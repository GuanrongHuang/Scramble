#!/usr/bin/env python3
"""
trunk_score.py — Boltz-2 trunk scoring using official --write_embeddings output.

Boltz-2 CLI supports --write_embeddings which writes trunk representations to:
  <pred_dir>/embeddings_<name>_model_0.npz

The file contains:
  single_embeddings: [N, d_single]  — per-token trunk single representations
  pair_embeddings:   [N, N, d_pair] — per-token-pair trunk pair representations

We use the cross-chain pair embeddings as an IPSAE proxy:
  proxy = mean(norm(pair_embeddings[:binder_len, binder_len:], axis=-1))

This is the norm of the pair representation between every binder token and
every target token — higher norm = stronger predicted interface interaction.

No model loading required. No Python API. No numpy incompatibility.
Just read the npz files already written by the CLI.

If embeddings are not found (--write_embeddings not added to run_boltz yet),
falls back to PAE-based scoring.
"""

import argparse, json, os, sys
import numpy as np


def score_via_embeddings(candidates, pred_dir, binder_len, target_len):
    """
    Score candidates using cross-chain pair embeddings from embeddings_*.npz.
    
    For each candidate substitution, we compare the cross-chain pair embedding
    norm of the parent prediction. Since we only have the parent prediction's
    embeddings (not re-predicted with the substitution), we use the per-residue
    cross-chain pair norm as a position-level confidence signal — positions with
    higher norm are more confidently placed at the interface and are better
    grafting candidates.
    
    Returns (scores_dict, method_str).
    """
    import glob

    # Find embeddings file — written by boltz with --write_embeddings
    emb_files = sorted(glob.glob(os.path.join(pred_dir, "embeddings_*.npz")))
    if not emb_files:
        raise FileNotFoundError(
            f"No embeddings_*.npz in {pred_dir}. "
            "Add --write_embeddings to run_boltz in Cell 5 and re-run Stage 2."
        )

    print(f"[trunk] found {len(emb_files)} embeddings file(s)", file=sys.stderr)

    # Average cross-chain pair norm across all diffusion samples
    expected    = binder_len + target_len
    accumulated = np.zeros(binder_len, dtype=np.float64)
    n_valid     = 0

    for ef in emb_files:
        data = np.load(ef)
        print(f"[trunk] {os.path.basename(ef)} keys: {list(data.keys())}",
              file=sys.stderr)

        if "pair_embeddings" not in data:
            print(f"[trunk] no pair_embeddings in {ef}", file=sys.stderr)
            continue

        pair = data["pair_embeddings"].astype(np.float32)  # [N, N, d_pair]
        print(f"[trunk] pair shape: {pair.shape}", file=sys.stderr)

        if pair.shape[0] != expected or pair.shape[1] != expected:
            print(f"[trunk] shape mismatch: expected {expected}x{expected}, "
                  f"got {pair.shape[0]}x{pair.shape[1]}", file=sys.stderr)
            continue

        # Cross-chain pair norm: binder rows × target columns
        cross = pair[:binder_len, binder_len:]             # [B, T, d_pair]
        cross_norm = np.linalg.norm(cross, axis=-1)        # [B, T]
        accumulated += cross_norm.mean(axis=1)             # [B]
        n_valid += 1

    if n_valid == 0:
        raise ValueError("No valid embeddings files found with correct shape")

    per_residue_score = accumulated / n_valid  # [binder_len] — higher = better interface

    print(f"[trunk] per-residue score range: "
          f"[{per_residue_score.min():.4f}, {per_residue_score.max():.4f}]",
          file=sys.stderr)

    scores = {}
    for cand in candidates:
        pos = cand["design_idx"]
        if pos < len(per_residue_score):
            scores[str(pos)] = float(per_residue_score[pos])
            print(f"[trunk] pos {pos} {cand['design_aa']}→{cand['known_aa']}: "
                  f"score={scores[str(pos)]:.4f}", file=sys.stderr)
        else:
            scores[str(pos)] = 0.0

    return scores, "pair_emb_norm"


def score_via_pae(candidates, pred_dir, binder_len, target_len):
    """Fallback: rank by per-residue cross-chain PAE (lower PAE = higher score)."""
    import glob

    pae_files   = sorted(glob.glob(os.path.join(pred_dir, "pae_*.npz")))
    expected    = binder_len + target_len
    accumulated = np.zeros(binder_len, dtype=np.float64)
    n_valid     = 0

    for pf in pae_files:
        data = np.load(pf)
        pae  = data["pae"] if "pae" in data else list(data.values())[0]
        if pae.ndim == 3: pae = pae[0]
        pae = pae.astype(np.float32)
        if pae.shape == (expected, expected):
            accumulated += pae[:binder_len, binder_len:].mean(axis=1)
            n_valid += 1

    if n_valid == 0:
        return {str(c["design_idx"]): 0.0 for c in candidates}, "fallback"

    cross_pae = accumulated / n_valid
    scores    = {
        str(c["design_idx"]): -float(cross_pae[c["design_idx"]])
        if c["design_idx"] < len(cross_pae) else 0.0
        for c in candidates
    }
    return scores, "pae_fallback"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir",    required=True)
    parser.add_argument("--binder_seq",  required=True)
    parser.add_argument("--binder_len",  type=int, required=True)
    parser.add_argument("--target_len",  type=int, required=True)
    parser.add_argument("--candidates",  required=True)
    parser.add_argument("--out",         required=True)
    args = parser.parse_args()

    candidates = json.loads(args.candidates)

    # Attempt 1: official trunk embeddings from --write_embeddings
    try:
        scores, method = score_via_embeddings(
            candidates, args.pred_dir,
            args.binder_len, args.target_len
        )
        print(f"[trunk] succeeded via {method}", file=sys.stderr)
    except Exception as e:
        print(f"[trunk] embeddings scoring failed: {e}", file=sys.stderr)
        # Attempt 2: PAE fallback
        try:
            scores, method = score_via_pae(
                candidates, args.pred_dir,
                args.binder_len, args.target_len
            )
            print(f"[trunk] PAE fallback succeeded", file=sys.stderr)
        except Exception as e2:
            print(f"[trunk] PAE fallback failed: {e2}", file=sys.stderr)
            scores = {str(c["design_idx"]): 0.0 for c in candidates}
            method = "fallback"

    with open(args.out, "w") as f:
        json.dump({"scores": scores, "method": method}, f)

    print(f"[trunk] wrote {len(scores)} scores via {method}", file=sys.stderr)


if __name__ == "__main__":
    main()
