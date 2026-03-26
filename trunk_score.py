#!/usr/bin/env python3
"""
trunk_score.py — Boltz-2 trunk scoring using official --write_embeddings output.

Boltz-2 CLI --write_embeddings writes to:
  <pred_dir>/embeddings_<name>.npz

Confirmed keys (from live inspection):
  s: [1, N, 384]       — per-token single representations
  z: [1, N, N, 128]    — per-token-pair representations

We use cross-chain z norm as IPSAE proxy:
  z[0, :binder_len, binder_len:].norm(axis=-1).mean(axis=1)
  Higher = stronger predicted interface signal per binder residue.

Falls back to PAE-based scoring if embeddings not found.
"""

import argparse, json, os, sys
import numpy as np


def score_via_embeddings(candidates, pred_dir, binder_len, target_len):
    import glob

    emb_files = sorted(glob.glob(os.path.join(pred_dir, "embeddings_*.npz")))
    if not emb_files:
        raise FileNotFoundError(f"No embeddings_*.npz in {pred_dir}")

    print(f"[trunk] found {len(emb_files)} embeddings file(s)", file=sys.stderr)

    expected    = binder_len + target_len
    accumulated = np.zeros(binder_len, dtype=np.float64)
    n_valid     = 0

    for ef in emb_files:
        data = np.load(ef)
        print(f"[trunk] keys: {list(data.keys())}", file=sys.stderr)

        # Confirmed key is 'z', fallback to 'pair_embeddings'
        if "z" in data:
            pair = data["z"].astype(np.float32)
        elif "pair_embeddings" in data:
            pair = data["pair_embeddings"].astype(np.float32)
        else:
            print(f"[trunk] no z or pair_embeddings in {ef}", file=sys.stderr)
            continue

        # Remove batch dimension if present
        if pair.ndim == 4:
            pair = pair[0]  # [N, N, d_pair]

        print(f"[trunk] pair shape after squeeze: {pair.shape}", file=sys.stderr)

        if pair.shape[0] != expected or pair.shape[1] != expected:
            print(f"[trunk] shape mismatch: expected {expected}, got {pair.shape[0]}",
                  file=sys.stderr)
            continue

        cross      = pair[:binder_len, binder_len:]       # [B, T, d_pair]
        cross_norm = np.linalg.norm(cross, axis=-1)       # [B, T]
        accumulated += cross_norm.mean(axis=1)            # [B]
        n_valid += 1

    if n_valid == 0:
        raise ValueError("No valid embeddings with correct shape")

    per_residue = accumulated / n_valid
    print(f"[trunk] score range: [{per_residue.min():.4f}, {per_residue.max():.4f}]",
          file=sys.stderr)

    scores = {}
    for cand in candidates:
        pos = cand["design_idx"]
        scores[str(pos)] = float(per_residue[pos]) if pos < len(per_residue) else 0.0
        print(f"[trunk] pos {pos} {cand['design_aa']}→{cand['known_aa']}: "
              f"{scores[str(pos)]:.4f}", file=sys.stderr)

    return scores, "pair_emb_norm"


def score_via_pae(candidates, pred_dir, binder_len, target_len):
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
    scores = {
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

    try:
        scores, method = score_via_embeddings(
            candidates, args.pred_dir,
            args.binder_len, args.target_len
        )
        print(f"[trunk] succeeded via {method}", file=sys.stderr)
    except Exception as e:
        print(f"[trunk] embeddings failed: {e}", file=sys.stderr)
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
