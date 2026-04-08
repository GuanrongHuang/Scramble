#!/usr/bin/env python3
"""
trunk_score.py — Boltz-2 trunk scoring combining z embeddings and PAE.

Confirmed file outputs from boltz --write_embeddings --write_full_pae:
  embeddings_<name>.npz  → keys: s [1,N,384], z [1,N,N,128]
  pae_<name>_model_N.npz → key: pae [N,N]

Combined score per binder residue:
  z_score   = mean(norm(z[:binder_len, binder_len:], axis=-1), axis=1)
              Higher = stronger trunk interface encoding at this position.

  pae_score = -mean(pae[:binder_len, binder_len:], axis=1)
              Higher (less negative) = more confident interface placement.

Both are normalised to [0,1] within the design and combined:
  combined = 0.5 * z_norm + 0.5 * pae_norm

This answers two complementary questions:
  z   → how much did the pairformer encode this position's interface relationship?
  PAE → how confidently does the model place this position relative to the target?

A position scoring high on both is genuinely interfacial AND confidently placed
— the strongest signal for prioritising grafting substitutions.
"""

import argparse, json, os, sys
import numpy as np


def load_z_scores(pred_dir, binder_len, target_len):
    """
    Load cross-chain z norm per binder residue from embeddings_*.npz.
    Returns [binder_len] float array, or None if not available.
    """
    import glob

    emb_files = sorted(glob.glob(os.path.join(pred_dir, "embeddings_*.npz")))
    if not emb_files:
        return None

    expected    = binder_len + target_len
    accumulated = np.zeros(binder_len, dtype=np.float64)
    n_valid     = 0

    for ef in emb_files:
        data = np.load(ef)

        if "z" in data:
            pair = data["z"].astype(np.float32)
        elif "pair_embeddings" in data:
            pair = data["pair_embeddings"].astype(np.float32)
        else:
            continue

        if pair.ndim == 4:
            pair = pair[0]

        if pair.shape[0] != expected or pair.shape[1] != expected:
            continue

        cross      = pair[:binder_len, binder_len:]   # [B, T, 128]
        cross_norm = np.linalg.norm(cross, axis=-1)   # [B, T]
        accumulated += cross_norm.mean(axis=1)        # [B]
        n_valid += 1

    if n_valid == 0:
        return None

    scores = accumulated / n_valid
    print(f"[trunk] z_score range: [{scores.min():.3f}, {scores.max():.3f}]",
          file=sys.stderr)
    return scores


def load_pae_scores(pred_dir, binder_len, target_len):
    """
    Load mean cross-chain PAE per binder residue from pae_*.npz.
    Returns [binder_len] float array of negated PAE (higher = more confident).
    Returns None if not available.
    """
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
        return None

    # Negate so higher = more confident (lower PAE = better)
    scores = -(accumulated / n_valid)
    print(f"[trunk] pae_score range: [{scores.min():.3f}, {scores.max():.3f}]",
          file=sys.stderr)
    return scores


def normalise(arr):
    """Normalise array to [0, 1]. Returns uniform 0.5 if all values identical."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.full_like(arr, 0.5)
    return (arr - mn) / (mx - mn)


def combine_scores(z_scores, pae_scores):
    """
    Combine z norm and PAE scores into a single per-residue score.
    Both are normalised to [0,1] first so neither dominates.
    """
    if z_scores is not None and pae_scores is not None:
        combined = 0.5 * normalise(z_scores) + 0.5 * normalise(pae_scores)
        method   = "z_pae_combined"
    elif z_scores is not None:
        combined = normalise(z_scores)
        method   = "pair_emb_norm"
    elif pae_scores is not None:
        combined = normalise(pae_scores)
        method   = "pae_fallback"
    else:
        combined = None
        method   = "fallback"

    return combined, method


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir",    required=True)
    parser.add_argument("--binder_seq",  required=True)
    parser.add_argument("--binder_len",  type=int, required=True)
    parser.add_argument("--target_len",  type=int, required=True)
    parser.add_argument("--candidates",  required=True)
    parser.add_argument("--out",         required=True)
    args = parser.parse_args()

    candidates  = json.loads(args.candidates)
    binder_len  = args.binder_len
    target_len  = args.target_len

    # Load both signals independently
    z_scores   = load_z_scores(args.pred_dir, binder_len, target_len)
    pae_scores = load_pae_scores(args.pred_dir, binder_len, target_len)

    combined, method = combine_scores(z_scores, pae_scores)

    if combined is not None:
        scores = {}
        for cand in candidates:
            pos = cand["design_idx"]
            scores[str(pos)] = float(combined[pos]) if pos < len(combined) else 0.0
            z_str   = f"{z_scores[pos]:.2f}"   if z_scores   is not None else "N/A"
            pae_str = f"{-pae_scores[pos]:.2f}" if pae_scores is not None else "N/A"
            print(f"[trunk] pos {pos} {cand['design_aa']}→{cand['known_aa']}: "
                  f"combined={scores[str(pos)]:.4f}  z={z_str}  pae={pae_str}",
                  file=sys.stderr)
    else:
        print("[trunk] no scoring signals available — using zero scores",
              file=sys.stderr)
        scores = {str(c["design_idx"]): 0.0 for c in candidates}

    print(f"[trunk] method: {method}", file=sys.stderr)

    with open(args.out, "w") as f:
        json.dump({"scores": scores, "method": method}, f)

    print(f"[trunk] wrote {len(scores)} scores via {method}", file=sys.stderr)


if __name__ == "__main__":
    main()
