#!/usr/bin/env python3
"""
trunk_score.py — Boltz-2 trunk scoring using z, s, pLDDT variance, and PDE.

Confirmed file outputs from boltz --write_embeddings --write_full_pae:
  embeddings_<name>.npz       -> keys: s [1,N,384], z [1,N,N,128]
  plddt_<name>_model_N.npz    -> key: plddt [N]
  pae_<name>_model_N.npz      -> key: pae [N,N]
  pde_<name>_model_N.npz      -> key: pde [N,N]

Scoring formula per binder residue:
  z_score         = mean(norm(z[0, :binder_len, binder_len:], axis=-1), axis=1)
                    Higher = stronger pairformer interface encoding.
                    Literature: z encodes residue-residue interactions (Boltz-2 paper).

  s_score         = norm(s[0, :binder_len, :], axis=-1)
                    Higher = higher predicted flexibility/disorder (B-factor proxy).
                    Literature: s single representation supervised on B-factors (Boltz-2 paper).

  plddt_var_score = variance of plddt[:binder_len] across 6 diffusion samples per position.
                    Higher = more structurally uncertain across samples = better substitution target.

  pde_score       = -mean(pde[0, :binder_len, binder_len:], axis=1)
                    Higher (less negative) = more confident interface distance geometry.
                    Complement to PAE for contact geometry confidence.

Combined:
  combined = 0.5 * z_norm + 0.25 * s_norm + 0.25 * plddt_var_norm

  PDE is loaded and printed for diagnostics but not yet in the formula.
  PAE is loaded and printed for diagnostics but not in the formula
  (empirically produced inferior IPSAE outcomes vs z-only).

All signals normalised to [0,1] within the design before combining.
"""

import argparse, glob, json, os, sys
import numpy as np


# ── Signal loaders ────────────────────────────────────────────────────────────

def load_z_scores(pred_dir, binder_len, target_len):
    """
    Cross-chain z norm per binder residue from embeddings_*.npz.
    Returns float32 [binder_len] or None.
    """
    emb_files = sorted(glob.glob(os.path.join(pred_dir, "embeddings_*.npz")))
    if not emb_files:
        return None

    expected    = binder_len + target_len
    accumulated = np.zeros(binder_len, dtype=np.float64)
    n_valid     = 0

    for ef in emb_files:
        try:
            data = np.load(ef)
        except Exception as e:
            print(f"[trunk] warn: could not load {ef}: {e}", file=sys.stderr)
            continue

        if "z" in data:
            pair = data["z"].astype(np.float32)
        elif "pair_embeddings" in data:
            pair = data["pair_embeddings"].astype(np.float32)
        else:
            continue

        if pair.ndim == 4:
            pair = pair[0]   # [N, N, 128]

        if pair.shape[0] != expected or pair.shape[1] != expected:
            continue

        cross      = pair[:binder_len, binder_len:]        # [B, T, 128]
        cross_norm = np.linalg.norm(cross, axis=-1)        # [B, T]
        accumulated += cross_norm.mean(axis=1)             # [B]
        n_valid += 1

    if n_valid == 0:
        return None

    scores = (accumulated / n_valid).astype(np.float32)
    print(f"[trunk] z_score range: [{scores.min():.3f}, {scores.max():.3f}]",
          file=sys.stderr)
    return scores


def load_s_scores(pred_dir, binder_len, target_len):
    """
    s norm per binder residue from embeddings_*.npz.
    s is supervised on B-factors in Boltz-2 training — proxy for flexibility.
    Returns float32 [binder_len] or None.
    """
    emb_files = sorted(glob.glob(os.path.join(pred_dir, "embeddings_*.npz")))
    if not emb_files:
        return None

    expected    = binder_len + target_len
    accumulated = np.zeros(binder_len, dtype=np.float64)
    n_valid     = 0

    for ef in emb_files:
        try:
            data = np.load(ef)
        except Exception as e:
            print(f"[trunk] warn: could not load {ef}: {e}", file=sys.stderr)
            continue

        if "s" not in data:
            continue

        single = data["s"].astype(np.float32)   # [1, N, 384] or [N, 384]
        if single.ndim == 3:
            single = single[0]                   # [N, 384]

        if single.shape[0] != expected:
            continue

        s_norm = np.linalg.norm(single[:binder_len, :], axis=-1)  # [B]
        accumulated += s_norm
        n_valid += 1

    if n_valid == 0:
        return None

    scores = (accumulated / n_valid).astype(np.float32)
    print(f"[trunk] s_score range: [{scores.min():.3f}, {scores.max():.3f}]",
          file=sys.stderr)
    return scores


def load_plddt_variance(pred_dir, binder_len, target_len):
    """
    Per-residue pLDDT variance across diffusion samples for binder residues.
    Higher variance = model is structurally uncertain across samples = better substitution target.
    Returns float32 [binder_len] or None.
    """
    plddt_files = sorted(glob.glob(os.path.join(pred_dir, "plddt_*.npz")))
    if not plddt_files:
        return None

    expected = binder_len + target_len
    arrays   = []

    for pf in plddt_files:
        try:
            data   = np.load(pf)
            plddt  = data["plddt"].astype(np.float32)   # [N]
        except Exception as e:
            print(f"[trunk] warn: could not load {pf}: {e}", file=sys.stderr)
            continue

        if plddt.shape[0] != expected:
            continue

        arrays.append(plddt[:binder_len])   # [B]

    if len(arrays) < 2:
        # Need at least 2 samples to compute variance
        return None

    stacked = np.stack(arrays, axis=0)      # [samples, B]
    scores  = stacked.var(axis=0)           # [B]
    print(f"[trunk] plddt_var range: [{scores.min():.4f}, {scores.max():.4f}]",
          file=sys.stderr)
    return scores.astype(np.float32)


def load_pde_scores(pred_dir, binder_len, target_len):
    """
    Cross-chain mean PDE (predicted distance error) per binder residue.
    Loaded for diagnostics — not currently in the combined formula.
    Returns float32 [binder_len] or None.
    """
    pde_files   = sorted(glob.glob(os.path.join(pred_dir, "pde_*.npz")))
    if not pde_files:
        return None

    expected    = binder_len + target_len
    accumulated = np.zeros(binder_len, dtype=np.float64)
    n_valid     = 0

    for pf in pde_files:
        try:
            data = np.load(pf)
            pde  = data["pde"].astype(np.float32)   # [N, N]
        except Exception as e:
            print(f"[trunk] warn: could not load {pf}: {e}", file=sys.stderr)
            continue

        if pde.shape != (expected, expected):
            continue

        accumulated += pde[:binder_len, binder_len:].mean(axis=1)
        n_valid += 1

    if n_valid == 0:
        return None

    scores = -(accumulated / n_valid).astype(np.float32)   # negate: lower PDE = better
    print(f"[trunk] pde_score range: [{scores.min():.3f}, {scores.max():.3f}]",
          file=sys.stderr)
    return scores


def load_pae_scores(pred_dir, binder_len, target_len):
    """
    Cross-chain mean PAE per binder residue.
    Loaded for diagnostics only — disabled from formula.
    Empirically produced inferior IPSAE outcomes vs z-only.
    Returns float32 [binder_len] or None.
    """
    pae_files   = sorted(glob.glob(os.path.join(pred_dir, "pae_*.npz")))
    if not pae_files:
        return None

    expected    = binder_len + target_len
    accumulated = np.zeros(binder_len, dtype=np.float64)
    n_valid     = 0

    for pf in pae_files:
        try:
            data = np.load(pf)
            pae  = data["pae"] if "pae" in data else list(data.values())[0]
            if pae.ndim == 3:
                pae = pae[0]
            pae = pae.astype(np.float32)
        except Exception as e:
            print(f"[trunk] warn: could not load {pf}: {e}", file=sys.stderr)
            continue

        if pae.shape != (expected, expected):
            continue

        accumulated += pae[:binder_len, binder_len:].mean(axis=1)
        n_valid += 1

    if n_valid == 0:
        return None

    scores = -(accumulated / n_valid).astype(np.float32)
    print(f"[trunk] pae_score range: [{scores.min():.3f}, {scores.max():.3f}]",
          file=sys.stderr)
    return scores


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalise(arr):
    """Normalise float array to [0, 1]. Returns 0.5 uniformly if all identical."""
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-8:
        return np.full_like(arr, 0.5, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


# ── Combination ───────────────────────────────────────────────────────────────

def combine_scores(z_scores, s_scores, plddt_var, pde_scores, pae_scores):
    """
    Combined score:
      0.50 * z_norm  (primary: pairformer interface encoding)
      0.25 * s_norm  (B-factor proxy: flexibility at interface)
      0.25 * plddt_var_norm (cross-sample structural uncertainty)

    PDE and PAE are loaded and printed for diagnostics but not in the formula.
    Falls back gracefully if s or plddt_var are unavailable.
    """
    if z_scores is None:
        # z unavailable — fall back through available signals
        if plddt_var is not None:
            return normalise(plddt_var), "plddt_var_fallback"
        if pae_scores is not None:
            return normalise(pae_scores), "pae_fallback"
        return None, "fallback"

    components = [("z", 0.50, normalise(z_scores))]
    weights_used = 0.50

    if s_scores is not None:
        components.append(("s", 0.25, normalise(s_scores)))
        weights_used += 0.25

    if plddt_var is not None:
        components.append(("plddt_var", 0.25, normalise(plddt_var)))
        weights_used += 0.25

    # Renormalise weights if some signals are missing
    total_weight = sum(w for _, w, _ in components)
    combined = sum((w / total_weight) * arr for _, w, arr in components)
    combined = combined.astype(np.float32)

    active = "+".join(name for name, _, _ in components)
    method = f"z_s_plddt_var" if len(components) == 3 else f"partial_{active}"

    return combined, method


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir",   required=True)
    parser.add_argument("--binder_seq", required=True)
    parser.add_argument("--binder_len", type=int, required=True)
    parser.add_argument("--target_len", type=int, required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out",        required=True)
    args = parser.parse_args()

    candidates = json.loads(args.candidates)
    binder_len = args.binder_len
    target_len = args.target_len

    # Load all signals
    z_scores   = load_z_scores(args.pred_dir, binder_len, target_len)
    s_scores   = load_s_scores(args.pred_dir, binder_len, target_len)
    plddt_var  = load_plddt_variance(args.pred_dir, binder_len, target_len)
    pde_scores = load_pde_scores(args.pred_dir, binder_len, target_len)     # diagnostic
    pae_scores = load_pae_scores(args.pred_dir, binder_len, target_len)     # diagnostic

    combined, method = combine_scores(z_scores, s_scores, plddt_var,
                                      pde_scores, pae_scores)

    if combined is not None:
        scores = {}
        for cand in candidates:
            pos = cand["design_idx"]
            if pos >= len(combined):
                scores[str(pos)] = 0.0
                continue

            scores[str(pos)] = float(combined[pos])

            z_str       = f"{z_scores[pos]:.3f}"      if z_scores   is not None else "N/A"
            s_str       = f"{s_scores[pos]:.3f}"      if s_scores   is not None else "N/A"
            pv_str      = f"{plddt_var[pos]:.4f}"     if plddt_var  is not None else "N/A"
            pde_str     = f"{-pde_scores[pos]:.3f}"   if pde_scores is not None else "N/A"
            pae_str     = f"{-pae_scores[pos]:.3f}"   if pae_scores is not None else "N/A"

            print(
                f"[trunk] pos {pos:3d} {cand['design_aa']}->{cand['known_aa']}: "
                f"score={scores[str(pos)]:.4f}  "
                f"z={z_str}  s={s_str}  plddt_var={pv_str}  "
                f"pde={pde_str}  pae={pae_str}",
                file=sys.stderr
            )
    else:
        print("[trunk] no scoring signals available — using zero scores",
              file=sys.stderr)
        scores = {str(c["design_idx"]): 0.0 for c in candidates}

    print(f"[trunk] method: {method}", file=sys.stderr)

    with open(args.out, "w") as fh:
        json.dump({"scores": scores, "method": method}, fh)

    print(f"[trunk] wrote {len(scores)} scores via {method}", file=sys.stderr)


if __name__ == "__main__":
    main()
