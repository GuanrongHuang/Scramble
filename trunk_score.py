#!/usr/bin/env python3
"""
trunk_score.py — Boltz-2 trunk scoring for candidate substitutions.

Run as a subprocess from Cell 9 to avoid numpy binary incompatibility.
Falls back to PAE-based scoring if trunk model loading fails.

Higher score = more promising substitution candidate.
"""

# ── Fix environment before any other imports ──────────────────────────────────
import subprocess, sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "numpy==1.26.4", "-q", "--quiet"],
    capture_output=True
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "boltz", "--upgrade", "-q", "--quiet"],
    capture_output=True
)

# Clear cached modules so fresh versions are used
for mod in list(sys.modules.keys()):
    if any(x in mod for x in ("numpy", "boltz", "torch")):
        del sys.modules[mod]

# ── Standard imports ──────────────────────────────────────────────────────────
import argparse, json, os
import numpy as np


def patch_boltz_for_compat():
    """
    Patch Boltz-2 classes to accept unknown kwargs from newer checkpoints.
    This handles version mismatches between the installed boltz package
    and the checkpoint file without requiring a full upgrade.
    """
    patches_applied = []

    # Patch AtomDiffusion
    try:
        from boltz.model.modules.diffusion.atom_diffusion import AtomDiffusion
        _orig = AtomDiffusion.__init__
        def _patched_atom(self, *args, **kwargs):
            kwargs.pop("mse_rotational_alignment", None)
            _orig(self, *args, **kwargs)
        AtomDiffusion.__init__ = _patched_atom
        patches_applied.append("AtomDiffusion")
    except Exception as e:
        print(f"[trunk] AtomDiffusion patch failed: {e}", file=sys.stderr)

    # Patch Boltz2 model itself in case it has unknown kwargs too
    try:
        from boltz.model.models.boltz2 import Boltz2
        _orig2 = Boltz2.__init__
        def _patched_boltz2(self, *args, **kwargs):
            for k in list(kwargs.keys()):
                try:
                    import inspect
                    sig = inspect.signature(_orig2)
                    if k not in sig.parameters:
                        kwargs.pop(k)
                except Exception:
                    pass
            _orig2(self, *args, **kwargs)
        Boltz2.__init__ = _patched_boltz2
        patches_applied.append("Boltz2")
    except Exception as e:
        print(f"[trunk] Boltz2 patch failed: {e}", file=sys.stderr)

    if patches_applied:
        print(f"[trunk] patched: {patches_applied}", file=sys.stderr)


def score_via_trunk(candidates, pred_dir, binder_seq, binder_len, target_len):
    """
    Load Boltz-2 model and score candidates via trunk pair representations.
    Returns (scores_dict, method_str).
    scores_dict: {design_idx_str: float} — higher = more promising
    """
    import torch

    # Apply compatibility patches before importing Boltz2
    patch_boltz_for_compat()

    from boltz.model.models.boltz2 import Boltz2

    checkpoint = "/root/.boltz/boltz2_conf.ckpt"
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"[trunk] loading model...", file=sys.stderr)
    model  = Boltz2.load_from_checkpoint(checkpoint, strict=False)
    model  = model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    top_modules = [n for n, _ in model.named_children()]
    print(f"[trunk] loaded. modules: {top_modules}", file=sys.stderr)

    # Find processed features directory
    search    = pred_dir
    processed = None
    for _ in range(6):
        p = os.path.join(search, "processed")
        if os.path.isdir(p):
            processed = p
            break
        search = os.path.dirname(search)
        if search == os.path.dirname(search):
            break

    if processed is None:
        raise FileNotFoundError(f"processed/ dir not found near {pred_dir}")

    import glob, pickle
    pkl_files = glob.glob(os.path.join(processed, "mols", "*.pkl"))
    npz_files = glob.glob(os.path.join(processed, "structures", "*.npz"))

    if not pkl_files or not npz_files:
        raise FileNotFoundError(f"Missing pkl/npz in {processed}")

    with open(pkl_files[0], "rb") as f:
        mol_data = pickle.load(f)
    struct_data = dict(np.load(npz_files[0], allow_pickle=True))
    print(f"[trunk] struct keys: {list(struct_data.keys())}", file=sys.stderr)

    AA_TO_IDX = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    N = binder_len + target_len

    # Build parent amino acid one-hot [N, 20]
    parent_onehot = torch.zeros(N, 20, device=device)
    for i, aa in enumerate(binder_seq):
        parent_onehot[i, AA_TO_IDX.get(aa, 0)] = 1.0

    def try_embed(x):
        """Try all known embedding module names, return single repr or None."""
        for attr in ["input_embedder", "token_embedder", "embedding",
                     "single_embedder", "residue_embedder"]:
            if hasattr(model, attr):
                try:
                    out = getattr(model, attr)(x)
                    print(f"[trunk] embedded via {attr}", file=sys.stderr)
                    return out
                except Exception as e:
                    print(f"[trunk] {attr} failed: {e}", file=sys.stderr)
        return None

    def try_pairformer(single):
        """Try all known pairformer/trunk module names, return (single, pair) or None."""
        for attr in ["pairformer", "trunk", "evoformer", "structure_module"]:
            if hasattr(model, attr):
                try:
                    result = getattr(model, attr)(single, None)
                    if isinstance(result, tuple) and len(result) == 2:
                        print(f"[trunk] pairformer via {attr}", file=sys.stderr)
                        return result
                    result2 = getattr(model, attr)(single)
                    if isinstance(result2, tuple) and len(result2) == 2:
                        print(f"[trunk] pairformer via {attr} (no pair arg)",
                              file=sys.stderr)
                        return result2
                except Exception as e:
                    print(f"[trunk] {attr} failed: {e}", file=sys.stderr)
        return None

    def run_trunk(onehot):
        """Return cross-chain pair repr norm as IPSAE proxy, or None."""
        with torch.no_grad():
            x      = onehot.unsqueeze(0)   # [1, N, 20]
            single = try_embed(x)
            if single is None:
                return None
            result = try_pairformer(single)
            if result is None:
                return None
            _, pair = result
            pair  = pair.squeeze(0)                    # [N, N, H]
            cross = pair[:binder_len, binder_len:]     # [B, T, H]
            return cross.norm(dim=-1).mean().item()

    parent_score = run_trunk(parent_onehot)
    if parent_score is None:
        raise RuntimeError("Trunk returned no pair representation")

    print(f"[trunk] parent_score={parent_score:.6f}", file=sys.stderr)

    scores = {}
    for cand in candidates:
        pos     = cand["design_idx"]
        new_idx = AA_TO_IDX.get(cand["known_aa"], 0)

        sub = parent_onehot.clone()
        sub[pos]         = 0.0
        sub[pos, new_idx] = 1.0

        sub_score = run_trunk(sub)
        scores[str(pos)] = (sub_score - parent_score) if sub_score is not None else 0.0

    return scores, "pair_repr_norm"


def score_via_pae(candidates, pred_dir, binder_len, target_len):
    """
    Fallback: rank by per-residue cross-chain PAE from existing .npz files.
    Lower PAE = more confident interface placement = higher score.
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

    # Attempt 1 — trunk
    try:
        scores, method = score_via_trunk(
            candidates, args.pred_dir,
            args.binder_seq, args.binder_len, args.target_len
        )
        print(f"[trunk] succeeded via {method}", file=sys.stderr)
    except Exception as e:
        print(f"[trunk] trunk failed: {e}", file=sys.stderr)
        # Attempt 2 — PAE fallback
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
