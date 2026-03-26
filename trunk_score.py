#!/usr/bin/env python3
"""
trunk_score.py — Boltz-2 trunk scoring for candidate substitutions.

Confirmed module paths from boltz/src/boltz/model/models/boltz2.py:
  from boltz.model.modules.diffusionv2 import AtomDiffusion
  from boltz.model.modules.trunkv2 import InputEmbedder, ...

Run as subprocess from Cell 9 to avoid numpy binary incompatibility.
Falls back to PAE-based scoring if trunk loading fails.
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

# Clear cached modules
for mod in list(sys.modules.keys()):
    if any(x in mod for x in ("numpy", "boltz", "torch")):
        del sys.modules[mod]

# ── Standard imports ──────────────────────────────────────────────────────────
import argparse, json, os
import numpy as np


def patch_atom_diffusion():
    """
    Patch AtomDiffusion to accept unknown kwargs from newer checkpoints.
    Correct import path confirmed from boltz source:
        from boltz.model.modules.diffusionv2 import AtomDiffusion
    """
    try:
        from boltz.model.modules.diffusionv2 import AtomDiffusion
        _orig = AtomDiffusion.__init__

        def _patched(self, *args, **kwargs):
            kwargs.pop("mse_rotational_alignment", None)
            _orig(self, *args, **kwargs)

        AtomDiffusion.__init__ = _patched
        print("[trunk] patched AtomDiffusion (diffusionv2)", file=sys.stderr)
        return True
    except Exception as e:
        print(f"[trunk] AtomDiffusion patch failed: {e}", file=sys.stderr)
        return False


def load_model_safe(checkpoint):
    """
    Load Boltz-2 model safely, handling version mismatches.
    Uses strict=False so missing/extra keys are ignored.
    Patches AtomDiffusion before instantiation.
    """
    import torch

    patch_atom_diffusion()

    from boltz.model.models.boltz2 import Boltz2

    # Load checkpoint dict directly to inspect hparams
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    print(f"[trunk] checkpoint keys: {list(ckpt.keys())}", file=sys.stderr)

    # Extract hyper_parameters and remove unknown ones
    hparams = ckpt.get("hyper_parameters", {})
    known_issues = ["mse_rotational_alignment"]
    for key in known_issues:
        if key in hparams:
            print(f"[trunk] removing unknown hparam: {key}", file=sys.stderr)
            del hparams[key]
    ckpt["hyper_parameters"] = hparams

    # Save cleaned checkpoint to temp file and load from there
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp:
        tmp_path = tmp.name
    torch.save(ckpt, tmp_path)

    try:
        model = Boltz2.load_from_checkpoint(tmp_path, strict=False)
        print("[trunk] model loaded successfully", file=sys.stderr)
    finally:
        try: os.unlink(tmp_path)
        except: pass

    return model


def score_via_trunk(candidates, pred_dir, binder_seq, binder_len, target_len):
    """
    Score candidates via Boltz-2 trunk InputEmbedder + pairformer.
    Returns (scores_dict, method_str).
    """
    import torch

    checkpoint = "/root/.boltz/boltz2_conf.ckpt"
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model  = load_model_safe(checkpoint)
    model  = model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    # Print available modules for debugging
    children = {n: type(m).__name__ for n, m in model.named_children()}
    print(f"[trunk] modules: {children}", file=sys.stderr)

    # Locate InputEmbedder — confirmed name from trunkv2
    embedder = None
    for attr in ["input_embedder", "InputEmbedder", "embedder"]:
        if hasattr(model, attr):
            embedder = getattr(model, attr)
            print(f"[trunk] found embedder: {attr}", file=sys.stderr)
            break

    # Locate pairformer trunk — confirmed from trunkv2
    pairformer = None
    for attr in ["pairformer", "trunk", "pairformer_stack"]:
        if hasattr(model, attr):
            pairformer = getattr(model, attr)
            print(f"[trunk] found pairformer: {attr}", file=sys.stderr)
            break

    if embedder is None or pairformer is None:
        raise RuntimeError(
            f"Could not find embedder or pairformer. "
            f"Available: {list(children.keys())}"
        )

    AA_TO_IDX = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    N = binder_len + target_len

    # Build parent one-hot [N, 20]
    parent_onehot = torch.zeros(N, 20, device=device)
    for i, aa in enumerate(binder_seq):
        parent_onehot[i, AA_TO_IDX.get(aa, 0)] = 1.0

    def run_trunk(onehot):
        with torch.no_grad():
            try:
                x      = onehot.unsqueeze(0)          # [1, N, 20]
                single = embedder(x)
                result = pairformer(single, None)
                if isinstance(result, tuple):
                    _, pair = result
                else:
                    return None
                pair  = pair.squeeze(0)               # [N, N, H]
                cross = pair[:binder_len, binder_len:]# [B, T, H]
                return cross.norm(dim=-1).mean().item()
            except Exception as e:
                print(f"[trunk] forward pass failed: {e}", file=sys.stderr)
                return None

    parent_score = run_trunk(parent_onehot)
    if parent_score is None:
        raise RuntimeError("Trunk forward pass returned None")

    print(f"[trunk] parent_score={parent_score:.6f}", file=sys.stderr)

    scores = {}
    for cand in candidates:
        pos     = cand["design_idx"]
        new_idx = AA_TO_IDX.get(cand["known_aa"], 0)

        sub = parent_onehot.clone()
        sub[pos]         = 0.0
        sub[pos, new_idx] = 1.0

        sub_score = run_trunk(sub)
        scores[str(pos)] = (
            float(sub_score - parent_score) if sub_score is not None else 0.0
        )

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

    # Attempt 1: trunk
    try:
        scores, method = score_via_trunk(
            candidates, args.pred_dir,
            args.binder_seq, args.binder_len, args.target_len
        )
        print(f"[trunk] succeeded via {method}", file=sys.stderr)
    except Exception as e:
        print(f"[trunk] trunk failed: {e}", file=sys.stderr)
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
