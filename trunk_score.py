#!/usr/bin/env python3
"""
trunk_score.py — Boltz-2 trunk scoring for candidate substitutions.

Run as a subprocess from Cell 9 to avoid numpy binary incompatibility
when importing boltz alongside gemmi/freesasa in the notebook process.

Usage:
    python trunk_score.py \
        --pred_dir   <path_to_stage2_pred_dir> \
        --binder_seq <sequence_string> \
        --binder_len <int> \
        --target_len <int> \
        --candidates <json_string> \
        --out        <output_json_path>

Output JSON:
    {
        "scores": {"<design_idx>": <float>, ...},
        "method": "pair_repr_norm" | "pae_fallback" | "fallback"
    }

Higher score = more promising substitution candidate.
"""

# ── Fix numpy binary incompatibility before any other imports ─────────────────
import subprocess, sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "numpy==1.26.4", "-q", "--quiet"],
    capture_output=True
)

# Force fresh numpy import with pinned version
import importlib
for mod in list(sys.modules.keys()):
    if "numpy" in mod:
        del sys.modules[mod]

# ── Standard imports ──────────────────────────────────────────────────────────
import argparse, json, os
import numpy as np


def score_via_trunk(candidates, pred_dir, binder_seq, binder_len, target_len):
    """
    Attempt to load Boltz-2 model and score candidates via trunk pair
    representations. Returns (scores_dict, method_str) or raises on failure.

    scores_dict: {design_idx_str: float}  — higher = more promising
    """
    import torch
    from boltz.model.models.boltz2 import Boltz2

    checkpoint = "/root/.boltz/boltz2_conf.ckpt"
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"[trunk] loading model from {checkpoint}", file=sys.stderr)
    model  = Boltz2.load_from_checkpoint(checkpoint)
    model  = model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    # Print top-level module names for debugging
    top_modules = [name for name, _ in model.named_children()]
    print(f"[trunk] top-level modules: {top_modules}", file=sys.stderr)

    # Locate processed features written by CLI during Stage 2
    # pred_dir is typically: .../boltz_results_X/predictions/X/
    # processed is at:       .../boltz_results_X/processed/
    search = pred_dir
    processed = None
    for _ in range(5):
        candidate_proc = os.path.join(search, "processed")
        if os.path.isdir(candidate_proc):
            processed = candidate_proc
            break
        parent = os.path.dirname(search)
        if parent == search:
            break
        search = parent

    # Also try sibling directory (boltz_results_X/processed)
    if processed is None:
        boltz_results = pred_dir
        for _ in range(4):
            boltz_results = os.path.dirname(boltz_results)
            candidate_proc = os.path.join(boltz_results, "processed")
            if os.path.isdir(candidate_proc):
                processed = candidate_proc
                break

    if processed is None:
        raise FileNotFoundError(
            f"Could not find processed/ directory near {pred_dir}"
        )

    print(f"[trunk] processed dir: {processed}", file=sys.stderr)

    import glob, pickle
    pkl_files = glob.glob(os.path.join(processed, "mols", "*.pkl"))
    npz_files = glob.glob(os.path.join(processed, "structures", "*.npz"))

    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files in {processed}/mols/")
    if not npz_files:
        raise FileNotFoundError(f"No .npz files in {processed}/structures/")

    print(f"[trunk] pkl: {pkl_files[0]}", file=sys.stderr)
    print(f"[trunk] npz: {npz_files[0]}", file=sys.stderr)

    with open(pkl_files[0], "rb") as f:
        mol_data = pickle.load(f)
    struct_data = dict(np.load(npz_files[0], allow_pickle=True))

    print(f"[trunk] struct_data keys: {list(struct_data.keys())}", file=sys.stderr)

    AA_TO_IDX = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    N = binder_len + target_len

    # Build parent amino acid one-hot
    parent_onehot = torch.zeros(N, 20, device=device)
    for i, aa in enumerate(binder_seq):
        parent_onehot[i, AA_TO_IDX.get(aa, 0)] = 1.0

    def run_trunk(onehot):
        """
        Try to run the pairformer trunk with the given one-hot sequence.
        Returns cross-chain pair representation norm as IPSAE proxy, or None.
        """
        with torch.no_grad():
            x = onehot.unsqueeze(0)  # [1, N, 20]

            # Try different module name conventions
            single = None

            # Attempt 1: input_embedder
            if hasattr(model, "input_embedder"):
                try:
                    single = model.input_embedder(x)
                    print(f"[trunk] used input_embedder", file=sys.stderr)
                except Exception as e:
                    print(f"[trunk] input_embedder failed: {e}", file=sys.stderr)

            # Attempt 2: token_embedder
            if single is None and hasattr(model, "token_embedder"):
                try:
                    single = model.token_embedder(x)
                    print(f"[trunk] used token_embedder", file=sys.stderr)
                except Exception as e:
                    print(f"[trunk] token_embedder failed: {e}", file=sys.stderr)

            # Attempt 3: embedding
            if single is None and hasattr(model, "embedding"):
                try:
                    single = model.embedding(x)
                    print(f"[trunk] used embedding", file=sys.stderr)
                except Exception as e:
                    print(f"[trunk] embedding failed: {e}", file=sys.stderr)

            if single is None:
                print(f"[trunk] no embedding module found", file=sys.stderr)
                return None

            # Run pairformer / trunk
            pair = None
            if hasattr(model, "pairformer"):
                try:
                    result = model.pairformer(single, pair)
                    if isinstance(result, tuple):
                        single, pair = result
                    else:
                        single = result
                    print(f"[trunk] used pairformer", file=sys.stderr)
                except Exception as e:
                    print(f"[trunk] pairformer failed: {e}", file=sys.stderr)

            if pair is None and hasattr(model, "trunk"):
                try:
                    result = model.trunk(single)
                    if isinstance(result, tuple):
                        single, pair = result
                    else:
                        single = result
                    print(f"[trunk] used trunk", file=sys.stderr)
                except Exception as e:
                    print(f"[trunk] trunk failed: {e}", file=sys.stderr)

            if pair is None:
                print(f"[trunk] no pair representation obtained", file=sys.stderr)
                return None

            # Cross-chain pair norm as IPSAE proxy
            pair   = pair.squeeze(0)                           # [N, N, H]
            cross  = pair[:binder_len, binder_len:]            # [B, T, H]
            return cross.norm(dim=-1).mean().item()

    parent_score = run_trunk(parent_onehot)
    if parent_score is None:
        raise RuntimeError("Trunk returned no pair representation")

    print(f"[trunk] parent_score={parent_score:.6f}", file=sys.stderr)

    scores = {}
    for cand in candidates:
        pos     = cand["design_idx"]
        new_aa  = cand["known_aa"]
        new_idx = AA_TO_IDX.get(new_aa, 0)

        sub_onehot = parent_onehot.clone()
        sub_onehot[pos]         = 0.0
        sub_onehot[pos, new_idx] = 1.0

        sub_score = run_trunk(sub_onehot)
        if sub_score is not None:
            scores[str(pos)] = sub_score - parent_score
        else:
            scores[str(pos)] = 0.0

    return scores, "pair_repr_norm"


def score_via_pae(candidates, pred_dir, binder_len, target_len):
    """
    Fallback: rank candidates by per-residue cross-chain PAE.
    Lower PAE = model more confident about interface placement = higher score.
    Returns (scores_dict, "pae_fallback").
    """
    import glob

    pae_files = sorted(glob.glob(os.path.join(pred_dir, "pae_*.npz")))
    if not pae_files:
        return {str(c["design_idx"]): 0.0 for c in candidates}, "fallback"

    expected    = binder_len + target_len
    accumulated = np.zeros(binder_len, dtype=np.float64)
    n_valid     = 0

    for pf in pae_files:
        data = np.load(pf)
        pae  = data["pae"] if "pae" in data else list(data.values())[0]
        if pae.ndim == 3:
            pae = pae[0]
        pae = pae.astype(np.float32)
        if pae.shape == (expected, expected):
            accumulated += pae[:binder_len, binder_len:].mean(axis=1)
            n_valid += 1

    if n_valid == 0:
        return {str(c["design_idx"]): 0.0 for c in candidates}, "fallback"

    cross_pae = accumulated / n_valid
    scores    = {}
    for cand in candidates:
        pos = cand["design_idx"]
        # Negate PAE so lower PAE → higher score
        scores[str(pos)] = -float(cross_pae[pos]) if pos < len(cross_pae) else 0.0

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

    # Attempt 1: full trunk scoring
    try:
        scores, method = score_via_trunk(
            candidates, args.pred_dir,
            args.binder_seq, args.binder_len, args.target_len
        )
        print(f"[trunk] trunk scoring succeeded via {method}", file=sys.stderr)
    except Exception as e:
        print(f"[trunk] trunk failed: {e}", file=sys.stderr)
        print(f"[trunk] falling back to PAE scoring", file=sys.stderr)

        # Attempt 2: PAE-based fallback
        try:
            scores, method = score_via_pae(
                candidates, args.pred_dir,
                args.binder_len, args.target_len
            )
            print(f"[trunk] PAE fallback succeeded", file=sys.stderr)
        except Exception as e2:
            print(f"[trunk] PAE fallback also failed: {e2}", file=sys.stderr)
            scores = {str(c["design_idx"]): 0.0 for c in candidates}
            method = "fallback"

    result = {"scores": scores, "method": method}
    with open(args.out, "w") as f:
        json.dump(result, f)

    print(
        f"[trunk] wrote {len(scores)} scores via {method}",
        file=sys.stderr
    )


if __name__ == "__main__":
    main()
