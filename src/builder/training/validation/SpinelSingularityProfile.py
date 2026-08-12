"""
SpinelSingularityProfile: diagnose rows that drive polish_negative_sp's batched
3x3 solve singular, and compare ground-truth vs. NN-predicted spinel
compositions for those rows.

`MidLevelNetwork.polish_negative_sp` (src/ngibbs/engine/NN.py) builds one 3x3
matrix per "illegal" row and solves the whole batch with a single
`torch.linalg.solve` call. If *any* row in that batch is singular, the whole
call raises; the code catches that, runs `polish_negative_spFe` (which mutates
FeO/spinel components), and recurses with `trial+1` -- up to trial 2, at which
point it gives up and re-raises. Because of that recursion, the composition
that is actually singular at the fatal attempt is usually NOT the raw NN
output: it's the raw output after one or two rounds of Fe-rebalancing. An
earlier version of this script only checked the pre-recursion (trial 0)
composition and could report zero singular rows even when the real code
crashes, simply because the failure only emerges after the mutation.

To stay faithful to the real control flow (including `polish_negative_spFe`'s
side effects between trials), this script does not reimplement the retry
logic. Instead it monkeypatches `model.polish_negative_sp` on the instance
(`_install_recorder` below) so every call -- including the recursive ones --
is intercepted: we rebuild the same 3x3 matrix from the exact
`intensiveComponents` tensor passed in, compute per-row determinants (instead
of letting one bad row crash the whole batched `torch.linalg.solve`), log any
singular rows together with their trial number and full component values,
then delegate to the original method so the real recursion/crash behavior is
unaffected. The final ValueError (if trial 2 still fails) is caught at the
top level so the script can keep going and report on what it found.

It also reproduces a real bug found while investigating this: NN.py:776 checks
`if 'chromitte' in sp` (double-t typo) instead of `'chromite' in sp`, so the
lookup never matches and chromite is *silently dropped* from the constraint
algebra for every model that has it (all MELTS spinels; HeFESTo's key is
'picro-chromite' and is unaffected by this particular typo, but is *also*
never included by name in this function). For every logged singular row we
also re-evaluate its exact (post-mutation) composition with the typo fixed,
to see how many rows the fix alone would have rescued vs. rows that are
genuinely singular regardless (e.g. true near-end-member compositions).

Usage:
    python SpinelSingularityProfile.py --bundle path/to/X_Valid.tar.gz --model path/to/model.pt

By default this walks the *entire* bundle in chunks (`--chunk-size`), not a
random subsample, since a row's singularity doesn't depend on which other
rows share its batch -- only on whether the row is included at all. Use
`--max-samples` to cap it for a quick pass.

Output goes to  <script_dir>/plots/spinel_singularity/<model_stem>/  by default;
override with --output-dir.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repo root and src are on path (same convention as RecoveryPlots.py)
repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
src_path = str(Path(__file__).resolve().parents[2])
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from ngibbs.engine.NN import rebuild_MELTS_model
from ngibbs.engine.emulator import NN_MELTS
from ngibbs.utils.file_utils import load_ml_bundle

# Order matters: matches the column order used everywhere below.
SPINEL_COMPONENTS = ["chromite", "hercynite", "magnetite", "spinel", "ulvospinel"]

# Determinant/component magnitudes below this are treated as "zero". Components
# live on a molar-fraction (0-1) scale, so 1e-6 is comfortably below any
# meaningful abundance while still being well above float32 noise.
DEFAULT_TOL = 1e-6


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def build_spinel_matrix(intensiveComponents: torch.Tensor, sp: dict):
    """
    Rebuild the exact per-row 3x3 system that polish_negative_sp solves
    (NN.py lines 767-853), for every row at once, reproducing the as-shipped
    'chromitte' typo (chromite is always treated as absent) since that's what
    actually executes in production.

    Returns
    -------
    M : (N, 3, 3) coefficient matrices
    illegal : (N,) bool, constraint-violation mask (same test NN.py uses to
        decide which rows need solving at all)
    """
    i2 = sp["hercynite"]
    i3 = sp["magnetite"]
    i4 = sp["spinel"]
    i5 = sp["ulvospinel"]

    # Mirrors NN.py:776 exactly, typo included.
    c1 = torch.zeros_like(intensiveComponents[:, i2])
    c2 = intensiveComponents[:, i2]
    c3 = intensiveComponents[:, i3]
    c4 = intensiveComponents[:, i4]
    c5 = intensiveComponents[:, i5]

    pos1 = c1 + c2 + c3 + 2.25 * c5
    neg1 = 19 * c4
    illegal1 = neg1 > pos1

    pos2 = c2 + c4
    neg2 = (2 / 3) * c3 + 0.25 * c5
    illegal2 = neg2 > pos2

    illegal = illegal1 | illegal2

    A = c1 + c2
    B = c3 + c5
    C = c4
    L1_c3c5 = c3 + 2.25 * c5
    L2_c3c5 = (2 / 3) * c3 + 0.25 * c5

    N = intensiveComponents.shape[0]
    M = torch.zeros((N, 3, 3), dtype=intensiveComponents.dtype, device=intensiveComponents.device)

    M[:, 0, 0] = A
    M[:, 0, 1] = B
    M[:, 0, 2] = C
    M[:, 1, 0] = A
    M[:, 1, 1] = L1_c3c5
    M[:, 1, 2] = -19 * C
    M[:, 2, 0] = c2
    M[:, 2, 1] = -L2_c3c5
    M[:, 2, 2] = C

    return M, illegal


def _det_from_values(vals: dict, tol: float) -> np.ndarray:
    """Numpy version of the same determinant, from named component arrays."""
    c1 = vals.get("chromite", np.zeros_like(vals["hercynite"]))
    c2 = vals["hercynite"]
    c3 = vals["magnetite"]
    c4 = vals["spinel"]
    c5 = vals["ulvospinel"]

    A = c1 + c2
    B = c3 + c5
    C = c4
    L1_c3c5 = c3 + 2.25 * c5
    L2_c3c5 = (2 / 3) * c3 + 0.25 * c5

    n = c2.shape[0]
    M = np.zeros((n, 3, 3), dtype=np.float64)
    M[:, 0, 0] = A
    M[:, 0, 1] = B
    M[:, 0, 2] = C
    M[:, 1, 0] = A
    M[:, 1, 1] = L1_c3c5
    M[:, 1, 2] = -19 * C
    M[:, 2, 0] = c2
    M[:, 2, 1] = -L2_c3c5
    M[:, 2, 2] = C
    return np.linalg.det(M)


def _classify_mode(vals: dict, tol: float) -> np.ndarray:
    c1 = vals.get("chromite", np.zeros_like(vals["hercynite"]))
    A = c1 + vals["hercynite"]
    B = vals["magnetite"] + vals["ulvospinel"]
    C = vals["spinel"]
    A0, B0, C0 = np.abs(A) < tol, np.abs(B) < tol, np.abs(C) < tol

    mode = np.full(A.shape[0], "other/near-singular", dtype=object)
    mode[C0 & ~A0 & ~B0] = "C=0 (no spinel end-member)"
    mode[B0 & ~A0 & ~C0] = "B=0 (no magnetite+ulvospinel)"
    mode[A0 & ~B0 & ~C0] = "A=0 (no chromite+hercynite)"
    mode[A0 & B0] = "A&B=0 (pure Mg-spinel end-member)"
    return mode


def _install_recorder(model, tol: float):
    """
    Monkeypatch model.polish_negative_sp (instance attribute, so it shadows
    the class method but the recursive `self.polish_negative_sp(...)` call
    inside the real method still resolves back to this wrapper) to log every
    singular row -- at whatever trial it's encountered -- without altering
    control flow: we delegate to the original implementation immediately
    after recording, so the real recursion/crash behavior is untouched.

    Returns (records, restore). `records` is appended to in place as the
    forward pass runs; each entry is a dict with keys: trial, local_row_idx
    (positions within the `intensiveComponents` tensor passed to *this*
    call), det, and one array per component in SPINEL_COMPONENTS present in
    `sp` (raw values, unaffected by the chromite-lookup bug -- these are read
    directly off intensiveComponents, not derived from the buggy A/B/C terms).
    """
    sp = model.detail_label_indices["spinel"]
    present_components = [c for c in SPINEL_COMPONENTS if c in sp]
    original_bound = model.polish_negative_sp
    records = []

    def recorder(intensiveComponents, trial=0):
        M, illegal = build_spinel_matrix(intensiveComponents, sp)
        if illegal.any():
            row_idx = torch.nonzero(illegal, as_tuple=False).squeeze(-1)
            det = torch.linalg.det(M[row_idx])
            singular_local = det.abs() < tol
            if singular_local.any():
                sel = row_idx[singular_local]
                rec = {
                    "trial": trial,
                    "local_row_idx": sel.detach().cpu().numpy(),
                    "det": det[singular_local].detach().cpu().numpy(),
                }
                for comp in present_components:
                    rec[comp] = intensiveComponents[sel, sp[comp]].detach().cpu().numpy()
                records.append(rec)
        return original_bound(intensiveComponents, trial=trial)

    model.polish_negative_sp = recorder

    def restore():
        model.polish_negative_sp = original_bound

    return records, restore


def _hist_panel(gt, pred, title, xlabel, out_path, n_bins=40):
    plt.figure(figsize=(6, 4))
    lo = float(min(np.min(gt) if gt.size else 0.0, np.min(pred) if pred.size else 0.0))
    hi = float(max(np.max(gt) if gt.size else 1.0, np.max(pred) if pred.size else 1.0))
    if hi <= lo:
        hi = lo + 1e-6
    bins = np.linspace(lo, hi, n_bins + 1)
    plt.hist(gt, bins=bins, alpha=0.5, color="blue", label=f"Ground truth (n={gt.size})", density=True)
    plt.hist(pred, bins=bins, alpha=0.5, color="red", label=f"Predicted at failing trial (n={pred.size})", density=True)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("Normalized frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_profile(
    bundle_path: str,
    model_path: str,
    output_dir: str,
    max_samples: int | None = None,
    chunk_size: int = 4096,
    seed: int = 1337,
    use_cuda: bool = False,
    tol: float = DEFAULT_TOL,
) -> None:
    bundle = load_ml_bundle(bundle_path)
    model = rebuild_MELTS_model(model_path)
    emulator = NN_MELTS(model, cuda=use_cuda)

    ml_indexer = model.ml_indexer
    if "spinel" not in ml_indexer.detail_label_indices:
        raise ValueError("This model/bundle has no 'spinel' phase; nothing to profile.")
    sp = ml_indexer.detail_label_indices["spinel"]
    present_components = [c for c in SPINEL_COMPONENTS if c in sp]
    missing = [c for c in SPINEL_COMPONENTS if c not in sp]
    if missing:
        print(f"Note: model spinel components are {list(sp.keys())}; missing from the standard "
              f"5-component set: {missing} (HeFESTo names its Cr end-member 'picro-chromite').")

    if "melts-liquid" not in model.label_indices_comp:
        print("WARNING: 'melts-liquid' is not in this model's label_indices_comp. "
              "MidLevelNetwork.forward() only calls polish_negative_sp inside the "
              "'melts-liquid' branch, so forward() will never invoke it for this model "
              "(this is the HeFESTo/subsolidus-only path) -- expect zero records below. "
              "If polish_negative_sp is being called directly elsewhere for this model, "
              "profile that call site instead of forward().")

    px_sp_sub = ml_indexer.PxSpTransform[np.ix_(ml_indexer.compositional_component_subset,
                                                 ml_indexer.compositional_component_subset)]

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    n_total = bundle.features.shape[0]
    if max_samples is not None and 0 < max_samples < n_total:
        rng = np.random.default_rng(seed)
        row_order = rng.choice(n_total, size=max_samples, replace=False)
    else:
        row_order = np.arange(n_total)

    all_records = []  # each entry gets a 'global_idx' array added, mapping into bundle rows
    n_chunks_crashed = 0

    for start in range(0, len(row_order), chunk_size):
        chunk_idx = row_order[start:start + chunk_size]
        features_tensor = torch.tensor(bundle.features[chunk_idx], device=device, dtype=torch.float32)
        features_norm = emulator.norm_features.norm(features_tensor)

        # Probe pass: NN_only=True never touches polish_negative_sp, so this is
        # side-effect-free. It reproduces the exact likelihoods/superliquidus
        # computation forward() does *before* the force-saturation loop and
        # before any spinel/px polishing, purely so we can recover which local
        # rows correspond to `non_super` (the subset polish_negative_sp is
        # actually called on) and map them back to global bundle indices.
        with torch.no_grad():
            likelihoods_probe, *_ = model.forward(features_norm, detailed=True, NN_only=True)
        binary_pred0 = (likelihoods_probe > 0.5).float()
        if "melts-liquid" in model.label_indices_comp:
            superliquidus = binary_pred0.sum(dim=1) == 1
        else:
            superliquidus = torch.zeros(binary_pred0.size(0), dtype=torch.bool, device=binary_pred0.device)
        non_super = ~superliquidus
        orig_idx_of_subset_row = torch.nonzero(non_super, as_tuple=False).squeeze(-1).detach().cpu().numpy()

        records, restore = _install_recorder(model, tol)
        try:
            with torch.no_grad():
                model.forward(features_norm, detailed=True)
        except ValueError as exc:
            n_chunks_crashed += 1
            print(f"[chunk starting at row_order[{start}]] forward() crashed as production does: {exc}")
        finally:
            restore()

        for rec in records:
            rec["global_idx"] = chunk_idx[orig_idx_of_subset_row[rec["local_row_idx"]]]
        all_records.extend(records)

    print(f"\n=== Spinel constraint-solve profile ===")
    print(f"Rows scanned: {len(row_order)} (of {n_total} total)  |  chunks that crashed: {n_chunks_crashed}")

    if not all_records:
        print("No singular rows encountered. If the real pipeline crashes on this bundle/model, "
              "try a larger --max-samples (or drop it to scan the whole bundle) and confirm the "
              "bundle/model paths match the run that failed.")
        return

    by_trial = {}
    for rec in all_records:
        by_trial.setdefault(rec["trial"], set()).update(rec["global_idx"].tolist())
    for trial in sorted(by_trial):
        print(f"  trial {trial}: {len(by_trial[trial])} distinct rows singular")

    max_trial_seen = max(by_trial)
    terminal_rows = by_trial[max_trial_seen]
    born_singular = terminal_rows & by_trial.get(0, set())
    print(f"\nTerminal (highest-trial) singular rows -- these are the ones actually "
          f"responsible for a crash: {len(terminal_rows)}")
    print(f"Of those, already singular at trial 0 (i.e. in the raw NN output, before "
          f"any polish_negative_spFe rebalancing): {len(born_singular)}")
    print(f"Of those, only became singular after Fe-rebalancing (trial 1 or 2): "
          f"{len(terminal_rows) - len(born_singular)}")

    # Take, for each terminal row, its last-recorded (highest-trial) component
    # snapshot -- the exact composition state at the failing solve attempt.
    terminal_records = [r for r in all_records if r["trial"] == max_trial_seen]
    global_to_vals = {}
    for rec in terminal_records:
        for local_i, gidx in enumerate(rec["global_idx"]):
            global_to_vals[int(gidx)] = {comp: rec[comp][local_i] for comp in present_components}

    terminal_global = np.array(sorted(global_to_vals.keys()))
    pred_vals = {comp: np.array([global_to_vals[g][comp] for g in terminal_global]) for comp in present_components}

    det_asshipped = _det_from_values(pred_vals, tol)
    mode_asshipped = _classify_mode(pred_vals, tol)
    print("\nFailure-mode breakdown (terminal singular rows, as-shipped 'chromitte' typo):")
    vals_u, counts_u = np.unique(mode_asshipped, return_counts=True)
    for v, c in sorted(zip(vals_u.tolist(), counts_u.tolist()), key=lambda kv: -kv[1]):
        print(f"  {v:<40s} {c}")

    if "chromite" in sp:
        fixed_vals = dict(pred_vals)  # chromite already present under its real name
        det_fixed = _det_from_values(fixed_vals, tol)
        rescued = (np.abs(det_asshipped) < tol) & (np.abs(det_fixed) >= tol)
        print(f"\nOf {len(terminal_global)} terminal singular rows, fixing the 'chromitte' typo "
              f"(folding real chromite into A) alone resolves: {int(rescued.sum())} "
              f"({100 * rescued.sum() / max(len(terminal_global), 1):.1f}%)")
        print("(Remaining rows are genuinely singular for other reasons -- B=0 or C=0 -- "
              "and would need separate handling.)")

    # ---- Ground truth comparison + histograms ----
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    gt_labels_trans = bundle.labels[terminal_global] @ px_sp_sub
    gt_np = np.asarray(gt_labels_trans)

    for comp in present_components:
        col = sp[comp]
        _hist_panel(
            gt_np[:, col],
            pred_vals[comp],
            title=f"Spinel '{comp}' composition\n(rows where polish_negative_sp's matrix is terminally singular)",
            xlabel=f"{comp} molar fraction",
            out_path=output_path / f"{_sanitize(comp)}_singular_rows_hist.png",
        )

    plt.figure(figsize=(6, 4))
    plt.hist(np.log10(np.abs(det_asshipped) + 1e-30), bins=60, alpha=0.7, color="red", label="As-shipped (chromitte typo)")
    if "chromite" in sp:
        plt.hist(np.log10(np.abs(det_fixed) + 1e-30), bins=60, alpha=0.5, color="green", label="Chromite-fixed")
    plt.axvline(np.log10(tol), color="black", linestyle="--", label=f"tol={tol:g}")
    plt.xlabel("log10(|det(M)|)")
    plt.ylabel("Count")
    plt.title("Determinant magnitude at the failing trial, terminal singular rows")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "determinant_log10_hist.png", dpi=200)
    plt.close()

    print(f"\nPlots written to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile rows that make polish_negative_sp's linear solve singular."
    )
    parser.add_argument("--bundle", required=True, help="Path to validation/test bundle (.tar.gz)")
    parser.add_argument("--model", required=True, help="Path to trained model (.pt or .zip)")
    parser.add_argument("--output-dir", default=None, help="Output directory for plots (default: script_dir/plots/spinel_singularity/<model_stem>)")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap rows scanned (default: scan the entire bundle)")
    parser.add_argument("--chunk-size", type=int, default=4096, help="Rows per forward() call while scanning")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed used only when --max-samples subsamples the bundle")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL, help="Abs. threshold below which det(M) or a component is treated as zero")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.output_dir is None:
        script_dir = Path(__file__).parent
        model_name = Path(args.model).stem
        output_dir = str(script_dir / "plots" / "spinel_singularity" / model_name)
    else:
        output_dir = args.output_dir

    run_profile(
        bundle_path=args.bundle,
        model_path=args.model,
        output_dir=output_dir,
        max_samples=args.max_samples,
        chunk_size=args.chunk_size,
        seed=args.seed,
        use_cuda=args.cuda,
        tol=args.tol,
    )


if __name__ == "__main__":
    main()
