"""
BundleHistograms: visualise training/test/validation distributions from ML-ready bundles.

Produces stacked 3-panel figures (one row per split) for:
  - Molar phase abundances (molar_labels, filtered to present-phase rows)
  - Bulk element mole fractions (from features)
  - Intensive phase component fractions (labels, filtered to present-phase rows)
  - Molar bulk Fe3/FeT (computed from labels + molar_labels via compToOx)
  - Scatter plots of every physical condition vs Pressure (T vs P, fO2 vs P, ...)

Usage (three individual paths):
    python BundleHistograms.py --train X_Train.tar.gz --test X_Test.tar.gz --val X_Valid.tar.gz

Usage (common stem, auto-discovers _Train / _Test / _Valid):
    python BundleHistograms.py --stem X

Output goes to  <script_dir>/plots/<stem>/  by default; override with --output-dir.
"""

from __future__ import annotations

import argparse
import gc
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- path setup (same depth as RecoveryPlots.py) ---
_here = Path(__file__).resolve()
_src = str(_here.parents[3])   # nGibbs/src
_builder = str(_here.parents[2])  # nGibbs/src/builder
for _p in (_src, _builder):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ngibbs.utils.file_utils import load_ml_bundle


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _resolve_bundle(path: Optional[str]) -> Optional[object]:
    if path is None:
        return None
    p = Path(path)
    # try adding .tar.gz if bare stem given
    for candidate in [p, p.with_suffix(".tar.gz"), Path(str(p) + ".tar.gz")]:
        if candidate.exists():
            try:
                return load_ml_bundle(str(candidate))
            except Exception as exc:
                print(f"Warning: failed to load {candidate}: {exc}")
                return None
    print(f"Warning: bundle not found at {path}")
    return None


def _stacked_hist(
    data_lists: list[np.ndarray],
    split_names: list[str],
    title: str,
    xlabel: str,
    out_path: Path,
    n_bins: int = 60,
    colors: list[str] | None = None,
    xlim: tuple[float, float] | None = None,
) -> None:
    """Save a figure with len(data_lists) vertically stacked histograms sharing the x-axis."""
    colors = colors or ["steelblue", "darkorange", "seagreen"]
    n = len(data_lists)

    # common range from all finite values
    all_vals = np.concatenate([d[np.isfinite(d)] for d in data_lists if len(d) > 0])
    if len(all_vals) == 0:
        return

    lo, hi = xlim if xlim else (float(np.nanmin(all_vals)), float(np.nanmax(all_vals)))
    if lo >= hi:
        lo, hi = lo - 0.5, hi + 0.5
    bins = np.linspace(lo, hi, n_bins + 1)

    fig, axes = plt.subplots(n, 1, figsize=(7, 2.8 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, data, name, color in zip(axes, data_lists, split_names, colors):
        finite = data[np.isfinite(data)] if len(data) > 0 else np.array([])
        if len(finite) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        else:
            ax.hist(finite, bins=bins, color=color, alpha=0.85, density=True)
        ax.set_ylabel("Density")
        ax.set_title(f"{name}   (n={len(data):,})", fontsize=9)

    axes[-1].set_xlabel(xlabel)
    fig.suptitle(title, fontsize=11, y=1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _stacked_scatter(
    x_lists: list[np.ndarray],
    y_lists: list[np.ndarray],
    split_names: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    max_pts: int = 5000,
    seed: int = 0,
    colors: list[str] | None = None,
) -> None:
    """Save a figure with len(x_lists) vertically stacked scatter subplots."""
    colors = colors or ["steelblue", "darkorange", "seagreen"]
    n = len(x_lists)
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(n, 1, figsize=(7, 3.0 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, x, y, name, color in zip(axes, x_lists, y_lists, split_names, colors):
        if len(x) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{name}", fontsize=9)
            continue
        idx = np.where(np.isfinite(x) & np.isfinite(y))[0]
        if len(idx) > max_pts:
            idx = rng.choice(idx, size=max_pts, replace=False)
        ax.scatter(x[idx], y[idx], s=2, alpha=0.4, color=color, rasterized=True)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{name}   (n_plotted={len(idx):,})", fontsize=9)

    axes[-1].set_xlabel(xlabel)
    fig.suptitle(title, fontsize=11, y=1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fe3/FeT bulk computation
# ---------------------------------------------------------------------------

def _compute_bulk_fe3_fet(
    molar_labels: np.ndarray,
    labels: np.ndarray,
    indexer,
) -> Optional[np.ndarray]:
    """
    Compute molar bulk Fe3/FeT from phase moles and intensive compositions.

    Builds extensive component moles (N, C), projects to oxide space via
    compToOx, then computes 2*n(Fe2O3) / (2*n(Fe2O3) + n(FeO)).
    Returns None when FeO or Fe2O3 is not in the Oxides list.
    """
    try:
        fe2o3_idx = indexer.Oxides.index("Fe2O3")
        feo_idx = indexer.Oxides.index("FeO")
    except ValueError:
        return None

    N = molar_labels.shape[0]
    ext = np.zeros((N, indexer.ncomps), dtype=np.float64)

    for phase, ph_idx in indexer.mass_phasedict.items():
        ph_moles = molar_labels[:, ph_idx]
        c_idx = indexer.label_indices[phase]
        if phase in indexer.label_indices_comp:
            vc_idx = indexer.label_indices_comp[phase]
            ext[:, c_idx] = ph_moles[:, None] * labels[:, vc_idx]
        elif len(c_idx) == 1:
            ext[:, c_idx[0]] = ph_moles

    bulk_ox = ext @ indexer.compToOx.astype(np.float64)  # (N, O) oxide moles
    fe3_moles = 2.0 * bulk_ox[:, fe2o3_idx]
    fe2_moles = bulk_ox[:, feo_idx]
    fe_total = fe3_moles + fe2_moles
    fe3_fet = np.where(fe_total > 1e-12, fe3_moles / fe_total, np.nan)
    return fe3_fet


# ---------------------------------------------------------------------------
# main plotting routine
# ---------------------------------------------------------------------------

def run_bundle_histograms(
    train_path: Optional[str],
    test_path: Optional[str],
    val_path: Optional[str],
    output_dir: str,
    n_bins: int = 60,
    max_scatter: int = 8000,
    seed: int = 42,
) -> None:
    named_paths = [("Train", train_path), ("Test", test_path), ("Valid", val_path)]
    split_bundles: list[tuple[str, object]] = []
    for name, path in named_paths:
        b = _resolve_bundle(path)
        if b is not None:
            split_bundles.append((name, b))

    if not split_bundles:
        raise RuntimeError("No bundles could be loaded. Check the provided paths.")

    split_names = [s for s, _ in split_bundles]
    bundles = [b for _, b in split_bundles]
    colors = ["steelblue", "darkorange", "seagreen"][: len(bundles)]

    # All bundles share the same indexer structure; use the first available one.
    indexer = bundles[0].ml_indexer

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out}")

    # -----------------------------------------------------------------------
    # 1. Physical conditions: T vs P and every other condition vs P
    # -----------------------------------------------------------------------
    feat_names: list[str] = list(indexer.featureNames)
    n_phys = len(feat_names)

    # Identify P and T column indices (first = P, second = T by convention)
    p_col = 0
    t_col = 1 if n_phys > 1 else None

    # Scatter: every physical feature vs P
    for fi, fname in enumerate(feat_names):
        if fi == p_col:
            continue
        x_lists = [b.features[:, fi] for b in bundles]
        y_lists = [b.features[:, p_col] for b in bundles]
        label_x = fname
        label_y = feat_names[p_col]
        _stacked_scatter(
            x_lists, y_lists,
            split_names,
            title=f"{fname}  vs  {feat_names[p_col]}",
            xlabel=label_x,
            ylabel=label_y,
            out_path=out / f"scatter_{_sanitize(fname)}_vs_{_sanitize(feat_names[p_col])}.png",
            max_pts=max_scatter,
            seed=seed,
            colors=colors,
        )
        print(f"  Scatter: {fname} vs {feat_names[p_col]}")

    # -----------------------------------------------------------------------
    # 2. Bulk element mole fraction histograms
    # -----------------------------------------------------------------------
    elkeys: list[str] = list(indexer.Elkeys)
    for k, el in enumerate(elkeys):
        col = n_phys + k
        data_lists = [
            b.features[:, col] if b.features is not None and b.features.shape[1] > col
            else np.array([], dtype=np.float32)
            for b in bundles
        ]
        _stacked_hist(
            data_lists, split_names,
            title=f"Bulk {el} (mole fraction)",
            xlabel=f"{el} mole fraction",
            out_path=out / f"bulk_{_sanitize(el)}.png",
            n_bins=n_bins,
            colors=colors,
        )
        print(f"  Bulk element histogram: {el}")

    # -----------------------------------------------------------------------
    # 3. Molar phase abundance histograms (present-phase rows only)
    # -----------------------------------------------------------------------
    for phase, ph_idx in indexer.mass_phasedict.items():
        data_lists = []
        for b in bundles:
            if b.molar_labels is None or b.binary_labels is None:
                data_lists.append(np.array([], dtype=np.float32))
                continue
            present = b.binary_labels[:, ph_idx] > 0.5
            data_lists.append(b.molar_labels[present, ph_idx])
        _stacked_hist(
            data_lists, split_names,
            title=f"{phase}  molar abundance (present rows)",
            xlabel="Phase moles (normalised)",
            out_path=out / f"phase_moles_{_sanitize(phase)}.png",
            n_bins=n_bins,
            colors=colors,
        )
        print(f"  Phase moles histogram: {phase}")

    # -----------------------------------------------------------------------
    # 4. Intensive phase component histograms (present-phase rows only)
    # -----------------------------------------------------------------------
    for phase, vc_indices in indexer.label_indices_comp.items():
        ph_idx = indexer.mass_phasedict[phase]
        c_indices = indexer.label_indices[phase]  # indices into C (for label_names)

        for k, (vc_idx, c_idx) in enumerate(zip(vc_indices, c_indices)):
            comp_name = indexer.label_names[c_idx]
            data_lists = []
            for b in bundles:
                if b.labels is None or b.binary_labels is None:
                    data_lists.append(np.array([], dtype=np.float32))
                    continue
                present = b.binary_labels[:, ph_idx] > 0.5
                data_lists.append(b.labels[present, vc_idx])
            _stacked_hist(
                data_lists, split_names,
                title=f"{phase}  |  {comp_name}  (present rows)",
                xlabel="Mole fraction",
                out_path=out / f"comp_{_sanitize(phase)}_{_sanitize(comp_name)}.png",
                n_bins=n_bins,
                xlim=(0.0, 1.0),
                colors=colors,
            )
        print(f"  Component histograms: {phase} ({len(vc_indices)} components)")

    # -----------------------------------------------------------------------
    # 5. Molar bulk Fe3/FeT
    # -----------------------------------------------------------------------
    fe3_computed = False
    for b_name, b in zip(split_names, bundles):
        if b.molar_labels is not None and b.labels is not None:
            test_val = _compute_bulk_fe3_fet(b.molar_labels[:1], b.labels[:1], indexer)
            if test_val is not None:
                fe3_computed = True
                break

    if fe3_computed:
        data_lists = []
        for b in bundles:
            if b.molar_labels is None or b.labels is None:
                data_lists.append(np.array([], dtype=np.float32))
                continue
            val = _compute_bulk_fe3_fet(b.molar_labels, b.labels, indexer)
            data_lists.append(val if val is not None else np.array([], dtype=np.float32))
        _stacked_hist(
            data_lists, split_names,
            title="Molar bulk Fe3/FeT",
            xlabel="Fe³⁺ / Fe_total (molar)",
            out_path=out / "bulk_Fe3_FeT.png",
            n_bins=n_bins,
            xlim=(0.0, 1.0),
            colors=colors,
        )
        print("  Fe3/FeT histogram generated.")
    else:
        print("  Skipping Fe3/FeT: FeO or Fe2O3 not found in Oxides list.")

    # -----------------------------------------------------------------------
    # 6. Physical condition histograms (P, T, fO2, ...)
    # -----------------------------------------------------------------------
    for fi, fname in enumerate(feat_names):
        data_lists = [b.features[:, fi] for b in bundles]
        _stacked_hist(
            data_lists, split_names,
            title=fname,
            xlabel=fname,
            out_path=out / f"condition_{_sanitize(fname)}.png",
            n_bins=n_bins,
            colors=colors,
        )
        print(f"  Condition histogram: {fname}")

    gc.collect()
    print(f"\nDone. {len(list(out.glob('*.png')))} plots saved to {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate stacked train/test/val histograms from ML-ready bundles."
    )
    group = p.add_argument_group("bundle paths (use --stem OR individual flags)")
    group.add_argument("--stem", default=None, help="Common path stem; appends _Train/_Test/_Valid.tar.gz")
    group.add_argument("--train", default=None, help="Path to training bundle (.tar.gz)")
    group.add_argument("--test", default=None, help="Path to test bundle (.tar.gz)")
    group.add_argument("--val", default=None, help="Path to validation bundle (.tar.gz)")

    p.add_argument("--output-dir", default=None, help="Directory for output plots (default: plots/<stem> or plots/histograms)")
    p.add_argument("--bins", type=int, default=60, help="Number of histogram bins (default: 60)")
    p.add_argument("--max-scatter", type=int, default=8000, help="Max points per split in scatter plots (default: 8000)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for scatter subsampling (default: 42)")
    return p


def main() -> None:
    args = build_parser().parse_args()

    train_path = args.train
    test_path = args.test
    val_path = args.val

    if args.stem:
        stem = args.stem
        train_path = train_path or f"{stem}_Train.tar.gz"
        test_path = test_path or f"{stem}_Test.tar.gz"
        val_path = val_path or f"{stem}_Valid.tar.gz"

    if not any([train_path, test_path, val_path]):
        print("Error: supply --stem or at least one of --train/--test/--val.", file=sys.stderr)
        sys.exit(1)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        plots_root = _here.parent / "plots"
        stem_name = Path(args.stem).name if args.stem else "histograms"
        output_dir = str(plots_root / stem_name)

    run_bundle_histograms(
        train_path=train_path,
        test_path=test_path,
        val_path=val_path,
        output_dir=output_dir,
        n_bins=args.bins,
        max_scatter=args.max_scatter,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
