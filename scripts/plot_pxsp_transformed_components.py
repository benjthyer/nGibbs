"""Histogram pyroxene and spinel component values after the PxSp change of basis.

Labels from an ML bundle are mapped through ``ml_indexer.PxSpTransform`` exactly
as in training (``labels @ PxSpTransform[subset, subset]``, see
``loadTrainData.load_ML_data``). The pyroxene (orthopyroxene, clinopyroxene) and
spinel columns are then pulled out and plotted:

1. ``*_components.png`` - every transformed component value, one histogram per
   phase overlaid on shared bins.
2. ``*_row_sums.png``   - for each row, the sum of that phase's transformed
   components, one histogram per phase overlaid on shared bins.

Bin edges are integer multiples of the bin width, so 0 is always a bin
boundary: negative values never share a bin with positive ones. Only rows where
the phase is present (``binary_labels``) are used unless ``--include-absent``.

Usage:
    python scripts/plot_pxsp_transformed_components.py \
        --bundle data/MLready/120/120ClosedFxBatch_Cr_NPT_Test.tar.gz \
        --output-dir /tmp/pxsp_plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

repo_src = str(Path(__file__).resolve().parents[1] / "src")
if repo_src not in sys.path:
    sys.path.insert(0, repo_src)

from ngibbs.utils.file_utils import load_ml_bundle

PXSP_PHASES = ["orthopyroxene", "clinopyroxene", "spinel"]
PHASE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def _resolve_bundle_path(path: Path) -> Path:
    if path.exists():
        return path
    if path.suffix == "":
        archive_path = path.with_name(path.name + ".tar.gz")
        if archive_path.exists():
            return archive_path
    raise FileNotFoundError(f"ML bundle not found: {path}")


def zero_aligned_bins(values: list[np.ndarray], n_bins: int,
                      value_range: tuple[float, float] | None = None) -> np.ndarray:
    """Edges spanning all ``values`` with width (hi - lo) / n_bins, placed at
    integer multiples of that width so that 0 is always an edge."""
    if value_range is None:
        lo = min(float(v.min()) for v in values if v.size)
        hi = max(float(v.max()) for v in values if v.size)
    else:
        lo, hi = value_range
    width = (hi - lo) / n_bins if hi > lo else max(abs(hi), 1.0) * 1e-6
    k_lo = int(np.floor(lo / width))
    k_hi = int(np.ceil(hi / width))
    if k_hi == k_lo:
        k_hi += 1
    return width * np.arange(k_lo, k_hi + 1, dtype=np.float64)


def transformed_phase_columns(bundle_path: Path, include_absent: bool) -> dict[str, dict]:
    bundle = load_ml_bundle(bundle_path, arrays=("labels", "binary_labels"))
    ix = bundle.ml_indexer
    subset = np.asarray(ix.compositional_component_subset)
    if bundle.labels.shape[1] != len(subset):
        raise ValueError(
            f"labels has {bundle.labels.shape[1]} columns but compositional_component_subset "
            f"has {len(subset)}; cannot apply the PxSp transform."
        )

    px_sp_sub = np.asarray(ix.PxSpTransform, dtype=np.float32)[np.ix_(subset, subset)]
    # Position of each full-label index inside the compositional subset (= labels columns).
    col_of = {int(full_idx): pos for pos, full_idx in enumerate(subset)}

    out = {}
    for phase in PXSP_PHASES:
        if phase not in ix.label_indices:
            print(f"Phase '{phase}' not in this bundle; skipping.")
            continue
        cols = np.array([col_of[int(i)] for i in ix.label_indices[phase]])
        rows = np.ones(bundle.labels.shape[0], dtype=bool)
        if not include_absent:
            rows = bundle.binary_labels[:, ix.mass_phasedict[phase]] > 0.5
        # PxSpTransform is block-diagonal by phase, so only this phase's columns of
        # the labels contribute to this phase's transformed columns.
        values = bundle.labels[rows][:, cols] @ px_sp_sub[np.ix_(cols, cols)]
        out[phase] = {
            "values": values,
            "names": [ix.label_names[i] for i in ix.label_indices[phase]],
        }
    return out


def _report(data: dict[str, dict]) -> None:
    for phase, d in data.items():
        v, sums = d["values"], d["values"].sum(axis=1)
        print(f"\n{phase}: {v.shape[0]:,} rows")
        for j, name in enumerate(d["names"]):
            col = v[:, j]
            print(f"  {name:<20s} min {col.min(): .4g}  max {col.max(): .4g}  "
                  f"negative {100 * (col < 0).mean():6.2f}%")
        print(f"  {'row sum':<20s} min {sums.min(): .4g}  max {sums.max(): .4g}  "
              f"mean {sums.mean(): .6g}")


def _overlaid_hist(series: dict[str, np.ndarray], n_bins: int, title: str, xlabel: str,
                   out_path: Path, log_y: bool,
                   value_range: tuple[float, float] | None) -> None:
    edges = zero_aligned_bins(list(series.values()), n_bins, value_range)
    fig, ax = plt.subplots(figsize=(9, 5))
    for (phase, vals), color in zip(series.items(), PHASE_COLORS):
        ax.hist(vals, bins=edges, histtype="step", linewidth=2, color=color,
                label=f"{phase} (n={vals.size:,})")
    if edges[0] < 0.0 < edges[-1]:
        ax.axvline(0.0, color="#52514e", linewidth=1, linestyle="--")
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(axis="y", color="#e5e4df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}  (bin width {edges[1] - edges[0]:.4g})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True, type=Path, help="ML-ready .tar.gz bundle")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write PNGs (default: <bundle dir>/Plots)")
    parser.add_argument("--bins", type=int, default=200, help="Approximate number of bins")
    parser.add_argument("--linear-y", action="store_true", help="Linear count axis (default log)")
    parser.add_argument("--include-absent", action="store_true",
                        help="Include rows where the phase is absent (all-zero labels)")
    parser.add_argument("--component-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"), help="Clip component histogram x-range")
    parser.add_argument("--sum-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"), help="Clip row-sum histogram x-range")
    args = parser.parse_args()

    bundle_path = _resolve_bundle_path(args.bundle)
    out_dir = args.output_dir or bundle_path.parent / "Plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = bundle_path.name.removesuffix(".tar.gz")

    data = transformed_phase_columns(bundle_path, args.include_absent)
    if not data:
        raise SystemExit("No pyroxene or spinel phases in this bundle.")
    _report(data)

    log_y = not args.linear_y
    _overlaid_hist({p: d["values"].ravel() for p, d in data.items()}, args.bins,
                   f"{stem}: PxSp-transformed component values",
                   "Transformed component value", out_dir / f"{stem}_pxsp_components.png",
                   log_y, args.component_range)
    _overlaid_hist({p: d["values"].sum(axis=1) for p, d in data.items()}, args.bins,
                   f"{stem}: per-row sum of PxSp-transformed components",
                   "Row sum within phase", out_dir / f"{stem}_pxsp_row_sums.png",
                   log_y, args.sum_range)


if __name__ == "__main__":
    main()
