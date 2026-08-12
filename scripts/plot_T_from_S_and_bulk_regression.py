"""
Plot the T = f(S, bulk composition) regression fit from fit_T_from_S_and_bulk.py.

Reuses that script's exact loading/filtering/fitting logic (same rows, same
features, same model) so the plots always match the numbers it reports, then
renders:
  1. Predicted vs. actual T, colored by signed residual, with the y=x line and
     outliers (|residual| > --threshold) ringed.
  2. Residual vs. S.
  3. Residual vs. P.
  4. Residual histogram with +/- threshold marked.

Also prints, sorted by |correlation|, how strongly each bulk element (and S, P)
correlates with the *residual* - useful for spotting which composition axis is
driving the outliers.

Usage:
    python scripts/plot_T_from_S_and_bulk_regression.py \
        --csv data/MELTStables/HeFESTo/HeFESTo_TrainsetMar2NTP.csv

    python scripts/plot_T_from_S_and_bulk_regression.py \
        --csv data/MELTStables/HeFESTo/HeFESTo_TrainsetMar2NTP.csv \
        --threshold 200 --out T_fit_diagnostics.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_T_from_S_and_bulk import P_MAX_DEFAULT, compute_T_from_S_and_bulk_regression  # noqa: E402

# Diverging blue -> neutral -> red, matching this repo's chart palette
# (blue #2a78d6 / red #e34948, neutral gray #f0efec midpoint).
RESIDUAL_CMAP = LinearSegmentedColormap.from_list(
    "residual_diverging", ["#184f95", "#2a78d6", "#f0efec", "#e34948", "#8a2323"]
)
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
MUTED_TEXT = "#898781"
PRIMARY_TEXT = "#0b0b0b"
OUTLIER_RING = "#0b0b0b"


def _style_axes(ax) -> None:
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_COLOR)
    ax.tick_params(colors=MUTED_TEXT, labelsize=9)
    ax.set_axisbelow(True)


def plot_regression_diagnostics(
    csv_path: Path, out_path: Path, threshold: float, p_max: float | None = P_MAX_DEFAULT
) -> None:
    result = compute_T_from_S_and_bulk_regression(csv_path, p_max=p_max)
    t = result["t"]
    s = result["s"]
    p = result["p"]
    t_pred = result["t_pred"]
    r2 = result["r2"]
    n = result["n"]
    bulk_cols = result["bulk_cols"]
    bulk = result["bulk"]

    residual = t - t_pred
    is_outlier = np.abs(residual) > threshold
    frac_outlier = is_outlier.mean()

    print(f"Rows: {n}   R^2 = {r2:.6g}")
    print(
        f"|residual| > {threshold:g} K: {is_outlier.sum()} / {n} "
        f"({frac_outlier:.1%})"
    )

    # Correlate residual against S, P, and each bulk element to help point at
    # which axis is driving the outliers.
    print("\nCorrelation of residual with each feature (sorted by |r|):")
    corr_rows = [("S", s), ("P", p)]
    for col, arr in zip(bulk_cols, bulk):
        corr_rows.append((col.replace("(Bulk_comp_elements)", "").strip(), arr))
    corrs = [(name, np.corrcoef(arr, residual)[0, 1]) for name, arr in corr_rows]
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, c in corrs:
        print(f"  {name:<8} r = {c:+.3f}")

    vmax = max(abs(residual.min()), abs(residual.max()), threshold)
    norm = Normalize(vmin=-vmax, vmax=vmax)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.patch.set_facecolor("#fcfcfb")
    ax_parity, ax_vs_s, ax_vs_p, ax_hist = axes.flat

    # 1. Predicted vs actual
    sc = ax_parity.scatter(
        t, t_pred, c=residual, cmap=RESIDUAL_CMAP, norm=norm,
        s=14, linewidths=0, alpha=0.85, zorder=2,
    )
    ax_parity.scatter(
        t[is_outlier], t_pred[is_outlier], facecolors="none",
        edgecolors=OUTLIER_RING, linewidths=0.8, s=32, zorder=3,
        label=f"|residual| > {threshold:g} K",
    )
    lims = [min(t.min(), t_pred.min()), max(t.max(), t_pred.max())]
    ax_parity.plot(lims, lims, color=AXIS_COLOR, linewidth=1.2, zorder=1)
    ax_parity.set_xlabel("Actual T (K)")
    ax_parity.set_ylabel("Predicted T (K)")
    ax_parity.set_title(f"Predicted vs. actual  (R² = {r2:.4f})", color=PRIMARY_TEXT, fontsize=11)
    ax_parity.legend(loc="upper left", frameon=False, fontsize=8)
    cbar = fig.colorbar(sc, ax=ax_parity, pad=0.02)
    cbar.set_label("Residual (K)", color=MUTED_TEXT, fontsize=8)
    cbar.ax.tick_params(colors=MUTED_TEXT, labelsize=7)
    _style_axes(ax_parity)

    # 2. Residual vs S
    ax_vs_s.axhline(0, color=AXIS_COLOR, linewidth=1.0, zorder=1)
    ax_vs_s.axhline(threshold, color=AXIS_COLOR, linewidth=0.8, linestyle="--", zorder=1)
    ax_vs_s.axhline(-threshold, color=AXIS_COLOR, linewidth=0.8, linestyle="--", zorder=1)
    ax_vs_s.scatter(s, residual, c=residual, cmap=RESIDUAL_CMAP, norm=norm, s=12, linewidths=0, alpha=0.8, zorder=2)
    ax_vs_s.set_xlabel("S")
    ax_vs_s.set_ylabel("Residual (K)")
    ax_vs_s.set_title("Residual vs. S", color=PRIMARY_TEXT, fontsize=11)
    _style_axes(ax_vs_s)

    # 3. Residual vs P
    ax_vs_p.axhline(0, color=AXIS_COLOR, linewidth=1.0, zorder=1)
    ax_vs_p.axhline(threshold, color=AXIS_COLOR, linewidth=0.8, linestyle="--", zorder=1)
    ax_vs_p.axhline(-threshold, color=AXIS_COLOR, linewidth=0.8, linestyle="--", zorder=1)
    ax_vs_p.scatter(p, residual, c=residual, cmap=RESIDUAL_CMAP, norm=norm, s=12, linewidths=0, alpha=0.8, zorder=2)
    ax_vs_p.set_xlabel("P (GPa)")
    ax_vs_p.set_ylabel("Residual (K)")
    ax_vs_p.set_title("Residual vs. P", color=PRIMARY_TEXT, fontsize=11)
    _style_axes(ax_vs_p)

    # 4. Residual histogram
    ax_hist.hist(residual, bins=60, color="#2a78d6", edgecolor="none", zorder=2)
    ax_hist.axvline(threshold, color=AXIS_COLOR, linewidth=0.8, linestyle="--", zorder=1)
    ax_hist.axvline(-threshold, color=AXIS_COLOR, linewidth=0.8, linestyle="--", zorder=1)
    ax_hist.set_xlabel("Residual (K)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(
        f"Residual distribution  ({frac_outlier:.1%} beyond ±{threshold:g} K)",
        color=PRIMARY_TEXT, fontsize=11,
    )
    _style_axes(ax_hist)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"\nFigure saved to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot diagnostics for the T = f(S, bulk composition) regression."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/MELTStables/HeFESTo/HeFESTo_TrainsetMar2NTP.csv"),
        help="Path to HeFESTo data (standalone CSV, or BigMetaTable .npy/.csv pair).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path (default: <csv stem>_regression_diagnostics.png next to the CSV).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=200.0,
        help="Residual magnitude (K) above which a point is flagged as an outlier (default: 200).",
    )
    parser.add_argument(
        "--p-max",
        type=float,
        default=P_MAX_DEFAULT,
        help=(
            "Drop rows with P >= this value (GPa) before fitting/plotting. "
            "Default: no pressure bound."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out
    if out_path is None:
        csv_base = args.csv if args.csv.suffix else args.csv
        out_path = csv_base.parent / f"{csv_base.stem}_regression_diagnostics.png"
    plot_regression_diagnostics(args.csv, out_path, args.threshold, p_max=args.p_max)


if __name__ == "__main__":
    main()
