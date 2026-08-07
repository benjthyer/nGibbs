"""
Plot a heat map of the Si-Mg bulk composition (element moles) distribution of a
HeFESTo training dataset CSV, with Mg/Si = 0.9 and 1.1 reference lines.

Usage:
    python scripts/plot_Si_Mg_heatmap.py
    python scripts/plot_Si_Mg_heatmap.py --csv path/to/train.csv --out heatmap.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_DEFAULT = Path("data/MELTStables/HeFESTo/HeFESTo_Trainset041026_low_noise_adiabats.csv")
SI_COL = "Si(Bulk_comp_elements)"
MG_COL = "Mg(Bulk_comp_elements)"
N_BINS = 200
LINE_COLOR = "#eb6834"  # categorical slot 2 (orange), per repo dataviz palette

# Sequential blue ramp (light -> dark), per repo dataviz palette.
BLUE_RAMP = [
    "#fcfcfb",  # surface-1, so near-zero bins recede into the background
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]


def plot_heatmap(si: np.ndarray, mg: np.ndarray, out: Path | None) -> None:
    cmap = mcolors.LinearSegmentedColormap.from_list("blue_sequential", BLUE_RAMP)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    h = ax.hist2d(
        si, mg,
        bins=N_BINS,
        norm=mcolors.LogNorm(),
        cmap=cmap,
    )

    cbar = fig.colorbar(h[3], ax=ax)
    cbar.set_label("Count (log scale)", color="#0b0b0b")
    cbar.ax.tick_params(colors="#52514e")
    cbar.outline.set_visible(False)

    x_ref = np.array([si.min(), si.max()])
    ax.plot(x_ref, 0.9 * x_ref, linestyle="--", linewidth=1.6, color=LINE_COLOR, label="Mg = 0.9·Si")
    ax.plot(x_ref, 1.1 * x_ref, linestyle=":", linewidth=2.0, color=LINE_COLOR, label="Mg = 1.1·Si")

    ax.set_xlim(si.min(), si.max())
    ax.set_ylim(mg.min(), mg.max())

    ax.set_xlabel("Si (mol, bulk composition)", color="#0b0b0b")
    ax.set_ylabel("Mg (mol, bulk composition)", color="#0b0b0b")
    ax.set_title(f"Training dataset Si–Mg distribution  (N={len(si):,})", color="#0b0b0b")
    ax.tick_params(colors="#52514e")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.grid(alpha=0.15, color="#52514e")

    legend = ax.legend(loc="upper left", frameon=False, labelcolor="#0b0b0b")

    fig.tight_layout()

    if out is not None:
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved to {out}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a Si-Mg heat map of a HeFESTo training CSV.")
    parser.add_argument("--csv", type=Path, default=CSV_DEFAULT, help="Training dataset CSV path.")
    parser.add_argument("--out", type=Path, default=None, help="Save figure to this path instead of showing it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading {args.csv} ...")
    df = pd.read_csv(args.csv, usecols=[SI_COL, MG_COL], dtype=np.float32)

    out = args.out
    if out is None:
        out = args.csv.parent / f"{args.csv.stem}_SiMg_heatmap.png"

    print("Plotting heat map ...")
    plot_heatmap(df[SI_COL].to_numpy(), df[MG_COL].to_numpy(), out)


if __name__ == "__main__":
    main()
