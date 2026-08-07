"""
Plot a heat map of the Pressure-Temperature distribution of a HeFESTo training
dataset CSV or .npy array.

A .npy input is assumed to be a header-less 2D array whose columns are named
by a sibling CSV of the same stem (e.g. train.npy + train.csv, where the CSV
holds only the header row) — the layout produced alongside the bundled
HeFESTo training arrays.

Usage:
    python scripts/plot_training_PT_heatmap.py
    python scripts/plot_training_PT_heatmap.py --csv path/to/train.csv --out heatmap.png
    python scripts/plot_training_PT_heatmap.py --csv path/to/train.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_DEFAULT = Path("data/MELTHeFESTo_Trainset041026_low_noise_adiabats.csv")
P_COL = "P(GPa)(System_main)"
T_COL = "T(K)(System_main)"
N_BINS = 200

# Sequential blue ramp (light -> dark), per repo dataviz palette.
BLUE_RAMP = [
    "#fcfcfb",  # surface-1, so near-zero bins recede into the background
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]


def load_columns(path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    if path.suffix.lower() == ".npy":
        header_csv = path.with_suffix(".csv")
        if not header_csv.exists():
            raise FileNotFoundError(
                f"{path} is a .npy file — expected a sibling column-name CSV at {header_csv}"
            )
        with open(header_csv) as f:
            header = f.readline().strip().split(",")

        col_idx = {}
        for col in columns:
            if col not in header:
                raise ValueError(f"Column {col!r} not found in {header_csv}")
            col_idx[col] = header.index(col)

        arr = np.load(path, mmap_mode="r")
        return {col: np.asarray(arr[:, idx], dtype=np.float32) for col, idx in col_idx.items()}

    df = pd.read_csv(path, usecols=columns, dtype=np.float32)
    return {col: df[col].to_numpy() for col in columns}


def plot_heatmap(p: np.ndarray, t: np.ndarray, out: Path | None) -> None:
    cmap = mcolors.LinearSegmentedColormap.from_list("blue_sequential", BLUE_RAMP)

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    h = ax.hist2d(
        t, p,
        bins=N_BINS,
        norm=mcolors.LogNorm(),
        cmap=cmap,
    )

    cbar = fig.colorbar(h[3], ax=ax)
    cbar.set_label("Count (log scale)", color="#0b0b0b")
    cbar.ax.tick_params(colors="#52514e")
    cbar.outline.set_visible(False)

    ax.set_xlabel("Temperature (K)", color="#0b0b0b")
    ax.set_ylabel("Pressure (GPa)", color="#0b0b0b")
    ax.set_title(f"Training dataset P–T distribution  (N={len(p):,})", color="#0b0b0b")
    ax.invert_yaxis()
    ax.tick_params(colors="#52514e")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.grid(alpha=0.15, color="#52514e")

    fig.tight_layout()

    if out is not None:
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved to {out}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a P-T heat map of a HeFESTo training CSV.")
    parser.add_argument("--csv", type=Path, default=CSV_DEFAULT, help="Training dataset .csv or .npy path.")
    parser.add_argument("--out", type=Path, default=None, help="Save figure to this path instead of showing it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading {args.csv} ...")
    cols = load_columns(args.csv, [P_COL, T_COL])

    out = args.out
    if out is None:
        out = args.csv.parent / f"{args.csv.stem}_PT_heatmap.png"

    print("Plotting heat map ...")
    plot_heatmap(cols[P_COL], cols[T_COL], out)


if __name__ == "__main__":
    main()
