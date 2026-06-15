"""
Plot 300 random Martian P-T paths using get_T_mars and GEOROC compositions.

Replicates the make_PT_path_martian workflow end-to-end without writing
simulation directories: samples compositions from GEOROC, speciates iron,
normalizes to 24 element moles, then generates the conductive + adiabatic
geotherm for random S and P_lit values drawn from the same distributions used
in prepare_HeFESTo_tree_Mars.

Paths are plotted noise-free so the geotherm structure is visible.

Usage:
    python scripts/plot_mars_PT_paths.py
    python scripts/plot_mars_PT_paths.py --georoc path/to/georoc.csv --n 300 --out plot.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

repo_root = Path(__file__).resolve().parents[1]
src_root = repo_root / "src"
for p in (str(repo_root), str(src_root)):
    if p not in sys.path:
        sys.path.insert(0, p)

from builder.HeFESTo.HeFESTo_functions import (
    _build_oxide_wt_from_row,
    _speciate_iron_and_normalize,
    _oxide_wt_to_element_moles,
    _normalize_total_moles,
    get_T_mars,
)

GEOROC_DEFAULT = repo_root / "data/MELTStables/GEOROC/GEOROC_PETDB_UNFILTERED_WHOLEROCK_TRAIN.csv"
N_DEFAULT = 300
P_GRID = np.linspace(0.01, 25.0, 200)   # GPa — avoids 0 for lithosphere ratio
COLD_TEMP = 220.0                         # K surface anchor (same as prepare_HeFESTo_tree_Mars)
TARGET_MOLES = 24.0


def _mars_geotherm(S: float, P_lit: float, element_moles: dict) -> np.ndarray:
    """Return T(K) along P_GRID for one Mars simulation (no noise)."""
    T = get_T_mars(S=S, P=P_GRID, **element_moles)

    conductive = P_GRID <= P_lit
    T_lit_base = get_T_mars(S=S, P=np.array([P_lit]), **element_moles)[0]
    T[conductive] = COLD_TEMP + (T_lit_base - COLD_TEMP) * (P_GRID[conductive] / P_lit)

    return T


def sample_compositions(georoc_df: pd.DataFrame, n: int, rng: np.random.Generator) -> list[dict]:
    mgo_col = "MgO"
    mgo_num = pd.to_numeric(georoc_df[mgo_col], errors="coerce").fillna(0.0)
    ultramafic = georoc_df[mgo_num > 20.0]
    mafic = georoc_df[(mgo_num > 5.5) & (mgo_num <= 20.0)]

    n_um = int(n * 3 / 5)
    n_m = n - n_um

    rows_um = ultramafic.sample(n=n_um, replace=True, random_state=int(rng.integers(1 << 31)))
    rows_m = mafic.sample(n=n_m, replace=True, random_state=int(rng.integers(1 << 31)))
    subset = pd.concat([rows_um, rows_m]).sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)

    fe3_fet = np.concatenate([
        np.linspace(0.0, 0.05, int(n * 4 / 5)),
        np.linspace(0.05, 0.20, n - int(n * 4 / 5)),
    ])
    rng.shuffle(fe3_fet)

    element_moles_list = []
    for i, (_, row) in enumerate(subset.iterrows()):
        base_wt = _build_oxide_wt_from_row(row)
        speciated = _speciate_iron_and_normalize(base_wt, float(fe3_fet[i]))
        moles = _oxide_wt_to_element_moles(speciated)
        moles_noised = {el: mol * rng.uniform(0.95,1.05) for el, mol in moles.items()}  # 5% noise
        moles_noised = _normalize_total_moles(moles_noised, TARGET_MOLES)
        element_moles_list.append(moles_noised)

    return element_moles_list


def plot_paths(element_moles_list: list[dict], Ss: np.ndarray, P_lits: np.ndarray, out: Path | None) -> None:
    norm = mcolors.Normalize(vmin=Ss.min(), vmax=Ss.max())
    
    cmap = cm.plasma

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    # ---- left panel: color by entropy S ----
    ax = axes[0]
    for i, (moles, S, P_lit) in enumerate(zip(element_moles_list, Ss, P_lits)):
        T = _mars_geotherm(S, P_lit, moles)
        color = cmap(norm(S))
        ax.plot(T, P_GRID, color=color, alpha=0.35, linewidth=0.8)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="S  (J/g/K)")
    ax.set_xlabel("Temperature  (K)")
    ax.set_ylabel("Pressure  (GPa)")
    ax.set_title("Mars P–T paths — colored by entropy")
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.grid(alpha=0.2)

    # ---- right panel: color by lithosphere base pressure ----
    norm_plit = mcolors.Normalize(vmin=P_lits.min(), vmax=P_lits.max())
    cmap_plit = cm.viridis

    ax2 = axes[1]
    for i, (moles, S, P_lit) in enumerate(zip(element_moles_list, Ss, P_lits)):
        T = _mars_geotherm(S, P_lit, moles)
        color = cmap_plit(norm_plit(P_lit))
        ax2.plot(T, P_GRID, color=color, alpha=0.35, linewidth=0.8)
        ax2.axhline(P_lit, color=color, linewidth=0.4, alpha=0.2, linestyle=":")

    sm2 = cm.ScalarMappable(cmap=cmap_plit, norm=norm_plit)
    sm2.set_array([])
    fig.colorbar(sm2, ax=ax2, label="P_lit  (GPa)")
    ax2.set_xlabel("Temperature  (K)")
    ax2.set_title("Mars P–T paths — colored by lithosphere thickness")
    ax2.invert_yaxis()
    ax2.set_xlim(left=0)
    ax2.grid(alpha=0.2)

    fig.suptitle(f"Martian geotherms  (N={len(Ss)}, get_T_mars R²=0.9929)", fontsize=13)
    plt.tight_layout()

    if out is not None:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot random Martian P-T paths from GEOROC compositions.")
    parser.add_argument("--georoc", type=Path, default=GEOROC_DEFAULT, help="GEOROC CSV file path.")
    parser.add_argument("--n", type=int, default=N_DEFAULT, help="Number of paths to plot.")
    parser.add_argument("--out", type=Path, default=None, help="Save figure to this path instead of showing it.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"Loading GEOROC data from {args.georoc} ...")
    georoc_df = pd.read_csv(args.georoc)

    print(f"Sampling {args.n} compositions ...")
    element_moles_list = sample_compositions(georoc_df, args.n, rng)

    Ss = rng.uniform(1.75, 2.9, args.n)
    P_lits = rng.uniform(1.5, 9.0, args.n)

    print("Generating and plotting P-T paths ...")
    plot_paths(element_moles_list, Ss, P_lits, args.out)


if __name__ == "__main__":
    main()
