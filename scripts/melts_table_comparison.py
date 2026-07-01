"""Compare bulk composition and P-T-S coverage between two MELTStable tables.

For two MELTStable-format CSVs (as produced by HeFESTo_functions.py /
import_HeFESTo_components, or the MELTS import pipeline), plots:
  - Stacked histograms of each oxide bulk composition   ('(Bulk_comp)' columns)
  - Stacked histograms of each element bulk composition ('(Bulk_comp_elements)' columns)
  - Stacked histogram of entropy S ('S(J/g/K)(System_main)')
  - Stacked density scatter (hexbin) of P vs S and P vs T

Composition histograms are built from one row per *unique* bulk composition:
a MELTStable row repeats once per pressure step along a simulation's P-T
path, so without deduplication a composition sampled at more pressures would
dominate the composition distribution. The S histogram and P-S / P-T density
scatters use every row instead, since those describe condition coverage
rather than composition diversity.

Usage:
    python scripts/melts_table_comparison.py \\
        --table-a data/MELTStables/HeFESTo/JiChingSims.csv \\
        --table-b data/MELTStables/HeFESTo/Training062426_MarsProfiles.csv \\
        --name-a JiChingSims --name-b "Mars training" \\
        --output-dir plots/melts_comparison
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONDITION_COLS = {
    'P': 'P(GPa)(System_main)',
    'T': 'T(K)(System_main)',
    'S': 'S(J/g/K)(System_main)',
}


def _sanitize(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('_')


def _short_name(col: str) -> str:
    return col.split('(')[0]


def _oxide_cols(columns) -> List[str]:
    return [c for c in columns if '(Bulk_comp)' in c]


def _element_cols(columns) -> List[str]:
    return [c for c in columns if '(Bulk_comp_elements)' in c]


def load_table(path: Path) -> pd.DataFrame:
    """Load only the composition and condition columns of a MELTStable CSV."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    oxide_cols = _oxide_cols(header)
    element_cols = _element_cols(header)
    condition_cols = [c for c in CONDITION_COLS.values() if c in header]
    usecols = sorted(set(oxide_cols) | set(element_cols) | set(condition_cols))
    if not usecols:
        raise ValueError(
            f'{path} does not look like a MELTStable CSV '
            "(no '(Bulk_comp)' / '(Bulk_comp_elements)' / condition columns found)"
        )
    print(f'Reading {path} ({len(usecols)} of {len(header)} columns)...')
    df = pd.read_csv(path, usecols=usecols)
    for col in usecols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f'  {len(df):,} rows')
    return df


def unique_compositions(df: pd.DataFrame, comp_cols: List[str]) -> pd.DataFrame:
    """One row per unique bulk composition (rounded to avoid float noise)."""
    if not comp_cols:
        return df.iloc[0:0]
    rounded_key = df[comp_cols].round(8)
    return df.loc[rounded_key.drop_duplicates().index].reset_index(drop=True)


def _stacked_hist(
    data_lists: List[np.ndarray],
    names: List[str],
    title: str,
    xlabel: str,
    out_path: Path,
    n_bins: int = 60,
    colors: Optional[List[str]] = None,
) -> None:
    colors = colors or ['steelblue', 'darkorange']
    all_vals = np.concatenate([d[np.isfinite(d)] for d in data_lists if len(d) > 0])
    if len(all_vals) == 0:
        print(f'  Skipping {title!r}: no finite data in either table')
        return

    lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    if lo >= hi:
        lo, hi = lo - 0.5, hi + 0.5
    bins = np.linspace(lo, hi, n_bins + 1)

    fig, axes = plt.subplots(len(data_lists), 1, figsize=(7, 2.8 * len(data_lists)), sharex=True)
    if len(data_lists) == 1:
        axes = [axes]

    for ax, data, name, color in zip(axes, data_lists, names, colors):
        finite = data[np.isfinite(data)] if len(data) > 0 else np.array([])
        if len(finite) == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
        else:
            ax.hist(finite, bins=bins, color=color, alpha=0.85, density=True)
        ax.set_ylabel('Density')
        ax.set_title(f'{name}   (n={len(finite):,})', fontsize=9)

    axes[-1].set_xlabel(xlabel)
    fig.suptitle(title, fontsize=11, y=1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {out_path}')


def _stacked_density_scatter(
    x_lists: List[np.ndarray],
    y_lists: List[np.ndarray],
    names: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    gridsize: int = 60,
) -> None:
    fig, axes = plt.subplots(len(x_lists), 1, figsize=(7.5, 3.4 * len(x_lists)), sharex=False)
    if len(x_lists) == 1:
        axes = [axes]

    for ax, x, y, name in zip(axes, x_lists, y_lists, names):
        mask = np.isfinite(x) & np.isfinite(y)
        x_f, y_f = x[mask], y[mask]
        if len(x_f) == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(name, fontsize=9)
            continue
        hb = ax.hexbin(x_f, y_f, gridsize=gridsize, cmap='viridis', mincnt=1, bins='log')
        fig.colorbar(hb, ax=ax, label='log10(count)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{name}   (n={len(x_f):,})', fontsize=9)

    axes[-1].set_xlabel(xlabel)
    fig.suptitle(title, fontsize=11, y=1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {out_path}')


def run(
    table_a_path: Path,
    table_b_path: Path,
    name_a: str,
    name_b: str,
    output_dir: str,
    n_bins: int = 60,
    gridsize: int = 60,
) -> None:
    df_a = load_table(table_a_path)
    df_b = load_table(table_b_path)
    names = [name_a, name_b]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {out}')

    oxide_cols = sorted(set(_oxide_cols(df_a.columns)) & set(_oxide_cols(df_b.columns)))
    element_cols = sorted(set(_element_cols(df_a.columns)) & set(_element_cols(df_b.columns)))
    missing_oxide = (set(_oxide_cols(df_a.columns)) | set(_oxide_cols(df_b.columns))) - set(oxide_cols)
    missing_element = (set(_element_cols(df_a.columns)) | set(_element_cols(df_b.columns))) - set(element_cols)
    if missing_oxide:
        print(f'  Skipping oxide column(s) not present in both tables: {sorted(missing_oxide)}')
    if missing_element:
        print(f'  Skipping element column(s) not present in both tables: {sorted(missing_element)}')

    comp_cols = oxide_cols + element_cols
    uniq_a = unique_compositions(df_a, comp_cols)
    uniq_b = unique_compositions(df_b, comp_cols)
    print(f'{name_a}: {len(uniq_a):,} unique composition(s) across {len(df_a):,} rows')
    print(f'{name_b}: {len(uniq_b):,} unique composition(s) across {len(df_b):,} rows')

    # --- Oxide-space composition histograms (one row per unique composition) ---
    for col in oxide_cols:
        _stacked_hist(
            [uniq_a[col].to_numpy(dtype=np.float64), uniq_b[col].to_numpy(dtype=np.float64)],
            names,
            title=f'Bulk {_short_name(col)}  (wt%, oxide space)',
            xlabel=f'{_short_name(col)} (wt%)',
            out_path=out / f'oxide_{_sanitize(_short_name(col))}.png',
            n_bins=n_bins,
        )

    # --- Element-space composition histograms (one row per unique composition) ---
    for col in element_cols:
        _stacked_hist(
            [uniq_a[col].to_numpy(dtype=np.float64), uniq_b[col].to_numpy(dtype=np.float64)],
            names,
            title=f'Bulk {_short_name(col)}  (moles, element space)',
            xlabel=f'{_short_name(col)} (moles)',
            out_path=out / f'element_{_sanitize(_short_name(col))}.png',
            n_bins=n_bins,
        )

    # --- Entropy histogram (every P-T-S row) ---
    s_col = CONDITION_COLS['S']
    if s_col in df_a.columns and s_col in df_b.columns:
        _stacked_hist(
            [df_a[s_col].to_numpy(dtype=np.float64), df_b[s_col].to_numpy(dtype=np.float64)],
            names,
            title='Entropy S',
            xlabel='S (J/g/K)',
            out_path=out / 'condition_S.png',
            n_bins=n_bins,
        )
    else:
        print('  Skipping S histogram: S column missing from one or both tables')

    # --- P-S and P-T sampling density (every row) ---
    p_col, t_col = CONDITION_COLS['P'], CONDITION_COLS['T']
    if p_col in df_a.columns and p_col in df_b.columns and s_col in df_a.columns and s_col in df_b.columns:
        _stacked_density_scatter(
            [df_a[p_col].to_numpy(dtype=np.float64), df_b[p_col].to_numpy(dtype=np.float64)],
            [df_a[s_col].to_numpy(dtype=np.float64), df_b[s_col].to_numpy(dtype=np.float64)],
            names,
            title='P vs S sampling density',
            xlabel='P (GPa)', ylabel='S (J/g/K)',
            out_path=out / 'density_P_vs_S.png',
            gridsize=gridsize,
        )
    else:
        print('  Skipping P vs S density scatter: P or S column missing from one or both tables')

    if p_col in df_a.columns and p_col in df_b.columns and t_col in df_a.columns and t_col in df_b.columns:
        _stacked_density_scatter(
            [df_a[p_col].to_numpy(dtype=np.float64), df_b[p_col].to_numpy(dtype=np.float64)],
            [df_a[t_col].to_numpy(dtype=np.float64), df_b[t_col].to_numpy(dtype=np.float64)],
            names,
            title='P vs T sampling density',
            xlabel='P (GPa)', ylabel='T (K)',
            out_path=out / 'density_P_vs_T.png',
            gridsize=gridsize,
        )
    else:
        print('  Skipping P vs T density scatter: P or T column missing from one or both tables')

    print(f'\nDone. {len(list(out.glob("*.png")))} plots saved to {out}')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--table-a', required=True, help='Path to first MELTStable CSV')
    p.add_argument('--table-b', required=True, help='Path to second MELTStable CSV')
    p.add_argument('--name-a', default=None, help='Display name for table A (default: filename stem)')
    p.add_argument('--name-b', default=None, help='Display name for table B (default: filename stem)')
    p.add_argument('--output-dir', required=True, help='Directory to save plots into')
    p.add_argument('--bins', type=int, default=60, help='Number of histogram bins (default: 60)')
    p.add_argument('--gridsize', type=int, default=60, help='Hexbin grid size for density scatter (default: 60)')
    return p


def main() -> None:
    args = build_parser().parse_args()
    table_a_path = Path(args.table_a)
    table_b_path = Path(args.table_b)
    run(
        table_a_path=table_a_path,
        table_b_path=table_b_path,
        name_a=args.name_a or table_a_path.stem,
        name_b=args.name_b or table_b_path.stem,
        output_dir=args.output_dir,
        n_bins=args.bins,
        gridsize=args.gridsize,
    )


if __name__ == '__main__':
    main()
