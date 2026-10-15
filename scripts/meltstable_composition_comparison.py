"""Compare two or three MELTStable tables in bulk-composition space.

For each oxide ('(Bulk_comp)' columns) and element ('(Bulk_comp_elements)'
columns) present in every table, draws a stacked histogram: one row per
table, all rows sharing the same bins and x axis. All components of a space
are tiled into a single grid figure:
  - composition_oxides.png    (wt%)
  - composition_elements.png  (moles)

Pass --per-component to also write one PNG per oxide/element, and
--xmax-percentile (e.g. 99.5) to trim long tails that squash the x axis.

Tables are loaded with the same loaders as melts_table_comparison.py, so
each may be a standalone CSV or a BigMetaTable .npy/.csv pair.

Histograms are built from one row per *unique* bulk composition (a MELTStable
row repeats once per P-T step along a simulation path); pass --all-rows to
histogram every row instead. Components that are identically zero in every
table (e.g. H2O in anhydrous sets) are skipped.

Usage:
    python scripts/meltstable_composition_comparison.py \\
        data/MELTStables/102/MELTS102_ValidsetOctober2closedCrFxAndBatch.npy \\
        data/MELTStables/102/MELTS102_ValidsetOctober2closedNoCrFxAndBatch.npy \\
        --names "Cr" "No Cr"

    python scripts/meltstable_composition_comparison.py A.csv B.csv C.csv \\
        --names Train Valid Test --ncols 5 --output-dir plots/comp_cmp
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plot_output import plots_dir
from melts_table_comparison import (
    _element_cols,
    _oxide_cols,
    _sanitize,
    _short_name,
    load_table,
    unique_compositions,
)

TABLE_COLORS = ['steelblue', 'darkorange', 'seagreen']
NON_COMPOSITION = {'mass'}   # bookkeeping columns that carry the '(Bulk_comp)' tag


def _shared_bins(arrays: List[np.ndarray], n_bins: int, percentile: float = 100.0) -> np.ndarray:
    """Common bin edges for all tables, spanning [min, `percentile`] of the pooled data."""
    finite = np.concatenate([a[np.isfinite(a)] for a in arrays])
    lo = float(finite.min())
    hi = float(finite.max()) if percentile >= 100 else float(np.percentile(finite, percentile))
    if lo >= hi:
        lo, hi = lo - 0.5, hi + 0.5
    return np.linspace(lo, hi, n_bins + 1)


def _draw_stack(axes, arrays: List[np.ndarray], bins: np.ndarray, colors: List[str]) -> None:
    """Draw one histogram per table on vertically stacked, x-shared axes."""
    for ax, data, color in zip(axes, arrays, colors):
        finite = data[np.isfinite(data)]
        if len(finite) == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', fontsize=7)
        else:
            # Values beyond a percentile-capped range are dropped, not piled into the last bin
            ax.hist(finite, bins=bins, color=color, alpha=0.85, density=True)
        ax.set_yticks([])
        ax.tick_params(axis='x', labelsize=7)
    axes[0].set_xlim(bins[0], bins[-1])


def _legend_handles(names: List[str], counts: List[int], colors: List[str]):
    return [Patch(color=c, label=f'{n}  (n={k:,})') for n, k, c in zip(names, counts, colors)]


def plot_grid(
    data: Dict[str, List[np.ndarray]],
    names: List[str],
    counts: List[int],
    title: str,
    unit: str,
    out_path: Path,
    n_bins: int,
    ncols: int,
    percentile: float = 100.0,
) -> None:
    """Tile every component's stacked histogram into one figure."""
    colors = TABLE_COLORS[:len(names)]
    n_comp = len(data)
    ncols = min(ncols, n_comp)
    nrows = math.ceil(n_comp / ncols)
    n_tab = len(names)

    fig = plt.figure(figsize=(3.2 * ncols, (0.75 * n_tab + 0.6) * nrows + 0.8), layout='constrained')
    subfigs = np.atleast_1d(fig.subfigures(nrows, ncols, wspace=0.03, hspace=0.05)).ravel()

    for sf, (label, arrays) in zip(subfigs, data.items()):
        axes = np.atleast_1d(sf.subplots(n_tab, 1, sharex=True))
        _draw_stack(axes, arrays, _shared_bins(arrays, n_bins, percentile), colors)
        sf.suptitle(f'{label} ({unit})', fontsize=9)
    for sf in subfigs[n_comp:]:
        sf.set_visible(False)

    fig.suptitle(title, fontsize=12)
    fig.legend(
        handles=_legend_handles(names, counts, colors),
        loc='outside lower center', ncols=n_tab, fontsize=9, frameon=False,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Saved -> {out_path}')


def plot_single(
    label: str,
    arrays: List[np.ndarray],
    names: List[str],
    unit: str,
    out_path: Path,
    n_bins: int,
    percentile: float = 100.0,
) -> None:
    colors = TABLE_COLORS[:len(names)]
    fig, axes = plt.subplots(len(names), 1, figsize=(6, 1.4 * len(names) + 0.8), sharex=True,
                             layout='constrained')
    _draw_stack(axes, arrays, _shared_bins(arrays, n_bins, percentile), colors)
    for ax, name in zip(axes, names):
        ax.set_ylabel(name, fontsize=8, rotation=0, ha='right', va='center')
    axes[-1].set_xlabel(f'{label} ({unit})')
    fig.suptitle(f'Bulk {label}', fontsize=11)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _common_cols(dfs: List[pd.DataFrame], selector) -> List[str]:
    per_table = [set(selector(df.columns)) for df in dfs]
    common = set.intersection(*per_table)
    skipped = set.union(*per_table) - common
    if skipped:
        print(f'  Skipping column(s) not present in every table: {sorted(skipped)}')
    return [c for c in selector(dfs[0].columns) if c in common and _short_name(c) not in NON_COMPOSITION]


def _component_arrays(tables: List[pd.DataFrame], cols: List[str]) -> Dict[str, List[np.ndarray]]:
    out = {}
    for col in cols:
        arrays = [t[col].to_numpy(dtype=np.float64) for t in tables]
        if all(np.nan_to_num(a).max(initial=0.0) == 0.0 and np.nan_to_num(a).min(initial=0.0) == 0.0
               for a in arrays):
            print(f'  Skipping {_short_name(col)}: zero in every table')
            continue
        out[_short_name(col)] = arrays
    return out


def run(
    table_paths: List[Path],
    names: List[str],
    output_dir: Path,
    n_bins: int = 60,
    ncols: int = 4,
    all_rows: bool = False,
    per_component: bool = False,
    percentile: float = 100.0,
) -> None:
    dfs = [load_table(p) for p in table_paths]
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {output_dir}')

    oxide_cols = _common_cols(dfs, _oxide_cols)
    element_cols = _common_cols(dfs, _element_cols)
    comp_cols = oxide_cols + element_cols
    if not comp_cols:
        raise ValueError('No composition columns are shared by all tables')

    if all_rows:
        tables = dfs
    else:
        tables = [unique_compositions(df, comp_cols) for df in dfs]
    for name, df, t in zip(names, dfs, tables):
        print(f'{name}: {len(t):,} {"rows" if all_rows else "unique composition(s)"} '
              f'(from {len(df):,} rows)')
    counts = [len(t) for t in tables]
    what = 'all rows' if all_rows else 'unique compositions'

    for cols, space, unit in ((oxide_cols, 'oxides', 'wt%'), (element_cols, 'elements', 'moles')):
        data = _component_arrays(tables, cols)
        if not data:
            print(f'  No {space} columns to plot')
            continue
        plot_grid(
            data, names, counts,
            title=f'Bulk composition ({space}, {what}) — normalized density',
            unit=unit,
            out_path=output_dir / f'composition_{space}.png',
            n_bins=n_bins, ncols=ncols, percentile=percentile,
        )
        if per_component:
            sub = output_dir / space
            sub.mkdir(exist_ok=True)
            for label, arrays in data.items():
                plot_single(label, arrays, names, unit, sub / f'{_sanitize(label)}.png', n_bins, percentile)
            print(f'  Saved {len(data)} per-component plots -> {sub}')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('tables', nargs='+',
                   help='2 or 3 MELTStable tables: CSVs, or .npy files / base names paired with a header-only .csv')
    p.add_argument('--names', nargs='+', default=None, help='Display names, one per table (default: filename stems)')
    p.add_argument('--output-dir', default=None,
                   help='Directory to save plots into (default: <repo>/plots/meltstable_composition_comparison)')
    p.add_argument('--bins', type=int, default=60, help='Number of histogram bins (default: 60)')
    p.add_argument('--ncols', type=int, default=4, help='Components per row in the grid figure (default: 4)')
    p.add_argument('--xmax-percentile', type=float, default=100.0,
                   help='Cap each x axis at this percentile of the pooled data, e.g. 99.5 to '
                        'trim long tails (default: 100, full range)')
    p.add_argument('--all-rows', action='store_true',
                   help='Histogram every row instead of one row per unique composition')
    p.add_argument('--per-component', action='store_true',
                   help='Also save one stacked-histogram PNG per oxide/element')
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not 2 <= len(args.tables) <= 3:
        parser.error(f'expected 2 or 3 tables, got {len(args.tables)}')
    paths = [Path(t) for t in args.tables]
    names = args.names or [p.stem for p in paths]
    if len(names) != len(paths):
        parser.error(f'--names got {len(names)} name(s) for {len(paths)} table(s)')
    run(
        table_paths=paths,
        names=names,
        output_dir=Path(args.output_dir) if args.output_dir else plots_dir('meltstable_composition_comparison'),
        n_bins=args.bins,
        ncols=args.ncols,
        all_rows=args.all_rows,
        per_component=args.per_component,
        percentile=args.xmax_percentile,
    )


if __name__ == '__main__':
    main()
