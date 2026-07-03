"""Compare bulk composition between a MELTStable reference table and a
directory of HeFESTo control files.

Like ``melts_table_comparison.py``, but instead of two MELTStable CSVs this
compares:
  - A reference MELTStable-format CSV (as produced by HeFESTo_functions.py /
    import_HeFESTo_components, or the MELTS import pipeline).
  - A directory tree of ``BatchNNNN/SimulationN`` (or flat ``SimulationN`` /
    ``model_NNNNNN``) subdirectories, each holding a HeFESTo ``control`` file
    that defines a starting bulk composition.

Produces the same stacked composition histograms as ``melts_table_comparison.py``:
  - Stacked histograms of each oxide bulk composition   ('(Bulk_comp)' columns)
  - Stacked histograms of each element bulk composition ('(Bulk_comp_elements)' columns)

As in ``melts_table_comparison.py``, composition histograms are built from
one row per *unique* bulk composition. There is no P-T-S coverage comparison
here: control files only specify a starting composition, not a simulation
P-T path, so there is nothing on the control-file side to compare against
the reference table's condition columns.

Usage:
    python scripts/melts_table_control_dir_comparison.py \\
        --reference-table data/MELTStables/HeFESTo/Training062426_MarsProfiles.csv \\
        --control-dir /path/to/workspace \\
        --name-reference "Mars training" --name-controls "New batch" \\
        --output-dir plots/melts_control_dir_comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
SRC_DIR = REPO_ROOT / 'src'
for p in (str(SCRIPTS_DIR), str(REPO_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from melts_table_comparison import (  # noqa: E402
    load_table,
    unique_compositions,
    _oxide_cols,
    _element_cols,
    _sanitize,
    _short_name,
    _stacked_hist,
)
from ngibbs.utils.file_utils import _parse_control_file  # noqa: E402
from builder.HeFESTo.HeFESTo_functions import (  # noqa: E402
    _compute_bulk_from_elements,
    _find_all_control_files,
)


def load_control_dir(control_dir: Path) -> pd.DataFrame:
    """Parse bulk compositions from every 'control' file under control_dir.

    Returns a DataFrame with oxide columns suffixed '(Bulk_comp)' and
    element-mole columns suffixed '(Bulk_comp_elements)', matching
    MELTStable column naming so it can be compared directly against a
    MELTStable table with the shared helpers from melts_table_comparison.py.
    """
    control_files = _find_all_control_files(str(control_dir))
    if not control_files:
        raise FileNotFoundError(f'No control files found under: {control_dir}')
    print(f'Found {len(control_files)} control file(s) under {control_dir}')

    records: List[Dict[str, float]] = []
    n_failed = 0
    for ctrl_path in control_files:
        try:
            element_moles, _ = _parse_control_file(ctrl_path)
            bulk_wt, bulk_elements, _ = _compute_bulk_from_elements(element_moles)
            record = {f'{ox}(Bulk_comp)': v for ox, v in bulk_wt.items()}
            record.update({f'{el}(Bulk_comp_elements)': v for el, v in bulk_elements.items()})
            records.append(record)
        except Exception:
            n_failed += 1

    if not records:
        raise ValueError(f'No valid compositions could be parsed from control files under {control_dir}')
    if n_failed:
        print(f'  Warning: {n_failed} control file(s) could not be parsed and were skipped.')

    df = pd.DataFrame(records).fillna(0.0)
    print(f'  {len(df):,} composition(s) parsed')
    return df


def run(
    table_path: Path,
    control_dir: Path,
    name_table: str,
    name_dir: str,
    output_dir: str,
    n_bins: int = 60,
) -> None:
    df_a = load_table(table_path)
    df_b = load_control_dir(control_dir)
    names = [name_table, name_dir]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {out}')

    oxide_cols = sorted(set(_oxide_cols(df_a.columns)) & set(_oxide_cols(df_b.columns)))
    element_cols = sorted(set(_element_cols(df_a.columns)) & set(_element_cols(df_b.columns)))
    missing_oxide = (set(_oxide_cols(df_a.columns)) | set(_oxide_cols(df_b.columns))) - set(oxide_cols)
    missing_element = (set(_element_cols(df_a.columns)) | set(_element_cols(df_b.columns))) - set(element_cols)
    if missing_oxide:
        print(f'  Skipping oxide column(s) not present in both sources: {sorted(missing_oxide)}')
    if missing_element:
        print(f'  Skipping element column(s) not present in both sources: {sorted(missing_element)}')

    comp_cols = oxide_cols + element_cols
    uniq_a = unique_compositions(df_a, comp_cols)
    uniq_b = unique_compositions(df_b, comp_cols)
    print(f'{name_table}: {len(uniq_a):,} unique composition(s) across {len(df_a):,} rows')
    print(f'{name_dir}: {len(uniq_b):,} unique composition(s) across {len(df_b):,} control file(s)')

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

    print(f'\nDone. {len(list(out.glob("*.png")))} plots saved to {out}')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--reference-table', required=True, help='Path to reference MELTStable CSV')
    p.add_argument('--control-dir', required=True, help='Root directory containing BatchNNNN/SimulationN or flat SimulationN subdirectories with control files')
    p.add_argument('--name-reference', default=None, help='Display name for the reference table (default: filename stem)')
    p.add_argument('--name-controls', default=None, help='Display name for the control-file directory (default: directory name)')
    p.add_argument('--output-dir', required=True, help='Directory to save plots into')
    p.add_argument('--bins', type=int, default=60, help='Number of histogram bins (default: 60)')
    return p


def main() -> None:
    args = build_parser().parse_args()
    table_path = Path(args.reference_table)
    control_dir = Path(args.control_dir)
    run(
        table_path=table_path,
        control_dir=control_dir,
        name_table=args.name_reference or table_path.stem,
        name_dir=args.name_controls or control_dir.name,
        output_dir=args.output_dir,
        n_bins=args.bins,
    )


if __name__ == '__main__':
    main()
