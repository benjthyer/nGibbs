#!/usr/bin/env python3
"""Realign derivative sidecars against a main table, without re-importing.

Why this exists
---------------
`import_HeFESTo_components` and `import_HeFESTo_derivatives` are two independent passes
over the same tree, each with its own hand-copied copy of the admission rules ("does this
simulation have control/fort.56/61/68/99, and how many rows does it contribute"). Any
divergence between those copies drops a simulation from one table and not the other.

That is why the sidecars carry `P(GPa)` and `T(K)` as their first two columns: the import
notes say they are there "so the join can be checked rather than assumed". Nothing was
actually checking them. This script does.

Two failure modes, and only one of them is loud
-----------------------------------------------
* Different ROW COUNTS -- `sidecar.attach` raises, which is what you saw.
* Same row count, different ROWS -- e.g. two simulations dropped from the main table and
  two *different* ones dropped from the sidecar. Lengths match, `attach` is happy, and
  every row's derivative belongs to some other row. Nothing downstream would notice; the
  training loss would simply be wrong. This script detects that case too, and it is the
  reason to run it even when the lengths agree.

What it does
------------
Aligns the sidecar's (P, T) sequence into the main table's as a subsequence, then writes a
corrected sidecar with NaN rows inserted wherever the main table has a row the sidecar
lacks. NaN, not zero: the trainer masks on `isfinite`, and a zero would read as a measured
"this component does not change".

Rows the SIDECAR has and the main table does not are dropped, and reported -- that
direction means the derivative pass admitted a simulation the main pass rejected.

Usage
-----
    python repair_derivative_alignment.py MAIN.csv [--suffixes _dndP _dndT]
                                          [--tol 1e-6] [--dry-run] [--out-suffix _aligned]

Reads `MAIN.csv` and `MAIN_dndP.csv` / `MAIN_dndT.csv`, writes `MAIN_dndP_aligned.csv`
unless `--out-suffix ''` is given, in which case it overwrites in place (after a .bak).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd

P_COL = 'P(GPa)(System_main)'
T_COL = 'T(K)(System_main)'
CHUNK = 500_000


def _read_pt(path, p_col=P_COL, t_col=T_COL):
    """(P, T) for every row of a CSV, read in chunks so a 10M-row table stays in RAM."""
    ps, ts = [], []
    for chunk in pd.read_csv(path, usecols=[p_col, t_col], chunksize=CHUNK):
        ps.append(chunk[p_col].to_numpy(np.float64))
        ts.append(chunk[t_col].to_numpy(np.float64))
    return np.concatenate(ps), np.concatenate(ts)


def align_subsequence(main_p, main_t, side_p, side_t, tol=1e-6):
    """Greedy subsequence match of the sidecar's rows into the main table's.

    Both tables are written in simulation order and, within a simulation, in row order --
    the only way they diverge is that one skipped whole simulations. So a single forward
    scan is sufficient and O(n): advance through the main table looking for the sidecar's
    next (P, T); if it is not found before the main table ends, that sidecar row has no
    home and is surplus.

    Returns `take` (for each main row, the sidecar row index or -1) and `surplus`
    (sidecar row indices that matched nothing).
    """
    n_main, n_side = len(main_p), len(side_p)
    take = np.full(n_main, -1, dtype=np.int64)
    surplus = []
    j = 0
    for i in range(n_main):
        if j >= n_side:
            break
        if abs(side_p[j] - main_p[i]) <= tol and abs(side_t[j] - main_t[i]) <= tol:
            take[i] = j
            j += 1
    # Anything left in the sidecar never matched, in order.
    if j < n_side:
        surplus = list(range(j, n_side))
    return take, surplus


def repair(main_csv, suffix, tol, out_suffix, dry_run):
    base = main_csv[:-4] if main_csv.lower().endswith('.csv') else main_csv
    side_csv = f'{base}{suffix}.csv'
    if not os.path.exists(side_csv):
        print(f'  {suffix}: no such file, skipped ({side_csv})')
        return None

    main_p, main_t = _read_pt(main_csv)
    side_p, side_t = _read_pt(side_csv)
    print(f'\n  {suffix}: main {len(main_p):,} rows, sidecar {len(side_p):,} rows '
          f'({len(main_p) - len(side_p):+,})')

    take, surplus = align_subsequence(main_p, main_t, side_p, side_t, tol)
    matched = int((take >= 0).sum())
    missing = len(main_p) - matched
    print(f'     matched {matched:,}  |  main rows with no derivative {missing:,}  |  '
          f'surplus sidecar rows {len(surplus):,}')

    if matched == len(main_p) and not surplus and len(main_p) == len(side_p):
        print('     already aligned; nothing to do')
        return 0

    if matched == 0 and len(side_p):
        print('     ALIGNMENT FAILED: not a single (P, T) pair matched. The two tables are '
              'not the same tree, or the P/T columns differ. Not writing anything.')
        return -1

    # Where the gaps fall tells you whether whole simulations went missing (one long run)
    # or rows are scattered (something worse).
    gaps = np.where(take < 0)[0]
    if gaps.size:
        breaks = np.where(np.diff(gaps) > 1)[0]
        runs = len(breaks) + 1
        print(f'     the {missing:,} missing rows fall in {runs} contiguous run(s) -- '
              f'{"consistent with whole simulations being skipped" if runs < 200 else "scattered, which is NOT a whole-simulation skip"}')
        first = gaps[0]
        print(f'     first gap at main row {first:,}  (P={main_p[first]:.4f}, T={main_t[first]:.2f})')

    if dry_run:
        print('     --dry-run: no file written')
        return missing

    header = list(pd.read_csv(side_csv, nrows=0).columns)
    out_csv = f'{base}{suffix}{out_suffix}.csv' if out_suffix else side_csv
    tmp = out_csv + '.tmp'

    side = pd.read_csv(side_csv).to_numpy(np.float64)     # (n_side, ncols)
    aligned = np.full((len(main_p), side.shape[1]), np.nan, dtype=np.float64)
    have = take >= 0
    aligned[have] = side[take[have]]

    pd.DataFrame(aligned, columns=header).to_csv(tmp, index=False)
    if not out_suffix:
        shutil.copyfile(side_csv, side_csv + '.bak')
        print(f'     original preserved at {os.path.basename(side_csv)}.bak')
    os.replace(tmp, out_csv)
    # A stale .npy would be picked up in preference to the repaired CSV.
    stale = f'{base}{suffix}.npy'
    if os.path.exists(stale) and not out_suffix:
        os.remove(stale)
        print(f'     removed stale {os.path.basename(stale)} so the CSV is rebuilt')
    print(f'     wrote {os.path.basename(out_csv)}: {len(main_p):,} rows, '
          f'{missing:,} NaN-filled')
    return missing


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('main_csv')
    ap.add_argument('--suffixes', nargs='*', default=['_dndP', '_dndT'])
    ap.add_argument('--tol', type=float, default=1e-6,
                    help='(P, T) match tolerance; both tables print the same precision')
    ap.add_argument('--out-suffix', default='_aligned',
                    help="written alongside the original; pass '' to overwrite in place")
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    print(f'main table: {a.main_csv}')
    codes = [repair(a.main_csv, s, a.tol, a.out_suffix, a.dry_run) for s in a.suffixes]
    bad = [c for c in codes if c == -1]
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
