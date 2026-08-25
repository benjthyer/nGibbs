"""
Verify that dndT/dndP derivative sidecar tables are row-parallel with their main
MELTS/HeFESTo datatable.

Sidecars ride alongside a main table as `<root>_dndT.{csv,npy}` and
`<root>_dndP.{csv,npy}` (see `src/builder/processing/sidecar.py`, SIDECAR_SPECS).
A `.npy` table (main or sidecar) may have a same-stem `.csv` that holds only
the header row of column names and no data (see `BigMetaTable`); `.npy` is
always checked first so that header-only file is never mistaken for the data.
The importer's guarantee is one sidecar row per main-table row, in the same
order; nothing downstream re-checks that once the files are on disk, so a
truncated write (killed job, disk full, partial re-run) can silently desync
them. This script re-derives the row counts independently and reports any
mismatch.

Row counts come from the file itself, not from loading it: `.npy` files are
opened with `mmap_mode='r'` (reads only the header), and `.csv` files are
counted by streaming raw bytes and counting newlines, so even the largest
tables (multi-GB) check in seconds without materialising any data.

Usage:
    python scripts/verify_dndt_dndp_row_counts.py data/MELTStables/HeFESTo/Training081026_EarthAdiabatsOnly
    python scripts/verify_dndt_dndp_row_counts.py data/MELTStables/HeFESTo
        (a directory scans every main table found directly inside it)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

SIDECAR_SUFFIXES = (('dndt', '_dndT'), ('dndp', '_dndP'))
# main-table candidates in a directory scan exclude anything that is itself a
# sidecar, a manifest, or the internal working copy sidecar.py writes.
NON_MAIN_SUFFIXES = ('_dndT', '_dndP', '_deriv_manifest', '_working')


def count_csv_rows(path: Path) -> int:
    """Data-row count of a CSV: total lines minus the header, streamed in binary."""
    newlines = 0
    last_byte = b''
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            newlines += chunk.count(b'\n')
            last_byte = chunk[-1:]
    if last_byte and last_byte != b'\n':
        newlines += 1  # trailing data line with no final newline
    return max(newlines - 1, 0)


def count_npy_rows(path: Path) -> int:
    import numpy as np
    return int(np.load(path, mmap_mode='r').shape[0])


def row_count(path: Path) -> int:
    return count_npy_rows(path) if path.suffix == '.npy' else count_csv_rows(path)


def find_table(root: str, suffix: str = '') -> Optional[Path]:
    """Prefer `<root><suffix>.npy`, fall back to `<root><suffix>.csv`.

    A `.npy` main table is often paired with a same-stem `.csv` that holds only
    the header row (column names, zero data rows) -- see `BigMetaTable` /
    `split_by_composition.py`. Checking `.npy` first avoids mistaking that
    header-only file for the data and reporting a bogus 0-row count.
    """
    for ext in ('.npy', '.csv'):
        candidate = Path(f'{root}{suffix}{ext}')
        if candidate.exists():
            return candidate
    return None


def discover_roots(directory: Path) -> list[str]:
    """Main-table roots directly inside `directory` (its own .csv/.npy files, minus sidecars).

    A root counts once even when it has both a `.npy` and a same-stem
    header-only `.csv` (see `find_table`).
    """
    roots = set()
    for pattern in ('*.csv', '*.npy'):
        for path in directory.glob(pattern):
            if any(path.stem.endswith(suffix) for suffix in NON_MAIN_SUFFIXES):
                continue
            roots.add(str(path.with_suffix('')))
    return sorted(roots)


def verify_root(root: str) -> bool:
    main = find_table(root)
    print(root)
    if main is None:
        print('  [MISSING] no main table (.csv or .npy) found')
        return False

    main_rows = row_count(main)
    print(f'  main  {main.name:<55} {main_rows:>14,} rows')

    ok = True
    for attr, suffix in SIDECAR_SUFFIXES:
        sidecar = find_table(root, suffix)
        if sidecar is None:
            print(f'  {attr:<5} MISSING  ({Path(root).name}{suffix}.npy / .csv)')
            ok = False
            continue
        sidecar_rows = row_count(sidecar)
        matched = sidecar_rows == main_rows
        ok &= matched
        status = 'OK' if matched else f'MISMATCH (expected {main_rows:,})'
        print(f'  {attr:<5} {sidecar.name:<55} {sidecar_rows:>14,} rows  [{status}]')
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root',
                         help='Root path of a main table (without extension), or a directory '
                              'to scan every main table found directly inside it.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = Path(args.root)

    if target.is_dir():
        roots = discover_roots(target)
        if not roots:
            print(f'No main tables (*.csv / *.npy) found directly under {target}')
            return 1
    else:
        roots = [args.root]

    all_ok = True
    for i, root in enumerate(roots):
        if i:
            print()
        if not verify_root(root):
            all_ok = False

    print()
    print('All tables row-aligned.' if all_ok else 'Row-count mismatch detected -- see above.')
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
