"""
CLI: extract a random (or sequential) row subset from a large CSV file.

Uses Python's mmap module to count rows without loading the file into RAM,
then reads only the selected rows via pandas.

Usage:
    python subset_csv.py input.csv output.csv --nrows 100000
    python subset_csv.py input.csv output.csv --nrows 50000 --seed 7 --sequential
"""

from __future__ import annotations

import argparse
import mmap
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _count_rows_mmap(path: Path) -> int:
    """Count data rows (excluding header) by scanning newlines via mmap."""
    with path.open("rb") as fh:
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            n = sum(1 for _ in iter(mm.readline, b""))
    return max(0, n - 1)  # subtract header


def _choose_row_indices(total: int, nrows: int, seed: int, sequential: bool) -> np.ndarray:
    if nrows >= total:
        print(f"  Requested {nrows:,} rows but only {total:,} available — returning all.")
        return np.arange(total)
    if sequential:
        return np.arange(nrows)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=nrows, replace=False))


def run(
    input_path: Path,
    output_path: Path,
    nrows: int,
    seed: int,
    sequential: bool,
    chunksize: int = 50_000,
) -> None:
    print(f"Counting rows in {input_path} ...")
    total = _count_rows_mmap(input_path)
    print(f"  {total:,} data rows found.")

    indices = _choose_row_indices(total, nrows, seed, sequential)
    mode = "first" if sequential else "random"
    print(f"  Selecting {len(indices):,} {mode} rows (seed={seed}) ...")

    # Build 1-based row-number keep set (row 0 = header, always kept by pandas default)
    keep_1based = set(indices + 1)

    # skiprows callable: skip data rows NOT in our keep set (row 0 = header → always False)
    def _skip(i: int) -> bool:
        return i != 0 and i not in keep_1based

    print("  Reading subset ...")
    df = pd.read_csv(input_path, skiprows=_skip)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df):,} rows → {output_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Extract a row subset from a large CSV. "
            "Row counting uses mmap (no full file load). "
            "Row reading uses pandas with a skip-list."
        )
    )
    p.add_argument("input", help="Path to source CSV file")
    p.add_argument("output", help="Destination path for output CSV")
    p.add_argument("--nrows", type=int, required=True, help="Number of output rows")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for random sampling (default: 42)")
    p.add_argument(
        "--sequential",
        action="store_true",
        help="Take the first --nrows rows instead of a random sample",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(
        input_path=Path(args.input),
        output_path=Path(args.output),
        nrows=args.nrows,
        seed=args.seed,
        sequential=args.sequential,
    )


if __name__ == "__main__":
    main()
