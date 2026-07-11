"""
CLI: extract a random (or sequential) row subset from a large CSV/TXT file.

Uses Python's mmap module to count rows without loading the file into RAM,
then reads only the selected rows via pandas.

Input and output are given as extensionless base paths. Whichever of
<base>.csv / <base>.txt exist for the input are subsetted; the same row
indices are used for both so paired files stay row-aligned. Each existing
input produces a matching output at <output_base>.<ext>.

.csv files are expected to carry a header row (inherited into the output,
not counted as data). .txt files are expected to be headerless — one data
line per row, as written by BigMetaTable's metadata sidecar files.

Usage:
    python subset_csv.py path/to/input path/to/output --nrows 100000
    python subset_csv.py path/to/input path/to/output --nrows 50000 --seed 7 --sequential
"""

from __future__ import annotations

import argparse
import mmap
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXTENSIONS = (".csv", ".txt")

# Whether each extension carries a header row. .csv is written with a column
# header; .txt is BigMetaTable's headerless metadata sidecar (one line per row).
HAS_HEADER = {".csv": True, ".txt": False}


def _with_ext(base: Path, ext: str) -> Path:
    return base.parent / (base.name + ext)


def _find_existing(base: Path) -> dict[str, Path]:
    found = {ext: p for ext in EXTENSIONS if (p := _with_ext(base, ext)).exists()}
    if not found:
        raise FileNotFoundError(
            f"No {' or '.join(_with_ext(base, e).name for e in EXTENSIONS)} found for base '{base}'"
        )
    return found


def _count_rows_mmap(path: Path, has_header: bool) -> int:
    """Count data rows by scanning newlines via mmap, excluding the header line if present."""
    with path.open("rb") as fh:
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            n = sum(1 for _ in iter(mm.readline, b""))
    return max(0, n - 1) if has_header else n


def _choose_row_indices(total: int, nrows: int, seed: int, sequential: bool) -> np.ndarray:
    if nrows >= total:
        print(f"  Requested {nrows:,} rows but only {total:,} available — returning all.")
        return np.arange(total)
    if sequential:
        return np.arange(nrows)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=nrows, replace=False))


def _subset_one(input_path: Path, output_path: Path, indices: np.ndarray, has_header: bool) -> int:
    if has_header:
        # Row 0 is the header line and is always kept; data row j lives at file line j+1.
        keep = set(indices + 1)
        skip = lambda i: i != 0 and i not in keep
        header = 0
    else:
        # No header line; data row j lives at file line j.
        keep = set(indices)
        skip = lambda i: i not in keep
        header = None

    df = pd.read_csv(input_path, skiprows=skip, header=header)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, header=has_header)
    return len(df)


def run(
    input_base: Path,
    output_base: Path,
    nrows: int,
    seed: int,
    sequential: bool,
    chunksize: int = 50_000,
) -> None:
    inputs = _find_existing(input_base)

    print(f"Counting rows for {input_base} ({', '.join(p.name for p in inputs.values())}) ...")
    counts = {ext: _count_rows_mmap(p, HAS_HEADER[ext]) for ext, p in inputs.items()}
    distinct_counts = set(counts.values())
    if len(distinct_counts) > 1:
        detail = ", ".join(f"{ext}={n:,}" for ext, n in counts.items())
        raise ValueError(
            f"Row counts differ between paired files ({detail}); cannot apply the same "
            "row indices to both. Reconcile the files or subset them separately."
        )
    total = next(iter(distinct_counts))
    print(f"  {total:,} data rows found.")

    indices = _choose_row_indices(total, nrows, seed, sequential)
    mode = "first" if sequential else "random"
    print(f"  Selecting {len(indices):,} {mode} rows (seed={seed}) ...")

    for ext, input_path in inputs.items():
        output_path = _with_ext(output_base, ext)
        print(f"  Reading subset from {input_path} ...")
        n = _subset_one(input_path, output_path, indices, HAS_HEADER[ext])
        print(f"  Saved {n:,} rows → {output_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Extract a row subset from a large CSV/TXT. Input/output are extensionless "
            "base paths; whichever of <base>.csv / <base>.txt exist are subsetted using "
            "the same row indices. Row counting uses mmap (no full file load). "
            "Row reading uses pandas with a skip-list."
        )
    )
    p.add_argument("input", help="Extensionless base path to source file(s) (.csv and/or .txt)")
    p.add_argument("output", help="Extensionless base path for output file(s)")
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
        input_base=Path(args.input),
        output_base=Path(args.output),
        nrows=args.nrows,
        seed=args.seed,
        sequential=args.sequential,
    )


if __name__ == "__main__":
    main()
