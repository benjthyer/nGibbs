#!/usr/bin/env python
"""Convert a .npy file to a human-readable .csv in the same directory.

Usage:
    python npy_to_csv.py path/to/array.npy [--fmt %.6g] [--max-rows N] [--allow-pickle]

The output is written next to the input as path/to/array.csv.

- 1D arrays become a single column.
- 2D arrays are written row by row, with a header row col_0, col_1, ...
- Arrays with more than 2 dimensions are flattened to 2D: the last axis
  becomes the columns, and the leading axes are written as index columns
  (dim_0, dim_1, ...) so every row can be traced back to its position.
- Structured (record) arrays use their field names as the header.

Large arrays are memory-mapped and written in chunks, so multi-GB files
do not need to fit in RAM.
"""

import argparse
import csv
import os
import sys

import numpy as np

CHUNK_ROWS = 100_000


def _format_value(value, fmt):
    if isinstance(value, (float, np.floating)):
        return fmt % value
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _load(path, allow_pickle):
    try:
        return np.load(path, mmap_mode="r", allow_pickle=allow_pickle)
    except ValueError:
        # Object arrays cannot be memory-mapped.
        return np.load(path, allow_pickle=allow_pickle)


def convert(npy_path, fmt="%.6g", max_rows=None, allow_pickle=False):
    arr = _load(npy_path, allow_pickle)
    if arr.ndim == 0:
        arr = arr.reshape(1)

    csv_path = os.path.splitext(npy_path)[0] + ".csv"

    if arr.dtype.names:
        header = list(arr.dtype.names)
        index_shape = arr.shape if arr.ndim > 1 else ()
        data = arr.reshape(-1)
    elif arr.ndim == 1:
        header = ["value"]
        index_shape = ()
        data = arr.reshape(-1, 1)
    else:
        n_cols = arr.shape[-1]
        index_shape = arr.shape[:-1] if arr.ndim > 2 else ()
        header = [f"col_{i}" for i in range(n_cols)]
        data = arr.reshape(-1, n_cols)

    index_header = [f"dim_{i}" for i in range(len(index_shape))]

    n_rows = data.shape[0] if max_rows is None else min(max_rows, data.shape[0])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(index_header + header)
        for start in range(0, n_rows, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, n_rows)
            chunk = np.asarray(data[start:stop])
            for offset, row in enumerate(chunk):
                prefix = (list(np.unravel_index(start + offset, index_shape))
                          if index_header else [])
                values = row.tolist() if arr.dtype.names else row
                writer.writerow([int(i) for i in prefix]
                                + [_format_value(v, fmt) for v in values])

    print(f"Wrote {n_rows} rows x {len(index_header) + len(header)} columns "
          f"(array shape {arr.shape}, dtype {arr.dtype}) to {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npy_path", help="Path to the .npy file")
    parser.add_argument("--fmt", default="%.6g",
                        help="printf-style format for floats (default: %%.6g)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Only write the first N rows (handy for previewing huge files)")
    parser.add_argument("--allow-pickle", action="store_true",
                        help="Allow loading object arrays (only for files you trust)")
    args = parser.parse_args()

    if not os.path.isfile(args.npy_path):
        sys.exit(f"File not found: {args.npy_path}")

    convert(args.npy_path, fmt=args.fmt, max_rows=args.max_rows,
            allow_pickle=args.allow_pickle)


if __name__ == "__main__":
    main()
