"""
Add a binary indicator column to a CSV or Parquet table and resave it.

The new column is 0 where the source column is 0, and 1 where it is nonzero.
Processing is done in chunks so large (multi-GB) tables fit in memory.

Usage:
    python add_binary_column.py --file <path> --after-column <col> --new-column <col>
                                [--chunk-size <rows>]

Example:
    python add_binary_column.py \\
        --file ./data/my_table.csv \\
        --after-column mass_olivine \\
        --new-column binary_olivine
"""

import argparse
import sys
import shutil
import tempfile
from pathlib import Path

import pandas as pd


DEFAULT_CHUNK_SIZE = 200_000


def _peek_columns_csv(file_path: Path) -> list[str]:
    return list(pd.read_csv(file_path, nrows=0).columns)


def _peek_columns_parquet(file_path: Path) -> list[str]:
    import pyarrow.parquet as pq
    return pq.read_schema(file_path).names


def _process_csv(file_path: Path, insert_pos: int, after_col: str, new_col: str, chunk_size: int):
    tmp_path = file_path.with_suffix(".tmp.csv")
    rows_written = 0
    try:
        first = True
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunk.insert(insert_pos, new_col, (chunk[after_col] != 0).astype(int))
            chunk.to_csv(tmp_path, mode="a", header=first, index=False)
            first = False
            rows_written += len(chunk)
            print(f"\r  {rows_written:,} rows written...", end="", flush=True)
        print()
        shutil.move(tmp_path, file_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _process_parquet(file_path: Path, insert_pos: int, after_col: str, new_col: str, chunk_size: int):
    import pyarrow as pa
    import pyarrow.parquet as pq

    tmp_path = file_path.with_suffix(".tmp.parquet")
    rows_written = 0
    writer = None
    try:
        pf = pq.ParquetFile(file_path)
        for batch in pf.iter_batches(batch_size=chunk_size):
            chunk = batch.to_pandas()
            chunk.insert(insert_pos, new_col, (chunk[after_col] != 0).astype(int))
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema)
            writer.write_table(table)
            rows_written += len(chunk)
            print(f"\r  {rows_written:,} rows written...", end="", flush=True)
        print()
        if writer is not None:
            writer.close()
            writer = None
        shutil.move(tmp_path, file_path)
    except Exception:
        if writer is not None:
            writer.close()
        tmp_path.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Add a binary indicator column after a specified column and resave."
    )
    parser.add_argument("--file", required=True, type=str,
                        help="Path to the input table (.csv or .parquet)")
    parser.add_argument("--after-column", required=True, type=str,
                        help="Name of the column the new binary column should follow")
    parser.add_argument("--new-column", required=True, type=str,
                        help="Name to give the new binary column")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Rows per chunk (default: {DEFAULT_CHUNK_SIZE:,})")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        columns = _peek_columns_csv(file_path)
    elif suffix == ".parquet":
        columns = _peek_columns_parquet(file_path)
    else:
        print(f"ERROR: Unsupported file format '{suffix}'. Use .csv or .parquet.")
        sys.exit(1)

    if args.after_column not in columns:
        print(f"ERROR: Column '{args.after_column}' not found in table.")
        print(f"Available columns: {columns}")
        sys.exit(1)

    if args.new_column in columns:
        print(f"ERROR: Column '{args.new_column}' already exists in table.")
        sys.exit(1)

    print("Columns before:")
    print(columns)

    insert_pos = columns.index(args.after_column) + 1

    print(f"\nProcessing in chunks of {args.chunk_size:,} rows...")
    if suffix == ".csv":
        _process_csv(file_path, insert_pos, args.after_column, args.new_column, args.chunk_size)
    else:
        _process_parquet(file_path, insert_pos, args.after_column, args.new_column, args.chunk_size)

    if suffix == ".csv":
        new_columns = list(pd.read_csv(file_path, nrows=0).columns)
    else:
        new_columns = _peek_columns_parquet(file_path)

    print("\nColumns after:")
    print(new_columns)
    print(f"\nSaved: {file_path}")


if __name__ == "__main__":
    main()
