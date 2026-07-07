"""
Merge two BigMetaTable datasets from extensionless base names.

Each table may be backed by a .npy memmap, a .csv, or both - BigMetaTable already
loads the .npy directly when present (falling back to building one from .csv
otherwise), so no conversion step is needed here regardless of which
representation either input uses.

The merged output always inherits table A's columns (name, count, and order).
Table B is reindexed to match: its columns are reordered to line up with A's,
any column A has that B lacks is filled with 0, and any column B has that A
lacks is an error (B cannot introduce columns unknown to A).

Usage:
    python scripts/merge_bigmetatables.py --table-a <base_a> --table-b <base_b> --output <merged_base>

Examples:
    python scripts/merge_bigmetatables.py \
        --table-a data/MELTStables/110/trainset \
        --table-b data/MELTStables/110/validset \
        --output data/MELTStables/110/train_valid_merged
"""

import argparse
import csv
import gc
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


repo_src = str(Path(__file__).resolve().parents[1] / "src")
if repo_src not in sys.path:
    sys.path.insert(0, repo_src)

from builder.processing.BigMetaTable import BigMetaTable


def _validate_extensionless(value: str, arg_name: str) -> str:
    path = Path(value)
    if path.suffix:
        raise ValueError(
            f"{arg_name} must be extensionless (received '{value}')."
        )
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge two BigMetaTable datasets from extensionless base names."
    )
    parser.add_argument(
        "--table-a",
        required=True,
        type=str,
        help="First BigMetaTable base filename (no extension).",
    )
    parser.add_argument(
        "--table-b",
        required=True,
        type=str,
        help="Second BigMetaTable base filename (no extension).",
    )
    parser.add_argument(
        "--output",
        required=False,
        type=str,
        help="Output merged BigMetaTable base filename (no extension).",
    )
    parser.add_argument(
        "--csv-output",
        choices=["full", "header", "none"],
        default="header",
        help=(
            "How much of a .csv companion to write for the merged output. 'full' (default) "
            "writes the entire merged table as CSV, matching prior behavior but re-serializing "
            "every row - expensive for large memmap-based datasets. 'header' writes just the "
            "header row, which is all BigMetaTable needs to load the .npy directly later. "
            "'none' skips it entirely (a later load must pass header= explicitly)."
        ),
    )
    return parser.parse_args()


def _require_table_files(name: str, label: str) -> None:
    """Check a BigMetaTable base name has at least one loadable representation."""
    has_npy = os.path.exists(f"{name}.npy")
    has_csv = os.path.exists(f"{name}.csv")
    if not (has_npy or has_csv):
        raise FileNotFoundError(f"{label} '{name}' has neither a .npy nor a .csv - nothing to load.")
    if has_npy and not has_csv:
        print(f"[INFO] {label} '{name}' is npy-only (no .csv); loading the memmap directly.")
    if not os.path.exists(f"{name}.txt"):
        print(f"[WARNING] {label} text metadata not found for base {name}")


def _column_discrepancies(table_a: BigMetaTable, table_b: BigMetaTable):
    """
    Compare table_a and table_b headers.

    Returns missing_in_b (columns of A absent from B - filled with 0 later).
    Raises ValueError if B has any column A does not have.
    """
    a_header = table_a.header
    b_set = set(table_b.header)

    extra_in_b = [c for c in table_b.header if c not in set(a_header)]
    if extra_in_b:
        raise ValueError(
            f"Table B has columns not present in Table A: {extra_in_b}. "
            "Table B cannot introduce columns unknown to Table A."
        )

    missing_in_b = [c for c in a_header if c not in b_set]
    return missing_in_b


def _warn_missing(missing_in_b, when: str) -> None:
    if missing_in_b:
        print(
            f"[WARNING] ({when}) Table A has {len(missing_in_b)} column(s) not present in Table B "
            f"(filled with 0 in the merged output): {missing_in_b}"
        )
    else:
        print(f"[INFO] ({when}) Table B contains every column of Table A.")


def _build_merged_table(table_a: BigMetaTable, table_b: BigMetaTable, output_name: str,
                         chunk_size: int = 100_000) -> BigMetaTable:
    """
    Build the merged table directly at `output_name`, in a single pass:
    table_a's rows are copied in row-chunks into the front of a new memmap
    (table_a is already in its own column order, so this is a plain
    contiguous copy); table_b's rows are then written into the tail,
    column-by-column, reordered/zero-filled to table_a's column layout.

    This replaces reindexing table_b into a full standalone copy and then
    handing both tables to merge_big_meta_tables for a second chunked copy -
    that path touched every row of table_b twice (once to align, once to
    merge) plus a further full rewrite in the final .save(). Building the
    output array once, directly, cuts that down to a single pass per table.
    """
    a_header = table_a.header
    b_index = {name: i for i, name in enumerate(table_b.header)}
    b_col_for_a = [b_index.get(name) for name in a_header]  # None -> missing from B, filled with 0

    rows_a, n_cols = table_a.table.shape
    rows_b = table_b.table.shape[0]
    dtype = table_a.table.dtype
    if table_b.table.dtype != dtype:
        raise ValueError("Dtype mismatch between Table A and Table B.")

    mmap_path = f"{output_name}.npy"
    merged = np.lib.format.open_memmap(mmap_path, mode='w+', dtype=dtype, shape=(rows_a + rows_b, n_cols))

    print(f"Copying Table A ({rows_a} rows)...")
    for start in tqdm(range(0, rows_a, chunk_size), desc="Copying Table A"):
        end = min(start + chunk_size, rows_a)
        merged[start:end] = table_a.table[start:end]
        merged.flush()

    print(f"Writing Table B ({rows_b} rows), reordered to Table A's columns...")
    # Row-chunked like Table A above: source/dest slices stay contiguous, and the
    # column reorder happens on a small in-RAM chunk rather than as a strided
    # column-at-a-time pass over the whole (row-major) memmap.
    zero_mask = np.array([b_col is None for b_col in b_col_for_a])
    safe_cols = np.array([b_col if b_col is not None else 0 for b_col in b_col_for_a])
    for start in tqdm(range(0, rows_b, chunk_size), desc="Aligning Table B columns"):
        end = min(start + chunk_size, rows_b)
        chunk = table_b.table[start:end][:, safe_cols]
        if zero_mask.any():
            chunk[:, zero_mask] = 0
        merged[rows_a + start:rows_a + end] = chunk
        merged.flush()

    # Blurred binaries are row-aligned labels unrelated to header columns - a plain concat.
    move_blurred = table_a.blurredbinaries is not None and table_b.blurredbinaries is not None
    if move_blurred:
        if table_a.blurredbinaries.shape[1] != table_b.blurredbinaries.shape[1]:
            raise ValueError("Blurredbinaries column mismatch between Table A and Table B.")
        print("Merging blurred boundaries...")
        merged_bb = np.lib.format.open_memmap(
            f"{output_name}blurredbinaries.npy", mode='w+',
            dtype=table_a.blurredbinaries.dtype,
            shape=(rows_a + rows_b, table_a.blurredbinaries.shape[1]),
        )
        merged_bb[:rows_a] = table_a.blurredbinaries
        merged_bb[rows_a:] = table_b.blurredbinaries
        merged_bb.flush()
        del merged_bb
    else:
        print("Not merging blurred boundaries")

    meta_rows = len(table_a.metadata) + len(table_b.metadata)
    assert merged.shape[0] == meta_rows, "Metadata length does not equal table rows!"
    with open(f"{output_name}.txt", "w") as f:
        table_a.write_meta_lines(f)
        table_b.write_meta_lines(f)

    del merged, table_a.table, table_b.table
    if table_a.blurredbinaries is not None:
        del table_a.blurredbinaries
    if table_b.blurredbinaries is not None:
        del table_b.blurredbinaries
    gc.collect()

    table_a._clear_metadata_rows()
    table_b._clear_metadata_rows()

    return BigMetaTable(output_name, header=list(a_header))


def main() -> None:
    args = parse_args()

    try:
        table_a_name = _validate_extensionless(args.table_a, "--table-a")
        table_b_name = _validate_extensionless(args.table_b, "--table-b")
        output_name = _validate_extensionless(args.output, "--output") if args.output else f"{table_a_name}__merged"
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    _require_table_files(table_a_name, "Table A")
    _require_table_files(table_b_name, "Table B")

    print(f"Loading table A: {table_a_name}")
    table_a = BigMetaTable(table_a_name)
    print(f"Loading table B: {table_b_name}")
    table_b = BigMetaTable(table_b_name)

    missing_in_b = _column_discrepancies(table_a, table_b)
    _warn_missing(missing_in_b, when="before merge")

    print("Merging tables...")
    merged_table = _build_merged_table(table_a, table_b, output_name)

    if args.csv_output == "full":
        merged_table.save_csv_streaming(name=output_name)
    elif args.csv_output == "header":
        with open(f"{output_name}.csv", "w", newline="") as f:
            csv.writer(f).writerow(merged_table.header)

    print("Merge complete.")
    print(f"Merged rows: {merged_table.table.shape[0]}")
    csv_note = {"full": f"{output_name}.csv (full)", "header": f"{output_name}.csv (header only)",
                "none": "no .csv"}[args.csv_output]
    print(f"Saved outputs: {output_name}.npy, {csv_note}, and {output_name}.txt")

    _warn_missing(missing_in_b, when="after merge")

    del merged_table
    gc.collect()


if __name__ == "__main__":
    main()
