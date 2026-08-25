"""
Merge an arbitrary number of BigMetaTable datasets from extensionless base names.

Each table may be backed by a .npy memmap, a .csv, or both - BigMetaTable already
loads the .npy directly when present (falling back to building one from .csv
otherwise), so no conversion step is needed here regardless of which
representation any input uses.

The merged output always inherits the first table's columns (name, count, and
order). Every other table is reindexed to match: its columns are reordered to
line up with the first table's, any column the first table has that a later one
lacks is filled with 0, and any column a later table has that the first lacks is
an error (no table after the first may introduce columns unknown to it).

Derivative sidecars (`<base>_dndP.npy`/`.csv`, `<base>_dndT.npy`/`.csv`) are
picked up automatically by BigMetaTable for any input that has them, and are
merged in parallel to the main table (same row concat as the main copy, in the
same table order). A table lacking a given sidecar is NaN-filled for its rows,
matching the importer's convention for unavailable derivatives. A column-count
mismatch between any two tables' sidecars is an error.

Usage:
    python scripts/merge_bigmetatables.py --tables <base_1> <base_2> [<base_3> ...] --output <merged_base>

Examples:
    python scripts/merge_bigmetatables.py \
        --tables data/MELTStables/110/trainset data/MELTStables/110/validset \
        --output data/MELTStables/110/train_valid_merged

    python scripts/merge_bigmetatables.py \
        --tables data/MELTStables/110/batch1 data/MELTStables/110/batch2 data/MELTStables/110/batch3 \
        --output data/MELTStables/110/all_batches_merged
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
from builder.processing import sidecar


def _validate_extensionless(value: str, arg_name: str) -> str:
    path = Path(value)
    if path.suffix:
        raise ValueError(
            f"{arg_name} must be extensionless (received '{value}')."
        )
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge an arbitrary number of BigMetaTable datasets from extensionless base names."
    )
    parser.add_argument(
        "--tables",
        required=True,
        nargs='+',
        type=str,
        help=(
            "BigMetaTable base filenames (no extension), in the order they should be "
            "concatenated. At least 2 required. The first table's columns (name, "
            "count, and order) become the merged output's columns; every other "
            "table is reindexed to match."
        ),
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
            "How much of a .csv companion to write for the merged output. 'full' "
            "writes the entire merged table as CSV, matching prior behavior but re-serializing "
            "every row - expensive for large memmap-based datasets. 'header' (default) writes just the "
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


def _column_discrepancies(reference: BigMetaTable, other: BigMetaTable, other_label: str):
    """
    Compare `other`'s header against the reference table's.

    Returns missing_in_other (columns of the reference absent from `other` -
    filled with 0 later). Raises ValueError if `other` has any column the
    reference does not have.
    """
    ref_header = reference.header
    other_set = set(other.header)

    extra_in_other = [c for c in other.header if c not in set(ref_header)]
    if extra_in_other:
        raise ValueError(
            f"{other_label} has columns not present in the reference table: {extra_in_other}. "
            f"{other_label} cannot introduce columns unknown to the reference table."
        )

    missing_in_other = [c for c in ref_header if c not in other_set]
    return missing_in_other


def _check_sidecar_row_parity(table: BigMetaTable, label: str) -> None:
    """Verify every attached derivative sidecar (dn/dP, dn/dT) has the same row
    count as its main table.

    BigMetaTable.__init__ already raises on this internally (via sidecar.attach,
    called on every load), so a mismatch should never reach here. This check exists
    to fail with a clear, table-labelled message at the point this script cares
    about it, rather than depending on that internal guard.
    """
    n_main = table.table.shape[0]
    for attr, suffix in sidecar.SIDECAR_SPECS:
        arr = getattr(table, attr, None)
        if arr is None:
            continue
        if arr.shape[0] != n_main:
            raise ValueError(
                f"{label}: sidecar '{attr}' ({suffix}.npy) has {arr.shape[0]} rows, "
                f"main table has {n_main}. Derivative tables must be row-parallel."
            )
        print(f"[INFO] {label}: sidecar '{attr}' row count matches main table ({arr.shape[0]} rows).")


def _warn_missing(missing_in_other, other_label: str, when: str) -> None:
    if missing_in_other:
        print(
            f"[WARNING] ({when}) Reference table has {len(missing_in_other)} column(s) not present "
            f"in {other_label} (filled with 0 in the merged output): {missing_in_other}"
        )
    else:
        print(f"[INFO] ({when}) {other_label} contains every column of the reference table.")


def _build_merged_table(tables: list[BigMetaTable], labels: list[str], output_name: str,
                         chunk_size: int = 100_000) -> BigMetaTable:
    """
    Build the merged table directly at `output_name`, in one pass per input table:
    each table's rows are copied, in the given order, into consecutive row-blocks
    of a new memmap sized for the full concatenation. The first (reference) table
    is already in its own column order, so its copy is a plain contiguous slice;
    every other table is reordered/zero-filled to the reference's column layout as
    it's written, on a small in-RAM chunk at a time rather than as a strided
    column-at-a-time pass over the whole (row-major) memmap.

    This builds the output array once, directly, rather than reindexing every
    non-reference table into a full standalone copy first and merging as a
    second pass - that would touch every row of those tables twice.
    """
    reference = tables[0]
    ref_header = reference.header
    rows = [t.table.shape[0] for t in tables]
    total_rows = sum(rows)
    n_cols = reference.table.shape[1]
    dtype = reference.table.dtype
    for table, label in zip(tables[1:], labels[1:]):
        if table.table.dtype != dtype:
            raise ValueError(f"Dtype mismatch: {label} does not match the reference table.")

    mmap_path = f"{output_name}.npy"
    merged = np.lib.format.open_memmap(mmap_path, mode='w+', dtype=dtype, shape=(total_rows, n_cols))

    offset = 0
    for table, label in zip(tables, labels):
        n = table.table.shape[0]
        if table is reference:
            print(f"Copying {label} ({n} rows)...")
            for start in tqdm(range(0, n, chunk_size), desc=f"Copying {label}"):
                end = min(start + chunk_size, n)
                merged[offset + start:offset + end] = table.table[start:end]
                merged.flush()
        else:
            other_index = {name: idx for idx, name in enumerate(table.header)}
            col_for_ref = [other_index.get(name) for name in ref_header]  # None -> missing, filled with 0
            zero_mask = np.array([c is None for c in col_for_ref])
            safe_cols = np.array([c if c is not None else 0 for c in col_for_ref])
            print(f"Writing {label} ({n} rows), reordered to the reference table's columns...")
            for start in tqdm(range(0, n, chunk_size), desc=f"Aligning {label} columns"):
                end = min(start + chunk_size, n)
                chunk = table.table[start:end][:, safe_cols]
                if zero_mask.any():
                    chunk[:, zero_mask] = 0
                merged[offset + start:offset + end] = chunk
                merged.flush()
        offset += n

    # Blurred binaries are row-aligned labels unrelated to header columns - a plain
    # concat, in table order. Only merged when every table has them (matching the
    # original two-table behaviour: a mix of present/absent isn't a defined merge).
    if all(t.blurredbinaries is not None for t in tables):
        n_bb_cols = reference.blurredbinaries.shape[1]
        for table, label in zip(tables[1:], labels[1:]):
            if table.blurredbinaries.shape[1] != n_bb_cols:
                raise ValueError(f"Blurredbinaries column mismatch: {label} vs the reference table.")
        print("Merging blurred boundaries...")
        merged_bb = np.lib.format.open_memmap(
            f"{output_name}blurredbinaries.npy", mode='w+',
            dtype=reference.blurredbinaries.dtype,
            shape=(total_rows, n_bb_cols),
        )
        offset = 0
        for table in tables:
            n = table.blurredbinaries.shape[0]
            merged_bb[offset:offset + n] = table.blurredbinaries
            offset += n
        merged_bb.flush()
        del merged_bb
    else:
        print("Not merging blurred boundaries")

    # Derivative sidecars (dn/dP, dn/dT). Already row-aligned to their own main
    # table (BigMetaTable.attach asserts this at load time), so merging is a plain
    # row concat like blurredbinaries above, in table order. Unlike blurredbinaries,
    # presence is allowed to differ per table - any table lacking a given sidecar is
    # NaN-filled for its rows, matching the importer's own convention for
    # unavailable derivatives (see sidecar.py's module docstring).
    for attr, suffix in sidecar.SIDECAR_SPECS:
        arrs = [getattr(t, attr, None) for t in tables]
        if all(arr is None for arr in arrs):
            continue
        present = [(arr, label) for arr, label in zip(arrs, labels) if arr is not None]
        n_cols_sc = present[0][0].shape[1]
        dtype_sc = present[0][0].dtype
        for arr, label in present[1:]:
            if arr.shape[1] != n_cols_sc:
                raise ValueError(f"Sidecar '{attr}' column mismatch: {label} vs {present[0][1]}.")
        missing_labels = [label for arr, label in zip(arrs, labels) if arr is None]
        if missing_labels:
            print(f"Merging sidecar '{attr}' ({', '.join(missing_labels)} lack it - NaN-filled)...")
        else:
            print(f"Merging sidecar '{attr}'...")
        merged_sc = np.lib.format.open_memmap(
            f"{output_name}{suffix}.npy", mode='w+', dtype=dtype_sc,
            shape=(total_rows, n_cols_sc),
        )
        offset = 0
        for table in tables:
            n = table.table.shape[0]
            arr = getattr(table, attr, None)
            merged_sc[offset:offset + n] = arr if arr is not None else np.nan
            offset += n
        merged_sc.flush()
        del merged_sc

    meta_rows = sum(len(t.metadata) for t in tables)
    assert merged.shape[0] == meta_rows, "Metadata length does not equal table rows!"
    with open(f"{output_name}.txt", "w") as f:
        for table in tables:
            table.write_meta_lines(f)

    del merged
    for table in tables:
        del table.table
        if table.blurredbinaries is not None:
            del table.blurredbinaries
        for attr, _ in sidecar.SIDECAR_SPECS:
            if getattr(table, attr, None) is not None:
                delattr(table, attr)
    gc.collect()

    for table in tables:
        table._clear_metadata_rows()

    return BigMetaTable(output_name, header=list(ref_header))


def main() -> None:
    args = parse_args()

    if len(args.tables) < 2:
        print("ERROR: --tables requires at least 2 table base names to merge.")
        sys.exit(1)

    try:
        table_names = [_validate_extensionless(t, "--tables") for t in args.tables]
        output_name = _validate_extensionless(args.output, "--output") if args.output else f"{table_names[0]}__merged"
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    labels = [f"Table {i + 1} [{name}]" for i, name in enumerate(table_names)]

    for name, label in zip(table_names, labels):
        _require_table_files(name, label)

    tables: list[BigMetaTable] = []
    for name, label in zip(table_names, labels):
        print(f"Loading {label}")
        table = BigMetaTable(name)
        _check_sidecar_row_parity(table, label)
        tables.append(table)

    reference = tables[0]
    missing_by_label: dict[str, list[str]] = {}
    for table, label in zip(tables[1:], labels[1:]):
        missing = _column_discrepancies(reference, table, label)
        missing_by_label[label] = missing
        _warn_missing(missing, label, when="before merge")

    print("Merging tables...")
    merged_table = _build_merged_table(tables, labels, output_name)
    _check_sidecar_row_parity(merged_table, "Merged output")

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

    sidecar_paths = [f"{output_name}{suffix}.npy" for _, suffix in sidecar.SIDECAR_SPECS
                      if os.path.exists(f"{output_name}{suffix}.npy")]
    if sidecar_paths:
        print(f"Saved sidecars: {', '.join(sidecar_paths)}")

    for label, missing in missing_by_label.items():
        _warn_missing(missing, label, when="after merge")

    del merged_table
    gc.collect()


if __name__ == "__main__":
    main()
