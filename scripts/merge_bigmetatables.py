"""
Merge two BigMetaTable datasets from extensionless base names.

Each table may be backed by a .npy memmap, a .csv, or both - BigMetaTable already
loads the .npy directly when present (falling back to building one from .csv
otherwise), so no conversion step is needed here regardless of which
representation either input uses.

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


repo_src = str(Path(__file__).resolve().parents[1] / "src")
if repo_src not in sys.path:
    sys.path.insert(0, repo_src)

from builder.processing.BigMetaTable import BigMetaTable, merge_big_meta_tables


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
        default="full",
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

    print("Merging tables...")
    merged_table = merge_big_meta_tables(
        [table_a, table_b],
        new_filename=output_name+'temp',
    )

    merged_table.save(output_name, save_csv=(args.csv_output == "full"))
    if args.csv_output == "header":
        with open(f"{output_name}.csv", "w", newline="") as f:
            csv.writer(f).writerow(merged_table.header)

    print("Merge complete.")
    print(f"Merged rows: {merged_table.table.shape[0]}")
    csv_note = {"full": f"{output_name}.csv (full)", "header": f"{output_name}.csv (header only)",
                "none": "no .csv"}[args.csv_output]
    print(f"Saved outputs: {output_name}.npy, {csv_note}, and {output_name}.txt")
    del merged_table, table_a, table_b
    gc.collect()


if __name__ == "__main__":
    main()
