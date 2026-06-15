"""
Merge two BigMetaTable datasets from extensionless base names.

Usage:
    python scripts/merge_bigmetatables.py --table-a <base_a> --table-b <base_b> --output <merged_base>

Examples:
    python scripts/merge_bigmetatables.py \
        --table-a data/MELTStables/110/trainset \
        --table-b data/MELTStables/110/validset \
        --output data/MELTStables/110/train_valid_merged
"""

import argparse
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        table_a_name = _validate_extensionless(args.table_a, "--table-a")
        table_b_name = _validate_extensionless(args.table_b, "--table-b")
        output_name = _validate_extensionless(args.output, "--output") if args.output else f"{table_a_name}__merged"
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    assert os.path.exists(f"{table_a_name}.csv") 
    if not os.path.exists(f"{table_a_name}.txt"):
        print(f"[WARNING] Table A text metadata not found for base {table_a_name}")
    assert os.path.exists(f"{table_b_name}.csv") 
    if not os.path.exists(f"{table_b_name}.txt"):
        print(f"[WARNING] Table B text metadata not found for base {table_b_name}")
   
    print(f"Loading table A: {table_a_name}")
    table_a = BigMetaTable(table_a_name)
    print(f"Loading table B: {table_b_name}")
    table_b = BigMetaTable(table_b_name)

    print("Merging tables...")
    merged_table = merge_big_meta_tables(
        [table_a, table_b],
        new_filename=output_name+'temp',
    )

    merged_table.save(output_name, save_csv=True)

    print("Merge complete.")
    print(f"Merged rows: {merged_table.table.shape[0]}")
    print(f"Saved outputs: {output_name}.npy,  {output_name}.csv, and {output_name}.txt")
    del merged_table, table_a, table_b
    gc.collect()


if __name__ == "__main__":
    main()
