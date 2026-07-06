"""
Recover masses for compositionally-constant phases from a BigMetaTable's text
metadata.

Some MELTS runs stabilize phases that show up in the per-row text metadata as
an untracked residual rather than being written into their own table columns.
For phases with a single fixed oxide formula (no compositional freedom, e.g.
graphite, quartz, calcite), that residual can be converted back into a mass
by projecting the bulk-composition residual (actual bulk oxide moles minus
tracked-phase oxide moles) onto the phase's stoichiometry row in the compToOx
projection CSV. See BigMetaTable.recover_constant_composition_phases.

Writes/fills "mass (gm)(<phasename>)" columns for each requested phase.
Existing non-zero values are left untouched, and the text metadata is never
modified.

Usage:
    python scripts/recover_constant_phases.py --table <base_name> --phases <phase> [<phase> ...] [--output <output_base>]

Examples:
    python scripts/recover_constant_phases.py \
        --table data/MELTStables/110/trainset \
        --phases graphite calcite dolomite magnesite

    python scripts/recover_constant_phases.py \
        --table data/MELTStables/110/trainset \
        --phases graphite quartz \
        --output data/MELTStables/110/trainset_recovered
"""

import argparse
import csv
import gc
import sys
from pathlib import Path


repo_src = str(Path(__file__).resolve().parents[1] / "src")
if repo_src not in sys.path:
    sys.path.insert(0, repo_src)

from builder.processing.BigMetaTable import BigMetaTable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover compositionally-constant phase masses from a BigMetaTable's text metadata."
    )
    parser.add_argument(
        "--table",
        required=True,
        type=str,
        help="BigMetaTable base filename (no extension).",
    )
    parser.add_argument(
        "--phases",
        required=True,
        nargs="+",
        help="Compositionally-constant phase names to recover (e.g. graphite calcite dolomite).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output BigMetaTable base filename (no extension). Defaults to overwriting --table.",
    )
    parser.add_argument(
        "--csv-output",
        choices=["full", "header", "none"],
        default="header",
        help=(
            "How much of a .csv companion to write for the resaved output. 'full' writes the "
            "entire table as CSV - expensive for large memmap-based datasets. 'header' "
            "(default) writes just the header row, which is all BigMetaTable needs to load "
            "the .npy directly later. 'none' skips it entirely (a later load must pass "
            "header= explicitly)."
        ),

    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_name = args.output if args.output else args.table

    print(f"Loading table: {args.table}")
    table = BigMetaTable(args.table)

    print(f"Recovering phases: {args.phases}")
    table.recover_constant_composition_phases(args.phases)

    table.save(output_name, save_csv=(args.csv_output == "full"))
    if args.csv_output == "header":
        with open(f"{output_name}.csv", "w", newline="") as f:
            csv.writer(f).writerow(table.header)

    csv_note = {"full": f"{output_name}.csv (full)", "header": f"{output_name}.csv (header only)",
                "none": "no .csv"}[args.csv_output]
    print(f"Saved outputs: {output_name}.npy, {csv_note}, and {output_name}.txt")

    del table
    gc.collect()


if __name__ == "__main__":
    main()
