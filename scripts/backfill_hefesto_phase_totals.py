"""
Backfill missing HeFESTo phase-total mole columns in an existing CSV table.

This script adds columns named:
    total (moles)(<phase>)

for phases that do not already have that column, by summing all non-state
component columns for each phase.

Usage examples:
    python scripts/backfill_hefesto_phase_totals.py \
        data/MELTStables/HeFESTo/HeFESTo_Validset041026_low_noise_adiabats

    python scripts/backfill_hefesto_phase_totals.py \
        data/MELTStables/HeFESTo/HeFESTo_Validset041026_low_noise_adiabats.csv \
        --output data/MELTStables/HeFESTo/HeFESTo_Validset041026_low_noise_adiabats_fixed.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


EXCLUDED_PHASES: Set[str] = {"System_main", "Bulk_comp", "Bulk_comp_elements"}

# Keep aligned with DatasetIndexer STATE_VARIABLES plus common HeFESTo system vars.
STATE_VARIABLE_COMPONENTS: Set[str] = {
    "P(GPa)",
    "T(K)",
    "rho(g/cm^3)",
    "mass (gm)",
    "VS(km/s)",
    "VP(km/s)",
    "H(kJ/g)",
    "cp(J/g/K)",
    "S(J/g/K)",
    "KS(GPa)",
    "alpha(1e5_K^-1)",
    "rho (gm/cc)",
    "H (kJ)",
    "S (J/K)",
    "V (cc)",
    "liq mass (gm)",
    "liq rho (gm/cc)",
    "liq vis (log 10 poise)",
    "liq H (kJ)",
    "liq S (J/K)",
    "liq V (cc)",
    "total (moles)",
}


def parse_component_phase(header: str) -> Optional[Tuple[str, str]]:
    """Parse headers like 'component(phase)' into ('component', 'phase')."""
    if "(" not in header or not header.endswith(")"):
        return None
    component, phase = header.rsplit("(", 1)
    return component, phase[:-1]


def resolve_csv_path(path_str: str) -> Path:
    """Allow extensionless base names and normalize to an existing .csv path."""
    path = Path(path_str)
    csv_path = path if path.suffix.lower() == ".csv" else path.with_suffix(".csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return csv_path


def collect_phase_groups(columns: List[str]) -> Dict[str, Dict[str, object]]:
    """Group columns by phase and classify as component/state/total."""
    groups: Dict[str, Dict[str, object]] = {}

    for col in columns:
        parsed = parse_component_phase(col)
        if parsed is None:
            continue

        component, phase = parsed
        if phase in EXCLUDED_PHASES:
            continue

        info = groups.setdefault(
            phase,
            {
                "all": [],
                "component": [],
                "state": [],
                "has_total": False,
            },
        )
        info["all"].append(col)

        if component == "total (moles)":
            info["has_total"] = True
        elif component in STATE_VARIABLE_COMPONENTS:
            info["state"].append(col)
        else:
            info["component"].append(col)

    return groups


def build_reordered_columns(
    original_columns: List[str],
    new_total_columns: Dict[str, str],
    phase_groups: Dict[str, Dict[str, object]],
) -> List[str]:
    """Insert each new total column before first state column of that phase."""
    first_state_by_phase: Dict[str, str] = {}
    last_col_by_phase: Dict[str, str] = {}

    for phase, info in phase_groups.items():
        state_cols = info["state"]
        all_cols = info["all"]
        if state_cols:
            first_state_by_phase[phase] = state_cols[0]
        if all_cols:
            last_col_by_phase[phase] = all_cols[-1]

    ordered: List[str] = []
    inserted: Set[str] = set()

    for col in original_columns:
        parsed = parse_component_phase(col)
        if parsed is not None:
            _, phase = parsed
            if phase in new_total_columns and phase not in inserted:
                if first_state_by_phase.get(phase) == col:
                    ordered.append(new_total_columns[phase])
                    inserted.add(phase)

        ordered.append(col)

        if parsed is not None:
            _, phase = parsed
            if phase in new_total_columns and phase not in inserted:
                if phase not in first_state_by_phase and last_col_by_phase.get(phase) == col:
                    ordered.append(new_total_columns[phase])
                    inserted.add(phase)

    expected = set(new_total_columns.keys())
    if inserted != expected:
        missing = sorted(expected - inserted)
        raise RuntimeError(
            "Failed to place total columns for phases: " + ", ".join(missing)
        )

    return ordered


def backfill_phase_totals(input_csv: Path, output_csv: Path) -> int:
    """Add missing total (moles)(phase) columns and save CSV. Returns count added."""
    df = pd.read_csv(input_csv)
    original_columns = list(df.columns)
    phase_groups = collect_phase_groups(original_columns)

    new_total_series: Dict[str, pd.Series] = {}
    new_total_columns: Dict[str, str] = {}

    for phase, info in phase_groups.items():
        if info["has_total"]:
            continue

        component_cols = list(info["component"])
        if not component_cols:
            raise ValueError(
                f"Phase '{phase}' is missing total (moles) but has no component columns."
            )

        component_frame = df[component_cols].apply(pd.to_numeric, errors="raise")
        total_col = f"total (moles)({phase})"
        new_total_series[phase] = component_frame.sum(axis=1)
        new_total_columns[phase] = total_col

    if not new_total_columns:
        print("No missing total (moles)(phase) columns were found.")
        return 0

    for phase, total_col in new_total_columns.items():
        if total_col in df.columns:
            raise RuntimeError(f"Unexpected duplicate column already exists: {total_col}")
        df[total_col] = new_total_series[phase]

    reordered_columns = build_reordered_columns(original_columns, new_total_columns, phase_groups)
    df = df[reordered_columns]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_csv.with_suffix(output_csv.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(output_csv)

    print(f"Added {len(new_total_columns)} total columns.")
    print(f"Saved: {output_csv}")
    return len(new_total_columns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill missing HeFESTo total (moles)(phase) columns in a CSV table."
    )
    parser.add_argument(
        "--input-table",
        type=str,
        help="Input table path, with or without .csv extension.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output CSV path. Defaults to overwriting input.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = resolve_csv_path(args.input_table)
    output_csv = Path(args.output) if args.output else input_csv
    if output_csv.suffix.lower() != ".csv":
        output_csv = output_csv.with_suffix(".csv")

    backfill_phase_totals(input_csv=input_csv, output_csv=output_csv)


if __name__ == "__main__":
    main()