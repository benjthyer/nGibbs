"""
Fit T as quadratic regression on entropy S and all bulk-composition elements.

Model form:
    T = b0 + bS*S + bS2*S^2
        + sum_i [ b_Xi*Xi + b_Xi2*Xi^2 ]

where Xi runs over every column matching *Bulk_comp_elements in the CSV.

Coefficients are saved to a text file whose name contains the R^2 value.

Accepts either a standalone CSV (header + data rows), or a BigMetaTable pair
(a .npy array plus a header-only .csv of column names) - pass the .npy file,
the shared base name with no extension, or the paired .csv itself.

Usage:
    python scripts/fit_T_from_S_and_bulk.py \
        --csv data/MELTStables/HeFESTo/HeFESTo_TrainsetMar2NTP.csv

    python scripts/fit_T_from_S_and_bulk.py \
        --csv data/MELTStables/HeFESTo/HeFESTo_TrainsetMar2NTP.csv \
        --out-dir data/MELTStables/HeFESTo

    python scripts/fit_T_from_S_and_bulk.py \
        --csv data/MELTStables/HeFESTo/HeFESTo_TrainsetMar2NTP
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

repo_src = str(Path(__file__).resolve().parents[1] / "src")
if repo_src not in sys.path:
    sys.path.insert(0, repo_src)

from builder.processing.BigMetaTable import BigMetaTable

KEEP_RANGE = [1.75, 2.9]  # Only keep rows with S in this range for fitting


def _load_table(path: Path) -> pd.DataFrame:
    """Load tabular data from a standalone CSV or a BigMetaTable .npy/.csv pair.

    A bare .npy file, an extensionless base name, or a .csv that has a sibling
    .npy (the paired BigMetaTable format, where the .csv holds just the header
    row) are all loaded through BigMetaTable so the array and its column names
    are read together. A .csv with no sibling .npy is read directly with
    pandas, preserving the original standalone-CSV behavior.
    """
    if path.suffix == ".npy":
        base = path.with_suffix("")
    elif path.suffix == "":
        base = path
    elif path.suffix == ".csv" and path.with_suffix(".npy").exists():
        base = path.with_suffix("")
    else:
        return pd.read_csv(path)

    table = BigMetaTable(str(base))
    return pd.DataFrame(np.asarray(table.table), columns=table.header)

def _find_column(df: pd.DataFrame, *, exact: list[str], prefixes: list[str], label: str) -> str:
    for name in exact:
        if name in df.columns:
            return name
    for prefix in prefixes:
        matches = [c for c in df.columns if c.startswith(prefix)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous {label} column for prefix '{prefix}': {matches}")
    raise ValueError(
        f"Could not find {label} column. Checked exact={exact}, prefixes={prefixes}."
    )


def fit_T_from_S_and_bulk(csv_path: Path, out_dir: Path) -> None:
    df = _load_table(csv_path)

    t_col = _find_column(
        df,
        exact=["T", "Temperature", "T(K)(System_main)"],
        prefixes=["T(", "Temperature("],
        label="temperature (T)",
    )
    s_col = _find_column(
        df,
        exact=["S", "Entropy", "S(J/g/K)(System_main)"],
        prefixes=["S(", "Entropy("],
        label="entropy (S)",
    )

    p_col = _find_column(
        df,
        exact=["P", "P(GPa)", "P(GPa)(System_main)"],
        prefixes=["P(", "Pressure("],
        label="pressure (P)",
    )

    bulk_cols = [c for c in df.columns if c.endswith("(Bulk_comp_elements)")]
    if not bulk_cols:
        raise ValueError(
            "No columns ending in '(Bulk_comp_elements)' found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    keep_cols = [t_col, s_col, p_col] + bulk_cols
    data = df[keep_cols].dropna()
    in_range = (data[s_col] > KEEP_RANGE[0]) & (data[s_col] < KEEP_RANGE[1]) & (data[p_col] < 25)
    data = data[in_range]
    n = len(data)

    t = data[t_col].to_numpy(dtype=float)
    s = data[s_col].to_numpy(dtype=float)
    p = data[p_col].to_numpy(dtype=float)
    bulk = [data[c].to_numpy(dtype=float) for c in bulk_cols]

    feature_parts = [s, s**2, p, p**2]
    feature_names = ["S", "S^2", "P", "P^2"]
    for col, arr in zip(bulk_cols, bulk):
        element = col.replace("(Bulk_comp_elements)", "").strip()
        feature_parts.append(arr)
        feature_names.append(element)
        feature_parts.append(arr**2)
        feature_names.append(f"{element}^2")

    X = np.column_stack(feature_parts)

    model = LinearRegression(fit_intercept=True)
    model.fit(X, t)
    t_pred = model.predict(X)
    r2 = r2_score(t, t_pred)

    b0 = float(model.intercept_)
    coefs = [float(v) for v in model.coef_]

    print("--- Temperature Regression Results ---")
    print(f"CSV:        {csv_path}")
    print(f"Rows used:  {n}")
    print(f"T column:   '{t_col}'")
    print(f"S column:   '{s_col}'")
    print(f"Bulk cols:  {bulk_cols}")
    print("")
    print("Model:")
    terms = " + ".join(f"b_{name}*{name}" for name in feature_names)
    print(f"  T = b0 + {terms}")
    print("")
    print("Parameters:")
    print(f"  b0  (intercept) = {b0:.12g}")
    for name, c in zip(feature_names, coefs):
        print(f"  b_{name:<10} = {c:.12g}")
    print("")
    print(f"R^2 = {r2:.12g}")

    # ---- save coefficients ----
    r2_str = f"{r2:.5f}".replace(".", "p")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"T_from_S_X_coefs_R2_{r2_str}.txt"

    lines = [
        "Temperature regression: T = f(S, bulk composition elements)",
        f"CSV: {csv_path}",
        f"Rows used: {n}",
        f"R^2: {r2:.12g}",
        "",
        f"Model: T = b0 + {terms}",
        "",
        "Coefficients:",
        f"  b0  (intercept) = {b0:.12g}",
    ]
    for name, c in zip(feature_names, coefs):
        lines.append(f"  b_{name:<10} = {c:.12g}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nCoefficients saved to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit T as quadratic regression on entropy S and all bulk-composition "
            "elements (*(Bulk_comp_elements) columns). Saves coefficients to a text "
            "file whose name contains the R^2 value."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/MELTStables/HeFESTo/HeFESTo_TrainsetMar2NTP.csv"),
        help=(
            "Path to HeFESTo data: either a standalone CSV, or a BigMetaTable "
            "(.npy array + header-only .csv) given as the .npy file, the "
            "extensionless base name, or the paired .csv."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write the coefficient file (defaults to same dir as CSV).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir is not None else args.csv.parent
    fit_T_from_S_and_bulk(args.csv, out_dir)


if __name__ == "__main__":
    main()
