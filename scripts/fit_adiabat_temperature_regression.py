"""
Fit temperature as a linear model of P, P^2, S, and S^2 with intercept.

Model form:
    T = b0 + bP*P + bP2*P^2 + bS*S + bS2*S^2

Usage example:
    python scripts/fit_adiabat_temperature_regression.py \
        --csv data/MELTStables/HeFESTo/HeFESTo_TrainsetApr4_AdiabatsNTP.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def _find_column(df: pd.DataFrame, *, exact: list[str], prefixes: list[str], label: str) -> str:
    for name in exact:
        if name in df.columns:
            return name

    for prefix in prefixes:
        matches = [c for c in df.columns if c.startswith(prefix)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous {label} column for prefix '{prefix}': {matches}"
            )

    raise ValueError(
        f"Could not find {label} column. Checked exact={exact}, prefixes={prefixes}."
    )


def fit_temperature_regression(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)

    p_col = _find_column(
        df,
        exact=["P", "Pressure", "P(GPa)(System_main)"],
        prefixes=["P(", "Pressure("],
        label="pressure (P)",
    )
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

    data = df[[p_col, s_col, t_col]].dropna()
    p = data[p_col].to_numpy(dtype=float)
    s = data[s_col].to_numpy(dtype=float)
    t = data[t_col].to_numpy(dtype=float)

    x = np.column_stack([p, p**2, s, s**2])

    model = LinearRegression(fit_intercept=True)
    model.fit(x, t)
    t_pred = model.predict(x)
    r2 = r2_score(t, t_pred)

    b0 = float(model.intercept_)
    b_p, b_p2, b_s, b_s2 = [float(v) for v in model.coef_]

    print("--- Temperature Regression Results ---")
    print(f"CSV: {csv_path}")
    print(f"Rows used: {len(data)}")
    print(f"Columns: P='{p_col}', S='{s_col}', T='{t_col}'")
    print("")
    print("Model:")
    print("  T = b0 + bP*P + bP2*P^2 + bS*S + bS2*S^2")
    print("")
    print("Parameters:")
    print(f"  b0  (intercept) = {b0:.12g}")
    print(f"  bP              = {b_p:.12g}")
    print(f"  bP2             = {b_p2:.12g}")
    print(f"  bS              = {b_s:.12g}")
    print(f"  bS2             = {b_s2:.12g}")
    print("")
    print(f"R^2 = {r2:.12g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit T as linear regression in [P, P^2, S, S^2] with intercept "
            "and print coefficients plus R^2."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/MELTStables/HeFESTo/HeFESTo_TrainsetApr4_AdiabatsNTP.csv"),
        help="Path to adiabat CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fit_temperature_regression(args.csv)


if __name__ == "__main__":
    main()
