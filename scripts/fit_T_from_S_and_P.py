"""
Fit T as a quadratic regression on entropy S and pressure P only.

Model form:
    T = b0 + bS*S + bS2*S^2 + bP*P + bP2*P^2

Coefficients are saved to a text file whose name contains the R^2 value.

Usage:
    python scripts/fit_T_from_S_and_P.py \
        --csv data/MELTStables/HeFESTo/HeFESTo_TrainsetMar2NTP.csv

    python scripts/fit_T_from_S_and_P.py \
        --csv data/MELTStables/HeFESTo/HeFESTo_TrainsetMar2NTP.csv \
        --out-dir data/MELTStables/HeFESTo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

KEEP_RANGE = [0,10] #[1.75, 2.9]  # S range used for fitting


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
    raise ValueError(f"Could not find {label} column. Checked exact={exact}, prefixes={prefixes}.")


def fit_T_from_S_and_P(csv_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_path)

    t_col = _find_column(df, exact=["T", "Temperature", "T(K)(System_main)"],
                         prefixes=["T(", "Temperature("], label="temperature (T)")
    s_col = _find_column(df, exact=["S", "Entropy", "S(J/g/K)(System_main)"],
                         prefixes=["S(", "Entropy("], label="entropy (S)")
    p_col = _find_column(df, exact=["P", "P(GPa)", "P(GPa)(System_main)"],
                         prefixes=["P(", "Pressure("], label="pressure (P)")

    data = df[[t_col, s_col, p_col]].dropna()
    in_range = (data[s_col] > KEEP_RANGE[0]) & (data[s_col] < KEEP_RANGE[1]) & (data[p_col] < 25)
    data = data[in_range]
    n = len(data)

    t = data[t_col].to_numpy(dtype=float)
    s = data[s_col].to_numpy(dtype=float)
    p = data[p_col].to_numpy(dtype=float)

    X = np.column_stack([s, s**2, p, p**2])
    feature_names = ["S", "S^2", "P", "P^2"]

    model = LinearRegression(fit_intercept=True)
    model.fit(X, t)
    r2 = r2_score(t, model.predict(X))

    b0 = float(model.intercept_)
    coefs = [float(v) for v in model.coef_]

    print(f"CSV:       {csv_path}")
    print(f"Rows used: {n}")
    print(f"R^2:       {r2:.6f}")
    print(f"b0 (intercept) = {b0:.12g}")
    for name, c in zip(feature_names, coefs):
        print(f"b_{name:<6} = {c:.12g}")

    r2_str = f"{r2:.5f}".replace(".", "p")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"T_from_S_P_coefs_R2_{r2_str}.txt"

    terms = " + ".join(f"b_{n}*{n}" for n in feature_names)
    lines = [
        "Temperature regression: T = f(S, P)",
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
        description="Fit T as quadratic regression on S and P only. Saves coefficients to a text file."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/MELTStables/HeFESTo/HeFESTo_TrainsetMar2NTP.csv"),
        help="Path to HeFESTo CSV file.",
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
    fit_T_from_S_and_P(args.csv, out_dir)


if __name__ == "__main__":
    main()
