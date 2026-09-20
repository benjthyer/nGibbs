"""
Fit T as quadratic regression on entropy S and all bulk-composition elements.

Model form:
    T = b0 + bS*S + bS2*S^2
        + sum_i [ b_Xi*Xi + b_Xi2*Xi^2 ]

where Xi runs over every column matching *Bulk_comp_elements in the CSV.

Coefficients are saved to a text file whose name contains the R^2 value.

Accepts a standalone CSV (header + data rows), a BigMetaTable pair (a .npy
array plus a header-only .csv of column names - pass the .npy file, the
shared base name with no extension, or the paired .csv itself), or an
ML-ready bundle (.tar.gz, or its extensionless base name) as produced by
prepareML.py: T is read out of free_outputs by name, P/S out of the first
two feature columns (named by ml_indexer.featureNames), and the bulk
elements out of the remaining feature columns (named by ml_indexer.Elkeys).

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

KEEP_RANGE = [1.75, 3.5]# 2.9]  # Only keep rows with S in this range for fitting
P_MAX_DEFAULT = None  # No pressure upper bound by default; override with --p-max


def _find_column(columns: list[str], *, exact: list[str], prefixes: list[str], label: str) -> str:
    for name in exact:
        if name in columns:
            return name
    # Bulk-composition columns are never T/S/P columns, but a bulk element can
    # collide with a prefix by coincidence (e.g. phosphorus's
    # "P(Bulk_comp_elements)" starts with the pressure prefix "P(") - exclude
    # them so that collision can't shadow the real column.
    candidates = [c for c in columns if not c.endswith("(Bulk_comp_elements)")]
    for prefix in prefixes:
        matches = [c for c in candidates if c.startswith(prefix)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous {label} column for prefix '{prefix}': {matches}")
    raise ValueError(
        f"Could not find {label} column. Checked exact={exact}, prefixes={prefixes}."
    )


def _load_bundle_as_frame(bundle_path: Path) -> pd.DataFrame:
    """Reshape an ML-ready bundle into the same T/S/P/bulk-element column
    layout a standalone CSV has, so it can be fit and plotted through the
    exact same code path.

    features[:, :len(featureNames)] are named by ml_indexer.featureNames
    (P and S for this fit), features[:, len(featureNames):] are the bulk
    elements in ml_indexer.Elkeys order, and T is pulled out of free_outputs
    by name.
    """
    from ngibbs.utils.file_utils import load_ml_bundle

    bundle = load_ml_bundle(bundle_path, arrays=["features", "free_outputs"])
    ml_indexer = bundle.ml_indexer
    feature_names = list(getattr(ml_indexer, "featureNames", None) or [])
    elkeys = list(ml_indexer.Elkeys)
    if not feature_names:
        raise ValueError(f"Bundle ml_indexer has no featureNames: {bundle_path}")

    features = np.asarray(bundle.features, dtype=float)
    expected_cols = len(feature_names) + len(elkeys)
    if features.shape[1] != expected_cols:
        raise ValueError(
            f"Bundle features has {features.shape[1]} columns, expected "
            f"{expected_cols} (= {len(feature_names)} featureNames + {len(elkeys)} Elkeys)."
        )

    data = {name: features[:, i] for i, name in enumerate(feature_names)}
    for j, el in enumerate(elkeys):
        data[f"{el}(Bulk_comp_elements)"] = features[:, len(feature_names) + j]

    free_output_names = list(getattr(ml_indexer, "free_outputs", None) or [])
    if bundle.free_outputs is None or not free_output_names:
        raise ValueError(f"Bundle missing free_outputs (needed for T): {bundle_path}")
    t_name = _find_column(
        free_output_names,
        exact=["T", "Temperature", "T(K)(System_main)"],
        prefixes=["T(", "Temperature("],
        label="temperature (T)",
    )
    t_idx = free_output_names.index(t_name)
    data[t_name] = np.asarray(bundle.free_outputs, dtype=float)[:, t_idx]

    return pd.DataFrame(data)


def _load_table(path: Path) -> pd.DataFrame:
    """Load tabular data from a standalone CSV, a BigMetaTable .npy/.csv
    pair, or an ML-ready bundle (.tar.gz).

    A bare .npy file, an extensionless base name, or a .csv that has a sibling
    .npy (the paired BigMetaTable format, where the .csv holds just the header
    row) are all loaded through BigMetaTable so the array and its column names
    are read together. A path ending in .tar.gz, or an extensionless base name
    with a .tar.gz sibling, is loaded as an ML-ready bundle. A .csv with none
    of the above is read directly with pandas, preserving the original
    standalone-CSV behavior.
    """
    if path.name.endswith(".tar.gz"):
        return _load_bundle_as_frame(path)
    if path.suffix == "" and path.with_suffix(".tar.gz").exists():
        return _load_bundle_as_frame(path.with_suffix(".tar.gz"))

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


def compute_T_from_S_and_bulk_regression(csv_path: Path, p_max: float | None = P_MAX_DEFAULT) -> dict:
    """Load, filter, fit, and return everything needed to report on or plot the fit.

    Shared by the CLI fitting entry point below and by plotting scripts, so the
    fitted model and the plotted data can never drift apart.

    p_max: if given, drop rows with P >= p_max before fitting. None (default)
    applies no pressure bound.
    """
    df = _load_table(csv_path)
    columns = list(df.columns)

    t_col = _find_column(
        columns,
        exact=["T", "Temperature", "T(K)(System_main)"],
        prefixes=["T(", "Temperature("],
        label="temperature (T)",
    )
    s_col = _find_column(
        columns,
        exact=["S", "Entropy", "S(J/g/K)(System_main)"],
        prefixes=["S(", "Entropy("],
        label="entropy (S)",
    )

    p_col = _find_column(
        columns,
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
    in_range = (data[s_col] > KEEP_RANGE[0]) & (data[s_col] < KEEP_RANGE[1])
    if p_max is not None:
        in_range &= data[p_col] < p_max
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

    return {
        "data": data,
        "n": n,
        "t_col": t_col,
        "s_col": s_col,
        "p_col": p_col,
        "bulk_cols": bulk_cols,
        "t": t,
        "s": s,
        "p": p,
        "bulk": bulk,
        "feature_names": feature_names,
        "X": X,
        "model": model,
        "t_pred": t_pred,
        "r2": r2,
        "b0": b0,
        "coefs": coefs,
    }


def fit_T_from_S_and_bulk(csv_path: Path, out_dir: Path, p_max: float | None = P_MAX_DEFAULT) -> None:
    result = compute_T_from_S_and_bulk_regression(csv_path, p_max=p_max)
    t_col = result["t_col"]
    s_col = result["s_col"]
    bulk_cols = result["bulk_cols"]
    n = result["n"]
    feature_names = result["feature_names"]
    b0 = result["b0"]
    coefs = result["coefs"]
    r2 = result["r2"]

    p_bound_str = "none" if p_max is None else f"< {p_max:g} GPa"

    print("--- Temperature Regression Results ---")
    print(f"CSV:        {csv_path}")
    print(f"Rows used:  {n}")
    print(f"P bound:    {p_bound_str}")
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
        f"P bound: {p_bound_str}",
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
            "Path to HeFESTo data: a standalone CSV, a BigMetaTable "
            "(.npy array + header-only .csv) given as the .npy file, the "
            "extensionless base name, or the paired .csv, or an ML-ready "
            "bundle (.tar.gz, or its extensionless base name)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write the coefficient file (defaults to same dir as CSV).",
    )
    parser.add_argument(
        "--p-max",
        type=float,
        default=P_MAX_DEFAULT,
        help=(
            "Drop rows with P >= this value (GPa) before fitting. "
            "Default: no pressure bound."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir is not None else args.csv.parent
    fit_T_from_S_and_bulk(args.csv, out_dir, p_max=args.p_max)


if __name__ == "__main__":
    main()
