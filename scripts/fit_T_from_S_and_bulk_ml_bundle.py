"""Fit temperature from an ML bundle's entropy, pressure, and bulk elements.

The bundle must contain ``features.npy``, ``free_outputs.npy``, and the
``ml_indexer`` metadata. Feature columns are named by ``featureNames`` followed
by ``Elkeys``; the temperature target is selected from the free-output names.

Usage:
    python scripts/fit_T_from_S_and_bulk_ml_bundle.py \
        --bundle data/MLready/120/120SedIgClosed_NoCr_NPS_Test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

repo_src = str(Path(__file__).resolve().parents[1] / "src")
if repo_src not in sys.path:
    sys.path.insert(0, repo_src)

from ngibbs.utils.file_utils import load_ml_bundle

KEEP_RANGE = [0,2000] #[1.75, 3.5]
P_MAX_DEFAULT = None


def _resolve_bundle_path(path: Path) -> Path:
    if path.exists():
        return path
    if path.suffix == "":
        archive_path = path.with_name(path.name + ".tar.gz")
        if archive_path.exists():
            return archive_path
    raise FileNotFoundError(f"ML bundle not found: {path}")


def _find_name(names: list[str], *, exact: list[str], prefixes: list[str], label: str) -> str:
    for name in exact:
        if name in names:
            return name
    for prefix in prefixes:
        matches = [name for name in names if name.startswith(prefix)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous {label} name for prefix '{prefix}': {matches}")
    raise ValueError(
        f"Could not find {label} name. Checked exact={exact}, prefixes={prefixes}."
    )


def compute_T_from_S_and_bulk_regression(
    bundle_path: Path, p_max: float | None = P_MAX_DEFAULT
) -> dict:
    """Load, filter, fit, and return the ML-bundle regression details."""
    bundle_path = _resolve_bundle_path(bundle_path)
    bundle = load_ml_bundle(bundle_path, arrays=("features", "free_outputs"))
    if bundle.free_outputs is None:
        raise ValueError("The ML bundle does not contain free_outputs.npy")

    indexer = bundle.ml_indexer
    feature_names = [str(name) for name in indexer.featureNames]
    bulk_names = [str(name) for name in indexer.Elkeys]
    expected_names = feature_names + bulk_names
    if bundle.features.ndim != 2 or bundle.features.shape[1] != len(expected_names):
        raise ValueError(
            "features.npy column count does not match featureNames + Elkeys: "
            f"got {bundle.features.shape[1:]}, expected {len(expected_names)} columns"
        )
    if bundle.free_outputs.ndim != 2 or bundle.free_outputs.shape[0] != bundle.features.shape[0]:
        raise ValueError(
            "free_outputs.npy must be a 2-D row-aligned array: "
            f"features rows={bundle.features.shape[0]}, free_outputs shape={bundle.free_outputs.shape}"
        )

    output_names = [str(name) for name in (indexer.free_outputs or [])]
    if len(output_names) != bundle.free_outputs.shape[1]:
        raise ValueError(
            "free_outputs metadata does not match free_outputs.npy: "
            f"{len(output_names)} names for {bundle.free_outputs.shape[1]} columns"
        )

    p_name = _find_name(
        feature_names,
        exact=["P", "Pressure", "Pressure(System_main)", "P(GPa)"],
        prefixes=["P(", "Pressure("],
        label="pressure (P)",
    )
    s_name = _find_name(
        feature_names,
        exact=["S", "Entropy", "S(System_main) / mass(System_main)"],
        prefixes=["S(", "Entropy("],
        label="entropy (S)",
    )
    t_name = _find_name(
        output_names,
        exact=["T", "Temperature", "Temperature(System_main)"],
        prefixes=["T(", "Temperature("],
        label="temperature (T)",
    )

    feature_indices = {name: index for index, name in enumerate(expected_names)}
    p = np.asarray(bundle.features[:, feature_indices[p_name]], dtype=float)
    s = np.asarray(bundle.features[:, feature_indices[s_name]], dtype=float)
    bulk = [np.asarray(bundle.features[:, feature_indices[name]], dtype=float) for name in bulk_names]
    t = np.asarray(bundle.free_outputs[:, output_names.index(t_name)], dtype=float)

    valid = np.isfinite(t) & np.isfinite(s) & np.isfinite(p)
    valid &= np.all(np.isfinite(np.column_stack(bulk)), axis=1)
    valid &= (s > KEEP_RANGE[0]) & (s < KEEP_RANGE[1])
    if p_max is not None:
        valid &= p < p_max

    s = s[valid]
    p = p[valid]
    t = t[valid]
    bulk = [values[valid] for values in bulk]

    # Bulk elements whose symbol is literally "S" or "P" (e.g. phosphorus)
    # collide with the reserved entropy/pressure term names below: both would
    # be written to the coefficient file as "b_P"/"b_P^2", and a downstream
    # dict-based parser can only keep one of the two (silently discarding the
    # other, or worse, applying the wrong one to the wrong feature). Exclude
    # them from the compositional fit rather than emit an ambiguous file.
    reserved_names = {"S", "P"}
    excluded_bulk_names = [name for name in bulk_names if name in reserved_names]
    if excluded_bulk_names:
        print(
            f"Warning: excluding bulk element(s) {excluded_bulk_names} from the "
            "compositional regression - their symbol collides with the reserved "
            "'S' (entropy) / 'P' (pressure) term names, which the saved "
            "coefficient file cannot disambiguate. Their contribution to T is "
            "not modeled."
        )
    fit_bulk_names = [name for name in bulk_names if name not in reserved_names]
    fit_bulk = [values for name, values in zip(bulk_names, bulk) if name not in reserved_names]

    input_values = [s, p] + fit_bulk
    input_names = ["S", "P"] + fit_bulk_names
    input_scales = []
    for name, values in zip(input_names, input_values):
        scale = float(np.max(np.abs(values)))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Cannot scale {name}: maximum absolute value is {scale}")
        input_scales.append(scale)

    scaled_s = s / input_scales[0]
    scaled_p = p / input_scales[1]
    scaled_bulk = [values / scale for values, scale in zip(fit_bulk, input_scales[2:])]
    feature_parts = [scaled_s, scaled_s**2, scaled_p, scaled_p**2]
    regression_names = ["S", "S^2", "P", "P^2"]
    for name, values in zip(fit_bulk_names, scaled_bulk):
        feature_parts.extend([values, values**2])
        regression_names.extend([name, f"{name}^2"])

    X = np.column_stack(feature_parts)
    model = LinearRegression(fit_intercept=True)
    model.fit(X, t)
    t_pred = model.predict(X)
    r2 = r2_score(t, t_pred)

    feature_scales = [
        input_scales[0],
        input_scales[0] ** 2,
        input_scales[1],
        input_scales[1] ** 2,
    ]
    for scale in input_scales[2:]:
        feature_scales.extend([scale, scale**2])
    raw_coefs = [
        float(coefficient / scale)
        for coefficient, scale in zip(model.coef_, feature_scales)
    ]

    return {
        "bundle_path": bundle_path,
        "n": len(t),
        "t_name": t_name,
        "s_name": s_name,
        "p_name": p_name,
        "bulk_names": fit_bulk_names,
        "feature_names": regression_names,
        "model": model,
        "t": t,
        "t_pred": t_pred,
        "r2": r2,
        "b0": float(model.intercept_),
        "coefs": raw_coefs,
        "input_names": input_names,
        "input_scales": input_scales,
    }


def fit_T_from_S_and_bulk(bundle_path: Path, out_dir: Path, p_max: float | None = P_MAX_DEFAULT) -> None:
    result = compute_T_from_S_and_bulk_regression(bundle_path, p_max=p_max)
    names = result["feature_names"]
    terms = " + ".join(f"b_{name}*{name}" for name in names)
    r2 = result["r2"]
    p_bound_str = "none" if p_max is None else f"< {p_max:g} GPa"

    print("--- Temperature Regression Results ---")
    print(f"Bundle:     {result['bundle_path']}")
    print(f"Rows used:  {result['n']}")
    print(f"P bound:    {p_bound_str}")
    print(f"T output:   '{result['t_name']}'")
    print(f"S feature:  '{result['s_name']}'")
    print(f"P feature:  '{result['p_name']}'")
    print(f"Bulk keys:  {result['bulk_names']}")
    print("Scaling factors (fit used value / factor):")
    for name, scale in zip(result["input_names"], result["input_scales"]):
        print(f"  {name:<8} = {scale:.12g}")
    print("")
    print(f"Model: T = b0 + {terms}")
    print(f"  b0  (intercept) = {result['b0']:.12g}")
    for name, coefficient in zip(names, result["coefs"]):
        print(f"  b_{name:<10} = {coefficient:.12g}")
    print(f"R^2 = {r2:.12g}")

    r2_str = f"{r2:.5f}".replace(".", "p")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"T_from_S_X_coefs_R2_{r2_str}.txt"
    lines = [
        "Temperature regression: T = f(S, bulk composition elements)",
        f"Bundle: {result['bundle_path']}",
        f"Rows used: {result['n']}",
        f"P bound: {p_bound_str}",
        f"R^2: {r2:.12g}",
        "",
        "Input scaling used during fit (stored coefficients are unscaled):",
    ]
    lines.extend(
        f"  {name:<8} = {scale:.12g}"
        for name, scale in zip(result["input_names"], result["input_scales"])
    )
    lines.extend([
        "",
        f"Model: T = b0 + {terms}",
        "",
        "Coefficients:",
        f"  b0  (intercept) = {result['b0']:.12g}",
    ])
    lines.extend(
        f"  b_{name:<10} = {coefficient:.12g}"
        for name, coefficient in zip(names, result["coefs"])
    )
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nCoefficients saved to: {out_path}")

    plot_path = out_dir / f"T_from_S_X_pred_R2_{r2_str}.png"
    t = result["t"]
    t_pred = result["t_pred"]
    limits = [min(t.min(), t_pred.min()), max(t.max(), t_pred.max())]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(limits, limits, color="black", linewidth=1.0, zorder=1, label="1:1")
    ax.scatter(
        t,
        t_pred,
        s=2,
        alpha=0.2,
        linewidths=0,
        rasterized=True,
        zorder=2,
        label="Regression samples",
    )
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("T (K)")
    ax.set_ylabel("T_pred(P, S, bulk) (K)")
    ax.set_title(f"Temperature regression parity (R² = {r2:.5f})")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Parity plot saved to: {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit T from entropy, pressure, and bulk elements in an ML bundle."
    )
    parser.add_argument(
        "--bundle", type=Path, required=True,
        help="ML bundle path, with or without the .tar.gz suffix.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Directory for coefficients (defaults to the bundle directory).",
    )
    parser.add_argument(
        "--p-max", type=float, default=P_MAX_DEFAULT,
        help="Drop rows with P >= this value (GPa). Default: no pressure bound.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_path = _resolve_bundle_path(args.bundle)
    out_dir = args.out_dir if args.out_dir is not None else bundle_path.parent
    fit_T_from_S_and_bulk(bundle_path, out_dir, p_max=args.p_max)


if __name__ == "__main__":
    main()