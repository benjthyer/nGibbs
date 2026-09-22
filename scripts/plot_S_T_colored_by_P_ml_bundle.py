"""Plot entropy against temperature for one or more ML bundles.

Each point is colored by pressure. At most ``--max-points`` rows are sampled
from each bundle so large bundles remain practical to inspect.

Usage:
    python scripts/plot_S_T_colored_by_P_ml_bundle.py \
        --bundle data/MLready/120/120SedIgClosed_NoCr_NPS_Test \
        --output /tmp/S_T_by_P.png

For multiple bundles, omit ``--output`` or provide an output directory; each
bundle gets a separate PNG named after the bundle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

repo_src = str(Path(__file__).resolve().parents[1] / "src")
if repo_src not in sys.path:
    sys.path.insert(0, repo_src)

from ngibbs.utils.file_utils import load_ml_bundle


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


def _load_plot_data(bundle_path: Path, max_points: int, seed: int) -> dict:
    bundle_path = _resolve_bundle_path(bundle_path)
    bundle = load_ml_bundle(bundle_path, arrays=("features", "free_outputs"))
    if bundle.free_outputs is None:
        raise ValueError(f"Bundle has no free_outputs.npy: {bundle_path}")

    indexer = bundle.ml_indexer
    feature_names = [str(name) for name in indexer.featureNames]
    element_names = [str(name) for name in indexer.Elkeys]
    names = feature_names + element_names
    if bundle.features.ndim != 2 or bundle.features.shape[1] != len(names):
        raise ValueError(
            "features.npy column count does not match featureNames + Elkeys: "
            f"got {bundle.features.shape[1:]}, expected {len(names)} columns"
        )

    output_names = [str(name) for name in (indexer.free_outputs or [])]
    if len(output_names) != bundle.free_outputs.shape[1]:
        raise ValueError(
            "free_outputs metadata does not match free_outputs.npy: "
            f"{len(output_names)} names for {bundle.free_outputs.shape[1]} columns"
        )

    s_name = _find_name(
        feature_names,
        exact=["S", "Entropy", "S(System_main) / mass(System_main)"],
        prefixes=["S(", "Entropy("],
        label="entropy (S)",
    )
    p_name = _find_name(
        feature_names,
        exact=["P", "Pressure", "Pressure(System_main)", "P(GPa)"],
        prefixes=["P(", "Pressure("],
        label="pressure (P)",
    )
    t_name = _find_name(
        output_names,
        exact=["T", "Temperature", "Temperature(System_main)"],
        prefixes=["T(", "Temperature("],
        label="temperature (T)",
    )

    feature_indices = {name: index for index, name in enumerate(names)}
    s = np.asarray(bundle.features[:, feature_indices[s_name]], dtype=float)
    p = np.asarray(bundle.features[:, feature_indices[p_name]], dtype=float)
    t = np.asarray(bundle.free_outputs[:, output_names.index(t_name)], dtype=float)
    valid = np.isfinite(s) & np.isfinite(p) & np.isfinite(t)
    valid_indices = np.flatnonzero(valid)
    if max_points and len(valid_indices) > max_points:
        rng = np.random.default_rng(seed)
        valid_indices = rng.choice(valid_indices, size=max_points, replace=False)

    return {
        "bundle_path": bundle_path,
        "s": s[valid_indices],
        "p": p[valid_indices],
        "t": t[valid_indices],
        "s_name": s_name,
        "p_name": p_name,
        "t_name": t_name,
        "total_valid": int(valid.sum()),
    }


def _default_output_path(bundle_path: Path) -> Path:
    stem = bundle_path.name
    if stem.endswith(".tar.gz"):
        stem = stem[:-len(".tar.gz")]
    return bundle_path.parent / f"{stem}_S_vs_T_colored_by_P.png"


def plot_bundle(data: dict, output_path: Path, point_size: float, alpha: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        data["s"],
        data["t"],
        c=data["p"],
        cmap="viridis",
        s=point_size,
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(data["p_name"])
    ax.set_xlabel(data["s_name"])
    ax.set_ylabel(data["t_name"])
    ax.set_title(f"{data['bundle_path'].name}: S vs T colored by P")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(
        f"Saved {output_path} ({len(data['s']):,} plotted / "
        f"{data['total_valid']:,} finite rows)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot S against T, colored by P, for ML bundles."
    )
    parser.add_argument(
        "--bundle", type=Path, nargs="+", required=True,
        help="One or more ML bundle paths, with or without .tar.gz.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="PNG path for one bundle, or output directory for multiple bundles.",
    )
    parser.add_argument(
        "--max-points", type=int, default=200_000,
        help="Maximum sampled points per bundle (default: 200000).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed used when sampling rows (default: 0).",
    )
    parser.add_argument(
        "--point-size", type=float, default=2.0,
        help="Scatter point area in points squared (default: 2).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.35,
        help="Point opacity (default: 0.35).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_points < 0:
        raise ValueError("--max-points must be non-negative")
    if not 0 < args.alpha <= 1:
        raise ValueError("--alpha must be greater than 0 and at most 1")
    if args.point_size <= 0:
        raise ValueError("--point-size must be greater than 0")
    if args.output is not None and len(args.bundle) > 1 and args.output.suffix:
        raise ValueError("For multiple bundles, --output must be a directory")

    for bundle_path in args.bundle:
        resolved_path = _resolve_bundle_path(bundle_path)
        data = _load_plot_data(resolved_path, args.max_points, args.seed)
        if args.output is None:
            output_path = _default_output_path(resolved_path)
        elif len(args.bundle) == 1:
            output_path = args.output
        else:
            output_path = args.output / _default_output_path(resolved_path).name
        plot_bundle(data, output_path, args.point_size, args.alpha)


if __name__ == "__main__":
    main()