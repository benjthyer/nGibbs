"""
Plot P-T paths from all ad.in files found under a workspace directory.

Each ad.in file is a fixed-width 3-column table written by make_PT_path /
make_PT_path_martian:
    col 0 : P  (GPa)
    col 1 : unused (0.0)
    col 2 : T  (K)

Paths are colored by the surface temperature T0 (T at the lowest P row),
which is a direct proxy for the isentropic potential temperature / entropy.

Usage:
    python scripts/plot_adin_PT_paths.py --root data/HeFESToWorkspace/myBatch
    python scripts/plot_adin_PT_paths.py --root data/HeFESToWorkspace/myBatch \\
        --max-paths 300 --out plot.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def _find_adin_files(root: Path) -> list[Path]:
    return sorted(root.rglob("ad.in"))


def _parse_adin(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (P, T) arrays from an ad.in file, or None if unreadable."""
    try:
        data = np.loadtxt(path, usecols=(0, 2))
    except Exception:
        return None
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
        return None
    return data[:, 0], data[:, 1]


def plot_paths(paths: list[Path], out: Path | None) -> None:
    parsed: list[tuple[np.ndarray, np.ndarray]] = []
    skipped = 0
    for p in paths:
        result = _parse_adin(p)
        if result is None:
            skipped += 1
        else:
            parsed.append(result)

    if not parsed:
        print("No readable ad.in files found.")
        return

    if skipped:
        print(f"Skipped {skipped} unreadable file(s).")

    # Color by T at the maximum-P row (surface temperature proxy)
    t0_values = np.array([pt[1][np.argmax(pt[0])] for pt in parsed])
    norm = mcolors.Normalize(vmin=np.percentile(t0_values, 2), vmax=np.percentile(t0_values, 98))
    cmap = cm.plasma

    fig, ax = plt.subplots(figsize=(9, 8))

    for (P, T), t0 in zip(parsed, t0_values):
        ax.plot(T, P, color=cmap(norm(t0)), alpha=0.30, linewidth=0.7)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Highest T  (K)  —  proxy for entropy")

    ax.set_xlabel("Temperature  (K)")
    ax.set_ylabel("Pressure  (GPa)")
    ax.set_title(f"P–T paths from ad.in files  (N={len(parsed)})\n{paths[0].parent.parent.parent}")
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.grid(alpha=0.2)
    plt.tight_layout()

    if out is not None:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot P-T paths from all ad.in files under a workspace directory."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root workspace directory to search recursively for ad.in files.",
    )
    parser.add_argument(
        "--max-paths",
        type=int,
        default=None,
        help="Maximum number of paths to plot (randomly sampled if exceeded).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Save figure to this path instead of showing it.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Searching for ad.in files under {args.root} ...")
    all_files = _find_adin_files(args.root)
    print(f"Found {len(all_files)} file(s).")

    if not all_files:
        print("Nothing to plot.")
        return

    if args.max_paths is not None and len(all_files) > args.max_paths:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(all_files), size=args.max_paths, replace=False)
        all_files = [all_files[i] for i in sorted(indices)]
        print(f"Subsampled to {args.max_paths} paths.")

    plot_paths(all_files, args.out)


if __name__ == "__main__":
    main()
