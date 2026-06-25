"""
CLI wrapper for plotting bulk compositions from HeFESTo control files.

Walks a workspace directory tree (BatchNNNN/SimulationN or flat SimulationN),
parses the bulk composition from each ``control`` file, and produces a
diagnostic figure with oxide scatter plots vs MgO and element-mole histograms.

Usage
-----
    python scripts/plot_bulk_compositions.py --workspace-dir /path/to/workspace
    python scripts/plot_bulk_compositions.py --workspace-dir /path/to/workspace --out fig.png
    python scripts/plot_bulk_compositions.py --workspace-dir /path/to/workspace --max-sims 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
src_root = repo_root / 'src'
for p in (str(repo_root), str(src_root)):
    if p not in sys.path:
        sys.path.insert(0, p)

from builder.HeFESTo.HeFESTo_functions import plot_bulk_compositions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot bulk compositions parsed from HeFESTo control files.',
    )
    parser.add_argument(
        '--workspace-dir',
        type=Path,
        required=True,
        help='Root directory containing BatchNNNN/SimulationN or flat SimulationN subdirectories.',
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=None,
        help='Save the figure to this path instead of displaying it interactively.',
    )
    parser.add_argument(
        '--max-sims',
        type=int,
        default=None,
        help='Maximum number of simulations to read (useful for quick previews).',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = plot_bulk_compositions(
        workspace_dir=str(args.workspace_dir),
        out_path=str(args.out) if args.out is not None else None,
        max_sims=args.max_sims,
    )
    print(f'Parsed {len(df)} compositions.')


if __name__ == '__main__':
    main()
