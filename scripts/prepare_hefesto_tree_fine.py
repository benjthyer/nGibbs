"""
CLI wrapper for preparing a HeFESTo simulation tree from phase-boundary rows.

Adds a --deep mode to the first-pass behaviour. Deep mode traverses a *single* bounded
scan at fine resolution instead of laying down a P-T grid, and it does not accept grid
bounds: an isotherm pair must share T and vary in P, an isobar pair must share P and
vary in T. Pairs that vary in both came from the first pass and are skipped with a
count rather than silently becoming grids.

Choosing --deep-dt
------------------
Do not pick it independently of --deep-dp. A boundary with Clapeyron slope dP/dT is
crossed over dT = dP / |dP/dT| in temperature, so matching resolution across the
boundary means dT = dP / |dP/dT|. At 0.01 GPa and the post-spinel slope of 1.71 MPa/K
that is ~5.8 K; for post-perovskite near 10 MPa/K it is ~1 K. A flat 1 K therefore
oversamples the transition zone about sixfold, and 0.1 K oversamples everywhere by
another order of magnitude. Leave --deep-dt unset and pass --clapeyron to have it
computed, or set it explicitly if you know what you want.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
src_root = repo_root / 'src'
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from builder.HeFESTo.HeFESTo_functions import (  # noqa: E402
    prepare_HeFESTo_tree_from_phase_changes,
)
from builder.HeFESTo.HeFESTo_deep_sampling import (  # noqa: E402
    prepare_deep_tree, deep_temperature_step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Prepare a batched HeFESTo tree from phase-boundary rows.',
    )
    parser.add_argument(
        '--directory',
        type=Path,
        required=True,
        help='Base output directory where Batch#### folders will be created.',
    )
    parser.add_argument(
        '--phase-path',
        type=Path,
        required=True,
        help='Path to the phase-boundary CSV with paired rows.',
    )
    parser.add_argument(
        '--control-dir',
        type=Path,
        required=True,
        help=(
            'Path to control template directory (or a template file path). '
            'If a directory is given, shallowHeFESTo/deepHeFESTo is preferred.'
        ),
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help=(
            'Limit the number of Simulation directories created by randomly '
            'sampling this many phase-path condition pairs from the CSV, '
            'instead of using every pair. Pairs are read directly off disk, '
            'so the rest of the CSV is never loaded into memory.'
        ),
    )
    parser.add_argument(
        '--deep',
        action='store_true',
        help=(
            'Fine-resolution traversal of a single bounded scan instead of a P-T '
            'grid. Requires bounds produced by --deep-phase-change-dataname.'
        ),
    )
    parser.add_argument(
        '--deep-axis',
        choices=('isotherm', 'isobar'),
        default='isotherm',
        help='Scan direction in --deep mode.',
    )
    parser.add_argument(
        '--deep-dp',
        type=float,
        default=0.01,
        help='Pressure step in GPa for --deep-axis isotherm (default 0.01).',
    )
    parser.add_argument(
        '--deep-dt',
        type=float,
        default=None,
        help=(
            'Temperature step in K for --deep-axis isobar. Default is computed as '
            'deep-dp / |Clapeyron|, which matches resolution across the boundary '
            'rather than picking a number blind.'
        ),
    )
    parser.add_argument(
        '--clapeyron',
        type=float,
        default=None,
        help=(
            'Clapeyron slope magnitude in MPa/K used to derive --deep-dt. Defaults to '
            '1.71 (post-spinel). Use ~10 for post-perovskite, ~2.5 for the 410.'
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.deep:
        dT = args.deep_dt
        if dT is None and args.deep_axis == 'isobar':
            dT = deep_temperature_step(args.clapeyron, args.deep_dp)
            print(f'[deep] dT = {dT:.2f} K from dP = {args.deep_dp} GPa and '
                  f'|dP/dT| = {args.clapeyron or 1.71} MPa/K')
        written = prepare_deep_tree(
            directory=args.directory,
            phase_path=args.phase_path,
            CONTROL_DIR=args.control_dir,
            axis=args.deep_axis,
            dP=args.deep_dp,
            dT=dT,
            clapeyron_MPa_per_K=args.clapeyron,
            limit=args.limit,
        )
        print(f'[deep] wrote {written} {args.deep_axis} simulations')
        return

    prepare_HeFESTo_tree_from_phase_changes(
        directory=args.directory,
        phase_path=args.phase_path,
        CONTROL_DIR=args.control_dir,
        limit=args.limit,
    )


if __name__ == '__main__':
    main()
