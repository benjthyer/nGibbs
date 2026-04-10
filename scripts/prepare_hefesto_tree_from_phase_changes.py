"""
CLI wrapper for preparing a HeFESTo phase-boundary simulation tree.

This script creates Batch####/SimulationN directories from a phase-boundary
CSV, copies the control template into each simulation folder, and writes bulk
composition values into each control file.
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_HeFESTo_tree_from_phase_changes(
        directory=args.directory,
        phase_path=args.phase_path,
        CONTROL_DIR=args.control_dir,
    )


if __name__ == '__main__':
    main()
