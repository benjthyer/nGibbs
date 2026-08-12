"""
CLI wrapper for preparing a HeFESTo surface conductive-lid simulation tree.

This script copies the control template into SimulationN folders, rewrites the
control file per simulation, and writes each simulation's ad.in path. Each
ad.in covers only the conductive lid itself: from the surface (300 K anchor)
down to a randomly sampled lithosphere-asthenosphere transition (0-250 km
deep), whose temperature is predicted with the Earth_Adiabats NN emulator.
This is meant to complement a separate whole-mantle adiabat dataset, not
duplicate it.
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
	prepare_HeFESTo_tree_surface,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description='Prepare a HeFESTo surface-lid SimulationN tree and per-run ad.in files.',
	)
	parser.add_argument(
		'--directory',
		type=Path,
		required=True,
		help='Output directory where SimulationN folders will be created.',
	)
	parser.add_argument(
		'--georoc-dir',
		type=Path,
		required=True,
		help='Path to the GEOROC CSV file used to sample compositions.',
	)
	parser.add_argument(
		'--control-path',
		type=Path,
		required=True,
		help='Path to the HeFESTo control template file (a shallow-mantle template, e.g. shallowHeFESTo).',
	)
	parser.add_argument(
		'--n',
		type=int,
		required=True,
		help='Number of simulations to create.',
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	prepare_HeFESTo_tree_surface(
		directory=args.directory,
		GEOROC_DIR=args.georoc_dir,
		control_path=args.control_path,
		N=args.n,
	)


if __name__ == '__main__':
	main()
