"""
CLI wrapper for preparing HeFESTo isobaric runs that fill the CMB-region gaps.

The whole-mantle adiabats (prepare_hefesto_tree_fulladiabat.py) run to ~140 GPa
but the conductive CMB boundary-layer runs (prepare_hefesto_tree_cmb.py) stop
at ~135-136 GPa, leaving everything hotter than the adiabats between ~136 and
140 GPa unsampled, plus a thin dead zone between the adiabat band and the
coolest conductive lines near 135 GPa. Each simulation here is one isobar at
P ~ U(134, 140) GPa sweeping T from the adiabat (sampled entropy) to ~5000 K,
written as a native HeFESTo P-T grid run code (no ad.in file).
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
	prepare_HeFESTo_tree_cmb_isobars,
	CMB_ISOBAR_P_RANGE_GPA,
	CMB_ISOBAR_T_MAX_RANGE_K,
	CMB_ISOBAR_DT_K,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description='Prepare a HeFESTo SimulationN tree of isobars filling the CMB-region gaps.',
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
		help='Path to the HeFESTo control template file (a deep-mantle template, e.g. deepHeFESTo).',
	)
	parser.add_argument(
		'--n',
		type=int,
		required=True,
		help='Number of simulations (isobars) to create.',
	)
	parser.add_argument(
		'--real-rocks',
		action='store_true',
		help=(
			'Skip the random Fe/Cr/Si/Mg/Ca/Al mole perturbation applied to each '
			'sampled composition, leaving the mixed GEOROC/PetDB rock chemistry '
			'unmodified (aside from Fe speciation to the sampled Fe3+/Fetotal ratio).'
		),
	)
	parser.add_argument(
		'--real-rock-proportion',
		type=float,
		default=None,
		help=(
			'Probability in [0, 1] that each simulation keeps its GEOROC/PetDB '
			'(or mixed) chemistry unperturbed, skipping the strong random '
			'Fe/Cr/Si/Mg/Ca/Al changes. Overrides --real-rocks when given.'
		),
	)
	parser.add_argument(
		'--p-range', type=float, nargs=2, default=list(CMB_ISOBAR_P_RANGE_GPA),
		metavar=('P_MIN', 'P_MAX'),
		help='Pressure window (GPa) for the isobars (default: %(default)s).',
	)
	parser.add_argument(
		'--t-max-range', type=float, nargs=2, default=list(CMB_ISOBAR_T_MAX_RANGE_K),
		metavar=('T_LO', 'T_HI'),
		help='Range (K) for the hot end of each isobar (default: %(default)s).',
	)
	parser.add_argument(
		'--dT', type=float, default=CMB_ISOBAR_DT_K,
		help='Temperature step (K) along each isobar (default: %(default)s).',
	)
	parser.add_argument(
		'--adiabat-overlap', type=float, default=0.0,
		help='Start each sweep this many K below its sampled adiabat temperature (default: %(default)s).',
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	prepare_HeFESTo_tree_cmb_isobars(
		directory=args.directory,
		GEOROC_DIR=args.georoc_dir,
		control_path=args.control_path,
		N=args.n,
		real_rocks=args.real_rocks,
		real_rock_proportion=args.real_rock_proportion,
		P_range=tuple(args.p_range),
		T_max_range=tuple(args.t_max_range),
		dT=args.dT,
		adiabat_overlap_K=args.adiabat_overlap,
	)


if __name__ == '__main__':
	main()
