"""
CLI wrapper for preparing matched HeFESTo Earth simulation trees.

Draws bulk compositions from GEOROC/PetDB (ultramafic, mafic, felsic) and the
Gard et al. (2019) sediments, plus paired linear mixtures of these, and runs
each composition down three temperature limbs in separate subtrees under
--directory: Adiabat/ (whole-mantle isentrope), Surface/ (conductive lid) and
CMB/ (conductive CMB boundary layer). SimulationN holds the same composition
in all three subtrees.

The composition databases are fixed below; --dataset only switches between
the train and validation splits.
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
	prepare_HeFESTo_Earth,
)


COMPOSITION_DIR = repo_root / 'data' / 'MELTStables' / 'GEOROC'
DATASETS = {
	'train': {
		'georoc': COMPOSITION_DIR / 'GEOROC_PETDB_UNFILTERED_WHOLEROCK_TRAIN.csv',
		'gard': COMPOSITION_DIR / 'Gard2019_sediments_train.csv',
	},
	'validation': {
		'georoc': COMPOSITION_DIR / 'GEOROC_PETDB_UNFILTERED_WHOLEROCK_VALIDATION.csv',
		'gard': COMPOSITION_DIR / 'Gard2019_sediments_valid.csv',
	},
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description='Prepare matched HeFESTo adiabat, surface-lid and CMB SimulationN trees.',
	)
	parser.add_argument(
		'--directory',
		type=Path,
		required=True,
		help='Output directory; Adiabat/, Surface/ and CMB/ subtrees are created under it.',
	)
	parser.add_argument(
		'--dataset',
		choices=sorted(DATASETS),
		required=True,
		help='Which GEOROC/PetDB + Gard composition split to sample from.',
	)
	parser.add_argument(
		'--adiabat-control-path',
		type=Path,
		default = repo_root / 'src' / 'builder' / 'HeFESTo' / 'batch' / 'shallowHeFESTo',
		help='HeFESTo control template for the whole-mantle adiabat limb.',
	)
	parser.add_argument(
		'--surface-control-path',
		type=Path,
		default = repo_root / 'src' / 'builder' / 'HeFESTo' / 'batch' / 'shallowHeFESTo',
		help='HeFESTo control template for the conductive surface-lid limb.',
	)
	parser.add_argument(
		'--cmb-control-path',
		type=Path,
		default = repo_root / 'src' / 'builder' / 'HeFESTo' / 'batch' / 'deepHeFESTo',
		help='HeFESTo control template for the conductive CMB boundary-layer limb.',
	)
	parser.add_argument(
		'--n',
		type=int,
		required=True,
		help='Number of compositions to create (each is run once per limb, 3N simulations in total).',
	)
	parser.add_argument(
		'--real-rocks',
		action='store_true',
		help=(
			'Skip the random Fe/Cr/Si/Mg/Ca/Al mole perturbation applied to each '
			'sampled composition, leaving the mixed rock chemistry unmodified '
			'(aside from Fe speciation to the sampled Fe3+/Fetotal ratio).'
		),
	)
	parser.add_argument(
		'--real-rock-proportion',
		type=float,
		default=0.5,
		help=(
			'Probability in [0, 1] that each composition keeps its database '
			'(or mixed) chemistry unperturbed, skipping the strong random '
			'Fe/Cr/Si/Mg/Ca/Al changes. Overrides --real-rocks when given.'
		),
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	dataset = DATASETS[args.dataset]
	prepare_HeFESTo_Earth(
		directory=args.directory,
		GEOROC_DIR=dataset['georoc'],
		GARD_DIR=dataset['gard'],
		adiabat_control_path=args.adiabat_control_path,
		surface_control_path=args.surface_control_path,
		cmb_control_path=args.cmb_control_path,
		N=args.n,
		real_rocks=args.real_rocks,
		real_rock_proportion=args.real_rock_proportion,
	)


if __name__ == '__main__':
	main()
