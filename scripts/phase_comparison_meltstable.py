"""CLI wrapper: MELTStable phase-abundance comparison for the HeFESTo emulator.

Thin front-end over ``ngibbs.deployment_tests.meltstable_comparison`` (the
packaged, ``builder``-free implementation also reached by
``HeFESToEmulatorCPU.test()``).  Draws stacked phase-abundance diagrams
(GT vs emulator, isothermal and isentropic pathways) for a directory of
MELTStable adiabat CSVs and writes a per-phase phase-fraction error table.
Pressure spans where the CSV contains a phase the emulator cannot represent
are hatched on the GT panel and listed.

    python scripts/phase_comparison_meltstable.py
    python scripts/phase_comparison_meltstable.py --table-dir some/dir --tables DMMadiabat
    python scripts/phase_comparison_meltstable.py --heavy --out-dir plots/meltstable
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / 'src'), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _plot_output import plots_dir  # noqa: E402
from ngibbs.deployment_tests.meltstable_comparison import (  # noqa: E402
    run_meltstable_phase_comparison, DEFAULT_TABLE_DIR,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--table-dir', type=str, default=str(DEFAULT_TABLE_DIR),
                        help='Directory of MELTStable CSVs (default: the shipped HeFESToAdiabatStandards)')
    parser.add_argument('--tables', type=str, nargs='+', default=None,
                        help='Explicit CSV paths or bare stems. Default: every *.csv in --table-dir.')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Directory for figures + stats CSV (default: <repo>/plots/meltstable)')
    parser.add_argument('--heavy', action='store_true',
                        help='Use HeFESToHeavyEmulatorCPU instead of the light model')
    args = parser.parse_args()

    from ngibbs.engine.models import HeFESToEmulatorCPU, HeFESToHeavyEmulatorCPU
    api = HeFESToHeavyEmulatorCPU if args.heavy else HeFESToEmulatorCPU
    out_dir = args.out_dir if args.out_dir is not None else plots_dir('meltstable')

    out = run_meltstable_phase_comparison(
        api, args.table_dir, out_dir, tables=args.tables,
    )
    if out['unrepresented_phases']:
        print('\nUnrepresented phases present in the tables:')
        for tbl, phases in out['unrepresented_phases'].items():
            for ph, info in sorted(phases.items(), key=lambda kv: kv[1]['peak_fraction'], reverse=True):
                print(f"  {tbl}: {ph}  peak={info['peak_fraction'] * 100:.3f}% at P={info['P_at_peak']:.1f} GPa")
    else:
        print('\nNo unrepresented phases — every phase in the tables is in the emulator.')

    print(f'\nWrote {out["stats_path"]}')
    for fig in out['figures'].values():
        print(f'Wrote {fig}')


if __name__ == '__main__':
    main()
