"""Superseded: the HeFESTo standard-rock phase and property comparisons are now
part of ``evaluate_workspace`` (``ngibbs.deployment_tests.workspace_benchmark``).

This wrapper runs it on the shipped standard workspace, or on --workspace:

    python scripts/phase_comparison_meltstable.py
    python scripts/phase_comparison_meltstable.py --workspace path/to/hefesto_runs --mars
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workspace', default=None,
                        help='raw HeFESTo runs (default: the model\'s standard workspace)')
    parser.add_argument('--out-dir', default=None, help='output directory (default: ./ngibbs_deployment_test/...)')
    parser.add_argument('--mars', action='store_true', help='use HeFESToMarsEmulatorCPU')
    args = parser.parse_args()

    from ngibbs import HeFESToEmulatorCPU, HeFESToMarsEmulatorCPU
    api = HeFESToMarsEmulatorCPU if args.mars else HeFESToEmulatorCPU
    r = api.evaluate_workspace(args.workspace, args.out_dir)
    print(f"\nWrote {r['output_dir']}")
    for name in ('property_errors', 'phase_errors'):
        if not r[name].empty:
            print(f'\n{name}:\n' + r[name].to_string(na_rep='--'))


if __name__ == '__main__':
    main()
