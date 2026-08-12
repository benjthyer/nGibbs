#!/usr/bin/env python3
"""
Run a prepared HeFESTo BatchNNNN/SimulationN tree locally with GNU parallel.

Local equivalent of run_sbatches.py / run_many_sbatches.py, for machines
without SLURM (analogous to forward_ensemble() in
src/builder/alphamelts/engine/alphamelts_functions.py, which also shells out
to GNU parallel: https://www.gnu.org/software/parallel/).

Before dispatching any work, every SimulationN directory under --base-dir is
checked for existing output: if it contains any file besides control/ad.in,
it's assumed to already be run and is skipped. This makes it safe to re-run
this script on a tree that a previous run only partially completed.

Every SimulationN directory needs a control file. ad.in is only required
when control's P-T path flag (field 6 of its first line) is -1, meaning
HeFESTo reads the path from ad.in; other control configurations embed the
P-T range directly in control and don't need an ad.in at all.

Requires the HeFESTo runtime environment (Benv activated, LD_LIBRARY_PATH /
LIBRARY_PATH set, as in ClusterHeFESTo.sh) to already be set up in the shell
this script is launched from -- it is inherited by the worker processes.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


SIM_DIR_PATTERN = re.compile(r'^Simulation\d+$', re.IGNORECASE)
INPUT_FILES = {'control', 'ad.in'}

DEFAULT_HEFESTO_CMD = '$HOME/HeFESTo/HeFESToRepository/main'


def find_simulation_dirs(base_dir: Path) -> list[Path]:
    """Find every SimulationN directory anywhere under base_dir."""
    sim_dirs = []
    for root, dirnames, _files in os.walk(base_dir):
        for dirname in dirnames:
            if SIM_DIR_PATTERN.match(dirname):
                sim_dirs.append(Path(root) / dirname)
    return sorted(sim_dirs)


def is_already_run(sim_dir: Path) -> bool:
    """True if sim_dir contains any file besides control/ad.in (i.e. HeFESTo output)."""
    file_names = {p.name for p in sim_dir.iterdir() if p.is_file()}
    return len(file_names - INPUT_FILES) > 0


def control_requires_adin(control_path: Path) -> bool:
    """True if this control file's P-T path flag tells HeFESTo to read ad.in.

    The first line of a HeFESTo control file is a comma-separated run_code;
    field index 6 is -1 when the P-T path is supplied via ad.in, and some
    other value when the P-T range is embedded directly in control (in
    which case no ad.in is needed at all).
    """
    with control_path.open('r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline()
    fields = [v.strip() for v in first_line.split(',')]
    if len(fields) <= 6:
        return False
    try:
        return float(fields[6]) == -1
    except ValueError:
        return False


def partition_simulation_dirs(
    sim_dirs: list[Path],
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split sim_dirs into (to_run, already_run, missing_inputs)."""
    to_run, already_run, missing_inputs = [], [], []
    for sim_dir in sim_dirs:
        control_path = sim_dir / 'control'
        if not control_path.is_file():
            missing_inputs.append(sim_dir)
        elif control_requires_adin(control_path) and not (sim_dir / 'ad.in').is_file():
            missing_inputs.append(sim_dir)
        elif is_already_run(sim_dir):
            already_run.append(sim_dir)
        else:
            to_run.append(sim_dir)
    return to_run, already_run, missing_inputs


def build_runall_script(sim_dirs: list[Path], hefesto_cmd: str) -> str:
    lines = [f'cd "{sim_dir.resolve()}" && {hefesto_cmd}' for sim_dir in sim_dirs]
    return '\n'.join(lines) + '\n'


def run_parallel(
    sim_dirs: list[Path],
    base_dir: Path,
    hefesto_cmd: str,
    jobs: int | None,
    joblog: Path | None,
    dry_run: bool,
) -> int:
    if not sim_dirs:
        print('Nothing to run.')
        return 0

    runall_path = base_dir / 'runall.sh'
    runall_path.write_text(build_runall_script(sim_dirs, hefesto_cmd))
    print(f'Wrote {len(sim_dirs)} job(s) to {runall_path}')

    cmd = ['parallel']
    if jobs is not None:
        cmd += ['-j', str(jobs)]
    if joblog is not None:
        cmd += ['--joblog', str(joblog)]
    cmd += ['--bar']

    if dry_run:
        print(f"[DRY RUN] Would run: {' '.join(cmd)} < {runall_path}")
        return 0

    print(f"Running: {' '.join(cmd)} < {runall_path}")
    with open(runall_path, 'r') as f:
        result = subprocess.run(cmd, stdin=f)
    return result.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run all not-yet-completed HeFESTo simulations under a prepared '
            'BatchNNNN/SimulationN tree locally, using GNU parallel.'
        ),
    )
    parser.add_argument(
        '--base-dir',
        type=Path,
        required=True,
        help='Root directory containing SimulationN folders (searched recursively).',
    )
    parser.add_argument(
        '--jobs', '-j',
        type=int,
        default=None,
        help='Number of concurrent GNU parallel workers (default: GNU parallel default, one per core).',
    )
    parser.add_argument(
        '--hefesto-cmd',
        type=str,
        default=DEFAULT_HEFESTO_CMD,
        help=f'Command used to invoke HeFESTo from within each SimulationN directory (default: {DEFAULT_HEFESTO_CMD}).',
    )
    parser.add_argument(
        '--joblog',
        type=Path,
        default=None,
        help='Optional path for a GNU parallel --joblog file (records per-job exit status/timing).',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be run without actually launching GNU parallel.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if shutil.which('parallel') is None:
        print('Error: GNU parallel not found on PATH.', file=sys.stderr)
        print('Install it: https://www.gnu.org/software/parallel/', file=sys.stderr)
        sys.exit(1)

    base_dir = args.base_dir.resolve()
    if not base_dir.exists():
        print(f'Error: {base_dir} does not exist', file=sys.stderr)
        sys.exit(1)

    sim_dirs = find_simulation_dirs(base_dir)
    if not sim_dirs:
        print(f'No SimulationN directories found under {base_dir}')
        sys.exit(1)

    to_run, already_run, missing_inputs = partition_simulation_dirs(sim_dirs)

    print(f'Found {len(sim_dirs)} SimulationN directories under {base_dir}')
    print(f'  Already run (skipping):        {len(already_run)}')
    print(f'  Missing required inputs (skipping): {len(missing_inputs)}')
    print(f'  To run:                        {len(to_run)}')
    for sim_dir in missing_inputs:
        print(f'  ! Missing control (or ad.in when required), skipping: {sim_dir}')
    print()

    returncode = run_parallel(
        to_run, base_dir, args.hefesto_cmd, args.jobs, args.joblog, args.dry_run,
    )
    if returncode != 0:
        print(f'Warning: GNU parallel exited with code {returncode}', file=sys.stderr)
        sys.exit(returncode)


if __name__ == '__main__':
    main()
