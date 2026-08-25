"""
Scan a directory tree for HeFESTo SimulationN directories whose output files
exist but are entirely empty (0 bytes).

A SimulationN directory that HeFESTo actually ran writes 'fort.56', 'fort.61',
'fort.68' and 'fort.99' (the four files import_HeFESTo_components() requires --
see HeFESTo_functions.py). If HeFESTo crashed or was killed immediately after
creating these files but before writing to them, the files exist with size 0.
The importer does not distinguish this from any other malformed simulation: an
empty file raises a pandas parse error inside the per-simulation try/except and
the simulation is folded into "Fault simulation IDs count" with no indication
it was specifically an empty-file problem. This script exists to measure that
specific failure mode directly, since seeing it call out Batch0043/Batch0044 by
name (rather than an aggregate fault count) is what makes it possible to judge
whether this is a one-off or something systemic worth investigating upstream
(e.g. a storage/quota issue, a killed job, a race during launch).

A simulation is classified as:
    not_run       -- none of the checked files exist yet (or dir has only
                      control/ad.in)
    fully_empty   -- at least one checked file exists, and every checked file
                      that exists is 0 bytes
    partly_empty  -- some checked files are 0 bytes, others have content
    missing_some  -- no empty files, but not all four checked files exist
    healthy       -- all four checked files exist and are non-empty

Usage:
    python scripts/check_empty_hefesto_outputs.py --base-dir <root>
    python scripts/check_empty_hefesto_outputs.py --base-dir <root> --list-affected
    python scripts/check_empty_hefesto_outputs.py --base-dir <root> \
        --files fort.56 fort.61 fort.68 fort.99 fort.29 fort.42 fort.59 qout
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_SIM_DIR_PATTERN = re.compile(r'^simulation\d+$', flags=re.IGNORECASE)

DEFAULT_FILES = ('fort.56', 'fort.61', 'fort.68', 'fort.99')


def find_simulation_dirs(base_dir: Path) -> list[Path]:
    sim_dirs = []
    for path in base_dir.rglob('*'):
        if path.is_dir() and _SIM_DIR_PATTERN.match(path.name):
            sim_dirs.append(path)
    return sorted(sim_dirs)


def classify_simulation(sim_dir: Path, checked_files: tuple[str, ...]) -> tuple[str, list[str], list[str]]:
    """Returns (category, empty_file_names, missing_file_names)."""
    empty, missing, present_nonempty = [], [], []
    for name in checked_files:
        target = sim_dir / name
        if not target.exists():
            missing.append(name)
            continue
        if target.stat().st_size == 0:
            empty.append(name)
        else:
            present_nonempty.append(name)

    if not empty and not present_nonempty:
        return 'not_run', empty, missing
    if empty and not present_nonempty:
        return 'fully_empty', empty, missing
    if empty and present_nonempty:
        return 'partly_empty', empty, missing
    if missing:
        return 'missing_some', empty, missing
    return 'healthy', empty, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Count HeFESTo SimulationN directories whose output files exist but are empty.'
    )
    parser.add_argument('--base-dir', type=Path, required=True,
                        help='Root directory to search recursively for SimulationN folders.')
    parser.add_argument('--files', type=str, nargs='+', default=list(DEFAULT_FILES),
                        help=f'Output filenames to check (default: {" ".join(DEFAULT_FILES)}).')
    parser.add_argument('--list-affected', action='store_true',
                        help='Print the full path of every fully_empty and partly_empty simulation.')
    parser.add_argument('--top', type=int, default=20,
                        help='Number of worst-offending workspace directories to show (default: 20).')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    checked_files = tuple(args.files)

    if not base_dir.is_dir():
        print(f'Error: {base_dir} is not a directory', file=sys.stderr)
        return 2

    sim_dirs = find_simulation_dirs(base_dir)
    if not sim_dirs:
        print(f'No SimulationN directories found under {base_dir}')
        return 0

    counts: Counter[str] = Counter()
    affected: list[tuple[Path, str, list[str]]] = []
    # Grouped by the SimulationN dir's parent -- the workspace/Batch directory --
    # so problems concentrated in a few batches (like the one that prompted this
    # script) are visible without scrolling through every individual simulation.
    per_workspace_fully_empty: Counter[Path] = Counter()
    per_workspace_total: Counter[Path] = Counter()

    for sim_dir in sim_dirs:
        category, empty, missing = classify_simulation(sim_dir, checked_files)
        counts[category] += 1
        workspace = sim_dir.parent
        per_workspace_total[workspace] += 1
        if category == 'fully_empty':
            per_workspace_fully_empty[workspace] += 1
        if category in ('fully_empty', 'partly_empty'):
            affected.append((sim_dir, category, empty))

    total = len(sim_dirs)
    print(f'Scanned {total} SimulationN directories under {base_dir}')
    print(f'  Checked files: {", ".join(checked_files)}')
    print()
    print('Breakdown:')
    for category in ('healthy', 'fully_empty', 'partly_empty', 'missing_some', 'not_run'):
        n = counts.get(category, 0)
        pct = 100.0 * n / total if total else 0.0
        print(f'  {category:<14} {n:>8}  ({pct:5.1f}%)')

    n_bad = counts.get('fully_empty', 0) + counts.get('partly_empty', 0)
    print()
    print(f'Total simulations with at least one empty output file: {n_bad} '
          f'({100.0 * n_bad / total:.1f}% of scanned)')

    if per_workspace_fully_empty:
        print()
        print(f'Worst-affected workspace directories (by fully_empty count), top {args.top}:')
        for workspace, n_empty in per_workspace_fully_empty.most_common(args.top):
            n_total = per_workspace_total[workspace]
            print(f'  {n_empty:>6} / {n_total:<6} {workspace}')

    if args.list_affected and affected:
        print()
        print('Affected simulations:')
        for sim_dir, category, empty in affected:
            print(f'  [{category}] {sim_dir}  (empty: {", ".join(empty)})')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
