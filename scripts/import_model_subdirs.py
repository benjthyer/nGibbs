"""
Import HeFESTo workspaces from a root path.

The script recursively searches for the lowest-level workspace directories
that directly contain model_XXXXXX folders.

For each discovered workspace directory:
1) Run import_HeFESTo_components() on that workspace.

This variant performs no cleanup or deletion.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from builder.HeFESTo.HeFESTo_functions import import_HeFESTo_components  # noqa: E402
from builder.indexer import DatasetIndexer, generate_column_headers_hefesto  # noqa: E402
from module.config.constants import COMPOSITIONAL_COMPONENTS_IN_PHASES_HEFESTO  # noqa: E402


_SIM_DIR_PATTERN = re.compile(r'^model_\d{6}$', flags=re.IGNORECASE)


def _is_model_dir(entry: Path) -> bool:
    return entry.is_dir() and _SIM_DIR_PATTERN.match(entry.name) is not None

def _build_hefesto_indexer() -> DatasetIndexer:
    excluded = {'System_main', 'Bulk_comp', 'Bulk_comp_elements'}
    phases = [
        phase_name
        for phase_name in COMPOSITIONAL_COMPONENTS_IN_PHASES_HEFESTO.keys()
        if phase_name not in excluded
    ]
    headers = generate_column_headers_hefesto(phases)
    return DatasetIndexer(headers, OXYGEN='closed', MODEL='HeFESTo')


def _contains_simulation_dirs(path: Path) -> bool:
    for entry in path.iterdir():
        if _is_model_dir(entry):
            return True
    return False


def _find_workspace_dirs(root: Path) -> list[Path]:
    workspace_dirs: list[Path] = []

    def visit(current_dir: Path) -> None:
        if _contains_simulation_dirs(current_dir):
            workspace_dirs.append(current_dir)
            return

        for entry in sorted(current_dir.iterdir()):
            if entry.is_dir():
                visit(entry)

    visit(root)
    return workspace_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Recursively find HeFESTo workspaces under a root directory and '
            'run import_HeFESTo_components() for each workspace.'
        )
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=Path('.'),
        help='Root path containing workspace subdirectories (default: current directory).',
    )
    parser.add_argument(
        '--dataname',
        type=str,
        default='DefaultHeFESTostorage.csv',
        help=(
            'Output CSV for all parsed rows. Path is used literally; default '
            'writes in the current working directory.'
        ),
    )
    parser.add_argument(
        '--phase-change-dataname',
        type=str,
        default=None,
        help=(
            'Optional output CSV for phase boundary rows. Path is used '
            'literally; if omitted, no phase-change CSV is written.'
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    if not root.exists() or not root.is_dir():
        print(f'Error: root path is not a directory: {root}', file=sys.stderr)
        return 2

    workspace_dirs = _find_workspace_dirs(root)
    if len(workspace_dirs) == 0:
        print(f'No HeFESTo workspaces found under: {root}')
        return 0

    indexer = _build_hefesto_indexer()

    total_faults = 0
    total_successful_imports = 0
    total_empty_dirs = 0
    total_malformed_dirs = 0
    dataname = args.dataname
    phase_change_dataname = args.phase_change_dataname

    for workspace_dir in workspace_dirs:
        passed_ids, malformed_ids, empty_ids = import_HeFESTo_components(
            workspace_dir=str(workspace_dir),
            indexer=indexer,
            dataname=dataname,
            phase_change_dataname=phase_change_dataname,
        )

        n_successful_imports = int(len(passed_ids))
        malformed_dirs = int(len(malformed_ids))
        empty_dirs = int(len(empty_ids))
        n_faults = malformed_dirs + empty_dirs

        total_successful_imports += n_successful_imports
        total_empty_dirs += empty_dirs
        total_malformed_dirs += malformed_dirs
        total_faults += n_faults

        print(f'Workspace: {workspace_dir}')
        print(f'  Successful imports: {n_successful_imports}')
        print(f'  Empty model dirs: {empty_dirs}')
        print(f'  Malformed model dirs: {malformed_dirs}')
        print(f'  Fault simulation IDs count: {n_faults}')

    print('Summary:')
    print(f'  Workspaces processed: {len(workspace_dirs)}')
    print(f'  Total successful imports: {total_successful_imports}')
    print(f'  Total empty model dirs: {total_empty_dirs}')
    print(f'  Total malformed model dirs: {total_malformed_dirs}')
    print(f'  Total fault simulation IDs: {total_faults}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())