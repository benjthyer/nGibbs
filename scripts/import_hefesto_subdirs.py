"""
Import HeFESTo workspaces from a root path.

The script recursively searches for the lowest-level workspace directories
that directly contain SimulationN folders.

For each discovered workspace directory:
1) Cleanup simulation subdirectories:
     - If a simulation directory contains only 'control' or 'control' + 'ad.in',
         delete that simulation directory.
     - Otherwise, delete files named 'fort.29' and 'qout' when present.
2) Run import_HeFESTo_components() on that workspace.
"""

from __future__ import annotations

import argparse
import re
import shutil
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
from ngibbs.config.constants import COMPOSITIONAL_COMPONENTS_IN_PHASES_HEFESTO  # noqa: E402


_SIM_DIR_PATTERN = re.compile(r'^simulation\d+$', flags=re.IGNORECASE)


def _build_hefesto_indexer() -> DatasetIndexer:
    excluded = {'System_main', 'Bulk_comp', 'Bulk_comp_elements'}
    phases = [
        phase_name
        for phase_name in COMPOSITIONAL_COMPONENTS_IN_PHASES_HEFESTO.keys()
        if phase_name not in excluded
    ]
    headers = generate_column_headers_hefesto(phases)
    return DatasetIndexer(headers, OXYGEN='closed', MODEL='HeFESTo')


def _looks_like_control_only_dir(sim_dir: Path) -> bool:
    files = {entry.name for entry in sim_dir.iterdir() if entry.is_file()}
    has_nested_dirs = any(entry.is_dir() for entry in sim_dir.iterdir())
    if has_nested_dirs:
        return False
    if 'control' not in files:
        return False
    return files.issubset({'control', 'ad.in'})


def _cleanup_simulation_dirs(workspace_dir: Path) -> tuple[int, int]:
    deleted_dirs = 0
    deleted_files = 0

    for entry in workspace_dir.iterdir():
        if not entry.is_dir() or _SIM_DIR_PATTERN.match(entry.name) is None:
            continue

        if _looks_like_control_only_dir(entry):
            shutil.rmtree(entry)
            deleted_dirs += 1
            continue

        for file_name in ('fort.29', 'qout'):
            target = entry / file_name
            if target.exists() and target.is_file():
                target.unlink()
                deleted_files += 1

    return deleted_dirs, deleted_files


def _contains_simulation_dirs(path: Path) -> bool:
    for entry in path.iterdir():
        if entry.is_dir() and _SIM_DIR_PATTERN.match(entry.name) is not None:
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
            'run import_HeFESTo_components() for each workspace, with optional '
            'phase-change export and pre-import simulation directory cleanup.'
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
    total_deleted_dirs = 0
    total_deleted_files = 0

    dataname = args.dataname
    phase_change_dataname = args.phase_change_dataname

    for workspace_dir in workspace_dirs:
        deleted_dirs, deleted_files = _cleanup_simulation_dirs(workspace_dir)
        total_deleted_dirs += deleted_dirs
        total_deleted_files += deleted_files

        _, malformed_ids, empty_ids = import_HeFESTo_components(
            workspace_dir=str(workspace_dir),
            indexer=indexer,
            dataname=dataname,
            phase_change_dataname=phase_change_dataname,
        )

        n_faults = int(len(malformed_ids) + len(empty_ids))
        total_faults += n_faults

        print(f'Workspace: {workspace_dir}')
        print(f'  Deleted control-only Simulation dirs: {deleted_dirs}')
        print(f'  Deleted fort.29/qout files: {deleted_files}')
        print(f'  Fault simulation IDs count: {n_faults}')

    print('Summary:')
    print(f'  Workspaces processed: {len(workspace_dirs)}')
    print(f'  Deleted control-only Simulation dirs: {total_deleted_dirs}')
    print(f'  Deleted fort.29/qout files: {total_deleted_files}')
    print(f'  Total fault simulation IDs: {total_faults}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
