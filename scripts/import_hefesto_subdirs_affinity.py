"""
Import HeFESTo workspaces *with* phase affinities parsed from ``qout``.

Same discovery, cleanup, tallying and phase-change export as
``import_hefesto_subdirs.py``, with two differences:

1) ``qout`` is NO LONGER DELETED during cleanup (``fort.29`` still is). It is
   parsed instead: the final ``phaseadd`` table of every P-T point gives the
   affinity of each absent phase at the converged assemblage, and the final
   ``lagcomp`` line gives the component (element) chemical potentials.
2) A shadow table ``<dataname stem>_affinity.csv`` is written with one row per row
   of the main table, in the same order (row parity is checked at the end). See
   ``builder/HeFESTo/HeFESTo_affinity_import.py`` for the column and flag
   definitions. Affinities keep HeFESTo's units and sign: kJ per mole of atoms,
   POSITIVE = undersaturated.

``--delete-qout-after-parse`` reclaims the space afterwards, but only for
simulations whose every fort.56 row was matched to a qout block; anything
partially matched keeps its qout so it can be inspected.

Phases that HeFESTo did not evaluate at the final assemblage (flag
NOT_EVALUATED: added-then-removed, spinodal, or redundant) are left NaN for a
downstream back-calculation; the ``mu_*`` columns are recorded for exactly that.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from builder.HeFESTo.HeFESTo_functions import import_HeFESTo_components  # noqa: E402
from builder.HeFESTo.HeFESTo_deep_sampling import collect_deep_phase_changes  # noqa: E402
from builder.HeFESTo.HeFESTo_affinity_import import (  # noqa: E402
    import_HeFESTo_affinities, FLAG_NAMES, ROW_NAMES, QOUT)
from builder.indexer import DatasetIndexer, generate_column_headers_hefesto  # noqa: E402
from ngibbs.config.constants import COMPOSITIONAL_COMPONENTS_IN_PHASES_HEFESTO  # noqa: E402
from ngibbs.utils.file_utils import get_dropped_rows, reset_dropped_rows  # noqa: E402


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
    """Delete control-only Simulation dirs and fort.29. qout is KEPT -- it is
    parsed for affinities after the main import."""
    deleted_dirs = 0
    deleted_files = 0

    for entry in workspace_dir.iterdir():
        if not entry.is_dir() or _SIM_DIR_PATTERN.match(entry.name) is None:
            continue

        if _looks_like_control_only_dir(entry):
            shutil.rmtree(entry)
            deleted_dirs += 1
            continue

        target = entry / 'fort.29'
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


def _count_rows(path: str) -> int:
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        return sum(1 for _ in fh) - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Recursively find HeFESTo workspaces under a root directory, run '
            'import_HeFESTo_components() for each, and write a row-aligned shadow '
            'table of phase affinities parsed from qout.'
        )
    )
    parser.add_argument('--root', type=Path, default=Path('.'),
                        help='Root path containing workspace subdirectories (default: cwd).')
    parser.add_argument('--dataname', type=str, default='DefaultHeFESTostorage.csv',
                        help='Output CSV for all parsed rows. Path is used literally.')
    parser.add_argument('--affinity-dataname', type=str, default=None,
                        help='Affinity shadow CSV. Default: <dataname stem>_affinity.csv')
    parser.add_argument('--affinity-manifest', type=str, default=None,
                        help='Per-simulation affinity manifest CSV. '
                             'Default: <dataname stem>_affinity_manifest.csv')
    parser.add_argument('--skip-affinities', action='store_true',
                        help='Do not parse qout (it is still kept on disk).')
    parser.add_argument('--delete-qout-after-parse', action='store_true',
                        help='After parsing, delete qout for simulations whose every '
                             'fort.56 row matched a qout block. Off by default.')
    parser.add_argument('--phase-change-dataname', type=str, default=None,
                        help='Optional output CSV for phase boundary rows.')
    parser.add_argument('--phase-change-offset-only', action='store_true',
                        help='Restrict the phase-change CSV to simulations whose fort.99 '
                             'was offset by stray HeFESTo diagnostic lines.')
    parser.add_argument('--deep-phase-change-dataname', type=str, default=None,
                        help='Optional output CSV of phase-change bounds from a workspace '
                             'that is already a phase-change resample (P-T grid per sim).')
    parser.add_argument('--deep-axis', choices=('isotherm', 'isobar', 'both'),
                        default='isotherm',
                        help='Scan direction for --deep-phase-change-dataname.')
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

    stem = Path(args.dataname)
    aff_name = args.affinity_dataname or str(stem.with_name(stem.stem + '_affinity.csv'))
    manifest = args.affinity_manifest or str(
        stem.with_name(stem.stem + '_affinity_manifest.csv'))

    indexer = _build_hefesto_indexer()

    tot = Counter()
    aff_totals = Counter()

    for workspace_dir in workspace_dirs:
        deleted_dirs, deleted_files = _cleanup_simulation_dirs(workspace_dir)
        tot['dirs'] += deleted_dirs
        tot['files'] += deleted_files

        reset_dropped_rows()

        passed_ids, malformed_ids, empty_ids = import_HeFESTo_components(
            workspace_dir=str(workspace_dir),
            indexer=indexer,
            dataname=args.dataname,
            phase_change_dataname=args.phase_change_dataname,
            phase_change_offset_only=args.phase_change_offset_only,
        )

        shifted = {p for p in get_dropped_rows() if p.endswith('fort.99')}
        tot['shifted'] += len(shifted)
        n_faults = int(len(malformed_ids) + len(empty_ids))
        tot['faults'] += n_faults

        print(f'Workspace: {workspace_dir}')
        print(f'  Deleted control-only Simulation dirs: {deleted_dirs}')
        print(f'  Deleted fort.29 files: {deleted_files}')
        print(f'  Fault simulation IDs count: {n_faults}')
        print(f'  Simulations with offset fort.99: {len(shifted)}')

        if args.deep_phase_change_dataname is not None:
            deep_passed, deep_bad, deep_pairs = collect_deep_phase_changes(
                workspace_dir=str(workspace_dir),
                indexer=indexer,
                out_csv=args.deep_phase_change_dataname,
                axis=args.deep_axis,
            )
            tot['deep_pairs'] += deep_pairs
            tot['deep_bad'] += len(deep_bad)
            print(f'  Deep phase-change pairs ({args.deep_axis}): {deep_pairs} '
                  f'from {len(deep_passed)} sims, {len(deep_bad)} malformed')

        if args.skip_affinities:
            continue

        # Only the simulations the main import accepted, so the shadow cannot gain
        # rows the main table lacks.
        full, partial, no_qout, failed, stats = import_HeFESTo_affinities(
            workspace_dir=str(workspace_dir),
            indexer=indexer,
            affinity_dataname=aff_name,
            manifest_name=manifest,
            only_sim_ids=passed_ids,
        )
        aff_totals.update(stats)
        tot['aff_full'] += len(full)
        tot['aff_partial'] += len(partial)
        tot['aff_noqout'] += len(no_qout)
        tot['aff_failed'] += len(failed)
        print(f'  Affinities: {len(full)} sims fully matched, {len(partial)} partial, '
              f'{len(no_qout)} without qout, {len(failed)} failed (NaN rows)')

        if args.delete_qout_after_parse:
            n_del = 0
            for sim_id in full:
                q = workspace_dir / f'Simulation{sim_id}' / QOUT
                if q.is_file():
                    q.unlink()
                    n_del += 1
            tot['qout_deleted'] += n_del
            print(f'  Deleted qout after parse: {n_del}')

    if not args.skip_affinities:
        try:
            n_main = _count_rows(args.dataname)
            n_aff = _count_rows(aff_name)
            match = n_main == n_aff
            print(f'Row parity: main {n_main}  affinity {n_aff}  '
                  f'{"MATCH" if match else "MISMATCH"}')
            if not match:
                print('  The affinity shadow does not align positionally with the main '
                      'table. If the main import resumed from a checkpoint, or the main '
                      'CSV already held rows from an earlier run, delete both outputs '
                      '(and the checkpoint) and rerun from scratch.')
                tot['parity_fail'] += 1
        except OSError as exc:
            print(f'Row parity: could not check ({exc})')

    print('Summary:')
    print(f'  Workspaces processed: {len(workspace_dirs)}')
    if args.deep_phase_change_dataname is not None:
        print(f'  Deep phase-change pairs ({args.deep_axis}): {tot["deep_pairs"]} '
              f'-> {args.deep_phase_change_dataname}')
    print(f'  Deleted control-only Simulation dirs: {tot["dirs"]}')
    print(f'  Deleted fort.29 files: {tot["files"]}')
    print(f'  Total fault simulation IDs: {tot["faults"]}')
    print(f'  Total simulations with offset fort.99: {tot["shifted"]}')
    if not args.skip_affinities:
        print(f'  Affinity simulations: {tot["aff_full"]} fully matched, '
              f'{tot["aff_partial"]} partial, {tot["aff_noqout"]} without qout, '
              f'{tot["aff_failed"]} failed')
        rows = {k[4:]: v for k, v in aff_totals.items() if k.startswith('row_')}
        flags = {k[5:]: v for k, v in aff_totals.items() if k.startswith('flag_')}
        print('  Row status:  ' + '  '.join(f'{n}={rows.get(n, 0)}' for n in ROW_NAMES.values()))
        print('  Phase flags: ' + '  '.join(f'{n}={flags.get(n, 0)}' for n in FLAG_NAMES.values()))
        if args.delete_qout_after_parse:
            print(f'  qout deleted after parse: {tot["qout_deleted"]}')
        print(f'  Affinity shadow: {aff_name}')
        print(f'  Manifest: {manifest}')
    return 1 if tot['parity_fail'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
