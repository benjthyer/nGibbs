"""
Import HeFESTo workspaces *with* composition derivatives.

Same discovery, cleanup and tallying as `import_hefesto_subdirs.py`, plus two shadow
tables carrying HeFESTo's own `dn/dT|_P` and `dn/dP|_T` from `fort.42`.

The shadows use the identical `indexer.database_headers` schema, one row per row of the
main table in the same order, with derivatives in the component slots and `P(GPa)` /
`T(K)` copied through so the join can be verified rather than assumed.

Two behaviours worth knowing before running:

* Most HeFESTo builds never wrote `fort.42`. A simulation without one has its
  derivatives **reconstructed** from `fort.99` + `fort.56` by default -- they are a
  deterministic function of the assemblage and the P-T state, the same Hessian system
  `physub.f` solves. Validated at mean correlation 0.9972 and rms error 2.6e-5 against
  a run that does carry one. `--no-recover` disables it; the manifest records
  `deriv_source` per simulation either way.
* `--verify N` runs the chain-rule check on N imported simulations:

      dn/dP|_isentrope = dn/dP|_T + dn/dT|_P * dT/dP

  Every ratio should read 1.000. This is the only cheap way to catch fort.42/fort.99
  misalignment, wrong species mapping, or a normalisation slip -- all silent otherwise.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
for _p in (REPO_ROOT, SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from builder.HeFESTo.HeFESTo_functions import import_HeFESTo_components  # noqa: E402
from builder.HeFESTo.HeFESTo_derivative_import import (  # noqa: E402
    import_HeFESTo_derivatives, verify_chain_rule)
from builder.HeFESTo.HeFESTo_deep_sampling import (  # noqa: E402
    collect_deep_phase_changes)
from builder.indexer import DatasetIndexer, generate_column_headers_hefesto  # noqa: E402
from ngibbs.config.constants import COMPOSITIONAL_COMPONENTS_IN_PHASES_HEFESTO  # noqa: E402
from ngibbs.utils.file_utils import get_dropped_rows, reset_dropped_rows  # noqa: E402

_SIM_DIR_PATTERN = re.compile(r'^simulation\d+$', flags=re.IGNORECASE)


def _build_hefesto_indexer() -> DatasetIndexer:
    excluded = {'System_main', 'Bulk_comp', 'Bulk_comp_elements'}
    phases = [p for p in COMPOSITIONAL_COMPONENTS_IN_PHASES_HEFESTO if p not in excluded]
    return DatasetIndexer(generate_column_headers_hefesto(phases),
                          OXYGEN='closed', MODEL='HeFESTo')


def _looks_like_control_only_dir(sim_dir: Path) -> bool:
    files = {e.name for e in sim_dir.iterdir() if e.is_file()}
    if any(e.is_dir() for e in sim_dir.iterdir()):
        return False
    if 'control' not in files:
        return False
    return files.issubset({'control', 'ad.in'})


def _cleanup_simulation_dirs(workspace_dir: Path, keep_fort42: bool = True):
    """Same cleanup as the original script.

    `fort.42` is NOT in the delete list and must not be added to it -- it is the
    derivative dataset. `fort.29` and `qout` are still removed.
    """
    deleted_dirs = deleted_files = 0
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
    return any(e.is_dir() and _SIM_DIR_PATTERN.match(e.name) for e in path.iterdir())


def _find_workspace_dirs(root: Path):
    found = []

    def visit(cur: Path):
        if _contains_simulation_dirs(cur):
            found.append(cur)
            return
        for entry in sorted(cur.iterdir()):
            if entry.is_dir():
                visit(entry)

    visit(root)
    return found


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Import HeFESTo workspaces and their fort.42 composition derivatives.')
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--dataname', type=str, default='DefaultHeFESTostorage.csv')
    p.add_argument('--phase-change-dataname', type=str, default=None)
    p.add_argument('--phase-change-offset-only', action='store_true')
    p.add_argument('--dndp-dataname', type=str, default=None,
                   help='Shadow CSV for dn/dP. Default: <dataname stem>_dndP.csv')
    p.add_argument('--dndt-dataname', type=str, default=None,
                   help='Shadow CSV for dn/dT. Default: <dataname stem>_dndT.csv')
    p.add_argument('--manifest', type=str, default=None,
                   help='Per-simulation provenance CSV (rows, N_el, offset flag). '
                        'Default: <dataname stem>_deriv_manifest.csv')
    p.add_argument('--normalise', action='store_true',
                   help='Divide derivatives by the run element total. OFF by default '
                        'and should stay off: the main import writes fort.99 moles raw, '
                        'so raw fort.42 values are the ones on the matching scale. '
                        'Normalising only the derivatives puts them 1/23.6 below their '
                        'own moles, which nothing downstream would catch.')
    p.add_argument('--deep-phase-change-dataname', type=str, default=None,
                   help='Collect phase-change bounds from a workspace that is ALREADY '
                        'a phase-change resample, i.e. each simulation is a P-T grid. '
                        'The grid is split into 1-D scans first and bounds are found '
                        'within each; treating it as one continuum would read every '
                        'scan seam as a transition. Same output schema as '
                        '--phase-change-dataname.')
    p.add_argument('--deep-axis', choices=('isotherm', 'isobar', 'both'),
                   default='isotherm',
                   help='Scan direction for --deep-phase-change-dataname. isotherm '
                        'fixes T and sweeps P; isobar fixes P and sweeps T.')
    p.add_argument('--skip-derivatives', action='store_true')
    p.add_argument('--no-recover', action='store_true',
                   help='Do not reconstruct derivatives for simulations lacking '
                        'fort.42. Default is to recover them from fort.99 + fort.56.')
    p.add_argument('--param-dir', type=str, default=None,
                   help='HeFESTo parameter directory for recovery. Default: the path '
                        'recorded in each control file, else the copy packaged with '
                        'ngibbs. Needed when runs came from another machine.')
    p.add_argument('--recover-nsmall-rel', type=float, default=0.0,
                   help='Active-set pruning for the recovery solve. Keep 0 for '
                        'HeFESTo-derived assemblages (see the module docstring).')
    p.add_argument('--write-recovered-fort42', action='store_true',
                   help='Write each reconstruction to fort.42 in its simulation '
                        'directory so a re-import reads it instead of recomputing. '
                        'Modifies the data directory.')
    p.add_argument('--verify', type=int, default=2, metavar='N',
                   help='Chain-rule check on the first N imported simulations per '
                        'workspace (0 disables).')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        print(f'Error: root path is not a directory: {root}', file=sys.stderr)
        return 2

    workspaces = _find_workspace_dirs(root)
    if not workspaces:
        print(f'No HeFESTo workspaces found under: {root}')
        return 0

    stem = Path(args.dataname)
    dndp_name = args.dndp_dataname or str(stem.with_name(stem.stem + '_dndP.csv'))
    dndt_name = args.dndt_dataname or str(stem.with_name(stem.stem + '_dndT.csv'))
    manifest = args.manifest or str(stem.with_name(stem.stem + '_deriv_manifest.csv'))

    indexer = _build_hefesto_indexer()
    tot = dict(faults=0, dirs=0, files=0, shifted=0,
               deriv_ok=0, deriv_bad=0, no42=0, recovered=0, verify_fail=0,
               deep_pairs=0, deep_bad=0)

    for ws in workspaces:
        d_dirs, d_files = _cleanup_simulation_dirs(ws)
        tot['dirs'] += d_dirs
        tot['files'] += d_files

        reset_dropped_rows()
        _, malformed, empty = import_HeFESTo_components(
            workspace_dir=str(ws), indexer=indexer, dataname=args.dataname,
            phase_change_dataname=args.phase_change_dataname,
            phase_change_offset_only=args.phase_change_offset_only)

        shifted = {p for p in get_dropped_rows() if p.endswith('fort.99')}
        tot['shifted'] += len(shifted)
        tot['faults'] += len(malformed) + len(empty)

        print(f'Workspace: {ws}')
        print(f'  Deleted control-only Simulation dirs: {d_dirs}')
        print(f'  Deleted fort.29/qout files: {d_files}')
        print(f'  Fault simulation IDs count: {len(malformed) + len(empty)}')
        print(f'  Simulations with offset fort.99: {len(shifted)}')

        if args.deep_phase_change_dataname is not None:
            dp_pass, dp_bad, dp_pairs = collect_deep_phase_changes(
                workspace_dir=str(ws), indexer=indexer,
                out_csv=args.deep_phase_change_dataname, axis=args.deep_axis)
            tot['deep_pairs'] += dp_pairs
            tot['deep_bad'] += len(dp_bad)
            print(f'  Deep phase-change pairs ({args.deep_axis}): {dp_pairs} '
                  f'from {len(dp_pass)} sims, {len(dp_bad)} malformed')

        if args.skip_derivatives:
            continue

        passed, bad, no42, recovered = import_HeFESTo_derivatives(
            workspace_dir=str(ws), indexer=indexer,
            dndp_dataname=dndp_name, dndt_dataname=dndt_name,
            manifest_name=manifest, normalise=args.normalise,
            recover=not args.no_recover, param_dir=args.param_dir,
            recover_nsmall_rel=args.recover_nsmall_rel,
            write_recovered=args.write_recovered_fort42)
        tot['deriv_ok'] += len(passed)
        tot['deriv_bad'] += len(bad)
        tot['no42'] += len(no42)
        tot['recovered'] += len(recovered)

        n_sims = len(passed) + len(bad) + len(no42)
        cov = 100.0 * len(passed) / n_sims if n_sims else 0.0
        native = len(passed) - len(recovered)
        print(f'  Derivative values for: {len(passed)}/{n_sims} sims ({cov:.0f}% coverage)')
        print(f'    from fort.42: {native}   reconstructed: {len(recovered)}')
        if no42:
            print(f'  NaN rows, fort.42 absent and recovery disabled: {len(no42)}')
        if bad:
            print(f'  NaN rows, derivatives failed: {len(bad)}')

        # Verify wherever a fort.42 exists on disk -- native, or one this tool wrote
        # with --write-recovered-fort42. Checking a written reconstruction is not
        # vacuous: it tests the write/read round-trip and the reconstruction against
        # fort.99, which is exactly the path that surfaced the stacked-profile bug.
        native_ids = [i for i in passed
                      if (ws / f'Simulation{i}' / 'fort.42').exists()]
        for sim_id in native_ids[:max(0, args.verify)]:
            sim_dir = ws / f'Simulation{sim_id}'
            if not sim_dir.is_dir():
                continue
            res = verify_chain_rule(str(sim_dir))
            ratios = '  '.join(f'{k}:{v:.3f}' for k, v in res.items() if not k.startswith('_'))
            ok = res.get('_pass', False)
            tot['verify_fail'] += 0 if ok else 1
            seg = res.get('_segments', 1)
            print(f'  [verify] Simulation{sim_id} chain rule {"PASS" if ok else "FAIL"} '
                  f'({seg} monotonic segment{"s" if seg != 1 else ""}, '
                  f'dP={res.get("_median_dP", float("nan")):.4g} GPa, '
                  f'tol={res.get("_tolerance", 0):.3f}): {ratios}')

    # Row parity is the property the shadows exist for -- check it rather than trust it.
    if not args.skip_derivatives:
        try:
            import pandas as _pd
            n_main = sum(1 for _ in open(args.dataname)) - 1
            n_dp = sum(1 for _ in open(dndp_name)) - 1
            n_dt = sum(1 for _ in open(dndt_name)) - 1
            match = (n_main == n_dp == n_dt)
            print(f'Row parity: main {n_main}  dndP {n_dp}  dndT {n_dt}  '
                  f'{"MATCH" if match else "MISMATCH"}')
            if not match:
                print('  Shadows do not align positionally with the main table. If the '
                      'main import resumed from a checkpoint, delete the shadows and '
                      'the checkpoint and rerun both from scratch.')
                tot['verify_fail'] += 1
        except OSError as exc:
            print(f'Row parity: could not check ({exc})')

    print('Summary:')
    print(f'  Workspaces processed: {len(workspaces)}')
    if args.deep_phase_change_dataname is not None:
        print(f'  Deep phase-change pairs ({args.deep_axis}): {tot["deep_pairs"]} '
              f'-> {args.deep_phase_change_dataname}')
    print(f'  Deleted control-only Simulation dirs: {tot["dirs"]}')
    print(f'  Deleted fort.29/qout files: {tot["files"]}')
    print(f'  Total fault simulation IDs: {tot["faults"]}')
    print(f'  Total simulations with offset fort.99: {tot["shifted"]}')
    if not args.skip_derivatives:
        print(f'  Derivative coverage: {tot["deriv_ok"]} sims '
              f'({tot["deriv_ok"] - tot["recovered"]} native, {tot["recovered"]} reconstructed, '
              f'{tot["no42"]} skipped, {tot["deriv_bad"]} malformed)')
        print(f'  Chain-rule verification failures: {tot["verify_fail"]}')
        print(f'  Shadow tables: {dndp_name}  {dndt_name}')
        print(f'  Manifest: {manifest}')
    return 1 if tot['verify_fail'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
