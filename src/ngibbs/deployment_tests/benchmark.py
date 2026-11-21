"""Benchmark generation and regression checking for the shipped emulators.

Two entry points (also exposed as ``HeFESToAPI`` / ``MELTSAPI`` methods
``.benchmark()`` and ``.check_benchmark()``):

* ``generate_benchmark(api, output_dir)`` scores one model on the two shipped
  datasets and writes every statistics table and plot, plus a
  ``benchmark_index.json`` describing them:

  - ``validationDataset``: the preprocessed ``*_Test_subset*.tar.gz``
    validation bundles shipped next to the checkpoints (``EmulatorAPI.test()``:
    phase quality metrics, bulk-reconstruction stats, coverage and abundance
    parity plots).
  - ``standardWorkspace``: the raw standard-rock runs shipped for the model
    (``workspace_benchmark.evaluate_workspace`` with its default path):
    per-assemblage quality tables plus the per-simulation property and phase
    comparisons against the built-in EOS.

  Each component runs independently; one that raises is recorded as ``error``
  in the index (with its message) and the rest still run. Custom workspaces
  are never part of a benchmark: they are measured with ``evaluate_workspace``.

* ``check_benchmark(api, baseline_dir)`` regenerates the benchmark with the
  baseline's recorded settings and passes ``compare_benchmarks`` on the two:
  every gated metric must be similar to or better than the baseline within
  ``BenchmarkTolerances``. Tables whose underlying data changed (different
  bundle file, different set of scored simulations) are reported as
  ``incomparable`` and fail, since a metric shift there says nothing about the
  model.

Baselines ship inside the package, next to each model's checkpoints
(``engine/TrainedModels/<family>/benchmark_baseline/<cpu|cuda>/``; plots excluded) so an installed copy can check itself. To (re)create
one, run ``generate_benchmark`` into that directory (e.g. after an
intentional model update) and commit it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

INDEX_NAME = 'benchmark_index.json'
FORMAT_VERSION = 2
PathLike = Union[str, Path]


# --------------------------------------------------------------------------- #
# tolerances
# --------------------------------------------------------------------------- #
@dataclass
class BenchmarkTolerances:
    """How far a candidate metric may move in the bad direction before it counts
    as a regression. A lower-is-better metric regresses when
    ``candidate > baseline + abs + rel * |baseline|`` (mirrored for
    higher-is-better). Exceeding the same margin in the good direction is
    reported as an improvement."""

    # phase quality tables (validation dataset and workspace): a phase is gated only when the
    # baseline has at least this many ground-truth-present rows
    min_rows_present: int = 100
    precision_recall_abs: float = 0.01
    abundance_mae_rel: float = 0.10
    abundance_mae_abs_wtpct: float = 0.05
    comp_mae_rel: float = 0.10
    comp_mae_abs_wtpct: float = 0.05
    relerr_pct_rel: float = 0.10
    relerr_pct_abs: float = 0.5          # percentage points
    recon_residual_rel: float = 0.50
    recon_residual_abs: float = 1e-5
    # standard-workspace EOS property tables
    property_mean_rel: float = 0.10
    property_mean_abs_pct: float = 0.05
    property_tail_rel: float = 0.25      # p95 / max relative errors
    property_tail_abs_pct: float = 0.25
    # standard-workspace property tables: count of rows whose predicted liquid the MELTS
    # EOS cannot represent (negative liquid basis component); NaN in the stats
    infeasible_liquid_rel: float = 0.25
    infeasible_liquid_abs: float = 5.0
    # standard-workspace phase-fraction tables (fractions of the system)
    phase_error_min_peak_fraction: float = 0.05
    phase_mean_abs_err_rel: float = 0.10
    phase_mean_abs_err_abs: float = 0.002
    phase_max_abs_err_rel: float = 0.25
    phase_max_abs_err_abs: float = 0.01

    def rule(self, kind: str, metric: str) -> Optional[Tuple[str, float, float]]:
        """(direction, rel, abs) for one metric, or None if it is informational."""
        if kind == 'phase_quality':
            if metric in ('precision', 'recall'):
                return 'higher', 0.0, self.precision_recall_abs
            if metric == 'abundance_mae_wtpct':
                return 'lower', self.abundance_mae_rel, self.abundance_mae_abs_wtpct
            if metric.endswith('_comp_mae_wtpct'):
                return 'lower', self.comp_mae_rel, self.comp_mae_abs_wtpct
            if metric.endswith('_relerr_pct'):
                return 'lower', self.relerr_pct_rel, self.relerr_pct_abs
            if metric.startswith('recon_residual_l2'):
                return 'lower', self.recon_residual_rel, self.recon_residual_abs
            return None
        if kind == 'property_errors':
            if metric.endswith(' mean_rel%'):
                return 'lower', self.property_mean_rel, self.property_mean_abs_pct
            if metric.endswith(' p95_rel%') or metric.endswith(' max_rel%'):
                return 'lower', self.property_tail_rel, self.property_tail_abs_pct
            if metric == 'n_infeasible_liquid':
                # rows whose predicted liquid MELTS's EOS can't represent
                return 'lower', self.infeasible_liquid_rel, self.infeasible_liquid_abs
            return None
        if kind == 'phase_errors':
            if metric == 'mean_abs_err':
                return 'lower', self.phase_mean_abs_err_rel, self.phase_mean_abs_err_abs
            if metric == 'max_abs_err':
                return 'lower', self.phase_max_abs_err_rel, self.phase_max_abs_err_abs
            return None
        raise ValueError(f'Unknown table kind {kind!r}')


class BenchmarkRegressionError(AssertionError):
    """Raised by ``check_benchmark`` when the candidate fails the comparison.
    ``.comparison`` holds the full result of ``compare_benchmarks``."""

    def __init__(self, comparison: dict):
        self.comparison = comparison
        super().__init__(comparison['summary_text'])


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def _frame_signature(df: pd.DataFrame) -> str:
    """Identifies which rows (and, if present, how many samples each) a
    standards comparison table covers."""
    keys = [list(map(str, k)) if isinstance(k, tuple) else [str(k)] for k in df.index]
    n = df['n'].tolist() if 'n' in df.columns else []
    return hashlib.sha1(json.dumps([keys, n]).encode()).hexdigest()


def _is_melts(api) -> bool:
    return hasattr(api, 'nocr') and hasattr(api, 'cr')


def _model_dir(api) -> Path:
    return Path(api._model_dir) if _is_melts(api) else Path(api.home_dir)


def _model_identity(api) -> dict:
    model_dir = _model_dir(api)
    checkpoints = sorted(list(model_dir.glob('*.tar')) + list(model_dir.glob('*.pt')))
    return {
        'api': type(api).__name__,
        'name': getattr(api, 'benchmark_name', None),
        'model_dir': str(model_dir),
        'checkpoints': {p.name: _sha256(p) for p in checkpoints},
    }


def _environment(api) -> dict:
    try:
        from importlib.metadata import version
        ngibbs_version = version('ngibbs')
    except Exception:
        ngibbs_version = None
    import torch
    repo = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
    except Exception:
        commit = None
    device = str(api.nocr.device if _is_melts(api) else api.device)
    return {'ngibbs_version': ngibbs_version, 'git_commit': commit, 'torch': torch.__version__,
            'python': platform.python_version(), 'device': device}


def _rel(path: Path, root: Path) -> str:
    return str(Path(path).resolve().relative_to(root.resolve()))


def _error(e: BaseException) -> dict:
    return {'status': 'error', 'message': f'{type(e).__name__}: {e}',
            'traceback': traceback.format_exc(limit=8)}


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def _run_validation_dataset(api, out: Path, root: Path, settings: dict, verbose: bool):
    from ngibbs.engine.API import EmulatorAPI
    from .workspace_benchmark import VALIDATION_DATASET
    tables, components = [], {}
    subs = [('nocr', api.nocr), ('cr', api.cr)] if _is_melts(api) else [('', api)]
    for tag, sub in subs:
        comp = f'{VALIDATION_DATASET}/{tag}' if tag else VALIDATION_DATASET
        sub_dir = out / VALIDATION_DATASET / tag if tag else out / VALIDATION_DATASET
        try:
            r = EmulatorAPI.test(
                sub, output_dir=sub_dir, max_samples=settings['max_samples'], seed=settings['seed'],
                write_outputs=True, verbose=verbose,
                allow_missing_models=settings['allow_missing_models'],
                plot_coverage=settings['plots'], plot_reconstruction=settings['plots'],
                plot_parity=settings['plots'],
            )
        except Exception as e:
            components[comp] = _error(e)
            continue
        components[comp] = {'status': 'ok', 'skipped_bundles': r['skipped_bundles']}
        for name, df in r['quality_metrics'].items():
            meta = r['meta'][name]
            bundle = Path(meta['bundle'])
            tables.append({
                'id': f'{comp}/{name}', 'component': comp, 'kind': 'phase_quality',
                'file': _rel(sub_dir / f'emulator_quality_{name}.csv', root), 'index_levels': 1,
                'dataset': {'bundle': bundle.name, 'bundle_sha256': _sha256(bundle),
                            'n_samples': meta['n_samples']},
            })
    return tables, components


def _run_standard_workspace(api, out: Path, root: Path, settings: dict, verbose: bool):
    from .workspace_benchmark import STANDARD_WORKSPACE, evaluate_workspace, standard_workspace
    comp = STANDARD_WORKSPACE
    path = standard_workspace(api)
    if path is None or not path.is_dir():
        return [], {comp: {'status': 'unavailable', 'path': str(path),
                           'message': 'the model ships no standard workspace'}}
    ws_out = out / STANDARD_WORKSPACE
    try:
        r = evaluate_workspace(api, None, ws_out, pathways=settings['pathways'],
                               plots=settings['plots'], verbose=verbose)
    except Exception as e:
        return [], {comp: _error(e)}
    m = r['manifest']
    components = {comp: {
        'status': 'ok', 'path': str(path),
        'n_simulations_found': m['n_simulations_found'],
        'n_simulations_benchmarked': m['n_simulations_benchmarked'],
        'n_assemblages': m['n_assemblages'],
        'n_skipped_simulations': len(m['skipped_simulations']),
        'comparison_failed': m['comparison']['failed'],
        'without_properties': m['comparison']['without_properties'],
    }}
    tables = []
    for key in r['quality_metrics']:
        g = m['groups'][key]
        tables.append({
            'id': f'{comp}/{key}', 'component': comp, 'kind': 'phase_quality',
            'file': _rel(ws_out / f'emulator_quality_{key}.csv', root), 'index_levels': 1,
            'dataset': {'n_simulations': g['n_simulations'], 'n_assemblages': g['n_assemblages'],
                        'signature': g['dataset_signature']},
        })
    for key, kind, levels, fname in (('property_errors', 'property_errors', 2, 'property_errors.csv'),
                                     ('phase_errors', 'phase_errors', 3, 'phase_errors.csv')):
        df = r[key]
        if df.empty:
            continue
        tables.append({'id': f'{comp}/{key}', 'component': comp, 'kind': kind,
                       'file': _rel(ws_out / fname, root), 'index_levels': levels,
                       'dataset': {'signature': _frame_signature(df)}})
    return tables, components


def generate_benchmark(
    api,
    output_dir: PathLike,
    *,
    pathways: Optional[List[str]] = None,
    include_validation_dataset: bool = True,
    include_standard_workspace: bool = True,
    max_samples: Optional[int] = None,
    seed: int = 1337,
    plots: bool = True,
    allow_missing_models: bool = False,
    verbose: bool = True,
) -> dict:
    """Score ``api`` on its validation dataset and standard workspace; write
    every table and plot under ``output_dir`` plus ``benchmark_index.json``.

    ``pathways`` applies to the standard workspace (default: every
    closed-system pathway the model has). Returns the index dict (also written
    to disk), with ``'output_dir'`` added.
    """
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    settings = {
        'pathways': list(pathways) if pathways is not None else None,
        'include_validation_dataset': include_validation_dataset,
        'include_standard_workspace': include_standard_workspace,
        'max_samples': max_samples, 'seed': seed, 'plots': plots,
        'allow_missing_models': allow_missing_models,
    }

    tables, components = [], {}
    if include_validation_dataset:
        t, c = _run_validation_dataset(api, out, out, settings, verbose)
        tables += t; components.update(c)
    if include_standard_workspace:
        t, c = _run_standard_workspace(api, out, out, settings, verbose)
        tables += t; components.update(c)

    index = {
        'format_version': FORMAT_VERSION,
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'model': _model_identity(api),
        'environment': _environment(api),
        'settings': settings,
        'components': components,
        'tables': tables,
    }
    (out / INDEX_NAME).write_text(json.dumps(index, indent=2))
    (out / 'benchmark_summary.md').write_text(_generation_summary(index))
    if verbose:
        bad = {k: v['status'] for k, v in components.items() if v['status'] != 'ok'}
        print(f'[benchmark] {len(tables)} table(s) -> {out}' + (f'; not ok: {bad}' if bad else ''))
    index['output_dir'] = out
    return index


def _generation_summary(index: dict) -> str:
    m = index['model']
    lines = [f"# Benchmark: {m['name'] or m['api']}", '',
             f"- created: {index['created_utc']}",
             f"- model dir: `{m['model_dir']}`",
             f"- git commit: {index['environment']['git_commit']}, device: {index['environment']['device']}",
             '', '## Components', '', '| component | status | detail |', '|---|---|---|']
    for name, c in index['components'].items():
        detail = c.get('message', '')
        if 'n_assemblages' in c:
            detail = (f"{c['n_simulations_benchmarked']}/{c['n_simulations_found']} simulations, "
                      f"assemblages {c['n_assemblages']}")
        lines.append(f"| {name} | {c['status']} | {str(detail).splitlines()[0] if detail else ''} |")
    lines += ['', '## Tables', '', '| id | kind | file |', '|---|---|---|']
    lines += [f"| {t['id']} | {t['kind']} | `{t['file']}` |" for t in index['tables']]
    return '\n'.join(lines) + '\n'


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
def load_index(benchmark_dir: PathLike) -> dict:
    path = Path(benchmark_dir) / INDEX_NAME
    if not path.is_file():
        raise FileNotFoundError(f'No {INDEX_NAME} in {benchmark_dir}; generate a benchmark there first.')
    return json.loads(path.read_text())


def _read_table(benchmark_dir: Path, entry: dict) -> pd.DataFrame:
    return pd.read_csv(benchmark_dir / entry['file'], index_col=list(range(entry['index_levels'])),
                       na_values='--')


def _check_cell(b, c, rule) -> Tuple[str, float]:
    direction, rel, abs_ = rule
    margin = abs_ + rel * abs(b)
    limit = b - margin if direction == 'higher' else b + margin
    if not np.isfinite(c):
        return 'regressed', limit
    if direction == 'higher':
        return ('regressed' if c < limit else 'improved' if c > b + margin else 'ok'), limit
    return ('regressed' if c > limit else 'improved' if c < b - margin else 'ok'), limit


def _gated_cells(kind: str, B: pd.DataFrame, tol: BenchmarkTolerances):
    """Yield (row_key, column, metric) for every baseline cell under a rule."""
    if kind == 'phase_quality':
        first = B.columns[0] if len(B.columns) else None
        for metric in ('recon_residual_l2_mean', 'recon_residual_l2_p95'):
            if first is not None and metric in B.index:
                yield metric, first, metric
        for phase in B.columns:
            n = B.at['n_rows_present', phase] if 'n_rows_present' in B.index else np.nan
            if not (n >= tol.min_rows_present):
                continue
            for metric in B.index:
                if not metric.startswith('recon_residual') and tol.rule(kind, metric):
                    yield metric, phase, metric
    else:
        for row in B.index:
            if kind == 'phase_errors':
                phase = row[-1] if isinstance(row, tuple) else row
                peak = B.at[row, 'gt_peak_fraction'] if 'gt_peak_fraction' in B.columns else np.nan
                if phase != '__assemblage_L1__' and not (peak >= tol.phase_error_min_peak_fraction):
                    continue
            for col in B.columns:
                if tol.rule(kind, col):
                    yield row, col, col


def compare_benchmarks(
    baseline_dir: PathLike,
    candidate_dir: PathLike,
    *,
    tolerances: Optional[BenchmarkTolerances] = None,
    report_dir: Optional[PathLike] = None,
) -> dict:
    """Compare two generated benchmarks table by table.

    Returns a dict with 'passed', 'tables' (one row per baseline/candidate
    table: status in ok / improved / regressed / missing / incomparable /
    skipped / new / baseline_error), 'cells' (every gated metric compared),
    'regressions' (the regressed cells), 'model_changed', 'summary_text'.
    Writes benchmark_comparison_{tables,cells}.csv and
    benchmark_comparison_summary.md to ``report_dir`` (default:
    ``<candidate_dir>/comparison``).
    """
    tol = tolerances or BenchmarkTolerances()
    base_dir, cand_dir = Path(baseline_dir).resolve(), Path(candidate_dir).resolve()
    base_idx, cand_idx = load_index(base_dir), load_index(cand_dir)
    cand_tables = {t['id']: t for t in cand_idx['tables']}
    base_ids = {t['id'] for t in base_idx['tables']}

    table_rows, cells = [], []

    def add_table(tid, status, message='', **counts):
        table_rows.append({'table': tid, 'status': status, 'message': message, **counts})

    for b in base_idx['tables']:
        c = cand_tables.get(b['id'])
        if c is None:
            comp = cand_idx['components'].get(b['component'], {})
            if comp.get('status') == 'unavailable':
                add_table(b['id'], 'skipped', comp.get('message', 'unavailable on this machine'))
            elif comp.get('status') == 'error':
                add_table(b['id'], 'missing', f"candidate {b['component']} errored: {comp['message']}")
            else:
                add_table(b['id'], 'missing', 'table not produced by the candidate')
            continue
        if c['dataset'] != b['dataset']:
            add_table(b['id'], 'incomparable',
                      f"underlying data changed: baseline {b['dataset']} vs candidate {c['dataset']}; "
                      'regenerate the baseline if this is intended')
            continue
        B, C = _read_table(base_dir, b), _read_table(cand_dir, c)
        counts = {'regressed': 0, 'improved': 0, 'ok': 0}
        for row, col, metric in _gated_cells(b['kind'], B, tol):
            bv = float(B.at[row, col])
            if not np.isfinite(bv):
                continue
            cv = float(C.at[row, col]) if (row in C.index and col in C.columns) else np.nan
            status, limit = _check_cell(bv, cv, tol.rule(b['kind'], metric))
            counts[status] += 1
            cells.append({'table': b['id'], 'row': ' / '.join(map(str, row)) if isinstance(row, tuple) else row,
                          'column': col, 'baseline': bv, 'candidate': cv, 'limit': limit,
                          'direction': tol.rule(b['kind'], metric)[0], 'status': status})
        status = 'regressed' if counts['regressed'] else 'improved' if counts['improved'] else 'ok'
        add_table(b['id'], status, n_compared=sum(counts.values()), n_regressed=counts['regressed'],
                  n_improved=counts['improved'])

    for t in cand_idx['tables']:
        if t['id'] not in base_ids:
            add_table(t['id'], 'new', 'not in the baseline (not gated)')
    for name, comp in base_idx['components'].items():
        if comp['status'] == 'error':
            add_table(name, 'baseline_error', comp['message'])

    tables_df = pd.DataFrame(table_rows)
    cells_df = pd.DataFrame(cells, columns=['table', 'row', 'column', 'baseline', 'candidate',
                                            'limit', 'direction', 'status'])
    failing = tables_df['status'].isin(['regressed', 'missing', 'incomparable'])
    passed = not bool(failing.any())
    model_changed = base_idx['model']['checkpoints'] != cand_idx['model']['checkpoints']
    summary = _comparison_summary(passed, base_idx, cand_idx, tables_df, cells_df, model_changed, tol)

    report_dir = Path(report_dir) if report_dir is not None else cand_dir / 'comparison'
    report_dir.mkdir(parents=True, exist_ok=True)
    tables_df.to_csv(report_dir / 'benchmark_comparison_tables.csv', index=False)
    cells_df.to_csv(report_dir / 'benchmark_comparison_cells.csv', index=False)
    (report_dir / 'benchmark_comparison_summary.md').write_text(summary)
    (report_dir / 'tolerances.json').write_text(json.dumps(asdict(tol), indent=2))

    return {'passed': passed, 'tables': tables_df, 'cells': cells_df,
            'regressions': cells_df[cells_df['status'] == 'regressed'],
            'model_changed': model_changed, 'summary_text': summary, 'report_dir': report_dir}


def _comparison_summary(passed, base_idx, cand_idx, tables_df, cells_df, model_changed, tol) -> str:
    name = base_idx['model']['name'] or base_idx['model']['api']
    lines = [f"# Benchmark comparison: {name} — {'PASSED' if passed else 'FAILED'}", '',
             f"- baseline: {base_idx['created_utc']} (commit {base_idx['environment']['git_commit']}, "
             f"{base_idx['environment']['device']})",
             f"- candidate: {cand_idx['created_utc']} (commit {cand_idx['environment']['git_commit']}, "
             f"{cand_idx['environment']['device']})",
             f"- checkpoints changed since baseline: {'yes' if model_changed else 'no'}"]
    b_dev, c_dev = base_idx['environment']['device'], cand_idx['environment']['device']
    if b_dev.split(':')[0] != c_dev.split(':')[0]:
        lines.append(f'- WARNING: cross-device comparison ({b_dev} baseline vs {c_dev} candidate). '
                     'Float32 rounding differs between devices and is amplified for minor '
                     'phases; compare against a baseline generated on the same device.')
    lines += ['',
             '| table | status | compared | regressed | improved | note |', '|---|---|---|---|---|---|']
    for _, r in tables_df.iterrows():
        def n(k):
            v = r.get(k)
            return '' if v is None or (isinstance(v, float) and np.isnan(v)) else int(v)
        lines.append(f"| {r['table']} | {r['status']} | {n('n_compared')} | {n('n_regressed')} | "
                     f"{n('n_improved')} | {str(r['message'])[:160]} |")
    reg = cells_df[cells_df['status'] == 'regressed']
    if len(reg):
        lines += ['', f'## Regressions ({len(reg)})', '',
                  '| table | row | column | baseline | candidate | limit |', '|---|---|---|---|---|---|']
        for _, r in reg.head(60).iterrows():
            lines.append(f"| {r['table']} | {r['row']} | {r['column']} | {r['baseline']:.4g} | "
                         f"{r['candidate']:.4g} | {r['limit']:.4g} |")
        if len(reg) > 60:
            lines.append(f'| ... {len(reg) - 60} more in benchmark_comparison_cells.csv | | | | | |')
    if (tables_df['status'] == 'improved').any() and passed:
        lines += ['', 'Some metrics improved beyond tolerance; consider regenerating the baseline.']
    return '\n'.join(lines) + '\n'


def check_benchmark(
    api,
    baseline_dir: Optional[PathLike] = None,
    output_dir: Optional[PathLike] = None,
    *,
    tolerances: Optional[BenchmarkTolerances] = None,
    raise_on_regression: bool = True,
    plots: Optional[bool] = None,
    verbose: bool = True,
) -> dict:
    """Regenerate the benchmark with the baseline's settings and compare.

    ``plots`` overrides the baseline's recorded plot setting (``None``: use
    it); plots do not affect any compared metric.

    ``baseline_dir`` defaults to the model's ``benchmark_baseline_dir``
    attribute (set in ``ngibbs.engine.models``: the packaged
    ``engine/TrainedModels/<family>/benchmark_baseline/<device>/``). The candidate is written
    to ``output_dir`` (default:
    ``./ngibbs_deployment_test/<model>/benchmark_candidate``).

    Raises ``BenchmarkRegressionError`` on failure unless
    ``raise_on_regression=False``; returns the ``compare_benchmarks`` dict
    (plus 'candidate_dir').
    """
    from .workspace_benchmark import default_output_root
    if baseline_dir is None:
        baseline_dir = getattr(api, 'benchmark_baseline_dir', None)
        if baseline_dir is None:
            raise ValueError('No baseline_dir given and the model has no benchmark_baseline_dir.')
    base_idx = load_index(baseline_dir)
    if base_idx.get('format_version') != FORMAT_VERSION:
        raise ValueError(f'Baseline {baseline_dir} has format version {base_idx.get("format_version")}, '
                         f'this ngibbs writes {FORMAT_VERSION}; regenerate it with '
                         '`python scripts/run_benchmarks.py --generate`.')
    s = base_idx['settings']
    if output_dir is None:
        output_dir = default_output_root(api) / 'benchmark_candidate'
    generate_benchmark(
        api, output_dir, pathways=s['pathways'],
        include_validation_dataset=s['include_validation_dataset'],
        include_standard_workspace=s['include_standard_workspace'],
        max_samples=s['max_samples'], seed=s['seed'],
        plots=s['plots'] if plots is None else plots,
        allow_missing_models=s['allow_missing_models'], verbose=verbose,
    )
    result = compare_benchmarks(baseline_dir, output_dir, tolerances=tolerances)
    result['candidate_dir'] = Path(output_dir)
    if verbose:
        print(result['summary_text'])
    if not result['passed'] and raise_on_regression:
        raise BenchmarkRegressionError(result)
    return result
