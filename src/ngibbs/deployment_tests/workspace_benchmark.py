"""Benchmark an emulator against a workspace of RAW MELTS or HeFESTo output.

``EmulatorAPI.test()`` scores a model against its shipped ML-ready
``*_Test_subset*.tar.gz`` bundle. This module scores it against any directory
tree of raw simulator output instead -- e.g. ``data/Workspace`` (alphaMELTS) or
``data/HeFESToSims/EarthValid103026/Adiabat/Batch0001`` (HeFESTo) -- with the
same metric table (``emulator_quality.phase_quality_table``), so the numbers are
directly comparable to the bundle test.

Discovery (recursive, any depth):

* MELTS: every directory holding a ``*.melts`` input file (any name, not just
  ``input.melts``). Ground truth is read from that run's alphaMELTS tables:
  ``System_main_tbl.txt`` (P, T, S, mass), ``Bulk_comp_tbl.txt`` (the system
  bulk at every step -- the emulator input; it evolves in fractionation runs, so
  the ``.melts`` file's initial composition is only the first row's),
  ``Phase_mass_tbl.txt`` (phase abundances) and each ``<phase>.tbl``
  (within-phase oxide wt%).
* HeFESTo: every directory holding a ``control`` file. Inputs are the control
  file's element moles plus P/T (or S) from ``fort.56``; ground truth is the
  extensive component moles in ``fort.99``.

Every ground-truth row is one benchmarked assemblage. Rows are dropped -- and
counted in the manifest -- when they contain a phase the model does not
represent, fall outside the model's training pressure range, or carry
non-finite values. A run that cannot be parsed at all is skipped with its
reason recorded rather than failing the benchmark.

MELTS runs are routed to the Cr or NoCr model per run through
``MELTSAPI._route`` (an all-zero Cr2O3 column selects NoCr), and scored in
separate ``<variant>_<pathway>`` tables because the two models' phase sets
differ.

The forward pass uses the same settings as ``evaluate_emulator_quality``
(``optimize_masses=True``) so workspace and bundle metrics are comparable.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

from .emulator_quality import NA_REP, legend_text, phase_quality_table

_EMULATOR_ATTR = {'isothermal': 'isothermal_emulator', 'isentropic': 'isentropic_emulator'}
_PRESSURE_FEATURES = ('P(GPa)(System_main)', 'Pressure(System_main)')
_FORWARD_OUTPUTS = ['phase_present', 'phase_tables', 'reconstruction_residual']


def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', str(path))]


def find_melts_runs(root: Union[str, Path]) -> List[Path]:
    """Directories under ``root`` (recursively) that hold a ``*.melts`` file."""
    root = Path(root)
    return sorted({p.parent for p in root.rglob('*.melts') if p.is_file()}, key=_natural_key)


def find_hefesto_runs(root: Union[str, Path]) -> List[Path]:
    """Directories under ``root`` (recursively) that hold a HeFESTo ``control`` file."""
    root = Path(root)
    return sorted({p.parent for p in root.rglob('control') if p.is_file()}, key=_natural_key)


def _pressure_bounds(emulator) -> Optional[Tuple[float, float]]:
    """Training pressure interval of ``emulator``, in its own pressure feature's
    units, read from its feature normalizer (None if unavailable)."""
    names = list(emulator.ml_indexer.featureNames)
    idx = next((names.index(n) for n in _PRESSURE_FEATURES if n in names), None)
    normalizer = getattr(emulator.ml_indexer, 'feature_normalizer', None)
    if idx is None or normalizer is None:
        return None
    lo = float(normalizer.miner[idx].item())
    return lo, lo + float(normalizer.ranger[idx].item())


def _to_np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


# --------------------------------------------------------------------------- #
# MELTS adapter
# --------------------------------------------------------------------------- #
class _MeltsAdapter:
    """Raw alphaMELTS run folder -> emulator inputs + ground truth."""

    kind = 'melts'
    # raw alphaMELTS phase name -> ml_indexer phase name (as the training importer,
    # builder/alphamelts/engine/alphamelts_functions.import_MELTS_components, maps them)
    RAW_TO_MODEL = {'liquid': 'melts-liquid', 'alkali-feldspar': 'k-feldspar', 'water': 'fluid'}
    # ml_indexer phase -> candidate <stem>.tbl files carrying its composition
    TBL_STEMS = {'melts-liquid': ('melts-liquid', 'liquid'),
                 'k-feldspar': ('alkali-feldspar', 'k-feldspar'),
                 'fluid': ('fluid', 'water')}

    def __init__(self, api):
        from .melts_comparison import _RAW_OXIDE_COLS
        self.api = api
        self.oxides = list(_RAW_OXIDE_COLS)

    find_runs = staticmethod(find_melts_runs)

    def compare(self, run_dir: Path, pathways: Sequence[str], properties: bool = True) -> dict:
        from .melts_comparison import melts_run_comparison
        return melts_run_comparison(self.api, run_dir, pathways=list(pathways) + ['openox'],
                                    properties=properties)

    def parse(self, run_dir: Path) -> dict:
        from .melts_comparison import (
            _ENTROPY_FEATURE, _dedupe_tbl_instances, _reindex_to, _system_main_feature,
            read_bulk_comp, read_phase_mass, read_phase_tbl, read_system_main,
        )
        for f in ('System_main_tbl.txt', 'Bulk_comp_tbl.txt', 'Phase_mass_tbl.txt'):
            if not (run_dir / f).is_file():
                raise FileNotFoundError(f'missing {f}')
        sysmain = read_system_main(run_dir)
        if sysmain.empty:
            raise ValueError('System_main_tbl.txt has no rows')
        index = sysmain['index'].to_numpy().astype(int)
        n = len(index)
        bulk = _reindex_to(index, read_bulk_comp(run_dir), index_col='index')

        conditions = {
            'Pressure(System_main)': sysmain['Pressure'].to_numpy(np.float64),
            'Temperature(System_main)': sysmain['Temperature'].to_numpy(np.float64),
        }
        try:
            conditions[_ENTROPY_FEATURE] = _system_main_feature(sysmain, _ENTROPY_FEATURE)
        except (KeyError, ValueError):
            pass
        composition = {ox: (bulk[ox].to_numpy(np.float64) if ox in bulk.columns else np.zeros(n))
                       for ox in self.oxides}

        mass_df, phase_cols = read_phase_mass(run_dir)
        mass_df = _reindex_to(index, mass_df, index_col='index')
        grams: Dict[str, np.ndarray] = {}
        for col, raw_name in phase_cols:
            name = self.RAW_TO_MODEL.get(raw_name, raw_name)
            grams[name] = grams.get(name, np.zeros(n)) + mass_df[col].fillna(0.0).to_numpy(np.float64)

        comp: Dict[str, np.ndarray] = {}   # phase -> (n, len(self.oxides)) wt%
        for name, g in grams.items():
            if not np.any(g > 0):
                continue
            for stem in self.TBL_STEMS.get(name, (name,)):
                tbl = read_phase_tbl(run_dir, stem)
                if tbl is None:
                    continue
                wt_cols = [f'wt% {ox}' for ox in self.oxides if f'wt% {ox}' in tbl.columns]
                if not wt_cols:
                    break
                tbl = _reindex_to(index, _dedupe_tbl_instances(tbl, wt_cols), index_col='Index')
                comp[name] = np.column_stack([
                    pd.to_numeric(tbl[f'wt% {ox}'], errors='coerce').fillna(0.0).to_numpy(np.float64)
                    if f'wt% {ox}' in tbl.columns else np.zeros(n) for ox in self.oxides])
                break

        return {'n': n, 'conditions': conditions, 'composition': composition,
                'grams': grams, 'comp': comp}

    def inputs(self, parsed: dict, pathway: str) -> Tuple[np.ndarray, List[str]]:
        from .melts_comparison import _ENTROPY_FEATURE
        second = 'Temperature(System_main)' if pathway == 'isothermal' else _ENTROPY_FEATURE
        if second not in parsed['conditions']:
            raise KeyError(f'no {second!r} available for the {pathway} pathway')
        headers = ['Pressure(System_main)', second] + self.oxides
        cols = [parsed['conditions']['Pressure(System_main)'], parsed['conditions'][second]]
        cols += [parsed['composition'][ox] for ox in self.oxides]
        return np.column_stack(cols), headers

    def route(self, table, headers):
        sub = self.api._route(table, headers)
        return ('cr' if sub is self.api.cr else 'nocr'), sub

    def ground_truth(self, parsed: dict, emulator) -> dict:
        mi = emulator.ml_indexer
        n = parsed['n']
        grams = parsed['grams']
        total = np.sum(np.stack(list(grams.values()), axis=1), axis=1) if grams else np.zeros(n)
        unmodelled = {p: g > 0 for p, g in grams.items() if p not in mi.mass_phasedict and np.any(g > 0)}

        n_phases = max(mi.mass_phasedict.values()) + 1
        mass = np.zeros((n, n_phases))
        for phase, col in mi.mass_phasedict.items():
            if phase in grams:
                mass[:, col] = grams[phase]
        safe = np.where(total > 0, total, 1.0)
        mass_wtpct = 100.0 * mass / safe[:, None]
        present = mass > 0

        model_oxides = list(mi.Oxides)
        ox_idx = [self.oxides.index(ox) if ox in self.oxides else None for ox in model_oxides]
        comp_wtpct = {}
        for phase in mi.comp_phasedict:
            raw = parsed['comp'].get(phase)
            if raw is None:
                continue
            wt = np.column_stack([raw[:, i] if i is not None else np.zeros(n) for i in ox_idx])
            s = wt.sum(axis=1, keepdims=True)
            comp_wtpct[phase] = np.where(s > 0, 100.0 * wt / np.where(s > 0, s, 1.0), 0.0)

        return {'present': present, 'mass_wtpct': mass_wtpct, 'comp_wtpct': comp_wtpct,
                'valid': np.isfinite(total) & (total > 0), 'unmodelled': unmodelled}


# --------------------------------------------------------------------------- #
# HeFESTo adapter
# --------------------------------------------------------------------------- #
class _HeFESToAdapter:
    """Raw HeFESTo simulation folder (control + fort.56 + fort.99) -> emulator
    inputs + ground truth."""

    kind = 'hefesto'

    def __init__(self, api):
        self.api = api

    find_runs = staticmethod(find_hefesto_runs)

    def compare(self, run_dir: Path, pathways: Sequence[str], properties: bool = True) -> dict:
        from .meltstable_comparison import hefesto_run_comparison
        return hefesto_run_comparison(self.api, run_dir, pathways=pathways, properties=properties)

    def parse(self, run_dir: Path) -> dict:
        from ngibbs.utils.file_utils import _safe_read_ws_table
        from .meltstable_comparison import (
            ELEMENT_KEYS, _load_composition, _load_fort99_phase_totals, _read_conditions,
        )
        for f in ('control', 'fort.56', 'fort.99'):
            p = run_dir / f
            if not p.is_file() or p.stat().st_size == 0:
                raise FileNotFoundError(f'missing or empty {f}')
        _, P, T, S = _read_conditions(run_dir)
        fort99 = _safe_read_ws_table(str(run_dir / 'fort.99'), skiprows=0)
        if len(fort99) != len(P):
            raise ValueError(f'fort.56 has {len(P)} rows but fort.99 has {len(fort99)}')
        # fort.99's leading columns are P (GPa) and T (K), rounded -- check row alignment.
        P99 = pd.to_numeric(fort99.iloc[:, 0], errors='coerce').to_numpy(np.float64)
        T99 = pd.to_numeric(fort99.iloc[:, 2], errors='coerce').to_numpy(np.float64)
        if not (np.allclose(P99, P, atol=0.01, equal_nan=True) and np.allclose(T99, T, atol=0.01, equal_nan=True)):
            raise ValueError('fort.56 and fort.99 rows are not aligned in P/T')
        comp = _load_composition(run_dir)
        n = len(P)
        return {
            'n': n, 'run_dir': run_dir,
            'conditions': {'P(GPa)(System_main)': P, 'T(K)(System_main)': T, 'S(J/g/K)(System_main)': S},
            'composition': {k: np.full(n, comp[k]) for k in ELEMENT_KEYS},
            'phase_totals': _load_fort99_phase_totals(run_dir),
        }

    def inputs(self, parsed: dict, pathway: str) -> Tuple[np.ndarray, List[str]]:
        second = 'T(K)(System_main)' if pathway == 'isothermal' else 'S(J/g/K)(System_main)'
        comp_keys = list(parsed['composition'])
        headers = ['P(GPa)(System_main)', second] + comp_keys
        cols = [parsed['conditions']['P(GPa)(System_main)'], parsed['conditions'][second]]
        cols += [parsed['composition'][k] for k in comp_keys]
        return np.column_stack(cols), headers

    def route(self, table, headers):
        return '', self.api

    def ground_truth(self, parsed: dict, emulator) -> dict:
        from ngibbs.utils.file_utils import load_fort99_componentMoles
        mi = emulator.ml_indexer
        unmodelled = {p: m > 0 for p, m in parsed['phase_totals'].items()
                      if p not in mi.mass_phasedict and np.any(m > 0)}
        # The loader prints a warning per component it cannot map; those rows are
        # already flagged in `unmodelled` (dropped and counted in the manifest).
        with contextlib.redirect_stdout(io.StringIO()):
            moles = load_fort99_componentMoles(str(parsed['run_dir']), mi)
        with torch.no_grad():
            comp_t, mass_t = emulator.make_phase_tables(
                torch.tensor(moles, dtype=torch.float32, device=emulator.dev), out='oxides')
        comp_np, mass_np = _to_np(comp_t).astype(np.float64), _to_np(mass_t).astype(np.float64)
        comp_wtpct = {phase: comp_np[:, idx, :] for phase, idx in mi.comp_phasedict.items()}
        total = moles.sum(axis=1)
        return {'present': mass_np > 0, 'mass_wtpct': mass_np, 'comp_wtpct': comp_wtpct,
                'valid': np.isfinite(total) & (total > 0), 'unmodelled': unmodelled}


def available_pathways(api) -> List[str]:
    """Closed-system pathways with a loaded emulator (on either MELTS variant)."""
    subs = [api.nocr, api.cr] if hasattr(api, 'nocr') else [api]
    return [p for p, attr in _EMULATOR_ATTR.items()
            if any(getattr(s, attr, None) is not None for s in subs)]


def _adapter_for(api):
    from ngibbs.engine.API import HeFESToAPI, MELTSAPI
    if isinstance(api, MELTSAPI):
        return _MeltsAdapter(api)
    if isinstance(api, HeFESToAPI):
        return _HeFESToAdapter(api)
    raise TypeError(f'No raw-output workspace benchmark for {type(api).__name__}; '
                    'expected a MELTSAPI or HeFESToAPI.')




# --------------------------------------------------------------------------- #
# output locations
# --------------------------------------------------------------------------- #
STANDARD_WORKSPACE = 'standardWorkspace'
CUSTOM_WORKSPACE = 'customWorkspace'
VALIDATION_DATASET = 'validationDataset'


def model_label(api) -> str:
    """The model's name (``benchmark_name``, set by ``ngibbs.make_ngibbs``),
    else its model directory's name."""
    name = getattr(api, 'benchmark_name', None)
    if name:
        return name
    home = api.nocr.home_dir if hasattr(api, 'nocr') else api.home_dir
    return Path(home).name


def default_output_root(api) -> Path:
    """Where test()/evaluate_workspace() write when no output_dir is given:
    ``./ngibbs_deployment_test/<model name>/`` in the current directory (an
    installed package directory may be read-only and is replaced on upgrade)."""
    return Path.cwd() / 'ngibbs_deployment_test' / model_label(api)


def standard_workspace(api) -> Optional[Path]:
    """The raw simulator runs shipped with this model (its standard workspace),
    or None if it ships none."""
    path = getattr(api, 'standard_workspace_dir', None)
    return Path(path) if path is not None else None


# --------------------------------------------------------------------------- #
# path classification (for the section plots)
# --------------------------------------------------------------------------- #
def classify_path(P: np.ndarray) -> Tuple[str, str]:
    """('isobar' | 'transect' | 'unplottable', reason) for one simulation's
    pressures. An isobar has a single pressure; a depth (pressure) transect has
    no repeated pressure. If any pressure repeats while others differ, the path
    cannot be drawn as a 1-D section. Temperature/entropy are unconstrained."""
    P = np.asarray(P, dtype=np.float64)
    P = P[np.isfinite(P)]
    if P.size < 2:
        return 'unplottable', f'only {P.size} row(s) in range'
    s = np.sort(P)
    same = np.isclose(s[1:], s[:-1], rtol=1e-9, atol=1e-12)
    n_unique = 1 + int((~same).sum())
    if n_unique == 1:
        return 'isobar', ''
    if n_unique == P.size:
        return 'transect', ''
    return 'unplottable', (f'pressure repeats but is not constant ({n_unique} distinct values '
                           f'in {P.size} rows): neither an isobar nor a depth transect')


# --------------------------------------------------------------------------- #
# per-simulation comparison statistics
# --------------------------------------------------------------------------- #
def _rel_stats(gt, pred) -> Dict[str, float]:
    from .meltstable_comparison import _rel_stats as rs
    return rs(np.asarray(gt, dtype=np.float64), np.asarray(pred, dtype=np.float64))


def _property_rows(name: str, r: dict) -> List[dict]:
    rows = []
    if not r['properties']:
        return rows
    n = len(r['P'])
    sources = [(f'emulation_{pw}', props, pw) for pw, props in r['eos_emulated'].items()]
    sources.append(('gt_assemblage_internal_eos', r['eos_gt'], None))
    for src, props, pathway in sources:
        row = {'simulation': name, 'source': src, 'variant': r['variant'] or '-', 'n': n}
        cov = r['coverage'].get(src)
        if cov is not None:
            cov = np.asarray(cov, dtype=np.float64)
            row['n_infeasible_liquid'] = int(np.isnan(cov).sum())
            row['coverage_min'] = float(np.nanmin(cov)) if np.isfinite(cov).any() else np.nan
            row['coverage_mean'] = float(np.nanmean(cov)) if np.isfinite(cov).any() else np.nan
        for prop, truth in r['properties'].items():
            for stat, val in _rel_stats(truth, props.get(prop, np.full(n, np.nan))).items():
                row[f'{prop} {stat}'] = val
        if pathway in r['T_pred']:
            # Relative T errors in Kelvin (Celsius makes them scale-dependent).
            off = r['T_K_offset']
            for stat, val in _rel_stats(r['T'] + off, r['T_pred'][pathway] + off).items():
                row[f'T(K) {stat}'] = val
        rows.append(row)
    return rows


def _phase_rows(name: str, r: dict) -> List[dict]:
    rows = []
    n = len(r['P'])
    zeros = np.zeros(n)
    for pathway, pred in r['phases_emulated'].items():
        total = np.zeros(n)
        for phase in r['phase_order']:
            if phase == 'unrepresented':
                continue
            gt_v, pr_v = r['phases_gt'].get(phase, zeros), pred.get(phase, zeros)
            if gt_v.max() <= 0 and pr_v.max() <= 0:
                continue
            err = np.abs(pr_v - gt_v)
            total = total + err
            rows.append({'simulation': name, 'pathway': pathway, 'phase': phase,
                         'variant': r['variant'] or '-', 'gt_peak_fraction': float(gt_v.max()),
                         'mean_abs_err': float(err.mean()), 'max_abs_err': float(err.max())})
        rows.append({'simulation': name, 'pathway': pathway, 'phase': '__assemblage_L1__',
                     'variant': r['variant'] or '-', 'gt_peak_fraction': np.nan,
                     'mean_abs_err': float(total.mean()), 'max_abs_err': float(total.max())})
    return rows


# --------------------------------------------------------------------------- #
# section plots (first few plottable simulations of a group)
# --------------------------------------------------------------------------- #
_PATHWAY_LABEL = {'isothermal': 'isothermal (P, T)', 'isentropic': 'isentropic (P, S)',
                  'openox': 'fO2-buffered (P, T, logfO2-QFM)'}


def _section_axis(r: dict, kind: str):
    if kind == 'isobar':
        return r['T'], r['T_label'], f"{r['P_label'].split()[0]} = {r['P'][0]:.4g} {r['P_label'].split()[1][1:-1]}"
    return r['P'], r['P_label'], 'depth transect'


def plot_property_sections(entries, pathway: str, save_path, title: str) -> Path:
    """One row per simulation, one column per property (+ temperature for the
    isentropic pathway): simulator (black), EOS on the emulated assemblage
    (red), EOS on the GT assemblage (blue), along the run's own path."""
    from matplotlib.figure import Figure
    from ._parity_plots import EMULATED_COLOR, GT_ASSEMBLAGE_COLOR

    props = list(entries[0][1]['properties'])
    show_T = any(pathway in r['T_pred'] for _, r, _ in entries)
    ncol = len(props) + int(show_T)
    fig = Figure(figsize=(2.9 * ncol, 2.7 * len(entries)), constrained_layout=True)
    axes = fig.subplots(len(entries), ncol, squeeze=False)
    for i, (name, r, kind) in enumerate(entries):
        x, xlabel, where = _section_axis(r, kind)
        order = np.argsort(x)
        xs = x[order]
        for j, prop in enumerate(props):
            ax = axes[i][j]
            ax.plot(xs, np.asarray(r['properties'][prop])[order], 'k-', lw=1.4, label='simulator')
            if pathway in r['eos_emulated']:
                ax.plot(xs, np.asarray(r['eos_emulated'][pathway].get(prop, np.full(len(x), np.nan)))[order],
                        '--', color=EMULATED_COLOR, lw=1.3, label='EOS on emulated assemblage')
            ax.plot(xs, np.asarray(r['eos_gt'].get(prop, np.full(len(x), np.nan)))[order],
                    ':', color=GT_ASSEMBLAGE_COLOR, lw=1.5, label='EOS on GT assemblage')
            unit = r['units'].get(prop, '')
            ax.set_title(f'{prop} ({unit})' if unit else prop, fontsize=9)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25)
            if j == 0:
                ax.set_ylabel(f'{name}\n{where}', fontsize=8)
            if i == 0 and j == 0:
                ax.legend(fontsize=6)
        if show_T:
            ax = axes[i][-1]
            ax.plot(xs, r['T'][order], 'k-', lw=1.4, label='simulator')
            if pathway in r['T_pred']:
                ax.plot(xs, r['T_pred'][pathway][order], '--', color=EMULATED_COLOR, lw=1.3,
                        label='temperature model')
            ax.set_title(r['T_label'], fontsize=9)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25)
            if i == 0:
                ax.legend(fontsize=6)
    fig.suptitle(title, fontsize=10)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, facecolor='white')
    return save_path


def plot_phase_sections(entries, pathway: str, save_path, title: str) -> Path:
    """One row per simulation: stacked phase fractions along the run's path,
    simulator (left) vs emulator (right)."""
    from matplotlib.figure import Figure
    from ._melts_phase_plotting import draw_melts_phase_stack

    fig = Figure(figsize=(11, 3.0 * len(entries)), constrained_layout=True)
    axes = fig.subplots(len(entries), 2, squeeze=False)
    for i, (name, r, kind) in enumerate(entries):
        x, xlabel, where = _section_axis(r, kind)
        for ax, data, who in ((axes[i][0], r['phases_gt'], 'simulator'),
                              (axes[i][1], r['phases_emulated'][pathway], 'emulator')):
            draw_melts_phase_stack(ax, data, x, r['phase_order'], r['phase_colors'],
                                   f'{name} ({where}) -- {who}')
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(f"phase {r['phase_basis']}", fontsize=8)
    fig.suptitle(title, fontsize=10)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, facecolor='white')
    return save_path


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def evaluate_workspace(
    api,
    workspace_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    *,
    pathways: Optional[Sequence[str]] = None,
    max_runs: Optional[int] = None,
    section_runs: int = 3,
    compare: bool = True,
    write_outputs: bool = True,
    plots: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """Score ``api`` against a workspace of raw simulator output.

    This only measures: nothing here fails on accuracy thresholds.

    Parameters
    ----------
    api : MELTSAPI or HeFESToAPI
    workspace_dir : path, optional
        Root searched recursively for simulation folders (see module
        docstring). Default: the model's standard workspace (the raw runs it
        ships with, ``api.standard_workspace_dir``).
    output_dir : path, optional
        Where to write. Default: ``<root>/standardWorkspace`` for the standard
        workspace, ``<root>/customWorkspace/<workspace folder name>`` for any
        other, with ``<root>`` = ``./ngibbs_deployment_test/<model name>``.
    pathways : sequence of {'isothermal', 'isentropic'}, optional
        Closed-system emulators to score (default: every one the model has).
        For MELTS, fO2-buffered runs are also compared through the open model
        when the model has one.
    max_runs : int, optional
        Only use the first ``max_runs`` discovered simulations (natural order).
    section_runs : int
        How many simulations to draw in the section plots (the first ones
        whose path is an isobar or a depth transect; see ``classify_path``).
    compare : bool
        Run the per-simulation phase and property comparison (tables, property
        parity plots, section plots). It is fault-tolerant: a simulation that
        fails is reported and skipped, and everything else is still produced.
    write_outputs, plots, verbose : bool

    Outputs (``write_outputs``)
    ---------------------------
    Every assemblage scored in isolation, over all simulations:
        ``emulator_quality_<key>.csv``, ``plots/abundance_parity_<key>.png``
    Per-simulation comparison, over all simulations (``compare``):
        ``property_errors.csv``, ``phase_errors.csv``,
        ``plots/property_parity_<key>.png`` (red: EOS on the emulated
        assemblage; blue: EOS on the GT assemblage; each with R^2)
    Sections, first ``section_runs`` plottable simulations:
        ``plots/property_section_<key>.png``, ``plots/phase_section_<key>.png``
    plus ``benchmark_manifest.json`` (what was used, dropped, skipped, and
    why) and ``emulator_quality_legend.txt``. ``<key>`` is the pathway, or
    ``<variant>_<pathway>`` for MELTS (Cr / NoCr routed per simulation).

    Returns
    -------
    dict with 'quality_metrics' ({key: DataFrame}), 'property_errors' and
    'phase_errors' (DataFrames, possibly empty), 'n_assemblages', 'simulations',
    'skipped_simulations', 'manifest', 'is_standard', 'workspace_dir', and
    'output_dir' (when written).
    """
    standard = standard_workspace(api)
    if workspace_dir is None:
        if standard is None:
            raise FileNotFoundError(f'{type(api).__name__} ships no standard workspace; '
                                    'pass the path of a workspace to evaluate.')
        workspace_dir = standard
    workspace_dir = Path(workspace_dir).expanduser().resolve()
    if not workspace_dir.is_dir():
        raise NotADirectoryError(f'Workspace directory does not exist: {workspace_dir}')
    is_standard = standard is not None and workspace_dir == standard.expanduser().resolve()
    if pathways is None:
        pathways = available_pathways(api)
    bad = [p for p in pathways if p not in _EMULATOR_ATTR]
    if bad:
        raise ValueError(f'Unknown pathway(s) {bad}; expected {list(_EMULATOR_ATTR)}')

    adapter = _adapter_for(api)
    runs = adapter.find_runs(workspace_dir)
    if not runs:
        raise FileNotFoundError(f'No {adapter.kind} simulation folders found under {workspace_dir}')
    n_found = len(runs)
    if max_runs is not None:
        runs = runs[:max_runs]

    def rel(p: Path) -> str:
        return str(p.relative_to(workspace_dir)) if p != workspace_dir else workspace_dir.name

    if verbose:
        kind = 'standard' if is_standard else 'custom'
        print(f'[workspace] {model_label(api)}: {kind} workspace {workspace_dir} ({len(runs)} simulations)')

    # ---- every assemblage in isolation --------------------------------------
    skipped: Dict[str, str] = {}
    groups: Dict[str, dict] = {}
    for i, run_dir in enumerate(runs):
        name = rel(run_dir)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                parsed = adapter.parse(run_dir)
        except Exception as e:
            skipped[name] = f'{type(e).__name__}: {e}'
            continue
        for pathway in pathways:
            try:
                table, headers = adapter.inputs(parsed, pathway)
                variant, sub = adapter.route(table, headers)
                emulator = getattr(sub, _EMULATOR_ATTR[pathway], None)
                if emulator is None:
                    raise RuntimeError(f'{pathway} emulator not loaded')
                gt = adapter.ground_truth(parsed, emulator)
            except Exception as e:
                skipped[f'{name} [{pathway}]'] = f'{type(e).__name__}: {e}'
                continue

            key = f'{variant}_{pathway}' if variant else pathway
            g = groups.setdefault(key, {
                'variant': variant or None, 'pathway': pathway, 'sub': sub, 'emulator': emulator,
                'headers': headers, 'tables': [], 'gt': [], 'simulations': {},
                'n_rows_parsed': 0, 'n_rows_dropped_unmodelled_phase': 0,
                'n_rows_dropped_out_of_pressure_range': 0, 'n_rows_dropped_invalid': 0,
                'unmodelled_phases': {},
            })
            keep = gt['valid'] & np.all(np.isfinite(table), axis=1)
            g['n_rows_dropped_invalid'] += int((~keep).sum())
            unmod_rows = np.zeros(parsed['n'], dtype=bool)
            for phase, mask in gt['unmodelled'].items():
                unmod_rows |= mask
                g['unmodelled_phases'][phase] = g['unmodelled_phases'].get(phase, 0) + int((mask & keep).sum())
            g['n_rows_dropped_unmodelled_phase'] += int((unmod_rows & keep).sum())
            keep &= ~unmod_rows
            bounds = _pressure_bounds(emulator)
            if bounds is not None:
                in_range = (table[:, 0] >= bounds[0]) & (table[:, 0] <= bounds[1])
                g['n_rows_dropped_out_of_pressure_range'] += int((keep & ~in_range).sum())
                keep &= in_range
            g['n_rows_parsed'] += parsed['n']
            if not keep.any():
                continue
            g['tables'].append(table[keep])
            g['gt'].append({
                'present': gt['present'][keep], 'mass_wtpct': gt['mass_wtpct'][keep],
                'comp_wtpct': {p: v[keep] for p, v in gt['comp_wtpct'].items()},
                'n': int(keep.sum()),
            })
            g['simulations'][name] = int(keep.sum())
        if verbose and (i + 1) % 100 == 0:
            print(f'[workspace] parsed {i + 1}/{len(runs)} simulations')

    quality: Dict[str, pd.DataFrame] = {}
    group_meta: Dict[str, dict] = {}
    parity_data: Dict[str, tuple] = {}
    for key, g in groups.items():
        meta = {k: g[k] for k in ('variant', 'pathway', 'n_rows_parsed', 'n_rows_dropped_unmodelled_phase',
                                  'n_rows_dropped_out_of_pressure_range', 'n_rows_dropped_invalid',
                                  'unmodelled_phases', 'simulations')}
        meta['n_simulations'] = len(g['simulations'])
        meta['n_assemblages'] = int(sum(x['n'] for x in g['gt']))
        # Identifies exactly which assemblages were scored, so a later comparison
        # can tell "the model changed" apart from "the workspace changed".
        meta['dataset_signature'] = hashlib.sha1(
            json.dumps(sorted(g['simulations'].items())).encode()).hexdigest()
        meta['mass_balance'] = getattr(g['emulator'], 'mass_balance', None)
        group_meta[key] = meta
        if not g['tables']:
            continue
        table = np.concatenate(g['tables']).astype(np.float32)
        sub, emulator = g['sub'], g['emulator']
        features, _ = sub.parse_input(table, headers=g['headers'])
        with torch.no_grad():
            out = sub._staged_forward(
                emulator.forwardMB, features, None,
                Normalize=True, optimize_masses=True, protect_opx=False, outputs=list(_FORWARD_OUTPUTS),
            )
        comp_pred, mass_pred = out['phase_tables']
        mi = emulator.ml_indexer
        gt_comp = {}
        for phase in mi.comp_phasedict:
            parts = [x['comp_wtpct'][phase] if phase in x['comp_wtpct']
                     else np.zeros((x['n'], len(mi.Oxides))) for x in g['gt']]
            gt_comp[phase] = np.concatenate(parts)
        gt_mass = np.concatenate([x['mass_wtpct'] for x in g['gt']])
        pred_mass = _to_np(mass_pred).astype(np.float64)
        quality[key], phase_order = phase_quality_table(
            mi,
            gt_present=np.concatenate([x['present'] for x in g['gt']]),
            pred_present=_to_np(out['phase_present']) > 0.5,
            gt_mass_wtpct=gt_mass,
            pred_mass_wtpct=pred_mass,
            gt_comp_wtpct=gt_comp,
            pred_comp_wtpct=_to_np(comp_pred).astype(np.float64),
            recon_residual=_to_np(out['reconstruction_residual']).reshape(-1),
        )
        parity_data[key] = (gt_mass, pred_mass, mi, phase_order)
        if verbose:
            print(f"[workspace] [{key}]: {meta['n_assemblages']:,} assemblages "
                  f"from {meta['n_simulations']} simulations "
                  f"(dropped: {meta['n_rows_dropped_unmodelled_phase']} unmodelled-phase, "
                  f"{meta['n_rows_dropped_out_of_pressure_range']} out-of-range, "
                  f"{meta['n_rows_dropped_invalid']} invalid)")

    # ---- per-simulation phase / property comparison --------------------------
    comparisons: Dict[str, dict] = {}         # name -> per-run result
    comparison_errors: Dict[str, str] = {}
    property_errors_by_run: Dict[str, str] = {}
    if compare:
        for i, run_dir in enumerate(runs):
            name = rel(run_dir)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    r = adapter.compare(run_dir, pathways)
            except Exception as e:  # noqa: BLE001 -- reported, the rest still runs
                comparison_errors[name] = f'{type(e).__name__}: {e}'
                continue
            comparisons[name] = r
            if r['property_error']:
                property_errors_by_run[name] = r['property_error']
            if verbose and (i + 1) % 50 == 0:
                print(f'[workspace] compared {i + 1}/{len(runs)} simulations')
        if comparison_errors or property_errors_by_run:
            msg = (f'[workspace] phase/property comparison: {len(comparison_errors)} simulation(s) failed, '
                   f'{len(property_errors_by_run)} without properties; other outputs are unaffected '
                   '(reasons in benchmark_manifest.json).')
            first = next(iter({**comparison_errors, **property_errors_by_run}.items()))
            warnings.warn(f'{msg} First: {first[0]}: {first[1]}')

    prop_rows = [row for name, r in comparisons.items() for row in _property_rows(name, r)]
    phase_rows = [row for name, r in comparisons.items() for row in _phase_rows(name, r)]
    property_errors = (pd.DataFrame(prop_rows).set_index(['simulation', 'source'])
                       if prop_rows else pd.DataFrame())
    phase_errors = (pd.DataFrame(phase_rows).set_index(['simulation', 'pathway', 'phase'])
                    if phase_rows else pd.DataFrame())

    # group comparisons by '<variant>_<pathway>' (or '<pathway>'), and classify paths
    section_groups: Dict[str, List[str]] = {}
    path_kind: Dict[str, Tuple[str, str]] = {}
    for name, r in comparisons.items():
        path_kind[name] = classify_path(r['P'])
        for pathway in r['phases_emulated']:
            key = f"{r['variant']}_{pathway}" if r['variant'] else pathway
            section_groups.setdefault(key, []).append(name)
    sections_meta = {}
    for key, names in section_groups.items():
        plottable = [n for n in names if path_kind[n][0] != 'unplottable']
        sections_meta[key] = {
            'drawn': plottable[:section_runs],
            'path_kind': {n: path_kind[n][0] for n in names},
            'unplottable': {n: path_kind[n][1] for n in names if path_kind[n][0] == 'unplottable'},
        }

    benchmarked = sorted({s for m in group_meta.values() for s in m['simulations']}, key=_natural_key)
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'api': type(api).__name__,
        'model': model_label(api),
        'model_dir': str(getattr(api, '_model_dir', None) or getattr(api, 'home_dir', '')),
        'workspace_dir': str(workspace_dir),
        'workspace_kind': 'standard' if is_standard else 'custom',
        'simulator': adapter.kind,
        'pathways': list(pathways),
        'n_simulations_found': n_found,
        'n_simulations_benchmarked': len(benchmarked),
        'n_assemblages': {k: m['n_assemblages'] for k, m in group_meta.items()},
        'groups': group_meta,
        'skipped_simulations': skipped,
        'comparison': {
            'n_simulations_compared': len(comparisons),
            'failed': comparison_errors,
            'without_properties': property_errors_by_run,
            'sections': sections_meta,
        },
    }
    if verbose and skipped:
        print(f'[workspace] skipped {len(skipped)} simulation(s)/pathway(s); see manifest')

    result = {'quality_metrics': quality, 'property_errors': property_errors,
              'phase_errors': phase_errors, 'n_assemblages': manifest['n_assemblages'],
              'simulations': benchmarked, 'skipped_simulations': skipped, 'manifest': manifest,
              'is_standard': is_standard, 'workspace_dir': workspace_dir}
    if not write_outputs:
        return result

    if output_dir is None:
        root = default_output_root(api)
        output_dir = (root / STANDARD_WORKSPACE if is_standard
                      else root / CUSTOM_WORKSPACE / workspace_dir.name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, df in quality.items():
        df.to_csv(output_dir / f'emulator_quality_{key}.csv', na_rep=NA_REP)
    (output_dir / 'emulator_quality_legend.txt').write_text(legend_text())
    if not property_errors.empty:
        property_errors.to_csv(output_dir / 'property_errors.csv', na_rep=NA_REP)
    if not phase_errors.empty:
        phase_errors.to_csv(output_dir / 'phase_errors.csv', na_rep=NA_REP)
    result['output_dir'] = output_dir

    if plots:
        result['figures'] = _write_plots(api, output_dir, workspace_dir, manifest, parity_data,
                                         comparisons, section_groups, sections_meta, path_kind,
                                         section_runs, verbose)
    (output_dir / 'benchmark_manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    if verbose:
        print(f'[workspace] wrote {len(quality)} quality table(s)'
              + (', property_errors.csv' if not property_errors.empty else '')
              + (', phase_errors.csv' if not phase_errors.empty else '')
              + f' -> {output_dir}')
    return result


def _write_plots(api, output_dir, workspace_dir, manifest, parity_data, comparisons,
                 section_groups, sections_meta, path_kind, section_runs, verbose) -> Dict[str, Path]:
    """Every plot for one workspace; each is independent, and a failure only
    warns (and is recorded in the manifest)."""
    from ._parity_plots import plot_abundance_parity, plot_property_parity

    plots_dir = output_dir / 'plots'
    model = model_label(api)
    figs: Dict[str, Path] = {}
    failures: Dict[str, str] = {}

    def attempt(label, fn):
        try:
            figs[label] = fn()
        except Exception as e:  # noqa: BLE001
            failures[label] = f'{type(e).__name__}: {e}'
            warnings.warn(f'[workspace] {label} plot failed: {failures[label]}')

    for key, (gt_mass, pred_mass, mi, phase_order) in parity_data.items():
        attempt(f'abundance_parity_{key}', lambda: plot_abundance_parity(
            gt_mass, pred_mass, mi, phase_order, plots_dir / f'abundance_parity_{key}.png',
            title=f'{model} [{key}] on {workspace_dir.name}: '
                  f'{manifest["n_assemblages"][key]:,} assemblages'))

    for key, names in section_groups.items():
        pathway = key.split('_', 1)[1] if '_' in key else key
        with_props = [n for n in names
                      if comparisons[n]['properties'] and pathway in comparisons[n]['eos_emulated']]
        if with_props:
            def parity(with_props=with_props, pathway=pathway, key=key):
                first = comparisons[with_props[0]]
                cat = lambda get: {p: np.concatenate([np.asarray(get(comparisons[n]).get(p, np.full(
                    len(comparisons[n]['P']), np.nan)), dtype=np.float64) for n in with_props])
                    for p in first['properties']}
                truth = cat(lambda r: r['properties'])
                emulated = cat(lambda r: r['eos_emulated'][pathway])
                gt_asm = cat(lambda r: r['eos_gt'])
                n_rows = sum(len(comparisons[n]['P']) for n in with_props)
                path = plots_dir / f'property_parity_{key}.png'
                manifest['comparison']['sections'][key]['property_r2'] = plot_property_parity(
                    truth, emulated, gt_asm, path, units=first['units'],
                    title=f'{model} [{key}] on {workspace_dir.name}: built-in EOS vs simulator, '
                          f'{n_rows:,} assemblages from {len(with_props)} simulations')
                return path
            attempt(f'property_parity_{key}', parity)

        drawn = [(n, comparisons[n], path_kind[n][0]) for n in sections_meta[key]['drawn']]
        if not drawn:
            if verbose:
                print(f'[workspace] [{key}]: no simulation is an isobar or depth transect; '
                      'section plots skipped')
            continue
        title = f'{model} [{key}], {_PATHWAY_LABEL.get(pathway, pathway)}'
        attempt(f'phase_section_{key}', lambda: plot_phase_sections(
            drawn, pathway, plots_dir / f'phase_section_{key}.png', f'{title}: phase {drawn[0][1]["phase_basis"]}'))
        drawn_props = [d for d in drawn if d[1]['properties'] and pathway in d[1]['eos_emulated']]
        if drawn_props:
            attempt(f'property_section_{key}', lambda: plot_property_sections(
                drawn_props, pathway, plots_dir / f'property_section_{key}.png', f'{title}: bulk properties'))

    manifest['comparison']['plot_failures'] = failures
    return figs
