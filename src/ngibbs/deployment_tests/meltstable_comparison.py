"""Deployable MELTStable comparisons for the HeFESTo emulator.

The packaged, ``builder``-free equivalent of
``scripts/property_comparison_meltstable.py`` and
``scripts/phase_comparison_meltstable.py``.  Ground truth is a directory of
MELTStable-format adiabat CSVs (one bulk composition each, swept over pressure);
the shipped set lives in ``deployment_tests/HeFESToAdiabatStandards/``.

Two entry points, both taking an already-constructed ``HeFESToAPI``:

* ``run_meltstable_property_comparison`` — rho / VP / VS / S / Cp / KS / alpha
  from three sources (isentropic emulation, isothermal emulation, real assemblage
  through the internal vectorised EOS) against the CSV's ``(System_main)``
  columns; writes the two comparison figures and a wide error-stats table with
  three rows per CSV.
* ``run_meltstable_phase_comparison`` — stacked phase-abundance diagrams
  (GT vs emulator, both pathways) plus a per-phase phase-fraction error table.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ngibbs.engine.EOS_arithmetic.hefesto_vec import compute_within_phase_frac
from ._phase_plotting import (
    phase_colors, build_ordered_phases, draw_phase_stack, flagged_pressure_spans,
)

ELEMENT_KEYS = ['Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr', 'O']

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_TABLE_DIR = PACKAGE_DIR / 'HeFESToAdiabatStandards'

SYSTEM_SUFFIX = '(System_main)'
P_COL = f'P(GPa){SYSTEM_SUFFIX}'
T_COL = f'T(K){SYSTEM_SUFFIX}'
S_COL = f'S(J/g/K){SYSTEM_SUFFIX}'

# EOS property key -> (MELTStable column token, short name, EOS->table unit scale).
# The scale brings the internally-computed EOS value into the table column's
# units: 1.0 everywhere except thermal expansivity, where the EOS returns
# ``alptot`` in 1/K and the table stores alptot * 1e5.
PROPERTY_MAP = {
    'rho':     ('rho(g/cm^3)',     'rho', 1.0),
    'Vp_fast': ('VP(km/s)',        'VP',  1.0),
    'Vs':      ('VS(km/s)',        'VS',  1.0),
    'S':       ('S(J/g/K)',        'S',   1.0),
    'cptot':   ('cp(J/g/K)',       'Cp',  1.0),
    'KStot':   ('KS(GPa)',         'KS',  1.0),
    'alptot':  ('alpha(1e5_K^-1)', 'alpha', 1.0e5),
}

MISSING_PHASE_FRACTION_THRESHOLD = 1.0e-3
_TOTAL_MOLES_RE = re.compile(r'total \(moles\)\((.+)\)$')


# --------------------------------------------------------------------------- #
# shared loading helpers
# --------------------------------------------------------------------------- #
def _to_numpy(x) -> np.ndarray:
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def find_tables(table_dir, tables: List[str] = None) -> List[Path]:
    table_dir = Path(table_dir)
    if tables:
        out = []
        for t in tables:
            p = Path(t)
            if not p.exists():
                p = table_dir / t
            if not p.exists() and not str(t).endswith('.csv'):
                p = table_dir / f'{t}.csv'
            if not p.exists():
                raise FileNotFoundError(f'MELTStable CSV not found: {t}')
            out.append(p)
        return out
    if not table_dir.is_dir():
        raise NotADirectoryError(f'MELTStable table directory does not exist: {table_dir}')
    out = sorted(table_dir.glob('*.csv'))
    if not out:
        raise FileNotFoundError(f'No *.csv MELTStable files in {table_dir}')
    return out


def _load_composition(df: pd.DataFrame) -> Dict[str, float]:
    comp = {}
    for k in ELEMENT_KEYS:
        col = f'{k}(Bulk_comp_elements)'
        if col not in df.columns:
            raise KeyError(f'MELTStable table missing bulk-composition column {col!r}')
        comp[k] = float(pd.to_numeric(df[col], errors='coerce').iloc[0])
    return comp


def _load_component_moles(df: pd.DataFrame, indexer) -> np.ndarray:
    """Extensive component moles aligned to ``indexer.label_names``.

    HeFESTo MELTStable phases store each endmember as an extensive mole count in
    a ``'<endmember>(<phase>)'`` column (they sum to ``'total (moles)(<phase>)'``).
    Iterating ``label_indices`` phase-by-phase keeps the two ``magnetite``
    components (spinel vs ferropericlase) in their correct slots.
    """
    n = len(df)
    cm = np.zeros((n, len(indexer.label_names)), dtype=np.float64)
    for phase, idxs in indexer.label_indices.items():
        for j in np.asarray(idxs, dtype=np.int64):
            col = f'{indexer.label_names[j]}({phase})'
            if col in df.columns:
                cm[:, j] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
    return cm


def _expand_chem_out_to_components(chem_out: np.ndarray, indexer) -> np.ndarray:
    B = chem_out.shape[0]
    full = np.ones((B, len(indexer.label_names)), dtype=np.float64)
    vc = np.asarray(indexer.compositionally_variable_subset, dtype=np.int64)
    full[:, vc] = chem_out
    return full


def _phase_totals(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    out = {}
    for col in df.columns:
        m = _TOTAL_MOLES_RE.match(col)
        if m:
            out[m.group(1)] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
    return out


def _read_conditions(csv_path: Path):
    df = pd.read_csv(csv_path)
    for col in (P_COL, T_COL, S_COL):
        if col not in df.columns:
            raise KeyError(f'{csv_path.name} missing required column {col!r}')
    P = pd.to_numeric(df[P_COL], errors='coerce').to_numpy(dtype=np.float64)
    T = pd.to_numeric(df[T_COL], errors='coerce').to_numpy(dtype=np.float64)
    S = pd.to_numeric(df[S_COL], errors='coerce').to_numpy(dtype=np.float64)
    return df, P, T, S


def _restrict_to_training_pressure_range(api, df, P, T, S):
    lower, upper = api.training_pressure_bounds('both')
    keep = np.isfinite(P) & (P >= lower) & (P <= upper)
    if not np.any(keep):
        raise ValueError(
            f'{len(P)} rows in the MELTStable table fall outside the model '
            f'pressure range [{lower:g}, {upper:g}] GPa'
        )
    if not np.all(keep):
        print(
            f'[MELTStable] restricting pressure range to [{lower:g}, {upper:g}] '
            f'GPa: keeping {int(keep.sum())}/{len(keep)} rows'
        )
    return df.loc[keep].reset_index(drop=True), P[keep], T[keep], S[keep]


def _feature_block(P, second, composition):
    comp_headers = list(composition.keys())
    comp_block = np.tile(
        np.array([[composition[k] for k in comp_headers]], dtype=np.float32), (len(P), 1)
    )
    return np.column_stack([P, second, comp_block]).astype(np.float32), comp_headers


def _normalize_phase_moles(phase_moles: np.ndarray) -> np.ndarray:
    row_totals = phase_moles.sum(axis=1, keepdims=True)
    row_totals = np.where(row_totals == 0, 1.0, row_totals)
    return phase_moles / row_totals


# --------------------------------------------------------------------------- #
# property comparison
# --------------------------------------------------------------------------- #
def _property_names() -> list:
    return list(PROPERTY_MAP.keys())


def _eval_assemblage(api, component_moles, P, T, within_phase_frac):
    props = api.get_property_hefesto_vectorized_from_assemblage(
        torch.tensor(component_moles, dtype=torch.float64),
        torch.tensor(np.stack([P, T], axis=1), dtype=torch.float64),
        property_names=_property_names(),
        within_phase_frac=torch.tensor(within_phase_frac, dtype=torch.float64),
    )
    return props


def _emulator_assemblage(api, P, second, composition, second_header, isentropic):
    features, comp_headers = _feature_block(P, second, composition)
    headers = ['P(GPa)(System_main)', second_header] + comp_headers
    outputs = ['component_moles', 'chem_out'] + (['temperature'] if isentropic else [])
    with torch.no_grad():
        out = api.ForwardMB(features, headers=headers, outputs=outputs)
    cm = _to_numpy(out['component_moles']).astype(np.float64)
    indexer = (api.isentropic_emulator if isentropic else api.isothermal_emulator).ml_indexer
    wpf = _expand_chem_out_to_components(_to_numpy(out['chem_out']).astype(np.float64), indexer)
    T_em = _to_numpy(out['temperature']).reshape(-1).astype(np.float64) if isentropic else None
    return cm, wpf, T_em


def _rel_stats(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    finite = np.isfinite(gt) & np.isfinite(pred)
    if not finite.any():
        return {'mae': np.nan, 'mean_rel%': np.nan, 'p95_rel%': np.nan, 'max_rel%': np.nan}
    ae = np.abs(pred[finite] - gt[finite])
    re_ = ae / np.maximum(np.abs(gt[finite]), 1e-12)
    return {
        'mae': float(ae.mean()),
        'mean_rel%': float(re_.mean() * 100),
        'p95_rel%': float(np.percentile(re_, 95) * 100),
        'max_rel%': float(re_.max() * 100),
    }


def run_meltstable_property_comparison(
    api,
    table_dir=None,
    out_dir='.',
    *,
    tables: List[str] = None,
    stats_name: str = 'meltstable_property_errors.csv',
    fig_prefix: str = 'meltstable_property_comparison',
) -> Dict[str, object]:
    """Compare bulk EOS properties (incl. thermal expansivity) against the
    MELTStable adiabat standards.  Writes two figures and one wide stats table
    (three rows per CSV: isentropic emulation, isothermal emulation, real
    assemblage through the internal EOS).
    """
    table_dir = Path(table_dir) if table_dir is not None else DEFAULT_TABLE_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = find_tables(table_dir, tables)
    indexer = api.isothermal_emulator.ml_indexer

    results = []
    stat_rows = []
    for csv_path in chosen:
        df, P, T, S = _read_conditions(csv_path)
        df, P, T, S = _restrict_to_training_pressure_range(api, df, P, T, S)
        composition = _load_composition(df)
        name = csv_path.stem

        gt_cm = _load_component_moles(df, indexer)
        gt_wpf = compute_within_phase_frac(gt_cm, list(indexer.label_indices.values()))
        gt_props = _eval_assemblage(api, gt_cm, P, T, gt_wpf)

        iso_cm, iso_wpf, _ = _emulator_assemblage(api, P, T, composition, 'T(K)(System_main)', False)
        iso_props = _eval_assemblage(api, iso_cm, P, T, iso_wpf)

        isen_cm, isen_wpf, T_em = _emulator_assemblage(
            api, P, S, composition, 'S(J/g/K)(System_main)', True)
        isen_props = _eval_assemblage(api, isen_cm, P, T_em, isen_wpf)

        table_vals = {}
        for key, (tok, short, scale) in PROPERTY_MAP.items():
            table_vals[key] = pd.to_numeric(
                df[f'{tok}{SYSTEM_SUFFIX}'], errors='coerce').to_numpy(dtype=np.float64)

        sources = {
            'emulation_isentropic': isen_props,
            'emulation_isothermal': iso_props,
            'gt_assemblage_internal_eos': gt_props,
        }
        for src_name, props in sources.items():
            row = {'table': name, 'source': src_name, 'n': int(len(P))}
            for key, (tok, short, scale) in PROPERTY_MAP.items():
                s = _rel_stats(table_vals[key],
                               np.asarray(props[key], dtype=np.float64) * scale)
                for stat, val in s.items():
                    row[f'{short} {stat}'] = val
            if src_name == 'emulation_isentropic':
                for stat, val in _rel_stats(T, T_em).items():
                    row[f'T {stat}'] = val
            stat_rows.append(row)

        results.append({
            'name': name, 'P': P, 'T': T, 'T_em': T_em, 'df': df,
            'table_vals': table_vals,
            'gt': gt_props, 'iso': iso_props, 'isen': isen_props,
        })

    stats = pd.DataFrame(stat_rows).set_index(['table', 'source'])
    stats_path = out_dir / stats_name
    stats.to_csv(stats_path, na_rep='--')

    figs = {
        'isothermal': _plot_properties(results, 'isothermal', out_dir / f'{fig_prefix}_isothermal.png'),
        'isentropic': _plot_properties(results, 'isentropic', out_dir / f'{fig_prefix}_isentropic.png'),
    }
    return {'property_errors': stats, 'stats_path': stats_path, 'figures': figs}


def _plot_properties(results, mode, save_path):
    prop_keys = list(PROPERTY_MAP.keys())
    show_T = mode == 'isentropic'
    n_cols = len(prop_keys) + (1 if show_T else 0)
    n_rows = len(results)
    col_w = 3.0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(col_w * n_cols, 3.2 * n_rows), squeeze=False)
    src_key = 'isen' if mode == 'isentropic' else 'iso'

    for r, res in enumerate(results):
        P = res['P']
        for c, key in enumerate(prop_keys):
            ax = axes[r][c]
            _, short, scale = PROPERTY_MAP[key]
            ax.plot(P, res['table_vals'][key], 'k-', lw=1.5, label='MELTStable (GT)')
            ax.plot(P, np.asarray(res[src_key][key]) * scale, 'r--', lw=1.4,
                    label='Emulator (pred. assemblage)')
            ax.plot(P, np.asarray(res['gt'][key]) * scale, 'b:', lw=1.6,
                    label='GT assemblage + internal EOS')
            ax.set_xlabel('P (GPa)')
            ax.set_ylabel(short if c > 0 else f"{res['name']}\n{short}", fontsize=9)
            ax.grid(True, alpha=0.25)
            if r == 0:
                ax.set_title(short, fontsize=10)
            if r == 0 and c == len(prop_keys) - 1:
                ax.legend(fontsize=7, loc='best')
        if show_T:
            ax = axes[r][-1]
            ax.plot(P, res['T'], 'k-', lw=1.5, label='MELTStable (GT)')
            ax.plot(P, res['T_em'], 'r--', lw=1.4, label='Emulator T(P,S)')
            ax.set_xlabel('P (GPa)')
            ax.set_ylabel('T (K)')
            ax.grid(True, alpha=0.25)
            if r == 0:
                ax.set_title('T (K)', fontsize=10)
                ax.legend(fontsize=7, loc='best')

    lbl = 'Isothermal (P, T)' if mode == 'isothermal' else 'Isentropic (P, S)'
    fig.suptitle(f'{lbl} pathway: emulator vs MELTStable ground truth', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=150, facecolor='white')
    plt.close(fig)
    return Path(save_path)


# --------------------------------------------------------------------------- #
# phase comparison
# --------------------------------------------------------------------------- #
def _gt_phase_fractions(df, mass_phasedict):
    totals = _phase_totals(df)
    n = len(df)
    n_phases = max(mass_phasedict.values()) + 1
    pm = np.zeros((n, n_phases), dtype=np.float64)
    for phase, col in mass_phasedict.items():
        if phase in totals:
            pm[:, col] = totals[phase]
    grand = np.sum(np.stack(list(totals.values()), axis=1), axis=1) if totals else np.ones(n)
    grand = np.where(grand == 0, 1.0, grand)
    return pm / grand[:, None]


def _check_missing_phases(df, mass_phasedict, P):
    totals = _phase_totals(df)
    n = len(df)
    total_all = np.sum(np.stack(list(totals.values()), axis=1), axis=1) if totals else np.zeros(n)
    safe = np.where(total_all > 0, total_all, 1.0)
    represented = set(mass_phasedict)
    missing_moles = np.zeros(n)
    unresolved = {}
    for phase, moles in totals.items():
        if phase in represented or not np.any(moles > 0):
            continue
        missing_moles += moles
        frac = np.where(total_all > 0, moles / safe, 0.0)
        pk = int(np.argmax(frac))
        unresolved[phase] = {
            'n_active_rows': int(np.sum(moles > 0)),
            'peak_fraction': float(frac[pk]),
            'P_at_peak': float(P[pk]),
            'significant': float(frac[pk]) >= MISSING_PHASE_FRACTION_THRESHOLD,
        }
    return {
        'missing_fraction': np.where(total_all > 0, missing_moles / safe, 0.0),
        'unresolved': unresolved,
    }


def _emulator_phase_fractions(api, P, second, composition, second_header):
    features, comp_headers = _feature_block(P, second, composition)
    headers = ['P(GPa)(System_main)', second_header] + comp_headers
    with torch.no_grad():
        out = api.ForwardMB(features, headers=headers, outputs=['phase_moles'])
    return _normalize_phase_moles(_to_numpy(out['phase_moles']).astype(np.float64))


def run_meltstable_phase_comparison(
    api,
    table_dir=None,
    out_dir='.',
    *,
    tables: List[str] = None,
    stats_name: str = 'meltstable_phase_errors.csv',
    fig_prefix: str = 'meltstable_phase_comparison',
) -> Dict[str, object]:
    table_dir = Path(table_dir) if table_dir is not None else DEFAULT_TABLE_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = find_tables(table_dir, tables)

    indexer = api.isothermal_emulator.ml_indexer
    mass_phasedict = indexer.mass_phasedict
    ordered_phases = build_ordered_phases(mass_phasedict)
    colors = phase_colors()

    results = []
    stat_rows = []
    for csv_path in chosen:
        df, P, T, S = _read_conditions(csv_path)
        df, P, T, S = _restrict_to_training_pressure_range(api, df, P, T, S)
        composition = _load_composition(df)
        name = csv_path.stem
        gt_pf = _gt_phase_fractions(df, mass_phasedict)
        iso_pf = _emulator_phase_fractions(api, P, T, composition, 'T(K)(System_main)')
        isen_pf = _emulator_phase_fractions(api, P, S, composition, 'S(J/g/K)(System_main)')
        coverage = _check_missing_phases(df, mass_phasedict, P)

        inv = {v: k for k, v in mass_phasedict.items()}
        for pathway, pred in (('isothermal', iso_pf), ('isentropic', isen_pf)):
            abs_err = np.abs(pred - gt_pf)
            for col in range(gt_pf.shape[1]):
                if gt_pf[:, col].max() <= 0 and pred[:, col].max() <= 0:
                    continue
                stat_rows.append({
                    'table': name, 'pathway': pathway, 'phase': inv.get(col, f'col{col}'),
                    'gt_peak_fraction': float(gt_pf[:, col].max()),
                    'mean_abs_err': float(abs_err[:, col].mean()),
                    'max_abs_err': float(abs_err[:, col].max()),
                })
            stat_rows.append({
                'table': name, 'pathway': pathway, 'phase': '__assemblage_L1__',
                'gt_peak_fraction': np.nan,
                'mean_abs_err': float(abs_err.sum(axis=1).mean()),
                'max_abs_err': float(abs_err.sum(axis=1).max()),
            })

        results.append({
            'name': name, 'P': P, 'gt': gt_pf, 'isothermal': iso_pf, 'isentropic': isen_pf,
            'coverage': coverage, 'mass_phasedict': mass_phasedict,
            'ordered_phases': ordered_phases, 'colors': colors,
        })

    stats = pd.DataFrame(stat_rows).set_index(['table', 'pathway', 'phase'])
    stats_path = out_dir / stats_name
    stats.to_csv(stats_path, na_rep='--')

    figs = {
        'isothermal': _plot_phases(results, 'isothermal', out_dir / f'{fig_prefix}_isothermal.png'),
        'isentropic': _plot_phases(results, 'isentropic', out_dir / f'{fig_prefix}_isentropic.png'),
    }
    unresolved = {r['name']: r['coverage']['unresolved'] for r in results if r['coverage']['unresolved']}
    return {'phase_errors': stats, 'stats_path': stats_path, 'figures': figs,
            'unrepresented_phases': unresolved}


def _plot_phases(results, mode, save_path):
    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 3.4 * n_rows), squeeze=False)
    for r, res in enumerate(results):
        pressures = res['P']
        row_idx = np.argsort(pressures)
        cov = res['coverage']
        gt_title = f"{res['name']} -- HeFESTo (GT)"
        if cov['unresolved']:
            gt_title += '  [UNREPRESENTED PHASE]'
        draw_phase_stack(axes[r][0], res['gt'], row_idx, pressures,
                         res['ordered_phases'], res['mass_phasedict'], res['colors'], gt_title)
        draw_phase_stack(axes[r][1], res[mode], row_idx, pressures,
                         res['ordered_phases'], res['mass_phasedict'], res['colors'],
                         f"{res['name']} -- Emulator ({mode})")
        flagged = cov['missing_fraction'][row_idx] > MISSING_PHASE_FRACTION_THRESHOLD
        if flagged.any():
            ps = pressures[row_idx]
            pad = 0.15 * (ps[-1] - ps[0]) / max(len(ps) - 1, 1)
            for a, b in flagged_pressure_spans(ps, flagged):
                if a == b:
                    a, b = a - pad, b + pad
                axes[r][0].axvspan(a, b, facecolor='none', edgecolor='black',
                                   hatch='//', alpha=0.6, zorder=5)
        for ax in axes[r]:
            ax.set_xlabel('P (GPa)')
    lbl = 'Isothermal (P, T)' if mode == 'isothermal' else 'Isentropic (P, S)'
    fig.suptitle(f'Phase abundances -- {lbl} pathway: GT vs emulator', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=150, facecolor='white')
    plt.close(fig)
    return Path(save_path)
