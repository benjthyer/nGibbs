"""Deployable MELTStable comparisons for the HeFESTo emulator.

The packaged, ``builder``-free equivalent of
``scripts/property_comparison_meltstable.py`` and
``scripts/phase_comparison_meltstable.py``.  Ground truth is a directory of
RAW HeFESTo simulation output folders (one per standard rock, each an
isobaric/isentropic-sweep run containing ``control`` + ``fort.56`` + ``fort.99``),
read straight off disk exactly the way ``melts_comparison.py`` reads raw
alphaMELTS output -- never a pre-baked MELTStable CSV. Each HeFESTo model ships
its set in its own folder, ``<model>/standards/<rock>/`` (HeFESToEarth and
HeFESToMars each run along their planet's own P-T conditions).

The low-level fort.*/``control`` parsers (``_parse_control_file``,
``_parse_fort56``, ``load_fort99_componentMoles``, ...) live in
``ngibbs.utils.file_utils`` -- the same "builder-free" module
``src/builder/HeFESTo/HeFESTo_functions.py`` itself imports them from -- so this
stays dependency-free of the dev-only ``builder`` package while sharing its
exact parsing logic.

Bulk composition comes from the ``control`` file's element-mole block: cation
moles plus oxygen, on HeFESTo's fixed total-moles basis (commonly ~24 for mantle
compositions -- see ``HeFESTo_functions._normalize_total_moles``). HeFESTo has no
separate ferric-iron entry; Fe3+ is implied by the oxygen in excess of an
all-ferrous charge balance. ``_load_composition`` returns that block unchanged
(including 'O'), which is the element-mole input format ``HeFESToAPI.ForwardMB``
already accepts: its ``InputParser`` converts O to Fe3 by charge balance
(``ngibbs.engine.API.add_ferric_column``) and renormalizes the cations to sum
to 1, so the emulator sees each standard with the redox state HeFESTo ran it
at. ``_load_cation_fractions`` applies the same conversion for consumers that
need the emulator's own composition space without calling the API (the
training-coverage Harker overlay in ``training_coverage.py``).

``hefesto_run_comparison(api, sim_dir)`` compares one simulation row by row
(phase mole fractions; with the EOS parameters, rho / VP / VS / S / Cp / KS /
alpha from fort.56, the built-in EOS on the fort.99 assemblage, and the
built-in EOS on each emulated assemblage). ``workspace_benchmark.
evaluate_workspace`` runs it over every simulation of a workspace and builds
the tables and plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from ngibbs.engine.EOS_arithmetic.hefesto_vec import compute_within_phase_frac
from ngibbs.utils.file_utils import (
    _parse_control_file,
    _parse_fort56,
    _safe_read_ws_table,
    load_fort99_componentMoles,
    _resolve_component_name_from_abbr,
    _resolve_component_phase,
    _build_reverse_component_phase_map,
)
from ._phase_plotting import (
    phase_colors, build_ordered_phases,
)

ELEMENT_KEYS = ['Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr', 'O']


# fort.56's own (bare, un-suffixed) column names for P/T/S.
P_COL = 'P(GPa)'
T_COL = 'T(K)'
S_COL = 'S(J/g/K)'

# EOS property key -> (fort.56 column token, short name, EOS->table unit scale).
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


# --------------------------------------------------------------------------- #
# shared loading helpers
# --------------------------------------------------------------------------- #
def _to_numpy(x) -> np.ndarray:
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def find_tables(table_dir, tables: List[str] = None) -> List[Path]:
    """Raw HeFESTo simulation directories (each holding ``control`` +
    ``fort.56`` + ``fort.99``) to use as standard-rock ground truth, one per
    rock -- the HeFESTo analogue of ``melts_comparison.find_tables`` finding
    raw alphaMELTS run folders."""
    table_dir = Path(table_dir)
    if tables:
        out = []
        for t in tables:
            p = Path(t)
            if not p.is_dir():
                p = table_dir / t
            if not p.is_dir():
                raise FileNotFoundError(f'HeFESTo simulation directory not found: {t}')
            out.append(p)
        return out
    if not table_dir.is_dir():
        raise NotADirectoryError(f'HeFESTo standards directory does not exist: {table_dir}')
    out = sorted(p for p in table_dir.iterdir()
                 if p.is_dir() and (p / 'control').exists() and (p / 'fort.56').exists())
    if not out:
        raise FileNotFoundError(
            f'No HeFESTo simulation directories (control + fort.56) found in {table_dir}')
    return out


def _load_composition(sim_dir: Path) -> Dict[str, float]:
    """Bulk composition of this standard rock exactly as HeFESTo was given it:
    element moles (cations AND 'O') from its ``control`` file, on HeFESTo's
    fixed total-moles basis.

    This is the HeFESTo element-mole input format that ``HeFESToAPI.ForwardMB``
    accepts directly: with 'O' among the headers, ``InputParser.parse_composition``
    replaces it by 'Fe3' via charge balance (``API.add_ferric_column``), leaving
    ferrous iron in 'Fe', and ``parse_input`` then renormalizes the cations to
    sum to 1. Passing the oxygen through (rather than dropping it and letting
    ``reorder_input_table(strict=False)`` zero-fill 'Fe3', as this function used
    to) keeps each standard's ferric iron -- e.g. Fe3+/sum(Fe) of ~0.12 for BASALT
    and ~0.19 for SHERG -- so emulator and HeFESTo see the same bulk.
    """
    element_moles, _ = _parse_control_file(str(Path(sim_dir) / 'control'))
    missing = [k for k in ELEMENT_KEYS if k not in element_moles]
    if missing:
        raise ValueError(f'{sim_dir}/control is missing element(s) {missing}')
    return {k: float(element_moles[k]) for k in ELEMENT_KEYS}


def _load_cation_fractions(sim_dir: Path) -> Dict[str, float]:
    """Bulk CATION mole fractions (sum == 1, no 'O') in the emulator's 'elements'
    space -- 'Fe' ferrous only, 'Fe3' ferric -- for this standard rock.

    Same conversion ``ForwardMB`` applies to ``_load_composition``'s output
    (``API.add_ferric_column``, then renormalization over cations), for callers
    that need the composition itself rather than an emulator prediction, such
    as the training-coverage Harker overlay.
    """
    from ngibbs.engine.API import add_ferric_column

    raw = _load_composition(sim_dir)
    headers = list(raw)
    values, headers = add_ferric_column(np.array([[raw[k] for k in headers]], dtype=np.float64),
                                        headers)
    comp = {h: float(v) for h, v in zip(headers, values[0])}
    total = sum(comp.values())
    if total <= 0:
        raise ValueError(f'Non-positive total cation moles in {sim_dir}/control')
    return {k: v / total for k, v in comp.items()}


def _load_component_moles(sim_dir: Path, indexer) -> np.ndarray:
    """Extensive component moles aligned to ``indexer.label_names``, read
    straight off this simulation's ``fort.99`` (see
    ``ngibbs.utils.file_utils.load_fort99_componentMoles``)."""
    return load_fort99_componentMoles(str(sim_dir), indexer)


def _expand_chem_out_to_components(chem_out: np.ndarray, indexer) -> np.ndarray:
    B = chem_out.shape[0]
    full = np.ones((B, len(indexer.label_names)), dtype=np.float64)
    vc = np.asarray(indexer.compositionally_variable_subset, dtype=np.int64)
    full[:, vc] = chem_out
    return full


def _load_fort99_phase_totals(sim_dir: Path) -> Dict[str, np.ndarray]:
    """{phase_name: (n,) extensive mole total}, from EVERY component column
    in this simulation's ``fort.99`` -- including phases this checkpoint's
    ml_indexer doesn't track (grouped under their own raw fort.99
    abbreviation instead of being silently dropped), so the true system
    total and ``_check_missing_phases``'s "unrepresented phase" detection
    both stay correct. The HeFESTo analogue of a MELTStable CSV's
    ``'total (moles)(<phase>)'`` columns, computed directly instead of
    reading a pre-aggregated column."""
    sim_dir = Path(sim_dir)
    _, control_component_to_phase_abbr = _parse_control_file(str(sim_dir / 'control'))
    comp_df = _safe_read_ws_table(str(sim_dir / 'fort.99'), skiprows=0)
    reverse_map = _build_reverse_component_phase_map()
    component_cols = list(comp_df.columns)[3:-2]
    n = len(comp_df)
    totals: Dict[str, np.ndarray] = {}
    for comp_abbr in component_cols:
        comp_abbr_str = str(comp_abbr).strip()
        comp_name = _resolve_component_name_from_abbr(comp_abbr_str)
        phase_name = _resolve_component_phase(
            component_abbr=comp_abbr_str, component_name=comp_name,
            reverse_component_phase_map=reverse_map,
            control_component_to_phase_abbr=control_component_to_phase_abbr,
        ) or comp_abbr_str
        values = pd.to_numeric(comp_df[comp_abbr], errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
        totals[phase_name] = totals.get(phase_name, np.zeros(n)) + values
    return totals


def _read_conditions(sim_dir: Path):
    """(fort.56 DataFrame, P_GPa, T_K, S_JgK) straight off this simulation's
    fort.56."""
    sim_dir = Path(sim_dir)
    df = _parse_fort56(str(sim_dir / 'fort.56'))
    for col in (P_COL, T_COL, S_COL):
        if col not in df.columns:
            raise KeyError(f'{sim_dir.name}/fort.56 missing required column {col!r}')
    P = pd.to_numeric(df[P_COL], errors='coerce').to_numpy(dtype=np.float64)
    T = pd.to_numeric(df[T_COL], errors='coerce').to_numpy(dtype=np.float64)
    S = pd.to_numeric(df[S_COL], errors='coerce').to_numpy(dtype=np.float64)
    return df, P, T, S


def _restrict_to_training_pressure_range(api, df, P, T, S):
    """Drop rows outside the model's training pressure range. Returns
    (df, P, T, S, keep); apply ``keep`` to any other per-row ground truth
    (fort.99 component moles / phase totals) so it stays row-aligned."""
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
    return df.loc[keep].reset_index(drop=True), P[keep], T[keep], S[keep], keep


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


# --------------------------------------------------------------------------- #
# phase comparison
# --------------------------------------------------------------------------- #
def _gt_phase_fractions(totals: Dict[str, np.ndarray], mass_phasedict, n: int):
    n_phases = max(mass_phasedict.values()) + 1
    pm = np.zeros((n, n_phases), dtype=np.float64)
    for phase, col in mass_phasedict.items():
        if phase in totals:
            pm[:, col] = totals[phase]
    grand = np.sum(np.stack(list(totals.values()), axis=1), axis=1) if totals else np.ones(n)
    grand = np.where(grand == 0, 1.0, grand)
    return pm / grand[:, None]


def _check_missing_phases(totals: Dict[str, np.ndarray], mass_phasedict, P):
    n = len(P)
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


HEFESTO_PROPERTY_UNITS = {'rho': 'g/cm³', 'VP': 'km/s', 'VS': 'km/s', 'S': 'J/g/K', 'Cp': 'J/g/K',
                          'KS': 'GPa', 'alpha': '1e-5/K'}


def hefesto_run_comparison(api, sim_dir, *, pathways=('isothermal', 'isentropic'),
                           properties: bool = True) -> Dict[str, object]:
    """Compare one raw HeFESTo simulation folder (``control`` + ``fort.56`` +
    ``fort.99``) against the emulator, row by row, over the rows inside the
    model's training pressure range.

    * Phase mole fractions: fort.99's phase totals (every phase, so an
      unrepresented one still counts in the total) vs the emulator's
      ``phase_moles`` for each pathway.
    * With ``properties`` (needs the HeFESTo EOS parameters): fort.56's bulk
      rho / VP / VS / S / Cp / KS / alpha, the built-in EOS on the fort.99
      assemblage, and the built-in EOS on each emulated assemblage (the
      isentropic one at the temperature model's T). A property failure is
      recorded in ``'property_error'`` and does not stop the phase comparison.

    Returns the per-run comparison dict consumed by
    ``workspace_benchmark.evaluate_workspace`` (see
    ``melts_comparison._new_run_result``).
    """
    from .melts_comparison import _new_run_result

    sim_dir = Path(sim_dir)
    df, P, T, S = _read_conditions(sim_dir)
    df, P, T, S, keep = _restrict_to_training_pressure_range(api, df, P, T, S)
    composition = _load_composition(sim_dir)
    indexer = api.isothermal_emulator.ml_indexer
    mass_phasedict = indexer.mass_phasedict

    result = _new_run_result(variant='', P=P, P_label='P (GPa)', T=T, T_label='T (K)',
                             phase_basis='mole fraction', units=HEFESTO_PROPERTY_UNITS)
    active = [p for p in pathways
              if getattr(api, {'isothermal': 'isothermal_emulator',
                               'isentropic': 'isentropic_emulator'}.get(p, ''), None) is not None]
    result['pathways'] = active
    second = {'isothermal': (T, 'T(K)(System_main)'), 'isentropic': (S, 'S(J/g/K)(System_main)')}

    # --- phases (mole fraction of the system), keyed by HeFESTo abbreviation ---
    ordered = build_ordered_phases(mass_phasedict)          # [(long_name, abbrev)]
    phase_totals = {k: v[keep] for k, v in _load_fort99_phase_totals(sim_dir).items()}
    gt_pf = _gt_phase_fractions(phase_totals, mass_phasedict, len(P))

    def by_abbrev(frac):
        return {abbrev: frac[:, mass_phasedict[name]] for name, abbrev in ordered}

    result['phases_gt'] = by_abbrev(gt_pf)
    unrepresented = _check_missing_phases(phase_totals, mass_phasedict, P)['missing_fraction']
    if np.any(unrepresented > 0):
        result['phases_gt']['unrepresented'] = unrepresented
    for pathway in active:
        x, header = second[pathway]
        result['phases_emulated'][pathway] = by_abbrev(
            _emulator_phase_fractions(api, P, x, composition, header))
    result['phase_order'] = [abbrev for _, abbrev in ordered] + (
        ['unrepresented'] if 'unrepresented' in result['phases_gt'] else [])
    result['phase_colors'] = {**phase_colors(), 'unrepresented': 'white'}

    if not properties:
        return result
    if api.hefesto_params is None:
        result['property_error'] = 'HeFESTo EOS parameters not loaded (no control file)'
        return result
    try:
        result['properties'] = {short: pd.to_numeric(df[tok], errors='coerce').to_numpy(np.float64)
                                for tok, short, _ in PROPERTY_MAP.values()}

        def scaled(props):
            return {short: np.asarray(_to_numpy(props[key]), dtype=np.float64) * scale
                    for key, (_, short, scale) in PROPERTY_MAP.items()}

        gt_cm = _load_component_moles(sim_dir, indexer)[keep]
        gt_wpf = compute_within_phase_frac(gt_cm, list(indexer.label_indices.values()))
        result['eos_gt'] = scaled(_eval_assemblage(api, gt_cm, P, T, gt_wpf))
        for pathway in active:
            x, header = second[pathway]
            isen = pathway == 'isentropic'
            cm, wpf, T_em = _emulator_assemblage(api, P, x, composition, header, isen)
            result['eos_emulated'][pathway] = scaled(_eval_assemblage(api, cm, P, T_em if isen else T, wpf))
            if isen:
                result['T_pred'][pathway] = T_em
    except Exception as e:  # noqa: BLE001 -- recorded, phases still usable
        result['property_error'] = f'{type(e).__name__}: {e}'
        result['eos_emulated'] = {}
    return result
