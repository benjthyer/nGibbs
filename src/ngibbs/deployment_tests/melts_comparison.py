"""MELTS analogue of ``meltstable_comparison.py``.

Ground truth here is a set of raw alphaMELTS isobaric-cooling run folders
(``MELTSIsobaricStandards/<model>/<Cr|NoCr>/<rock>/``) rather than a single
MELTStable CSV per rock: ``System_main_tbl.txt`` (P/T/S/mass/fO2 conditions),
``Bulk_comp_tbl.txt`` (evolving bulk oxide composition, including a real,
already-speciated FeO/Fe2O3 split -- these are fO2-buffered runs), and one
``<phase>.tbl`` per phase actually saturated during that run (endmember mole
fractions for solution phases, already in melts_vec's own column order --
verified against real .tbl headers, see ``_GT_SOLUTION_ENDMEMBERS``).

Two independent "sources" of bulk properties/phase proportions are compared
against this ground truth, mirroring HeFESTo's three-source structure minus
the one source MELTS has no way to produce (an emulator-predicted assemblage
run through melts_vec, split by isothermal/isentropic pathway):
  - ``gt_assemblage_internal_eos``: the REAL alphaMELTS phase assemblage
    (composition + molar abundance, read straight from the .tbl files) fed
    through melts_vec's internal EOS via
    ``MELTSAPI.get_property_melts_vectorized_from_assemblage``. Liquid goes
    through the oxide->component bridge (``oxides_to_liquid_components``)
    using the GT's own real (already fO2-buffered) oxide wt%, no
    speciation approximation needed.
  - ``emulation_isothermal`` / ``emulation_isentropic``: the trained MELTS120
    emulator's OWN predicted assemblage (fed the row's real, evolving bulk
    oxide composition -- including the real Fe2O3/FeO split, which is what
    "provide the evolving Fe3/FeT ratio" means in practice for a closed-model
    forward pass) run through the *same* internal EOS, via the
    ml_indexer -> melts_vec bridge in ``_ml_indexer_to_melts_vec_assemblage``.
    The isentropic pathway has no MELTS120 temperature model to predict T
    from S, so it borrows the row's real GT temperature to evaluate the
    internal EOS -- properties are still "assemblage predicted by the
    isentropic model", just evaluated at the true T rather than a modelled
    one; this is noted wherever isentropic property results are reported.

Phase-proportion ground truth and comparisons are on a MASS basis throughout
(matching Phase_mass_tbl.txt and the isentropic model's own gram-mass
normalization), not mole fraction.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ngibbs.config.constants import OXIDE_MOLAR_MASSES, get_oxide_molar_mass
from ngibbs.engine.EOS_arithmetic_MELTS.melts_vec import molar_masses as _melts_molar_masses
from ngibbs.engine.EOS_arithmetic_MELTS.melts_vec import oxides_to_liquid_components as _oxides_to_liquid_components

from ._melts_phase_plotting import (
    MELTS_PHASE_STACK_ORDER,
    build_ordered_melts_phases,
    draw_melts_phase_stack,
    melts_phase_colors,
)

ROCKS = ('BishopTuff', 'Lherzolite', 'MORB')
VARIANTS = ('Cr', 'NoCr')

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_STANDARDS_DIR = PACKAGE_DIR / 'MELTSIsobaricStandards' / '120'

_BARS_PER_GPA = 10000.0

# --------------------------------------------------------------------------- #
# GT solid-solution endmember columns, IN melts_vec'S OWN ORDER -- verified
# against real MELTSIsobaricStandards .tbl headers this session (every
# solution-phase .tbl file already carries its endmember mole fractions as
# its trailing columns, in exactly this order; nothing needs reindexing).
# --------------------------------------------------------------------------- #
_GT_SOLUTION_ENDMEMBERS: Dict[str, List[str]] = {
    'olivine': ['tephroite', 'fayalite', 'co-olivine', 'ni-olivine', 'monticellite', 'forsterite'],
    'clinopyroxene': ['diopside', 'clinoenstatite', 'hedenbergite', 'alumino-buffonite',
                       'buffonite', 'essenite', 'jadeite'],
    'orthopyroxene': ['diopside', 'clinoenstatite', 'hedenbergite', 'alumino-buffonite',
                       'buffonite', 'essenite', 'jadeite'],
    'spinel': ['chromite', 'hercynite', 'magnetite', 'spinel', 'ulvospinel'],
    'plagioclase': ['albite', 'anorthite', 'sanidine'],
    'alkali-feldspar': ['albite', 'anorthite', 'sanidine'],
    'rhm-oxide': ['geikielite', 'hematite', 'ilmenite', 'pyrophanite', 'corundum'],
}
# melts_vec's own _MELTS_SOLUTION_PHASES recognises 'feldspar'/'plagioclase' but
# has no 'alkali-feldspar' entry (alphaMELTS models plagioclase and alkali
# feldspar as two immiscible solutions over the SAME albite-anorthite-sanidine
# ternary -- confirmed by alkali-feldspar.tbl and plagioclase.tbl sharing
# identical endmember columns). MELTSAPI._MELTS_SOLUTION_PHASES is extended
# with this alias at import time in `_ensure_alkali_feldspar_alias` below
# rather than routing alkali-feldspar through the (wrong) pure-phase fallback.
_GT_PURE_PHASES = ('quartz', 'tridymite', 'apatite', 'whitlockite')
# Phase_mass_tbl.txt names the liquid column 'liquid1' (de-suffixed: 'liquid'),
# while its own per-row data lives in 'melts-liquid.tbl' -- two different
# names for the same phase across these two raw-output files.
_GT_LIQUID_NAMES = ('liquid', 'melts-liquid')
_GT_SKIP_PHASES = ('fluid',)  # H2O/CO2 fluid: not part of melts_vec's solid+liquid EOS scope

# wt% oxide columns present in every raw .tbl / Bulk_comp/Liquid_comp file.
_RAW_OXIDE_COLS = ['SiO2', 'TiO2', 'Al2O3', 'Fe2O3', 'Cr2O3', 'FeO', 'MnO', 'MgO',
                    'NiO', 'CoO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'H2O', 'CO2']

# The oxide set the trained MELTS120 emulator actually tracks (NoCr; Cr adds
# Cr2O3) -- confirmed against a loaded checkpoint's ml_indexer.Oxides this
# session. MnO/NiO/CoO are not part of this emulator's tracked composition
# space at all and are dropped when building its input feature row.
_EMULATOR_OXIDE_COLS_NOCR = ['SiO2', 'TiO2', 'Al2O3', 'Fe2O3', 'FeO', 'MgO', 'CaO',
                             'Na2O', 'K2O', 'P2O5', 'H2O', 'CO2']
_EMULATOR_OXIDE_COLS_CR = _EMULATOR_OXIDE_COLS_NOCR + ['Cr2O3']


def _ensure_alkali_feldspar_alias(api) -> None:
    """Extend MELTSAPI's (class-level) _MELTS_SOLUTION_PHASES with
    'alkali-feldspar' (see module docstring / _GT_SOLUTION_ENDMEMBERS comment
    above). ``api`` is the top-level MELTSAPI object (get_property_melts_
    vectorized_from_assemblage and _MELTS_SOLUTION_PHASES live there, not on
    its .nocr/.cr EmulatorAPI sub-instances). Idempotent."""
    sp = type(api)._MELTS_SOLUTION_PHASES
    if 'alkali-feldspar' not in sp:
        sp['alkali-feldspar'] = sp['feldspar']


def _to_float(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return np.nan


# --------------------------------------------------------------------------- #
# Raw alphaMELTS table loaders
# --------------------------------------------------------------------------- #
def _read_whitespace_table(path: Path) -> pd.DataFrame:
    """Parser for the whitespace-delimited System_main_tbl.txt / Bulk_comp_tbl.txt
    / Phase_mass_tbl.txt / Phase_vol_tbl.txt style files: a free-text preamble,
    then a header line starting with 'index', then space-delimited data rows
    (some cells 'n/a')."""
    with open(path) as f:
        lines = f.readlines()
    header = None
    data_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('index'):
            header = stripped.split()
            data_start = i + 1
            break
    if header is None:
        raise ValueError(f"Could not find 'index ...' header line in {path}")
    rows = []
    for line in lines[data_start:]:
        parts = line.split()
        if not parts:
            continue
        rows.append([_to_float(x) for x in parts])
    df = pd.DataFrame(rows, columns=header)
    return df


def read_system_main(rock_dir: Path) -> pd.DataFrame:
    """index, Pressure(bar), Temperature(C), mass(g), ..., S(J/K total), ...,
    fO2(absolute), fO2-(QFM)."""
    return _read_whitespace_table(Path(rock_dir) / 'System_main_tbl.txt')


def read_bulk_comp(rock_dir: Path) -> pd.DataFrame:
    """index, Pressure, Temperature, mass, <oxide wt%>... (real, already
    fO2-buffered FeO/Fe2O3 split)."""
    return _read_whitespace_table(Path(rock_dir) / 'Bulk_comp_tbl.txt')


def read_phase_mass(rock_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    """index, Pressure, Temperature, <phase><instance#> mass(g)... Returns the
    frame plus the de-suffixed phase name list, in column order."""
    df = _read_whitespace_table(Path(rock_dir) / 'Phase_mass_tbl.txt')
    phase_cols = [c for c in df.columns if c not in ('index', 'Pressure', 'Temperature')]
    phase_names = [re.sub(r'\d+$', '', c) for c in phase_cols]
    return df, list(zip(phase_cols, phase_names))


def read_phase_tbl(rock_dir: Path, phase_file_stem: str) -> Optional[pd.DataFrame]:
    """One <phase>.tbl (comma-separated, header row with 'Index', 'T (C)',
    'wt% <oxide>' columns and, for solution phases, trailing endmember mole
    fraction columns). Returns None if the phase never saturated in this run
    (no .tbl file written for it at all)."""
    path = Path(rock_dir) / f'{phase_file_stem}.tbl'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


# --------------------------------------------------------------------------- #
# GT-assemblage -> melts_vec internal EOS
# --------------------------------------------------------------------------- #
def _mean_molar_mass_from_oxide_wtpct(oxide_wtpct: Dict[str, np.ndarray]) -> np.ndarray:
    """Standard mean molar mass (g/mol) from wt% oxide composition:
    100 / sum(wt%_i / MM_i). Rows with zero total wt% -> nan (never used as a
    divisor for a phase with zero mass/moles anyway)."""
    total_wt = None
    inv = None
    for ox, wt in oxide_wtpct.items():
        wt = np.asarray(wt, dtype=np.float64)
        mm = get_oxide_molar_mass(ox)
        contrib = wt / mm
        inv = contrib if inv is None else inv + contrib
        total_wt = wt if total_wt is None else total_wt + wt
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(inv > 0, total_wt / np.where(inv > 0, inv, 1.0), np.nan)


def _reindex_to(full_index: np.ndarray, df: pd.DataFrame, index_col: str = 'Index') -> pd.DataFrame:
    """Left-join df (keyed by index_col) onto every value in full_index,
    filling absent rows with NaN (caller decides how to zero-fill).

    Requires ``index_col`` to be unique in ``df`` -- a duplicated Index (two
    co-existing instances of the same phase reported as separate rows, e.g.
    BishopTuff's two rhm-oxide grains) inflates the merge past
    ``len(full_index)``. Callers reading a per-phase .tbl file should collapse
    duplicates with ``_dedupe_tbl_instances`` first.
    """
    base = pd.DataFrame({index_col: full_index.astype(int)})
    merged = base.merge(df, on=index_col, how='left')
    return merged


def _dedupe_tbl_instances(tbl: pd.DataFrame, value_cols: Sequence[str],
                           index_col: str = 'Index') -> pd.DataFrame:
    """Collapse a phase .tbl's multiple co-existing instances at the same
    step (e.g. two rhm-oxide grains both saturated at the same Index, written
    as two separate rows) into one row per Index, via a mass-weighted average
    of ``value_cols`` using each instance's own 'mass (gm)' column (equal
    weights if that column is absent). No-op when ``index_col`` is already
    unique. Only ``index_col`` and ``value_cols`` survive in the output --
    the only columns load_gt_assemblage actually reads downstream."""
    if not tbl[index_col].duplicated().any():
        return tbl
    w = (pd.to_numeric(tbl['mass (gm)'], errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
         if 'mass (gm)' in tbl.columns else np.ones(len(tbl)))
    tmp = tbl[[index_col] + list(value_cols)].copy()
    for c in value_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors='coerce').fillna(0.0) * w
    tmp['__w__'] = w
    grouped = tmp.groupby(index_col, as_index=False).sum()
    for c in value_cols:
        grouped[c] = np.where(grouped['__w__'] > 0, grouped[c] / grouped['__w__'], 0.0)
    return grouped.drop(columns='__w__')


def load_gt_assemblage(rock_dir: Path, api) -> Dict[str, np.ndarray]:
    """Build the GT assemblage's melts_vec phase_composition/phase_moles/PT/
    liquid_oxides inputs for every row of one isobaric-cooling run, ready to
    pass straight to ``api.get_property_melts_vectorized_from_assemblage``
    (``api`` is the top-level MELTSAPI object -- that method and
    melts_solid_params/melts_liquid_params live there, not on .nocr/.cr).

    Returns a dict with 'PT' (B,2) [GPa, K], 'phase_composition',
    'phase_moles', 'liquid_oxides', 'liquid_oxide_labels', plus 'gt' (a small
    DataFrame of index/P_bar/T_K/S_specific/mass_g for downstream property
    comparison).
    """
    rock_dir = Path(rock_dir)
    sysmain = read_system_main(rock_dir)
    full_index = sysmain['index'].values.astype(int)
    B = len(full_index)

    T_K = sysmain['Temperature'].values + 273.15
    P_bar = sysmain['Pressure'].values
    mass_g = sysmain['mass'].values
    S_total_JK = sysmain['S'].values
    S_specific = S_total_JK / np.where(mass_g > 0, mass_g, np.nan)

    PT = np.column_stack([P_bar / _BARS_PER_GPA, T_K])

    phase_mass_df, phase_col_names = read_phase_mass(rock_dir)
    phase_mass_df = _reindex_to(full_index, phase_mass_df, index_col='index')

    # Two co-existing instances of the same phase (e.g. BishopTuff's two
    # rhm-oxide grains) appear as separate <phase><N> columns here, all
    # de-suffixed to the SAME phase_name (see read_phase_mass) -- sum their
    # masses into one combined per-phase array up front, so iterating below
    # by phase_name (not by raw column) never lets a later instance silently
    # overwrite an earlier one's contribution.
    combined_mass: Dict[str, np.ndarray] = {}
    for col, phase_name in phase_col_names:
        m = phase_mass_df[col].fillna(0.0).values.astype(np.float64)
        combined_mass[phase_name] = combined_mass.get(phase_name, np.zeros(B)) + m

    phase_composition: Dict[str, np.ndarray] = {}
    phase_moles: Dict[str, np.ndarray] = {}

    liquid_oxides = None
    liquid_oxide_labels = None

    for phase_name, mass_phase in combined_mass.items():
        if phase_name in _GT_SKIP_PHASES:
            continue

        if phase_name in _GT_LIQUID_NAMES:
            liq_df = read_phase_tbl(rock_dir, 'melts-liquid')
            if liq_df is None:
                phase_moles[phase_name] = np.zeros(B)
                continue
            wt_cols_present = [f'wt% {ox}' for ox in _RAW_OXIDE_COLS if f'wt% {ox}' in liq_df.columns]
            liq_df = _dedupe_tbl_instances(liq_df, wt_cols_present)
            liq_df = _reindex_to(full_index, liq_df, index_col='Index')
            wtcols = {ox: liq_df[f'wt% {ox}'].fillna(0.0).values for ox in _RAW_OXIDE_COLS
                      if f'wt% {ox}' in liq_df.columns}
            # Both oxides_to_liquid_components and get_property_melts_vectorized_
            # from_assemblage's own liquid_oxides slot want MOLAR oxide
            # composition (see oxides_to_liquid_components's own docstring:
            # "Oxide moles"), not wt% -- convert wt% (per 100 g nominal) to
            # moles-per-100g first. Feeding raw wt% straight through here
            # (an earlier version of this loader's bug) silently overstates
            # every oxide's mole count by its own molar mass, badly inflating
            # the derived liquid mole count and every specific (per-gram)
            # property downstream.
            mole_cols = {ox: wt / get_oxide_molar_mass(ox) for ox, wt in wtcols.items()}
            liquid_oxide_labels = list(mole_cols.keys())
            liquid_oxides = np.column_stack([mole_cols[ox] for ox in liquid_oxide_labels])

            comp_moles = _oxides_to_liquid_components(mole_cols, api.melts_liquid_params)
            total_comp_moles_per_100g = comp_moles.sum(axis=1)
            moles = total_comp_moles_per_100g * mass_phase / 100.0
            phase_moles[phase_name] = moles
            # placeholder composition slot: liquid_oxides overrides it entirely.
            phase_composition[phase_name] = np.ones((B, 1))
            continue

        if phase_name in _GT_PURE_PHASES:
            phase_composition[phase_name] = np.ones((B, 1))
            tbl = read_phase_tbl(rock_dir, phase_name)
            if tbl is None:
                phase_moles[phase_name] = np.zeros(B)
                continue
            tbl = _reindex_to(full_index, tbl, index_col='Index')
            mm = float(_melts_molar_masses(api.melts_solid_params, names=[phase_name])[0])
            phase_moles[phase_name] = np.where(mass_phase > 0, mass_phase / mm, 0.0)
            continue

        if phase_name in _GT_SOLUTION_ENDMEMBERS:
            endmembers = _GT_SOLUTION_ENDMEMBERS[phase_name]
            tbl = read_phase_tbl(rock_dir, phase_name)
            if tbl is None:
                phase_composition[phase_name] = np.zeros((B, len(endmembers)))
                phase_composition[phase_name][:, -1] = 1.0  # inert placeholder
                phase_moles[phase_name] = np.zeros(B)
                continue
            em_cols_present = [em for em in endmembers if em in tbl.columns]
            tbl = _dedupe_tbl_instances(tbl, em_cols_present)
            tbl = _reindex_to(full_index, tbl, index_col='Index')
            X = np.zeros((B, len(endmembers)))
            for j, em in enumerate(endmembers):
                if em in tbl.columns:
                    X[:, j] = tbl[em].fillna(0.0).values
            row_sum = X.sum(axis=1)
            present = row_sum > 1e-8
            X_norm = np.where(present[:, None], X / np.where(present[:, None], row_sum[:, None], 1.0),
                               0.0)
            X_norm[~present, -1] = 1.0  # inert placeholder for absent rows (moles=0 there anyway)
            phase_composition[phase_name] = X_norm

            endmember_mm = _melts_molar_masses(api.melts_solid_params, names=endmembers)
            mm = X_norm @ endmember_mm
            phase_moles[phase_name] = np.where((mass_phase > 0) & present, mass_phase / np.where(mm > 0, mm, 1.0), 0.0)
            continue

        warnings.warn(f"GT phase '{phase_name}' has no melts_vec mapping; excluded from assemblage.")

    return {
        'PT': PT, 'phase_composition': phase_composition, 'phase_moles': phase_moles,
        'liquid_oxides': liquid_oxides, 'liquid_oxide_labels': liquid_oxide_labels,
        'index': full_index, 'T_K': T_K, 'P_bar': P_bar, 'mass_g': mass_g,
        'S_specific_JgK': S_specific, 'S_total_JK': S_total_JK,
        'fO2_QFM': sysmain['fO2-(QFM)'].values if 'fO2-(QFM)' in sysmain.columns else None,
    }


# --------------------------------------------------------------------------- #
# Emulator-predicted assemblage -> melts_vec internal EOS bridge
# --------------------------------------------------------------------------- #
def _embed_by_name(ml_component_moles: np.ndarray, ml_component_names: Sequence[str],
                    melts_vec_endmembers: Sequence[str], warn_prefix: str = '') -> np.ndarray:
    """(B, n_ml) ml_indexer component moles -> (B, n_mv) melts_vec endmember
    array, matched by exact endmember NAME. Any melts_vec endmember this
    checkpoint doesn't track (e.g. tephroite for a training set with no
    Mn-olivine, because MnO isn't in this checkpoint's tracked oxide set at
    all) is zero-filled -- a real statement that its abundance is
    structurally 0 given this checkpoint's tracked composition space, not an
    approximation."""
    B = ml_component_moles.shape[0]
    out = np.zeros((B, len(melts_vec_endmembers)), dtype=np.float64)
    idx = {name: i for i, name in enumerate(melts_vec_endmembers)}
    for j, name in enumerate(ml_component_names):
        if name in idx:
            out[:, idx[name]] = ml_component_moles[:, j]
        else:
            warnings.warn(f"{warn_prefix}ml_indexer component {name!r} has no melts_vec "
                           f"endmember match; dropped.")
    return out


def ml_indexer_to_melts_vec_assemblage(
    api_sub, api, emulator, component_moles: torch.Tensor, mass_wtpct: torch.Tensor,
    comp_wtpct: torch.Tensor,
) -> Dict[str, object]:
    """Bridge one ForwardMB(...) call's raw 'component_moles' + 'phase_tables'
    mass output (api_sub's own ml_indexer basis) into melts_vec's own
    phase_composition/phase_moles/liquid_oxides inputs for
    ``api.get_property_melts_vectorized_from_assemblage``.

    Solid solution phases: ml_indexer's own components_in_phases columns are
    embedded into melts_vec's endmember order by exact NAME match (verified
    against a real checkpoint this session -- clinopyroxene/orthopyroxene
    already match melts_vec's order exactly; olivine/spinel/rhm-oxide are
    checkpoint-tracked SUBSETS of melts_vec's full endmember list, missing
    only Mn/Ni/Co-bearing endmembers this checkpoint's oxide set doesn't
    track at all -- see _embed_by_name). alphaMELTS's 'k-feldspar' is routed
    through melts_vec's 'feldspar'/'plagioclase' mixing model too (same
    albite-anorthite-sanidine ternary; MELTSAPI's own _MELTS_SOLUTION_PHASES
    has no separate 'k-feldspar' key -- see _ensure_alkali_feldspar_alias).

    Liquid: NOT bridged via component_moles/get_liquid_oxides -- empirically,
    for this checkpoint, component_moles's liquid slice (indexed through
    label_indices_comp['melts-liquid'], labelled 'wt% <oxide>') comes back
    structurally wrong (major oxides SiO2/Al2O3/MgO/CaO/FeO all exactly 0,
    nonzero only in K2O/P2O5/Cr2O3/Fe2O3/CO2) even on a real, in-range,
    100%-liquid row, while 'phase_tables'' comp_tens output for the SAME row
    gives a fully plausible liquid oxide composition. So liquid instead
    reads its oxide wt% straight from comp_wtpct (the comp_tens half of
    'phase_tables', already validated), converts wt%->moles/100g (same
    convention as load_gt_assemblage), and feeds THAT into
    oxides_to_liquid_components -- component_moles/get_liquid_oxides is
    only used for the solid solution phases above, where it checks out.

    Per-phase molar abundance is NOT taken from ForwardMB's own 'phase_moles'
    output (that output is normalized within the ml_indexer's own internal
    basis, not a cross-phase-consistent extensive quantity) -- instead each
    phase's own real molar mass (from its bridged composition, via
    melts_vec.molar_masses / the liquid's own oxide molar mass) converts its
    'phase_tables' mass fraction into an internally-consistent mole count,
    exactly mirroring how ``load_gt_assemblage`` derives GT phase_moles from
    real mass. Bulk 'phase_moles' units are arbitrary/self-consistent -- the
    aggregation in get_property_melts_vectorized_from_assemblage only uses
    relative proportions between phases.

    api_sub : EmulatorAPI
        api.nocr or api.cr -- whichever sub-API actually ran the forward
        pass (needed for its ml_indexer).
    api : MELTSAPI
        Top-level object -- get_property_melts_vectorized_from_assemblage and
        melts_solid_params/melts_liquid_params live here.
    emulator : NN_MELTS
        The specific wrapped emulator that produced component_moles/
        mass_wtpct/comp_wtpct (api_sub.isothermal_emulator /
        .isentropic_emulator / .open_emulator) -- only its ml_indexer
        (component/oxide/phasedict layout) is used here.
    component_moles, mass_wtpct, comp_wtpct : torch.Tensor
        'component_moles' and both halves (mass, comp) of 'phase_tables'
        from ``emulator``'s own forwardMB (or api_sub.ForwardMB) call --
        i.e. call ForwardMB(..., outputs=['component_moles', 'phase_tables'])
        and pass phase_tables' (comp_tens, mass_tens) here as
        (comp_wtpct, mass_wtpct).

    Returns dict with 'phase_composition', 'phase_moles', 'liquid_oxides',
    'liquid_oxide_labels', 'mass_wtpct_by_phase' (mass-phasedict-indexed
    mass-fraction array, for the phase-comparison diagrams), 'mass_phasedict'
    (for indexing mass_wtpct_by_phase).
    """
    mi = emulator.ml_indexer
    B = component_moles.shape[0]
    cm_np = component_moles.detach().cpu().numpy().astype(np.float64)
    mass_np = mass_wtpct.detach().cpu().numpy().astype(np.float64)  # (B, n_phases) mass fraction of system

    phase_composition: Dict[str, np.ndarray] = {}
    phase_moles: Dict[str, np.ndarray] = {}

    for phase_name, comp_idx in mi.label_indices_comp.items():
        if phase_name in _GT_LIQUID_NAMES:
            continue
        solution_key = 'plagioclase' if phase_name == 'k-feldspar' else phase_name
        if solution_key not in _GT_SOLUTION_ENDMEMBERS:
            continue  # tracked by this checkpoint but melts_vec has no mixing model for it
        endmembers = _GT_SOLUTION_ENDMEMBERS[solution_key]
        ml_names = mi.components_in_phases[phase_name]
        ml_moles = cm_np[:, comp_idx]
        X = _embed_by_name(ml_moles, ml_names, endmembers, warn_prefix=f"[{phase_name}] ")
        row_sum = X.sum(axis=1)
        present = row_sum > 1e-10
        X_norm = np.where(present[:, None], X / np.where(present[:, None], row_sum[:, None], 1.0), 0.0)
        X_norm[~present, -1] = 1.0

        gt_key = 'alkali-feldspar' if phase_name == 'k-feldspar' else phase_name
        mass_idx = mi.mass_phasedict[phase_name]
        # mass_np is percent-of-SYSTEM (rows sum to 100 across phases, verified
        # empirically), i.e. mass_frac numerically equals grams of this phase
        # per 100 g of system -- so dividing by molar mass alone already gives
        # moles per 100 g system; no extra *100 (that would overcount moles by
        # 100x, as an earlier version of this function did -- see validation
        # note below).
        mass_frac = mass_np[:, mass_idx]
        endmember_mm = _melts_molar_masses(api.melts_solid_params, names=endmembers)
        mm = X_norm @ endmember_mm
        moles = np.where(present & (mm > 0), mass_frac / np.where(mm > 0, mm, 1.0), 0.0)

        phase_composition[gt_key] = X_norm
        phase_moles[gt_key] = moles

    for phase_name in _GT_PURE_PHASES:
        if phase_name not in mi.mass_phasedict:
            continue
        mass_idx = mi.mass_phasedict[phase_name]
        mass_frac = mass_np[:, mass_idx]  # percent-of-system == grams per 100g system
        mm = float(_melts_molar_masses(api.melts_solid_params, names=[phase_name])[0])
        phase_composition[phase_name] = np.ones((B, 1))
        phase_moles[phase_name] = mass_frac / mm if mm > 0 else np.zeros(B)

    # Liquid: NOT sourced from component_moles/get_liquid_oxides (empirically
    # structurally wrong for this phase on this checkpoint -- see docstring
    # above). Instead read the liquid's own oxide wt% straight out of
    # comp_wtpct (the comp_tens half of the SAME 'phase_tables' ForwardMB
    # output that mass_wtpct comes from), indexed the same way API.py itself
    # indexes it (comp_wtpct[:, comp_phasedict[phase], :], oxide columns in
    # mi.Oxides order) -- then follow the EXACT same wt%->moles/100g->
    # oxides_to_liquid_components pipeline load_gt_assemblage uses for the
    # real GT liquid, so both pathways treat liquid identically apart from
    # where the wt% numbers come from.
    liq_key = 'melts-liquid' if 'melts-liquid' in mi.mass_phasedict else 'liquid'
    comp_wtpct_np = comp_wtpct.detach().cpu().numpy().astype(np.float64)
    liq_comp_idx = mi.comp_phasedict[liq_key]
    liq_wtpct = comp_wtpct_np[:, liq_comp_idx, :]  # (B, n_oxides), columns = mi.Oxides order
    liquid_oxide_labels = list(mi.Oxides)
    mole_cols = {ox: liq_wtpct[:, i] / get_oxide_molar_mass(ox)
                 for i, ox in enumerate(liquid_oxide_labels)}
    liquid_moles_np = np.column_stack([mole_cols[ox] for ox in liquid_oxide_labels])

    comp_moles_liq = _oxides_to_liquid_components(mole_cols, api.melts_liquid_params)
    # comp_moles_liq's row sum is moles per 100 g of PURE liquid (liq_wtpct is
    # a within-phase wt% that itself sums to 100 -- verified above). To get
    # moles per 100 g of SYSTEM, scale by how many grams of liquid are
    # actually in 100 g of system: mass_np is percent-of-system (grams per
    # 100 g system), so that scale factor is liq_mass_frac / 100 -- matching
    # the *100 removal above, and mirroring load_gt_assemblage's own
    # `total_comp_moles_per_100g * mass_phase / 100.0` (there mass_phase is
    # real grams over a real total system mass, here it's the same ratio
    # expressed as a percent-of-100g).
    total_comp_moles_per_100g = comp_moles_liq.sum(axis=1)

    liq_mass_idx = mi.mass_phasedict[liq_key]
    liq_mass_frac = mass_np[:, liq_mass_idx]
    phase_composition[liq_key] = np.ones((B, 1))
    phase_moles[liq_key] = total_comp_moles_per_100g * liq_mass_frac / 100.0  # per-100g-system now

    return {
        'phase_composition': phase_composition, 'phase_moles': phase_moles,
        'liquid_oxides': liquid_moles_np, 'liquid_oxide_labels': liquid_oxide_labels,
        'mass_wtpct_by_phase': mass_np, 'mass_phasedict': dict(mi.mass_phasedict),
    }


# --------------------------------------------------------------------------- #
# phase-name reconciliation: raw-file / ml_indexer naming -> the single
# plotting/GT-comparison convention used by _melts_phase_plotting.py
# (MELTS_PHASE_STACK_ORDER) -- 'melts-liquid' and 'alkali-feldspar'.
# --------------------------------------------------------------------------- #
def _gt_to_plot_phase_name(name: str) -> str:
    return 'melts-liquid' if name in _GT_LIQUID_NAMES else name


def _ml_to_plot_phase_name(name: str) -> str:
    return 'alkali-feldspar' if name == 'k-feldspar' else name


def _rel_stats(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
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
# property comparison
# --------------------------------------------------------------------------- #
# EOS property key -> (short label, EOS-output -> GT-column unit scale). GT
# System_main_tbl.txt reports S/Cp/V/H as TOTAL-system extensive quantities
# (J/K, J/K, cc, J respectively) -- both sides are converted to per-gram
# specific values before comparison (see _specific_properties), matching the
# isentropic model's own gram-mass normalization. melts_vec's 'V' is J/bar
# (*10 -> cc, since 1 J/bar = 10 cc); S/Cp/H need no unit scale, only the
# molar->specific conversion. No rho/VP/VS here (unlike HeFESTo/burnman,
# melts_vec's parameter tables don't carry per-endmember elastic moduli).
_MELTS_PROPERTY_KEYS = ('S', 'Cp', 'V', 'H')
_MELTS_PROPERTY_SCALE = {'S': 1.0, 'Cp': 1.0, 'V': 10.0, 'H': 1.0}


def _phase_moles_total(phase_moles: Dict[str, np.ndarray]) -> np.ndarray:
    total = None
    for v in phase_moles.values():
        v = np.nan_to_num(np.asarray(v, dtype=np.float64))
        total = v.copy() if total is None else total + v
    return total


def _specific_properties(props: Dict[str, np.ndarray], phase_moles: Dict[str, np.ndarray],
                          system_mass: np.ndarray) -> Dict[str, np.ndarray]:
    """Convert get_property_melts_vectorized_from_assemblage's bulk MOLAR
    outputs (per total moles of the assemblage) into per-gram specific
    values, via each row's own mean molar mass = system_mass / total_moles.

    ``system_mass``: the REAL total system mass in grams for
    ``gt_assemblage_internal_eos`` (phase_moles there -- from
    load_gt_assemblage -- are real extensive moles for that real mass); the
    constant ``100.0`` for the emulator-predicted pathways (phase_moles there
    -- from ml_indexer_to_melts_vec_assemblage -- are moles per 100 g of a
    nominal reference system, since mass_wtpct is percent-of-system).
    """
    total_moles = _phase_moles_total(phase_moles)
    mean_mm = np.where(total_moles > 0, system_mass / total_moles, np.nan)  # g/mol
    out = {}
    for key in _MELTS_PROPERTY_KEYS:
        if key not in props:
            continue
        out[key] = (np.asarray(props[key], dtype=np.float64) * _MELTS_PROPERTY_SCALE[key]) / mean_mm
    return out


def _gt_property_table(sysmain: pd.DataFrame, mass_g: np.ndarray) -> Dict[str, np.ndarray]:
    return {key: sysmain[key].values.astype(np.float64) / mass_g for key in _MELTS_PROPERTY_KEYS}


def _emulator_forward(api, oxide_cols, P_bar, second_col, second_header, bulk):
    """Run ForwardMB for one condition column (T or S) against the real
    evolving bulk oxide composition (including its real, already fO2-
    buffered FeO/Fe2O3 split -- this IS "the evolving Fe3/FeT ratio" for a
    closed-system emulator input; see module docstring). Returns
    (component_moles, comp_wtpct, mass_wtpct)."""
    headers = ['Pressure(System_main)', second_header] + list(oxide_cols)
    table = np.column_stack(
        [P_bar, second_col] + [bulk[ox].fillna(0.0).values for ox in oxide_cols]
    ).astype(np.float32)
    with torch.no_grad():
        out = api.ForwardMB(table, headers=headers, outputs=['component_moles', 'phase_tables'])
    comp_wtpct, mass_wtpct = out['phase_tables']
    return out['component_moles'], comp_wtpct, mass_wtpct


def run_melts_property_comparison(
    api,
    standards_dir=None,
    out_dir='.',
    *,
    rocks: Sequence[str] = ROCKS,
    variants: Sequence[str] = VARIANTS,
    stats_name: str = 'melts_property_errors.csv',
    fig_prefix: str = 'melts_property_comparison',
) -> Dict[str, object]:
    """Compare bulk specific properties (S, Cp, V, H; J or cc per gram) across
    the 6 rock x Cr/NoCr rows against the raw alphaMELTS ground truth. Writes
    two figures (isothermal-pathway, isentropic-pathway; one row per rock x
    Cr/NoCr combination, matching HeFESTo's run_meltstable_property_comparison
    layout) and one wide stats table (three rows per rock x Cr/NoCr: the
    'emulation_isentropic' / 'emulation_isothermal' / 'gt_assemblage_internal_eos'
    sources described in the module docstring).
    """
    standards_dir = Path(standards_dir) if standards_dir is not None else DEFAULT_STANDARDS_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_alkali_feldspar_alias(api)

    results = []
    stat_rows = []
    for variant in variants:
        oxide_cols = _EMULATOR_OXIDE_COLS_CR if variant == 'Cr' else _EMULATOR_OXIDE_COLS_NOCR
        api_sub = api.cr if variant == 'Cr' else api.nocr
        for rock in rocks:
            rock_dir = standards_dir / variant / rock
            if not rock_dir.exists():
                warnings.warn(f"No standards directory for {rock}/{variant}: {rock_dir}")
                continue
            name = f'{rock}_{variant}'

            sysmain = read_system_main(rock_dir)
            bulk = read_bulk_comp(rock_dir)
            # T_C: alphaMELTS's own native Celsius value, straight from
            # System_main_tbl.txt -- this is what the isothermal/openox
            # emulator's 'Temperature(System_main)' feature actually expects
            # (confirmed against API.py's make_ptt_out, which reads that same
            # feature back directly as T_C with no offset). T_K is Kelvin,
            # needed ONLY for melts_vec's internal Berman-EOS PT tuple (via
            # get_property_melts_vectorized_from_assemblage) -- NEVER pass T_K
            # to the emulator under the 'Temperature(System_main)' header, or
            # every isothermal-pathway forward pass silently runs ~273 degrees
            # too hot (this was a real bug here previously -- see §15 update).
            T_C = sysmain['Temperature'].values
            T_K = T_C + 273.15
            P_bar = sysmain['Pressure'].values
            P_GPa = P_bar / _BARS_PER_GPA
            mass_g = sysmain['mass'].values
            S_specific_gt = sysmain['S'].values / mass_g

            gt_table = _gt_property_table(sysmain, mass_g)

            # source 1: real GT assemblage through the internal vectorized EOS
            gt_asm = load_gt_assemblage(rock_dir, api)
            gt_props_raw = api.get_property_melts_vectorized_from_assemblage(
                gt_asm['phase_composition'], gt_asm['phase_moles'], gt_asm['PT'],
                property_names=(*_MELTS_PROPERTY_KEYS, 'melts_coverage_fraction'),
                liquid_oxides=gt_asm['liquid_oxides'], liquid_oxide_labels=gt_asm['liquid_oxide_labels'],
            )
            gt_cov = gt_props_raw.pop('melts_coverage_fraction')
            gt_props = _specific_properties(gt_props_raw, gt_asm['phase_moles'], mass_g)

            # source 2: emulator-predicted assemblage, isothermal (P, T) pathway.
            # Feed T_C (Celsius, alphaMELTS's own native units), NOT T_K --
            # the emulator's 'Temperature(System_main)' feature is Celsius.
            cm_iso, comp_iso, mass_iso = _emulator_forward(
                api, oxide_cols, P_bar, T_C, 'Temperature(System_main)', bulk)
            bridged_iso = ml_indexer_to_melts_vec_assemblage(
                api_sub, api, api_sub.isothermal_emulator, cm_iso, mass_iso, comp_iso)
            PT_iso = np.column_stack([P_GPa, T_K])
            props_iso_raw = api.get_property_melts_vectorized_from_assemblage(
                bridged_iso['phase_composition'], bridged_iso['phase_moles'], PT_iso,
                property_names=(*_MELTS_PROPERTY_KEYS, 'melts_coverage_fraction'),
                liquid_oxides=bridged_iso['liquid_oxides'], liquid_oxide_labels=bridged_iso['liquid_oxide_labels'],
            )
            cov_iso = props_iso_raw.pop('melts_coverage_fraction')
            props_iso = _specific_properties(props_iso_raw, bridged_iso['phase_moles'], 100.0)

            # source 3: emulator-predicted assemblage, isentropic (P, S) pathway.
            # No MELTS120 temperature model exists to predict T from S, so the
            # downstream EOS evaluation borrows the row's real GT T -- the
            # predicted ASSEMBLAGE itself is still driven purely from real P, S.
            cm_isen, comp_isen, mass_isen = _emulator_forward(
                api, oxide_cols, P_bar, S_specific_gt, 'S(System_main)', bulk)
            bridged_isen = ml_indexer_to_melts_vec_assemblage(
                api_sub, api, api_sub.isentropic_emulator, cm_isen, mass_isen, comp_isen)
            PT_isen = np.column_stack([P_GPa, T_K])
            props_isen_raw = api.get_property_melts_vectorized_from_assemblage(
                bridged_isen['phase_composition'], bridged_isen['phase_moles'], PT_isen,
                property_names=(*_MELTS_PROPERTY_KEYS, 'melts_coverage_fraction'),
                liquid_oxides=bridged_isen['liquid_oxides'], liquid_oxide_labels=bridged_isen['liquid_oxide_labels'],
            )
            cov_isen = props_isen_raw.pop('melts_coverage_fraction')
            props_isen = _specific_properties(props_isen_raw, bridged_isen['phase_moles'], 100.0)

            sources = {
                'emulation_isentropic': (props_isen, cov_isen),
                'emulation_isothermal': (props_iso, cov_iso),
                'gt_assemblage_internal_eos': (gt_props, gt_cov),
            }
            for src_name, (props, cov) in sources.items():
                row = {
                    'rock': rock, 'variant': variant, 'source': src_name, 'n': int(len(P_bar)),
                    'coverage_min': float(np.nanmin(cov)), 'coverage_mean': float(np.nanmean(cov)),
                }
                for key in _MELTS_PROPERTY_KEYS:
                    s = _rel_stats(gt_table[key], props.get(key, np.full(len(P_bar), np.nan)))
                    for stat, val in s.items():
                        row[f'{key} {stat}'] = val
                stat_rows.append(row)

            results.append({
                # These are isobaric-COOLING runs (fixed P, varying T within a
                # rock's own run -- confirmed against System_main_tbl.txt/
                # Phase_mass_tbl.txt: P is constant per rock, T decreases
                # monotonically with row index), so T (not P, which is a
                # single value here) is the meaningful x-axis for plotting.
                'name': name, 'rock': rock, 'variant': variant,
                'T_C': sysmain['Temperature'].values, 'P_GPa': P_GPa,
                'gt_table': gt_table, 'gt': gt_props, 'iso': props_iso, 'isen': props_isen,
            })

    stats = pd.DataFrame(stat_rows).set_index(['rock', 'variant', 'source'])
    stats_path = out_dir / stats_name
    stats.to_csv(stats_path, na_rep='--')

    figs = {
        'isothermal': _plot_melts_properties(results, 'isothermal', out_dir / f'{fig_prefix}_isothermal.png'),
        'isentropic': _plot_melts_properties(results, 'isentropic', out_dir / f'{fig_prefix}_isentropic.png'),
    }
    return {'property_errors': stats, 'stats_path': stats_path, 'figures': figs}


def _plot_melts_properties(results, mode, save_path):
    prop_keys = _MELTS_PROPERTY_KEYS
    n_cols = len(prop_keys)
    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.2 * n_rows), squeeze=False)
    src_key = 'isen' if mode == 'isentropic' else 'iso'

    for r, res in enumerate(results):
        T = res['T_C']
        for c, key in enumerate(prop_keys):
            ax = axes[r][c]
            ax.plot(T, res['gt_table'][key], 'k-', lw=1.5, label='Raw alphaMELTS (GT)')
            ax.plot(T, res[src_key].get(key, np.full_like(T, np.nan)), 'r--', lw=1.4,
                    label='Emulator (pred. assemblage)')
            ax.plot(T, res['gt'].get(key, np.full_like(T, np.nan)), 'b:', lw=1.6,
                    label='GT assemblage + internal EOS')
            ax.set_xlabel('T (°C)')
            if c == 0:
                ax.set_ylabel(f"{res['name']} (P={res['P_GPa'][0]:.3g} GPa)\n{key}", fontsize=9)
            else:
                ax.set_ylabel(key, fontsize=9)
            ax.grid(True, alpha=0.25)
            if r == 0:
                ax.set_title(key, fontsize=10)
            if r == 0 and c == n_cols - 1:
                ax.legend(fontsize=7, loc='best')

    lbl = 'Isothermal (P, T)' if mode == 'isothermal' else 'Isentropic (P, S) -- T borrowed from GT'
    fig.suptitle(f'{lbl} pathway: MELTS120 emulator vs raw alphaMELTS ground truth', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=150, facecolor='white')
    plt.close(fig)
    return Path(save_path)


# --------------------------------------------------------------------------- #
# phase (mass-fraction) comparison
# --------------------------------------------------------------------------- #
def _gt_phase_mass_fractions(rock_dir: Path, full_index: np.ndarray,
                              mass_g: np.ndarray) -> Dict[str, np.ndarray]:
    """{plot_phase_name: (B,) mass fraction of system}, from Phase_mass_tbl.txt
    (absolute grams per phase) over the row's real system mass -- matching
    Phase_mass_tbl.txt's own mass basis (see module docstring)."""
    phase_mass_df, phase_col_names = read_phase_mass(rock_dir)
    phase_mass_df = _reindex_to(full_index, phase_mass_df, index_col='index')
    out: Dict[str, np.ndarray] = {}
    for col, phase_name in phase_col_names:
        plot_name = _gt_to_plot_phase_name(phase_name)
        grams = phase_mass_df[col].fillna(0.0).values.astype(np.float64)
        frac = np.where(mass_g > 0, grams / mass_g, 0.0)
        out[plot_name] = out.get(plot_name, np.zeros_like(frac)) + frac
    return out


def _emulator_phase_mass_fractions(api, emulator, oxide_cols, P_bar, second_col,
                                    second_header, bulk) -> Dict[str, np.ndarray]:
    """{plot_phase_name: (B,) mass fraction of system}, from the SAME
    ForwardMB 'phase_tables' mass output used by ml_indexer_to_melts_vec_
    assemblage (percent-of-system, /100 -> fraction)."""
    headers = ['Pressure(System_main)', second_header] + list(oxide_cols)
    table = np.column_stack(
        [P_bar, second_col] + [bulk[ox].fillna(0.0).values for ox in oxide_cols]
    ).astype(np.float32)
    with torch.no_grad():
        out = api.ForwardMB(table, headers=headers, outputs=['phase_tables'])
    _, mass_wtpct = out['phase_tables']
    mass_np = mass_wtpct.detach().cpu().numpy().astype(np.float64) / 100.0
    mi = emulator.ml_indexer
    result: Dict[str, np.ndarray] = {}
    for phase_name, idx in mi.mass_phasedict.items():
        plot_name = _ml_to_plot_phase_name(phase_name)
        result[plot_name] = result.get(plot_name, np.zeros(mass_np.shape[0])) + mass_np[:, idx]
    return result


def run_melts_phase_comparison(
    api,
    standards_dir=None,
    out_dir='.',
    *,
    rocks: Sequence[str] = ROCKS,
    variants: Sequence[str] = VARIANTS,
    stats_name: str = 'melts_phase_errors.csv',
    fig_prefix: str = 'melts_phase_comparison',
) -> Dict[str, object]:
    """Stacked phase-mass-fraction diagrams (GT vs emulator-predicted
    assemblage, both isothermal and isentropic pathways) across the 6 rock x
    Cr/NoCr rows, plus a per-phase mass-fraction error table. Mirrors
    HeFESTo's run_meltstable_phase_comparison layout: two figures (one row
    per rock x Cr/NoCr combination), one stats table.
    """
    standards_dir = Path(standards_dir) if standards_dir is not None else DEFAULT_STANDARDS_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_alkali_feldspar_alias(api)

    ordered_phases = build_ordered_melts_phases(MELTS_PHASE_STACK_ORDER)
    colors = melts_phase_colors()

    results = []
    stat_rows = []
    for variant in variants:
        oxide_cols = _EMULATOR_OXIDE_COLS_CR if variant == 'Cr' else _EMULATOR_OXIDE_COLS_NOCR
        api_sub = api.cr if variant == 'Cr' else api.nocr
        for rock in rocks:
            rock_dir = standards_dir / variant / rock
            if not rock_dir.exists():
                warnings.warn(f"No standards directory for {rock}/{variant}: {rock_dir}")
                continue
            name = f'{rock}_{variant}'

            sysmain = read_system_main(rock_dir)
            bulk = read_bulk_comp(rock_dir)
            full_index = sysmain['index'].values.astype(int)
            # T_C (Celsius, alphaMELTS's own native units) -- NOT T_K -- is
            # what the emulator's 'Temperature(System_main)' feature expects;
            # see the matching comment in run_melts_property_comparison.
            T_C = sysmain['Temperature'].values
            P_bar = sysmain['Pressure'].values
            P_GPa = P_bar / _BARS_PER_GPA
            mass_g = sysmain['mass'].values
            S_specific_gt = sysmain['S'].values / mass_g

            gt_mf = _gt_phase_mass_fractions(rock_dir, full_index, mass_g)
            iso_mf = _emulator_phase_mass_fractions(
                api, api_sub.isothermal_emulator, oxide_cols, P_bar, T_C,
                'Temperature(System_main)', bulk)
            isen_mf = _emulator_phase_mass_fractions(
                api, api_sub.isentropic_emulator, oxide_cols, P_bar, S_specific_gt,
                'S(System_main)', bulk)

            zeros = np.zeros_like(P_bar)
            for pathway, pred in (('isothermal', iso_mf), ('isentropic', isen_mf)):
                assemblage_abs_err = np.zeros_like(P_bar)
                for phase in ordered_phases:
                    gt_v = gt_mf.get(phase, zeros)
                    pr_v = pred.get(phase, zeros)
                    if gt_v.max() <= 0 and pr_v.max() <= 0:
                        continue
                    abs_err = np.abs(pr_v - gt_v)
                    assemblage_abs_err = assemblage_abs_err + abs_err
                    stat_rows.append({
                        'rock': rock, 'variant': variant, 'pathway': pathway, 'phase': phase,
                        'gt_peak_fraction': float(gt_v.max()),
                        'mean_abs_err': float(abs_err.mean()), 'max_abs_err': float(abs_err.max()),
                    })
                stat_rows.append({
                    'rock': rock, 'variant': variant, 'pathway': pathway, 'phase': '__assemblage_L1__',
                    'gt_peak_fraction': np.nan,
                    'mean_abs_err': float(assemblage_abs_err.mean()),
                    'max_abs_err': float(assemblage_abs_err.max()),
                })

            results.append({
                # Isobaric-cooling runs (fixed P, varying T) -- see the T_C
                # comment in run_melts_property_comparison; T is the x-axis.
                'name': name, 'rock': rock, 'variant': variant,
                'T_C': sysmain['Temperature'].values, 'P_GPa': P_GPa,
                'gt': gt_mf, 'isothermal': iso_mf, 'isentropic': isen_mf,
            })

    stats = pd.DataFrame(stat_rows).set_index(['rock', 'variant', 'pathway', 'phase'])
    stats_path = out_dir / stats_name
    stats.to_csv(stats_path, na_rep='--')

    figs = {
        'isothermal': _plot_melts_phases(results, 'isothermal', ordered_phases, colors,
                                          out_dir / f'{fig_prefix}_isothermal.png'),
        'isentropic': _plot_melts_phases(results, 'isentropic', ordered_phases, colors,
                                          out_dir / f'{fig_prefix}_isentropic.png'),
    }
    return {'phase_errors': stats, 'stats_path': stats_path, 'figures': figs}


def _plot_melts_phases(results, mode, ordered_phases, colors, save_path):
    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 3.4 * n_rows), squeeze=False)
    for r, res in enumerate(results):
        T = res['T_C']
        p_lbl = f"(P={res['P_GPa'][0]:.3g} GPa)"
        draw_melts_phase_stack(axes[r][0], res['gt'], T, ordered_phases, colors,
                                f"{res['name']} {p_lbl} -- Raw alphaMELTS (GT)")
        draw_melts_phase_stack(axes[r][1], res[mode], T, ordered_phases, colors,
                                f"{res['name']} {p_lbl} -- Emulator ({mode})")
        for ax in axes[r]:
            ax.set_xlabel('T (°C)')
    lbl = 'Isothermal (P, T)' if mode == 'isothermal' else 'Isentropic (P, S)'
    fig.suptitle(f'Phase mass fractions -- {lbl} pathway: GT vs emulator', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=150, facecolor='white')
    plt.close(fig)
    return Path(save_path)
