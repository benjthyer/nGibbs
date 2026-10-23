"""
melts_assemblage.py
-------------------
Bridge from a MELTS emulator's raw forward-pass outputs to the vectorised
MELTS EOS (melts_vec), and from there to PetThermoTools (ptt) style
per-phase property tables.

Two layers:

* ``ml_indexer_to_melts_vec_assemblage`` -- one ``ForwardMB(...,
  outputs=['component_moles', 'phase_tables'])`` call (in the emulator's own
  ml_indexer basis) -> melts_vec's ``phase_composition`` / ``phase_moles`` /
  ``liquid_oxides`` inputs for
  ``MELTSAPI.get_property_melts_vectorized_from_assemblage``. Moved here
  verbatim from ``deployment_tests/melts_comparison.py`` (which now imports
  it from here) so inference no longer depends on the deployment-test
  module and its matplotlib import.
* ``ptt_phase_props`` -- evaluates melts_vec on that assemblage (only on the
  rows where each phase is present) and writes ``'<phase>_prop'`` tables plus
  bulk ``'Conditions'`` columns into a ``make_ptt_out`` dict, with the same
  layout PetThermoTools' alphaMELTS-backed ``calc_phase_props_MELTS_parallel``
  produces. This is what ``EmulatorAPI.ForwardMB(..., outputs=['ptt_out',
  'ptt_props'])`` calls for MELTS models.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ngibbs.config.constants import get_oxide_molar_mass
from ngibbs.engine.EOS_arithmetic_MELTS.melts_vec import molar_masses as _melts_molar_masses
from ngibbs.engine.EOS_arithmetic_MELTS.melts_vec import oxides_to_liquid_components as _oxides_to_liquid_components
from ngibbs.engine.EOS_arithmetic_MELTS.melts_vec import molar_mass_from_formula as _melts_molar_mass_from_formula
from ngibbs.engine.EOS_arithmetic_MELTS.melts_vec import MELTS_CARBON_PHASES as _MELTS_CARBON_PHASES

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
    'garnet': ['almandine', 'grossular', 'pyrope'],
    'leucite': ['leucite', 'analcime', 'na-leucite'],
    'biotite': ['annite', 'phlogopite'],
    'hornblende': ['pargasite', 'ferropargasite', 'magnesiohastingsite'],
    'nepheline': ['na-nepheline', 'k-nepheline', 'vc-nepheline', 'ca-nepheline'],
}
# melts_vec's own _MELTS_SOLUTION_PHASES recognises 'feldspar'/'plagioclase' but
# has no 'alkali-feldspar' entry (alphaMELTS models plagioclase and alkali
# feldspar as two immiscible solutions over the SAME albite-anorthite-sanidine
# ternary -- confirmed by alkali-feldspar.tbl and plagioclase.tbl sharing
# identical endmember columns). MELTSAPI._MELTS_SOLUTION_PHASES carries
# 'alkali-feldspar' / 'k-feldspar' aliases of its feldspar model for this
# (deployment_tests' `_ensure_alkali_feldspar_alias` predates them and is now
# a no-op) rather than routing alkali-feldspar through the (wrong) pure-phase
# fallback.
_GT_PURE_PHASES = ('quartz', 'tridymite', 'apatite', 'whitlockite', 'muscovite') + tuple(_MELTS_CARBON_PHASES)
# Phase_mass_tbl.txt names the liquid column 'liquid1' (de-suffixed: 'liquid'),
# while its own per-row data lives in 'melts-liquid.tbl' -- two different
# names for the same phase across these two raw-output files.
_GT_LIQUID_NAMES = ('liquid', 'melts-liquid')
# fluid.tbl: 1.1/1.2 Duan H2O-CO2 fluid (h2oduan, co2duan mole fractions);
# 1.0.2 writes its pure-water phase to fluid.tbl as well (no endmember columns).
_GT_FLUID_NAMES = ('fluid', 'water')
_FLUID_MOLAR_MASSES = np.array([_melts_molar_mass_from_formula('H2O'), _melts_molar_mass_from_formula('CO2')])


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

    Solid solution phases: component_moles is first mapped to native MELTS
    endmember moles with ``emulator.native_component_moles`` -- pyroxenes and
    spinel are learned in the transformed PxSp_Comp_TransformV2 basis, and
    reading those columns as endmember amounts gave badly wrong pyroxene and
    spinel compositions (e.g. orthopyroxene FeO 0.1 wt% instead of 16), the
    cause of the MORB emulator-pathway H/V/Cp errors. The native columns are
    then embedded into melts_vec's endmember order by exact NAME match (verified
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
    cm_np = np.asarray(emulator.native_component_moles(
        component_moles.detach().cpu().numpy().astype(np.float64)))
    mass_np = mass_wtpct.detach().cpu().numpy().astype(np.float64)  # (B, n_phases) mass fraction of system

    phase_composition: Dict[str, np.ndarray] = {}
    phase_moles: Dict[str, np.ndarray] = {}

    # A phase's component_moles columns are read from phaseToCompMap (the
    # matrix make_phase_tables itself uses), NOT label_indices_comp: on the
    # released MELTS checkpoints label_indices_comp is stale for every phase
    # stored after the pure phases (fluid, hornblende, melts-liquid) -- e.g.
    # it points hornblende at the quartz/tridymite/muscovite columns, which
    # gave hornblende zero moles. phaseToCompMap's rows follow mass_phasedict
    # and its columns reproduce components_in_phases' names exactly.
    phase_to_comp = np.asarray(mi.phaseToCompMap)
    for phase_name in mi.label_indices_comp:
        if phase_name in _GT_LIQUID_NAMES:
            continue
        solution_key = 'plagioclase' if phase_name == 'k-feldspar' else phase_name
        if solution_key not in _GT_SOLUTION_ENDMEMBERS:
            continue  # tracked by this checkpoint but melts_vec has no mixing model for it
        endmembers = _GT_SOLUTION_ENDMEMBERS[solution_key]
        ml_names = mi.components_in_phases[phase_name]
        comp_idx = np.nonzero(phase_to_comp[mi.mass_phasedict[phase_name]])[0]
        if len(comp_idx) != len(ml_names):
            raise ValueError(f"[{phase_name}] phaseToCompMap gives {len(comp_idx)} component columns "
                             f"but components_in_phases lists {len(ml_names)} ({ml_names}).")
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

    # Fluid: same route as the liquid -- its own oxide wt% (H2O, CO2 columns of
    # comp_wtpct) gives the [H2O, CO2] mole fractions, 'phase_tables' mass the
    # abundance. A 1.0.2 fluid carrying CO2 is rejected downstream by the API.
    # A 1.0.2 checkpoint (no CO2 tracked) has only a fluid mass column: its
    # pure-water fluid has a fixed composition, so none is predicted -- it is
    # bridged as pure H2O.
    fluid_key = next((k for k in _GT_FLUID_NAMES if k in mi.mass_phasedict), None)
    if fluid_key is not None:
        ox = list(mi.Oxides)
        flu_mass = mass_np[:, mi.mass_phasedict[fluid_key]]
        if fluid_key not in mi.comp_phasedict:
            if 'CO2' in ox:
                raise KeyError(f"ml_indexer has a '{fluid_key}' mass column but no composition slot for it.")
            present = flu_mass > 0
            X = np.tile([1.0, 0.0], (B, 1))
        else:
            if 'H2O' not in ox:
                raise KeyError(f"ml_indexer Oxides {ox} lack H2O; cannot build the fluid composition.")
            flu_wt = comp_wtpct_np[:, mi.comp_phasedict[fluid_key], :]
            co2 = flu_wt[:, ox.index('CO2')] if 'CO2' in ox else np.zeros(B)
            n = np.column_stack([flu_wt[:, ox.index('H2O')], co2]) / _FLUID_MOLAR_MASSES
            tot = n.sum(1)
            present = (tot > 0) & (flu_mass > 0)
            X = np.where(present[:, None], n / np.where(tot > 0, tot, 1.0)[:, None], 0.0)
            X[~present] = [1.0, 0.0]
        phase_composition['fluid'] = X
        phase_moles['fluid'] = np.where(present, flu_mass / (X @ _FLUID_MOLAR_MASSES), 0.0)

    return {
        'phase_composition': phase_composition, 'phase_moles': phase_moles,
        'liquid_oxides': liquid_moles_np, 'liquid_oxide_labels': liquid_oxide_labels,
        'mass_wtpct_by_phase': mass_np, 'mass_phasedict': dict(mi.mass_phasedict),
    }


# --------------------------------------------------------------------------- #
# melts_vec assemblage -> PetThermoTools '<phase>_prop' tables
# --------------------------------------------------------------------------- #
# Same columns, in the same order, as PetThermoTools'
# MELTS.calc_phase_props_MELTS_parallel, so downstream ptt code sees no
# difference. Extensive columns are for the phase's actual mass in the 100 g
# system (mass_g); 'dPdT_bar/K', 'molwt' and 'rho_kg/m^3' are intensive.
# ('Cp_J/(kg.K^2)' / 'dCpdT_J/(kg.K^2)' keep ptt's names, but -- exactly like
# the alphaMELTS path -- hold the phase's extensive Cp (J/K) / dCp/dT (J/K^2).)
# The second-volume-derivative columns are not computed by melts_vec and are
# left NaN.
PTT_PROP_COLUMNS = [
    'mass_g', 'rho_kg/m^3', 'V_cm^3', 'G_J', 'H_J', 'S_J/K',
    'Cp_J/(kg.K^2)', 'dCpdT_J/(kg.K^2)', 'dVdT_cm^3/K', 'dPdT_bar/K',
    'dVdP_cm^3/bar', 'd2VdT2_cm^3/K^2', 'd2VdTdP_cm^3/(bar.K)',
    'd2VdP2_cm^3/bar^2', 'molwt',
]
# ptt column -> (melts_vec per-phase key, factor turning its per-mole value
# into the column's unit; V-like keys are J/bar per mole -> *10 for cm^3).
_PTT_EXTENSIVE = {
    'V_cm^3': ('V', 10.0),
    'G_J': ('G', 1.0),
    'H_J': ('H', 1.0),
    'S_J/K': ('S', 1.0),
    'Cp_J/(kg.K^2)': ('Cp', 1.0),
    'dCpdT_J/(kg.K^2)': ('dCpdT', 1.0),
    'dVdT_cm^3/K': ('dVdT', 10.0),
    'dVdP_cm^3/bar': ('dVdP', 10.0),
}
_BARS_PER_GPA = 10000.0


def _as_rows(v) -> np.ndarray:
    """(B,) float64 view of a per-phase EOS output; the pure-phase branch of
    get_property_melts_vectorized_from_assemblage leaves some keys (B, 1)."""
    v = np.asarray(v, dtype=np.float64)
    return v.reshape(v.shape[0]) if v.ndim == 2 and v.shape[1] == 1 else v


def _assemblage_key(internal_name: str) -> str:
    """ml_indexer phase name -> the key ml_indexer_to_melts_vec_assemblage
    stores that phase under."""
    if internal_name == 'k-feldspar':
        return 'alkali-feldspar'
    if internal_name in _GT_FLUID_NAMES:
        return 'fluid'
    return internal_name


def ptt_phase_props(api, ptt_out: Dict[str, pd.DataFrame], bridged: Dict[str, object],
                    phase_mappings: Sequence[tuple], T_C: np.ndarray, P_bar: np.ndarray
                    ) -> Dict[str, pd.DataFrame]:
    """Add melts_vec phase and bulk properties to a ``make_ptt_out`` dict.

    api : MELTSAPI
        Holds the melts_vec parameter tables and
        get_property_melts_vectorized_from_assemblage.
    ptt_out : dict of DataFrames
        ``EmulatorAPI.make_ptt_out`` output for the same rows. Modified in
        place (and returned): a ``'<ptt phase>_prop'`` DataFrame is added for
        every phase in ``ptt_out['mass_g']`` that melts_vec covers, and
        ``'Conditions'`` gains the bulk columns (mass_g, H_J, S_J/K, V_cm^3,
        rho_kg/m^3, log10(fO2), dVdP_cm^3/bar, Cp_J/K) summed over those
        phases.
    bridged : dict
        ``ml_indexer_to_melts_vec_assemblage`` output for the same forward pass.
    phase_mappings : sequence of (ptt_long, ptt_short, internal_name, mass_col)
        ``EmulatorAPI._ptt_phase_mappings`` for the same forward pass.
    T_C, P_bar : (B,) arrays
        Conditions to evaluate the EOS at (Celsius, bar).

    The EOS is evaluated only on the rows where a phase has nonzero moles, so
    absent phases cost nothing. A phase melts_vec cannot evaluate (no model,
    or non-finite properties, e.g. nepheline without vc-nepheline) gets an
    all-NaN prop table, a warning, and is left out of the bulk sums.
    """
    T_C = np.asarray(T_C, dtype=np.float64)
    P_bar = np.asarray(P_bar, dtype=np.float64)
    n = T_C.shape[0]
    PT = np.column_stack([P_bar / _BARS_PER_GPA, T_C + 273.15])  # [GPa, K]
    phase_composition = bridged['phase_composition']
    phase_moles = bridged['phase_moles']
    mass_g = ptt_out['mass_g']

    totals = {k: np.zeros(n) for k in ('mass_g', 'H_J', 'S_J/K', 'V_cm^3', 'dVdP_cm^3/bar', 'Cp_J/(kg.K^2)')}
    for ptt_long, _short, internal_name, _col in phase_mappings:
        prop = np.full((n, len(PTT_PROP_COLUMNS)), np.nan)
        col = {c: i for i, c in enumerate(PTT_PROP_COLUMNS)}
        key = _assemblage_key(internal_name)
        phase_mass = mass_g[ptt_long].to_numpy(dtype=np.float64)
        if key not in phase_composition:
            warnings.warn(f"nGibbs ptt_props: no melts_vec model for phase '{internal_name}'; "
                          f"'{ptt_long}_prop' left NaN and excluded from the bulk properties.")
            ptt_out[ptt_long + '_prop'] = pd.DataFrame(prop, columns=PTT_PROP_COLUMNS)
            continue

        moles = np.asarray(phase_moles[key], dtype=np.float64)
        rows = np.nonzero((moles > 0.0) & np.isfinite(moles) & (phase_mass > 0.0))[0]
        if rows.size:
            kwargs = {}
            if key in _GT_LIQUID_NAMES:
                kwargs = dict(liquid_oxides=np.asarray(bridged['liquid_oxides'])[rows],
                              liquid_oxide_labels=bridged['liquid_oxide_labels'])
            try:
                res = api.get_property_melts_vectorized_from_assemblage(
                    {key: np.asarray(phase_composition[key])[rows]}, {key: moles[rows]}, PT[rows],
                    property_names=('per_phase',), **kwargs)['per_phase'][key]
            except ValueError as err:
                warnings.warn(f"nGibbs ptt_props: melts_vec could not evaluate '{internal_name}' "
                              f"({err}); '{ptt_long}_prop' left NaN and excluded from the bulk properties.")
                ptt_out[ptt_long + '_prop'] = pd.DataFrame(prop, columns=PTT_PROP_COLUMNS)
                continue

            n_ph = moles[rows]
            prop[rows, col['mass_g']] = phase_mass[rows]
            for c, (k, f) in _PTT_EXTENSIVE.items():
                prop[rows, col[c]] = _as_rows(res[k]) * f * n_ph
            molwt = _as_rows(res['mass'])
            V_molar_cc = _as_rows(res['V']) * 10.0
            dVdT, dVdP = _as_rows(res['dVdT']), _as_rows(res['dVdP'])
            with np.errstate(divide='ignore', invalid='ignore'):
                prop[rows, col['molwt']] = molwt
                prop[rows, col['rho_kg/m^3']] = np.where(V_molar_cc > 0, molwt / V_molar_cc * 1000.0, np.nan)
                prop[rows, col['dPdT_bar/K']] = np.where(dVdP != 0.0, -dVdT / dVdP, 0.0)
            for c in totals:
                totals[c][rows] += prop[rows, col[c]]
        ptt_out[ptt_long + '_prop'] = pd.DataFrame(prop, columns=PTT_PROP_COLUMNS)

    old = ptt_out['Conditions']
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = np.where(totals['V_cm^3'] > 0, totals['mass_g'] / totals['V_cm^3'] * 1000.0, np.nan)
    if 'log10(fO2)' in old.columns:
        fO2 = old['log10(fO2)'].to_numpy()
    else:
        fO2 = np.full(n, np.nan)
    cond = pd.DataFrame({
        'T_C': T_C,
        'P_bar': P_bar,
        'mass_g': totals['mass_g'],
        'H_J': totals['H_J'],
        'S_J/K': totals['S_J/K'],
        'V_cm^3': totals['V_cm^3'],
        'rho_kg/m^3': rho,
        'log10(fO2)': fO2,
        'dVdP_cm^3/bar': totals['dVdP_cm^3/bar'],
        'Cp_J/K': totals['Cp_J/(kg.K^2)'],
    }, index=old.index)
    for c in old.columns:  # keep anything else make_ptt_out put there (e.g. 's')
        if c not in cond.columns:
            cond[c] = old[c].to_numpy()
    ptt_out['Conditions'] = cond
    return ptt_out
