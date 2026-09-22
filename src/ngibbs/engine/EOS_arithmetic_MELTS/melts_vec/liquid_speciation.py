"""
Convert speciated liquid oxide moles into melts_vec's 19 liquid endmember
components.

Input basis: nGibbs's own oxide/element convention (config.constants.
default_Oxides plus a few oxides nGibbs itself never tracks) -- SiO2, TiO2,
Al2O3, Fe2O3, Cr2O3, FeO, MnO, MgO, NiO, CoO, CaO, Na2O, K2O, P2O5, H2O,
CO2, SO3, Cl, F. Fe2O3/FeO must already be correctly speciated by the
caller (see emulator.NN_MELTS.get_liquid_oxides / Iron_Speciator) -- this
module does no fO2 chemistry at all.

Output basis: melts_vec's 19 liquid components, in melts_vec.liquid_params.
load_liquid().labels order (SiO2, TiO2, Al2O3, Fe2O3, MgCr2O4, Fe2SiO4,
MnSi0.5O2, Mg2SiO4, NiSi0.5O2, CoSi0.5O2, CaSiO3, Na2SiO3, KAlSiO4,
Ca3(PO4)2, CO2, SO3, Cl2O-1, F2O-1, H2O) -- confirmed against MAGMA's own
sources/liquid_v34.c testLiq_v34() NAMES/FORMULAS arrays and liq_struct_
data.json's own `label` field (which IS each component's formula for the
meltsLiquid table).

Why this can't be a single matrix multiply
-------------------------------------------
Every other oxide<->component map in nGibbs (ml_indexer's compToOx, and
this module's own liquid_components_to_oxides() below) is linear, because
going from components to oxides is just "add up what each component
contributes." Going the other way is not linear: SiO2 (and, to a lesser
extent, Al2O3) is a shared "pool" oxide that several components draw from
-- Fe2SiO4, Mg2SiO4, MnSi0.5O2, NiSi0.5O2, CoSi0.5O2, CaSiO3, Na2SiO3 and
KAlSiO4 all consume some SiO2, and KAlSiO4 also consumes Al2O3 -- and what
is left over after all of them have taken their share becomes the "SiO2"
/ "Al2O3" component itself. That "take your share, then see what's left"
structure is exactly what a CIPW norm does, and like a CIPW norm it isn't
something a fixed matrix can express; it has to be evaluated in a specific
order.

MELTS's fixed order (mirrored here) is:
  1. Cr2O3 -> MgCr2O4 (all of it; borrows 1 mol MgO per mol Cr2O3)
  2. P2O5  -> Ca3(PO4)2 (all of it; borrows 3 mol CaO per mol P2O5)
  3. FeO, MnO, remaining MgO, NiO, CoO, remaining CaO, Na2O each go
     entirely to their own single-cation silicate component (Fe2SiO4,
     MnSi0.5O2, Mg2SiO4, NiSi0.5O2, CoSi0.5O2, CaSiO3, Na2SiO3
     respectively), each drawing its own share of SiO2.
  4. K2O -> KAlSiO4 (all of it; borrows Al2O3).
  5. Whatever Al2O3 is left over is the Al2O3 component.
  6. Whatever SiO2 is left over after every step above is the SiO2
     component -- SiO2 is nobody's minority partner, so it is always
     resolved last.
TiO2, Fe2O3, H2O, CO2, SO3, and the two halogen components are already
"pure" (they never share an oxide with anything else), so they are just
copied straight across.

Every per-component "how much SiO2 / Al2O3 does one mole of me consume"
number below is cross-checked in this module's own self-test against the
component's own formula string (parsed the same way melts_vec.molar_mass
already parses solid-endmember formulas), rather than trusting a single
hand-transcribed set of coefficients -- only the priority ORDER above
(which cation gets first claim on a shared pool oxide) is a genuine
modeling choice that formula-parsing alone can't recover, so that part
stays hand-written.

Vectorization: every step here is closed-form elementwise arithmetic
across the batch dimension -- no iteration, no Newton solve -- so despite
not being a single matrix multiply, this is fully vectorized.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

try:
    from .molar_mass import _parse_formula, FORMULA_OVERRIDES as _FORMULA_OVERRIDES
except ImportError:  # pragma: no cover - allows standalone testing/import
    from molar_mass import _parse_formula, FORMULA_OVERRIDES as _FORMULA_OVERRIDES


# meltsLiquid component labels, melts_vec's own canonical order
# (== melts_vec.liquid_params.load_liquid().labels for the standard
# rhyolite-MELTS/pMELTS "meltsLiquid" table).
LIQUID_COMPONENT_LABELS = [
    'SiO2', 'TiO2', 'Al2O3', 'Fe2O3', 'MgCr2O4', 'Fe2SiO4', 'MnSi0.5O2',
    'Mg2SiO4', 'NiSi0.5O2', 'CoSi0.5O2', 'CaSiO3', 'Na2SiO3', 'KAlSiO4',
    'Ca3(PO4)2', 'CO2', 'SO3', 'Cl2O-1', 'F2O-1', 'H2O',
]

# The two labels the general element-counting parser can't handle: a
# literal "-1" subscript on oxygen isn't valid chemistry and doesn't
# appear anywhere else in this project's formula strings. These
# components' whole role is "carry N mol of a halogen"; they don't draw
# on any pool oxide, so their O count is irrelevant here and omitted.
# (Now the single shared definition in molar_mass.FORMULA_OVERRIDES --
# imported above as _FORMULA_OVERRIDES -- so this module and
# molar_mass.liquid_molar_masses() can't silently drift apart.)


def _formula_counts(label: str) -> Dict[str, float]:
    if label in _FORMULA_OVERRIDES:
        return dict(_FORMULA_OVERRIDES[label])
    return _parse_formula(label)


def liquid_component_si_al_coeffs(labels=LIQUID_COMPONENT_LABELS):
    """Per-component (SiO2, Al2O3) consumption coefficients -- moles of the
    pool oxide one mole of this component consumes -- parsed directly from
    each label's own formula string. SiO2 and Al2O3 themselves are pool
    components, not consumers of themselves, so their own coefficients are
    forced to 0 regardless of what the (trivial, single-element) parse of
    "SiO2"/"Al2O3" would otherwise give.
    """
    si = np.zeros(len(labels))
    al = np.zeros(len(labels))
    for i, lbl in enumerate(labels):
        if lbl in ('SiO2', 'Al2O3'):
            continue
        counts = _formula_counts(lbl)
        si[i] = counts.get('Si', 0.0) / 1.0   # 1 Si atom per mol SiO2
        al[i] = counts.get('Al', 0.0) / 2.0   # 2 Al atoms per mol Al2O3
    return si, al


def oxides_to_liquid_components(
    oxides: Dict[str, np.ndarray],
    liquid_params=None,
    return_diagnostics: bool = False,
):
    """
    Convert speciated liquid oxide moles into melts_vec's 19 liquid
    endmember-component moles, following MELTS's own fixed component-
    construction order (see module docstring).

    Parameters
    ----------
    oxides : dict[str, array-like]
        Oxide moles, batched (B,) arrays (plain floats broadcast fine).
        Recognised keys: 'SiO2', 'TiO2', 'Al2O3', 'Fe2O3', 'Cr2O3', 'FeO',
        'MnO', 'MgO', 'NiO', 'CoO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'H2O',
        'CO2', 'SO3', 'Cl', 'F'. Any key not supplied defaults to 0 -- this
        covers every oxide/element nGibbs's own config.constants.
        default_Elkeys/default_Oxides does not currently track (CoO, SO3,
        Cl, F all default to 0 for every current nGibbs checkpoint).
        Fe2O3 and FeO must already be correctly speciated -- this function
        does no fO2 chemistry (see emulator.NN_MELTS.get_liquid_oxides).
    liquid_params : MELTSLiquidParams, optional
        From melts_vec.liquid_params.load_liquid(). If given and its
        .labels order differs from LIQUID_COMPONENT_LABELS, the returned
        array's columns are reindexed to match it.
    return_diagnostics : bool, default=False
        If True, also return {'negative_mask': {...}} flagging rows where
        SiO2, Al2O3, MgO, or CaO went negative under this fixed allocation
        order -- i.e. the bulk composition is more silica-/alumina-/
        magnesia-/lime-undersaturated (relative to its other oxides) than
        this allocation order can support. MELTS's own C code doesn't
        guard against this either (see liquid_v34.c's actLiq_v34/
        gmixLiq_v34 "CAUTION ... negative mole fraction" printfs) -- it's
        a real property of unusual bulk compositions, not a bug, but worth
        surfacing rather than silently feeding a negative mole fraction
        into the mixing model.

    Returns
    -------
    np.ndarray, shape (B, 19)
        Component moles, NOT normalized to mole fractions -- divide by the
        row sum (or let melts_vec.compute_liquid_bulk normalize
        internally) before use.
    dict, optional
        Only when return_diagnostics=True.
    """
    def g(key):
        return np.asarray(oxides.get(key, 0.0), dtype=np.float64)

    keys = ('SiO2', 'TiO2', 'Al2O3', 'Fe2O3', 'Cr2O3', 'FeO', 'MnO', 'MgO',
            'NiO', 'CoO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'H2O', 'CO2', 'SO3',
            'Cl', 'F')
    vals = [g(k) for k in keys]
    B = np.broadcast_shapes(*[v.shape for v in vals])
    (nSiO2, nTiO2, nAl2O3, nFe2O3, nCr2O3, nFeO, nMnO, nMgO, nNiO, nCoO,
     nCaO, nNa2O, nK2O, nP2O5, nH2O, nCO2, nSO3, nCl, nF) = [
        np.broadcast_to(v, B).astype(np.float64) for v in vals
    ]

    # 1. Cr-limited: all Cr2O3 -> MgCr2O4, borrowing 1 mol MgO per mol Cr2O3.
    m_MgCr2O4 = nCr2O3.copy()
    mg_avail = nMgO - m_MgCr2O4

    # 2. P-limited: all P2O5 -> Ca3(PO4)2, borrowing 3 mol CaO per mol P2O5.
    m_Ca3PO42 = nP2O5.copy()
    ca_avail = nCaO - 3.0 * m_Ca3PO42

    # 3. Single-cation silicate components, each drawing its own SiO2 share.
    m_Fe2SiO4 = 0.5 * nFeO          # Fe2SiO4: 2 Fe/formula
    m_MnSi0_5O2 = nMnO.copy()       # MnSi0.5O2: 1 Mn/formula
    m_Mg2SiO4 = 0.5 * mg_avail      # Mg2SiO4: 2 Mg/formula (gets Cr's leftovers)
    m_NiSi0_5O2 = nNiO.copy()       # NiSi0.5O2: 1 Ni/formula
    m_CoSi0_5O2 = nCoO.copy()       # CoSi0.5O2: 1 Co/formula
    m_CaSiO3 = ca_avail.copy()      # CaSiO3: 1 Ca/formula (gets P's leftovers)
    m_Na2SiO3 = nNa2O.copy()        # Na2SiO3: 2 Na/formula = 1 mol Na2O

    # 4. K-limited: all K2O -> KAlSiO4, borrowing 0.5 mol Al2O3 per mol KAlSiO4.
    m_KAlSiO4 = 2.0 * nK2O           # KAlSiO4: 1 K/formula = 0.5 mol K2O
    al_avail = nAl2O3 - 0.5 * m_KAlSiO4

    # 5. Leftover Al2O3 is its own component.
    m_Al2O3 = al_avail

    # Pure (never-shared) components: copied straight across.
    m_TiO2 = nTiO2.copy()
    m_Fe2O3 = nFe2O3.copy()
    m_H2O = nH2O.copy()
    m_CO2 = nCO2.copy()
    m_SO3 = nSO3.copy()
    m_Cl2O_neg1 = 0.5 * nCl
    m_F2O_neg1 = 0.5 * nF

    # 6. Leftover SiO2, last: total SiO2 consumed by every silicate
    #    component above (each component's own Si-atom-count / 1).
    si_consumed = (
        1.0 * m_Fe2SiO4 + 0.5 * m_MnSi0_5O2 + 1.0 * m_Mg2SiO4
        + 0.5 * m_NiSi0_5O2 + 0.5 * m_CoSi0_5O2 + 1.0 * m_CaSiO3
        + 1.0 * m_Na2SiO3 + 1.0 * m_KAlSiO4
    )
    m_SiO2 = nSiO2 - si_consumed

    components = np.stack([
        m_SiO2, m_TiO2, m_Al2O3, m_Fe2O3, m_MgCr2O4, m_Fe2SiO4, m_MnSi0_5O2,
        m_Mg2SiO4, m_NiSi0_5O2, m_CoSi0_5O2, m_CaSiO3, m_Na2SiO3, m_KAlSiO4,
        m_Ca3PO42, m_CO2, m_SO3, m_Cl2O_neg1, m_F2O_neg1, m_H2O,
    ], axis=-1)

    if liquid_params is not None and list(liquid_params.labels) != LIQUID_COMPONENT_LABELS:
        my_index = {lbl: i for i, lbl in enumerate(LIQUID_COMPONENT_LABELS)}
        reindex = [my_index[lbl] for lbl in liquid_params.labels]
        components = components[..., reindex]

    if return_diagnostics:
        diagnostics = {
            'negative_mask': {
                'SiO2': m_SiO2 < 0.0,
                'Al2O3': m_Al2O3 < 0.0,
                'MgO': mg_avail < 0.0,
                'CaO': ca_avail < 0.0,
            }
        }
        return components, diagnostics
    return components


def liquid_components_to_oxides(components, labels=LIQUID_COMPONENT_LABELS) -> Dict[str, np.ndarray]:
    """
    Forward map: melts_vec component moles -> the same speciated-oxide
    basis oxides_to_liquid_components() takes as input. Unlike that
    direction, this one genuinely is a fixed linear map -- provided mainly
    so a round trip (oxides -> components -> oxides) can be checked as a
    mass-balance closure test.
    """
    components = np.asarray(components, dtype=np.float64)
    label_idx = {lbl: i for i, lbl in enumerate(labels)}

    def col(name):
        return components[..., label_idx[name]]

    out = {
        'SiO2': (col('SiO2') + col('Fe2SiO4') + 0.5 * col('MnSi0.5O2')
                  + col('Mg2SiO4') + 0.5 * col('NiSi0.5O2') + 0.5 * col('CoSi0.5O2')
                  + col('CaSiO3') + col('Na2SiO3') + col('KAlSiO4')),
        'TiO2': col('TiO2'),
        'Al2O3': col('Al2O3') + 0.5 * col('KAlSiO4'),
        'Fe2O3': col('Fe2O3'),
        'Cr2O3': col('MgCr2O4'),
        'FeO': 2.0 * col('Fe2SiO4'),
        'MnO': col('MnSi0.5O2'),
        'MgO': col('MgCr2O4') + 2.0 * col('Mg2SiO4'),
        'NiO': col('NiSi0.5O2'),
        'CoO': col('CoSi0.5O2'),
        'CaO': 3.0 * col('Ca3(PO4)2') + col('CaSiO3'),
        'Na2O': col('Na2SiO3'),
        'K2O': 0.5 * col('KAlSiO4'),
        'P2O5': col('Ca3(PO4)2'),
        'H2O': col('H2O'),
        'CO2': col('CO2'),
        'SO3': col('SO3'),
        'Cl': 2.0 * col('Cl2O-1'),
        'F': 2.0 * col('F2O-1'),
    }
    return out
