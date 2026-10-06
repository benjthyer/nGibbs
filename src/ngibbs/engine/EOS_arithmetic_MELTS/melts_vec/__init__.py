"""
melts_vec -- Vectorized Python implementation of the MELTS EOS, translated
from MAGMA's `sources/gibbs.c` and friends.

Quick start
-----------
>>> from melts_vec import load_solids, compute
>>> params = load_solids()   # defaults to the 'meltsSolids' (rhyolite-MELTS) table
>>> result = compute(T, P, params, names=['forsterite', 'fayalite'])
>>> result['V'], result['K'], result['Cp']   # (B, 2) arrays

>>> from melts_vec import load_liquid, compute_liquid_bulk
>>> lparams = load_liquid()
>>> liquid = compute_liquid_bulk_mode(T, P, X, 'MELTS120')   # rhyolite-MELTS 1.2 liquid

>>> from melts_vec import load_solids, compute_feldspar_solution, compute_olivine_solution
>>> sparams = load_solids()
>>> fsp = compute_feldspar_solution(T, P, X_fsp, sparams)
>>> olv = compute_olivine_solution(T, P, X_olv, sparams)

>>> from melts_vec import compute_clinopyroxene_solution, compute_orthopyroxene_solution
>>> cpx = compute_clinopyroxene_solution(T, P, X_cpx, sparams)
>>> opx = compute_orthopyroxene_solution(T, P, X_opx, sparams)

>>> from melts_vec import compute_spinel_solution
>>> spn = compute_spinel_solution(T, P, X_spn, sparams)

>>> from melts_vec import compute_rhm_oxide_solution
>>> rhm = compute_rhm_oxide_solution(T, P, X_rhm, sparams)

>>> from melts_vec import oxides_to_liquid_components
>>> X_liq = oxides_to_liquid_components({'SiO2': ..., 'MgO': ..., ...})   # (B, 19) component moles
>>> liquid = compute_liquid_bulk_mode(T, P, X_liq, 'MELTS120')   # rows normalized inside

Scope
-----
- DONE, verified bit-for-bit against a standalone C harness built from
  MAGMA source formulas (see melts_vec/tests/): the solid-endmember EOS
  (Berman polynomial + Vinet finite-strain volume, Berman/Saxena heat
  capacity), the liquid Kress volume EOS + ideal mixing, and the
  feldspar, olivine, clinopyroxene, orthopyroxene, spinel, and
  rhombohedral-oxide solid-solution mixing models. clinopyroxene.c and
  orthopyroxene.c are
  confirmed (via direct `diff`) to share a byte-identical Taylor-
  coefficient macro engine, forked from one ancestor PYROXENE.C --
  orthopyroxene.py reuses clinopyroxene.py's pure-endmember reference
  frame, essenite ordering, site-fraction, and Darken-matrix code
  directly rather than re-deriving it, adding only its own MIX-branch
  (clino=FALSE) Taylor coefficients, including a nonzero H0/S0/V0
  constant term that has no analogue in clinopyroxene.py (there it is
  exactly 0 and so is omitted by construction) -- see orthopyroxene.py's
  module docstring for the full architecture. Note gmixOpx() itself is
  NOT ~0 at orthopyroxene's own pure-endmember vertices (unlike
  clinopyroxene's gmixCpx()): it deliberately subtracts a clino=TRUE
  ("monoclinic reference frame") pure reference from a clino=FALSE bulk
  G, by the source's own design (see orthopyroxene.py's module
  docstring and the benchmark test file).

  spinel.py has a different structure from either pyroxene phase: there
  is no separate PURE-vs-MIX Taylor-coefficient reference frame at all
  (pureOrder() and the bulk order() share the exact same coefficients).
  Its order() (3 site-occupancy parameters, analytic Hessian, step
  truncation that keeps site fractions in [0, 1]) and pureOrder() are
  translated verbatim, with analytic Cp/dCpdT along the ordering
  equilibrium; gmixSpn() is ~0 at all five vertices. It matches MAGMA
  also for infeasible emulator compositions (negative total Al).

  rhomsghiorso.py (rhombohedral-oxide) has a large, bespoke (non-generic-
  polynomial) parameter surface and its own short-range-order cubic-
  spline function (`fSRO`, currently calibrated to a constant -- see its
  module docstring), but is architecturally SIMPLER than spinel.py:
  pureOrder() solves the same three DECOUPLED 1-D Landau parameters
  (s0<->ilmenite, s1<->geikielite, s2<->pyrophanite; hematite/corundum
  have none) shared by every endmember and every bulk composition alike
  (no per-vertex dispatch), and the bulk order()'s plain (undamped)
  Newton step converges reliably with only a relative-to-r[i] clip bound
  -- no line search or physical-gating fix was needed. One genuine
  subtlety: `hmixMsg`/`smixMsg` (both bulk and pure) report H/S via
  G-T*dG/dT / -dG/dT rather than the raw named H/S macros -- numerically
  identical to the raw macros in the CURRENT calibration but implemented
  via the general form for correctness beyond it. pMELTS omits the
  corundum endmember (`sol_struct_data.json`'s `pMeltsSolids` table has
  only 4 rhm-oxide rows, no Al2O3) -- this general 5-endmember model
  reduces correctly to that case at X_corundum=0, no separate code path
  needed.
- MELTS-version-aware liquid (liquid_modes.py, 2026-09-23): 'MELTS102' /
  'MELTS110' / 'MELTS120' select the liquid table, the W table and gibbs.c's
  SiO2/H2O/CO2/CaCO3 special cases; 1.1/1.2 include the CaCO3 speciation.
  compute_liquid_bulk_mode() is exact against the MAGMA C library for all
  three versions. compute_liquid_bulk() (liquid_eos.py) is the older
  version-agnostic path WITHOUT those special cases (dissolved H2O/CO2 get
  zero standard-state properties) and is kept only for regression tests.
- Fluids: compute_fluid_solution() (1.1/1.2 Duan H2O-CO2 "fluid",
  fluid_duan.py) and compute_water_phase() (1.0.2 pure "water", water.py).
- Garnet, leucite, biotite, hornblende, nepheline (2026-09-23; garnet.c,
  leucite.c, biotite.c, hornblende.c, nepheline.c incl. its K-Na ordering
  parameter): compute_solution(phase, T, P, X, solid_params). Also the
  endmember special cases they need (vc-/ca-nepheline, compute.py) and
  albite's displacive + Al-Si ordering (feldspar_disorder.py). Exact against
  the MAGMA C library. biotiteTaj.c is not used by any MELTS table; kalsilite
  and melilite are out of nGibbs's scope.
- Carbon solids (1.1/1.2): load_solids(table='meltsFluidSolids') adds
  calcite, aragonite, magnesite, siderite, dolomite, spurrite, tilleyite,
  graphite and diamond; they go through compute() like any pure phase.
"""

from .params    import load_solids, MELTSSolidParams, DEFAULT_TABLE
from .compute   import compute
from .solid_eos import eos_berman, eos_vinet
from .thermal   import berman_ref_state, saxena_ref_state, berman_cp
from .constants import (
    Rgas, Tr, Pr, Trl, BARS_PER_GPA,
    EOS_BERMAN, EOS_VINET, EOS_SAXENA,
    CP_BERMAN, CP_SAXENA,
)
from .liquid_params import load_liquid, MELTSLiquidParams
from .liquid_eos import kress_component, compute_liquid_components, compute_liquid_bulk
from .solution_model import newton_solve_ordering, darken_activities
from . import feldspar
from . import olivine
from . import clinopyroxene
from . import orthopyroxene
from . import spinel
from . import rhomsghiorso
from . import garnet, leucite, biotite, hornblende, nepheline
from . import molar_mass
from .molar_mass import molar_mass_from_formula, molar_masses, liquid_molar_masses
from .solid_solutions import (
    compute_feldspar_solution, compute_olivine_solution, compute_clinopyroxene_solution,
    compute_orthopyroxene_solution, compute_spinel_solution, compute_rhm_oxide_solution,
    compute_solution, SOLUTION_MODULES,
)
from .liquid_speciation import (
    oxides_to_liquid_components, liquid_components_to_oxides,
    liquid_component_si_al_coeffs, LIQUID_COMPONENT_LABELS,
)
from .liquid_modes import (
    MODES as MELTS_MODES, SOLID_TABLE as MELTS_SOLID_TABLE, CARBON_PHASES as MELTS_CARBON_PHASES,
    load_liquid_mode, liquid_species_properties, compute_liquid_bulk_mode,
)
from .fluid_duan import compute_fluid_solution
from .water import compute_water_phase

__all__ = [
    'load_solids', 'MELTSSolidParams', 'DEFAULT_TABLE',
    'compute',
    'eos_berman', 'eos_vinet',
    'berman_ref_state', 'saxena_ref_state', 'berman_cp',
    'Rgas', 'Tr', 'Pr', 'Trl', 'BARS_PER_GPA',
    'EOS_BERMAN', 'EOS_VINET', 'EOS_SAXENA',
    'CP_BERMAN', 'CP_SAXENA',
    'load_liquid', 'MELTSLiquidParams',
    'kress_component', 'compute_liquid_components', 'compute_liquid_bulk',
    'newton_solve_ordering', 'darken_activities',
    'feldspar', 'olivine', 'clinopyroxene', 'orthopyroxene', 'spinel', 'rhomsghiorso',
    'garnet', 'leucite', 'biotite', 'hornblende', 'nepheline', 'compute_solution', 'SOLUTION_MODULES',
    'molar_mass', 'molar_mass_from_formula', 'molar_masses', 'liquid_molar_masses',
    'compute_feldspar_solution', 'compute_olivine_solution', 'compute_clinopyroxene_solution',
    'compute_orthopyroxene_solution', 'compute_spinel_solution', 'compute_rhm_oxide_solution',
    'oxides_to_liquid_components', 'liquid_components_to_oxides',
    'liquid_component_si_al_coeffs', 'LIQUID_COMPONENT_LABELS',
    'MELTS_MODES', 'MELTS_SOLID_TABLE', 'MELTS_CARBON_PHASES',
    'load_liquid_mode', 'liquid_species_properties', 'compute_liquid_bulk_mode',
    'compute_fluid_solution', 'compute_water_phase',
]
