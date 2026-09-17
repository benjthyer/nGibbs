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
>>> liquid = compute_liquid_bulk(T, P, X, lparams)   # ideal-mixing bulk liquid

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
  (pureOrder() and the bulk order() share the exact same coefficients),
  and its internal-ordering Newton solve (3 site-occupancy parameters)
  needed a combined backtracking-line-search + physical-gating fix,
  beyond the plain solver every other phase uses, both to converge
  reliably for interior/mixed compositions and to correctly handle the
  genuine physical degeneracy at 3 of its 5 pure-endmember vertices (an
  entire cation type absent, e.g. Fe2+ at pure spinel MgAl2O4) -- see
  spinel.py's module docstring and the benchmark test file for the full
  derivation. gmixSpn() is ~0 at 4 of the 5 vertices (chromite,
  hercynite, magnetite, ulvospinel) but genuinely nonzero at the spinel
  vertex itself, by the same physical-degeneracy mechanism.

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
- NOT IMPLEMENTED: MELTS liquid's non-ideal (quasi-chemical / regular
  solution) excess Gibbs energy of mixing -- see liquid_eos.py's
  docstring. Liquid mixing here is ideal only (confirmed exact for
  volume; an approximation for G/H/S/activities).
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
from . import molar_mass
from .molar_mass import molar_mass_from_formula, molar_masses
from .solid_solutions import (
    compute_feldspar_solution, compute_olivine_solution, compute_clinopyroxene_solution,
    compute_orthopyroxene_solution, compute_spinel_solution, compute_rhm_oxide_solution,
)

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
    'molar_mass', 'molar_mass_from_formula', 'molar_masses',
    'compute_feldspar_solution', 'compute_olivine_solution', 'compute_clinopyroxene_solution',
    'compute_orthopyroxene_solution', 'compute_spinel_solution', 'compute_rhm_oxide_solution',
]
