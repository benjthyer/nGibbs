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

>>> from melts_vec import compute_clinopyroxene_solution
>>> cpx = compute_clinopyroxene_solution(T, P, X_cpx, sparams)

Scope
-----
- DONE, verified bit-for-bit against a standalone C harness built from
  MAGMA source formulas (see melts_vec/tests/): the solid-endmember EOS
  (Berman polynomial + Vinet finite-strain volume, Berman/Saxena heat
  capacity), the liquid Kress volume EOS + ideal mixing, and the
  feldspar, olivine, and clinopyroxene solid-solution mixing models.
- NOT YET IMPLEMENTED (deferred -- see MELTS_EOS_Vectorization_Plan.md in
  the nGibbs project for the full writeup and rationale): orthopyroxene,
  spinel, and rhombohedral-oxide solid-solution mixing models. These have
  a large parameter surface and correspondingly higher transcription-
  error risk than feldspar/olivine; they were explicitly scoped out of
  this pass rather than translated without the same level of C-harness
  verification the other phases received. Bulk assemblage properties for
  these phases can still be computed at the pure-endmember level via
  `compute()` with `solid_solutions.py`'s combination rule applied
  manually (ideal mixing only, i.e. omitting their W(i,j) excess terms)
  as a stop-gap.
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
from .solid_solutions import (
    compute_feldspar_solution, compute_olivine_solution, compute_clinopyroxene_solution,
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
    'feldspar', 'olivine', 'clinopyroxene',
    'compute_feldspar_solution', 'compute_olivine_solution', 'compute_clinopyroxene_solution',
]
