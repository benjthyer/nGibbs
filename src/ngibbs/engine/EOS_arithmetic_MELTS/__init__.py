"""
EOS_arithmetic_MELTS -- Vectorized Python implementation of the MELTS EOS,
mirroring the role `EOS_arithmetic/` plays for HeFESTo.

Current status (see MELTS_EOS_Vectorization_Plan.md in the nGibbs project
for the full roadmap):

- DONE: solid-endmember EOS (`melts_vec`) -- Berman polynomial + Vinet
  finite-strain volume models, Berman/Saxena heat capacity, translated
  directly from MAGMA's sources/gibbs.c and validated against a standalone
  C harness built from the same source formulas (see
  melts_vec/tests/benchmark_test.py).
- DONE: liquid volume/mixing model (Kress polynomial volume EOS + ideal
  mixing, from sources/gibbs.c's generic liquid branch -- the
  Ghiorso-Kress compositional model is confirmed dead code for the
  default rhyolite-MELTS mode this package targets, see liquid_eos.py),
  validated against a C harness (melts_vec/tests/benchmark_liquid_test.py).
  Liquid's non-ideal excess Gibbs energy of mixing is NOT implemented
  (flagged in liquid_eos.py's docstring).
- DONE: feldspar, olivine, and clinopyroxene solid-solution mixing models
  (the per-phase Margules/Landau-expansion excess-G functions, the
  internal-ordering Newton solve(s), and -- for clinopyroxene -- its
  additional essenite-only internal ordering and non-standard per-
  component-normalized Darken activity formula), validated against a C
  harness (melts_vec/tests/benchmark_solutions_test.py,
  melts_vec/tests/benchmark_clinopyroxene_test.py).
- DEFERRED: orthopyroxene, spinel, and rhombohedral-oxide solid-solution
  mixing models -- explicitly scoped out of this pass due to their large
  parameter surface and correspondingly higher transcription-error risk
  without a more automated extraction approach; see melts_vec's own
  __init__.py docstring and the project plan doc for the full rationale.
- DONE: a MELTSAPI.get_property_melts_vectorized_from_assemblage() wiring
  in engine/API.py, parallel to HeFESToAPI's existing method.

>>> from EOS_arithmetic_MELTS.melts_vec import load_solids, compute
>>> params = load_solids()
>>> result = compute(T, P, params)
"""

from .melts_vec import (
    load_solids, MELTSSolidParams, compute,
    load_liquid, MELTSLiquidParams, compute_liquid_bulk, compute_liquid_components,
    compute_feldspar_solution, compute_olivine_solution, compute_clinopyroxene_solution,
)

__all__ = [
    'load_solids', 'MELTSSolidParams', 'compute',
    'load_liquid', 'MELTSLiquidParams', 'compute_liquid_bulk', 'compute_liquid_components',
    'compute_feldspar_solution', 'compute_olivine_solution', 'compute_clinopyroxene_solution',
]
