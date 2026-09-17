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
- DONE: feldspar, olivine, clinopyroxene, orthopyroxene, and spinel
  solid-solution mixing models (the per-phase Margules/Landau-expansion
  excess-G functions, the internal-ordering Newton solve(s), and -- for
  clino-/orthopyroxene -- their additional essenite-only internal
  ordering and non-standard per-component-normalized Darken activity
  formula), validated against a C harness (
  melts_vec/tests/benchmark_solutions_test.py,
  melts_vec/tests/benchmark_clinopyroxene_test.py,
  melts_vec/tests/benchmark_orthopyroxene_test.py,
  melts_vec/tests/benchmark_spinel_test.py). clinopyroxene.c and
  orthopyroxene.c share a byte-identical Taylor-coefficient macro engine
  (confirmed via direct `diff`, both literal forks of one ancestor
  PYROXENE.C), so orthopyroxene.py reuses clinopyroxene.py's pure-
  endmember reference frame and essenite-ordering code directly and adds
  only its own MIX-branch coefficients -- see melts_vec/orthopyroxene.py's
  module docstring for the full architecture. spinel.c has no such
  PURE-vs-MIX split, and its internal-ordering solve needed a combined
  backtracking-line-search + physical-gating Newton fix (beyond the
  plain solver every other phase uses) to converge correctly, including
  at 3 of its 5 pure-endmember vertices where an entire cation type is
  physically absent -- see melts_vec/spinel.py's module docstring for
  the full derivation.
- DONE: the rhombohedral-oxide (geikielite, hematite, ilmenite,
  pyrophanite, corundum) solid-solution mixing model, validated against a
  C harness (melts_vec/tests/benchmark_rhomsghiorso_test.py). Its large,
  bespoke (non-generic-polynomial) parameter surface and its own short-
  range-order cubic-spline function (`fSRO`) had no analogue in any phase
  translated before it, but architecturally it needed no line-search/
  gating fix analogous to spinel.c's -- see melts_vec's own __init__.py
  and rhomsghiorso.py's module docstrings for the full rationale. pMELTS
  omits the corundum endmember; this general 5-endmember model reduces
  correctly to that case at X_corundum=0.
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
    compute_orthopyroxene_solution, compute_spinel_solution, compute_rhm_oxide_solution,
)

__all__ = [
    'load_solids', 'MELTSSolidParams', 'compute',
    'load_liquid', 'MELTSLiquidParams', 'compute_liquid_bulk', 'compute_liquid_components',
    'compute_feldspar_solution', 'compute_olivine_solution', 'compute_clinopyroxene_solution',
    'compute_orthopyroxene_solution', 'compute_spinel_solution', 'compute_rhm_oxide_solution',
]
