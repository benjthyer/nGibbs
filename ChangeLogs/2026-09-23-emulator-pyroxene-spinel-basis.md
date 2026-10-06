2026-09-23
- Cause of the MORB emulator-pathway H/V/Cp errors (~0.8% H, ~0.35% V):
  the melts_comparison bridge read the NN's pyroxene and spinel component
  moles as native MELTS endmember moles. Those phases are learned in the
  PxSp_Comp_TransformV2 basis (c = x_native @ T), so their compositions
  were badly wrong (MORB opx FeO 0.06 vs 16.7 wt%, spinel Al2O3 16.6 vs
  4.6) although phase proportions looked right. cpx alone cost -66 J/g.
- emulator.py: new NN_MELTS.native_component_moles(componentMoles) ->
  componentMoles @ inv(PxSpTransform) (numpy or torch). Reproduces the
  emulator's own phase_tables oxides for every phase.
- melts_comparison: ml_indexer_to_melts_vec_assemblage maps component
  moles to the native basis before building melts_vec compositions.
- melts_vec/spinel.py: order() and pureOrder() now translated verbatim
  (analytic D2GDS2, feasibility step truncation, 10*DBL_EPSILON stop,
  MAX_ITER 200, sOld returned, site fractions clipped as in the source).
  Cp and dCp/dT are analytic along the ordering equilibrium (cpmixSpn
  FIRST|SECOND); dCpdT_mix was 0 before. dV/dT, dV/dP of mixing are 0.
  hmix now includes the WV1/WV2 (p-1) volume term, as hmixSpn does
  (it was missing: G, H off by ~9 J on BishopTuff spinel, S exact).
- Removed from spinel.py: the FD-Jacobian/line-search/gated solver, FD Cp,
  _newton_1d and pure_endmember_ghsv. That solver diverged on emulated
  spinel with negative total Al (BishopTuff: dG up to 6 kJ, Cp x1e3)
  and gave gmix 1.7 kJ at MgAl2O4 (now ~0 at all five vertices).
  Now matches MAGMA to ~1e-15 relative on random, near-vertex and
  negative-Al compositions (GT and emulated MORB/BishopTuff spinel).
- order() hitting MAX_ITER is reported with a RuntimeWarning (MAGMA uses
  the last iterate silently); a singular Hessian raises LinAlgError.
- compute_spinel_solution: n_iter argument removed (no longer used).
- tests: benchmark_spinel_test.py now checks the ordering solve and all
  properties against frozen MAGMA output (spinel block added to
  solid_solutions_oracle.json via make_solid_solutions_oracle.py; other
  phases' frozen data unchanged). verify_spinel.c is kept only for its
  fixed-(r,s) formula checks.
- Emulator pathways, mean rel. error before -> after (NoCr isothermal):
  MORB H 0.81 -> 0.011%, Cp 0.39 -> 0.093%, V 0.34 -> 0.062%;
  Lherzolite H 0.05 -> 0.015%, V 0.05 -> 0.021%. BishopTuff H 0.04 ->
  0.08%: now at the GT-assemblage floor (0.068%, alphaMELTS plagioclase
  H/S offset); the old value was cancellation against wrong compositions.
- Open (emulator, not EOS): BishopTuff emulated spinel has negative
  native hercynite in 200 of 232 rows (total Al down to -0.27; GT min
  +0.04). The EOS now treats it as MAGMA would; the NN output is
  unphysical there.
- Exactly pure MgAl2O4 (no Fe): order()'s Hessian has cond ~1e16, so Cp
  and dCp/dT there are rounding noise in MAGMA and here alike.
