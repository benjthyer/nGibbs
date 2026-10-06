2026-09-23
- The iterative mass balance now re-applies the MELTS pyroxene/spinel
  legality polish after every step. Its correction + clamp perturbs
  components inside a phase and re-created negative CaO (opx) or FeO/Al2O3
  (spinel) in the transformed basis; polishing last makes the returned
  component moles legal.
- mass_balance.py: MassBalanceProjector.forward(..., polish=None) calls
  polish(n) after each clamp.
- emulator.py: NN_MELTS.polish_component_moles(cm) divides each opx and
  spinel block by its phase moles, runs polish_negative_px /
  polish_negative_sp, and rescales (phase moles unchanged). Passed to the
  projector in 'iterative' mode for MELTS models ('melts-liquid' present);
  'pinv' and 'none' are unchanged. A MELTS model without the polish
  methods raises TypeError.
- NN_continuous.py: ContinuousModel reuses MidLevelNetwork's polish
  methods (as it reuses save). Its forward() still does not polish; the
  deployed 1.2 models are ContinuousModel, so before this change their
  opx/spinel were never polished at all.
- NN.py polish_negative_sp: for rows with all three pools present, only
  the violated constraint is imposed (Al-only: spinel fixed, {chromite,
  hercynite} vs {magnetite, ulvospinel}; FeO-only: {magnetite,
  ulvospinel} fixed, {chromite, hercynite} vs spinel), as the degenerate
  branches already did. Rows violating both, or pushed past the other
  bound, still get the joint 3x3 solve. Imposing both equalities on an
  Al-only violation set FeO = 0: BishopTuff spinel went to FeO 4, MgO 20
  wt% (alphaMELTS: 34, 0.8), and the EOS spinel Hessian became singular.
- polish_negative_sp: violation tests use tol = 1e-6, so float32
  round-off no longer re-flags rows a previous polish put on a bound.
- polish_negative_spFe: ulvospinel's FeO coefficient is 2.25 (compToOx),
  not 1 + 2.25 (only reached on the singular-retry path).
- BishopTuff NoCr isothermal spinel after MB, mean wt% (TiO2, Al2O3,
  Fe2O3, FeO, MgO): raw 4.3, -2.2, 63.8, 33.0, 1.1 -> 3.9, 0.3, 61.6,
  33.1, 1.1 (alphaMELTS 3.9, 2.0, 59.6, 33.8, 0.8). Mass-balance residual
  median 4.0e-4 -> 4.2e-4. No negative opx/cpx/spinel oxide remains in any
  1.2 standard. MORB/Lherzolite unchanged (nothing to polish); bulk
  property errors unchanged except BishopTuff H (0.082 -> 0.077%).
- tests/unit_tests/test_mass_balance.py: polish hook, single-constraint
  polish, and a BishopTuff end-to-end legality check (skipped without the
  1.2 bundle and standards).
- Not updated: builder/training/validation/SpinelSingularityProfile.py
  still mirrors the older (always-joint) solve.
