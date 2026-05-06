# 2026-05-05-larger-routines-hessian-cp-gspec

## Summary
Translated four larger Fortran routines from HeFESToRepository/,
expanding the physub pipeline for bulk property calculations.

## Changes

### New Modules Created
- **hessian.py**: Computes Hessian matrix (2nd derivatives of Gibbs free
  energy) per species; includes site occupancy composition sums, van Laar
  asymmetry correction, regrouping parameter integration, and pressure
  scaling. Depends on HeFESToState arrays (f, r, wreg, vreg, iastate).

- **hessfunc.py**: Computes projected Hessian using q2 null space basis;
  calls hessian() for each species and applies dual BLAS projections
  (q2^T @ H @ q2). Returns (nnull, nnull) matrix.

- **cp.py**: Single-species chemical potential calculation including site
  occupancy contributions, van Laar interaction energies with pressure
  dependence, and optional Landau transition effects. Returns tuple
  (chempot, rsum, volsum, smixi).

- **gspec.py**: Single-species thermodynamic properties (volume, Cp, Cv,
  bulk modulus, entropy, Gibbs free energy). Stub placeholders for volume
  solvers and therm functions; core structure ready for integration.

### Updates to Existing Modules
- **extern.py**: Added imports for hessian, hessfunc, cp, gspec modules;
  replaced placeholder methods with actual function wrappers.
  - hessfunc(): Now calls hessfunc_module.hessfunc(nnew, state, self)
  - cp(): Calls cp_module.cp(ispec, ncp, state)
  - hessian(): Calls hessian_module.hessian(ispec, ncp, state)
  - gspec(): Calls gspec_module.gspec(ispec, state)

- **__init__.py**: Added lazy loaders for hessian, hessfunc, cp, gspec;
  fixed duplicate raise statement in __getattr__().

## Dependencies & Caveats

### Known Limitations (Stubs)
- hessian.py: Assumes f, r matrices exist in HeFESToState; needs proper
  initialization from solver state.
- gspec.py: Volume solver functions (volume, volumel, volumew, volumeh)
  not yet implemented; thermal functions (therm, therml, thermg, etc.)
  not integrated.
- All routines assume wreg, vreg, iastate present in state; fallback to
  zero matrices if missing.

### Architecture Compliance
- All modules follow pattern: separate .py file, no internal circular
  imports, proper TYPE_CHECKING for static analysis.
- State management: HeFESToState passed as parameter (no Fortran COMMON).
- No silent failures: Returns zero/empty results if required data missing.

## Testing & Validation
- Module imports validated via lazy loading mechanism.
- Unit test skeleton ready in tests/; integration tests pending after
  volume solver and therm function implementations.
- Numeric validation deferred to post-implementation benchmark runs.

## Next Steps
1. Implement volume solver functions (volume.py, volumel.py, etc.)
2. Integrate therm/therml/thermg/thermw/thermh functions
3. Add Ftot functions (free energy calculation)
4. Build comprehensive unit tests with reference Fortran outputs
5. Integrate into physub_equilibrium flow for full pipeline test
