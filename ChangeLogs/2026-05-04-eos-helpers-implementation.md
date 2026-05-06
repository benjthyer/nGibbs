## 2026-05-04 - Implement small EOS_arithmetic helpers

### Summary
Translated five Fortran helper modules to Python and integrated them into
the EOS_arithmetic engine to unblock higher-level routine translations.

### Changes
- **bserch.py**: Binary search helper (exact Fortran semantics).
- **thetacal.py**: Theta calibration via Heat table inversion.
- **qr19.py**: Seismic Q-factor (attenuation) model with temperature/age
  corrections and adiabat reference.
- **landau.py**: Landau phase transition properties (standard reference).
- **landauqr.py**: Landau transition (Q-referenced ordered state).
- **extern.py**: Wired helpers and fixed tlindeman typo (fnaggF -> fnagg).
- **test_eos_arithmetic_helpers.py**: Unit tests for all five helpers
  (13 tests, all passing).
- **__init__.py**: Reorganized imports to avoid circular dependencies
  via lazy loading.

### Pre-existing fixes
- Commented unused import in param_state.py (hefesto_thermal_properties).
- Deferred entrop imports due to broken dependencies in current codebase.

### Testing
- All 13 unit tests pass (bserch, thetacal, qr19, landau, landauqr).
- bserch: binary search boundary cases.
- thetacal: Heat inversion edge cases and intermediate values.
- qr19: loads adqref.inc, computes Q with temp corrections.
- landau/landauqr: Landau formulas with and without transitions.

### Next Steps
1. Translate hessian.f and hessfunc.f.
2. Translate cp.f (largest remaining routine).
3. Translate gspec.f to call new helpers and return structured results.
4. Revisit svdsub batched SVD semantics.
5. Compare numeric outputs to Fortran benchmarks (NewBenchmark).
