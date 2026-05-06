## 2026-05-05 - Root finding and solid Ftotsub integration

- Added `cage.py`, `zeroin.py`, and `nlmin_V.py` translations.
- Replaced solid `volume.py` fallback with bracketing + minimization.
- Added solid `Ftotsub.py` and wired `gspec.py` to use it for `htl = 0`.
- Kept liquid, gas, water, and hydrogen branches on the existing fallback
  path per the anhydrous/subsolidus scope.
- Added smoke tests for `Ftotsub` and retained the prior helper coverage.
- Validation: helper + larger-routine tests pass (`28` total).