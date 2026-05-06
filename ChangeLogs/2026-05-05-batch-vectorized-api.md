# Batch and Vectorized API Support

## Date
2026-05-05

## Summary
Updated `calculate_bulk_properties` API to accept batch inputs and process multiple assemblages efficiently. Added parameter singleton integration and fixed import paths.

## Changes

### Core API Changes
- **`calculate_bulk_properties()`**: Now detects batch mode automatically
  - Single mode: `nnew` (nspec,), `P` scalar, `T` scalar -> dict
  - Batch mode: `nnew` (N, nspec), `P` (N,), `T` (N,) -> dict with (N, nprops) output
  - Inputs are numpy-ified and array-like is supported
  
- **`_compute_bulk_properties_batch()`**: New helper for vectorized processing
  - Processes N assemblages row-by-row (loop over rows; individual kernels remain scalar)
  - Returns dict with keys: 'output' (N, nprops), 'property_names', 'timing' (seconds)
  - Current output properties: density, Cp, Cv, alpha, K_S, gamma
  
- **`initialize_hefesto_state()`**: Wired to parameter store
  - Calls `get_parameter_store()` to load HeFESTo parameter files (cached singleton)
  - Populates `state.apar` (nspecp × nparp) with parameter values
  - Falls back safely if species names don't match parameter store

### Parameter Loader
- **`param_loader.py`**: New module with cached singleton
  - `ParameterStore`: Container for parsed HeFESTo parameter records
  - `get_parameter_store()`: LRU-cached factory (maxsize=1)
  - `build_apar()`: Constructs (ns × npar) matrix with optional padding to nspecp

### Fixes
- Fixed ml_indexer import: Changed `from nMELTS.utils...` to relative import
- Parameter store is compatible with both HeFESTo files and fallback modes

## Vectorization Notes
Current implementation processes batch rows in a loop, with each row passing through scalar-path kernels. For ~60K rows, this may be memory-efficient but not compute-efficient. Next phases will:
1. Profile individual kernel times (volume, Ftotsub, therm, gspec)
2. Vectorize hot-path kernels (e.g., therm computation can use numpy broadcasting)
3. Consider GPU acceleration for matrix-heavy operations

## Testing
- All existing unit tests pass (27/27)
- Batch API imports successfully
- Parameter store singleton caches and fallbacks work
