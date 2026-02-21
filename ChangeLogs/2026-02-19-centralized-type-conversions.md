# Centralized Type Conversion System for Config Serialization

## Motivation
Type conversion logic was scattered across multiple files with manual loops.
Default_dtype parameter adds JSON serialization safety by converting unmatched
keys to strings, preventing serialization errors during model saving.

## Changes

### `src/nMELTS/config/constants.py`
- Added TYPE_CONVERSION_MAP: centralized mapping of parameter names to
  target dtypes (float for floatTypes, int for intTypes)
- Consolidates previously scattered type definitions

### `src/nMELTS/utils/string_utils.py`
- Updated apply_type_conversions() with default_dtype parameter
- When specified, converts non-matching keys to default type for safety
- Recursively applies conversions through nested dicts while preserving
  structure
- Enables JSON-safe serialization

### `src/nMELTS/engine/NN.py`
- Imported apply_type_conversions and TYPE_CONVERSION_MAP
- Updated save() method to apply type conversions before JSON dump
- Uses default_dtype=str to convert unspecified values for JSON safety

### `src/builder/training/main.py`
- Replaced manual type conversion loops with apply_type_conversions
- Removed floatTypes and intTypes local definitions
- Now uses centralized TYPE_CONVERSION_MAP from constants
- Applied at three points: global config, episode_cfg, and tune_params
- Simpler, more maintainable code with fewer lines

## Benefits
- Single source of truth for type conversion rules
- Reduces code duplication (3 loop pairs -> 1 function call)
- JSON serialization is safe by default
- Easier to add new parameters: just update TYPE_CONVERSION_MAP
- Consistent behavior across training and model saving
