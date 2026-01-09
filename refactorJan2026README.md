# nMELTS Refactoring Log - January 2026

This document tracks the refactoring of the nMELTS codebase to follow software engineering best practices.

## Overview
Refactoring the codebase to improve organization, maintainability, and follow best practices by:
- Separating concerns into logical modules
- Moving constants and configuration to dedicated modules
- Consolidating utility functions
- Creating clear module boundaries

## Changes Made

### 2026-01-XX: Initial Structure Creation
- Created new module structure in `src/nMELTS/`
- Created blank files for planned modules
- **Note**: Original files remain unchanged in Legacy directories

### 2026-01-XX: Renamed component_indices to MELTS_indices
- **Action**: Renamed `component_indices` to `MELTS_indices` throughout constants.py for better clarity
- **Changes**:
  - Updated all 20+ references to use `MELTS_indices` instead of `component_indices`
  - Updated `__init__.py` to export `MELTS_indices` instead of `component_indices`
- **Files Modified**:
  - `src/nMELTS/config/constants.py` - All references renamed
  - `src/nMELTS/config/__init__.py` - Updated imports and exports

### 2026-01-XX: Updated CSV Loading in constants.py
- **Action**: Updated CSV file paths to use relative paths from module location
- **Changes**:
  - CSV files now load from `config/projections/` and `config/old_transforms/` folders
  - Paths are resolved relative to the `constants.py` file location using `Path(__file__).parent`
  - CSV files are automatically loaded when the `constants` module is imported
  - Updated `__init__.py` to import constants and re-export commonly used constants
- **Files Modified**:
  - `src/nMELTS/config/constants.py` - Updated CSV loading paths
  - `src/nMELTS/config/__init__.py` - Added imports to ensure constants load on module import

### 2026-01-XX: EmulatorLibrary.py Refactoring
- **Source**: `Legacy/BackEnds/EmulatorLibrary.py`
- **Action**: Subdivided functions and variables into config and utils modules
- **Files Created**:
  - `src/nMELTS/config/constants.py` - Constants, mappings, and configuration data
    - Component indices, phase mappings, label indices
    - Oxide and element key lists
    - Transformation matrices (with CSV file loading)
    - Phase dictionaries and component mappings
  - `src/nMELTS/utils/string_utils.py` - String manipulation utilities
    - `pull_number()`, `pull_letter()`, `concat_all()`, `random_char()`
  - `src/nMELTS/utils/math_utils.py` - Mathematical utilities
    - `QFM_fO2()`, `QFM_fO2_torch()`, `Fe2O3_FeO_ratio()`
    - `identify_binaries()`, `blur_binary_boundaries()`, `grid_sample()`
    - `squash_to_range()`, `unsquash_from_range()`, `projected_nnls()`, `masked_column_assign()`
  - `src/nMELTS/utils/file_utils.py` - File operation utilities
    - `delete_files_with_keyword()`, `move_file()`, `move_files_with_extension()`
  - `src/nMELTS/config/settings.py` - Configuration settings
    - `external_base`, `internal_dir()`, `external_dir()`

## Module Organization

### Config Module (`src/nMELTS/config/`)
Contains constants, mappings, and configuration data:
- Component indices and phase mappings
- Oxide and element key lists
- Transformation matrices
- Label indices and dictionaries
- Configuration settings (paths, etc.)

### Utils Module (`src/nMELTS/utils/`)
Contains reusable utility functions:
- String manipulation (pull_number, pull_letter, etc.)
- Mathematical operations (QFM_fO2, Fe2O3_FeO_ratio, etc.)
- File operations (move_file, delete_files_with_keyword, etc.)
- Data processing utilities

## Migration Notes
- All original files remain in Legacy directories
- New modules are created alongside, not replacing original code
- Backward compatibility maintained during transition

## Notes on Remaining Code
The following classes from `Legacy/BackEnds/EmulatorLibrary.py` have not yet been moved:
- `Normalizer` class - Normalization utility (lines 811-843)
- `MELTS_Converter` class - Conversion functionality (lines 675-797)

These can be moved to appropriate modules in a future refactoring step. The `Normalizer` class could go in `utils/math_utils.py` or a new `utils/normalization.py`. The `MELTS_Converter` class might belong in a separate module like `engine/converter.py` or `data/converter.py`.
