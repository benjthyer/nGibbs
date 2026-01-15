---
name: Dynamic Dataset Indexing System
overview: Refactor the static indexing system in constants.py to use a dynamic DatasetIndexer class that generates all indices from dataset column headers, enabling flexible addition/removal of phases and components.
todos:
  - id: create_indexer_class
    content: Create DatasetIndexer class in src/nMELTS/config/indexer.py with header parsing and dynamic mapping generation
    status: completed
  - id: update_constants
    content: Remove static index definitions from constants.py, keep only chemistry transforms and add indexer factory function
    status: pending
    dependencies:
      - create_indexer_class
  - id: update_parser
    content: Update BigMetaTable in parser.py to create and use DatasetIndexer from CSV headers
    status: pending
    dependencies:
      - create_indexer_class
  - id: update_import_function
    content: Update import_MELTS_components in alphamelts_functions.py to use DatasetIndexer for column lookups
    status: pending
    dependencies:
      - create_indexer_class
  - id: update_emulator
    content: Update NN_MELTS in emulator.py to accept DatasetIndexer parameter and use it instead of static constants
    status: pending
    dependencies:
      - create_indexer_class
  - id: update_init_exports
    content: Update __init__.py files to export DatasetIndexer and update import statements
    status: pending
    dependencies:
      - create_indexer_class
      - update_constants
  - id: test_integration
    content: Test with existing datasets to ensure all mappings are generated correctly
    status: pending
    dependencies:
      - update_parser
      - update_import_function
      - update_emulator
---

# Dynamic Dataset Indexing System Refactor

## Overview

Replace static index definitions in `src/nMELTS/config/constants.py` with a dynamic `DatasetIndexer` class that parses dataset column headers and generates all indices and mappings on-the-fly.

## Architecture

### Core Components

1. **DatasetIndexer Class** (`src/nMELTS/config/indexer.py`)

   - Parses headers in format `component(phase)` 
   - Generates `MELTS_indices` dict dynamically
   - Builds all derived mappings (label_indices, mass_indices, phaseToCompMap, etc.)
   - Handles special cases (System_main, melts-liquid, excluded phases)

2. **Header Parser**

   - Regex pattern: `^(.+)\((.+)\)$` to extract component and phase
   - Handles edge cases (special liquid properties, system properties)
   - Validates header format

3. **Dynamic Mapping Generation**

   - All current static mappings become dynamic methods
   - Maintains same structure but generated from parsed headers

## Implementation Plan

### Phase 1: Create DatasetIndexer Class

**File: `src/nMELTS/config/indexer.py`** (new file)

```python
class DatasetIndexer:
    """Dynamic indexer that generates all mappings from dataset headers."""
    
    def __init__(self, headers: List[str]):
        """Initialize indexer from column headers."""
        self.headers = headers
        self.MELTS_indices = {}
        self.database_headers = headers.copy()
        # Parse and build all indices
        self._parse_headers()
        self._build_derived_mappings()
    
    def _parse_headers(self):
        """Parse headers into MELTS_indices structure."""
        # Extract component(phase) pairs
        # Build MELTS_indices dict
        
    def _build_derived_mappings(self):
        """Generate all derived mappings dynamically."""
        # mass_indices, label_indices, phaseToCompMap, etc.
```

**Key Methods:**

- `_parse_headers()` - Extract phase/component pairs from headers
- `_build_mass_indices()` - Find all mass columns
- `_build_label_indices()` - Generate ML-ready mappings
- `_build_phase_mappings()` - Create phaseToCompMap, variedToAllComp, etc.
- `_identify_phases()` - Extract unique phase list
- `_identify_components()` - Extract component lists per phase

### Phase 2: Update Constants Module

**File: `src/nMELTS/config/constants.py`**

- Remove static `MELTS_indices` definition
- Remove static `database_headers` definition  
- Remove static derived mappings (label_indices, mass_indices, etc.)
- Keep only:
  - Transform constants (compToOx, oxToEl, etc.) - these are chemistry transforms, not dataset-specific
  - Oxide/element key lists (WRkeys, Oxides, Elkeys) - chemistry definitions
  - Active oxide dictionaries - chemistry rules
- Add factory function: `create_indexer_from_headers(headers)` or `create_indexer_from_csv(csv_path)`

### Phase 3: Update Data Parser

**File: `src/nMELTS/data/parser.py`**

- `BigMetaTable.__init__()` should create a `DatasetIndexer` from `self.header`
- Store indexer as instance attribute: `self.indexer = DatasetIndexer(self.header)`
- Replace all `MELTS_indices`, `mass_indices`, etc. references with `self.indexer.MELTS_indices`, `self.indexer.mass_indices`
- Update `collect_indices()` to accept indexer instead of mass_indices

### Phase 4: Update MELTS Import Function

**File: `src/wslMELTS/engine/alphamelts_functions.py`**

- `import_MELTS_components()` should:
  - Load or create CSV with headers first
  - Create `DatasetIndexer` from headers
  - Use indexer for all column lookups instead of static `MELTS_indices`
  - Build `workbase` array size from indexer's max index

### Phase 5: Update Emulator

**File: `src/nMELTS/engine/emulator.py`**

- `NN_MELTS.__init__()` should accept a `DatasetIndexer` instance
- Store indexer: `self.indexer = indexer`
- Replace all static constant references with `self.indexer.*`
- Update methods that use phaseToCompMap, label_indices, etc.

### Phase 6: Update Scripts

**Files: `scripts/MELTedMORB.py`, `scripts/MELTedGEOROC.py`**

- When creating datasets, ensure headers follow `component(phase)` format
- After importing, create indexer from generated CSV headers
- Pass indexer to any functions that need it

## Header Format Specification

Headers must follow pattern: `component(phase)`

Examples:

- `mass (gm)(olivine)`
- `forsterite(olivine)`
- `Pressure(System_main)`
- `wt% SiO2(melts-liquid)`

Special handling:

- System properties: `viscosity(System_main)`, `H(System_main)`, etc.
- Liquid properties: `liq mass (gm)(melts-liquid)` or `wt% SiO2(melts-liquid)`
- Excluded phases: `alkali-feldspar`, `water` (if present, skip or handle specially)

## Migration Strategy

1. Create `DatasetIndexer` class with full functionality
2. Update `BigMetaTable` to use indexer (backward compatible during transition)
3. Update `import_MELTS_components` to generate headers and use indexer
4. Update `NN_MELTS` to require indexer parameter
5. Remove static definitions from constants.py
6. Update all call sites

## Testing Considerations

- Test with existing dataset headers
- Test with missing phases/components
- Test with extra phases/components not in original schema
- Verify all derived mappings are correct
- Test edge cases (special characters, parentheses in names)

## Files to Modify

1. **New:** `src/nMELTS/config/indexer.py` - DatasetIndexer class
2. **Modify:** `src/nMELTS/config/constants.py` - Remove static indices, add factory
3. **Modify:** `src/nMELTS/config/__init__.py` - Export DatasetIndexer
4. **Modify:** `src/nMELTS/data/parser.py` - Use DatasetIndexer
5. **Modify:** `src/wslMELTS/engine/alphamelts_functions.py` - Use DatasetIndexer
6. **Modify:** `src/nMELTS/engine/emulator.py` - Accept and use DatasetIndexer
7. **Modify:** `src/nMELTS/engine/NN.py` - Update if it directly uses constants

## Benefits

- Flexible: Add/remove phases without code changes
- Dataset-specific: Each dataset can have different schema
- Maintainable: Single source of truth (headers)
- Testable: Can test with synthetic headers
- Extensible: Easy to add new component types