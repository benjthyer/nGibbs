# HeFESTo Notebook Setup and Bulk Property Generalization

Date: 2026-04-28

## Summary
Completed HeFESTo simulation data loading workflow for tutorial.ipynb with
generalized bulk property extraction utility. Fixed notebook import issues
and gracefully handled missing model files.

## Changes

### API.py (src/nMELTS/engine/API.py)
- Lines 790-820: Modified module-level HeFESToEmulatorCPU/GPU
  instantiation to conditionally lazy-load when model files exist
- Prevents import failures when TrainedModel directory unavailable
- Allows clean imports even in environments without pre-trained models

### HeFESTo_functions.py (src/builder/HeFESTo/HeFESTo_functions.py)
- Lines 16-27: Fixed file_utils import with fallback chain
  (relative -> absolute -> dummy implementation)
- Lines 1130-1280: Added new generalized function
  `extract_bulk_properties_from_simulation_dir(sim_dir, ...)`
  - Loads single HeFESTo simulation directory
  - Extracts thermodynamic properties via physub context
  - Computes density, bulk/shear moduli, velocities
  - Returns dict with component moles + bulk properties
  - Pattern extracted from test_hefesto_physub_benchmark.py

### tutorial.ipynb (notebooks/tutorial.ipynb)
- Cell 1: Fixed __file__ context detection for Jupyter
  - Supports both notebook and script execution modes
  - Handles missing model files with informative message
  - Proper sys.path configuration for nMELTS imports
- Cell 2: Load three PS simulations (basalt_PS, DMM_PS, htz_PS)
  - Each composition gets separate DataFrame with bulk properties
  - Displays T, P, VS, VP, component moles, physub properties
  - Provides summary statistics for each simulation

## Technical Details

### Bulk Property Extraction Workflow
1. Parse fort files from simulation directory (control, fort.56+)
2. Extract component moles from fort.99 (component abundances)
3. Build attributes dict for compute_physub_bulk_matrix()
4. Compute Hill-averaged bulk properties + seismic velocities
5. Return organized dict with T/P grids and properties

### Import Architecture
- HeFESTo_functions.py imports are now robust to different sys.path
  configurations (builder as root, src as root, etc.)
- Fallback saves_fixed_width_table prevents errors if file_utils
  unavailable at runtime

## Benefits
- Users can now load individual HeFESTo simulations without custom code
- Bulk property extraction is reusable library utility, not hardcoded
- Notebook works in environments with/without pre-trained models
- Cleaner separation between API initialization and data loading

## Testing Status
✓ Cell 1 imports execute without error
✓ Cell 2 loads basalt_PS, DMM_PS, htz_PS successfully
✓ Bulk properties computed for all three simulations
✓ DataFrames created with component + thermodynamic data
