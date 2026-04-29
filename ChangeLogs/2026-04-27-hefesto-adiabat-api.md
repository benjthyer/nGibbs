# 2026-04-27 - HeFESTo Adiabat API Implementation

## Summary
Built comprehensive high-level API for HeFESTo adiabat modeling with ensemble
emulator management and thermodynamic workflows.

## Components Implemented

### 1. InputParser Class
- Detects oxide vs element composition space from headers
- Auto-recalculates Fe/Fe3+ when O present in elemental composition
- Wraps emulator's reorder_input_table for column normalization
- Returns parsed tables with consistent alignment

### 2. HeFESToAdiabatAPI Main Class
- Manages isothermal and isentropic emulator pair
- Loads temperature prediction FCNN via _load_temperature_model()
- Initializes HeFESTo physub context for thermodynamic ops
- Provides three main workflow methods

### 3. get_T() Lightweight Frontend
- Takes features through isentropic emulator
- Builds temperature model inputs from component moles
- Applies temperature normalizers (Normalizer class)
- Returns Kelvin temperatures

### 4. get_isentrope() Adiabat Computation
Four-step workflow:
- [1/4] Isothermal model evaluates entropy constraint
- [2/4] Creates (T,P) design grid with arbitrary spacing
- [3/4] Staged isentropic evaluation (_STAGE helper pattern)
- [4/4] Temperature FCNN prediction + reorganization

Features:
- Handles single or multiple input feature sets
- Optional potential_temperatures with validation
- Batch processing for memory efficiency
- Outputs (T,P) shaped adiabats

### 5. Design Matrix Functions
- create_isentrope_design_matrix(): Arbitrary (T,P) grid generation
- _create_iso_design_matrix(): Method variant for API workflows
- Supports tile/repeat patterns for arbitrary spacing

### 6. Batch Processing Support
- _staged_forward(): Wrapper around _STAGE pattern
- Handles both single tensors and list/tuple outputs
- Concatenates results across batches

### 7. parse_input() Convenience Method
- Orchestrates composition parsing + reordering
- Auto-detects oxide/element space
- Returns properly aligned torch.Tensor on device

## Key Design Decisions

- Never create utility functions outside src/nMELTS (follows architecture)
- Fe3+ column initialized to zero; actual speciation in emulator forward
- Temperature model inputs default to component moles (flexible for future)
- Normalizers instantiated as Normalizer objects for consistency
- Batch size defaults to 2^16 for GPU memory management
- Returns torch.Tensor for downstream operations (not numpy)

## Dependencies
- NN.py: _load_temperature_model()
- emulator.py: NN_MELTS class
- math_utils.py: Normalizer class
- EOS_arithmetic: get_hefesto_physub_context()

## Error Handling
- Comprehensive input validation (shapes, lengths, types)
- Detailed error messages for debugging
- Path existence checks for model files
- Device management (CPU/CUDA)

## Future Extensions
- Entropy model outputs when available
- Composition space conversion utilities
- Uncertainty quantification from ensembles
- Plotting/visualization workflows
