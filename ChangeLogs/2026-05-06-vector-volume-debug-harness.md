2026-05-06
- Refactored VectorSLBBase.volume() to fix and diagnose convergence.
- CRITICAL: Fixed K_T evaluation at P=0 -> now uses actual pressure.
- Removed aggressive dP_dV clamping that suppressed Newton corrections.
- Added verbose mode with iteration-by-iteration diagnostics.
- Added checks for pathological K_T (negative) and huge step sizes.
- Simplified convergence criterion (absolute residual only).
