# Comprehensive Bulk Property Output from physub

## Date
2026-05-05

## Summary
Updated `physub()` to save all computed bulk properties to the `HeFESToState` object and return them in a comprehensive result dictionary. This ensures physical properties are accessible downstream and directly available for batch processing.

## Changes

### physub.py
- **State saving**: All computed properties now persisted to state object fields:
  - `state.vol` ← volagg (aggregate volume)
  - `state.Cap` ← cpagg (Cp)
  - `state.Cv` ← cvagg (Cv)
  - `state.alp` ← alpagg (thermal expansivity)
  - `state.gamma` ← gruagg (Gruneisen parameter)
  - `state.K` ← baggh (Hill bulk modulus)
  - `state.Ks` ← btaggh (adiabatic bulk modulus)
  - `state.Gsh` ← gaggh (Hill shear modulus)
  - `state.deltas` ← delagg (Debye parameter)
  - `state.ent` ← entagg (entropy)
  - `state.Ftot` ← freeagg (free energy)
  - And 10+ more elastic, thermal, and modal properties

- **Return dictionary**: Expanded from 3 keys to 25+ keys including:
  - Elastic moduli: K_Hill, K_Voigt, K_Reuss, G_Hill, G_Voigt, G_Reuss
  - Velocities: Vb_Hill, Vs_Hill, Vp_Hill (bulk, shear, P-wave)
  - Reuss/Voigt alternatives for all velocities
  - Thermodynamic: Cp, Cv, alpha, gamma, entropy, enthalpy
  - QfactorS: Q_shear, Q_pressure
  - Anelastic velocities: Vs_anelastic, Vp_anelastic
  - Debye: theta_Debye, gamma_Debye

### api.py (`_compute_bulk_properties_batch`)
- **Property extraction**: Now extracts 13 core properties (expanded from 6):
  - Density, Cp, Cv, alpha, K_S, gamma (original 6)
  - K_Hill, G_Hill, Vb_Hill, Vs_Hill, Vp_Hill (new moduli/velocities)
  - entropy, enthalpy (new thermodynamic)

- **Fallback logic**: Uses result dict keys with state fallbacks
  - ```python
    output[i, j] = result.get('K_Hill', state.K)
    ```
  - Ensures robustness if any property fails to compute

- **Output shape**: (N, 13) instead of (N, 6)

## Behavior Changes
- **Single-mode (`calculate_bulk_properties` scalar calls)**: Return dict now contains 25+ keys instead of 3
- **Batch-mode**: Output array has 13 columns (property_names expanded)
- **API.py line 603**: Calling code can now access full bulk property matrix directly

```python
output = calculate_bulk_properties(nnew=componentMoles, P=PT[:,0], T=PT[:,1], ...)
# output['output'] is now (N, 13) with full elastic + thermal props
# output['property_names'] lists: density, Cp, Cv, alpha, K_S, gamma, ...
```

## Benchmark Impact
- **Memory**: Small increase (~2× array size for 13 vs 6 properties)
- **Speed**: No change (properties already computed, now just returned/saved)
- **Data richness**: 60K rows now return ~780K values instead of ~360K

## Testing
- Imports: PASS
- Batch API: PASS (extracts 13 properties)
- State persistence: Properties now accessible via state object fields

## Next Steps
1. Run benchmark with updated 13-property output
2. Validate property ranges and physical reasonableness
3. Consider adding optional property selectors (return subset if memory/speed critical)
4. Add unit tests validating property extraction and state consistency
