# Why 8.5% of assemblages fail bulk reconstruction

**Your instinct was right: it is the magnetite overlap.** The mechanism is a silent
name mismatch in the component importer, and `magnetite(spinel)` has been an all-zero
column in every HeFESTo table ever imported.

Reproduced on `HeFESTo_demoset_derivs` (8715 rows, the table in `data/MELTStables/HeFESTo`).

---

## The chain

**1. Two naming tables disagree.**

```
HeFESTo_snames_long['smag']              == 'magnetite-spinel'
MELTS_indices['spinel']                  has the key 'magnetite'
```

`_resolve_component_name_from_abbr('smag')` returns `magnetite-spinel`; the phase resolves
correctly to `spinel`. Verified directly:

```
smag -> (magnetite-spinel, spinel)      mag -> (magnetite, ferropericlase)
```

The long name exists *because* `magnetite` is ambiguous — it belongs to both `spinel` and
`ferropericlase` — but the schema column is `magnetite(spinel)`, i.e. the ambiguous short
form. So the resolved name is not a key in that phase's map.

**2. `_safe_assign` swallowed the miss.**

```python
col_idx = phase_map.get(component, None)
if col_idx is None:
    return                     # <- no write, no warning, no record
```

`smag`'s moles were computed, resolved, and then written nowhere.

**3. The column is empty, and it should not be.** In HeFESTo's own `fort.99`
(`somesims/Simulation1`, a low-P run), on the spinel-bearing row:

| species | moles | share of spinel phase |
|---|---|---|
| `smag` (magnetite) | 2.456e-2 | **80.6%** |
| `hc` (hercynite) | 3.28e-3 | 10.8% |
| `picr` | 2.54e-3 | 8.3% |
| `sp` | 9.91e-5 | 0.3% |

Spinel at low pressure is *dominantly* magnetite. Meanwhile in the imported CSV,
`magnetite(spinel)` has **0 non-zero entries** while `magnetite(ferropericlase)` has 4318.
Not an off-by-one: the memmap is exactly column-aligned with the header, and the CSV column
itself is empty.

**4. So the reconstruction loses that Fe₃O₄.** Residual oxide moles unaccounted for on
spinel rows, after scaling both sides to a common basis:

```
FeO    +5.204e-04        Al2O3  0.0     MgO   0.0
Fe2O3  +5.204e-04        CaO   ~1e-09   SiO2 ~6e-08
```

Identical to four digits — the FeO·Fe₂O₃ signature. Fitting the residual to every
`compToOxLoad` row: **magnetite explains 100.0%**, amount 5.204e-4 mol.

**5. Which deletes exactly the spinel rows.**

```
P(fail | spinel present) = 100.0%          P(fail | no spinel) = 0.0%
non-spinel rows reconstruct to 1e-9        spinel rows: Fe2O3 60% short, total Fe 0.89% short
```

The demoset is 1.70% spinel-bearing, so it loses 1.70%. Your EarthAdiabats set spends more
time in the spinel field, hence 8.5%. The tolerance is 5000 ppm; the Fe deficit is ~8900 ppm.

**6. And it is self-perpetuating.** Because the column reads all-zero, the indexer excludes
it: `EXCLUDED_COMPONENTS_BY_PHASE == {'spinel': ['magnetite'], ...}`. So spinel loses ~80% of
its mass in the model's component list even before the filter runs.

**7. The derivative sidecars inherit it identically** — same resolver, same call pattern.
`magnetite(spinel)` in `_dndP.csv`: 0 non-zero. `magnetite(ferropericlase)`: 4318.

---

## The fix

Three files, all in the delivered set.

**`file_utils.py`** — new `reconcile_component_name(indexer, phase, component_name, abbr)`.
Maps a resolved name onto the key the phase's schema actually uses. It strips only the
suffix `-<phase>`, and only when it matches the phase being written, so `magnetite-spinel`
in `spinel` becomes `magnetite` while `high-pressure-magnetite` in `ca-ferrite` is left
alone. Returns `None` when nothing matches — callers must treat that as an error.

Verified:

```
smag -> (magnetite-spinel, spinel)         -> 'magnetite'      col 34
mag  -> (magnetite, ferropericlase)        -> 'magnetite'      col 143
```

**`HeFESTo_functions.py`** — the component loop reconciles before assigning, and **raises**
if a species with non-zero moles has no column. `_safe_assign` now records every
unresolved `(phase, component)` in `UNRESOLVED_ASSIGNMENTS` (readable via
`get_unresolved_assignments()`) instead of returning silently. Several callers legitimately
probe optional names (`moles`, `phase moles`), so the record is a tally, not a raise — the
raise lives where it can see that real data is being dropped.

**`HeFESTo_derivative_import.py`** — same reconciliation and same refusal, so the sidecars
stay consistent with the abundances.

### What this costs you

**The tables have to be re-imported.** The zero column is baked into every existing CSV, and
so is the indexer's exclusion of `magnetite(spinel)`. Re-running the import will change the
component count (spinel regains a component) and therefore the bundle's label width, so
existing checkpoints will not load against the new indexer.

Expected after re-import: bulk-mismatch deletions drop from 8.5% to well under 0.1%, and
spinel assemblages survive.

### Worth considering separately

The root ambiguity — `magnetite` appearing in two phases in `COMPONENTS_IN_PHASES_HEFESTO`
— is still there. Renaming the spinel one to `magnetite-spinel` to match
`HeFESTo_snames_long` would remove the trap at source rather than reconciling around it.
That is a schema change, so it is your call; the reconciler is correct either way and
becomes a no-op if you do it.

---

## Isentropic derivatives — implemented

`resampling_to_datasets(..., isentropic_derivatives='auto')` now also exports
`dndp_s_labels.npy` and `dnds_labels.npy` on the component axis, derived from the
isothermal sidecars and the table's own `T`, `cp`, `alpha`, `rho`:

```
dS/dT|_P = cp / T
dS/dP|_T = -alpha * V,     V = 1/rho          [alpha_col * 1e-2 / rho  ->  J/(g K GPa)]
dT/dP|_S = alpha * V * T / cp                  <- the adiabatic gradient

dn/dS|_P = (dn/dT|_P) * T / cp
dn/dP|_S =  dn/dP|_T + (dn/dT|_P) * dT/dP|_S
```

`'auto'` adds them when both isothermal sidecars and all four System_main columns exist;
`True` raises naming what is missing; `False` skips. Rows with non-positive `cp`, `T` or
`rho` come out NaN rather than as a large finite number — the trainer masks on `isfinite`,
and a silently huge derivative would dominate a scaled Huber.

Verified on the real demoset export:

```
[derivatives] exporting ['dndp', 'dndp_s', 'dnds', 'dndt'] on the component axis (68 columns each)
[derivatives] isentropic pair enabled: dn/dP|S and dn/dS|P derived via alpha*V*T/cp
[derivatives] median adiabatic gradient dT/dP|S = 6.89 K/GPa
```

and on the packaged bundle:

| array | median \|Aᵀdn\| | max |
|---|---|---|
| dn/dP\|T | 0 | 4.59e-04 |
| dn/dT\|P | 0 | 5.61e-07 |
| **dn/dP\|S** | **0** | **4.28e-04** |
| **dn/dS\|P** | **0** | **7.27e-04** |

The mass-balance tangent identity holds for the isentropic pair at the same precision as
the isothermal one, which it must: `Aᵀ dn/dP|_S = Aᵀ dn/dP|_T + (Aᵀ dn/dT|_P)·dT/dP = 0`.
Implied `dT/dP|_S` recovered from the exported arrays: median 7.3 K/GPa, 5–95th percentile
4.4–19.9 — right for a mantle adiabat spanning upper mantle to lower mantle.

**The one thing still unverified.** `cp` must be the *equilibrium* heat capacity including
the latent-heat term `T·Σ(dnᵢ/dT)·Sᵢ`. If it is the fixed-assemblage value, `dT/dP|_S` is
wrong precisely at a phase transition — the only place any of this matters. The exporter now
prints the median gradient and warns outside 2–60 K/GPa, which catches a gross error but not
a subtle one. The decisive test is comparing `alpha*V*T/cp` against `dT/dP` finite-differenced
along a smooth, finely spaced adiabat. Run that before training on `dndp_s`/`dnds`.
