# Derivative sidecars in BigMetaTable

`dn/dP` and `dn/dT` ride alongside `self.table` as parallel memmaps, following the
`blurredbinaries` pattern already in the class.

| file | role |
|---|---|
| `src/builder/processing/sidecar.py` | all the mechanics |
| `src/builder/processing/BigMetaTable.py` | seven insertions, no logic moved |
| `src/builder/processing/guardrails.py` | the `[1,1]` and MELTS guards |

## Insertion points

| location | what it does |
|---|---|
| `__init__` | `attach()` — loads `<base>_dndP.npy` / `_dndT.npy`, building them from the importer's CSVs on first use |
| `delete` | `apply_mask()` — same keep-mask, same chunked rewrite |
| `split` | `split_into()` — moved rows onto the new table, kept rows filtered here |
| `manual_split` | same |
| `resample_rare_phase` | `grow_with_repeats()` — duplicates the same rows |
| `save` | `save()` |
| `retrieve_component_moles` | allocates `self.dmolar['dndp'/'dndt']`, the molar-space twins |

Absent sidecars leave the attributes at `None` and every path keeps its previous
behaviour, so this is inert on a workspace without derivative tables.

## Verified

```
attach -> ['dndp', 'dndt']   dndp(20, 4) dndt(20, 4)
delete  -> rows 17  dndp 17  values preserved: True
split   -> moved 5 kept 12  moved ok: True  kept ok: True
grow    -> 12 -> 18 rows   duplicated block exact: True
save    -> ['final_dndP.npy', 'final_dndT.npy']
guard   -> misalignment raises
```

## Three things worth knowing

**The filename base is recorded at attach time.** `__init__` rewrites `self.filename`
to `<name>_working`, but the importer wrote the shadows under the original stem. The
base is stored as `_sidecar_base` rather than re-derived later, so a later rename
cannot quietly point the sidecars at the wrong files.

**`grow_with_repeats` checks row counts, not `self.table`.** `resample_rare_phase`
grows the main table *after* the sidecars, so mid-operation the main table is still at
its old height. An alignment assert against it fired spuriously in testing; the check
now compares against the known `old_rows` and `new_total`, which removes the ordering
hazard entirely rather than depending on call order.

**No multiplier is applied on duplication.** Abundance resampling is guarded to
`[1, 1]`, so a duplicated row's derivatives are the originals. If that guard is lifted,
the same per-row factor must be applied to the derivatives too — `n` and `dn/dP` are
each homogeneous of degree 1 in the bulk amount. Both `sidecar.py` and the guardrail
message say so at the point where it would matter.

## Still open

`retrieve_component_moles` allocates the molar twins and asserts their row count, but
the per-phase fill loop is not yet wired — it needs the same `component_inds` selection
the moles use, inside the `for phase in phases` body. That is the one remaining edit,
and it is confined to the HeFESTo branch (`self.Model == 'HeFESTo'`), since MELTS has
no dn/dP source.
