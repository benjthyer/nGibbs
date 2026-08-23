# Derivative-supervised training — what's implemented

Files and where they go:

| file | destination |
|---|---|
| `check_derivatives.py` | `scripts/` |
| `NN.py`, `NN_continuous.py` | `src/ngibbs/engine/` |
| `sobolev.py`, `trainer.py`, `tuners.py`, `main.py` | `src/builder/training/` |
| `sidecar.py`, `BigMetaTable.py`, `MLexporter.py` | `src/builder/processing/` |
| `file_utils.py` | `src/ngibbs/utils/` |

One file is **specified but not written** — `loadTrainData.py`, which is not on my
filesystem. Its contract is at the end.

Verified in this session: the legacy tuning path is still bit-identical (9 trials, same
order, same losses); the units chain round-trips to 7e-17; and the derivative loss reduces
a student's error against a teacher by **99.6% (dn/dP)** and **98.7% (dn/dT)**.

---

## 1. `scripts/check_derivatives.py` — run this first

Standalone, no pipeline changes needed. On the runs available here:

```
$ python scripts/check_derivatives.py data/HeFESToWorkspace/Htz_transition --chain
   rows 501   species 73   tracked 62   untracked 11
   [0] species map injective on (phase, component): True
   [1] A^T dndp : max |resid| over tracked 7.00e-08   rows>tol 0
   [1] A^T dndt : max |resid| over tracked 7.00e-08   rows>tol 0
   [1b] tracked bulk at row 0 matches control file: True
        Fe_total  row0 0.580520  control 0.58052   drift 1.00e-07
        O         row0 13.639470  control 13.63947   drift 4.70e-07
   [2] derivative coverage: 100.0% of rows
   [3] chain rule median ratio: 0.9999 over 2 segment-species
   -> ok
```

Three statuses, and the distinction between the last two is the point:

- **ok** — clean.
- **out-of-domain** — internally consistent, but the run spends time in phases the trained
  model does not carry. `somesims/Simulation1` reports 37.5% of rows with an untracked
  phase present, peaking at 7.78% of cations (quartz, coesite, kyanite, NAL phases). Those
  rows' tracked-subset derivatives *cannot* satisfy the tangent identity, so they fight the
  projection. Usable with `row_weights`, not a fault.
- **CHECK** — an unexplained residual, a non-injective map, or a bulk deficit with nothing
  untracked present. This is the one that means something is wrong.

The rank-1 report only runs on *unexplained* rows. That matters: `alpha-iron` shares its
element signature (`Fe=+3, Fe3=-2`) with `gamma-iron`, so leakage would otherwise read as
"100% along gamma-iron — looks like a mapping error" when nothing is wrong.

Step 0 asserts the species map is injective on `(phase, component)`. That single line is
what would have caught the `magnetite` collision that produced two wrong drafts of the
plan; `sidecar.component_columns` now enforces the same key, so the collision is
structurally prevented rather than remembered.

## 2. `ContinuousModel` — three new methods

- **`network_component_moles(x)`** — the body up to but not including the projector. Split
  out purely so a JVP can target it: 29 ms against 12,807 ms through the full `forward`.
  `forward` calls it, so there is still one definition of the body and its output is
  unchanged (verified identical after a save/load round trip).
- **`tangent_project(dn, n)`** — the analytic null-space projection. Measured on an
  untrained net: `||A^T dn||` 3.59e-2 → 1.65e-8.
- **`derivative_outputs(x, feature_indices, project=True)`** — one JVP per direction,
  returning the primal for free from the first.

## 3. `builder/training/sobolev.py`

`train_Upper_Sobolev` imports `_upper_forward`, `_upper_loss`,
`_resolve_heads_to_freeze`, `_regularization_spec`, `_make_train_loader` and the
checkpointing helpers from `trainer.py` rather than copying them. The new code is the two
JVPs, the projection, and a per-component-scaled Huber.

Two guards, both tested:

- a 4-tuple bundle raises by name, saying which arrays are missing and how to fix it;
- a model without `derivative_outputs` raises, explaining that `NN.py`'s `PhaseHead`
  writes into a softmax output in place and cannot be differentiated twice.

It warns on `batchnorm` (forward-mode AD makes the tangent batch-coupled, which the physics
is not) and notes that dropout draws a fresh mask per direction.

Noise injection goes into the value path only. Perturbing the input of a derivative target
asks the network to reproduce `dn/dP` at a pressure it was not given.

## 4. Export path

- **`BigMetaTable.retrieve_component_moles`** — fills `self.dmolar[attr]` alongside
  `self.molar` in the HeFESTo branch, using `sidecar.component_columns` keyed on
  `(phase, component)`. Unwritten slots are **NaN, not zero**: a component this chunk's
  phases never touch is unknown, and zero would read downstream as a measured "does not
  change".
- **`sidecar.py`** — persists the column header next to the `.npy` as
  `<stem>_dndP_header.json` and reads it back on attach. Without it a rebuilt memmap is an
  anonymous float block. Adds `component_columns`.
- **`MLexporter`** — allocates `dndp_labels.npy` / `dndt_labels.npy` on the **component**
  axis (not contracted by `phaseToCompMap.T`), divides by the same `InTot_chunk` as the
  moles, and writes `derivative_stats.json` with per-component scales and coverage. Both
  arrays are registered in `_ROW_ALIGNED_ARRAYS` — a missing entry there would leave one
  array unpermuted by `shuffle_bundle_rows` while everything else moved, silently pairing
  each row's derivative with a different row's composition.

  Allocation is gated on the **sidecars**, not on `self.dmolar`, because
  `retrieve_component_moles()` runs later inside the resample loop.
- **`file_utils`** — `MLDataBundle` grows `dndp_labels`, `dndt_labels`,
  `derivative_stats` and a `has_derivatives` property. The two arrays join
  `_OPTIONAL_ARRAYS`, so an old bundle loads unchanged instead of raising; the loud failure
  belongs in the trainer, where it can name the config key that asked for them.

## 5. Config

```yaml
derivatives:
  enabled: true          # bundle MUST carry dndp_labels/dndt_labels
  dndp_weight: 1.0
  dndt_weight: 1.0
  project_tangent: true
  subsample: 1.0         # fraction of each batch that gets the JVP
  huber_delta: 3.0       # in units of the per-component robust scale
```

`main._derivative_settings` reads it and returns `(train_fn, kwargs)`. Absent block means
`enabled: false`, so every existing recipe is untouched. `enabled: true` against a
derivative-free dataset **raises** — no silent fallback.

`tune_Upper_MELTS` gained `train_fn` / `train_fn_kwargs`, defaulting to
`train_Upper_MELTS`. The search logic is unchanged; a trial is compared on whatever scalar
the trainer returns. One caveat is in the docstring: losses are only comparable *within* a
sweep, so do not seed `best_loss` from a value-only episode into a Sobolev sweep.

Suggested first run: `project_tangent: true`, `subsample: 1.0`, `mole_regularization:
layernorm` or `none`, dropout 0. Budget ~3x the plain step time.

---

## `loadTrainData.py` — the contract I could not write

1. Load `dndp_labels` and `dndt_labels` from the bundle alongside the existing four
   (`load_ml_bundle` already returns them, and `bundle.has_derivatives` reports presence).
2. `TensorDatasetFour` becomes arity-agnostic — the name encodes the arity, so either
   rename it or have it accept `*tensors`. `trainer.py` and `sobolev.py` both index
   (`batch[:4]`, `batch[4:6]`) rather than destructuring, so a 6-tuple is safe for both.
3. `ChunkedMemmapTrainLoader`'s memmap set widens by two arrays, yielding 6-tuples.
4. Expose two attributes on whatever the loaders return:
   - `has_derivatives` (bool) — `main._derivative_settings` prefers it and falls back to
     probing `len(train_set[0]) >= 6`, which forces a row read.
   - `derivative_scale` (optional) — `derivative_stats.json`'s `scale[attr]`. Without it
     the trainer estimates from the first test batches, which is adequate but makes runs
     less comparable to each other.
5. Fit the feature `Normalizer` as now. `sobolev._feature_ranges` **raises** if it is
   absent rather than defaulting to 1.0 — a missing factor of ~140 on P would look like a
   badly weighted loss rather than a wrong one.

Derivative arrays are stored in **physical units** (mol/GPa, mol/K, per element mole).
The trainer applies the `ranger` factor itself. Do not pre-scale them in the loader.

## Open questions

1. **`alpha-iron`** — untracked, carries up to 7.3e-4 mol in `Simulation2`. Below your
   abundance threshold, presumably deliberately. Worth confirming it stays below at the
   pressures you care about.
2. **Bundle size** — two extra `(rows, 62)` float32 arrays roughly doubles it. float16
   would be fine after scale normalisation if disk is tight.
3. **The duplicate `magnetite` name** in `components_in_phases`. Renaming the spinel one to
   `magnetite-spinel`, matching `HeFESTo_snames_long`, would remove the trap at the source.
   Anything downstream depend on the current name?
