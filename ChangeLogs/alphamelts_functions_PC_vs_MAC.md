# alphamelts_functions.py — PC vs MAC Comparison

Files:
- `src/builder/alphamelts/engine/alphamelts_functions.py` (PC/Linux)
- `src/builder/alphameltsMAC/engineMAC/alphamelts_functions.py` (MAC)

---

## 1. Binary path

The two files point to different alphaMELTS app directories and use different initializer filenames (note the underscore vs hyphen):

| | PC | MAC |
|---|---|---|
| App directory | `alphamelts-app-2.3.1-linux` (versioned) | `alphamelts-app` (unversioned) |
| Initializer | `run_alphamelts.command` (underscore) | `run-alphamelts.command` (hyphen) |
| Variable name | `alphameltsLocation` (full path to binary) | `alphaMELTSLocation` (directory only; binary appended at call time) |

PC also exposes `settingsLocation` as a module-level variable pointing to the batch settings file. MAC hardcodes the settings path from `__file__` inside `forward_ensemble`.

---

## 2. `forward_ensemble` — phase suppression strategy

This is the most significant functional difference. The two files implement phase suppression differently and offer different APIs.

### PC — suppress via MELTS input file only
Writes `Suppress:` lines directly into the `.melts` input string using `suppressAllBut`. No batch-file editing.

```python
if only_phases:
    if 'p' in batchname[i].lower():
        only_phases.append('sillimanite')   # ← mutates the input list (bug)
    MELTSStr = suppressAllBut(MELTSStr, only_phases)
```

### MAC — two-strategy suppression, controlled by `suppress_with_batch`
Has a `suppress_with_batch=True` default. When true, phases are suppressed by editing a copy of the batch file using `suppressAllButBatch` (which inserts phase/1/x blocks after the `8` marker). When false, falls back to the MELTS-input approach, same as PC.

```python
active_only_phases = list(only_phases)   # ← copy, not mutation
if 'p' in batchname[i].lower():
    if 'sillimanite' not in [p.lower() for p in active_only_phases]:
        active_only_phases.append('sillimanite')   # ← guarded add
if not suppress_with_batch:
    MELTSStr = suppressAllBut(MELTSStr, active_only_phases)
# ...
if active_only_phases and suppress_with_batch:
    batch_contents = suppressAllButBatch(batch_contents, active_only_phases)
```

MAC also imports `systemNames` (the complete alphaMELTS phase registry) to enumerate which phases to suppress in the batch. PC does not import or use `systemNames`.

---

## 3. `only_phases` mutation bug (PC only)

PC's `forward_ensemble` appends `'sillimanite'` directly to the `only_phases` list passed in by the caller. Because the same list object is reused across the simulation loop, all subsequent simulations — including non-pMELTS ones — will have sillimanite in their suppression list after the first pMELTS simulation is encountered.

MAC avoids this by working on `active_only_phases = list(only_phases)` and also guards against duplicate adds.

---

## 4. `forward_ensemble` signature differences

| Parameter | PC | MAC |
|---|---|---|
| Engine path | `alphameltsLocation` (full path, keyword arg) | `initializer='run-alphamelts.command'` (filename only, joined to `alphaMELTSLocation` dir at runtime) |
| Settings file | `settingsLocation` (keyword arg, defaults to module-level var) | Hardcoded from `__file__` |
| `delta` default | `None` | `-3` |
| `suppress_with_batch` | Absent | `True` |

---

## 5. `suppressAllButBatch` function — MAC only

MAC adds a function that edits an alphaMELTS **batch file** to suppress phases, rather than editing the `.melts` input. It finds the `8` marker line in the batch file and inserts blocks of:

```
<phase>
1
x
```

for every phase in `systemNames` that is not in the keep-list. PC has no equivalent.

---

## 6. `import_MELTS_components` — sanidine/plagioclase fix (MAC only)

MAC handles an alphaMELTS 2.3.1 quirk in pMELTS where the plagioclase K-feldspar end-member is reported as `sanidine` in the output table but stored under `highsanidine` in the component dictionary:

```python
# MAC only:
if fillname == 'sanidine' and phasename == 'plagioclase':
    meltsobj[rowsfill, indexer.MELTS_indices[phasename][fillname]] = table[:, melt_dict['highsanidine']]
else:
    meltsobj[rowsfill, ...] = table[:, melt_dict[fillname]]
```

PC uses the generic path for all components and will silently fail (caught by the `except` block) on pMELTS plagioclase data.

---

## Summary table

| Difference | PC | MAC |
|---|---|---|
| App directory | versioned `alphamelts-app-2.3.1-linux` | unversioned `alphamelts-app` |
| Initializer filename | `run_alphamelts.command` (underscore) | `run-alphamelts.command` (hyphen) |
| Phase suppression method | MELTS input file only | Batch file (default) or MELTS input file |
| `suppressAllButBatch` function | Absent | Present |
| `systemNames` import | Absent | Present |
| `only_phases` mutation | Yes — input list mutated (bug) | No — local copy used |
| Sillimanite duplicate guard | No | Yes |
| `delta` default | `None` | `-3` |
| Settings file arg | Keyword param | Hardcoded from `__file__` |
| Sanidine/plagioclase pMELTS fix | Absent | Present |
