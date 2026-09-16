# Derivative importer

Two new files. `import_hefesto_subdirs.py` and `HeFESTo_functions.py` are untouched.

| file | contents |
|---|---|
| `src/builder/HeFESTo/HeFESTo_derivative_import.py` | `load_fort42`, `import_HeFESTo_derivatives`, `verify_chain_rule` |
| `scripts/import_hefesto_subdirs_derivs.py` | drop-in replacement for the existing script |

```bash
python scripts/import_hefesto_subdirs_derivs.py \
    --root data/HeFESToWorkspace \
    --dataname HeFESTo_all.csv \
    --phase-change-dataname HeFESTo_pc.csv \
    --verify 3
```

Writes `HeFESTo_all.csv` exactly as before, plus `HeFESTo_all_dndP.csv`,
`HeFESTo_all_dndT.csv` and `HeFESTo_all_deriv_manifest.csv`.

## Shadow tables

**Row parity is guaranteed.** The shadows emit one row for every row the main import
emits, so the three tables concatenate positionally with no join:

```
Row parity: main 1503  dndP 1503  dndT 1503  MATCH
```

That means the accept/reject logic mirrors the main import exactly — same ordering,
same required-file checks, same `nrows = min(len(table))` truncation — and a simulation
whose *derivatives* fail still contributes its `nrows`, filled with NaN, rather than
vanishing. Skipping would shift every subsequent row, the same class of bug as the
stray `fort.99` diagnostic lines. A simulation the main import rejects is rejected
here too, so it contributes no rows to either table.

The script checks parity at the end rather than trusting it.

### Columns

Trimmed to what a derivative table can actually carry: `P(GPa)`, `T(K)`, and the
per-component values. **185 columns → 76.** The fort.56 properties, the bulk
composition and the per-phase rho/V/mass columns are either constant down a simulation
or not composition derivatives at all, so carrying them would just repeat the main
table.

```
first columns: P(GPa)(System_main), T(K)(System_main),
               anorthite(plagioclase), albite(plagioclase), ...
```

Names are byte-identical to the main table's, so `main[shadow.columns]` lines the two
up directly.

### The NaN convention

A row whose derivatives could not be produced carries NaN in every component column,
but **`P(GPa)` and `T(K)` are still filled** — so the row keeps its place *and* stays
identifiable. The manifest's `deriv_source` says which route each simulation took:

```
sim_id  n_rows     N_el  n_species  n_placed deriv_source  fort99_offset
     1     501 23.63948        73        73      fort.42          False
     2     501 23.63948        73        73    recovered          False
     3     501 23.63948         0         0  unavailable          False
```

Mask on `isna()` of any component column, or join the manifest on `sim_id`, to drop
those rows from the derivative loss while keeping them in the main table.

## Verified end to end

Test workspace: the same HeFESTo run three times — once with its `fort.42`, once with
it deleted, once with it corrupted.

```
Derivative values for: 2/3 sims (67% coverage)
  from fort.42: 1   reconstructed: 1
NaN rows, derivatives failed: 1
Row parity: main 1503  dndP 1503  dndT 1503  MATCH

main (1503, 185)   dndP (1503, 76)      109 columns dropped
shadow columns are a subset of main, same names: True
Simulation3 block all-NaN in component cols: True,  P/T still populated: True
P column matches main row for row: True
normalise: max |shadow * N_el - raw fort.42| = 7.1e-15
```

## Three things to know before running it

**`--verify N` is not optional in spirit.** It runs

```
dn/dP|_isentrope = dn/dP|_T + dn/dT|_P * dT/dP
```

on N imported simulations per workspace. Every ratio should read 1.000. This is the
only cheap way to catch `fort.42`/`fort.99` misalignment, a wrong species mapping, or a
normalisation slip — all silent otherwise. The script exits non-zero if any check fails.

**Normalisation is by the element total, read per run.** `forward_phase_moles` divides
component moles by `reconBulkUnNormed.sum(dim=1)`, the element total, which is fixed
along a profile — so `dN/dP = 0` and the quotient rule collapses to a constant scale.
But that constant is per-run: Htz_transition is 23.63948, BENCHMARK is 24.00450. The
invariant is cations = 10, with oxygen floating with Fe3+ and Si/Mg. It is read from
each simulation's own control file and recorded in the manifest. `--no-normalise`
writes raw mol/GPa and mol/K if you'd rather scale downstream.

**`fort.42` must survive cleanup.** The delete list is still `fort.29` and `qout` only.
Do not add `fort.42` — it is the derivative dataset.

## Recovery is automatic

Most HeFESTo builds never wrote `fort.42`. Simulations lacking one have their
derivatives **reconstructed** from `fort.99` + `fort.56` — they are a deterministic
function of the assemblage and the P-T state, the same Hessian system `physub.f`
solves. This is on by default; `--no-recover` turns it off and reverts to reporting
those simulations as `missing fort.42`.

End-to-end check: the test workspace holds the same HeFESTo run twice, once with its
`fort.42` and once with it deleted, so the native and reconstructed blocks of the
shadow table must agree.

```
sim_id  n_rows     N_el  n_species  n_placed deriv_source  fort99_offset
     1     501 23.63948        73        73      fort.42          False
     2     501 23.63948        73        73    recovered          False

dn/dP: 28 moving components  rms diff 3.16e-06  signal rms 5.87e-02  rel 5.4e-05  corr 0.99871
dn/dT: 28 moving components  rms diff 1.71e-07  signal rms 1.76e-04  rel 9.7e-04  corr 0.99880
```

Relative agreement of 5e-5 on `dn/dP`. The manifest's `deriv_source` column records
which route each simulation took, so recovered rows can be excluded or down-weighted
if you ever want to.

Notes:

* `--param-dir` is needed when runs came from another machine. Resolution order is the
  flag, then the path recorded in the run's own control file, then the copy packaged
  with ngibbs. Control files record the path on the machine that *ran* them, which
  usually does not exist locally.
* `build_tables` parses the whole parameter set, so it is cached per parameter
  directory rather than rebuilt per simulation.
* `--write-recovered-fort42` caches each reconstruction to disk so a re-import reads it
  instead of recomputing. Off by default — it modifies the data directory.
* Keep `--recover-nsmall-rel 0`. Active-set pruning exists for emulator-predicted
  assemblages; HeFESTo's own minimiser does not need it, and pruning its output removes
  species it actually solved for.
* `--verify` now runs only on simulations with a genuine `fort.42`. The chain rule
  differences `fort.99` against `fort.42`, so on a reconstructed run it would compare
  the reconstruction with itself and pass vacuously.

## Partial coverage

Coverage is reported per workspace and in the summary, split into native and
reconstructed. With recovery on it should be 100%. If you disable it, train with
per-term loss normalisation by the count of *labelled* rows (not batch size) and
stratified batches, so the effective derivative weight does not drift with batch
composition — §0.3 of the plan.
