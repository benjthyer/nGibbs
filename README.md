# nGibbs: Neural Network Emulators for Free Energy Minimizing Silicate Phase Equilibria

nGibbs provides fast neural-network emulators of MELTS (rhyolite-MELTS 1.0.2 and 1.2.0) and HeFESTo for silicate multi-component systems. Given pressure, temperature (or entropy) and a bulk composition, an emulator predicts the equilibrium phase assemblage, phase abundances and phase compositions roughly 1000× faster than running the simulator. Bundled, vectorized equations of state then turn those assemblages into thermodynamic and seismic properties.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20766922.svg)](https://doi.org/10.5281/zenodo.20766922)

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Shipped models](#shipped-models)
- [Input format](#input-format)
- [Automatic model selection](#automatic-model-selection)
- [HeFESTo emulators](#hefesto-emulators)
- [MELTS emulators](#melts-emulators)
- [Equation-of-state property calculators](#equation-of-state-property-calculators)
- [Performance: GPU, batch sizes and the CPU fallback](#performance-gpu-batch-sizes-and-the-cpu-fallback)
- [Training range warnings](#training-range-warnings)
- [Validating the models](#validating-the-models)
- [Tutorials](#tutorials)
- [Package layout](#package-layout)
- [Roadmap](#roadmap)
- [Citing nGibbs](#citing-ngibbs)
- [License](#license)

---

## Installation

```bash
pip install nGibbs
```

This installs the inference package (`ngibbs`): trained model weights, the emulator APIs, the EOS property calculators and the self-tests. Requirements: Python ≥ 3.9 (developed and tested on 3.11), PyTorch, NumPy, pandas, tqdm and matplotlib. A CUDA GPU is optional; every model also runs on CPU.

> **PyTorch:** install the build that matches your hardware (CPU-only or a CUDA version) by following [pytorch.org](https://pytorch.org/get-started/locally/) *before* `pip install nGibbs`. Otherwise pip installs its default build.

To work from source, or to generate data and train models (see [DEVELOPMENT.md](DEVELOPMENT.md)):

```bash
git clone https://github.com/benjthyer/nGibbs.git
cd nGibbs
pip install -e .                    # inference package, editable
pip install -r requirements.txt     # plus the data-generation / training dependencies
```

**PyPI package vs. this repository.** The PyPI package contains only `src/ngibbs/`. The repository adds `src/builder/` (running alphaMELTS/HeFESTo at scale, data processing and PyTorch training), `scripts/`, `recipes/`, and `notebooks/`.

---

## Quick start

```python
import numpy as np
from ngibbs import MELTS120EmulatorCPU   # or ...GPU when CUDA is available

MORB = {'SiO2': 48.68, 'TiO2': 1.01, 'Al2O3': 17.64, 'FeO': 7.59, 'Fe2O3': 0.89,
        'MgO': 9.10, 'CaO': 12.45, 'Na2O': 2.65, 'K2O': 0.03, 'P2O5': 0.08, 'H2O': 0.20}

T_C     = np.linspace(1250, 1000, 100)                       # °C
table   = np.column_stack([np.full_like(T_C, 2000), T_C,     # 2 kbar isobaric cooling path
                           np.tile(list(MORB.values()), (len(T_C), 1))])
headers = ['P(bar)', 'T(C)'] + list(MORB)            # same as 'Pressure(System_main)', ...

out = MELTS120EmulatorCPU.ForwardMB(table, headers=headers, outputs=['ptt_out'])
out['ptt_out']['Conditions']        # P, T, mass, ... per row
out['ptt_out']['liquid1']           # liquid composition and mass per row, PetThermoTools-style
```

`import ngibbs` by itself loads no models. Importing the first model loads all of them, which takes a few seconds; after that, every import returns the same objects.

---

## Shipped models

Import model objects directly from `ngibbs` (e.g. `from ngibbs import MELTS120EmulatorCPU`); `ngibbs.engine.models` exposes the same objects. Each model has a `...CPU` object and a `...GPU` object; the `...GPU` object is `None` when CUDA is unavailable. The same objects are also available as dictionaries, `ngibbs.CPU_MODELS` and `ngibbs.GPU_MODELS`.

| Object | Simulator | Pathways (inputs) | Oxides |
|---|---|---|---|
| `HeFESToEmulatorCPU/GPU` | HeFESTo, Earth mantle | isothermal (P, T); isentropic (P, S) with a temperature model | SiO2, Al2O3, FeO, Fe2O3, MgO, CaO, Na2O, Cr2O3 |
| `HeFESToMarsEmulatorCPU/GPU` | HeFESTo, Martian mantle | isothermal (P, T); isentropic (P, S) with a temperature model | same as Earth |
| `MELTS102EmulatorCPU/GPU` | rhyolite-MELTS 1.0.2 | closed-system isothermal (P, T); fO2-buffered (P, T, ΔQFM); each in Cr and NoCr variants | SiO2, TiO2, Al2O3, FeO, Fe2O3, MgO, CaO, Na2O, K2O, P2O5, H2O, (Cr2O3) |
| `MELTS120EmulatorCPU/GPU` | rhyolite-MELTS 1.2.0 | closed-system isothermal (P, T), Cr and NoCr variants | as 1.0.2, plus CO2 |

A missing column for an oxide the model knows is treated as zero. The HeFESTo models cover about 0–141 GPa, and `api.training_pressure_bounds()` returns the exact range. MELTS training data were generated with alphaMELTS 2.3.1.

Each model predicts:

- phase presence (the stable assemblage)
- the mass of each phase (wt% of the system)
- phase compositions (oxide wt%), and component moles in the reduced component space
- a temperature, from the temperature network (isentropic HeFESTo models only)

Mass balance is enforced after the network runs (`ForwardMB`), so the phases always add back to the input bulk composition. `ForwardNN` takes the same arguments and returns the raw network output without that correction. It is faster but does not conserve mass.

---

## Input format

Inputs are a 2-D array plus a list of headers, a pandas DataFrame (its column names are the headers), or a `{header: column}` dict. The condition headers select the pathway.

Condition headers are written `<quantity>(<unit>)`, and the trailing `(System_main)` is optional. The unit must be the one the model was trained in. Nothing is converted, so a wrong or missing unit is an error rather than a silent rescale.

| Quantity | Accepted names | HeFESTo unit | MELTS unit |
|---|---|---|---|
| pressure | `P`, `Pressure` | `GPa` | `bar` (or `bars`) |
| temperature (isothermal) | `T`, `Temp`, `Temperature` | `K` | `C` (or `°C`, `degC`, `Celsius`) |
| specific entropy (isentropic) | `S`, `Entropy` | `J/g/K` | — |
| log fO2 relative to QFM (selects the buffered model) | `logfO2(QFM)`, `fO2(QFM)`, `logfO2-QFM`, `ΔQFM`, `dQFM` (FMQ also accepted) | — | QFM offset (MELTS 1.0.2 only) |
| composition | `'SiO2'`, `'MgO'`, … | oxide wt% | oxide wt% |

So `'T(K)'`, `'Temperature(K)'` and `'T(K)(System_main)'` all work for HeFESTo, and `'T(C)'`, `'Temperature(°C)'` and `'P(bar)'` work for MELTS. `'T(C)'` is an error for HeFESTo, and `'T(K)'` is an error for MELTS.

The full canonical names still work, including MELTS's unitless `'Pressure(System_main)'` and `'Temperature(System_main)'`. Other headers without a unit, such as `'Temperature'` or `'logfO2'`, are rejected.

- **Iron (HeFESTo):** if you give only `'FeO'`, all iron is treated as ferrous. If you give both `'FeO'` and `'Fe2O3'`, that speciation is used.
- **Elements:** HeFESTo models also accept element amounts (`'Si'`, `'Mg'`, …) in place of oxides. Pass `composition_space='elements'` to force this; by default it is detected from the headers.
- **Buffered MELTS runs:** oxygen is open, so the buffer fixes Fe³⁺/Fe²⁺. Give total iron as `'FeO'`.

---

## Automatic model selection

You never pick a sub-model by hand. Each call is routed using the headers and data you pass:

```
Input features + headers
        │
        ├─ (MELTS) any nonzero Cr2O3 (or Cr) value?  ── yes → Cr model
        │                                             └─ no  → NoCr model  (missing or all-zero column)
        │
        ├─ entropy header (e.g. 'S(J/g/K)')?          ── yes → isentropic model (+ temperature model)
        │                                              └─ no  → isothermal model
        │
        └─ (MELTS) a log fO2 (QFM) header?          ── yes → fO2-buffered model
                                                       └─ no  → closed-oxygen model (Fe2O3 conserved)
```

- **Cr routing applies to the whole table at once.** One row with Cr2O3 > 0 sends every row to the Cr model. To use the NoCr model for Cr-free rows, split the table yourself.
- **Separate training data:** the Cr and NoCr models were trained on separate datasets. Each model therefore only sees compositions from its own training distribution.

---

## HeFESTo emulators

### `get_isentrope()` — mantle adiabats

```python
import numpy as np
from ngibbs import HeFESToEmulatorCPU

# Bulk Silicate Earth (McDonough & Sun, 1995)
BSE = {'SiO2': 45.0, 'Al2O3': 4.45, 'FeO': 7.75, 'Fe2O3': 0.35,
       'MgO': 37.8, 'CaO': 3.55, 'Na2O': 0.36, 'Cr2O3': 0.38}

headers    = ['P(GPa)', 'T(K)'] + list(BSE)
comp_array = np.array([0, 0] + list(BSE.values()), dtype=np.float32).reshape(1, -1)

T_grid, P_grid, props = HeFESToEmulatorCPU.get_isentrope(
    features               = comp_array,
    headers                = headers,
    pressures              = np.linspace(0, 140, 300),           # GPa
    potential_temperatures = np.array([1500, 1600, 1700, 1800]), # K
    properties             = ['S', 'rho', 'Vp_fast', 'Vs', 'Kh'],
)
# T_grid, P_grid: torch tensors, shape (4, 300); props[key]: numpy arrays, shape (4, 300)
```

`get_isentrope` has two modes:

- **Potential temperatures given** (shown above): `features` must be a single row.
- **No potential temperatures:** `features` may have many rows, and each row's temperature column is its own potential temperature. Rows may have different compositions:

```python
# An approximate depleted-MORB-mantle composition for comparison
DMM = {'SiO2': 44.7, 'Al2O3': 3.98, 'FeO': 7.9, 'Fe2O3': 0.3,
       'MgO': 38.7, 'CaO': 3.17, 'Na2O': 0.13, 'Cr2O3': 0.57}

comp_array = np.array([[0, 1800] + list(BSE.values()),
                       [0, 1500] + list(DMM.values())], dtype=np.float32)

T_grid, P_grid, props = HeFESToEmulatorCPU.get_isentrope(
    features=comp_array, headers=headers,
    pressures=np.linspace(0, 140, 300), properties=['rho', 'Vs'],
)   # shapes (2, 300)
```

`properties` accepts any key of the HeFESTo EOS calculator (see [below](#hefesto-get_property_hefesto_vectorized_from_assemblage)). Omit it to get only `(T_grid, P_grid)`. The Martian model, `HeFESToMarsEmulatorCPU`, has the same API.

### `ForwardMB()` — direct forward pass

```python
out = HeFESToEmulatorCPU.ForwardMB(
    features,                       # (N, F) array with an entropy column → isentropic model
    headers = headers,
    outputs = ['component_moles', 'phase_tables', 'temperature'],
)
out['component_moles']              # (N, n_components) torch tensor
comp_wtpct, mass_wtpct = out['phase_tables']   # phase compositions and phase masses (wt%)
out['temperature']                  # (N,) temperature in K; isentropic models only
```

Available `outputs`:

| Output | Contents |
|---|---|
| `'component_moles'` | component moles in the reduced component space (the default) |
| `'phase_moles'` | moles of each phase |
| `'phase_tables'` | `(phase composition wt%, phase mass wt%)` |
| `'phase_present'` | predicted phase presence |
| `'temperature'` | temperature in K (isentropic models only) |
| `'ptt_out'` | PetThermoTools-style tables (see [MELTS](#melts-emulators)) |

---

## MELTS emulators

### `ForwardMB()` — phase assemblages along P–T paths

```python
import numpy as np
from ngibbs import MELTS102EmulatorCPU

MORB = {'SiO2': 48.68, 'TiO2': 1.01, 'Al2O3': 17.64, 'Cr2O3': 0.03,
        'FeO': 7.59, 'Fe2O3': 0.89, 'MgO': 9.10, 'CaO': 12.45,
        'Na2O': 2.65, 'K2O': 0.03, 'P2O5': 0.08, 'H2O': 0.20}

P_bar = np.repeat([2000, 5000, 10000], 250)          # 3 isobars ...
T_C   = np.tile(np.linspace(1400, 900, 250), 3)      # ... × 250 temperatures
grid  = np.column_stack([P_bar, T_C, np.tile(list(MORB.values()), (len(P_bar), 1))])
headers = ['P(bar)', 'T(C)'] + list(MORB)

out = MELTS102EmulatorCPU.ForwardMB(grid, headers=headers, outputs=['ptt_out'])
ptt = out['ptt_out']   # {'Conditions': DataFrame, 'liquid1': DataFrame, 'olivine1': ..., ...}

# Split into one set of tables per isobar
_, table_id = np.unique(grid[:, 0], return_inverse=True)
per_isobar = MELTS102EmulatorCPU.divide_ptt_tables(ptt, table_id)
```

`'ptt_out'` uses the table layout of [PetThermoTools](https://github.com/gleesonm1/PetThermoTools), so nGibbs can stand in for alphaMELTS calls in existing PetThermoTools workflows.

### Thermodynamic properties: `outputs=['ptt_props']`

Ask for `'ptt_props'` instead of `'ptt_out'` to also run the built-in vectorized MELTS EOS on the predicted assemblage. It adds bulk columns to `ptt['Conditions']`: `H_J`, `S_J/K`, `V_cm^3`, `rho_kg/m^3`, `Cp_J/K`, `dVdP_cm^3/bar`, `log10(fO2)`. It also adds a `<phase>_prop` table per phase, in the layout of PetThermoTools' `calc_phase_props_MELTS_parallel`:

```python
out = MELTS102EmulatorCPU.ForwardMB(grid, headers=headers, outputs=['ptt_props'])
out['ptt_out']['Conditions'][['T_C', 'P_bar', 'rho_kg/m^3', 'S_J/K']]
```

### fO2-buffered runs (MELTS 1.0.2)

Add a log fO2 column, given as an offset from QFM (e.g. `'logfO2(QFM)'`), and give total iron as `FeO`:

```python
oxides  = {k: v for k, v in MORB.items() if k != 'Fe2O3'}
grid_fo2 = np.column_stack([P_bar, T_C, np.full_like(T_C, 0.0),          # QFM+0
                            np.tile(list(oxides.values()), (len(P_bar), 1))])
headers_fo2 = ['P(bar)', 'T(C)', 'logfO2(QFM)'] + list(oxides)
out = MELTS102EmulatorCPU.ForwardMB(grid_fo2, headers=headers_fo2, outputs=['ptt_out'])
```

---

## Equation-of-state property calculators

Both calculators are deterministic Python ports of the simulators' own thermodynamics; they are not neural networks. You can feed them assemblages predicted by an emulator, or ground-truth assemblages read from simulator output.

### HeFESTo: `get_property_hefesto_vectorized_from_assemblage`

This is a vectorized translation of HeFESTo's `physub` routines. Aggregate properties are Voigt–Reuss–Hill averages over the phases.

```python
import numpy as np, torch
from ngibbs import HeFESToEmulatorCPU as H

# Emulate an assemblage on a (P, S) grid, then evaluate the EOS at the emulated T
P = np.linspace(1, 135, 300); S = np.full_like(P, 2.2)                # GPa, J/g/K
feats = np.column_stack([P, S, np.tile(list(BSE.values()), (len(P), 1))])
hdrs  = ['P(GPa)', 'S(J/g/K)'] + list(BSE)

out = H.ForwardMB(feats, headers=hdrs, outputs=['component_moles', 'temperature'])
PT  = torch.tensor(np.column_stack([P, out['temperature'].cpu().numpy()]), dtype=torch.float64)
props = H.get_property_hefesto_vectorized_from_assemblage(
    out['component_moles'].double(), PT, property_names=['rho', 'Vp_fast', 'Vs', 'cptot'])
```

You can also read component moles straight from HeFESTo's `fort.99` files (`ngibbs.utils.file_utils.load_fort99_componentMoles`).

| Key | Quantity | Units |
|---|---|---|
| `rho` | density | g/cm³ |
| `Vp`, `Vs`, `Vb` | P, S and bulk-sound velocity (isomorphic, VRH) | km/s |
| `Vp_fast`, `Vb_fast` | velocities including order–disorder relaxation; **compare these to HeFESTo's fort.56 VP/VB** | km/s |
| `S` | entropy | J/g/K |
| `Kh`, `Gh` | Hill adiabatic bulk and shear moduli | GPa |
| `Kv`, `Kr`, `Gv`, `Gr` | Voigt / Reuss bounds | GPa |
| `Kh_fast`, `Kv_fast`, `Kr_fast` | relaxed bulk moduli | GPa |
| `cptot`, `cvtot` | isobaric / isochoric heat capacity, including phase-change terms | J/g/K |
| `alptot` | thermal expansivity, including phase-change terms | 1/K |
| `gamtot` | Grüneisen parameter | — |
| `KStot`, `KTtot` | adiabatic / isothermal bulk modulus, including phase-change terms | GPa |
| `cpiso`, `alpiso`, … (`*iso`) | the same, without the phase-change term | |
| `ClapeyronSlope`, `deltaent`, `deltavol`, `dndt`, `dndp` | phase-change diagnostics | |

`Vp` without `_fast` is slightly faster to compute but up to ~0.26% high in the lower mantle relative to HeFESTo. The `*tot`/`*iso`/phase-change keys and the `_fast` keys each require an extra linear solve, so they are computed only when requested. The method's docstring lists per-species outputs as well.

### MELTS: `get_property_melts_vectorized_from_assemblage`

This is a vectorized port of the MELTS liquid and solid-solution thermodynamics. The simplest route is `ForwardMB(..., outputs=['ptt_props'])` (above). To call it directly, pass a per-phase composition dict, a per-phase moles dict and `PT` (P in GPa, T in K). It returns bulk (whole-assemblage) values of `V`, `Cp`, `K`, `alpha`, `G`, `H` and `S` by default; `rho` and `melts_coverage_fraction` (the share of the assemblage the EOS could evaluate) are also available. Use `property_names=('bulk', 'per_phase')` to get every bulk value plus a phase-by-phase breakdown.

Some predicted liquids cannot be represented in MELTS's liquid basis. For example, a slightly peralkaline residual melt with K > Al gives a negative component. For those rows the EOS raises an error rather than extrapolating.

---

## Performance: GPU, batch sizes and the CPU fallback

- **GPU objects** (`...GPU`) process inputs in chunks, and on GPU objects small inputs are automatically sent to the matching CPU model, because transfer overhead dominates for small inputs.
- **Batch sizes** default to conservative values. Tune them once per GPU; the result is saved and reused in later sessions:

```python
import ngibbs
ngibbs.tune_batch_size()                       # tunes the packaged HeFESTo GPU model
ngibbs.tune_batch_size(model='MELTS120Emulator')
ngibbs.show_batch_tuning()                     # inspect the saved profile
ngibbs.clear_batch_tuning()                    # forget it
```

Set `NGIBBS_BATCH_TUNING=0` to ignore tuned values, or `NGIBBS_CONFIG_DIR` to change where they are stored. Any call also accepts an explicit `batch_size=`.

---

## Training range warnings

Rows outside a model's training range (pressure, temperature/entropy, or composition) still get predictions, but these are extrapolations. A `TrainingRangeWarning` lists the offending rows. To make it an error:

```python
import warnings
from ngibbs.engine.emulator import TrainingRangeWarning
warnings.simplefilter('error', TrainingRangeWarning)
```

If a model file is missing from your installation, the error message also reports whether a newer nGibbs is available on PyPI.

---

## Validating the models

Every model object can check itself against ground truth that ships with the package. These checks need matplotlib, which is a core dependency.

```python
from ngibbs import HeFESToEmulatorCPU as api

report = api.test()          # metrics tables + figures
api.assert_quality()         # raises EmulatorQualityError below the shipped thresholds
```

`test()` scores the model on two datasets that ship with it:

- **`validationDataset/`:** held-out samples from the model's preprocessed test set: per-phase precision and recall, abundance and composition errors, mass-balance residuals, training-coverage plots and phase-abundance parity plots.
- **`standardWorkspace/`:** the raw simulator runs shipped for the model (standard rocks along a HeFESTo adiabat or a MELTS isobaric cooling path), scored exactly as a custom workspace is (below).

By default everything is written to `./ngibbs_deployment_test/<model name>/` in the current directory; pass `output_dir=` to choose another location.

### Testing your own simulations (custom workspaces)

`evaluate_workspace(path)` scores the model against a directory of your own raw simulator output, without rerunning anything else. It only measures: it never fails on accuracy thresholds. It searches the directory recursively for runs:

- **MELTS:** any folder with a `*.melts` input file and its alphaMELTS `*_tbl.txt` / `*.tbl` outputs. Each run is routed to the Cr or NoCr model (an all-zero Cr2O3 column selects NoCr).
- **HeFESTo:** any folder with a `control` file and its `fort.56` and `fort.99` outputs.

```python
from ngibbs import MELTS120EmulatorCPU as api

res = api.evaluate_workspace('path/to/alphamelts_runs')
# -> ./ngibbs_deployment_test/MELTS120Emulator/customWorkspace/alphamelts_runs/

api.evaluate_workspace()     # no path: the model's standard workspace -> .../standardWorkspace/
```

The standard and custom workspaces go through the same code and produce the same files:

| Output | What it covers |
|---|---|
| `emulator_quality_<key>.csv`, `plots/abundance_parity_<key>.png` | Every assemblage of every run, each scored on its own |
| `property_errors.csv`, `plots/property_parity_<key>.png` | Bulk properties for every run. Red is the built-in EOS on the emulated assemblage, blue is the same EOS on the simulator's assemblage, both against the simulator's own value, with R² for each |
| `phase_errors.csv` | Phase-fraction errors for every run |
| `plots/phase_section_<key>.png`, `plots/property_section_<key>.png` | The first three runs (`section_runs=`) along their own path |
| `benchmark_manifest.json` | Which runs were used, rows dropped (unmodelled phase, out of pressure range, invalid), runs skipped or failed, and why |

`<key>` is the pathway (`isothermal`, `isentropic`, and for buffered MELTS runs `openox`), prefixed with `nocr_`/`cr_` for MELTS.

Section plots need a 1-D path. A run whose pressure never changes is drawn as an isobar against temperature. A run whose pressure never repeats is drawn as a depth transect against pressure. A run with repeated but varying pressures is listed in the manifest and left out of the sections. Its assemblages still count in every table and parity plot.

The phase and property comparison is fault-tolerant. If it fails for a run, or the model has no EOS parameters, you get a warning naming the run and the reason, and every other output is still written.

### Benchmarks and regression checks

`benchmark()` writes the validation-dataset and standard-workspace results as one indexed set. `check_benchmark()` reruns them and compares with a stored baseline. It fails if any metric is worse than the baseline by more than the allowed tolerance, or if the underlying data changed:

```python
api.check_benchmark()                            # against the baseline shipped with the package
api.benchmark('my_benchmark')                    # write your own
api.check_benchmark(baseline_dir='my_benchmark') # raises BenchmarkRegressionError on a regression
```

Baselines ship inside the package next to each model's checkpoints, in `ngibbs/engine/TrainedModels/<family>/benchmark_baseline/<cpu|cuda>/`, so an installed copy can check itself. They are kept per device because CPU and GPU float32 arithmetic round differently. Custom workspaces are never part of a benchmark. In the repository, `python scripts/run_benchmarks.py [--device cuda]` checks every shipped model, and the PyPI release workflow runs it on CPU before publishing.

---

## Tutorials

Jupyter notebooks in [`notebooks/`](notebooks/):

- [`HeFESTo_tutorial.ipynb`](notebooks/HeFESTo_tutorial.ipynb): inputs, `ForwardMB`, and EOS properties for the HeFESTo emulators
- [`HeFESTo_EarthAdiabatTutorial.ipynb`](notebooks/HeFESTo_EarthAdiabatTutorial.ipynb): mantle adiabats with `get_isentrope`
- [`MELTS_tutorial.ipynb`](notebooks/MELTS_tutorial.ipynb): MELTS emulators, PetThermoTools-style output, and buffered runs

---

## Package layout

```
src/ngibbs/                         # the pip-installable package
├── __init__.py                     # model objects (lazy), tune_batch_size, show/clear_batch_tuning
├── engine/
│   ├── models.py                   # model objects (also importable from ngibbs directly)
│   ├── API.py                      # EmulatorAPI, HeFESToAPI, MELTSAPI (routing, ForwardMB, get_isentrope, ...)
│   ├── emulator.py                 # NN_MELTS inference wrapper, TrainingRangeWarning
│   ├── mass_balance.py             # mass-balance projection
│   ├── melts_assemblage.py         # emulator output → MELTS EOS assemblage / ptt_props
│   ├── NN.py, NN_continuous.py     # network architectures
│   ├── EOS_arithmetic/hefesto_vec/ # vectorized HeFESTo EOS
│   ├── EOS_arithmetic_MELTS/melts_vec/  # vectorized MELTS EOS
│   └── TrainedModels/              # HeFESToEarth/, HeFESToMars/, 102/, 120/ checkpoints + test subsets
├── deployment_tests/               # test(), evaluate_workspace(), benchmark(), standard workspaces, baselines
├── config/                         # constants, ml_indexer, projection tables
└── utils/                          # file parsing (fort.*), math helpers, batch_tuning
```

---

## Roadmap

- **In progress:** isentropic MELTS emulators; pMELTS emulators (including an isentropic, low-melt-fraction model with and without Cr); phMELTS.
- **Planned:** native integration into PetThermoTools; Apple Neural Engine support.

Data generation, processing and training are documented in [DEVELOPMENT.md](DEVELOPMENT.md).

---

## Citing nGibbs

If you use nGibbs in your research, please cite it with the metadata in [`CITATION.cff`](CITATION.cff). The concept DOI [10.5281/zenodo.20766922](https://doi.org/10.5281/zenodo.20766922) always resolves to the latest version. Please also cite MELTS or HeFESTo as appropriate.

## License

AGPL-3.0. See [LICENSE](LICENSE).
