# nGibbs: Neural Network Emulators for Silicate Thermodynamics

A collection of fast neural network emulators for MELTS, HeFESTo, and related free-energy codes for silicate multi-component systems. nGibbs replaces computationally expensive thermodynamic simulations with GPU-accelerated neural network predictions that are orders of magnitude faster.

---

## Quick Install

The core inference package is available on PyPI. This is the recommended starting point for most users — it includes pre-trained models and the full inference API, but not the data-generation or training machinery.

```bash
pip install nGibbs
```

```python
from ngibbs import MELTS102EmulatorCPU, HeFESToEmulatorCPU
```

**PyPI package vs. this repository.** The `pip install` package (`src/ngibbs/`) is a lightweight, stable distribution containing trained model weights, the inference engine, and user-facing APIs. This GitHub repository is the full development environment: it adds `src/builder/` for data generation (running MELTS/alphaMELTS/HeFESTo at scale), data processing pipelines, and PyTorch training infrastructure. If you only want to *use* the emulators, `pip install nGibbs` is all you need. If you want to generate new training data or retrain models, you need the full repository and its extended dependencies.

---

## Table of Contents

- [Overview](#overview)
- [Project Status](#project-status)
- [Architecture](#architecture)
- [Dependencies](#dependencies)
- [Workflow](#workflow)
  - [1. Data Generation — MELTS](#1-data-generation--melts)
  - [2. Data Generation — HeFESTo on the HPC Cluster](#2-data-generation--hefesto-on-the-hpc-cluster)
  - [3. Data Processing](#3-data-processing)
  - [4. Model Training](#4-model-training)
  - [5. Emulator Usage](#5-emulator-usage)
    - [HeFESTo Emulator API](#hefesto-emulator-api)
    - [MELTS Emulator API](#melts-emulator-api)
    - [Vectorized HeFESTo Bulk Property Calculator](#vectorized-hefesto-bulk-property-calculator)
- [Module Structure](#module-structure)
- [Configuration](#configuration)
- [License](#license)

---

## Overview

**nGibbs** provides ~1000× faster predictions of phase equilibria and thermodynamic properties compared to running MELTS or HeFESTo directly. The emulators predict:

- Phase assemblages at equilibrium
- Phase compositions in a reduced component space
- Phase masses and mole fractions
- Molar abundances of phases
- Key thermodynamic properties: temperature, entropy, density, seismic velocities, elastic moduli

**Key Features:**

- Flexible training pipelines produce datasets for different models covering different use cases and composition spaces, optimizing accuracy where it matters.
  - Dynamic indexers: `DatasetIndexer` class in `indexer.py`. Builds indexers from table values and column names.
    - Fundamental object: `Elkeys` is controlled by the oxides present in `Bulk_comp` columns of MELTS/HeFESTo tables.
    - Spawns `ml_indexer` class, which is embedded in each trained model.

- Training datasets are well-documented: configuration files and statistics files are carried alongside each dataset.
- Training data for MELTS is generated with alphaMELTS 2.0.3 running on GNU parallel in WSL. Native Python parallelism is also supported on macOS.
- Training data for HeFESTo is generated on the Caltech HPC cluster using SLURM array jobs.
- Large datasets can be manipulated; `.csv` files up to ~10 GB work on standard hardware.

---

## Project Status

### ✅ Completed

- **Data Processing Pipeline** ([src/builder/processing/](src/builder/processing/))
  - MELTS CSV parsing with memory-mapped arrays and assemblage metadata ([BigMetaTable.py](src/builder/processing/BigMetaTable.py))
  - Data filtering and quality checks ([filters.py](src/builder/processing/filters.py))
  - Upsampling of assemblages containing rare phases, or by perturbing modal abundances within an equilibrium assemblage
  - Export to ML-ready tar.gz bundles ([MLexporter.py](src/builder/processing/MLexporter.py))
  - Configuration-driven processing workflow ([prepareML.py](src/builder/processing/prepareML.py))

- **Feature/Label Generation**
  - Dynamic indexing from dataset headers ([src/builder/indexer.py](src/builder/indexer.py))
    - `ml_indexer` built from the larger `DatasetIndexer`. Contains the information necessary to interface with neural network models.
    - Transformation/projection matrices for oxide ↔ element ↔ component conversions
  - Molar abundance, normalized so the total sum of system elements = 1 (excluding oxygen)
  - Mass proportion of each phase, normalized so total system oxide mass = 100 (all iron as FeO)
  - Binary (present/absent) phase assemblage labels

- **Model Infrastructure**
  - PyTorch Dataset and DataLoader integration ([src/builder/training/torchDataClass.py](src/builder/training/torchDataClass.py))
  - Neural network architecture ([src/ngibbs/engine/NN.py](src/ngibbs/engine/NN.py))
  - Emulator inference with mass-balance enforcement ([src/ngibbs/engine/emulator.py](src/ngibbs/engine/emulator.py))
  - HeFESTo and MELTS high-level emulator APIs ([src/ngibbs/engine/API.py](src/ngibbs/engine/API.py))

- **Model Training**
  - Training loop implementation ([src/builder/training/trainer.py](src/builder/training/trainer.py))
  - Hyperparameter optimization via Optuna ([src/builder/training/tuners.py](src/builder/training/tuners.py))
  - YAML-driven sequential train/tune episodes ([src/builder/training/main.py](src/builder/training/main.py))

- **HeFESTo EOS Arithmetic Engine** ([src/ngibbs/engine/EOS_arithmetic/](src/ngibbs/engine/EOS_arithmetic/))
  - Full Python translation of HeFESTo's physub thermodynamic routines
  - Vectorized batch API: compute bulk properties for N assemblages simultaneously
  - Returns density, Vp, Vs, Cp, Cv, α, K, G, entropy, and more from component mole arrays

- **HeFESTo HPC Data Generation Workflow** ([src/builder/HeFESTo/](src/builder/HeFESTo/))
  - Composition sampling from GEOROC/PetDB databases
  - Automated SimulationN directory tree generation with per-simulation control files and P–T paths
  - SLURM array job submission scripts with queue-size monitoring
  - Batch import of fort.56/61/68/99 output files into training CSV tables

- **MELTS and HeFESTo Inference APIs** with automatic model selection (see [Emulator Usage](#5-emulator-usage))

- **Testing & Validation**
  - Unit tests for processing pipeline
  - Benchmark harnesses comparing EOS outputs against HeFESTo ground truth (fort.56)

### 🟡 In Progress

- **Trained Models**
  - Geodynamically-relevant isentropic low-melt-fraction pMELTS emulator (w/ and w/o Cr)
  - General (any melt fraction) emulators for MELTS 1.0, 1.2, and pMELTS
  - HeFESTo models

- **fO2-Buffered MELTS Emulator**
  - Infrastructure for open-oxygen (buffered fO2) emulators is already in place (the `OXYGEN='open'` flag in `DatasetIndexer` and the model-selection routing in the API)
  - Training data generation for buffered runs is underway; buffered models will be available in an upcoming release

- **Deployment Packaging and Notebooks**
  - Tutorial notebooks for HeFESTo and MELTS emulators
  - Integration tests for end-to-end workflows

### 🔵 Planned

- Vectorized MELTS bulk property calculator (equivalent to the existing HeFESTo version — see [below](#vectorized-hefesto-bulk-property-calculator))
- Integration into PetThermoTools
- Support for Apple's Neural Engine
- User-side automated performance testing against representative datasets

---

## Architecture

```
Raw MELTS (or HeFESTo) Simulations
  (alphaMELTS .tbl files  /  HeFESTo fort.* files)

    ↓ Compiled and reduced into

Annotated CSV + TXT files
  (data/MELTStables  /  data/HeFESToWorkspace)

    ↓
┌──────────────────────────────────────────┐
│  Data Processing & Filtering             │
│  (BigMetaTable, filters, MLexporter)     │
└──────────────────────────────────────────┘
    ↓
ML-Ready Dataset Bundle (tar.gz : data/MLready)
  ├─ features.npy          (P, T/S, fO2, element fractions)
  ├─ molar_labels.npy      (phase mole fractions)
  ├─ binary_labels.npy     (present/absent phases)
  ├─ mass_labels.npy       (wt% phase masses)
  ├─ labels.npy            (intensive component fractions)
  ├─ ml_indexer/           (dataset/model indexer)
  │   ├─ indexer_metadata.json
  │   ├─ indexer_structure.json
  │   └─ indexer_arrays.npz  (projection matrices, normalizer state)
  ├─ free_outputs.npy      (OPTIONAL: non-chemical outputs)
  ├─ stats.txt             (composition range, phase abundances)
  └─ processing.yaml       (config used for generation)
    ↓
┌──────────────────────────────────────────┐
│  PyTorch Training Pipeline               │
│  (loadTrainData, torchDataClass, tuners) │
└──────────────────────────────────────────┘
    ↓
Trained Neural Network Model (.tar)
  └─ model_name.pt  (zip package)
      ├─ state_dict.pt
      ├─ config.json
      ├─ metadata.json
      ├─ ml_indexer/
      ├─ model.yaml  (optional)
      ├─ training.yaml  (optional)
      ├─ stats.txt  (optional)
      └─ log.txt  (optional)
    ↓
┌──────────────────────────────────────────┐
│  Emulator Inference (pip package)        │
│  (API.py, emulator.py, NN.py)            │
└──────────────────────────────────────────┘
    ↓
Phase Predictions + Thermodynamic Properties (fast!)
```

---

## Dependencies

### Python Version
- Python 3.9+ (built and tested on 3.10.16)

### Core (pip package)
```
numpy>=1.20
torch>=1.9
pandas>=1.3
tqdm>=4.67
```

### Development / Training (full repo)
```
scikit-learn>=0.24
matplotlib>=3.3
PyYAML>=5.4
optuna>=3.0
```

### Installation

```bash
# End-user: install the inference package from PyPI
pip install nGibbs

# Developer: install from source in editable mode
git clone https://github.com/...
cd nGibbs
```

---

## Workflow

### 1. Data Generation — MELTS

MELTS simulations are orchestrated through alphaMELTS by parallelized (GNU parallel) terminal calls in WSL. Bulk compositions are sampled from the GEOROC and PetDB databases.

Raw MELTS data comes from MELTS/alphaMELTS simulations of natural bulk compositions. Each simulation produces text files within the alphaMELTS working directory. These are compiled/reduced into:
- **CSV file**: numerical data (temperatures, pressures, phase masses, thermodynamic state variables, compositions, etc.)
- **TXT file**: metadata (run IDs, MELTS version, simulation parameters)

Training data generation scripts live in `scripts/MELTedGEOROC*.py`. The key control parameters are:

```python
MELTSmodels = ['102']  # MELTS version: '102', '110', '120', or 'p' for pMELTS
calctype    = 'Cooling'  # 'Cooling' (isobaric) or 'Compression'
FXes        = ['Batch']  # 'Batch' (equilibrium) or 'FxCryst' (fractional crystallization)
ZeroOxides  = ['MnO', 'NiO']  # oxides set to zero in all runs
```

Separate datasets are generated for Cr-bearing and Cr-free compositions (see [Automatic Model Selection](#automatic-model-selection) below).

---

### 2. Data Generation — HeFESTo on the HPC Cluster

HeFESTo training data is generated on the Caltech HPC cluster using SLURM. The workflow has three stages: **Preparation → Execution → Import**.

```
GEOROC/PetDB Database
   (natural rock compositions)
         │
         ▼ scripts/prepare_hefesto_tree_fulladiabat.py
         │    - Sample N compositions (mafic/ultramafic filter)
         │    - Speciate iron across a Fe³⁺/Fetotal grid (0–10%)
         │    - Convert oxide wt% → element moles
         │    - Compute isentropic P–T path for each composition
         │      using get_S(T=Tp, Ca) → get_T(S, P)
         │
         ▼
BatchNNNN/
  ├── Simulation1/
  │     ├── control   ← HeFESTo input: element amounts + run parameters
  │     └── ad.in     ← P–T path file: (P, _, T) rows along isentrope
  ├── Simulation2/
  │     ├── control
  │     └── ad.in
  ├── ...
  └── Simulation1000/
        ├── control
        └── ad.in
         │
         ▼ scripts/run_sbatches.py  (or run_many_sbatches.py for large jobs)
         │    - Creates a SLURM array job for each BatchNNNN/
         │    - Each task: cd SimulationN && $HOME/HeFESTo/HeFESToRepository/main
         │    - run_many_sbatches.py monitors queue size (max ~9500 jobs)
         │      and staggers submission as the queue drains
         │
         ▼  [SLURM runs HeFESTo on the cluster]
         │
BatchNNNN/
  ├── Simulation1/
  │     ├── control, ad.in
  │     ├── fort.56   ← bulk system properties (P, T, S, rho, Vp, Vs, …)
  │     ├── fort.61   ← phase densities
  │     ├── fort.68   ← phase volumes
  │     └── fort.99   ← phase-component molar abundances
  │
         ▼ scripts/import_hefesto_subdirs.py
         │    - Recursively finds all workspace directories
         │    - Cleans up control-only (failed) simulation dirs
         │    - Calls import_HeFESTo_components() on each workspace:
         │        * Parses control + fort.56/61/68/99 for each simulation
         │        * Appends rows to a master CSV matching DatasetIndexer layout
         │        * Optionally writes a separate CSV of phase-boundary rows
         │
         ▼
HeFESTo_TrainsetXXX.csv
  (training-ready table with P, T, S, bulk composition,
   phase assemblage, component moles, bulk properties)
```

#### Preparation in Detail

The preparation script `prepare_HeFESTo_tree_fulladiabat()` (and variants for Martian conditions) does the following for each of N simulations:

1. Samples a rock composition from GEOROC, filtering for MgO > 20 wt% to target mantle-relevant rocks.
2. Speciates iron: distributes total iron between Fe²⁺ (FeO) and Fe³⁺ (Fe₂O₃) across a grid from Fe³⁺/Fetotal = 0.0 to 0.10.
3. Converts the oxide weight-percent composition to element molar abundances, normalized to a target of 24 total moles.
4. Writes a HeFESTo `control` file into each `SimulationN/` directory, substituting the sampled element values and run parameters (pressure range, number of steps).
5. Computes an isentropic P–T path using the mantle potential-temperature regression (`get_S`, `get_T`) and writes it to `ad.in` with small random noise to improve coverage.

Each batch contains at most 1000 simulations. If more are requested, additional `Batch0002/`, `Batch0003/`, … directories are created automatically.

#### Job Submission

`run_sbatches.py` creates one SLURM array job per batch directory and submits them all:

```bash
python scripts/run_sbatches.py \
    --base-dir $HOME/HeFESTo/Simulations \
    --minutes 10
```

For very large runs (tens of thousands of simulations), `run_many_sbatches.py` monitors the queue and submits new jobs as capacity becomes available:

```bash
python scripts/run_many_sbatches.py \
    --base-dir $HOME/HeFESTo/Simulations \
    --max-queued 9500 \
    --check-interval 60 \
    --minutes 10
```

Each SLURM task `cd`s into its `SimulationN/` directory and runs `$HOME/HeFESTo/HeFESToRepository/main`. The full environment setup (Python virtual environment, library paths) is written into the SLURM script at submission time.

A worked end-to-end example for the Caltech cluster:

```bash
# 1. Prepare the directory tree from the login node
python $HOME/HeFESTo/nGibbs/scripts/prepare_hefesto_tree_fulladiabat.py \
    --directory    $HOME/HeFESTo/DemoSimulations \
    --georoc-dir   $HOME/HeFESTo/nGibbs/data/GEOROC_PETDB_UNFILTERED_WHOLEROCK_TRAIN.csv \
    --control-path $HOME/HeFESTo/nGibbs/src/builder/alphamelts/batch/shallowHeFESTo \
    --n 5000

# 2. Submit array jobs (staggers if queue is large)
python $HOME/HeFESTo/nGibbs/scripts/run_many_sbatches.py \
    --base-dir $HOME/HeFESTo/DemoSimulations \
    --max-queued 9500 --minutes 10

# 3. After jobs complete, import results into a training CSV
python $HOME/HeFESTo/nGibbs/scripts/import_hefesto_subdirs.py \
    --root     $HOME/HeFESTo/DemoSimulations \
    --dataname $HOME/HeFESTo/data/HeFESTo_TrainsetDemo.csv
```

---

### 3. Data Processing

Converts raw MELTS/HeFESTo CSV tables into ML-ready bundles.

**Pipeline Steps:**

1. **Load & Index** ([BigMetaTable.py](src/builder/processing/BigMetaTable.py))
   - Parse CSV into memory-mapped arrays for memory efficiency on large datasets
   - Build dynamic indexers from column headers
   - Associate metadata from TXT files

2. **Filter & Clean** ([filters.py](src/builder/processing/filters.py))
   - Remove rows with unsupported phases
   - Apply oxide composition bounds
   - Filter physically invalid assemblages

3. **Balance Dataset** ([filters.py](src/builder/processing/filters.py))
   - Upsample rare phase assemblages
   - Apply liquid fraction balancing
   - Ensure diverse phase stability fields are represented

4. **Generate Features & Labels** ([MLexporter.py](src/builder/processing/MLexporter.py))
   - Extract P, T, fO2 as input features
   - Transform oxide compositions to element molar fractions
   - Calculate phase molar abundances
   - Generate binary phase presence/absence labels

5. **Export Bundle** ([MLexporter.py](src/builder/processing/MLexporter.py))
   - Save all `.npy` arrays to memory-mapped files
   - Archive with transformation matrices and indexer
   - Include configuration used

**Usage:**

```python
from src.builder.processing.prepareML import process_for_ML

process_for_ML(
    MELTSModel='102',
    Date='Jun10',
    Mode='BatchCooling',
    upsample=True,
)
```

---

### 4. Model Training

Trains neural networks on ML-ready bundles.

Training is controlled by YAML files that execute a sequence of train/tune episodes. Configuration is merged in layered precedence:

1. Per-episode overrides (e.g. `train1`, `tune1`, `train2`, …) OVERRIDES
2. Global values in the selected training YAML… OVERRIDES
3. Defaults YAML in code (no episodes in default, only globals)

Per-episode overrides can update both optimization hyperparameters and model architecture. Architecture changes between episodes are supported via warm-start loading of compatible weights.

**Pipeline Steps:**

1. **Load Data** ([loadTrainData.py](src/builder/training/loadTrainData.py)) — extract bundles, split train/val/test
2. **Create Datasets** ([torchDataClass.py](src/builder/training/torchDataClass.py)) — wrap `.npy` arrays in PyTorch Dataset classes
3. **Train Model** ([tuners.py](src/builder/training/tuners.py)) — forward pass, backpropagation, mass-balance penalty terms
4. **Evaluate** — 1/2/3-stage residual relaxation to enforce mass conservation; comparison to validation set

---

### 5. Emulator Usage

---

#### HeFESTo Emulator API

The HeFESTo emulator models isentropic mantle adiabats from surface to the core–mantle boundary. Two singleton instances are available after `pip install nGibbs`:

```python
from ngibbs.engine.API import HeFESToEmulatorCPU   # always available
from ngibbs.engine.API import HeFESToEmulatorGPU   # available if CUDA detected
```

The emulator manages a *pair* of neural networks — an isothermal model and an isentropic model — along with a temperature prediction FCNN. These work together to map a composition and thermodynamic state onto a full adiabatic profile.

##### Input Format

The emulator accepts a 2D numpy array of features and a matching list of header strings. Headers follow the `'quantity(phase)'` convention used throughout nGibbs tables:

| Header | Description |
|---|---|
| `'P(GPa)(System_main)'` | Pressure in GPa |
| `'T(K)(System_main)'` | Temperature in K (isothermal mode) |
| `'S(J/g/K)(System_main)'` | Specific entropy in J/g/K (isentropic mode) |
| `'SiO2'`, `'MgO'`, `'FeO'`, `'Fe2O3'`, `'CaO'`, `'Al2O3'`, `'Na2O'`, `'Cr2O3'` | Oxide weight percents |
| `'Si'`, `'Mg'`, `'Fe'`, `'Ca'`, `'Al'`, `'Na'`, `'Cr'`, `'O'` | Alternative: element mole fractions |

Oxide and element inputs are both supported. When both `'FeO'` and `'Fe2O3'` are supplied, iron speciation is used directly. If only `'FeO'` is given, all iron is treated as ferrous.

##### `get_isentrope()` — Compute Mantle Adiabats

The primary high-level method. Given a composition and a set of potential temperatures, it computes full isentropic P–T profiles and the corresponding aggregate physical properties.

```python
import numpy as np
from ngibbs.engine.API import HeFESToEmulatorCPU

# Bulk Silicate Earth (McDonough & Sun, 1995)
BSE = {
    'SiO2': 45.0, 'Al2O3': 4.45, 'FeO': 7.75, 'Fe2O3': 0.35,
    'MgO': 37.8,  'CaO': 3.55,   'Na2O': 0.36, 'Cr2O3': 0.38
}

headers     = ['P(GPa)(System_main)', 'T(K)(System_main)'] + list(BSE.keys())
comp_array  = np.array([0, 0] + list(BSE.values()), dtype=np.float32).reshape(1, -1)
pressures   = np.linspace(0, 140, 300)   # GPa
Tpotentials = np.array([1500, 1600, 1700, 1800])  # K

T_grid, P_grid, props = HeFESToEmulatorCPU.get_isentrope(
    features               = comp_array,
    headers                = headers,
    pressures              = pressures,
    potential_temperatures = Tpotentials,
    properties             = ['S', 'rho', 'Vp', 'Vs', 'Kh'],
)

# T_grid  : shape (4, 300) — temperature (K) along each isentrope
# P_grid  : shape (4, 300) — pressure (GPa) along each isentrope
# props   : dict with keys 'S', 'rho', 'Vp', 'Vs', 'Kh', each shape (4, 300)
```

Multiple distinct compositions can be evaluated simultaneously by providing one row per composition in `comp_array`. If `potential_temperatures` is omitted, the temperature column in `features` is used directly as the surface condition for each row:

```python
# Two compositions, each with their own starting temperature
comp_array = np.zeros((2, len(headers)), dtype=np.float32)
comp_array[0, 1] = 1800   # Tp for first composition
comp_array[1, 1] = 1500   # Tp for second composition
comp_array[0, 2:] = list(BSE.values())
comp_array[1, 2:] = list(BSMars.values())

T_grid, P_grid, props = HeFESToEmulatorCPU.get_isentrope(
    features   = comp_array,
    headers    = headers,
    pressures  = pressures,
    properties = ['rho', 'Vp', 'Vs'],
)
# T_grid, P_grid, and each property have shape (2, 300)
```

Available output properties: `'S'` (entropy, J/g/K), `'rho'` (density, g/cm³), `'Vp'` (P-wave velocity, km/s), `'Vs'` (S-wave velocity, km/s), `'Kh'` (Hill adiabatic bulk modulus, GPa), `'Gh'` (Hill shear modulus, GPa).

##### `ForwardMB()` — Low-Level Forward Pass

For more direct control, `ForwardMB` runs the neural network(s) on an arbitrary input grid and returns component-level outputs:

```python
output = HeFESToEmulatorCPU.ForwardMB(
    features,
    headers = headers,
    outputs = ['component_moles', 'temperature'],
)

component_moles = output['component_moles']   # torch.Tensor (N, n_components)
temperature     = output['temperature']        # torch.Tensor (N,)
```

This is used internally by `get_isentrope` and is also useful when you need the raw mineral assemblage — for example, to pass directly into the vectorized bulk property calculator (see below).

---

#### MELTS Emulator API

The MELTS emulator targets crustal and lithospheric processes: isobaric crystallization, polybaric fractionation, and related workflows. Emulators are available for MELTS 1.0.2 and are being trained for MELTS 1.2 and pMELTS.

```python
from ngibbs import MELTS102EmulatorCPU   # always available
from ngibbs import MELTS102EmulatorGPU   # available if CUDA detected
```

##### Input Format

Inputs are a 2D array of (pressure, temperature, oxide wt%) rows, with matching headers:

| Header | Description |
|---|---|
| `'Pressure(System_main)'` | Pressure in bar, MELTS |  `'P(GPa)(System_main)'` | Pressure in GPa, HeFESTo |
| `'Temperature(System_main)'` | Temperature in °C, MELTS | `'T(K)(System_main)'` | Temperature in °K, HeFESTo |
| `'SiO2'`, `'TiO2'`, `'Al2O3'`, `'FeO'`, `'Fe2O3'`, `'MgO'`, `'CaO'`, `'Na2O'`, `'K2O'`, `'P2O5'`, `'H2O'` | Oxide weight percents (sum to 100) |
| `'Cr2O3'` | Optional; triggers Cr-bearing model (see below) |

##### `ForwardMB()` — Predict Phase Assemblages

```python
import numpy as np
from ngibbs import MELTS102EmulatorCPU

MORB = {
    'SiO2': 48.68, 'TiO2': 1.01,  'Al2O3': 17.64, 'Cr2O3': 0.03,
    'FeO':   7.59, 'Fe2O3': 0.89, 'MgO':    9.10,  'CaO':  12.45,
    'Na2O':  2.65, 'K2O':   0.03, 'P2O5':   0.08,  'H2O':   0.20,
}

# Evaluate at a grid of P–T conditions
P_bar = np.repeat([2000, 5000, 10000], 250)        # 3 pressures × 250 temperatures
T_C   = np.tile(np.linspace(900, 1400, 250), 3)    # °C

comp_cols   = np.tile(list(MORB.values()), (len(P_bar), 1))
input_grid  = np.hstack([P_bar.reshape(-1, 1), T_C.reshape(-1, 1), comp_cols])
headers     = ['Pressure(System_main)', 'Temperature(System_main)'] + list(MORB.keys())

output = MELTS102EmulatorCPU.ForwardMB(
    input_grid,
    headers = headers,
    outputs = ['ptt_out'],   # PetThermoTools-compatible output format
)
ptt_result = output['ptt_out']
```

The `'ptt_out'` output format matches the table structure returned by PetThermoTools, making it easy to substitute nGibbs for direct MELTS calls in existing PTT workflows. Separate isobaric tables can be recovered with:

```python
P_bar_unique, tableNo = np.unique(input_grid[:, 0], return_inverse=True)
ptt_tables = MELTS102EmulatorCPU.divide_ptt_tables(ptt_result, tableNo)
# ptt_tables : list of DataFrames, one per unique pressure
```

##### Automatic Model Selection

The MELTS emulator automatically selects the appropriate trained model for each call based on three criteria:

**1. Chromium content.** If `Cr2O3 > 0` in the input, the Cr-bearing model is selected; otherwise the Cr-free (NoCr) model is used. The Cr and NoCr models are trained on separate datasets to ensure each model only encounters compositions within its training distribution.

**2. Temperature vs. entropy input.** The header names determine which model type is used:
- `'Temperature(System_main)'` (MELTS) or `'T(K)(System_main)'` (HeFESTo) → isothermal (T-input) model
- `'S(System_main)'` (MELTS) or `'S(J/g/K)(System_main)'` (HeFESTo) → isentropic (S-input) model

**3. Oxygen fugacity buffering (coming soon).** The infrastructure for open-oxygen (fO2-buffered) emulators is already in place. When released, specifying a log oxygen fugacity (in terms of an fO2 buffer with offsets) will route the call to a model trained on buffered runs, where oxygen fugacity is held constant rather than Fe₂O₃ being conserved.

```
Input features + headers
        │
        ├─ Cr2O3 > 0 ?─── YES ──→ Cr model family
        │                 NO  ──→ NoCr model family
        │
        ├─ Header contains 'S(J/g/K)' ?─── YES ──→ Isentropic model
        │                                   NO  ──→ Isothermal model
        │
        └─ fO2_buffer specified ?─── YES ──→ Buffered model (coming soon)
                                      NO  ──→ Closed-oxygen model (Fe₂O₃ conserved)
```

---

#### Vectorized HeFESTo Bulk Property Calculator

This is a separate, deterministic component — not a neural network, but a Python translation of HeFESTo's own `physub` thermodynamic routines. Given a mineral assemblage expressed as component moles and a set of (P, T) conditions, it computes aggregate physical properties by applying Voigt–Reuss–Hill averaging over all phases.

```python
from ngibbs.engine.API import HeFESToEmulatorCPU
import torch
import numpy as np

# component_moles : (N, n_components) — moles of each mineral component
# PT              : (N, 2)            — [P in GPa, T in K] for each row

component_moles = torch.tensor(my_assemblage, dtype=torch.float64)
PT              = torch.tensor(np.stack([P_gpa, T_kelvin], axis=1), dtype=torch.float64)

props = HeFESToEmulatorCPU.get_property_hefesto_vectorized_from_assemblage(
    component_moles,
    PT,
    property_names = ['S', 'rho', 'Vp', 'Vs', 'Kh', 'Gh'],
)

# props['rho']  : (N,) array — density in g/cm³
# props['Vp']   : (N,) array — P-wave velocity in km/s
# props['Vs']   : (N,) array — S-wave velocity in km/s
# props['S']    : (N,) array — entropy in J/g/K
# props['Kh']   : (N,) array — Hill adiabatic bulk modulus in GPa
# props['Gh']   : (N,) array — Hill shear modulus in GPa
```

The component moles can come from two sources: (1) imported directly from HeFESTo's `fort.99` output files via `load_fort99_componentMoles()`, or (2) predicted by the isentropic neural network emulator via `ForwardMB(..., outputs=['component_moles'])`. The second path lets you chain the emulator and the EOS calculator to go from a composition + entropy to a full suite of physical properties without running HeFESTo directly.

Full list of available properties:

| Property | Key | Units |
|---|---|---|
| Density | `'rho'` | g/cm³ |
| P-wave velocity | `'Vp'` | km/s |
| S-wave velocity | `'Vs'` | km/s |
| Bulk velocity | `'Vb'` | km/s |
| Entropy | `'S'` | J/g/K |
| Specific heat (isobaric) | `'Cp'` | J/g/K |
| Specific heat (isochoric) | `'Cv'` | J/g/K |
| Thermal expansivity | `'alpha'` | 1/K |
| Grüneisen parameter | `'gamma'` | — |
| Hill bulk modulus | `'Kh'` | GPa |
| Hill shear modulus | `'Gh'` | GPa |
| Voigt bulk modulus | `'Kv'` | GPa |
| Reuss bulk modulus | `'Kr'` | GPa |
| Enthalpy | `'H'` | kJ/g |
| Debye temperature | `'theta_D'` | K |

For a full mantle column (~60 000 rows), typical CPU wall-clock times are 10's of seconds; GPU acceleration reduces this by another factor of 2-5. **An equivalent vectorized bulk property calculator for MELTS is planned but not yet implemented.**

##### End-to-End Example: Emulator → EOS

```python
import numpy as np
import torch
from ngibbs.engine.API import HeFESToEmulatorCPU

BSE_oxides = {
    'SiO2': 45.0, 'Al2O3': 4.45, 'FeO': 7.75, 'Fe2O3': 0.35,
    'MgO': 37.8,  'CaO': 3.55,   'Na2O': 0.36, 'Cr2O3': 0.38
}

# 1. Build a grid of (P, S) conditions for one composition
P_values  = np.linspace(0, 135, 300)
S_values  = np.linspace(1.9, 2.5, 200)
P_grid    = np.repeat(P_values, len(S_values))
S_grid    = np.tile(S_values, len(P_values))
comp_cols = np.tile(list(BSE_oxides.values()), (len(P_grid), 1))
features  = np.hstack([P_grid.reshape(-1, 1), S_grid.reshape(-1, 1), comp_cols])

headers = (
    ['P(GPa)(System_main)', 'S(J/g/K)(System_main)']
    + list(BSE_oxides.keys())
)

# 2. Run the isentropic emulator → component moles + temperature
output = HeFESToEmulatorCPU.ForwardMB(
    features, headers=headers,
    outputs=['component_moles', 'temperature'],
)
comp_moles = output['component_moles']             # (N, n_components)
T_emul     = output['temperature'].cpu().numpy()   # (N,)

# 3. Compute EOS properties on the emulated assemblage
PT = torch.tensor(
    np.stack([P_grid, T_emul], axis=1), dtype=torch.float64
)
props = HeFESToEmulatorCPU.get_property_hefesto_vectorized_from_assemblage(
    comp_moles.double(), PT,
    property_names=['rho', 'Vp', 'Vs', 'S'],
)
# props['rho'] etc. are (N,) numpy arrays in native HeFESTo units
```

---

## Module Structure

```
src/
├── builder/                     # Dev-only: data generation, processing, training
│   ├── HeFESTo/
│   │   └── HeFESTo_functions.py # HPC tree preparation, simulation import, bulk
│   │                            #   property extraction from fort.* files
│   ├── alphamelts/              # MELTS/alphaMELTS interfacing scripts
│   ├── alphameltsMAC/           # macOS variant
│   ├── indexer.py               # Dynamic dataset indexing from table headers
│   ├── plotting.py              # Diagnostic plots
│   ├── processing/              # MELTS data processing pipeline
│   │   ├── BigMetaTable.py      # Parse and manage large MELTS datasets
│   │   ├── filters.py           # Data quality and balancing filters
│   │   ├── MLexporter.py        # Feature/label generation and bundle export
│   │   ├── export_only.py       # Direct CSV → ML bundle workflow
│   │   └── prepareML.py         # Main processing orchestration
│   └── training/                # YAML-driven training and tuning infrastructure
│       ├── main.py              # CLI orchestration for sequential episodes
│       ├── trainer.py           # Core training loops
│       ├── tuners.py            # Hyperparameter tuning loops (Optuna)
│       ├── optimizer_factory.py # Optimizer and scheduler construction
│       ├── loadTrainData.py     # Load ML bundles into PyTorch DataLoaders
│       ├── torchDataClass.py    # Dataset wrappers
│       ├── logger.py            # Training log utilities
│       └── validation/          # Validation and diagnostic utilities
│
└── ngibbs/                      # ← pip-installable package (pip install nGibbs)
    ├── __init__.py              # Exports: MELTS102EmulatorCPU/GPU,
    │                            #          HeFESToEmulatorCPU/GPU
    ├── config/
    │   ├── constants.py         # Phase–component mappings, oxide lists,
    │   │                        #   molar masses, abbreviation tables
    │   ├── ml_indexer.py        # ML transformation matrices
    │   ├── projections/         # Oxide ↔ element ↔ component projection CSVs
    │   └── README_MLIndexer.md  # Detailed ml_indexer structure and usage
    ├── engine/
    │   ├── NN.py                # Neural network architecture (multi-head MLP)
    │   ├── emulator.py          # NN_MELTS: inference wrapper with mass balance
    │   ├── API.py               # High-level APIs: HeFESToAdiabatAPI,
    │   │                        #   HeFESToEmulatorCPU/GPU, MELTS*EmulatorCPU/GPU
    │   ├── EOS_arithmetic/      # Python translation of HeFESTo physub routines
    │   │   ├── hefesto_vec.py   # Vectorized batch EOS calculator
    │   │   ├── physub.py        # Core thermodynamic subroutine
    │   │   ├── gspec.py         # Species-level free energy
    │   │   ├── volume.py        # Equation-of-state volume solver
    │   │   ├── therm.py         # Thermal properties (Debye/Einstein)
    │   │   ├── Ftotsub.py       # Solid-phase free energy
    │   │   └── ...              # Additional EOS helpers
    │   └── TrainedModel/        # Bundled pre-trained .tar model files
    └── utils/
        ├── math_utils.py        # Grid sampling, composition mixing,
        │                        #   adiabat regression, QFM buffer
        ├── file_utils.py        # HeFESTo fort.* file parsing, fixed-width I/O
        └── string_utils.py      # String parsing utilities
```

---

## Configuration

### Processing Configuration

Configure the data processing pipeline via YAML:

```yaml
dataset:
  MELTSModel: '102'          # MELTS version (102, 110, 120, p)
  Date: 'Jun10'              # Dataset date identifier
  Mode: 'BatchCooling'       # Simulation mode
  subset: false

preprocessing:
  preprocessed: false
  filter_full_metadata: true
  separate_analcime: true

upsampling:
  enabled: true
  phases:
    nepheline:
      n_resamples: 10
      multiplier_bounds: [0.8, 1.1]

resampling:
  train_bounds:
    - [1, 1]
    - [0.8, 1]
    - [0.5, 1]
  test_bounds:
    - [1, 1]

deep_filter:
  oxide_lower_bounds: [...]
  oxide_upper_bounds: [...]
```

### Training Configuration

Training is controlled by YAML files in `recipes/training/`. Episodes are discovered automatically from keys like `train1`, `tune1`, `train2`:

```yaml
# recipes/training/HeFESToIsentropic.yaml
model:
  hidden_size: 512
  n_layers: 6
  activation: gelu

train1:
  epochs: 200
  lr: 1e-3
  batch_size: 4096

tune1:
  epochs: 100
  lr: 1e-4
  n_trials: 50   # Optuna trials
```

### Project Notes

Copilot agents operating in this repository follow strict structure and changelog rules. See [`.github/copilot-instructions.md`](.github/copilot-instructions.md).

---

## License

See [LICENSE](LICENSE).
