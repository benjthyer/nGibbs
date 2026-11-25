# nGibbs development guide

This guide is for working on nGibbs itself: generating training data with alphaMELTS and HeFESTo, turning it into ML-ready bundles, training emulators and maintaining the regression benchmarks. To *use* the emulators, see [README.md](README.md).

Everything here needs the full repository and its extra dependencies:

```bash
git clone https://github.com/benjthyer/nGibbs.git
cd nGibbs
pip install -e .
pip install -r requirements.txt     # scikit-learn, scipy, PyYAML, optuna, psutil, ...
```

Repository-only code lives in `src/builder/` (it is not part of the PyPI package), `scripts/` and `recipes/`.

## Contents

- [Pipeline overview](#pipeline-overview)
- [1. Data generation — MELTS](#1-data-generation--melts)
- [2. Data generation — HeFESTo on the HPC cluster](#2-data-generation--hefesto-on-the-hpc-cluster)
- [3. Data processing](#3-data-processing)
- [4. Model training](#4-model-training)
- [5. Shipping a model](#5-shipping-a-model)
- [6. Benchmarks and the release gate](#6-benchmarks-and-the-release-gate)
- [7. Releasing](#releasing)
- [builder layout](#builder-layout)

---

## Pipeline overview

- **Dynamic indexers:**
  - `DatasetIndexer` in `src/builder/indexer.py` builds an indexer from table values and column headers.
  - The element set (`Elkeys`) follows the oxides present in the `Bulk_comp` columns of the MELTS/HeFESTo tables.
  - It spawns the `ml_indexer` embedded in every trained model; see `src/ngibbs/config/README_MLIndexer.md`.
- **Documented datasets:** configuration and statistics files are carried alongside each dataset bundle.
- **Scale:** large tables are memory-mapped; `.csv` files of about 10 GB work on standard hardware.

### Data flow

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

## 1. Data Generation — MELTS

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

Separate datasets are generated for Cr-bearing and Cr-free compositions (see "Automatic model selection" in the README).

---

## 2. Data Generation — HeFESTo on the HPC Cluster

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

### Preparation in Detail

The preparation script `prepare_HeFESTo_tree_fulladiabat()` (and variants for Martian conditions) does the following for each of N simulations:

1. Samples a rock composition from GEOROC, filtering for MgO > 20 wt% to target mantle-relevant rocks.
2. Speciates iron: distributes total iron between Fe²⁺ (FeO) and Fe³⁺ (Fe₂O₃) across a grid from Fe³⁺/Fetotal = 0.0 to 0.10.
3. Converts the oxide weight-percent composition to element molar abundances, normalized to a target of 24 total moles.
4. Writes a HeFESTo `control` file into each `SimulationN/` directory, substituting the sampled element values and run parameters (pressure range, number of steps).
5. Computes an isentropic P–T path using the mantle potential-temperature regression (`get_S`, `get_T`) and writes it to `ad.in` with small random noise to improve coverage.

Each batch contains at most 1000 simulations. If more are requested, additional `Batch0002/`, `Batch0003/`, … directories are created automatically.

### Job Submission

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

## 3. Data Processing

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
from builder.processing.prepareML import process_for_ML

process_for_ML(
    MELTSModel='102',
    Date='Jun10',
    Mode='BatchCooling',
    upsample=True,
)
```

### Processing configuration

The processing pipeline is configured with YAML:

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

---

## 4. Model Training

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


### Training configuration

Training recipes live in `recipes/training/`. Top-level keys set the dataset and the global architecture/regularization. Numbered `tuneN` / `trainN` blocks are episodes, run in order, and each one overrides the globals:

```yaml
# recipes/training/HeFESToIsentropic.yaml (abridged)
tarname: 'HeFESTo_041026_NPT_low_noise_adiabats'   # loads <tarname>_Train / <tarname>_Test bundles
warm_start: None                                    # or a checkpoint name to start from

encoderLayerUp: 1
middleLayerUp: 2
low_regularization: batchnormdropout0.6
high_regularization: batchnormdropout0.3
lowWD: 1E-3
highWD: 1E-4
noise: 0.05

tune1:                       # grid-tunes the listed parameters, in order
  strategy: lower            # which part of the network trains (others frozen)
  batch_size: 1024
  epochs: 15
  learning_rate: 5E-4
  early_stopping_patience: 2
  max_N: 5E5
  tune_params:
    low_regularization: [dropout0.3, batchnormdropout0.3, batchnormdropout0.6]
    encoderLayer: [[1,0], [1,1], [2,2], [3,2]]
    noise: [0, 0.1, 0.05, 0.01]

train1:
  strategy: lower
  batch_size: 1024
  epochs: 70
  learning_rate: 1E-3
  early_stopping_patience: 15
  scheduler:
    type: CosineAnnealingLR  # any torch.optim.lr_scheduler class
    args: {T_max: 7}
  binweights:                # up-weight rare phases in the presence loss
    akimotoite: 10
    plagioclase: 10
```

Run a recipe from the repository root:

```bash
python -m builder.training.main train --config recipes/training/HeFESToIsentropic.yaml
```

The CLI also takes `--stage {all,lower,upper,finetune}` and `--tarname` (to override the recipe's dataset); see `python -m builder.training.main --help`. The `MELTSft_*.yaml` recipes are the fine-tuning recipes for the shipped MELTS models.

---

## 5. Shipping a model

A model is one folder, `src/ngibbs/engine/TrainedModels/<Model>/`, holding the current (working) copy of that model. Its folder name is its public name (`make_ngibbs('<Model>')`). Only the working copy is kept in git; every released version lives on [Hugging Face](https://huggingface.co/benjthyer/nGibbs) as `<Model>/v<X.Y>/`, so delete old copies locally freely.

1. **Files:** the checkpoints (`*.tar`, plus a temperature `*.pt` for isentropic models) and their `*_Test_subset15000.tar.gz` validation bundles at the top of the folder (`EmulatorAPI` matches them by filename: `NPT`/`NPS`/`Open`, `Cr`/`NoCr`), and the raw standard-rock simulator runs in `standards/`.
2. **Model card:** a `README.md` beginning with a YAML header, which the release checks:
   ```yaml
   ---
   model: MELTS120              # = the folder name
   api: MELTSAPI                # or HeFESToAPI
   api_kwargs:                  # optional, passed to the API class
     melts_version: MELTS120
   summary: one line, shown in the model index
   requires_ngibbs: ">=1.3,<3"  # optional; default: same major series, not newer than nGibbs
   license: agpl-3.0
   ---
   ```
   followed by a description of the model, its data and how to cite it.
3. **Baselines:** generate them (next section) into `benchmarks/<Model>/` and commit them.
4. **Release** it in the next X.Y.0 release ([Releasing](#releasing)): it becomes `<Model>vX.Y`. Only files git tracks are published; local outputs in the folder are ignored.

Changing anything but `README.md` in a released model makes it a new version at the next minor release. Editing only the model card updates it in place, in any release.

---

## 6. Benchmarks and the release gate

`scripts/run_benchmarks.py` runs `benchmark()` / `check_benchmark()` (see the README) for every model working copy:

```bash
python scripts/run_benchmarks.py                        # check all models on CPU
python scripts/run_benchmarks.py --device cuda          # check against the cuda baselines
python scripts/run_benchmarks.py --generate [--device cuda] [--models MELTS120]
```

Each run scores two things:

- **Validation dataset:** the model's `*_Test_subset15000` bundles.
- **Standard workspace:** the model's raw standard-rock runs (`standards/`), through `evaluate_workspace`: per-assemblage quality tables plus the per-run property and phase comparisons.

Custom workspaces are measured with `evaluate_workspace(path)` and are never gated.

Pathways a model doesn't ship (e.g. isentropic MELTS) are skipped with a warning.

**Baselines:**

- **Location:** `benchmarks/<Model>/<cpu|cuda>/`, in git but outside the package: they record what this code gives for each working copy, so they change with the code, while released model versions never change. Plots are drawn by default (`--no-plots` to skip) and are git-ignored.
- **Per device:** baselines are kept per device because CPU and GPU float32 kernels round differently. Minor-phase compositions amplify that difference past any tolerance tight enough to be useful.
- **When to regenerate:** whenever a model or its reference data change on purpose. A changed dataset shows up as *incomparable* and fails the check, so a data change can't pass as a model change.
- **Tolerances:** in `BenchmarkTolerances` (`src/ngibbs/deployment_tests/benchmark.py`).

**Release gate:** the release workflow calls `.github/workflows/benchmark.yml`, which runs the CPU check on a GitHub runner; nothing is published unless every model passes. The benchmark workflow can also be run alone from the Actions tab; tick *generate* to produce CPU baselines on the runner and download them as an artifact, if the runner's CPU rounds differently from your machine's.

---

## Releasing

Releases are started by hand: **Actions → Release → Run workflow**, with the version (`1.1.0`, `1.1.1`, `1.2.0rc1`) and *dry run* ticked or not. A version number is used up only by the PyPI upload, which comes after every check, so a release that fails anywhere before it can simply be run again with the same number.

| Step | What it does | Fails when |
|---|---|---|
| plan | compares every working copy with Hugging Face and writes the plan to the run's summary page | the version is malformed, already on PyPI or tagged, or not newer than PyPI's latest; a patch release (Z ≠ 0) has new or changed models; a new or changed model has no valid model card |
| benchmark | `scripts/run_benchmarks.py` on CPU | any model regresses |
| build | wheel + sdist with that version | a file is over 95 MB |
| stage | opens a Hugging Face pull request with the new `<Model>/vX.Y/` folders and `index.json` | |
| smoke | installs the wheel in a clean venv and loads every model from that pull request | a model is missing, incomplete or the wrong version |
| publish | **waits for your approval**, then uploads to PyPI | |
| finalize | merges the Hugging Face pull request (the models go live), tags `v<version>`, creates the GitHub release, updates `CITATION.cff` | |
| discard | dry runs and stopped releases: closes the Hugging Face pull request | |

The plan reads, for each model:

```
New model detected: MELTS120 -> will be uploaded to Hugging Face as MELTS120v1.0
Changed model detected: MELTS120v1.0 -> MELTS120v1.1: will be uploaded to Hugging Face as MELTS120v1.1; ...
Unchanged: MELTS102 (stays MELTS102v1.0)
```

Run it locally first; it changes nothing: `python scripts/release.py plan 1.1.0`.

**Rules:**

- **Model versions** are named after the minor release that ships them (`vX.Y`) and are final once `X.Y.0` is released: never overwritten, never deleted by the workflow (delete one by hand on Hugging Face if you ever must).
- **Pre-releases** (`X.Y.0rcN`, `aN`, `bN`) can ship models as a pre-release `vX.Y`, which later pre-releases of the same `X.Y.0` may replace. `X.Y.0` makes it final (re-uploading it if it changed).
- **Major releases** (`X.0.0`) republish every model in the checkout as `vX.0` (same files), since nGibbs X.x does not load older-major models. Models you dropped from the checkout stay on Hugging Face for older installs; the plan warns about them.
- **Dry runs** (the default) do everything up to and including the smoke test against a staging pull request, then close it: use them to test the workflow itself.

**One-time setup** (Settings of the GitHub repository):

- **Secrets:** `HUGGINGFACEAUTH` (a Hugging Face write token for `benjthyer/nGibbs`) and `PYPI_TOKEN`.
- **Environment:** create an environment named `release` with yourself as a required reviewer. The publish step waits for that approval, and the plan step refuses a real release if the environment has no required reviewer.

If `finalize` fails after PyPI has the release, re-run that job alone; each of its steps is safe to repeat.

---

## builder layout

```
src/builder/                     # repository only; not in the PyPI package
├── HeFESTo/HeFESTo_functions.py # HPC tree preparation, fort.* import, bulk properties
├── alphamelts/                  # alphaMELTS 2.3.1 driver, .melts file builder, batch files
├── alphameltsMAC/               # macOS variant
├── indexer.py                   # DatasetIndexer: dynamic indexing from table headers
├── plotting.py                  # diagnostic plots
├── processing/                  # BigMetaTable, filters, MLexporter, prepareML, affinity labels
└── training/                    # main.py CLI, trainer, tuners, data loaders, temperature FCNN
```
