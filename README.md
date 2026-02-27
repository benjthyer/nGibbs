# nMELTS: Neural Network Emulator for MELTS Thermodynamic Modeling

A machine learning emulator for MELTS (and other) thermodynamic phasse equilibria models for silicate systems. This project replaces computationally expensive simulations with fast neural network predictions.

NOTE: This Repository supports data generation, processing and training to produce new models. It is NOT intended at this time to be made public, and therefore needn't be strictly stable. 
The public release will be limited to everything within ([src/nMELTS](src/nMELTS)). This portion is standalone and includes all model infrastructure and operations necesary to use
nMELTS with pretrained models, allowing for tighter quality control and implementation simplicity.

## Table of Contents

- [Overview](#overview)
- [Project Status](#project-status)
- [Architecture](#architecture)
- [Dependencies](#dependencies)
- [Workflow](#workflow)
  - [1. Data Generation](#1-data-generation)
  - [2. Data Processing](#2-data-processing)
  - [3. Model Training](#3-model-training)
  - [4. Emulator Usage](#4-emulator-usage)
- [Module Structure](#module-structure)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**nMELTS** is a CUDA-powered neural network emulator trained on MELTS thermodynamic simulations. Instead of running MELTS directly, nMELTS (on CUDA) provides ~1000 times faster predictions of:

- Phase assemblages at equilibrium 
- Phase compositions in component space of reduced dimensionality.
- Phase masses and mole fractions
- Molar abundances of phases
- Key thermodynamic properties of the system

**Key Features:**
- This pipeline flexibly produces diverse MELTS datasets to train different models that cover different use-cases and composition spaces, optimizing for accuracy where it counts for the user. 
  - Dynamic indexers: DatasetIndexer() class in indexer.py. Builds indexers based on table values and column names. 
    - Fundamental object: Elkeys is controlled by oxides present in 'Bulk_comp' columns of MELTS tables. 
    - Spawn ml_indexer() class, which is carried by nMELTS models during training. 


- I have done my best to keep these training datasets well-documented by carrying along their stats files and the configuration files used in their processing.
- Training data is generated with alphamelts 2.0.3 running on GNU parallel on WSL. Native python parallelism may be more effectively supported for Macs-- currently untested.
- Large datasets can be manipulated. On my machine, .csv files up to ~10 GB can be manipulated. The training products are vastly reduced in size
- The inputs and outputs of nMELTS are flexible, accomadating a wide variety of use cases.

---

## Project Status

### ✅ Completed

- **Data Processing Pipeline** ([src/builder/processing/](src/builder/processing/))
  - MELTS CSV parsing with memory-mapped arrays and assemblage metadata ([BigMetaTable.py](src/builder/processing/BigMetaTable.py))
  - Data filtering and quality checks ([filters.py](src/builder/processing/filters.py))
  - Upsampling of assemblages containing rare phases, or upsampling by perturbing modal abundances to change bulk composition / properties within an EQM assemblage
  - Export to ML-ready tar.gz bundles ([MLexporter.py](src/builder/processing/MLexporter.py))
  - Configuration-driven processing workflow, human inteligible for reproducibility/transparency ([prepareML.py](src/builder/processing/prepareML.py))

- **Feature/Label Generation**
  - Dynamic indexing from dataset headers ([src/builder/indexer.py](src/builder/indexer.py))
    - ml_indexer built from this larger indexer. It contains the important information necesary to interface with the NN models. 
    - Transformation/projection matrices for oxide↔element↔component conversions
    
  - Molar abundance, normalized to total sum of system elements = 1, excluding oxygen
  - Mass proportion of each, normalized to total sum of system oxide mass = 100, with all iron represented as FeO
  - Binary (present/absent) phase assemblage labels

- **Utilities**
  - Mathematical functions (QFM buffer, normalization, transformations) ([src/nMELTS/utils/math_utils.py](src/nMELTS/utils/math_utils.py))
  - File operations and directory management ([src/nMELTS/utils/file_utils.py](src/nMELTS/utils/file_utils.py))
  - String parsing utilities ([src/nMELTS/utils/string_utils.py](src/nMELTS/utils/string_utils.py))

- **Model Infrastructure**
  - PyTorch Dataset and DataLoader integration ([src/builder/training/torchDataClass.py](src/builder/training/torchDataClass.py))
  - Neural network architecture ([src/nMELTS/engine/NN.py](src/nMELTS/engine/NN.py))
  - Emulator inference wrapper with mass balance enforcement ([src/nMELTS/engine/emulator.py](src/nMELTS/engine/emulator.py))

- **Model Training**
  - Training loop implementation ([src/builder/training/tuners.py](src/builder/training/tuners.py))
  - Hyperparameter optimization ([src/builder/training/tuners.py](src/builder/training/tuners.py))

- **Testing & Validation**
  - Unit tests for processing pipeline

### 🟡 In Progress (req'd for release)
- **Training Models**
  - First targets: 
    - Geodynamically relevant isentropic low melt fraction pMELTS emulator (w/wo Cr Models)
    - "General" (any melt fraction) emulators for MELTS 1.0, MELTS 1.2, and pMELTS
    - HeFESTo Models

- **Inference API and Deployment Hardening**
  - Inference support is nearly complete in `NN_MELTS` (table-based input handling, internal transformations, and chemistry reconstruction)
  - User-side mass balance is enabled by default and will be configureable for 0/1/2/3-stage residual fitting to balance speed vs. fidelity
  - Ongoing API polish for user-facing wrappers and additional refinement of spinel/pyroxene perturbation handling
  - Deployment packaging and notebook tutorials


### 🔵 Planned

- Support for Apple's Neural Engine
- Integration into PTT
- Integration tests for end-to-end workflows
- User-side automated testing of performance and quality of canned models against representative datasets. Execute upon installation/updates?

---

## Architecture

```
Raw MELTS (or HeFESTo / MAGEmin) Data (.tbl files: data/Workspace)

    ↓ Irreversibly Distilled and Compiled into

(CSV + TXT files: data/MELTStables)

    ↓
┌─────────────────────────────────────────┐
│  Data Processing & Filtering            │
│  (BigMetaTable, filters, MLexporter)    │
└─────────────────────────────────────────┘
    ↓
ML-Ready Dataset Bundle (tar.gz : data/MLready)
  ├─ features.npy          (Three Thermodynamic state variables (e.g. P, T, fO2) + element fractions)
  ├─ molar_labels.npy      (mole fractions of phases)
  ├─ binary_labels.npy     (present/absent phases)
  ├─ mass_labels.npy       (wt% phase masses)
  ├─ labels.npy            (intensive component fractions)
  ├─ ml_indexer/           (dataset/model indexer saved as state files)
  │   ├─ indexer_metadata.json   (indexer metadata and feature/free-output names)
  │   ├─ indexer_structure.json  (phase/component dictionaries and mappings)
  │   └─ indexer_arrays.npz      (projection matrices and normalizer state arrays)
  ├─ free_outputs.npy      (OPTIONAL: non-chemical outputs not tied into mass conservation)
  ├─ stats.txt             (dataset statistics: composition range, phase abundances, liquid fraction)
  └─ processing.yaml       (config used for generation)
    ↓
┌─────────────────────────────────────────┐
│  PyTorch Training Pipeline              │
│  (loadTrainData, torchDataClass, tuners)│
└─────────────────────────────────────────┘
    ↓
Trained Neural Network Model (tar)
  └─ model_name.pt (zip package)
    ├─ state_dict.pt           (PyTorch weights)
    ├─ config.json             (model architecture)
    ├─ metadata.json           (save-time metadata)
    ├─ ml_indexer/              
    │   ├─ indexer_metadata.json
    │   ├─ indexer_structure.json
    │   └─ indexer_arrays.npz
    ├─ model.yaml              (optional: model config used during training)
    ├─ training.yaml           (optional: training orchestration YAML)
    ├─ data_processing.yaml    (optional: processing YAML for training data)
    ├─ stats.txt               (optional: dataset stats)
    └─ log.txt                 (optional: training/tuning log)
    ↓
┌─────────────────────────────────────────┐
│  Emulator Inference                     │
│  (emulator.py, NN.py)                   │
└─────────────────────────────────────────┘
    ↓
Phase Predictions (fast!)
```

---

## Dependencies

### Python Version
- Python 3.8+ (built on 3.10.16)

### Core Libraries
```
numpy>=1.19
pandas>=1.2
PyYAML>=5.4
```

### Machine Learning
```
torch>=1.9          # PyTorch for neural network training (built on 2.5.1)
scikit-learn>=0.24  # Data preprocessing and utilities
```

### Data Processing
```
tqdm>=4.60          # Progress bars
matplotlib>=3.3     # Plotting (Harker diagrams, T-P-fO2 distributions)
```

### Installation

```bash
# Activate your environment
conda activate torch-env

# Install from requirements.txt
pip install -r requirements.txt
```

For details on specific dependency versions and optional packages, see [requirements.txt](requirements.txt).

---

## Workflow

### 1. Data Generation

External dependency: alphamelts
MELTS simulations are orchestrated through alphamelts by parallelized (GNU parallel) terminal calls in WSL.

Raw MELTS data comes from MELTS/alphaMELTS simulations of bulk compositions from GEOROC and PetDB. Each simulation produces 
text files within the working directory of alphamelts (folders within data/Workspace)
These text files contain 100s-1000s of EQM assemblages

**Status:** ✅ External process (uses MELTS/alphaMELTS software directly)

These are compiled/reduced into:
- **CSV file**: Numerical data (temperatures, pressures, phase masses, thermodynamic state variables, compositions, etc.)
- **TXT file**: Metadata (run IDs, MELTS version, simulation parameters)

Then the folders with the MELTS-generated tables are cleared. 

**To Be Expanded:**
- Detailed guide on running MELTS simulations
- Example input files and workflows
- Best practices for generating diverse, balanced datasets

---

### 2. Data Processing

Converts raw MELTS CSV/TXT into ML-ready bundles.

**Data Processing Pipeline Steps:**

1. **Load & Index** ([BigMetaTable.py](src/builder/processing/BigMetaTable.py))
   - Parse CSV into memory-mapped arrays (for memory efficiency on large datasets)
   - Build dynamic indexers from column headers
   - Associate metadata from TXT files

2. **Filter & Clean** ([filters.py](src/builder/processing/filters.py))
   - Remove rows with unsupported phases
   - Apply oxide composition bounds
   - Filter outliers and physically invalid assemblages
   - Separate analcime from leucite phases

3. **Balance Dataset** ([filters.py](src/builder/processing/filters.py))
   - Upsample rare phase assemblages
   - Apply liquid fraction balancing
   - Ensure diverse phase stability fields represented

4. **Generate Features & Labels** ([MLexporter.py](src/builder/processing/MLexporter.py))
   - Extract P, T, fO2 as input features
   - Transform oxide compositions to element molar fractions
   - Calculate phase molar abundances
   - Generate binary phase presence/absence labels
   - Apply random resampling to augment dataset

5. **Export Bundle** ([MLexporter.py](src/builder/processing/MLexporter.py))
   - Save all .npy arrays to memory-mapped files
   - Archive with transformation matrices and indexer
   - Include configuration used (processing.yaml)

**Configuration:** [config/processing.yaml](config/processing.yaml)

**Usage:**

```python
from src.builder.processing.prepareML import process_for_ML

# Use default config
process_for_ML()

# Or customize
process_for_ML(
    MELTSModel='110',
    Date='Feb3',
    Mode='BatchCooling',
    upsample=True,
    use_external=False
)
```

---

### 3. Model Training

Trains neural networks on ML-ready bundles.

Training is controlled by YAML files that can execute a sequence of train/tune episodes.
Configuration is merged in layered precedence:

1) Per-episode overrides (e.g., `train1`, `tune1`, `train2`, ...) OVERRIDES
2) Global values in the selected training YAML... OVERRIDES
3) defaults yaml in code (no episodes in default, only globals)

Per-episode overrides can update both optimization hyperparameters and model
architecture. Architecture changes are supported between episodes via warm-start
loading of compatible weights.


**Model Architecture:** ([NN.py](src/nMELTS/engine/NN.py))
- Multi-layer perceptron with customizable depth/width
- Output heads for each phase (molar abundance, mass, composition)
- Physical constraints enforced during inference (mass balance, bounds)

**Training Orchestration:** ([main.py](src/builder/training/main.py))
- Sequential train/tune episodes are discovered from YAML (`train1`, `tune1`, ...)
- Later episodes can inherit tuned state from earlier episodes
- Global and per-episode overrides are deep-merged with centralized type conversion
- YAML can drive schedulers, regularization, architecture, and loop behavior without code edits


**Pipeline Steps:**

1. **Load Data** ([loadTrainData.py](src/builder/training/loadTrainData.py))
   - Extract tar.gz bundles
   - Load .npy files into PyTorch DataLoaders
   - Split into train/validation/test sets

2. **Create Datasets** ([torchDataClass.py](src/builder/training/torchDataClass.py))
   - Wrap .npy arrays in PyTorch Dataset classes
   - Support multiple label configurations (binary, molar, mass)
   - Batching and shuffling

3. **Train Model** ([tuners.py](src/builder/training/tuners.py))
   - Forward pass with architecture and loss behavior controlled by YAML episode config
   - Backpropagation and gradient updates
   - Penalize mass-balance mismatch during training objectives where configured


4. **Evaluate** 
   - 1/2/3 step Routine to relax system component abundances to enforce massconservation
   - Compare predictions to validation set
   - Compute phase prediction accuracy


---

### 4. Emulator Usage

Run the trained emulator for fast phase predictions.

The current implementation is centered on `NN_MELTS` and supports
table-based user inputs in wt% oxides, with internal reordering and
transformations to model feature space.

**Schematic High-Level Usage (wrapper-style)** 

```python
from src.nMELTS.engine.emulator import NN_MELTS

def run_emulator_table(model_name, input_table, columns=None, fit_stages=1):
    """
    Schematic convenience wrapper for user-side workflows.
    input_table: tabular rows with input conditions (P, T, fO2, H, S) and bulk oxide columns (wt%). Constant Fe2O3 can and often should be specified instead of fO2.
    fit_stages: 0, 1, 2, or 3 mass-balance fitting stages.
    """
    if columns is None:
      try: 
        columns = np.array(input_table.columns.values)
      except:
        raise ValueError('If column headers not explicitly passed, the input_table must have be a pandas dataFrame with headers')

    emulator = NN_MELTS(load_model_from_zip(path_dict[model_name]))
    required_inputs = np.array(emulator.featureNames + [OXIDES_FROM_ELEMENTS[el] for el in emulator.Elkeys])
    assert np.all(np.isin(required_inputs, columns))
    input_table = emulator.reorderMELTStable(input_table)
    input_table = emulator.condition_oxide_table(input_table)
    predictions = emulator.forward(input_table, fitStages=fit_stages) # many outputs, few desired.
    predictions = emulator.condition_predictions(predictions) # Output format configurable?
    return predictions

# This wrapper is schematic documentation for planned user-facing API.
# Current internals already support table ingestion and mass-balance workflows.
```

**Status:** 🟡 Nearly complete for inference, with ongoing user-facing API polish

**Mass Balance Behavior:**
- User-side mass-balance enforcement is default behavior in inference workflows
- Stage depth is configurable (0/1/2/3) to trade runtime against correction strength
- Remaining refinement targets include robust handling of spinel/pyroxene perturbations

**To Be Expanded:**
- Detailed API documentation
- Final convenience API signatures and naming
- Input/output format specifications and examples
- Example prediction workflows
- Comparison with original MELTS predictions
- Uncertainty quantification

---

## Module Structure

```
src/
├── builder/                          # Data processing, alphamelts tools, and model training, not necesary for inference with trained models, not publically distributed
│   ├── alphamelts/                   # MELTS/alphaMELTS interfacing scripts
│   ├── indexer.py                    # Dynamic dataset indexing from table headers
│   ├── processing/                   # MELTS data processing pipeline
│   │   ├── BigMetaTable.py           # Parse and manage large MELTS datasets
│   │   ├── filters.py                # Data quality and balancing filters
│   │   ├── MLexporter.py             # Feature/label generation and bundle export
│   │   ├── export_only.py            # Direct csv/txt -> ML bundle export workflow
│   │   ├── prepareML.py              # Main processing orchestration
│   │   └── __init__.py
│   ├── training/                     # YAML-driven training and tuning infrastructure
│   │   ├── main.py                   # CLI orchestration for sequential train/tune episodes
│   │   ├── trainer.py                # Core training loops
│   │   ├── tuners.py                 # Hyperparameter tuning loops
│   │   ├── optimizer_factory.py      # Optimizer and scheduler construction
│   │   ├── loadTrainData.py          # Load ML bundles into PyTorch DataLoaders
│   │   ├── torchDataClass.py         # Dataset wrappers
│   │   ├── logger.py                 # Training log utilities
│   │   ├── validation/               # Validation and diagnostic utilities
│   │   └── __init__.py
│   └── __init__.py
│
└── nMELTS/                           # Core, deployable emulator package. Installable (future) with pip
    ├── config/                       # Configuration & constants
    │   ├── constants.py              # Phase-component mappings, oxide lists
    │   ├── ml_indexer.py             # ML transformation matrices
    │   ├── projections/              # Projection tables used by indexer/model transforms
    │   ├── README_MLIndexer.md       # Detailed ml_indexer structure and usage notes
    │   └── __init__.py
    ├── engine/                       # Neural network & inference
    │   ├── NN.py                     # Neural network architecture
    │   ├── emulator.py               # High-level emulator interface
    │   ├── TrainedModel/             # Stored trained models for local usage: potentially updated often
    │   └── __init__.py
    ├── utils/                        # Utility functions
    │   ├── math_utils.py             # Math operations (QFM, normalization, etc.)
    │   ├── file_utils.py             # File & directory management
    │   ├── string_utils.py           # String parsing
    │   └── __init__.py
    └── __init__.py
```

---

## Configuration

### Processing Configuration

Configure the data processing pipeline via [config/processing.yaml](config/processing.yaml):

```yaml
dataset:
  MELTSModel: '102'          # MELTS version (102, 110, 120, p)
  Date: 'Feb3'               # Dataset date identifier
  Mode: 'BatchCooling'       # Simulation mode
  subset: false              # Use subset for testing
  use_external: false        # Use external storage

preprocessing:
  preprocessed: false        # Skip if already processed
  filter_full_metadata: true # Filter invalid phases
  separate_analcime: true    # Separate analcime from leucite

upsampling:
  enabled: true              # Upsample rare phases
  phases:
    nepheline:
      n_resamples: 10
      multiplier_bounds: [0.8, 1.1]
    # ... more phases ...

resampling:
  train_bounds:
    - [1, 1]                 # Identity
    - [0.8, 1]               # Shifted proportions (~20%)
    - [0.5, 1]               # Shifted proportions (~50%)
  test_bounds:
    - [1, 1]                 # Identity only

plot:
  enabled: false             # Generate diagnostic plots

deep_filter:
  oxide_lower_bounds: [...]  # Composition constraints
  oxide_upper_bounds: [...]
  component_upper_bounds: [...]
```

### Project Configuration

Note: This Copilot agents are to follow strict structure and changelog rules.
See `.github/copilot-instructions.md`.


Set paths and storage locations in [config/config.yaml](config/config.yaml):

```yaml
# Internal vs. external storage paths
# Configure before running large datasets
```

---
