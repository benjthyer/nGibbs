# nMELTS: Neural Network Emulator for MELTS Thermodynamic Modeling

A machine learning emulator for MELTS thermodynamic modeling software. This project replaces computationally expensive MELTS simulations with fast neural network predictions while maintaining accuracy.

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

**nMELTS** is a neural network emulator trained on MELTS thermodynamic simulations. Instead of running MELTS directly, nMELTS provides ~1000 times faster predictions of:

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

### 🟡 In Progress

- **Model Infrastructure**
  - PyTorch Dataset and DataLoader integration ([src/builder/training/torchDataClass.py](src/builder/training/torchDataClass.py))
  - Neural network architecture with mass balance enforcement ([src/nMELTS/engine/NN.py](src/nMELTS/engine/NN.py))
  - Emulator inference wrapper ([src/nMELTS/engine/emulator.py](src/nMELTS/engine/emulator.py))

- **Model Training**
  - Training loop implementation ([src/builder/training/tuners.py](src/builder/training/tuners.py))
  - Hyperparameter optimization ([src/builder/training/tuners.py](src/builder/training/tuners.py))
  - Validation metrics and convergence checks

- **Testing & Validation**
  - Unit tests for processing pipeline
  - Integration tests for end-to-end workflows
  - Straightforward validation against original MELTS predictions

### 🔵 Planned

- Error analysis and uncertainty quantification
- GPU optimization for others' machines
- Deployment as an installable module with .toml file
- Detailed usage documentation and tutorials
- Integration into PTT

---

## Architecture

```
Raw MELTS Data (.tbl files: data/Workspace)

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
  ├─ ml_indexer.pkl        (dataset/model specific indexers, labels, and transformation matrices)
  ├─ free_outputs.npy      (OPTIONAL: non-chemical outputs not tied into mass conservation)
  ├─ stats.txt             (dataset statistics: composition range, phase abundances, liquid fraction)
  └─ processing.yaml       (config used for generation)
    ↓
┌─────────────────────────────────────────┐
│  PyTorch Training Pipeline              │
│  (loadTrainData, torchDataClass, tuners)│
└─────────────────────────────────────────┘
    ↓
Trained Neural Network Model
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

**Status:** ✅ Nearly Fully implemented

**To Be Expanded:**
- Automatic tests for filters (other reductions' tests are completed)
- Support for External Memory Management is untested in current version

---

### 3. Model Training

Trains neural networks on ML-ready bundles.

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
   - Forward pass with mass balance enforcement
   - Backpropagation and gradient updates
   - Track loss and validation metrics

4. **Evaluate** 
   - Compare predictions to validation set
   - Assess mass balance accuracy
   - Compute phase prediction accuracy

**Model Architecture:** ([NN.py](src/nMELTS/engine/NN.py))
- Multi-layer perceptron with customizable depth/width
- Output heads for each phase (molar abundance, mass, composition)
- Physical constraints enforced during inference (mass balance, bounds)

**Status:** 🟡 In progress (architecture complete, training loop and hyperparameter tuning ongoing)

**To Be Expanded:**
- Complete training loop with logging and checkpointing
- Hyperparameter optimization workflow (in progress with tuners.py)
- Convergence analysis and early stopping (could use more inteligent ways to )
- Loss function design and weighting strategies
- Example training runs with results

---

### 4. Emulator Usage

Run the trained emulator for fast phase predictions.

**Basic Usage:**

```python
from src.nMELTS.engine.emulator import Emulator

# Load trained model
emulator = Emulator(model_path='path/to/trained_model.pt', indexer_path='path/to/indexer.pkl')

# Predict phases at given P, T, fO2, bulk composition
# (Input format to be documented)
predictions = emulator.predict(
    pressure=1.0,           # GPa
    temperature=1200,       # °C
    logfo2=0,              # relative to QFM buffer
    bulk_composition=[...]  # element molar fractions
)

# Results include:
# - Phase assemblage (which phases present)
# - Phase masses (wt%)
# - Phase compositions (wt% oxides or molar fractions)
```

**Status:** ✅ Infrastructure ready, 🟡 training needed for production use

**To Be Expanded:**
- Detailed API documentation
- Input/output format specifications
- Example prediction workflows
- Comparison with original MELTS predictions
- Uncertainty quantification

---

## Module Structure

```
src/
├── builder/                          # Data processing & model training. Models are made here, nothinig here is necesary for inference.
│   ├── indexer.py                    # Dynamic dataset indexing from CSV headers
│   ├── processing/                   # MELTS data processing pipeline
│   │   ├── BigMetaTable.py           # Parse & manage large MELTS datasets
│   │   ├── filters.py                # Data quality & balancing filters
│   │   ├── MLexporter.py             # Feature/label generation & bundling
│   │   ├── prepareML.py              # Main workflow orchestration
│   │   └── __init__.py
│   ├── training/                     # PyTorch training infrastructure
│   │   ├── loadTrainData.py          # Load ML bundles into DataLoaders
│   │   ├── torchDataClass.py         # PyTorch Dataset wrappers
│   │   ├── tuners.py                 # Training loops & optimization
│   │   └── __init__.py
│   └── __init__.py
│
└── nMELTS/                           # Core, deployable emulator package
    ├── config/                       # Configuration & constants
    │   ├── constants.py              # Phase-component mappings, oxide lists
    │   ├── ml_indexer.py             # ML transformation matrices
    │   ├── settings.py               # Path & directory configuration
    │   └── __init__.py
    ├── engine/                       # Neural network & inference
    │   ├── NN.py                     # Neural network architecture
    │   ├── emulator.py               # High-level emulator interface
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
