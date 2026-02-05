# WSL MELTS Data Processing

This directory contains scripts to execute MELTS simulations through alphamelts (v2.3.1), as well as
to gather the generated data into tables (.csv files) and corresponding metadata (.txt files).

## Directory Structure

```
processing/
├── DataProducts/         # .npy binary files and .txt headers
│   ├── 102/              # MELTS 1.0.2 data
│   │   ├── Plots/        # Generated visualization plots (Harker diagrams, T distributions)
│   │   └── *.npy         # Binary data files (features, labels, molar abundances, etc.)
│   ├── 110/              # MELTS 1.1.0 data 
│   └── ...
└── README.md             # This file
```

## Data Files

Each MELTS model version directory contains the following processed datasets:

### Training Data
- `MELTS{Version}_Trainset{Date}{Mode}*molar_labels.npy` - Molar abundances of phases
- `MELTS{Version}_Trainset{Date}{Mode}*binary_labels.npy` - Phase presence/absence flags
- `MELTS{Version}_Trainset{Date}{Mode}*mass_labels.npy` - Phase masses (normalized to 100g), used for validation only
- `MELTS{Version}_Trainset{Date}{Mode}*features.npy` - Input features (P, T, fO2, bulk composition)
- `MELTS{Version}_Trainset{Date}{Mode}*labels.npy` - Intensive compositional labels
- `MELTS{Version}_Trainset{Date}{Mode}*free_outputs.npy` - Optional outputs not bounded by mass conservation (if available)

### Validation & Test Data
Same naming convention as training data, with `Validset` or `Testset` instead of `Trainset`.
- Validation data is not used at all during training
- Test data is used to evaluate model generalization during training. Model performance is estimated against inference to this dataset

## Processing Pipeline

Data is processed through the following stages:

1. **Filtering** - Remove simulations that produce MELTS idiosynchrasies or otherwise improbable compositions
2. **Resampling** - Augment rare phase abundance and/or by melt fraction to balance dataset
3. **Feature Engineering** - Convert component masses to element moles (bulk chemistry)
4. **ML Preparation** - Create memmaps with normalized features and labels
5. **Quality Control** - Verify bulk reconstruction (forward + backward mass balance)
6. **Output** - Save standardized `.npy` files for training

## Usage

### Processing New Data

MELTS tables and metadata are pulled from wslMELTS/DataProducts, and the generated ML ready memmaps are moved to processing/DataProducts.

```python
from src.builder.processing.prepareML import process_for_ML

process_for_ML(
    MELTSModel='102',
    Date='Jan22',
    Mode='BatchCooling',
    upsample=True,
    preprocessed=False,
    subset=False,
    use_external=False,
    balance_function=None
)
```

### Loading Processed Data

```python
import numpy as np

# Load features (P, T, fO2, bulk chemistry in element moles)
features = np.load('DataProducts/102/MELTS102_Trainset...features.npy')

# Load labels (phase compositions)
labels = np.load('DataProducts/102/MELTS102_Trainset...labels.npy')

# Load binary labels (phase presence)
binaries = np.load('DataProducts/102/MELTS102_Trainset...binary_labels.npy')

# Load molar abundances
molars = np.load('DataProducts/102/MELTS102_Trainset...molar_labels.npy')
```

## Key Concepts

### Features ()
- **Pressure** (bars)
- **Temperature** (°C) | or **Enthalpy per 100 g** (kJ) (not implemented as of v0.0.0)
- **logfO2-QFM** (oxygen fugacity relative to Quartz-Fayalite-Magnetite buffer) | or **Fe3+ mol fraction** (not implemented as of v0.0.0)
- **Bulk Composition** (element moles normalized to sum = 1)
  - Elements: Si, Ti, Al, Fe, Mg, Ca, Na, K, P, H, Cr, Mn, Ni (as applicable)

### Labels
- **Phase Compositions** (component mole fractions for compositionally variable phases)
- **Phase Abundances** (molar and mass amounts)
- **Binary Indicators** (presence/absence of each phase)

### Compositionally Variable Phases
Some phases vary in composition across conditions. For example:
- **Olivine** (Fo-Fa solid solution)
- **Pyroxenes** (orthopyroxene, clinopyroxene)
- **Plagioclase** (An-Ab solid solution)
- **Feldspar** (plagioclase, k-feldspar)
- **Spinel** (Mg-Fe-Al oxide solid solution)
- **Liquid/Melt** (silicate melt composition)
- etc.

### Fixed Phases
Single-composition/component, pure phases. Mole fraction and Phase presence are only outputs. For example:
- **Quartz**, **Tridymite**, **Cristobalite**
- **Apatite**, **Whitlockite**
- **Fluid** (water)

## Bulk Reconstruction

Quality control includes verification of mass balance. Also verification of projection tensors

```
Input bulk composition (oxides from elements) ≈ 
  Reconstructed bulk composition (from phase abundances + compositions)
```

Samples failing this check (threshold ±0.01 wt%) are flagged and can be filtered. See `tests/test_bulk_reconstruction.py` for diagnostics.

## Troubleshooting

### All samples of a phase missing after filtering
Check `filter_legal()` and `filter_full_metadata()` - they may be too aggressive for your data, or . Use the phase abundance diagnostic output in `process_for_ML()` to see what survives filtering.

### Bulk reconstruction failures
Run the diagnostic script:
```bash
python tests/test_bulk_reconstruction.py --file path/to/melts/data --samples 10
```

This shows which oxides are mismatching and which phases are common in failures.

### Resampling not detecting phases
Even if phase abundances are non-zero, `resample_rare_phase()` may not detect them if:
- Values are extremely small (< threshold)
- Data contains NaN or inf values
- The method has internal abundance requirements

Check phase statistics before and after resampling in debug output.

## File Management

The processing pipeline includes automatic cleanup:
- **Temporary files** created during processing are removed after successful completion
- A baseline snapshot is taken at start and only new files are removed on exit
- Memory maps are properly closed in finally blocks to prevent resource leaks

## References

- **MELTS**: Ghiorso & Sack (1995), Asimow & Ghiorso (1998)
- **nMELTS**: Antoshechkina & Ghiorso (2014)
- **Machine Learning Integration**: Custom pipeline in `src/builder/processing/`

## Contact

For questions about data processing or this directory, see the main repository README.
