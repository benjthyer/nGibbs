# MLIndexer Documentation

The `MLIndexer` class creates ML-ready indexers and transformation matrices from MELTS component specifications. It handles the mapping between MELTS components, oxides, and elements, while managing phase compositions and building projection matrices for neural network training.

## Dimension Notation

Throughout this document, we use the following dimension notation:

- **C** = `ncomps` - Total number of components across all phases
- **P** = `nphases` - Total number of phases
- **VP** = Number of compositionally variable phases (phases with >1 component). VP = len(compositionally_variable_phases)
- **VC** = `ncompsVaried` - Total number of components in compositionally variable phases. This is the composition output dimension for nMELTS
- **E** = Number of elements (length of `Elkeys`)
- **O** = Number of oxides (length of `Oxides`, includes Fe2O3), always O = E + 1

## Constructor Parameters

### `components_in_phases` : Dict[str, List[str]], optional
Dictionary mapping phase names to their component lists. This is created and passed to ml_indexer by DatasetIndexer. 
- **Example**: `{'olivine': ['fayalite', 'forsterite'], 'melts-liquid': ['Si', 'Ti', 'Al', ...]}`

### `Elkeys` : List[str], optional
List of element symbols within the system. This is passed to ml_indexer by DatasetIndexer
- **Default**: Extracted from `default_Elkeys` if not provided
- **Example**: `['Si', 'Ti', 'Al', 'Fe', 'Mg', 'Ca', 'Na', 'K', 'P', 'H', 'Cr', 'Mn', 'Ni']`

---

## Core Attributes

### Phase and Component Indices

#### `label_names` : List[str]
**Dimension**: (C,)  
Ordered list of all component names across all phases.
- Components are ordered by phase (`'melts-liquid'` comes last)
- **Example**: `['fayalite', 'forsterite', 'diopside', 'hedenbergite', ..., 'Si', 'Ti', 'Al', ...]`
- *Test*:
    - len

#### `label_indices` : Dict[str, np.ndarray] 
**Dimension**: Phase → List of indices into (C,)  
Maps each phase name to its component indices in `label_names`.
- **Type**: `np.ndarray` of dtype `int`
- **Example**: `{'olivine': [0, 1], 'clinopyroxene': [2, 3], 'melts-liquid': [50, 51, ..., 62]}`
- *Test*:
    - Sum of all list lengths equals C
    - max value in compiled list = C-1

#### `label_indices_comp` : Dict[str, np.ndarray]
**Dimension**: Phase → List of indices into (VC,)  
Maps phase names to arrays of their component indices in the varied-composition space.
- Only includes compositionally variable phases
- **Type**: `np.ndarray` of dtype `int`
- **Example**: `{'olivine': array([0, 1]), 'clinopyroxene': array([2, 3])}`
- *Test*:
    - Sum of all list lengths equals VC
    - max value in compiled list = VC-1

#### `comp_map` : Dict[str, np.ndarray]
Identical to `label_indices_comp`, kept for backward compatibility
- *Test*:
    - identical to label_indices_comp

#### `detail_label_indices` : Dict[str, Dict[str, int]]
**Dimension**: Phase → Component → index into (VC,)  
Nested dictionary mapping phase and component names to indices in the compositionally-variable component space. 
- Only includes phases with >1 component
- **Example**: `{'olivine': {'fayalite': 0, 'forsterite': 1}, 'clinopyroxene': {'diopside': 2, 'hedenbergite': 3}}`
- *Test*:
    - each nested dictionary contains > 1 entry
    - the amount of 1st-level entries = VP
    - the amount of all nested entries = VC
    - max value in compiled list = VC-1
---

### Phase Organization

#### `all_phases` : List[str]
**Dimension**: (P,)  
Ordered list of all phase names.
- **Example**: `['olivine', 'clinopyroxene', 'plagioclase', 'spinel', 'melts-liquid']`
- *Test*:
    - len

#### `compositionally_variable_phases` : List[str]
**Dimension**: (VP,)  
List of phases with multiple components (solid solutions).
VP = len(compositionally_variable_phases)
- **Example**: `['olivine', 'clinopyroxene', 'plagioclase', 'melts-liquid']`
- Excludes phases like pure `'quartz'` or `'Apatite'`
- *Test*:
    - len = VP

#### `mass_phasedict` : Dict[str, int]
**Dimension**: Phase → index into (P,)  
Maps phase names to their indices in the full phase list. Used by nMELTS for indexing phase abundances
- **Example**: `{'olivine': 0, 'clinopyroxene': 1, 'plagioclase': 2, 'spinel': 3, 'melts-liquid': 4}`
- *Test*:
    - len = P

#### `comp_phasedict` : Dict[str, int]
**Dimension**: Phase → index into (VP,)  
Maps compositionally variable phase names to their indices among variable phases only.
- **Example**: `{'olivine': 0, 'clinopyroxene': 1, 'plagioclase': 2, 'melts-liquid': 3}`
- Only includes phases in `compositionally_variable_phases`
- *Test*:
    - len = VP

---

### Size Counters

#### `ncomps` : int
Total number of components across all phases (C).

#### `ncompsVaried` : int
Total number of components in compositionally variable phases (VC).

#### `nphases` : int
Total number of phases (P).

---

## Transformation Matrices

### Component-Oxide Transformations

#### `compToOxLoad` : np.ndarray
**Dimension**: (C, O)  
**Type**: `float32`  
Raw projection matrix from components to oxides, loaded and filtered from CSV.
- Maps each component to its oxide composition
- Includes Fe2O3 column
- *Test*:
    - dtype
    - dims

#### `PxSpTransform` : np.ndarray
**Dimension**: (C, C)  
**Type**: `float32`  
Component transformation matrix for Px-Sp all-positive remappings.
- Square matrix transforming component space
- Projects components into an all-positive space for ease of use with softmax activation function to predict chemistry. 
- Saved to `projections_dir/PxSp_Comp_Transform_gathered.csv` during initialization
- *Test*:
    - dims
    - dtype
    - Is Invertible
    - The submatrix that excludes spinel (except chromite), cpx, and opx is an identity matrix (diagonal = 1, else 0)

#### `compToOx` : np.ndarray
**Dimension**: (C, O)  
**Type**: `float32`  
Final component-to-oxide projection matrix.
- Computed as: `inv(PxSpTransform) @ compToOxLoad`
- Maps component compositions to oxide weight fractions
- *Test*:
    - dims
    - dtype
    - The submatrix that excludes spinel (except chromite), cpx, and opx is identical to the corresponding submatrix of compToOxLoad

#### `boolTransCompToOx` : np.ndarray or None
**Dimension**: (C, E)  
**Type**: `int` (0 or 1)  
Boolean mask indicating which oxides each component contributes to.
- Derived from `compToOx` by converting non-zero values to 1
- Fe2O3 is folded into FeO column for whole-rock compatibility
- Used for masking and sparsity analysis
- *Test*:
    - dims
    - dtype

### Oxide-Element Transformations

#### `OxToEl` : np.ndarray
**Dimension**: (O, E) = (E+1 , E) 
**Type**: `float32`  
Projection matrix from oxides to elements.
- Maps oxide mole fractions to elemental mole fractions
- Includes Fe2O3 row
- *Test*:
    - dims
    - dtype

#### `ElToOx` : np.ndarray
**Dimension**: (E, E)  
**Type**: `float32`  
Inverse transformation from elements back to oxides.
- Computed as: `inv(OxToEl[:E, :])`
- Maps elemental composition to total iron oxides (FeO equivalent)
- does not map back to Fe2O3, FeO partitioning can be solved for in the liquid given PTfO2
- *Test*:
    - dims
    - dtype


### Molar Mass Matrices

#### `MM` : np.ndarray
**Dimension**: (O, O)  
**Type**: `float32`  
Diagonal matrix of oxide molar masses.
- `MM[i, i]` = molar mass of `Oxides[i]` in g/mol
- Used for mass-to-mole conversions
- *Test*:
    - dims
    - diagonal matrix

#### `Minv` : np.ndarray
**Dimension**: (O, O)  
**Type**: `float32`  
Diagonal matrix of inverse molar masses.
- `Minv[i, i]` = 1 / (molar mass of `Oxides[i]`)
- Used for mole-to-mass conversions


#### `Mtot` : np.ndarray
**Dimension**: (O, 1)  
**Type**: `float32`  
Column vector of oxide molar masses.
- Used for broadcasting operations in mass/mole calculations
- *Test*:
    - dims
    - identical to diag of MM
---

## ML-Ready Mapping Matrices

### Phase-Component Mappings

#### `phaseToCompMap` : np.ndarray
**Dimension**: (P, C)  
**Type**: `float32`  
Binary matrix mapping phases to their components.
- `phaseToCompMap[p, c] = 1.0` if component `c` belongs to phase `p`
- Used to aggregate component masses/compositions to phase level
    - For example: to multiply weights by composition to get an extensive component matrix 
- *Test*:
    - dims
    - columwise sum is all ones
    - dtype

#### `variedToAllComp` : np.ndarray
**Dimension**: (VC, C)  
**Type**: `float32`  
Maps varied-composition component indices to full component indices.
- `variedToAllComp[vc, c] = 1.0` if varied component `vc` corresponds to full component `c`
- Identity-like matrix for subsetting compositionally variable components
    - Used as an indexer to pull chemically variable phase components from full component matrix 
- *Test*:
    - dims
    - rowwise sum is all ones
    - dtype

#### `compositionally_variable_binaries` : np.ndarray
**Dimension**: (P,)  
**Type**: `bool` (0 or 1)  
Binary indicator for compositionally variable phases.
- `compositionally_variable_binaries[p] = 1` if phase `p` has >1 component
- Used for masking and conditional operations, or to take VP subset of P
- *Test*:
    - len
    - dtype
    - sum = VP

#### `compositionally_variable_subset` : np.ndarray
**Dimension**: (VC,)  
**Type**: `int`  
Indices of compositionally variable components in the full component list.
- Selects components belonging to variable-composition phases from (C,) arrays
- **Example**: `[0, 1, 2, 3, 50, 51, 52, ...]` for olivine, cpx, and liquid components
- Used for indexing VC subset of C. 1D alternative to variedToAllComp.T
- *Test*:
    - len
    - dtype

#### `compositional_component_subset` : np.ndarray
**Dimension**: (VC,)  
**Type**: `int`  
Copy of `compositionally_variable_subset` (maintained for backward compatibility).
- *Test*:
    - identical to compositionally_variable_subset

#### `fixed_phaseToCompMap` : np.ndarray
**Dimension**: (1, C)  
**Type**: `float32`  
Aggregated mapping for all fixed-composition phases.
- Row vector where entry `c` is 1.0 if component `c` belongs to any fixed phase
- Computed as: `(~is_variable).reshape(1, -1) @ phaseToCompMap`
- Could be used to take 1D vector of fixed composition phases from a phase ownership matrix of shape (C, P)
- *Test*:
    - dim
    - dtype
    - sum = P-VP


#### `comp_variable_IDMAT` : torch.Tensor
**Dimension**: (P, P)  
**Type**: `torch.float` 
Diagonal identity matrix for compositionally variable phases.
- `comp_variable_IDMAT[p, p] = 1.0` if phase `p` is compositionally variable, else 0
- Used to zero out pure phases
- *Test*:
    - dim
    - dtype
    - sum = VP

---

## Backward-Compatibility Structures

These attributes maintain compatibility with older code:

#### `comp_binaries` : np.ndarray
**Dimension**: (VP,)  
**Type**: `int`  
Phase indices (in `all_phases`) of compositionally variable phases.
- *Test*:
    - dim
    - dtype

#### `comp_mappings` : np.ndarray
**Dimension**: (VP, VC)  
**Type**: `float32`  
Maps compositionally variable phases to their components.
- `comp_mappings[vp, vc] = 1.0` if component `vc` belongs to variable phase `vp`
- *Test*:
    - dim
    - dtype
    - columnwise sums = 1
---

## Element and Oxide Lists

#### `Elkeys` : List[str]
**Dimension**: (E,)  
Element symbols for liquid composition.
E = len(Elkeys)
- **Example**: `['Si', 'Ti', 'Al', 'Fe', 'Mg', 'Ca', 'Na', 'K', 'P', 'H', 'Cr', 'Mn', 'Ni']`

#### `WRkeys` : List[str]
**Dimension**: (E,)  
Whole-rock oxide names (excludes Fe2O3).
- **Example**: `['SiO2', 'TiO2', 'Al2O3', 'FeO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'H2O', 'Cr2O3', 'MnO', 'NiO']`
- *Test*:
    - len = E + 1

#### `Oxides` : List[str]
**Dimension**: (O,)  
All oxide names including Fe2O3.
- **Example**: `['SiO2', 'TiO2', 'Al2O3', 'FeO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'H2O', 'Cr2O3', 'MnO', 'NiO', 'Fe2O3']`
- *Test*:
    - len = E + 1
---

## Usage Example

```python
from src.nMELTS.config.ml_indexer import MLIndexer

# Define components for each phase
components_in_phases = {
    'olivine': ['fayalite', 'forsterite'],
    'clinopyroxene': ['diopside', 'hedenbergite'],
    'melts-liquid': None  # Will use Elkeys
}

# Create indexer
indexer = MLIndexer(
    components_in_phases=components_in_phases,
    Elkeys=['Si', 'Ti', 'Al', 'Fe', 'Mg', 'Ca', 'Na', 'K']
)

# Access attributes
print(f"Total components: {indexer.ncomps}")
print(f"Number of phases: {indexer.nphases}")
print(f"Variable phases: {indexer.compositionally_variable_phases}")

# Use transformation matrices
component_vector = np.random.rand(indexer.ncomps)
oxide_composition = indexer.compToOx.T @ component_vector

# Map phases to components
phase_masses = np.random.rand(indexer.nphases)
component_masses = indexer.phaseToCompMap.T @ phase_masses
```

---

## Key Relationships

1. **Component Counts**:
   - `len(label_names) == ncomps == C`
   - `ncompsVaried ≤ ncomps` (only counts components in variable phases)

2. **Phase Counts**:
   - `len(all_phases) == nphases == P`
   - `len(compositionally_variable_phases) ≤ nphases`

3. **Matrix Shapes**:
   - `phaseToCompMap @ component_vector` → phase aggregation (C,) → (P,)
   - `compToOx.T @ component_vector` → oxide composition (C,) → (O,)
   - `OxToEl.T @ oxide_vector` → element composition (O,) → (E,)

4. **Special Phase Handling**:
   - `'melts-liquid'` always appears last in orderings
   - `'melts-liquid'` components come from `Elkeys`, not `components_in_phases`

---

## Notes

- **Thread Safety**: Not thread-safe; create separate instances for parallel processing
- **PyTorch**: If unavailable, `comp_variable_IDMAT` will be `None`
- **File Dependencies**: Requires projection CSVs in `projections_dir`
- **Immutability**: Attributes should be treated as read-only after initialization
