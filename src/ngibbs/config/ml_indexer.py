"""
ML Indexer for nMELTS.

Takes a DatasetIndexer and creates ML-ready indexers and transformation matrices.
Transformation matrices are built dynamically based on the components in label_names.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
import json

from .constants import default_Elkeys, all_Elkeys, get_oxide_molar_mass
from ..utils.math_utils import Normalizer

# Make torch optional for alphamelts in WSL
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


def _load_element_oxide_mappings() -> tuple:
    """Derive element<->oxide mappings from OxToElV2_ferric.csv (diagonal matrix)."""
    csv_path = Path(__file__).parent / 'projections' / 'OxToElV2_ferric.csv'
    df = pd.read_csv(csv_path, index_col=0)
    oxides = list(df.index)
    elements = list(df.columns)
    element_to_oxide = {elements[i]: oxides[i] for i in range(len(elements))}
    oxide_to_element = {oxides[i]: elements[i] for i in range(len(oxides))}
    return element_to_oxide, oxide_to_element


ELEMENT_TO_OXIDE, OXIDE_TO_ELEMENT = _load_element_oxide_mappings()


class MLIndexer:
    """
    ML-ready indexer that builds transformation matrices and mappings from either:
    - a DatasetIndexer instance, or
    - a `components_in_phases` dictionary (phase -> list of components)

    The class builds label indices, phase dictionaries, phase->component matrices,
    loads projection matrices if available, and exposes ML-ready mappings.
    """

    def __init__(self,
                 components_in_phases: Optional[Dict[str, List[str]]] = None,
                 Elkeys: Optional[List[str]] = None,
                 projections_dir: Optional[Path] = None,
                 Model: str = 'MELTS'):

        # Store provided configuration
        self.components_in_phases = components_in_phases or {}
        self.Elkeys = Elkeys or []

        # Calculate basic containers to be filled by builder
        self.label_indices: Dict[str, np.ndarray] = {}
        self.label_names: List[str] = []
        self.detail_label_indices: Dict[str, Dict[str, int]] = {}
        self.label_indices_comp: Dict[str, np.ndarray] = {}
        self.all_phases: List[str] = []
        self.mass_phasedict: Dict[str, int] = {}
        self.comp_phasedict: Dict[str, int] = {}
        self.compositionally_variable_phases: List[str] = []
        self.ncomps: int = 0
        self.ncompsVaried: int = 0
        self.nphases: int = 0
        self.do_bulk: bool = True # Tells nMELTS object whether to reconstruct bulk calculations or not
        self.Model = Model
        # Feature normalizers (to be set by data loading pipeline)
        self.feature_normalizer: Optional[Any] = None  # Normalizer for input features
        self.output_normalizer: Optional[Any] = None   # Normalizer for free outputs (if applicable)

        # Set up projections directory
        if projections_dir is None:
            _CONFIG_DIR = Path(__file__).parent
            projections_dir = _CONFIG_DIR / 'projections'
        self.projections_dir = Path(projections_dir)

        # Run builders
        self._extract_elements_and_oxides()
        self._build_label_indices_from_components()
        self._load_transformation_matrices()

        self._build_phase_dictionaries()
        self._build_phase_mappings()
        # Extract elements and oxides from label_names (for melts-liquid)
        # Load and build transformation matrices
        # Build ML-ready mappings
        self._build_ml_mappings()
        
    def _build_label_indices_from_components(self):
        """Generate label indices from `self.components_in_phases` and `self.Elkeys`."""
        detail_ind = 0
        index = 0

        for phase, chemical_components in self.components_in_phases.items():
            # For melts-liquid, use Elkeys as components
            if phase == 'melts-liquid':
                chemical_components = self.Elkeys
            
            phase_inds = []
            is_detailed = len(chemical_components) > 1

            if is_detailed:
                self.detail_label_indices[phase] = {}
                comp_inds = []

                for component in chemical_components:
                    phase_inds.append(index)
                    self.detail_label_indices[phase][component] = detail_ind
                    comp_inds.append(detail_ind)
                    detail_ind += 1
                    index += 1
                    self.label_names.append(component)

                self.label_indices_comp[phase] = np.array(comp_inds, dtype=int)
            else:
                phase_inds.append(index)
                index += 1
                self.label_names.append(chemical_components[0])

            self.label_indices[phase] = np.array(phase_inds, dtype=int)
        
        # Set component counts directly from loop counters
        self.ncomps = index
        self.ncompsVaried = detail_ind
        self.all_phases = list(self.label_indices.keys())

    def _build_phase_dictionaries(self):
        """Build phase dictionaries and identify compositionally variable phases."""
        cj = 0
        for i, phase in enumerate(self.label_indices.keys()):
            #self.all_phases.append(phase)
            self.mass_phasedict[phase] = i
            if len(self.label_indices[phase]) > 1:
                self.compositionally_variable_phases.append(phase)
                self.comp_phasedict[phase] = cj
                cj += 1

        self.nphases = len(self.all_phases)

    def _build_phase_mappings(self):
        """Build phase-to-component and related mapping matrices."""
        if self.nphases == 0 or self.ncomps == 0:
            self.phaseToCompMap = np.zeros((0, 0), dtype=np.float32)
            self.variedToAllComp = np.zeros((0, 0), dtype=np.float32)
            self.compositionally_variable_binaries = np.array([], dtype=int)
            self.compositionally_variable_subset = np.array([], dtype=int)
            self.compositional_component_subset = np.array([], dtype=int)
            self.fixed_phaseToCompMap = np.zeros((1, 0), dtype=np.float32)
            return

        self.phaseToCompMap = np.zeros((self.nphases, self.ncomps), dtype=np.float32)
        compositionally_variable_subset_list = []

        if self.ncompsVaried == 0:
            max_varied = 0
            for comp_inds in self.label_indices_comp.values():
                if len(comp_inds) > 0:
                    max_varied = max(max_varied, max(comp_inds))
            self.ncompsVaried = max_varied + 1 if max_varied > 0 else 0

        if self.ncompsVaried > 0:
            self.variedToAllComp = np.zeros((self.ncompsVaried, self.ncomps), dtype=np.float32)
        else:
            self.variedToAllComp = np.zeros((0, self.ncomps), dtype=np.float32)

        for p, phase in enumerate(self.all_phases):
            if phase in self.label_indices:
                phase_inds = self.label_indices[phase]
                self.phaseToCompMap[p, phase_inds] = 1.0

                if phase in self.compositionally_variable_phases:
                    compositionally_variable_subset_list.extend(phase_inds)
                    if phase in self.label_indices_comp:
                        comp_inds = self.label_indices_comp[phase]
                        label_inds = self.label_indices[phase]
                        for varied_idx, label_idx in zip(comp_inds, label_inds):
                            if varied_idx < self.ncompsVaried and label_idx < self.ncomps:
                                self.variedToAllComp[varied_idx, label_idx] = 1.0

        self.compositionally_variable_binaries = np.array([1 if ph in self.compositionally_variable_phases else 0 for ph in self.all_phases], dtype=int)
        self.compositionally_variable_subset = np.array(compositionally_variable_subset_list, dtype=int)
        self.compositional_component_subset = np.copy(self.compositionally_variable_subset)

        if TORCH_AVAILABLE and len(self.compositionally_variable_binaries) > 0:
            self.comp_variable_IDMAT = torch.tensor(np.diag(self.compositionally_variable_binaries), dtype=torch.float)
        else:
            self.comp_variable_IDMAT = None

        is_fixed = ~(self.compositionally_variable_binaries.astype(bool))
        self.fixed_phaseToCompMap = (is_fixed.reshape(1, -1) @ self.phaseToCompMap).astype(np.float32)

        # Build component mappings for backward compatibility
        self._build_component_mappings()
    
    def _extract_elements_and_oxides(self):
        """Extract Elkeys and WRkeys from label_names (melts-liquid components)."""
        # Infer Elkeys if not explicitly passed
        """if not len(self.Elkeys) and 'melts-liquid' in self.label_indices:
            melts_start = self.label_indices['melts-liquid'][0]
            melts_end = self.label_indices['melts-liquid'][-1] + 1
            self.Elkeys = self.label_names[melts_start:melts_end]
        else:
            self.Elkeys = default_Elkeys.copy()"""

        # Build WRkeys (oxides without Fe2O3) deterministically from Elkeys
        self.WRkeys = []
        for el in self.Elkeys:
            if el in ELEMENT_TO_OXIDE:
                ox = ELEMENT_TO_OXIDE[el]
                if ox not in self.WRkeys:
                    self.WRkeys.append(ox)

        # Build Oxides (include Fe2O3 if Fe present)
        self.Oxides = self.WRkeys.copy()
        if 'Fe' in self.Elkeys and 'Fe2O3' not in self.Oxides:
            self.Oxides.append('Fe2O3')
    
    def _load_transformation_matrices(self):
        """Load transformation CSV files and build matrices based on label_names."""
    
        # Load compToOx.csv (components to oxides)
        compToOx_path = self.projections_dir / 'compToOxV2.csv'
        compToOx_df = pd.read_csv(compToOx_path, index_col=0)
        
        # Load PxSp_Comp_Transform.csv (component transformation matrix)
        if self.Model == 'MELTS':
            PxSpTransform_path = self.projections_dir / 'PxSp_Comp_TransformV2.csv'
            PxSpTransform_df = pd.read_csv(PxSpTransform_path, index_col=0)
        

        
        # The assumptions that the components in rows of compToOx and PxSpTransform square matric are matching is baked into the construction of these projection matrices
        #assert compToOx_df.shape[0] == PxSpTransform_df.shape[0], f"Component count mismatch between compToOx rows:{compToOx_df.shape[0]} and PxSp_Comp_Transform matrices:{PxSpTransform_df.shape}."
        #assert compToOx_df.shape[0] == PxSpTransform_df.shape[1], f"Component count mismatch between compToOx rows:{compToOx_df.shape[0]} and PxSp_Comp_Transform matrices:{PxSpTransform_df.shape}."

        # Load OxToEl.csv (oxides to elements)
        # print(f"ELKEYS: {self.Elkeys}")
        if 'Fe3' in self.Elkeys:
            oxToEl_path = self.projections_dir / 'OxToElV2_ferric.csv'
            # print("Loading OxToElV2_ferric.csv with Fe2O3 -> Fe3+ mapping for closed system oxygen")
        else:
            oxToEl_path = self.projections_dir / 'OxToElV2.csv'
        oxToEl_df = pd.read_csv(oxToEl_path, index_col=0)

        # Match CSV rows to our phase components
        compMappings = {idx: i for i, idx in enumerate(compToOx_df.index)}

        row_idx = []
        for phase_name, IDXes in self.label_indices.items():
            for comp_name in [self.label_names[i] for i in IDXes]:
                if comp_name in compMappings:
                    row_idx.append(compMappings[comp_name])
                else:
                    partname = f"{comp_name} : {phase_name}" # Some components are named with phase. We actually need the right order to avoid doubling repeated components
                    if partname in compMappings:
                        row_idx.append(compMappings[partname])
                    else:
                        raise ValueError(f"Component {comp_name} or {partname} not found in compToOx projection matrix.")

        compToOxLoad_data = np.array(compToOx_df.values)[row_idx, :].astype(np.float32)
        oxide_indices = np.array([compToOx_df.columns.get_loc(col) for col in self.Oxides])

        self.compToOxLoad = compToOxLoad_data[:, oxide_indices]
        if np.isnan(self.compToOxLoad).any():
            nan_count = int(np.isnan(self.compToOxLoad).sum())
            raise ValueError(f"Warning: compToOxLoad contains {nan_count} NaN values. ")
        if self.Model == 'MELTS':
            self.PxSpTransform = np.array(PxSpTransform_df.values)[np.ix_(row_idx, row_idx)].astype(np.float32)
        else:
            self.PxSpTransform = np.diag(np.ones(len(row_idx), dtype=np.float32)) # Identity if no transformation provided for non-MELTS models

        self.compToOx = np.linalg.inv(self.PxSpTransform) @ self.compToOxLoad
        
        # Build oxToEl matrix: rows=Oxides, columns=Elkeys
        # First, get column indices for elements
        el_idx = [] 
        for el in self.Elkeys:
            if el in oxToEl_df.columns:
                el_idx.append(oxToEl_df.columns.get_loc(el))
            else:
                raise ValueError(f"Element {el} not found in oxToEl projection matrix.")
        
        # Second, get row indices for oxides (which rows to select)
        ox_idx = []
        for ox in self.Oxides:
            if ox in oxToEl_df.index:
                ox_idx.append(oxToEl_df.index.get_loc(ox))
            else:
                raise ValueError(f"Oxide {ox} not found in oxToEl projection matrix rows.")
        
        # Now select rows (oxides) and columns (elements)
        self.OxToEl = np.array(oxToEl_df.values)[np.ix_(ox_idx, el_idx)].astype(np.float32)
        if np.isnan(self.OxToEl).any():
            nan_count = int(np.isnan(self.OxToEl).sum())
            raise ValueError(f"Warning: OxToEl contains {nan_count} NaN values. ")
        self.ElToOx = np.linalg.inv(self.OxToEl[:len(self.Elkeys)]) # Can map back to oxides, but only total iron. 

        # Build molar mass matrices
        molar_masses = [get_oxide_molar_mass(ox) for ox in self.Oxides]
        self.MM = np.diag(molar_masses).astype(np.float32)
        self.Minv = np.diag([1.0 / mass for mass in molar_masses]).astype(np.float32)
        self.Mtot = np.array(molar_masses, dtype=np.float32).reshape(-1, 1)
        
    
    def _build_ml_mappings(self):
        """Build ML-ready phase-to-component mappings."""
        # Build phaseToCompMap: (nphases, ncomps)
        self.phaseToCompMap = np.zeros((self.nphases, self.ncomps), dtype=np.float32)
        
        for p, phase in enumerate(self.all_phases):
            if phase in self.label_indices:
                phase_inds = self.label_indices[phase]
                self.phaseToCompMap[p, phase_inds] = 1.0
        
        # Build variedToAllComp: (ncompsVaried, ncomps)
        self.variedToAllComp = np.zeros((self.ncompsVaried, self.ncomps), dtype=np.float32)
        
        for phase in self.compositionally_variable_phases:
            if phase in self.label_indices_comp and phase in self.label_indices:
                comp_inds = self.label_indices_comp[phase]
                label_inds = self.label_indices[phase]
                self.variedToAllComp[comp_inds, label_inds] = 1.0
        
        # Build compositionally_variable_binaries
        self.compositionally_variable_binaries = []
        self.compositionally_variable_subset = []
        
        for phase in self.all_phases:
            if phase in self.compositionally_variable_phases:
                self.compositionally_variable_binaries.append(1)
                if phase in self.label_indices:
                    self.compositionally_variable_subset.extend(self.label_indices[phase])
            else:
                self.compositionally_variable_binaries.append(0)
        
        self.compositionally_variable_binaries = np.array(self.compositionally_variable_binaries, dtype=int)
        self.compositionally_variable_subset = np.array(self.compositionally_variable_subset, dtype=int)
        self.compositional_component_subset = np.copy(self.compositionally_variable_subset)
        
        # Build comp_variable_IDMAT (torch tensor if available)
        if TORCH_AVAILABLE:
            self.comp_variable_IDMAT = torch.tensor(
                np.diag(self.compositionally_variable_binaries),
                dtype=torch.float
            )
        else:
            self.comp_variable_IDMAT = None
        
        # Build fixed_phaseToCompMap
        is_fixed = ~(self.compositionally_variable_binaries.astype(bool))
        self.fixed_phaseToCompMap = (is_fixed.reshape(1, -1) @ self.phaseToCompMap).astype(np.float32)
        
        # Build component mappings (for backward compatibility)
        self._build_component_mappings()
    
    def _build_component_mappings(self):
        """Build component mapping structures (comp_map, comp_binaries, comp_mappings)."""
        
        self.comp_map = self.label_indices_comp  # For backward compatibility

        comp_binaries_list = []
        comp_mappings_list = []
        j = 0
        
        for phase in self.all_phases:
            if phase in self.label_indices_comp:
                n_components = len(self.label_indices[phase])
                comp_binaries_list.append(self.all_phases.index(phase))
                comp_mappings_list.extend([j] * n_components)
                j += 1
        
        self.comp_binaries = np.array(comp_binaries_list, dtype=int) if comp_binaries_list else np.array([], dtype=int)
        self.comp_mappings = np.zeros((j, len(comp_mappings_list)), dtype=np.float32) if comp_mappings_list else np.zeros((0, 0), dtype=np.float32)
        
        for col, row in enumerate(comp_mappings_list):
            self.comp_mappings[row, col] = 1.0
        
    # Build boolTransCompToOx 
        self.boolTransCompToOx = np.copy(self.compToOx)
        # Add Fe2O3 column to FeO if Oxygen is buffered
        if 'Fe2O3' in self.Oxides and 'Fe3' not in self.Elkeys:
            feo_idx = self.WRkeys.index('FeO')
            fe2o3_idx = self.Oxides.index('Fe2O3')
            self.boolTransCompToOx[:, feo_idx] += self.boolTransCompToOx[:, fe2o3_idx]
        # Convert to boolean (non-zero -> 1)
        self.boolTransCompToOx = (self.boolTransCompToOx != 0).astype(int)

    def _rebuild_vc_structures(self) -> None:
        """
        Rebuild all VC (variable chemistry) dependent attributes after filtering phases.
        
        This is called internally after compositionally_variable_phases has been filtered.
        It rebuilds from scratch all attributes with VC dimensions:
        - detail_label_indices: Cleared and rebuilt with only kept phases
        - label_indices_comp: Re-enumerated to be contiguous starting from 0
        - ncompsVaried: Total count of components in restricted variable phases
        - comp_phasedict: Rebuilt with new VP ordering
        - variedToAllComp: Rebuilt (VC, C) mapping
        - compositionally_variable_subset: Rebuilt (VC,) indices
        - All other phase and component mappings via _build_phase_mappings()
        
        This method should be called whenever compositionally_variable_phases is modified.
        """
        # Step 1: Completely rebuild detail_label_indices and label_indices_comp from scratch
        # This ensures no old removed phases remain
        new_detail_label_indices = {}
        new_label_indices_comp = {}
        new_vc_index = 0
        
        for phase in self.compositionally_variable_phases:
            if phase in self.label_indices:
                # Get component names for this phase from label_names
                phase_component_indices = self.label_indices[phase]
                component_names = [self.label_names[idx] for idx in phase_component_indices]
                
                # Build new detail_label_indices for this phase
                new_detail_label_indices[phase] = {}
                new_comp_inds = []
                
                for comp_name in component_names:
                    new_detail_label_indices[phase][comp_name] = new_vc_index
                    new_comp_inds.append(new_vc_index)
                    new_vc_index += 1
                
                # Store new VC indices for this phase
                new_label_indices_comp[phase] = np.array(new_comp_inds, dtype=int)
        
        # Replace old structures with new ones
        self.detail_label_indices = new_detail_label_indices
        self.label_indices_comp = new_label_indices_comp
        self.comp_map = new_label_indices_comp # Identical object for backwards compatibility
        self.ncompsVaried = new_vc_index
        
        # Step 2: Rebuild comp_phasedict (VP ordering)
        self.comp_phasedict = {}
        for vp_idx, phase in enumerate(self.compositionally_variable_phases):
            self.comp_phasedict[phase] = vp_idx
        
        # Step 3: Rebuild variedToAllComp (VC, C)
        if self.ncompsVaried > 0:
            self.variedToAllComp = np.zeros((self.ncompsVaried, self.ncomps), dtype=np.float32)
            for phase in self.compositionally_variable_phases:
                if phase in self.label_indices_comp and phase in self.label_indices:
                    comp_inds = self.label_indices_comp[phase]
                    label_inds = self.label_indices[phase]
                    self.variedToAllComp[comp_inds, label_inds] = 1.0
        else:
            self.variedToAllComp = np.zeros((0, self.ncomps), dtype=np.float32)
        
        # Step 4: Rebuild compositionally_variable_subset (VC,) 
        compositionally_variable_subset_list = []
        for phase in self.compositionally_variable_phases:
            if phase in self.label_indices:
                compositionally_variable_subset_list.extend(self.label_indices[phase])
        self.compositionally_variable_subset = np.array(compositionally_variable_subset_list, dtype=int)
        self.compositional_component_subset = np.copy(self.compositionally_variable_subset)
        
        # Step 5: Rebuild all other phase mapping structures (calls _build_component_mappings)
        self._build_phase_mappings()

    def restrictVC(self, phase_names: List[str]) -> None:
        """
        Restrict variable chemistry (VC) structures to only include the specified phases.
        
        This method filters compositionally_variable_phases to only include phases
        in the provided list and rebuilds all related VC structures.
        
        Parameters
        ----------
        phase_names : List[str]
            List of phase names to keep in the variable chemistry structures.
            Only phases that are both in this list AND in compositionally_variable_phases
            will be retained.
        
        Raises
        ------
        ValueError
            If the provided phase_names list is empty or contains phases not in all_phases.

        Notes
        -----
        If this is done to an ml_indexer within a DatasetIndexer, the DatasetIndexer must be reexposed to the MLIndexer
        after this operation to ensure consistency. This is never done as of 1/28/26.
        """
        self.do_bulk = False # Disable bulk calculations when restricting VC, we no longer have enough information

        if not phase_names:
            raise ValueError("phase_names list cannot be empty")
        
        # Validate that all provided phases exist in all_phases
        invalid_phases = set(phase_names) - set(self.all_phases)
        if invalid_phases:
            raise ValueError(f"Invalid phases: {invalid_phases}. Must be subset of {self.all_phases}")
        
        # Filter compositionally_variable_phases to only include requested phases
        self.compositionally_variable_phases = [
            phase for phase in self.compositionally_variable_phases 
            if phase in phase_names
        ]
        
        # Rebuild all VC-dependent structures with proper renumbering
        self._rebuild_vc_structures()

    def save(self, directory: str) -> None:
        """
        Save MLIndexer state to directory without rebuilding from CSVs.
        
        Exports all critical state in JSON and numpy formats that allows
        reconstruction via load_ml_indexer_from_state() without calling __init__
        or requiring access to original CSV projection matrices.
        
        Parameters
        ----------
        directory : str or Path
            Directory path where state files will be saved. Created if not exists.
        
        Notes
        -----
        Files created:
        - indexer_metadata.json: Configuration (components_in_phases, Elkeys, do_bulk, featureNames, freeOutputs)
        - indexer_structure.json: All dictionary mappings (indices, phases, detail mappings)
        - indexer_arrays.npz: All numpy arrays (indices, matrices, mappings, normalizer state)
        
        Normalizers (feature_normalizer, output_normalizer) are saved as state dicts in the NPZ file
        with keys 'normalizer_feature_state' and 'normalizer_output_state'. These contain only
        the 1D min and range arrays needed to reconstruct Normalizer objects via from_state_dict().
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # === PART 1: Metadata (JSON) ===
        metadata = {
            'components_in_phases': self.components_in_phases,
            'Elkeys': self.Elkeys,
            'Oxides': self.Oxides,
            'WRkeys': self.WRkeys,
            'do_bulk': bool(self.do_bulk),
            'ncomps': int(self.ncomps),
            'ncompsVaried': int(self.ncompsVaried),
            'nphases': int(self.nphases),
            'label_names': self.label_names,
            'all_phases': self.all_phases,
            'compositionally_variable_phases': self.compositionally_variable_phases,
            'featureNames': getattr(self, 'featureNames', None),  # May not exist
            'freeOutputs': getattr(self, 'freeOutputs', None),     # May not exist
            'has_feature_normalizer': self.feature_normalizer is not None,
            'has_output_normalizer': self.output_normalizer is not None,
            'molar_epsilon': getattr(self, 'molar_epsilon', None),
        }
        
        with open(directory / 'indexer_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # === PART 2: Structure dictionaries (JSON) ===
        # Convert dictionaries with numpy arrays to serializable format
        structure_dict = {
            'label_indices': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                            for k, v in self.label_indices.items()},
            'label_indices_comp': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                  for k, v in self.label_indices_comp.items()},
            'mass_phasedict': self.mass_phasedict,
            'comp_phasedict': self.comp_phasedict,
            'detail_label_indices': {
                phase: {comp: int(idx) for comp, idx in comp_dict.items()}
                for phase, comp_dict in self.detail_label_indices.items()
            },
            'compositionally_variable_binaries': self.compositionally_variable_binaries.tolist() 
                if isinstance(self.compositionally_variable_binaries, np.ndarray) else self.compositionally_variable_binaries,
        }
        
        with open(directory / 'indexer_structure.json', 'w') as f:
            json.dump(structure_dict, f, indent=2)
        
        # === PART 3: Large numpy arrays (NPZ) ===
        # Store all matrices efficiently + normalizer states
        arrays_dict = {
            'compToOxLoad': self.compToOxLoad,
            'PxSpTransform': self.PxSpTransform,
            'compToOx': self.compToOx,
            'boolTransCompToOx': self.boolTransCompToOx if self.boolTransCompToOx is not None else np.array([]),
            'OxToEl': self.OxToEl,
            'ElToOx': self.ElToOx,
            'MM': self.MM,
            'Minv': self.Minv,
            'Mtot': self.Mtot,
            'phaseToCompMap': self.phaseToCompMap,
            'variedToAllComp': self.variedToAllComp,
            'fixed_phaseToCompMap': self.fixed_phaseToCompMap,
            'compositionally_variable_subset': self.compositionally_variable_subset,
            'compositional_component_subset': self.compositional_component_subset,
            'comp_binaries': self.comp_binaries,
            'comp_mappings': self.comp_mappings,
        }
        
        # Add normalizer states (only 1D min/range arrays, no objects)
        if self.feature_normalizer is not None:
            feature_state = self.feature_normalizer.to_state_dict()
            arrays_dict['normalizer_feature_min'] = feature_state['min']
            arrays_dict['normalizer_feature_range'] = feature_state['range']
        
        if self.output_normalizer is not None:
            output_state = self.output_normalizer.to_state_dict()
            arrays_dict['normalizer_output_min'] = output_state['min']
            arrays_dict['normalizer_output_range'] = output_state['range']
        
        np.savez_compressed(directory / 'indexer_arrays.npz', **arrays_dict)


def load_ml_indexer_from_state(directory: str) -> 'MLIndexer':
    """
    Load MLIndexer instance from saved state directory without calling __init__.
    
    Reconstructs a complete MLIndexer instance from files saved by MLIndexer.save().
    This bypasses __init__ entirely, avoiding CSV projection matrix reloading and
    expensive rebuilds. Useful for deployment and checkpointing.
    
    Parameters
    ----------
    directory : str or Path
        Directory containing the saved indexer state files.
        Expected files:
        - indexer_metadata.json: Configuration and metadata
        - indexer_structure.json: Dictionary mappings
        - indexer_arrays.npz: Large numpy arrays (matrices, mappings)
        - indexer_normalizers.pkl: Normalizers and torch tensors
    
    Returns
    -------
    MLIndexer
        Fully reconstructed MLIndexer instance with all attributes populated.
    
    Raises
    ------
    FileNotFoundError
        If required state files are missing from directory.
    """
    directory = Path(directory)
    
    # === LOAD PART 1: Metadata (JSON) ===
    with open(directory / 'indexer_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # === LOAD PART 2: Structure (JSON) ===
    with open(directory / 'indexer_structure.json', 'r') as f:
        structure_dict = json.load(f)
    
    # === LOAD PART 3: Arrays (NPZ) ===
    arrays_data = np.load(directory / 'indexer_arrays.npz', allow_pickle=True)
    arrays_dict = {key: arrays_data[key] for key in arrays_data.files}
    
    # === CREATE BLANK INDEXER INSTANCE ===
    # Create instance without running __init__ to avoid CSV loading
    indexer = object.__new__(MLIndexer)
    
    # === RESTORE METADATA ===
    indexer.components_in_phases = metadata['components_in_phases']
    indexer.Elkeys = metadata['Elkeys']
    indexer.Oxides = metadata['Oxides']
    indexer.WRkeys = metadata['WRkeys']
    indexer.do_bulk = metadata['do_bulk']
    indexer.ncomps = metadata['ncomps']
    indexer.ncompsVaried = metadata['ncompsVaried']
    indexer.nphases = metadata['nphases']
    indexer.label_names = metadata['label_names']
    indexer.all_phases = metadata['all_phases']
    indexer.compositionally_variable_phases = metadata['compositionally_variable_phases']
    indexer.projections_dir = Path(__file__).parent / 'projections'  # Set default; may not be used
    indexer.molar_epsilon = metadata.get('molar_epsilon', None) # May not exist before training.
    
    # Restore optional attributes
    if metadata.get('featureNames') is not None:
        indexer.featureNames = metadata['featureNames']
    if metadata.get('freeOutputs') is not None:
        indexer.freeOutputs = metadata['freeOutputs']
    
    # === RESTORE STRUCTURE DICTIONARIES ===
    # Convert lists back to numpy arrays
    indexer.label_indices = {
        k: np.array(v, dtype=int) 
        for k, v in structure_dict['label_indices'].items()
    }
    indexer.label_indices_comp = {
        k: np.array(v, dtype=int) 
        for k, v in structure_dict['label_indices_comp'].items()
    }
    indexer.mass_phasedict = structure_dict['mass_phasedict']
    indexer.comp_phasedict = structure_dict['comp_phasedict']
    
    # Restore detail_label_indices
    indexer.detail_label_indices = {
        phase: {comp: int(idx) for comp, idx in comp_dict.items()}
        for phase, comp_dict in structure_dict['detail_label_indices'].items()
    }
    
    # Restore binary array
    indexer.compositionally_variable_binaries = np.array(
        structure_dict['compositionally_variable_binaries'], dtype=int
    )
    
    # === RESTORE ARRAYS ===
    indexer.compToOxLoad = arrays_dict['compToOxLoad'].astype(np.float32)
    indexer.PxSpTransform = arrays_dict['PxSpTransform'].astype(np.float32)
    indexer.compToOx = arrays_dict['compToOx'].astype(np.float32)
    
    # Handle boolTransCompToOx which might be empty
    bool_array = arrays_dict['boolTransCompToOx']
    indexer.boolTransCompToOx = bool_array.astype(int) if bool_array.size > 0 else None
    
    indexer.OxToEl = arrays_dict['OxToEl'].astype(np.float32)
    indexer.ElToOx = arrays_dict['ElToOx'].astype(np.float32)
    indexer.MM = arrays_dict['MM'].astype(np.float32)
    indexer.Minv = arrays_dict['Minv'].astype(np.float32)
    indexer.Mtot = arrays_dict['Mtot'].astype(np.float32)
    indexer.phaseToCompMap = arrays_dict['phaseToCompMap'].astype(np.float32)
    indexer.variedToAllComp = arrays_dict['variedToAllComp'].astype(np.float32)
    indexer.fixed_phaseToCompMap = arrays_dict['fixed_phaseToCompMap'].astype(np.float32)
    indexer.compositionally_variable_subset = arrays_dict['compositionally_variable_subset'].astype(int)
    indexer.compositional_component_subset = arrays_dict['compositional_component_subset'].astype(int)
    indexer.comp_binaries = arrays_dict['comp_binaries'].astype(int)
    indexer.comp_mappings = arrays_dict['comp_mappings'].astype(np.float32)
    
    # === RESTORE NORMALIZERS ===
    # Reconstruct from state dicts if they exist
    
    if metadata.get('has_feature_normalizer', False):
        feature_state = {
            'min': arrays_dict['normalizer_feature_min'],
            'range': arrays_dict['normalizer_feature_range'],
            'device': 'cpu'
        }
        indexer.feature_normalizer = Normalizer.from_state_dict(feature_state, cuda=False)
    else:
        indexer.feature_normalizer = None
    
    if metadata.get('has_output_normalizer', False):
        output_state = {
            'min': arrays_dict['normalizer_output_min'],
            'range': arrays_dict['normalizer_output_range'],
            'device': 'cpu'
        }
        indexer.output_normalizer = Normalizer.from_state_dict(output_state, cuda=False)
    else:
        indexer.output_normalizer = None
    
    # comp_variable_IDMAT is not saved as it's torch-specific and can be None
    indexer.comp_variable_IDMAT = None
    if TORCH_AVAILABLE and len(indexer.compositionally_variable_binaries) > 0:
        indexer.comp_variable_IDMAT = torch.tensor(
            np.diag(indexer.compositionally_variable_binaries), dtype=torch.float
        )
    
    # === BACKWARD COMPATIBILITY ===
    indexer.comp_map = indexer.label_indices_comp
    
    return indexer

