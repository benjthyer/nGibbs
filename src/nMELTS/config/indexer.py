"""
Dynamic dataset indexer for nMELTS.

Generates all index mappings dynamically from dataset column headers,
enabling flexible addition/removal of phases and components.
"""

import re
import numpy as np
# Make torch optional for WSL scripts that don't need ML features
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
from typing import List, Dict, Set, Tuple, Optional, Any

# Import defaults from constants for melts-liquid label generation and oxide indexing
from .constants import COMPONENTS_IN_PHASES, default_Elkeys

# Element to oxide mapping dictionary
ELEMENT_TO_OXIDE = {
    'Si': 'SiO2',
    'Ti': 'TiO2',
    'Al': 'Al2O3',
    'Fe': 'FeO',  # FeO also represents FeOT, Fe2O3 is added separately 
    'Mg': 'MgO',
    'Ca': 'CaO',
    'Na': 'Na2O',
    'K': 'K2O',
    'P': 'P2O5',
    'H': 'H2O',
    'Cr': 'Cr2O3',
    'Mn': 'MnO',
    'Ni': 'NiO'
}

# Required elements (must always be present)
REQUIRED_ELEMENTS = {'Si', 'Al', 'Fe', 'Ca', 'Mg'}
# ============================================================================
#  Phase-to-Components/Attributes Dictionary
# ============================================================================
# This dictionary maps phases to their chemical components (excluding state variables).
# Used by both DatasetIndexer and scripts that generate column headers for new datasets.
# State variables (mass, density, thermodynamic properties) for non melts-liquid phases are added separately.



def generate_column_headers(phases: List[str]) -> List[str]:
    """
    Generate column headers from a list of phase names using COMPONENTS_IN_PHASES.
    
    Parameters
    ----------
    phases : List[str]
        List of phase names to generate columns for
        
    Returns
    -------
    List[str]
        List of column headers in format 'component(phase)'
        
    Examples
    --------
    >>> phases = ['olivine', 'plagioclase']
    >>> headers = generate_column_headers(phases)
    >>> headers[:3]
    ['mass (gm)(olivine)', 'forsterite(olivine)', 'fayalite(olivine)']
    """
    column_headers = []
    
    # Automatically add 'analcime' if 'leucite' is in the phases list
    phases_set = set(phases)
    if 'leucite' in phases_set and 'analcime' not in phases_set:
        phases_set.add('analcime')
        phases = list(phases_set)
    
    #Begin with system_main:
    components = COMPONENTS_IN_PHASES['System_main']
            
    for component in components:
        column_headers.append(f"{component}({'System_main'})")
    # We are going to sort the phases by the order in COMPONENTS_IN_PHASES
    for phase in phases:
        if phase not in COMPONENTS_IN_PHASES:
            if phase == 'liquid': # Special case for melts-liquid, which is called 'liquid' when building the .melts files. 
                for component in COMPONENTS_IN_PHASES['melts-liquid']:
                    column_headers.append(f"{component}({'melts-liquid'})")
            else:
                raise ValueError(f"Phase '{phase}' not found in COMPONENTS_IN_PHASES. "
                           f"Available phases: {list(COMPONENTS_IN_PHASES.keys())}")
        
        else:
            components = COMPONENTS_IN_PHASES[phase]
            
            for component in components:
                column_headers.append(f"{component}({phase})")
    
    return column_headers


class DatasetIndexer:
    """
    Dynamic indexer that generates all mappings from dataset headers.
    
    Headers must follow the format: component(phase)
    Examples:
        - mass (gm)(olivine)
        - forsterite(olivine)
        - Pressure(System_main)
        - wt% SiO2(melts-liquid)

    ADD FEATURE TO RAISE AN ERROR IF THERE ARE NONZERO VALUES IN THE DATASET FOR EXCLUDED COMPONENTS OR PHASES, EXCEPT SYSTEM_MAIN
    """
    
    def __init__(self, headers: List[str], # For now, does not handle NiO and MnO
        EXCLUDED_PHASES = {'System_main'},
        EXCLUDED_COMPONENTS = {'tephroite', 'co-olivine', 'ni-olivine', 'pyrophanite', 'Mn', 'Ni', 'Fe-metal', 'Ni-metal', 'Fe-liquid', 'Ni-liquid'}, 
        STATE_VARIABLES = {
            'mass (gm)', 'rho (gm/cc)', 'H (kJ)', 'S (J/K)', 'V (cc)',
            'liq mass (gm)', 'liq rho (gm/cc)', 'liq vis (log 10 poise)',
            'liq H (kJ)', 'liq S (J/K)', 'liq V (cc)'
            },
        Elkeys: Optional[List[str]] = None):
        """
        Initialize indexer from column headers.
        
        Parameters
        ----------
        headers : List[str]
            List of column header strings in format 'component(phase)'
        EXCLUDED_PHASES : Set[str]
            Set of phase names to exclude from ML indexing (default: {'System_main'})
        EXCLUDED_COMPONENTS : Set[str]
            Set of component names to exclude from ML indexing
        STATE_VARIABLES : Set[str]
            Set of state variable names to exclude from chemical component indexing
        Elkeys : Optional[List[str]]
            List of element names. If None, uses default from constants.
            Required elements (Si, Al, Fe, Ca, Mg) will be automatically added if missing.
            WRkeys and Oxides are built automatically from Elkeys.
        """
        self.headers = headers
        self.database_headers = headers.copy()
        self.EXCLUDED_COMPONENTS = EXCLUDED_COMPONENTS
        self.STATE_VARIABLES = STATE_VARIABLES
        self.EXCLUDED_PHASES = EXCLUDED_PHASES
        if 'System_main' not in self.EXCLUDED_PHASES:
            self.EXCLUDED_PHASES.add('System_main') # If users don't pass this, we will.

        # Build Elkeys - use defaults if not provided, and force required elements
        if Elkeys is None:
            self.Elkeys = default_Elkeys.copy()
        else:
            self.Elkeys = list(Elkeys)  # Make a copy
        
        # Force required elements to be present
        for required_el in REQUIRED_ELEMENTS:
            if required_el not in self.Elkeys:
                self.Elkeys.append(required_el)
        
        # Remove duplicates while preserving order
        seen = set()
        self.Elkeys = [el for el in self.Elkeys if not (el in seen or seen.add(el))]
        
        # Build WRkeys from Elkeys (all oxides except Fe2O3)
        self.WRkeys = []
        for el in self.Elkeys:
            if el in ELEMENT_TO_OXIDE:
                ox = ELEMENT_TO_OXIDE[el]
                if ox not in self.WRkeys:
                    self.WRkeys.append(ox)
        
        # Build Oxides from WRkeys + Fe2O3 (if Fe is present)
        self.Oxides = self.WRkeys.copy()
        if 'Fe' in self.Elkeys and 'Fe2O3' not in self.Oxides:
            self.Oxides.append('Fe2O3')
        
        # Build oxide_dict from Oxides list
        self.oxide_dict: Dict[str, int] = {}
        for i, ox in enumerate(self.Oxides):
            self.oxide_dict[ox] = i
        
        # Build active_ox_dict (phase to oxide indices mapping)
        # This maps each phase to the indices of oxides it uses
        self.active_ox_dict: Dict[str, List[int]] = {}
        active_oxide_lists = {
            'olivine': ['SiO2','MgO', 'FeO', 'CaO'],
            'orthopyroxene': ['SiO2', 'CaO', 'MgO', 'FeO', 'Fe2O3', 'Na2O', 'Al2O3', 'TiO2'],
            'clinopyroxene': ['SiO2', 'CaO', 'MgO', 'FeO', 'Fe2O3', 'Na2O', 'Al2O3', 'TiO2'],
            'spinel': ['Cr2O3', 'MgO', 'FeO', 'Fe2O3', 'Al2O3', 'TiO2'],
            'plagioclase': ['SiO2', 'Na2O', 'K2O', 'CaO'],
            'k-feldspar': ['SiO2', 'Na2O', 'K2O', 'CaO'],
            'garnet': ['SiO2', 'FeO', 'CaO', 'MgO'],
            'nepheline': ['SiO2', 'Na2O', 'K2O', 'CaO', 'Al2O3'],
            'leucite': ['SiO2', 'Na2O', 'K2O', 'H2O'],
            'biotite': ['SiO2','FeO','MgO'],
            'rhm-oxide': ['MgO', 'FeO', 'Fe2O3', 'TiO2'],
            'analcime': ['SiO2', 'Na2O', 'K2O', 'H2O'],
            'melts-liquid': self.Oxides  # melts-liquid uses all oxides
        }
        
        for phase, oxide_list in active_oxide_lists.items():
            # Only include oxides that are in our Oxides list
            phase_oxide_indices = []
            for ox in oxide_list:
                if ox in self.oxide_dict:
                    phase_oxide_indices.append(self.oxide_dict[ox])
            if phase_oxide_indices:  # Only add if there are valid oxides
                self.active_ox_dict[phase] = phase_oxide_indices

        # Core mappings
        self.MELTS_indices: Dict[str, Dict[str, int]] = {}
        self.mass_indices: np.ndarray = np.array([], dtype=int)
        
        # Phase to chemical components mapping (single-level dict)
        self.components_in_phases: Dict[str, List[str]] = {}
        
        # ML-ready mappings
        self.label_indices: Dict[str, List[int]] = {}
        self.label_names: List[str] = []
        self.detail_label_indices: Dict[str, Dict[str, int]] = {}
        self.label_indices_comp: Dict[str, np.ndarray] = {}
        
        # Phase dictionaries
        self.all_phases: List[str] = []
        self.mass_phasedict: Dict[str, int] = {}
        self.comp_phasedict: Dict[str, int] = {}
        self.compositionally_variable_phases: List[str] = []
        
        # Phase-component matrices
        self.phaseToCompMap: np.ndarray = np.array([])
        self.variedToAllComp: np.ndarray = np.array([])
        self.comp_variable_IDMAT: Optional[Any] = None  # torch.Tensor when torch is available
        self.fixed_phaseToCompMap: np.ndarray = np.array([])
        self.compositionally_variable_subset: np.ndarray = np.array([])
        self.compositional_component_subset: np.ndarray = np.array([])
        
        # Component mappings
        self.comp_map: Dict[str, np.ndarray] = {}
        self.comp_binaries: np.ndarray = np.array([])
        self.comp_mappings: np.ndarray = np.array([])
        
        # Derived counts
        self.ncomps: int = 0
        self.ncompsVaried: int = 0
        self.nphases: int = 0
        
        # Parse and build all indices
        self._parse_headers() # Build MELTS_indicies to index all of the MELTS table
        self._build_components_in_phases() # Build compositional indices
        self._build_mass_indices()
        self._build_label_indices()
        self._build_phase_dictionaries()
        self._build_phase_mappings()
        self._build_component_mappings()
    
    def _parse_headers(self):
        """Parse headers into MELTS_indices structure."""
        # Parse headers in format component(phase)
        # Components may contain parentheses (e.g., "mass (gm)"), so we need to
        # find the last opening parenthesis before the final closing parenthesis
        for idx, header in enumerate(self.headers):
            header = header.strip()
            
            # Find the last opening parenthesis (this separates component from phase)
            last_open_paren = header.rfind('(')
            last_close_paren = header.rfind(')')
            
            # Must have both parentheses and opening must come before closing
            if last_open_paren == -1 or last_close_paren == -1 or last_open_paren >= last_close_paren:
                # Skip headers that don't match the pattern
                continue
            
            # Extract component (everything before last opening paren)
            component = header[:last_open_paren].strip()
            # Extract phase (everything between last opening and closing paren)
            phase = header[last_open_paren + 1:last_close_paren].strip()
            
            # Skip if either is empty
            if not component or not phase:
                continue
            
            # Initialize phase dict if needed
            if phase not in self.MELTS_indices:
                self.MELTS_indices[phase] = {}
            
            # Store component index
            self.MELTS_indices[phase][component] = idx
    
    def _build_components_in_phases(self):
        """
        #Build single-level dictionary mapping phases to lists of chemical components.
        #First extracts chemical components, then adds state variables (mass, density, etc.).
        """
        # Extract chemical components only (exclude state variables and excluded components)
        for phase, components in self.MELTS_indices.items():
            chemical_components = []

            # Skip excluded phases
            if phase in self.EXCLUDED_PHASES:
                continue

            for component in components.keys():
                # Skip state variables
                if component in self.STATE_VARIABLES:
                    continue
                
                # Skip excluded components
                if component in self.EXCLUDED_COMPONENTS:
                    continue
    
                if 'mass' in component.lower():
                    #print('mass skip!')
                    continue

                
                chemical_components.append(component)
            if len(chemical_components) == 0: 
                chemical_components = [phase] # Pure Phases are one component
            # Store chemical components for this phase
            # This is an initermediate used to guide the building of the ML indexers. Includes both pure and chemically variable phases
            self.components_in_phases[phase] = chemical_components 
        

        
    
    def _build_mass_indices(self):
        """Find all mass column indices."""
        mass_indices_list = []
        
        # Get solid phases that aren't excluded 
        # Liquid has diferent labels unfortunately, so we need to handle it separately: phases_in_order + ['melts-liquid']
        phases_with_mass = set()
        for phase in self.MELTS_indices.keys():
            if phase not in self.EXCLUDED_PHASES:
                phases_with_mass.add(phase)
        
        for phase, components in self.MELTS_indices.items():
            if phase in phases_with_mass:
                for component, idx in components.items():
                    if 'mass' in component.lower():
                        mass_indices_list.append(idx)
        
        self.mass_indices = np.unique(np.array(mass_indices_list, dtype=int))
    

    def _build_label_indices(self):
        """Generate ML-ready label indices from components_in_phases."""
        detail_ind = 0
        index = 0
        
        # Process phases,
        for phase, chemical_components in self.components_in_phases.items():
            
            phase_inds = []
            
            # Determine if phase should have detailed component mapping
            # A phase is detailed if it has chemical components (excluding state variables)
            # Filter out state variables from the chemical_components list
            
            is_detailed = len(chemical_components) > 0
            
            if is_detailed:
                self.detail_label_indices[phase] = {}
                comp_inds = []
                
                for component in chemical_components:
                    phase_inds.append(index)
                    self.detail_label_indices[phase][component] = detail_ind # This object indexes composition in the compositionally variant submatrix
                    comp_inds.append(detail_ind)
                    detail_ind += 1
                    index += 1
                    self.label_names.append(component)
                
                self.label_indices_comp[phase] = np.array(comp_inds)
            else:
                # Simple phase mapping (single index) - phase has only state variables
                phase_inds.append(index)
                index += 1
                self.label_names.append(phase)
            
            self.label_indices[phase] = phase_inds
        
        # Special handling for melts-liquid: use Elkeys directly (not components from headers)
        # melts-liquid label_indices are based on Elkeys (from self.Elkeys, which may be customized)
        if 'melts-liquid' in self.MELTS_indices:
            kept_Elkeys = []
            for key in self.Elkeys:
                if key not in self.EXCLUDED_COMPONENTS:
                    kept_Elkeys.append(key)

            elkeys_len = len(kept_Elkeys)
            self.label_indices['melts-liquid'] = list(range(index, index + elkeys_len))
            self.label_names.extend(kept_Elkeys)
            
            # Build detail_label_indices for melts-liquid
            self.detail_label_indices['melts-liquid'] = {}
            comp_inds = []
            for key in kept_Elkeys:
                self.detail_label_indices['melts-liquid'][key] = detail_ind
                comp_inds.append(detail_ind)
                detail_ind += 1
            self.label_indices_comp['melts-liquid'] = np.array(comp_inds)
    
    def _build_phase_dictionaries(self):
        """Build phase dictionaries and identify compositionally variable phases. Called after building label_indices."""
        cj = 0
        
        for i, phase in enumerate(self.label_indices.keys()):
            self.all_phases.append(phase)
            self.mass_phasedict[phase] = i # This indexes a matrix of all phase masses
            
            # Phase is compositionally variable if it has multiple label indices
            if len(self.label_indices[phase]) > 1:
                self.compositionally_variable_phases.append(phase)
                self.comp_phasedict[phase] = cj # This indexes a matrix of compositionally variable phases
                cj += 1
        
        # Calculate dimensions
        # nphases should always be len(all_phases) since all_phases contains all phases in label_indices
        self.nphases = len(self.all_phases)
        
        if 'melts-liquid' in self.label_indices and len(self.label_indices['melts-liquid']) > 0:
            self.ncomps = self.label_indices['melts-liquid'][-1] + 1
            if 'melts-liquid' in self.label_indices_comp and len(self.label_indices_comp['melts-liquid']) > 0:
                self.ncompsVaried = self.label_indices_comp['melts-liquid'][-1] + 1
            else:
                self.ncompsVaried = 0
        else:
            # Fallback if melts-liquid not present, Not tested as of 1/9/2026
            if len(self.label_indices) > 0:
                max_idx = max(max(inds) for inds in self.label_indices.values() if len(inds) > 0)
                self.ncomps = max_idx + 1
                if len(self.label_indices_comp) > 0:
                    max_varied = max(max(comp_inds) for comp_inds in self.label_indices_comp.values() if len(comp_inds) > 0)
                    self.ncompsVaried = max_varied + 1 if max_varied >= 0 else 0
                else:
                    self.ncompsVaried = 0
            else:
                self.ncomps = 0
                self.ncompsVaried = 0
    
    def _build_phase_mappings(self):
        """Build phase-to-component mapping matrices, for reduced size ML tensor structures"""
        if self.nphases == 0 or self.ncomps == 0:
            return
        
        self.phaseToCompMap = np.zeros((self.nphases, self.ncomps), dtype=float)
        self.compositionally_variable_binaries = []
        compositionally_variable_subset_list = []
        
        # Determine ncompsVaried for variedToAllComp
        if self.ncompsVaried == 0:
            # Calculate from label_indices_comp
            max_varied = 0
            for comp_inds in self.label_indices_comp.values():
                if len(comp_inds) > 0:
                    max_varied = max(max_varied, max(comp_inds))
            self.ncompsVaried = max_varied + 1 if max_varied > 0 else 0
        
        if self.ncompsVaried > 0:
            self.variedToAllComp = np.zeros((self.ncompsVaried, self.ncomps), dtype=float)
        else:
            self.variedToAllComp = np.zeros((0, self.ncomps), dtype=float)
        
        for p, phase in enumerate(self.all_phases):
            if phase in self.label_indices:
                phase_inds = self.label_indices[phase]
                self.phaseToCompMap[p, phase_inds] = 1
                
                if phase in self.compositionally_variable_phases:
                    self.compositionally_variable_binaries.append(1)
                    compositionally_variable_subset_list.extend(phase_inds)
                    
                    if phase in self.label_indices_comp:
                        comp_inds = self.label_indices_comp[phase]
                        label_inds = self.label_indices[phase]
                        # Map varied components to all components
                        for varied_idx, label_idx in zip(comp_inds, label_inds):
                            if varied_idx < self.ncompsVaried and label_idx < self.ncomps:
                                self.variedToAllComp[varied_idx, label_idx] = 1
                else:
                    self.compositionally_variable_binaries.append(0)
        
        self.compositionally_variable_binaries = np.array(self.compositionally_variable_binaries)
        self.compositionally_variable_subset = np.array(compositionally_variable_subset_list, dtype=int)
        self.compositional_component_subset = np.copy(self.compositionally_variable_subset)
        
        # Build torch tensor for comp_variable_IDMAT (only if torch is available)
        if TORCH_AVAILABLE and len(self.compositionally_variable_binaries) > 0:
            self.comp_variable_IDMAT = torch.tensor(
                np.diag(self.compositionally_variable_binaries), 
                dtype=torch.float
            )
        else:
            # Set to None if torch not available - this is fine for WSL scripts
            self.comp_variable_IDMAT = None
        
        # Build fixed_phaseToCompMap
        is_fixed = ~(self.compositionally_variable_binaries.astype(bool))
        self.fixed_phaseToCompMap = (is_fixed.reshape(1, -1) @ self.phaseToCompMap)
    
    def _build_component_mappings(self):
        """Build component mapping structures."""
        j = 0
        k = 0
        comp_binaries_list = []
        comp_mappings_list = []
        
        # First pass: build comp_map
        for label, inds in self.label_indices.items():
            n_components = len(inds)
            if n_components > 1:
                comp_list = np.arange(k, k + n_components)
                k += n_components
                self.comp_map[label] = comp_list
        
        # Second pass: build comp_binaries and comp_mappings
        for i, (label, inds) in enumerate(self.label_indices.items()):
            n_components = len(inds)
            if n_components > 1:
                comp_binaries_list.append(i)
                comp_mappings_list.extend([j] * n_components)
                j += 1
        
        self.comp_binaries = np.array(comp_binaries_list, dtype=int)
        
        if len(comp_mappings_list) > 0:
            self.comp_mappings = np.zeros((j, len(comp_mappings_list)), dtype=float)
            for col, row in enumerate(comp_mappings_list):
                self.comp_mappings[row, col] = 1
        else:
            self.comp_mappings = np.zeros((0, 0), dtype=float)
    
    def get_max_index(self) -> int:
        """Get the maximum index value across all mappings."""
        max_idx = -1
        for phase_dict in self.MELTS_indices.values():
            for idx in phase_dict.values():
                max_idx = max(max_idx, idx)
        return max_idx
    
    def get_phase_list(self) -> List[str]:
        """Get list of all phases in the dataset."""
        return list(self.MELTS_indices.keys())
    
    def get_components_for_phase(self, phase: str) -> List[str]:
        """Get list of components for a given phase."""
        return list(self.MELTS_indices.get(phase, {}).keys())


def verify_csv_headers_match(csv_path: str, indexer: DatasetIndexer) -> Tuple[bool, Optional[str]]:
    """
    Verify that an existing CSV file's headers match the expected headers from a DatasetIndexer.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file to check
    indexer : DatasetIndexer
        DatasetIndexer object with expected headers
        
    Returns
    -------
    Tuple[bool, Optional[str]]
        (True, None) if headers match, (False, error_message) if they don't match
        Returns (False, "File does not exist") if the file doesn't exist
        
    Examples
    --------
    >>> indexer = DatasetIndexer(headers)
    >>> matches, error = verify_csv_headers_match('data.csv', indexer)
    >>> if not matches:
    ...     print(f"Header mismatch: {error}")
    """
    import pandas as pd
    import os
    
    if not os.path.exists(csv_path):
        return (False, f"File does not exist: {csv_path}")
    
    try:
        # Read just the header row
        df = pd.read_csv(csv_path, nrows=0)
        existing_headers = list(df.columns)
        expected_headers = indexer.database_headers
        
        # Check if lengths match
        if len(existing_headers) != len(expected_headers):
            return (False, f"Header count mismatch: expected {len(expected_headers)} columns, "
                          f"found {len(existing_headers)} columns")
        
        # Check if headers match exactly
        if existing_headers != expected_headers:
            # Find differences
            mismatches = []
            for i, (existing, expected) in enumerate(zip(existing_headers, expected_headers)):
                if existing != expected:
                    mismatches.append(f"Column {i}: expected '{expected}', found '{existing}'")
            
            error_msg = f"Header mismatch:\n" + "\n".join(mismatches[:10])  # Limit to first 10 mismatches
            if len(mismatches) > 10:
                error_msg += f"\n... and {len(mismatches) - 10} more mismatches"
            
            return (False, error_msg)
        
        return (True, None)
        
    except Exception as e:
        return (False, f"Error reading CSV file: {str(e)}")