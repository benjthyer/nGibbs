"""
Dynamic dataset indexer for nMELTS.

Generates all index mappings dynamically from dataset column headers,
enabling flexible addition/removal of phases and components.
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
# Make torch optional for WSL scripts that don't need ML features
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
from typing import List, Dict, Set, Tuple, Optional, Any

# Ensure src is on path
import sys
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import defaults from constants for melts-liquid label generation and oxide indexing
from nMELTS.config.constants import (
    COMPONENTS_IN_PHASES,
    COMPONENTS_IN_PHASES_HEFESTO,
    COMPOSITIONAL_COMPONENTS_IN_PHASES,
    COMPOSITIONAL_COMPONENTS_IN_PHASES_HEFESTO,
    default_Elkeys,
    REQUIRED_ELEMENTS,
)
from nMELTS.config.ml_indexer import MLIndexer

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
    'Ni': 'NiO',
    'Fe3': 'Fe2O3' # Closed oxygen, added for completeness
}

# Reverse mapping: oxide to element
OXIDE_TO_ELEMENT = {oxide: element for element, oxide in ELEMENT_TO_OXIDE.items()}

# ============================================================================
#  Phase-to-Components/Attributes Dictionary
# ============================================================================
# This dictionary maps phases to their chemical components (excluding state variables).
# Used by both DatasetIndexer and scripts that generate column headers for new datasets.
# State variables (mass, density, thermodynamic properties) for non melts-liquid phases are added separately.

def generate_column_headers(phases: List[str], mode: str = 'None', zeroOxides: List[str] = []) -> List[str]:
    """
    Generate column headers from a list of phase names using COMPONENTS_IN_PHASES.
    
    Parameters
    ----------
    phases : List[str]
        List of phase names to generate columns for

    mode : str       
        MELTS model mode, which affects how certain phases are labeled. pMELTS doesn't include MnO, NiO, CoO or Corundum in rhm-oxide
        Default does not force these exclusions
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
    
    # Ensure bulk and system info included: 
    phases = ['System_main', 'Bulk_comp'] + phases

    # Automatically add 'analcime' if 'leucite' is in the phases list
    if 'leucite' in phases and 'analcime' not in phases:
        phases.append('analcime')

    is_liquid = False # Flag to handle special case for melts-liquid

    # We are going to sort the phases by the order in COMPONENTS_IN_PHASES
    for phase in phases:
        if phase not in COMPONENTS_IN_PHASES:
            if phase == 'liquid': # Special case for melts-liquid, which is called 'liquid' when building the .melts files. 
                is_liquid = True
            else:
                raise ValueError(f"Phase '{phase}' not found in COMPONENTS_IN_PHASES. "
                           f"Available phases: {list(COMPONENTS_IN_PHASES.keys())}")
        
        else:
            components = COMPONENTS_IN_PHASES[phase]
            
            for component in components:
                if (mode.lower() != 'p' or (not any([pExclude in component for pExclude in ['CoO', 'corundum']]))): # except oxides, components lowercase
                    if not any([oxide in component for oxide in zeroOxides]):
                        column_headers.append(f"{component}({phase})")
                else:
                    print(f"Excluding component '{component}' from phase '{phase}' for pMELTS mode.")

    
    if is_liquid: # I want the liquid at the end for clarity and continuity with previous editions
        for component in COMPONENTS_IN_PHASES['melts-liquid']:
            if not any([oxide in component for oxide in zeroOxides]):
                column_headers.append(f"{component}({'melts-liquid'})")

    return column_headers


def generate_column_headers_hefesto(
    phases: List[str]
) -> List[str]:
    """
    Generate column headers for HeFESTo phases using the HeFESTo phase/species dictionary.

    Parameters
    ----------
    phases : List[str]
        List of full HeFESTo phase names (e.g., 'clinopyroxene').
        Abbreviations are not accepted here and should be resolved by parser code.

    Returns
    -------
    List[str]
        Column headers in format 'component(phase)'.

    Examples
    --------
    >>> headers = generate_column_headers_hefesto(['orthopyroxene', 'clinopyroxene'])
    >>> headers[:4]
    ['enstatite(orthopyroxene)', 'ferrosilite(orthopyroxene)',
     'mg-tschermaks(orthopyroxene)', 'ortho-diopside(orthopyroxene)']
    """
    column_headers: List[str] = []

    phases = ['System_main', 'Bulk_comp', 'Bulk_comp_elements'] + phases

    for phase_name in phases:

        if phase_name not in COMPONENTS_IN_PHASES_HEFESTO:
            raise ValueError(
                f"HeFESTo phase '{phase_name}' not found in COMPONENTS_IN_PHASES_HEFESTO. "
                f"Available phases: {list(COMPONENTS_IN_PHASES_HEFESTO.keys())}"
            )
        
        for component in COMPONENTS_IN_PHASES_HEFESTO[phase_name]:
            column_headers.append(f"{component}({phase_name})")

        # Keep phase-total moles as part of schema generation (not importer runtime logic).
        if phase_name not in {'System_main', 'Bulk_comp', 'Bulk_comp_elements'}:
            header_name = f"total (moles)({phase_name})"
            if header_name not in column_headers:
                column_headers.append(header_name)

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
        #EXCLUDED_PHASES = {'System_main', 'Bulk_comp'},
        #EXCLUDED_COMPONENTS_BY_PHASE: Optional[Dict[str, Set[str]]] = None,
        STATE_VARIABLES = {
            'mass (gm)', 'rho (gm/cc)', 'H (kJ)', 'S (J/K)', 'V (cc)',
            'liq mass (gm)', 'liq rho (gm/cc)', 'liq vis (log 10 poise)',
            'liq H (kJ)', 'liq S (J/K)', 'liq V (cc)', 'total (moles)'
            },
        OXYGEN = None, #'closed',
        MODEL = 'MELTS'): 
        #Elkeys: Optional[List[str]] = None):
        """
        Initialize indexer from column headers.
        
        Parameters
        ----------
        headers : List[str]
            List of column header strings in format 'component(phase)'
        EXCLUDED_PHASES : Set[str]
            Set of phase names to exclude from ML indexing (default: {'System_main'})
        EXCLUDED_COMPONENTS_BY_PHASE : Optional[Dict[str, Set[str]]]
            Mapping of phase -> components to exclude from ML indexing. Components are excluded only within their phase.
        STATE_VARIABLES : Set[str]
            Set of state variable names to exclude from chemical component indexing
        OXYGEN: Determines whether Fe2O3 is conserved ('closed') or if oxygen is buffered ('open')
        MODEL: Determines if model is melts or other. HeFESTo doesn't use PxSp transformation matrices 
        """
        self.headers = headers
        self.database_headers = headers.copy()

        # Core mappings
        self.MELTS_indices: Dict[str, Dict[str, int]] = {}
        self.mass_indices: np.ndarray = np.array([], dtype=int)

        # Initialize empty exclusions, to be populated later dynamically based on data. These exclude phases and components from being
        # carried further into the ml_indexer object, for which ALL COLUMNS of EVERY TABLE is expected to be nonzero (else the model carries dead neurons)
        self.EXCLUDED_COMPONENTS_BY_PHASE = {}

        self.STATE_VARIABLES = STATE_VARIABLES
        self.EXCLUDED_PHASES = {'System_main', 'Bulk_comp', 'Bulk_comp_elements'}  # Default excluded phases
        self.OXYGEN = OXYGEN
        self.MODEL = MODEL

        assert self.OXYGEN in ['closed', 'open'], "OXYGEN parameter must be 'closed' (constant Fe2O3) or 'open' (buffered)"

        self._parse_headers() # Build MELTS_indicies to index all of the MELTS table

        # Dynamically populate Elkeys from Bulk_comp components
        self._populate_elkeys_from_bulk_comp()

        self._build_fundamentals() # Build WRkeys and Oxides from Elkeys
        

        # Exclude components whose oxides fall outside the active Oxides list
        projections_dir = Path(__file__).resolve().parent.parent / 'nMELTS' / 'config' / 'projections'
        comp_to_ox_path = projections_dir / 'compToOxV2.csv'
        if self.OXYGEN == 'open':
            ox_to_el_path = projections_dir / 'OxToElV2.csv'
        elif self.OXYGEN == 'closed':
            ox_to_el_path = projections_dir / 'OxToElV2_ferric.csv'
        self.components_with_extra_oxides: Dict[str, List[str]] = {}
        self.oxides_to_elements: Dict[str, List[str]] = {}
        self.compToOx_df = pd.read_csv(comp_to_ox_path, index_col=0)
        oxToEl_df = pd.read_csv(ox_to_el_path, index_col=0)

        # Build oxide -> elements lookup
        for ox in oxToEl_df.columns:
            elems = [el for el, val in oxToEl_df[ox].items() if float(val) != 0]
            if elems:
                self.oxides_to_elements[ox] = elems
        
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
        
        # Parse and build initial indices
        self._repopulate_indexer()

    def _populate_elkeys_from_bulk_comp(self):
        """
        Dynamically populate self.Elkeys from Bulk_comp components.
        
        Extracts oxide names from Bulk_comp headers and maps them to elements
        using the OXIDE_TO_ELEMENT mapping. Avoids duplicate entries.
        """
        elkeys_set = set()  # Use set to avoid duplicates (including duplicate Fe)
        
        if 'Bulk_comp' in self.MELTS_indices:
            for component in self.MELTS_indices['Bulk_comp'].keys():
                # Skip state variables
                if component in self.STATE_VARIABLES:
                    continue
                
                # Extract oxide name (remove 'wt% ' prefix if present)
                oxide_name = component.replace('wt% ', '').strip()
                
                # Map oxide to element
                if oxide_name in OXIDE_TO_ELEMENT:
                    if oxide_name == 'Fe2O3':
                        if self.OXYGEN == 'open':
                            print("Skipping Fe3 in Elkeys because oxygen is buffered (open system)")
                            # In open oxygen case, we treat Fe as buffered and don't include Fe2O3 as an active oxide, so we skip adding Fe to Elkeys
                            continue
                        print("Adding Fe3 in Elkeys because oxygen is buffered (open system)")
                    elkeys_set.add(OXIDE_TO_ELEMENT[oxide_name])
        
        # Convert to sorted list for consistent, reproducible ordering
        self.Elkeys = sorted(list(elkeys_set))

    def _build_fundamentals(self):
        
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

    def _repopulate_indexer(self):
        self._look_for_illegal_oxides()
        self._look_for_dead_phases()
        self._build_components_in_phases() # Build compositional indices
        self._build_mass_indices()
        self._build_ml_indexer()

    def _build_ml_indexer(self):
        
        # Delegate building of ML-ready indexers to MLIndexer (constructed from components_in_phases)
    
        self.ml_indexer = MLIndexer(components_in_phases=self.components_in_phases,
                                    Elkeys=self.Elkeys, Model=self.MODEL)

        self.expose_ml_indexer_attributes()


    def expose_ml_indexer_attributes(self):
        # Expose ML indexer attributes on DatasetIndexer for compatibility. Just in case ml_indexer is changed underneath indexer
        self.label_indices = self.ml_indexer.label_indices
        self.label_names = self.ml_indexer.label_names
        self.detail_label_indices = self.ml_indexer.detail_label_indices
        self.label_indices_comp = self.ml_indexer.label_indices_comp
        self.all_phases = self.ml_indexer.all_phases
        self.mass_phasedict = self.ml_indexer.mass_phasedict
        self.comp_phasedict = self.ml_indexer.comp_phasedict
        self.compositionally_variable_phases = self.ml_indexer.compositionally_variable_phases

        self.phaseToCompMap = self.ml_indexer.phaseToCompMap
        self.variedToAllComp = self.ml_indexer.variedToAllComp
        self.comp_variable_IDMAT = self.ml_indexer.comp_variable_IDMAT
        self.fixed_phaseToCompMap = self.ml_indexer.fixed_phaseToCompMap
        self.compositionally_variable_subset = getattr(self.ml_indexer, 'compositionally_variable_subset', np.array([], dtype=int))
        self.compositional_component_subset = getattr(self.ml_indexer, 'compositional_component_subset', np.array([], dtype=int))

        self.comp_map = getattr(self.ml_indexer, 'comp_map', {})
        self.comp_binaries = getattr(self.ml_indexer, 'comp_binaries', np.array([], dtype=int))
        self.comp_mappings = getattr(self.ml_indexer, 'comp_mappings', np.array([], dtype=float))

        self.ncomps = self.ml_indexer.ncomps
        self.ncompsVaried = self.ml_indexer.ncompsVaried
        self.nphases = self.ml_indexer.nphases
    

    
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
            #test: max val = # len(headers) -1
            #test: len(values) == len(headers)
    
    def _build_components_in_phases(self):
        """
        #Build single-level dictionary mapping phases to lists of chemical components.
        #First extracts chemical components, then adds state variables (mass, density, etc.).
        """
        self.components_in_phases = {} # Reinitialize the empty Dictionary 

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
                
                # Skip excluded components for this phase
                if component in self.EXCLUDED_COMPONENTS_BY_PHASE.get(phase, set()):
                    continue
    
                if 'mass' in component.lower():
                    #print('mass skip!')
                    continue

                
                chemical_components.append(component)
            if len(chemical_components) == 0: 
                chemical_components = [phase] # Pure Phases are one component
            # Store chemical components for this phase
            # This is an intermediate used to guide the building of the ML indexers. Includes both pure and chemically variable phases
            self.components_in_phases[phase] = chemical_components 
        print(f"""Components in phases after building: {self.components_in_phases}""")
        #test: every phase has at least one component

        
    
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
                    if 'mass' in component.lower() or 'moles' in component.lower():
                        mass_indices_list.append(idx)
        
        self.mass_indices = np.array(mass_indices_list, dtype=int)

    def _look_for_illegal_oxides(self):
        """
        Identify and exclude components that use oxides outside self.Oxides.
        """
        # Identify components whose oxide usage is not a subset of self.Oxides
        for comp_full in self.compToOx_df.index:
            comp_name = comp_full.split(' : ')[0].strip()
            row = self.compToOx_df.loc[comp_full]
            present_oxides = [col for col, val in row.items() if float(val) != 0]
            extra_oxides = [ox for ox in present_oxides if ox not in self.Oxides]
            if extra_oxides:
                self.components_with_extra_oxides[comp_name] = extra_oxides
                for phase, components in self.MELTS_indices.items():
                    if comp_name in components:
                        self.EXCLUDED_COMPONENTS_BY_PHASE.setdefault(phase, set()).add(comp_name)

    def _look_for_dead_phases(self):
        """
        Identify and exclude compositionally variable phases
        whose components are all excluded.
        """

        # Only act on compositionally-variable phases (len>1) and only when *all* components
        # are currently excluded. Pure phases (len==1) must remain, even if their lone
        # component is excluded elsewhere.
        if self.MODEL == 'HeFESTo':
            compositional_components_in_phases = COMPOSITIONAL_COMPONENTS_IN_PHASES_HEFESTO
        else:
            compositional_components_in_phases = COMPOSITIONAL_COMPONENTS_IN_PHASES
        for phase, components in compositional_components_in_phases.items():
            if phase in self.EXCLUDED_PHASES:
                continue

            """if len(components) <= 1:
                # Keep pure phases; they need to stay even if their single component is excluded WHY??
                continue"""

            # If every component for this phase is excluded, mark the phase as excluded
            excluded_for_phase = self.EXCLUDED_COMPONENTS_BY_PHASE.get(phase, set())
            all_excluded = all(comp in excluded_for_phase for comp in components)
            print(f"Checking phase '{phase}' with components: {components} -> all_excluded={all_excluded}")
            if all_excluded:
                """if phase in ['analcime', 'leucite']:
                    print("Can't remove leucite or analcime unless mass is zero because of overlap in names of components.")
                    continue # Special case: We let these be excluded only if the mass is zero"""
                self.EXCLUDED_PHASES.add(phase)
                print(f"Excluding phase '{phase}' as all its components are excluded.")

    def exclude_zero_sum_components(self, data_matrix: np.ndarray, tolerance: float = 1e-10):
        """
        Exclude components with zero columnwise sums from the dataset.
        
        This method analyzes a data matrix and identifies components that have
        zero (or near-zero) values across all samples. Such components are added
        to the EXCLUDED_COMPONENTS_BY_PHASE mapping, and the indexer is repopulated to reflect
        the updated exclusions.
        
        Parameters
        ----------
        data_matrix : np.ndarray
            Data matrix with shape (n_samples, n_features) where n_features must
            match the length of self.headers
        tolerance : float, default=1e-10
            Threshold below which a component sum is considered zero
            
        Raises
        ------
        ValueError
            If data_matrix column count doesn't match headers length
            
        Notes
        -----
        This method modifies the indexer in-place by:
        1. Adding zero-sum components to EXCLUDED_COMPONENTS_BY_PHASE (phase-specific)
        2. Calling _look_for_dead_phases() to check for phases with all components excluded
        3. Calling _repopulate_indexer() to rebuild all indices and mappings
        
        Examples
        --------
        >>> indexer = DatasetIndexer(headers)
        >>> data = np.loadtxt('dataset.csv', delimiter=',', skiprows=1)
        >>> indexer.exclude_zero_sum_components(data)
        """
        # Verify column count matches
        if data_matrix.shape[1] != len(self.headers):
            raise ValueError(
                f"Data matrix has {data_matrix.shape[1]} columns, "
                f"but headers list has {len(self.headers)} entries. "
                f"Column counts must match."
            )
        
        # Calculate columnwise sums
        column_sums = np.sum(np.abs(data_matrix), axis=0)
        
        # Track newly excluded components for reporting
        newly_excluded = []
        
        # Iterate through all phases and their components
        for phase, components_dict in self.MELTS_indices.items():
            # Skip already-excluded phases
            if phase in self.EXCLUDED_PHASES:
                continue
            
            for component, col_idx in components_dict.items():
                # Skip already-excluded components for this phase
                if component in self.EXCLUDED_COMPONENTS_BY_PHASE.get(phase, set()):
                    continue
                
                # Skip state variables (they may legitimately be zero). But look at mass: we need a way to catch pure phases. 
                if component in self.STATE_VARIABLES and 'mass' not in component.lower():
                    continue
                
                # Check if this component's sum is zero
                if column_sums[col_idx] <= tolerance:
                    if 'mass' in component.lower():
                        self.EXCLUDED_PHASES.add(phase) # Exclude phases if their mass is zero
                        newly_excluded.append(f"{phase} (mass zero)")
                    else:
                        self.EXCLUDED_COMPONENTS_BY_PHASE.setdefault(phase, set()).add(component)
                        newly_excluded.append(f"{component}({phase})")
                    print(f"Excluding zero-sum component: {component} from phase {phase}")
        
        # Report summary
        if newly_excluded:
            print(f"\nExcluded {len(newly_excluded)} zero-sum components:")
            for comp_str in newly_excluded[:20]:  # Limit output
                print(f"  - {comp_str}")
            if len(newly_excluded) > 20:
                print(f"  ... and {len(newly_excluded) - 20} more")
        else:
            print("No zero-sum components found.")
        
        # Check for dead phases (all components excluded)
        self._look_for_dead_phases()
        
        # Rebuild all indexer structures
        self._repopulate_indexer()
        
        return newly_excluded

    def exclude_zero_sum_oxides(self, data_matrix: np.ndarray, tolerance: float = 1e-10):
        """
        Remove oxides with zero columnwise sums in Bulk_comp from the active oxide list.
        
        This method analyzes the Bulk_comp phase columns and identifies oxides that have
        zero (or near-zero) values across all samples. Such oxides are removed from
        Oxides, WRkeys, and Elkeys lists. The indexer is then repopulated.
        
        Parameters
        ----------
        data_matrix : np.ndarray
            Data matrix with shape (n_samples, n_features) where n_features must
            match the length of self.headers
        tolerance : float, default=1e-10
            Threshold below which an oxide sum is considered zero
            
        Raises
        ------
        ValueError
            If data_matrix column count doesn't match headers length, or if a
            required element has zero sum in Bulk_comp
            
        Notes
        -----
        This method modifies the indexer in-place by:
        1. Identifying zero-sum oxides in Bulk_comp phase
        2. Removing corresponding oxides from Oxides list
        3. Removing corresponding entries from WRkeys and Elkeys
        4. Rebuilding oxide_dict
        5. Calling _repopulate_indexer() to rebuild all indices and mappings
        
        For iron, the sum of FeO + Fe2O3 is checked rather than each individually.
        
        Examples
        --------
        >>> indexer = DatasetIndexer(headers)
        >>> data = pd.read_csv('dataset.csv').values
        >>> indexer.exclude_zero_sum_oxides(data)
        """
        # Verify column count matches
        if data_matrix.shape[1] != len(self.headers):
            raise ValueError(
                f"Data matrix has {data_matrix.shape[1]} columns, "
                f"but headers list has {len(self.headers)} entries. "
                f"Column counts must match."
            )
        
        # Check if Bulk_comp phase exists in MELTS_indices
        if 'Bulk_comp' not in self.MELTS_indices:
            print("Warning: Bulk_comp phase not found in dataset. Skipping oxide exclusion.")
            return []
        
        # Calculate columnwise sums
        column_sums = np.sum(np.abs(data_matrix), axis=0)
        
        # Track which oxides to remove
        oxides_to_remove = []
        oxide_sum_status = {}  # For reporting
        
        # Special handling for iron: check FeO + Fe2O3 combined
        fe_total_sum = 0.0
        feo_checked = False
        fe2o3_checked = False
        
        # Check each oxide in Bulk_comp
        bulk_comp_components = self.MELTS_indices['Bulk_comp']
        
        for oxide in self.Oxides:
            # Find the column name for this oxide in Bulk_comp
            # Format is typically "wt% {oxide}(Bulk_comp)"
            oxide_col_name = oxide#f"wt% {oxide}" # name is simply the oxide. A bit simpler logic than the liquid phase.
            
            if oxide_col_name not in bulk_comp_components:
                print(f"Warning: Expected column '{oxide_col_name}' not found in Bulk_comp")
                continue
            
            col_idx = bulk_comp_components[oxide_col_name]
            oxide_sum = column_sums[col_idx]
            oxide_sum_status[oxide] = oxide_sum
            
            # Special handling for iron
            if oxide == 'FeO':
                fe_total_sum += oxide_sum
                feo_checked = True
            elif oxide == 'Fe2O3':
                fe_total_sum += oxide_sum
                fe2o3_checked = True
            elif oxide_sum <= tolerance:
                # Non-iron oxide with zero sum
                oxides_to_remove.append(oxide)
        
        # Check iron total (only if both were checked)
        if feo_checked or fe2o3_checked:
            if fe_total_sum <= tolerance:
                # Total iron is zero
                if 'FeO' in self.Oxides:
                    oxides_to_remove.append('FeO')
                if 'Fe2O3' in self.Oxides:
                    oxides_to_remove.append('Fe2O3')
                print(f"Total iron (FeO + Fe2O3) sum: {fe_total_sum:.2e} - excluding both")
        
        # Check if any required elements would be removed
        elements_to_remove = []
        for oxide in oxides_to_remove:
            # Map oxide back to element
            for el, ox in ELEMENT_TO_OXIDE.items():
                if ox == oxide:
                    if el in self.Elkeys:
                        elements_to_remove.append(el)
                        if el in REQUIRED_ELEMENTS:
                            raise ValueError(
                                f"Cannot exclude oxide {oxide} (element {el}) because it is a required element. "
                                f"Required elements are: {REQUIRED_ELEMENTS}. "
                                f"The Bulk_comp column for {oxide} has zero sum ({oxide_sum_status.get(oxide, 0):.2e})."
                            )
        
        # Report what will be removed
        if oxides_to_remove:
            print(f"\nRemoving {len(oxides_to_remove)} zero-sum oxides from Bulk_comp:")
            for oxide in oxides_to_remove:
                print(f"  - {oxide}: sum = {oxide_sum_status.get(oxide, 0):.2e}")
            print(f"\nCorresponding elements to remove: {elements_to_remove}")
        else:
            print("No zero-sum oxides found in Bulk_comp.")
            return []
        
        # Remove oxides from lists
        self.Oxides = [ox for ox in self.Oxides if ox not in oxides_to_remove]
        
        # Rebuild WRkeys (exclude Fe2O3, which is in Oxides but not WRkeys)
        self.WRkeys = [ox for ox in self.Oxides if ox != 'Fe2O3']
        
        # Remove corresponding elements from Elkeys
        self.Elkeys = [el for el in self.Elkeys if el not in elements_to_remove]
        
        # Rebuild oxide_dict with new indices
        self.oxide_dict = {}
        for i, ox in enumerate(self.Oxides):
            self.oxide_dict[ox] = i
        
        print(f"\nUpdated lists:")
        print(f"  - Oxides: {len(self.Oxides)} ({self.Oxides})")
        print(f"  - WRkeys: {len(self.WRkeys)} ({self.WRkeys})")
        print(f"  - Elkeys: {len(self.Elkeys)} ({self.Elkeys})")
        
        self._look_for_illegal_oxides()
        self._look_for_dead_phases()

        # Repopulate indexer with new oxide/element lists
        self._repopulate_indexer()
        
        return oxides_to_remove

    def table_update(self, data_matrix: np.ndarray, tolerance: float = 1e-10):
        self.exclude_zero_sum_components(data_matrix, tolerance)
        self.exclude_zero_sum_oxides(data_matrix, tolerance)

    def _build_label_indices(self):
        raise NotImplementedError("ML label indices are built by MLIndexer (available at self.ml_indexer)")

    def _build_phase_dictionaries(self):
        raise NotImplementedError("ML phase dictionaries are built by MLIndexer (available at self.ml_indexer)")

    def _build_phase_mappings(self):
        raise NotImplementedError("ML phase mappings are built by MLIndexer (available at self.ml_indexer)")

    def _build_component_mappings(self):
        raise NotImplementedError("Component mappings are built by MLIndexer (available at self.ml_indexer)")
    
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