"""
ML Indexer for nMELTS.

Takes a DatasetIndexer and creates ML-ready indexers and transformation matrices.
Transformation matrices are built dynamically based on the components in label_names.
"""

import numpy as np
import pandas as pd
import molmass as ms
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any

# Make torch optional
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from .indexer import DatasetIndexer


class MLIndexer:
    """
    ML-ready indexer that builds transformation matrices and mappings from a DatasetIndexer.
    
    Takes the label_indices from DatasetIndexer and creates:
    - Phase-to-component mappings
    - Transformation matrices (compToOx, oxToEl, etc.)
    - ML-ready data structures
    """
    
    def __init__(self, dataset_indexer: DatasetIndexer, projections_dir: Optional[Path] = None):
        """
        Initialize ML indexer from a DatasetIndexer.
        
        Parameters
        ----------
        dataset_indexer : DatasetIndexer
            The dataset indexer containing label_indices, label_names, etc.
        projections_dir : Path, optional
            Directory containing transformation CSV files. Defaults to config/projections.
        """
        self.dataset_indexer = dataset_indexer
        
        # Extract label mappings from dataset_indexer
        self.label_indices = dataset_indexer.label_indices
        self.label_names = dataset_indexer.label_names
        self.detail_label_indices = dataset_indexer.detail_label_indices
        self.label_indices_comp = dataset_indexer.label_indices_comp
        self.all_phases = dataset_indexer.all_phases
        self.mass_phasedict = dataset_indexer.mass_phasedict
        self.comp_phasedict = dataset_indexer.comp_phasedict
        self.compositionally_variable_phases = dataset_indexer.compositionally_variable_phases
        
        # Calculate dimensions
        self.ncomps = dataset_indexer.ncomps
        self.ncompsVaried = dataset_indexer.ncompsVaried
        self.nphases = dataset_indexer.nphases
        
        # Set up projections directory
        if projections_dir is None:
            _CONFIG_DIR = Path(__file__).parent
            projections_dir = _CONFIG_DIR / 'projections'
        self.projections_dir = Path(projections_dir)
        
        # Extract elements and oxides from label_names (for melts-liquid)
        self._extract_elements_and_oxides()
        
        # Load and build transformation matrices
        self._load_transformation_matrices()
        
        # Build ML-ready mappings
        self._build_ml_mappings()
    
    def _extract_elements_and_oxides(self):
        """Extract Elkeys and WRkeys from label_names (melts-liquid components)."""
        # Get elements from melts-liquid label_names (they should be Elkeys)
        if 'melts-liquid' in self.label_indices:
            melts_start = self.label_indices['melts-liquid'][0]
            melts_end = self.label_indices['melts-liquid'][-1] + 1
            self.Elkeys_used = self.label_names[melts_start:melts_end]
        else:
            # Fallback to DatasetIndexer's Elkeys if no melts-liquid
            self.Elkeys_used = self.dataset_indexer.Elkeys
        
        # Determine which oxides are needed based on Elkeys
        # Map: Si->SiO2, Ti->TiO2, Al->Al2O3, Fe->FeO/Fe2O3, etc.
        element_to_oxide = {
            'Si': 'SiO2', 'Ti': 'TiO2', 'Al': 'Al2O3', 'Fe': 'FeO',
            'Mg': 'MgO', 'Ca': 'CaO', 'Na': 'Na2O', 'K': 'K2O',
            'P': 'P2O5', 'H': 'H2O', 'Cr': 'Cr2O3', 'Mn': 'MnO', 'Ni': 'NiO'
        }
        
        # Build WRkeys_used (oxides without Fe2O3)
        self.WRkeys_used = []
        for el in self.Elkeys_used:
            if el in element_to_oxide:
                ox = element_to_oxide[el]
                if ox not in self.WRkeys_used:
                    self.WRkeys_used.append(ox)
        
        # Oxides_used includes Fe2O3 if Fe is present
        self.Oxides_used = self.WRkeys_used.copy()
        if 'Fe' in self.Elkeys_used and 'Fe2O3' not in self.Oxides_used:
            self.Oxides_used.append('Fe2O3')
    
    def _load_transformation_matrices(self):
        """Load transformation CSV files and build matrices based on label_names."""
        try:
            # Load compToOx.csv (components to oxides)
            compToOx_path = self.projections_dir / 'compToOx.csv'
            compToOx_df = pd.read_csv(compToOx_path, index_col=0)
            
            # Load PxSp_Comp_Transform.csv (component transformation matrix)
            PxSpTransform_path = self.projections_dir / 'PxSp_Comp_Transform.csv'
            PxSpTransform_df = pd.read_csv(PxSpTransform_path, index_col=0)
            
            # Load OxToEl.csv (oxides to elements)
            oxToEl_path = self.projections_dir / 'OxToEl.csv'
            oxToEl_df = pd.read_csv(oxToEl_path, index_col=0)
            
            # Extract components from label_names (excluding melts-liquid elements)
            # Components are phase-specific (e.g., "forsterite", "fayalite")
            # Elements are from melts-liquid (e.g., "Si", "Ti", "Al")
            components_in_labels = []
            for name in self.label_names:
                # Skip elements (they're in Elkeys)
                if name not in self.dataset_indexer.Elkeys:
                    components_in_labels.append(name)
            
            # Build compToOx matrix for components in label_names
            # Format in CSV: "component : phase" -> oxides
            self.compToOxLoad = None
            self.compToOx = None
            self.PxSpTransform = None
            
            if len(components_in_labels) > 0:
                # Find matching rows in compToOx_df
                # Need to match "component : phase" format
                comp_rows = []
                comp_full_names = []  # Store full "component : phase" names
                comp_names_only = []  # Store just component names
                
                # Build a mapping from component name to possible phase combinations
                comp_to_phases = {}
                for phase in self.all_phases:
                    if phase in self.detail_label_indices:
                        for comp in self.detail_label_indices[phase].keys():
                            if comp not in comp_to_phases:
                                comp_to_phases[comp] = []
                            comp_to_phases[comp].append(phase)
                
                # Match CSV rows to our components
                for idx in compToOx_df.index:
                    # Extract component and phase from "component : phase"
                    parts = idx.split(' : ')
                    if len(parts) == 2:
                        comp_name = parts[0].strip()
                        phase_name = parts[1].strip()
                        
                        # Check if this component is in our labels
                        if comp_name in components_in_labels:
                            # Check if this phase matches one of our phases for this component
                            if comp_name in comp_to_phases and phase_name in comp_to_phases[comp_name]:
                                comp_rows.append(compToOx_df.loc[idx].values)
                                comp_full_names.append(idx)
                                comp_names_only.append(comp_name)
                
                if len(comp_rows) > 0:
                    # Build compToOxLoad: components x oxides
                    compToOxLoad_data = np.array(comp_rows, dtype=np.float32)
                    # Select only the oxides we're using
                    oxide_cols = [col for col in compToOx_df.columns if col in self.Oxides_used]
                    if len(oxide_cols) > 0:
                        oxide_indices = [compToOx_df.columns.get_loc(col) for col in oxide_cols]
                        self.compToOxLoad = compToOxLoad_data[:, oxide_indices]
                        
                        # Build PxSpTransform for these components
                        # PxSpTransform has same index/column format as compToOx
                        comp_rows_px = []
                        comp_cols_px = []
                        
                        for comp_full in comp_full_names:
                            if comp_full in PxSpTransform_df.index:
                                comp_rows_px.append(PxSpTransform_df.loc[comp_full].values)
                        
                        # Get columns that match our components
                        for comp_full in comp_full_names:
                            if comp_full in PxSpTransform_df.columns:
                                comp_cols_px.append(comp_full)
                        
                        if len(comp_rows_px) > 0 and len(comp_cols_px) > 0:
                            # Build square matrix
                            col_indices = [PxSpTransform_df.columns.get_loc(col) for col in comp_cols_px]
                            PxSp_data = np.array(comp_rows_px, dtype=np.float32)
                            # Only use columns that match our rows
                            if len(comp_cols_px) <= PxSp_data.shape[1]:
                                self.PxSpTransform = PxSp_data[:, col_indices[:len(comp_rows_px)]]
                                
                                # Compute compToOx = inv(PxSpTransform) @ compToOxLoad
                                if self.PxSpTransform.shape[0] == self.PxSpTransform.shape[1] and self.PxSpTransform.shape[0] == self.compToOxLoad.shape[0]:
                                    try:
                                        self.compToOx = np.linalg.inv(self.PxSpTransform) @ self.compToOxLoad
                                    except np.linalg.LinAlgError:
                                        # If matrix is singular, use pseudoinverse
                                        self.compToOx = np.linalg.pinv(self.PxSpTransform) @ self.compToOxLoad
            
            # Build oxToEl matrix: Elkeys (rows) x WRkeys (columns)
            # oxToEl_df has elements as rows, oxides as columns
            el_rows = []
            for el in self.Elkeys_used:
                if el in oxToEl_df.index:
                    el_rows.append(oxToEl_df.loc[el].values)
                else:
                    # If element not found, create zero row
                    el_rows.append(np.zeros(len(oxToEl_df.columns), dtype=np.float32))
            
            if len(el_rows) > 0:
                # Select only the oxides we're using (WRkeys)
                oxide_cols_el = [col for col in oxToEl_df.columns if col in self.WRkeys_used]
                if len(oxide_cols_el) > 0:
                    oxide_indices_el = [oxToEl_df.columns.get_loc(col) for col in oxide_cols_el]
                    elToOx_data = np.array(el_rows, dtype=np.float32)
                    # Build ElToOx: (Elkeys, WRkeys) - elements as rows, oxides as columns
                    self.ElToOx = elToOx_data[:, oxide_indices_el]
                    # Also keep oxToEl for backward compatibility: (WRkeys, Elkeys) - transpose
                    self.oxToEl = self.ElToOx.T
                else:
                    self.ElToOx = None
                    self.oxToEl = None
            else:
                self.ElToOx = None
                self.oxToEl = None
            
            # Build molar mass matrices
            self.MM = np.diag([ms.Formula(ox).mass for ox in self.Oxides_used]).astype(np.float32)
            self.Minv = np.diag([1.0 / ms.Formula(ox).mass for ox in self.Oxides_used]).astype(np.float32)
            self.Mtot = np.array([ms.Formula(ox).mass for ox in self.Oxides_used], dtype=np.float32).reshape(-1, 1)
            
        except FileNotFoundError as e:
            import warnings
            warnings.warn(f"Could not load transform CSV files from {self.projections_dir}: {e}", UserWarning)
            # Set to None if files don't exist
            self.compToOxLoad = None
            self.compToOx = None
            self.PxSpTransform = None
            self.ElToOx = None
            self.oxToEl = None
            self.MM = None
            self.Minv = None
            self.Mtot = None
    
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
        self.comp_map = {}
        j = 0
        k = 0
        
        for phase in self.all_phases:
            if phase in self.label_indices:
                n_components = len(self.label_indices[phase])
                if n_components > 1:
                    comp_list = np.arange(k, k + n_components, dtype=int)
                    self.comp_map[phase] = comp_list
                    k += n_components
        
        comp_binaries_list = []
        comp_mappings_list = []
        j = 0
        
        for phase in self.all_phases:
            if phase in self.label_indices:
                n_components = len(self.label_indices[phase])
                if n_components > 1:
                    comp_binaries_list.append(self.all_phases.index(phase))
                    comp_mappings_list.extend([j] * n_components)
                    j += 1
        
        self.comp_binaries = np.array(comp_binaries_list, dtype=int) if comp_binaries_list else np.array([], dtype=int)
        self.comp_mappings = np.zeros((j, len(comp_mappings_list)), dtype=np.float32) if comp_mappings_list else np.zeros((0, 0), dtype=np.float32)
        
        for col, row in enumerate(comp_mappings_list):
            self.comp_mappings[row, col] = 1.0
        
        # Build boolTransCompToOx (if compToOx exists)
        if self.compToOx is not None:
            self.boolTransCompToOx = np.copy(self.compToOx)
            # Add Fe2O3 column to FeO if Fe2O3 exists
            if 'Fe2O3' in self.Oxides_used and 'FeO' in self.WRkeys_used:
                feo_idx = self.WRkeys_used.index('FeO')
                fe2o3_idx = self.Oxides_used.index('Fe2O3')
                if feo_idx < self.boolTransCompToOx.shape[1] and fe2o3_idx < self.boolTransCompToOx.shape[1]:
                    self.boolTransCompToOx[:, feo_idx] += self.boolTransCompToOx[:, fe2o3_idx]
            # Convert to boolean (non-zero -> 1)
            self.boolTransCompToOx = (self.boolTransCompToOx != 0).astype(int)
        else:
            self.boolTransCompToOx = None
