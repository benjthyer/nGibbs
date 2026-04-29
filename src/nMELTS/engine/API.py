"""
High-level API for HeFESTo adiabat modeling.

Provides user-friendly interfaces for:
- Getting temperatures from isentropic states
- Computing isentropic adiabats along pressure transects
- Parsing and converting input compositions
- Managing emulator ensembles (isothermal, isentropic, temperature models)
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union, Dict, Any

from .NN import _load_temperature_model
from .emulator import NN_MELTS
from ..utils.math_utils import Normalizer
from .EOS_arithmetic import get_hefesto_physub_context, compute_physub_bulk_matrix

aliases = {
    'T': 'T(K)(System_main)',
    'T(K)': 'T(K)(System_main)',
    'Temperature': 'T(K)(System_main)',
    'Temperature(K)': 'T(K)(System_main)',
    'temperature': 'T(K)(System_main)',
    'temperature(K)': 'T(K)(System_main)',
    'S': 'S(J/mol/K)(System_main)',
    'S(J/mol/K)': 'S(J/mol/K)(System_main)',
    'Entropy': 'S(J/mol/K)(System_main)',
    'Entropy(J/mol/K)': 'S(J/mol/K)(System_main)',
    'entropy': 'S(J/mol/K)(System_main)',
    'entropy(J/mol/K)': 'S(J/mol/K)(System_main)',
    'P': 'P(GPa)(System_main)',
    'P(GPa)': 'P(GPa)(System_main)',
    'Pressure': 'P(GPa)(System_main)',
    'Pressure(GPa)': 'P(GPa)(System_main)',
    'pressure': 'P(GPa)(System_main)',
    'pressure(GPa)': 'P(GPa)(System_main)',
}

class InputParser:
    """
    Parser for input composition data.
    
    Handles:
    - Detection of oxide vs element input
    - Iron speciation when oxygen is present
    - Column reordering via emulator interface
    """
    
    def __init__(self, emulator: NN_MELTS):
        """
        Initialize parser with an emulator reference.
        
        Parameters
        ----------
        emulator : NN_MELTS
            Reference emulator for column mapping and indexing
        """
        self.emulator = emulator
        self.Elkeys = list(emulator.Elkeys)
        self.Oxides = list(emulator.Oxides)
    
    def parse_composition(
        self,
        table: Union[np.ndarray, torch.Tensor],
        headers: Optional[Sequence[str]] = None,
        composition_space: Optional[str] = None,
    ) -> Tuple[np.ndarray, Sequence[str], str]:
        """
        Parse input composition table and detect composition space.
        
        If 'O' (oxygen) is present in elements and Fe is present, will
        recalculate elemental mole fractions to differentiate Fe2+ and Fe3+.
        
        Parameters
        ----------
        table : array-like
            Input table with intensive conditions + composition
        headers : sequence of str, optional
            Column headers. Required if table lacks column labels.
        composition_space : str, optional
            If provided, uses this ('elements' or 'oxides').
            Otherwise, auto-detects from headers.
        
        Returns
        -------
        tuple
            (parsed_table, headers_out, composition_space)
            parsed_table is np.ndarray with Fe3+ column added if applicable.
            headers_out contains updated headers.
            composition_space is 'elements' or 'oxides'.
        """
        # Convert to numpy if needed
        if torch.is_tensor(table):
            values = table.detach().cpu().numpy()
        else:
            values = np.asarray(table, dtype=np.float32)
        
        if values.ndim != 2:
            raise ValueError(f"table must be 2D, got shape {values.shape}")
        
        # Determine headers
        if headers is None:
            if hasattr(table, 'columns'):
                headers = [str(col) for col in table.columns]
            else:
                raise ValueError(
                    "headers must be provided when table lacks column labels"
                )
        else:
            headers = [str(h) for h in headers]
        
        if len(values[0]) != len(headers):
            raise ValueError(
                f"Header count ({len(headers)}) does not match "
                f"table columns ({values.shape[1]})"
            )
        
        # Auto-detect composition space if not provided
        if composition_space is None:
            composition_space = self._detect_composition_space(headers)
        
        composition_space = composition_space.lower()
        if composition_space not in ['elements', 'oxides']:
            raise ValueError("composition_space must be 'elements' or 'oxides'")
        
        # Handle iron speciation for elements with oxygen.
        # Replace O with Fe3 so the emulator sees the expected feature space.
        headers_out = list(headers)
        if composition_space == 'elements':
            if 'O' in headers:
                values, headers_out = self._add_ferric_column(values, headers)
        
        return values, headers_out, composition_space
    
    def _detect_composition_space(self, headers: Sequence[str]) -> str:
        """
        Detect whether input is oxides or elements.
        
        Parameters
        ----------
        headers : sequence of str
            Column headers
        
        Returns
        -------
        str
            'elements' or 'oxides'
        """
        header_set = set(str(h).strip() for h in headers)
        
        # Check for oxide indicators
        oxide_keywords = {'SiO2', 'FeO', 'Fe2O3', 'MgO', 'CaO', 'Al2O3', 'Na2O', 'K2O'}
        if any(ox in header_set for ox in oxide_keywords):
            return 'oxides'
        
        # Check for element indicators
        element_keywords = {'Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'K', 'Cr', 'O'}
        if any(el in header_set for el in element_keywords):
            return 'elements'
        
        # Default to elements
        return 'elements'
    
    def _add_ferric_column(
        self,
        values: np.ndarray,
        headers: Sequence[str],
    ) -> Tuple[np.ndarray, Sequence[str]]:
        """
        Replace O with Fe3+ by calculating ferric iron from charge balance.
        
        Implements charge-balance oxygen speciation to calculate Fe3+/Fe2+ split
        based on total Fe and O available in the composition. Uses the approach
        inverse to _oxide_wt_to_element_moles in HeFESTo_functions.py.
        
        Parameters
        ----------
        values : np.ndarray
            (N, F) input table with elemental composition
        headers : sequence of str
            Column headers matching input table
        
        Returns
        -------
        tuple
            (table_with_fe3, updated_headers) where O has been replaced by Fe3+
        """
        # Get column indices for Fe and O
        header_map = {str(h).strip(): i for i, h in enumerate(headers)}
        
        if 'Fe' not in header_map or 'O' not in header_map:
            raise ValueError("Elemental inputs with oxygen must include both 'Fe' and 'O'")
        
        fe_idx = header_map['Fe']
        o_idx = header_map['O']
        
        # Extract Fe and O columns
        fe_total = values[:, fe_idx].astype(np.float32)  # Total Fe in moles
        o_total = values[:, o_idx].astype(np.float32)    # Total O in moles
        
        element_moles: Dict[str, np.ndarray] = {
            'Fe': fe_total,
            'O': o_total,
        }
        for elem in ('Si', 'Mg', 'Ca', 'Al', 'Na', 'K', 'Cr'):
            if elem in header_map:
                element_moles[elem] = values[:, header_map[elem]].astype(np.float32)

        fe3_moles = speciate_iron_from_charge_balance(element_moles)

        # Clip to valid range: 0 <= Fe3+ <= Fe_total
        fe3_moles = np.clip(fe3_moles, 0.0, fe_total).astype(np.float32)

        o_idx = header_map['O']
        updated_values = np.delete(values, o_idx, axis=1).astype(np.float32, copy=False)
        updated_values = np.insert(updated_values, o_idx, fe3_moles, axis=1).astype(np.float32, copy=False)

        updated_headers = list(headers)
        updated_headers[o_idx] = 'Fe3'
        return updated_values, updated_headers


def speciate_iron_from_charge_balance(
    element_moles: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Speciate Fe into Fe2+ and Fe3+ based on charge balance with oxygen.
    
    Inverse of the oxide-to-element conversion in HeFESTo_functions.py.
    Given elemental molar composition with Fe and O, calculates Fe3+ required
    for charge neutrality based on all cation oxidation states.
    
    Charge balance equation:
    sum(ox_state_i * cation_i) + 2*Fe2+ + 3*Fe3+ = 2*O_total
    
    Where Fe2+ + Fe3+ = Fe_total
    
    Solving for Fe3+:
    Fe3+ = 2*O_total - sum(ox_state_i * cation_i) - 2*Fe_total
    
    Parameters
    ----------
    element_moles : dict[str, np.ndarray]
        Dictionary mapping element names to molar amounts (1D or 2D arrays)
        Must include 'Fe' and 'O'. Other cations ('Si', 'Mg', 'Ca', 'Al', 'Na', 'K', 'Cr')
        are optional and assumed 0 if missing.
    
    Returns
    -------
    np.ndarray
        Fe3+ molar amounts, clipped to [0, Fe_total]
    
    Raises
    ------
    KeyError
        If 'Fe' or 'O' missing from element_moles dict
    """
    if 'Fe' not in element_moles or 'O' not in element_moles:
        raise KeyError("element_moles must include 'Fe' and 'O'")
    
    fe_total = np.asarray(element_moles['Fe'], dtype=np.float32)
    o_total = np.asarray(element_moles['O'], dtype=np.float32)
    
    # Ensure arrays are at least 1D
    if fe_total.ndim == 0:
        fe_total = fe_total.reshape(1)
    if o_total.ndim == 0:
        o_total = o_total.reshape(1)
    
    # Calculate cation charge contributions
    cation_charges = np.zeros_like(fe_total, dtype=np.float32)
    
    cation_specs = {
        'Si': 4,
        'Mg': 2,
        'Ca': 2,
        'Al': 3,
        'Na': 1,
        'K': 1,
        'Cr': 3,
    }
    
    for elem, ox_state in cation_specs.items():
        if elem in element_moles:
            moles = np.asarray(element_moles[elem], dtype=np.float32)
            if moles.ndim == 0:
                moles = moles.reshape(1)
            cation_charges = cation_charges + ox_state * moles
    
    # Solve for Fe3+ from charge balance
    fe3_moles = 2.0 * o_total - cation_charges - 2.0 * fe_total
    
    # Clip to valid range
    fe3_moles = np.clip(fe3_moles, 0.0, fe_total)
    
    return fe3_moles.astype(np.float32)


def create_isentrope_design_matrix(
    temperatures: Union[np.ndarray, torch.Tensor],
    pressures: Union[np.ndarray, torch.Tensor],
    base_features: Union[np.ndarray, torch.Tensor],
    temperature_idx: int,
    pressure_idx: int,
) -> np.ndarray:
    """
    Create a design matrix for isentropic adiabat exploration.
    
    Generates a matrix of shape (T*P, F) where each row represents a unique
    (temperature, pressure) combination, with all other features held constant.
    
    Parameters
    ----------
    temperatures : array-like
        1D array of temperatures (K)
    pressures : array-like
        1D array of pressures (Pa or consistent units)
    base_features : array-like
        (T, F) or (1, F) base feature set to tile/repeat
    temperature_idx : int
        Column index for temperature in features
    pressure_idx : int
        Column index for pressure in features
    
    Returns
    -------
    np.ndarray
        Design matrix of shape (T*P, F)
    """
    temperatures = np.asarray(temperatures, dtype=np.float32).flatten()
    pressures = np.asarray(pressures, dtype=np.float32).flatten()
    base_features = np.asarray(base_features, dtype=np.float32)
    
    if base_features.ndim == 1:
        base_features = base_features.reshape(1, -1)
    
    n_temps = temperatures.shape[0]
    n_press = pressures.shape[0]
    n_feat = base_features.shape[1]
    
    # If base_features has multiple rows, must match temperatures
    if base_features.shape[0] != 1 and base_features.shape[0] != n_temps:
        raise ValueError(
            f"base_features must have 1 or {n_temps} rows, "
            f"got {base_features.shape[0]}"
        )
    
    # Create grid: repeat base_features for each (T, P) pair
    design = np.tile(base_features, (n_temps * n_press, 1))
    
    # Set temperature column: repeat each temp n_press times
    design[:, temperature_idx] = np.repeat(temperatures, n_press)
    
    # Set pressure column: tile pressures for each temperature
    design[:, pressure_idx] = np.tile(pressures, n_temps)
    
    return design


class HeFESToAPI:
    """
    High-level API for HeFESTo Emulator.
    
    Manages:
    - Isothermal and isentropic emulators
    - Temperature prediction model (FCNN)
    - Input parsing and composition handling
    - Adiabat computation workflows
    """
    
    def __init__(
        self,
        isothermal_model_path: Union[str, Path],
        isentropic_model_path: Union[str, Path],
        temperature_model_path: Union[str, Path],
        device: str = 'cpu',
        verbose: bool = False,
    ):
        """
        Initialize the adiabat API with emulator and temperature models.
        
        Parameters
        ----------
        isothermal_model_path : str or Path
            Path to isothermal emulator checkpoint
        isentropic_model_path : str or Path
            Path to isentropic emulator checkpoint
        temperature_model_path : str or Path
            Path to temperature FCNN checkpoint
        device : str, default='cpu'
            Torch device ('cpu' or 'cuda')
        verbose : bool, default=False
            Print initialization messages
        """
        if verbose:
            print("[INFO] Initializing HeFESToAdiabatAPI...")
        
        self.device = torch.device(device)
        self.verbose = verbose
        
        # Load emulators
        if verbose:
            print(f"  Loading isothermal emulator: {isothermal_model_path}")
        self.isothermal_emulator = self._load_emulator(
            isothermal_model_path, device
        )
        
        if verbose:
            print(f"  Loading isentropic emulator: {isentropic_model_path}")
        self.isentropic_emulator = self._load_emulator(
            isentropic_model_path, device
        )
        
        # Load temperature model
        if verbose:
            print(f"  Loading temperature FCNN: {temperature_model_path}")
        self._setup_temperature_model(temperature_model_path)
        
        # Load HeFESTo physub context
        if verbose:
            print("  Loading HeFESTo physub context...")
        self.hefesto_context = get_hefesto_physub_context()
        
        # Setup parser
        self.parser = InputParser(self.isothermal_emulator)
        
        if verbose:
            print("[INFO] HeFESToAdiabatAPI initialized successfully.")
    
    @staticmethod
    def _load_emulator(model_path: Union[str, Path], device: str) -> NN_MELTS:
        """Load and wrap a checkpoint as NN_MELTS emulator."""
        from ..engine.NN import rebuild_MELTS_model
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = rebuild_MELTS_model(str(model_path))
        return NN_MELTS(model, cuda=(device == 'cuda'))
    
    def _setup_temperature_model(self, checkpoint_path: Union[str, Path]) -> None:
        """Load temperature FCNN and setup normalizers."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Temperature model not found: {checkpoint_path}")
        
        # Load model
        model, payload, x_min, x_range, y_min, y_range = _load_temperature_model(
            checkpoint_path, self.device
        )
        
        self.temperature_model = model
        self.temperature_payload = payload
        
        # Setup normalizers as torch tensors
        self.temp_input_normalizer = Normalizer(
            torch.tensor(x_min, dtype=torch.float32),
            torch.tensor(x_range, dtype=torch.float32),
            cuda=(self.device.type == 'cuda')
        )
        self.temp_output_normalizer = Normalizer(
            torch.tensor(y_min, dtype=torch.float32),
            torch.tensor(y_range, dtype=torch.float32),
            cuda=(self.device.type == 'cuda')
        )
    
    def get_T(
        self,
        features: Union[np.ndarray, torch.Tensor],
        normalize_features: bool = True,
    ) -> torch.Tensor:
        """
        Get temperature from isentropic emulator output.
        
        Lightweight frontend that:
        1. Passes features through isentropic emulator
        2. Builds temperature model inputs from emulator outputs
        3. Passes through temperature FCNN
        4. Returns temperature predictions
        
        Parameters
        ----------
        features : array-like
            Input features (B, F) with intensive variables + composition
        normalize_features : bool, default=True
            Whether to normalize input features
        
        Returns
        -------
        torch.Tensor
            Temperatures (B,) in Kelvin
        """
        features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        
        # Forward through isentropic emulator
        iso_output = self.isentropic_emulator.forwardMB(
            features, Normalize=normalize_features, outputs=['component_moles']
        )
        component_moles = iso_output['component_moles']
        
        # Build temperature model inputs
        # This depends on the temperature model's training design
        # For now, use component moles as input
        temp_input = component_moles
        
        # Normalize for temperature model
        temp_input_norm = self.temp_input_normalizer.norm(temp_input)
        
        # Forward through temperature FCNN
        with torch.no_grad():
            temp_output_norm = self.temperature_model(temp_input_norm)
        
        # Denormalize output
        temperature = self.temp_output_normalizer.denorm(temp_output_norm)
        
        return temperature.squeeze(-1)
    
    def get_isentrope(
        self,
        features: Union[np.ndarray, torch.Tensor, Sequence],
        pressures: Union[np.ndarray, torch.Tensor],
        potential_temperatures: Optional[Union[np.ndarray, torch.Tensor]] = None,
        batch_size: int = 2**16,
        normalize_features: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute isentropic adiabats along pressure transects.
        
        Process:
        1. Use isothermal model to compute entropy at each input state (P=0)
        2. Create grid of (T, P) states for entropy-constrained search
        3. Forward through isentropic model for all (T, P) combinations
        4. Predict temperatures via temperature FCNN
        5. Return organized (T, P) adiabats
        
        Parameters
        ----------
        features : array-like
            Input features (T, F) or (N, F). If multiple rows:
            - Each row is a distinct composition/condition
            - potential_temperatures must match length or be None
        pressures : array-like
            1D array of pressures for adiabat search
        potential_temperatures : array-like, optional
            Potential temperatures (K) for each feature row.
            If None, will be computed from isothermal model at reference pressure.
            If features is (1, F), can be array of any length.
            If features is (N, F) with N > 1, must match N or be None.
        batch_size : int, default=2**16
            Batch size for staged evaluation
        normalize_features : bool, default=True
            Whether to normalize input features
        
        Returns
        -------
        tuple of torch.Tensor
            (temperatures, pressures_grid)
            temperatures shape: (N_input, len(pressures))
            pressures_grid shape: (N_input, len(pressures))
            Each row represents an isentrope along pressure
        """
        features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        pressures = np.asarray(pressures, dtype=np.float32).flatten()
        
        if features.ndim != 2:
            raise ValueError(f"features must be 2D, got shape {features.shape}")
        
        n_input = features.shape[0]
        n_press = len(pressures)
        
        # Validate potential_temperatures
        if potential_temperatures is not None:
            potential_temperatures = np.asarray(
                potential_temperatures, dtype=np.float32
            ).flatten()
            if potential_temperatures.shape[0] not in [1, n_input]:
                raise ValueError(
                    f"potential_temperatures must have length 1 or {n_input}, "
                    f"got {potential_temperatures.shape[0]}"
                )
        
        if self.verbose:
            print(f"[INFO] Computing isentropes:")
            print(f"  Input features: {features.shape}")
            print(f"  Pressures: {n_press} values")
            print(f"  Total states to evaluate: {n_input * n_press}")
        
        # Step 1: Get entropy values from isothermal model at reference pressure
        # Use isothermal model with features to extract entropy-related state
        # For now, we'll get phase compositions which are entropy-constrained
        if self.verbose:
            print("  [1/4] Evaluating isothermal model for entropy constraint...")
        
        iso_therm_output = self.isothermal_emulator.forwardMB(
            features, Normalize=normalize_features,
            outputs=['component_moles', 'phase_moles']
        )
        component_moles_iso = iso_therm_output['component_moles']
        
        # Step 2: Create design matrix for (T, P) grid
        if self.verbose:
            print("  [2/4] Creating (T, P) design matrix...")
        
        # Get feature indices
        iso_feat_names = list(self.isothermal_emulator.ml_indexer.featureNames)
        temp_idx = iso_feat_names.index('Temperature')
        pressure_idx = iso_feat_names.index('Pressure')
        
        # If single input, broadcast to allow multiple potential temperatures
        if n_input == 1 and potential_temperatures is not None:
            base_feat = features.repeat(potential_temperatures.shape[0], 1)
            potential_temperatures_used = potential_temperatures
        else:
            base_feat = features
            if potential_temperatures is not None:
                potential_temperatures_used = potential_temperatures
            else:
                potential_temperatures_used = None
        
        # Create grid (broadcast each feature to all pressures)
        design_matrix = self._create_iso_design_matrix(
            base_feat,
            pressures,
            temp_idx,
            pressure_idx,
            potential_temperatures_used,
        )
        
        # Step 3: Forward through isentropic model
        if self.verbose:
            print("  [3/4] Evaluating isentropic model (staged)...")
        
        design_tensor = torch.as_tensor(
            design_matrix, dtype=torch.float32, device=self.device
        )
        
        iso_outputs = self._staged_forward(
            self.isentropic_emulator.forwardMB,
            design_tensor,
            batch_size,
            Normalize=normalize_features,
            outputs=['component_moles']
        )
        pred_component_moles = iso_outputs[0]
        
        # Step 4: Predict temperatures
        if self.verbose:
            print("  [4/4] Predicting temperatures (staged)...")
        
        temp_input_norm = self.temp_input_normalizer.norm(pred_component_moles)
        
        with torch.no_grad():
            temp_output_norm = self._staged_forward(
                self.temperature_model,
                temp_input_norm,
                batch_size,
            )
            if not isinstance(temp_output_norm, (list, tuple)):
                temp_output_norm = [temp_output_norm]
        
        temperatures_norm = temp_output_norm[0]
        temperatures_pred = self.temp_output_normalizer.denorm(temperatures_norm)
        temperatures_pred = temperatures_pred.squeeze(-1)
        
        # Step 5: Reorganize output to (N_input, P) shape
        temperatures_grid = temperatures_pred.reshape(n_input, n_press)
        pressures_grid = torch.tensor(
            np.tile(pressures, (n_input, 1)), device=self.device, dtype=torch.float32
        )
        
        if self.verbose:
            print(f"  Output shapes: {temperatures_grid.shape}")
        
        return temperatures_grid, pressures_grid
    
    def _create_iso_design_matrix(
        self,
        base_features: torch.Tensor,
        pressures: np.ndarray,
        temp_idx: int,
        pressure_idx: int,
        potential_temperatures: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Create design matrix for isentropic evaluation."""
        base_feat_np = base_features.detach().cpu().numpy()
        n_input = base_feat_np.shape[0]
        n_feat = base_feat_np.shape[1]
        n_press = len(pressures)
        
        # Create output design matrix
        design = np.zeros(
            (n_input * n_press, n_feat), dtype=np.float32
        )
        
        for i in range(n_input):
            start_idx = i * n_press
            end_idx = (i + 1) * n_press
            
            # Tile base features for all pressures
            design[start_idx:end_idx] = np.tile(
                base_feat_np[i:i+1], (n_press, 1)
            )
            
            # Set pressure column
            design[start_idx:end_idx, pressure_idx] = pressures
            
            # Set temperature column if provided
            if potential_temperatures is not None:
                if len(potential_temperatures) == n_input:
                    design[start_idx:end_idx, temp_idx] = potential_temperatures[i]
                elif len(potential_temperatures) == 1:
                    design[start_idx:end_idx, temp_idx] = potential_temperatures[0]
        
        return design
    
    def _staged_forward(
        self,
        func,
        input_tensor: torch.Tensor,
        batch_size: int,
        **kwargs
    ) -> Union[torch.Tensor, list]:
        """
        Execute function in batches for memory efficiency.
        
        Parameters
        ----------
        func : callable
            Function to apply (e.g., model forward pass)
        input_tensor : torch.Tensor
            Input batch tensor
        batch_size : int
            Batch size
        **kwargs :
            Additional arguments to pass to func
        
        Returns
        -------
        torch.Tensor or list of torch.Tensor
            Concatenated outputs from all batches
        """
        def _merge_outputs(existing, batch):
            if isinstance(existing, dict) and isinstance(batch, dict):
                merged = {}
                for key in existing:
                    if key not in batch:
                        raise KeyError(f"Missing key '{key}' in staged batch output")
                    merged[key] = _merge_outputs(existing[key], batch[key])
                return merged

            if isinstance(existing, list) and isinstance(batch, (list, tuple)):
                if len(existing) != len(batch):
                    raise ValueError("Staged batch output list length mismatch")
                return [_merge_outputs(existing[i], batch[i]) for i in range(len(existing))]

            if isinstance(existing, tuple) and isinstance(batch, (list, tuple)):
                if len(existing) != len(batch):
                    raise ValueError("Staged batch output tuple length mismatch")
                return tuple(_merge_outputs(existing[i], batch[i]) for i in range(len(existing)))

            if isinstance(existing, torch.Tensor) and isinstance(batch, torch.Tensor):
                return torch.cat([existing, batch], dim=0)

            if isinstance(existing, np.ndarray) and isinstance(batch, np.ndarray):
                return np.concatenate([existing, batch], axis=0)

            return batch

        n_samples = input_tensor.size(0)
        
        if n_samples <= batch_size:
            return func(input_tensor, **kwargs)
        
        # Process first batch
        outputs = func(input_tensor[:batch_size], **kwargs)
        
        # Process remaining batches
        n_batches = (n_samples + batch_size - 1) // batch_size
        for batch_idx in range(1, n_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, n_samples)
            
            batch_outputs = func(input_tensor[start:end], **kwargs)
            outputs = _merge_outputs(outputs, batch_outputs)
        
        return outputs
    
    def parse_input(
        self,
        table: Union[Dict[str, Any], np.ndarray, torch.Tensor],
        headers: Optional[Sequence[str]] = None,
        composition_space: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Parse and reorder input composition table.
        
        Parameters
        ----------
        table : array-like
            Input table with conditions and composition
        headers : sequence of str, optional
            Column headers
        composition_space : str, optional
            'elements' or 'oxides'. Auto-detected if not provided.
        
        Returns
        -------
        torch.Tensor
            Reordered features on correct device
        """
        def _canonicalize_header(name: str) -> str:
            key = str(name).strip()
            return aliases.get(key, key)

        if isinstance(table, dict):
            if headers is None:
                headers = [str(key) for key in table.keys()]
            else:
                headers = [str(h) for h in headers]

            column_arrays = []
            expected_length = None
            for key in headers:
                if key not in table:
                    raise KeyError(f"Dictionary input is missing required key '{key}'")
                column_values = table[key]
                if torch.is_tensor(column_values):
                    column_array = column_values.detach().cpu().numpy()
                else:
                    column_array = np.asarray(column_values, dtype=np.float32)

                if column_array.ndim == 0:
                    column_array = column_array.reshape(1)
                elif column_array.ndim > 1:
                    column_array = column_array.reshape(-1)

                if expected_length is None:
                    expected_length = column_array.shape[0]
                elif column_array.shape[0] != expected_length:
                    raise ValueError(
                        "All dictionary values passed to parse_input must have the same length; "
                        f"got {expected_length} and {column_array.shape[0]} for key '{key}'"
                    )

                column_arrays.append(column_array)

            if len(column_arrays) == 0:
                raise ValueError("table dictionary must contain at least one column")

            values = np.column_stack(column_arrays).astype(np.float32, copy=False)
        else:
            if torch.is_tensor(table):
                values = table.detach().cpu().numpy()
            else:
                values = np.asarray(table, dtype=np.float32)

            if headers is not None:
                headers = [str(h) for h in headers]

        headers = [_canonicalize_header(header) for header in headers]

        has_temperature = 'T(K)(System_main)' in headers
        has_entropy = 'S(J/mol/K)(System_main)' in headers
        if has_temperature and has_entropy:
            raise ValueError(
                "Input headers cannot include both temperature and entropy features"
            )

        if has_entropy:
            emulator = self.isentropic_emulator
        else:
            emulator = self.isothermal_emulator

        parsed_table, headers_out, comp_space = self.parser.parse_composition(
            values, headers, composition_space
        )
        
        # Use the emulator that matches the canonicalized thermodynamic feature.
        reordered = emulator.reorder_input_table(
            parsed_table,
            headers=headers_out,
            composition_space=comp_space,
            strict=False,
            return_type='torch'
        )
        
        return reordered.to(self.device)



    def ForwardMB(
        self,
        table: Union[pd.DataFrame, np.ndarray, torch.Tensor, Sequence],
        headers: Optional[Sequence[str]] = None,
        composition_space: Optional[str] = None,
        model: str = 'isothermal',
        batch_size: int = 2**16,
        normalize_features: bool = True,
        wt_percent: bool = False,
        comp_table_out: str = 'oxides',
        outputs: Optional[Sequence[str]] = None,
    ) -> Union[torch.Tensor, tuple, list, Dict[str, torch.Tensor]]:
        """
        Parse input features and run a staged forwardMB pass on an emulator.

        Parameters
        ----------
        table : pandas.DataFrame or array-like
            Input table containing conditions plus composition columns.
        headers : sequence of str, optional
            Column headers if table is array-like.
        composition_space : str, optional
            'elements' or 'oxides'. Auto-detected if omitted.
        model : str, default='isothermal'
            Emulator to use: 'isothermal' or 'isentropic'.
        batch_size : int, default=2**16
            Batch size for staged evaluation.
        normalize_features : bool, default=True
            Whether to normalize features before model evaluation.
        wt_percent : bool, default=False
            Whether composition inputs are weight percent.
        comp_table_out : str, default='oxides'
            Composition output format passed through to the emulator.
        outputs : sequence[str], optional
            Output selectors forwarded to emulator.forwardMB.

        Returns
        -------
        torch.Tensor, tuple, list, or dict
            Whatever the underlying emulator.forwardMB returns, assembled over
            batches if needed.
        """
        features = self.parse_input(table, headers=headers, composition_space=composition_space)

        model_key = str(model).strip().lower()
        if model_key in {'isothermal', 'iso'}:
            emulator = self.isothermal_emulator
        elif model_key in {'isentropic', 'isentrope', 'adiabat', 'adiabatic'}:
            emulator = self.isentropic_emulator
        else:
            raise ValueError("model must be 'isothermal' or 'isentropic'")

        return self._staged_forward(
            emulator.forwardMB,
            features,
            batch_size,
            Normalize=normalize_features,
            WtPercent=wt_percent,
            comp_table_out=comp_table_out,
            outputs=outputs,
        )

    def ForwardNN(
        self,
        table: Union[pd.DataFrame, np.ndarray, torch.Tensor, Sequence],
        headers: Optional[Sequence[str]] = None,
        composition_space: Optional[str] = None,
        model: str = 'isothermal',
        batch_size: int = 2**16,
        normalize_features: bool = True,
        wt_percent: bool = False,
        comp_table_out: str = 'oxides',
        outputs: Optional[Sequence[str]] = None,
    ) -> Union[torch.Tensor, tuple, list, Dict[str, torch.Tensor]]:
        """
        Parse input features and run a staged forwardMB pass on an emulator.

        Parameters
        ----------
        table : pandas.DataFrame or array-like
            Input table containing conditions plus composition columns.
        headers : sequence of str, optional
            Column headers if table is array-like.
        composition_space : str, optional
            'elements' or 'oxides'. Auto-detected if omitted.
        model : str, default='isothermal'
            Emulator to use: 'isothermal' or 'isentropic'.
        batch_size : int, default=2**16
            Batch size for staged evaluation.
        normalize_features : bool, default=True
            Whether to normalize features before model evaluation.
        wt_percent : bool, default=False
            Whether composition inputs are weight percent.
        comp_table_out : str, default='oxides'
            Composition output format passed through to the emulator.
        outputs : sequence[str], optional
            Output selectors forwarded to emulator.forwardMB.

        Returns
        -------
        torch.Tensor, tuple, list, or dict
            Whatever the underlying emulator.forwardMB returns, assembled over
            batches if needed.
        """
        features = self.parse_input(table, headers=headers, composition_space=composition_space)

        model_key = str(model).strip().lower()
        if model_key in {'isothermal', 'iso'}:
            emulator = self.isothermal_emulator
        elif model_key in {'isentropic', 'isentrope', 'adiabat', 'adiabatic'}:
            emulator = self.isentropic_emulator
        else:
            raise ValueError("model must be 'isothermal' or 'isentropic'")

        return self._staged_forward(
            emulator.forwardNN,
            features,
            batch_size,
            Normalize=normalize_features,
            WtPercent=wt_percent,
            comp_table_out=comp_table_out,
            outputs=outputs,
        )
        



# Model paths - resolved relative to this file's location
_this_file_dir = Path(__file__).parent
_models_dir = _this_file_dir / "TrainedModels" / "HeFESTo_Adiabats"

adiabat_NPT_path = _models_dir / "HeFESTo_adiabats_NPT.tar"
adiabat_NPS_path = _models_dir / "HeFESTo_adiabats_NPS.tar"
adiabat_TfromS_path = _models_dir / "T_from_S_HeFESTo_adiabats.pt"

# Only instantiate emulators if model files exist
HeFESToEmulatorCPU = None
HeFESToEmulatorGPU = None

try:
    if adiabat_NPT_path.exists() and adiabat_NPS_path.exists() and adiabat_TfromS_path.exists():
        HeFESToEmulatorCPU = HeFESToAPI(
            isothermal_model_path=str(adiabat_NPT_path),
            isentropic_model_path=str(adiabat_NPS_path),
            temperature_model_path=str(adiabat_TfromS_path),
            device='cpu'
        )

        if torch.cuda.is_available():
            HeFESToEmulatorGPU = HeFESToAPI(
                isothermal_model_path=str(adiabat_NPT_path),
                isentropic_model_path=str(adiabat_NPS_path),
                temperature_model_path=str(adiabat_TfromS_path),
                device='cuda'
            )
except Exception as _e:
    pass  # Model files not available at import time

