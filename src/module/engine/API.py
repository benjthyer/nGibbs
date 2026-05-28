"""
High-level API for thermodynamic emulators.

Provides user-friendly interfaces for:
- Getting temperatures from isentropic states
- Computing isentropic adiabats along pressure transects
- Parsing and converting input compositions
- Managing emulator ensembles (isothermal, isentropic, temperature models)

Base class: EmulatorAPI — model-agnostic adiabat workflows
Subclass:   HeFESToAPI  — adds Burnman EOS backend for HeFESTo
"""

import gc
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union, Dict, Any
import sys
import time
import tqdm

file_path = str(Path(__file__).parent)
if file_path not in sys.path:
    sys.path.insert(0, file_path)
module_path = str(Path(__file__).parent.parent)
if module_path not in sys.path:
    sys.path.insert(0, module_path)
print(f"Module path added to sys.path: {module_path}")

from engine.EOS_arithmetic.vector_composite import VectorComposite
import EOS_arithmetic.burnman as burnman

from config.constants import OXIDE_MOLAR_MASSES as oxide_molar_masses, ptt_longs, ptt_order_oxides, ptt_to_short, ptt_oxide_indexer, HeFESTosnames_long

ferric_to_ferrous_ratio =  (2*oxide_molar_masses['FeO'])/oxide_molar_masses['Fe2O3']
snames_dict = {name: i for i, name in enumerate(HeFESTosnames_long)}

from NN import _load_temperature_model
from emulator import NN_MELTS
from utils.math_utils import Normalizer

aliases = { # 
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

burnman_phase_aliases = {
    'spinel':'mg_fe_aluminous_spinel',
    'hp-clinopyroxene':'c2c_pyroxene',
    'ca-perovskite':'ca_perovskite',
    'akimotoite':'ilmenite',
    'post-perovskite':'post_perovskite',
    'ca-ferrite':'cf',
    'gamma-iron':'gamma_fcc_iron',
    'epsilon-iron':'epsilon_hcp_iron',
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
                #print("Adding ferric column...")
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


class EmulatorAPI:
    """
    Base API for thermodynamic emulators.

    Manages:
    - Isothermal and isentropic emulators
    - Temperature prediction model (FCNN)
    - Input parsing and composition handling
    - Adiabat computation workflows

    Subclasses must implement `_compute_bulk_EOS_properties` to provide
    an equation-of-state backend for entropy and bulk property evaluation.
    """

    def __init__(
        self,
        isothermal_model_path: Union[str, Path],
        isentropic_model_path: Union[str, Path],
        temperature_model_path: Optional[Union[str, Path]] = None,
        device: str = 'cpu',
        verbose: bool = False,
    ):
        """
        Initialize the emulator API with neural network models.

        Parameters
        ----------
        isothermal_model_path : str or Path
            Path to isothermal emulator checkpoint
        isentropic_model_path : str or Path
            Path to isentropic emulator checkpoint
        temperature_model_path : str or Path, optional
            Path to temperature FCNN checkpoint
        device : str, default='cpu'
            Torch device ('cpu' or 'cuda')
        verbose : bool, default=False
            Print initialization messages
        """
        if verbose:
            print("[INFO] Initializing EmulatorAPI...")

        self.device = torch.device(device)
        self.verbose = verbose

        if verbose:
            print(f"  Loading isothermal emulator: {isothermal_model_path}")
        self.isothermal_emulator = self._load_emulator(isothermal_model_path, device)

        if verbose:
            print(f"  Loading isentropic emulator: {isentropic_model_path}")
        self.isentropic_emulator = self._load_emulator(isentropic_model_path, device)

        if temperature_model_path is not None:
            if verbose:
                print(f"  Loading temperature FCNN: {temperature_model_path}")
            self._setup_temperature_model(temperature_model_path)

        self.parser = InputParser(self.isothermal_emulator)

        if verbose:
            print("[INFO] EmulatorAPI initialized successfully.")

    def _compute_bulk_EOS_properties(
        self,
        component_moles: torch.Tensor,
        PT: torch.Tensor,
        property_names: Sequence[str],
    ) -> Dict[str, np.ndarray]:
        """
        Compute bulk EOS properties for a given assemblage.

        Must be implemented by subclasses to provide a thermodynamic
        equation-of-state backend (e.g. Burnman for HeFESTo).

        Parameters
        ----------
        component_moles : torch.Tensor, shape (N, C)
            Component mole fractions from emulator output
        PT : torch.Tensor, shape (N, 2)
            Pressure (GPa) and temperature (K): PT[:, 0] = P, PT[:, 1] = T
        property_names : sequence of str
            Property names to compute (e.g. 'entropy_by_mass', 'density')

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from property name to array of shape (N,)
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _compute_bulk_EOS_properties"
        )

    @staticmethod
    def _load_emulator(model_path: Union[str, Path], device: str) -> NN_MELTS:
        """Load and wrap a checkpoint as NN_MELTS emulator."""
        from engine.NN import rebuild_MELTS_model

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

    #NOTE: This assumes that the isentropic and isothermal emulators have S and T in the same position within the features
    def get_isentrope(
        self,
        features: Union[np.ndarray, torch.Tensor, Sequence],
        headers: Sequence[str],
        pressures: Union[np.ndarray, torch.Tensor] = None,
        potential_temperatures: Optional[Union[np.ndarray, torch.Tensor]] = None,
        batch_size: int = 2**16,
        normalize_features: bool = True,
        outputs=None,
        properties: Optional[Sequence[str]] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, np.ndarray]]]:
        """
        Compute isentropic adiabats along a pressure grid.

        Two modes of operation:

        Mode A — potential_temperatures provided:
            features must have exactly 1 row. That row is tiled to
            len(potential_temperatures) rows and the temperature column is
            overwritten by each potential temperature. The isothermal emulator
            evaluates each state at P=0.0001 GPa; `_compute_bulk_EOS_properties`
            then computes the reference entropy S (J/g/K) for each. Each S is
            swept across the pressure grid via the isentropic model.
            Output shape: (len(potential_temperatures), len(pressures)).

        Mode B — no potential_temperatures:
            features can have any number of rows N; composition may vary
            freely across rows. The T column of each row acts as that row's
            potential temperature. Entropy is computed per row at P=0.0001 GPa,
            then each S is swept across the pressure grid.
            Output shape: (N, len(pressures)).

        Parameters
        ----------
        features : array-like (N, F)
            Input features. Mode A requires N == 1.
        headers : sequence of str
            Column headers for features.
        pressures : array-like, optional
            Pressure grid in GPa. Defaults to linspace(0, 140, 300).
        potential_temperatures : array-like, optional
            Potential temperatures (K). None → use T column from features.
        batch_size : int
            Batch size for staged evaluation.
        normalize_features : bool
            Whether to normalize input features before emulator calls.
        properties : sequence of str, optional
            EOS property names to evaluate at each isentropic (P, T) state
            (e.g. ['density', 'p_wave_velocity', 'entropy_by_mass']). Each
            property is returned as an np.ndarray of shape (n_S, n_P).
            When provided, a third return value (dict) is included.

        Returns
        -------
        temperatures_grid : torch.Tensor, shape (n_S, n_P)
        pressures_grid : torch.Tensor, shape (n_S, n_P)
        properties_dict : dict[str, np.ndarray], shape (n_S, n_P) per key
            Only returned when `properties` is not None.
        """
        features_arr = np.asarray(features, dtype=np.float32)
        if features_arr.ndim != 2:
            raise ValueError(f"features must be 2D, got shape {features_arr.shape}")

        if pressures is None:
            pressures = np.linspace(0, 140, 300)
        pressures = np.asarray(pressures, dtype=np.float32).flatten()
        n_press = len(pressures)
        pressures_t = torch.tensor(pressures, dtype=torch.float32, device=self.device)

        iso_feat_names = list(self.isothermal_emulator.ml_indexer.featureNames)
        temp_idx = iso_feat_names.index('T(K)(System_main)')
        iso_pressure_idx = iso_feat_names.index('P(GPa)(System_main)')

        isen_feat_names = list(self.isentropic_emulator.ml_indexer.featureNames)
        entropy_idx = isen_feat_names.index('S(J/g/K)(System_main)')
        isen_pressure_idx = isen_feat_names.index('P(GPa)(System_main)')

        # --- Build isothermal reference features at P=0.0001 ---
        if potential_temperatures is not None:
            # Mode A: single composition tiled over potential temperatures
            if features_arr.shape[0] != 1:
                raise ValueError(
                    f"When potential_temperatures is provided, features must have exactly 1 row, "
                    f"got {features_arr.shape[0]}"
                )
            potential_temperatures = np.asarray(potential_temperatures, dtype=np.float32).flatten()
            n_S = len(potential_temperatures)

            iso_single, _ = self.parse_input(features_arr, headers=headers)  # (1, F_iso)
            iso_ref = iso_single.repeat(n_S, 1)                               # (n_S, F_iso)
            iso_ref[:, temp_idx] = torch.tensor(
                potential_temperatures, dtype=torch.float32, device=self.device
            )
            iso_ref[:, iso_pressure_idx] = 0.0001
            T_for_EOS = torch.tensor(
                potential_temperatures, dtype=torch.float64, device=self.device
            )

        else:
            # Mode B: N rows, T column provides each row's potential temperature
            n_S = features_arr.shape[0]
            iso_ref, _ = self.parse_input(features_arr, headers=headers)     # (n_S, F_iso)
            T_for_EOS = iso_ref[:, temp_idx].to(dtype=torch.float64, device=self.device)
            iso_ref = iso_ref.clone()
            iso_ref[:, iso_pressure_idx] = 0.0001

        if self.verbose:
            print(f"[get_isentrope] n_S={n_S}, n_P={n_press}, total states={n_S * n_press}")

        # --- Step 1: Isothermal emulator at P=0.0001 → component moles ---
        iso_out = self._staged_forward(
            self.isothermal_emulator.forwardMB,
            iso_ref,
            batch_size,
            Normalize=normalize_features,
            outputs=['component_moles'],
        )
        component_moles = iso_out['component_moles'].to(device=self.device, dtype=torch.float64)

        # --- Step 2: EOS backend → reference entropy S (J/g/K) ---
        P_ref = torch.full((n_S,), 0.0001, dtype=torch.float64, device=self.device)
        PT_ref = torch.stack([P_ref, T_for_EOS], dim=1)  # (n_S, 2): columns = [P, T]

        eos_props = self._compute_bulk_EOS_properties(
            component_moles,
            PT_ref,
            property_names=['entropy_by_mass'],
        )
        S_values = torch.tensor(
            eos_props['entropy_by_mass']/1000, dtype=torch.float32, device=self.device
        )  # (n_S,)
        print(f"Reference entropies (J/g/K) at P=0.0001 GPa: {S_values.cpu().numpy()}")

        # --- Step 3: Isentropic design matrix (n_S * n_press, F) ---
        # Each of the n_S base rows is repeated n_press times, then S and P
        # columns are overwritten with target entropy and pressure values.
        isen_design = iso_ref.repeat_interleave(n_press, dim=0)  # (n_S * n_press, F)
        isen_design[:, entropy_idx] = S_values.repeat_interleave(n_press)
        isen_design[:, isen_pressure_idx] = pressures_t.repeat(n_S)

        # --- Step 4: Isentropic model → component moles ---
        isen_out = self._staged_forward(
            self.isentropic_emulator.forwardMB,
            isen_design,
            batch_size,
            Normalize=normalize_features,
            outputs=['phase_moles', 'chem_out', 'component_moles'],
        )
        pred_component_moles = isen_out['component_moles'].to(device=self.device, dtype=torch.float64)
        temperatures_pred = self.get_T(torch.concatenate([isen_design, isen_out['phase_moles'], isen_out['chem_out']], dim=1), normalize_features=normalize_features) # This is another neural network!

        # --- Step 6: Reshape → (n_S, n_press) ---
        temperatures_grid = temperatures_pred.reshape(n_S, n_press)
        pressures_grid = pressures_t.unsqueeze(0).expand(n_S, -1)

        if self.verbose:
            print(f"[get_isentrope] Output temperatures shape: {temperatures_grid.shape}")

        if properties is None:
            return temperatures_grid, pressures_grid

        # --- Step 7: EOS properties at isentropic (P, T) states ---
        # PT columns = [P, T], matching _compute_bulk_EOS_properties convention
        P_flat = pressures_grid.reshape(-1).to(dtype=torch.float64, device=self.device)
        T_flat = temperatures_grid.reshape(-1).to(dtype=torch.float64, device=self.device)
        PT_isen = torch.stack([P_flat, T_flat], dim=1)  # (n_S * n_press, 2)

        raw_props = self._compute_bulk_EOS_properties(
            pred_component_moles.to(device=self.device, dtype=torch.float64),
            PT_isen,
            property_names=list(properties),
        )

        properties_dict = {
            name: np.asarray(arr).reshape(n_S, n_press)
            for name, arr in raw_props.items()
        }

        return temperatures_grid, pressures_grid, properties_dict


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
                return torch.cat([existing, batch.detach().cpu()], dim=0)

            if isinstance(existing, np.ndarray) and isinstance(batch, np.ndarray):
                return np.concatenate([existing, batch], axis=0)

            return batch

        n_samples = input_tensor.size(0)

        if n_samples <= batch_size:
            return func(input_tensor, **kwargs)

        # Process first batch
        with torch.no_grad():
            outputs = func(input_tensor[:batch_size], **kwargs)

            # Process remaining batches
            n_batches = (n_samples + batch_size - 1) // batch_size
            for batch_idx in tqdm.tqdm(range(1, n_batches), desc="Processing batches of size " + str(batch_size)):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, n_samples)

                batch_outputs = func(input_tensor[start:end], **kwargs)
                outputs = _merge_outputs(outputs, batch_outputs)
                del batch_outputs
                gc.collect()

                torch.cuda.empty_cache()

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
        """def _canonicalize_header(name: str) -> str:
            key = str(name).strip()
            return aliases.get(key, key)"""

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

        #headers = [_canonicalize_header(header) for header in headers]

        has_temperature = 'T(K)(System_main)' in headers or 'Temperature(System_main)' in headers
        has_entropy = 'S(J/g/K)(System_main)' in headers or 'S(System_main)' in headers
        if has_temperature and has_entropy:
            raise ValueError(
                "Input headers cannot include both temperature and entropy features"
            )

        if has_entropy:
            emulator = self.isentropic_emulator
        elif has_temperature:
            emulator = self.isothermal_emulator
        else:
            raise ValueError("Input headers must include either temperature or entropy features: T(K)(System_main) / 'Temperature(System_main)' or S(J/g/K)(System_main) / 'S(System_main)'.\n You have: {}".format(headers))

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

        non_chem_features = len(self.isentropic_emulator.ml_indexer.featureNames) # The emulator type doesn't matter: But they both need to agree on open or closed oxygen systematics
        chem_sum = reordered[:,non_chem_features:].sum(dim=1)
        reordered[:,non_chem_features:] = reordered[:,non_chem_features:] / chem_sum.unsqueeze(1)

        return reordered.to(self.device), 'isentropic' if has_entropy else 'isothermal'



    def ForwardMB(
        self,
        table: Union[pd.DataFrame, np.ndarray, torch.Tensor, Sequence],
        headers: Optional[Sequence[str]] = None,
        composition_space: Optional[str] = None,
        batch_size: int = 2**15,
        normalize_features: bool = True,
        wt_percent: bool = False,
        comp_table_out: str = 'oxides',
        outputs: Optional[Sequence[str]] = 'component_moles',
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
            - 'transcomponent_hat'
            - 'chem_out'
            - 'phase_tables'
            - 'component_moles'
            - 'wt_del_component_moles'
            - 'phase_moles'
            - 'temperature' #NN temp output for isentropic models
            - 'ptt_out' Phase table output formatting like ptt
        Returns
        -------
        torch.Tensor, tuple, list, or dict
            Whatever the underlying emulator.forwardMB returns, assembled over
            batches if needed.
        """
        features, modeltype = self.parse_input(table, headers=headers, composition_space=composition_space)

        if modeltype == 'isothermal':
            if 'temperature' in outputs:
                raise ValueError("'temperature' output was passed for an isothermal model... Did you mean to use an isentropic model?")
            emulator = self.isothermal_emulator
        elif modeltype == 'isentropic':
            emulator = self.isentropic_emulator
        else:
            raise ValueError(f"parser determined model is not recognized: {modeltype} must be 'isothermal' or 'isentropic'")

        if 'temperature' in outputs:
            if 'chem_out' not in outputs:
                outputs = list(outputs) + ['chem_out']
                print("[INFO] Adding 'chem_out' to outputs for temperature calculation.")
            if 'phase_moles' not in outputs:
                outputs = list(outputs) + ['phase_moles']
                print("[INFO] Adding 'phase_moles' to outputs for temperature calculation.")
            get_temp = True
            outputs.remove('temperature') # temperature not recognized as arg for NN.
        else:
            get_temp = False

        if 'ptt_out' in outputs:
            get_ptt_tables = True
            if 'phase_tables' not in outputs:
                outputs = list(outputs) + ['phase_tables']
                print("[INFO] Adding 'phase_tables' to outputs for ptt_out formatting.")
            outputs.remove('ptt_out') # ptt_out not recognized by lower level funcs.
        else:            
            get_ptt_tables = False

        results = self._staged_forward(
            emulator.forwardMB,
            features,
            batch_size,
            Normalize=normalize_features,
            WtPercent=wt_percent,
            comp_table_out=comp_table_out,
            outputs=outputs,
        )

        if get_temp:
            results['temperature'] = self.get_T(torch.concatenate([features, results['phase_moles'].to(self.device), results['chem_out'].to(self.device)], dim=1), normalize_features=normalize_features)

        if get_ptt_tables:
            results['ptt_out'] = self.make_ptt_out(features, results['phase_tables'][0], results['phase_tables'][1], isentropic=(modeltype=='isentropic'))

        return results


    def ForwardNN(
        self,
        table: Union[pd.DataFrame, np.ndarray, torch.Tensor, Sequence],
        headers: Optional[Sequence[str]] = None,
        composition_space: Optional[str] = None,
        batch_size: int = 2**15,
        normalize_features: bool = True,
        wt_percent: bool = False,
        outputs: Optional[Sequence[str]] = None,
    ) -> Union[torch.Tensor, tuple, list, Dict[str, torch.Tensor]]:
        """
        Parse input features and run a staged forwardNN pass on an emulator.

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
        features, modeltype = self.parse_input(table, headers=headers, composition_space=composition_space)

        if modeltype == 'isothermal':
            if 'temperature' in outputs:
                raise ValueError("'temperature' output was passed for an isothermal model... Did you mean to use an isentropic model?")
            emulator = self.isothermal_emulator
        elif modeltype == 'isentropic':
            emulator = self.isentropic_emulator
        else:
            raise ValueError(f"parser determined model is not recognized: {modeltype} must be 'isothermal' or 'isentropic'")

        if 'temperature' in outputs:
            if 'chem_out' not in outputs:
                outputs = list(outputs) + ['chem_out']
                print("[INFO] Adding 'chem_out' to outputs for temperature calculation.")
            if 'phase_moles' not in outputs:
                outputs = list(outputs) + ['phase_moles']
                print("[INFO] Adding 'phase_moles' to outputs for temperature calculation.")
            get_temp = True
            outputs.remove('temperature') # temperature not recognized as arg for NN.
        else:
            get_temp = False

        if 'ptt_out' in outputs:
            get_ptt_tables = True
            if 'phase_tables' not in outputs:
                outputs = list(outputs) + ['phase_tables']
                print("[INFO] Adding 'phase_tables' to outputs for ptt_out formatting.")
            outputs.remove('ptt_out') # ptt_out not recognized by lower level funcs.
        else:            
            get_ptt_tables = False

        results = self._staged_forward(
            emulator.forwardNN,
            features,
            batch_size,
            Normalize=normalize_features,
            WtPercent=wt_percent,
            outputs=outputs,
        )

        if get_temp:
            results['temperature'] = self.get_T(torch.concatenate([features, results['phase_moles'].to(self.device), results['chem_out'].to(self.device)], dim=1), normalize_features=normalize_features)

        if get_ptt_tables:
            results['ptt_out'] = self.make_ptt_out(features, results['phase_tables'][0], results['phase_tables'][1], isentropic=(modeltype=='isentropic'))

        return results

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

        # Normalize for temperature model
        temp_input_norm = self.temp_input_normalizer.norm(features)

        # Forward through temperature FCNN
        with torch.no_grad():
            temp_output_norm = self.temperature_model(temp_input_norm)

        # Denormalize output
        temperature = self.temp_output_normalizer.denorm(temp_output_norm)

        return temperature.squeeze(-1)
    
    def make_ptt_out(self, features, phaseOxWt, phaseMassNorm, isentropic=None):
        """
        Create output in format of ptt pandas tables. This should only be used internally.
        [TODO]: Calculate thermodynamic properties for bulk and each phase to include in output tables.
        
        Parameters
        ----------
        features : torch.Tensor
            Input features
        phaseOxWt : torch.Tensor
            Phase oxide weight percent tables, 1st output of make_phase_tables
        phaseMassNorm : torch.Tensor
            Phase mass fractions, 2nd output of make_phase_tables
        isentropic : bool, optional
            Whether the input features are isentropic

        Returns
        -------
        dict
            of pandas DataFrames matching ptt format for each phase
        """
    

        output = {} # Initialize dict of Pandas outputs
        P_bar = features[:, self.isentropic_emulator.ml_indexer.featureNames.index('Pressure(System_main)')].detach().cpu().numpy()

        indexer = self.isentropic_emulator.ml_indexer if isentropic else self.isothermal_emulator.ml_indexer # get appropriate indexer

        if not isentropic: # Isothermal models
            T_C = features[:, indexer.featureNames.index('Temperature(System_main)')].detach().cpu().numpy()
            s = np.full((features.size(0),), fill_value=np.nan)
        else: # Isentropic models - we have entropy but not temperature as input features.
            T_C = np.full((features.size(0),), fill_value=np.nan) 
            s = features[:, indexer.featureNames.index('S(System_main)')].detach().cpu().numpy()

        output['Conditions'] = pd.DataFrame([P_bar, T_C, s], index=['P_bar', 'T_C', 's']).T

        # Build oxide mappings for ptt output
        ptt_Ox_Idx = np.array([ptt_oxide_indexer[ox] for ox in indexer.Oxides]).astype(int)

        # Build correct-order name mappings for ptt output
        phasePresent = (phaseMassNorm > 1e-3).any(dim=0).detach().cpu().numpy().astype(int)  # (P,)
        accountedPhases = np.zeros_like(phasePresent, dtype=int) # Track which phases we've mapped to ptt namings
        phaseMappings = []
        mass_array = np.zeros((features.size(0), int(phasePresent.sum())))
        mass_cols = []
        all_mass_cols = []
        

        for ptt_long, ptt_short in ptt_to_short.items():
            if ptt_long in ptt_longs:
                internal_name = ptt_longs[ptt_long] # special mappings for K-spar and liquid.
            else:
                internal_name = ptt_long[:-1] # Exclude number at the end
            internalPhaseCol = indexer.mass_phasedict[internal_name]
            if phasePresent[internalPhaseCol]:
                accountedPhases[internalPhaseCol] = 1
                phaseMappings.append((ptt_long, ptt_short, internal_name, internalPhaseCol))
        
        unaccountedPhases = phasePresent - accountedPhases


        if np.sum(unaccountedPhases):
            for idx in np.where(unaccountedPhases)[0]:
                intName = indexer.all_phases[idx]
                pttName = intName + '1'
                phaseMappings.append((pttName, '_' + pttName, intName, idx)) # Map phases with no naming scheme in ptt


        # Now build output tables!
        for i, (ptt_long, ptt_short, internal_name, internalPhaseCol) in enumerate(phaseMappings):
            print(internal_name)
            if internal_name in indexer.compositionally_variable_phases:
                print(f"Phase {internal_name} is compositionally variable!.")
                colnames = [Ox + ptt_short for Ox in ptt_order_oxides]

                outTable = np.zeros((features.size(0), len(ptt_order_oxides)))
                outTable[:, ptt_Ox_Idx] = phaseOxWt[:, indexer.comp_phasedict[internal_name], :]

                # Handle iron shenanigans.
                Fe2 = outTable[:, ptt_oxide_indexer['FeO']]/oxide_molar_masses['FeO']
                Fe3 = outTable[:, ptt_oxide_indexer['Fe2O3']]/oxide_molar_masses['Fe2O3']*2
                Fet = Fe2 + Fe3
                outTable[:, ptt_oxide_indexer['Fe3Fet']] = Fe3/Fet
                outTable[:, ptt_oxide_indexer['FeOt']] = Fet * oxide_molar_masses['FeO']

                output[ptt_long] = pd.DataFrame(outTable, columns=colnames)

            mass_array[:,i] = phaseMassNorm[:, internalPhaseCol].detach().cpu().numpy()
            mass_cols.append(ptt_long)
            all_mass_cols.append(ptt_short)
        all_mass_cols = list(output['Conditions'].columns) + all_mass_cols # Combine condition and mass column names for "All" table

        output['mass_g'] = pd.DataFrame(mass_array, columns=mass_cols) # Finally, mass table. 
        output['All'] = pd.DataFrame(np.concatenate([output['Conditions'], mass_array], axis=1), columns=all_mass_cols) # A version of the mass table with short column names for easy parsing in ptt scripts such as phase diagrams

        return output
    
    def divide_ptt_tables(self, ptt_out, tableIDX):
        """
        Divide ptt rows in output tables according to tableIDX
        """

        tableIDX = tableIDX.astype(int)
        IDs = np.unique(tableIDX)
        divided_output = {}
        for id in IDs: # initialize dict of divided outputs for each unique tableIDX
            divided_output[f"Run {id}"] = {}

        for key, df in ptt_out.items(): 
            for id in IDs:
                divided_output[f"Run {id}"][key] = df.iloc[tableIDX == id]

        return divided_output




class HeFESToAPI(EmulatorAPI):
    """
    HeFESTo emulator API with Burnman EOS backend.

    Extends EmulatorAPI with Burnman-based bulk property evaluation,
    enabling isentrope computation and physical property retrieval for
    the HeFESTo mineral physics database.
    """

    def __init__(
        self,
        isothermal_model_path: Union[str, Path],
        isentropic_model_path: Union[str, Path],
        temperature_model_path: Optional[Union[str, Path]] = None,
        device: str = 'cpu',
        verbose: bool = False,
    ):
        """
        Initialize HeFESTo API with emulator models and Burnman phases.

        Parameters
        ----------
        isothermal_model_path : str or Path
            Path to isothermal emulator checkpoint
        isentropic_model_path : str or Path
            Path to isentropic emulator checkpoint
        temperature_model_path : str or Path, optional
            Path to temperature FCNN checkpoint
        device : str, default='cpu'
            Torch device ('cpu' or 'cuda')
        verbose : bool, default=False
            Print initialization messages
        """
        if verbose:
            print("[INFO] Initializing HeFESToAPI...")

        super().__init__(
            isothermal_model_path,
            isentropic_model_path,
            temperature_model_path,
            device,
            verbose,
        )

        # Build Burnman input translator
        phase_names = self.isothermal_emulator.ml_indexer.all_phases
        self.burnman_translator = {phase: phase for phase in phase_names}
        for hefesto_name, burnman_name in burnman_phase_aliases.items():
            self.burnman_translator[hefesto_name] = burnman_name

        self.burnman_minerals = {
            phase_name: getattr(burnman.minerals.SLB_2024, self.burnman_translator[phase_name])()
            for phase_name in self.isothermal_emulator.ml_indexer.all_phases
        }
        
        assert self.isothermal_emulator.ml_indexer.label_names == self.isentropic_emulator.ml_indexer.label_names # Assume the label names are model-agnostic

        self.PropertyIDX = np.array([snames_dict[name] for name in self.isothermal_emulator.ml_indexer.label_names])


        if verbose:
            print("[INFO] HeFESToAPI initialized successfully.")

    def _compute_bulk_EOS_properties(
        self,
        component_moles: torch.Tensor,
        PT: torch.Tensor,
        property_names: Sequence[str],
    ) -> Dict[str, np.ndarray]:
        """Delegate to Burnman backend."""
        return self.get_property_burnman_vectorized_from_assemblage(
            component_moles, PT, property_names
        )

    def get_property_burnman_vectorized_from_assemblage(
        self,
        componentMoles,
        PT,
        property_names = ['entropy_by_mass', 'density', 'p_wave_velocity', 's_wave_velocity', 'isothermal_bulk_modulus_reuss']):

        """
        Compute properties for a given assemblage using Burnman.

        Parameters
        ----------
        intensive_moles : np.array
            (N, C) tensor of intensive component moles from emulator output
        phase_moles : np.array
            (N, P) tensor of phase moles from emulator output
        PT : np.array
            (N, 2) tensor of pressure and temperature values
        property_names : sequence of str
            List of properties recongnized by burnman to compute (e.g., ['bulk_modulus', 'density'])

        Returns
        -------
        numpy.ndarray
            Array of shape (N, len(property_names)) with computed properties for each assemblage
        """

        time_start = time.time()

        self.VecEOS = VectorComposite(
        #component_names=[self.burnman_translator[phase] for phase in self.isentropic_emulator.ml_indexer.label_names],
        #phase_names=[self.burnman_translator[phase] for phase in self.isothermal_emulator.ml_indexer.all_phases],
        phase_identity=torch.tensor(self.isothermal_emulator.ml_indexer.phaseToCompMap.T, dtype=torch.float64, device=self.device),
        component_abundances=componentMoles,
        phase_models=self.burnman_minerals,
        pressure=PT[:,0],
        temperature=PT[:,1],
        device=self.device,
        dtype=torch.float64
        )

        #self.VecEOS.evaluate_backend_properties()

        nrows = componentMoles.size()[0]

        assert nrows == PT.size()[0], "Batch size of intensive_moles and phase_moles must match"

        #outmatrix = np.zeros((nrows, len(property_names)), dtype=np.float32)
        outdict = {}

        for j, prop in enumerate(property_names):
            try:
                outdict[prop] = getattr(self.VecEOS, prop).detach().cpu().numpy()
            except Exception as e:
                raise ValueError(f"Error computing property '{prop}': {e}")

        print(f"Burnman property computation for {nrows} assemblages and properties {property_names} took {time.time() - time_start:.2f} seconds")
        return outdict

class MELTSAPI:
    """
    Dispatcher API for MELTS emulators with Cr / NoCr model variants.

    Holds two EmulatorAPI sub-instances and routes every call to the
    correct one by inspecting the input headers: if 'Cr' or 'Cr2O3'
    appears, the Cr sub-API is used; otherwise the NoCr sub-API is used.

    The public interface mirrors EmulatorAPI (ForwardMB, ForwardNN,
    get_isentrope, parse_input). For methods that don't carry headers
    (e.g. get_T) access the sub-API directly via .nocr or .cr.
    """

    def __init__(
        self,
        isothermal_NoCr_model_path: Union[str, Path],
        isothermal_Cr_model_path: Union[str, Path],
        isentropic_NoCr_model_path: Union[str, Path],
        isentropic_Cr_model_path: Union[str, Path],
        temperature_NoCr_model_path: Optional[Union[str, Path]] = None,
        temperature_Cr_model_path: Optional[Union[str, Path]] = None,
        device: str = 'cpu',
        verbose: bool = False,
    ):
        """
        Initialize MELTS API, building NoCr and Cr sub-APIs.

        Parameters
        ----------
        isothermal_NoCr_model_path : str or Path
            Path to isothermal emulator checkpoint (NoCr)
        isothermal_Cr_model_path : str or Path
            Path to isothermal emulator checkpoint (Cr)
        isentropic_NoCr_model_path : str or Path
            Path to isentropic emulator checkpoint (NoCr)
        isentropic_Cr_model_path : str or Path
            Path to isentropic emulator checkpoint (Cr)
        temperature_NoCr_model_path : str or Path, optional
            Path to temperature FCNN checkpoint (NoCr)
        temperature_Cr_model_path : str or Path, optional
            Path to temperature FCNN checkpoint (Cr)
        device : str, default='cpu'
            Torch device ('cpu' or 'cuda')
        verbose : bool, default=False
            Print initialization messages
        """
        if verbose:
            print("[INFO] Initializing MELTSAPI (NoCr)...")
        self.nocr = EmulatorAPI(
            isothermal_NoCr_model_path,
            isentropic_NoCr_model_path,
            temperature_NoCr_model_path,
            device,
            verbose,
        )
        if verbose:
            print("[INFO] Initializing MELTSAPI (Cr)...")
        self.cr = EmulatorAPI(
            isothermal_Cr_model_path,
            isentropic_Cr_model_path,
            temperature_Cr_model_path,
            device,
            verbose,
        )
        self.device = torch.device(device)
        self.verbose = verbose
        if verbose:
            print("[INFO] MELTSAPI initialized successfully.")

    def _route(
        self,
        table=None,
        headers: Optional[Sequence[str]] = None,
    ) -> EmulatorAPI:
        """
        Select NoCr or Cr sub-API from headers.

        Checks `headers` first; if None, falls back to DataFrame column
        names. Routes to .cr if 'Cr' or 'Cr2O3' is present, else .nocr.
        """
        header_list = headers
        if header_list is None and hasattr(table, 'columns'):
            header_list = list(table.columns)
        if header_list is not None:
            header_set = {str(h).strip() for h in header_list}
            if 'Cr' in header_set or 'Cr2O3' in header_set:
                return self.cr
        return self.nocr

    def ForwardMB(self, table, headers=None, **kwargs):
        return self._route(table, headers).ForwardMB(table, headers=headers, **kwargs)

    def ForwardNN(self, table, headers=None, **kwargs):
        return self._route(table, headers).ForwardNN(table, headers=headers, **kwargs)

    def get_isentrope(self, features, headers, **kwargs):
        return self._route(headers=headers).get_isentrope(features, headers, **kwargs)

    def parse_input(self, table, headers=None, **kwargs):
        return self._route(table, headers).parse_input(table, headers=headers, **kwargs)
    
    def divide_ptt_tables(self, ptt_out, tableIDX):
        return self.cr.divide_ptt_tables(ptt_out, tableIDX) # Function is agnostic of model.

# Initialize APIs for best models.

# Model paths - resolved relative to this file's location
_this_file_dir = Path(__file__).parent
_models_dir = _this_file_dir / "TrainedModels" / "HeFESTo_Adiabats"

adiabat_NPT_path = _models_dir / "HeFESTo_adiabats_NPT.tar"
adiabat_NPS_path = _models_dir / "HeFESTo_adiabats_NPS.tar"
adiabat_TfromS_path = _models_dir / "T_from_S_HeFESTo_adiabats.pt"
print(f"[INFO] Looking for HeFESTo model at: {adiabat_NPT_path}")

MELTS102_dir = _this_file_dir / "TrainedModels" / "MELTS102"



HeFESToEmulatorCPU = None
HeFESToEmulatorGPU = None
MELTS102EmulatorCPU = None
MELTS102EmulatorGPU = None




HeFESToEmulatorCPU = HeFESToAPI(
    isothermal_model_path=str(adiabat_NPT_path),
    isentropic_model_path=str(adiabat_NPS_path),
    temperature_model_path=str(adiabat_TfromS_path),
    device='cpu'
)

MELTS102EmulatorCPU = MELTSAPI(
    isothermal_NoCr_model_path = str(MELTS102_dir / "Isothermal_NoCr.tar"),
    isothermal_Cr_model_path = str(MELTS102_dir / "Isothermal_Cr.tar"),
    isentropic_NoCr_model_path = str(MELTS102_dir / "Isentropic_NoCr.tar"),
    isentropic_Cr_model_path = str(MELTS102_dir / "Isentropic_Cr.tar"),
    device='cpu'
)

if torch.cuda.is_available():
    HeFESToEmulatorGPU = HeFESToAPI(
        isothermal_model_path=str(adiabat_NPT_path),
        isentropic_model_path=str(adiabat_NPS_path),
        temperature_model_path=str(adiabat_TfromS_path),
        device='cuda'
    )

    MELTS102EmulatorGPU = MELTSAPI(
    isothermal_NoCr_model_path = str(MELTS102_dir / "Isothermal_NoCr.tar"),
    isothermal_Cr_model_path = str(MELTS102_dir / "Isothermal_Cr.tar"),
    isentropic_NoCr_model_path = str(MELTS102_dir / "Isentropic_NoCr.tar"),
    isentropic_Cr_model_path = str(MELTS102_dir / "Isentropic_Cr.tar"),
    device='cuda'
    )

if __name__ == "__main__": # For testing API loading...
    print("Available APIs:")
    print(f"HeFESToEmulatorCPU: {HeFESToEmulatorCPU}")
    print(f"HeFESToEmulatorGPU: {HeFESToEmulatorGPU}")
    print(f"MELTS102EmulatorCPU: {MELTS102EmulatorCPU}")
    print(f"MELTS102EmulatorGPU: {MELTS102EmulatorGPU}")