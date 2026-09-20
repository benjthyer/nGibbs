"""
High-level API for thermodynamic emulators.

Provides user-friendly interfaces for:
- Getting temperatures from isentropic states
- Computing isentropic adiabats along pressure transects
- Parsing and converting input compositions
- Managing emulator ensembles (isothermal, isentropic, temperature models)

Base class: EmulatorAPI — model-agnostic adiabat workflows
Subclass:   HeFESToAPI  — add Fortranslation EOS backend for HeFESTo
"""

import functools
import gc
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union, Dict, Any
import time
import tqdm

from .EOS_arithmetic.hefesto_vec import (
    compute as hefesto_compute,
    load_control,
    build_tables as hefesto_build_tables,
    add_metamorphic as hefesto_add_metamorphic,
    compute_within_phase_frac,
    NSMALL_REL,
)

from .EOS_arithmetic_MELTS.melts_vec import (
    load_solids as load_melts_solids,
    load_liquid as load_melts_liquid,
    compute_liquid_bulk as melts_compute_liquid_bulk,
    compute_feldspar_solution as melts_compute_feldspar_solution,
    compute_olivine_solution as melts_compute_olivine_solution,
    compute_clinopyroxene_solution as melts_compute_clinopyroxene_solution,
    compute_orthopyroxene_solution as melts_compute_orthopyroxene_solution,
    compute_spinel_solution as melts_compute_spinel_solution,
    compute_rhm_oxide_solution as melts_compute_rhm_oxide_solution,
    feldspar as _melts_feldspar,
    olivine as _melts_olivine,
    clinopyroxene as _melts_clinopyroxene,
    orthopyroxene as _melts_orthopyroxene,
    spinel as _melts_spinel,
    rhomsghiorso as _melts_rhomsghiorso,
    BARS_PER_GPA as _MELTS_BARS_PER_GPA,
    oxides_to_liquid_components as _melts_oxides_to_liquid_components,
)

from ..config.constants import (
    OXIDE_MOLAR_MASSES as oxide_molar_masses, ptt_longs, ptt_order_oxides, ptt_to_short,
    ptt_oxide_indexer, HeFESTo_snames_long, default_Oxides as _DEFAULT_LIQUID_OXIDE_LABELS,
)

ferric_to_ferrous_ratio =  (2*oxide_molar_masses['FeO'])/oxide_molar_masses['Fe2O3']
snames_dict = {name: i for i, name in enumerate(HeFESTo_snames_long)}

from .NN import _load_temperature_model
from .emulator import NN_MELTS
from ..utils.math_utils import Normalizer, get_T as _reference_adiabat


def _eval_adiabat_poly(
    P: np.ndarray,
    S,
    coefs: dict,
    features: Optional[np.ndarray] = None,
    comp_indices: Optional[Dict] = None,
) -> np.ndarray:
    """Evaluate the reference adiabat polynomial (P in GPa, T in K).

    Base: T = b0 + b_S*S + b_S2*S^2 + b_P*P + b_P2*P^2
    Compositional extension (when features and comp_indices provided): each
    element term X_i is the per-row reconstruction of the fit's total=norm
    formula-unit composition from bundle atom fractions. comp_indices carries
    {elem_weighted, cation_cols, norm_total}; see
    train_temperature_residual_fcnn._resolve_comp_feature_indices. Kept in exact
    sync with that module so reconstructed T is not systematically biased.
    """
    T = (
        coefs["b0"]
        + coefs["b_S"] * S
        + coefs["b_S2"] * S ** 2
        + coefs["b_P"] * P
        + coefs["b_P2"] * P ** 2
    )
    if comp_indices and features is not None:
        elem_weighted = comp_indices["elem_weighted"]
        cation_cols = comp_indices["cation_cols"]
        norm_total = comp_indices["norm_total"]
        n = features.shape[0]
        raw = {}
        for elem, weighted_cols in elem_weighted.items():
            v = np.zeros(n, dtype=np.float64)
            for col_idx, weight in weighted_cols:
                v += weight * features[:, col_idx]
            raw[elem] = v
        cation_sum = np.zeros(n, dtype=np.float64)
        for col_idx in cation_cols:
            cation_sum += features[:, col_idx]
        grand_total = cation_sum + raw["O"] if "O" in raw else cation_sum
        scale = np.where(np.abs(grand_total) < 1e-12, 0.0, norm_total / grand_total)
        for elem, v in raw.items():
            X = scale * v
            if f"b_{elem}" in coefs:
                T = T + coefs[f"b_{elem}"] * X
            if f"b_{elem}2" in coefs:
                T = T + coefs[f"b_{elem}2"] * X ** 2
    return np.asarray(T, dtype=np.float32)


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


def _check_ngibbs_update():
    """
    Check whether a newer nGibbs release is available on PyPI.

    Returns
    -------
    tuple[str | None, str | None, bool]
        (installed_version, latest_version, update_available)
        Any element may be None if it could not be determined.
    """
    import importlib.metadata
    import urllib.request
    import json as _json

    try:
        installed = importlib.metadata.version('nGibbs')
    except Exception:
        installed = None

    latest = None
    try:
        with urllib.request.urlopen('https://pypi.org/pypi/nGibbs/json', timeout=3) as resp:
            data = _json.loads(resp.read())
        latest = data['info']['version']
    except Exception:
        pass

    update_available = installed is not None and latest is not None and installed != latest
    return installed, latest, update_available


def _model_unavailable_error(label: str, path: str) -> RuntimeError:
    """Build a descriptive RuntimeError for a model whose file was not found at load time."""
    installed, latest, update_available = _check_ngibbs_update()

    lines = [
        f"The '{label}' emulator model is not available.",
        f"  Expected file: {path}",
        "",
    ]
    if update_available:
        lines += [
            f"A newer version of nGibbs ({latest}) is available "
            f"(you have {installed}). This model may be included in the update:",
            "    pip install --upgrade nGibbs",
        ]
    else:
        if installed:
            lines.append(
                f"You are running nGibbs {installed}, which appears to be the latest version."
            )
        lines += [
            "If you believe this model should be included, please open an issue at:",
            "    https://github.com/benjthyer/nGibbs/issues/new",
        ]
    return RuntimeError("\n".join(lines))


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

        # 'Fe' and 'Fe3' are disjoint ferrous-only/ferric-only pools in the
        # model's compositional basis (compToEl gives every component either
        # a pure 'Fe' or a pure 'Fe3' formula, never both -- see
        # HeFESToEmulatorCPU.get_property_hefesto_vectorized_from_assemblage).
        # Subtract the estimated ferric fraction out of 'Fe' so the two
        # columns partition, rather than double-count, the true total Fe.
        # Leaving 'Fe' at its original total here would inflate the bulk
        # mass-balance target that polish_masses corrects the predicted
        # assemblage against by exactly fe3_moles, biasing every predicted
        # assemblage iron-rich relative to the true composition.
        updated_values = values.astype(np.float32, copy=True)
        updated_values[:, fe_idx] = fe_total - fe3_moles

        o_idx = header_map['O']
        updated_values = np.delete(updated_values, o_idx, axis=1)
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


# --------------------------------------------------------------------------- #
# Single-directory model discovery (EmulatorAPI / HeFESToAPI / MELTSAPI)
#
# Replaces the old "long list of explicit checkpoint-path kwargs" constructors:
# callers now point at one directory (typically engine/TrainedModels/<Family>/)
# and everything else -- which checkpoint is isothermal/isentropic/openox,
# which file is the temperature FCNN, which *_Test_subset*.tar.gz quality-eval
# bundle and which standards directory belong to it -- is found by filename
# convention. See EmulatorAPI.__init__'s docstring for the convention itself.
# --------------------------------------------------------------------------- #
import re as _re
import warnings as _warnings


def _deployment_tests_root() -> Path:
    """The ``ngibbs/deployment_tests/`` directory (sibling of ``engine/``),
    searched recursively for *_Test_subset*.tar.gz quality bundles and,
    one level deep, for standards directories."""
    return Path(__file__).resolve().parent.parent / 'deployment_tests'


def _checkpoint_tokens(path: Path) -> list:
    """Lowercase '_'/non-alnum-delimited tokens from a checkpoint's filename
    (extension stripped)."""
    return [t for t in _re.split(r'[^A-Za-z0-9]+', path.stem.lower()) if t]


def _classify_checkpoint_kind(path: Path):
    """'isothermal' | 'isentropic' | 'openox' | None, from filename tokens.

    Checked in this order because MELTS's open-system checkpoints (e.g.
    '120SedIgOpen_NoCr_train2.tar') carry neither an 'npt' nor an 'nps'
    token at all -- only the closed-system isothermal/isentropic checkpoints
    do. The 'open'/'openox' check is a plain case-insensitive substring
    search on the filename, rather than a check against ``_checkpoint_tokens``,
    because real MELTS filenames glue it to its neighbor with no delimiter
    ('SedIgOpen') -- underscore-only tokenization would never isolate it.
    """
    if 'open' in path.stem.lower():
        return 'openox'
    tokens = _checkpoint_tokens(path)
    if 'npt' in tokens or 'isothermal' in tokens:
        return 'isothermal'
    if 'nps' in tokens or 'isentropic' in tokens:
        return 'isentropic'
    return None


def _classify_checkpoint_variant(path: Path):
    """'cr' | 'nocr' | None, from filename tokens (MELTS Cr/NoCr split).
    'nocr' is checked as its own whole token, never mistaken for 'cr' inside
    a longer word, since tokens are already split on non-alnum boundaries."""
    tokens = _checkpoint_tokens(path)
    if 'nocr' in tokens:
        return 'nocr'
    if 'cr' in tokens:
        return 'cr'
    return None


def _discover_checkpoints(model_dir: Path, variant: Optional[str] = None) -> dict:
    """Scan ``model_dir`` (non-recursive) for checkpoint files.

    Returns {'isothermal': Path|None, 'isentropic': Path|None,
    'openox': Path|None, 'temperature': Path|None}. When ``variant`` is
    given ('cr'/'nocr'), a *.tar whose filename carries the OTHER variant's
    token is skipped (a *.tar with no variant token at all, e.g. every
    HeFESTo checkpoint, is never filtered — HeFESTo has no Cr/NoCr concept).
    """
    model_dir = Path(model_dir)
    result = {'isothermal': None, 'isentropic': None, 'openox': None, 'temperature': None}
    for p in sorted(model_dir.glob('*.tar')):
        v = _classify_checkpoint_variant(p)
        if variant is not None and v is not None and v != variant:
            continue
        kind = _classify_checkpoint_kind(p)
        if kind is None:
            continue
        if result[kind] is not None:
            _warnings.warn(
                f"Multiple '{kind}' checkpoints found in {model_dir}"
                + (f" for variant={variant!r}" if variant else "")
                + f"; using {result[kind].name}, ignoring {p.name}."
            )
            continue
        result[kind] = p

    pt_files = sorted(model_dir.glob('*.pt'))
    if variant is not None:
        filtered = [p for p in pt_files if _classify_checkpoint_variant(p) in (None, variant)]
        if filtered:
            pt_files = filtered
    if len(pt_files) == 1:
        result['temperature'] = pt_files[0]
    elif len(pt_files) > 1:
        _warnings.warn(
            f"Multiple candidate temperature checkpoints found in {model_dir}"
            + (f" for variant={variant!r}" if variant else "")
            + f": {[p.name for p in pt_files]}; none picked (ambiguous)."
        )
    return result


def _discover_test_bundle(
    deployment_tests_root: Path,
    checkpoint_path: Path,
    model_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Find the ``*_Test_subset*.tar.gz`` quality-eval bundle matching
    ``checkpoint_path``.

    Candidates are gathered from ``model_dir`` itself first (non-recursive --
    the normal layout keeps a checkpoint and its bundle side by side, e.g.
    'TrainedModels/120/120SedIgClosed_NoCr_NPT.tar' next to
    '120SedIgClosed_NoCr_NPT_Test_subset15000.tar.gz'), then, as a fallback
    for older layouts, recursively under ``deployment_tests_root``.

    Matching is primarily by (kind, variant) classification -- the same
    ``_classify_checkpoint_kind``/``_classify_checkpoint_variant`` used for
    checkpoints themselves -- rather than a literal filename-prefix match,
    so an arbitrary training-run or descriptor suffix on either the
    checkpoint or the bundle ('_train2', '_light', '_tune3', ...) never
    breaks the match. Falls back to the old prefix heuristic (checkpoint
    basename, with a trailing ``_train<N>`` and its extension stripped, as a
    prefix of the bundle's basename) only if no bundle's kind/variant can be
    classified.

    ``model_dir`` is tried alone first, and only if it yields nothing is the
    search widened to ``deployment_tests_root`` -- so a bundle that already
    resolves unambiguously next to its checkpoint is never second-guessed
    against (and never spuriously warns about) an unrelated same-named file
    living elsewhere under ``deployment_tests_root``.
    """
    ckpt_kind = _classify_checkpoint_kind(checkpoint_path)
    ckpt_variant = _classify_checkpoint_variant(checkpoint_path)

    def _best(candidates):
        if not candidates:
            return None
        matches = [
            c for c in candidates
            if _classify_checkpoint_kind(c) == ckpt_kind
            and _classify_checkpoint_variant(c) == ckpt_variant
        ]
        if not matches:
            prefix = _re.sub(r'(_train\d+)?\.tar$', '', checkpoint_path.name, flags=_re.IGNORECASE)
            matches = [c for c in candidates if c.name.lower().startswith(prefix.lower())]
        if not matches:
            return None
        if len(matches) > 1:
            ckpt_stem = checkpoint_path.stem.lower()
            matches.sort(key=lambda c: (not c.name.lower().startswith(ckpt_stem), c.name))
            _warnings.warn(
                f"Multiple test bundles match checkpoint {checkpoint_path.name}: "
                f"{[m.name for m in matches]}; using {matches[0].name}."
            )
        return matches[0]

    if model_dir is not None and Path(model_dir).is_dir():
        found = _best(sorted(Path(model_dir).glob('*Test_subset*.tar.gz')))
        if found is not None:
            return found

    if deployment_tests_root.is_dir():
        return _best(sorted(deployment_tests_root.rglob('*Test_subset*.tar.gz')))
    return None


def _name_tokens(name: str) -> set:
    """Fuzzy, case/plural-insensitive token set for a directory or file name,
    splitting on non-alnum boundaries, camelCase, ALLCAPSWord boundaries, and
    letter-digit boundaries (so 'MELTS120' -> {'melts', '120'} and
    'MELTSIsobaricStandards' -> {'melts', 'isobaric', 'standard'})."""
    s = _re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', name)
    s = _re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', '_', s)
    s = _re.sub(r'(?<=[A-Za-z])(?=[0-9])', '_', s)
    s = _re.sub(r'(?<=[0-9])(?=[A-Za-z])', '_', s)
    out = set()
    for t in _re.split(r'[^A-Za-z0-9]+', s):
        t = t.lower()
        if not t:
            continue
        if t.endswith('s') and len(t) > 3 and not t.isdigit():
            t = t[:-1]
        out.add(t)
    return out


def _discover_standards_dir(deployment_tests_root: Path, model_dir: Path) -> Optional[Path]:
    """Best-effort match of a 'standards' comparison directory (MELTS's raw
    alphaMELTS isobaric-cooling standards, or HeFESTo's MELTStable-format CSV
    directory) under ``deployment_tests_root`` to ``model_dir``'s model
    family, by fuzzy token overlap (see ``_name_tokens``).

    A model_dir named after a bare version number (MELTS's '120', say) shares
    no name token at all with a directory like 'MELTSIsobaricStandards' --
    the overlap is only findable one level down, in a numeric-named
    subdirectory ('MELTSIsobaricStandards/120'). So for each 'standard'
    candidate, a numeric-named subdirectory matching one of model_dir's own
    numeric tokens counts as a match in its own right (and is what gets
    returned for that candidate), on top of plain top-level name overlap.
    Returns None if nothing scores > 0 either way.
    """
    if not deployment_tests_root.is_dir():
        return None
    target = _name_tokens(model_dir.name)
    numeric_tokens = {t for t in target if t.isdigit()}
    candidates = [d for d in deployment_tests_root.iterdir()
                  if d.is_dir() and 'standard' in _name_tokens(d.name)]
    best_score, best_result = 0, None
    for c in candidates:
        score = len(_name_tokens(c.name) & target)
        result = c
        if numeric_tokens:
            for sub in sorted(c.iterdir()):
                if sub.is_dir() and sub.name in numeric_tokens:
                    score = max(score, 1)
                    result = sub
                    break
        if score > best_score:
            best_score, best_result = score, result
    return best_result


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

    # Composition space of this API's own Test_subset bundles, for
    # test()'s training-coverage plots (see training_coverage.py):
    # 'oxides' (oxide mole fraction, MELTS's native convention -- also the
    # right default for a bare EmulatorAPI, since MELTSAPI builds its two
    # sub-APIs directly as plain EmulatorAPI instances) or 'elements'
    # (elemental mole fraction, HeFESTo's convention -- see
    # emulator.py's ``reorder_input_table``). HeFESToAPI overrides this.
    _composition_space = 'oxides'

    def __init__(
        self,
        model_dir: Union[str, Path],
        *,
        device: str = 'cpu',
        verbose: bool = False,
        mass_balance: str = 'iterative',
        variant: Optional[str] = None,
    ):
        """
        Initialize the emulator API by pointing at a single directory.

        Parameters
        ----------
        model_dir : str or Path
            Directory holding this model family's checkpoint files (e.g.
            ``engine/TrainedModels/HeFESTo_Adiabats/``). Checkpoints are
            auto-discovered by filename convention, scanning ``model_dir``
            itself (non-recursive):

            - a ``*.tar`` whose filename contains 'NPT' or 'isothermal' is
              the isothermal (closed, T-input) emulator
            - a ``*.tar`` containing 'NPS' or 'isentropic' is the isentropic
              (closed, S-input) emulator
            - a ``*.tar`` containing 'open' or 'openox' is the open-oxygen
              (fO2-buffered) emulator
            - a single ``*.pt`` file is the temperature FCNN

            At least one of the isothermal/isentropic checkpoints must be
            found, or construction raises ``FileNotFoundError``.

            The matching ``*_Test_subset*.tar.gz`` quality-eval bundle for
            each discovered checkpoint (ground truth for ``self.test()``) is
            auto-discovered by kind/variant, checked first inside
            ``model_dir`` itself (bundles normally sit right next to their
            checkpoint) and, as a fallback, recursively under the sibling
            ``deployment_tests/`` directory — wherever you've organized it.
            A checkpoint with no matching bundle is simply skipped from
            ``self.test()`` (see its docstring).
        device : str, default='cpu'
            Torch device ('cpu' or 'cuda')
        verbose : bool, default=False
            Print initialization messages
        mass_balance : {'iterative', 'pinv', 'none'}, default='iterative'
            Mass-balance correction every wrapped `NN_MELTS` applies by default
            (see `NN_MELTS.forwardMB`). The choice is model-agnostic.
        variant : {'cr', 'nocr'}, optional
            Internal — restricts checkpoint discovery within ``model_dir`` to
            filenames carrying this Cr/NoCr token (used by ``MELTSAPI`` to
            build its two sub-APIs out of one shared ``model_dir``, since
            both variants' checkpoints live side by side there). Leave as
            None when constructing ``EmulatorAPI``/``HeFESToAPI`` directly.
        """
        if verbose:
            print("[INFO] Initializing EmulatorAPI...")

        self.device = torch.device(device)
        self.verbose = verbose
        self.modelType = self.__class__.__name__
        self._mass_balance = mass_balance
        self._model_dir = str(Path(model_dir).resolve())
        self._variant = variant

        found = _discover_checkpoints(Path(model_dir), variant=variant)
        isothermal_model_path = found['isothermal']
        isentropic_model_path = found['isentropic']
        openox_model_path = found['openox']
        temperature_model_path = found['temperature']

        if isothermal_model_path is None and isentropic_model_path is None:
            raise FileNotFoundError(
                f"No isothermal (NPT) or isentropic (NPS) checkpoint (*.tar) found in "
                f"{model_dir}" + (f" for variant={variant!r}" if variant else "") +
                ". Expected a filename containing 'NPT' or 'NPS'."
            )

        # Store paths so _clone_as_cpu can rebuild this instance on another device
        self._iso_path  = str(isothermal_model_path) if isothermal_model_path is not None else None
        self._isen_path = str(isentropic_model_path) if isentropic_model_path is not None else None
        self._temp_path = str(temperature_model_path) if temperature_model_path is not None else None
        self._open_path = str(openox_model_path) if openox_model_path is not None else None

        # Ground-truth data for self.test(): one ML-ready bundle per emulator,
        # keyed by {'isothermal', 'isentropic', 'openox'}, auto-discovered
        # under the sibling deployment_tests/ directory. self.test() scores
        # every configured emulator whose bundle file exists and writes the
        # metrics tables into self.home_dir (== model_dir) / 'deployment_test'.
        _dep_root = _deployment_tests_root()
        self._test_bundles = {}
        for _kind, _path in (('isothermal', isothermal_model_path),
                              ('isentropic', isentropic_model_path),
                              ('openox', openox_model_path)):
            if _path is None:
                continue
            _bundle = _discover_test_bundle(_dep_root, _path, model_dir=Path(model_dir))
            if _bundle is not None:
                self._test_bundles[_kind] = str(_bundle)

        if isothermal_model_path is not None:
            if verbose:
                print(f"  Loading isothermal emulator: {isothermal_model_path}")
            self.isothermal_emulator = self._load_emulator(isothermal_model_path, device, verbose, mass_balance)
        else:
            self.isothermal_emulator = None

        if isentropic_model_path is not None:
            if verbose:
                print(f"  Loading isentropic emulator: {isentropic_model_path}")
            self.isentropic_emulator = self._load_emulator(isentropic_model_path, device, verbose, mass_balance)
        else:
            self.isentropic_emulator = None

        if openox_model_path is not None:
            if verbose:
                print(f"  Loading open oxygen emulator: {openox_model_path}")
            self.open_emulator = self._load_emulator(openox_model_path, device, verbose, mass_balance)
        else:
            self.open_emulator = None

        if temperature_model_path is not None:
            if verbose:
                print(f"  Loading temperature FCNN: {temperature_model_path}")
            self._setup_temperature_model(temperature_model_path, verbose)
        elif verbose:
            print("  No temperature model found; temperature predictions will be unavailable.")

        # Use first available emulator to build the composition parser
        _parser_src = self.isothermal_emulator or self.isentropic_emulator or self.open_emulator
        self.parser = InputParser(_parser_src) if _parser_src is not None else None

        # _cpu_api is populated here for direct EmulatorAPI instances; for
        # subclasses the __init_subclass__ wrapper calls _clone_as_cpu after
        # the subclass __init__ has finished storing its own extra state.
        self._cpu_api = None
        if type(self) is EmulatorAPI and self.device.type != 'cpu':
            self._cpu_api = self._clone_as_cpu()

        if verbose:
            print("[INFO] EmulatorAPI initialized successfully.")

    def __init_subclass__(cls, **kwargs):
        """Wrap every subclass __init__ to auto-create a CPU twin after init.

        After a GPU subclass finishes its own __init__ (including all super()
        calls and subclass-specific setup), this wrapper calls _clone_as_cpu()
        if no CPU twin was already set.  Subclasses just need to:
          1. Store any extra constructor args they need for cloning.
          2. Override _clone_as_cpu() to forward those args.
        No manual _finalize_cpu() call is required.
        """
        super().__init_subclass__(**kwargs)
        original_init = cls.__init__

        @functools.wraps(original_init)
        def _auto_cpu(self, *args, **kw):
            original_init(self, *args, **kw)
            # Only fire from the leaf class being constructed.
            # When a super().__init__() chain is running, intermediate wrappers
            # see type(self) != cls and skip — so _clone_as_cpu() is only called
            # once, after the most-derived __init__ has stored all its state.
            if type(self) is cls and self.device.type != 'cpu' and self._cpu_api is None:
                self._cpu_api = self._clone_as_cpu()

        cls.__init__ = _auto_cpu

    def _clone_as_cpu(self):
        """Return a CPU-device copy of this EmulatorAPI.

        Subclasses must override this to forward their extra constructor
        arguments (e.g. control_path for HeFESToAPI).
        """
        return EmulatorAPI(
            self._model_dir,
            device='cpu', verbose=False,
            mass_balance=self._mass_balance,
            variant=self._variant,
        )

    @property
    def home_dir(self) -> Path:
        """The model_dir this emulator was constructed from. ``test()`` writes
        its outputs into ``home_dir / 'deployment_test'``.
        """
        return Path(self._model_dir)

    def _coverage_standards(self, kind: str):
        """Hook for ``test()``'s training-coverage plots: return a
        ``(P, T, S, comp_dict, rock_ids)`` standard-rock overlay tuple (as
        ``training_coverage.melts_standard_points``/``hefesto_standard_points``
        return) for the emulator named ``kind`` ('isothermal'/'isentropic'/
        'openox'), or None to omit the 'x' overlay entirely.

        The base implementation defers to an instance attribute,
        ``self._coverage_standards_fn``, when one is set -- this is how
        ``MELTSAPI`` wires standards into its two plain-``EmulatorAPI``
        sub-APIs without either of them needing their own subclass.
        ``HeFESToAPI`` overrides this method directly instead.
        """
        fn = getattr(self, '_coverage_standards_fn', None)
        return fn(kind) if fn is not None else None

    def test(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        *,
        max_samples: Optional[int] = None,
        seed: int = 1337,
        write_outputs: bool = True,
        verbose: bool = True,
        allow_missing_models: bool = False,
        plot_coverage: bool = True,
    ) -> Dict[str, object]:
        """Run the deployable emulator quality test against the bundled ground truth.

        Scores every configured emulator (``self._test_bundles`` maps
        'isothermal' / 'isentropic' / 'openox' to an ML-ready bundle) the way
        ``ModelComparison`` scores a field of models, but for the single model
        this API wraps: per-phase precision / recall / proportion-in-dataset /
        abundance error / per-oxide within-phase composition error, assembled
        into one ``metrics x phases`` table per emulator (phase columns ordered
        most- to least-abundant in the bundle).

        A bundle is "missing" when its emulator was never loaded (no checkpoint
        path given at construction — e.g. MELTS120 has no temperature model at
        all, and its Cr variant has no openox checkpoint) or its bundle file
        isn't present on disk. Either way it's always skipped from evaluation
        (only the available portions are ever scored) — ``allow_missing_models``
        controls whether that gap is also fatal:

        - ``False`` (default): evaluate every available emulator exactly as
          below, then raise ``RuntimeError`` naming whichever configured
          emulator(s) were skipped. Use this for a deployment gate where every
          configured model is expected to be present.
        - ``True``: keep the previous behaviour — skipped emulators are noted
          in the returned ``'skipped_bundles'`` list and never fail the test.
          Use this when some models are known/expected to be absent (e.g. no
          temperature model, or a Cr-variant model that was never trained) and
          you only want the available portions graded.

        Quality metrics only — tolerances / pass-fail on the metrics THEMSELVES
        are applied by a separate layer that consumes the returned tables
        (``assert_quality``); ``allow_missing_models`` only governs whether
        missing models are themselves treated as a failure.

        ``plot_coverage`` (default True): for each evaluated emulator, also
        render the 4 training-data-context diagnostic plots (P vs S, P vs T,
        SiO2/Si Harker grid, MgO/Mg Harker grid -- see
        ``ngibbs.deployment_tests.training_coverage``) of that emulator's own
        Test_subset bundle, with real standard-rock conditions overlaid as
        'x' markers when a subclass provides them (``_coverage_standards``).
        Written to ``<output_dir>/plots/`` alongside the CSVs; skipped
        entirely when ``write_outputs`` is False. A plotting failure for one
        emulator only warns -- it never fails the test.

        Returns
        -------
        dict with:
            'quality_metrics' : {emulator_name: DataFrame}
            'meta'            : {emulator_name: dict}
            'skipped_bundles' : list of emulator names whose bundle was absent
            'coverage_plots'  : {emulator_name: {plot_key: Path}} (only present
                when ``plot_coverage`` and ``write_outputs`` are both True)
        Each table is written to ``<output_dir>/emulator_quality_<name>.csv``
        (NaN as ``--``) when ``write_outputs``, alongside
        ``emulator_quality_legend.txt``.
        """
        from ngibbs.deployment_tests import evaluate_emulator_quality
        from ngibbs.deployment_tests.emulator_quality import legend_text

        if not self._test_bundles:
            raise RuntimeError(
                f"{self.__class__.__name__} has no test bundles configured "
                "(test_bundles={}). Wire them in the constructor / model spec."
            )
        out_dir = Path(output_dir) if output_dir is not None else self.home_dir / 'deployment_test'

        quality: Dict[str, object] = {}
        meta: Dict[str, object] = {}
        skipped = []
        for name, bundle in self._test_bundles.items():
            emu = getattr(self, {'openox': 'open_emulator'}.get(name, f'{name}_emulator'), None)
            if emu is None or not Path(bundle).exists():
                skipped.append(name)
                continue
            if verbose:
                print(f"[test] {self.__class__.__name__} [{name}]: {Path(bundle).name}")
            r = evaluate_emulator_quality(
                self, bundle, emulator_name=name, max_samples=max_samples, seed=seed,
            )
            quality[name] = r['phase_quality_metrics']
            meta[name] = r['meta']
            if write_outputs:
                out_dir.mkdir(parents=True, exist_ok=True)
                csv_path = out_dir / f'emulator_quality_{name}.csv'
                r['phase_quality_metrics'].to_csv(csv_path, na_rep='--')
                if verbose:
                    print(f"[test]   n={r['meta']['n_samples']:,}  "
                          f"recon_resid_l2_mean={r['meta']['recon_residual_l2_mean']:.4g}  "
                          f"-> {csv_path.name}")

        if skipped and verbose:
            print(f"[test]   skipped (bundle file or emulator absent): {skipped}")
        if write_outputs and quality:
            (out_dir / 'emulator_quality_legend.txt').write_text(legend_text())

        if skipped and not allow_missing_models:
            raise RuntimeError(
                f"{self.__class__.__name__}.test(): configured emulator(s) "
                f"{skipped} have no available checkpoint/bundle and "
                "allow_missing_models=False. Pass allow_missing_models=True to "
                "evaluate only the available portions instead of failing."
            )

        result = {'quality_metrics': quality, 'meta': meta, 'skipped_bundles': skipped}

        if plot_coverage and write_outputs:
            from ngibbs.deployment_tests.training_coverage import plot_training_coverage
            plots_dir = out_dir / 'plots'
            coverage_plots = {}
            for name, bundle in self._test_bundles.items():
                if name in skipped or not Path(bundle).exists():
                    continue
                try:
                    coverage_plots[name] = plot_training_coverage(
                        bundle, plots_dir,
                        composition_space=self._composition_space,
                        model_label=f'{self.modelType}_{name}',
                        standards=self._coverage_standards(name),
                    )
                    if verbose:
                        print(f"[test]   coverage plots -> {plots_dir}/{self.modelType}_{name}_*.png")
                except Exception as e:
                    _warnings.warn(f"Training-coverage plots failed for '{name}' bundle: {e}")
            result['coverage_plots'] = coverage_plots

        return result

    def assert_quality(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        *,
        max_samples: Optional[int] = None,
        seed: int = 1337,
        write_outputs: bool = True,
        verbose: bool = True,
        allow_missing_models: bool = False,
    ) -> Dict[str, object]:
        """Run ``self.test()`` and enforce the deployment quality gate on the result.

        Applies the fixed pass/fail tolerances in
        ``ngibbs.deployment_tests.quality_thresholds`` to every ``quality_metrics``
        table returned by ``test()`` -- phase-presence precision/recall (bucketed
        by average GT modal abundance), and for major phases the abundance and
        MgO/FeO/SiO2 composition errors. Subclasses whose ``test()`` also returns
        ``meltstable_property_errors`` (currently ``HeFESToAPI``) additionally get
        those EOS-property tolerances checked.

        ``allow_missing_models`` is forwarded to ``test()`` unchanged: with the
        default ``False``, a configured emulator with no available checkpoint/
        bundle fails the gate (via ``test()``'s own ``RuntimeError``) just like
        any other quality failure; ``True`` lets ``test()`` silently grade only
        the available portions (see ``test()``'s docstring).

        Raises
        ------
        EmulatorQualityError
            If any threshold is missed; the exception's ``.failures`` lists every
            violation (see ``QualityFailure``).
        RuntimeError
            If ``allow_missing_models=False`` and a configured emulator has no
            available checkpoint/bundle (raised by ``test()``).

        Returns
        -------
        The same dict ``test()`` returns, plus ``'failures': []`` on success.
        """
        from ngibbs.deployment_tests.quality_thresholds import (
            check_meltstable_quality,
            check_phase_quality,
            EmulatorQualityError,
        )

        result = self.test(
            output_dir=output_dir, max_samples=max_samples, seed=seed,
            write_outputs=write_outputs, verbose=verbose,
            allow_missing_models=allow_missing_models,
        )
        failures = check_phase_quality(result['quality_metrics'])
        if 'meltstable_property_errors' in result:
            failures += check_meltstable_quality(result['meltstable_property_errors'])

        result['failures'] = failures
        if failures:
            raise EmulatorQualityError(self.__class__.__name__, failures, result=result)
        if verbose:
            print(f"[test]   {self.__class__.__name__}: all deployment quality checks passed.")
        return result

    def _get_cpu_func(self, func):
        """Return the CPU-API equivalent of a GPU emulator bound method, or None.

        Inspects func.__self__ to identify which of the two emulators owns the
        method, then looks up the same method name on self._cpu_api's emulator.
        Returns None when no CPU twin exists or func is not an emulator method.
        """
        if self._cpu_api is None:
            return None
        if not (hasattr(func, '__self__') and hasattr(func, '__func__')):
            return None
        owner = func.__self__
        name  = func.__func__.__name__
        if owner is self.isothermal_emulator:
            return getattr(self._cpu_api.isothermal_emulator, name, None)
        if owner is self.isentropic_emulator:
            return getattr(self._cpu_api.isentropic_emulator, name, None)
        return None

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
    def _load_emulator(model_path: Union[str, Path], device: str, verbose: bool = True,
                       mass_balance: str = 'iterative') -> Optional[NN_MELTS]:
        """Load and wrap a checkpoint as NN_MELTS emulator, or return None with a warning."""
        import warnings
        from .NN import rebuild_MELTS_model

        model_path = Path(model_path)
        if not model_path.exists():
            if verbose:
                print(
                f"[nGibbs] Model file not found: {model_path}. "
                "This model will be unavailable."
                )
            return None

        model = rebuild_MELTS_model(str(model_path))
        return NN_MELTS(model, cuda=(device == 'cuda'), mass_balance=mass_balance)

    def _setup_temperature_model(self, checkpoint_path: Union[str, Path], verbose: bool = False) -> None:
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
        if self.isothermal_emulator is None:
            raise _model_unavailable_error('isothermal', self._iso_path)
        if self.isentropic_emulator is None:
            raise _model_unavailable_error('isentropic', self._isen_path)

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
            property_names=['S'],
        )
        S_values = torch.tensor(
            eos_props['S'], dtype=torch.float32, device=self.device
        )  # (n_S,)
        # print(f"Reference entropies (J/g/K) at P=0.0001 GPa: {S_values.cpu().numpy()}")

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


    # Batches smaller than this threshold are forwarded to the CPU API (when
    # one exists) to avoid the GPU kernel-launch overhead that dominates for
    # tiny inputs.  8 192 rows is a conservative crossover empirically valid
    # for both NN forward passes and the HeFESTo EOS kernel.
    _CPU_BATCH_THRESHOLD: int = 8_192

    def _staged_forward(
        self,
        func,
        input_tensor: torch.Tensor,
        batch_size: int,
        **kwargs
    ) -> Union[torch.Tensor, list]:
        """Execute function in batches, routing small inputs to the CPU API.

        Parameters
        ----------
        func : callable
            Function to apply (e.g., model forward pass).
        input_tensor : torch.Tensor
            Input batch tensor.
        batch_size : int
            Maximum rows per GPU chunk.
        _cpu_fallback : callable, keyword-only (popped before forwarding)
            Explicit CPU function to call for small batches.  Used by
            _compute_bulk_EOS_properties; NN forward calls are routed
            automatically via _get_cpu_func.
        **kwargs :
            Additional arguments forwarded to func (and cpu_fallback).

        Returns
        -------
        torch.Tensor or list of torch.Tensor
            Concatenated outputs from all batches.
        """
        # _cpu_fallback is an internal sentinel — pop before forwarding to func
        cpu_fallback = kwargs.pop('_cpu_fallback', None)

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

        # ── Small-batch CPU routing ───────────────────────────────────────────
        # For NN forward passes, _get_cpu_func resolves the CPU twin
        # automatically.  For EOS calls, an explicit _cpu_fallback is supplied.
        if n_samples < self._CPU_BATCH_THRESHOLD:
            cpu_fn = cpu_fallback or self._get_cpu_func(func)
            if cpu_fn is not None:
                return cpu_fn(input_tensor.detach().cpu(), **kwargs)

        # ── GPU path (existing batched logic) ─────────────────────────────────
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
        has_logfO2 = 'logfO2-QFM(System_main)' in headers

        if has_temperature and has_entropy:
            raise ValueError(
                "Input headers cannot include both temperature and entropy features"
            )
        if has_logfO2 and has_entropy:
            raise ValueError(
                "Input headers cannot include both logfO2-QFM and entropy features: "
                "an open-system isentropic model is non-physical and not supported."
            )

        if has_logfO2:
            if self.open_emulator is None:
                raise _model_unavailable_error('open oxygen (fO2-buffered)', self._open_path or '<not specified>')
            emulator = self.open_emulator
        elif has_entropy:
            if self.isentropic_emulator is None:
                raise _model_unavailable_error('isentropic', self._isen_path)
            emulator = self.isentropic_emulator
        elif has_temperature:
            if self.isothermal_emulator is None:
                raise _model_unavailable_error('isothermal', self._iso_path)
            emulator = self.isothermal_emulator
        else:
            raise ValueError("Input headers must include either temperature or entropy features: T(K)(System_main) / 'Temperature(System_main)' or S(J/g/K)(System_main) / 'S(System_main)'.\n You have: {}".format(headers))

        if self.parser is None:
            raise RuntimeError(
                "No emulator models could be loaded — all specified model files were missing. "
                "Check the warnings printed at initialization for details."
            )

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

        non_chem_features = len(emulator.ml_indexer.featureNames)
        chem_sum = reordered[:,non_chem_features:].sum(dim=1)
        reordered[:,non_chem_features:] = reordered[:,non_chem_features:] / chem_sum.unsqueeze(1)

        if has_logfO2:
            modeltype = 'openox'
        elif has_entropy:
            modeltype = 'isentropic'
        else:
            modeltype = 'isothermal'

        return reordered.to(self.device), modeltype



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
        elif modeltype == 'openox':
            if 'temperature' in outputs:
                raise ValueError("'temperature' output was passed for an open oxygen model... Did you mean to use an isentropic model?")
            emulator = self.open_emulator
        else:
            raise ValueError(f"parser determined model is not recognized: {modeltype} must be 'isothermal', 'isentropic', or 'openox'")

        if 'temperature' in outputs:
            if 'chem_out' not in outputs:
                outputs = list(outputs) + ['chem_out']
                # print("[INFO] Adding 'chem_out' to outputs for temperature calculation.")
            if 'phase_moles' not in outputs:
                outputs = list(outputs) + ['phase_moles']
                # print("[INFO] Adding 'phase_moles' to outputs for temperature calculation.")
            get_temp = True
            outputs.remove('temperature') # temperature not recognized as arg for NN.
        else:
            get_temp = False

        if 'ptt_out' in outputs:
            get_ptt_tables = True
            if 'phase_tables' not in outputs:
                outputs = list(outputs) + ['phase_tables']
                # print("[INFO] Adding 'phase_tables' to outputs for ptt_out formatting.")
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
            results['ptt_out'] = self.make_ptt_out(features, results['phase_tables'][0], results['phase_tables'][1], isentropic=(modeltype=='isentropic'), openox=(modeltype=='openox'))

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
        elif modeltype == 'openox':
            if 'temperature' in outputs:
                raise ValueError("'temperature' output was passed for an open oxygen model... Did you mean to use an isentropic model?")
            emulator = self.open_emulator
        else:
            raise ValueError(f"parser determined model is not recognized: {modeltype} must be 'isothermal', 'isentropic', or 'openox'")

        if 'temperature' in outputs:
            if 'chem_out' not in outputs:
                outputs = list(outputs) + ['chem_out']
                # print("[INFO] Adding 'chem_out' to outputs for temperature calculation.")
            if 'phase_moles' not in outputs:
                outputs = list(outputs) + ['phase_moles']
                # print("[INFO] Adding 'phase_moles' to outputs for temperature calculation.")
            get_temp = True
            outputs.remove('temperature') # temperature not recognized as arg for NN.
        else:
            get_temp = False

        if 'ptt_out' in outputs:
            get_ptt_tables = True
            if 'phase_tables' not in outputs:
                outputs = list(outputs) + ['phase_tables']
                # print("[INFO] Adding 'phase_tables' to outputs for ptt_out formatting.")
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
            results['ptt_out'] = self.make_ptt_out(features, results['phase_tables'][0], results['phase_tables'][1], isentropic=(modeltype=='isentropic'), openox=(modeltype=='openox'))

        return results

    def get_T(
        self,
        features: Union[np.ndarray, torch.Tensor],
        normalize_features: bool = True,
    ) -> torch.Tensor:
        """
        Get temperature from isentropic emulator output.

        If the loaded checkpoint contains p_feature_idx / s_feature_idx (written by
        train_temperature_residual_fcnn.py), the NN predicts a residual against
        reference_adiabat(P, S) and the reference is added back.  Older checkpoints
        that lack these keys fall back to direct temperature prediction.

        Parameters
        ----------
        features : array-like
            Input features (B, F) with intensive variables + composition (raw, unnormalized)
        normalize_features : bool, default=True
            Whether to normalize input features

        Returns
        -------
        torch.Tensor
            Temperatures (B,) in Kelvin
        """
        features = torch.as_tensor(features, dtype=torch.float32, device=self.device)

        # Compute reference adiabat if checkpoint carries P/S indices
        p_idx = self.temperature_payload.get("p_feature_idx")
        s_idx = self.temperature_payload.get("s_feature_idx")
        adiabat_coefs = self.temperature_payload.get("adiabat_coefs")
        comp_indices = self.temperature_payload.get("coef_feature_indices")
        if p_idx is not None and s_idx is not None:
            P_raw = features[:, p_idx].detach().cpu().numpy().astype(np.float64)
            S_raw = features[:, s_idx].detach().cpu().numpy().astype(np.float64)
            if self.modelType == "MELTSAPI":
                P_raw = P_raw / 10000.0  # bars → GPa
            if adiabat_coefs is not None:
                features_cpu = features.detach().cpu().numpy().astype(np.float64) if comp_indices else None
                T_ref_K = _eval_adiabat_poly(P_raw, S_raw, adiabat_coefs, features=features_cpu, comp_indices=comp_indices)
            else:
                T_ref_K = np.asarray(_reference_adiabat(P_raw, S_raw), dtype=np.float32)
            if self.modelType == "MELTSAPI":
                T_ref_K = T_ref_K - 273.15  # K → Celsius
            T_ref = torch.tensor(T_ref_K, dtype=torch.float32, device=self.device)
        else:
            T_ref = None

        temp_input_norm = self.temp_input_normalizer.norm(features)
        with torch.no_grad():
            output_norm = self.temperature_model(temp_input_norm)
        output_denorm = self.temp_output_normalizer.denorm(output_norm).squeeze(-1)

        return output_denorm if T_ref is None else T_ref + output_denorm
    
    def make_ptt_out(self, features, phaseOxWt, phaseMassNorm, isentropic=None, openox=False):
        """
        Create output in format of ptt pandas tables. This should only be used internally.
        Future dev? Integrate thermodynamic properties calcs already done in ptt. Only once an independent parallel pythonic method implemented

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
        openox : bool, default=False
            Whether the input features are from an open oxygen (fO2-buffered) model

        Returns
        -------
        dict
            of pandas DataFrames matching ptt format for each phase
        """

        if openox:
            indexer = self.open_emulator.ml_indexer
        elif isentropic:
            indexer = self.isentropic_emulator.ml_indexer
        else:
            indexer = self.isothermal_emulator.ml_indexer

        output = {} # Initialize dict of Pandas outputs
        P_bar = features[:, indexer.featureNames.index('Pressure(System_main)')].detach().cpu().numpy()

        if isentropic: # Isentropic models - we have entropy but not temperature as input features.
            T_C = np.full((features.size(0),), fill_value=np.nan)
            s = features[:, indexer.featureNames.index('S(System_main)')].detach().cpu().numpy()
        else: # Isothermal and open oxygen models have T as input
            T_C = features[:, indexer.featureNames.index('Temperature(System_main)')].detach().cpu().numpy()
            s = np.full((features.size(0),), fill_value=np.nan)

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
            # print(internal_name)
            if internal_name in indexer.compositionally_variable_phases:
                # print(f"Phase {internal_name} is compositionally variable!.")
                colnames = [Ox + ptt_short for Ox in ptt_order_oxides]

                outTable = np.zeros((features.size(0), len(ptt_order_oxides)))
                outTable[:, ptt_Ox_Idx] = phaseOxWt[:, indexer.comp_phasedict[internal_name], :]

                # Handle iron shenanigans.
                Fe2 = outTable[:, ptt_oxide_indexer['FeO']]/oxide_molar_masses['FeO']
                Fe3 = outTable[:, ptt_oxide_indexer['Fe2O3']]/oxide_molar_masses['Fe2O3']*2
                Fet = Fe2 + Fe3
                outTable[:, ptt_oxide_indexer['Fe3Fet']] = np.where(Fet > 0, Fe3 / np.where(Fet > 0, Fet, 1.0), 0.0)
                outTable[:, ptt_oxide_indexer['FeOt']] = Fet * oxide_molar_masses['FeO']

                output[ptt_long] = pd.DataFrame(outTable, columns=colnames)

            mass_array[:,i] = phaseMassNorm[:, internalPhaseCol].detach().cpu().numpy()
            mass_cols.append(ptt_long)
            all_mass_cols.append('mass_g' + ptt_short)
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
    HeFESTo emulator API with elastic/rigid EOS backend.

    Extends EmulatorAPI with elastic/rigid bulk property evaluation,
    enabling isentrope computation and physical property retrieval for
    the HeFESTo mineral physics database.
    """

    _composition_space = 'elements'

    def __init__(
        self,
        model_dir: Union[str, Path],
        *,
        device: str = 'cpu',
        verbose: bool = False,
        control_path: Optional[Union[str, Path]] = None,
        param_dir: Optional[Union[str, Path]] = None,
        npz_path: Optional[Union[str, Path]] = None,
        mass_balance: str = 'iterative',
    ):
        """
        Initialize HeFESTo API by pointing at a single directory.

        Parameters
        ----------
        model_dir : str or Path
            Directory holding this HeFESTo model's checkpoint files (e.g.
            ``engine/TrainedModels/HeFESTo_Adiabats/``). See
            ``EmulatorAPI.__init__``'s docstring for the checkpoint/test-
            bundle discovery convention (isothermal/isentropic/temperature
            checkpoints found in ``model_dir`` itself; the matching
            ``*_Test_subset*.tar.gz`` quality bundles found recursively under
            the sibling ``deployment_tests/``). The MELTStable-format
            standards directory for the HeFESTo-specific bulk-property/
            phase-abundance comparison in ``self.test()`` is auto-discovered
            the same way, by fuzzy name match against ``model_dir``'s name
            (e.g. ``engine/TrainedModels/HeFESTo_Adiabats`` ->
            ``deployment_tests/HeFESToAdiabatStandards``); left unconfigured
            (comparison skipped, not failed) if nothing matches.
        device : str, default='cpu'
            Torch device ('cpu' or 'cuda')
        verbose : bool, default=False
            Print initialization messages
        control_path : str or Path, optional
            Path to a HeFESTo control file. Required to use
            get_property_hefesto_vectorized_from_assemblage. These are shared
            physics-parameter files, not part of ``model_dir``'s per-checkpoint
            discovery — defaults to the packaged BENCHMARK control file.
        param_dir : str or Path, optional
            Override for the parameter directory embedded in the control file
            (e.g. path to HeFESTo_Parameters_010123/).
        npz_path : str or Path, optional
            Path to a pre-built DOS-table .npz file for the HeFESTo EOS kernel.
        """
        if verbose:
            print("[INFO] Initializing HeFESToAPI...")

        super().__init__(
            model_dir,
            device=device,
            verbose=verbose,
            mass_balance=mass_balance,
        )

        # Directory of MELTStable adiabat CSVs for the HeFESTo-specific part of
        # self.test() (bulk-property + phase-abundance comparison vs the
        # internal vectorised EOS), auto-discovered by fuzzy name match
        # against model_dir under the sibling deployment_tests/ directory.
        _standards = _discover_standards_dir(_deployment_tests_root(), Path(model_dir))
        self._test_meltstable_dir = str(_standards) if _standards is not None else None

        assert self.isothermal_emulator is not None and self.isentropic_emulator is not None, (
            "HeFESToAPI requires both an isothermal (NPT) and isentropic (NPS) "
            f"checkpoint in {model_dir}."
        )
        assert self.isothermal_emulator.ml_indexer.label_names == self.isentropic_emulator.ml_indexer.label_names # Assume the label names are model-agnostic

        # Known label collision: some checkpoints' ml_indexer.label_names carry the
        # bare string 'magnetite' at BOTH ml component index 4 (3rd member of the
        # 'spinel' phase group -- physically HeFESTo's 'smag', the spinel-
        # solid-solution magnetite endmember) and index 56 (5th member of
        # 'ferropericlase' -- physically 'mag', plain oxide-phase magnetite).
        # Confirmed by matching phase position: ML 'spinel' = [spinel,
        # hercynite, magnetite, picro-chromite] lines up 1:1 with physics
        # phase 'sp' = [sp, hc, smag, picr]. snames_dict is keyed on the current
        # composite 'species : phase' convention (see constants.COMPONENT_KEY_SEP),
        # which has no bare 'magnetite' entry at all, so a plain lookup raises
        # KeyError for both occurrences before either can be disambiguated. The
        # root cause is a stale COMPONENT_ABBREVIATION_OVERRIDES['smag'] entry
        # (ngibbs/utils/file_utils.py) that produced this label at the time this
        # model's label metadata was generated; fixed there for future models
        # (whose label_names already arrive as the composite name and resolve
        # directly), but that fix cannot retroactively change this checkpoint's
        # saved label_names, so bare 'magnetite' is disambiguated by position here.
        def _resolve_snames_idx(i, name):
            if name == 'magnetite':
                composite = 'magnetite : spinel' if i == 4 else 'magnetite : ferropericlase'
                return snames_dict[composite]
            return snames_dict[name]

        self.PropertyIDX = np.array([
            _resolve_snames_idx(i, name)
            for i, name in enumerate(self.isothermal_emulator.ml_indexer.label_names)
        ])

        # Load vectorised HeFESTo EOS params if a control file is provided
        _eos_dir = Path(__file__).parent / "EOS_arithmetic"
        if control_path is None:
            control_path = _eos_dir / "BENCHMARK" / "control"
        if param_dir is None:
            param_dir = _eos_dir / "HeFESTo_Parameters_010123"

        # Store resolved paths so _clone_as_cpu can recreate this instance
        self._control_path = str(control_path)
        self._param_dir    = str(param_dir)
        self._npz_path     = str(npz_path) if npz_path is not None else None

        self.hefesto_params = None
        self.hefesto_npz_path = self._npz_path
        if control_path is not None:
            self.hefesto_params = load_control(
                str(control_path),
                param_dir_override=str(param_dir) if param_dir is not None else None,
            )
            if verbose:
                print(f"[INFO] Loaded HeFESTo params: {self.hefesto_params.nspec} species")

        if verbose:
            print("[INFO] HeFESToAPI initialized successfully.")
        # __init_subclass__ wrapper creates the CPU twin automatically
        # after this __init__ returns — no manual call needed here.

    def _coverage_standards(self, kind: str):
        """The same standard adiabats (BASALT/DMM/HTZ) overlay every kind of
        emulator ('isothermal'/'isentropic') -- HeFESTo has no Cr/NoCr split
        to distinguish them by. Returns None if no standards dir was
        auto-discovered for this ``model_dir`` (``self._test_meltstable_dir``)."""
        if self._test_meltstable_dir is None:
            return None
        from ngibbs.deployment_tests.training_coverage import hefesto_standard_points
        return hefesto_standard_points(self._test_meltstable_dir)

    def _clone_as_cpu(self):
        """Return a CPU HeFESToAPI with the same model_dir and EOS params."""
        return HeFESToAPI(
            self._model_dir,
            device='cpu',
            verbose=False,
            control_path=self._control_path,
            param_dir=self._param_dir,
            npz_path=self._npz_path,
            mass_balance=self._mass_balance,
        )

    def test(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        *,
        max_samples: Optional[int] = None,
        seed: int = 1337,
        write_outputs: bool = True,
        verbose: bool = True,
        allow_missing_models: bool = False,
    ) -> Dict[str, object]:
        """HeFESTo deployable test: the base emulator quality metrics (one table
        per configured NPT / NPS bundle) plus a MELTStable comparison of bulk EOS
        properties (rho, VP, VS, S, Cp, KS, thermal expansivity) and phase
        abundances against the internal vectorised HeFESTo EOS.

        ``allow_missing_models`` is forwarded to the base ``EmulatorAPI.test()``
        unchanged -- see its docstring. It does not affect the MELTStable EOS
        comparison below, which is skipped (not failed) whenever no MELTStable
        directory is configured, independent of this flag.

        All outputs (a quality-metrics CSV per emulator, two property + two phase
        figures, a property-error table and a phase-error table) are written to
        ``self.home_dir / 'deployment_test'``.
        """
        from ngibbs.deployment_tests import (
            run_meltstable_phase_comparison,
            run_meltstable_property_comparison,
        )

        out_dir = Path(output_dir) if output_dir is not None else self.home_dir / 'deployment_test'
        result = super().test(
            output_dir=out_dir, max_samples=max_samples, seed=seed,
            write_outputs=write_outputs, verbose=verbose,
            allow_missing_models=allow_missing_models,
        )

        if self._test_meltstable_dir is None:
            if verbose:
                print("[test]   no MELTStable directory configured; skipping EOS comparison.")
            return result

        if self.hefesto_params is None:
            raise RuntimeError(
                "HeFESTo EOS params not loaded; the MELTStable comparison needs a control file."
            )
        if verbose:
            print(f"[test]   MELTStable EOS comparison: {self._test_meltstable_dir}")

        prop = run_meltstable_property_comparison(self, self._test_meltstable_dir, out_dir)
        phase = run_meltstable_phase_comparison(self, self._test_meltstable_dir, out_dir)
        result['meltstable_property_errors'] = prop['property_errors']
        result['meltstable_phase_errors'] = phase['phase_errors']
        result['meltstable_unrepresented_phases'] = phase['unrepresented_phases']
        result['figures'] = {**prop['figures'], **phase['figures']}
        if write_outputs and verbose:
            print(f"[test]   wrote MELTStable tables + figures to {out_dir}")
        return result

    # EOS chunks: large enough to amortise Python overhead, small enough to
    # avoid OOM on systems where GPU VRAM is the bottleneck.
    _EOS_BATCH_SIZE: int = 2 ** 15

    # ── Metamorphic (phase-change / latent-heat) properties ──────────────────
    # Requesting any of these from get_property_hefesto_vectorized_from_assemblage
    # triggers the extra dn/dT, dn/dP solve.  Callers that only want the
    # isomorphic quantities (rho, Vp, Vs, S, Kh, ...) pay nothing.
    #
    # The velocity keys are here because HeFESTo's reported VP/VB are NOT
    # isomorphic: they carry the intra-phase order-disorder relaxation (see
    # aggregate.apply_fast_metamorphic).  'Vp'/'Vb' therefore stay isomorphic
    # for backwards compatibility, and the corrected values are exposed under
    # explicit names.
    _METAMORPHIC_KEYS = frozenset({
        'alptot', 'cptot', 'cvtot', 'gamtot', 'KTtot', 'KStot',
        'alpiso', 'cpiso', 'cviso', 'gamiso', 'KTiso', 'KSiso',
        'alpmet', 'cpmet', 'bmet',
        'dndt', 'dndp', 'sspeca', 'vspeca',
        'deltaent', 'deltavol', 'ClapeyronSlope',
    })
    # Subset that additionally needs the per-phase order-disorder ("fast") pass.
    # Active-set smallness threshold for the metamorphic solve, as a fraction of
    # total moles.  This is load-bearing when the composition comes from the
    # emulator: an NN never emits exact zeros, and a trace species that is the
    # sole occupant of an otherwise-absent phase is completely unregularised in
    # the dn/dT solve, producing Cp spikes that are invisible in rho/Vp/Vs.
    # See hefesto_vec.metamorphic.prune_active_set.
    #
    # It must sit ABOVE the emulator's absolute noise floor on component moles.
    # Raise it if you see Cp/KS spikes on emulated assemblages that are absent
    # from the fort.99 ground truth.
    metamorphic_nsmall_rel: float = NSMALL_REL

    # Single-component-phase sweep (see hefesto_vec.metamorphic.prune_active_set):
    # catches a phase's lone surviving member -- e.g. a spinel-group phase
    # where every other endmember has gone to exactly zero but one leftover
    # component lingers at trace level for many GPa/steps. That survivor
    # carries no configurational regularisation (nothing else in its phase to
    # be diluted against), independent of how small metamorphic_nsmall_rel is
    # set to. Only engages when within_phase_frac is supplied to
    # get_property_hefesto_vectorized_from_assemblage.
    metamorphic_single_component_dominance: float = 0.98
    metamorphic_single_component_nsmall_rel: Optional[float] = None  # default: 5x metamorphic_nsmall_rel

    _FAST_KEYS = frozenset({
        'Vp_fast', 'Vb_fast', 'Kh_fast', 'Kv_fast', 'Kr_fast',
        'dndt_fast', 'dndp_fast',
        'bmet_fast_phases', 'alpmet_fast_phases', 'cpmet_fast_phases',
    })

    def _compute_bulk_EOS_properties(
        self,
        component_moles: torch.Tensor,
        PT: torch.Tensor,
        property_names = ('rho', 'Vp', 'Vs', 'S'),
    ) -> Dict[str, np.ndarray]:
        """Route EOS evaluation through _staged_forward.

        Gives automatic chunking for large inputs and CPU fallback for
        small inputs — both handled transparently by _staged_forward.
        """
        nc = int(component_moles.shape[1])

        def _as_f64(x):
            if torch.is_tensor(x):
                return x.detach().to(device=self.device, dtype=torch.float64)
            return torch.tensor(np.asarray(x, dtype=np.float64), device=self.device)

        combined = torch.cat([_as_f64(component_moles), _as_f64(PT)], dim=1)

        def _eos_gpu(batch):
            return self.get_property_hefesto_vectorized_from_assemblage(
                batch[:, :nc], batch[:, nc:], property_names
            )

        cpu_fn = None
        if self._cpu_api is not None:
            def _eos_cpu(batch):
                return self._cpu_api.get_property_hefesto_vectorized_from_assemblage(
                    batch[:, :nc], batch[:, nc:], property_names
                )
            cpu_fn = _eos_cpu

        return self._staged_forward(
            _eos_gpu, combined, self._EOS_BATCH_SIZE, _cpu_fallback=cpu_fn
        )

    def get_property_hefesto_vectorized_from_assemblage(
        self,
        component_moles: torch.Tensor,
        PT: torch.Tensor,
        property_names: Sequence[str] = ('rho', 'Vp', 'Vs', 'S'),
        within_phase_frac: Optional[torch.Tensor] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute bulk EOS properties using the vectorised HeFESTo EOS.

        Bulk scalar outputs (shape (B,)):
            rho   g/cm³       aggregate density
            Vp    km/s        P-wave velocity (VRH)
            Vs    km/s        S-wave velocity (VRH)
            Vb    km/s        bulk sound velocity
            S     J/g/K       aggregate entropy
            Kh    GPa         adiabatic bulk modulus (VRH Hill)
            Gh    GPa         shear modulus (VRH Hill)
            Kv/Kr GPa         Voigt / Reuss adiabatic bulk moduli
            Gv/Gr GPa         Voigt / Reuss shear moduli

        Metamorphic (phase-change) outputs, shape (B,) — computed only if
        requested, since they cost an extra dn/dT, dn/dP linear solve:
            cptot  J/g/K      isobaric heat capacity  (fort.56 'cp')
            KStot  GPa        adiabatic bulk modulus  (fort.56 'KS')
            KTtot  GPa        isothermal bulk modulus (fort.56 'btot')
            alptot 1/K        thermal expansivity     (fort.56 'alpha'/1e5)
            cvtot, gamtot     isochoric Cp, Gruneisen
            *iso              the same quantities without the phase-change term
                              (fort.59), for isolating the metamorphic part
            alpmet, cpmet, bmet, deltaent, deltavol, ClapeyronSlope
            dndt, dndp  (B, nspec)   the underlying composition derivatives

        Order-disorder ('fast') outputs, shape (B,) — these cost a second solve:
            Vp_fast, Vb_fast  km/s   velocities WITH the intra-phase relaxation
            Kh_fast, Kv_fast, Kr_fast  GPa

        Note that 'Vp'/'Vb' remain the *isomorphic* velocities.  HeFESTo's
        fort.56 VP/VB include the order-disorder relaxation, so 'Vp_fast' is the
        one to compare against fort.56 — plain 'Vp' is biased high by up to
        0.26% in the lower mantle.  Vs is identical either way.

        Per-species outputs (shape (B, nspec)) — useful for diagnostics:
            V     cm³/mol     converged molar volume
            _K    GPa         isothermal bulk modulus
            _Ks   GPa         adiabatic bulk modulus
            _Gsh  GPa         shear modulus
            _alp  K⁻¹         thermal expansivity
            _Cp   J/mol/K     isobaric heat capacity
            _rho  g/cm³       species density
            _S    J/mol/K     species molar entropy

        Parameters
        ----------
        component_moles : torch.Tensor, shape (B, C)
            Component mole fractions from emulator output. Columns correspond
            to self.isothermal_emulator.ml_indexer.label_names; self.PropertyIDX
            maps them into the full HeFESTo species array.
        PT : torch.Tensor, shape (B, 2)
            Columns: [P (GPa), T (K)].
        property_names : sequence of str
            Keys to return from the compute output dict.
        within_phase_frac : torch.Tensor, shape (B, C), optional
            Each component's mole fraction *within its own phase* -- same
            column space as ``component_moles``. Enables prune_active_set's
            single-component sweep (see
            hefesto_vec.metamorphic.compute_within_phase_frac / .prune_active_set),
            which catches a phase's lone surviving member -- e.g. every other
            spinel-group endmember at exactly zero but one leftover component
            lingering at trace level -- that ordinary total-system-fraction
            pruning misses because that survivor's phase share can sit above
            metamorphic_nsmall_rel even though it carries no configurational
            regularisation. For the emulator, pass 'chem_out' (already computed
            per forward pass, so this is free) expanded back to component
            space via ml_indexer.variedToAllComp. When omitted, it is derived
            for you from ``component_moles`` itself via a direct
            moles-over-phase-total division -- correct, but redundant with (and
            noisier than) a genuine intensive-label output, so prefer passing
            the real one when you have it. Only used when a metamorphic
            property is requested.

        Returns
        -------
        dict[str, np.ndarray]
            Requested properties keyed by name.
        """
        if self.hefesto_params is None:
            raise RuntimeError(
                "HeFESTo params not loaded. Pass control_path= (and optionally "
                "param_dir=) to HeFESToAPI.__init__."
            )

        if torch.is_tensor(component_moles):
            cm_np = component_moles.detach().cpu().numpy().astype(np.float64)
        else:
            cm_np = np.asarray(component_moles, dtype=np.float64)

        if torch.is_tensor(PT):
            PT_np = PT.detach().cpu().numpy().astype(np.float64)
        else:
            PT_np = np.asarray(PT, dtype=np.float64)

        P = PT_np[:, 0]  # (B,) GPa
        T = PT_np[:, 1]  # (B,) K

        B = cm_np.shape[0]
        X_input = np.zeros((B, self.hefesto_params.nspec), dtype=np.float64)
        X_input[:, self.PropertyIDX] = cm_np

        if within_phase_frac is not None:
            if torch.is_tensor(within_phase_frac):
                wpf_np = within_phase_frac.detach().cpu().numpy().astype(np.float64)
            else:
                wpf_np = np.asarray(within_phase_frac, dtype=np.float64)
            within_phase_frac_full = np.zeros((B, self.hefesto_params.nspec), dtype=np.float64)
            within_phase_frac_full[:, self.PropertyIDX] = wpf_np
        else:
            # Computed lazily below, only if a metamorphic property is actually
            # requested (this is cheap either way, but no need to pay it for
            # rho/Vp/Vs/S-only calls).
            within_phase_frac_full = None

        eos_device = self.device.type if hasattr(self.device, 'type') else str(self.device)
        t0 = time.time()
        result = hefesto_compute(
            P, T, X_input, self.hefesto_params,
            npz_path=self.hefesto_npz_path,
            device=eos_device if eos_device != 'cpu' else None,
        )
        if self.verbose:
            print(f"HeFESTo EOS for {B} assemblages took {time.time() - t0:.2f} s")

        # Metamorphic pass, only if something asked for it.
        wanted = set(property_names)
        need_fast = bool(wanted & self._FAST_KEYS)
        if need_fast or (wanted & self._METAMORPHIC_KEYS):
            if within_phase_frac_full is None:
                within_phase_frac_full = compute_within_phase_frac(
                    X_input, self.hefesto_params.phase_members
                )
            t1 = time.time()
            met = hefesto_add_metamorphic(
                result, X_input, T, P, self._get_metamorphic_tables(),
                include_fast=need_fast,
                nsmall_rel=self.metamorphic_nsmall_rel,
                within_phase_frac=within_phase_frac_full,
                single_component_dominance=self.metamorphic_single_component_dominance,
                single_component_nsmall_rel=self.metamorphic_single_component_nsmall_rel,
            )
            if need_fast:
                # add_metamorphic overwrites Vp/Vb/Kh/Kv/Kr in-place with the
                # softened values.  Re-key them so 'Vp' keeps meaning the
                # isomorphic velocity for every existing caller.
                for key in ('Vp', 'Vb', 'Kh', 'Kv', 'Kr'):
                    if key in met:
                        met[f'{key}_fast'] = met.pop(key)
                met.pop('Vs', None)          # unchanged by construction
                for key in ('Vp_iso', 'Vb_iso', 'Kh_iso'):
                    met.pop(key, None)       # identical to result[key]
            result.update(met)
            if self.verbose:
                print(f"  metamorphic terms took {time.time() - t1:.2f} s"
                      f"{' (incl. order-disorder pass)' if need_fast else ''}")

        try:
            return {prop: result[prop] for prop in property_names}
        except KeyError as e:
            raise ValueError(
                f"Unknown property {e}. Available keys: {sorted(result.keys())}"
            ) from e

    def _get_metamorphic_tables(self):
        """Static tables for the metamorphic solve, built once and cached.

        Depends only on the parameter set and species list, not on P/T/X, so a
        single build serves every batch (and every chunk of a staged batch).

        The parameter directory is taken from ``hefesto_params.param_dir``, not
        from ``self._param_dir``: scripts routinely swap ``hefesto_params`` out
        after construction to override a stale control-file path (see
        scripts/property_comparison.py), and the tables must be read from the
        same parameter set the EOS is using.  The cache is keyed on the params
        object so that swap invalidates it.
        """
        params = self.hefesto_params
        cached = getattr(self, '_metamorphic_tables', None)
        if cached is not None and cached[0] is params:
            return cached[1]
        param_dir = getattr(params, 'param_dir', '') or self._param_dir
        tables = hefesto_build_tables(params, param_dir)
        self._metamorphic_tables = (params, tables)
        return tables


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
        model_dir: Union[str, Path],
        *,
        device: str = 'cpu',
        verbose: bool = False,
        mass_balance: str = 'iterative',
        load_melts_eos: bool = True,
        melts_solid_params_path: Optional[Union[str, Path]] = None,
        melts_liquid_params_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize MELTS API by pointing at a single directory, building the
        NoCr and Cr sub-APIs from the Cr/NoCr-tagged checkpoints inside it.

        Each sub-API holds up to three emulators: isothermal (closed, T input),
        isentropic (closed, S input), and open oxygen (fO2-buffered, T + logfO2
        input). Routing between Cr and NoCr is done by inspecting composition
        headers; routing between the three emulator types is done by inspecting
        thermodynamic condition headers.

        Parameters
        ----------
        model_dir : str or Path
            Directory holding both the Cr- and NoCr-tagged checkpoints for
            this MELTS model family (e.g. ``engine/TrainedModels/MELTS120/``,
            which holds ``120SedIgClosed_{Cr,NoCr}_{NPT,NPS}_train2.tar`` and
            ``120SedIgOpen_{Cr,NoCr}_train2.tar`` side by side). Each
            checkpoint's Cr/NoCr sub-API and isothermal/isentropic/openox
            role are auto-discovered from its filename — see
            ``EmulatorAPI.__init__``'s docstring for the exact convention.
            The matching ``*_Test_subset*.tar.gz`` quality bundles (up to six,
            consumed by ``self.test()``) are located the same way, under the
            sibling ``deployment_tests/`` directory.
        device : str, default='cpu'
            Torch device ('cpu' or 'cuda')
        verbose : bool, default=False
            Print initialization messages
        load_melts_eos : bool, default=True
            Load the vectorised MELTS EOS parameter tables (melts_vec) needed
            by get_property_melts_vectorized_from_assemblage. These ship as
            JSON inside EOS_arithmetic_MELTS/MELTS_Parameters/ and load fast,
            so this defaults on (unlike HeFESToAPI's control_path, which is
            optional because it points at a large external control file).
        melts_solid_params_path, melts_liquid_params_path : str or Path, optional
            Override the packaged sol_struct_data.json / liq_struct_data.json
            (see EOS_arithmetic_MELTS/melts_vec/params.py, liquid_params.py).
        """
        self._model_dir = str(Path(model_dir).resolve())
        if verbose:
            print("[INFO] Initializing MELTSAPI (NoCr)...")
        self.nocr = EmulatorAPI(
            model_dir,
            device=device,
            verbose=verbose,
            mass_balance=mass_balance,
            variant='nocr',
        )
        if verbose:
            print("[INFO] Initializing MELTSAPI (Cr)...")
        self.cr = EmulatorAPI(
            model_dir,
            device=device,
            verbose=verbose,
            mass_balance=mass_balance,
            variant='cr',
        )
        self.device = torch.device(device)
        self.verbose = verbose
        self.nocr.modelType = "MELTSAPI"
        self.cr.modelType = "MELTSAPI"

        # Wire the MELTSIsobaricStandards overlay into each sub-API's own
        # test()-time training-coverage plots (see EmulatorAPI._coverage_standards)
        # -- auto-discovered the same way MELTSAPI.test() finds its own
        # standards_dir default, just done once here up front.
        _std_dir = _discover_standards_dir(_deployment_tests_root(), Path(model_dir))
        if _std_dir is not None:
            from ngibbs.deployment_tests.training_coverage import melts_standard_points
            self.nocr._coverage_standards_fn = (
                lambda kind, d=_std_dir: melts_standard_points(d, variant='NoCr')
            )
            self.cr._coverage_standards_fn = (
                lambda kind, d=_std_dir: melts_standard_points(d, variant='Cr')
            )

        # Vectorised MELTS EOS (melts_vec): solid-endmember + liquid + the
        # feldspar/olivine solid-solution mixing models. See
        # get_property_melts_vectorized_from_assemblage's docstring for what
        # is and isn't covered.
        self._melts_solid_params_path = str(melts_solid_params_path) if melts_solid_params_path is not None else None
        self._melts_liquid_params_path = str(melts_liquid_params_path) if melts_liquid_params_path is not None else None
        self.melts_solid_params = None
        self.melts_liquid_params = None
        if load_melts_eos:
            self.melts_solid_params = load_melts_solids(self._melts_solid_params_path)
            self.melts_liquid_params = load_melts_liquid(self._melts_liquid_params_path)
            if verbose:
                print(f"[INFO] Loaded melts_vec EOS params: "
                      f"{self.melts_solid_params.nspec} solid endmembers, "
                      f"{self.melts_liquid_params.nspec} liquid components")

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

    def test(self, output_dir=None, *, standards_dir=None, allow_missing_models: bool = False,
             **kwargs) -> dict:
        """MELTS deployable test: the base per-sub-API (NoCr/Cr) emulator
        quality metrics, exactly as before, PLUS the MELTSIsobaricStandards
        property/phase comparison (``run_melts_property_comparison`` /
        ``run_melts_phase_comparison`` in ``ngibbs.deployment_tests.
        melts_comparison``) against the internal vectorised MELTS EOS -- the
        MELTS analogue of ``HeFESToAPI.test()``'s MELTStable comparison.

        Each sub-API scores every configured emulator (isothermal / isentropic /
        openox) whose bundle file is present — up to six tables in total. The
        NoCr and Cr outputs are split into ``deployment_test/nocr`` and
        ``deployment_test/cr`` (the two share a model directory).

        ``allow_missing_models`` is forwarded to each sub-API's own
        ``EmulatorAPI.test()`` unchanged (see its docstring) -- with the
        default ``False``, a configured-but-absent emulator bundle (for
        either sub-API) fails the test. MELTS120 has no temperature model at
        all and no Cr openox checkpoint; as long as those aren't wired into
        ``test_*_bundles`` in the first place they're never seen as "missing"
        here, so ``allow_missing_models`` only matters for bundles that ARE
        configured but whose checkpoint/bundle file happens to be absent.
        This flag does not gate the MELTSIsobaricStandards comparison below
        (that comparison reports actual emulator-vs-GT error, not model
        presence/absence, and is skipped rather than failed when
        ``standards_dir`` doesn't exist or ``load_melts_eos=False``).

        ``standards_dir`` : optional override for the auto-discovered
        standards directory (``_discover_standards_dir``, matched to this
        MELTSAPI's own ``model_dir`` by name, e.g. 'TrainedModels/120' ->
        'deployment_tests/MELTSIsobaricStandards/120'), which itself falls
        back to ``melts_comparison.DEFAULT_STANDARDS_DIR`` if nothing is
        found.

        Returns ``{'nocr': <result dict>, 'cr': <result dict>,
        'melts_property_errors': DataFrame, 'melts_phase_errors': DataFrame,
        'figures': {...}}`` (the last three keys omitted if the
        MELTSIsobaricStandards comparison was skipped).
        """
        out = {}
        for tag, sub in (('nocr', self.nocr), ('cr', self.cr)):
            sub_dir = (Path(output_dir) / tag if output_dir is not None
                       else sub.home_dir / 'deployment_test' / tag)
            out[tag] = sub.test(output_dir=sub_dir, allow_missing_models=allow_missing_models, **kwargs)

        if self.melts_solid_params is None or self.melts_liquid_params is None:
            # load_melts_eos=False at construction: the internal-EOS
            # comparison needs melts_vec's own parameter tables.
            return out

        from ngibbs.deployment_tests.melts_comparison import (
            DEFAULT_STANDARDS_DIR, run_melts_phase_comparison, run_melts_property_comparison,
        )
        if standards_dir is not None:
            std_dir = Path(standards_dir)
        else:
            std_dir = _discover_standards_dir(_deployment_tests_root(), Path(self._model_dir))
            if std_dir is None:
                std_dir = DEFAULT_STANDARDS_DIR
        if not std_dir.exists():
            return out

        comp_out_dir = Path(output_dir) if output_dir is not None else self.nocr.home_dir / 'deployment_test'
        prop = run_melts_property_comparison(self, std_dir, comp_out_dir)
        phase = run_melts_phase_comparison(self, std_dir, comp_out_dir)
        out['melts_property_errors'] = prop['property_errors']
        out['melts_phase_errors'] = phase['phase_errors']
        # prop['figures'] and phase['figures'] both use the keys
        # 'isothermal'/'isentropic' -- prefix so neither clobbers the other.
        out['figures'] = {
            **{f'property_{k}': v for k, v in prop['figures'].items()},
            **{f'phase_{k}': v for k, v in phase['figures'].items()},
        }
        return out

    def assert_quality(self, output_dir=None, **kwargs) -> dict:
        """Run ``assert_quality()`` on both sub-APIs (NoCr and Cr).

        Raises ``EmulatorQualityError`` (from whichever sub-API fails first) if
        either misses a threshold. Returns ``{'nocr': <result dict>, 'cr': <result dict>}``.
        """
        out = {}
        for tag, sub in (('nocr', self.nocr), ('cr', self.cr)):
            sub_dir = (Path(output_dir) / tag if output_dir is not None
                       else sub.home_dir / 'deployment_test' / tag)
            out[tag] = sub.assert_quality(output_dir=sub_dir, **kwargs)
        return out

    # ── Vectorised MELTS EOS (melts_vec) ──────────────────────────────────────
    # Phase-name -> (endmember list, solution-model function) for the phases
    # melts_vec currently implements a real (non-ideal) mixing model for.
    # Both the canonical melts_vec module name and a couple of common nMELTS
    # ml_indexer phase-name spellings are accepted so callers can pass
    # whichever convention their composition dict already uses.
    _MELTS_SOLUTION_PHASES = {
        'feldspar':    (_melts_feldspar.ENDMEMBERS, melts_compute_feldspar_solution),
        'plagioclase': (_melts_feldspar.ENDMEMBERS, melts_compute_feldspar_solution),
        'olivine':     (_melts_olivine.ENDMEMBERS,  melts_compute_olivine_solution),
        'clinopyroxene': (_melts_clinopyroxene.ENDMEMBERS, melts_compute_clinopyroxene_solution),
        'cpx':           (_melts_clinopyroxene.ENDMEMBERS, melts_compute_clinopyroxene_solution),
        'orthopyroxene': (_melts_orthopyroxene.ENDMEMBERS, melts_compute_orthopyroxene_solution),
        'opx':           (_melts_orthopyroxene.ENDMEMBERS, melts_compute_orthopyroxene_solution),
        'spinel':        (_melts_spinel.ENDMEMBERS, melts_compute_spinel_solution),
        'sp':            (_melts_spinel.ENDMEMBERS, melts_compute_spinel_solution),
        'rhm-oxide':          (_melts_rhomsghiorso.ENDMEMBERS, melts_compute_rhm_oxide_solution),
        'rhombohedral-oxide': (_melts_rhomsghiorso.ENDMEMBERS, melts_compute_rhm_oxide_solution),
        'rhm_oxide':          (_melts_rhomsghiorso.ENDMEMBERS, melts_compute_rhm_oxide_solution),
    }
    # Phases MELTS models as solid solutions that melts_vec does NOT yet
    # implement a mixing model for. Recognised purely so
    # get_property_melts_vectorized_from_assemblage can name them in its
    # coverage report / strict-mode error rather than just silently ignoring
    # an unrecognised key.
    _MELTS_UNSUPPORTED_SOLUTION_PHASES = frozenset({
        # 'pyroxene' is deliberately left here too: unlike 'clinopyroxene'/'cpx'
        # and 'orthopyroxene'/'opx' (both unambiguous), a bare 'pyroxene' key
        # could mean either structural state, so it stays unsupported rather
        # than silently guessing which one a caller meant.
        'pyroxene',
    })
    _MELTS_LIQUID_PHASE_NAMES = frozenset({'liquid', 'melts-liquid', 'melt'})

    def get_property_melts_vectorized_from_assemblage(
        self,
        phase_composition: Dict[str, Union[torch.Tensor, np.ndarray]],
        phase_moles: Dict[str, Union[torch.Tensor, np.ndarray]],
        PT: torch.Tensor,
        property_names: Sequence[str] = ('V', 'Cp', 'K', 'alpha', 'G', 'H', 'S'),
        strict: bool = False,
        liquid_oxides: Optional[Union[torch.Tensor, np.ndarray]] = None,
        liquid_oxide_labels: Optional[Sequence[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute bulk thermodynamic properties of a MELTS assemblage using the
        vectorised melts_vec EOS -- the MELTS analogue of
        HeFESToAPI.get_property_hefesto_vectorized_from_assemblage, but
        operating directly in melts_vec's own per-phase composition space
        (see "Scope and known gaps" below for why this isn't yet wired
        straight to the trained MELTS emulator's raw output).

        Parameters
        ----------
        phase_composition : dict[str, array-like]
            {phase_name: (B, n_endmembers) mole fractions of that phase's own
            endmembers, each row should sum to ~1}. Recognised phase names:
              - 'melts-liquid' / 'liquid' / 'melt': (B, 19) mole fractions in
                melts_vec.liquid_params.load_liquid().labels order (SiO2,
                TiO2, Al2O3, Fe2O3, MgCr2O4, Fe2SiO4, MnSi0.5O2, Mg2SiO4,
                NiSi0.5O2, CoSi0.5O2, CaSiO3, Na2SiO3, KAlSiO4, Ca3(PO4)2,
                CO2, SO3, Cl2O-1, F2O-1, H2O).
              - 'feldspar' / 'plagioclase': (B, 3), melts_vec.feldspar.
                ENDMEMBERS order (albite, anorthite, sanidine).
              - 'olivine': (B, 6), melts_vec.olivine.ENDMEMBERS order
                (tephroite, fayalite, co-olivine, ni-olivine, monticellite,
                forsterite).
              - 'clinopyroxene' / 'cpx': (B, 7), melts_vec.clinopyroxene.
                ENDMEMBERS order (diopside, clinoenstatite, hedenbergite,
                alumino-buffonite, buffonite, essenite, jadeite).
              - 'orthopyroxene' / 'opx': (B, 7), melts_vec.orthopyroxene.
                ENDMEMBERS order (diopside, clinoenstatite, hedenbergite,
                alumino-buffonite, buffonite, essenite, jadeite) -- the same
                names/order as clinopyroxene (both reference the same
                sol_struct_data.json endmember block).
              - 'spinel' / 'sp': (B, 5), melts_vec.spinel.ENDMEMBERS order
                (chromite, hercynite, magnetite, spinel, ulvospinel).
              - 'rhm-oxide' / 'rhombohedral-oxide' / 'rhm_oxide': (B, 5),
                melts_vec.rhomsghiorso.ENDMEMBERS order (geikielite, hematite,
                ilmenite, pyrophanite, corundum). For a pMELTS-calibrated
                composition (which omits corundum entirely) pass 0 in the
                last column.
            Any other key is treated as a single (possibly pure) phase
            evaluated via the pure-endmember EOS only (see "ideal-only
            phases" below) -- pass its melts_vec solid-endmember label(s)
            as the dict values' implicit column order isn't defined for
            those, so for anything beyond feldspar/olivine/liquid this is
            only meaningful for a SINGLE pure phase (n_endmembers = 1,
            (B, 1) all-ones), e.g. quartz.
        phase_moles : dict[str, array-like]
            {phase_name: (B,) molar amount of that phase in the assemblage},
            same keys as phase_composition. Used as the bulk-aggregation
            weights (mirrors HeFESTo's component_moles carrying both
            identity and abundance in one array; kept separate here because
            melts_vec's per-phase functions want composition normalized to
            sum to 1, not raw extensive moles).
        PT : torch.Tensor, shape (B, 2)
            Columns: [P (GPa), T (K)] -- same convention as
            get_property_hefesto_vectorized_from_assemblage; converted
            internally to melts_vec's bars.
        property_names : sequence of str
            Bulk (B,) keys to return: 'V' (J/bar), 'dVdT', 'dVdP', 'K' (bar),
            'alpha' (1/K), 'Cp' (J/mol/K), 'dCpdT', 'G', 'H' (J/mol), 'S'
            (J/mol/K). (No 'rho' -- melts_vec's extracted parameter tables
            don't carry a verified per-endmember molar mass yet, see Scope
            below.) Also accepts 'melts_coverage_fraction' (the mole
            fraction of the total assemblage actually covered by a
            supported phase -- see below) and 'per_phase' (returns the raw
            per-phase dicts, keyed by the phase names in phase_composition,
            instead of/alongside the bulk aggregate).
        strict : bool, default=False
            If True, raise ValueError when phase_composition contains a key
            in _MELTS_UNSUPPORTED_SOLUTION_PHASES with nonzero phase_moles
            (rather than silently excluding it from the bulk aggregate --
            see below).

        liquid_oxides : array-like, shape (B, O), optional
            Molar oxide composition of the liquid, with iron ALREADY
            speciated into FeO/Fe2O3 -- e.g. straight from
            NN_MELTS.get_liquid_oxides(). Column order defaults to
            config.constants.default_Oxides (SiO2, TiO2, Al2O3, FeO, MgO,
            CaO, Na2O, K2O, P2O5, H2O, Cr2O3, MnO, NiO, CO2, Fe2O3); pass
            liquid_oxide_labels for a different order/set of columns
            (unrecognised oxides -- e.g. CoO, SO3, Cl, F, which no current
            nGibbs checkpoint tracks -- default to 0, see melts_vec.
            oxides_to_liquid_components's own docstring for the full
            recognised-key list).

            When given, this REPLACES whatever the 'melts-liquid'/'liquid'/
            'melt' key in phase_composition carries: melts_vec's own 19-
            component basis is built from these oxides via melts_vec.
            oxides_to_liquid_components, following MELTS's own fixed
            component-construction order (chromite before olivine's
            forsterite component claims the remaining MgO, apatite before
            wollastonite claims the remaining CaO, leucite/kalsilite before
            whatever Al2O3 is left over becomes the Al2O3 component itself,
            SiO2 resolved last as everyone else's leftover -- see that
            function's own docstring for the full derivation). This is the
            intended way to feed the trained emulator's raw liquid output
            into this method -- see "Scope and known gaps" below.

            phase_composition must still include a 'melts-liquid' (or
            alias) key when liquid_oxides is given, so its phase_moles
            entry is still counted in the bulk aggregate -- that key's own
            composition VALUE is ignored and can be any correctly-shaped
            placeholder (e.g. zeros); only its phase_moles weight matters.
        liquid_oxide_labels : sequence of str, optional
            Column labels for liquid_oxides, if not config.constants.
            default_Oxides's own order (length must match liquid_oxides's
            last dimension).

        Bulk aggregation
        ----------------
        Bulk V/dVdT/dVdP/H/S/Cp/dCpdT are the phase-mole-weighted sum over
        every phase in phase_composition that melts_vec can actually
        compute (liquid, feldspar, olivine, or a single pure endmember);
        G = H - T*S is then recomputed for consistency, and K/alpha are
        derived from V/dVdT/dVdP exactly as compute()/solid_solutions.py do
        for a single phase. This is NOT necessarily the properties of the
        WHOLE assemblage: any phase in _MELTS_UNSUPPORTED_SOLUTION_PHASES
        (pyroxene) is excluded from the sum entirely
        (its moles still count toward the coverage denominator). Always
        check 'melts_coverage_fraction' -- the fraction of total assemblage
        moles actually covered -- before trusting the bulk numbers for a
        pyroxene-bearing assemblage; request it explicitly via
        property_names or read it off the returned dict's
        'melts_coverage_fraction' key, which is always included.

        Scope and known gaps
        ---------------------
        This method's composition input is melts_vec's OWN endmember space,
        not the trained MELTS neural-network emulator's raw output space.
        Bridging the two is a real, separate translation task that was
        looked into but deliberately NOT done this pass:
          - For 'melts-liquid': DONE, via the liquid_oxides parameter above,
            rather than by reworking this method's own phase_composition
            slot. nMELTS's ml_indexer represents the liquid's ML-output
            composition as ELEMENTAL mole fractions (Si, Ti, Al, Fe, Mg,
            Ca, Na, K, P, H, Cr, Mn, Ni -- see config/README_MLIndexer.md:
            "'melts-liquid' components come from Elkeys, not
            components_in_phases"), not the 19-component meltsLiquid
            oxide-component table melts_vec.liquid_eos expects, and
            getting from one to the other is genuinely two separate
            problems, both now addressed:
              (a) the fO2-dependent Fe2+/Fe3+ equilibrium partition --
                  MAGMA's own conLiq_v34() (sources/liquid_v34.c) turned
                  out to already have an independent Python translation in
                  this codebase (emulator.Fe2O3_FeO_ratio /
                  QFM_fO2_torch, matching conLiq_v34's Kress & Carmichael
                  1991 coefficients exactly), just not wired up to
                  melts_vec -- NN_MELTS.get_liquid_oxides() now exposes it
                  for this purpose (via the existing Iron_Speciator, for
                  open/fO2-buffered models only; closed models carry Fe3+
                  as their own tracked component already and need no
                  speciation step at all -- see get_liquid_oxides's own
                  docstring);
              (b) going from 14 elemental oxides to 19 melts_vec
                  components is NOT a fixed linear map either (several
                  components share SiO2/Al2O3 as a "pool" oxide) -- this
                  is genuine stoichiometric bookkeeping, not equilibrium
                  chemistry, and is unrelated to conLiq; melts_vec.
                  oxides_to_liquid_components() implements MELTS's own
                  fixed component-construction order for it (see that
                  function's module docstring for the full derivation),
                  verified by a 20,000-row round-trip mass-balance closure
                  test against melts_vec's own (linear) component->oxide
                  map.
            Passing raw ml_indexer liquid output straight into this
            method's 'melts-liquid' phase_composition slot still WILL give
            wrong numbers -- use the liquid_oxides parameter instead, fed
            from NN_MELTS.get_liquid_oxides()'s output.
          - For 'feldspar'/'olivine', nMELTS's components_in_phases already
            appears to use plain MELTS endmember names directly as its
            per-phase component labels (per the README's own
            ``{'olivine': ['fayalite', 'forsterite'], ...}`` example, which
            matches melts_vec.olivine.ENDMEMBERS's naming) -- so mapping a
            real ml_indexer's label_indices_comp['olivine'] /
            ['feldspar' or 'plagioclase'] columns into this method's
            expected column order is likely a straight reindex by label
            name, not a unit-system conversion like the liquid case. This
            has NOT been cross-checked against an actual trained
            checkpoint's ml_indexer this session, though, so verify the
            label order (e.g. via ``ml_indexer.detail_label_indices``)
            before wiring it up for real use.
          - clinopyroxene, orthopyroxene, spinel, and rhombohedral-oxide are
            all supported now (see melts_vec's __init__.py docstring for each
            phase's own architecture notes). Only a bare 'pyroxene' key stays
            unsupported (structural state is ambiguous). Any assemblage
            containing an unsupported phase is necessarily partially
            covered here, see 'melts_coverage_fraction'.
        """
        if self.melts_solid_params is None or self.melts_liquid_params is None:
            raise RuntimeError(
                "melts_vec EOS params not loaded. Pass load_melts_eos=True "
                "(the default) to MELTSAPI.__init__, or supply "
                "melts_solid_params_path=/melts_liquid_params_path=."
            )

        def _np(x):
            if torch.is_tensor(x):
                return x.detach().cpu().numpy().astype(np.float64)
            return np.asarray(x, dtype=np.float64)

        PT_np = _np(PT)
        P_bar = PT_np[:, 0] * _MELTS_BARS_PER_GPA
        T = PT_np[:, 1]
        B = PT_np.shape[0]

        # liquid_oxides, if given, takes over the 'melts-liquid' slot entirely:
        # build melts_vec's 19-component mole-fraction array from it up front,
        # via MELTS's own (non-linear, order-dependent) oxide->component
        # construction -- see oxides_to_liquid_components's own docstring and
        # this method's "Scope and known gaps" section above.
        liquid_X_override = None
        if liquid_oxides is not None:
            labels = list(liquid_oxide_labels) if liquid_oxide_labels is not None else list(_DEFAULT_LIQUID_OXIDE_LABELS)
            liq_ox_np = _np(liquid_oxides)
            if liq_ox_np.shape[-1] != len(labels):
                raise ValueError(
                    f"liquid_oxides has {liq_ox_np.shape[-1]} columns but "
                    f"{len(labels)} liquid_oxide_labels were given/defaulted "
                    f"({labels})."
                )
            oxide_dict = {lbl: liq_ox_np[:, i] for i, lbl in enumerate(labels)}
            liquid_component_moles = _melts_oxides_to_liquid_components(oxide_dict, self.melts_liquid_params)
            row_sum = liquid_component_moles.sum(axis=-1, keepdims=True)
            liquid_X_override = np.divide(
                liquid_component_moles, row_sum,
                out=np.zeros_like(liquid_component_moles), where=(row_sum != 0.0),
            )

        per_phase: Dict[str, Dict[str, np.ndarray]] = {}
        total_moles = np.zeros(B, dtype=np.float64)
        covered_moles = np.zeros(B, dtype=np.float64)

        acc_keys = ('V', 'dVdT', 'dVdP', 'H', 'S', 'Cp', 'dCpdT')
        acc = {k: np.zeros(B, dtype=np.float64) for k in acc_keys}

        for phase_name, X in phase_composition.items():
            X_np = _np(X)
            moles = _np(phase_moles[phase_name]) if phase_name in phase_moles else np.zeros(B)
            total_moles = total_moles + moles

            phase_key = phase_name.strip().lower()
            if phase_key in self._MELTS_LIQUID_PHASE_NAMES and liquid_X_override is not None:
                X_np = liquid_X_override  # liquid_oxides overrides phase_composition's own liquid entry

            result = None
            if phase_key in self._MELTS_LIQUID_PHASE_NAMES:
                result = melts_compute_liquid_bulk(T, P_bar, X_np, self.melts_liquid_params)
            elif phase_key in self._MELTS_SOLUTION_PHASES:
                endmembers, fn = self._MELTS_SOLUTION_PHASES[phase_key]
                result = fn(T, P_bar, X_np, self.melts_solid_params)
            elif phase_key in self._MELTS_UNSUPPORTED_SOLUTION_PHASES:
                if strict and np.any(moles != 0.0):
                    raise ValueError(
                        f"phase {phase_name!r} is a MELTS solid solution melts_vec "
                        f"does not yet implement a mixing model for (see "
                        f"get_property_melts_vectorized_from_assemblage's docstring); "
                        f"pass strict=False to exclude it from the bulk aggregate instead."
                    )
                # excluded: contributes to total_moles (coverage denominator)
                # but not to covered_moles or the property sums.
                continue
            else:
                # Single pure phase fallback: X_np expected (B, 1), all ones.
                from .EOS_arithmetic_MELTS.melts_vec import compute as melts_compute
                result = melts_compute(T, P_bar, self.melts_solid_params, names=[phase_name])
                for k in acc_keys:
                    result[k] = result[k][:, 0]

            per_phase[phase_name] = result
            covered_moles = covered_moles + moles
            for k in acc_keys:
                if k in result:
                    acc[k] = acc[k] + moles * result[k]

            del X_np, moles, result

        with np.errstate(divide='ignore', invalid='ignore'):
            for k in acc_keys:
                acc[k] = np.where(covered_moles != 0.0, acc[k] / np.where(covered_moles != 0.0, covered_moles, 1.0), 0.0)

        G = acc['H'] - T * acc['S']
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            K = np.where(acc['dVdP'] != 0.0, -acc['V'] / acc['dVdP'], np.inf)
            alpha = np.where(acc['V'] != 0.0, acc['dVdT'] / acc['V'], 0.0)
        coverage = np.where(total_moles != 0.0, covered_moles / np.where(total_moles != 0.0, total_moles, 1.0), 0.0)

        full = dict(acc)
        full['G'] = G
        full['K'] = K
        full['alpha'] = alpha
        full['melts_coverage_fraction'] = coverage

        out = {}
        for prop in property_names:
            if prop == 'per_phase':
                out['per_phase'] = per_phase
            elif prop in full:
                out[prop] = full[prop]
            else:
                raise ValueError(
                    f"Unknown property {prop!r}. Available keys: "
                    f"{sorted(full.keys()) + ['per_phase']}"
                )
        out['melts_coverage_fraction'] = coverage
        return out
