"""
HeFESTo workspace parser and simulation management utilities.

This module provides three groups of functionality:

1. **Simulation tree preparation** – Sample bulk compositions from GEOROC/PetDB,
   speciate iron, convert oxides to element moles, and write per-simulation
   ``control`` + ``ad.in`` files into BatchNNNN/SimulationN directory trees
   ready for SLURM submission on the HPC cluster.

2. **Forward simulation wrapper** – ``forward_HeFESTo`` creates a SimulationN
   directory tree from a pre-built input array and runs all simulations via GNU
   parallel (intended for small local ensembles).

3. **Result import** – ``import_HeFESTo_components`` walks a workspace directory
   tree of completed SimulationN folders, parses fort.56/61/68/99 output files,
   and appends rows to a master CSV table compatible with DatasetIndexer.

Typical HPC workflow::

    prepare_HeFESTo_tree_fulladiabat(directory, georoc_dir, control_path, N=5000)
    # → submit with scripts/run_many_sbatches.py
    # → after completion:
    import_HeFESTo_components(workspace_dir, indexer, dataname='out.csv')
"""

import os
import re
import shutil
import argparse
import random
import subprocess
from time import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import sys
import numpy as np
rng = np.random.default_rng()
    
try: 
    import torch
except ImportError:
    print("PyTorch is not installed. Some functions may not be available")
    pass

src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)
file_path = str(Path(__file__).parent)
if file_path not in sys.path:
    sys.path.insert(0, file_path)
base_path = str(Path(__file__).parent.parent.parent.parent)
if base_path not in sys.path:
    sys.path.insert(0, base_path)

try:
    from ngibbs.utils.file_utils import (
        save_fixed_width_table,
        PHASE_ABBREVIATION_OVERRIDES,
        COMPONENT_ABBREVIATION_OVERRIDES,
        _safe_read_ws_table,
        _resolve_phase_name_from_abbr,
        _resolve_component_name_from_abbr,
        _build_reverse_component_phase_map,
        _parse_control_file,
        _parse_fort56,
        _resolve_component_phase,
        load_fort99_component_moles_and_labels,
        load_fort99_componentMoles,
        extract_bulk_properties_from_simulation_dir,
    )
except ImportError:
        from src.ngibbs.utils.file_utils import (
            save_fixed_width_table,
            PHASE_ABBREVIATION_OVERRIDES,
            COMPONENT_ABBREVIATION_OVERRIDES,
            _safe_read_ws_table,
            _resolve_phase_name_from_abbr,
            _resolve_component_name_from_abbr,
            _build_reverse_component_phase_map,
            _parse_control_file,
            _parse_fort56,
            _resolve_component_phase,
            load_fort99_component_moles_and_labels,
            load_fort99_componentMoles,
            extract_bulk_properties_from_simulation_dir,
        )


import numpy as np
import pandas as pd

from src.ngibbs.utils.math_utils import get_S, get_T

from ngibbs.config.constants import (
    COMPOSITIONAL_COMPONENTS_IN_PHASES_HEFESTO,
    HEFESTO_ABBREVIATION_TO_SHORT_NAMES,
    get_oxide_molar_mass,
    OXIDE_MOLAR_MASSES
)

_SIMULATION_DIR_PATTERN = re.compile(r'^(?:Simulation(\d+)|model_(\d{6}))$', re.IGNORECASE)
_T_COEF_LINE = re.compile(r'^\s*(b[^\s(=]+)\s*(?:\([^)]*\))?\s*=\s*([+-]?\d[\d.eE+-]*)')


EnsembleLocation = None

def make_PT_path(S, P, func, out_path=None):
    """Generate an ``ad.in`` pressure–temperature path file for HeFESTo.

    Writes a fixed-width three-column table (P, unused, T) where T is the
    isentropic temperature at each pressure point, with a small amount of
    Gaussian noise added to improve training coverage.

    Parameters
    ----------
    S : float
        Specific entropy in J/g/K used to compute the isentrope.
    P : array-like
        Pressure values in GPa along the path.
    func : callable
        Temperature function with signature ``func(S, P) -> T`` (K).
        Typically ``get_T`` from ``ngibbs.utils.math_utils``.
    out_path : str or Path, optional
        Output file path. If ``None`` the table is not written to disk.
    """
    AdIn = np.zeros((len(P), 3))  # Middle column unused
    AdIn[:, 0] = P
    AdIn[:, 2] = func(S=S, P=P) + np.random.normal(0, 5, size=len(P))
    save_fixed_width_table(AdIn, out_path=out_path)

def get_T_mars_simple(S, P):
    return 723.717303134 + -694.269778182 * S + 419.656745453 * S**2 + 13.9481733773 * P + -0.0816921679556 * P**2

def make_PT_path_martian(S, P, func, P_lit, cold_temp = 220, out_path=None):
    """Generate an ``ad.in`` pressure–temperature path file for Martian conditions.

    Like ``make_PT_path`` but overlays a conductive lithosphere: pressures
    below ``P_lit`` are replaced with a linear temperature profile that
    rises from ``cold_temp`` at the surface to the adiabat temperature at
    the lithosphere base. Gaussian noise is added to all points.

    Parameters
    ----------
    S : float
        Specific entropy in J/g/K for the adiabatic mantle.
    P : array-like
        Pressure values in GPa along the path.
    func : callable
        Adiabatic temperature function ``func(S, P) -> T`` (K).
    P_lit : float
        Lithosphere base pressure in GPa. Points with P ≤ P_lit are
        replaced with the conductive profile.
    cold_temp : float, default=220
        Surface temperature in K used as the cold-end anchor of the
        conductive regime.
    out_path : str or Path, optional
        Output file path. If ``None`` the table is not written to disk.
    """
    AdIn = np.zeros((len(P), 3))  # Middle column unused
    AdIn[:, 0] = P
    AdIn[:, 2] = func(S=S, P=P) 
    # Overwrite linear conductive regime
    conductive_mask = P <= P_lit
    AdIn[conductive_mask, 2] = cold_temp + (AdIn[conductive_mask, 2] - cold_temp) * (AdIn[conductive_mask, 0] / P_lit)
    AdIn[:, 2] = AdIn[:, 2] + np.random.normal(0, 5, size=len(P))
    save_fixed_width_table(AdIn, out_path=out_path)



def make_T_from_coef_file(coef_path: str):
    """Build a T(P, S, *, Si, Mg, Fe, Ca, Al, Na, Cr, O) function from a regression file.

    The coefficient file is read once when this factory is called; the returned
    function captures the values in its closure and performs no file I/O at
    call time.  The expected file format is::

        b0  (intercept) = <float>
        b_S             = <float>
        b_S^2           = <float>
        b_P             = <float>
        b_P^2           = <float>
        b_Si            = <float>
        b_Si^2          = <float>
        ... (Mg, Fe, Ca, Al, Na, Cr, O)

    The returned function has the same signature as ``get_T_mars``.

    Parameters
    ----------
    coef_path : str or Path
        Path to the coefficient text file.

    Returns
    -------
    callable
        ``get_T(P, S, *, Si, Mg, Fe, Ca, Al, Na, Cr, O) -> float | np.ndarray``
    """
    raw: Dict[str, float] = {}
    with open(coef_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            m = _T_COEF_LINE.match(line)
            if m:
                raw[m.group(1)] = float(m.group(2))

    def _get(key: str) -> float:
        if key not in raw:
            raise KeyError(f"Coefficient '{key}' not found in {coef_path}")
        return raw[key]

    b0   = _get('b0')
    bS,  bS2  = _get('b_S'),  _get('b_S^2')
    bP,  bP2  = _get('b_P'),  _get('b_P^2')
    bSi, bSi2 = _get('b_Si'), _get('b_Si^2')
    bMg, bMg2 = _get('b_Mg'), _get('b_Mg^2')
    bFe, bFe2 = _get('b_Fe'), _get('b_Fe^2')
    bCa, bCa2 = _get('b_Ca'), _get('b_Ca^2')
    bAl, bAl2 = _get('b_Al'), _get('b_Al^2')
    bNa, bNa2 = _get('b_Na'), _get('b_Na^2')
    bCr, bCr2 = _get('b_Cr'), _get('b_Cr^2')
    bO,  bO2  = _get('b_O'),  _get('b_O^2')

    def get_T(P, S, *, Si, Mg, Fe, Ca, Al, Na, Cr, O):
        return (
            b0
            + bS  * S  + bS2  * S**2
            + bP  * P  + bP2  * P**2
            + bSi * Si + bSi2 * Si**2
            + bMg * Mg + bMg2 * Mg**2
            + bFe * Fe + bFe2 * Fe**2
            + bCa * Ca + bCa2 * Ca**2
            + bAl * Al + bAl2 * Al**2
            + bNa * Na + bNa2 * Na**2
            + bCr * Cr + bCr2 * Cr**2
            + bO  * O  + bO2  * O**2
        )

    return get_T

#get_T_mars = make_T_from_coef_file("src/builder/HeFESTo/T_from_S_X_coefs_R2_0p99456.txt")

def _clean_workspace(workspace_dir: str) -> None:
    if os.path.exists(workspace_dir):
        for item in os.listdir(workspace_dir):
            item_path = os.path.join(workspace_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)


def _normalize_element_label(label: str) -> str:
    text = str(label).strip()
    if not text:
        return text
    if text.upper() == 'O':
        return 'O'
    return text[0].upper() + text[1:].lower()


def _build_control_lines(
    template_lines: List[str],
    element_values: Dict[str, float],
    run_code: List,
) -> List[str]:
    if len(template_lines) == 0:
        raise ValueError('Control template file is empty')

    lines = list(template_lines)
    lines[0] = ','.join(str(value) for value in run_code)

    oxides_start = None
    for i, line in enumerate(lines):
        if line.strip().lower() == 'oxides':
            oxides_start = i + 1
            break
    if oxides_start is None:
        raise ValueError("No 'oxides' block found in control template")

    for i in range(oxides_start, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            continue
        if stripped.lower().startswith('phase '):
            break
        if ',' in stripped:
            continue

        parts = stripped.split()
        if len(parts) < 3:
            continue

        element = _normalize_element_label(parts[0])
        if element not in element_values:
            continue

        value = float(element_values[element])
        value_text = f'{value:.5f}'
        third_col = parts[3] if len(parts) >= 4 else '0'
        lines[i] = f'{element:<2} {value_text:>12} {value_text:>11} {third_col}'

    return lines


def _resolve_control_template_path(control_template: str) -> str:
    if not isinstance(control_template, str) or not control_template.strip():
        raise ValueError('control_template must be a non-empty string')

    template_name = control_template.strip()
    if os.path.isabs(template_name):
        resolved = template_name
    else:
        batch_dir = os.path.join(Path(__file__).parent.parent.absolute(), 'batch')
        resolved = os.path.join(batch_dir, template_name)

    if not os.path.exists(resolved):
        raise FileNotFoundError(f'Missing HeFESTo control template: {resolved}')

    if not os.path.isfile(resolved):
        raise FileNotFoundError(f'Control template is not a file: {resolved}')

    return resolved


def _resolve_phase_change_control_template_path(control_dir: Path) -> Path:
    control_dir = Path(control_dir)

    if control_dir.is_file():
        return control_dir

    if not control_dir.exists():
        raise FileNotFoundError(f'Control template directory not found: {control_dir}')

    preferred_names = ('shallowHeFESTo', 'deepHeFESTo')
    for template_name in preferred_names:
        candidate = control_dir / template_name
        if candidate.is_file():
            return candidate

    template_files = sorted(item for item in control_dir.iterdir() if item.is_file())
    if len(template_files) == 0:
        raise FileNotFoundError(
            f'No control template files found in: {control_dir}'
        )

    return template_files[0]


def _contains_simulation_dirs(root_dir: Path) -> bool:
    root_dir = Path(root_dir)
    if not root_dir.exists():
        return False

    for current_root, dirnames, _ in os.walk(root_dir):
        for dirname in dirnames:
            if _SIMULATION_DIR_PATTERN.match(dirname):
                return True

    return False


def _next_available_batch_dir(root_dir: Path, batch_number: int) -> Path:
    candidate = root_dir / f'Batch{batch_number:04d}'
    while candidate.exists():
        batch_number += 1
        candidate = root_dir / f'Batch{batch_number:04d}'
    return candidate


def _normalize_run_code_rows(run_code, n_simulations: int) -> List[List]:
    if n_simulations <= 0:
        raise ValueError('n_simulations must be positive')

    if isinstance(run_code, np.ndarray):
        if run_code.ndim == 1:
            shared = run_code.tolist()
            if len(shared) == 0:
                raise ValueError('run_code cannot be empty')
            return [list(shared) for _ in range(n_simulations)]

        if run_code.ndim == 2:
            if run_code.shape[0] != n_simulations:
                raise ValueError(
                    '2D run_code must have one row per simulation in input_array'
                )
            if run_code.shape[1] == 0:
                raise ValueError('run_code rows cannot be empty')
            return [list(run_code[i, :]) for i in range(n_simulations)]

        raise ValueError('run_code numpy array must be 1D or 2D')

    if not isinstance(run_code, (list, tuple)):
        raise TypeError('run_code must be list-like or numpy array')

    if len(run_code) == 0:
        raise ValueError('run_code cannot be empty')

    first_item = run_code[0]
    is_2d = isinstance(first_item, (list, tuple, np.ndarray))

    if is_2d:
        if len(run_code) != n_simulations:
            raise ValueError(
                '2D run_code must have one row per simulation in input_array'
            )
        rows: List[List] = []
        for row in run_code:
            if not isinstance(row, (list, tuple, np.ndarray)):
                raise TypeError('All rows in 2D run_code must be list-like')
            row_list = list(row)
            if len(row_list) == 0:
                raise ValueError('run_code rows cannot be empty')
            rows.append(row_list)
        return rows

    shared = list(run_code)
    if len(shared) == 0:
        raise ValueError('run_code cannot be empty')
    return [list(shared) for _ in range(n_simulations)]


def forward_HeFESTo(
    input_array,
    keys,
    run_code,
    EnsembleLocation=EnsembleLocation,
    control_template: str = 'shallowHeFESTo',
):
    """
    Create and execute an ensemble of HeFESTo simulations.

    Parameters
    ----------
    input_array : np.ndarray
        Array of elemental conditions, shape (n_simulations, n_elements).
    keys : np.ndarray or list
        Element labels corresponding to columns in input_array.
    run_code : list
        Either:
        - 1D list-like values written to all simulation control files, or
        - 2D list-like/array with one row per simulation in input_array.
    EnsembleLocation : str
        Directory where SimulationN folders and runall.sh are written.
    control_template : str, default='shallowHeFESTo'
        Template filename in src/builder/alphamelts/batch (or absolute path)
        used to seed each simulation control file.
    """
    if EnsembleLocation is None:
        raise ValueError('EnsembleLocation must be set for forward_HeFESTo()')

    input_array = np.asarray(input_array)
    if input_array.ndim != 2:
        raise ValueError('input_array must be a 2D array')

    if input_array.shape[1] != len(keys):
        raise IndexError("Condition columns don't match keys")

    normalized_keys = [_normalize_element_label(key) for key in keys]
    run_code_rows = _normalize_run_code_rows(run_code, input_array.shape[0])

    control_template_path = _resolve_control_template_path(control_template)

    with open(control_template_path, 'r', encoding='utf-8', errors='ignore') as handle:
        template_lines = [line.rstrip('\n') for line in handle]

    _clean_workspace(EnsembleLocation)
    os.makedirs(EnsembleLocation, exist_ok=True)

    runall_lines: List[str] = []
    for i in range(input_array.shape[0]):
        sim_dir = os.path.join(EnsembleLocation, f'Simulation{i}')
        os.makedirs(sim_dir, exist_ok=True)

        control_path = os.path.join(sim_dir, 'control')
        shutil.copy(control_template_path, control_path)

        element_values = {
            normalized_keys[j]: float(input_array[i, j])
            for j in range(input_array.shape[1])
        }
        updated_control_lines = _build_control_lines(
            template_lines,
            element_values,
            run_code_rows[i],
        )

        with open(control_path, 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(updated_control_lines) + '\n')

        runall_lines.append(f'cd "{sim_dir}" ; HeFESTo')

    runall_path = os.path.join(EnsembleLocation, 'runall.sh')
    with open(runall_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(runall_lines) + '\n')

    os.system('cd "' + EnsembleLocation + '"; parallel < runall.sh; cd -')


def _validate_absolute_path(path_value: str, arg_name: str) -> str:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f'{arg_name} must be a non-empty string')

    resolved = path_value.strip()
    if not os.path.isabs(resolved):
        raise ValueError(f'{arg_name} must be an absolute path: {resolved}')

    return resolved


def _copy_template_tree(template_dir: str, destination_dir: str) -> None:
    for item in os.listdir(template_dir):
        src = os.path.join(template_dir, item)
        dst = os.path.join(destination_dir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

def _build_oxide_wt_from_row(row: pd.Series) -> Dict[str, float]:
    """Extract oxide weight-percent values from a GEOROC/PetDB DataFrame row.

    Missing or NaN values for SiO2, MgO, and FeO are replaced with random
    plausible values so that all simulations have a valid starting composition.
    An auxiliary key ``'Fe_total_moles'`` is included for downstream iron
    speciation. Other oxides (CaO, Al2O3, Na2O, Cr2O3) default to 0.

    Parameters
    ----------
    row : pd.Series
        A single row from a GEOROC/PetDB composition table.

    Returns
    -------
    dict
        Oxide wt% values keyed by oxide name, plus ``'Fe_total_moles'``.
    """
    oxide_cols = {
        'SiO2': 'SiO2',
        'MgO': 'MgO',
        'FeO': 'FeO',
        'CaO': 'CaO',
        'Al2O3': 'Al2O3',
        'Na2O': 'Na2O',
        'Cr2O3': 'Cr2O3',
    }

    wt: Dict[str, float] = {}
    for oxide, col_name in oxide_cols.items():
        value = row.get(col_name, 0.0)
        if pd.isna(value):
            value = 0.0
        wt[oxide] = float(value)*np.random.normal(1.0, 0.1)  # Add random noise to search more continuous composition space

    for oxide in ['SiO2', 'MgO', 'FeO']:
        if wt[oxide] > 0.0:
            continue
        if oxide == 'SiO2':
            wt[oxide] = float(np.random.uniform(30.0, 55.0))
        elif oxide == 'MgO':
            wt[oxide] = float(np.random.uniform(5.0, 40.0))
        elif oxide == 'FeO':
            wt[oxide] = float(np.random.uniform(0.1, 15.0))

    wt['Fe_total_moles'] = wt['FeO'] / OXIDE_MOLAR_MASSES['FeO']
    return wt

def _speciate_iron_and_normalize(oxide_wt: Dict[str, float], fe3_fet: float) -> Dict[str, float]:
    """Distribute total iron between Fe²⁺ and Fe³⁺, then renormalize to 100 wt%.

    Takes the ``'Fe_total_moles'`` from ``oxide_wt`` and splits it according
    to the requested Fe³⁺ fraction. FeO and Fe₂O₃ masses are recalculated
    from the resulting Fe²⁺ and Fe³⁺ moles. Minor oxides (CaO, Al₂O₃, Na₂O,
    Cr₂O₃) receive a small epsilon floor so they are never exactly zero in
    the HeFESTo control file.

    Parameters
    ----------
    oxide_wt : dict
        Output from ``_build_oxide_wt_from_row``, containing raw oxide wt%
        values and the ``'Fe_total_moles'`` auxiliary key.
    fe3_fet : float
        Target Fe³⁺/Fetotal molar ratio in [0, 0.1]. Values outside this
        range are clipped.

    Returns
    -------
    dict
        Renormalized oxide wt% values (sum ≈ 100) keyed by oxide name.
        Includes both ``'FeO'`` and ``'Fe2O3'``.
    """
    fe_total_moles = float(oxide_wt['Fe_total_moles'])
    fe3_fet = float(fe3_fet)

    fe3_moles = fe_total_moles * fe3_fet
    fe2_moles = fe_total_moles - fe3_moles
    fe2o3_moles = fe3_moles / 2.0
    feo_moles = fe2_moles

    eps = 0.001 # Force small non-zero values for minor oxides

    oxide_masses = {
        'SiO2': oxide_wt['SiO2'],
        'MgO': oxide_wt['MgO'],
        'FeO': feo_moles * OXIDE_MOLAR_MASSES['FeO'],
        'Fe2O3': fe2o3_moles * OXIDE_MOLAR_MASSES['Fe2O3'],
        'CaO': oxide_wt['CaO'] + eps,
        'Al2O3': oxide_wt['Al2O3'] + eps,
        'Na2O': oxide_wt['Na2O'] + eps,
        'Cr2O3': oxide_wt['Cr2O3'] + eps #* np.random.uniform(0.01, 0.5)
    }

    total_mass = float(sum(oxide_masses.values()))
    if total_mass <= 0:
        raise ValueError('Non-positive oxide mass after Fe speciation')

    return {oxide: 100.0 * mass / total_mass for oxide, mass in oxide_masses.items()}


def _oxide_wt_to_element_moles(oxide_wt_norm: Dict[str, float]) -> Dict[str, float]:
    """Convert normalized oxide wt% to element mole counts.

    Converts the 8-oxide system (SiO₂, MgO, FeO, Fe₂O₃, CaO, Al₂O₃, Na₂O,
    Cr₂O₃) to element moles (Si, Mg, Fe, Ca, Al, Na, Cr, O) using the
    stoichiometric molar mass of each oxide. Total iron is expressed as a
    single Fe mole count (Fe²⁺ + Fe³⁺ combined); oxygen is computed from
    the full oxide stoichiometry.

    These element moles are the direct input to HeFESTo ``control`` files.

    Parameters
    ----------
    oxide_wt_norm : dict
        Renormalized oxide wt% values from ``_speciate_iron_and_normalize``.

    Returns
    -------
    dict
        Element moles keyed by element symbol (Si, Mg, Fe, Ca, Al, Na, Cr, O).
    """
    n_sio2 = oxide_wt_norm['SiO2'] / OXIDE_MOLAR_MASSES['SiO2']
    n_mgo = oxide_wt_norm['MgO'] / OXIDE_MOLAR_MASSES['MgO']
    n_feo = oxide_wt_norm['FeO'] / OXIDE_MOLAR_MASSES['FeO']
    n_fe2o3 = oxide_wt_norm['Fe2O3'] / OXIDE_MOLAR_MASSES['Fe2O3']
    n_cao = oxide_wt_norm['CaO'] / OXIDE_MOLAR_MASSES['CaO']
    n_al2o3 = oxide_wt_norm['Al2O3'] / OXIDE_MOLAR_MASSES['Al2O3']
    n_na2o = oxide_wt_norm['Na2O'] / OXIDE_MOLAR_MASSES['Na2O']
    n_cr2o3 = oxide_wt_norm['Cr2O3'] / OXIDE_MOLAR_MASSES['Cr2O3']

    return {
        'Si': n_sio2,
        'Mg': n_mgo,
        'Fe': n_feo + 2.0 * n_fe2o3,
        'Ca': n_cao,
        'Al': 2.0 * n_al2o3,
        'Na': 2.0 * n_na2o,
        'Cr': 2.0 * n_cr2o3,
        'O': (
            2.0 * n_sio2 + n_mgo + n_feo + 3.0 * n_fe2o3 + n_cao +
            3.0 * n_al2o3 + n_na2o + 3.0 * n_cr2o3
        ),
    }


def _normalize_total_moles(element_moles: Dict[str, float], target_total_moles: float) -> Dict[str, float]:
    """Scale element mole counts so the total sum equals ``target_total_moles``.

    HeFESTo control files expect element quantities on a fixed total-moles
    scale (typically 24 moles for mantle compositions). This function applies
    a uniform scale factor to bring the raw mole counts to that target.

    Parameters
    ----------
    element_moles : dict
        Unnormalized element moles, e.g. from ``_oxide_wt_to_element_moles``.
    target_total_moles : float
        Desired sum of all element moles in the output (e.g. 24.0).

    Returns
    -------
    dict
        Rescaled element moles with the same keys as ``element_moles``.
    """
    total_moles = float(sum(element_moles.values()))
    if total_moles <= 0.0:
        raise ValueError('Cannot normalize element moles with non-positive total')

    scale = float(target_total_moles) / total_moles
    return {key: value * scale for key, value in element_moles.items()}



def prepare_HeFESTo_tree_fulladiabat(directory: Path, GEOROC_DIR: Path, control_path: Path, N: int) -> None:
    """Prepare a BatchNNNN/SimulationN directory tree for HeFESTo isentrope runs.

    Samples N bulk compositions from the GEOROC/PetDB database as 30% pure
    ultramafic (MgO > 20 wt%), 20% pure mafic (5 < MgO ≤ 20 wt%), and 50%
    linear mixtures of a mafic and an ultramafic end-member with ultramafic
    fraction α ~ U(0, 1). Iron is then speciated across a Fe³⁺/Fetotal grid
    from 0 to 10%, oxide wt% is converted to element moles (normalized to 24
    total moles), and per-simulation ``control`` and ``ad.in`` files are
    written. Simulations are grouped into batches of 1000 in ``BatchNNNN/``
    subdirectories for SLURM array job submission.

    The P–T path in each ``ad.in`` file is an isentrope computed from a random
    mantle potential temperature (Tp ∈ [1473, 1923] K) using ``get_S`` and
    ``get_T``, with small Gaussian noise (σ = 5 K) added for coverage diversity.

    Parameters
    ----------
    directory : Path
        Root directory under which ``BatchNNNN/`` subdirectories are created.
    GEOROC_DIR : Path
        Path to a GEOROC/PetDB CSV file with oxide wt% columns.
    control_path : Path
        Path to a HeFESTo control template file (e.g. ``shallowHeFESTo``).
    N : int
        Total number of simulations to generate.
    """
    
    directory = Path(directory)
    GEOROC_DIR = Path(GEOROC_DIR)
    control_path = Path(control_path)
    if not control_path.exists() or not control_path.is_file():
        raise FileNotFoundError(f'Control template not found: {control_path}')
    directory.mkdir(parents=True, exist_ok=True)

    batch_dir = _next_available_batch_dir(directory, 1)
    batch_dir.mkdir(parents=True, exist_ok=False)
    simulations_in_batch = 0
    batch_number = int(batch_dir.name.removeprefix('Batch'))

    # Example: Generate ad.in files for a range of mantle potential temperatures and calcium contents
    Mps = 273 + 1200 + np.random.uniform(0, 1, N) * (1650 - 1200)  #  mantle potential temperatures
    target_total_moles = 24.0

    georoc_df = pd.read_csv(GEOROC_DIR)
    mgo_col = 'MgO'
    with open(control_path, 'r', encoding='utf-8', errors='ignore') as handle:
        template_lines = [line.rstrip('\n') for line in handle]
    mgo_numeric = pd.to_numeric(georoc_df[mgo_col], errors='coerce').fillna(0.0)
    ultramafic_df = georoc_df[mgo_numeric > 20.0]
    mafic_df = georoc_df[(mgo_numeric > 5.0) & (mgo_numeric <= 20.0)]

    # Composition split: 30% ultramafic, 20% mafic, 50% mixtures of the two
    N_ultra = int(round(N * 0.3))
    N_mafic = int(round(N * 0.2))
    N_mix = N - N_ultra - N_mafic

    ultra_pure = ultramafic_df.sample(n=N_ultra, replace=True).reset_index(drop=True)
    mafic_pure = mafic_df.sample(n=N_mafic, replace=True).reset_index(drop=True)
    mix_ultra = ultramafic_df.sample(n=N_mix, replace=True).reset_index(drop=True)
    mix_mafic = mafic_df.sample(n=N_mix, replace=True).reset_index(drop=True)
    mix_alphas = np.random.uniform(0.0, 1.0, N_mix)  # ultramafic fraction in mixture

    # Build spec list: (category, row_a, row_b_or_None, alpha_or_None). ``category``
    # records which pool(s) the spec was drawn from so a rejected sample (see
    # MAX_COMPOSITION_ATTEMPTS below) can be redrawn from the same pool(s).
    specs: List[Tuple] = (
        [('ultra', row, None, None) for _, row in ultra_pure.iterrows()] +
        [('mafic', row, None, None) for _, row in mafic_pure.iterrows()] +
        [('mix', mix_ultra.iloc[i], mix_mafic.iloc[i], mix_alphas[i]) for i in range(N_mix)]
    )
    order = rng.permutation(N)
    specs = [specs[i] for i in order]

    logfo2 = np.random.uniform(-6, 3, N) # logfO2 relative to FMQ
    logfe3_fe2 = (0.2*logfo2) - 1
    fe3_fe2 = 10**logfe3_fe2
    rand0 = 1 #np.random.choice([0, 1], size=N, p=[0.1, 0.9]) # Randomly assign some simulations to have 0 Fe3+/FeT
    fe3_fet_grid = rand0 * fe3_fe2 / (1 + fe3_fe2) # Convert to Fe3+/Fetotal
    element_keys = np.array(['Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr', 'O'])
    P0s = np.random.uniform(0, 1, size=N)
    run_code = [[float(P0), float(P0 + 139), 138, 0, 0, 0, -1, 0, 0, 0, 0] for P0 in P0s] # ad.in files made downstream
    element_rows: List[List[float]] = []
    wts: List[str] = []

    for sim_idx, spec in enumerate(specs):
        if simulations_in_batch >= 1000:
            batch_number += 1
            batch_dir = _next_available_batch_dir(directory, batch_number)
            batch_dir.mkdir(parents=True, exist_ok=False)
            simulations_in_batch = 0

        sim_dir = batch_dir / f'Simulation{simulations_in_batch + 1}'
        sim_dir.mkdir(parents=True, exist_ok=True)
        simulations_in_batch += 1

        ratio = float(fe3_fet_grid[sim_idx])
        category, row_a, row_b, alpha = spec

        # Reject and redraw compositions that end up too iron-rich (FeOT > 30
        # wt%) or too silica-poor (SiO2 < 25 wt%) for HeFESTo to handle
        # robustly. The first attempt uses the pre-sampled spec; subsequent
        # attempts redraw fresh row(s) from the same pool(s) the spec came from.
        MAX_COMPOSITION_ATTEMPTS = 25
        for attempt in range(MAX_COMPOSITION_ATTEMPTS):
            if attempt > 0:
                if category == 'ultra':
                    row_a = ultramafic_df.sample(n=1).iloc[0]
                elif category == 'mafic':
                    row_a = mafic_df.sample(n=1).iloc[0]
                else:  # 'mix'
                    row_a = ultramafic_df.sample(n=1).iloc[0]
                    row_b = mafic_df.sample(n=1).iloc[0]
                    alpha = float(np.random.uniform(0.0, 1.0))

            if row_b is None:
                # Pure mafic or pure ultramafic
                base_oxide_wt = _build_oxide_wt_from_row(row_a)
            else:
                # Linear mixture: alpha = ultramafic fraction, (1-alpha) = mafic fraction
                ultra_oxide = _build_oxide_wt_from_row(row_a)
                mafic_oxide = _build_oxide_wt_from_row(row_b)
                base_oxide_wt = {
                    key: float(alpha) * ultra_oxide.get(key, 0.0) + (1.0 - float(alpha)) * mafic_oxide.get(key, 0.0)
                    for key in set(ultra_oxide) | set(mafic_oxide)
                }

            # Build element moles with total Fe treated as fully ferrous (Fe3+/FeT = 0)
            # for now. The target fe3_fet ratio is applied further below, on the
            # *final* total Fe -- i.e. after the random Fe/Cr/Ca/Al/Si/Mg perturbation
            # -- since HeFESTo's control file has no separate Fe3+ element and can only
            # be told the intended oxidation state via the bulk O/Fe ratio. Speciating
            # before that perturbation would bake the oxidation signal into a Fe
            # budget that then gets overwritten, diluting (and effectively
            # randomizing) the intended fe3_fet signal.
            unspeciated_wt = _speciate_iron_and_normalize(base_oxide_wt, 0.0)
            element_moles = _oxide_wt_to_element_moles(unspeciated_wt)

            # Normalize to total moles = 1, then perturb Fe/Cr/Ca/Al/Si/Mg to add
            # compositional diversity beyond the GEOROC end-members. Each
            # perturbation also adjusts O to preserve mass balance for the default
            # oxide it represents (FeO: O:Fe=1, Cr2O3: O:Cr=1.5, CaO: O:Ca=1,
            # Al2O3: O:Al=1.5, SiO2: O:Si=2, MgO: O:Mg=1) -- otherwise cations are
            # added/removed with no matching oxygen, which distorts the bulk O
            # budget (and, for Fe, would swamp the oxidation-state signal applied
            # just below).
            element_moles = _normalize_total_moles(element_moles, 1.0)

            mg_val = element_moles.get('Mg', 0.0)

            d_fe = float(np.random.uniform(0.00, 0.05))*float(np.random.uniform()>0.3) # Add Fe to some compositions, but not all
            d_cr = float(np.random.uniform(0.0, 0.01))
            d_si = float(np.random.uniform(-0.5/24, 2.0/24)) # Read these values off of norm 24 histograms
            d_mg = float(np.random.uniform(1.5/24, 2.75/24)) * (mg_val < 1.5/24)
            d_mg += float(np.random.uniform(0.25/24, 2.0/24))
            d_ca = -(float(np.random.uniform(0.0, 0.1)) * (element_moles.get('Ca', 0.0) > 0.1))
            d_al = -(float(np.random.uniform(0.0, 0.05)) * (element_moles.get('Al', 0.0) > 0.05))

            element_moles['Fe'] = element_moles.get('Fe', 0.0) + d_fe
            element_moles['Cr'] = element_moles.get('Cr', 0.0) + d_cr
            element_moles['Ca'] = element_moles.get('Ca', 0.0) + d_ca
            element_moles['Al'] = element_moles.get('Al', 0.0) + d_al
            element_moles['Si'] = element_moles.get('Si', 0.0) + d_si
            element_moles['Mg'] = element_moles.get('Mg', 0.0) + d_mg
            element_moles['O'] = (
                element_moles.get('O', 0.0)
                + 1.0 * d_fe   # FeO baseline for the added Fe (ferrous by default)
                + 1.5 * d_cr   # Cr2O3
                + 1.0 * d_ca   # CaO
                + 1.5 * d_al   # Al2O3
                + 2.0 * d_si   # SiO2
                + 1.0 * d_mg   # MgO
            )

            # Apply the target Fe3+/FeT ratio now, on the final total Fe, by adding
            # the extra oxygen that a Fe2O3 (vs FeO) stoichiometry requires beyond
            # the FeO baseline just added above: 1.5 O per Fe3+ atom instead of 1.0
            # O per Fe2+ atom, i.e. +0.5 O per Fe3+ atom. This is algebraically
            # identical to speciating the final Fe budget directly through
            # _speciate_iron_and_normalize.
            fe_total = element_moles.get('Fe', 0.0)
            element_moles['O'] = element_moles.get('O', 0.0) + 0.5 * ratio * fe_total

            element_moles = _normalize_total_moles(element_moles, target_total_moles)

            # Reject compositions that are too iron-rich or too silica-poor for
            # HeFESTo, and redraw. Checked on the final (post-perturbation,
            # 24-mole) composition, since that's what actually gets written out.
            bulk_comp_wt, _, _ = _compute_bulk_from_elements(element_moles)
            feot_wt = bulk_comp_wt['FeO'] + bulk_comp_wt['Fe2O3'] * (
                2.0 * OXIDE_MOLAR_MASSES['FeO'] / OXIDE_MOLAR_MASSES['Fe2O3']
            )
            sio2_wt = bulk_comp_wt['SiO2']
            if feot_wt <= 30.0 and sio2_wt >= 25.0:
                break
        else:
            print(
                f'Sim {sim_idx+1}: no composition satisfied FeOT<=30%, SiO2>=25% '
                f'after {MAX_COMPOSITION_ATTEMPTS} attempts; using last draw '
                f'(FeOT={feot_wt:.2f}%, SiO2={sio2_wt:.2f}%)'
            )

        wt_debug = ', '.join(f'{key}={value:.4f}' for key, value in element_moles.items())
        print(f'Sim {sim_idx+1} Fe3/FeT={ratio:.4f} -> {wt_debug}')
        wts.append(wt_debug)

        element_rows.append([element_moles[key] for key in element_keys])

        control_copy_path = sim_dir / 'control'
        shutil.copy2(control_path, control_copy_path)
        updated_control_lines = _build_control_lines(
            template_lines=template_lines,
            element_values=element_moles,
            run_code=run_code[sim_idx],
        )
        with open(control_copy_path, 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(updated_control_lines) + '\n')

        make_PT_path( # adds noise
            P=np.linspace(run_code[sim_idx][0], run_code[sim_idx][1], run_code[sim_idx][2] + 2),
            S=get_S(T=Mps[sim_idx], Ca=element_moles['Ca']),
            func=get_T,
            out_path=sim_dir / 'ad.in',
        )

def prepare_HeFESTo_tree_Mars(directory: Path, GEOROC_DIR: Path, control_path: Path, N: int) -> None:
    """Prepare a BatchNNNN/SimulationN directory tree for Martian HeFESTo runs.

    Variant of ``prepare_HeFESTo_tree_fulladiabat`` tuned for Mars mantle
    conditions. Key differences:

    - Pressure range is 0–30 GPa (upper mantle / transition zone only).
    - Compositions are drawn as 20% pure mafic (5.5 < MgO ≤ 20 wt%), 20% pure
      ultramafic (MgO > 20 wt%), and 60% linear mixtures of a mafic and an
      ultramafic end-member with mafic fraction α ~ U(0.1, 0.9). After oxide
      conversion, compositions are normalized to total moles = 1, a random Fe
      addition Δfe ~ U(0, 0.2) is applied, then renormalized to 24 total moles.
    - Entropy is drawn uniformly from [1.75, 2.9] J/g/K instead of being
      derived from a potential-temperature regression.
    - Each ``ad.in`` file includes a conductive lithosphere overlay: for
      pressures below a randomly sampled lithosphere base pressure (P_lit ∈
      [1.5, 9] GPa), temperature increases linearly from a cold surface
      anchor rather than following the mantle adiabat.

    Parameters
    ----------
    directory : Path
        Root directory under which ``BatchNNNN/`` subdirectories are created.
    GEOROC_DIR : Path
        Path to a GEOROC/PetDB CSV file with oxide wt% columns.
    control_path : Path
        Path to a HeFESTo control template file.
    N : int
        Total number of simulations to generate.
    """

    directory = Path(directory)
    GEOROC_DIR = Path(GEOROC_DIR)
    control_path = Path(control_path)
    if not control_path.exists() or not control_path.is_file():
        raise FileNotFoundError(f'Control template not found: {control_path}')
    directory.mkdir(parents=True, exist_ok=True)

    batch_dir = _next_available_batch_dir(directory, 1)
    batch_dir.mkdir(parents=True, exist_ok=False)
    simulations_in_batch = 0
    batch_number = int(batch_dir.name.removeprefix('Batch'))

    Ss = np.random.uniform(1.75, 2.85, N) # NOTE: THIS DOESN'T SEEM TO BE CALIBRATED CORRECTLY? PLOT YOUR PT PATHS!!!
    P_lit = np.random.uniform(1.5, 9, N)  # GPa, conductive regime lithosphere thicknesses
    target_total_moles = 24.0

    georoc_df = pd.read_csv(GEOROC_DIR)
    mgo_col = 'MgO'
    with open(control_path, 'r', encoding='utf-8', errors='ignore') as handle:
        template_lines = [line.rstrip('\n') for line in handle]

    mgo_numeric = pd.to_numeric(georoc_df[mgo_col], errors='coerce').fillna(0.0)
    ultramafic_df = georoc_df[mgo_numeric > 20.0]
    mafic_df = georoc_df[(mgo_numeric > 5.5) & (mgo_numeric <= 20.0)]

    # Composition split: 20% mafic, 20% ultramafic, 60% mixtures
    N_mafic = int(round(N * 0.2))
    N_ultra = int(round(N * 0.2))
    N_mix = N - N_mafic - N_ultra

    mafic_pure = mafic_df.sample(n=N_mafic, replace=True).reset_index(drop=True)
    ultra_pure = ultramafic_df.sample(n=N_ultra, replace=True).reset_index(drop=True)
    mix_mafic = mafic_df.sample(n=N_mix, replace=True).reset_index(drop=True)
    mix_ultra = ultramafic_df.sample(n=N_mix, replace=True).reset_index(drop=True)
    mix_alphas = np.random.uniform(0.1, 0.9, N_mix)  # mafic fraction in mixture

    # Build spec list: (row_a, row_b_or_None, alpha_or_None)
    specs: List[Tuple] = (
        [(row, None, None) for _, row in mafic_pure.iterrows()] +
        [(row, None, None) for _, row in ultra_pure.iterrows()] +
        [(mix_mafic.iloc[i], mix_ultra.iloc[i], mix_alphas[i]) for i in range(N_mix)]
    )
    order = rng.permutation(N)
    specs = [specs[i] for i in order]

    logfo2 = np.random.uniform(-6, 3, N)  # logfO2 relative to FMQ
    logfe3_fe2 = (0.2 * logfo2) - 1
    fe3_fe2 = 10 ** logfe3_fe2
    fe3_fet_grid = fe3_fe2 / (1 + fe3_fe2)  # Convert to Fe3+/Fetotal
    element_keys = np.array(['Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr', 'O'])
    P0s = np.random.uniform(0, 0.5, size=N)
    run_code = [[float(P0), float(P0 + 25.5), 50, 0, 0, 0, -1, 0, 0, 0, 0] for P0 in P0s]  # ad.in files made downstream
    element_rows: List[List[float]] = []
    wts: List[str] = []

    for sim_idx, spec in enumerate(specs):
        if simulations_in_batch >= 1000:
            batch_number += 1
            batch_dir = _next_available_batch_dir(directory, batch_number)
            batch_dir.mkdir(parents=True, exist_ok=False)
            simulations_in_batch = 0

        sim_dir = batch_dir / f'Simulation{simulations_in_batch + 1}'
        sim_dir.mkdir(parents=True, exist_ok=True)
        simulations_in_batch += 1

        ratio = float(fe3_fet_grid[sim_idx])
        row_a, row_b, alpha = spec

        # Build element moles with total Fe treated as fully ferrous (Fe3+/FeT = 0)
        # for now. The target fe3_fet ratio is applied further below, on the
        # *final* total Fe -- i.e. after the random Fe/Cr/Ca/Al perturbation --
        # since HeFESTo's control file has no separate Fe3+ element and can only
        # be told the intended oxidation state via the bulk O/Fe ratio. Speciating
        # before that perturbation would bake the oxidation signal into a Fe
        # budget that then gets overwritten, diluting (and effectively
        # randomizing) the intended fe3_fet signal.
        if row_b is None:
            # Pure mafic or pure ultramafic
            base_oxide_wt = _build_oxide_wt_from_row(row_a)
            unspeciated_wt = _speciate_iron_and_normalize(base_oxide_wt, 0.0)
            element_moles = _oxide_wt_to_element_moles(unspeciated_wt)
        else:
            # Linear mixture: alpha = mafic fraction, (1-alpha) = ultramafic fraction
            mafic_oxide = _build_oxide_wt_from_row(row_a)
            ultra_oxide = _build_oxide_wt_from_row(row_b)
            mafic_unspec = _speciate_iron_and_normalize(mafic_oxide, 0.0)
            ultra_unspec = _speciate_iron_and_normalize(ultra_oxide, 0.0)
            mafic_moles = _oxide_wt_to_element_moles(mafic_unspec)
            ultra_moles = _oxide_wt_to_element_moles(ultra_unspec)
            element_moles = {
                key: float(alpha) * mafic_moles[key] + (1.0 - float(alpha)) * ultra_moles[key]
                for key in mafic_moles
            }

        # Normalize to total moles = 1, then perturb Fe/Cr/Ca/Al to add
        # compositional diversity beyond the two GEOROC end-members. Each
        # perturbation also adjusts O to preserve mass balance for the default
        # oxide it represents (FeO: O:Fe=1, Cr2O3: O:Cr=1.5, CaO: O:Ca=1,
        # Al2O3: O:Al=1.5) -- otherwise cations are added/removed with no
        # matching oxygen, which distorts the bulk O budget (and, for Fe,
        # would swamp the oxidation-state signal applied just below).
        element_moles = _normalize_total_moles(element_moles, 1.0)

        mg_val = element_moles.get('Mg', 0.0)

        d_fe = float(np.random.uniform(0.0, 0.1))
        d_cr = float(np.random.uniform(0.0, 0.01))
        d_si = float(np.random.uniform(-0.5/24, 2.0/24)) # Read these values off of norm 24 histograms
        d_mg = float(np.random.uniform(1.5/24, 2.75/24)) * (mg_val < 2.5/24)
        d_mg += float(np.random.uniform(0.25/24, 2.0/24))
        d_ca = -(float(np.random.uniform(0.0, 0.1)) * (element_moles.get('Ca', 0.0) > 0.1))
        d_al = -(float(np.random.uniform(0.0, 0.05)) * (element_moles.get('Al', 0.0) > 0.05))

        element_moles['Fe'] = element_moles.get('Fe', 0.0) + d_fe
        element_moles['Cr'] = element_moles.get('Cr', 0.0) + d_cr
        element_moles['Ca'] = element_moles.get('Ca', 0.0) + d_ca
        element_moles['Al'] = element_moles.get('Al', 0.0) + d_al
        element_moles['Si'] = element_moles.get('Si', 0.0) + d_si
        element_moles['Mg'] = element_moles.get('Mg', 0.0) + d_mg
        element_moles['O'] = (
            element_moles.get('O', 0.0)
            + 1.0 * d_fe   # FeO baseline for the added Fe (ferrous by default)
            + 1.5 * d_cr   # Cr2O3
            + 1.0 * d_ca   # CaO
            + 1.5 * d_al   # Al2O3
            + 2.0 * d_si   # SiO2
            + 1.0 * d_mg   # MgO
        )

        # Apply the target Fe3+/FeT ratio now, on the final total Fe, by adding
        # the extra oxygen that a Fe2O3 (vs FeO) stoichiometry requires beyond
        # the FeO baseline just added above: 1.5 O per Fe3+ atom instead of 1.0
        # O per Fe2+ atom, i.e. +0.5 O per Fe3+ atom. This is algebraically
        # identical to speciating the final Fe budget directly through
        # _speciate_iron_and_normalize.
        fe_total = element_moles.get('Fe', 0.0)
        element_moles['O'] = element_moles.get('O', 0.0) + 0.5 * ratio * fe_total

        element_moles = _normalize_total_moles(element_moles, target_total_moles)

        wt_debug = ', '.join(f'{key}={value:.4f}' for key, value in element_moles.items())
        print(f'Sim {sim_idx+1} Fe3/FeT={ratio:.4f} -> {wt_debug}')
        wts.append(wt_debug)

        element_rows.append([element_moles[key] for key in element_keys])

        control_copy_path = sim_dir / 'control'
        shutil.copy2(control_path, control_copy_path)
        updated_control_lines = _build_control_lines(
            template_lines=template_lines,
            element_values=element_moles,
            run_code=run_code[sim_idx],
        )
        with open(control_copy_path, 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(updated_control_lines) + '\n')

        make_PT_path_martian( # built ad.in file
            P=np.linspace(run_code[sim_idx][0], run_code[sim_idx][1], run_code[sim_idx][2] + 2),
            S=Ss[sim_idx],
            P_lit=P_lit[sim_idx],
            #func=lambda S, P, _e=element_moles: get_T_mars(S=S, P=P, **_e),
            func = get_T_mars_simple,
            out_path=sim_dir / 'ad.in',
            )

def _find_all_control_files(workspace_dir: str) -> List[str]:
    """Return sorted paths to every ``control`` file inside a SimulationN dir at any depth."""
    paths: List[str] = []
    for root, _dirs, files in os.walk(workspace_dir):
        if _SIMULATION_DIR_PATTERN.match(os.path.basename(root)) and 'control' in files:
            paths.append(os.path.join(root, 'control'))
    return sorted(paths)


def plot_bulk_compositions(
    workspace_dir: str,
    out_path: Optional[str] = None,
    max_sims: Optional[int] = None,
) -> pd.DataFrame:
    """Parse bulk compositions from all control files in a workspace and plot them.

    Walks ``workspace_dir`` recursively, finds every ``control`` file inside a
    ``SimulationN`` (or ``model_NNNNNN``) subdirectory, and converts each to
    oxide wt% and element moles via ``_parse_control_file`` +
    ``_compute_bulk_from_elements``.  Produces a two-row matplotlib figure:

    - **Row 1** – scatter plots of SiO2, FeO_T, CaO, and Al2O3 vs MgO (wt%).
    - **Row 2** – histograms of the four major element mole counts
      (Si, Mg, Fe, O) on the 24-mole scale used by HeFESTo.
    - **Row 3** – Si vs Mg (cation mole fraction) scatter, with Mg = 0.9·Si
      and Mg = 1.1·Si reference lines.

    Parameters
    ----------
    workspace_dir : str
        Root directory containing ``BatchNNNN/SimulationN`` or flat
        ``SimulationN`` subdirectories.
    out_path : str or None, default=None
        If given, the figure is saved here instead of displayed interactively.
    max_sims : int or None, default=None
        Cap on simulations to read (useful for quick previews on large trees).

    Returns
    -------
    pd.DataFrame
        One row per simulation with oxide wt% columns (SiO2, MgO, FeO, Fe2O3,
        CaO, Al2O3, Na2O, Cr2O3, FeOT_wt) and element mole columns
        (Si_mol, Mg_mol, Fe_mol, Ca_mol, Al_mol, Na_mol, Cr_mol, O_mol).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('matplotlib is required for plot_bulk_compositions()')

    control_files = _find_all_control_files(workspace_dir)
    if max_sims is not None:
        control_files = control_files[:int(max_sims)]
    if not control_files:
        raise FileNotFoundError(f'No control files found under: {workspace_dir}')

    records: List[Dict[str, float]] = []
    n_failed = 0
    for ctrl_path in control_files:
        try:
            element_moles, _ = _parse_control_file(ctrl_path)
            bulk_wt, bulk_elements, _ = _compute_bulk_from_elements(element_moles)
            record: Dict[str, float] = {f'{k}_mol': v for k, v in bulk_elements.items()}
            record.update(bulk_wt)
            records.append(record)
        except Exception:
            n_failed += 1

    if not records:
        raise ValueError('No valid compositions could be parsed from control files.')
    if n_failed:
        print(f'Warning: {n_failed} control file(s) could not be parsed and were skipped.')

    df = pd.DataFrame(records).fillna(0.0)

    # Normalize so cation moles (excluding O) sum to 1; O is rescaled by the same factor
    mol_cols = [c for c in df.columns if c.endswith('_mol')]
    cation_cols = [c for c in mol_cols if c != 'O_mol']
    cation_totals = df[cation_cols].sum(axis=1)
    df[mol_cols] = df[mol_cols].div(cation_totals, axis=0)

    # Total-iron as FeO equivalent for scatter plots
    df['FeOT_wt'] = (
        df['FeO']
        + df['Fe2O3'] * (2.0 * OXIDE_MOLAR_MASSES['FeO'] / OXIDE_MOLAR_MASSES['Fe2O3'])
    )

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 4, figsize=(16, 15))
    fig.suptitle(
        f'Bulk compositions  —  {len(df):,} simulations  —  {workspace_dir}',
        fontsize=10,
    )

    # Row 0: oxide scatter plots vs MgO
    scatter_specs = [
        ('SiO2',    'SiO2 (wt%)'),
        ('FeOT_wt', 'FeO_T (wt%)'),
        ('CaO',     'CaO (wt%)'),
        ('Al2O3',   'Al2O3 (wt%)'),
    ]
    mgo = df['MgO']
    for ax, (col, ylabel) in zip(axes[0], scatter_specs):
        ax.scatter(mgo, df[col], s=3, alpha=0.35, color='steelblue', rasterized=True)
        ax.set_xlabel('MgO (wt%)')
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.4, alpha=0.5)

    # Rows 1–2: one histogram per element (all 8, normalized to sum=1)
    hist_specs = [
        ('Si_mol', 'Si (cation mol fraction)'),
        ('Mg_mol', 'Mg (cation mol fraction)'),
        ('Fe_mol', 'Fe (cation mol fraction)'),
        ('Ca_mol', 'Ca (cation mol fraction)'),
        ('Al_mol', 'Al (cation mol fraction)'),
        ('Na_mol', 'Na (cation mol fraction)'),
        ('Cr_mol', 'Cr (cation mol fraction)'),
        ('O_mol',  'O (mol / cation mol)'),
    ]
    for ax, (col, xlabel) in zip(axes[1:3].flat, hist_specs):
        ax.hist(df[col], bins=50, color='steelblue', alpha=0.75, edgecolor='none')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.grid(True, linewidth=0.4, alpha=0.5, axis='y')

    # Row 3: Si vs Mg (cation mole fraction), with Mg/Si = 0.9 and 1.1 reference lines
    ax_simg = axes[3, 0]
    si_mol = df['Si_mol']
    mg_mol = df['Mg_mol']
    ax_simg.scatter(si_mol, mg_mol, s=3, alpha=0.35, color='steelblue', rasterized=True)

    line_color = '#eb6834'  # categorical slot 2 (orange), per repo dataviz palette
    x_ref = np.array([si_mol.min(), si_mol.max()])
    ax_simg.plot(x_ref, 0.9 * x_ref, linestyle='--', linewidth=1.6, color=line_color, label='Mg = 0.9·Si')
    ax_simg.plot(x_ref, 1.1 * x_ref, linestyle=':', linewidth=2.0, color=line_color, label='Mg = 1.1·Si')
    ax_simg.set_xlim(x_ref[0], x_ref[1])
    ax_simg.set_xlabel('Si (cation mol fraction)')
    ax_simg.set_ylabel('Mg (cation mol fraction)')
    ax_simg.grid(True, linewidth=0.4, alpha=0.5)
    ax_simg.legend(loc='best', fontsize=8, frameon=False)

    for ax in axes[3, 1:]:
        ax.axis('off')

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Figure saved to {out_path}')
    else:
        plt.show()

    return df


def prepare_HeFESTo_tree_from_phase_changes(
    directory: Path,
    phase_path: Path,
    CONTROL_DIR: Path,
    limit: Optional[int] = None,
) -> None:
    """
    Prepare a directory tree from a phase-boundary CSV.

    The CSV is expected to contain pairs of rows that bound a phase change.
    Each pair is copied into its own SimulationN directory, the bulk element
    abundances are written verbatim into a HeFESTo control file, and the paired
    rows are preserved in a per-simulation CSV for downstream reuse.

    Parameters
    ----------
    limit : int, optional
        If given, only ``limit`` phase-change pairs are loaded. Pair indices
        are drawn as random integers in ``[0, total_pairs)`` and each pair's
        two rows (``idx * 2`` and ``idx * 2 + 1``) are read directly off disk
        via ``skiprows``, so the rest of the CSV is never materialized in
        memory.
    """

    directory = Path(directory)
    phase_path = Path(phase_path)


    if not phase_path.exists():
        raise FileNotFoundError(f'Phase boundary file not found: {phase_path}')
    if not phase_path.is_file():
        raise FileNotFoundError(f'Phase boundary file is not a file: {phase_path}')

    """if _contains_simulation_dirs(directory):
        raise FileExistsError(
            f'Target directory already contains Simulation<number> subdirectories: {directory}'
        )"""

    if limit is None:
        phase_df = pd.read_csv(phase_path, dtype=str)
        if phase_df.empty:
            raise ValueError('Phase boundary file is empty')
        if len(phase_df) % 2 != 0:
            raise ValueError(
                'Phase boundary file should contain an even number of rows, '
                'with each pair bounding one phase change'
            )
    else:
        if limit <= 0:
            raise ValueError('limit must be a positive integer')

        with open(phase_path, 'r', encoding='utf-8', errors='ignore') as handle:
            total_data_rows = sum(1 for _ in handle) - 1
        if total_data_rows <= 0:
            raise ValueError('Phase boundary file is empty')
        if total_data_rows % 2 != 0:
            raise ValueError(
                'Phase boundary file should contain an even number of rows, '
                'with each pair bounding one phase change'
            )

        total_pairs = total_data_rows // 2
        if limit > total_pairs:
            raise ValueError(
                f'limit ({limit}) exceeds the number of available phase-change '
                f'pairs ({total_pairs})'
            )

        pair_idx = random.sample(range(total_pairs), limit)
        keep_rows = {row for idx in pair_idx for row in (idx * 2, idx * 2 + 1)}

        phase_df = pd.read_csv(
            phase_path,
            dtype=str,
            skiprows=lambda line_no: line_no != 0 and (line_no - 1) not in keep_rows,
        )

    bulk_element_cols = [
        column for column in phase_df.columns
        if str(column).endswith('(Bulk_comp_elements)')
    ]
    if len(bulk_element_cols) == 0:
        raise ValueError(
            'Phase boundary file does not contain Bulk_comp_elements columns'
        )

    if 'P(GPa)(System_main)' not in phase_df.columns or 'T(K)(System_main)' not in phase_df.columns:
        raise ValueError("Phase boundary file must contain 'P(GPa)(System_main)' and 'T(K)(System_main)' columns")

    directory.mkdir(parents=True, exist_ok=True)

    batch_dir = _next_available_batch_dir(directory, 1)
    batch_dir.mkdir(parents=True, exist_ok=False)
    simulations_in_batch = 0
    batch_number = int(batch_dir.name.removeprefix('Batch'))

    for sim_idx in range(len(phase_df) // 2):
        window_df = phase_df.iloc[sim_idx * 2:(sim_idx + 1) * 2].reset_index(drop=True)
        reference_row = window_df.iloc[0]
        comparison_row = window_df.iloc[1]

        p_values = pd.to_numeric(window_df['P(GPa)(System_main)'], errors='coerce')
        t_values = pd.to_numeric(window_df['T(K)(System_main)'], errors='coerce')
        if p_values.isna().any() or t_values.isna().any():
            raise ValueError(
                f'Non-numeric P or T in phase boundary rows for Simulation{sim_idx + 1}'
            )

        p_min = float(p_values.min())
        p_max = float(p_values.max())
        t_min = float(t_values.min())
        t_max = float(t_values.max())

        p_steps = 0 if np.isclose(p_min, p_max) else 5
        t_steps = 0 if np.isclose(t_min, t_max) else 1
        t_steps += (t_max-t_min)//25

        run_code = [
            p_min,
            p_max,
            int(p_steps),
            t_min,
            t_max,
            int(t_steps),
            0,
            0,
            0,
            0,
            0,
        ]

        if simulations_in_batch >= 1000:
            batch_number += 1
            batch_dir = _next_available_batch_dir(directory, batch_number)
            batch_dir.mkdir(parents=True, exist_ok=False)
            simulations_in_batch = 0

        sim_dir = batch_dir / f'Simulation{simulations_in_batch + 1}'
        sim_dir.mkdir(parents=True, exist_ok=True)
        simulations_in_batch += 1
        # Define Appropriate Template
        if CONTROL_DIR.is_file() or CONTROL_DIR.name in ('shallowHeFESTo', 'deepHeFESTo'):
            control_template_path = CONTROL_DIR
        elif float(window_df['P(GPa)(System_main)'].min()) < 23:
            control_template_path = CONTROL_DIR / 'shallowHeFESTo'
        else:
            control_template_path = CONTROL_DIR / 'deepHeFESTo'

        control_path = sim_dir / 'control'
        shutil.copy2(control_template_path, control_path)

        with open(control_path, 'r', encoding='utf-8', errors='ignore') as handle:
            control_lines = [line.rstrip('\n') for line in handle]

        element_values: Dict[str, float] = {}
        for column in bulk_element_cols:
            element_name = _normalize_element_label(column.split('(', 1)[0])
            element_values[element_name] = float(reference_row[column])

            reference_value = str(reference_row[column]).strip()
            comparison_value = str(comparison_row[column]).strip()
            if reference_value != comparison_value:
                try:
                    if not np.isclose(float(reference_value), float(comparison_value)):
                        raise ValueError(
                            f'Bulk element mismatch for {element_name} in '
                            f'Simulation{sim_idx + 1}: {reference_value} != {comparison_value}'
                        )
                except ValueError as exc:
                    raise ValueError(
                        f'Bulk element mismatch for {element_name} in '
                        f'Simulation{sim_idx + 1}: {reference_value} != {comparison_value}'
                    ) from exc

        control_lines = _build_control_lines(
            template_lines=control_lines,
            element_values=element_values,
            run_code=run_code,
        )

        with open(control_path, 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(control_lines) + '\n')

        boundary_copy_path = sim_dir / phase_path.name
        #window_df.to_csv(boundary_copy_path, index=False)


def prepare_HeFESTo_single_composition_directory(
    directory: Path,
    oxide_wt: Dict[str, float],
    p_min: float,
    p_max: float,
    p_steps: int,
    t_min: float,
    t_max: float,
    t_steps: int,
    CONTROL_DIR: Path,
    fe3_fet: float = 0.0,
    target_total_moles: float = 24.0,
) -> None:
    """
    Prepare a single directory for a HeFESTo run over one bulk composition and a P-T grid.

    Modeled on ``prepare_HeFESTo_tree_from_phase_changes``, but takes a single
    caller-supplied composition and pressure/temperature range directly
    (instead of pairs of rows from a phase-boundary CSV), and writes the
    ``control`` file straight into ``directory`` rather than a Batch/SimulationN
    tree.

    Parameters
    ----------
    directory : Path
        Destination directory for the HeFESTo run. Created if missing; an
        existing ``control`` file inside it is overwritten.
    oxide_wt : dict
        Bulk composition in wt% oxides with keys ``SiO2``, ``MgO``, ``FeO``
        (total iron as FeO), ``CaO``, ``Al2O3``, ``Na2O``, ``Cr2O3``. Iron is
        speciated into FeO/Fe2O3 using ``fe3_fet`` before conversion to
        element moles, matching the 8-oxide HeFESTo system used elsewhere in
        this module.
    p_min, p_max : float
        Pressure range in GPa.
    p_steps : int
        Number of pressure steps in the grid.
    t_min, t_max : float
        Temperature range in K.
    t_steps : int
        Number of temperature steps in the grid.
    CONTROL_DIR : Path
        Path to a control template file, or a directory containing
        ``shallowHeFESTo``/``deepHeFESTo`` templates (selected by ``p_min``,
        using the same 23 GPa cutoff as ``prepare_HeFESTo_tree_from_phase_changes``).
    fe3_fet : float, optional
        Target Fe3+/Fetotal molar ratio used to speciate the total iron in
        ``oxide_wt['FeO']``. Defaults to 0.0 (all iron as Fe2+).
    target_total_moles : float, optional
        Total element moles the composition is rescaled to (default 24.0,
        matching the other ``prepare_HeFESTo_tree_*`` functions).
    """
    directory = Path(directory)
    CONTROL_DIR = Path(CONTROL_DIR)

    if not CONTROL_DIR.exists():
        raise FileNotFoundError(f'Control template not found: {CONTROL_DIR}')

    if CONTROL_DIR.is_file() or CONTROL_DIR.name in ('shallowHeFESTo', 'deepHeFESTo'):
        control_template_path = CONTROL_DIR
    elif p_min < 23:
        control_template_path = CONTROL_DIR / 'shallowHeFESTo'
    else:
        control_template_path = CONTROL_DIR / 'deepHeFESTo'

    if not control_template_path.exists():
        raise FileNotFoundError(f'Control template not found: {control_template_path}')

    directory.mkdir(parents=True, exist_ok=True)

    control_path = directory / 'control'
    shutil.copy2(control_template_path, control_path)

    with open(control_path, 'r', encoding='utf-8', errors='ignore') as handle:
        template_lines = [line.rstrip('\n') for line in handle]

    base_oxide_wt = dict(oxide_wt)
    base_oxide_wt['Fe_total_moles'] = base_oxide_wt['FeO'] / OXIDE_MOLAR_MASSES['FeO']

    speciated_wt = _speciate_iron_and_normalize(base_oxide_wt, fe3_fet)
    element_moles = _oxide_wt_to_element_moles(speciated_wt)
    element_moles = _normalize_total_moles(element_moles, target_total_moles)

    run_code = [
        float(p_min), float(p_max), int(p_steps),
        float(t_min), float(t_max), int(t_steps),
        0, 0, 0, 0, 0,
    ]

    control_lines = _build_control_lines(
        template_lines=template_lines,
        element_values=element_moles,
        run_code=run_code,
    )

    with open(control_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(control_lines) + '\n')


def _safe_read_ws_table(path: str, skiprows: int = 0) -> pd.DataFrame:
    return pd.read_csv(path, sep=r'\s+', engine='python', skiprows=skiprows)


def load_fort99_component_moles_and_labels(sim_dir: str, indexer) -> Tuple[np.ndarray, np.ndarray]:
    """Load fort.99 components into an extensive component-moles array and
    compute intensive (VC) and molar (P) phase labels using an ml_indexer.

    Parameters
    ----------
    sim_dir : str
        Directory containing a `fort.99` file.
    indexer : object
        ml_indexer providing `label_names` (component labels) and
        `phaseToCompMap` (2D array-like with shape [n_phases, n_components]).

    Returns
    -------
    VC : np.ndarray
        Intensive phase labels with shape (n_rows, n_phases). Contains component proportions that sum to 1 across each phase
    P : np.ndarray
        Molar phase labels (phase moles) with shape (n_rows, n_phases).
    """
    sim_dir = Path(sim_dir)
    fort99_path = sim_dir / 'fort.99'
    if not fort99_path.exists() or not fort99_path.is_file():
        raise FileNotFoundError(f'fort.99 not found in: {sim_dir}')

    comp_df = _safe_read_ws_table(str(fort99_path), skiprows=0)
    if comp_df.shape[1] < 6:
        raise ValueError('fort.99 has insufficient columns to parse components')

    nrows = len(comp_df)
    component_count = len(getattr(indexer, 'label_names', []))
    if component_count == 0:
        raise ValueError('Indexer must expose non-empty `label_names`')

    # Prepare an (nrows, n_components) array filled with zeros
    component_moles = np.zeros((nrows, component_count), dtype=float)

    # fort.99 component columns convention: skip first 3 and last 2 columns
    component_cols = list(comp_df.columns)[3:-2]
    for comp_abbr in component_cols:
        comp_abbr_str = str(comp_abbr).strip()
        comp_name = _resolve_component_name_from_abbr(comp_abbr_str)
        try:
            comp_idx = list(indexer.label_names).index(comp_name)
        except ValueError:
            # component not present in indexer; skip it
            continue
        values = pd.to_numeric(comp_df[comp_abbr], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        component_moles[:, comp_idx] = values

    # Validate phaseToCompMap and compute phase molar amounts P
    p_to_c = np.asarray(getattr(indexer, 'phaseToCompMap', None), dtype=float)
    if p_to_c is None or p_to_c.ndim != 2 or p_to_c.shape[1] != component_count:
        raise ValueError('Indexer must expose `phaseToCompMap` with shape [n_phases, n_components]')

    # P: molar phase amounts per row (nrows, n_phases)
    P = component_moles @ p_to_c.T

    # VC: intensive labels (normalized component fractions of variable composition phases per row)
    phaseComponentMoles = component_moles[:, None, :] * p_to_c[None, :, :]  # (nrows, n_phases, n_components)

    comp_sums = np.sum(phaseComponentMoles, axis=2, keepdims=True)
    print(comp_sums)
    # avoid divide-by-zero: if sum==0 leave row as zeros
    with np.errstate(invalid='ignore', divide='ignore'):
        VC =np.einsum('bpc,pc->bc', np.divide(phaseComponentMoles, comp_sums, where=(comp_sums > 0)), p_to_c) @ indexer.variedToAllComp.T # (nrows, n_variable_comps)
    print(f"Shape of VC: {VC.shape}, should be {len(indexer.compositionally_variable_subset)} Shape of P: {P.shape}")
    print(VC)
    return VC, P


def load_fort99_componentMoles(sim_dir: str, indexer) -> np.ndarray:
    """Load fort.99 components into an extensive component-moles array.

    Parses the whitespace-delimited fort.99 file, resolves each column's
    abbreviation to a canonical component name via the indexer, and returns
    the component moles as a dense array aligned to ``indexer.label_names``.
    Components not present in the indexer are silently skipped.

    This is the lower-level companion to ``load_fort99_component_moles_and_labels``:
    it returns only the raw mole array without computing phase-label projections.
    The result can be passed directly to
    ``HeFESToEmulatorCPU.get_property_hefesto_vectorized_from_assemblage``.

    Parameters
    ----------
    sim_dir : str
        Directory containing a ``fort.99`` file.
    indexer : object
        ml_indexer providing ``label_names`` (list of component names) used to
        align fort.99 columns to the correct output positions.

    Returns
    -------
    component_moles : np.ndarray
        Array of shape ``(n_rows, n_components)`` where ``n_components`` is
        ``len(indexer.label_names)``. Each row corresponds to one P–T step
        in the simulation; each column is the molar abundance of the
        corresponding component.
    """
    sim_dir = Path(sim_dir)
    fort99_path = sim_dir / 'fort.99'
    if not fort99_path.exists() or not fort99_path.is_file():
        raise FileNotFoundError(f'fort.99 not found in: {sim_dir}')

    comp_df = _safe_read_ws_table(str(fort99_path), skiprows=0)
    if comp_df.shape[1] < 6:
        raise ValueError('fort.99 has insufficient columns to parse components')

    nrows = len(comp_df)
    component_count = len(getattr(indexer, 'label_names', []))
    if component_count == 0:
        raise ValueError('Indexer must expose non-empty `label_names`')

    # Prepare an (nrows, n_components) array filled with zeros
    component_moles = np.zeros((nrows, component_count), dtype=float)

    # fort.99 component columns convention: skip first 3 and last 2 columns
    component_cols = list(comp_df.columns)[3:-2]
    for comp_abbr in component_cols:
        comp_abbr_str = str(comp_abbr).strip()
        comp_name = _resolve_component_name_from_abbr(comp_abbr_str)
        try:
            comp_idx = list(indexer.label_names).index(comp_name)
        except ValueError:
            # component not present in indexer; skip it
            continue
        values = pd.to_numeric(comp_df[comp_abbr], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        component_moles[:, comp_idx] = values
    return component_moles



def _extract_sim_id(sim_name: str) -> Optional[int]:
    match = _SIMULATION_DIR_PATTERN.match(sim_name)
    if match is None:
        return None

    simulation_number = match.group(1) or match.group(2)
    if simulation_number is None:
        return None

    return int(simulation_number)


def _list_simulation_dirs(workspace_dir: str) -> List[Tuple[int, str]]:
    sims: List[Tuple[int, str]] = []
    for name in os.listdir(workspace_dir):
        full = os.path.join(workspace_dir, name)
        if not os.path.isdir(full):
            continue
        sim_id = _extract_sim_id(name)
        if sim_id is None:
            continue
        sims.append((sim_id, full))
    sims.sort(key=lambda x: x[0])
    return sims


def _compute_bulk_from_elements(element_moles: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float], float]:
    required = ['Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr', 'O']
    missing = [el for el in required if el not in element_moles]
    if missing:
        raise ValueError(f'Missing required element(s) in control oxides block: {missing}')

    n_si = float(element_moles['Si'])
    n_mg = float(element_moles['Mg'])
    n_fe = float(element_moles['Fe'])
    n_ca = float(element_moles['Ca'])
    n_al = float(element_moles['Al'])
    n_na = float(element_moles['Na'])
    n_cr = float(element_moles['Cr'])
    n_o = float(element_moles['O'])

    # Oxygen tied up in non-iron oxides.
    o_non_fe = 2.0 * n_si + n_mg + n_ca + 1.5 * n_al + 0.5 * n_na + 1.5 * n_cr
    o_for_fe = n_o - o_non_fe

    # Fe2O3 moles are the oxygen excess relative to all-FeO partitioning.
    n_fe2o3 = max(0.0, o_for_fe - n_fe)
    n_feo = max(0.0, n_fe - 2.0 * n_fe2o3)

    if n_feo + 2.0 * n_fe2o3 > n_fe + 1e-9:
        n_fe2o3 = 0.0
        n_feo = n_fe

    oxide_moles = {
        'SiO2': n_si,
        'MgO': n_mg,
        'FeO': n_feo,
        'Fe2O3': n_fe2o3,
        'CaO': n_ca,
        'Al2O3': n_al / 2.0,
        'Na2O': n_na / 2.0,
        'Cr2O3': n_cr / 2.0,
    }

    oxide_masses = {ox: moles * get_oxide_molar_mass(ox) for ox, moles in oxide_moles.items()}
    system_mass = float(np.sum(list(oxide_masses.values())))
    if system_mass <= 0:
        raise ValueError('Computed non-positive system mass from control oxides block')

    bulk_comp_wt = {ox: 100.0 * mass / system_mass for ox, mass in oxide_masses.items()}
    bulk_elements = {el: float(element_moles[el]) for el in required}
    return bulk_comp_wt, bulk_elements, system_mass


def _safe_assign(
    table: np.ndarray,
    indexer,
    phase: str,
    component: str,
    values: np.ndarray,
    add: bool = False,
) -> None:
    phase_map = indexer.MELTS_indices.get(phase, None)
    if phase_map is None:
        return
    col_idx = phase_map.get(component, None)
    if col_idx is None:
        return
    if add:
        table[:, col_idx] += values
    else:
        table[:, col_idx] = values


def _extract_phase_series_from_prefixed_columns(
    df: pd.DataFrame,
    prefix: str,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for col in df.columns:
        col_str = str(col)
        if not col_str.startswith(prefix):
            continue
        abbr = col_str[len(prefix):].strip()
        phase_name = _resolve_phase_name_from_abbr(abbr)
        values = pd.to_numeric(df[col], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        out.setdefault(phase_name, np.zeros_like(values))
        out[phase_name] += values
    return out


def _write_block_to_csv(dataname: str, headers: List[str], block: np.ndarray) -> None:
    if not os.path.exists(dataname):
        pd.DataFrame(columns=headers).to_csv(dataname, index=False)
    pd.DataFrame(block, columns=headers).to_csv(dataname, mode='a', header=False, index=False)


def _ensure_existing_csv_headers_match(dataname: str, expected_headers: List[str]) -> None:
    if not os.path.exists(dataname):
        return

    existing_headers = list(pd.read_csv(dataname, nrows=0).columns)
    if existing_headers != list(expected_headers):
        raise ValueError(
            'Existing output CSV headers do not match indexer.database_headers. '
            'Refuse to append due to schema mismatch.'
        )


def _checkpoint_path_for_csv(dataname: str) -> str:
    csv_path = Path(dataname)
    return str(csv_path.with_name(f'{csv_path.name}.checkpoint.txt'))


def _read_checkpoint_sim_id(checkpoint_path: str) -> Optional[int]:
    if not os.path.exists(checkpoint_path):
        return None

    raw_value = Path(checkpoint_path).read_text(encoding='utf-8').strip()
    if raw_value == '':
        raise ValueError(f'Checkpoint file is empty: {checkpoint_path}')

    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f'Invalid checkpoint value in {checkpoint_path}: {raw_value!r}') from exc


def _write_checkpoint_sim_id(checkpoint_path: str, sim_id: int) -> None:
    Path(checkpoint_path).write_text(f'{int(sim_id)}\n', encoding='utf-8')


def _remove_file_if_exists(path_value: str) -> None:
    if os.path.exists(path_value):
        os.remove(path_value)


def _flush_import_buffers(
    dataname: str,
    phase_change_dataname: Optional[str],
    headers: List[str],
    data_blocks: List[np.ndarray],
    phase_change_blocks: List[np.ndarray],
) -> None:
    data_block = None
    if len(data_blocks) > 0:
        data_block = np.concatenate(data_blocks, axis=0) if len(data_blocks) > 1 else data_blocks[0]

    phase_change_block = None
    if phase_change_dataname is not None and len(phase_change_blocks) > 0:
        phase_change_block = (
            np.concatenate(phase_change_blocks, axis=0)
            if len(phase_change_blocks) > 1
            else phase_change_blocks[0]
        )

    if data_block is not None:
        _write_block_to_csv(dataname, headers, data_block)

    if phase_change_block is not None:
        _write_block_to_csv(phase_change_dataname, headers, phase_change_block)

    data_blocks.clear()
    phase_change_blocks.clear()


def import_HeFESTo_components(
    workspace_dir,
    indexer,
    dataname: str = 'DefaultHeFESTostorage.csv',
    phase_change_dataname: Optional[str] = None,
):
    """
    Parse HeFESTo SimulationN directories into one CSV table.

    File usage per simulation:
    - control: bulk element inventory (and phase/component abbreviation blocks)
    - fort.56: System_main intensive/thermo attributes
    - fort.61: phase densities (rh* columns)
    - fort.68: phase volumes (vol* columns)
    - fort.99: phase-component molar abundances

    Parameters
    ----------
    workspace_dir : str
        Directory containing SimulationN folders.
    indexer : DatasetIndexer
        DatasetIndexer with MELTS_indices/database_headers defining output schema.
    dataname : str, default='DefaultHeFESTostorage.csv'
        Output CSV path for all parsed rows.
    phase_change_dataname : str or None, default=None
        Optional output CSV path for rows that bound fort.99 component
        zero/non-zero transitions. A boundary includes the transition row
        and its immediately previous row, using tolerance 1e-6.

    A checkpoint text file is written next to ``dataname`` after each flushed
    batch of 128 simulations. If the import is restarted and that checkpoint
    remains, the function resumes from the next simulation ID. The checkpoint
    file is removed when the import completes successfully.

    Returns
    -------
    tuple[list[int], list[int], list[int]]
        ``(passed_ids, malformed_ids, empty_ids)`` where each list contains
        unique Simulation IDs classified during parsing.
    """
    passed_ids: List[int] = []
    malformed_ids: List[int] = []
    empty_ids: List[int] = []
    reverse_component_phase_map = _build_reverse_component_phase_map()
    sim_dirs = _list_simulation_dirs(workspace_dir)
    _ensure_existing_csv_headers_match(dataname, indexer.database_headers)
    if phase_change_dataname is not None:
        _ensure_existing_csv_headers_match(phase_change_dataname, indexer.database_headers)

    checkpoint_path = _checkpoint_path_for_csv(dataname)
    resume_after_sim_id = _read_checkpoint_sim_id(checkpoint_path)

    if not os.path.exists(dataname):
        pd.DataFrame(columns=indexer.database_headers).to_csv(dataname, index=False)
    if phase_change_dataname is not None and not os.path.exists(phase_change_dataname):
        pd.DataFrame(columns=indexer.database_headers).to_csv(phase_change_dataname, index=False)

    data_blocks: List[np.ndarray] = []
    phase_change_blocks: List[np.ndarray] = []
    processed_since_flush = 0
    last_flushed_sim_id: Optional[int] = resume_after_sim_id

    for sim_id, sim_dir in sim_dirs:
        if resume_after_sim_id is not None and sim_id <= resume_after_sim_id:
            continue

        control_path = os.path.join(sim_dir, 'control')
        fort56_path = os.path.join(sim_dir, 'fort.56')
        fort61_path = os.path.join(sim_dir, 'fort.61')
        fort68_path = os.path.join(sim_dir, 'fort.68')
        fort99_path = os.path.join(sim_dir, 'fort.99')

        sim_contents = [
            entry
            for entry in os.listdir(sim_dir)
            if os.path.isfile(os.path.join(sim_dir, entry))
        ]
        if len(sim_contents) == 0:
            empty_ids.append(sim_id)
            continue

        if not os.path.exists(control_path):
            malformed_ids.append(sim_id)
            continue

        missing_outputs = [
            req_path
            for req_path in [fort56_path, fort61_path, fort68_path, fort99_path]
            if not os.path.exists(req_path)
        ]
        if len(missing_outputs) > 0:
            malformed_ids.append(sim_id)
            continue

        try:
            element_moles, control_component_to_phase_abbr = _parse_control_file(control_path)
            bulk_comp_wt, bulk_elements, system_mass = _compute_bulk_from_elements(element_moles)

            sys_df = _parse_fort56(fort56_path)
            rho_df = _safe_read_ws_table(fort61_path, skiprows=0)
            vol_df = _safe_read_ws_table(fort68_path, skiprows=0)
            comp_df = _safe_read_ws_table(fort99_path, skiprows=0)

            nrows = min(len(sys_df), len(rho_df), len(vol_df), len(comp_df))
            if nrows <= 0:
                raise ValueError('No rows found across required HeFESTo tables')

            sys_df = sys_df.iloc[:nrows].reset_index(drop=True)
            rho_df = rho_df.iloc[:nrows].reset_index(drop=True)
            vol_df = vol_df.iloc[:nrows].reset_index(drop=True)
            comp_df = comp_df.iloc[:nrows].reset_index(drop=True)

            out = np.zeros((nrows, indexer.get_max_index() + 1), dtype=float)

            # System_main from fort.56
            _safe_assign(out, indexer, 'System_main', 'P(GPa)', pd.to_numeric(sys_df.get('P(GPa)'), errors='coerce').fillna(0.0).to_numpy(dtype=float))
            _safe_assign(out, indexer, 'System_main', 'T(K)', pd.to_numeric(sys_df.get('T(K)'), errors='coerce').fillna(0.0).to_numpy(dtype=float))
            _safe_assign(out, indexer, 'System_main', 'rho(g/cm^3)', pd.to_numeric(sys_df.get('rho(g/cm^3)'), errors='coerce').fillna(0.0).to_numpy(dtype=float))
            _safe_assign(out, indexer, 'System_main', 'mass (gm)', np.full(nrows, system_mass, dtype=float))
            _safe_assign(out, indexer, 'System_main', 'VS(km/s)', pd.to_numeric(sys_df.get('VS(km/s)'), errors='coerce').fillna(0.0).to_numpy(dtype=float))
            _safe_assign(out, indexer, 'System_main', 'VP(km/s)', pd.to_numeric(sys_df.get('VP(km/s)'), errors='coerce').fillna(0.0).to_numpy(dtype=float))
            _safe_assign(out, indexer, 'System_main', 'H(kJ/g)', pd.to_numeric(sys_df.get('H(kJ/g)'), errors='coerce').fillna(0.0).to_numpy(dtype=float))
            _safe_assign(out, indexer, 'System_main', 'cp(J/g/K)', pd.to_numeric(sys_df.get('cp(J/g/K)'), errors='coerce').fillna(0.0).to_numpy(dtype=float))
            _safe_assign(out, indexer, 'System_main', 'S(J/g/K)', pd.to_numeric(sys_df.get('S(J/g/K)'), errors='coerce').fillna(0.0).to_numpy(dtype=float))
            _safe_assign(out, indexer, 'System_main', 'KS(GPa)', pd.to_numeric(sys_df.get('KS(GPa)'), errors='coerce').fillna(0.0).to_numpy(dtype=float))
            _safe_assign(out, indexer, 'System_main', 'alpha(1e5_K^-1)', pd.to_numeric(sys_df.get('alpha(1e5_K^-1)'), errors='coerce').fillna(0.0).to_numpy(dtype=float))

            # Bulk_comp and Bulk_comp_elements from control-derived chemistry.
            for oxide, wt in bulk_comp_wt.items():
                _safe_assign(out, indexer, 'Bulk_comp', oxide, np.full(nrows, wt, dtype=float))

            for element, value in bulk_elements.items():
                _safe_assign(out, indexer, 'Bulk_comp_elements', element, np.full(nrows, value, dtype=float))

            # Phase density and volume tables.
            rho_by_phase = _extract_phase_series_from_prefixed_columns(rho_df, 'rh')
            vol_by_phase = _extract_phase_series_from_prefixed_columns(vol_df, 'vol')

            for phase_name, rho_values in rho_by_phase.items():
                _safe_assign(out, indexer, phase_name, 'rho (gm/cc)', rho_values)

            for phase_name, vol_values in vol_by_phase.items():
                _safe_assign(out, indexer, phase_name, 'V (cc)', vol_values)

            # Rebuild phase mass from rho*V where available.
            for phase_name in indexer.MELTS_indices.keys():
                if phase_name in {'System_main', 'Bulk_comp', 'Bulk_comp_elements'}:
                    continue
                phase_map = indexer.MELTS_indices[phase_name]
                mass_idx = phase_map.get('mass (gm)')
                vol_idx = phase_map.get('V (cc)')
                rho_idx = phase_map.get('rho (gm/cc)')
                if mass_idx is None or vol_idx is None or rho_idx is None:
                    continue
                out[:, mass_idx] = out[:, rho_idx] * out[:, vol_idx]

            # fort.99 components: skip first 3 and last 2 columns.
            if comp_df.shape[1] < 6:
                raise ValueError('fort.99 has insufficient columns to parse components')

            component_cols = list(comp_df.columns)[3:-2]
            transition_tol = 1e-6
            for comp_abbr in component_cols:
                comp_abbr_str = str(comp_abbr).strip()
                component_name = _resolve_component_name_from_abbr(comp_abbr_str)
                phase_name = _resolve_component_phase(
                    component_abbr=comp_abbr_str,
                    component_name=component_name,
                    reverse_component_phase_map=reverse_component_phase_map,
                    control_component_to_phase_abbr=control_component_to_phase_abbr,
                )
                if phase_name is None:
                    continue
                values = pd.to_numeric(comp_df[comp_abbr], errors='coerce').fillna(0.0).to_numpy(dtype=float)

                _safe_assign(out, indexer, phase_name, component_name, values)

            # Compute total phase moles from component-space abundances.
            for phase_name, c_indices in getattr(indexer, 'label_indices', {}).items():
                phase_map = indexer.MELTS_indices.get(phase_name, None)
                if phase_map is None:
                    continue

                total_values = np.zeros(nrows, dtype=float)
                for c_idx in np.atleast_1d(c_indices):
                    c_idx_int = int(c_idx)
                    if c_idx_int < 0 or c_idx_int >= len(indexer.label_names):
                        continue
                    component_label = str(indexer.label_names[c_idx_int])
                    out_col_idx = phase_map.get(component_label, None)
                    if out_col_idx is None:
                        continue
                    total_values += out[:, out_col_idx]

                # Populate in-schema names when present.
                _safe_assign(out, indexer, phase_name, 'total (moles)', total_values)
                _safe_assign(out, indexer, phase_name, 'moles', total_values)
                _safe_assign(out, indexer, phase_name, 'phase moles', total_values)
                _safe_assign(out, indexer, phase_name, 'total moles', total_values)

            phase_change_boundary_rows: List[int] = []
            if phase_change_dataname is not None and nrows > 1:
                if not hasattr(indexer, 'phaseToCompMap'):
                    raise ValueError('Indexer must expose phaseToCompMap for phase-change export')
                if not hasattr(indexer, 'label_indices') or not hasattr(indexer, 'label_names'):
                    raise ValueError('Indexer must expose label indices/names for phase-change export')

                component_count = len(indexer.label_names)
                p_to_c = np.asarray(indexer.phaseToCompMap, dtype=float)
                if p_to_c.ndim != 2:
                    raise ValueError('Indexer phaseToCompMap must be a 2D array')
                if p_to_c.shape[1] != component_count:
                    raise ValueError(
                        'Indexer phaseToCompMap/component label dimension mismatch: '
                        f'{p_to_c.shape[1]} != {component_count}'
                    )

                # Build C-space component molar array from populated out, then project C -> P.
                component_moles = np.zeros((nrows, component_count), dtype=float)
                for phase_name, c_indices in indexer.label_indices.items():
                    phase_map = indexer.MELTS_indices.get(phase_name, None)
                    if phase_map is None:
                        continue
                    for c_idx in np.atleast_1d(c_indices):
                        c_idx_int = int(c_idx)
                        component_label = str(indexer.label_names[c_idx_int])
                        out_col_idx = phase_map.get(component_label, None)
                        if out_col_idx is None:
                            continue
                        component_moles[:, c_idx_int] = out[:, out_col_idx]

                phase_moles = component_moles @ p_to_c.T
                transition_rows = []
                for phase_idx in range(phase_moles.shape[1]):
                    is_present = np.abs(phase_moles[:, phase_idx]) > transition_tol
                    transition_rows += (np.where(is_present[1:] != is_present[:-1])[0] + 1).tolist()
                for row_idx in sorted(set(transition_rows)):
                    phase_change_boundary_rows.append(int(row_idx - 1))
                    phase_change_boundary_rows.append(int(row_idx))

            data_blocks.append(out)
            if phase_change_dataname is not None and len(phase_change_boundary_rows) > 0:
                phase_change_blocks.append(out[phase_change_boundary_rows, :])

            processed_since_flush += 1
            last_flushed_sim_id = sim_id

            if processed_since_flush >= 16:
                _flush_import_buffers(
                    dataname=dataname,
                    phase_change_dataname=phase_change_dataname,
                    headers=indexer.database_headers,
                    data_blocks=data_blocks,
                    phase_change_blocks=phase_change_blocks,
                )
                _write_checkpoint_sim_id(checkpoint_path, sim_id)
                processed_since_flush = 0

            passed_ids.append(sim_id)

        except Exception as exc:
            print(f'Simulation{sim_id} FAILED at {sim_dir}: {type(exc).__name__}: {exc}')
            malformed_ids.append(sim_id)

    if len(data_blocks) > 0 or len(phase_change_blocks) > 0:
        _flush_import_buffers(
            dataname=dataname,
            phase_change_dataname=phase_change_dataname,
            headers=indexer.database_headers,
            data_blocks=data_blocks,
            phase_change_blocks=phase_change_blocks,
        )
        if last_flushed_sim_id is not None:
            _write_checkpoint_sim_id(checkpoint_path, last_flushed_sim_id)

    _remove_file_if_exists(checkpoint_path)

    def _unique_sorted(ids: List[int]) -> List[int]:
        return sorted(set(int(value) for value in ids))

    return _unique_sorted(passed_ids), _unique_sorted(malformed_ids), _unique_sorted(empty_ids)


def extract_bulk_properties_from_simulation_dir(
    sim_dir: str,
    include_phase_properties: bool = False,
    repeat: int = 1
) -> Dict[str, Any]:
    """
    Extract bulk thermodynamic properties from a single HeFESTo simulation.

    This function reads all usable bulk properties from fort.56 (all numeric
    columns), along with key tables from fort.61/68/99 for component-based
    diagnostics.
    
    Parameters
    ----------
    sim_dir : str
        Path to HeFESTo simulation directory (containing fort.*, control files)
    include_phase_properties : bool, default=False
        Reserved for compatibility.
    
    Returns
    -------
    dict
        Dictionary containing:
        - Core arrays: 'P(GPa)', 'T(K)', 'VS(km/s)', 'VP(km/s)'
        - One entry per numeric fort.56 bulk column
        - 'fort56_bulk': dict[str, np.ndarray] mirror of numeric fort.56 columns
        - 'fort56_dominant_phase': optional dominant phase labels
        - 'component_names', 'component_moles'
        - Optional physub outputs: 'bulk_properties', 'bulk_property_names'
    
    Raises
    ------
    FileNotFoundError
        If required fort files are missing
    ImportError
        If torch is required but unavailable
    """

    # Check required files
    required_files = ['control', 'fort.56', 'fort.61', 'fort.68', 'fort.99']
    for fname in required_files:
        fpath = os.path.join(sim_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f'Missing required file: {fpath}')

    # Parse control and basic data
    element_moles, control_component_to_phase_abbr = _parse_control_file(
        os.path.join(sim_dir, 'control')
    )

    # Read fort tables
    sys_df = _parse_fort56(os.path.join(sim_dir, 'fort.56'))
    rho_df = _safe_read_ws_table(os.path.join(sim_dir, 'fort.61'), skiprows=0)
    vol_df = _safe_read_ws_table(os.path.join(sim_dir, 'fort.68'), skiprows=0)
    comp_df = _safe_read_ws_table(os.path.join(sim_dir, 'fort.99'), skiprows=0)
    fort59_path = os.path.join(sim_dir, 'fort.59')
    fort59_df = None
    if os.path.exists(fort59_path):
        fort59_df = _safe_read_ws_table(fort59_path, skiprows=0)

    # Ensure all tables have same number of rows
    table_lengths = [len(sys_df), len(rho_df), len(vol_df), len(comp_df)]
    if fort59_df is not None:
        table_lengths.append(len(fort59_df))
    nrows = min(table_lengths)
    if nrows <= 0:
        raise ValueError('No rows found across required HeFESTo tables')

    sys_df = sys_df.iloc[:nrows].reset_index(drop=True)
    rho_df = rho_df.iloc[:nrows].reset_index(drop=True)
    vol_df = vol_df.iloc[:nrows].reset_index(drop=True)
    comp_df = comp_df.iloc[:nrows].reset_index(drop=True)
    if fort59_df is not None:
        fort59_df = fort59_df.iloc[:nrows].reset_index(drop=True)

    # Read every numeric bulk property available in fort.56.
    fort56_bulk: Dict[str, np.ndarray] = {}
    for col in sys_df.columns:
        numeric_series = pd.to_numeric(sys_df[col], errors='coerce')
        if numeric_series.notna().any():
            fort56_bulk[str(col)] = numeric_series.fillna(0.0).to_numpy(dtype=np.float32)

    # Optional fort.59 numeric bulk properties.
    fort59_bulk: Dict[str, np.ndarray] = {}
    if fort59_df is not None:
        for col in fort59_df.columns:
            numeric_series = pd.to_numeric(fort59_df[col], errors='coerce')
            if numeric_series.notna().any():
                fort59_bulk[str(col)] = numeric_series.fillna(0.0).to_numpy(dtype=np.float32)

    if 'P(GPa)' not in fort56_bulk or 'T(K)' not in fort56_bulk:
        raise ValueError("fort.56 is missing required numeric columns 'P(GPa)' and/or 'T(K)'")

    dominant_phase_col = None
    for col in sys_df.columns:
        if str(col).strip().lower() == 'dominant_phase':
            dominant_phase_col = col
            break

    # Build component moles array from fort.99
    reverse_component_phase_map = _build_reverse_component_phase_map()
    component_cols = list(comp_df.columns)[3:-2]  # Skip first 3 and last 2 columns

    # Create mapping of component abbreviations to component indices
    component_names = []
    component_moles_list = []

    for comp_abbr in component_cols:
        comp_abbr_str = str(comp_abbr).strip()
        component_name = _resolve_component_name_from_abbr(comp_abbr_str)
        phase_name = _resolve_component_phase(
            component_abbr=comp_abbr_str,
            component_name=component_name,
            reverse_component_phase_map=reverse_component_phase_map,
            control_component_to_phase_abbr=control_component_to_phase_abbr,
        )
        
        if component_name and phase_name:
            component_names.append(component_name)
            values = pd.to_numeric(comp_df[comp_abbr], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
            component_moles_list.append(values)

    if not component_moles_list:
        raise ValueError('No valid components found in fort.99')

    # Stack component moles into (nrows, n_components) array
    component_moles = np.stack(component_moles_list, axis=1)  # (nrows, n_comps)

    # Build result dictionary with backward-compatible keys.
    result = {
        'P(GPa)': fort56_bulk['P(GPa)'],
        'T(K)': fort56_bulk['T(K)'],
        'VS(km/s)': fort56_bulk.get('VS(km/s)', np.zeros(nrows, dtype=np.float32)),
        'VP(km/s)': fort56_bulk.get('VP(km/s)', np.zeros(nrows, dtype=np.float32)),
        'fort56_bulk': fort56_bulk,
        'fort59_bulk': fort59_bulk,
        'component_names': component_names,
        'component_moles': component_moles,
        #'bulk_properties': None,
        #'bulk_property_names': tuple(),
    }

    # Also expose each fort.56 numeric column directly for convenience.
    for col_name, values in fort56_bulk.items():
        result[col_name] = values

    # Expose fort.59 columns with a namespace prefix to avoid key collisions.
    for col_name, values in fort59_bulk.items():
        result[f'fort59::{col_name}'] = values

    if dominant_phase_col is not None:
        result['fort56_dominant_phase'] = sys_df[dominant_phase_col].astype(str).to_numpy()

    if repeat > 1: # Repeat functionality to test scalability
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                result[key] = np.repeat(value, repeat, axis=0)
            elif isinstance(value, pd.DataFrame):
                result[key] = pd.concat([value] * repeat, ignore_index=True)
            elif isinstance(value, pd.Series):
                result[key] = pd.concat([value] * repeat, ignore_index=True)
            elif isinstance(value, (list, tuple)):
                result[key] = value * repeat
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        result[key][k] = np.repeat(v, repeat, axis=0)
                    elif isinstance(v, pd.DataFrame):
                        result[key][k] = pd.concat([v] * repeat, ignore_index=True)
                    elif isinstance(v, pd.Series):
                        result[key][k] = pd.concat([v] * repeat, ignore_index=True)
                    elif isinstance(v, (list, tuple)):
                        result[key][k] = v * repeat
                    elif isinstance(v, dict):
                        result[key][k] = {kk: vv * repeat for kk, vv in v.items()}


    # Optional physub bulk matrix (kept for compatibility with existing callers).
    if torch is not None:
        try:
            molar_masses = np.array(
                [get_oxide_molar_mass(name) for name in component_names],
                dtype=np.float32,
            )
        except Exception:
            molar_masses = np.ones(len(component_names), dtype=np.float32) * 100.0

        device = torch.device('cpu')
        """attrs = {
            'molar_volume': torch.ones((nrows, len(component_names)), dtype=torch.float32, device=device),
            'bulk_modulus': torch.ones((nrows, len(component_names)), dtype=torch.float32, device=device) * 130.0,
            'shear_modulus': torch.ones((nrows, len(component_names)), dtype=torch.float32, device=device) * 80.0,
            'heat_capacity_p': torch.ones((nrows, len(component_names)), dtype=torch.float32, device=device) * 1.2,
            'heat_capacity_v': torch.ones((nrows, len(component_names)), dtype=torch.float32, device=device) * 0.9,
            'thermal_expansivity': torch.ones((nrows, len(component_names)), dtype=torch.float32, device=device) * 3.0e-5,
            'entropy': torch.ones((nrows, len(component_names)), dtype=torch.float32, device=device) * 5.0,
            'enthalpy': torch.ones((nrows, len(component_names)), dtype=torch.float32, device=device) * 10.0,
            'gibbs': torch.ones((nrows, len(component_names)), dtype=torch.float32, device=device) * 8.0,
        }"""

        #component_moles_t = torch.from_numpy(component_moles).to(device)
        #molar_mass_t = torch.from_numpy(molar_masses).to(device)
        #bulk_props, bulk_names = compute_physub_bulk_matrix(
        #    component_moles_t,
        #    molar_mass_t,
        #    attrs,
        #)
        #result['bulk_properties'] = bulk_props.detach().cpu().numpy()
        #result['bulk_property_names'] = bulk_names

    return result




