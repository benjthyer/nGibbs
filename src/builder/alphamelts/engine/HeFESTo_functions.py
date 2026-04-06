"""
HeFESTo workspace parser utilities.

Parses SimulationN folders and compiles HeFESTo outputs into one CSV table
matching a DatasetIndexer header layout.
"""

import os
import re
import shutil
import argparse
import subprocess
from time import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from src.nMELTS.utils.file_utils import save_fixed_width_table


import numpy as np
import pandas as pd

from nMELTS.config.constants import (
    COMPOSITIONAL_COMPONENTS_IN_PHASES_HEFESTO,
    HEFESTO_ABBREVIATION_TO_SHORT_NAMES,
    get_oxide_molar_mass,
    OXIDE_MOLAR_MASSES
)


PHASE_ABBREVIATION_OVERRIDES: Dict[str, str] = {
    'c2c': 'hp-clinopyroxene',
    'il': 'akimotoite',
    'pv': 'bridgmanite',
    'mw': 'ferropericlase',
    'fea': 'iron',
    'feg': 'iron',
    'fee': 'iron',
}

COMPONENT_ABBREVIATION_OVERRIDES: Dict[str, str] = {
    'mgil': 'mg-akimotoite',
    'feil': 'fe-akimotoite',
    'mgpv': 'mg-bridgmanite',
    'fepv': 'fe-bridgmanite',
    'alpv': 'al-bridgmanite',
    'hepv': 'ferric-bridgmanite',
    'hlpv': 'ferric-bridgmanite-ls',
    'fapv': 'ferric-al-bridgmanite',
    'crpv': 'cr-bridgmanite',
    'smag': 'magnetite',
    'fea': 'alpha-iron',
    'feg': 'gamma-iron',
    'fee': 'epsilon-iron',
    'mgc2': 'hp-clinoenstatite',
    'fec2': 'hp-clinoferrosilite',
}


EnsembleLocation = None




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


def forward_HeFESTo_single(
    hefesto_executable: str,
    simulation_id: int,
    workspace_dir: str,
) -> str:
    """
    Create and execute one HeFESTo simulation from a template directory.

    Parameters
    ----------
    hefesto_executable : str
        Absolute path to the HeFESTo executable.
    template_dir : str
        Absolute path to directory containing template files to copy.
    simulation_id : int
        Integer N used to create SimulationN.
    workspace_dir : str
        Absolute path where SimulationN will be created.

    Returns
    -------
    str
        Absolute path to the created SimulationN directory.
    """
    hefesto_executable = _validate_absolute_path(
        hefesto_executable,
        'hefesto_executable',
    )
    template_dir = _validate_absolute_path(template_dir, 'template_dir')
    workspace_dir = _validate_absolute_path(workspace_dir, 'workspace_dir')

    if not os.path.isfile(hefesto_executable):
        raise FileNotFoundError(
            f'HeFESTo executable path is not a file: {hefesto_executable}'
        )

    if not os.path.isdir(template_dir):
        raise FileNotFoundError(f'Template directory not found: {template_dir}')

    if not isinstance(simulation_id, int):
        raise TypeError('simulation_id must be an integer')
    if simulation_id < 0:
        raise ValueError('simulation_id must be >= 0')

    os.makedirs(workspace_dir, exist_ok=True)
    sim_dir = os.path.join(workspace_dir, f'Simulation{simulation_id}')

    #if os.path.exists(sim_dir):
        #shutil.rmtree(sim_dir)
    os.makedirs(sim_dir, exist_ok=False) # Don't overwrite existing simulation directory

    _copy_template_tree(template_dir, sim_dir)

    subprocess.run([hefesto_executable], cwd=sim_dir, check=True)
    return sim_dir


def forward_HeFESTo_single_from_cli(cli_args: Optional[List[str]] = None) -> str:
    """
    Run one HeFESTo simulation using command line switch arguments.

    Switches
    --------
    --hefesto-path : absolute path to HeFESTo executable
    --template-dir : absolute path to template directory
    --simulation-id : integer simulation id
    --workspace-dir : absolute path where SimulationN is created
    """
    parser = argparse.ArgumentParser(
        prog='forward_HeFESTo_single',
        description='Run a single HeFESTo simulation in SimulationN',
    )
    parser.add_argument(
        '--hefesto-path',
        required=True,
        help='Absolute path to HeFESTo executable',
    )
    parser.add_argument(
        '--template-dir',
        required=True,
        help='Absolute path to template directory',
    )
    parser.add_argument(
        '--simulation-id',
        required=True,
        type=int,
        help='Integer simulation id used for SimulationN',
    )
    parser.add_argument(
        '--workspace-dir',
        required=True,
        help='Absolute path where SimulationN will be created',
    )

    args = parser.parse_args(cli_args)
    return forward_HeFESTo_single(
        hefesto_executable=args.hefesto_path,
        template_dir=args.template_dir,
        simulation_id=args.simulation_id,
        workspace_dir=args.workspace_dir,
    )

def _build_oxide_wt_from_row(row: pd.Series) -> Dict[str, float]:
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
        wt[oxide] = float(value)

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
    fe_total_moles = float(oxide_wt['Fe_total_moles'])
    fe3_fet = float(np.clip(fe3_fet, 0.0, 0.1))

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
    total_moles = float(sum(element_moles.values()))
    if total_moles <= 0.0:
        raise ValueError('Cannot normalize element moles with non-positive total')

    scale = float(target_total_moles) / total_moles
    return {key: value * scale for key, value in element_moles.items()}


def get_S(T, Ca):
    """Get Entropy as a function of mantle potential temperature and Molar Ca (24 molar basis!)"""
    S = 0.732261764 - 0.0220453381 * Ca + 0.00147915640 * T - 0.0160980560 * Ca**2 - 2.16492530e-07 * T**2
    return S

def get_T(S, P):
    """Get Temperature as a function of Entropy, Pressure, and Mantle Potential Temperature"""
    T = 6029.33080527 * S - 787.844288209  * S**2 + 10.3970768783 * P - 0.0186356282034 * P**2 - 8587.97619083
    return T

def prepare_HeFESTo_tree_fulladiabat(directory: Path, GEOROC_DIR: Path, control_path: Path, N: int) -> None:
    """
    Prepare the directory structure and files needed to run HeFESTo simulations. This includes changing template files according to randomly chosen compositions and
    generating ad.in files for each simulation based on a range of random mantle potential temperatures and calcium contents,
    ----------
    directory : Path
        The directory where the HeFESTo input files will be generated.
    GEOROC_DIR : Path
        The directory containing the GEOROC data file.
    control_path : Path
        The path to the control file.
    N : int
        The number of simulations to run.
    """
    
    directory = Path(directory)
    GEOROC_DIR = Path(GEOROC_DIR)
    control_path = Path(control_path)
    if not control_path.exists() or not control_path.is_file():
        raise FileNotFoundError(f'Control template not found: {control_path}')
    directory.mkdir(parents=True, exist_ok=True)

    # Example: Generate ad.in files for a range of mantle potential temperatures and calcium contents
    Mps = 273 + 1200 + np.random.uniform(0, 1, N) * (1650 - 1200)  #  mantle potential temperatures
    target_total_moles = 24.0

    georoc_df = pd.read_csv(GEOROC_DIR)
    mgo_col = 'MgO'
    with open(control_path, 'r', encoding='utf-8', errors='ignore') as handle:
        template_lines = [line.rstrip('\n') for line in handle]
    mafic_df = georoc_df[pd.to_numeric(georoc_df[mgo_col], errors='coerce').fillna(0.0) > 20.0]
    subset = mafic_df.sample(n=N, replace=True) # Multiply every element value by a random number between 0.95 and 1.05
    reduced_N = int(N*(4/5))

    fe3_fet_grid = np.append(np.linspace(0.0, 0.05, reduced_N), np.linspace(0.05, 0.10, int(N - reduced_N)))
    element_keys = np.array(['Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr', 'O'])
    P0s = np.random.uniform(0, 1, size=N)
    run_code = [[float(P0), float(P0 + 139), 138, 0, 0, 0, -1, 0, 0, 0, 0] for P0 in P0s] # ad.in files made downstream
    element_rows: List[List[float]] = []
    wts: List[str] = []

    for sim_idx, (_, row) in enumerate(subset.iterrows()):
        sim_dir = directory / f'Simulation{sim_idx}'
        sim_dir.mkdir(parents=True, exist_ok=True)

        ratio = float(fe3_fet_grid[sim_idx])
        base_oxide_wt = _build_oxide_wt_from_row(row)
        speciated_wt = _speciate_iron_and_normalize(base_oxide_wt, ratio)
        wt_debug = ', '.join(f'{key}={value:.4f}' for key, value in speciated_wt.items())
        print(f'Sim {sim_idx} Fe3/FeT={ratio:.4f} -> {wt_debug}')
        wts.append(wt_debug)

        element_moles = _oxide_wt_to_element_moles(speciated_wt)
        element_moles = _normalize_total_moles(element_moles, target_total_moles)
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


def _safe_read_ws_table(path: str, skiprows: int = 0) -> pd.DataFrame:
    return pd.read_csv(path, sep=r'\s+', engine='python', skiprows=skiprows)


def _extract_sim_id(sim_name: str) -> Optional[int]:
    match = re.search(r'simulation(\d+)$', sim_name.lower())
    if match is None:
        return None
    return int(match.group(1))


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


def _resolve_phase_name_from_abbr(phase_abbr: str) -> str:
    if phase_abbr in PHASE_ABBREVIATION_OVERRIDES:
        return PHASE_ABBREVIATION_OVERRIDES[phase_abbr]
    return HEFESTO_ABBREVIATION_TO_SHORT_NAMES.get(phase_abbr, phase_abbr)


def _resolve_component_name_from_abbr(component_abbr: str) -> str:
    if component_abbr in COMPONENT_ABBREVIATION_OVERRIDES:
        return COMPONENT_ABBREVIATION_OVERRIDES[component_abbr]
    return HEFESTO_ABBREVIATION_TO_SHORT_NAMES.get(component_abbr, component_abbr)


def _build_reverse_component_phase_map() -> Dict[str, List[str]]:
    reverse_map: Dict[str, List[str]] = {}
    for phase_name, comp_list in COMPOSITIONAL_COMPONENTS_IN_PHASES_HEFESTO.items():
        if phase_name in {'System_main', 'Bulk_comp', 'Bulk_comp_elements'}:
            continue
        for component in comp_list:
            reverse_map.setdefault(component, []).append(phase_name)
    return reverse_map


def _parse_control_file(control_path: str) -> Tuple[Dict[str, float], Dict[str, str]]:
    with open(control_path, 'r', encoding='utf-8', errors='ignore') as handle:
        lines = [line.rstrip('\n') for line in handle]

    element_moles: Dict[str, float] = {}
    control_component_to_phase_abbr: Dict[str, str] = {}

    oxides_start = None
    for i, line in enumerate(lines):
        if line.strip().lower() == 'oxides':
            oxides_start = i + 1
            break
    if oxides_start is None:
        raise ValueError(f"No 'oxides' block found in {control_path}")

    for i in range(oxides_start, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            continue
        if stripped.lower().startswith('phase '):
            break
        if ',' in stripped:
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        symbol = parts[0]
        if symbol.upper() == 'O':
            symbol = 'O'
        try:
            value = float(parts[1])
        except ValueError:
            continue
        element_moles[symbol] = value

    active_phase_abbr: Optional[str] = None
    expect_flag_line = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith('phase '):
            fields = stripped.split()
            if len(fields) >= 2:
                active_phase_abbr = fields[1].strip()
                expect_flag_line = True
            continue
        if active_phase_abbr is None:
            continue
        if expect_flag_line:
            expect_flag_line = False
            continue
        component_abbr = stripped.split()[0]
        control_component_to_phase_abbr[component_abbr] = active_phase_abbr

    return element_moles, control_component_to_phase_abbr


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


def _resolve_component_phase(
    component_abbr: str,
    component_name: str,
    reverse_component_phase_map: Dict[str, List[str]],
    control_component_to_phase_abbr: Dict[str, str],
) -> Optional[str]:
    candidates = reverse_component_phase_map.get(component_name, [])

    if len(candidates) == 1:
        return candidates[0]

    if component_abbr in control_component_to_phase_abbr:
        phase_abbr = control_component_to_phase_abbr[component_abbr]
        phase_name = _resolve_phase_name_from_abbr(phase_abbr)
        if len(candidates) == 0:
            return phase_name
        if phase_name in candidates:
            return phase_name

    if len(candidates) > 0:
        return candidates[0]

    return None


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


def _parse_fort56(path: str) -> pd.DataFrame:
    return _safe_read_ws_table(path, skiprows=1)


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


def import_HeFESTo_components(workspace_dir, indexer, dataname='DefaultHeFESTostorage.csv'):
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
        Output CSV path.

    Returns
    -------
    np.ndarray
        Unique failing Simulation IDs.
    """
    faultIDs: List[int] = []
    reverse_component_phase_map = _build_reverse_component_phase_map()
    sim_dirs = _list_simulation_dirs(workspace_dir)
    _ensure_existing_csv_headers_match(dataname, indexer.database_headers)

    if not os.path.exists(dataname):
        pd.DataFrame(columns=indexer.database_headers).to_csv(dataname, index=False)

    for sim_id, sim_dir in sim_dirs:
        control_path = os.path.join(sim_dir, 'control')
        fort56_path = os.path.join(sim_dir, 'fort.56')
        fort61_path = os.path.join(sim_dir, 'fort.61')
        fort68_path = os.path.join(sim_dir, 'fort.68')
        fort99_path = os.path.join(sim_dir, 'fort.99')

        try:
            for req_path in [control_path, fort56_path, fort61_path, fort68_path, fort99_path]:
                if not os.path.exists(req_path):
                    raise FileNotFoundError(f'Missing required file: {req_path}')

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

            _write_block_to_csv(dataname, indexer.database_headers, out)

        except Exception as exc:
            print(f'Simulation{sim_id} FAILED at {sim_dir}: {type(exc).__name__}: {exc}')
            faultIDs.append(sim_id)

    return np.unique(np.asarray(faultIDs, dtype=int))


def make_PT_path(S, P, func, out_path = None):
    """Args:
    S: scalar or array of entropy values
    P: array of pressure in GPa

    Generates ad.in file that is read by HeFESTo to follow a PT path. Useful for loosely describing general adiabatic paths in the mantle.
    """
    AdIn = np.zeros((len(P), 3)) # Middle column unused
    AdIn[:, 0] = P
    AdIn[:, 2] = func(S=S, P=P)*np.random.uniform(0.95, 1.05, size=len(P)) # Add small random noise to T to avoid HeFESTo interpolation issues with perfectly linear PT paths.
    save_fixed_width_table(AdIn, out_path=out_path)

