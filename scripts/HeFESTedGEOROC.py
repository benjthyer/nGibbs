"""Note: This script is for generating training data and will not be exported to users in the release.
The linux virtual environment is loaded by calling: source ~/melts_env/venv/bin/activate"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'

# Add repo root and src to path so repo-local packages resolve from repo root.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.settings import internal_data_dir, internal_scratch_dir
#from builder.alphamelts.engine import alphamelts_functions # The essential ensemble MELTS functions
#from nMELTS.utils.string_utils import pull_number, random_char
from builder.alphamelts.engine import HeFESTo_functions as MELTER
from builder.indexer import generate_column_headers_hefesto, DatasetIndexer
import numpy as np
import pandas as pd
from nMELTS.config.constants import HEFESTO_ABBREVIATION_TO_SHORT_NAMES, OXIDE_MOLAR_MASSES
from scipy.stats import skewnorm

def skewed_peak(x: np.ndarray, center: float, width: float, skew: float = 0, height: float = 1.0) -> np.ndarray:
    """
    Generate a skewed peak over x centered at `center`.

    Parameters
    ----------
    x : np.ndarray
        1D array of independent variable values.
    center : flo
        Location of the peak's maximum (mode).
    width : float
        Controls the width (standard deviation-like). Must be > 0.
    skew : float
        Skewness: 0 = symmetric, >0 = right-skewed, <0 = left-skewed.
    height : float
        Maximum height of the peak (default 1.0).

    Returns
    -------
    np.ndarray
        Peak-shaped array y(x) of same shape as input x.
    """
    if width <= 0:
        raise ValueError("width must be > 0")

    # Standardize x relative to peak center and width
    z = (x - center) / width

    # Use skew-normal PDF
    y = skewnorm.pdf(z, skew)
    
    # Normalize to specified height
    return np.array(height * y / y.max())

depths = np.linspace(0,2900,(2900*5)+1)
dPDF = np.ones_like(depths)*0.15

dPDF += skewed_peak(depths, center=50, width=50, height=1)
dPDF += skewed_peak(depths, center=410, width=30, height=2)
dPDF += skewed_peak(depths, center=660, width=50, height=3)
dPDF += skewed_peak(depths, center=520, width=30, height=2)
dPDF += skewed_peak(depths, center=2890, width=100, height=3)
dPDF += skewed_peak(depths, center=900, width=100, height=1)

PrEMpressure = pd.read_csv(REPO_ROOT / 'config' / 'PrEMPressureDepth.csv').to_numpy()
Adiabat = pd.read_csv(REPO_ROOT / 'config' / 'AdiabatTempDepth.csv').to_numpy()


def get_PT(out_dir: Path):
    """
    NOT USED
    Generate a 2D array of n depth values from 0 to 2900 km,  favoring depths that contain phase changes.
    Save this to a file called ad.in at the output directory. 
    Columns: Pressure, Unused, Temperature

    Parameters
    ----------
    n : int
        Number of depth values to generate (default 50).

    Returns
    -------
    np.ndarray
        Array of depth values in km.
    """

    depth = np.random.choice(depths, size=n, replace=False, p=dPDF/dPDF.sum())

    pressure = np.interp(depth, PrEMpressure[:, 0], PrEMpressure[:, 1]) * 0.1

    #T = 

    return np.column_stack((depth, pressure))

def _find_first_column(df: pd.DataFrame, names: List[str]) -> str:
    lookup = {str(col).strip().lower(): str(col).strip() for col in df.columns}
    for name in names:
        key = name.strip().lower()
        if key in lookup:
            return lookup[key]
    return ''


def _safe_value(row: pd.Series, col: str) -> float:
    if not col:
        return 0.0
    value = row.get(col, 0.0)
    if pd.isna(value):
        return 0.0
    return float(value)


def _build_oxide_wt_from_row(row: pd.Series) -> Dict[str, float]:
    oxide_cols = {
        'SiO2': _find_first_column(row.to_frame().T, ['SiO2']),
        'MgO': _find_first_column(row.to_frame().T, ['MgO']),
        'FeO': _find_first_column(row.to_frame().T, ['FeO']),
        'CaO': _find_first_column(row.to_frame().T, ['CaO']),
        'Al2O3': _find_first_column(row.to_frame().T, ['Al2O3']),
        'Na2O': _find_first_column(row.to_frame().T, ['Na2O']),
        'Cr2O3': _find_first_column(row.to_frame().T, ['Cr2O3']),
    }

    
            

    wt = {oxide: _safe_value(row, col_name) for oxide, col_name in oxide_cols.items()}

    for  ox in ['SiO2', 'MgO', 'FeO']: # force non-zero values for key oxides
        print(f'Row {row.name} oxide {ox} value: {wt[ox]}')
        if not wt[ox]:
            print(f'Row {row.name} missing {ox}, assigning random value')
            if ox == 'SiO2':
                wt[ox] = np.random.uniform(30,55)
            elif ox == 'MgO':
                wt[ox] = np.random.uniform(5, 40)
            elif ox == 'FeO':
                wt[ox] = np.random.uniform(0.1, 15)

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

def main() -> None:
    start_time = time.time()

    ensemble_location = str(internal_scratch_dir())
    georoc_path = REPO_ROOT / 'data' / 'MELTStables' / 'GEOROC' / 'GEOROC_PETDB_UNFILTERED_WHOLEROCK_TRAIN.csv'

    os.makedirs(ensemble_location, exist_ok=True)

    date = 'Mar4'
    melts_model = 'HeFESTo'
    #run_code = [P0, P0+139, 140, S, 0, 0, -2, 0, 0, 0, 0]
    run_code = [0.1, 0.1, 0, 2500, 2500, 0, 0, 0, 0, 0, 0]

    total_to_run = int(1)
    simcycle = int(4)
    random_seed = None
    target_total_moles = 24.0

    nameCodes = [
        'plg', 'sp', 'opx', 'cpx', 'c2c', 'wo', 'pwo', 'gt', 'cpv', 'ol',
        'wa', 'ri', 'il', 'pv', 'ppv', 'cf', 'nal', 'mw', 'qtz', 'coes',
        'st', 'apbo', 'ky', 'neph', 'fea', 'feg', 'fee'
    ]
    allowed_phases = [HEFESTO_ABBREVIATION_TO_SHORT_NAMES[code] for code in nameCodes]
    headers = generate_column_headers_hefesto(allowed_phases)
    indexer = DatasetIndexer(headers, OXYGEN='closed', MODEL='HeFESTo')

    out_folder = Path(internal_data_dir(melts_model))
    os.makedirs(out_folder, exist_ok=True)
    out_base = out_folder / f'HeFESTo_Trainset{date}NTP'
    out_csv = str(out_base) + '.csv'

    georoc_df = pd.read_csv(georoc_path)
    if georoc_df.empty:
        raise ValueError('GEOROC file has no rows')

    feot_col = _find_first_column(georoc_df, ['FeOT', 'FeOt', 'FeO_total'])
    feo_col = _find_first_column(georoc_df, ['FeO'])
    fe2o3_col = _find_first_column(georoc_df, ['Fe2O3'])
    if not (feot_col or feo_col or fe2o3_col):
        raise ValueError('GEOROC file lacks FeOT/FeO/Fe2O3 for iron speciation')

    required_cols = ['SiO2', 'MgO', 'CaO', 'Al2O3']
    available = [bool(_find_first_column(georoc_df, [col])) for col in required_cols]
    if not all(available):
        missing = [required_cols[i] for i, ok in enumerate(available) if not ok]
        raise ValueError(f'GEOROC file missing required oxide columns: {missing}')

    mgo_col = _find_first_column(georoc_df, ['MgO'])
    cro_col = _find_first_column(georoc_df, ['Cr2O3'])
    if not mgo_col:
        raise ValueError('GEOROC file missing MgO column for mafic filtering')

    mafic_df = georoc_df[pd.to_numeric(georoc_df[mgo_col], errors='coerce').fillna(0.0) > 20.0]
    #mafic_df.iloc[:,cro_col] = np.random.uniform(0.01, 0.5, size=len(mafic_df))
    if mafic_df.empty:
        raise ValueError('No mafic compositions found (MgO > 5 wt%)')

    total_sims = total_to_run * simcycle
    if len(mafic_df) < total_sims:
        raise ValueError(
            f'Not enough unique mafic rows for requested simulations: '
            f'need {total_sims}, found {len(mafic_df)}'
        )

    subset = mafic_df.sample(n=total_sims, replace=False, random_state=random_seed)
    fe3_fet_grid = np.linspace(0.0, 0.1, simcycle)

    element_keys = np.array(['Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr', 'O'])
    element_rows: List[List[float]] = []
    wts = []
    for sim_idx, (_, row) in enumerate(subset.iterrows()):
        ratio = float(fe3_fet_grid[sim_idx % simcycle])
        base_oxide_wt = _build_oxide_wt_from_row(row)
        speciated_wt = _speciate_iron_and_normalize(base_oxide_wt, ratio)
        wt_debug = ', '.join(f'{key}={value:.4f}' for key, value in speciated_wt.items())
        print(f'Sim {sim_idx} Fe3/FeT={ratio:.4f} -> {wt_debug}')
        wts.append(wt_debug)

        element_moles = _oxide_wt_to_element_moles(speciated_wt)
        element_moles = _normalize_total_moles(element_moles, target_total_moles)
        element_rows.append([element_moles[key] for key in element_keys])

    input_array = np.asarray(element_rows, dtype=float)
    print(f'Prepared {input_array.shape[0]} HeFESTo simulations')

    MELTER.forward_HeFESTo(
        input_array=input_array,
        keys=element_keys,
        run_code=run_code,
        EnsembleLocation=ensemble_location,
    )

    fault_ids = MELTER.import_HeFESTo_components(
        workspace_dir=ensemble_location,
        indexer=indexer,
        dataname=out_csv,
    )
    print(f'HeFESTo parse complete. Fault IDs: {fault_ids}')
    print(f'Compiled output written to: {out_csv}')
    print(f'Wt debug info:')
    for wt in wts:
        print(wt)

    elapsed_seconds = time.time() - start_time
    print(f'main() execution time: {elapsed_seconds:.2f} seconds')

if __name__ == '__main__':
    main()
