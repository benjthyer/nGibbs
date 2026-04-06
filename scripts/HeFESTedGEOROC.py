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




def main() -> None:
    start_time = time.time()

    ensemble_location = str(internal_scratch_dir())
    georoc_path = REPO_ROOT / 'data' / 'MELTStables' / 'GEOROC' / 'GEOROC_PETDB_UNFILTERED_WHOLEROCK_TRAIN.csv'

    os.makedirs(ensemble_location, exist_ok=True)

    date = 'Apr4_Adiabats'
    melts_model = 'HeFESTo'
   
    #run_code = [0.1, 0.1, 0, 2500, 2500, 0, 0, 0, 0, 0, 0]
    #run_code = [0,140,140,2.5330109805002023,0,0,-2,0,0,0,0]

    total_cycles = int(10) # How many iterations of cycle, not implemented yet
    simcycle = int(4) # How many simulations to run per cycle
    total_sims = total_cycles * simcycle
    random_seed = None
    target_total_moles = 24.0
    Mps = 273 + 1200 + np.arange(total_sims)*(1650-1200)/total_sims

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
    
    if len(mafic_df) < total_sims:
            raise ValueError(
                f'Not enough unique mafic rows for requested simulations: '
                f'need {total_sims}, found {len(mafic_df)}'
            )
    subset = mafic_df.sample(n=total_sims, replace=False, random_state=random_seed)
    reduced_N = int(total_sims*(4/5))
    fe3_fet_grid = np.append(np.linspace(0.0, 0.05, reduced_N), np.linspace(0.05, 0.10, int(total_sims - reduced_N)))

    element_keys = np.array(['Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr', 'O'])

    for cycle in range(total_cycles):
        
        P0s = np.random.uniform(0, 1, size=simcycle)
        #run_code = [[P0, P0+139, 138, 'S', 0, 0, -2, 0, 0, 0, 0] for P0 in P0s] # Placeholder S evaluated downstream
        run_code = [[P0, P0+139, 138, 0, 0, 0, -1, 0, 0, 0, 0] for P0 in P0s] # ad.in files made downstream
        #total_sims = total_to_run * simcycle 
        
        
        element_rows: List[List[float]] = []
        wts = []

        for sim_idx, (_, row) in enumerate(subset.iloc[(simcycle*cycle):(simcycle*cycle)+simcycle].iterrows()):
            ratio = float(fe3_fet_grid[(simcycle*cycle)+sim_idx])
            base_oxide_wt = _build_oxide_wt_from_row(row)
            speciated_wt = _speciate_iron_and_normalize(base_oxide_wt, ratio)
            wt_debug = ', '.join(f'{key}={value:.4f}' for key, value in speciated_wt.items())
            print(f'Sim {sim_idx} Fe3/FeT={ratio:.4f} -> {wt_debug}')
            wts.append(wt_debug)

            element_moles = _oxide_wt_to_element_moles(speciated_wt)
            element_moles = _normalize_total_moles(element_moles, target_total_moles)
            element_rows.append([element_moles[key] for key in element_keys])
            #run_code[sim_idx][3] = get_S(T=Mps[(simcycle*cycle)+sim_idx], Ca=element_moles["Ca"]) # Potential temps between 1200 and 1650 C
            MELTER.make_PT_path(P=np.linspace(run_code[sim_idx][0], run_code[sim_idx][1], run_code[sim_idx][2]+2), S=get_S(T=Mps[(simcycle*cycle)+sim_idx], Ca=element_moles["Ca"]), func=get_T, out_path=ensemble_location + '/' + f"Simulation{sim_idx}")

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
