"""Note: This script is for generating training data and will not be exported to users in the release.
The linux virtual environment is loaded by calling: source ~/melts_env/venv/bin/activate"""

import sys
import os
# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# Add src to path so we can import modules without src prefix
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from nMELTS.config.settings import internal_data_dir, internal_scratch_dir
#from builder.alphamelts.engine import alphamelts_functions # The essential ensemble MELTS functions
#from nMELTS.utils.string_utils import pull_number, random_char
from builder.alphamelts.engine import RandomMelters as RM
from builder.indexer import generate_column_headers, DatasetIndexer
import numpy as np
import pandas as pd
#import warnings
#warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path
import shutil

    
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / 'data'

alphaMELTSLocation = os.path.join(REPO_ROOT, 'src', 'builder', 'alphamelts', 'engine', 'alphamelts-app-2.3.1-linux', 'alphamelts_linux')
# Location to where to put the computed files.
EnsembleLocation = str(internal_scratch_dir())

GEOROC_DIR = os.path.join(REPO_ROOT, 'data', 'MELTStables', 'GEOROC')
os.makedirs(EnsembleLocation, exist_ok=True)

total_to_run = int(1) # How many total simulations to run
simcycle = 40 # How many simulations to run per iteration
calctype = 'Cooling' # Isobaric: 'Cooling', 'Compression'. To add: Isentropic, Isochoric, Isenthalpic  # 'FxCryst', 'FxMelt', 'Batch'
MELTSModel= 'p'
fractionate = 'Batch'

date = 'Feb11'
startT = 1600
max_liquid_fraction = 15

ZeroOxides = ['MnO', 'NiO'] # List of oxides to set to zero

Out_Folder = Path(internal_data_dir(MELTSModel))
os.makedirs(Out_Folder, exist_ok=True)

Trainfilename = str(Out_Folder / f'MELTS{MELTSModel}_Trainset{date}{fractionate}{calctype}')
Validfilename = str(Out_Folder / f'MELTS{MELTSModel}_Validset{date}{fractionate}{calctype}')

if MELTSModel == 'p':
    allowed_phases = ['olivine','orthopyroxene','clinopyroxene','spinel','plagioclase','k-feldspar','garnet',
    'rhm-oxide','alloy-solid','alloy-liquid','quartz','tridymite','cristobalite','fluid','liquid'] # Apatite/Whitlockite don't seem to crystallize at all for me?
    for zeroOx in ['MnO', 'NiO', 'P2O5']: #pMELTS must exclude these...
        if zeroOx not in ZeroOxides:
            ZeroOxides.append(zeroOx)
     
else:
    allowed_phases = ['olivine','orthopyroxene','clinopyroxene','spinel','plagioclase','k-feldspar','garnet',
        'nepheline','leucite','biotite','rhm-oxide','alloy-solid','alloy-liquid','apatite','whitlockite','quartz','tridymite','cristobalite','muscovite','fluid','liquid']

# Generate headers and create indexer for this set of phases
headers = generate_column_headers(allowed_phases, mode=MELTSModel, zeroOxides=ZeroOxides) # pMELTS doesn't include Corundum in rhm-oxide

indexer = DatasetIndexer(headers)

assert fractionate in ['Batch', 'FxCryst'], "fractionate argument must be one of ['Batch', 'FxCryst'], 'FxMelt' not yet implemented"
assert calctype in ['Cooling', 'Compression'], "calctype argument must be one of ['Cooling', 'Compression'], isoentropic, isoenthalpic, isochroic not yet implemented"
assert MELTSModel in ['102', '110', '120', 'p'], "MELTSModel argument must be one of ['102', '110', '120', 'p'], MAGEmin not yet implemented"

#logic trees to direct dataset generation:
if calctype == 'Cooling':
    MELTER = RM.alphaMELTScooling
elif calctype == 'Compression':
    MELTER = RM.alphaMELTScompress

GEOROC = np.genfromtxt(GEOROC_DIR + '/GEOROC_PETDB_UNFILTERED_WHOLEROCK_TRAIN.csv', delimiter=',',skip_header=1) # Skip index column

# Define keys for input compositions (oxides)
keys = np.array(pd.read_csv(GEOROC_DIR + '/GEOROC_PETDB_UNFILTERED_WHOLEROCK_TRAIN.csv').columns)[1:] # Skip index column
col_dict = {key:i for i, key in enumerate(keys)} # Create col_dict mapping keys to indices

maficsGEOROC = GEOROC[GEOROC[:, col_dict['MgO']+1] > 25] # Must add 1 to account for indexing. This is confusing.

args = {'MELTSModel':MELTSModel, 'GEOROC':maficsGEOROC, 'col_dict':col_dict, 'indexer':indexer, 'itercode':f'a{total_to_run}',
         'simcycle':simcycle, 'fxtal': (fractionate == 'FxCryst'), 'ExFailures':False, 'startT': startT, 
         'max_liquid_fraction': max_liquid_fraction, 'zeroOxides': ZeroOxides}

MELTER(output_file=Trainfilename, **args)
MELTER(output_file=Validfilename, **args)