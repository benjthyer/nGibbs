"""Note: This script is for generating training data and will not be exported to users in the release.
The linux virtual environment is loaded by calling: source ~/melts_env/venv/bin/activate"""

import sys
import os
# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.nMELTS.config.settings import internal_data_dir
#from src.wslMELTS.engine import alphamelts_functions # The essential ensemble MELTS functions
#from src.nMELTS.utils.string_utils import pull_number, random_char
import RandomMelters as RM
from src.nMELTS.config.indexer import generate_column_headers, DatasetIndexer
import numpy as np
import pandas as pd
#import warnings
#warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path
import shutil


Out_Folder = os.path.join(Path(__file__).parent.parent.absolute(), 'src', 'nMELTS', 'data', 'DataProducts')
    
alphaMELTSLocation = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'wslMELTS', 'engine', 'alphamelts-app-2.3.1-linux', 'alphamelts_linux')
# Location to where to put the computed files.
EnsembleLocation = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'wslMELTS', 'Workspace')

os.makedirs(EnsembleLocation, exist_ok=True)

total_to_run = int(1) # How many total simulations to run
simcycle = 8 # How many simulations to run per iteration
calctype = 'Cooling' # Isobaric: 'Cooling', 'Compression'. To add: Isentropic, Isochoric, Isenthalpic  # 'FxCryst', 'FxMelt', 'Batch'
MELTSModel= '102'
fractionate = 'Batch'

date = 'Jan14'
Trainfilename = Out_Folder + f'/MELTS{MELTSModel}_Trainset{date}{fractionate}{calctype}'
Validfilename = Out_Folder + f'/MELTS{MELTSModel}_Validset{date}{fractionate}{calctype}'

if MELTSModel == 'p':
        allowed_phases = ['olivine','orthopyroxene','clinopyroxene','spinel','plagioclase','k-feldspar','garnet',
    'rhm-oxide','alloy-solid','alloy-liquid','apatite','whitlockite','quartz','tridymite','cristobalite','fluid','liquid']
else:
    allowed_phases = ['olivine','orthopyroxene','clinopyroxene','spinel','plagioclase','k-feldspar','garnet',
        'nepheline','leucite','biotite','rhm-oxide','alloy-solid','alloy-liquid','apatite','whitlockite','quartz','tridymite','cristobalite','muscovite','fluid','liquid']

# Generate headers and create indexer for this set of phases
headers = generate_column_headers(allowed_phases)
indexer = DatasetIndexer(headers)

assert fractionate in ['Batch', 'FxCryst'], "fractionate argument must be one of ['Batch', 'FxCryst'], 'FxMelt' not yet implemented"
assert calctype in ['Cooling', 'Compression'], "calctype argument must be one of ['Cooling', 'Compression'], isoentropic, isoenthalpic, isochroic not yet implemented"
assert MELTSModel in ['102', '110', '120', 'p'], "MELTSModel argument must be one of ['102', '110', '120', 'p'], MAGEmin not yet implemented"

#logic trees to direct dataset generation:
if calctype == 'Cooling':
    MELTER = RM.alphaMELTScooling
elif calctype == 'Compression':
    MELTER = RM.alphaMELTScompress

GEOROC = np.genfromtxt(internal_data_dir('GEOROC') / 'GEOROC_PETDB_UNFILTERED_WHOLEROCK_TRAIN.csv', delimiter=',',skip_header=1)

# Define keys for input compositions (oxides)
keys = np.array(pd.read_csv(internal_data_dir('GEOROC') / 'GEOROC_PETDB_UNFILTERED_WHOLEROCK_TRAIN.csv', delimiter=',').columns)[1:] # Skip index column

# Create col_dict mapping keys to indices
col_dict = {}
for i, k in enumerate(keys):
    col_dict[k] = i

args = {'MELTSModel':MELTSModel, 'GEOROC':GEOROC, 'col_dict':col_dict, 'indexer':indexer, 'iter':total_to_run, 'simcycle':simcycle, 'fxtal': (fractionate == 'FxCryst')}

MELTER(output_file=Trainfilename, **args)
MELTER(output_file=Validfilename, **args)