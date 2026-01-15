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
calctype = 'Cooling' # Isobaric: 'Cooling', 'Compression'. To add: Isentropic, Isochoric, Isenthalpic  # 'FxCryst', 'FxMelt', 'Batch'
date = 'Nov20'

total_to_run = int(160) # How many total simulations to run
simcycle = 50 # How many simulations to run per iteration

#storage_directory = f'/mnt/d/Workspace/{MELTSModel}Datasets/'

# Check that arguments are valid


#batch_file = MELTSModel + 'batch'

for N, MELTSModel in enumerate(['p', '102', '120']): 
    for fractionate in ['Batch', 'FxCryst']:

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

        Trainfilename = Out_Folder +f'/MELTS{MELTSModel}_Trainset{date}{fractionate}{calctype}'
        Validfilename = Out_Folder +f'/MELTS{MELTSModel}_Validset{date}{fractionate}{calctype}'
        

        #logic trees to direct dataset generation:
        if calctype == 'Cooling':
            MELTER = RM.alphaMELTScooling
        elif calctype == 'Compression':
            MELTER = RM.alphaMELTScompress

        # Clunky. Distributing the types of data to be run. This may be superceded by a more elegant solution soon.
        if MELTSModel == 'p' and fractionate == 'batch':
            mafics_to_run = int(total_to_run * 0.9)
            full_to_run = int(total_to_run * 0.1)
        elif MELTSModel == 'p':
            mafics_to_run = int(total_to_run * 0.975)
            full_to_run = int(total_to_run * 0.025)
        elif MELTSModel == '120' and fractionate == 'batch':
            mafics_to_run = int(0)
            full_to_run = total_to_run
        elif MELTSModel == '120':
            mafics_to_run = int(total_to_run * 0.2)
            full_to_run = int(total_to_run * 0.8)
        elif fractionate == 'batch':
            mafics_to_run = int(total_to_run * 0.3)
            full_to_run = int(total_to_run * 0.7)
        else:
            mafics_to_run = int(total_to_run * 0.7)
            full_to_run = int(total_to_run * 0.3)


        # Generate Training Dataset
        
        
        GEOROC = np.genfromtxt(internal_data_dir('GEOROC/GEOROC_PETDB_UNFILTERED_WHOLEROCK_TRAIN.csv'), delimiter=',',skip_header=1)
        
        # Define keys for input compositions (oxides)
        keys = np.array(pd.read_csv(internal_data_dir('GEOROC') / 'GEOROC_PETDB_UNFILTERED_WHOLEROCK_TRAIN.csv', delimiter=',').columns)[1:]

        # Create col_dict mapping keys to indices
        col_dict = {}
        for i, k in enumerate(keys):
            col_dict[k] = i

        args = {'MELTSModel':MELTSModel, 'GEOROC':GEOROC, 'col_dict':col_dict, 'indexer':indexer, 'iter':full_to_run, 'simcycle':simcycle, 'fxtal': (fractionate == 'FxCryst')}
        MELTER(output_file=Trainfilename, **args)

        args['iter'] = int(full_to_run//4)
        MELTER(output_file=Validfilename, **args)

        if mafics_to_run != 0:
            mafics = GEOROC[:,5]>=5 # MgO above 5
           
            args['GEOROC'] = GEOROC[mafics]
            args['iter'] = mafics_to_run

            MELTER(output_file=Trainfilename, **args)

            args['iter'] = int(mafics_to_run//4)
            MELTER(output_file=Validfilename, **args)

