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

calctype = 'Cooling' # Isobaric: 'Cooling', 'Compression'. To add: Isentropic, Isochoric, Isenthalpic  # 'FxCryst', 'FxMelt', 'Batch'
date = 'Feb5'

ZeroOxides = ['MnO', 'NiO', 'CoO'] # List of oxides to set to zero
MELTSmodels = ['p', '102'] # MELTS models to run. To add: MAGEmin
FXes = ['Batch']#, 'FxCryst']

startTs = [1400, 1800]
max_liquid_fractions = [15, 100]
total_to_run = int(80) # How many total simulations to run
simcycle = 50 # How many simulations to run per iteration

#storage_directory = f'/mnt/d/Workspace/{MELTSModel}Datasets/'

# Check that arguments are valid


#batch_file = MELTSModel + 'batch'

for N, MELTSModel in enumerate(MELTSmodels):#, '102', '120']): 
    startT = startTs[N]
    max_liquid_fraction = max_liquid_fractions[N]
    for fractionate in FXes:#, 'FxCryst']:

        if MELTSModel == 'p':
            allowed_phases = ['olivine','orthopyroxene','clinopyroxene','spinel','plagioclase','k-feldspar','garnet',
            'rhm-oxide','alloy-solid','alloy-liquid','apatite','whitlockite','quartz','tridymite','cristobalite','fluid','liquid']
            for zeroOx in['MnO', 'NiO', 'CoO']: #pMELTS must exclude these...
                if zeroOx not in ZeroOxides:
                    ZeroOxides.append(zeroOx)
        else:
            allowed_phases = ['olivine','orthopyroxene','clinopyroxene','spinel','plagioclase','k-feldspar','garnet',
                'nepheline','leucite','biotite','rhm-oxide','alloy-solid','alloy-liquid','apatite','whitlockite','quartz','tridymite','cristobalite','muscovite','fluid','liquid']

        # Generate headers and create indexer for this set of phases
        headers = generate_column_headers(allowed_phases, mode=MELTSModel, zeroOxides=ZeroOxides)
        indexer = DatasetIndexer(headers)

        assert fractionate in ['Batch', 'FxCryst'], "fractionate argument must be one of ['Batch', 'FxCryst'], 'FxMelt' not yet implemented"
        assert calctype in ['Cooling', 'Compression'], "calctype argument must be one of ['Cooling', 'Compression'], isoentropic, isoenthalpic, isochroic not yet implemented"
        assert MELTSModel in ['102', '110', '120', 'p'], "MELTSModel argument must be one of ['102', '110', '120', 'p'], MAGEmin not yet implemented"


        Out_Folder = Path(internal_data_dir(MELTSModel))
        
        os.makedirs(Out_Folder, exist_ok=True)

        Trainfilename = str(Out_Folder / f'MELTS{MELTSModel}_Trainset{date}{fractionate}{calctype}')
        Validfilename = str(Out_Folder / f'MELTS{MELTSModel}_Validset{date}{fractionate}{calctype}')
        Train_progress_file = f'{Trainfilename}_progress.txt'
        Valid_progress_file = f'{Validfilename}_progress.txt'

        #logic trees to direct dataset generation:
        if calctype == 'Cooling':
            MELTER = RM.alphaMELTScooling
        elif calctype == 'Compression':
            MELTER = RM.alphaMELTScompress

        # Clunky. Distributing the types of data to be run. This may be superceded by a more elegant solution soon.
        if MELTSModel == 'p' and fractionate == 'Batch':
            mafics_to_run = int(total_to_run * 0.9)
            full_to_run = int(total_to_run * 0.1)
        elif MELTSModel == 'p':
            mafics_to_run = int(total_to_run * 0.975)
            full_to_run = int(total_to_run * 0.025)
        elif MELTSModel == '120' and fractionate == 'Batch':
            mafics_to_run = int(0)
            full_to_run = total_to_run
        elif MELTSModel == '120':
            mafics_to_run = int(total_to_run * 0.2)
            full_to_run = int(total_to_run * 0.8)
        elif fractionate == 'Batch':
            mafics_to_run = int(total_to_run * 0.3)
            full_to_run = int(total_to_run * 0.7)
        else:
            mafics_to_run = int(total_to_run * 0.7)
            full_to_run = int(total_to_run * 0.3)


        # Generate Training Dataset
        
        
        
        GEOROC = np.genfromtxt(GEOROC_DIR + '/GEOROC_PETDB_UNFILTERED_WHOLEROCK_TRAIN.csv', delimiter=',',skip_header=1)
        
        # Define keys for input compositions (oxides)
        keys = np.array(pd.read_csv(GEOROC_DIR + '/GEOROC_PETDB_UNFILTERED_WHOLEROCK_TRAIN.csv').columns)[1:]

        # Create col_dict mapping keys to indices
        col_dict = {}
        for i, k in enumerate(keys):
            col_dict[k] = i

        # Run full GEOROC training dataset
        args = {'MELTSModel':MELTSModel, 'GEOROC':GEOROC, 'col_dict':col_dict, 'indexer':indexer, 
                'itercode':f'a{full_to_run}', 'simcycle':simcycle, 'fxtal': (fractionate == 'FxCryst'), 
                'startT': startT, 'max_liquid_fraction': max_liquid_fraction}
        
        MELTER(output_file=Trainfilename, **args)

        # Run full GEOROC validation dataset
        args['itercode'] = f'a{int(full_to_run//4)}'
        MELTER(output_file=Validfilename, **args)

        if mafics_to_run != 0:
            mafics = GEOROC[:,col_dict['MgO']+1]>=5 # MgO above 5
           
            args['GEOROC'] = GEOROC[mafics]
            # Run mafic GEOROC training dataset
            args['itercode'] = f'm{mafics_to_run}'

            MELTER(output_file=Trainfilename, **args)

            # Run mafic GEOROC validation dataset
            args['itercode'] = f'm{int(mafics_to_run//4)}'
            MELTER(output_file=Validfilename, **args)


        # Clean up progress files upon completion
        if os.path.exists(Train_progress_file):
            os.remove(Train_progress_file)
            print(f"Training simulations completed. Progress file removed.")

            # Clean up progress file upon completion
        if os.path.exists(Valid_progress_file):
            os.remove(Valid_progress_file)
            print(f"Validation simulations completed. Progress file removed.")
