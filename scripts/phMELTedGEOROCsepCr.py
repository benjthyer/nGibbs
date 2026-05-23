"""Note: This script is for generating training data and will not be exported to users in the release.
The linux virtual environment is loaded by calling: source ~/melts_env/venv/bin/activate"""

import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'

# Add repo root and src to path so repo-local packages resolve from repo root.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recipes.settings import internal_data_dir, internal_scratch_dir
#from builder.alphamelts.engine import alphamelts_functions # The essential ensemble MELTS functions
#from nMELTS.utils.string_utils import pull_number, random_char
from builder.alphamelts.engine import RandomMelters as RM
from builder.indexer import generate_column_headers, DatasetIndexer
import numpy as np
import pandas as pd
#import warnings
#warnings.filterwarnings("ignore", category=UserWarning)
import shutil

    
DATA_DIR = REPO_ROOT / 'data'

alphaMELTSLocation = os.path.join(REPO_ROOT, 'src', 'builder', 'alphamelts', 'engine', 'linux_alphamelts_1-9', 'run_alphamelts.command')
settings_location = os.path.join(REPO_ROOT, 'src', 'builder', 'alphamelts', 'batch', 'phMELTS_settings.txt')

# Location to where to put the computed files.
EnsembleLocation = str(internal_scratch_dir())

GEOROC_DIR = os.path.join(REPO_ROOT, 'data', 'MELTStables', 'GEOROC')


os.makedirs(EnsembleLocation, exist_ok=True)


"""print("Waiting")
import time
time.sleep(8000)"""

calctype = 'Decompression' # Isobaric: 'Cooling', 'Compression'. To add: Isentropic, Isochoric, Isenthalpic  # 'FxCryst', 'FxMelt', 'Batch'
input_date = 'Feb16UM'

input_ZeroOxides = ['MnO', 'NiO'] # List of oxides to set to zero
MELTSmodels = ['ph']#, '102'] # MELTS models to run. To add: MAGEmin
FXes = ['Batch']#, 'FxCryst']
Prange = None # Auto if None, for lithosphere/aesthenospere (p)

total_to_run = int(300) # How many total simulations to run
ultramafics_to_run = total_to_run*(0.70) #int(total_to_run * 0.5)
mafics_to_run = total_to_run*(0.20) #int(total_to_run * 0.4)
full_to_run = total_to_run*(0.10) #int(total_to_run * 0.1)

input_liquid_fractions = [15]#, 100] # Make above 100 to allow for superliquidus
simcycle = 50 # How many simulations to run per iteration

#storage_directory = f'/mnt/d/Workspace/{MELTSModel}Datasets/'

# Check that arguments are valid


#batch_file = MELTSModel + 'batch'

for N, MELTSModel in enumerate(MELTSmodels):#, '102', '120']): 
    for C, Tag in enumerate(['NoCr', 'Cr']):
        ZeroOxides = input_ZeroOxides.copy()
        if Tag == 'NoCr':
            ZeroOxides.append('Cr2O3')
        date = input_date + Tag
        max_liquid_fraction = input_liquid_fractions[N]
        #total_to_run = total_to_run_input/(C+1) # Half the size of Cr dataset

        for fractionate in FXes:#, 'FxCryst']:

            allowed_phases = ['olivine','orthopyroxene','clinopyroxene','spinel','plagioclase','feldspar','garnet',
                'rhm-oxide','alloy-solid','alloy-liquid','quartz','tridymite','cristobalite','fluid','liquid']
            for zeroOx in['MnO', 'NiO', 'P2O5']: #pMELTS must exclude these...
                if zeroOx not in ZeroOxides:
                    ZeroOxides.append(zeroOx)


            # Generate headers and create indexer for this set of phases
            headers = generate_column_headers(allowed_phases, mode=MELTSModel, zeroOxides=ZeroOxides)
            indexer = DatasetIndexer(headers, OXYGEN='closed', MODEL='MELTS') # No use of ml_indexer here, but we need to specify the same OXYGEN and MODEL to sidestep errors, even though they are not used

            assert fractionate in ['Batch', 'FxCryst'], "fractionate argument must be one of ['Batch', 'FxCryst'], 'FxMelt' not yet implemented"

            Out_Folder = Path(internal_data_dir(MELTSModel))
            
            os.makedirs(Out_Folder, exist_ok=True)

            Trainfilename = str(Out_Folder / f'MELTS{MELTSModel}_Trainset{date}{fractionate}{calctype}')
            Validfilename = str(Out_Folder / f'MELTS{MELTSModel}_Validset{date}{fractionate}{calctype}')
            Train_progress_file = f'{Trainfilename}_progress.txt'
            Valid_progress_file = f'{Validfilename}_progress.txt'

            MELTER = RM.alphaMELTSERph # Simpler melter. Let settings file do the work. 

            # Clunky. Distributing the types of data to be run. This may be superceded by a more elegant solution soon.
            """if MELTSModel == 'p' and fractionate == 'Batch':
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
                full_to_run = int(total_to_run * 0.3)"""


            # Generate Training Dataset
            
            
            
            GEOROC = np.genfromtxt(GEOROC_DIR + '/GEOROC_PETDB_UNFILTERED_WHOLEROCK_TRAIN.csv', delimiter=',',skip_header=1)
            
            # Define keys for input compositions (oxides)
            keys = np.array(pd.read_csv(GEOROC_DIR + '/GEOROC_PETDB_UNFILTERED_WHOLEROCK_TRAIN.csv').columns)[1:]

            # Create col_dict mapping keys to indices
            col_dict = {}
            for i, k in enumerate(keys):
                col_dict[k] = i

            # Run full GEOROC training dataset
            args = {'GEOROC':GEOROC, 'col_dict':col_dict, 'indexer':indexer, 
                    'itercode':f'a{full_to_run}', 'simcycle':simcycle, 'fxtal': (fractionate == 'FxCryst'), 
                     'max_liquid_fraction': max_liquid_fraction, 'zeroOxides': ZeroOxides, 
                     'settingsLocation': settings_location, 'alphameltsLocation': alphaMELTSLocation}
            
            if full_to_run != 0:
                MELTER(output_file=Trainfilename, **args)

                # Run full GEOROC validation dataset
                args['itercode'] = f'a{int(full_to_run//4)}'
                MELTER(output_file=Validfilename, **args)

            if ultramafics_to_run != 0:
                ultramafics = GEOROC[:,col_dict['MgO']+1]>=25 # MgO above 25
            
                args['GEOROC'] = GEOROC[ultramafics]
                # Run ultramafic GEOROC training dataset
                args['itercode'] = f'u{ultramafics_to_run}'

                MELTER(output_file=Trainfilename, **args)

                # Run ultramafic GEOROC validation dataset
                args['itercode'] = f'u{int(ultramafics_to_run//4)}'
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
