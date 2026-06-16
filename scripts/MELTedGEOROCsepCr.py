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

alphaMELTSLocation = os.path.join(REPO_ROOT, 'src', 'builder', 'alphamelts', 'engine', 'alphamelts-app-2.3.1-linux', 'alphamelts_linux')
# Location to where to put the computed files.
EnsembleLocation = str(internal_scratch_dir())

GEOROC_DIR = os.path.join(REPO_ROOT, 'data', 'MELTStables', 'GEOROC')


os.makedirs(EnsembleLocation, exist_ok=True)


"""print("Waiting")
import time
time.sleep(8000)"""

calctype = 'Cooling' # Isobaric: 'Cooling', 'Compression'. To add: Isentropic, Isochoric, Isenthalpic  # 'FxCryst', 'FxMelt', 'Batch'
input_date = 'CO2TESTopen'

input_ZeroOxides = ['MnO', 'NiO'] # List of oxides to set to zero
MELTSmodels = ['102']#, '102'] # MELTS models to run. To add: MAGEmin
FXes = ['Batch']#, 'FxCryst']
Prange = None # Auto if None, for lithosphere/aesthenospere (p)

Oxygen = 'Open' # 'Closed' or 'Open' system with respect to oxygen. Buffered or constant oxygen?

total_to_run = int(300) # How many total simulations to run
ultramafics_to_run = int(total_to_run * 0.1)
mafics_to_run = int(total_to_run * 0.4)
full_to_run = int(total_to_run * 0.5)

startTs = [1800]#, 1800]
delta = -4
input_liquid_fractions = [101]#, 100] # Make above 100 to allow for superliquidus
simcycle = 8 # How many simulations to run per iteration

#storage_directory = f'/mnt/d/Workspace/{MELTSModel}Datasets/'

# Check that arguments are valid


#batch_file = MELTSModel + 'batch'

for N, MELTSModel in enumerate(MELTSmodels):#, '102', '120']): 
    for C, Tag in enumerate(['NoCr', 'Cr']):
        ZeroOxides = input_ZeroOxides.copy()
        if Tag == 'NoCr':
            ZeroOxides.append('Cr2O3')
        date = input_date + Tag
        startT = startTs[N]
        max_liquid_fraction = input_liquid_fractions[N]
        #total_to_run = total_to_run_input/(C+1) # Half the size of Cr dataset

        for fractionate in FXes:#, 'FxCryst']:

            if MELTSModel == 'p':
                allowed_phases = ['olivine','orthopyroxene','clinopyroxene','spinel','plagioclase','k-feldspar','garnet',
                'rhm-oxide','alloy-solid','alloy-liquid','quartz','tridymite','cristobalite','fluid','liquid']
                for zeroOx in['MnO', 'NiO', 'P2O5']: #pMELTS must exclude these...
                    if zeroOx not in ZeroOxides:
                        ZeroOxides.append(zeroOx)
            else:
                allowed_phases = ['olivine','orthopyroxene','clinopyroxene','spinel','plagioclase','k-feldspar','garnet',
                    'nepheline','leucite','biotite','rhm-oxide','apatite','whitlockite','quartz','tridymite',#'cristobalite',
                    'muscovite','fluid','liquid', 'hornblende', 'alloy-solid','alloy-liquid']
            if MELTSModel != '120':
                if 'CO2' not in ZeroOxides:
                    ZeroOxides.append('CO2')

            # Generate headers and create indexer for this set of phases
            headers = generate_column_headers(allowed_phases, mode=MELTSModel, zeroOxides=ZeroOxides)
            indexer = DatasetIndexer(headers, OXYGEN='closed', MODEL='MELTS') # No use of ml_indexer here, but we need to specify the same OXYGEN and MODEL to sidestep errors, even though they are not used

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
            
            
            
            #GEOROC = np.genfromtxt(GEOROC_DIR + '/GEOROC_PETDB_UNFILTERED_WHOLEROCK_TRAIN.csv', delimiter=',',skip_header=1)
            GEOtab = pd.read_csv(GEOROC_DIR + '/GEOROC_PETDB_UNFILTERED_WHOLEROCK_TRAIN.csv')
            if MELTSModel == '120':
                GEOtab['CO2'] = 0 # Add CO2 column with zeros since MELTS 120 requires it, even though GEOROC has no CO2 data. Will be filled in later by random melter.
            GEOROC = GEOtab.to_numpy()
            # Define keys for input compositions (oxides)
            keys = np.array(GEOtab.columns)[1:]

            # Create col_dict mapping keys to indices
            col_dict = {}
            for i, k in enumerate(keys):
                col_dict[k] = i

            GEOtab_valid = pd.read_csv(GEOROC_DIR + '/GEOROC_PETDB_UNFILTERED_WHOLEROCK_VALIDATION.csv')
            if MELTSModel == '120':
                GEOtab_valid['CO2'] = 0
            GEOROC_valid = GEOtab_valid.to_numpy()
            keys_valid = np.array(GEOtab_valid.columns)[1:]
            col_dict_valid = {}
            for i, k in enumerate(keys_valid):
                col_dict_valid[k] = i

            shared_kwargs = {'MELTSModel':MELTSModel, 'indexer':indexer, 'simcycle':simcycle,
                    'fxtal': (fractionate == 'FxCryst'), 'startT': startT,
                    'max_liquid_fraction': max_liquid_fraction, 'zeroOxides': ZeroOxides,
                    'Prange': Prange, 'delta': delta, 'Oxygen': Oxygen}

            # Run full GEOROC training dataset
            args = {**shared_kwargs, 'GEOROC':GEOROC, 'col_dict':col_dict, 'itercode':f'a{full_to_run}'}
            valid_args = {**shared_kwargs, 'GEOROC':GEOROC_valid, 'col_dict':col_dict_valid, 'itercode':f'a{int(full_to_run//4)}'}

            if full_to_run != 0:
                MELTER(output_file=Trainfilename, **args)

                # Run full GEOROC validation dataset
                MELTER(output_file=Validfilename, **valid_args)

            if ultramafics_to_run != 0:
                ultramafics = GEOROC[:,col_dict['MgO']+1]>=25 # MgO above 25
                ultramafics_valid = GEOROC_valid[:,col_dict_valid['MgO']+1]>=25

                args['GEOROC'] = GEOROC[ultramafics]
                args['itercode'] = f'u{ultramafics_to_run}'
                MELTER(output_file=Trainfilename, **args)

                valid_args['GEOROC'] = GEOROC_valid[ultramafics_valid]
                valid_args['itercode'] = f'u{int(ultramafics_to_run//4)}'
                MELTER(output_file=Validfilename, **valid_args)

            if mafics_to_run != 0:
                mafics = GEOROC[:,col_dict['MgO']+1]>=5 # MgO above 5
                mafics_valid = GEOROC_valid[:,col_dict_valid['MgO']+1]>=5

                args['GEOROC'] = GEOROC[mafics]
                args['itercode'] = f'm{mafics_to_run}'
                MELTER(output_file=Trainfilename, **args)

                valid_args['GEOROC'] = GEOROC_valid[mafics_valid]
                valid_args['itercode'] = f'm{int(mafics_to_run//4)}'
                MELTER(output_file=Validfilename, **valid_args)



            # Clean up progress files upon completion
            if os.path.exists(Train_progress_file):
                os.remove(Train_progress_file)
                print(f"Training simulations completed. Progress file removed.")

                # Clean up progress file upon completion
            if os.path.exists(Valid_progress_file):
                os.remove(Valid_progress_file)
                print(f"Validation simulations completed. Progress file removed.")
