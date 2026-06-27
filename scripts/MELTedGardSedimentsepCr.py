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
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--no-training', action='store_true', help='Skip training dataset generation')
parser.add_argument('--no-validation', action='store_true', help='Skip validation dataset generation')
parser.add_argument('--mac', action='store_true', help='Use MAC engine (alphameltsMAC) instead of Linux alphamelts')
parsed_args = parser.parse_args()
from builder.alphamelts.engine import RandomMelters as RM

if parsed_args.mac:
    #from builder.alphameltsMAC.engineMAC import RandomMelters20 as RM
    from builder.alphameltsMAC.engineMAC import alphamelts_functions as af
else:
    from builder.alphamelts.engine import alphamelts_functions as af

DATA_DIR = REPO_ROOT / 'data'

alphaMELTSLocation = os.path.join(REPO_ROOT, 'src', 'builder', 'alphamelts', 'engine', 'alphamelts-app-2.3.1-linux', 'alphamelts_linux')
# Location to where to put the computed files.
EnsembleLocation = str(internal_scratch_dir())

GEOROC_DIR = os.path.join(REPO_ROOT, 'data', 'MELTStables', 'GEOROC')


os.makedirs(EnsembleLocation, exist_ok=True)

"""sleepytime = 3600 * 4
print("Waiting for " + str(np.round(sleepytime/3600, 2)) + " hours before starting dataset generation ")
import time
time.sleep(sleepytime)"""

calctype = 'Cooling' # Isobaric: 'Cooling', 'Compression'. To add: Isentropic, Isochoric, Isenthalpic  # 'FxCryst', 'FxMelt', 'Batch'
input_date = 'June26Sediments'

input_ZeroOxides = ['MnO', 'NiO'] # List of oxides to set to zero
MELTSmodels = ['120']#, '120']#, '102'] # MELTS models to run. To add: MAGEmin
FXes = ['Batch']#, 'FxCryst']
Prange = None # Auto if None, for lithosphere/aesthenospere (p)

Oxygen = 'Open' # 'Closed' or 'Open' system with respect to oxygen. Buffered or constant oxygen?

total_to_run = int(300) # How many total simulations to run

startTs = [1800, 1800]#, 1800]
delta = -2
input_liquid_fractions = [102, 102]#, 100] # Make above 100 to allow for superliquidus
simcycle = 50 # How many simulations to run per iteration

#storage_directory = f'/mnt/d/Workspace/{MELTSModel}Datasets/'

# Check that arguments are valid


#batch_file = MELTSModel + 'batch'

for N, MELTSModel in enumerate(MELTSmodels):#, '102', '120']): 

  
    carbonates_to_run = int(total_to_run * 0.40) # Most fail
    noncarbonates_to_run = int(total_to_run * 0.70) # almost as many fail

    for C, Tag in enumerate(['Cr']): ###################################################################
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
                    'muscovite','fluid','liquid', 'hornblende', 'alloy-solid','alloy-liquid', 'graphite', 'calcite', 'dolomite', 'magnesite']
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
            #if calctype == 'Cooling':
            MELTER = RM.alphaMELTScooling
            """elif calctype == 'Compression':
                MELTER = RM.alphaMELTScompress"""
            
            #GEOROC = np.genfromtxt(GEOROC_DIR + '/GEOROC_PETDB_UNFILTERED_WHOLEROCK_TRAIN.csv', delimiter=',',skip_header=1)
            GEOtab = pd.read_csv(GEOROC_DIR + '/Gard2019_sediments_train.csv')
            GEOROC = GEOtab.to_numpy()
            # Define keys for input compositions (oxides)
            keys = np.array(GEOtab.columns)[1:]

            # Create col_dict mapping keys to indices
            col_dict = {}
            for i, k in enumerate(keys):
                col_dict[k] = i

            GEOtab_valid = pd.read_csv(GEOROC_DIR + '/Gard2019_sediments_valid.csv')
            GEOROC_valid = GEOtab_valid.to_numpy()

            if "Cr2O3" not in ZeroOxides: # Sprinkle Cr2O3 into 35% of the training and validation datasets
                Cr_spiked = np.random.choice(np.arange(len(GEOROC)), size=int(len(GEOROC)*0.35), replace=False)
                GEOROC[Cr_spiked, col_dict['Cr2O3']] = np.random.uniform(high=1, size=len(Cr_spiked)) # Spike Cr2O3 in 35% of training dataset
                Cr_spiked_valid = np.random.choice(np.arange(len(GEOROC_valid)), size=int(len(GEOROC_valid)*0.35), replace=False)
                GEOROC_valid[Cr_spiked_valid, col_dict['Cr2O3']] = np.random.uniform(high=1, size=len(Cr_spiked_valid)) # Spike Cr2O3 in 35% of validation dataset
                
            keys_valid = np.array(GEOtab_valid.columns)[1:]
            col_dict_valid = {}
            for i, k in enumerate(keys_valid):
                col_dict_valid[k] = i

            shared_kwargs = {'MELTSModel':MELTSModel, 'indexer':indexer, 'simcycle':simcycle,
                    'fxtal': (fractionate == 'FxCryst'), 'startT': startT,
                    'max_liquid_fraction': max_liquid_fraction, 'zeroOxides': ZeroOxides,
                    'Prange': Prange, 'delta': delta, 'Oxygen': Oxygen, 'replace_CO2': False, 'af': af} # Keep reported CO2 values

            args = {**shared_kwargs, 'GEOROC':GEOROC, 'col_dict':col_dict}
            valid_args = {**shared_kwargs, 'GEOROC':GEOROC_valid, 'col_dict':col_dict_valid}

            if carbonates_to_run != 0:
                carbonates = GEOROC[:,col_dict['CO2']+1]>=10 # CO2 above 10
                carbonates_valid = GEOROC_valid[:,col_dict_valid['CO2']+1]>=10

                if not parsed_args.no_training:
                    args['GEOROC'] = GEOROC[carbonates]
                    args['itercode'] = f'c{carbonates_to_run}'
                    MELTER(output_file=Trainfilename, **args)

                if not parsed_args.no_validation:
                    valid_args['GEOROC'] = GEOROC_valid[carbonates_valid]
                    valid_args['itercode'] = f'c{int(carbonates_to_run//4)}'
                    MELTER(output_file=Validfilename, **valid_args)

            if noncarbonates_to_run != 0:
                noncarbonates = GEOROC[:,col_dict['CaO']+1]<10 # CaO below 10
                noncarbonates_valid = GEOROC_valid[:,col_dict_valid['CaO']+1]<10

                if not parsed_args.no_training:
                    args['GEOROC'] = GEOROC[noncarbonates]
                    args['itercode'] = f'n{noncarbonates_to_run}'
                    MELTER(output_file=Trainfilename, **args)

                if not parsed_args.no_validation:
                    valid_args['GEOROC'] = GEOROC_valid[noncarbonates_valid]
                    valid_args['itercode'] = f'n{int(noncarbonates_to_run//4)}'
                    MELTER(output_file=Validfilename, **valid_args)



            # Clean up progress files upon completion
            if os.path.exists(Train_progress_file):
                os.remove(Train_progress_file)
                print(f"Training simulations completed. Progress file removed.")

                # Clean up progress file upon completion
            if os.path.exists(Valid_progress_file):
                os.remove(Valid_progress_file)
                print(f"Validation simulations completed. Progress file removed.")
