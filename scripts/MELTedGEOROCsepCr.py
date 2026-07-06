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
from builder.indexer import generate_column_headers, DatasetIndexer
from builder.alphamelts.engine.composition_pool import build_ratio_pool
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
# Location to where to put the computed files.
EnsembleLocation = str(internal_scratch_dir())

GEOROC_DIR = os.path.join(REPO_ROOT, 'data', 'MELTStables', 'GEOROC')


os.makedirs(EnsembleLocation, exist_ok=True)

"""sleepytime = 3600 * 4
print("Waiting for " + str(np.round(sleepytime/3600, 2)) + " hours before starting dataset generation ")
import time
time.sleep(sleepytime)"""

calctype = 'Cooling' # Isobaric: 'Cooling', 'Compression'. To add: Isentropic, Isochoric, Isenthalpic  # 'FxCryst', 'FxMelt', 'Batch'
input_date = 'July1closed'

input_ZeroOxides = ['MnO', 'NiO'] # List of oxides to set to zero
MELTSmodels = ['102']#, '120']#, '102'] # MELTS models to run. To add: MAGEmin
FXes = ['Batch']#, 'FxCryst']
Prange = None # Auto if None, for lithosphere/aesthenospere (p)

Oxygen = 'Closed' # 'Closed' or 'Open' system with respect to oxygen. Buffered or constant oxygen?

total_to_run = int(300) # How many total simulations to run. Now 

startTs = [1800, 1800]#, 1800]
delta = -1
input_liquid_fractions = [102, 102]#, 100] # Make above 100 to allow for superliquidus
simcycle = 50 # How many simulations to run per iteration

#storage_directory = f'/mnt/d/Workspace/{MELTSModel}Datasets/'

# Check that arguments are valid

# Tunable: final table row count for each memmap (~30x scale-up target).
target_rows_train = 90_000_000
target_rows_valid = 2_500_000 # Change pipeline later to keep all this as test, no 3rd valid split. Existing set is fine
#batch_file = MELTSModel + 'batch'

for N, MELTSModel in enumerate(MELTSmodels):#, '102', '120']): 

    if MELTSModel == 'p':
        ultramafics_to_run = int(total_to_run * 0.4)
        mafics_to_run = int(total_to_run * 0.4)
        full_to_run = int(total_to_run * 0.2)
    else:
        ultramafics_to_run = int(total_to_run * 0.2) #0.15)
        mafics_to_run = int(total_to_run * 0.3)
        full_to_run = int(total_to_run * 0.5) #0.55)

    for C, Tag in enumerate(['NoCr', 'Cr']): ###################################################################
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
                    'muscovite','fluid','liquid', 'hornblende', 'alloy-solid','alloy-liquid', 'graphite']
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
                    'Prange': Prange, 'delta': delta, 'Oxygen': Oxygen, 'af': af}

            # Pre-mix full/ultramafic/mafic compositions into a single pool per dataset,
            # weighted by the same proportions the old per-group iteration counts used,
            # and let the melter run to a fixed row target instead of calling it three
            # times with hand-tuned per-group iteration counts.
            ultramafics = GEOROC[:,col_dict['MgO']+1]>=25 # MgO above 25
            mafics = GEOROC[:,col_dict['MgO']+1]>=5 # MgO above 5
            ultramafics_valid = GEOROC_valid[:,col_dict_valid['MgO']+1]>=25
            mafics_valid = GEOROC_valid[:,col_dict_valid['MgO']+1]>=5

            train_weights = [full_to_run, ultramafics_to_run, mafics_to_run]
            train_pool = build_ratio_pool(
                [GEOROC, GEOROC[ultramafics], GEOROC[mafics]], train_weights, pool_size=20_000)
            valid_pool = build_ratio_pool(
                [GEOROC_valid, GEOROC_valid[ultramafics_valid], GEOROC_valid[mafics_valid]],
                train_weights, pool_size=5_000)

            if not parsed_args.no_training and target_rows_train > 0:
                MELTER(output_file=Trainfilename, GEOROC=train_pool, col_dict=col_dict,
                       itercode='mix', target_rows=target_rows_train, **shared_kwargs)

            if not parsed_args.no_validation and target_rows_valid > 0:
                MELTER(output_file=Validfilename, GEOROC=valid_pool, col_dict=col_dict_valid,
                       itercode='mix', target_rows=target_rows_valid, **shared_kwargs)



            # Clean up progress files upon completion
            if os.path.exists(Train_progress_file):
                os.remove(Train_progress_file)
                print(f"Training simulations completed. Progress file removed.")

                # Clean up progress file upon completion
            if os.path.exists(Valid_progress_file):
                os.remove(Valid_progress_file)
                print(f"Validation simulations completed. Progress file removed.")
