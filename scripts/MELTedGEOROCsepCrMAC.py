"""Note: This script is for generating training data with BJT's old MAC:
uses builder/alphameltsMAC instead of builder/alphamelts."""

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
from builder.alphameltsMAC.engineMAC import RandomMelters20 as RM
from builder.indexer import generate_column_headers, DatasetIndexer
import numpy as np
import pandas as pd
import shutil


DATA_DIR = REPO_ROOT / 'data'

# Location to where to put the computed files.
EnsembleLocation = str(internal_scratch_dir())

GEOROC_DIR = os.path.join(REPO_ROOT, 'data', 'MELTStables', 'GEOROC')


os.makedirs(EnsembleLocation, exist_ok=True)


calctype = 'Cooling' # Isobaric: 'Cooling', 'Compression'. To add: Isentropic, Isochoric, Isenthalpic  # 'FxCryst', 'FxMelt', 'Batch'
input_date = 'June9'

input_ZeroOxides = ['MnO', 'NiO'] # List of oxides to set to zero
MELTSmodels = ['120']#, '102'] # MELTS models to run. To add: MAGEmin
FXes = ['Batch']#, 'FxCryst']
Prange = None # Auto if None, for lithosphere/aesthenospere (p)


Oxygen = 'closed' # 'closed' or 'open' system with respect to oxygen. Buffered or constant oxygen?

total_to_run = int(300) # How many total simulations to run
ultramafics_to_run = int(total_to_run * 0.15)
mafics_to_run = int(total_to_run * 0.3)
full_to_run = int(total_to_run * 0.55)

startTs = [1800]#, 1800]
delta = -4
input_liquid_fractions = [102]#, 100] # Make above 100 to allow for superliquidus
simcycle = 50 # How many simulations to run per iteration


for N, MELTSModel in enumerate(MELTSmodels):
    for C, Tag in enumerate(['NoCr', 'Cr']):
        ZeroOxides = input_ZeroOxides.copy()
        if Tag == 'NoCr':
            ZeroOxides.append('Cr2O3')
        date = input_date + Tag
        startT = startTs[N]
        max_liquid_fraction = input_liquid_fractions[N]

        for fractionate in FXes:

            if MELTSModel == 'p':
                allowed_phases = ['olivine','orthopyroxene','clinopyroxene','spinel','plagioclase','k-feldspar','garnet',
                'rhm-oxide','alloy-solid','alloy-liquid','quartz','tridymite','cristobalite','fluid','liquid']
                for zeroOx in['MnO', 'NiO', 'P2O5']: #pMELTS must exclude these...
                    if zeroOx not in ZeroOxides:
                        ZeroOxides.append(zeroOx)
            else:
                allowed_phases = ['olivine','orthopyroxene','clinopyroxene','spinel','plagioclase','k-feldspar','garnet',
                    'nepheline','leucite','biotite','rhm-oxide','apatite','whitlockite','quartz','tridymite',
                    'muscovite','fluid','liquid', 'hornblende', 'alloy-solid','alloy-liquid']

            # Generate headers and create indexer for this set of phases
            headers = generate_column_headers(allowed_phases, mode=MELTSModel, zeroOxides=ZeroOxides)
            indexer = DatasetIndexer(headers, OXYGEN='closed', MODEL='MELTS')

            assert fractionate in ['Batch', 'FxCryst'], "fractionate argument must be one of ['Batch', 'FxCryst'], 'FxMelt' not yet implemented"
            assert calctype in ['Cooling', 'Compression'], "calctype argument must be one of ['Cooling', 'Compression'], isoentropic, isoenthalpic, isochroic not yet implemented"
            assert MELTSModel in ['102', '110', '120', 'p'], "MELTSModel argument must be one of ['102', '110', '120', 'p'], MAGEmin not yet implemented"


            Out_Folder = Path(internal_data_dir(MELTSModel))

            os.makedirs(Out_Folder, exist_ok=True)

            Trainfilename = str(Out_Folder / f'MELTS{MELTSModel}_Trainset{date}{fractionate}{calctype}')
            Validfilename = str(Out_Folder / f'MELTS{MELTSModel}_Validset{date}{fractionate}{calctype}')
            Train_progress_file = f'{Trainfilename}_progress.txt'
            Valid_progress_file = f'{Validfilename}_progress.txt'

            if calctype == 'Cooling':
                MELTER = RM.alphaMELTScooling
            elif calctype == 'Compression':
                MELTER = RM.alphaMELTScompress

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
                    'startT': startT, 'max_liquid_fraction': max_liquid_fraction, 'zeroOxides': ZeroOxides,
                    'Prange': Prange, 'delta': delta, 'Oxygen': Oxygen}

            if full_to_run != 0:
                MELTER(output_file=Trainfilename, **args)

                # Run full GEOROC validation dataset
                args['itercode'] = f'a{int(full_to_run//4)}'
                MELTER(output_file=Validfilename, **args)

            if ultramafics_to_run != 0:
                ultramafics = GEOROC[:,col_dict['MgO']+1]>=25 # MgO above 25

                args['GEOROC'] = GEOROC[ultramafics]
                args['itercode'] = f'u{ultramafics_to_run}'

                MELTER(output_file=Trainfilename, **args)

                args['itercode'] = f'u{int(ultramafics_to_run//4)}'
                MELTER(output_file=Validfilename, **args)

            if mafics_to_run != 0:
                mafics = GEOROC[:,col_dict['MgO']+1]>=5 # MgO above 5

                args['GEOROC'] = GEOROC[mafics]
                args['itercode'] = f'm{mafics_to_run}'

                MELTER(output_file=Trainfilename, **args)

                args['itercode'] = f'm{int(mafics_to_run//4)}'
                MELTER(output_file=Validfilename, **args)


            # Clean up progress files upon completion
            if os.path.exists(Train_progress_file):
                os.remove(Train_progress_file)
                print(f"Training simulations completed. Progress file removed.")

            if os.path.exists(Valid_progress_file):
                os.remove(Valid_progress_file)
                print(f"Validation simulations completed. Progress file removed.")
