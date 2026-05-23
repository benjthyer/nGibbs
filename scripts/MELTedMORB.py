"""Note: This script is for operating MELTS simulations on custom compositions and will not be exported to users in the release. 
The linux virtual environment is loaded by calling from linux home: source ~/melts_env/venv/bin/activate"""

from copy import copy
import sys
import os
# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# Add src to path so we can import modules without src prefix
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from builder.alphamelts.engine import alphamelts_functions # The essential ensemble MELTS functions
from nMELTS.utils.string_utils import  random_char
from nMELTS.utils.math_utils import  grid_sample
from nMELTS.config.constants import OXIDE_MOLAR_MASSES
from builder.indexer import generate_column_headers, DatasetIndexer
from recipes.settings import internal_data_dir, internal_scratch_dir

import numpy as np
import time
import shutil
import pandas as pd
from pathlib import Path

#from EmulatorLibrary import *

# These are the phases that will be simulated and recorded in the output files.
allowed_phases = ['olivine','orthopyroxene','clinopyroxene','spinel','plagioclase','k-feldspar','garnet','nepheline','leucite',
          'biotite','rhm-oxide','alloy-solid','apatite','whitlockite','quartz','tridymite','cristobalite','muscovite','fluid','liquid']

headers = generate_column_headers(allowed_phases)
indexer = DatasetIndexer(headers)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / 'data'

Out_Folder = Path(internal_data_dir('MORB'))
os.makedirs(Out_Folder, exist_ok=True)
    
alphaMELTSLocation = os.path.join(REPO_ROOT, 'src', 'builder', 'alphamelts', 'engine', 'alphamelts-app-2.3.1-linux', 'alphamelts_linux')
# Location to where to put the computed files.
EnsembleLocation = str(internal_scratch_dir())

os.makedirs(EnsembleLocation, exist_ok=True)

ferric_to_ferrous = (2 * OXIDE_MOLAR_MASSES['FeO'] / OXIDE_MOLAR_MASSES['Fe2O3'])

#keys = ['Pressure', 'Temperature', 'fO2', 'SiO2', 'Al2O3', 'CaO', 'MgO', 'Na2O', 'K2O', 'Fe2O3', 'FeO', 'TiO2', 'MnO', 'Cr2O3', 'NiO', 'CoO', 'P2O5', 'H2O']
#keys = ['Pressure', 'Temperature', 'fO2', 'SiO2','TiO2', 'Al2O3', 'FeO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'MnO', 'H2O', 'Cr2O3', 'NiO']
keys = ['Pressure', 'Temperature', 'fO2'] #+ Oxides

CrCondition_str = {'SiO2': 48.68,
'TiO2': 1.01,
'Al2O3':17.64,
'Cr2O3': 0.03,
'FeO': 7.59+(ferric_to_ferrous*0.89),
'MgO': 9.10,
'CaO': 12.45,
'Na2O': 2.65,
'K2O': 0.03,
'P2O5': 0.08,
'H2O': 0.20}

#Peridotite
"""CrCondition_str = {'SiO2': 45.14,
'TiO2': 0.1,
'Al2O3':3.17,
'Cr2O3': 0.42,
'FeO': 7.69+(ferric_to_ferrous*0.89),
'MgO': 39.92,
'CaO': 0.6,
'Na2O': 0.24,
'K2O': 0.01,
'P2O5': 0.1,
'H2O': 0.05}"""

NoCrCondition_str = copy(CrCondition_str)
del NoCrCondition_str['Cr2O3']



for L, MELTSmodel in enumerate(['102']): #Which MELTS models to use
    for j in range(2):
        condition_str = [NoCrCondition_str, CrCondition_str][j]
        suffix = ["NoCr", "Cr"][j]
        #if (j == 0) and (L == 0): # Temp, Skip completed runs during Debugging
        #    continue


        keys = ['Pressure', 'Temperature', 'fO2']
        total = np.sum(list(condition_str.values()))
        for key, val in condition_str.items(): #Renormalize to total = 100
            condition_str[key] = np.round(val*100/total,2)
        print(condition_str)

        key_dict = {}
        for i, k in enumerate(keys):
            key_dict[k] = i



        Fxtal = False
        fxLabel = 'Fxtal' if Fxtal else 'Batch' 
        csv_name = os.path.join(Out_Folder, f'GTMELTS{MELTSmodel}_{suffix}_MORB_{fxLabel}_phasediagrams.csv')

        BatchName = f"{MELTSmodel}Batch"

        in_array = grid_sample([[1,10000,50],[1600,1600,1],[-4,4,5]]) #Phase diagrams
        #in_array = grid_sample([[1,10000,50],[1600,1600,1],[0,0,1]]) #Phase diagrams
        #in_array = grid_sample([[1000,1000,1],[1600,1600,1],[0,0,1]]) #Small Test


        #in_array = grid_sample([[1000,10000,3],[2000,2000,1],[-4,4,5]]) # Harkers
        #in_array = grid_sample([[1000,10000,3],[1600,1600,1],[-4,4,5]]) # Harkers
        #in_array = grid_sample([[1000,8000,4],[1600,1600,1],[-1,1,2]]) # performance benchmarking

        nrow = in_array.shape[0]
        for key, val in condition_str.items():
            keys.append(key)
            in_array = np.append(in_array,np.repeat(val,nrow).reshape(-1,1),axis = 1)


            #in_array = grid_sample([[1000,10000,3],[1600,1600,1],[-4,4,5]], MarsArr.reshape(1,-1))
            #rockname = names[j]
            

        start = time.time()
        batch_size = int(50)
        total_batches = np.ceil(in_array.shape[0]/batch_size)
        assert total_batches == int(total_batches)

        for batch in np.arange(total_batches).astype(int): # Handle remainder at final batch
            if (batch+1)*batch_size > in_array.shape[0]:
                end = in_array.shape[0]
                sims = end-(batch*batch_size)
                for simno in np.arange(sims,batch_size):
                    if os.path.exists(f'{Out_Folder}/Simulation{simno}'):
                        shutil.rmtree(f'{Out_Folder}/Simulation{simno}') # Clear unused directories so that they are not read by import_MELTS_components
            else:
                end = (batch+1)*batch_size
            batchname = np.empty(end-(batch_size*batch), dtype=object)
            batchname[:] = BatchName

            alphamelts_functions.forward_ensemble(in_array[(batch*batch_size):end], keys, only_phases=allowed_phases, batchname = batchname, 
                                            end = 800, EnsembleLocation=EnsembleLocation, WSL = True, compression=False, delta = -1, fxtal=Fxtal)
            for j, name in enumerate(batchname):
                    i = (batch*batch_size) + j
                    batchname[j] = f"{i}:{random_char(4)}:{name}" # Put PTX Index in metadata to link one simulation
            alphamelts_functions.import_MELTS_components(EnsembleLocation=EnsembleLocation, batchname=batchname, indexer=indexer, fO2Arr=in_array[(batch*batch_size):end,2],dataname = csv_name)#, dataname = f'Applications/{rockname}_BatchMELTS.csv')

        print(f"Ordered, Completed, and Read {nrow} simulations in {time.time()-start} seconds")