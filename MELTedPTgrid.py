import ensemble_MELTSV2 # The essential ensemble MELTS functions
import numpy as np
import random
import pickle
import os   
import string
import time
import molmass as ms
import shutil
import pandas as pd
from EmulatorLibrary import *


allowed_phases = ['olivine','orthopyroxene','clinopyroxene','spinel','plagioclase','k-feldspar','garnet','nepheline','leucite','biotite',
                  'rhm-oxide','alloy-solid','apatite','whitlockite','quartz','tridymite','cristobalite','muscovite','fluid','liquid']
Out_Folder = 'MORB_SIMS'



start = time.time()

def random_char(y):
       return ''.join(random.choice(string.ascii_letters) for x in range(y))

def pull_number(string):
    string_number = ''
    for char in string:
        if char in '1234567890.-':
            string_number += char
    try:
        return float(string_number)
    except: 
        return np.nan
    
def grid_sample(params, table=np.array([])):
    """Generates a numpy array grid sample recursively for arbitrary parameters.
    Let params be a nested list, with each sublist of [min, max, len] passed to np.linspace.
    Order of params determines column order in the output table."""
    
    params = list(params)  # Copy to avoid side-effects
    param = params.pop()
    new_col = np.linspace(*param).reshape((-1,1))
    
    if not table.shape[0]:
        table = new_col
    else:
        table = np.append(np.repeat(new_col, table.shape[0], axis=0),
                          np.tile(table, (new_col.shape[0], 1)), axis=1)
    
    if len(params):
        return grid_sample(params, table)
    else:
        return table

alphaMELTSLocation = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alphamelts')
# Location to where to put the computed files.
EnsembleLocation = os.path.join(os.path.dirname(os.path.abspath(__file__)), Out_Folder)

if Out_Folder not in os.listdir():
    os.makedirs(EnsembleLocation)

ferric_to_ferrous = (2*ms.Formula('FeO').mass/ms.Formula('Fe2O3').mass)

#keys = ['Pressure', 'Temperature', 'fO2', 'SiO2', 'Al2O3', 'CaO', 'MgO', 'Na2O', 'K2O', 'Fe2O3', 'FeO', 'TiO2', 'MnO', 'Cr2O3', 'NiO', 'CoO', 'P2O5', 'H2O']
#keys = ['Pressure', 'Temperature', 'fO2', 'SiO2','TiO2', 'Al2O3', 'FeO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'MnO', 'H2O', 'Cr2O3', 'NiO']
keys = ['Pressure', 'Temperature', 'fO2'] #+ Oxides

condition_str = {'SiO2': 48.68,
'TiO2': 1.01,
'Al2O3':17.64,
#'Cr2O3': 0.03,
'FeO': 7.59+(ferric_to_ferrous*0.89),
'MgO': 9.10,
'CaO': 12.45,
'Na2O': 2.65,
'K2O': 0.03,
'P2O5': 0.08,
'H2O': 0.20}


#Renormalize
total = np.sum(list(condition_str.values()))
for key, val in condition_str.items():
    condition_str[key] = np.round(val*100/total,2)
print(condition_str)

key_dict = {}
for i, k in enumerate(keys):
    key_dict[k] = i

#in_array = grid_sample([[1,10000,50],[1600,1600,1],[-5,5,5]]) #Phase diagrams
in_array = grid_sample([[1000,10000,3],[1600,1600,1],[-4,4,5]]) # Harkers
#in_array = grid_sample([[1000,8000,8],[1600,1600,1],[-1,1,1]])

nrow = in_array.shape[0]
for key, val in condition_str.items():
    keys.append(key)
    in_array = np.append(in_array,np.repeat(val,nrow).reshape(-1,1),axis = 1)


    #in_array = grid_sample([[1000,10000,3],[1600,1600,1],[-4,4,5]], MarsArr.reshape(1,-1))
    #rockname = names[j]
    


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
    batchname[:] = 'Crustal'

    ensemble_MELTSV2.forward_ensemble(in_array[(batch*batch_size):end], keys, only_phases=allowed_phases, batchname = batchname, 
                                    end = 800, EnsembleLocation=EnsembleLocation, WSL = True, compression=False, delta = -2, fxtal=True)
    for j, name in enumerate(batchname):
            i = (batch*batch_size) + j
            batchname[j] = f"{i}:{random_char(4)}:{name}" # Put PTX Index in metadata to link one simulation
    ensemble_MELTSV2.import_MELTS_components(EnsembleLocation=EnsembleLocation, batchname=batchname, fO2Arr=in_array[(batch*batch_size):end,2],dataname = f"MORB_Phase_Diagram_PTgridNoCrHarkersFxtal.csv")#, dataname = f'Applications/{rockname}_BatchMELTS.csv')

print(f"Ordered, Completed, and Read {nrow} simulations in {time.time()-start} seconds")