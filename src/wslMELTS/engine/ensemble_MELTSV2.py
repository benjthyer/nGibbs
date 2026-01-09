"""UPDATED APRIL 2025, USING SMALLER DATA STORAGE AND DIFFERENT, CLEANER PHASE INDICES"""
#from doctest import OutputChecker
import os
#import sys, os
import shutil
import time
import numpy as np
import pandas as pd
from pathlib import Path
from io import StringIO
import re
import pickle
#import random
import subprocess

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# You Need GNU parallel to run this! https://build.opensuse.org/package/show/home:tange/parallel

alphaMELTSLocation = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'alphamelts-app-2.3.1-linux')
# Location to where to put the computed files.
EnsembleLocation = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GEOROC_SIMS')
#batchname = 'mybatch'
#os.makedirs(EnsembleLocation, exist_ok= True)

def pull_number(string):
    string_number = ''
    for char in string:
        if char in '1234567890.-':
            string_number += char
    try:
        return float(string_number)
    except: 
        return np.nan

def expand_MC(conditions, deviation, length = 20):
    """Give two slices of equal lenth to generate an array with normal distributions of 
    conditions with std deviations as given in 'deviation'. """
    if np.shape(conditions) != np.shape(deviation):
        raise IndexError('Errors and Means unequal size')
    vals = np.shape(conditions)[0]
    expanded_array = np.empty((length, vals))

    for i in range(vals):
        expanded_array[:,i] = np.random.normal(
            conditions[i], deviation[i], length)
    return expanded_array

def AddMELTSLine(MELTSStr, key, val, end = 0, delta = -2):
    """Add a line to a string from two values to build .MELTS file"""
    if key == 'fO2':
        #if val != 0:
        MELTSStr += 'Log fo2 Path: FMQ\n'
        MELTSStr += f'Log fo2 Offset: {val}\n'
    elif key == 'Pressure':
        MELTSStr += f'Initial Pressure: {val}\n'
        MELTSStr += f'Final Pressure: {val}\n'
        MELTSStr += 'Increment Pressure: 0\n'
    elif key == 'Temperature':
        MELTSStr += f'Initial Temperature: {val}\n'
        if end:
            MELTSStr += f'Final Temperature: {end}\n' 
        else:
            MELTSStr += f'Final Temperature: {700}\n' 
        MELTSStr += f'Increment Temperature: {delta}\n'
    elif 'O' in key:
        MELTSStr += f'Initial Composition: {key} {val}\n'
    else:
        MELTSStr += f'Initial Trace: {key} {val}\n'
    return MELTSStr

def AddMELTSLineCompression(MELTSStr, key, val, end = 0, delta=50):
    """Add a line to a string from two values to build .MELTS file. Isothermal Compression runs"""
    if key == 'fO2':
        #if val != 0:
        MELTSStr += 'Log fo2 Path: FMQ\n'
        MELTSStr += f'Log fo2 Offset: {val}\n'
    elif key == 'Pressure':
        #Make pressures depending on what MELTS domain is implied by the value passed here
        if not end:
            if val > 10000:
                beginP = 8000 + (val % 20)
                endP = 45000 - (val % 20) #Maybe pMELTS craps out at 30000 bars???
            if val <= 10000:
                beginP = 1 + (val % 20)
                endP = 12000 - (val % 20)
        else:
            beginP = val
            endP = end
        deltaP = delta
        MELTSStr += f'Initial Pressure: {beginP}\n'
        MELTSStr += f'Final Pressure: {endP}\n'
        MELTSStr += f'Increment Pressure: {deltaP}\n'
    elif key == 'Temperature':
        MELTSStr += f'Initial Temperature: {val}\n'
        MELTSStr += f'Final Temperature: {val}\n' 
        MELTSStr += 'Increment Temperature: 0\n'
    elif 'O' in key:
        MELTSStr += f'Initial Composition: {key} {val}\n'
    else:
        MELTSStr += f'Initial Trace: {key} {val}\n'
    return MELTSStr

def makeMELTSStr(conditions, keys, end = True, fxtal = False, compression = False, delta = -3):
    """Tranform slice of conditions with labels ('keys') into MELTS String"""
    MELTSStr = 'Output: both\n'
    if np.shape(conditions)[0] != np.shape(keys)[0]:
        raise IndexError('Conditions and Keys unequal size')
    #if 'MnO' in keys:
    #    if np.array(conditions)[np.array(keys) == 'MnO'][0] > 0: #and np.array(conditions)[np.array(keys) == 'SiO2'][0] < 54: # Handle glitch in pMELTS (maybe all MELTS) where some Ni is necesary to precipitate olivine
    #        if 'NiO' in keys:
    #            conditions[np.where(np.array(keys) == 'NiO')[0][0]] += 0.000015
    #        else:
    #            MELTSStr = AddMELTSLine(MELTSStr, 'NiO', 0.00001, end)
    for i in range(np.shape(conditions)[0]):
        if compression:
            MELTSStr = AddMELTSLineCompression(MELTSStr, keys[i], conditions[i], end=end,delta=delta)
        else:
            MELTSStr = AddMELTSLine(MELTSStr, keys[i], conditions[i], end=end, delta=delta)
    #MELTSStr +='Suppress: tridymite\n'
    #MELTSStr +='Suppress: sillimanite\n'
    MELTSStr +='Suppress: rutile\n'
    #MELTSStr +='Suppress: alloy-solid\n'
    #MELTSStr +='Suppress: alloy-liquid\n'
    if fxtal:
        MELTSStr += 'mode: fractionate solids\n'
        #for sysName in systemNames:
        #    if sysName not in ['tridymite', 'sillimanite', 'rutile', 'liquid', 'fluid','water']:
        #        MELTSStr += f'Fractionate: {sysName}\n'
    MELTSStr += ''

    return MELTSStr

systemNames = ['liquid',
 'olivine',
 'sphene',
 'garnet',  
 'melilite',
 'orthopyroxene',
 'clinopyroxene',
 'aegirine',
 'aenigmatite',
 'cummingtonite',
 'clinoamphibole',
 'orthoamphibole',
 'hornblende',
 'biotite',
 'muscovite',
 'k-feldspar',
 'plagioclase',
 'quartz',
 'tridymite',
 'cristobalite',
 'nepheline',
 'kalsilite',
 'leucite',
 'corundum',
 'rutile',
 'perovskite',
 'spinel',
 'rhm-oxide',
 'ortho-oxide',
 'whitlockite',
 'apatite',
 'alloy-solid',
 'alloy-liquid',
 'sillimanite']

def suppressAllBut(MELTSStr, phase_names):
    """Suppresses everything except for specified phases by appending lines to MELTS file"""
    for phase in systemNames:
        if phase not in phase_names:
            MELTSStr +=f'Suppress: {phase}\n'
    MELTSStr += '' # I don't know why this is here, but it's in makeMELTSstr so I add it here as well
    return MELTSStr

def forward_ensemble(input_array, keys, batchname, only_phases = None, end = 0, EnsembleLocation = EnsembleLocation, fxtal = False, initializer = 'run-alphamelts.command', WSL = True, compression = False, delta=-3): 
    """Performs ensemble MELTS calculation starting with numpy arrays corresponding to column labels: 'keys'
    Let exclude_phases be a nested list of phase names"""
    #if only_phases:
    #    assert len(only_phases) == np.shape(input_array)[0], "only_phases argument should be a nested list with length equal to rows of input array (number of MELTS simulations)"
    RunAll = '' # The shell script to be passed to the terminal eventually
    simulations = np.shape(input_array)[0]
    if np.shape(input_array)[1] != np.shape(keys)[0]:
        raise IndexError("Condition columns don't match Keys")
    if isinstance(batchname, str):
        batchname = [batchname]*np.shape(input_array)[0]
    for i in range(simulations): # Build folders and prepare terminal command
        dirname = f'Simulation{i}'
        ComputeDir = os.path.join(EnsembleLocation, dirname)
        print(ComputeDir)
        if np.ndim(end):
            endparam = end[i]
        else:
            endparam = end
        if np.ndim(delta):
            deltaparam = delta[i]
        else:
            deltaparam = delta
        MELTSStr = makeMELTSStr(input_array[i,:], keys, end=endparam, fxtal=fxtal, compression=compression, delta=deltaparam)
        if only_phases:
            if 'p' in batchname[i].lower():
                print('No Sillimanite in PMELTS Calculations') # Sillimanite not in pMELTS, breaks calculation. 
                only_phases.append('sillimanite')
            MELTSStr = suppressAllBut(MELTSStr, only_phases)
        #print(ComputeDir)
        if not os.path.exists(ComputeDir):
            #inp = input(f'Making Folder {i}')
            os.makedirs(ComputeDir)
        else:
            for filename in os.listdir(ComputeDir): # This comes through and deletes previous mineral table outputs to avoid mixing simulations
                if 'tbl' in filename:
                    file_path = os.path.join(ComputeDir, filename)
                    os.remove(file_path)
        with open(os.path.join(ComputeDir, 'input.melts'), 'w') as f:
            f.write(MELTSStr)
        #print(batchname)
        #print(ComputeDir)
        #for FileName in ['alphamelts_settings.txt', batchname[i]]:
        for FileName in [ batchname[i]]:
            shutil.copy(os.path.join(Path(__file__).parent.absolute(), FileName), os.path.join(ComputeDir, FileName))
        RunAll += 'cd "' + ComputeDir + '" && "'
        RunAll += os.path.join(alphaMELTSLocation, initializer) + f'" -b {batchname[i]}\n'
        #RunAll += initializer + f'" -f alphamelts_settings.txt -b {batchname[i]}\n'

        #RunAll += 'cd ' + ComputeDir + ' && '
        #RunAll += initializer + f' -f alphamelts_settings.txt -b {batchname[i]}\n'


    # Run the script
    if WSL:
        with open(os.path.join(EnsembleLocation, 'runall.sh'), 'w') as f:
            f.write(RunAll)
        os.system('cd "' + EnsembleLocation + '"; parallel < runall.sh; cd -')
    else: # On Windows we are more creative. 
        commands = RunAll.split('\n')
        active_procs = []
        for command in commands:
            proc = subprocess.Popen(command, shell=True)
            active_procs.append(proc)

            # Limit to 4 concurrent processes
            while len(active_procs) >= 4:
                # Remove finished processes from list
                active_procs = [p for p in active_procs if p.poll() is None]
                time.sleep(0.5)
        for proc in active_procs:
            proc.wait()

################ Data Import ####################


### Zach Gainsforth Functions ###
def GetalphaMELTSSectionAsTxt(data, start):
    """ GetAlphaMELTSSection(): Given the output file from a alphaMELTS calculation, extract just one section.
        Input:
            data (str): The output of the melts calculation (from file alphaMELTS_tbl.txt).
            start (str): The name of the section.
        Output:
            Returns the string containing the entire section except for the line containing the start string.
    """

    # The end of the section is a double <CR>
    stop = '\n\n'
    # We are looking for all the text (any characters) between the start string and \n\n
    reout = re.compile(r'%s.*?%s' % (start, stop), re.S)
    try:
        SectionStr = reout.search(data).group(0)
    except:
        # It is possible that this MELTS computation didn't produce this mineral.  If so, just bail.
        return None

    # This is handling a bug in alphaMELTS where alloy-solid doesn't include a label for the structure column.
    if ('alloy-solid' in start) or ('alloy-liquid' in start) or ('neph' in start) or ('kalsilite' in start):
        SectionStr = SectionStr.replace('formula', 'structure formula')

    return StringIO(SectionStr)


def ReadOnePhaseFromMELTSOutputFile(MELTSData, header):
    # Get the chunk of text for this phase
    DataRaw = GetalphaMELTSSectionAsTxt(MELTSData, header)
   
    if DataRaw is not None:
        # Read text as a CSV and default everything to floats, except for a couple fields that are strings.
        Data = pd.read_csv(DataRaw, header=1, delimiter=' ') 
        for c in Data.columns:
            if c not in ['formula', 'structure', 'neph']:
                Data[c] = Data[c].astype(float)
            else:
                del Data[c]

        # Convert Kelvin to Celcius.
        #Data['Temperature'] -= 273.15
        
        return Data
    else:
        return None
 

### Original Functions and mappings###
try:
    phase_ind =  pickle.load(open('PhaseDict.pkl', 'rb'))
except: # If not existing, create new and write down for editing
    phase_ind = {'Bulk Composition': 0,
    'liquid1': 1,
    'fluid1': 2,
    'olivine1': 3,
    'spinel1': 4,
    'orthopyroxene1': 5,
    'orthopyroxene2': 6,
    'clinopyroxene1': 7,
    'clinopyroxene2': 8} #etc.... Letting it build itself tbh
    pickle.dump(phase_ind, open('PhaseDict.pkl', 'wb'))


chem_ind = {
'Pressure':0, 
'Temperature':1, 
'mass':2,
'SiO2':3,
'Al2O3':4,
'CaO':5,
'MgO':6,
'Na2O':7,
'K2O':8,
'Fe2O3':9, 
'FeO':10,
'TiO2':11,
'MnO':12,
'Cr2O3':13,
'NiO':14,
'CoO':15,
'P2O5':16, 
'H2O':17, 
'CO2':18,
'H':19, 
'S':20, 
'V':21, 
'Cp':22,
'viscosity':23
}

# ALL ITEMS IN COMPONENT INDICES BUT BE REAL COLUMNS IN ALPHAMELTS OUTPUT. THIS DETERMINES WHAT DATA IS SAVED IN LARGE CSV TABLE
component_indices = {
    'System_main': {
        'Pressure': 0,
        'Temperature': 1,
        'logfO2-QFM': 2
    },
    'olivine': {
        'mass (gm)': 3,
        'tephroite': 4,
        'fayalite': 5,
        'ni-olivine': 6,
        'monticellite': 7,
        'forsterite': 8
    },
    'orthopyroxene': {
        'mass (gm)': 9,
        'diopside': 10,
        'clinoenstatite': 11,
        'hedenbergite': 12,
        'alumino-buffonite': 13,
        'buffonite': 14,
        'essenite': 15,
        'jadeite': 16
    },
    'clinopyroxene': {
        'mass (gm)': 17,
        'diopside': 18,
        'clinoenstatite': 19,
        'hedenbergite': 20,
        'alumino-buffonite': 21,
        'buffonite': 22,
        'essenite': 23,
        'jadeite': 24
    },
    'spinel': {
        'mass (gm)': 25,
        'chromite': 26,
        'hercynite': 27,
        'magnetite': 28,
        'spinel': 29,
        'ulvospinel': 30
    },
    'plagioclase': {
        'mass (gm)': 31,
        'albite': 32,
        'anorthite': 33,
        'sanidine': 34 # alphamelts 2.3.1 uses exlusively 'sanidine'. Previous versions used 'highsanidine'
    },
    'alkali-feldspar': {
        'mass (gm)': 35,
        'albite': 36,
        'anorthite': 37,
        'sanidine': 38
    },
    'k-feldspar': {
        'mass (gm)': 35,
        'albite': 36,
        'anorthite': 37,
        'sanidine': 38
    },
    'garnet': {
        'mass (gm)': 39,
        'almandine': 40,
        'grossular': 41,
        'pyrope': 42
    },
    'nepheline': {
        'mass (gm)': 43,
        'na-nepheline': 44,
        'k-nepheline': 45,
        'vc-nepheline': 46,
        'ca-nepheline': 47
    },
    'leucite': {
        'mass (gm)': 48,
        'leucite': 49,
        'analcime': 50,
        'na-leucite': 51
    },
    'biotite': {
        'mass (gm)': 52,
        'annite': 53,
        'phlogopite': 54
    },
    'rhm-oxide': {
        'mass (gm)': 55,
        'geikielite': 56,
        'hematite': 57,
        'ilmenite': 58,
        'pyrophanite': 59,
        'corundum':60 
    },
    'alloy-solid': {
        'mass (gm)': 61,
        'Fe-metal': 62,
        'Ni-metal': 63
    },
    'alloy-liquid': {
        'mass (gm)': 64,
        'Fe-liquid': 65,
        'Ni-liquid': 66
    },
    'apatite': {
        'mass (gm)': 67
    },
    'whitlockite': {
        'mass (gm)': 68
    },
    'quartz': {
        'mass (gm)': 69
    },
    'tridymite': {
        'mass (gm)': 70
    },
    'cristobalite': {
        'mass (gm)': 71
    },
    'amphibole': {
        'mass (gm)': 72
    },
    'muscovite': {
        'mass (gm)': 73
    },
    'fluid': {
        'mass (gm)': 74
    },
    'water': {
        'mass (gm)': 74
    },
    'melts-liquid': {
        'liq mass (gm)': 75,
        'wt% SiO2': 76,
        'wt% TiO2': 77,
        'wt% Al2O3': 78,
        'wt% FeO': 79,
        'wt% MgO': 80,
        'wt% CaO': 81,
        'wt% Na2O': 82,
        'wt% K2O': 83,
        'wt% P2O5': 84,
        'wt% MnO': 85,
        'wt% H2O': 86,
        'wt% Cr2O3': 87,
        'wt% NiO': 88,
        'wt% Fe2O3': 89
    }
}

#### WILL Break if new names for single phase is added
database_headers = []
for pha, val in component_indices.items():
    if pha not in ['alkali-feldspar', 'water']:
        for com in list(val.keys()):
            database_headers.append(f"{com}({pha})")

component_indices['System_main']['viscosity'] = 90
database_headers.append("viscocity(System_main)")
component_indices['System_main']['H'] = 91
database_headers.append("H(System_main)")
component_indices['System_main']['Cp'] = 92
database_headers.append("Cp(System_main)")
component_indices['System_main']['S'] = 93
database_headers.append('S(system_main)')
component_indices['System_main']['V'] = 94
database_headers.append('V(System_main)')
component_indices['System_main']['dVdP*10^6'] = 95
database_headers.append('dVdP*10^6(System_main)')
component_indices['System_main']['dVdT*10^6'] = 96
database_headers.append('dVdP*10^6(System_main)')
# Now add rho and viscosity fields sequentially

next_index = 97

# List of phases (in order of appearance)
phases_in_order = [
    'olivine', 'orthopyroxene', 'clinopyroxene', 'spinel',
    'plagioclase', 'alkali-feldspar', 'garnet', 'nepheline', 'leucite',
    'biotite', 'rhm-oxide', 'alloy-solid', 'alloy-liquid', 'apatite', 'whitlockite', 'quartz', 'tridymite', 'cristobalite',
    'amphibole', 'muscovite', 'fluid'
]

# Add density, enthalpy field
for phase in phases_in_order:
    component_indices[phase]['rho (gm/cc)'] = next_index
    next_index += 1
    database_headers.append(f"{'rho (gm/cc)'}({phase})")
    component_indices[phase]['H (kJ)'] = next_index
    next_index += 1
    database_headers.append(f"{'H'}({phase})")
    component_indices[phase]['S (J/K)'] = next_index
    next_index += 1
    database_headers.append(f"{'S (J/K)'}({phase})")
    component_indices[phase]['V (cc)'] = next_index
    next_index += 1
    database_headers.append(f"{'V (cc)'}({phase})")

# Add viscosity field for melt only
component_indices['melts-liquid']['liq rho (gm/cc)'] = next_index
component_indices['melts-liquid']['liq vis (log 10 poise)'] = next_index + 1
component_indices['melts-liquid']['liq H (kJ)'] = next_index + 2
component_indices['melts-liquid']['liq S (J/K)'] = next_index + 3
component_indices['melts-liquid']['liq V (cc)'] = next_index + 4
database_headers.append('rho (gm/cc)(melts-liquid)')
database_headers.append('liq vis (log 10 poise)')
database_headers.append('liq H (kJ)')
database_headers.append('liq S (J/K)')
database_headers.append('liq V (cc)')

mass_indices = [] #exclude amphibole
for phase, props in component_indices.items():
    if phase in (phases_in_order + ['melts-liquid']):
        for key, ind in list(props.items()):
            if 'mass' in key:
                mass_indices.append(ind)
mass_indices = np.unique(mass_indices)

def append_phase(phase, phase_tbl, MELTSobj):
    """phase_tbl is a pandas dataframe, 
    MELTSobj is a numpy redering of a single MELTS Simulation
    phase is the string phase label (i.e. 'apatite1') """
    # Turn float indices retrieved from MELTS to integers for python indexing
    indices = list(map(int, phase_tbl['index'].to_numpy() - 1)) 
    del phase_tbl['index'] 
    for label in phase_tbl.columns:
        MELTSobj[indices, int(phase_ind[phase]), int(chem_ind[label])] = phase_tbl[label].to_numpy()
    return MELTSobj


def import_MELTS_components(EnsembleLocation, batchname, fO2Arr = None, dataname = 'DefaultMELTSstorage.csv'): 
    """New as of 04/03/2025: Load Components to component object
    Handle distinction in pMELTS (as of alphamelts 2.3.1) where plagioclase K component is sanidine, not high sanidine."""
    contents = os.listdir(EnsembleLocation)
    folders = len(contents)-1
    sim_metadata_name = dataname.split('.')[0] + '.txt'
    metadata = []
    workbase = np.empty((0, 1+max(list(component_indices.values())[-1].values())))
    #database_headers = database_headers

    if not os.path.exists(sim_metadata_name):
        with open('emptyfile.txt', 'w') as f:
            pass
    if not os.path.exists(dataname):
        newbase = pd.DataFrame(columns = database_headers)
        newbase.to_csv(dataname, index = False)
    faultIDs = []
    for folderNo in range(folders):
        folder = 'Simulation'+str(folderNo)
        run = os.path.join(EnsembleLocation, folder)
        tablename = 'System_main_tbl.txt'
        fault = False
        try:
            table = np.genfromtxt(os.path.join(run, tablename), skip_header=2)
            nrows = np.shape(table)[0]
            print(np.shape(table))
            if table.ndim <= 1:
                go = False
            else:
                go = True
        except:
            go = False
            print(f"Simulation{folderNo} FAILED and was not read!")
        if go:
            fault = False
            working_database_rows = []
            for nr in range(nrows):
                working_database_rows.append(batchname[folderNo]+f' {nr}') # add step index
            meltsobj = np.zeros((nrows, 1+max(list(component_indices.values())[-1].values()))) # Prepare empty container for data
            for tablename in os.listdir(run):
                if 'tbl' in tablename and tablename not in ['Solid_comp_tbl.txt','Phase_vol_tbl.txt','Phase_mass_tbl.txt','Phase_main_tbl.txt','Liquid_comp_tbl.txt','Bulk_comp_tbl.txt']: # METLS output we are not interested in
                    phasename = tablename.split('tbl')[0][:-1]
                    if phasename in ['orthoamphibole', 'clinoamphibole', 'hornblende']:
                        phasename = 'amphibole'
                    skipline = 1
                    delim = ','
                    if tablename == 'System_main_tbl.txt': # Formatting for the bulk reports is space deliminated and contains an additional header
                        skipline = 2
                        delim = ' '
                    with open(os.path.join(run, tablename), 'r') as text:
                        headers = (text.read().split('\n')[skipline-1]).split(delim)
                        melt_dict = {}
                        for i, header in enumerate(headers): 
                            melt_dict[header] = i
                    try:
                        table = np.genfromtxt(os.path.join(run, tablename), delimiter=delim, skip_header=skipline)
                    except: 
                        print(f"Bad data table for {phasename} in {folder}. Skipping it!")
                        fault = True
                        break
                    print(phasename)
                    print(folder)
                    print(f"Dims of table: {table.ndim}")
                    print(np.shape(table))
                    if len(np.shape(table)) <= 1:
                        table = np.atleast_2d(table)
                        print(f'Reshaping table! {np.shape(table)}')
                    try:
                        rowsfill = table[:,0].astype(int)-1 # Get indices from METLS table
                    except:
                        print(f"Bad data table for {phasename} in {folder}. Skipping it!")
                        fault = True
                        break
                    if phasename in list(component_indices.keys()): 
                        compnames = list(component_indices[phasename].keys()) 
                        for fillname in compnames: # ALL ITEMS IN COMPONENT INDICES BUT BE REAL COLUMNS IN ALPHAMELTS OUTPUT
                            if fillname == 'corundum' and 'pBatch' in batchname[folderNo]: # pMELTS does not have corundum component in rhm-oxides as of alphamelts 2.3.1
                                continue
                            if fillname == 'logfO2-QFM': #Handle MELTS variable column name for the buffered fO2
                                if fO2Arr is None:
                                    for key in list(melt_dict.keys()):
                                        if 'QFM' in key:
                                            fO2key = key
                                            print(fO2key)
                                            delta = pull_number(key[4:]) 
                                            print(delta)
                                            if np.isnan(delta):
                                                delta = 0
                                else: 
                                    delta = fO2Arr[folderNo]
                                #meltsobj[rowsfill,component_indices[phasename][fillname]] = table[:,melt_dict[fO2key]] + delta
                                try:
                                    meltsobj[rowsfill,component_indices[phasename][fillname]] =  delta
                                except:
                                    fault = True
                                    faultIDs.append(folderNo)
                            elif phasename == 'amphibole':
                                try:
                                    meltsobj[rowsfill,component_indices[phasename][fillname]] += table[:,melt_dict[fillname]] 
                                except:
                                    fault = True
                                    faultIDs.append(folderNo)
                            else: 
                                try:
                                    meltsobj[rowsfill,component_indices[phasename][fillname]] = table[:,melt_dict[fillname]] # Populate table 
                                except:
                                    fault = True
                                    faultIDs.append(folderNo)
                    else: 
                        for i, row in enumerate(rowsfill): # If a phase comes up that is not recorded in the table, put its mass and name in the metadata
                            try:
                                working_database_rows[row] += f" {phasename}:{table[i,melt_dict['mass (gm)']]}"
                            except:
                                fault = True
                                faultIDs.append(folderNo)
            assert len(working_database_rows) == meltsobj.shape[0], f'Unequal Length run metadata and meltobj rows sim: {folderNo}'
            #assert len(np.unique(meltsobj[:,1])) == 1, f'Multiple Temperatures: {folderNo}' #FOR COMPRESSION RUNS ONLY, DEBUGGING,
            #assert len(np.unique(meltsobj[:,2])) == 1, f'Multiple fO2s: {folderNo}'#FOR COMPRESSION RUNS ONLY, DEBUGGING

            if not fault:
                for wdr in working_database_rows:
                    metadata.append(wdr + '\n')
                workbase = np.vstack([workbase,meltsobj])
        if fault or not go:
            faultIDs.append(folderNo) # Record failures for exfail function
            print(f"FAILURE AT FOLDER {folderNo}")
    if len(metadata) != np.shape(workbase)[0]:
        raise Exception('Metadata different length than rows of csv!')
    
    #New as of 10/08/25: Filter out much of the superliquidus assemblage to save space, balance dataset

    # Step 1: Identify nonzero rows in selected columns
    nonzero_mask = (workbase[:, mass_indices[:-1]] != 0).any(axis=1)
    print(nonzero_mask.shape)
    print(workbase.shape)
    # Step 2: Separate indices
    nonzero_indices = np.where(nonzero_mask)[0]
    zero_indices = np.where(~nonzero_mask)[0]

    # Step 3: Choose one-fourth as many zero rows as nonzero rows to add back
    n_add = len(nonzero_indices) // 4
    if len(zero_indices) > 0:
        add_back_indices = np.random.choice(zero_indices, size=min(n_add, len(zero_indices)), replace=False)
    else:
        add_back_indices = np.array([], dtype=int)

    final_indices = np.sort(np.concatenate([nonzero_indices, add_back_indices]))

    # Step 5: Extract subset and matching metadata
    filtered_workbase = workbase[final_indices]
    print(len(metadata))

    print(len(final_indices))
    filtered_rows = [metadata[L] for L in final_indices]

    with open(sim_metadata_name, 'a') as f:
        f.writelines(filtered_rows)
    workDF = pd.DataFrame(filtered_workbase)

    workDF.to_csv(dataname, mode = 'a', index = False, header = False)
    return np.unique(faultIDs)

def pick_exsolution_failure(EnsembleLocation, input_array, keys, batchname=['exampleWOW']*8, dataname = '2_phasePTX.csv', faultIDs = []): 
    """saves 2+ phase conditions (PTXfO2) for later study"""
    
    sim_metadata_name = dataname.split('.')[0] + '.txt'
    metadata = []
    workbase = np.empty((0, 1+len(keys)))
    if not os.path.exists(dataname):
        database_headers = keys + ['Failed']
        df = pd.DataFrame(columns = database_headers)
        df.to_csv(dataname, index = False)
    if not os.path.exists(sim_metadata_name):
        with open('emptyfile.txt', 'w') as f:
            pass
    contents = os.listdir(EnsembleLocation)
    folders = len(contents)-1
    for folderNo in range(folders):
        if folderNo in faultIDs:
            workbase = np.vstack([workbase, np.append(input_array[folderNo],[1])]) # Record Failures from import_MELTS
            metadata.append(batchname[folderNo] + '\n')
            continue
        folder = 'Simulation'+str(folderNo)
        proceed = True
        try:
            main_file = os.path.join(EnsembleLocation, folder, 'Phase_main_tbl.txt')
            with open(main_file, 'r') as myfile:
                mainstr=myfile.read() + '\n'
        except: # Record failures 
            metadata.append(batchname[folderNo] + '\n')
            workbase = np.vstack([workbase, np.append(input_array[folderNo],[1])])
            proceed = False
        if proceed:
            chunks = mainstr.split("\n\n")
            del chunks[-1]
            save = False #Only save exsolutions
            working_label = batchname[folderNo]
            for chunk in chunks:
                ind = 0
                lines = chunk.split('\n')
                while chunk.split('\n')[ind] == '':
                    ind += 1 #Ignore headspace
                header = lines[ind]
                phase = header.split()[0]
                if pull_number(phase) in [2,3,4,5,6]: # Detect plural phases
                    col = ReadOnePhaseFromMELTSOutputFile(mainstr, header).to_numpy()[:,0]
                    save = True
                    working_label += f' {phase}-{int(min(col))}-{int(max(col))}'
            if save:
                metadata.append(working_label + '\n')
                workbase = np.vstack([workbase, np.append(input_array[folderNo],[0])])
    if len(metadata) != np.shape(workbase)[0]:
        raise Exception('Metadata different length than rows of csv!')
    with open(sim_metadata_name, 'a') as f:
        f.writelines(metadata)
    workDF = pd.DataFrame(workbase)
    workDF.to_csv(dataname, mode = 'a', index = False, header = False)


def import_MELTS(EnsembleLocation=EnsembleLocation, fail_placeholders = False, dataname = None): 
    contents = os.listdir(EnsembleLocation)
    folders = len(contents)-1
    try:
        database = pickle.load(open(dataname, 'rb'))
        init_data = False
    except:
        init_data = True
        #if dataname == None:
        #    dataname = 'unnamed_MELTS_sims.pkl'
    for folderNo in range(folders):
        folder = 'Simulation'+str(folderNo)
        if 'Simulation' in folder:
            #print(folder)
            main_file = os.path.join(EnsembleLocation, folder, 'Phase_main_tbl.txt')
            BC_file = os.path.join(EnsembleLocation, folder, 'Bulk_comp_tbl.txt')
            cont = True # Go forward with function if loading successful 
            try:
                with open(main_file, 'r') as myfile:
                    mainstr=myfile.read() + '\n' 
                    #Add space to end of string to so last section is readable to GetAlphaMELTSSectionAsTxt
            except:
                print('Failed to open ' + folder + '.  Likely the MELTS computation failed in this portion of phase space.  Skipping.')
                cont = False
            
            if cont:
                # Identify Phases and headers to retrieve tables from MELTS Simultations
                chunks = mainstr.split("\n\n")
                del chunks[-1]
                headers = []
                phases = []

                for chunk in chunks:
                    ind = 0
                    lines = chunk.split('\n')
                    while chunk.split('\n')[ind] == '':
                        ind += 1 #Ignore headspace
                    header = lines[ind]
                    headers.append(header)
                    phase = header.split()[0]
                    if phase == 'index': 
                        cont = False
                        break
                    if phase not in phase_ind.keys() and phase != 'index':
                        phase_ind[phase] = len(phase_ind)
                        pickle.dump(phase_ind, open('PhaseDict.pkl', 'wb')) #Save new phase
                if cont == False:
                    if fail_placeholders:
                        MELTSobj = f'Failed to open {folder}'
                        print("Placeholder saved")
                    continue
                
                with open(BC_file, 'r') as myBCfile:
                    BCstr=StringIO(myBCfile.read())
                raw_pd = pd.read_csv(BCstr, header=1, delimiter=' ').iloc[:,:-1]

                del raw_pd['index']
                raw_np = raw_pd.to_numpy()
                timesteps = np.shape(raw_np)[0]
                MELTSobj = np.zeros((timesteps, len(phase_ind), len(chem_ind))) 
                for i, label in enumerate(raw_pd.columns): # Move columns to proper spots
                    MELTSobj[:,0,chem_ind[label]] = raw_np[:,i] 

                for header in headers:
                    table = ReadOnePhaseFromMELTSOutputFile(mainstr, header)
                    phase = header.split()[0]
                    MELTSobj = append_phase(phase, table, MELTSobj) # Attach phase to array
            elif fail_placeholders:
                MELTSobj = f'Failed to open {folder}'
                print("Placeholder saved")
            if init_data: 
                database = list([MELTSobj])
                init_data = False
            else:    
                database.append(MELTSobj) # First index references which simulation you are on
    if dataname != None: # By Default, do not write file. If there is a name presented, the function will save/ append accordingly.
        pickle.dump(database, open(dataname, 'wb'))
    return database

    
#keys = ['SiO2', 'TiO2',	'Al2O3', 'FeO',	'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'MnO', 'H2O', 'Cr2O3',	'NiO']
#batchname = 'mybatch'