"""
Ensemble MELTS simulation execution code.

Contains functions for running ensemble MELTS calculations and importing results.
"""

import os
import shutil
import time
import numpy as np
import pandas as pd
from pathlib import Path
from io import StringIO
import re
import subprocess
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Import file building functions
from .melts_file_builder import (
    makeMELTSStr,
    suppressAllBut,
    #alphaMELTSLocation,
    #EnsembleLocation,
    #pull_number
)

from ...nMELTS.utils.string_utils import pull_number
from ...nMELTS.config.indexer import DatasetIndexer

# You Need GNU parallel to run this! https://build.opensuse.org/package/show/home:tange/parallel

EnsembleLocation = None # let this error if not set
alphaMELTSLocation = os.path.join(Path(__file__).parent.absolute(), 'alphamelts-app-2.3.1-linux')

### Zach Gainsforth Functions ### https://github.com/ZGainsforth/alphaMELTSEnsemble?tab=readme-ov-file
def GetalphaMELTSSectionAsTxt(data, start):
    """
    GetAlphaMELTSSection(): Given the output file from a alphaMELTS calculation, extract just one section.
    
    Parameters:
    -----------
    data : str
        The output of the melts calculation (from file alphaMELTS_tbl.txt).
    start : str
        The name of the section.
        
    Returns:
    --------
    StringIO or None
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
    """
    Read one phase from MELTS output file.
    
    Parameters:
    -----------
    MELTSData : str
        MELTS output data as string
    header : str
        Phase header string
        
    Returns:
    --------
    pd.DataFrame or None
        Phase data as DataFrame
    """
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

        return Data
    else:
        return None

### Here the functions are my own, or highly modified from Zach Gainsforth's functions
################ Ensemble Execution Functions ####################

def _clean_workspace(EnsembleLocation):
    """
    Delete all directories and files in the EnsembleLocation workspace.
    
    Parameters
    ----------
    EnsembleLocation : str
        Path to the workspace directory to clean
    """
    if os.path.exists(EnsembleLocation):
        for item in os.listdir(EnsembleLocation):
            item_path = os.path.join(EnsembleLocation, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception as e:
                print(f"Warning: Could not delete {item_path}: {e}")


def forward_ensemble(input_array, keys, batchname, only_phases=None, end=0, EnsembleLocation=EnsembleLocation,
                     fxtal=False, initializer='run-alphamelts.command', WSL=True, compression=False, delta=-3):
    """
    Performs ensemble MELTS calculation starting with numpy arrays corresponding to column labels: 'keys'.
    
    Parameters:
    -----------
    input_array : np.ndarray
        Array of input conditions, shape (n_simulations, n_conditions)
    keys : np.ndarray or list
        Column labels corresponding to conditions
    batchname : str or list
        Batch name(s) for simulations
    only_phases : list, optional
        List of phase names to allow (all others suppressed)
    end : float or np.ndarray, default=0
        End temperature value(s)
    EnsembleLocation : str, default=EnsembleLocation
        Location for computed files
    fxtal : bool, default=False
        Whether to enable fractional crystallization
    initializer : str, default='run-alphamelts.command'
        Initializer script name
    WSL : bool, default=True
        Whether running on WSL (uses GNU parallel) or Windows. Windows is deprecated.
    compression : bool, default=False
        Whether this is a compression run
    delta : float or np.ndarray, default=-3
        Temperature increment(s)
    """
    # Clean workspace before starting new simulations
    _clean_workspace(EnsembleLocation)
    
    # Ensure EnsembleLocation exists
    os.makedirs(EnsembleLocation, exist_ok=True)
    
    RunAll = ''  # The shell script to be passed to the terminal eventually
    simulations = np.shape(input_array)[0]
    if np.shape(input_array)[1] != np.shape(keys)[0]:
        raise IndexError("Condition columns don't match Keys")
    if isinstance(batchname, str):
        batchname = [batchname] * np.shape(input_array)[0]
    
    for i in range(simulations):  # Build folders and prepare terminal command
        dirname = f'Simulation{i}'
        ComputeDir = os.path.join(EnsembleLocation, dirname)
        print(ComputeDir)
        if np.ndim(end): # These can be scalars or sample-wise arrays.
            endparam = end[i]
        else:
            endparam = end
        if np.ndim(delta):
            deltaparam = delta[i]
        else:
            deltaparam = delta
        MELTSStr = makeMELTSStr(input_array[i, :], keys, end=endparam, fxtal=fxtal,
                                compression=compression, delta=deltaparam)
        if only_phases:
            #if 'p' in batchname[i].lower(): # This asks: Are we doing a PMELTS calculation?
                #print('No Sillimanite in PMELTS Calculations')  # Sillimanite not in pMELTS, breaks calculation, enforce later...
                #only_phases.append('sillimanite') # Vestigial for when this argument held phases to exclude, now it is phases that are kept
            MELTSStr = suppressAllBut(MELTSStr, only_phases)
        
        # Create simulation directory (workspace is already clean)
        os.makedirs(ComputeDir, exist_ok=True)
        
        with open(os.path.join(ComputeDir, 'input.melts'), 'w') as f:
            f.write(MELTSStr)
        
        for FileName in [batchname[i]]:
            shutil.copy(os.path.join(Path(__file__).parent.parent.absolute(), 'batch', FileName),
                       os.path.join(ComputeDir, FileName))
        RunAll += 'cd "' + ComputeDir + '" && "'
        RunAll += os.path.join(alphaMELTSLocation, initializer) + f'" -b {batchname[i]}\n'

    # Run the script
    if WSL:
        with open(os.path.join(EnsembleLocation, 'runall.sh'), 'w') as f:
            f.write(RunAll)
        os.system('cd "' + EnsembleLocation + '"; parallel < runall.sh; cd -')
    else:  # On Windows we are more creative.
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


################ Data Import Functions ####################
# Parsing MELTS output to .csv database
def import_MELTS_components(EnsembleLocation, batchname, indexer, fO2Arr=None, dataname='DefaultMELTSstorage.csv'):
    """
    New as of 04/03/2025: Load Components to component object.
    
    Handle distinction in pMELTS (as of alphamelts 2.3.1) where plagioclase K component
    is sanidine, not high sanidine.
    
    Parameters:
    -----------
    EnsembleLocation : str
        Location of simulation folders
    batchname : list
        Batch names for each simulation
    indexer : DatasetIndexer
        DatasetIndexer object containing MELTS_indices, database_headers, and mass_indices
    fO2Arr : np.ndarray, optional
        Array of fO2 offsets
    dataname : str, default='DefaultMELTSstorage.csv'
        Output CSV filename
        
    Returns:
    --------
    np.ndarray
        Array of fault IDs
    """
    contents = os.listdir(EnsembleLocation)
    folders = len(contents) - 1
    sim_metadata_name = dataname.split('.')[0] + '.txt'
    metadata = []
    workbase = np.empty((0, indexer.get_max_index()+1))

    if not os.path.exists(sim_metadata_name):
        with open('emptyfile.txt', 'w') as f:
            pass
    if not os.path.exists(dataname):
        newbase = pd.DataFrame(columns=indexer.database_headers)
        newbase.to_csv(dataname, index=False)
    faultIDs = []
    
    for folderNo in range(folders):
        folder = 'Simulation' + str(folderNo)
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
                working_database_rows.append(batchname[folderNo] + f' {nr}')  # add step index
            meltsobj = np.zeros((nrows, indexer.get_max_index()+1))  # Prepare empty container
            
            for tablename in os.listdir(run):
                if 'tbl' in tablename and tablename not in ['Solid_comp_tbl.txt', 'Phase_vol_tbl.txt',
                                                             'Phase_mass_tbl.txt', 'Phase_main_tbl.txt',
                                                             'Liquid_comp_tbl.txt', 'Bulk_comp_tbl.txt']:
                    phasename = tablename.split('tbl')[0][:-1]
                    if phasename in ['orthoamphibole', 'clinoamphibole', 'hornblende']:
                        phasename = 'amphibole'
                    skipline = 1
                    delim = ','
                    if tablename == 'System_main_tbl.txt':  # Formatting for the bulk reports is space delimited
                        skipline = 2
                        delim = ' '
                    with open(os.path.join(run, tablename), 'r') as text:
                        headers = (text.read().split('\n')[skipline - 1]).split(delim)
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
                        rowsfill = table[:, 0].astype(int) - 1  # Get indices from MELTS table
                    except:
                        print(f"Bad data table for {phasename} in {folder}. Skipping it!")
                        fault = True
                        break
                    if phasename in list(indexer.MELTS_indices.keys()):
                        compnames = list(indexer.MELTS_indices[phasename].keys())
                        for fillname in compnames:
                            if fillname == 'corundum' and 'pBatch' in batchname[folderNo]:
                                # pMELTS does not have corundum component in rhm-oxides as of alphamelts 2.3.1
                                continue
                            if fillname == 'logfO2-QFM':  # Handle MELTS variable column name for the buffered fO2
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
                                try:
                                    meltsobj[rowsfill, indexer.MELTS_indices[phasename][fillname]] = delta
                                except:
                                    fault = True
                                    faultIDs.append(folderNo)
                                """elif phasename == 'amphibole':
                                try:
                                    meltsobj[rowsfill, indexer.MELTS_indices[phasename][fillname]] += table[:, melt_dict[fillname]]
                                except:
                                    fault = True
                                    faultIDs.append(folderNo)"""
                            else:
                                try:
                                    meltsobj[rowsfill, indexer.MELTS_indices[phasename][fillname]] = table[:, melt_dict[fillname]]
                                except:
                                    fault = True
                                    faultIDs.append(folderNo)
                    else:
                        # If a phase comes up that is not recorded in the table, put its mass and name in the metadata
                        for i, row in enumerate(rowsfill):
                            try:
                                working_database_rows[row] += f" {phasename}:{table[i, melt_dict['mass (gm)']]}"
                            except:
                                fault = True
                                faultIDs.append(folderNo)
            
            assert len(working_database_rows) == meltsobj.shape[0], f'Unequal Length run metadata and meltobj rows sim: {folderNo}'

            if not fault:
                for wdr in working_database_rows:
                    metadata.append(wdr + '\n')
                workbase = np.vstack([workbase, meltsobj])
        
        if fault or not go:
            faultIDs.append(folderNo)  # Record failures for exfail function
            print(f"FAILURE AT FOLDER {folderNo}")
    
    if len(metadata) != np.shape(workbase)[0]:
        raise Exception('Metadata different length than rows of csv!')

    # New as of 10/08/25: Filter out much of the superliquidus assemblage to save space, balance dataset
    # Step 1: Identify nonzero rows in selected columns
    nonzero_mask = (workbase[:, indexer.mass_indices[:-1]] != 0).any(axis=1)
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
    workDF.to_csv(dataname, mode='a', index=False, header=False)
    return np.unique(faultIDs)


def pick_exsolution_failure(EnsembleLocation, input_array, keys, batchname=['exampleWOW'] * 8,
                            dataname='2_phasePTX.csv', faultIDs=[]):
    """
    Saves 2+ phase conditions (PTXfO2) for later study.
    This function is unnecessary in most cases, as multiple instances of the same phase are prohibited in MELTS directly using the Batch file
    Parameters:
    -----------
    EnsembleLocation : str
        Location of simulation folders
    input_array : np.ndarray
        Input conditions array
    keys : list
        Column labels
    batchname : list, default=['exampleWOW']*8
        Batch names
    dataname : str, default='2_phasePTX.csv'
        Output CSV filename
    faultIDs : list, default=[]
        List of fault IDs to include
        
    Returns:
    --------
    None
    """
    sim_metadata_name = dataname.split('.')[0] + '.txt'
    metadata = []
    workbase = np.empty((0, 1 + len(keys)))
    if not os.path.exists(dataname):
        database_headers = keys + ['Failed']
        df = pd.DataFrame(columns=database_headers)
        df.to_csv(dataname, index=False)
    if not os.path.exists(sim_metadata_name):
        with open('emptyfile.txt', 'w') as f:
            pass
    contents = os.listdir(EnsembleLocation)
    folders = len(contents) - 1
    
    for folderNo in range(folders):
        if folderNo in faultIDs:
            workbase = np.vstack([workbase, np.append(input_array[folderNo], [1])])  # Record Failures from import_MELTS
            metadata.append(batchname[folderNo] + '\n')
            continue
        folder = 'Simulation' + str(folderNo)
        proceed = True
        try:
            main_file = os.path.join(EnsembleLocation, folder, 'Phase_main_tbl.txt')
            with open(main_file, 'r') as myfile:
                mainstr = myfile.read() + '\n'
        except:  # Record failures
            metadata.append(batchname[folderNo] + '\n')
            workbase = np.vstack([workbase, np.append(input_array[folderNo], [1])])
            proceed = False
        if proceed:
            chunks = mainstr.split("\n\n")
            del chunks[-1]
            save = False  # Only save exsolutions
            working_label = batchname[folderNo]
            for chunk in chunks:
                ind = 0
                lines = chunk.split('\n')
                while chunk.split('\n')[ind] == '':
                    ind += 1  # Ignore headspace
                header = lines[ind]
                phase = header.split()[0]
                if pull_number(phase) in [2, 3, 4, 5, 6]:  # Detect plural phases
                    col = ReadOnePhaseFromMELTSOutputFile(mainstr, header).to_numpy()[:, 0]
                    save = True
                    working_label += f' {phase}-{int(min(col))}-{int(max(col))}'
            if save:
                metadata.append(working_label + '\n')
                workbase = np.vstack([workbase, np.append(input_array[folderNo], [0])])
    
    if len(metadata) != np.shape(workbase)[0]:
        raise Exception('Metadata different length than rows of csv!')
    with open(sim_metadata_name, 'a') as f:
        f.writelines(metadata)
    workDF = pd.DataFrame(workbase)
    workDF.to_csv(dataname, mode='a', index=False, header=False)
