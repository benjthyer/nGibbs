"""
Test script to debug import_MELTS_components function.

Prints detailed internal state to diagnose why CSV files are empty.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.wslMELTS.engine import alphamelts_functions
from src.nMELTS.config.indexer import generate_column_headers, DatasetIndexer

def debug_import_MELTS_components(EnsembleLocation, batchname, indexer, fO2Arr=None, dataname='DefaultMELTSstorage.csv'):
    """
    Debug version of import_MELTS_components with extensive print statements.
    """
    print("=" * 80)
    print("DEBUG: import_MELTS_components")
    print("=" * 80)
    
    print(f"\n[1] Input Parameters:")
    print(f"    EnsembleLocation: {EnsembleLocation}")
    print(f"    dataname: {dataname}")
    print(f"    batchname type: {type(batchname)}, length: {len(batchname) if hasattr(batchname, '__len__') else 'N/A'}")
    print(f"    fO2Arr: {fO2Arr is not None}")
    print(f"    indexer.get_max_index(): {indexer.get_max_index()}")
    print(f"    indexer.database_headers length: {len(indexer.database_headers)}")
    
    # Check EnsembleLocation
    print(f"\n[2] Checking EnsembleLocation:")
    if not os.path.exists(EnsembleLocation):
        print(f"    ERROR: EnsembleLocation does not exist: {EnsembleLocation}")
        return
    print(f"    EnsembleLocation exists: {os.path.exists(EnsembleLocation)}")
    
    contents = os.listdir(EnsembleLocation)
    print(f"    Contents: {contents}")
    folders = len(contents) - 1  # Subtract 1 for runall.sh or other files
    print(f"    Number of folders (len-1): {folders}")
    print(f"    Full contents list: {contents}")
    
    sim_metadata_name = dataname.split('.')[0] + '.txt'
    print(f"\n[3] Output Files:")
    print(f"    CSV file: {dataname}")
    print(f"    Metadata file: {sim_metadata_name}")
    print(f"    CSV exists: {os.path.exists(dataname)}")
    print(f"    Metadata exists: {os.path.exists(sim_metadata_name)}")
    
    metadata = []
    workbase = np.empty((0, indexer.get_max_index()+1))
    print(f"\n[4] Initial State:")
    print(f"    workbase shape: {workbase.shape}")
    print(f"    metadata length: {len(metadata)}")
    
    if not os.path.exists(sim_metadata_name):
        with open('emptyfile.txt', 'w') as f:
            pass
    if not os.path.exists(dataname):
        newbase = pd.DataFrame(columns=indexer.database_headers)
        newbase.to_csv(dataname, index=False)
        print(f"    Created new CSV file with {len(indexer.database_headers)} columns")
    
    faultIDs = []
    
    print(f"\n[5] Processing Folders:")
    print(f"    Will process folders: 0 to {folders-1}")
    
    for folderNo in range(folders):
        print(f"\n    --- Folder {folderNo} ---")
        folder = 'Simulation' + str(folderNo)
        run = os.path.join(EnsembleLocation, folder)
        print(f"    Folder path: {run}")
        print(f"    Folder exists: {os.path.exists(run)}")
        
        if not os.path.exists(run):
            print(f"    SKIP: Folder does not exist")
            continue
        
        # List files in folder
        try:
            folder_files = os.listdir(run)
            print(f"    Files in folder: {folder_files}")
        except Exception as e:
            print(f"    ERROR listing files: {e}")
            continue
        
        tablename = 'System_main_tbl.txt'
        fault = False
        go = False
        
        # Check System_main_tbl.txt
        system_main_path = os.path.join(run, tablename)
        print(f"    System_main_tbl.txt exists: {os.path.exists(system_main_path)}")
        
        try:
            table = np.genfromtxt(system_main_path, skip_header=2)
            nrows = np.shape(table)[0] if table.size > 0 else 0
            print(f"    System_main table shape: {np.shape(table)}")
            print(f"    nrows: {nrows}")
            if table.ndim <= 1:
                go = False
                print(f"    SKIP: Table is 1D or empty")
            else:
                go = True
                print(f"    OK: Table is 2D, proceeding")
        except Exception as e:
            go = False
            print(f"    ERROR reading System_main_tbl.txt: {e}")
            print(f"    SKIP: Simulation{folderNo} FAILED and was not read!")
        
        if go:
            fault = False
            working_database_rows = []
            for nr in range(nrows):
                if folderNo < len(batchname):
                    working_database_rows.append(batchname[folderNo] + f' {nr}')
                else:
                    working_database_rows.append(f'batch{folderNo} {nr}')
            
            print(f"    Created {len(working_database_rows)} database rows")
            meltsobj = np.zeros((nrows, indexer.get_max_index()+1))
            print(f"    meltsobj shape: {meltsobj.shape}")
            
            # Find all table files
            table_files = [f for f in folder_files if 'tbl' in f and f not in 
                          ['Solid_comp_tbl.txt', 'Phase_vol_tbl.txt', 'Phase_mass_tbl.txt', 
                           'Phase_main_tbl.txt', 'Liquid_comp_tbl.txt', 'Bulk_comp_tbl.txt']]
            print(f"    Phase table files found: {table_files}")
            
            phases_processed = 0
            for tablename in table_files:
                print(f"\n      Processing: {tablename}")
                phasename = tablename.split('tbl')[0][:-1]
                if phasename in ['orthoamphibole', 'clinoamphibole', 'hornblende']:
                    phasename = 'amphibole'
                print(f"      Phase name: {phasename}")
                
                skipline = 1
                delim = ','
                if tablename == 'System_main_tbl.txt':
                    skipline = 2
                    delim = ' '
                
                table_path = os.path.join(run, tablename)
                try:
                    with open(table_path, 'r') as text:
                        lines = text.readlines()
                        if len(lines) >= skipline:
                            headers = lines[skipline - 1].split(delim)
                            print(f"      Headers: {headers[:5]}... (showing first 5)")
                        else:
                            print(f"      WARNING: Not enough lines in file")
                            headers = []
                    
                    melt_dict = {}
                    for i, header in enumerate(headers):
                        melt_dict[header.strip()] = i
                    print(f"      Found {len(melt_dict)} columns in header")
                    
                    table = np.genfromtxt(table_path, delimiter=delim, skip_header=skipline)
                    print(f"      Table shape: {np.shape(table)}")
                    
                    if len(np.shape(table)) <= 1:
                        table = np.atleast_2d(table)
                        print(f"      Reshaped to 2D: {np.shape(table)}")
                    
                    try:
                        rowsfill = table[:, 0].astype(int) - 1
                        print(f"      rowsfill shape: {rowsfill.shape}, range: [{rowsfill.min()}, {rowsfill.max()}]")
                    except Exception as e:
                        print(f"      ERROR getting rowsfill: {e}")
                        fault = True
                        break
                    
                    if phasename in list(indexer.MELTS_indices.keys()):
                        print(f"      Phase '{phasename}' found in indexer.MELTS_indices")
                        compnames = list(indexer.MELTS_indices[phasename].keys())
                        print(f"      Components for this phase: {compnames}")
                        
                        components_filled = 0
                        for fillname in compnames:
                            if fillname == 'corundum' and folderNo < len(batchname) and 'pBatch' in batchname[folderNo]: # Confused why I have folderno < len(batchname)
                                print(f"        SKIP corundum (pMELTS)")
                                continue
                            
                            if fillname == 'logfO2-QFM':
                                if fO2Arr is None:
                                    for key in list(melt_dict.keys()):
                                        if 'QFM' in key:
                                            fO2key = key
                                            delta = float(key[4:]) if len(key) > 4 else 0.0
                                            if np.isnan(delta):
                                                delta = 0
                                            break
                                else:
                                    delta = fO2Arr[folderNo] if folderNo < len(fO2Arr) else 0
                                
                                try:
                                    idx = indexer.MELTS_indices[phasename][fillname]
                                    meltsobj[rowsfill, idx] = delta
                                    components_filled += 1
                                    print(f"        Filled {fillname} at index {idx} with delta={delta}")
                                except Exception as e:
                                    print(f"        ERROR filling {fillname}: {e}")
                                    fault = True
                            elif phasename == 'amphibole':
                                if fillname in melt_dict:
                                    try:
                                        idx = indexer.MELTS_indices[phasename][fillname]
                                        meltsobj[rowsfill, idx] += table[:, melt_dict[fillname]]
                                        components_filled += 1
                                        print(f"        Filled {fillname} at index {idx} (amphibole, additive)")
                                    except Exception as e:
                                        print(f"        ERROR filling {fillname}: {e}")
                                        fault = True
                                else:
                                    print(f"        WARNING: {fillname} not in melt_dict")
                            else:
                                if fillname in melt_dict:
                                    try:
                                        idx = indexer.MELTS_indices[phasename][fillname]
                                        meltsobj[rowsfill, idx] = table[:, melt_dict[fillname]]
                                        components_filled += 1
                                        print(f"        Filled {fillname} at index {idx}")
                                    except Exception as e:
                                        print(f"        ERROR filling {fillname}: {e}")
                                        fault = True
                                else:
                                    print(f"        WARNING: {fillname} not in melt_dict (available: {list(melt_dict.keys())[:5]}...)")
                        
                        print(f"      Filled {components_filled} components for {phasename}")
                        phases_processed += 1
                    else:
                        print(f"      WARNING: Phase '{phasename}' not in indexer.MELTS_indices")
                        print(f"      Available phases: {list(indexer.MELTS_indices.keys())[:10]}...")
                        # Try to get mass anyway
                        if 'mass (gm)' in melt_dict:
                            for i, row in enumerate(rowsfill):
                                try:
                                    working_database_rows[row] += f" {phasename}:{table[i, melt_dict['mass (gm)']]}"
                                except Exception as e:
                                    print(f"        ERROR adding to metadata: {e}")
                
                except Exception as e:
                    print(f"      ERROR processing {tablename}: {e}")
                    import traceback
                    traceback.print_exc()
                    fault = True
                    break
            
            print(f"\n    Summary for folder {folderNo}:")
            print(f"      Phases processed: {phases_processed}")
            print(f"      Fault: {fault}")
            print(f"      meltsobj shape: {meltsobj.shape}")
            print(f"      meltsobj non-zero elements: {np.count_nonzero(meltsobj)}")
            print(f"      working_database_rows length: {len(working_database_rows)}")
            
            assert len(working_database_rows) == meltsobj.shape[0], f'Unequal Length run metadata and meltobj rows sim: {folderNo}'
            
            if not fault:
                for wdr in working_database_rows:
                    metadata.append(wdr + '\n')
                workbase = np.vstack([workbase, meltsobj])
                print(f"      Added to workbase. New workbase shape: {workbase.shape}")
            else:
                print(f"      SKIP: Fault detected, not adding to workbase")
        
        if fault or not go:
            faultIDs.append(folderNo)
            print(f"    FAILURE AT FOLDER {folderNo}")
    
    print(f"\n[6] Final Summary:")
    print(f"    Total metadata entries: {len(metadata)}")
    print(f"    Final workbase shape: {workbase.shape}")
    print(f"    Fault IDs: {faultIDs}")
    
    if len(metadata) != np.shape(workbase)[0]:
        print(f"    ERROR: Metadata length {len(metadata)} != workbase rows {np.shape(workbase)[0]}")
        raise Exception('Metadata different length than rows of csv!')
    
    # Filtering step
    print(f"\n[7] Filtering Step:")
    if workbase.shape[0] > 0 and len(indexer.mass_indices) > 0:
        nonzero_mask = (workbase[:, indexer.mass_indices[:-1]] != 0).any(axis=1)
        print(f"    nonzero_mask shape: {nonzero_mask.shape}")
        print(f"    Non-zero rows: {np.sum(nonzero_mask)}")
        
        nonzero_indices = np.where(nonzero_mask)[0]
        zero_indices = np.where(~nonzero_mask)[0]
        print(f"    Non-zero indices count: {len(nonzero_indices)}")
        print(f"    Zero indices count: {len(zero_indices)}")
        
        n_add = len(nonzero_indices) // 4
        print(f"    Will add back {n_add} zero rows")
        
        if len(zero_indices) > 0:
            add_back_indices = np.random.choice(zero_indices, size=min(n_add, len(zero_indices)), replace=False)
        else:
            add_back_indices = np.array([], dtype=int)
        
        final_indices = np.sort(np.concatenate([nonzero_indices, add_back_indices]))
        print(f"    Final indices count: {len(final_indices)}")
        
        filtered_workbase = workbase[final_indices]
        filtered_rows = [metadata[L] for L in final_indices]
        
        print(f"    Filtered workbase shape: {filtered_workbase.shape}")
        print(f"    Filtered metadata length: {len(filtered_rows)}")
    else:
        print(f"    SKIP: workbase is empty or no mass_indices")
        filtered_workbase = workbase
        filtered_rows = metadata
        final_indices = np.arange(len(metadata)) if len(metadata) > 0 else np.array([], dtype=int)
    
    print(f"\n[8] Saving Files:")
    print(f"    Saving to: {dataname}")
    print(f"    Saving metadata to: {sim_metadata_name}")
    print(f"    Data shape: {filtered_workbase.shape}")
    
    with open(sim_metadata_name, 'a') as f:
        f.writelines(filtered_rows)
    print(f"    Metadata file written")
    
    workDF = pd.DataFrame(filtered_workbase)
    workDF.to_csv(dataname, mode='a', index=False, header=False)
    print(f"    CSV file written")
    
    print(f"\n[9] Verification:")
    if os.path.exists(dataname):
        df_check = pd.read_csv(dataname)
        print(f"    CSV file exists, shape: {df_check.shape}")
        print(f"    CSV file columns: {len(df_check.columns)}")
    else:
        print(f"    WARNING: CSV file was not created!")
    
    return np.unique(faultIDs)


if __name__ == "__main__":
    # Set up similar to MELTedMORB.py
    allowed_phases = ['olivine','orthopyroxene','clinopyroxene','spinel','plagioclase','k-feldspar','garnet','nepheline','leucite','biotite',
                      'rhm-oxide','alloy-solid','apatite','whitlockite','quartz','tridymite','cristobalite','muscovite','fluid','liquid']
    
    headers = generate_column_headers(allowed_phases)
    indexer = DatasetIndexer(headers)
    
    Out_Folder = 'MORB_SIMS'
    EnsembleLocation = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'wslMELTS', 'Workspace', Out_Folder)
    
    # Check if EnsembleLocation exists
    if not os.path.exists(EnsembleLocation):
        print(f"ERROR: EnsembleLocation does not exist: {EnsembleLocation}")
        print("Please run forward_ensemble first to create simulation folders.")
        sys.exit(1)
    
    # Try to find existing batch files or create dummy batchname
    # Look for any existing simulation folders to determine batchname
    contents = os.listdir(EnsembleLocation)
    simulation_folders = [f for f in contents if f.startswith('Simulation')]
    
    if len(simulation_folders) == 0:
        print(f"ERROR: No Simulation folders found in {EnsembleLocation}")
        sys.exit(1)
    
    # Create a dummy batchname array (you may need to adjust this)
    num_sims = len(simulation_folders)
    batchname = [f'102Batch' for _ in range(num_sims)]
    
    csv_name = 'GTMELTS102_NoCr_MORB_Batch_PsuedoSections_smallOx2.csv'
    
    print("Running debug version of import_MELTS_components...")
    print(f"Will process {num_sims} simulation folders")
    
    faultIDs = debug_import_MELTS_components(
        EnsembleLocation=EnsembleLocation,
        batchname=batchname,
        indexer=indexer,
        fO2Arr=None,
        dataname=csv_name
    )
    
    print(f"\n{'='*80}")
    print("DEBUG COMPLETE")
    print(f"{'='*80}")
    print(f"Fault IDs: {faultIDs}")
