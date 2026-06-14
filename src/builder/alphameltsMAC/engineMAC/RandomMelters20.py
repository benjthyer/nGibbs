"""
Random MELTS simulation execution script.

This script provides functions for running ensemble MELTS calculations
with random compositions from the GEOROC database.
"""

import sys
import os
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / 'src'

# Add repo root and src to path so repo-local packages resolve from repo root.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import MELTS ensemble functions
from . import alphamelts_functions

# Import utility functions
from ngibbs.utils.string_utils import pull_number, random_char
from ngibbs.utils.file_utils import count_csv_rows, count_file_lines

# Import indexer and header generation
from builder.indexer import DatasetIndexer, verify_csv_headers_match
from recipes.settings import internal_scratch_dir

# Set up paths
EnsembleLocation = str(internal_scratch_dir())
os.makedirs(EnsembleLocation, exist_ok=True)


def _validate_existing_files(dataname, sim_metadata_name, indexer):
    """
    Validate that existing output files have compatible headers and matching line counts.
    
    Parameters
    ----------
    dataname : str
        Path to CSV file
    sim_metadata_name : str
        Path to text metadata file
    indexer : DatasetIndexer
        Indexer to validate headers against
        
    Raises
    ------
    ValueError
        If headers don't match or line counts are inconsistent
    """
    if os.path.exists(dataname):
        matches, error_msg = verify_csv_headers_match(dataname, indexer)
        if not matches:
            raise ValueError(
                f"Existing file {dataname} has incompatible headers:\n{error_msg}\n"
                f"Please remove the file or use a different output_file name."
            )
        print(f"Found existing file {dataname} with compatible headers. Appending data...")
        
        # Verify CSV and text file have matching line counts
        if os.path.exists(sim_metadata_name):
            csv_rows = count_csv_rows(dataname, has_header=True)
            txt_lines = count_file_lines(sim_metadata_name, skip_header=False)
            if csv_rows != txt_lines:
                raise ValueError(
                    f"Line count mismatch between {dataname} and {sim_metadata_name}:\n"
                    f"  CSV has {csv_rows} data rows (excluding header)\n"
                    f"  Text file has {txt_lines} lines\n"
                    f"These should match. Please check the files for corruption."
                )
            print(f"Verified: CSV ({csv_rows} rows) and text file ({txt_lines} lines) have matching counts.")


def _process_compositions(compositions, col_dict, simcycle, MELTSModel, zeroOxides = ['MnO', 'NiO']):
    """
    Process and normalize compositions, setting constraints on various oxides.

    Water handling is implicit per model:
      p   → all anhydrous
      102 → half gaussian-hydrous (sigma=0.5), half anhydrous
      110/120 → half gaussian-hydrous, ~1/6 soaked (uniform 0-5 wt%), ~1/3 anhydrous

    Parameters
    ----------
    compositions : np.ndarray
        Array of compositions to process
    col_dict : dict
        Dictionary mapping oxide names to column indices
    simcycle : int
        Number of simulations in this cycle
    MELTSModel : str
        MELTS model version ('p', '102', '110', '120')

    Returns
    -------
    np.ndarray
        Processed and normalized compositions
    """
    # Zero water up front
    compositions[:, col_dict['H2O']] = 0
    if MELTSModel in ('110', '120'):
        remaining_idx = np.arange(simcycle)
        hydrous = np.random.choice(remaining_idx, size=int(simcycle/2), replace=False)
        compositions[hydrous, col_dict['H2O']] = np.abs(np.random.normal(size=len(hydrous), scale=0.5))
        out_mask = np.ones(simcycle)
        out_mask[hydrous] = 0
        remaining_idx = remaining_idx[out_mask.astype(bool)]
        soaked = np.random.choice(remaining_idx, size=int(simcycle/3), replace=False)
        compositions[soaked, col_dict['H2O']] = np.random.uniform(size=len(soaked), high=5)
    elif MELTSModel == '102':
        hydrous = np.random.choice(np.arange(simcycle), size=int(simcycle/2), replace=False)
        compositions[hydrous, col_dict['H2O']] = np.abs(np.random.normal(size=len(hydrous), scale=0.5))
    # 'p': all anhydrous — already zeroed above
    
    # Set zeroOxides to zero
    for oxide in zeroOxides:
        compositions[:, col_dict[oxide]] = 0
    
    # For pMELTS, randomly cancel out poorly handled volatiles. Consider excluding these oxides altogether from pMELTS.
    if MELTSModel == 'p':
        NoKP = np.random.randint(simcycle, high=None, size=int(simcycle/2))
        compositions[NoKP, col_dict['P2O5']] = 0
        compositions[NoKP, col_dict['K2O']] = 0
    
    # Set small Cr2O3 to zero (mostly for felsic compositions)
    smallCr2O = compositions[:, col_dict['Cr2O3']] < 0.01
    compositions[smallCr2O, col_dict['Cr2O3']] = 0
    
    # Normalize to 100%
    compositions = 100 * compositions / (np.sum(compositions, axis=1))[:, np.newaxis]
    
    return compositions



def alphaMELTScompress(output_file, MELTSModel, GEOROC, col_dict, indexer, itercode='a1', 
                       simcycle=50, fxtal=False, ExFailures=False):
    """
    Perform ensemble MELTS compression calculations with random compositions.
    
    Note: Compression not implemented for pMELTS yet.
    
    Parameters
    ----------
    output_file : str
        Base name for output files (without extension)
    MELTSModel : str
        MELTS model version ('102', '110', '120', 'p')
    GEOROC : np.ndarray
        Array of GEOROC compositions (first column is index, rest are oxides)
    col_dict : dict
        Dictionary mapping oxide names to column indices in compositions
    indexer : DatasetIndexer
        DatasetIndexer object for the dataset
    iter : int, default=750
        Number of iterations to run
    simcycle : int, default=50
        Number of simulations per iteration
    fxtal : bool, default=False
        Whether to enable fractional crystallization
    """

    iterLetter = itercode[0]
    iter = int(itercode[1:])

    if iter <= 0 or iter != int(iter):
        print('Must have integer positive nonzero for iteration "iter" argument. Substituting 1.')
        iter = int(1)
    
    batch_file = MELTSModel + 'batch'
    dataname = f'{output_file}.csv'
    sim_metadata_name = f'{output_file}.txt'
    progress_file = f'{output_file}_progress.txt'
    
    _validate_existing_files(dataname, sim_metadata_name, indexer)
    
    # Initialize or resume from progress file
    start_iter = 0
    existing_lines = []
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                all_lines = f.read().strip().split('\n')
            
            if all_lines and all_lines[0]:  # Check if file has content
                # Find and extract the line with current letter code if it exists
                found_current_letter = False
                for line in all_lines:
                    if line.strip().startswith(iterLetter):
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            start_iter = int(parts[1].split('/')[0])
                            print(f"Resuming {iterLetter} from iteration {start_iter}/{iter}")
                            found_current_letter = True
                        break
                
                # Preserve all lines except the one with current letter code (will be rewritten)
                existing_lines = [line for line in all_lines if not line.strip().startswith(iterLetter)]
        except Exception as e:
            print(f"Warning: Could not read progress file {progress_file}: {e}. Starting fresh.")
            existing_lines = []
            start_iter = 0
    
    def PTF_initialize(conditions, length=simcycle):
        """Initialize Pressure-Temperature-fO2 arrays for compression runs."""
        out_array = np.zeros((length, np.shape(conditions)[1] + 3))
        out_array[:, 0] = np.random.uniform(1, 21, size=length)  # Pressure in kbar (compression)
        out_array[:, 1] = np.random.uniform(700, 2000, size=length)  # Temperature in K
        out_array[:, 2] = np.random.uniform(-5, 5, size=length)  # fO2 offset from FMQ
        #PDF = 
        #conditions = np.copy(input_conditions)
        
        out_array[:, 3:] = conditions
        return out_array
    
    keys = np.array(["Pressure", "Temperature", 'Fe2O3'] + list(col_dict.keys()))
    
    for j in range(start_iter, iter): # Will not run if start_iter == iter, so safe to use range for both iteration and progress tracking
        # Select random compositions
        choices = np.random.randint(np.shape(GEOROC)[0], size=simcycle, dtype=int)
        compositions = GEOROC[choices, 1:].copy()
        indices = GEOROC[choices, 0]
        
        # Process and normalize compositions
        compositions = _process_compositions(compositions, col_dict, simcycle, MELTSModel)
        
        # Initialize PTX conditions
        in_array = np.round(PTF_initialize(compositions, length=simcycle), 2)
        
        # Set batch names
        batchname = np.full(simcycle, batch_file, dtype=object)
        
        # Run ensemble simulation
        alphamelts_functions.forward_ensemble(
            in_array, keys=keys, batchname=batchname,
            only_phases=indexer.get_phase_list(),
            end=12000 + in_array[:, 0], fxtal=fxtal,
            EnsembleLocation=EnsembleLocation, WSL=True,
            compression=True, delta=12000/200
        )
        
        # Update batch names with metadata
        for i, name in enumerate(batchname):
            batchname[i] = f"{pull_number(str(indices[i]))}:{random_char(4)}:{name}"
        
        # Import results
        failure_IDs = alphamelts_functions.import_MELTS_components(
            EnsembleLocation=EnsembleLocation, batchname=batchname,
            indexer=indexer, fO2Arr=in_array[:, 2], dataname=dataname
        )

        if ExFailures:
            alphamelts_functions.pick_exsolution_failure(EnsembleLocation, in_array, keys, batchname=batchname,
                                dataname=dataname, faultIDs=failure_IDs)
        
        # Update progress file
        current_iter = j + 1
        progress_content = '\n'.join(existing_lines) if existing_lines else ''
        if progress_content:
            progress_content += f"\n{iterLetter} {current_iter}/{iter}"
        else:
            progress_content = f"{iterLetter} {current_iter}/{iter}"
        
        with open(progress_file, 'w') as f:
            f.write(progress_content)
    

def alphaMELTScooling(output_file, MELTSModel, GEOROC, col_dict, indexer, itercode='a1', simcycle=50, fxtal=False,
                      ExFailures=False, zeroOxides = ['MnO', 'NiO'], startT=1925, max_liquid_fraction=100, end=700,
                      Prange = None, delta = -1, Oxygen = 'Closed'):
    """
    Perform ensemble MELTS cooling calculations with random compositions.

    Parameters
    ----------
    output_file : str
        Base name for output files (without extension)
    MELTSModel : str
        MELTS model version ('102', '110', '120', 'p')
    GEOROC : np.ndarray
        Array of GEOROC compositions (first column is index, rest are oxides)
    col_dict : dict
        Dictionary mapping oxide names to column indices in compositions
    indexer : DatasetIndexer
        DatasetIndexer object for the dataset
    itercode : str, default='a1'
        Code for the total iteration (used for tracking progress. a for all melts, m for mafics)
    simcycle : int, default=50
        Number of simulations per iteration
    fxtal : bool, default=False
        Whether to enable fractional crystallization
    zeroOxides : list, default=['MnO', 'NiO']
        List of oxides to set to zero across all compositions
    startT : int, default=1925
        Starting temperature in Celsius
    max_liquid_fraction : int, default=100
        Maximum allowed liquid fraction for the simulation. 100 Still subsamples superliquidus
    end : int, default=700
        End temperature in Celsius
    Prange : list, default=None
        Pressure range in bars
    delta : int, default=-1
        Temperature change per simulation
    Oxygen : str, default='Closed'
        Oxygen fugacity condition ('Closed' or 'Open')
    """

    iterLetter = itercode[0]
    iter = int(itercode[1:])

    # Set pressure range and end temperature based on MELTS model
    if Prange is None:
        if MELTSModel == 'p':
            Prange = [8000, 30000]  # Pressure in bars
        else:
            Prange = [1, 12000]  # Pressure in bars

    if iter <= 0 or iter != int(iter):
        print('Must have integer positive nonzero for iteration "iter" argument. Substituting 1.')
        iter = int(1)
    
    batch_file = MELTSModel + 'batch'
    dataname = f'{output_file}.csv'
    sim_metadata_name = f'{output_file}.txt'
    progress_file = f'{output_file}_progress.txt'
    
    _validate_existing_files(dataname, sim_metadata_name, indexer)
    
    # Initialize or resume from progress file
    start_iter = 0
    existing_lines = []
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                all_lines = f.read().strip().split('\n')
            
            if all_lines and all_lines[0]:  # Check if file has content
                # Find and extract the line with current letter code if it exists
                found_current_letter = False
                for line in all_lines:
                    if line.strip().startswith(iterLetter):
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            start_iter = int(parts[1].split('/')[0])
                            print(f"Resuming {iterLetter} from iteration {start_iter}/{iter}")
                            found_current_letter = True
                        break
                
                # Preserve all lines except the one with current letter code (will be rewritten)
                existing_lines = [line for line in all_lines if not line.strip().startswith(iterLetter)]
        except Exception as e:
            print(f"Warning: Could not read progress file {progress_file}: {e}. Starting fresh.")
            existing_lines = []
            start_iter = 0
    
    def PTF_initialize(conditions, length=simcycle):
        """Initialize Pressure-Temperature-fO2 arrays for cooling runs."""
        conditions = conditions.copy()
        out_array = np.zeros((length, np.shape(conditions)[1] + 3))
        out_array[:, 0] = np.random.uniform(*Prange, size=length)  # Pressure in bars
        out_array[:, 1] = startT + 20 + np.arange(length)  # Temperature in K (starting high, decreasing)
        logfo2 = np.random.uniform(-5, 5, size=length) # log fO2 delta QFM
        
        if Oxygen.lower() == 'closed':
            ferric_to_ferrous = 0.8998084799181955 #wt ratio conserving Fe atoms
            logR = (logfo2*0.2)-1
            R32 = 10**logR
            R_chosen = R32/(R32+1) #Fe3 / FeT Molar
            ferric = conditions[:,col_dict['FeO']]*R_chosen*(1/ferric_to_ferrous)
            conditions[:,col_dict['FeO']] = conditions[:,col_dict['FeO']]*(1-R_chosen)
            total = (np.sum(conditions, axis = 1) + ferric).reshape(-1,1) # Renormalize to new wt%
            conditions = conditions*(100/total)
            ferric = ferric*(100/total.flatten())
            out_array[:, 2] = ferric
        elif Oxygen.lower() == 'open':
            # Open system with respect to oxygen, set fO2 to random values
            out_array[:, 2] = logfo2  # fO2 offset from FMQ
        else:           
            raise ValueError(f"Oxygen condition '{Oxygen}' not recognized. Use 'closed' or 'open'")
        out_array[:, 3:] = conditions
        return out_array
    
    if Oxygen.lower() == 'closed':
        keys = np.array(["Pressure", "Temperature", 'Fe2O3'] + list(col_dict.keys()))
    elif Oxygen.lower() == 'open':
        keys = np.array(["Pressure", "Temperature", 'fO2'] + list(col_dict.keys()))
    
    for j in range(start_iter, iter):
        # Select random compositions
        choices = np.random.randint(np.shape(GEOROC)[0], size=simcycle, dtype=int)
        compositions = GEOROC[choices, 1:].copy()
        indices = GEOROC[choices, 0]
        
        # Process and normalize compositions
        compositions = _process_compositions(compositions, col_dict, simcycle, MELTSModel, zeroOxides=zeroOxides)
        
        # Initialize PTX conditions
        in_array = np.round(PTF_initialize(compositions, length=simcycle), 2)
        
        # Set batch names
        batchname = np.full(simcycle, batch_file, dtype=object)
        
        # Run ensemble simulation
        alphamelts_functions.forward_ensemble(
            in_array, keys=keys, batchname=batchname,
            only_phases=indexer.get_phase_list(),
            end=end, fxtal=fxtal,
            EnsembleLocation=EnsembleLocation, WSL=True,
            compression=False, delta=delta
        )
        
        # Update batch names with metadata
        for i, name in enumerate(batchname):
            batchname[i] = f"{pull_number(str(indices[i]))}:{random_char(4)}:{name}"
        
        # Import results
        failure_IDs = alphamelts_functions.import_MELTS_components(
            EnsembleLocation=EnsembleLocation, batchname=batchname,
            indexer=indexer, fO2Arr=in_array[:, 2], dataname=dataname,
            max_liquid_fraction=max_liquid_fraction
        )

        if ExFailures:
            alphamelts_functions.pick_exsolution_failure(EnsembleLocation, in_array, keys, batchname=batchname,
                                dataname=dataname, faultIDs=failure_IDs)
        
        # Update progress file
        current_iter = j + 1
        progress_content = '\n'.join(existing_lines) if existing_lines else ''
        if progress_content:
            progress_content += f"\n{iterLetter} {current_iter}/{iter}"
        else:
            progress_content = f"{iterLetter} {current_iter}/{iter}"
        
        with open(progress_file, 'w') as f:
            f.write(progress_content)
