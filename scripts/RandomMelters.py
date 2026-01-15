"""
Random MELTS simulation execution script.

This script provides functions for running ensemble MELTS calculations
with random compositions from the GEOROC database.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Import MELTS ensemble functions
from src.wslMELTS.engine import alphamelts_functions

# Import utility functions
from src.nMELTS.utils.string_utils import pull_number, random_char
from src.nMELTS.utils.file_utils import count_csv_rows, count_file_lines

# Import indexer and header generation
from src.nMELTS.config.indexer import DatasetIndexer, verify_csv_headers_match

# Set up paths
EnsembleLocation = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'wslMELTS', 'Workspace')
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


def _process_compositions(compositions, col_dict, simcycle, MELTSModel):
    """
    Process and normalize compositions, setting constraints on various oxides.
    
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
    # Randomly set some compositions to anhydrous
    anhydrous = np.random.randint(simcycle, high=None, size=int(simcycle/4))
    compositions[anhydrous, col_dict['H2O']] = 0
    
    # Cap H2O at 5%
    too_wet = compositions[:, col_dict['H2O']] > 5
    compositions[too_wet, col_dict['H2O']] = 5
    
    # Set MnO and NiO to zero
    compositions[:, col_dict['MnO']] = 0
    compositions[:, col_dict['NiO']] = 0
    
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



def alphaMELTScompress(output_file, MELTSModel, GEOROC, col_dict, indexer, iter=int(1), simcycle=50, fxtal=False):
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
    if iter <= 0 or iter != int(iter):
        print('Must have integer positive nonzero for iteration "iter" argument. Substituting 1.')
        iter = int(1)
    
    batch_file = MELTSModel + 'batch'
    dataname = f'{output_file}.csv'
    sim_metadata_name = f'{output_file}.txt'
    
    _validate_existing_files(dataname, sim_metadata_name, indexer)
    
    def PTF_initialize(conditions, length=simcycle):
        """Initialize Pressure-Temperature-fO2 arrays for compression runs."""
        out_array = np.zeros((length, np.shape(conditions)[1] + 3))
        out_array[:, 0] = np.random.uniform(1, 21, size=length)  # Pressure in kbar (compression)
        out_array[:, 1] = np.random.uniform(700, 2000, size=length)  # Temperature in K
        out_array[:, 2] = np.random.uniform(-5, 5, size=length)  # fO2 offset from FMQ
        out_array[:, 3:] = conditions
        return out_array
    
    keys = np.array(["Pressure", "Temperature", 'fO2'] + list(col_dict.keys()))
    
    for j in range(iter):
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
        alphamelts_functions.import_MELTS_components(
            EnsembleLocation=EnsembleLocation, batchname=batchname,
            indexer=indexer, fO2Arr=in_array[:, 2], dataname=dataname
        )

def alphaMELTScooling(output_file, MELTSModel, GEOROC, col_dict, indexer, iter=750, simcycle=50, fxtal=False):
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
    iter : int, default=750
        Number of iterations to run
    simcycle : int, default=50
        Number of simulations per iteration
    fxtal : bool, default=False
        Whether to enable fractional crystallization
    """
    # Set pressure range and end temperature based on MELTS model
    if MELTSModel == 'p':
        Prange = [8000, 30000]  # Pressure in bars
        end = 1000  # End temperature in Celsius
    else:
        Prange = [1, 12000]  # Pressure in bars
        end = 700  # End temperature in Celsius
    
    if iter <= 0 or iter != int(iter):
        print('Must have integer positive nonzero for iteration "iter" argument. Substituting 1.')
        iter = int(1)
    
    batch_file = MELTSModel + 'batch'
    dataname = f'{output_file}.csv'
    sim_metadata_name = f'{output_file}.txt'
    
    _validate_existing_files(dataname, sim_metadata_name, indexer)
    
    def PTF_initialize(conditions, length=simcycle):
        """Initialize Pressure-Temperature-fO2 arrays for cooling runs."""
        out_array = np.zeros((length, np.shape(conditions)[1] + 3))
        out_array[:, 0] = np.random.uniform(*Prange, size=length)  # Pressure in bars
        out_array[:, 1] = 1925 + 20 + np.arange(length)  # Temperature in K (starting high, decreasing)
        out_array[:, 2] = np.random.uniform(-5, 5, size=length)  # fO2 offset from FMQ
        out_array[:, 3:] = conditions
        return out_array
    
    keys = np.array(["Pressure", "Temperature", 'fO2'] + list(col_dict.keys()))
    
    for j in range(iter):
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
            end=end, fxtal=fxtal,
            EnsembleLocation=EnsembleLocation, WSL=True,
            compression=False, delta=-1
        )
        
        # Update batch names with metadata
        for i, name in enumerate(batchname):
            batchname[i] = f"{pull_number(str(indices[i]))}:{random_char(4)}:{name}"
        
        # Import results
        alphamelts_functions.import_MELTS_components(
            EnsembleLocation=EnsembleLocation, batchname=batchname,
            indexer=indexer, fO2Arr=in_array[:, 2], dataname=dataname
        )