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

# Import utility functions
from ngibbs.utils.string_utils import pull_number, random_char
from ngibbs.utils.file_utils import count_csv_rows, count_file_lines

# Import indexer and header generation
from builder.indexer import DatasetIndexer, verify_csv_headers_match
from recipes.settings import internal_scratch_dir

# Set up paths
EnsembleLocation = str(internal_scratch_dir())
os.makedirs(EnsembleLocation, exist_ok=True)


def _repair_torn_tail_and_count(path):
    """
    Count confirmed (newline-terminated) lines in an append-only metadata file,
    truncating a torn trailing line first if a previous run crashed mid-write.

    Uses a backward byte-seek to find the last newline rather than reading the
    whole file into memory, so repair itself stays cheap even on a multi-GB
    metadata file; the line count that follows is still a single sequential
    scan (count_file_lines), done once here rather than once per cycle.
    """
    with open(path, 'rb') as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size == 0:
            return 0
        f.seek(-1, os.SEEK_END)
        ends_clean = f.read(1) == b'\n'
        if not ends_clean:
            pos = size - 1
            while pos > 0:
                pos -= 1
                f.seek(pos)
                if f.read(1) == b'\n':
                    break
            else:
                pos = -1  # no newline anywhere; the whole file was one torn line
            truncate_at = pos + 1

    if not ends_clean:
        with open(path, 'r+b') as f:
            f.truncate(truncate_at)

    return count_file_lines(path)


class MemmapWriter:
    """
    Fixed-size binary sink for import_MELTS_components, replacing the CSV append path.

    Pre-allocates `{output_file}.npy` sized to `target_rows` (the same layout
    BigMetaTable eventually builds from CSV, so BigMetaTable can load it directly
    with rebuild_memmap=False).

    Resuming after an interrupted run (crash or Ctrl+C) is driven by the metadata
    `.txt` file, not by trusting the cursor sidecar file blindly: within write(),
    the array segment is written and flushed *before* its metadata lines are
    appended, so "N confirmed metadata lines" always implies "the first N array
    rows are durably on disk" - the reverse is never true. Re-deriving the cursor
    from the metadata file's confirmed line count at startup (self-healing a torn
    last line if one crashed mid-write) is therefore always safe, and is a single
    sequential scan done once per resume rather than once per cycle.
    """

    def __init__(self, output_file, target_rows, n_cols):
        self.path = f'{output_file}.npy'
        self.cursor_file = f'{output_file}_cursor.txt'
        self.sim_metadata_name = f'{output_file}.txt'

        if os.path.exists(self.path):
            self.array = np.lib.format.open_memmap(self.path, mode='r+')
            if self.array.shape[1] != n_cols:
                raise ValueError(
                    f"Existing memmap {self.path} has {self.array.shape[1]} columns, "
                    f"expected {n_cols}. Indexer/phase list changed since this run started?"
                )
            if self.array.shape[0] != target_rows:
                raise ValueError(
                    f"Existing memmap {self.path} has {self.array.shape[0]} rows, "
                    f"expected target_rows={target_rows}. Remove it or use a new output_file."
                )
        else:
            self.array = np.lib.format.open_memmap(
                self.path, mode='w+', dtype=np.float32, shape=(target_rows, n_cols))

        confirmed_rows = 0
        if os.path.exists(self.sim_metadata_name):
            confirmed_rows = _repair_torn_tail_and_count(self.sim_metadata_name)

        recorded_cursor = None
        if os.path.exists(self.cursor_file):
            with open(self.cursor_file, 'r') as f:
                recorded_cursor = int(f.read().strip())

        if recorded_cursor is not None and recorded_cursor != confirmed_rows:
            print(f"[MemmapWriter] Cursor file said {recorded_cursor} rows but {self.sim_metadata_name} "
                  f"has {confirmed_rows} confirmed lines (likely an interrupted previous run). "
                  f"Resuming from {confirmed_rows}.")

        self.cursor = confirmed_rows
        with open(self.cursor_file, 'w') as f:
            f.write(str(self.cursor))

    @property
    def full(self):
        return self.cursor >= self.array.shape[0]

    def write(self, rows, meta_lines, sim_metadata_name):
        """Write as many of rows/meta_lines as fit before the memmap fills. Returns rows written."""
        n_avail = self.array.shape[0] - self.cursor
        n = min(n_avail, len(rows))
        if n <= 0:
            return 0
        self.array[self.cursor:self.cursor + n] = rows[:n]
        self.array.flush()
        with open(sim_metadata_name, 'a') as f:
            f.writelines(meta_lines[:n])
        self.cursor += n
        with open(self.cursor_file, 'w') as f:
            f.write(str(self.cursor))
        return n

    def finalize(self):
        """
        Shrink the memmap in place from its preallocated target_rows down to the
        rows actually written (self.cursor). A run stopped before target_rows was
        reached (early Ctrl+C, or simply not resumed further) otherwise leaves the
        file at full target_rows size padded with unwritten (zero) rows, which
        BigMetaTable has no way to distinguish from real data - call this before
        handing a not-yet-full dataset to BigMetaTable or the merge script.
        """
        if self.cursor == self.array.shape[0]:
            return  # already exactly full; nothing to trim
        tmp_path = self.path + '.trimtmp'
        trimmed = np.lib.format.open_memmap(
            tmp_path, mode='w+', dtype=self.array.dtype, shape=(self.cursor, self.array.shape[1]))
        trimmed[:] = self.array[:self.cursor]
        trimmed.flush()
        del trimmed
        del self.array
        os.replace(tmp_path, self.path)
        self.array = np.lib.format.open_memmap(self.path, mode='r+')


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


def _process_compositions(compositions, col_dict, simcycle, MELTSModel, zeroOxides = ['MnO', 'NiO'], replace_CO2=True):
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
        if 'H2O' not in zeroOxides:
            compositions[hydrous, col_dict['H2O']] = np.abs(np.random.normal(size=len(hydrous), scale=0.5))
        out_mask = np.ones(simcycle)
        out_mask[hydrous] = 0
        remaining_idx = remaining_idx[out_mask.astype(bool)]
        soaked = np.random.choice(remaining_idx, size=int(simcycle/3), replace=False)
        if 'H2O' not in zeroOxides:
            compositions[soaked, col_dict['H2O']] = np.random.uniform(size=len(soaked), high=5)
    if (MELTSModel == '120') and ('CO2' not in zeroOxides) and replace_CO2:
        allWet = np.unique(np.append(hydrous, soaked))
        print(f"Adding CO2 to {allWet} hydrous/soaked compositions for MELTSModel {MELTSModel}.")
        co2_hydrous = np.random.choice(allWet, size=int(simcycle/3), replace=False)
        CO2_arr = np.zeros((simcycle, 1))
        CO2_arr[co2_hydrous] = np.random.uniform(size=len(co2_hydrous), high=0.5).reshape(-1, 1)
        random_draws = np.random.uniform(high=1, size = allWet.shape[0])
        co2_soaked = allWet[random_draws < 0.1]
        if len(co2_soaked) > 0:
            CO2_arr[co2_soaked] = np.random.uniform(size=len(co2_soaked), high=1).reshape(-1, 1)
        compositions[:,col_dict['CO2']] = CO2_arr[:, 0].flatten() # Add co2 column earlier
    if (MELTSModel == '102') and ('H2O' not in zeroOxides):
        hydrous = np.random.choice(np.arange(simcycle), size=int(simcycle/2), replace=False)
        compositions[hydrous, col_dict['H2O']] = np.abs(np.random.normal(size=len(hydrous), scale=0.5))
    # 'p': all anhydrous — already zeroed above

    # Set zeroOxides to zero

    for oxide in zeroOxides:
        try:
            compositions[:, col_dict[oxide]] = 0
        except KeyError:
            print(f"No '{oxide}' in compositions to zero out.")
    
    # For pMELTS, randomly cancel out poorly handled volatiles. Consider excluding these oxides altogether from pMELTS.
    if MELTSModel == 'p':
        NoKP = np.random.randint(simcycle, high=None, size=int(simcycle/2))
        compositions[NoKP, col_dict['P2O5']] = 0
        compositions[NoKP, col_dict['K2O']] = 0
    
    # Set small Cr2O3 to zero (mostly for felsic compositions)
    smallCr2O = compositions[:, col_dict['Cr2O3']] < 0.01
    compositions[smallCr2O, col_dict['Cr2O3']] = 0

    # Add Noise: 
    compositions *= np.random.normal(loc=1.0, scale=0.05, size=compositions.shape) # Add 5% noise to all oxides
    
    # Normalize to 100%
    compositions = 100 * compositions / (np.sum(compositions, axis=1))[:, np.newaxis]
    
    return compositions
    

def alphaMELTScooling(output_file, MELTSModel, GEOROC, col_dict, indexer, itercode='a1', simcycle=50, fxtal=False,
                      ExFailures=False, zeroOxides=['MnO', 'NiO'], startT=1925, max_liquid_fraction=100, end=700,
                      Prange=None, delta=-1, Oxygen='Closed', replace_CO2=True, af=None, target_rows=None):
    """
    Perform ensemble MELTS cooling calculations with random compositions.

    Parameters
    ----------
    output_file : str
        Base name for output files (without extension)
    MELTSModel : str
        MELTS model version ('102', '110', '120', 'p')
    GEOROC : np.ndarray
        Array of GEOROC compositions (first column is index, rest are oxides).
        If target_rows is given, GEOROC is expected to already be pre-mixed to
        the desired compositional-group ratio (see composition_pool.build_ratio_pool),
        since every cycle draws simcycle rows from it uniformly at random.
    col_dict : dict
        Dictionary mapping oxide names to column indices in compositions
    indexer : DatasetIndexer
        DatasetIndexer object for the dataset
    itercode : str, default='a1'
        Code for the total iteration (used for tracking progress. a for all melts, m for mafics).
        Ignored (beyond its use as a log label) when target_rows is given.
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
    Oxygen : str, default='Closed'
        Oxygen fugacity condition ('Closed' or 'Open')
    target_rows : int, optional
        If given, ignore itercode's iteration count and instead cycle until a
        memmap of this many rows (`{output_file}.npy`) is filled, resuming from
        a persisted cursor rather than the legacy progress-file/iteration scheme.
    """

    iterLetter = itercode[0]

    # Set pressure range and end temperature based on MELTS model
    if Prange is None:
        if MELTSModel == 'p':
            Prange = [8000, 30000]  # Pressure in bars
        else:
            Prange = [1, 12000]  # Pressure in bars

    batch_file = MELTSModel + 'batch'
    dataname = f'{output_file}.csv'
    sim_metadata_name = f'{output_file}.txt'

    memmap_writer = None
    iter = None
    start_iter = 0
    existing_lines = []
    progress_file = None

    if target_rows is not None:
        memmap_writer = MemmapWriter(output_file, target_rows, indexer.get_max_index() + 1)
    else:
        iter = int(itercode[1:])
        if iter <= 0 or iter != int(iter):
            print('Must have integer positive nonzero for iteration "iter" argument. Substituting 1.')
            iter = int(1)

        progress_file = f'{output_file}_progress.txt'
        _validate_existing_files(dataname, sim_metadata_name, indexer)

        # Initialize or resume from progress file
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    all_lines = f.read().strip().split('\n')

                if all_lines and all_lines[0]:  # Check if file has content
                    # Find and extract the line with current letter code if it exists
                    for line in all_lines:
                        if line.strip().startswith(iterLetter):
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                start_iter = int(parts[1].split('/')[0])
                                print(f"Resuming {iterLetter} from iteration {start_iter}/{iter}")
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
        if Oxygen.lower() == 'closed':
            ferric_to_ferrous = 0.8998084799181955  # wt ratio conserving Fe atoms
            logfo2 = np.random.uniform(-5, 5, size=length)
            logR = (logfo2 * 0.2) - 1
            R32 = 10 ** logR
            R_chosen = R32 / (R32 + 1)  # Fe3+/FeT molar ratio coupled to fO2
            ferric = conditions[:, col_dict['FeO']] * R_chosen * (1 / ferric_to_ferrous)
            conditions[:, col_dict['FeO']] = conditions[:, col_dict['FeO']] * (1 - R_chosen)
            total = (np.sum(conditions, axis=1) + ferric).reshape(-1, 1)
            conditions = conditions * (100 / total)
            ferric = ferric * (100 / total.flatten())
            out_array[:, 2] = ferric
        elif Oxygen.lower() == 'open':
            out_array[:, 2] = np.random.uniform(-5, 5, size=length)  # fO2 offset from FMQ
        else:
            raise ValueError(f"Oxygen condition '{Oxygen}' not recognized. Use 'Closed' or 'Open'")
        out_array[:, 3:] = conditions
        return out_array

    if Oxygen.lower() == 'closed':
        keys = np.array(["Pressure", "Temperature", 'Fe2O3'] + list(col_dict.keys()))
    elif Oxygen.lower() == 'open':
        keys = np.array(["Pressure", "Temperature", 'fO2'] + list(col_dict.keys()))

    def _run_cycle():
        # Select random compositions
        choices = np.random.randint(np.shape(GEOROC)[0], size=simcycle, dtype=int)
        compositions = GEOROC[choices, 1:].copy()
        indices = GEOROC[choices, 0]

        # Process and normalize compositions
        compositions = _process_compositions(compositions, col_dict, simcycle, MELTSModel, zeroOxides=zeroOxides, replace_CO2=replace_CO2)

        # Initialize PTX conditions
        in_array = np.round(PTF_initialize(compositions, length=simcycle), 2)

        # Set batch names
        batchname = np.full(simcycle, batch_file, dtype=object)

        # Run ensemble simulation
        af.forward_ensemble(
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
        failure_IDs = af.import_MELTS_components(
            EnsembleLocation=EnsembleLocation, batchname=batchname,
            indexer=indexer, fO2Arr=in_array[:, 2], dataname=dataname,
            max_liquid_fraction=max_liquid_fraction, memmap_writer=memmap_writer
        )

        if ExFailures:
            af.pick_exsolution_failure(EnsembleLocation, in_array, keys, batchname=batchname,
                                dataname=dataname, faultIDs=failure_IDs)

    if memmap_writer is not None:
        try:
            while not memmap_writer.full:
                _run_cycle()
                print(f"[{output_file}] {memmap_writer.cursor}/{memmap_writer.array.shape[0]} rows filled")
        except KeyboardInterrupt:
            print(f"[{output_file}] Interrupted at {memmap_writer.cursor}/{memmap_writer.array.shape[0]} rows. "
                  f"Re-run with the same output_file/target_rows to resume, or call "
                  f"MemmapWriter(...).finalize() to trim the file to the rows actually written.")
            raise
    else:
        for j in range(start_iter, iter):
            _run_cycle()

            # Update progress file
            current_iter = j + 1
            progress_content = '\n'.join(existing_lines) if existing_lines else ''
            if progress_content:
                progress_content += f"\n{iterLetter} {current_iter}/{iter}"
            else:
                progress_content = f"{iterLetter} {current_iter}/{iter}"

            with open(progress_file, 'w') as f:
                f.write(progress_content)

def alphaMELTSERph(output_file, GEOROC, col_dict, indexer, itercode='a1', simcycle=50, fxtal=False,
                      ExFailures=False, zeroOxides=['MnO', 'NiO'],
                      settingsLocation=None, alphameltsLocation=None,
                      max_liquid_fraction=120, af=None):
    """
    Perform ensemble MELTS calculations controlled by settings and batch files with random compositions.

    Parameters
    ----------
    output_file : str
        Base name for output files (without extension)
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
    settingsLocation : str, optional
        Path to settings.txt; defaults to af.settingsLocation
    alphameltsLocation : str, optional
        Path to alphamelts binary; defaults to af.alphameltsLocation
    max_liquid_fraction : int, default=120
        Maximum allowed liquid fraction for the simulation
    af : module
        alphamelts_functions module to use for simulation (required)
    """
    if af is None:
        raise ValueError("af (alphamelts_functions module) must be provided")
    if settingsLocation is None:
        settingsLocation = af.settingsLocation
    if alphameltsLocation is None:
        alphameltsLocation = af.alphameltsLocation

    iterLetter = itercode[0]
    iter = int(float(itercode[1:]))

    MELTSModel = 'ph'
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
        out_array = np.zeros((length, np.shape(conditions)[1] + 2))
        conditions = conditions.copy() * np.random.uniform(0.9, 1.1, size=conditions.shape) # Add some random noise to compositions to expand the range of conditions sampled
        out_array[:, 0] = 1400 + ((2000-1300)/length) * (np.arange(length))  # Temperature in K (starting high, decreasing)
        ferric_to_ferrous = 0.8998084799181955 #wt ratio conserving Fe atoms
        R = np.linspace(0,0.15,100) # Proportion of total FeO wt to Fe2O3
        P = (np.exp(-20*R))#)+(0.5/7)) #Empiracally fit elsewhere, arbitrary
        P = P/P.sum() 
        R_chosen = np.random.choice(R, size=length, replace=True, p=P)
        ferric = conditions[:,col_dict['FeO']]*R_chosen*(1/ferric_to_ferrous)
        conditions[:,col_dict['FeO']] = conditions[:,col_dict['FeO']]*(1-R_chosen)
        water_sampling = np.random.uniform(0, 0.1, size=int(length*0.5))  # Randomize H2O content up to 
        water_sampling = np.append(water_sampling, np.random.uniform(0.1, 3, size=int(length*0.25)))  # Add some higher H2O values up to 5%
        water_sampling = np.append(water_sampling, np.zeros(length-water_sampling.shape[0]))  # Add some higher anhydrous samples
        conditions[:,col_dict['H2O']] = np.random.choice(water_sampling, size=length, replace=False) # weird shuffle workaround
        total = (np.sum(conditions, axis = 1) + ferric).reshape(-1,1) # Renormalize to new wt%
        #print(total.shape)
        conditions = conditions*(100/total)
        #print(conditions.shape)
        #print(ferric.shape)
        ferric = ferric*(100/total.flatten())
        #print(ferric.shape)
        #out_array[:, 2] = np.random.uniform(-5, 5, size=length)  # fO2 offset from FMQ
        out_array[:, 1] = ferric
        out_array[:, 2:] = conditions
        return out_array
    
    keys = np.array(["Temperature", 'Fe2O3'] + list(col_dict.keys()))

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
        af.forward_ensemble(
            in_array, keys=keys, batchname=batchname,
            only_phases=indexer.get_phase_list(),
            end=None, fxtal=fxtal,
            EnsembleLocation=EnsembleLocation, WSL=True,
            compression=False, delta=None, settingsLocation=settingsLocation, alphameltsLocation=alphameltsLocation
        )
        
        # Update batch names with metadata
        for i, name in enumerate(batchname):
            batchname[i] = f"{pull_number(str(indices[i]))}:{random_char(4)}:{name}"
        
        # Import results
        failure_IDs = af.import_MELTS_components(
            EnsembleLocation=EnsembleLocation, batchname=batchname,
            indexer=indexer, fO2Arr=in_array[:, 2], dataname=dataname,
            max_liquid_fraction=max_liquid_fraction
        )

        if ExFailures:
            af.pick_exsolution_failure(EnsembleLocation, in_array, keys, batchname=batchname,
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