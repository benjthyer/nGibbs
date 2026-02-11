"""
MELTS output file parsing.

Contains BigMetaTable class for handling large MELTS datasets with memory-mapped arrays.
Moved from BigMetaTableLibrary.py for better organization.
"""

import code
import os
import csv
import gc
import time
import mmap
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

# Ensure src is on path
import sys
from pathlib import Path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import from refactored modules
from builder.indexer import DatasetIndexer
from nMELTS.utils.string_utils import pull_number
from nMELTS.utils.math_utils import blur_binary_boundaries
from nMELTS.utils.file_utils import move_file


def strip_filename_suffixes(filename, suffixes=None):
    """
    Remove suffixes from a filename. Helper for BigMetaTable looking for columns to describe its dataset
    
    Parameters:
    - filename (str): The filename to process
    - suffixes (list): List of suffixes to remove (e.g., ['.npy', '.csv', '_working'])
                      If None, uses default list of common suffixes
    - strip_trailing_underscores (bool): Whether to remove trailing underscores after suffix removal
    
    Returns:
    - str: Filename with suffixes and optionally trailing underscores removed
    
    Examples:
        >>> strip_filename_suffixes('table_temp_filtered.csv')
        'table.npy'
        >>> strip_filename_suffixes('data_working.npy')
        'data.npy'
        >>> strip_filename_suffixes('file.npy', suffixes=['.npy', '.csv'])
        'file.npy'
    """

    filetype = ('.' + filename.rsplit('.', 1)[-1]) if '.' in filename else ''

    if suffixes is None:
        # Default suffixes used by BigMetaTable
        suffixes = [
            '.npy', '.csv', '.txt', '.pkl',
            'working', 'temp', 'filtered', 'resampled',
            'preprocessed', 'split', 'remaining'
        ]
    
    result = filename[:-len(filetype)].rstrip('_') # Get filename without extension
    
    # Keep removing suffixes until none match (handles multiple suffixes)
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if result.endswith(suffix):
                result = result[:-len(suffix)].rstrip('_')
                changed = True
                break
    
    return result.rstrip('_') + filetype

class BigMetaTable:
    def __init__(self, filename, read_dir=None, memmap_mode='r+', rebuild_memmap=False,
                 allow_differing_lengths=False, headers=None):
        
        self.filename = filename
        self.memmap_file = self.filename + '.npy'
        self.csv_file =  self.filename + '.csv'
        self.txt_file =  self.filename + '.txt'
        

        if read_dir is None:
            self.read_dir = ''
        else:
            self.read_dir = read_dir
            if os.path.exists(self.read_dir + self.memmap_file) and not os.path.exists(self.memmap_file):
                print(f'Moving {self.read_dir + self.memmap_file} to current directory')
                move_file(src_path=self.read_dir + self.memmap_file, dst_path=self.memmap_file, overwrite=False) # bring memmap into working drive.
            if os.path.exists(self.read_dir + self.csv_file) and not os.path.exists(self.csv_file):
                print(f'Moving {self.read_dir + self.csv_file} to current directory')
                move_file(src_path=self.read_dir + self.csv_file, dst_path=self.csv_file, overwrite=False) # bring memmap into working drive.
            if os.path.exists(self.read_dir + self.txt_file) and not os.path.exists(self.txt_file):
                print(f'Moving {self.read_dir + self.txt_file} to current directory')
                move_file(src_path=self.read_dir + self.txt_file, dst_path=self.txt_file, overwrite=False) # bring memmap into working drive.
            # Also move header file if it exists
            """header_file = self.memmap_file.replace('.npy', '_headers.pkl')
            if os.path.exists(self.read_dir + header_file) and not os.path.exists(header_file):
                print(f'Moving {self.read_dir + header_file} to current directory')
                move_file(src_path=self.read_dir + header_file, dst_path=header_file, overwrite=False)"""
           
                

        
        working_text = False # Do we change text file name to protect it after loading?

      
        self.meta = []
        self.run_indices = []
        self.MELTSversion = []
        self.metadata = []
        
        t_start = time.time()
        
        print(f"{self.memmap_file}  {['DOES NOT EXIST', 'exists'][int(os.path.exists(self.memmap_file))]}")

        # --- Build memmap if needed
        if rebuild_memmap or not os.path.exists(self.memmap_file):
            with open(self.csv_file, newline='') as f:
                reader = csv.reader(f)
                self.header = next(reader)
            with open(self.csv_file, newline='') as f:
                file_rows = sum(1 for _ in f) - 1
                self.file_rows = file_rows
            
            # Build DatasetIndexer from headers
            self.indexer = DatasetIndexer(self.header)
        
                
            total_rows = file_rows
            self.total_rows = total_rows
            self.filename = filename + '_working'
            self.memmap_file = self.filename + '.npy'
            working_text = True
    
            self._csv_to_memmap()
            self.table = np.load(self.memmap_file, mmap_mode=memmap_mode, allow_pickle=True)
            
            """# Save headers for future loading
            
            header_file = self.memmap_file.replace('.npy', '_headers.pkl')
            with open(header_file, 'wb') as f:
                pickle.dump(self.header, f)"""
            

        else:
            self.table = np.load(self.memmap_file, mmap_mode=memmap_mode, allow_pickle=True)
            def _infer_headers(csv_file):
                with open(csv_file, newline='') as f:
                    reader = csv.reader(f)
                    self.header = next(reader)
            file_rows = self.table.shape[0]
            if headers is not None:
                self.header = headers
                if len(self.header) != self.table.shape[1]:
                    # Try to load header from a saved pickle file
                    if os.path.exists(self.csv_file):
                        _infer_headers(self.csv_file)
                    else:
                        raise ValueError(f"Header length {len(self.header)} does not match table columns {self.table.shape[1]} and {self.csv_file} does not exist to infer headers from.")
            elif os.path.exists(self.csv_file):
                _infer_headers(self.csv_file)
            elif os.path.exists(strip_filename_suffixes(self.csv_file)):
                _infer_headers(strip_filename_suffixes(self.csv_file))
            else:
                raise ValueError(f"Headers are not provided and {self.csv_file} or {strip_filename_suffixes(self.csv_file)} does not exist to infer headers from.")
            
            self.file_rows = file_rows
           
            # Build DatasetIndexer from loaded headers
            self.indexer = DatasetIndexer(self.header)
            
            total_rows = file_rows
            self.total_rows = total_rows
            
        if not os.path.exists(self.txt_file):
            print('NO TEXT FILE DETECTED: PROCEEDING WITH PLACEHOLDER .txt FILE. SOME FILTERING AND PROCESSING FEATURES CANNOT BE USED')
            self.read_metadata = False
            with open(self.txt_file, 'w') as f:
                for i in range(file_rows):
                    value = 1_000_000.000001 + i * 0.000001
                    line = f"{value:.6f}:XXXX:555batch\n"
                    f.write(line)
        else:
            self.read_metadata = True

        print(f"Preparing to read {file_rows} rows of text")
        
        # --- Read TXT metadata (only the slice)
        with open(self.txt_file, 'r') as f:
            for i, line in tqdm(enumerate(f), desc = 'Reading Text...', total=int(file_rows)):
                line = line.strip()
                self.meta.append(line)
                parts = line.split(' ', 1)
                id_parts = parts[0].split(':')
                if len(id_parts) == 1:
                    self.run_indices.append('')
                    self.MELTSversion.append(id_parts[0])
                else:
                    self.run_indices.append(id_parts[0] + id_parts[1])
                    self.MELTSversion.append(id_parts[2])
                self.metadata.append(parts[1] if len(parts) > 1 else '')
        print(f"[TIMER] Parsed {total_rows} of {i+1} Metadata to Lists: completed in {time.time() - t_start:.2f} seconds")
        
        if working_text:
            self.txt_file = self.filename + '.txt' # Now change name if we are using a 'working' version after reading text.
            
        # --- Convert lists to arrays
        t_start = time.time()
        self.meta = np.array(self.meta, dtype=str)
        self.run_indices = np.array(self.run_indices, dtype=str)
        self.MELTSversion = np.array(self.MELTSversion, dtype=str)
        self.metadata = np.array(self.metadata, dtype=object)
        gc.collect()
        print(f"[TIMER] Metadata Lists to arrays: completed in {time.time() - t_start:.2f} seconds")

        

        # --- Handle blurred binaries
        if os.path.exists(filename + 'blurredbinaries.npy'):
            self.blurredbinaries = np.load(filename + 'blurredbinaries.npy', mmap_mode='r+')
            if self.blurredbinaries.shape[0] != self.table.shape[0]:
                print('Blurred binaries have differing rows than main table! Deleting Blurredbinaries')
                del self.blurredbinaries
                gc.collect()
                self.blurredbinaries = None
        else:
            self.blurredbinaries = None

        self.csv_file = self.filename + '.csv'
        self.filename = filename + '_working'
        self.memmap_file = self.filename + '.npy'
        self.txt_file = self.filename + '.txt'

        len_problem = False
        row_expected = self.table.shape[0]
        if len(self.meta) != row_expected:
            print(f'WARNING: Metadata length {len(self.meta)} does not match data rows: {row_expected}')
            len_problem = True
        if len(self.run_indices) != row_expected:
            print(f'WARNING: Run Indices length {len(self.run_indices)} does not match data rows: {row_expected}')
            len_problem = True
        if len(self.MELTSversion) != row_expected:
            print(f'WARNING: MELTSversion length {len(self.MELTSversion)} does not match data rows: {row_expected}')
            len_problem = True
        if len(self.metadata) != row_expected:
            print(f'WARNING: Metadata length {len(self.metadata)} does not match data rows: {row_expected}')
            len_problem = True

        if len_problem and not allow_differing_lengths:
            rows = self.table.shape[0]
            del self.table
            gc.collect()
            raise ValueError(f'Metadata length does not match data rows: {rows}')

    
    def _csv_to_memmap(self, chunk_size=100000):
        """Convert CSV to memmap, storing header."""
        t_start = time.time()
        num_cols = len(self.header)

        # Count total rows (done_earlier)
        file_rows = self.file_rows
        total_batches = np.ceil(file_rows/chunk_size)

        nrows = self.total_rows

        print(f"[INFO] Preparing to process {nrows} of {file_rows} total rows with {num_cols} columns...")

        # Create memmap with provisional shape (max size)
        if os.path.exists(self.memmap_file):
            try:
                os.remove(self.memmap_file)
                print(f"Existing memmap file {self.memmap_file} removed.")
            except OSError as exc:
                print(f"[WARN] Failed to remove existing memmap file {self.memmap_file}: {exc}")
        memmap = np.lib.format.open_memmap(
            self.memmap_file,
            mode='w+',
            dtype=np.float32,
            shape=(nrows, num_cols)
        )


        valid_row_count = 0
        skip_indices = []

        with open(self.csv_file, newline='') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            buffer = []

            for i, row in enumerate(tqdm(reader, desc=f'Reading {self.csv_file}', leave=False, total=int(total_batches))):
                try:
                    values = [float(x) if x.strip() != '' else 0.0 for x in row]
                    assert len(values) == num_cols, f"Row {i} has {len(values)} columns, expected {num_cols}\nValues: {values}"
                except:
                    print(f"Failure at row {i}: {row}")
                    break
                buffer.append(values)
                
                if len(buffer) >= chunk_size:
                    chunk = np.array(buffer, dtype=np.float32)
                    memmap[valid_row_count:valid_row_count + len(chunk)] = chunk
                    valid_row_count += len(chunk)
                    buffer.clear()

            if buffer:
                chunk = np.array(buffer, dtype=np.float32)
                memmap[valid_row_count:valid_row_count + len(chunk)] = chunk
                valid_row_count += len(chunk)

        print(f"[INFO] Loaded {valid_row_count} valid rows into memmap")# (skipped {len(skip_indices)})")
        print(f"[TIMER] Completed in {time.time() - t_start:.2f} seconds")

        memmap.flush()
        del memmap
        gc.collect()
            
    def _try_close_memmap_array(self, arr):
        """Try to close underlying mmap for numpy memmap-like arrays. Not functioning nor used 10/3/25"""
        if arr is None:
            return
        # numpy.memmap objects (and arrays from open_memmap) may expose _mmap
        m = getattr(arr, '_mmap', None)
        if m is not None:
            try:
                m.close()
            except Exception:
                pass
        # Sometimes the mmap object is in .base
        base = getattr(arr, 'base', None)
        if base is not None:
            # base might itself be an mmap or numpy.memmap
            if hasattr(base, '_mmap'):
                try:
                    base._mmap.close()
                except Exception:
                    pass
            elif isinstance(base, mmap.mmap):
                try:
                    base.close()
                except Exception:
                    pass
        # As a last-ditch attempt, delete the object and gc
        try:
            del arr
        except Exception:
            pass
        gc.collect()
            
    def _mark_nan_rows_for_deletion(self, nan_mask):
        """Delete corresponding metadata rows after filtering NaN rows."""
        nan_indices = np.where(nan_mask)[0]
        self.meta = np.delete(self.meta, nan_indices)
        self.run_indices = np.delete(self.run_indices, nan_indices)
        self.MELTSversion = np.delete(self.MELTSversion, nan_indices)
        self.metadata = np.delete(self.metadata, nan_indices)

    
    def save_csv_streaming(self, name, chunk_size=100000):
        with open(name + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.header)  # write header

            for i in tqdm(range(0, self.table.shape[0], chunk_size), desc="Saving Large csv file ...", leave=False):
                chunk = self.table[i:i+chunk_size]
                writer.writerows(chunk)

    def run_ind(self, indices):
        return self.run_indices[np.array(indices)]

    def version(self, indices):
        return self.MELTSversion[np.array(indices)]

    def ID(self, code):
        return np.where(self.run_indices == code)[0] # Boolean for clarity
        #return np.arange(len(self.metadata))[self.run_indices == code]

    def delete(self, indices, save_text = True):
        """Efficient delete that preserves memory-mapping by rewriting."""
        keep_mask = np.ones(self.table.shape[0], dtype=bool)
        keep_mask[indices] = False
        new_shape = (np.sum(keep_mask), self.table.shape[1])
        
        # Apply operation to blurred binary data first to call assertion before we get too deep.
        if self.blurredbinaries is not None:
            binary_name = self.filename + 'blurredbinaries.npy'
            if os.path.exists(binary_name):
                temp_binary_filename = self.filename + 'blurredbinaries_temp.npy'
            else: 
                temp_binary_filename = binary_name
                
            assert np.shape(self.blurredbinaries)[0] == np.shape(self.table)[0], "Shape of binary and main tables are not equal!"
            new_binary_memmap = np.lib.format.open_memmap(temp_binary_filename, dtype='float32', mode='w+', shape=(new_shape[0], self.blurredbinaries.shape[1]))

            # Write filtered rows to new memmap file
            new_binary_memmap[:] = self.blurredbinaries[keep_mask]
            new_binary_memmap.flush()

            # Clear memory mapping and update file
            del new_binary_memmap, self.blurredbinaries
            gc.collect()
            
            if os.path.exists(self.filename + 'blurredbinaries_temp.npy'):
                os.replace(temp_binary_filename, binary_name)

            # Update reference to new memmap in read mode
            self.blurredbinaries = np.load(binary_name, mmap_mode='r+')
        
        # Prepare output memmap
        temp_filename = self.filename + '_temp.npy'
        new_memmap = np.lib.format.open_memmap(temp_filename, dtype='float32', mode='w+', shape=new_shape)

        # Write filtered rows to new memmap file
        new_memmap[:] = self.table[keep_mask]
        new_memmap.flush()
        
        # Clear memory mapping and update file
        del new_memmap
        del self.table
        gc.collect()
        os.replace(temp_filename, self.memmap_file)
        
        # Update reference to new memmap
        self.table = np.load(self.memmap_file, mmap_mode='r+')

        # Update metadata arrays (all in RAM)
        if save_text:
            self.meta = self.meta[keep_mask]
            self.run_indices = self.run_indices[keep_mask]
            self.MELTSversion = self.MELTSversion[keep_mask]
            self.metadata = self.metadata[keep_mask]
            self.save_txt()


    def save(self, name=None, save_csv = True):
        if name is None:
            name = self.filename + '_filtered'
        np.save(name + '.npy', self.table)
        if self.blurredbinaries is not None:
            np.save(name + 'blurredbinaries.npy', self.blurredbinaries)
        if save_csv:
            self.save_csv_streaming(name = name)
        with open(name + '.txt', 'w') as f:
            for line in self.meta:
                f.write(line + '\n')
    
    def save_txt(self, name=None):
        if name is None:
            name = self.filename + '.txt'
        if '.txt' not in name:
            name += '.txt'
        with open(name, 'w') as f:
            for line in self.meta:
                f.write(line + '\n')
                
    def split(self, proportion):
        """
        Splits the BigMetaTable into two, moving a proportion of entries to a new BigMetaTable.

        Parameters:
        - proportion (float): Between 0 and 1. Fraction of entries to move to the new table.

        Returns:
        - new_table (BigMetaTable): A new BigMetaTable with the specified proportion of rows.
        """

        if not (0.0 < proportion < 1.0):
            raise ValueError("Proportion must be a float between 0 and 1 (exclusive).")

        total_rows = self.table.shape[0]
        num_to_move = int(total_rows * proportion)

        # Randomly select indices to move (without replacement)
        all_indices = np.arange(total_rows)
        move_indices = np.random.choice(all_indices, size=num_to_move, replace=False)
        keep_mask = np.ones(total_rows, dtype=bool)
        keep_mask[move_indices] = False
        keep_indices = all_indices[keep_mask]

        # Sort indices to preserve row order
        move_indices.sort()
        keep_indices.sort()

        # Create memmap files for both splits
        dtype = self.table.dtype
        col_count = self.table.shape[1]

        new_filename = f"{self.filename}_split"
        new_path = f"{new_filename}.npy"
        remaining_filename = f"{self.filename}_remaining"

        # Write moved data
        new_table_data = np.lib.format.open_memmap(
            new_path, mode='w+', dtype=dtype, shape=(num_to_move, col_count)
        )
        new_table_data[:] = self.table[move_indices]
        new_table_data.flush()

        #...and moved text
        with open(new_filename+'.txt', 'w') as f:
            for line in self.meta[move_indices]:
                f.write(line + '\n')
        
        # Save headers for new table
        
        new_header_file = new_path.replace('.npy', '_headers.pkl')
        with open(new_header_file, 'wb') as f:
            pickle.dump(self.header, f)
        
        # Write kept data
        remaining_data = np.lib.format.open_memmap(
            remaining_filename+'.npy', mode='w+', dtype=dtype, shape=(total_rows - num_to_move, col_count)
        )
        remaining_data[:] = self.table[keep_indices]
        remaining_data.flush()
        
        # Save headers for remaining table
        remaining_header_file = remaining_filename+'_headers.pkl'
        with open(remaining_header_file, 'wb') as f:
            pickle.dump(self.header, f)
        
        #...and kept text
        self.meta = self.meta[keep_indices]
        self.run_indices = self.run_indices[keep_indices]
        self.MELTSversion = self.MELTSversion[keep_indices]
        self.metadata = self.metadata[keep_indices]
            
        # Now clear memmaps, reinitialize new memmap, and replace memmaps for this object
        del new_table_data
        del remaining_data
        del self.table
        gc.collect()
        
        
        
        #self.table = np.load(remaining_filename+'.npy', mmap_mode='r+')
        self.filename = remaining_filename 
        self.csv_file = self.filename + '.csv'
        self.memmap_file = self.filename + '.npy'
        self.txt_file = self.filename + '.txt'
        
        self.save_txt()
        
        # Build new BigMetaTable
        new_table = BigMetaTable(new_filename)
        
        # Move BlurredBinaries
        if self.blurredbinaries is not None:
            new_table.blurredbinaries = np.lib.format.open_memmap(
                new_filename+'blurredbinaries.npy', mode='w+', dtype=np.float32, shape=(num_to_move, self.indexer.nphases)
                )
            new_table.blurredbinaries[:] = self.blurredbinaries[move_indices]
            new_table.blurredbinaries.flush()
            print(f"First Row blurredbinaries split Memmap write mode:{new_table.blurredbinaries[0]}")
            #Reopen blurred binaries for split data in read mode
            del new_table.blurredbinaries
            gc.collect()
            new_table.blurredbinaries = np.load(new_filename+'blurredbinaries.npy', mmap_mode = 'r')
            assert np.shape(new_table.blurredbinaries)[0] == np.shape(new_table.table)[0], "Shape of split binary and main tables are not equal!"
            print(f"First Row blurredbinaries split Memmap read mode:{new_table.blurredbinaries[0]}")
            
            
            #Update remaining blurredboundaries object
            binary_name = self.filename + 'blurredbinaries.npy'
            if os.path.exists(binary_name):
                temp_binary_filename = self.filename + 'blurredbinaries_temp.npy'
            else: 
                temp_binary_filename = binary_name
                
            new_binary_memmap = np.lib.format.open_memmap(temp_binary_filename, dtype=np.float32, mode='w+', shape=(total_rows - num_to_move, self.indexer.nphases))

            # Write filtered rows to new memmap file
            new_binary_memmap[:] = self.blurredbinaries[keep_indices]
            new_binary_memmap.flush()

            # Clear memory mapping and update file
            del new_binary_memmap, self.blurredbinaries
            gc.collect()
            
            if os.path.exists(self.filename + 'blurredbinaries_temp.npy'):
                os.replace(temp_binary_filename, binary_name)

            # Update reference to new memmap in read mode
            #self.blurredbinaries = np.load(binary_name, mmap_mode='r')
            
        del self
        gc.collect()
        newself = BigMetaTable(remaining_filename)
            
        return newself, new_table


    def filter_twophase(self, ExFail): # Function used to delete simulations that predicted multiple phases. Now deprecated, this is better handled in MELTS directly
        to_delete = []

        for i, failsim in enumerate(tqdm(ExFail.metadata, desc="Filtering Instances of Multiple Phases", leave=False)):
            if not failsim:
                continue

            clusters = failsim.split()
            lowlist = [int(c.split('-')[1]) for c in clusters]
            highlist = [pull_number(c.split('-')[2]) for c in clusters]
            lo, hi = min(lowlist), max(highlist)

            code = ExFail.run_indices[i]
            inds = self.ID(code)
            if not len(inds):
                continue

            stepnos = self.metadata[inds]
            stepnos_num = np.array([pull_number(stp) for stp in stepnos], dtype=int)
            mask = (stepnos_num >= lo) & (stepnos_num <= hi)

            if np.any(mask):
                to_delete.extend(inds[mask])

        if to_delete:
            self.delete(np.array(to_delete, dtype=int))
            print(f"Deleted: {len(to_delete)} Assemblages With 2 Phases")
        else:
            print("Deleted: 0 Assemblages With 2 Phases")

    def filter_jumps(self, thresholds = np.array([6,12,4,4,4,5])):
        IDs = np.unique(self.run_indices)
        to_delete = set()

        relevant_cols = [self.indexer.MELTS_indices['System_main']['Temperature'],
                 self.indexer.MELTS_indices['melts-liquid']['liq mass (gm)'],
                 self.indexer.MELTS_indices['melts-liquid']['wt% TiO2'],
                 self.indexer.MELTS_indices['melts-liquid']['wt% P2O5'],
                 self.indexer.MELTS_indices['melts-liquid']['wt% MnO'],
                 self.indexer.MELTS_indices['melts-liquid']['wt% NiO']]

        for ID in tqdm(IDs, desc="Trimming Unstable Jumps", leave=False):
            indices = self.ID(ID)
            
            if len(indices) < 2:
                continue
      
            data = self.table[np.ix_(indices, relevant_cols)]
            deltas = np.abs(data[:-1] - data[1:])

            # Check where any delta exceeds the threshold
            over_threshold = deltas > thresholds
            any_jump = np.any(over_threshold, axis=1)

            if np.any(any_jump):
                jump_index = np.argmax(any_jump)  # First occurrence
                to_delete.update(indices[jump_index:])  # delete from this index onward

        if to_delete:
            self.delete(np.array(sorted(to_delete), dtype=int))
            print(f"Deleted total {len(to_delete)} simulations with large jumps")
            
    def filter_jumps_with_phase_matrix(self,  thresholds = np.array([6,12,4,4,4,5,100])):
        IDs = np.unique(self.run_indices)
        to_delete = set()
        num_rows = np.shape(self.table)[0]
        num_phases = self.indexer.nphases
        broke_count = 0
        
        
        relevant_cols = [self.indexer.MELTS_indices['System_main']['Temperature'],
                 self.indexer.MELTS_indices['melts-liquid']['liq mass (gm)'],
                 self.indexer.MELTS_indices['melts-liquid']['wt% TiO2'],
                 self.indexer.MELTS_indices['melts-liquid']['wt% P2O5'],
                 self.indexer.MELTS_indices['melts-liquid']['wt% MnO'],
                 self.indexer.MELTS_indices['melts-liquid']['wt% NiO'],
                 self.indexer.MELTS_indices['System_main']['Pressure']]
        
        table_labels = ['Temperature','liq mass (gm)', 'wt% TiO2', 'wt% P2O5','wt% MnO','wt% NiO','Pressure']
        
        self.blurredbinaries = np.lib.format.open_memmap( # Flags of present phases
                self.filename+'blurredbinaries.npy',
                mode='w+',
                dtype=np.float32,
                shape=(num_rows, num_phases)
        )

        for ID in tqdm(IDs, desc="Trimming Unstable Jumps", leave=False):
            indices = self.ID(ID)
            
            if np.any(np.abs(np.diff(indices))>1):
                raise ValueError(f'Non-consectutive indices for ID: {ID}')
            
            if len(indices) < 3:
                print(f"Small simulation (Less that 3 assemblages) for ID: {ID}")
                to_delete.update(indices)
                continue
      
            data = self.table[np.ix_(indices, relevant_cols)]
            
            deltas = np.abs(data[:-1] - data[1:])
            # Check where any delta exceeds the threshold
            over_threshold = deltas > thresholds

            # Build Binary labels with blurred boundaries
            binary_mask = (self.table[np.ix_(indices,self.indexer.mass_indices)]>0).astype(np.float32)
            # Check where any delta exceeds the threshold
            any_jump = np.any(over_threshold, axis=1)

            if np.any(any_jump):
                jump_index = np.argmax(any_jump)  # First occurrence
                to_delete.update(indices[jump_index:])  # delete from this index onward
                #to_delete.update(real_indices[jump_index:])  # delete from this index onward
                if jump_index == 0:
                    print(f'We had to delete ALL of run {ID}')
                else:
                    self.blurredbinaries[indices[:jump_index]] = blur_binary_boundaries(binary_mask[:jump_index]) # Do not blur for excluded steps with potentially weird phases 
                    #self.blurredbinaries[real_indices[:jump_index]] = blur_binary_boundaries(binary_mask[:jump_index]) # Do not blur for excluded steps with potentially weird phases
            else:
                #self.blurredbinaries[real_indices] = blur_binary_boundaries(binary_mask)
                self.blurredbinaries[indices] = blur_binary_boundaries(binary_mask)
                
        #Now reload binary_labels in safer read mode.
        self.blurredbinaries.flush()
        del self.blurredbinaries
        gc.collect()        

        self.blurredbinaries = np.load(self.filename+'blurredbinaries.npy', mmap_mode='r+')
        
        if to_delete:
            self.delete(np.array(sorted(to_delete), dtype=int))
            print(f"Deleted total {len(to_delete)} simulations with large jumps")
            
    def filter_min_phase_proportion(self, min_proportion):
        """
        Delete all rows that contain phases whose presence falls below a minimum proportion.

        Parameters:
        - min_proportion (float): Minimum fraction of rows in which a phase must appear to be kept.
        """
        if not isinstance(min_proportion, (float, int)):
            raise TypeError("min_proportion must be a float between 0 and 1.")
        if not (0.0 < float(min_proportion) <= 1.0):
            raise ValueError("min_proportion must be a float between 0 and 1 (exclusive of 0).")

        min_proportion = float(min_proportion)
        total_rows = self.table.shape[0]
        if total_rows == 0:
            print("No rows available; skipping phase proportion filtering.")
            self.indexer.table_update(self.table)
            return

        delete_mask = np.zeros(total_rows, dtype=bool)
        phases_removed = []

        for phase, components in self.indexer.MELTS_indices.items():
            if phase in self.indexer.EXCLUDED_PHASES:
                continue

            if phase == 'melts-liquid':
                mass_col = components.get('liq mass (gm)')
            else:
                mass_col = components.get('mass (gm)')

            if mass_col is None:
                print(f"Skipping phase '{phase}': no mass column found.")
                continue

            phase_mask = self.table[:, mass_col] > 0
            phase_count = int(np.sum(phase_mask))
            phase_proportion = phase_count / total_rows

            if phase_proportion < min_proportion:
                delete_mask |= phase_mask
                phases_removed.append(phase)
                print(
                    f"Phase '{phase}' proportion {phase_proportion:.6f} below {min_proportion:.6f}; "
                    f"marking {phase_count} rows for deletion."
                )
            else:
                print(
                    f"Phase '{phase}' proportion {phase_proportion:.6f} meets threshold {min_proportion:.6f}; keeping."
                )

        if np.any(delete_mask):
            indices_to_delete = np.where(delete_mask)[0]
            print(
                f"Deleting {len(indices_to_delete)} rows from {len(phases_removed)} underrepresented phases: "
                f"{phases_removed}"
            )
            self.delete(indices_to_delete)
        else:
            print("No phases below threshold; no rows deleted.")

        self.indexer.table_update(self.table)
    
    def filter_phases_not_in_ml_indexer(self):
        """
        Delete all rows that contain phases not present in self.indexer.ml_indexer.all_phases.
        """
        if not hasattr(self.indexer, 'ml_indexer'):
            raise AttributeError("DatasetIndexer has no ml_indexer attached.")
        if not hasattr(self.indexer.ml_indexer, 'all_phases'):
            raise AttributeError("ml_indexer has no all_phases attribute.")

        total_rows = self.table.shape[0]
        if total_rows == 0:
            print("No rows available; skipping ml_indexer phase filtering.")
            self.indexer.table_update(self.table)
            return

        allowed_phases = set(self.indexer.ml_indexer.all_phases)
        delete_mask = np.zeros(total_rows, dtype=bool)
        phases_removed = []

        for phase, components in self.indexer.MELTS_indices.items():

            if phase in allowed_phases:
                continue

            if phase == 'melts-liquid':
                mass_col = components.get('liq mass (gm)')
            else:
                mass_col = components.get('mass (gm)')

            if mass_col is None:
                print(f"Skipping phase '{phase}': no mass column found.")
                continue

            phase_mask = self.table[:, mass_col] > 0
            phase_count = int(np.sum(phase_mask))
            if phase_count == 0:
                print(f"Phase '{phase}' not in ml_indexer; no rows contain it.")
                continue

            delete_mask |= phase_mask
            phases_removed.append(phase)
            print(
                f"Phase '{phase}' not in ml_indexer; marking {phase_count} rows for deletion."
            )

        if np.any(delete_mask):
            indices_to_delete = np.where(delete_mask)[0]
            print(
                f"Deleting {len(indices_to_delete)} rows from {len(phases_removed)} non-ml phases: "
                f"{phases_removed}"
            )
            self.delete(indices_to_delete)
        else:
            print("No rows matched phases outside ml_indexer; no rows deleted.")

        self.indexer.table_update(self.table)
    
    def filter_inconsistent_phase_data(self):
        """
        Delete rows with inconsistent phase data: mass = 0 but non-zero attributes.
        
        Checks all phases in ml_indexer and removes rows where a phase has zero mass
        but has non-zero values in any of its other attributes (composition, etc.).
        This indicates corrupted or inconsistent data.
        """
        if not hasattr(self.indexer, 'ml_indexer'):
            raise AttributeError("DatasetIndexer has no ml_indexer attached.")
        if not hasattr(self.indexer.ml_indexer, 'all_phases'):
            raise AttributeError("ml_indexer has no all_phases attribute.")

        total_rows = self.table.shape[0]
        if total_rows == 0:
            print("No rows available; skipping inconsistent phase data filtering.")
            return

        delete_mask = np.zeros(total_rows, dtype=bool)
        inconsistent_phases = {}

        for phase in self.indexer.ml_indexer.all_phases:
            if phase not in self.indexer.MELTS_indices:
                continue

            components = self.indexer.MELTS_indices[phase]

            # Get mass column
            if phase == 'melts-liquid':
                mass_col = components.get('liq mass (gm)')
            else:
                mass_col = components.get('mass (gm)')

            if mass_col is None:
                continue

            # Get all other attribute columns for this phase
            other_cols = [col_idx for key, col_idx in components.items() 
                         if col_idx != mass_col]

            if not other_cols:
                continue

            # Find rows where mass = 0
            zero_mass_mask = self.table[:, mass_col] == 0

            # Check if any other attributes are non-zero when mass = 0
            other_data = self.table[zero_mass_mask][:, other_cols]
            has_nonzero_attrs = np.any(other_data != 0, axis=1)

            # Mark these rows for deletion
            zero_mass_indices = np.where(zero_mass_mask)[0]
            inconsistent_indices = zero_mass_indices[has_nonzero_attrs]

            if len(inconsistent_indices) > 0:
                delete_mask[inconsistent_indices] = True
                inconsistent_phases[phase] = len(inconsistent_indices)

        # Count total inconsistent rows
        num_to_delete = int(np.sum(delete_mask))
        
        if num_to_delete > 0:
            proportion = num_to_delete / total_rows
            print(f"\n=== Filtering Inconsistent Phase Data ===")
            print(f"Total rows: {total_rows}")
            print(f"Rows with inconsistent data: {num_to_delete} ({proportion:.2%})")
            print(f"\nInconsistent phases detected:")
            for phase, count in sorted(inconsistent_phases.items(), key=lambda x: -x[1]):
                phase_proportion = count / total_rows
                print(f"  {phase:25s}: {count:6d} rows ({phase_proportion:.4%})")
            
            indices_to_delete = np.where(delete_mask)[0]
            self.delete(indices_to_delete)
            print(f"\nDeleted {num_to_delete} rows with inconsistent phase data.")
        else:
            print("No inconsistent phase data found; no rows deleted.")

        self.indexer.table_update(self.table)
            
            
    
    def filter_legal(self):
        """OLD CODE TO FILTER BY LIQUID COMPOSITION; DEPRECATED."""
        TiO2_col = self.indexer.MELTS_indices['melts-liquid']['wt% TiO2']
        SiO2_col = self.indexer.MELTS_indices['melts-liquid']['wt% SiO2']
        FO2_col = self.indexer.MELTS_indices['System_main']['logfO2-QFM']
        
        TiO2_excess = self.table[:, TiO2_col] > self.table[:, SiO2_col]
        FO2_out = (self.table[:, FO2_col] > 5) | (self.table[:, FO2_col] < -5)
        #amphibole_bearing = self.table[:, self.indexer.MELTS_indices['amphibole']['mass (gm)']] > 0
        #cristobalite_bearing = self.table[:, self.indexer.MELTS_indices['cristobalite']['mass (gm)']] > 0
        SiO2_deficit = (self.table[:, SiO2_col] < 20) * (self.table[:, SiO2_col] != 0)
        
        # Check for nonzero values in excluded phases (skip System_main as it's always present)
        excluded_phase_cols = []
        for phase in self.indexer.EXCLUDED_PHASES:
            if phase not in ['System_main', 'Bulk_comp'] and phase in self.indexer.MELTS_indices:
                # Get all column indices for this excluded phase
                phase_cols = list(self.indexer.MELTS_indices[phase].values())
                excluded_phase_cols.extend(phase_cols)
        
        excluded_phase_mask = np.zeros(self.table.shape[0], dtype=bool)
        if excluded_phase_cols:
            # Check if any excluded phase column has nonzero values
            excluded_phase_data = self.table[:, excluded_phase_cols]
            excluded_phase_mask = np.any(excluded_phase_data != 0, axis=1)
      
        #self.delete(np.where(TiO2_excess | FO2_out | amphibole_bearing | cristobalite_bearing | SiO2_deficit | excluded_phase_mask)[0])
        self.delete(np.where(TiO2_excess | FO2_out | SiO2_deficit | excluded_phase_mask)[0])
        
        print(f"Deleted {np.sum(TiO2_excess)} for TiO2 dominant liquids")
        print(f"Deleted {np.sum(FO2_out)} for out-of-bounds fO2")
        #print(f"Deleted {np.sum(amphibole_bearing)} for having amphibole")
        #print(f"Deleted {np.sum(cristobalite_bearing)} for having cristobalite")
        print(f"Deleted {np.sum(SiO2_deficit)} for low (<20%) SiO2 liquids")
        print(f"Deleted {np.sum(excluded_phase_mask)} for having nonzero values in excluded phases: {self.indexer.EXCLUDED_PHASES}")
          
    def filter_full_metadata(self):
        """Deletes rows where metadata contain unsupported phases."""
        # Try parsing each metadata entry
        mask_keep = np.array([
            entry.strip().isdigit() for entry in self.metadata
        ])

        # Delete where mask is False
        if np.sum(~mask_keep):
            print(f"Deleted {np.sum(~mask_keep)} entries for unsupported phases")
            indices_to_delete = np.where(~mask_keep)[0]
            self.delete(indices_to_delete)

    def filter_all(self, generate_binaries = False, thresholds = None):#, ExFail, name = None):
        """INELEGANT HARD CODE; DEPRECATED. 'Blanket filtering from scratch'"""

        if generate_binaries:
            if thresholds is None:
                thresholds = np.array([6,12,4,4,4,5,100])
            t_start = time.time()
            self.filter_jumps_with_phase_matrix()
            print(f"[TIMER] Filtering Large Jumps and Blurred Binaries Generated in {time.time() - t_start:.2f} seconds")
        else:
            if thresholds is None:
                thresholds = np.array([6,12,4,4,4,5])
            t_start = time.time()
            self.filter_jumps()
            print(f"[TIMER] Filtering Large Jumps Completed in {time.time() - t_start:.2f} seconds")
        
        t_start = time.time()
        self.filter_legal()
        print(f"[TIMER] Filtering For legal TiO2 and fO2 Completed in {time.time() - t_start:.2f} seconds")
      
    def resample_rare_phase(self, phase_column, multiplier_bounds, n_resamples, overwrite = False):
        """
        Resamples entries that contain a rare phase (nonzero in `phase_column`).

        Parameters:
        - phase_column (int): Index of the column indicating presence of the rare phase.
        - multiplier_bounds (tuple/list of two floats): (min, max) range for random multipliers.
        - n_resamples (int): Number of times to replicate/resample each rare-phase entry.

        Appends the resampled entries (with perturbed mass columns) to the table.
        """
        min_multiplier, max_multiplier = multiplier_bounds
        if min_multiplier >= max_multiplier:
            raise ValueError("Invalid multiplier bounds: min must be less than max.")
        if not isinstance(n_resamples, int) or n_resamples < 1:
            raise ValueError("n_resamples must be a positive integer.")

        # Identify entries with the rare phase present
        rare_mask = self.table[:, phase_column] > 0
        rare_rows = self.table[rare_mask]
        
        if rare_rows.shape[0] == 0:
            print("[INFO] No rare-phase entries found. No resampling performed.")
            return

        #Define Sizings
        n_rare = rare_rows.shape[0]
        n_masscols = self.indexer.nphases
        total_new = n_rare * n_resamples
        old_rows, n_cols = self.table.shape
        new_total = old_rows + total_new
        
        #Apply all operations to blurred binary too, done early for assertion to occur before opening new memmap
        if self.blurredbinaries is not None:
            assert np.shape(self.blurredbinaries)[0] == np.shape(self.table)[0], 'Shape mismatch between main table and binarylabel table!'
            rare_binaries = self.blurredbinaries[rare_mask]
            resampled_binaries = np.repeat(rare_binaries, n_resamples, axis=0)
            old_binary_filename = self.blurredbinaries.filename
            new_binary_path = f"{old_binary_filename.split('.')[0]}_resampledblurredbinaries.npy"
            new_binary_table = np.lib.format.open_memmap(
                new_binary_path, mode='w+', dtype=self.blurredbinaries.dtype, shape=(new_total, len(mass_indices))
            )
        
            new_binary_table[:old_rows] = self.blurredbinaries
            new_binary_table[old_rows:] = resampled_binaries
            new_binary_table.flush()
            
            
            
            #reload resampled binaries in read mode
            del new_binary_table, self.blurredbinaries
            gc.collect()
            
            if overwrite:
                os.replace(new_binary_path, old_binary_filename)
                self.blurredbinaries = np.load(old_binary_filename, mmap_mode='r+')
            else:
                self.blurredbinaries = np.load(new_binary_path, mmap_mode='r+')
                

        multipliers = np.random.uniform(
            low=min_multiplier, high=max_multiplier, size=(total_new, n_masscols)
        )
        

        # Repeat original rare rows
        resampled = np.repeat(rare_rows, n_resamples, axis=0)

        # Apply multipliers to specified mass columns
        resampled[:,self.indexer.mass_indices] = resampled[:,self.indexer.mass_indices] * multipliers
        
        totals = np.sum(resampled[:,self.indexer.mass_indices], axis = 1).reshape((-1,1))
        
        resampled[:,self.indexer.mass_indices] = resampled[:,self.indexer.mass_indices] * 100/totals

        # Expand underlying memmap
        new_path = f"{self.table.filename.split('.')[0]}_resampled.npy"
        old_path = self.table.filename
        new_table = np.lib.format.open_memmap(
            new_path, mode='w+', dtype=self.table.dtype, shape=(new_total, n_cols)
        )

        new_table[:old_rows] = self.table
        new_table[old_rows:] = resampled
        new_table.flush()
            
        del self.table, new_table
        gc.collect()
        
        # Update self in-place
        if overwrite:
            os.replace(new_path, old_path)
            self.table = np.load(old_path, mmap_mode='r+')
        else:
            self.table = np.load(new_path, mmap_mode='r+')

        self.meta = np.append(self.meta,[m + 'RESAMPLE' for m in np.repeat(self.meta[rare_mask], n_resamples)])
        self.run_indices = np.append(self.run_indices, np.repeat(self.run_indices[rare_mask], n_resamples))
        self.MELTSversion = np.append(self.MELTSversion, np.repeat(self.MELTSversion[rare_mask], n_resamples))
        self.metadata = np.append(self.metadata, np.repeat(self.metadata[rare_mask], n_resamples))
        
    def separate_analcime(self):
        
        """Used to move leucite data to analcime columns when leucite component < 0.4 (indicating Na/H2O rich analcime)"""
        
        # Find rows where leucite phase has mass > 0 and leucite component < 0.4 (indicating analcime)
        analcime_pres = np.where((self.table[:,self.indexer.MELTS_indices['leucite']['mass (gm)']]>0) & (self.table[:,self.indexer.MELTS_indices['leucite']['leucite']]<0.4))[0]
        print(f"Total Length:{np.shape(self.table)[0]}, Leucites: {np.sum(self.table[:,self.indexer.MELTS_indices['leucite']['mass (gm)']]>0)}, of which {len(analcime_pres)} are analcime")
        
        if len(analcime_pres) == 0:
            print("No analcime assemblages found to separate.")
            return
        
        # Map leucite columns to analcime columns
        oldIDX = []
        newIDX = []
        for key, idx in self.indexer.MELTS_indices['leucite'].items():
            #if key in self.indexer.MELTS_indices['analcime']:
            oldIDX.append(idx)
            newIDX.append(self.indexer.MELTS_indices['analcime'][key])
            
        oldIDX = np.array(oldIDX)
        newIDX = np.array(newIDX)

        # Move data from leucite columns to analcime columns
        self.table[np.ix_(analcime_pres,newIDX)] = self.table[np.ix_(analcime_pres,oldIDX)]
        # Clear the leucite columns for these rows
        self.table[np.ix_(analcime_pres,oldIDX)] = 0
        self.table.flush() # Write to disc
        
        print(f"Moved {len(analcime_pres)} assemblages from leucite to analcime columns")

        
    def retrieve_component_moles(self, multiplier_bounds=[1, 1]):
        """
        Convert phase assemblages from mass (grams) to molar component form.
        
        This method transforms the MELTS table data into absolute molar quantities for each
        component in each phase. It handles three types of phases differently:
        
        1. Solid phases with variable composition (e.g., olivine with Mg-Fe substitution)
        2. Solid phases with fixed composition (e.g., quartz)
        3. Liquid phase (stored as wt% oxides in MELTS table)
        
        Args:
            multiplier_bounds: [min, max] range for random mass multipliers (for resampling)
        
        Requires:
            self.table1 must be initialized (done by resampling_to_datasets, the parent function)
        
        Creates:
            self.molar: memmap array (n_rows, n_components) with absolute component moles, normalized to sum(element moles) = 1
        """
        # ========== Extract indexer data ==========
        label_indices = self.indexer.label_indices  # Phase -> component indices in label space
        label_names = self.indexer.label_names      # List of all component names
        component_indices = self.indexer.MELTS_indices  # Phase -> column indices in MELTS table
        
        # Transformation matrices
        compToOxLoad = self.indexer.ml_indexer.compToOxLoad  # Components -> oxides
        OxToEl = self.indexer.ml_indexer.OxToEl              # Oxides -> elements
        Mtot = self.indexer.ml_indexer.Mtot                  # Oxide molar masses (column vector)
        Minv = self.indexer.ml_indexer.Minv                  # Inverse molar mass matrix (diagonal)
        Oxides = self.indexer.ml_indexer.Oxides              # List of oxide names

        print(compToOxLoad)
        print(self.table1)
        
        # ========== Initialize output memmap ==========
        total_rows = self.table1.shape[0]
        num_cols = self.indexer.ml_indexer.ncomps
        
        self.molar = np.lib.format.open_memmap(
            self.filename + 'component_moles.npy',
            mode='w+',
            dtype=np.float32,
            shape=(total_rows, num_cols)
        )
        
        # ========== Process each phase ==========
        for phase in list(label_indices.keys()):
            # Generate random multipliers for resampling (1.0 for no resampling)
            mass_multipliers = np.random.uniform(
                *multiplier_bounds, 
                size=total_rows
            ).reshape(-1, 1)
            
            # --- CASE 1: Solid phases (olivine, pyroxene, etc.) ---
            if phase != 'melts-liquid':
                
                # Get indices for this phase's components
                phase_label_inds = label_indices[phase]
                
                if len(phase_label_inds) > 1:
                    # Variable composition solid (e.g., olivine: Fo + Fa + Monticellite)
                    # Components are stored as mole fractions that sum to 1
                    
                    # Get component names and their column indices in table
                    component_names = np.array(label_names)[phase_label_inds]
                    component_col_indices = np.array([
                        component_indices[phase][comp_name] 
                        for comp_name in component_names
                    ])
                    
                    # Extract component mole fractions from table (sum to 1 per row)
                    component_fractions = self.table1[:, component_col_indices]
                    
                    # Calculate phase molar mass from composition
                    # component_fractions -> oxides -> molar mass
                    phase_molar_mass = (
                        component_fractions @ compToOxLoad[phase_label_inds]
                    ) @ Mtot
                    
                    # Get phase mass (grams) from table
                    phase_mass_grams = self.table1[:, component_indices[phase]['mass (gm)']]
                    phase_mass_col = np.atleast_2d(phase_mass_grams).T
                    
                    # Calculate total moles of phase: moles = mass / molar_mass
                    # Handle division by zero with safe divide
                    zero_mat = np.zeros_like(phase_molar_mass, dtype=float)
                    total_phase_moles = np.divide(
                        phase_mass_col,
                        phase_molar_mass,
                        out=zero_mat,
                        where=phase_molar_mass != 0
                    )
                    
                    # Moles of each component = component_fraction * total_moles * multiplier
                    self.molar[:, phase_label_inds] = (
                        component_fractions * 
                        mass_multipliers * 
                        total_phase_moles
                    )
                    
                else:
                    # Invariant composition solid (e.g., quartz = 100% SiO2)
                    # Only one component, so moles = mass / molar_mass
                    
                    phase_mass_grams = self.table1[:, component_indices[phase]['mass (gm)']]
                    phase_mass_col = phase_mass_grams.reshape(-1, 1)
                    
                    # Get molar mass of this single component
                    phase_molar_mass = compToOxLoad[phase_label_inds, :] @ Mtot
                    
                    # Calculate moles: mass / MM, with multiplier
                    self.molar[:, phase_label_inds] = (
                        (mass_multipliers * phase_mass_col / phase_molar_mass).T
                    ).T
            
            # --- CASE 2: Liquid phase (special handling) ---
            else:
                # Liquid is stored as wt% oxides in MELTS table
                # Need to convert: wt% oxides -> mole oxides -> mole elements
                
                # Get column indices for oxide weight percents
                oxide_col_indices = np.array([
                    component_indices['melts-liquid'][f"wt% {oxide}"] 
                    for oxide in Oxides
                ])
                
                # Extract wt% oxides from table
                wt_percent_oxides = self.table1[:, oxide_col_indices]
                
                # Convert wt% to unnormalized moles: wt% / molar_mass
                # Minv is diagonal matrix of 1/MM for each oxide
                unnormed_mole_oxides = wt_percent_oxides @ Minv
                
                # Normalize to sum = 1 (mole fractions)
                total_unnormed = np.sum(unnormed_mole_oxides, axis=1).reshape(-1, 1)
                zero_mat = np.zeros_like(unnormed_mole_oxides, dtype=float)
                mole_fraction_oxides = np.divide(
                    unnormed_mole_oxides,
                    total_unnormed,
                    out=zero_mat,
                    where=total_unnormed != 0
                )
                
                # Calculate liquid molar mass from composition
                liquid_molar_mass = mole_fraction_oxides @ Mtot
                
                # Get liquid mass (grams) from table
                liquid_mass_grams = self.table1[:, component_indices[phase]['liq mass (gm)']]
                liquid_mass_col = liquid_mass_grams.reshape(-1, 1)
                
                # Calculate total moles of liquid: mass / molar_mass
                zero_mat = np.zeros_like(liquid_molar_mass, dtype=float)
                total_liquid_moles = np.divide(
                    liquid_mass_col,
                    liquid_molar_mass.reshape(-1, 1),
                    out=zero_mat,
                    where=liquid_molar_mass.reshape(-1, 1) != 0
                )
                
                # Convert from mole oxides to mole elements
                # mole_fraction_oxides @ OxToEl gives element mole fractions
                # Multiply by total moles and multiplier to get absolute moles
                self.molar[:, label_indices[phase]] = (
                    mole_fraction_oxides * 
                    mass_multipliers * 
                    total_liquid_moles
                ) @ OxToEl                                          