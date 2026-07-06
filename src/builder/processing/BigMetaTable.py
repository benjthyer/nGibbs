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
from ngibbs.utils.string_utils import pull_number
from ngibbs.utils.math_utils import blur_binary_boundaries
from ngibbs.utils.file_utils import move_file


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

class BigMetaTable():
    def __init__(self, filename, read_dir=None, memmap_mode='r+', rebuild_memmap=False,
                 allow_differing_lengths=False, header=None, Model='MELTS', OXYGEN='closed'):
        
        self.filename = filename
        self.memmap_file = self.filename + '.npy'
        self.csv_file =  self.filename + '.csv'
        self.txt_file =  self.filename + '.txt'
        if 'hefesto' in filename.lower() and Model.lower() != 'hefesto': # Overwrite if HeFESTo is detected in filename, but only if not already set to HeFESTo 
            self.Model = 'HeFESTo'
            print('OVERRIDING MODEL TO HeFESTo BASED ON FILENAME, AFFECTS PXSP TRANSFORMATIONS!')
        else:
            self.Model = Model

        self.OXYGEN = OXYGEN

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
            
            # Backward compatibility: rename highsanidine(plagioclase) to sanidine(plagioclase)
            if 'highsanidine(plagioclase)' in self.header:
                col_idx = self.header.index('highsanidine(plagioclase)')
                self.header[col_idx] = 'sanidine(plagioclase)'
                print("[Backward Compatibility] Renamed column: 'highsanidine(plagioclase)' -> 'sanidine(plagioclase)'")
            
            with open(self.csv_file, newline='') as f:
                file_rows = sum(1 for _ in f) - 1
                self.file_rows = file_rows
            
            # Build DatasetIndexer from headers
            self.indexer = DatasetIndexer(self.header, MODEL = self.Model, OXYGEN = self.OXYGEN)
        
                
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
            if header is not None:
                self.header = header
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
            
            # Backward compatibility: rename highsanidine(plagioclase) to sanidine(plagioclase)
            if 'highsanidine(plagioclase)' in self.header:
                col_idx = self.header.index('highsanidine(plagioclase)')
                self.header[col_idx] = 'sanidine(plagioclase)'
                print("[Backward Compatibility] Renamed column: 'highsanidine(plagioclase)' -> 'sanidine(plagioclase)'")
            
            self.file_rows = file_rows
           
            # Build DatasetIndexer from loaded headers
            self.indexer = DatasetIndexer(self.header, MODEL = self.Model, OXYGEN = self.OXYGEN)
            
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
        print(f"[TIMER] Parsed {total_rows} of Metadata to Lists: completed in {time.time() - t_start:.2f} seconds")
        
        if working_text:
            self.txt_file = self.filename + '.txt' # Now change name if we are using a 'working' version after reading text.
            
        # --- Convert lists to arrays
        t_start = time.time()
        self.meta = np.array(self.meta, dtype=object)
        self.run_indices = np.array(self.run_indices, dtype=object)
        self.MELTSversion = np.array(self.MELTSversion, dtype=object)
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
        new_table = BigMetaTable(new_filename, Model=self.Model, OXYGEN=self.OXYGEN)
        
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
            
        OXYGEN = self.OXYGEN
        Model = self.Model
        del self
        gc.collect()
        newself = BigMetaTable(remaining_filename, Model=Model, OXYGEN=OXYGEN)
            
        return newself, new_table


    def manual_split(self, sep_idx):
        """
        Splits the BigMetaTable into two, moving the rows given by the sep_idx argument to a new BigMetaTable.

        Parameters:
        - sep_idx (arr, type: int): Array of row indices to move to the new table.

        Returns:
        - remainder, new_table (BigMetaTable): Self without sep_idx, A new BigMetaTable with the rows specified by sep_idx.
        """

        if not isinstance(sep_idx, np.ndarray):
            sep_idx = np.array(sep_idx)
        if not np.all((sep_idx >= 0) & (sep_idx < self.table.shape[0])):
            raise ValueError("Some indices in sep_idx are out of bounds.")


        total_rows = self.table.shape[0]
       

        # Randomly select indices to move (without replacement)
        all_indices = np.arange(total_rows)
        keep_mask = np.ones(total_rows, dtype=bool)
        keep_mask[sep_idx] = False
        keep_indices = all_indices[keep_mask]
        num_to_move = len(sep_idx)
        move_indices = sep_idx

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
        new_table_data[:] = self.table[sep_idx]
        new_table_data.flush()

        #...and moved text
        with open(new_filename+'.txt', 'w') as f:
            for line in self.meta[sep_idx]:
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
        new_table = BigMetaTable(new_filename, Model=self.Model, OXYGEN=self.OXYGEN)
        
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
            
        OXYGEN = self.OXYGEN
        Model = self.Model
        del self # Deletion for memory management. 
        gc.collect()
        newself = BigMetaTable(remaining_filename, Model=Model, OXYGEN=OXYGEN)
            
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

            if mass_col is None: # HeFESTo sims different
                mass_col = components.get('total (moles)')
                print(f"Phase '{phase}' missing 'mass (gm)', trying 'total (moles)' column: {mass_col}")

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

    def exclude_oxides(self, oxides, tolerance=1e-10):
        """
        Explicitly exclude oxides from the indexer and refresh mappings.

        Intended to be called shortly after BigMetaTable instantiation.
        """
        removed = self.indexer.exclude_oxides(oxides)
        self.indexer.table_update(self.table, tolerance=tolerance)
        return removed
    
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
            return

        non_phase_keys = set(getattr(self.indexer, 'EXCLUDED_PHASES', set())) | {
            'System_main', 'Bulk_comp', 'Bulk_comp_elements'
        }
        allowed_phases = set(self.indexer.ml_indexer.all_phases) | non_phase_keys
        delete_mask = np.zeros(total_rows, dtype=bool)
        phases_removed = []

        for phase, components in self.indexer.MELTS_indices.items():

            if phase in non_phase_keys:
                continue

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

        non_phase_keys = set(getattr(self.indexer, 'EXCLUDED_PHASES', set())) | {
            'System_main', 'Bulk_comp', 'Bulk_comp_elements'
        }
        delete_mask = np.zeros(total_rows, dtype=bool)
        inconsistent_phases = {}

        for phase in self.indexer.ml_indexer.all_phases:
            if phase in non_phase_keys:
                continue

            if phase not in self.indexer.MELTS_indices:
                continue

            components = self.indexer.MELTS_indices[phase]

            # Get mass column
            if phase == 'melts-liquid':
                mass_col = components.get('liq mass (gm)')
            else:
                mass_col = components.get('mass (gm)')

            if mass_col is None:
                mass_col = components.get('mass(gm)')

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
            assert delete_mask.sum()/self.table.shape[0] < 0.03, f'Too many ({100*delete_mask.sum()/self.table.shape[0]} %) broken assemblages... What is going on? '

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
                new_binary_path, mode='w+', dtype=self.blurredbinaries.dtype, shape=(new_total, self.blurredbinaries.shape[1])
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

        if 'leucite' not in self.indexer.MELTS_indices:
            raise KeyError("Cannot separate analcime: 'leucite' phase not found in indexer.")
        if 'analcime' not in self.indexer.MELTS_indices:
            raise KeyError("Cannot separate analcime: 'analcime' phase not found in indexer.")

        leucite_cols = self.indexer.MELTS_indices['leucite']
        analcime_cols = self.indexer.MELTS_indices['analcime']

        if 'mass (gm)' not in leucite_cols or 'leucite' not in leucite_cols:
            raise KeyError(
                "Cannot separate analcime: 'leucite' phase is missing "
                "'mass (gm)' or 'leucite' component columns."
            )

        analcime_mass_key = None
        for mass_key in ('mass (gm)', 'mass(gm)', 'total (moles)'):
            if mass_key in analcime_cols:
                analcime_mass_key = mass_key
                break
        if analcime_mass_key is None:
            raise KeyError(
                "Cannot separate analcime: destination phase 'analcime' has no "
                "mass column among ['mass (gm)', 'mass(gm)', 'total (moles)']."
            )
        
        # Find rows where leucite phase has mass > 0 and leucite component < 0.4 (indicating analcime)
        analcime_pres = np.where(
            (self.table[:, leucite_cols['mass (gm)']] > 0)
            & (self.table[:, leucite_cols['leucite']] < 0.4)
        )[0]
        print(
            f"Total Length:{np.shape(self.table)[0]}, Leucites: "
            f"{np.sum(self.table[:, leucite_cols['mass (gm)']] > 0)}, "
            f"of which {len(analcime_pres)} are analcime"
        )
        
        if len(analcime_pres) == 0:
            print("No analcime assemblages found to separate.")
            return

        # Failsafe: do not overwrite rows where analcime is already present.
        analcime_already_present = self.table[:, analcime_cols[analcime_mass_key]] > 0
        conflict_rows = analcime_pres[analcime_already_present[analcime_pres]]
        move_rows = analcime_pres[~analcime_already_present[analcime_pres]]
        
        # Map leucite columns to analcime columns
        oldIDX = []
        newIDX = []
        for key, idx in leucite_cols.items():
            if key not in analcime_cols:
                continue
            oldIDX.append(idx)
            newIDX.append(analcime_cols[key])

        if len(oldIDX) == 0:
            raise ValueError(
                "Cannot separate analcime: no shared columns between leucite "
                "and analcime phases."
            )
            
        oldIDX = np.array(oldIDX)
        newIDX = np.array(newIDX)

        # Move data from leucite columns to analcime columns
        if len(move_rows) > 0:
            self.table[np.ix_(move_rows, newIDX)] = self.table[np.ix_(move_rows, oldIDX)]
            # Clear the leucite columns for moved rows
            self.table[np.ix_(move_rows, oldIDX)] = 0
            self.table.flush() # Write to disk
        
        print(f"Moved {len(move_rows)} assemblages from leucite to analcime columns")

        if len(conflict_rows) > 0:
            print(
                f"Failsafe triggered: {len(conflict_rows)} rows had both leucite "
                "(analcime-like) and analcime present. Deleting those rows."
            )
            self.delete(np.unique(conflict_rows).astype(int))

    def separate_k_feldspar(self):
        """
        Move plagioclase data to k-feldspar columns when orthoclase fraction > 0.5.

        Failsafe: rows where k-feldspar is already present are not overwritten; they
        are collected and deleted at the end via self.delete().
        """

        source_phase = 'plagioclase'
        if source_phase not in self.indexer.MELTS_indices:
            raise KeyError("Cannot separate k-feldspar: 'plagioclase' phase not found in indexer.")

        destination_phase = None
        for phase_name in ('k-feldspar', 'alkali-feldspar'):
            if phase_name in self.indexer.MELTS_indices:
                destination_phase = phase_name
                break
        if destination_phase is None:
            raise KeyError(
                "Cannot separate k-feldspar: destination phase not found. "
                "Expected one of ['k-feldspar', 'alkali-feldspar']."
            )

        source_cols = self.indexer.MELTS_indices[source_phase]
        destination_cols = self.indexer.MELTS_indices[destination_phase]

        source_mass_key = None
        for mass_key in ('mass (gm)', 'mass(gm)', 'total (moles)'):
            if mass_key in source_cols:
                source_mass_key = mass_key
                break
        if source_mass_key is None:
            raise KeyError(
                "Cannot separate k-feldspar: source plagioclase has no mass column "
                "among ['mass (gm)', 'mass(gm)', 'total (moles)']."
            )

        destination_mass_key = None
        for mass_key in ('mass (gm)', 'mass(gm)', 'total (moles)'):
            if mass_key in destination_cols:
                destination_mass_key = mass_key
                break
        if destination_mass_key is None:
            raise KeyError(
                "Cannot separate k-feldspar: destination has no mass column among "
                "['mass (gm)', 'mass(gm)', 'total (moles)']."
            )

        if 'sanidine' not in source_cols:
            raise KeyError(
                "Cannot separate k-feldspar: 'sanidine' component is missing "
                "from plagioclase columns."
            )

        kfeldspar_candidates = np.where(
            (self.table[:, source_cols[source_mass_key]] > 0)
            & (self.table[:, source_cols['sanidine']] > 0.5)
        )[0]

        print(
            f"Total Length:{np.shape(self.table)[0]}, Plagioclase-bearing: "
            f"{np.sum(self.table[:, source_cols[source_mass_key]] > 0)}, "
            f"of which {len(kfeldspar_candidates)} are k-feldspar-like"
        )

        if len(kfeldspar_candidates) == 0:
            print("No k-feldspar assemblages found to separate.")
            return

        # Failsafe: do not overwrite rows where destination phase is already present.
        destination_already_present = self.table[:, destination_cols[destination_mass_key]] > 0
        conflict_rows = kfeldspar_candidates[destination_already_present[kfeldspar_candidates]]
        move_rows = kfeldspar_candidates[~destination_already_present[kfeldspar_candidates]]

        # Map plagioclase columns to k-feldspar columns where keys are shared.
        oldIDX = []
        newIDX = []
        for key, idx in source_cols.items():
            if key not in destination_cols:
                continue
            oldIDX.append(idx)
            newIDX.append(destination_cols[key])

        if len(oldIDX) == 0:
            raise ValueError(
                f"Cannot separate k-feldspar: no shared columns between "
                f"{source_phase} and {destination_phase}."
            )

        oldIDX = np.array(oldIDX)
        newIDX = np.array(newIDX)

        if len(move_rows) > 0:
            self.table[np.ix_(move_rows, newIDX)] = self.table[np.ix_(move_rows, oldIDX)]
            # Clear source phase columns for moved rows.
            self.table[np.ix_(move_rows, oldIDX)] = 0
            self.table.flush() # Write to disk

        print(
            f"Moved {len(move_rows)} assemblages from {source_phase} "
            f"to {destination_phase} columns"
        )

        if len(conflict_rows) > 0:
            print(
                f"Failsafe triggered: {len(conflict_rows)} rows had both "
                f"{source_phase} (k-feldspar-like) and {destination_phase} present. "
                "Deleting those rows."
            )
            self.delete(np.unique(conflict_rows).astype(int))

        
    def retrieve_component_moles(self, multiplier_bounds=[1, 1]):
        """
        Convert phase assemblages to molar component form.
        
        This method transforms the table data into absolute molar quantities for each
        component in each phase. MELTS phases are reconstructed from masses, while
        HeFESTo phases already store component moles directly.

        It handles three types of phases differently:
        
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
            phase_multipliers = np.random.uniform(
                *multiplier_bounds, 
                size=total_rows
            ).reshape(-1, 1)
            
            if phase != 'melts-liquid':
                # --- CASE 0: HeFESTo solids phases (olivine, pyroxene, etc.) ---

                if self.Model == 'HeFESTo':
                    # HeFESTo solids are already stored as extensive component moles.
                    # Keep them directly, but assert the phase-total column matches the
                    # sum of component columns so schema mismatches fail loudly.
                    phase_label_inds = label_indices[phase]
                    component_names = np.array(label_names)[phase_label_inds]
                    component_col_indices = np.array([
                        component_indices[phase][comp_name]
                        for comp_name in component_names
                    ])

                    component_moles = self.table1[:, component_col_indices]
                   

                    self.molar[:, phase_label_inds] = component_moles * phase_multipliers
                
                # --- CASE 1: MELTS solid phases (olivine, pyroxene, etc.) ---
                else:
                        

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
                            phase_multipliers * 
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
                            (phase_multipliers * phase_mass_col / phase_molar_mass).T
                        ).T
                
            # --- CASE 2: Liquid phase in MELTS (special handling) ---
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
                    phase_multipliers * 
                    total_liquid_moles
                ) @ OxToEl

    def recover_untracked_phases(self):
        """
        Recover graphite mass and hornblende composition from the bulk-composition
        residual for rows whose metadata lists only 'amphibole' and/or 'graphite'
        as untracked phases.  (MELTS labels the hornblende group 'amphibole' in its
        metadata output; the table phase is 'hornblende'.)  Rows with any other
        phase annotation are skipped.

        Recovery order (graphite always first):
          1. Graphite  : residual CO2 moles → graphite grams (12.011 g/mol C)
          2. Hornblende: NNLS fit of residual oxide moles to
                         [pargasite, ferropargasite, magnesiohastingsite]

        Hornblende results are written in-place to the existing
        ``mass (gm)(hornblende)``, ``pargasite(hornblende)``, etc. columns.
        A new ``mass (gm)(graphite)`` column is appended (if absent).
        Metadata text file is NOT modified.  Working CSV is re-saved.
        """
        from scipy.optimize import nnls as _nnls
        from ngibbs.config.constants import OXIDE_MOLAR_MASSES

        # Metadata uses 'amphibole' as MELTS' catch-all; table phase is 'hornblende'
        ALLOWED_META_PHASES = {'amphibole', 'graphite'}
        HORN_ENDMEMBERS     = ['pargasite', 'ferropargasite', 'magnesiohastingsite']

        # ------------------------------------------------------------------ #
        # 1. Classify rows from metadata                                       #
        # ------------------------------------------------------------------ #
        n_rows         = self.table.shape[0]
        has_graphite   = np.zeros(n_rows, dtype=bool)
        has_hornblende = np.zeros(n_rows, dtype=bool)
        do_recovery    = np.zeros(n_rows, dtype=bool)

        for i, meta in enumerate(self.metadata):
            tokens      = str(meta).strip().split()
            phase_names = set()
            for tok in tokens:
                name = tok.split(':')[0].strip().lower()
                if not name.lstrip('-').replace('.', '', 1).isdigit():
                    phase_names.add(name)
            if not phase_names:
                continue
            if phase_names - ALLOWED_META_PHASES:
                continue                           # unrecognised phase → skip
            has_graphite[i]   = 'graphite'  in phase_names
            has_hornblende[i] = 'amphibole' in phase_names   # metadata token
            do_recovery[i]    = True

        target_idx = np.where(do_recovery)[0]
        if len(target_idx) == 0:
            print("[RECOVER] No rows with recoverable untracked phases found.")
            return

        # ------------------------------------------------------------------ #
        # 1b. Guard: skip rows that already have non-zero recovered data.     #
        #     If the graphite column does not yet exist, always attempt.      #
        # ------------------------------------------------------------------ #
        horn_cols_guard     = self.indexer.MELTS_indices.get('hornblende', {})
        horn_mass_col_guard = horn_cols_guard.get('mass (gm)')
        if horn_mass_col_guard is not None:
            existing_horn = self.table[target_idx, horn_mass_col_guard].astype(np.float64)
            already_h     = existing_horn != 0
            n_skip_h      = int((has_hornblende[target_idx] & already_h).sum())
            if n_skip_h:
                print(f"[RECOVER] Skipping {n_skip_h} hornblende rows already non-zero.")
            has_hornblende[target_idx[already_h]] = False

        g_hdr_guard = 'mass (gm)(graphite)'
        if g_hdr_guard in self.header:
            g_col_guard = self.header.index(g_hdr_guard)
            existing_g  = self.table[target_idx, g_col_guard].astype(np.float64)
            already_g   = existing_g != 0
            n_skip_g    = int((has_graphite[target_idx] & already_g).sum())
            if n_skip_g:
                print(f"[RECOVER] Skipping {n_skip_g} graphite rows already non-zero.")
            has_graphite[target_idx[already_g]] = False
        # If graphite column doesn't exist yet, all g-flagged rows proceed.

        do_recovery = has_graphite | has_hornblende
        target_idx  = np.where(do_recovery)[0]
        if len(target_idx) == 0:
            print("[RECOVER] All target rows already recovered; nothing to do.")
            return

        n_target = len(target_idx)
        g_mask   = has_graphite[target_idx]
        h_mask   = has_hornblende[target_idx]
        print(f"[RECOVER] {n_target} target rows to recover — "
              f"{g_mask.sum()} graphite, {h_mask.sum()} hornblende")

        # ------------------------------------------------------------------ #
        # 2. Oxide helpers                                                     #
        # ------------------------------------------------------------------ #
        ml     = self.indexer.ml_indexer
        Oxides = self.indexer.Oxides
        n_ox   = len(Oxides)
        ox_idx = {ox: i for i, ox in enumerate(Oxides)}
        MM_ox  = np.array([OXIDE_MOLAR_MASSES[ox] for ox in Oxides], dtype=np.float64)

        compToOx_df = self.indexer.compToOx_df
        horn_stoich = np.zeros((3, n_ox), dtype=np.float64)
        for j, em in enumerate(HORN_ENDMEMBERS):
            key = f"{em} : hornblende"
            if key not in compToOx_df.index:
                raise KeyError(f"[RECOVER] '{key}' not found in compToOx projection CSV.")
            for k, ox in enumerate(Oxides):
                if ox in compToOx_df.columns:
                    horn_stoich[j, k] = float(compToOx_df.loc[key, ox])

        horn_MM = horn_stoich @ MM_ox   # molar mass per hornblende endmember (3,)

        co2_col = ox_idx.get('CO2', -1)
        if co2_col < 0:
            print("[RECOVER] CO2 not in active oxide list — graphite recovery disabled.")
            has_graphite[:] = False
            g_mask = has_graphite[target_idx]

        # ------------------------------------------------------------------ #
        # 3. Actual bulk oxide moles                                           #
        # ------------------------------------------------------------------ #
        bulk_cols = self.indexer.MELTS_indices.get('Bulk_comp', {})
        if 'mass' in bulk_cols:
            sys_mass = self.table[target_idx, bulk_cols['mass']].astype(np.float64)
        else:
            sys_mass = np.full(n_target, 100.0, dtype=np.float64)
            print("[RECOVER] No 'mass(Bulk_comp)' column — assuming 100 g system mass.")

        actual_ox_moles = np.zeros((n_target, n_ox), dtype=np.float64)
        for k, ox in enumerate(Oxides):
            if ox in bulk_cols:
                wt_pct = self.table[target_idx, bulk_cols[ox]].astype(np.float64)
                actual_ox_moles[:, k] = (wt_pct / 100.0) * sys_mass / MM_ox[k]

        # ------------------------------------------------------------------ #
        # 4. Tracked-phase oxide moles (hornblende & graphite excluded)       #
        # ------------------------------------------------------------------ #
        SKIP = self.indexer.EXCLUDED_PHASES | {'hornblende', 'amphibole', 'graphite'}
        tracked_ox_moles = np.zeros((n_target, n_ox), dtype=np.float64)

        for phase in ml.all_phases:
            if phase in SKIP:
                continue
            phase_cols = self.indexer.MELTS_indices.get(phase, {})

            # ---- Liquid ---------------------------------------------------
            if phase == 'melts-liquid':
                liq_col = phase_cols.get('liq mass (gm)')
                if liq_col is None:
                    continue
                liq_mass = self.table[target_idx, liq_col].astype(np.float64)
                for k, ox in enumerate(Oxides):
                    wt_col = phase_cols.get(f'wt% {ox}')
                    if wt_col is not None:
                        ox_g = (self.table[target_idx, wt_col].astype(np.float64)
                                / 100.0) * liq_mass
                        tracked_ox_moles[:, k] += ox_g / MM_ox[k]
                continue

            # ---- Solid phases ---------------------------------------------
            mass_col = phase_cols.get('mass (gm)', phase_cols.get('total (moles)'))
            if mass_col is None:
                continue
            phase_mass = self.table[target_idx, mass_col].astype(np.float64)
            if np.all(phase_mass == 0):
                continue

            phase_label_inds = ml.label_indices.get(phase)
            if phase_label_inds is None:
                continue

            comp_names = [ml.label_names[i] for i in phase_label_inds]
            if len(phase_label_inds) == 1:
                comp_fracs = np.ones((n_target, 1), dtype=np.float64)
            else:
                comp_fracs = np.zeros((n_target, len(phase_label_inds)), dtype=np.float64)
                for j, cname in enumerate(comp_names):
                    ccol = phase_cols.get(cname)
                    if ccol is not None:
                        comp_fracs[:, j] = self.table[target_idx, ccol].astype(np.float64)

            sub_compToOx   = ml.compToOxLoad[phase_label_inds, :].astype(np.float64)
            phase_ox_fracs = comp_fracs @ sub_compToOx
            phase_MM_row   = phase_ox_fracs @ MM_ox
            phase_moles    = np.where(phase_MM_row > 0, phase_mass / phase_MM_row, 0.0)
            tracked_ox_moles += phase_moles[:, np.newaxis] * phase_ox_fracs

        # ------------------------------------------------------------------ #
        # 5. Residual                                                          #
        # ------------------------------------------------------------------ #
        residual = actual_ox_moles - tracked_ox_moles

        # ------------------------------------------------------------------ #
        # 6. Graphite recovery (CO2 residual, 1 : 1 molar)                    #
        # ------------------------------------------------------------------ #
        graphite_mass_arr = np.zeros(n_target, dtype=np.float64)
        if co2_col >= 0 and g_mask.any():
            co2_res = residual[g_mask, co2_col]
            g_moles = np.maximum(0.0, co2_res)
            graphite_mass_arr[g_mask] = g_moles * 12.011
            residual[g_mask, co2_col] -= g_moles

        # ------------------------------------------------------------------ #
        # 7. Hornblende NNLS per row                                           #
        # ------------------------------------------------------------------ #
        horn_comp_fracs_arr = np.zeros((n_target, 3), dtype=np.float64)
        horn_mass_arr       = np.zeros(n_target,      dtype=np.float64)
        A_horn = horn_stoich.T          # (n_ox, 3)

        for i in tqdm(np.where(h_mask)[0], desc="[RECOVER] Hornblende NNLS", leave=False):
            x, _ = _nnls(A_horn, residual[i])
            horn_mass_arr[i] = float(x @ horn_MM)
            tot = x.sum()
            if tot > 0:
                horn_comp_fracs_arr[i] = x / tot

        # ------------------------------------------------------------------ #
        # 8. Write hornblende into existing 'hornblende' table columns         #
        # ------------------------------------------------------------------ #
        horn_cols = self.indexer.MELTS_indices.get('hornblende', {})
        if not horn_cols:
            print("[RECOVER] Warning: 'hornblende' not found in MELTS_indices — "
                  "hornblende data will not be written.")
        else:
            rows_h        = target_idx[h_mask]
            horn_mass_col = horn_cols.get('mass (gm)')
            if horn_mass_col is not None:
                self.table[rows_h, horn_mass_col] = horn_mass_arr[h_mask].astype(np.float32)
            for j, em in enumerate(HORN_ENDMEMBERS):
                col = horn_cols.get(em)
                if col is not None:
                    self.table[rows_h, col] = horn_comp_fracs_arr[h_mask, j].astype(np.float32)
            self.table.flush()
            print(f"[RECOVER] Wrote hornblende → {h_mask.sum()} rows "
                  f"(total mass = {horn_mass_arr[h_mask].sum():.3f} g).")

        # ------------------------------------------------------------------ #
        # 9. Graphite: fill existing column or append new one                  #
        # ------------------------------------------------------------------ #
        g_hdr = 'mass (gm)(graphite)'
        if g_mask.any():
            rows_g = target_idx[g_mask]
            if g_hdr in self.header:
                g_col = self.header.index(g_hdr)
                self.table[rows_g, g_col] = graphite_mass_arr[g_mask].astype(np.float32)
                self.table.flush()
            else:
                self.table.flush()
                new_col          = np.zeros(n_rows, dtype=np.float32)
                new_col[rows_g]  = graphite_mass_arr[g_mask].astype(np.float32)
                old_ncols        = self.table.shape[1]
                tmp_path         = self.memmap_file.replace('.npy', '_graphite_expand.npy')
                new_mm = np.lib.format.open_memmap(
                    tmp_path, mode='w+', dtype=self.table.dtype,
                    shape=(n_rows, old_ncols + 1))
                new_mm[:, :old_ncols] = self.table
                new_mm[:, old_ncols]  = new_col
                new_mm.flush()
                del new_mm, self.table
                gc.collect()
                os.replace(tmp_path, self.memmap_file)
                self.table = np.load(self.memmap_file, mmap_mode='r+')
                self.header.append(g_hdr)
                self.indexer = DatasetIndexer(self.header, MODEL = self.Model, OXYGEN = self.OXYGEN) # Reinistall Indexer
                print(f"[RECOVER] Appended '{g_hdr}' column.")
            print(f"[RECOVER] Wrote graphite → {g_mask.sum()} rows "
                  f"(total mass = {graphite_mass_arr[g_mask].sum():.3f} g).")

        # ------------------------------------------------------------------ #
        # 10. Re-save working CSV                                              #
        # ------------------------------------------------------------------ #
        print("[RECOVER] Saving updated CSV …")
        self.save_csv_streaming(name=self.filename)
        print("[RECOVER] Done.")

    def recover_constant_composition_phases(self, phase_names):
        """
        Recover mass for one or more compositionally-constant phases (a single
        fixed oxide formula, e.g. graphite, quartz, calcite) from the bulk-
        composition residual, for rows whose metadata lists only phases named
        in `phase_names` as untracked. Generalizes the graphite branch of
        recover_untracked_phases to an arbitrary set of fixed-stoichiometry
        phases, using the stoichiometry rows already present in the compToOx
        projection CSV (self.indexer.compToOx_df). Variable-composition
        phases (e.g. hornblende) are out of scope - use
        recover_untracked_phases for those.

        Parameters:
        - phase_names (iterable of str): Table phase names to recover. Each
          must have a stoichiometry row in self.indexer.compToOx_df keyed by
          its bare name (no ' : phase' suffix - that suffix only applies to
          variable-composition endmembers), and must match the name used for
          this phase in the per-row text metadata.

        Writes ``mass (gm)(<phase>)`` columns, appending them if absent. Rows
        already non-zero in an existing column are left untouched. Metadata
        text file is NOT modified. Does not write a CSV - the .npy memmap is
        updated (and flushed) in place; call .save() afterwards if a CSV
        companion is wanted.
        """
        from scipy.optimize import nnls as _nnls
        from ngibbs.config.constants import OXIDE_MOLAR_MASSES

        phase_names = list(dict.fromkeys(phase_names))  # de-dupe, preserve order
        if not phase_names:
            raise ValueError("phase_names must be non-empty.")
        ALLOWED_META_PHASES = set(phase_names)
        phase_pos = {p: i for i, p in enumerate(phase_names)}

        compToOx_df = self.indexer.compToOx_df
        Oxides = self.indexer.Oxides
        n_ox = len(Oxides)
        MM_ox = np.array([OXIDE_MOLAR_MASSES[ox] for ox in Oxides], dtype=np.float64)

        # ------------------------------------------------------------------ #
        # Stoichiometry (over active Oxides) and molar mass per phase.         #
        # ------------------------------------------------------------------ #
        stoich = np.zeros((len(phase_names), n_ox), dtype=np.float64)
        for p_i, phase in enumerate(phase_names):
            if phase not in compToOx_df.index:
                raise KeyError(f"[RECOVER] '{phase}' not found in compToOx projection CSV.")
            row = compToOx_df.loc[phase]
            for k, ox in enumerate(Oxides):
                if ox in compToOx_df.columns:
                    stoich[p_i, k] = float(row[ox])
        phase_MM = stoich @ MM_ox
        if np.any(phase_MM <= 0):
            bad = [phase_names[i] for i in np.where(phase_MM <= 0)[0]]
            raise ValueError(
                f"[RECOVER] Non-positive molar mass computed for phase(s) {bad} "
                "from the active Oxides - check compToOx entries / excluded oxides."
            )

        # ------------------------------------------------------------------ #
        # 1. Classify rows from metadata.                                      #
        # ------------------------------------------------------------------ #
        n_rows = self.table.shape[0]
        has_phase = np.zeros((len(phase_names), n_rows), dtype=bool)
        do_recovery = np.zeros(n_rows, dtype=bool)

        for i, meta in enumerate(self.metadata):
            tokens = str(meta).strip().split()
            names = set()
            for tok in tokens:
                name = tok.split(':')[0].strip().lower()
                if not name.lstrip('-').replace('.', '', 1).isdigit():
                    names.add(name)
            if not names:
                continue
            if names - ALLOWED_META_PHASES:
                continue                           # unrecognised phase → skip
            for name in names:
                has_phase[phase_pos[name], i] = True
            do_recovery[i] = True

        target_idx = np.where(do_recovery)[0]
        if len(target_idx) == 0:
            print("[RECOVER] No rows with recoverable constant-composition phases found.")
            return

        # ------------------------------------------------------------------ #
        # 1b. Guard: skip rows already non-zero in an existing mass column.    #
        # ------------------------------------------------------------------ #
        for p_i, phase in enumerate(phase_names):
            hdr = f"mass (gm)({phase})"
            if hdr in self.header:
                col = self.header.index(hdr)
                existing = self.table[target_idx, col].astype(np.float64)
                already = existing != 0
                n_skip = int((has_phase[p_i, target_idx] & already).sum())
                if n_skip:
                    print(f"[RECOVER] Skipping {n_skip} '{phase}' rows already non-zero.")
                has_phase[p_i, target_idx[already]] = False

        do_recovery = np.any(has_phase, axis=0)
        target_idx = np.where(do_recovery)[0]
        if len(target_idx) == 0:
            print("[RECOVER] All target rows already recovered; nothing to do.")
            return

        n_target = len(target_idx)
        print(f"[RECOVER] {n_target} target rows to recover — " + ", ".join(
            f"{phase} ({int(has_phase[p_i, target_idx].sum())})"
            for p_i, phase in enumerate(phase_names)
        ))

        # ------------------------------------------------------------------ #
        # 2. Actual bulk oxide moles for target rows.                          #
        # ------------------------------------------------------------------ #
        bulk_cols = self.indexer.MELTS_indices.get('Bulk_comp', {})
        if 'mass' in bulk_cols:
            sys_mass = self.table[target_idx, bulk_cols['mass']].astype(np.float64)
        else:
            sys_mass = np.full(n_target, 100.0, dtype=np.float64)
            print("[RECOVER] No 'mass(Bulk_comp)' column — assuming 100 g system mass.")

        actual_ox_moles = np.zeros((n_target, n_ox), dtype=np.float64)
        for k, ox in enumerate(Oxides):
            if ox in bulk_cols:
                wt_pct = self.table[target_idx, bulk_cols[ox]].astype(np.float64)
                actual_ox_moles[:, k] = (wt_pct / 100.0) * sys_mass / MM_ox[k]

        # ------------------------------------------------------------------ #
        # 3. Tracked-phase oxide moles (recovered phases excluded).            #
        # ------------------------------------------------------------------ #
        SKIP = self.indexer.EXCLUDED_PHASES | ALLOWED_META_PHASES
        ml = self.indexer.ml_indexer
        tracked_ox_moles = np.zeros((n_target, n_ox), dtype=np.float64)

        for phase in ml.all_phases:
            if phase in SKIP:
                continue
            phase_cols = self.indexer.MELTS_indices.get(phase, {})

            if phase == 'melts-liquid':
                liq_col = phase_cols.get('liq mass (gm)')
                if liq_col is None:
                    continue
                liq_mass = self.table[target_idx, liq_col].astype(np.float64)
                for k, ox in enumerate(Oxides):
                    wt_col = phase_cols.get(f'wt% {ox}')
                    if wt_col is not None:
                        ox_g = (self.table[target_idx, wt_col].astype(np.float64)
                                / 100.0) * liq_mass
                        tracked_ox_moles[:, k] += ox_g / MM_ox[k]
                continue

            mass_col = phase_cols.get('mass (gm)', phase_cols.get('total (moles)'))
            if mass_col is None:
                continue
            phase_mass = self.table[target_idx, mass_col].astype(np.float64)
            if np.all(phase_mass == 0):
                continue

            phase_label_inds = ml.label_indices.get(phase)
            if phase_label_inds is None:
                continue

            comp_names = [ml.label_names[i] for i in phase_label_inds]
            if len(phase_label_inds) == 1:
                comp_fracs = np.ones((n_target, 1), dtype=np.float64)
            else:
                comp_fracs = np.zeros((n_target, len(phase_label_inds)), dtype=np.float64)
                for j, cname in enumerate(comp_names):
                    ccol = phase_cols.get(cname)
                    if ccol is not None:
                        comp_fracs[:, j] = self.table[target_idx, ccol].astype(np.float64)

            sub_compToOx   = ml.compToOxLoad[phase_label_inds, :].astype(np.float64)
            phase_ox_fracs = comp_fracs @ sub_compToOx
            phase_MM_row   = phase_ox_fracs @ MM_ox
            phase_moles    = np.where(phase_MM_row > 0, phase_mass / phase_MM_row, 0.0)
            tracked_ox_moles += phase_moles[:, np.newaxis] * phase_ox_fracs

        # ------------------------------------------------------------------ #
        # 4. Residual, then solve per row for the phase(s) actually present.  #
        #    Rows with a single present phase are solved with a vectorized     #
        #    least-squares projection; rows with several simultaneous          #
        #    constant-composition phases fall back to per-row NNLS (mirrors    #
        #    the hornblende NNLS fit in recover_untracked_phases).             #
        # ------------------------------------------------------------------ #
        residual = actual_ox_moles - tracked_ox_moles
        phase_moles_arr = np.zeros((len(phase_names), n_target), dtype=np.float64)

        present_counts = has_phase[:, target_idx].sum(axis=0)
        single_mask = present_counts == 1
        multi_mask  = present_counts > 1

        if single_mask.any():
            single_rows  = np.where(single_mask)[0]
            phase_of_row = np.argmax(has_phase[:, target_idx[single_rows]], axis=0)
            for p_i in np.unique(phase_of_row):
                rows_p = single_rows[phase_of_row == p_i]
                s = stoich[p_i]
                denom = s @ s
                if denom <= 0:
                    continue
                proj = (residual[rows_p] @ s) / denom
                phase_moles_arr[p_i, rows_p] = np.maximum(0.0, proj)

        if multi_mask.any():
            for row in tqdm(np.where(multi_mask)[0],
                             desc="[RECOVER] Constant-composition NNLS", leave=False):
                present = np.where(has_phase[:, target_idx[row]])[0]
                A = stoich[present].T
                x, _ = _nnls(A, residual[row])
                phase_moles_arr[present, row] = x

        phase_mass_arr = phase_moles_arr * phase_MM[:, np.newaxis]

        # ------------------------------------------------------------------ #
        # 5. Write results: fill existing columns / append new ones as needed.#
        # ------------------------------------------------------------------ #
        to_append = [p for p in phase_names if f"mass (gm)({p})" not in self.header
                     and has_phase[phase_pos[p]].any()]
        if to_append:
            old_ncols = self.table.shape[1]
            tmp_path = self.memmap_file.replace('.npy', '_constphase_expand.npy')
            new_mm = np.lib.format.open_memmap(
                tmp_path, mode='w+', dtype=self.table.dtype,
                shape=(n_rows, old_ncols + len(to_append)))
            new_mm[:, :old_ncols] = self.table
            new_mm[:, old_ncols:] = 0
            new_mm.flush()
            del new_mm, self.table
            gc.collect()
            os.replace(tmp_path, self.memmap_file)
            self.table = np.load(self.memmap_file, mmap_mode='r+')
            self.header.extend(f"mass (gm)({p})" for p in to_append)
            self.indexer = DatasetIndexer(self.header, MODEL=self.Model, OXYGEN=self.OXYGEN)
            print(f"[RECOVER] Appended columns: {[f'mass (gm)({p})' for p in to_append]}")

        for p_i, phase in enumerate(phase_names):
            mask = has_phase[p_i, target_idx]
            if not mask.any():
                continue
            hdr = f"mass (gm)({phase})"
            col = self.header.index(hdr)
            rows = target_idx[mask]
            self.table[rows, col] = phase_mass_arr[p_i, mask].astype(np.float32)
            print(f"[RECOVER] Wrote {phase} → {mask.sum()} rows "
                  f"(total mass = {phase_mass_arr[p_i, mask].sum():.3f} g).")
        self.table.flush()
        print("[RECOVER] Done.")


def merge_big_meta_tables(tables, new_filename, chunk_size=100_000, clear_old_tables = True):
    """
    Merges multiple BigMetaTable instances into a new memmapped BigMetaTable.
    Edited 6/30/25 to initialize merged BigMetaTable using .__init__() for stability
    MUST BE DONE AFTER ANALCIME IS ISOLATED BECAUSE THE NEW TABLE WILL INHERIT THE GLOBAL COLUMNS

    Parameters:
    - tables: list/tuple of BigMetaTable instances to merge.
    - new_filename: base name for output .npy file (no extension needed).
    - chunk_size: number of rows per chunk when copying (default: 100,000).

    Returns:
    - A new BigMetaTable instance with combined memmapped data.
    """
    
    if not tables:
        raise ValueError("No tables were provided for merging.")

    if len(tables) == 2 and tables[0] is None: # SAFEGUARD FOR ITERATIVE USE
        return tables[1]

    tables = [t for t in tables if t is not None]
    if not tables:
        raise ValueError("All provided tables were None.")
    
    # Ensure consistent shape/dtype
    col_count = tables[0].table.shape[1]
    dtype = tables[0].table.dtype
    
    
        
    
    for t in tables:
        if t.table.shape[1] != col_count:
            raise ValueError("Column count mismatch among tables.")
        if t.table.dtype != dtype:
            raise ValueError("Dtype mismatch among tables.")

    # Compute total number of rows
    total_rows = sum(t.table.shape[0] for t in tables)

    # Create memmap file
    mmap_path = f"{new_filename}.npy"
    merged_array = np.lib.format.open_memmap(
        mmap_path, mode='w+', dtype=dtype, shape=(total_rows, col_count)
    )
    
    # Move BlurredBinaries
    if np.all(np.array([t.blurredbinaries is not None for t in tables])):
        move_blurred = True
        print('MERGING BLURRED BOUNDARIES')

        blurred_cols = tables[0].blurredbinaries.shape[1]
        blurred_dtype = tables[0].blurredbinaries.dtype
        for t in tables:
            if t.blurredbinaries.shape[1] != blurred_cols:
                raise ValueError("Blurredbinaries column mismatch among tables.")

        merged_blurredbinaries = np.lib.format.open_memmap(
        new_filename+'blurredbinaries.npy', mode='w+', dtype=blurred_dtype, shape=(total_rows, blurred_cols)
        )
    else:
        move_blurred = False
        print('Not merging blurred boundaries')
        

    # Copy in chunks
    current_row = 0
    new_meta = []

    for t in tables:
        table_rows = t.table.shape[0]
        for start in range(0, table_rows, chunk_size):
            end = min(start + chunk_size, table_rows)
            merged_array[current_row:current_row + (end - start)] = t.table[start:end]
            
            if move_blurred:
                merged_blurredbinaries[current_row:current_row + (end - start)] = t.blurredbinaries[start:end]
            
            current_row += (end - start)
            
        # merge metadata
        new_meta.append(t.meta)
    
    # Save metadata
    new_meta = np.concatenate(new_meta)
    
    assert merged_array.shape[0] == len(new_meta), 'Metadata length does not equal table rows!'
            
    # Flush memmap(s) to disk
    merged_array.flush()
    del merged_array
    if move_blurred:
        merged_blurredbinaries.flush()
        del merged_blurredbinaries
    if clear_old_tables:
        for t in tables:
            del t.table
            if t.blurredbinaries is not None:
                del t.blurredbinaries
            del t
            
    gc.collect()
    
    # Write metadata 
    with open(new_filename+'.txt', 'w') as f:
        for line in new_meta:
            f.write(line + '\n')

    # Build new BigMetaTable
    new_table = BigMetaTable(new_filename, header=tables[0].header)

    return new_table