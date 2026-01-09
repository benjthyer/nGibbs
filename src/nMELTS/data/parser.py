"""
MELTS output file parsing.

Contains BigMetaTable class for handling large MELTS datasets with memory-mapped arrays.
Moved from BigMetaTableLibrary.py for better organization.
"""

import os
import csv
import gc
import time
import mmap
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import from refactored modules
from ..config import (
    MELTS_indices,
    database_headers,
    mass_indices,
    mass_phasedict,
)
from ..utils.string_utils import pull_number
from ..utils.math_utils import blur_binary_boundaries
from ..utils.file_utils import move_file


def collect_indices(csv_path, mass_indices, known_total_rows=None, 
                    batch_size=100_000, maximum_indices=4_000_000, 
                    random_state=None, has_header=True):
    """
    Memory-efficient function
    Gathers indices of subliquidus rows and supplements with random superliquidus rows. 
    This process is also automatically applied when importing melts data, so this is rarely necessary to run manually. 
    """
    rng = np.random.default_rng(random_state)

    selected = []   # indices of rows with nonzero
    total_rows = 0

    usecols = mass_indices[:-1]
    #header_offset = 1 if has_header else 0
    header_offset = 0 # I think this was adding 1 to all indices incorrectly, below pandas reader takes care of the header.
    
    # Stream the CSV in chunks
    reader = pd.read_csv(
        csv_path,
        chunksize=batch_size,
        header=0 if has_header else None
    )

    for chunk in tqdm(reader, desc="Collecting Subliquidus Indices..."):
        # Absolute index offset for this chunk
        idx_offset = total_rows + header_offset
        total_rows += len(chunk)

        # Boolean mask for nonzero rows
        mask = (chunk.iloc[:, usecols] != 0).any(axis=1)
        nonzero_idx = np.flatnonzero(mask) + idx_offset

        if len(nonzero_idx) > 0:
            selected.append(nonzero_idx)

    # Concatenate results into one array
    if selected:
        selected = np.concatenate(selected)
    else:
        selected = np.array([], dtype=np.int64)

    # Compute remaining candidates *without* materializing full setdiff
    all_indices = np.arange(header_offset, total_rows, dtype=np.int64)
    mask = np.ones(len(all_indices), dtype=bool)

    pos = selected - header_offset
    pos = pos[(pos >= 0) & (pos < len(mask))]  # safety check
    mask[pos] = False

    remaining = all_indices[mask]

    # Balance positives and random negatives
    n_to_sample = min(len(selected), len(remaining))
    random_indices = rng.choice(remaining, size=n_to_sample, replace=False)

    final_indices = np.sort(np.concatenate([selected, random_indices]))

    # Reduce if too big
    if len(final_indices) > maximum_indices:
        print(f"{len(final_indices)} collected; reducing to {maximum_indices}")
        final_indices = rng.choice(final_indices, size=int(maximum_indices), replace=False)
        final_indices.sort()

    return final_indices


class BigMetaTable:
    def __init__(self, filename, read_dir=None, memmap_mode='r+', rebuild_memmap=False,
                 allow_differing_lengths=False, reduce_superliquidus = False):
        

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
            if reduce_superliquidus:
                idx = collect_indices(csv_path = self.csv_file, mass_indices = mass_indices, known_total_rows = file_rows)
            else:
                idx = np.arange(file_rows)
            
            self.idx = idx
            self.idx_set = set(idx)
                
            total_rows = len(idx)
            self.total_rows = total_rows
            self.filename = filename + '_working'
            self.memmap_file = self.filename + '.npy'
            working_text = True
    
            self._csv_to_memmap()
            self.table = np.load(self.memmap_file, mmap_mode=memmap_mode, allow_pickle=True)
            
        else:
            self.table = np.load(self.memmap_file, mmap_mode=memmap_mode, allow_pickle=True)
            file_rows = self.table.shape[0]
            self.file_rows = file_rows
            print('No Header! Assuming global database_headers apply')
            self.header = database_headers
            total_rows = file_rows
            self.total_rows = total_rows
            
            idx = np.arange(total_rows)
        
            self.idx = idx
            self.idx_set = set(idx)
            
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
                if i not in self.idx_set:
                    continue
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

        if len(self.meta) != self.table.shape[0] and not allow_differing_lengths:
            rows = self.table.shape[0]
            del self.table
            gc.collect()
            raise ValueError(f'Metadata length {len(self.meta)} does not match data rows: {rows}')

    
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
                if i not in self.idx_set:
                    continue
                try:
                    values = [float(x) if x.strip() != '' else 0.0 for x in row]
                except:
                    print(len(values))
                    print(row)
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
        return np.arange(len(self.metadata))[self.run_indices == code]

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
        new_table = BigMetaTable(new_filename)
        
        # Move BlurredBinaries
        if self.blurredbinaries is not None:
            new_table.blurredbinaries = np.lib.format.open_memmap(
            new_filename+'blurredbinaries.npy', mode='w+', dtype=np.float32, shape=(num_to_move, len(mass_indices))
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
                
            new_binary_memmap = np.lib.format.open_memmap(temp_binary_filename, dtype=np.float32, mode='w+', shape=(total_rows - num_to_move, len(mass_indices)))

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

        relevant_cols = [MELTS_indices['System_main']['Temperature'],
                 MELTS_indices['melts-liquid']['liq mass (gm)'],
                 MELTS_indices['melts-liquid']['wt% TiO2'],
                 MELTS_indices['melts-liquid']['wt% P2O5'],
                 MELTS_indices['melts-liquid']['wt% MnO'],
                 MELTS_indices['melts-liquid']['wt% NiO']]

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
        num_phases = mass_phasedict['melts-liquid']+1
        broke_count = 0
        
        
        relevant_cols = [MELTS_indices['System_main']['Temperature'],
                 MELTS_indices['melts-liquid']['liq mass (gm)'],
                 MELTS_indices['melts-liquid']['wt% TiO2'],
                 MELTS_indices['melts-liquid']['wt% P2O5'],
                 MELTS_indices['melts-liquid']['wt% MnO'],
                 MELTS_indices['melts-liquid']['wt% NiO'],
                 MELTS_indices['System_main']['Pressure']]
        
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
            binary_mask = (self.table[np.ix_(indices,mass_indices)]>0).astype(np.float32)
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
            
            
    
    def filter_legal(self):
        TiO2_col = MELTS_indices['melts-liquid']['wt% TiO2']
        SiO2_col = MELTS_indices['melts-liquid']['wt% SiO2']
        FO2_col = MELTS_indices['System_main']['logfO2-QFM']
        
        TiO2_excess = self.table[:, TiO2_col] > self.table[:, SiO2_col]
        FO2_out = (self.table[:, FO2_col] > 5) | (self.table[:, FO2_col] < -5)
        amphibole_bearing = self.table[:, MELTS_indices['amphibole']['mass (gm)']] > 0
        cristobalite_bearing = self.table[:, MELTS_indices['cristobalite']['mass (gm)']] > 0
        SiO2_deficit = (self.table[:, SiO2_col] < 20) * (self.table[:, SiO2_col] != 0)
      
        self.delete(np.where(TiO2_excess | FO2_out | amphibole_bearing | cristobalite_bearing | SiO2_deficit)[0])
        print(f"Deleted {np.sum(TiO2_excess)} for TiO2 dominant liquids")
        print(f"Deleted {np.sum(FO2_out)} for out-of-bounds fO2")
        print(f"Deleted {np.sum(amphibole_bearing)} for having amphibole")
        print(f"Deleted {np.sum(SiO2_deficit)} for low (<20%) SiO2 liquids")
          
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
        """Blanket filtering from scratch"""

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
        n_masscols = len(mass_indices)
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
        resampled[:,mass_indices] = resampled[:,mass_indices] * multipliers
        
        totals = np.sum(resampled[:,mass_indices], axis = 1).reshape((-1,1))
        
        resampled[:,mass_indices] = resampled[:,mass_indices] * 100/totals

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
        
        """Used to move columns into right place after changing structure of MELTS metatable to separate K and Na/H2O rich
        leucites"""
        
        #Go through and delete rare instances of alloy-liquid
        alloy_pres = np.where(self.table[:,MELTS_indices['analcime']['leucite']] > 0)[0] # Old alloy-liquid mass slot
        if len(alloy_pres):
            self.delete(alloy_pres)
            print(f"Deleting {len(alloy_pres)} assemblages for alloy liquid presence")
        
        analcime_pres = np.where((self.table[:,MELTS_indices['leucite']['mass (gm)']]>0) & (self.table[:,MELTS_indices['leucite']['leucite']]<0.4))[0]
        print(f"Total Length:{np.shape(self.table)[0]}, Leucites: {np.sum(self.table[:,MELTS_indices['leucite']['mass (gm)']]>0)}, of which {len(analcime_pres)} are analcime")
        
        
        oldIDX = []
        newIDX = []
        for key, idx in MELTS_indices['leucite'].items():
            oldIDX.append(idx)
            newIDX.append(MELTS_indices['analcime'][key])
            
        oldIDX = np.array(oldIDX)
        newIDX = np.array(newIDX)

        self.table[np.ix_(analcime_pres,newIDX)] = self.table[np.ix_(analcime_pres,oldIDX)]
        self.table[np.ix_(analcime_pres,oldIDX)] = 0
        self.table.flush() # Write to disc
        
        self.header = database_headers # Need to track the change in the table when saving


