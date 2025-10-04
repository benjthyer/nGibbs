"""BigMetaTable for very large datasets with rowwise text file metadata: memmap"""

import numpy as np 
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cm
import pickle
import os
import pandas as pd
import random
plt.rcParams['figure.figsize'] = [10, 7]
import csv
import gc
import time
from tqdm import tqdm
import torch
import mmap
from EmulatorLibrary import *

def collect_indices(csv_path, mass_indices, known_total_rows=None, 
                    batch_size=100_000, maximum_indices=4_000_000, 
                    random_state=None, has_header=True):
    """
    Memory-efficient version of collect_indices.
    Gathers indices of subliquidus rows and supplements with random superliquidus rows.
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
    def __init__(self, filename, memmap_mode='r+', rebuild_memmap=False,
                 allow_differing_lengths=False, reduce_superliquidus = False):
        
        self.filename = filename
        self.memmap_file = self.filename + '.npy'
        self.csv_file = self.filename + '.csv'
        self.txt_file = self.filename + '.txt'
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
            name = self.txt_file
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


    def filter_twophase(self, ExFail):
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

        relevant_cols = [component_indices['System_main']['Temperature'],
                 component_indices['melts-liquid']['liq mass (gm)'],
                 component_indices['melts-liquid']['wt% TiO2'],
                 component_indices['melts-liquid']['wt% P2O5'],
                 component_indices['melts-liquid']['wt% MnO'],
                 component_indices['melts-liquid']['wt% NiO']]

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
        
        
        relevant_cols = [component_indices['System_main']['Temperature'],
                 component_indices['melts-liquid']['liq mass (gm)'],
                 component_indices['melts-liquid']['wt% TiO2'],
                 component_indices['melts-liquid']['wt% P2O5'],
                 component_indices['melts-liquid']['wt% MnO'],
                 component_indices['melts-liquid']['wt% NiO'],
                 component_indices['System_main']['Pressure']]
        
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
        TiO2_col = component_indices['melts-liquid']['wt% TiO2']
        SiO2_col = component_indices['melts-liquid']['wt% SiO2']
        FO2_col = component_indices['System_main']['logfO2-QFM']
        
        TiO2_excess = self.table[:, TiO2_col] > self.table[:, SiO2_col]
        FO2_out = (self.table[:, FO2_col] > 5) | (self.table[:, FO2_col] < -5)
        amphibole_bearing = self.table[:, component_indices['amphibole']['mass (gm)']] > 0
        cristobalite_bearing = self.table[:, component_indices['cristobalite']['mass (gm)']] > 0
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
        alloy_pres = np.where(self.table[:,component_indices['analcime']['leucite']] > 0)[0] # Old alloy-liquid mass slot
        if len(alloy_pres):
            self.delete(alloy_pres)
            print(f"Deleting {len(alloy_pres)} assemblages for alloy liquid presence")
        
        analcime_pres = np.where((self.table[:,component_indices['leucite']['mass (gm)']]>0) & (self.table[:,component_indices['leucite']['leucite']]<0.4))[0]
        print(f"Total Length:{np.shape(self.table)[0]}, Leucites: {np.sum(self.table[:,component_indices['leucite']['mass (gm)']]>0)}, of which {len(analcime_pres)} are analcime")
        
        
        oldIDX = []
        newIDX = []
        for key, idx in component_indices['leucite'].items():
            oldIDX.append(idx)
            newIDX.append(component_indices['analcime'][key])
            
        oldIDX = np.array(oldIDX)
        newIDX = np.array(newIDX)

        self.table[np.ix_(analcime_pres,newIDX)] = self.table[np.ix_(analcime_pres,oldIDX)]
        self.table[np.ix_(analcime_pres,oldIDX)] = 0
        self.table.flush() # Write to disc
        
        self.header = database_headers # Need to track the change in the table when saving
        
def merge_big_meta_tables(tables, new_filename, chunk_size=100_000, clear_old_tables = False):
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
    
    if table[0] is None: # SAFEGUARD FOR ITERATIVE USE
        return table[1]
    
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
        merged_blurredbinaries = np.lib.format.open_memmap(
        new_filename+'blurredbinaries.npy', mode='w+', dtype=dtype, shape=(total_rows, len(mass_indices))
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
    new_table = BigMetaTable(new_filename)

    return new_table


def count_assemblages(MELTS, filename): #CURRENTLY CHOKES 5/21/25
    important_phase_ind = []
    important_phase_names = list(label_indices.keys())
    totalassemblages = np.shape(MELTS.table)[0]
    with open(filename, 'w') as F:
        for name in important_phase_names:
            ind = list(component_indices[name].values())[0]
            percent_rep = 100*np.sum(MELTS.table[:,ind]>0)/totalassemblages
            important_phase_ind.append(ind)
            F.write(f'{name} present in {round(percent_rep,2)}% of dataset assemblages\n')

    important_phase_ind = np.array(important_phase_ind)

    permutations = identify_binaries(len(important_phase_ind)).astype(bool)
    present_phases_bool = MELTS.table[:,important_phase_ind]>0

    permutation_catchment_bins = np.zeros(np.shape(permutations)[0])
    sim = present_phases_bool[0]

    for sim in present_phases_bool:
        match = np.all(sim == permutations, axis = 1)
        permutation_catchment_bins += match

    with open(filename, 'a') as F:
        for i, permutation in enumerate(tqdm(permutations, desc = f'Counting Unique Assemblages', leave = False)):
            phase_txt = ' + '.join(np.array(important_phase_names, dtype = object)[permutation])
            if not len(phase_txt):
                phase_txt = 'None'
            if permutation_catchment_bins[i] > 0:
                print(phase_txt + ': ' + str(int(permutation_catchment_bins[i])))
                F.write(phase_txt + ': ' + str(int(permutation_catchment_bins[i]))+' \n')
            
def safe_delete_batched(filename, delete_indices, batch_size=200000):
    """
    Filters a .npy file by removing rows listed in delete_indices.
    Creates a temporary file and replaces the original.
    """
    delete_indices_set = set(delete_indices)
    original = np.load(filename, mmap_mode='r')
    n_rows = original.shape[0]

    # Create temporary file
    tmp_filename = filename + '.tmp.npy'
    with open(tmp_filename, 'wb') as f:
        # We don't know final shape yet, so write later
        kept_rows = []
        print("Deleting!")
        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            batch_indices = np.arange(start, end)
            mask = [i not in delete_indices_set for i in batch_indices]
            batch = original[start:end][mask]  # Only keep good rows
            kept_rows.append(batch)

        # Stack and write to temp file
        print('Saving!')
        result = np.vstack(kept_rows)
        np.save(f, result)
    
    del original
    gc.collect()
    
    os.replace(tmp_filename, filename)  # Overwrite original

def deep_filter(filename, Component_Lower_Bounds=None, Component_Upper_Bounds=None, Oxide_Lower_Bounds=None, Oxide_Upper_Bounds=None, Mass_Upper_Bounds=None, batch_size=200_000):
    
    components = np.load(filename + 'labels.npy', mmap_mode='r')
    binary_labels = np.load(filename + 'binary_labels.npy', mmap_mode='r') 
    
    
    delete_indices = np.array([], dtype=int)
    
        # === Full-array filters for components (cheap)
    if Component_Lower_Bounds is not None:
        for phase, comp, bound in Component_Lower_Bounds:
            idx = detail_label_indices[phase][comp]
            to_delete = np.where((components[:, idx] < bound)*(components[:, idx] != 0))[0]
            print(f"Deleting {len(to_delete)} for {bound} Lower Bound {phase} {comp}")
            delete_indices = np.append(delete_indices, to_delete)
            
    if Component_Upper_Bounds is not None:
        for phase, comp, bound in Component_Upper_Bounds:
            idx = detail_label_indices[phase][comp]
            to_delete = np.where(components[:, idx] > bound)[0]
            print(f"Deleting {len(to_delete)} for {bound} Upper Bound {phase} {comp}")
            delete_indices = np.append(delete_indices, to_delete)

    # === Batch filtering for expensive mass/oxide filters
    n_rows = components.shape[0]
    
    for phase in all_phases:
        print(f"{phase} present in {(100*np.sum(binary_labels[:,mass_phasedict[phase]]>0.5))/n_rows}% of assemblages")
    
    print(f"Rows before deleting: {n_rows}")
  
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)

        comp_batch = components[start:end]

        batch_indices = np.arange(start, end)

        # Oxide Lower Bounds
        if Oxide_Lower_Bounds is not None:
            for phase, ox, bound in Oxide_Lower_Bounds:
                oxides_GT = (comp_batch[:,label_indices_comp[phase]] @ compToOx[label_indices[phase]]) 

                oxides_GT = oxides_GT @ MM
                oxides_GT = oxides_GT * (100/np.sum(oxides_GT,axis=1)).reshape(-1,1)

                failing = np.where((oxides_GT[:,oxide_dict[ox]] < bound)*(oxides_GT[:,oxide_dict[ox]] != 0))[0]
                print(f"Deleting {len(failing)} for {bound} Lower Bound {phase} {ox}")
                delete_indices = np.append(delete_indices, batch_indices[failing])

        # Oxide Upper Bounds
        if Oxide_Upper_Bounds is not None:
            for phase, ox, bound in Oxide_Upper_Bounds:
                oxides_GT = (comp_batch[:,label_indices_comp[phase]] @ compToOx[label_indices[phase]]) 

                oxides_GT = oxides_GT @ MM
                oxides_GT = oxides_GT * (100/np.sum(oxides_GT,axis=1)).reshape(-1,1)

                failing = np.where((oxides_GT[:,oxide_dict[ox]] > bound)*(oxides_GT[:,oxide_dict[ox]] != 0))[0]
                print(f"Deleting {len(failing)} for {bound} Upper Bound {phase} {ox}")
                delete_indices = np.append(delete_indices, batch_indices[failing])

        del comp_batch, batch_indices, oxides_GT
        gc.collect()
            
    delete_indices = np.unique(delete_indices)
    print(f"Rows after deleting: {n_rows-len(delete_indices)}")

    del components, binary_labels
    gc.collect()
    
    # Perform safe batch delete
    safe_delete_batched(filename + 'labels.npy', delete_indices)
    safe_delete_batched(filename + 'features.npy', delete_indices)
    safe_delete_batched(filename + 'binary_labels.npy', delete_indices)
    safe_delete_batched(filename + 'molar_labels.npy', delete_indices)
    safe_delete_batched(filename + 'mass_labels.npy', delete_indices)
    
    gc.collect()
    
    return delete_indices

    
def report_phase_presence_by_melt_bin(filename, bin_width=10):
    """Prints how often each phase appears in each melt-fraction bin (0%, 0–10%, ..., 90–100%)"""
    # Load data
    components = np.load(filename + 'labels.npy', mmap_mode='r')
    features = np.load(filename + 'features.npy', mmap_mode='r')
    binary_labels = np.load(filename + 'binary_labels.npy', mmap_mode='r')

    # Convert molar to intensive wt%
    massTens, _ = Converter.convertMolToIntensiveWt(torch.tensor(components), torch.tensor(features))

    melt_idx = mass_phasedict['melts-liquid']
    melt_mass = massTens[:, melt_idx]

    # Define melt bins: subsolidus + 10% melt fraction bins up to 100
    bins = [(None, 0)] + [(i, i + bin_width) for i in range(0, 100, bin_width)]

    for low, high in bins:
        if low is None:
            # Subsolidus
            mask = melt_mass == 0
            label = "Subsolidus (0%)"
        else:
            mask = (melt_mass > low) & (melt_mass <= high)
            label = f"{low:2d}–{high:3d}% melt"

        indices = np.where(mask)[0]
        sampleNo = len(indices)
        print(f"\n{label} — {sampleNo} samples\n{'-' * (len(label) + 15)}")

        if len(indices) == 0:
            print("No samples in this bin.")
            continue

        for phase_name, phase_idx in mass_phasedict.items():
            count = binary_labels[indices, phase_idx].sum()
            print(f"{phase_name:<20}: {100*int(count)/sampleNo}%")

    # Cleanup
    del components, features, binary_labels

def F_phase_plots(partition_directory, filename):
    """"Old function, probably won't work. 10/3/25
    Plots liquid chemistry against melt fraction"""

    if not os.path.exists(partition_directory):
        os.makedirs(partition_directory)

    validation_binaries = np.load(filename+"binary_labels.npy", mmap_mode='r')
    validation_labels = np.load(filename+"labels.npy", mmap_mode='r')
    validation_features = np.load(filename+"features.npy", mmap_mode='r')
    Converter = MELTS_Converter()

    if len(validation_binaries) > 500000:
        subset = np.random.choice(np.arange(0, len(validation_binaries)), size=500000, replace=False)
    else: 
        subset = np.arange(len(validation_binaries))

    massMatTrue, compMatTrue = Converter.convertMolToIntensiveWt(components=torch.tensor(validation_labels[subset]), features = torch.tensor(validation_features[subset]))
    
    compMatTrue = compMatTrue.detach().numpy()
    massMatTrue = massMatTrue.detach().numpy()

    correct_phases = np.ones_like(subset).astype(bool)
    print(np.shape(correct_phases))
    datalen = len(validation_labels[subset])
    j = 0
    gc.collect()

    
    for i, (phase, indices) in enumerate(label_indices.items()):
        if phase == 'cristobalite':
            continue
        
        realPos = (validation_binaries[subset,i] > 0.5)
        plotable = realPos
        prop_correct_plotable = 100*np.sum(plotable*correct_phases)/np.sum(plotable)

        plt.title(f'{phase} mass 1:1 Plot')
       
        plt.scatter(massMatTrue[plotable*correct_phases,mass_phasedict['melts-liquid']], 
                    massMatTrue[plotable*correct_phases,i], color = 'blue', s = 0.1*(datalen/np.sum(realPos)), 
                    alpha = 0.2, label =f'Complete Assemblage Recovered ({prop_correct_plotable}%)'
                    )

        plt.ylabel(f'True wt% {phase}')
        plt.xlabel(f'Melt F')
        plt.tight_layout()
        plt.savefig(partition_directory + '/ '[0] + phase + '_mass.jpg', dpi = 256)
        plt.show()



        if phase in compositionally_variable_phases:
            for relOxInd in active_ox_dict[phase]: 
                plt.title(f'{phase} {Oxides[relOxInd]} wt% 1:1 Plot')

                plt.scatter(massMatTrue[plotable*correct_phases,mass_phasedict['melts-liquid']],
                compMatTrue[plotable*correct_phases,comp_phasedict[phase],relOxInd], color = 'blue', 
                s = 0.1*(datalen/np.sum(realPos)), alpha = 0.2
                )

                plt.ylabel(f'True {Oxides[relOxInd]} wt% {phase}')
                plt.xlabel(f'Melt F')
                plt.tight_layout()
                plt.savefig(partition_directory + '/ '[0] + phase + Oxides[relOxInd] + 'wt_Intesive.jpg', dpi = 256)
                plt.show()


        for ind in indices:
            name = phase
            if name != label_names[ind]:
                name += ' ' + label_names[ind]



            plt.title(f'{name} 1:1 Plot')

            plt.scatter(massMatTrue[plotable*correct_phases,mass_phasedict['melts-liquid']], 
            validation_labels[subset[plotable*correct_phases],ind], color = 'blue', s = 0.1*(datalen/np.sum(realPos)),
             alpha = 0.2, label =f'Complete Assemblage Recovered ({prop_correct_plotable}%)'
             )
            plt.ylabel(f'True molar {label_names[ind]}')
            plt.xlabel(f'Melt F')
            plt.tight_layout()
            plt.savefig(partition_directory + '/ '[0] + name + '.jpg', dpi = 256)
            plt.show()
            j += 1


    del validation_binaries, validation_labels, validation_features
    gc.collect()
    
def retrieve_component_moles(self, multiplier_bounds = [1,1]):
    """Generates matrix of assemblage in absolute component form. Multipliers to reweight """
    total_rows = np.shape(self.table1)[0]
    num_cols = label_indices['melts-liquid'][-1]+1
    
    self.molar = np.lib.format.open_memmap(
            self.filename+'component_moles.npy',
            mode='w+',
            dtype=np.float32,
            shape=(total_rows, num_cols)
        )
    
    for phase in list(label_indices.keys()):
        mass_multipliers = np.random.uniform(*multiplier_bounds, size = total_rows).reshape(-1,1) # Vary proportions of equilibrium assemblages
        #First, Find molar mass of phase
        if phase != 'melts-liquid':
            if len(label_indices[phase]) > 1: #Variable componsition. Components to Molar oxides per formula unit to Molar Mass
                compnames = np.array(label_names)[label_indices[phase]]
                X_ind = np.array([component_indices[phase][compname] for compname in compnames])
                MM = (self.table1[:,X_ind] @ compToOx[label_indices[phase]]) @ Mtot
                #Then moles of each component is multiplied by the total moles of phase (phase mass/MM)
                zero_mat = np.zeros_like(MM, dtype=float) 
                self.molar[:,label_indices[phase]] = self.table1[:,X_ind] * mass_multipliers* np.divide(np.atleast_2d(self.table1[:,list(component_indices[phase].values())[0]]).T, MM, out=zero_mat, where=MM != 0)                                                           
            else: #Invariant Phase Composition, moles = mass / MM
                self.molar[:,label_indices[phase]] = ((mass_multipliers*self.table1[:,component_indices[phase]['mass (gm)']].reshape(-1,1) / (compToOx[label_indices[phase],:] @ Mtot)).T).T
        else: # Liquid is weight percent. First, go to moles, then do the rest of the calculation
            X_ind = np.array([component_indices['melts-liquid'][f"wt% {wrk}"] for wrk in Oxides])
            UnNormed = self.table1[:,X_ind] @ Minv[:len(X_ind), :len(X_ind)] # Moles oxides
            tot_unnormed = np.sum(UnNormed, axis = 1).reshape(-1,1)
            zero_mat = np.zeros_like(UnNormed, dtype=float) 
            molar_X_liquid = np.divide(UnNormed,tot_unnormed, out=zero_mat, where=tot_unnormed != 0)
            MM = molar_X_liquid @ Mtot[:len(X_ind), :len(X_ind)]
            zero_mat = np.zeros_like(MM, dtype=float) 
            self.molar[:,label_indices[phase]] = (molar_X_liquid * mass_multipliers * np.divide(self.table1[:,list(component_indices[phase].values())[0]].reshape(-1,1), MM.reshape(-1,1), out=zero_mat, where=MM != 0)) @ oxToEl[:len(X_ind), :len(X_ind)]                                                  

def retrieve_bulk_elements(self):
    """Generates matrix of assemblage in absolute component form. Multipliers to reweight """
    total_rows = np.shape(self.table1)[0]
    num_cols = len(Elkeys)
    
    self.bulk = np.lib.format.open_memmap(
            self.filename+'bulk_elements.npy',
            mode='w+',
            dtype=np.float32,
            shape=(total_rows, num_cols)
        )
    retrieve_component_moles(self)
    try:
        self.bulk[:] = self.molar @ compToOx @ oxToEl
        row_sums = np.sum(self.bulk, axis=1)
        row_sums[row_sums == 0] = 1.0
        self.bulk[:] = self.bulk / row_sums[np.newaxis, :]
    finally:
        self.bulk.flush()
    
    
    
    
def resampling_to_datasets(self, resample_bounds = [[1,1]], clear_old_tables=False):
    """Builds features and labels for training"""
    sampleNo = len(resample_bounds)
    
    num_rows = np.shape(self.table)[0]
    total_rows = np.shape(self.table)[0]
    num_components_intensive = label_indices_comp['melts-liquid'][-1]+1
    #num_components_extensive = label_indices['melts-liquid'][-1]+1
    num_phases = mass_phasedict['melts-liquid']+1
    print('INITIALIZE')
    
    if clear_old_tables:
        del self.binarylabels
        del self.features
        del self.labels
        del self.masslabels
        del self.table1
        del self.molar
        gc.collect()
        
    new_file = self.filename + '_temp.npy'
    self.table1 = np.lib.format.open_memmap(
            new_file,
            mode='w+',
            dtype=self.table.dtype,
            shape=self.table.shape
    )

    # Copy data in blocks or all at once (depending on size)
    self.table1[:] = self.table[:]
    self.table1.flush()
    
    if self.blurredbinaries is None:
        binary_mask = (self.table[:,mass_indices]>0).astype(int)
    else:
        assert self.blurredbinaries.shape[0] == self.table.shape[0], "Table of labels and blurred binaries must have the same number of rows!"
        print('Using Blurred Binaries to generate binary labels...')
        binary_mask = self.blurredbinaries
        
    self.molarlabels =np.lib.format.open_memmap( # Molar abundances
            self.filename+ 'molar_labels.npy',
            mode='w+',
            dtype=np.float32,
            shape=(num_rows*len(resample_bounds), num_phases)
    )
        
    self.binarylabels = np.lib.format.open_memmap( # Flags of present phases
            self.filename+'binary_labels.npy',
            mode='w+',
            dtype=np.float32,
            shape=(num_rows*len(resample_bounds), num_phases)
    )
    
    self.masslabels = np.lib.format.open_memmap( # Masses in grams, normed to 100
            self.filename+'mass_labels.npy',
            mode='w+',
            dtype=np.float32,
            shape=(num_rows*len(resample_bounds), num_phases)
    )
    
    self.features = np.lib.format.open_memmap( # Input, PTfO2, Bulk chemistry X in element moles, normed to sum of 1
            self.filename+'features.npy',
            mode='w+',
            dtype=np.float32,
            shape=(num_rows*len(resample_bounds), 3 + len(Elkeys))
        )
    
    self.labels = np.lib.format.open_memmap( # Components in moles, intensive only
            self.filename+'labels.npy',
            mode='w+',
            dtype=np.float32,
            shape=(num_rows*len(resample_bounds), num_components_intensive)
        )
    
    try:
        for i, bounds in enumerate(tqdm(resample_bounds, desc = 'Generating Molar Features and Labels', leave = False)):
            print(f"SAMPLE: {i}")
            mass_multipliers = np.random.uniform(*bounds, size = total_rows*num_phases).reshape(total_rows,num_phases) # Vary proportions of equilibrium assemblages
            self.table1[:,mass_indices] *= mass_multipliers
            self.table1[:,mass_indices] *= 100/np.sum(self.table1[:,mass_indices], axis = 1, keepdims = True)
            self.table1.flush()
            
            # Get molar quantities
            retrieve_component_moles(self)
            Inmoles = (self.molar @ compToOx) @ oxToEl
            InTot = np.sum(Inmoles, axis = 1).reshape(-1,1)
            
            # --- Binary labels
            sl = np.s_[i*num_rows:(i+1)*num_rows]
            self.binarylabels[sl] = binary_mask
            self.binarylabels.flush()
            del sl

            # --- Mass labels
            sl = np.s_[i*num_rows:(i+1)*num_rows]
            self.masslabels[sl] = self.table1[:, mass_indices]
            self.masslabels.flush()
            del sl

            # --- Features (bulk composition slices)
            sl = np.s_[i*num_rows:(i+1)*num_rows, 3:]
            self.features[sl] = (Inmoles / InTot)
            self.features.flush()
            del sl

            sl = np.s_[i*num_rows:(i+1)*num_rows, :3]
            self.features[sl] = self.table[:, :3]
            self.features.flush()
            del sl

            # --- Molar labels
            sl = np.s_[i*num_rows:(i+1)*num_rows]
            self.molarlabels[sl] = (self.molar / InTot) @ phaseToCompMap.T
            self.molarlabels.flush()
            del sl

            # Explicitly collect to close lingering references
            gc.collect()
            
            for phase, compdict in detail_label_indices.items(): # Move components into the right space. Already Normed to 1.
                for component, ind in compdict.items():
                    if phase != 'melts-liquid':
                        
                        # --- Labels (phase components)
                        sl = np.s_[i*num_rows:(i+1)*num_rows, ind]
                        self.labels[sl] = self.table[:, component_indices[phase][component]]
                        self.labels.flush()
                        del sl
                        
                    else:
                        pass
                if phase == 'melts-liquid':
                    liq_mol = self.molar[:,label_indices[phase]] # Non-normalized liquid element moles
                    liq_tot = np.sum(liq_mol, axis = 1)
                    liqNonzero = liq_tot != 0
                    liq_mol[liqNonzero] = liq_mol[liqNonzero] / liq_tot[liqNonzero].reshape(-1,1) # Normalize to sum 1
                    
                    # --- Labels (phase components)
                    sl = np.s_[i*num_rows:(i+1)*num_rows, label_indices_comp[phase]]
                    self.labels[sl] = liq_mol
                    self.labels.flush()
                    
                    del liq_mol, liq_tot, liqNonzero, sl
                    gc.collect()

            del self.molar, Inmoles, InTot #  Delete ALL memmap references!
            gc.collect()

        ## Filter out data where features are improperly summed. Why is that? Very small proportion, so we ignore for now. 

        bulk_wt_ox = (self.features[:,3:] @ np.linalg.inv(oxToEl[:-1])) @ MM[:-1,:-1]
        bulk_wt_ox = 100*bulk_wt_ox/np.sum(bulk_wt_ox, axis = 1).reshape(-1,1)

        GT_comps = np.zeros((self.features.shape[0],label_indices['melts-liquid'][-1]+1))

        for phase in np.array(list(label_indices.keys())):
            if phase in compositionally_variable_phases:
                GT_comps[:,label_indices[phase]] = (self.molarlabels[:, mass_phasedict[phase]]).reshape(-1,1) * self.labels[:,label_indices_comp[phase]]
            else:
                GT_comps[:,label_indices[phase]] = (self.molarlabels[:, mass_phasedict[phase]]).reshape(-1,1)

        GTReconBulk_oxides = (((GT_comps @ compToOx) @ oxToEl) @ np.linalg.inv(oxToEl[:-1])) @ MM[:-1,:-1]
        GTReconBulk_oxides =  GTReconBulk_oxides*100/np.sum(GTReconBulk_oxides,axis=1, keepdims=True)


        # This is the indices of data to remove from self.binarylabels, self.masslabels, self.features, self.labels, and self.molarlabels
        mismatches = np.unique(np.where(np.round(bulk_wt_ox,2) != np.round(GTReconBulk_oxides,2))[0])

        # Indices of rows that are good
        keep_mask = np.ones(num_rows*len(resample_bounds), dtype=bool)
        keep_mask[mismatches] = False
        
        
        #Clearing more references...
        del bulk_wt_ox, GTReconBulk_oxides, GT_comps
        gc.collect()
       
    finally: #Close Memmaps
        del self.binarylabels, self.masslabels, self.features, self.labels, self.table1, self.molarlabels
        gc.collect()
        
    return filter_invalid_rows(self, mismatches)
    
    
def filter_invalid_rows(self, mismatches):
    """Remove invalid rows from all generated memmaps."""
    if mismatches.size == 0:
        print("No mismatches to filter.")
        return

    _delete_memmap_rows(self.filename + "binary_labels.npy", mismatches)
    _delete_memmap_rows(self.filename + "mass_labels.npy", mismatches)
    _delete_memmap_rows(self.filename + "features.npy", mismatches)
    _delete_memmap_rows(self.filename + "labels.npy", mismatches)
    _delete_memmap_rows(self.filename + "molar_labels.npy", mismatches)
    print(f"Removed {len(mismatches)} invalid rows.")


def _delete_memmap_rows(filename, indices_to_delete):
    arr = np.load(filename, mmap_mode="r+")
    keep_mask = np.ones(arr.shape[0], dtype=bool)
    keep_mask[indices_to_delete] = False
    new_filename = filename.replace(".npy", "_filtered.npy")

    new_arr = np.lib.format.open_memmap(
        new_filename, mode="w+", dtype=arr.dtype, shape=(keep_mask.sum(), arr.shape[1])
    )
    new_arr[:] = arr[keep_mask]
    new_arr.flush()
    del arr, new_arr
    os.replace(new_filename, filename)
    
def balance_lowF(self, sacred_phases=None, batch_size=200_000):

    if sacred_phases:
        sacredIDX = np.array([mass_phasedict[phase] for phase in sacred_phases])
        check_sacred = True
    else:
        sacredIDX = None
        check_sacred = False

    delete_indices = np.array([], dtype=int)

    melt_mass = self.table[:, component_indices['melts-liquid']['liq mass (gm)']]*(100 /(np.sum(self.table[:,mass_indices],axis = 1)+1E-8))
    

    # Determine target bin size from middle melt fraction range (40–60%)
    middleMasses = (melt_mass > 40) & (melt_mass < 60)
    targetBinNo = int(middleMasses.sum() / 2)

    print(f"Target bin amount: {targetBinNo}")

    def select_deletable_indices(mask, num_to_delete):
        """Helper to select deletable entries that lack sacred phases."""
        if num_to_delete <= 0:
            print('None to delete for balancing melt fraction')
            return np.array([], dtype=int)
        # Only consider entries in the block
        block_indices = np.where(mask)[0]
        if len(block_indices) == 0:
            print('No potential sims passed to delete for balancing melt fraction?')
            return np.array([], dtype=int)

        if check_sacred:
            # Check for sacred phases
            sacred_presence = np.sum(self.binary_labels[block_indices][:, sacredIDX], axis = 1)
            deletable = block_indices[sacred_presence == 0]
            print(f"{num_to_delete} of {len(deletable)} will be deleted; {sacred_presence.sum()} sacred phases avoided.")
        else:
            # If no sacred phases, all are deletable
            deletable = block_indices
            print(f"{num_to_delete} of {len(deletable)} will be deleted.")
            
        
        if len(deletable) == 0:
            return np.array([], dtype=int)

        # Randomly sample deletable indices
        sample_size = min(len(deletable), num_to_delete)
        return np.random.choice(deletable, size=sample_size, replace=False)
    
    # Tertiary melt fraction block: 10–20%
    tertiaryBlock = (melt_mass > 10) & (melt_mass <= 20)
    numToDelete = int(tertiaryBlock.sum() - targetBinNo)
    delete_indices = np.append(delete_indices, select_deletable_indices(tertiaryBlock, numToDelete))

    # Secondary melt fraction block: 0–10%
    secondaryBlock = (melt_mass > 0) & (melt_mass <= 10)
    numToDelete = int(secondaryBlock.sum() - targetBinNo)
    delete_indices = np.append(delete_indices, select_deletable_indices(secondaryBlock, numToDelete))
    
    # Near-Liquidus melt fraction block: 0–10%
    upperBlock = (melt_mass >= 90) & (melt_mass < 100)
    numToDelete = int(upperBlock.sum() - (2*targetBinNo))
    delete_indices = np.append(delete_indices, select_deletable_indices(upperBlock, numToDelete))

    # Subsolidus block: exactly 0%
    solidBlock = melt_mass == 0
    numToDelete = int(solidBlock.sum() - targetBinNo)
    delete_indices = np.append(delete_indices, select_deletable_indices(solidBlock, numToDelete))
    
    # Superliquidus block: exactly 100%
    liquidBlock = (melt_mass >= 99) #Account for not always perfectly summing to 1
    numToDelete = int(liquidBlock.sum() - targetBinNo)
    delete_indices = np.append(delete_indices, select_deletable_indices(liquidBlock, numToDelete))
    
    #Delete ALL references to memory map
    del melt_mass, tertiaryBlock, secondaryBlock, solidBlock, liquidBlock, upperBlock
    gc.collect()
    
    print(f"Deleting {len(delete_indices)} entries")

    self.delete(delete_indices)
    
def balance_Superliquidus_fxtal(self, sacred_phases=None, batch_size=200_000):

    if sacred_phases:
        sacredIDX = np.array([mass_phasedict[phase] for phase in sacred_phases])
        check_sacred = True
    else:
        sacredIDX = None
        check_sacred = False

    delete_indices = np.array([], dtype=int)

    melt_mass = self.table[:, component_indices['melts-liquid']['liq mass (gm)']]
    melt_mass = melt_mass * 100/(np.sum(self.table[:,mass_indices],axis = 1)+1E-8)

    # Determine target bin size from middle melt fraction range (40–60%)
    middleMasses = (melt_mass > 95) & (melt_mass < 99.95)
    targetBinNo = int(middleMasses.sum() / 3)

    print(f"Target bin amount: {targetBinNo}")

    def select_deletable_indices(mask, num_to_delete):
        """Helper to select deletable entries that lack sacred phases."""
        if num_to_delete <= 0:
            print('None to delete for balancing melt fraction')
            return np.array([], dtype=int)
        # Only consider entries in the block
        block_indices = np.where(mask)[0]
        if len(block_indices) == 0:
            print('No potential sims passed to delete for balancing melt fraction?')
            return np.array([], dtype=int)

        if check_sacred:
            # Check for sacred phases
            sacred_presence = np.sum(self.binary_labels[block_indices][:, sacredIDX], axis = 1)
            deletable = block_indices[sacred_presence == 0]
            print(f"{num_to_delete} of {len(deletable)} will be deleted; {sacred_presence.sum()} sacred phases avoided.")
        else:
            # If no sacred phases, all are deletable
            deletable = block_indices
            print(f"{num_to_delete} of {len(deletable)} will be deleted.")
            
        
        if len(deletable) == 0:
            return np.array([], dtype=int)

        # Randomly sample deletable indices
        sample_size = min(len(deletable), num_to_delete)
        return np.random.choice(deletable, size=sample_size, replace=False)
    

    # Superliquidus block
    liquidBlock = (melt_mass >= 99.95) #Account for not always perfectly summing to 1
    numToDelete = int(liquidBlock.sum() - targetBinNo)
    delete_indices = np.append(delete_indices, select_deletable_indices(liquidBlock, numToDelete))
    
    # Subsolidus block: exactly 0%
    solidBlock = melt_mass == 0
    numToDelete = int(solidBlock.sum() - (targetBinNo/2))
    delete_indices = np.append(delete_indices, select_deletable_indices(solidBlock, numToDelete))
    
    #Delete ALL references to memory map
    del melt_mass, liquidBlock, solidBlock
    gc.collect()
    
    print(f"Deleting {len(delete_indices)} entries")

    self.delete(delete_indices)
    
def make_Tplots(MELTS, plot_directory, colormap = 'turbo'):
    if not os.path.exists(plot_directory[:-1]):
        os.makedirs(plot_directory[:-1])
    
    conditional = MELTS.table[:,component_indices['melts-liquid']['liq mass (gm)']]>0
    
    if np.shape(MELTS.table)[0]> 200000:
        conditional = np.where(conditional)[0]
        conditional = conditional[np.random.randint(0,len(conditional),200000)]
        
    SiO2 = MELTS.table[conditional, component_indices['melts-liquid']['wt% SiO2']]
    Temp = MELTS.table[conditional,component_indices['System_main']['Temperature']]
    Pres = MELTS.table[conditional,component_indices['System_main']['Pressure']]
    fO2 = MELTS.table[conditional,component_indices['System_main']['logfO2-QFM']]
    plt.hist(Temp)
    plt.xlabel('Temperature °C')
    plt.ylabel('Counts')
    plt.title('MELTS Dataset Temperature Histogram')
    plt.savefig(plot_directory+'Temperature_Hist')
    plt.show()

    plt.hist(Pres)
    plt.xlabel('Pressure (Bars)')
    plt.ylabel('Counts')
    plt.title('MELTS Dataset Pressure Histogram')
    plt.savefig(plot_directory+'Pressure_Hist')
    plt.show()

    plt.hist(fO2)
    plt.xlabel(r"$log_{10}\left(fO_2\right) - QFM$")
    plt.ylabel('Counts')
    plt.title(r'MELTS Dataset $log_{10}\left(fO_2\right) - QFM$ Histogram')
    plt.savefig(plot_directory+'fO2_Hist')
    plt.show()

    plt.scatter(Pres, fO2, s=5, alpha = 0.1, c = Temp)
    plt.xlabel('Pressure (Bars)')
    plt.ylabel(r"$log_{10}\left(fO_2\right) - QFM$")
    plt.title('MELTS Dataset Pressure x fO2')
    plt.savefig(plot_directory+'fO2XPressure')
    plt.show()

    for label, ind in component_indices['melts-liquid'].items():
        if label == 'liq rho (gm/cc)':
            break
        data = MELTS.table[conditional,ind]
        if label == 'liq mass (gm)':
            literal_label = 'Liquid Mass Fraction (%)'
        else:
            literal_label = ""
            for char in label:
                if np.isnan(pull_number(char)):
                    if char == ' ':
                        literal_label += ' $'
                    else:
                        literal_label += char
                else:
                    literal_label += r'_' + char
        literal_label += r'$'
        plt.hist(data)
        plt.xlabel(label)
        plt.savefig(plot_directory+f'{label}_Hist')
        plt.show()
        #if label != 'wt% SiO2':
        fig, ax = plt.subplots()
        plt.scatter(Temp, data, s=5, alpha = 0.2, cmap=colormap, c = SiO2)#, c = Temp)
        plt.xlabel('Temperature, C')#r'$wt\% SiO_2$')
        plt.ylabel(literal_label)
        norm = plt.Normalize(vmin=np.min(SiO2), vmax=np.max(SiO2))
        sm = cm.ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('wt% SiO2')
        cbar.set_alpha(0.8)
        plt.title(f'Liquid {literal_label}')
        plt.tight_layout()
        plt.savefig(plot_directory+f'{label}XTemp')
        plt.show()
            
def make_harkers(MELTS, plot_directory, colormap = 'turbo', hist = False):
    if not os.path.exists(plot_directory[:-1]):
        os.makedirs(plot_directory[:-1])
        
    conditional = MELTS.table[:,component_indices['melts-liquid']['liq mass (gm)']]>0
    if np.shape(MELTS.table)[0]> 200000:
        conditional = np.where(conditional)[0]
        conditional = conditional[np.random.randint(0,len(conditional),200000)]
        
    SiO2 = MELTS.table[conditional, component_indices['melts-liquid']['wt% SiO2']]
    Temp = MELTS.table[conditional,component_indices['System_main']['Temperature']]
    Pres = MELTS.table[conditional,component_indices['System_main']['Pressure']]
    fO2 = MELTS.table[conditional,component_indices['System_main']['logfO2-QFM']]
    if hist:
        plt.hist(Temp)
        plt.xlabel('Temperature °C')
        plt.ylabel('Counts')
        plt.title('MELTS Dataset Temperature Histogram')
        plt.savefig(plot_directory+'Temperature_Hist')
        plt.show()

        plt.hist(Pres)
        plt.xlabel('Pressure (Bars)')
        plt.ylabel('Counts')
        plt.title('MELTS Dataset Pressure Histogram')
        plt.savefig(plot_directory+'Pressure_Hist')
        plt.show()

        plt.hist(fO2)
        plt.xlabel(r"$log_{10}\left(fO_2\right) - QFM$")
        plt.ylabel('Counts')
        plt.title(r'MELTS Dataset $log_{10}\left(fO_2\right) - QFM$ Histogram')
        plt.savefig(plot_directory+'fO2_Hist')
        plt.show()

        plt.scatter(Pres, fO2, s=5, alpha = 0.2, c = Temp)
        plt.xlabel('Pressure (Bars)')
        plt.ylabel(r"$log_{10}\left(fO_2\right) - QFM$")
        plt.title('MELTS Dataset Pressure x fO2')
        plt.savefig(plot_directory+'fO2XPressure')
        plt.show()

    for label, ind in component_indices['melts-liquid'].items():
        if label == 'liq rho (gm/cc)':
            break
        
        data = MELTS.table[conditional,ind]
        if label == 'liq mass (gm)':
            literal_label = 'Liquid Mass Fraction (%)'
        else:
            literal_label = ""
            for char in label:
                if np.isnan(pull_number(char)):
                    if char == ' ':
                        literal_label += ' $'
                    else:
                        literal_label += char
                else:
                    literal_label += r'_' + char
            literal_label += r'$'
        if hist:
            plt.hist(data)
            plt.xlabel(label)
            plt.savefig(plot_directory+f'{label}_Hist.jpg', dpi = 256)
            plt.show()
        if label != 'wt% SiO2':
            fig, ax = plt.subplots()
            plt.scatter(SiO2, data, s=5, alpha = 0.2, cmap=colormap, c = Temp)#, c = Temp)
            plt.xlabel(r'$wt\% SiO_2$')#r'$wt\% SiO_2$')
            plt.ylabel(literal_label)
            norm = plt.Normalize(vmin=np.min(Temp), vmax=np.max(Temp))
            sm = cm.ScalarMappable(norm=norm, cmap=colormap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Temperature (C)')
            cbar.set_alpha(0.8)
            plt.title(f'Liquid {literal_label}')
            plt.tight_layout()
            plt.savefig(plot_directory+f'{label}XSiO2')
            plt.show()


def extract_rows(filename, indices, new_suffix="_subset"):
    """
    Extracts specific rows by index from a CSV and TXT file pair.
    
    Args:
        filename (str): Base filename (no extension).
        indices (list[int]): Row indices to extract (0-based, excluding header for csv).
        new_suffix (str): Suffix for new files (default "_subset").
    """
    indices = sorted(set(indices))  # ensure sorted & unique
    indices_iter = iter(indices)
    try:
        next_idx = next(indices_iter)
    except StopIteration:
        next_idx = None  # no indices to write

    # --- Handle CSV ---
    csv_in = f"{filename}.csv"
    csv_out = f"{filename}{new_suffix}.csv"

    with open(csv_in, "r", newline="", encoding="utf-8") as fin, \
         open(csv_out, "w", newline="", encoding="utf-8") as fout:
        
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        # Write header
        header = next(reader)
        writer.writerow(header)

        # Now rows start at index 0 (after header)
        for row_idx, row in enumerate(reader):
            if next_idx is None:
                break
            if row_idx == next_idx:
                writer.writerow(row)
                try:
                    next_idx = next(indices_iter)
                except StopIteration:
                    next_idx = None

    # --- Handle TXT ---
    txt_in = f"{filename}.txt"
    txt_out = f"{filename}{new_suffix}.txt"

    with open(txt_in, "r", encoding="utf-8") as fin, \
         open(txt_out, "w", encoding="utf-8") as fout:
        
        indices_iter = iter(indices)
        try:
            next_idx = next(indices_iter)
        except StopIteration:
            next_idx = None

        for row_idx, line in enumerate(fin):
            if next_idx is None:
                break
            if row_idx == next_idx:
                fout.write(line)
                try:
                    next_idx = next(indices_iter)
                except StopIteration:
                    next_idx = None