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
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

try:
    import resource  # POSIX only (Linux/WSL/macOS) - absent on Windows.
except ImportError:  # pragma: no cover - platform-dependent
    resource = None

# Ensure src is on path
import sys
from pathlib import Path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import from refactored modules
from builder.indexer import DatasetIndexer
from ngibbs.utils.file_utils import move_file, chunked_mask_copy
from .guardrails import (  # guardrails: see module docstring
    assert_identity_multipliers, assert_no_derivative_sidecars,
)
from . import sidecar as _sidecar  # dn/dP, dn/dT tables carried parallel to self.table


def _log_mem(label):
    """Print current process peak RSS so far (stdlib-only, no psutil dependency).

    On Linux, ``ru_maxrss`` from ``getrusage`` is the high-water-mark resident
    set size in KB - i.e. it only ever grows within a process, so consecutive
    calls show how peak memory climbs across pipeline stages. Intended as a
    diagnostic breadcrumb trail for large (tens-of-millions-of-rows) datasets
    where a crash can otherwise only be localized to "somewhere in this call".
    No-ops quietly if `resource` isn't available (e.g. native Windows).
    """
    if resource is None:
        return
    try:
        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        print(f"[MEM] {label}: peak RSS so far = {rss_gb:.2f} GB")
    except Exception:
        pass


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
                 allow_differing_lengths=False, header=None, Model='MELTS', OXYGEN='closed',
                 chunk_size=100_000):

        # Default row-chunk size for every large-table scan/copy/filter/resample
        # method below that doesn't get an explicit chunk_size argument (see YAML
        # `performance.chunk_size`, threaded in from prepareML_fullvalid.py).
        self.chunk_size = chunk_size
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
    
            self._csv_to_memmap(chunk_size=self.chunk_size)
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
        _log_mem("After CSV/memmap load, before metadata text parsing")

        # --- Read TXT metadata, encoding the low-cardinality fields (run id, value
        # tag, MELTS version) as int32 codes into small vocab tables rather than
        # boxing every row as its own Python string. A run's steps repeat the same
        # run/value/version tokens over and over, so a run_id/version array does
        # not need one Python object per row - just an int per row plus one string
        # per distinct value.
        run_group_vocab, value_vocab, run_id_vocab, version_vocab = {}, {}, {}, {}
        metadata_vocab = {}
        run_group_codes, value_codes, run_id_codes, version_codes = [], [], [], []
        metadata_codes = []

        def _code_for(vocab_map, key):
            code = vocab_map.get(key)
            if code is None:
                code = len(vocab_map)
                vocab_map[key] = code
            return code

        with open(self.txt_file, 'r') as f:
            for i, line in tqdm(enumerate(f), desc = 'Reading Text...', total=int(file_rows)):
                line = line.strip()
                parts = line.split(' ', 1)
                id_parts = parts[0].split(':')
                if len(id_parts) == 1:
                    value_str, run_id_str, version_str = '', '', id_parts[0]
                else:
                    value_str, run_id_str, version_str = id_parts[0], id_parts[1], id_parts[2]
                run_group_codes.append(_code_for(run_group_vocab, value_str + run_id_str))
                value_codes.append(_code_for(value_vocab, value_str))
                run_id_codes.append(_code_for(run_id_vocab, run_id_str))
                version_codes.append(_code_for(version_vocab, version_str))
                # Metadata (the free-text suffix after run/value/version) is vocab-coded
                # exactly like the other fields, rather than stored as a raw per-row
                # string. At 90M-row scale, np.array(list_of_python_strs) picks a
                # SINGLE fixed unicode width equal to the LONGEST string across every
                # row, then pays that width for all 90M rows (e.g. a 50-char outlier
                # row -> 50 * 4 bytes * 90M = ~18 GB for this array alone). Since
                # metadata values are highly repetitive (mostly '' plus a handful of
                # phase-annotation patterns), coding them into a small vocab collapses
                # this to a 90M-entry int32 array (~360 MB) plus a tiny string table.
                metadata_codes.append(_code_for(metadata_vocab, parts[1] if len(parts) > 1 else ''))
        print(f"[TIMER] Parsed {total_rows} of Metadata to Lists: completed in {time.time() - t_start:.2f} seconds")

        if working_text:
            self.txt_file = self.filename + '.txt' # Now change name if we are using a 'working' version after reading text.

        # --- Convert to compact typed storage. run_indices/MELTSversion are int32
        # codes into small vocab arrays (the concatenated value+run-id key and the
        # MELTS version string are both highly repeated within a run); value_codes
        # and run_id_codes are kept separately only so the original line's
        # value:run_id:version structure can be reconstructed for save/save_txt.
        # metadata (the free-text suffix, not reliably numeric - see
        # recover_untracked_phases) is vocab-coded the same way as run/value/
        # version below, NOT stored as a per-row fixed-width string array: at
        # 90M rows, np.array() over Python strings picks one unicode width equal
        # to the single longest string across the whole table and pays that
        # width for every row, which is the dominant cause of OOM on large
        # datasets (a 50-char outlier row -> ~18 GB for this array alone).
        t_start = time.time()

        def _vocab_array(vocab_map):
            return np.array(list(vocab_map.keys()), dtype=str) if vocab_map else np.array([], dtype='<U1')

        self.run_indices = np.array(run_group_codes, dtype=np.int32)
        self._run_group_vocab = _vocab_array(run_group_vocab)
        self._value_codes = np.array(value_codes, dtype=np.int32)
        self._value_vocab = _vocab_array(value_vocab)
        self._run_id_codes = np.array(run_id_codes, dtype=np.int32)
        self._run_id_vocab = _vocab_array(run_id_vocab)
        self.MELTSversion = np.array(version_codes, dtype=np.int32)
        self._melts_version_vocab = _vocab_array(version_vocab)
        self._metadata_codes = np.array(metadata_codes, dtype=np.int32)
        self._metadata_vocab = _vocab_array(metadata_vocab)
        del run_group_codes, value_codes, run_id_codes, version_codes, metadata_codes
        gc.collect()
        print(f"[TIMER] Metadata Lists to arrays: completed in {time.time() - t_start:.2f} seconds")
        _log_mem("After metadata vocab-coding, before phase-abundance reporting")

        

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

        # --- Handle derivative sidecars (dn/dP, dn/dT). Located from the ORIGINAL
        # --- base name, since that is what the importer wrote; absent sidecars leave
        # --- the attributes at None and every path below keeps its previous behaviour.
        _sidecar.attach(self, filename, memmap_mode=memmap_mode,
                        chunk_size=self.chunk_size)
        self.memmap_file = self.filename + '.npy'
        self.txt_file = self.filename + '.txt'

        len_problem = False
        row_expected = self.table.shape[0]
        if len(self._value_codes) != row_expected:
            print(f'WARNING: Metadata length {len(self._value_codes)} does not match data rows: {row_expected}')
            len_problem = True
        if len(self.run_indices) != row_expected:
            print(f'WARNING: Run Indices length {len(self.run_indices)} does not match data rows: {row_expected}')
            len_problem = True
        if len(self.MELTSversion) != row_expected:
            print(f'WARNING: MELTSversion length {len(self.MELTSversion)} does not match data rows: {row_expected}')
            len_problem = True
        if len(self._metadata_codes) != row_expected:
            print(f'WARNING: Metadata length {len(self._metadata_codes)} does not match data rows: {row_expected}')
            len_problem = True

        if len_problem and not allow_differing_lengths:
            rows = self.table.shape[0]
            del self.table
            gc.collect()
            raise ValueError(f'Metadata length does not match data rows: {rows}')

    @property
    def meta(self):
        """Reconstruct every raw metadata line from the compact encoded fields.

        Kept for backward compatibility (e.g. interactive use); materializes the
        whole table's lines at once, so prefer `write_meta_lines`/`_build_meta_lines`
        for anything operating on a large table.
        """
        return self._build_meta_lines()

    @property
    def metadata(self):
        """Full per-row metadata (free-text) strings, reconstructed from the
        compact vocab-coded storage (`_metadata_codes`/`_metadata_vocab`; see
        __init__). Kept for backward compatibility / interactive use only -
        this allocates an array sized (widest distinct metadata string) x
        (row count), which is exactly the allocation this vocab coding was
        added to avoid for a 90M-row table. Internal code should index
        `_metadata_vocab[_metadata_codes[...]]` for a bounded subset instead.
        """
        return self._metadata_vocab[self._metadata_codes]

    def _build_meta_lines(self, indices=None):
        """Reconstruct raw ``value:run_id:version metadata`` lines for `indices`
        (an index array/boolean mask/slice, or None for all rows)."""
        if indices is None:
            indices = slice(None)
        value_strs = self._value_vocab[self._value_codes[indices]]
        run_id_strs = self._run_id_vocab[self._run_id_codes[indices]]
        version_strs = self._melts_version_vocab[self.MELTSversion[indices]]
        # Reconstruct only the requested subset's metadata strings from the vocab -
        # not self.metadata (which would materialize every row at once; see the
        # `metadata` property docstring). Callers that want the whole table pass
        # `indices=None` deliberately (e.g. the `meta` property); `write_meta_lines`
        # always passes a bounded chunk, so peak RAM here stays O(chunk size).
        metadata_strs = self._metadata_vocab[self._metadata_codes[indices]]
        n = len(metadata_strs)
        lines = np.empty(n, dtype=object)
        for i in range(n):
            if value_strs[i] == '' and run_id_strs[i] == '':
                head = version_strs[i]
            else:
                head = f"{value_strs[i]}:{run_id_strs[i]}:{version_strs[i]}"
            lines[i] = f"{head} {metadata_strs[i]}" if metadata_strs[i] else head
        return lines

    def write_meta_lines(self, fileobj, indices=None, chunk_size=None):
        """Stream reconstructed metadata lines to `fileobj`, chunked so peak RAM
        stays bounded regardless of table size. `indices` may be an index array
        (e.g. the rows moved by `split`) or None to write every row in order."""
        if chunk_size is None:
            chunk_size = self.chunk_size
        n = len(self._metadata_codes) if indices is None else len(indices)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_indices = slice(start, end) if indices is None else indices[start:end]
            for line in self._build_meta_lines(chunk_indices):
                fileobj.write(line + '\n')

    def _filter_metadata_rows(self, keep):
        """Restrict every metadata field to the rows selected by `keep` (a
        boolean mask or integer index array)."""
        self.run_indices = self.run_indices[keep]
        self.MELTSversion = self.MELTSversion[keep]
        self._metadata_codes = self._metadata_codes[keep]
        self._value_codes = self._value_codes[keep]
        self._run_id_codes = self._run_id_codes[keep]

    def _clear_metadata_rows(self):
        del self.run_indices, self.MELTSversion, self._metadata_codes, self._value_codes, self._run_id_codes
        gc.collect()

    def _append_metadata_rows(self, source_selector, n_repeats=1):
        """Append `n_repeats` copies of the rows selected by `source_selector`
        (a boolean mask or integer index array) to every metadata field."""
        self.run_indices = np.append(self.run_indices, np.repeat(self.run_indices[source_selector], n_repeats))
        self.MELTSversion = np.append(self.MELTSversion, np.repeat(self.MELTSversion[source_selector], n_repeats))
        # Repeated rows reference the same vocab entries as their source rows, so
        # only the (small) int32 code array needs to grow - the vocab itself is
        # unchanged.
        self._metadata_codes = np.append(self._metadata_codes, np.repeat(self._metadata_codes[source_selector], n_repeats))
        self._value_codes = np.append(self._value_codes, np.repeat(self._value_codes[source_selector], n_repeats))
        self._run_id_codes = np.append(self._run_id_codes, np.repeat(self._run_id_codes[source_selector], n_repeats))

    def _csv_to_memmap(self, chunk_size=None):
        """Convert CSV to memmap, storing header."""
        if chunk_size is None:
            chunk_size = self.chunk_size
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
            
    def save_csv_streaming(self, name, chunk_size=None):
        if chunk_size is None:
            chunk_size = self.chunk_size
        with open(name + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.header)  # write header

            for i in tqdm(range(0, self.table.shape[0], chunk_size), desc="Saving Large csv file ...", leave=False):
                chunk = self.table[i:i+chunk_size]
                writer.writerows(chunk)

    def run_ind(self, indices):
        return self._run_group_vocab[self.run_indices[np.array(indices)]]

    def version(self, indices):
        return self._melts_version_vocab[self.MELTSversion[np.array(indices)]]

    def ID(self, code):
        return np.where(self.run_indices == code)[0] # Boolean for clarity
        #return np.arange(len(self.metadata))[self.run_indices == code]

    def run_code_to_label(self, codes):
        """Decode run_indices code(s) (e.g. from np.unique(self.run_indices) or
        whatever ID() was called with) back to the human-readable 'value:run_id'
        string(s) - run_indices itself holds compact int codes, not strings, so a
        raw code isn't meaningful on its own in diagnostics/log messages."""
        return self._run_group_vocab[np.asarray(codes)]

    def _chunked_copy_into(self, src, dst, dst_offset=0, chunk_size=None):
        """Copy every row of `src` into `dst` starting at `dst_offset`, one
        contiguous row-chunk at a time with a flush per chunk - avoids a single
        huge one-shot assignment that leaves the whole copy as dirty pages
        before any of it reaches disk."""
        if chunk_size is None:
            chunk_size = self.chunk_size
        for start in range(0, src.shape[0], chunk_size):
            end = min(start + chunk_size, src.shape[0])
            dst[dst_offset + start:dst_offset + end] = src[start:end]
            dst.flush()

    def _chunked_mask_copy(self, src, dst_path, keep_mask, chunk_size=None):
        """Thin wrapper around the shared `chunked_mask_copy` helper (see
        ngibbs.utils.file_utils), defaulting chunk_size from this instance."""
        if chunk_size is None:
            chunk_size = self.chunk_size
        return chunked_mask_copy(src, dst_path, keep_mask, chunk_size)

    def delete(self, indices, save_text = True, chunk_size = None):
        """Efficient delete that preserves memory-mapping by rewriting, one
        row-chunk at a time via `_chunked_mask_copy` instead of materializing
        every kept row through `self.table[keep_mask]` at once."""
        if chunk_size is None:
            chunk_size = self.chunk_size
        keep_mask = np.ones(self.table.shape[0], dtype=bool)
        keep_mask[indices] = False

        # Apply operation to blurred binary data first to call assertion before we get too deep.
        if self.blurredbinaries is not None:
            binary_name = self.filename + 'blurredbinaries.npy'
            if os.path.exists(binary_name):
                temp_binary_filename = self.filename + 'blurredbinaries_temp.npy'
            else:
                temp_binary_filename = binary_name

            assert np.shape(self.blurredbinaries)[0] == np.shape(self.table)[0], "Shape of binary and main tables are not equal!"
            self._chunked_mask_copy(self.blurredbinaries, temp_binary_filename, keep_mask, chunk_size)

            # Clear memory mapping and update file
            del self.blurredbinaries
            gc.collect()

            if os.path.exists(self.filename + 'blurredbinaries_temp.npy'):
                os.replace(temp_binary_filename, binary_name)

            # Update reference to new memmap in read mode
            self.blurredbinaries = np.load(binary_name, mmap_mode='r+')

        # Derivative sidecars follow the identical mask.
        _sidecar.apply_mask(self, keep_mask, chunk_size)

        # Prepare output memmap
        temp_filename = self.filename + '_temp.npy'
        self._chunked_mask_copy(self.table, temp_filename, keep_mask, chunk_size)

        # Clear memory mapping and update file
        del self.table
        gc.collect()
        os.replace(temp_filename, self.memmap_file)

        # Update reference to new memmap
        self.table = np.load(self.memmap_file, mmap_mode='r+')

        # Update metadata arrays (all in RAM)
        if save_text:
            self._filter_metadata_rows(keep_mask)
            self.save_txt()

    def save(self, name=None, save_csv = True):
        if name is None:
            name = self.filename + '_filtered'
        np.save(name + '.npy', self.table)
        if self.blurredbinaries is not None:
            np.save(name + 'blurredbinaries.npy', self.blurredbinaries)
        _sidecar.save(self, name)
        if save_csv:
            self.save_csv_streaming(name = name)
        with open(name + '.txt', 'w') as f:
            self.write_meta_lines(f)

    def save_txt(self, name=None):
        if name is None:
            name = self.filename + '.txt'
        if '.txt' not in name:
            name += '.txt'
        with open(name, 'w') as f:
            self.write_meta_lines(f)
                
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

        new_filename = f"{self.filename}_split"
        new_path = f"{new_filename}.npy"
        remaining_filename = f"{self.filename}_remaining"

        # Write moved data, one row-chunk at a time (see _chunked_mask_copy)
        move_mask = ~keep_mask
        self._chunked_mask_copy(self.table, new_path, move_mask)

        #...and moved text
        with open(new_filename+'.txt', 'w') as f:
            self.write_meta_lines(f, indices=move_indices)

        # Save headers for new table

        new_header_file = new_path.replace('.npy', '_headers.pkl')
        with open(new_header_file, 'wb') as f:
            pickle.dump(self.header, f)

        # Write kept data
        remaining_data_path = remaining_filename + '.npy'
        self._chunked_mask_copy(self.table, remaining_data_path, keep_mask)

        # Save headers for remaining table
        remaining_header_file = remaining_filename+'_headers.pkl'
        with open(remaining_header_file, 'wb') as f:
            pickle.dump(self.header, f)

        #...and kept text
        self._filter_metadata_rows(keep_indices)

        # Now clear memmaps, reinitialize new memmap, and replace memmaps for this object
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

        # Derivative sidecars: moved rows onto the new table, kept rows filtered here.
        _sidecar.split_into(self, new_table, move_mask, keep_mask, new_filename,
                            self.chunk_size)
        
        # Move BlurredBinaries, one row-chunk at a time (see _chunked_mask_copy)
        if self.blurredbinaries is not None:
            self._chunked_mask_copy(self.blurredbinaries, new_filename+'blurredbinaries.npy', move_mask)
            new_table.blurredbinaries = np.load(new_filename+'blurredbinaries.npy', mmap_mode = 'r')
            assert np.shape(new_table.blurredbinaries)[0] == np.shape(new_table.table)[0], "Shape of split binary and main tables are not equal!"
            print(f"First Row blurredbinaries split Memmap read mode:{new_table.blurredbinaries[0]}")

            #Update remaining blurredboundaries object
            binary_name = self.filename + 'blurredbinaries.npy'
            if os.path.exists(binary_name):
                temp_binary_filename = self.filename + 'blurredbinaries_temp.npy'
            else:
                temp_binary_filename = binary_name

            self._chunked_mask_copy(self.blurredbinaries, temp_binary_filename, keep_mask)

            # Clear memory mapping and update file
            del self.blurredbinaries
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
        move_mask = ~keep_mask

        new_filename = f"{self.filename}_split"
        new_path = f"{new_filename}.npy"
        remaining_filename = f"{self.filename}_remaining"

        # Write moved data, one row-chunk at a time (see _chunked_mask_copy)
        self._chunked_mask_copy(self.table, new_path, move_mask)

        #...and moved text
        with open(new_filename+'.txt', 'w') as f:
            self.write_meta_lines(f, indices=sep_idx)

        # Save headers for new table

        new_header_file = new_path.replace('.npy', '_headers.pkl')
        with open(new_header_file, 'wb') as f:
            pickle.dump(self.header, f)

        # Write kept data
        remaining_data_path = remaining_filename + '.npy'
        self._chunked_mask_copy(self.table, remaining_data_path, keep_mask)

        #...and kept text
        self._filter_metadata_rows(keep_indices)

        # Now clear memmaps, reinitialize new memmap, and replace memmaps for this object
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

        # Derivative sidecars: moved rows onto the new table, kept rows filtered here.
        _sidecar.split_into(self, new_table, move_mask, keep_mask, new_filename,
                            self.chunk_size)
        
        # Move BlurredBinaries, one row-chunk at a time (see _chunked_mask_copy)
        if self.blurredbinaries is not None:
            self._chunked_mask_copy(self.blurredbinaries, new_filename+'blurredbinaries.npy', move_mask)
            new_table.blurredbinaries = np.load(new_filename+'blurredbinaries.npy', mmap_mode = 'r')
            assert np.shape(new_table.blurredbinaries)[0] == np.shape(new_table.table)[0], "Shape of split binary and main tables are not equal!"
            print(f"First Row blurredbinaries split Memmap read mode:{new_table.blurredbinaries[0]}")

            #Update remaining blurredboundaries object
            binary_name = self.filename + 'blurredbinaries.npy'
            if os.path.exists(binary_name):
                temp_binary_filename = self.filename + 'blurredbinaries_temp.npy'
            else:
                temp_binary_filename = binary_name

            self._chunked_mask_copy(self.blurredbinaries, temp_binary_filename, keep_mask)

            # Clear memory mapping and update file
            del self.blurredbinaries
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


    def filter_min_phase_proportion(self, min_proportion, chunk_size=None):
        """
        Delete all rows that contain phases whose presence falls below a minimum proportion.

        Parameters:
        - min_proportion (float): Minimum fraction of rows in which a phase must appear to be kept.
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        if not isinstance(min_proportion, (float, int)):
            raise TypeError("min_proportion must be a float between 0 and 1.")
        if not (0.0 < float(min_proportion) <= 1.0):
            raise ValueError("min_proportion must be a float between 0 and 1 (exclusive of 0).")

        min_proportion = float(min_proportion)
        total_rows = self.table.shape[0]
        if total_rows == 0:
            print("No rows available; skipping phase proportion filtering.")
            return

        # Resolve each phase's mass column once, up front.
        phase_mass_cols = []
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
            phase_mass_cols.append((phase, mass_col))

        # Single row-chunked pass building a compact (rows x phases) presence matrix,
        # instead of reading each phase's mass column across every row separately -
        # every phase's presence is then just an in-RAM column of this matrix.
        mass_col_indices = np.array([c for _, c in phase_mass_cols])
        presence = np.zeros((total_rows, len(phase_mass_cols)), dtype=bool)
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            presence[start:end] = self.table[start:end][:, mass_col_indices] > 0

        below_threshold = []
        phases_removed = []
        for j, (phase, mass_col) in enumerate(phase_mass_cols):
            phase_count = int(presence[:, j].sum())
            phase_proportion = phase_count / total_rows

            if phase_proportion < min_proportion:
                below_threshold.append(j)
                phases_removed.append(phase)
                print(
                    f"Phase '{phase}' proportion {phase_proportion:.6f} below {min_proportion:.6f}; "
                    f"marking {phase_count} rows for deletion."
                )
            else:
                print(
                    f"Phase '{phase}' proportion {phase_proportion:.6f} meets threshold {min_proportion:.6f}; keeping."
                )

        if below_threshold:
            delete_mask = presence[:, below_threshold].any(axis=1)
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
        self.indexer.table_update(self.table, tolerance=tolerance, chunk_size=self.chunk_size)
        return removed
    
    def filter_phases_not_in_ml_indexer(self, chunk_size=None):
        """
        Delete all rows that contain phases not present in self.indexer.ml_indexer.all_phases.
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
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

        # Resolve the (usually few) disallowed phases' mass columns once, up front.
        disallowed_mass_cols = []
        for phase, components in self.indexer.MELTS_indices.items():
            if phase in non_phase_keys or phase in allowed_phases:
                continue
            mass_col = components.get('liq mass (gm)') if phase == 'melts-liquid' else components.get('mass (gm)')
            if mass_col is None:
                print(f"Skipping phase '{phase}': no mass column found.")
                continue
            disallowed_mass_cols.append((phase, mass_col))

        if not disallowed_mass_cols:
            print("No rows matched phases outside ml_indexer; no rows deleted.")
            return

        # Single row-chunked pass building a compact (rows x disallowed phases)
        # presence matrix, instead of reading each phase's mass column across every
        # row separately.
        mass_col_indices = np.array([c for _, c in disallowed_mass_cols])
        presence = np.zeros((total_rows, len(disallowed_mass_cols)), dtype=bool)
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            presence[start:end] = self.table[start:end][:, mass_col_indices] > 0

        keep_cols = []
        phases_removed = []
        for j, (phase, mass_col) in enumerate(disallowed_mass_cols):
            phase_count = int(presence[:, j].sum())
            if phase_count == 0:
                print(f"Phase '{phase}' not in ml_indexer; no rows contain it.")
                continue
            keep_cols.append(j)
            phases_removed.append(phase)
            print(
                f"Phase '{phase}' not in ml_indexer; marking {phase_count} rows for deletion."
            )

        if keep_cols:
            delete_mask = presence[:, keep_cols].any(axis=1)
            indices_to_delete = np.where(delete_mask)[0]
            print(
                f"Deleting {len(indices_to_delete)} rows from {len(phases_removed)} non-ml phases: "
                f"{phases_removed}"
            )
            self.delete(indices_to_delete)
        else:
            print("No rows matched phases outside ml_indexer; no rows deleted.")

    def report_phase_abundances(self, label=None, chunk_size=None):
        """
        Print the non-zero-mass row count and percentage for every phase, in one
        row-chunked pass over the table rather than one full-column read per phase.

        Parameters:
        - label (str, optional): Included in the printed header, e.g. "Raw" or
          "After Filtering", to distinguish reports taken at different pipeline stages.
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        if not hasattr(self.indexer, 'ml_indexer'):
            raise AttributeError("DatasetIndexer has no ml_indexer attached.")

        header = "Phase Abundance" if label is None else f"Phase Abundance ({label})"
        print(f"\n################ {header} ####################")

        total_rows = self.table.shape[0]
        if total_rows == 0:
            print("No rows available.")
            print("########################################################################\n")
            return

        phase_mass_cols = []
        for phase in self.indexer.ml_indexer.all_phases:
            if phase not in self.indexer.MELTS_indices:
                continue
            col_idx = self.indexer.MELTS_indices[phase].get('mass (gm)')
            if col_idx is not None:
                phase_mass_cols.append((phase, col_idx))

        if not phase_mass_cols:
            print("No phases with a 'mass (gm)' column found.")
            print("########################################################################\n")
            return

        mass_col_indices = np.array([c for _, c in phase_mass_cols])
        counts = np.zeros(len(phase_mass_cols), dtype=np.int64)
        n_chunks = int(np.ceil(total_rows / chunk_size))
        _log_mem(f"report_phase_abundances('{label}') start, {n_chunks} chunks over {total_rows:,} rows")
        for chunk_i, start in enumerate(range(0, total_rows, chunk_size)):
            end = min(start + chunk_size, total_rows)
            counts += np.sum(self.table[start:end][:, mass_col_indices] > 0, axis=0)
            # Periodic checkpoint (not every chunk, to avoid flooding stdout) - lets a
            # crash mid-scan be localized to "memory was already climbing steadily"
            # (WSL2 page-cache buildup from the full-table sequential read) versus
            # "memory jumped once" (a one-shot allocation elsewhere).
            if chunk_i % 100 == 0:
                _log_mem(f"report_phase_abundances('{label}') chunk {chunk_i}/{n_chunks}")

        for j, (phase, col_idx) in enumerate(phase_mass_cols):
            non_zero_count = int(counts[j])
            total_pct = 100 * non_zero_count / total_rows
            print(f"{phase:20} {non_zero_count:>10,} samples ({total_pct:>6.2f}%)")
        print("########################################################################\n")
        _log_mem(f"report_phase_abundances('{label}') done")

    def filter_inconsistent_phase_data(self, chunk_size=None):
        """
        Delete rows with inconsistent phase data: mass = 0 but non-zero attributes.

        Checks all phases in ml_indexer and removes rows where a phase has zero mass
        but has non-zero values in any of its other attributes (composition, etc.).
        This indicates corrupted or inconsistent data.
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
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

        # Resolve each phase's (mass_col, other_cols) once up front - this used to be
        # interleaved with a full-table column read per phase (self.table[:, mass_col],
        # then self.table[zero_mass_mask] materializing every zero-mass row's full width
        # for that phase alone), so a table with N phases paid for N separate full-table
        # passes. Below, a single row-chunked pass checks every phase against each chunk,
        # so the whole table is only read once regardless of phase count.
        phase_cols = []
        for phase in self.indexer.ml_indexer.all_phases:
            if phase in non_phase_keys:
                continue
            if phase not in self.indexer.MELTS_indices:
                continue
            components = self.indexer.MELTS_indices[phase]
            if phase == 'melts-liquid':
                mass_col = components.get('liq mass (gm)')
            else:
                mass_col = components.get('mass (gm)')
            if mass_col is None:
                mass_col = components.get('mass(gm)')
            if mass_col is None:
                continue
            other_cols = [col_idx for key, col_idx in components.items()
                         if col_idx != mass_col]
            if not other_cols:
                continue
            phase_cols.append((phase, mass_col, other_cols))

        delete_mask = np.zeros(total_rows, dtype=bool)
        inconsistent_phases = {}

        for start in tqdm(range(0, total_rows, chunk_size), desc="Checking phase consistency"):
            end = min(start + chunk_size, total_rows)
            chunk = self.table[start:end]

            for phase, mass_col, other_cols in phase_cols:
                zero_mass_mask = chunk[:, mass_col] == 0
                if not zero_mass_mask.any():
                    continue

                other_data = chunk[zero_mass_mask][:, other_cols]
                has_nonzero_attrs = np.any(other_data != 0, axis=1)
                if not has_nonzero_attrs.any():
                    continue

                zero_mass_indices = np.where(zero_mass_mask)[0]
                inconsistent_indices = zero_mass_indices[has_nonzero_attrs]
                delete_mask[start + inconsistent_indices] = True
                inconsistent_phases[phase] = inconsistent_phases.get(phase, 0) + len(inconsistent_indices)

            # Checked per chunk (covering every phase seen so far) rather than per phase
            # across the whole table, since phases are now the inner loop - still catches
            # the same "data is overwhelmingly broken" condition.
            assert delete_mask.sum()/total_rows < 0.03, f'Too many ({100*delete_mask.sum()/total_rows} %) broken assemblages... What is going on? '

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
            
            
    
    def filter_full_metadata(self):
        """Deletes rows where metadata contain unsupported phases."""
        # Classify each DISTINCT vocab entry once (typically a handful to a few
        # hundred), then broadcast that classification across all rows via the
        # int32 code array - instead of a 90M-row Python loop over materialized
        # per-row strings (both much slower and, before vocab-coding, required
        # materializing the whole `self.metadata` array first).
        if len(self._metadata_vocab):
            vocab_is_digit = np.array(
                [str(entry).strip().isdigit() for entry in self._metadata_vocab]
            )
            mask_keep = vocab_is_digit[self._metadata_codes]
        else:
            mask_keep = np.ones(self.table.shape[0], dtype=bool)

        # Delete where mask is False
        if np.sum(~mask_keep):
            print(f"Deleted {np.sum(~mask_keep)} entries for unsupported phases")
            indices_to_delete = np.where(~mask_keep)[0]
            self.delete(indices_to_delete)

    def resample_rare_phase(self, phase_column, multiplier_bounds, n_resamples, overwrite = False, chunk_size = None):
        """
        Resamples entries that contain a rare phase (nonzero in `phase_column`).

        Parameters:
        - phase_column (int): Index of the column indicating presence of the rare phase.
        - multiplier_bounds (tuple/list of two floats): (min, max) range for random multipliers.
        - n_resamples (int): Number of times to replicate/resample each rare-phase entry.

        Appends the resampled entries (with perturbed mass columns) to the table.

        NOTE ON NAMING: despite the name (and the YAML config it's driven from being
        called "upsampling"/"rare phase" resampling), this method does not itself
        check that the phase passed in is actually rare - it resamples whatever
        phase_column is given. If a phase present in a large fraction of rows is
        passed (e.g. olivine in a mostly-mafic dataset), the number of matching
        rows can be comparable to the whole table. Everything below is written
        chunk-wise so that case stays memory-bounded instead of materializing an
        "all matching rows" array that can be a large fraction of the whole table.
        """
        assert_identity_multipliers(multiplier_bounds, 'resample_rare_phase')
        if chunk_size is None:
            chunk_size = self.chunk_size
        min_multiplier, max_multiplier = multiplier_bounds
        if not isinstance(n_resamples, int) or n_resamples < 1:
            raise ValueError("n_resamples must be a positive integer.")

        old_rows, n_cols = self.table.shape

        # --- Pass 1: count matching rows only, one chunk (single column) at a
        # time, so the output memmap can be allocated at its final size up front.
        # This never holds more than `chunk_size` rows of a single column in RAM -
        # contrast with the previous `self.table[:, phase_column] > 0` (a
        # full-table column read) followed by `self.table[rare_mask]`, which
        # materialized every matching row (all columns) as one in-RAM array. That
        # was fine when "rare" really meant rare, but blew up memory for a common
        # phase like olivine, where matching rows can be tens of millions.
        n_rare = 0
        for start in range(0, old_rows, chunk_size):
            end = min(start + chunk_size, old_rows)
            n_rare += int(np.count_nonzero(self.table[start:end, phase_column] > 0))

        if n_rare == 0:
            print("[INFO] No rare-phase entries found. No resampling performed.")
            return

        #Define Sizings
        n_masscols = self.indexer.nphases
        mass_indices = self.indexer.mass_indices
        total_new = n_rare * n_resamples
        new_total = old_rows + total_new

        has_binaries = self.blurredbinaries is not None
        if has_binaries:
            assert np.shape(self.blurredbinaries)[0] == np.shape(self.table)[0], 'Shape mismatch between main table and binarylabel table!'
            old_binary_filename = self.blurredbinaries.filename
            new_binary_path = f"{old_binary_filename.split('.')[0]}_resampledblurredbinaries.npy"
            new_binary_table = np.lib.format.open_memmap(
                new_binary_path, mode='w+', dtype=self.blurredbinaries.dtype, shape=(new_total, self.blurredbinaries.shape[1])
            )
            self._chunked_copy_into(self.blurredbinaries, new_binary_table, chunk_size=chunk_size)

        # Expand underlying memmap
        new_path = f"{self.table.filename.split('.')[0]}_resampled.npy"
        old_path = self.table.filename
        new_table = np.lib.format.open_memmap(
            new_path, mode='w+', dtype=self.table.dtype, shape=(new_total, n_cols)
        )
        self._chunked_copy_into(self.table, new_table, chunk_size=chunk_size)

        # --- Pass 2: walk the table again one chunk at a time. For each chunk,
        # pull out just its matching rows (bounded by chunk_size, not by how
        # common the phase is), repeat + perturb + renormalize that small piece,
        # and write it straight into the new memmap. Peak RAM added by this loop
        # is O(chunk_size * n_resamples), never O(n_rare). The per-chunk mask is
        # reused for both the main table and blurred binaries so the two stay
        # row-aligned, and accumulated to reconstruct the full `rare_mask` needed
        # by `_append_metadata_rows` at the end (cheap: booleans, same size as
        # the old table's row count either way).
        rare_mask_chunks = []
        repeat_plan = []          # (start, end, chunk_mask, n_resamples) for the sidecars
        write_ptr = old_rows
        for start in range(0, old_rows, chunk_size):
            end = min(start + chunk_size, old_rows)
            table_chunk = self.table[start:end]
            chunk_mask = table_chunk[:, phase_column] > 0
            rare_mask_chunks.append(chunk_mask)
            repeat_plan.append((start, end, chunk_mask, n_resamples))
            if not chunk_mask.any():
                continue

            rare_chunk = table_chunk[chunk_mask]
            resampled_chunk = np.repeat(rare_chunk, n_resamples, axis=0)

            multipliers = np.random.uniform(
                low=min_multiplier, high=max_multiplier,
                size=(resampled_chunk.shape[0], n_masscols)
            )
            resampled_chunk[:, mass_indices] = resampled_chunk[:, mass_indices] * multipliers
            totals = np.sum(resampled_chunk[:, mass_indices], axis=1).reshape((-1, 1))
            resampled_chunk[:, mass_indices] = resampled_chunk[:, mass_indices] * 100 / totals

            n = resampled_chunk.shape[0]
            new_table[write_ptr:write_ptr + n] = resampled_chunk

            if has_binaries:
                rare_binary_chunk = self.blurredbinaries[start:end][chunk_mask]
                resampled_binary_chunk = np.repeat(rare_binary_chunk, n_resamples, axis=0)
                new_binary_table[write_ptr:write_ptr + n] = resampled_binary_chunk

            write_ptr += n

        assert write_ptr == new_total, f"Resampled row count mismatch: wrote {write_ptr}, expected {new_total}"

        new_table.flush()
        rare_mask = np.concatenate(rare_mask_chunks)

        # Sidecars grow by duplicating the same rows. Multipliers are guarded to
        # [1, 1], so a duplicated row's derivatives are simply the originals.
        _sidecar.grow_with_repeats(self, new_total, old_rows, repeat_plan, chunk_size)

        del self.table, new_table
        if has_binaries:
            new_binary_table.flush()
            del new_binary_table, self.blurredbinaries
        gc.collect()

        # Update self in-place
        if overwrite:
            os.replace(new_path, old_path)
            self.table = np.load(old_path, mmap_mode='r+')
        else:
            self.table = np.load(new_path, mmap_mode='r+')

        if has_binaries:
            if overwrite:
                os.replace(new_binary_path, old_binary_filename)
                self.blurredbinaries = np.load(old_binary_filename, mmap_mode='r+')
            else:
                self.blurredbinaries = np.load(new_binary_path, mmap_mode='r+')

        self._append_metadata_rows(rare_mask, n_resamples)
        
    def separate_analcime(self):
        assert_no_derivative_sidecars(self, 'separate_analcime')
        
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
        assert_no_derivative_sidecars(self, 'separate_k_feldspar')
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

        
    def retrieve_component_moles(self, multiplier_bounds=[1, 1], chunk_size=None):
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
        assert_identity_multipliers(multiplier_bounds, 'retrieve_component_moles')
        if chunk_size is None:
            chunk_size = self.chunk_size
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

        # Molar-space twins of the derivative sidecars. Multipliers are guarded to 1,
        # so these carry the same linear selection the moles do; if that guard is ever
        # lifted the same per-row factor must be applied to both, since n and dn/dP are
        # each homogeneous of degree 1 in the bulk amount.
        self.dmolar = {}
        for _attr, _suffix, _arr in _sidecar.items(self):
            if _arr.shape[0] != total_rows:
                raise AssertionError(
                    f'retrieve_component_moles: sidecar {_attr} has {_arr.shape[0]} rows '
                    f'against table1 {total_rows}')
            self.dmolar[_attr] = np.lib.format.open_memmap(
                self.filename + f'component_moles_{_attr}.npy',
                mode='w+', dtype=np.float32, shape=(total_rows, num_cols))

        phases = list(label_indices.keys())

        # ========== Process one row-chunk at a time, every phase within it ==========
        # Each phase used to do its own full-height column read from self.table1 and
        # full-height write into self.molar - a strided pass over every row, repeated
        # once per phase. Looping chunk-outer/phase-inner instead means self.table1 is
        # read contiguously exactly once (per chunk) and self.molar written once, no
        # matter how many phases there are.
        for start in tqdm(range(0, total_rows, chunk_size), desc="Computing component moles"):
            end = min(start + chunk_size, total_rows)
            chunk = self.table1[start:end]
            chunk_len = end - start
            molar_chunk = np.zeros((chunk_len, num_cols), dtype=np.float32)

            for phase in phases:
                # Generate random multipliers for resampling (1.0 for no resampling)
                phase_multipliers = np.random.uniform(
                    *multiplier_bounds,
                    size=chunk_len
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

                        component_moles = chunk[:, component_col_indices]

                        molar_chunk[:, phase_label_inds] = component_moles * phase_multipliers

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
                            component_fractions = chunk[:, component_col_indices]

                            # Calculate phase molar mass from composition
                            # component_fractions -> oxides -> molar mass
                            phase_molar_mass = (
                                component_fractions @ compToOxLoad[phase_label_inds]
                            ) @ Mtot

                            # Get phase mass (grams) from table
                            phase_mass_grams = chunk[:, component_indices[phase]['mass (gm)']]
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
                            molar_chunk[:, phase_label_inds] = (
                                component_fractions *
                                phase_multipliers *
                                total_phase_moles
                            )

                        else:
                            # Invariant composition solid (e.g., quartz = 100% SiO2)
                            # Only one component, so moles = mass / molar_mass

                            phase_mass_grams = chunk[:, component_indices[phase]['mass (gm)']]
                            phase_mass_col = phase_mass_grams.reshape(-1, 1)

                            # Get molar mass of this single component
                            phase_molar_mass = compToOxLoad[phase_label_inds, :] @ Mtot

                            # Calculate moles: mass / MM, with multiplier
                            molar_chunk[:, phase_label_inds] = (
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
                    wt_percent_oxides = chunk[:, oxide_col_indices]

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
                    liquid_mass_grams = chunk[:, component_indices[phase]['liq mass (gm)']]
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
                    molar_chunk[:, label_indices[phase]] = (
                        mole_fraction_oxides *
                        phase_multipliers *
                        total_liquid_moles
                    ) @ OxToEl

            self.molar[start:end] = molar_chunk
            self.molar.flush()

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

        # Iterate the small int32 code array and look up each row's string from the
        # (small) vocab on demand, rather than enumerate(self.metadata) - the latter
        # would materialize every row's metadata string into one giant fixed-width
        # array before the loop even starts.
        for i, _code in enumerate(self._metadata_codes):
            meta        = self._metadata_vocab[_code]
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

        # See recover_untracked_phases: iterate codes + small vocab, not the
        # materialized self.metadata array.
        for i, _code in enumerate(self._metadata_codes):
            meta = self._metadata_vocab[_code]
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
