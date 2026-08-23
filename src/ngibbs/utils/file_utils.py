"""
File operation utilities.

Extracted from Legacy/BackEnds/EmulatorLibrary.py
"""

import os
import shutil
import gc
import warnings
from pathlib import Path
import json
import tarfile
import tempfile
import pickle
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Repo root, used to keep temp directories on disk (not a RAM-backed /tmp)
_repo_root = Path(__file__).resolve().parents[3]


def chunked_mask_copy(src, dst_path, keep_mask, chunk_size=100_000):
    """Copy the rows of `src` selected by boolean `keep_mask` into a new
    memmap at `dst_path`, one row-chunk at a time.

    `src[start:end]` and the write into `dst` both stay contiguous; only the
    boolean sub-select of a chunk is fancy-indexed, so peak RAM is
    O(chunk_size) instead of materializing every selected row at once via
    `src[keep_mask]` (or, equivalently, `np.vstack` over a list of kept
    batches) - either of which holds the entire surviving array in RAM before
    anything is written back out.

    Shared by BigMetaTable.delete/split/manual_split (row-major .npy tables)
    and filters.safe_delete_batched (standalone ML-bundle .npy arrays with no
    live BigMetaTable around) so there is one implementation of "delete rows
    from a memmap" rather than several with diverging memory behavior.

    Returns the number of rows written (`n_keep`).
    """
    n_keep = int(np.count_nonzero(keep_mask))
    dst = np.lib.format.open_memmap(dst_path, mode='w+', dtype=src.dtype, shape=(n_keep,) + src.shape[1:])
    write_ptr = 0
    for start in range(0, src.shape[0], chunk_size):
        end = min(start + chunk_size, src.shape[0])
        chunk_mask = keep_mask[start:end]
        if not chunk_mask.any():
            continue
        kept = src[start:end][chunk_mask]
        n = kept.shape[0]
        dst[write_ptr:write_ptr + n] = kept
        write_ptr += n
    dst.flush()
    del dst
    gc.collect()
    return n_keep


def chunked_permutation_copy(src, dst_path, permutation, chunk_size=100_000):
    """Write `dst[i] = src[permutation[i]]` for every row, one output row-chunk
    at a time.

    The *write* into `dst` is always a contiguous chunk; the *read* from `src`
    is a scattered fancy-index gather (`src[permutation[start:end]]`), so this
    is only as fast as random access to `src` allows - but peak RAM is bounded
    to O(chunk_size) regardless of how large `src`/`permutation` are, since only
    one chunk's worth of gathered rows exists at a time. `permutation` itself
    (a single (n,) int array) is expected to already be in RAM - cheap even at
    tens of millions of rows, unlike the row data itself.

    This is the one-time "shuffle a table's row order" primitive: pay a
    scattered-read cost once during processing so that later, repeated
    sequential/chunked reads (e.g. a chunked training data loader) never need
    to worry about the underlying row order carrying any structure (such as
    upsampled rows appended as a block at the end of a table).
    """
    n = src.shape[0]
    dst = np.lib.format.open_memmap(dst_path, mode='w+', dtype=src.dtype, shape=(n,) + src.shape[1:])
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        dst[start:end] = src[permutation[start:end]]
    dst.flush()
    del dst
    gc.collect()


def delete_files_with_keyword(directory, keyword, dry_run=True):
    """
    Deletes all files in `directory` whose names contain `keyword`.

    Args:
        directory (str): Path to the target directory.
        keyword (str): Keyword to match in filenames.
        dry_run (bool): If True, only print files that would be deleted.

    Example:
        delete_files_with_keyword("/path/to/folder", "temp", dry_run=False)
    """
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        return

    deleted_count = 0
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and keyword in filename:
            if dry_run:
                print(f"[DRY RUN] Would delete: {filepath}")
            else:
                try:
                    os.remove(filepath)
                    print(f"Deleted: {filepath}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {filepath}: {e}")

    if dry_run:
        print("\nDry run complete. No files were actually deleted.")
    else:
        print(f"\nDeleted {deleted_count} files containing '{keyword}'.")
        

def move_file(src_filename, dst_dir, overwrite=False):
    """
    Move a file from the current working directory to a destination directory.

    Args:
        src_filename (str): Name of the file in the current working directory.
        dst_dir (str): Destination directory path.
        overwrite (bool): If True, overwrite any existing file with the same name.
    """
    # Ensure the source file exists
    src_path = os.path.join(os.getcwd(), src_filename)
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"Source file not found: {src_path}")

    # Ensure destination directory exists
    #if not os.path.exists(dst_dir):
    #   os.makedirs(dst_dir)

    # Destination file path
    dst_path = os.path.join(dst_dir, src_filename)

    # Handle overwrite
    if os.path.exists(dst_path):
        if overwrite:
            os.remove(dst_path)
        else:
            raise FileExistsError(f"Destination file already exists: {dst_path}")

    # Move the file
    shutil.move(src_path, dst_path)
    print(f"Moved '{src_filename}' to '{dst_dir}' successfully.")

def move_files_with_extension(extension, dst_dir, src_dir=None, overwrite=False):
    """
    Move all files with a given extension or filename ending from a directory.

    Args:
        extension (str): File extension (e.g., '.csv') or filename ending (e.g., 'labels.npy')
        dst_dir (str or Path): Destination directory path (absolute or relative)
        src_dir (str or Path): Source directory path (absolute or relative). Default is cwd.
        overwrite (bool): If True, overwrite existing files in destination
    """
    # Handle source directory - accept both absolute and relative paths
    if src_dir is not None:
        src_path = Path(src_dir)
        if not src_path.is_absolute():
            src_path = Path.cwd() / src_dir
    else:
        src_path = Path.cwd()
    
    # Handle destination directory
    dst_dir = Path(dst_dir)
    if not dst_dir.is_absolute():
        dst_dir = Path.cwd() / dst_dir
    
    # Don't move if source and destination are the same
    if src_path.resolve() == dst_dir.resolve():
        return
    
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Find matching files - check both suffix and filename ending
    if extension.startswith('.'):
        # Traditional extension match (e.g., '.csv', '.npy')
        files_to_move = [f for f in src_path.iterdir() if f.is_file() and f.suffix == extension]
    else:
        # Filename ending match (e.g., 'molar_labels.npy', '_processed.txt')
        files_to_move = [f for f in src_path.iterdir() if f.is_file() and f.name.endswith(extension)]

    if not files_to_move:
        print(f"No files matching '{extension}' found in {src_path}.")
        return

    moved_count = 0
    for file_path in files_to_move:
        dest_path = dst_dir / file_path.name

        if dest_path.exists():
            if overwrite:
                dest_path.unlink()  # Remove existing file
            else:
                print(f"Skipping {file_path.name}, already exists in destination.")
                continue

        shutil.move(str(file_path), str(dest_path))
        moved_count += 1
    
    print(f"Moved {moved_count} file(s) matching '{extension}' from {src_path} -> {dst_dir}")

def move_files_with_keyword(keyword, dst_dir, src_dir=None, overwrite=False):
    """
    Move all files whose names contain a given keyword from the source directory
    (or current working directory) to the destination directory.

    Args:
        keyword (str): Substring to search for in filenames.
        dst_dir (str or Path): Destination directory path.
        src_dir (str or Path, optional): Source directory path. Defaults to current working directory.
        overwrite (bool): If True, overwrite existing files in destination.
    """
    src_path = Path(src_dir) if src_dir else Path.cwd()
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    # Find all files containing the keyword in their filename
    files_to_move = [f for f in src_path.iterdir() if f.is_file() and keyword in f.name]

    if not files_to_move:
        print(f"No files containing '{keyword}' found in {src_path}.")
        return

    for file_path in files_to_move:
        dest_file = dst_path / file_path.name

        if dest_file.exists() and not overwrite:
            print(f"Skipping {file_path.name}, already exists in destination.")
            continue

        if dest_file.exists() and overwrite:
            dest_file.unlink()

        shutil.move(str(file_path), str(dest_file))
        print(f"Moved {file_path.name} -> {dst_path}")


def count_file_lines(file_path, skip_header=False):
    """
    Efficiently count lines in a text or CSV file without loading into RAM.
    
    For files > 1GB, this is much more memory-efficient than loading the entire file.
    Uses line-by-line reading which minimizes memory usage.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the file to count lines in
    skip_header : bool, default=False
        If True, skip the first line (useful for CSV files with headers)
        
    Returns
    -------
    int
        Number of lines (excluding header if skip_header=True)
        
    Examples
    --------
    >>> # Count all lines in a text file
    >>> count = count_file_lines('metadata.txt')
    >>> 
    >>> # Count data rows in a CSV (excluding header)
    >>> count = count_file_lines('data.csv', skip_header=True)
    """
    file_path = Path(file_path) if not isinstance(file_path, str) else file_path
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    count = 0
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        if skip_header:
            try:
                next(f)  # Skip header
            except StopIteration:
                return 0  # Empty file
        for _ in f:
            count += 1
    return count


def count_csv_rows(csv_path, has_header=True):
    """
    Count data rows in a CSV file without loading into RAM.
    
    Convenience wrapper around count_file_lines for CSV files.
    
    Parameters
    ----------
    csv_path : str or Path
        Path to CSV file
    has_header : bool, default=True
        Whether the CSV file has a header row
        
    Returns
    -------
    int
        Number of data rows (excluding header if has_header=True)
        
    Examples
    --------
    >>> row_count = count_csv_rows('large_file.csv')
    >>> print(f"CSV has {row_count} data rows")
    """
    return count_file_lines(csv_path, skip_header=has_header)


def get_baseline_files(directory):
    """
    Take a snapshot of all files currently in a directory.
    
    Useful for cleanup operations - capture baseline before work starts,
    then use with clear_new_files() in a finally block to clean up only
    files created during execution.
    
    Parameters
    ----------
    directory : str or Path
        Directory to scan
        
    Returns
    -------
    set
        Set of filenames (just names, not full paths) currently in directory
        
    Examples
    --------
    >>> baseline = get_baseline_files('/path/to/dir')
    >>> try:
    ...     # Do some work that creates files
    ...     process_data()
    ... finally:
    ...     clear_new_files('/path/to/dir', baseline)
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"'{directory}' is not a valid directory")
    
    return {f.name for f in directory.iterdir() if f.is_file()}


def clear_new_files(directory, baseline_files, protected_extensions=None, dry_run=False):
    """
    Delete files that were created after a baseline snapshot.
    
    Compares current files in directory against a baseline set and removes
    only files that are new (not in baseline). Useful for cleanup in 
    try/finally blocks or exception handlers.
    
    Parameters
    ----------
    directory : str or Path
        Directory to scan for new files
    baseline_files : set
        Set of baseline filenames (from get_baseline_files())
        
    dry_run : bool, default=False
        If True, print files that would be deleted but do not delete them

    Returns
    -------
    int
        Number of files deleted (or that would be deleted if dry_run=True)
        
    Examples
    --------
    >>> baseline = get_baseline_files('/path/to/dir')
    >>> try:
    ...     # Do some work that creates files
    ...     process_data()
    ... finally:
    ...     deleted_count = clear_new_files('/path/to/dir', baseline)
    ...     print(f"Cleaned up {deleted_count} temporary files")
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"'{directory}' is not a valid directory")
    
    deleted_count = 0
    current_files = {f.name for f in directory.iterdir() if f.is_file()}
    new_files = current_files - baseline_files
    
    for filename in new_files:
        if protected_extensions and any(filename.endswith(ext) for ext in protected_extensions):
            print(f"Skipping protected file: {filename}")
            continue
        filepath = directory / filename
        if dry_run:
            print(f"[DRY RUN] Would delete temporary file: {filepath}")
            deleted_count += 1
            continue
        try:
            filepath.unlink()
            print(f"Deleted temporary file: {filepath}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {filepath}: {e}")
    
    return deleted_count

# Arrays a bundle may legitimately lack: `free_outputs` was always optional, and the two
# derivative arrays are absent from every bundle exported before derivative support. A
# missing one is reported through `MLDataBundle.has_derivatives`, not raised here -- the
# loud failure belongs in the trainer, where it can name the config key that asked for them.
_OPTIONAL_ARRAYS = ('free_outputs', 'dndp_labels', 'dndt_labels')


class MLDataBundle:
    """Container for ML dataset bundle loaded from .tar.gz file."""
    def __init__(self):
        self.molar_labels = None
        self.binary_labels = None
        self.mass_labels = None
        self.features = None
        self.labels = None
        self.free_outputs = None
        # Composition derivatives on the component (species) axis. None for a bundle
        # exported before derivative support -- `has_derivatives` is the flag training
        # should branch on rather than testing for None in three different places.
        self.dndp_labels = None
        self.dndt_labels = None
        self.derivative_stats = None
        self.ml_indexer = None

    @property
    def has_derivatives(self):
        return self.dndp_labels is not None and self.dndt_labels is not None


def load_ml_bundle(bundle_path, arrays=None):
    """
    Load a .tar.gz bundle created by resampling_to_datasets.

    Extracts all .npy files and the ml_indexer.pkl from the tarball and returns
    them as a MLDataBundle object with attributes for each file.

    Parameters
    ----------
    bundle_path : str or Path
        Path to the .tar.gz bundle file
    arrays : iterable of str, optional
        Which arrays to actually load into RAM (any of 'molar_labels',
        'binary_labels', 'mass_labels', 'features', 'labels', 'free_outputs',
        'dndp_labels', 'dndt_labels').
        Arrays not listed are left as None on the returned bundle and never
        read off disk - use this to skip arrays nothing downstream needs
        (e.g. 'mass_labels' is unused by every current training/inference
        consumer). Default (None) loads every array present, matching the
        original behavior.

    Returns
    -------
    MLDataBundle
        Object with attributes: molar_labels, binary_labels, mass_labels,
        features, labels, free_outputs (if present), ml_indexer
    """
    bundle_path = Path(bundle_path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle file not found: {bundle_path}")

    # Create temporary directory for extraction
    tmp_base = _repo_root / "data" / "tmp"
    tmp_base.mkdir(parents=True, exist_ok=True)
    temp_dir = tempfile.mkdtemp(dir=tmp_base)
    try:
        # Extract tarball
        with tarfile.open(bundle_path, 'r:gz') as tar:
            tar.extractall(path=temp_dir)

        # Create bundle object
        bundle = MLDataBundle()

        # Load .npy files
        npy_files = {
            'molar_labels': 'molar_labels.npy',
            'binary_labels': 'binary_labels.npy',
            'mass_labels': 'mass_labels.npy',
            'features': 'features.npy',
            'labels': 'labels.npy',
            'free_outputs': 'free_outputs.npy',
            'dndp_labels': 'dndp_labels.npy',
            'dndt_labels': 'dndt_labels.npy',
        }
        wanted = set(npy_files) if arrays is None else set(arrays)

        for attr_name, filename in npy_files.items():
            if attr_name not in wanted:
                continue  # not requested - never touched on disk or in RAM
            file_path = Path(temp_dir) / filename
            if file_path.exists():
                setattr(bundle, attr_name, np.load(file_path))
            elif attr_name not in _OPTIONAL_ARRAYS:
                raise FileNotFoundError(f"Expected file not found in bundle: {filename}")

        # Present only in bundles exported with derivative support; absence is normal and
        # is what `bundle.has_derivatives` reports.
        stats_json = Path(temp_dir) / 'derivative_stats.json'
        if stats_json.exists():
            bundle.derivative_stats = json.loads(stats_json.read_text())
        
        # Load ml_indexer state directory (preferred)
        indexer_dir = Path(temp_dir) / 'ml_indexer'
        if indexer_dir.exists():
            from ngibbs.config.ml_indexer import load_ml_indexer_from_state
            bundle.ml_indexer = load_ml_indexer_from_state(indexer_dir)
        else:
            # Backward compatibility: fallback to pickle if present
            indexer_path = Path(temp_dir) / 'ml_indexer.pkl'
            if indexer_path.exists():
                with open(indexer_path, 'rb') as f:
                    bundle.ml_indexer = pickle.load(f)
            else:
                raise FileNotFoundError("ml_indexer state directory or pickle not found in bundle")
        
        return bundle
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


def save_ml_bundle(bundle, output_path):
    """
    Save an MLBundle object to a .tar.gz file.
    
    Saves all numpy arrays, the ml_indexer state, and creates a tar.gz archive.
    Complementary function to load_ml_bundle.
    
    Parameters
    ----------
    bundle : MLDataBundle or dict-like object
        Bundle object with attributes or dict keys:
        - features: np.ndarray
        - binary_labels: np.ndarray
        - mass_labels: np.ndarray
        - molar_labels: np.ndarray
        - labels: np.ndarray
        - ml_indexer: MLIndexer instance
        - free_outputs: np.ndarray (optional)
        
    output_path : str or Path
        Path where to save the .tar.gz bundle file
        
    Examples
    --------
    >>> from nMELTS.utils.file_utils import load_ml_bundle, save_ml_bundle
    >>> bundle = load_ml_bundle('existing_bundle.tar.gz')
    >>> # ... modify bundle data ...
    >>> save_ml_bundle(bundle, 'modified_bundle.tar.gz')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory for staging
    tmp_base = _repo_root / "data" / "tmp"
    tmp_base.mkdir(parents=True, exist_ok=True)
    temp_dir = tempfile.mkdtemp(dir=tmp_base)
    try:
        # Helper to get attribute or dict key
        def get_attr(obj, name, default=None):
            if hasattr(obj, name):
                return getattr(obj, name)
            elif isinstance(obj, dict) and name in obj:
                return obj[name]
            else:
                return default
        
        # Save .npy files
        arrays_to_save = {
            'features': get_attr(bundle, 'features'),
            'binary_labels': get_attr(bundle, 'binary_labels'),
            'mass_labels': get_attr(bundle, 'mass_labels'),
            'molar_labels': get_attr(bundle, 'molar_labels'),
            'labels': get_attr(bundle, 'labels'),
            'free_outputs': get_attr(bundle, 'free_outputs'),  # optional
        }
        
        for filename, array in arrays_to_save.items():
            if array is not None:
                np.save(Path(temp_dir) / f'{filename}.npy', array)
        
        # Save ml_indexer
        ml_indexer = get_attr(bundle, 'ml_indexer')
        if ml_indexer is None:
            raise ValueError("Bundle must have an ml_indexer attribute")
        
        # Save indexer state directory
        indexer_state_dir = Path(temp_dir) / 'ml_indexer'
        ml_indexer.save(str(indexer_state_dir))
        
        # Create tar.gz archive
        with tarfile.open(output_path, 'w:gz') as tar:
            tar.add(temp_dir, arcname='.')
        
        print(f"Saved bundle to {output_path}")
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

def save_fixed_width_table(
    table,
    out_path = None,
    columns= None,
    width = 16,
    precision = 5,
) -> Path:

    """
        table: pd.DataFrame | np.ndarray,
    out_path: Path | str | None = None,
    columns: list[str] | None = None,
    width: int = 16,
    precision: int = 5,
    ) -> Path:
    Save tabular data in fort.59-style fixed-width text format.

    Parameters
    ----------
    table : pd.DataFrame | np.ndarray
        Input data to serialize. Must be 2D.
    out_path : Path | str
        Output file path.
    columns : Sequence[str] | None
        Optional column names when `table` is a numpy array.
        For DataFrames, defaults to existing DataFrame columns.
    width : int
        Fixed character width for each field.
    precision : int
        Decimal precision for numeric fields.

    Returns
    -------
    Path
        Path of the written file.
    """
    if width < 8:
        raise ValueError('width must be >= 8')
    if precision < 0:
        raise ValueError('precision must be >= 0')

    if isinstance(table, pd.DataFrame):
        values = table.to_numpy()
        col_names = [str(col) for col in table.columns]
    else:
        values = np.asarray(table)
        if values.ndim != 2:
            raise ValueError('numpy input must be a 2D array')
        if columns is None:
            col_names = None
        else:
            if len(columns) != values.shape[1]:
                raise ValueError('Length of columns must match number of table columns')
            col_names = columns

    if values.ndim != 2:
        raise ValueError('table must be 2D')

    if out_path is None:
        output_path = Path.cwd() / 'ad.in' # default filename used by HeFESTo for convenience
    else:
        output_path = Path(out_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)


    with output_path.open('w', encoding='utf-8', newline='\n') as handle:
        if col_names is not None:
            header_line = ''.join(f'{col:>{width}}' for col in col_names)
            handle.write(header_line + '\n')
        for row in values:
            fields: List[str] = []
            for value in row:
                if pd.isna(value):
                    fields.append(f'{float("nan"):>{width}.{precision}f}')
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    fields.append(f'{float(value):>{width}.{precision}f}')
                else:
                    fields.append(f'{str(value):>{width}}')
            handle.write(''.join(fields) + '\n')

    return output_path


# ---------------------------------------------------------------------------
# HeFESTo fort-file I/O
# ---------------------------------------------------------------------------

try:
    from ngibbs.config.constants import (
        COMPOSITIONAL_COMPONENTS_IN_PHASES_HEFESTO,
        HEFESTO_ABBREVIATION_TO_SHORT_NAMES,
        get_oxide_molar_mass,
    )
except ImportError:
    try:
        from src.ngibbs.config.constants import (
            COMPOSITIONAL_COMPONENTS_IN_PHASES_HEFESTO,
            HEFESTO_ABBREVIATION_TO_SHORT_NAMES,
            get_oxide_molar_mass,
        )
    except ImportError:
        COMPOSITIONAL_COMPONENTS_IN_PHASES_HEFESTO: Dict = {}
        HEFESTO_ABBREVIATION_TO_SHORT_NAMES: Dict = {}

        def get_oxide_molar_mass(name: str) -> float:  # type: ignore[misc]
            return 100.0

try:
    import torch
    from nMELTS.engine.EOS_arithmetic.hefesto_physub import (
        get_hefesto_physub_context,
        compute_physub_bulk_matrix,
    )
except ImportError:
    torch = None  # type: ignore[assignment]


PHASE_ABBREVIATION_OVERRIDES: Dict[str, str] = {
    'c2c': 'hp-clinopyroxene',
    'il': 'akimotoite',
    'pv': 'bridgmanite',
    'mw': 'ferropericlase',
    'fea': 'iron',
    'feg': 'iron',
    'fee': 'iron',
}

COMPONENT_ABBREVIATION_OVERRIDES: Dict[str, str] = {
    'mgil': 'mg-akimotoite',
    'feil': 'fe-akimotoite',
    'mgpv': 'mg-bridgmanite',
    'fepv': 'fe-bridgmanite',
    'alpv': 'al-bridgmanite',
    'hepv': 'ferric-bridgmanite',
    'hlpv': 'ferric-bridgmanite-ls',
    'fapv': 'ferric-al-bridgmanite',
    'crpv': 'cr-bridgmanite',
    'fea': 'alpha-iron',
    'feg': 'gamma-iron',
    'fee': 'epsilon-iron',
    'mgc2': 'hp-clinoenstatite',
    'fec2': 'hp-clinoferrosilite',
}


# Files from which _safe_read_ws_table dropped stray rows, path -> row count.
# Recorded as a by-product of the read that already happens, so callers can
# report how many simulations were affected without re-walking the tree.
_DROPPED_ROWS: Dict[str, int] = {}


def reset_dropped_rows() -> None:
    """Clear the dropped-row tally (call before a batch to get per-batch counts)."""
    _DROPPED_ROWS.clear()


def get_dropped_rows() -> Dict[str, int]:
    """Return {file path: rows dropped} since the last :func:`reset_dropped_rows`."""
    return dict(_DROPPED_ROWS)


def _safe_read_ws_table(path: str, skiprows: int = 0) -> pd.DataFrame:
    """Read a whitespace-delimited HeFESTo fort.* table.

    HeFESTo interleaves diagnostic lines into its output tables, e.g.

        WARNING: Phase Rule      0.00000   220.00122   27    8    0
        Invalid Solut 0.00000 ...

    ``pd.read_csv`` happily admits these as *data* rows (short rows are
    NaN-padded), which silently shifts every subsequent row down by one.  For
    fort.99 that means the assemblage no longer lines up with the P-T grid in
    fort.56: properties get evaluated one pressure step behind, and any phase
    transition appears one step LATE.

    This is easy to miss because it barely moves rho/Vp/Vs — they are smooth in
    composition — while it badly distorts the metamorphic Cp/K_S terms, which
    depend on dn/dT and so are most sensitive exactly at a transition.  It hits
    21% of the JiChingSims models (83/402), and the two cases where the stray
    line appears mid-file are worse still, since only part of the profile shifts.

    Rows whose first field is not numeric are therefore dropped, and the index
    is reset so positional slicing stays aligned with fort.56.
    """
    df = pd.read_csv(path, sep=r'\s+', engine='python', skiprows=skiprows)
    if df.shape[1] == 0:
        return df
    first_numeric = pd.to_numeric(df.iloc[:, 0], errors='coerce').notna()
    n_bad = int((~first_numeric).sum())
    if n_bad:
        _DROPPED_ROWS[str(path)] = n_bad
        warnings.warn(
            f'{path}: dropped {n_bad} non-numeric row(s) (HeFESTo diagnostic '
            f'lines such as "WARNING: Phase Rule"). Left in place these shift '
            f'the table against the fort.56 P-T grid.',
            RuntimeWarning, stacklevel=2,
        )
        df = df.loc[first_numeric].reset_index(drop=True)
    return df


def _resolve_phase_name_from_abbr(phase_abbr: str) -> str:
    if phase_abbr in PHASE_ABBREVIATION_OVERRIDES:
        return PHASE_ABBREVIATION_OVERRIDES[phase_abbr]
    return HEFESTO_ABBREVIATION_TO_SHORT_NAMES.get(phase_abbr, phase_abbr)


def _resolve_component_name_from_abbr(component_abbr: str) -> str:
    if component_abbr in COMPONENT_ABBREVIATION_OVERRIDES:
        return COMPONENT_ABBREVIATION_OVERRIDES[component_abbr]
    return HEFESTO_ABBREVIATION_TO_SHORT_NAMES.get(component_abbr, component_abbr)


def _build_reverse_component_phase_map() -> Dict[str, List[str]]:
    reverse_map: Dict[str, List[str]] = {}
    for phase_name, comp_list in COMPOSITIONAL_COMPONENTS_IN_PHASES_HEFESTO.items():
        if phase_name in {'System_main', 'Bulk_comp', 'Bulk_comp_elements'}:
            continue
        for component in comp_list:
            reverse_map.setdefault(component, []).append(phase_name)
    return reverse_map


def _parse_control_file(control_path: str) -> Tuple[Dict[str, float], Dict[str, str]]:
    with open(control_path, 'r', encoding='utf-8', errors='ignore') as handle:
        lines = [line.rstrip('\n') for line in handle]

    element_moles: Dict[str, float] = {}
    control_component_to_phase_abbr: Dict[str, str] = {}

    oxides_start = None
    for i, line in enumerate(lines):
        if line.strip().lower() == 'oxides':
            oxides_start = i + 1
            break
    if oxides_start is None:
        raise ValueError(f"No 'oxides' block found in {control_path}")

    for i in range(oxides_start, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            continue
        if stripped.lower().startswith('phase '):
            break
        if ',' in stripped:
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        symbol = parts[0]
        if symbol.upper() == 'O':
            symbol = 'O'
        try:
            value = float(parts[1])
        except ValueError:
            continue
        element_moles[symbol] = value

    active_phase_abbr: Optional[str] = None
    expect_flag_line = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith('phase '):
            fields = stripped.split()
            if len(fields) >= 2:
                active_phase_abbr = fields[1].strip()
                expect_flag_line = True
            continue
        if active_phase_abbr is None:
            continue
        if expect_flag_line:
            expect_flag_line = False
            continue
        component_abbr = stripped.split()[0]
        control_component_to_phase_abbr[component_abbr] = active_phase_abbr

    return element_moles, control_component_to_phase_abbr


def _parse_fort56(path: str) -> pd.DataFrame:
    return _safe_read_ws_table(path, skiprows=1)


def _resolve_component_phase(
    component_abbr: str,
    component_name: str,
    reverse_component_phase_map: Dict[str, List[str]],
    control_component_to_phase_abbr: Dict[str, str],
) -> Optional[str]:
    candidates = reverse_component_phase_map.get(component_name, [])

    if len(candidates) == 1:
        return candidates[0]

    if component_abbr in control_component_to_phase_abbr:
        phase_abbr = control_component_to_phase_abbr[component_abbr]
        phase_name = _resolve_phase_name_from_abbr(phase_abbr)
        if len(candidates) == 0:
            return phase_name
        if phase_name in candidates:
            return phase_name

    if len(candidates) > 0:
        return candidates[0]

    return None


def load_fort99_component_moles_and_labels(sim_dir: str, indexer) -> Tuple[np.ndarray, np.ndarray]:
    """Load fort.99 components into an extensive component-moles array and
    compute intensive (VC) and molar (P) phase labels using an ml_indexer.

    Parameters
    ----------
    sim_dir : str
        Directory containing a `fort.99` file.
    indexer : object
        ml_indexer providing `label_names` (component labels) and
        `phaseToCompMap` (2D array-like with shape [n_phases, n_components]).

    Returns
    -------
    VC : np.ndarray
        Intensive phase labels with shape (n_rows, n_phases). Contains component proportions that sum to 1 across each phase
    P : np.ndarray
        Molar phase labels (phase moles) with shape (n_rows, n_phases).
    """
    from pathlib import Path as _Path
    sim_dir = _Path(sim_dir)
    fort99_path = sim_dir / 'fort.99'
    if not fort99_path.exists() or not fort99_path.is_file():
        raise FileNotFoundError(f'fort.99 not found in: {sim_dir}')

    comp_df = _safe_read_ws_table(str(fort99_path), skiprows=0)
    if comp_df.shape[1] < 6:
        raise ValueError('fort.99 has insufficient columns to parse components')

    nrows = len(comp_df)
    component_count = len(getattr(indexer, 'label_names', []))
    if component_count == 0:
        raise ValueError('Indexer must expose non-empty `label_names`')

    component_moles = np.zeros((nrows, component_count), dtype=float)

    component_cols = list(comp_df.columns)[3:-2]
    for comp_abbr in component_cols:
        comp_abbr_str = str(comp_abbr).strip()
        comp_name = _resolve_component_name_from_abbr(comp_abbr_str)
        try:
            comp_idx = list(indexer.label_names).index(comp_name)
        except ValueError:
            print(f"[WARNING] Component {comp_name} not found in indexer!")
            continue
        values = pd.to_numeric(comp_df[comp_abbr], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        component_moles[:, comp_idx] = values

    p_to_c = np.asarray(getattr(indexer, 'phaseToCompMap', None), dtype=float)
    if p_to_c is None or p_to_c.ndim != 2 or p_to_c.shape[1] != component_count:
        raise ValueError('Indexer must expose `phaseToCompMap` with shape [n_phases, n_components]')

    P = component_moles @ p_to_c.T

    phaseComponentMoles = component_moles[:, None, :] * p_to_c[None, :, :]

    comp_sums = np.sum(phaseComponentMoles, axis=2, keepdims=True)
    # print(comp_sums)
    with np.errstate(invalid='ignore', divide='ignore'):
        VC = np.einsum('bpc,pc->bc', np.divide(phaseComponentMoles, comp_sums, where=(comp_sums > 0)), p_to_c) @ indexer.variedToAllComp.T
    # print(f"Shape of VC: {VC.shape}, should be {len(indexer.compositionally_variable_subset)} Shape of P: {P.shape}")
    # print(VC)
    return VC, P


def load_fort99_componentMoles(sim_dir: str, indexer) -> np.ndarray:
    """Load fort.99 components into an extensive component-moles array.

    Parameters
    ----------
    sim_dir : str
        Directory containing a `fort.99` file.
    indexer : object
        ml_indexer providing `label_names` (component labels).

    Returns
    -------
    componentMoles : np.ndarray
        Extensive component moles with shape (n_rows, n_components).
    """
    from pathlib import Path as _Path
    sim_dir = _Path(sim_dir)
    fort99_path = sim_dir / 'fort.99'
    #if not fort99_path.exists() or not fort99_path.is_file():
    #    raise FileNotFoundError(f'fort.99 not found in: {sim_dir}')

    comp_df = _safe_read_ws_table(str(fort99_path), skiprows=0)
    if comp_df.shape[1] < 6:
        raise ValueError('fort.99 has insufficient columns to parse components')

    nrows = len(comp_df)
    component_count = len(getattr(indexer, 'label_names', []))
    if component_count == 0:
        raise ValueError('Indexer must expose non-empty `label_names`')

    component_moles = np.zeros((nrows, component_count), dtype=float)

    component_cols = list(comp_df.columns)[3:-2]
    for comp_abbr in component_cols:
        comp_abbr_str = str(comp_abbr).strip()
        comp_name = _resolve_component_name_from_abbr(comp_abbr_str)
        values = pd.to_numeric(comp_df[comp_abbr], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        try:
            comp_idx = list(indexer.label_names).index(comp_name)
        except ValueError:
            if values.any():
                print(f"[WARNING] Component {comp_name} is present in fort.99 but not found in indexer!")
            continue
        component_moles[:, comp_idx] = values
    return component_moles


def extract_bulk_properties_from_simulation_dir(
    sim_dir: str,
    include_phase_properties: bool = False,
    repeat: int = 1,
) -> Dict[str, Any]:
    """Extract bulk thermodynamic properties from a single HeFESTo simulation.

    Reads all usable bulk properties from fort.56 (all numeric columns), along
    with key tables from fort.61/68/99 for component-based diagnostics.

    Parameters
    ----------
    sim_dir : str
        Path to HeFESTo simulation directory (containing fort.*, control files).
    include_phase_properties : bool, default=False
        Reserved for compatibility.
    repeat : int, default=1
        Repeat every array N times (for scalability testing).

    Returns
    -------
    dict
        - Core arrays: 'P(GPa)', 'T(K)', 'VS(km/s)', 'VP(km/s)'
        - One entry per numeric fort.56 bulk column
        - 'fort56_bulk': dict[str, np.ndarray] mirror of numeric fort.56 columns
        - 'fort56_dominant_phase': optional dominant phase labels
        - 'component_names', 'component_moles'
    """
    import os as _os
    from pathlib import Path as _Path

    required_files = ['control', 'fort.56', 'fort.61', 'fort.68', 'fort.99']
    for fname in required_files:
        fpath = _os.path.join(sim_dir, fname)
        if not _os.path.exists(fpath):
            raise FileNotFoundError(f'Missing required file: {fpath}')

    element_moles, control_component_to_phase_abbr = _parse_control_file(
        _os.path.join(sim_dir, 'control')
    )

    sys_df = _parse_fort56(_os.path.join(sim_dir, 'fort.56'))
    rho_df = _safe_read_ws_table(_os.path.join(sim_dir, 'fort.61'), skiprows=0)
    vol_df = _safe_read_ws_table(_os.path.join(sim_dir, 'fort.68'), skiprows=0)
    comp_df = _safe_read_ws_table(_os.path.join(sim_dir, 'fort.99'), skiprows=0)
    fort59_path = _os.path.join(sim_dir, 'fort.59')
    fort59_df = None
    if _os.path.exists(fort59_path):
        fort59_df = _safe_read_ws_table(fort59_path, skiprows=0)

    table_lengths = [len(sys_df), len(rho_df), len(vol_df), len(comp_df)]
    if fort59_df is not None:
        table_lengths.append(len(fort59_df))
    nrows = min(table_lengths)
    if nrows <= 0:
        raise ValueError('No rows found across required HeFESTo tables')

    sys_df = sys_df.iloc[:nrows].reset_index(drop=True)
    rho_df = rho_df.iloc[:nrows].reset_index(drop=True)
    vol_df = vol_df.iloc[:nrows].reset_index(drop=True)
    comp_df = comp_df.iloc[:nrows].reset_index(drop=True)
    if fort59_df is not None:
        fort59_df = fort59_df.iloc[:nrows].reset_index(drop=True)

    fort56_bulk: Dict[str, np.ndarray] = {}
    for col in sys_df.columns:
        numeric_series = pd.to_numeric(sys_df[col], errors='coerce')
        if numeric_series.notna().any():
            fort56_bulk[str(col)] = numeric_series.fillna(0.0).to_numpy(dtype=np.float32)

    fort59_bulk: Dict[str, np.ndarray] = {}
    if fort59_df is not None:
        for col in fort59_df.columns:
            numeric_series = pd.to_numeric(fort59_df[col], errors='coerce')
            if numeric_series.notna().any():
                fort59_bulk[str(col)] = numeric_series.fillna(0.0).to_numpy(dtype=np.float32)

    if 'P(GPa)' not in fort56_bulk or 'T(K)' not in fort56_bulk:
        raise ValueError("fort.56 is missing required numeric columns 'P(GPa)' and/or 'T(K)'")

    dominant_phase_col = None
    for col in sys_df.columns:
        if str(col).strip().lower() == 'dominant_phase':
            dominant_phase_col = col
            break

    reverse_component_phase_map = _build_reverse_component_phase_map()
    component_cols = list(comp_df.columns)[3:-2]

    component_names: List[str] = []
    component_moles_list: List[np.ndarray] = []

    for comp_abbr in component_cols:
        comp_abbr_str = str(comp_abbr).strip()
        component_name = _resolve_component_name_from_abbr(comp_abbr_str)
        phase_name = _resolve_component_phase(
            component_abbr=comp_abbr_str,
            component_name=component_name,
            reverse_component_phase_map=reverse_component_phase_map,
            control_component_to_phase_abbr=control_component_to_phase_abbr,
        )

        if component_name and phase_name:
            component_names.append(component_name)
            values = pd.to_numeric(comp_df[comp_abbr], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
            component_moles_list.append(values)

    if not component_moles_list:
        raise ValueError('No valid components found in fort.99')

    component_moles = np.stack(component_moles_list, axis=1)

    result: Dict[str, Any] = {
        'P(GPa)': fort56_bulk['P(GPa)'],
        'T(K)': fort56_bulk['T(K)'],
        'VS(km/s)': fort56_bulk.get('VS(km/s)', np.zeros(nrows, dtype=np.float32)),
        'VP(km/s)': fort56_bulk.get('VP(km/s)', np.zeros(nrows, dtype=np.float32)),
        'fort56_bulk': fort56_bulk,
        'fort59_bulk': fort59_bulk,
        'component_names': component_names,
        'component_moles': component_moles,
    }

    for col_name, values in fort56_bulk.items():
        result[col_name] = values

    for col_name, values in fort59_bulk.items():
        result[f'fort59::{col_name}'] = values

    if dominant_phase_col is not None:
        result['fort56_dominant_phase'] = sys_df[dominant_phase_col].astype(str).to_numpy()

    if repeat > 1:
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                result[key] = np.repeat(value, repeat, axis=0)
            elif isinstance(value, (list, tuple)):
                result[key] = value * repeat
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        result[key][k] = np.repeat(v, repeat, axis=0)
                    elif isinstance(v, (list, tuple)):
                        result[key][k] = v * repeat

    if torch is not None:
        try:
            molar_masses = np.array(
                [get_oxide_molar_mass(name) for name in component_names],
                dtype=np.float32,
            )
        except Exception:
            molar_masses = np.ones(len(component_names), dtype=np.float32) * 100.0

    return result

    return output_path

