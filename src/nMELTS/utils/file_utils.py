"""
File operation utilities.

Extracted from Legacy/BackEnds/EmulatorLibrary.py
"""

import os
import shutil
from pathlib import Path
import tarfile
import tempfile
import pickle
import numpy as np


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


def clear_new_files(directory, baseline_files, protected_extensions=None):
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
        
    Returns
    -------
    int
        Number of files deleted
        
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
        try:
            filepath.unlink()
            print(f"Deleted temporary file: {filepath}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {filepath}: {e}")
    
    return deleted_count

class MLDataBundle:
    """Container for ML dataset bundle loaded from .tar.gz file."""
    def __init__(self):
        self.molar_labels = None
        self.binary_labels = None
        self.mass_labels = None
        self.features = None
        self.labels = None
        self.free_outputs = None
        self.ml_indexer = None


def load_ml_bundle(bundle_path):
    """
    Load a .tar.gz bundle created by resampling_to_datasets.
    
    Extracts all .npy files and the ml_indexer.pkl from the tarball and returns
    them as a MLDataBundle object with attributes for each file.
    
    Parameters
    ----------
    bundle_path : str or Path
        Path to the .tar.gz bundle file
        
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
    temp_dir = tempfile.mkdtemp()
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
        }
        
        for attr_name, filename in npy_files.items():
            file_path = Path(temp_dir) / filename
            if file_path.exists():
                setattr(bundle, attr_name, np.load(file_path))
            elif attr_name != 'free_outputs':
                raise FileNotFoundError(f"Expected file not found in bundle: {filename}")
        
        # Load ml_indexer state directory (preferred)
        indexer_dir = Path(temp_dir) / 'ml_indexer'
        if indexer_dir.exists():
            from nMELTS.config.ml_indexer import load_ml_indexer_from_state
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
    temp_dir = tempfile.mkdtemp()
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


