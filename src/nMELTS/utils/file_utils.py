"""
File operation utilities.

Extracted from Legacy/BackEnds/EmulatorLibrary.py
"""

import os
import shutil
from pathlib import Path


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


def move_file(src_path, dst_path, overwrite=False):
    """
    Move a file from the current working directory, to a destination directory.

    Args:
        src_path (str): Name of the file in the current working directory.
        dst_path (str): Destination directory
        overwrite (bool): If True, overwrite any existing file with the same name.
    """
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"Source file not found: {src_path}")

    if src_path == dst_path:
        return

    # Handle overwrite
    if os.path.exists(dst_path):
        if overwrite:
            os.remove(dst_path)
        else:
            raise FileExistsError(f"Destination file already exists: {dst_path}")

    # Move the file
    shutil.move(src_path, dst_path)
    print(f"Moved '{src_path}' to '{dst_path}' successfully.")


def move_files_with_extension(extension, dst_dir, src_dir=None, overwrite=False):
    """
    Move all files with a given extension from the current working directory
    to the destination directory.

    Args:
        extension (str): File extension, e.g., '.csv' or '.dat'
        dst_dir (str or Path): Destination directory path
        src_dir (str or Path): Source location from working directory, Default none is cwd. 
        overwrite (bool): If True, overwrite existing files in destination
    """
    if src_dir == dst_dir: 
        return

    if src_dir is not None:
        src_path = Path.cwd() / src_dir
    else:
        src_path = Path.cwd()

    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Find all matching files in current directory
    files_to_move = [f for f in src_path.iterdir() if f.is_file() and f.suffix == extension]

    if not files_to_move:
        print(f"No files with extension '{extension}' found.")
        return

    for file_path in files_to_move:
        dest_path = dst_dir / file_path.name

        if dest_path.exists():
            if overwrite:
                dest_path.unlink()  # Remove existing file
            else:
                print(f"Skipping {file_path.name}, already exists in destination.")
                continue

        shutil.move(str(file_path), str(dest_path))
        print(f"Moved {file_path.name} -> {dst_dir}")
