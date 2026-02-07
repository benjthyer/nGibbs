"""
Filter functions for balancing and filtering MELTS datasets.

These functions are designed to be used as methods on BigMetaTable objects,
but can also be called as standalone functions with an explicit indexer parameter.
"""

import os
import numpy as np
import gc
import tarfile
import tempfile
import shutil
import pickle

# Ensure src is on path
import sys
from pathlib import Path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

top_path = str(Path(__file__).parent.parent.parent.parent)
if top_path not in sys.path:
    sys.path.insert(0, top_path)

# Import DatasetIndexer type for type hints
from typing import Optional
from builder.indexer import DatasetIndexer
from tests.unit_tests.test_processing.ML_export_tests import sanity_check_bundle


# Oxide bounds configuration
Oxide_Lower_Bounds = [
    ['k-feldspar', 'K2O', 8],
    ['clinopyroxene', 'CaO', 7]
]

easy_build_oxide_upper_bounds = {
    'k-feldspar': [['CaO', 1]],
    'plagioclase': [['K2O', 8]],
    'clinopyroxene': [['TiO2', 6], ['Al2O3', 8]],
    'orthopyroxene': [['TiO2', 2], ['Al2O3', 8], ['CaO', 4]],
}

Oxide_Upper_Bounds = []
for phase, boundlist in easy_build_oxide_upper_bounds.items():
    for ox, bound in boundlist:
        Oxide_Upper_Bounds.append([phase, ox, bound])

Component_Upper_Bounds = []  # [ ['plagioclase', 'highsanidine', 0.1] ]


def balance_lowF(MetaTable, indexer: Optional[DatasetIndexer] = None, sacred_phases=None, batch_size=200_000):
    """
    Balance dataset by removing excess entries in extreme melt fraction ranges.
    
    Balances the dataset to have roughly equal representation across different
    melt fraction ranges, with target bin size based on middle range (40-60%).
    
    Parameters
    ----------
    MetaTable : BigMetaTable
        BigMetaTable instance (when used as method)
    indexer : DatasetIndexer, optional
        DatasetIndexer instance. If None, uses MetaTable.indexer
    sacred_phases : list, optional
        List of phase names to protect from deletion
    batch_size : int, default=200000
        Batch size for processing (not currently used but kept for compatibility)
    """
    # Get indexer from self if not provided
    if indexer is None:
        if not hasattr(MetaTable, 'indexer'):
            raise ValueError("indexer must be provided if MetaTable.indexer is not available")
        indexer = MetaTable.indexer
    
    # Extract indexing dictionaries from indexer
    mass_phasedict = indexer.mass_phasedict
    MELTS_indices = indexer.MELTS_indices
    mass_indices = indexer.mass_indices

    if sacred_phases:
        sacredIDX = np.array([mass_phasedict[phase] for phase in sacred_phases])
        check_sacred = True
    else:
        sacredIDX = None
        check_sacred = False

    delete_indices = np.array([], dtype=int)

    # Calculate melt mass percentage
    melt_mass = MetaTable.table[:, MELTS_indices['melts-liquid']['liq mass (gm)']] * (
        100 / (np.sum(MetaTable.table[:, mass_indices], axis=1) + 1E-8)
    )

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
            sacred_presence = np.sum(MetaTable.binary_labels[block_indices][:, sacredIDX], axis=1)
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

    # Near-Liquidus melt fraction block: 90–100%
    upperBlock = (melt_mass >= 90) & (melt_mass < 100)
    numToDelete = int(upperBlock.sum() - (2 * targetBinNo))
    delete_indices = np.append(delete_indices, select_deletable_indices(upperBlock, numToDelete))

    # Subsolidus block: exactly 0%
    solidBlock = melt_mass == 0
    numToDelete = int(solidBlock.sum() - targetBinNo)
    delete_indices = np.append(delete_indices, select_deletable_indices(solidBlock, numToDelete))

    # Superliquidus block: exactly 100%
    liquidBlock = (melt_mass >= 99)  # Account for not always perfectly summing to 1
    numToDelete = int(liquidBlock.sum() - targetBinNo)
    delete_indices = np.append(delete_indices, select_deletable_indices(liquidBlock, numToDelete))

    # Delete ALL references to memory map
    del melt_mass, tertiaryBlock, secondaryBlock, solidBlock, liquidBlock, upperBlock
    gc.collect()

    print(f"Deleting {len(delete_indices)} entries")

    MetaTable.delete(delete_indices)


def balance_Superliquidus_fxtal(MetaTable, indexer: Optional[DatasetIndexer] = None, sacred_phases=None, batch_size=200_000):
    """
    Balance dataset for fractional crystallization runs.
    
    Focuses on balancing superliquidus and subsolidus regions for fractional
    crystallization datasets.
    
    Parameters
    ----------
    MetaTable : BigMetaTable
        BigMetaTable instance (when used as method)
    indexer : DatasetIndexer, optional
        DatasetIndexer instance. If None, uses MetaTable.indexer
    sacred_phases : list, optional
        List of phase names to protect from deletion
    batch_size : int, default=200000
        Batch size for processing (not currently used but kept for compatibility)
    """
    # Get indexer from self if not provided
    if indexer is None:
        if not hasattr(MetaTable, 'indexer'):
            raise ValueError("indexer must be provided if MetaTable.indexer is not available")
        indexer = MetaTable.indexer
    
    # Extract indexing dictionaries from indexer
    mass_phasedict = indexer.mass_phasedict
    MELTS_indices = indexer.MELTS_indices
    mass_indices = indexer.mass_indices

    if sacred_phases:
        sacredIDX = np.array([mass_phasedict[phase] for phase in sacred_phases])
        check_sacred = True
    else:
        sacredIDX = None
        check_sacred = False

    delete_indices = np.array([], dtype=int)

    # Calculate melt mass percentage
    melt_mass = MetaTable.table[:, MELTS_indices['melts-liquid']['liq mass (gm)']]
    melt_mass = melt_mass * 100 / (np.sum(MetaTable.table[:, mass_indices], axis=1) + 1E-8)

    # Determine target bin size from middle melt fraction range (95–99.95%)
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
            sacred_presence = np.sum(MetaTable.binary_labels[block_indices][:, sacredIDX], axis=1)
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
    liquidBlock = (melt_mass >= 99.95)  # Account for not always perfectly summing to 1
    numToDelete = int(liquidBlock.sum() - targetBinNo)
    delete_indices = np.append(delete_indices, select_deletable_indices(liquidBlock, numToDelete))

    # Subsolidus block: exactly 0%
    solidBlock = melt_mass == 0
    numToDelete = int(solidBlock.sum() - (targetBinNo / 2))
    delete_indices = np.append(delete_indices, select_deletable_indices(solidBlock, numToDelete))

    # Delete ALL references to memory map
    del melt_mass, liquidBlock, solidBlock
    gc.collect()

    print(f"Deleting {len(delete_indices)} entries")

    MetaTable.delete(delete_indices)


def balance_geodynamics(MetaTable, indexer: Optional[DatasetIndexer] = None, sacred_phases=None, batch_size=200_000):
    """
    Balance dataset for geodynamic applications.
    
    Removes all data with melt fraction > 10% and balances low melt fraction regions
    so that the 5-10% range has at most 1/3 as many entries as the 0.01-5% range.
    
    Parameters
    ----------
    MetaTable : BigMetaTable
        BigMetaTable instance (when used as method)
    indexer : DatasetIndexer, optional
        DatasetIndexer instance. If None, uses MetaTable.indexer
    sacred_phases : list, optional
        List of phase names to protect from deletion
    batch_size : int, default=200000
        Batch size for processing (not currently used but kept for compatibility)
    """
    # Get indexer from self if not provided
    if indexer is None:
        if not hasattr(MetaTable, 'indexer'):
            raise ValueError("indexer must be provided if MetaTable.indexer is not available")
        indexer = MetaTable.indexer
    
    # Extract indexing dictionaries from indexer
    mass_phasedict = indexer.mass_phasedict
    MELTS_indices = indexer.MELTS_indices
    mass_indices = indexer.mass_indices

    if sacred_phases:
        sacredIDX = np.array([mass_phasedict[phase] for phase in sacred_phases])
        check_sacred = True
    else:
        sacredIDX = None
        check_sacred = False

    delete_indices = np.array([], dtype=int)

    # Calculate melt mass percentage
    melt_mass = MetaTable.table[:, MELTS_indices['melts-liquid']['liq mass (gm)']] * (
        100 / (np.sum(MetaTable.table[:, mass_indices], axis=1) + 1E-8)
    )

    print(f"Initial dataset size: {len(melt_mass)}")

    def select_deletable_indices(mask, num_to_delete):
        """Helper to select deletable entries that lack sacred phases."""
        if num_to_delete <= 0:
            print('None to delete for geodynamics balancing')
            return np.array([], dtype=int)
        # Only consider entries in the block
        block_indices = np.where(mask)[0]
        if len(block_indices) == 0:
            print('No potential sims in this block')
            return np.array([], dtype=int)

        if check_sacred:
            # Check for sacred phases
            sacred_presence = np.sum(MetaTable.binary_labels[block_indices][:, sacredIDX], axis=1)
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

    # Remove all data with melt fraction > 10%
    highMeltBlock = melt_mass > 10
    numHighMelt = highMeltBlock.sum()
    print(f"Removing {numHighMelt} entries with melt fraction > 10%")
    delete_indices = np.append(delete_indices, select_deletable_indices(highMeltBlock, numHighMelt))

    # Count low melt fraction bins
    lowMeltBlock = (melt_mass > 0.01) & (melt_mass <= 5)
    midMeltBlock = (melt_mass > 5) & (melt_mass <= 10)
    
    lowMeltCount = lowMeltBlock.sum()
    midMeltCount = midMeltBlock.sum()
    
    print(f"Low melt (0.01-5%): {lowMeltCount} entries")
    print(f"Mid melt (5-10%): {midMeltCount} entries")
    
    # Target for mid melt range: at most 1/3 of low melt count
    targetMidMelt = int(lowMeltCount / 3)
    numToDelete = midMeltCount - targetMidMelt
    
    if numToDelete > 0:
        print(f"Balancing 5-10% range to {targetMidMelt} entries (1/3 of {lowMeltCount})")
        delete_indices = np.append(delete_indices, select_deletable_indices(midMeltBlock, numToDelete))
    else:
        print(f"5-10% range already balanced (has {midMeltCount}, target ≤ {targetMidMelt})")

    # Delete ALL references to memory map
    del melt_mass, highMeltBlock, lowMeltBlock, midMeltBlock
    gc.collect()

    print(f"Deleting {len(delete_indices)} total entries for geodynamics filtering")

    MetaTable.delete(delete_indices)


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
    

def deep_filter(tarball_path, Component_Lower_Bounds=None, Component_Upper_Bounds=None, 
                        Oxide_Lower_Bounds=None, Oxide_Upper_Bounds=None, Mass_Upper_Bounds=None, 
                        batch_size=200_000):
    """Filter files within a tar.gz bundle."""
    from .MLexporter import generate_dataset_stats
    
    tarball_path = Path(tarball_path)
    sanity_check_bundle(tarball_path)

    temp_dir = tempfile.mkdtemp()
    
    try:
        # Extract tarball
        print(f"Extracting {tarball_path}...")
        with tarfile.open(tarball_path, 'r:gz') as tar:
            tar.extractall(path=temp_dir)
        
        temp_path = Path(temp_dir)
        
        # Rename stats.txt to stats_prefilter.txt
        stats_file = temp_path / 'stats.txt'
        if stats_file.exists():
            prefilter_stats = temp_path / 'stats_prefilter.txt'
            os.rename(stats_file, prefilter_stats)
            print(f"Renamed stats.txt → stats_prefilter.txt")
        
        # Load ml_indexer for filtering and regenerating stats
        ml_indexer_path = temp_path / 'ml_indexer.pkl'
        if ml_indexer_path.exists():
            with open(ml_indexer_path, 'rb') as f:
                ml_indexer = pickle.load(f)
        else:
            ml_indexer = None
        
        # Apply filtering to the .npy files using temporary filename prefix
        temp_filename_prefix = str(temp_path / 'filtered_')
        
        # Copy .npy files to temp location with prefix
        npy_files = ['labels.npy', 'features.npy', 'binary_labels.npy', 'molar_labels.npy', 'mass_labels.npy']
        for npy_file in npy_files:
            src = temp_path / npy_file
            if src.exists():
                dst = Path(f"{temp_filename_prefix}{npy_file}")
                shutil.copy(src, dst)
        
        # Apply filtering
        _deep_filter_npy(
            temp_filename_prefix,
            Component_Lower_Bounds=Component_Lower_Bounds,
            Component_Upper_Bounds=Component_Upper_Bounds,
            Oxide_Lower_Bounds=Oxide_Lower_Bounds,
            Oxide_Upper_Bounds=Oxide_Upper_Bounds,
            Mass_Upper_Bounds=Mass_Upper_Bounds,
            batch_size=batch_size,
            ml_indexer=ml_indexer
        )
        
        
        # Generate stats_postfilter.txt
        if ml_indexer is not None:
            print("Generating post-filter statistics...")
            generate_dataset_stats(
                dataset_name=str(temp_path / 'filtered_'),
                ml_indexer=ml_indexer,
                output_dir=temp_path
            )

            # Move filtered files back to original names
            for npy_file in npy_files:
                filtered_src = Path(f"{temp_filename_prefix}{npy_file}")
                if filtered_src.exists():
                    dst = temp_path / npy_file
                    shutil.move(filtered_src, dst)

            # Rename to stats_postfilter.txt
            postfilter_tmp = temp_path / 'filtered__stats.txt'
            postfilter_stats = temp_path / 'stats_postfilter.txt'
            if postfilter_tmp.exists():
                os.rename(postfilter_tmp, postfilter_stats)
                print(f"Created stats_postfilter.txt")
        
        # Repack tarball
        print(f"Repacking filtered data into {tarball_path}...")
        with tarfile.open(tarball_path, 'w:gz') as tar:
            for file in temp_path.glob('*'):
                if file.is_file() and not file.name.startswith('filtered_'):
                    tar.add(file, arcname=file.name)
        
        print(f"Filtering complete: {tarball_path}")
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


def _deep_filter_tarball(tarball_path, Component_Lower_Bounds=None, Component_Upper_Bounds=None, 
                        Oxide_Lower_Bounds=None, Oxide_Upper_Bounds=None, Mass_Upper_Bounds=None, 
                        indexer: Optional[DatasetIndexer] = None, batch_size=200_000):
    """Filter files within a tar.gz bundle."""
    from .MLexporter import generate_dataset_stats
    
    tarball_path = Path(tarball_path)
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Extract tarball
        print(f"Extracting {tarball_path}...")
        with tarfile.open(tarball_path, 'r:gz') as tar:
            tar.extractall(path=temp_dir)
        
        temp_path = Path(temp_dir)
        
        # Rename stats.txt to stats_prefilter.txt
        stats_file = temp_path / 'stats.txt'
        if stats_file.exists():
            prefilter_stats = temp_path / 'stats_prefilter.txt'
            os.rename(stats_file, prefilter_stats)
            print(f"Renamed stats.txt → stats_prefilter.txt")
        
        # Load ml_indexer for filtering and regenerating stats
        ml_indexer_path = temp_path / 'ml_indexer.pkl'
        if ml_indexer_path.exists():
            with open(ml_indexer_path, 'rb') as f:
                ml_indexer = pickle.load(f)
        else:
            ml_indexer = None
        
        # Apply filtering to the .npy files using temporary filename prefix
        temp_filename_prefix = str(temp_path / 'filtered_')
        
        # Copy .npy files to temp location with prefix
        npy_files = ['labels.npy', 'features.npy', 'binary_labels.npy', 'molar_labels.npy', 'mass_labels.npy']
        for npy_file in npy_files:
            src = temp_path / npy_file
            if src.exists():
                dst = Path(f"{temp_filename_prefix}{npy_file}")
                shutil.copy(src, dst)
        
        # Apply filtering
        _deep_filter_npy(
            temp_filename_prefix,
            Component_Lower_Bounds=Component_Lower_Bounds,
            Component_Upper_Bounds=Component_Upper_Bounds,
            Oxide_Lower_Bounds=Oxide_Lower_Bounds,
            Oxide_Upper_Bounds=Oxide_Upper_Bounds,
            Mass_Upper_Bounds=Mass_Upper_Bounds,
            batch_size=batch_size,
            ml_indexer=ml_indexer
        )
        
        # Move filtered files back to original names
        for npy_file in npy_files:
            filtered_src = Path(f"{temp_filename_prefix}{npy_file}")
            if filtered_src.exists():
                dst = temp_path / npy_file
                shutil.move(filtered_src, dst)
        
        # Generate stats_postfilter.txt
        if ml_indexer is not None:
            print("Generating post-filter statistics...")
            generate_dataset_stats(
                dataset_name=str(temp_path / 'filtered_'),
                ml_indexer=ml_indexer,
                output_dir=temp_path
            )
            # Rename to stats_postfilter.txt
            postfilter_tmp = temp_path / 'filtered__stats.txt'
            postfilter_stats = temp_path / 'stats_postfilter.txt'
            if postfilter_tmp.exists():
                os.rename(postfilter_tmp, postfilter_stats)
                print(f"Created stats_postfilter.txt")
        
        # Repack tarball
        print(f"Repacking filtered data into {tarball_path}...")
        with tarfile.open(tarball_path, 'w:gz') as tar:
            for file in temp_path.glob('*'):
                if file.is_file() and not file.name.startswith('filtered_'):
                    tar.add(file, arcname=file.name)
        
        print(f"Filtering complete: {tarball_path}")
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


def _deep_filter_npy(filename, Component_Lower_Bounds=None, Component_Upper_Bounds=None, Oxide_Lower_Bounds=None, Oxide_Upper_Bounds=None, Mass_Upper_Bounds=None, batch_size=200_000, ml_indexer=None):
    """Filter an on-disk dataset using the provided ml_indexer for lookups."""

    if ml_indexer is None:
        raise ValueError("ml_indexer is required for deep_filter")

    detail_label_indices = ml_indexer.detail_label_indices
    all_phases = ml_indexer.all_phases
    mass_phasedict = ml_indexer.mass_phasedict
    label_indices_comp = ml_indexer.label_indices_comp
    label_indices = ml_indexer.label_indices
    oxide_dict = {ox: i for i, ox in enumerate(ml_indexer.Oxides)}

    compToOxLoad = getattr(ml_indexer, "compToOxLoad", None)
    MM = getattr(ml_indexer, "MM", None)

    if (Oxide_Lower_Bounds or Oxide_Upper_Bounds) and (compToOxLoad is None or MM is None):
        raise ValueError("Oxide filtering requires compToOxLoad and MM from the ml_indexer")

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

        oxides_GT = None

        batch_indices = np.arange(start, end)

        # Oxide Lower Bounds
        if Oxide_Lower_Bounds is not None:
            for phase, ox, bound in Oxide_Lower_Bounds:
                print(f"Processing {phase} {ox} Lower Bound filter...")
                print(f"comp_batch shape: {comp_batch.shape}, label_indices_comp[phase]: {label_indices_comp[phase]}, compToOxLoad shape: {compToOxLoad.shape}")
                oxides_GT = (comp_batch[:,label_indices_comp[phase]] @ compToOxLoad[label_indices[phase]]) 

                oxides_GT = oxides_GT @ MM
                oxides_GT = oxides_GT * (100/np.sum(oxides_GT,axis=1)).reshape(-1,1)

                failing = np.where((oxides_GT[:,oxide_dict[ox]] < bound)*(oxides_GT[:,oxide_dict[ox]] != 0))[0]
                print(f"Deleting {len(failing)} for {bound} Lower Bound {phase} {ox}")
                delete_indices = np.append(delete_indices, batch_indices[failing])

        # Oxide Upper Bounds
        if Oxide_Upper_Bounds is not None:
            for phase, ox, bound in Oxide_Upper_Bounds:
                oxides_GT = (comp_batch[:,label_indices_comp[phase]] @ compToOxLoad[label_indices[phase]]) 

                oxides_GT = oxides_GT @ MM
                oxides_GT = oxides_GT * (100/np.sum(oxides_GT,axis=1)).reshape(-1,1)

                failing = np.where((oxides_GT[:,oxide_dict[ox]] > bound)*(oxides_GT[:,oxide_dict[ox]] != 0))[0]
                print(f"Deleting {len(failing)} for {bound} Upper Bound {phase} {ox}")
                delete_indices = np.append(delete_indices, batch_indices[failing])

        if oxides_GT is not None:
            del oxides_GT
        del comp_batch, batch_indices
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

    free_outputs_path = filename + 'free_outputs.npy'
    if os.path.exists(free_outputs_path):
        safe_delete_batched(free_outputs_path, delete_indices)
    
    gc.collect()
    

    return delete_indices