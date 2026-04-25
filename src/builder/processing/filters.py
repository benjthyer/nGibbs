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
from nMELTS.config.ml_indexer import MLIndexer, load_ml_indexer_from_state


def _normalize_bulk_oxide_bounds(bounds):
    """Validate and normalize bulk oxide bounds mapping."""
    if bounds is None:
        return {}
    if not isinstance(bounds, dict):
        raise ValueError("Bulk_Oxide_Bounds must be a dict like {'SiO2': [min, max]}")

    normalized = {}
    for oxide, limits in bounds.items():
        if not isinstance(oxide, str):
            raise ValueError(f"Bulk oxide key must be a string, got {type(oxide)}")
        if not isinstance(limits, (list, tuple)) or len(limits) != 2:
            raise ValueError(
                f"Bulk oxide bounds for {oxide} must be [min, max], got {limits}"
            )

        lower, upper = limits
        lower = -np.inf if lower is None else float(lower)
        upper = np.inf if upper is None else float(upper)

        if lower > upper:
            raise ValueError(f"Bulk oxide bounds invalid for {oxide}: min > max")

        normalized[oxide] = (lower, upper)

    return normalized


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
    
    # Explicitly close mmap handles to release file locks on Windows
    if hasattr(arr, '_mmap') and arr._mmap is not None:
        arr._mmap.close()
    if hasattr(new_arr, '_mmap') and new_arr._mmap is not None:
        new_arr._mmap.close()
    
    del arr, new_arr
    gc.collect()
    
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
    
    # Explicitly close memmap if it has _mmap attribute (helps on Windows)
    if hasattr(original, '_mmap') and original._mmap is not None:
        original._mmap.close()
    del original
    gc.collect()
    
    os.replace(tmp_filename, filename)  # Overwrite original
    

def bundle_insanity_filter(tarball_path, tolerance=1e-3, bulk_tol_frac=1e-3, batch_size=200_000):
    """
    Filter rows that fail sanity checks from a tar.gz bundle.
    
    Performs the same checks as sanity_check_bundle but deletes failing rows
    instead of raising errors. Asserts that less than 0.1% of dataset is deleted.
    
    Checks:
    - Variable-phase label rows sum to 1 when phase present, else 0
    - (phase moles > 0) matches binary labels
    - Labels only nonzero where binary labels allow
    - Reconstructed bulk element composition matches features within tolerance
    
    Parameters
    ----------
    tarball_path : str or Path
        Path to the .tar.gz bundle file
    tolerance : float, default=1e-3
        Tolerance for component sum checks
    bulk_tol_frac : float, default=1e-3
        Fractional tolerance for bulk composition reconstruction
    batch_size : int, default=200000
        Batch size for processing
    """
    from .MLexporter import generate_dataset_stats
    
    tarball_path = Path(tarball_path)
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    ml_indexer = None

    def _save_array_preview_csv(npy_path, csv_path, max_rows=20_000, columns=None):
        if not npy_path.exists():
            return
        arr = np.load(npy_path, mmap_mode='r')
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        row_count = min(arr.shape[0], max_rows)
        data = np.asarray(arr[:row_count])
        if columns is None or len(columns) != data.shape[1]:
            columns = [f"col_{i}" for i in range(data.shape[1])]
        header = ",".join(columns)
        np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
        if hasattr(arr, '_mmap') and arr._mmap is not None:
            arr._mmap.close()
        del arr

    def _export_failure_csvs(max_rows=20_000):
        base_name = tarball_path.name.replace('.tar.gz', '')
        out_dir = tarball_path.parent / 'logs' / f"{base_name}_insanity_failure_csv"
        out_dir.mkdir(parents=True, exist_ok=True)

        if ml_indexer is not None:
            feature_cols = list(ml_indexer.featureNames) + list(ml_indexer.Elkeys)
            label_cols = [
                ml_indexer.label_names[i]
                for i in ml_indexer.compositionally_variable_subset
            ]
            phase_cols = list(ml_indexer.all_phases)
        else:
            feature_cols = None
            label_cols = None
            phase_cols = None

        _save_array_preview_csv(
            temp_path / 'features.npy',
            out_dir / 'features_first20000.csv',
            max_rows=max_rows,
            columns=feature_cols,
        )
        _save_array_preview_csv(
            temp_path / 'labels.npy',
            out_dir / 'labels_first20000.csv',
            max_rows=max_rows,
            columns=label_cols,
        )
        _save_array_preview_csv(
            temp_path / 'molar_labels.npy',
            out_dir / 'molar_labels_first20000.csv',
            max_rows=max_rows,
            columns=phase_cols,
        )
        _save_array_preview_csv(
            temp_path / 'binary_labels.npy',
            out_dir / 'binary_labels_first20000.csv',
            max_rows=max_rows,
            columns=phase_cols,
        )
        _save_array_preview_csv(
            temp_path / 'mass_labels.npy',
            out_dir / 'mass_labels_first20000.csv',
            max_rows=max_rows,
            columns=phase_cols,
        )
        _save_array_preview_csv(
            temp_path / 'free_outputs.npy',
            out_dir / 'free_outputs_first20000.csv',
            max_rows=max_rows,
            columns=getattr(ml_indexer, 'freeOutputs', None) if ml_indexer is not None else None,
        )
        print(f"[INSANITY FILTER] Exported failure CSV previews to {out_dir}")
    
    try:
        # Extract tarball
        print(f"\n[INSANITY FILTER] Extracting {tarball_path}...")
        with tarfile.open(tarball_path, 'r:gz') as tar:
            tar.extractall(path=temp_dir)
        
        # Load ml_indexer and arrays
        ml_indexer_path = temp_path / 'ml_indexer'
        ml_indexer = load_ml_indexer_from_state(ml_indexer_path)
        
        # Load arrays into memory (not mmap) to avoid file handle issues on Windows
        labels = np.load(temp_path / 'labels.npy')
        molar_labels = np.load(temp_path / 'molar_labels.npy')
        binary_labels = np.load(temp_path / 'binary_labels.npy')
        features = np.load(temp_path / 'features.npy')
        
        total_rows = labels.shape[0]
        delete_mask = np.zeros(total_rows, dtype=bool)
        
        print(f"[INSANITY FILTER] Total rows: {total_rows}")
        
        # Check 1: Variable-phase label row sums
        print("[INSANITY FILTER] Check 1: Variable-phase label row sums...")
        for phase, idxs in ml_indexer.label_indices_comp.items():
            idxs = np.asarray(idxs, dtype=int)
            phase_idx = ml_indexer.mass_phasedict[phase]
            phase_present = binary_labels[:, phase_idx] > 0.5
            row_sums = np.sum(labels[:, idxs], axis=1)
            
            present_mask = phase_present
            absent_mask = ~phase_present
            present_diff = np.abs(row_sums[present_mask] - 1.0)
            absent_diff = np.abs(row_sums[absent_mask] - 0.0)
            
            present_fail_local = present_diff > tolerance
            absent_fail_local = absent_diff > tolerance
            
            # Map back to global indices
            present_indices = np.where(present_mask)[0]
            absent_indices = np.where(absent_mask)[0]
            
            delete_mask[present_indices[present_fail_local]] = True
            delete_mask[absent_indices[absent_fail_local]] = True
            
            if present_fail_local.any() or absent_fail_local.any():
                print(f"  {phase}: {present_fail_local.sum()} present failures, {absent_fail_local.sum()} absent failures")
        
        # Check 2: Phase moles > 0 matches binary labels
        print("[INSANITY FILTER] Check 2: Phase moles vs binary labels...")
        phase_present_from_moles = molar_labels > 0
        phase_present_from_binary = binary_labels > 0.5
        mismatch = phase_present_from_moles != phase_present_from_binary
        mismatch_rows = np.any(mismatch, axis=1)
        delete_mask[mismatch_rows] = True
        print(f"  {mismatch_rows.sum()} rows with phase mole/binary mismatches")
        
        # Check 3: Labels only nonzero where binary labels allow. Also fails if intensive components do not sum to 1 when present within a given phase.
        print("[INSANITY FILTER] Check 3: Labels vs binary label constraints...")
        comp_mappings = ml_indexer.comp_mappings
        comp_binaries = np.asarray(ml_indexer.comp_binaries, dtype=int)
        label_implied_binary = labels @ comp_mappings.T
        binary_target = binary_labels[:, comp_binaries]
        binary_diff = np.abs(label_implied_binary - binary_target)
        binary_fail = np.any(binary_diff > tolerance, axis=1)
        delete_mask[binary_fail] = True
        print(f"  {binary_fail.sum()} rows with label/binary constraint violations")
        
        # Check 4: Bulk element reconstruction
        print("[INSANITY FILTER] Check 4: Bulk element reconstruction...")
        phase_to_comp = ml_indexer.phaseToCompMap
        comp_moles = molar_labels @ phase_to_comp
        
        varied_to_all = ml_indexer.variedToAllComp
        labels_full = labels @ varied_to_all
        var_idx = np.asarray(ml_indexer.compositionally_variable_subset, dtype=int)
        
        comp_frac = np.ones_like(labels_full)
        comp_frac[:, var_idx] = labels_full[:, var_idx]
        comp_moles = comp_moles * comp_frac
        
        bulk_el = (comp_moles @ ml_indexer.compToOxLoad) @ ml_indexer.OxToEl
        bulk_el = bulk_el / np.sum(bulk_el, axis=1, keepdims=True)
        
        feature_offset = len(ml_indexer.featureNames)
        expected_el = features[:, feature_offset:]
        rel_diff = np.abs(bulk_el - expected_el) / (expected_el + 1e-10)
        near0 = expected_el < 1e-6
        rel_diff[near0] = np.abs(bulk_el[near0] - expected_el[near0])
        per_row_max = np.max(rel_diff, axis=1)
        bulk_fail = per_row_max > bulk_tol_frac
        delete_mask[bulk_fail] = True
        print(f"  {bulk_fail.sum()} rows with bulk composition mismatches")
        
        # Cleanup loaded arrays
        del labels, molar_labels, binary_labels, features
        gc.collect()
        
        # Get indices to delete
        delete_indices = np.where(delete_mask)[0]
        num_to_delete = len(delete_indices)
        deletion_fraction = num_to_delete / total_rows
        
        print(f"\n[INSANITY FILTER] Total rows to delete: {num_to_delete} ({deletion_fraction:.4%})")
        
        # Assert less than 0.1% deleted
        assert deletion_fraction < 0.001, (
            f"Insanity filter would delete {deletion_fraction:.4%} of dataset "
            f"(threshold: 0.1%). Dataset may have serious quality issues."
        )
        
        if num_to_delete == 0:
            print("[INSANITY FILTER] No rows to delete. Bundle passed all sanity checks.")
            # Clean up ml_indexer reference to avoid handle issues
            del ml_indexer
            gc.collect()
            return
        
        # Apply deletion to all .npy files
        temp_filename_prefix = str(temp_path / 'filtered_')
        npy_files = ['labels.npy', 'features.npy', 'binary_labels.npy', 'molar_labels.npy', 'mass_labels.npy']
        
        for npy_file in npy_files:
            src = temp_path / npy_file
            if src.exists():
                dst = Path(f"{temp_filename_prefix}{npy_file}")
                shutil.copy(src, dst)
        
        # Copy free_outputs if present
        free_outputs_src = temp_path / 'free_outputs.npy'
        if free_outputs_src.exists():
            shutil.copy(free_outputs_src, Path(f"{temp_filename_prefix}free_outputs.npy"))
        
        # Delete rows from all files
        for npy_file in npy_files:
            file_path = f"{temp_filename_prefix}{npy_file}"
            if Path(file_path).exists():
                safe_delete_batched(file_path, delete_indices, batch_size=batch_size)
        
        if Path(f"{temp_filename_prefix}free_outputs.npy").exists():
            safe_delete_batched(f"{temp_filename_prefix}free_outputs.npy", delete_indices, batch_size=batch_size)
        
        # Generate post-filter stats
        print("[INSANITY FILTER] Generating post-filter statistics...")
        generate_dataset_stats(
            dataset_name=temp_filename_prefix,
            ml_indexer=ml_indexer,
            output_dir=temp_path
        )
        
        # Move filtered files back to original names (use os.replace for atomic overwrite on Windows)
        for npy_file in npy_files:
            filtered_src = Path(f"{temp_filename_prefix}{npy_file}")
            if filtered_src.exists():
                dst = temp_path / npy_file
                if dst.exists():
                    os.remove(dst)
                os.replace(filtered_src, dst)
        
        filtered_free_outputs = Path(f"{temp_filename_prefix}free_outputs.npy")
        if filtered_free_outputs.exists():
            dst = temp_path / 'free_outputs.npy'
            if dst.exists():
                os.remove(dst)
            os.replace(filtered_free_outputs, dst)
        
        # Rename stats
        postfilter_tmp = temp_path / 'filtered__stats.txt'
        if postfilter_tmp.exists():
            # Rename existing stats.txt to stats_prefilter.txt if present
            stats_file = temp_path / 'stats.txt'
            if stats_file.exists():
                os.rename(stats_file, temp_path / 'stats_prefilter.txt')
            # Rename new stats to stats.txt
            os.rename(postfilter_tmp, temp_path / 'stats.txt')
        
        # Clean up ml_indexer reference before repacking to release any file handles
        del ml_indexer
        gc.collect()
        
        # Repack tarball
        print(f"[INSANITY FILTER] Repacking filtered data into {tarball_path}...")
        with tarfile.open(tarball_path, 'w:gz') as tar:
            for file in temp_path.glob('*'):
                if file.is_file() and not file.name.startswith('filtered_'):
                    tar.add(file, arcname=file.name)
            
            # Re-add ml_indexer directory
            ml_indexer_dir = temp_path / 'ml_indexer'
            if ml_indexer_dir.is_dir():
                tar.add(ml_indexer_dir, arcname='ml_indexer')
        
        # Ensure tarball is fully closed before cleanup
        gc.collect()
        
        print(f"[INSANITY FILTER] Complete. Deleted {num_to_delete} rows ({deletion_fraction:.4%})")

    except Exception as e:
        print(f"[INSANITY FILTER] Error: {e}")
        print(f"[INSANITY FILTER] Exporting first 20000 rows to CSV.")
    
        try:
            _export_failure_csvs(max_rows=20_000)
        except Exception as export_error:
            print(f"[INSANITY FILTER] Failed to export failure CSV previews: {export_error}")
        raise
        
    finally:
        # Clean up temporary directory with retry on Windows
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            # Windows may still have file handles open, force garbage collection and retry
            gc.collect()
            import time
            time.sleep(0.1)  # Brief delay to allow handles to close
            try:
                shutil.rmtree(temp_dir)
            except PermissionError:
                # If still failing, try to remove files individually
                print(f"[INSANITY FILTER] Warning: Could not remove temp directory {temp_dir} due to open file handles")
                print(f"[INSANITY FILTER] You may need to manually delete: {temp_dir}")


def deep_filter(tarball_path, Component_Lower_Bounds=None, Component_Upper_Bounds=None,
                        Oxide_Lower_Bounds=None, Oxide_Upper_Bounds=None, Mass_Upper_Bounds=None,
                        Bulk_Oxide_Bounds=None, batch_size=200_000):
    """Filter files within a tar.gz bundle."""
    from .MLexporter import generate_dataset_stats
    
    tarball_path = Path(tarball_path)
    #sanity_check_bundle(tarball_path)

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
        """ml_indexer_path = temp_path / 'ml_indexer.pkl' #OLD PKL. DEPRECATED. 
        if ml_indexer_path.exists():
            with open(ml_indexer_path, 'rb') as f:
                ml_indexer = pickle.load(f)
        else:
            ml_indexer = None"""
        ml_indexer_path = temp_path / 'ml_indexer'
        ml_indexer = load_ml_indexer_from_state(ml_indexer_path)
        
        # Apply filtering to the .npy files using temporary filename prefix
        temp_filename_prefix = str(temp_path / 'filtered_')
        
        # Copy .npy files to temp location with prefix
        npy_files = ['labels.npy', 'features.npy', 'binary_labels.npy', 'molar_labels.npy', 'mass_labels.npy']
        for npy_file in npy_files:
            src = temp_path / npy_file
            if src.exists():
                dst = Path(f"{temp_filename_prefix}{npy_file}")
                shutil.copy(src, dst)

        # Preserve alignment for optional free outputs during filtering.
        free_outputs_src = temp_path / 'free_outputs.npy'
        if free_outputs_src.exists():
            shutil.copy(free_outputs_src, Path(f"{temp_filename_prefix}free_outputs.npy"))
        
        # Apply filtering
        _deep_filter_npy(
            temp_filename_prefix,
            Component_Lower_Bounds=Component_Lower_Bounds,
            Component_Upper_Bounds=Component_Upper_Bounds,
            Oxide_Lower_Bounds=Oxide_Lower_Bounds,
            Oxide_Upper_Bounds=Oxide_Upper_Bounds,
            Mass_Upper_Bounds=Mass_Upper_Bounds,
            Bulk_Oxide_Bounds=Bulk_Oxide_Bounds,
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

            # Move filtered files back to original names (use os.replace for atomic overwrite on Windows)
            for npy_file in npy_files:
                filtered_src = Path(f"{temp_filename_prefix}{npy_file}")
                if filtered_src.exists():
                    dst = temp_path / npy_file
                    if dst.exists():
                        os.remove(dst)
                    os.replace(filtered_src, dst)

            filtered_free_outputs = Path(f"{temp_filename_prefix}free_outputs.npy")
            if filtered_free_outputs.exists():
                dst = temp_path / 'free_outputs.npy'
                if dst.exists():
                    os.remove(dst)
                os.replace(filtered_free_outputs, dst)

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
            
            # Re-add ml_indexer directory if present
            ml_indexer_dir = temp_path / 'ml_indexer'
            if ml_indexer_dir.is_dir():
                print("Re-packing ml_indexer state directory...")
                tar.add(ml_indexer_dir, arcname='ml_indexer')
        

        #sanity_check_bundle(tarball_path)
        print(f"Filtering complete: {tarball_path}")
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


def _deep_filter_npy(filename, Component_Lower_Bounds=None, Component_Upper_Bounds=None,
                     Oxide_Lower_Bounds=None, Oxide_Upper_Bounds=None, Mass_Upper_Bounds=None,
                     Bulk_Oxide_Bounds=None, batch_size=200_000, ml_indexer=None):
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
    phase_to_comp = getattr(ml_indexer, "phaseToCompMap", None)
    varied_to_all = getattr(ml_indexer, "variedToAllComp", None)
    var_idx = np.asarray(getattr(ml_indexer, "compositionally_variable_subset", []), dtype=int)

    bulk_oxide_bounds = _normalize_bulk_oxide_bounds(Bulk_Oxide_Bounds)

    if (Oxide_Lower_Bounds or Oxide_Upper_Bounds) and (compToOxLoad is None or MM is None):
        raise ValueError("Oxide filtering requires compToOxLoad and MM from the ml_indexer")

    if bulk_oxide_bounds and (
        compToOxLoad is None or MM is None or phase_to_comp is None or varied_to_all is None
    ):
        raise ValueError(
            "Bulk oxide filtering requires compToOxLoad, MM, phaseToCompMap, and variedToAllComp"
        )

    for oxide in bulk_oxide_bounds:
        if oxide not in oxide_dict:
            raise ValueError(
                f"Bulk oxide filter requested unknown oxide '{oxide}'. "
                f"Known oxides: {list(oxide_dict.keys())}"
            )

    components = np.load(filename + 'labels.npy', mmap_mode='r')
    binary_labels = np.load(filename + 'binary_labels.npy', mmap_mode='r')
    molar_labels = None
    if bulk_oxide_bounds:
        molar_labels = np.load(filename + 'molar_labels.npy', mmap_mode='r')

    delete_indices = np.array([], dtype=int)

    # === Full-array filters for components (cheap)
    if Component_Lower_Bounds is not None:
        for phase, comp, bound in Component_Lower_Bounds:
            if phase not in detail_label_indices or comp not in detail_label_indices[phase]:
                print(f"Warning: {phase} or {comp} not found in detail_label_indices. Skipping this filter.")
                continue
            idx = detail_label_indices[phase][comp]
            to_delete = np.where((components[:, idx] < bound)*(components[:, idx] != 0))[0]
            print(f"Deleting {len(to_delete)} for {bound} Lower Bound {phase} {comp}")
            delete_indices = np.append(delete_indices, to_delete)
            
    if Component_Upper_Bounds is not None:
        for phase, comp, bound in Component_Upper_Bounds:
            if phase not in detail_label_indices or comp not in detail_label_indices[phase]:
                print(f"Warning: {phase} or {comp} not found in detail_label_indices. Skipping this filter.")
                continue
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
                if phase not in detail_label_indices or ox not in oxide_dict:
                    print(f"Warning: {phase} or {ox} not found in detail_label_indices or oxide_dict. Skipping this filter.")
                    continue
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
                if phase not in detail_label_indices or ox not in oxide_dict:
                    print(f"Warning: {phase} or {ox} not found in detail_label_indices or oxide_dict. Skipping this filter.")
                    continue
                oxides_GT = (comp_batch[:,label_indices_comp[phase]] @ compToOxLoad[label_indices[phase]]) 

                oxides_GT = oxides_GT @ MM
                oxides_GT = oxides_GT * (100/np.sum(oxides_GT,axis=1)).reshape(-1,1)

                failing = np.where((oxides_GT[:,oxide_dict[ox]] > bound)*(oxides_GT[:,oxide_dict[ox]] != 0))[0]
                print(f"Deleting {len(failing)} for {bound} Upper Bound {phase} {ox}")
                delete_indices = np.append(delete_indices, batch_indices[failing])

        # Whole-assemblage bulk oxide bounds
        if bulk_oxide_bounds:
            comp_moles_batch = molar_labels[start:end] @ phase_to_comp
            labels_full = comp_batch @ varied_to_all
            comp_frac = np.ones_like(labels_full)
            comp_frac[:, var_idx] = labels_full[:, var_idx]
            comp_moles_batch = comp_moles_batch * comp_frac

            bulk_oxide_wt = comp_moles_batch @ compToOxLoad
            bulk_oxide_wt = bulk_oxide_wt @ MM
            bulk_oxide_wt = bulk_oxide_wt * (
                100.0 / (np.sum(bulk_oxide_wt, axis=1, keepdims=True) + 1e-12)
            )

            for oxide, (lower, upper) in bulk_oxide_bounds.items():
                oxide_vals = bulk_oxide_wt[:, oxide_dict[oxide]]
                failing = np.where((oxide_vals < lower) | (oxide_vals > upper))[0]
                print(
                    f"Deleting {len(failing)} for bulk {oxide} outside [{lower}, {upper}]"
                )
                delete_indices = np.append(delete_indices, batch_indices[failing])

            del comp_moles_batch, labels_full, comp_frac, bulk_oxide_wt

        if oxides_GT is not None:
            del oxides_GT
        del comp_batch, batch_indices
        gc.collect()
            
    delete_indices = np.unique(delete_indices)
    print(f"Rows after deleting: {n_rows-len(delete_indices)}")

    # Explicitly close mmap handles before deletion to release file locks on Windows
    if hasattr(components, '_mmap') and components._mmap is not None:
        components._mmap.close()
    if hasattr(binary_labels, '_mmap') and binary_labels._mmap is not None:
        binary_labels._mmap.close()
    if molar_labels is not None and hasattr(molar_labels, '_mmap') and molar_labels._mmap is not None:
        molar_labels._mmap.close()
    
    del components, binary_labels, molar_labels
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