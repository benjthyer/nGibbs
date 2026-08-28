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
from tqdm import tqdm

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
from ngibbs.config.ml_indexer import MLIndexer, load_ml_indexer_from_state
from ngibbs.utils.file_utils import chunked_mask_copy, ROW_ALIGNED_BUNDLE_ARRAYS


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


def filter_bulk_composition_mismatch(MetaTable, indexer: Optional[DatasetIndexer] = None,
                                      tolerance_ppm=5000, chunk_size=100_000):
    """
    Delete rows where MELTS's own reported bulk composition (Bulk_comp oxides)
    disagrees with the bulk composition computed by summing/converting the
    individual phase assemblages (phase masses -> component moles -> oxides ->
    elements). A mismatch means the simulation's own per-phase output doesn't add
    up to its own reported system totals - most commonly seen as a low-temperature
    MELTS solver artifact at unstable conditions - rather than anything introduced
    by this pipeline.

    Must run on a BigMetaTable whose indexer.ml_indexer projection matrices
    (compToOxLoad, OxToEl, MM, Oxides) are already built (i.e. after
    exclude_oxides/table_update), and before resampling_to_datasets() perturbs the
    table's mass columns - the comparison is only meaningful against the
    as-simulated data.

    Parameters
    ----------
    MetaTable : BigMetaTable
        BigMetaTable instance (when used as method)
    indexer : DatasetIndexer, optional
        DatasetIndexer instance. If None, uses MetaTable.indexer
    tolerance_ppm : float, default=5000
        Relative tolerance (ppm) between reported and phase-derived bulk element
        fractions before a row is considered inconsistent.
    chunk_size : int, default=100_000
        Row-chunk size used by the component-moles computation this reuses.
    """
    if indexer is None:
        if not hasattr(MetaTable, 'indexer'):
            raise ValueError("indexer must be provided if MetaTable.indexer is not available")
        indexer = MetaTable.indexer

    ml_indexer = indexer.ml_indexer
    compToOxLoad = ml_indexer.compToOxLoad
    OxToEl = ml_indexer.OxToEl
    MM = ml_indexer.MM
    Oxides = ml_indexer.Oxides
    Elkeys = ml_indexer.Elkeys
    MM_diag = np.diag(MM).reshape(1, -1)

    # Component moles computed directly from the table, with no resampling
    # perturbation (retrieve_component_moles defaults to multiplier_bounds=[1,1]).
    # table1 is aliased rather than copied since retrieve_component_moles only
    # ever reads it.
    MetaTable.table1 = MetaTable.table
    MetaTable.retrieve_component_moles(chunk_size=chunk_size)
    molar = MetaTable.molar  # memmap (n_rows, ncomps)

    oxide_indices = np.array([indexer.MELTS_indices['Bulk_comp'][ox] for ox in Oxides])

    # Graphite is reduced carbon (2Fe2O3 + C -> 4FeO + CO2), so its compToOxLoad row
    # carries large FeO/Fe2O3 coefficients (+4/-2) against its own tiny molar mass
    # (12.011 g/mol). MELTS's own graphite mass output is only reported to 1e-4 g
    # precision, and for the trace amounts graphite is usually present in, that
    # rounding gets amplified ~24-27x once run through the FeO/Fe2O3 stoichiometry -
    # landing almost entirely on Fe3 (confirmed empirically: graphite-bearing rows
    # that fail this check fail on Fe3 alone, never on any other element). This is
    # output-precision noise on a trace phase, not a real inconsistency, so Fe3 is
    # overlooked specifically for rows where graphite is present.
    graphite_mass_col = indexer.MELTS_indices.get('graphite', {}).get('mass (gm)')
    fe3_idx = Elkeys.index('Fe3') if (graphite_mass_col is not None and 'Fe3' in Elkeys) else None

    # Phase -> mass column lookup, used below to profile the assemblage of rows
    # flagged for deletion as we go (liquid uses a differently-named mass column).
    phase_mass_cols = []
    for phase in ml_indexer.all_phases:
        phase_cols = indexer.MELTS_indices.get(phase, {})
        mass_col = (phase_cols.get('liq mass (gm)') if phase == 'melts-liquid'
                    else phase_cols.get('mass (gm)'))
        if mass_col is not None:
            phase_mass_cols.append((phase, mass_col))
    phase_mass_col_array = np.array([col for _, col in phase_mass_cols])
    phase_presence_counts = np.zeros(len(phase_mass_cols), dtype=np.int64)

    total_rows = molar.shape[0]
    tolerance_fraction = tolerance_ppm * 1e-6

    # Processed one row-chunk at a time (matching retrieve_component_moles above) so
    # that neither the (n_rows, n_oxides/n_elements) intermediates nor the table's
    # oxide columns are ever materialized for the full dataset at once - on an
    # 80GB+ memmapped table those full-height arrays alone can exceed available RAM.
    delete_chunks = []
    max_rel_diff_overall = 0.0
    graphite_row_count = 0
    for start in tqdm(range(0, total_rows, chunk_size), desc="Checking bulk composition mismatch"):
        end = min(start + chunk_size, total_rows)
        molar_chunk = np.asarray(molar[start:end])

        # Bulk element fractions computed from phases (component moles -> oxides -> elements)
        Inmoles_chunk = (molar_chunk @ compToOxLoad) @ OxToEl
        bulk_el_from_phases_chunk = Inmoles_chunk / (np.sum(Inmoles_chunk, axis=1, keepdims=True) + 1e-10)

        # Bulk element fractions MELTS reports directly via Bulk_comp
        bulk_wt_oxides_chunk = np.asarray(MetaTable.table[start:end, oxide_indices])
        bulk_moles_oxides_chunk = bulk_wt_oxides_chunk / MM_diag
        bulk_moles_elements_chunk = bulk_moles_oxides_chunk @ OxToEl
        bulk_el_from_reported_chunk = bulk_moles_elements_chunk / (
            np.sum(bulk_moles_elements_chunk, axis=1, keepdims=True) + 1e-10
        )

        rel_diff_chunk = np.abs(bulk_el_from_phases_chunk - bulk_el_from_reported_chunk) / (
            bulk_el_from_reported_chunk + 1e-10
        )
        near_zero_chunk = bulk_el_from_reported_chunk < 1e-6
        rel_diff_chunk[near_zero_chunk] = np.abs(
            bulk_el_from_phases_chunk[near_zero_chunk] - bulk_el_from_reported_chunk[near_zero_chunk]
        )

        if fe3_idx is not None:
            has_graphite_chunk = np.asarray(MetaTable.table[start:end, graphite_mass_col]) > 0
            if has_graphite_chunk.any():
                rel_diff_chunk[has_graphite_chunk, fe3_idx] = 0.0
                graphite_row_count += int(has_graphite_chunk.sum())

        max_rel_diff_per_row_chunk = np.max(rel_diff_chunk, axis=1)
        max_rel_diff_overall = max(max_rel_diff_overall, float(np.max(max_rel_diff_per_row_chunk)))

        chunk_delete = np.where(max_rel_diff_per_row_chunk > tolerance_fraction)[0]
        if chunk_delete.size:
            delete_chunks.append(chunk_delete + start)
            if phase_mass_col_array.size:
                deletable_mass_chunk = np.asarray(
                    MetaTable.table[start:end][chunk_delete][:, phase_mass_col_array]
                )
                phase_presence_counts += np.sum(deletable_mass_chunk > 0, axis=0)

    del MetaTable.molar, MetaTable.table1
    gc.collect()

    if graphite_row_count:
        print(
            f"[BULK MISMATCH FILTER] Overlooking Fe3 for {graphite_row_count} graphite-bearing "
            f"rows (known amplified noise from MELTS's coarse graphite-mass precision)."
        )

    delete_indices = np.concatenate(delete_chunks) if delete_chunks else np.array([], dtype=np.int64)

    if delete_indices.size > 0:
        deletion_fraction = delete_indices.size / MetaTable.table.shape[0]
        print(
            f"[BULK MISMATCH FILTER] Deleting {delete_indices.size} rows ({deletion_fraction:.4%}) "
            f"where reported Bulk_comp disagrees with phase-derived bulk composition "
            f"beyond {tolerance_fraction:.2e} relative (max observed: {max_rel_diff_overall:.2e})."
        )
        print("[BULK MISMATCH FILTER] Phase assemblage of deleted rows:")
        for (phase, _), count in zip(phase_mass_cols, phase_presence_counts):
            if count > 0:
                pct = 100 * int(count) / delete_indices.size
                print(f"  {phase:20} present in {int(count):>8,} deleted rows ({pct:>6.2f}%)")

        log_dir = Path(MetaTable.filename).parent / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        out_path = log_dir / f"{Path(MetaTable.filename).name}_bulk_composition_mismatch_deleted_rows.txt"
        with open(out_path, 'w') as f:
            MetaTable.write_meta_lines(f, indices=delete_indices)
        print(f"[BULK MISMATCH FILTER] Wrote run metadata for {delete_indices.size} deleted rows to {out_path}")

        MetaTable.delete(delete_indices)
    else:
        print("[BULK MISMATCH FILTER] No rows found with bulk composition mismatch.")

    return delete_indices


def safe_delete_batched(filename, delete_indices, batch_size=200000):
    """
    Filters a .npy file by removing rows listed in delete_indices.
    Creates a temporary file and replaces the original.

    Uses the shared `chunked_mask_copy` helper (see ngibbs.utils.file_utils)
    so peak RAM stays O(batch_size) - this previously accumulated every kept
    batch in a Python list and materialized the whole surviving array via
    `np.vstack` before writing, which for a wide array (e.g. labels.npy at
    tens of millions of rows) held the entire post-filter result in RAM at
    once.
    """
    original = np.load(filename, mmap_mode='r')
    n_rows = original.shape[0]

    keep_mask = np.ones(n_rows, dtype=bool)
    keep_mask[np.asarray(delete_indices, dtype=np.int64)] = False

    tmp_filename = filename + '.tmp.npy'
    chunked_mask_copy(original, tmp_filename, keep_mask, chunk_size=batch_size)

    # Explicitly close memmap if it has _mmap attribute (helps on Windows)
    if hasattr(original, '_mmap') and original._mmap is not None:
        original._mmap.close()
    del original
    gc.collect()

    os.replace(tmp_filename, filename)  # Overwrite original


def insanity_filter_npy(file_prefix, ml_indexer, tolerance=1e-3, bulk_tol_frac=1e-3,
                         batch_size=200_000, log_dir=None):
    """
    Filter rows that fail sanity checks directly on an unpacked bundle's .npy files.

    Same checks as `bundle_insanity_filter`, but operates in place on
    `file_prefix + 'labels.npy'` etc. - the on-disk layout produced by
    `resampling_to_datasets` before packaging, or by extracting a bundle's
    .tar.gz - so no tar extraction/repacking is required. Asserts that less
    than 0.1% of the dataset is deleted.

    Checks:
    - Variable-phase label rows sum to 1 when phase present, else 0
    - (phase moles > 0) matches binary labels
    - Labels only nonzero where binary labels allow
    - Reconstructed bulk element composition matches features within tolerance

    Parameters
    ----------
    file_prefix : str
        Prefix such that `file_prefix + 'labels.npy'` etc. resolve to the
        dataset's .npy files.
    ml_indexer : MLIndexer
        Already-loaded indexer describing the dataset schema.
    tolerance : float, default=1e-3
        Tolerance for component sum checks
    bulk_tol_frac : float, default=1e-3
        Fractional tolerance for bulk composition reconstruction
    batch_size : int, default=200000
        Batch size for processing
    log_dir : str or Path, optional
        Directory to write failure-preview CSVs to if a sanity check trips the
        0.1% assertion. Defaults to a `logs/` directory next to `file_prefix`.

    Returns
    -------
    np.ndarray
        Indices of rows deleted (empty if the dataset passed all checks).
    """
    if log_dir is None:
        base_name = Path(file_prefix).name or Path(file_prefix).parent.name
        log_dir = Path(file_prefix).parent / 'logs' / f"{base_name}_insanity_failure_csv"
    else:
        log_dir = Path(log_dir)

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
        log_dir.mkdir(parents=True, exist_ok=True)

        feature_cols = list(ml_indexer.featureNames) + list(ml_indexer.Elkeys)
        label_cols = [
            ml_indexer.label_names[i]
            for i in ml_indexer.compositionally_variable_subset
        ]
        phase_cols = list(ml_indexer.all_phases)

        _save_array_preview_csv(
            Path(f"{file_prefix}features.npy"),
            log_dir / 'features_first20000.csv',
            max_rows=max_rows,
            columns=feature_cols,
        )
        _save_array_preview_csv(
            Path(f"{file_prefix}labels.npy"),
            log_dir / 'labels_first20000.csv',
            max_rows=max_rows,
            columns=label_cols,
        )
        _save_array_preview_csv(
            Path(f"{file_prefix}molar_labels.npy"),
            log_dir / 'molar_labels_first20000.csv',
            max_rows=max_rows,
            columns=phase_cols,
        )
        _save_array_preview_csv(
            Path(f"{file_prefix}binary_labels.npy"),
            log_dir / 'binary_labels_first20000.csv',
            max_rows=max_rows,
            columns=phase_cols,
        )
        _save_array_preview_csv(
            Path(f"{file_prefix}mass_labels.npy"),
            log_dir / 'mass_labels_first20000.csv',
            max_rows=max_rows,
            columns=phase_cols,
        )
        _save_array_preview_csv(
            Path(f"{file_prefix}free_outputs.npy"),
            log_dir / 'free_outputs_first20000.csv',
            max_rows=max_rows,
            columns=getattr(ml_indexer, 'free_outputs', None),
        )
        print(f"[INSANITY FILTER] Exported failure CSV previews to {log_dir}")

    try:
        # Memory-mapped rather than loaded in full - all four checks below are
        # row-independent (none needs data from another row, only the small
        # schema-sized ml_indexer matrices pulled out before the loop), so the
        # whole scan can run in row-chunks with peak RAM bounded by batch_size
        # instead of holding all four full-size arrays in RAM at once (tens of
        # GB combined on a large dataset).
        labels = np.load(f"{file_prefix}labels.npy", mmap_mode='r')
        molar_labels = np.load(f"{file_prefix}molar_labels.npy", mmap_mode='r')
        binary_labels = np.load(f"{file_prefix}binary_labels.npy", mmap_mode='r')
        features = np.load(f"{file_prefix}features.npy", mmap_mode='r')

        total_rows = labels.shape[0]
        delete_mask = np.zeros(total_rows, dtype=bool)

        print(f"[INSANITY FILTER] Total rows: {total_rows}")

        label_indices_comp_items = list(ml_indexer.label_indices_comp.items())
        comp_mappings = ml_indexer.comp_mappings
        comp_binaries = np.asarray(ml_indexer.comp_binaries, dtype=int)
        phase_to_comp = ml_indexer.phaseToCompMap
        varied_to_all = ml_indexer.variedToAllComp
        var_idx = np.asarray(ml_indexer.compositionally_variable_subset, dtype=int)
        feature_offset = len(ml_indexer.featureNames)

        present_fail_counts = {phase: 0 for phase, _ in label_indices_comp_items}
        absent_fail_counts = {phase: 0 for phase, _ in label_indices_comp_items}
        mismatch_total = 0
        binary_fail_total = 0
        bulk_fail_total = 0

        print("[INSANITY FILTER] Checks 1-4: scanning in chunks...")
        for start in tqdm(range(0, total_rows, batch_size), desc="[INSANITY FILTER] Scanning"):
            end = min(start + batch_size, total_rows)
            labels_chunk = np.asarray(labels[start:end])
            molar_chunk = np.asarray(molar_labels[start:end])
            binary_chunk = np.asarray(binary_labels[start:end])
            features_chunk = np.asarray(features[start:end])
            chunk_delete = np.zeros(end - start, dtype=bool)

            # Check 1: Variable-phase label row sums
            for phase, idxs in label_indices_comp_items:
                idxs = np.asarray(idxs, dtype=int)
                phase_idx = ml_indexer.mass_phasedict[phase]
                phase_present = binary_chunk[:, phase_idx] > 0.5
                row_sums = np.sum(labels_chunk[:, idxs], axis=1)

                present_mask = phase_present
                absent_mask = ~phase_present
                present_diff = np.abs(row_sums[present_mask] - 1.0)
                absent_diff = np.abs(row_sums[absent_mask] - 0.0)

                present_fail_local = present_diff > tolerance
                absent_fail_local = absent_diff > tolerance

                # Map back to chunk-local indices
                present_indices = np.where(present_mask)[0]
                absent_indices = np.where(absent_mask)[0]

                chunk_delete[present_indices[present_fail_local]] = True
                chunk_delete[absent_indices[absent_fail_local]] = True

                present_fail_counts[phase] += int(present_fail_local.sum())
                absent_fail_counts[phase] += int(absent_fail_local.sum())

            # Check 2: Phase moles > 0 matches binary labels
            phase_present_from_moles = molar_chunk > 0
            phase_present_from_binary = binary_chunk > 0.5
            mismatch_rows = np.any(phase_present_from_moles != phase_present_from_binary, axis=1)
            chunk_delete |= mismatch_rows
            mismatch_total += int(mismatch_rows.sum())

            # Check 3: Labels only nonzero where binary labels allow. Also fails if intensive components do not sum to 1 when present within a given phase.
            label_implied_binary = labels_chunk @ comp_mappings.T
            binary_target = binary_chunk[:, comp_binaries]
            binary_fail = np.any(np.abs(label_implied_binary - binary_target) > tolerance, axis=1)
            chunk_delete |= binary_fail
            binary_fail_total += int(binary_fail.sum())

            # Check 4: Bulk element reconstruction
            comp_moles = molar_chunk @ phase_to_comp
            labels_full = labels_chunk @ varied_to_all
            comp_frac = np.ones_like(labels_full)
            comp_frac[:, var_idx] = labels_full[:, var_idx]
            comp_moles = comp_moles * comp_frac

            bulk_el = (comp_moles @ ml_indexer.compToOxLoad) @ ml_indexer.OxToEl
            bulk_el = bulk_el / np.sum(bulk_el, axis=1, keepdims=True)

            expected_el = features_chunk[:, feature_offset:]
            rel_diff = np.abs(bulk_el - expected_el) / (expected_el + 1e-10)
            near0 = expected_el < 1e-6
            rel_diff[near0] = np.abs(bulk_el[near0] - expected_el[near0])
            bulk_fail = np.max(rel_diff, axis=1) > bulk_tol_frac
            chunk_delete |= bulk_fail
            bulk_fail_total += int(bulk_fail.sum())

            delete_mask[start:end] = chunk_delete

        print("[INSANITY FILTER] Check 1: Variable-phase label row sums:")
        for phase, _ in label_indices_comp_items:
            if present_fail_counts[phase] or absent_fail_counts[phase]:
                print(f"  {phase}: {present_fail_counts[phase]} present failures, {absent_fail_counts[phase]} absent failures")
        print(f"[INSANITY FILTER] Check 2: {mismatch_total} rows with phase mole/binary mismatches")
        print(f"[INSANITY FILTER] Check 3: {binary_fail_total} rows with label/binary constraint violations")
        print(f"[INSANITY FILTER] Check 4: {bulk_fail_total} rows with bulk composition mismatches")

        # Cleanup mmap handles
        for arr in (labels, molar_labels, binary_labels, features):
            if hasattr(arr, '_mmap') and arr._mmap is not None:
                arr._mmap.close()
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
            print("[INSANITY FILTER] No rows to delete. Dataset passed all sanity checks.")
            return delete_indices

        # Delete rows directly from the real .npy files (safe_delete_batched
        # already writes to a sibling .tmp.npy and atomically replaces, so no
        # separate working copy is needed here). Every row-aligned array must be
        # trimmed with the same indices -- a derivative/free_outputs array left
        # behind keeps its pre-filter row count and desyncs from features.npy,
        # which later trips the assertion in chunked_permutation_copy (or, worse,
        # silently misaligns rows) during shuffle_bundle_rows.
        for npy_file in ROW_ALIGNED_BUNDLE_ARRAYS:
            file_path = f"{file_prefix}{npy_file}"
            if Path(file_path).exists():
                safe_delete_batched(file_path, delete_indices, batch_size=batch_size)

        print(f"[INSANITY FILTER] Complete. Deleted {num_to_delete} rows ({deletion_fraction:.4%})")
        return delete_indices

    except Exception as e:
        print(f"[INSANITY FILTER] Error: {e}")
        print(f"[INSANITY FILTER] Exporting first 20000 rows to CSV.")

        try:
            _export_failure_csvs(max_rows=20_000)
        except Exception as export_error:
            print(f"[INSANITY FILTER] Failed to export failure CSV previews: {export_error}")
        raise


def bundle_insanity_filter(tarball_path, tolerance=1e-3, bulk_tol_frac=1e-3, batch_size=200_000):
    """
    Filter rows that fail sanity checks from a tar.gz bundle.

    Thin wrapper around `insanity_filter_npy` for filtering an already-packaged
    bundle: extracts the tarball, runs the checks in place, regenerates stats,
    and repacks. New pipelines should call `insanity_filter_npy` directly on
    the unpacked .npy files *before* packaging, to avoid this extract/repack
    cost - see `MLexporter.resampling_to_datasets`.

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
    tmp_base = Path(top_path) / "data" / "tmp"
    tmp_base.mkdir(parents=True, exist_ok=True)
    temp_dir = tempfile.mkdtemp(dir=tmp_base)
    temp_path = Path(temp_dir)

    try:
        print(f"\n[INSANITY FILTER] Extracting {tarball_path}...")
        with tarfile.open(tarball_path, 'r:gz') as tar:
            tar.extractall(path=temp_dir)

        ml_indexer = load_ml_indexer_from_state(temp_path / 'ml_indexer')
        file_prefix = f"{temp_path}{os.sep}"
        base_name = tarball_path.name.replace('.tar.gz', '')
        log_dir = tarball_path.parent / 'logs' / f"{base_name}_insanity_failure_csv"

        try:
            delete_indices = insanity_filter_npy(
                file_prefix,
                ml_indexer,
                tolerance=tolerance,
                bulk_tol_frac=bulk_tol_frac,
                batch_size=batch_size,
                log_dir=log_dir,
            )
        finally:
            del ml_indexer
            gc.collect()

        if len(delete_indices) == 0:
            return

        ml_indexer = load_ml_indexer_from_state(temp_path / 'ml_indexer')
        try:
            stats_file = temp_path / 'stats.txt'
            if stats_file.exists():
                os.rename(stats_file, temp_path / 'stats_prefilter.txt')

            print("[INSANITY FILTER] Generating post-filter statistics...")
            stats_path = generate_dataset_stats(
                dataset_name=file_prefix,
                ml_indexer=ml_indexer,
                output_dir=temp_path,
                chunk_size=batch_size,
            )
            os.replace(stats_path, temp_path / 'stats.txt')
        finally:
            del ml_indexer
            gc.collect()

        print(f"[INSANITY FILTER] Repacking filtered data into {tarball_path}...")
        with tarfile.open(tarball_path, 'w:gz') as tar:
            for file in temp_path.glob('*'):
                if file.is_file():
                    tar.add(file, arcname=file.name)

            ml_indexer_dir = temp_path / 'ml_indexer'
            if ml_indexer_dir.is_dir():
                tar.add(ml_indexer_dir, arcname='ml_indexer')

        gc.collect()

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
    """
    Filter files within a tar.gz bundle.

    Thin wrapper around `deep_filter_npy` for filtering an already-packaged
    bundle: extracts the tarball, filters in place, regenerates stats, and
    repacks. New pipelines should call `deep_filter_npy` directly on the
    unpacked .npy files *before* packaging, to avoid this extract/repack cost
    - see `MLexporter.resampling_to_datasets`.
    """
    from .MLexporter import generate_dataset_stats

    tarball_path = Path(tarball_path)

    tmp_base = Path(top_path) / "data" / "tmp"
    tmp_base.mkdir(parents=True, exist_ok=True)
    temp_dir = tempfile.mkdtemp(dir=tmp_base)
    temp_path = Path(temp_dir)
    try:
        # Extract tarball
        print(f"Extracting {tarball_path}...")
        with tarfile.open(tarball_path, 'r:gz') as tar:
            tar.extractall(path=temp_dir)

        # Rename stats.txt to stats_prefilter.txt
        stats_file = temp_path / 'stats.txt'
        if stats_file.exists():
            prefilter_stats = temp_path / 'stats_prefilter.txt'
            os.rename(stats_file, prefilter_stats)
            print(f"Renamed stats.txt → stats_prefilter.txt")

        ml_indexer_path = temp_path / 'ml_indexer'
        ml_indexer = load_ml_indexer_from_state(ml_indexer_path)

        file_prefix = f"{temp_path}{os.sep}"
        deep_filter_npy(
            file_prefix,
            ml_indexer,
            Component_Lower_Bounds=Component_Lower_Bounds,
            Component_Upper_Bounds=Component_Upper_Bounds,
            Oxide_Lower_Bounds=Oxide_Lower_Bounds,
            Oxide_Upper_Bounds=Oxide_Upper_Bounds,
            Mass_Upper_Bounds=Mass_Upper_Bounds,
            Bulk_Oxide_Bounds=Bulk_Oxide_Bounds,
            batch_size=batch_size,
        )

        print("Generating post-filter statistics...")
        stats_path = generate_dataset_stats(
            dataset_name=file_prefix,
            ml_indexer=ml_indexer,
            output_dir=temp_path,
            chunk_size=batch_size
        )
        os.replace(stats_path, temp_path / 'stats.txt')
        print("Created stats_postfilter (as stats.txt)")

        # Repack tarball
        print(f"Repacking filtered data into {tarball_path}...")
        with tarfile.open(tarball_path, 'w:gz') as tar:
            for file in temp_path.glob('*'):
                if file.is_file():
                    tar.add(file, arcname=file.name)

            ml_indexer_dir = temp_path / 'ml_indexer'
            if ml_indexer_dir.is_dir():
                print("Re-packing ml_indexer state directory...")
                tar.add(ml_indexer_dir, arcname='ml_indexer')

        print(f"Filtering complete: {tarball_path}")

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


def deep_filter_npy(filename, ml_indexer, Component_Lower_Bounds=None, Component_Upper_Bounds=None,
                     Oxide_Lower_Bounds=None, Oxide_Upper_Bounds=None, Mass_Upper_Bounds=None,
                     Bulk_Oxide_Bounds=None, batch_size=200_000):
    """Filter an on-disk dataset (given as a `filename` prefix, so that
    `filename + 'labels.npy'` etc. resolve to the dataset's .npy files) using
    the provided ml_indexer for lookups."""

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
    
    # Perform safe batch delete on every row-aligned array (see
    # ROW_ALIGNED_BUNDLE_ARRAYS) so derivative/free_outputs arrays stay the same
    # length as features.npy -- an untrimmed one desyncs the bundle and trips
    # shuffle_bundle_rows.
    for npy_file in ROW_ALIGNED_BUNDLE_ARRAYS:
        file_path = filename + npy_file
        if os.path.exists(file_path):
            safe_delete_batched(file_path, delete_indices)
    
    gc.collect()
    

    return delete_indices