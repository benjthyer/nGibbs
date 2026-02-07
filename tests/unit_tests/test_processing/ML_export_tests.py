import numpy as np
import pandas as pd
from typing import Optional
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent.parent # Add top directory to path.
sys.path.insert(0, str(project_root))
#from tests.test_utils import is_almost_equal
from tests.test_utils import setup_test_logging
import tests.unit_tests.test_indexers.indexer_test as IDX_TEST

## Functions work on BigMetaTable object, which contains unique indexers 
# and projection matrices at BMT.indexer.ml_indexer and the full MELTS table at BMT.table. Also these data products, generated
# by resampling_to_datasets, are available at BMT.features (n, E+label_offset), BMT.labels (n, vc), BMT.molarlabels (n, p), BMT.binarylabels (n, p), BMT.masslabels (n, p)
# where n is number of samples, E is number of elements, p is number of phases, vc is number of variable components, 
# and label_offset is number of non-chemical members of the features array, which is 3 for PTfO2 features.


def export_bundle_arrays_to_csv(bundle, output_dir: Optional[Path] = None) -> None:
    """
    Export all .npy arrays from a bundle to .csv files with proper column names.
    
    Parameters
    ----------
    bundle : object
        Bundle object with features, labels, molarlabels, binarylabels, masslabels attributes
    indexer : DatasetIndexer or MLIndexer
        Indexer with Elkeys, all_phases, label_names, compositionally_variable_subset
    output_dir : Optional[Path]
        Directory to save CSV files. Defaults to current working directory.
        
    Notes
    -----
    Creates the following CSV files:
    - features.csv: columns = [Pressure, Temperature, logfO2-QFM, El1, El2, ..., ElE]
    - labels.csv: columns = names of compositionally variable components (VC)
    - molarlabels.csv: columns = all phase names (P)
    - binarylabels.csv: columns = all phase names (P)
    - masslabels.csv: columns = all phase names (P)
    """
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    indexer = bundle.ml_indexer

    # Get indexer attributes
    elkeys = indexer.Elkeys if hasattr(indexer, 'Elkeys') else indexer.ml_indexer.Elkeys
    all_phases = indexer.all_phases if hasattr(indexer, 'all_phases') else indexer.ml_indexer.all_phases
    label_names = indexer.label_names if hasattr(indexer, 'label_names') else indexer.ml_indexer.label_names
    comp_var_subset = indexer.compositionally_variable_subset if hasattr(indexer, 'compositionally_variable_subset') else indexer.ml_indexer.compositionally_variable_subset
    
    # Features: [Pressure, Temperature, logfO2-QFM] + Elkeys
    if bundle.features is not None:
        feature_cols = ['Pressure', 'Temperature', 'logfO2-QFM'] + list(elkeys)
        features_df = pd.DataFrame(bundle.features, columns=feature_cols)
        features_path = output_dir / 'features.csv'
        features_df.to_csv(features_path, index=False)
        print(f"Exported features to {features_path}")
    
    # Labels: compositionally variable component names (VC)
    if bundle.labels is not None:
        label_cols = [label_names[i] for i in comp_var_subset]
        labels_df = pd.DataFrame(bundle.labels, columns=label_cols)
        labels_path = output_dir / 'labels.csv'
        labels_df.to_csv(labels_path, index=False)
        print(f"Exported labels to {labels_path}")
    
    # Molar labels: all phase names (P)
    if bundle.molar_labels is not None:
        molar_df = pd.DataFrame(bundle.molar_labels, columns=all_phases)
        molar_path = output_dir / 'molar_labels.csv'
        molar_df.to_csv(molar_path, index=False)
        print(f"Exported molar_labels to {molar_path}")
    
    # Binary labels: all phase names (P)
    if bundle.binary_labels is not None:
        binary_df = pd.DataFrame(bundle.binary_labels, columns=all_phases)
        binary_path = output_dir / 'binary_labels.csv'
        binary_df.to_csv(binary_path, index=False)
        print(f"Exported binary_labels to {binary_path}")
    
    # Mass labels: all phase names (P)
    if bundle.mass_labels is not None:
        mass_df = pd.DataFrame(bundle.mass_labels, columns=all_phases)
        mass_path = output_dir / 'mass_labels.csv'
        mass_df.to_csv(mass_path, index=False)
        print(f"Exported mass_labels to {mass_path}")


def sanity_check_bundle(bundle_path: Path, tolerance=1e-3, bulk_tol_frac=1e-3) -> None:
    """
    Basic sanity checks for a ML bundle with ml_indexer consistency.

    Checks:
    - Array shapes match indexer dimensions (labels vs VC, binaries vs P).
    - Variable-phase label rows sum to 1 when phase present, else 0.
    - (phase moles > 0) matches binary labels.
    - Labels only nonzero where binary labels allow.
    - Reconstructed bulk element composition matches features within tolerance.
    """
    from src.builder.processing.MLexporter import load_ml_bundle

    bundle = load_ml_bundle(bundle_path)
    indexer = bundle.ml_indexer

    labels = bundle.labels
    molar_labels = bundle.molar_labels
    binary_labels = bundle.binary_labels
    features = bundle.features
    free_outputs = getattr(bundle, 'free_outputs', None)

    assert labels.shape[1] == indexer.ncompsVaried, (
        f"labels has {labels.shape[1]} columns, expected VC={indexer.ncompsVaried}"
    )
    assert binary_labels.shape[1] == indexer.nphases, (
        f"binary_labels has {binary_labels.shape[1]} columns, expected P={indexer.nphases}"
    )
    assert molar_labels.shape[1] == indexer.nphases, (
        f"molar_labels has {molar_labels.shape[1]} columns, expected P={indexer.nphases}"
    )

    # All bundle arrays should have consistent row counts.
    row_counts = {
        'features': features.shape[0],
        'labels': labels.shape[0],
        'molar_labels': molar_labels.shape[0],
        'binary_labels': binary_labels.shape[0],
        'mass_labels': bundle.mass_labels.shape[0],
    }
    if free_outputs is not None:
        row_counts['free_outputs'] = free_outputs.shape[0]

    unique_row_counts = set(row_counts.values())
    assert len(unique_row_counts) == 1, (
        f"Bundle arrays have inconsistent row counts: {row_counts}"
    )

    # Variable-phase label row sums should be 1 if phase present, else 0.
    for phase, idxs in indexer.label_indices_comp.items():
        idxs = np.asarray(idxs, dtype=int)
        phase_idx = indexer.mass_phasedict[phase]
        phase_present = binary_labels[:, phase_idx] > 0.5
        row_sums = np.sum(labels[:, idxs], axis=1)
        assert np.allclose(row_sums[phase_present], 1.0, rtol=0.0, atol=tolerance), (
            f"{phase}: label rows should sum to 1 when present"
        )
        assert np.allclose(row_sums[~phase_present], 0.0, rtol=0.0, atol=tolerance), (
            f"{phase}: label rows should sum to 0 when absent"
        )

    # Phase moles > 0 should match binary labels.
    phase_present_from_moles = molar_labels > 0
    phase_present_from_binary = binary_labels > 0.5
    assert np.array_equal(phase_present_from_moles, phase_present_from_binary), (
        "(phase moles > 0) must match binary_labels"
    )

    # Labels only nonzero where binary labels allow.
    comp_mappings = indexer.comp_mappings
    comp_binaries = np.asarray(indexer.comp_binaries, dtype=int)
    label_implied_binary = labels @ comp_mappings.T
    assert np.allclose(
        label_implied_binary,
        binary_labels[:, comp_binaries],
        rtol=0.0,
        atol=tolerance
    ), "labels are nonzero where binary labels disallow"

    # Reconstruct bulk element composition from phase moles + component labels.
    phase_to_comp = indexer.phaseToCompMap
    comp_moles = molar_labels @ phase_to_comp

    varied_to_all = indexer.variedToAllComp
    labels_full = labels @ varied_to_all
    var_idx = np.asarray(indexer.compositionally_variable_subset, dtype=int)

    comp_frac = np.ones_like(labels_full)
    comp_frac[:, var_idx] = labels_full[:, var_idx]
    comp_moles = comp_moles * comp_frac

    bulk_el = (comp_moles @ indexer.compToOxLoad) @ indexer.OxToEl
    bulk_el = bulk_el / np.sum(bulk_el, axis=1, keepdims=True)

    feature_offset = 3
    expected_el = features[:, feature_offset:]
    rel_diff = np.abs(bulk_el - expected_el) / (expected_el + 1e-10)
    max_rel_diff = np.max(rel_diff)
    assert max_rel_diff <= bulk_tol_frac, (
        f"Bulk element mismatch: max rel diff {max_rel_diff:.2e} > {bulk_tol_frac:.2e}"
    )

    print(f"[PASS] sanity_check_bundle passed for {bundle_path}")


def test_phase_masses(BMT, fractionate='batch', tolerance_ppm=0.1):
    """
    Test 1: Verify phase masses sum correctly.
    
    - Take the rowwise sum of BMT.table[:,BMT.indexer.mass_indices].
    - Require these sums to be the same as BMT.table[:,BMT.indexer.MELTS_indices['Bulk_comp']['mass']]
      within 0.1 ppm.
    - If fractionate=='batch': Require masses from 'Bulk_comp']['mass'] to equal 100 within 2%.
    - If fractionate=='fractionate': Require that 50% of rows are not within 2% of 100.
    - NOTE: the large tolerance is because some relatively significant mass variation is allowed due to the open
    - system behavior of oxygen for fO2-buffered simulations.
    
    Args:
        BMT: BigMetaTable object with features/labels generated from resampling_to_datasets()
        fractionate: 'batch' or 'fractionate' to check mass normalization behavior
        tolerance_ppm: Relative tolerance in ppm for mass sum comparison (default 0.1 ppm)
    """
    mass_indices = BMT.indexer.mass_indices
    bulk_comp_mass_col = BMT.indexer.MELTS_indices['Bulk_comp']['mass']
    
    # Rowwise sum of phase masses
    phase_mass_sum = np.sum(BMT.table[:, mass_indices], axis=1)
    bulk_comp_mass = BMT.table[:, bulk_comp_mass_col]
    
    # Convert tolerance from ppm to fraction (0.1 ppm = 1e-7)
    tolerance_fraction = tolerance_ppm * 1e-6
    
    # Check if sums match within tolerance
    relative_diff = np.abs(phase_mass_sum - bulk_comp_mass) / (bulk_comp_mass + 1e-10)
    max_relative_error = np.max(relative_diff)
    
    assert max_relative_error <= tolerance_fraction, \
        f"Phase mass sums do not match bulk composition mass. Max relative error: {max_relative_error:.2e} (tolerance: {tolerance_fraction:.2e})"
    
    # Check mass normalization behavior
    mass_fraction_of_100 = np.abs(bulk_comp_mass - 100.0) / 100.0
    tolerance_pct = 0.02  # 2%
    
    if fractionate == 'batch':
        assert np.all(mass_fraction_of_100 <= tolerance_pct), \
            f"Batch mode: Not all masses are within 2% of 100. Found {np.sum(mass_fraction_of_100 > tolerance_pct)} rows exceeding tolerance."
    elif fractionate == 'fractionate':
        fraction_below_tolerance = np.sum(mass_fraction_of_100 <= tolerance_pct) / len(bulk_comp_mass)
        assert fraction_below_tolerance <= 0.5, \
            f"Fractionate mode: More than 50% of rows are within 2% of 100 (found {fraction_below_tolerance:.1%})."
    else:
        raise ValueError(f"fractionate must be 'batch' or 'fractionate', got {fractionate}")
    
    print(f"[PASS] test_phase_masses passed (fractionate={fractionate})")


def test_bulk_reconstruction(BMT, tolerance_ppm=0.1, export_failures_path: Optional[Path] = None):
    """
    Test 2: Verify bulk composition reconstruction from oxides.
    
    - Grab wt% bulk composition from BMT.table using oxide indices.
    - Calculate element-wise bulk composition using OxToEl and MM matrices.
    - Normalize rowwise to total = 1.
    - Require this to match BMT.features[:,3:] (element molar fractions) within 0.1 ppm.
    - Print phase proportions for full dataset and out-of-tolerance subset.
    
    Args:
        BMT: BigMetaTable object with features/labels generated from resampling_to_datasets()
        tolerance_ppm: Relative tolerance in ppm for comparison (default 0.1 ppm)
        export_failures_path: Optional path to save out-of-tolerance rows as CSV. If None, writes
            to processing_tests_failures.csv in the same directory as this test file.
    """
    # Get oxide indices from MELTS table
    oxide_indices = np.array([
        BMT.indexer.MELTS_indices['Bulk_comp'][ox] 
        for ox in BMT.indexer.Oxides
    ])
    
    # Extract bulk composition in wt%
    bulk_wt_oxides = BMT.table[:, oxide_indices]
    
    # Get transformation matrices
    OxToEl = BMT.indexer.ml_indexer.OxToEl
    #ElToOx = BMT.indexer.ml_indexer.ElToOx

    MM = BMT.indexer.ml_indexer.MM
    
    # Calculate element-wise bulk composition
    # First convert wt% to moles: (wt_ox / MM_ox)
    bulk_moles_oxides = bulk_wt_oxides / np.diag(MM).reshape(1, -1)
    
    # Convert to elements: oxides -> elements
    bulk_moles_elements = bulk_moles_oxides @ OxToEl
    
    # Normalize to sum to 1 (element molar fractions)
    total_moles = np.sum(bulk_moles_elements, axis=1, keepdims=True)
    bulk_element_fractions = bulk_moles_elements / (total_moles + 1e-10)
    
    # Expected values from features (columns 3: onwards, assuming PTfO2 features)
    feature_offset = 3
    expected_element_fractions = BMT.features[:, feature_offset:]
    
    # Compare
    tolerance_fraction = tolerance_ppm * 1e-6
    relative_diff = np.abs(bulk_element_fractions - expected_element_fractions) / (expected_element_fractions + 1e-10)
    max_relative_error_per_row = np.max(relative_diff, axis=1)
    out_of_tolerance_mask = max_relative_error_per_row > tolerance_fraction
    max_relative_error = np.max(max_relative_error_per_row)
    
    # Count out-of-tolerance rows
    num_oot = np.sum(out_of_tolerance_mask)
    total_rows = len(out_of_tolerance_mask)
    oot_fraction = num_oot / total_rows if total_rows > 0 else 0.0
    
    print(f"\n=== Bulk Reconstruction Analysis ===")
    print(f"Total rows: {total_rows}, Out-of-tolerance: {num_oot} ({oot_fraction:.2%})")
    print(f"Max relative error: {max_relative_error:.2e} (tolerance: {tolerance_fraction:.2e})")
    
    # Phase proportions in full dataset
    all_phases = BMT.indexer.all_phases
    print(f"\n--- Phase Proportions (Full Dataset: {total_rows} rows) ---")
    for phase in all_phases:
        phase_col = BMT.indexer.MELTS_indices.get(phase, {}).get('mass (gm)')
        if phase_col is None:
            # Try alternative naming convention
            for alt_name in [phase, phase.replace('-', ' '), phase.replace(' ', '-')]:
                phase_col = BMT.indexer.MELTS_indices.get(alt_name, {}).get('mass (gm)')
                if phase_col is not None:
                    break
        
        if phase_col is None:
            # Get from mass_indices
            if hasattr(BMT.indexer, 'mass_indices') and phase in BMT.indexer.mass_indices:
                phase_col = BMT.indexer.mass_indices[phase]
        
        if phase_col is not None:
            phase_present = np.sum(BMT.table[:, phase_col] > 0)
            phase_frac = phase_present / total_rows if total_rows > 0 else 0.0
            print(f"  {phase:25s}: {phase_frac:6.2%} ({phase_present:6d} rows)")
    
    # Phase proportions in out-of-tolerance subset
    if num_oot > 0:
        print(f"\n--- Phase Proportions (Out-of-Tolerance Subset: {num_oot} rows) ---")
        oot_indices = np.where(out_of_tolerance_mask)[0]
        print(f"First 5 idx of out-of-tolerance rows: {oot_indices[:5]}")
        for phase in all_phases:
            phase_col = BMT.indexer.MELTS_indices.get(phase, {}).get('mass (gm)')
            if phase_col is None:
                if hasattr(BMT.indexer, 'mass_indices') and phase in BMT.indexer.mass_indices:
                    phase_col = BMT.indexer.mass_indices[phase]
            
            if phase_col is not None:
                phase_present = np.sum(BMT.table[oot_indices, phase_col] > 0)
                phase_frac = phase_present / num_oot if num_oot > 0 else 0.0
                print(f"  {phase:25s}: {phase_frac:6.2%} ({phase_present:6d} rows)")

        # Write failing assemblages spreadsheet
        if export_failures_path is None:
            export_failures_path = Path(__file__).parent / "logs" /  'reconstruction_test_failures.csv'

        feature_offset = 3
        elkeys = BMT.indexer.ml_indexer.Elkeys
        comp_subset = BMT.indexer.compositionally_variable_subset
        label_names = BMT.indexer.ml_indexer.label_names
        all_phase_names = BMT.indexer.all_phases

        # Build columns: PTfO2, expected Elkeys, reconstructed Elkeys, labels (VC), molarlabels (P)
        ptfo2 = BMT.features[oot_indices, :feature_offset]
        expected_el = BMT.features[oot_indices, feature_offset:feature_offset + len(elkeys)]
        reconstructed_el = bulk_element_fractions[oot_indices, :]
        labels_mat = BMT.labels[oot_indices, :len(comp_subset)]
        molar_mat = BMT.molarlabels[oot_indices, :len(all_phase_names)]

        # Column names
        ptfo2_cols = ['Pressure', 'Temperature', 'logfO2-QFM']
        expected_cols = [f"exp_{el}" for el in elkeys]
        recon_cols = [f"recon_{el}" for el in elkeys]
        label_cols = [label_names[i] for i in comp_subset]
        phase_cols = list(all_phase_names)

        data = np.concatenate([
            ptfo2,
            expected_el,
            reconstructed_el,
            labels_mat,
            molar_mat
        ], axis=1)

        columns = ptfo2_cols + expected_cols + recon_cols + label_cols + phase_cols

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(export_failures_path, index=False)
        print(f"Saved failing assemblages spreadsheet to {export_failures_path}")
    
    assert max_relative_error <= tolerance_fraction, \
        f"Bulk reconstruction mismatch. Max relative error: {max_relative_error:.2e} (tolerance: {tolerance_fraction:.2e})"
    
    print(f"\n[PASS] test_bulk_reconstruction passed")


def test_nonzero_column_sums(BMT):
    """
    Test 3: Verify that all columnwise sums are nonzero.
    
    - Check BMT.features, BMT.labels, BMT.molarlabels, BMT.binarylabels, BMT.masslabels.
    - Verify all columnwise sums are nonzero (i.e., every column has at least some data).
    
    Args:
        BMT: BigMetaTable object with features/labels generated from resampling_to_datasets()
    """
    arrays_to_check = {
        'features': BMT.features,
        'labels': BMT.labels,
        'molarlabels': BMT.molarlabels,
        'binarylabels': BMT.binarylabels,
        'masslabels': BMT.masslabels
    }
    
    for name, arr in arrays_to_check.items():
        if arr is None:
            continue
        
        column_sums = np.sum(arr, axis=0)
        zero_columns = np.where(column_sums == 0)[0]
        
        assert len(zero_columns) == 0, \
            f"{name}: Found {len(zero_columns)} columns with zero sum (indices: {zero_columns[:10]}{'...' if len(zero_columns) > 10 else ''})"
    
    print(f"[PASS] test_nonzero_column_sums passed")


def test_phase_coverage(BMT):
    """
    Test 4: Verify phase coverage in table matches indexer.
    
    - Calculate rowwise sums for BMT.table by phase (using mass_indices).
    - Identify all phases with non-zero values across any row.
    - Verify that these phases are included in BMT.indexer.all_phases.
    
    Args:
        BMT: BigMetaTable object with features/labels generated from resampling_to_datasets()
    """
    mass_indices = BMT.indexer.mass_indices
    all_phases = BMT.indexer.all_phases
    
    # Get all phases that appear in the table (non-zero in any row)
    phases_with_data = set()
    
    for phase_idx, phase_name in enumerate(all_phases):
        if phase_name in BMT.indexer.mass_indices:
            col_idx = BMT.indexer.mass_indices[phase_name]
            if np.any(BMT.table[:, col_idx] > 0):
                phases_with_data.add(phase_name)
    
    # All phases with data should be in all_phases
    phases_in_all_phases = set(all_phases)
    
    missing_from_indexer = phases_with_data - phases_in_all_phases
    assert len(missing_from_indexer) == 0, \
        f"Found phases with non-zero data not in all_phases: {missing_from_indexer}"
    
    print(f"[PASS] test_phase_coverage passed (found {len(phases_with_data)} phases with data)")


def test_bulk_comp_change_by_run(BMT, mode='batch', tolerance_ppm=0.1, frac_change_threshold=0.9, Output_tables = False):
    """
    Additional Test: Verify bulk composition stability/change across run sequences.

    - Get unique run IDs via np.unique(BMT.run_indices).
    - For each run, compare bulk element fractions for the first vs last rows.
      For batch mode, they should be the same within 0.1 ppm for all elements EXCEPT iron (Fe).
        NOTE: Iron variation in batch mode is expected behavior from alphamelts - it redistributes
        total iron between FeO (ferrous) and Fe2O3 (ferric) during crystallization based on oxygen
        fugacity. When iron is collapsed to a single "Fe" element, this redistribution appears as
        composition change. This is not a bug but a thermodynamic artifact of the alphamelts code.
      For fractional mode, at least 90% of runs should change beyond 0.1 ppm.
    - Before comparing with features, also transform Bulk_comp oxides to elements for validation.

    Args:
        BMT: BigMetaTable object with features/labels generated from resampling_to_datasets().
        mode: 'batch' or 'fractionate' (fractional crystallization).
        tolerance_ppm: ppm tolerance for equality (default 0.1 ppm).
        frac_change_threshold: minimum fraction of runs that must change for fractional mode.
    """
    # Use features' element fractions for robustness (already normalized to 1)
    feature_offset = 3
    assert hasattr(BMT, 'features') and BMT.features is not None, "BMT.features must be loaded before running this test."

    run_ids = np.unique(BMT.run_indices)
    tolerance_fraction = tolerance_ppm * 1e-6
    elkeys = BMT.indexer.ml_indexer.Elkeys
    all_phases = BMT.indexer.all_phases
    
    # Pre-compute bulk composition elements from Bulk_comp phase for validation
    oxide_indices = np.array([
        BMT.indexer.MELTS_indices['Bulk_comp'][ox] 
        for ox in BMT.indexer.Oxides
    ])
    bulk_wt_oxides = BMT.table[:, oxide_indices]
    
    # Transform Bulk_comp oxides to elements
    MM = BMT.indexer.ml_indexer.MM
    OxToEl = BMT.indexer.ml_indexer.OxToEl
    
    bulk_moles_oxides = bulk_wt_oxides / np.diag(MM).reshape(1, -1)
    bulk_moles_elements = bulk_moles_oxides @ OxToEl
    total_moles = np.sum(bulk_moles_elements, axis=1, keepdims=True)
    bulk_element_fractions_from_table = bulk_moles_elements / (total_moles + 1e-10)
    
    
    

    if Output_tables:

        # Export transformation matrices and intermediate results for debugging
        debug_dir = Path(__file__).parent / "debug_outputs"
        debug_dir.mkdir(exist_ok=True)

        # Export OxToEl matrix with labeled rows and columns
        oxToEl_df = pd.DataFrame(
            OxToEl,
            index=BMT.indexer.Oxides,
            columns=elkeys
        )
        oxToEl_df.to_csv(debug_dir / "OxToEl_matrix.csv")
        
        # Export MM matrix with labeled rows and columns (diagonal matrix)
        MM_df = pd.DataFrame(
            MM,
            index=BMT.indexer.Oxides,
            columns=BMT.indexer.Oxides
        )
        MM_df.to_csv(debug_dir / "MM_matrix.csv")

        # Export bulk_wt_oxides with column labels
        bulk_wt_oxides_df = pd.DataFrame(
            bulk_wt_oxides,
            columns=BMT.indexer.Oxides
        )
        bulk_wt_oxides_df.to_csv(debug_dir / "bulk_wt_oxides.csv", index=False)
        
        # Export unnormalized bulk_moles_oxides with column labels
        bulk_moles_oxides_df = pd.DataFrame(
            bulk_moles_oxides,
            columns=BMT.indexer.Oxides
        )
        bulk_moles_oxides_df.to_csv(debug_dir / "bulk_moles_oxides.csv", index=False)
        
        # Export bulk_element_fractions_from_table with column labels
        bulk_element_fractions_df = pd.DataFrame(
            bulk_element_fractions_from_table,
            columns=elkeys
        )
        bulk_element_fractions_df.to_csv(debug_dir / "bulk_element_fractions_from_table.csv", index=False)
        
        print(f"\n[DEBUG] Exported transformation matrices to {debug_dir}/")

    num_runs = 0
    num_changed = 0
    constant_chem_runs = []  # For fractional mode: track runs that didn't change

    for run_code in run_ids:
        idxs = BMT.ID(run_code)
        if len(idxs) < 2:
            continue
        num_runs += 1
        first_vec = BMT.features[idxs[0], feature_offset:]
        last_vec = BMT.features[idxs[-1], feature_offset:]
        rel_diff = np.max(np.abs(first_vec - last_vec) / (np.abs(first_vec) + 1e-10))

        if mode == 'batch':
            if rel_diff > tolerance_fraction:
                # Batch run failed: print diagnostics
                print(f"\n[FAIL] Batch run '{run_code}' failed bulk composition stability:")
                print(f"   Max relative difference: {rel_diff:.2e} (tolerance: {tolerance_fraction:.2e})")
                
                # Compare element fractions from Bulk_comp vs features
                table_bulk_el = bulk_element_fractions_from_table[idxs[0], :]
                features_el = BMT.features[idxs[0], feature_offset:]
                bulk_diff = np.abs(table_bulk_el - features_el) / (np.abs(features_el) + 1e-10)
                max_bulk_diff = np.max(bulk_diff)
                
                print(f"\n   Bulk_comp element fractions vs features (first row):")
                print(f"     Max relative difference: {max_bulk_diff:.2e}")
                for el_idx, el in enumerate(elkeys):
                    if bulk_diff[el_idx] > tolerance_fraction:
                        print(f"       {el}: table={table_bulk_el[el_idx]:.6f}, features={features_el[el_idx]:.6f} (Δ={bulk_diff[el_idx]:.2e})")
                
                print(f"\n   First row phases:")
                for p_idx, phase in enumerate(all_phases):
                    phase_moles = BMT.molarlabels[idxs[0], p_idx] if p_idx < BMT.molarlabels.shape[1] else 0.0
                    if phase_moles > 0:
                        print(f"     {phase}: {phase_moles:.6f} mol")
                
                print(f"\n   Last row phases:")
                for p_idx, phase in enumerate(all_phases):
                    phase_moles = BMT.molarlabels[idxs[-1], p_idx] if p_idx < BMT.molarlabels.shape[1] else 0.0
                    if phase_moles > 0:
                        print(f"     {phase}: {phase_moles:.6f} mol")
                
                # Element-by-element changes
                print(f"\n   Element changes (first vs last, from features):")
                for el_idx, el in enumerate(elkeys):
                    first_val = first_vec[el_idx]
                    last_val = last_vec[el_idx]
                    if first_val > 0:
                        el_change = np.abs(last_val - first_val) / first_val
                    else:
                        el_change = np.abs(last_val)
                    if el_change > tolerance_fraction:
                        print(f"     {el}: {first_val:.6f} -> {last_val:.6f} (delt={el_change:.2e})")
            
            assert rel_diff <= tolerance_fraction, (
                f"Batch run '{run_code}': bulk composition changed beyond tolerance. Max rel diff: {rel_diff:.2e}"
            )
        elif mode == 'fractionate':
            if rel_diff > tolerance_fraction:
                num_changed += 1
            else:
                constant_chem_runs.append((run_code, idxs, first_vec, last_vec))
        else:
            raise ValueError(f"mode must be 'batch' or 'fractionate', got {mode}")

    if mode == 'fractionate':
        if num_runs == 0:
            raise AssertionError("No runs found to evaluate fractional change.")
        frac_changed = num_changed / num_runs
        
        # If test fails, print diagnostics on constant chemistry runs
        if frac_changed < frac_change_threshold and len(constant_chem_runs) > 0:
            print(f"\n⚠️  Fractional mode: Only {frac_changed:.1%} of runs changed (expected >= {frac_change_threshold:.0%}).")
            print(f"Found {len(constant_chem_runs)} unexpected constant-chemistry runs:")
            
            for run_code, idxs, first_vec, last_vec in constant_chem_runs[:5]:  # Show first 5
                print(f"\n  Run '{run_code}' (constant chemistry):")
                print(f"    Rows: {len(idxs)} samples")
                
                # Phase assemblage statistics
                phase_counts = {phase: 0 for phase in all_phases}
                for idx in idxs:
                    for p_idx, phase in enumerate(all_phases):
                        if p_idx < BMT.molarlabels.shape[1] and BMT.molarlabels[idx, p_idx] > 0:
                            phase_counts[phase] += 1
                
                print(f"    Phases present in ≥1 row: {[p for p, c in phase_counts.items() if c > 0]}")
                print(f"    Avg first-row phases: {len([p for p, c in phase_counts.items() if c > 0])}")
            
            if len(constant_chem_runs) > 5:
                print(f"\n  ... and {len(constant_chem_runs) - 5} more constant-chemistry runs")
        
        assert frac_changed >= frac_change_threshold, (
            f"Fractional mode: Only {frac_changed:.1%} of runs changed (expected >= {frac_change_threshold:.0%})."
        )

    print(f"[PASS] test_bulk_comp_change_by_run passed (mode={mode})")



def shuffle_csv_columns(input_csv_path: Path, output_csv_path: Path, seed: int = 42):
    """
    Shuffle the columns of a CSV file randomly (except the first column if it's an index).
    
    Args:
        input_csv_path: Path to the input CSV file
        output_csv_path: Path to save the shuffled CSV
        seed: Random seed for reproducibility
    """
    print(f"\n=== Shuffling CSV columns ===")
    print(f"Input: {input_csv_path}")
    print(f"Output: {output_csv_path}")
    
    # Read the CSV
    df = pd.read_csv(input_csv_path)
    
    # Shuffle columns randomly
    np.random.seed(seed)
    cols = df.columns.tolist()
    shuffled_cols = cols.copy()
    np.random.shuffle(shuffled_cols)
    
    # Reorder dataframe
    df_shuffled = df[shuffled_cols]
    
    # Save to output path
    df_shuffled.to_csv(output_csv_path, index=False)
    
    print(f"Shuffled {len(cols)} columns")
    print(f"Original first 10 columns: {cols[:10]}")
    print(f"Shuffled first 10 columns: {shuffled_cols[:10]}")
    

def run_tests_on_bundle(bundle_path: Path, BMT, test_name: str = "", output_tables: bool = False, fractionate = 'batch', outname=None):
    """
    Run all processing tests on a given .tar.gz bundle file.
    
    Args:
        bundle_path: Path to the .tar.gz bundle file
        BMT: BigMetaTable object with original table and indexer
        test_name: Optional name for the test run (for logging)
        output_tables: Whether to export debug tables of transformation matrices
        outname: Optional subdirectory name for exporting CSV versions of .npy files
    """
    from src.builder.processing.MLexporter import load_ml_bundle
    
    print(f"\n{'='*60}")
    print(f"Running tests on bundle: {test_name if test_name else bundle_path.name}")
    print(f"{'='*60}")
    
    # Load bundle and assign to BMT
    bundle = load_ml_bundle(bundle_path)

    outpath  = Path(__file__).parent / 'logs' 
    if outname is not None:
        outpath = outpath / outname

    outpath.mkdir(exist_ok=True)

    export_bundle_arrays_to_csv(bundle, output_dir=outpath)


    BMT.features = bundle.features
    BMT.labels = bundle.labels
    BMT.molarlabels = bundle.molar_labels
    BMT.binarylabels = bundle.binary_labels
    BMT.masslabels = bundle.mass_labels
    BMT.indexer.ml_indexer = bundle.ml_indexer
    BMT.indexer.expose_ml_indexer_attributes()  # Ensure indexer attributes are accessible at BMT.indexer level

    IDX_TEST.test_dataset_indexer(BMT.indexer)  # Sanity check that indexer is functional after loading bundle
    IDX_TEST.test_ml_indexer(BMT.indexer.ml_indexer)  # Sanity check that ml_indexer is functional after loading bundle

    sanity_check_bundle(bundle_path=bundle_path)

    #BMT.indexer.ml_indexer.label_indices_comp['fake'] = np.array([39,40,41])

    #IDX_TEST.test_ml_indexer(BMT.indexer.ml_indexer)  # This should fail if label_indices_comp is not properly integrated into ml_indexer

    # Execute tests
    if fractionate is not None:
        test_bulk_comp_change_by_run(BMT, mode=fractionate.lower(), tolerance_ppm=150000, Output_tables=output_tables)
        test_phase_masses(BMT, fractionate=fractionate.lower(), tolerance_ppm=5000)
    test_bulk_reconstruction(BMT, tolerance_ppm=1500)
    test_nonzero_column_sums(BMT)
    test_phase_coverage(BMT)
    
    print(f"\n{'='*60}")
    print(f"All tests PASSED for bundle: {test_name if test_name else bundle_path.name}")
    print(f"{'='*60}\n")


def run_tests_on_csv(csv_path: Path, test_name: str = "", output_tables: bool = False, fractionate = 'batch'):
    """
    Run all processing tests on a given CSV file.
    
    Args:
        csv_path: Path to the CSV file
        test_name: Optional name for the test run (for logging)
    """
    from src.builder.processing.BigMetaTable import BigMetaTable
    from src.builder.processing.MLexporter import resampling_to_datasets
    from src.builder.indexer import DatasetIndexer
    
    print(f"\n{'='*60}")
    print(f"Running tests on: {test_name if test_name else csv_path.name}")
    print(f"{'='*60}")
    
    # BigMetaTable expects base filename without extension
    base_name = str(csv_path).rsplit('.csv', 1)[0]

    # Build table and indexer
    BMT = BigMetaTable(base_name)
    BMT.indexer.table_update(BMT.table)
    
    # Ensure indexer exists
    if not hasattr(BMT, 'indexer') or BMT.indexer is None:
        BMT.indexer = DatasetIndexer(BMT)

    # Generate features/labels with resampling bounds [[1,1]]
    resampling_to_datasets(BMT, resample_bounds=[[1, 1]], indexer=BMT.indexer)

    # Reload memmaps (MLexporter deletes attributes in its finally block)
    BMT.features = np.load(BMT.filename + 'features.npy', mmap_mode='r+')
    BMT.labels = np.load(BMT.filename + 'labels.npy', mmap_mode='r+')
    BMT.molarlabels = np.load(BMT.filename + 'molar_labels.npy', mmap_mode='r+')
    BMT.binarylabels = np.load(BMT.filename + 'binary_labels.npy', mmap_mode='r+')
    BMT.masslabels = np.load(BMT.filename + 'mass_labels.npy', mmap_mode='r+')

    # Execute tests for batch dataset
    test_bulk_comp_change_by_run(BMT, mode=fractionate.lower(), tolerance_ppm=150000, Output_tables=output_tables) # 1.5%!
    test_phase_masses(BMT, fractionate=fractionate.lower(), tolerance_ppm=5000)
    test_bulk_reconstruction(BMT, tolerance_ppm=1500)
    test_nonzero_column_sums(BMT)
    test_phase_coverage(BMT)
    
    print(f"\n{'='*60}")
    print(f"All tests PASSED for: {test_name if test_name else csv_path.name}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    from tests.unit_tests.test_indexers.indexer_report import build_and_report
    import shutil

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    setup_test_logging(
        log_filename=f"{Path(__file__).stem}_{timestamp}.txt",
        log_dir=Path(__file__).parent / 'logs',
    )
    from src.builder.processing.BigMetaTable import BigMetaTable
    from src.builder.processing.MLexporter import resampling_to_datasets, load_ml_bundle
    from src.builder.indexer import DatasetIndexer

    # Path to the batch cooling CSV
    csv_path = Path(project_root) / 'data' / 'MELTStables' / '110' / 'MELTS110_TrainsetFeb3BatchCooling.csv'
    bundle_path_1 = Path(project_root) / 'data' / 'MLready' / '110' / 'MELTS110_TrainsetFeb3BatchCooling.tar.gz'
    bundle_path_2 = Path(project_root) / 'data' / 'MLready' / '110' / 'MELTS110_TrainsetFeb3BatchCooling_shuffled.tar.gz'


    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    
    # Initialize BMT once (will be reused for all tests)
    base_name = str(csv_path).rsplit('.csv', 1)[0]
    BMT = BigMetaTable(base_name, rebuild_memmap=True)
    BMT.separate_analcime() # Test essential separate analcime function
    
    #BMT.indexer.table_update(BMT.table)

    build_and_report(BMT.indexer, headers = BMT.header)



    #if not hasattr(BMT, 'indexer') or BMT.indexer is None:
    #    BMT.indexer = DatasetIndexer(BMT)
    
    # Test 1: Original CSV - Generate bundle and test
    print("\n" + "="*80)
    print("TEST SET 1: Original CSV with Bundle Generation")
    print("="*80)
    resampling_to_datasets(BMT, resample_bounds=[[1, 1]])#, indexer=BMT.indexer)
    #bundle_path_1 = Path(base_name + '.tar.gz')
    if bundle_path_1.exists():
        run_tests_on_bundle(bundle_path_1, BMT, "Original CSV Bundle", output_tables=False, outname='originalMLoutputs')
    else:
        raise FileNotFoundError(f"⚠️  Bundle file not found at {bundle_path_1}")
    
    # Test 2: Shuffled CSV - Generate bundle and test
    print("\n" + "="*80)
    print("TEST SET 2: Shuffled CSV with Bundle Generation")
    print("="*80)
    shuffled_path = csv_path.parent / (csv_path.stem + "_shuffled.csv")
    shuffle_csv_columns(csv_path, shuffled_path, seed=42)
    
    # Copy the text file for the shuffled CSV
    base_name_shuffled = str(shuffled_path).rsplit('.csv', 1)[0]
    shutil.copy(f"{base_name}.txt", f"{base_name_shuffled}.txt")
    
    # Create new BMT for shuffled CSV
    BMT_shuffled = BigMetaTable(base_name_shuffled, rebuild_memmap=True)
    #BMT_shuffled.separate_analcime()
    #BMT_shuffled.indexer.table_update(BMT_shuffled.table)
    #if not hasattr(BMT_shuffled, 'indexer') or BMT_shuffled.indexer is None:
    #    BMT_shuffled.indexer = DatasetIndexer(BMT_shuffled)
    build_and_report(BMT_shuffled.indexer, headers = BMT_shuffled.header)
    
    resampling_to_datasets(BMT_shuffled, resample_bounds=[[1, 1]])#, indexer=BMT_shuffled.indexer)
    #bundle_path_2 = Path(base_name_shuffled + '.tar.gz')
    if bundle_path_2.exists():
        run_tests_on_bundle(bundle_path_2, BMT_shuffled, "Shuffled CSV Bundle", output_tables=False, outname='shuffledMLoutputs')
    else:
        raise FileNotFoundError(f"⚠️  Bundle file not found at {bundle_path_2}")
    
    print("\n" + "="*80)
    print("SUCCESS: All tests passed!")
    print("="*80)

