"""
Test bulk reconstruction to diagnose failures in the mass balance check.

This script replicates the bulk reconstruction test from MLexporter.py (lines 236-252)
and provides detailed diagnostics on failures.
"""

import sys
from pathlib import Path
import numpy as np
from collections import Counter

# Add parent directory (repository root) to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
# Add src to path so we can import modules without src prefix
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from builder.processing.BigMetaTable import BigMetaTable


def test_bulk_reconstruction(melts_obj, sample_size=10):
    """
    Test bulk reconstruction and analyze failures.
    
    Parameters:
    -----------
    melts_obj : MELTStable
        MELTS table object with processed data
    sample_size : int
        Number of random failures to display in detail
    """
    
    # Extract indexer components
    indexer = melts_obj.indexer
    ml_indexer = indexer.ml_indexer
    label_indices_comp = indexer.label_indices_comp
    label_indices = indexer.label_indices
    mass_phasedict = indexer.mass_phasedict
    mass_indices = indexer.mass_indices
    compToOxLoad = ml_indexer.compToOxLoad
    OxToEl = ml_indexer.OxToEl
    compositionally_variable_phases = indexer.compositionally_variable_phases
    MM = ml_indexer.MM
    Elkeys = ml_indexer.Elkeys
    
    # Get data arrays (assuming they exist from resampling_to_datasets)
    features = melts_obj.features
    molarlabels = melts_obj.molarlabels
    labels = melts_obj.labels
    binarylabels = melts_obj.binarylabels
    all_phases = indexer.all_phases
    
    feature_offset = len(ml_indexer.featureNames)
    
    print("="*80)
    print("BULK RECONSTRUCTION TEST")
    print("="*80)
    
    # Step 1: Calculate bulk composition from features (element moles -> oxides)
    print("\n[Step 1] Computing bulk_wt_ox from features...")
    bulk_wt_ox = (
        features[:, feature_offset:]
        @ np.linalg.inv(OxToEl[:len(Elkeys)])
    ) @ MM[:len(Elkeys), :len(Elkeys)]
    bulk_wt_ox = 100 * bulk_wt_ox / np.sum(bulk_wt_ox, axis=1).reshape(-1, 1)
    print(f"  Shape: {bulk_wt_ox.shape}")
    print(f"  Sum check (should be ~100): {bulk_wt_ox[0].sum():.4f}")
    
    # Step 2: Reconstruct bulk composition from labels
    print("\n[Step 2] Reconstructing GT_comps from labels...")
    GT_comps = np.zeros((features.shape[0], ml_indexer.ncomps))
    
    for phase in np.array(list(label_indices.keys())):
        if phase in compositionally_variable_phases:
            GT_comps[:, label_indices[phase]] = (
                molarlabels[:, mass_phasedict[phase]].reshape(-1, 1) * 
                labels[:, label_indices_comp[phase]]
            )
        else:
            GT_comps[:, label_indices[phase]] = molarlabels[:, mass_phasedict[phase]].reshape(-1, 1)
    
    print(f"  Shape: {GT_comps.shape}")
    
    # Step 3: Convert GT_comps to oxides
    print("\n[Step 3] Converting GT_comps to GTReconBulk_oxides...")
    GTReconBulk_oxides = (
        (((GT_comps @ compToOxLoad) @ OxToEl)
         @ np.linalg.inv(OxToEl[:len(Elkeys)]))
        @ MM[:len(Elkeys), :len(Elkeys)]
    )
    GTReconBulk_oxides = GTReconBulk_oxides * 100 / np.sum(GTReconBulk_oxides, axis=1, keepdims=True)
    print(f"  Shape: {GTReconBulk_oxides.shape}")
    print(f"  Sum check (should be ~100): {GTReconBulk_oxides[0].sum():.4f}")
    
    # Step 4: Find mismatches
    print("\n[Step 4] Finding mismatches...")
    mismatches = np.unique(np.where(np.round(bulk_wt_ox, 2) != np.round(GTReconBulk_oxides, 2))[0])
    
    total_samples = features.shape[0]
    n_failures = len(mismatches)
    n_successes = total_samples - n_failures
    
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples:     {total_samples:,}")
    print(f"Successful:        {n_successes:,} ({100*n_successes/total_samples:.2f}%)")
    print(f"Failed:            {n_failures:,} ({100*n_failures/total_samples:.2f}%)")
    
    if n_failures == 0:
        print("\n✓ All samples passed bulk reconstruction test!")
        return
    
    # Analyze which phases are present in failures
    print(f"\n{'='*80}")
    print(f"PHASE OCCURRENCE IN FAILURES")
    print(f"{'='*80}")
    
    phase_counter = Counter()
    for idx in mismatches:
        present_phases = [all_phases[i] for i, val in enumerate(binarylabels[idx]) if val > 0]
        phase_counter.update(present_phases)
    
    print(f"\n{'Phase':<25} {'Count':>10} {'% of Failures':>15}")
    print("-" * 55)
    for phase in all_phases:
        count = phase_counter[phase]
        pct = 100 * count / n_failures if n_failures > 0 else 0
        print(f"{phase:<25} {count:>10,} {pct:>14.2f}%")
    
    # Sample random failures for detailed inspection
    print(f"\n{'='*80}")
    print(f"DETAILED INSPECTION OF {min(sample_size, n_failures)} RANDOM FAILURES")
    print(f"{'='*80}")
    
    if n_failures > 0:
        sample_indices = np.random.choice(mismatches, size=min(sample_size, n_failures), replace=False)
        oxide_names = indexer.ml_indexer.WRkeys  # Oxide names (without Fe2O3)
        
        for i, idx in enumerate(sample_indices, 1):
            print(f"\n{'─'*80}")
            print(f"FAILURE #{i} (Row {idx})")
            print(f"{'─'*80}")
            
            # Show phases present
            present_phases = [all_phases[j] for j, val in enumerate(binarylabels[idx]) if val > 0]
            print(f"\nPhases present: {', '.join(present_phases)}")
            
            # Show features
            print(f"\nFeatures (P, T, fO2):")
            print(f"  Pressure:    {features[idx, 0]:.2f} bars")
            print(f"  Temperature: {features[idx, 1]:.2f} °C")
            print(f"  logfO2-QFM:  {features[idx, 2]:.4f}")
            
            # Show oxide comparison
            print(f"\nOxide Comparison (wt%):")
            print(f"{'Oxide':<10} {'From Features':>15} {'From Labels':>15} {'Difference':>15} {'Match':>8}")
            print("-" * 68)
            
            for j, oxide in enumerate(oxide_names):
                feat_val = bulk_wt_ox[idx, j]
                recon_val = GTReconBulk_oxides[idx, j]
                diff = feat_val - recon_val
                match = "✓" if np.round(feat_val, 2) == np.round(recon_val, 2) else "✗"
                print(f"{oxide:<10} {feat_val:>15.6f} {recon_val:>15.6f} {diff:>15.6f} {match:>8}")
            
            # Show totals
            feat_sum = bulk_wt_ox[idx].sum()
            recon_sum = GTReconBulk_oxides[idx].sum()
            print("-" * 68)
            print(f"{'TOTAL':<10} {feat_sum:>15.6f} {recon_sum:>15.6f} {feat_sum - recon_sum:>15.6f}")
            
            # Show phase masses
            print(f"\nPhase Molar Abundances:")
            for phase in present_phases:
                phase_idx = mass_phasedict[phase]
                molar_val = molarlabels[idx, phase_idx]
                print(f"  {phase:<25} {molar_val:.6f}")
    
    print(f"\n{'='*80}\n")
    
    return mismatches, phase_counter


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test bulk reconstruction on MELTS data")
    parser.add_argument("--file", type=str, help="Path to MELTS data file (without extension)", required=True)
    parser.add_argument("--samples", type=int, default=10, help="Number of random failures to display")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    print(f"Loading MELTS data from: {args.file}")
    
    # Load MELTS table
    melts = BigMetaTable(args.file)
    
    # Check if resampled data exists
    if not hasattr(melts, 'features') or melts.features is None:
        print("\nError: No resampled data found. Please run resampling_to_datasets() first.")
        sys.exit(1)
    
    # Run test
    mismatches, phase_stats = test_bulk_reconstruction(melts, sample_size=args.samples)
