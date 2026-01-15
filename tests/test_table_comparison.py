"""
Test script to compare csv projection matrices.

This test verifies that the reduced version 
is identical to the original file. 
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_compToOx_identical():
    """Test that two csv projection matrices are identical."""
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    projections_dir = project_root / 'src' / 'nMELTS' / 'config' / 'projections'
    
    # File paths
    original_file = projections_dir / 'OneHotcompToOx.csv'
    reduced_file = projections_dir / 'boolcompToOx_with_Mn_Ni.csv'
    
    # Check if files exist
    if not original_file.exists():
        raise FileNotFoundError(f"Original file not found: {original_file}")
    if not reduced_file.exists():
        raise FileNotFoundError(f"Reduced file not found: {reduced_file}")
    
    # Read both CSV files
    print(f"Reading {original_file.name}...")
    df_original = pd.read_csv(original_file, index_col=0)
    
    print(f"Reading {reduced_file.name}...")
    df_reduced = pd.read_csv(reduced_file, index_col=0)
    
    # Compare shapes
    print(f"\nOriginal shape: {df_original.shape}")
    print(f"Reduced shape: {df_reduced.shape}")
    
    if df_original.shape != df_reduced.shape:
        print(f"\n❌ FAILED: Shapes do not match!")
        print(f"  Original: {df_original.shape[0]} rows × {df_original.shape[1]} columns")
        print(f"  Reduced: {df_reduced.shape[0]} rows × {df_reduced.shape[1]} columns")
        return False
    
    # Compare column names
    if list(df_original.columns) != list(df_reduced.columns):
        print(f"\n❌ FAILED: Column names do not match!")
        print(f"  Original columns: {list(df_original.columns)}")
        print(f"  Reduced columns: {list(df_reduced.columns)}")
        
        # Find differences
        orig_cols = set(df_original.columns)
        reduced_cols = set(df_reduced.columns)
        only_in_original = orig_cols - reduced_cols
        only_in_reduced = reduced_cols - orig_cols
        
        if only_in_original:
            print(f"  Only in original: {only_in_original}")
        if only_in_reduced:
            print(f"  Only in reduced: {only_in_reduced}")
        return False
    
    # Compare index (row names)
    if list(df_original.index) != list(df_reduced.index):
        print(f"\n❌ FAILED: Row indices do not match!")
        
        # Find differences
        orig_idx = set(df_original.index)
        reduced_idx = set(df_reduced.index)
        only_in_original = orig_idx - reduced_idx
        only_in_reduced = reduced_idx - orig_idx
        
        if only_in_original:
            print(f"  Only in original ({len(only_in_original)} rows):")
            for idx in sorted(only_in_original):
                print(f"    - {idx}")
        if only_in_reduced:
            print(f"  Only in reduced ({len(only_in_reduced)} rows):")
            for idx in sorted(only_in_reduced):
                print(f"    - {idx}")
        return False
    
    # Compare values
    print("\nComparing values...")
    
    # Align dataframes to ensure same order
    df_original_sorted = df_original.sort_index()
    df_reduced_sorted = df_reduced.sort_index()
    
    # Compare values element-wise
    comparison = df_original_sorted == df_reduced_sorted
    
    # Check for NaN differences (where one is NaN and other is not)
    nan_diff_orig = df_original_sorted.isna() & ~df_reduced_sorted.isna()
    nan_diff_reduced = df_reduced_sorted.isna() & ~df_original_sorted.isna()
    
    # Find all differences
    all_diffs = ~comparison | nan_diff_orig | nan_diff_reduced
    
    if all_diffs.any().any():
        print(f"\n❌ FAILED: Values do not match!")
        
        # Report differences
        diff_count = all_diffs.sum().sum()
        print(f"  Total number of differing cells: {diff_count}")
        
        # Show first few differences
        diff_rows, diff_cols = np.where(all_diffs)
        print(f"\n  First 10 differences:")
        for i in range(min(10, len(diff_rows))):
            row_idx = df_original_sorted.index[diff_rows[i]]
            col_name = df_original_sorted.columns[diff_cols[i]]
            orig_val = df_original_sorted.loc[row_idx, col_name]
            reduced_val = df_reduced_sorted.loc[row_idx, col_name]
            print(f"    Row '{row_idx}', Column '{col_name}':")
            print(f"      Original: {orig_val}")
            print(f"      Reduced:  {reduced_val}")
        
        return False
    
    # If we get here, files are identical
    print("\n✅ SUCCESS: Files are identical!")
    print(f"  Both files have {df_original.shape[0]} rows and {df_original.shape[1]} columns")
    print(f"  All {df_original.shape[0] * df_original.shape[1]} values match")
    
    return True


if __name__ == '__main__':
    success = test_compToOx_identical()
    sys.exit(0 if success else 1)
