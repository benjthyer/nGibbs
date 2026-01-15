"""
Script to create a one-hot matrix from compToOx_with_Mn_Ni.csv.

Converts all nonzero values to 1 and zero values to 0,
creating a binary (one-hot) representation of the component-to-oxide mapping.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_onehot_compToOx():
    """Create one-hot matrix from compToOx_with_Mn_Ni.csv."""
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    projections_dir = project_root / 'src' / 'nMELTS' / 'config' / 'projections'
    
    # Input and output file paths
    input_file = projections_dir / 'compToOx_with_Mn_Ni.csv'
    output_file = projections_dir / 'OneHotcompToOx.csv'
    
    # Check if input file exists
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Reading {input_file.name}...")
    df = pd.read_csv(input_file, index_col=0)
    
    print(f"Original shape: {df.shape}")
    print(f"Creating one-hot matrix (nonzero -> 1, zero -> 0)...")
    
    # Create one-hot matrix: 1 for nonzero values, 0 for zero values
    df_onehot = (df != 0).astype(int)
    
    # Save to CSV
    print(f"Saving to {output_file.name}...")
    df_onehot.to_csv(output_file)
    
    print(f"✅ Success! One-hot matrix saved to {output_file}")
    print(f"  Shape: {df_onehot.shape}")
    print(f"  Total nonzero entries in original: {(df != 0).sum().sum()}")
    print(f"  Total ones in one-hot matrix: {df_onehot.sum().sum()}")
    
    return df_onehot


if __name__ == '__main__':
    create_onehot_compToOx()
