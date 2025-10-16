"""
Example usage of the new create_3D_memmaps method.

This script demonstrates how to use the new method to create numpy memmap datasets
with [I,T,O] geometry combining the functionality of filter_jumps_with_phase_matrix()
and resampling_to_datasets() in a single loop.
"""

from BackEnds.BigMetaTableLibrary import BigMetaTable

def main():
    # Initialize BigMetaTable (replace with your actual data file)
    melts_data = BigMetaTable("your_data_file", read_dir="path/to/data/")
    
    # Create 3D memmaps - single call creates all arrays at once
    print("Creating 3D memmaps...")
    melts_data.create_3D_memmaps(
        output_prefix="threeD_data",
        T=800,  # 800 temperature steps
        thresholds=None  # Use default thresholds
    )
    
    # Access the memmap arrays directly from the object
    print(f"\nBinary labels shape: {melts_data.binarylabels.shape}")
    print(f"Mass labels shape: {melts_data.masslabels.shape}")
    print(f"Molar labels shape: {melts_data.molarlabels.shape}")
    print(f"Features shape: {melts_data.features.shape}")
    print(f"Labels shape: {melts_data.labels.shape}")
    print(f"Temperatures shape: {melts_data.temperatures.shape}")
    
    # Example: Access data for the first ID
    print(f"\nFirst ID temperature range: {melts_data.temperatures[0, 0]:.1f}°C to {melts_data.temperatures[0, -1]:.1f}°C")
    print(f"First ID binary phase data shape: {melts_data.binarylabels[0].shape}")
    print(f"First ID features shape: {melts_data.features[0].shape}")
    print(f"First ID labels shape: {melts_data.labels[0].shape}")
    
    # Example: Check phase presence for first ID
    print(f"\nPhase presence for first ID (first 5 temperature steps):")
    print(f"Binary labels: {melts_data.binarylabels[0, :5, :5]}")  # First 5 phases, first 5 temp steps
    print(f"Mass fractions: {melts_data.masslabels[0, :5, :5]}")  # First 5 phases, first 5 temp steps

if __name__ == "__main__":
    main()
