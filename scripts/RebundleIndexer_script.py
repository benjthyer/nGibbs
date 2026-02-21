"""
Script to rebundle an ML-ready dataset with a new MLIndexer.

Usage:
    python RebundleIndexer_script.py --bundle <path> --new-indexer <path> [--output <path>]

Example:
    python RebundleIndexer_script.py \\
        --bundle ./data/MLready/110/MELTS110_ValidsetFeb13BatchCooling_Valid.tar.gz \\
        --new-indexer ./data/MLready/110/ml_indexer/ \\
        --output ./data/MLready/110/MELTS110_ValidsetFeb13BatchCooling_Valid_rebundled.tar.gz
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from typing import Optional


# Add repo root to path
repo_root = str(Path(__file__).resolve().parents[1] / 'src')
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print(repo_root)
from nMELTS.config.ml_indexer import load_ml_indexer_from_state
from nMELTS.config.ml_indexer import MLIndexer
from nMELTS.utils.file_utils import load_ml_bundle, save_ml_bundle, MLDataBundle

def main():
    parser = argparse.ArgumentParser(
        description="Rebundle an ML-ready dataset with a new MLIndexer"
    )
    parser.add_argument(
        "--bundle",
        required=True,
        type=str,
        help="Path to bundle (.tar.gz file)"
    )
    parser.add_argument(
        "--new-indexer",
        required=True,
        type=str,
        help="Path to new MLIndexer saved state directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for rebundled file (default: <input>_rebundled.tar.gz)"
    )
    
    args = parser.parse_args()
    
    bundle_path = Path(args.bundle)
    new_indexer_path = Path(args.new_indexer)
    output_path = Path(args.output) if args.output else None
    
    # Validate inputs
    if not bundle_path.exists():
        print(f"ERROR: Bundle not found: {bundle_path}")
        sys.exit(1)
    
    if not new_indexer_path.exists():
        print(f"ERROR: New indexer directory not found: {new_indexer_path}")
        sys.exit(1)
    
    print(f"Bundle: {bundle_path.name}")
    print(f"New indexer: {new_indexer_path}")
    
    # Load new indexer
    print(f"\nLoading new MLIndexer...")
    try:
        new_indexer = load_ml_indexer_from_state(str(new_indexer_path))
        print(f"  ✓ Loaded: P={new_indexer.nphases}, C={new_indexer.ncomps}, VC={new_indexer.ncompsVaried}")
    except Exception as e:
        print(f"ERROR loading indexer: {e}")
        sys.exit(1)
    
    # Rebundle
    print(f"\n{'='*70}")
    print(f"Starting rebundle operation...")
    print(f"{'='*70}\n")
    
    try:
        output = rebundle_with_new_indexer(
            bundle_path,
            new_indexer,
            output_path=output_path,
            verbose=True
        )
        
        print(f"\n{'='*70}")
        print(f"SUCCESS: Bundle rebundled and saved")
        print(f"{'='*70}")
        print(f"\nOutput: {output}")
        
    except Exception as e:
        print(f"\nERROR during rebundling: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)





def _remap_phase_array(
    old_array: np.ndarray,
    old_indexer: MLIndexer,
    new_indexer: MLIndexer,
    axis: int = 1,
    fill_value: float = 0.0
) -> np.ndarray:
    """
    Remap an array that is indexed by phases (typically axis=1, after rows).
    
    This function handles mapping phase-indexed arrays from old indexer to new indexer.
    Phases present in old but absent in new are dropped. Phases in new but absent in old
    are filled with fill_value.
    
    Parameters
    ----------
    old_array : np.ndarray
        Array with shape (n_rows, n_old_phases) or similar
    old_indexer : MLIndexer
        The old MLIndexer used to create the array
    new_indexer : MLIndexer
        The new MLIndexer to remap to
    axis : int
        Axis along which phases are indexed (default: 1, for 2D arrays)
    fill_value : float
        Value to use for new phases not in old data (default: 0.0)
    
    Returns
    -------
    np.ndarray
        Remapped array with shape (..., n_new_phases, ...)
    """
    # Create mapping from old phase names to new phase indices
    phase_mapping = {}
    for new_idx, phase_name in enumerate(new_indexer.all_phases):
        if phase_name in old_indexer.mass_phasedict:
            old_idx = old_indexer.mass_phasedict[phase_name]
            phase_mapping[old_idx] = new_idx
        # If phase not in old indexer, it will be filled with fill_value
    
    # Initialize new array
    new_shape = list(old_array.shape)
    new_shape[axis] = new_indexer.nphases
    new_array = np.full(new_shape, fill_value, dtype=old_array.dtype)
    
    # Copy data from old to new positions
    for old_idx, new_idx in phase_mapping.items():
        if axis == 1:
            new_array[:, new_idx] = old_array[:, old_idx]
        elif axis == 0:
            new_array[new_idx, :] = old_array[old_idx, :]
        else:
            raise ValueError(f"Unsupported axis: {axis}")
    
    return new_array


def _remap_component_array(
    old_array: np.ndarray,
    old_indexer: MLIndexer,
    new_indexer: MLIndexer
) -> np.ndarray:
    """
    Remap an array that is indexed by components (typically axis=1, after rows).
    
    This function handles mapping component-indexed arrays from old indexer to new indexer.
    Components are matched by phase and then by position within the phase. Components
    present in old but absent in new are dropped. Components in new but absent in old
    are filled with fill_value.
    
    Parameters
    ----------
    old_array : np.ndarray
        Array with shape (n_rows, n_old_components) or similar
    old_indexer : MLIndexer
        The old MLIndexer used to create the array
    new_indexer : MLIndexer
        The new MLIndexer to remap to

    Returns
    -------
    np.ndarray
        Remapped array with shape (..., n_new_components, ...)
    """
    
    rows = old_array.shape[0]
    columns = len(old_indexer.compositional_component_subset)
    new_arr = np.zeros((rows, columns), dtype = old_array.dtype)

    for new_phase_name, new_phase_dict in new_indexer.detail_label_indices.items():
        #print(new_phase_name)
        for comp_name, new_idx in new_phase_dict.items():
            old_idx = old_indexer.detail_label_indices[new_phase_name][comp_name]
            new_arr[:, new_idx] = old_array[:, old_idx]
            #print(f"{new_phase_name} {comp_name} at new idx {new_idx} mapped from old idx {old_idx}")
    
    return new_arr


def rebundle_with_new_indexer(
    bundle_path: Path,
    new_indexer: MLIndexer,
    output_path: Optional[Path] = None,
    verbose: bool = True
) -> Path:
    """
    Remap an ML-ready bundle to use a new MLIndexer.
    
    This function reads an existing bundle (created with an old MLIndexer), extracts
    its data, and reorganizes it to match a new MLIndexer's component/phase structure.
    Row counts are preserved. Components/phases present in old but not in new are dropped.
    Components/phases in new but not in old are filled with 0.
    
    Parameters
    ----------
    bundle_path : Path
        Path to the existing bundle (.tar.gz file)
    new_indexer : MLIndexer
        The new MLIndexer to remap to
    output_path : Path, optional
        Where to save the new bundle. If None, uses input filename in same directory.
    verbose : bool
        If True, prints progress messages
    
    Returns
    -------
    Path
        Path to the newly created bundle
    
    Raises
    ------
    FileNotFoundError
        If bundle_path does not exist
    AssertionError
        If array shapes do not match expectations
    """
    bundle_path = Path(bundle_path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")
    
    # Default output path: same directory with _rebundled suffix
    if output_path is None:
        stem = bundle_path.stem.replace('.tar', '')  # Remove .tar from double extension
        output_path = bundle_path.parent / f"{stem}_rebundled.tar.gz"
    else:
        output_path = Path(output_path)
    
    if verbose:
        print(f"\n=== Rebundling with new MLIndexer ===")
        print(f"Input bundle:  {bundle_path}")
        print(f"Output bundle: {output_path}")
    
    # Load old bundle
    if verbose:
        print(f"\nLoading old bundle...")
    old_bundle = load_ml_bundle(bundle_path)
    old_indexer = old_bundle.ml_indexer
    
    if verbose:
        print(f"Old indexer: P={old_indexer.nphases}, C={old_indexer.ncomps}, VC={old_indexer.ncompsVaried}")
        print(f"New indexer: P={new_indexer.nphases}, C={new_indexer.ncomps}, VC={new_indexer.ncompsVaried}")
    
    # Extract old arrays
    n_rows = old_bundle.features.shape[0]
    
    if verbose:
        print(f"\nRemapping arrays (n_rows={n_rows})...")
    
    # === 1. REMAP BINARY LABELS (phase-indexed) ===
    # Shape: (n_rows, P)
    if verbose:
        print(f"  - binary_labels ({old_bundle.binary_labels.shape} -> ({n_rows}, {new_indexer.nphases}))")
    new_binary_labels = _remap_phase_array(
        old_bundle.binary_labels,
        old_indexer,
        new_indexer,
        axis=1,
        fill_value=0.0
    )
    assert new_binary_labels.shape == (n_rows, new_indexer.nphases), \
        f"binary_labels shape mismatch: {new_binary_labels.shape} vs ({n_rows}, {new_indexer.nphases})"
    
    # === 2. REMAP MASS LABELS (phase-indexed) ===
    # Shape: (n_rows, P)
    if verbose:
        print(f"  - mass_labels ({old_bundle.mass_labels.shape} -> ({n_rows}, {new_indexer.nphases}))")
    new_mass_labels = _remap_phase_array(
        old_bundle.mass_labels,
        old_indexer,
        new_indexer,
        axis=1,
        fill_value=0.0
    )
    assert new_mass_labels.shape == (n_rows, new_indexer.nphases), \
        f"mass_labels shape mismatch: {new_mass_labels.shape} vs ({n_rows}, {new_indexer.nphases})"
    
    # === 3. REMAP MOLAR LABELS (phase-indexed) ===
    # Shape: (n_rows, P)
    if verbose:
        print(f"  - molar_labels ({old_bundle.molar_labels.shape} -> ({n_rows}, {new_indexer.nphases}))")
    new_molar_labels = _remap_phase_array(
        old_bundle.molar_labels,
        old_indexer,
        new_indexer,
        axis=1,
        fill_value=0.0
    )
    assert new_molar_labels.shape == (n_rows, new_indexer.nphases), \
        f"molar_labels shape mismatch: {new_molar_labels.shape} vs ({n_rows}, {new_indexer.nphases})"
    
    # === 4. REMAP LABELS (component-indexed, all components including fixed) ===
    # Shape: (n_rows, C)
    if verbose:
        print(f"  - labels ({old_bundle.labels.shape} -> ({n_rows}, {len(new_indexer.compositional_component_subset)}))")
    new_labels = _remap_component_array(
        old_bundle.labels,
        old_indexer,
        new_indexer
    )
    
    # === 5. FEATURES (unchanged) ===
    if verbose:
        print(f"  - features ({old_bundle.features.shape} unchanged)")
    new_features = np.copy(old_bundle.features)
    assert new_features.shape[0] == n_rows, f"features row count mismatch"
    
    # === 6. FREE OUTPUTS (if present) ===
    new_free_outputs = None
    if old_bundle.free_outputs is not None:
        if verbose:
            print(f"  - free_outputs ({old_bundle.free_outputs.shape} unchanged)")
        new_free_outputs = np.copy(old_bundle.free_outputs)
    
    if verbose:
        print(f"\n✓ All arrays remapped successfully")
    
    # === 7. CREATE NEW BUNDLE WITH NEW INDEXER ===
    if verbose:
        print(f"Creating new bundle with new indexer...")
    
    new_bundle = MLDataBundle()
    new_bundle.features = new_features
    new_bundle.binary_labels = new_binary_labels
    new_bundle.mass_labels = new_mass_labels
    new_bundle.molar_labels = new_molar_labels
    new_bundle.labels = new_labels
    new_bundle.free_outputs = new_free_outputs
    new_bundle.ml_indexer = new_indexer
    
    # === 8. SAVE NEW BUNDLE ===
    if verbose:
        print(f"Saving new bundle to {output_path}...")
    
    save_ml_bundle(new_bundle, str(output_path))
    
    if verbose:
        print(f"✓ Bundle saved to {output_path}")
        print(f"\n=== Rebundle Complete ===")
    
    return output_path


if __name__ == "__main__":
    main()
