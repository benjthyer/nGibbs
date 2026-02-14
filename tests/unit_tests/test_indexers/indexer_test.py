"""
Comprehensive unit tests for MLIndexer and DatasetIndexer.

This module provides test functions that validate all attributes and structural
properties of MLIndexer and DatasetIndexer instances.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys
from datetime import datetime
import tempfile


project_root = Path(__file__).parent.parent.parent.parent # Add top directory to path.
sys.path.insert(0, str(project_root))
from tests.test_utils import is_almost_equal, setup_test_logging

def save_failed_matrix(matrix, matrix_name, test_name):
    """
    Save a failed matrix to logs folder for debugging.
    
    Parameters
    ----------
    matrix : np.ndarray or other
        The matrix that failed the test
    matrix_name : str
        Name of the matrix (e.g., 'PxSpTransform', 'compToOx')
    test_name : str
        Name of the test that failed (e.g., 'test_ml_indexer')
    """
    log_dir = Path.cwd() / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    if isinstance(matrix, np.ndarray):
        filename = log_dir / f'FAILED_{test_name}_{matrix_name}.csv'
        try:
            pd.DataFrame(matrix).to_csv(filename, index=False, header=False)
            print(f"\n⚠️  Failed matrix saved to: {filename}")
        except Exception as e:
            print(f"\n⚠️  Could not save matrix: {e}")
    else:
        # For non-ndarray types, save repr
        filename = log_dir / f'FAILED_{test_name}_{matrix_name}.txt'
        try:
            with open(filename, 'w') as f:
                f.write(f"Type: {type(matrix)}\n")
                f.write(f"Value:\n{repr(matrix)}\n")
            print(f"\n⚠️  Failed value saved to: {filename}")
        except Exception as e:
            print(f"\n⚠️  Could not save value: {e}")


def test_ml_indexer(indexer):
    """
    Comprehensive test suite for MLIndexer instance.
    
    Validates all attributes, dimensions, data types, and mathematical relationships
    defined in README_MLIndexer.md.
    
    Parameters
    ----------
    indexer : MLIndexer
        An initialized MLIndexer instance to test
        
    Raises
    ------
    AssertionError
        If any test condition fails
    """
    
    # Extract dimension notation for readability
    C = indexer.ncomps
    P = indexer.nphases
    VP = len(indexer.compositionally_variable_phases)
    VC = indexer.ncompsVaried
    E = len(indexer.Elkeys)
    O = len(indexer.Oxides)
    WR = len(indexer.WRkeys)
    
    print(f"\n=== MLIndexer Test Suite ===")
    print(f"Dimensions: C={C}, P={P}, VP={VP}, VC={VC}, E={E}, O={O}, WR={WR}")
    print("=" * 50)
    
    # ========================================================================
    # 1. LABEL NAMES AND INDICES TESTS
    # ========================================================================
    print("\n[1] Testing label_names...")
    assert isinstance(indexer.label_names, list), "label_names must be a list"
    assert len(indexer.label_names) == C, f"len(label_names)={len(indexer.label_names)} != C={C}"
    assert all(isinstance(name, str) for name in indexer.label_names), "All label_names must be strings"
    print(f"✓ label_names: len={len(indexer.label_names)}")
    
    # ========================================================================
    print("\n[2] Testing label_indices...")
    assert isinstance(indexer.label_indices, dict), "label_indices must be a dict"
    assert len(indexer.label_indices) == P, f"len(label_indices)={len(indexer.label_indices)} != P={P}"
    
    # Verify indices are contiguous and cover all components
    all_indices = []
    for phase_name, indices in indexer.label_indices.items():
        assert isinstance(indices, (list, np.ndarray)), f"Indices for {phase_name} must be list or ndarray"
        all_indices.extend(indices)
    
    all_indices_sorted = sorted(all_indices)
    assert len(all_indices) == C, f"Sum of all indices lengths {len(all_indices)} != C={C}"
    assert all_indices_sorted == list(range(C)), "Indices must be contiguous from 0 to C-1"
    assert max(all_indices_sorted) == C - 1, f"Max index {max(all_indices_sorted)} != C-1={C-1}"
    print(f"✓ label_indices: phases={len(indexer.label_indices)}, total_components={len(all_indices)}")
    
    # ========================================================================
    print("\n[3] Testing label_indices_comp...")
    assert isinstance(indexer.label_indices_comp, dict), "label_indices_comp must be a dict"
    
    all_varied_indices = []
    for phase_name, indices in indexer.label_indices_comp.items():
        assert isinstance(indices, np.ndarray), f"Indices for {phase_name} must be ndarray"
        all_varied_indices.extend(indices)
    
    all_varied_indices_sorted = sorted(all_varied_indices)
    assert len(all_varied_indices) == VC, f"Sum of all varied indices {len(all_varied_indices)} != VC={VC}"
    if VC > 0:
        assert all_varied_indices_sorted == list(range(VC)), "Varied indices must be contiguous from 0 to VC-1"
        assert max(all_varied_indices_sorted) == VC - 1, f"Max varied index {max(all_varied_indices_sorted)} != VC-1={VC-1}"
    print(f"✓ label_indices_comp: phases={len(indexer.label_indices_comp)}, total_varied_components={len(all_varied_indices)}")
    
    # ========================================================================
    print("\n[4] Testing comp_map (backward compatibility)...")
    assert indexer.comp_map.keys() == indexer.label_indices_comp.keys(),  "comp_map must be identical to label_indices_comp"
    for key in indexer.comp_map:
        np.testing.assert_array_equal(indexer.comp_map[key], indexer.label_indices_comp[key])
    print(f"✓ comp_map: identical to label_indices_comp")
    
    # ========================================================================
    print("\n[5] Testing detail_label_indices...")
    assert isinstance(indexer.detail_label_indices, dict), "detail_label_indices must be a dict"
    assert len(indexer.detail_label_indices) == VP, f"len(detail_label_indices)={len(indexer.detail_label_indices)} != VP={VP}"
    
    all_detail_indices = []
    for phase_name, comp_dict in indexer.detail_label_indices.items():
        assert isinstance(comp_dict, dict), f"detail_label_indices[{phase_name}] must be dict"
        assert len(comp_dict) > 1, f"detail_label_indices[{phase_name}] must have >1 entry"
        all_detail_indices.extend(comp_dict.values())
    
    all_detail_indices_sorted = sorted(all_detail_indices)
    assert len(all_detail_indices) == VC, f"Sum of all detail indices {len(all_detail_indices)} != VC={VC}"
    if VC > 0:
        assert all_detail_indices_sorted == list(range(VC)), "Detail indices must be contiguous"
        assert max(all_detail_indices_sorted) == VC - 1, f"Max detail index != VC-1={VC-1}"
    print(f"✓ detail_label_indices: phases={len(indexer.detail_label_indices)}, total_varied_components={len(all_detail_indices)}")
    
    # ========================================================================
    # 2. PHASE ORGANIZATION TESTS
    # ========================================================================
    print("\n[6] Testing all_phases...")
    assert isinstance(indexer.all_phases, list), "all_phases must be a list"
    assert len(indexer.all_phases) == P, f"len(all_phases)={len(indexer.all_phases)} != P={P}"
    assert all(isinstance(p, str) for p in indexer.all_phases), "All phase names must be strings"
    assert 'melts-liquid' in indexer.all_phases, "melts-liquid must be in all_phases"
    #assert indexer.all_phases[-1] == 'melts-liquid', "melts-liquid must be last in all_phases"
    print(f"✓ all_phases: len={len(indexer.all_phases)}")
    
    # ========================================================================
    print("\n[7] Testing compositionally_variable_phases...")
    assert isinstance(indexer.compositionally_variable_phases, list), "compositionally_variable_phases must be a list"
    assert len(indexer.compositionally_variable_phases) == VP, f"len(compositionally_variable_phases)={len(indexer.compositionally_variable_phases)} != VP={VP}"
    for phase in indexer.compositionally_variable_phases:
        assert phase in indexer.all_phases, f"Variable phase {phase} not in all_phases"
        assert len(indexer.label_indices[phase]) > 1, f"Variable phase {phase} must have >1 component"
    print(f"✓ compositionally_variable_phases: len={len(indexer.compositionally_variable_phases)}")
    
    # ========================================================================
    print("\n[8] Testing mass_phasedict...")
    assert isinstance(indexer.mass_phasedict, dict), "mass_phasedict must be a dict"
    assert len(indexer.mass_phasedict) == P, f"len(mass_phasedict)={len(indexer.mass_phasedict)} != P={P}"
    mass_dict_values = sorted(indexer.mass_phasedict.values())
    assert mass_dict_values == list(range(P)), f"mass_phasedict values must be 0 to P-1"
    for phase in indexer.all_phases:
        assert phase in indexer.mass_phasedict, f"Phase {phase} missing from mass_phasedict"
    print(f"✓ mass_phasedict: len={len(indexer.mass_phasedict)}")
    
    # ========================================================================
    print("\n[9] Testing comp_phasedict...")
    assert isinstance(indexer.comp_phasedict, dict), "comp_phasedict must be a dict"
    assert len(indexer.comp_phasedict) == VP, f"len(comp_phasedict)={len(indexer.comp_phasedict)} != VP={VP}"
    for phase in indexer.comp_phasedict.keys():
        assert phase in indexer.compositionally_variable_phases, f"Phase {phase} in comp_phasedict but not in compositionally_variable_phases"
    comp_dict_values = sorted(indexer.comp_phasedict.values())
    assert comp_dict_values == list(range(VP)), f"comp_phasedict values must be 0 to VP-1"
    print(f"✓ comp_phasedict: len={len(indexer.comp_phasedict)}")
    
    # ========================================================================
    # 3. SIZE COUNTERS TESTS
    # ========================================================================
    print("\n[10] Testing size counters...")
    assert indexer.ncomps == C, f"ncomps={indexer.ncomps} != C={C}"
    assert indexer.ncompsVaried == VC, f"ncompsVaried={indexer.ncompsVaried} != VC={VC}"
    assert indexer.nphases == P, f"nphases={indexer.nphases} != P={P}"
    assert C >= VC, f"Total components C={C} must be >= varied components VC={VC}"
    assert P >= VP, f"Total phases P={P} must be >= variable phases VP={VP}"
    print(f"✓ Size counters: C={C}, P={P}, VC={VC}, VP={VP}")
    
    # ========================================================================
    # 4. COMPONENT-OXIDE TRANSFORMATION TESTS
    # ========================================================================
    print("\n[11] Testing compToOxLoad...")
    try:
        assert isinstance(indexer.compToOxLoad, np.ndarray), "compToOxLoad must be ndarray"
        assert indexer.compToOxLoad.dtype == np.float32, f"compToOxLoad dtype={indexer.compToOxLoad.dtype} != float32"
        assert indexer.compToOxLoad.shape == (C, O), f"compToOxLoad shape {indexer.compToOxLoad.shape} != (C={C}, O={O})"
        print(f"✓ compToOxLoad: shape={indexer.compToOxLoad.shape}, dtype={indexer.compToOxLoad.dtype}")
    except AssertionError as e:
        save_failed_matrix(indexer.compToOxLoad, "compToOxLoad", "test_ml_indexer")
        raise
    
    # ========================================================================
    print("\n[12] Testing PxSpTransform...")
    try:
        assert isinstance(indexer.PxSpTransform, np.ndarray), "PxSpTransform must be ndarray"
        assert indexer.PxSpTransform.dtype == np.float32, f"PxSpTransform dtype != float32"
        assert indexer.PxSpTransform.shape == (C, C), f"PxSpTransform shape {indexer.PxSpTransform.shape} != (C={C}, C={C})"
        
        # Test invertibility (The determinant approaches zero when correct, because there is a redundant dimension (closure) encoded within the transform)
        # det = np.linalg.det(indexer.PxSpTransform)
        # assert abs(det) > 1e-6, f"PxSpTransform determinant {det} is near zero; matrix may not be invertible"
        try:
            invMat = np.linalg.inv(indexer.PxSpTransform)
            print(f"✓ PxSpTransform: shape={indexer.PxSpTransform.shape})")
        except np.linalg.LinAlgError:
            raise AssertionError("PxSpTransform is not invertible")
        ok, agree_mask = is_almost_equal(np.linalg.inv(invMat), indexer.PxSpTransform, abs_tol=1E-6) #(32 bit floating point cannot be better than ~ 1E-7)
        if not ok:
            save_failed_matrix(indexer.PxSpTransform, "PxSpTransform", "test_ml_indexer")
            save_failed_matrix(agree_mask.astype(np.int32), "PxSpTransform_agreeMask", "test_ml_indexer")
            raise AssertionError("PxSpTransform is not reversibly invertible!")

    except AssertionError as e:
        save_failed_matrix(indexer.PxSpTransform, "PxSpTransform", "test_ml_indexer")
        raise
    
    # ========================================================================
    print("\n[13] Testing compToOx...")
    try:
        assert isinstance(indexer.compToOx, np.ndarray), "compToOx must be ndarray"
        assert indexer.compToOx.dtype == np.float32, f"compToOx dtype != float32"
        assert indexer.compToOx.shape == (C, O), f"compToOx shape {indexer.compToOx.shape} != (C={C}, O={O})"
        
        # Verify compToOx = inv(PxSpTransform) @ compToOxLoad
        expected_compToOx = np.linalg.inv(indexer.PxSpTransform) @ indexer.compToOxLoad
        np.testing.assert_allclose(indexer.compToOx, expected_compToOx.astype(np.float32), rtol=1e-5, 
                                   err_msg="compToOx != inv(PxSpTransform) @ compToOxLoad")
        print(f"✓ compToOx: shape={indexer.compToOx.shape}, correctly computed from PxSpTransform")
    except AssertionError as e:
        save_failed_matrix(indexer.compToOx, "compToOx", "test_ml_indexer")
        raise
    
    # ========================================================================
    print("\n[14] Testing boolTransCompToOx...")
    try:
        assert isinstance(indexer.boolTransCompToOx, np.ndarray), "boolTransCompToOx must be ndarray"
        assert indexer.boolTransCompToOx.dtype in [np.int32, np.int64, int], f"boolTransCompToOx dtype must be int"
        assert indexer.boolTransCompToOx.shape[0] == C, f"boolTransCompToOx rows {indexer.boolTransCompToOx.shape[0]} != C={C}"
        assert np.all((indexer.boolTransCompToOx == 0) | (indexer.boolTransCompToOx == 1)), "boolTransCompToOx must be binary (0 or 1)"
        print(f"✓ boolTransCompToOx: shape={indexer.boolTransCompToOx.shape}, dtype={indexer.boolTransCompToOx.dtype}")
    except AssertionError as e:
        save_failed_matrix(indexer.boolTransCompToOx, "boolTransCompToOx", "test_ml_indexer")
        raise
    
    # ========================================================================
    # 5. OXIDE-ELEMENT TRANSFORMATION TESTS
    # ========================================================================
    print("\n[15] Testing OxToEl...")
    try:
        assert isinstance(indexer.OxToEl, np.ndarray), "OxToEl must be ndarray"
        assert indexer.OxToEl.dtype == np.float32, f"OxToEl dtype != float32"
        assert indexer.OxToEl.shape == (O, E), f"OxToEl shape {indexer.OxToEl.shape} != (O={O}, E={E})"
        print(f"✓ OxToEl: shape={indexer.OxToEl.shape}, dtype={indexer.OxToEl.dtype}")
    except AssertionError as e:
        save_failed_matrix(indexer.OxToEl, "OxToEl", "test_ml_indexer")
        raise
    
    # ========================================================================
    print("\n[16] Testing ElToOx...")
    try:
        assert isinstance(indexer.ElToOx, np.ndarray), "ElToOx must be ndarray"
        assert indexer.ElToOx.dtype == np.float32, f"ElToOx dtype != float32"
        assert indexer.ElToOx.shape == (E, E), f"ElToOx shape {indexer.ElToOx.shape} != (E={E}, E={E})"
        
        # Verify ElToOx = inv(OxToEl[:E, :])
        expected_ElToOx = np.linalg.inv(indexer.OxToEl[:E, :])
        np.testing.assert_allclose(indexer.ElToOx, expected_ElToOx.astype(np.float32), rtol=1e-5,
                                   err_msg="ElToOx != inv(OxToEl[:E, :])")
        print(f"✓ ElToOx: shape={indexer.ElToOx.shape}, correctly computed from OxToEl")
    except AssertionError as e:
        save_failed_matrix(indexer.ElToOx, "ElToOx", "test_ml_indexer")
        raise
    
    # ========================================================================
    # 6. MOLAR MASS MATRICES TESTS
    # ========================================================================
    print("\n[17] Testing MM...")
    try:
        assert isinstance(indexer.MM, np.ndarray), "MM must be ndarray"
        assert indexer.MM.dtype == np.float32, f"MM dtype != float32"
        assert indexer.MM.shape == (O, O), f"MM shape {indexer.MM.shape} != (O={O}, O={O})"
        
        # Verify diagonal matrix
        is_diagonal = np.allclose(indexer.MM, np.diag(np.diag(indexer.MM)))
        assert is_diagonal, "MM must be a diagonal matrix"
        print(f"✓ MM: shape={indexer.MM.shape}, is diagonal matrix")
    except AssertionError as e:
        save_failed_matrix(indexer.MM, "MM", "test_ml_indexer")
        raise
    
    # ========================================================================
    print("\n[18] Testing Minv...")
    try:
        assert isinstance(indexer.Minv, np.ndarray), "Minv must be ndarray"
        assert indexer.Minv.dtype == np.float32, f"Minv dtype != float32"
        assert indexer.Minv.shape == (O, O), f"Minv shape {indexer.Minv.shape} != (O={O}, O={O})"
        
        # Verify diagonal and inverse relationship
        is_diagonal = np.allclose(indexer.Minv, np.diag(np.diag(indexer.Minv)))
        assert is_diagonal, "Minv must be a diagonal matrix"
        
        product = indexer.MM @ indexer.Minv
        np.testing.assert_allclose(product, np.eye(O, dtype=np.float32), rtol=1e-5,
                                   err_msg="MM @ Minv != I")
        print(f"✓ Minv: shape={indexer.Minv.shape}, is diagonal, MM @ Minv = I")
    except AssertionError as e:
        save_failed_matrix(indexer.Minv, "Minv", "test_ml_indexer")
        raise
    
    # ========================================================================
    print("\n[19] Testing Mtot...")
    try:
        assert isinstance(indexer.Mtot, np.ndarray), "Mtot must be ndarray"
        assert indexer.Mtot.dtype == np.float32, f"Mtot dtype != float32"
        assert indexer.Mtot.shape == (O, 1), f"Mtot shape {indexer.Mtot.shape} != (O={O}, 1)"
        
        # Verify Mtot is identical to MM diagonal
        expected_Mtot = np.diag(indexer.MM).reshape(-1, 1)
        np.testing.assert_allclose(indexer.Mtot, expected_Mtot.astype(np.float32),
                                   err_msg="Mtot != diag(MM)")
        print(f"✓ Mtot: shape={indexer.Mtot.shape}, identical to MM diagonal")
    except AssertionError as e:
        save_failed_matrix(indexer.Mtot, "Mtot", "test_ml_indexer")
        raise
    
    # ========================================================================
    # 7. ML-READY MAPPING MATRICES TESTS
    # ========================================================================
    print("\n[20] Testing phaseToCompMap...")
    try:
        assert isinstance(indexer.phaseToCompMap, np.ndarray), "phaseToCompMap must be ndarray"
        assert indexer.phaseToCompMap.dtype == np.float32, f"phaseToCompMap dtype != float32"
        assert indexer.phaseToCompMap.shape == (P, C), f"phaseToCompMap shape {indexer.phaseToCompMap.shape} != (P={P}, C={C})"
        
        # Verify binary values
        assert np.all((indexer.phaseToCompMap == 0) | (indexer.phaseToCompMap == 1)), "phaseToCompMap must be binary"
        
        # Verify each component belongs to exactly one phase
        column_sums = indexer.phaseToCompMap.sum(axis=0)
        assert np.allclose(column_sums, 1.0), "Each component must belong to exactly one phase (column sum = 1)"
        print(f"✓ phaseToCompMap: shape={indexer.phaseToCompMap.shape}, each component in exactly one phase")
    except AssertionError as e:
        save_failed_matrix(indexer.phaseToCompMap, "phaseToCompMap", "test_ml_indexer")
        raise
    
    # ========================================================================
    print("\n[21] Testing variedToAllComp...")
    try:
        assert isinstance(indexer.variedToAllComp, np.ndarray), "variedToAllComp must be ndarray"
        assert indexer.variedToAllComp.dtype == np.float32, f"variedToAllComp dtype != float32"
        assert indexer.variedToAllComp.shape == (VC, C), f"variedToAllComp shape {indexer.variedToAllComp.shape} != (VC={VC}, C={C})"
        
        # Verify binary values
        assert np.all((indexer.variedToAllComp == 0) | (indexer.variedToAllComp == 1)), "variedToAllComp must be binary"
        
        # Verify each varied component maps to exactly one full component
        if VC > 0:
            row_sums = indexer.variedToAllComp.sum(axis=1)
            assert np.allclose(row_sums, 1.0), "Each varied component must map to exactly one full component (row sum = 1)"
        print(f"✓ variedToAllComp: shape={indexer.variedToAllComp.shape}, each varied component maps to one full component")
    except AssertionError as e:
        save_failed_matrix(indexer.variedToAllComp, "variedToAllComp", "test_ml_indexer")
        raise
    
    # ========================================================================
    print("\n[22] Testing compositionally_variable_binaries...")
    assert isinstance(indexer.compositionally_variable_binaries, np.ndarray), "compositionally_variable_binaries must be ndarray"
    assert indexer.compositionally_variable_binaries.dtype in [bool, np.bool_, np.int32, np.int64], f"compositionally_variable_binaries dtype must be bool or int"
    assert len(indexer.compositionally_variable_binaries) == P, f"len(compositionally_variable_binaries)={len(indexer.compositionally_variable_binaries)} != P={P}"
    
    # Verify sum equals VP
    binary_sum = indexer.compositionally_variable_binaries.sum()
    assert binary_sum == VP, f"Sum of compositionally_variable_binaries {binary_sum} != VP={VP}"
    print(f"✓ compositionally_variable_binaries: len={len(indexer.compositionally_variable_binaries)}, sum={binary_sum} (VP={VP})")
    
    # ========================================================================
    print("\n[23] Testing compositionally_variable_subset...")
    assert isinstance(indexer.compositionally_variable_subset, np.ndarray), "compositionally_variable_subset must be ndarray"
    assert indexer.compositionally_variable_subset.dtype in [np.int32, np.int64, int], f"compositionally_variable_subset dtype must be int"
    assert len(indexer.compositionally_variable_subset) == VC, f"len(compositionally_variable_subset)={len(indexer.compositionally_variable_subset)} != VC={VC}"
    
    # Verify all indices are valid and within range
    assert np.all(indexer.compositionally_variable_subset >= 0), "All indices must be >= 0"
    assert np.all(indexer.compositionally_variable_subset < C), f"All indices must be < C={C}"
    print(f"✓ compositionally_variable_subset: len={len(indexer.compositionally_variable_subset)}, valid indices")
    
    # ========================================================================
    print("\n[24] Testing compositional_component_subset...")
    assert isinstance(indexer.compositional_component_subset, np.ndarray), "compositional_component_subset must be ndarray"
    assert indexer.compositional_component_subset.dtype in [np.int32, np.int64, int], f"compositional_component_subset dtype must be int"
    
    # Verify identity with compositionally_variable_subset
    np.testing.assert_array_equal(indexer.compositional_component_subset, indexer.compositionally_variable_subset,
                                  err_msg="compositional_component_subset must be identical to compositionally_variable_subset")
    print(f"✓ compositional_component_subset: identical to compositionally_variable_subset")
    
    # ========================================================================
    print("\n[25] Testing fixed_phaseToCompMap...")
    try:
        assert isinstance(indexer.fixed_phaseToCompMap, np.ndarray), "fixed_phaseToCompMap must be ndarray"
        assert indexer.fixed_phaseToCompMap.dtype == np.float32, f"fixed_phaseToCompMap dtype != float32"
        assert indexer.fixed_phaseToCompMap.shape == (1, C), f"fixed_phaseToCompMap shape {indexer.fixed_phaseToCompMap.shape} != (1, C={C})"
        
        # Verify sum equals P - VP (number of fixed phases)
        fixed_sum = indexer.fixed_phaseToCompMap.sum()
        if indexer.do_bulk:
            assert fixed_sum == P - VP, f"Sum of fixed_phaseToCompMap {fixed_sum} != P-VP={P-VP}"
            print(f"✓ fixed_phaseToCompMap: shape={indexer.fixed_phaseToCompMap.shape}, sum={fixed_sum} (P-VP={P-VP})")
    except AssertionError as e:
        save_failed_matrix(indexer.fixed_phaseToCompMap, "fixed_phaseToCompMap", "test_ml_indexer")
        raise
    
    # ========================================================================
    print("\n[26] Testing comp_variable_IDMAT...")
    try:
        import torch
        assert indexer.comp_variable_IDMAT is not None, "comp_variable_IDMAT should not be None if torch is available"
        assert isinstance(indexer.comp_variable_IDMAT, torch.Tensor), "comp_variable_IDMAT must be torch.Tensor"
        assert indexer.comp_variable_IDMAT.shape == (P, P), f"comp_variable_IDMAT shape != (P={P}, P={P})"
        assert indexer.comp_variable_IDMAT.dtype == torch.float, f"comp_variable_IDMAT dtype != torch.float"
        
        # Verify diagonal matrix with VP ones on diagonal
        diag_sum = indexer.comp_variable_IDMAT.diag().sum().item()
        assert diag_sum == VP, f"Diagonal sum {diag_sum} != VP={VP}"
        print(f"✓ comp_variable_IDMAT: shape={indexer.comp_variable_IDMAT.shape}, diagonal sum={diag_sum} (VP={VP})")
    except ImportError:
        assert indexer.comp_variable_IDMAT is None, "comp_variable_IDMAT should be None if torch is not available"
        print(f"✓ comp_variable_IDMAT: None (torch not available)")
    
    # ========================================================================
    # 8. BACKWARD-COMPATIBILITY STRUCTURE TESTS
    # ========================================================================
    print("\n[27] Testing comp_binaries...")
    assert isinstance(indexer.comp_binaries, np.ndarray), "comp_binaries must be ndarray"
    assert indexer.comp_binaries.dtype in [np.int32, np.int64, int], f"comp_binaries dtype must be int"
    assert len(indexer.comp_binaries) == VP, f"len(comp_binaries)={len(indexer.comp_binaries)} != VP={VP}"
    print(f"✓ comp_binaries: len={len(indexer.comp_binaries)}")
    
    # ========================================================================
    print("\n[28] Testing comp_mappings...")
    try:
        assert isinstance(indexer.comp_mappings, np.ndarray), "comp_mappings must be ndarray"
        assert indexer.comp_mappings.dtype == np.float32, f"comp_mappings dtype != float32"
        assert indexer.comp_mappings.shape == (VP, VC), f"comp_mappings shape {indexer.comp_mappings.shape} != (VP={VP}, VC={VC})"
        
        # Verify column sums are 1 (each component belongs to one variable phase)
        if VC > 0 and VP > 0:
            column_sums = indexer.comp_mappings.sum(axis=0)
            assert np.allclose(column_sums, 1.0), "Each varied component must belong to exactly one variable phase (column sum = 1)"
        print(f"✓ comp_mappings: shape={indexer.comp_mappings.shape}, each component in one variable phase")
    except AssertionError as e:
        save_failed_matrix(indexer.comp_mappings, "comp_mappings", "test_ml_indexer")
        raise
    
    # ========================================================================
    # 9. ELEMENT AND OXIDE LISTS TESTS
    # ========================================================================
    print("\n[29] Testing Elkeys...")
    assert isinstance(indexer.Elkeys, list), "Elkeys must be a list"
    assert len(indexer.Elkeys) == E, f"len(Elkeys)={len(indexer.Elkeys)} != E={E}"
    assert all(isinstance(el, str) for el in indexer.Elkeys), "All Elkeys must be strings"
    print(f"✓ Elkeys: len={len(indexer.Elkeys)}")
    
    # ========================================================================
    print("\n[30] Testing WRkeys...")
    assert isinstance(indexer.WRkeys, list), "WRkeys must be a list"
    assert len(indexer.WRkeys) == E, f"len(WRkeys)={len(indexer.WRkeys)} != E={E}"
    assert all(isinstance(ox, str) for ox in indexer.WRkeys), "All WRkeys must be strings"
    # Verify no Fe2O3 in WRkeys
    assert 'Fe2O3' not in indexer.WRkeys, "Fe2O3 must not be in WRkeys"
    print(f"✓ WRkeys: len={len(indexer.WRkeys)}, no Fe2O3")
    
    # ========================================================================
    print("\n[31] Testing Oxides...")
    assert isinstance(indexer.Oxides, list), "Oxides must be a list"
    assert len(indexer.Oxides) == O, f"len(Oxides)={len(indexer.Oxides)} != O={O}"
    assert all(isinstance(ox, str) for ox in indexer.Oxides), "All Oxides must be strings"
    # Verify O = E + 1 and Fe2O3 is included
    assert O == E + 1, f"O={O} must equal E+1={E+1}"
    assert 'Fe2O3' in indexer.Oxides, "Fe2O3 must be in Oxides"
    # WRkeys should be all Oxides except Fe2O3
    oxides_without_fe2o3 = [ox for ox in indexer.Oxides if ox != 'Fe2O3']
    assert len(oxides_without_fe2o3) == E, f"Oxides without Fe2O3 should have length E={E}"
    print(f"✓ Oxides: len={len(indexer.Oxides)}, Fe2O3 included, O=E+1")
    
    # ========================================================================
    # 10. CROSS-ATTRIBUTE CONSISTENCY TESTS
    # ========================================================================
    print("\n[32] Testing cross-attribute consistency...")
    
    # Verify label_indices keys match all_phases
    assert set(indexer.label_indices.keys()) == set(indexer.all_phases), "label_indices keys must match all_phases"
    
    # Verify mass_phasedict keys match all_phases
    assert set(indexer.mass_phasedict.keys()) == set(indexer.all_phases), "mass_phasedict keys must match all_phases"
    
    # Verify compositionally_variable_phases is subset of all_phases
    assert set(indexer.compositionally_variable_phases).issubset(set(indexer.all_phases)), \
        "compositionally_variable_phases must be subset of all_phases"
    
    # Verify detail_label_indices keys are subset of compositionally_variable_phases
    assert set(indexer.detail_label_indices.keys()).issubset(set(indexer.compositionally_variable_phases)), \
        "detail_label_indices keys must be subset of compositionally_variable_phases"
    
    print(f"✓ All cross-attribute relationships consistent")
    
    # ========================================================================
    # 11. DIMENSIONAL LOGIC TESTS
    # ========================================================================
    print("\n[33] Testing dimensional logic...")
    
    # Verify phaseToCompMap * 1_C = 1_P (each phase has at least one component)
    ones_vector = np.ones(C)
    result = indexer.phaseToCompMap @ ones_vector
    assert np.all(result >= 1), "Each phase must have at least one component"
    
    # Verify variedToAllComp can select from compToOx
    if VC > 0 and C > 0:
        test_comp = np.random.randn(C)
        varied_selection = indexer.variedToAllComp @ test_comp
        assert varied_selection.shape == (VC,), "variedToAllComp selection should yield VC components"
    
    print(f"✓ Dimensional logic validated")
    
    # ========================================================================
    # 12. TRANSFORMATION CHAIN TESTS
    # ========================================================================
    print("\n[34] Testing transformation chains...")
    
    # Test component -> oxide -> element chain
    test_comp = np.random.rand(C).astype(np.float32)
    
    # Component to oxide
    test_oxide = indexer.compToOx.T @ test_comp
    assert test_oxide.shape == (O,), "compToOx transformation should yield O oxides"
    
    # Oxide to element
    test_elem = indexer.OxToEl.T @ test_oxide
    assert test_elem.shape == (E,), "OxToEl transformation should yield E elements"
    
    print(f"✓ Transformation chains validated: component -> oxide -> element")
    
    # ========================================================================
    # 13. PHASE AGGREGATION TESTS
    # ========================================================================
    print("\n[35] Testing phase aggregation...")
    
    # Test that phaseToCompMap can aggregate components to phases
    test_comp_vec = np.random.rand(C).astype(np.float32)
    phase_vec = indexer.phaseToCompMap @ test_comp_vec
    assert phase_vec.shape == (P,), "Phase aggregation should yield P phases"
    assert np.all(np.isfinite(phase_vec)), "Phase aggregation should yield finite values"
    
    print(f"✓ Phase aggregation validated")
    
    # ========================================================================
    # 14. SAVE/LOAD ROUND-TRIP TESTS
    # ========================================================================
    print("\n[36] Testing save/load round-trip...")
    from nMELTS.config.ml_indexer import load_ml_indexer_from_state

    with tempfile.TemporaryDirectory() as tmp_dir:
        indexer.save(tmp_dir)
        reloaded = load_ml_indexer_from_state(tmp_dir)

    # Basic metadata checks
    assert reloaded.ncomps == indexer.ncomps, "ncomps mismatch after reload"
    assert reloaded.ncompsVaried == indexer.ncompsVaried, "ncompsVaried mismatch after reload"
    assert reloaded.nphases == indexer.nphases, "nphases mismatch after reload"
    assert reloaded.label_names == indexer.label_names, "label_names mismatch after reload"
    assert reloaded.all_phases == indexer.all_phases, "all_phases mismatch after reload"
    assert reloaded.compositionally_variable_phases == indexer.compositionally_variable_phases, "compositionally_variable_phases mismatch"
    assert reloaded.Elkeys == indexer.Elkeys, "Elkeys mismatch after reload"
    assert reloaded.Oxides == indexer.Oxides, "Oxides mismatch after reload"
    assert reloaded.WRkeys == indexer.WRkeys, "WRkeys mismatch after reload"
    assert reloaded.do_bulk == indexer.do_bulk, "do_bulk mismatch after reload"

    # Dictionary + array checks
    dict_array_attrs = [
        "label_indices",
        "label_indices_comp",
    ]
    for attr in dict_array_attrs:
        original = getattr(indexer, attr)
        loaded = getattr(reloaded, attr)
        assert original.keys() == loaded.keys(), f"{attr} keys mismatch after reload"
        for key in original:
            np.testing.assert_array_equal(loaded[key], original[key], err_msg=f"{attr}[{key}] mismatch")

    dict_attrs = [
        "mass_phasedict",
        "comp_phasedict",
        "detail_label_indices",
    ]
    for attr in dict_attrs:
        assert getattr(reloaded, attr) == getattr(indexer, attr), f"{attr} mismatch after reload"

    array_float_attrs = [
        "compToOxLoad",
        "PxSpTransform",
        "compToOx",
        "OxToEl",
        "ElToOx",
        "MM",
        "Minv",
        "Mtot",
        "phaseToCompMap",
        "variedToAllComp",
        "fixed_phaseToCompMap",
        "comp_mappings",
    ]
    for attr in array_float_attrs:
        np.testing.assert_allclose(
            getattr(reloaded, attr),
            getattr(indexer, attr),
            rtol=1e-6,
            err_msg=f"{attr} mismatch after reload"
        )

    array_int_attrs = [
        "compositionally_variable_subset",
        "compositional_component_subset",
        "comp_binaries",
        "compositionally_variable_binaries",
    ]
    for attr in array_int_attrs:
        np.testing.assert_array_equal(
            getattr(reloaded, attr),
            getattr(indexer, attr),
            err_msg=f"{attr} mismatch after reload"
        )

    if indexer.boolTransCompToOx is None:
        assert reloaded.boolTransCompToOx is None, "boolTransCompToOx mismatch after reload"
    else:
        np.testing.assert_array_equal(
            reloaded.boolTransCompToOx,
            indexer.boolTransCompToOx,
            err_msg="boolTransCompToOx mismatch after reload"
        )

    # Normalizer state checks (if present)
    if indexer.feature_normalizer is None:
        assert reloaded.feature_normalizer is None, "feature_normalizer mismatch after reload"
    else:
        original_state = indexer.feature_normalizer.to_state_dict()
        reloaded_state = reloaded.feature_normalizer.to_state_dict()
        np.testing.assert_allclose(original_state["min"], reloaded_state["min"], rtol=1e-6)
        np.testing.assert_allclose(original_state["range"], reloaded_state["range"], rtol=1e-6)

    if indexer.output_normalizer is None:
        assert reloaded.output_normalizer is None, "output_normalizer mismatch after reload"
    else:
        original_state = indexer.output_normalizer.to_state_dict()
        reloaded_state = reloaded.output_normalizer.to_state_dict()
        np.testing.assert_allclose(original_state["min"], reloaded_state["min"], rtol=1e-6)
        np.testing.assert_allclose(original_state["range"], reloaded_state["range"], rtol=1e-6)

    # Torch-specific matrix (optional)
    try:
        import torch
    except ImportError:
        torch = None

    if getattr(indexer, "comp_variable_IDMAT", None) is None:
        assert reloaded.comp_variable_IDMAT is None, "comp_variable_IDMAT mismatch after reload"
    else:
        assert reloaded.comp_variable_IDMAT is not None, "comp_variable_IDMAT missing after reload"
        if torch is not None:
            assert torch.allclose(reloaded.comp_variable_IDMAT, indexer.comp_variable_IDMAT), "comp_variable_IDMAT mismatch"

    print("✓ save/load round-trip: all checked attributes match")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 50)
    print("✓ ALL TESTS PASSED")
    print("=" * 50)
    print(f"\nDimensions verified: C={C}, P={P}, VP={VP}, VC={VC}, E={E}, O={O}, WR={WR}")
    print(f"All {36} test categories completed successfully")


def test_restrictVC(ml_indexer):
    """
    Test suite for MLIndexer.restrictVC() method.
    
    This test verifies that restrictVC() correctly filters variable chemistry
    structures to only include specified phases, and that all dependent structures
    are properly rebuilt.
    
    Parameters
    ----------
    ml_indexer : MLIndexer
        An initialized MLIndexer instance to test
        
    Raises
    ------
    AssertionError
        If any test condition fails or VC is not reduced
    """
    
    print(f"\n=== MLIndexer.restrictVC() Test Suite ===")
    
    # Get initial dimensions
    initial_VC = ml_indexer.ncompsVaried
    initial_VP = len(ml_indexer.compositionally_variable_phases)
    initial_phases = ml_indexer.compositionally_variable_phases.copy()
    
    print(f"\nInitial state:")
    print(f"  - Variable chemistry phases (VP): {initial_VP}")
    print(f"  - Variable chemistry components (VC): {initial_VC}")
    print(f"  - Phases: {initial_phases}")
    
    # ========================================================================
    # 1. TEST RESTRICTION TO LIQUID AND CLINOPYROXENE
    # ========================================================================
    print(f"\nRestricting to ['melts-liquid', 'clinopyroxene']...")
    
    # Verify clinopyroxene is in the original variable phases
    if 'clinopyroxene' not in initial_phases:
        print(f"  Note: clinopyroxene not in variable phases. Available: {initial_phases}")
    
    restrict_phases = ['melts-liquid', 'clinopyroxene']
    ml_indexer.restrictVC(restrict_phases)
    
    
    # Get new dimensions
    new_VC = ml_indexer.ncompsVaried
    new_VP = len(ml_indexer.compositionally_variable_phases)
    new_phases = ml_indexer.compositionally_variable_phases.copy()
    
    print(f"\nAfter restriction:")
    print(f"  - Variable chemistry phases (VP): {new_VP}")
    print(f"  - Variable chemistry components (VC): {new_VC}")
    print(f"  - Phases: {new_phases}")
    
    # ========================================================================
    # 2. VERIFY VC WAS REDUCED
    # ========================================================================
    print(f"\n[1] Verifying VC reduction...")
    assert new_VC < initial_VC, f"VC should decrease: {initial_VC} -> {new_VC}"
    reduction_percent = (1 - new_VC / initial_VC) * 100 if initial_VC > 0 else 0
    print(f"✓ VC reduced from {initial_VC} to {new_VC} ({reduction_percent:.1f}% reduction)")
    
    # ========================================================================
    # 3. VERIFY PHASE FILTERING
    # ========================================================================
    print(f"\n[2] Verifying phase filtering...")
    for phase in new_phases:
        assert phase in restrict_phases, f"Phase {phase} not in restrict list {restrict_phases}"
    print(f"✓ All {new_VP} remaining phases are in restrict list")
    
    # ========================================================================
    # 4. VERIFY comp_phasedict WAS REBUILT
    # ========================================================================
    print(f"\n[3] Verifying comp_phasedict rebuild...")
    assert len(ml_indexer.comp_phasedict) == new_VP, \
        f"comp_phasedict size {len(ml_indexer.comp_phasedict)} != VP {new_VP}"
    for phase in ml_indexer.comp_phasedict.keys():
        assert phase in new_phases, f"comp_phasedict contains {phase} not in new_phases"
    comp_dict_values = sorted(ml_indexer.comp_phasedict.values())
    assert comp_dict_values == list(range(new_VP)), \
        f"comp_phasedict values {comp_dict_values} != 0..{new_VP-1}"
    print(f"✓ comp_phasedict correctly rebuilt with {new_VP} entries")
    
    # ========================================================================
    # 5. VERIFY DEPENDENT STRUCTURES WERE REBUILT
    # ========================================================================
    print(f"\n[4] Verifying dependent structures rebuild...")
    
    # Check compositionally_variable_binaries
    assert len(ml_indexer.compositionally_variable_binaries) == ml_indexer.nphases, \
        f"compositionally_variable_binaries length mismatch"
    binary_sum = ml_indexer.compositionally_variable_binaries.sum()
    assert binary_sum == new_VP, f"Sum of binaries {binary_sum} != VP {new_VP}"
    print(f"✓ compositionally_variable_binaries rebuilt correctly (sum={binary_sum})")
    
    # Check variedToAllComp
    if new_VC > 0:
        assert ml_indexer.variedToAllComp.shape == (new_VC, ml_indexer.ncomps), \
            f"variedToAllComp shape {ml_indexer.variedToAllComp.shape} != ({new_VC}, {ml_indexer.ncomps})"
        print(f"✓ variedToAllComp reshaped to ({new_VC}, {ml_indexer.ncomps})")
    
    # Check compositionally_variable_subset
    assert isinstance(ml_indexer.compositionally_variable_subset, np.ndarray), \
        "compositionally_variable_subset must be ndarray"
    assert len(ml_indexer.compositionally_variable_subset) == new_VC, \
        f"compositionally_variable_subset length {len(ml_indexer.compositionally_variable_subset)} != VC {new_VC}"
    print(f"✓ compositionally_variable_subset rebuilt with {new_VC} indices")
    
    # Check fixed_phaseToCompMap
    is_fixed = ~(ml_indexer.compositionally_variable_binaries.astype(bool))
    expected_fixed_shape = (1, ml_indexer.ncomps)
    assert ml_indexer.fixed_phaseToCompMap.shape == expected_fixed_shape, \
        f"fixed_phaseToCompMap shape {ml_indexer.fixed_phaseToCompMap.shape} != {expected_fixed_shape}"
    print(f"✓ fixed_phaseToCompMap rebuilt with shape {expected_fixed_shape}")
    
    # ========================================================================
    # 6. RUN FULL TEST SUITE ON RESTRICTED INDEXER
    # ========================================================================
    print(f"\n[5] Running full test suite on restricted indexer...")
    print(f"\n" + "=" * 70)
    test_ml_indexer(ml_indexer)
    print("=" * 70)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n" + "=" * 50)
    print("✓ RESTRICTVC TEST PASSED")
    print("=" * 50)
    print(f"\nRestriction Summary:")
    print(f"  - Reduced variable phases: {initial_VP} → {new_VP}")
    print(f"  - Reduced variable components: {initial_VC} → {new_VC} ({reduction_percent:.1f}%)")
    print(f"  - All dependent structures rebuilt and validated")
    print(f"  - Full MLIndexer test suite passed on restricted indexer")


def test_dataset_indexer(indexer):
    """
    Test suite for DatasetIndexer instance.
    
    Validates core attributes, header parsing, exclusion logic, and delegation
    to MLIndexer for ML-ready structures.
    
    Parameters
    ----------
    indexer : DatasetIndexer
        An initialized DatasetIndexer instance to test
        
    Raises
    ------
    AssertionError
        If any test condition fails
    """
    
    print(f"\n=== DatasetIndexer Test Suite ===")
    print(f"Headers: {len(indexer.headers)} columns")
    print(f"Phases: {len(indexer.MELTS_indices)} in MELTS_indices")
    print(f"Excluded phases: {len(indexer.EXCLUDED_PHASES)}")
    print(f"Excluded components: {len(indexer.EXCLUDED_COMPONENTS_BY_PHASE)}")
    print("=" * 50)
    
    # ========================================================================
    # 1. HEADER AND MELTS_INDICES TESTS
    # ========================================================================
    print("\n[1] Testing headers and MELTS_indices structure...")
    assert isinstance(indexer.headers, list), "headers must be a list"
    assert isinstance(indexer.database_headers, list), "database_headers must be a list"
    assert indexer.headers == indexer.database_headers, "headers should equal database_headers"
    assert len(indexer.headers) > 0, "headers must not be empty"
    
    assert isinstance(indexer.MELTS_indices, dict), "MELTS_indices must be a dict"
    assert len(indexer.MELTS_indices) > 0, "MELTS_indices must not be empty"
    
    # Verify all indices point to valid header positions
    all_indices_list = []
    for phase, comp_dict in indexer.MELTS_indices.items():
        assert isinstance(comp_dict, dict), f"MELTS_indices[{phase}] must be dict"
        for component, idx in comp_dict.items():
            assert isinstance(idx, int), f"Index for {component}({phase}) must be int"
            assert 0 <= idx < len(indexer.headers), f"Index {idx} out of range for headers (len={len(indexer.headers)})"
            all_indices_list.append(idx)
    
    # Test: max val = len(headers) - 1
    if all_indices_list:
        max_index = max(all_indices_list)
        assert max_index <= len(indexer.headers) - 1, f"Max index {max_index} exceeds headers range"
        print(f"✓ MELTS_indices: {len(indexer.MELTS_indices)} phases, max index={max_index}, headers range=[0, {len(indexer.headers)-1}]")
    
    # ========================================================================
    # 2. COMPONENTS_IN_PHASES TESTS
    # ========================================================================
    print("\n[2] Testing components_in_phases...")
    assert isinstance(indexer.components_in_phases, dict), "components_in_phases must be a dict"
    
    # Test: every phase has at least one component
    for phase, components in indexer.components_in_phases.items():
        assert isinstance(components, list), f"components_in_phases[{phase}] must be a list"
        assert len(components) > 0, f"Phase {phase} must have at least one component"
        assert all(isinstance(c, str) for c in components), f"All components in {phase} must be strings"
    
    # Verify excluded phases are not in components_in_phases
    for excluded_phase in indexer.EXCLUDED_PHASES:
        assert excluded_phase not in indexer.components_in_phases, \
            f"Excluded phase {excluded_phase} should not be in components_in_phases"
    
    print(f"✓ components_in_phases: {len(indexer.components_in_phases)} phases, all have ≥1 component")
    
    # ========================================================================
    # 3. MASS_INDICES TESTS
    # ========================================================================
    print("\n[3] Testing mass_indices...")
    assert isinstance(indexer.mass_indices, np.ndarray), "mass_indices must be ndarray"
    assert indexer.mass_indices.dtype in [np.int32, np.int64, int], "mass_indices dtype must be int"
    
    # Verify all mass indices are valid and unique
    if len(indexer.mass_indices) > 0:
        assert np.all(indexer.mass_indices >= 0), "All mass indices must be >= 0"
        assert np.all(indexer.mass_indices < len(indexer.headers)), "All mass indices must be < len(headers)"
        assert len(indexer.mass_indices) == len(np.unique(indexer.mass_indices)), "mass_indices must be unique"
        
        # Verify these indices actually correspond to mass columns
        for idx in indexer.mass_indices:
            header = indexer.headers[idx]
            assert 'mass' in header.lower(), f"mass_indices contains non-mass column: {header}"
    
    print(f"✓ mass_indices: {len(indexer.mass_indices)} mass columns, all valid and unique")
    
    # ========================================================================
    # 4. EXCLUSION LOGIC TESTS
    # ========================================================================
    print("\n[4] Testing exclusion logic...")
    assert isinstance(indexer.EXCLUDED_PHASES, set), "EXCLUDED_PHASES must be a set"
    assert isinstance(indexer.EXCLUDED_COMPONENTS_BY_PHASE, dict), "EXCLUDED_COMPONENTS_BY_PHASE must be a dict"
    assert isinstance(indexer.STATE_VARIABLES, set), "STATE_VARIABLES must be a set"
    
    # System_main and Bulk_comp should always be excluded
    assert 'System_main' in indexer.EXCLUDED_PHASES, "System_main must be in EXCLUDED_PHASES"
    assert 'Bulk_comp' in indexer.EXCLUDED_PHASES, "Bulk_comp must be in EXCLUDED_PHASES"
    
    # Verify excluded components don't appear in components_in_phases
    for phase, components in indexer.components_in_phases.items():
        for component in components:
            assert component not in indexer.EXCLUDED_COMPONENTS_BY_PHASE.get(phase, set()), \
                f"Excluded component {component} found in components_in_phases[{phase}]"
            assert component not in indexer.STATE_VARIABLES, \
                f"State variable {component} found in components_in_phases[{phase}]"
    
    print(f"✓ Exclusions: {len(indexer.EXCLUDED_PHASES)} phases, {sum(len(comps) for comps in indexer.EXCLUDED_COMPONENTS_BY_PHASE.values())} components excluded")
    
    # ========================================================================
    # 5. ELEMENT AND OXIDE LISTS TESTS
    # ========================================================================
    print("\n[5] Testing element and oxide lists...")
    assert isinstance(indexer.Elkeys, list), "Elkeys must be a list"
    assert len(indexer.Elkeys) > 0, "Elkeys must not be empty"
    assert all(isinstance(el, str) for el in indexer.Elkeys), "All Elkeys must be strings"
    
    # Required elements should be present
    from src.nMELTS.config.constants import REQUIRED_ELEMENTS
    for req_el in REQUIRED_ELEMENTS:
        assert req_el in indexer.Elkeys, f"Required element {req_el} missing from Elkeys"
    
    assert isinstance(indexer.WRkeys, list), "WRkeys must be a list"
    assert len(indexer.WRkeys) == len(indexer.Elkeys), "WRkeys length must equal Elkeys length"
    assert 'Fe2O3' not in indexer.WRkeys, "Fe2O3 must not be in WRkeys"
    
    assert isinstance(indexer.Oxides, list), "Oxides must be a list"
    assert len(indexer.Oxides) == len(indexer.Elkeys) + 1, "Oxides length must equal Elkeys + 1"
    assert 'Fe2O3' in indexer.Oxides, "Fe2O3 must be in Oxides"
    
    assert isinstance(indexer.oxide_dict, dict), "oxide_dict must be a dict"
    assert len(indexer.oxide_dict) == len(indexer.Oxides), "oxide_dict length must equal Oxides length"
    for i, ox in enumerate(indexer.Oxides):
        assert ox in indexer.oxide_dict, f"Oxide {ox} missing from oxide_dict"
        assert indexer.oxide_dict[ox] == i, f"oxide_dict[{ox}] should be {i}"
    
    print(f"✓ Elements/Oxides: {len(indexer.Elkeys)} elements, {len(indexer.Oxides)} oxides")
    
    # ========================================================================
    # 6. ML INDEXER DELEGATION TESTS
    # ========================================================================
    print("\n[6] Testing MLIndexer delegation...")
    assert hasattr(indexer, 'ml_indexer'), "DatasetIndexer must have ml_indexer attribute"
    from nMELTS.config.ml_indexer import MLIndexer
    assert isinstance(indexer.ml_indexer, MLIndexer), "ml_indexer must be an MLIndexer instance"
    
    # Verify key attributes are exposed from ml_indexer
    ml_attrs = [
        'label_indices', 'label_names', 'detail_label_indices', 'label_indices_comp',
        'all_phases', 'mass_phasedict', 'comp_phasedict', 'compositionally_variable_phases',
        'phaseToCompMap', 'variedToAllComp', 'comp_variable_IDMAT', 'fixed_phaseToCompMap',
        'ncomps', 'ncompsVaried', 'nphases'
    ]
    
    for attr in ml_attrs:
        assert hasattr(indexer, attr), f"DatasetIndexer must expose {attr} from ml_indexer"
        assert getattr(indexer, attr) is getattr(indexer.ml_indexer, attr), \
            f"{attr} should reference ml_indexer.{attr}"
    
    print(f"✓ MLIndexer delegation: {len(ml_attrs)} attributes properly exposed")
    
    # ========================================================================
    # 7. HELPER METHODS TESTS
    # ========================================================================
    print("\n[7] Testing helper methods...")
    
    # Test get_max_index()
    max_idx = indexer.get_max_index()
    assert isinstance(max_idx, int), "get_max_index() must return int"
    assert max_idx >= -1, "get_max_index() must return >= -1"
    if len(indexer.MELTS_indices) > 0:
        assert max_idx < len(indexer.headers), "get_max_index() must be < len(headers)"
    
    # Test get_phase_list()
    phase_list = indexer.get_phase_list()
    assert isinstance(phase_list, list), "get_phase_list() must return list"
    assert phase_list == list(indexer.MELTS_indices.keys()), "get_phase_list() should return MELTS_indices keys"
    
    # Test get_components_for_phase()
    for phase in indexer.MELTS_indices.keys():
        components = indexer.get_components_for_phase(phase)
        assert isinstance(components, list), f"get_components_for_phase('{phase}') must return list"
        assert components == list(indexer.MELTS_indices[phase].keys()), \
            f"get_components_for_phase('{phase}') should return correct components"
    
    # Test with non-existent phase
    empty_components = indexer.get_components_for_phase('nonexistent_phase_xyz')
    assert empty_components == [], "get_components_for_phase() should return empty list for nonexistent phase"
    
    print(f"✓ Helper methods: get_max_index()={max_idx}, get_phase_list() returns {len(phase_list)} phases")
    
    # ========================================================================
    # 8. CONSISTENCY BETWEEN MELTS_INDICES AND COMPONENTS_IN_PHASES
    # ========================================================================
    print("\n[8] Testing consistency between MELTS_indices and components_in_phases...")
    
    for phase in indexer.components_in_phases.keys():
        assert phase in indexer.MELTS_indices, \
            f"Phase {phase} in components_in_phases but not in MELTS_indices"
        
        # Verify components in components_in_phases are subset of MELTS_indices components
        # (MELTS_indices may have state variables that are filtered out)
        melts_components = set(indexer.MELTS_indices[phase].keys())
        cip_components = set(indexer.components_in_phases[phase])
        
        assert cip_components.issubset(melts_components.union(indexer.components_in_phases.keys())) , \
            f"components_in_phases[{phase}] contains components not in MELTS_indices[{phase}]"
    
    print(f"✓ Consistency: components_in_phases aligns with MELTS_indices")
    
    # ========================================================================
    # 9. PROJECTION FILE LOADING TESTS
    # ========================================================================
    print("\n[9] Testing projection file loading...")
    assert hasattr(indexer, 'compToOx_df'), "DatasetIndexer must have compToOx_df"
    assert isinstance(indexer.compToOx_df, pd.DataFrame), "compToOx_df must be DataFrame"
    assert len(indexer.compToOx_df) > 0, "compToOx_df must not be empty"
    
    assert hasattr(indexer, 'components_with_extra_oxides'), "DatasetIndexer must have components_with_extra_oxides"
    assert isinstance(indexer.components_with_extra_oxides, dict), "components_with_extra_oxides must be dict"
    
    assert hasattr(indexer, 'oxides_to_elements'), "DatasetIndexer must have oxides_to_elements"
    assert isinstance(indexer.oxides_to_elements, dict), "oxides_to_elements must be dict"
    
    print(f"✓ Projection files: compToOx_df loaded with {len(indexer.compToOx_df)} components")
    
    # ========================================================================
    # 10. MELTS-LIQUID OXIDE CONSISTENCY TESTS (I don't think this is assumed anywhere)
    # ========================================================================
    """ print("\n[10] Testing melts-liquid oxide names match Oxides list...")
    assert 'melts-liquid' in indexer.MELTS_indices, "melts-liquid must be in MELTS_indices"
    
    melts_liquid_components = list(indexer.MELTS_indices['melts-liquid'].keys())
    
    # Extract oxide names from melts-liquid components (format: "wt% SiO2" or similar)
    melts_liquid_oxides = []
    for comp in melts_liquid_components:
        # Skip mass and other non-oxide components
        if 'mass' in comp.lower() or 'liq' in comp.lower() or 'rho' in comp.lower() or 'vis' in comp.lower() or 'H' == comp or 'S' == comp or 'V' == comp:
            continue
        # Extract oxide name (remove "wt% " prefix if present)
        oxide_name = comp.replace('wt% ', '').strip()
        melts_liquid_oxides.append(oxide_name)
    
    # Verify melts_liquid_oxides matches Oxides exactly (same names, same order)
    assert len(melts_liquid_oxides) == len(indexer.Oxides), \
        f"melts-liquid oxide count {len(melts_liquid_oxides)} != Oxides count {len(indexer.Oxides)}" \
        f"\nmelts-liquid oxides {melts_liquid_oxides} != {indexer.Oxides}"
    
    
    for i, (mlox, oxox) in enumerate(zip(melts_liquid_oxides, indexer.Oxides)):
        assert mlox == oxox, \
            f"Oxide mismatch at position {i}: melts-liquid has '{mlox}', but Oxides has '{oxox}'"
    
    print(f"✓ melts-liquid oxides: {len(melts_liquid_oxides)} oxides, names and order match Oxides list exactly")
    """
    # ========================================================================
    # 11. RUN ML_INDEXER TESTS
    # ========================================================================
    print("\n[11] Running embedded MLIndexer tests...")
    try:
        test_ml_indexer(indexer.ml_indexer)
        print(f"✓ Embedded MLIndexer passed all tests")
    except AssertionError as e:
        raise AssertionError(f"Embedded MLIndexer test failed: {e}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 50)
    print("✓ ALL DATASETINDEXER TESTS PASSED")
    print("=" * 50)
    print(f"\nDataset structure:")
    print(f"  - {len(indexer.headers)} total columns")
    print(f"  - {len(indexer.MELTS_indices)} phases in MELTS_indices")
    print(f"  - {len(indexer.components_in_phases)} phases in components_in_phases")
    print(f"  - {len(indexer.mass_indices)} mass columns")
    print(f"  - {len(indexer.Elkeys)} elements, {len(indexer.Oxides)} oxides")
    print(f"  - {indexer.nphases} phases, {indexer.ncomps} components in ML indexer")
    print(f"All {11} test categories completed successfully")
    
if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    setup_test_logging(
        log_filename=f"{Path(__file__).stem}_{timestamp}.txt",
        log_dir=Path(__file__).parent / 'logs',
    )
    
    

    from src.builder.indexer import DatasetIndexer, generate_column_headers
    # Test DatasetIndexer
    #from tests.unit_tests.test_indexers.indexer_test import test_dataset_indexer
    


    phases = [
        'olivine','orthopyroxene','clinopyroxene','spinel','plagioclase','k-feldspar','garnet',
        'nepheline','leucite','biotite','rhm-oxide','alloy-solid','alloy-liquid','apatite',
        'whitlockite','quartz','tridymite','cristobalite','muscovite','fluid','liquid'
    ]

    headers = generate_column_headers(phases)

    indexer = DatasetIndexer(headers=headers)

    print("\n" + "=" * 70)
    print("RUNNING INITIAL ML INDEXER TEST")
    print("=" * 70)
    test_ml_indexer(indexer.ml_indexer)

    print("\n\n" + "=" * 70)
    print("RUNNING DATASET INDEXER TEST")
    print("=" * 70)
    test_dataset_indexer(indexer)

    #NOW TRY WITH AUTO EXCLUDE! 

    csv_path = Path(project_root) / 'data' / 'MELTStables' / '110' / 'MELTS110_TrainsetFeb3BatchCooling.csv'
    DF = pd.read_csv(csv_path)
    headers = list(DF.columns)
    indexer_auto = DatasetIndexer(headers=headers)
    indexer_auto.table_update(DF.to_numpy())
    test_dataset_indexer(indexer_auto)

    print("\n\n" + "=" * 70)
    print("RUNNING RESTRICTVC TEST")
    print("=" * 70)
    test_restrictVC(indexer.ml_indexer)
