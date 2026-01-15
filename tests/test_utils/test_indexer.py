"""
Test script for DatasetIndexer class.

Generates sample headers based on the expected format and tests the indexer.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
from nMELTS.config.indexer import DatasetIndexer

# Generate sample headers based on the expected format from constants.py
# This mimics what would come from a real MELTS dataset CSV

def generate_sample_headers():
    """Generate sample headers in component(phase) format."""
    headers = []
    idx = 0
    
    # System_main properties
    headers.append(f"Pressure(System_main)")
    headers.append(f"Temperature(System_main)")
    headers.append(f"logfO2-QFM(System_main)")
    
    # Olivine components
    headers.append(f"mass (gm)(olivine)")
    headers.append(f"tephroite(olivine)")
    headers.append(f"fayalite(olivine)")
    headers.append(f"ni-olivine(olivine)")
    headers.append(f"monticellite(olivine)")
    headers.append(f"forsterite(olivine)")
    
    # Orthopyroxene components
    headers.append(f"mass (gm)(orthopyroxene)")
    headers.append(f"diopside(orthopyroxene)")
    headers.append(f"clinoenstatite(orthopyroxene)")
    headers.append(f"hedenbergite(orthopyroxene)")
    headers.append(f"alumino-buffonite(orthopyroxene)")
    headers.append(f"buffonite(orthopyroxene)")
    headers.append(f"essenite(orthopyroxene)")
    headers.append(f"jadeite(orthopyroxene)")
    
    # Clinopyroxene components
    headers.append(f"mass (gm)(clinopyroxene)")
    headers.append(f"diopside(clinopyroxene)")
    headers.append(f"clinoenstatite(clinopyroxene)")
    headers.append(f"hedenbergite(clinopyroxene)")
    headers.append(f"alumino-buffonite(clinopyroxene)")
    headers.append(f"buffonite(clinopyroxene)")
    headers.append(f"essenite(clinopyroxene)")
    headers.append(f"jadeite(clinopyroxene)")
    
    # Spinel components
    headers.append(f"mass (gm)(spinel)")
    headers.append(f"chromite(spinel)")
    headers.append(f"hercynite(spinel)")
    headers.append(f"magnetite(spinel)")
    headers.append(f"spinel(spinel)")
    headers.append(f"ulvospinel(spinel)")
    
    # Plagioclase components
    headers.append(f"mass (gm)(plagioclase)")
    headers.append(f"albite(plagioclase)")
    headers.append(f"anorthite(plagioclase)")
    headers.append(f"sanidine(plagioclase)")
    
    # Garnet components
    headers.append(f"mass (gm)(garnet)")
    headers.append(f"almandine(garnet)")
    headers.append(f"grossular(garnet)")
    headers.append(f"pyrope(garnet)")
    
    # Nepheline components
    headers.append(f"mass (gm)(nepheline)")
    headers.append(f"na-nepheline(nepheline)")
    headers.append(f"k-nepheline(nepheline)")
    headers.append(f"vc-nepheline(nepheline)")
    headers.append(f"ca-nepheline(nepheline)")
    
    # Leucite components
    headers.append(f"mass (gm)(leucite)")
    headers.append(f"leucite(leucite)")
    headers.append(f"analcime(leucite)")
    headers.append(f"na-leucite(leucite)")
    
    # Biotite components
    headers.append(f"mass (gm)(biotite)")
    headers.append(f"annite(biotite)")
    headers.append(f"phlogopite(biotite)")
    
    # rhm-oxide components
    headers.append(f"mass (gm)(rhm-oxide)")
    headers.append(f"geikielite(rhm-oxide)")
    headers.append(f"hematite(rhm-oxide)")
    headers.append(f"ilmenite(rhm-oxide)")
    headers.append(f"pyrophanite(rhm-oxide)")
    headers.append(f"corundum(rhm-oxide)")
    
    # alloy-solid components
    headers.append(f"mass (gm)(alloy-solid)")
    headers.append(f"Fe-metal(alloy-solid)")
    headers.append(f"Ni-metal(alloy-solid)")
    
    # alloy-liquid components
    headers.append(f"mass (gm)(alloy-liquid)")
    headers.append(f"Fe-metal(alloy-liquid)")
    headers.append(f"Ni-metal(alloy-liquid)")
    
    # analcime components
    headers.append(f"mass (gm)(analcime)")
    headers.append(f"leucite(analcime)")
    headers.append(f"analcime(analcime)")
    headers.append(f"na-leucite(analcime)")
    
    # Simple phases (mass only)
    headers.append(f"mass (gm)(apatite)")
    headers.append(f"mass (gm)(whitlockite)")
    headers.append(f"mass (gm)(quartz)")
    headers.append(f"mass (gm)(tridymite)")
    headers.append(f"mass (gm)(muscovite)")
    headers.append(f"mass (gm)(fluid)")
    
    # System_main additional properties
    headers.append(f"viscosity(System_main)")
    headers.append(f"H(System_main)")
    headers.append(f"Cp(System_main)")
    headers.append(f"S(System_main)")
    headers.append(f"V(System_main)")
    headers.append(f"dVdP*10^6(System_main)")
    headers.append(f"dVdT*10^6(System_main)")
    
    # Add density, enthalpy, entropy, volume for phases
    phases_with_props = ['olivine', 'orthopyroxene', 'clinopyroxene', 'spinel',
                        'plagioclase', 'garnet', 'nepheline', 'leucite',
                        'biotite', 'rhm-oxide', 'alloy-solid', 'alloy-liquid', 
                        'analcime', 'apatite', 'whitlockite', 'quartz', 
                        'tridymite', 'muscovite', 'fluid']
    
    for phase in phases_with_props:
        headers.append(f"rho (gm/cc)({phase})")
        headers.append(f"H (kJ)({phase})")
        headers.append(f"S (J/K)({phase})")
        headers.append(f"V (cc)({phase})")
    
    # Melts-liquid components
    headers.append(f"liq mass (gm)(melts-liquid)")
    headers.append(f"wt% SiO2(melts-liquid)")
    headers.append(f"wt% TiO2(melts-liquid)")
    headers.append(f"wt% Al2O3(melts-liquid)")
    headers.append(f"wt% FeO(melts-liquid)")
    headers.append(f"wt% MgO(melts-liquid)")
    headers.append(f"wt% CaO(melts-liquid)")
    headers.append(f"wt% Na2O(melts-liquid)")
    headers.append(f"wt% K2O(melts-liquid)")
    headers.append(f"wt% P2O5(melts-liquid)")
    headers.append(f"wt% MnO(melts-liquid)")
    headers.append(f"wt% H2O(melts-liquid)")
    headers.append(f"wt% Cr2O3(melts-liquid)")
    headers.append(f"wt% NiO(melts-liquid)")
    headers.append(f"wt% Fe2O3(melts-liquid)")
    
    # Melts-liquid additional properties
    headers.append(f"liq rho (gm/cc)(melts-liquid)")
    headers.append(f"liq vis (log 10 poise)(melts-liquid)")
    headers.append(f"liq H (kJ)(melts-liquid)")
    headers.append(f"liq S (J/K)(melts-liquid)")
    headers.append(f"liq V (cc)(melts-liquid)")
    
    return headers


def print_indexer_results(indexer: DatasetIndexer):
    """Print all indexer results and their shapes."""
    print("=" * 80)
    print("DATASET INDEXER RESULTS")
    print("=" * 80)
    
    print(f"\nTotal Headers: {len(indexer.headers)}")
    print(f"Max Index: {indexer.get_max_index()}")
    
    # MELTS_indices
    print("\n" + "=" * 80)
    print("MELTS_INDICES (Phase -> Component -> Index)")
    print("=" * 80)
    for phase, components in indexer.MELTS_indices.items():
        print(f"\n{phase}:")
        for component, idx in components.items():
            print(f"  {component}: {idx}")
    
    # Mass indices
    print("\n" + "=" * 80)
    print(f"MASS_INDICES (shape: {indexer.mass_indices.shape})")
    print("=" * 80)
    print(indexer.mass_indices)
    
    # Label indices
    print("\n" + "=" * 80)
    print("LABEL_INDICES (Phase -> List of indices)")
    print("=" * 80)
    for phase, indices in indexer.label_indices.items():
        print(f"{phase}: {indices}")
    
    # Label names
    print("\n" + "=" * 80)
    print(f"LABEL_NAMES (length: {len(indexer.label_names)})")
    print("=" * 80)
    print(indexer.label_names[:20], "..." if len(indexer.label_names) > 20 else "")
    
    # Detail label indices
    print("\n" + "=" * 80)
    print("DETAIL_LABEL_INDICES (Phase -> Component -> Detail Index)")
    print("=" * 80)
    for phase, components in indexer.detail_label_indices.items():
        print(f"\n{phase}:")
        for component, idx in components.items():
            print(f"  {component}: {idx}")
    
    # Label indices comp
    print("\n" + "=" * 80)
    print("LABEL_INDICES_COMP (Phase -> Array of component indices)")
    print("=" * 80)
    for phase, comp_inds in indexer.label_indices_comp.items():
        print(f"{phase}: {comp_inds}")
    
    # Phase dictionaries
    print("\n" + "=" * 80)
    print("PHASE DICTIONARIES")
    print("=" * 80)
    print(f"all_phases: {indexer.all_phases}")
    print(f"\nmass_phasedict: {indexer.mass_phasedict}")
    print(f"\ncomp_phasedict: {indexer.comp_phasedict}")
    print(f"\ncompositionally_variable_phases: {indexer.compositionally_variable_phases}")
    
    # Dimensions
    print("\n" + "=" * 80)
    print("DIMENSIONS")
    print("=" * 80)
    print(f"ncomps: {indexer.ncomps}")
    print(f"ncompsVaried: {indexer.ncompsVaried}")
    print(f"nphases: {indexer.nphases}")
    
    # Phase-to-component mappings
    print("\n" + "=" * 80)
    print(f"PHASE_TO_COMP_MAP (shape: {indexer.phaseToCompMap.shape})")
    print("=" * 80)
    print(f"First 10 rows, first 20 cols:\n{indexer.phaseToCompMap[:10, :20]}")
    
    print("\n" + "=" * 80)
    print(f"VARIED_TO_ALL_COMP (shape: {indexer.variedToAllComp.shape})")
    print("=" * 80)
    print(f"First 10 rows, first 20 cols:\n{indexer.variedToAllComp[:10, :20]}")
    
    print("\n" + "=" * 80)
    print(f"COMP_VARIABLE_IDMAT (shape: {indexer.comp_variable_IDMAT.shape})")
    print("=" * 80)
    print(f"First 10x10:\n{indexer.comp_variable_IDMAT[:10, :10]}")
    
    print("\n" + "=" * 80)
    print(f"FIXED_PHASE_TO_COMP_MAP (shape: {indexer.fixed_phaseToCompMap.shape})")
    print("=" * 80)
    print(f"First 20 cols:\n{indexer.fixed_phaseToCompMap[0, :20]}")
    
    print("\n" + "=" * 80)
    print(f"COMPOSITIONALLY_VARIABLE_SUBSET (shape: {indexer.compositionally_variable_subset.shape})")
    print("=" * 80)
    print(f"First 20 values: {indexer.compositionally_variable_subset[:20]}")
    
    # Component mappings
    print("\n" + "=" * 80)
    print("COMPONENT MAPPINGS")
    print("=" * 80)
    print(f"comp_map: {list(indexer.comp_map.keys())}")
    for phase, comp_list in indexer.comp_map.items():
        print(f"  {phase}: {comp_list}")
    
    print(f"\ncomp_binaries (shape: {indexer.comp_binaries.shape}): {indexer.comp_binaries}")
    print(f"\ncomp_mappings (shape: {indexer.comp_mappings.shape})")
    print(f"First 10x10:\n{indexer.comp_mappings[:10, :10]}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total phases: {len(indexer.MELTS_indices)}")
    print(f"Total components: {sum(len(comps) for comps in indexer.MELTS_indices.values())}")
    print(f"Phases with mass: {len(indexer.mass_indices)}")
    print(f"Compositionally variable phases: {len(indexer.compositionally_variable_phases)}")
    print(f"Label indices phases: {len(indexer.label_indices)}")
    print(f"Total label names: {len(indexer.label_names)}")


def load_headers_from_csv(csv_path):
    """Load headers from a CSV or Excel file."""
    import pandas as pd
    try:
        # Try Excel first
        if csv_path.endswith('.xlsx') or csv_path.endswith('.xls'):
            df = pd.read_excel(csv_path, nrows=0)  # Read only headers
        else:
            df = pd.read_csv(csv_path, nrows=0)  # Read only headers
        return list(df.columns)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test DatasetIndexer')
    parser.add_argument('--csv', type=str, help='Path to CSV file to test with')
    args = parser.parse_args()
    
    if args.csv and os.path.exists(args.csv):
        print(f"Loading headers from CSV: {args.csv}")
        headers = load_headers_from_csv(args.csv)
        if headers is None:
            print("Failed to load headers from CSV, using sample headers instead")
            headers = generate_sample_headers()
    else:
        print("Generating sample headers...")
        headers = generate_sample_headers()
    
    print(f"Using {len(headers)} headers")
    print(f"First 10 headers: {headers[:10]}")
    
    print("\nCreating DatasetIndexer...")
    try:
        indexer = DatasetIndexer(headers)
        print("DatasetIndexer created successfully!")
    except Exception as e:
        print(f"Error creating DatasetIndexer: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\nPrinting results...")
    print_indexer_results(indexer)
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
