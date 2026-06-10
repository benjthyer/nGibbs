"""
Generate MELTS column headers from a phase list, build a DatasetIndexer, and
print key values/stats for quick inspection.

Usage:
    python tests/indexer_report.py --phases olivine clinopyroxene plagioclase "melts-liquid"

If no phases are provided, a default demo list is used.
"""

import argparse
import sys
import pandas as pd
from pprint import pprint
from datetime import datetime

# Ensure project root is on sys.path
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.builder.indexer import DatasetIndexer, generate_column_headers
from src.ngibbs.config.constants import COMPONENTS_IN_PHASES, COMPOSITIONAL_COMPONENTS_IN_PHASES
from tests.test_utils import setup_test_logging



def build_and_report(idx, headers=None):
    
    #headers = generate_column_headers(phases)
    if headers is not None:
        print("\n=== Generated Headers ===")
        pprint(headers)

        print("\n=== Header count ===")
        print(len(headers))

    print("\n=== MELTS indices ===")
    print(idx.MELTS_indices)

    print("\n=== Mass indices ===")
    print(idx.mass_indices)

    print("\n=== Components in phases ===")
    pprint(idx.components_in_phases)

    print("\n=== Excluded components (by phase) ===")
    pprint(idx.EXCLUDED_COMPONENTS_BY_PHASE)

    print("\n=== Excluded phases===")
    print(idx.EXCLUDED_PHASES)

    print("\n=== Oxides to elements (if available) ===")
    ox_el = getattr(idx, "oxides_to_elements", {})
    pprint(ox_el)

    print("\n=== Components with extra oxides (if any) ===")
    extra = getattr(idx, "components_with_extra_oxides", {})
    pprint(extra)

    print("\n=== All phases ===")
    print(idx.all_phases)

    print("\n=== Label names ===")
    print(idx.label_names)

    print("\n=== Label indices ===")
    pprint(idx.label_indices)

    print("\n=== Detail label indices ===")
    pprint(idx.detail_label_indices)

    print("\n=== label indices comp ===")
    pprint(idx.label_indices_comp)

    print("\n=== Mass phase dict ===")
    pprint(idx.mass_phasedict)

    print("\n=== Compositionally variable phases ===")
    pprint(idx.compositionally_variable_phases)

    print("\n=== Comp phase dict ===")
    pprint(idx.comp_phasedict)

    print("\n=== ncomps / ncompsVaried / nphases ===")
    print(idx.ncomps, idx.ncompsVaried, idx.nphases)

    print("\n=== phaseToCompMap shape ===")
    print(getattr(idx, "phaseToCompMap", None).shape if hasattr(idx, "phaseToCompMap") else None)

    print("\n=== variedToAllComp shape ===")
    print(getattr(idx, "variedToAllComp", None).shape if hasattr(idx, "variedToAllComp") else None)

    print("\n=== compToOx shape ===")
    print(getattr(idx.ml_indexer, "compToOx").shape if hasattr(idx.ml_indexer, "compToOx") and getattr(idx.ml_indexer, "compToOx") is not None else None)

    print("\n=== compToOxLoad shape ===")
    print(getattr(idx.ml_indexer, "compToOxLoad").shape if hasattr(idx.ml_indexer, "compToOxLoad") and getattr(idx.ml_indexer, "compToOxLoad") is not None else None)

    # === Detailed ML Indexer Debug Info ===
    print("\n" + "="*60)
    print("=== ML INDEXER DETAILED ATTRIBUTES ===")
    print("="*60)

    ml = idx.ml_indexer

    print("\n=== ML Indexer: Elkeys ===")
    print(ml.Elkeys)

    print("\n=== ML Indexer: WRkeys ===")
    print(ml.WRkeys)

    print("\n=== ML Indexer: Oxides ===")
    print(ml.Oxides)

    print("\n=== ML Indexer: components_in_phases ===")
    pprint(ml.components_in_phases)

    print("\n=== ML Indexer: label_names (first 20) ===")
    print(ml.label_names[:20] if len(ml.label_names) > 20 else ml.label_names)

    print("\n=== ML Indexer: all_phases ===")
    print(ml.all_phases)

    print("\n=== ML Indexer: compositionally_variable_phases ===")
    print(ml.compositionally_variable_phases)

    print("\n=== ML Indexer: ncomps / ncompsVaried / nphases ===")
    print(f"ncomps: {ml.ncomps}, ncompsVaried: {ml.ncompsVaried}, nphases: {ml.nphases}")

    print("\n=== ML Indexer: projections_dir ===")
    print(ml.projections_dir)
    print(f"Directory exists: {ml.projections_dir.exists()}")

    # Transformation matrices
    print("\n=== ML Indexer: compToOxLoad ===")
    if ml.compToOxLoad is not None:
        print(f"Shape: {ml.compToOxLoad.shape}")
        print(f"Sample (first 3 rows, first 5 cols):\n{ml.compToOxLoad[:3, :5]}")
    else:
        print("None")

    print("\n=== ML Indexer: PxSpTransform ===")
    if ml.PxSpTransform is not None:
        print(f"Shape: {ml.PxSpTransform.shape}")
        print(f"Sample (first 3 rows, first 5 cols):\n{ml.PxSpTransform[:3, :5]}")
    else:
        print("None")

    print("\n=== ML Indexer: compToOx ===")
    if ml.compToOx is not None:
        print(f"Shape: {ml.compToOx.shape}")
        print(f"Sample (first 3 rows, first 5 cols):\n{ml.compToOx[:3, :5]}")
    else:
        print("None")

    print("\n=== ML Indexer: ElToOx ===")
    if ml.ElToOx is not None:
        print(f"Shape: {ml.ElToOx.shape}")
        print(f"Sample:\n{ml.ElToOx}")
    else:
        print("None")

    print("\n=== ML Indexer: oxToEl ===")
    if ml.OxToEl is not None:
        print(f"Shape: {ml.OxToEl.shape}")
        print(f"Sample (first 5 rows, first 5 cols):\n{ml.OxToEl[:5, :5]}")
    else:
        print("None")

    print("\n=== ML Indexer: boolTransCompToOx ===")
    if ml.boolTransCompToOx is not None:
        print(f"Shape: {ml.boolTransCompToOx.shape}")
        print(f"Sample (first 3 rows, first 5 cols):\n{ml.boolTransCompToOx[:3, :5]}")
    else:
        print("None")

    print("\n=== ML Indexer: MM (molar mass matrix) ===")
    if hasattr(ml, 'MM') and ml.MM is not None:
        print(f"Shape: {ml.MM.shape}")
        print(f"Diagonal (first 5): {ml.MM.diagonal()[:5]}")
    else:
        print("None")

    print("\n=== ML Indexer: phaseToCompMap ===")
    if ml.phaseToCompMap is not None:
        print(f"Shape: {ml.phaseToCompMap.shape}")
        print(f"Non-zero entries: {(ml.phaseToCompMap != 0).sum()}")
    else:
        print("None")

    print("\n=== ML Indexer: variedToAllComp ===")
    if ml.variedToAllComp is not None:
        print(f"Shape: {ml.variedToAllComp.shape}")
        print(f"Non-zero entries: {(ml.variedToAllComp != 0).sum()}")
    else:
        print("None")

    print("\n=== ML Indexer: compositionally_variable_binaries ===")
    print(ml.compositionally_variable_binaries)

    print("\n=== ML Indexer: compositionally_variable_subset ===")
    print(ml.compositionally_variable_subset)

    print("\n=== ML Indexer: comp_map ===")
    pprint(ml.comp_map)

    print("\n=== ML Indexer: comp_binaries ===")
    print(ml.comp_binaries)

    print("\n=== ML Indexer: comp_mappings ===")
    if ml.comp_mappings is not None:
        print(f"Shape: {ml.comp_mappings.shape}")
        print(f"Non-zero entries: {(ml.comp_mappings != 0).sum()}")
    else:
        print("None")

    print("\n" + "="*60)


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    setup_test_logging(
        log_filename=f"{Path(__file__).stem}_{timestamp}.txt",
        log_dir=Path(__file__).parent / "logs",
    )
    parser = argparse.ArgumentParser(description="DatasetIndexer reporter")
    parser.add_argument("--phases", nargs="*", help="List of phases", default=None)
    switches = parser.parse_args()

    pprint(COMPONENTS_IN_PHASES)
    pprint(COMPOSITIONAL_COMPONENTS_IN_PHASES)

    phases = switches.phases if switches.phases else [
        'olivine','orthopyroxene','clinopyroxene','spinel','plagioclase','k-feldspar','garnet',
        'nepheline','leucite','biotite','rhm-oxide','alloy-solid','alloy-liquid','apatite',
        'whitlockite','quartz','tridymite','cristobalite','muscovite','fluid','liquid'
    ]
    args = {
        'EXCLUDED_PHASES' : {'System_main'},
        'EXCLUDED_COMPONENTS_BY_PHASE' : {},
        'Elkeys' : ['Si', 'Ti', 'Al', 'Fe', 'Mg', 'Ca', 'Na', 'H', 'Mn', 'Ni']
    }

    #headers = generate_column_headers(phases)
    headers = pd.read_csv('src/builder/wslMELTS/DataProducts/102/MELTS102_ValidsetJan14BatchCooling.csv').columns.tolist()

    idx = DatasetIndexer(headers, **args)

    build_and_report(idx, headers)


if __name__ == "__main__":
    main()
