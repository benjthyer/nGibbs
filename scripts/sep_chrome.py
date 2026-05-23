"""Split MELTS datasets into Cr-present and Cr-absent subsets."""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

# Add repo root and src to path so repo-local packages resolve from repo root.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recipes.settings import internal_data_dir
from builder.processing.BigMetaTable import BigMetaTable
from nMELTS.utils.file_utils import delete_files_with_keyword

# Loading Params
MELTSModel = '102'          # MELTS model version: '102', '110', '120', 'p'
Date = 'Feb14'               # Date identifier for dataset naming
Mode = 'BatchCooling'     # Calculation mode: FxCrystCooling, BatchCooling, etc.

INTERNAL_DIR = Path(internal_data_dir(MELTSModel))
internal_data_path = str(INTERNAL_DIR)
ValidName = f"{internal_data_path}/MELTS{MELTSModel}_Validset{Date}{Mode}"
TrainName = f"{internal_data_path}/MELTS{MELTSModel}_Trainset{Date}{Mode}"


def _build_name(set_kind, date_tag):
    return str(INTERNAL_DIR / f"MELTS{MELTSModel}_{set_kind}{date_tag}{Mode}")


def _split_by_bulk_cr2o3(source_name, set_kind):
    table = BigMetaTable(source_name)
    bulk_indices = table.indexer.MELTS_indices.get("Bulk_comp", {})
    cr_col = bulk_indices.get("Cr2O3")
    if cr_col is None:
        raise KeyError("Cr2O3 column not found in Bulk_comp.")

    cr_indices = np.where(table.table[:, cr_col] > 0)[0]
    no_cr_table, cr_table = table.manual_split(cr_indices)

    cr_name = _build_name(set_kind, f"{Date}Cr")
    no_cr_name = _build_name(set_kind, f"{Date}NoCr")

    cr_table.save_csv_streaming(cr_name)
    cr_table.save_txt(cr_name)
    no_cr_table.save_csv_streaming(no_cr_name)
    no_cr_table.save_txt(no_cr_name)

    return cr_name, no_cr_name

if __name__ == "__main__":
    _split_by_bulk_cr2o3(ValidName, "Validset")
    _split_by_bulk_cr2o3(TrainName, "Trainset")
    delete_files_with_keyword(INTERNAL_DIR, "working", dry_run=False)