"""Prepare a single HeFESTo run directory for one bulk composition over a P-T grid.

No command-line configurability, edit the CONFIG block below and run.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'

# Add repo root and src to path so repo-local packages resolve from repo root.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from builder.HeFESTo.HeFESTo_functions import prepare_HeFESTo_single_composition_directory

# ----------------------------- CONFIG ------------------------------------

# Where to write the 'control' file for this run.
DIRECTORY = REPO_ROOT / 'data' / 'HeFESToWorkspace' / 'SingleComposition'

# Bulk composition in wt% oxides. FeO carries total iron; it is speciated
# into FeO/Fe2O3 below using FE3_FET.
OXIDE_WT = {
    'SiO2': 44.71,
    'MgO': 39.61,
    'FeO': 8.18,
    'CaO': 3.17,
    'Al2O3': 3.59,
    'Na2O': 0.29,
    'Cr2O3': 0.45,
}
FE3_FET = 0.0  # Target Fe3+/Fetotal molar ratio

# Pressure range (GPa) and number of steps in the grid.
P_MIN, P_MAX, P_STEPS = 0.0, 24.0, 24

# Temperature range (K) and number of steps in the grid.
T_MIN, T_MAX, T_STEPS = 1600.0, 2200.0, 24

TARGET_TOTAL_MOLES = 24.0

# Path to a control template file, or a directory containing
# shallowHeFESTo/deepHeFESTo templates (auto-selected using a 23 GPa cutoff
# on P_MIN).
CONTROL_DIR = SRC_DIR / 'builder' / 'HeFESTo' / 'batch'

# ---------------------------------------------------------------------------

if __name__ == '__main__':
    prepare_HeFESTo_single_composition_directory(
        directory=DIRECTORY,
        oxide_wt=OXIDE_WT,
        p_min=P_MIN,
        p_max=P_MAX,
        p_steps=P_STEPS,
        t_min=T_MIN,
        t_max=T_MAX,
        t_steps=T_STEPS,
        CONTROL_DIR=CONTROL_DIR,
        fe3_fet=FE3_FET,
        target_total_moles=TARGET_TOTAL_MOLES,
    )
    print(f'HeFESTo run directory prepared at: {DIRECTORY}')
