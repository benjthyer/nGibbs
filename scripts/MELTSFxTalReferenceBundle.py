"""Build an ML-ready bundle from the MELTS 1.2.0 (120) fractional crystallization reference runs.

1. Import the FxTal reference runs (MORB, Lherzolite) with import_MELTS_components().
2. Pick N_PICKS evenly spaced subliquidus bulk compositions from each FxTal run.
3. Run batch (equilibrium) cooling calculations on each picked composition at the
   original pressure, fO2 buffer offset and temperature path of its parent run.
4. Copy the original FxTal runs into the same workspace and import all of them.
5. Read the combined MELTStable with BigMetaTable and export with resampling_to_datasets().

The linux virtual environment is loaded by calling: source ~/melts_env/venv/bin/activate
"""

import sys
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from recipes.settings import internal_data_dir, internal_train_dir
from builder.indexer import generate_column_headers, DatasetIndexer
from builder.alphamelts.engine import alphamelts_functions as af
from builder.processing.BigMetaTable import BigMetaTable
from builder.processing.MLexporter import resampling_to_datasets

MELTSModel = '120'
batch_file = '120batch'
N_PICKS = 20
delta = -1

STANDARDS_DIR = REPO_ROOT / 'src' / 'ngibbs' / 'deployment_tests' / 'MELTSIsobaricStandards' / '120'
FXTAL_DIR = STANDARDS_DIR / 'NoCr_v232_dndT_FxTal_Test'
FXTAL_IMPORT_WS = STANDARDS_DIR / 'NoCr_v232_dndT_FxTal_Import'     # FxTal runs only (step 1)
BATCH_WS = STANDARDS_DIR / 'NoCr_v232_dndT_FxTal_BatchWorkspace'   # batch runs + FxTal runs (steps 3-4)

# Conditions of the original FxTal runs (from their input.melts files)
references = {
    'MORB':       {'Pressure': 500,  'startT': 1400, 'endT': 700, 'fO2': 0.0},
    'Lherzolite': {'Pressure': 9000, 'startT': 1800, 'endT': 700, 'fO2': -2.0},
}
ref_names = list(references.keys())

OUT_DIR = Path(internal_data_dir(MELTSModel))
OUT_DIR.mkdir(parents=True, exist_ok=True)
FxTalName = str(OUT_DIR / 'MELTS120_RefStandardsFxTalOpenNoCr')
CombinedName = str(OUT_DIR / 'MELTS120_RefStandardsFxTalBatchOpenNoCr')

# Same phase list / zeroed oxides as MELTedGEOROCsepCr.py uses for 120 NoCr
allowed_phases = ['olivine', 'orthopyroxene', 'clinopyroxene', 'spinel', 'plagioclase', 'k-feldspar', 'garnet',
                  'nepheline', 'leucite', 'biotite', 'rhm-oxide', 'apatite', 'whitlockite', 'quartz', 'tridymite',
                  'muscovite', 'fluid', 'liquid', 'hornblende', 'alloy-solid', 'alloy-liquid', 'graphite', 'calcite',
                  'dolomite', 'magnesite']
ZeroOxides = ['MnO', 'NiO', 'Cr2O3']
headers = generate_column_headers(allowed_phases, mode=MELTSModel, zeroOxides=ZeroOxides)
indexer = DatasetIndexer(headers, OXYGEN='open', MODEL='MELTS')


def _remove_outputs(name):
    """import_MELTS_components appends, so start from clean output files."""
    for ext in ['.csv', '.txt', '.npy']:
        if os.path.exists(name + ext):
            os.remove(name + ext)


def _copy_run(src, dst):
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('*:Zone.Identifier'))


# ---------------------------------------------------------------------------
# 1. Import the FxTal reference runs
# ---------------------------------------------------------------------------
if FXTAL_IMPORT_WS.exists():
    shutil.rmtree(FXTAL_IMPORT_WS)
FXTAL_IMPORT_WS.mkdir(parents=True)
for i, ref in enumerate(ref_names):
    _copy_run(FXTAL_DIR / ref, FXTAL_IMPORT_WS / f'Simulation{i}')
# import_MELTS_components expects one non-simulation file (runall.sh) in the workspace
(FXTAL_IMPORT_WS / 'runall.sh').touch()

_remove_outputs(FxTalName)
fx_batchnames = np.array([f'{i}:{ref}-FxTal:{batch_file}' for i, ref in enumerate(ref_names)], dtype=object)
af.import_MELTS_components(EnsembleLocation=str(FXTAL_IMPORT_WS), batchname=fx_batchnames, indexer=indexer,
                           fO2Arr=np.array([references[r]['fO2'] for r in ref_names]), dataname=FxTalName + '.csv')

# ---------------------------------------------------------------------------
# 2. Pick evenly spaced subliquidus bulk compositions from each FxTal run
# ---------------------------------------------------------------------------
fx_table = pd.read_csv(FxTalName + '.csv')
with open(FxTalName + '.txt') as f:
    fx_runs = np.array([line.split(' ')[0].split(':')[1] for line in f])
assert len(fx_runs) == len(fx_table), 'FxTal metadata and table lengths differ'

bulk_cols = [h for h in fx_table.columns if h.endswith('(Bulk_comp)') and not h.startswith('mass')]
oxides = [h.split('(')[0] for h in bulk_cols]
solid_mass_cols = [indexer.database_headers[i] for i in indexer.mass_indices[:-1]]

in_rows, end_temps, batch_meta, batch_fO2 = [], [], [], []
for ref in ref_names:
    cond = references[ref]
    run_rows = fx_table[fx_runs == f'{ref}-FxTal']
    subliquidus = run_rows[(run_rows[solid_mass_cols] > 0).any(axis=1)].sort_values('Temperature(System_main)', ascending=False)
    picks = np.unique(np.round(np.linspace(0, len(subliquidus) - 1, N_PICKS)).astype(int))
    assert len(picks) == N_PICKS, f'{ref}: only {len(subliquidus)} subliquidus rows to pick from'
    for k, p in enumerate(picks):
        row = subliquidus.iloc[p]
        comp = row[bulk_cols].to_numpy(dtype=float)
        comp = 100 * comp / comp.sum()
        in_rows.append([cond['Pressure'], cond['startT'], cond['fO2']] + list(comp))
        end_temps.append(cond['endT'])
        batch_fO2.append(cond['fO2'])
        T_pick = row['Temperature(System_main)']
        batch_meta.append(f'{len(batch_meta)}:{ref}-Batch{k:02d}-T{T_pick:.0f}:{batch_file}')
        print(f'{ref} pick {k:02d}: FxTal T = {T_pick:.0f} C')

in_array = np.round(np.array(in_rows), 5)
keys = np.array(['Pressure', 'Temperature', 'fO2'] + oxides)
pd.DataFrame(in_array, columns=keys, index=[m.split(':')[1] for m in batch_meta]).to_csv(
    CombinedName + '_batch_inputs.csv')

# ---------------------------------------------------------------------------
# 3. Batch cooling calculations on the picked compositions
# ---------------------------------------------------------------------------
af.forward_ensemble(in_array, keys=keys, batchname=batch_file, only_phases=indexer.get_phase_list(),
                    end=np.array(end_temps), fxtal=False, EnsembleLocation=str(BATCH_WS), WSL=True,
                    compression=False, delta=delta)

# ---------------------------------------------------------------------------
# 4. Add the original FxTal runs and import everything
# ---------------------------------------------------------------------------
n_batch = len(in_array)
for i, ref in enumerate(ref_names):
    _copy_run(FXTAL_DIR / ref, BATCH_WS / f'Simulation{n_batch + i}')
    batch_meta.append(f'{n_batch + i}:{ref}-FxTal:{batch_file}')
    batch_fO2.append(references[ref]['fO2'])

_remove_outputs(CombinedName)
failures = af.import_MELTS_components(EnsembleLocation=str(BATCH_WS), batchname=np.array(batch_meta, dtype=object),
                                      indexer=indexer, fO2Arr=np.array(batch_fO2), dataname=CombinedName + '.csv')
print(f'Failed simulations: {failures}')

# ---------------------------------------------------------------------------
# 5. ML-ready bundle
# ---------------------------------------------------------------------------
RefMELTS = BigMetaTable(CombinedName, Model='MELTS', OXYGEN='open')
bundle_path = resampling_to_datasets(
    RefMELTS, [[1, 1]],
    featureNames=['Pressure(System_main)', 'Temperature(System_main)', 'logfO2-QFM(System_main)'],
    bundle_name=Path(CombinedName).name + '.tar.gz',
)
train_dir = Path(internal_train_dir(MELTSModel))
train_dir.mkdir(parents=True, exist_ok=True)
final_path = shutil.move(str(bundle_path), str(train_dir / Path(bundle_path).name))
print(f'Bundle written to {final_path}')
