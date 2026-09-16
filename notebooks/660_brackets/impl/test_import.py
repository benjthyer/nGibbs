import sys, warnings, os, glob
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/claude/imp')
import numpy as np, pandas as pd
from builder.indexer import DatasetIndexer, generate_column_headers_hefesto
from ngibbs.config.constants import COMPOSITIONAL_COMPONENTS_IN_PHASES_HEFESTO as CC
from builder.HeFESTo.HeFESTo_functions import import_HeFESTo_components
from builder.HeFESTo.HeFESTo_derivative_import import (
    import_HeFESTo_derivatives, verify_chain_rule, load_fort42)

ex = {'System_main','Bulk_comp','Bulk_comp_elements'}
idx = DatasetIndexer(generate_column_headers_hefesto([p for p in CC if p not in ex]),
                     OXYGEN='closed', MODEL='HeFESTo')
for f in glob.glob('/home/claude/imp/out_*'): os.remove(f)

print("[verify] chain rule on Simulation1 (independent of the importer):")
v = verify_chain_rule('/home/claude/imp/ws/Simulation1')
print("   " + "  ".join(f"{k}:{val:.3f}" for k,val in v.items() if not k.startswith('_'))
      + f"   PASS={v['_pass']}")

main_csv='/home/claude/imp/out_main.csv'
dp_csv='/home/claude/imp/out_dndP.csv'; dt_csv='/home/claude/imp/out_dndT.csv'
mf='/home/claude/imp/out_manifest.csv'

p,m,e = import_HeFESTo_components(workspace_dir='/home/claude/imp/ws', indexer=idx, dataname=main_csv)
print(f"\n[main]  passed={len(p)} malformed={len(m)} empty={len(e)}")
pd_, md, m42 = import_HeFESTo_derivatives('/home/claude/imp/ws', idx, dp_csv, dt_csv, manifest_name=mf)
print(f"[deriv] passed={pd_} malformed={md} missing_fort42={m42}")

A = pd.read_csv(main_csv); DP = pd.read_csv(dp_csv); DT = pd.read_csv(dt_csv)
print(f"\n[shape] main {A.shape}  dndP {DP.shape}  dndT {DT.shape}")
print(f"[schema] identical headers: {list(A.columns)==list(DP.columns)==list(DT.columns)}")

# Simulation2 has no fort.42, so main has 2 sims and the shadows have 1.
n1 = len(pd.read_csv(mf).iloc[0].to_dict() and DP)
print(f"[align] P(GPa) of shadow rows equals first {len(DP)} main rows: "
      f"{np.allclose(DP['P(GPa)(System_main)'].values, A['P(GPa)(System_main)'].values[:len(DP)])}")
print(f"[align] T(K)   likewise: "
      f"{np.allclose(DP['T(K)(System_main)'].values, A['T(K)(System_main)'].values[:len(DP)])}")

man = pd.read_csv(mf); print(f"\n[manifest]\n{man.to_string(index=False)}")

# normalisation: shadow value * N_el must equal the raw fort.42 number
names, dndt_raw, dndp_raw = load_fort42('/home/claude/imp/ws/Simulation1/fort.42')
N_el = float(man['N_el'].iloc[0])
col = [c for c in DP.columns if c.startswith('mg-ringwoodite(')]
k = names.index('mgri')
err = np.abs(DP[col[0]].values*N_el - dndp_raw[:len(DP), k]).max()
print(f"\n[normalise] col {col[0]!r}: max |shadow*N_el - fort.42| = {err:.3e}  (N_el={N_el:.5f})")

# derivative columns must not be identically zero where the abundance moves
mov = (np.abs(DP[col[0]].values) > 0).sum()
print(f"[content]  {mov}/{len(DP)} rows carry nonzero d(mg-ringwoodite)/dP")
