"""
Benchmark test comparing hefesto_vec against HeFESTo Fortran outputs.

Per-species comparison uses fort.813 (exact match expected).
Aggregate comparison uses fort.58 with APPROXIMATE X (pure end-members per phase)
- expect ~1% error in Vp/Vs until exact n(ispec) are available from Fortran output.
"""
import sys, warnings, numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from hefesto_vec.params    import load_control
from hefesto_vec.volume    import solve_volume
from hefesto_vec.therm_props import compute_therm_props
from hefesto_vec.aggregate import vrh_average

BENCH = THIS_DIR / 'BENCHMARK'
CTRL = BENCH / 'control'
PDIR = THIS_DIR / 'HeFESTo_Parameters_010123'

params = load_control(str(CTRL), param_dir_override=str(PDIR))
S = params.nspec
print(params.snames)

# ── Parse fort.813: per-species thermo at each P,T ───────────────────────────
# Format (from summary): ispec T Vi Vo Cp Cv γ q K Ks alp Ftot ph ent deltas
#                        tcal ζ Gsh uth uto thet pzp Vdeb gamdeb
# Column indices (0-based after stripping 'THERM' tag and ispec):
#   0=T, 1=Vi, 2=Vo, 3=Cp, 4=Cv, 5=γ, 6=q, 7=K, 8=Ks, 9=alp, 10=Ftot,
#   11=ph, 12=ent, 13=deltas, 14=thet (tcal), 15=pzp, 16=Gsh, 17=uth, 18=uto,
#   19=thet2, 20=?, 21=Vdeb, 22=gamdeb
f813 = {}   # {ispec_1based: {P_idx: row_array}}
with open(BENCH / 'fort.813') as fh:
    for line in fh:
        parts = line.split()
        if not parts or parts[0] != 'THERM':
            continue
        isp = int(parts[1])   # 1-based
        vals = np.array(parts[2:], dtype=float)
        f813.setdefault(isp, []).append(vals)

# ── Parse fort.58: aggregate VRH data ────────────────────────────────────────
# Header: Pi depth Ti rho KS G VBh VSh VPh KSr Gr VBr VSr VPr KSv Gv VBv VSv VPv
f58 = []
with open(BENCH / 'fort.58') as fh:
    for line in fh:
        parts = line.split()
        if not parts or not parts[0].replace('.','').replace('-','').replace('e','').replace('E','').isdigit():
            continue
        if len(parts) >= 9:
            try:
                f58.append([float(x) for x in parts[:19]])
            except ValueError:
                pass
f58 = np.array(f58)

# ── Parse fort.66: phase mole fractions ──────────────────────────────────────
f66 = []
with open(BENCH / 'fort.66') as fh:
    lines = fh.readlines()
header_66 = lines[0].split()   # Pi depth Ti plg sp opx c2c cpx wo ... ol ...
ph_names_66 = header_66[3:]    # phase names in fort.66 order
for line in lines[1:]:
    parts = line.split()
    if not parts:
        continue
    try:
        f66.append([float(x) for x in parts])
    except ValueError:
        pass
f66 = np.array(f66)

print("Benchmark data loaded:")
print(f"  fort.813: {len(f813)} species, {len(f813[1])} P,T points")
print(f"  fort.58:  {len(f58)} P,T points")
print(f"  fort.66:  {len(f66)} P,T points, {len(ph_names_66)} phases")
print(f"  fort.66 phase names: {ph_names_66}")

# ════════════════════════════════════════════════════════════════════════════
# Test 1: Per-species properties for fo at P=0, T=1600K
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 1: Per-species properties for fo at P=0, T=1600 K")
print("="*70)

fo_idx = params.spec_index['fo']   # 0-based
P0 = np.array([0.0])
T0 = np.array([1600.0])
apar = params.apar
active_fo = np.zeros((1, S), dtype=bool)
active_fo[0, fo_idx] = True

# pre-load DOS tables
from hefesto_vec.dos_tables import load_tables
load_tables(str(THIS_DIR / 'dos_tables.npz'))
sol = solve_volume(P0, T0, apar, active_fo, max_iter=50, tol_GPa=1e-10)
V_fo = sol['V'][0, fo_idx]

B, Sp = 1, S
Ti_bs = T0[:, None] * np.ones((B, Sp))
Pi_bs = P0[:, None] * np.ones((B, Sp))
To_bs = apar[None, :, 3] * np.ones((B, Sp))

iltyp_arr = np.zeros(S, dtype=np.int32)
for ispec in range(S):
    iph = params.lphase[ispec]
    if iph >= 0:
        iltyp_arr[ispec] = params.phase_iltyp[iph]

props = compute_therm_props(
    V=sol['V'], Ti=Ti_bs, To=To_bs, Pi=Pi_bs,
    gamma=sol['gamma'], q=sol['q'], qp=sol['qp'],
    etas=sol['etas'], detasdv=sol['detasdv'],
    Uth=sol['Uth'], Uto=sol['Uto'], Cv=sol['Cv'], Cvo=sol['Cvo'],
    apar=apar, iltyp_arr=iltyp_arr,
)

# fort.813 for fo (species 29 in 1-based = index 0 in f813[29])
f813_fo = np.array(f813[fo_idx + 1][0])   # first P,T point (P=0)
# Column mapping: T=0, Vi=1, Vo=2, Cp=3, Cv=4, γ=5, q=6, K=7, Ks=8, alp=9,
#                Ftot=10, ph=11, ent=12, deltas=13, tcal=14, pzp=15, Gsh=16,
#                uth=17, uto=18, thet=19, ?=20, Vdeb=21, gamdeb=22

print(f"\n{'Property':12s}  {'Python':>14s}  {'Fortran':>14s}  {'Err %':>8s}")
print("-"*55)

checks = [
    ('V (cm³/mol)',   V_fo,                           f813_fo[1]),
    ('K (GPa)',       props['K'][0, fo_idx],          f813_fo[7]),
    ('Ks (GPa)',      props['Ks'][0, fo_idx],         f813_fo[8]),
    ('Gsh (GPa)',     props['Gsh'][0, fo_idx],        f813_fo[16]),
    ('alp (1e-5/K)',  props['alp'][0, fo_idx]*1e5,    f813_fo[9]*1e5),
    ('Cp (J/mol/K)',  props['Cp'][0, fo_idx],         f813_fo[3]),
    ('Cv (J/mol/K)',  props['Cv'][0, fo_idx],         f813_fo[4]),
    ('Uth (J/mol)',   sol['Uth'][0, fo_idx],          f813_fo[17]),
    ('Uto (J/mol)',   sol['Uto'][0, fo_idx],          f813_fo[18]),
    ('gamma',         sol['gamma'][0, fo_idx],        f813_fo[5]),
]

all_pass = True
for name, py_val, fort_val in checks:
    err = abs(py_val - fort_val) / max(abs(fort_val), 1e-15) * 100
    status = "✓" if err < 0.01 else ("~" if err < 1.0 else "✗")
    if err >= 0.01:
        all_pass = False
    print(f"{name:12s}  {py_val:14.6f}  {fort_val:14.6f}  {err:8.4f}%  {status}")

# ════════════════════════════════════════════════════════════════════════════
# Test 2: Aggregate at P=0, T=1600K (approximate pure end-member composition)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 2: Aggregate at P=0, T=1600 K (pure end-members per phase)")
print("="*70)
print("NOTE: X uses fort.66 phase fractions mapped to single dominant species")
print("      (fo, en, di, an, sp). Fe-substitution effects are ~1% level.")

# fort.66 at P=0:  Pi depth Ti [phase fractions...]
row0 = f66[0]   # P=0 row
Pi_agg = row0[0]
Ti_agg = row0[2]
ph_fracs_66 = row0[3:]   # one per phase in fort.66 order

# Map fort.66 phase names to our phase_members
# fort.66 header (after Pi,depth,Ti): plg sp opx c2c cpx wo pwo gt cpv ol wa ri il pv ppv cf nal mw qtz coes st apbo ky neph fea feg fee
ph66_map = {name: frac for name, frac in zip(ph_names_66, ph_fracs_66)}
print(f"\nActive phase fractions: { {k:v for k,v in ph66_map.items() if v > 1e-6} }")

# Build X (B=1, S=73) using dominant end-member per phase
X = np.zeros((1, S))
# ol → fo (species 28 0-indexed)
if 'ol' in ph66_map and ph66_map['ol'] > 0:
    X[0, params.spec_index['fo']] = ph66_map['ol']
# opx → en (species 6)
if 'opx' in ph66_map and ph66_map['opx'] > 0:
    X[0, params.spec_index['en']] = ph66_map['opx']
# cpx → di (species 12)
if 'cpx' in ph66_map and ph66_map['cpx'] > 0:
    X[0, params.spec_index['di']] = ph66_map['cpx']
# plg → an (species 0)
if 'plg' in ph66_map and ph66_map['plg'] > 0:
    X[0, params.spec_index['an']] = ph66_map['plg']
# sp → sp (species 2)
if 'sp' in ph66_map and ph66_map['sp'] > 0:
    X[0, params.spec_index['sp']] = ph66_map['sp']

active_agg = X > 0.0

print(f"Active species: {[params.snames[i] for i in range(S) if X[0,i]>0]}")
print(f"X (non-zero): {[(params.snames[i], f'{X[0,i]:.5f}') for i in range(S) if X[0,i]>0]}")

# Solve volumes for all active species
P_agg = np.array([Pi_agg])
T_agg = np.array([Ti_agg])
sol_agg = solve_volume(P_agg, T_agg, apar, active_agg, max_iter=50, tol_GPa=1e-10)

Ti_bs_agg = T_agg[:, None] * np.ones((1, S))
Pi_bs_agg = P_agg[:, None] * np.ones((1, S))
To_bs_agg = apar[None, :, 3] * np.ones((1, S))

props_agg = compute_therm_props(
    V=sol_agg['V'], Ti=Ti_bs_agg, To=To_bs_agg, Pi=Pi_bs_agg,
    gamma=sol_agg['gamma'], q=sol_agg['q'], qp=sol_agg['qp'],
    etas=sol_agg['etas'], detasdv=sol_agg['detasdv'],
    Uth=sol_agg['Uth'], Uto=sol_agg['Uto'],
    Cv=sol_agg['Cv'], Cvo=sol_agg['Cvo'],
    apar=apar, iltyp_arr=iltyp_arr,
)

wm_arr = apar[:, 2]
agg = vrh_average(
    X=X,
    V=props_agg['V_corrected'],
    K=props_agg['K'],
    Ks=props_agg['Ks'],
    Gsh=props_agg['Gsh'],
    alp=props_agg['alp'],
    Cp=props_agg['Cp'],
    wm=wm_arr,
    Ti=T_agg,
    phase_members=params.phase_members,
    active=active_agg,
)

# fort.58 at P=0: Pi depth Ti rho KS G VBh VSh VPh KSr Gr VBr VSr VPr KSv Gv VBv VSv VPv
fort58_p0 = f58[0]
print(f"\n{'Property':12s}  {'Python':>12s}  {'Fortran':>12s}  {'Err %':>8s}")
print("-"*55)

checks_agg = [
    ('rho (g/cm³)', agg['rho'][0],  fort58_p0[3]),
    ('Ks_h (GPa)',  agg['Kh'][0],   fort58_p0[4]),
    ('G_h (GPa)',   agg['Gh'][0],   fort58_p0[5]),
    ('Vb_h (km/s)', agg['Vb'][0],   fort58_p0[6]),
    ('Vs_h (km/s)', agg['Vs'][0],   fort58_p0[7]),
    ('Vp_h (km/s)', agg['Vp'][0],   fort58_p0[8]),
    ('Ks_r (GPa)',  agg['Kr'][0],   fort58_p0[9]),
    ('G_r (GPa)',   agg['Gr'][0],   fort58_p0[10]),
    ('Vb_r (km/s)', np.sqrt(np.maximum(agg['Kr'][0]/agg['rho'][0],0)), fort58_p0[11]),
    ('Vs_r (km/s)', agg['Vsr'][0],  fort58_p0[12]),
    ('Vp_r (km/s)', agg['Vpr'][0],  fort58_p0[13]),
    ('Ks_v (GPa)',  agg['Kv'][0],   fort58_p0[14]),
    ('G_v (GPa)',   agg['Gv'][0],   fort58_p0[15]),
    ('Vs_v (km/s)', agg['Vsv'][0],  fort58_p0[17]),
    ('Vp_v (km/s)', agg['Vpv'][0],  fort58_p0[18]),
]

for name, py_val, fort_val in checks_agg:
    err = abs(py_val - fort_val) / max(abs(fort_val), 1e-15) * 100
    status = "✓" if err < 0.1 else ("~" if err < 2.0 else "✗")
    print(f"{name:12s}  {py_val:12.5f}  {fort_val:12.5f}  {err:8.4f}%  {status}")

print(f"\nNote: Fortran reference rho={fort58_p0[3]:.5f}, Vs={fort58_p0[7]:.5f}, Vp={fort58_p0[8]:.5f}")
print(f"Expected discrepancy ~1-2% due to Fe solid solution (fo vs Fo89Fa11 etc.)")

# Hill velocity check
vsv_fort = fort58_p0[17]
vsr_fort = fort58_p0[12]
vsh_fort = fort58_p0[7]
print(f"\nHill velocity verification (Fortran):")
print(f"  (Vsv+Vsr)/2 = ({vsv_fort:.5f}+{vsr_fort:.5f})/2 = {(vsv_fort+vsr_fort)/2:.5f}")
print(f"  Vsh (stored) = {vsh_fort:.5f}")
print(f"  Match: {abs((vsv_fort+vsr_fort)/2 - vsh_fort) < 0.0001}")
