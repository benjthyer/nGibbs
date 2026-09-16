import warnings, numpy as np, torch, re
from pathlib import Path
warnings.filterwarnings('ignore')
from ngibbs.engine.models import HeFESToEmulatorCPU as E

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
D = str(REPO / 'data' / 'HeFESToWorkspace' / 'Htz_transition') + '/'

# ---- HeFESTo ground truth (fort.56)
hdr = open(D+'fort.56').readlines()[1].split()
raw = np.genfromtxt(D+'fort.56', skip_header=2, usecols=range(len(hdr)-1))
H = {k: raw[:, i] for i, k in enumerate(hdr[:-1])}
dom = [l.split()[-1] for l in open(D+'fort.56').readlines()[2:] if l.strip()]
P = H['P(GPa)']
print(f"HeFESTo: {len(P)} rows, P {P[0]:.2f}-{P[-1]:.2f} GPa, dP={np.median(np.diff(P))*1000:.0f} MPa")
print(f"  S = {H['S(J/g/K)'].mean():.6f} +/- {H['S(J/g/K)'].std():.2e} J/g/K   T {H['T(K)'].min():.0f}-{H['T(K)'].max():.0f} K")
print(f"  dominant phases: {sorted(set(dom))}")

# ---- control file composition (element moles)
lines = [l.rstrip('\n') for l in open(D+'control')]
els = {}
for l in lines[3:11]:
    p = l.split(); els[p[0]] = float(p[1])
print("  element moles:", {k: round(v,4) for k,v in els.items()},
      f"  Mg/Si = {els['Mg']/els['Si']:.4f}")

# ---- emulator at the SAME composition and the SAME entropy
S0 = float(H['S(J/g/K)'].mean())
EL = ['Si','Mg','Fe','Ca','Al','Na','Cr','O']
frac = np.array([els[k] for k in EL]); frac = frac/frac.sum()
hS = ['P(GPa)(System_main)','S(J/g/K)(System_main)'] + EL
f = np.zeros((len(P), len(hS)), dtype=np.float32)
f[:,0] = P; f[:,1] = S0; f[:,2:] = frac
out = E.ForwardMB(f, headers=hS, outputs=['component_moles','temperature'])
Te = out['temperature'].detach().numpy().ravel()
cm = out['component_moles'].to(torch.float64)
pr = E.get_property_hefesto_vectorized_from_assemblage(
        cm, torch.tensor(np.stack([P, Te],1), dtype=torch.float64),
        ('rho','Vp','Vs','S'))
np.savez(str(HERE / 'cmp.npz'), P=P, Th=H['T(K)'], Te=Te, rhoh=H['rho(g/cm^3)'],
         rhoe=pr['rho'], Vsh=H['VS(km/s)'], Vse=pr['Vs'], Vph=H['VP(km/s)'],
         Vpe=pr['Vp'], Se=pr['S'], cm=cm.numpy(), S0=S0,
         frac=frac, depth=H['depth_PREM(km)'])
r = lambda a,b: (a-b)
print(f"\n  emulator S check: {pr['S'].mean():.4f} (target {S0:.4f})")
for nm, h, e in (('T (K)', H['T(K)'], Te), ('rho', H['rho(g/cm^3)'], pr['rho']),
                 ('Vs', H['VS(km/s)'], pr['Vs']), ('Vp', H['VP(km/s)'], pr['Vp']),
                 ):
    d = r(e,h); print(f"  {nm:8s} mean {d.mean():+9.4f}  rms {np.sqrt((d**2).mean()):8.4f}  max|d| {np.abs(d).max():8.4f}")
