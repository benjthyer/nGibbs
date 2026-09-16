import numpy as np, re
from pathlib import Path
HERE = Path(__file__).resolve().parent
D = str(HERE.parents[2] / 'data' / 'HeFESToWorkspace' / 'Htz_transition') + '/'
d=np.load(HERE / 'dnd.npz'); names=[str(x) for x in d['names']]; dndt,dndp=d['dndt'],d['dndp']
L=open(D+'fort.99').readlines(); cols=L[0].split()
num=re.compile(r'^[\s\-\d\.eEdD\+]+$')
A=np.array([[float(x) for x in l.split()] for l in L[1:]
            if num.match(l.rstrip('\n')) and len(l.split())==len(cols)])
P,T = A[:,0], A[:,2]
sp = cols[3:]
print(f"fort.99 has {len(sp)} data columns vs {len(names)} species in fort.42")
print("  columns not in fort.42:", [c for c in sp if c not in names])
k = [sp.index(s) for s in names]                     # name-matched, 73 species
N = A[:, 3:][:, k]
tot = N.sum(1)
print(f"\nSPECIES total: {tot.min():.4f} -> {tot.max():.4f}  (varies {100*(tot.max()-tot.min())/tot.mean():.1f}%)  NOT conserved")
dTdP = np.gradient(T,P)
i = int(np.argmax(np.abs(np.gradient(tot,P))))
lhs = np.gradient(tot,P)[i]; rhs = dndp.sum(1)[i] + dndt.sum(1)[i]*dTdP[i]
print(f"  chain rule on the total at P={P[i]:.2f}: numeric {lhs:+.3f}  chain {rhs:+.3f}  ratio {rhs/lhs:.3f}")
