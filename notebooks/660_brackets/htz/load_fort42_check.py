import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
D = str(HERE.parents[2] / 'data' / 'HeFESToWorkspace' / 'Htz_transition') + '/'
lines=[l for l in open(D+'fort.42')]
blocks=[i for i,l in enumerate(lines) if 'dndt and dndp' in l]
nsp=blocks[1]-blocks[0]-1
names=[lines[blocks[0]+1+k].split()[1] for k in range(nsp)]
dndt=np.zeros((len(blocks),nsp)); dndp=np.zeros_like(dndt)
for b,i in enumerate(blocks):
    for k in range(nsp):
        p=lines[i+1+k].split(); dndt[b,k]=float(p[2]); dndp[b,k]=float(p[3])
print(f"{len(blocks)} blocks x {nsp} species  (fort.56 has 501 rows)")
d=np.load(HERE / 'cmp.npz'); P=d['P']
assert len(blocks)==len(P)
act=np.abs(dndt).max(0)>0
print("species with nonzero dn/dT anywhere:", [names[k] for k in np.where(act)[0]])
i=int(np.argmax(np.abs(dndp).sum(1)))
print(f"\npeak |dn/dP| at P={P[i]:.3f} GPa (transition was 23.34-23.42):")
for k in np.argsort(-np.abs(dndp[i]))[:8]:
    print(f"   {names[k]:6s} dndt={dndt[i,k]:+.6f}  dndp={dndp[i,k]:+.6f}")
# width of the dn/dP feature vs the Vs window
tot=np.abs(dndp).sum(1); half=tot.max()/2
sup=np.where(tot>half)[0]
Z=d['depth']
print(f"\nFWHM of total |dn/dP|: {Z[sup[-1]]-Z[sup[0]]:.2f} km   (Vs d2 window was 2.00 km)")
print(f"nonzero |dn/dP| support: {Z[np.where(tot>1e-9)[0][-1]]-Z[np.where(tot>1e-9)[0][0]]:.2f} km")
np.savez(HERE / 'dnd.npz', dndt=dndt, dndp=dndp, names=np.array(names), P=P)
