import numpy as np, re
from pathlib import Path
HERE = Path(__file__).resolve().parent
D = str(HERE.parents[2] / 'data' / 'HeFESToWorkspace' / 'Htz_transition') + '/'
d = np.load(HERE / 'dnd.npz'); dndt, dndp, names, P = d['dndt'], d['dndp'], [str(x) for x in d['names']], d['P']
T = np.load(HERE / 'cmp.npz')['Th']

L = open(D+'fort.99').readlines()
cols = L[0].split()
num = re.compile(r'^[\s\-\d\.eEdD\+]+$')
rows = [l for l in L[1:] if num.match(l.rstrip('\n')) and len(l.split()) == len(cols)]
print(f"fort.99: {len(cols)} cols, {len(L)-1} lines, {len(rows)} numeric rows kept "
      f"({len(L)-1-len(rows)} WARNING/diagnostic lines dropped)")
A = np.array([[float(x) for x in l.split()] for l in rows])
sp = cols[3:]
N  = A[:, 3:]
print("species columns match fort.42 order:", sp == names[:len(sp)])
assert len(A) == len(P)

dTdP = np.gradient(T, P)
print("\nchain rule   dn/dP|_isentrope  =  dn/dP|_T + dn/dT|_P * dT/dP")
print("species   numeric      chain     ratio    corr")
for s in ('mgri','mgpv','pe','mgil','st','mgmj','feri','fepv'):
    if s not in sp: continue
    k = sp.index(s); j = names.index(s)
    nu = np.gradient(N[:, k], P)
    ch = dndp[:, j] + dndt[:, j]*dTdP
    m = np.abs(nu) > 0.02*np.abs(nu).max()
    if m.sum() < 5: continue
    print(f"  {s:6s} {np.abs(nu[m]).mean():9.4f} {np.abs(ch[m]).mean():9.4f} "
          f"{np.median(ch[m]/nu[m]):7.3f}  {np.corrcoef(nu[m], ch[m])[0,1]:6.3f}")
