"""What survives the binary phase gate: transition LOCATION, total property
jumps, and the latent-heat temperature deficit (a state-function difference
between two well-defined endpoints).  The transition WIDTH does not."""
import numpy as np, json, sys, time
sys.path.insert(0,'/home/claude/bracket'); import bracket_lib as B
BSE={'SiO2':45.0,'Al2O3':4.45,'FeO':7.75,'Fe2O3':0.35,'MgO':37.8,'CaO':3.55,'Na2O':0.36,'Cr2O3':0.38}
zP=B.prem_depth_interpolator('/mnt/user-data/uploads/nGibbs/recipes/PrEMPressureDepth.csv')

# ---- 1. Clapeyron slope from the isothermal net: locate the rw-out surface vs T
def clapeyron(Ts=np.arange(1500.,2050.,25.)):
    Ps=[]
    for T0 in Ts:
        P=np.arange(22.6,24.6,0.001)
        (ri,bm,fp),_=B.isothermal_path(BSE,np.full(len(P),T0,dtype=np.float32),P)
        ok=(ri>1e-6)&(bm>1e-3)          # rw-out edge of the reacting field
        Ps.append(P[np.where(ok)[0][-1]] if ok.any() else np.nan)
    Ts,Ps=np.asarray(Ts),np.asarray(Ps)
    g=np.isfinite(Ps)
    # robust fit over the well-behaved segment
    sl,ic=np.polyfit(Ts[g],Ps[g],1)
    return Ts,Ps,sl,ic

Ts,Ps,slope,ic=clapeyron()
print("Clapeyron dP/dT = %.4f GPa/K = %.2f MPa/K"%(slope,slope*1000))
for t,p in zip(Ts,Ps): print(f"   T={t:.0f}  P_rw-out={p:.3f}")

# ---- 2. latent-heat deficit and location vs potential temperature
def one(Tp):
    S=B.reference_entropy(BSE,Tp)
    Pc=np.arange(18.,30.,0.02)
    T,(ri,bm,fp),_=B.isentropic_path(BSE,S,Pc,props=('rho',))
    Pin0,Pout0=B.field_edges(Pc,ri,fp)
    if Pin0 is None: return None
    P=np.arange(Pin0-0.9,Pout0+0.9,0.001)
    T,(ri,bm,fp),pr=B.isentropic_path(BSE,S,P)
    Pin,Pout=B.field_edges(P,ri,fp)
    if Pin is None: return None
    deep=(P>Pout+0.20)&(P<Pout+0.70)
    if deep.sum()<20: return None
    G=float(np.polyfit(P[deep],T[deep],1)[0])          # clean lower-mantle adiabatic gradient
    i0=np.searchsorted(P,Pin)-1; i1=min(np.searchsorted(P,Pout)+1,len(P)-1)
    dT=float(T[i0]+G*(P[i1]-P[i0])-T[i1])              # latent-heat deficit
    a=np.interp(Pin-0.05,P,pr['Vs']); b=np.interp(Pout+0.05,P,pr['Vs'])
    ra=np.interp(Pin-0.05,P,pr['rho']); rb=np.interp(Pout+0.05,P,pr['rho'])
    return dict(Tp=float(Tp),S=S,gamma=G,P_in=Pin,P_out=Pout,
                z=float(zP(0.5*(Pin+Pout))),dT_dip=dT,T_tr=float(T[i0]),
                dVs_pct=float(200*(b-a)/(a+b)),drho_pct=float(200*(rb-ra)/(ra+rb)))

rows=[r for r in (one(t) for t in np.arange(1300.,1776.,25.)) if r]
dPdz=float(1.0/(zP(24.0)-zP(23.0)))*-1                 # GPa per km (positive)
print("\nPREM dP/dz near 660: %.4f GPa/km  (1 GPa = %.1f km)"%(abs(dPdz),1/abs(dPdz)))
print("\n  Tp    z(km)  T_tr(K)  gamma  dT_dip  self-broaden(km)  dVs%  drho%")
for r in rows:
    r['self_broaden_km']=abs(slope)*r['dT_dip']/abs(dPdz)
    print(f"{r['Tp']:6.0f} {r['z']:7.1f} {r['T_tr']:8.1f} {r['gamma']:6.1f} {r['dT_dip']:7.1f}"
          f"      {r['self_broaden_km']:6.2f}       {r['dVs_pct']:5.2f} {r['drho_pct']:5.2f}")
json.dump(dict(rows=rows,clapeyron_GPa_per_K=slope,dPdz_GPa_per_km=abs(dPdz),
               clap_T=Ts.tolist(),clap_P=Ps.tolist()),
          open('/home/claude/bracket/robust.json','w'),indent=1)
