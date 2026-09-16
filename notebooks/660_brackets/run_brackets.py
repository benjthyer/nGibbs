import numpy as np, time, json, sys
sys.path.insert(0, '/home/claude/bracket')
import bracket_lib as B

PREM = '/mnt/user-data/uploads/nGibbs/recipes/PrEMPressureDepth.csv'
zP = B.prem_depth_interpolator(PREM)

# Pyrolite / Bulk Silicate Earth, McDonough & Sun (1995)
BSE = {'SiO2':45.0,'Al2O3':4.45,'FeO':7.75,'Fe2O3':0.35,
       'MgO':37.8,'CaO':3.55,'Na2O':0.36,'Cr2O3':0.38}

COARSE = np.arange(18.0, 30.0, 0.02)
DP_FINE = 0.001           # 1 MPa == 25 m of depth
HALFWIN = 0.45            # GPa around the coarse-located field

def one_Mp(Tp, comp=BSE):
    S = B.reference_entropy(comp, Tp)
    T_c, (ri_c, bm_c, fp_c), _ = B.isentropic_path(comp, S, COARSE, props=('rho',))
    P_in0, P_out0 = B.field_edges(COARSE, ri_c, fp_c)
    if P_in0 is None:
        return None
    P = np.arange(P_in0 - HALFWIN, P_out0 + HALFWIN, DP_FINE)

    # ---- Pe -> inf : the isentrope itself
    T_i, (ri_i, bm_i, fp_i), pr_i = B.isentropic_path(comp, S, P)
    P_in, P_out_i = B.field_edges(P, ri_i, fp_i)
    if P_in is None:
        return None
    gamma = B.background_gradient(P, T_i, ri_i, fp_i)

    # ---- Pe -> 0 : reaction-free adiabat, linear through the interval
    T_onset = float(np.interp(P_in, P, T_i))
    T_cond = T_onset + gamma * (P - P_in)
    T_cond = np.where(P < P_in, T_i, T_cond)          # identical above the onset
    (ri_c2, bm_c2, fp_c2), pr_c = B.isothermal_path(comp, T_cond.astype(np.float32), P)
    P_in_c, P_out_c = B.field_edges(P, ri_c2, fp_c2)

    # ---- control: same isothermal net, but along the ISENTROPIC T(P).
    # Any difference from the isentropic run is inter-network bias, not physics.
    (ri_x, bm_x, fp_x), _ = B.isothermal_path(comp, T_i.astype(np.float32), P)
    P_in_x, P_out_x = B.field_edges(P, ri_x, fp_x)

    # latent-heat deficit at the isentropic completion pressure
    dT_dip = float(np.interp(P_out_i, P, T_cond) - np.interp(P_out_i, P, T_i))

    def jump(arr):
        a = np.interp(P_in - 0.05, P, arr); b = np.interp(P_out_i + 0.05, P, arr)
        return float(b - a), float(200*(b-a)/(a+b))

    dVs, dVs_pct = jump(pr_i['Vs']); drho, drho_pct = jump(pr_i['rho'])
    z_in = float(zP(P_in))
    km = lambda p1, p0: float(zP(p1) - zP(p0))

    return dict(
        Tp=float(Tp), S=S, gamma=gamma,
        P_in=P_in, P_out_isen=P_out_i, P_out_cond=P_out_c,
        P_in_cond=P_in_c, P_in_ctrl=P_in_x, P_out_ctrl=P_out_x,
        z_in=z_in, z_out_isen=float(zP(P_out_i)), z_out_cond=float(zP(P_out_c)),
        thick_isen_km=km(P_out_i, P_in), thick_cond_km=km(P_out_c, P_in),
        thick_ctrl_km=km(P_out_x, P_in_x) if P_out_x else np.nan,
        dT_dip=dT_dip, T_onset=T_onset,
        dVs=dVs, dVs_pct=dVs_pct, drho=drho, drho_pct=drho_pct,
        n_states=int(len(P)),
    )

if __name__ == '__main__':
    Tps = np.arange(1300., 1951., 25.)
    t0 = time.time(); rows = []
    for Tp in Tps:
        try:
            r = one_Mp(Tp)
        except Exception as e:
            print(f"Tp={Tp}: FAILED {type(e).__name__}: {e}"); continue
        if r is None:
            print(f"Tp={Tp}: no rw+bm+fp field found"); continue
        rows.append(r)
        print(f"Tp={Tp:6.0f}  z_in={r['z_in']:6.1f} km  "
              f"thick: isen={r['thick_isen_km']:5.2f} cond={r['thick_cond_km']:5.2f} "
              f"ctrl={r['thick_ctrl_km']:5.2f} km  dTdip={r['dT_dip']:5.1f} K  "
              f"dVs={r['dVs_pct']:4.1f}%  gamma={r['gamma']:.1f} K/GPa")
    print(f"\n{len(rows)} temperatures in {time.time()-t0:.1f}s "
          f"({sum(r['n_states'] for r in rows)*3} emulator states)")
    json.dump(rows, open('/home/claude/bracket/brackets.json', 'w'), indent=1)
