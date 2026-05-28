"""
Optimised batched Newton-Raphson volume solver for HeFESTo EOS.

Finds V such that P(V, T) = Pi_target for all (batch, species) pairs.

Key optimisations
-----------------
1.  **Flat active-only layout**: only the ~15/73 active (X>0) cells per row
    are ever processed.  Typical reduction: 3-5x fewer array elements.

2.  **Shrinking work set**: converged cells are dropped from the working
    array after each iteration, so late-iteration cost falls rapidly.

3.  **Stuck-patience early exit**: only activated AFTER at least one cell
    has converged. Once some cells converge but the remaining work set
    stops shrinking for `stuck_patience` consecutive iterations, those
    are true BM3-divergence cases (Kop<4 minerals at extreme P) and
    further iterations are useless.

4.  **Ko-floored Jacobian**: K_eff >= Ko prevents the Newton step from
    diverging when the isothermal K accidentally goes near-zero.

5.  **arg-safe gamset**: clips sqrt(arg) to sqrt(1e-12) instead of
    producing NaN for highly-expanded minerals during transients.
"""

from __future__ import annotations
import numpy as np

from .gamset   import gamset_vec
from .thermal  import Etherm_vec, Ctherm_vec
from .pressure import pressure_vec, cold_bulk_modulus_vec
from .constants import Rgas


def solve_volume(
    Pi    : np.ndarray,   # (B,) target pressures (GPa)
    Ti    : np.ndarray,   # (B,) temperatures (K)
    apar  : np.ndarray,   # (S, NPARP)  -- already column-reduced
    active: np.ndarray,   # (B, S) bool
    *,
    max_iter       : int   = 50,
    tol_GPa        : float = 1.0e-8,
    stuck_patience : int   = 5,
    verbose        : bool  = False,
) -> dict:
    """Solve V(P,T) for all active (batch, species) pairs.

    Returns a dict with all (B, S) arrays needed by compute_therm_props:
    V, converged, gamma, q, qp, etas, detasdv, scale, wd1..wol, Uth, Uto, Cv, Cvo.
    Inactive cells are filled with sensible defaults (V=Vo, converged=True).
    """
    B, S = active.shape

    # -- 0. Flatten to active-only 1D arrays ----------------------------------
    row_idx, col_idx = np.nonzero(active)   # each (n_flat,)
    n_flat = len(row_idx)

    # Species parameters -> (n_flat,)
    Pi_f   = Pi[row_idx]
    Ti_f   = Ti[row_idx]
    To_f   = apar[col_idx,  3]
    fn_f   = apar[col_idx,  0]
    zu_f   = apar[col_idx,  1]
    Vo_f   = apar[col_idx,  5]
    Ko_f   = apar[col_idx,  6]
    Kop_f  = apar[col_idx,  7]
    Kopp_f = apar[col_idx,  8]
    wd1o_f = apar[col_idx,  9]
    wd2o_f = apar[col_idx, 10]
    wd3o_f = apar[col_idx, 11]
    ws1o_f = apar[col_idx, 12]
    ws2o_f = apar[col_idx, 13]
    ws3o_f = apar[col_idx, 14]
    we1o_f = apar[col_idx, 15];  qe1_f = apar[col_idx, 16]
    we2o_f = apar[col_idx, 17];  qe2_f = apar[col_idx, 18]
    we3o_f = apar[col_idx, 19];  qe3_f = apar[col_idx, 20]
    we4o_f = apar[col_idx, 21];  qe4_f = apar[col_idx, 22]
    wouo_f = apar[col_idx, 23]
    wolo_f = apar[col_idx, 24]
    gam0_f = apar[col_idx, 25]
    qo_f   = apar[col_idx, 26]
    be_f   = apar[col_idx, 27]
    ge_f   = apar[col_idx, 28]
    q2A2_f = apar[col_idx, 29]
    ibv_f  = apar[col_idx, 31].astype(int)
    izp_f  = apar[col_idx, 33].astype(int)
    Got_f  = apar[col_idx, 36]
    a5_f   = apar[col_idx, 42]

    # -- 1. Murnaghan initial guess -------------------------------------------
    Kop_safe_f = np.where(Kop_f == 0.0, 4.0, Kop_f)
    V_f = Vo_f * (1.0 + (Kop_safe_f / Ko_f) * Pi_f) ** (-1.0 / Kop_safe_f)
    V_f = np.clip(V_f, Vo_f / 10.0, Vo_f * 10.0)

    conv_f  = np.zeros(n_flat, dtype=bool)
    P_res_f = np.zeros(n_flat, dtype=np.float64)

    # work_ids: integer indices into flat arrays that are still unconverged
    work_ids      = np.arange(n_flat, dtype=np.intp)
    prev_nw       = n_flat
    stuck_cnt     = 0
    any_converged = False   # gate: only count stuck after first convergence

    # -- 2. Newton-Raphson loop -----------------------------------------------
    for iteration in range(max_iter):
        nw = len(work_ids)
        if nw == 0:
            break

        # Stuck-patience exit: only activated after >=1 cell has converged.
        # Before any convergence we are still in the warm-up phase (NR
        # approaching the root but not yet within tol). After first
        # convergence, if the work set stops shrinking for stuck_patience
        # consecutive iterations, those are true BM3-divergence cases and
        # further iterations waste time.
        if any_converged:
            if nw == prev_nw:
                stuck_cnt += 1
            else:
                stuck_cnt = 0
        prev_nw = nw

        if stuck_cnt >= stuck_patience:
            if verbose:
                print(f"  iter {iteration:3d}: stuck for {stuck_patience} iters "
                      f"with {nw} remaining -> exit (BM3 divergence cases)")
            break

        wid = work_ids

        # Extract working slices
        V_w   = V_f[wid]
        Vo_w  = Vo_f[wid]
        Pi_w  = Pi_f[wid]
        Ti_w  = Ti_f[wid]
        To_w  = To_f[wid]

        # Mode frequencies + Gruneisen at current V
        gs_w = gamset_vec(
            wd1o_f[wid], wd2o_f[wid], wd3o_f[wid],
            ws1o_f[wid], ws2o_f[wid], ws3o_f[wid],
            we1o_f[wid], we2o_f[wid], we3o_f[wid], we4o_f[wid],
            wouo_f[wid], wolo_f[wid],
            gam0_f[wid], qo_f[wid], Got_f[wid],
            V_w, Vo_w,
        )

        kw_w = dict(
            fn=fn_f[wid], zu=zu_f[wid],
            wd1=gs_w['wd1'], wd2=gs_w['wd2'], wd3=gs_w['wd3'],
            ws1=gs_w['ws1'], ws2=gs_w['ws2'], ws3=gs_w['ws3'],
            wou=gs_w['wou'], wol=gs_w['wol'],
            we1=gs_w['we1'], we2=gs_w['we2'],
            we3=gs_w['we3'], we4=gs_w['we4'],
            qe1=qe1_f[wid], qe2=qe2_f[wid],
            qe3=qe3_f[wid], qe4=qe4_f[wid],
        )

        Uth_w = Etherm_vec(Ti=Ti_w, **kw_w)
        Uto_w = Etherm_vec(Ti=To_w, **kw_w)

        P_res_w = pressure_vec(
            Vi=V_w, Ti=Ti_w, Pi_target=Pi_w,
            Vo=Vo_w, Ko=Ko_f[wid], Kop=Kop_f[wid], Kopp=Kopp_f[wid],
            gamma=gs_w['gamma'], q=gs_w['q'],
            fn=fn_f[wid], wd1=wd1o_f[wid],
            a5=a5_f[wid], izp=izp_f[wid], be=be_f[wid], ge=ge_f[wid],
            q2A2=q2A2_f[wid],
            To=To_w, Uth=Uth_w, Uto=Uto_w,
            wd1_Vi=gs_w['wd1'], ibv=ibv_f[wid],
        )

        # Convergence check
        new_conv_mask = np.abs(P_res_w) < tol_GPa
        P_res_f[wid]               = P_res_w
        conv_f[wid[new_conv_mask]] = True

        # Shrink work set
        not_conv_mask = ~new_conv_mask
        work_ids = wid[not_conv_mask]
        nw_new   = len(work_ids)

        if new_conv_mask.any():
            any_converged = True

        if verbose:
            mx = float(np.nanmax(np.abs(P_res_w)))
            print(f"  iter {iteration:3d}: max|P_res|={mx:.3e} GPa, "
                  f"n_converged={conv_f.sum()}/{n_flat}  "
                  f"(work {nw}->{nw_new})")

        if nw_new == 0:
            break

        # -- NR update for still-unconverged cells ----------------------------
        nc    = not_conv_mask          # boolean within wid
        V_nc  = V_w[nc]
        Vo_nc = Vo_f[work_ids]
        Ti_nc = Ti_w[nc]
        To_nc = To_w[nc]

        kw_nc = {k: (v[nc] if isinstance(v, np.ndarray) else v)
                 for k, v in kw_w.items()}

        Cv_nc  = Ctherm_vec(Ti=Ti_nc, **kw_nc)
        Cvo_nc = Ctherm_vec(Ti=To_nc, **kw_nc)

        Ko_nc  = Ko_f[work_ids]
        Kc_nc  = cold_bulk_modulus_vec(
            V_nc, Vo_nc, Ko_nc, Kop_f[work_ids], Kopp_f[work_ids],
            a5_f[work_ids], ibv=ibv_f[work_ids],
        )

        gam_nc = gs_w['gamma'][nc]
        q_nc   = gs_w['q'][nc]
        Uth_nc = Uth_w[nc]
        Uto_nc = Uto_w[nc]

        ph_nc  = 1.0e-3 * (gam_nc / V_nc) * (Uth_nc - Uto_nc)
        Kth_nc = ((gam_nc + 1.0 - q_nc) * ph_nc
                  - 1.0e-3 * (gam_nc**2 / V_nc)
                    * (Ti_nc * Cv_nc - To_nc * Cvo_nc))

        # Ko-floored K_eff to prevent step divergence
        K_eff_nc = np.maximum(Kc_nc + Kth_nc, Ko_nc)

        dV_nc = P_res_w[nc] * V_nc / K_eff_nc
        V_f[work_ids] = np.clip(V_nc + dV_nc, Vo_nc / 10.0, Vo_nc * 10.0)

    # -- 3. Final evaluation at converged V (all active cells) ----------------
    gs_f = gamset_vec(
        wd1o_f, wd2o_f, wd3o_f,
        ws1o_f, ws2o_f, ws3o_f,
        we1o_f, we2o_f, we3o_f, we4o_f,
        wouo_f, wolo_f,
        gam0_f, qo_f, Got_f,
        V_f, Vo_f,
    )
    kw_f = dict(
        fn=fn_f, zu=zu_f,
        wd1=gs_f['wd1'], wd2=gs_f['wd2'], wd3=gs_f['wd3'],
        ws1=gs_f['ws1'], ws2=gs_f['ws2'], ws3=gs_f['ws3'],
        wou=gs_f['wou'], wol=gs_f['wol'],
        we1=gs_f['we1'], we2=gs_f['we2'],
        we3=gs_f['we3'], we4=gs_f['we4'],
        qe1=qe1_f, qe2=qe2_f, qe3=qe3_f, qe4=qe4_f,
    )
    Uth_f  = Etherm_vec(Ti=Ti_f, **kw_f)
    Uto_f  = Etherm_vec(Ti=To_f, **kw_f)
    Cv_f   = Ctherm_vec(Ti=Ti_f, **kw_f)
    Cvo_f  = Ctherm_vec(Ti=To_f, **kw_f)

    # -- 4. Scatter flat results back to (B, S) arrays ------------------------
    def _scatter(arr_f, fill=0.0):
        out = np.full((B, S), fill, dtype=np.float64)
        out[row_idx, col_idx] = arr_f
        return out

    def _scatter_bool(arr_f, fill=True):
        out = np.full((B, S), fill, dtype=bool)
        out[row_idx, col_idx] = arr_f
        return out

    # V: fill inactive cells with Vo to prevent downstream division by zero
    V_out = np.broadcast_to(apar[np.newaxis, :, 5], (B, S)).copy()
    V_out[row_idx, col_idx] = V_f

    return dict(
        V         = V_out,
        converged = _scatter_bool(conv_f, fill=True),
        gamma     = _scatter(gs_f['gamma']),
        q         = _scatter(gs_f['q']),
        qp        = _scatter(gs_f['qp']),
        etas      = _scatter(gs_f['etas']),
        detasdv   = _scatter(gs_f['detasdv']),
        scale     = _scatter(gs_f['scale']),
        wd1       = _scatter(gs_f['wd1']),
        wd2       = _scatter(gs_f['wd2']),
        wd3       = _scatter(gs_f['wd3']),
        ws1       = _scatter(gs_f['ws1']),
        ws2       = _scatter(gs_f['ws2']),
        ws3       = _scatter(gs_f['ws3']),
        we1       = _scatter(gs_f['we1']),
        we2       = _scatter(gs_f['we2']),
        we3       = _scatter(gs_f['we3']),
        we4       = _scatter(gs_f['we4']),
        wou       = _scatter(gs_f['wou']),
        wol       = _scatter(gs_f['wol']),
        Uth       = _scatter(Uth_f),
        Uto       = _scatter(Uto_f),
        Cv        = _scatter(Cv_f),
        Cvo       = _scatter(Cvo_f),
    )
