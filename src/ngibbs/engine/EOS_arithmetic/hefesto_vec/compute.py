"""
Top-level compute function: (P, T, X) -> (rho, Vp, Vs, S).

Vectorised HeFESTo EOS.  Key performance optimisations:
  - Active-species column reduction: only species present in any batch item
    are passed through the heavy volume-solver / thermal-props kernels.
    For a typical mantle assemblage with ~15 active species out of 73 total
    this gives a ~5x reduction in array size.
  - Accurate Newton-Raphson Jacobian: Ctherm is computed in the NR loop so
    the full isothermal K (not just cold BM) drives the Newton step, giving
    ~3x fewer iterations.
"""

from __future__ import annotations
import numpy as np

from .params      import HeFESToParams
from .volume      import solve_volume, solve_volume_torch
from .therm_props import compute_therm_props, compute_therm_props_torch
from .aggregate   import vrh_average, vrh_average_torch
from .thermal     import Ftherm_vec, Ftherm_torch


def compute(
    P      : np.ndarray,   # (B,) GPa
    T      : np.ndarray,   # (B,) K
    X      : np.ndarray,   # (B, S)  mole fractions
    params : HeFESToParams,
    *,
    max_iter : int   = 50,
    tol_GPa  : float = 1.0e-8,
    verbose  : bool  = False,
    npz_path : str | None = None,
    device   : str | None = None,   # e.g. 'cuda', 'cuda:0', 'cpu'; None → numpy path
) -> dict:
    """Compute density, Vp, Vs, S for a batch of (P, T, X) conditions.

    Pass ``device='cuda'`` (or any torch device string) to run the heavy
    volume-solver and property kernels on GPU.  Outputs are always numpy
    arrays regardless of device.
    """
    if device is not None and device != 'cpu':
        return _compute_torch(P, T, X, params,
                              max_iter=max_iter, tol_GPa=tol_GPa,
                              verbose=verbose, npz_path=npz_path,
                              device=device)

    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.asarray(P, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)

        B = P.shape[0]
        S = params.nspec
        assert X.shape == (B, S), f"X must be ({B}, {S}), got {X.shape}"

        if npz_path is not None:
            from .dos_tables import load_tables
            load_tables(npz_path)

        apar = params.apar   # (S, 100)

        # is_stishovite mask: species whose name starts with "st" get stishtran.
        is_stishovite = np.array(
            [n.strip().startswith('st') for n in params.snames],
            dtype=bool
        )  # (S,)

        # Active-species column reduction
        active = X > 0.0                            # (B, S)
        active_any = active.any(axis=0)             # (S,) species present in any row
        ac = np.where(active_any)[0]                # (n_ac,) column indices
        n_ac = len(ac)

        apar_r          = apar[ac, :]               # (n_ac, 100)
        active_r        = active[:, ac]             # (B, n_ac)
        is_stishovite_r = is_stishovite[ac]         # (n_ac,)

        iltyp_full = np.full(S, params.iltyp, dtype=np.int32)
        iltyp_r = iltyp_full[ac]                    # (n_ac,)

        # 1. Solve for V(P, T) per species
        sol = solve_volume(
            Pi=P, Ti=T, apar=apar_r, active=active_r,
            max_iter=max_iter, tol_GPa=tol_GPa, verbose=verbose,
        )

        V_r = sol['V']   # (B, n_ac)

        # 2. Thermodynamic properties at converged V
        Ti_r  = T[:, None] * np.ones((B, n_ac))
        Pi_r  = P[:, None] * np.ones((B, n_ac))
        To_r  = apar_r[None, :, 3] * np.ones((B, n_ac))

        props = compute_therm_props(
            V=V_r, Ti=Ti_r, To=To_r, Pi=Pi_r,
            gamma   = sol['gamma'],
            q       = sol['q'],
            qp      = sol['qp'],
            etas    = sol['etas'],
            detasdv = sol['detasdv'],
            Uth     = sol['Uth'],
            Uto     = sol['Uto'],
            Cv      = sol['Cv'],
            Cvo     = sol['Cvo'],
            apar    = apar_r,
            iltyp_arr = iltyp_r,
            ivtyp   = params.ivtyp,
            ittyp   = params.ittyp,
            is_stishovite = is_stishovite_r,
        )

        # 2b. Per-species molar entropy S [J/mol/K]
        #   Fortran Ftotsub.f: entve = (Uth - Fth)/Ti + beta*Ti + slan
        #   where beta = be*(V/Vo)^ge (electronic term), slan = Landau entropy
        Fth_r = Ftherm_vec(
            Ti=Ti_r,
            fn=apar_r[None, :, 0],   zu=apar_r[None, :, 1],
            wd1=sol["wd1"],   wd2=sol["wd2"],   wd3=sol["wd3"],
            ws1=sol["ws1"],   ws2=sol["ws2"],   ws3=sol["ws3"],
            wou=sol["wou"],   wol=sol["wol"],
            we1=sol["we1"],   we2=sol["we2"],
            we3=sol["we3"],   we4=sol["we4"],
            qe1=apar_r[None, :, 16], qe2=apar_r[None, :, 18],
            qe3=apar_r[None, :, 20], qe4=apar_r[None, :, 22],
        )   # (B, n_ac)  J/mol

        # Vibrational + electronic entropy (pre-Landau V from NR solver)
        Vo_r   = apar_r[None, :, 5]
        be_r   = apar_r[None, :, 27]
        ge_r   = apar_r[None, :, 28]
        beta_r = be_r * (V_r / Vo_r) ** ge_r            # (B, n_ac)
        Ti_safe_r = np.where(Ti_r > 0.0, Ti_r, 1.0)
        S_vib_r   = np.where(Ti_r > 0.0,
                             (sol["Uth"] - Fth_r) / Ti_safe_r,
                             0.0)                        # vibrational (J/mol/K)
        S_el_r    = beta_r * Ti_r                        # electronic (J/mol/K)
        S_r       = S_vib_r + S_el_r + props['S_landau'] # (B, n_ac)  J/mol/K

        # 3. Expand reduced outputs back to full (B, S)
        Vo_full = apar[None, :, 5]   # (1, S) reference volumes

        def _expand(arr_r, fill=0.0):
            out = np.full((B, S), fill, dtype=np.float64)
            out[:, ac] = arr_r
            return out

        def _expand_safe(arr_r, fill=0.0):
            # Expand (B, n_ac) -> (B, S) AND zero inactive (X=0) cells.
            # therm_props computes values for every (B, n_ac) cell including rows
            # where a species is inactive (X=0 but it is active in another row).
            # Those cells can yield nan (e.g. alp = gamma*Cv/(V*K) at the
            # odd V=Vo, P=0 state).  Zeroing them prevents 0*nan=nan from
            # contaminating the aggregate phase-level sums.
            out = _expand(arr_r, fill=fill)
            out[~active] = fill
            return out

        V_full = _expand(props['V_corrected'], fill=0.0)
        # Inactive species get V=Vo so VRH denominators stay finite
        V_full[~active] = np.broadcast_to(Vo_full, (B, S))[~active]

        # 4. VRH phase averaging
        wm = apar[:, 2]   # (S,) molar mass

        agg = vrh_average(
            X=X,
            V=V_full,
            K   = _expand_safe(props['K']),
            Ks  = _expand_safe(props['Ks']),
            Gsh = _expand_safe(props['Gsh']),
            alp = _expand_safe(props['alp']),
            Cp  = _expand_safe(props['Cp']),
            S   = _expand_safe(S_r),
            wm=wm,
            Ti=T,
            phase_members=params.phase_members,
            active=active,
            site_data=params.site_data if params.site_data else None,
        )

        return dict(
            rho       = agg['rho'],
            Vp        = agg['Vp'],
            Vs        = agg['Vs'],
            Vb        = agg['Vb'],
            S         = agg['S'],       # aggregate entropy (J/g/K)
            Kh        = agg['Kh'],
            Gh        = agg['Gh'],
            Kv        = agg['Kv'],
            Kr        = agg['Kr'],
            Gv        = agg['Gv'],
            Gr        = agg['Gr'],
            # Isomorphic aggregate derivative properties (fort.59 columns).
            # Feed these to metamorphic.add_metamorphic() to obtain the
            # phase-change-corrected fort.56 values.
            KTr       = agg['KTr'],
            alpagg    = agg['alpagg'],
            cpagg     = agg['cpagg'],
            volagg    = agg['volagg'],
            wmagg     = agg['wmagg'],
            V         = V_full,
            converged = _expand(sol['converged'], fill=True).astype(bool),
            _K    = _expand(props['K']),
            _Ks   = _expand(props['Ks']),
            _Gsh  = _expand(props['Gsh']),
            _alp  = _expand(props['alp']),
            _Cp   = _expand(props['Cp']),
            _rho  = _expand(props['rho']),
            _S    = _expand_safe(S_r),  # per-species entropy (B, S) J/mol/K
            # Per-phase intermediates for aggregate.apply_fast_metamorphic():
            # lets the velocity chain be redone with the order-disorder K_T
            # softening without re-running the EOS.
            phase_cache = agg['phase_cache'],
        )


# ---------------------------------------------------------------------------
# Torch dispatch implementation
# ---------------------------------------------------------------------------

def _compute_torch(P, T, X, params: HeFESToParams, *,
                   max_iter, tol_GPa, verbose, npz_path, device: str) -> dict:
    """GPU-accelerated compute path.  Inputs numpy, outputs numpy."""
    import torch
    from .dos_tables import load_tables, _ensure_loaded_torch

    dev = torch.device(device)

    if npz_path is not None:
        load_tables(npz_path)
    _ensure_loaded_torch(dev)   # pre-load DOS tables on device

    def _t(arr, dtype=torch.float64):
        return torch.as_tensor(np.asarray(arr, dtype=np.float64), dtype=dtype, device=dev)

    P_t = _t(P);  T_t = _t(T);  X_t = _t(X)
    B   = P_t.shape[0]
    S   = params.nspec
    assert X_t.shape == (B, S)

    # apar as a torch tensor (kept on device for all kernel calls)
    apar_t = _t(params.apar)    # (S, NPARP)
    apar   = params.apar        # keep numpy ref for expand helpers below

    is_stishovite_t = torch.tensor(
        [n.strip().startswith('st') for n in params.snames],
        dtype=torch.bool, device=dev
    )

    active_t    = X_t > 0.0                          # (B, S)
    active_any  = active_t.any(dim=0)                # (S,)
    ac          = torch.where(active_any)[0]          # (n_ac,) indices
    n_ac        = int(ac.shape[0])

    apar_r_t    = apar_t[ac, :]                      # (n_ac, NPARP)
    active_r_t  = active_t[:, ac]                    # (B, n_ac)
    is_st_r_t   = is_stishovite_t[ac]               # (n_ac,)

    iltyp_r_t   = torch.full((n_ac,), params.iltyp,
                              dtype=torch.long, device=dev)

    # 1. Volume solver
    sol = solve_volume_torch(
        Pi=P_t, Ti=T_t, apar=apar_r_t, active=active_r_t,
        device=dev, max_iter=max_iter, tol_GPa=tol_GPa, verbose=verbose,
    )
    V_r = sol['V']   # (B, n_ac)

    # 2. Thermodynamic properties
    Ti_r = T_t[:, None].expand(B, n_ac)
    Pi_r = P_t[:, None].expand(B, n_ac)
    To_r = apar_r_t[None, :, 3].expand(B, n_ac)

    props = compute_therm_props_torch(
        V=V_r, Ti=Ti_r, To=To_r, Pi=Pi_r,
        gamma=sol['gamma'], q=sol['q'], qp=sol['qp'],
        etas=sol['etas'], detasdv=sol['detasdv'],
        Uth=sol['Uth'], Uto=sol['Uto'],
        Cv=sol['Cv'], Cvo=sol['Cvo'],
        apar_t=apar_r_t, iltyp_arr=iltyp_r_t,
        ivtyp=params.ivtyp, ittyp=params.ittyp,
        is_stishovite=is_st_r_t,
    )

    # 2b. Per-species molar entropy
    # has_aniso was already resolved in solve_volume_torch (1 sync total).
    # Reuse it here for Ftherm so we pay 0 extra syncs.
    from .thermal import _has_aniso_from_modes
    _ha = _has_aniso_from_modes(apar_r_t[:, 10], apar_r_t[:, 11],
                                apar_r_t[:, 13], apar_r_t[:, 14])
    Fth_r = Ftherm_torch(
        Ti=Ti_r,
        fn=apar_r_t[None, :, 0],  zu=apar_r_t[None, :, 1],
        wd1=sol['wd1'], wd2=sol['wd2'], wd3=sol['wd3'],
        ws1=sol['ws1'], ws2=sol['ws2'], ws3=sol['ws3'],
        wou=sol['wou'], wol=sol['wol'],
        we1=sol['we1'], we2=sol['we2'],
        we3=sol['we3'], we4=sol['we4'],
        qe1=apar_r_t[None, :, 16], qe2=apar_r_t[None, :, 18],
        qe3=apar_r_t[None, :, 20], qe4=apar_r_t[None, :, 22],
        has_aniso=_ha,
    )

    Vo_r    = apar_r_t[None, :, 5]
    be_r    = apar_r_t[None, :, 27]
    ge_r    = apar_r_t[None, :, 28]
    beta_r  = be_r * (V_r / Vo_r) ** ge_r
    Ti_safe_r = torch.where(Ti_r > 0.0, Ti_r, torch.ones_like(Ti_r))
    S_vib_r   = torch.where(Ti_r > 0.0,
                            (sol['Uth'] - Fth_r) / Ti_safe_r,
                            torch.zeros_like(Ti_r))
    S_el_r    = beta_r * Ti_r
    S_r       = S_vib_r + S_el_r + props['S_landau']   # (B, n_ac)

    # 3. Expand reduced → full (B, S)
    Vo_full_t = apar_t[None, :, 5]   # (1, S)

    def _exp(arr_r, fill=0.0):
        out = torch.full((B, S), fill, dtype=torch.float64, device=dev)
        out[:, ac] = arr_r
        return out

    def _exp_safe(arr_r, fill=0.0):
        out = _exp(arr_r, fill=fill)
        out[~active_t] = fill
        return out

    V_full_t = _exp(props['V_corrected'], fill=0.0)
    V_full_t[~active_t] = Vo_full_t.expand(B, S)[~active_t]

    # 4. VRH averaging
    wm_t = _t(params.apar[:, 2])   # (S,)

    agg = vrh_average_torch(
        X=X_t, V=V_full_t,
        K   = _exp_safe(props['K']),
        Ks  = _exp_safe(props['Ks']),
        Gsh = _exp_safe(props['Gsh']),
        alp = _exp_safe(props['alp']),
        Cp  = _exp_safe(props['Cp']),
        S   = _exp_safe(S_r),
        wm=wm_t, Ti=T_t,
        phase_members=params.phase_members,
        active=active_t,
        site_data=params.site_data if params.site_data else None,
    )

    # 5. Convert all outputs to numpy
    def _np(t):
        return t.detach().cpu().numpy() if torch.is_tensor(t) else t

    return dict(
        rho       = _np(agg['rho']),
        Vp        = _np(agg['Vp']),
        Vs        = _np(agg['Vs']),
        Vb        = _np(agg['Vb']),
        S         = _np(agg['S']),
        Kh        = _np(agg['Kh']),
        Gh        = _np(agg['Gh']),
        Kv        = _np(agg['Kv']),
        Kr        = _np(agg['Kr']),
        Gv        = _np(agg['Gv']),
        Gr        = _np(agg['Gr']),
        # Isomorphic aggregate derivative properties (fort.59 columns) — kept in
        # step with the numpy branch so metamorphic.add_metamorphic() works on a
        # GPU result too.  phase_cache is already host-side numpy.
        KTr       = _np(agg['KTr']),
        alpagg    = _np(agg['alpagg']),
        cpagg     = _np(agg['cpagg']),
        volagg    = _np(agg['volagg']),
        wmagg     = _np(agg['wmagg']),
        phase_cache = agg['phase_cache'],
        V         = _np(V_full_t),
        converged = _np(_exp(sol['converged'].to(dtype=torch.float64), fill=1.0)).astype(bool),
        _K    = _np(_exp(props['K'])),
        _Ks   = _np(_exp(props['Ks'])),
        _Gsh  = _np(_exp(props['Gsh'])),
        _alp  = _np(_exp(props['alp'])),
        _Cp   = _np(_exp(props['Cp'])),
        _rho  = _np(_exp(props['rho'])),
        _S    = _np(_exp_safe(S_r)),
    )
