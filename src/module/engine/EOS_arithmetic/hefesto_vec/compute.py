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
from .volume      import solve_volume
from .therm_props import compute_therm_props
from .aggregate   import vrh_average
from .thermal     import Ftherm_vec


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
) -> dict:
    """Compute density, Vp, Vs, S for a batch of (P, T, X) conditions."""
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
        V         = V_full,
        converged = _expand(sol['converged'], fill=True).astype(bool),
        _K    = _expand(props['K']),
        _Ks   = _expand(props['Ks']),
        _Gsh  = _expand(props['Gsh']),
        _alp  = _expand(props['alp']),
        _Cp   = _expand(props['Cp']),
        _rho  = _expand(props['rho']),
        _S    = _expand_safe(S_r),  # per-species entropy (B, S) J/mol/K
    )
