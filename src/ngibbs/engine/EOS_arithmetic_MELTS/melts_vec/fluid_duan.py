"""
Duan & Zhang (2006) H2O-CO2 fluid equation of state as used by rhyolite-MELTS
1.1/1.2 (MAGMA `sources/fluidPhase.c`): pure-endmember properties
(`propertiesOfPureH2O` / `propertiesOfPureCO2`) and the binary fluid's
mixing properties (`gmixFlu` / `hmixFlu` / `smixFlu` / `cpmixFlu` / `vmixFlu`).

What is verbatim and what is not
--------------------------------
Verbatim from the source: every coefficient; the virial mixing rules (B..F,
beta, gamma, including the cube-root `powSum` combining rule and the k1/k2/k3
binary factors); the Z(V) equation; the closed-form ln(phi_i) expression; the
ideal-gas Cp/H/S polynomials; the low-/high-pressure coefficient switch at
2000 bar and its splice (above 2000 bar ln(phi) = ln phi_high(P) +
ln phi_low(2000) - ln phi_high(2000)); the G/H/S/Cp/V assembly; and the
source's volume root-finder (damped fixed-point iteration from the ideal-gas
volume, then bisection), so the same root is selected.

Not transcribed: the ~1500 lines of hand-written T- and V-derivatives. The
T/P derivatives of ln(phi) and V are instead taken by automatic
differentiation (torch, float64) through the verbatim expressions, with the
converged volume re-attached to the graph by three in-graph Newton steps
(which makes first, second and third derivatives of V(T, P) exact at the
root). These are the exact derivatives of the same functions; they agree with
the source's analytic derivatives to floating-point noise (checked against
the MAGMA C oracle), except `dcpdt`, which the source itself obtains by a
forward finite difference (step sqrt(DBL_EPSILON)) -- ours is the exact
derivative, so it differs from MELTS's at the ~1e-7 relative level.

Units follow the source: T in K, P in bar, V in J/bar, energies in J/mol.
"""
from __future__ import annotations
import numpy as np
import torch

_R = 8.3143          # fluidPhase.c #define R (energies)
_RV = 8.314467       # constant used in the volume / Vc expressions
_DBL_EPS = float(np.finfo(float).eps)

H2O, CO2 = 0, 1
_TC = (647.25, 304.1282)
_PC = (221.19, 73.773)
_VC = (_RV*_TC[0]/_PC[0], _RV*_TC[1]/_PC[1])

# a1..a12, a (F), b (beta), c (gamma); [low-P, high-P][H2O, CO2]
_A = {
    True: ({1: 4.38269941E-02, 2: -1.68244362E-01, 3: -2.36923373E-01, 4: 1.13027462E-02,
            5: -7.67764181E-02, 6: 9.71820593E-02, 7: 6.62674916E-05, 8: 1.06637349E-03,
            9: -1.23265258E-03, 10: -8.93953948E-06, 11: -3.88124606E-05, 12: 5.61510206E-05,
            'a': 7.51274488E-03, 'b': 2.51598931E+00, 'c': 3.94000000E-02},
           {1: 1.14400435E-01, 2: -9.38526684E-01, 3: 7.21857006E-01, 4: 8.81072902E-03,
            5: 6.36473911E-02, 6: -7.70822213E-02, 7: 9.01506064E-04, 8: -6.81834166E-03,
            9: 7.32364258E-03, 10: -1.10288237E-04, 11: 1.26524193E-03, 12: -1.49730823E-03,
            'a': 7.81940730E-03, 'b': -4.22918013E+00, 'c': 1.58500000E-01}),
    False: ({1: 4.68071541E-02, 2: -2.81275941E-01, 3: -2.43926365E-01, 4: 1.10016958E-02,
             5: -3.86603525E-02, 6: 9.30095461E-02, 7: -1.15747171E-05, 8: 4.19873848E-04,
             9: -5.82739501E-04, 10: 1.00936000E-06, 11: -1.01713593E-05, 12: 1.63934213E-05,
             'a': -4.49505919E-02, 'b': -3.15028174E-01, 'c': 1.25000000E-02},
            {1: 5.72573440E-03, 2: 7.94836769E+00, 3: -3.84236281E+01, 4: 3.71600369E-02,
             5: -1.92888994E+00, 6: 6.64254770E+00, 7: -7.02203950E-06, 8: 1.77093234E-02,
             9: -4.81892026E-02, 10: 3.88344869E-06, 11: -5.54833167E-04, 12: 1.70489748E-03,
             'a': -4.13039220E-01, 'b': -8.47988634E+00, 'c': 2.80000000E-02}),
}

_IDEAL = np.array([
    [3.10409601236035e+01, -0.18188731e+01], [-3.91422080460869e+01, 0.12903022e+02],
    [3.79695277233575e+01, -0.96634864e+01], [-2.18374910952284e+01, 0.42251879e+01],
    [7.42251494566339e+00, -0.10421640e+01], [-1.38178929609470e+00, 0.12683515e+00],
    [1.08807067571454e-01, -0.49939675e-02], [-1.20771176848589e+01, 0.24950242e+01],
    [3.39105078851732e+00, -0.82723750e+00], [-5.84520979955060e-01, 0.15372481e+00],
    [5.89930846488082e-02, -0.15861243e-01], [-3.12970001415882e-03, 0.86017150e-03],
    [6.57460740981757e-05, -0.19222165e-04]])
_IDEAL_SHIFT = {H2O: (-355665.4136, 359.6505), CO2: (-385358.2260, 210.0304)}


def _ideal_gas(t: np.ndarray, i: int):
    """idealGasH2O / idealGasCO2: (cp, s0, h0, dcpdt), J/mol units."""
    c = _IDEAL[:, i]
    tt = t/1000.0
    cp = sum(c[k]*tt**k for k in range(7)) + sum(c[k]/tt**(k-6) for k in range(7, 13))
    dcpdt = sum(k*c[k]*tt**(k-1) for k in range(1, 7)) + sum(-(k-6)*c[k]/tt**(k+1-6) for k in range(7, 13))
    h0 = sum(c[k]*tt**(k+1)/(k+1) for k in range(7)) + c[7]*np.log(tt) \
        + sum(c[k]/tt**(k-7)/(7-k) for k in range(8, 13))
    s0 = c[0]*np.log(tt) + sum(c[k]*tt**k/k for k in range(1, 7)) + sum(c[k]/tt**(k-6)/(6-k) for k in range(7, 13))
    R = 8.31451
    h, s = _IDEAL_SHIFT[i]
    return cp*R, s0*R + s, h0*R*1000.0 + h, dcpdt*R/1000.0


def _cbrt(a):
    return torch.sign(a)*torch.abs(a).pow(1.0/3.0)


def _powsum(a, fa, b, fb):
    """fluidPhase.c powSum(): sign-preserving cube-root weighted mean, cubed."""
    return ((fa*_cbrt(a) + fb*_cbrt(b))/(fa + fb))**3


def _vc_coeffs(t, low: bool):
    """End-member virial parameters (bEnd..fEnd, beta, gamma) and binary k1/k2/k3 at T."""
    A = _A[low]
    tr = [t/_TC[H2O], t/_TC[CO2]]
    end = {}
    for key, (i1, i2, i3) in {'b': (1, 2, 3), 'c': (4, 5, 6), 'd': (7, 8, 9), 'e': (10, 11, 12)}.items():
        end[key] = [A[j][i1] + A[j][i2]/tr[j]/tr[j] + A[j][i3]/tr[j]/tr[j]/tr[j] for j in (H2O, CO2)]
    end['f'] = [A[j]['a']/tr[j]/tr[j]/tr[j] for j in (H2O, CO2)]
    end['beta'] = [A[j]['b'] for j in (H2O, CO2)]
    end['gamma'] = [A[j]['c'] for j in (H2O, CO2)]
    if low:
        k1 = 3.131 - 5.0624e-3*t + 1.8641e-6*t*t - 31.409/t
        k2 = -46.646 + 4.2877e-2*t - 1.0892e-5*t*t + 1.5782e4/t
        k3 = 0.9
    else:
        k1 = 9.034 - 7.9212e-3*t + 2.3285e-6*t*t - 2.4221e3/t
        k2 = -1.068 + 1.8756e-3*t - 4.9371e-7*t*t + 6.6180e2/t
        k3 = 1.0
    return end, k1, k2, k3


def _mix(t, x, low: bool):
    """Mixed virial coefficients (bv, cv, dv, ev, fv, beta, gammav) and their
    composition 'Prime' derivatives (d/dx_i with x_i independent), exactly the
    general-composition branch of B/C/D/E/F/Beta/GammaVcAndDerivative."""
    end, k1, k2, k3 = _vc_coeffs(t, low)
    V = _VC
    xh, xc = x[..., H2O], x[..., CO2]
    tv = lambda v: torch.as_tensor(v, dtype=torch.float64)
    # B: quadratic, k1 binary factor
    bH = end['b'][H2O]*V[H2O]; bC = end['b'][CO2]*V[CO2]
    bX = _powsum(end['b'][H2O], 1.0, end['b'][CO2], 1.0)*k1*float(_powsum(tv(V[H2O]), 1.0, tv(V[CO2]), 1.0))
    bv = bH*xh*xh + 2.0*bX*xh*xc + bC*xc*xc
    bP = torch.stack([2.0*bH*xh + 2.0*bX*xc, 2.0*bX*xh + 2.0*bC*xc], -1)
    # C: cubic, k2
    cH = end['c'][H2O]*V[H2O]**2; cC = end['c'][CO2]*V[CO2]**2
    c21 = 3.0*_powsum(end['c'][H2O], 2.0, end['c'][CO2], 1.0)*k2*float(_powsum(tv(V[H2O]), 2.0, tv(V[CO2]), 1.0))**2
    c12 = 3.0*_powsum(end['c'][H2O], 1.0, end['c'][CO2], 2.0)*k2*float(_powsum(tv(V[H2O]), 1.0, tv(V[CO2]), 2.0))**2
    cv = cH*xh**3 + c21*xh*xh*xc + c12*xh*xc*xc + cC*xc**3
    cP = torch.stack([3.0*cH*xh*xh + 2.0*c21*xh*xc + c12*xc*xc,
                      c21*xh*xh + 2.0*c12*xh*xc + 3.0*cC*xc*xc], -1)
    # D: quintic
    def w(e, n, m, p):
        return _powsum(e[H2O], float(n), e[CO2], float(m))*float(_powsum(tv(V[H2O]), float(n), tv(V[CO2]), float(m)))**p
    dE = end['d']
    dH = dE[H2O]*V[H2O]**4; dC = dE[CO2]*V[CO2]**4
    d41 = 5.0*w(dE, 4, 1, 4); d32 = 10.0*w(dE, 3, 2, 4); d23 = 10.0*w(dE, 2, 3, 4); d14 = 5.0*w(dE, 1, 4, 4)
    dv = dH*xh**5 + d41*xh**4*xc + d32*xh**3*xc**2 + d23*xh**2*xc**3 + d14*xh*xc**4 + dC*xc**5
    dP = torch.stack([5.0*dH*xh**4 + 4.0*d41*xh**3*xc + 3.0*d32*xh**2*xc**2 + 2.0*d23*xh*xc**3 + d14*xc**4,
                      d41*xh**4 + 2.0*d32*xh**3*xc + 3.0*d23*xh**2*xc**2 + 4.0*d14*xh*xc**3 + 5.0*dC*xc**4], -1)
    # E: sextic
    eE = end['e']
    eH = eE[H2O]*V[H2O]**5; eC = eE[CO2]*V[CO2]**5
    e51 = 6.0*w(eE, 5, 1, 5); e42 = 15.0*w(eE, 4, 2, 5); e33 = 20.0*w(eE, 3, 3, 5)
    e24 = 15.0*w(eE, 2, 4, 5); e15 = 6.0*w(eE, 1, 5, 5)
    ev = eH*xh**6 + e51*xh**5*xc + e42*xh**4*xc**2 + e33*xh**3*xc**3 + e24*xh**2*xc**4 + e15*xh*xc**5 + eC*xc**6
    eP = torch.stack([6.0*eH*xh**5 + 5.0*e51*xh**4*xc + 4.0*e42*xh**3*xc**2 + 3.0*e33*xh**2*xc**3
                      + 2.0*e24*xh*xc**4 + e15*xc**5,
                      e51*xh**5 + 2.0*e42*xh**4*xc + 3.0*e33*xh**3*xc**2 + 4.0*e24*xh**2*xc**3
                      + 5.0*e15*xh*xc**4 + 6.0*eC*xc**5], -1)
    # F: quadratic, no binary factor
    fE = end['f']
    fH = fE[H2O]*V[H2O]**2; fC = fE[CO2]*V[CO2]**2
    fX = _powsum(fE[H2O], 1.0, fE[CO2], 1.0)*float(_powsum(tv(V[H2O]), 1.0, tv(V[CO2]), 1.0))**2
    fv = fH*xh*xh + 2.0*fX*xh*xc + fC*xc*xc
    fP = torch.stack([2.0*fH*xh + 2.0*fX*xc, 2.0*fX*xh + 2.0*fC*xc], -1)
    # beta: linear
    be = end['beta']
    beta = be[H2O]*xh + be[CO2]*xc
    betaP = torch.stack([torch.full_like(xh, be[H2O]), torch.full_like(xh, be[CO2])], -1)
    # gamma: cubic, k3
    gE = end['gamma']
    gH = gE[H2O]*V[H2O]**2; gC = gE[CO2]*V[CO2]**2
    g21 = 3.0*float(_powsum(tv(gE[H2O]), 2.0, tv(gE[CO2]), 1.0))*k3*float(_powsum(tv(V[H2O]), 2.0, tv(V[CO2]), 1.0))**2
    g12 = 3.0*float(_powsum(tv(gE[H2O]), 1.0, tv(gE[CO2]), 2.0))*k3*float(_powsum(tv(V[H2O]), 1.0, tv(V[CO2]), 2.0))**2
    gv = gH*xh**3 + g21*xh*xh*xc + g12*xh*xc*xc + gC*xc**3
    gP = torch.stack([3.0*gH*xh*xh + 2.0*g21*xh*xc + g12*xc*xc,
                      g21*xh*xh + 2.0*g12*xh*xc + 3.0*gC*xc*xc], -1)
    return dict(bv=bv, cv=cv, dv=dv, ev=ev, fv=fv, beta=beta, gv=gv,
                bP=bP, cP=cP, dP=dP, eP=eP, fP=fP, betaP=betaP, gP=gP)


def _z(v, c):
    ex = torch.exp(-c['gv']/v/v)
    return 1.0 + c['bv']/v + c['cv']/v/v + c['dv']/v**4 + c['ev']/v**5 + (c['fv']/v/v)*(c['beta'] + c['gv']/v/v)*ex


def _solve_v(t, p, c):
    """The source's own root finder (duanDriver/duanH2ODriver/duanCO2Driver),
    vectorized, on detached values: damped fixed point from v = RT/P until the
    residual changes sign, then bisection."""
    cd = {k: val.detach() for k, val in c.items()}
    t = t.detach(); p = p.detach()
    v = _RV*t/p
    vprev = torch.ones_like(v); dprev = torch.ones_like(v)
    active = torch.ones_like(v, dtype=torch.bool)
    brk_sign = torch.zeros_like(active)
    delv = torch.zeros_like(v)
    for it in range(200):
        z = _z(v, cd)
        d = z*_RV*t/p - v
        stop = ((it > 1) & (d*dprev < 0.0)) | (torch.abs(d) < v*100.0*_DBL_EPS)
        newly = active & stop
        brk_sign = brk_sign | (newly & ~(torch.abs(d) < v*100.0*_DBL_EPS))
        delv = torch.where(active, d, delv)
        active = active & ~stop
        if not bool(active.any()):
            break
        vprev = torch.where(active, v, vprev)
        dprev = torch.where(active, d, dprev)
        v = torch.where(active, (z*_RV*t/p + v)/2.0, v)
    if bool(brk_sign.any()):
        dx = torch.where(delv < 0.0, vprev - v, v - vprev)
        rtb = torch.where(delv < 0.0, v, vprev)
        act = brk_sign.clone()
        for _ in range(200):
            dx = torch.where(act, dx*0.5, dx)
            vn = rtb + dx
            z = _z(vn, cd)
            d = z*_RV*t/p - vn
            rtb = torch.where(act & (d <= 0.0), vn, rtb)
            v = torch.where(act, vn, v)
            act = act & ~((torch.abs(dx) < 100.0*_DBL_EPS) | (d == 0.0))
            if not bool(act.any()):
                break
    return v


def _driver(t, p, x, low: bool):
    """ln(phi_i) (B, 2), v, z -- differentiable in t and p."""
    c = _mix(t, x, low)
    v = _solve_v(t, p, c)
    for _ in range(3):  # re-attach v(t, p) to the autograd graph: Newton steps with the
        # analytic dZ/dV (duanDriver's dzdv) -- exact derivatives of V at the root
        ex = torch.exp(-c['gv']/v/v)
        dzdv = (-c['bv']/v/v - 2.0*c['cv']/v**3 - 4.0*c['dv']/v**5 - 5.0*c['ev']/v**6
                - 2.0*(c['fv']/v**3)*(c['beta'] + c['gv']/v/v)*ex
                - 2.0*(c['fv']/v/v)*(c['gv']/v**3)*ex
                + 2.0*(c['fv']/v/v)*(c['beta'] + c['gv']/v/v)*(c['gv']/v**3)*ex)
        f = _z(v, c)*_RV*t/p - v
        v = v - f/(dzdv*_RV*t/p - 1.0)
    z = _z(v, c)
    ex = torch.exp(-c['gv']/v/v)
    gv, fv, beta = c['gv'], c['fv'], c['beta']
    out = []
    for i in (H2O, CO2):
        bP, cP, dP, eP, fP = c['bP'][..., i], c['cP'][..., i], c['dP'][..., i], c['eP'][..., i], c['fP'][..., i]
        betaP, gP = c['betaP'][..., i], c['gP'][..., i]
        ln = -torch.log(z) + bP/v + cP/2.0/v/v + dP/4.0/v**4 + eP/5.0/v**5
        ln = ln + ((fP*beta + betaP*fv)/2.0/gv)*(1.0 - ex)
        ln = ln + ((fP*gv + gP*fv - fv*beta*(gP - gv))/2.0/gv/gv)*(1.0 - (gv/v/v + 1.0)*ex)
        ln = ln + ((gP - gv)*fv/2.0/gv/gv)*(-2.0 + (gv*gv/v**4 + 2.0*gv/v/v + 2.0)*ex)
        out.append(ln)
    return torch.stack(out, -1), v, z


def _lnphi(t, p, x):
    """duan(): low-P coefficients for p <= 2000 bar; above, the source's splice
    ln phi_high(T, P) + ln phi_low(T, 2000) - ln phi_high(T, 2000). v is always the
    EOS volume at (T, P) of whichever coefficient set applies."""
    low = p <= 2000.0
    res_ln = torch.zeros(t.shape + (2,), dtype=torch.float64)
    res_v = torch.zeros_like(t)
    if bool(low.any()):
        idx = low.nonzero(as_tuple=True)[0]
        ln, v, _ = _driver(t[idx], p[idx], x[idx], True)
        res_ln = res_ln.index_put((idx,), ln)
        res_v = res_v.index_put((idx,), v)
    hi = ~low
    if bool(hi.any()):
        idx = hi.nonzero(as_tuple=True)[0]
        th, ph, xh = t[idx], p[idx], x[idx]
        ln, v, _ = _driver(th, ph, xh, False)
        p2k = torch.full_like(th, 2000.0)
        lnL, _, _ = _driver(th, p2k, xh, True)
        lnH, _, _ = _driver(th, p2k, xh, False)
        res_ln = res_ln.index_put((idx,), ln + lnL - lnH)
        res_v = res_v.index_put((idx,), v)
    return res_ln, res_v


def _grad(y, x, create_graph=True):
    return torch.autograd.grad(y.sum(), x, create_graph=create_graph, retain_graph=True, allow_unused=True)[0]


def _state(T, P, x, need_third=False):
    """ln(phi), V and their T/P derivatives needed downstream."""
    t = torch.as_tensor(T, dtype=torch.float64).clone().requires_grad_(True)
    p = torch.as_tensor(P, dtype=torch.float64).clone().requires_grad_(True)
    xx = torch.as_tensor(x, dtype=torch.float64)
    ln, v = _lnphi(t, p, xx)
    d = {'ln': ln.detach().numpy(), 'v': v.detach().numpy()}
    dlnT, d2lnT, d3lnT, dlnP = [], [], [], []
    for i in (H2O, CO2):
        g1 = _grad(ln[..., i], t)
        g2 = _grad(g1, t, create_graph=need_third)
        dlnT.append(g1.detach().numpy()); d2lnT.append(g2.detach().numpy())
        if need_third:
            d3lnT.append(_grad(g2, t, create_graph=False).detach().numpy())
        dlnP.append(_grad(ln[..., i], p, create_graph=False).detach().numpy())
    dvdt = _grad(v, t)
    dvdp = _grad(v, p)
    d['dlnT'] = np.stack(dlnT, -1); d['d2lnT'] = np.stack(d2lnT, -1); d['dlnP'] = np.stack(dlnP, -1)
    if need_third:
        d['d3lnT'] = np.stack(d3lnT, -1)
    d['dvdt'] = dvdt.detach().numpy(); d['dvdp'] = dvdp.detach().numpy()
    d['d2vdt2'] = _grad(dvdt, t, create_graph=False).detach().numpy()
    d['d2vdtdp'] = _grad(dvdt, p, create_graph=False).detach().numpy()
    d['d2vdp2'] = _grad(dvdp, p, create_graph=False).detach().numpy()
    return d


def _pure_state(T, P, i):
    x = np.zeros(T.shape + (2,)); x[..., i] = 1.0
    return _state(T, P, x, need_third=True)


def pure_properties(T, P, which: str, st=None):
    """propertiesOfPureH2O / propertiesOfPureCO2 (which='H2O'|'CO2'): dict g, h, s, cp,
    dcpdt, v, dvdt, dvdp, d2vdt2, d2vdtdp, d2vdp2 (J, J/K, J/bar). `st` is an
    optional precomputed _pure_state for the same (T, P) (reused by compute_fluid)."""
    T = np.atleast_1d(np.asarray(T, dtype=np.float64)); P = np.atleast_1d(np.asarray(P, dtype=np.float64))
    T, P = np.broadcast_arrays(T, P)
    if which not in ('H2O', 'CO2'):
        raise ValueError(f"which must be 'H2O' or 'CO2', got {which!r}")
    i = H2O if which == 'H2O' else CO2
    if st is None:
        st = _pure_state(T, P, i)
    cp, s, h, dcpdt = _ideal_gas(T, i)
    lnphi, d1, d2, d3 = st['ln'][..., i], st['dlnT'][..., i], st['d2lnT'][..., i], st['d3lnT'][..., i]
    lnfp = lnphi + np.log(P)                      # log(phi*p)
    R = _R
    g = h - T*s + R*T*lnfp
    s_out = s - (R*lnfp + R*T*d1)
    h_out = h + R*T*lnfp - T*(R*lnfp + R*T*d1)
    cp_out = cp - T*(2.0*R*d1 + R*T*d2)
    dcpdt_out = dcpdt - (2.0*R*d1 + R*T*d2) - T*(3.0*R*d2 + R*T*d3)
    return dict(g=g, h=h_out, s=s_out, cp=cp_out, dcpdt=dcpdt_out, v=st['v'], dvdt=st['dvdt'],
                dvdp=st['dvdp'], d2vdt2=st['d2vdt2'], d2vdtdp=st['d2vdtdp'], d2vdp2=st['d2vdp2'])


def fluid_mixing(T, P, x_co2, pure_states=None):
    """gmixFlu/hmixFlu/smixFlu/cpmixFlu/vmixFlu (FIRST, plus vmix FOURTH/FIFTH) per mole of
    fluid, x_co2 = r[0] = CO2 mole fraction. Zero (as in the source) when either
    component's mole fraction is below 100*DBL_EPSILON. Also returns 'dcpdt', the
    exact T-derivative of cpmix (cpmixFlu's own SECOND output is a finite
    difference and is not used). `pure_states` is an optional (H2O, CO2) pair of
    precomputed _pure_state results for the same (T, P) rows."""
    T = np.atleast_1d(np.asarray(T, dtype=np.float64)); P = np.atleast_1d(np.asarray(P, dtype=np.float64))
    xc = np.atleast_1d(np.asarray(x_co2, dtype=np.float64))
    T, P, xc = np.broadcast_arrays(T, P, xc)
    B = T.shape[0]
    xs = np.stack([1.0 - xc, xc], -1)
    out = {k: np.zeros(B) for k in ('g', 'h', 's', 'cp', 'dcpdt', 'v', 'dvdt', 'dvdp')}
    ok = (np.abs(xc) >= 100.0*_DBL_EPS) & (np.abs(1.0 - xc) >= 100.0*_DBL_EPS)
    if not ok.any():
        return out
    Tm, Pm, xm = T[ok], P[ok], xs[ok]
    mix = _state(Tm, Pm, xm, need_third=True)
    if pure_states is None:
        ph, pc = _pure_state(Tm, Pm, H2O), _pure_state(Tm, Pm, CO2)
    else:
        ph, pc = ({k: val[ok] for k, val in st.items()} for st in pure_states)
    R = _R
    lnrel = mix['ln'] - np.stack([ph['ln'][..., H2O], pc['ln'][..., CO2]], -1)
    dTrel = mix['dlnT'] - np.stack([ph['dlnT'][..., H2O], pc['dlnT'][..., CO2]], -1)
    d2Trel = mix['d2lnT'] - np.stack([ph['d2lnT'][..., H2O], pc['d2lnT'][..., CO2]], -1)
    d3Trel = mix['d3lnT'] - np.stack([ph['d3lnT'][..., H2O], pc['d3lnT'][..., CO2]], -1)
    xlogx = xm*np.log(xm)
    g = R*Tm*np.sum(xlogx + xm*lnrel, -1)
    s = -(R*np.sum(xlogx + xm*lnrel, -1) + R*Tm*np.sum(xm*dTrel, -1))
    h = g + Tm*s
    d2gdt2 = 2.0*R*np.sum(xm*dTrel, -1) + R*Tm*np.sum(xm*d2Trel, -1)
    d3gdt3 = 3.0*R*np.sum(xm*d2Trel, -1) + R*Tm*np.sum(xm*d3Trel, -1)
    out['g'][ok] = g; out['s'][ok] = s; out['h'][ok] = h; out['cp'][ok] = -Tm*d2gdt2
    out['dcpdt'][ok] = -d2gdt2 - Tm*d3gdt3
    out['v'][ok] = mix['v'] - xm[:, 0]*ph['v'] - xm[:, 1]*pc['v']
    out['dvdt'][ok] = mix['dvdt'] - xm[:, 0]*ph['dvdt'] - xm[:, 1]*pc['dvdt']
    out['dvdp'][ok] = mix['dvdp'] - xm[:, 0]*ph['dvdp'] - xm[:, 1]*pc['dvdp']
    return out


def compute_fluid(T, P, n_h2o, n_co2):
    """Extensive properties of an H2O-CO2 fluid (moles n_h2o, n_co2) exactly as
    alphaMELTS reports them (read_write.c: moles*mix + sum_i n_i * endmember):
    dict G, H, S, V (J/bar), Cp, dCpdT, dVdT, dVdP."""
    T = np.atleast_1d(np.asarray(T, dtype=np.float64)); P = np.atleast_1d(np.asarray(P, dtype=np.float64))
    nh = np.atleast_1d(np.asarray(n_h2o, dtype=np.float64)); nc = np.atleast_1d(np.asarray(n_co2, dtype=np.float64))
    T, P, nh, nc = np.broadcast_arrays(T, P, nh, nc)
    n = nh + nc
    with np.errstate(invalid='ignore', divide='ignore'):
        xc = np.where(n > 0, nc/n, 0.0)
    keys = (('G', 'g'), ('H', 'h'), ('S', 's'), ('V', 'v'), ('Cp', 'cp'), ('dCpdT', 'dcpdt'),
            ('dVdT', 'dvdt'), ('dVdP', 'dvdp'))
    res = {K: np.zeros(T.shape) for K, _ in keys}
    # Endmember states only where that endmember is present (the Duan volume
    # solve dominates the cost); each is reused by the mixing term.
    states = {}
    for i, which, ni in ((H2O, 'H2O', nh), (CO2, 'CO2', nc)):
        m = ni > 0
        if m.any():
            st = _pure_state(T[m], P[m], i)
            e = pure_properties(T[m], P[m], which, st=st)
            for K, k in keys:
                res[K][m] += ni[m]*e[k]
            states[i] = (m, st)
    both = (nh > 0) & (nc > 0)
    if both.any():
        sub = [{k: val[both[m]] for k, val in st.items()} for m, st in (states[H2O], states[CO2])]
        mix = fluid_mixing(T[both], P[both], xc[both], pure_states=tuple(sub))
        for K, k in keys:
            res[K][both] += n[both]*mix[k]
    return res


def compute_fluid_solution(T, P, X):
    """Per-mole properties of the rhyolite-MELTS 1.1/1.2 H2O-CO2 fluid phase
    ("fluid") for (B, 2) mole fractions X = [H2O, CO2] (MELTS's own endmember
    order, h2oduan then co2duan); rows are normalized here. Returns dict of (B,):
    G, H, S, V (J/bar), Cp, dCpdT, dVdT, dVdP, K (bar), alpha (1/K)."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if X.shape[-1] != 2:
        raise ValueError(f"fluid composition must be (B, 2) [H2O, CO2], got shape {X.shape}")
    n = X.sum(-1)
    if np.any(n <= 0.0):
        raise ValueError(f"fluid composition rows must have positive total moles; bad rows: {np.nonzero(n <= 0.0)[0][:10]}")
    out = compute_fluid(T, P, X[:, 0]/n, X[:, 1]/n)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        out['K'] = np.where(out['dVdP'] != 0.0, -out['V']/out['dVdP'], np.inf)
        out['alpha'] = out['dVdT']/out['V']
    return out
