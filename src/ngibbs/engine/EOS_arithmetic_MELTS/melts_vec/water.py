"""
Haar-Gallagher-Kell (1984) equation of state for water, and the Delaney &
Helgeson (1978) >10 kbar correction -- vectorized translations of MAGMA's
`sources/water.c` (`whaar()` and `wdh78()`), verbatim in structure and
constants.

Used by:
- the liquid H2O component's standard state in every rhyolite-MELTS mode
  (`gibbs.c` calls ``whaar(1.0, t, ...)``, i.e. water at 1 bar), see
  `liquid_special.py`;
- the pure-water fluid phase ("water") of rhyolite-MELTS 1.0.2 (`gibbs.c`
  calls ``whaar(min(p, 10000), t, ...)`` and adds ``wdh78`` above 10 kbar),
  see `fluid.py`.

Units follow the source exactly: T in K, P in bar; G, H in J/mol; S, Cp in
J/(mol K); V in cm^3/mol (whaar's own convention -- gibbs.c divides by 10
to get J/bar where it needs to).
"""
from __future__ import annotations
import numpy as np

_GI = np.array([0.0,
    -.53062968529023e4,  .22744901424408e5,  .78779333020687e4,
    -.69830527374994e3,  .17863832875422e6, -.39514731563338e6,
     .33803884280753e6, -.13855050202703e6, -.25637436613260e7,
     .48212575981415e7, -.34183016969660e7,  .12223156417448e7,
     .11797433655832e8, -.21734810110373e8,  .10829952168620e8,
    -.25441998064049e7, -.31377774947767e8,  .52911910757704e8,
    -.13802577177877e8, -.25109914369001e7,  .46561826115608e8,
    -.72752773275387e8,  .41774246148294e7,  .14016358244614e8,
    -.31555231392127e8,  .47929666384584e8,  .40912664781209e7,
    -.13626369388386e8,  .69625220862664e7, -.10834900096447e8,
    -.22722827401688e7,  .38365486000660e7,  .68833257944332e5,
     .21757245522644e6, -.26627944829770e5, -.70730418082074e6,
    -.225e1,            -1.68e1,             .055e1,
    -93.0e1])
_KI = np.array([0,
    1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
    6, 6, 6, 6, 7, 7, 7, 7, 9, 9, 9, 9, 3, 3, 1, 5, 2, 2, 2, 4])
_LI = np.array([0,
    1, 2, 4, 6, 1, 2, 4, 6, 1, 2, 4, 6, 1, 2, 4, 6, 1, 2, 4, 6,
    1, 2, 4, 6, 1, 2, 4, 6, 1, 2, 4, 6, 0, 3, 3, 3, 0, 2, 0, 0])
_CI = np.array([0.0,
    .19730271018e2,      .209662681977e2,   -.483429455355,
    .605743189245e1,   22.56023885,        -9.87532442,
   -.43135538513e1,      .458155781,        -.47754901883e-1,
    .41238460633e-2,    -.27929052852e-3,    .14481695261e-4,
   -.56473658748e-6,     .16200446e-7,      -.3303822796e-9,
    .451916067368e-11,  -.370734122708e-13,  .137546068238e-15])
_RHOI = {37: 0.319, 38: 0.310, 39: 0.310, 40: 1.55}
_TTTI = {37: 640.0, 38: 640.0, 39: 641.6, 40: 270.0}
_ALPI = {37: 34.0, 38: 40.0, 39: 30.0, 40: 1050.0}
_BETI = {37: 2.0e4, 38: 2.0e4, 39: 4.0e4, 40: 25.0}
_BI  = (0.7478629, -0.3540782, 0.0, 0.007159876, 0.0, -0.003528426)
_BBI = (1.1278334, -0.5944001, -5.010996, 0.0, 0.63684256, 0.0)

_R, _GREF, _HREF = 4.6152, -54955.23970014, -34099.89230644
_T0, _RR, _ALPHA, _BETA, _GAMMA, _P0 = 647.073, 8.31441, 11.0, 133.0/3.0, 7.0/2.0, 1.01325
_DBL_EPS = np.finfo(float).eps


def _psat2(t):
    """water.c psat2() (only called for t <= 647.25)."""
    a = (0.0, -7.8889166, 2.5514255, -6.716169, 33.239495,
         -105.38479, 174.35319, -148.39348, 48.631602)
    low = np.exp(6.3573118 - 8858.843/t + 607.56335/np.power(t, 0.6))
    v = t/647.25
    w = np.abs(1.0 - v)
    wsq = np.sqrt(w)
    ff = np.zeros_like(t)
    for i in range(1, 9):
        ff = ff + a[i]*w
        w = w*wsq
    return np.where(t <= 314.0, low, 220.93*np.exp(ff/v))


def _kubik(b, c, d):
    """water.c kubik(): real roots of x^3 + b x^2 + c x + d (vectorized)."""
    pi = 3.14159263538979
    x1 = np.zeros_like(b); x2 = np.zeros_like(b); x2i = np.zeros_like(b); x3 = np.zeros_like(b)
    trivial = (c == 0.0) & (d == 0.0)
    q = (2.0*b**3/27.0 - b*c/3.0 + d)/2.0
    p = (3.0*c - b*b)/9.0
    r = np.sqrt(np.abs(p))
    r = np.where(r*q < 0.0, -r, r)
    with np.errstate(all='ignore'):
        ff = q/r**3
        br1 = p > 0.0
        phi3 = np.log(ff + np.sqrt(ff*ff + 1.0))/3.0
        x1_1 = -r*(np.exp(phi3) - np.exp(-phi3)) - b/3.0
        br2 = (~br1) & (q*q + p*p*p > 0.0)
        phi3b = np.log(ff + np.sqrt(ff*ff - 1.0))/3.0
        x1_2 = -r*(np.exp(phi3b) + np.exp(-phi3b)) - b/3.0
        br3 = (~br1) & (~br2)
        phi3c = np.arctan(np.sqrt(1.0 - ff*ff)/ff)/3.0
        x1_3 = -2.0*r*np.cos(phi3c) - b/3.0
        x2_3 = 2.0*r*np.cos(pi/3.0 - phi3c) - b/3.0
        x3_3 = 2.0*r*np.cos(pi/3.0 + phi3c) - b/3.0
    x1 = np.where(br1, x1_1, np.where(br2, x1_2, x1_3))
    x2 = np.where(br3, x2_3, 0.0)
    x3 = np.where(br3, x3_3, 0.0)
    x2i = np.where(br3, 0.0, 1.0)
    x1 = np.where(trivial, -b, x1)
    x2 = np.where(trivial, 0.0, x2); x3 = np.where(trivial, 0.0, x3); x2i = np.where(trivial, 0.0, x2i)
    return x1, x2, x2i, x3


def whaar(p, t, melts_mode: bool = True):
    """Haar EOS water properties at (p [bar], t [K]), vectorized.

    Returns dict g, h, s, cp, dcpdt, v, dvdt, dvdp, d2vdt2, d2vdtdp, d2vdp2 --
    exactly whaar()'s 11 outputs (v and its derivatives in cm^3/mol).

    melts_mode=True applies the rhyolite-MELTS (MODE__MELTS / MELTSandCO2 /
    MELTSandCO2_H2O / xMELTS) Berman-1988 reference-state shift; False the
    pMELTS one -- exactly the `calculationMode` branch at the end of whaar().
    """
    p = np.asarray(p, dtype=np.float64); t = np.asarray(t, dtype=np.float64)
    p, t = np.broadcast_arrays(p, t); p = p.astype(np.float64).copy(); t = t.astype(np.float64).copy()
    GI, KI, LI = _GI, _KI, _LI
    taui = [np.ones_like(t), t/_T0]
    for i in range(2, 7):
        taui.append(taui[i-1]*taui[1])
    bi, bbi = _BI, _BBI
    b = bi[1]*np.log(taui[1]) + bi[0] + bi[3]/taui[3] + bi[5]/taui[5]
    bb = bbi[0] + bbi[1]/taui[1] + bbi[2]/taui[2] + bbi[4]/taui[4]

    ps = np.where(t <= 647.25, _psat2(t), 220.55)

    # --- initial density guess: Redlich-Kwong fit ---
    ark = 1.279186e8 - 2.241415e4*t
    brk = 1.428062e1 + 6.092237e-4*t
    oft = ark/(p*np.sqrt(t))
    buk = -10.0*_RR*t/p
    cuk = oft - brk*brk + brk*buk
    duk = -brk*oft
    x1, x2, x2i, x3 = _kubik(buk, cuk, duk)
    vol = np.where(x2i != 0.0, x1,
                   np.where(p < ps, np.maximum(x1, np.maximum(x2, x3)), np.minimum(x1, np.minimum(x2, x3))))
    with np.errstate(divide='ignore'):
        rhn = np.where(vol <= 0.0, 1.9, (1.0/vol)*18.0152)

    # --- Newton iteration on density (per element, frozen once converged) ---
    dp = np.full_like(t, np.inf); dr = np.full_like(t, np.inf)
    rh = rhn.copy()
    for _count in range(100):
        act = (dp > 10.0*_DBL_EPS) | (dr > 10.0*_DBL_EPS)
        if not act.any():
            break
        rh_a = np.where(act, rhn, rh)
        rh_a = np.where(rh_a <= 0.0, 1.e-8, rh_a)
        rh_a = np.where(rh_a > 1.9, 1.9, rh_a)
        y = rh_a*b/4.0
        e1 = 1.0 - np.exp(-rh_a)
        ermi = [np.ones_like(t), e1]
        for i in range(2, 10):
            ermi.append(ermi[i-1]*e1)
        pr = np.zeros_like(t); dpr = np.zeros_like(t)
        for i in range(1, 37):
            term = GI[i]/taui[LI[i]]*ermi[KI[i]-1]
            pr = pr + term
            dpr = dpr + (2.0 + rh_a*(KI[i]*np.exp(-rh_a) - 1.0)/ermi[1])*term
        for i in range(37, 41):
            dl = rh_a/_RHOI[i] - 1.0
            tau = t/_TTTI[i] - 1.0
            abc = -_ALPI[i]*dl**KI[i] - _BETI[i]*tau*tau
            q10 = np.where(abc > -100.0, GI[i]*dl**LI[i]*np.exp(abc), 0.0)
            qm = LI[i]/dl - KI[i]*_ALPI[i]*dl**(KI[i]-1)
            pr = pr + q10*qm*rh_a*rh_a/_RHOI[i]
            dpr = dpr + (q10*qm*rh_a*rh_a/_RHOI[i])*(2.0/rh_a + qm/_RHOI[i]) \
                - rh_a*rh_a/(_RHOI[i]*_RHOI[i])*q10*(LI[i]/dl/dl + KI[i]*(KI[i]-1)*_ALPI[i]*dl**(KI[i]-2))
        pr = rh_a*(rh_a*np.exp(-rh_a)*pr + _R*t*((1.0 + _ALPHA*y + _BETA*y*y)/(1.0-y)**3
                                               + 4.0*y*(bb/b - _GAMMA)))
        dpr = rh_a*np.exp(-rh_a)*dpr + _R*t*((1.0 + 2.0*_ALPHA*y + 3.0*_BETA*y*y)/(1.0-y)**3
                                           + 3.0*y*(1.0 + _ALPHA*y + _BETA*y*y)/(1.0-y)**4
                                           + 2.0*4.0*y*(bb/b - _GAMMA))
        neg = dpr <= 0.0
        dpr_c = np.where(dpr < 0.01, 0.01, dpr)
        x = (p - pr)/dpr_c
        with np.errstate(invalid='ignore', divide='ignore'):
            x = np.where(np.abs(x) > 0.1, 0.1*x/np.abs(x), x)
        rhn_new = np.where(neg, rhn*np.where(p <= ps, 0.95, 1.05), rh_a + x)
        dp_new = np.abs(1.0 - pr/p)
        dr_new = np.abs(1.0 - rhn_new/rh_a)
        rh = np.where(act, rh_a, rh)
        rhn = np.where(act, rhn_new, rhn)
        dp = np.where(act, dp_new, dp)
        dr = np.where(act, dr_new, dr)
    rh = rhn

    t2, t3 = t*t, t*t*t
    dbdt = bi[1]/t - 3.0*bi[3]/(taui[3]*t) - 5.0*bi[5]/(taui[5]*t)
    d2bdt2 = -bi[1]/t2 + 4.0*3.0*bi[3]/(taui[3]*t2) + 6.0*5.0*bi[5]/(taui[5]*t2)
    d3bdt3 = 2.0*bi[1]/t3 - 5.0*4.0*3.0*bi[3]/(taui[3]*t3) - 7.0*6.0*5.0*bi[5]/(taui[5]*t3)
    dbbdt = -bbi[1]/(taui[1]*t) - 2.0*bbi[2]/(taui[2]*t) - 4.0*bbi[4]/(taui[4]*t)
    d2bbdt2 = 2.0*bbi[1]/(taui[1]*t2) + 3.0*2.0*bbi[2]/(taui[2]*t2) + 5.0*4.0*bbi[4]/(taui[4]*t2)
    d3bbdt3 = -3.0*2.0*bbi[1]/(taui[1]*t3) - 4.0*3.0*2.0*bbi[2]/(taui[2]*t3) - 6.0*5.0*4.0*bbi[4]/(taui[4]*t3)

    y = rh*b/4.0
    dydt = rh*dbdt/4.0
    dydrh = b/4.0
    d2ydt2 = rh*d2bdt2/4.0
    d2ydtdrh = dbdt/4.0
    d2ydrh2 = 0.0
    d3ydt3 = rh*d3bdt3/4.0
    d3ydt2drh = d2bdt2/4.0
    d3ydtdrh2 = 0.0
    d3ydrh3 = 0.0

    e1 = 1.0 - np.exp(-rh)
    ermi = [np.ones_like(t), e1]
    for i in range(2, 10):
        ermi.append(ermi[i-1]*e1)

    al, be, ga = _ALPHA, _BETA, _GAMMA
    omy = 1.0 - y
    SQ = lambda v: v*v
    CU = lambda v: v*v*v
    QU = lambda v: v*v*v*v
    QI = lambda v: v*v*v*v*v
    bdiff = b*dbbdt - bb*dbdt
    b2diff = b*d2bbdt2 - bb*d2bdt2

    Z = -np.log(omy) - (be-1.0)/omy + (al+be+1.0)/(2.0*omy*omy) + 4*y*(bb/b - ga) \
        - (al-be+3.0)/2.0 + np.log(rh*_R*t/_P0)
    dZdt = 1.0/t + 4.0*bdiff*y/(b*b) + 4.0*(bb/b-ga)*dydt + dydt/omy \
        + (al+be+1.0)*dydt/CU(omy) - (be-1.0)*dydt/SQ(omy)
    dZdrh = 1.0/rh + 4.0*(bb/b-ga)*dydrh + dydrh/omy + (al+be+1.0)*dydrh/CU(omy) \
        - (be-1.0)*dydrh/SQ(omy)
    d2Zdt2 = -1.0/t2 + 4.0*((b2diff*(b*b) - 2.0*bdiff*b*dbdt)*y)/QU(b) \
        + SQ(dydt/omy) + 3.0*(al+be+1.0)*SQ(dydt)/QU(omy) - 2.0*(be-1.0)*SQ(dydt)/CU(omy) \
        + 8.0*bdiff*dydt/(b*b) + 4.0*(bb/b-ga)*d2ydt2 + d2ydt2/omy \
        + (al+be+1.0)*d2ydt2/CU(omy) - (be-1.0)*d2ydt2/SQ(omy)
    d2Zdtdrh = 4.0*(bb/b-ga)*d2ydtdrh + d2ydtdrh/omy + (al+be+1.0)*d2ydtdrh/CU(omy) \
        - (be-1.0)*d2ydtdrh/SQ(omy) + 4.0*bdiff*dydrh/(b*b) + dydt*dydrh/SQ(omy) \
        + 3.0*(al+be+1.0)*dydt*dydrh/QU(omy) - 2.0*(be-1.0)*dydt*dydrh/CU(omy)
    d2Zdrh2 = -1.0/(rh*rh) + SQ(dydrh/omy) + 3.0*(al+be+1.0)*SQ(dydrh)/QU(omy) \
        - 2.0*(be-1.0)*SQ(dydrh)/CU(omy) + 4.0*(bb/b-ga)*d2ydrh2 + d2ydrh2/omy \
        + (al+be+1.0)*d2ydrh2/CU(omy) - (be-1.0)*d2ydrh2/SQ(omy)
    d3Zdt3 = 2.0/t3 + 4.0*(((-2.0*bdiff*SQ(dbdt)
                             + (b*d3bbdt3 + d2bbdt2*dbdt - dbbdt*d2bdt2 - bb*d3bdt3)*SQ(b)
                             - 2.0*bdiff*b*d2bdt2)*y
                            + (b2diff*SQ(b) - 2.0*bdiff*b*dbdt)*dydt)*QU(b)
                           - 4.0*(b2diff*SQ(b) - 2.0*bdiff*b*dbdt)*CU(b)*y*dbdt) / (QU(b)*QU(b)) \
        + 2.0*CU(dydt/omy) + 12.0*(al+be+1.0)*CU(dydt)/QI(omy) - 6.0*(be-1.0)*CU(dydt)/QU(omy) \
        + 8.0*(b2diff*SQ(b) - 2.0*bdiff*b*dbdt)*dydt/QU(b) + 12.0*bdiff*d2ydt2/SQ(b) \
        + 3.0*dydt*d2ydt2/SQ(omy) + 9.0*(al+be+1.0)*dydt*d2ydt2/QU(omy) \
        - 6.0*(be-1.0)*dydt*d2ydt2/CU(omy) + 4.0*(bb/b-ga)*d3ydt3 + d3ydt3/omy \
        + (al+be+1.0)*d3ydt3/CU(omy) - (be-1.0)*d3ydt3/SQ(omy)
    d3Zdt2drh = 4.0*((b2diff*SQ(b) - 2.0*bdiff*b*dbdt)*dydrh)/QU(b) \
        + 4.0*(bb/b-ga)*d3ydt2drh + d3ydt2drh/omy + (al+be+1.0)*d3ydt2drh/CU(omy) \
        - (be-1.0)*d3ydt2drh/SQ(omy) + 8.0*bdiff*d2ydtdrh/SQ(b) + 2.0*dydt*d2ydtdrh/SQ(omy) \
        + 6.0*(al+be+1.0)*dydt*d2ydtdrh/QU(omy) - 4.0*(be-1.0)*dydt*d2ydtdrh/CU(omy) \
        + 2.0*SQ(dydt)*dydrh/CU(omy) + 12.0*(al+be+1.0)*SQ(dydt)*dydrh/QI(omy) \
        - 6.0*(be-1.0)*SQ(dydt)*dydrh/QU(omy) + d2ydt2*dydrh/SQ(omy) \
        + 3.0*(al+be+1.0)*d2ydt2*dydrh/QU(omy) - 2.0*(be-1.0)*d2ydt2*dydrh/CU(omy)
    d3Zdtdrh2 = 2.0*dydt*SQ(dydrh)/CU(omy) + 12.0*(al+be+1.0)*dydt*SQ(dydrh)/QI(omy) \
        - 6.0*(be-1.0)*dydt*SQ(dydrh)/QU(omy) + 4.0*(bb/b-ga)*d3ydtdrh2 + d3ydtdrh2/omy \
        + (al+be+1.0)*d3ydtdrh2/CU(omy) - (be-1.0)*d3ydtdrh2/SQ(omy) \
        + 2.0*d2ydtdrh*dydrh/SQ(omy) + 6.0*(al+be+1.0)*d2ydtdrh*dydrh/QU(omy) \
        - 4.0*(be-1.0)*d2ydtdrh*dydrh/CU(omy) + 4.0*bdiff*d2ydrh2/SQ(b) + dydt*d2ydrh2/SQ(omy) \
        + 3.0*(al+be+1.0)*dydt*d2ydrh2/QU(omy) - 2.0*(be-1.0)*dydt*d2ydrh2/CU(omy)
    d3Zdrh3 = 2.0/CU(rh) + 2.0*CU(dydrh/omy) + 12.0*(al+be+1.0)*CU(dydrh)/QI(omy) \
        - 6.0*(be-1.0)*CU(dydrh)/QU(omy) + 3.0*dydrh*d2ydrh2/SQ(omy) \
        + 9.0*(al+be+1.0)*dydrh*d2ydrh2/QU(omy) - 6.0*(be-1.0)*dydrh*d2ydrh2/CU(omy) \
        + 4.0*(bb/b-ga)*d3ydrh3 + d3ydrh3/omy + (al+be+1.0)*d3ydrh3/CU(omy) \
        - (be-1.0)*d3ydrh3/SQ(omy)

    r = _R
    Ab = r*t*Z
    dAbdt = r*Z + r*t*dZdt
    dAbdrh = r*t*dZdrh
    d2Abdt2 = 2.0*r*dZdt + r*t*d2Zdt2
    d2Abdtdrh = r*dZdrh + r*t*d2Zdtdrh
    d2Abdrh2 = r*t*d2Zdrh2
    d3Abdt3 = 3.0*r*d2Zdt2 + r*t*d3Zdt3
    d3Abdt2drh = 2.0*r*d2Zdtdrh + r*t*d3Zdt2drh
    d3Abdtdrh2 = r*d2Zdrh2 + r*t*d3Zdtdrh2
    d3Abdrh3 = r*t*d3Zdrh3

    # --- residual function ---
    Ar = dArdt = dArdrh = d2Ardt2 = d2Ardtdrh = d2Ardrh2 = 0.0
    d3Ardt3 = d3Ardt2drh = d3Ardtdrh2 = d3Ardrh3 = 0.0
    er = np.exp(-rh)
    for i in range(1, 37):
        k, l, g = KI[i], LI[i], GI[i]
        tl = taui[l]
        Ar = Ar + g/k/tl*ermi[k]
        dArdt = dArdt + -l*g/k/(tl*t)*ermi[k]
        dArdrh = dArdrh + g/tl*ermi[k-1]*er
        d2Ardt2 = d2Ardt2 + (l+1.0)*l*g/k/(tl*t2)*ermi[k]
        d2Ardtdrh = d2Ardtdrh + -l*g/(tl*t)*ermi[k-1]*er
        if k > 1:
            d2Ardrh2 = d2Ardrh2 + g/tl*((k-1.0)*ermi[k-2]*er - ermi[k-1])*er
        else:
            d2Ardrh2 = d2Ardrh2 + -g/tl*er
        d3Ardt3 = d3Ardt3 + -(l+2.0)*(l+1.0)*l*g/k/(tl*t3)*ermi[k]
        d3Ardt2drh = d3Ardt2drh + (l+1.0)*l*g/(tl*t2)*ermi[k-1]*er
        if k > 1:
            d3Ardtdrh2 = d3Ardtdrh2 + -l*g/(tl*t)*((k-1.0)*ermi[k-2]*er - ermi[k-1])*er
        else:
            d3Ardtdrh2 = d3Ardtdrh2 + l*g/(tl*t)*er
        if k > 2:
            d3Ardrh3 = d3Ardrh3 + g/tl*(((k-2.0)*ermi[k-3]*er - 3.0*ermi[k-2])*(k-1.0)*er + ermi[k-1])*er
        elif k > 1:
            d3Ardrh3 = d3Ardrh3 + -g/tl*(4.0*er - 1.0)*er
        else:
            d3Ardrh3 = d3Ardrh3 + g/tl*er
    for i in range(37, 41):
        k, l, g = KI[i], LI[i], GI[i]
        rhoi, ttti, alpi, beti = _RHOI[i], _TTTI[i], _ALPI[i], _BETI[i]
        dl = rh/rhoi - 1.0
        tau = t/ttti - 1.0
        Q = -alpi*dl**k - beti*tau*tau
        dQdt = -beti*2.0*tau/ttti
        dQdrh = 0.0 if k == 0 else -alpi*k*dl**(k-1)/rhoi
        d2Qdt2 = -beti*2.0/SQ(ttti)
        d2Qdtdrh = 0.0
        d2Qdrh2 = 0.0 if k in (0, 1) else -alpi*k*(k-1.0)*dl**(k-2)/SQ(rhoi)
        d3Qdt3 = 0.0
        d3Qdt2drh = 0.0
        d3Qdtdrh2 = 0.0
        d3Qdrh3 = 0.0 if k in (0, 1, 2) else -alpi*k*(k-1.0)*(k-2.0)*dl**(k-3)/CU(rhoi)
        expQ = np.where(Q > -100.0, np.exp(Q), 0.0)
        Ar = Ar + g*dl**l*expQ
        dArdt = dArdt + g*dl**l*expQ*dQdt
        if l == 0:
            dArdrh = dArdrh + g*expQ*dQdrh
        else:
            dArdrh = dArdrh + g*l*dl**(l-1)*expQ/rhoi + g*dl**l*expQ*dQdrh
        d2Ardt2 = d2Ardt2 + g*dl**l*expQ*(SQ(dQdt) + d2Qdt2)
        if l == 0:
            d2Ardtdrh = d2Ardtdrh + g*expQ*dQdt*dQdrh + g*expQ*d2Qdtdrh
        else:
            d2Ardtdrh = d2Ardtdrh + g*l*dl**(l-1)*expQ*dQdt/rhoi + g*dl**l*expQ*(dQdt*dQdrh + d2Qdtdrh)
        if l == 0:
            d2Ardrh2 = d2Ardrh2 + g*expQ*SQ(dQdrh) + g*expQ*d2Qdrh2
        elif l == 1:
            d2Ardrh2 = d2Ardrh2 + 2.0*g*expQ*dQdrh/rhoi + g*dl*expQ*SQ(dQdrh) + g*dl*expQ*d2Qdrh2
        else:
            d2Ardrh2 = d2Ardrh2 + g*l*(l-1.0)*dl**(l-2)*expQ/SQ(rhoi) \
                + 2.0*g*l*dl**(l-1)*expQ*dQdrh/rhoi + g*dl**l*(expQ*SQ(dQdrh) + expQ*d2Qdrh2)
        d3Ardt3 = d3Ardt3 + g*dl**l*expQ*(CU(dQdt) + 3.0*dQdt*d2Qdt2 + d3Qdt3)
        if l == 0:
            d3Ardt2drh = d3Ardt2drh + g*(expQ*SQ(dQdt)*dQdrh + expQ*(d2Qdt2*dQdrh + dQdt*d2Qdtdrh)) \
                + g*(expQ*dQdt*d2Qdtdrh + expQ*d3Qdt2drh)
        else:
            d3Ardt2drh = d3Ardt2drh + g*l*dl**(l-1)*(expQ*SQ(dQdt) + expQ*d2Qdt2)/rhoi \
                + g*dl**l*(expQ*dQdt*(dQdt*dQdrh + d2Qdtdrh) + expQ*(d2Qdt2*dQdrh + dQdt*d2Qdtdrh + d3Qdt2drh))
        if l == 1:
            d3Ardtdrh2 = d3Ardtdrh2 + 2.0*g*l*(expQ*dQdt*dQdrh + expQ*d2Qdtdrh)/rhoi \
                + g*dl*(expQ*dQdt*SQ(dQdrh) + expQ*2.0*dQdrh*d2Qdtdrh + expQ*dQdt*d2Qdrh2 + expQ*d3Qdtdrh2)
        else:
            # NB: for l == 0 this evaluates pow(del, -2)/pow(del, -1) with a zero prefactor,
            # exactly as the C source does.
            d3Ardtdrh2 = d3Ardtdrh2 + g*l*(l-1.0)*dl**(l-2.0)*expQ*dQdt/SQ(rhoi) \
                + 2.0*g*l*dl**(l-1.0)*(expQ*dQdt*dQdrh + expQ*d2Qdtdrh)/rhoi \
                + g*dl**l*(expQ*dQdt*SQ(dQdrh) + expQ*2.0*dQdrh*d2Qdtdrh + expQ*dQdt*d2Qdrh2 + expQ*d3Qdtdrh2)
        if l == 0:
            d3Ardrh3 = d3Ardrh3 + g*(expQ*CU(dQdrh) + expQ*2.0*dQdrh*d2Qdrh2 + expQ*dQdrh*d2Qdrh2 + expQ*d3Qdrh3)
        elif l == 1:
            d3Ardrh3 = d3Ardrh3 + 3.0*g*(expQ*SQ(dQdrh) + expQ*d2Qdrh2)/rhoi \
                + g*dl*(expQ*CU(dQdrh) + expQ*2.0*dQdrh*d2Qdrh2 + expQ*dQdrh*d2Qdrh2 + expQ*d3Qdrh3)
        elif l == 2:
            d3Ardrh3 = d3Ardrh3 + 3.0*g*2.0*expQ*dQdrh/SQ(rhoi) \
                + 3.0*g*2.0*dl*(expQ*SQ(dQdrh) + expQ*d2Qdrh2)/rhoi \
                + g*SQ(dl)*(expQ*CU(dQdrh) + expQ*2.0*dQdrh*d2Qdrh2 + expQ*dQdrh*d2Qdrh2 + expQ*d3Qdrh3)
        else:
            d3Ardrh3 = d3Ardrh3 + g*l*(l-1.0)*(l-2.0)*dl**(l-3)*expQ/CU(rhoi) \
                + 3.0*g*l*(l-1.0)*dl**(l-2)*expQ*dQdrh/SQ(rhoi) \
                + 3.0*g*l*dl**(l-1)*(expQ*SQ(dQdrh) + expQ*d2Qdrh2)/rhoi \
                + g*dl**l*(expQ*CU(dQdrh) + expQ*2.0*dQdrh*d2Qdrh2 + expQ*dQdrh*d2Qdrh2 + expQ*d3Qdrh3)

    # --- ideal gas function ---
    ci = _CI
    tr = t/1.0e2
    dtrdt = 1.0/100.0
    Zi = 1.0 + (ci[1]/tr + ci[2])*np.log(tr)
    dZidt = (-ci[1]*dtrdt/SQ(tr))*np.log(tr) + (ci[1]/tr + ci[2])*dtrdt/tr
    d2Zidt2 = (2.0*ci[1]*SQ(dtrdt)/CU(tr))*np.log(tr) + (-ci[1]*dtrdt/SQ(tr))*dtrdt/tr \
        + (-ci[1]*dtrdt/SQ(tr))*dtrdt/tr - (ci[1]/tr + ci[2])*SQ(dtrdt/tr)
    d3Zidt3 = (-3.0*2.0*ci[1]*CU(dtrdt)/QU(tr))*np.log(tr) \
        + (2.0*ci[1]*SQ(dtrdt)/CU(tr))*dtrdt/tr + (2.0*ci[1]*SQ(dtrdt)/CU(tr))*dtrdt/tr \
        - (-ci[1]*dtrdt/SQ(tr))*SQ(dtrdt/tr) + (2.0*ci[1]*SQ(dtrdt)/CU(tr))*dtrdt/tr \
        - (-ci[1]*dtrdt/SQ(tr))*SQ(dtrdt/tr) - (-ci[1]*dtrdt/SQ(tr))*SQ(dtrdt/tr) \
        + 2.0*(ci[1]/tr + ci[2])*CU(dtrdt/tr)
    for i in range(3, 19):
        Zi = Zi + ci[i]*tr**float(i-6)
        dZidt = dZidt + (i-6.0)*ci[i]*tr**float(i-7)*dtrdt
        d2Zidt2 = d2Zidt2 + (i-6.0)*(i-7.0)*ci[i]*tr**float(i-8)*SQ(dtrdt)
        d3Zidt3 = d3Zidt3 + (i-6.0)*(i-7.0)*(i-8.0)*ci[i]*tr**float(i-9)*CU(dtrdt)
    Ai = -r*t*Zi
    dAidt = -r*Zi - r*t*dZidt
    d2Aidt2 = -2.0*r*dZidt - r*t*d2Zidt2
    d3Aidt3 = -3.0*r*d2Zidt2 - r*t*d3Zidt3

    A = Ab + Ar + Ai
    dAdt = dAbdt + dArdt + dAidt
    dAdrh = dAbdrh + dArdrh
    d2Adt2 = d2Abdt2 + d2Ardt2 + d2Aidt2
    d2Adtdrh = d2Abdtdrh + d2Ardtdrh
    d2Adrh2 = d2Abdrh2 + d2Ardrh2
    d3Adt3 = d3Abdt3 + d3Ardt3 + d3Aidt3
    d3Adt2drh = d3Abdt2drh + d3Ardt2drh
    d3Adtdrh2 = d3Abdtdrh2 + d3Ardtdrh2
    d3Adrh3 = d3Abdrh3 + d3Ardrh3

    pp = rh*rh*dAdrh
    dpdrh = 2.0*rh*dAdrh + rh*rh*d2Adrh2
    dpdt = rh*rh*d2Adtdrh
    drhdt = -dpdt/dpdrh
    d2pdt2 = rh*rh*d3Adt2drh
    d2pdtdrh = 2.0*rh*d2Adtdrh + rh*rh*d3Adtdrh2
    d2pdrh2 = 2.0*dAdrh + 4.0*rh*d2Adrh2 + rh*rh*d3Adrh3

    g = A + pp/rh
    h = A + pp/rh - t*dAdt
    s = -dAdt
    cp = -t*d2Adt2 + (t/(rh*rh))*SQ(dpdt)/dpdrh
    v = 1.0/rh
    dvdt = -(1.0/SQ(rh))*drhdt
    dvdp = -(1.0/SQ(rh))/dpdrh
    temp = (-d2pdt2 - 2.0*d2pdtdrh*drhdt - d2pdrh2*SQ(drhdt))/dpdrh
    d2vdt2 = 2.0*SQ(drhdt)/CU(rh) - temp/SQ(rh)
    temp = (-d2pdtdrh/dpdrh - d2pdrh2*drhdt/dpdrh)/dpdrh
    d2vdtdp = -2.0*dpdt/(CU(rh)*SQ(dpdrh)) - temp/SQ(rh)
    d2vdp2 = (1.0/SQ(rh))*d2pdrh2/CU(dpdrh) + (2.0/CU(rh))/SQ(dpdrh)
    temp = -d2Adt2 - t*d3Adt3 + SQ(dpdt/rh)/dpdrh \
        + t*(2.0*dpdrh*dpdt*d2pdt2 - SQ(dpdt)*d2pdtdrh)/SQ(rh*dpdrh)
    dcpdt = temp + t*d2vdt2*dpdt

    m1, m2 = 1.80152, 18.0152
    out = dict(g=g*m1, h=h*m1, s=s*m1, cp=cp*m1, dcpdt=dcpdt*m1,
               v=v*m2, dvdt=dvdt*m2, dvdp=dvdp*m2, d2vdt2=d2vdt2*m2, d2vdtdp=d2vdtdp*m2, d2vdp2=d2vdp2*m2)
    if melts_mode:
        out['g'] = out['g'] + (-285829.96 - (298.15*69.9146) - _GREF)
        out['h'] = out['h'] + (-285829.96 - _HREF)
    else:
        out['g'] = out['g'] + (-237130.00 - _GREF)
        out['h'] = out['h'] + (-285830.00 - _HREF)
    return out


_A78 = np.array([
    [-5.6130073e+04,  3.8101798e-01, -2.1167697e-06, 2.0266445e-11, -8.3225572e-17],
    [-1.5285559e+01,  1.3752390e-04, -1.5586868e-09, 6.6329577e-15,  0.0],
    [-2.6092451e-02,  3.5988857e-08, -2.7916588e-14, 0.0,            0.0],
    [ 1.7140501e-05, -1.6860893e-11,  0.0,           0.0,            0.0],
    [-6.0126987e-09,  0.0,            0.0,           0.0,            0.0]])


def _dh78_poly(p1, t, tk):
    """One pass of wdh78()'s double loop, term for term (including the source's own
    d3gdtdp2 indexing quirks, which only feed d2vdtdp)."""
    a = _A78
    z = np.zeros_like(t)
    g = z.copy(); dgdt = z.copy(); dgdp = z.copy(); d2gdt2 = z.copy(); d2gdtdp = z.copy()
    d2gdp2 = z.copy(); d3gdt3 = z.copy(); d3gdt2dp = z.copy(); d3gdtdp2 = z.copy(); d3gdp3 = z.copy()
    for j in range(5):
        for l in range(5 - j):
            c = a[j][l]
            if j == 0 and l == 0:
                g = g + c
            elif j == 0:
                g = g + c*p1**l
                dgdp = dgdp + (l*c if l == 1 else 0.0) + (l*c*p1**(l-1) if l > 1 else 0.0)
                d2gdp2 = d2gdp2 + (l*(l-1)*c if l == 2 else 0.0) + (l*(l-1)*c*p1**(l-2) if l > 2 else 0.0)
                d3gdp3 = d3gdp3 + (l*(l-1)*(l-2)*c if l == 3 else 0.0) + (l*(l-1)*(l-2)*c*p1**(l-3) if l > 3 else 0.0)
            elif l == 0:
                g = g + c*t**j
                dgdt = dgdt + (j*c if j == 1 else 0.0) + (j*c*t**(j-1) if j > 1 else 0.0)
                d2gdt2 = d2gdt2 + (j*(j-1)*c if j == 2 else 0.0) + (j*(j-1)*c*t**(j-2) if j > 2 else 0.0)
                d3gdt3 = d3gdt3 + (j*(j-1)*(j-2)*c if j == 3 else 0.0) + (j*(j-1)*(j-2)*c*t**(j-3) if j > 3 else 0.0)
            else:
                g = g + c*t**j*p1**l
                dgdp = dgdp + (l*c*t**j if l == 1 else 0.0) + (l*c*t**j*p1**(l-1) if l > 1 else 0.0)
                d2gdp2 = d2gdp2 + (l*(l-1)*c*t**j if l == 2 else 0.0) + (l*(l-1)*c*t**j*p1**(l-2) if l > 2 else 0.0)
                d3gdp3 = d3gdp3 + (l*(l-1)*(l-2)*c*t**j if l == 3 else 0.0) \
                    + (l*(l-1)*(l-2)*c*t**j*p1**(l-3) if l > 3 else 0.0)
                dgdt = dgdt + (j*c*p1**l if j == 1 else 0.0) + (j*c*t**(j-1)*p1**l if j > 1 else 0.0)
                d2gdt2 = d2gdt2 + (j*(j-1)*c*p1**l if j == 2 else 0.0) + (j*(j-1)*c*t**(j-2)*p1**l if j > 2 else 0.0)
                d3gdt3 = d3gdt3 + (j*(j-1)*(j-2)*c*p1**l if j == 3 else 0.0) \
                    + (j*(j-1)*(j-2)*c*t**(j-3)*p1**l if j > 3 else 0.0)
                d2gdtdp = d2gdtdp + (j*l*c if (j == 1 and l == 1) else 0.0) \
                    + (j*l*c*t**(j-1) if (j > 1 and l == 1) else 0.0) \
                    + (j*l*c*p1**(l-1) if (j == 1 and l > 1) else 0.0) \
                    + (j*l*c*t**(j-1)*p1**(l-1) if (j > 1 and l > 1) else 0.0)
                d3gdt2dp = d3gdt2dp + (j*(j-1)*l*c if (j == 2 and l == 1) else 0.0) \
                    + (j*(j-1)*l*c*t**(j-2) if (j > 2 and l == 1) else 0.0) \
                    + (j*(j-1)*l*c*p1**(l-1) if (j == 2 and l > 1) else 0.0) \
                    + (j*(j-1)*l*c*t**(j-2)*p1**(l-1) if (j > 2 and l > 1) else 0.0)
                d3gdtdp2 = d3gdtdp2 + (j*l*(l-1)*c if (j == 1 and l == 2) else 0.0) \
                    + (j*(j-1)*l*(l-1)*c*t**(j-1) if (j > 1 and l == 2) else 0.0) \
                    + (j*l*(l-1)*c*p1**(l-2) if (j == 1 and l > 2) else 0.0) \
                    + (j*l*(l-1)*c*t**(j-2)*p1**(l-1) if (j > 1 and l > 2) else 0.0)
    return dict(g=g, h=g + tk*dgdt, s=-dgdt, cp=-tk*d2gdt2, dcpdt=-d2gdt2 - tk*d3gdt3,
                v=dgdp, dvdt=d2gdtdp, dvdp=d2gdp2, d2vdt2=d3gdt2dp, d2vdtdp=d3gdtdp2, d2vdp2=d3gdp3)


def wdh78(p, tk):
    """Delaney & Helgeson (1978) water: returns the DELTA (p minus 10 kbar) for g, h, s,
    cp, dcpdt (J) and the TOTAL v, dvdt, dvdp, d2vdt2, d2vdtdp, d2vdp2 (cm^3/mol) --
    exactly water.c wdh78()."""
    p = np.asarray(p, dtype=np.float64); tk = np.asarray(tk, dtype=np.float64)
    p, tk = np.broadcast_arrays(p, tk)
    t = tk - 273.15
    ref = _dh78_poly(np.full_like(t, 10000.0), t, tk)
    cur = _dh78_poly(p.astype(np.float64), t, tk)
    out = {k: (cur[k] - ref[k])*4.184 for k in ('g', 'h', 's', 'cp', 'dcpdt')}
    out.update({k: cur[k]*41.84 for k in ('v', 'dvdt', 'dvdp', 'd2vdt2', 'd2vdtdp', 'd2vdp2')})
    return out


def water_phase_properties(p, t):
    """The pure-water fluid phase ("water") of the rhyolite-MELTS modes -- gibbs.c's
    `strcmp(name, "water")` branch: whaar at min(p, 10 kbar), plus the wdh78
    difference above 10 kbar (whose volume replaces whaar's). V in J/bar.
    Returns dict g, h, s, cp, dcpdt, v, dvdt, dvdp, d2vdt2, d2vdtdp, d2vdp2."""
    p = np.asarray(p, dtype=np.float64); t = np.asarray(t, dtype=np.float64)
    p, t = np.broadcast_arrays(p, t)
    w = whaar(np.minimum(p, 10000.0), t, melts_mode=True)
    hi = p > 10000.0
    if hi.any():
        d = wdh78(np.where(hi, p, 10000.0), t)
        for k in ('g', 'h', 's', 'cp', 'dcpdt'):
            w[k] = np.where(hi, w[k] + d[k], w[k])
        for k in ('v', 'dvdt', 'dvdp', 'd2vdt2', 'd2vdtdp', 'd2vdp2'):
            w[k] = np.where(hi, d[k], w[k])
    for k in ('v', 'dvdt', 'dvdp', 'd2vdt2', 'd2vdtdp', 'd2vdp2'):
        w[k] = w[k]/10.0
    return w


def compute_water_phase(T, P):
    """Per-mole properties of rhyolite-MELTS 1.0.2's pure-water fluid phase
    ("water"; see water_phase_properties) in the same dict layout as the other
    melts_vec phase functions: G, H, S, V (J/bar), Cp, dCpdT, dVdT, dVdP, K (bar),
    alpha (1/K), each (B,)."""
    T = np.atleast_1d(np.asarray(T, dtype=np.float64)); P = np.atleast_1d(np.asarray(P, dtype=np.float64))
    w = water_phase_properties(P, T)
    out = {K: w[k] for K, k in (('G', 'g'), ('H', 'h'), ('S', 's'), ('V', 'v'), ('Cp', 'cp'),
                                ('dCpdT', 'dcpdt'), ('dVdT', 'dvdt'), ('dVdP', 'dvdp'))}
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        out['K'] = np.where(out['dVdP'] != 0.0, -out['V']/out['dVdP'], np.inf)
        out['alpha'] = out['dVdT']/out['V']
    return out
