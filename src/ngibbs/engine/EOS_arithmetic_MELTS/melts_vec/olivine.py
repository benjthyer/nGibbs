"""
Olivine (tephroite-fayalite-Co-olivine-Ni-olivine-monticellite-forsterite)
solid-solution mixing model -- vectorized translation of `sources/olivine.c`
(Hirschmann 1991; Sack & Ghiorso 1989; Hirschmann & Ghiorso 1994), a
two-site (M1/M2) Bragg-Williams/asymmetric-Margules model with internal
Mn/Fe/Co/Ni M1-vs-M2 site-preference ("ordering") parameters solved by
Newton iteration at each (bulk composition, T, P) -- structurally the
most complex piece implemented so far.

Every coefficient below (G0, GR1..GR5, GS1..GS4, the ~45 quadratic
GRiRj/GRiSj/GSiSj cross terms, and their V-analogues) is a MECHANICAL,
verbatim transcription of the `#define` macros at olivine.c lines
121-637 -- not re-derived. This is deliberate: those macros are
themselves simple fixed linear combinations of ~50 base Margules/
exchange parameters (HEX**, VEX**, HX**, VX**, WH1**/WH2**, WV1**/WV2**,
WH2CA*/WV2CA*, F_*), so transcribing the combination *formulas* exactly
(rather than trying to re-derive or simplify them) is both the lowest-
risk and the most maintainable path -- and it means this file can be
diffed line-by-line against olivine.c if MELTS's own parameters are ever
updated.

See `solution_model.py`'s module docstring for why dG/dr (activities),
G, and S are evaluated analytically at the converged ordering parameters
s* (exact, by the envelope theorem) while dV/dT, dV/dP, and Cp are
obtained by finite-differencing V(T,P) and G(T,P) with s* re-solved at
each stencil point (capturing the real ordering contribution to thermal
expansion/heat capacity, which a naive fixed-s* analytic formula would
miss).
"""
from __future__ import annotations
import numpy as np

from .constants import Rgas
from .solution_model import newton_solve_ordering, darken_activities

ENDMEMBERS = ["tephroite", "fayalite", "co-olivine", "ni-olivine", "monticellite", "forsterite"]

_DBL_EPS = 1e-12

# ---------------------------------------------------------------------
# Base Margules/exchange parameters, verbatim from olivine.c lines 121-217
# (all in J or J/bar).
# ---------------------------------------------------------------------
_HEX = dict(MGMN=15.80e3, MGFE=0.0, MGCO=-15.00e3, MGNI=-19.75e3,
            MNFE=-11.80e3, MNCO=0.0, MNNI=0.0, FECO=0.0, FENI=-20.00e3, CONI=0.0)
_VEX = dict(MGMN=0.0, MGFE=0.0, MGCO=0.0, MGNI=0.0,
            MNFE=0.0, MNCO=0.0, MNNI=0.0, FECO=0.0, FENI=0.0, CONI=0.0)
_HX = dict(MGMN=8.75e3, MGFE=10.15e3, MGCO=3.00e3, MGNI=2.20e3,
           MNFE=0.50e3, MNCO=0.0, MNNI=0.0, FECO=3.00e3, FENI=10.00e3, CONI=0.0)
_VX = dict(MGMN=0.0, MGFE=0.015, MGCO=0.0, MGNI=0.0,
           MNFE=0.0, MNCO=0.0, MNNI=0.0, FECO=0.0, FENI=0.0, CONI=0.0)
_WH1 = dict(MGMN=6.625e3, MGFE=5.075e3, MGCO=1.50e3, MGNI=-0.600e3,
            MNFE=1.75e3, MNCO=0.0, MNNI=0.0, FECO=1.50e3, FENI=5.000e3, CONI=0.0)
_WH2 = dict(MGMN=6.625e3, MGFE=5.075e3, MGCO=1.50e3, MGNI=2.800e3,
            MNFE=1.75e3, MNCO=0.0, MNNI=0.0, FECO=1.50e3, FENI=5.000e3, CONI=0.0)
_WV1 = dict(MGMN=0.0, MGFE=0.0, MGCO=0.0, MGNI=0.0,
            MNFE=0.0, MNCO=0.0, MNNI=0.0, FECO=0.0, FENI=0.0, CONI=0.0)
_WV2 = dict(MGMN=0.0, MGFE=0.0, MGCO=0.0, MGNI=0.0,
            MNFE=0.0, MNCO=0.0, MNNI=0.0, FECO=0.0, FENI=0.0, CONI=0.0)
_WH2CA = dict(MG=34.50e3, MN=16.00e3, FE=21.90e3, CO=30.00e3, NI=40.00e3)
_WV2CA = dict(MG=0.35, MN=0.0, FE=0.0, CO=0.0, NI=0.0)
_F = dict(MN=9.50e3, FE=9.50e3, CO=0.0, NI=0.0)


def _combo(P, hex_dict, vex_dict, key):
    """GEXxx / GXxx style: HEXxx + (P-1)*VEXxx, broadcasting against P."""
    return hex_dict[key] + (P - 1.0)*vex_dict[key]


def _coeffs(P):
    """All G-version (P-dependent) combined parameters, verbatim from
    olivine.c lines 224-271: GEX**, GX**, W1**, W2**, W2CA**."""
    GEX = {k: _combo(P, _HEX, _VEX, k) for k in _HEX}
    GX  = {k: _combo(P, _HX,  _VX,  k) for k in _HX}
    W1  = {k: _combo(P, _WH1, _WV1, k) for k in _WH1}
    W2  = {k: _combo(P, _WH2, _WV2, k) for k in _WH2}
    W2CA = {k: _combo(P, {k2: _WH2CA[k2] for k2 in _WH2CA},
                          {k2: _WV2CA[k2] for k2 in _WV2CA}, k)
             for k in _WH2CA}
    return GEX, GX, W1, W2, W2CA


def _g_taylor_coeffs(P):
    """G0, GR1..GR5, GS1..GS4, and the quadratic GRiRj/GRiSj/GSiSj terms,
    verbatim from olivine.c lines 278-459. Returns a dict."""
    GEX, GX, W1, W2, W2CA = _coeffs(P)
    F = _F
    c = {}

    c['G0'] = 0.25*((GX['MNFE']+W1['MNFE']+W2['MNFE'])
                     + (GX['MNCO']+W1['MNCO']+W2['MNCO'])
                     + (GX['MNNI']+W1['MNNI']+W2['MNNI'])
                     + (GX['FECO']+W1['FECO']+W2['FECO'])
                     + (GX['FENI']+W1['FENI']+W2['FENI'])
                     + (GX['CONI']+W1['CONI']+W2['CONI'])
                     - 2.0*(GX['MGMN']+W1['MGMN']+W2['MGMN'])
                     - 2.0*(GX['MGFE']+W1['MGFE']+W2['MGFE'])
                     - 2.0*(GX['MGCO']+W1['MGCO']+W2['MGCO'])
                     - 2.0*(GX['MGNI']+W1['MGNI']+W2['MGNI']))

    c['GR1'] = 0.25*(-3.0*(GX['MGMN']+W1['MGMN']+W2['MGMN'])
                      - (GX['MGFE']+W1['MGFE']+W2['MGFE'])
                      - (GX['MGCO']+W1['MGCO']+W2['MGCO'])
                      - (GX['MGNI']+W1['MGNI']+W2['MGNI'])
                      + (GX['MNFE']+W1['MNFE']+W2['MNFE'])
                      + (GX['MNCO']+W1['MNCO']+W2['MNCO'])
                      + (GX['MNNI']+W1['MNNI']+W2['MNNI']))
    c['GR2'] = 0.25*(-(GX['MGMN']+W1['MGMN']+W2['MGMN'])
                      - 3.0*(GX['MGFE']+W1['MGFE']+W2['MGFE'])
                      - (GX['MGCO']+W1['MGCO']+W2['MGCO'])
                      - (GX['MGNI']+W1['MGNI']+W2['MGNI'])
                      + (GX['MNFE']+W1['MNFE']+W2['MNFE'])
                      + (GX['FECO']+W1['FECO']+W2['FECO'])
                      + (GX['FENI']+W1['FENI']+W2['FENI']))
    c['GR3'] = 0.25*(-(GX['MGMN']+W1['MGMN']+W2['MGMN'])
                      - (GX['MGFE']+W1['MGFE']+W2['MGFE'])
                      - 3.0*(GX['MGCO']+W1['MGCO']+W2['MGCO'])
                      - (GX['MGNI']+W1['MGNI']+W2['MGNI'])
                      + (GX['MNCO']+W1['MNCO']+W2['MNCO'])
                      + (GX['FECO']+W1['FECO']+W2['FECO'])
                      + (GX['CONI']+W1['CONI']+W2['CONI']))
    c['GR4'] = 0.25*(-(GX['MGMN']+W1['MGMN']+W2['MGMN'])
                      - (GX['MGFE']+W1['MGFE']+W2['MGFE'])
                      - (GX['MGCO']+W1['MGCO']+W2['MGCO'])
                      - 3.0*(GX['MGNI']+W1['MGNI']+W2['MGNI'])
                      + (GX['MNNI']+W1['MNNI']+W2['MNNI'])
                      + (GX['FENI']+W1['FENI']+W2['FENI'])
                      + (GX['CONI']+W1['CONI']+W2['CONI']))
    c['GR5'] = (-W2CA['MG']
                - 0.25*((F['MN']+GEX['MGMN']+GX['MGMN']-2.0*W2CA['MN']+2.0*W2['MGMN'])
                        + (F['FE']+GEX['MGFE']+GX['MGFE']-2.0*W2CA['FE']+2.0*W2['MGFE'])
                        + (F['CO']+GEX['MGCO']+GX['MGCO']-2.0*W2CA['CO']+2.0*W2['MGCO'])
                        + (F['NI']+GEX['MGNI']+GX['MGNI']-2.0*W2CA['NI']+2.0*W2['MGNI'])))

    c['GS1'] = 0.25*((GEX['MGMN']+3.0*W1['MGMN']-3.0*W2['MGMN'])
                      - (GEX['MGFE']-W1['MGFE']+W2['MGFE'])
                      - (GEX['MGCO']-W1['MGCO']+W2['MGCO'])
                      - (GEX['MGNI']-W1['MGNI']+W2['MGNI'])
                      + (GEX['MNFE']-W1['MNFE']+W2['MNFE'])
                      + (GEX['MNCO']-W1['MNCO']+W2['MNCO'])
                      + (GEX['MNNI']-W1['MNNI']+W2['MNNI']))
    c['GS2'] = 0.25*((-GEX['MGMN']+W1['MGMN']-W2['MGMN'])
                      + (GEX['MGFE']+3.0*W1['MGFE']-3.0*W2['MGFE'])
                      - (GEX['MGCO']-W1['MGCO']+W2['MGCO'])
                      - (GEX['MGNI']-W1['MGNI']+W2['MGNI'])
                      - (GEX['MNFE']+W1['MNFE']-W2['MNFE'])
                      + (GEX['FECO']-W1['FECO']+W2['FECO'])
                      + (GEX['FENI']-W1['FENI']+W2['FENI']))
    c['GS3'] = 0.25*((GEX['MGMN']-W1['MGMN']+W2['MGMN'])
                      + (GEX['MGFE']-W1['MGFE']+W2['MGFE'])
                      - (GEX['MGCO']+3.0*W1['MGCO']-3.0*W2['MGCO'])
                      + (GEX['MGNI']-W1['MGNI']+W2['MGNI'])
                      + (GEX['MNCO']+W1['MNCO']-W2['MNCO'])
                      + (GEX['FECO']+W1['FECO']-W2['FECO'])
                      - (GEX['CONI']-W1['CONI']+W2['CONI']))
    c['GS4'] = 0.25*((GEX['MGMN']-W1['MGMN']+W2['MGMN'])
                      + (GEX['MGFE']-W1['MGFE']+W2['MGFE'])
                      + (GEX['MGCO']-W1['MGCO']+W2['MGCO'])
                      - (GEX['MGNI']+3.0*W1['MGNI']-3.0*W2['MGNI'])
                      + (GEX['MNNI']+W1['MNNI']-W2['MNNI'])
                      + (GEX['FENI']+W1['FENI']-W2['FENI'])
                      + (GEX['CONI']+W1['CONI']-W2['CONI']))

    c['GR1R1'] = -0.25*(GX['MGMN']+W1['MGMN']+W2['MGMN'])
    c['GR1R2'] = 0.25*((GX['MNFE']+W1['MNFE']+W2['MNFE'])
                        - (GX['MGMN']+W1['MGMN']+W2['MGMN'])
                        - (GX['MGFE']+W1['MGFE']+W2['MGFE']))
    c['GR1R3'] = 0.25*((GX['MNCO']+W1['MNCO']+W2['MNCO'])
                        - (GX['MGMN']+W1['MGMN']+W2['MGMN'])
                        - (GX['MGCO']+W1['MGCO']+W2['MGCO']))
    c['GR1R4'] = 0.25*((GX['MNNI']+W1['MNNI']+W2['MNNI'])
                        - (GX['MGMN']+W1['MGMN']+W2['MGMN'])
                        - (GX['MGNI']+W1['MGNI']+W2['MGNI']))
    c['GR1R5'] = -0.25*(F['MN']+GEX['MGMN']+GX['MGMN']
                         + 2.0*W2CA['MG']-2.0*W2CA['MN']+2.0*W2['MGMN'])
    c['GR1S1'] = 0.5*(W1['MGMN']-W2['MGMN'])
    c['GR1S2'] = 0.25*((-GEX['MGMN']-W1['MGMN']+W2['MGMN'])
                        + (GEX['MGFE']+W1['MGFE']-W2['MGFE'])
                        - (GEX['MNFE']+W1['MNFE']-W2['MNFE']))
    c['GR1S3'] = 0.25*((GEX['MGMN']-W1['MGMN']+W2['MGMN'])
                        - (GEX['MGCO']+W1['MGCO']-W2['MGCO'])
                        + (GEX['MNCO']+W1['MNCO']-W2['MNCO']))
    c['GR1S4'] = 0.25*((GEX['MGMN']-W1['MGMN']+W2['MGMN'])
                        - (GEX['MGNI']+W1['MGNI']-W2['MGNI'])
                        + (GEX['MNNI']+W1['MNNI']-W2['MNNI']))

    c['GR2R2'] = -0.25*(GX['MGFE']+W1['MGFE']+W2['MGFE'])
    c['GR2R3'] = 0.25*((GX['FECO']+W1['FECO']+W2['FECO'])
                        - (GX['MGFE']+W1['MGFE']+W2['MGFE'])
                        - (GX['MGCO']+W1['MGCO']+W2['MGCO']))
    c['GR2R4'] = 0.25*((GX['FENI']+W1['FENI']+W2['FENI'])
                        - (GX['MGFE']+W1['MGFE']+W2['MGFE'])
                        - (GX['MGNI']+W1['MGNI']+W2['MGNI']))
    c['GR2R5'] = -0.25*(F['FE']+GEX['MGFE']+GX['MGFE']
                         + 2.0*W2CA['MG']-2.0*W2CA['FE']+2.0*W2['MGFE'])
    c['GR2S1'] = 0.25*((GEX['MGMN']+W1['MGMN']-W2['MGMN'])
                        - (GEX['MGFE']-W1['MGFE']+W2['MGFE'])
                        + (GEX['MNFE']-W1['MNFE']+W2['MNFE']))
    c['GR2S2'] = 0.5*(W1['MGFE']-W2['MGFE'])
    c['GR2S3'] = 0.25*((GEX['MGFE']-W1['MGFE']+W2['MGFE'])
                        - (GEX['MGCO']+W1['MGCO']-W2['MGCO'])
                        + (GEX['FECO']+W1['FECO']-W2['FECO']))
    c['GR2S4'] = 0.25*((GEX['MGFE']-W1['MGFE']+W2['MGFE'])
                        - (GEX['MGNI']+W1['MGNI']-W2['MGNI'])
                        + (GEX['FENI']+W1['FENI']-W2['FENI']))

    c['GR3R3'] = -0.25*(GX['MGCO']+W1['MGCO']+W2['MGCO'])
    c['GR3R4'] = 0.25*((GX['CONI']+W1['CONI']+W2['CONI'])
                        - (GX['MGCO']+W1['MGCO']+W2['MGCO'])
                        - (GX['MGNI']+W1['MGNI']+W2['MGNI']))
    c['GR3R5'] = -0.25*(F['CO']+GEX['MGCO']+GX['MGCO']
                         + 2.0*W2CA['MG']-2.0*W2CA['CO']+2.0*W2['MGCO'])
    c['GR3S1'] = 0.25*((GEX['MGMN']+W1['MGMN']-W2['MGMN'])
                        - (GEX['MGCO']-W1['MGCO']+W2['MGCO'])
                        + (GEX['MNCO']-W1['MNCO']+W2['MNCO']))
    c['GR3S2'] = 0.25*((GEX['MGFE']+W1['MGFE']-W2['MGFE'])
                        - (GEX['MGCO']-W1['MGCO']+W2['MGCO'])
                        + (GEX['FECO']-W1['FECO']+W2['FECO']))
    c['GR3S3'] = 0.5*(-W1['MGCO']+W2['MGCO'])
    c['GR3S4'] = 0.25*((GEX['MGCO']+W1['MGCO']-W2['MGCO'])
                        - (GEX['MGNI']-W1['MGNI']+W2['MGNI'])
                        + (GEX['CONI']-W1['CONI']+W2['CONI']))

    c['GR4R4'] = -0.25*(GX['MGNI']+W1['MGNI']+W2['MGNI'])
    c['GR4R5'] = -0.25*(F['NI']+GEX['MGNI']+GX['MGNI']
                         + 2.0*W2CA['MG']-2.0*W2CA['NI']+2.0*W2['MGNI'])
    c['GR4S1'] = 0.25*((GEX['MGMN']+W1['MGMN']-W2['MGMN'])
                        - (GEX['MGNI']-W1['MGNI']+W2['MGNI'])
                        + (GEX['MNNI']-W1['MNNI']+W2['MNNI']))
    c['GR4S2'] = 0.25*((GEX['MGFE']+W1['MGFE']-W2['MGFE'])
                        - (GEX['MGNI']-W1['MGNI']+W2['MGNI'])
                        + (GEX['FENI']-W1['FENI']+W2['FENI']))
    c['GR4S3'] = 0.25*((-GEX['MGCO']+W1['MGCO']-W2['MGCO'])
                        + (GEX['MGNI']-W1['MGNI']+W2['MGNI'])
                        - (GEX['CONI']-W1['CONI']+W2['CONI']))
    c['GR4S4'] = 0.5*(-W1['MGNI']+W2['MGNI'])

    c['GR5R5'] = -1.0*W2CA['MG']
    c['GR5S1'] = 0.25*(F['MN']+GEX['MGMN']+GX['MGMN']
                        - 2.0*W2CA['MG']+2.0*W2CA['MN']-2.0*W2['MGMN'])
    c['GR5S2'] = 0.25*(F['FE']+GEX['MGFE']+GX['MGFE']
                        - 2.0*W2CA['MG']+2.0*W2CA['FE']-2.0*W2['MGFE'])
    c['GR5S3'] = -0.25*(F['CO']+GEX['MGCO']+GX['MGCO']
                         - 2.0*W2CA['MG']+2.0*W2CA['CO']-2.0*W2['MGCO'])
    c['GR5S4'] = -0.25*(F['NI']+GEX['MGNI']+GX['MGNI']
                         - 2.0*W2CA['MG']+2.0*W2CA['NI']-2.0*W2['MGNI'])

    c['GS1S1'] = 0.25*(GX['MGMN']-W1['MGMN']-W2['MGMN'])
    c['GS1S2'] = 0.25*((GX['MGMN']-W1['MGMN']-W2['MGMN'])
                        + (GX['MGFE']-W1['MGFE']-W2['MGFE'])
                        - (GX['MNFE']-W1['MNFE']-W2['MNFE']))
    c['GS1S3'] = 0.25*(-(GX['MGMN']-W1['MGMN']-W2['MGMN'])
                        - (GX['MGCO']-W1['MGCO']-W2['MGCO'])
                        + (GX['MNCO']-W1['MNCO']-W2['MNCO']))
    c['GS1S4'] = 0.25*(-(GX['MGMN']-W1['MGMN']-W2['MGMN'])
                        - (GX['MGNI']-W1['MGNI']-W2['MGNI'])
                        + (GX['MNNI']-W1['MNNI']-W2['MNNI']))
    c['GS2S2'] = 0.25*(GX['MGFE']-W1['MGFE']-W2['MGFE'])
    c['GS2S3'] = 0.25*(-(GX['MGFE']-W1['MGFE']-W2['MGFE'])
                        - (GX['MGCO']-W1['MGCO']-W2['MGCO'])
                        + (GX['FECO']-W1['FECO']-W2['FECO']))
    c['GS2S4'] = 0.25*(-(GX['MGFE']-W1['MGFE']-W2['MGFE'])
                        - (GX['MGNI']-W1['MGNI']-W2['MGNI'])
                        + (GX['FENI']-W1['FENI']-W2['FENI']))
    c['GS3S3'] = 0.25*(GX['MGCO']-W1['MGCO']-W2['MGCO'])
    c['GS3S4'] = 0.25*((GX['MGCO']-W1['MGCO']-W2['MGCO'])
                        + (GX['MGNI']-W1['MGNI']-W2['MGNI'])
                        - (GX['CONI']-W1['CONI']-W2['CONI']))
    c['GS4S4'] = 0.25*(GX['MGNI']-W1['MGNI']-W2['MGNI'])

    return c


def _v_taylor_coeffs():
    """V0, VR1..VR5, VS1..VS4, and quadratic VRiRj/VRiSj/VSiSj terms --
    the V-analogue of _g_taylor_coeffs, verbatim from olivine.c lines
    462-637. These use ONLY the *EX/*X/*W1/*W2/*W2CA "V" (not P-combined)
    base constants directly (no P-dependence at all -- V is a pure
    function of composition, matching gibbs.c's `#define V ...`)."""
    HEX, VEX, HX, VX = _HEX, _VEX, _HX, _VX  # noqa: reuse names for brevity
    WV1, WV2, WV2CA, F = _WV1, _WV2, _WV2CA, _F

    def VX_(k): return VX[k]
    def VEX_(k): return VEX[k]

    c = {}
    c['V0'] = 0.25*((VX_('MNFE')+WV1['MNFE']+WV2['MNFE'])
                     + (VX_('MNCO')+WV1['MNCO']+WV2['MNCO'])
                     + (VX_('MNNI')+WV1['MNNI']+WV2['MNNI'])
                     + (VX_('FECO')+WV1['FECO']+WV2['FECO'])
                     + (VX_('FENI')+WV1['FENI']+WV2['FENI'])
                     + (VX_('CONI')+WV1['CONI']+WV2['CONI'])
                     - 2.0*(VX_('MGMN')+WV1['MGMN']+WV2['MGMN'])
                     - 2.0*(VX_('MGFE')+WV1['MGFE']+WV2['MGFE'])
                     - 2.0*(VX_('MGCO')+WV1['MGCO']+WV2['MGCO'])
                     - 2.0*(VX_('MGNI')+WV1['MGNI']+WV2['MGNI']))
    c['VR1'] = 0.25*(-3.0*(VX_('MGMN')+WV1['MGMN']+WV2['MGMN'])
                      - (VX_('MGFE')+WV1['MGFE']+WV2['MGFE'])
                      - (VX_('MGCO')+WV1['MGCO']+WV2['MGCO'])
                      - (VX_('MGNI')+WV1['MGNI']+WV2['MGNI'])
                      + (VX_('MNFE')+WV1['MNFE']+WV2['MNFE'])
                      + (VX_('MNCO')+WV1['MNCO']+WV2['MNCO'])
                      + (VX_('MNNI')+WV1['MNNI']+WV2['MNNI']))
    c['VR2'] = 0.25*(-(VX_('MGMN')+WV1['MGMN']+WV2['MGMN'])
                      - 3.0*(VX_('MGFE')+WV1['MGFE']+WV2['MGFE'])
                      - (VX_('MGCO')+WV1['MGCO']+WV2['MGCO'])
                      - (VX_('MGNI')+WV1['MGNI']+WV2['MGNI'])
                      + (VX_('MNFE')+WV1['MNFE']+WV2['MNFE'])
                      + (VX_('FECO')+WV1['FECO']+WV2['FECO'])
                      + (VX_('FENI')+WV1['FENI']+WV2['FENI']))
    c['VR3'] = 0.25*(-(VX_('MGMN')+WV1['MGMN']+WV2['MGMN'])
                      - (VX_('MGFE')+WV1['MGFE']+WV2['MGFE'])
                      - 3.0*(VX_('MGCO')+WV1['MGCO']+WV2['MGCO'])
                      - (VX_('MGNI')+WV1['MGNI']+WV2['MGNI'])
                      + (VX_('MNCO')+WV1['MNCO']+WV2['MNCO'])
                      + (VX_('FECO')+WV1['FECO']+WV2['FECO'])
                      + (VX_('CONI')+WV1['CONI']+WV2['CONI']))
    c['VR4'] = 0.25*(-(VX_('MGMN')+WV1['MGMN']+WV2['MGMN'])
                      - (VX_('MGFE')+WV1['MGFE']+WV2['MGFE'])
                      - (VX_('MGCO')+WV1['MGCO']+WV2['MGCO'])
                      - 3.0*(VX_('MGNI')+WV1['MGNI']+WV2['MGNI'])
                      + (VX_('MNNI')+WV1['MNNI']+WV2['MNNI'])
                      + (VX_('FENI')+WV1['FENI']+WV2['FENI'])
                      + (VX_('CONI')+WV1['CONI']+WV2['CONI']))
    c['VR5'] = (-WV2CA['MG']
                - 0.25*((VEX_('MGMN')+VX_('MGMN')-2.0*WV2CA['MN']+2.0*WV2['MGMN'])
                        + (VEX_('MGFE')+VX_('MGFE')-2.0*WV2CA['FE']+2.0*WV2['MGFE'])
                        + (VEX_('MGCO')+VX_('MGCO')-2.0*WV2CA['CO']+2.0*WV2['MGCO'])
                        + (VEX_('MGNI')+VX_('MGNI')-2.0*WV2CA['NI']+2.0*WV2['MGNI'])))

    c['VS1'] = 0.25*((VEX_('MGMN')+3.0*WV1['MGMN']-3.0*WV2['MGMN'])
                      - (VEX_('MGFE')-WV1['MGFE']+WV2['MGFE'])
                      - (VEX_('MGCO')-WV1['MGCO']+WV2['MGCO'])
                      - (VEX_('MGNI')-WV1['MGNI']+WV2['MGNI'])
                      + (VEX_('MNFE')-WV1['MNFE']+WV2['MNFE'])
                      + (VEX_('MNCO')-WV1['MNCO']+WV2['MNCO'])
                      + (VEX_('MNNI')-WV1['MNNI']+WV2['MNNI']))
    c['VS2'] = 0.25*((-VEX_('MGMN')+WV1['MGMN']-WV2['MGMN'])
                      + (VEX_('MGFE')+3.0*WV1['MGFE']-3.0*WV2['MGFE'])
                      - (VEX_('MGCO')-WV1['MGCO']+WV2['MGCO'])
                      - (VEX_('MGNI')-WV1['MGNI']+WV2['MGNI'])
                      - (VEX_('MNFE')+WV1['MNFE']-WV2['MNFE'])
                      + (VEX_('FECO')-WV1['FECO']+WV2['FECO'])
                      + (VEX_('FENI')-WV1['FENI']+WV2['FENI']))
    c['VS3'] = 0.25*((VEX_('MGMN')-WV1['MGMN']+WV2['MGMN'])
                      + (VEX_('MGFE')-WV1['MGFE']+WV2['MGFE'])
                      - (VEX_('MGCO')+3.0*WV1['MGCO']-3.0*WV2['MGCO'])
                      + (VEX_('MGNI')-WV1['MGNI']+WV2['MGNI'])
                      + (VEX_('MNCO')+WV1['MNCO']-WV2['MNCO'])
                      + (VEX_('FECO')+WV1['FECO']-WV2['FECO'])
                      - (VEX_('CONI')-WV1['CONI']+WV2['CONI']))
    c['VS4'] = 0.25*((VEX_('MGMN')-WV1['MGMN']+WV2['MGMN'])
                      + (VEX_('MGFE')-WV1['MGFE']+WV2['MGFE'])
                      + (VEX_('MGCO')-WV1['MGCO']+WV2['MGCO'])
                      - (VEX_('MGNI')+3.0*WV1['MGNI']-3.0*WV2['MGNI'])
                      + (VEX_('MNNI')+WV1['MNNI']-WV2['MNNI'])
                      + (VEX_('FENI')+WV1['FENI']-WV2['FENI'])
                      + (VEX_('CONI')+WV1['CONI']-WV2['CONI']))

    c['VR1R1'] = -0.25*(VX_('MGMN')+WV1['MGMN']+WV2['MGMN'])
    c['VR1R2'] = 0.25*((VX_('MNFE')+WV1['MNFE']+WV2['MNFE'])
                        - (VX_('MGMN')+WV1['MGMN']+WV2['MGMN'])
                        - (VX_('MGFE')+WV1['MGFE']+WV2['MGFE']))
    c['VR1R3'] = 0.25*((VX_('MNCO')+WV1['MNCO']+WV2['MNCO'])
                        - (VX_('MGMN')+WV1['MGMN']+WV2['MGMN'])
                        - (VX_('MGCO')+WV1['MGCO']+WV2['MGCO']))
    c['VR1R4'] = 0.25*((VX_('MNNI')+WV1['MNNI']+WV2['MNNI'])
                        - (VX_('MGMN')+WV1['MGMN']+WV2['MGMN'])
                        - (VX_('MGNI')+WV1['MGNI']+WV2['MGNI']))
    c['VR1R5'] = -0.25*(VEX_('MGMN')+VX_('MGMN')
                         + 2.0*WV2CA['MG']-2.0*WV2CA['MN']+2.0*WV2['MGMN'])
    c['VR1S1'] = 0.5*(WV1['MGMN']-WV2['MGMN'])
    c['VR1S2'] = 0.25*((-VEX_('MGMN')-WV1['MGMN']+WV2['MGMN'])
                        + (VEX_('MGFE')+WV1['MGFE']-WV2['MGFE'])
                        - (VEX_('MNFE')+WV1['MNFE']-WV2['MNFE']))
    c['VR1S3'] = 0.25*((VEX_('MGMN')-WV1['MGMN']+WV2['MGMN'])
                        - (VEX_('MGCO')+WV1['MGCO']-WV2['MGCO'])
                        + (VEX_('MNCO')+WV1['MNCO']-WV2['MNCO']))
    c['VR1S4'] = 0.25*((VEX_('MGMN')-WV1['MGMN']+WV2['MGMN'])
                        - (VEX_('MGNI')+WV1['MGNI']-WV2['MGNI'])
                        + (VEX_('MNNI')+WV1['MNNI']-WV2['MNNI']))

    c['VR2R2'] = -0.25*(VX_('MGFE')+WV1['MGFE']+WV2['MGFE'])
    c['VR2R3'] = 0.25*((VX_('FECO')+WV1['FECO']+WV2['FECO'])
                        - (VX_('MGFE')+WV1['MGFE']+WV2['MGFE'])
                        - (VX_('MGCO')+WV1['MGCO']+WV2['MGCO']))
    c['VR2R4'] = 0.25*((VX_('FENI')+WV1['FENI']+WV2['FENI'])
                        - (VX_('MGFE')+WV1['MGFE']+WV2['MGFE'])
                        - (VX_('MGNI')+WV1['MGNI']+WV2['MGNI']))
    c['VR2R5'] = -0.25*(VEX_('MGFE')+VX_('MGFE')
                         + 2.0*WV2CA['MG']-2.0*WV2CA['FE']+2.0*WV2['MGFE'])
    c['VR2S1'] = 0.25*((VEX_('MGMN')+WV1['MGMN']-WV2['MGMN'])
                        - (VEX_('MGFE')-WV1['MGFE']+WV2['MGFE'])
                        + (VEX_('MNFE')-WV1['MNFE']+WV2['MNFE']))
    c['VR2S2'] = 0.5*(WV1['MGFE']-WV2['MGFE'])
    c['VR2S3'] = 0.25*((VEX_('MGFE')-WV1['MGFE']+WV2['MGFE'])
                        - (VEX_('MGCO')+WV1['MGCO']-WV2['MGCO'])
                        + (VEX_('FECO')+WV1['FECO']-WV2['FECO']))
    c['VR2S4'] = 0.25*((VEX_('MGFE')-WV1['MGFE']+WV2['MGFE'])
                        - (VEX_('MGNI')+WV1['MGNI']-WV2['MGNI'])
                        + (VEX_('FENI')+WV1['FENI']-WV2['FENI']))

    c['VR3R3'] = -0.25*(VX_('MGCO')+WV1['MGCO']+WV2['MGCO'])
    c['VR3R4'] = 0.25*((VX_('CONI')+WV1['CONI']+WV2['CONI'])
                        - (VX_('MGCO')+WV1['MGCO']+WV2['MGCO'])
                        - (VX_('MGNI')+WV1['MGNI']+WV2['MGNI']))
    c['VR3R5'] = -0.25*(VEX_('MGCO')+VX_('MGCO')
                         + 2.0*WV2CA['MG']-2.0*WV2CA['CO']+2.0*WV2['MGCO'])
    c['VR3S1'] = 0.25*((VEX_('MGMN')+WV1['MGMN']-WV2['MGMN'])
                        - (VEX_('MGCO')-WV1['MGCO']+WV2['MGCO'])
                        + (VEX_('MNCO')-WV1['MNCO']+WV2['MNCO']))
    c['VR3S2'] = 0.25*((VEX_('MGFE')+WV1['MGFE']-WV2['MGFE'])
                        - (VEX_('MGCO')-WV1['MGCO']+WV2['MGCO'])
                        + (VEX_('FECO')-WV1['FECO']+WV2['FECO']))
    c['VR3S3'] = 0.5*(-WV1['MGCO']+WV2['MGCO'])
    c['VR3S4'] = 0.25*((VEX_('MGCO')+WV1['MGCO']-WV2['MGCO'])
                        - (VEX_('MGNI')-WV1['MGNI']+WV2['MGNI'])
                        + (VEX_('CONI')-WV1['CONI']+WV2['CONI']))

    c['VR4R4'] = -0.25*(VX_('MGNI')+WV1['MGNI']+WV2['MGNI'])
    c['VR4R5'] = -0.25*(VEX_('MGNI')+VX_('MGNI')
                         + 2.0*WV2CA['MG']-2.0*WV2CA['NI']+2.0*WV2['MGNI'])
    c['VR4S1'] = 0.25*((VEX_('MGMN')+WV1['MGMN']-WV2['MGMN'])
                        - (VEX_('MGNI')-WV1['MGNI']+WV2['MGNI'])
                        + (VEX_('MNNI')-WV1['MNNI']+WV2['MNNI']))
    c['VR4S2'] = 0.25*((VEX_('MGFE')+WV1['MGFE']-WV2['MGFE'])
                        - (VEX_('MGNI')-WV1['MGNI']+WV2['MGNI'])
                        + (VEX_('FENI')-WV1['FENI']+WV2['FENI']))
    c['VR4S3'] = 0.25*((-VEX_('MGCO')+WV1['MGCO']-WV2['MGCO'])
                        + (VEX_('MGNI')-WV1['MGNI']+WV2['MGNI'])
                        - (VEX_('CONI')-WV1['CONI']+WV2['CONI']))
    c['VR4S4'] = 0.5*(-WV1['MGNI']+WV2['MGNI'])

    c['VR5R5'] = -WV2CA['MG']
    c['VR5S1'] = 0.25*(VEX_('MGMN')+VX_('MGMN')
                        - 2.0*WV2CA['MG']+2.0*WV2CA['MN']-2.0*WV2['MGMN'])
    c['VR5S2'] = 0.25*(VEX_('MGFE')+VX_('MGFE')
                        - 2.0*WV2CA['MG']+2.0*WV2CA['FE']-2.0*WV2['MGFE'])
    c['VR5S3'] = -0.25*(VEX_('MGCO')+VX_('MGCO')
                         - 2.0*WV2CA['MG']+2.0*WV2CA['CO']-2.0*WV2['MGCO'])
    c['VR5S4'] = -0.25*(VEX_('MGNI')+VX_('MGNI')
                         - 2.0*WV2CA['MG']+2.0*WV2CA['NI']-2.0*WV2['MGNI'])

    c['VS1S1'] = 0.25*(VX_('MGMN')-WV1['MGMN']-WV2['MGMN'])
    c['VS1S2'] = 0.25*((VX_('MGMN')-WV1['MGMN']-WV2['MGMN'])
                        + (VX_('MGFE')-WV1['MGFE']-WV2['MGFE'])
                        - (VX_('MNFE')-WV1['MNFE']-WV2['MNFE']))
    c['VS1S3'] = 0.25*(-(VX_('MGMN')-WV1['MGMN']-WV2['MGMN'])
                        - (VX_('MGCO')-WV1['MGCO']-WV2['MGCO'])
                        + (VX_('MNCO')-WV1['MNCO']-WV2['MNCO']))
    c['VS1S4'] = 0.25*(-(VX_('MGMN')-WV1['MGMN']-WV2['MGMN'])
                        - (VX_('MGNI')-WV1['MGNI']-WV2['MGNI'])
                        + (VX_('MNNI')-WV1['MNNI']-WV2['MNNI']))
    c['VS2S2'] = 0.25*(VX_('MGFE')-WV1['MGFE']-WV2['MGFE'])
    c['VS2S3'] = 0.25*(-(VX_('MGFE')-WV1['MGFE']-WV2['MGFE'])
                        - (VX_('MGCO')-WV1['MGCO']-WV2['MGCO'])
                        + (VX_('FECO')-WV1['FECO']-WV2['FECO']))
    c['VS2S4'] = 0.25*(-(VX_('MGFE')-WV1['MGFE']-WV2['MGFE'])
                        - (VX_('MGNI')-WV1['MGNI']-WV2['MGNI'])
                        + (VX_('FENI')-WV1['FENI']-WV2['FENI']))
    c['VS3S3'] = 0.25*(VX_('MGCO')-WV1['MGCO']-WV2['MGCO'])
    c['VS3S4'] = 0.25*((VX_('MGCO')-WV1['MGCO']-WV2['MGCO'])
                        + (VX_('MGNI')-WV1['MGNI']-WV2['MGNI'])
                        - (VX_('CONI')-WV1['CONI']-WV2['CONI']))
    c['VS4S4'] = 0.25*(VX_('MGNI')-WV1['MGNI']-WV2['MGNI'])

    return c


_V_COEFFS = _v_taylor_coeffs()   # composition-only, P-independent: compute once


def _quadratic_form(c, prefix, r, s):
    """sum of (prefix)RiRj*ri*rj + (prefix)RiSj*ri*sj + (prefix)SiSj*si*sj
    terms, i.e. the shared structure of the olivine.c H/V macros (lines
    955-989), given the appropriate coefficient dict `c` (G- or
    V-flavoured) and r (...,5), s (...,4) arrays."""
    r0, r1, r2, r3, r4 = r[..., 0], r[..., 1], r[..., 2], r[..., 3], r[..., 4]
    s0, s1, s2, s3 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    g = (c[f'{prefix}0']
         + c[f'{prefix}R1']*r0 + c[f'{prefix}R2']*r1 + c[f'{prefix}R3']*r2
         + c[f'{prefix}R4']*r3 + c[f'{prefix}R5']*r4
         + c[f'{prefix}S1']*s0 + c[f'{prefix}S2']*s1 + c[f'{prefix}S3']*s2 + c[f'{prefix}S4']*s3
         + c[f'{prefix}R1R1']*r0*r0 + c[f'{prefix}R1R2']*r0*r1 + c[f'{prefix}R1R3']*r0*r2
         + c[f'{prefix}R1R4']*r0*r3 + c[f'{prefix}R1R5']*r0*r4
         + c[f'{prefix}R1S1']*r0*s0 + c[f'{prefix}R1S2']*r0*s1 + c[f'{prefix}R1S3']*r0*s2 + c[f'{prefix}R1S4']*r0*s3
         + c[f'{prefix}R2R2']*r1*r1 + c[f'{prefix}R2R3']*r1*r2 + c[f'{prefix}R2R4']*r1*r3 + c[f'{prefix}R2R5']*r1*r4
         + c[f'{prefix}R2S1']*r1*s0 + c[f'{prefix}R2S2']*r1*s1 + c[f'{prefix}R2S3']*r1*s2 + c[f'{prefix}R2S4']*r1*s3
         + c[f'{prefix}R3R3']*r2*r2 + c[f'{prefix}R3R4']*r2*r3 + c[f'{prefix}R3R5']*r2*r4
         + c[f'{prefix}R3S1']*r2*s0 + c[f'{prefix}R3S2']*r2*s1 + c[f'{prefix}R3S3']*r2*s2 + c[f'{prefix}R3S4']*r2*s3
         + c[f'{prefix}R4R4']*r3*r3 + c[f'{prefix}R4R5']*r3*r4
         + c[f'{prefix}R4S1']*r3*s0 + c[f'{prefix}R4S2']*r3*s1 + c[f'{prefix}R4S3']*r3*s2 + c[f'{prefix}R4S4']*r3*s3
         + c[f'{prefix}R5R5']*r4*r4
         + c[f'{prefix}R5S1']*r4*s0 + c[f'{prefix}R5S2']*r4*s1 + c[f'{prefix}R5S3']*r4*s2 + c[f'{prefix}R5S4']*r4*s3
         + c[f'{prefix}S1S1']*s0*s0 + c[f'{prefix}S1S2']*s0*s1 + c[f'{prefix}S1S3']*s0*s2 + c[f'{prefix}S1S4']*s0*s3
         + c[f'{prefix}S2S2']*s1*s1 + c[f'{prefix}S2S3']*s1*s2 + c[f'{prefix}S2S4']*s1*s3
         + c[f'{prefix}S3S3']*s2*s2 + c[f'{prefix}S3S4']*s2*s3
         + c[f'{prefix}S4S4']*s3*s3)
    return g


def site_fractions(r, s):
    """(r,s) -> the 11 M1/M2 site mole fractions, verbatim from olivine.c
    lines 1915-1925. r: (...,5), s: (...,4). Returns a dict of (...,)
    arrays, clipped away from exactly 0/1 to avoid log(0) (matching
    olivine.c's own DBL_EPSILON floor, lines 1927-1937)."""
    r0, r1, r2, r3, r4 = r[..., 0], r[..., 1], r[..., 2], r[..., 3], r[..., 4]
    s0, s1, s2, s3 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]

    x = {}
    x['m1mn'] = (r0 - s0 + 1.0)/2.0
    x['m2mn'] = (r0 + s0 + 1.0)/2.0
    x['m1fe'] = (r1 - s1 + 1.0)/2.0
    x['m2fe'] = (r1 + s1 + 1.0)/2.0
    x['m1co'] = (r2 + s2 + 1.0)/2.0
    x['m2co'] = (r2 - s2 + 1.0)/2.0
    x['m1ni'] = (r3 + s3 + 1.0)/2.0
    x['m2ni'] = (r3 - s3 + 1.0)/2.0
    x['m2ca'] = r4
    x['m1mg'] = 1.0 - x['m1mn'] - x['m1fe'] - x['m1co'] - x['m1ni']
    x['m2mg'] = 1.0 - x['m2mn'] - x['m2fe'] - x['m2co'] - x['m2ni'] - x['m2ca']
    for k in x:
        x[k] = np.clip(x[k], _DBL_EPS, None)
    return x


def _entropy(x):
    return -Rgas*(x['m1mn']*np.log(x['m1mn']) + x['m2mn']*np.log(x['m2mn'])
                  + x['m1fe']*np.log(x['m1fe']) + x['m2fe']*np.log(x['m2fe'])
                  + x['m1co']*np.log(x['m1co']) + x['m2co']*np.log(x['m2co'])
                  + x['m1ni']*np.log(x['m1ni']) + x['m2ni']*np.log(x['m2ni'])
                  + x['m2ca']*np.log(x['m2ca'])
                  + x['m1mg']*np.log(x['m1mg']) + x['m2mg']*np.log(x['m2mg']))


def gibbs_mixing(r, s, T, P):
    """G(r, s, T, P) = H(r,s,P) - T*S(sitefractions), verbatim assembly
    from olivine.c (H macro line 955, S macro line 949, G = H - t*(S)).
    r: (...,5), s: (...,4), T,P broadcastable against r/s's leading dims.
    """
    Gc = _g_taylor_coeffs(P)
    H = _quadratic_form(Gc, 'G', r, s)
    x = site_fractions(r, s)
    S = _entropy(x)
    return H - T*S, H, S


def volume_mixing(r, s):
    """V(r,s), verbatim from olivine.c's V macro (line 974) -- no T or P
    dependence (P-dependence already fully absorbed into H via GEX/GX/W1/
    W2/W2CA)."""
    return _quadratic_form(_V_COEFFS, 'V', r, s)


# ---------------------------------------------------------------------
# dG/dr (activities) and dG/ds (ordering-parameter Newton solve),
# verbatim from olivine.c lines 998-1033.
# ---------------------------------------------------------------------

def _dgdr(Gc, r, s, T, x):
    r0, r1, r2, r3, r4 = r[..., 0], r[..., 1], r[..., 2], r[..., 3], r[..., 4]
    s0, s1, s2, s3 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    c = Gc
    d0 = (c['GR1'] + 2.0*c['GR1R1']*r0
          + c['GR1R2']*r1 + c['GR1R3']*r2 + c['GR1R4']*r3 + c['GR1R5']*r4
          + c['GR1S1']*s0 + c['GR1S2']*s1 + c['GR1S3']*s2 + c['GR1S4']*s3
          + 0.5*Rgas*T*np.log(x['m1mn']*x['m2mn']/(x['m1mg']*x['m2mg'])))
    d1 = (c['GR2'] + 2.0*c['GR2R2']*r1
          + c['GR1R2']*r0 + c['GR2R3']*r2 + c['GR2R4']*r3 + c['GR2R5']*r4
          + c['GR2S1']*s0 + c['GR2S2']*s1 + c['GR2S3']*s2 + c['GR2S4']*s3
          + 0.5*Rgas*T*np.log(x['m1fe']*x['m2fe']/(x['m1mg']*x['m2mg'])))
    d2 = (c['GR3'] + 2.0*c['GR3R3']*r2
          + c['GR1R3']*r0 + c['GR2R3']*r1 + c['GR3R4']*r3 + c['GR3R5']*r4
          + c['GR3S1']*s0 + c['GR3S2']*s1 + c['GR3S3']*s2 + c['GR3S4']*s3
          + 0.5*Rgas*T*np.log(x['m1co']*x['m2co']/(x['m1mg']*x['m2mg'])))
    d3 = (c['GR4'] + 2.0*c['GR4R4']*r3
          + c['GR1R4']*r0 + c['GR2R4']*r1 + c['GR3R4']*r2 + c['GR4R5']*r4
          + c['GR4S1']*s0 + c['GR4S2']*s1 + c['GR4S3']*s2 + c['GR4S4']*s3
          + 0.5*Rgas*T*np.log(x['m1ni']*x['m2ni']/(x['m1mg']*x['m2mg'])))
    d4 = (c['GR5'] + 2.0*c['GR5R5']*r4
          + c['GR1R5']*r0 + c['GR2R5']*r1 + c['GR3R5']*r2 + c['GR4R5']*r3
          + c['GR5S1']*s0 + c['GR5S2']*s1 + c['GR5S3']*s2 + c['GR5S4']*s3
          + Rgas*T*np.log(x['m2ca']/x['m2mg']))
    return np.stack([d0, d1, d2, d3, d4], axis=-1)


def _dgds(Gc, r, s, T, x):
    r0, r1, r2, r3, r4 = r[..., 0], r[..., 1], r[..., 2], r[..., 3], r[..., 4]
    s0, s1, s2, s3 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    c = Gc
    d0 = (c['GS1'] + 2.0*c['GS1S1']*s0
          + c['GR1S1']*r0 + c['GR2S1']*r1 + c['GR3S1']*r2 + c['GR4S1']*r3 + c['GR5S1']*r4
          + c['GS1S2']*s1 + c['GS1S3']*s2 + c['GS1S4']*s3
          + 0.5*Rgas*T*np.log(x['m2mn']*x['m1mg']/(x['m1mn']*x['m2mg'])))
    d1 = (c['GS2'] + 2.0*c['GS2S2']*s1
          + c['GR1S2']*r0 + c['GR2S2']*r1 + c['GR3S2']*r2 + c['GR4S2']*r3 + c['GR5S2']*r4
          + c['GS1S2']*s0 + c['GS2S3']*s2 + c['GS2S4']*s3
          + 0.5*Rgas*T*np.log(x['m2fe']*x['m1mg']/(x['m1fe']*x['m2mg'])))
    d2 = (c['GS3'] + 2.0*c['GS3S3']*s2
          + c['GR1S3']*r0 + c['GR2S3']*r1 + c['GR3S3']*r2 + c['GR4S3']*r3 + c['GR5S3']*r4
          + c['GS1S3']*s0 + c['GS2S3']*s1 + c['GS3S4']*s3
          + 0.5*Rgas*T*np.log(x['m1co']*x['m2mg']/(x['m2co']*x['m1mg'])))
    d3 = (c['GS4'] + 2.0*c['GS4S4']*s3
          + c['GR1S4']*r0 + c['GR2S4']*r1 + c['GR3S4']*r2 + c['GR4S4']*r3 + c['GR5S4']*r4
          + c['GS1S4']*s0 + c['GS2S4']*s1 + c['GS3S4']*s2
          + 0.5*Rgas*T*np.log(x['m1ni']*x['m2mg']/(x['m2ni']*x['m1mg'])))
    return np.stack([d0, d1, d2, d3], axis=-1)


def _dgds_func(s, r, T, P):
    """(B,4) dG/ds callable for newton_solve_ordering: r, T, P (all
    plain (B,) / (B,5)) are the fixed batch state; s (B,4) varies."""
    Gc = _g_taylor_coeffs(P)
    x = site_fractions(r, s)
    return _dgds(Gc, r, s, T, x)


def x_to_r(X):
    """Bulk mole fractions [x_teph, x_fay, x_co, x_ni, x_mont, x_fo]
    (..., 6) -> composition variables r (..., 5), per olivine.c's
    `actOlv` FOURTH-mask inverse mapping (x[i] = (r[i]+1)/2 for i=0..3,
    x[4] = r[4])."""
    X = np.asarray(X, dtype=np.float64)
    r0 = 2.0*X[..., 0] - 1.0
    r1 = 2.0*X[..., 1] - 1.0
    r2 = 2.0*X[..., 2] - 1.0
    r3 = 2.0*X[..., 3] - 1.0
    r4 = X[..., 4]
    return np.stack([r0, r1, r2, r3, r4], axis=-1)


def _fr_matrix(r):
    """FR matrix (component i, r-index j), verbatim from olivine.c's
    FR0(i)..FR4(i) macros (lines 922-926): FRj(i) = (i==j) ? 1-r_j :
    -(1+r_j) for j=0..3, and FR4(i) = (i==4) ? 1-r4 : -r4."""
    B = r.shape[0]
    r0, r1, r2, r3, r4 = r[:, 0], r[:, 1], r[:, 2], r[:, 3], r[:, 4]
    fr = np.empty((B, 6, 5), dtype=np.float64)
    for i in range(6):
        fr[:, i, 0] = (1.0 - r0) if i == 0 else -(1.0 + r0)
        fr[:, i, 1] = (1.0 - r1) if i == 1 else -(1.0 + r1)
        fr[:, i, 2] = (1.0 - r2) if i == 2 else -(1.0 + r2)
        fr[:, i, 3] = (1.0 - r3) if i == 3 else -(1.0 + r3)
        fr[:, i, 4] = (1.0 - r4) if i == 4 else -r4
    return fr


def solve_ordering(r, T, P, n_iter=60):
    """Batched Newton solve for the equilibrium ordering parameters
    s* (B,4), given bulk composition r (B,5) and T,P (B,)."""
    r = np.asarray(r, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    B = r.shape[0]
    s0 = np.zeros((B, 4), dtype=np.float64)
    # r_i == -1 (a completely absent minor component) forces s_i == 0
    # exactly (olivine.c lines 1910-1913) -- clip toward that with tight
    # bounds to keep the Newton solve well-posed there too.
    s_bound = 0.999*np.ones(4)
    s = newton_solve_ordering(_dgds_func, s0, r, T, P, n_iter=n_iter,
                                s_min=-s_bound, s_max=s_bound)
    for i in range(4):
        s[:, i] = np.where(np.abs(r[:, i] + 1.0) < 1e-8, 0.0, s[:, i])
    return s


def _g_and_v(r, T, P, n_iter):
    """s*(r,T,P), gmix, H_mix, S_mix, V_mix -- all plain (B,) T,P in,
    (B,) g/H/S/V out."""
    s_eq = solve_ordering(r, T, P, n_iter=n_iter)
    g, H, S = gibbs_mixing(r, s_eq, T, P)
    V = volume_mixing(r, s_eq)
    return g, H, S, V, s_eq


def solution_thermo(r, T, P, n_iter=60, dT=0.02, dP=0.02):
    """Full olivine solid-solution mixing thermodynamics at bulk
    composition r (B,5) and T,P (B,): solves for s*, then returns gmix,
    H_mix, S_mix (analytic, exact at s* -- envelope theorem), V_mix
    (analytic in composition, s*-dependent), dVdT_mix/dVdP_mix/Cp_mix
    (finite-differenced with s* re-solved at each T,P stencil point --
    see solution_model.py docstring), and Darken activities for the 6
    endmembers.
    """
    r = np.asarray(r, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)

    g0, H0, S0, V0, s_eq0 = _g_and_v(r, T, P, n_iter)

    # dV/dT, dV/dP via central FD with re-equilibrated s*.
    _, _, _, V_Tp, _ = _g_and_v(r, T + dT, P, n_iter)
    _, _, _, V_Tm, _ = _g_and_v(r, T - dT, P, n_iter)
    _, _, _, V_Pp, _ = _g_and_v(r, T, P + dP, n_iter)
    _, _, _, V_Pm, _ = _g_and_v(r, T, P - dP, n_iter)
    dVdT_mix = (V_Tp - V_Tm)/(2.0*dT)
    dVdP_mix = (V_Pp - V_Pm)/(2.0*dP)

    # Cp = -T * d^2G/dT^2 via central FD (re-equilibrated s* at each T).
    g_Tp, _, _, _, _ = _g_and_v(r, T + dT, P, n_iter)
    g_Tm, _, _, _, _ = _g_and_v(r, T - dT, P, n_iter)
    d2GdT2 = (g_Tp - 2.0*g0 + g_Tm)/(dT*dT)
    Cp_mix = -T*d2GdT2
    dCpdT_mix = np.zeros_like(Cp_mix)  # not needed downstream; higher-order FD omitted

    Gc = _g_taylor_coeffs(P)
    x = site_fractions(r, s_eq0)
    dgdr = _dgdr(Gc, r, s_eq0, T, x)
    fr = _fr_matrix(r)
    mu, a = darken_activities(g0, dgdr, fr, Rgas, T)

    return dict(
        gmix=g0, H_mix=H0, S_mix=S0, V_mix=V0, Cp_mix=Cp_mix, dCpdT_mix=dCpdT_mix,
        dVdT_mix=dVdT_mix, dVdP_mix=dVdP_mix,
        mu=mu, activities=a, endmembers=ENDMEMBERS, s_eq=s_eq0,
    )
