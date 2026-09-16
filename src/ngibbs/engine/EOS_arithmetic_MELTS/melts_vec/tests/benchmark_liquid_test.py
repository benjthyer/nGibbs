"""
Regression test cross-checking melts_vec.liquid_eos.kress_component
against a standalone C harness (`verify_liquid.c`, shipped alongside this
file) built from formulas copied verbatim out of MAGMA's `sources/gibbs.c`
(the generic liquid branch) -- same method as `benchmark_test.py` for the
solid EOS and `benchmark_solutions_test.py` for feldspar/olivine.

This test's three components were chosen deliberately, not at random:
SiO2 and Al2O3 have a plain Berman solid reference (no order-disorder
correction), while KAlSiO4's solid reference phase (sanidine-ish) DOES
have a nonzero lambda transition (Tt = 800.15 K, well below its
Tfusion = 2023.15 K) that contributes to the fusion-state H, S. An
earlier draft of `kress_component` silently dropped that term (it wasn't
wired through from `liquid_params.py` at all) -- this test's KAlSiO4
case is what catches a regression of that bug.

Both (T, P) test points per component were also chosen deliberately:
(T=Trl=1673, P=Pr=1) makes every pressure-correction term in the Kress
EOS vanish identically (dT=dP=0), so it exercises only the fusion+cp_liquid
integration; (T=1200, P=10000) has dT!=0 and dP!=0 and is what caught a
second real bug -- a sign error in the H pressure correction
(`H = hl + press_integral - T*s_press_corr` should have been `+`, since
gibbs.c's own H correction is `-t*(dvdt*dP + 0.5*d2vdtp*dP^2)` which
equals `+T*s_press_corr` given how `s_press_corr` is defined here) that
the first (degenerate) test point could not have detected on its own.

To regenerate: rebuild and rerun verify_liquid.c and update EXPECTED
below.

    gcc -O0 -std=c99 -o verify_liquid verify_liquid.c -lm && ./verify_liquid
"""
from __future__ import annotations

import numpy as np
import pytest

from ..liquid_eos import kress_component

_COMPS = {
    "SiO2": dict(ref_h=-906377.0, ref_s=46.029, ref_k0=83.51, ref_k1=-374.7,
                 ref_k2=-2455400.0, ref_k3=280070000.0,
                 ref_cp_t=0.0, ref_cp_h=0.0, ref_l1=0.0, ref_l2=0.0,
                 v_liq=2.69, dvdt=0.0, dvdp=-1.89e-05, d2vdtp=1.3e-08, d2vdp2=3.6e-10,
                 t_fusion=1999.0, s_fusion=4.46, cp_liquid=82.6),
    "Al2O3": dict(ref_h=-1675700.0, ref_s=50.82, ref_k0=155.02, ref_k1=-828.4,
                  ref_k2=-3861400.0, ref_k3=409080000.0,
                  ref_cp_t=0.0, ref_cp_h=0.0, ref_l1=0.0, ref_l2=0.0,
                  v_liq=3.711, dvdt=0.000262, dvdp=-2.26e-05, d2vdtp=2.7e-08, d2vdp2=4.0e-10,
                  t_fusion=2319.65, s_fusion=48.61, cp_liquid=170.3),
    "KAlSiO4": dict(ref_h=-2111813.55, ref_s=133.9653, ref_k0=186.0, ref_k1=0.0,
                    ref_k2=-13106700.0, ref_k3=2138930000.0,
                    ref_cp_t=800.15, ref_cp_h=1154.0, ref_l1=-0.07096454, ref_l2=0.00021682,
                    v_liq=6.8375, dvdt=0.0007265, dvdp=-6.395e-05, d2vdtp=-4.6e-08, d2vdp2=1.21e-09,
                    t_fusion=2023.15, s_fusion=24.5, cp_liquid=217.0),
}

# (label, T, P) -> (G, H, S, Cp, V, dVdT, dVdP), verbatim ./verify_liquid stdout.
EXPECTED = {
    ("SiO2", 1673.0, 1.0):        (-1072809.534, -808383.8467, 158.0548041, 82.6, 2.69, 0.0, -1.89e-05),
    ("SiO2", 1200.0, 10000.0):    (-978477.0538, -822528.3982, 129.957213, 82.6, 2.457531449, 0.000129987, -2.144936e-05),
    ("Al2O3", 1673.0, 1.0):       (-1895092.861, -1419905.021, 284.0333767, 170.3, 3.711, 0.000262, -2.26e-05),
    ("Al2O3", 1200.0, 10000.0):   (-1739223.183, -1471054.680, 223.4737526, 170.3, 3.253395371, 0.000531973, -3.13714e-05),
    ("KAlSiO4", 1673.0, 1.0):     (-2562165.789, -1840807.15, 431.176712, 217.0, 6.837500001, 0.0007265, -6.395e-05),
    ("KAlSiO4", 1200.0, 10000.0): (-2311305.511, -1886381.241, 354.1035588, 217.0, 6.132475593, 0.000266546, -3.009321e-05),
}


@pytest.mark.parametrize("label,T,P", list(EXPECTED.keys()))
def test_liquid_component_matches_c_harness(label, T, P):
    kw = _COMPS[label]
    out = kress_component(np.array([T]), np.array([P]), **kw)
    exp_G, exp_H, exp_S, exp_Cp, exp_V, exp_dVdT, exp_dVdP = EXPECTED[(label, T, P)]

    np.testing.assert_allclose(out["G"][0], exp_G, rtol=1e-6, atol=1e-2)
    np.testing.assert_allclose(out["H"][0], exp_H, rtol=1e-6, atol=1e-2)
    np.testing.assert_allclose(out["S"][0], exp_S, rtol=1e-6, atol=1e-4)
    np.testing.assert_allclose(out["Cp"][0], exp_Cp, rtol=1e-6, atol=1e-4)
    np.testing.assert_allclose(out["V"][0], exp_V, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out["dVdT"][0], exp_dVdT, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(out["dVdP"][0], exp_dVdP, rtol=1e-6, atol=1e-10)
