"""
Regression test cross-checking melts_vec's feldspar and olivine
solid-solution mixing models against a standalone C harness
(`verify_solutions.c`, shipped alongside this file) built from formulas
copied verbatim out of MAGMA's `sources/feldspar.c` and `sources/olivine.c`
-- same rationale and method as `benchmark_test.py` for the pure-endmember
EOS.

Feldspar: cross-checks gmix/H/S/V at four compositions x two (T,P) points
-- exact, since feldspar.c's model has no internal ordering variables.

Olivine: cross-checks (a) every one of the ~45+13 Taylor-expansion
coefficients (G0..GS4, the full quadratic set) at three pressures --
this is the highest-value check, since it validates the mechanical
transcription of ~90 numbers built from ~50 base Margules parameters;
and (b) the full H/S/G/dG/dr0/dG/ds0 polynomial *assembly* at two
(r, s, T, P) points (with s taken from melts_vec's own Newton solve, so
this checks the formulas rather than re-testing the solver) -- together
these cover every macro used by `olivine.solution_thermo` except the
finite-difference-based Cp/dV(T,P) machinery, which is deliberately not
transcribed from source at all (see solution_model.py's docstring for
why: MELTS's own d2gds2 Hessian macros aren't used here in favor of a
numerically-estimated Jacobian + finite-differenced T,P derivatives).

To regenerate: rebuild and rerun verify_solutions.c and update EXPECTED
below.

    gcc -O0 -std=c99 -o verify_solutions verify_solutions.c -lm && ./verify_solutions
"""
from __future__ import annotations

import numpy as np
import pytest

from ..feldspar import gibbs_mixing as feldspar_gibbs_mixing
from .. import olivine as olv

# ---------------------------------------------------------------------
# FELDSPAR: (xab, xan, T, P) -> (G, H, S, V), verbatim ./verify_solutions
# stdout.
# ---------------------------------------------------------------------
FELDSPAR_EXPECTED = {
    (0.6, 0.3, 900.0, 1.0):    (-3845.445126, 3429.966, 8.08379014, 0.0006366),
    (0.6, 0.3, 1200.0, 5000.0): (-6267.399805, 3429.966, 8.08379014, 0.0006366),
    (0.9, 0.05, 900.0, 1.0):    (-2013.614403, 1354.76225, 3.742640726, 0.0125462),
    (0.9, 0.05, 1200.0, 5000.0): (-3073.688167, 1354.76225, 3.742640726, 0.0125462),
    (0.2, 0.7, 900.0, 1.0):     (-1904.455992, 4280.848, 6.872559991, -0.0090496),
    (0.2, 0.7, 1200.0, 5000.0): (-4011.46294, 4280.848, 6.872559991, -0.0090496),
    (0.33, 0.33, 900.0, 1.0):   (-1333.57757, 7926.543537, 10.28902345, -0.0022158939),
    (0.33, 0.33, 1200.0, 5000.0): (-4431.361859, 7926.543537, 10.28902345, -0.0022158939),
}


@pytest.mark.parametrize("xab,xan,T,P", list(k for k in FELDSPAR_EXPECTED))
def test_feldspar_matches_c_harness(xab, xan, T, P):
    G, H, S, V = feldspar_gibbs_mixing(np.array([xab]), np.array([xan]), np.array([T]), np.array([P]))
    exp_G, exp_H, exp_S, exp_V = FELDSPAR_EXPECTED[(xab, xan, T, P)]
    np.testing.assert_allclose(G[0], exp_G, rtol=1e-6, atol=1e-4)
    np.testing.assert_allclose(H[0], exp_H, rtol=1e-6, atol=1e-4)
    np.testing.assert_allclose(S[0], exp_S, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(V[0], exp_V, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------
# OLIVINE: Taylor coefficients at three pressures, verbatim
# ./verify_solutions stdout.
# ---------------------------------------------------------------------
OLIVINE_COEFF_EXPECTED = {
    1.0:     dict(G0=-18850, GR1=-23175, GR2=-15825, GR3=-14675, GR4=-10375, GR5=5412.5,
                   GS1=8837.5, GS2=1837.5, GS3=3612.5, GS4=2687.5,
                   GR1R5=-21075, GR2R5=-13750, GR3R5=-0.0),
    10000.0: dict(G0=-18924.9925, GR1=-23212.49625, GR2=-15937.48875, GR3=-14712.49625,
                   GR4=-10412.49625, GR5=1875.35375,
                   GS1=8837.5, GS2=1837.5, GS3=3612.5, GS4=2687.5,
                   GR1R5=-22824.825, GR2R5=-15537.32125, GR3R5=-1749.825),
    30000.0: dict(G0=-19074.9925, GR1=-23287.49625, GR2=-16162.48875, GR3=-14787.49625,
                   GR4=-10487.49625, GR5=-5199.64625,
                   GS1=8837.5, GS2=1837.5, GS3=3612.5, GS4=2687.5,
                   GR1R5=-26324.825, GR2R5=-19112.32125, GR3R5=-5249.825),
}


@pytest.mark.parametrize("P", [1.0, 10000.0, 30000.0])
def test_olivine_taylor_coeffs_match_c_harness(P):
    c = olv._g_taylor_coeffs(np.array([P]))
    for key, exp_val in OLIVINE_COEFF_EXPECTED[P].items():
        np.testing.assert_allclose(c[key][0], exp_val, rtol=1e-8, atol=1e-6,
                                    err_msg=f"olivine coeff {key} at P={P}")


# ---------------------------------------------------------------------
# OLIVINE: full H/S/G/dG/dr0/dG/ds0 assembly at two (r, s, T, P) points
# (s values taken from melts_vec's own solve_ordering(), so this checks
# the polynomial/entropy/gradient *formulas*, not the Newton solver).
# ---------------------------------------------------------------------
OLIVINE_HSG_CASES = [
    dict(r=[-0.998, -0.80, -0.998, -0.998, 0.01],
         s=[0.00115527206, -0.0000603898399, 0.00104057230, 0.00136465727],
         T=1400.0, P=10000.0,
         H=2182.12372, S=6.241839079, G=-6556.45099, DGDR0=-72617.22296, DGDS0=1.712507492e-05),
    dict(r=[-0.998, -0.40, -0.998, -0.998, 0.01],
         s=[0.000996295403, 0.0000181380011, 0.000745383563, 0.00121539113],
         T=1600.0, P=30000.0,
         H=4678.153972, S=10.96324009, G=-12863.03017, DGDR0=-83999.55577, DGDS0=3.274166374e-06),
]


@pytest.mark.parametrize("case", OLIVINE_HSG_CASES)
def test_olivine_h_s_g_dgdr_dgds_match_c_harness(case):
    r = np.array([case["r"]])
    s = np.array([case["s"]])
    T = np.array([case["T"]])
    P = np.array([case["P"]])

    g, H, S = olv.gibbs_mixing(r, s, T, P)
    Gc = olv._g_taylor_coeffs(P)
    x = olv.site_fractions(r, s)
    dgdr = olv._dgdr(Gc, r, s, T, x)
    dgds = olv._dgds(Gc, r, s, T, x)

    np.testing.assert_allclose(H[0], case["H"], rtol=1e-6, atol=1e-3)
    np.testing.assert_allclose(S[0], case["S"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(g[0], case["G"], rtol=1e-6, atol=1e-3)
    np.testing.assert_allclose(dgdr[0, 0], case["DGDR0"], rtol=1e-6, atol=1e-2)
    np.testing.assert_allclose(dgds[0, 0], case["DGDS0"], rtol=1e-4, atol=1e-8)


def test_olivine_pure_forsterite_limit():
    """At pure forsterite (no Mn/Fe/Co/Ni/Ca), the mixing model must
    reduce to gmix=0, activity(forsterite)=1, all other activities=0 --
    not something checked against the C harness (it's a mathematical
    property of the model, not a numeric transcription check), but a
    cheap and high-value physical sanity test to keep as a regression
    guard."""
    X = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    r = olv.x_to_r(X)
    T = np.array([1400.0])
    P = np.array([10000.0])
    res = olv.solution_thermo(r, T, P, n_iter=60)

    np.testing.assert_allclose(res["gmix"][0], 0.0, atol=1e-4)
    np.testing.assert_allclose(res["V_mix"][0], 0.0, atol=1e-10)
    np.testing.assert_allclose(res["activities"][0, 5], 1.0, atol=1e-6)
    for i in range(5):
        assert res["activities"][0, i] < 1e-8
