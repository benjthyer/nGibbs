"""
Regression test cross-checking melts_vec's clinopyroxene solid-solution
mixing model against a standalone C harness (`verify_clinopyroxene.c`,
shipped alongside this file). Unlike verify_solutions.c (which re-derives
feldspar/olivine formulas by hand), verify_clinopyroxene.c's macro
definitions (verify_cpx_macros.h) are extracted VERBATIM (byte-for-byte
`sed` ranges) straight out of sources/clinopyroxene.c -- so this test is
checking melts_vec.clinopyroxene's translation against the literal source
text, not against a second hand-transcription that could carry its own
errors.

Coverage:
  - A spot-check of 17 Taylor-expansion coefficients (out of ~150),
    chosen to span every coefficient "family" (linear/quadratic/cubic,
    H/S/V) -- most valuable is SX5/SX6/SS1, which must come out to
    *exactly* 0.0 (a real bug during development had these three
    referencing the wrong base constant, "W55" instead of the distinct,
    always-zero "S55"; see clinopyroxene.py git history / PR notes).
  - Full H/S/G/V/DGDR0-5/DGDS0-1 assembly at three interior (r, s, T, P)
    points, including one point with a clipped near-zero/negative site
    fraction (xal3tet at the second test point) -- this is what caught
    a second real bug (Python's DBL_EPSILON clip constant didn't match
    C's float.h DBL_EPSILON, causing an ~1e5 J/mol divergence in the
    affected dG/dr, dG/ds ideal-mixing terms at that point).
  - Essenite's own internal-ordering Newton solve and ES_G/H/S/V, at one
    (T, P) point.
  - Pure-endmember gmix ~ 0 sanity checks at all 7 vertices (a
    mathematical property of the ENDMEMBERS-subtraction construction,
    not a C-harness check per se, but a cheap high-value regression
    guard -- this is what caught a third real bug, a missing term in
    the configurational entropy SIC formula that only showed up as a
    nonzero residual at the alumino-buffonite/buffonite vertices,
    where SIC itself is nonzero-but-should-cancel rather than ~0).

To regenerate: rebuild and rerun verify_clinopyroxene.c and update
EXPECTED below.

    gcc -O0 -o verify_clinopyroxene verify_clinopyroxene.c -lm && ./verify_clinopyroxene
"""
from __future__ import annotations

import numpy as np
import pytest

from .. import clinopyroxene as cpx

# ---------------------------------------------------------------------
# Taylor coefficient spot-check, verbatim ./verify_clinopyroxene stdout
# (=== CPX_COEFFS ===).
# ---------------------------------------------------------------------
COEFF_EXPECTED = dict(
    H0=0.0, HX2=7029.12, HX7=11610.10336, HS1=-2677.12, HS2=-9811.976641,
    HX3X7=22496.35168, HX7X7=-17564.45836, HX7X7X7=590.99, HX7S2S2=-10695.35,
    SX7=-3.97551128, SX2X3=0.0, SX7X7=5.7997562, SX5=0.0, SX6=0.0, SS1=0.0,
    VX2=0.0, VX7=-0.0106789, VX7X7=0.04624288266,
)


def test_taylor_coefficients_match_c_harness():
    # H_COEF/S_COEF/V_COEF key their entries WITHOUT the leading H/S/V
    # letter (e.g. H_COEF["X2"], not H_COEF["HX2"]) -- see clinopyroxene.py.
    for name, exp_val in COEFF_EXPECTED.items():
        table = cpx.H_COEF if name.startswith('H') else (cpx.S_COEF if name.startswith('S') else cpx.V_COEF)
        got = table.get(name[1:], 0.0)
        np.testing.assert_allclose(got, exp_val, rtol=1e-8, atol=1e-6,
                                    err_msg=f"coefficient {name}")


# ---------------------------------------------------------------------
# Full H/S/G/V/DGDR0-5/DGDS0-1 assembly, verbatim ./verify_clinopyroxene
# stdout (=== CPX_MIXED_G_H_S_V_DGDR_DGDS ===).
# ---------------------------------------------------------------------
MIXED_CASES = [
    dict(r=[0.15, 0.05, 0.04, 0.03, 0.02, 0.30], s=[0.02, -0.05], T=1350.0, P=5000.0,
         H=5544.298639, S=10.76702313, G=-9020.494015, V=-0.005863459637,
         DGDR=[-35262.48355, -4702.319252, -8102.129657, -31311.36954, -42138.45521, 8590.724128],
         DGDS=[5519.752996, 9598.558032]),
    dict(r=[0.30, 0.02, 0.10, 0.01, 0.05, 0.10], s=[-0.03, 0.08], T=1500.0, P=12000.0,
         H=3191.559009, S=10.26017525, G=-12235.25064, V=-0.003045817989,
         DGDR=[-13731.65026, -416712.3696, -5109.866796, -252141.9731, 63673.56942, -16560.58414],
         DGDS=[-225209.4514, 13283.74241]),
    dict(r=[0.05, 0.20, 0.02, 0.15, 0.08, 0.45], s=[0.10, -0.10], T=1250.0, P=1000.0,
         H=9811.493202, S=15.34828045, G=-9384.626334, V=-0.01077974767,
         DGDR=[-370336.8874, 1978.746325, -15529.50178, -19187.55766, -17156.17954, 181164.2585],
         DGDS=[10777.55367, 178428.5241]),
]


@pytest.mark.parametrize("case", MIXED_CASES)
def test_h_s_g_v_dgdr_dgds_match_c_harness(case):
    r = np.array([case["r"]])
    s = np.array([case["s"]])
    T = np.array([case["T"]])
    P = np.array([case["P"]])

    G, H, S, V = cpx.gibbs_total(r, s, T, P)
    dgdr = cpx.dgdr(r, s, T, P)
    dgds = cpx.dgds(r, s, T, P)

    np.testing.assert_allclose(H[0], case["H"], rtol=1e-6, atol=1e-2)
    np.testing.assert_allclose(S[0], case["S"], rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(G[0], case["G"], rtol=1e-6, atol=1e-2)
    np.testing.assert_allclose(V[0], case["V"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(dgdr[0], case["DGDR"], rtol=1e-5, atol=1e-1)
    np.testing.assert_allclose(dgds[0], case["DGDS"], rtol=1e-5, atol=1e-1)


# ---------------------------------------------------------------------
# Essenite's own internal ordering + ES_G/H/S/V, verbatim
# ./verify_clinopyroxene stdout (=== CPX_ESSENITE_ORDER ===, T=1500/P=8000
# row -- the T=1300/P=1000 row NaN'd in the harness's own toy Newton
# starting guess and isn't used here).
# ---------------------------------------------------------------------
def test_essenite_internal_ordering_matches_c_harness():
    T = np.array([1500.0])
    P = np.array([8000.0])
    s = cpx._pure_es_order(T, P)
    g, h, ss, v = cpx._pure_essenite_ghsv(T, P)

    np.testing.assert_allclose(s[0], 0.4973203062, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(g[0], -4966.889367, rtol=1e-6, atol=1e-2)
    np.testing.assert_allclose(h[0], 9095.975905, rtol=1e-6, atol=1e-2)
    np.testing.assert_allclose(ss[0], 9.375243515, rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(v[0], 0.0, atol=1e-10)


# ---------------------------------------------------------------------
# Pure-endmember gmix ~ 0 at all 7 vertices -- a mathematical property
# of the ENDMEMBERS-subtraction construction (not a C-harness numeric
# check per se), but a cheap, high-value regression guard: this is what
# caught the missing "-(1-xmg2m1-xfe2m1)*log(1-xmg2m1-xfe2m1)" term in
# SIC (the alumino-buffonite/buffonite vertices have nonzero SIC that
# must cancel between the mixed-phase G and the model's own pure-G
# reference; DI/EN/HD/ES/JD have SIC~0 at their vertices so wouldn't
# have caught this).
# ---------------------------------------------------------------------
def test_pure_endmember_gmix_is_zero():
    T = np.array([1300.0] * 7)
    P = np.array([1000.0] * 7)
    X = np.eye(7)
    r = cpx.x_to_r(X)
    res = cpx.solution_thermo(r, T, P, n_iter=80)
    np.testing.assert_allclose(res["gmix"], np.zeros(7), atol=2e-3)


def test_alumino_buffonite_and_buffonite_vertex_gfull_matches_c_harness():
    """The two endmembers with intrinsic (nonzero, but self-cancelling)
    site-mixing entropy at their own 'pure' vertex -- direct numeric
    check against the C harness's Gfull column (not just the gmix~0
    regression guard above)."""
    T = np.array([1300.0, 1500.0])
    P = np.array([1000.0, 8000.0])
    r = np.array([[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=np.float64)
    s = np.zeros((2, 2))
    G, H, S, V = cpx.gibbs_total(r, s, T, P)
    np.testing.assert_allclose(G, [-7.803464541e-10, -9.003997548e-10], atol=1e-6)

    r_cf = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]], dtype=np.float64)
    G_cf, _, _, _ = cpx.gibbs_total(r_cf, s, T, P)
    np.testing.assert_allclose(G_cf, [-7.785274647e-10, -8.985807654e-10], atol=1e-6)
