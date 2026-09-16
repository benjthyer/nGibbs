"""
Regression test cross-checking melts_vec's Berman/Vinet solid EOS against a
standalone C harness (`verify_eos.c`, shipped alongside this file) built
from formulas copied verbatim out of MAGMA's `sources/gibbs.c`
(`intEOSsolid()` lines ~449-533, and the CP_BERMAN branch of the generic
solid Gibbs energy calculation, lines ~2474-2535).

Why this test exists
---------------------
`melts_vec` is a from-scratch numpy re-implementation of that C arithmetic,
not a wrapper around the real library, so there's real risk of a subtle
translation error (sign flip, wrong exponent, swapped P/Pr, etc.) that
still produces physically-plausible-looking numbers. To rule that out, we
compiled `verify_eos.c` with gcc and ran it for four endmembers -- 3 of
type EOS_BERMAN (forsterite, fayalite-as-COMPONENT, diopside) and 1 of
type EOS_VINET (fayalite-as-PHASE, exercising the Newton-Raphson
compression solve) -- across three (T, P) points spanning ~1 bar to 5 GPa
and 1000-2000 K, using parameters read directly out of
`MELTS_Parameters/sol_struct_data.json` (the same source `load_solids()`
reads) rather than hand-transcribed, to eliminate transcription error as a
variable.

The expected values below are that C harness's raw stdout, unedited. The
observed max relative difference between the two across all
endmembers x (T,P) x quantities was ~3e-10, consistent with %.10g print
rounding in the C harness rather than any actual algorithmic discrepancy.
This test freezes that cross-check into something CI can run without
needing a C compiler at test time; to regenerate, rebuild and rerun
verify_eos.c and update EXPECTED below.

    gcc -O0 -std=c99 -o verify_eos verify_eos.c -lm && ./verify_eos
"""
from __future__ import annotations

import numpy as np
import pytest

from ..compute import compute
from ..params import MELTSSolidParams, load_solids

# label -> (type, index used to build the (T,P) diagonal below)
_ENDMEMBERS = [
    ("forsterite", "COMPONENT"),
    ("fayalite", "COMPONENT"),   # EOS_BERMAN fayalite
    ("fayalite", "PHASE"),       # EOS_VINET fayalite
    ("diopside", "COMPONENT"),
]
_NAMES = ["forsterite", "fayalite_berman", "fayalite_vinet", "diopside"]

_T = np.array([1000.0, 1500.0, 2000.0])
_P = np.array([1.0, 10000.0, 50000.0])

# label, T, P -> (G, H, S, Cp, V, dVdT, dVdP), verbatim ./verify_eos stdout.
EXPECTED = {
    ("forsterite", 1000.0, 1.0):        (-2341132.873, -2064986.562, 276.1463113, 175.2370972, 4.475347831, 1.829590384e-04, -3.453506000e-06),
    ("forsterite", 1500.0, 10000.0):    (-2453034.692, -1931880.302, 347.4362601, 185.7714158, 4.542559764, 2.216562062e-04, -3.335548477e-06),
    ("forsterite", 2000.0, 50000.0):    (-2456949.155, -1674078.807, 391.4351740, 186.1357628, 4.539077766, 2.603533740e-04, -2.863671197e-06),
    ("fayalite_berman", 1000.0, 1.0):     (-1709908.221, -1360599.405, 349.3088159, 187.9518401, 4.734390485, 1.745643730e-04, -3.379900000e-06),
    ("fayalite_berman", 1500.0, 10000.0): (-1857205.041, -1218554.793, 425.7668318, 198.1100062, 4.797077093, 2.113645390e-04, -3.379900000e-06),
    ("fayalite_berman", 2000.0, 50000.0): (-1891413.549, -943880.1698, 473.7666895, 198.5330147, 4.776763404, 2.481647050e-04, -3.379900000e-06),
    ("fayalite_vinet", 1000.0, 1.0):      (-1709908.221, -1360599.405, 349.3088159, 187.9518401, 4.732200720, 1.557993596e-04, -4.134896523e-06),
    ("fayalite_vinet", 1500.0, 10000.0):  (-1857426.714, -1218134.184, 426.1950202, 198.6509893, 4.769662432, 1.636640534e-04, -4.343624564e-06),
    ("fayalite_vinet", 2000.0, 50000.0):  (-1895013.645, -939525.4766, 477.7440843, 202.1044483, 4.682886778, 1.460622680e-04, -3.876475268e-06),
    ("diopside", 1000.0, 1.0):          (-3446723.120, -3044454.813, 402.2683064, 248.4144458, 6.776235213, 2.612067987e-04, -5.772640000e-06),
    ("diopside", 1500.0, 10000.0):      (-3606331.949, -2852523.119, 502.5392203, 259.4100656, 6.863997864, 3.162070827e-04, -5.546655801e-06),
    ("diopside", 2000.0, 50000.0):      (-3600084.682, -2473575.619, 563.2545313, 256.8472382, 6.832065789, 3.712073667e-04, -4.642628601e-06),
}


def _build_test_params() -> MELTSSolidParams:
    """The 4 endmembers above, resolved to specific (label, type) rows so
    the EOS_BERMAN vs EOS_VINET fayalite ambiguity is explicit."""
    all_solids = load_solids(prefer_type=None)
    idx = []
    for label, typ in _ENDMEMBERS:
        matches = [
            i for i, (l, t) in enumerate(zip(all_solids.labels, all_solids.types))
            if l == label and t == typ
        ]
        assert matches, f"no ({label}, {typ}) row in meltsSolids"
        idx.append(matches[0])
    idx = np.array(idx, dtype=np.intp)
    return MELTSSolidParams(
        labels=_NAMES,
        types=[all_solids.types[i] for i in idx],
        formulas=[all_solids.formulas[i] for i in idx],
        h=all_solids.h[idx], s=all_solids.s[idx], v0=all_solids.v0[idx],
        cp_type=all_solids.cp_type[idx], cp_coeffs=all_solids.cp_coeffs[idx],
        eos_type=all_solids.eos_type[idx], eos_coeffs=all_solids.eos_coeffs[idx],
        label_index={n: i for i, n in enumerate(_NAMES)},
    )


@pytest.mark.parametrize("name", _NAMES)
def test_matches_c_harness(name):
    params = _build_test_params()
    result = compute(_T, _P, params)
    i = params.label_index[name]

    for b in range(len(_T)):
        t, p = float(_T[b]), float(_P[b])
        exp_G, exp_H, exp_S, exp_Cp, exp_V, exp_dVdT, exp_dVdP = EXPECTED[(name, t, p)]

        got = dict(
            G=result["G"][b, i], H=result["H"][b, i], S=result["S"][b, i],
            Cp=result["Cp"][b, i], V=result["V"][b, i],
            dVdT=result["dVdT"][b, i], dVdP=result["dVdP"][b, i],
        )
        exp = dict(G=exp_G, H=exp_H, S=exp_S, Cp=exp_Cp, V=exp_V, dVdT=exp_dVdT, dVdP=exp_dVdP)

        for key, exp_val in exp.items():
            np.testing.assert_allclose(
                got[key], exp_val, rtol=1e-6, atol=1e-6,
                err_msg=f"{name} {key} at T={t}, P={p} vs C harness",
            )
