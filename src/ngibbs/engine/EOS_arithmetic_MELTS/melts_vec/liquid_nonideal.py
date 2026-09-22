"""
MELTS's non-ideal liquid mixing correction, translated verbatim from
`sources/liquid_v34.c`'s `gmixLiq_v34`/`hmixLiq_v34`/`smixLiq_v34` (the
active liquid mixing model for both `MODE__MELTS` and `MODE_pMELTS` --
confirmed by direct inspection of `liquid.c`'s own `gmixLiq()` dispatcher,
which routes both modes to `gmixLiq_v34` unconditionally).

Root cause this closes
-----------------------
`liquid_eos.py`'s `compute_liquid_bulk()` implemented IDEAL (Raoult's-law)
mixing only -- flagged prominently in its own docstring as a known,
unimplemented gap (`§4`/`§5`/`§9` of the project plan doc) and, per the raw-
MELTS-output benchmark (`§13`), likely the single largest remaining
precision gap in the whole translation. This module implements the
missing non-ideal excess term.

What the real model actually adds, and why it collapses to something simple
-----------------------------------------------------------------------------
`gmixLiq_v34`'s `mask & FIRST` branch computes (`x[]` = 19 component mole
fractions, `r[]` = the 18 independent ones, `x[0] = 1 - sum(r)`):

    gmix = sum_{i<j} x_i*x_j*(WH(i,j) - t*WS(i,j) + (p-1)*WV(i,j))     (A)
         + sum_i [x_i != 0 ? R*t*x_i*log(x_i) : 0]                     (B)
         + [x_18 != 0 ? R*t*(x_18*log(x_18) + (1-x_18)*log(1-x_18)) : 0]  (C)

(C) is `hmixLiq_v34`/`smixLiq_v34`'s counterpart too: `hmixLiq_v34` sums
ONLY the `WH(i,j)` part of (A) (no `t*WS`/`(p-1)*WV` terms, matching that
`W(i,j)` volume AND entropy interaction parameters are IDENTICALLY ZERO for
every one of the 171 off-diagonal pairs in this calibration -- confirmed
by `MELTS_Parameters/extract_liquid_wij_params.py`'s own assertion, cross-
checked against the connected MAGMA source's `includes/param_struct_data.h`
directly). `cpmixLiq_v34` returns exactly 0 unconditionally, and
`vmixLiq_v34`'s own `mask&FIRST` term is `sum x_i*x_j*WV(i,j)` -- also
exactly 0, matching `liquid_eos.py`'s existing "volume mixing is exactly
ideal" finding. Term (B) already exactly matches `compute_liquid_bulk()`'s
existing `-R*sum(x*log(x))` ideal-entropy term (`R*t*sum(x*log(x)) =
-t*S_config`), so it needs no change.

That leaves exactly two additive corrections on top of the existing ideal
bulk liquid, both closed-form (no Newton solve, no internal ordering
parameter -- unlike every solid-solution mixing model in this package):

  1. A regular-solution (symmetric Margules) EXCESS ENTHALPY, purely from
     term (A) with WS=WV=0: `Gex = Hex = sum_{i<j} x_i*x_j*WH(i,j)`,
     T,P-INDEPENDENT (so it contributes 0 to Sex and Vex, exactly as (A)'s
     `-t*WS`/`(p-1)*WV` sub-terms vanish identically).
  2. An EXTRA ideal-mixing term for H2O specifically (term (C)), on top of
     H2O's own share of the ordinary term (B) -- a known, deliberate
     feature of the Ghiorso & Sack liquid model (water mixes with its own
     additional quasi-binary ideal term, not just as one more component in
     the flat (B) sum). This term is purely entropic (it has the form
     `R*t*f(x)`, i.e. `-t*(-R*f(x))`, with f(x) itself T,P-independent),
     so it contributes 0 to Hex and a genuine `-R*f(x)` to Sex.

Where the W(i,j) numbers come from
------------------------------------
`sources/liquid_v34.c` as uploaded to this project (dated 2006, an
xMELTS/MELTS-5.x-era snapshot) references two mode-specific arrays,
`meltsModelParameters`/`pMeltsModelParameters`, inside its own `WH(i,j)`
macro. The CONNECTED, current MAGMA source's `includes/param_struct_data.h`
(dated 2010, and the same snapshot every other W(i,j)/ThermoRef/EOS table
in this package is sourced from) has unified this into a single
`originalModelParameters[]` table, matching the single `ModelParameters
*modelParameters` pointer `includes/silmin.h` declares and `liquid.c`'s own
(newer) `gmixLiq()` wrapper uses to dispatch BOTH `MODE__MELTS` and
`MODE_pMELTS` to `gmixLiq_v34` -- i.e. the connected, current source treats
MELTS and pMELTS as sharing identical liquid interaction parameters. This
module therefore uses the FORMULAS (A)/(B)/(C) verbatim from the uploaded
`liquid_v34.c` (see `tests/verify_liquid_nonideal.c`'s byte-for-byte
`sed`-extracted function bodies) but sources the numeric W(i,j) VALUES from
`MELTS_Parameters/liq_wij_data.json` (extracted from the connected,
current `param_struct_data.h` -- see that extraction script's own
docstring for the full reasoning).

Units: T in K, P in bars (unused here -- WV=0 identically), mole fractions
dimensionless. Returns J/mol (dG, dH) and J/(mol K) (dS); dCp = dV = 0
identically (see derivation above), so callers may skip adding them.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from .constants import Rgas

_DEFAULT_WIJ_JSON = (Path(__file__).resolve().parent.parent
                      / "MELTS_Parameters" / "liq_wij_data.json")


def load_wij_matrix(labels, json_path: str | Path | None = None) -> np.ndarray:
    """Build a symmetric (N, N) WH(i,j) matrix (zero diagonal, J) in the
    given `labels` order from `liq_wij_data.json`. WS/WV are not returned
    -- they are identically zero for every pair in this calibration (see
    module docstring), so `liquid_nonideal_correction` below never needs
    them; a future recalibration with nonzero WS/WV would need this
    function (and the correction below) extended, not silently ignored."""
    path = Path(json_path) if json_path is not None else _DEFAULT_WIJ_JSON
    with open(path) as fh:
        data = json.load(fh)
    pairs = data["pairs"]
    n = len(labels)
    WH = np.zeros((n, n), dtype=np.float64)
    idx = {lbl: i for i, lbl in enumerate(labels)}
    for key, vals in pairs.items():
        a, b = key.split("|")
        if a not in idx or b not in idx:
            continue  # caller selected a subset of components -- skip
        i, j = idx[a], idx[b]
        WH[i, j] = vals["WH"]
        WH[j, i] = vals["WH"]
    return WH


def liquid_nonideal_correction(X, T, WH_matrix, h2o_index: int) -> dict:
    """Additive (dG, dH, dS) corrections to `compute_liquid_bulk()`'s
    existing ideal-mixing G/H/S -- dCp and dV are identically 0 (see module
    docstring) and are not returned; callers add nothing to Cp/V/dVdT/dVdP.

    Parameters
    ----------
    X : (B, N) mole fractions (same convention as `compute_liquid_bulk`).
    T : (B,) or (B, 1) temperature, K.
    WH_matrix : (N, N) symmetric, zero-diagonal, from `load_wij_matrix`.
    h2o_index : index of the H2O component in X's column order (18 for the
        standard 19-component `meltsLiquid` order; `liquid_v34.c`'s own
        `x[NA-1]` is always the LAST component in both the real source and
        this package's column order, confirmed identical -- see
        `MELTS_Parameters/liq_wij_data.json`'s docstring).
    """
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    if T.ndim == 1:
        T = T[:, None]

    # (A) with WS=WV=0: Gex = Hex = sum_{i<j} x_i*x_j*WH(i,j)
    #    = 0.5 * X @ WH_matrix @ X^T (off-diagonal only, since WH_matrix's
    #    diagonal is exactly 0 -- matches gmixLiq_v34's own `for (j=i+1;...)`
    #    strict upper-triangle loop bit-for-bit, just vectorized).
    gex = 0.5 * np.einsum("bi,ij,bj->b", X, WH_matrix, X)
    dH = gex
    dG_excess = gex  # T,P-independent -- Sex=Vex=0

    # (C): extra H2O ideal term, purely entropic.
    xw = X[:, h2o_index]
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(xw > 0.0, xw*np.log(np.where(xw > 0.0, xw, 1.0)), 0.0) + \
               np.where((1.0 - xw) > 0.0, (1.0 - xw)*np.log(np.where((1.0 - xw) > 0.0, 1.0 - xw, 1.0)), 0.0)
    # gibbs.c/gmixLiq_v34 gates the WHOLE (C) term on x[NA-1] != 0 (not on
    # 1-x[NA-1] != 0 separately) -- matched here for bit-for-bit fidelity,
    # though the (1-xw) log(1-xw) sub-term is individually well-defined
    # (-> 0) at xw=0 regardless.
    term = np.where(xw != 0.0, term, 0.0)
    dS_h2o = -Rgas*term
    dG_h2o = Rgas*T[:, 0]*term

    dG = dG_excess + dG_h2o
    dS = dS_h2o  # dS_excess = 0
    return dict(dG=dG, dH=dH, dS=dS)
