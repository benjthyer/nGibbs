"""
Clinopyroxene (quadrilateral + non-quadrilateral) solid-solution mixing
model -- vectorized translation of `sources/clinopyroxene.c`'s
`gmixCpx`/`actCpx`/`order`, verified against the file directly.

Endmembers (NA=7, matching `testCpx`'s NAMES/FORMULAS arrays and the
`purePyx` component-index order 0..6):
    0 diopside            CaMgSi2O6
    1 clinoenstatite       Mg2Si2O6
    2 hedenbergite         CaFeSi2O6
    3 alumino-buffonite    CaTi0.5Mg0.5AlSiO6
    4 buffonite            CaTi0.5Mg0.5FeSiO6
    5 essenite             CaFeAlSiO6
    6 jadeite              NaAlSi2O6

Independent composition variables r[0..5] (NR=6, the C source's "X2".."X7"
minus 1, see `ENDMEMBERS`'s mole-fraction weights below) and two internal
ordering parameters s[0],s[1] (NS=2, "S1"/"S2" -- M1-site Al/Fe3+ ordering
and M1/M2-site Fe2+/Mg ordering respectively).

Structural state ("clino" vs "ortho"/"pigeonite"): `clinopyroxene.c`
compiles with `ISCLINO` undefined ("If undefined, structural state is
always monoclinic" -- see the file's own top-of-file comment), which
makes `isClino()` unconditionally `return TRUE` (its `#ifdef ISCLINO`
branch, containing the real clino-vs-ortho G-comparison logic, is dead
code for the actual compiled behavior). Every one of this file's many
`(clino) ? cFOO : oFOO` Taylor-coefficient ternaries therefore ALWAYS
selects the `c`-prefixed (clinopyroxene) constant, and every vertex term
`H1..H7`/`S1..S7`/`V1..V7` (all of the form `(clino) ? 0.0 : DFOO`)
collapses to exactly zero -- while the "sliding pigeonite standard
state" terms `pH1..pH7`/`pS1..pS7`/`pV1..pV7` (of the form
`(clino) ? DcTOpFOO : 0.0`) and the "delta" terms `dH027`/`dHEX`/`dHX`/
etc. (of the form `(clino) ? pFOO-cFOO : 0.0`) are ALWAYS active. This
module hardcodes that reduction throughout rather than carrying the
`clino` ternary machinery, which is why there is no `oXXX`-branch code
here at all (confirmed dead for this build).

Design notes carried over from `solution_model.py` / `olivine.py`:
  - a batched Newton solve with a NUMERICALLY estimated Jacobian (not
    the analytic `d2gds2` Hessian `clinopyroxene.c` itself uses) finds
    the equilibrium ordering parameters s* at each (r, T, P);
  - G, H, S and the Darken activities are read off analytically at s*
    (envelope theorem); Cp/dV(T,P) are obtained by finite-differencing
    with s* re-solved at each stencil point.

Additional structural feature NOT present in feldspar.py/olivine.py:
essenite (component 5) has its OWN one-parameter internal ordering
variable (Fe3+/Al on the tetrahedral/M1 sites), entirely separate from
the main two-parameter (s0, s1) solution ordering, solved once per
(T, P) via `pureOrder`/`purePyx` in the C source (its equilibrium
value doesn't depend on the bulk composition r at all -- only T, P).
This is reproduced here as `_pure_es_order`.

Because every Taylor-expansion coefficient in this model (`HX2`, `SX2X3`,
`VX2X7X7`, ...) turns out to be a plain compile-time CONSTANT -- unlike
olivine.c, T and P enter this model only at the final
`G = H - T*S + (P-1)*V`-style assembly, exactly as in feldspar.c -- this
module represents the full ~90-term H/S/V Taylor expansion as three
flat {monomial_name: float} dictionaries built once at import time, and
uses one small generic polynomial-evaluation/differentiation engine
(`_poly_value`/`_poly_dr`/`_poly_ds`) shared by the total-G assembly,
the pure-endmember G/H/S/V macros (DI_G, EN_G, ..., JD_G -- each of
which is confirmed, by direct inspection, to equal exactly the SAME
G/H/S/V polynomial evaluated at that endmember's own vertex (r, s), see
`_pure_vertex_value` below) and the dG/dr, dG/ds derivatives. This
means only ONE thing needs to be gotten right per term (the numeric
coefficient, verified against a C harness) rather than three consistent
hand-transcriptions of value/gradient/vertex formulas -- deliberately
lower transcription-risk for a model this size, per the same reasoning
`solution_model.py` gives for using a numerical Jacobian.

The one place this module does NOT use the generic engine is the
ideal-mixing configurational-entropy contribution to dG/dr and dG/ds
(the `R*t*(log(...))` leading terms in `DGDR0`.."DGDR5"/`DGDS0`/`DGDS1`)
-- those come from differentiating `SIC` (a bespoke, non-polynomial
function of the site fractions), and are transcribed verbatim.
"""
from __future__ import annotations
import re
import numpy as np

from .constants import Rgas
from .solution_model import newton_solve_ordering, newton_solve_ordering as _newton  # noqa: F401

ENDMEMBERS = ["diopside", "clinoenstatite", "hedenbergite",
              "alumino-buffonite", "buffonite", "essenite", "jadeite"]

_DBL_EPS = 2.220446049250313e-16   # C's DBL_EPSILON (float.h), matched exactly so
                                     # near-boundary site-fraction clipping agrees
                                     # bit-for-bit with clinopyroxene.c's order()

# =============================================================================
# Base constants, verbatim from clinopyroxene.c lines 213-368 (only the
# "c"-prefixed / plain / "p"-prefixed / "DcTOp*" constants are needed --
# see module docstring on why the "o"-prefixed ones are dead for this
# build; they are read here ONLY where a "c" value is defined in terms
# of them, per the dependency chain in lines 374-401).
# =============================================================================
_DH1 = 0.5573 * 1000.0 * 4.184
_DS1 = 0.00033 * 1000.0 * 4.184
_DV1 = 0.018 * 4.184
_DH2 = 1.200 * 1000.0 * 4.184
_DS2 = 0.000675 * 1000.0 * 4.184
_DV2 = 0.0148 * 4.184
_DH3 = 0.6991252 * 1000.0 * 4.184
_DS3 = 0.0005926 * 1000.0 * 4.184
_DV3 = 0.00865 * 4.184
_DH4 = -0.49 * 1000.0 * 4.184
_DS4 = -0.000132 * 1000.0 * 4.184

_cV0DI = 6.620
_cV0HD = 6.7894
_oV0HD = 1.64620 * 4.184
_oV0EN = 3.133 * 2.0
_oV0FS = 3.296 * 2.0

_oH027 = -3.30 * 1000.0 * 4.184
_oS027 = -0.00055428 * 1000.0 * 4.184
_pH027 = -1.97547480 * 1000.0 * 4.184
_pS027 = 0.00071134 * 1000.0 * 4.184
_pV027 = -0.00920929 * 4.184

_oHEX = -1.87 * 1000.0 * 4.184
_oVEX = -0.029 * 4.184
_cHEX = -2.2 * 1000.0 * 4.184
_cVEX = 0.01 * 4.184
_pHEX = -0.65 * 1000.0 * 4.184
_pVEX = 0.0 * 4.184

_oHX = -0.45 * 1000.0 * 4.184
_oVX = 0.00675 * 4.184
_cHX = -0.10 * 1000.0 * 4.184
_cVX = 0.0 * 4.184
_pHX = -0.10 * 1000.0 * 4.184
_pVX = 0.0 * 4.184

_cWHFEMG = 1.15 * 1000.0 * 4.184
_cWVFEMG = -0.0035 * 4.184

_cWH12 = 1.68 * 1000.0 * 4.184
_cWV12 = 0.0 * 4.184

_cWHCAMG = 6.72 * 1000.0 * 4.184
_cWVCAMG = -0.009 * 4.184

_cWHCAFE = 4.515 * 1000.0 * 4.184
_cWVCAFE = 0.003 * 4.184

_cDWHCAMG = -0.7 * 1000.0 * 4.184
_cDWVCAMG = 0.008 * 4.184

_cDWHCAFE = -0.5375 * 1000.0 * 4.184
_cDWVCAFE = 0.0025 * 4.184

_W13 = 16.318 * 1000.0
_W14 = 18.82800 * 1000.0
_W15 = 20.920 * 1000.0
_W15P = 16.78900 * 1000.0
_W16 = 0.00000 * 1000.0
_W25 = 3.98176 * 1000.0
_W25P = 7.49005 * 1000.0
_W26 = 0.00000 * 1000.0
_W34 = 16.78900 * 1000.0
_W35 = 12.02812 * 1000.0
_W35P = 47.38600 * 1000.0
_W36 = 34.35062 * 1000.0
_cW37 = 28.65091 * 1000.0
_cW3U7U = 34.90070 * 1000.0
_W45 = 27.14312 * 1000.0
_W45P = 16.318 * 1000.0
_W46 = 40.40662 * 1000.0
_cW47 = 35.38104 * 1000.0
_cW4U7U = 28.54527 * 1000.0
_cW55 = 35.670 * 1000.0
_W56 = 0.00000 * 1000.0
_cW57 = 33.25291 * 1000.0
_cW57U = 15.29737 * 1000.0
_W5P6 = 0.00000 * 1000.0
_cW5P7 = 36.24989 * 1000.0
_cW5P7U = 2.35321 * 1000.0
_cW67 = 0.00000 * 1000.0
_cW67U = 0.00000 * 1000.0

_H23 = -2.71700 * 1000.0
_S23 = 0.0
_H24 = -7.36648 * 1000.0
_S24 = 0.0
_cH55 = 9.48524 * 1000.0
_S55 = 0.0

_DcTOoH3, _DcTOoS3, _DcTOoV3 = 25.11924 * 1000.0, -2.00000, -0.05129
_DcTOoH4, _DcTOoS4, _DcTOoV4 = 25.11924 * 1000.0, -2.00000, -0.05129
_DcTOoH5, _DcTOoS5, _DcTOoV5 = 25.11924 * 1000.0, -2.00000, -0.05129
_DcTOoH6, _DcTOoS6, _DcTOoV6 = 25.11924 * 1000.0, -2.00000, -0.05129

_DcTOpH1 = -0.9 * 1000.0 * 4.184
_DcTOpS1 = -0.0006920 * 1000.0 * 4.184
_DcTOpV1 = 0.0 * 4.184
_DcTOpH2 = -1.88 * 1000.0 * 4.184
_DcTOpS2 = -0.00156401 * 1000.0 * 4.184
_DcTOpV2 = 0.0 * 4.184
_DcTOpH3, _DcTOpS3, _DcTOpV3 = 21.11924 * 1000.0, -2.00000, -0.05129
_DcTOpH4, _DcTOpS4, _DcTOpV4 = 21.11924 * 1000.0, -2.00000, -0.05129
_DcTOpH5, _DcTOpS5, _DcTOpV5 = 21.11924 * 1000.0, -2.00000, -0.05129
_DcTOpHj, _DcTOpSj, _DcTOpVj = 21.11924 * 1000.0, -2.00000, -0.05129
_DcTOpH7 = 0.0 * 1000.0 * 4.184
_DcTOpS7 = 0.0 * 1000.0 * 4.184
_DcTOpV7 = 0.0 * 4.184

# --- Dependent parameters (clinopyroxene.c lines 374-486), clino=TRUE ------
_oV0DI = _cV0DI + _DV1
_cV0EN = _oV0EN + _DV2
_cV0FS = _oV0FS + _DV3
_DV4 = _cV0HD - _oV0HD
_oV027 = _oV0FS - _oV0EN + 2.0 * _oV0DI - 2.0 * _oV0HD
_cV027 = _cV0FS - _cV0EN + 2.0 * _cV0DI - 2.0 * _cV0HD
_cH027 = _oH027 + _DH3 - _DH2 - 2.0 * _DH1 - 2.0 * _DH4
_cS027 = _oS027 + _DS3 - _DS2 - 2.0 * _DS1 - 2.0 * _DS4

_H027, _S027, _V027 = _cH027, _cS027, _cV027           # clino=TRUE
_dH027 = _pH027 - _cH027
_dS027 = _pS027 - _cS027
_dV027 = _pV027 - _cV027

_HEX, _VEX = _cHEX, _cVEX
_dHEX = _pHEX - _cHEX
_dVEX = _pVEX - _cVEX

_HX, _VX = _cHX, _cVX
_dHX = _pHX - _cHX
_dVX = _pVX - _cVX

_WHFEMG, _WVFEMG = _cWHFEMG, _cWVFEMG
_WH12, _WV12 = _cWH12, _cWV12
_WHCAMG, _WVCAMG = _cWHCAMG, _cWVCAMG
_DWHCAMG, _DWVCAMG = _cDWHCAMG, _cDWVCAMG
_WHCAFE, _WVCAFE = _cWHCAFE, _cWVCAFE
_DWHCAFE, _DWVCAFE = _cDWHCAFE, _cDWVCAFE

_W37, _W3U7U = _cW37, _cW3U7U
_W47, _W4U7U = _cW47, _cW4U7U
_W55 = _cW55
_W57, _W57U = _cW57, _cW57U
_W5P7, _W5P7U = _cW5P7, _cW5P7U
_W67, _W67U = _cW67, _cW67U
_H55 = _cH55

# --- Vertices (clinopyroxene.c lines 492-563), clino=TRUE ------------------
# H1..H7 = S1..S7 = V1..V7 = 0.0 always (all "(clino)?0.0:DFOO" ternaries).
_pH1, _pS1, _pV1 = _DcTOpH1, _DcTOpS1, _DcTOpV1
_pH2, _pS2, _pV2 = _DcTOpH2, _DcTOpS2, _DcTOpV2
_pH3, _pS3, _pV3 = _DcTOpH3, _DcTOpS3, _DcTOpV3
_pH4, _pS4, _pV4 = _DcTOpH4, _DcTOpS4, _DcTOpV4
_pH5, _pS5, _pV5 = _DcTOpH5, _DcTOpS5, _DcTOpV5
_pHj, _pSj, _pVj = _DcTOpHj, _DcTOpSj, _DcTOpVj
_pH7, _pS7, _pV7 = _DcTOpH7, _DcTOpS7, _DcTOpV7

# =============================================================================
# Taylor-expansion coefficients (clinopyroxene.c lines 566-898), verbatim
# with H1..H7/S1..S7/V1..V7 -> 0.0 substituted throughout.
# =============================================================================
H0 = 0.0
HX2 = _WH12
HX3 = _W13
HX4 = _W14
HX5 = 0.5 * (_W15 + _W15P + _H55)
HX6 = (-0.5 * _W13 + 0.5 * _W14 + 0.25 * _W15 - 0.75 * _W15P + _W16 - 0.75 * _H55)
HX7 = (_pH1 + 0.5 * (_WHCAFE + _WHCAMG - _WH12) + 0.25 * (_HEX + _HX + _H027)
       + 0.5 * _DWHCAMG - 1.5 * _DWHCAFE)

HS1 = 0.5 * (_W15 - _W15P - _H55)
HS2 = (0.5 * (_WHCAFE - _WHCAMG - _WH12) + 0.25 * (_HEX + _HX + _H027)
       - 0.5 * _DWHCAMG - 1.5 * _DWHCAFE)

HX2X2 = -_WH12
HX2X3 = 2.0 * _H23 - 0.5 * _WH12
HX2X4 = 2.0 * _H24 - 0.5 * _WH12
HX2X5 = 0.5 * (_W25 + _W25P - _W15 - _W15P) - _WH12
HX2X6 = (_W26 - _W16 + 0.25 * (_W25 - _W15) - 0.75 * (_W25P - _W15P)
         + _H24 - _H23 - 0.5 * _WH12)
HX2X7 = (_pH2 - _pH1 + _WH12 + 0.5 * (_H027 - _HEX) - 2.0 * _DWHCAMG
         + 2.0 * _DWHCAFE)
HX2S1 = 0.5 * (_W25 - _W25P - _W15 + _W15P)
HX2S2 = _WH12 - 0.5 * _HX + 2.0 * _DWHCAMG + 2.0 * _DWHCAFE

HX3X3 = -_W13
HX3X4 = _W34 - _W13 - _W14
HX3X5 = 0.5 * (_W35 + _W35P - _W15 - _W15P) - _W13
HX3X6 = (_W36 - _W16 + 0.25 * (_W35 - _W15) + 0.5 * (_W34 - _W14)
         - 0.75 * (_W35P - _W15P))
HX3X7 = (0.125 * (_H027 - _HEX - _HX) + 0.5 * (_W37 + _W3U7U - _WHCAFE - _WHCAMG)
         - 1.5 * _H23 - _W13 + 0.25 * _WH12 - _DWHCAMG + _DWHCAFE + _pH3 - _pH1)
HX3S1 = 0.5 * (_W35 - _W35P - _W15 + _W15P)
HX3S2 = (0.125 * (_H027 - _HEX - _HX) + 0.5 * (_W3U7U - _W37 + _WHCAMG - _WHCAFE)
          - 1.5 * _H23 + 0.25 * _WH12 + _DWHCAMG + _DWHCAFE)

HX4X4 = -_W14
HX4X5 = 0.5 * (_W45 + _W45P - _W15 - _W15P) - _W14
HX4X6 = (_W46 - _W16 - _W14 + 0.25 * (_W45 - _W15) - 0.5 * (_W34 - _W13)
         - 0.75 * (_W45P - _W15P))
HX4X7 = (0.125 * (_H027 - _HEX - _HX) + 0.5 * (_W47 + _W4U7U - _WHCAFE - _WHCAMG)
         - 1.5 * _H24 - _W14 + 0.25 * _WH12 - _DWHCAMG + _DWHCAFE + _pH4 - _pH1)
HX4S1 = 0.5 * (_W45 - _W45P - _W15 + _W15P)
HX4S2 = (0.125 * (_H027 - _HEX - _HX) + 0.5 * (_W4U7U - _W47 + _WHCAMG - _WHCAFE)
          - 1.5 * _H24 + 0.25 * _WH12 + _DWHCAMG + _DWHCAFE)

HX5X5 = 0.25 * _W55 - 0.5 * (_W15 + _W15P)
HX5X6 = (0.5 * (_W56 + _W5P6 - _W15 + _W15P - 2.0 * _W16) - 0.25 * _W55
         + 0.25 * (_W45 + _W45P - 2.0 * _W14) - 0.25 * (_W35 + _W35P - 2.0 * _W13))
HX5X7 = (0.25 * (_H027 - _HEX - _HX) + 0.25 * (_W57U + _W5P7U + _W57 + _W5P7)
         - 0.5 * (_WHCAFE + _WHCAMG + _W25 + _W25P - _WH12) - _DWHCAMG
         + 2.0 * _DWHCAFE + _pH5 - _pH1)
HX5S1 = -0.5 * (_W15 - _W15P)
HX5S2 = (0.25 * (_H027 - _HEX - _HX) + 0.25 * (_W57U + _W5P7U - _W57 - _W5P7)
          + 0.5 * (_W15 + _W15P - _W25 - _W25P + _WHCAMG - _WHCAFE + _WH12)
          + _DWHCAMG + 2.0 * _DWHCAFE)

HX6X6 = (0.25 * _W56 - 0.75 * _W5P6 + 0.5 * (_W46 - _W36 - _W16)
         - 0.1875 * _W55 + 0.125 * (_W45 - _W35 - _W15)
         - 0.375 * (_W45P - _W35P - _W15P) - 0.25 * (_W34 + _W14 - _W13))
HX6X7 = (0.5 * (_W67 + _W67U) - 0.375 * (_W5P7 + _W5P7U)
         + 0.125 * (_W57 + _W57U) + 0.25 * (_W47 + _W4U7U)
         - 0.25 * (_W37 + _W3U7U) - _W26 - 0.25 * _W25 + 0.75 * _W25P
         - 0.5 * (_W14 - _W13) + 0.25 * _WH12 - 0.25 * (_WHCAFE + _WHCAMG)
         - 0.125 * (_HEX + _HX - _H027) - 0.75 * (_H24 - _H23)
         - 0.50 * _DWHCAMG + _DWHCAFE + _pHj - 0.5 * (_pH1 + _pH3 - _pH4 + _pH5))
HX6S1 = (0.5 * (_W56 - _W5P6) + 0.25 * (_W45 - _W45P) - 0.25 * (_W35 - _W35P)
          - 0.25 * (_W15 - _W15P) - 0.5 * _W55)
HX6S2 = (0.5 * (_W67U - _W67) + 0.125 * (_W57U - _W57)
          + 0.375 * (_W5P7 - _W5P7U) + 0.25 * (_W4U7U - _W47)
          + 0.25 * (_W37 - _W3U7U) + _W16 - _W26 + 0.75 * (_W25P - _W15P)
          + 0.25 * (_W15 - _W25) + 0.25 * _WH12 - 0.25 * (_WHCAFE - _WHCAMG)
          - 0.125 * (_HEX + _HX - _H027) - 0.75 * (_H24 - _H23)
          + 0.50 * _DWHCAMG + _DWHCAFE)

HX7X7 = (0.25 * (_HEX - _H027) - 0.5 * (_WHCAFE + _WHCAMG)
         + 0.25 * (_WHFEMG - _WH12) + 0.75 * _DWHCAMG + 0.25 * _DWHCAFE
         + _pH7 - _pH1 + 0.25 * (_dH027 + _dHEX + _dHX))
HX7S1 = 0.25 * (_W57U + _W57 - _W5P7U - _W5P7) - 0.5 * (_W25 - _W25P)
HX7S2 = (0.5 * (_WHCAMG - _WHCAFE - _WH12) + 0.25 * (_HEX + _HX - _H027)
          + 1.5 * _DWHCAMG - 1.5 * _DWHCAFE + 0.25 * (_dH027 + _dHEX + _dHX))

HS1S1 = -0.25 * _W55
HS1S2 = (0.25 * (_W57U + _W5P7 - _W5P7U - _W57)
         + 0.5 * (_W25P - _W25 + _W15 - _W15P))
HS2S2 = 0.25 * (_HX - _WHFEMG - _WH12) - 2.25 * _DWHCAMG - 1.75 * _DWHCAFE

HX2X2X7 = -0.5 * _DWHCAMG + 0.5 * _DWHCAFE
HX2X2S2 = 0.5 * _DWHCAMG - 0.5 * _DWHCAFE
HX2X3X7 = _DWHCAMG - _DWHCAFE
HX2X3S2 = -_DWHCAMG - _DWHCAFE
HX2X4X7 = _DWHCAMG - _DWHCAFE
HX2X4S2 = -_DWHCAMG - _DWHCAFE
HX2X5X7 = _DWHCAMG - _DWHCAFE
HX2X5S2 = -_DWHCAMG - _DWHCAFE
HX2X6X7 = 0.5 * _DWHCAMG - 0.5 * _DWHCAFE
HX2X6S2 = -0.5 * _DWHCAMG - 0.5 * _DWHCAFE
HX2X7X7 = 2.75 * _DWHCAMG - 2.75 * _DWHCAFE + 0.5 * (_dH027 - _dHEX)
HX2X7S2 = -2.5 * _DWHCAMG - 1.5 * _DWHCAFE - 0.5 * _dHX
HX2S2S2 = -0.25 * _DWHCAMG + 0.25 * _DWHCAFE
HX3X3X7 = 0.5 * _DWHCAMG
HX3X3S2 = -0.5 * _DWHCAMG
HX3X4X7 = _DWHCAMG + 0.25 * _DWHCAFE
HX3X4S2 = -_DWHCAMG + 0.25 * _DWHCAFE
HX3X5X7 = _DWHCAMG - 0.25 * _DWHCAFE
HX3X5S2 = -_DWHCAMG - 0.25 * _DWHCAFE
HX3X6X7 = 0.5 * _DWHCAMG
HX3X6S2 = -0.5 * _DWHCAMG
HX3X7X7 = 0.25 * _DWHCAMG + 0.625 * _DWHCAFE + 0.125 * (_dH027 - _dHEX - _dHX)
HX3X7S2 = -1.5 * _DWHCAMG + 1.5 * _DWHCAFE + 0.125 * (_dH027 - _dHEX - _dHX)
HX3S2S2 = 1.25 * _DWHCAMG + 0.875 * _DWHCAFE
HX4X4X7 = 0.5 * _DWHCAMG
HX4X4S2 = -0.5 * _DWHCAMG
HX4X5X7 = _DWHCAMG - 0.25 * _DWHCAFE
HX4X5S2 = -_DWHCAMG - 0.25 * _DWHCAFE
HX4X6X7 = 0.5 * _DWHCAMG - 0.25 * _DWHCAFE
HX4X6S2 = -0.5 * _DWHCAMG - 0.25 * _DWHCAFE
HX4X7X7 = 0.25 * _DWHCAMG + 0.625 * _DWHCAFE + 0.125 * (_dH027 - _dHEX - _dHX)
HX4X7S2 = -1.5 * _DWHCAMG + 1.5 * _DWHCAFE + 0.125 * (_dH027 - _dHEX - _dHX)
HX4S2S2 = 1.25 * _DWHCAMG + 0.875 * _DWHCAFE
HX5X5X7 = 0.5 * _DWHCAMG - 0.5 * _DWHCAFE
HX5X5S2 = -0.5 * _DWHCAMG - 0.5 * _DWHCAFE
HX5X6X7 = 0.5 * _DWHCAMG - 0.5 * _DWHCAFE
HX5X6S2 = -0.5 * _DWHCAMG - 0.5 * _DWHCAFE
HX5X7X7 = 0.25 * _DWHCAMG - 0.25 * _DWHCAFE + 0.25 * (_dH027 - _dHEX - _dHX)
HX5X7S2 = -1.5 * _DWHCAMG + 0.5 * _DWHCAFE + 0.25 * (_dH027 - _dHEX - _dHX)
HX5S2S2 = 1.25 * _DWHCAMG + 0.75 * _DWHCAFE
HX6X6X7 = 0.125 * _DWHCAMG - 0.1875 * _DWHCAFE
HX6X6S2 = -0.125 * _DWHCAMG - 0.1875 * _DWHCAFE
HX6X7X7 = 0.125 * _DWHCAMG - 0.125 * _DWHCAFE + 0.125 * (_dH027 - _dHEX - _dHX)
HX6X7S2 = -0.75 * _DWHCAMG + 0.25 * _DWHCAFE + 0.125 * (_dH027 - _dHEX - _dHX)
HX6S2S2 = 0.625 * _DWHCAMG + 0.375 * _DWHCAFE
HX7X7X7 = -1.5 * _DWHCAMG + 1.5 * _DWHCAFE + 0.25 * (_dHEX - _dH027)
HX7X7S2 = -_DWHCAMG + 3.0 * _DWHCAFE + 0.25 * (_dHEX + _dHX - _dH027)
HX7S2S2 = 2.5 * _DWHCAMG + 1.5 * _DWHCAFE + 0.25 * _dHX

# --- S coefficients (lines 709-779). RHYOLITE_ADJUSTMENTS build: SS2=0.0. --
S0 = 0.0
SX2 = 0.0
SX3 = 0.0
SX4 = 0.0
SX5 = 0.5 * _S55
SX6 = -0.75 * _S55
SX7 = _pS1 + 0.25 * _S027

SS1 = -0.5 * _S55
SS2 = 0.0    # RHYOLITE_ADJUSTMENTS build (confirmed active this project)

SX2X2 = 0.0
SX2X3 = 2.0 * _S23
SX2X4 = 2.0 * _S24
SX2X5 = 0.0
SX2X6 = _S24 - _S23
SX2X7 = _pS2 - _pS1 + 0.5 * _S027
SX2S1 = 0.0
SX2S2 = 0.0

SX3X3 = 0.0
SX3X4 = 0.0
SX3X5 = 0.0
SX3X6 = 0.0
SX3X7 = 0.125 * _S027 - 1.5 * _S23 + _pS3 - _pS1
SX3S1 = 0.0
SX3S2 = 0.125 * _S027 - 1.5 * _S23

SX4X4 = 0.0
SX4X5 = 0.0
SX4X6 = 0.0
SX4X7 = 0.125 * _S027 - 1.5 * _S24 + _pS4 - _pS1
SX4S1 = 0.0
SX4S2 = 0.125 * _S027 - 1.5 * _S24

SX5X5 = 0.0
SX5X6 = 0.0
SX5X7 = 0.25 * _S027 + _pS5 - _pS1
SX5S1 = 0.0
SX5S2 = 0.25 * _S027

SX6X6 = 0.0
SX6X7 = 0.125 * _S027 - 0.75 * (_S24 - _S23) + _pSj - 0.5 * (_pS1 + _pS3 - _pS4 + _pS5)
SX6S1 = 0.0
SX6S2 = 0.125 * _S027 - 0.75 * (_S24 - _S23)

SX7X7 = -0.25 * _S027 + _pS7 - _pS1 + 0.25 * _dS027
SX7S1 = 0.0
SX7S2 = -0.25 * _S027 + 0.25 * _dS027

SS1S1 = 0.0
SS1S2 = 0.0
SS2S2 = 0.0

SX2X7X7 = 0.5 * _dS027
SX3X7X7 = 0.125 * _dS027
SX3X7S2 = 0.125 * _dS027
SX4X7X7 = 0.125 * _dS027
SX4X7S2 = 0.125 * _dS027
SX5X7X7 = 0.25 * _dS027
SX5X7S2 = 0.25 * _dS027
SX6X7X7 = 0.125 * _dS027
SX6X7S2 = 0.125 * _dS027
SX7X7X7 = -0.25 * _dS027
SX7X7S2 = -0.25 * _dS027

# --- V coefficients (lines 781-898). ---------------------------------------
V0 = 0.0
VX2 = _WV12
VX3 = 0.0
VX4 = 0.0
VX5 = 0.0
VX6 = 0.0
VX7 = (_pV1 + 0.5 * (_WVCAFE + _WVCAMG - _WV12) + 0.25 * (_VEX + _VX + _V027)
       + 0.5 * _DWVCAMG - 1.5 * _DWVCAFE)

VS1 = 0.0
VS2 = (0.5 * (_WVCAFE - _WVCAMG - _WV12) + 0.25 * (_VEX + _VX + _V027)
       - 0.5 * _DWVCAMG - 1.5 * _DWVCAFE)

VX2X2 = -_WV12
VX2X3 = -0.5 * _WV12
VX2X4 = -0.5 * _WV12
VX2X5 = -_WV12
VX2X6 = -0.5 * _WV12
VX2X7 = _pV2 - _pV1 + _WV12 + 0.5 * (_V027 - _VEX) - 2.0 * _DWVCAMG + 2.0 * _DWVCAFE
VX2S1 = 0.0
VX2S2 = _WV12 - 0.5 * _VX + 2.0 * _DWVCAMG + 2.0 * _DWVCAFE

VX3X3 = 0.0
VX3X4 = 0.0
VX3X5 = 0.0
VX3X6 = 0.0
VX3X7 = (0.125 * (_V027 - _VEX - _VX) - 0.5 * (_WVCAFE + _WVCAMG)
         + 0.25 * _WV12 - _DWVCAMG + _DWVCAFE + _pV3 - _pV1)
VX3S1 = 0.0
VX3S2 = (0.125 * (_V027 - _VEX - _VX) + 0.5 * (_WVCAMG - _WVCAFE)
          + 0.25 * _WV12 + _DWVCAMG + _DWVCAFE)

VX4X4 = 0.0
VX4X5 = 0.0
VX4X6 = 0.0
VX4X7 = (0.125 * (_V027 - _VEX - _VX) - 0.5 * (_WVCAFE + _WVCAMG)
         + 0.25 * _WV12 - _DWVCAMG + _DWVCAFE + _pV4 - _pV1)
VX4S1 = 0.0
VX4S2 = (0.125 * (_V027 - _VEX - _VX) + 0.5 * (_WVCAMG - _WVCAFE)
          + 0.25 * _WV12 + _DWVCAMG + _DWVCAFE)

VX5X5 = 0.0
VX5X6 = 0.0
VX5X7 = (0.25 * (_V027 - _VEX - _VX) - 0.5 * (_WVCAFE + _WVCAMG - _WV12)
         - _DWVCAMG + 2.0 * _DWVCAFE + _pV5 - _pV1)
VX5S1 = 0.0
VX5S2 = (0.25 * (_V027 - _VEX - _VX) + 0.5 * (_WVCAMG - _WVCAFE + _WV12)
          + _DWVCAMG + 2.0 * _DWVCAFE)

VX6X6 = 0.0
VX6X7 = (0.25 * _WV12 - 0.25 * (_WVCAFE + _WVCAMG)
         - 0.125 * (_VEX + _VX - _V027) - 0.5 * _DWVCAMG + _DWVCAFE
         + _pVj - 0.5 * (_pV1 + _pV3 - _pV4 + _pV5))
VX6S1 = 0.0
VX6S2 = (0.25 * _WV12 - 0.25 * (_WVCAFE - _WVCAMG)
          - 0.125 * (_VEX + _VX - _V027) + 0.5 * _DWVCAMG + _DWVCAFE)

VX7X7 = (0.25 * (_VEX - _V027) - 0.5 * (_WVCAFE + _WVCAMG)
         + 0.25 * (_WVFEMG - _WV12) + 0.75 * _DWVCAMG + 0.25 * _DWVCAFE
         + _pV7 - _pV1 + 0.25 * (_dV027 + _dVEX + _dVX))
VX7S1 = 0.0
VX7S2 = (0.5 * (_WVCAMG - _WVCAFE - _WV12) + 0.25 * (_VEX + _VX - _V027)
          + 1.5 * _DWVCAMG - 1.5 * _DWVCAFE + 0.25 * (_dV027 + _dVEX + _dVX))

VS1S1 = 0.0
VS1S2 = 0.0
VS2S2 = 0.25 * (_VX - _WVFEMG - _WV12) - 2.25 * _DWVCAMG - 1.75 * _DWVCAFE

VX2X2X7 = -0.5 * _DWVCAMG + 0.5 * _DWVCAFE
VX2X2S2 = 0.5 * _DWVCAMG - 0.5 * _DWVCAFE
VX2X3X7 = _DWVCAMG - _DWVCAFE
VX2X3S2 = -_DWVCAMG - _DWVCAFE
VX2X4X7 = _DWVCAMG - _DWVCAFE
VX2X4S2 = -_DWVCAMG - _DWVCAFE
VX2X5X7 = _DWVCAMG - _DWVCAFE
VX2X5S2 = -_DWVCAMG - _DWVCAFE
VX2X6X7 = 0.5 * _DWVCAMG - 0.5 * _DWVCAFE
VX2X6S2 = -0.5 * _DWVCAMG - 0.5 * _DWVCAFE
VX2X7X7 = 2.75 * _DWVCAMG - 2.75 * _DWVCAFE + 0.5 * (_dV027 - _dVEX)
VX2X7S2 = -2.5 * _DWVCAMG - 1.5 * _DWVCAFE - 0.5 * _dVX
VX2S2S2 = -0.25 * _DWVCAMG + 0.25 * _DWVCAFE
VX3X3X7 = 0.5 * _DWVCAMG
VX3X3S2 = -0.5 * _DWVCAMG
VX3X4X7 = _DWVCAMG + 0.25 * _DWVCAFE
VX3X4S2 = -_DWVCAMG + 0.25 * _DWVCAFE
VX3X5X7 = _DWVCAMG - 0.25 * _DWVCAFE
VX3X5S2 = -_DWVCAMG - 0.25 * _DWVCAFE
VX3X6X7 = 0.5 * _DWVCAMG
VX3X6S2 = -0.5 * _DWVCAMG
VX3X7X7 = 0.25 * _DWVCAMG + 0.625 * _DWVCAFE + 0.125 * (_dV027 - _dVEX - _dVX)
VX3X7S2 = -1.5 * _DWVCAMG + 1.5 * _DWVCAFE + 0.125 * (_dV027 - _dVEX - _dVX)
VX3S2S2 = 1.25 * _DWVCAMG + 0.875 * _DWVCAFE
VX4X4X7 = 0.5 * _DWVCAMG
VX4X4S2 = -0.5 * _DWVCAMG
VX4X5X7 = _DWVCAMG - 0.25 * _DWVCAFE
VX4X5S2 = -_DWVCAMG - 0.25 * _DWVCAFE
VX4X6X7 = 0.5 * _DWVCAMG - 0.25 * _DWVCAFE
VX4X6S2 = -0.5 * _DWVCAMG - 0.25 * _DWVCAFE
VX4X7X7 = 0.25 * _DWVCAMG + 0.625 * _DWVCAFE + 0.125 * (_dV027 - _dVEX - _dVX)
VX4X7S2 = -1.5 * _DWVCAMG + 1.5 * _DWVCAFE + 0.125 * (_dV027 - _dVEX - _dVX)
VX4S2S2 = 1.25 * _DWVCAMG + 0.875 * _DWVCAFE
VX5X5X7 = 0.5 * _DWVCAMG - 0.5 * _DWVCAFE
VX5X5S2 = -0.5 * _DWVCAMG - 0.5 * _DWVCAFE
VX5X6X7 = 0.5 * _DWVCAMG - 0.5 * _DWVCAFE
VX5X6S2 = -0.5 * _DWVCAMG - 0.5 * _DWVCAFE
VX5X7X7 = 0.25 * _DWVCAMG - 0.25 * _DWVCAFE + 0.25 * (_dV027 - _dVEX - _dVX)
VX5X7S2 = -1.5 * _DWVCAMG + 0.5 * _DWVCAFE + 0.25 * (_dV027 - _dVEX - _dVX)
VX5S2S2 = 1.25 * _DWVCAMG + 0.75 * _DWVCAFE
VX6X6X7 = 0.125 * _DWVCAMG - 0.1875 * _DWVCAFE
VX6X6S2 = -0.125 * _DWVCAMG - 0.1875 * _DWVCAFE
VX6X7X7 = 0.125 * _DWVCAMG - 0.125 * _DWVCAFE + 0.125 * (_dV027 - _dVEX - _dVX)
VX6X7S2 = -0.75 * _DWVCAMG + 0.25 * _DWVCAFE + 0.125 * (_dV027 - _dVEX - _dVX)
VX6S2S2 = 0.625 * _DWVCAMG + 0.375 * _DWVCAFE
VX7X7X7 = -1.5 * _DWVCAMG + 1.5 * _DWVCAFE + 0.25 * (_dVEX - _dV027)
VX7X7S2 = -_DWVCAMG + 3.0 * _DWVCAFE + 0.25 * (_dVEX + _dVX - _dV027)
VX7S2S2 = 2.5 * _DWVCAMG + 1.5 * _DWVCAFE + 0.25 * _dVX

_LOCAL = dict(locals())
H_COEF = {k[1:]: v for k, v in _LOCAL.items() if k.startswith('H') and k[1:2] in ('X', 'S') and not k.startswith('H_')}
S_COEF = {k[1:]: v for k, v in _LOCAL.items() if k.startswith('S') and k[1:2] in ('X', 'S') and not k.startswith('S_') and k != 'SIC'}
V_COEF = {k[1:]: v for k, v in _LOCAL.items() if k.startswith('V') and k[1:2] in ('X', 'S')}
del _LOCAL

# =============================================================================
# Generic polynomial engine over the 8 "variables" X2,X3,X4,X5,X6,X7,S1,S2
# (<-> r[0..5], s[0], s[1]), shared by the total G/H/S/V assembly, the
# pure-endmember vertex evaluations, and the dG/dr, dG/ds polynomial parts.
# =============================================================================
_TOKEN_RE = re.compile(r'[XS]\d')
_TOKEN_INDEX = {'X2': 0, 'X3': 1, 'X4': 2, 'X5': 3, 'X6': 4, 'X7': 5, 'S1': 6, 'S2': 7}


def _var(idx, r, s):
    return r[..., idx] if idx < 6 else s[..., idx - 6]


def _parsed(coef: dict) -> list:
    """[(value, (idx0,[idx1,[idx2]])), ...] for every nonzero coefficient."""
    out = []
    for key, val in coef.items():
        if val == 0.0:
            continue
        idxs = tuple(_TOKEN_INDEX[t] for t in _TOKEN_RE.findall(key))
        out.append((val, idxs))
    return out


def _poly_value(parsed, r, s):
    out = 0.0
    for val, idxs in parsed:
        m = val
        for i in idxs:
            m = m * _var(i, r, s)
        out = out + m
    return out


def _poly_d(parsed, r, s, wrt):
    """d(poly)/d(var wrt), wrt in 0..7."""
    out = 0.0
    for val, idxs in parsed:
        n = idxs.count(wrt)
        if n == 0:
            continue
        rest = list(idxs)
        rest.remove(wrt)
        m = val * n
        for i in rest:
            m = m * _var(i, r, s)
        out = out + m
    return out


_H_PARSED = _parsed(H_COEF)
_S_PARSED = _parsed(S_COEF)
_V_PARSED = _parsed(V_COEF)


def _pure_vertex_value(parsed_h, parsed_s, parsed_v, r_vertex, s_vertex, T, P):
    """G,H,S,V of the polynomial (NOT including SIC) at a fixed vertex
    (r_vertex, s_vertex) -- used for the 6 non-essenite pure endmembers,
    whose vertex (r, s) is stoichiometrically fixed (no equilibrium solve
    needed, see module docstring)."""
    r_vertex = np.asarray(r_vertex, dtype=np.float64)
    s_vertex = np.asarray(s_vertex, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    rb = np.broadcast_to(r_vertex, T.shape + (6,))
    sb = np.broadcast_to(s_vertex, T.shape + (2,))
    H = _poly_value(parsed_h, rb, sb)
    S = _poly_value(parsed_s, rb, sb)
    V = _poly_value(parsed_v, rb, sb)
    G = H - T * S + (P - 1.0) * V
    return G, H, S, V


# Vertices (r, s) for the 6 non-essenite endmembers, read off directly from
# the DI_G/EN_G/HD_G/CA_G/CF_G/JD_G macros (clinopyroxene.c lines 1324-1481)
# by matching each macro's explicit monomial coefficients against the
# generic polynomial evaluated at a candidate (r, s) point -- confirmed
# term-by-term for every one of these six (e.g. EN_G's "-(HS2)+...-(HX7S2)+
# ...+(HX7S2S2)" pattern matches evaluating the generic H polynomial at
# r5=1, s2=-1 exactly, term for term; JD_G's 20-term expansion matches
# r=[0,.5,-.5,.5,1,0], s=[-1,0] exactly, term for term).
_VERTEX = {
    'diopside':         ([0, 0, 0, 0, 0, 0], [0, 0]),
    'clinoenstatite':   ([0, 0, 0, 0, 0, 1], [0, -1]),
    'hedenbergite':     ([1, 0, 0, 0, 0, 0], [0, 0]),
    'alumino-buffonite': ([0, 1, 0, 0, 0, 0], [0, 0]),
    'buffonite':        ([0, 0, 1, 0, 0, 0], [0, 0]),
    'jadeite':          ([0, 0.5, -0.5, 0.5, 1, 0], [-1, 0]),
}


def _pure_endmember_ghsv(name, T, P):
    r_v, s_v = _VERTEX[name]
    return _pure_vertex_value(_H_PARSED, _S_PARSED, _V_PARSED, r_v, s_v, T, P)


# --- Essenite: its own 1-parameter internal ordering (DES_GDS1/pureOrder) --
def _es_dgds1(s, T, P):
    """DES_GDS1, clinopyroxene.c lines 1424-1426, verbatim (s here is
    essenite's OWN ordering parameter, unrelated to the main s0/s1)."""
    r0 = np.zeros(T.shape + (6,))
    s0 = np.zeros(T.shape + (2,))
    # HX5, HX5S1 etc. do not depend on s, so evaluate the constant parts once:
    HS1v, SS1v, VS1v = HS1, SS1, VS1
    HX5S1v, SX5S1v, VX5S1v = HX5S1, SX5S1, VX5S1
    HS1S1v, SS1S1v, VS1S1v = HS1S1, SS1S1, VS1S1
    return (Rgas * T * (np.log(1.0 + s) - np.log(1.0 - s))
            + HS1v + HX5S1v + HS1S1v * s * 2.0
            - T * (SS1v + SX5S1v + SS1S1v * s * 2.0)
            + (P - 1.0) * (VS1v + VX5S1v + VS1S1v * s * 2.0))


def _pure_es_order(T, P, n_iter=60):
    """Batched Newton solve for essenite's own ordering parameter (a
    single scalar per (T,P), independent of bulk composition r)."""
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    s = np.full(T.shape, 0.0, dtype=np.float64)
    eps = 1e-6
    for _ in range(n_iter):
        f0 = _es_dgds1(s, T, P)
        fp = _es_dgds1(np.clip(s + eps, -1 + 1e-9, 1 - 1e-9), T, P)
        fm = _es_dgds1(np.clip(s - eps, -1 + 1e-9, 1 - 1e-9), T, P)
        jac = (fp - fm) / (2.0 * eps)
        jac = np.where(jac == 0.0, 1.0, jac)
        s = s - f0 / jac
        s = np.clip(s, -1.0 + 1e-12, 1.0 - 1e-12)
    return s


def _pure_essenite_ghsv(T, P):
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    s = _pure_es_order(T, P)
    H = H0 + HX5 + HS1 * s + HX5X5 + HX5S1 * s + HS1S1 * s * s
    S = (-Rgas * ((1.0 - s) * np.log(1.0 - s) + (1.0 + s) * np.log(1.0 + s) - 2.0 * np.log(2.0))
         + S0 + SX5 + SS1 * s + SX5X5 + SX5S1 * s + SS1S1 * s * s)
    V = V0 + VX5 + VS1 * s + VX5X5 + VX5S1 * s + VS1S1 * s * s
    G = H - T * S + (P - 1.0) * V
    return G, H, S, V


_PURE_FUNCS = {
    'diopside':          lambda T, P: _pure_endmember_ghsv('diopside', T, P),
    'clinoenstatite':    lambda T, P: _pure_endmember_ghsv('clinoenstatite', T, P),
    'hedenbergite':      lambda T, P: _pure_endmember_ghsv('hedenbergite', T, P),
    'alumino-buffonite': lambda T, P: _pure_endmember_ghsv('alumino-buffonite', T, P),
    'buffonite':         lambda T, P: _pure_endmember_ghsv('buffonite', T, P),
    'essenite':          _pure_essenite_ghsv,
    'jadeite':           lambda T, P: _pure_endmember_ghsv('jadeite', T, P),
}


def pure_endmember_ghsv(T, P):
    """G,H,S,V (B,7) for the 7 endmembers in ENDMEMBERS order, using this
    model's OWN internal Taylor-coefficient-based reference frame (NOT
    sol_struct_data.json's Berman/Vinet pure-endmember EOS -- see module
    docstring: these are only ever used differentially, against the same
    reference frame, in `ENDMEMBERS`/gmix and the Darken normalization)."""
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    G = np.empty(T.shape + (7,))
    H = np.empty(T.shape + (7,))
    S = np.empty(T.shape + (7,))
    V = np.empty(T.shape + (7,))
    for i, name in enumerate(ENDMEMBERS):
        g, h, s, v = _PURE_FUNCS[name](T, P)
        G[..., i], H[..., i], S[..., i], V[..., i] = g, h, s, v
    return G, H, S, V


# =============================================================================
# Endmember mole fractions <-> r (ENDMEMBERS macro, clinopyroxene.c lines
# 1965-1968, inverted).
# =============================================================================
def endmember_mole_fractions(r):
    r = np.asarray(r, dtype=np.float64)
    x_di = 1.0 - r[..., 0] - r[..., 1] - r[..., 2] - r[..., 3] - r[..., 4] / 2.0 - r[..., 5]
    x_en = r[..., 5]
    x_hd = r[..., 0]
    x_ca = r[..., 1] - r[..., 4] / 2.0
    x_cf = r[..., 2] + r[..., 4] / 2.0
    x_es = r[..., 3] - r[..., 4] / 2.0
    x_jd = r[..., 4]
    return np.stack([x_di, x_en, x_hd, x_ca, x_cf, x_es, x_jd], axis=-1)


def x_to_r(X):
    """(B,7) mole fractions [diopside, clinoenstatite, hedenbergite,
    alumino-buffonite, buffonite, essenite, jadeite] -> (B,6) r."""
    X = np.asarray(X, dtype=np.float64)
    x_di, x_en, x_hd, x_ca, x_cf, x_es, x_jd = (X[..., i] for i in range(7))
    r4 = x_jd
    r0 = x_hd
    r5 = x_en
    r1 = x_ca + r4 / 2.0
    r2 = x_cf - r4 / 2.0
    r3 = x_es + r4 / 2.0
    return np.stack([r0, r1, r2, r3, r4, r5], axis=-1)


# =============================================================================
# Site fractions (clinopyroxene.c lines 3583-3662, the converged/current-
# iterate formulas -- boundary clipping at DBL_EPSILON matches the source).
# =============================================================================
def site_fractions(r, s):
    r = np.asarray(r, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    r0, r1, r2, r3, r4, r5 = (r[..., i] for i in range(6))
    s0, s1 = s[..., 0], s[..., 1]

    def clip(x):
        return np.clip(x, _DBL_EPS, 1.0 - _DBL_EPS)

    xti4m1 = clip((r1 + r2) / 2.0)
    xca2m2 = clip(1.0 - r4 - r5)
    xna1m2 = clip(r4)
    xsi4tet = clip((4.0 - 2.0 * r1 - 2.0 * r2 - 2.0 * r3 + r4) / 4.0)

    xal3m1 = clip((2.0 * r3 + r4 - 2.0 * s0) / 4.0)
    xfe2m1 = clip((2.0 * r0 - r5 - s1) / 2.0)
    xfe3m1 = clip((2.0 * r3 + r4 + 2.0 * s0) / 4.0)
    xmg2m1 = clip(1.0 - r0 - r3 - 0.5 * (r1 + r2 + r4 - r5 - s1))

    xfe2m2 = clip((r5 + s1) / 2.0)
    xmg2m2 = clip((r5 - s1) / 2.0)

    xal3tet = clip((4.0 * r1 + 2.0 * r3 - r4 + 2.0 * s0) / 8.0)
    xfe3tet = clip((4.0 * r2 + 2.0 * r3 - r4 - 2.0 * s0) / 8.0)

    return dict(xti4m1=xti4m1, xca2m2=xca2m2, xna1m2=xna1m2, xsi4tet=xsi4tet,
                xal3m1=xal3m1, xfe2m1=xfe2m1, xfe3m1=xfe3m1, xmg2m1=xmg2m1,
                xfe2m2=xfe2m2, xmg2m2=xmg2m2, xal3tet=xal3tet, xfe3tet=xfe3tet)


def _entropy_SIC(x):
    """clinopyroxene.c lines 1981-1998, verbatim."""
    xmg2m1, xfe2m1, xal3m1, xfe3m1, xti4m1 = (x[k] for k in
        ('xmg2m1', 'xfe2m1', 'xal3m1', 'xfe3m1', 'xti4m1'))
    xsi4tet, xal3tet, xfe3tet = x['xsi4tet'], x['xal3tet'], x['xfe3tet']
    xca2m2, xna1m2, xmg2m2, xfe2m2 = x['xca2m2'], x['xna1m2'], x['xmg2m2'], x['xfe2m2']

    a = np.clip(xmg2m1 + xfe2m1 - xti4m1, _DBL_EPS, None)
    b = np.clip(1.0 - xti4m1, _DBL_EPS, None)
    c = np.clip(xmg2m1 + xfe2m1, _DBL_EPS, None)
    d = np.clip(1.0 - xsi4tet, _DBL_EPS, None)
    e = np.clip(1.0 - xmg2m1 - xfe2m1 - xna1m2, _DBL_EPS, None)
    c2 = np.clip(1.0 - xmg2m1 - xfe2m1, _DBL_EPS, None)
    f = np.clip(1.0 - xna1m2, _DBL_EPS, None)

    return -Rgas * (xmg2m1 * np.log(xmg2m1) + xfe2m1 * np.log(xfe2m1)
                     + xal3m1 * np.log(xal3m1) + xfe3m1 * np.log(xfe3m1)
                     + xti4m1 * np.log(xti4m1)
                     + a * np.log(a) - b * np.log(b) - c * np.log(c)
                     - 2.0 * d * np.log(d)
                     + 2.0 * xal3tet * np.log(xal3tet) + 2.0 * xfe3tet * np.log(xfe3tet)
                     + xca2m2 * np.log(xca2m2) + xna1m2 * np.log(xna1m2)
                     + xmg2m2 * np.log(xmg2m2) + xfe2m2 * np.log(xfe2m2)
                     + e * np.log(e) - c2 * np.log(c2)
                     - f * np.log(f))


# =============================================================================
# Total G, H, S, V (the "G"/"H"/"S"/"V" macros, i.e. NOT gmix -- see module
# docstring; combine with pure_endmember_ghsv + endmember_mole_fractions to
# get gmix/hmix/smix/vmix, exactly mirroring gmixCpx/hmixCpx/smixCpx/vmixCpx).
# =============================================================================
def gibbs_total(r, s, T, P):
    r = np.asarray(r, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    x = site_fractions(r, s)

    H = _poly_value(_H_PARSED, r, s)
    S_poly = _poly_value(_S_PARSED, r, s)
    V = _poly_value(_V_PARSED, r, s)
    S = _entropy_SIC(x) + S_poly
    G = H - T * S + (P - 1.0) * V
    return G, H, S, V


# =============================================================================
# DGDR0-5 / DGDS0-1 (clinopyroxene.c lines 2194-2408): ideal-mixing
# (SIC-derivative) leading term, verbatim, plus the generic polynomial
# derivative for the rest.
# =============================================================================
def _combined_d(wrt, r, s, T, P):
    return (_poly_d(_H_PARSED, r, s, wrt) - T * _poly_d(_S_PARSED, r, s, wrt)
            + (P - 1.0) * _poly_d(_V_PARSED, r, s, wrt))


def dgdr(r, s, T, P, x=None):
    r = np.asarray(r, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    if x is None:
        x = site_fractions(r, s)
    xmg2m1, xfe2m1, xal3m1, xfe3m1, xti4m1 = (x[k] for k in
        ('xmg2m1', 'xfe2m1', 'xal3m1', 'xfe3m1', 'xti4m1'))
    xsi4tet, xal3tet, xfe3tet = x['xsi4tet'], x['xal3tet'], x['xfe3tet']
    xca2m2, xna1m2, xmg2m2, xfe2m2 = x['xca2m2'], x['xna1m2'], x['xmg2m2'], x['xfe2m2']

    # As in _entropy_SIC: the raw macro combinations (xmg2m1+xfe2m1-xti4m1),
    # (1-xti4m1), (xmg2m1+xfe2m1), (1-xsi4tet), (1-xmg2m1-xfe2m1-xna1m2),
    # (1-xmg2m1-xfe2m1), (1-xna1m2) can land at/below zero even though each
    # individual site fraction was already clipped (e.g. two fractions
    # clipped to the same DBL_EPSILON can subtract to ~0 or slightly
    # negative) -- clip these composite quantities too so dG/dr stays
    # finite there (matches the physical T*dSIC/dr limit; verified not to
    # perturb any non-degenerate test point in benchmark_clinopyroxene_test.py).
    a = np.clip(xmg2m1 + xfe2m1 - xti4m1, _DBL_EPS, None)
    b = np.clip(1.0 - xti4m1, _DBL_EPS, None)
    c = np.clip(xmg2m1 + xfe2m1, _DBL_EPS, None)
    d = np.clip(1.0 - xsi4tet, _DBL_EPS, None)
    e = np.clip(1.0 - xmg2m1 - xfe2m1 - xna1m2, _DBL_EPS, None)
    c2 = np.clip(1.0 - xmg2m1 - xfe2m1, _DBL_EPS, None)
    f = np.clip(1.0 - xna1m2, _DBL_EPS, None)

    ideal0 = Rgas * T * (np.log(xfe2m1) - np.log(xmg2m1))
    ideal1 = Rgas * T * (0.5 * np.log(xti4m1) - 0.5 * np.log(xmg2m1)
                          + 0.5 * np.log(b) - np.log(a)
                          + 0.5 * np.log(c) + 0.5 * np.log(e)
                          - np.log(d) - 0.5 * np.log(c2) + np.log(xal3tet))
    ideal2 = Rgas * T * (0.5 * np.log(xti4m1) - 0.5 * np.log(xmg2m1)
                          + 0.5 * np.log(b) - np.log(a)
                          + 0.5 * np.log(c) + 0.5 * np.log(e)
                          - np.log(d) - 0.5 * np.log(c2) + np.log(xfe3tet))
    ideal3 = Rgas * T * (0.5 * np.log(xal3m1) + 0.5 * np.log(xfe3m1) - np.log(xmg2m1)
                          - np.log(a) + np.log(c)
                          - np.log(d) + 0.5 * np.log(xal3tet) + 0.5 * np.log(xfe3tet)
                          + np.log(e) - np.log(c2))
    ideal4 = Rgas * T * (0.25 * np.log(xal3m1) + 0.25 * np.log(xfe3m1) - 0.5 * np.log(xmg2m1)
                          - 0.5 * np.log(a) + 0.5 * np.log(c)
                          + 0.5 * np.log(d) - 0.25 * np.log(xal3tet) - 0.25 * np.log(xfe3tet)
                          - np.log(xca2m2) + np.log(xna1m2) - 0.5 * np.log(e)
                          - 0.5 * np.log(c2) + np.log(f))
    ideal5 = 0.5 * Rgas * T * (np.log(xmg2m1) - 2.0 * np.log(xca2m2)
                                + np.log(xmg2m2) + np.log(xfe2m2 / xfe2m1))

    out = np.empty(r.shape)
    out[..., 0] = ideal0 + _combined_d(0, r, s, T, P)
    out[..., 1] = ideal1 + _combined_d(1, r, s, T, P)
    out[..., 2] = ideal2 + _combined_d(2, r, s, T, P)
    out[..., 3] = ideal3 + _combined_d(3, r, s, T, P)
    out[..., 4] = ideal4 + _combined_d(4, r, s, T, P)
    out[..., 5] = ideal5 + _combined_d(5, r, s, T, P)
    return out


def dgds(r, s, T, P, x=None):
    r = np.asarray(r, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    if x is None:
        x = site_fractions(r, s)
    xmg2m1, xfe2m1, xal3m1, xfe3m1 = x['xmg2m1'], x['xfe2m1'], x['xal3m1'], x['xfe3m1']
    xal3tet, xfe3tet, xmg2m2, xfe2m2 = x['xal3tet'], x['xfe3tet'], x['xmg2m2'], x['xfe2m2']

    ideal0 = 0.5 * Rgas * T * (np.log(xfe3m1) - np.log(xal3m1) + np.log(xal3tet) - np.log(xfe3tet))
    ideal1 = 0.5 * Rgas * T * (np.log(xmg2m1) - np.log(xfe2m1) + np.log(xfe2m2) - np.log(xmg2m2))

    out = np.empty(s.shape)
    out[..., 0] = ideal0 + _combined_d(6, r, s, T, P)
    out[..., 1] = ideal1 + _combined_d(7, r, s, T, P)
    return out


def _ab_initio_guess_and_gates(r):
    """clinopyroxene.c lines 3547-3575, 3674-3675: the closed-form initial
    guess for s (sNew[0], sNew[1]) and the "gate" that forces DGDS0/DGDS1
    to exactly zero -- rather than the real (and, at these degenerate
    compositions, singular/log(0)) gradient -- whenever the corresponding
    pair of site occupants is totally absent from the bulk composition
    (e.g. no Al *and* no Fe3+ at all means there is no real M1-site
    Al/Fe3+ ordering freedom; s0 is then just whatever the stoichiometry
    forces, held fixed at the initial guess). Without this gating, the
    ideal-mixing (SIC-derivative) term in dgds blows up at these
    boundaries (multiple site fractions simultaneously clipped to the
    same DBL_EPSILON) and Newton diverges to the s=+-1 bound instead of
    finding the true (degenerate, non-Newton) answer."""
    r = np.asarray(r, dtype=np.float64)
    totAl = r[:, 1] + r[:, 3]
    totCa = 1.0 - r[:, 4] - r[:, 5]
    totFe2 = r[:, 0]
    totFe3 = r[:, 2] + r[:, 3]
    totMg = 1.0 - r[:, 0] - 0.5 * r[:, 1] - 0.5 * r[:, 2] - r[:, 3] - 0.5 * r[:, 4] + r[:, 5]
    totNa = r[:, 4]
    totTi = 0.5 * (r[:, 1] + r[:, 2])
    totM2 = totCa + totNa
    totM1 = totTi + (totMg + totFe2 - (1.0 - totM2))

    denom0 = totFe3 + totAl
    s0_guess = np.where(denom0 != 0.0, (1.0 - totM1) * (totFe3 - totAl) / np.where(denom0 == 0.0, 1.0, denom0), 0.0)
    denom1 = totFe2 + totMg
    s1_guess = np.where(denom1 != 0.0, (1.0 - totM2) * (totFe2 - totMg) / np.where(denom1 == 0.0, 1.0, denom1), 0.0)

    gate0 = (totFe3 != 0.0) & (totAl != 0.0)
    gate1 = (totFe2 != 0.0) & (totMg != 0.0)
    s_guess = np.stack([s0_guess, s1_guess], axis=-1)
    gate = np.stack([gate0, gate1], axis=-1)
    return s_guess, gate


def solve_ordering(r, T, P, n_iter=60):
    r = np.asarray(r, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    s_guess, gate = _ab_initio_guess_and_gates(r)

    def _dgds_gated(s, rr, TT, PP):
        d = dgds(rr, s, TT, PP)
        return np.where(gate, d, 0.0)

    s = newton_solve_ordering(_dgds_gated, s_guess, r, T, P, n_iter=n_iter,
                               s_min=-1.0 + 1e-9, s_max=1.0 - 1e-9)
    return s


# =============================================================================
# Darken activities (actCpx, clinopyroxene.c lines 4583-4658): NOTE the
# extra per-component normalization by the model's own pure-endmember
# value -- unlike feldspar.py/olivine.py, "g" here is the TOTAL G
# polynomial (not an excess-only gmix), so mu_i = g + sum_j fr_ij dG/dr_j
# is normalized against g_pure_i to give a proper mixing-only mu.
# =============================================================================
def _fr_matrix(r):
    """(B,7,6) Darken FR matrix, clinopyroxene.c FR2(i)..FR7(i) macros
    (lines 1944-1949), component order = ENDMEMBERS."""
    r = np.asarray(r, dtype=np.float64)
    B = r.shape[0]
    r0, r1, r2, r3, r4, r5 = (r[:, i] for i in range(6))
    fr = np.zeros((B, 7, 6), dtype=np.float64)
    # FR2(i): X2 column (r0) -- component 2 (hedenbergite, index 2) is 1-r0, else -r0
    fr[:, 2, 0] = 1.0 - r0
    for i in (0, 1, 3, 4, 5, 6):
        fr[:, i, 0] = -r0
    # FR3(i): X3 column (r1) -- component 3 (alumino-buffonite, idx3) is 1-r1;
    # component 6 (jadeite, idx6) is 0.5-r1; else -r1
    fr[:, 3, 1] = 1.0 - r1
    fr[:, 6, 1] = 0.5 - r1
    for i in (0, 1, 2, 4, 5):
        fr[:, i, 1] = -r1
    # FR4(i): X4 column (r2) -- component 4 (buffonite, idx4) is 1-r2;
    # component 6 (jadeite) is -0.5-r2; else -r2
    fr[:, 4, 2] = 1.0 - r2
    fr[:, 6, 2] = -0.5 - r2
    for i in (0, 1, 2, 3, 5):
        fr[:, i, 2] = -r2
    # FR5(i): X5 column (r3) -- component 5 (essenite, idx5) is 1-r3;
    # component 6 (jadeite) is 0.5-r3; else -r3
    fr[:, 5, 3] = 1.0 - r3
    fr[:, 6, 3] = 0.5 - r3
    for i in (0, 1, 2, 3, 4):
        fr[:, i, 3] = -r3
    # FR6(i): X6 column (r4) -- component 6 (jadeite, idx6) is 1-r4; else -r4
    fr[:, 6, 4] = 1.0 - r4
    for i in (0, 1, 2, 3, 4, 5):
        fr[:, i, 4] = -r4
    # FR7(i): X7 column (r5) -- component 1 (clinoenstatite, idx1) is 1-r5; else -r5
    fr[:, 1, 5] = 1.0 - r5
    for i in (0, 2, 3, 4, 5, 6):
        fr[:, i, 5] = -r5
    return fr


def solution_thermo(r, T, P, n_iter=60, dT=0.02, dP=0.02):
    """Full clinopyroxene solid-solution mixing thermodynamics.

    Returns gmix, H_mix, S_mix, V_mix (analytic, at converged s*),
    Cp_mix/dVdT_mix/dVdP_mix (central-differenced, s* re-solved at each
    stencil point), mu, activities (7,) and s_eq (2,).
    """
    r = np.asarray(r, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)

    def _g_h_s_v(rr, TT, PP, n_it=n_iter):
        s = solve_ordering(rr, TT, PP, n_iter=n_it)
        G, H, S, V = gibbs_total(rr, s, TT, PP)
        Gp, Hp, Sp, Vp = pure_endmember_ghsv(TT, PP)
        x = endmember_mole_fractions(rr)
        gmix = G - np.sum(x * Gp, axis=-1)
        hmix = H - np.sum(x * Hp, axis=-1)
        smix = S - np.sum(x * Sp, axis=-1)
        vmix = V - np.sum(x * Vp, axis=-1)
        return gmix, hmix, smix, vmix, s, G

    gmix, hmix, smix, vmix, s_eq, G_total = _g_h_s_v(r, T, P)

    # Cp_mix, dVdT_mix, dVdP_mix via central differences (envelope theorem
    # does not extend to second derivatives -- see solution_model.py).
    _, h_pT, _, v_pT, _, _ = _g_h_s_v(r, T + dT, P)
    _, h_mT, _, v_mT, _, _ = _g_h_s_v(r, T - dT, P)
    _, _, _, v_pP, _, _ = _g_h_s_v(r, T, P + dP)
    _, _, _, v_mP, _, _ = _g_h_s_v(r, T, P - dP)

    Cp_mix = (h_pT - h_mT) / (2.0 * dT)
    dVdT_mix = (v_pT - v_mT) / (2.0 * dT)
    dVdP_mix = (v_pP - v_mP) / (2.0 * dP)

    # Darken activities (envelope theorem: dG/dr at fixed s* is exact).
    dgdr_val = dgdr(r, s_eq, T, P)
    fr = _fr_matrix(r)
    mu_bulk = G_total[:, None] + np.einsum('bij,bj->bi', fr, dgdr_val)
    Gp, _, _, _ = pure_endmember_ghsv(T, P)
    mu = mu_bulk - Gp
    a = np.exp(mu / (Rgas * T[:, None]))

    dCpdT_mix = np.zeros_like(Cp_mix)  # not needed downstream; higher-order FD omitted (matches olivine.py)

    return dict(gmix=gmix, H_mix=hmix, S_mix=smix, V_mix=vmix,
                Cp_mix=Cp_mix, dCpdT_mix=dCpdT_mix, dVdT_mix=dVdT_mix, dVdP_mix=dVdP_mix,
                mu=mu, activities=a, s_eq=s_eq, endmembers=ENDMEMBERS)
