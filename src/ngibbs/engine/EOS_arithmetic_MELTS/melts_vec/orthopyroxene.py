"""
Orthopyroxene (quadrilateral + non-quadrilateral) solid-solution mixing
model -- vectorized translation of `sources/orthopyroxene.c`'s
`gmixOpx`/`actOpx`/`order`, verified against the file directly.

Relationship to `clinopyroxene.py`
-----------------------------------
`orthopyroxene.c`'s own header comment says it "Canalbalized PYROXENE.C
(now CLINOPYROXENE.C)" (May 1997) -- i.e. both files are literal forks of
one shared ancestor. Comparing the two files directly (`diff`) confirms
this at the source level: the ENTIRE Taylor-coefficient macro block
(`#define NR 6`/`NS 2`/`NA 7`, every `H1..H7`/`HX2..HX7X7X7`/`S...`/`V...`
macro, `DECLARE_SITE_FRACTIONS`, the `SIC` configurational-entropy
formula, the `order()` Newton-solve gating logic, and the `FR2..FR7`/
`ENDMEMBERS` Darken-matrix macros) is BYTE-IDENTICAL text between
`clinopyroxene.c` and `orthopyroxene.c`. The two files also define
byte-identical numeric values for every "c"-prefixed (monoclinic
reference) base constant. What differs is:

  1. Which branch of every `(clino) ? cFOO : oFOO` ternary the file's
     OWN main `order()`/`gibbs()` engine uses: `clinopyroxene.c` forces
     `clino=TRUE` (see that module's docstring), `orthopyroxene.c` forces
     `clino=FALSE` (its own `isClino()` is `#ifdef ISCLINO`-gated dead
     code that collapses to `return FALSE;`, the mirror image of
     clinopyroxene's `return TRUE;`). This module's MIXED-composition
     Taylor coefficients (`H_COEF`/`S_COEF`/`V_COEF` below) are therefore
     genuinely different numbers from clinopyroxene's, built from the
     "o"-prefixed base constants -- AND, unlike clinopyroxene.c (where
     clino=TRUE makes every `H1..H7`/`pH1..pH7` vertex-offset term
     collapse to 0), here `H1..H7` are NONZERO (`(clino)?0.0:DFOO` picks
     `DFOO` for FALSE) while `pH1..pH7` collapse to 0 instead. This
     module explicitly carries the nonzero `H1..H7`/`S1..S7`/`V1..V7`
     terms through the coefficient derivation below, INCLUDING the
     leading `H0`/`S0`/`V0` term, which clinopyroxene.py could safely
     omit from its generic polynomial engine (it was exactly 0.0) but
     must be added explicitly here since it is not.
  2. The PURE-ENDMEMBER properties (`purePyx`/`pureOrder` in the C
     source, `pure_endmember_ghsv`/essenite's own ordering here) are a
     DIFFERENT story: both `purePyx()` and `pureOrder()` declare a LOCAL
     `static const int clino = TRUE;` that C-scopes over (shadows) the
     file's own global `clino` for every macro they expand -- the
     source's own comment explains why: "a static variable clino
     overrides the current global variable clino. This insures that the
     solution properties returned by the global functions in this file
     are always reference to a monoclinic state (i.e. the thermodynamic
     constants stored [in] SOLID_STRUCT_DATA.H and used in GIBBS.C refer
     to the monoclinic structure)." Concretely: BOTH `clinopyroxene.c`
     AND `orthopyroxene.c` always evaluate pure-endmember G/H/S/V (and
     essenite's own internal ordering) at clino=TRUE, using the SAME
     "c"-prefixed constants -- confirmed both by reading `orthopyroxene
     .c`'s `pureOrder`/`purePyx` directly and by `sol_struct_data.json`'s
     own `meltsSolids` table, which lists the SAME 7 endmember records
     (identical h/s/v/Cp/EOS fields, byte for byte) once for
     "clinopyroxene" and again for "orthopyroxene". This module therefore
     does NOT re-derive a second pure-endmember coefficient set: it
     imports and reuses `clinopyroxene`'s `pure_endmember_ghsv`,
     `_es_dgds1`/`_pure_es_order`/`_pure_essenite_ghsv`, and `_VERTEX`
     machinery directly, since they are proven (by the two source files'
     own byte-identical constants) to be the same numbers. The C harness
     (`verify_orthopyroxene.c`) checks this equivalence explicitly by
     dumping `orthopyroxene.c`'s OWN `purePyx()` output and comparing it
     against `clinopyroxene.pure_endmember_ghsv()`, rather than assuming
     it from source inspection alone.

Endmembers, independent composition variables, ordering parameters, the
`x_to_r`/`endmember_mole_fractions`/`site_fractions`/`_fr_matrix`
Darken-matrix machinery, and the essenite-analog one-parameter internal
ordering are otherwise identical in form to `clinopyroxene.py` (same
`FR2..FR7`/`ENDMEMBERS` macros, same `DECLARE_SITE_FRACTIONS` variable
list, same vertex-to-endmember binding -- H1=diopside, H2=hedenbergite,
H3=alumino-buffonite-type, H4=buffonite-type, H5=essenite, H6=jadeite,
H7=clinoenstatite/enstatite-type -- confirmed identical between the two
files' vertex comment blocks), so this module reuses those functions
from `clinopyroxene.py` directly rather than re-deriving them, keeping
transcription risk to just the genuinely-new part: the MIX Taylor
coefficients below.

Units, DBL_EPSILON convention, generic polynomial engine, and the
Newton-solve gating logic are all as documented in `clinopyroxene.py`'s
module docstring; not repeated here.
"""
from __future__ import annotations
import numpy as np

from .constants import Rgas
from .solution_model import newton_solve_ordering
from . import clinopyroxene as _cpx
from .clinopyroxene import (
    ENDMEMBERS, _DBL_EPS, _TOKEN_INDEX, _var, _parsed, _poly_value, _poly_d,
    x_to_r, endmember_mole_fractions, site_fractions, _entropy_SIC, _fr_matrix,
    pure_endmember_ghsv,
)

# =============================================================================
# Base constants, verbatim from orthopyroxene.c lines 97-247 (both "o"- and
# "c"-prefixed values are needed here, unlike clinopyroxene.py, because this
# module's MIX coefficients use the "o" branch while the imported pure-
# endmember machinery above uses the "c" branch -- see module docstring).
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

_oHEX = -1.87 * 1000.0 * 4.184
_oVEX = -0.029 * 4.184
_oHX = -0.45 * 1000.0 * 4.184
_oVX = 0.00675 * 4.184

_oWHFEMG = 2.0 * 1000.0 * 4.184
_oWVFEMG = 0.003375 * 4.184
_oWH12 = 2.0 * 1000.0 * 4.184
_oWV12 = 0.003375 * 4.184
_oWHCAMG = 7.56 * 1000.0 * 4.184
_oWVCAMG = 0.008 * 4.184
_oWHCAFE = 4.12 * 1000.0 * 4.184
_oWVCAFE = 0.011 * 4.184
_oDWHCAMG = -1.3 * 1000.0 * 4.184
_oDWVCAMG = 0.012 * 4.184
_oDWHCAFE = -1.1 * 1000.0 * 4.184
_oDWVCAFE = 0.005 * 4.184

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
_W45 = 27.14312 * 1000.0
_W45P = 16.318 * 1000.0
_W46 = 40.40662 * 1000.0
_W56 = 0.00000 * 1000.0
_W5P6 = 0.00000 * 1000.0
_oW37 = 28.65091 * 1000.0
_oW3U7U = 34.90070 * 1000.0
_oW47 = 35.38104 * 1000.0
_oW4U7U = 28.54527 * 1000.0
_oW55 = 35.670 * 1000.0
_oW57 = 33.25291 * 1000.0
_oW57U = 15.29737 * 1000.0
_oW5P7 = 36.24989 * 1000.0
_oW5P7U = 2.35321 * 1000.0
_oW67 = 0.00000 * 1000.0
_oW67U = 0.00000 * 1000.0

_H23 = -2.71700 * 1000.0
_S23 = 0.0
_H24 = -7.36648 * 1000.0
_S24 = 0.0
_S55 = 0.0
_oH55 = 9.48524 * 1000.0

_DcTOoH3, _DcTOoS3, _DcTOoV3 = 25.11924 * 1000.0, -2.00000, -0.05129
_DcTOoH4, _DcTOoS4, _DcTOoV4 = 25.11924 * 1000.0, -2.00000, -0.05129
_DcTOoH5, _DcTOoS5, _DcTOoV5 = 25.11924 * 1000.0, -2.00000, -0.05129
_DcTOoH6, _DcTOoS6, _DcTOoV6 = 25.11924 * 1000.0, -2.00000, -0.05129

# --- Dependent parameters (orthopyroxene.c lines 253-365), clino=FALSE -----
_oV0DI = _cV0DI + _DV1
_oV027 = _oV0FS - _oV0EN + 2.0 * _oV0DI - 2.0 * _oV0HD

_H027, _S027, _V027 = _oH027, _oS027, _oV027   # clino=FALSE
_dH027 = _dS027 = _dV027 = 0.0                 # (clino) ? p-c : 0.0 -> 0.0
_HEX, _VEX = _oHEX, _oVEX
_dHEX = _dVEX = 0.0
_HX, _VX = _oHX, _oVX
_dHX = _dVX = 0.0

_WHFEMG, _WVFEMG = _oWHFEMG, _oWVFEMG
_WH12, _WV12 = _oWH12, _oWV12
_WHCAMG, _WVCAMG = _oWHCAMG, _oWVCAMG
_DWHCAMG, _DWVCAMG = _oDWHCAMG, _oDWVCAMG
_WHCAFE, _WVCAFE = _oWHCAFE, _oWVCAFE
_DWHCAFE, _DWVCAFE = _oDWHCAFE, _oDWVCAFE

_W37, _W3U7U = _oW37, _oW3U7U
_W47, _W4U7U = _oW47, _oW4U7U
_W55 = _oW55
_W57, _W57U = _oW57, _oW57U
_W5P7, _W5P7U = _oW5P7, _oW5P7U
_W67, _W67U = _oW67, _oW67U
_H55 = _oH55

# --- Vertices (orthopyroxene.c lines 371-405), clino=FALSE -> NONZERO ------
_H1, _S1, _V1 = _DH1, _DS1, _DV1
_H2, _S2, _V2 = -_DH4, -_DS4, -(_cV0HD - _oV0HD)
_H3, _S3, _V3 = _DcTOoH3, _DcTOoS3, _DcTOoV3
_H4, _S4, _V4 = _DcTOoH4, _DcTOoS4, _DcTOoV4
_H5, _S5, _V5 = _DcTOoH5, _DcTOoS5, _DcTOoV5
_H6, _S6, _V6 = _DcTOoH6, _DcTOoS6, _DcTOoV6
_H7, _S7, _V7 = -_DH2, -_DS2, -_DV2
# pH1..pH7 (orthopyroxene.c lines 409-442), clino=FALSE -> 0.0
_pH1 = _pH2 = _pH3 = _pH4 = _pH5 = _pHj = _pH7 = 0.0
_pS1 = _pS2 = _pS3 = _pS4 = _pS5 = _pSj = _pS7 = 0.0
_pV1 = _pV2 = _pV3 = _pV4 = _pV5 = _pVj = _pV7 = 0.0

# =============================================================================
# Taylor-expansion coefficients (orthopyroxene.c lines 449-778), verbatim,
# clino=FALSE branch -- confirmed byte-identical FORMULAS to clinopyroxene.c
# via `diff`; H0/S0/V0 and H1..H7/S1..S7/V1..V7 are nonzero here (unlike
# clinopyroxene.py) so they are carried through explicitly.
# =============================================================================
H0 = _H1
HX2 = _H2 - _H1 + _WH12
HX3 = _H3 - _H1 + _W13
HX4 = _H4 - _H1 + _W14
HX5 = _H5 - _H1 + 0.5 * (_W15 + _W15P + _H55)
HX6 = (_H6 + 0.5 * (_H4 - _H1 - _H3 - _H5) - 0.5 * _W13 + 0.5 * _W14
       + 0.25 * _W15 - 0.75 * _W15P + _W16 - 0.75 * _H55)
HX7 = (_H7 - _H1 + _pH1 + 0.5 * (_WHCAFE + _WHCAMG - _WH12)
       + 0.25 * (_HEX + _HX + _H027) + 0.5 * _DWHCAMG - 1.5 * _DWHCAFE)

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

# --- S coefficients (lines 588-658). RHYOLITE_ADJUSTMENTS build: SS2=0.0. --
S0 = _S1
SX2 = _S2 - _S1
SX3 = _S3 - _S1
SX4 = _S4 - _S1
SX5 = _S5 - _S1 + 0.5 * _S55
SX6 = _S6 + 0.5 * (_S4 - _S1 - _S3 - _S5) - 0.75 * _S55
SX7 = _S7 - _S1 + _pS1 + 0.25 * _S027

SS1 = -0.5 * _S55
SS2 = 0.0

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

# --- V coefficients (lines 660-777). ---------------------------------------
V0 = _V1
VX2 = _V2 - _V1 + _WV12
VX3 = _V3 - _V1
VX4 = _V4 - _V1
VX5 = _V5 - _V1
VX6 = _V6 + 0.5 * (_V4 - _V1 - _V3 - _V5)
VX7 = (_V7 - _V1 + _pV1 + 0.5 * (_WVCAFE + _WVCAMG - _WV12)
       + 0.25 * (_VEX + _VX + _V027) + 0.5 * _DWVCAMG - 1.5 * _DWVCAFE)

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

_H_PARSED = _parsed(H_COEF)
_S_PARSED = _parsed(S_COEF)
_V_PARSED = _parsed(V_COEF)


# =============================================================================
# Total G, H, S, V (the MIX "G"/"H"/"S"/"V" macros): H0/S0/V0 are explicit
# here (nonzero, unlike clinopyroxene.py -- see module docstring).
# =============================================================================
def gibbs_total(r, s, T, P):
    r = np.asarray(r, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    x = site_fractions(r, s)

    H = H0 + _poly_value(_H_PARSED, r, s)
    S_poly = S0 + _poly_value(_S_PARSED, r, s)
    V = V0 + _poly_value(_V_PARSED, r, s)
    S = _entropy_SIC(x) + S_poly
    G = H - T * S + (P - 1.0) * V
    return G, H, S, V


# =============================================================================
# DGDR0-5 / DGDS0-1: ideal-mixing (SIC-derivative) leading term, verbatim
# (identical formula to clinopyroxene.py -- SIC's functional form and the
# site-fraction <-> r,s relations are byte-identical between the two source
# files), plus the generic polynomial derivative for the MIX coefficients
# (H0/S0/V0 drop out of the derivative since they are r,s-independent).
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
    """orthopyroxene.c lines ~3426-3454, 3553-3566: identical gating logic
    to clinopyroxene.py's (see there for the rationale)."""
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
# Darken activities (actOpx): same non-standard per-component pure-G
# normalization as clinopyroxene.py's actCpx -- see there for rationale.
# Pure-endmember G comes from clinopyroxene.pure_endmember_ghsv (see module
# docstring: proven identical between the two files).
# =============================================================================
def solution_thermo(r, T, P, n_iter=60, dT=0.02, dP=0.02):
    """Full orthopyroxene solid-solution mixing thermodynamics. Same
    return contract as clinopyroxene.solution_thermo."""
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

    _, h_pT, _, v_pT, _, _ = _g_h_s_v(r, T + dT, P)
    _, h_mT, _, v_mT, _, _ = _g_h_s_v(r, T - dT, P)
    _, _, _, v_pP, _, _ = _g_h_s_v(r, T, P + dP)
    _, _, _, v_mP, _, _ = _g_h_s_v(r, T, P - dP)

    Cp_mix = (h_pT - h_mT) / (2.0 * dT)
    dVdT_mix = (v_pT - v_mT) / (2.0 * dT)
    dVdP_mix = (v_pP - v_mP) / (2.0 * dP)

    dgdr_val = dgdr(r, s_eq, T, P)
    fr = _fr_matrix(r)
    mu_bulk = G_total[:, None] + np.einsum('bij,bj->bi', fr, dgdr_val)
    Gp, _, _, _ = pure_endmember_ghsv(T, P)
    mu = mu_bulk - Gp
    a = np.exp(mu / (Rgas * T[:, None]))

    dCpdT_mix = np.zeros_like(Cp_mix)

    return dict(gmix=gmix, H_mix=hmix, S_mix=smix, V_mix=vmix,
                Cp_mix=Cp_mix, dCpdT_mix=dCpdT_mix, dVdT_mix=dVdT_mix, dVdP_mix=dVdP_mix,
                mu=mu, activities=a, s_eq=s_eq, endmembers=ENDMEMBERS)
