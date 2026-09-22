#!/usr/bin/env python3
"""
Extract the liquid regular-solution interaction-parameter table (the W(i,j)
enthalpy/entropy/volume Margules parameters gmixLiq_v34/hmixLiq_v34/
smixLiq_v34/vmixLiq_v34 in `sources/liquid_v34.c` sum over all i<j component
pairs) out of MAGMA's `includes/param_struct_data.h` into JSON.

Why this file, not a "meltsModelParameters"/"pMeltsModelParameters" pair
------------------------------------------------------------------------
The `liquid_v34.c` snapshot uploaded to this project (dated 2006, an
xMELTS/MELTS-5.x-era file per its own RCS log) references two separate
global arrays, `meltsModelParameters`/`pMeltsModelParameters`, selected by
`calculationMode` inside its own `WH(i,j)`/`WS(i,j)`/`WV(i,j)` macros. The
CONNECTED, current MAGMA source snapshot's `param_struct_data.h` (dated
2010-05-18, four years later) has no such split -- it defines exactly ONE
table, `originalModelParameters[]`, matching the single `ModelParameters
*modelParameters` extern pointer declared in `includes/silmin.h` and used
throughout the connected `liquid.c` (whose own `gmixLiq()`/etc. wrapper
functions dispatch to `gmixLiq_v34` for BOTH `MODE__MELTS` and
`MODE_pMELTS` -- confirmed by direct inspection, see `liquid.c` around line
8690). This means the connected source's actual calibration treats MELTS
and pMELTS as sharing identical liquid W(i,j) interaction parameters (the
per-mode difference is which SOLID phases/components are active, not the
liquid mixing parameters themselves) -- consistent with there being only
one array to extract. `originalModelParameters` is therefore the
authoritative, current source of the numeric W(i,j) values used here, same
as every other MAGMA-derived parameter table in this project (extracted
from the CONNECTED source snapshot, not from the separately-uploaded,
older `liquid_v34.c`, whose own formulas are still used verbatim -- see
`melts_vec/liquid_nonideal.py` and `melts_vec/tests/
verify_liquid_nonideal.c` for where the function bodies themselves are
extracted from).

Format of the source block
---------------------------
`static ModelParameters originalModelParameters[] = { { "W(i ,j )", enthalpy,
entropy, volume, activeH, activeS, activeV, activeF }, ... };` -- entries
are grouped by a second ("j") component, each block preceded by a "Basis
species" or similar comment, and the file's own preamble force-undefines
`USE_NEW_MELTS_WATER_MODEL`, so `#ifdef USE_NEW_MELTS_WATER_MODEL / #else /
#endif` pairs (used only for two W(H2O,*) entries) always resolve to the
`#else` branch here.

A legacy 20th component, CaCO3, appears throughout the table (evidently a
component MELTS once carried and has since dropped) but is not one of the
19 `meltsLiquid` components this package targets (`liq_struct_data.json`);
its rows are dropped. After dropping CaCO3 and resolving the water-model
ifdef, this extraction produces exactly C(19,2) = 171 unique unordered
pairs, one per off-diagonal (i,j) combination of the 19 canonical liquid
components -- verified by this script's own assertion at the end.
"""
import json
import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/mnt/user-data/uploads/MAGMA/includes/param_struct_data.h")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else
           str(Path(__file__).resolve().parent / "liq_wij_data.json"))

# Canonical 19 meltsLiquid components, in the SAME order as
# MELTS_Parameters/liq_struct_data.json's "meltsLiquid" table (which is
# also `sources/liquid_v34.c`'s own FORMULAS[] order, confirmed by direct
# comparison) -- H2O is deliberately last (index 18) in both.
CANON19 = ["SiO2", "TiO2", "Al2O3", "Fe2O3", "MgCr2O4", "Fe2SiO4", "MnSi0.5O2",
           "Mg2SiO4", "NiSi0.5O2", "CoSi0.5O2", "CaSiO3", "Na2SiO3", "KAlSiO4",
           "Ca3(PO4)2", "CO2", "SO3", "Cl2O-1", "F2O-1", "H2O"]

# param_struct_data.h uses the OLDER, pre-speciation-rename oxide/element
# labels for three components -- confirmed by cross-referencing against
# `testLiq_v34`'s own oxide-name vs. component-formula arrays in
# liquid_v34.c (NAMES[] has "SO3"/"Cl2O-1"/"F2O-1" already, but the
# interaction-parameter table's OWN row labels predate that rename).
ALIAS = {"S": "SO3", "Cl": "Cl2O-1", "F": "F2O-1"}


def resolve_water_model_ifdef(text: str) -> str:
    """Resolve `#ifdef USE_NEW_MELTS_WATER_MODEL / #else / #endif` blocks by
    always keeping the #else branch -- the file's own preamble
    unconditionally `#undef`s this macro, so the #else branch is always
    the one actually compiled."""
    out = []
    i = 0
    pat_if = re.compile(r"#ifdef\s+USE_NEW_MELTS_WATER_MODEL")
    while True:
        m = pat_if.search(text, i)
        if not m:
            out.append(text[i:])
            break
        out.append(text[i:m.start()])
        else_m = re.search(r"#else", text[m.end():])
        endif_m = re.search(r"#endif", text[m.end():])
        assert else_m and endif_m and else_m.start() < endif_m.start(), \
            "malformed #ifdef USE_NEW_MELTS_WATER_MODEL block"
        else_start = m.end() + else_m.end()
        endif_start = m.end() + endif_m.start()
        out.append(text[else_start:endif_start])
        i = m.end() + endif_m.end()
    return "".join(out)


def parse_label(label: str):
    """'W(i ,j )' -> (canon(i), canon(j)). Split on the FIRST comma only --
    some component formulas (Ca3(PO4)2) contain their own parentheses, so a
    naive balanced-paren regex on the whole label is unsafe; the W(...)
    argument list itself only ever has one comma separating i and j."""
    lab = label.strip()
    assert lab.startswith("W(") and lab.endswith(")"), lab
    inner = lab[2:-1]
    i_str, j_str = inner.split(",", 1)
    canon = lambda s: ALIAS.get(s.strip(), s.strip())
    return canon(i_str), canon(j_str)


def main():
    text = SRC.read_text()
    start = text.index("static ModelParameters originalModelParameters[] = {")
    brace = text.index("{", start)
    end = text.index("\n};", brace)
    block = resolve_water_model_ifdef(text[brace:end])

    row_pat = re.compile(
        r'\{\s*"([^"]*)"\s*,\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*,')
    rows = row_pat.findall(block)

    canon_set = set(CANON19)
    pairs = {}
    for label, h, s, v in rows:
        if not label.strip().startswith("W("):
            continue  # stray match from a trailing/adjacent array, ignore
        a, b = parse_label(label)
        if a not in canon_set or b not in canon_set:
            continue  # legacy CaCO3 rows (or anything else outside our 19)
        key = tuple(sorted((a, b)))
        val = (float(h), float(s), float(v))
        if key in pairs:
            assert pairs[key] == val, f"conflicting duplicate row for {key}: {pairs[key]} vs {val}"
        pairs[key] = val

    expected = len(CANON19) * (len(CANON19) - 1) // 2
    missing = [tuple(sorted((CANON19[i], CANON19[j])))
               for i in range(len(CANON19)) for j in range(i + 1, len(CANON19))
               if tuple(sorted((CANON19[i], CANON19[j]))) not in pairs]
    assert not missing, f"missing {len(missing)} W(i,j) pairs: {missing}"
    assert len(pairs) == expected, f"got {len(pairs)} pairs, expected {expected}"
    assert all(s == 0.0 and v == 0.0 for (h, s, v) in pairs.values()), \
        "expected all W(i,j) entropy/volume terms to be zero in this calibration"

    out = {
        "_comment": ("Liquid regular-solution W(i,j) enthalpy interaction "
                     "parameters (J), extracted from the connected MAGMA "
                     "source's includes/param_struct_data.h "
                     "(originalModelParameters[]), verbatim (byte-identical "
                     "numeric literals, no hand-retyping) -- see this "
                     "script's module docstring for why this table (not "
                     "liquid_v34.c's own meltsModelParameters/"
                     "pMeltsModelParameters split) is authoritative. "
                     "Entropy and volume interaction terms are all "
                     "identically zero in this calibration -- the liquid "
                     "non-ideal mixing correction is purely enthalpic "
                     "(see melts_vec/liquid_nonideal.py)."),
        "labels": CANON19,
        "pairs": {f"{a}|{b}": {"WH": h, "WS": s, "WV": v}
                  for (a, b), (h, s, v) in sorted(pairs.items())},
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {len(pairs)} W(i,j) pairs to {OUT}")


if __name__ == "__main__":
    main()
