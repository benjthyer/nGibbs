"""
Per-endmember molar mass from its chemical formula string.

`sol_struct_data.json`'s own `mw` field is unpopulated (`'0.0'` for every
entry -- noted in the project plan doc, section 9, as a known gap), but
every record does carry a `formula` field (e.g. "Mg2SiO4" for forsterite,
"CaTi0.5Mg0.5AlSiO6" for alumino-buffonite, "Ca5(PO4)3OH" for apatite) --
see `MELTSSolidParams.formulas`, aligned with `.labels`. This module parses
those formula strings into elemental atom counts and sums standard atomic
weights to get g/mol, rather than trying to decompose each formula into
oxide components (which would need a bespoke, per-mineral-group mapping
rule and is unnecessary: an oxide's molar mass is itself just the sum of
its own atomic weights, so parsing straight to elements gives the
identical number with a single, general-purpose parser).

The atomic weights below are standard IUPAC values and are internally
consistent with `ngibbs.config.constants.OXIDE_MOLAR_MASSES` (e.g.
O=15.9994 + Si=28.0855 reproduces that table's SiO2=60.0843 to 4 decimal
places) -- so results here agree with any oxide-based molar-mass
convention used elsewhere in nGibbs.

Formula syntax handled (covers every `formula` string in `sol_struct_data
.json`'s meltsSolids/pMeltsSolids tables, verified by parsing all of them):
element symbols (one uppercase letter + optional lowercase letters),
optional integer or decimal subscripts (e.g. "Ti0.5"), and parenthesized
groups with their own subscript multiplier (e.g. "(PO4)3", "(OH)2").
"""
from __future__ import annotations
import re
from typing import Dict

import numpy as np

# Standard atomic weights (g/mol), IUPAC, for every element that appears in
# sol_struct_data.json's formula strings.
ATOMIC_WEIGHTS: Dict[str, float] = {
    'H': 1.00794, 'C': 12.0107, 'O': 15.9994, 'F': 18.998403, 'Na': 22.98977,
    'Mg': 24.305, 'Al': 26.98154, 'Si': 28.0855, 'P': 30.973762, 'S': 32.065,
    'Cl': 35.453, 'K': 39.0983, 'Ca': 40.078, 'Ti': 47.867, 'Cr': 51.9961,
    'Mn': 54.938, 'Fe': 55.845, 'Co': 58.9332, 'Ni': 58.6934,
}
# 'S', 'Cl', 'F' were added for melts_vec's 19 meltsLiquid components (SO3
# needs S; the two halogen components below need Cl/F) -- absent before
# because no *solid*-endmember formula in sol_struct_data.json uses them.

_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)|(\()|(\))(\d*\.?\d*)")


def _parse_formula(formula: str) -> Dict[str, float]:
    """Parse a chemical formula string into {element: atom_count}.

    Recursive-descent over a flat token stream; handles nested parenthesized
    groups with their own multiplier (only one level deep appears in this
    project's formulas, but arbitrary nesting works).
    """
    pos = 0
    n = len(formula)

    def parse_group() -> Dict[str, float]:
        nonlocal pos
        counts: Dict[str, float] = {}
        while pos < n and formula[pos] != ')':
            if formula[pos] == '(':
                pos += 1
                sub = parse_group()
                if pos >= n or formula[pos] != ')':
                    raise ValueError(f"Unbalanced parens in formula {formula!r}")
                pos += 1
                m = re.match(r"\d*\.?\d*", formula[pos:])
                mult = float(m.group()) if m.group() else 1.0
                pos += len(m.group())
                for el, ct in sub.items():
                    counts[el] = counts.get(el, 0.0) + ct*mult
            else:
                m = re.match(r"[A-Z][a-z]?", formula[pos:])
                if not m:
                    raise ValueError(
                        f"Could not parse element at position {pos} in {formula!r}")
                el = m.group()
                pos += len(el)
                m2 = re.match(r"\d*\.?\d*", formula[pos:])
                ct = float(m2.group()) if m2.group() else 1.0
                pos += len(m2.group())
                counts[el] = counts.get(el, 0.0) + ct
        return counts

    result = parse_group()
    if pos != n:
        raise ValueError(f"Unexpected trailing characters in formula {formula!r} at {pos}")
    return result


# Labels that aren't valid chemical formulas as literally written -- a
# component NAME, not real chemistry. Currently only melts_vec's two liquid
# halogen components: their trailing "-1" is a MELTS naming convention (a
# formal negative-oxygen "charge balance" marker), not a real oxygen atom --
# confirmed against MAGMA's own liq_struct_data.h and against
# liquid_speciation.py's identical, independently-derived treatment of these
# two labels (each component's whole role is "carry N mol of one halogen").
# This is the single source of truth for that exception: liquid_speciation.py
# imports it from here rather than keeping its own copy.
FORMULA_OVERRIDES: Dict[str, Dict[str, float]] = {
    'Cl2O-1': {'Cl': 2.0},
    'F2O-1': {'F': 2.0},
}


def _counts_for_label(label: str) -> Dict[str, float]:
    if label in FORMULA_OVERRIDES:
        return dict(FORMULA_OVERRIDES[label])
    return _parse_formula(label)


def molar_mass_from_formula(formula: str) -> float:
    """g/mol for one chemical formula string (e.g. "Mg2SiO4" -> 140.6931),
    or one of the two FORMULA_OVERRIDES labels (e.g. "Cl2O-1" -> 70.906)."""
    counts = _counts_for_label(formula)
    unknown = set(counts) - set(ATOMIC_WEIGHTS)
    if unknown:
        raise KeyError(
            f"Formula {formula!r} uses element(s) {sorted(unknown)} not in "
            f"molar_mass.ATOMIC_WEIGHTS")
    return float(sum(ATOMIC_WEIGHTS[el]*ct for el, ct in counts.items()))


def molar_masses(solid_params, names=None) -> np.ndarray:
    """g/mol for each endmember in `names` (default: all of
    `solid_params.labels`, first match per label), via its `formulas` entry.

    Parameters
    ----------
    solid_params : MELTSSolidParams (from params.load_solids()).
    names : optional list of endmember labels; defaults to solid_params.labels.
    """
    if names is None:
        names = solid_params.labels
    idx = solid_params.index(names)
    return np.array([molar_mass_from_formula(solid_params.formulas[i]) for i in idx],
                     dtype=np.float64)


def liquid_molar_masses(liquid_params, names=None) -> np.ndarray:
    """g/mol for each meltsLiquid component in `names` (default: all of
    `liquid_params.labels`).

    Unlike the solid endmembers (which need a separate `formulas` field --
    see `molar_masses` above), every meltsLiquid component's own LABEL
    already IS its chemical formula (e.g. "Mg2SiO4", "KAlSiO4",
    "Ca3(PO4)2") -- confirmed against liq_struct_data.json's own `label`
    field for the standard meltsLiquid table -- except the two halogen
    components resolved via FORMULA_OVERRIDES above. So this parses labels
    directly, with no extra field to thread through.

    Parameters
    ----------
    liquid_params : MELTSLiquidParams (from liquid_params.load_liquid()).
    names : optional list of component labels; defaults to liquid_params.labels.
    """
    if names is None:
        names = liquid_params.labels
    idx = liquid_params.index(names)
    labels = [liquid_params.labels[i] for i in idx]
    return np.array([molar_mass_from_formula(lbl) for lbl in labels], dtype=np.float64)
