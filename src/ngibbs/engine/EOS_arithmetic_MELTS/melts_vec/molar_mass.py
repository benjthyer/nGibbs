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
    'H': 1.00794, 'C': 12.0107, 'O': 15.9994, 'Na': 22.98977, 'Mg': 24.305,
    'Al': 26.98154, 'Si': 28.0855, 'P': 30.973762, 'K': 39.0983,
    'Ca': 40.078, 'Ti': 47.867, 'Cr': 51.9961, 'Mn': 54.938, 'Fe': 55.845,
    'Co': 58.9332, 'Ni': 58.6934,
}

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


def molar_mass_from_formula(formula: str) -> float:
    """g/mol for one chemical formula string (e.g. "Mg2SiO4" -> 140.6931)."""
    counts = _parse_formula(formula)
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
