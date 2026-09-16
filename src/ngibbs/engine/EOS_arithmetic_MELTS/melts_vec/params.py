"""
Load MELTS solid-endmember thermodynamic reference data (extracted from
MAGMA's `includes/sol_struct_data.h` -- see `MELTS_Parameters/README.md`
for provenance) into flat numpy arrays suitable for vectorized evaluation,
mirroring the role `EOS_arithmetic/hefesto_vec/params.py` plays for
HeFESTo's `apar` array.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .constants import CP_BERMAN, CP_SAXENA, EOS_BERMAN, EOS_VINET

_DEFAULT_JSON = Path(__file__).resolve().parent.parent / "MELTS_Parameters" / "sol_struct_data.json"

_CP_TYPE_CODE = {"CP_BERMAN": CP_BERMAN, "CP_SAXENA": CP_SAXENA}
_EOS_TYPE_CODE = {"EOS_BERMAN": EOS_BERMAN, "EOS_VINET": EOS_VINET}

# Default table: rhyolite-MELTS 1.0.x/1.1.x/1.2.x -- the mode nGibbs's
# existing MELTS102/MELTS120 NN emulators were trained against.
DEFAULT_TABLE = "meltsSolids"


@dataclass
class MELTSSolidParams:
    """Flat, vectorization-friendly view of a `Solids <table>[]` array.

    All arrays are indexed by endmember, length N.
    """
    labels    : list      # (N,) str -- endmember name, e.g. "forsterite"
    types     : list      # (N,) str -- "PHASE" or "COMPONENT"
    formulas  : list      # (N,) str -- chemical formula
    h         : np.ndarray  # (N,) J        -- reference enthalpy at (Tr, Pr)
    s         : np.ndarray  # (N,) J/K      -- reference entropy at (Tr, Pr)
    v0        : np.ndarray  # (N,) J/bar    -- reference volume at (Tr, Pr)
    cp_type   : np.ndarray  # (N,) int      -- CP_BERMAN / CP_SAXENA
    cp_coeffs : np.ndarray  # (N,8)         -- Berman: k0,k1,k2,k3,Tt,deltah,l1,l2
                             #                  Saxena: a,b,c,d,e,g,h,(unused)
    eos_type  : np.ndarray  # (N,) int      -- EOS_BERMAN / EOS_VINET
    eos_coeffs: np.ndarray  # (N,4)         -- Berman: v1,v2,v3,v4
                             #                  Vinet:  alpha,K,Kp,(unused)
    label_index: dict = field(default_factory=dict)  # label -> row index (first match)

    @property
    def nspec(self) -> int:
        return len(self.labels)

    def index(self, names) -> np.ndarray:
        """Row indices for a list of endmember labels (first match each)."""
        return np.array([self.label_index[n] for n in names], dtype=np.intp)

    def select(self, names) -> "MELTSSolidParams":
        """Return a new MELTSSolidParams containing only the given labels,
        in the given order (convenience for testing / small batches)."""
        idx = self.index(names)
        return MELTSSolidParams(
            labels=[self.labels[i] for i in idx],
            types=[self.types[i] for i in idx],
            formulas=[self.formulas[i] for i in idx],
            h=self.h[idx], s=self.s[idx], v0=self.v0[idx],
            cp_type=self.cp_type[idx], cp_coeffs=self.cp_coeffs[idx],
            eos_type=self.eos_type[idx], eos_coeffs=self.eos_coeffs[idx],
            label_index={n: i for i, n in enumerate([self.labels[i] for i in idx])},
        )


def load_solids(json_path: str | Path | None = None,
                 table: str = DEFAULT_TABLE,
                 prefer_type: str | None = "COMPONENT") -> MELTSSolidParams:
    """Load one `Solids <table>[]` array from the extracted JSON.

    Parameters
    ----------
    json_path : path to sol_struct_data.json (defaults to the copy shipped
        in MELTS_Parameters/).
    table : one of "xMeltsSolids", "meltsSolids", "meltsFluidSolids",
        "pMeltsSolids" (see MELTS_Parameters/README.md).
    prefer_type : when a label appears more than once in the table (e.g.
        "fayalite" appears both as a COMPONENT, for use inside the olivine
        solid solution, and as a standalone EOS_VINET-parameterized PHASE),
        keep the entry whose `type` matches this string and drop the other.
        Pass None to keep every entry (label_index then points at the
        *last* occurrence, matching a plain dict-from-list construction).
    """
    path = Path(json_path) if json_path is not None else _DEFAULT_JSON
    with open(path) as fh:
        all_tables = json.load(fh)
    if table not in all_tables:
        raise KeyError(f"table {table!r} not in {sorted(all_tables)} ({path})")
    records = all_tables[table]

    if prefer_type is not None:
        by_label: dict[str, dict] = {}
        for r in records:
            cur = by_label.get(r["label"])
            if cur is None or (cur["type"] != prefer_type and r["type"] == prefer_type):
                by_label[r["label"]] = r
        records = list(by_label.values())

    n = len(records)
    labels   = [r["label"] for r in records]
    types    = [r["type"] for r in records]
    formulas = [r["formula"] for r in records]
    h  = np.array([r["h"] for r in records], dtype=np.float64)
    s  = np.array([r["s"] for r in records], dtype=np.float64)
    v0 = np.array([r["v"] for r in records], dtype=np.float64)

    cp_type = np.array([_CP_TYPE_CODE[r["cp_type"]] for r in records], dtype=np.int32)
    cp_coeffs = np.zeros((n, 8), dtype=np.float64)
    for i, r in enumerate(records):
        c = r["cp_coeffs"]
        cp_coeffs[i, :len(c)] = c

    eos_type = np.array([_EOS_TYPE_CODE[r["eos_type"]] for r in records], dtype=np.int32)
    eos_coeffs = np.zeros((n, 4), dtype=np.float64)
    for i, r in enumerate(records):
        c = r["eos_coeffs"]
        eos_coeffs[i, :len(c)] = c

    label_index = {}
    for i, lbl in enumerate(labels):
        label_index.setdefault(lbl, i)

    return MELTSSolidParams(
        labels=labels, types=types, formulas=formulas,
        h=h, s=s, v0=v0,
        cp_type=cp_type, cp_coeffs=cp_coeffs,
        eos_type=eos_type, eos_coeffs=eos_coeffs,
        label_index=label_index,
    )
