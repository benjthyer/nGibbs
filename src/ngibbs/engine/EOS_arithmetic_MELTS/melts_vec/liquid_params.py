"""
Load MELTS liquid-component thermodynamic reference data (extracted from
MAGMA's `includes/liq_struct_data.h` -- see `MELTS_Parameters/README.md`)
into flat numpy arrays for `liquid_eos.py`, mirroring `params.py`'s role
for the solid endmembers.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_DEFAULT_JSON = Path(__file__).resolve().parent.parent / "MELTS_Parameters" / "liq_struct_data.json"

DEFAULT_TABLE = "meltsLiquid"


@dataclass
class MELTSLiquidParams:
    """Flat, vectorization-friendly view of a `Liquid <table>[]` array.

    All arrays indexed by liquid component, length N (19 for meltsLiquid:
    the standard rhyolite-MELTS oxide/silicate component list).
    """
    labels : list
    # ThermoRef block (crystalline reference phase, used only to integrate
    # Cp from Tr up to Tfusion -- see liquid_eos.kress_component). k0-k3 are
    # the plain Berman polynomial; cp_t/cp_h/l1/l2 are the optional
    # order-disorder ("lambda transition") correction -- zero for most
    # components, but genuinely nonzero for Fe2O3, KAlSiO4 and Ca3(PO4)2
    # (all with cp_t < t_fusion, so it DOES contribute to the fusion-state
    # H,S -- dropping it silently would be a real, if small, error for
    # exactly those three components).
    ref_h : np.ndarray
    ref_s : np.ndarray
    ref_k0: np.ndarray
    ref_k1: np.ndarray
    ref_k2: np.ndarray
    ref_k3: np.ndarray
    ref_cp_t: np.ndarray
    ref_cp_h: np.ndarray
    ref_l1  : np.ndarray
    ref_l2  : np.ndarray
    # ThermoLiq block:
    v_liq     : np.ndarray   # J/bar, at (Trl, Pr)
    dvdt      : np.ndarray
    dvdp      : np.ndarray
    d2vdtp    : np.ndarray
    d2vdp2    : np.ndarray
    t_fusion  : np.ndarray   # K
    s_fusion  : np.ndarray   # J/(mol K)
    cp_liquid : np.ndarray   # J/(mol K), constant (not T-dependent)
    label_index: dict = field(default_factory=dict)

    @property
    def nspec(self) -> int:
        return len(self.labels)

    def index(self, names) -> np.ndarray:
        return np.array([self.label_index[n] for n in names], dtype=np.intp)

    def select(self, names) -> "MELTSLiquidParams":
        idx = self.index(names)
        return MELTSLiquidParams(
            labels=[self.labels[i] for i in idx],
            ref_h=self.ref_h[idx], ref_s=self.ref_s[idx],
            ref_k0=self.ref_k0[idx], ref_k1=self.ref_k1[idx],
            ref_k2=self.ref_k2[idx], ref_k3=self.ref_k3[idx],
            ref_cp_t=self.ref_cp_t[idx], ref_cp_h=self.ref_cp_h[idx],
            ref_l1=self.ref_l1[idx], ref_l2=self.ref_l2[idx],
            v_liq=self.v_liq[idx], dvdt=self.dvdt[idx], dvdp=self.dvdp[idx],
            d2vdtp=self.d2vdtp[idx], d2vdp2=self.d2vdp2[idx],
            t_fusion=self.t_fusion[idx], s_fusion=self.s_fusion[idx],
            cp_liquid=self.cp_liquid[idx],
            label_index={n: i for i, n in enumerate([self.labels[i] for i in idx])},
        )


def load_liquid(json_path: str | Path | None = None, table: str = DEFAULT_TABLE) -> MELTSLiquidParams:
    path = Path(json_path) if json_path is not None else _DEFAULT_JSON
    with open(path) as fh:
        all_tables = json.load(fh)
    if table not in all_tables:
        raise KeyError(f"table {table!r} not in {sorted(all_tables)} ({path})")
    records = all_tables[table]

    n = len(records)
    labels = [r["label"] for r in records]

    def arr(key):
        return np.array([r[key] for r in records], dtype=np.float64)

    ref_k = np.array([r["ref_cp_coeffs"] for r in records], dtype=np.float64)
    kress = np.array([r["kress_coeffs"] for r in records], dtype=np.float64)

    label_index = {}
    for i, lbl in enumerate(labels):
        label_index.setdefault(lbl, i)

    return MELTSLiquidParams(
        labels=labels,
        ref_h=arr("ref_h"), ref_s=arr("ref_s"),
        ref_k0=ref_k[:, 0], ref_k1=ref_k[:, 1], ref_k2=ref_k[:, 2], ref_k3=ref_k[:, 3],
        ref_cp_t=ref_k[:, 4], ref_cp_h=ref_k[:, 5], ref_l1=ref_k[:, 6], ref_l2=ref_k[:, 7],
        v_liq=arr("v_liq"),
        dvdt=kress[:, 0], dvdp=kress[:, 1], d2vdtp=kress[:, 2], d2vdp2=kress[:, 3],
        t_fusion=arr("t_fusion"), s_fusion=arr("s_fusion"), cp_liquid=arr("cp_liquid"),
        label_index=label_index,
    )
