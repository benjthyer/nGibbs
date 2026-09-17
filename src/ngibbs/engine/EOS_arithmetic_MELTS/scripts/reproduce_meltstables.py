#!/usr/bin/env python3
"""
Reproduce MELTStables CSV bulk/phasewise properties with melts_vec.

Ben's request: rather than using PetThermoTools as the benchmarking-data
source (PTT is a live front end that runs actual MELTS/nGibbs calculations
-- useful for interactive cross-checks, but not a static reference table),
use an existing MELTStables CSV export (e.g.
`data/MELTStables/110/MELTS110_TrainsetFeb13BatchCooling.csv`, a
rhyolite-MELTS v1.0.2 batch-cooling run) as ground truth: it already
contains, per equilibrium step, each stable solid solution's own
component (endmember) mole fractions plus its bulk rho/H/S/V and mass
directly from MELTS itself. This script feeds those endmember mole
fractions through melts_vec's `compute_*_solution()` functions at the
table's own (T, P) and checks whether the results agree.

Units / comparison method
--------------------------
The table reports EXTENSIVE per-phase quantities on a 100 g bulk-system
basis (mass in g, H in kJ, S in J/K, V in cc), while melts_vec's
`compute_*_solution()` returns INTENSIVE per-mole quantities (V in J/bar,
H in J/mol, S in J/(mol K)) for one mole of the solution's own formula
unit. There is no verified per-endmember molar mass anywhere in the
extracted parameter tables (`sol_struct_data.json`'s `mw` field is
unpopulated) -- but every endmember record DOES carry its chemical
formula (e.g. "Mg2SiO4"), so `melts_vec.molar_mass` parses that directly
into a g/mol figure using standard atomic weights (internally consistent
with `ngibbs.config.constants.OXIDE_MOLAR_MASSES`, see that module's
docstring). With the mole-fraction-weighted molar mass MW(g/mol) of a row's
own composition in hand, per-mole and per-gram (specific) quantities
convert into each other with no separate "total moles of phase" bookkeeping
needed:

    our_V_cc_per_mol = V_Jbar * 10.0          (1 J/bar = 10 cm^3, standard)
    our_rho          = MW / our_V_cc_per_mol                    (g/cc)
    our_H_specific   = (H_J_per_mol / 1000) / MW                (kJ/g)
    our_S_specific   = S_J_per_molK / MW                        (J/(K g))

compared directly against the table's own `rho (gm/cc)(<phase>)` column
and against `H (kJ)(<phase>)/mass (gm)(<phase>)` /
`S (J/K)(<phase>)/mass (gm)(<phase>)` -- both intensive, so this
comparison never depends on the table's 100 g normalization or on knowing
the phase's total mole count.

Coverage
--------
olivine, orthopyroxene, clinopyroxene, spinel, plagioclase, and
k-feldspar (plagioclase/k-feldspar are the same feldspar mixing model,
just fed the table's separate plagioclase/k-feldspar columns) all appear
with nonzero mass somewhere in the Feb13 batch-cooling table and are
checked here. rhm-oxide never saturates in this particular run (0 rows
with nonzero mass -- not surprising for a single cooling path at moderate
fO2/TiO2) but is wired in for whichever table it turns up in later.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent  # EOS_arithmetic_MELTS/
sys.path.insert(0, str(repo_root.parent))            # so `import EOS_arithmetic_MELTS` works
sys.path.insert(0, str(repo_root))                    # so `import melts_vec` works directly too

from melts_vec.params import load_solids
from melts_vec.molar_mass import molar_masses
from melts_vec import (
    feldspar as _feldspar, olivine as _olivine,
    clinopyroxene as _clinopyroxene, orthopyroxene as _orthopyroxene,
    spinel as _spinel, rhomsghiorso as _rhomsghiorso,
    compute_feldspar_solution, compute_olivine_solution,
    compute_clinopyroxene_solution, compute_orthopyroxene_solution,
    compute_spinel_solution, compute_rhm_oxide_solution,
)

DEFAULT_CSV = (repo_root.parent.parent.parent.parent
               / "data" / "MELTStables" / "110"
               / "MELTS110_TrainsetFeb13BatchCooling.csv")

# phase_key -> (csv column-suffix used in "<endmember>(<suffix>)" /
#               "mass (gm)(<suffix>)" etc., compute_fn, ENDMEMBERS list in
#               the order compute_fn expects)
PHASE_SPECS = {
    "olivine":       ("olivine",       compute_olivine_solution,       _olivine.ENDMEMBERS),
    "orthopyroxene": ("orthopyroxene", compute_orthopyroxene_solution, _orthopyroxene.ENDMEMBERS),
    "clinopyroxene": ("clinopyroxene", compute_clinopyroxene_solution, _clinopyroxene.ENDMEMBERS),
    "spinel":        ("spinel",        compute_spinel_solution,        _spinel.ENDMEMBERS),
    "plagioclase":   ("plagioclase",   compute_feldspar_solution,      _feldspar.ENDMEMBERS),
    "k-feldspar":    ("k-feldspar",    compute_feldspar_solution,      _feldspar.ENDMEMBERS),
    "rhm-oxide":     ("rhm-oxide",     compute_rhm_oxide_solution,     _rhomsghiorso.ENDMEMBERS),
}


def build_X(df: pd.DataFrame, idx, csv_suffix: str, endmembers: list[str]) -> np.ndarray:
    """(len(idx), len(endmembers)) mole-fraction matrix, reading each
    endmember's own CSV column by name and defaulting to 0 for any
    endmember melts_vec models but this table's phase doesn't carry a
    column for (e.g. olivine's co-olivine)."""
    X = np.zeros((len(idx), len(endmembers)), dtype=np.float64)
    for j, name in enumerate(endmembers):
        col = f"{name}({csv_suffix})"
        if col in df.columns:
            X[:, j] = df.loc[idx, col].values
    return X


def run_phase(df: pd.DataFrame, solid_params, phase_key: str,
              mass_threshold: float = 1.0, max_rows: int = 300,
              seed: int = 0) -> dict | None:
    csv_suffix, compute_fn, endmembers = PHASE_SPECS[phase_key]
    mass_col = f"mass (gm)({csv_suffix})"
    if mass_col not in df.columns:
        return None
    idx_all = df.index[df[mass_col] > mass_threshold]
    if len(idx_all) == 0:
        return {"phase": phase_key, "n_rows": 0}
    if len(idx_all) > max_rows:
        rng = np.random.default_rng(seed)
        idx = pd.Index(rng.choice(idx_all.to_numpy(), size=max_rows, replace=False))
    else:
        idx = idx_all

    T_K = df.loc[idx, "Temperature(System_main)"].values + 273.15
    P_bar = df.loc[idx, "Pressure(System_main)"].values
    X = build_X(df, idx, csv_suffix, endmembers)

    mw_endmembers = molar_masses(solid_params, endmembers)  # (N,) g/mol
    MW = X @ mw_endmembers                                   # (B,) g/mol, row composition

    result = compute_fn(T_K, P_bar, X, solid_params)

    our_V_cc = result["V"]*10.0
    our_rho = MW/our_V_cc
    our_H_specific = (result["H"]/1000.0)/MW   # kJ/g
    our_S_specific = result["S"]/MW            # J/(K g)

    table_mass = df.loc[idx, mass_col].values
    table_rho = df.loc[idx, f"rho (gm/cc)({csv_suffix})"].values
    table_H_specific = df.loc[idx, f"H (kJ)({csv_suffix})"].values/table_mass
    table_S_specific = df.loc[idx, f"S (J/K)({csv_suffix})"].values/table_mass

    def relerr(ours, table):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.abs(ours - table)/np.abs(table)

    err_rho = relerr(our_rho, table_rho)
    err_H = relerr(our_H_specific, table_H_specific)
    err_S = relerr(our_S_specific, table_S_specific)

    return dict(
        phase=phase_key, n_rows=len(idx),
        idx=idx, T_K=T_K, P_bar=P_bar, MW=MW,
        our_rho=our_rho, table_rho=table_rho, err_rho=err_rho,
        our_H_specific=our_H_specific, table_H_specific=table_H_specific, err_H=err_H,
        our_S_specific=our_S_specific, table_S_specific=table_S_specific, err_S=err_S,
    )


def summarize(res: dict) -> str:
    if res is None or res["n_rows"] == 0:
        return f"{res['phase'] if res else '?':14s}  no rows with nonzero mass in this table"

    def stats(a):
        finite = a[np.isfinite(a)]
        if len(finite) == 0:
            return "n/a"
        return f"median={np.median(finite)*100:7.3f}%  mean={np.mean(finite)*100:7.3f}%  max={np.max(finite)*100:7.3f}%"

    return (f"{res['phase']:14s} n={res['n_rows']:4d}  "
            f"rho: {stats(res['err_rho'])}\n"
            f"{'':14s}         "
            f"H  : {stats(res['err_H'])}\n"
            f"{'':14s}         "
            f"S  : {stats(res['err_S'])}")


def main(csv_path: str | Path = DEFAULT_CSV, max_rows: int = 300, out_csv: str | Path | None = None):
    csv_path = Path(csv_path)
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    solid_params = load_solids()

    print("\n" + "="*78)
    print(f"Reproducing MELTStables phase properties with melts_vec ({csv_path.name})")
    print("="*78)

    all_results = []
    for phase_key in PHASE_SPECS:
        res = run_phase(df, solid_params, phase_key, max_rows=max_rows)
        all_results.append(res)
        print(summarize(res))

    if out_csv is not None:
        rows = []
        for res in all_results:
            if res is None or res["n_rows"] == 0:
                continue
            for i in range(res["n_rows"]):
                rows.append(dict(
                    phase=res["phase"], row=int(res["idx"][i]),
                    T_K=res["T_K"][i], P_bar=res["P_bar"][i], MW=res["MW"][i],
                    our_rho=res["our_rho"][i], table_rho=res["table_rho"][i], err_rho=res["err_rho"][i],
                    our_H_specific=res["our_H_specific"][i], table_H_specific=res["table_H_specific"][i], err_H=res["err_H"][i],
                    our_S_specific=res["our_S_specific"][i], table_S_specific=res["table_S_specific"][i], err_S=res["err_S"][i],
                ))
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\nPer-row results written to {out_csv}")

    return all_results


if __name__ == "__main__":
    main()
