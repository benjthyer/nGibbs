#!/usr/bin/env python3
"""
Benchmark melts_vec against RAW MELTS output tables (not the aggregated
MELTStables CSV `reproduce_meltstables.py` uses).

Ben's request: the MELTStables CSV export re-normalizes every step onto a
100 g bulk-system basis, and real MELTS runs are never exactly conserved to
100 g (closed-system Fe3+/FeT bookkeeping alone drifts the total by up to
~1%) -- so a benchmark built on that CSV has an extra, avoidable source of
noise baked in on top of whatever melts_vec itself gets wrong. The raw
per-phase `.tbl` files each MELTS run writes directly (one row per
equilibrium step, one file per phase) sidestep that entirely: each row
already carries its own mass (g), density (g/cc), extensive G/H/S/V/Cp
(kJ/kJ/J/K/cc/J/K) for exactly that mass of that phase, and the phase's own
endmember mole fractions -- nothing renormalized, nothing re-aggregated.

Data layout: `data/{pMELTS,MELTS102,MELTS120}_RawOutputs/Simulation<i>/`
(i = 0..49), each one isobaric-cooling run. About 15% of runs fail outright
(no `System_main_tbl.txt` written at all) -- skipped, and counted in the
run-level summary this script prints. Within a surviving run, `<phase>.tbl`
exists only for phases that saturate along that particular cooling path, so
which phases get how many rows varies run to run; this script pools rows
across all 50 runs (all 3 model variants), phase by phase.

Units in the raw files: T in Celsius (+273.15 -> K), P in KBARS (x1000 ->
bars -- NOTE this differs from `reproduce_meltstables.py`'s source CSV,
which already reports P in bars), mass in g, rho in g/cc, G/H/S/V/Cp
EXTENSIVE for that row's own mass (kJ, kJ, J/K, cc, J/K).

Comparison method: identical in spirit to `reproduce_meltstables.py` --
convert both sides to per-gram (specific, intensive) quantities so no
"total moles of phase" bookkeeping or `sol_struct_data.json` molar-mass gap
is in the way (`melts_vec.molar_mass` supplies MW(g/mol) from each
endmember's own formula string):

    our_V_cc_per_mol = V_Jbar * 10.0
    our_rho          = MW / our_V_cc_per_mol
    our_H_specific   = (H_J_per_mol / 1000) / MW      (kJ/g, matches table's own kJ)
    our_S_specific   = S_J_per_molK / MW               (J/(K g))
    our_Cp_specific  = Cp_J_per_molK / MW              (J/(K g))

Coverage: every melts_vec solid-solution phase (olivine, orthopyroxene,
clinopyroxene, spinel, plagioclase, alkali-feldspar, rhm-oxide -- note
rhm-oxide has real nonzero-mass rows here, unlike the MELTStables CSV used
in `reproduce_meltstables.py`, which never saturated it) plus four PURE
phases with no mixing model at all (quartz, tridymite, whitlockite,
apatite) checked directly against `melts_vec.compute()`'s pure-endmember
EOS -- a bonus check of the underlying solid EOS itself against real MELTS
output, not just the mixing models. Liquid is deliberately NOT included:
converting the table's raw bulk-oxide composition into meltsLiquid's own
19-component basis is MAGMA's `conLiq()` (an fO2-dependent Fe2+/Fe3+
partition), which is a separate, not-yet-implemented translation task (see
the project plan doc's Task #5 notes) -- there is no way to build a correct
liquid input composition without it, so testing liquid here would just be
testing that gap, not the liquid EOS itself.

Output: one CSV, one row per (source, phase), columns matching the
project's existing `deployment_test` quality-table convention (n, then
<property> mae / mean_rel% / p95_rel% / max_rel% for rho, H, S, V, Cp).
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent  # EOS_arithmetic_MELTS/
sys.path.insert(0, str(repo_root.parent))
sys.path.insert(0, str(repo_root))

from melts_vec.params import load_solids
from melts_vec.molar_mass import molar_masses
from melts_vec import (
    feldspar as _feldspar, olivine as _olivine,
    clinopyroxene as _clinopyroxene, orthopyroxene as _orthopyroxene,
    spinel as _spinel, rhomsghiorso as _rhomsghiorso,
    compute as compute_pure,
    compute_feldspar_solution, compute_olivine_solution,
    compute_clinopyroxene_solution, compute_orthopyroxene_solution,
    compute_spinel_solution, compute_rhm_oxide_solution,
)

# nGibbs/data/ -- 4 .parent above EOS_arithmetic_MELTS (…/src/ngibbs/engine/EOS_arithmetic_MELTS)
DEFAULT_DATA_ROOT = repo_root.parent.parent.parent.parent / "data"

MODEL_DIRS = {
    "pMELTS_RawOutputs":  "pMELTS",
    "MELTS102_RawOutputs": "MELTS1.0.2",
    "MELTS120_RawOutputs": "MELTS1.2.0",
}

SOLUTION_PHASES = {
    "olivine":         (compute_olivine_solution, _olivine.ENDMEMBERS),
    "orthopyroxene":   (compute_orthopyroxene_solution, _orthopyroxene.ENDMEMBERS),
    "clinopyroxene":   (compute_clinopyroxene_solution, _clinopyroxene.ENDMEMBERS),
    "spinel":          (compute_spinel_solution, _spinel.ENDMEMBERS),
    "plagioclase":     (compute_feldspar_solution, _feldspar.ENDMEMBERS),
    "alkali-feldspar": (compute_feldspar_solution, _feldspar.ENDMEMBERS),
    "rhm-oxide":       (compute_rhm_oxide_solution, _rhomsghiorso.ENDMEMBERS),
}

# Pure phases: no mixing model exists or is needed (Ben's own note on
# whitlockite/apatite) -- checked directly against compute()'s pure-
# endmember EOS, which was already C-harness-verified elsewhere but had
# never been cross-checked against real MELTS run output before this pass.
PURE_PHASES = ["quartz", "tridymite", "whitlockite", "apatite"]

MASS_THRESHOLD_G = 0.01   # skip rows where the phase mass is negligible/at the numerical floor


def _run_is_valid(sim_dir: Path) -> bool:
    """A run's own failure signature: no System_main_tbl.txt at all (~15%
    of runs, per Ben's estimate -- confirmed by direct count, see main())."""
    f = sim_dir / "System_main_tbl.txt"
    return f.exists() and f.stat().st_size > 0


def _read_phase_tbl(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if len(df) == 0 or "mass (gm)" not in df.columns:
        return None
    df = df[df["mass (gm)"] > MASS_THRESHOLD_G].copy()
    if len(df) == 0:
        return None
    df["T_K"] = df["T (C)"].values + 273.15
    df["P_bar"] = df["P (kbars)"].values * 1000.0
    return df


def _build_X(df: pd.DataFrame, endmembers: list[str]) -> np.ndarray:
    X = np.zeros((len(df), len(endmembers)), dtype=np.float64)
    for j, name in enumerate(endmembers):
        if name in df.columns:
            X[:, j] = df[name].values
    return X


def _relerr(ours, table):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(ours - table)/np.abs(table)


def _stats(err: np.ndarray) -> tuple[float, float, float]:
    finite = err[np.isfinite(err)]
    if len(finite) == 0:
        return (np.nan, np.nan, np.nan)
    return (float(np.mean(finite)*100), float(np.percentile(finite, 95)*100), float(np.max(finite)*100))


def process_solution_phase(df: pd.DataFrame, phase_key: str, solid_params) -> dict:
    compute_fn, endmembers = SOLUTION_PHASES[phase_key]
    T_K = df["T_K"].values
    P_bar = df["P_bar"].values
    X = _build_X(df, endmembers)

    mw_endmembers = molar_masses(solid_params, endmembers)
    MW = X @ mw_endmembers

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compute_fn(T_K, P_bar, X, solid_params)

    our_V_cc = result["V"]*10.0
    our_rho = MW/our_V_cc
    our_H_specific = (result["H"]/1000.0)/MW     # kJ/g
    our_S_specific = result["S"]/MW               # J/(K g)
    our_Cp_specific = result["Cp"]/MW              # J/(K g)

    table_mass = df["mass (gm)"].values
    table_rho = df["rho (gm/cc)"].values
    table_H_specific = df["H (kJ)"].values/table_mass
    table_S_specific = df["S (J/K)"].values/table_mass
    table_V_specific = df["V (cc)"].values/table_mass
    table_Cp_specific = df["Cp (J/K)"].values/table_mass
    our_V_specific = our_V_cc/MW

    return dict(
        n=len(df),
        rho=(np.abs(our_rho - table_rho), _relerr(our_rho, table_rho)),
        H=(np.abs(our_H_specific - table_H_specific), _relerr(our_H_specific, table_H_specific)),
        S=(np.abs(our_S_specific - table_S_specific), _relerr(our_S_specific, table_S_specific)),
        V=(np.abs(our_V_specific - table_V_specific), _relerr(our_V_specific, table_V_specific)),
        Cp=(np.abs(our_Cp_specific - table_Cp_specific), _relerr(our_Cp_specific, table_Cp_specific)),
    )


def process_pure_phase(df: pd.DataFrame, phase_key: str, solid_params) -> dict:
    T_K = df["T_K"].values
    P_bar = df["P_bar"].values
    MW = float(molar_masses(solid_params, [phase_key])[0])

    result = compute_pure(T_K, P_bar, solid_params, names=[phase_key])
    our_V_cc = result["V"][:, 0]*10.0
    our_rho = MW/our_V_cc
    our_H_specific = (result["H"][:, 0]/1000.0)/MW
    our_S_specific = result["S"][:, 0]/MW
    our_Cp_specific = result["Cp"][:, 0]/MW
    our_V_specific = our_V_cc/MW

    table_mass = df["mass (gm)"].values
    table_rho = df["rho (gm/cc)"].values
    table_H_specific = df["H (kJ)"].values/table_mass
    table_S_specific = df["S (J/K)"].values/table_mass
    table_V_specific = df["V (cc)"].values/table_mass
    table_Cp_specific = df["Cp (J/K)"].values/table_mass

    return dict(
        n=len(df),
        rho=(np.abs(our_rho - table_rho), _relerr(our_rho, table_rho)),
        H=(np.abs(our_H_specific - table_H_specific), _relerr(our_H_specific, table_H_specific)),
        S=(np.abs(our_S_specific - table_S_specific), _relerr(our_S_specific, table_S_specific)),
        V=(np.abs(our_V_specific - table_V_specific), _relerr(our_V_specific, table_V_specific)),
        Cp=(np.abs(our_Cp_specific - table_Cp_specific), _relerr(our_Cp_specific, table_Cp_specific)),
    )


def main(data_root: str | Path = DEFAULT_DATA_ROOT, n_sims: int = 50,
         out_csv: str | Path = "melts_raw_property_errors.csv"):
    data_root = Path(data_root)
    solid_params = load_solids()

    all_phases = {**{k: None for k in SOLUTION_PHASES}, **{k: None for k in PURE_PHASES}}
    # accumulator[(source, phase)] -> dict(n=int, rho=[abs_err arrays], rho_rel=[...], ...)
    acc: dict[tuple[str, str], dict] = {}
    run_counts = {}   # source -> (n_valid, n_total)

    for model_dir, source_label in MODEL_DIRS.items():
        root = data_root / model_dir
        if not root.is_dir():
            print(f"[warn] {root} not found, skipping {source_label}")
            continue
        sim_dirs = sorted(root.glob("Simulation*"),
                           key=lambda p: int(p.name.replace("Simulation", "")))
        sim_dirs = sim_dirs[:n_sims]
        n_valid = 0
        for sim_dir in sim_dirs:
            if not _run_is_valid(sim_dir):
                continue
            n_valid += 1
            for phase_key in all_phases:
                tbl_path = sim_dir / f"{phase_key}.tbl"
                if not tbl_path.exists():
                    continue
                df = _read_phase_tbl(tbl_path)
                if df is None:
                    continue
                if phase_key in SOLUTION_PHASES:
                    res = process_solution_phase(df, phase_key, solid_params)
                else:
                    res = process_pure_phase(df, phase_key, solid_params)

                key = (source_label, phase_key)
                if key not in acc:
                    acc[key] = dict(n=0, rho=[], rho_rel=[], H=[], H_rel=[],
                                     S=[], S_rel=[], V=[], V_rel=[], Cp=[], Cp_rel=[])
                a = acc[key]
                a["n"] += res["n"]
                for prop in ("rho", "H", "S", "V", "Cp"):
                    abs_err, rel_err = res[prop]
                    a[prop].append(abs_err)
                    a[f"{prop}_rel"].append(rel_err)
        run_counts[source_label] = (n_valid, len(sim_dirs))

    print("=" * 78)
    print("Run-level yield (raw MELTS outputs)")
    print("=" * 78)
    for source_label, (n_valid, n_total) in run_counts.items():
        fail_pct = 100.0*(n_total - n_valid)/n_total if n_total else float("nan")
        print(f"{source_label:12s}  {n_valid}/{n_total} runs valid  "
              f"({fail_pct:.1f}% failed -- no System_main_tbl.txt written)")

    rows = []
    for (source_label, phase_key), a in acc.items():
        if a["n"] == 0:
            continue
        row = {"source": source_label, "phase": phase_key, "n": a["n"]}
        for prop in ("rho", "H", "S", "V", "Cp"):
            abs_all = np.concatenate(a[prop])
            rel_all = np.concatenate(a[f"{prop}_rel"])
            mae = float(np.mean(abs_all[np.isfinite(abs_all)])) if np.any(np.isfinite(abs_all)) else np.nan
            mean_rel, p95_rel, max_rel = _stats(rel_all)
            row[f"{prop} mae"] = mae
            row[f"{prop} mean_rel%"] = mean_rel
            row[f"{prop} p95_rel%"] = p95_rel
            row[f"{prop} max_rel%"] = max_rel
        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values(["source", "phase"]).reset_index(drop=True)
    out_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 78)
    print("Per-phase property errors (pooled across all valid runs, all 3 MELTS variants)")
    print("=" * 78)
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.4f}".format):
        print(out_df[["source", "phase", "n",
                       "rho mean_rel%", "H mean_rel%", "S mean_rel%", "V mean_rel%", "Cp mean_rel%"]])

    print(f"\nFull table (mae + mean/p95/max relative error for rho, H, S, V, Cp) written to {out_csv}")
    return out_df


if __name__ == "__main__":
    main()
