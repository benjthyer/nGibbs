"""
Process major-element compositions from CompleteGardEtAl2019.csv.

Filters to rgroup_id in [82, 86, 87, 88], computes a clean oxide table,
renormalises to 100 wt%, and produces summary histograms.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Allow running from any working directory
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ngibbs.config.constants import OXIDE_MOLAR_MASSES

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CSV_PATH = REPO_ROOT / "data" / "MELTStables" / "GEOROC" / "CompleteGardEtAl2019.csv"
OUT_CSV  = REPO_ROOT / "data" / "MELTStables" / "GEOROC" / "Gard2019_processed.csv"
OUT_FIG  = REPO_ROOT / "data" / "MELTStables" / "GEOROC" / "Gard2019_histograms.png"

# ---------------------------------------------------------------------------
# Molar mass constants
# ---------------------------------------------------------------------------
MW_FeO   = OXIDE_MOLAR_MASSES["FeO"]      # 71.844
MW_Fe2O3 = OXIDE_MOLAR_MASSES["Fe2O3"]    # 159.688
MW_CaO   = OXIDE_MOLAR_MASSES["CaO"]      # 56.0774
MW_MgO   = OXIDE_MOLAR_MASSES["MgO"]      # 40.3044
MW_CO2   = OXIDE_MOLAR_MASSES["CO2"]      # 44.0095
# Carbonate molar masses (not in OXIDE_MOLAR_MASSES)
MW_CaCO3 = 100.0869   # Ca(40.078) + C(12.011) + 3*O(15.999)
MW_MgCO3 =  84.3139   # Mg(24.305) + C(12.011) + 3*O(15.999)

FE2O3_TO_FEO = (2 * MW_FeO) / MW_Fe2O3   # ≈ 0.8998

# ---------------------------------------------------------------------------
# Load and filter
# ---------------------------------------------------------------------------
print(f"Reading {CSV_PATH} …")
df = pd.read_csv(CSV_PATH, low_memory=False, encoding="latin-1")
df.columns = df.columns.str.strip().str.lower()

RGROUP_IDS = [82, 86, 87, 88]
df = df[df["rgroup_id"].isin(RGROUP_IDS)].copy()
print(f"  {len(df)} rows after rgroup_id filter")

# Keep a working copy with float columns we will build up
def to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

# Pull raw columns (all as float, NaN where absent/blank)
raw = {
    "feo":       to_float(df.get("feo",       pd.Series(dtype=float, index=df.index))),
    "feo_tot":   to_float(df.get("feo_tot",   pd.Series(dtype=float, index=df.index))),
    "fe2o3":     to_float(df.get("fe2o3",     pd.Series(dtype=float, index=df.index))),
    "fe2o3_tot": to_float(df.get("fe2o3_tot", pd.Series(dtype=float, index=df.index))),
    "sio2":      to_float(df.get("sio2",      pd.Series(dtype=float, index=df.index))),
    "tio2":      to_float(df.get("tio2",      pd.Series(dtype=float, index=df.index))),
    "al2o3":     to_float(df.get("al2o3",     pd.Series(dtype=float, index=df.index))),
    "cr2o3":     to_float(df.get("cr2o3",     pd.Series(dtype=float, index=df.index))),
    "mgo":       to_float(df.get("mgo",       pd.Series(dtype=float, index=df.index))),
    "cao":       to_float(df.get("cao",       pd.Series(dtype=float, index=df.index))),
    "mno":       to_float(df.get("mno",       pd.Series(dtype=float, index=df.index))),
    "nio":       to_float(df.get("nio",       pd.Series(dtype=float, index=df.index))),
    "k2o":       to_float(df.get("k2o",       pd.Series(dtype=float, index=df.index))),
    "na2o":      to_float(df.get("na2o",      pd.Series(dtype=float, index=df.index))),
    "p2o5":      to_float(df.get("p2o5",      pd.Series(dtype=float, index=df.index))),
    "h2o_tot":   to_float(df.get("h2o_tot",   pd.Series(dtype=float, index=df.index))),
    "co2":       to_float(df.get("co2",       pd.Series(dtype=float, index=df.index))),
    "caco3":     to_float(df.get("caco3",     pd.Series(dtype=float, index=df.index))),
    "mgco3":     to_float(df.get("mgco3",     pd.Series(dtype=float, index=df.index))),
    "loi":       to_float(df.get("loi",       pd.Series(dtype=float, index=df.index))),
}

# ---------------------------------------------------------------------------
# 1. Total iron as FeO
# ---------------------------------------------------------------------------
# Priority: feo_tot ->fe2o3_tot→FeO ->fe2o3→FeO + feo
feo_total = pd.Series(np.nan, index=df.index)

# Lowest priority: component sum
has_fe2o3 = raw["fe2o3"].notna()
has_feo   = raw["feo"].notna()
feo_total = np.where(
    has_fe2o3 & has_feo,
    raw["fe2o3"] * FE2O3_TO_FEO + raw["feo"],
    np.where(has_feo,   raw["feo"],
    np.where(has_fe2o3, raw["fe2o3"] * FE2O3_TO_FEO, np.nan))
)
feo_total = pd.Series(feo_total, index=df.index)

# Override with fe2o3_tot if present
mask = raw["fe2o3_tot"].notna()
feo_total[mask] = raw["fe2o3_tot"][mask] * FE2O3_TO_FEO

# Override with feo_tot if present (highest priority)
mask = raw["feo_tot"].notna()
feo_total[mask] = raw["feo_tot"][mask]

# ---------------------------------------------------------------------------
# 2. Decompose CaCO3 and MgCO3 into CaO, MgO, CO2
# ---------------------------------------------------------------------------
cao = raw["cao"].fillna(0.0).copy()
mgo = raw["mgo"].fillna(0.0).copy()
co2 = raw["co2"].copy()   # may be NaN

caco3 = raw["caco3"].fillna(0.0)
mgco3 = raw["mgco3"].fillna(0.0)

cao += caco3 * (MW_CaO  / MW_CaCO3)
mgo += mgco3 * (MW_MgO  / MW_MgCO3)

co2_from_carbonates = caco3 * (MW_CO2 / MW_CaCO3) + mgco3 * (MW_CO2 / MW_MgCO3)

# Add carbonate-derived CO2 to existing CO2 (treat NaN as 0 for the add)
co2_new = co2.fillna(0.0) + co2_from_carbonates
# But if neither source was present keep NaN
had_co2 = co2.notna() | (caco3 > 0) | (mgco3 > 0)
co2_new = co2_new.where(had_co2, other=np.nan)

# ---------------------------------------------------------------------------
# 3. LOI ->CO2 if CO2 still undefined and LOI > 5 wt%
# ---------------------------------------------------------------------------
loi = raw["loi"]
use_loi = co2_new.isna() & loi.notna() & (loi > 5.0)
co2_new = co2_new.where(~use_loi, other=loi)

# ---------------------------------------------------------------------------
# 4. Assemble output columns (NaN where no data)
# ---------------------------------------------------------------------------
out = pd.DataFrame(index=df.index)
out["rgroup_id"] = df["rgroup_id"].values
out["sample_id"] = df["sample_id"].values

out["SiO2"]  = raw["sio2"]
out["TiO2"]  = raw["tio2"]
out["Al2O3"] = raw["al2o3"]
out["Cr2O3"] = raw["cr2o3"]
out["FeO"]   = feo_total
out["MgO"]   = mgo.where(mgo > 0, other=raw["mgo"])   # keep NaN if no MgO at all
out["CaO"]   = cao.where(cao > 0, other=raw["cao"])
out["MnO"]   = raw["mno"]
out["NiO"]   = raw["nio"]
out["K2O"]   = raw["k2o"]
out["Na2O"]  = raw["na2o"]
out["P2O5"]  = raw["p2o5"]
out["H2O"]   = raw["h2o_tot"]
out["CO2"]   = co2_new

# Restore genuine zeros for MgO/CaO that were zero to begin with
out["MgO"] = mgo.where(raw["mgo"].notna() | (mgco3 > 0), other=raw["mgo"])
out["CaO"] = cao.where(raw["cao"].notna() | (caco3 > 0), other=raw["cao"])

OXIDE_COLS = ["SiO2", "TiO2", "Al2O3", "Cr2O3", "FeO", "MgO", "CaO",
              "MnO", "NiO", "K2O", "Na2O", "P2O5", "H2O", "CO2"]

# ---------------------------------------------------------------------------
# 5. Oxide sum, filter 85–115%, renormalise to 100%
# ---------------------------------------------------------------------------
oxide_sum = out[OXIDE_COLS].fillna(0.0).sum(axis=1)
keep = (oxide_sum >= 85.0) & (oxide_sum <= 115.0)
print(f"  {keep.sum()} / {len(out)} rows pass the 85–115 wt% oxide sum filter")

out = out[keep].copy()
oxide_sum_kept = oxide_sum[keep]

# Drop rows that report fewer than 5 of the 14 oxide columns
reported = out[OXIDE_COLS].notna().sum(axis=1)
enough = reported >= 5
print(f"  {(~enough).sum()} rows removed for reporting fewer than 5 oxides; {enough.sum()} remaining")
out = out[enough].copy()
oxide_sum_kept = oxide_sum_kept[enough]

for col in OXIDE_COLS:
    out[col] = out[col].fillna(0.0) / oxide_sum_kept.values * 100.0

out["oxide_sum_prenorm"] = oxide_sum_kept.values

# Zero out any negative values
for col in OXIDE_COLS:
    out[col] = out[col].clip(lower=0.0)

# Drop compositions with H2O > 10 wt% (post-renorm)
pre_h2o = len(out)
out = out[out["H2O"].fillna(0.0) <= 10.0].copy()
print(f"  {pre_h2o - len(out)} rows removed for H2O > 10 wt%; {len(out)} remaining")

# ---------------------------------------------------------------------------
# 6. Save CSVs (full + 80/20 train/validation split)
# ---------------------------------------------------------------------------
OUT_TRAIN = REPO_ROOT / "data" / "MELTStables" / "GEOROC" / "Gard2019_train.csv"
OUT_VAL   = REPO_ROOT / "data" / "MELTStables" / "GEOROC" / "Gard2019_validation.csv"

val = out.sample(frac=0.2, random_state=42)
train = out.drop(val.index)

out.to_csv(OUT_CSV, index=False)
train.to_csv(OUT_TRAIN, index=False)
val.to_csv(OUT_VAL, index=False)
print(f"  Saved {len(out)} rows -> {OUT_CSV}")
print(f"  Train: {len(train)} rows -> {OUT_TRAIN}")
print(f"  Validation: {len(val)} rows -> {OUT_VAL}")

# ---------------------------------------------------------------------------
# 7. Histograms
# ---------------------------------------------------------------------------
RGROUP_COLORS = {82: "#4C72B0", 86: "#DD8452", 87: "#55A868", 88: "#C44E52"}
RGROUP_LABELS = {g: f"rgroup {g}" for g in RGROUP_IDS}

n_oxide = len(OXIDE_COLS)
n_cols  = 4
n_rows  = (n_oxide + 1 + n_cols - 1) // n_cols   # +1 for rgroup bar chart

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.2))
axes = axes.flatten()

# --- rgroup_id bar chart (categorical) ---
ax = axes[0]
counts = out["rgroup_id"].value_counts().reindex(RGROUP_IDS, fill_value=0)
bars = ax.bar(
    [str(g) for g in RGROUP_IDS],
    counts.values,
    color=[RGROUP_COLORS[g] for g in RGROUP_IDS],
    edgecolor="black", linewidth=0.6
)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            str(val), ha="center", va="bottom", fontsize=8)
ax.set_title("rgroup_id", fontsize=10)
ax.set_xlabel("rgroup_id")
ax.set_ylabel("count")

# --- Oxide histograms ---
for i, col in enumerate(OXIDE_COLS):
    ax = axes[i + 1]
    for grp_id in RGROUP_IDS:
        subset = out.loc[out["rgroup_id"] == grp_id, col].dropna()
        if len(subset) == 0:
            continue
        ax.hist(subset, bins=30, alpha=0.55,
                color=RGROUP_COLORS[grp_id], label=RGROUP_LABELS[grp_id],
                edgecolor="none")
    ax.set_title(f"{col} (wt%)", fontsize=10)
    ax.set_xlabel("wt%")
    ax.set_ylabel("count")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

# Legend on last used oxide axis
handles = [plt.Rectangle((0, 0), 1, 1, color=RGROUP_COLORS[g], alpha=0.7)
           for g in RGROUP_IDS]
axes[1].legend(handles, [RGROUP_LABELS[g] for g in RGROUP_IDS],
               fontsize=7, loc="upper right")

# Hide unused axes
for ax in axes[n_oxide + 1:]:
    ax.set_visible(False)

fig.suptitle("Gard et al. 2019 - renormalised major-element compositions", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
print(f"  Saved histogram figure -> {OUT_FIG}")
plt.close(fig)

# ---------------------------------------------------------------------------
# 8. SiO2 vs CO2 scatter, coloured by FeO (wt%)
# ---------------------------------------------------------------------------
OUT_SCATTER = REPO_ROOT / "data" / "MELTStables" / "GEOROC" / "Gard2019_SiO2_CO2_scatter.png"
scatter_data = out[["SiO2", "CO2", "FeO"]].dropna()
fig2, ax2 = plt.subplots(figsize=(7.5, 5))
sc = ax2.scatter(
    scatter_data["SiO2"], scatter_data["CO2"],
    c=scatter_data["FeO"], cmap="plasma",
    s=8, alpha=0.5, linewidths=0,
)
cbar = fig2.colorbar(sc, ax=ax2)
cbar.set_label("FeO (wt%)", fontsize=9)
ax2.set_xlabel("SiO2 (wt%)")
ax2.set_ylabel("CO2 (wt%)")
ax2.set_title("SiO2 vs CO2 - Gard et al. 2019")
fig2.tight_layout()
fig2.savefig(OUT_SCATTER, dpi=150, bbox_inches="tight")
print(f"  Saved scatter figure -> {OUT_SCATTER}")
plt.close(fig2)
