"""
Unit test for NN_MELTS.Iron_Speciator().

Loads a MELTS BigMetaTable, selects rows where melt fraction F > 10%,
consolidates FeO + Fe2O3 into total-iron FeO equivalents, re-speciates
with Iron_Speciator (Kress & Carmichael 1991), then reports and asserts
on the disparity vs the original MELTS-reported FeO / Fe2O3 wt%.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
BUILDER_ROOT = SRC_ROOT / "builder"

for _p in (str(REPO_ROOT), str(SRC_ROOT), str(BUILDER_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from builder.processing.BigMetaTable import BigMetaTable  # noqa: E402
from ngibbs.engine.emulator import NN_MELTS  # noqa: E402


# ---------------------------------------------------------------------------
# Oxide / element metadata (standard MELTS ordering)
# ---------------------------------------------------------------------------

WR_OXIDES: List[str] = [
    "SiO2", "TiO2", "Al2O3", "FeO", "MgO", "CaO",
    "Na2O", "K2O", "P2O5", "H2O", "Cr2O3", "MnO", "NiO",
]
ELKEYS: List[str] = [
    "Si", "Ti", "Al", "Fe", "Mg", "Ca",
    "Na", "K",  "P",  "H",  "Cr", "Mn", "Ni",
]
OXIDE_DICT: Dict[str, int] = {ox: i for i, ox in enumerate(WR_OXIDES + ["Fe2O3"])}

MOLAR_MASSES: Dict[str, float] = {
    "SiO2":  60.0843,
    "TiO2":  79.866,
    "Al2O3": 101.9613,
    "FeO":   71.844,
    "MgO":   40.3044,
    "CaO":   56.0774,
    "Na2O":  61.9788,
    "K2O":   94.2,
    "P2O5":  141.9437,
    "H2O":   18.0152,
    "Cr2O3": 151.9902,
    "MnO":   70.9374,
    "NiO":   74.6928,
    "Fe2O3": 159.688,
}
MM_ARR = np.array(
    [MOLAR_MASSES[ox] for ox in WR_OXIDES + ["Fe2O3"]], dtype=np.float32
)


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

_CANDIDATE_TABLES = [
    REPO_ROOT / "data" / "MELTStables" / "110" / "MELTS110_TrainsetFeb13BatchCooling",
    REPO_ROOT / "data" / "MELTStables" / "110" / "MELTS110_ValidsetFeb13BatchCooling",
    REPO_ROOT / "data" / "MELTStables" / "110" / "MELTS110_TrainsetFeb3BatchCooling",
]


def _find_bmt_base() -> Path | None:
    for base in _CANDIDATE_TABLES:
        if (Path(str(base) + ".csv")).exists():
            return base
    return None


# ---------------------------------------------------------------------------
# Minimal mock emulator for Iron_Speciator
# ---------------------------------------------------------------------------

class _IdentityNorm:
    """Normalizer mock: denorm is identity (result unused when P/T/fO2 provided)."""
    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _FakeEmulator:
    """
    Provides only the attributes accessed by Iron_Speciator when P, T, and fO2
    are supplied explicitly (so no normalizer extraction is needed).
    """
    oxide_dict = OXIDE_DICT
    Elkeys = ELKEYS
    dev = "cpu"
    feature_offset = 3            # dummy; slice result is discarded when P/T/fO2 given
    norm_features = _IdentityNorm()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bmt_and_liquid_rows():
    """
    Load BigMetaTable and return (bmt, liquid_mask, liquid_rows array).
    Skips the test module if no suitable data file is found.
    """
    base = _find_bmt_base()
    if base is None:
        pytest.skip("No MELTS BigMetaTable CSV found under data/MELTStables/; skipping.")

    bmt = BigMetaTable(str(base))

    sys_idx  = bmt.indexer.MELTS_indices.get("System_main", {})
    liq_idx  = bmt.indexer.MELTS_indices.get("melts-liquid", {})

    if "F" not in sys_idx:
        pytest.skip("F(System_main) column not found in loaded BigMetaTable.")
    if "wt% FeO" not in liq_idx or "wt% Fe2O3" not in liq_idx:
        pytest.skip("wt% FeO or wt% Fe2O3 not found in melts-liquid columns.")

    liquid_mask = bmt.table[:, sys_idx["F"]] > 0.10
    if not liquid_mask.any():
        pytest.skip("No rows with F > 10% liquid in BigMetaTable.")

    return bmt, liquid_mask


def test_iron_speciator_conserves_total_iron(bmt_and_liquid_rows):
    """
    Iron_Speciator must conserve total Fe (as FeO-equiv moles) exactly.
    """
    bmt, liquid_mask = bmt_and_liquid_rows
    _check_conservation(bmt, liquid_mask)


def test_iron_speciator_disparity_vs_melts(bmt_and_liquid_rows):
    """
    Measure wt% MAE between Iron_Speciator (K&C 1991) and MELTS-reported
    FeO / Fe2O3.  Reports statistics and asserts a loose (<= 5 wt%) tolerance.
    """
    bmt, liquid_mask = bmt_and_liquid_rows
    _evaluate_disparity(bmt, liquid_mask)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_input_tensor(bmt: BigMetaTable, liquid_mask: np.ndarray):
    """
    Extract liquid-bearing rows, build:
      - oxide_moles  (n, 13) float32 tensor  – input to Iron_Speciator
      - total_iron   (n,)    float32 ndarray  – total FeO-equiv moles per row
      - orig_wt_FeO  (n,)    float32 ndarray  – original MELTS wt% FeO
      - orig_wt_Fe2O3(n,)    float32 ndarray  – original MELTS wt% Fe2O3
    """
    liq_idx  = bmt.indexer.MELTS_indices["melts-liquid"]
    sys_idx  = bmt.indexer.MELTS_indices["System_main"]
    rows     = bmt.table[liquid_mask]
    n        = rows.shape[0]

    # PT conditions
    P   = rows[:, sys_idx["Pressure"]].astype(np.float32)     # bars
    T   = rows[:, sys_idx["Temperature"]].astype(np.float32)  # °C
    fO2 = rows[:, sys_idx["logfO2-QFM"]].astype(np.float32)  # Δ log fO2 vs QFM

    # Original iron speciation (wt% within liquid)
    orig_wt_FeO   = rows[:, liq_idx["wt% FeO"]].astype(np.float32)
    orig_wt_Fe2O3 = rows[:, liq_idx["wt% Fe2O3"]].astype(np.float32)

    # Build oxide mole tensor (13 columns, WR_OXIDES ordering, no Fe2O3)
    oxide_moles = np.zeros((n, 13), dtype=np.float32)

    non_fe_oxides = [
        ("SiO2",  "wt% SiO2"),
        ("TiO2",  "wt% TiO2"),
        ("Al2O3", "wt% Al2O3"),
        ("MgO",   "wt% MgO"),
        ("CaO",   "wt% CaO"),
        ("Na2O",  "wt% Na2O"),
        ("K2O",   "wt% K2O"),
        ("P2O5",  "wt% P2O5"),
        ("H2O",   "wt% H2O"),
        ("Cr2O3", "wt% Cr2O3"),
        ("MnO",   "wt% MnO"),
        ("NiO",   "wt% NiO"),
    ]
    for oxide, col_key in non_fe_oxides:
        if col_key in liq_idx:
            wt = rows[:, liq_idx[col_key]].astype(np.float32)
            oxide_moles[:, OXIDE_DICT[oxide]] = wt / MOLAR_MASSES[oxide]

    # Consolidate iron: FeO_total_equiv = FeO_moles + 2 * Fe2O3_moles
    # (1 Fe2O3 → 2 FeO when fully reduced)
    feo_moles   = orig_wt_FeO   / MOLAR_MASSES["FeO"]
    fe2o3_moles = orig_wt_Fe2O3 / MOLAR_MASSES["Fe2O3"]
    total_iron  = feo_moles + 2.0 * fe2o3_moles
    oxide_moles[:, OXIDE_DICT["FeO"]] = total_iron

    return oxide_moles, total_iron, orig_wt_FeO, orig_wt_Fe2O3, P, T, fO2


def _run_iron_speciator(oxide_moles: np.ndarray, P, T, fO2) -> np.ndarray:
    """
    Run NN_MELTS.Iron_Speciator as an unbound call with the minimal fake emulator.
    Returns result (n, 14) numpy array.
    """
    n = oxide_moles.shape[0]
    fake = _FakeEmulator()

    oxide_tensor  = torch.tensor(oxide_moles, dtype=torch.float32)
    dummy_normed  = torch.zeros((n, fake.feature_offset), dtype=torch.float32)
    P_t   = torch.tensor(P,   dtype=torch.float32)
    T_t   = torch.tensor(T,   dtype=torch.float32)
    fO2_t = torch.tensor(fO2, dtype=torch.float32)

    result = NN_MELTS.Iron_Speciator(
        fake,
        oxides=oxide_tensor,
        Normedfeatures=dummy_normed,
        P=P_t,
        T=T_t,
        fO2=fO2_t,
    )
    return result.detach().cpu().numpy()


def _check_conservation(bmt: BigMetaTable, liquid_mask: np.ndarray) -> None:
    oxide_moles, total_iron, *_ = _build_input_tensor(bmt, liquid_mask)
    P, T, fO2 = (
        bmt.table[liquid_mask, bmt.indexer.MELTS_indices["System_main"]["Pressure"]].astype(np.float32),
        bmt.table[liquid_mask, bmt.indexer.MELTS_indices["System_main"]["Temperature"]].astype(np.float32),
        bmt.table[liquid_mask, bmt.indexer.MELTS_indices["System_main"]["logfO2-QFM"]].astype(np.float32),
    )
    result = _run_iron_speciator(oxide_moles, P, T, fO2)

    recovered_FeO_mol   = result[:, OXIDE_DICT["FeO"]]
    recovered_Fe2O3_mol = result[:, OXIDE_DICT["Fe2O3"]]
    recovered_iron_equiv = recovered_FeO_mol + 2.0 * recovered_Fe2O3_mol

    np.testing.assert_allclose(
        recovered_iron_equiv,
        total_iron,
        rtol=1e-4,
        atol=1e-6,
        err_msg=(
            "Iron_Speciator failed to conserve total iron in FeO-equivalent moles. "
            "new_FeO + 2 * new_Fe2O3 must equal original consolidated iron."
        ),
    )


def _evaluate_disparity(bmt: BigMetaTable, liquid_mask: np.ndarray) -> None:
    (
        oxide_moles, total_iron,
        orig_wt_FeO, orig_wt_Fe2O3,
        P, T, fO2,
    ) = _build_input_tensor(bmt, liquid_mask)

    result = _run_iron_speciator(oxide_moles, P, T, fO2)

    # Convert result moles back to wt%
    masses     = result * MM_ARR[np.newaxis, :]          # (n, 14)
    total_mass = masses.sum(axis=1, keepdims=True)       # (n, 1)
    safe_mass  = np.where(total_mass > 0, total_mass, 1.0)
    wt_pct     = 100.0 * masses / safe_mass              # (n, 14)

    recovered_wt_FeO   = wt_pct[:, OXIDE_DICT["FeO"]]
    recovered_wt_Fe2O3 = wt_pct[:, OXIDE_DICT["Fe2O3"]]

    err_FeO   = recovered_wt_FeO   - orig_wt_FeO
    err_Fe2O3 = recovered_wt_Fe2O3 - orig_wt_Fe2O3

    mae_FeO    = float(np.mean(np.abs(err_FeO)))
    mae_Fe2O3  = float(np.mean(np.abs(err_Fe2O3)))
    max_FeO    = float(np.max(np.abs(err_FeO)))
    max_Fe2O3  = float(np.max(np.abs(err_Fe2O3)))
    bias_FeO   = float(np.mean(err_FeO))
    bias_Fe2O3 = float(np.mean(err_Fe2O3))

    n = orig_wt_FeO.shape[0]
    print(f"\n{'='*60}")
    print(f"[Iron_Speciator vs MELTS Disparity]  n_rows={n}")
    print(f"  FeO   MAE={mae_FeO:.3f} wt%  bias={bias_FeO:+.3f}  max={max_FeO:.3f}")
    print(f"  Fe2O3 MAE={mae_Fe2O3:.3f} wt%  bias={bias_Fe2O3:+.3f}  max={max_Fe2O3:.3f}")
    print(f"{'='*60}")

    # K&C (1991) vs MELTS; allow up to 5 wt% MAE
    assert mae_FeO < 5.0, (
        f"FeO MAE {mae_FeO:.2f} wt% exceeds 5 wt% threshold — "
        "check Iron_Speciator implementation or data quality."
    )
    assert mae_Fe2O3 < 5.0, (
        f"Fe2O3 MAE {mae_Fe2O3:.2f} wt% exceeds 5 wt% threshold — "
        "check Iron_Speciator implementation or data quality."
    )


if __name__ == "__main__":
    base = _find_bmt_base()
    if base is None:
        print("No BigMetaTable CSV found; cannot run standalone.")
    else:
        bmt_obj = BigMetaTable(str(base))
        mask = bmt_obj.table[:, bmt_obj.indexer.MELTS_indices["System_main"]["F"]] > 0.10
        print(f"Rows with >10% liquid: {mask.sum()} / {mask.shape[0]}")
        _check_conservation(bmt_obj, mask)
        print("Iron conservation: PASSED")
        _evaluate_disparity(bmt_obj, mask)
