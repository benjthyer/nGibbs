"""Unit tests for the emulator mass-balance correction.

`MassBalanceProjector` (engine/mass_balance.py) is tested in isolation with a synthetic
stoichiometry matrix -- no model checkpoint needed. An optional integration check runs
all three `NN_MELTS(mass_balance=...)` modes against a real light bundle when one is
present.

Run:  pytest tests/unit_tests/test_mass_balance.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ngibbs.engine.mass_balance import MassBalanceProjector


def _synthetic_system(B=32, C=12, E=5, seed=0):
    """Random nonneg component moles `n` (with a zeroed support) + a feasible bulk
    direction `b_dir`, and a random nonnegative `compToEl` (C, E)."""
    g = torch.Generator().manual_seed(seed)
    compToEl = torch.rand(C, E, generator=g) * (torch.rand(C, E, generator=g) > 0.3)
    n_true = torch.rand(B, C, generator=g)
    # zero out ~40% of phases per row (the "absent" support)
    n_true = n_true * (torch.rand(B, C, generator=g) > 0.4)
    bl = n_true @ compToEl
    b_dir = bl / bl.sum(dim=1, keepdim=True).clamp(min=1e-9)
    # perturb n away from the feasible point so there is something to correct
    n_pred = (n_true + 0.15 * torch.randn(B, C, generator=g)).clamp(min=0.0)
    return n_pred, compToEl, b_dir, n_true


def _resid(n, compToEl, b_dir):
    bl = n @ compToEl
    return (bl / bl.sum(dim=1, keepdim=True).clamp(min=1e-9) - b_dir).norm(dim=1)


@pytest.mark.parametrize("relative", [True, False])
def test_projector_reduces_residual(relative):
    n_pred, compToEl, b_dir, _ = _synthetic_system()
    proj = MassBalanceProjector(iters=5, relative=relative)
    n_corr, resid = proj(n_pred, compToEl, b_dir)

    r0 = _resid(n_pred, compToEl, b_dir)
    assert (resid <= r0 + 1e-6).all()
    assert resid.mean() < 0.25 * r0.mean()
    # residual returned matches a fresh recompute
    assert torch.allclose(resid, _resid(n_corr, compToEl, b_dir), atol=1e-5)


@pytest.mark.parametrize("relative", [True, False])
def test_projector_nonnegative_and_support_preserving(relative):
    n_pred, compToEl, b_dir, _ = _synthetic_system(seed=1)
    absent = n_pred == 0
    n_corr, _ = MassBalanceProjector(iters=5, relative=relative)(n_pred, compToEl, b_dir)
    assert (n_corr >= 0).all()
    # a zeroed phase is never revived by the correction
    assert (n_corr[absent] == 0).all()


def test_relative_weighting_spares_low_abundance_phases():
    """The weighted (relative) objective should perturb a small phase less, in relative
    terms, than the unweighted one."""
    n_pred, compToEl, b_dir, _ = _synthetic_system(seed=2)
    present = n_pred > 0
    small = present & (n_pred < n_pred[present].median())

    n_rel, _ = MassBalanceProjector(iters=5, relative=True)(n_pred, compToEl, b_dir)
    n_uni, _ = MassBalanceProjector(iters=5, relative=False)(n_pred, compToEl, b_dir)

    rel_change_weighted = ((n_rel - n_pred).abs() / n_pred.clamp(min=1e-9))[small].mean()
    rel_change_uniform = ((n_uni - n_pred).abs() / n_pred.clamp(min=1e-9))[small].mean()
    assert rel_change_weighted < rel_change_uniform


def test_projector_is_device_generic():
    """No hard-coded device: runs wherever the inputs live (CUDA path exercised only if
    a GPU is available)."""
    n_pred, compToEl, b_dir, _ = _synthetic_system(seed=3)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    proj = MassBalanceProjector(iters=3)
    n_corr, resid = proj(n_pred.to(dev), compToEl.to(dev), b_dir.to(dev))
    assert n_corr.device.type == dev
    assert torch.isfinite(resid).all()


# --------------------------------------------------------------------------- #
#  Integration: real bundle, all three modes share one code path
# --------------------------------------------------------------------------- #
_LIGHT_BUNDLE = (Path(__file__).resolve().parents[2] / "src" / "ngibbs" / "engine" /
                 "TrainedModels" / "HeFESTo_Adiabats_Light" /
                 "HeFESTo_Earth_Adiabat_NPT_light.tar")


@pytest.mark.skipif(not _LIGHT_BUNDLE.exists(), reason="light bundle not checked out")
@pytest.mark.parametrize("mode", ["none", "iterative", "pinv"])
def test_forwardMB_modes_share_one_path(mode):
    from ngibbs.engine.NN import rebuild_MELTS_model
    from ngibbs.engine.emulator import NN_MELTS

    model = rebuild_MELTS_model(str(_LIGHT_BUNDLE))
    mli = model.ml_indexer
    fn, elk = list(mli.featureNames), list(mli.Elkeys)
    rng = np.random.default_rng(0)
    feat = np.zeros((16, len(fn) + len(elk)), dtype=np.float32)
    for i, nm in enumerate(fn):
        if 'P(GPa)' in nm:
            feat[:, i] = rng.uniform(1, 25, 16)
        elif 'T(K)' in nm:
            feat[:, i] = rng.uniform(1200, 2200, 16)
    approx = dict(Si=0.20, Mg=0.24, Fe=0.04, Ca=0.018, Al=0.021, Na=0.002, O=0.46, Cr=1e-3)
    for j, el in enumerate(elk):
        feat[:, len(fn) + j] = approx.get(el, 0.0) * rng.uniform(0.7, 1.3, 16)
    ft = torch.tensor(feat)

    emu = NN_MELTS(model, mass_balance=mode)
    out = emu.forwardMB(ft, outputs=["phase_tables", "component_moles",
                                     "reconstruction_residual", "phase_present"])
    cm = out["component_moles"]
    _, mass_tbl = out["phase_tables"]

    assert (cm >= -1e-6).all()
    assert torch.allclose(mass_tbl.sum(dim=1), torch.full((16,), 100.0), atol=1e-2)
    resid = out["reconstruction_residual"]
    assert torch.isfinite(resid).all()
    if mode == "none":
        assert resid.mean() > 1e-3          # raw heads are off the manifold
    else:
        assert resid.mean() < 5e-3          # corrected
