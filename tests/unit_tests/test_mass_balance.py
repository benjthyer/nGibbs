"""Unit tests for the emulator mass-balance correction.

`MassBalanceProjector` (engine/mass_balance.py) is tested in isolation with a synthetic
stoichiometry matrix -- no model checkpoint needed. An optional integration check runs
all three `NN_MELTS(mass_balance=...)` modes against a real light bundle when one is
present. The MELTS pyroxene/spinel polish (`polish_negative_sp`, run after every
projector step) is tested on synthetic intensive compositions, and end to end on the
MELTS 1.2 bundle and a standard when they are present.

Run:  pytest tests/unit_tests/test_mass_balance.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ngibbs.engine.mass_balance import MassBalanceProjector
from ngibbs.engine.NN import MidLevelNetwork


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


# --------------------------------------------------------------------------- #
#  Pyroxene/spinel legality polish inside the iterative mass balance
# --------------------------------------------------------------------------- #
def test_projector_runs_polish_after_every_step():
    n_pred, compToEl, b_dir, _ = _synthetic_system(seed=4)
    seen = []

    def polish(n):
        seen.append(n.clone())
        out = n.clone()
        out[:, 0] = 0.0
        return out

    n_corr, resid = MassBalanceProjector(iters=4)(n_pred, compToEl, b_dir, polish=polish)
    assert len(seen) == 4
    assert (n_corr[:, 0] == 0).all()          # the last thing applied is the polish
    assert torch.allclose(resid, _resid(n_corr, compToEl, b_dir), atol=1e-5)


class _SpinelHost:
    """The MidLevelNetwork spinel polish methods on a bare object (they need only
    detail_label_indices)."""
    detail_label_indices = {'spinel': {'hercynite': 0, 'magnetite': 1, 'spinel': 2, 'ulvospinel': 3}}
    polish_negative_sp = MidLevelNetwork.polish_negative_sp
    polish_negative_spFe = MidLevelNetwork.polish_negative_spFe


def _sp_constraints(ic):
    c2, c3, c4, c5 = ic.T
    feo = c2 + c3 + 2.25 * c5 - 19 * c4          # FeO >= 0 (compToOx, transformed basis)
    al = c2 + c4 - (2 / 3) * c3 - 0.25 * c5      # Al2O3 >= 0
    return feo, al


def test_polish_negative_sp_imposes_only_the_violated_constraint():
    # rows: Al-only violation (an Al-poor Fe-Ti spinel), FeO-only violation (a
    # Mg-rich spinel), both violated, and a legal row
    ic = torch.tensor([[0.33, 0.56, 0.003, 0.107],
                       [0.40, 0.05, 0.53, 0.02],
                       [0.02, 0.80, 0.06, 0.12],
                       [0.50, 0.30, 0.02, 0.18]], dtype=torch.float64)
    feo0, al0 = _sp_constraints(ic)
    assert al0[0] < 0 < feo0[0] and feo0[1] < 0 < al0[1]
    assert feo0[2] < 0 and al0[2] < 0 and feo0[3] > 0 and al0[3] > 0
    out = _SpinelHost().polish_negative_sp(ic.clone())
    feo, al = _sp_constraints(out)
    assert (feo > -1e-9).all() and (al > -1e-9).all()
    assert torch.allclose(out.sum(1), ic.sum(1))                  # block total kept
    # Al-only: spinel (MgAl2O4) untouched, FeO NOT driven to zero
    assert out[0, 2] == ic[0, 2] and abs(al[0]) < 1e-9 and feo[0] > 0.5 * feo0[0]
    # FeO-only: magnetite and ulvospinel untouched, Al2O3 stays positive
    assert out[1, 1] == ic[1, 1] and out[1, 3] == ic[1, 3]
    assert abs(feo[1]) < 1e-9 and al[1] > 0
    # both violated: both equalities
    assert abs(feo[2]) < 1e-9 and abs(al[2]) < 1e-9
    assert torch.equal(out[3], ic[3])


_MELTS_120 = Path(__file__).resolve().parents[2] / "src" / "ngibbs" / "engine" / "TrainedModels" / "120"
_STANDARDS = (Path(__file__).resolve().parents[2] / "src" / "ngibbs" / "deployment_tests" /
              "MELTSIsobaricStandards" / "120" / "NoCr" / "BishopTuff")


@pytest.mark.skipif(not (_MELTS_120.exists() and _STANDARDS.exists()),
                    reason="MELTS 1.2 bundle or BishopTuff standard not checked out")
def test_melts_iterative_mass_balance_returns_legal_spinel_and_opx():
    """BishopTuff: the emulator's spinel is Al-deficient; after the iterative mass
    balance (polish after every step) no orthopyroxene or spinel oxide is negative."""
    import warnings
    from ngibbs.engine.API import MELTSAPI
    import ngibbs.deployment_tests.melts_comparison as mc

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        api = MELTSAPI(_MELTS_120)
        em = api.nocr.isothermal_emulator
        sm = mc.read_system_main(_STANDARDS)
        bulk = mc.read_bulk_comp(_STANDARDS)
        ox = list(mc._emulator_input_oxides(em))
        table = np.column_stack([sm['Pressure'].values, sm['Temperature'].values]
                                + [mc._bulk_oxide(bulk, o) for o in ox]).astype(np.float32)
        out = api.ForwardMB(table, headers=['Pressure(System_main)', 'Temperature(System_main)'] + ox,
                            outputs=['component_moles', 'reconstruction_residual'])
    cm = out['component_moles'].double()
    compToOx = torch.as_tensor(np.asarray(em.ml_indexer.compToOx, dtype=np.float64))
    lic = em.ml_indexer.label_indices_comp
    for ph in ('orthopyroxene', 'spinel'):
        blk = torch.zeros_like(cm)
        blk[:, lic[ph]] = cm[:, lic[ph]]
        tot = blk.sum(1)
        present = tot > 0
        assert present.any()
        oxides = (blk[present] @ compToOx) / tot[present, None]
        assert oxides.min() > -1e-6, ph
    assert out['reconstruction_residual'].max() < 5e-3
