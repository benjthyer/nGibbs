"""Tests for the cumulate inverter (engine/cumulate_inversion.py).

Uses the 102 closed NoCr isothermal checkpoint and its deployment-test bundle; the
module-level fixture skips everything if they are not present.

Run:  pytest tests/unit_tests/test_cumulate_inversion.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from ngibbs.engine import cumulate_inversion as ci

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / 'src/ngibbs/engine/TrainedModels/102'
CKPT = MODEL_DIR / '102Closed_NoCr_NPT.tar'
BUNDLE = MODEL_DIR / '102Closed_NoCr_NPT_Test_subset15000.tar.gz'


@pytest.fixture(scope='module')
def setup():
    if not (CKPT.exists() and BUNDLE.exists()):
        pytest.skip('102 closed NoCr NPT checkpoint / test bundle not available')
    from ngibbs.engine.NN import rebuild_MELTS_model
    from ngibbs.engine.emulator import NN_MELTS, TrainingRangeWarning
    warnings.simplefilter('ignore', TrainingRangeWarning)
    em = NN_MELTS(rebuild_MELTS_model(str(CKPT)))
    bundle = ci.load_ml_bundle(BUNDLE)
    return em, bundle, ci._Geometry(em)


def test_bundle_loader(setup):
    em, b, geo = setup
    assert b.features.shape[1] == geo.n_feat + geo.E
    assert b.binary_labels.shape[1] == len(geo.phases)
    assert np.allclose(b.features[:, geo.n_feat:].sum(1), 1.0, atol=1e-3)


def test_oxide_roundtrip(setup):
    _, b, geo = setup
    x = torch.as_tensor(b.features[:64, geo.n_feat:])
    assert torch.allclose(geo.oxwt_to_el(geo.el_to_oxwt(x)), x / x.sum(1, keepdim=True), atol=1e-5)


def test_label_basis_matches_network_basis(setup):
    """Bundle labels are native endmembers, the chem heads emit the PxSp-transformed
    basis. Converted correctly, network and labels agree at the true bulk."""
    _, b, geo = setup
    F = torch.as_tensor(b.features)
    L = torch.as_tensor(b.labels)
    with torch.no_grad():
        _, chem, _ = geo.network(F[:, geo.P_idx], F[:, geo.T_idx], F[:, geo.n_feat:])
    for p in ('olivine', 'clinopyroxene', 'orthopyroxene', 'plagioclase', 'spinel'):
        m = torch.as_tensor(b.binary_labels[:, geo.phase_col[p]] > 0.5)
        d = geo.phase_oxwt(chem[m], p) - geo.phase_oxwt(L[m], p, native=True)
        rmse = d.pow(2).mean(1).sqrt()
        assert torch.nanquantile(rmse, 0.5) < 0.5, p


def test_find_liquidi_brackets(setup):
    em, b, geo = setup
    rows = np.nonzero(b.binary_labels[:, geo.liq] > 0.5)[0][:96]
    x = b.features[rows, geo.n_feat:]
    P = b.features[rows, geo.P_idx]
    r = ci.find_liquidi(em, x, P, n_grid=24, tol=0.5)
    assert r['passes'] == 24 + 7 + 1          # 51 C grid step -> 7 halvings to <= 0.5 C
    ok = r['status'] == 0
    assert ok.mean() > 0.8
    solid = [geo.phase_col[p] for p in geo.phases if p not in ('melts-liquid', 'fluid')]
    with torch.no_grad():
        for dT, want in ((+1.0, False), (-1.0, True)):
            _, _, g = geo.network(torch.as_tensor(P[ok]), torch.as_tensor(r['T_liquidus'][ok] + dT),
                                  torch.as_tensor(x[ok]))
            assert ((g[:, solid].max(1).values > 0).numpy() == want).mean() > 0.97
    # a liquid-bearing bundle row cannot have its liquidus below its own temperature
    # (the network agrees with MELTS on "partly molten" for most rows)
    T = b.features[rows, geo.T_idx]
    assert np.mean(r['T_liquidus'][ok] >= T[ok] - 5) > 0.9


def test_from_bundle_target(setup):
    em, b, geo = setup
    rows = ci.select_cumulate_rows(em, b, ['olivine', 'clinopyroxene', 'plagioclase'])
    assert len(rows)
    t = ci.CumulateTarget.from_bundle(em, b, int(rows[0]))
    assert set(t.phases) == {'olivine', 'clinopyroxene', 'plagioclase'}
    for p, w in t.oxide_wt.items():
        assert abs(w.sum() - 100) < 1e-3 and (w > -1e-6).all(), p
    assert abs(t.truth['liquid_oxide_wt'].sum() - 100) < 1e-3


def _target(em, b, geo, asm=('olivine', 'clinopyroxene', 'plagioclase')):
    rows = ci.select_cumulate_rows(em, b, list(asm))
    rows = rows[b.mass_labels[rows, geo.liq] > 30]
    return ci.CumulateTarget.from_bundle(em, b, int(rows[0]))


def test_liquidus_tolerance_controls_cost(setup):
    em, b, geo = setup
    x = b.features[:32, geo.n_feat:]
    P = b.features[:32, geo.P_idx]
    coarse = ci.find_liquidi(em, x, P, n_grid=12, tol=20.0, return_phase=False)
    fine = ci.find_liquidi(em, x, P, n_grid=12, tol=0.25)
    assert coarse['passes'] < fine['passes'] and 'liquidus_phase' not in coarse
    ok = (coarse['status'] == 0) & (fine['status'] == 0)
    assert np.all(np.abs(coarse['T_liquidus'][ok] - fine['T_liquidus'][ok]) <= 10.0 + 0.125 + 1e-3)


def test_constrained_basis(setup):
    """Oxides absent from every cumulus phase are outside the basis; H2O is carried."""
    em, b, geo = setup
    inv = ci.CumulateInverter(em, _target(em, b, geo, ('olivine',)))
    assert set(inv.constrained_oxides) <= {'SiO2', 'MgO', 'FeO', 'CaO', 'Fe2O3'}
    assert inv.volatile_oxides == ['H2O']
    assert 'K2O' not in inv.basis_oxides and 'P2O5' not in inv.basis_oxides


def test_prior_nodes(setup):
    em, b, geo = setup
    inv = ci.CumulateInverter(em, _target(em, b, geo), ci.InversionConfig(seed=3))
    pr = inv.initial_nodes(256)
    wt = geo.el_to_oxwt(torch.as_tensor(pr['x'])).numpy()
    outside = [geo.Oxides.index(o) for o in geo.Oxides if o not in inv.basis_oxides]
    assert np.all(wt[:, outside] == 0)
    h = wt[:, geo.Oxides.index('H2O')]
    assert (h >= 0).all() and 0.2 < np.mean(h == 0) < 0.5          # a third zeroed (+ clamped)
    assert np.mean(pr['prior_status'] == 'liquidus') > 0.9
    assert (pr['prior_T'][pr['prior_moves'] == 0] <= 1600 + 1e-3).all()
    assert (pr['T'] >= inv.T_bounds[0]).all()


def test_inversion_smoke(setup):
    """A short run: the loss falls, bounds hold, fixed oxides never move, and most
    nodes end with the cumulus phases + liquid present."""
    em, b, geo = setup
    t = _target(em, b, geo)
    cfg = ci.InversionConfig(steps=120, batch_size=64, seed=1, log_every=10 ** 9)
    inv = ci.CumulateInverter(em, t, cfg)
    res = inv.run(64, verbose=False)
    h = res.history
    assert h['best_median'].iloc[-1] < 0.2 * h['best_median'].iloc[0]
    d = res.nodes
    assert d['assemblage_ok'].mean() > 0.5
    assert d['comp_rmse_wt'].min() < 1.0
    lo, hi = inv.P_bounds
    assert ((d['P_bar'] >= lo) & (d['P_bar'] <= hi)).all()
    lo, hi = inv.T_bounds
    assert ((d['T_C'] >= lo) & (d['T_C'] <= hi)).all()
    bulk = d[[f'bulk_{o}' for o in geo.Oxides]].values
    assert np.allclose(bulk.sum(1), 100, atol=1e-2)
    # oxides outside the basis stay exactly zero; carried volatiles keep their start
    for o in geo.Oxides:
        if o not in res.basis_oxides:
            assert (d[f'bulk_{o}'] == 0).all() and (d[f'liq_{o}'].abs() < 1e-4).all(), o
    xb = geo.oxwt_to_el(torch.as_tensor(bulk, dtype=torch.float32)).numpy()
    xi = geo.oxwt_to_el(torch.as_tensor(d[[f'init_{o}' for o in geo.Oxides]].values,
                                        dtype=torch.float32)).numpy()
    for o in inv.volatile_oxides:
        j = geo.el_of_ox[geo.Oxides.index(o)]
        assert np.allclose(xb[:, j], xi[:, j], atol=1e-5), o
    # best state re-evaluates to the recorded loss
    top = d.nsmallest(4, 'loss')
    x = geo.oxwt_to_el(torch.as_tensor(top[[f'bulk_{o}' for o in geo.Oxides]].values, dtype=torch.float32))
    ev = inv.evaluate(torch.as_tensor(top['P_bar'].values, dtype=torch.float32),
                      torch.as_tensor(top['T_C'].values, dtype=torch.float32), x)
    assert np.allclose(ev['total'].numpy(), top['loss'].values, rtol=1e-2, atol=1e-3)
    # truth renormalised onto the basis
    tl = res.truth_liquid()
    assert abs(sum(tl[o] for o in res.basis_oxides) - 100) < 1e-6
    # summaries
    acc = res.accepted()
    if len(acc) > 5:
        C = res.covariance(subset=acc)
        assert list(C.index) == ['P_bar', 'T_C'] + [f'liq_{o}' for o in res.basis_oxides]
        X, Y, Z = res.density2d(subset=acc, gridsize=40)
        area = (X[0, 1] - X[0, 0]) * (Y[1, 0] - Y[0, 0])
        assert 0.9 < Z.sum() * area < 1.01


def test_extra_phases_free_vs_forbid(setup):
    em, b, geo = setup
    t = _target(em, b, geo)
    free = ci.CumulateInverter(em, t, ci.InversionConfig(extra_phases='free'))
    forbid = ci.CumulateInverter(em, t, ci.InversionConfig(extra_phases='forbid'))
    assert free.forbidden == [] and len(forbid.forbidden) == len(free.extra) > 0
    pr = free.initial_nodes(32)
    P, T, x = (torch.as_tensor(pr[k]) for k in ('P', 'T', 'x'))
    assert (forbid.evaluate(P, T, x)['assemblage'] >= free.evaluate(P, T, x)['assemblage'] - 1e-6).all()


def test_rejects_gated_or_open_models(setup):
    em, *_ = setup

    class Fake:
        model = torch.nn.Linear(1, 1)
    with pytest.raises(TypeError):
        ci._Geometry(Fake())


def test_history_axes_and_plots(setup, tmp_path):
    """Convergence history, derived NKC/2A index, axis choice and the figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    em, b, geo = setup
    t = _target(em, b, geo)
    res = ci.CumulateInverter(em, t, ci.InversionConfig(steps=20, batch_size=64, seed=2)).run(64, verbose=False)
    h = res.history
    assert len(h) == 21
    for k in ('sum_loss', 'sum_assemblage', 'sum_composition', 'loss_p10', 'loss_p50', 'loss_p90', 'best_min'):
        assert k in h and np.isfinite(h[k]).all(), k
    assert np.allclose(h['sum_loss'], h['mean_loss'] * len(res.nodes), rtol=1e-4)
    # molar (Na2O + K2O + 2CaO) / (2 Al2O3)
    from ngibbs.config.constants import OXIDE_MOLAR_MASSES as MM
    r = res.nodes.iloc[0]
    ref = (r['liq_Na2O'] / MM['Na2O'] + r['liq_K2O'] / MM['K2O'] + 2 * r['liq_CaO'] / MM['CaO']) \
        / (2 * r['liq_Al2O3'] / MM['Al2O3'])
    assert np.isclose(r['liq_NKC2A'], ref)
    x, y = res.composition_axes()
    assert (x, y) == ('SiO2', 'MgO' if res.truth_liquid()['MgO'] >= 4 else 'NKC2A')
    for make in (lambda: res.plot_prior_vs_final(loss_quantile=0.5),
                 lambda: res.plot_convergence(), lambda: res.plot_prior()):
        plt.close(make())


def test_recovery_row_placeholders(setup):
    import importlib.util
    spec = importlib.util.spec_from_file_location('demo', REPO / 'scripts' / 'cumulate_inversion_demo.py')
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    em, b, geo = setup
    t = _target(em, b, geo, ('olivine',))
    res = ci.CumulateInverter(em, t, ci.InversionConfig(steps=5, batch_size=32)).run(32, verbose=False)
    rec = demo.recovery_row(res, res.accepted(), 0.0)
    for o in geo.Oxides:
        if o in res.basis_oxides:
            assert isinstance(rec[f'acc_err_{o}'], float) and rec[f'role_{o}'] in ('constrained', 'carried')
        else:
            assert rec[f'acc_err_{o}'] == '--' and rec[f'role_{o}'] == '--'
    assert rec['role_H2O'] == 'carried' and isinstance(rec['best_err_H2O'], float)


def test_emulator_baseline(setup):
    """from_bundle stores the forward model at the true (P, T, bulk); on the bundle
    rows the emulator reproduces the MELTS liquid closely."""
    em, b, geo = setup
    t = _target(em, b, geo)
    emu = t.truth['emulator']
    assert abs(emu['liquid_oxide_wt'].sum() - 100) < 1e-3 and emu['liquid_present']
    assert np.sqrt(np.mean((emu['liquid_oxide_wt'] - t.truth['liquid_oxide_wt']) ** 2)) < 1.5
    assert set(emu['solids_true']) >= set(t.phases)
    res = ci.CumulateInverter(em, t, ci.InversionConfig(steps=3, batch_size=16)).run(16, verbose=False)
    ev = res.truth_vector(['liq_SiO2', 'P_bar', 'T_C'], source='emulator')
    assert ev[1] == t.truth['P_bar'] and ev[2] == t.truth['T_C']
    assert 'nGibbs_at_truth' in res.summary(subset=res.nodes)


def test_prior_oxide_noise(setup):
    """With the per-oxide noise off, a single-phase cumulate's prior bulks are that
    phase's composition exactly; with it on (default U(0.7, 1.3)) they scatter."""
    em, b, geo = setup
    t = _target(em, b, geo, ('olivine',))
    rows = {}
    for noise in (None, (0.7, 1.3)):
        inv = ci.CumulateInverter(em, t, ci.InversionConfig(seed=4, prior_oxide_noise=noise, volatiles=()))
        rows[noise] = inv.initial_nodes(64)['prior_bulk_wt']
    ref = inv.phase_oxwt_target['olivine'] * np.array([o in inv.constrained_oxides for o in geo.Oxides])
    ref = 100 * ref / ref.sum()
    assert np.allclose(rows[None], ref, atol=1e-6)
    assert rows[(0.7, 1.3)].std(0).max() > 0.5
    assert np.allclose(rows[(0.7, 1.3)].sum(1), 100)


def test_prior_trace_phase_fractions(setup):
    """Accessory phases are drawn from U(0, 10%) of the mix; the rest share the remainder."""
    em, b, geo = setup
    t = ci.CumulateTarget(phases=['clinopyroxene', 'plagioclase', 'apatite', 'whitlockite'],
                          oxide_wt={'clinopyroxene': np.eye(geo.O)[geo.Oxides.index('SiO2')] * 100,
                                    'plagioclase': np.eye(geo.O)[geo.Oxides.index('Al2O3')] * 100})
    inv = ci.CumulateInverter(em, t)
    w = inv._prior_phase_fractions(np.random.default_rng(0), 5000)
    assert np.allclose(w.sum(1), 1.0)
    tr = [t.phases.index('apatite'), t.phases.index('whitlockite')]
    assert (w[:, tr] >= 0).all() and (w[:, tr] <= 0.10).all()
    assert abs(w[:, tr].mean() - 0.05) < 0.005
    assert w[:, [0, 1]].max() > 0.7
    only = ci.CumulateInverter(em, ci.CumulateTarget(phases=['apatite', 'whitlockite'], oxide_wt={}))
    w2 = only._prior_phase_fractions(np.random.default_rng(1), 100)
    assert np.allclose(w2.sum(1), 1.0) and w2.max() > 0.5
