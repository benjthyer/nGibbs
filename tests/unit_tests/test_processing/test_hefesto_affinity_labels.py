"""Synthetic checks for builder.processing.hefesto_affinity_labels."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from builder.processing import hefesto_affinity_labels as hal  # noqa: E402


def _scan(x, n, A, F):
    return (np.asarray(n, np.float32)[:, None], np.asarray(A, np.float32)[:, None],
            np.asarray(F, np.int8)[:, None], np.asarray(x, np.float64),
            np.zeros(len(x), dtype=np.int64))


def _linear_boundary(lam, x0, x_star, step, k, dndx):
    """Phase present for x < x_star with n = dndx*(x_star-x); absent A = (dndx/lam)*(x-x_star)."""
    x = x0 + step * np.arange(k)
    n = np.where(x < x_star, dndx * (x_star - x), 0.0)
    A = np.where(x < x_star, 0.0, dndx / lam * (x - x_star))
    F = np.where(x < x_star, hal.FLAG_PRESENT, hal.FLAG_MINIMIZED)
    return x, n, A, F


def test_continuous_crossing_recovers_lambda():
    prm = hal.hefesto_params()
    x, n, A, F = _linear_boundary(lam=0.3, x0=0, x_star=4.37, step=1.0, k=10, dndx=0.02)
    rec = hal.find_crossings(*_scan(x, n, A, F), prm)
    assert rec['kind'].tolist() == [0]
    assert np.isclose(rec['lam'][0], 0.3, rtol=1e-4)


def test_scan_edge_one_sided_estimates():
    prm = hal.hefesto_params()
    # absent side has a single row (edge): lambda from s and x*_n
    x, n, A, F = _linear_boundary(lam=0.3, x0=0, x_star=4.37, step=1.0, k=6, dndx=0.02)
    rec = hal.find_crossings(*_scan(x, n, A, F), prm)
    assert rec['kind'].tolist() == [1] and np.isclose(rec['lam'][0], 0.3, rtol=1e-4)
    # present side has a single row (edge): lambda from sA and x*_A
    x, n, A, F = _linear_boundary(lam=0.3, x0=4, x_star=4.37, step=1.0, k=6, dndx=0.02)
    rec = hal.find_crossings(*_scan(x, n, A, F), prm)
    assert rec['kind'].tolist() == [2] and np.isclose(rec['lam'][0], 0.3, rtol=1e-4)


def test_jump_and_flicker_rejection():
    prm = hal.hefesto_params()
    x = np.arange(8.0)
    n = np.array([0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0])            # finite jump: polymorph
    A = np.array([0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4])
    F = np.where(n > 0, hal.FLAG_PRESENT, hal.FLAG_MINIMIZED)
    n[0:4] += 0.001 * np.arange(4)[::-1]                        # tiny slope
    rec = hal.find_crossings(*_scan(x, n, A, F), prm)
    assert rec['kind'].tolist() == [3] and np.isclose(rec['jump'][0], 0.5)
    # supersaturated absent neighbour: rejected
    x, n, A, F = _linear_boundary(lam=0.3, x0=0, x_star=4.37, step=1.0, k=10, dndx=0.02)
    A[5], F[5] = -0.5, hal.FLAG_SUPERSATURATED
    assert hal.find_crossings(*_scan(x, n, A, F), prm)['phase'].size == 0


def test_label_rows_rules():
    prm = hal.hefesto_params()
    n = np.array([[0.2, 0, 0, 0, 0, 0]], np.float32)
    A = np.array([[0, 0.5, 20.0, -0.3, np.nan, np.nan]], np.float32)
    F = np.array([[0, hal.FLAG_MINIMIZED, hal.FLAG_UPPER_BOUND, hal.FLAG_SUPERSATURATED, 3, -1]], np.int8)
    lam = np.full(6, 0.1)
    g, s, cnt, _ = hal.label_rows(n, A, F, lam, prm)
    np.testing.assert_allclose(g[0, :4], [0.2, -0.05, -2.0, 0.0], rtol=1e-6)
    assert np.isnan(g[0, 4:]).all() and np.isnan(s[0, 4:]).all()
    np.testing.assert_allclose(s[0, :4], 0.1)
    assert cnt['absent_unlabelled'] == 2


def test_summarize_polymorph_and_fallback():
    prm = hal.hefesto_params(dict(min_crossings=3))
    rec = dict(phase=np.array([0, 0, 0, 1, 1, 1, 1]),
               kind=np.array([0, 1, 2, 3, 3, 3, 0]),
               lam=np.array([0.1, 0.1, 0.1, np.nan, np.nan, np.nan, 5.0]),
               jump=np.array([0, 0, 0, 0.2, 0.2, 0.4, 0]),
               axis=np.array(['isobar'] * 7))
    cal = hal.summarize_crossings(rec, ['a', 'b', 'c'], prm)
    assert cal['method'] == ['crossings', 'polymorph', 'fallback']
    np.testing.assert_allclose(cal['lam'], [0.1, 0.2 / prm['A_window'], 0.1])


def test_c_floor():
    from builder.training.affinity_targets import AffinityTargets
    cs = np.array([[1e-4, 0.05], [-1.0, 0.3], [-1.0, 0.3], [0.0, 0.0]])
    med = [0.04, 0.1]
    off = AffinityTargets(cs, params=dict(min_crossings=1), crossings_per_phase=[100, 100],
                          present_median=med)
    np.testing.assert_allclose(off.c, [1e-4, 0.05])
    on = AffinityTargets(cs, params=dict(min_crossings=1, c_floor_frac=0.1),
                         crossings_per_phase=[100, 100], present_median=med)
    np.testing.assert_allclose(on.c, [0.004, 0.05])
    assert on.c_floored.tolist() == [True, False]
    assert off.fingerprint() != on.fingerprint()
    try:
        AffinityTargets(cs, params=dict(c_floor_frac=0.1), crossings_per_phase=[100, 100])
        raise AssertionError('expected ValueError without present_median')
    except ValueError:
        pass
