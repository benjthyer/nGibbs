"""
Unit tests for the four newly translated large routines:
hessian, hessfunc, cp, gspec.

These tests verify basic module loading and stub structure.
Integration tests with realistic HeFESToState fixtures will follow
after volume solver and therm functions are implemented.
"""
import sys
from pathlib import Path

# Ensure the src directory is in the path so we can import nMELTS
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "src"))

import unittest
import numpy as np

from nMELTS.engine.EOS_arithmetic.state import HeFESToState
from nMELTS.engine.EOS_arithmetic import (
    hessian, hessfunc, cp, gspec, Ftotsub, volume, volumel, volumew, volumeh, therm, therml, thermg, thermw, thermh
)


class TestHessianModule(unittest.TestCase):
    """Test hessian module basic structure."""

    def test_hessian_import(self):
        """Verify hessian module imports."""
        self.assertIsNotNone(hessian)
        self.assertTrue(hasattr(hessian, 'hessian'))

    def test_hessian_dkron(self):
        """Verify dkron helper function."""
        self.assertEqual(hessian.dkron(1, 1), 1.0)
        self.assertEqual(hessian.dkron(1, 2), 0.0)


class TestHessfuncModule(unittest.TestCase):
    """Test hessfunc module basic structure."""

    def test_hessfunc_import(self):
        """Verify hessfunc module imports."""
        self.assertIsNotNone(hessfunc)
        self.assertTrue(hasattr(hessfunc, 'hessfunc'))

    def test_hessfunc_empty_state(self):
        """Test hessfunc with minimal state."""
        state = HeFESToState()
        state.nspec = 0
        state.nnull = 0
        
        nnew = np.array([])
        result = hessfunc.hessfunc(nnew, state)
        self.assertEqual(result.shape, (0, 0))


class TestCpModule(unittest.TestCase):
    """Test cp module basic structure."""

    def test_cp_import(self):
        """Verify cp module imports."""
        self.assertIsNotNone(cp)
        self.assertTrue(hasattr(cp, 'cp'))

    def test_cp_ourlog(self):
        """Verify ourlog helper function."""
        self.assertLess(cp.ourlog(0.0), -1e99)  # Should return large negative
        self.assertAlmostEqual(cp.ourlog(1.0), 0.0, places=5)
        self.assertGreater(cp.ourlog(2.0), 0.0)

    def test_cp_dkron(self):
        """Verify dkron helper function."""
        self.assertEqual(cp.dkron(0, 0), 1.0)
        self.assertEqual(cp.dkron(0, 1), 0.0)

    def test_cp_minimal_state(self):
        """Test cp with minimal state."""
        state = HeFESToState()
        state.nspec = 0
        state.Ti = 1000.0
        state.Pi = 1.0
        
        ncp = np.array([])
        chempot, rsum, volsum, smixi = cp.cp(0, ncp, state)
        
        self.assertEqual(chempot, 0.0)
        self.assertEqual(rsum, 0.0)
        self.assertEqual(volsum, 0.0)
        self.assertEqual(smixi, 0.0)


class TestGspecModule(unittest.TestCase):
    """Test gspec module basic structure."""

    def test_gspec_import(self):
        """Verify gspec module imports."""
        self.assertIsNotNone(gspec)
        self.assertTrue(hasattr(gspec, 'gspec'))

    def test_gspec_result_dataclass(self):
        """Verify GspecResult dataclass structure."""
        result = gspec.GspecResult(
            vol=50.0, Cp=50.0, Cv=40.0, gamma=2.0, K=100.0, Ks=110.0,
            alp=1e-5, Ftot=-1000.0, ph=1000.0, ent=100.0, deltas=0.0,
            tcal=1000.0, zeta=0.0, Gsh=50.0, uth=500.0, uto=400.0,
            thet=1000.0, qq=0.0, etas=0.0, dGdT=-0.1, pzp=0.0, Vdeb=50.0,
            gamdeb=1.5, spinodal=False
        )
        self.assertEqual(result.vol, 50.0)
        self.assertFalse(result.spinodal)

    def test_gspec_minimal_state(self):
        """Test gspec with minimal state."""
        state = HeFESToState()
        state.nspecp = 1
        state.nspec = 1
        state.Ti = 1000.0
        state.Pi = 1.0
        state.apar = np.zeros((1, 60))
        state.apar[0, 5] = 12.0   # Vo (apar 6)
        state.apar[0, 6] = 160.0  # Ko (apar 7)
        state.apar[0, 7] = 4.0    # Kop (apar 8)
        state.apar[0, 50] = 1.0   # vlow
        state.apar[0, 51] = 100.0 # vupp
        state.apar[0, 52] = 1.0   # vsplow
        state.apar[0, 53] = 100.0 # vspupp
        
        result, spinodal = gspec.gspec(0, state)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(spinodal, bool)
        self.assertIsInstance(result.vol, float)

    def test_ftotsub_minimal_state(self):
        state = HeFESToState()
        state.nspecp = 1
        state.nspec = 1
        state.Ti = 1000.0
        state.Pi = 1.0
        state.apar = np.zeros((1, 60))
        state.apar[0, 0] = 1.0
        state.apar[0, 1] = 1.0
        state.apar[0, 2] = 100.0
        state.apar[0, 3] = 298.15
        state.apar[0, 4] = -100.0
        state.apar[0, 5] = 12.0
        state.apar[0, 6] = 160.0
        state.apar[0, 7] = 4.0
        state.apar[0, 15] = 800.0

        ftot = Ftotsub.Ftotsub(0, 12.0, state)
        self.assertTrue(np.isfinite(ftot))

    def test_gspec_htl_dispatch_smoke(self):
        """Ensure each htl branch executes with finite outputs."""
        state = HeFESToState()
        state.nspecp = 1
        state.nspec = 1
        state.Ti = 1200.0
        state.Pi = 1.5
        state.apar = np.zeros((1, 60))
        state.apar[0, 0] = 1.0    # fn
        state.apar[0, 1] = 1.0    # zu
        state.apar[0, 2] = 100.0  # wm
        state.apar[0, 3] = 298.15 # To
        state.apar[0, 4] = -100.0 # Fo
        state.apar[0, 5] = 12.0   # Vo
        state.apar[0, 6] = 160.0  # Ko
        state.apar[0, 7] = 4.0    # Kop
        state.apar[0, 8] = 0.0    # Kopp
        state.apar[0, 15] = 800.0 # wd1o-like seed
        state.apar[0, 50] = 1.0
        state.apar[0, 51] = 2000.0
        state.apar[0, 52] = 1.0
        state.apar[0, 53] = 2000.0

        for htl in (0, 1, 2, 3, 4, 5):
            state.apar[0, 30] = float(htl)  # apar(31)
            result, spinodal = gspec.gspec(0, state)
            self.assertTrue(np.isfinite(result.vol))
            self.assertTrue(np.isfinite(result.Ftot))
            self.assertIsInstance(spinodal, bool)


class TestVolumeThermModules(unittest.TestCase):
    """Smoke tests for new volume*/therm* modules."""

    def _make_state(self):
        s = HeFESToState()
        s.nspecp = 1
        s.nspec = 1
        s.Ti = 1000.0
        s.Pi = 1.0
        s.apar = np.zeros((1, 60))
        s.apar[0, 0] = 1.0
        s.apar[0, 1] = 1.0
        s.apar[0, 2] = 100.0
        s.apar[0, 3] = 298.15
        s.apar[0, 4] = -50.0
        s.apar[0, 5] = 12.0
        s.apar[0, 6] = 160.0
        s.apar[0, 7] = 4.0
        s.apar[0, 8] = 0.0
        s.apar[0, 15] = 800.0
        s.apar[0, 50] = 1.0
        s.apar[0, 51] = 5000.0
        s.apar[0, 52] = 1.0
        s.apar[0, 53] = 5000.0
        return s

    def test_module_imports(self):
        self.assertIsNotNone(volume)
        self.assertIsNotNone(volumel)
        self.assertIsNotNone(volumew)
        self.assertIsNotNone(volumeh)
        self.assertIsNotNone(therm)
        self.assertIsNotNone(therml)
        self.assertIsNotNone(thermg)
        self.assertIsNotNone(thermw)
        self.assertIsNotNone(thermh)

    def test_volume_and_therm_smoke(self):
        s = self._make_state()
        v0 = volume.volume(0, 12.0, s)
        vl = volumel.volumel(0, 12.0, s)
        vw = volumew.volumew(0, 1000.0, s)
        vh = volumeh.volumeh(0, 12.0, s)

        self.assertTrue(np.isfinite(v0))
        self.assertTrue(np.isfinite(vl))
        self.assertTrue(np.isfinite(vw))
        self.assertTrue(np.isfinite(vh))

        t0 = therm.therm(0, max(v0, 1.0), max(v0, 1.0), s)
        tl = therml.therml(0, max(vl, 1.0), max(vl, 1.0), s)
        tg = thermg.thermg(0, max(vh, 1.0), max(vh, 1.0), s)
        tw = thermw.thermw(0, max(vw, 1.0), max(vw, 1.0), s)
        th = thermh.thermh(0, max(vh, 1.0), max(vh, 1.0), s)

        self.assertTrue(np.isfinite(t0.Cp))
        self.assertTrue(np.isfinite(tl.Cp))
        self.assertTrue(np.isfinite(tg.Cp))
        self.assertTrue(np.isfinite(tw.Cp))
        self.assertTrue(np.isfinite(th.Cp))


if __name__ == '__main__':
    unittest.main()
