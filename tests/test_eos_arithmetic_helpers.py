"""
Unit tests for newly translated Fortran helpers.

Tests: bserch, thetacal, qr19, landau, landauqr.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add src to path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import unittest
from unittest.mock import MagicMock

import numpy as np

# Direct imports from modules to avoid circular deps
from nMELTS.engine.EOS_arithmetic.bserch import bserch
from nMELTS.engine.EOS_arithmetic.thetacal import thetacal
from nMELTS.engine.EOS_arithmetic.qr19 import qr19, _load_adqref_tables
from nMELTS.engine.EOS_arithmetic.landau import landau
from nMELTS.engine.EOS_arithmetic.landauqr import landauqr
from nMELTS.engine.EOS_arithmetic.state import HeFESToState


class TestBserch(unittest.TestCase):
    """Tests for binary search helper."""

    def test_bserch_basic(self):
        """Test basic binary search."""
        xx = (1.0, 2.0, 3.0, 4.0, 5.0)
        self.assertEqual(bserch(xx, 2.5), 1)  # Between 2 and 3
        self.assertEqual(bserch(xx, 1.5), 0)  # Between 1 and 2
        self.assertEqual(bserch(xx, 4.9), 3)  # Between 4 and 5

    def test_bserch_boundaries(self):
        """Test edge cases."""
        xx = (1.0, 2.0, 3.0)
        self.assertEqual(bserch(xx, 0.5), -1)  # Below min
        self.assertEqual(bserch(xx, 1.0), 0)   # At min
        self.assertEqual(bserch(xx, 3.0), 2)   # At max
        self.assertEqual(bserch(xx, 3.5), 2)   # Above max

    def test_bserch_empty(self):
        """Test empty array."""
        xx = ()
        self.assertEqual(bserch(xx, 1.0), -1)


class TestThetacal(unittest.TestCase):
    """Tests for theta calibration."""

    def test_thetacal_edge_cases(self):
        """Test edge cases."""
        self.assertEqual(thetacal(0.0), 0.0)
        self.assertEqual(thetacal(-0.5), 0.0)
        self.assertEqual(thetacal(1.0), 1.0)
        self.assertEqual(thetacal(1.5), 1.0)

    def test_thetacal_intermediate(self):
        """Test intermediate values."""
        # Should be between 0 and 1 for valid input
        result = thetacal(0.5)
        self.assertGreater(result, 0.0)
        self.assertLessEqual(result, 1.0)


class TestQR19(unittest.TestCase):
    """Tests for attenuation Q calculation."""

    def test_qr19_basic(self):
        """Test basic Q computation."""
        qs, qp = qr19(100.0, 1500.0)
        self.assertGreater(qs, 0.0)
        self.assertGreater(qp, 0.0)
        self.assertGreater(qp, qs)  # Qp > Qs for Poisson solid

    def test_qr19_ti_zero(self):
        """Test Ti <= 0 edge case."""
        qs, qp = qr19(100.0, 0.0)
        # Should return large Q values (qslarge ~ 9999)
        self.assertGreater(qs, 1000.0)

    def test_qr19_adqref_tables(self):
        """Test that adiabat tables load."""
        tables = _load_adqref_tables()
        self.assertIn("dad", tables)
        self.assertIn("tad", tables)
        self.assertEqual(len(tables["dad"]), 1401)
        self.assertEqual(len(tables["tad"]), 1401)


class TestLandau(unittest.TestCase):
    """Tests for Landau transition properties."""

    def setUp(self):
        """Set up test state."""
        self.state = HeFESToState()
        self.state.nspecp = 10
        self.state.apar = np.zeros((10, 50))
        self.state.Ti = 1000.0
        self.state.Pi = 0.0  # GPa

    def test_landau_no_transition(self):
        """Test when Tco <= 0 (no Landau transition)."""
        # Leave apar as zeros (Tco = 0)
        ispec = 0
        Vi = 40.0
        qorder, Tc, glan, cplan, alplan, betlan, slan, vlan = landau(ispec, Vi, self.state)
        self.assertEqual(qorder, 0.0)
        self.assertEqual(Tc, 0.0)

    def test_landau_above_transition(self):
        """Test when Ti > Tc (no transition)."""
        ispec = 0
        self.state.apar[ispec, 37] = 500.0  # Tco = 500 K
        self.state.apar[ispec, 38] = 10.0   # smax
        self.state.apar[ispec, 39] = 5.0    # vmax
        self.state.Ti = 1000.0
        self.state.Pi = 0.0
        
        qorder, Tc, glan, cplan, alplan, betlan, slan, vlan = landau(ispec, Vi=40.0, state=self.state)
        self.assertEqual(qorder, 0.0)

    def test_landau_below_transition(self):
        """Test when Ti < Tc (active transition)."""
        ispec = 0
        self.state.apar[ispec, 37] = 1000.0  # Tco = 1000 K
        self.state.apar[ispec, 38] = 10.0    # smax
        self.state.apar[ispec, 39] = 5.0     # vmax
        self.state.Ti = 500.0
        self.state.Pi = 1.0  # 1 GPa pressure
        
        qorder, Tc, glan, cplan, alplan, betlan, slan, vlan = landau(ispec, Vi=40.0, state=self.state)
        self.assertGreater(qorder, 0.0)
        self.assertGreater(Tc, 1000.0)  # Tc > Tco due to 1 GPa pressure
        self.assertLess(qorder, 2.0)  # Should be order param in [0, ~1.7]


class TestLandauQR(unittest.TestCase):
    """Tests for Landau Q-referenced transition properties."""

    def setUp(self):
        """Set up test state."""
        self.state = HeFESToState()
        self.state.nspecp = 10
        self.state.apar = np.zeros((10, 50))
        self.state.Ti = 1000.0
        self.state.Pi = 0.0

    def test_landauqr_no_transition(self):
        """Test when Tco <= 0."""
        ispec = 0
        qorder, Tc, glan, cplan, alplan, betlan, slan, vlan = landauqr(ispec, 40.0, self.state)
        self.assertEqual(qorder, 0.0)

    def test_landauqr_below_transition(self):
        """Test active Q-referenced transition."""
        ispec = 0
        self.state.apar[ispec, 37] = 1000.0
        self.state.apar[ispec, 38] = 10.0
        self.state.apar[ispec, 39] = 5.0
        self.state.Ti = 500.0
        self.state.Pi = 0.0
        
        qorder, Tc, glan, cplan, alplan, betlan, slan, vlan = landauqr(ispec, 40.0, self.state)
        self.assertGreater(qorder, 0.0)


if __name__ == "__main__":
    unittest.main()
