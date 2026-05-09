# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit
# for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2025 by the BurnMan team, released under the GNU
# GPL v2 or later.


import numpy as np
from sympy import Matrix, nsimplify
import warnings

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

from burnman.classes.material import Material, material_property, cached_property
from burnman.classes.mineral import Mineral
from burnman.classes.solution import Solution
from burnman.classes import averaging_schemes

from burnman.utils.reductions import independent_row_indices
from burnman.utils.chemistry import sum_formulae, sort_element_list_to_IUPAC_order
from burnman.utils.chemistry import reaction_matrix_as_strings
from burnman.eos import debye, birch_murnaghan as bm


def check_pairs(phases, fractions):
    if len(fractions) < 1:
        raise Exception("ERROR: we need at least one phase")

    if len(phases) != len(fractions):
        raise Exception("ERROR: different array lengths for phases and fractions")

    total = sum(fractions)
    if abs(total - 1.0) > 1e-10:
        raise Exception("ERROR: list of molar fractions does not add up to one")
    for p in phases:
        if not isinstance(p, Mineral):
            raise Exception(
                "ERROR: object of type " "%s" " is not of type Mineral" % (type(p))
            )


def _require_torch():
    if torch is None:
        raise ImportError(
            "VectorComposite requires PyTorch. Install torch to use the batched backend."
        )


def _as_tensor(value, device=None, dtype=None):
    _require_torch()
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _normalize_rows(tensor, eps=1.0e-12):
    total = tensor.sum(dim=-1, keepdim=True)
    if torch.any(total.abs() <= eps):
        raise ValueError("Cannot normalize a batch row with zero total abundance")
    return tensor / total


# ============================================================================
# Vectorized SLB3/SLBBase EOS Classes for Batched Thermodynamic Calculations
# ============================================================================

def _vector_grueneisen_parameter_slb(V_0, volume, gruen_0, q_0):
    """
    Vectorized Grüneisen parameter for SLB EOS.
    Accepts batched tensors [B, ...] for all arguments.
    Returns [B, ...] tensor.
    """
    x = V_0 / volume
    f = 0.5 * (torch.pow(x, 2.0 / 3.0) - 1.0)
    a1_ii = 6.0 * gruen_0
    a2_iikk = -12.0 * gruen_0 + 36.0 * gruen_0 * gruen_0 - 18.0 * q_0 * gruen_0
    nu_o_nu0_sq = 1.0 + a1_ii * f + 0.5 * a2_iikk * f * f
    return (1.0 / 6.0 / nu_o_nu0_sq) * (2.0 * f + 1.0) * (a1_ii + a2_iikk * f)


class VectorSLBBase:
    """
    Vectorized finite strain-Mie-Grueneisen-Debye equation of state.
    
    Accepts batched pressure [B], temperature [B], and volume [B] tensors.
    Returns batched property tensors [B].
    
    This class does not inherit from EquationOfState because it is a standalone
    batched EOS kernel designed to replace the scalar loop in evaluate_backend_properties().
    """

    def __init__(self, order=3, conductive=False):
        """
        Parameters
        ----------
        order : int
            Order of finite strain expansion (2 or 3). Default: 3
        conductive : bool
            Whether to include electronic contributions. Default: False
        """
        self.order = order
        self.conductive = conductive

    def _debye_temperature(self, x, params):
        """
        Finite strain approximation for Debye Temperature [K].
        x = ref_vol/vol (can be batched [B] or scalar)
        Returns debye temperature with same shape as x.
        """
        f = 0.5 * (torch.pow(x, 2.0 / 3.0) - 1.0)
        a1_ii = 6.0 * params["grueneisen_0"]
        a2_iikk = (
            -12.0 * params["grueneisen_0"]
            + 36.0 * torch.pow(params["grueneisen_0"], 2.0)
            - 18.0 * params["q_0"] * params["grueneisen_0"]
        )
        nu_o_nu0_sq = 1.0 + a1_ii * f + 0.5 * a2_iikk * f * f
        
        # Check for invalid volumes
        if torch.any(nu_o_nu0_sq <= 0.0):
            warnings.warn("Some volumes exceed valid range for SLB EOS")
        
        debye_T = params["Debye_0"] * torch.sqrt(torch.clamp(nu_o_nu0_sq, min=1e-10))
        return debye_T

    def _volume_dependent_q(self, x, params):
        """
        Finite strain approximation for q, the isotropic volume strain
        derivative of the grueneisen parameter.
        """
        f = 0.5 * (torch.pow(x, 2.0 / 3.0) - 1.0)
        a1_ii = 6.0 * params["grueneisen_0"]
        a2_iikk = (
            -12.0 * params["grueneisen_0"]
            + 36.0 * torch.pow(params["grueneisen_0"], 2.0)
            - 18.0 * params["q_0"] * params["grueneisen_0"]
        )
        nu_o_nu0_sq = 1.0 + a1_ii * f + 0.5 * a2_iikk * f * f
        gr = (1.0 / 6.0 / nu_o_nu0_sq) * (2.0 * f + 1.0) * (a1_ii + a2_iikk * f)
        
        # Avoid divide by zero if grueneisen_0 is near zero
        gruen_abs = torch.abs(params["grueneisen_0"])
        is_near_zero = gruen_abs < 1.0e-10
        
        q_standard = (
            1.0 / 9.0 * (
                18.0 * gr - 6.0
                - 0.5 / nu_o_nu0_sq * (2.0 * f + 1.0) * (2.0 * f + 1.0) * a2_iikk / gr
            )
        )
        q_simple = 1.0 / 9.0 * (18.0 * gr - 6.0)
        
        # Use torch.where for safe branching with batched tensors
        if isinstance(is_near_zero, torch.Tensor):
            q = torch.where(is_near_zero, q_simple, q_standard)
        else:
            q = q_standard if not is_near_zero else q_simple
        
        return q

    def _isotropic_eta_s(self, x, params):
        """
        Finite strain approximation for eta_s, the isotropic shear strain
        derivative of the grueneisen parameter.
        """
        f = 0.5 * (torch.pow(x, 2.0 / 3.0) - 1.0)
        a2_s = -2.0 * params["grueneisen_0"] - 2.0 * params["eta_s_0"]
        a1_ii = 6.0 * params["grueneisen_0"]
        a2_iikk = (
            -12.0 * params["grueneisen_0"]
            + 36.0 * torch.pow(params["grueneisen_0"], 2.0)
            - 18.0 * params["q_0"] * params["grueneisen_0"]
        )
        nu_o_nu0_sq = 1.0 + a1_ii * f + 0.5 * a2_iikk * torch.pow(f, 2.0)
        gr = (1.0 / 6.0 / nu_o_nu0_sq) * (2.0 * f + 1.0) * (a1_ii + a2_iikk * f)
        
        eta_s = -gr - (
            0.5 * torch.pow(nu_o_nu0_sq, -1.0) * torch.pow((2.0 * f) + 1.0, 2.0) * a2_s
        )
        return eta_s

    def gibbs_energy(self, pressure, temperature, volume, params):
        """
        Returns the Gibbs free energy [J/mol].
        All inputs should be torch tensors with leading batch dimension [B].
        """
        x = params["V_0"] / volume
        f = 0.5 * (torch.pow(x, 2.0 / 3.0) - 1.0)
        debye_T = self._debye_temperature(x, params)
        
        # Helmholtz free energy calculation
        F_quasiharmonic = debye.helmholtz_energy(
            temperature, debye_T, params["n"]
        ) - debye.helmholtz_energy(params["T_0"], debye_T, params["n"])
        
        b_iikk = 9.0 * params["K_0"]
        b_iikkmm = 27.0 * params["K_0"] * (params["Kprime_0"] - 4.0)
        
        F = (
            params["F_0"]
            + 0.5 * b_iikk * f * f * params["V_0"]
            + (1.0 / 6.0) * params["V_0"] * b_iikkmm * f * f * f
            + F_quasiharmonic
        )
        
        # Gibbs energy
        G = F + pressure * volume
        return G

    def entropy(self, pressure, temperature, volume, params):
        """
        Returns entropy [J/K/mol].
        """
        x = params["V_0"] / volume
        debye_T = self._debye_temperature(x, params)
        S = debye.entropy(temperature, debye_T, params["n"])
        return S

    def molar_heat_capacity_p(self, pressure, temperature, volume, params):
        """
        Returns heat capacity at constant pressure [J/K/mol].
        """
        alpha = self.thermal_expansivity(pressure, temperature, volume, params)
        K_T = self.isothermal_bulk_modulus_reuss(pressure, temperature, volume, params)
        C_v = self._molar_heat_capacity_v(pressure, temperature, volume, params)
        C_p = C_v + alpha * alpha * K_T * volume * temperature
        return C_p

    def _molar_heat_capacity_v(self, pressure, temperature, volume, params):
        """
        Returns heat capacity at constant volume [J/K/mol].
        """
        x = params["V_0"] / volume
        debye_T = self._debye_temperature(x, params)
        C_v = debye.molar_heat_capacity_v(temperature, debye_T, params["n"])
        return C_v

    def thermal_expansivity(self, pressure, temperature, volume, params):
        """
        Returns thermal expansivity [1/K].
        """
        x = params["V_0"] / volume
        debye_T = self._debye_temperature(x, params)
        C_v = debye.molar_heat_capacity_v(temperature, debye_T, params["n"])
        gr_slb = _vector_grueneisen_parameter_slb(
            params["V_0"], volume, params["grueneisen_0"], params["q_0"]
        )
        K = self.isothermal_bulk_modulus_reuss(pressure, temperature, volume, params)
        alpha = gr_slb * C_v / volume / K
        return alpha

    def isothermal_bulk_modulus_reuss(self, pressure, temperature, volume, params):
        """
        Returns isothermal bulk modulus [Pa].
        """
        x = params["V_0"] / volume
        T_0 = params["T_0"]
        debye_T = self._debye_temperature(x, params)
        gr = _vector_grueneisen_parameter_slb(
            params["V_0"], volume, params["grueneisen_0"], params["q_0"]
        )
        
        # Thermal energy contributions
        E_th = debye.thermal_energy(temperature, debye_T, params["n"])
        E_th_ref = debye.thermal_energy(T_0, debye_T, params["n"])
        
        C_v = debye.molar_heat_capacity_v(temperature, debye_T, params["n"])
        C_v_ref = debye.molar_heat_capacity_v(T_0, debye_T, params["n"])
        
        q = self._volume_dependent_q(x, params)
        
        # Bulk modulus from Birch-Murnaghan
        K_bm = bm.bulk_modulus_third_order(volume, params)
        
        K = (
            K_bm
            + (gr + 1.0 - q) * (gr / volume) * (E_th - E_th_ref)
            - (gr * gr / volume) * (C_v * temperature - C_v_ref * T_0)
        )
        
        return K

    def shear_modulus(self, pressure, temperature, volume, params):
        """
        Returns shear modulus [Pa].
        """
        x = params["V_0"] / volume
        T_0 = params["T_0"]
        debye_T = self._debye_temperature(x, params)
        eta_s = self._isotropic_eta_s(x, params)
        
        E_th = debye.thermal_energy(temperature, debye_T, params["n"])
        E_th_ref = debye.thermal_energy(T_0, debye_T, params["n"])
        
        if self.order == 2:
            G_bm = bm.shear_modulus_second_order(volume, params)
        elif self.order == 3:
            G_bm = bm.shear_modulus_third_order(volume, params)
        else:
            raise NotImplementedError("Only order 2 and 3 are supported")
        
        G = G_bm - eta_s * (E_th - E_th_ref) / volume
        return G


class VectorSLB3(VectorSLBBase):
    """
    Vectorized third-order SLB equation of state.
    
    This is a batched version of the SLB3 EOS that accepts and returns
    torch tensors with a leading batch dimension [B].
    """

    def __init__(self):
        super().__init__(order=3, conductive=False)
