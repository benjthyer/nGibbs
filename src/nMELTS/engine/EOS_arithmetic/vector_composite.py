# This file is a vectorized implementation of the composite class of BurnMan - a thermoelastic and thermodynamic toolkit
# for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2025 by the BurnMan team, released under the GNU
# GPL v2 or later.

# Vectorized implementation built by Ben J. Thyer 2026


import sys
import numpy as np
import warnings
from pathlib import Path


import torch




file_path = str(Path(__file__).parent)
if file_path not in sys.path:
    sys.path.insert(0, file_path)


from burnman.classes.mineral import Mineral
from burnman.eos import birch_murnaghan as bm
import burnman.classes.solutionmodel as vsm
from burnman.classes.solution import Solution

from .qr19 import qr19_vec as _qr19_vec
_HAS_QR19 = True
"""except (ImportError, ValueError):

    _HAS_QR19 = False
"""
logish_eps = [np.finfo(float).eps]

# When True, Bukowinski (1977) electronic contributions are added to SLB EOS
# thermodynamic properties for phases that define bel_0 and gel parameters.
INCLUDE_ELECTRONIC_CONTRIBUTIONS = True
INCLUDE_ANELASTIC_CONTRIBUTIONS = False

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

def Qk(Qp, Qs, Ks, G):
    """
    Compute the quality factor K for a given set of elastic parameters.
    Not used currently, as Qs and Qp are observational and not calculated phasewise.
    
    Qp: Compressional quality factor
    Qs: Shear quality factor
    Ks: Elastic Bulk modulus
    G:  Elastic Shear modulus
    """
    L = Ks + ((4/3) * G)

    return (Ks/L) / (  (1/Qp) - ( (1/Qs) * ((4/3)*(G/L)) )  )


# ============================================================================
# Vectorized SLB3/SLBBase EOS Classes for Batched Thermodynamic Calculations
# ============================================================================

# --- PyTorch batched helpers (operate on torch tensors with leading batch axis) ---
def torch_logish(x, eps=logish_eps):
    eps0 = float(eps[0])
    # x: torch tensor
    f_eps = 1.0 - x / eps0
    mask = x > eps0
    ln = torch.where(x <= eps0, torch.log(torch.tensor(eps0)) - f_eps - f_eps * f_eps / 2.0, torch.zeros_like(x))
    if mask.any():
        ln = torch.where(mask, torch.log(x), ln)
    return ln


def torch_inverseish(x, eps=logish_eps):
    eps0 = float(eps[0])
    mask = x > eps0
    oneoverx = torch.where(x <= eps0, torch.tensor(2.0 / eps0) - x / (eps0 * eps0), torch.zeros_like(x))
    if mask.any():
        oneoverx = torch.where(mask, 1.0 / x, oneoverx)
    return oneoverx


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

def bulk_modulus_third_order_elastic(volume, params):
    """
    Bulk modulus for the third order Birch-Murnaghan equation of state
    :cite:`Birch1947`.

    :param volume: Volume of the material in the same units as
        the reference volume.
    :type volume: float

    :param params: Parameter dictionary
    :type params: dict

    :return: Bulk modulus in the same units as the reference bulk modulus.
    :rtype: float
    """

    x = params["V_0"] / volume
    #print(f"bm3 volume: {volume}")
    #print(f"bm3 V_0: {params['V_0']}")
    f = 0.5 * (pow(x, 2.0 / 3.0) - 1.0)

    K = pow(1.0 + 2.0 * f, 5.0 / 2.0) * (
        params["K_0"]
        + (3.0 * params["K_0"] * params["Kprime_0"] - 5 * params["K_0"]) * f
        + 27.0
        / 2.0
        * (params["K_0"] * params["Kprime_0"] - 4.0 * params["K_0"])
        * f
        * f
    )
    return K

def _vector_debye_fn_cheb(x):
    """
    Torch-vectorized approximation of the Debye function D(x).
    Valid for x > 0.
    """
    x = torch.clamp(x, min=1.0e-12)
    coeffs = x.new_tensor(
        [
            2.707737068327440945 / 2.0,
            0.340068135211091751,
            -0.12945150184440869e-01,
            0.7963755380173816e-03,
            -0.546360009590824e-04,
            0.39243019598805e-05,
            -0.2894032823539e-06,
            0.217317613962e-07,
            -0.16542099950e-08,
            0.1272796189e-09,
            -0.987963460e-11,
            0.7725074e-12,
            -0.607797e-13,
            0.48076e-14,
            -0.3820e-15,
            0.305e-16,
            -0.24e-17,
        ]
    )

    t = x * x / 8.0 - 1.0
    x2 = 2.0 * t
    c0 = torch.full_like(x, coeffs[-2])
    c1 = torch.full_like(x, coeffs[-1])
    for i in range(3, coeffs.numel() + 1):
        tmp = c0
        c0 = coeffs[-i] - c1
        c1 = tmp + c1 * x2
    poly = c0 + c1 * t

    small = 1.0 - 3.0 * x / 8.0 + x * x / 20.0
    mid = poly - 0.375 * x
    large = 19.4818182068004875 / (x * x * x)
    return torch.where(x <= 4.0, torch.where(x < 2.0e-5, small, mid), large)


def _vector_thermal_energy(temperature, debye_temperature, n):
    x = debye_temperature / torch.clamp(temperature, min=1.0e-12)
    return 3.0 * n * 8.31446261815324 * temperature * _vector_debye_fn_cheb(x)


def _vector_molar_heat_capacity_v(temperature, debye_temperature, n):
    x = debye_temperature / torch.clamp(temperature, min=1.0e-12)
    debye_val = _vector_debye_fn_cheb(x)
    ex = torch.exp(torch.clamp(x, max=200.0))
    return 3.0 * n * 8.31446261815324 * (4.0 * debye_val - 3.0 * x / (ex - 1.0))


def _vector_entropy(temperature, debye_temperature, n):
    x = debye_temperature / torch.clamp(temperature, min=1.0e-12)
    debye_val = _vector_debye_fn_cheb(x)
    return n * 8.31446261815324 * (
        4.0 * debye_val - 3.0 * torch.log1p(-torch.exp(-torch.clamp(x, max=200.0)))
    )


def _vector_helmholtz_energy(temperature, debye_temperature, n):
    x = debye_temperature / torch.clamp(temperature, min=1.0e-12)
    return n * 8.31446261815324 * temperature * (
        3.0 * torch.log1p(-torch.exp(-torch.clamp(x, max=200.0)))
        - _vector_debye_fn_cheb(x)
    )


# ============================================================================
# Vectorized Bukowinski (1977) electronic contribution helpers
# Each function mirrors bukowinski_electronic.py but operates on torch tensors.
# ============================================================================

def _vec_el_helmholtz(temperature, volume, T_0, V_0, bel_0, gel):
    """Electronic Helmholtz energy [J/mol]: F_el = -0.5*bel_0*(V/V_0)^gel*(T^2 - T_0^2)"""
    return -0.5 * bel_0 * torch.pow(volume / V_0, gel) * (temperature ** 2 - T_0 ** 2)


def _vec_el_pressure(temperature, volume, T_0, V_0, bel_0, gel):
    """Electronic pressure [Pa]: P_el = 0.5*gel*bel_0*(V/V_0)^gel*(T^2-T_0^2)/V"""
    return 0.5 * gel * bel_0 * torch.pow(volume / V_0, gel) * (temperature ** 2 - T_0 ** 2) / volume


def _vec_el_KToverV(temperature, volume, T_0, V_0, bel_0, gel):
    """Electronic K_T/V [Pa/m^3/mol]: -(gel-1)*P_el/V"""
    return -(gel - 1.0) * _vec_el_pressure(temperature, volume, T_0, V_0, bel_0, gel) / volume


def _vec_el_entropy(temperature, volume, V_0, bel_0, gel):
    """Electronic entropy [J/K/mol]: S_el = bel_0*(V/V_0)^gel*T"""
    return bel_0 * torch.pow(volume / V_0, gel) * temperature


def _vec_el_CVoverT(volume, V_0, bel_0, gel):
    """Electronic Cv/T [J/K^2/mol]: Cv_el/T = bel_0*(V/V_0)^gel"""
    return bel_0 * torch.pow(volume / V_0, gel)


def _vec_el_aKT(temperature, volume, V_0, bel_0, gel):
    """Electronic alpha*K_T [Pa/K]: aKT_el = gel*bel_0*(V/V_0)^gel*T/V"""
    return gel * bel_0 * torch.pow(volume / V_0, gel) * temperature / volume


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

    def pressure(self, temperature, volume, params):
        """
        Vectorized pressure P(T, V) [Pa].
        """
        T_0 = params["T_0"]
        x = params["V_0"] / volume
        f = 0.5 * (torch.pow(x, 2.0 / 3.0) - 1.0)

        a1_ii = 6.0 * params["grueneisen_0"]
        a2_iikk = (
            -12.0 * params["grueneisen_0"]
            + 36.0 * torch.pow(params["grueneisen_0"], 2.0)
            - 18.0 * params["q_0"] * params["grueneisen_0"]
        )
        nu_o_nu0_sq = 1.0 + a1_ii * f + 0.5 * a2_iikk * f * f
        gr = (1.0 / 6.0 / nu_o_nu0_sq) * (2.0 * f + 1.0) * (a1_ii + a2_iikk * f)

        debye_temperature = params["Debye_0"] * torch.sqrt(torch.clamp(nu_o_nu0_sq, min=1.0e-10))
        E_th = _vector_thermal_energy(temperature, debye_temperature, params["n"])
        E_th_ref = _vector_thermal_energy(
            torch.full_like(temperature, T_0), debye_temperature, params["n"]
        )

        P = (
            (1.0 / 3.0)
            * torch.pow(1.0 + 2.0 * f, 2.5)
            * (9.0 * params["K_0"] * f + 0.5 * 27.0 * params["K_0"] * (params["Kprime_0"] - 4.0) * f * f)
            + gr * (E_th - E_th_ref) / volume
        )
        if INCLUDE_ELECTRONIC_CONTRIBUTIONS:
            P = P + _vec_el_pressure(
                temperature, volume, T_0, params["V_0"],
                params["bel_0"], params["gel"],
            )
        return P



    def volume(self, pressure, temperature, params, max_iter=12, tol=1, verbose=False):
        """
        Batched Newton-Raphson solve for V(P, T) using dP/dV = -K_T/V.
        
        Args:
            pressure: Target pressure [Pa], shape [B]
            temperature: Temperature [K], shape [B]
            params: Parameter dictionary with V_0, K_0, etc.
            max_iter: Maximum iterations per sample
            tol: Absolute residual tolerance [Pa]
            verbose: Enable detailed diagnostics
        
        Returns:
            V: Converged volumes, shape [B]
            converged: Convergence mask, shape [B]
        """
        V = torch.full_like(pressure, params["V_0"])
        converged = torch.zeros_like(pressure, dtype=torch.bool)
        
        residual_history = []  # Track convergence per iteration

        for iteration in range(max_iter):
            P_pred = self.pressure(temperature, V, params)
            residual = P_pred - pressure
            #residual_history.append(residual.detach().clone())
            
            # Convergence check: absolute residual within tolerance
            with torch.no_grad():
                converged_now = torch.abs(residual) <= tol
                converged = converged | converged_now

            if verbose and iteration == 0:
                print(f"\n[volume() diagnostics]")
                print(f"  Initial V_0: {params['V_0']}, target P: {pressure[0]:.2e} Pa")
            
            if verbose:
                n_conv = torch.sum(converged).item()
                print(f"  Iter {iteration}: converged {n_conv}/{len(pressure)}, "
                      f"max|residual|={torch.max(torch.abs(residual)).item():.2e} Pa")

            if torch.all(converged):
                if verbose:
                    print(f"  Converged in {iteration} iterations")
                break

            # Bulk modulus evaluated at CURRENT pressure (not zero!)
            K_T = self.isothermal_bulk_modulus_reuss(
                pressure, temperature, V, params
            )
            if verbose:
                print(f"  Iter {iteration}: K_T = {K_T} Pa")
            
            # Derivative: dP/dV = -K_T / V
            # Both should be positive; result should be negative
            V_safe = torch.clamp(V, min=1.0e-16)
            K_T_safe = torch.clamp(K_T, min=1.0e-16)  # K_T must be positive
            dP_dV = -K_T_safe / V_safe
            
            # Newton step
            delta_V = residual / dP_dV
            large_step = torch.abs(delta_V) > 0.1 * V
            #delta_V = torch.where(large_step, 0.1 * V, delta_V) # Cut off big steps?

            V_new = V - delta_V # Smaller steps to avoid negative K_T? 
            
            # Restrict volume excursions to avoid unphysical states
            V_min = 0.4 * params["V_0"]
            V_max = 1.6 * params["V_0"]
            V_clamped = torch.clamp(V_new, min=V_min, max=V_max)
            
            if verbose:
                # Check for pathological K_T or step sizes on first iteration
                bad_kt = K_T < 0
                if torch.any(bad_kt):
                    print(f"  WARNING: {torch.sum(bad_kt)} samples have K_T < 0")
                if torch.any(large_step):
                    print(f"  WARNING: {torch.sum(large_step)} samples have |delta_V| > 10% V")
            
            V = V_clamped

        n_unconverged = torch.sum(~converged).item()
        if verbose:
            if n_unconverged > 0:
                print(f"  FAILED TO CONVERGE: {n_unconverged} samples after {max_iter} iterations")
                #final_residuals = residual_history[-1]
                #print(f"    Final max|residual|: {torch.max(torch.abs(final_residuals)).item():.2e} Pa")

        if n_unconverged > 0:
            V[~converged] = params["V_0"]
            if verbose:
                print(f"min pressure unconverged: {torch.min(pressure[~converged]).item():.2e} Pa")
                print(f"max pressure unconverged: {torch.max(pressure[~converged]).item():.2e} Pa")
            #raise ValueError("Failed to converge")

        return V, torch.ones_like(converged, dtype=torch.bool) # Defaulting to V0, assuming EOS poorly calibrated outside of stability where we don't care anyway.

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
        F_quasiharmonic = _vector_helmholtz_energy(
            temperature, debye_T, params["n"]
        ) - _vector_helmholtz_energy(torch.full_like(temperature, params["T_0"]), debye_T, params["n"])
        
        b_iikk = 9.0 * params["K_0"]
        b_iikkmm = 27.0 * params["K_0"] * (params["Kprime_0"] - 4.0)
        
        F = (
            params["F_0"]
            + 0.5 * b_iikk * f * f * params["V_0"]
            + (1.0 / 6.0) * params["V_0"] * b_iikkmm * f * f * f
            + F_quasiharmonic
        )
        if INCLUDE_ELECTRONIC_CONTRIBUTIONS:
            F = F + _vec_el_helmholtz(
                temperature, volume, params["T_0"], params["V_0"],
                params["bel_0"], params["gel"],
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
        S = _vector_entropy(temperature, debye_T, params["n"])
        if INCLUDE_ELECTRONIC_CONTRIBUTIONS:
            S = S + _vec_el_entropy(
                temperature, volume, params["V_0"],
                params["bel_0"], params["gel"],
            )
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
        C_v = _vector_molar_heat_capacity_v(temperature, debye_T, params["n"])
        if INCLUDE_ELECTRONIC_CONTRIBUTIONS:
            C_v = C_v + temperature * _vec_el_CVoverT(
                volume, params["V_0"], params["bel_0"], params["gel"],
            )
        return C_v

    def thermal_expansivity(self, pressure, temperature, volume, params):
        """
        Returns thermal expansivity [1/K].
        """
        x = params["V_0"] / volume
        debye_T = self._debye_temperature(x, params)
        C_v = _vector_molar_heat_capacity_v(temperature, debye_T, params["n"])
        gr_slb = _vector_grueneisen_parameter_slb(
            params["V_0"], volume, params["grueneisen_0"], params["q_0"]
        )
        K = self.isothermal_bulk_modulus_reuss(pressure, temperature, volume, params)
        alpha = gr_slb * C_v / volume / K
        if INCLUDE_ELECTRONIC_CONTRIBUTIONS:
            alpha = alpha + _vec_el_aKT(
                temperature, volume, params["V_0"],
                params["bel_0"], params["gel"],
            ) / K
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
        E_th = _vector_thermal_energy(temperature, debye_T, params["n"])
        E_th_ref = _vector_thermal_energy(torch.full_like(temperature, T_0), debye_T, params["n"])
        
        C_v = _vector_molar_heat_capacity_v(temperature, debye_T, params["n"])
        C_v_ref = _vector_molar_heat_capacity_v(torch.full_like(temperature, T_0), debye_T, params["n"])
        
        q = self._volume_dependent_q(x, params)
        
        # Bulk modulus from Birch-Murnaghan
        K_bm = bulk_modulus_third_order_elastic(volume, params)
        K = (
            K_bm
            + (gr + 1.0 - q) * (gr / volume) * (E_th - E_th_ref)
            - (gr * gr / volume) * (C_v * temperature - C_v_ref * T_0)
        )
        if INCLUDE_ELECTRONIC_CONTRIBUTIONS:
            K = K + volume * _vec_el_KToverV(
                temperature, volume, T_0, params["V_0"],
                params["bel_0"], params["gel"],
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
        
        E_th = _vector_thermal_energy(temperature, debye_T, params["n"])
        E_th_ref = _vector_thermal_energy(torch.full_like(temperature, T_0), debye_T, params["n"])
        
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

vector_eos = VectorSLB3()

class VectorComposite:
    """
    Batched composite backend for tensor-based thermodynamic reductions.

    This first implementation focuses on P,T thermodynamics and batched
    composition reduction. It intentionally omits equilibration, reaction
    affinities, and other relaxation workflows.

    Expected batch layout:
    - component_abundances: [B, C]
    - phase_identity: [C, P] binary assignment matrix mapping each component
      to a phase
    - component properties: [B, C]
    - phase properties: [B, P]
    """

    def __init__(
        self,
        #component_names,
        #phase_names,
        phase_identity,
        component_abundances,
        phase_models=None,
        pressure=None,
        temperature=None,
        device=None,
        dtype=None,
    ):
        _require_torch()

        #self.component_names = list(component_names)
        #self.phase_names = list(phase_names)
        self.phase_identity = _as_tensor(phase_identity, device=device, dtype=dtype)
        self.component_abundances = _normalize_rows(
            _as_tensor(component_abundances, device=device, dtype=dtype)
        )

        if self.phase_identity.ndim != 2:
            raise ValueError("phase_identity must be a 2D tensor with shape [C, P]")
        if self.component_abundances.ndim != 2:
            raise ValueError("component_abundances must be a 2D tensor with shape [B, C]")
        if self.phase_identity.shape[0] != self.component_abundances.shape[1]:
            raise ValueError("phase_identity and component_abundances have incompatible shapes")

        self.pressure = None
        self.temperature = None
        self.phase_models = None
        self.endmembers_per_phase = None
        self.endmember_slices = None
        self.component_properties = {}
        self.phase_properties = {}
        self.endmember_properties = {}
        self.Qs = None
        self.Qp = None

        if phase_models is not None:
            self.set_phase_models(phase_models)

        if pressure is not None or temperature is not None:
            self.set_state(pressure, temperature)

    @property
    def batch_size(self):
        return self.component_abundances.shape[0]

    """@property
    def n_components(self):
        return len(self.component_names)"""

    @property
    def n_phases(self):
        return self.phase_identity.shape[1]

    @property
    def n_endmembers(self):
        if self.endmembers_per_phase is None:
            return None
        return int(sum(self.endmembers_per_phase))

    def to(self, device=None, dtype=None):
        self.phase_identity = self.phase_identity.to(device=device, dtype=dtype)
        self.component_abundances = self.component_abundances.to(device=device, dtype=dtype)
        self.pressure = None if self.pressure is None else self.pressure.to(device=device, dtype=dtype)
        self.temperature = None if self.temperature is None else self.temperature.to(device=device, dtype=dtype)
        self.Qs = None if self.Qs is None else self.Qs.to(device=device, dtype=dtype)
        self.Qp = None if self.Qp is None else self.Qp.to(device=device, dtype=dtype)
        self.component_properties = {
            name: value.to(device=device, dtype=dtype)
            for name, value in self.component_properties.items()
        }
        self.phase_properties = {
            name: value.to(device=device, dtype=dtype)
            for name, value in self.phase_properties.items()
        }
        return self

    def set_state(self, pressure, temperature):
        self.pressure = _as_tensor(
            pressure,
            device=self.component_abundances.device,
            dtype=self.component_abundances.dtype,
        )
        self.temperature = _as_tensor(
            temperature,
            device=self.component_abundances.device,
            dtype=self.component_abundances.dtype,
        )
        if self.pressure.ndim == 0:
            self.pressure = self.pressure.expand(self.batch_size)
        if self.temperature.ndim == 0:
            self.temperature = self.temperature.expand(self.batch_size)
        if self.pressure.shape != self.temperature.shape:
            raise ValueError("pressure and temperature must have the same batch shape")
        if self.pressure.shape[0] != self.batch_size:
            raise ValueError("pressure/temperature batch size must match component_abundances")

        if _HAS_QR19:
            P_GPa = self.pressure.detach().cpu().numpy() * 1e-9
            T_K = self.temperature.detach().cpu().numpy()
            Qs_np, Qp_np = _qr19_vec(P_GPa, T_K)
            print(f"Qs: {Qs_np[::1000]}, Qp: {Qp_np[::1000]}")
            dev, dt = self.pressure.device, self.pressure.dtype
            self.Qs = torch.as_tensor(Qs_np, device=dev, dtype=dt)
            self.Qp = torch.as_tensor(Qp_np, device=dev, dtype=dt)
            print(self.Qp)

        if self.phase_models is not None:
            self.evaluate_backend_properties()

    def set_composition(self, component_abundances):
        self.component_abundances = _normalize_rows(
            _as_tensor(
                component_abundances,
                device=self.component_abundances.device,
                dtype=self.component_abundances.dtype,
            )
        )
        if self.component_abundances.ndim != 2:
            raise ValueError("component_abundances must be a 2D tensor with shape [B, C]")
        if self.component_abundances.shape[1] != self.phase_identity.shape[0]:
            raise ValueError("component_abundances has incompatible component dimension")

    def set_phase_models(self, phase_models):
        if len(phase_models) != self.n_phases:
            raise ValueError("phase_models must match the number of phase names")

        #self.phase_models = list(phase_models)
        self.phase_models = dict(phase_models)

        self.endmembers_per_phase = []
        self.endmember_slices = []

        
        start = 0
        for phase_name, model in self.phase_models.items():
            #print(model)
            if getattr(model, "endmembers", None) is not None:
                count = model.n_endmembers
                """endmembers = model.endmembers
                smName = model.solution_model.__class__.__name__
                print(smName)
               
                self.phase_models[phase_name].solution_model = getattr(vsm, smName)(endmembers=endmembers) # overwrite solution models with vectorized code
                """
            else:
                #print(f"Mineral phase: {phase_name}")
                count = 1
            """ else:
                print(f'Unexpected phase type: {type(model)}: {model}')
                print(model.__class__.__name__)
                raise TypeError(
                    'phase_models must contain only burnman.Mineral or burnman.Solution objects'
                )"""

            self.endmembers_per_phase.append(count)
            self.endmember_slices.append(slice(start, start + count))
            start += count

        self.endmember_properties = {}

    def _print_scalar_fallback_warning(self, context):
        print(f"[VectorComposite Warning] Falling back to scalar computation: {context}")

    def evaluate_backend_properties(self):
        """
        Populate batched phase and endmember properties from attached phase models.
        
        For Mineral phases with SLB3 EOS, this uses vectorized torch kernels to
        compute properties over the entire batch simultaneously.
        For other phases (Solutions, etc.), falls back to scalar evaluation.
        """

        if self.phase_models is None:
            raise AttributeError("No phase_models have been attached to the VectorComposite")

        batch_size = self.batch_size
        device = self.component_abundances.device
        dtype = self.component_abundances.dtype

        phase_property_names = (
            "molar_gibbs",
            "molar_volume",
            "molar_mass",
            "molar_entropy",
            "molar_enthalpy",
            "molar_internal_energy",
            "molar_heat_capacity_p",
            "isothermal_bulk_modulus_reuss",
            "isentropic_bulk_modulus_reuss",
            "thermal_expansivity",
            "shear_modulus",
        )
        endmember_property_names = (
            "partial_gibbs",
            "partial_volumes",
            "partial_entropies",
        )

        self.phase_properties = {
            name: torch.empty((batch_size, self.n_phases), device=device, dtype=dtype)
            for name in phase_property_names
        }
        self.endmember_properties = {
            name: torch.empty((batch_size, self.n_endmembers), device=device, dtype=dtype)
            for name in endmember_property_names
        }

        for phase_index, (phase_name, phase) in enumerate(self.phase_models.items()):
            # Check if this is a Mineral with SLB3 EOS; if so, use vectorized kernel
            if phase.method == "SolutionMethod":
                self._evaluate_solution_with_vector(phase, phase_index, batch_size, device, dtype)
            else:
                self._evaluate_phase_with_vector_slb3(
                        phase, phase_index, batch_size, device, dtype, phase_property_names
                    )
            """if isinstance(phase, Mineral) and hasattr(phase, "method"): # Removed fallbacks 5/6/26
                method_name = phase.method.__class__.__name__
                print(f"Phase {phase_name} has method: {phase.method}, class {phase.method.__class__}, name {phase.method.__class__.__name__}")
                if method_name == "SLB3":
                    # Use vectorized EOS for this phase
                    self._evaluate_phase_with_vector_slb3(
                        phase, phase_index, batch_size, device, dtype, phase_property_names
                    )
                else:
                    # Fallback to scalar evaluation
                    self._print_scalar_fallback_warning(
                        f"phase '{getattr(phase, 'name', phase_index)}' uses EOS '{method_name}', not vectorized"
                    )
                    self._evaluate_phase_scalar(
                        phase, phase_index, batch_size, device, dtype, 
                        phase_property_names, endmember_property_names
                    )
            else:
                # If this is a Solution use a specialized vectorized handler
                if isinstance(phase, Solution):
                    try:
                        self._evaluate_solution_with_vector(phase, phase_index, batch_size, device, dtype)
                    except Exception:
                        self._print_scalar_fallback_warning(
                            f"solution phase '{getattr(phase, 'name', phase_index)}' vectorized handler failed; falling back to scalar"
                        )
                        self._evaluate_phase_scalar(
                            phase, phase_index, batch_size, device, dtype, 
                            phase_property_names, endmember_property_names
                        )
                else:
                    # Fallback to scalar evaluation for other non-vectorized types
                    self._print_scalar_fallback_warning(
                        f"phase '{getattr(phase, 'name', phase_index)}' type '{type(phase).__name__}' is not vectorized"
                    )
                    self._evaluate_phase_scalar(
                        phase, phase_index, batch_size, device, dtype, 
                        phase_property_names, endmember_property_names
                    )"""

        return self.phase_properties, self.endmember_properties

    def _evaluate_phase_with_vector_slb3(
        self, phase, phase_index, batch_size, device, dtype, phase_property_names
    ):
        """
        Use VectorSLB3 EOS to compute all batch samples at once for a Mineral phase.
        """
        # Create vectorized EOS kernel
        
        # Get phase parameters
        params = phase.params.copy()
        
        # Convert params to tensors for batched computation
        params_tensor = {
            k: _as_tensor(v, device=device, dtype=dtype) if isinstance(v, (int, float))
               else v
            for k, v in params.items()
        }
        
        # Ensure pressure and temperature are expanded to batch size
        P = self.pressure.to(device=device, dtype=dtype)
        T = self.temperature.to(device=device, dtype=dtype)
        
        # Strategy 1: batched Newton-Raphson volume solve.
        # Fall back to scalar per-sample set_state if any items do not converge.
        params_tensor.setdefault("T_0", _as_tensor(300.0, device=device, dtype=dtype))
        params_tensor.setdefault("F_0", _as_tensor(0.0, device=device, dtype=dtype))
        params_tensor.setdefault("bel_0", _as_tensor(0.0, device=device, dtype=dtype))
        params_tensor.setdefault("gel", _as_tensor(1.0, device=device, dtype=dtype))
        volumes, converged = vector_eos.volume(P, T, params_tensor)
        if not torch.all(converged):
            self._print_scalar_fallback_warning(
                f"volume solve did not converge for {int(torch.sum(~converged).item())} samples in phase '{getattr(phase, 'name', phase_index)}'"
            )
            for batch_idx in torch.where(~converged)[0].tolist():
                p_scalar = float(P[batch_idx].detach().cpu().item())
                t_scalar = float(T[batch_idx].detach().cpu().item())
                phase.set_state(p_scalar, t_scalar)
                volumes[batch_idx] = phase.molar_volume
        
        # Now call vectorized EOS methods with batched P, T, V
        try:
            g = vector_eos.gibbs_energy(P, T, volumes, params_tensor)
            self.phase_properties["molar_gibbs"][:, phase_index] = _as_tensor(g, device=device, dtype=dtype)
        except Exception:
            # Fallback if vectorized fails
            self._print_scalar_fallback_warning(
                f"molar_gibbs in phase '{getattr(phase, 'name', phase_index)}'"
            )
            for i in range(batch_size):
                self.phase_properties["molar_gibbs"][i, phase_index] = phase.molar_gibbs
        
        try:
            s = vector_eos.entropy(P, T, volumes, params_tensor)
            self.phase_properties["molar_entropy"][:, phase_index] = _as_tensor(s, device=device, dtype=dtype)
        except Exception:
            self._print_scalar_fallback_warning(
                f"molar_entropy in phase '{getattr(phase, 'name', phase_index)}'"
            )
            for i in range(batch_size):
                self.phase_properties["molar_entropy"][i, phase_index] = phase.molar_entropy
        
        try:
            h = self.phase_properties["molar_gibbs"][:, phase_index] + T * _as_tensor(s, device=device, dtype=dtype)
            self.phase_properties["molar_enthalpy"][:, phase_index] = h
        except Exception:
            self._print_scalar_fallback_warning(
                f"molar_enthalpy in phase '{getattr(phase, 'name', phase_index)}'"
            )
            for i in range(batch_size):
                self.phase_properties["molar_enthalpy"][i, phase_index] = phase.molar_enthalpy
        
        try:
            u = (
                self.phase_properties["molar_gibbs"][:, phase_index] 
                - P * _as_tensor(volumes, device=device, dtype=dtype) 
                + T * _as_tensor(s, device=device, dtype=dtype)
            )
            self.phase_properties["molar_internal_energy"][:, phase_index] = u
        except Exception:
            self._print_scalar_fallback_warning(
                f"molar_internal_energy in phase '{getattr(phase, 'name', phase_index)}'"
            )
            for i in range(batch_size):
                self.phase_properties["molar_internal_energy"][i, phase_index] = phase.molar_internal_energy
        
        try:
            v = _as_tensor(volumes, device=device, dtype=dtype)
            self.phase_properties["molar_volume"][:, phase_index] = v
        except Exception:
            self._print_scalar_fallback_warning(
                f"molar_volume in phase '{getattr(phase, 'name', phase_index)}'"
            )
            for i in range(batch_size):
                self.phase_properties["molar_volume"][i, phase_index] = phase.molar_volume

        try:
            self.phase_properties["molar_mass"][:, phase_index] = _as_tensor(
                phase.molar_mass, device=device, dtype=dtype
            ).expand(batch_size)
        except Exception:
            self._print_scalar_fallback_warning(
                f"molar_mass in phase '{getattr(phase, 'name', phase_index)}'"
            )
            for i in range(batch_size):
                self.phase_properties["molar_mass"][i, phase_index] = phase.molar_mass
        
        try:
            cp = vector_eos.molar_heat_capacity_p(P, T, volumes, params_tensor)
            self.phase_properties["molar_heat_capacity_p"][:, phase_index] = _as_tensor(cp, device=device, dtype=dtype)
        except Exception:
            self._print_scalar_fallback_warning(
                f"molar_heat_capacity_p in phase '{getattr(phase, 'name', phase_index)}'"
            )
            for i in range(batch_size):
                self.phase_properties["molar_heat_capacity_p"][i, phase_index] = phase.molar_heat_capacity_p
        
        try:
            K = vector_eos.isothermal_bulk_modulus_reuss(P, T, volumes, params_tensor)
            self.phase_properties["isothermal_bulk_modulus_reuss"][:, phase_index] = _as_tensor(K, device=device, dtype=dtype)
        except Exception:
            self._print_scalar_fallback_warning(
                f"isothermal_bulk_modulus_reuss in phase '{getattr(phase, 'name', phase_index)}'"
            )
            for i in range(batch_size):
                self.phase_properties["isothermal_bulk_modulus_reuss"][i, phase_index] = phase.isothermal_bulk_modulus_reuss
        
        try:
            alpha = vector_eos.thermal_expansivity(P, T, volumes, params_tensor)
            self.phase_properties["thermal_expansivity"][:, phase_index] = _as_tensor(alpha, device=device, dtype=dtype)
        except Exception:
            self._print_scalar_fallback_warning(
                f"thermal_expansivity in phase '{getattr(phase, 'name', phase_index)}'"
            )
            for i in range(batch_size):
                self.phase_properties["thermal_expansivity"][i, phase_index] = phase.thermal_expansivity
        
        try:
            G = vector_eos.shear_modulus(P, T, volumes, params_tensor)
            self.phase_properties["shear_modulus"][:, phase_index] = _as_tensor(G, device=device, dtype=dtype)
        except Exception:
            self._print_scalar_fallback_warning(
                f"shear_modulus in phase '{getattr(phase, 'name', phase_index)}'"
            )
            for i in range(batch_size):
                if hasattr(phase, "shear_modulus"):
                    self.phase_properties["shear_modulus"][i, phase_index] = phase.shear_modulus
                else:
                    self.phase_properties["shear_modulus"][i, phase_index] = 0.0

        # Per-phase K_S: compute before assemblage averaging so VRH of K_S is correct
        try:
            K_T_ph = self.phase_properties["isothermal_bulk_modulus_reuss"][:, phase_index]
            alpha_ph = self.phase_properties["thermal_expansivity"][:, phase_index]
            Cp_ph = self.phase_properties["molar_heat_capacity_p"][:, phase_index]
            V_ph = _as_tensor(volumes, device=device, dtype=dtype)
            denom = 1.0 - (alpha_ph ** 2) * K_T_ph * T * V_ph / Cp_ph.clamp(min=1e-30)
            denom = denom.clamp(min=1e-10)
            self.phase_properties["isentropic_bulk_modulus_reuss"][:, phase_index] = K_T_ph / denom
        except Exception:
            self._print_scalar_fallback_warning(
                f"isentropic_bulk_modulus_reuss in phase '{getattr(phase, 'name', phase_index)}'"
            )

        # Handle endmember properties
        endmember_slice = self.endmember_slices[phase_index]
        self.endmember_properties["partial_gibbs"][:, endmember_slice] = self.phase_properties[
            "molar_gibbs"
        ][:, phase_index].unsqueeze(-1)
        self.endmember_properties["partial_volumes"][:, endmember_slice] = self.phase_properties[
            "molar_volume"
        ][:, phase_index].unsqueeze(-1)
        self.endmember_properties["partial_entropies"][:, endmember_slice] = self.phase_properties[
            "molar_entropy"
        ][:, phase_index].unsqueeze(-1)

    def _evaluate_phase_scalar(
        self, phase, phase_index, batch_size, device, dtype,
        phase_property_names, endmember_property_names
    ):
        """
        Scalar evaluation loop for each batch sample.
        Used as fallback for phases without vectorized EOS support.
        """
        for batch_index in range(batch_size):
            pressure = float(self.pressure[batch_index].detach().cpu().item())
            temperature = float(self.temperature[batch_index].detach().cpu().item())

            phase.set_state(pressure, temperature)

            for name in phase_property_names:
                if name == "shear_modulus" and not hasattr(phase, "shear_modulus"):
                    value = 0.0
                else:
                    value = getattr(phase, name)
                self.phase_properties[name][batch_index, phase_index] = value

            endmember_slice = self.endmember_slices[phase_index]
            if isinstance(phase, Solution):
                self.endmember_properties["partial_gibbs"][batch_index, endmember_slice] = _as_tensor(
                    phase.partial_gibbs, device=device, dtype=dtype
                )
                self.endmember_properties["partial_volumes"][batch_index, endmember_slice] = _as_tensor(
                    phase.partial_volumes, device=device, dtype=dtype
                )
                self.endmember_properties["partial_entropies"][batch_index, endmember_slice] = _as_tensor(
                    phase.partial_entropies, device=device, dtype=dtype
                )
            else:
                self.endmember_properties["partial_gibbs"][batch_index, endmember_slice] = _as_tensor(
                    [phase.molar_gibbs], device=device, dtype=dtype
                )
                self.endmember_properties["partial_volumes"][batch_index, endmember_slice] = _as_tensor(
                    [phase.molar_volume], device=device, dtype=dtype
                )
                self.endmember_properties["partial_entropies"][batch_index, endmember_slice] = _as_tensor(
                    [phase.molar_entropy], device=device, dtype=dtype
                )

    def _evaluate_solution_with_vector(self, phase, phase_index, batch_size, device, dtype):
        """
        Vectorized handling for Solution phases: evaluate each endmember across
        the batch (vectorized when possible) and then compute excess terms.
        For MechanicalSolution the excesses are zero and fully vectorized;
        for other solution models we compute excess terms per-sample (fallback).
        """
        endmember_slice = self.endmember_slices[phase_index]
        endmembers = phase.solution_model.endmembers
        n_end = len(endmembers)

        P = self.pressure.to(device=device, dtype=dtype)
        T = self.temperature.to(device=device, dtype=dtype)

        # Buffers for endmember properties: shape [B, n_end]
        gibbs = torch.empty((batch_size, n_end), device=device, dtype=dtype)
        vol = torch.empty((batch_size, n_end), device=device, dtype=dtype)
        ent = torch.empty((batch_size, n_end), device=device, dtype=dtype)
        K_end = torch.empty((batch_size, n_end), device=device, dtype=dtype)
        G_end = torch.empty((batch_size, n_end), device=device, dtype=dtype)
        cp_end = torch.empty((batch_size, n_end), device=device, dtype=dtype)

        # Evaluate each endmember (vectorized across batch where supported)
        for j, mbr in enumerate(endmembers):
            em = mbr[0]
            #print(em)
            #print(mbr)
            # If the endmember uses SLB3 we can vector-evaluate using VectorSLB3
            if hasattr(em, "method") and em.method.__class__.__name__ == "SLB3":

                params = em.params.copy()
                params_tensor = {
                    k: _as_tensor(v, device=device, dtype=dtype) if isinstance(v, (int, float)) else v
                    for k, v in params.items()
                }
                params_tensor.setdefault("T_0", _as_tensor(300.0, device=device, dtype=dtype))
                params_tensor.setdefault("F_0", _as_tensor(0.0, device=device, dtype=dtype))
                params_tensor.setdefault("bel_0", _as_tensor(0.0, device=device, dtype=dtype))
                params_tensor.setdefault("gel", _as_tensor(1.0, device=device, dtype=dtype))

                vols_j, converged = vector_eos.volume(P, T, params_tensor)
                if not torch.all(converged):
                    self._print_scalar_fallback_warning(
                        f"volume solve did not converge for {int(torch.sum(~converged).item())} samples in endmember '{getattr(em, 'name', j)}'"
                    )
                    for idx in torch.where(~converged)[0].tolist():
                        p_scalar = float(P[idx].detach().cpu().item())
                        t_scalar = float(T[idx].detach().cpu().item())
                        em.set_state(p_scalar, t_scalar)
                        vols_j[idx] = em.molar_volume

                # Compute vectorized properties
                try:
                    g_j = vector_eos.gibbs_energy(P, T, vols_j, params_tensor)
                    s_j = vector_eos.entropy(P, T, vols_j, params_tensor)
                    K_j = vector_eos.isothermal_bulk_modulus_reuss(P, T, vols_j, params_tensor)
                    G_j = vector_eos.shear_modulus(P, T, vols_j, params_tensor)
                    cp_j = vector_eos.molar_heat_capacity_p(P, T, vols_j, params_tensor)

                    gibbs[:, j] = _as_tensor(g_j, device=device, dtype=dtype)
                    vol[:, j] = _as_tensor(vols_j, device=device, dtype=dtype)
                    ent[:, j] = _as_tensor(s_j, device=device, dtype=dtype)
                    K_end[:, j] = _as_tensor(K_j, device=device, dtype=dtype)
                    G_end[:, j] = _as_tensor(G_j, device=device, dtype=dtype)
                    cp_end[:, j] = _as_tensor(cp_j, device=device, dtype=dtype)
                except Exception:
                    self._print_scalar_fallback_warning(
                        f"vectorized endmember evaluation failed for '{getattr(em, 'name', j)}' - falling back to scalar per-sample"
                    )
                    for i in range(batch_size):
                        p_i = float(P[i].detach().cpu().item())
                        t_i = float(T[i].detach().cpu().item())
                        em.set_state(p_i, t_i)
                        gibbs[i, j] = em.gibbs
                        vol[i, j] = em.molar_volume
                        ent[i, j] = em.molar_entropy
                        try:
                            K_end[i, j] = em.isothermal_bulk_modulus_reuss
                        except Exception:
                            K_end[i, j] = 0.0
                        try:
                            G_end[i, j] = em.shear_modulus
                        except Exception:
                            G_end[i, j] = 0.0
                        try:
                            cp_end[i, j] = em.molar_heat_capacity_p
                        except Exception:
                            cp_end[i, j] = 0.0
            else:
                # Non-vectorized endmember: evaluate scalar for each batch sample
                self._print_scalar_fallback_warning(
                    f"endmember '{getattr(em, 'name', j)}' not vectorized; computing per-sample"
                )
                for i in range(batch_size):
                    p_i = float(P[i].detach().cpu().item())
                    t_i = float(T[i].detach().cpu().item())
                    em.set_state(p_i, t_i)
                    gibbs[i, j] = em.gibbs
                    vol[i, j] = em.molar_volume
                    ent[i, j] = em.molar_entropy
                    try:
                        K_end[i, j] = em.isothermal_bulk_modulus_reuss
                    except Exception:
                        K_end[i, j] = 0.0
                    try:
                        G_end[i, j] = em.shear_modulus
                    except Exception:
                        G_end[i, j] = 0.0
                    try:
                        cp_end[i, j] = em.molar_heat_capacity_p
                    except Exception:
                        cp_end[i, j] = 0.0

        # Compute per-phase normalized endmember fractions for each batch sample
        comp_abund = self.component_abundances[:, endmember_slice]
        sums = comp_abund.sum(-1, keepdim=True)
        # Avoid division by zero: where sums == 0, leave fractions as zeros
        mask = sums > 0
        f = torch.zeros_like(comp_abund, device=device, dtype=dtype)
        if mask.any():
            f[mask.squeeze(-1)] = comp_abund[mask.squeeze(-1)] / sums[mask.squeeze(-1)]

        sm = phase.solution_model
        #print(sm.__class__.__name__)

        # Prepare excess arrays
        excess_partial_gibbs = torch.zeros_like(gibbs, device=device, dtype=dtype)
        excess_partial_volumes = torch.zeros_like(vol, device=device, dtype=dtype)
        excess_partial_entropies = torch.zeros_like(ent, device=device, dtype=dtype)
        excess_gibbs = torch.zeros((batch_size,), device=device, dtype=dtype)
        excess_volume = torch.zeros((batch_size,), device=device, dtype=dtype)
        excess_entropy = torch.zeros((batch_size,), device=device, dtype=dtype)

        # Vectorized for MechanicalSolution (zero excess); otherwise compute via batched model if available
        if not (sm.__class__.__name__ == "MechanicalSolution"):
            # If the solution model provides batched torch interfaces, use them
            if hasattr(sm, "batched_excess_partial_gibbs_energies"):
                try:
                    epg = sm.batched_excess_partial_gibbs_energies(P, T, f)
                    epv = sm.batched_excess_partial_volumes(P, T, f)
                    epe = sm.batched_excess_partial_entropies(P, T, f)
                    eg = sm.batched_excess_gibbs_energy(P, T, f)

                    excess_partial_gibbs = epg.to(device=device, dtype=dtype)
                    excess_partial_volumes = epv.to(device=device, dtype=dtype)
                    excess_partial_entropies = epe.to(device=device, dtype=dtype)
                    excess_gibbs = eg.to(device=device, dtype=dtype)

                    # aggregate scalar excesses from partials
                    excess_volume = (f * excess_partial_volumes).sum(-1)
                    excess_entropy = (f * excess_partial_entropies).sum(-1)
                except Exception as e:
                    raise RuntimeError(f"Batched solution-model excess calculation failed for phase '{getattr(phase, 'name', phase_index)}': {e}")
                    self._print_scalar_fallback_warning(
                        f"batched solution-model excess calculation failed for phase '{getattr(phase, 'name', phase_index)}'; falling back to per-sample"
                    )
                    # fallback to per-sample numpy-based methods
                    for i in range(batch_size):
                        p_i = float(P[i].detach().cpu().item())
                        t_i = float(T[i].detach().cpu().item())
                        f_i = f[i].detach().cpu().numpy()
                        try:
                            epg = sm.excess_partial_gibbs_energies(p_i, t_i, f_i)
                            epv = sm.excess_partial_volumes(p_i, t_i, f_i)
                            epe = sm.excess_partial_entropies(p_i, t_i, f_i)
                            eg = sm.excess_gibbs_energy(p_i, t_i, f_i)
                            ev = sm.excess_volume(p_i, t_i, f_i)
                            ee = sm.excess_entropy(p_i, t_i, f_i)
                        except Exception as e:
                            print(e)
                            raise Exception("Solution model excess calculation failed")
                            self._print_scalar_fallback_warning(
                                f"solution model excess calculation failed for sample {i} in phase '{getattr(phase, 'name', phase_index)}'"
                            )
                            continue
                        excess_partial_gibbs[i] = torch.from_numpy(np.asarray(epg)).to(device=device, dtype=dtype)
                        excess_partial_volumes[i] = torch.from_numpy(np.asarray(epv)).to(device=device, dtype=dtype)
                        excess_partial_entropies[i] = torch.from_numpy(np.asarray(epe)).to(device=device, dtype=dtype)
                        excess_gibbs[i] = float(eg)
                        excess_volume[i] = float(ev)
                        excess_entropy[i] = float(ee)
            else:
                # per-sample fallback for excesses using solution model's numpy-based methods
                for i in range(batch_size):
                    p_i = float(P[i].detach().cpu().item())
                    t_i = float(T[i].detach().cpu().item())
                    f_i = f[i].detach().cpu().numpy()
                    try:
                        epg = sm.excess_partial_gibbs_energies(p_i, t_i, f_i)
                        epv = sm.excess_partial_volumes(p_i, t_i, f_i)
                        epe = sm.excess_partial_entropies(p_i, t_i, f_i)
                        eg = sm.excess_gibbs_energy(p_i, t_i, f_i)
                        ev = sm.excess_volume(p_i, t_i, f_i)
                        ee = sm.excess_entropy(p_i, t_i, f_i)
                    except Exception as e:
                        # If model fails, warn and leave excesses zero for this sample
                        print(e)
                        raise Exception("Solution model excess calculation failed")
                        self._print_scalar_fallback_warning(
                            f"solution model excess calculation failed for sample {i} in phase '{getattr(phase, 'name', phase_index)}'"
                        )
                        continue
                    excess_partial_gibbs[i] = torch.from_numpy(np.asarray(epg)).to(device=device, dtype=dtype)
                    excess_partial_volumes[i] = torch.from_numpy(np.asarray(epv)).to(device=device, dtype=dtype)
                    excess_partial_entropies[i] = torch.from_numpy(np.asarray(epe)).to(device=device, dtype=dtype)
                    excess_gibbs[i] = float(eg)
                    excess_volume[i] = float(ev)
                    excess_entropy[i] = float(ee)

        # Partial properties = endmember contribution + excess partials
        # (excess already contains ideal Temkin mixing for IdealSolution/AsymmetricRegularSolution)
        partial_gibbs = gibbs + excess_partial_gibbs
        partial_volumes = vol + excess_partial_volumes
        partial_entropies = ent + excess_partial_entropies

        # Store endmember partials into global endmember_properties
        self.endmember_properties["partial_gibbs"][:, endmember_slice] = partial_gibbs
        self.endmember_properties["partial_volumes"][:, endmember_slice] = partial_volumes
        self.endmember_properties["partial_entropies"][:, endmember_slice] = partial_entropies

        # Aggregate to phase properties
        # excess_gibbs / excess_entropy already include ideal Temkin configurational terms
        molar_gibbs_phase = (f * gibbs).sum(-1) + excess_gibbs
        molar_volume_phase = (f * vol).sum(-1) + excess_volume
        molar_entropy_phase = (f * ent).sum(-1) + excess_entropy

        self.phase_properties["molar_gibbs"][:, phase_index] = molar_gibbs_phase
        self.phase_properties["molar_volume"][:, phase_index] = molar_volume_phase
        self.phase_properties["molar_entropy"][:, phase_index] = molar_entropy_phase

        # Molar mass: weighted average of endmember molar masses
        masses = torch.tensor([mbr[0].molar_mass for mbr in endmembers], device=device, dtype=dtype)
        self.phase_properties["molar_mass"][:, phase_index] = (f * masses).sum(-1)

        # Enthalpy and internal energy
        h_phase = molar_gibbs_phase + T * molar_entropy_phase
        u_phase = molar_gibbs_phase - P * molar_volume_phase + T * molar_entropy_phase
        self.phase_properties["molar_enthalpy"][:, phase_index] = h_phase
        self.phase_properties["molar_internal_energy"][:, phase_index] = u_phase

        # Cp_p: weighted sum + excess (use batched if available, otherwise per-sample)
        try:
            cp_phase = (f * cp_end).sum(-1)
            if hasattr(sm, "batched_Cp_excess") and not (sm.__class__.__name__ == "MechanicalSolution"):
                try:
                    cp_excess = sm.batched_Cp_excess(P, T, f)
                    cp_phase = cp_phase + cp_excess.to(device=device, dtype=dtype)
                except Exception:
                    self._print_scalar_fallback_warning(
                        f"batched Cp_excess calculation failed for phase '{getattr(phase, 'name', phase_index)}'; using per-sample fallback"
                    )
                    if hasattr(sm, "Cp_excess"):
                        cp_excess = torch.zeros((batch_size,), device=device, dtype=dtype)
                        for i in range(batch_size):
                            p_i = float(P[i].detach().cpu().item())
                            t_i = float(T[i].detach().cpu().item())
                            f_i = f[i].detach().cpu().numpy()
                            try:
                                cp_ex = sm.Cp_excess(p_i, t_i, f_i)
                                cp_excess[i] = float(cp_ex)
                            except Exception:
                                pass
                        cp_phase = cp_phase + cp_excess
            elif hasattr(sm, "Cp_excess") and not (sm.__class__.__name__ == "MechanicalSolution"):
                cp_excess = torch.zeros((batch_size,), device=device, dtype=dtype)
                for i in range(batch_size):
                    p_i = float(P[i].detach().cpu().item())
                    t_i = float(T[i].detach().cpu().item())
                    f_i = f[i].detach().cpu().numpy()
                    try:
                        cp_ex = sm.Cp_excess(p_i, t_i, f_i)
                        cp_excess[i] = float(cp_ex)
                    except Exception:
                        pass
                cp_phase = cp_phase + cp_excess
            self.phase_properties["molar_heat_capacity_p"][:, phase_index] = cp_phase
        except Exception:
            self._print_scalar_fallback_warning(
                f"molar_heat_capacity_p aggregation failed for phase '{getattr(phase, 'name', phase_index)}'"
            )

        # Isothermal bulk modulus (Reuss mixing): V/K = sum Xi Vi/Ki + excess
        # Compute V_over_K from endmembers
        # Avoid division by zero in K_end
        K_safe = K_end.clone()
        K_safe[K_safe == 0] = 1e-30
        VoverK = (f * (vol / K_safe)).sum(-1)
        # Add model excess V/K if available (use batched if present)
        if hasattr(sm, "batched_VoverKT_excess") and not (sm.__class__.__name__ == "MechanicalSolution"):
            try:
                VoverK_ex = sm.batched_VoverKT_excess(P, T, f)
                VoverK = VoverK + VoverK_ex.to(device=device, dtype=dtype)
            except Exception:
                self._print_scalar_fallback_warning(
                    f"batched VoverKT_excess calculation failed for phase '{getattr(phase, 'name', phase_index)}'; using per-sample fallback"
                )
                if hasattr(sm, "VoverKT_excess"):
                    VoverK_ex = torch.zeros((batch_size,), device=device, dtype=dtype)
                    for i in range(batch_size):
                        p_i = float(P[i].detach().cpu().item())
                        t_i = float(T[i].detach().cpu().item())
                        f_i = f[i].detach().cpu().numpy()
                        try:
                            VoverK_ex[i] = float(sm.VoverKT_excess(p_i, t_i, f_i))
                        except Exception:
                            pass
                    VoverK = VoverK + VoverK_ex
        elif hasattr(sm, "VoverKT_excess") and not (sm.__class__.__name__ == "MechanicalSolution"):
            VoverK_ex = torch.zeros((batch_size,), device=device, dtype=dtype)
            for i in range(batch_size):
                p_i = float(P[i].detach().cpu().item())
                t_i = float(T[i].detach().cpu().item())
                f_i = f[i].detach().cpu().numpy()
                try:
                    VoverK_ex[i] = float(sm.VoverKT_excess(p_i, t_i, f_i))
                except Exception:
                    pass
            VoverK = VoverK + VoverK_ex

        # Finally compute mixed K_T
        K_phase = molar_volume_phase / VoverK
        self.phase_properties["isothermal_bulk_modulus_reuss"][:, phase_index] = K_phase

        # Thermal expansivity: alphaV = sum Xi alpha_i V_i + excess -> alpha = alphaV / V
        try:
            # For endmembers that have thermal_expansivity attribute
            alpha_end = torch.empty_like(vol, device=device, dtype=dtype)
            for j in range(n_end):
                try:
                    # Many endmembers have scalar alpha
                    a = endmembers[j][0].thermal_expansivity
                    alpha_end[:, j] = _as_tensor(a, device=device, dtype=dtype).expand(batch_size)
                except Exception:
                    alpha_end[:, j] = 0.0
            alphaV = (f * (alpha_end * vol)).sum(-1)
            # add excess alphaV if model provides (use batched if available)
            if hasattr(sm, "batched_alphaV_excess") and not (sm.__class__.__name__ == "MechanicalSolution"):
                try:
                    alphaV_ex = sm.batched_alphaV_excess(P, T, f)
                    alphaV = alphaV + alphaV_ex.to(device=device, dtype=dtype)
                except Exception:
                    self._print_scalar_fallback_warning(
                        f"batched alphaV_excess calculation failed for phase '{getattr(phase, 'name', phase_index)}'; using per-sample fallback"
                    )
                    if hasattr(sm, "alphaV_excess"):
                        alphaV_ex = torch.zeros((batch_size,), device=device, dtype=dtype)
                        for i in range(batch_size):
                            p_i = float(P[i].detach().cpu().item())
                            t_i = float(T[i].detach().cpu().item())
                            f_i = f[i].detach().cpu().numpy()
                            try:
                                alphaV_ex[i] = float(sm.alphaV_excess(p_i, t_i, f_i))
                            except Exception:
                                pass
                        alphaV = alphaV + alphaV_ex
            elif hasattr(sm, "alphaV_excess") and not (sm.__class__.__name__ == "MechanicalSolution"):
                alphaV_ex = torch.zeros((batch_size,), device=device, dtype=dtype)
                for i in range(batch_size):
                    p_i = float(P[i].detach().cpu().item())
                    t_i = float(T[i].detach().cpu().item())
                    f_i = f[i].detach().cpu().numpy()
                    try:
                        alphaV_ex[i] = float(sm.alphaV_excess(p_i, t_i, f_i))
                    except Exception:
                        pass
                alphaV = alphaV + alphaV_ex
            alpha_phase = alphaV / molar_volume_phase
            self.phase_properties["thermal_expansivity"][:, phase_index] = alpha_phase
        except Exception:
            self._print_scalar_fallback_warning(
                f"thermal_expansivity aggregation failed for phase '{getattr(phase, 'name', phase_index)}'"
            )

        # Shear modulus: Voigt (linear) average of endmember shear moduli within a solution
        # For a homogeneous solid solution the crystal is uniform; linear mixing in G is appropriate.
        try:
            G_phase = (f * G_end).sum(-1)
            self.phase_properties["shear_modulus"][:, phase_index] = G_phase
        except Exception:
            self._print_scalar_fallback_warning(
                f"shear_modulus aggregation failed for phase '{getattr(phase, 'name', phase_index)}'"
            )

        # Isentropic bulk modulus per phase: K_S = K_T / (1 - α²·K_T·T·V/Cp)
        # Must be computed per-phase (not from assembled properties) to allow
        # correct Reuss/VRH averaging of K_S across phases.
        try:
            K_T_ph = self.phase_properties["isothermal_bulk_modulus_reuss"][:, phase_index]
            alpha_ph = self.phase_properties["thermal_expansivity"][:, phase_index]
            Cp_ph = self.phase_properties["molar_heat_capacity_p"][:, phase_index]
            V_ph = molar_volume_phase
            denom = 1.0 - (alpha_ph ** 2) * K_T_ph * T * V_ph / Cp_ph.clamp(min=1e-30)
            denom = denom.clamp(min=1e-10)
            K_S_phase = K_T_ph / denom
            self.phase_properties["isentropic_bulk_modulus_reuss"][:, phase_index] = K_S_phase
        except Exception:
            self._print_scalar_fallback_warning(
                f"isentropic_bulk_modulus_reuss aggregation failed for phase '{getattr(phase, 'name', phase_index)}'"
            )

    def set_component_properties(self, **properties):
        for name, value in properties.items():
            tensor = _as_tensor(
                value,
                device=self.component_abundances.device,
                dtype=self.component_abundances.dtype,
            )
            if tensor.shape[:2] != self.component_abundances.shape:
                raise ValueError(f"component property '{name}' must start with shape [B, C]")
            self.component_properties[name] = tensor

    def set_phase_properties(self, **properties):
        for name, value in properties.items():
            tensor = _as_tensor(
                value,
                device=self.component_abundances.device,
                dtype=self.component_abundances.dtype,
            )
            if tensor.shape[:2] != (self.batch_size, self.n_phases):
                raise ValueError(f"phase property '{name}' must start with shape [B, P]")
            self.phase_properties[name] = tensor

    def phase_abundances(self):
        return self.component_abundances @ self.phase_identity

    def _reduce_component_property(self, name):
        if name not in self.component_properties:
            raise AttributeError(f"component property '{name}' has not been set")
        values = self.component_properties[name]
        print(f"Reducing component property '{name}':")
        #print(values)
        #print(torch.sum(self.component_abundances * values, dim=-1))
        return torch.nansum(self.component_abundances * values, dim=-1)

    def _reduce_phase_property(self, name, VRH=False):
        if name not in self.phase_properties:
            raise AttributeError(f"phase property '{name}' has not been set")
        values = self.phase_properties[name]
        mole_fracs = _normalize_rows(self.phase_abundances())  # [B, P]

        # Volume fractions: phi_i = x_i * V_i / sum_j(x_j * V_j)
        # Fall back to mole fractions when reducing molar_volume itself (circular)
        if 'molar' not in name and 'entropy' not in name and 'heat_capacity' not in name:
            vol_unnorm = mole_fracs * self.phase_properties["molar_volume"]
            print(self.phase_properties["molar_volume"].size())
            weights = vol_unnorm / vol_unnorm.nansum(dim=-1, keepdim=True)# torch.clamp(vol_unnorm.sum(dim=-1, keepdim=True), min=1e-30)
            print(vol_unnorm.nansum(dim=-1).size())
            print('################### WEIGHTS ##################')
            print(weights)
        else:
                weights = mole_fracs

        voight = torch.nansum(weights * values, dim=-1)
        if not VRH:
            return voight
        else:
            reuss = 1 / (torch.nansum(weights * (1 / values), dim=-1))
            return (voight + reuss) / 2

    @property
    def molar_gibbs(self):
        if "molar_gibbs" in self.component_properties:
            return self._reduce_component_property("molar_gibbs")
        return self._reduce_phase_property("molar_gibbs")

    @property
    def molar_entropy(self):
        if "molar_entropy" in self.component_properties:
            return self._reduce_component_property("molar_entropy")
        return self._reduce_phase_property("molar_entropy")
    
    @property
    def entropy_by_mass(self):
        ME = self.molar_entropy
        MM = self.molar_mass
        return ME / MM

    @property
    def molar_enthalpy(self):
        if "molar_enthalpy" in self.component_properties:
            return self._reduce_component_property("molar_enthalpy")
        return self._reduce_phase_property("molar_enthalpy")

    @property
    def molar_internal_energy(self):
        if "molar_internal_energy" in self.component_properties:
            return self._reduce_component_property("molar_internal_energy")
        return self._reduce_phase_property("molar_internal_energy")

    @property
    def molar_volume(self):
        if "molar_volume" in self.component_properties:
            return self._reduce_component_property("molar_volume")
        return self._reduce_phase_property("molar_volume")

    @property
    def molar_mass(self):
        if "molar_mass" in self.component_properties:
            return self._reduce_component_property("molar_mass")
        return self._reduce_phase_property("molar_mass")

    @property
    def molar_heat_capacity_p(self):
        if "molar_heat_capacity_p" in self.component_properties:
            return self._reduce_component_property("molar_heat_capacity_p")
        return self._reduce_phase_property("molar_heat_capacity_p")

    @property
    def shear_modulus(self):
        #if "shear_modulus" in self.component_properties:
            #return self._reduce_component_property("shear_modulus")
        return self._reduce_phase_property("shear_modulus", VRH=True)

    @property
    def isothermal_bulk_modulus_reuss(self):
        #if "isothermal_bulk_modulus_reuss" in self.component_properties:
            #return self._reduce_component_property("isothermal_bulk_modulus_reuss")
        return self._reduce_phase_property("isothermal_bulk_modulus_reuss", VRH=True)
    
    @property
    def thermal_expansivity(self):
        if "thermal_expansivity" in self.component_properties:
            return self._reduce_component_property("thermal_expansivity")
        return self._reduce_phase_property("thermal_expansivity")

    
    @property
    def molar_heat_capacity_v(self):
        """Isochoric molar heat capacity [J/K/mol]. Derived via Cv = Cp - T*V*alpha^2*K_T."""
        C_p = self.molar_heat_capacity_p
        K_T = self.isothermal_bulk_modulus_reuss
        V   = self.molar_volume
        alpha = self.thermal_expansivity
        return C_p - self.temperature * V * alpha ** 2 * K_T

    @property
    def heat_capacity_p_by_mass(self):
        """Isobaric heat capacity per unit mass [J/kg/K]."""
        return self.molar_heat_capacity_p / self.molar_mass

    @property
    def heat_capacity_v_by_mass(self):
        """Isochoric heat capacity per unit mass [J/kg/K]."""
        return self.molar_heat_capacity_v / self.molar_mass

    @property
    def isentropic_bulk_modulus_reuss(self):
        """
        Returns adiabatic bulk modulus of the assemblage [Pa] via VRH of per-phase K_S.

        Per-phase K_S is computed in _evaluate_solution_with_vector to ensure the
        nonlinear K_T → K_S conversion happens before phase averaging (matching HeFESTo).
        """
        if "isentropic_bulk_modulus_reuss" in self.phase_properties:
            return self._reduce_phase_property("isentropic_bulk_modulus_reuss", VRH=True)
        # Fallback: derive from assembled K_T (less accurate for multi-phase assemblages)
        K_T = self.isothermal_bulk_modulus_reuss
        alpha = self.thermal_expansivity
        C_p = self.molar_heat_capacity_p
        V = self.molar_volume
        T = self.temperature
        return K_T / (1.0 - (alpha**2) * K_T * T * V / C_p)

    @property
    def density(self):
        if "density" in self.phase_properties:
            return self._reduce_phase_property("density")
        return self.molar_mass / self.molar_volume

    @property
    def p_wave_velocity(self):
        if INCLUDE_ANELASTIC_CONTRIBUTIONS:
            prefactor = (1-(1/(torch.pi*self.Qp)))
        else:
            prefactor = 1.0
        return prefactor * torch.sqrt(
            (
                self.isentropic_bulk_modulus_reuss
                + (4.0 / 3.0) * self.shear_modulus
            )
            / self.density
        )


    @property
    def bulk_sound_velocity(self):
        return torch.sqrt((self.p_wave_velocity ** 2) - ((4.0 / 3.0) * (self.s_wave_velocity ** 2)))

    @property
    def s_wave_velocity(self):
        if INCLUDE_ANELASTIC_CONTRIBUTIONS:
            prefactor = (1-(1/(torch.pi*self.Qs)))
        else:
            prefactor = 1.0
        return prefactor * torch.sqrt(self.shear_modulus / self.density)

    @property
    def available_component_properties(self):
        return tuple(sorted(self.component_properties.keys()))

    @property
    def available_phase_properties(self):
        return tuple(sorted(self.phase_properties.keys()))

    @property
    def available_endmember_properties(self):
        return tuple(sorted(self.endmember_properties.keys()))
