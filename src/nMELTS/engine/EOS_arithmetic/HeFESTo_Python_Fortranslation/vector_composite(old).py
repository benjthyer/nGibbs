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

        return (
            (1.0 / 3.0)
            * torch.pow(1.0 + 2.0 * f, 2.5)
            * (9.0 * params["K_0"] * f + 0.5 * 27.0 * params["K_0"] * (params["Kprime_0"] - 4.0) * f * f)
            + gr * (E_th - E_th_ref) / volume
        )

    def volume(self, pressure, temperature, params, max_iter=12, tol=1.0e-10):
        """
        Batched Newton-Raphson solve for V(P, T) using dP/dV = -K_T/V.
        """
        V = torch.full_like(pressure, params["V_0"])
        converged = torch.zeros_like(pressure, dtype=torch.bool)

        for _ in range(max_iter):
            P_pred = self.pressure(temperature, V, params)
            residual = P_pred - pressure

            with torch.no_grad():
                converged_now = torch.abs(residual) <= tol * torch.clamp(torch.abs(pressure), min=1.0)
                converged = converged | converged_now

            if torch.all(converged):
                break

            K_T = self.isothermal_bulk_modulus_reuss(
                torch.zeros_like(pressure), temperature, V, params
            )
            dP_dV = -torch.clamp(K_T / torch.clamp(V, min=1.0e-16), min=-1.0e30, max=-1.0e-16)
            delta_V = residual / dP_dV

            V_new = V - delta_V
            # Keep volumes in a physically plausible range around V0.
            V = torch.clamp(V_new, min=0.4 * params["V_0"], max=1.6 * params["V_0"])

        return V, converged

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
        component_names,
        phase_names,
        phase_identity,
        component_abundances,
        phase_models=None,
        pressure=None,
        temperature=None,
        device=None,
        dtype=None,
    ):
        _require_torch()

        self.component_names = list(component_names)
        self.phase_names = list(phase_names)
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

        if phase_models is not None:
            self.set_phase_models(phase_models)

        if pressure is not None or temperature is not None:
            self.set_state(pressure, temperature)

    @property
    def batch_size(self):
        return self.component_abundances.shape[0]

    @property
    def n_components(self):
        return len(self.component_names)

    @property
    def n_phases(self):
        return len(self.phase_names)

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

        self.phase_models = list(phase_models)
        self.endmembers_per_phase = []
        self.endmember_slices = []

        start = 0
        for phase in self.phase_models:
            if isinstance(phase, Solution):
                count = phase.n_endmembers
            elif isinstance(phase, Mineral):
                count = 1
            else:
                raise TypeError(
                    "phase_models must contain only burnman.Mineral or burnman.Solution objects"
                )

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

        for phase_index, phase in enumerate(self.phase_models):
            # Check if this is a Mineral with SLB3 EOS; if so, use vectorized kernel
            if isinstance(phase, Mineral) and hasattr(phase, "method"):
                method_name = phase.method.__class__.__name__
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
                # Fallback to scalar evaluation for Solutions and other types
                self._print_scalar_fallback_warning(
                    f"phase '{getattr(phase, 'name', phase_index)}' type '{type(phase).__name__}' is not vectorized"
                )
                self._evaluate_phase_scalar(
                    phase, phase_index, batch_size, device, dtype, 
                    phase_property_names, endmember_property_names
                )

        return self.phase_properties, self.endmember_properties

    def _evaluate_phase_with_vector_slb3(
        self, phase, phase_index, batch_size, device, dtype, phase_property_names
    ):
        """
        Use VectorSLB3 EOS to compute all batch samples at once for a Mineral phase.
        """
        # Create vectorized EOS kernel
        vector_eos = VectorSLB3()
        
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
        return torch.sum(self.component_abundances * values, dim=-1)

    def _reduce_phase_property(self, name):
        if name not in self.phase_properties:
            raise AttributeError(f"phase property '{name}' has not been set")
        values = self.phase_properties[name]
        weights = _normalize_rows(self.phase_abundances())
        return torch.sum(weights * values, dim=-1)

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
        if "shear_modulus" in self.component_properties:
            return self._reduce_component_property("shear_modulus")
        return self._reduce_phase_property("shear_modulus")

    @property
    def isothermal_bulk_modulus_reuss(self):
        if "isothermal_bulk_modulus_reuss" in self.component_properties:
            return self._reduce_component_property("isothermal_bulk_modulus_reuss")
        return self._reduce_phase_property("isothermal_bulk_modulus_reuss")

    @property
    def isentropic_bulk_modulus_reuss(self):
        if "isentropic_bulk_modulus_reuss" in self.component_properties:
            return self._reduce_component_property("isentropic_bulk_modulus_reuss")
        if "isentropic_bulk_modulus_reuss" in self.phase_properties:
            return self._reduce_phase_property("isentropic_bulk_modulus_reuss")

        self._print_scalar_fallback_warning(
            "isentropic_bulk_modulus_reuss not available; using isothermal_bulk_modulus_reuss as approximation"
        )
        return self.isothermal_bulk_modulus_reuss

    @property
    def density(self):
        if "density" in self.phase_properties:
            return self._reduce_phase_property("density")
        return self.molar_mass / self.molar_volume

    @property
    def p_wave_velocity(self):
        return torch.sqrt(
            (
                self.isentropic_bulk_modulus_reuss
                + (4.0 / 3.0) * self.shear_modulus
            )
            / self.density
        )

    @property
    def bulk_sound_velocity(self):
        return torch.sqrt(self.isentropic_bulk_modulus_reuss / self.density)

    @property
    def shear_wave_velocity(self):
        return torch.sqrt(self.shear_modulus / self.density)

    @property
    def available_component_properties(self):
        return tuple(sorted(self.component_properties.keys()))

    @property
    def available_phase_properties(self):
        return tuple(sorted(self.phase_properties.keys()))

    @property
    def available_endmember_properties(self):
        return tuple(sorted(self.endmember_properties.keys()))
