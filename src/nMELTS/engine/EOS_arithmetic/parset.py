"""Exact Python translation of HeFESTo ``parset.f``."""

from __future__ import annotations

from typing import Sequence

from .param_state import HeFESToParameterRecord
from .hefesto_thermal_data import (
    HCOK,
    HeFESToParsetResult,
    HeFESToThermalModeParameters,
    HeFESToThermalModeTensors,
)


def parset(
    parameter_records: Sequence[HeFESToParameterRecord],
) -> HeFESToThermalModeTensors:
    """
    Vectorized extraction of thermal parameters used by HeFESTo ``therm`` and ``Ctherm``.

    This function processes a sequence of parameter records and bundles them
    into a single `HeFESToThermalModeTensors` object, which contains all
    parameters as torch tensors for efficient, batched computation.
    """
    names = [record.species_label for record in parameter_records]
    results = [
        HeFESToParsetResult(
            fn=record.value("atoms_per_formula_unit"),
            zu=record.value("formula_units_per_cell"),
            wm=record.value("formula_mass_g_mol"),
            To=record.value("t0_k"),
            Fo=record.value("f0_kj_mol"),
            Vo=record.value("v0_cm3_mol"),
            Ko=record.value("k0_gpa"),
            Kop=record.value("k0_prime"),
            Kopp=record.value("k0_double_prime"),
            modes=HeFESToThermalModeParameters(
                wd1=record.value("theta0_k"),
                wd2=record.value("debye_acoustic_branch_2"),
                wd3=record.value("debye_acoustic_branch_3"),
                ws1=record.value("sin_acoustic_branch_1") * HCOK,
                ws2=record.value("sin_acoustic_branch_2") * HCOK,
                ws3=record.value("sin_acoustic_branch_3") * HCOK,
                we1=record.value("einstein_oscillator_1") * HCOK,
                qe1=record.value("einstein_weight_1"),
                we2=record.value("einstein_oscillator_2") * HCOK,
                qe2=record.value("einstein_weight_2"),
                we3=record.value("einstein_oscillator_3") * HCOK,
                qe3=record.value("einstein_weight_3"),
                we4=record.value("einstein_oscillator_4") * HCOK,
                qe4=record.value("einstein_weight_4"),
                wou=record.value("optic_continuum_upper") * HCOK,
                wol=record.value("optic_continuum_lower") * HCOK,
            ),
            gam=record.value("gamma_0"),
            qo=record.value("q_0"),
            be=record.value("beta"),
            ge=record.value("gamma_el_0"),
            q2A2=record.value("q2_a2"),
            htl=record.value("high_temperature_approximation"),
            ibv=record.integer_value("eos_type_flag"),
            ied=record.integer_value("debye_or_einstein_flag"),
            izp=record.integer_value("zero_point_pressure_flag"),
            Go=record.value("ambient_shear_modulus_gpa"),
            Gop=record.value("shear_pressure_derivative"),
            Got=record.value("shear_temperature_derivative"),
        )
        for record in parameter_records
    ]
    return HeFESToThermalModeTensors.from_parset_results(names, results)