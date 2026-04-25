"""Python translation layer for HeFESTo ``physub``-style property arithmetic.

This module is intentionally self-contained. It focuses on the arithmetic that
comes after an equilibrium assemblage has already been determined: phase-level
aggregation, bulk Voigt-Reuss-Hill style combinations, and the simple
derivative-based corrections that depend on ``dn/dT`` and ``dn/dP``.

The implementation does not call back into the Fortran code. It expects the
caller to provide the species/phase thermodynamic state that the original
``physub`` subroutine would have read from COMMON blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import copysign, sqrt
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DEFAULT_PARAMETER_DIR = Path(__file__).resolve().parent / "HeFESTo_Parameters_010123"


PARAMETER_FIELD_NAMES: Tuple[str, ...] = (
    "atoms_per_formula_unit",
    "formula_units_per_cell",
    "formula_mass_g_mol",
    "t0_k",
    "f0_kj_mol",
    "v0_cm3_mol",
    "k0_gpa",
    "k0_prime",
    "k0_double_prime",
    "theta0_k",
    "debye_acoustic_branch_2",
    "debye_acoustic_branch_3",
    "sin_acoustic_branch_1",
    "sin_acoustic_branch_2",
    "sin_acoustic_branch_3",
    "einstein_oscillator_1",
    "einstein_weight_1",
    "einstein_oscillator_2",
    "einstein_weight_2",
    "einstein_oscillator_3",
    "einstein_weight_3",
    "einstein_oscillator_4",
    "einstein_weight_4",
    "optic_continuum_upper",
    "optic_continuum_lower",
    "gamma_0",
    "q_0",
    "beta",
    "gamma_el_0",
    "q2_a2",
    "high_temperature_approximation",
    "eos_type_flag",
    "debye_or_einstein_flag",
    "zero_point_pressure_flag",
    "ambient_shear_modulus_gpa",
    "shear_pressure_derivative",
    "shear_temperature_derivative",
    "critical_temperature_k",
    "critical_entropy_j_mol_k",
    "critical_volume_cm3_mol",
    "van_laar_size_parameter",
    "c12_prime",
    "c44_prime",
)


def asqrt(value: float) -> float:
    """Signed square root matching the Fortran helper."""

    return copysign(sqrt(abs(value)), value)


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) == 0.0:
        return default
    return numerator / denominator


@dataclass(frozen=True)
class HeFESToParameterRecord:
    """Parsed contents of one HeFESTo parameter file."""

    source_path: Path
    species_label: str
    phase_label: str
    values: Tuple[float, ...]
    raw_lines: Tuple[str, ...] = field(default_factory=tuple)

    def value(self, name: str) -> float:
        index = PARAMETER_FIELD_NAMES.index(name)
        return float(self.values[index])

    def integer_value(self, name: str) -> int:
        return int(round(self.value(name)))

    def compute_volume_bounds(self) -> Dict[str, float]:
        """Reproduce the small computed bounds block from ``parset.f``."""

        gamma_0 = self.value("gamma_0")
        q_0 = self.value("q_0")
        k0_prime = self.value("k0_prime")
        v0 = self.value("v0_cm3_mol")

        vsquaredminimum = 0.1
        vsmall = 1.0e-6
        vfactor = 10.0

        c = 1.0 - vsquaredminimum
        b = 6.0 * gamma_0
        a = 0.5 * gamma_0 * (36.0 * gamma_0 - 18.0 * q_0 - 12.0)

        vlow = 1.0e-15
        vupp = 1.0e15
        det = b * b - 4.0 * a * c
        if det >= 0.0:
            if a == 0.0:
                f1 = -c / b
                if f1 > 0.0:
                    vlow = v0 * (2.0 * f1 + 1.0) ** (-3.0 / 2.0)
                elif f1 < 0.0:
                    vupp = v0 * (2.0 * f1 + 1.0) ** (-3.0 / 2.0)
            else:
                f1 = (-b - sqrt(det)) / (2.0 * a)
                f2 = (-b + sqrt(det)) / (2.0 * a)
                if max(f1, f2) < 0.0:
                    vupp = v0 * (2.0 * max(f1, f2) + 1.0) ** (-3.0 / 2.0)
                elif min(f1, f2) > 0.0:
                    vlow = v0 * (2.0 * min(f1, f2) + 1.0) ** (-3.0 / 2.0)
                else:
                    vupp = v0 * (2.0 * min(f1, f2) + 1.0) ** (-3.0 / 2.0)
                    vlow = v0 * (2.0 * max(f1, f2) + 1.0) ** (-3.0 / 2.0)

        if a != 0.0:
            fextremum = -b / (2.0 * a)
            vextremum = v0 * (2.0 * fextremum + 1.0) ** (-3.0 / 2.0)
            if fextremum > 0.0:
                vlow = max(vlow, vextremum)
            elif fextremum < 0.0:
                vupp = min(vupp, vextremum)

        vibrational_lower = max(vlow, v0 / vfactor) + vsmall
        vibrational_upper = min(vupp, v0 * vfactor) - vsmall

        c = 1.0
        bsp = 3.0 * k0_prime - 5.0
        asp = 27.0 / 2.0 * (k0_prime - 4.0)
        detsp = bsp * bsp - 4.0 * asp * c
        vsplow = 1.0e-15
        vspupp = 1.0e15

        if detsp >= 0.0:
            if asp == 0.0:
                f1 = -c / bsp
                vspupp = v0 * (2.0 * f1 + 1.0) ** (-3.0 / 2.0)
            else:
                f1 = (-bsp - sqrt(detsp)) / (2.0 * asp)
                f2 = (-bsp + sqrt(detsp)) / (2.0 * asp)
                if max(f1, f2) < 0.0:
                    vspupp = v0 * (2.0 * max(f1, f2) + 1.0) ** (-3.0 / 2.0)
                elif min(f1, f2) > 0.0:
                    vsplow = v0 * (2.0 * min(f1, f2) + 1.0) ** (-3.0 / 2.0)
                else:
                    vspupp = v0 * (2.0 * min(f1, f2) + 1.0) ** (-3.0 / 2.0)
                    vsplow = v0 * (2.0 * max(f1, f2) + 1.0) ** (-3.0 / 2.0)

        spinodal_lower = max(vsplow, v0 / vfactor) + vsmall
        spinodal_upper = min(vspupp, v0 * vfactor) - vsmall

        return {
            "vibrational_lower": vibrational_lower,
            "vibrational_upper": vibrational_upper,
            "spinodal_lower": spinodal_lower,
            "spinodal_upper": spinodal_upper,
        }


def parse_hefesto_parameter_file(path: Path | str) -> HeFESToParameterRecord:
    """Parse one HeFESTo parameter file using the fixed record order."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing HeFESTo parameter file: {path}")

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        raise ValueError(f"Parameter file is empty: {path}")

    header_tokens = lines[0].split()
    if not header_tokens:
        raise ValueError(f"Parameter file header is empty: {path}")

    species_label = header_tokens[0]
    phase_label = " ".join(header_tokens[1:]) if len(header_tokens) > 1 else ""

    numeric_values: List[float] = []
    raw_lines: List[str] = []
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            numeric_values.append(float(stripped.split()[0]))
            raw_lines.append(line)
        except ValueError:
            continue

    if len(numeric_values) < len(PARAMETER_FIELD_NAMES):
        raise ValueError(
            f"Parameter file {path} has {len(numeric_values)} numeric records, "
            f"expected at least {len(PARAMETER_FIELD_NAMES)}"
        )

    return HeFESToParameterRecord(
        source_path=path,
        species_label=species_label,
        phase_label=phase_label,
        values=tuple(numeric_values[: len(PARAMETER_FIELD_NAMES)]),
        raw_lines=tuple(raw_lines),
    )


def load_hefesto_parameter_directory(directory: Path | str = DEFAULT_PARAMETER_DIR) -> Dict[str, HeFESToParameterRecord]:
    """Load all parameter files in the HeFESTo parameter directory."""

    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing HeFESTo parameter directory: {directory}")

    records: Dict[str, HeFESToParameterRecord] = {}
    for entry in sorted(directory.iterdir()):
        if not entry.is_file():
            continue
        if entry.name.startswith("README") or entry.name.startswith("."):
            continue
        try:
            records[entry.name] = parse_hefesto_parameter_file(entry)
        except ValueError:
            continue
    return records


@dataclass(frozen=True)
class HeFESToSpeciesState:
    """Species-level thermodynamic state supplied to the Python translation."""

    name: str
    phase_name: str
    amount: float
    molar_mass: float
    molar_volume: float
    density: float
    bulk_modulus_t: float
    bulk_modulus_s: float
    shear_modulus: float
    heat_capacity_p: float
    heat_capacity_v: float
    thermal_expansivity: float
    entropy: float
    enthalpy: float
    gibbs: float
    delta_s: float = 0.0
    delta_gdt: float = 0.0
    vdeb: float = 0.0
    gamdeb: float = 0.0
    pressure_derivative_gibbs: float = 0.0
    temperature_derivative_entropy: float = 0.0
    dndt: float = 0.0
    dndp: float = 0.0
    is_absent: bool = False


@dataclass(frozen=True)
class HeFESToPhaseState:
    """A phase and its constituent species after equilibrium has been solved."""

    name: str
    species: Tuple[HeFESToSpeciesState, ...]


@dataclass(frozen=True)
class HeFESToPhaseProperties:
    name: str
    n_moles: float
    mass: float
    volume: float
    density: float
    bulk_modulus_t: float
    bulk_modulus_s: float
    shear_modulus: float
    bulk_sound_velocity: float
    shear_velocity: float
    pressure_velocity: float
    heat_capacity_p: float
    heat_capacity_v: float
    thermal_expansivity: float
    entropy: float
    enthalpy: float
    gibbs: float
    dlnvb_dt: float = 0.0
    dgdtdt: float = 0.0
    dndt_fast: float = 0.0
    dndp_fast: float = 0.0


@dataclass(frozen=True)
class HeFESToBulkProperties:
    pressure_gpa: float
    depth_km: float
    temperature_k: float
    density: float
    volume: float
    mass: float
    bulk_modulus_reuss: float
    bulk_modulus_voigt: float
    bulk_modulus_hill: float
    shear_modulus_reuss: float
    shear_modulus_voigt: float
    shear_modulus_hill: float
    bulk_sound_velocity: float
    shear_velocity: float
    pressure_velocity: float
    debye_velocity: float
    debye_velocity_signed: float
    heat_capacity_p: float
    heat_capacity_v: float
    thermal_expansivity: float
    gruneisen_parameter: float
    entropy: float
    enthalpy: float
    gibbs: float
    phases: Tuple[HeFESToPhaseProperties, ...] = field(default_factory=tuple)


def _combine_reuss(volume: float, weighted_sum: float) -> float:
    if weighted_sum == 0.0:
        return 0.0
    return volume / weighted_sum


def _combine_hill(voigt_value: float, reuss_value: float) -> float:
    return 0.5 * (voigt_value + reuss_value)


def _phase_wave_speeds(density: float, bulk_modulus: float, shear_modulus: float) -> Tuple[float, float, float]:
    bulk_velocity = asqrt(_safe_divide(bulk_modulus, density))
    shear_velocity = asqrt(_safe_divide(shear_modulus, density))
    pressure_velocity = asqrt(_safe_divide(bulk_modulus + 4.0 / 3.0 * shear_modulus, density))
    return bulk_velocity, shear_velocity, pressure_velocity


def _compute_phase_properties(phase: HeFESToPhaseState, temperature_k: float) -> HeFESToPhaseProperties:
    phase_volume = 0.0
    phase_mass = 0.0
    phase_cp = 0.0
    phase_cv = 0.0
    phase_alpha = 0.0
    phase_delta = 0.0
    phase_shear_reuss = 0.0
    phase_bulk_reuss = 0.0
    phase_bulk_voigt = 0.0
    phase_shear_voigt = 0.0
    phase_entropy = 0.0
    phase_enthalpy = 0.0
    phase_gibbs = 0.0
    phase_dgdt = 0.0
    phase_n = 0.0

    for species in phase.species:
        if species.is_absent:
            continue

        amount = species.amount
        phase_n += amount
        phase_volume += amount * species.molar_volume
        phase_mass += amount * species.molar_mass
        phase_cp += amount * species.heat_capacity_p
        phase_cv += amount * species.heat_capacity_v
        phase_alpha += amount * species.molar_volume * species.thermal_expansivity
        phase_delta += amount * species.molar_volume * species.thermal_expansivity * (1.0 + species.delta_s) / species.bulk_modulus_s
        phase_shear_reuss += amount * species.molar_volume / species.shear_modulus if species.shear_modulus != 0.0 else 0.0
        phase_bulk_reuss += amount * species.molar_volume / species.bulk_modulus_t if species.bulk_modulus_t != 0.0 else 0.0
        phase_bulk_voigt += amount * species.molar_volume * species.bulk_modulus_t
        phase_shear_voigt += amount * species.molar_volume * species.shear_modulus
        phase_entropy += amount * species.entropy
        phase_enthalpy += amount * species.enthalpy
        phase_gibbs += amount * species.gibbs
        phase_dgdt += amount * species.molar_volume / (species.shear_modulus ** 2 if species.shear_modulus != 0.0 else 1.0) * species.delta_gdt

    if phase_volume == 0.0 or phase_mass == 0.0 or phase_n == 0.0:
        return HeFESToPhaseProperties(
            name=phase.name,
            n_moles=phase_n,
            mass=phase_mass,
            volume=phase_volume,
            density=0.0,
            bulk_modulus_t=0.0,
            bulk_modulus_s=0.0,
            shear_modulus=0.0,
            bulk_sound_velocity=0.0,
            shear_velocity=0.0,
            pressure_velocity=0.0,
            heat_capacity_p=0.0,
            heat_capacity_v=0.0,
            thermal_expansivity=0.0,
            entropy=0.0,
            enthalpy=0.0,
            gibbs=0.0,
        )

    phase_density = phase_mass / phase_volume
    phase_alpha_total = phase_alpha / phase_volume
    bulk_reuss = _combine_reuss(phase_volume, phase_bulk_reuss)
    shear_reuss = _combine_reuss(phase_volume, phase_shear_reuss)
    bulk_voigt = phase_bulk_voigt / phase_volume if phase_volume != 0.0 else 0.0
    shear_voigt = phase_shear_voigt / phase_volume if phase_volume != 0.0 else 0.0

    phase_delta_total = phase_delta * bulk_reuss / (phase_volume * phase_alpha) - 1.0 if phase_alpha != 0.0 else 0.0

    bulk_hill = _combine_hill(bulk_voigt, bulk_reuss)
    shear_hill = _combine_hill(shear_voigt, shear_reuss)

    phase_cp_total = phase_cp / phase_mass
    phase_cv_total = phase_cv / phase_mass

    bulk_sound_velocity, shear_velocity, pressure_velocity = _phase_wave_speeds(phase_density, bulk_hill, shear_hill)

    return HeFESToPhaseProperties(
        name=phase.name,
        n_moles=phase_n,
        mass=phase_mass,
        volume=phase_volume,
        density=phase_density,
        bulk_modulus_t=bulk_hill,
        bulk_modulus_s=bulk_reuss,
        shear_modulus=shear_hill,
        bulk_sound_velocity=bulk_sound_velocity,
        shear_velocity=shear_velocity,
        pressure_velocity=pressure_velocity,
        heat_capacity_p=phase_cp_total,
        heat_capacity_v=phase_cv_total,
        thermal_expansivity=phase_alpha_total,
        entropy=phase_entropy / phase_n,
        enthalpy=phase_enthalpy / phase_n,
        gibbs=phase_gibbs / phase_n,
        dlnvb_dt=0.5 * phase_alpha_total * (phase_delta_total - 1.0),
        dgdtdt=phase_dgdt * (shear_hill ** 2),
        dndt_fast=0.0,
        dndp_fast=0.0,
    )


def compute_physub_properties(
    phases: Sequence[HeFESToPhaseState],
    pressure_gpa: float,
    temperature_k: float,
    depth_km: float = 0.0,
) -> HeFESToBulkProperties:
    """Compute phase and bulk properties from an equilibrium assemblage."""

    phase_outputs: List[HeFESToPhaseProperties] = []
    total_volume = 0.0
    total_mass = 0.0
    total_entropy = 0.0
    total_enthalpy = 0.0
    total_gibbs = 0.0

    baggv = 0.0
    baggr = 0.0
    gaggv = 0.0
    gaggr = 0.0
    alpagg = 0.0
    cvagg = 0.0
    cpagg = 0.0

    for phase in phases:
        phase_output = _compute_phase_properties(phase, temperature_k)
        phase_outputs.append(phase_output)

        total_volume += phase_output.volume
        total_mass += phase_output.mass
        total_entropy += phase_output.entropy * phase_output.n_moles
        total_enthalpy += phase_output.enthalpy * phase_output.n_moles
        total_gibbs += phase_output.gibbs * phase_output.n_moles

        if phase_output.volume != 0.0:
            baggv += phase_output.volume * phase_output.bulk_modulus_t
            baggr += _safe_divide(phase_output.volume, phase_output.bulk_modulus_t)
            gaggv += phase_output.volume * phase_output.shear_modulus
            gaggr += _safe_divide(phase_output.volume, phase_output.shear_modulus)

        alpagg += phase_output.thermal_expansivity * phase_output.volume
        cpagg += phase_output.heat_capacity_p * phase_output.mass
        cvagg += phase_output.heat_capacity_v * phase_output.mass

    if total_volume == 0.0 or total_mass == 0.0:
        return HeFESToBulkProperties(
            pressure_gpa=pressure_gpa,
            depth_km=depth_km,
            temperature_k=temperature_k,
            density=0.0,
            volume=total_volume,
            mass=total_mass,
            bulk_modulus_reuss=0.0,
            bulk_modulus_voigt=0.0,
            bulk_modulus_hill=0.0,
            shear_modulus_reuss=0.0,
            shear_modulus_voigt=0.0,
            shear_modulus_hill=0.0,
            bulk_sound_velocity=0.0,
            shear_velocity=0.0,
            pressure_velocity=0.0,
            debye_velocity=0.0,
            debye_velocity_signed=0.0,
            heat_capacity_p=0.0,
            heat_capacity_v=0.0,
            thermal_expansivity=0.0,
            gruneisen_parameter=0.0,
            entropy=0.0,
            enthalpy=0.0,
            gibbs=0.0,
            phases=tuple(phase_outputs),
        )

    density = total_mass / total_volume

    bulk_modulus_voigt = baggv / total_volume if total_volume != 0.0 else 0.0
    bulk_modulus_reuss = _combine_reuss(total_volume, baggr)
    shear_modulus_voigt = gaggv / total_volume if total_volume != 0.0 else 0.0
    shear_modulus_reuss = _combine_reuss(total_volume, gaggr)
    bulk_modulus_hill = _combine_hill(bulk_modulus_voigt, bulk_modulus_reuss)
    shear_modulus_hill = _combine_hill(shear_modulus_voigt, shear_modulus_reuss)

    heat_capacity_p = _safe_divide(cpagg, total_mass)
    heat_capacity_v = _safe_divide(cvagg, total_mass)
    thermal_expansivity = _safe_divide(alpagg, total_volume)
    gruneisen_parameter = _safe_divide(1000.0 * bulk_modulus_hill * thermal_expansivity, density * heat_capacity_p)

    bulk_sound_velocity, shear_velocity, pressure_velocity = _phase_wave_speeds(density, bulk_modulus_hill, shear_modulus_hill)
    debye_velocity_cubed = 2.0 / 3.0 / (shear_velocity ** 3 if shear_velocity != 0.0 else 1.0) + 1.0 / 3.0 / (pressure_velocity ** 3 if pressure_velocity != 0.0 else 1.0)
    debye_velocity = 1.0 / (debye_velocity_cubed ** (1.0 / 3.0)) if debye_velocity_cubed > 0.0 else 0.0
    debye_velocity_signed = 1.0 / (abs(debye_velocity_cubed) ** (1.0 / 3.0)) if debye_velocity_cubed != 0.0 else 0.0
    if debye_velocity_cubed < 0.0:
        debye_velocity_signed = -debye_velocity_signed

    return HeFESToBulkProperties(
        pressure_gpa=pressure_gpa,
        depth_km=depth_km,
        temperature_k=temperature_k,
        density=density,
        volume=total_volume,
        mass=total_mass,
        bulk_modulus_reuss=bulk_modulus_reuss,
        bulk_modulus_voigt=bulk_modulus_voigt,
        bulk_modulus_hill=bulk_modulus_hill,
        shear_modulus_reuss=shear_modulus_reuss,
        shear_modulus_voigt=shear_modulus_voigt,
        shear_modulus_hill=shear_modulus_hill,
        bulk_sound_velocity=bulk_sound_velocity,
        shear_velocity=shear_velocity,
        pressure_velocity=pressure_velocity,
        debye_velocity=debye_velocity,
        debye_velocity_signed=debye_velocity_signed,
        heat_capacity_p=heat_capacity_p,
        heat_capacity_v=heat_capacity_v,
        thermal_expansivity=thermal_expansivity,
        gruneisen_parameter=gruneisen_parameter,
        entropy=total_entropy,
        enthalpy=total_enthalpy,
        gibbs=total_gibbs,
        phases=tuple(phase_outputs),
    )
