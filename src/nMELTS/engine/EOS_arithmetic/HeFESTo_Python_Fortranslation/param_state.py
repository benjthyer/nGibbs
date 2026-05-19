from __future__ import annotations

from dataclasses import dataclass, field
from math import copysign, sqrt
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from ...config.constants import HEFESTO_ABBREVIATION_TO_SHORT_NAMES
# from . import hefesto_thermal_properties as htp  # TODO: Not used, causes import error


DEFAULT_PARAMETER_DIR = Path(__file__).resolve().parent / "HeFESTo_Parameters_010123"


PHYSUB_BULK_ATTRIBUTE_NAMES: Tuple[str, ...] = (
    "density",
    "K_Reuss",
    "K_Voigt",
    "K_Hill",
    "G_Reuss",
    "G_Voigt",
    "G_Hill",
    "Vb",
    "Vs",
    "Vp",
    "Vd",
    "Cp",
    "Cv",
    "alpha",
    "gamma",
    "entropy",
    "enthalpy",
    "Gibbs",
)

_PHYSUB_COMPONENT_ATTRIBUTE_NAMES: Tuple[str, ...] = (
    "molar_volume",
    "bulk_modulus",
    "shear_modulus",
    "heat_capacity_p",
    "heat_capacity_v",
    "thermal_expansivity",
    "entropy",
    "enthalpy",
    "gibbs",
)

_HEFESTO_SHORT_NAME_TO_ABBREVIATIONS: Dict[str, Tuple[str, ...]] = {}
for abbreviation, short_name in HEFESTO_ABBREVIATION_TO_SHORT_NAMES.items():
    key = short_name.lower()
    _HEFESTO_SHORT_NAME_TO_ABBREVIATIONS.setdefault(key, tuple())
    _HEFESTO_SHORT_NAME_TO_ABBREVIATIONS[key] = tuple(
        dict.fromkeys((*_HEFESTO_SHORT_NAME_TO_ABBREVIATIONS[key], abbreviation))
    )


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


def _load_hefesto_parameter_directory_uncached(directory: Path | str) -> Dict[str, HeFESToParameterRecord]:
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
