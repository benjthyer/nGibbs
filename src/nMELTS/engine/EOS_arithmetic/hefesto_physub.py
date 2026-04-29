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
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from ...config.constants import HEFESTO_ABBREVIATION_TO_SHORT_NAMES
from . import hefesto_thermal_properties as htp


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


@dataclass
class HeFESToPhysubContext:
    """Singleton-style context for cached HeFESTo parameter and projection tensors."""

    parameter_dir: Path = DEFAULT_PARAMETER_DIR
    parameter_records: Dict[str, HeFESToParameterRecord] = field(init=False)
    component_names: Tuple[str, ...] = field(init=False)
    component_index: Dict[str, int] = field(init=False)
    formula_mass_g_mol: torch.Tensor = field(init=False)
    projection_component_to_mass: torch.Tensor = field(init=False)
    projection_component_to_attributes: torch.Tensor = field(init=False)
    parse_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        records = _load_hefesto_parameter_directory_uncached(self.parameter_dir)
        self.parse_count = len(records)
        self.parameter_records = records
        self.component_names = tuple(sorted(records.keys()))
        self.component_index = {name: i for i, name in enumerate(self.component_names)}

        if len(self.component_names) == 0:
            self.formula_mass_g_mol = torch.empty((0,), dtype=torch.float32)
            self.projection_component_to_mass = torch.empty((0, 1), dtype=torch.float32)
            self.projection_component_to_attributes = torch.empty(
                (0, len(_PHYSUB_COMPONENT_ATTRIBUTE_NAMES)), dtype=torch.float32
            )
            return

        masses = [records[name].value("formula_mass_g_mol") for name in self.component_names]
        
        # Note: thermal properties (Cp, Cv, alpha, entropy) are temperature-dependent.
        # Initialize placeholder values; they will be computed at runtime via
        # compute_component_attributes_at_temperature().
        attrs = []
        for name in self.component_names:
            rec = records[name]
            attrs.append(
                [
                    rec.value("v0_cm3_mol"),
                    rec.value("k0_gpa"),
                    rec.value("ambient_shear_modulus_gpa"),
                    0.0,  # heat_capacity_p (computed at runtime)
                    0.0,  # heat_capacity_v (computed at runtime)
                    0.0,  # thermal_expansivity (computed at runtime)
                    0.0,  # entropy (computed at runtime)
                    rec.value("f0_kj_mol"),
                    rec.value("f0_kj_mol"),
                ]
            )

        self.formula_mass_g_mol = torch.tensor(masses, dtype=torch.float32)
        self.projection_component_to_mass = self.formula_mass_g_mol.unsqueeze(-1)
        self.projection_component_to_attributes = torch.tensor(attrs, dtype=torch.float32)

    def compute_component_attributes_at_temperature(
        self,
        temperature_k: float,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, torch.Tensor]:
        """Compute temperature-dependent component attributes using thermal models.
        
        This creates per-component tensors for Cp, Cv, alpha, and entropy based on
        the given temperature and HeFESTo parameters.
        
        Parameters
        ----------
        temperature_k : float
            Temperature in Kelvin
        batch_size : int, optional
            Batch dimension size. Attributes will be expanded to (batch_size, num_components).
        device : torch.device
            Device to place output tensors on
            
        Returns
        -------
        dict
            Mapping from attribute name to tensor with shape (batch_size, C) where C is the
            number of components. Includes: heat_capacity_p, heat_capacity_v,
            thermal_expansivity, entropy, molar_volume, bulk_modulus, shear_modulus,
            enthalpy, gibbs.
        """
        if temperature_k <= 0.0:
            raise ValueError(f"Temperature must be positive, got {temperature_k} K")

        num_components = len(self.component_names)
        attrs: Dict[str, torch.Tensor] = {}

        # Initialize with reference (static) properties
        v0 = []
        k0 = []
        g0 = []
        f0 = []
        cp_vals = []
        cv_vals = []
        alpha_vals = []
        s_vals = []

        for name in self.component_names:
            rec = self.parameter_records[name]

            v0.append(rec.value("v0_cm3_mol"))
            k0.append(rec.value("k0_gpa"))
            g0.append(rec.value("ambient_shear_modulus_gpa"))
            f0.append(rec.value("f0_kj_mol"))

            # Compute thermal properties at this temperature
            fn = rec.value("atoms_per_formula_unit")
            theta_d = rec.value("theta0_k")
            
            # Collect Einstein oscillator data
            einstein_temps = []
            einstein_weights = []
            for i in range(1, 5):  # Modes 1-4
                we_key = f"einstein_oscillator_{i}"
                qe_key = f"einstein_weight_{i}"
                try:
                    we_val = rec.value(we_key) * htp.HCOK if rec.value(we_key) else 0.0
                    qe_val = rec.value(qe_key) if rec.value(qe_key) else 0.0
                    if we_val > 0.0 and qe_val > 0.0:
                        einstein_temps.append(we_val)
                        einstein_weights.append(qe_val)
                except (KeyError, TypeError):
                    pass

            # Compute Cp with Ctherm-like branch composition.
            cp = htp.compute_component_heat_capacity_p(
                temperature=temperature_k,
                atoms_per_formula=fn,
                debye_temp=theta_d,
                formula_units_per_cell=rec.value("formula_units_per_cell"),
                debye_temps_2_3=(
                    rec.value("debye_acoustic_branch_2"),
                    rec.value("debye_acoustic_branch_3"),
                ),
                sin_temps=(
                    rec.value("sin_acoustic_branch_1") * htp.HCOK,
                    rec.value("sin_acoustic_branch_2") * htp.HCOK,
                    rec.value("sin_acoustic_branch_3") * htp.HCOK,
                ),
                optic_continuum_upper=rec.value("optic_continuum_upper") * htp.HCOK,
                optic_continuum_lower=rec.value("optic_continuum_lower") * htp.HCOK,
                einstein_temps=tuple(einstein_temps),
                einstein_weights=tuple(einstein_weights),
            )
            formula_mass = max(rec.value("formula_mass_g_mol"), 1.0e-12)
            cp_vals.append(cp / (formula_mass * 1000.0))

            # Compute Cv using thermodynamic relation
            cv = htp.compute_component_heat_capacity_v(
                temperature=temperature_k,
                cp=cp,
                thermal_expansivity=0.0,  # Placeholder: will update after computing alpha
                volume=v0[-1],
                bulk_modulus=k0[-1],
            )
            cv_vals.append(cv / (formula_mass * 1000.0))

            # Compute thermal expansivity
            gamma = rec.value("gamma_0")  # Grüneisen parameter
            alpha = htp.compute_component_thermal_expansivity(
                temperature=temperature_k,
                gruneisen_parameter=gamma,
                heat_capacity_v=cv,
                volume=v0[-1],
                bulk_modulus=k0[-1],
            )
            alpha_vals.append(alpha)

            # Recompute Cv with actual alpha
            cv_corrected = htp.compute_component_heat_capacity_v(
                temperature=temperature_k,
                cp=cp,
                thermal_expansivity=alpha,
                volume=v0[-1],
                bulk_modulus=k0[-1],
            )
            cv_vals[-1] = cv_corrected / (formula_mass * 1000.0)

            # Compute entropy
            s = htp.compute_component_entropy(
                temperature=temperature_k,
                atoms_per_formula=fn,
                debye_temp=theta_d,
                formula_units_per_cell=rec.value("formula_units_per_cell"),
                debye_temps_2_3=(
                    rec.value("debye_acoustic_branch_2"),
                    rec.value("debye_acoustic_branch_3"),
                ),
                sin_temps=(
                    rec.value("sin_acoustic_branch_1") * htp.HCOK,
                    rec.value("sin_acoustic_branch_2") * htp.HCOK,
                    rec.value("sin_acoustic_branch_3") * htp.HCOK,
                ),
                optic_continuum_upper=rec.value("optic_continuum_upper") * htp.HCOK,
                optic_continuum_lower=rec.value("optic_continuum_lower") * htp.HCOK,
                einstein_temps=tuple(einstein_temps),
                einstein_weights=tuple(einstein_weights),
                reference_entropy_j_mol_k=rec.value("critical_entropy_j_mol_k"),
            )
            s_vals.append(s)

        # Convert lists to tensors and expand to batch dimension
        attrs["molar_volume"] = torch.tensor(v0, dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1)
        attrs["bulk_modulus"] = torch.tensor(k0, dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1)
        attrs["shear_modulus"] = torch.tensor(g0, dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1)
        attrs["heat_capacity_p"] = torch.tensor(cp_vals, dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1) * 1000.0
        attrs["heat_capacity_v"] = torch.tensor(cv_vals, dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1) * 1000.0
        attrs["thermal_expansivity"] = torch.tensor(alpha_vals, dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1)
        attrs["entropy"] = torch.tensor(s_vals, dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1)
        attrs["enthalpy"] = torch.tensor(f0, dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1)
        attrs["gibbs"] = torch.tensor(f0, dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1)

        return attrs

    def align_component_tensor(
        self,
        component_moles: torch.Tensor,
        component_names: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        """Align arbitrary component ordering to canonical context ordering."""

        if component_moles.ndim != 2:
            raise ValueError(f"component_moles must be 2D (B, C), got shape {tuple(component_moles.shape)}")

        if component_names is None:
            if component_moles.shape[1] != len(self.component_names):
                raise ValueError(
                    "component_names must be provided when C does not match context component count"
                )
            return component_moles

        if len(component_names) != component_moles.shape[1]:
            raise ValueError("component_names length must match component_moles.shape[1]")

        aligned = torch.zeros(
            (component_moles.shape[0], len(self.component_names)),
            dtype=component_moles.dtype,
            device=component_moles.device,
        )
        for in_col, name in enumerate(component_names):
            component_key = name if name in self.component_index else None
            if component_key is None:
                aliases = _HEFESTO_SHORT_NAME_TO_ABBREVIATIONS.get(str(name).lower(), ())
                for alias in aliases:
                    if alias in self.component_index:
                        component_key = alias
                        break
            if component_key is None:
                raise KeyError(
                    f"Unknown HeFESTo component name '{name}'. "
                    "Expected a known shorthand or a full mineral name present in HEFESTO_ABBREVIATION_TO_SHORT_NAMES."
                )
            aligned[:, self.component_index[component_key]] = component_moles[:, in_col]
        return aligned


_HEFESTO_PHYSUB_CONTEXT: HeFESToPhysubContext = HeFESToPhysubContext()


def get_hefesto_physub_context() -> HeFESToPhysubContext:
    """Return module-level singleton context initialized at import time."""

    return _HEFESTO_PHYSUB_CONTEXT


def load_hefesto_parameter_directory(
    directory: Path | str = DEFAULT_PARAMETER_DIR,
    use_cache: bool = True,
) -> Dict[str, HeFESToParameterRecord]:
    """Load HeFESTo parameter files, defaulting to singleton cache."""

    resolved_dir = Path(directory).resolve()
    if use_cache and resolved_dir == _HEFESTO_PHYSUB_CONTEXT.parameter_dir.resolve():
        return dict(_HEFESTO_PHYSUB_CONTEXT.parameter_records)
    return _load_hefesto_parameter_directory_uncached(resolved_dir)


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


@dataclass(frozen=True)
class HeFESToBenchmarkComparisonResult:
    passed: bool
    n_rows: int
    mean_errors: Dict[str, float]
    max_errors: Dict[str, float]


def _read_benchmark_table(path: Path) -> Tuple[List[str], List[Dict[str, float]]]:
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
    header = None
    rows: List[Dict[str, float]] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if header is None:
            header = tokens
            continue

        if len(tokens) < len(header):
            continue

        try:
            values = [float(token) for token in tokens[: len(header)]]
        except ValueError:
            continue

        rows.append({header[i]: values[i] for i in range(len(header))})

    if header is None:
        raise ValueError(f"No header found in table: {path}")

    return header, rows


def _parse_control_species_map(path: Path) -> Tuple[List[str], Dict[str, str], Dict[str, List[str]]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    species_order: List[str] = []
    species_to_phase: Dict[str, str] = {}
    phase_to_species: Dict[str, List[str]] = {}

    current_phase = ""
    skip_next = False
    for line in lines:
        if line.lower().startswith("phase "):
            current_phase = line.split()[1]
            phase_to_species[current_phase] = []
            skip_next = True
            continue

        if skip_next:
            skip_next = False
            continue

        if current_phase:
            token = line.split()[0]
            if token.lower() == "phase":
                continue
            species_order.append(token)
            species_to_phase[token] = current_phase
            phase_to_species[current_phase].append(token)

    return species_order, species_to_phase, phase_to_species


def _parse_fort42_blocks(path: Path, expected_species: List[str]) -> List[Dict[str, Tuple[float, float]]]:
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
    blocks: List[Dict[str, Tuple[float, float]]] = []
    current: Dict[str, Tuple[float, float]] = {}

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("dndt and dndp by species"):
            if current:
                blocks.append(current)
                current = {}
            continue

        parts = stripped.split()
        if len(parts) < 4:
            continue

        try:
            int(parts[0])
            species = parts[1]
            dndt = float(parts[2])
            dndp = float(parts[3])
        except ValueError:
            continue

        if species in expected_species:
            current[species] = (dndt, dndp)

    if current:
        blocks.append(current)

    return blocks


def _build_phase_states_from_benchmark_row(
    row_index: int,
    param_records,
    hefesto_context,
    temperature_k: float,
    phase_to_species: Dict[str, List[str]],
    species_moles_rows: List[Dict[str, float]],
    phase_density_rows: List[Dict[str, float]],
    fort63_rows: List[Dict[str, float]],
    fort64_rows: List[Dict[str, float]],
    dndt_dndp_blocks: List[Dict[str, Tuple[float, float]]],
) -> List[HeFESToPhaseState]:
    thermal_attrs = hefesto_context.compute_component_attributes_at_temperature(
        temperature_k=float(temperature_k),
        batch_size=1,
        device=torch.device("cpu"),
    )

    species_row = species_moles_rows[row_index]
    density_row = phase_density_rows[row_index]
    vb_row = fort63_rows[row_index]
    vs_row = fort64_rows[row_index]
    derivative_block = dndt_dndp_blocks[row_index]

    phase_states: List[HeFESToPhaseState] = []
    for phase_name, species_names in phase_to_species.items():
        phase_moles = sum(species_row.get(species, 0.0) for species in species_names)
        if phase_moles <= 0.0:
            continue

        phase_mass = 0.0
        for species in species_names:
            amount = species_row.get(species, 0.0)
            if amount > 0.0:
                phase_mass += amount * param_records[species].value("formula_mass_g_mol")

        if phase_mass <= 0.0:
            continue

        phase_density = density_row.get(f"rh{phase_name}", 0.0)
        if phase_density <= 0.0:
            continue

        phase_volume = phase_mass / phase_density
        vb = vb_row.get(f"vb{phase_name}", 0.0)
        vs = vs_row.get(f"vs{phase_name}", 0.0)
        if vb == 0.0 or vs == 0.0:
            continue

        bulk_modulus = phase_density * vb * vb
        shear_modulus = phase_density * vs * vs
        molar_volume = phase_volume / phase_moles

        species_states: List[HeFESToSpeciesState] = []
        for species in species_names:
            amount = species_row.get(species, 0.0)
            if amount <= 0.0:
                continue

            dndt, dndp = derivative_block.get(species, (0.0, 0.0))

            thermodynamic_state = htp.compute_component_thermodynamic_state(
                temperature=float(temperature_k),
                volume=max(molar_volume, 1.0e-12),
                reference_volume=max(param_records[species].value("v0_cm3_mol"), 1.0e-12),
                bulk_modulus=max(bulk_modulus, 1.0e-12),
                atoms_per_formula=param_records[species].value("atoms_per_formula_unit"),
                formula_units_per_cell=param_records[species].value("formula_units_per_cell"),
                debye_temp=param_records[species].value("theta0_k"),
                debye_temps_2_3=(
                    param_records[species].value("debye_acoustic_branch_2"),
                    param_records[species].value("debye_acoustic_branch_3"),
                ),
                sin_temps=(
                    param_records[species].value("sin_acoustic_branch_1") * htp.HCOK,
                    param_records[species].value("sin_acoustic_branch_2") * htp.HCOK,
                    param_records[species].value("sin_acoustic_branch_3") * htp.HCOK,
                ),
                optic_continuum_upper=param_records[species].value("optic_continuum_upper") * htp.HCOK,
                optic_continuum_lower=param_records[species].value("optic_continuum_lower") * htp.HCOK,
                einstein_temps=(
                    param_records[species].value("einstein_oscillator_1") * htp.HCOK,
                    param_records[species].value("einstein_oscillator_2") * htp.HCOK,
                    param_records[species].value("einstein_oscillator_3") * htp.HCOK,
                    param_records[species].value("einstein_oscillator_4") * htp.HCOK,
                ),
                einstein_weights=(
                    param_records[species].value("einstein_weight_1"),
                    param_records[species].value("einstein_weight_2"),
                    param_records[species].value("einstein_weight_3"),
                    param_records[species].value("einstein_weight_4"),
                ),
                gamma_0=param_records[species].value("gamma_0"),
                q_0=param_records[species].value("q_0"),
                got=param_records[species].value("ambient_shear_modulus_gpa"),
                reference_entropy_j_mol_k=param_records[species].value("critical_entropy_j_mol_k"),
                ityp=3,
            )

            cp_val = thermodynamic_state.heat_capacity_p
            cv_val = thermodynamic_state.heat_capacity_v
            alpha_val = thermodynamic_state.thermal_expansivity
            entropy_val = thermodynamic_state.entropy
            enthalpy_val = param_records[species].value("f0_kj_mol")
            gibbs_val = param_records[species].value("f0_kj_mol")

            species_states.append(
                HeFESToSpeciesState(
                    name=species,
                    phase_name=phase_name,
                    amount=amount,
                    molar_mass=param_records[species].value("formula_mass_g_mol"),
                    molar_volume=molar_volume,
                    density=phase_density,
                    bulk_modulus_t=bulk_modulus,
                    bulk_modulus_s=bulk_modulus,
                    shear_modulus=shear_modulus,
                    heat_capacity_p=cp_val,
                    heat_capacity_v=cv_val,
                    thermal_expansivity=alpha_val,
                    entropy=entropy_val,
                    enthalpy=enthalpy_val,
                    gibbs=gibbs_val,
                    dndt=dndt,
                    dndp=dndp,
                )
            )

        if species_states:
            phase_states.append(HeFESToPhaseState(name=phase_name, species=tuple(species_states)))

    return phase_states


def compare_physub_against_benchmark_directory(
    benchmark_dir: Path | str,
    param_records,
    hefesto_context,
    verbose: bool = False,
) -> HeFESToBenchmarkComparisonResult:
    benchmark_dir = Path(benchmark_dir)

    def _vprint(message: str) -> None:
        if verbose:
            print(message)

    required = ["control", "fort.42", "fort.58", "fort.61", "fort.62", "fort.63", "fort.99"]
    if not all((benchmark_dir / name).exists() for name in required):
        raise FileNotFoundError(f"Benchmark tables are unavailable at {benchmark_dir}")

    species_order, _, phase_to_species = _parse_control_species_map(benchmark_dir / "control")
    fort58_header, fort58_rows = _read_benchmark_table(benchmark_dir / "fort.58")
    fort61_header, fort61_rows = _read_benchmark_table(benchmark_dir / "fort.61")
    fort62_header, fort62_rows = _read_benchmark_table(benchmark_dir / "fort.62")
    fort63_header, fort63_rows = _read_benchmark_table(benchmark_dir / "fort.63")
    fort99_header, fort99_rows = _read_benchmark_table(benchmark_dir / "fort.99")
    dndt_dndp_blocks = _parse_fort42_blocks(benchmark_dir / "fort.42", species_order)

    from importlib import import_module

    extracted_bulk = import_module("HeFESTo.HeFESTo_functions").extract_bulk_properties_from_simulation_dir(str(benchmark_dir))
    fort56_bulk = extracted_bulk.get("fort56_bulk", {})
    fort59_bulk = extracted_bulk.get("fort59_bulk", {})

    n_rows = min(
        len(fort58_rows),
        len(fort61_rows),
        len(fort62_rows),
        len(fort63_rows),
        len(fort99_rows),
        len(dndt_dndp_blocks),
        len(extracted_bulk.get("P(GPa)", [])),
    )

    if n_rows <= 0:
        raise ValueError("No common benchmark rows were found across required fort.* files")

    def _pick_reference_column(source: Dict[str, torch.Tensor], candidates: Tuple[str, ...]):
        lower_map = {str(key).lower(): key for key in source.keys()}
        for candidate in candidates:
            if candidate in source:
                return source[candidate]
            matched = lower_map.get(candidate.lower())
            if matched is not None:
                return source[matched]
        raise KeyError(f"None of {candidates} found in source columns: {list(source.keys())}")

    cp_ref_vec = _pick_reference_column(fort56_bulk, ("cp(J/g/K)", "cp(j/g/k)"))
    alpha_ref_vec = _pick_reference_column(fort56_bulk, ("alpha(1e5_K^-1)", "alpha(1e5 k^-1)", "alpha"))
    entropy_ref_vec = _pick_reference_column(fort56_bulk, ("S(J/g/K)", "s(j/g/k)", "S"))
    try:
        cv_ref_vec = _pick_reference_column(fort59_bulk, ("Heat C", "Heat", "cv", "Cv"))
    except KeyError:
        cv_ref_vec = cp_ref_vec
    gamma_ref_vec = _pick_reference_column(fort59_bulk, ("g", "gamma"))

    density_rel_err: List[float] = []
    ks_rel_err: List[float] = []
    g_rel_err: List[float] = []
    vb_rel_err: List[float] = []
    vs_rel_err: List[float] = []
    vp_rel_err: List[float] = []
    cp_rel_err: List[float] = []
    cv_rel_err: List[float] = []
    alpha_rel_err: List[float] = []
    gamma_rel_err: List[float] = []
    entropy_rel_err: List[float] = []

    for row_index in range(n_rows):
        target = fort58_rows[row_index]
        phase_states = _build_phase_states_from_benchmark_row(
            row_index=row_index,
            param_records=param_records,
            hefesto_context=hefesto_context,
            temperature_k=target["Ti"],
            phase_to_species=phase_to_species,
            species_moles_rows=fort99_rows,
            phase_density_rows=fort61_rows,
            fort63_rows=fort62_rows,
            fort64_rows=fort63_rows,
            dndt_dndp_blocks=dndt_dndp_blocks,
        )

        output = compute_physub_properties(
            phases=phase_states,
            pressure_gpa=target["Pi"],
            temperature_k=target["Ti"],
            depth_km=target["depth"],
        )

        def _rel_err(observed: float, expected: float) -> float:
            return abs(observed - expected) / max(abs(expected), 1.0e-12)

        density_rel_err.append(_rel_err(output.density, target["rho"]))
        ks_rel_err.append(_rel_err(output.bulk_modulus_hill, target["KS"]))
        g_rel_err.append(_rel_err(output.shear_modulus_hill, target["G"]))
        vb_rel_err.append(_rel_err(output.bulk_sound_velocity, target["VBh"]))
        vs_rel_err.append(_rel_err(output.shear_velocity, target["VSh"]))
        vp_rel_err.append(_rel_err(output.pressure_velocity, target["VPh"]))

        cp_rel_err.append(_rel_err(float(output.heat_capacity_p), float(cp_ref_vec[row_index])))
        cv_rel_err.append(_rel_err(float(output.heat_capacity_v), float(cv_ref_vec[row_index])))
        alpha_rel_err.append(_rel_err(float(output.thermal_expansivity) * 1.0e5, float(alpha_ref_vec[row_index])))
        gamma_rel_err.append(_rel_err(float(output.gruneisen_parameter), float(gamma_ref_vec[row_index])))
        entropy_rel_err.append(
            _rel_err(float(output.entropy) / max(abs(float(output.mass)), 1.0e-12), float(entropy_ref_vec[row_index]))
        )

        _vprint(
            " | ".join(
                [
                    f"row={row_index:03d}",
                    f"P={target['Pi']:.3f} GPa",
                    f"T={target['Ti']:.2f} K",
                    f"depth={target['depth']:.2f} km",
                    f"phases={len(phase_states)}",
                    f"species={sum(len(phase.species) for phase in phase_states)}",
                ]
            )
        )
        _vprint(
            "    "
            + " | ".join(
                [
                    f"density pred={output.density:.6f} ref={target['rho']:.6f}",
                    f"KS pred={output.bulk_modulus_hill:.6f} ref={target['KS']:.6f}",
                    f"G pred={output.shear_modulus_hill:.6f} ref={target['G']:.6f}",
                ]
            )
        )
        _vprint(
            "    "
            + " | ".join(
                [
                    f"VBh pred={output.bulk_sound_velocity:.6f} ref={target['VBh']:.6f}",
                    f"VSh pred={output.shear_velocity:.6f} ref={target['VSh']:.6f}",
                    f"VPh pred={output.pressure_velocity:.6f} ref={target['VPh']:.6f}",
                ]
            )
        )
        _vprint(
            "    "
            + " | ".join(
                [
                    f"cp pred={float(output.heat_capacity_p):.6f} ref={float(cp_ref_vec[row_index]):.6f}",
                    f"cv pred={float(output.heat_capacity_v):.6f} ref={float(cv_ref_vec[row_index]):.6f}",
                    f"alpha(1e5) pred={float(output.thermal_expansivity) * 1.0e5:.6f} ref={float(alpha_ref_vec[row_index]):.6f}",
                ]
            )
        )
        _vprint(
            "    "
            + " | ".join(
                [
                    f"gamma pred={float(output.gruneisen_parameter):.6f} ref={float(gamma_ref_vec[row_index]):.6f}",
                    f"S pred={float(output.entropy) / max(abs(float(output.mass)), 1.0e-12):.6f} ref={float(entropy_ref_vec[row_index]):.6f}",
                ]
            )
        )

    mean_errors = {
        "density": sum(density_rel_err) / len(density_rel_err),
        "KS": sum(ks_rel_err) / len(ks_rel_err),
        "G": sum(g_rel_err) / len(g_rel_err),
        "VBh": sum(vb_rel_err) / len(vb_rel_err),
        "VSh": sum(vs_rel_err) / len(vs_rel_err),
        "VPh": sum(vp_rel_err) / len(vp_rel_err),
        "cp": sum(cp_rel_err) / len(cp_rel_err),
        "cv": sum(cv_rel_err) / len(cv_rel_err),
        "alpha": sum(alpha_rel_err) / len(alpha_rel_err),
        "gamma": sum(gamma_rel_err) / len(gamma_rel_err),
        "S": sum(entropy_rel_err) / len(entropy_rel_err),
    }
    max_errors = {
        "density": max(density_rel_err),
        "KS": max(ks_rel_err),
        "G": max(g_rel_err),
        "VBh": max(vb_rel_err),
        "VSh": max(vs_rel_err),
        "VPh": max(vp_rel_err),
        "cp": max(cp_rel_err),
        "cv": max(cv_rel_err),
        "alpha": max(alpha_rel_err),
        "gamma": max(gamma_rel_err),
        "S": max(entropy_rel_err),
    }

    passed = (
        mean_errors["density"] < 0.001
        and mean_errors["KS"] < 0.001
        and mean_errors["G"] < 0.001
        and mean_errors["VBh"] < 0.001
        and mean_errors["VSh"] < 0.001
        and mean_errors["VPh"] < 0.001
        and mean_errors["cp"] < 0.01
        and mean_errors["cv"] < 0.01
        and mean_errors["alpha"] < 0.01
        and mean_errors["gamma"] < 0.01
        and mean_errors["S"] < 0.01
    )

    return HeFESToBenchmarkComparisonResult(
        passed=passed,
        n_rows=n_rows,
        mean_errors=mean_errors,
        max_errors=max_errors,
    )


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


def _component_attribute_tensor(
    component_moles: torch.Tensor,
    component_attributes: Optional[Mapping[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    attrs: Dict[str, torch.Tensor] = {}
    for name in _PHYSUB_COMPONENT_ATTRIBUTE_NAMES:
        if component_attributes is None or name not in component_attributes:
            attrs[name] = torch.zeros_like(component_moles)
            continue
        value = component_attributes[name]
        if value.shape != component_moles.shape:
            raise ValueError(
                f"component attribute '{name}' must match component_moles shape; "
                f"got {tuple(value.shape)} vs {tuple(component_moles.shape)}"
            )
        attrs[name] = value
    return attrs


def _resolve_bulk_selectors(selectors: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if selectors is None:
        return PHYSUB_BULK_ATTRIBUTE_NAMES

    cleaned: List[str] = []
    for selector in selectors:
        if selector not in PHYSUB_BULK_ATTRIBUTE_NAMES:
            raise KeyError(
                f"Unknown bulk attribute selector '{selector}'. "
                f"Valid selectors: {', '.join(PHYSUB_BULK_ATTRIBUTE_NAMES)}"
            )
        if selector not in cleaned:
            cleaned.append(selector)
    return tuple(cleaned)


def compute_physub_bulk_matrix(
    component_moles: torch.Tensor,
    molar_mass: torch.Tensor,
    component_attributes: Optional[Mapping[str, torch.Tensor]] = None,
    selectors: Optional[Sequence[str]] = None,
    temperature_k: Optional[float] = None,
    hefesto_context: Optional[HeFESToPhysubContext] = None,
    eps: float = 1.0e-12,
) -> Tuple[torch.Tensor, Tuple[str, ...]]:
    """Torch-native matrix-first bulk evaluator.

    Parameters
    ----------
    component_moles
        Tensor with shape (B, C) containing extensive component moles.
    molar_mass
        Tensor with shape (C,) or (1, C) for component molar masses.
    component_attributes
        Optional mapping of per-component attribute tensors with shape (B, C).
        Supported keys: molar_volume, bulk_modulus, shear_modulus,
        heat_capacity_p, heat_capacity_v, thermal_expansivity, entropy,
        enthalpy, gibbs.
    selectors
        Optional ordered subset of PHYSUB_BULK_ATTRIBUTE_NAMES.
    temperature_k
        Optional temperature in Kelvin. If provided with hefesto_context,
        will compute temperature-dependent thermal properties (Cp, Cv, alpha, entropy).
    hefesto_context
        Optional HeFESToPhysubContext context for computing thermal attributes.
        Required if temperature_k is provided.

    Returns
    -------
    tuple
        (bulk_tensor, attribute_names) where bulk_tensor has shape (B, A).
    """

    if component_moles.ndim != 2:
        raise ValueError(f"component_moles must have shape (B, C), got {tuple(component_moles.shape)}")

    if molar_mass.ndim == 2 and molar_mass.shape[0] == 1:
        molar_mass = molar_mass.squeeze(0)
    if molar_mass.ndim != 1:
        raise ValueError(f"molar_mass must be rank-1 (C,), got {tuple(molar_mass.shape)}")
    if molar_mass.shape[0] != component_moles.shape[1]:
        raise ValueError("molar_mass length must match component_moles.shape[1]")

    # If temperature and context are provided, compute thermal attributes
    if temperature_k is not None and hefesto_context is not None:
        thermal_attrs = hefesto_context.compute_component_attributes_at_temperature(
            temperature_k=temperature_k,
            batch_size=component_moles.shape[0],
            device=component_moles.device,
        )
        # Merge thermal attributes with provided attributes
        if component_attributes is None:
            component_attributes = thermal_attrs
        else:
            merged = dict(component_attributes)
            merged.update(thermal_attrs)
            component_attributes = merged

    requested = _resolve_bulk_selectors(selectors)
    attrs = _component_attribute_tensor(component_moles, component_attributes)

    mass_components = component_moles * molar_mass.unsqueeze(0)
    total_mass = mass_components.sum(dim=1)
    total_volume = (component_moles * attrs["molar_volume"]).sum(dim=1)
    density = total_mass / torch.clamp(total_volume, min=eps)

    vol_weight = component_moles * attrs["molar_volume"]
    k_voigt_num = (vol_weight * attrs["bulk_modulus"]).sum(dim=1)
    g_voigt_num = (vol_weight * attrs["shear_modulus"]).sum(dim=1)
    k_reuss_den = (vol_weight / torch.clamp(attrs["bulk_modulus"], min=eps)).sum(dim=1)
    g_reuss_den = (vol_weight / torch.clamp(attrs["shear_modulus"], min=eps)).sum(dim=1)

    k_voigt = k_voigt_num / torch.clamp(total_volume, min=eps)
    g_voigt = g_voigt_num / torch.clamp(total_volume, min=eps)
    k_reuss = total_volume / torch.clamp(k_reuss_den, min=eps)
    g_reuss = total_volume / torch.clamp(g_reuss_den, min=eps)
    k_hill = 0.5 * (k_voigt + k_reuss)
    g_hill = 0.5 * (g_voigt + g_reuss)

    vb = torch.sign(k_hill) * torch.sqrt(torch.abs(k_hill / torch.clamp(density, min=eps)))
    vs = torch.sign(g_hill) * torch.sqrt(torch.abs(g_hill / torch.clamp(density, min=eps)))
    vp = torch.sign(k_hill + (4.0 / 3.0) * g_hill) * torch.sqrt(
        torch.abs((k_hill + (4.0 / 3.0) * g_hill) / torch.clamp(density, min=eps))
    )
    vd_inv_cubed = 2.0 / (3.0 * torch.clamp(vs.abs() ** 3, min=eps)) + 1.0 / (
        3.0 * torch.clamp(vp.abs() ** 3, min=eps)
    )
    vd = torch.sign(vd_inv_cubed) / torch.clamp(torch.abs(vd_inv_cubed) ** (1.0 / 3.0), min=eps)

    cp = (attrs["heat_capacity_p"] * mass_components).sum(dim=1) / torch.clamp(total_mass, min=eps)
    cv = (attrs["heat_capacity_v"] * mass_components).sum(dim=1) / torch.clamp(total_mass, min=eps)
    alpha = (attrs["thermal_expansivity"] * vol_weight).sum(dim=1) / torch.clamp(total_volume, min=eps)
    gamma = 1000.0 * k_hill * alpha / torch.clamp(density * cp, min=eps)
    # Clamp gamma to a reasonable physical range to avoid extreme outliers
    gamma = torch.clamp(gamma, min=0.0, max=100.0)

    entropy = (attrs["entropy"] * component_moles).sum(dim=1)
    enthalpy = (attrs["enthalpy"] * component_moles).sum(dim=1)
    gibbs = (attrs["gibbs"] * component_moles).sum(dim=1)

    output_lookup: Dict[str, torch.Tensor] = {
        "density": density,
        "K_Reuss": k_reuss,
        "K_Voigt": k_voigt,
        "K_Hill": k_hill,
        "G_Reuss": g_reuss,
        "G_Voigt": g_voigt,
        "G_Hill": g_hill,
        "Vb": vb,
        "Vs": vs,
        "Vp": vp,
        "Vd": vd,
        "Cp": cp,
        "Cv": cv,
        "alpha": alpha,
        "gamma": gamma,
        "entropy": entropy,
        "enthalpy": enthalpy,
        "Gibbs": gibbs,
    }

    output = torch.stack([output_lookup[name] for name in requested], dim=1)
    return output, requested


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

    phase_count = len(phase_outputs)
    phase_moles = torch.tensor(
        [[phase.n_moles for phase in phase_outputs]], dtype=torch.float32
    )
    phase_molar_masses = torch.tensor(
        [
            _safe_divide(phase.mass, phase.n_moles, default=0.0)
            for phase in phase_outputs
        ],
        dtype=torch.float32,
    )
    phase_molar_volumes = torch.tensor(
        [
            _safe_divide(phase.volume, phase.n_moles, default=0.0)
            for phase in phase_outputs
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    phase_bulk = torch.tensor(
        [[phase.bulk_modulus_t for phase in phase_outputs]], dtype=torch.float32
    )
    phase_shear = torch.tensor(
        [[phase.shear_modulus for phase in phase_outputs]], dtype=torch.float32
    )
    phase_cp = torch.tensor(
        [[phase.heat_capacity_p for phase in phase_outputs]], dtype=torch.float32
    )
    phase_cv = torch.tensor(
        [[phase.heat_capacity_v for phase in phase_outputs]], dtype=torch.float32
    )
    phase_alpha = torch.tensor(
        [[phase.thermal_expansivity for phase in phase_outputs]], dtype=torch.float32
    )
    phase_entropy = torch.tensor(
        [[phase.entropy for phase in phase_outputs]], dtype=torch.float32
    )
    phase_enthalpy = torch.tensor(
        [[phase.enthalpy for phase in phase_outputs]], dtype=torch.float32
    )
    phase_gibbs = torch.tensor(
        [[phase.gibbs for phase in phase_outputs]], dtype=torch.float32
    )

    bulk_matrix, _ = compute_physub_bulk_matrix(
        component_moles=phase_moles,
        molar_mass=phase_molar_masses,
        component_attributes={
            "molar_volume": phase_molar_volumes,
            "bulk_modulus": phase_bulk,
            "shear_modulus": phase_shear,
            "heat_capacity_p": phase_cp,
            "heat_capacity_v": phase_cv,
            "thermal_expansivity": phase_alpha,
            "entropy": phase_entropy,
            "enthalpy": phase_enthalpy,
            "gibbs": phase_gibbs,
        },
        selectors=PHYSUB_BULK_ATTRIBUTE_NAMES,
    )
    bulk_values = {name: float(bulk_matrix[0, i].item()) for i, name in enumerate(PHYSUB_BULK_ATTRIBUTE_NAMES)}

    density = bulk_values["density"]
    bulk_modulus_reuss = bulk_values["K_Reuss"]
    bulk_modulus_voigt = bulk_values["K_Voigt"]
    bulk_modulus_hill = bulk_values["K_Hill"]
    shear_modulus_reuss = bulk_values["G_Reuss"]
    shear_modulus_voigt = bulk_values["G_Voigt"]
    shear_modulus_hill = bulk_values["G_Hill"]
    bulk_sound_velocity = bulk_values["Vb"]
    shear_velocity = bulk_values["Vs"]
    pressure_velocity = bulk_values["Vp"]
    debye_velocity = abs(bulk_values["Vd"])
    debye_velocity_signed = bulk_values["Vd"]
    heat_capacity_p = bulk_values["Cp"]
    heat_capacity_v = bulk_values["Cv"]
    thermal_expansivity = bulk_values["alpha"]
    gruneisen_parameter = bulk_values["gamma"]

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
        entropy=bulk_values["entropy"],
        enthalpy=bulk_values["enthalpy"],
        gibbs=bulk_values["Gibbs"],
        phases=tuple(phase_outputs),
    )
