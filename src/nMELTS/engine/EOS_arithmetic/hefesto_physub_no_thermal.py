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
)

_PHYSUB_COMPONENT_ATTRIBUTE_NAMES: Tuple[str, ...] = (
    "molar_volume",
    "bulk_modulus",
    "shear_modulus",
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
            self.projection_component_to_attributes = torch.empty((0, len(_PHYSUB_COMPONENT_ATTRIBUTE_NAMES)), dtype=torch.float32)
            return

        masses = [records[name].value("formula_mass_g_mol") for name in self.component_names]
        attrs = [
            [
                records[name].value("v0_cm3_mol"),
                records[name].value("k0_gpa"),
                records[name].value("ambient_shear_modulus_gpa"),
            ]
            for name in self.component_names
        ]

        self.formula_mass_g_mol = torch.tensor(masses, dtype=torch.float32)
        self.projection_component_to_mass = self.formula_mass_g_mol.unsqueeze(-1)
        self.projection_component_to_attributes = torch.tensor(attrs, dtype=torch.float32)

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
    """Species-level state for elastic-only aggregation."""

    name: str
    phase_name: str
    amount: float
    molar_mass: float
    molar_volume: float
    bulk_modulus_t: float
    shear_modulus: float
    is_absent: bool = False


@dataclass(frozen=True)
class HeFESToPhaseState:
    """A phase and its constituent species after equilibrium has been solved."""

    name: str
    species: Tuple[HeFESToSpeciesState, ...]


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
                f"component attribute '{name}' must match component_moles shape; got {tuple(value.shape)} vs {tuple(component_moles.shape)}"
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
                f"Unknown bulk attribute selector '{selector}'. Valid selectors: {', '.join(PHYSUB_BULK_ATTRIBUTE_NAMES)}"
            )
        if selector not in cleaned:
            cleaned.append(selector)
    return tuple(cleaned)


def compute_physub_bulk_matrix(
    component_moles: torch.Tensor,
    molar_mass: torch.Tensor,
    component_attributes: Optional[Mapping[str, torch.Tensor]] = None,
    selectors: Optional[Sequence[str]] = None,
    component_names: Optional[Sequence[str]] = None,
    hefesto_context: Optional[HeFESToPhysubContext] = None,
    temperature_k: Optional[float] = None,
    eps: float = 1.0e-12,
) -> Tuple[torch.Tensor, Tuple[str, ...]]:
    """Torch-native matrix-first bulk evaluator."""

    if component_moles.ndim != 2:
        raise ValueError(f"component_moles must have shape (B, C), got {tuple(component_moles.shape)}")

    # If a HeFESTo context is provided along with component names, align
    # the incoming component ordering to the context's canonical ordering.
    if hefesto_context is not None and component_names is not None:
        # If the incoming tensor already has the canonical number of columns
        # for the context, assume it is already aligned and ignore the
        # provided `component_names` ordering. Otherwise, require the
        # provided names to match the tensor width and perform alignment.
        if component_moles.shape[1] == len(hefesto_context.component_names):
            # already aligned to context ordering
            pass
        else:
            if component_moles.shape[1] != len(component_names):
                raise ValueError("component_names length must match component_moles.shape[1]")
            if tuple(component_names) != tuple(hefesto_context.component_names):
                component_moles = hefesto_context.align_component_tensor(component_moles, component_names)

    if molar_mass.ndim == 2 and molar_mass.shape[0] == 1:
        molar_mass = molar_mass.squeeze(0)
    if molar_mass.ndim != 1:
        raise ValueError(f"molar_mass must be rank-1 (C,), got {tuple(molar_mass.shape)}")
    if molar_mass.shape[0] != component_moles.shape[1]:
        # If context is available, prefer its canonical molar mass vector.
        if hefesto_context is not None:
            molar_mass = hefesto_context.formula_mass_g_mol
        else:
            raise ValueError("molar_mass length must match component_moles.shape[1]")

    requested = _resolve_bulk_selectors(selectors)

    # If no explicit component attributes were supplied, but we have a
    # HeFESTo context and a temperature, compute the temperature-dependent
    # per-component attributes using the context.
    if component_attributes is None and hefesto_context is not None and temperature_k is not None:
        # compute_component_attributes_at_temperature returns tensors of shape (batch, C)
        computed = hefesto_context.compute_component_attributes_at_temperature(
            temperature_k, batch_size=component_moles.shape[0], device=component_moles.device
        )
        # Ensure we only keep the attributes the reducer expects
        component_attributes = {
            name: computed[name] for name in _PHYSUB_COMPONENT_ATTRIBUTE_NAMES if name in computed
        }

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
    }

    output = torch.stack([output_lookup[name] for name in requested], dim=1)
    return output, requested


def compute_physub_properties(
    phases: Sequence[HeFESToPhaseState],
    pressure_gpa: float,
    temperature_k: float,
    depth_km: float = 0.0,
) -> HeFESToBulkProperties:
    """Compute bulk elastic properties from an equilibrium assemblage."""

    species_states = [species for phase in phases for species in phase.species if not species.is_absent and species.amount != 0.0]
    if not species_states:
        return HeFESToBulkProperties(
            pressure_gpa=pressure_gpa,
            depth_km=depth_km,
            temperature_k=temperature_k,
            density=0.0,
            volume=0.0,
            mass=0.0,
            bulk_modulus_reuss=0.0,
            bulk_modulus_voigt=0.0,
            bulk_modulus_hill=0.0,
            shear_modulus_reuss=0.0,
            shear_modulus_voigt=0.0,
            shear_modulus_hill=0.0,
            bulk_sound_velocity=0.0,
            shear_velocity=0.0,
            pressure_velocity=0.0,
        )

    component_moles = torch.tensor([[species.amount for species in species_states]], dtype=torch.float32)
    component_names = [species.name for species in species_states]

    # Use HeFESTo context to compute temperature-dependent attributes and
    # align component ordering automatically. Pass the canonical molar mass
    # vector from the context so shapes match after alignment.
    hefesto_ctx = get_hefesto_physub_context()
    molar_mass = hefesto_ctx.formula_mass_g_mol

    bulk_matrix, _ = compute_physub_bulk_matrix(
        component_moles=component_moles,
        molar_mass=molar_mass,
        component_attributes=None,
        selectors=PHYSUB_BULK_ATTRIBUTE_NAMES,
        component_names=component_names,
        hefesto_context=hefesto_ctx,
        temperature_k=temperature_k,
    )
    bulk_values = {name: float(bulk_matrix[0, index].item()) for index, name in enumerate(PHYSUB_BULK_ATTRIBUTE_NAMES)}
    total_mass = float((component_moles * molar_mass.unsqueeze(0)).sum().item())
    # Compute per-component attributes to evaluate total volume
    comp_attrs = hefesto_ctx.compute_component_attributes_at_temperature(
        temperature_k, batch_size=component_moles.shape[0], device=component_moles.device
    )
    total_volume = float((component_moles * comp_attrs["molar_volume"]).sum().item())

    return HeFESToBulkProperties(
        pressure_gpa=pressure_gpa,
        depth_km=depth_km,
        temperature_k=temperature_k,
        density=bulk_values["density"],
        volume=total_volume,
        mass=total_mass,
        bulk_modulus_reuss=bulk_values["K_Reuss"],
        bulk_modulus_voigt=bulk_values["K_Voigt"],
        bulk_modulus_hill=bulk_values["K_Hill"],
        shear_modulus_reuss=bulk_values["G_Reuss"],
        shear_modulus_voigt=bulk_values["G_Voigt"],
        shear_modulus_hill=bulk_values["G_Hill"],
        bulk_sound_velocity=bulk_values["Vb"],
        shear_velocity=bulk_values["Vs"],
        pressure_velocity=bulk_values["Vp"],
    )
