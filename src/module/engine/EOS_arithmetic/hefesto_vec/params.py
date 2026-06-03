"""
Parameter loader for HeFESTo.

Reads the control file and all species parameter files from
HeFESTo_Parameters_010123/ (or any parameter directory specified in the
control file).  Applies unit conversions at load time so downstream code
works with consistent units throughout:

    - F0        : kJ/mol  →  J/mol   (× 1000)
    - ws1-ws3   : cm⁻¹   →  K       (× hcok)
    - we1-we4   : cm⁻¹   →  K       (× hcok)
    - wou, wol  : cm⁻¹   →  K       (× hcok)

All other parameters are dimensionally as stored in the parameter files.

apar column index mapping (1-indexed, matching parset.f):
    1   fn      atoms in formula unit
    2   zu      Z, formula units in unit cell
    3   wm      formula mass (g/mol)
    4   To      T_0 reference temperature (K)
    5   Fo      F_0 standard Helmholtz free energy (J/mol after conversion)
    6   Vo      V_0 reference volume (cm³/mol)
    7   Ko      K_0 isothermal bulk modulus (GPa)
    8   Kop     K_0' pressure derivative
    9   Kopp    K_0'' (0 → 3rd-order BM)
    10  wd1     Debye acoustic branch 1 θ (K)
    11  wd2     Debye acoustic branch 2 θ (K)
    12  wd3     Debye acoustic branch 3 θ (K)
    13  ws1     Sinusoidal branch 1 (K, converted from cm⁻¹)
    14  ws2     Sinusoidal branch 2 (K)
    15  ws3     Sinusoidal branch 3 (K)
    16  we1     Einstein oscillator 1 (K, converted from cm⁻¹)
    17  qe1     weight of Einstein 1
    18  we2     Einstein oscillator 2 (K)
    19  qe2     weight of Einstein 2
    20  we3     Einstein oscillator 3 (K)
    21  qe3     weight of Einstein 3
    22  we4     Einstein oscillator 4 (K)
    23  qe4     weight of Einstein 4
    24  wou     optic continuum upper (K, converted from cm⁻¹)
    25  wol     optic continuum lower (K, converted from cm⁻¹)
    26  gam     γ_0 Grüneisen parameter
    27  qo      q_0 volume exponent
    28  be      β  electronic heat capacity coefficient
    29  ge      γ_el electronic Grüneisen
    30  q2A2    anharmonic coefficient
    31  htl     high-temperature limit flag (1=yes)
    32  ibv     EOS type: 0=Birch-Murnaghan, 1=Vinet, 2=Lagrangian
    33  ied     DOS type flag (0=Einstein, 1=Debye) — controls ityp interaction
    34  izp     zero-point pressure flag (±1=yes)
    35  Go      G_0 ambient shear modulus (GPa)
    36  Gop     G_0' pressure derivative of G
    37  Got     η_S0 temperature derivative of G (shear thermal parameter)
    38  Tc      Landau critical temperature (K)
    39  Smax    Landau critical entropy (J/mol/K)
    40  Vmax    Landau critical volume (cm³/mol)
    41  (Van Laar size parameter)
    42  C12'    default 2.0 (set by readin.f before reading file)
    43  C44'

Parameters 44–100 reserved (zero by default).
apar[:,51:54] (1-indexed 51-54) are volume bounds computed by parset.f.

Note
----
Inelastic parameters (Hessian of molar second derivatives w.r.t. composition)
are not loaded here — they are irrelevant for the fixed-composition Vp/Vs path.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np

from .constants import hcok

# Number of parameter columns in apar (≥ 43 + volume bounds at 51-54)
NPARP = 100


@dataclass
class HeFESToParams:
    """All parameters loaded from a control file + parameter directory."""
    # Global theory flags
    ityp  : int   # 1=full quadratic gamset (benchmark default)
    ivtyp : int   # 1=full G expansion
    ittyp : int   # 1=full G thermal correction
    iltyp : int   # 1=landauqr

    # Species list
    nspec     : int
    snames    : List[str]              # length nspec
    apar      : np.ndarray             # (nspec, NPARP), 0-indexed species, 0-indexed cols

    # Phase structure
    phase_names   : List[str]          # length nphase
    phase_members : List[List[int]]    # phase_members[iph] = list of 0-indexed ispec
    phase_iltyp   : List[int]          # per-phase Landau flag (from control file integers)
    lphase        : np.ndarray         # (nspec,) 0-indexed phase index for each species

    # Convenience: species name → 0-indexed index
    spec_index: Dict[str, int] = field(default_factory=dict)

    # Site occupancy data for mixing entropy (from chemical formulas).
    # site_data[ispec] = list of (element_str, stoich_int) per non-O site position.
    # Indexed identically to snames / apar rows.
    site_data: List[List[Tuple[str, int]]] = field(default_factory=list)


def load_control(control_path: str, param_dir_override: str | None = None) -> HeFESToParams:
    """Parse a HeFESTo control file and load all species parameters.

    Parameters
    ----------
    control_path : str
        Path to the ``control`` file (e.g., BENCHMARK/control).
    param_dir_override : str, optional
        If given, use this as the parameter directory instead of the one
        embedded in the control file (useful for portability).

    Returns
    -------
    HeFESToParams
    """
    with open(control_path, 'r') as fh:
        lines = fh.read().splitlines()

    line_iter = iter(lines)

    def next_nonempty():
        for ln in line_iter:
            ln = ln.strip()
            if ln:
                return ln
        raise StopIteration

    # ── Line 1: global run parameters (not needed for EOS) ───────────────────
    next_nonempty()   # e.g. "0,140,14,2.533...,0,0,-2,0,0,0,0"

    # ── Line 2: nc, ne, ... (bulk chemistry metadata) ────────────────────────
    next_nonempty()

    # ── Lines 3..3+nc: component type and bulk composition ───────────────────
    comp_type = next_nonempty()   # e.g. "oxides"
    # Detect number of component lines by reading until we hit the "1,1,1,1" line
    while True:
        ln = next_nonempty()
        # The theory-flags line has only digits, commas, spaces
        if re.fullmatch(r'[\d,\s]+', ln.strip()):
            theory_line = ln
            break

    # ── Theory flags: ityp, ivtyp, ittyp, iltyp ──────────────────────────────
    flags = [int(v.strip()) for v in theory_line.split(',')]
    ityp, ivtyp, ittyp, iltyp = flags[0], flags[1], flags[2], flags[3]

    # ── Parameter directory ───────────────────────────────────────────────────
    param_dir_line = next_nonempty()
    if param_dir_override is not None:
        param_dir = param_dir_override
    else:
        param_dir = param_dir_line.strip()

    # ── Number of species ─────────────────────────────────────────────────────
    nspec = int(next_nonempty())

    # ── Phase / species block ─────────────────────────────────────────────────
    phase_names   : List[str]       = []
    phase_members : List[List[int]] = []
    phase_iltyp   : List[int]       = []
    snames        : List[str]       = []
    lphase        : List[int]       = []   # 0-indexed phase per species

    current_phase_idx = -1
    ispec_count = 0

    remaining_lines: List[str] = []
    for ln in line_iter:
        remaining_lines.append(ln)

    rl_iter = iter(remaining_lines)

    def next_rl():
        for ln in rl_iter:
            ln = ln.strip()
            if ln:
                return ln
        return None

    while ispec_count < nspec:
        ln = next_rl()
        if ln is None:
            break
        if ln.lower().startswith('phase'):
            parts = ln.split()
            ph_name = parts[1] if len(parts) > 1 else f'phase{len(phase_names)}'
            phase_names.append(ph_name)
            phase_members.append([])
            # Read per-phase Landau integer (0=absent or no-Landau, 1=Landau)
            ph_flag_str = next_rl()
            ph_flag = int(float(ph_flag_str)) if ph_flag_str else 0
            phase_iltyp.append(ph_flag)
            current_phase_idx = len(phase_names) - 1
        else:
            # This line is a species name
            snames.append(ln)
            lphase.append(current_phase_idx)
            if current_phase_idx >= 0:
                phase_members[current_phase_idx].append(ispec_count)
            ispec_count += 1

    # ── Load parameter files ──────────────────────────────────────────────────
    apar = np.zeros((nspec, NPARP), dtype=np.float64)
    # Default: C12' = 2.0 (set by readin.f before reading file)
    apar[:, 41] = 2.0   # 0-indexed col 41 = 1-indexed col 42

    formulas: List[str] = []
    for ispec, sname in enumerate(snames):
        fpath = os.path.join(param_dir, sname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"Parameter file not found: {fpath!r}\n"
                f"(species '{sname}', index {ispec})"
            )
        formula_str = _read_param_file(fpath, ispec, apar)
        formulas.append(formula_str)

    # ── Apply unit conversions (matching parset.f) ───────────────────────────
    # Fo: kJ/mol → J/mol
    apar[:, 4] *= 1000.0   # col 5 (0-indexed: 4)

    # ws1-ws3 (cols 13-15, 0-indexed 12-14): cm⁻¹ → K
    apar[:, 12] *= hcok
    apar[:, 13] *= hcok
    apar[:, 14] *= hcok

    # we1-we4 (cols 16,18,20,22 → 0-indexed 15,17,19,21): cm⁻¹ → K
    apar[:, 15] *= hcok
    apar[:, 17] *= hcok
    apar[:, 19] *= hcok
    apar[:, 21] *= hcok

    # wou, wol (cols 24-25 → 0-indexed 23-24): cm⁻¹ → K
    apar[:, 23] *= hcok
    apar[:, 24] *= hcok

    # ── Build species name → index dict ──────────────────────────────────────
    spec_index = {name: i for i, name in enumerate(snames)}

    # ── Parse formulas into per-site occupancy data ───────────────────────────
    site_data = [_parse_formula(f) for f in formulas]

    return HeFESToParams(
        ityp=ityp, ivtyp=ivtyp, ittyp=ittyp, iltyp=iltyp,
        nspec=nspec,
        snames=snames,
        apar=apar,
        phase_names=phase_names,
        phase_members=phase_members,
        phase_iltyp=phase_iltyp,
        lphase=np.asarray(lphase, dtype=np.int32),
        spec_index=spec_index,
        site_data=site_data,
    )


def _parse_formula(formula_str: str) -> List[List[Tuple[str, int]]]:
    """Parse a HeFESTo chemical formula into per-site component lists.

    Each site is a list of (element, stoich) pairs.  Single-component sites
    have one entry; multi-component sites (parenthesised groups) have several.

    Examples
    --------
    'Mg_1Si_1O_3'           → [[(Mg,1)], [(Si,1)]]
    'Mg_2Mg_2O_4'           → [[(Mg,2)], [(Mg,2)]]
    '(Na_2Mg_1)Si_1Si_3O_12'→ [[(Na,2),(Mg,1)], [(Si,1)], [(Si,3)]]
    'Ca_1Al_1(Si_1Al_1)O_6' → [[(Ca,1)], [(Al,1)], [(Si,1),(Al,1)]]

    Parenthesised groups pack multiple components onto a single site,
    matching the HeFESTo Fortran convention used in cformula.f.
    Oxygen-containing sites are excluded.
    """
    sites: List[List[Tuple[str, int]]] = []
    i = 0
    s = formula_str.strip()
    while i < len(s):
        if s[i] == '(':
            # Parenthesised multi-component site
            j = s.index(')', i)
            comps = [(el, int(n))
                     for el, n in re.findall(r'([A-Z][a-z]*)_(\d+)', s[i+1:j])
                     if el != 'O']
            if comps:
                sites.append(comps)
            i = j + 1
        elif s[i].isupper():
            # Single-component site: El_N
            m = re.match(r'([A-Z][a-z]*)_(\d+)', s[i:])
            if m:
                el, n = m.group(1), int(m.group(2))
                if el != 'O':
                    sites.append([(el, n)])
                i += m.end()
            else:
                i += 1
        else:
            i += 1
    return sites


def _read_param_file(fpath: str, ispec: int, apar: np.ndarray) -> str:
    """Read up to 43 numeric values from a HeFESTo parameter file into apar.

    The file format has one numeric value per line, possibly preceded by
    whitespace and followed by a text label.  Lines that cannot be parsed as
    a float are silently skipped (this matches Fortran's ``read(1,*,...) err=``
    behaviour).

    Returns the first-line formula string (e.g. 'Mg_1Si_1O_3').
    """
    values: List[float] = []
    formula_str = ''
    with open(fpath, 'r') as fh:
        # First line is the chemical formula (and mineral name after whitespace)
        first_line = fh.readline()
        formula_str = first_line.split()[0] if first_line.split() else ''
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # First token is the numeric value
            token = line.split()[0]
            try:
                values.append(float(token))
            except ValueError:
                continue

    npar = min(len(values), 43)
    for i, v in enumerate(values[:npar]):
        apar[ispec, i] = v

    return formula_str


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build species-abundance array from a phase-abundance dict
# ─────────────────────────────────────────────────────────────────────────────

def phase_to_species_fractions(
    params: HeFESToParams,
    phase_mole_fracs: Dict[str, float],
    species_site_fracs: Dict[str, float] | None = None,
) -> np.ndarray:
    """Convert phase mole fractions → species mole fractions X (nspec,).

    For simple end-member phases (single species per phase) this is trivial.
    For solid solutions you must supply ``species_site_fracs`` mapping each
    species name to its site occupancy fraction within its phase.

    Parameters
    ----------
    params            : HeFESToParams
    phase_mole_fracs  : dict  phase_name → mole fraction (sum should be ≤ 1)
    species_site_fracs: dict  species_name → site fraction within phase
                              (required for multi-species phases)

    Returns
    -------
    X : (nspec,)  species mole fractions
    """
    X = np.zeros(params.nspec, dtype=np.float64)
    for iph, ph_name in enumerate(params.phase_names):
        ph_frac = phase_mole_fracs.get(ph_name, 0.0)
        if ph_frac == 0.0:
            continue
        members = params.phase_members[iph]
        if len(members) == 1:
            X[members[0]] = ph_frac
        else:
            if species_site_fracs is None:
                raise ValueError(
                    f"Phase '{ph_name}' has {len(members)} species; "
                    "supply species_site_fracs."
                )
            for ispec in members:
                sname = params.snames[ispec]
                X[ispec] = ph_frac * species_site_fracs.get(sname, 0.0)
    return X
