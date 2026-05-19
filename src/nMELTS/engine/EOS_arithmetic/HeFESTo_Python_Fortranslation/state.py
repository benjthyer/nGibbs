"""
Centralized state management for the HeFESTo EOS arithmetic engine,
designed to replace Fortran COMMON blocks with a unified Python object.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class HeFESToState:
    """
    A container for the shared state variables used throughout the HeFESTo
    physical properties calculations. This object is passed through the
    call stack to provide access to system conditions, compositions,
    and intermediate results.

    This class replaces the multiple Fortran COMMON blocks.
    """

    # System conditions
    Pi: float = 0.0  # Pressure (GPa)
    Ti: float = 0.0  # Temperature (K)

    # Dimensions
    nspec: int = 0        # Number of species in the current equilibrium assemblage
    nph: int = 0          # Number of phases in the current equilibrium assemblage
    nc: int = 0           # Number of chemical components 
    nco: int = 0          # Number of cation components (subset of nc, excludes oxygen)
    nspecp: int = 100     # Padded dimension for species (>= nspec, for array allocation)
    nphasep: int = 40     # Padded dimension for phases (>= nph, for array allocation)
    ncompp: int = 0       # Padded dimension for components (>= nc, for array allocation)
    nparp: int = 0        # Number of thermodynamic parameters per species (length of apar columns)
    nnull: int = 0        # Dimension of the null space basis (stoichiometric constraints)
    nnulls: int = 0       # Auxiliary null space dimension for SVD/linear algebra operations

    # Composition and Stoichiometry
    n: np.ndarray = field(default_factory=lambda: np.array([]))  # Species amounts (nspecp,)
    n1: np.ndarray = field(default_factory=lambda: np.array([])) # Species amounts in null space
    b: np.ndarray = field(default_factory=lambda: np.array([]))  # Bulk composition (ncompp,)
    q2: np.ndarray = field(default_factory=lambda: np.array([])) # Null space basis matrix (nspecp, nspecp)
    s: np.ndarray = field(default_factory=lambda: np.array([]))  # Stoichiometry matrix (ncompp, nspecp)
    f: np.ndarray = field(default_factory=lambda: np.array([]))  # Phase membership matrix (nphasep, nspecp)
    absents: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool)) # Absent species flags (nspecp,)

    # Species-level properties
    cpa: np.ndarray = field(default_factory=lambda: np.array([]))      # Chemical potentials (nspecp,)
    sspeca: np.ndarray = field(default_factory=lambda: np.array([]))   # Species entropy derivatives (nspecp,)
    vspeca: np.ndarray = field(default_factory=lambda: np.array([]))   # Species volume derivatives (nspecp,)
    apar: np.ndarray = field(default_factory=lambda: np.array([]))     # Raw parameters from files (nspecp, nparp)

    # Aggregate properties (many are outputs from physub)
    vol: float = 0.0
    Cap: float = 0.0
    Cv: float = 0.0
    gamma: float = 0.0
    K: float = 0.0
    Ks: float = 0.0
    alp: float = 0.0
    Ftot: float = 0.0
    ph: float = 0.0
    ent: float = 0.0
    deltas: float = 0.0
    tcal: float = 0.0
    zeta: float = 0.0
    Gsh: float = 0.0
    uth: float = 0.0
    uto: float = 0.0
    thet: float = 0.0
    qq: float = 0.0
    etas: float = 0.0
    dGdT: float = 0.0
    pzp: float = 0.0
    Vdeb: float = 0.0
    gamdeb: float = 0.0

    # Phase and species info
    phname: List[str] = field(default_factory=list)
    sname: List[str] = field(default_factory=list)
    iophase: np.ndarray = field(default_factory=lambda: np.array([]))
    mophase: np.ndarray = field(default_factory=lambda: np.array([]))
    iphase: np.ndarray = field(default_factory=lambda: np.array([]))

    # Spinodal stability
    spinod: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    spinph: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    # Optimization and search state
    lagc: np.ndarray = field(default_factory=lambda: np.array([])) # Lagrange multipliers
    iphyflag: int = 0

    # BLAS/LAPACK constants
    one: float = 1.0
    zero: float = 0.0
    ione: int = 1
    zervec: np.ndarray = field(default_factory=lambda: np.zeros(1))

    # File handles for logging (optional)
    file_handles: Dict[int, Any] = field(default_factory=dict)

    # Other Fortran state
    icfe: int = 0
    Rgas: float = 8.314462618
    Ksp: float = 0.0
    Gshp: float = 0.0
    fn: float = 0.0
    zu: float = 0.0
    stox: np.ndarray = field(default_factory=lambda: np.array([]))
    wox: np.ndarray = field(default_factory=lambda: np.array([]))
    wcomp: np.ndarray = field(default_factory=lambda: np.array([]))
    stcomp: np.ndarray = field(default_factory=lambda: np.array([]))
    atom: np.ndarray = field(default_factory=lambda: np.array([]))
    comp: np.ndarray = field(default_factory=lambda: np.array([]))
    tlast: float = 0.0
    vtarg: float = 0.0
    starg: float = 0.0
    phugo: float = 0.0
    vhugo: float = 0.0
    ehugo: float = 0.0
    dvdpmol: float = 0.0
    dsdtmol: float = 0.0
    dhdpmol: float = 0.0
    dhdtmol: float = 0.0
    wmaggc: float = 0.0
    chcalc: bool = False
    adcalc: bool = False
    hucalc: bool = False
    nvet: int = 0
    nvep: int = 0
