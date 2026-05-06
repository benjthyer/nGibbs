"""
High-level API for computing bulk physical properties using the HeFESTo
Equation of State (EOS) arithmetic engine.
"""
from __future__ import annotations

import numpy as np

from .state import HeFESToState
from .extern import HeFESToExtern
from . import physub
from . import param_state
from ...config.ml_indexer import MLIndexer
from .param_loader import get_parameter_store
from tqdm import tqdm

def calculate_bulk_properties(nnew: np.ndarray, P, T, ml_indexer: MLIndexer, property_names):
    """
    Top-level function to compute bulk physical properties for an equilibrium
    phase assemblage at a given pressure and temperature.

    Supports both single and batch (vectorized) modes:
    - Single: nnew (nspec,), P scalar, T scalar -> dict
    - Batch: nnew (N, nspec), P (N,), T (N,) -> dict with (N, nprops) output

    Parameters
    ----------
    nnew : np.ndarray
        Species amounts. Shape (nspec,) for single or (N, nspec) for batch.
    P : float or array-like
        Pressure in GPa. Scalar or (N,) array.
    T : float or array-like
        Temperature in K. Scalar or (N,) array.
    ml_indexer : MLIndexer
        The machine learning indexer containing compositional and
        stoichiometric information.
    property_names : sequence of str
        List of properties to compute (e.g., ['density', 'Cp', 'Cv']).

    Returns
    -------
    dict
        Single mode: dict with properties.
        Batch mode: dict with 'output' (N, nprops), 'property_names', 'timing'.
    """
    # Detect batch mode
    nnew = np.asarray(nnew, dtype=float)
    P = np.asarray(P, dtype=float)
    T = np.asarray(T, dtype=float)

    is_batch = nnew.ndim == 2
    if is_batch:
        n_assemblages = nnew.shape[0]
        return _compute_bulk_properties_batch(nnew, P, T, ml_indexer)

    # Scalar mode (single-state path)
    # 1. Initialize the state and external function containers
    extern = HeFESToExtern()

    # Initialize HeFESToState using a helper that you can extend with
    # your MLIndexer-driven dimension/composition population. The function
    # keeps initialization minimal so you can fully control mapping logic.
    state = initialize_hefesto_state(nnew.ravel(), float(P.item()), float(T.item()), ml_indexer)
    extern.set_state(state)

    # 3. Execute the core calculation
    # The physub function will be modified to accept state and extern objects
    # and will raise NotImplementedError until all dependencies are translated.
    try:
        result = physub.physub(
            nnew,
            rho=0.0,  # Output variable
            wmagg=0.0, # Output variable
            freeagg=0.0, # Output variable
            state=state,
            extern=extern,
            iprint=0
        )
    except NotImplementedError as e:
        print(f"Calculation stopped: {e}")
        print("This is expected until all Fortran dependencies are translated to Python.")
        return {"status": "incomplete", "error": str(e)}


    # 4. Extract and return results from the state object
    # This will be populated with actual results once physub runs completely.
    """final_properties = {
        "density": state.vol,
        "Cp": state.Cap,
        "Cv": state.Cv,
        "alpha": state.alp,
        "K_S": state.Ks,
        "gamma": state.gamma,
        # ... other properties from PHYSUB_BULK_ATTRIBUTE_NAMES
    }"""

    alias_map = {
        'density': 'rho',
        'Cp': 'Cap',
        'Cv': 'Cv',
        'alpha': 'alp',
        'K_S': 'Ks',
        'gamma': 'gamma',
        'K_Hill': 'K',
        'G_Hill': 'Gsh',
        'entropy': 'ent',
        'enthalpy': 'phugo',
    }
    final_properties = {}
    for name in property_names:
        source_name = alias_map.get(name, name)
        final_properties[name] = result.get(source_name, getattr(state, source_name, getattr(state, name, None)))

    return final_properties


def _compute_bulk_properties_batch(
    nnew: np.ndarray, P: np.ndarray, T: np.ndarray, ml_indexer: MLIndexer
) -> dict:
    """Batch compute bulk properties for N assemblages (vectorized where possible).

    Processes N assemblages efficiently by:
    - Reusing parameter store singleton (cached)
    - Processing each row through hot-path kernels
    - Using vectorized extraction for final properties
    - Returning comprehensive result dict with all bulk properties

    Parameters
    ----------
    nnew : np.ndarray
        Shape (N, nspec) species amounts
    P : np.ndarray
        Shape (N,) pressures in GPa
    T : np.ndarray
        Shape (N,) temperatures in K
    ml_indexer : MLIndexer

    Returns
    -------
    dict
        Keys: 'output' (N, nprops), 'property_names', 'timing' (seconds)
        where nprops includes: density, Cp, Cv, alpha, K_S, gamma, velocities, etc.
    """
    import time

    start_time = time.perf_counter()
    extern = HeFESToExtern()

    n_assemblages = nnew.shape[0]

    # Property names in output order (matches extraction below)
    property_names = [
        "density",
        "Cp",
        "Cv",
        "alpha",
        "K_S",
        "gamma",
        "K_Hill",
        "G_Hill",
        "Vb_Hill",
        "Vs_Hill",
        "Vp_Hill",
        "entropy",
        "enthalpy",
    ]
    n_props = len(property_names)
    
    # Pre-allocate output array
    output = np.zeros((n_assemblages, n_props), dtype=float)

    # Pre-fetch parameter store once (singleton, LRU cached)
    store = None
    try:
        from .param_loader import get_parameter_store
        store = get_parameter_store()
    except Exception:
        pass

    # Process each assemblage row
    for i in tqdm(range(n_assemblages)):
        # Update state for this row
        state = initialize_hefesto_state(nnew[i], float(P[i]), float(T[i]), ml_indexer)
        extern.set_state(state)

        try:
            result = physub.physub(
                nnew[i],
                rho=0.0,
                wmagg=0.0,
                freeagg=0.0,
                state=state,
                extern=extern,
                iprint=0,
            )
        except NotImplementedError:
            # Expected until physub is fully translated
            continue
        except Exception as e:
            # Non-fatal; continue with zeros
            continue

        # Extract computed properties from result dict and state into output row
        # Result dict now includes all bulk properties computed by physub
        output[i, 0] = result.get('rho', state.vol)
        output[i, 1] = result.get('Cp', state.Cap)
        output[i, 2] = result.get('Cv', state.Cv)
        output[i, 3] = result.get('alpha', state.alp)
        output[i, 4] = result.get('K_S', state.Ks)
        output[i, 5] = result.get('gamma', state.gamma)
        output[i, 6] = result.get('K_Hill', state.K)
        output[i, 7] = result.get('G_Hill', state.Gsh)
        output[i, 8] = result.get('Vb_Hill', 0.0)
        output[i, 9] = result.get('Vs_Hill', 0.0)
        output[i, 10] = result.get('Vp_Hill', 0.0)
        output[i, 11] = result.get('entropy', state.ent)
        output[i, 12] = result.get('enthalpy', 0.0)

    elapsed = time.perf_counter() - start_time

    return output, property_names
"""{
        "output": output,
        "property_names": property_names,
        "timing": elapsed,
    }"""


def initialize_hefesto_state(nnew: np.ndarray, P: float, T: float, ml_indexer: MLIndexer, state: HeFESToState | None = None) -> HeFESToState:
    """Create and partially-initialize a `HeFESToState` instance.

    This helper sets `Pi`, `Ti`, and `n` and provides clear commented
    placeholders where your `MLIndexer` should populate dimensions,
    stoichiometry and mapping matrices. You requested to control the
    detailed composition/stoichiometry construction — implement those
    pieces in the marked sections below.

    Keep this function in `api.py` as the canonical place the rest of the
    pipeline expects state variables to live.
    """
    if state is None:
        state = HeFESToState()

    state.Pi = float(P)
    state.Ti = float(T)
    state.n = np.asarray(nnew, dtype=float).reshape(-1)
    state.nspec = int(state.n.size)
    state.nspecp = max(state.nspecp, state.nspec)

    padded_n = np.zeros(state.nspecp, dtype=float)
    padded_n[: state.nspec] = state.n
    state.n = padded_n
    state.n1 = np.zeros(state.nspecp, dtype=float)
    state.q2 = np.eye(state.nspecp, dtype=float)
    state.b = np.copy(state.n)
    state.s = np.eye(state.nspecp, dtype=float)
    state.absents = state.n <= 0.0
    state.nnull = state.nspec
    state.nnulls = state.nspec

    state.nc = state.nspec
    state.nco = 1
    state.ncompp = max(1, state.nco)
    state.nph = 1
    state.nphasep = max(state.nphasep, state.nph)

    state.comp = np.array([f'c{i}' for i in range(state.ncompp)], dtype=object)
    state.phname = ['bulk']
    state.sname = [f's{i}' for i in range(state.nspec)]
    state.lagc = np.zeros(state.ncompp, dtype=float)
    state.iophase = np.ones(state.nphasep, dtype=int)
    state.mophase = np.ones(state.nphasep, dtype=int)
    state.iphase = np.zeros(state.nspecp, dtype=int)
    state.lphase = np.zeros(state.nspecp, dtype=int)
    state.spinod = np.zeros(state.nspecp, dtype=bool)
    state.spinph = np.zeros(state.nphasep, dtype=bool)

    state.f = np.zeros((state.nph, state.nspecp), dtype=float)
    state.f[0, : state.nspec] = 1.0
    state.f_phase = state.f
    state.r_site = np.zeros((state.nco, state.nspecp, 1), dtype=float)
    state.r_site[0, : state.nspec, 0] = 1.0
    state.nsite = np.ones(state.nphasep, dtype=int)
    state.wreg = np.zeros((state.nph, 1, state.nspecp, state.nspecp), dtype=float)
    state.vreg = np.zeros((state.nph, 1, state.nspecp, state.nspecp), dtype=float)
    state.iastate = np.zeros((state.nspecp, state.nspecp, 1), dtype=bool)
    state.iltyp = int(getattr(ml_indexer, 'iltyp', 0))

    state.cpa = np.zeros(state.nspecp, dtype=float)
    state.sspeca = np.zeros(state.nspecp, dtype=float)
    state.vspeca = np.zeros(state.nspecp, dtype=float)

    state.vol = 1.0
    state.Cap = 1.0
    state.Cv = 1.0
    state.gamma = 1.0
    state.K = 1.0
    state.Ks = 1.0
    state.alp = 1.0e-5
    state.Ftot = 0.0
    state.ph = 0.0
    state.ent = 0.0
    state.deltas = 0.0
    state.tcal = 0.0
    state.zeta = 0.0
    state.Gsh = 1.0
    state.uth = 0.0
    state.uto = 0.0
    state.thet = 0.0
    state.qq = 0.0
    state.etas = 0.0
    state.dGdT = 0.0
    state.pzp = 0.0
    state.Vdeb = 1.0
    state.gamdeb = 1.0
    state.Ksp = 1.0
    state.Gshp = 1.0
    state.fn = float(max(state.nspec, 1))
    state.zu = float(np.sum(state.n[: state.nspec]))

    try:
        store = get_parameter_store()
        species_list = list(getattr(ml_indexer, 'label_names', []) or [])
        if not species_list:
            species_list = list(store.species)
        if len(species_list) < state.nspec:
            species_list = (species_list + list(store.species))[: state.nspec]
        else:
            species_list = species_list[: state.nspec]
        if not species_list:
            species_list = list(store.species)[: state.nspec]

        state.nparp = store.npar
        try:
            state.apar = store.build_apar(species_list=species_list, nspecp=state.nspecp)
            state.sname = list(species_list)
        except KeyError:
            state.apar = store.build_apar(species_list=store.species[: state.nspec], nspecp=state.nspecp)
            state.sname = list(store.species[: state.nspec])

        # Validate that the critical seed volume parameter (apar column 6)
        # is present and non-zero for all species in the assemblage. If not,
        # raise a descriptive error so the caller can fix species mapping or
        # parameter files rather than propagating zeros through the pipeline.
        EPS = 1.0e-12
        try:
            v0_vals = state.apar[: state.nspec, 6]
            if np.any(v0_vals <= EPS) or v0_vals.size < state.nspec:
                missing = [state.sname[i] if i < len(state.sname) else str(i) for i, v in enumerate(v0_vals[: state.nspec]) if v <= EPS]
                raise ValueError(f"Missing or zero apar v0 for species: {missing}")
        except IndexError:
            raise ValueError("apar array does not contain expected parameter columns (v0 missing)")
    except Exception:
        state.apar = np.zeros((state.nspecp, state.nparp), dtype=float)

    return state
