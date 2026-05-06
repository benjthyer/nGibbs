## Plan: Getting physub API Working for Testing & Debugging

### Phase 1: Dimension and Composition Initialization
**Goal**: Use MLIndexer to populate all dimension variables, composition matrices, and stoichiometry in HeFESToState.

**Task 1a**: Extract dimensions from MLIndexer
- Read: nspec, nph, nc, nco from ml_indexer
- Set: nspecp = nspec + padding_factor (typically ~10% for working arrays)
- Set: nphasep = nph + padding_factor
- Set: ncompp = nc + padding_factor
- Set: nparp = 60 (standard HeFESTo parameter count)
- Compute: nnull, nnulls from stoichiometry (nc - nspec degrees of freedom)

**Task 1b**: Populate composition arrays
- Map ml_indexer component labels → state.comp array
- Populate state.stcomp, state.wcomp (stoichiometric and molar masses)
- Set state.atom (atomic composition)

**Task 1c**: Populate stoichiometry and phase matrices
- Build state.s (ncompp × nspecp) stoichiometry matrix from ml_indexer
- Build state.f (nphasep × nspecp) phase membership matrix
- Compute null space basis q2 via SVD or QR decomposition
- Set state.n1, state.zervec

### Phase 2: Parameter Loading Pipeline
**Goal**: Load HeFESTo thermodynamic parameters for all species into state.apar.

**Task 2a**: Create parameter loader
- Read from HeFESTo_Parameters_010123/ directory for each species
- Use parse_hefesto_parameter_file() from param_state.py
- Build (nspecp × nparp) apar array with species parameters

**Task 2b**: Initialize species-level derivatives
- Compute state.sspeca (entropy derivatives w.r.t. composition)
- Compute state.vspeca (volume derivatives w.r.t. composition)
- These come from therm/volume branch computations at reference conditions

### Phase 3: State Initialization and Null Space Projection
**Goal**: Set up all intermediate state variables and project initial composition into null space.

**Task 3a**: Initialize working arrays
- state.n ← nnew (species amounts, user-provided)
- state.b ← stoichiometric projection of nnew via s matrix
- state.absents ← identify species with zero amounts
- state.lagc ← initialize Lagrange multipliers (zeros initially)

**Task 3b**: Set up BLAS/LAPACK constants
- state.one = 1.0, state.zero = 0.0, state.ione = 1
- state.zervec = [0.0] * nnull (for null space padding)

### Phase 4: Create Unified API Wrapper
**Goal**: Package the initialization sequence into a high-level API function.

```python
def initialize_hefesto_state(
    nnew: np.ndarray,
    P: float,
    T: float,
    ml_indexer: MLIndexer
) -> HeFESToState:
    """Initialize complete HeFESToState from equilibrium assemblage and conditions."""
    state = HeFESToState()
    
    # Set P, T
    state.Pi = P
    state.Ti = T
    
    # Populate dimensions (Task 1a)
    # Populate compositions (Task 1b)
    # Populate stoichiometry (Task 1c)
    
    # Load parameters (Task 2a, 2b)
    
    # Initialize working arrays (Task 3a, 3b)
    
    return state
```

### Phase 5: Validation Harness
**Goal**: Build smoke tests to verify state initialization before physub execution.

**Test 5a**: Dimension consistency
- Check nspec ≤ nspecp, nph ≤ nphasep, etc.
- Verify matrix shapes (s, f, q2)

**Test 5b**: Stoichiometry validity
- Verify s · n = b (composition conservation)
- Verify q2 is orthonormal and spans null space

**Test 5c**: Parameter bounds
- Check apar values are within physical ranges (T > 0, V > 0, K > 0, etc.)

### Phase 6: Debugging Infrastructure
**Goal**: Instrument physub for detailed logging and error tracking.

**Task 6a**: Add debug output
- Log state before/after each major physub block
- Track intermediate property computations (K, G, alpha, etc.)
- Record species-level data (vol, Ftot, ent for each ispec)

**Task 6b**: Comparison vs Fortran reference
- Extract benchmark data from Fortran fort.* files (fort.56, fort.99)
- Compute per-species and aggregate property errors
- Profile execution time by subfunction (gspec, volume, therm, cp, etc.)

### Phase 7: Performance Testing
**Goal**: Measure and optimize physub execution.

**Task 7a**: Micro-benchmarks
- Time individual species gspec calls
- Profile vectorizable loops (species loop, phase loop)
- Identify BLAS/LAPACK bottlenecks

**Task 7b**: Full-assemblage benchmarks
- Time end-to-end physub for realistic 5–10 phase assemblages
- Compare Python vs Fortran wall-clock time
- Profile memory usage

### Ordering & Dependencies
1. Phase 1 (Initialization) → Requires MLIndexer API study
2. Phase 2 (Parameters) → Requires understanding HeFESTo file format
3. Phase 3 (State) → Depends on Phase 1 & 2
4. Phase 4 (API) → Depends on Phase 3
5. Phase 5 (Validation) → Can begin parallel to Phase 4
6. Phase 6 (Debugging) → Enables Phase 7
7. Phase 7 (Performance) → Final refinement

### Key Unknowns to Resolve
- Which ml_indexer methods/attributes provide dimensions, stoichiometry, and species labels?
- What is the expected nspecp padding strategy?
- Are sspeca, vspeca provided by ml_indexer or computed on-the-fly?
- Where are HeFESTo parameter files stored relative to workspace?
- What is the fallback behavior for absent species or missing parameters?
