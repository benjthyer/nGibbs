# HeFESTo Vectorized Python Reimplementation — Technical Analysis

**Date**: 2026-05-23  
**Purpose**: Planning document for a vectorized NumPy reimplementation of the HeFESTo EOS routine that matches Fortran output to numerical precision, targeting ≥2^15 assemblages in <10 seconds.  
**Scope**: Density, P-wave velocity, S-wave velocity only. Elastic moduli only (no anelastic corrections). Fixed phase compositions as inputs (no Gibbs minimization).

---

## 1. EOS Computation Path

### Call Chain: (P, T, X) → (ρ, Vp, Vs)

The relevant path through the Fortran (rooted in `physub.f`, called from `main.f` after equilibration) is:

```
physub(nnew, rho, wmagg, freeagg, iprint=1)           [physub.f]
  For each phase (iph=1..nph):
    For each species (ispec in phase):
      parset(ispec, apar, ...)                          [parset.f]
        → extracts scalar params from apar array
      cp(ispec, n, chempot, ...)                        [cp.f]
        → chemical potential + configurational entropy (not needed for fixed-X)
      gspec(ispec)                                      [gspec.f]
        → calls volume() then therm() then Ftotsub()
        → populates /prop/ COMMON block

        volume(ispec, x1)                               [volume.f]
          → cage()      — bracket the root
          → zeroin()    — Brent's method: finds V s.t. pressure(V) = Pi
              pressure(V)   [pressure.f]
                = pc(BM) + ph(MG thermal) + pa(anharmonic) + pel(electronic) + pzp(zero-point)
          Falls back to nlmin_V() → cages() for spinodal/convergence failures

        therm(ispec, vol, volnl, ...)                   [therm.f]
          → gamset()     — computes γ(V), etas(V)
          → Etherm(), Ctherm(), Ftherm(), Ztherm()     [Etherm.f, Ctherm.f, Ftherm.f, Ztherm.f]
              → table lookups in dos.inc for each vibrational mode
          → thetacal()   — effective Debye temperature
          → landauqr() or landau()   — Landau transition corrections
          → Assembles: Kc (cold BM), Kth (thermal), K = Kc + Kth + Kel
          → Ks = K*(1 + α*γ*T)    [adiabatic]
          → Gsh from finite-strain + thermal etas correction
          → alp (thermal expansivity), Cp, Cv, deltas

      Ftotsub(ispec, volnl, Ftot)                       [Ftotsub.f]
          → Gibbs free energy (needed only for Gibbs minimization, not for fixed-X path)

    [Per-phase Reuss average of K, G:]
      1/K_reuss = Σ_i  n_i * V_i / Ks_i
      1/G_reuss = Σ_i  n_i * V_i / Gsh_i

  [Multi-phase Voigt, Reuss, Hill, Hashin-Shtrikman averaging:]
    K_voigt = Σ_ph  (V_ph/V_total) * K_reuss_ph
    K_reuss_agg = V_total / Σ_ph (V_ph / K_reuss_ph)
    K_Hill = (K_voigt + K_reuss_agg) / 2    ← used for fort.56 output
    ρ = W_total / V_total
    Vb = sqrt(K_Hill / ρ)
    Vs = sqrt(G_Hill / ρ)
    Vp = sqrt((K_Hill + 4/3*G_Hill) / ρ)
```

### Key Intermediate Quantities

| Symbol | Meaning | Units |
|--------|---------|-------|
| `V` | Molar volume at (P,T) for species i | cm³/mol |
| `f` | Eulerian strain: `0.5*((V/V₀)^(-2/3) - 1)` | dimensionless |
| `γ` | Volume-dependent Grüneisen parameter | dimensionless |
| `q`, `etas` | Logarithmic volume derivatives of γ and G | dimensionless |
| `Uth`, `Uto` | Thermal internal energy at T and T₀ | J/mol |
| `Cv`, `Cvo` | Isochoric heat capacity at T and T₀ | J/(mol·K) |
| `Kc` | Cold (BM) bulk modulus | GPa |
| `Kth` | Thermal correction to bulk modulus | GPa |
| `K` | Isothermal bulk modulus = Kc + Kth + Kel | GPa |
| `Ks` | Adiabatic bulk modulus = K*(1 + α*γ*T) | GPa |
| `Gsh` | Shear modulus (finite-strain + thermal) | GPa |
| `ph` | Mie-Grüneisen thermal pressure | GPa |
| `alp` | Thermal expansivity | 1/K |
| `deltas` | Anderson-Grüneisen δS parameter | dimensionless |

---

## 2. Thermal EOS Details

### Vibrational Model

HeFESTo uses the **Mie-Grüneisen** framework with a generalized VDOS parameterized as a weighted sum of four mode types (not a simple single-Debye model):

| `idos` | Mode Type | Parameters | Units in file |
|--------|-----------|------------|---------------|
| 1 | Debye acoustic | `wd1, wd2, wd3` | **Kelvin** (stored directly) |
| 3 | Sinusoidal acoustic | `ws1, ws2, ws3` | **cm⁻¹** → converted to K via `×hcok` |
| 2 | Einstein oscillators (×4) | `we1-we4`, weights `qe1-qe4` | **cm⁻¹** → converted to K |
| 4 | Optic continuum | `wou, wol` (upper/lower) | **cm⁻¹** → converted to K |

The mode weight normalization is:
- `su = fn * zu` = total atoms per formula unit × formula units per cell = 3N vibrational DOF
- Acoustic fraction = `1/su` (per mode); Einstein weight = `qe = Σ qei`; Optic weight = `qo = 1 - 1/su - qe`

**Critical conversion**: `hcok = 1.438775 K·cm` (= h*c/k_B). Applied in `parset.f` to ws, we, wou, wol before use. Must be applied in Python when reading from `apar` array.

### Volume Dependence (Grüneisen Scaling)

Controlled by `ityp` global flag (from `control` file). **Benchmark uses `ityp=1`**.

- **`ityp=1`** (default/benchmark): Full finite-strain quadratic:  
  `f = 0.5*((V/V₀)^(-2/3) - 1)`  
  `a = 6*γ₀`, `b = γ₀*(36*γ₀ - 18*q₀ - 12)`  
  `θ(V) = θ₀ * sqrt(1 + a*f + 0.5*b*f²)`  
  γ(V) derived analytically from d ln θ / d ln V.

- **`ityp=2`**: Linear FS: `θ(V) = θ₀ * (1 + a*f + 0.5*b*f²)`

- **`ityp=3`**: Classic MGD: `q=q₀=const`, `γ = γ₀*(V/V₀)^q₀`, `θ = θ₀*exp((γ₀-γ)/q₀)`

### Volume Solver

At each (P, T, species), solve for V such that:

```
P_total(V) - P_target = 0

P_total(V) = pc(V) + ph(V,T) + pa(V,T) + pel(V,T) + pzp(V)

pc  = cold BM pressure (3rd-order BM unless K'' ≠ 0)
ph  = 0.001 * (γ/V) * (Uth(T) - Uth(T₀))     [Mie-Grüneisen, T₀=300 K]
pa  = 0.001 * 3*fn*R * anh * (T² - T₀²) / V  [anharmonic; usually 0]
pel = 0.001 * 0.5*ge*be * (V/V₀)^ge * (T² - T₀²) / V  [electronic]
pzp = izp * 0.001 * (9/8)*fn*R*wd1*γ/V        [zero-point; izp=0 for mantle minerals]
```

Root-finding sequence: `cage()` → `zeroin()` (Brent's method). Fallback: `nlmin_V()` → `cages()`.  
Volume bounds from `apar(51-54)` (spinodal and frequency positivity constraints).

---

## 3. Elastic Moduli Calculation

### Cold Bulk Modulus (3rd-order Birch-Murnaghan)

```
f = 0.5 * ((V/V₀)^(-2/3) - 1)
Kc = K₀ * (1+2f)^2.5 * [1 + (7 + a₃)*f + (4.5*a₃ + 0.5*a₄)*f² + ...]
  where a₃ = 3*(K₀' - 4),  a₄ = 9*(K₀'' + K₀'*(K₀'-7) + 143/9)
```
Vinet (ibv=1) and Lagrangian (ibv=2) variants also supported.

### Thermal Correction to K

```
Kth = (γ + 1 - q) * (ph + pzp) - 0.001 * (γ²/V) * (T*Cv - T₀*Cvo)
K = Kc + Kth + Kel
```

### Adiabatic Bulk Modulus

```
agT = alp * γ * T
Ks  = K * (1 + agT)
```

### Shear Modulus (ivtyp=1, ittyp=1 — benchmark settings)

```
# Cold shear (finite-strain BM expansion):
b₀ = G₀
b₁ = 3*K₀*G₀' - 5*G₀
b₂ = 6*K₀*G₀' - 24*K₀ - 14*G₀ + 4.5*K₀*K₀'
Gsh_cold = (1+2f)^2.5 * (b₀ + b₁*f + b₂*f²)

# Pressure derivative:
Gshp = (1/(3*K)) * (5*Gsh_cold + (1+2f)^3.5 * (b₁ + 2*b₂*f))

# Thermal correction (etas-based):
Gsh = Gsh_cold - 0.001 * etas/V * (Uth - Uto + Ezp)

# Temperature derivative (for α, Cp calculations):
dGdT = -0.001*etas*Cv/V - alp*K*Gshp
```

`etas` comes from `gamset()` and depends on `Got` = `η_S₀` = apar(37) (NOT apar(43)).

### Landau Corrections (iltyp=1 — benchmark setting)

Applied to phases where `apar(38)` = Tc > 0 (quartz, coesite, stishovite, some pyroxenes).  
`landauqr.f` formulation (Q-R transition):  
- Transition temperature: `Tc = Tc₀ + Vmax/(0.001*Smax)*P`  
- Order parameter: `Q⁴ = (Tc - T)/Tc₀`, capped at `Qmax=1.5`  
- Adds corrections to F, S, V, Cp, β

For most major mantle phases (olivine, garnet, Mg-perovskite), Tc=0 and Landau corrections are zero.

---

## 4. Wave Velocity Computation

### Single-Phase Velocities (physub.f, per phase)

```
ρ_ph  = W_ph / V_ph           [g/cm³, with W in g/mol, V in cm³/mol]
Vb_ph = sqrt(K_reuss_ph / ρ_ph)   [km/s, if K in GPa and ρ in g/cm³]
Vs_ph = sqrt(G_reuss_ph / ρ_ph)
Vp_ph = sqrt((K_reuss_ph + 4/3*G_reuss_ph) / ρ_ph)
```

Note: 1 GPa / (g/cm³) = 1 km²/s², so sqrt gives km/s directly. ✓

### Multi-Phase Averaging

The benchmark output (fort.56) uses **Voigt-Reuss-Hill (VRH) Hill averages** for Ks and G:

```
K_voigt = Σ_ph (V_ph/V_total) * K_ph       [stiffness-weighted]
K_reuss = V_total / Σ_ph (V_ph/K_ph)       [compliance-weighted]
K_Hill  = (K_voigt + K_reuss) / 2

G_voigt = Σ_ph (V_ph/V_total) * G_ph
G_reuss = V_total / Σ_ph (V_ph/G_ph)
G_Hill  = (G_voigt + G_reuss) / 2

ρ = W_total / V_total

Vb = sqrt(K_Hill / ρ)
Vs = sqrt(G_Hill / ρ)
Vp = sqrt((K_Hill + 4/3*G_Hill) / ρ)
```

Hashin-Shtrikman bounds are also computed and written to fort.58, but are **not** used for the primary fort.56 output.

### Anelastic Reduction (to EXCLUDE from elastic-only computation)

Applied only to output columns, via `qr19()` and `vred()`:  
`VSQ = Vs * vsred`, `VPQ = Vp * vpred`  
The elastic Vs, Vp columns in fort.56 are unaffected. Simply omit these calls.

---

## 5. Parameter File Structure (apar array, 1-indexed in Fortran, 0-indexed in Python)

| Fortran Index | Python Index | Parameter | Units | Notes |
|--------------|-------------|-----------|-------|-------|
| 1 | 0 | fn | – | Atoms per formula unit |
| 2 | 1 | zu | – | Formula units per unit cell (Z) |
| 3 | 2 | wm | g/mol | Formula mass |
| 4 | 3 | To | K | Reference temperature (typically 300 K) |
| 5 | 4 | Fo | kJ/mol | Reference Helmholtz free energy |
| 6 | 5 | Vo | cm³/mol | Reference molar volume |
| 7 | 6 | Ko | GPa | Reference bulk modulus |
| 8 | 7 | K₀' | – | Pressure derivative of K |
| 9 | 8 | K₀'' | GPa⁻¹ | Second pressure derivative (0 = pure 3rd-order BM) |
| 10 | 9 | wd1 | **K** | Debye temperature branch 1 (stored in Kelvin) |
| 11 | 10 | wd2 | **K** | Debye branch 2 |
| 12 | 11 | wd3 | **K** | Debye branch 3 |
| 13 | 12 | ws1 | **cm⁻¹** | Sinusoidal branch 1 → multiply by `hcok` |
| 14 | 13 | ws2 | cm⁻¹ | Sinusoidal branch 2 |
| 15 | 14 | ws3 | cm⁻¹ | Sinusoidal branch 3 |
| 16 | 15 | we1 | **cm⁻¹** | Einstein oscillator 1 → multiply by `hcok` |
| 17 | 16 | qe1 | – | Weight of Einstein oscillator 1 |
| 18 | 17 | we2 | cm⁻¹ | Einstein oscillator 2 |
| 19 | 18 | qe2 | – | Weight of Einstein oscillator 2 |
| 20 | 19 | we3 | cm⁻¹ | Einstein oscillator 3 |
| 21 | 20 | qe3 | – | Weight of Einstein oscillator 3 |
| 22 | 21 | we4 | cm⁻¹ | Einstein oscillator 4 |
| 23 | 22 | qe4 | – | Weight of Einstein oscillator 4 |
| 24 | 23 | wou | **cm⁻¹** | Optic continuum upper limit → multiply by `hcok` |
| 25 | 24 | wol | **cm⁻¹** | Optic continuum lower limit → multiply by `hcok` |
| 26 | 25 | γ₀ | – | Reference Grüneisen parameter |
| 27 | 26 | q₀ | – | d ln γ / d ln V at reference |
| 28 | 27 | be | J/(mol·K²) | Electronic Cv coefficient (0 for most mantle phases) |
| 29 | 28 | ge | – | Electronic volume-scaling exponent |
| 30 | 29 | anh (q₂A₂) | K⁻¹ | Anharmonic correction coefficient |
| 31 | 30 | htl | flag | EOS type: 0=solid BM, 1=liquid, 3=water, 4=ideal gas, 5=H-EOS |
| 32 | 31 | ibv | flag | BM(0), Vinet(1), Lagrange(2) |
| 33 | 32 | ied | flag | Integrand: Einstein(0), Debye(1) |
| 34 | 33 | izp | flag | Zero-point pressure: ±1=yes, 0=no |
| 35 | 34 | G₀ | GPa | Reference shear modulus |
| 36 | 35 | G₀' | – | Pressure derivative of G |
| 37 | 36 | Got (η_S₀) | – | Thermal derivative parameter: -d ln G / d ln V |
| 38 | 37 | Tc | K | Landau critical temperature at P=0 (0 = no transition) |
| 39 | 38 | Smax | J/(mol·K) | Landau maximum entropy |
| 40 | 39 | Vmax | cm³/mol | Landau maximum volume change |
| 41 | 40 | VanLaar | – | Van Laar size parameter for mixing |
| 42 | 41 | C₁₂' | – | (unused in current branches) |
| 43 | 42 | a₅ (C₄₄') | – | 4th-order BM correction (typically 0) |

**Critical unit facts**:
- `hcok = 1.438775 K·cm` converts cm⁻¹ → Kelvin  
- `Rgas = 8.314472 J/(mol·K)`  
- Pressures throughout: GPa  
- Volumes: cm³/mol  
- Energies: J/mol (except Fo in kJ/mol: multiply by 1000 on read)  
- Thermal pressure requires factor 0.001: (J/mol) / (cm³/mol) = J/cm³ = 10 GPa → ×0.001 gives GPa  
- Cold BM energy: factor 4500 = 4.5 × 1000 (GPa·cm³ → J/mol)

---

## 6. Existing Python Translation Errors

These are confirmed bugs requiring correction before any benchmarking is possible. Listed roughly in order of severity.

### Bug 1 (CRITICAL): γ₀, q₀ Read from Wrong apar Indices

**Files**: `volume.py`, `Ftotsub.py`, any caller of `gamset()`  
**Fortran**: `gam = apar(ispec,26)`, `qo = apar(ispec,27)`  
**Python**: reads indices 38 and 39 (= Tc and Smax — both 0 for most mantle minerals)  
**Effect**: Every mineral without a Landau transition gets γ=0, q=0, completely eliminating thermal pressure and making volume-finding wrong.

### Bug 2 (CRITICAL): be, ge, anh Read from Wrong apar Indices

**Files**: `volume.py`, `therm.py`  
**Fortran**: `be=apar(28)`, `ge=apar(29)`, `anh=apar(30)`  
**Python**: reads indices 35, 36, 37 (= G₀, G₀', Got — shear modulus parameters)  
**Effect**: Electronic and anharmonic contributions are computed from shear modulus parameters, giving nonsense corrections.

### Bug 3 (CRITICAL): Got vs C44' Confusion — Thermal Shear Correction Eliminated

**File**: `Ftotsub.py` line ~52-53  
**Fortran**: `Got = apar(ispec,37)` (typically 1–3 for mantle minerals)  
**Python**: `got = a5 = apar(43)` (= 4th-order BM coefficient, typically 0)  
**Effect**: `etas=0` everywhere, eliminating the entire thermal correction to the shear modulus (and the dominant temperature-dependence of Vs).

### Bug 4 (CRITICAL): wd1 Read from Wrong apar Index

**File**: `therm.py` (`_thermal_modes_from_apar`)  
**Fortran**: `wd1 = apar(ispec,10)` (Debye temperature in Kelvin, typically ~800 K)  
**Python**: reads `apar(16)` = `we1` (Einstein oscillator 1, often 0; in cm⁻¹, not K)  
**Effect**: Primary acoustic Debye temperature is wrong for all species.

### Bug 5 (CRITICAL): All Secondary Vibrational Modes Zeroed in therm.py

**File**: `therm.py` (`_thermal_modes_from_apar`)  
Hard-zeros wd2, wd3, ws1-ws3, we1-we4, qe1-qe4, wou, wol. Only one Debye branch used. All Sin, Einstein, and optic modes are ignored.  
**Effect**: Thermal energy, heat capacity, and Grüneisen pressure are wrong for all species.

### Bug 6 (CRITICAL): cm⁻¹ → Kelvin Conversion Missing

**Files**: `therm.py`, `volume.py`, anywhere that reads ws1-ws3, we1-we4, wou, wol from apar  
**Fortran**: `parset.f` multiplies these by `hcok=1.438775` before use  
**Python**: reads raw file values without conversion — off by factor ~1440  
**Effect**: All sinusoidal and Einstein mode frequencies (used in thermal kernel x=θ/T) are ~1440× too small.

### Bug 7 (CRITICAL): Gsh Set to Zero Everywhere

**File**: `therm.py`  
Replaces the BM + thermal etas shear modulus calculation with: `gsh = max(apar_value(apar, ispec, 44, 0.0), 0.0)`.  
apar index 44 does not exist (only 43 params). Default 0.0 is returned.  
**Effect**: Gsh=0 for all species → Vs=0 everywhere.

### Bug 8: zu Hardcoded to 1.0

**Files**: `Ftotsub.py`, `volume.py` (thermal kernel calls)  
**Fortran**: `zu = apar(ispec,2)` (e.g., 4 for olivine, 4 for Mg-perovskite, 2 for periclase)  
**Effect**: Thermal energy and Cv wrong by factor zu (typically 2–4).

### Bug 9: gamset.py Defaults to ityp=3, Benchmark Uses ityp=1

**File**: `gamset.py`  
Most callers don't pass `ityp`, so it defaults to 3 (classic MGD with q=const).  
The benchmark control specifies `ityp=1` (full finite-strain quadratic theory).  
**Effect**: Grüneisen parameter wrong at all V ≠ V₀.

### Bug 10: deltas Not Computed

**File**: `therm.py`  
Returns `deltas=0.0`. Correct formula:  
`deltas = (Ksp + q - gamma - 1) / (1 + agT)`  
**Effect**: Downstream calculations using deltas (aggregate compressibility) are wrong.

### Bug 11: Hessian / Metamorphic Contributions Block-Commented Out

**File**: `physub.py` lines 165-215  
The dndt, dndp block (phase-transition reaction derivatives) is inside a triple-quoted string and never executes. For **fixed compositions** (the target use case), these terms are identically zero anyway, so this bug does not affect the fixed-X path.

### Bug 12: Ks and K Swapped in physub.py State Update

**File**: `physub.py` near end  
`s.K = baggh` (Hill KT) is correct, but `s.Ks = btaggh` assigns the isothermal Hill K to the adiabatic slot.

### Bug 13: Volume Array Index Shifts for Sinusoidal Modes

**File**: `volume.py`  
`ws1o = apar_value(apar, ispec, 19)` — index 19 is `qe2` (Einstein weight).  
Correct index for ws1 is 13 (Fortran 14, 0-indexed 13).  
The ws1-ws3 sequence is shifted by 6 positions.

### Bug 14: thetacal.py Uses Bisection Instead of Table Lookup

**File**: `thetacal.py`  
Fortran uses pre-computed Debye function table (`dos.inc`) with Neville interpolation. Python uses bisection on `Heat()`.  
**Effect**: Functionally correct but ~10× slower and may produce minor numerical differences at table boundaries.

### Bug 15: landau.py Uses Wrong Q Formula

**File**: `landau.py`  
Uses `(Ti - Tc)*q² + 1/3*Tco*q⁶`. The `landauqr.f` variant (used with `iltyp=1`) has Q⁴ = (Tc-T)/Tc₀ (denominator differs) and different sign conventions.

### Bug 16: vector_EOS.py Implements SLB EOS, Not HeFESTo

**File**: `HeFESTo_Python_Fortranslation/vector_EOS.py`  
Implements the Stixrude-Lithgow-Bertelloni (SLB) parameterization using BurnMan's Debye module and SLB VDOS model. This is **not** the HeFESTo parameterization. HeFESTo uses the multi-mode VDOS described in Section 2; SLB uses a single Debye model with different Grüneisen scaling. Numerically different even for minerals with identical parameters. **This file should not be used for the HeFESTo-matching implementation.**

---

## 7. Vectorization Strategy

### Input/Output Layout

```python
# Inputs:
P_bar   : (B,)        # Pressure in bar → convert to GPa: / 10000
T_K     : (B,)        # Temperature in Kelvin
X       : (B, nspec)  # Mole amounts / phase abundances of each species

# Pre-loaded constants (shape (nspec, 43), loaded once at startup):
params  : (nspec, 43) # Full apar matrix, 0-indexed, post-converted (cm⁻¹ → K applied)

# Working array (to be solved):
V       : (B, nspec)  # Molar volume at (P,T) for each species

# Outputs:
rho     : (B,)        # Density, g/cm³
Vp      : (B,)        # P-wave velocity, km/s
Vs      : (B,)        # S-wave velocity, km/s
```

### Embarrassingly Parallel Operations (vectorize with broadcasting)

All of these can be batched across the `(B, nspec)` grid simultaneously:

1. **gamset()**: Given `(V, V₀, γ₀, q₀, Got)` all as `(B, nspec)` arrays — pure arithmetic, no loops needed.

2. **Thermal kernel lookups**: Pre-load dos.inc tables as 1D numpy arrays at module init. Replace Neville interpolation with `np.interp()`. For each mode, compute `x = θ(V)/T` as a `(B, nspec)` array and call `np.interp` once → `(B, nspec)` output.

3. **Etherm, Ctherm, Ftherm, Ztherm**: These are weighted sums over modes. With all modes as `(nspec, nmodes)` arrays and T as `(B, 1)`, compute all `x = θ/T` values as `(B, nspec, nmodes)` in one step, apply table lookup, weight and sum.

4. **Cold pressure and moduli (BM)**: `f = 0.5*((V/V₀)^(-2/3) - 1)` as `(B, nspec)` → all polynomial evaluations vectorized.

5. **Landau corrections**: Compute for all species; mask to zero where `Tc == 0` using `np.where`.

6. **Phase averaging (Voigt, Reuss, Hill)**: Simple weighted sums along the species axis, using `X` (phase abundances) as weights.

### Volume Solver (the Bottleneck)

Each species at each condition requires solving `P_total(V, P_target, T) = 0` independently. For 2^15 assemblages × ~10 species = ~330K solves.

**Recommended approach — Batched Newton-Raphson:**

```python
def solve_volume_batch(P_target, T, params):
    # params: (nspec, 43); P_target: (B,); T: (B,)
    P_GPa = P_target[:, None] / 10000.0   # (B, nspec)
    T_arr  = T[:, None]                    # (B, nspec)

    # Initial guess: Murnaghan approximation
    V0 = params[None, :, 5]               # (1, nspec)
    K0 = params[None, :, 6]               # (1, nspec)
    Kp = params[None, :, 7]               # (1, nspec)
    V  = V0 * (1 + P_GPa * Kp / K0) ** (-1.0/Kp)  # (B, nspec)

    # Newton-Raphson iterations
    for _ in range(10):
        F  = P_total(V, P_GPa, T_arr, params)   # (B, nspec)
        dF = dPdV(V, P_GPa, T_arr, params)      # (B, nspec), analytical dP/dV
        dV = -F / dF
        dV = np.clip(dV, -0.3*V, 0.3*V)         # clamp step
        V  = V + dV
        V  = np.clip(V, V_low[None,:], V_high[None,:])

    return V
```

For ~330K conditions × 10 Newton iterations × ~20 float ops per iteration ≈ 66M FLOPs — well within <1 second on modern CPU with numpy.

For robustness, flag non-converged conditions (`abs(F) > tol`) and handle with scalar `scipy.optimize.brentq` as fallback.

### Phase Averaging Vectorization

```python
# V_phase: (B, nph)  mass-weighted phase volumes
# K_ph:    (B, nph)  Reuss-averaged K per phase
# G_ph:    (B, nph)  Reuss-averaged G per phase

# Voigt:
K_voigt = np.sum((V_phase / V_total[:, None]) * K_ph, axis=1)   # (B,)
G_voigt = np.sum((V_phase / V_total[:, None]) * G_ph, axis=1)

# Reuss (harmonic mean):
K_reuss = V_total / np.sum(V_phase / K_ph, axis=1)
G_reuss = V_total / np.sum(V_phase / G_ph, axis=1)

# Hill:
K_Hill = 0.5 * (K_voigt + K_reuss)
G_Hill = 0.5 * (G_voigt + G_reuss)
```

### Data Dependencies (Sequential Within Each Condition)

1. gamset(V, params) → γ, etas  
2. Etherm, Ctherm (need γ from step 1 for mode scaling) → Uth, Cv  
3. P_total (needs Uth from step 2) → residual for Newton  
4. converged V → therm() → K, Ks, G, alp  
5. phase averaging → ρ, Vp, Vs

Steps 1-4 iterate together in the Newton loop. Step 5 is post-processing.

---

## 8. Inelastic Contributions (Excluded from Target Implementation)

These terms are present in the Fortran but should be **excluded** for the elastic-only, fixed-composition implementation:

| Term | Location | Dependency | Why Excludable |
|------|----------|------------|----------------|
| `qr19()`, `vred()` | physub.f near output | PREM Q model | Only applied to output columns VSQ, VPQ — elastic Vs, Vp unchanged |
| `hessfunc()`, `hessian()`, `svdsub()` | physub.f, hessian.f | Gibbs Hessian matrix | Only needed for dndt, dndp (reaction derivatives). Zero for fixed X. |
| `alpmet`, `cpmet`, `bmet` | physub.f | dndt, dndp | Metamorphic contributions to α, Cp, K — zero for fixed X |
| Inelastic Cv term | therm.f | Hessian | Not applied to elastic moduli |

**Notes on Ztherm and deltas**: These are small corrections to Ks through the pressure derivative of the thermal K. They should be **retained** in the full implementation for accuracy, even though they are not strictly "inelastic". They can be approximated as zero initially for debugging.

**To mark clearly in code**: Add comments like:
```python
# INELASTIC TERM (omitted): Fortran adds alpmet contribution here via Hessian.
# Requires: hessfunc.f, svdsub.f, dndt/dndp arrays.
# Include if implementing full equilibrium path.
```

---

## 9. Open Questions & Uncertainties

### Unit Ambiguities

1. **Pressure input in fort.56**: The benchmark P column appears to be in GPa, but HeFESTo internally uses bar (1 GPa = 10000 bar). The `depth.f` function converts Pi (in bar) → depth in km. Need to confirm P units in `control` file and fort.56 output.

2. **`Fo` (apar 4 = index 4)**: Stored in the parameter file — is this in kJ/mol or J/mol? Fortran `readin.f` likely scales by 1000. Check `parset.f` for the read statement.

3. **`4500` factor in cold BM energy** (`Ftotsub.f`): `Fbm = 4500.*Ko*Vo*f*f*(...)`. Factor = 4.5 × 1000, converting GPa·cm³/mol to J/mol. Likely correct but should be verified against dimensional analysis.

4. **`0.001` factor throughout thermal pressure**: Converts J/cm³ → GPa (since 1 J/cm³ = 1 kJ/L = 1 GPa). Should be consistent throughout but easy to misapply.

5. **Pressure in `ph` formula**: `ph = 0.001*(γ/V)*(Uth-Uto)`. Units: (J/mol) / (cm³/mol) × 0.001 = J/cm³ × 0.001 = GPa. ✓

### Variables with Uncertain Meaning

1. **`zeta` in therm.f**: Returned by `Ztherm()` — the mode-weighted logarithmic derivative of Cv with respect to mode frequencies. Used to compute `Kpth` (pressure derivative of Kth). Physical meaning: `ζ = (V/Cv) * (∂Cv/∂V)|_T = d ln Cv / d ln V`. Small correction to Ks.

2. **`etas` vs `Got`**: `Got` = apar(37) = reference `η_S₀ = -d ln G / d ln V` at (V₀, T₀). `etas` is its volume-dependent value: `etas = -γ - 0.5*(some FS polynomial)`. **The connection between `etas` and `Got` via `gamset.f` must be carefully verified** — the formula differs between ityp modes.

3. **`fn*zu` (= `su`)**: Total number of vibrational DOF per formula unit = 3N (where N = atoms per formula unit × Z). Appears in VDOS weight normalization. Must be computed from `apar(0)*apar(1)`.

4. **`q` in Kth formula**: The `q` in `Kth = (γ + 1 - q)*(ph+pzp) - ...` is the **volume logarithmic derivative of γ** at current volume, **not** the reference q₀. This is the `q` output of `gamset()`.

5. **`volnl` vs `vol`**: `volume.f` returns both `vol` and `volnl`. The former appears to be in cm³/mol normalized to formula units, the latter to the unit cell. Need to verify which is passed to `therm.f` for which computation.

6. **`Ezp` in Gsh thermal correction**: `Gsh = Gsh_cold - 0.001*etas/V*(Uth - Uto + Ezp)`. The `Ezp` term is the zero-point energy correction. For mantle minerals (izp=0), Ezp=0. Confirm this is the case for all target phases.

### Fortran-Python Disagreements Without Obvious Reason

1. **`zu=1.0` hardcoded in Ftotsub.py**: Fortran uses actual `zu = apar(2)`. No explanation for why Python forces zu=1. Possibly a simplification that was forgotten to be fixed.

2. **gamset.py does not read `ityp` from state**: gamset's `ityp` parameter determines the entire Grüneisen parameterization. The state object in `param_state.py` does have an `ityp` field, but none of the thermal callers pass it to gamset. Likely an oversight.

3. **`wm` vs `wmagg`**: physub.f accumulates `wmagg` (aggregate formula weight) by summing over species contributions. Python physub.py uses a state attribute that may not accumulate correctly if species ordering differs.

4. **Pressure units in volume.py**: Fortran converts input `Pi` (in bar) by dividing by 10000 to get GPa before pressure-matching. Python volume.py must do the same. Verify this conversion is present and not applied twice.

### Fortran Routines Without Python Counterparts

| Fortran Routine | Purpose | Required for Elastic Path? |
|-----------------|---------|---------------------------|
| `tlindeman.f` | Lindemann melting temperature | No |
| `icebcc.f` | Ice VII-X special EOS | No (unless ice in system) |
| `stishtran.f` | Stishovite softening near transition | Yes for stishovite-bearing systems |
| `aliqset.f` | Liquid EOS parameter setup | Only for melts |
| `depth.f` | PREM P→depth lookup | No (only for output labeling) |
| `svdsub.f` | SVD solver for Hessian system | No (fixed-X path) |
| `spinrem.f` | Spin transition (Fe) | **Possibly yes** — check if any target phases use it |
| `Tspin.f` | Fe spin crossover contribution | **Possibly yes** for Fe-bearing phases |
| `thermlel.f` | Linear elastic limit for liquid | No |
| `eoswater.f` | Water EOS | Only for hydrous systems |

**Priority check**: Examine whether `Tspin.f` is called during therm() for Fe-bearing phases (fp, bridgmanite). If so, this must be implemented.

---

## 10. Implementation Plan

### Phase 1: Parameter Loading and Pre-Processing

1. Write `load_params(param_dir)` that reads the HeFESTo_Parameters_010123 files into a `(nspec, 43)` array, **applying unit conversions at load time**:
   - `params[:, 12:15] *= hcok`  — ws1, ws2, ws3: cm⁻¹ → K
   - `params[:, 15] *= hcok`     — we1: cm⁻¹ → K (index 15)
   - `params[:, 17] *= hcok`     — we2: cm⁻¹ → K
   - `params[:, 19] *= hcok`     — we3
   - `params[:, 21] *= hcok`     — we4
   - `params[:, 23] *= hcok`     — wou
   - `params[:, 24] *= hcok`     — wol
   - `params[:, 4]  *= 1000.0`   — Fo: kJ/mol → J/mol (confirm this)

2. Pre-load `dos.inc` table arrays into numpy for fast interpolation.

3. Build `phase_map: dict[str → list[int]]` (phase name → species indices).

### Phase 2: Thermal Kernel Library

```python
# vectorized table lookup (B, nspec) or (B, nspec, nmodes)
def heat_vec(x, d, idos):   # x = θ/T, d = optic bandwidth
    ...
def ener_vec(x, d, idos):   ...
def helm_vec(x, d, idos):   ...

# Grüneisen scaling — all inputs (B, nspec)
def gamset_vec(V, params, ityp=1):
    → returns gamma, q, etas   # all (B, nspec)

# Mode thermal integrals — (B, nspec)
def Etherm_vec(T, V, params): → Uth   (B, nspec)   [J/mol]
def Ctherm_vec(T, V, params): → Cv    (B, nspec)   [J/(mol·K)]
def Ftherm_vec(T, V, params): → Fth   (B, nspec)   [J/mol]
def Ztherm_vec(T, V, params): → zeta  (B, nspec)   [dimensionless]
```

### Phase 3: Pressure Function and Volume Solver

```python
def pressure_vec(V, P_target_GPa, T, params):
    → total pressure residual: (B, nspec)

def dPdV_vec(V, T, params):
    → analytical dP/dV: (B, nspec)

def solve_volume(P_bar, T, params, tol=1e-8, maxiter=20):
    → V: (B, nspec)
    # Newton-Raphson with Murnaghan initial guess
    # Fallback to scipy.brentq for non-converged points
```

### Phase 4: Thermodynamic Properties at Converged Volume

```python
def compute_therm(V, P_bar, T, params):
    → dict with keys: K, Ks, Gsh, alp, Cv, Cp, rho, deltas
    # All arrays (B, nspec)
    # Includes Landau corrections (masked where Tc==0)
```

### Phase 5: Phase Averaging and Output

```python
def aggregate_properties(Ks, Gsh, V_spec, X, phase_map):
    → rho: (B,), Vp: (B,), Vs: (B,)
    # Voigt, Reuss, Hill averaging
```

### Testing Strategy

1. **Thermal kernel unit tests**: For forsterite at T₀=300 K, `Etherm(300, ...)` should give the zero-point + reference thermal energy consistent with `Uto = apar(5) - Fo_reference`. For T=1600 K, compare numerical derivative of `Ftherm` with `Ctherm`.

2. **Single-phase volume test**: Use benchmark row at P=0 GPa. For each species with known ρ (e.g., fo: ρ ≈ 3.23 g/cm³, wm ≈ 140.7 g/mol → V ≈ 43.5 cm³/mol = V₀), verify solver returns V₀ at T=T₀=300 K, P=0.

3. **Intermediate output extraction**: Instrument the Fortran (or examine `fort.821`, `fort.822` trace files if enabled) to extract per-species V, Uth, Cv, K, Ks, Gsh at each P-T point. Compare Python output field by field.

4. **Full benchmark comparison**: Run on pyrolite benchmark composition across all 16 P-T points in fort.56.head. Targets: ρ within 0.01%, Vs within 0.01%, Vp within 0.01%.

5. **Performance profiling**: Profile the 2^15 batch after Phase 5 is complete. Expected bottleneck is the Newton-Raphson loop (Phase 3). If slow, consider Numba JIT on the innermost loop.

---

## 11. Benchmark Output Format (fort.56.head)

The fort.56 output contains (by column, with Hill averages):

| Column | Quantity | Units |
|--------|----------|-------|
| 1 | Pressure | GPa (or bar? — **verify from control**) |
| 2 | Depth | km |
| 3 | Temperature | K |
| 4 | Density ρ | g/cm³ |
| 5 | Bulk sound Vb | km/s |
| 6 | S-wave Vs | km/s |
| 7 | P-wave Vp | km/s |
| 8 | Isothermal K_T | GPa |
| 9 | Adiabatic K_S | GPa |
| 10 | Shear G | GPa |
| 11+ | Per-phase contributions, HS bounds, etc. |

**Primary validation targets**: Columns 4 (ρ), 6 (Vs), 7 (Vp).

---

## 12. Constants (from const.f / const.inc)

```python
Rgas    = 8.314472      # J/(mol·K)
hcok    = 1.438775      # K·cm  (= h*c/k_B, for cm⁻¹ → K)
kB      = 1.380650e-23  # J/K
hplanck = 6.626070e-34  # J·s
avogad  = 6.022142e23   # mol⁻¹
T_ref   = 300.0         # K, standard reference temperature
P_ref   = 0.0           # GPa
```

---

*Document generated from analysis of HeFESToRepository/*.f and HeFESTo_Python_Fortranslation/*.py, cross-referenced against BENCHMARK/fort.56.head and BENCHMARK/control.*
