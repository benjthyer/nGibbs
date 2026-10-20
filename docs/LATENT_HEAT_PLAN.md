# Latent-heat / "metamorphic" terms in HeFESTo — what nGibbs is missing and how to get it

**Status:** analysis + plan. No code written yet (the nGibbs repo is not mounted in this session — see §7).

---

## 1. The answer: the derivative you need is `dn/dT` and `dn/dP`

Not `dG/dT` or `dG/dP`. Those are just `-S` and `V`, which the EOS already gives you analytically at
fixed `n`. Your second instinct was right: **the missing quantity is the composition derivative.**

HeFESTo calls them `dndt(ispec)` and `dndp(ispec)` — the derivatives of the *species* mole vector
along the equilibrium path, at fixed bulk composition `b`:

```
dn_i/dT |_{P, b}      [mol / K]
dn_i/dP |_{T, b}      [mol / GPa]
```

They are computed in **`physub.f` lines 156–186** and dumped verbatim to **`fort.42`** ("dndt and dndp
by species") — so you already have a per-species, per-PT benchmark target in
`BENCHMARK/fort.42`. That file is the single most useful artifact for this task.

## 2. Exactly where they enter the reported properties

Three accumulators, all in `physub.f`:

```fortran
! aggregate ("slow" — all species), physub.f:590-594
alpmet = alpmet + dndt(i)*vspeca(i)/volagg     ! metamorphic thermal expansivity
cpmet  = cpmet  + Ti*dndt(i)*sspeca(i)         ! <-- LATENT HEAT
bmet   = bmet   + dndp(i)*dmdp(i)              ! = -sum dndp_i * V_i, a compliance
```

then (physub.f:596–601):

```
alptot = alpagg + alpmet
cptot  = cpagg  + cpmet/wmagg
btot   = volagg / (volagg/btaggr + bmet)       ! softened K_T
cvtot  = cptot - T*volagg*alptot^2*btot/wmagg*1000
gamtot = volagg*alptot*btot/(cvtot*wmagg)*1000
bstot  = btot*(1 + alptot*gamtot*T)            ! softened K_S
```

`fort.56` writes `1e5*alptot`, `cptot`, `bstot`, `btot` — i.e. **the reported α, Cp, K_S and K_T all
include the full metamorphic term.** Your isochemical nGibbs+EOS reproduces only `alpagg`, `cpagg`,
`btaggr`, `bsiso`.

### 2a. Important correction to the project brief: velocities are *not* fully isomorphic

`Vbh` / `Vph` in `fort.56` **do** carry a metamorphic contribution — the "fast" one.
`physub.f:414–431` computes a per-phase `bmet` from `dndtfast` / `dndpfast`, which are restricted to
species in the range `iophase(iph) .. iophase(iph)+mophase(iph)-1`. From `readin.f:222–247`, that
range is the set of **order–disorder species** of a phase (species with identical stoichiometry, e.g.
opx ordering, the Fe spin-state pairs in ferropericlase/bridgmanite). The reasoning is that
intra-phase cation ordering relaxes fast enough to be seismic-frequency, while cross-phase reactions
do not.

So:

| Quantity | metamorphic content |
|---|---|
| `alptot`, `cptot`, `btot`, `bstot` (fort.56) | **full** — all species |
| `bukph` → `baggv/baggr` → `Vbh`, `Vph` (fort.56, fort.58) | **fast only** — order–disorder species within each phase |
| `gshph` → `Vsh` | none (shear modulus is unaffected) |
| `Vsh*vsred`, `Vph*vpred` | fast metamorphic + anelastic Q correction (`qr19`/`vred`) |

If nGibbs' Vp is currently off by a small amount in regions with active ordering (opx, mw spin
crossover), this is a candidate cause. In assemblages with no order–disorder species active,
`nfast = 0`, `bmet = 0`, and velocities are purely isomorphic — which is why the discrepancy would be
patchy, not systematic.

## 3. How HeFESTo actually computes `dn/dT`, `dn/dP`

This is a **linear solve, not a finite difference**. From `physub.f:156–178`:

1. RHS from partial molar quantities (already available from the vectorized EOS):
   ```
   dmdt_i = S_i / 1000      ! sspeca, J/K/mol -> kJ/K/mol   (sign pre-flipped)
   dmdp_i = -V_i            ! vspeca, cm^3/mol == kJ/GPa/mol
   ```
2. Project both onto the null space of the bulk-composition constraint:
   `dmdt^P = Q2^T dmdt`, with `Q2` (nspec × nnull) from `svdsub` on the stoichiometry matrix `s`.
3. Build the **projected Hessian** `H^P = Q2^T H Q2` (`hessfunc.f`), where
   `H_ij = ∂²G/∂n_i∂n_j`.
4. Solve `H^P x = dmdt^P` (SVD pseudo-inverse, `svdsub`), then lift: `dn/dT = Q2 x`.
5. Same with `dmdp` for `dn/dP`.
6. Sanity check they already run: `μ·dn/dT` and `μ·dn/dP` should both be 0.

### Two facts that make this very cheap to vectorize

**(a) `Q2` is a constant across the whole batch.** It is the null space of `s` (the
component × species stoichiometry matrix), computed once in `sform.f` / `setup.f`. It depends on
`s`, **not** on `b`. Since your batch varies `X` (which is `b`) but not the species list, `Q2` is one
fixed `(nspec, nnull)` matrix for all 2^15 rows. `nnull = nspec - nc`.

**(b) The Hessian contains no EOS second derivatives.** `hessian.f` shows `H` comes *entirely* from
the mixing model:
- ideal configurational term: `-T·R/1000 ·` [site-multiplicity sums built from `r(ic,ispec,kst)`,
  `f(iph,ispec)`, and the `iastate` mask]
- excess/regular-solution term: `rsuma`, built from `wreg`, `vreg`, the size parameters
  `apar(:,41)`, and `P`.

Endmember Gibbs energy is linear in `n`, so it drops out of `∂²G/∂n∂n` entirely. **You do not need to
differentiate the EOS.** You need only the solution-model parameters, which are already in
`HeFESTo_Parameters_010123`, plus `n`, `T`, `P`.

`iastate(i,j,k)` (set in `readin.f:249–280`) is a *static* boolean mask flagging species pairs in
which iron sits in different valence/spin states on the same site — those pairs are excluded from
the ideal-mixing sum. Precompute once as an `(nspec, nspec, nsite)` bool array.

**(c) Absent species are handled by regularization, not by set reduction.** `physub.f:160` sets
`n(i) = 1e-16` for absent species rather than removing them. Their Hessian entries then blow up as
`1/n`, which drives their `dn/dT` to ~0. This is *ideal for you*: no ragged per-row active-species
sets, no dynamic masking. Every row in the batch uses the same fixed-shape dense arrays. Mirror the
floor exactly (`nsmall/10 = 1e-16`) and use an SVD/lstsq solve with a cutoff, not a direct `solve`,
because `H^P` will be badly conditioned by construction.

## 4. Recommended plan

### Route A (recommended) — analytic, one nGibbs call, fully batched

Implement `hessian.f` + the projection as vectorized torch/numpy. No extra NN evaluations.

**New module `ngibbs/metamorphic.py`:**

```
build_static_tables(params) -> {Q2 (nspec,nnull), r, f, iastate, wreg, vreg, size}   # once, at import
hessian_batch(n, T, P)      -> H       (B, nspec, nspec)
molar_derivatives(n, T, P, S_i, V_i)   -> dndt (B,nspec), dndp (B,nspec)
metamorphic_terms(dndt, dndp, S_i, V_i, T, volagg, wmagg)
                            -> alpmet, cpmet, bmet        (B,)
metamorphic_terms_fast(...) -> per-phase alpmet/cpmet/bmet using order-disorder mask only
```

**Cost estimate for B = 2^15, nspec ≈ 73, nnull ≈ 73 - nc:**
- `H`: pure einsum over precomputed static tables → a few `(B, nspec, nspec)` tensors, ~700 MB in
  float64 at B=32768. **Chunk the batch (e.g. 4096 rows) to stay in cache/VRAM.**
- `H^P = Q2^T H Q2`: two batched GEMMs, `(B,nnull,nspec)×(B,nspec,nspec)` — this dominates.
  ~2·B·nspec²·nnull flops ≈ 2·3.3e4·73²·60 ≈ 2e10 flops. Trivial on GPU, ~seconds on CPU with BLAS.
- Batched `lstsq` on `(B, nnull, nnull)` with two RHS. `torch.linalg.lstsq` (driver `gelsd`) or an
  explicit `svd` + thresholded reciprocal. `nnull ≈ 60` → cheap.

This should add well under a second on GPU and is the only route that gets you the *exact* Fortran
answer, including in the pathological near-singular regions where finite differences fail.

**Effort:** the Hessian is the only real work — `hessian.f` is 115 lines of triple-nested loops that
need to become einsums. Budget a day, most of it on the `rsuma` (subregular, size-parameter-weighted)
block, which is fiddly.

### Route B (fallback / cross-check) — 5-point stencil on nGibbs

Your original idea, with one important change: **do not finite-difference G.** `S` and `V` are first
derivatives of `G` that the EOS returns analytically, so differencing *them* is one order better
conditioned:

```
S_tot(P,T) = S_agg(n*(P,T), P, T)
Cp_tot = T · [S_tot(P,T+dT) - S_tot(P,T-dT)] / (2 dT) / wmagg
α_tot  =     [V_tot(P,T+dT) - V_tot(P,T-dT)] / (2 dT) / V
1/K_T,tot = -[V_tot(P+dP,T) - V_tot(P-dP,T)] / (2 dP) / V
```

Chain rule confirms this is exactly `cpagg + cpmet/wmagg` etc.:
`dS/dT = ∂S/∂T|_n + Σ_i S_i · dn_i/dT`.

Equivalently, and more diagnostically, difference the mole vector directly and reuse HeFESTo's own
assembly:
`dndt_i ≈ [n_i(T+dT) - n_i(T-dT)] / (2 dT)`, then apply the §2 formulas verbatim.

**Cost:** 5 nGibbs NN calls instead of 1. Given the 10 s / 2^15 budget, this is only viable if a
single nGibbs pass is ≲ 1 s.

**The real risk is NN noise.** If nGibbs' `n` output has ~1e-4 absolute error, then `dndt ≈ 2e-4/K`
(the magnitude in `fort.42` line 30: `fo → -0.000206`) is buried unless `dT` is large enough that
signal ≫ noise. Empirically you'll need `dT ~ 10–25 K` and `dP ~ 0.1–0.25 GPa` — which then smears
Clapeyron-sharp transitions, exactly where the latent heat matters most. **This is why Route A is the
recommendation.** Route B is still worth building as a 30-minute validation harness.

### Route C — hybrid

Use Route A everywhere; use Route B only to validate. Do not ship Route B.

## 5. Validation ladder

Do these in order; do not skip ahead.

1. **`dndt` / `dndp` per species vs `BENCHMARK/fort.42`.** Direct, unambiguous, species-resolved.
   Target: agreement to the 8 decimals printed (`f16.8`). Rows that are `±0.00000000` in the
   benchmark are a free test of the absent-species regularization.
2. **`checkt` / `checkp`**: `Σ_i μ_i dn_i/dT` and `Σ_i μ_i dn_i/dP` must be ≈ 0 (physub.f:185-186).
   Free, and catches null-space/projection bugs immediately.
3. **`alpmet`, `cpmet`, `bmet`** vs the `fort.31` debug lines
   (`'alpmet,alpagg,alptot'`, `'cpmet,cpagg,cptot,cvtot,gamtot'`, `'Pi,X_Fe,bmet,btaggh,btot,bstot'`).
   Also `fort.69` carries `phasebuoyancyparameter, ClapeyronSlope, deltaent, deltavol, alpagg,
   alpmet, alptot, cvtot, bstot` at f25.16 — the cleanest machine-readable target.
4. **`cptot`, `alptot`, `btot`, `bstot`** vs `fort.56` columns 12–15.
5. **Fast terms**: `bmet` per phase vs the `'bmet calc ispec,bmet,dndpfast,dmdp'` lines in
   `BENCHMARK/qout`, then `Vph` vs `fort.56` col 7 in assemblages with active ordering.

If you need more intermediate values than `fort.42` and `fort.31` provide, the cheapest instrumentation
is to add a `write(43,...)` next to the existing `write(42,...)` in `physub.f` dumping
`hespro`, `dmdt`, `dmdp`, `sspeca`, `vspeca` — the `iprint`-independent dump block is already there
at line 179 as a template.

## 6. Things to notate in the code but not implement (per project scope)

- Anelastic attenuation (`qr19.f`, `vred.f`) — applies only to the `VSQ`/`VPQ` columns.
- `dfdp` / `phasebuoyancyparameter` / `ClapeyronSlope` (physub.f:610–631) — by-products of `dndp`,
  free once you have it, but not needed for ρ/Vp/Vs.
- `dndttest` (physub.f:191–213) — an alternative `nspec×nspec` route to the same `dndt`. Slower;
  ignore, but note it exists as an independent check if the projected route misbehaves.

## 7. Blockers for this session

- **The nGibbs repo is not mounted.** Only `HeFESToRepository` (WSL) and `DissertationBJT` are
  connected. Please connect the nGibbs folder so I can write `metamorphic.py` into it.
- **The shell is unavailable this session.** The WSL UNC mount (`\\wsl.localhost\...`) breaks the
  sandbox mount layer, so every bash call fails with `UNC paths are not supported`, and
  `request_cowork_directory` fails for the same reason. I can read/write files but cannot run or test
  Python. If you can expose the repos under a drive-letter path (e.g. a `subst` or a Windows-side
  clone) instead of `\\wsl.localhost\`, I get the shell back and can actually run the validation in
  §5.
