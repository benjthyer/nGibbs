# Derivative-supervised training — implementation plan

Everything below that is stated as a measurement was measured in this session, on the
real `ml_indexer` (62 components, 18 phases, 8 elements) and the real
`Htz_transition/fort.42` and `somesims/Simulation2/fort.42`. Numbers are quoted so you can
re-derive them. §0c has been rewritten twice after Ben caught errors in it; the note at the
end of that section says what each draft got wrong.

---

## 0. Three findings that shape the design

**(a) Do not autodiff through the mass-balance projector.** Taking a JVP through
`ContinuousModel.forward` including `MassBalanceProjector` took **12,807 ms**; through
the network alone, **29 ms**. That is 440×, and it buys nothing — see §2.

**(b) Use forward-mode AD, not reverse.** You want ∂n_c/∂P for *all* C components in
*one* input direction. That is a Jacobian-**vector** product — forward mode, one pass.
Measured at B=128, C=62: forward mode for both directions plus backward, **76 ms/step**;
reverse mode, one backward per component, **~383 ms** for the Jacobian alone. The gap
grows linearly in C, and C grows if you ever add phases. `impl/losses.py:77-93`
(`per_component_derivative_loss`) is a `for c in range(C)` reverse loop — that is the
approach to abandon, and its own docstring says so.

`torch.func.jvp` is differentiable w.r.t. parameters — confirmed, gradients flow to
`encoder`, `chem_heads` and `mole_head`.

**(c) HeFESTo's mass balance is clean. The violation I reported in the first two drafts
of this plan was a bug in my own species mapping.** *(Corrected twice — see the note.)*

`ml_indexer.components_in_phases` contains the name **`magnetite` twice**: once under
`spinel`, once under `ferropericlase`. HeFESTo distinguishes them (`smag` in phase `sp`,
`mag` in phase `mw`) and so does `HeFESTo_snames_long` (`magnetite-spinel` vs
`magnetite`), but a lookup keyed on component name alone collapses them. Mine did: both
model components read fort.42's `mag` column, `smag` was never read, and the error vector
was `(n_mag - n_smag)` times the Fe3O4 composition — which is why the residual looked so
convincingly rank-1 along `Fe = 1, Fe3 = 2`, and why it switched on exactly where
ferropericlase magnetite becomes non-zero.

With the mapping keyed on **(phase, species)** instead:

| run | rows | violating rows | max abs residual | Fe_total drift | O drift |
|---|---|---|---|---|---|
| `Htz_transition` | 501 | **0** (was 166) | 7e-8 | 1.0e-7 | 4.7e-7 |
| `somesims/Simulation2` | 4089 | **1** (was 1690) | 6.8e-3 | 7.3e-4 | 5.0e-7 |

Everything is at fort.42's printed precision. Row 0 of `Htz_transition` reconstructs the
control file exactly — `Fe_total 0.580520`, `O 13.639470`, `cations 10.000010` — and stays
there for all 501 rows. `Aᵀ dn/dP = 0` and `Aᵀ dn/dT = 0` hold to 7e-8 everywhere.

Your framing was right on every point: O is a fixed input, only Fe is redox-active, so the
`(Fe_total, O)` ↔ `(Fe2, Fe3)` remap is unique and both are conserved. `compToEl` encodes
γ- and ε-iron as `Fe = +3, Fe3 = -2` — exactly `Fe(0) = 3FeO - Fe2O3` — giving
`Fe_total = 1`, `O = 0` for metal, so the Fe0/Fe2/Fe3 speciation moves freely without
touching either conserved quantity.

**The real importer does not have this bug.** `_resolve_component_phase` keys on the
fort.42 abbreviation through the control file, returning `smag -> (magnetite, spinel)` and
`mag -> (magnetite, ferropericlase)`, and `_safe_assign` writes to
`indexer.MELTS_indices[phase][component]`. Verified by calling them directly. Sidecar
tables already built are fine.

**Two things that survive:**

- `somesims/Simulation2` row 3959 (P = 11.000 GPa, T = 2337.7 K) has a residual of 6.8e-3
  with monotonic P on both sides — not a seam, just one badly converged row in 4089. A
  residual threshold drops it. That is the whole row filter.
- The 11 genuinely untracked species carry up to **7.33e-4 mol** and **2.26e-3 mol/GPa** in
  `Simulation2` — small but not zero, and it accounts for that run's entire 7.3e-4
  `Fe_total` drift. The likely carrier is **`alpha-iron`**, which the model does not track
  even though it does track γ- and ε-iron. Worth deciding deliberately rather than by
  omission; 0.13% of total Fe, presumably confined to low P.

> **What the first two drafts said, and why both were wrong.** v1 read the Fe/Fe3 drift as
> HeFESTo treating ferric/ferrous as a redox freedom and proposed masking both elements out
> of the projection. v2 corrected the physics — O is an input, so both must be conserved —
> but then read the residual as a real solver artifact and proposed projecting the target to
> repair it. Both were the same mistake: interpreting an artifact of my own column mapping
> as a property of the data. The 99.98%-rank-1 residual should have been the tell — a
> perfectly clean stoichiometric direction is what a mislabelled column looks like, not what
> a converging solver looks like. **The check that would have caught it in one line is
> asserting the species map is injective** (`len(set(cols)) == len(cols)`), which is now
> step 0 of the QC gate in §4g.

---

## 1. What to supervise, and why the species axis

`fort.42` is per-**species** (endmember), 73 of them; the model tracks 62 and those 62
carry **100.00%** of |dn/dP| — nothing leaks into untracked species.

Supervise on the species axis `C`, not the phase axis. The identity is

    n_c = n_phase(c) · x_c        →        dn_c/dP = (dn_phase/dP)·x_c + n_phase·(dx_c/dP)

Contracting to phases (`@ phaseToCompMap.T`, which is what `molar_labels.npy` already
does) annihilates the second term. That term is the change of *phase composition* with
pressure — Fe partitioning between ringwoodite and bridgmanite+ferropericlase — and it is
the term that sets the transition width. Nothing else in the loss constrains it. Summing
over components (what `impl/losses.py:73` currently does) throws away even more.

**You do not need a new value target.** The C-axis moles are already reconstructible from
what the bundle carries:

```python
n_C_target = (chem_target @ variedToAllComp + fixed_phaseToCompMap) * (mole_target @ phaseToCompMap)
```

which is exactly the model's own `phaseProportions * compMultipliers`. Only the two
derivative arrays are new.

---

## 2. The core move: differentiate the constraint, not the solver

Let `n̂` be the raw network component moles (pre-projector) and `Π` the mass-balance
projection onto `{Aᵀn = b}` restricted to the active support `Ω = (n > 0)`. Then

    dn/dP = ∂Π/∂n̂ · dn̂/dP + ∂Π/∂b · db/dP,      db/dP = 0  (all eight elements; §0c)

and at the projection's fixed point `∂Π/∂n̂` is just the orthogonal projector onto the
tangent space `null(A_Ωᵀ)`. So:

1. JVP through the **network only** — no `linalg.solve` in the graph (29 ms, not 12,807).
2. Apply the tangent projector analytically, one batched 8×8 solve:

```python
def tangent_project(jvp, n):                      # used on BOTH prediction and target
    A_om = compToEl.T.unsqueeze(0) * (n > 0).unsqueeze(1)      # (B, E, C)
    M    = A_om @ A_om.transpose(1, 2) + tikhonov * I
    lam  = torch.linalg.solve(M, (jvp @ compToEl).unsqueeze(-1))
    return jvp - (A_om.transpose(1, 2) @ lam).squeeze(-1)
```

**Apply this to the prediction only.** The target already satisfies `A^T dn = 0` to 7e-8
(§0c), so projecting it is a no-op — measured at 4.4e-8 on clean rows, which is now *all*
rows. Running it on the target anyway, as a cheap assertion rather than a repair, is
reasonable: if it ever moves a target by more than the noise floor, either the mapping
broke or the row is bad, and both are things you want to hear about.

Measured on an untrained net: `‖Aᵀ dn‖` **1.82e-2 → 1.12e-8**, and it stays differentiable
(44 params get gradients, same set as without it). The projection changes the raw tangent
by 70% at init — that is 70% of the derivative signal being structure the network would
otherwise have to learn from scratch.

This is the implicit differentiation you asked about. It is not implicit differentiation
of the *equilibrium* — that would need the Gibbs Hessian, which the network does not
emit — it is implicit differentiation of the projection layer, and it is both exact at
the fixed point and 440× cheaper than unrolling.

**Note on the support.** `Ω = (n > 0)` is piecewise constant, so it contributes no
gradient — which is correct: `clamp(leaky_relu(g), min=0)` already gives `dn/dP = 0` for
an absent phase, matching HeFESTo. The flip side is that the derivative loss *cannot make
a phase appear* — only 44/144 parameter tensors receive gradient from it, because only
13 of 62 components are active in a typical row. Keep the value loss carrying that job.

---

## 3. The units chain (verified end to end)

Four factors, none optional:

| step | factor | where it lives |
|---|---|---|
| `fort.42` raw | mol/GPa, mol/K | `P(GPa)`, `T(K)` are literally the HeFESTo feature units — no unit conversion |
| element-total basis | ÷ `InTot` | `MLexporter.py:360-361`, `:420`. `InTot` is constant along a scan (Mg/Si/Ca/Al/Cr/Na constant to 1e-7), so **no quotient-rule term** |
| feature normalization | × `ranger[P_idx]` | `Normalizer` fit in `load_ML_data`, stored as `ml_indexer.feature_normalizer`; bounds in `feature_bounds.json` |
| model output space | linear moles | `ContinuousModel` predicts linear `m`; the gated model's `log10(n+eps)` would add a `1/((n+eps)ln10)` factor that is singular at `n=0` — another reason this rides on `ContinuousModel` only |

The `ranger` factor is a happy accident as well as a necessity: normalized P spans [0,1],
so `dn/dx_norm ≈ 140 × dn/dP ÷ 23.6 ≈ 0.6` against `n ≈ 0.1` — same order of magnitude as
the value target. The derivative loss is naturally non-dimensionalized and does not need a
hand-tuned global weight to be numerically comparable.

`impl/losses.py:59` assumes the target is "already normalised by the run's element total"
but never applies the `ranger` factor. That is a real bug in the existing sketch.

---

## 4. Work items

### 4a. `BigMetaTable.retrieve_component_moles` — finish the `dmolar` fill
The allocation and the row-count assertion exist (`BigMetaTable.py:1574-1582`); the fill
loop inside `for phase in phases` under `self.Model == 'HeFESTo'` was deferred. This is a
prerequisite for everything else — without it `self.dmolar` is allocated and empty.

### 4b. `MLexporter.resampling_to_datasets` — export the derivative labels
Currently `grep -c "sidecar\|dmolar" MLexporter.py` → **0**. The sidecar work stops at
`BigMetaTable`. Add, in the chunk loop at `MLexporter.py:355-443`:

```python
self.dndp_labels[out_start:out_end] = dmolar_dndp_chunk / InTot_chunk[:, None]   # (rows, C)
self.dndt_labels[out_start:out_end] = dmolar_dndt_chunk / InTot_chunk[:, None]
```

Note **no** `@ phaseToCompMap.T` — these stay on the C axis (§1). Then:

- allocate two `(R·rows, ncomps)` memmaps alongside `MLexporter.py:239-283`
- add to `_ROW_ALIGNED_ARRAYS` (`:38-45`) so the post-hoc shuffle keeps rows aligned
- add to `file_mappings` (`:546-552`) and to `load_ml_bundle`'s `npy_files`
  (`utils/file_utils.py:465-472`) and `MLDataBundle` (`:410-419`)

Also write into `stats` / a new `derivative_stats.json`:
- **`massbalance_residual`** — per-row `||A^T dn||`, both partials. Expect ~1e-7; use it
  to drop the rare bad row (1 in 4089 in `Simulation2`) and to catch a future mapping
  regression. It is a sharp test precisely because the clean value is machine noise.
- **`derivative_scale`** — per-component robust scale (95th percentile of |dn/dx_norm|)
  for the loss normalization in §5.
- **`has_derivative`** — per-row mask. Rows without `fort.42` are NaN-filled, never
  dropped, to preserve row parity (`HeFESTo_derivative_import.py:220-228`). Store the
  mask rather than re-deriving it from NaN at every batch.

Cost: two `(rows, 62)` float32 arrays. Against the current `(rows, 18+18+62+10)` that is
roughly a doubling of bundle size — worth confirming against your disk budget before a
full re-export.

### 4c. `loadTrainData.py` — arity
`TensorDatasetFour`'s name encodes the arity, and `trainer.py:147` and `:467` destructure
a 4-tuple. Generalize to `batch[:4]` + optional `batch[4:6]`, widen
`ChunkedMemmapTrainLoader`'s memmap set, and expose a `bundle.has_derivatives` flag. I
don't have this file — it is not on this filesystem — so this item is specified, not
drafted.

### 4d. `ContinuousModel` — two hooks
```python
def network_component_moles(self, x):
    """Pre-projector componentMoles on the C axis. Split out of forward() so the JVP
    can target it without dragging linalg.solve into the tangent graph."""

def tangent_projector(self, dn, n, conserved_mask):
    """Project a tangent onto null(A_Ω^T) restricted to conserved elements."""
```
`forward` then calls `network_component_moles` and the existing projector, so there is
one definition of the network body. No change to the gated model — `NN.py`'s `PhaseHead`
does an in-place softmax write and is not double-backward-safe, which is already
documented in `NN_continuous.py:106-132`.

### 4e. New module `builder/training/sobolev.py`
`train_Upper_Sobolev(model, ...)` — same signature as `train_Upper_MELTS` plus
`dndp_alpha`, `dndt_alpha`, `project_tangent`, `deriv_subsample`. It reuses the pieces
from last week's generalization rather than forking them: `_upper_forward`, `_upper_loss`,
`_resolve_heads_to_freeze`, `_regularization_spec`, the adaptive-dropout block and the
early-stopping/checkpoint block all import from `trainer.py` unchanged. The only new code
is the two JVPs, the tangent projection and the derivative loss term.

`tuners.py` gets a `train_fn=train_Upper_MELTS` parameter (one line in `_run_trial`), so
sweeps work against either objective with no other change.

### 4f. `main.py` — config switch
```yaml
derivatives:
  enabled: true          # bundle MUST carry dndp_labels/dndt_labels
  dndp_weight: 1.0
  dndt_weight: 1.0
  project_tangent: true
  subsample: 1.0         # fraction of rows in a batch that get the JVP
```
`main.py` chooses `train_fn = train_Upper_Sobolev if episode_cfg['derivatives']['enabled']
else train_Upper_MELTS` and passes it to the tuner. That is the whole change — about ten
lines, plus registering the new keys in `TYPE_CONVERSION_MAP`.

**On your "prioritize the new workflow" instruction:** `enabled: true` against a bundle
with no derivative arrays should raise, naming the bundle path and the re-export command —
never silently fall back. A silent fallback is how you discover six weeks later that the
long run you cared about trained on the old objective.

### 4g. QC gate — `scripts/check_derivatives.py`
Run before any long training, print/fail on:
0. **Assert the species map is injective** — `len(set(cols)) == len(cols)` — and that it
   is keyed on `(phase, species)`, not species name alone. `ml_indexer` contains
   `magnetite` twice (spinel and ferropericlase); a name-keyed map silently double-counts
   one and never reads the other. This one line would have saved two wrong drafts of §0c.
1. `Aᵀ dn/dP` and `Aᵀ dn/dT` per element, absolute and relative — no finite differencing at
   all, and the clean value is machine precision (7e-8), so anything larger is real.
   Report the rank-1 decomposition of any residual, not just its norm: a residual that is
   ~100% along one clean stoichiometric direction is the signature of a *mapping* error,
   not a solver error. That is the lesson of §0c and it is worth encoding in the check.
1b. Reconstruct the bulk from `fort.99` and compare against the `control` file's own
   cation and **oxygen** amounts. Row 0 must match to the last printed digit, and so must
   every subsequent row; `Htz_transition` does for all 501.
1c. Report the moles and derivatives carried by *untracked* species. Not zero:
   `Simulation2` reaches 7.3e-4 mol, probably `alpha-iron`.
2. Chain rule against `np.gradient` on monotonic P segments — you already have
   `monotonic_segments` and `verify_chain_rule`, including the derivative-linearity guard.
3. Coverage: fraction of rows with derivatives, per simulation.
4. The species name map: `fort.42` uses short codes in parameter-file order (`an, ab, sp,
   hc, smag, picr, en, ...`) which is **not** `fort.99`'s order and **not** the indexer's
   order. `HEFESTO_ABBREVIATION_TO_SHORT_NAMES` / `HeFESTo_snames_short|long` in
   `config/constants.py` resolve it; 62/62 matched once you go through them. Assert the
   count, don't assume it.

---

## 5. Loss design

```
L = L_value  +  α_P · L_dP  +  α_T · L_dT
L_dP = mean over (row, component) of  huber( (dn_pred - dn_target) / s_c )
```

- **`s_c`** — per-component robust scale from §4b, so a trace component is not drowned by
  forsterite and not amplified into the dominant term either.
- **Masking** — `has_derivative` row mask AND the active support. An absent component has
  target 0 and prediction 0 with no gradient; including it just dilutes the mean.
- **Huber, not L2.** Near a phase-in/out the true derivative is genuinely large and
  genuinely discontinuous; L2 will let those rows dominate. They are also exactly the rows
  you care about, so clip rather than drop — a Huber with the transition at ~3·`s_c`.
- **Tangent-consistency penalty** as a cheap extra (optional): `‖Aᵀ dn_pred‖²`. Redundant
  if `project_tangent: true`, useful as a diagnostic printout either way.
- **Row filter** from `massbalance_residual`: drop rows above ~1e-6. At 1 row in 4089 this
  is a hard cut with no cost, not a soft weight.

**Practical settings for derivative episodes:**
- Prefer `layernorm` or `none` over `batchnorm`. BatchNorm makes the JVP batch-coupled —
  the tangent of one row depends on the other rows in the batch, which is not a property
  the physics has.
- Keep dropout low or zero. The dropout mask applies to primal and tangent alike, so it
  injects noise directly into the derivative target.
- Budget ~3× the plain step time (two JVPs + backward). `subsample` is the knob if that
  hurts; JVP on a random 25% of rows still gives an unbiased gradient estimate.

---

## 6. Suggested order

1. **4g first, on existing data.** The QC script is a day's work, needs no pipeline change,
   and would have caught §0c before it silently biased a run.
2. 4a → 4b → re-export one small bundle. Verify row parity and the units chain by
   finite-differencing the *exported* arrays, not the raw ones.
3. 4d + 4e against that small bundle; check that the derivative loss decreases and that
   `‖Aᵀ dn‖` stays at projection precision.
4. 4c + 4f, then a full re-export and a real run.

**Acceptance test — not the loss curve.** The number that matters is the post-spinel
transition width: HeFESTo ground truth 2.00 km against the emulator's 0.50 km from the
earlier session. If derivative supervision is doing what this plan claims, that gap closes,
and it closes because `dx/dP` is now constrained. Check the isentropic chain rule
`dn/dP|_S = dn/dP|_T + dn/dT|_P · dT/dP` on held-out adiabats as the intermediate metric —
both partials are supervised independently, so their combination is a real generalization
test rather than a fit statistic.

---

## 7. Open questions for you

1. **`alpha-iron`.** The model tracks γ- and ε-iron but not α-iron, which carries up to
   7.3e-4 mol in `Simulation2`. Deliberate (α-iron is only stable at low P, outside the
   range you care about) or an oversight? It is the one remaining leak in an otherwise
   exactly-closed system.
1b. **The duplicate `magnetite` name** in `components_in_phases`. Renaming one to
   `magnetite-spinel` to match `HeFESTo_snames_long` would remove the trap permanently. Any
   reason not to, or does something downstream depend on the current name?
2. **Bundle size.** Two extra `(rows, 62)` float32 arrays roughly doubles it. Acceptable,
   or do you want float16 for the derivative arrays? Their dynamic range is narrow after
   `s_c` normalization, so float16 is probably fine and I'd rather ask than guess.
3. **`loadTrainData.py`** — send it and I'll write 4c properly instead of specifying it.
4. **Does every training simulation now have `fort.42`?** If coverage is partial, say what
   fraction; it changes whether `subsample` is a cost knob or a correctness concern.
