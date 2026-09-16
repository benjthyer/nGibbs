# Plan — continuous phase-saturation model

Replaces the binary saturation gate with a signed, continuous saturation variable
whose ReLU *is* the molar abundance. New classes only; `NN.py` is not modified.

---

## 0. Two questions settled first

### 0.1 Does clamp-then-reproject converge in three iterations?

Not as generic alternating projection — that converges asymptotically, at a linear
rate set by the angle between `{An = b}` and the nonnegative orthant, and the angle
gets shallow exactly where a phase is going out. So the naive loop is slowest in the
regime we care about.

But the loop we actually want has a much better guarantee, and it is the same thing
you asked for — *absent phases must not participate in reactions*. Restrict the
correction to the active support:

```
m = leaky_relu(g)                 # signed saturation, B x C
n = clamp(m, min=0)               # exact zeros, exact sparsity
Omega = (n > 0)                   # active support, B x C bool

repeat k = 1..3:
    r     = b - n @ compToEl                       # B x E residual
    A_Om  = compToEl.T * Omega.unsqueeze(1)        # B x E x C, columns off Omega zeroed
    M     = A_Om @ A_Om.transpose(1,2) + lam*I     # B x E x E   (E = 8)
    lam_v = torch.linalg.solve(M, r.unsqueeze(-1)) # B x E x 1
    delta = (A_Om.transpose(1,2) @ lam_v).squeeze(-1)
    n     = clamp(n + delta, min=0)
    Omega = (n > 0)
```

`delta` is the minimum-norm mass-balance correction **supported on Omega**, so a
zeroed phase can never be reactivated by it. Because entries can only leave the
support and never re-enter, `Omega` is monotonically non-increasing, the iteration
is finite, and it terminates rather than merely converging. Two to three passes is
right in practice because only a few components overshoot.

The solve is `B x 8 x 8` — trivially batched, differentiable, negligible cost.

Two guards:
- Tikhonov `lam ~ 1e-6` keeps `M` invertible if the support shrinks below the rank
  needed to span `b`.
- Log the final `||r||`. A persistent residual means the network has zeroed a phase
  the bulk composition requires — the continuous analogue of what the old phase
  forcing loop was patching, and now visible instead of silently repaired.

Enable the projection after warmup (see §5), not from step 0.

### 0.2 HeFESTo already writes both partials — `fort.42`

`fort.42` is `dndt and dndp by species`: one block per pressure step, 73 species per
block, 501 blocks aligned 1:1 with `fort.56` rows. These are the **true partials**
`dn/dT|_P` and `dn/dP|_T`, not a directional derivative.

Verified by chain rule against `fort.99` species moles on the Htz run:

```
dn/dP|_isentrope  =  dn/dP|_T  +  dn/dT|_P * dT/dP
```

| species | numeric | chain | ratio | corr |
|---|---|---|---|---|
| mgri | 6.047 | 6.069 | 1.000 | 0.938 |
| mgpv | 4.389 | 4.637 | 1.000 | 0.903 |
| pe   | 1.778 | 1.790 | 1.000 | 0.894 |
| mgil | 1.657 | 1.932 | 1.000 | 0.870 |
| st   | 0.324 | 0.312 | 1.000 | 0.961 |

Median ratio 1.000 across the board; residual scatter is finite-difference noise at
the sharp steps. **No stencil generation is needed and no cross-run linkage is
needed** — every row can carry both partials, and rows stay independent.

Parser (validated on Htz_transition):

```python
def load_fort42(path):
    """Returns names (S,), dndt (nP, S), dndp (nP, S). Blocks align 1:1 with fort.56 rows."""
    lines  = open(path).readlines()
    blocks = [i for i, l in enumerate(lines) if 'dndt and dndp' in l]
    nsp    = blocks[1] - blocks[0] - 1
    names  = [lines[blocks[0] + 1 + k].split()[1] for k in range(nsp)]
    dndt   = np.zeros((len(blocks), nsp)); dndp = np.zeros_like(dndt)
    for b, i in enumerate(blocks):
        for k in range(nsp):
            p = lines[i + 1 + k].split()
            dndt[b, k] = float(p[2]); dndp[b, k] = float(p[3])
    return names, dndt, dndp
```

Three implementation gotchas:

1. **Species order differs from `fort.99`.** `fort.42` uses parameter-file order
   (`an, ab, sp, hc, smag, picr, en, ...`); `fort.99`'s columns do not match it
   positionally. Map by name, never by index.
2. **Normalisation — which denominator decides whether the quotient rule applies.**
   `fort.42` is absolute mol/GPa and mol/K.
   - Normalised by the **species total** `sum_s n_s`: the quotient rule is mandatory.
     Measured on the Htz run, that total swings 3.13 -> 4.37, a **33.6% variation**,
     because reactions like `ri -> pv + pe` do not conserve mole count.
   - Normalised by the **element/bulk total** `sum_e b_e`: `dN/dP = 0`, because the
     bulk is fixed along an isentrope. The quotient rule collapses to a single
     constant scale factor per run: `d(n_i/N_el)/dP = (1/N_el)(dn_i/dP)`.

   `forward_phase_moles` divides by `reconBulkUnNormed.sum(dim=1)`, i.e. the element
   total, so **the constant-scale case applies** — no quotient rule needed, just
   divide the `fort.42` values by that run's element total.

   The trap is that `N_el` is constant *along* a profile but not *across* runs:
   Htz_transition has 23.63948, BENCHMARK has 24.00450. The invariant in both control
   files is **cations = 10.00000**, with oxygen (and therefore the total) floating with
   Fe3+ content and the Si/Mg ratio. Hardcoding 24 would introduce a scale error that
   is correlated with composition -- it would masquerade as an Mg/Si trend. Read
   `N_el` per run from the control file.
3. **`fort.99` has 75 data columns, not 73** -- the 73 species plus `Gibbs` and
   `Quality`. Positional slicing silently mixes them into species arrays. Match by name.
4. **`fort.99` needs the WARNING-line filter** when read alongside `fort.42`, or block
   alignment slips (see the existing note on interleaved diagnostic lines).

Also worth knowing: on the Htz run the total `|dn/dP|` has a 25 km FWHM and 123 km of
nonzero support, against a 2.00 km `d2Vs/dP2` velocity window. Reactions run
continuously — Fe-Mg exchange, garnet dissolution — far outside the sharp step, and
the peak `|dn/dP|` is at 22.40 GPa (`mgil -> mgpv`), not at the post-spinel step at
23.34-23.42 GPa. So derivative supervision carries signal across the whole profile,
not only at velocity transitions, and it is complementary to the `Vs` width target
rather than redundant with it.

### 0.3 If some rows still lack derivative labels

`fort.42` is standard output (it is present in `EOS_arithmetic/BENCHMARK` too), so
newly imported data should be fully labelled. For legacy runs where it was not
retained:

- **Normalise the derivative loss by the count of labelled rows in the batch**, not by
  batch size, or the effective weight fluctuates with batch composition.
- **Stratify batches** — a fixed number of labelled and unlabelled rows per batch makes
  the effective weight deterministic and removes any covariance between batch
  composition and loss scale.
- The subtle failure is not estimator bias but gradient budget: a row carrying two loss
  terms contributes more total gradient than a row carrying one, which quietly upweights
  the labelled subpopulation in the shared trunk. Per-term normalisation plus
  stratification handles it.
- Do **not** label only the 10 MPa resample windows. Given the 123 km support above,
  that would discard most of the derivative signal and teach sharpness at transitions
  with no notion of the smooth background.

---

## 1. Module layout

New file `src/ngibbs/engine/NN_continuous.py`:

| class | role |
|---|---|
| `ContinuousPhaseModel(TunableModel)` | drops `sat_head`, adds per-phase mole branches |
| `ContinuousMidLevelNetwork(ContinuousPhaseModel, MidLevelNetwork)` | inference/forward parity with `MidLevelNetwork` |
| `MassBalanceProjector` | the §0.1 loop as a standalone `nn.Module` |

Subclassing `TunableModel` inherits `_set_indexer`, the registered stoichiometry
buffers (`compToEl`, `phaseToCompMap`, `variedToAllComp`, `fixed_phaseToCompMap`),
`save()`, and the `polish_negative_*` helpers without touching `NN.py`.
`__init__` calls `super().__init__(...)` then `del self.sat_head`.

---

## 2. Architecture

### 2.1 What is removed

- `sat_head` (the per-phase binary heads)
- `likelihoods`, `binary_pred`, `zero_mask` in the forward path
- the phase-forcing `while force_phases` loop — **a second, independent source of
  discontinuity**, since `binary_pred[row, idx] = 1.0` is itself a jump
- `binary_mask` multiplication in `forward_phase_moles`

### 2.2 What replaces it

The mole head becomes the saturation model. One branch per phase, geometry
configurable exactly as the encoder and middle brain are:

```python
def __init__(self,
             encoderLayerUp=1, encoderLayerDown=0,
             middleLayerUp=2,  middleLayerDown=1,
             moleLayerUp=2,    moleLayerDown=1,     # NEW: per-phase branch geometry
             moleBranchBase=32,                     # NEW: first width of each branch
             mole_activation='leakyrelu0.05',       # NEW: leaky | softplus_anneal
             mole_regularization='none',            # NEW: own reg, branches are small
             ...)
```

Branch construction mirrors `finish_build`'s middle brain, one per phase:

```
self.mole_head = nn.ModuleList()
for _ in range(self.n_phases):
    w = moleBranchBase
    layers = [nn.Linear(middle_out, w)] + mole_reg_modules(w)
    for _ in range(moleLayerUp):    w2 = w*2;          layers += [nn.Linear(w, w2)] + mole_reg_modules(w2); w = w2
    for _ in range(moleLayerDown):  w2 = max(1, w//2); layers += [nn.Linear(w, w2)] + mole_reg_modules(w2); w = w2
    layers.append(nn.Linear(w, 1))          # NO activation here
    self.mole_head.append(nn.Sequential(*layers))
```

Assert `moleLayerUp >= moleLayerDown`, matching the existing convention.

The final `Linear(w, 1)` output is `g_phi`, the **signed saturation variable**. Its
physical reading is the complementarity condition `n_phi >= 0`, `A_phi >= 0`,
`n_phi * A_phi = 0`: negative `g` is undersaturation, positive `g` is the amount
that precipitates. That is what the old classifier was discretising.

### 2.3 Activation

`leaky_relu(g, 0.05)` in training, `clamp(min=0)` at inference and in the projector.

Do **not** use softplus. Two reasons:
1. `softplus > 0` strictly, so a phase never truly leaves. `compute.py` builds
   `active = X > 0.0` and relies on ~15 of 73 species being active — under softplus
   every species goes active and the EOS kernel cost multiplies.
2. `clamp(leaky(x), min=0)` is continuous at `x = 0` because the leaky output passes
   *through* zero. `clamp(softplus(x), min=eps)` jumps by `eps`.

Optional `softplus_anneal` mode: train `softplus(beta*x)/beta` with `beta` on a
schedule to hard ReLU. Keep it configurable; leaky is the default.

### 2.4 Unchanged

`chem_heads` (softmax over endmembers within a phase) stay as they are. The
`n_phase * x_endmember -> componentMoles` construction in `forward_phase_moles` is
already the right factorisation — only the `* binary_mask` and the log/eps inversion
come out.

---

## 3. Mass balance

`MassBalanceProjector` implements §0.1 against `self.compToEl`. Config:

```
massbalance_iters: 3          # 0 disables, projector becomes identity
massbalance_tikhonov: 1.0e-6
massbalance_warmup_epochs: 5  # identity before this
```

Keep the existing `reconBulk` loss as well. The projector makes balance exact for
the support the network chose; the loss is what teaches it to choose a support that
*can* be balanced.

The nullspace form `n = A^+ b + N z` is the same object viewed differently — the
columns of `null(A)` are balanced reactions, and the projector's `delta` is a step in
`null(A_Omega)`, the reactions available to the active support. Predicting `z`
directly would make balance exact without iteration but spends 65 coordinates
describing mostly zeros, and cannot express the support restriction you need. The
projector is the better trade here.

---

## 4. Losses

| term | change |
|---|---|
| phase abundance | **keep active for absent phases.** `n = 0` is real ground truth and is the only signal that teaches where zero is |
| endmember composition | **mask by presence** — no ground truth for an absent phase's chemistry |
| composition weighting | weight by `sqrt(n_phi)` so capacity is not spent on chemistry that cannot affect an aggregate property |
| `reconBulk` | unchanged |
| saturation aux | optional BCE on `sign(g)` vs presence, as deep supervision only. **Never multiplies the output** |
| **derivative (new)** | `lambda_P * || dn_hat/dP - dn/dP ||` and `lambda_T * || dn_hat/dT - dn/dT ||`, both by autograd w.r.t. the corresponding input, both targets straight from `fort.42` |

Derivative term: start `lambda = 0`, ramp in after the abundance loss plateaus.
Weight it by `1/(1 + |dn/dP|)` or normalise per phase — the raw magnitudes span
orders of magnitude across the transition.

---

## 5. Training schedule

1. **Warmup (epochs 0-N1)** — projector off, derivative loss off, aux saturation loss
   on. Gets abundances roughly right.
2. **Projection on (N1-N2)** — network learns to emit supports that survive
   projection. Monitor final `||r||`.
3. **Derivative on (N2-end)** — ramp `lambda`. This is where transition width becomes
   supervised rather than emergent.

Warm-start the encoder and middle brain from the current trained checkpoint; the
mole branches are new and initialise fresh.

---

## 6. Data

Generation changes:
- coarse profile resolution **1.0 -> 0.25 GPa** (tighter bracket for the phase-change
  detector, so the fine resample window can be narrower)
- fine resample at **10 MPa** — validated by the Htz run, which resolves a 2.00 km
  (0.08 GPa) transition cleanly at that spacing
- resample window **+/- 0.5 to 1.0 GPa** around each detected transition, not tighter.
  The post-garnet ramp runs ~1 GPa above the post-spinel step; a tight window teaches
  the step without its context
- entropy spacing **dS ~ 0.014 J/g/K**. Since `(dT/dS)_P = T/c_p`, at `T ~ 1800 K` and
  `c_p ~ 1.25` this puts adjacent isentropes ~20 K apart, so their transitions overlap
  (a 20 K shift moves the transition ~0.85 km against a 2 km width)

Loader changes (`loadTrainData.py`, new columns, rows stay independent):
- parse `fort.42` per simulation dir (see §0.2), map species **by name**, apply the
  quotient-rule renormalisation, emit `dn_dP` and `dn_dT` columns per component
- retain `fort.42` in the importer's keep-list — it is the derivative dataset
- no stencil generation, no cross-run linkage, no finite differencing

Hold out **entire isentropes**, not random rows, or neighbouring samples leak and the
width metric flatters itself.

---

## 7. Acceptance gate

Aggregate metrics are blind to this failure — the current model has `Vs` rms 0.42%
against HeFESTo while being 4x wrong on transition width. Add to the validation suite,
run against `data/HeFESToWorkspace/Htz_transition`:

| metric | current | target |
|---|---|---|
| `d2Vs/dP2` window width | 0.50 km | **2.00 km** |
| `dVs` across window | 1.90% | **3.97%** |
| base depth | 662.9 km | **659.6 km** |
| latent-heat `dT` | 43.2 K | **18.3 K** |
| active species per row | ~15 | stays ~15 (guards against softplus creep) |
| `T` rms, `rho` rms, `Vs` rms | 2.0 K / 0.34% / 0.42% | no regression |

`notebooks/660_brackets/htz/htz_compare.py` already computes the first four.

---

## 8. Risks

- **Optimisation is harder.** The factorised classifier x regressor is easier to fit
  than a constrained nonnegative head. Mitigation: warm start, aux saturation loss,
  staged schedule.
- **Dead phases.** Plain ReLU gives no gradient to revive a wrongly-absent phase.
  The leak is what prevents this; do not set `activation_leak = 0`.
- **Support collapse.** The network may zero a phase the bulk requires. The projector
  will not fix it, by design. Monitor `||r||` and the count of rows where it exceeds
  tolerance; that number replaces the old forcing-loop counter.
- **Legacy data may lack `fort.42`.** Newly generated runs are fully labelled, but
  archived workspaces may not have retained it. Check coverage before setting
  `lambda_P`, `lambda_T`; fall back to §0.3 if coverage is partial.
- **Quotient-rule renormalisation is easy to get wrong** and fails silently, biasing
  every derivative target near a transition where it matters most. Unit-test it by
  reproducing the §0.2 chain-rule table.
