# Continuous saturation model — first implementation

Four new files. `NN.py` is untouched.

| file | contents |
|---|---|
| `NN_continuous.py` | `ContinuousMidLevelNetwork`, `MassBalanceProjector`, `ContinuousPhaseHead` |
| `losses.py` | abundance hinge, masked composition, Sobolev derivative, aux saturation |
| `hefesto_derivatives.py` | `fort.42` / `fort.99` / control-file loaders and normalisation |
| `test_impl.py` | the smoke tests below |

Drop `NN_continuous.py`, `losses.py`, `hefesto_derivatives.py` into `src/ngibbs/engine/`.

## Verified

```
[loader]      fort.42 (501, 73), fort.99 species 73 (Gibbs/Quality dropped), N_el=23.63948
[loader]      normalised chain rule (mgri): ratio 0.9995
[model]       phases=18, mole branches=18, params=0.50M, sat_head removed
[massbalance] normalised bulk residual: 0:0.385  1:0.218  2:0.179  3:0.179  5:0.105  8:0.105
              monotone, 3-iter reduction 54%
[continuity]  4000 steps over 6 GPa (1.5 MPa each)
              max |d phaseMoles| between adjacent samples: 3.9e-07  (0.000% of max)
[autograd]    d(componentMoles)/dP finite, double backward OK
[train]       full step incl. Sobolev term; gradient reaches all 18 mole branches
```

The continuity number is the point. The binary-gate model jumped **0.171 of 1.0** in a
single 0.2 MPa step; this one moves 3.9e-07 across 1.5 MPa steps, i.e. float32 noise.

## Two bugs found while building this

**1. `PhaseHead` blocks Sobolev training outright.** The existing head does
`proportions[torch.isnan(proportions)] = 0.0`, an in-place write to a softmax output.
On the *double* backward that derivative training requires (`create_graph=True`) this
raises

```
one of the variables needed for gradient computation has been modified by an
inplace operation ... output 0 of Softmax, is at version 1
```

`ContinuousPhaseHead` reproduces the masking semantics out-of-place with `torch.where`.
It is weight-compatible with the original (single `Linear`), so `finish_build`'s heads
are copied across at construction and warm-starting is unaffected. **Any derivative
training against the current head would have failed at the first backward pass**, so
this would have blocked the work regardless of the architecture change.

**2. Holding the mass-balance scale fixed across iterations makes the residual
non-monotone.** First pass fixed the target as `b_target * scale` with `scale` computed
once outside the loop; the residual then went 0.56 -> 0.69 -> 0.15 with iteration count.
Each clamp changes the element total, so a target pinned to a stale scale is
inconsistent. The projector now constrains the composition *direction* only and rebuilds
the scale from the current `n` every iteration.

## Design notes worth carrying

**The abundance loss is a hinge, not an MSE.** `clamp(min=0)` has zero gradient below
zero, so clamping alone would give dead phases. The forward returns the *signed*
saturation `m` alongside the rectified `n`; the loss uses `m`:

```
L = sum_present (m - n_true)^2  +  sum_absent relu(m)^2
```

Present phases are regressed normally; absent phases are only penalised for being
positive, so `m` is free to sit anywhere below zero and keeps gradient. That is the
complementarity condition `n >= 0`, `A >= 0`, `n*A = 0` written as a loss.

**`prior_sat_head`** is a cheap shared `Linear(enc_out, n_phases)` filling the slot the
logits occupied in the middle-brain input. It gates nothing. Its purpose is to keep the
middle-brain input dimension identical so the existing encoder and middle brain
warm-start unchanged, and to give the auxiliary saturation loss somewhere to attach.

**Mass-balance residual is the new forcing-loop counter.** It plateaus at ~0.10 here
because the network is randomly initialised and many random target bulks are simply
unreachable within the support it picked. On a trained model that number should fall;
if it does not, the network is zeroing a phase the bulk requires — which the projector
will not repair, by design. Log it.

## Not yet done

- `per_component_derivative_loss` does C backward passes. Replace with
  `torch.func.jacrev` / `vmap`, or subsample components per step, before using it in
  anger. The contracted `derivative_loss` is cheap and is what the smoke test exercises.
- Importer wiring: `fort.42` needs adding to the keep-list, and `normalised_derivatives`
  called per simulation dir.
- Config plumbing for the new YAML keys.
- Warm-start script from an existing checkpoint.
