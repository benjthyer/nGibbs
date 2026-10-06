# Cumulate inversion (prototype)

`src/ngibbs/engine/cumulate_inversion.py` — recovers the liquid, pressure and
temperature a cumulate crystallised from by running a closed, isothermal MELTS
`ContinuousModel` backwards: gradient descent on the network **inputs**
(P, T, bulk composition) with the weights frozen.

## Problem

A cumulate is a set of cumulus phases (the assemblage) with measured
compositions. The emulator's input is a bulk composition, not a liquid, so one
cumulate is compatible with a family of bulks that differ only in modal
abundance (liquid plus more or less of each cumulus phase). The family is
collapsed by also maximising the liquid fraction: among the bulks that are
multiply saturated in exactly the imposed assemblage, prefer the one with the
most liquid. That optimum sits just inside the multiple-saturation surface, so
the bulk converges onto the liquid composition and (P, T) onto the near-liquidus
condition.

## Formulation

`ContinuousModel` gives one signed affinity per phase, `g_phi`, with
`n_phi = clamp(leaky_relu(g_phi), 0)`: a phase is present iff `g_phi > 0`, and
`g_phi` is continuous through the phase boundary, so it can carry gradients into
the inputs. Scaled by the per-phase vanishing-abundance scale
`ml_indexer.T0` (5th percentile of the nonzero training abundance),
`a_phi = g_phi / T0_phi`.

Per node:

| term | form | default weight |
|---|---|---|
| assemblage | required (cumulus + liquid): `hinge(1 - a)`; forbidden (all other phases): `hinge(a + 1)`; free (fluid): none | 1 |
| composition | mean over cumulus phases of mean over oxides of `((wt_pred - wt_target) / 1 wt%)^2` | 1 |
| liquid | `1 - f_liq`, `f_liq` = element-mole fraction of the raw (not mass-balanced) system held by liquid | 1, ramped in over the first 25% of steps |
| bounds | squared excursion of bulk oxide wt% outside the training range | 1 |

`hinge` is a one-sided Huber: 0 once the constraint is met, `d^2` for
`0 < d < 1`, `2d - 1` beyond. Being exactly zero inside the feasible set, it
does not fight the liquid term there; being linear far outside, one badly wrong
phase cannot swamp every other gradient.

Parameterisation: P, T = sigmoid onto the training range (box constraint);
bulk = softmax over `Elkeys` (closed, positive; any element can be pinned to 0).
Adam, cosine learning-rate decay (0.05 → 0.0025), 600 steps.

No mass balance is used anywhere in the loop. The phase compositions come from
the chemistry heads and the presence test from the affinities, neither of which
depends on the post-hoc projection.

Target compositions from bundles: labels store pyroxene/spinel in the native
MELTS endmember basis (`compToOxLoad`); the chemistry heads emit the
PxSp-transformed basis (`compToOx`). Both are converted to oxide wt% before
comparison.

## Multi-start and degeneracy

Each run optimises many independent nodes (target 2^15). Nodes start at random
pressures (uniform over the training range) and noisy GEOROC/PetDB validation
compositions (log-normal σ = 0.1 per oxide, random Fe³⁺/ΣFe in 0.03–0.25,
clipped into the training box), each at its own liquidus from `find_liquidi`.
The best state seen by each node is kept. The cloud of solutions is summarised
by `InversionResult`: accepted-node filters, summary statistics, weighted
covariance/correlation of (P, T, liquid oxides), and 2-D KDEs.

The density of accepted nodes is **not a posterior**. It is the GEOROC prior
pushed through the optimiser's basins of attraction, then filtered. Where the
objective is flat along a direction (e.g. elements the cumulate does not
constrain, such as K₂O in an olivine–clinopyroxene–plagioclase cumulate), the
solutions simply keep their starting values along it.

## Liquidus finder

`find_liquidi(emulator, compositions, pressures)` finds the liquidi of many
(composition, P) pairs in one vectorised pass. It runs only
`network_component_moles`, with no mass balance and no phase tables. It uses a
descending 48-point grid, then 10 bisections on the highest crossing of
`max_solid(g) = 0` (fluid ignored). Resolution is about 0.05 °C. It returns T,
a status (0 bracketed, +1 solid already at Tmax, −1 no solid by Tmin) and the
liquidus phase. 2048 GEOROC compositions take 12 s on 2 CPU cores; 2^15 take
3.5 min. On the grid-bracketed rows, no row has a solid 0.5 °C above its
liquidus and every row has one 0.5 °C below.

## Validation on the deployment-test bundles

Method: take a liquid-bearing row of the 102 / 120 closed NoCr NPT test subsets,
drop the liquid, and invert its solids. Compare the results with the row's true
(P, T, liquid). Settings: 1024 nodes, 600 steps, CPU. Liquid RMSE is over all
model oxides, in wt%.

Control ("truth basin"): 64 nodes started within ±300 bar, ±10 °C and 3% of
the true liquid. If that basin's best loss is no lower than the GEOROC
ensemble's, then a miss by the ensemble reflects degeneracy of the emulator
objective, not an optimiser failure.

### 102 (rhyolite-MELTS 1.0.2), closed NoCr NPT

| target | true P (bar) | true T (°C) | ΔP best | ΔT best | liq RMSE best | ensemble best loss | truth-basin loss | truth-basin liq RMSE |
|---|---|---|---|---|---|---|---|---|
| ol | 3197 | 1594 | +8325 | +274 | 6.78 | 0.027 | 0.028 | 1.35 |
| ol+cpx | 3634 | 1167 | +1225 | +5 | 1.63 | 0.042 | 0.043 | 1.14 |
| ol+cpx+pl | 397 | 1109 | +442 | −37 | 3.05 | 0.068 | 0.063 | 0.46 |
| cpx+pl | 6939 | 1241 | −288 | −49 | 1.59 | 0.033 | 0.034 | 1.64 |
| ol+opx | 3082 | 1381 | +2636 | +11 | 1.40 | 0.045 | 0.054 | 0.80 |
| ol+opx+cpx+sp | 10046 | 1315 | +1598 | +11 | 0.87 | 0.377 | 0.515 | 0.52 |
| opx+cpx+pl | 5908 | 1051 | −50 | +31 | 0.69 | 0.070 | 0.077 | 0.52 |

### 120 (rhyolite-MELTS 1.2.0), closed NoCr NPT

| target | true P (bar) | true T (°C) | ΔP best | ΔT best | liq RMSE best | ensemble best loss | truth-basin loss | truth-basin liq RMSE |
|---|---|---|---|---|---|---|---|---|
| ol | 4226 | 1732 | +6833 | −521 | 12.16 | 0.027 | 0.028 | 1.09 |
| ol+cpx | 5016 | 1228 | +4122 | +81 | 3.74 | 0.045 | 0.057 | 0.77 |
| ol+cpx+pl | 1086 | 1160 | −312 | +12 | 1.29 | 0.063 | 0.061 | 0.13 |
| cpx+pl | 11587 | 1261 | −1112 | −68 | 0.76 | 0.032 | 0.033 | 0.45 |
| ol+opx | 7578 | 1515 | −5264 | −476 | 9.36 | 0.039 | 0.044 | 0.64 |
| ol+opx+cpx+sp | 10017 | 1253 | −9212 | −33 | 4.52 | 0.069 | 0.158 | 1.96 |
| opx+cpx+pl | 4800 | 1018 | −22 | −11 | 0.74 | 0.055 | 0.042 | 0.45 |

### Reading the results

* The optimiser works. Started at the truth, every target converges back to it:
  liquid RMSE 0.1–2 wt%, P within 0.8 kbar, T within 50 °C. From GEOROC
  starts, 71–100% of nodes end in the imposed assemblage. Their median
  cumulus-composition misfit is 0.05–0.3 wt% (0.4–0.9 wt% for the four-phase
  spinel lherzolites and the 1.2 ol+cpx target), and the median liquid
  fraction is 0.81–0.97.
* Misses are mostly degeneracy, not optimisation failure. In most cases the
  best GEOROC node reaches a loss equal to or lower than the truth basin's,
  but at a different (liquid, P, T). The emulator reproduces the cumulus
  compositions equally well at both points.
* In the SiO₂–MgO plane the degeneracy is a ridge. Along it, liquid MgO and
  SiO₂ trade off at roughly constant Mg# (the Fe–Mg exchange with olivine and
  pyroxene fixes the ratio, not the absolute level). P is typically the worst
  constrained quantity (accepted-node σ_P ≈ 0.6–3.4 kbar). T is better
  (σ_T ≈ 15–110 °C for multiply saturated targets, lowest when plagioclase
  is present).
* Constraint grows with the number of saturated phases. Single-phase olivine
  cumulates are unconstrained (only liquid Mg# is fixed). Three-phase
  plagioclase-bearing cumulates recover the liquid to about 1 wt% and T to
  about 10–50 °C.
* In three cases the truth basin has a lower loss than anything the 1024-node
  GEOROC ensemble found: 102 ol+cpx+pl (0.063 vs 0.068), 120 ol+cpx+pl (0.061
  vs 0.063) and 120 opx+cpx+pl (0.042 vs 0.055). Those are sampling misses,
  and larger ensembles (2^15) are the remedy.
* The loss evaluated exactly at the truth (bulk = true liquid, true P and T) is
  higher than either basin minimum (0.3–10). Two things cause this. At
  bulk = liquid the cumulus phases sit exactly on their saturation boundary
  (a ≈ 0), below the 1·T0 margin. And the emulator's own error means the
  imposed phases sometimes read as just absent there (`assemblage_ok` false
  in 9 of 14 targets). The truth basin's minimum lies a few hundred bar and
  some tens of °C away.

### 2^15-node run (102 opx+cpx+pl)

32768 nodes, 400 steps, batch 8192, 2-core CPU: liquidus initialisation 3.5 min,
optimisation 37 min (≈ 5.5 s/step; a GPU should be far faster). 97.1% of nodes
end in the imposed assemblage, with a median cumulus misfit of 0.23 wt%.
Accepted nodes = assemblage satisfied and loss in the lowest quartile (7956 nodes).

| | truth | best node | accepted p50 | accepted p05–p95 |
|---|---|---|---|---|
| P (bar) | 5908 | 5678 | 6935 | 5943–8703 |
| T (°C) | 1051 | 1080 | 1051 | 1021–1071 |
| liq SiO₂ | 56.46 | 54.79 | 60.57 | 55.2–67.1 |
| liq MgO | 1.41 | 1.53 | 0.97 | 0.76–1.40 |
| liq CaO | 4.79 | 3.93 | 4.19 | 3.44–4.79 |
| liq Al₂O₃ | 18.70 | 19.36 | 18.49 | 16.9–20.3 |
| liq K₂O | 1.25 | 1.41 | 2.18 | 0.94–4.43 |

The lowest-loss node sits near the truth (ΔP −230 bar, ΔT +29 °C). The
accepted cloud is a single ridge running from the truth toward high-SiO₂,
low-MgO, higher-P liquids. The ridge shows up in the correlations:
corr(P, SiO₂) = 0.90, corr(P, Al₂O₃) = −0.78, corr(SiO₂, Na₂O) = −0.91 and
corr(FeO, MgO) = 0.97 (constant Mg#). The density maximum at about 67 wt%
SiO₂ and 8.6 kbar is where GEOROC starts pile up on the ridge. It is not
evidence against the truth end.

## Files

* `src/ngibbs/engine/cumulate_inversion.py`: `load_ml_bundle`,
  `load_georoc_pool`, `find_liquidi`, `CumulateTarget` (`from_bundle`,
  `from_oxides`), `select_cumulate_rows`, `InversionConfig`,
  `CumulateInverter`, `InversionResult`.
* `scripts/cumulate_inversion_demo.py`: bundle-based validation (the tables
  above).
* `tests/unit_tests/test_cumulate_inversion.py`.

## Usage

```python
from ngibbs.engine.NN import rebuild_MELTS_model
from ngibbs.engine.emulator import NN_MELTS
from ngibbs.engine import cumulate_inversion as ci

em = NN_MELTS(rebuild_MELTS_model('.../102Closed_NoCr_NPT.tar'), cuda=True)
pool = ci.load_georoc_pool('data/MELTStables/GEOROC/GEOROC_PETDB_UNFILTERED_WHOLEROCK_VALIDATION.csv', em)

# measured cumulate (Oxides not carried by the model are ignored)
tgt = ci.CumulateTarget.from_oxides(em, {
    'olivine':       {'SiO2': 39.2, 'FeO': 17.5, 'MgO': 42.9, 'CaO': 0.3},
    'clinopyroxene': {...},
    'plagioclase':   {...},
})
res = ci.CumulateInverter(em, tgt, ci.InversionConfig(steps=600)).run(2**15, pool=pool)
acc = res.accepted(loss_quantile=0.25)
res.summary(subset=acc); res.covariance(subset=acc, correlation=True)
res.plot_density('liq_SiO2', 'liq_MgO', subset=acc, path='density.png')

# liquidi alone
out = ci.find_liquidi(em, pool.values, pressures=5000.0, composition_space='oxides')
```

## Open items

* Microprobe data give FeO-total: set `InversionConfig(combine_iron=True)` to
  compare FeO + Fe₂O₃ as FeOT.
* The accepted-node density depends on the starting prior. A likelihood
  temperature (`InversionResult.weights(temperature=...)`) or a proper sampler
  (e.g. Langevin steps around each node's optimum) would be needed to call it
  a posterior.
* Oxides the cumulate does not constrain keep their starting values. A weak
  prior toward a reference liquid, or a penalty on distance from the start,
  could be added if that is preferred to reporting them as unconstrained.
* The margins (1·T0) set how "near" near-liquidus is; smaller margins move
  the solution closer to the saturation surface, where the emulator is least
  certain about presence.
* Gated (`MidLevelNetwork`) checkpoints, open-fO₂ and isentropic models are not
  supported.
