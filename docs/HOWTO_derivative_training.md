# Processing and training a derivative dataset, end to end

Worked against `HeFESTo_Trainset082026EarthAdiabats` / `_Validset...` in
`data/MELTStables/HeFESTo`. Every command below was validated on the real
`HeFESTo_demoset_derivs` table in that directory, which already has its sidecars.

---

## 0. What you need on disk

Per split, three files sharing a base name:

```
HeFESTo_Trainset082026EarthAdiabats.csv          the BigMetaTable
HeFESTo_Trainset082026EarthAdiabats_dndP.csv     sidecar, 2 + n_components columns
HeFESTo_Trainset082026EarthAdiabats_dndT.csv
```

**Both splits need sidecars**, not just Train. `BigMetaTable.__init__` attaches them from
the original base name, and a split without them exports a bundle without derivative
arrays — which then fails the `derivatives.enabled: true` gate, correctly but late.

Written by `scripts/import_hefesto_subdirs_derivs.py` alongside the main import.

## 1. QC the derivatives first

```
python scripts/check_derivatives.py <HeFESTo workspace dir> --chain
```

Costs nothing and is sharp — the clean value of its primary test is machine precision
(7e-8), not a tolerance. Three outcomes:

- **ok** — proceed.
- **out-of-domain** — internally consistent, but the run spends time in phases the trained
  model will not carry. Report gives the row fraction; usable, worth `row_weights`.
- **CHECK** — unexplained residual or a non-injective species map. Fix before exporting.

## 2. Export the bundles

`prepareML.py` builds names as `{Model}_{Split}set{Date}{Mode}` and emits bundles named
`{stem}_{Train,Test,Valid}.tar.gz` where `stem = HeFESTo082026EarthAdiabats`:

```
python src/builder/processing/prepareML.py --MELTSModel HeFESTo \
    --Date 082026 --Mode EarthAdiabats
```

**Two things will bite you here.**

*Naming.* `main.py` loads `<tarname>_Train` and `<tarname>_Test` — `_Valid` is produced but
never read. With Train/Valid inputs and no Testset, either name your validation CSV
`HeFESTo_Testset082026EarthAdiabats.csv`, or export the two splits individually and name
the outputs yourself:

```
python src/builder/processing/export_only.py \
    --input_path data/MELTStables/HeFESTo/HeFESTo_Trainset082026EarthAdiabats \
    --name HeFESTo082026EarthAdiabats_Train
```

*featureNames.* `export_only.py` and `resampling_to_datasets` default to the MELTS pair and
raise `KeyError: Feature Pressure(System_main) not found in MELTS_indices` on a HeFESTo
table. Set them to `['P(GPa)(System_main)', 'T(K)(System_main)']` — in `processing.yaml`
for the `prepareML` route, or pass `featureNames=` if you call the exporter directly.
(`export_only.py` has no flag for this today; it is a two-line addition if you use that
route often.)

**What a successful export prints:**

```
[table1] aliasing self.table1 -> self.table (identity multipliers, HeFESTo)
[derivatives] exporting ['dndp', 'dndt'] on the component axis (68 columns each)
[derivatives] coverage 99.6% of rows; scale median 1.932e-09
```

`68`, not 62 — the component count is dataset-dependent. This table excluded 11 zero-sum
components including `magnetite(spinel)`, which is exactly the abundance threshold at
work: the derivative tables carry every species HeFESTo knows, the trained model carries
only what clears the threshold, and `check_derivatives.py` reports the difference rather
than confusing it with a fault.

Verify before training:

```python
import tarfile
print([n for n in tarfile.open('..._Train.tar.gz').getnames() if n.endswith('.npy')])
# must include dndp_labels.npy, dndt_labels.npy  (+ derivative_stats.json)
```

## 3. Train

Copy `recipes/training/HeFESToContinuousDerivs.yaml`, set `tarname`, then:

```
python src/builder/training/main.py train \
    --config recipes/training/HeFESToContinuousDerivs.yaml
```

The recipe's `train1` is a 3-epoch, `max_N: 2e5` smoke episode. Look for these lines before
letting `train2` run:

```
[derivatives] enabled: {'dndp_alpha': 1.0, ..., 'project_tangent': True, ...}
Derivative directions: feature[0] (dn/dP, range 140), feature[1] (dn/dT, range 2700)
Derivative scale estimated from N rows: median ..., max ...
[TRAIN] dn/dP 8.2e-03   dn/dT 1.4e-02
```

If `[derivatives] enabled` is missing, the episode is running `train_Upper_MELTS` and the
derivative arrays are doing nothing.

Three settings that are not arbitrary, all in the recipe's trailing comment:
`which_heads_to_freeze: []` (the `upper` strategy would freeze `mole_head`, which is what
the derivative loss trains), `layernorm` over `batchnorm` (forward-mode AD makes a
batchnorm tangent batch-coupled), and dropout at 0 (each direction draws its own mask,
straight into the derivative target).

---

## Would dn/dS help the isentropic model?

**It is not optional — it is required, and the isentropic model cannot use derivative
training at all without it.**

```
isothermal_emulator   featureNames = ['P(GPa)(System_main)', 'T(K)(System_main)']
isentropic_emulator   featureNames = ['P(GPa)(System_main)', 'S(J/g/K)(System_main)']
```

The loss differentiates the network with respect to *its own inputs*. An isentropic
network has no `T` input to differentiate, so `sobolev._feature_index(model, 'T(',
'Temperature')` raises `KeyError` — honestly, but it means the isentropic path is
currently blocked.

**The good news: no new HeFESTo output is needed.** The main table already carries `S`,
`cp`, `alpha` and `rho`, so the coordinate change is exact and local:

```
dS/dT|_P = cp / T
dS/dP|_T = -alpha * V            (Maxwell), V = 1/rho
                                  units: alpha[1e-5/K] / rho[g/cm3] * 1e3  ->  J/(g K GPa)
                                  i.e.  dS/dP|_T = -(alpha * 1e-2) / rho

dn/dS|_P = (dn/dT|_P) * T / cp
dn/dP|_S = dn/dP|_T + (dn/dT|_P) * (alpha * V * T / cp)
```

That second line is the adiabatic gradient `dT/dP|_S = alpha*V*T/cp` — the same expression
the bracket exercise used, so the two are consistent by construction.

**Is it more information?** No — the Jacobian in (P,S) is the (P,T) Jacobian times an
invertible 2x2. Same content, right coordinates. But two things make it better than a
formality:

1. **It is the quantity you actually care about.** `dn/dP|_S` is what sets the isentropic
   width of the 660. Supervising it directly beats supervising two partials and trusting
   their combination — which is the chain-rule identity `check_derivatives.py --chain`
   already verifies (median ratio 0.9999 on `Htz_transition`).
2. **It is numerically better behaved.** At a transition `cp` spikes with latent heat, so
   `dn/dS = (dn/dT)·T/cp` is *bounded* exactly where `dn/dT` blows up. The Huber clipping
   would bite far less, and the loss would stop being dominated by the handful of rows it
   currently has to clip.

**One thing to verify first, and it is load-bearing.** `cp` must be the **equilibrium**
heat capacity, including the latent-heat term `cpmet = T * sum_i (dn_i/dT) * S_i`. If the
table's column is the frozen, fixed-assemblage `cp`, then `dT/dP|_S` is wrong precisely at
a phase transition, where the two differ by a large factor — and that is the only place any
of this matters. `hefesto_vec/metamorphic.py` computes `cpmet` as a separate term, which
hints the base may be frozen.

The decisive test is to compare `alpha*V*T/cp` from the table columns against `dT/dP`
finite-differenced along an actual isentrope. I ran it on `HeFESTo_demoset_derivs` and got
a per-segment ratio of `[0.35, 0.80, 1.75, 1.55]` — **inconclusive**, because that table
samples adaptively around phase changes and `np.gradient` cannot resolve `dT/dP` on an
irregular grid. Run it on a smooth, finely-spaced adiabat (`somesims/Simulation2` has 29)
before trusting the conversion. If the ratio comes back at 1.00, `cp` is the equilibrium
value and `dn/dS` is a ~40-line addition to the exporter: two more sidecar-derived arrays
computed from columns already in the table, riding the same plumbing.
