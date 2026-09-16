# Standalone continuous model

`ContinuousModel` no longer subclasses `MidLevelNetwork`. No lower model, so no feature
encoder feeding saturation logits and no prior-saturation head. Weights are not
interchangeable with the gated model.

## Naming and shape

`middleBrain` is now `self.encoder`, taking `encoderLayerUp` / `encoderLayerDown`.
`middleLayerUp` / `middleLayerDown` are gone.

```
core = cat([encoder(x), x])          # core_dim = encoder_out + input_dim
chem = softmax heads on core
g    = per-phase branches on cat([core, chem])
```

## Verified

```
chem_first : params 0.53M  core_dim 138  mole_in 196  resid 0.1133
parallel   : params 0.50M  core_dim 138  mole_in 138  resid 0.2912
no sat_head / middleBrain / prior_sat_head: True
save/load  : m identical True  recon identical True  head_order restored
double backward OK; grads on 178/178 params
continuity : 6 GPa / 3000 steps, max |d phaseMoles| = 1.3e-05
```

## Head order: chem first, and it is switchable

`head_order='chem_first'` (default) feeds `chem_out` into the mole heads; `'parallel'`
runs both off `core` alone.

The argument for chem-first is mass balance. Given the endmember fractions and the bulk,
the moles are nearly determined -- `A(chem) n = b` -- so handing `chem` to the mole heads
lets them approximate that solve rather than rediscover it. The reverse ordering has no
comparable justification: at equilibrium a phase's composition is set by chemical
potentials, not by how much of it is present.

Your worry that this trades assemblage quality for bulk residual does not apply, because
`chem` is **concatenated** rather than substituted -- a mole head keeps full access to
`core` and can ignore `chem` if it turns out unhelpful. And the two are coupled in your
favour here: the projector restricts corrections to the active support, which the *moles*
choose, so a better assemblage widens what the projection can reach. Improving the moles
is the right priority and chem-first serves it.

The one real cost is that every mole head now depends on every chem head, which enlarges
the double-backward graph for Sobolev training. Modest, and `'parallel'` is one config
key away if it bites.

## Reuse

`save` and `_set_indexer` are **borrowed by assignment** from `NN.py`
(`save = MidLevelNetwork.save`), not copied -- one implementation to maintain.
`save()` only touches `state_dict()`, `.config` and `.ml_indexer`, all of which this
class provides.

`load_model_from_zip` gains `model_class=None`, defaulting to `MidLevelNetwork`, so
existing calls are unaffected. `load_continuous_from_zip` is a three-line wrapper. The
zip layout is unchanged -- `state_dict.pt`, `config.json`, `ml_indexer/`, plus the
yaml/stats/log extras -- so packaging and deployment need no changes.

## One trap worth knowing

`MidLevelNetwork.save` runs the config through
`apply_type_conversions(..., default_dtype=str)`, so **any key absent from
`TYPE_CONVERSION_MAP` round-trips as a string**. The first load failed with
`'str' object cannot be interpreted as an integer` on `moleLayerUp`.

Rather than registering each new key in `constants.py`, the constructor coerces its own
arguments. That keeps the class loadable regardless of what the map knows, and means
adding a config later cannot silently break `load_continuous_from_zip`. Worth applying
the same habit to any new config on the gated model.

## Still open

`tuners.py` and `trainer.py` are untouched. Generalising `train_Upper_MELTS` /
`tune_Upper_MELTS` to an arbitrary NN class, and branching the architectural tuning on
`'Layer' in key`, is the remaining work -- I have not read either file, and rewriting a
tuning loop blind is how you get a silently mis-scored sweep.
