---
model: MELTS120
api: MELTSAPI
api_kwargs:
  melts_version: MELTS120
summary: 'rhyolite-MELTS 1.2.0 emulator: closed-system isothermal (P, T), Cr and NoCr variants'
license: agpl-3.0
---

# MELTS120: rhyolite-MELTS 1.2.0 emulator

Emulates rhyolite-MELTS 1.2.0 phase equilibria (training data generated with
alphaMELTS 2.3.1) for inputs SiO2, TiO2, Al2O3, FeO, Fe2O3, MgO, CaO, Na2O, K2O,
P2O5, H2O, CO2 and, in the Cr variant, Cr2O3. Inputs with any Cr2O3 are routed
to the Cr emulator, others to the NoCr one. Predicts the stable assemblage,
phase masses, phase compositions and component moles, mass-balanced to the
input bulk.

| File | Contents |
|---|---|
| `120Closed_{Cr,NoCr}_NPT.tar` | closed-system isothermal emulators (inputs P, T) |
| `*_Test_subset15000.tar.gz` | validation datasets, 15,000 MELTS states per emulator from the test split |
| `standards/` | alphaMELTS isobaric runs of three standard rocks (Bishop Tuff, lherzolite, MORB), Cr and NoCr |

## Use

```python
from ngibbs import make_ngibbs
api = make_ngibbs('MELTS120')            # newest version compatible with your nGibbs
api = make_ngibbs('MELTS120v1.0')        # this version exactly
report = api.test()                    # score it on the validation datasets below
api.evaluate_workspace()               # ... on the standard runs below
api.evaluate_workspace('my_runs/')     # ... on your own alphaMELTS output
```

## Citation and license

nGibbs, Ben J. Thyer, https://github.com/benjthyer/nGibbs (concept DOI
10.5281/zenodo.20766922). Released under the AGPL-3.0 license.
