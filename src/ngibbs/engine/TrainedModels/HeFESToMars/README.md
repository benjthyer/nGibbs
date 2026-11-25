---
model: HeFESToMars
api: HeFESToAPI
summary: 'HeFESTo emulator, Martian mantle: isothermal (P, T) and isentropic (P, S) with a temperature network'
license: agpl-3.0
---

# HeFESToMars: HeFESTo emulator for the Martian mantle

Emulates HeFESTo phase equilibria of Martian mantle compositions in the
elemental system Si-Mg-Fe-Ca-Al-Na-Cr-O (inputs as SiO2, Al2O3, FeO, Fe2O3, MgO,
CaO, Na2O, Cr2O3); `api.training_pressure_bounds()` gives the pressure range.
Predicts the stable assemblage, phase masses, phase compositions and component
moles, mass-balanced to the input bulk.

| File | Contents |
|---|---|
| `HeFESToMars_NPT.tar` | isothermal emulator (inputs P, T) |
| `HeFESToMars_NPS.tar` | isentropic emulator (inputs P, S) |
| `HeFESToMars_Temperature.pt` | temperature network for isentropic predictions |
| `*_NPT/NPS_Test_subset15000.tar.gz` | validation datasets, 15,000 HeFESTo states each from the test split |
| `standards/` | HeFESTo adiabats of three standard rocks (basalt, HTZ, shergottite), run along Martian P-T conditions |

## Use

```python
from ngibbs import make_ngibbs
api = make_ngibbs('HeFESToMars')            # newest version compatible with your nGibbs
api = make_ngibbs('HeFESToMarsv1.0')        # this version exactly
report = api.test()                    # score it on the validation datasets below
api.evaluate_workspace()               # ... on the standard runs below
api.evaluate_workspace('my_runs/')     # ... on your own HeFESTo output
```

## Citation and license

nGibbs, Ben J. Thyer, https://github.com/benjthyer/nGibbs (concept DOI
10.5281/zenodo.20766922). Released under the AGPL-3.0 license.
