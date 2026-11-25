---
model: HeFESToEarth
api: HeFESToAPI
summary: 'HeFESTo emulator, Earth mantle: isothermal (P, T) and isentropic (P, S) with a temperature network'
license: agpl-3.0
---

# HeFESToEarth: HeFESTo emulator for the Earth's mantle

Emulates HeFESTo phase equilibria of Earth mantle compositions in the
elemental system Si-Mg-Fe-Ca-Al-Na-Cr-O (inputs as SiO2, Al2O3, FeO, Fe2O3, MgO,
CaO, Na2O, Cr2O3), over about 0-141 GPa (`api.training_pressure_bounds()`
gives the exact range). Predicts the stable assemblage, phase masses, phase
compositions and component moles, mass-balanced to the input bulk.

| File | Contents |
|---|---|
| `HeFESToEarth_NPT.tar` | isothermal emulator (inputs P, T) |
| `HeFESToEarth_NPS.tar` | isentropic emulator (inputs P, S) |
| `HeFESToEarth_Temperature.pt` | temperature network for isentropic predictions |
| `*_NPT/NPS_Test_subset15000.tar.gz` | validation datasets, 15,000 HeFESTo states each from the test split |
| `standards/` | HeFESTo adiabats of three standard rocks (basalt, DMM, silica-rich), run along Earth P-T conditions |

## Use

```python
from ngibbs import make_ngibbs
api = make_ngibbs('HeFESToEarth')            # newest version compatible with your nGibbs
api = make_ngibbs('HeFESToEarthv1.0')        # this version exactly
report = api.test()                    # score it on the validation datasets below
api.evaluate_workspace()               # ... on the standard runs below
api.evaluate_workspace('my_runs/')     # ... on your own HeFESTo output
```

## Citation and license

nGibbs, Ben J. Thyer, https://github.com/benjthyer/nGibbs (concept DOI
10.5281/zenodo.20766922). Released under the AGPL-3.0 license.
