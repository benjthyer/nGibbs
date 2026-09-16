# Emulator vs real HeFESTo — Htz_transition

`data/HeFESToWorkspace/Htz_transition` is an isentropic HeFESTo run: harzburgite
(Mg/Si = 1.577 from the control file's element moles), S = 2.45 J/g/K held to
2.7e-10, 20–25 GPa in 501 steps of 10 MPa. That is the resolution the emulator
could not supply, so it is the first ground truth for transition **width**.

The emulator was evaluated at the same element moles, the same entropy and the
same pressures, and both were put through the same `d2Vs/dP2` window detector.

## Away from the transition — good

| | rms | max |
|---|---|---|
| T | 2.0 K | 9.9 K |
| rho | 0.34% | 2.3% |
| Vs | 0.42% | 2.5% |
| Vp | 0.23% | 1.4% |

No T-from-S excursions in this run. The 500 K excursions seen in the Mg/Si × Tp
notebook are absent here, so they are not universal — this profile is cool
(1773–1812 K) and Mg-rich, i.e. well inside the training distribution.

## Through the transition — not good

| | HeFESTo | emulator |
|---|---|---|
| window width | **2.00 km** | 0.50 km |
| ΔVs across window | **3.97%** | 1.90% |
| base depth | 659.6 km | 662.9 km |
| latent-heat deficit ΔT | **18.3 K** | 43.2 K |
| reaction-free dT/dP | 8.48 K/GPa | 8.17 K/GPa |

The emulator is 4× too sharp, recovers half the velocity contrast, puts the base
3.3 km too deep, and more than doubles the latent-heat deficit. All four follow
from the binary phase gate turning a finite divariant loop into a step.

## The number Weiyi asked about

HeFESTo's own isentropic post-spinel transition for this harzburgite is
**2.0 km**, not ~7 km. Subtracting the self-broadening term
|dP/dT|·ΔT/(dP/dz) with ΔT = 18.3 K gives the conductive limit:

| Clapeyron | self-broadening | conductive width | adiabatic width |
|---|---|---|---|
| −1.71 MPa/K | 0.77 km | 1.23 km | 2.00 km |
| −2.5 MPa/K | 1.13 km | 0.87 km | 2.00 km |

So the full Pe bracket for this composition is roughly **0.9–2.0 km**, entirely
inside the ≤2 km seismic constraint. Latent-heat broadening along an isentrope
does not put the 660 in conflict with a sharp discontinuity.

## Note on the window detector

The `d2Vs/dP2` max-then-min definition works cleanly on real HeFESTo — the
detector was never the problem. It failed in the notebook because the emulator's
second derivative is dominated by gate steps rather than by the reaction.

## Files
- `htz_compare.py` — loads fort.56, runs the emulator at matched composition/entropy
- `htz_figure.py` — the four-panel comparison
- `htz_results.json` — the numbers above
