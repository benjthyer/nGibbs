# 660-km sharpness brackets — nGibbs HeFESTo

## What this computes

The thickness of the ringwoodite → bridgmanite + ferropericlase transition in two
limits, for pyrolite (BSE, McDonough & Sun 1995) swept over mantle potential
temperature:

- **Pe → ∞** — the true isentrope. Latent heat cools the reacting interval; with a
  negative Clapeyron slope the cooling pushes completion deeper, so the transition
  self-broadens.
- **Pe → 0** — conduction resupplies the latent heat. Steady state with negligible
  advection solves d/dz(k dT/dz) = 0, i.e. T linear in depth matched to the
  reaction-free adiabat. The temperature deficit vanishes and the interval collapses
  to the intrinsic divariant width.

## Headline numbers (pyrolite)

| quantity | value |
|---|---|
| Clapeyron slope, fitted from the emulator's own rw-out surface | −1.71 MPa/K |
| latent-heat deficit ΔT across the loop | 25 K (Mp 1300) → 39 K (Mp 1700) |
| **isentropic self-broadening = bracket separation** | **1.1 – 1.7 km** |
| transition depth | 673 km (Mp 1300) → 643 km (Mp 1750) |
| depth = PREM's 660 km at | Mp ≈ 1475 K |
| Δρ across the post-spinel step / across 22–24 GPa | 5.1% / 8.4% |
| ΔVs across the post-spinel step | 6.0% |

**The isentropic self-broadening is ~1.5 km, not ~7 km.** The latent-heat term alone
does not push the 660 outside the seismically resolved ≤ 2 km, so an isentropic
mantle is not by itself in conflict with a sharp discontinuity.

## Known limitation — read before quoting a width

`binary_pred = (likelihoods > 0.5)` in `engine/NN.py` gates phase presence with a
hard threshold, so a phase leaves the assemblage discontinuously rather than
decreasing to zero. At Mp = 1600 K, 61% of the reaction occurs in two zero-width
steps (Fig. 2). Consequences:

- **Not measurable:** the absolute transition width, and any 10–90% velocity or
  impedance metric. Sampling finer does not help — the steps are true
  discontinuities, still zero-width at 0.2 MPa (≈5 m).
- **Measurable:** transition depth, total ΔVs / Δρ / ΔT — all endpoint differences
  of state functions, unaffected by how the change is distributed in pressure.

The bracket *separation* reported here is therefore computed semi-analytically as
|dP/dT| · ΔT / (dP/dz) from quantities that do survive, not by differencing two
measured widths. The intrinsic divariant width sits underneath it and needs HeFESTo
run directly at ~1 MPa spacing.

## Consistency checks

- entropy constant along each isentrope to 0.15%
- lower-mantle adiabatic gradient 7–8 K/GPa ≈ 0.30 K/km, matches αgT/c_p
- Clapeyron −1.71 MPa/K against experimental −2 to −3 MPa/K
- ΔT 25–39 K against the analytic bound −T·ΔS_rxn/c_p ≈ 50–60 K for complete
  olivine-component conversion (partial here — much of the olivine component has
  already gone to bridgmanite through the post-garnet reaction)
- Δρ 8.4% over 22–24 GPa against PREM's 9.3% at 670

## Figure 2 — two scales, both paths

`fig2_gate_diagnostic.png` shows the transition at 3.0 GPa (~74 km) and, ×10 in,
at 0.30 GPa (~7.4 km), with the isentropic (solid) and conductive (dashed) paths
overlaid on temperature, phase assemblage, Vp and Vs.

Both paths' assemblages and velocities are evaluated with the same isothermal
network at prescribed (P, T), so the solid–dashed difference is physics rather
than inter-network bias. The isentrope completes the reaction 0.062 GPa deeper
— **1.56 km** — which is an independent confirmation of the 1.4 km semi-analytic
self-broadening in Figure 1, arrived at by a different route.

The step in the thermal-path panel sits ~0.09 GPa above the gates in the other
panels because T comes from the isentropic network while assemblage and
velocities come from the isothermal one. That offset is the inter-network
decision-surface bias made visible, and is exactly why both paths in this figure
share a single network.

## Files

- `bracket_lib.py` — emulator wrappers, phase-field edge detection, PREM depth map
- `robust.py` — Clapeyron fit, ΔT sweep, self-broadening → `robust.json`
- `make_figures.py` — Figure 1
- `make_fig2.py` — Figure 2 (two scales, both bracket paths)
- `run_brackets.py` — first attempt, measuring widths directly. Kept because its
  control (isothermal net evaluated along the isentropic T path) is what exposed
  the gating: it returned 0.4–1.0 km where the isentropic net returned 2–4 km.
