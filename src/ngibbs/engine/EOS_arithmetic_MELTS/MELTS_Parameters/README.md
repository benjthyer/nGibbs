# sol_struct_data.json

Endmember standard-state thermodynamic reference data (H, S, V, Berman/Saxena
heat-capacity coefficients, Berman/Vinet equation-of-state coefficients),
extracted from `includes/sol_struct_data.h` in Paula Antoshechkina's MAGMA
fork of Mark Ghiorso's xMELTS repository
(https://github.com/magmasource/MAGMA), which is itself a fork/continuation
of the original MELTS (Ghiorso & Sack, 1995) source distribution.

Four tables are included, one per MELTS calculation mode:

- `xMeltsSolids`      — MODE_xMELTS (associated solutions, multiple liquids)
- `meltsSolids`       — MODE__MELTS (rhyolite-MELTS 1.0.x/1.1.x/1.2.x) —
                         **default table used by `melts_vec.params`**, since
                         this is the mode nGibbs's existing MELTS102/MELTS120
                         NN emulators were trained against.
- `meltsFluidSolids`  — MODE__MELTSandCO2 / MODE__MELTSandCO2_H2O
- `pMeltsSolids`      — MODE_pMELTS

Extraction method: `extract_sol_params.py` (kept alongside this file for
reproducibility) is a small brace/quote-aware tokenizer — not a full C
parser — that walks each `Solids <table>[] = { ... };` initializer and pulls
out `label`, `type` (`PHASE`/`COMPONENT`), `formula`, and the nested
`ThermoRef` struct (`h`, `s`, `v`, `cp_type` + coefficients, `eos_type` +
coefficients), skipping entries whose H/S/V are the all-zero dummy
placeholder used for solid-solution `PHASE` headers (those get their
properties from a mixing model elsewhere, not directly from a `ThermoRef`).
Units follow the source exactly: H in J, S in J/K, V in J/bar (equal to
`10 x cm^3/mol`), Cp in J/K, pressure in bars, temperature in K — see
`melts_vec/constants.py`.

The underlying numbers are the published Berman (1988) / Ghiorso & Sack
(1995) standard-state values as transcribed into MAGMA's source; several
entries also carry an inline provenance comment in the original header
(e.g. "left to parallel calcite (from Berman, 1988)") which this extraction
does not currently preserve. As with the HeFESTo parameter set already
vendored in `EOS_arithmetic/HeFESTo_Parameters_010123/`, check MAGMA's own
`LICENSE.txt` before redistributing this file outside the group.
