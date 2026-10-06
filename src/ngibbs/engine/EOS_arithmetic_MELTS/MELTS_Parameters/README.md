# sol_struct_data.json

Endmember standard-state thermodynamic reference data (H, S, V, Berman/Saxena
heat-capacity coefficients, Berman/Vinet equation-of-state coefficients),
extracted from `includes/sol_struct_data.h` in Paula Antoshechkina's MAGMA
fork of Mark Ghiorso's xMELTS repository
(https://github.com/magmasource/MAGMA), which is itself a fork/continuation
of the original MELTS (Ghiorso & Sack, 1995) source distribution.

Four tables are included, one per MELTS calculation mode:

- `xMeltsSolids`      — MODE_xMELTS (associated solutions, multiple liquids)
- `meltsSolids`       — MODE__MELTS (rhyolite-MELTS 1.0.2) — default table
                         of `melts_vec.params.load_solids`.
- `meltsFluidSolids`  — MODE__MELTSandCO2 / MODE__MELTSandCO2_H2O
                         (rhyolite-MELTS 1.1.0 / 1.2.0): meltsSolids minus
                         `water`, plus the Duan fluid endmembers and the
                         carbon phases (calcite ... graphite, diamond).
                         MELTSAPI picks the table from its `melts_version`
                         (`melts_vec.liquid_modes.SOLID_TABLE`).
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

# liq_mode_wij_data.json

Liquid interaction-parameter tables for each rhyolite-MELTS version, used by
`melts_vec/liquid_modes.py` (extracted by `extract_liquid_mode_wij_params.py`):

- `MELTS102` — `includes/param_struct_data_v34.h` `meltsModelParameters`
  (19 components). This is the table liquid_v34.c's WH() macro reads in
  MODE__MELTS. It differs from `param_struct_data.h`'s
  `originalModelParameters` (the source of `liq_wij_data.json`) in the
  eleven nonzero CO2 pairs; the MAGMA library itself confirms the v34 table.
- `MELTS110` — `param_struct_data_CO2.h` `meltsAndCO2ModelParameters`
- `MELTS120` — `param_struct_data_CO2_H2O.h` `meltsAndCO2_H2OModelParameters`

1.1/1.2 have 20 species (the 19 basis components + CaCO3) and one H/S/V
adjustment per species (all zero). Entries are positional (pair n = i<l, i
outer); the C labels are kept as `W_labels` and checked on extraction.
