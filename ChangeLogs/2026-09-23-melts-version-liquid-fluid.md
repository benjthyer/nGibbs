2026-09-23
- melts_vec: new liquid_modes.py. MELTS-version switch 'MELTS102' /
  'MELTS110' / 'MELTS120' for the liquid: table (meltsLiquid vs
  meltsFluidLiquid), W table, and gibbs.c's special cases, all verbatim:
  SiO2 (glass transition at 1480 K), H2O (1.0.2/1.1 phiP volume; 1.2
  Ochs & Lange volume + a/b shifts), CO2 (Duan 1 bar + offsets) and CaCO3.
- 1.1/1.2 liquid: 20-species regular solution with CaCO3 speciation
  (CaSiO3 + CO2 = CaCO3 + SiO2), solved per row; Cp, dV/dT, dV/dP and
  dCp/dT include the speciation terms.
- 1.0.2 W table is now param_struct_data_v34.h's meltsModelParameters (what
  liquid_v34.c reads in MODE__MELTS). liq_wij_data.json
  (originalModelParameters) differs in the CO2 pairs.
- New MELTS_Parameters/liq_mode_wij_data.json + its extraction script.
- melts_vec: new water.py (whaar / wdh78 water; 1.0.2 'water' phase) and
  fluid_duan.py (Duan & Zhang H2O-CO2 fluid of 1.1/1.2, T/P derivatives
  by torch autograd). compute_water_phase / compute_fluid_solution.
- Carbon solids (calcite ... graphite, diamond) come from the
  meltsFluidSolids table; checked exact against MAGMA.
- All of the above match the MAGMA C library to ~1e-9 or better
  (melts_vec/tests/benchmark_liquid_modes_test.py, frozen oracle output;
  melts_oracle.c + make_liquid_modes_oracle.py regenerate it).
- MELTSAPI: new melts_version argument (default: inferred from the
  model_dir name, e.g. TrainedModels/120 -> 'MELTS120'; raises if it
  cannot be inferred). Selects the solid table, liquid model and fluid.
- get_property_melts_vectorized_from_assemblage: liquid uses
  compute_liquid_bulk_mode; new 'fluid'/'water' phase ((B, 2) [H2O, CO2]);
  absent-phase rows get a placeholder composition, zero composition with
  nonzero moles raises; CO2 in a MELTS102 fluid raises.
- melts_comparison.py: the GT fluid is no longer skipped (fluid.tbl
  h2oduan/co2duan, or pure H2O for 1.0.2; mismatch with melts_version
  raises). Emulator bridge adds the fluid (from its H2O/CO2 wt%) and the
  carbon pure phases.
- Result (120 standards, GT assemblage + internal EOS): liquid H/S/V/Cp
  errors <=0.01% (were up to -12% V, -16% S), fluid exact. Bulk mean errors
  now <=0.07% except BishopTuff Cp (-0.33%, feldspar Cp, still open).
  Emulator pathways: BishopTuff Cp 3.1% -> 0.3%, V 1.4% -> 0.05%;
  MORB V 1.6% -> 0.3%, S 0.7% -> 0.05%.
- tests: benchmark_bulk_rho_test's bare MELTSAPI sets the new attributes.
