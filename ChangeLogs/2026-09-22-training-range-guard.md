2026-09-22
- NN.read_training_bounds(): parse CONDITION / BULK COMPOSITION bounds
  out of a bundle's stats.txt.
- load_model_from_zip() attaches them as model.training_bounds
  (None when the bundle has no stats.txt).
- NN_MELTS: new TrainingRangeWarning + check_training_bounds(), run by
  forwardMB/forwardNN on every batch before normalization. Reports each
  out-of-range input: name, training range, #rows, value span, first row.
  Composition checked as wt% oxide, as in stats.txt. NaN/inf flagged.
- NN_MELTS warns at load if stats.txt and the feature normalizer disagree
  on a condition's range, or if the bundle has no stats.txt.
- reorder_input_table(): a missing CONDITION column now raises, even with
  strict=False. It was silently filled with 0.0 (fill_value).
- Found via the above: melts_comparison.py passed 'S(System_main)', but
  MELTS120 NPS expects 'S(System_main) / mass(System_main)', so every
  isentropic row ran at S = 0.
- melts_comparison.py: _system_main_feature() builds a condition column by
  the recipe in its feature name ('X / Y' reads X and Y, returns X / Y).
  Isentropic input is now S / mass under the model's own feature name.
- API.py: _ENTROPY_FEATURES tuple; parse_input and make_ptt_out now accept
  'S(System_main) / mass(System_main)' as the entropy feature.
- Result: isentropic phase-mass L1 error on the 6 MELTS standards fell
  from ~1.0-1.4 to ~0.01-0.02, on par with the isothermal pathway.
- API.get_T: removed the modelType == "MELTSAPI" unit conversion (bar->GPa,
  K->C). MELTS120 temperature models were predicting T 275-330 C too low.
  Now ~1 C MAE on the 6 MELTS standards.
- Removed the vestigial is_melts / --isMELTS flag everywhere (adiabat_utils,
  residual_workspace, train_temperature_residual_fcnn, Validate script,
  get_T). T_ref is always computed in the bundle's native units; no shipped
  checkpoint used is_melts=True. Cached residual workspaces rebuild once
  (fingerprint changed).
- melts_comparison.py: isentropic properties are evaluated at the
  temperature model's T (ForwardMB 'temperature'), not the GT T, as in
  HeFESTo's meltstable_comparison. Adds a T column to the isentropic
  property figure and 'T(K)' error stats to the isentropic rows.
