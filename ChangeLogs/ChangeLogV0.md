_____________________ Unreleased _______________________

2026-04-10: Made HeFESTo importer recursive. 
- Also high level python orchestration of SLURM calls for subdirectories (to limit sizes of jobs to 1000)

2026-04-09: Made phase-boundary HeFESTo tree builder.
- Limited to 1000 staged simulations per directory, then moves to a new directory.

2026-04-09: Fixed .gitignore for builder alphamelts linux engine directory.
- Ignored src/builder/alphamelts/engine/linux_alphamelts_1_9/.

2026-04-09: Added phase-boundary extractor in import_HeFESTo function to subsample interesting regions near phase changes
- import_HeFESTo_components() now accepts phase_change_dataname.
- Writes full rows to a 2nd CSV when phase mass toggles zero/non-zero.
- Boundary export includes each transition row and prior row.
Cluster ready import-HeFESTo:
- Deletes control-only Simulation dirs (did not run); and removes fort.29 and qout which are memory intensive. 

2026-04-07: Got alphamelts 1.9 working, prep for phMELTS. (Decided to deprioritize until after thesis)
- Added plain "feldspar" entry into projection matrices to support 1.9
- realized that I need a generalizable thermodynamic orchestration pipeline that builds directories, 
moves/generates settings/imports, then runs general function. Can greatly simplify existing pipeline.

2026-04-05: Completed HeFESTo adiabat tree writer.
- Edit copied control file
- Wrote ad.in into each simulation directory with the requested path.

2026-04-05: Generated Adiabat sims to direct roughly adiabatic HeFESTo sims on cluster.
- getting noisy PT from rough adiabat from reference calculations
- Added scripts/fit_adiabat_temperature_regression.py CLI tool.


Commit: ae372a9738d9b58959e4318ae953644dec9456c5
2026-04-04: Added single-run HeFESTo simulation launcher.
- Added forward_HeFESTo_single() for one SimulationN run.
- Added CLI switch wrapper for paths and simulation id input.
- Copies full template directory into SimulationN before run.

2026-03-11: Improved recoveryplot.py binary metrics readability.

2026-03-05: Added oxide-space coverage validation script.
Pursuing issue where the predicted phase assemblage is insufficient to form a basis that covers the composition. 
Should force saturation of highest scoring phase that contains
- Added scripts/check_phase_oxide_coverage.py CLI tool.
Phase weighting now supported through YAML!
- Training now supports separate episode binweights and compweights maps.
- main.py builds [1,P] binary and [1,VC] component weights from config.
- Lower trainer now applies phase-weighted BCE loss each epoch.
- Lower trainer now prints phasewise test precision and recall by epoch.

2026-03-04: HeFESTo control template and run code flexibility.
- forward_HeFESTo now accepts 1D run_code applied to all simulations.
- forward_HeFESTo now accepts 2D run_code with one row per simulation.
- Added control_template arg; default is shallowHeFESTo in batch dir.

2026-03-03: Training with constant iron and arbitrary features!
- Added scripts/merge_bigmetatables.py CLI for two-table BigMetaTable merges.
- Added forward_HeFESTo() to build and run SimulationN ensembles with gnu parallel
- HeFESTo input comps are uniformly spaced 3Fe/Fet between 0-0.1. (Too high?)
- Wired script to run forward_HeFESTo() and import_HeFESTo_components().
- HeFESTo is too slow on my personal machine: I estimate it will require ~60 cpu days to compute 4E6 assemblages. 
- **Removed Molmass dependency with hard-coded oxide molar masses**
- Training Changes: 
    - Fixed bug where loss was not normalized correctly in trainers.py
    - Fixed bug where tuners wouldn't properly test values
    - Now best_loss is recalculated at the beginning of a new tuning episode by default
    - Warm-start model config now overwrites global config baseline in training main. 
- Merged Oxygen Dev back to main! 

2026-02-27: HeFESTo!
- Updated README
- Added HeFESTo phase->species name mapping
- Added generate_column_headers_hefesto() for HeFESTo phase header generation.
- Made HeFESTo header generator full-name only; abbreviations handled in parser.
- Added HeFESTo workspace parser for control/fort.56/.61/.68/.99 -> CSV.

2026-02-26
- Removed assumption that liquid is last phase (or there at all).
- Beyond PTX: Replaced hardcoded feature offsets with len(ml_indexer.featureNames).

2026-02-25
- Removed test forcing compToOx to have same rows as PsSp transform
    - This transform will not be applied to HeFESTo data.
**Will need separate loading script for HeFESTo data.**
- Fixed tuning logic bug where WD/Noise parameters could be skipped in certain cases
2026-02-23
- Tested model molar_epsilon 1E-3. Not as effective as linear softmax it seems 
    - Fixed overwriting bug for molar epsilon by high level NN_MELTS object
- ***Adaptive dropout now treats regularization config value as upper limit, not initial rate.***
- Added CLI Phase Diagrams validation tool. Buggy. 

Commit: 12a4a971804a360e39f30129038319c8de70ccca
2026-02-20
- Training Debugging
- Stabilized Bulk loss (cannot use dropout at all when using bulk loss...)
- Selectively apply weight decay to active parameters only, and not to mole heads. 
    - Low and High models now can use distinct weight decay
- Excluded mole_head and chem_heads from adaptive dropout updates.

2026-02-16
- Adapted 1:1 Recovery Plots. Old bugs still present (some phases do not plot missed assemblages at all)
- Refactored NN_MELTS Emulator Class to use ml_indexer Architecture
- Added bundle_insanity_filter() to filter out bad runs from MELTS. Req'd to be less than 0.1% of data
- Added Cr/NoCr split workflow to separate chrome


Commit: 9760356478a2072402fcb8275276c1e9d4cd3d28
2026-02-13: Training focus
- Added/Debugged Detailed model metadata inclusion with .tar files
- Removed automatic .update_table() calls to ml_indices for more granular control of indexer 
- Implemented log normalized molar predictions
- Debugging training.

2026-02-12
- **MLIndexer and Normalization Save State, not Pickled objects**
    - Added `MLIndexer.save(directory)` method that exports all state to JSON and NPZ files
    - Implemented `load_ml_indexer_from_state(state_dict)` function to reconstruct MLIndexer bypassing `__init__`
    - Serialization includes: metadata, structure mappings, transformation matrices, normalizer states

- **NN Models Now Stored in Zip Packaging** - Complete model state bundling with metadata
    - Zip contents: state_dict.pt, config.json, ml_indexer/ (full state), metadata.json, optional model.yaml, training.yaml, stats.txt, log.txt


- **Sequential Training/Tuning Orchestration** - Multi-episode episode-based training from YAML
    - Implemented `_discover_episodes()` for automatic discovery and ordering of tune1, train1, tune2, train2, ... episodes
    - Episode-based execution: Sequential numbered train/tune commands execute. Before only one of each allowwed! 

- Updated spinel and orthopyroxene correction methods in NN.py to use name indexers and not hard coded positional indexers, which now are meaningless after archetecture made flexible
- Refactored MLexporter feature parsing to accept MELTStable column-style strings and wired featureNames/freeOutputs from processing .yaml into dataset generation.
- Updated tune_Lower_MELTS to require Model instance and pass ml_indexer to all new MidLevelNetwork instantiations for consistency.
- **nMELTS Package Isolated from builder/ dir for pip distribution**: 
    - Moved config/settings.py out of src/nMELTS/ to config/ directory (builder-only)

2026-02-11
- Training now by .yaml configuration. Now accepts pytorch lr schedulers.
- Branched off training and optuna work onto trainDev branch. Focus on getting old code working/adapted on main, maybe revisit optuna later. 
    - Painfully slow, very little feedback for me to know what was going on
- Debugged Training! Mostly. 
- Added training logger to capture all training/tuning output to timestamped log files in src/builder/training/logs directory.

    
Merge Commit onto main : 950b8e4e675afd6e958537453365b1f8e663f54e
2026-02-10 
- Debuged resampling tests -- now functioning
- sanity_check_bundle to export bundle arrays to CSV only when assertions fail (not unconditionally).
- Add filter_inconsistent_phase_data() to BigMetaTable to detect and remove rows with zero mass but non-zero attributes.
- Refactor loadTrainData into load_train_data() to build torch datasets from bundle paths.
- Wire training main CLI to resolve bundle names from training.yaml and run lower/upper/finetune stages.
- 

2026-02-09
- Added BigMetaTable phase proportion filter that removes rows containing underrepresented phases and refreshes the indexer after filtering.
- Added BigMetaTable filter to drop rows containing phases not present in the ml_indexer for train/validation consistency.
- Added optional bundle_name parameter to MLexporter.resampling_to_datasets for renaming output tarballs.
- Added detailed failure diagnostics to ML bundle sanity checks (counts, indices, and worst values).

2026-02-06 

Commit: 88899b57a2105940337b437beb03325c133f9ef3
- Added training scripts under src/builder/training with config, data, design, trainer, and tuning modules.
- Added training .yaml config capability and a template config/training.yaml (including separate validation bundle pointer).
- Refactored MidLevelNetwork to accept and persist ml_indexer
- Add config file to ML bundle tarballs during creation, preserving the original filename.
- Add bundle sanity check helper for ml_indexer consistency, bulk reconstruction, and row-count alignment, plugged into bundle ops


2026-02-05  

Commit: 9952463ada64e54f1b1b5dedd8070a81a636a189
**(!) Random Melters now keep track of alphamelts progress with temporary text file. iter arg does not exist any longer, now itercode in form 'a15'**
- Add `max_liquid_fraction` parameter to `import_MELTS_components()` function for memory management of large data products.
    - Default value of 100 retains previous behavior of superliquidus subsampling
    - When set below 100, filters out all rows where liquid mass exceeds the specified threshold
    - Motivation: Enable more efficient storage and processing by excluding high-liquid-fraction conditions that may not be relevant for certain analyses
- Add pMELTS oxide exclusion logic to avoid errors by defining ZeroOxides and passing mode/zeroOxides into generate_column_headers.
This excludes NiO and MnO at all times, avoiding errors because these oxides are not found in the Bulk_comp table.
- Adjust alloy-liquid component names to Fe-liquid/Ni-liquid in constants and projection tables to match MELTS output naming
- Add .github/ to .gitignore to keep agent instruction files out of the repo (codebase-tailored instructions).
- Indexer now build Elkeys from bulk_comp columns, to avoid missing absent oxides when excluding components. Unsure if completely necesary. 
- Processing config defines name of output folder, blank is same behavior as before. 


2026-02-04

- Fixed issue where DatasetIndexer() object phase/component exclusions were affected by previous instantiations. 
    - Unclear why this happened in the first place. 
    - Removed exclusion input; now relies on columns and analysis of tables alone to build dynamic indexers. 
    - Updated README.md to follow
- Generated Baseline README.md file at top level directory
- Expanded BigMetaTable's ability to look for column names to generate indexer (& ml_indexer) objects so the .split() method doesn't fail
- Add more detailed failure mode descriptions for import_MELTS_components() alphamelts compiler function, assert less than 50% failure rate.
- Data Processing pipeline now relies on .yaml configuration files, which are carried by .tar.gz files containing the data.

-Verified with tests that dynamic indexing and export logic works as intended! Data processing tests remain incomplete, need to be redesigned not to rely on exporting tests, which intercompare MELTStables and output bundles. This can be done only if the BigMetaTable used in testing is instantiated on the final filtered MELTStable .npy file, rather than the unfiltered version. Other option is to exclude ALL filtering with override in .yaml file. 

- Added logging of printed info for tests

**COMMITED CHANGES AND PUSHED TO REMOTE** dfdd107e70c25b0276d9b06e47e87fb2eb0fbb9a