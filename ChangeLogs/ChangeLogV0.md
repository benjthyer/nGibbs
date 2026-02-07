_____________________ Unreleased _______________________

2026-02-06 

Commit: 
- Added training scripts under src/builder/training with config, data, design, trainer, and tuning modules.
- Added training .yaml config capability and a template config/training.yaml (including separate validation bundle pointer).
- Refactored MidLevelNetwork to accept and persist ml_indexer
- Add config file to ML bundle tarballs during creation, preserving the original filename.
- Add bundle sanity check helper for ml_indexer consistency, bulk reconstruction, and row-count alignment, plugged into bundle ops
- Allow ChangeLogs directory to be tracked by Git by adding an explicit exception to .gitignore.
- Remove the later *logs/ ignore rule so the ChangeLogs exception is not overridden.

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