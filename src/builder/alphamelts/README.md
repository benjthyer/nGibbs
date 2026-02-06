# WSL MELTS Data Processing

This directory contains scripts to execute MELTS simulations through alphamelts (v2.3.1) using a ubuntu Windows Subsystem
for Linux (WSL) as well as to gather the generated data into tables (.csv files) and corresponding metadata (.txt files).
MELTS output data is gathered here.

## Directory Structure

```
wslMELTS/
├── batch/                # batch files (extentionless) used to send commands to alphamelts.
├── DataProducts/         # Tables (.csv files) and metadata (.txt files) of data gathered from alphamelts simulations
│   ├── 102/              # MELTS 1.0.2 data
│   ├── 110/              # MELTS 1.1.0 data (etc.)
│   ├── GEOROC/           # GEOROC and PetDB whole rock compositions, randomly selected to generate nMELTS training data
│   └── MORB/             # MORB (and other names aside from p, 102, 110, 120) are analogous MELTS data gathered from such rocks
├── engine/               # Python scripts to build and exectute MELTS simulations with alphamelts in WSL
│   └── alphamelts-app-2.3.1-linux      # This directory contains the alphamelts installation
├── Workspace/            # Folder where alphamelts output is collected
└── README.md             # This file
```

## Engine Scripts

- [src/builder/wslMELTS/engine/alphamelts_functions.py](src/builder/wslMELTS/engine/alphamelts_functions.py) orchestrates ensemble runs. Key pieces:
	- `forward_ensemble(...)` builds per-simulation workspaces, writes `input.melts`, copies the requested batch file, and executes alphamelts via GNU parallel on WSL (Windows execution is deprecated). Takes compositional (and PTfO2) as inputs
	- `import_MELTS_components(...)` ingests alphamelts tables into a consolidated CSV using `DatasetIndexer`, handles pMELTS quirks (e.g., corundum component), and balances rows by downsampling super-liquidus states.
	- `pick_exsolution_failure(...)` records conditions that produced multiple occurrences of the same phase or outright failure to help debug issues. Seldom used, as multiple instances of the same phase can be forbidden directly within alphamelts (using batch files)

- [src/builder/wslMELTS/engine/melts_file_builder.py](src/builder/wslMELTS/engine/melts_file_builder.py) builds MELTS input strings:
	- `makeMELTSStr(...)` converts condition arrays and headers into input .melts files, supports fractional crystallization or batch melting/crystallization and compression runs. Isobaric only as of v0.0.0
	- `suppressAllBut(...)` appends suppression lines to keep only specified phases.
	- `expand_MC(...)` jitters condition vectors with Gaussian noise for quick Monte Carlo style sampling.

- [src/builder/wslMELTS/engine/RandomMelters.py](src/builder/wslMELTS/engine/RandomMelters.py) is a driver for generating training datasets from GEOROC compositions:
	- `alphaMELTScooling(...)` runs cooling paths (pressure range depends on MELTS model), normalizes and sanitizes GEOROC oxide arrays, and streams results into `<output_file>.csv` plus matching metadata.
	- `alphaMELTScompress(...)` runs compression paths (not implemented for pMELTS as of v0.0.0), varying pressure while keeping temperature fixed per run.
	- Both functions validate any existing output files to ensure headers/row counts match before appending.

## Prerequisites

- alphamelts 2.3.1 installed at `engine/alphamelts-app-2.3.1-linux` (already vendored here).
- GNU parallel available inside WSL Ubuntu (used by `forward_ensemble`).
- Python dependencies from the project `requirements.txt` (includes pandas, numpy, etc.).

## Typical Workflow

From a Python session, call a driver such as:
	 ```python
	 from src.builder.wslMELTS.engine.RandomMelters import alphaMELTScooling
	 alphaMELTScooling(
			 output_file="MELTS102Batch",
			 MELTSModel="102",
			 GEOROC=georoc_array,
			 col_dict=oxide_to_col_index,
			 indexer=dataset_indexer,
			 iter=10,          # number of iterations
			 simcycle=50,      # simulations per iteration
			 fxtal=False       # set True to fractionate solids
	 )
	 ```
Outputs appear under `engine/Workspace/` during execution; consolidated results append to `<output_file>.csv` and `<output_file>.txt` in the working directory.

## Notes and Tips

- Batch files live in `batch/` and should match the MELTS model you request (e.g., `102batch`).
- `forward_ensemble` cleans the Workspace before each call; ensure you have copied out any prior runs you wish to keep.
- Cooling runs use a default end temperature of 700 °C (1000 °C for pMELTS); compression runs set pressure start/stop automatically when `end` is not provided.
- DatasetIndexer object reads phases and elements implied by columns and values of MELTS tables. 
- To restrict allowed phases, pass `only_phases` (list of MELTS phase names) through to `forward_ensemble` or via the `indexer.get_phase_list()` used in the drivers.



## References

- **MELTS**: Ghiorso & Sack (1995), Asimow & Ghiorso (1998)
- **nMELTS**: Antoshechkina & Ghiorso (2014)
- **Machine Learning Integration**: Custom pipeline in `src/builder/processing/`

## Contact

For questions about data processing or this directory, see the main repository README.
