# Benchmark: MELTS120Emulator

- created: 2026-11-20T02:44:40+00:00
- model dir: `/home/bthyer/nGibbs/src/ngibbs/engine/TrainedModels/120`
- git commit: 0b584f24a3352a062e710bbc7d3bfba3bf7e54b4, device: cuda

## Components

| component | status | detail |
|---|---|---|
| validationDataset/nocr | ok |  |
| validationDataset/cr | ok |  |
| standardWorkspace | ok | 6/6 simulations, assemblages {'nocr_isothermal': 2186, 'cr_isothermal': 1171} |

## Tables

| id | kind | file |
|---|---|---|
| validationDataset/nocr/isothermal | phase_quality | `validationDataset/nocr/emulator_quality_isothermal.csv` |
| validationDataset/cr/isothermal | phase_quality | `validationDataset/cr/emulator_quality_isothermal.csv` |
| standardWorkspace/nocr_isothermal | phase_quality | `standardWorkspace/emulator_quality_nocr_isothermal.csv` |
| standardWorkspace/cr_isothermal | phase_quality | `standardWorkspace/emulator_quality_cr_isothermal.csv` |
| standardWorkspace/property_errors | property_errors | `standardWorkspace/property_errors.csv` |
| standardWorkspace/phase_errors | phase_errors | `standardWorkspace/phase_errors.csv` |
