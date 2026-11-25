# Benchmark: MELTS120Emulator

- created: 2026-11-24T04:58:54+00:00
- model dir: `/home/bthyer/nGibbs/src/ngibbs/engine/TrainedModels/120`
- git commit: 431a672609a2a2b9140c21ce3cb7a4bb353063a0, device: cpu

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
