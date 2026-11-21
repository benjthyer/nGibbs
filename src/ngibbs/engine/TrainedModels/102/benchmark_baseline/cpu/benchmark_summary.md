# Benchmark: MELTS102Emulator

- created: 2026-11-21T00:32:55+00:00
- model dir: `/home/bthyer/nGibbs/src/ngibbs/engine/TrainedModels/102`
- git commit: 3301474d65bcd152ecb74d5bb5d0e9355fd96ae3, device: cpu

## Components

| component | status | detail |
|---|---|---|
| validationDataset/nocr | ok |  |
| validationDataset/cr | ok |  |
| standardWorkspace | ok | 6/6 simulations, assemblages {'nocr_isothermal': 2804, 'cr_isothermal': 1802} |

## Tables

| id | kind | file |
|---|---|---|
| validationDataset/nocr/isothermal | phase_quality | `validationDataset/nocr/emulator_quality_isothermal.csv` |
| validationDataset/nocr/openox | phase_quality | `validationDataset/nocr/emulator_quality_openox.csv` |
| validationDataset/cr/isothermal | phase_quality | `validationDataset/cr/emulator_quality_isothermal.csv` |
| validationDataset/cr/openox | phase_quality | `validationDataset/cr/emulator_quality_openox.csv` |
| standardWorkspace/nocr_isothermal | phase_quality | `standardWorkspace/emulator_quality_nocr_isothermal.csv` |
| standardWorkspace/cr_isothermal | phase_quality | `standardWorkspace/emulator_quality_cr_isothermal.csv` |
| standardWorkspace/property_errors | property_errors | `standardWorkspace/property_errors.csv` |
| standardWorkspace/phase_errors | phase_errors | `standardWorkspace/phase_errors.csv` |
