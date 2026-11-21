# Benchmark: HeFESToEmulator

- created: 2026-11-21T01:45:04+00:00
- model dir: `/home/bthyer/nGibbs/src/ngibbs/engine/TrainedModels/HeFESToEarth`
- git commit: 3301474d65bcd152ecb74d5bb5d0e9355fd96ae3, device: cpu

## Components

| component | status | detail |
|---|---|---|
| validationDataset | ok |  |
| standardWorkspace | ok | 3/3 simulations, assemblages {'isothermal': 420, 'isentropic': 420} |

## Tables

| id | kind | file |
|---|---|---|
| validationDataset/isothermal | phase_quality | `validationDataset/emulator_quality_isothermal.csv` |
| validationDataset/isentropic | phase_quality | `validationDataset/emulator_quality_isentropic.csv` |
| standardWorkspace/isothermal | phase_quality | `standardWorkspace/emulator_quality_isothermal.csv` |
| standardWorkspace/isentropic | phase_quality | `standardWorkspace/emulator_quality_isentropic.csv` |
| standardWorkspace/property_errors | property_errors | `standardWorkspace/property_errors.csv` |
| standardWorkspace/phase_errors | phase_errors | `standardWorkspace/phase_errors.csv` |
