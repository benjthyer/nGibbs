# Benchmark: HeFESToMarsEmulator

- created: 2026-11-20T02:41:31+00:00
- model dir: `/home/bthyer/nGibbs/src/ngibbs/engine/TrainedModels/HeFESToMars`
- git commit: 0b584f24a3352a062e710bbc7d3bfba3bf7e54b4, device: cuda

## Components

| component | status | detail |
|---|---|---|
| validationDataset | ok |  |
| standardWorkspace | ok | 3/3 simulations, assemblages {'isothermal': 153, 'isentropic': 153} |

## Tables

| id | kind | file |
|---|---|---|
| validationDataset/isothermal | phase_quality | `validationDataset/emulator_quality_isothermal.csv` |
| validationDataset/isentropic | phase_quality | `validationDataset/emulator_quality_isentropic.csv` |
| standardWorkspace/isothermal | phase_quality | `standardWorkspace/emulator_quality_isothermal.csv` |
| standardWorkspace/isentropic | phase_quality | `standardWorkspace/emulator_quality_isentropic.csv` |
| standardWorkspace/property_errors | property_errors | `standardWorkspace/property_errors.csv` |
| standardWorkspace/phase_errors | phase_errors | `standardWorkspace/phase_errors.csv` |
