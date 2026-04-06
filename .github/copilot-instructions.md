---
name: nMELTSagent.md
description: This is an evolving guide to the nMELTS agent, to be applied generally whenever an agent is changing code or writing new code.
---


```markdown


## Architecture
{Source code is divided in two parts: one is a deployable nMELTS package that can be pip installed, and the other is a builder that contains scripts to generate data products and train models. Any code that interfaces with alphamelts should be run in a linux subsystem, where it can take advantage of GNU parallel processing of CPUs.}

Anything within src/nMELTS should never access files from outside src/nMELTS. 
Outside files though are free to reference the contents of src/nMELTS.

## Dynamic Indexing
{The DatasetIndexer() object is used to generate dynamic indexers for the data tables. It spawns an ml_indexer() object that is carried by nMELTS models through training and deployment: carrying names of phases and components and how they map to matrix inputs and outputs. Notably, it contains the matrices used to project data between component, oxide, and element space.}

Always refer to [ml_indexer_readme.md](..\src\nMELTS\config\README_MLIndexer.md) for the most up-to-date documentation on how the contents of ml_indexer(), which is used all over the codebase.

## Build and Test
{For now, no automatic testing, although test suggestions are welcome}

## Project Conventions
{Every time the agent codes, the changes should be documented in the ChangeLogs folder. 
Value Brevity — when describing changes, be concise and high level. Limit each line in the changelog to 80 charecters.
Aggregate changes by date following the format YYYY-MM-DD
Let there be a ChangeLog for each major version of the codebase.
}
{The python environment used in this repository is loaded with `conda activate torch-env`}
{If the motivation, goals, or mechanism to accomplish some requested code is unclear, please ask for clarification!}
{Never look in or do anything within any folder labeled "legacy" without explicit instructions to do so. This code is old and will be deleted soon. It is kept for human reference only.}
{Never use emojis or unusual characters in print statements: they can cause errors when writing to and parsing log files.
For instance, instead of '→', use '->'}
```