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

When the user speaks about "intensive components", they are speaking about phase-wise normalized component fractions in tensor of shape (B, VC). 
When the user speaks about "extensive components", they are speaking about the total amounts of each component in the system in a tensor of shape (B, C)

## Project Conventions
{Every time the agent codes, the changes should be documented in the ChangeLogs folder. 
{Never write code to silently ignore upstream failures. It is much better to even include assertions to test that the upstream result is valid, and if not, to raise an error that can be caught and debugged.}
Value Brevity — when describing changes, be concise and high level. Limit each line in the changelog to 80 charecters.
Aggregate changes by date following the format YYYY-MM-DD
Let there be a ChangeLog for each major version of the codebase.
}
{Always use this python environment: `conda activate torch-env`}
{If the motivation, goals, or mechanism to accomplish some requested code is unclear, please ask for clarification!}
{Never look in or do anything within any folder labeled "legacy" without explicit instructions to do so. This code is old and will be deleted soon. It is kept for human reference only.}
{Never use emojis or unusual characters in print statements: they can cause errors when writing to and parsing log files.
For instance, instead of '→', use '->'}
{Avoid short helper functions. If a function can be written in 3 lines or less, write it inline.}
{If something can be vectorized, vectorize it. This module is built for speed.}
{When programming tests, always use the code from the repository directly. Never copy or redundantly reproduce code from the repository, as it does not truly represent the state of the tested codebase.}
{Silent failures are bad. Never hide failures; errors should stop code execution. Never write something like `try: ... except: pass` }
```