"""
Constants and mappings for nMELTS.

Extracted from Legacy/BackEnds/EmulatorLibrary.py

CSV files are automatically loaded from the 'projections' and 'old_transforms'
folders when this module is imported.
"""

import numpy as np
import pandas as pd
import molmass as ms
from typing import List, Dict, Set, Tuple
from pathlib import Path
from types import MappingProxyType

# Get the (config) directory containing this file (constants.py)
_CONFIG_DIR = Path(__file__).parent
_PROJECTIONS_DIR = _CONFIG_DIR / 'projections'
_OLD_TRANSFORMS_DIR = _CONFIG_DIR / 'old_transforms'

# Required elements (must always be present)
REQUIRED_ELEMENTS = {'Si','Ti', 'Al', 'Fe', 'Mg', 'Ca', 'Na'}

default_WRkeys = ['SiO2', 'TiO2', 'Al2O3', 'FeO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'H2O', 'Cr2O3', 'MnO', 'NiO']
default_Oxides = default_WRkeys + ['Fe2O3']
default_Elkeys = ['Si', 'Ti', 'Al', 'Fe', 'Mg', 'Ca', 'Na', 'K', 'P', 'H', 'Cr', 'Mn', 'Ni']
all_Elkeys = default_Elkeys #+['Mn', 'Ni']
all_Oxides = default_Oxides #+ ['MnO', 'NiO']

# This object is used to build the indexer and column headers for the MELTSdataset.
COMPOSITIONAL_COMPONENTS_IN_PHASES: Dict[str, List[str]] = { # Will need to be expanded as more phases are supported. May need to be flexible to MELTS model
    'System_main': ['Pressure', 'Temperature', 'logfO2-QFM', 'mass', 'F', 'viscosity', 'H', 'Cp', 'S', 'V', 'dVdP*10^6', 'dVdT*10^6'],
    'Bulk_comp': ['mass'] + all_Oxides,
    'olivine': ['forsterite', 'fayalite', 'monticellite', 'tephroite', 'ni-olivine'],
    'orthopyroxene': ['diopside', 'clinoenstatite', 'hedenbergite', 'alumino-buffonite', 'buffonite', 'essenite', 'jadeite'],
    'clinopyroxene': ['diopside', 'clinoenstatite', 'hedenbergite', 'alumino-buffonite', 'buffonite', 'essenite', 'jadeite'],
    'spinel': ['chromite', 'hercynite', 'magnetite', 'spinel', 'ulvospinel'],
    'plagioclase': ['albite', 'anorthite', 'sanidine'],
    'alkali-feldspar': ['albite', 'anorthite', 'sanidine'],
    'k-feldspar': ['albite', 'anorthite', 'sanidine'],
    'alkali-feldspar': ['albite', 'anorthite', 'sanidine'],
    'garnet': ['almandine', 'grossular', 'pyrope'],
    'nepheline': ['na-nepheline', 'k-nepheline', 'vc-nepheline', 'ca-nepheline'],
    'leucite': ['leucite', 'analcime', 'na-leucite'],
    'biotite': ['annite', 'phlogopite'],
    'rhm-oxide': ['geikielite', 'hematite', 'ilmenite', 'pyrophanite', 'corundum'],
    'alloy-solid': ['Fe-metal', 'Ni-metal'],
    'alloy-liquid': ['Fe-metal', 'Ni-metal'],
    'analcime': ['leucite', 'analcime', 'na-leucite'],
    'apatite': [],  # No chemical components, only state variables
    'whitlockite': [],  # No chemical components, only state variables
    'quartz': [],  # No chemical components, only state variables
    'tridymite': [],  # No chemical components, only state variables
    'cristobalite': [],  # No chemical components, only state variables
    'amphibole': [],  # No chemical components, only state variables
    'muscovite': [],  # No chemical components, only state variables
    'fluid': [],  # No chemical components, only state variables
    'water': [],  # No chemical components, only state variables
    'melts-liquid': ['liq mass (gm)', 
                     'wt% SiO2', 'wt% TiO2', 'wt% Al2O3', 'wt% FeO', 'wt% MgO', 
                     'wt% CaO', 'wt% Na2O', 'wt% K2O', 'wt% P2O5','wt% H2O', 'wt% Cr2O3',
                     'wt% MnO', 'wt% NiO', 'wt% Fe2O3',
                     'liq rho (gm/cc)',
                     'liq vis (log 10 poise)',
                     'liq H (kJ)',
                     'liq S (J/K)',
                     'liq V (cc)']
    }

COMPONENTS_IN_PHASES = COMPOSITIONAL_COMPONENTS_IN_PHASES.copy()

state_vars_to_add = [
            'mass (gm)',
            'rho (gm/cc)',
            'H (kJ)',
            'S (J/K)',
            'V (cc)'
        ]

        
for phase in COMPONENTS_IN_PHASES.keys():
    if phase not in ['melts-liquid', 'System_main', 'Bulk_comp']:
        for state_var in state_vars_to_add:
            if state_var == 'mass (gm)':
                COMPONENTS_IN_PHASES[phase] = ['mass (gm)'] + COMPONENTS_IN_PHASES[phase] # Add mass to the front, the rest to the back. 
            else:
                COMPONENTS_IN_PHASES[phase].append(state_var)

# Convert to immutable MappingProxyType to prevent accidental modifications
# that could affect other DatasetIndexer / ml_indexer instances
COMPOSITIONAL_COMPONENTS_IN_PHASES = MappingProxyType(COMPOSITIONAL_COMPONENTS_IN_PHASES)
COMPONENTS_IN_PHASES = MappingProxyType(COMPONENTS_IN_PHASES)


# Used for plotting wt% oxides in phases
active_oxide_lists = {
    'olivine': ['SiO2','MgO', 'FeO', 'CaO'],
    'orthopyroxene': ['SiO2', 'CaO', 'MgO', 'FeO', 'Fe2O3', 'Na2O', 'Al2O3', 'TiO2'],
    'clinopyroxene': ['SiO2', 'CaO', 'MgO', 'FeO', 'Fe2O3', 'Na2O', 'Al2O3', 'TiO2'],
    'spinel': ['Cr2O3', 'MgO', 'FeO', 'Fe2O3', 'Al2O3', 'TiO2'],
    'plagioclase': ['SiO2', 'Na2O', 'K2O', 'CaO'],
    'k-feldspar': ['SiO2', 'Na2O', 'K2O', 'CaO'],
    'garnet': ['SiO2', 'FeO', 'CaO', 'MgO'],
    'nepheline': ['SiO2', 'Na2O', 'K2O', 'CaO', 'Al2O3'],
    'leucite': ['SiO2', 'Na2O', 'K2O', 'H2O'],
    'biotite': ['SiO2','FeO','MgO'],
    'rhm-oxide': ['MgO', 'FeO', 'Fe2O3', 'TiO2'],
    'analcime': ['SiO2', 'Na2O', 'K2O', 'H2O'],
    'melts-liquid': default_Oxides
        }