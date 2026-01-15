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

# Get the (config) directory containing this file (constants.py)
_CONFIG_DIR = Path(__file__).parent
_PROJECTIONS_DIR = _CONFIG_DIR / 'projections'
_OLD_TRANSFORMS_DIR = _CONFIG_DIR / 'old_transforms'


default_WRkeys = ['SiO2', 'TiO2', 'Al2O3', 'FeO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'H2O', 'Cr2O3']
default_Oxides = default_WRkeys + ['Fe2O3']
default_Elkeys = ['Si', 'Ti', 'Al', 'Fe', 'Mg', 'Ca', 'Na', 'K', 'P', 'H', 'Cr']

# Aliases for backward compatibility
#WRkeys = default_WRkeys
#Oxides = default_Oxides
#Elkeys = default_Elkeys

# This object is used to build the indexer and column headers for the MELTSdataset.
COMPONENTS_IN_PHASES: Dict[str, List[str]] = { # Will need to be expanded as more phases are supported
    'System_main': ['Pressure', 'Temperature', 'logfO2-QFM', 'mass', 'F', 'viscosity', 'H', 'Cp', 'S', 'V', 'dVdP*10^6', 'dVdT*10^6'],
    'olivine': ['forsterite', 'fayalite', 'monticellite', 'tephroite', 'ni-olivine'],
    'orthopyroxene': ['diopside', 'clinoenstatite', 'hedenbergite', 'alumino-buffonite', 'buffonite', 'essenite', 'jadeite'],
    'clinopyroxene': ['diopside', 'clinoenstatite', 'hedenbergite', 'alumino-buffonite', 'buffonite', 'essenite', 'jadeite'],
    'spinel': ['chromite', 'hercynite', 'magnetite', 'spinel', 'ulvospinel'],
    'plagioclase': ['albite', 'anorthite', 'sanidine'],
    'alkali-feldspar': ['albite', 'anorthite', 'sanidine'],
    'k-feldspar': ['albite', 'anorthite', 'sanidine'],
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
                     'wt% CaO', 'wt% Na2O', 'wt% K2O', 'wt% P2O5', 'wt% MnO', 
                     'wt% H2O', 'wt% Cr2O3', 'wt% NiO', 'wt% Fe2O3',
                     'liq rho (gm/cc)',
                     'liq vis (log 10 poise)',
                     'liq H (kJ)',
                     'liq S (J/K)',
                     'liq V (cc)']
}

state_vars_to_add = [
            'mass (gm)',
            'rho (gm/cc)',
            'H (kJ)',
            'S (J/K)',
            'V (cc)'
        ]

        
for phase in COMPONENTS_IN_PHASES.keys():
    if phase not in ['melts-liquid', 'System_main']:
        for state_var in state_vars_to_add:
            if state_var == 'mass (gm)':
                COMPONENTS_IN_PHASES[phase] = ['mass (gm)'] + COMPONENTS_IN_PHASES[phase] # Add mass to the front, the rest to the back. 
            else:
                COMPONENTS_IN_PHASES[phase].append(state_var)