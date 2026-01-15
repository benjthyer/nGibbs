"""
Constants and mappings for nMELTS.

Extracted from Legacy/BackEnds/EmulatorLibrary.py

CSV files are automatically loaded from the 'projections' and 'old_transforms'
folders when this module is imported.
"""

import numpy as np
import pandas as pd
import molmass as ms
import torch
from pathlib import Path

# Get the (config) directory containing this file (constants.py)
_CONFIG_DIR = Path(__file__).parent
_PROJECTIONS_DIR = _CONFIG_DIR / 'projections'
_OLD_TRANSFORMS_DIR = _CONFIG_DIR / 'old_transforms'

#I NEED TO CHANGE THIS INFRASTRUCTURE TO GENERATE CONSTANTS AND INDEXERS GIVEN THE SPECIFICATIONS OF WHICH PHASES TO INCLUDE, TO AID IN FORWARD COMPATIBILITY AS PHASES ARE CHANGED

# ============================================================================
# Component Indices - Phase to Component Mappings
# ============================================================================


MELTS_indices = {
    'System_main': {
        'Pressure': 0,
        'Temperature': 1,
        'logfO2-QFM': 2
    },
    'olivine': {
        'mass (gm)': 3,
        'tephroite': 4,
        'fayalite': 5,
        'ni-olivine': 6,
        'monticellite': 7,
        'forsterite': 8
    },
    'orthopyroxene': {
        'mass (gm)': 9,
        'diopside': 10,
        'clinoenstatite': 11,
        'hedenbergite': 12,
        'alumino-buffonite': 13,
        'buffonite': 14,
        'essenite': 15,
        'jadeite': 16
    },
    'clinopyroxene': {
        'mass (gm)': 17,
        'diopside': 18,
        'clinoenstatite': 19,
        'hedenbergite': 20,
        'alumino-buffonite': 21,
        'buffonite': 22,
        'essenite': 23,
        'jadeite': 24
    },
    'spinel': {
        'mass (gm)': 25,
        'chromite': 26,
        'hercynite': 27,
        'magnetite': 28,
        'spinel': 29,
        'ulvospinel': 30
    },
    'plagioclase': {
        'mass (gm)': 31,
        'albite': 32,
        'anorthite': 33,
        'sanidine': 34  # alphamelts 2.3.1 uses exclusively 'sanidine'. Previous versions used 'highsanidine'
    },
    'alkali-feldspar': {
        'mass (gm)': 35,
        'albite': 36,
        'anorthite': 37,
        'sanidine': 38
    },
    'k-feldspar': {
        'mass (gm)': 35,
        'albite': 36,
        'anorthite': 37,
        'sanidine': 38
    },
    'garnet': {
        'mass (gm)': 39,
        'almandine': 40,
        'grossular': 41,
        'pyrope': 42
    },
    'nepheline': {
        'mass (gm)': 43,
        'na-nepheline': 44,
        'k-nepheline': 45,
        'vc-nepheline': 46,
        'ca-nepheline': 47
    },
    'leucite': {
        'mass (gm)': 48,
        'leucite': 49,
        'analcime': 50,
        'na-leucite': 51
    },
    'biotite': {
        'mass (gm)': 52,
        'annite': 53,
        'phlogopite': 54
    },
    'rhm-oxide': {
        'mass (gm)': 55,
        'geikielite': 56,
        'hematite': 57,
        'ilmenite': 58,
        'pyrophanite': 59,
        'corundum': 60
    },
    'alloy-solid': {
        'mass (gm)': 61,
        'Fe-metal': 62,
        'Ni-metal': 63
    },
    'alloy-liquid': {
        'mass (gm)': 64,
        'Fe-metal': 65,
        'Ni-metal': 66
    },
    'analcime': {  # This is populated from hydrous leucite during processing. Does not correspond to real phase in MELTS.
        'mass (gm)': 67,
        'leucite': 68,
        'analcime': 69,
        'na-leucite': 70
    },
    'apatite': {
        'mass (gm)': 71
    },
    'whitlockite': {
        'mass (gm)': 72
    },
    'quartz': {
        'mass (gm)': 73
    },
    'tridymite': {
        'mass (gm)': 74
    },
    'cristobalite': {
        'mass (gm)': 75
    },
    'amphibole': {
        'mass (gm)': 76
    },
    'muscovite': {
        'mass (gm)': 77
    },
    'fluid': {
        'mass (gm)': 78
    },
    'water': {
        'mass (gm)': 78
    },
    'melts-liquid': {
        'liq mass (gm)': 79,
        'wt% SiO2': 80,
        'wt% TiO2': 81,
        'wt% Al2O3': 82,
        'wt% FeO': 83,
        'wt% MgO': 84,
        'wt% CaO': 85,
        'wt% Na2O': 86,
        'wt% K2O': 87,
        'wt% P2O5': 88,
        'wt% MnO': 89,
        'wt% H2O': 90,
        'wt% Cr2O3': 91,
        'wt% NiO': 92,
        'wt% Fe2O3': 93
    }
}

# Build database headers
database_headers = []

for pha, val in MELTS_indices.items():
    for com, idx in val.items():
        if pha not in ['alkali-feldspar', 'water']:
            database_headers.append(f"{com}({pha})") # Handle more explicity the case where multiple names for a single phase are used. 

# Add system main properties
MELTS_indices['System_main']['viscosity'] = 94
database_headers.append("viscosity(System_main)")
MELTS_indices['System_main']['H'] = 95
database_headers.append("H(System_main)")
MELTS_indices['System_main']['Cp'] = 96
database_headers.append("Cp(System_main)")
MELTS_indices['System_main']['S'] = 97
database_headers.append('S(system_main)')
MELTS_indices['System_main']['V'] = 98
database_headers.append('V(System_main)')
MELTS_indices['System_main']['dVdP*10^6'] = 99
database_headers.append('dVdP*10^6(System_main)')
MELTS_indices['System_main']['dVdT*10^6'] = 100
database_headers.append('dVdP*10^6(System_main)')

# List of phases (in order of appearance)
phases_in_order = [
    'olivine', 'orthopyroxene', 'clinopyroxene', 'spinel',
    'plagioclase', 'alkali-feldspar', 'garnet', 'nepheline', 'leucite',
    'biotite', 'rhm-oxide', 'alloy-solid', 'alloy-liquid', 'analcime', 'apatite', 'whitlockite', 
    'quartz', 'tridymite', 'muscovite', 'fluid'
]

# Add density, enthalpy, entropy, volume fields
next_index = 101
for phase in phases_in_order:
    MELTS_indices[phase]['rho (gm/cc)'] = next_index
    next_index += 1
    database_headers.append(f"{'rho (gm/cc)'}({phase})")
    MELTS_indices[phase]['H (kJ)'] = next_index
    next_index += 1
    database_headers.append(f"{'H (kJ)'}({phase})")
    MELTS_indices[phase]['S (J/K)'] = next_index
    next_index += 1
    database_headers.append(f"{'S (J/K)'}({phase})")
    MELTS_indices[phase]['V (cc)'] = next_index
    next_index += 1
    database_headers.append(f"{'V (cc)'}({phase})")

# Add viscosity field for melt only
MELTS_indices['melts-liquid']['liq rho (gm/cc)'] = next_index
MELTS_indices['melts-liquid']['liq vis (log 10 poise)'] = next_index + 1
MELTS_indices['melts-liquid']['liq H (kJ)'] = next_index + 2
MELTS_indices['melts-liquid']['liq S (J/K)'] = next_index + 3
MELTS_indices['melts-liquid']['liq V (cc)'] = next_index + 4
database_headers.append('rho (gm/cc)(melts-liquid)')
database_headers.append('liq vis (log 10 poise)')
database_headers.append('liq H (kJ)')
database_headers.append('liq S (J/K)')
database_headers.append('liq V (cc)')

# Mass indices
mass_indices = []  # exclude amphibole
for phase, props in MELTS_indices.items(): #properties
    if phase in (phases_in_order + ['melts-liquid']):
        for key, ind in list(props.items()):
            if 'mass' in key:
                mass_indices.append(ind)
mass_indices = np.unique(mass_indices)

# ============================================================================
# End Indexer Generation
#============================================================================

# ============================================================================
# Old/Older Transform Constants (for backward compatibility)
# ============================================================================

WRkeysOld = ['SiO2', 'TiO2', 'Al2O3', 'FeO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'MnO', 'H2O', 'Cr2O3', 'NiO']
OxidesOld = WRkeysOld + ['Fe2O3']
ElkeysOld = ['Si', 'Ti', 'Al', 'Fe', 'Mg', 'Ca', 'Na', 'K', 'P', 'Mn', 'H', 'Cr', 'Ni']
MtotOld = np.array([ms.Formula(MM).mass for MM in OxidesOld]).reshape(14, 1)  # 14x1, For phase molar mass from oxide moles

# Load old transform CSV files from old_transforms folder
try:
    compToOxOld_path = _OLD_TRANSFORMS_DIR / 'compToOx - Copy (2).csv'
    compToOxOld = pd.read_csv(compToOxOld_path).to_numpy()[:, 1:].astype(np.float32)
    MinvOld = np.diag([1/ms.Formula(MM).mass for MM in OxidesOld])
    MMOld = np.diag([ms.Formula(mm).mass for mm in OxidesOld])
    oxToElOld_path = _OLD_TRANSFORMS_DIR / 'OxToEl_olderJuly2525.csv'
    oxToElOld = pd.read_csv(oxToElOld_path).to_numpy()[:, 1:].astype(np.float32).T
except FileNotFoundError as e:
    # If files don't exist, create placeholders
    compToOxOld = None
    MinvOld = None
    MMOld = None
    oxToElOld = None
    import warnings
    warnings.warn(f"Could not load old transform CSV files: {e}", UserWarning)

# ============================================================================
# Current Transform Constants
# ============================================================================

WRkeys = ['SiO2', 'TiO2', 'Al2O3', 'FeO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'H2O', 'Cr2O3']
Oxides = WRkeys + ['Fe2O3']
Elkeys = ['Si', 'Ti', 'Al', 'Fe', 'Mg', 'Ca', 'Na', 'K', 'P', 'H', 'Cr']
Mtot = np.array([ms.Formula(MM).mass for MM in Oxides]).reshape(len(Oxides), 1)  # For phase molar mass from oxide moles

# Load current transform CSV files from projections folder
# These are automatically loaded when this module is imported
try:
    compToOxLoad_path = _PROJECTIONS_DIR / 'compToOx.csv'
    PxSpTransform_path = _PROJECTIONS_DIR / 'PxSp_Comp_Transform.csv'
    oxToEl_path = _PROJECTIONS_DIR / 'OxToEl.csv'
    
    compToOxLoad = pd.read_csv(compToOxLoad_path).to_numpy()[:, 1:].astype(np.float32)
    PxSpTransform = pd.read_csv(PxSpTransform_path).to_numpy()[:, 1:].astype(np.float32)
    compToOx = np.linalg.inv(PxSpTransform) @ compToOxLoad  # pc,co->po
    MM = np.diag([ms.Formula(MM).mass for MM in Oxides])
    Minv = np.diag([1/ms.Formula(MM).mass for MM in Oxides])
    oxToEl = pd.read_csv(oxToEl_path).to_numpy()[:, 1:].astype(np.float32).T
except FileNotFoundError as e:
    # If files don't exist, create placeholders
    compToOxLoad = None
    PxSpTransform = None
    compToOx = None
    MM = None
    Minv = None
    oxToEl = None
    import warnings
    warnings.warn(f"Could not load transform CSV files from projections folder: {e}. "
                  f"Expected files in: {_PROJECTIONS_DIR}", UserWarning)

# ============================================================================
# Label Indices - ML-ready data mappings
# ============================================================================

label_indices = {}  # Used for mapping phases to full components matrix
label_names = []
detail_label_indices = {}  # More used for index mapping
label_indices_comp = {}  # Mapping to compositionally variable intensive component output

detail_ind = 0
index = 0

for phase, components in MELTS_indices.items():  # Build indices for ML-ready data, will direct translation to element moles
    if phase not in ['System_main', 'alkali-feldspar', 'water', 'amphibole', 'melts-liquid', 'cristobalite']:
        phase_inds = []

        if phase == 'k-feldspar' or (len(components) > 5 and phase != 'alloy-solid'):
            detail_label_indices[phase] = {}
            comp_inds = []
            for component in components:
                if component not in ['liq rho (gm/cc)', 'liq vis (log 10 poise)', 'liq H (kJ)',
                                     'liq mass (gm)', 'liq H (kJ)', 'liq S (J/K)', 'liq V (cc)',
                                     'mass (gm)', 'rho (gm/cc)', 'H (kJ)', 'S (J/K)', 'V (cc)']:  # State variables, not chemistry
                    if component not in ['tephroite', 'co-olivine', 'ni-olivine', 'pyrophanite', 'Mn', 'Ni']:  # Components that are not used
                        phase_inds += [index]
                        if component != 'Fe-metal':
                            detail_label_indices[phase][component] = detail_ind
                            comp_inds.append(detail_ind)
                            detail_ind += 1
                        index += 1
                        label_names.append(component)
            label_indices_comp[phase] = np.array(comp_inds)
        else:
            phase_inds += [index]
            index += 1
            label_names.append(phase)
        label_indices[phase] = phase_inds

label_indices['melts-liquid'] = list(range(index, index+len(Elkeys)))
label_names += Elkeys

detail_label_indices['melts-liquid'] = {}
comp_inds = []
for key in Elkeys:
    detail_label_indices['melts-liquid'][key] = detail_ind
    comp_inds.append(detail_ind)
    detail_ind += 1
label_indices_comp['melts-liquid'] = np.array(comp_inds)

# ============================================================================
# Phase Dictionaries
# ============================================================================

compositionally_variable_phases = []
cj = 0
comp_phasedict = {}
mass_phasedict = {}
all_phases = []
for i, phase in enumerate(list(label_indices.keys())):
    all_phases.append(phase)
    mass_phasedict[phase] = i
    if len(label_indices[phase]) > 1:
        compositionally_variable_phases.append(phase)
        comp_phasedict[phase] = cj
        cj += 1

ncomps = label_indices['melts-liquid'][-1] + 1  # (C)
ncompsVaried = label_indices_comp['melts-liquid'][-1] + 1  # (V)
nphases = mass_phasedict['melts-liquid'] + 1  # (P)

compositionally_variable_binaries = []
compositionally_variable_subset = []
phaseToCompMap = np.zeros((nphases, ncomps))  # (P,C) General
variedToAllComp = np.zeros((ncompsVaried, ncomps))  # (V,C)

for p, phase in enumerate(list(label_indices.keys())):
    phaseToCompMap[p, label_indices[phase]] = 1
    if phase in compositionally_variable_phases:
        compositionally_variable_binaries.append(1)
        compositionally_variable_subset += label_indices[phase]
        variedToAllComp[label_indices_comp[phase], label_indices[phase]] = 1
    else:
        compositionally_variable_binaries.append(0)

comp_variable_IDMAT = torch.tensor(np.diag(compositionally_variable_binaries), dtype=torch.float)
is_fixed = ~(np.array(compositionally_variable_binaries).astype(bool))
fixed_phaseToCompMap = (is_fixed.reshape(1, -1) @ phaseToCompMap)
compositionally_variable_subset = np.array(compositionally_variable_subset)
compositional_component_subset = np.copy(compositionally_variable_subset)

# ============================================================================
# Oxide Dictionary
# ============================================================================

oxide_dict = {}
for i, ox in enumerate(Oxides):
    oxide_dict[ox] = i

# ============================================================================
# Component Mappings
# ============================================================================

chem_list = []
j = 0
k = 0
comp_binariesL = []
comp_mappingsL = []
comp_map = {}
for i, (label, inds) in enumerate(label_indices.items()):
    n_components = len(inds)
    if n_components > 1:
        comp_list = np.arange(k, k+n_components)
        k += n_components
        comp_map[label] = comp_list

for i, (label, inds) in enumerate(label_indices.items()):
    n_components = len(inds)
    if n_components > 1:
        comp_binariesL.append(i)
        comp_mappingsL = comp_mappingsL + np.repeat(j, n_components).tolist()
        j += 1
comp_binaries = np.array(comp_binariesL)
comp_mappings = np.zeros((j, len(comp_mappingsL)))  # Build Binary Matrix to project phases to components
for col, row in enumerate(comp_mappingsL):
    comp_mappings[row, col] = 1

boolTransCompToOx = np.copy(compToOx) if compToOx is not None else None
if boolTransCompToOx is not None:
    boolTransCompToOx[:, 3] += boolTransCompToOx[:, -1]  # Compile non-negative irons for use with element inputs
    boolTransCompToOx = (boolTransCompToOx[:, :-1] != 0).astype(int)  # For sending logits to neg infinity to prevent prediction of components with absent atoms

# ============================================================================
# Active Oxide Dictionary
# ============================================================================

active_ox_dict = {
    'olivine': [oxide_dict[ox] for ox in ['MgO', 'FeO', 'CaO']],
    'orthopyroxene': [oxide_dict[ox] for ox in ['CaO', 'MgO', 'FeO', 'Fe2O3', 'Na2O', 'Al2O3', 'TiO2']],
    'clinopyroxene': [oxide_dict[ox] for ox in ['CaO', 'MgO', 'FeO', 'Fe2O3', 'Na2O', 'Al2O3', 'TiO2']],
    'spinel': [oxide_dict[ox] for ox in ['Cr2O3', 'MgO', 'FeO', 'Fe2O3', 'Al2O3', 'TiO2']],
    'plagioclase': [oxide_dict[ox] for ox in ['Na2O', 'K2O', 'CaO']],
    'k-feldspar': [oxide_dict[ox] for ox in ['Na2O', 'K2O', 'CaO']],
    'garnet': [oxide_dict[ox] for ox in ['FeO', 'CaO', 'MgO']],
    'nepheline': [oxide_dict[ox] for ox in ['Na2O', 'K2O', 'CaO', 'Al2O3']],
    'leucite': [oxide_dict[ox] for ox in ['Na2O', 'K2O', 'H2O']],
    'biotite': [oxide_dict['MgO']],
    'rhm-oxide': [oxide_dict[ox] for ox in ['MgO', 'FeO', 'Fe2O3', 'TiO2']],
    'analcime': [oxide_dict[ox] for ox in ['Na2O', 'K2O', 'H2O']],
    'melts-liquid': [oxide_dict[ox] for ox in Oxides]
}
