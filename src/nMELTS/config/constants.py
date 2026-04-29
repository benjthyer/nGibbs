"""
Constants and mappings for nMELTS.

Extracted from Legacy/BackEnds/EmulatorLibrary.py

CSV files are automatically loaded from the 'projections' and 'old_transforms'
folders when this module is imported.
"""

from typing import List, Dict, Set, Tuple
from pathlib import Path
from types import MappingProxyType

# Get the (config) directory containing this file (constants.py)
_CONFIG_DIR = Path(__file__).parent
_PROJECTIONS_DIR = _CONFIG_DIR / 'projections'
_OLD_TRANSFORMS_DIR = _CONFIG_DIR / 'old_transforms'

# Type conversion mappings for training/model configuration
# These define which config parameters should be converted to specific types
TYPE_CONVERSION_MAP = {
    'activation_leak': float,
    'lowWD': float,
    'highWD': float,
    'noise': float,
    'lr': float,
    'learning_rate': float,
    'encoderLayerUp': int,
    'encoderLayerDown': int,
    'middleLayerUp': int,
    'middleLayerDown': int,
    'encoderLayer': int,
    'middleLayer': int,
    'batch_size': int,
    'epochs': int,
    'early_stopping_patience': int,
}

# Required elements (must always be present)
REQUIRED_ELEMENTS = {'Si','Ti', 'Al', 'Fe', 'Mg', 'Ca', 'Na'}

default_WRkeys = ['SiO2', 'TiO2', 'Al2O3', 'FeO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'H2O', 'Cr2O3', 'MnO', 'NiO']
default_Oxides = default_WRkeys + ['Fe2O3']
default_Elkeys = ['Si', 'Ti', 'Al', 'Fe', 'Mg', 'Ca', 'Na', 'K', 'P', 'H', 'Cr', 'Mn', 'Ni']
all_Elkeys = default_Elkeys #+['Mn', 'Ni']
all_Oxides = default_Oxides #+ ['MnO', 'NiO']

# Centralized molar masses (g/mol) for oxide formulas used across nMELTS.
OXIDE_MOLAR_MASSES: Dict[str, float] = {
    'SiO2': 60.0843,
    'TiO2': 79.866,
    'Al2O3': 101.9613,
    'FeO': 71.8440,
    'Fe2O3': 159.688,
    'MgO': 40.3044,
    'CaO': 56.0774,
    'Na2O': 61.9789,
    'K2O': 94.196,
    'P2O5': 141.9445,
    'H2O': 18.01528,
    'Cr2O3': 151.9904,
    'MnO': 70.9374,
    'NiO': 74.6928,
    'CoO': 74.9326,
}


def get_oxide_molar_mass(oxide: str) -> float:
    """Return oxide molar mass in g/mol, raising on unknown oxides."""
    if oxide not in OXIDE_MOLAR_MASSES:
        raise KeyError(
            f"Oxide '{oxide}' is missing from OXIDE_MOLAR_MASSES in constants.py"
        )
    return float(OXIDE_MOLAR_MASSES[oxide])

# This object is used to build the indexer and column headers for the MELTSdataset.
COMPOSITIONAL_COMPONENTS_IN_PHASES: Dict[str, List[str]] = { # Will need to be expanded as more phases are supported. May need to be flexible to MELTS model
    'System_main': ['Pressure', 'Temperature', 'logfO2-QFM', 'mass', 'F', 'viscosity', 'H', 'Cp', 'S', 'V', 'dVdP*10^6', 'dVdT*10^6', 'rhol', 'rhos'],
    'Bulk_comp': ['mass'] + all_Oxides,
    'olivine': ['forsterite', 'fayalite', 'monticellite', 'tephroite', 'ni-olivine'],
    'orthopyroxene': ['diopside', 'clinoenstatite', 'hedenbergite', 'alumino-buffonite', 'buffonite', 'essenite', 'jadeite'],
    'clinopyroxene': ['diopside', 'clinoenstatite', 'hedenbergite', 'alumino-buffonite', 'buffonite', 'essenite', 'jadeite'],
    'spinel': ['chromite', 'hercynite', 'magnetite', 'spinel', 'ulvospinel'],
    'plagioclase': ['albite', 'anorthite', 'sanidine'],
    'alkali-feldspar': ['albite', 'anorthite', 'sanidine'],
    'k-feldspar': ['albite', 'anorthite', 'sanidine'],
    'feldspar': ['albite', 'anorthite', 'sanidine'],
    'garnet': ['almandine', 'grossular', 'pyrope'],
    'nepheline': ['na-nepheline', 'k-nepheline', 'vc-nepheline', 'ca-nepheline'],
    'leucite': ['leucite', 'analcime', 'na-leucite'],
    'biotite': ['annite', 'phlogopite'],
    'rhm-oxide': ['geikielite', 'hematite', 'ilmenite', 'pyrophanite', 'corundum'], # Corundum is not actually in rhm-oxide in pMELTS
    'alloy-solid': ['Fe-metal', 'Ni-metal'],
    'alloy-liquid': ['Fe-liquid', 'Ni-liquid'], # Definitely called this in pMELTS... Possible that components are 'Fe-metal' etc for others? 
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
                     'liq V (cc)'],
    'hornblende': ['pargasite', 'ferropargasite', 'magnesiohastingsite'],
    }


# HeFESTo compositional species by phase, using full species names from tables.
COMPOSITIONAL_COMPONENTS_IN_PHASES_HEFESTO: Dict[str, List[str]] = {
    'Bulk_comp':['SiO2', 'MgO', 'FeO', 'Fe2O3', 'CaO', 'Al2O3', 'Na2O', 'Cr2O3'],
    'System_main': ['P(GPa)', 'T(K)', 'rho(g/cm^3)', 'mass (gm)', 'VS(km/s)', 'VP(km/s)',  'H(kJ/g)', 'cp(J/g/K)', 'S(J/g/K)', 'KS(GPa)', 'alpha(1e5_K^-1)'],
    'Bulk_comp_elements':['Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr', 'O'],
    'plagioclase': ['anorthite', 'albite'],
    'spinel': ['spinel', 'hercynite', 'magnetite', 'picro-chromite'],
    'olivine': ['forsterite', 'fayalite'],
    'wadsleyite': ['mg-wadsleyite', 'fe-wadsleyite'],
    'ringwoodite': ['mg-ringwoodite', 'fe-ringwoodite'],
    'orthopyroxene': ['enstatite', 'ferrosilite', 'mg-tschermaks', 'ortho-diopside'],
    'clinopyroxene': ['diopside', 'hedenbergite', 'clinoenstatite', 'ca-tschermaks', 'jadeite', 'acmite'],
    'wollastonite': ['wollastonite'],
    'pseudowollastonite': ['pseudowollastonite'],
    'hp-clinopyroxene': ['hp-clinoenstatite', 'hp-clinoferrosilite'],
    'ca-perovskite': ['ca-perovskite'],
    'akimotoite': ['mg-akimotoite', 'fe-akimotoite', 'corundum', 'hematite', 'eskolaite'],
    'garnet': ['pyrope', 'almandine', 'grossular', 'mg-majorite', 'na-majorite', 'andradite', 'knorringite'],
    'quartz': ['quartz'],
    'coesite': ['coesite'],
    'stishovite': ['stishovite'],
    'seifertite': ['seifertite'],
    'bridgmanite': ['mg-bridgmanite', 'fe-bridgmanite', 'al-bridgmanite',
                    'ferric-bridgmanite', 'ferric-bridgmanite-ls',
                    'ferric-al-bridgmanite', 'cr-bridgmanite'],
    'post-perovskite': ['mg-post-perovskite', 'fe-post-perovskite', 'al-post-perovskite',
                        'ferric-post-perovskite', 'cr-post-perovskite'],
    'ferropericlase': ['periclase', 'wustite', 'wustite-ls', 'alpha-naalo2', 'magnetite'],
    'ca-ferrite': ['mg-ca-ferrite', 'fe-ca-ferrite', 'na-ca-ferrite',
                   'high-pressure-magnetite', 'cr-ca-ferrite'],
    'nal-phase': ['mg-nal-phase', 'fe-nal-phase', 'na-nal-phase'],
    'kyanite': ['kyanite'],
    'nepheline': ['nepheline'],
    'alpha-iron': ['alpha-iron'], 
    'gamma-iron': ['gamma-iron'],
    'epsilon-iron': ['epsilon-iron'],
    'alpha-pbo-type-sio2': ['alpha-pbo-type-sio2']
}

# Abbreviation mapping from table codes to short names.
HEFESTO_ABBREVIATION_TO_SHORT_NAMES: Dict[str, str] = {
    'plg': 'plagioclase', #
    'sp': 'spinel', #
    'ol': 'olivine',#
    'wa': 'wadsleyite',#
    'ri': 'ringwoodite',#
    'opx': 'orthopyroxene',#
    'cpx': 'clinopyroxene',#
    'wo': 'wollastonite',#
    'pwo': 'pseudowollastonite',#
    #'hpcpx': 'hp-clinopyroxene',
    'c2c': 'hp-clinopyroxene',#
    'cpv': 'ca-perovskite',#
    #'ak': 'akimotoite',
    'il': 'akimotoite',#
    'mgil': 'mg-akimotoite',#
    'feil': 'fe-akimotoite',#
    'gt': 'garnet',#
    'qtz': 'quartz',#
    'coes': 'coesite',#
    'st': 'stishovite',#
    'apbo': 'alpha-pbo-type-sio2',#
    'seif': 'seifertite',
    #'bg': 'bridgmanite',
    'pv': 'bridgmanite',#
    'ppv': 'post-perovskite',#
    'fp': 'ferropericlase',
    'cf': 'ca-ferrite',#
    'nal': 'nal-phase',#
    'ky': 'kyanite',#
    'neph': 'nepheline',#
    'an': 'anorthite',
    'ab': 'albite',
    'hc': 'hercynite',
    'smag': 'magnetite-spinel',
    'picr': 'picro-chromite',
    'fo': 'forsterite',
    'fa': 'fayalite',
    'mgwa': 'mg-wadsleyite',
    'fewa': 'fe-wadsleyite',
    'mgri': 'mg-ringwoodite',
    'feri': 'fe-ringwoodite',
    'en': 'enstatite',
    'fs': 'ferrosilite',
    'mgts': 'mg-tschermaks',
    'odi': 'ortho-diopside',
    'di': 'diopside',
    'he': 'hedenbergite',
    'hem': 'hematite',
    'cen': 'clinoenstatite',
    'cats': 'ca-tschermaks',
    'jd': 'jadeite',
    'acm': 'acmite',
    'hpcen': 'hp-clinoenstatite',
    'mgc2': 'hp-clinoenstatite',
    'hpcfs': 'hp-clinoferrosilite',
    'fec2': 'hp-clinoferrosilite',
    'capv': 'ca-perovskite',
    'mgak': 'mg-akimotoite',
    'feak': 'fe-akimotoite',
    'co': 'corundum',
    'esk': 'eskolaite',
    'py': 'pyrope',
    'al': 'almandine',
    'gr': 'grossular',
    'mgmj': 'mg-majorite',
    'namj': 'na-majorite',
    'andr': 'andradite',
    'knor': 'knorringite',
    #'mgbg': 'mg-bridgmanite',
    #'febg': 'fe-bridgmanite',
    #'albg': 'al-bridgmanite',
    #'hebg': 'ferric-bridgmanite',
    #'hlbg': 'ferric-bridgmanite-ls',
    #'fabg': 'ferric-al-bridgmanite',
    #'crbg': 'cr-bridgmanite',
    'mgpv': 'mg-bridgmanite',
    'fepv': 'fe-bridgmanite',
    'alpv': 'al-bridgmanite',
    'hepv': 'ferric-bridgmanite',
    'hlpv': 'ferric-bridgmanite-ls',
    'fapv': 'ferric-al-bridgmanite',
    'crpv': 'cr-bridgmanite',
    'mppv': 'mg-post-perovskite',
    'fppv': 'fe-post-perovskite',
    'appv': 'al-post-perovskite',
    'hppv': 'ferric-post-perovskite',
    'cppv': 'cr-post-perovskite',
    'pe': 'periclase',
    'wu': 'wustite',
    #'mw': 'wustite',#
    'mw' : 'ferropericlase',#
    'wuls': 'wustite-ls',
    'anao': 'alpha-naalo2',
    'mag': 'magnetite',
    'mgcf': 'mg-ca-ferrite',
    'fecf': 'fe-ca-ferrite',
    'nacf': 'na-ca-ferrite',
    'hmag': 'high-pressure-magnetite',
    'crcf': 'cr-ca-ferrite',
    'mnal': 'mg-nal-phase',
    'fnal': 'fe-nal-phase',
    'nnal': 'na-nal-phase',
    'fea': 'alpha-iron',#
    'feg': 'gamma-iron',#
    'fee': 'epsilon-iron',#
}





# Deep-copy dict-of-lists so appending state variables does not mutate
# the compositional-only dictionaries.
COMPONENTS_IN_PHASES = {
    phase: list(components)
    for phase, components in COMPOSITIONAL_COMPONENTS_IN_PHASES.items()
}
COMPONENTS_IN_PHASES_HEFESTO = {
    phase: list(components)
    for phase, components in COMPOSITIONAL_COMPONENTS_IN_PHASES_HEFESTO.items()
}


state_vars_to_add = [
            'mass (gm)',
            'rho (gm/cc)',
            'H (kJ)',
            'S (J/K)',
            'V (cc)'
        ]

state_vars_to_add_HeFESTo = [
            'total (moles)',
            'rho (gm/cc)',
            'V (cc)'
        ]

        
for phase in COMPONENTS_IN_PHASES.keys():
    if phase not in ['melts-liquid', 'System_main', 'Bulk_comp']:
        for state_var in state_vars_to_add:
            if state_var == 'mass (gm)':
                COMPONENTS_IN_PHASES[phase] = ['mass (gm)'] + COMPONENTS_IN_PHASES[phase] # Add mass to the front, the rest to the back. 
            else:
                COMPONENTS_IN_PHASES[phase].append(state_var)

for phase in COMPONENTS_IN_PHASES_HEFESTO.keys():
    if phase not in ['System_main', 'Bulk_comp', 'Bulk_comp_elements']:
        for state_var in state_vars_to_add_HeFESTo:
            if state_var == 'mass (gm)':
                COMPONENTS_IN_PHASES_HEFESTO[phase] = ['mass (gm)'] + COMPONENTS_IN_PHASES_HEFESTO[phase] # Add mass to the front, the rest to the back. 
            else:
                COMPONENTS_IN_PHASES_HEFESTO[phase].append(state_var)


# Convert to immutable MappingProxyType to prevent accidental modifications
# that could affect other DatasetIndexer / ml_indexer instances
COMPOSITIONAL_COMPONENTS_IN_PHASES = MappingProxyType(COMPOSITIONAL_COMPONENTS_IN_PHASES)
COMPONENTS_IN_PHASES = MappingProxyType(COMPONENTS_IN_PHASES)
COMPONENTS_IN_PHASES_HEFESTO = MappingProxyType(COMPONENTS_IN_PHASES_HEFESTO)
COMPOSITIONAL_COMPONENTS_IN_PHASES_HEFESTO = MappingProxyType(COMPOSITIONAL_COMPONENTS_IN_PHASES_HEFESTO)

HEFESTO_ABBREVIATION_TO_SHORT_NAMES = MappingProxyType(HEFESTO_ABBREVIATION_TO_SHORT_NAMES)
OXIDE_MOLAR_MASSES = MappingProxyType(OXIDE_MOLAR_MASSES)


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

