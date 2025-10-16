# Getting relevant libraries

import numpy as np 
import random
import matplotlib.pyplot as plt
import pandas as pd
import random
plt.rcParams['figure.figsize'] = [10, 7]
import re
import molmass as ms
import torch
import os
import shutil
from pathlib import Path

# SET WHERE YOUR DATA IS KEPT HERE. IF IN YOUR WORKING DIRECTORY, MAKE IT "", an empty string
external_base = "D:/"


if external_base[-1] != '/' and len(external_base): # Assumed to end in slash in this code
    external_base += '/'

def internal_dir(MELTSModel): 
    return f'Workspace/{MELTSModel}Datasets/'
def external_dir(MELTSModel):
    return external_base + internal_dir(MELTSModel)

def delete_files_with_keyword(directory, keyword, dry_run=True):
    """
    Deletes all files in `directory` whose names contain `keyword`.

    Args:
        directory (str): Path to the target directory.
        keyword (str): Keyword to match in filenames.
        dry_run (bool): If True, only print files that would be deleted.

    Example:
        delete_files_with_keyword("/path/to/folder", "temp", dry_run=False)
    """
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        return

    deleted_count = 0
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and keyword in filename:
            if dry_run:
                print(f"[DRY RUN] Would delete: {filepath}")
            else:
                try:
                    os.remove(filepath)
                    print(f"Deleted: {filepath}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {filepath}: {e}")

    if dry_run:
        print("\nDry run complete. No files were actually deleted.")
    else:
        print(f"\nDeleted {deleted_count} files containing '{keyword}'.")


def QFM_fO2(P, K):
    trans1 = 573 + (0.025 * P)
    if K > trans1:
        A = -25096.3
        B = 8.735
        D = 0.11
    else:
        A = -26455.3
        B = 10.344
        D = 0.092
    K += 273.15 # Celsius to Kelvin
    logfo2 = (A/K) + B + ((D * (P-1)) / K)
    return(logfo2)

# Compile once, use many times
_number_pattern = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

def random_char(y):
    return ''.join(random.choice(string.ascii_letters) for x in range(y))
_number_pattern = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')
def pull_number(string):
    match = _number_pattern.search(string)
    return float(match.group()) if match else np.nan
def pull_letter(string, symbols = False):
    letters = ''
    accepted_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    if symbols:
        accepted_chars += '_+=-,.<>?;[]{}\|!@#$%^&*() '
    for char in string:
        if char in accepted_chars:
            letters += char
    return letters
def concat_all(*args):
    return ''.join(str(arg) for arg in args)
def identify_binaries(digits):
    """Returns numpy array of all unique binaries possible given a number of digits"""
    if 2**digits > 1E7:
        return(str(f"imagine there are {2**digits} of combinations supplied here. We aren't paid enough to actually generate them :P"))
    digits = int(digits)
    binaries = np.zeros((2,digits))
    binaries[1,0] = 1
    
    for b in range(1,digits):
        new_binaries = np.copy(binaries)
        new_binaries[:,b] = 1
        binaries = np.append(binaries, new_binaries, axis = 0)
        
    return binaries.astype(int)

def safe_float(x):
    try:
        return float(x)
    except:
        return 0.0
    
safe_convert = np.vectorize(safe_float)

def blur_binary_boundaries(arr):
    """
    For blurring boundaries between phase occurence and disappearence for training.
    Given a 2D binary array (0s and 1s), modifies boundary transitions (column-wise)
    such that:
    - A 1 that borders a 0 is set to 0.7
    - A 0 that borders a 1 is set to 0.3

    Returns a new array with blurred boundaries.
    """
    arr = arr.astype(float)  # Ensure output is float to allow fractional values
    blurred = arr.copy()

    # Shifted arrays for comparison: above and below
    above = np.roll(arr, 1, axis=0)
    below = np.roll(arr, -1, axis=0)

    # Identify boundaries (only between 1s and 0s)
    boundary_mask = (arr != above) | (arr != below)

    # Only consider cases where neighbors are valid (not from wrap-around)
    boundary_mask[0, :] &= (arr[0, :] != below[0, :])  # Top row: compare only below
    boundary_mask[-1, :] &= (arr[-1, :] != above[-1, :])  # Bottom row: compare only above

    # Set blurred values
    blurred[(boundary_mask) & (arr == 1)] = 0.7
    blurred[(boundary_mask) & (arr == 0)] = 0.3

    return blurred

    
def grid_sample(params, table=np.array([])):
    """Generates a numpy array grid sample recursively for arbitrary parameters.
    Let params be a nested list, with each sublist of [min, max, len] passed to np.linspace.
    Order of params determines column order in the output table."""
    
    params = list(params)  # Copy to avoid side-effects
    param = params.pop()
    new_col = np.linspace(*param).reshape((-1,1))
    
    if not table.shape[0]:
        table = new_col
    else:
        table = np.append(np.repeat(new_col, table.shape[0], axis=0),
                          np.tile(table, (new_col.shape[0], 1)), axis=1)
    
    if len(params):
        return grid_sample(params, table)
    else:
        return table

def squash_to_range(x, min_=0.1, max_=0.95):
    return x * (max_ - min_) + min_

def unsquash_from_range(x, min_=0.1, max_=0.95):
    return (x - min_) / (max_ - min_)

def QFM_fO2(P, K, use_torch=False):
    """Calculate log10 fO2 along the QFM buffer.
    
    Parameters:
    P : array-like
        Pressure in bars.
    K : array-like
        Temperature in Kelvin.
    use_torch : bool
        If True, use PyTorch tensors; otherwise, use NumPy arrays.
        
    Returns:
    log10(fO2) : array-like
        Logarithm base 10 of oxygen fugacity.
    """
    xp = torch if use_torch else np  # shorthand for backend

    trans1 = 573 + (0.025 * P)
    lowKind = K > trans1

    output = xp.zeros_like(K)

    if xp.any(lowKind):
        A = -25096.3
        B = 8.735
        D = 0.11
        output = output.clone() if use_torch else output  # avoid modifying shared memory
        output[lowKind] = (A / K[lowKind]) + B + ((D * (P[lowKind] - 1)) / K[lowKind])

    if xp.any(~lowKind):
        A = -26455.3
        B = 10.344
        D = 0.092
        output = output.clone() if use_torch else output
        output[~lowKind] = (A / K[~lowKind]) + B + ((D * (P[~lowKind] - 1)) / K[~lowKind])

    return output

def Fe2O3_FeO_ratio(fO2, T, P, composition, use_torch = False, device = 'cpu'):
    """
    Calculate ln(X_Fe2O3 / X_FeO) using Equation 7 from Kress & Carmichael (1991).

    Parameters:
    - fO2: oxygen fugacity (in atm or bar, same unit used in the original calibration)
    - T: temperature in Kelvin
    - P: pressure in Pa
    - composition: array (n x 5) of oxide compositions, NORMALIZED to sigma(X_i) = 1
        'Al2O3', 'FeO*', 'CaO', 'Na2O', 'K2O'

    Returns:
    - X_Fe2O3 / X_FeO
    """

    # Constants from Table 7 for natural melts
    a = 0.196
    b = 1.1492e4
    c = -6.675
    """d = {
        'Al2O3': -2.243,
        'FeO*': -1.828,
        'CaO': 3.201,
        'Na2O': 5.854,
        'K2O': 6.215
    }"""
    d = np.array([-2.243, -1.828, 3.201, 5.854, 6.215]).reshape(5,1) # set up for matrix multiplication for sum term
    e = -3.36
    f = -7.01e-7
    g = -1.54e-10
    h = 3.85e-17
    T0 = 1673.0  # Kelvin

    if use_torch:
        #print(f"fO2 size: {fO2.size()}")
        #print(f"T size: {T.size()}")
        #print(f"P size: {P.size()}")
        d = torch.tensor(d, dtype = torch.float32, device = device)
        
        dX_sum = composition @ d
        #print(f"Sum Size inner function: {dX_sum.size()}")
        
        ln_ratio = (
            a * torch.log(fO2) +
            b / T +
            c +
            dX_sum.flatten() +
            e * (1 - (T0 / T) - (torch.log(T / T0))) +
            f * P / T +
            g * (T - T0) * P / T +
            h * P ** 2 / T
        )
        #print(f"Return size inner function: {torch.exp(ln_ratio)}")
        return torch.exp(ln_ratio)
    
    else:
        dX_sum = composition @ d
        
        ln_ratio = (
            a * np.log(fO2) +
            b / T +
            c +
            dX_sum +
            e * (1 - (T0 / T) - (np.log(T / T0))) +
            f * P / T +
            g * (T - T0) * P / T +
            h * P ** 2 / T
        )
    
        return np.exp(ln_ratio)


"""Define all labeling systems / dictionaries for this module"""

component_indices = {
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
        'highsanidine': 34
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
        'corundum':60 
    },
    'alloy-solid': {
        'mass (gm)': 61,
        'Fe-metal': 62
        #'Ni-metal': 63
    },
    'analcime': { #USED TO BE ALLOY LIQUID
        'mass (gm)': 63,
        'leucite': 64,
        'analcime': 65,
        'na-leucite': 66
        #'Fe-liquid': 64,
        #'Ni-liquid': 65
    },
    'apatite': {
        'mass (gm)': 67
    },
    'whitlockite': {
        'mass (gm)': 68
    },
    'quartz': {
        'mass (gm)': 69
    },
    'tridymite': {
        'mass (gm)': 70
    },
    'cristobalite': {
        'mass (gm)': 71
    },
    'amphibole': {
        'mass (gm)': 72
    },
    'muscovite': {
        'mass (gm)': 73
    },
    'fluid': {
        'mass (gm)': 74
    },
    'water': {
        'mass (gm)': 74
    },
    'melts-liquid': {
        'liq mass (gm)': 75,
        'wt% SiO2': 76,
        'wt% TiO2': 77,
        'wt% Al2O3': 78,
        'wt% FeO': 79,
        'wt% MgO': 80,
        'wt% CaO': 81,
        'wt% Na2O': 82,
        'wt% K2O': 83,
        'wt% P2O5': 84,
        'wt% MnO': 85,
        'wt% H2O': 86,
        'wt% Cr2O3': 87,
        'wt% NiO': 88,
        'wt% Fe2O3': 89
    }
}

database_headers = []
for pha, val in component_indices.items():
    if pha not in ['alkali-feldspar', 'water']:
        for com in list(val.keys()):
            database_headers.append(f"{com}({pha})")

component_indices['System_main']['viscosity'] = 90
database_headers.append("viscosity(System_main)")
component_indices['System_main']['H'] = 91
database_headers.append("H(System_main)")
component_indices['System_main']['Cp'] = 92
database_headers.append("Cp(System_main)")
component_indices['System_main']['S'] = 93
database_headers.append('S(system_main)')
component_indices['System_main']['V'] = 94
database_headers.append('V(System_main)')
component_indices['System_main']['dVdP*10^6'] = 95
database_headers.append('dVdP*10^6(System_main)')
component_indices['System_main']['dVdT*10^6'] = 96
database_headers.append('dVdP*10^6(System_main)')
# Now add rho and viscosity fields sequentially

next_index = 97
# Now add rho and viscosity fields sequentially

# List of phases (in order of appearance)
phases_in_order = [
    'olivine', 'orthopyroxene', 'clinopyroxene', 'spinel',
    'plagioclase', 'alkali-feldspar', 'garnet', 'nepheline', 'leucite',
    'biotite', 'rhm-oxide', 'alloy-solid', 'analcime', 'apatite', 'whitlockite', 'quartz', 'tridymite', 
    'muscovite', 'fluid'
]


# Add density, enthalpy field
for phase in phases_in_order:
    component_indices[phase]['rho (gm/cc)'] = next_index
    next_index += 1
    database_headers.append(f"{'rho (gm/cc)'}({phase})")
    component_indices[phase]['H (kJ)'] = next_index
    next_index += 1
    database_headers.append(f"{'H (kJ)'}({phase})")
    component_indices[phase]['S (J/K)'] = next_index
    next_index += 1
    database_headers.append(f"{'S (J/K)'}({phase})")
    component_indices[phase]['V (cc)'] = next_index
    next_index += 1
    database_headers.append(f"{'V (cc)'}({phase})")

# Add viscosity field for melt only
component_indices['melts-liquid']['liq rho (gm/cc)'] = next_index
component_indices['melts-liquid']['liq vis (log 10 poise)'] = next_index + 1
component_indices['melts-liquid']['liq H (kJ)'] = next_index + 2
component_indices['melts-liquid']['liq S (J/K)'] = next_index + 3
component_indices['melts-liquid']['liq V (cc)'] = next_index + 4
database_headers.append('rho (gm/cc)(melts-liquid)')
database_headers.append('liq vis (log 10 poise)')
database_headers.append('liq H (kJ)')
database_headers.append('liq S (J/K)')
database_headers.append('liq V (cc)')
print(component_indices)

mass_indices = [] #exclude amphibole
for phase, props in component_indices.items():
    if phase in (phases_in_order + ['melts-liquid']):
        for key, ind in list(props.items()):
            if 'mass' in key:
                mass_indices.append(ind)
mass_indices = np.unique(mass_indices)


WRkeysOld = ['SiO2','TiO2', 'Al2O3', 'FeO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'MnO', 'H2O', 'Cr2O3', 'NiO']
OxidesOld = WRkeysOld + ['Fe2O3']
ElkeysOld = ['Si','Ti','Al','Fe','Mg','Ca','Na','K','P','Mn','H','Cr','Ni']
MtotOld = np.array([ms.Formula(MM).mass for MM in OxidesOld]).reshape(14,1) #  14x1 , For phase molar mass from oxide moles
compToOxOld = pd.read_csv('BackEnds/Old_Transforms/compToOx - Copy (2).csv').to_numpy()[:,1:].astype(np.float32)
MinvOld = np.diag([1/ms.Formula(MM).mass for MM in OxidesOld])
MMOld = np.diag([ms.Formula(mm).mass for mm in OxidesOld])
oxToElOld = pd.read_csv('BackEnds/Old_Transforms/OxToEl_olderJuly2525.csv').to_numpy()[:,1:].astype(np.float32).T

WRkeys = ['SiO2','TiO2', 'Al2O3', 'FeO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'H2O', 'Cr2O3']
Oxides = WRkeys + ['Fe2O3']
Elkeys = ['Si','Ti','Al','Fe','Mg','Ca','Na','K','P','H','Cr']
Mtot = np.array([ms.Formula(MM).mass for MM in Oxides]).reshape(len(Oxides),1) # For phase molar mass from oxide moles
compToOxLoad = pd.read_csv('BackEnds/compToOx.csv').to_numpy()[:,1:].astype(np.float32)
PxSpTransform = pd.read_csv('BackEnds/PxSp_Comp_Transform.csv').to_numpy()[:,1:].astype(np.float32)
compToOx = np.linalg.inv(PxSpTransform) @ compToOxLoad #pc,co->po
MM = np.diag([ms.Formula(MM).mass for MM in Oxides])

Minv = np.diag([1/ms.Formula(MM).mass for MM in Oxides])
oxToEl = pd.read_csv('BackEnds/OxToEl.csv').to_numpy()[:,1:].astype(np.float32).T

label_indices = {} # Used for mapping phases to full components matrix

label_names = []
detail_label_indices = {} # More used for index mapping
label_indices_comp = {} # Mapping to compositionally variable intensive component output

detail_ind = 0
index = 0

for phase, components in component_indices.items(): # Build indices for ML-ready data, will direct translation to element moles

    if phase not in ['System_main', 'alkali-feldspar','water', 'amphibole', 'melts-liquid',
                     'cristobalite']:
        phase_inds = []

        if phase == 'k-feldspar' or (len(components) > 5 and phase != 'alloy-solid'):
            detail_label_indices[phase] = {}
            comp_inds = []
            for component in components:
                if component not in ['liq rho (gm/cc)', 'liq vis (log 10 poise)','liq H (kJ)', 
                                     'liq mass (gm)',  'liq H (kJ)', 'liq S (J/K)', 'liq V (cc)',
                                     'mass (gm)', 'rho (gm/cc)', 'H (kJ)', 'S (J/K)', 'V (cc)']: #State variables, not chemistry

                        if component not in ['tephroite', 'co-olivine', 'ni-olivine', 'pyrophanite',
                                        'Mn', 'Ni']: # Components that are not used, because NiO and MnO are not
                            phase_inds += [index]
                            if component != 'Fe-metal':
                                detail_label_indices[phase][component] = detail_ind
                                comp_inds.append(detail_ind)
                                detail_ind +=1
                            index += 1
                            label_names.append(component)
            label_indices_comp[phase] = np.array(comp_inds)
        else: 
            phase_inds += [index]
            index += 1

            label_names.append(phase)
        label_indices[phase] = phase_inds

label_indices['melts-liquid'] = list(range(index,index+len(Elkeys)))

label_names += Elkeys

detail_label_indices['melts-liquid'] = {}
comp_inds = []
for key in Elkeys:
    detail_label_indices['melts-liquid'][key] = detail_ind
    comp_inds.append(detail_ind)
    detail_ind += 1
label_indices_comp['melts-liquid'] = np.array(comp_inds)
    
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
        
ncomps = label_indices['melts-liquid'][-1]+1 #(C)
ncompsVaried = label_indices_comp['melts-liquid'][-1]+1 #(V)
nphases = mass_phasedict['melts-liquid']+1 #(P)
        
compositionally_variable_binaries = []
compositionally_variable_subset = []
phaseToCompMap = np.zeros((nphases,ncomps)) #(P,C) General
variedToAllComp = np.zeros((ncompsVaried,ncomps)) #(V,C)

for p, phase in enumerate(list(label_indices.keys())):
    phaseToCompMap[p,label_indices[phase]] = 1
    if phase in compositionally_variable_phases:
        compositionally_variable_binaries.append(1)
        compositionally_variable_subset += label_indices[phase]
        variedToAllComp[label_indices_comp[phase], label_indices[phase]] = 1
    else:
        compositionally_variable_binaries.append(0)

comp_variable_IDMAT = torch.tensor(np.diag(compositionally_variable_binaries), dtype = torch.float)
is_fixed = ~ (np.array(compositionally_variable_binaries).astype(bool))
fixed_phaseToCompMap = (is_fixed.reshape(1,-1) @ phaseToCompMap)
#compositional_component_subset = np.array(compositional_component_subset)
compositionally_variable_subset = np.array(compositionally_variable_subset)
compositional_component_subset = np.copy(compositionally_variable_subset)
print(compositional_component_subset)

oxide_dict = {}
for i, ox in enumerate(Oxides):
    oxide_dict[ox] = i


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
comp_mappings = np.zeros((j, len(comp_mappingsL))) # Build Binary Matrix to project phases to components
#comp_mappings_tensor = torch.zeros(j, len(comp_mappingsL))
for col, row in enumerate(comp_mappingsL):
    comp_mappings[row, col] = 1

boolTransCompToOx = np.copy(compToOx)
boolTransCompToOx[:,3] += boolTransCompToOx[:,-1] #Compile non-negative irons for use with element inputs
boolTransCompToOx = (boolTransCompToOx[:,:-1] !=0).astype(int) # For sending logits to neg inifinity to prevent prediction of components with absent atoms

active_ox_dict = {'olivine':[oxide_dict[ox] for ox in ['MgO', 'FeO','CaO']],
 'orthopyroxene':[oxide_dict[ox] for ox in ['CaO', 'MgO', 'FeO', 'Fe2O3', 'Na2O', 'Al2O3', 'TiO2']],
 'clinopyroxene':[oxide_dict[ox] for ox in ['CaO', 'MgO', 'FeO', 'Fe2O3', 'Na2O', 'Al2O3', 'TiO2']],
 'spinel':[oxide_dict[ox] for ox in ['Cr2O3', 'MgO', 'FeO', 'Fe2O3', 'Al2O3', 'TiO2']],
 'plagioclase':[oxide_dict[ox] for ox in ['Na2O', 'K2O', 'CaO']],
 'k-feldspar':[oxide_dict[ox] for ox in ['Na2O', 'K2O', 'CaO']],
 'garnet':[oxide_dict[ox] for ox in ['FeO', 'CaO', 'MgO']],
 'nepheline':[oxide_dict[ox] for ox in ['Na2O', 'K2O', 'CaO', 'Al2O3']],
 'leucite':[oxide_dict[ox] for ox in ['Na2O', 'K2O', 'H2O']],
 'biotite':[oxide_dict['MgO']],
 'rhm-oxide':[oxide_dict[ox] for ox in ['MgO', 'FeO', 'Fe2O3', 'TiO2']],
 'analcime':[oxide_dict[ox] for ox in ['Na2O', 'K2O', 'H2O']],
 'melts-liquid':[oxide_dict[ox] for ox in Oxides]}

class MELTS_Converter():
    """Conversion functionality of the Emulator class without neural network models"""
    def __init__(self, 
                 Norm_Stat_Name = 'binary_saturation_renormalization_July7_stats.txt',#'binary_saturation_resampled_normalization_stats.txt', 
                 Component_Stat_Name = 'components_renormalization_July7_stats.txt'):#'components_resampled_normalization_stats.txt'):
        
        self.compToOx = torch.tensor(compToOx[:label_indices['fluid'][-1]+1], dtype = torch.float)
        self.oxToEl = torch.tensor(oxToEl, dtype = torch.float)
        self.Minv = torch.tensor(Minv, dtype = torch.float)
        self.MM = torch.tensor(MM, dtype = torch.float)
        self.Mtot = torch.tensor(Mtot, dtype = torch.float).flatten()
        
        features_min = []
        features_range = []
        component_min = []
        component_range = []
        
        with open(Norm_Stat_Name, 'r') as BSF:
            for i, line in enumerate(BSF):
                if i == 0:
                    continue
                if not len(line):
                    break
                parts = line.split(' ')
                if 'min' in parts[0]:
                    minimum = pull_number(parts[1])
                    features_min.append(minimum)
                if 'max' in parts[0]:
                    Drange = pull_number(parts[1]) - minimum
                    features_range.append(Drange)
                
        with open(Component_Stat_Name, 'r') as CSF:
            for i, line in enumerate(CSF):
                if i == 0:
                    continue
                if not len(line):
                    break
                parts = line.split(' ')
                if 'min' in parts[0]:
                    minimum = pull_number(parts[1])
                    component_min.append(minimum)
                if 'max' in parts[0]:
                    Drange = pull_number(parts[1]) - minimum
                    component_range.append(Drange)
        
        self.norm_features = Normalizer(torch.tensor(features_min)[:15], torch.tensor(features_range)[:15]) # Exclude Nickel
        self.norm_components = Normalizer(torch.tensor(component_min)[:label_indices['fluid'][-1]+1], torch.tensor(component_range)[:label_indices['fluid'][-1]+1])
        self.norm_liquid = Normalizer(torch.tensor(component_min)[label_indices['fluid'][-1]+1:], torch.tensor(component_range)[label_indices['fluid'][-1]+1:])
        self.norm_labels = Normalizer(torch.tensor(component_min), torch.tensor(component_range))
        
   
    """def convertMolToIntensiveWt(self, components, features):
        #For old model archetecture predicting ONLY extensive component moles. Code could be adapted but there are other working versions
        liquid = components[:,-13:]
        solids = components[:,:-13]
        nrows = components.size()[0]
        massTens = torch.zeros((nrows, len(all_phases)))
        compTens = torch.zeros((nrows, len(compositionally_variable_phases), len(Oxides)))
        
        
        for phase in all_phases:
            if phase != 'melts-liquid':
                massTens[:,mass_phasedict[phase]] = components[:,label_indices[phase]] @ self.compToOx[label_indices[phase]] @ self.Mtot # Not Normalized!
                if phase in compositionally_variable_phases: #Variable componsition. 
                    unNormed = components[:,label_indices[phase]] @ self.compToOx[label_indices[phase]] @ self.MM
                    row_sums = unNormed.sum(dim=1, keepdim=True)
                    nonzero_mask = (row_sums != 0) # Replace zeros with 1 to avoid division by zero
                    row_sums[~nonzero_mask] = 1.0
                    renormed = unNormed * (100.0 / row_sums)# Normalize to 100 wt%
                    renormed[~nonzero_mask.expand_as(unNormed)] = 0.0 # Set zero-sum rows to 0 to be extra careful
                    compTens[:, comp_phasedict[phase], :] = renormed 
            else: 
                #For the liquid, We go to moles oxide and then speciate the iron between ferric and ferrous based on fO2.
                fO2_composition_Nos = torch.tensor([oxide_dict[ox] for ox in ['Al2O3', 'FeO', 'CaO', 'Na2O', 'K2O']]) # For Feeding in relevant molar oxides for Kress and Carmichael, 
                fO2_composition_ind = torch.zeros(13) # For Feeding in relevant molar oxides for Kress and Carmichael, 
                fO2_composition_ind[fO2_composition_Nos] = 1
                fO2_composition_ind = fO2_composition_ind.to(torch.bool)
                unNormed = components[:,label_indices[phase]] @ torch.linalg.inv(self.oxToEl[:13]) # mole oxides, no ferric iron
                row_sums = unNormed.sum(dim=1, keepdim=True) # Temporary renormalization for iron speciation equation
                nonzero_mask = row_sums != 0 # Replace zeros with 1 to avoid division by zero
                row_sums[~nonzero_mask] = 1.0
                temp_renorm = unNormed * (1 / row_sums)# Normalize to one oxide mole
                temp_renorm[~nonzero_mask.expand_as(unNormed)] = 0.0 # Set zero-sum rows to 0 to be extra careful
                IronR = Fe2O3_FeO_ratio(fO2=10**(QFM_fO2(K = features[nonzero_mask.flatten()][:,1]+273, P = features[nonzero_mask.flatten()][:,0], use_torch = True)+features[nonzero_mask.flatten()][:,2]), 
                                        T=features[nonzero_mask.flatten()][:,1]+273, P=1e5*features[nonzero_mask.flatten()][:,0], composition=temp_renorm[nonzero_mask.flatten()][:,fO2_composition_ind], use_torch = True)
                ferricPerTot = 1/(2+(1/IronR))
                ferrousPerTot = 1/((2*IronR)+1)
                
                ferric = torch.zeros((nrows,1)) # Placeholder column to recieve ferric iron
                # Get indices where the mask is True
                idx = nonzero_mask.flatten().nonzero(as_tuple=True)[0]

                # Modify ferric[:, 0]
                ferric[idx, 0] = unNormed[idx, oxide_dict['FeO']] * ferricPerTot

                # Modify unNormed[:, FeO] in-place
                unNormed[idx, oxide_dict['FeO']] *= ferrousPerTot

                #ferric[nonzero_mask.flatten()][:,0] = (unNormed[nonzero_mask.flatten()][:, oxide_dict['FeO']] * ferricPerTot)
                #print(ferric)
                #print(unNormed[0])
                #unNormed[nonzero_mask.flatten()][:,oxide_dict['FeO']] = unNormed[nonzero_mask.flatten()][:,oxide_dict['FeO']]*ferrousPerTot # Edit ferrous iron in place
                #print(unNormed[0])
                unNormed = torch.cat([unNormed,ferric], dim = 1) # moles oxides with Fe2O3
                
                massTens[:,mass_phasedict[phase]] = unNormed @ self.Mtot # Not Normalized!
        
                unNormedWt = (unNormed @ self.MM) 
                row_sums = unNormedWt.sum(dim=1, keepdim=True) # Now Renormalizing for intensive mass: 100 wt%
                renormed = unNormedWt * (100.0 / row_sums)# Normalize to 100 wt%
                renormed[~nonzero_mask.expand_as(unNormed)] = 0.0 # Set zero-sum rows to 0 to be extra careful
                compTens[:, comp_phasedict[phase], :] = renormed #Finally
        
        row_sums = massTens.sum(dim=1, keepdim=True) # Normalize Mass Table to 100 wt
        massTens = massTens * (100.0 / row_sums)
        
        return massTens, compTens"""
                    
    def convertOxToMol(self, features):
        colsize = features.shape[1]-3
        unclosed = features[:,3:] @ self.Minv[:colsize,:colsize] @ oxToEl[:colsize]
        closedmoles = unclosed / unclosed.sum(dim=1, keepdim=True)
        return torch.cat([features[:,:3], closedmoles], dim = 1)
    
def projected_nnls(A, b, max_iter=10, lr=0.1):
    # A: (batch, n_elements, n_phases): Element contribution from each phase
    # b: (batch, n_elements): Negative Element deficits in liquid
    r = torch.zeros(A.shape[0], A.shape[2], device=A.device)

    for _ in range(max_iter):
        residual = (A @ r.unsqueeze(2)).squeeze(2) - b  # (batch, n_elements)
        grad = (A.transpose(1, 2) @ residual.unsqueeze(2)).squeeze(2)  # (batch, n_phases)
        r = r - lr * grad
        r = torch.clamp(r, min=0.0)  # projection
    return r  # shape: (batch, n_phases)

class Normalizer:
    """Quick Normalizing object that holds minima and ranges for a dataset and converts into and out of [0,1]
    min-max normalization for interfacing with neural networks"""
    
    def __init__(self, min_tensor, range_tensor, cuda = False):
        
        if cuda:
            self.miner = min_tensor.cuda()
            self.ranger = range_tensor.cuda()
            self.dev = 'cuda'
        else:
            self.miner = min_tensor.cpu()
            self.ranger = range_tensor.cpu()
            self.dev = 'cpu'


    def denorm(self, x):

        return x * self.ranger + self.miner
    
    def norm(self, x):
        if isinstance(x, np.ndarray):
            out = np.zeros_like(x, dtype=float)
            mask = self.ranger != 0
            out[:,mask] = (x[:,mask] - self.miner[mask]) / self.ranger[mask]
            return out
        elif isinstance(x, torch.Tensor):
            out = torch.zeros_like(x, dtype=torch.float, device = self.dev)
            mask = self.ranger != 0
            out[:,mask] = (x[:,mask] - self.miner[mask]) / self.ranger[mask]
            return out
        else:
            raise TypeError("Input must be a NumPy array or a PyTorch tensor.")



def move_file(src_path, dst_path, overwrite=False):
    """
    Move a file from the current working directory to a destination directory.

    Args:
        src_path (str): Name of the file in the current working directory.
        dst_path (str): Destination directory
        overwrite (bool): If True, overwrite any existing file with the same name.
    """

    # Ensure the source file exists
    #src_path = os.path.join(os.getcwd(), src_path)
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"Source file not found: {src_path}")

    # Ensure destination directory exists
    #s.makedirs(dst_path, exist_ok=True)

    # Destination file path
    #dst_path = os.path.join(dst_dir, src_filename)

    if src_path == dst_path:
        return

    # Handle overwrite
    if os.path.exists(dst_path):
        if overwrite:
            os.remove(dst_path)
        else:
            raise FileExistsError(f"Destination file already exists: {dst_path}")

    # Move the file
    shutil.move(src_path, dst_path)
    print(f"Moved '{src_path}' to '{dst_path}' successfully.")

def move_files_with_extension(extension, dst_dir, src_dir = None, overwrite=False):
    """
    Move all files with a given extension from the current working directory
    to the destination directory.

    Args:
        extension (str): File extension, e.g., '.csv' or '.dat'
        dst_dir (str or Path): Destination directory path
        src_dir (str or Path): Source location from working directory, Default none is cwd. 
        overwrite (bool): If True, overwrite existing files in destination
    """

    if src_dir == dst_dir: 
        return

    if src_dir is not None:
        src_path = Path.cwd() / src_dir
    else:
        src_path = Path.cwd()


    #cwd = Path.cwd()
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Find all matching files in current directory
    files_to_move = [f for f in src_path.iterdir() if f.is_file() and f.suffix == extension]

    if not files_to_move:
        print(f"No files with extension '{extension}' found in {cwd}.")
        return

    for file_path in files_to_move:
        dest_path = dst_dir / file_path.name

        if dest_path.exists():
            if overwrite:
                dest_path.unlink()  # Remove existing file
            else:
                print(f"Skipping {file_path.name}, already exists in destination.")
                continue

        shutil.move(str(file_path), str(dest_path))
        print(f"Moved {file_path.name} -> {dst_dir}")

def masked_column_assign(target: torch.Tensor,
                         row_mask: torch.Tensor,
                         col_idx: torch.Tensor,
                         values: torch.Tensor):
    """
    Efficient in-place assignment for a masked subset of rows and explicit column indices.

    Args:
        target (torch.Tensor): 2D tensor to modify in place.
        row_mask (BoolTensor): Mask selecting which rows to modify (shape [n_rows]).
        col_idx (1D LongTensor): Indices of columns to assign to.
        values (Tensor): Values to assign (shape [sum(row_mask), len(col_idx)]).
    """
    # Sanity checks
    assert target.dim() == 2, "target must be 2D"
    assert values.dim() == 2, "values must be 2D"
    assert values.size(1) == col_idx.numel(), "values must match number of columns"
    assert values.size(0) == row_mask.sum(), "values must match number of selected rows"

    # Get actual row indices (1D LongTensor)
    row_idx = torch.nonzero(row_mask, as_tuple=False).squeeze(1)

    # Perform in-place indexed assignment efficiently
    target.index_put_((row_idx.unsqueeze(1).expand(-1, col_idx.numel()),
                       col_idx.unsqueeze(0).expand(row_idx.numel(), -1)),
                      values)



"""        
def Iron_Speciator(self, oxides, Normedfeatures):
    #Let oxides input be a tensor of size nxO, where columns are liquid molar oxides (except for Fe2O3)
    #Features are assumed to be normalized. Output is oxides, now with Fe2O3
    features = self.norm_features.denorm(Normedfeatures)[:,:3]
    unNormed = oxides.clone()[:,:len(Elkeys)] # Ensure we don't grab the potentially empty ferric column
    #print(f"unNormed: {unNormed[:2]}")
    #print(f"Features: {features[3]}")
    fO2_composition_Nos = torch.tensor([oxide_dict[ox] for ox in ['Al2O3', 'FeO', 'CaO', 'Na2O', 'K2O']], device = self.dev) # For Feeding in relevant molar oxides for Kress and Carmichael, 
    fO2_composition_ind = torch.zeros(len(Elkeys), device = self.dev).to(torch.bool) # For Feeding in relevant molar oxides for Kress and Carmichael, 
    fO2_composition_ind[fO2_composition_Nos] = True
    row_sums = unNormed.sum(dim=1, keepdim=True) # Temporary renormalization for iron speciation equation
    nonzero_mask = row_sums != 0 # Replace zeros with 1 to avoid division by zero
    row_sums[~nonzero_mask] = 1.0
    temp_renorm = unNormed * (1 / row_sums)# Normalize to one oxide mole
    temp_renorm[~nonzero_mask.expand_as(unNormed)] = 0.0 # Set zero-sum rows to 0 to be extra careful
    #print(f"Temp renorm: {temp_renorm}")
    #print(f"Composition input: {temp_renorm[nonzero_mask.flatten()][:,fO2_composition_ind]}")
    # Get ferric/ferrous 
    IronR = Fe2O3_FeO_ratio(fO2=10**(QFM_fO2(K = features[nonzero_mask.flatten()][:,1]+273, P = features[nonzero_mask.flatten()][:,0], use_torch = True)+features[nonzero_mask.flatten()][:,2]), 
                            T=features[nonzero_mask.flatten()][:,1]+273, P=1e5*features[nonzero_mask.flatten()][:,0], composition=temp_renorm[nonzero_mask.flatten()][:,fO2_composition_ind], use_torch = True, device = self.dev)
    #print(f"IronR: {IronR}")
    ferricPerTot = 1/(2+(1/IronR))
    ferrousPerTot = 1/((2*IronR)+1)

    ferric = torch.zeros((unNormed.size()[0],1), device = self.dev, dtype = torch.float32) # Placeholder column to recieve ferric iron
    # Get indices where the mask is True
    idx = nonzero_mask.flatten().nonzero(as_tuple=True)[0]

    # Modify ferric[:, 0]
    ferric[idx, 0] = unNormed[idx, oxide_dict['FeO']] * ferricPerTot
    #print(f"Ferric: {ferric}")
    # Modify unNormed[:, FeO] in-place
    unNormed[idx, oxide_dict['FeO']] *= ferrousPerTot

    unNormed_out = torch.cat([unNormed,ferric], dim = 1) # moles oxides with Fe2O3
    #print(f"Out unNormed: {unNormed_out[:2]}")
    return unNormed_out"""