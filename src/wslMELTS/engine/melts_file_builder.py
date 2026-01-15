"""
MELTS file generation and formatting.

Contains functions for building MELTS input files from conditions arrays.
"""

import os
import numpy as np
from pathlib import Path
from ...nMELTS.utils.string_utils import pull_number

# Configuration constants (Define these in top level scripts?)
#alphaMELTSLocation = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'alphamelts-app-2.3.1-linux')
#EnsembleLocation = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GEOROC_SIMS')

# System names for phase suppression
systemNames = [
    'liquid',
    'olivine',
    'sphene',
    'garnet',
    'melilite',
    'orthopyroxene',
    'clinopyroxene',
    'aegirine',
    'aenigmatite',
    'cummingtonite',
    'clinoamphibole',
    'orthoamphibole',
    'hornblende',
    'biotite',
    'muscovite',
    'k-feldspar',
    'plagioclase',
    'quartz',
    'tridymite',
    'cristobalite',
    'nepheline',
    'kalsilite',
    'leucite',
    'corundum',
    'rutile',
    'perovskite',
    'spinel',
    'rhm-oxide',
    'ortho-oxide',
    'whitlockite',
    'apatite',
    'alloy-solid',
    'alloy-liquid',
    'sillimanite'
]

def expand_MC(conditions, deviation, length=20):
    """
    Take a 1D array, add random guassian noise to get MCMC sampling of normally distributed, uncorrelated conditions

    Generate an array with normal distributions of conditions.
    
    Give two slices of equal length to generate an array with normal distributions of
    conditions with std deviations as given in 'deviation'.
    
    Parameters:
    -----------
    conditions : np.ndarray
        Mean values for each condition
    deviation : np.ndarray
        Standard deviations for each condition
    length : int, default=20
        Number of samples to generate
        
    Returns:
    --------
    np.ndarray
        Expanded array with shape (length, n_conditions)
    """
    if np.shape(conditions) != np.shape(deviation):
        raise IndexError('Errors and Means unequal size')
    vals = np.shape(conditions)[0]
    expanded_array = np.empty((length, vals))

    for i in range(vals):
        expanded_array[:, i] = np.random.normal(
            conditions[i], deviation[i], length)
    return expanded_array


def AddMELTSLine(MELTSStr, key, val, end=0, delta=-2):
    """
    Use Headers and values to build string for .MELTS files calling ISOTHERMAL runs.
    
    Parameters:
    -----------
    MELTSStr : str
        Current MELTS file string
    key : str
        Parameter name (e.g., 'fO2', 'Pressure', 'Temperature', or oxide name)
    val : float
        Parameter value
    end : float, default=0
        End value for temperature (if applicable)
    delta : float, default=-2
        Increment value for temperature
        
    Returns:
    --------
    str
        Updated MELTS file string
    """
    if key == 'fO2':
        MELTSStr += 'Log fo2 Path: FMQ\n'
        MELTSStr += f'Log fo2 Offset: {val}\n'
    elif key == 'Pressure':
        MELTSStr += f'Initial Pressure: {val}\n'
        MELTSStr += f'Final Pressure: {val}\n'
        MELTSStr += 'Increment Pressure: 0\n'
    elif key == 'Temperature':
        MELTSStr += f'Initial Temperature: {val}\n'
        if end:
            MELTSStr += f'Final Temperature: {end}\n'
        else:
            MELTSStr += f'Final Temperature: {700}\n'
        MELTSStr += f'Increment Temperature: {delta}\n'
    elif 'O' in key:
        MELTSStr += f'Initial Composition: {key} {val}\n'
    else:
        MELTSStr += f'Initial Trace: {key} {val}\n'
    return MELTSStr


def AddMELTSLineCompression(MELTSStr, key, val, end=0, delta=50):
    """
     Use Headers and values to build string for .MELTS files calling ISOTHERMAL runs.
    
    Parameters:
    -----------
    MELTSStr : str
        Current MELTS file string
    key : str
        Parameter name
    val : float
        Parameter value
    end : float, default=0
        End pressure value (if applicable)
    delta : float, default=50
        Pressure increment
        
    Returns:
    --------
    str
        Updated MELTS file string
    """
    if key == 'fO2':
        MELTSStr += 'Log fo2 Path: FMQ\n'
        MELTSStr += f'Log fo2 Offset: {val}\n'
    elif key == 'Pressure':
        # Make pressures depending on what MELTS domain is implied by the value passed here
        if not end:
            if val > 10000:
                beginP = 8000 + (val % 20)
                endP = 45000 - (val % 20)  # Maybe pMELTS craps out at 30000 bars???
            if val <= 10000:
                beginP = 1 + (val % 20)
                endP = 12000 - (val % 20)
        else:
            beginP = val
            endP = end
        deltaP = delta
        MELTSStr += f'Initial Pressure: {beginP}\n'
        MELTSStr += f'Final Pressure: {endP}\n'
        MELTSStr += f'Increment Pressure: {deltaP}\n'
    elif key == 'Temperature':
        MELTSStr += f'Initial Temperature: {val}\n'
        MELTSStr += f'Final Temperature: {val}\n'
        MELTSStr += 'Increment Temperature: 0\n'
    elif 'O' in key:
        MELTSStr += f'Initial Composition: {key} {val}\n'
    else:
        MELTSStr += f'Initial Trace: {key} {val}\n'
    return MELTSStr


def makeMELTSStr(conditions, keys, end=True, fxtal=False, compression=False, delta=-3):
    """
    Transform slice of conditions with labels ('keys') into MELTS String. 
    I should make the input a dictionary.
    
    Parameters:
    -----------
    conditions : np.ndarray
        Array of condition values
    keys : np.ndarray or list
        Array of parameter names corresponding to conditions
    end : bool or float, default=True
        End value for temperature or True to use default
    fxtal : bool, default=False
        Whether to enable fractional crystallization mode
    compression : bool, default=False
        Whether this is a compression run
    delta : float, default=-3
        Temperature increment
        
    Returns:
    --------
    str
        Complete MELTS input file string
    """
    MELTSStr = 'Output: both\n'
    if np.shape(conditions)[0] != np.shape(keys)[0]:
        raise IndexError('Conditions and Keys unequal size')
    
    for i in range(np.shape(conditions)[0]):
        if compression:
            MELTSStr = AddMELTSLineCompression(MELTSStr, keys[i], conditions[i], end=end, delta=delta)
        else:
            MELTSStr = AddMELTSLine(MELTSStr, keys[i], conditions[i], end=end, delta=delta)
    
    MELTSStr += 'Suppress: rutile\n'
    
    if fxtal:
        MELTSStr += 'mode: fractionate solids\n'


    return MELTSStr


def suppressAllBut(MELTSStr, phase_names):
    """
    Suppresses everything except for specified phases by appending lines to MELTS file.
    
    Parameters:
    -----------
    MELTSStr : str
        Current MELTS file string
    phase_names : list
        List of phase names to keep (all others will be suppressed)
        
    Returns:
    --------
    str
        Updated MELTS file string with suppression lines
    """
    for phase in systemNames:
        if phase not in phase_names:
            MELTSStr += f'Suppress: {phase}\n'
    return MELTSStr
