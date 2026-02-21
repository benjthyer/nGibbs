"""
String manipulation utilities.

Extracted from Legacy/BackEnds/EmulatorLibrary.py
"""

import re
import string
import random
import numpy as np

# Compile once, use many times
_number_pattern = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')


def pull_number(string_val):
    """
    Extract the first number from a string.
    
    Args:
        string_val (str): String to extract number from
        
    Returns:
        float: Extracted number, or np.nan if no number found
    """
    match = _number_pattern.search(string_val)
    return float(match.group()) if match else np.nan


def pull_letter(string_val, symbols=False):
    """
    Extract letters (and optionally symbols) from a string.
    
    Args:
        string_val (str): String to extract from
        symbols (bool): If True, include symbols in extraction
        
    Returns:
        str: Extracted letters/symbols
    """
    letters = ''
    accepted_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    if symbols:
        accepted_chars += '_+=-,.<>?;[]{}\|!@#$%^&*() '
    for char in string_val:
        if char in accepted_chars:
            letters += char
    return letters


def concat_all(*args):
    """
    Concatenate all arguments into a single string.
    
    Args:
        *args: Variable number of arguments to concatenate
        
    Returns:
        str: Concatenated string
    """
    return ''.join(str(arg) for arg in args)


def random_char(y):
    """
    Generate a random string of letters.
    
    Args:
        y (int): Length of random string
        
    Returns:
        str: Random string of letters
    """
    return ''.join(random.choice(string.ascii_letters) for x in range(y))

def _deep_typer(lst, dtype = int):
    """Recursive (very soft) algorithm to transform all values within a list of arbitrary shape to a different dtype"""
    out_lst = []
    if isinstance(lst, list):
        for obj in lst:
            if isinstance(obj, list):
                out_lst.append(_deep_typer(obj, dtype))
            else:
                out_lst.append(dtype(obj))
    else:
        try:
            return dtype(lst) # not an iterable
        except:
            return lst
    
    return out_lst


def apply_type_conversions(nested_dict, conversions, default_dtype=None):
    """
    Recursively traverse a nested dictionary and apply type conversions to matching keys.
    
    Preserves dictionary structure while selectively converting values based on key names.
    Non-matching keys can optionally be converted to a default type for safety (e.g., str).
    
    Parameters
    ----------
    nested_dict : dict
        Nested dictionary potentially containing dicts, lists, and scalars
    conversions : dict
        Mapping of {key_name: dtype} where dtype is the target type.
        Example: {'highWD': float, 'noise': float, 'encoderLayer': int}
    default_dtype : type or None, default=None
        If specified, apply this type conversion to all values whose keys are NOT
        in the conversions dict. Useful for JSON serialization safety (e.g., str).
        If None, non-matching values are left unchanged.
        
    Returns
    -------
    dict
        New nested dictionary with conversions applied, original unchanged
        
    Examples
    --------
    >>> config = {
    ...     'highWD': '1e-5',
    ...     'nested': {'noise': ['0.01', '0.05'], 'other': 'unchanged'},
    ...     'tune_params': {'highWD': ['1e-6', '1e-5'], 'name': 'test'}
    ... }
    >>> result = apply_type_conversions(config, {'highWD': float, 'noise': float})
    >>> result['highWD']
    1e-05
    >>> result['nested']['noise']
    [0.01, 0.05]
    >>> result['tune_params']['highWD']
    [1e-06, 1e-05]
    
    For JSON safety, convert unmatched keys to strings:
    >>> result = apply_type_conversions(config, {'highWD': float}, default_dtype=str)
    >>> result['nested']['other']  # Now converted to string if not already
    """
    if not isinstance(nested_dict, dict):
        return nested_dict
    
    result = {}
    for key, value in nested_dict.items():
        # If key matches a conversion target
        if key in conversions:
            target_dtype = conversions[key]
            result[key] = _deep_typer(value, target_dtype)
        # If value is a dict, recurse
        elif isinstance(value, dict):
            result[key] = apply_type_conversions(value, conversions, default_dtype=default_dtype)
        # If default_dtype specified, apply it to non-matching keys
        elif default_dtype is not None:
            result[key] = _deep_typer(value, default_dtype)
        # Otherwise preserve value as-is
        else:
            result[key] = value
    
    return result