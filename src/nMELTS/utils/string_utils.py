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
