"""
Configuration settings for nMELTS builder tools.

This file is used by builder and processing scripts, NOT by the distributable nMELTS package.
Data file paths and external storage configuration for development/training workflows.
"""
from pathlib import Path
import yaml


def _load_user_config():
    config_path = Path(__file__).parent / 'config.yaml'
    if not config_path.exists():
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


_CFG = _load_user_config()

# SET WHERE YOUR DATA IS KEPT HERE. IF IN YOUR WORKING DIRECTORY, MAKE IT "", an empty string
external_base = _CFG.get('external_base', "D:/") or ""

if external_base and external_base[-1] != '/':  # Assumed to end in slash in this code
    external_base += '/'


_internal_data_dir = _CFG.get('internal_data_dir')
if _internal_data_dir:
    _internal_data_dir = Path(_internal_data_dir)
    if not _internal_data_dir.is_absolute():
        _internal_data_dir = Path(__file__).resolve().parent.parent / _internal_data_dir
    INT_DATA_DIR = _internal_data_dir
else:
    INT_DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

DATA_PATH = _CFG.get('data_path', 'MELTStable/') or ''
if DATA_PATH and not DATA_PATH.endswith('/'):
    DATA_PATH += '/'  # Path from internal data folder, or from external_base

TRAIN_PATH = _CFG.get('train_path', 'MLready/') or ''
if TRAIN_PATH and not TRAIN_PATH.endswith('/'):
    TRAIN_PATH += '/'

SCRATCH_DATA_PATH = _CFG.get('scratch_data_path', 'Workspace/') or ''
if SCRATCH_DATA_PATH and not SCRATCH_DATA_PATH.endswith('/'):
    SCRATCH_DATA_PATH += '/'




def internal_data_dir(MELTSModel): 
    """
    Get internal directory path for a MELTS model.
    
    Args:
        MELTSModel (str): MELTS model identifier
        
    Returns:
        str: Internal directory path
    """
    return INT_DATA_DIR / (DATA_PATH + f'{MELTSModel}')


def external_data_dir(MELTSModel):
    """
    Get external directory path for a MELTS model.
    
    Args:
        MELTSModel (str): MELTS model identifier
        
    Returns:
        str: External directory path
    """
    return external_base + (DATA_PATH + f'{MELTSModel}')


def internal_train_dir(MELTSModel):
    """
    Get internal directory path for ML-ready training bundles.
    """
    return INT_DATA_DIR / (TRAIN_PATH + f'{MELTSModel}')


def external_train_dir(MELTSModel):
    """
    Get external directory path for ML-ready training bundles.
    """
    return external_base + (TRAIN_PATH + f'{MELTSModel}')


def internal_scratch_dir():
    """
    Get internal directory path for temporary MELTS workspace simulations.
    
    Returns:
        Path: Internal scratch directory path
    """
    return INT_DATA_DIR / SCRATCH_DATA_PATH


def external_scratch_dir():
    """
    Get external directory path for temporary MELTS workspace simulations.
    
    Returns:
        str: External scratch directory path
    """
    return external_base + SCRATCH_DATA_PATH
