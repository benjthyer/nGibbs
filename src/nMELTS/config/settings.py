"""
Configuration settings.

For now, just the paths.?
TO DO: NEW CODE THAT CREATES THE EXTERNAL DIRECTORY STRUCTURE IF IT DON'T EXIST. 

"""
from pathlib import Path


# SET WHERE YOUR DATA IS KEPT HERE. IF IN YOUR WORKING DIRECTORY, MAKE IT "", an empty string
external_base = "D:/"


if external_base[-1] != '/' and len(external_base):  # Assumed to end in slash in this code
    external_base += '/'


_DATA_DIR = Path(__file__).resolve().parent.parent + '/'
DATA_PATH = 'Workspace/' # Path from internal data folder, or from external_base




def internal_data_dir(MELTSModel): 
    """
    Get internal directory path for a MELTS model.
    
    Args:
        MELTSModel (str): MELTS model identifier
        
    Returns:
        str: Internal directory path
    """
    return _DATA_DIR + DATA_PATH + f'{MELTSModel}Datasets/'


def external_data__dir(MELTSModel):
    """
    Get external directory path for a MELTS model.
    
    Args:
        MELTSModel (str): MELTS model identifier
        
    Returns:
        str: External directory path
    """
    return external_base + DATA_PATH + f'{MELTSModel}Datasets/'
