import sys
import os
from pathlib import Path

# Add parent directory (repository root) to path
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add src to path so we can import modules without src prefix
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from builder.processing.prepareML import process_for_ML

# Option 1: Use default configuration from config/processing.yaml
process_for_ML()

# Option 2: Use custom config file
# process_for_ML(config_path='path/to/custom_config.yaml')

# Option 3: Use default config but override specific parameters
# process_for_ML(MELTSModel='110', upsample=False)

# Option 4: Specify all parameters programmatically (ignores config file)
# from builder.processing import filters
# process_for_ML(
#     MELTSModel='102',
#     Date='Jan14',
#     Mode='BatchCooling',
#     preprocessed=False,
#     subset=False,
#     use_external=False,
#     upsample=True,
#     balance_function=filters.balance_geodynamics
# )
