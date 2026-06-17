"""
Batch Melting data processing workflow.

Processes MELTS simulation data for machine learning training, validation, and testing.
"""

from copy import deepcopy
import os
import gc
import shutil
from pathlib import Path
import yaml

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - save plots without displaying

# Ensure repo root and src are on path
import sys
repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

file_path = str(Path(__file__).parent)
if file_path not in sys.path:
    sys.path.insert(0, file_path)

# Import BigMetaTable
from builder.processing.BigMetaTable import BigMetaTable

# Import filter functions
from builder.processing import filters

# Import settings and utilities
from recipes.settings import internal_data_dir, external_data_dir, external_base, internal_train_dir, external_train_dir
from ngibbs.utils.file_utils import delete_files_with_keyword, move_files_with_extension, get_baseline_files, clear_new_files

# Exporter functions
from builder.processing.MLexporter import resampling_to_datasets, make_harkers, make_Tplots

# Perhaps migrate the chemistry filters to their own module? 
from builder.processing.filters import deep_filter, bundle_insanity_filter
from tests.unit_tests.test_processing.ML_export_tests import sanity_check_bundle



def process_for_ML(config_path=None, MELTSModel=None, Date=None, Mode=None, upsample=None,
                   preprocessed=None, subset=None, use_external=None, balance_function=None,
                   skip_train=False):
    """
    Process MELTS data for machine learning.
    
    Parameters
    ----------
    config_path : str or Path, optional
        Path to processing.yaml configuration file. If provided, other arguments are ignored.
        Defaults to config/processing.yaml relative to repository root.
    MELTSModel : str, optional
        MELTS model version identifier (overrides config if both provided)
    Date : str, optional
        Date identifier for dataset naming (overrides config if both provided)
    Mode : str, optional
        Calculation mode string (overrides config if both provided)
    upsample : bool, optional
        Whether to upsample rare phases (overrides config if both provided)
    preprocessed : bool, optional
        Whether data has already been preprocessed (overrides config if both provided)
    subset : bool, optional
        Whether to use subset versions of datasets (overrides config if both provided)
    use_external : bool, optional
        Whether to use external storage directory (overrides config if both provided)
    balance_function : callable, optional
        Balance function to apply (overrides config if both provided)
    """
    
    # Load configuration
    if config_path is None:
        REPO_ROOT = Path(__file__).resolve().parents[3]
        config_path = REPO_ROOT / 'config' / 'processing.yaml'
    else:
        config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract configuration with command-line overrides
    dataset_cfg = config['dataset']
    preproc_cfg = config['preprocessing']
    upsample_cfg = config['upsampling']
    resampling_cfg = config['resampling']
    feature_names_cfg = config.get('featureNames')
    free_outputs_cfg = config.get('freeOutputs')
    if 'logfO2-QFM(System_main)' in feature_names_cfg:
        OXYGEN = 'open'
        print("fO2 Feature Passed: Open System Oxygen")
    else:   
        OXYGEN = 'closed'
        print("No fO2 Feature Passed: Closed System Oxygen")
    
    balance_cfg = config['balancing']
    filter_cfg = config['deep_filter']
    min_phase_cfg = config['min_phase_proportion']
    excluded_oxides_cfg = preproc_cfg.get('excluded_oxides', [])

    if excluded_oxides_cfg is None:
        excluded_oxides_cfg = []
    if isinstance(excluded_oxides_cfg, str):
        excluded_oxides_cfg = [excluded_oxides_cfg]
    if not isinstance(excluded_oxides_cfg, list):
        raise ValueError("preprocessing.excluded_oxides must be a list of oxide names")

    plot_cfg = config.get('plot', {})
    outname = config.get('outname', '').strip()

    resampling_kwargs = {}
    if feature_names_cfg is not None:
        resampling_kwargs['featureNames'] = feature_names_cfg
    if free_outputs_cfg is not None:
        resampling_kwargs['freeOutputs'] = free_outputs_cfg
    
    MELTSModel = MELTSModel or dataset_cfg['MELTSModel']
    Date = Date or dataset_cfg['Date']
    Mode = Mode or dataset_cfg['Mode']
    subset = subset if subset is not None else dataset_cfg['subset']
    use_external = use_external if use_external is not None else dataset_cfg['use_external']

    if 'hefesto' in MELTSModel.lower():
        MODEL = 'HeFESTo'
    else:
        MODEL = 'MELTS' 
    
    preprocessed = preprocessed if preprocessed is not None else preproc_cfg['preprocessed']
    
    upsample = upsample if upsample is not None else upsample_cfg['enabled']
    plot_enabled = bool(plot_cfg.get('enabled', False))
    
    if balance_function is None:
        balance_func_name = balance_cfg['function']
        if balance_func_name == 'balance_lowF':
            balance_function = filters.balance_lowF
        elif balance_func_name == 'balance_geodynamics':
            balance_function = filters.balance_geodynamics
        # else: balance_function remains None

    
    # Directory definitions
    REPO_ROOT = Path(__file__).resolve().parents[3]
    DATA_DIR = REPO_ROOT / 'data'
    INTERNAL_DIR = Path(internal_data_dir(MELTSModel))
    EXTERNAL_DIR = Path(external_data_dir(MELTSModel))
    DESTINATION_DIR = EXTERNAL_DIR if use_external else INTERNAL_DIR
    if use_external:
        EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
        external_data_path = external_data_dir(MELTSModel)
        Path(external_data_path).mkdir(parents=True, exist_ok=True)
        external_mlready_dir = Path(external_train_dir(MELTSModel))
        external_mlready_dir.mkdir(parents=True, exist_ok=True)

    # Build source filenames
    internal_data_path = str(INTERNAL_DIR)
    if MODEL == 'MELTS': # Overwrite if HeFESTo is detected in MELTSModel, but only if not already set to HeFESTo
        ValidName = f"{internal_data_path}/MELTS{MELTSModel}_Validset{Date}{Mode}"
        TestName = f"{internal_data_path}/MELTS{MELTSModel}_Testset{Date}{Mode}"
        TrainName = f"{internal_data_path}/MELTS{MELTSModel}_Trainset{Date}{Mode}"
    else:
        
        ValidName = f"{internal_data_path}/{MELTSModel}_Validset{Date}{Mode}"
        TestName = f"{internal_data_path}/{MELTSModel}_Testset{Date}{Mode}"
        TrainName = f"{internal_data_path}/{MELTSModel}_Trainset{Date}{Mode}"

    # Create destination directories if they don't exist
    out_Dir = Path(internal_train_dir(MELTSModel))
    train_dir = Path(external_train_dir(MELTSModel)) if use_external else out_Dir
    PLOT_DIR = out_Dir / 'Plots' / f'MELTS{MELTSModel}_{Date}{Mode}'
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if subset:
        ValidName += '_subset'
        TestName += '_subset'
        TrainName += '_subset'

    if preprocessed:
        ValidName += '_processed'
        TestName += '_processed'
        TrainName += '_processed'

    # Build a shared output stem so all bundle names differ only by split suffix.
    bundle_stem = outname or Path(TrainName).name.replace('_Trainset', '')

    # Helper function to generate bundle names based on outname config
    def get_bundle_name(data_type):
        """
        Generate bundle filename.
        
        Parameters
        ----------
        data_type : str
            Type of data ('Train', 'Test', or 'Valid')
            
        Returns
        -------
        str
            Bundle filename
        """
        return f"{bundle_stem}_{data_type}.tar.gz"

    def ensure_bundle_in_train_dir(bundle_path):
        """Ensure bundle exists in final MLready directory before filtering."""
        bundle_path = Path(bundle_path)
        target_path = train_dir / bundle_path.name

        if bundle_path.exists() and bundle_path.parent != train_dir:
            train_dir.mkdir(parents=True, exist_ok=True)
            if target_path.exists():
                target_path.unlink()
            moved_path = shutil.move(str(bundle_path), str(target_path))
            return Path(moved_path)

        if target_path.exists():
            return target_path

        if bundle_path.exists():
            return bundle_path

        raise FileNotFoundError(
            f"Bundle not found at expected locations: {bundle_path} or {target_path}"
        )

    gc.collect()
    
    # Capture baseline files for cleanup on exit
    baseline_files = get_baseline_files(INTERNAL_DIR)
    
    try:
        # Process training data
        read_dir = str(external_base) if use_external else None

        saved_ml_indexer = None  # populated when skip_train=True
        if not skip_train:
            TrainMELTS = BigMetaTable(TrainName, read_dir=read_dir, Model=MODEL, OXYGEN=OXYGEN) # Assume closed system for training data
            if excluded_oxides_cfg:
                print(f"Applying configured oxide exclusions at table load: {excluded_oxides_cfg}")
                TrainMELTS.exclude_oxides(excluded_oxides_cfg)
            header = TrainMELTS.header  # Capture header for indexer construction
            pre_filter = TrainMELTS.table.shape[0]

            TrainMELTS.filter_inconsistent_phase_data()

            assert TrainMELTS.table.shape[0] > pre_filter*0.97, "More than 3% of the dataset has inconsistent phase data! (Zero mass but non-zero other properties)."

            if not preprocessed:
                if preproc_cfg.get('recover_untracked_phases', False):
                    TrainMELTS.recover_untracked_phases()
                if preproc_cfg['separate_analcime']:
                    TrainMELTS.separate_analcime()
                if preproc_cfg['separate_feldspar']:
                    TrainMELTS.separate_k_feldspar()
                if preproc_cfg['filter_full_metadata']:
                    TrainMELTS.filter_full_metadata()
                #TrainMELTS.filter_legal() Deprecated. Filtering handled in deep_filtering step and metadata filtering

                # Diagnostic: Check which phases have non-zero abundance after filtering
                if upsample:
                    print("\n################ Phase Abundance After Filtering ####################")
                    phase_list = TrainMELTS.indexer.ml_indexer.all_phases
                    for phase in phase_list:
                        if phase in TrainMELTS.indexer.MELTS_indices:
                            col_idx = TrainMELTS.indexer.MELTS_indices[phase].get('mass (gm)')
                            if col_idx is not None:
                                non_zero_count = np.sum(TrainMELTS.table[:, col_idx] > 0)
                                total_pct = 100 * non_zero_count / len(TrainMELTS.table)
                                print(f"{phase:20} {non_zero_count:>10,} samples ({total_pct:>6.2f}%)")
                    print("########################################################################\n")

                if upsample:
                    # Resample configured phases from YAML (excluding test_set_phases)
                    for phase_name, phase_config in upsample_cfg['phases'].items():
                        if phase_name == 'test_set_phases':
                            continue  # Skip test set configuration

                        if phase_name in TrainMELTS.indexer.MELTS_indices:
                            try:
                                TrainMELTS.resample_rare_phase(
                                    TrainMELTS.indexer.MELTS_indices[phase_name]['mass (gm)'],
                                    multiplier_bounds=phase_config['multiplier_bounds'],
                                    n_resamples=phase_config['n_resamples'],
                                    overwrite=True
                                )
                            except KeyError as e:
                                print(f"Warning: Phase '{phase_name}' not found in MELTS_indices: {e}")
                        else:
                            print(f"Warning: Phase '{phase_name}' not available in current MELTS model")

                if balance_function is not None:
                    balance_function(TrainMELTS)

                # Exclude exceptionally low-abundance phases, these won't be learned well and may add noise to training. Configured in YAML.
                if min_phase_cfg != 0:
                    TrainMELTS.filter_min_phase_proportion(min_proportion=min_phase_cfg) # Remove samples where any phase is below the minimum proportion threshold after upsampling

            TrainMELTS.indexer.table_update(
                TrainMELTS.table,
                excluded_oxides=excluded_oxides_cfg
            )

            TrainMELTS.filename = TrainName
            train_bundle = train_dir / get_bundle_name('Train')

            if upsample:
                train_bundle_path = resampling_to_datasets(
                    TrainMELTS,
                    resampling_cfg['train_bounds'],
                    config_path=config_path,
                    bundle_name=train_bundle,
                    **resampling_kwargs,
                )
            else:
                train_bundle_path = resampling_to_datasets(
                    TrainMELTS,
                    [[1, 1]],
                    config_path=config_path,
                    bundle_name=train_bundle,
                    **resampling_kwargs,
                )
                train_bundle_path = ensure_bundle_in_train_dir(train_bundle_path)

            TrainIndexer = deepcopy(TrainMELTS.indexer)  # Capture indexer for consistency with validation/test dataset

            del TrainMELTS.table
            del TrainMELTS
            gc.collect()

            # Clear intermediate memory maps
            delete_files_with_keyword(str(INTERNAL_DIR), keyword='working', dry_run=False)
            delete_files_with_keyword(str(INTERNAL_DIR), keyword='temp', dry_run=False)

            if filter_cfg['enabled']:
                deep_filter(
                    str(train_bundle_path),
                    Oxide_Lower_Bounds=filter_cfg['oxide_lower_bounds'] or None,
                    Oxide_Upper_Bounds=filter_cfg['oxide_upper_bounds'] or None,
                    Component_Upper_Bounds=filter_cfg['component_upper_bounds'] or None,
                    Component_Lower_Bounds=filter_cfg.get('component_lower_bounds', []),
                    Bulk_Oxide_Bounds=filter_cfg.get('bulk_oxide_bounds') or None,
                    batch_size=filter_cfg['batch_size']
                )

            bundle_insanity_filter(train_bundle_path, bulk_tol_frac=5E-3)

        else:
            # Load pre-built training artifacts; skip the full training pipeline.
            header = open(TrainName + '.csv').readline().strip().split(',')
            from ngibbs.config.ml_indexer import load_ml_indexer_from_state
            saved_ml_indexer = load_ml_indexer_from_state(TrainName + 'ml_indexer')
            train_bundle_path = train_dir / get_bundle_name('Train')
            if not Path(train_bundle_path).exists():
                raise FileNotFoundError(
                    f"--skip-train requires a pre-built training bundle at: {train_bundle_path}")
            TrainIndexer = None
            print(f"[SKIP-TRAIN] Loaded header ({len(header)} cols) and ml_indexer "
                  f"from {TrainName}ml_indexer/")


        # Process validation and test data
        ValidMELTS = BigMetaTable(ValidName, read_dir=read_dir, Model=MODEL, OXYGEN=OXYGEN)
        try:
            assert ValidMELTS.header == header, "Validation dataset header does not match training dataset header! Trying without last training column." #(This is assummed in later steps, but maybe doesn't need to be?)
        except AssertionError as e:
            assert ValidMELTS.header == header[:-1], "Validation dataset header does not match training dataset header!" #(This is assummed in later steps, but maybe doesn't need to be?)

        if skip_train:
            if excluded_oxides_cfg:
                ValidMELTS.exclude_oxides(excluded_oxides_cfg)
            ValidMELTS.indexer.ml_indexer = saved_ml_indexer
        else:
            ValidMELTS.indexer = TrainIndexer  # Assign identical indexer for consistency with training dataset
            ValidMELTS.indexer.ml_indexer = TrainIndexer.ml_indexer



        pre_filter = ValidMELTS.table.shape[0]

        ValidMELTS.filter_inconsistent_phase_data()  

        assert ValidMELTS.table.shape[0] > pre_filter*0.97, "More than 3% of the Validation dataset has inconsistent phase data! (Zero mass but non-zero other properties)."

        if not preprocessed:
            if preproc_cfg.get('recover_untracked_phases', False):
                ValidMELTS.recover_untracked_phases()
            if preproc_cfg['separate_analcime']:
                ValidMELTS.separate_analcime()
            if preproc_cfg['separate_feldspar']:
                ValidMELTS.separate_k_feldspar()
            if preproc_cfg['filter_full_metadata']:
                ValidMELTS.filter_full_metadata()
            ValidMELTS.filter_phases_not_in_ml_indexer() # Propagate prohibitively rare phase removal from training set to validation/test data
            #ValidMELTS.filter_legal() Deprecated. Filtering handled in deep_filtering step and metadata filtering

            ValidMELTS, TestMELTS = ValidMELTS.split(0.30) # Future: make configurable. 
            if skip_train:
                TestMELTS.indexer.ml_indexer  = saved_ml_indexer
                ValidMELTS.indexer.ml_indexer = saved_ml_indexer
            else:
                TestMELTS.indexer = TrainIndexer  # Assign identical indexer for consistency with training dataset
                TestMELTS.indexer.ml_indexer = TrainIndexer.ml_indexer
                ValidMELTS.indexer = TrainIndexer  # ValidMELTS is reinitialized; must transfer indexer again after split.
                ValidMELTS.indexer.ml_indexer = TrainIndexer.ml_indexer
            if upsample:
                # Resample configured test set phases from YAML
                test_phases = upsample_cfg['phases'].get('test_set_phases', {})
                for phase_name, phase_config in test_phases.items():
                    if phase_name in TestMELTS.indexer.MELTS_indices:
                        try:
                            TestMELTS.resample_rare_phase(
                                TestMELTS.indexer.MELTS_indices[phase_name]['mass (gm)'],
                                multiplier_bounds=phase_config['multiplier_bounds'],
                                n_resamples=phase_config['n_resamples'],
                                overwrite=True
                            )
                        except KeyError as e:
                            print(f"Warning: Phase '{phase_name}' not found in MELTS_indices: {e}")
                    else:
                        print(f"Warning: Phase '{phase_name}' not available in current MELTS model")

            print("################ After resampling ####################")

            if balance_function is not None:
                balance_function(TestMELTS)
                balance_function(ValidMELTS)

            #TestMELTS.save(name=f"{TestName}Filtered", save_csv=False)
            #ValidMELTS.save(name=f"{ValidName}Filtered", save_csv=False)
        else:
            TestMELTS = BigMetaTable(TestName, read_dir=read_dir, Model=MODEL, OXYGEN=OXYGEN)

        TestMELTS.filename = TestName
        ValidMELTS.filename = ValidName

        test_bundle = train_dir / get_bundle_name('Test')
        valid_bundle = train_dir / get_bundle_name('Valid')

        test_bundle_path = resampling_to_datasets(
            TestMELTS,
            resampling_cfg['test_bounds'],
            config_path=config_path,
            bundle_name=test_bundle,
            **resampling_kwargs,
        )
        valid_bundle_path = resampling_to_datasets(
            ValidMELTS,
            resampling_cfg['test_bounds'],
            config_path=config_path,
            bundle_name=valid_bundle,
            **resampling_kwargs,
        )

        test_bundle_path = ensure_bundle_in_train_dir(test_bundle_path)
        valid_bundle_path = ensure_bundle_in_train_dir(valid_bundle_path)

        # Generate plots (saved to PLOT_DIR, not displayed due to 'Agg' backend)
        if plot_enabled:
            make_harkers(ValidMELTS, str(PLOT_DIR) + '/')
            make_Tplots(ValidMELTS, str(PLOT_DIR) + '/')

        del TestMELTS.table, ValidMELTS.table
        del TestMELTS, ValidMELTS
        gc.collect()

        delete_files_with_keyword(str(INTERNAL_DIR), keyword='working', dry_run=False)
        delete_files_with_keyword(str(INTERNAL_DIR), keyword='temp', dry_run=False)

        if filter_cfg['enabled']:
            deep_filter(
                str(test_bundle_path),
                Oxide_Lower_Bounds=filter_cfg['oxide_lower_bounds'] or None,
                Oxide_Upper_Bounds=filter_cfg['oxide_upper_bounds'] or None,
                Component_Upper_Bounds=filter_cfg['component_upper_bounds'] or None,
                Component_Lower_Bounds=filter_cfg.get('component_lower_bounds', []),
                Bulk_Oxide_Bounds=filter_cfg.get('bulk_oxide_bounds') or None,
                batch_size=filter_cfg['batch_size']
            )
        if filter_cfg['enabled']:
            deep_filter(
                str(valid_bundle_path),
                Oxide_Lower_Bounds=filter_cfg['oxide_lower_bounds'] or None,
                Oxide_Upper_Bounds=filter_cfg['oxide_upper_bounds'] or None,
                Component_Upper_Bounds=filter_cfg['component_upper_bounds'] or None,
                Component_Lower_Bounds=filter_cfg.get('component_lower_bounds', []),
                Bulk_Oxide_Bounds=filter_cfg.get('bulk_oxide_bounds') or None,
                batch_size=filter_cfg['batch_size']
            )

        bundle_insanity_filter(test_bundle_path, bulk_tol_frac=5E-3)
        bundle_insanity_filter(valid_bundle_path, bulk_tol_frac=5E-3)
        #sanity_check_bundle(test_bundle)  # Verify test bundle integrity before proceeding
        #sanity_check_bundle(valid_bundle)  # Verify validation bundle integrity before proceeding

        

        # Rename processed data
        """ if not preprocessed: # Preprocessing not supported
            os.rename(ValidName + 'Filtered.npy', ValidName + '_processed.npy')
            os.rename(ValidName + 'Filtered.txt', ValidName + '_processed.txt')
            os.rename(TestName + 'Filtered.npy', TestName + '_processed.npy')
            os.rename(TestName + 'Filtered.txt', TestName + '_processed.txt')
            """
        # Move ALL files to external directory if requested
        if use_external:
            external_data_path = str(external_data_dir(MELTSModel))
            move_files_with_extension(
                extension='.npy',
                dst_dir=external_data_path,
                src_dir=str(INTERNAL_DIR),
                overwrite=True
            )
            move_files_with_extension(
                extension='.txt',
                dst_dir=external_data_path,
                src_dir=str(INTERNAL_DIR),
                overwrite=True
            )
            move_files_with_extension(
                extension='.csv',
                dst_dir=external_data_path,
                src_dir=str(INTERNAL_DIR),
                overwrite=True
            )
            move_files_with_extension(
                extension='.tar.gz',
                dst_dir=str(external_mlready_dir),
                src_dir=str(INTERNAL_DIR),
                overwrite=True
            )

        else: # Move ML-ready bundles to data/MLready
            move_files_with_extension(
                extension='.tar.gz',
                dst_dir=str(out_Dir),
                src_dir=str(INTERNAL_DIR),
                overwrite=True)
    
    finally:
        # Close any open memory maps to prevent resource leaks
        print("\n[Cleanup] Closing open memory maps...")
        
        # Helper function to close memory map attributes
        def close_memmap_attrs(obj, obj_name):
            """Close memory map attributes of a MELTS object."""
            memmap_attrs = ['table', 'features', 'labels', 'binarylabels', 'masslabels', 
                           'molarlabels', 'freeOutputs', 'molar', 'table1', 'blurredbinaries']
            closed_count = 0
            for attr in memmap_attrs:
                if hasattr(obj, attr):
                    try:
                        delattr(obj, attr)
                        closed_count += 1
                    except Exception as e:
                        pass  # Attribute already deleted or not a memmap
            if closed_count > 0:
                print(f"[Cleanup] Closed {closed_count} memory maps from {obj_name}")
            return closed_count
        
        # Close TrainMELTS if it exists
        if 'TrainMELTS' in locals():
            try:
                close_memmap_attrs(TrainMELTS, 'TrainMELTS')
                del TrainMELTS
            except Exception as e:
                print(f"[Cleanup] Error closing TrainMELTS: {e}")
        
        # Close TestMELTS if it exists
        if 'TestMELTS' in locals():
            try:
                close_memmap_attrs(TestMELTS, 'TestMELTS')
                del TestMELTS
            except Exception as e:
                print(f"[Cleanup] Error closing TestMELTS: {e}")
        
        # Close ValidMELTS if it exists
        if 'ValidMELTS' in locals():
            try:
                close_memmap_attrs(ValidMELTS, 'ValidMELTS')
                del ValidMELTS
            except Exception as e:
                print(f"[Cleanup] Error closing ValidMELTS: {e}")
        
        # Force garbage collection
        gc.collect()
        
        # Clean up any temporary files created during processing
        print("\n[Cleanup] Removing remaining new files created during processing...")
        deleted_count = clear_new_files(INTERNAL_DIR, baseline_files, protected_extensions=['.tar.gz'])
        print(f"[Cleanup] Removed {deleted_count} temporary files from {INTERNAL_DIR}")

        return train_bundle_path, valid_bundle_path, test_bundle_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Process MELTS simulation data for machine learning training.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Use default configuration from config/processing.yaml
  python prepareML.py
  
  # Use custom configuration file
  python prepareML.py --config /path/to/custom_config.yaml
  
  # Override specific parameters
  python prepareML.py --MELTSModel 110 --Date Jan15 --no-upsample
  
  # Use custom config with overrides
  python prepareML.py --config custom.yaml --MELTSModel 102 --use-external
        ''')
    
    parser.add_argument('--config', type=str, default=None,
                        help='Path to processing.yaml configuration file (default: config/processing.yaml)')
    parser.add_argument('--MELTSModel', type=str, default=None,
                        help='MELTS model version (102, 110, 120, p)')
    parser.add_argument('--Date', type=str, default=None,
                        help='Date identifier for dataset naming')
    parser.add_argument('--Mode', type=str, default=None,
                        help='Calculation mode (e.g., FxCrystCooling, BatchCooling)')
    parser.add_argument('--upsample', dest='upsample', action='store_true', default=None,
                        help='Enable rare phase upsampling')
    parser.add_argument('--no-upsample', dest='upsample', action='store_false',
                        help='Disable rare phase upsampling')
    parser.add_argument('--preprocessed', action='store_true',
                        help='Skip preprocessing (data already preprocessed)')
    parser.add_argument('--subset', action='store_true',
                        help='Use subset versions of datasets')
    parser.add_argument('--use-external', dest='use_external', action='store_true',
                        help='Use external storage directory')
    parser.add_argument('--balance-function', type=str, default=None,
                        choices=['none', 'balance_lowF', 'balance_geodynamics', 'balance_superliquidus'],
                        help='Balance function to apply')
    parser.add_argument('--skip-train', dest='skip_train', action='store_true', default=False,
                        help='Skip training-set processing; load header and ml_indexer from the '
                             'pre-built training bundle instead (requires an existing bundle)')

    args = parser.parse_args()
    
    # Convert balance function string to actual function
    balance_func = None
    if args.balance_function == 'balance_lowF':
        balance_func = filters.balance_lowF
    elif args.balance_function == 'balance_geodynamics':
        balance_func = filters.balance_geodynamics
    elif args.balance_function == 'balance_superliquidus':
        balance_func = filters.balance_Superliquidus_fxtal
    
    """print('Waiting')
    import time
    time.sleep(4500)"""
        # Run processing
    process_for_ML(
        config_path=args.config,
        MELTSModel=args.MELTSModel,
        Date=args.Date,
        Mode=args.Mode,
        upsample=args.upsample if args.upsample is not None else None,
        preprocessed=args.preprocessed if args.preprocessed else None,
        subset=args.subset if args.subset else None,
        use_external=args.use_external if args.use_external else None,
        balance_function=balance_func,
        skip_train=args.skip_train,
    )