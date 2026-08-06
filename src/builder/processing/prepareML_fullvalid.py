"""
Batch Melting data processing workflow (full-validation-as-test variant).

Processes MELTS simulation data for machine learning training and testing.
Unlike prepareML.py, this variant does not split the validation dataset into
separate validation/test subsets - the entire validation dataset is used as
the test dataset, and no separate Valid bundle is produced.
"""

from copy import deepcopy
import os
import gc
import shutil
import tempfile
from pathlib import Path
import yaml

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
from builder.processing.MLexporter import resampling_to_datasets, make_harkers, make_Tplots, shuffle_bundle_rows

# Perhaps migrate the chemistry filters to their own module?
# deep_filter/bundle_insanity_filter run inside resampling_to_datasets() via
# deep_filter_kwargs/insanity_filter_kwargs below, so they aren't called directly here.
from builder.processing.filters import filter_bulk_composition_mismatch
from tests.unit_tests.test_processing.ML_export_tests import sanity_check_bundle



def process_for_ML(config_path=None, MELTSModel=None, Date=None, Mode=None, outname=None, upsample=None,
                   preprocessed=None, subset=None, use_external=None, balance_function=None,
                   skip_train=False):
    """
    Process MELTS data for machine learning.

    The entire validation dataset is used as the test dataset (no 70/30 split,
    no separate Valid bundle). Otherwise identical to prepareML.process_for_ML.

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
    outname : str, optional
        Output bundle base name (overrides config's top-level `outname` if both provided)
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
    free_outputs_cfg = config.get('free_outputs', config.get('freeOutputs'))
    if 'logfO2-QFM(System_main)' in feature_names_cfg:
        OXYGEN = 'open'
        print("fO2 Feature Passed: Open System Oxygen")
    else:
        OXYGEN = 'closed'
        print("No fO2 Feature Passed: Closed System Oxygen")

    balance_cfg = config['balancing']
    filter_cfg = config['deep_filter']
    min_phase_cfg = config['min_phase_proportion']

    # Forwarded into resampling_to_datasets() so deep_filter/bundle_insanity_filter
    # run on the unpacked .npy files before packaging, instead of extracting and
    # repacking the finished .tar.gz bundle.
    deep_filter_kwargs = None
    if filter_cfg['enabled']:
        deep_filter_kwargs = dict(
            Oxide_Lower_Bounds=filter_cfg['oxide_lower_bounds'] or None,
            Oxide_Upper_Bounds=filter_cfg['oxide_upper_bounds'] or None,
            Component_Upper_Bounds=filter_cfg['component_upper_bounds'] or None,
            Component_Lower_Bounds=filter_cfg.get('component_lower_bounds', []),
            Bulk_Oxide_Bounds=filter_cfg.get('bulk_oxide_bounds') or None,
            batch_size=filter_cfg['batch_size'],
        )
    insanity_filter_kwargs = dict(bulk_tol_frac=5E-3)
    excluded_oxides_cfg = preproc_cfg.get('excluded_oxides', [])
    performance_cfg = config.get('performance', {})
    chunk_size = performance_cfg.get('chunk_size', 100_000)

    if excluded_oxides_cfg is None:
        excluded_oxides_cfg = []
    if isinstance(excluded_oxides_cfg, str):
        excluded_oxides_cfg = [excluded_oxides_cfg]
    if not isinstance(excluded_oxides_cfg, list):
        raise ValueError("preprocessing.excluded_oxides must be a list of oxide names")

    plot_cfg = config.get('plot', {})
    outname = (outname or config.get('outname', '')).strip()

    resampling_kwargs = {'chunk_size': chunk_size}
    if feature_names_cfg is not None:
        resampling_kwargs['featureNames'] = feature_names_cfg
    if free_outputs_cfg is not None:
        resampling_kwargs['free_outputs'] = free_outputs_cfg

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
    else:
        # balance_function was passed in directly (e.g. from a CLI override) rather
        # than resolved from balance_cfg above - recover a name for it so the
        # archived config below still records what actually ran.
        balance_func_name = getattr(balance_function, '__name__', balance_cfg.get('function'))

    # The bundle archiving below (resampling_to_datasets' config_path argument)
    # copies the file at config_path verbatim into the .tar.gz - it has no idea
    # about MELTSModel/Date/Mode/outname/subset/use_external/preprocessed/upsample/
    # balance_function overrides applied above. Write out the config actually in
    # effect and archive that instead, so the bundled yaml is a faithful record of
    # what ran rather than a copy of the un-overridden file on disk.
    effective_config = deepcopy(config)
    effective_config.setdefault('dataset', {})
    effective_config['dataset']['MELTSModel'] = MELTSModel
    effective_config['dataset']['Date'] = Date
    effective_config['dataset']['Mode'] = Mode
    effective_config['dataset']['subset'] = subset
    effective_config['dataset']['use_external'] = use_external
    effective_config.setdefault('preprocessing', {})['preprocessed'] = preprocessed
    effective_config.setdefault('upsampling', {})['enabled'] = upsample
    effective_config.setdefault('balancing', {})['function'] = balance_func_name
    effective_config['outname'] = outname

    archive_config_dir = Path(tempfile.mkdtemp(prefix='ngibbs_processing_config_'))
    archive_config_path = archive_config_dir / Path(config_path).name
    with open(archive_config_path, 'w') as f:
        yaml.safe_dump(effective_config, f, sort_keys=False)

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
            TrainMELTS = BigMetaTable(TrainName, read_dir=read_dir, Model=MODEL, OXYGEN=OXYGEN, chunk_size=chunk_size) # Assume closed system for training data
            if excluded_oxides_cfg:
                print(f"Applying configured oxide exclusions at table load: {excluded_oxides_cfg}")
                TrainMELTS.exclude_oxides(excluded_oxides_cfg)
            header = TrainMELTS.header  # Capture header for indexer construction
            pre_filter = TrainMELTS.table.shape[0]

            # Raw phase abundances before any row filtering, so surprising abundances
            # can be traced back to the source data rather than to a filtering step.
            TrainMELTS.report_phase_abundances(label="Raw, Before Any Filtering")

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
                    TrainMELTS.report_phase_abundances(label="After Filtering")

                # Catch rows where MELTS's own reported Bulk_comp disagrees with the
                # bulk composition summed from its own per-phase output (seen as a
                # low-temperature MELTS solver artifact). Must run before
                # resampling_to_datasets(), which perturbs the mass columns this compares.
                print("Filtering rows with bulk composition mismatch...")
                pre_filter = TrainMELTS.table.shape[0]
                filter_bulk_composition_mismatch(TrainMELTS)
                assert TrainMELTS.table.shape[0] > pre_filter*0.9, "More than 10 percent of the training dataset has a bulk composition mismatch!"

                if upsample:
                    # Resample configured phases from YAML (excluding test_set_phases)
                    for phase_name, phase_config in upsample_cfg['phases'].items():
                        if phase_name == 'test_set_phases':
                            continue  # Skip test set configuration
                        print(f"Resampling phase '{phase_name}' with config: {phase_config}")
                        if phase_name in TrainMELTS.indexer.MELTS_indices:
                            try:
                                TrainMELTS.resample_rare_phase(
                                    TrainMELTS.indexer.MELTS_indices[phase_name]['mass (gm)'], # NOT GENERALIZABLE TO HEFESTO
                                    multiplier_bounds=phase_config['multiplier_bounds'],
                                    n_resamples=phase_config['n_resamples'],
                                    overwrite=True
                                )
                            except KeyError as e:
                                print(f"Warning: Phase '{phase_name}' not found in MELTS_indices: {e}")
                        else:
                            print(f"Warning: Phase '{phase_name}' not available in current MELTS model")

                if balance_function is not None:
                    print(f"Applying balance function: {balance_function.__name__}")
                    balance_function(TrainMELTS)

                # Exclude exceptionally low-abundance phases, these won't be learned well and may add noise to training. Configured in YAML.
                # Run iteratively: deleting rows for rare phases can push co-occurring borderline phases below the threshold.
                if min_phase_cfg != 0:
                    print(f"Filtering phases with proportion below {min_phase_cfg}")
                    TrainMELTS.filter_min_phase_proportion(min_proportion=min_phase_cfg)
                    """
                    while True:
                        before = TrainMELTS.table.shape[0]
                        TrainMELTS.filter_min_phase_proportion(min_proportion=min_phase_cfg)
                        if TrainMELTS.table.shape[0] == before:
                            break"""

            print(f"Table update now after filtering and resampling.")
            TrainMELTS.indexer.table_update(
                TrainMELTS.table,
                excluded_oxides=excluded_oxides_cfg,
                chunk_size=chunk_size
            )

            TrainMELTS.filename = TrainName
            train_bundle = train_dir / get_bundle_name('Train')

            print("Running resampling to datasets for training data...")
            if upsample:
                train_bundle_path = resampling_to_datasets(
                    TrainMELTS,
                    resampling_cfg['train_bounds'],
                    config_path=archive_config_path,
                    bundle_name=train_bundle,
                    deep_filter_kwargs=deep_filter_kwargs,
                    insanity_filter_kwargs=insanity_filter_kwargs,
                    **resampling_kwargs,
                )
            else:
                train_bundle_path = resampling_to_datasets(
                    TrainMELTS,
                    [[1, 1]],
                    config_path=archive_config_path,
                    bundle_name=train_bundle,
                    deep_filter_kwargs=deep_filter_kwargs,
                    insanity_filter_kwargs=insanity_filter_kwargs,
                    **resampling_kwargs,
                )
                train_bundle_path = ensure_bundle_in_train_dir(train_bundle_path)

            # One-time out-of-core shuffle, run here (post-packaging, on the
            # already deep_filter/insanity_filter/resampling-trimmed bundle)
            # rather than on the raw pre-filter table, so later out-of-core/
            # chunked training readers never have to worry about row-order
            # structure (e.g. upsampled rows resample_rare_phase appends as a
            # block at the end of the table) - see MLexporter.shuffle_bundle_rows.
            print("Shuffling training row order (one-time, out-of-core, post-packaging)...")
            shuffle_bundle_rows(train_bundle_path, chunk_size=chunk_size)

            TrainIndexer = deepcopy(TrainMELTS.indexer)  # Capture indexer for consistency with validation/test dataset

            del TrainMELTS.table
            del TrainMELTS
            gc.collect()

            # Clear intermediate memory maps
            delete_files_with_keyword(str(INTERNAL_DIR), keyword='working', dry_run=False)
            delete_files_with_keyword(str(INTERNAL_DIR), keyword='temp', dry_run=False)

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


        # Process test data - the entire validation dataset is used as the test
        # dataset (no split, no separate Valid bundle).
        TestMELTS = BigMetaTable(ValidName, read_dir=read_dir, Model=MODEL, OXYGEN=OXYGEN, chunk_size=chunk_size)
        try:
            assert TestMELTS.header == header, "Validation dataset header does not match training dataset header! Trying without last training column." #(This is assummed in later steps, but maybe doesn't need to be?)
        except AssertionError as e:
            assert TestMELTS.header == header[:-1], "Validation dataset header does not match training dataset header!" #(This is assummed in later steps, but maybe doesn't need to be?)

        if skip_train:
            if excluded_oxides_cfg:
                TestMELTS.exclude_oxides(excluded_oxides_cfg)
            TestMELTS.indexer.ml_indexer = saved_ml_indexer
            TestMELTS.indexer.expose_ml_indexer_attributes()  # Sync label_indices/label_names/ncomps etc. to the loaded (train) ml_indexer, else they stay from the ephemeral indexer built from this validation set's own header
        else:
            TestMELTS.indexer = TrainIndexer  # Assign identical indexer for consistency with training dataset
            TestMELTS.indexer.ml_indexer = TrainIndexer.ml_indexer



        pre_filter = TestMELTS.table.shape[0]

        TestMELTS.filter_inconsistent_phase_data()

        assert TestMELTS.table.shape[0] > pre_filter*0.9, "More than 10% of the Validation dataset has inconsistent phase data! (Zero mass but non-zero other properties)."

        # Catch rows where MELTS's own reported Bulk_comp disagrees with the bulk
        # composition summed from its own per-phase output. Must run before
        # resampling_to_datasets(), which perturbs the mass columns this compares.
        pre_filter = TestMELTS.table.shape[0]
        filter_bulk_composition_mismatch(TestMELTS)
        assert TestMELTS.table.shape[0] > pre_filter*0.9, "More than 10% of the test dataset has a bulk composition mismatch!"
        
        if not preprocessed:
            if preproc_cfg.get('recover_untracked_phases', False):
                TestMELTS.recover_untracked_phases()
            if preproc_cfg['separate_analcime']:
                TestMELTS.separate_analcime()
            if preproc_cfg['separate_feldspar']:
                TestMELTS.separate_k_feldspar()
            if preproc_cfg['filter_full_metadata']:
                TestMELTS.filter_full_metadata()
            TestMELTS.filter_phases_not_in_ml_indexer() # Propagate prohibitively rare phase removal from training set to test data
            #TestMELTS.filter_legal() Deprecated. Filtering handled in deep_filtering step and metadata filtering

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

            #TestMELTS.save(name=f"{TestName}Filtered", save_csv=False)

        TestMELTS.filename = TestName

        test_bundle = train_dir / get_bundle_name('Test')

        test_bundle_path = resampling_to_datasets(
            TestMELTS,
            resampling_cfg['test_bounds'],
            config_path=archive_config_path,
            bundle_name=test_bundle,
            deep_filter_kwargs=deep_filter_kwargs,
            insanity_filter_kwargs=insanity_filter_kwargs,
            **resampling_kwargs,
        )

        test_bundle_path = ensure_bundle_in_train_dir(test_bundle_path)

        # Generate plots of the test data (saved to PLOT_DIR, not displayed due to 'Agg' backend)
        if plot_enabled:
            make_harkers(TestMELTS, str(PLOT_DIR) + '/')
            make_Tplots(TestMELTS, str(PLOT_DIR) + '/')

        del TestMELTS.table
        del TestMELTS
        gc.collect()

        delete_files_with_keyword(str(INTERNAL_DIR), keyword='working', dry_run=False)
        delete_files_with_keyword(str(INTERNAL_DIR), keyword='temp', dry_run=False)
        #sanity_check_bundle(test_bundle)  # Verify test bundle integrity before proceeding

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
                           'molarlabels', 'free_outputs', 'molar', 'table1', 'blurredbinaries']
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

        # Force garbage collection
        gc.collect()

        # Clean up any temporary files created during processing
        print("\n[Cleanup] Removing remaining new files created during processing...")
        deleted_count = clear_new_files(INTERNAL_DIR, baseline_files, protected_extensions=['.tar.gz'])
        print(f"[Cleanup] Removed {deleted_count} temporary files from {INTERNAL_DIR}")

        if 'archive_config_dir' in locals():
            shutil.rmtree(archive_config_dir, ignore_errors=True)

        return train_bundle_path, test_bundle_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Process MELTS simulation data for machine learning training '
                     '(full-validation-as-test variant: no valid/test split, no Valid bundle).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Use default configuration from config/processing.yaml
  python prepareML_fullvalid.py

  # Use custom configuration file
  python prepareML_fullvalid.py --config /path/to/custom_config.yaml

  # Override specific parameters
  python prepareML_fullvalid.py --MELTSModel 110 --Date Jan15 --no-upsample

  # Use custom config with overrides
  python prepareML_fullvalid.py --config custom.yaml --MELTSModel 102 --use-external
        ''')

    parser.add_argument('--config', type=str, default=None,
                        help='Path to processing.yaml configuration file (default: config/processing.yaml)')
    parser.add_argument('--MELTSModel', type=str, default=None,
                        help='MELTS model version (102, 110, 120, p)')
    parser.add_argument('--Date', type=str, default=None,
                        help='Date identifier for dataset naming')
    parser.add_argument('--Mode', type=str, default=None,
                        help='Calculation mode (e.g., FxCrystCooling, BatchCooling)')
    parser.add_argument('--outname', type=str, default=None,
                        help='Output bundle base name (overrides config: outname)')
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

    # Run processing
    process_for_ML(
        config_path=args.config,
        MELTSModel=args.MELTSModel,
        Date=args.Date,
        Mode=args.Mode,
        outname=args.outname,
        upsample=args.upsample if args.upsample is not None else None,
        preprocessed=args.preprocessed if args.preprocessed else None,
        subset=args.subset if args.subset else None,
        use_external=args.use_external if args.use_external else None,
        balance_function=balance_func,
        skip_train=args.skip_train,
    )
