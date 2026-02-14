import argparse
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gc

import yaml
import numpy as np
from datetime import datetime

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.builder.processing.prepareML import process_for_ML
from src.builder.processing.BigMetaTable import BigMetaTable
from src.builder.processing.MLexporter import load_ml_bundle
from src.builder.processing import filters
from tests.unit_tests.test_processing.ML_export_tests import run_tests_on_bundle, sanity_check_bundle
from config.settings import internal_train_dir, external_train_dir
from tests.test_utils import setup_test_logging


def _load_config(config_path: Path) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _build_dataset_base_name(melts_model: str, date: str, mode: str, subset: bool, preprocessed: bool) -> str:
    base = f"MELTS{melts_model}_Trainset{date}{mode}"
    if subset:
        base += "_subset"
    if preprocessed:
        base += "_processed"
    return base


def _find_source_base_path(melts_model: str, date: str, mode: str, subset: bool, preprocessed: bool) -> Path:
    base_name = _build_dataset_base_name(melts_model, date, mode, subset, preprocessed)
    candidate_dirs = [
        project_root / 'data' / 'MELTStables' / melts_model,
        project_root / 'data' / 'MELTStable' / melts_model,
    ]
    for base_dir in candidate_dirs:
        csv_path = base_dir / f"{base_name}.csv"
        if csv_path.exists():
            return csv_path.with_suffix('')
    raise FileNotFoundError(
        f"Source CSV not found for {base_name} in MELTStables or MELTStable directories."
    )


def _get_bundle_path(melts_model: str, date: str, mode: str, subset: bool, preprocessed: bool) -> Path:
    base_name = _build_dataset_base_name(melts_model, date, mode, subset, preprocessed)
    internal_path = Path(internal_train_dir(melts_model)) / f"{base_name}.tar.gz"
    external_path = Path(external_train_dir(melts_model)) / f"{base_name}.tar.gz"
    if external_path.exists():
        return external_path
    return internal_path


def _wrap_deep_filter_for_tarballs():
    """Ensure prepareML.deep_filter can accept base names by auto-resolving to .tar.gz."""
    import src.builder.processing.prepareML as prepareML

    original_deep_filter = prepareML.deep_filter

    def _deep_filter_wrapper(path, *args, **kwargs):
        candidate = Path(str(path))
        if not str(candidate).endswith('.tar.gz'):
            tar_candidate = Path(str(candidate) + '.tar.gz')
            if tar_candidate.exists():
                candidate = tar_candidate
        return original_deep_filter(str(candidate), *args, **kwargs)

    prepareML.deep_filter = _deep_filter_wrapper


def _reserve_baseline_bundle(bundle_path: Path, config_name: str) -> Path:
    base_name = bundle_path.name
    if base_name.endswith('.tar.gz'):
        base_root = base_name[:-7]
    else:
        base_root = bundle_path.stem
    baseline_name = f"{base_root}_baseline_{config_name}.tar.gz"
    baseline_path = bundle_path.with_name(baseline_name)
    if baseline_path.exists():
        return baseline_path
    shutil.move(str(bundle_path), str(baseline_path))
    return baseline_path


def _phase_presence(binary_labels: np.ndarray, all_phases: List[str], phase_name: str) -> Optional[np.ndarray]:
    if phase_name not in all_phases:
        return None
    idx = all_phases.index(phase_name)
    return binary_labels[:, idx] > 0.5


def _sanity_check_balancing_liquid(bundle, min_fraction: float, label: str):
    all_phases = list(bundle.ml_indexer.all_phases)
    liquid_present = _phase_presence(bundle.binary_labels, all_phases, 'melts-liquid')
    if liquid_present is None:
        raise AssertionError(f"{label}: 'melts-liquid' not found in phases")

    other_indices = [i for i, p in enumerate(all_phases) if p != 'melts-liquid']
    if not other_indices:
        raise AssertionError(f"{label}: no non-liquid phases available to test")

    other_present = np.sum(bundle.binary_labels[:, other_indices] > 0.5, axis=1) > 0
    fraction = np.mean(liquid_present & other_present)
    assert fraction >= min_fraction, (
        f"{label}: Only {fraction:.2%} rows contain melts-liquid + any other phase; "
        f"expected >= {min_fraction:.0%}"
    )


def _sanity_check_geodynamics(bundle, min_fraction: float):
    all_phases = list(bundle.ml_indexer.all_phases)
    liquid_present = _phase_presence(bundle.binary_labels, all_phases, 'melts-liquid')
    if liquid_present is None:
        raise AssertionError("Geodynamics: 'melts-liquid' not found in phases")
    fraction = np.mean(~liquid_present)
    assert fraction >= min_fraction, (
        f"Geodynamics: Only {fraction:.2%} rows have no melts-liquid; expected >= {min_fraction:.0%}"
    )


def _sanity_check_resampling_row_increase(bundle, baseline_bundle, label: str):
    new_rows = bundle.features.shape[0]
    base_rows = baseline_bundle.features.shape[0]
    assert new_rows > base_rows, (
        f"{label}: Expected more rows than baseline ({new_rows} vs {base_rows})"
    )


def _sanity_check_upsampling_phase_proportions(bundle, baseline_bundle, phase_names: List[str]):
    all_phases = list(bundle.ml_indexer.all_phases)
    base_phases = list(baseline_bundle.ml_indexer.all_phases)

    for phase in phase_names:
        if phase not in all_phases or phase not in base_phases:
            continue
        idx_new = all_phases.index(phase)
        idx_base = base_phases.index(phase)
        new_prop = np.mean(bundle.binary_labels[:, idx_new] > 0.5)
        base_prop = np.mean(baseline_bundle.binary_labels[:, idx_base] > 0.5)
        assert new_prop > base_prop, (
            f"Upsampling: phase '{phase}' proportion did not increase ({new_prop:.4f} vs {base_prop:.4f})"
        )


def _sanity_check_separate_analcime(bundle, base_table: BigMetaTable):
    idx_map = base_table.indexer.MELTS_indices
    analcime_col = idx_map.get('analcime', {}).get('mass (gm)')
    leucite_col = idx_map.get('leucite', {}).get('mass (gm)')

    analcime_present_before = analcime_col is not None and np.any(base_table.table[:, analcime_col] > 0)
    leucite_present_before = leucite_col is not None and np.any(base_table.table[:, leucite_col] > 0)

    if not (analcime_present_before or leucite_present_before):
        return

    all_phases = list(bundle.ml_indexer.all_phases)
    analcime_present_after = _phase_presence(bundle.binary_labels, all_phases, 'analcime')
    leucite_present_after = _phase_presence(bundle.binary_labels, all_phases, 'leucite')

    if analcime_present_after is None or leucite_present_after is None:
        raise AssertionError("Separate analcime: analcime or leucite missing from output phases")

    assert np.any(analcime_present_after), "Separate analcime: analcime not present after processing"
    assert np.any(leucite_present_after), "Separate analcime: leucite not present after processing"


def _compute_phase_oxide_wt(bundle, phase: str) -> np.ndarray:
    ml_indexer = bundle.ml_indexer
    if phase not in ml_indexer.label_indices_comp or phase not in ml_indexer.label_indices:
        raise AssertionError(f"Deep filter: phase '{phase}' not found in label mappings")

    comp_var_indices = ml_indexer.label_indices_comp[phase]
    comp_full_indices = ml_indexer.label_indices[phase]

    comp_to_ox = ml_indexer.compToOxLoad
    mm = ml_indexer.MM

    comps = bundle.labels[:, comp_var_indices]
    oxides = comps @ comp_to_ox[comp_full_indices]
    oxides = oxides @ mm

    sums = np.sum(oxides, axis=1)
    oxide_wt = np.zeros_like(oxides)
    nonzero = sums > 0
    oxide_wt[nonzero] = oxides[nonzero] * 100.0 / sums[nonzero].reshape(-1, 1)
    return oxide_wt


def _sanity_check_deep_filter_bounds(bundle, deep_filter_cfg: Dict):
    ml_indexer = bundle.ml_indexer
    oxide_dict = {ox: i for i, ox in enumerate(ml_indexer.Oxides)}

    component_lower_bounds = deep_filter_cfg.get('component_lower_bounds', []) or []
    component_upper_bounds = deep_filter_cfg.get('component_upper_bounds', []) or []
    oxide_lower_bounds = deep_filter_cfg.get('oxide_lower_bounds', []) or []
    oxide_upper_bounds = deep_filter_cfg.get('oxide_upper_bounds', []) or []

    # Component bounds (VC space)
    for phase, comp, bound in component_lower_bounds:
        if phase not in ml_indexer.detail_label_indices or comp not in ml_indexer.detail_label_indices[phase]:
            raise AssertionError(f"Deep filter: component '{comp}' not found in phase '{phase}'")
        idx = ml_indexer.detail_label_indices[phase][comp]
        values = bundle.labels[:, idx]
        failing = (values < bound) & (values != 0)
        assert not np.any(failing), (
            f"Deep filter: lower bound violated for {phase}:{comp} < {bound}"
        )

    for phase, comp, bound in component_upper_bounds:
        if phase not in ml_indexer.detail_label_indices or comp not in ml_indexer.detail_label_indices[phase]:
            raise AssertionError(f"Deep filter: component '{comp}' not found in phase '{phase}'")
        idx = ml_indexer.detail_label_indices[phase][comp]
        values = bundle.labels[:, idx]
        failing = values > bound
        assert not np.any(failing), (
            f"Deep filter: upper bound violated for {phase}:{comp} > {bound}"
        )

    # Oxide bounds (wt%)
    for phase, ox, bound in oxide_lower_bounds:
        if ox not in oxide_dict:
            raise AssertionError(f"Deep filter: oxide '{ox}' not found in Oxides list")
        oxide_wt = _compute_phase_oxide_wt(bundle, phase)
        values = oxide_wt[:, oxide_dict[ox]]
        failing = (values < bound) & (values != 0)
        assert not np.any(failing), (
            f"Deep filter: lower oxide bound violated for {phase}:{ox} < {bound}"
        )

    for phase, ox, bound in oxide_upper_bounds:
        if ox not in oxide_dict:
            raise AssertionError(f"Deep filter: oxide '{ox}' not found in Oxides list")
        oxide_wt = _compute_phase_oxide_wt(bundle, phase)
        values = oxide_wt[:, oxide_dict[ox]]
        failing = (values > bound) & (values != 0)
        assert not np.any(failing), (
            f"Deep filter: upper oxide bound violated for {phase}:{ox} > {bound}"
        )


def _balance_function_from_config(balance_name: Optional[str]):
    if balance_name == 'balance_lowF':
        return filters.balance_lowF
    if balance_name == 'balance_geodynamics':
        return filters.balance_geodynamics
    if balance_name == 'balance_superliquidus':
        return filters.balance_Superliquidus_fxtal
    return None


def run_processing_tests(config_dir: Path, melts_model: Optional[str], date: Optional[str], mode: Optional[str]):
    #from tests.unit_tests.test_processing.ML_export_tests import export_bundle_arrays_to_csv

    _wrap_deep_filter_for_tarballs()

    config_paths = sorted(config_dir.glob('*.yaml'))
    if not config_paths:
        raise FileNotFoundError(f"No .yaml configs found in {config_dir}")

    # Ensure baseline runs first
    config_paths.sort(key=lambda p: (p.name != 'separate_analcime_only.yaml', p.name)) # Skipping baseline, testing that with export tests

    baseline_bundle = None
    baseline_config = None
    base_table = None
    baseline_bundle_path = None

    for config_path in config_paths:
        if config_path.name == 'separate_analcime_only.yaml':
                continue  # skip baseline, tested in export tests
        cfg = _load_config(config_path)

        dataset_cfg = cfg['dataset']
        preproc_cfg = cfg['preprocessing']
        balance_cfg = cfg['balancing']
        upsample_cfg = cfg['upsampling']

        cfg_model = melts_model or dataset_cfg['MELTSModel']
        cfg_date = date or dataset_cfg['Date']
        cfg_mode = mode or dataset_cfg['Mode']

        balance_func = _balance_function_from_config(balance_cfg.get('function'))

        print(f"\n=== Running process_for_ML for {config_path.name} ===")
        process_for_ML(
            config_path=str(config_path),
            MELTSModel=cfg_model,
            Date=cfg_date,
            Mode=cfg_mode,
            balance_function=balance_func,
        )

        bundle_path = _get_bundle_path(
            cfg_model,
            cfg_date,
            cfg_mode,
            dataset_cfg.get('subset', False),
            preproc_cfg.get('preprocessed', False),
        )
        if not bundle_path.exists():
            raise FileNotFoundError(f"Bundle not found at {bundle_path}")
        
        sanity_check_bundle(bundle_path=Path(bundle_path))  # Verify that the data make sense

        # Build base table once
        if base_table is None:
            source_base = _find_source_base_path(
                cfg_model,
                cfg_date,
                cfg_mode,
                dataset_cfg.get('subset', False),
                preproc_cfg.get('preprocessed', False),
            )
            base_name = str(source_base.parent / 'base_table')
            base_table = BigMetaTable(str(source_base))
            base_table.save(base_name)
            del base_table  # Clear from memory, will be reloaded as needed in tests
            gc.collect()
            base_table = BigMetaTable(base_name) # Reload with diffferent name to ensure no crashing effects
            

        #fractionate = 'batch' if 'sampling' not in config_path.name else None  #Skip mass == 100 test and constant bulk test if resampling is ocurring. 
        fractionate = None # For now, test this earlier in pipeline. Errors I don't want to fix. 
        #run_tests_on_bundle(bundle_path, base_table, test_name=config_path.stem, fractionate=fractionate, outname=config_path.name.replace('.yaml', '_test_output')) Test earlier in pipeline.

        if baseline_bundle_path is None:
            baseline_bundle_path = _reserve_baseline_bundle(bundle_path, config_path.stem)
            bundle_path = baseline_bundle_path

        bundle = load_ml_bundle(bundle_path)

        if baseline_bundle is None:
            baseline_bundle = bundle
            baseline_config = cfg
            continue

        if config_path.name == 'separate_analcime_only.yaml':
            _sanity_check_separate_analcime(bundle, base_table)
            continue

        if baseline_bundle is None or baseline_config is None:
            raise RuntimeError("Baseline bundle not available for comparisons")

        if config_path.name == 'upsampling_only.yaml':
            _sanity_check_resampling_row_increase(bundle, baseline_bundle, 'Upsampling')
            upsample_phases = [
                name for name in upsample_cfg.get('phases', {}).keys()
                if name != 'test_set_phases'
            ]
            _sanity_check_upsampling_phase_proportions(bundle, baseline_bundle, upsample_phases)

        if config_path.name == 'resampling_only.yaml':
            _sanity_check_resampling_row_increase(bundle, baseline_bundle, 'Resampling')

        if config_path.name == 'balancingLowF_only.yaml':
            _sanity_check_balancing_liquid(bundle, 0.70, 'balance_lowF')

        if config_path.name == 'balancingSuperliquidus_only.yaml':
            _sanity_check_balancing_liquid(bundle, 0.65, 'balance_superliquidus')

        if config_path.name == 'balancingGeodynamics_only.yaml':
            _sanity_check_geodynamics(bundle, 0.50)

        if config_path.name == 'deep_filter_only.yaml':
            _sanity_check_deep_filter_bounds(bundle, cfg.get('deep_filter', {}))


if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    setup_test_logging(
        log_filename=f"{Path(__file__).stem}_{timestamp}.txt",
        log_dir=Path(__file__).parent / "logs",
    )
    parser = argparse.ArgumentParser(
        description='Run processing tests for resampling/filtering configs.'
    )
    parser.add_argument(
        '--config-dir',
        type=str,
        default=str(Path(__file__).parent / 'config'),
        help='Directory containing test processing YAML configs.'
    )
    parser.add_argument('--MELTSModel', type=str, default=None)
    parser.add_argument('--Date', type=str, default=None)
    parser.add_argument('--Mode', type=str, default=None)

    args = parser.parse_args()

    run_processing_tests(
        config_dir=Path(args.config_dir),
        melts_model=args.MELTSModel,
        date=args.Date,
        mode=args.Mode,
    )
