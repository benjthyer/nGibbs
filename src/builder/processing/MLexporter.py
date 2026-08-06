from tqdm import tqdm
import numpy as np
import gc
import os
import csv
import tarfile
import tempfile
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
import argparse
import json

# Ensure repo root and src are on path
import sys
repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from recipes.settings import internal_train_dir, external_train_dir, external_base
from ngibbs.utils.string_utils import pull_number
from tests.unit_tests.test_processing.ML_export_tests import sanity_check_bundle
from ngibbs.utils.file_utils import load_ml_bundle, MLDataBundle, chunked_permutation_copy
from ngibbs.utils.math_utils import Normalizer
from ngibbs.config.ml_indexer import load_ml_indexer_from_state
from builder.processing.BigMetaTable import BigMetaTable

# Row-aligned arrays that may be present in an ML-ready bundle. Every one that
# actually exists gets shuffled with the *same* permutation in shuffle_bundle_rows,
# to keep rows aligned across arrays. Not all bundles have mass_labels/free_outputs
# (the latter is optional; the former is always written by resampling_to_datasets
# today, but guarded here in case of older/hand-built bundles).
_ROW_ALIGNED_ARRAYS = (
    "features.npy",
    "labels.npy",
    "binary_labels.npy",
    "molar_labels.npy",
    "mass_labels.npy",
    "free_outputs.npy",
)

#featureNames = ['Pressure(System_main)', 'Temperature(System_main)', 'logfO2-QFM(System_main)']
#free_outputs = ['viscocity(System_main)', 'liq H (kJ)(melts-liquid)', 'Temperature(System_main)']
#free_outputs = None





def resampling_to_datasets(self, resample_bounds = [[1,1]], clear_old_tables=False, featureNames=["Pressure(System_main)", "Temperature(System_main)"],
                            free_outputs=None, indexer=None, config_path=None, bundle_name=None, chunk_size=None,
                            deep_filter_kwargs=None, insanity_filter_kwargs=None):

    """Builds features and labels for training. Converts MELTS tables to .npy files fit for ML work.
    Self: BigMetaTable Instance.

    Parameters
    ----------
    bundle_name : str, optional
        Optional .tar.gz filename to use for the final bundle (stored in the dataset directory).
    chunk_size : int, optional
        Row-chunk size for every large-array pass below. Defaults to `self.chunk_size`
        (the BigMetaTable instance's configured chunk size - see YAML `performance.chunk_size`)
        when not given explicitly.
    deep_filter_kwargs : dict, optional
        If given, applies `filters.deep_filter_npy` to the exported .npy files
        in place before packaging (kwargs forwarded as-is, e.g.
        Oxide_Lower_Bounds, Component_Upper_Bounds, batch_size). Filtering
        before packaging avoids the extract/repack round trip that applying
        these filters to the finished .tar.gz bundle would require.
    insanity_filter_kwargs : dict, optional
        If given, applies `filters.insanity_filter_npy` to the exported .npy
        files in place before packaging (kwargs forwarded as-is, e.g.
        tolerance, bulk_tol_frac, batch_size).
    """

    if chunk_size is None:
        chunk_size = getattr(self, 'chunk_size', 100_000)

    sampleNo = len(resample_bounds)
    debug_dump_enabled = os.getenv('NMELTS_DEBUG_DUMP_MOLAR', '').lower() in ('1', 'true', 'yes', 'on')
    debug_dump_done = False

    def _write_labeled_csv(path, data, row_labels=None, col_labels=None, row_label_name='row_id'):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            if col_labels is not None:
                if row_labels is not None:
                    writer.writerow([row_label_name] + list(col_labels))
                else:
                    writer.writerow(list(col_labels))

            if row_labels is not None:
                for label, row in zip(row_labels, data):
                    writer.writerow([label] + list(row))
            else:
                writer.writerows(data)


    def _parse_feature_entry(entry):
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            return entry[0], entry[1]
        if isinstance(entry, str):
            entry = entry.strip()
            open_idx = entry.rfind('(')
            close_idx = entry.rfind(')')
            if open_idx == -1 or close_idx != len(entry) - 1 or open_idx > close_idx:
                raise ValueError(f"Feature entry must match 'Component(Phase)', got: {entry}")
            comp = entry[:open_idx].strip()
            phase = entry[open_idx + 1:close_idx].strip()
            if not comp or not phase:
                raise ValueError(f"Feature entry must match 'Component(Phase)', got: {entry}")
            return comp, phase
        raise ValueError(f"Feature entry must be 'Component(Phase)' or [component, phase], got: {entry}")

    def _is_ratio_entry(entry):
        """Return True if entry uses ' / ' division syntax."""
        return isinstance(entry, str) and ' / ' in entry

    def _split_ratio(entry):
        """Split 'A(P1) / B(P2)' into the two half-strings."""
        parts = entry.split(' / ', 1)
        return parts[0].strip(), parts[1].strip()

    if sampleNo > 1:
        allowed_pairs = {
            ('Pressure', 'System_main'),
            ('Temperature', 'System_main'),
            ('logfO2-QFM', 'System_main'),
        }
        for FN in featureNames:
            if _is_ratio_entry(FN):
                for part in _split_ratio(FN):
                    comp, phase = _parse_feature_entry(part)
                    if ((comp, phase) not in allowed_pairs) and (resample_bounds != [[1, 1]]):
                        raise NotImplementedError(
                            f"(Feature {comp}({phase}) in ratio) Extensive features not implemented for resampling; only PTfO2 allowed"
                        )
            else:
                comp, phase = _parse_feature_entry(FN)
                if ((comp, phase) not in allowed_pairs) and (resample_bounds != [[1,1]]):
                    raise NotImplementedError(
                        f"(Feature {comp}({phase})) Extensive features not implemented for resampling; only PTfO2 allowed"
                    )
        print(f"Resampling dataset {sampleNo} times with bounds: {resample_bounds}")


    # Use the provided indexer or fall back to self.indexe
    if indexer is None:
        if hasattr(self, 'indexer'):
            indexer = self.indexer
        else:
            raise ValueError("No indexer provided and self.indexer not found")

    #if hasattr(indexer, 'ml_indexer'):
    indexer.ml_indexer.featureNames = featureNames
    indexer.ml_indexer.free_outputs = free_outputs
    
    # Extract all needed indices and matrices from the indexer
    label_indices_comp = indexer.label_indices_comp
    label_indices = indexer.label_indices
    mass_phasedict = indexer.mass_phasedict
    mass_indices = indexer.mass_indices
    compToOxLoad = indexer.ml_indexer.compToOxLoad
    OxToEl = indexer.ml_indexer.OxToEl
    phaseToCompMap = indexer.ml_indexer.phaseToCompMap
    detail_label_indices = indexer.detail_label_indices
    component_indices = indexer.MELTS_indices
    compositionally_variable_phases = indexer.compositionally_variable_phases
    MM = indexer.ml_indexer.MM
    Elkeys = indexer.ml_indexer.Elkeys

    # Parse feature names (as [component, phase]) to column indices in MELTS table
    def _simple_feature_to_index(entry) -> int:
        comp, phase = _parse_feature_entry(entry)
        if phase not in component_indices or comp not in component_indices[phase]:
            raise KeyError(f"Feature {comp}({phase}) not found in MELTS_indices.")
        return component_indices[phase][comp]

    def _feature_to_index(entry):
        """Return int for simple features, (int, int) for 'A(P) / B(P)' ratio features."""
        if _is_ratio_entry(entry):
            num_str, den_str = _split_ratio(entry)
            return (_simple_feature_to_index(num_str), _simple_feature_to_index(den_str))
        return _simple_feature_to_index(entry)

    feature_indices = [ _feature_to_index(n) for n in featureNames ]
    feature_offset = len(featureNames)
    
    # Optional free outputs: indices for arbitrary labels not constrained by phase equilibria
    free_output_indices = []
    if free_outputs is not None:
        if len(free_outputs) > 0:
            free_output_indices = [ _feature_to_index(n) for n in free_outputs ]
    
    num_rows = np.shape(self.table)[0]
    total_rows = np.shape(self.table)[0]
    num_components_intensive = indexer.ml_indexer.ncompsVaried
    #num_components_extensive = indexer.ml_indexer.ncomps
    num_phases = indexer.ml_indexer.nphases
    print('INITIALIZE')
    
    if clear_old_tables:
        del self.binarylabels
        del self.features
        del self.labels
        del self.masslabels
        del self.table1
        del self.molar
        if hasattr(self, 'free_outputs'):
            del self.free_outputs
        gc.collect()
        
    new_file = self.filename + '_temp.npy'
    self.table1 = np.lib.format.open_memmap(
            new_file,
            mode='w+',
            dtype=self.table.dtype,
            shape=self.table.shape
    )

    # Copy data one row-chunk at a time, so this never holds the whole table's worth of
    # dirty pages in RAM before anything reaches disk. This is done to avoid mutating the
    # original table during resampling.
    self._chunked_copy_into(self.table, self.table1, chunk_size=chunk_size)

    if self.blurredbinaries is None:
        binary_source = None  # derived per-chunk below from self.table; never held in full
    else:
        assert self.blurredbinaries.shape[0] == self.table.shape[0], "Table of labels and blurred binaries must have the same number of rows!"
        print('Using Blurred Binaries to generate binary labels...')
        binary_source = self.blurredbinaries
    
    self.molarlabels =np.lib.format.open_memmap( # Molar abundances
            self.filename + 'molar_labels.npy',
            mode='w+',
            dtype=np.float32,
            shape=(num_rows*len(resample_bounds), num_phases)
    )
        
    self.binarylabels = np.lib.format.open_memmap( # Flags of present phases
            self.filename + 'binary_labels.npy',
            mode='w+',
            dtype=np.float32,
            shape=(num_rows*len(resample_bounds), num_phases)
    )
    
    self.masslabels = np.lib.format.open_memmap( # Masses in grams, normed to 100
            self.filename + 'mass_labels.npy',
            mode='w+',
            dtype=np.float32,
            shape=(num_rows*len(resample_bounds), num_phases)
    )
    
    self.features = np.lib.format.open_memmap( # Input features + bulk chemistry X in element moles, normed to sum of 1
            self.filename + 'features.npy',
            mode='w+',
            dtype=np.float32,
            shape=(num_rows*len(resample_bounds), feature_offset + len(Elkeys))
        )
    
    
    
    # Allocate free outputs memmap if requested
    if free_outputs is not None and len(free_output_indices) > 0:
        self.free_outputs = np.lib.format.open_memmap(
            self.filename + 'free_outputs.npy',
            mode='w+',
            dtype=np.float32,
            shape=(num_rows*len(resample_bounds), len(free_output_indices))
        )
    
    self.labels = np.lib.format.open_memmap( # Components in moles, intensive only
            self.filename + 'labels.npy',
            mode='w+',
            dtype=np.float32,
            shape=(num_rows*len(resample_bounds), num_components_intensive)
        )
    
    try:
        for i, bounds in enumerate(tqdm(resample_bounds, desc = 'Generating Molar Features and Labels', leave = False)):
            print(f"SAMPLE: {i}")
            # Vary proportions of equilibrium assemblages, one row-chunk at a time -
            # `self.table1[:, mass_indices]` across every row is a strided read/write
            # over the whole (row-major) memmap; chunking keeps each pass contiguous.
            for start in tqdm(range(0, total_rows, chunk_size), desc = "Chunking Molar Feature Calculations...", leave=False):
                end = min(start + chunk_size, total_rows)
                chunk_t1 = self.table1[start:end]
                mass_multipliers = np.random.uniform(*bounds, size=(end - start, num_phases))
                chunk_t1[:, mass_indices] *= mass_multipliers
                chunk_t1[:, mass_indices] *= 100/np.sum(chunk_t1[:, mass_indices], axis=1, keepdims=True)
                self.table1[start:end] = chunk_t1
            self.table1.flush()

            # Get molar quantities
            self.retrieve_component_moles()

            if debug_dump_enabled and not debug_dump_done:
                print("Debug dump of molar and compToOxLoad previews enabled. Saving to debug/ subdirectory...")
                debug_dir = Path(self.filename).parent / 'debug'
                debug_dir.mkdir(parents=True, exist_ok=True)

                comp_labels = list(getattr(indexer.ml_indexer, 'label_names', []))
                if len(comp_labels) != self.molar.shape[1]:
                    comp_labels = [f"comp_{j}" for j in range(self.molar.shape[1])]

                oxide_labels = list(getattr(indexer.ml_indexer, 'Oxides', []))
                if len(oxide_labels) != compToOxLoad.shape[1]:
                    oxide_labels = [f"oxide_{j}" for j in range(compToOxLoad.shape[1])]

                max_rows_str = os.getenv('NMELTS_DEBUG_DUMP_MOLAR_ROWS', '2000')
                try:
                    max_rows = max(1, int(max_rows_str))
                except ValueError:
                    max_rows = 2000

                row_count = min(self.molar.shape[0], max_rows)
                molar_preview = np.asarray(self.molar[:row_count])
                molar_row_labels = [f"sample_{j}" for j in range(row_count)]

                np.save(debug_dir / f"{Path(self.filename).name}_molar_preview.npy", molar_preview)
                _write_labeled_csv(
                    debug_dir / f"{Path(self.filename).name}_molar_preview.csv",
                    molar_preview,
                    row_labels=molar_row_labels,
                    col_labels=comp_labels,
                    row_label_name='sample',
                )

                _write_labeled_csv(
                    debug_dir / f"{Path(self.filename).name}_compToOxLoad.csv",
                    np.asarray(compToOxLoad),
                    row_labels=comp_labels,
                    col_labels=oxide_labels,
                    row_label_name='component',
                )

                print(
                    f"Saved debug molar/compToOxLoad previews to {debug_dir} "
                    f"(rows={row_count})"
                )
                debug_dump_done = True
                        
            # Every downstream label/feature array is derived from self.molar/self.table1/
            # self.table, one row-chunk at a time. This used to read self.molar in full
            # (once for Inmoles, again for masslabels, again for molarlabels, again per
            # phase in the labels loop below) and self.table/self.table1 in full per
            # feature/free-output column - each chunk is now read once and every output
            # for that row range is derived from it before moving to the next chunk.
            for start in tqdm(range(0, num_rows, chunk_size), desc=f"Building labels/features (sample {i})", leave=False):
                end = min(start + chunk_size, num_rows)
                out_start, out_end = i*num_rows + start, i*num_rows + end

                molar_chunk = self.molar[start:end]
                Inmoles_chunk = (molar_chunk @ compToOxLoad) @ OxToEl
                InTot_chunk = np.sum(Inmoles_chunk, axis=1).reshape(-1, 1)

                # --- Binary labels (computed per-chunk rather than precomputed for all
                # rows up front, since it doesn't depend on the resample index and a
                # full (total_rows, num_phases) array is large enough to OOM on
                # multi-hundred-million-row tables)
                #print(f"Building binary labels for rows {start}:{end} (sample {i})")
                if binary_source is None:
                    self.binarylabels[out_start:out_end] = (self.table[start:end, mass_indices] > 0).astype(np.float32)
                else:
                    self.binarylabels[out_start:out_end] = binary_source[start:end]

                #print(f"Building mass labels for rows {start}:{end} (sample {i})")
                # --- Mass labels
                if self.Model == 'HeFESTo':
                    phaseComps = molar_chunk[:, :, np.newaxis] * phaseToCompMap.T  # (B, C, P)

                    # Convert to oxides per phase (plug in iron speciator). Moles, then grams
                    phaseOxMolar = np.einsum("bcp,co->bpo", phaseComps, compToOxLoad)

                    phaseOxMass = np.einsum("bpo,oo->bpo", phaseOxMolar, MM)

                    # Compute total phase masses
                    phaseMass = np.sum(phaseOxMass, axis=-1)  # (B, P)

                    # Normalize systemwide to 100%
                    systemTotal = phaseMass.sum(axis=-1, keepdims=True)  # (B, 1)
                    phaseMassNorm = 100.0 * phaseMass / (systemTotal)
                    self.masslabels[out_start:out_end] = phaseMassNorm

                else: # For MELTS, we can just use the mass columns directly (already wt%)
                    self.masslabels[out_start:out_end] = self.table1[start:end, mass_indices]

                # --- Free outputs (if any): replicate values from original table across resamples
                if free_outputs is not None and len(free_output_indices) > 0:
                    #print(f"Building free outputs for rows {start}:{end} (sample {i})")
                    for k, fidx in enumerate(free_output_indices):
                        if isinstance(fidx, tuple):
                            num_col = self.table[start:end, fidx[0]].astype(np.float64)
                            den_col = self.table[start:end, fidx[1]].astype(np.float64)
                            self.free_outputs[out_start:out_end, k] = np.where(den_col != 0, num_col / den_col, 0.0).astype(np.float32)
                        else:
                            self.free_outputs[out_start:out_end, k] = self.table[start:end, fidx]

                # --- Features (bulk chemistry in elements normalized to 1)
                #print(f"Building features for rows {start}:{end} (sample {i})")
                self.features[out_start:out_end, feature_offset:] = (Inmoles_chunk / InTot_chunk)

                # --- Features (selected input variables from table by featureNames)
                for k, fidx in enumerate(feature_indices):
                    if isinstance(fidx, tuple):
                        num_col = self.table[start:end, fidx[0]].astype(np.float64)
                        den_col = self.table[start:end, fidx[1]].astype(np.float64)
                        self.features[out_start:out_end, k] = np.where(den_col != 0, num_col / den_col, 0.0).astype(np.float32)
                    else:
                        self.features[out_start:out_end, k] = self.table[start:end, fidx]

                # --- Molar labels
                #print(f"Building molar labels for rows {start}:{end} (sample {i})")
                self.molarlabels[out_start:out_end] = (molar_chunk / InTot_chunk) @ phaseToCompMap.T

                for phase, idx in label_indices.items(): # Move components into the right space. Using molar_chunk to support HeFESTo and MELTS with same code
                    if len(idx) == 1:
                        continue # Skip pure phases

                    if phase != 'melts-liquid':
                        input_idx = label_indices_comp[phase]
                        # --- Labels (phase components)
                        row_tot = np.sum(molar_chunk[:, idx], axis=1)
                        nonzeros = row_tot != 0
                        normed = np.zeros_like(molar_chunk[:, idx])
                        normed[nonzeros] = molar_chunk[np.ix_(nonzeros, idx)] / row_tot[nonzeros].reshape(-1, 1)
                        self.labels[out_start:out_end, input_idx] = normed

                    if phase == 'melts-liquid':
                        liq_mol = molar_chunk[:, label_indices[phase]] # Non-normalized liquid element moles
                        liq_tot = np.sum(liq_mol, axis=1)
                        liqNonzero = liq_tot != 0
                        liq_mol[liqNonzero] = liq_mol[liqNonzero] / liq_tot[liqNonzero].reshape(-1, 1) # Normalize to sum 1

                        # --- Labels (phase components)
                        self.labels[out_start:out_end, label_indices_comp[phase]] = liq_mol

            print(f"Finished sample {i} of {len(resample_bounds)}. Flushing memmaps to disk...")
            self.binarylabels.flush()
            self.masslabels.flush()
            if free_outputs is not None and len(free_output_indices) > 0:
                self.free_outputs.flush()
            self.features.flush()
            self.molarlabels.flush()
            self.labels.flush()

            # Explicitly collect to close lingering references
            del self.molar #  Delete memmap reference!
            gc.collect()

        ## Filter out data where features are improperly summed. Why is that? 
        """
        bulk_wt_ox = (
            self.features[:, feature_offset:]
            @ np.linalg.inv(OxToEl[:len(Elkeys), :len(Elkeys)])
        ) @ MM[:len(Elkeys), :len(Elkeys)]
        bulk_wt_ox = 100*bulk_wt_ox/np.sum(bulk_wt_ox, axis = 1).reshape(-1,1)

        GT_comps = np.zeros((self.features.shape[0], indexer.ml_indexer.ncomps))

        for phase in np.array(list(label_indices.keys())):
            if phase in compositionally_variable_phases:
                GT_comps[:,label_indices[phase]] = (self.molarlabels[:, mass_phasedict[phase]]).reshape(-1,1) * self.labels[:,label_indices_comp[phase]]
            else:
                GT_comps[:,label_indices[phase]] = (self.molarlabels[:, mass_phasedict[phase]]).reshape(-1,1)

        GTReconBulk_oxides = (
            (((GT_comps @ compToOxLoad) @ OxToEl)
             @ np.linalg.inv(OxToEl[:len(Elkeys), :len(Elkeys)]))
            @ MM[:len(Elkeys), :len(Elkeys)]
        )
        GTReconBulk_oxides =  GTReconBulk_oxides*100/np.sum(GTReconBulk_oxides,axis=1, keepdims=True)


        # This is the indices of data to remove from self.binarylabels, self.masslabels, self.features, self.labels, and self.molarlabels
        mismatches = np.unique(np.where(np.round(bulk_wt_ox,2) != np.round(GTReconBulk_oxides,2))[0])

        # Indices of rows that are good
        keep_mask = np.ones(num_rows*len(resample_bounds), dtype=bool)
        keep_mask[mismatches] = False
        
        
        #Clearing more references...
        del bulk_wt_ox, GTReconBulk_oxides, GT_comps
        gc.collect()"""
       
    finally: #Close Memmaps
        print("Closing memmaps and cleaning up...")
        del self.binarylabels, self.masslabels, self.features, self.labels, self.table1, self.molarlabels
        if hasattr(self, 'free_outputs'):
            del self.free_outputs
        gc.collect()

    indexer_dir = self.filename + 'ml_indexer'
    indexer.ml_indexer.save(indexer_dir)

    print("Generating dataset statistics...")
    # Generate dataset statistics
    stats_path = generate_dataset_stats(
        dataset_name=self.filename,
        ml_indexer=indexer.ml_indexer,
        output_dir=Path(self.filename).parent,
        chunk_size=chunk_size
    )
    # generate_dataset_stats() also writes a "{stem}_feature_bounds.json"
    # companion (featureNames' min/max) next to stats_path - same stem, so
    # derive its path the same way rather than changing generate_dataset_stats'
    # return signature.
    feature_bounds_path = Path(self.filename).parent / f"{Path(self.filename).stem}_feature_bounds.json"

    # Apply deep/insanity filters directly to the exported .npy files, before
    # packaging - filtering the finished .tar.gz instead would mean extracting
    # and repacking it, which is an expensive and unnecessary round trip on
    # large binary datasets.
    if deep_filter_kwargs is not None or insanity_filter_kwargs is not None:
        from builder.processing.filters import deep_filter_npy, insanity_filter_npy

        if stats_path and stats_path.exists():
            os.replace(stats_path, stats_path.with_name(f"{stats_path.stem}_prefilter.txt"))
        if feature_bounds_path.exists():
            os.replace(feature_bounds_path, feature_bounds_path.with_name(f"{feature_bounds_path.stem}_prefilter.json"))

        if deep_filter_kwargs is not None:
            print("Applying deep_filter to unpacked dataset (pre-packaging)...")
            deep_filter_npy(self.filename, indexer.ml_indexer, **deep_filter_kwargs)

        if insanity_filter_kwargs is not None:
            print("Applying insanity_filter to unpacked dataset (pre-packaging)...")
            insanity_filter_npy(self.filename, indexer.ml_indexer, **insanity_filter_kwargs)

        print("Regenerating dataset statistics after filtering...")
        stats_path = generate_dataset_stats(
            dataset_name=self.filename,
            ml_indexer=indexer.ml_indexer,
            output_dir=Path(self.filename).parent,
            chunk_size=chunk_size
        )

    # Map full file paths to simple archive names (without self.filename prefix for readability)
    file_mappings = {
        self.filename + 'molar_labels.npy': 'molar_labels.npy',
        self.filename + 'binary_labels.npy': 'binary_labels.npy',
        self.filename + 'mass_labels.npy': 'mass_labels.npy',
        self.filename + 'features.npy': 'features.npy',
        self.filename + 'labels.npy': 'labels.npy',
    }
    if config_path:
        config_basename = Path(config_path).name
        file_mappings[str(config_path)] = config_basename
    if free_outputs is not None:
        file_mappings[self.filename + 'free_outputs.npy'] = 'free_outputs.npy'
    
    # Add stats file
    if stats_path and stats_path.exists():
        file_mappings[str(stats_path)] = 'stats.txt'

    # Add feature bounds file (featureNames' min/max - see generate_dataset_stats)
    if feature_bounds_path.exists():
        file_mappings[str(feature_bounds_path)] = 'feature_bounds.json'

    bundle_base = self.filename.split('_working')[0]
    if bundle_name:
        bundle_filename = Path(bundle_name).name
        if not bundle_filename.endswith('.tar.gz'):
            bundle_filename += '.tar.gz'
        bundle_path = str(Path(bundle_base).parent / bundle_filename)
    else:
        bundle_path = bundle_base + '.tar.gz'

    print("Packaging dataset bundle...")
    with tarfile.open(bundle_path, 'w:gz') as tar:
        for fpath, arcname in file_mappings.items():
            if os.path.exists(fpath):
                tar.add(fpath, arcname=arcname)
        if os.path.isdir(indexer_dir):
            tar.add(indexer_dir, arcname='ml_indexer')
        
    #sanity_check_bundle(bundle_path=Path(bundle_path)) # Verify that the data make sense. Expensive for large files
    

    print(f"Moving dataset bundle")
    # Move bundle to configured training directory
    model_match = re.search(r"MELTS([^_]+)_", Path(bundle_base).name)
    if model_match:
        melts_model = model_match.group(1)
        bundle_base_path = Path(bundle_base)
        if external_base and str(bundle_base_path).startswith(external_base):
            target_dir = Path(external_train_dir(melts_model))
        else:
            target_dir = Path(internal_train_dir(melts_model))
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / Path(bundle_path).name
        shutil.move(bundle_path, target_path)

        return target_path
    
    return bundle_path


def make_Tplots(MELTS, plot_directory, colormap = 'turbo'):
    if not os.path.exists(plot_directory[:-1]):
        os.makedirs(plot_directory[:-1])

    component_indices = MELTS.indexer.MELTS_indices
    
    conditional = MELTS.table[:,component_indices['melts-liquid']['liq mass (gm)']]>0
    
    if np.shape(MELTS.table)[0]> 200000:
        conditional = np.where(conditional)[0]
        conditional = conditional[np.random.randint(0,len(conditional),200000)]
        
    SiO2 = MELTS.table[conditional, component_indices['melts-liquid']['wt% SiO2']]
    Temp = MELTS.table[conditional,component_indices['System_main']['Temperature']]
    Pres = MELTS.table[conditional,component_indices['System_main']['Pressure']]
    fO2 = MELTS.table[conditional,component_indices['System_main']['logfO2-QFM']]
    plt.hist(Temp)
    plt.xlabel('Temperature °C')
    plt.ylabel('Counts')
    plt.title('MELTS Dataset Temperature Histogram')
    plt.savefig(plot_directory+'Temperature_Hist')
    plt.show()

    plt.hist(Pres)
    plt.xlabel('Pressure (Bars)')
    plt.ylabel('Counts')
    plt.title('MELTS Dataset Pressure Histogram')
    plt.savefig(plot_directory+'Pressure_Hist')
    plt.show()

    plt.hist(fO2)
    plt.xlabel(r"$log_{10}\left(fO_2\right) - QFM$")
    plt.ylabel('Counts')
    plt.title(r'MELTS Dataset $log_{10}\left(fO_2\right) - QFM$ Histogram')
    plt.savefig(plot_directory+'fO2_Hist')
    plt.show()

    plt.scatter(Pres, fO2, s=5, alpha = 0.1, c = Temp)
    plt.xlabel('Pressure (Bars)')
    plt.ylabel(r"$log_{10}\left(fO_2\right) - QFM$")
    plt.title('MELTS Dataset Pressure x fO2')
    plt.savefig(plot_directory+'fO2XPressure')
    plt.show()

    for label, ind in component_indices['melts-liquid'].items():
        if label == 'liq rho (gm/cc)':
            break
        data = MELTS.table[conditional,ind]
        if label == 'liq mass (gm)':
            literal_label = 'Liquid Mass Fraction (%)'
        else:
            literal_label = ""
            for char in label:
                if np.isnan(pull_number(char)):
                    if char == ' ':
                        literal_label += ' $'
                    else:
                        literal_label += char
                else:
                    literal_label += r'_' + char
        literal_label += r'$'
        plt.hist(data)
        plt.xlabel(label)
        plt.savefig(plot_directory+f'{label}_Hist')
        plt.show()
        #if label != 'wt% SiO2':
        fig, ax = plt.subplots()
        plt.scatter(Temp, data, s=5, alpha = 0.2, cmap=colormap, c = SiO2)#, c = Temp)
        plt.xlabel('Temperature, C')#r'$wt\% SiO_2$')
        plt.ylabel(literal_label)
        norm = plt.Normalize(vmin=np.min(SiO2), vmax=np.max(SiO2))
        sm = cm.ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('wt% SiO2')
        cbar.set_alpha(0.8)
        plt.title(f'Liquid {literal_label}')
        plt.tight_layout()
        plt.savefig(plot_directory+f'{label}XTemp')
        plt.show()
            
def make_harkers(MELTS, plot_directory, colormap = 'turbo', hist = False):
    if not os.path.exists(plot_directory[:-1]):
        os.makedirs(plot_directory[:-1])

    component_indices = MELTS.indexer.MELTS_indices

    conditional = MELTS.table[:,component_indices['melts-liquid']['liq mass (gm)']]>0
    if np.shape(MELTS.table)[0]> 200000:
        conditional = np.where(conditional)[0]
        conditional = conditional[np.random.randint(0,len(conditional),200000)]
        
    SiO2 = MELTS.table[conditional, component_indices['melts-liquid']['wt% SiO2']]
    Temp = MELTS.table[conditional,component_indices['System_main']['Temperature']]
    Pres = MELTS.table[conditional,component_indices['System_main']['Pressure']]
    fO2 = MELTS.table[conditional,component_indices['System_main']['logfO2-QFM']]
    if hist:
        plt.hist(Temp)
        plt.xlabel('Temperature °C')
        plt.ylabel('Counts')
        plt.title('MELTS Dataset Temperature Histogram')
        plt.savefig(plot_directory+'Temperature_Hist')
        plt.show()

        plt.hist(Pres)
        plt.xlabel('Pressure (Bars)')
        plt.ylabel('Counts')
        plt.title('MELTS Dataset Pressure Histogram')
        plt.savefig(plot_directory+'Pressure_Hist')
        plt.show()

        plt.hist(fO2)
        plt.xlabel(r"$log_{10}\left(fO_2\right) - QFM$")
        plt.ylabel('Counts')
        plt.title(r'MELTS Dataset $log_{10}\left(fO_2\right) - QFM$ Histogram')
        plt.savefig(plot_directory+'fO2_Hist')
        plt.show()

        plt.scatter(Pres, fO2, s=5, alpha = 0.2, c = Temp)
        plt.xlabel('Pressure (Bars)')
        plt.ylabel(r"$log_{10}\left(fO_2\right) - QFM$")
        plt.title('MELTS Dataset Pressure x fO2')
        plt.savefig(plot_directory+'fO2XPressure')
        plt.show()

    for label, ind in component_indices['melts-liquid'].items():
        if label == 'liq rho (gm/cc)':
            break
        
        data = MELTS.table[conditional,ind]
        if label == 'liq mass (gm)':
            literal_label = 'Liquid Mass Fraction (%)'
        else:
            literal_label = ""
            for char in label:
                if np.isnan(pull_number(char)):
                    if char == ' ':
                        literal_label += ' $'
                    else:
                        literal_label += char
                else:
                    literal_label += r'_' + char
            literal_label += r'$'
        if hist:
            plt.hist(data)
            plt.xlabel(label)
            plt.savefig(plot_directory+f'{label}_Hist.jpg', dpi = 256)
            plt.show()
        if label != 'wt% SiO2':
            fig, ax = plt.subplots()
            plt.scatter(SiO2, data, s=5, alpha = 0.2, cmap=colormap, c = Temp)#, c = Temp)
            plt.xlabel(r'$wt\% SiO_2$')#r'$wt\% SiO_2$')
            plt.ylabel(literal_label)
            norm = plt.Normalize(vmin=np.min(Temp), vmax=np.max(Temp))
            sm = cm.ScalarMappable(norm=norm, cmap=colormap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Temperature (C)')
            cbar.set_alpha(0.8)
            plt.title(f'Liquid {literal_label}')
            plt.tight_layout()
            plt.savefig(plot_directory+f'{label}XSiO2')
            plt.show()


def generate_dataset_stats(dataset_name, ml_indexer, output_dir=None, chunk_size=None):
    """
    Generate comprehensive statistics for a processed ML dataset.

    Creates a stats.txt file containing:
    - Dataset size
    - Phase abundances from binary_labels
    - Bulk composition bounds in wt% oxide (using ElToOx, iron as FeOT)
    - Condition bounds (P, T, fO2)
    - Liquid fraction distribution

    Parameters
    ----------
    dataset_name : str
        Base path to dataset files (without extensions)
    ml_indexer : MLIndexer
        ML indexer with transformation matrices and phase information
    output_dir : str or Path, optional
        Directory for output stats.txt. If None, uses dataset directory.
    chunk_size : int, optional
        Row-chunk size for the scan below. Defaults to 100_000.
    """
    from pathlib import Path

    if chunk_size is None:
        chunk_size = 100_000

    dataset_path = Path(dataset_name)
    if output_dir is None:
        output_dir = dataset_path.parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_file = output_dir / f"{dataset_path.stem}_stats.txt"

    # Memory-mapped rather than loaded in full - every statistic below (phase
    # abundance counts, bulk-oxide min/max, condition min/max, liquid-fraction
    # bin counts) is a simple reduction across rows, so a single row-chunked
    # pass accumulates all of them at once instead of holding
    # features.npy/binary_labels.npy/mass_labels.npy fully in RAM (three
    # full-size arrays simultaneously on a large dataset).
    features = np.load(f"{dataset_name}features.npy", mmap_mode='r')
    binary_labels = np.load(f"{dataset_name}binary_labels.npy", mmap_mode='r')
    mass_labels = np.load(f"{dataset_name}mass_labels.npy", mmap_mode='r')

    n_samples = features.shape[0]
    n_conditions = len(ml_indexer.featureNames)  # P, T, fO2
    n_chem_features = features.shape[1] - n_conditions

    all_phases = ml_indexer.all_phases
    n_phase_cols = min(len(all_phases), binary_labels.shape[1])

    ElToOx = ml_indexer.ElToOx
    MM_ox = ml_indexer.MM[:len(ml_indexer.Elkeys), :len(ml_indexer.Elkeys)]
    # ElToOx/MM_ox are both sized len(Elkeys) x len(Elkeys) (the reconstructed-from-
    # elements oxide space), which is ml_indexer.WRkeys - not ml_indexer.Oxides. The
    # two only coincide when every Elkey maps to a distinct oxide 1:1 (closed-system
    # iron, tracked as separate Fe/Fe3 elements); in open-system (buffered) mode,
    # Oxides carries an extra trailing 'Fe2O3' entry (needed elsewhere to read the
    # raw liquid wt% column) that has no corresponding column here.
    oxide_names = ml_indexer.WRkeys
    n_oxide_cols = len(oxide_names)

    liquid_idx = None
    for i, phase in enumerate(all_phases):
        if phase == 'melts-liquid':
            liquid_idx = i
            break
    has_liquid = liquid_idx is not None and liquid_idx < mass_labels.shape[1]

    bin_edges = np.arange(0, 100, 5)  # 0-5, 5-10, ..., 95-100

    # --- Accumulators: all sized by column count (phases/oxides/conditions/bins),
    # never by row count, so they stay tiny regardless of dataset size.
    phase_present_counts = np.zeros(n_phase_cols, dtype=np.int64)
    oxide_min = np.full(n_oxide_cols, np.inf)
    oxide_max = np.full(n_oxide_cols, -np.inf)
    cond_min = np.full(n_conditions, np.inf)
    cond_max = np.full(n_conditions, -np.inf)
    superliquidus = 0
    subsolidus = 0
    bin_counts = np.zeros(len(bin_edges), dtype=np.int64)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)

        binary_chunk = np.asarray(binary_labels[start:end, :n_phase_cols])
        phase_present_counts += np.sum(binary_chunk > 0, axis=0)

        features_chunk = np.asarray(features[start:end])

        # Bulk composition: element moles -> oxide moles -> wt% oxide
        chem_chunk = features_chunk[:, n_conditions:]
        oxide_moles_chunk = chem_chunk @ ElToOx
        oxide_wt_chunk = oxide_moles_chunk @ MM_ox
        oxide_wt_pct_chunk = 100 * oxide_wt_chunk / np.sum(oxide_wt_chunk, axis=1, keepdims=True)
        # np.fmin/fmax (not minimum/maximum) so an all-NaN column in one chunk
        # doesn't NaN-poison a running min/max that's valid in every other chunk.
        oxide_min = np.fmin(oxide_min, np.nanmin(oxide_wt_pct_chunk, axis=0))
        oxide_max = np.fmax(oxide_max, np.nanmax(oxide_wt_pct_chunk, axis=0))

        cond_chunk = features_chunk[:, :n_conditions]
        cond_min = np.fmin(cond_min, np.nanmin(cond_chunk, axis=0))
        cond_max = np.fmax(cond_max, np.nanmax(cond_chunk, axis=0))

        if has_liquid:
            liquid_wt_frac_chunk = np.asarray(mass_labels[start:end, liquid_idx]) / 100.0
            superliquidus += int(np.sum(liquid_wt_frac_chunk >= 0.995))
            subsolidus += int(np.sum(liquid_wt_frac_chunk < 0.005))
            pct_chunk = liquid_wt_frac_chunk * 100
            for i in range(len(bin_edges)):
                lower = bin_edges[i]
                upper = bin_edges[i] + 5 if i < len(bin_edges) - 1 else 100
                if upper <= 0.5 or lower >= 99.5:
                    continue
                bin_counts[i] += int(np.sum((pct_chunk >= lower) & (pct_chunk < upper)))

    for arr in (features, binary_labels, mass_labels):
        if hasattr(arr, '_mmap') and arr._mmap is not None:
            arr._mmap.close()
    del features, binary_labels, mass_labels
    gc.collect()

    # Helper function for horizontal histogram
    def make_horizontal_bar(percent, max_width=50):
        """Create horizontal bar: '19.75% ||||||||||||'"""
        n_bars = int(round(percent / 100.0 * max_width))
        return '|' * n_bars

    with open(stats_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"DATASET STATISTICS: {dataset_path.stem}\n")
        f.write("=" * 80 + "\n\n")

        # 1. Dataset size
        f.write(f"Dataset Size: {n_samples:,} samples\n\n")

        # 2. Phase abundances from binary_labels
        f.write("-" * 80 + "\n")
        f.write("PHASE ABUNDANCES (from binary labels)\n")
        f.write("-" * 80 + "\n")

        for i, phase in enumerate(all_phases):
            if i < n_phase_cols:
                percent = 100.0 * phase_present_counts[i] / n_samples
                bar = make_horizontal_bar(percent)
                f.write(f"{phase:20} {percent:6.2f}% {bar}\n")

        f.write("\n")

        # 3. Bulk composition bounds in wt% oxide space
        f.write("-" * 80 + "\n")
        f.write("BULK COMPOSITION BOUNDS (wt% oxide, FeO as FeOT)\n")
        f.write("-" * 80 + "\n")

        f.write(f"{'Oxide':>10} {'Min (wt%)':>12} {'Max (wt%)':>12}\n")
        f.write("-" * 40 + "\n")

        for i, oxide in enumerate(oxide_names):
            if i < n_oxide_cols:
                f.write(f"{oxide:>10} {oxide_min[i]:12.4f} {oxide_max[i]:12.4f}\n")

        f.write("\n")

        # 4. Condition bounds (P, T, fO2)
        f.write("-" * 80 + "\n")
        f.write("CONDITION BOUNDS\n")
        f.write("-" * 80 + "\n")

        condition_names = ml_indexer.featureNames #['Pressure (bars)', 'Temperature (°C)', 'logfO2-QFM']
        f.write(f"{'Condition':>20} {'Min':>15} {'Max':>15}\n")
        f.write("-" * 55 + "\n")

        for i, cond_name in enumerate(condition_names):
            if i < n_conditions:
                f.write(f"{cond_name:>20} {cond_min[i]:15.2f} {cond_max[i]:15.2f}\n")

        f.write("\n")

        # 5. Liquid fraction distribution
        f.write("-" * 80 + "\n")
        f.write("LIQUID FRACTION DISTRIBUTION (from mass labels)\n")
        f.write("-" * 80 + "\n")

        if has_liquid:
            # Superliquidus (liquid >= 99.5%)
            superliq_pct = 100.0 * superliquidus / n_samples
            bar = make_horizontal_bar(superliq_pct)
            f.write(f"{'Superliquidus (>99.5%)':30} {superliq_pct:6.2f}% {bar}\n")

            # Subsolidus (liquid < 0.5%)
            subsol_pct = 100.0 * subsolidus / n_samples
            bar = make_horizontal_bar(subsol_pct)
            f.write(f"{'Subsolidus (<0.5%)':30} {subsol_pct:6.2f}% {bar}\n")

            f.write("\n")

            # Bin remainder by 5% increments
            f.write("Intermediate liquid fractions (5% bins):\n")

            for i in range(len(bin_edges)):
                lower = bin_edges[i]
                upper = bin_edges[i] + 5 if i < len(bin_edges) - 1 else 100

                # Skip bins outside intermediate range
                if upper <= 0.5 or lower >= 99.5:
                    continue

                bin_pct = 100.0 * bin_counts[i] / n_samples

                if bin_pct > 0:  # Only show non-empty bins
                    bar = make_horizontal_bar(bin_pct)
                    f.write(f"{lower:3.0f}-{upper:3.0f}% liquid {bin_pct:6.2f}% {bar}\n")
        else:
            f.write("Warning: melts-liquid phase not found in mass labels\n")

        f.write("\n")
        f.write("=" * 80 + "\n")

    # Machine-readable companion to the "CONDITION BOUNDS" section above -
    # featureNames' min/max are an immutable property of this dataset, so they
    # belong here (computed once, during preprocessing) rather than being
    # rescanned by every training run that loads this bundle. Same stem as
    # stats_file so callers that already derive stats_file's path (e.g.
    # resampling_to_datasets) can derive this one identically.
    feature_bounds_file = output_dir / f"{dataset_path.stem}_feature_bounds.json"
    with open(feature_bounds_file, 'w') as f:
        json.dump(
            {
                "featureNames": list(ml_indexer.featureNames),
                "min": [float(v) for v in cond_min],
                "max": [float(v) for v in cond_max],
            },
            f,
            indent=2,
        )

    print(f"Generated statistics file: {stats_file}")
    print(f"Generated feature bounds file: {feature_bounds_file}")
    return stats_file


def shuffle_bundle_rows(bundle_path, seed=None, chunk_size=1_000_000):
    """Shuffle every row-aligned array in an already-packaged ML-ready bundle,
    in place, out-of-core.

    This is the same one-time out-of-core shuffle previously run against the
    raw (pre-filter, pre-resample) BigMetaTable via BigMetaTable.shuffle_rows(),
    moved to run here instead: on the small, already deep_filter/insanity_filter/
    resampling_to_datasets-trimmed bundle, rather than on the far larger table
    that produced it. On out-of-core-sized tables the old call site meant every
    chunked read was a fully-random gather across a file many times larger than
    RAM, with effectively zero page-cache locality - a pathological I/O pattern
    that could take substantially longer than shuffling the finished bundle.

    Parameters
    ----------
    bundle_path : str or Path
        Path to the .tar.gz ML-ready bundle to shuffle in place.
    seed : int, optional
        Seed for the row permutation (reproducibility). None (default) draws
        fresh entropy.
    chunk_size : int, optional
        Row-chunk size for the underlying chunked_permutation_copy passes and
        the stats regeneration scan.

    Returns
    -------
    Path
        `bundle_path`, unchanged (the bundle is shuffled in place).
    """
    bundle_path = Path(bundle_path)
    extract_dir = Path(tempfile.mkdtemp(dir=bundle_path.parent))

    try:
        with tarfile.open(bundle_path, "r:gz") as tar:
            original_names = [m.name for m in tar.getmembers() if m.isfile()]
            tar.extractall(path=extract_dir)

        present_arrays = [name for name in _ROW_ALIGNED_ARRAYS if (extract_dir / name).exists()]

        n_rows = np.load(extract_dir / "features.npy", mmap_mode="r").shape[0]
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(n_rows)
        print(f"[shuffle_bundle_rows] Shuffling {n_rows:,} rows in {bundle_path.name} (seed={seed})...")

        for name in present_arrays:
            src_path = extract_dir / name
            src = np.load(src_path, mmap_mode="r")
            tmp_path = extract_dir / f"_shuffled_{name}"
            chunked_permutation_copy(src, tmp_path, permutation, chunk_size=chunk_size)
            del src
            gc.collect()
            tmp_path.replace(src_path)
            print(f"[shuffle_bundle_rows]   shuffled {name}")

        print("[shuffle_bundle_rows] Regenerating stats.txt + feature_bounds.json...")
        ml_indexer = load_ml_indexer_from_state(str(extract_dir / "ml_indexer"))
        dataset_name = str(extract_dir) + "/"
        stats_path = generate_dataset_stats(
            dataset_name=dataset_name,
            ml_indexer=ml_indexer,
            output_dir=extract_dir,
            chunk_size=chunk_size,
        )
        feature_bounds_path = extract_dir / f"{Path(dataset_name).stem}_feature_bounds.json"

        # Repack: shuffled arrays + regenerated stats/feature_bounds under their
        # canonical arcnames, plus every other member the original bundle had
        # (e.g. a copied processing.yaml, ml_indexer/) carried through unchanged.
        handled_top_names = set(present_arrays) | {"stats.txt", "feature_bounds.json"}
        with tarfile.open(bundle_path, "w:gz") as tar:
            for name in present_arrays:
                tar.add(extract_dir / name, arcname=name)
            if stats_path and Path(stats_path).exists():
                tar.add(stats_path, arcname="stats.txt")
            if feature_bounds_path.exists():
                tar.add(feature_bounds_path, arcname="feature_bounds.json")
            indexer_dir = extract_dir / "ml_indexer"
            if indexer_dir.is_dir():
                tar.add(indexer_dir, arcname="ml_indexer")
                handled_top_names.add("ml_indexer")
            for name in original_names:
                top_name = Path(name).parts[0] if name else name
                if top_name in handled_top_names:
                    continue
                member_path = extract_dir / name
                if member_path.exists():
                    tar.add(member_path, arcname=name)

        print(f"[shuffle_bundle_rows] Done shuffling {bundle_path}")
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)

    return bundle_path


def _parse_resample_bounds(value):
    """Parse a JSON list of [low, high] mass-multiplier pairs, e.g. '[[1,1]]'."""
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(isinstance(b, list) and len(b) == 2 for b in parsed):
        raise argparse.ArgumentTypeError(
            f"--resample-bounds must be a JSON list of [low, high] pairs, got: {value}"
        )
    return parsed


def main():
    parser = argparse.ArgumentParser(
        description="Load a BigMetaTable and export resampled ML training datasets via resampling_to_datasets()."
    )

    # BigMetaTable construction args
    parser.add_argument("input_path", type=str, help="Path to table base name or to .csv/.txt/.npy file.")
    parser.add_argument("--read-dir", type=str, default=None,
                         help="Optional source directory used by BigMetaTable(read_dir=...).")
    parser.add_argument("--memmap-mode", type=str, default="r+", help="Mode used to open the table memmap.")
    parser.add_argument("--rebuild-memmap", action="store_true",
                         help="Force rebuilding the .npy memmap from the .csv/.txt source.")
    parser.add_argument("--allow-differing-lengths", action="store_true",
                         help="Allow header/table column count mismatch.")
    parser.add_argument("--model", type=str, default="MELTS", choices=["MELTS", "HeFESTo"],
                         help="Thermodynamic model used to build the table (BigMetaTable Model=).")
    parser.add_argument("--oxygen", type=str, default="closed", choices=["closed", "open"],
                         help="Oxygen buffering mode (BigMetaTable OXYGEN=).")
    parser.add_argument("--chunk-size", type=int, default=100_000,
                         help="Row-chunk size for BigMetaTable, and for resampling_to_datasets unless "
                              "--resample-chunk-size is given.")

    # resampling_to_datasets args
    parser.add_argument("--resample-bounds", type=_parse_resample_bounds, default=[[1, 1]],
                         help="JSON list of [low, high] mass-multiplier bounds, e.g. '[[1,1]]' or "
                              "'[[0.8,1.2],[0.9,1.1]]'.")
    parser.add_argument("--clear-old-tables", action="store_true",
                         help="Delete any previously computed label/feature arrays on the table before resampling.")
    parser.add_argument("--feature-names", nargs="+",
                         default=["Pressure(System_main)", "Temperature(System_main)"],
                         help="Feature column names, e.g. 'Pressure(System_main)'.")
    parser.add_argument("--free-outputs", nargs="+", default=None,
                         help="Optional free-output column names, not constrained by phase equilibria.")
    parser.add_argument("--config-path", type=str, default=None,
                         help="Optional config file to bundle alongside the exported dataset.")
    parser.add_argument("--bundle-name", type=str, default=None,
                         help="Optional .tar.gz filename for the final bundle.")
    parser.add_argument("--resample-chunk-size", type=int, default=None,
                         help="Row-chunk size override for resampling_to_datasets only (defaults to --chunk-size).")

    args = parser.parse_args()

    print(f"Loading BigMetaTable from: {args.input_path}")
    bmt = BigMetaTable(
        args.input_path,
        read_dir=args.read_dir,
        memmap_mode=args.memmap_mode,
        rebuild_memmap=args.rebuild_memmap,
        allow_differing_lengths=args.allow_differing_lengths,
        Model=args.model,
        OXYGEN=args.oxygen,
        chunk_size=args.chunk_size,
    )
    bmt.indexer.table_update(bmt.table)

    bundle_path = resampling_to_datasets(
        bmt,
        resample_bounds=args.resample_bounds,
        clear_old_tables=args.clear_old_tables,
        featureNames=args.feature_names,
        free_outputs=args.free_outputs,
        config_path=args.config_path,
        bundle_name=args.bundle_name,
        chunk_size=args.resample_chunk_size,
    )
    print(f"Export complete: {bundle_path}")


if __name__ == "__main__":
    main()