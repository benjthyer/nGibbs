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

# Ensure repo root and src are on path
import sys
repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from .filters import filter_invalid_rows
from recipes.settings import internal_train_dir, external_train_dir, external_base
from ngibbs.utils.string_utils import pull_number
from tests.unit_tests.test_processing.ML_export_tests import sanity_check_bundle
from ngibbs.utils.file_utils import load_ml_bundle, MLDataBundle
from ngibbs.utils.math_utils import Normalizer

#featureNames = ['Pressure(System_main)', 'Temperature(System_main)', 'logfO2-QFM(System_main)']
#freeOutputs = ['viscocity(System_main)', 'liq H (kJ)(melts-liquid)', 'Temperature(System_main)']
#freeOutputs = None





def resampling_to_datasets(self, resample_bounds = [[1,1]], clear_old_tables=False, featureNames=["Pressure(System_main)", "Temperature(System_main)"],
                            freeOutputs=None, indexer=None, config_path=None, bundle_name=None):

    """Builds features and labels for training. Converts MELTS tables to .npy files fit for ML work.
    Self: BigMetaTable Instance.

    Parameters
    ----------
    bundle_name : str, optional
        Optional .tar.gz filename to use for the final bundle (stored in the dataset directory).
    """

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
    indexer.ml_indexer.freeOutputs = freeOutputs
    
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
    if freeOutputs is not None:
        if len(freeOutputs) > 0:
            free_output_indices = [ _feature_to_index(n) for n in freeOutputs ]
    
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
        if hasattr(self, 'freeOutputs'):
            del self.freeOutputs
        gc.collect()
        
    new_file = self.filename + '_temp.npy'
    self.table1 = np.lib.format.open_memmap(
            new_file,
            mode='w+',
            dtype=self.table.dtype,
            shape=self.table.shape
    )

    # Copy data in blocks or all at once (depending on size). This is done to avoid mutating the original table during resampling.
    self.table1[:] = self.table[:]
    self.table1.flush()
    
    if self.blurredbinaries is None:
        binary_mask = (self.table[:,mass_indices]>0).astype(int)
    else:
        assert self.blurredbinaries.shape[0] == self.table.shape[0], "Table of labels and blurred binaries must have the same number of rows!"
        print('Using Blurred Binaries to generate binary labels...')
        binary_mask = self.blurredbinaries
    
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
    if freeOutputs is not None and len(free_output_indices) > 0:
        self.freeOutputs = np.lib.format.open_memmap(
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
            mass_multipliers = np.random.uniform(*bounds, size = total_rows*num_phases).reshape(total_rows,num_phases) # Vary proportions of equilibrium assemblages
            self.table1[:,mass_indices] *= mass_multipliers
            self.table1[:,mass_indices] *= 100/np.sum(self.table1[:,mass_indices], axis = 1, keepdims = True)
            self.table1.flush()
            
            # Get molar quantities
            self.retrieve_component_moles()

            if debug_dump_enabled and not debug_dump_done:
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
                        
            Inmoles = (self.molar @ compToOxLoad) @ OxToEl
            InTot = np.sum(Inmoles, axis = 1).reshape(-1,1)
            
            # --- Binary labels
            sl = np.s_[i*num_rows:(i+1)*num_rows]
            self.binarylabels[sl] = binary_mask
            self.binarylabels.flush()
            del sl

            # --- Mass labels
            sl = np.s_[i*num_rows:(i+1)*num_rows]
            if self.Model == 'HeFESTo':
                phaseComps = self.molar[:, :, np.newaxis] * phaseToCompMap.T  # (B, C, P)

                # Convert to oxides per phase (plug in iron speciator). Moles, then grams
                phaseOxMolar = np.einsum("bcp,co->bpo", phaseComps, compToOxLoad)
                

                phaseOxMass = np.einsum("bpo,oo->bpo", phaseOxMolar, MM)
                
                # Compute total phase masses
                phaseMass = np.sum(phaseOxMass, axis=-1)  # (B, P)

                # Normalize systemwide to 100%
                systemTotal = phaseMass.sum(axis=-1, keepdims=True)  # (B, 1)
                phaseMassNorm = 100.0 * phaseMass / (systemTotal)
                self.masslabels[sl] = phaseMassNorm

            else: # For MELTS, we can just use the mass columns directly (already wt%)
                self.masslabels[sl] = self.table1[:, mass_indices]

            self.masslabels.flush()
            del sl
            
            # --- Free outputs (if any): replicate values from original table across resamples
            if freeOutputs is not None and len(free_output_indices) > 0:
                sl_rows = slice(i*num_rows, (i+1)*num_rows)
                for k, fidx in enumerate(free_output_indices):
                    if isinstance(fidx, tuple):
                        num_col = self.table[:, fidx[0]].astype(np.float64)
                        den_col = self.table[:, fidx[1]].astype(np.float64)
                        self.freeOutputs[sl_rows, k] = np.where(den_col != 0, num_col / den_col, 0.0).astype(np.float32)
                    else:
                        self.freeOutputs[sl_rows, k] = self.table[:, fidx]
                self.freeOutputs.flush()

            # --- Features (bulk chemistry in elements normalized to 1)
            sl = np.s_[i*num_rows:(i+1)*num_rows, feature_offset:]
            print(feature_offset)
            print(Inmoles.shape)
            print(self.features.shape)
            print(len(self.indexer.ml_indexer.Elkeys))
            self.features[sl] = (Inmoles / InTot)
            self.features.flush()
            del sl

            # --- Features (selected input variables from table by featureNames)
            sl_rows = slice(i*num_rows, (i+1)*num_rows)
            for k, fidx in enumerate(feature_indices):
                if isinstance(fidx, tuple):
                    num_col = self.table[:, fidx[0]].astype(np.float64)
                    den_col = self.table[:, fidx[1]].astype(np.float64)
                    self.features[sl_rows, k] = np.where(den_col != 0, num_col / den_col, 0.0).astype(np.float32)
                else:
                    self.features[sl_rows, k] = self.table[:, fidx]
            self.features.flush()

            # --- Molar labels
            sl = np.s_[i*num_rows:(i+1)*num_rows]
            self.molarlabels[sl] = (self.molar / InTot) @ phaseToCompMap.T
            self.molarlabels.flush()
            del sl

            # Explicitly collect to close lingering references
            gc.collect()
            
            for phase, idx in label_indices.items(): # Move components into the right space. Using self.molar to support HeFESTo and MELTS with same code
                if len(idx) == 1:
                    continue # Skip pure phases
              
                if phase != 'melts-liquid':
                    input_idx = label_indices_comp[phase]
                    # --- Labels (phase components)
                    sl = np.s_[i*num_rows:(i+1)*num_rows, input_idx]
                    row_tot = np.sum(self.molar[:, idx], axis=1)
                    nonzeros = row_tot != 0
                    normed = np.zeros_like(self.molar[:, idx])
                    normed[nonzeros] = self.molar[np.ix_(nonzeros, idx)] / row_tot[nonzeros].reshape(-1, 1)
                    self.labels[sl] = normed
                    self.labels.flush()
                    del sl
                        

                if phase == 'melts-liquid':
                    liq_mol = self.molar[:,label_indices[phase]] # Non-normalized liquid element moles
                    liq_tot = np.sum(liq_mol, axis = 1)
                    liqNonzero = liq_tot != 0
                    liq_mol[liqNonzero] = liq_mol[liqNonzero] / liq_tot[liqNonzero].reshape(-1,1) # Normalize to sum 1
                    
                    # --- Labels (phase components)
                    sl = np.s_[i*num_rows:(i+1)*num_rows, label_indices_comp[phase]]
                    self.labels[sl] = liq_mol
                    self.labels.flush()
                    
                    del liq_mol, liq_tot, liqNonzero, sl
                    gc.collect()

            del self.molar, Inmoles, InTot #  Delete ALL memmap references!
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
        del self.binarylabels, self.masslabels, self.features, self.labels, self.table1, self.molarlabels
        if hasattr(self, 'freeOutputs'):
            del self.freeOutputs
        gc.collect()

    indexer_dir = self.filename + 'ml_indexer'
    indexer.ml_indexer.save(indexer_dir)

    # Generate dataset statistics
    stats_path = generate_dataset_stats(
        dataset_name=self.filename,
        ml_indexer=indexer.ml_indexer,
        output_dir=Path(self.filename).parent
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
    if freeOutputs is not None:
        file_mappings[self.filename + 'free_outputs.npy'] = 'free_outputs.npy'
    
    # Add stats file
    if stats_path and stats_path.exists():
        file_mappings[str(stats_path)] = 'stats.txt'

    bundle_base = self.filename.split('_working')[0]
    if bundle_name:
        bundle_filename = Path(bundle_name).name
        if not bundle_filename.endswith('.tar.gz'):
            bundle_filename += '.tar.gz'
        bundle_path = str(Path(bundle_base).parent / bundle_filename)
    else:
        bundle_path = bundle_base + '.tar.gz'


    with tarfile.open(bundle_path, 'w:gz') as tar:
        for fpath, arcname in file_mappings.items():
            if os.path.exists(fpath):
                tar.add(fpath, arcname=arcname)
        if os.path.isdir(indexer_dir):
            tar.add(indexer_dir, arcname='ml_indexer')
        
    #sanity_check_bundle(bundle_path=Path(bundle_path)) # Verify that the data make sense. Expensive for large files
    


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
    
    return bundle_path  #filter_invalid_rows(self, mismatches)


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


def generate_dataset_stats(dataset_name, ml_indexer, output_dir=None):
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
    """
    from pathlib import Path
    
    dataset_path = Path(dataset_name)
    if output_dir is None:
        output_dir = dataset_path.parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_file = output_dir / f"{dataset_path.stem}_stats.txt"
    
    # Load dataset arrays
    features = np.load(f"{dataset_name}features.npy")
    binary_labels = np.load(f"{dataset_name}binary_labels.npy")
    mass_labels = np.load(f"{dataset_name}mass_labels.npy")
    
    n_samples = features.shape[0]
    n_conditions = len(ml_indexer.featureNames)  # P, T, fO2
    n_chem_features = features.shape[1] - n_conditions
    
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
        
        all_phases = ml_indexer.all_phases
        for i, phase in enumerate(all_phases):
            if i < binary_labels.shape[1]:
                phase_present = np.sum(binary_labels[:, i] > 0)
                percent = 100.0 * phase_present / n_samples

                bar = make_horizontal_bar(percent)
                f.write(f"{phase:20} {percent:6.2f}% {bar}\n")
        
        f.write("\n")
        
        # 3. Bulk composition bounds in wt% oxide space
        f.write("-" * 80 + "\n")
        f.write("BULK COMPOSITION BOUNDS (wt% oxide, FeO as FeOT)\n")
        f.write("-" * 80 + "\n")
        
        # Extract chemical features (skip conditions)
        chem_features = features[:, n_conditions:]
        
        # Convert from element moles to wt% oxide using ElToOx
        ElToOx = ml_indexer.ElToOx
        
        # Element moles to oxide moles
        oxide_moles = chem_features @ ElToOx
        
        # Oxide moles to wt% (multiply by molar mass and normalize)
        MM_ox = ml_indexer.MM[:len(ml_indexer.Elkeys), :len(ml_indexer.Elkeys)]
        oxide_names = ml_indexer.Oxides
        
        # Calculate oxide wt% for each sample
        oxide_wt = oxide_moles @ MM_ox
        oxide_wt_pct = 100 * oxide_wt / np.sum(oxide_wt, axis=1, keepdims=True)
        
        f.write(f"{'Oxide':>10} {'Min (wt%)':>12} {'Max (wt%)':>12}\n")
        f.write("-" * 40 + "\n")
        
        for i, oxide in enumerate(oxide_names):
            if i < oxide_wt_pct.shape[1]:
                oxide_col = oxide_wt_pct[:, i]
                min_val = np.nanmin(oxide_col)
                max_val = np.nanmax(oxide_col)
                f.write(f"{oxide:>10} {min_val:12.4f} {max_val:12.4f}\n")

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
                cond_col = features[:, i]
                min_val = np.nanmin(cond_col)
                max_val = np.nanmax(cond_col)
                f.write(f"{cond_name:>20} {min_val:15.2f} {max_val:15.2f}\n")
        
        f.write("\n")
        
        # 5. Liquid fraction distribution
        f.write("-" * 80 + "\n")
        f.write("LIQUID FRACTION DISTRIBUTION (from mass labels)\n")
        f.write("-" * 80 + "\n")
        
        # Find liquid column in mass_labels
        liquid_idx = None
        for i, phase in enumerate(all_phases):
            if phase == 'melts-liquid':
                liquid_idx = i
                break
        
        if liquid_idx is not None and liquid_idx < mass_labels.shape[1]:
            liquid_wt_frac = mass_labels[:, liquid_idx] / 100.0  # Convert from wt% to fraction
            
            # Superliquidus (liquid >= 99.5%)
            superliquidus = np.sum(liquid_wt_frac >= 0.995)
            superliq_pct = 100.0 * superliquidus / n_samples
            bar = make_horizontal_bar(superliq_pct)
            f.write(f"{'Superliquidus (>99.5%)':30} {superliq_pct:6.2f}% {bar}\n")
            
            # Subsolidus (liquid < 0.5%)
            subsolidus = np.sum(liquid_wt_frac < 0.005)
            subsol_pct = 100.0 * subsolidus / n_samples
            bar = make_horizontal_bar(subsol_pct)
            f.write(f"{'Subsolidus (<0.5%)':30} {subsol_pct:6.2f}% {bar}\n")
            
            f.write("\n")
            
            # Bin remainder by 5% increments
            f.write("Intermediate liquid fractions (5% bins):\n")
            bin_edges = np.arange(0, 100, 5)  # 0-5, 5-10, ..., 95-100
            
            for i in range(len(bin_edges)):
                lower = bin_edges[i]
                upper = bin_edges[i] + 5 if i < len(bin_edges) - 1 else 100
                
                # Skip bins outside intermediate range
                if upper <= 0.5 or lower >= 99.5:
                    continue
                
                # Count samples in this bin
                in_bin = np.sum((liquid_wt_frac * 100 >= lower) & (liquid_wt_frac * 100 < upper))
                bin_pct = 100.0 * in_bin / n_samples
                
                if bin_pct > 0:  # Only show non-empty bins
                    bar = make_horizontal_bar(bin_pct)
                    f.write(f"{lower:3.0f}-{upper:3.0f}% liquid {bin_pct:6.2f}% {bar}\n")
        else:
            f.write("Warning: melts-liquid phase not found in mass labels\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
    
    print(f"Generated statistics file: {stats_file}")
    return stats_file