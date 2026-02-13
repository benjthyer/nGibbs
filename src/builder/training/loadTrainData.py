#time.sleep(3600*3)
import sys
import os
import gc
import pickle
import tarfile
import tempfile
import numpy as np
import torch
from pathlib import Path

# Ensure src is on path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

base_path = str(Path(__file__).parent.parent.parent.parent)
if base_path not in sys.path:
    sys.path.insert(0, base_path)

config_path = str(Path(__file__).parent.parent.parent.parent / 'config')
if config_path not in sys.path:
    sys.path.insert(0, config_path)

from builder.training.torchDataClass import TensorDatasetFour, TensorDatasetFive
from nMELTS.config.ml_indexer import MLIndexer
from settings import external_base
from nMELTS.utils.file_utils import load_ml_bundle, MLDataBundle
from tests.unit_tests.test_processing.ML_export_tests import sanity_check_bundle


MELTSModel = '102'
CalcType = 'FxCryst'
date = 'Nov9'
use_external = True  # Is data on external drive? Path defined in nMELTS.config.settings
subset = False
molar_epsilon = 1e-4 # Used for log-scaling phase molar abundances

####################
# This is a list that will be passed to ml_indexer.restrictVC(), or None to not restrict. Makes smaller model.
# It limits the chemical labels the model maps to to only the specified phases. This breaks the models ability to mass balance
only_VP = None
####################

REPO_ROOT = Path(__file__).resolve().parents[3]
MLREADY_DIR = REPO_ROOT / 'data' / 'MLready' / f'{MELTSModel}'
external_mlready_dir = Path(external_base) / 'MLready' / f'{MELTSModel}'

Trainfilename = str(MLREADY_DIR / f'MELTS{MELTSModel}_Trainset{date}{CalcType}Cooling')
Testfilename = str(MLREADY_DIR / f'MELTS{MELTSModel}_Testset{date}{CalcType}Cooling')
modelname = f"MELTS{MELTSModel}{CalcType}"

if subset:
    Trainfilename += '_subset'
    Testfilename += '_subset'

if not os.path.exists(f'{Trainfilename}.tar.gz') or use_external == True:
    Trainfilename = str(external_mlready_dir / f'MELTS{MELTSModel}_Trainset{date}{CalcType}Cooling')
    if subset:
        Trainfilename += '_subset'
if not os.path.exists(f'{Testfilename}.tar.gz') or use_external == True:
    Testfilename = str(external_mlready_dir / f'MELTS{MELTSModel}_Testset{date}{CalcType}Cooling')
    if subset:
        Testfilename += '_subset'

class Normalizer:
    """Quick Normalizing object that holds minima and ranges for a dataset and converts into and out of [0,1]
    minmax normalization for interfacing with neural networks"""

    def __init__(self, min_tensor, range_tensor):
        assert len(min_tensor) == len(range_tensor), 'Minimum and range are not equal!'
        self.miner = min_tensor
        self.ranger = range_tensor

    def __len__(self):
        return len(self.miner)

    def denorm(self, x):
        return x * self.ranger + self.miner

    def norm(self, x):
        return (x - self.miner) / self.ranger

# ============================================================================
# LOAD TRAINING DATA
# ============================================================================
def load_ML_data(Trainpath, only_VP=None):

    #sanity_check_bundle(Path(f'{Trainpath}.tar.gz')) # Good check. Cost time, so we skip for now, 

    print(f"Loading training data from {Trainpath}")
    train_data = load_ml_bundle(f'{Trainpath}.tar.gz')

    ml_indexer = train_data.ml_indexer
    featureMap = train_data.features
    binaryMap = train_data.binary_labels
    labelMap = train_data.labels
    moleMap = train_data.molar_labels
    has_free_outputs = getattr(train_data, 'freeOutputs', None) is not None

    # Extract indexer components for easier access
    label_indices = ml_indexer.label_indices
    label_indices_comp = ml_indexer.label_indices_comp
    compositionally_variable_phases = ml_indexer.compositionally_variable_phases
    mass_phasedict = ml_indexer.mass_phasedict
    compositional_component_subset = ml_indexer.compositional_component_subset
    compToOx = ml_indexer.compToOx
    PxSpTransform = ml_indexer.PxSpTransform
    oxToEl = ml_indexer.OxToEl
    elToOx = ml_indexer.ElToOx
    MM = ml_indexer.MM
    Elkeys = ml_indexer.Elkeys

    print(f"Feature Shape: {featureMap.shape}")
    print(f"Binary Shape: {binaryMap.shape}")
    print(f"Label Shape: {labelMap.shape}")
    print(f"Mole Shape: {moleMap.shape}")
    if has_free_outputs:
        print(f"Free Outputs Shape: {train_data.freeOutputs.shape}")

    # ============================================================================
    # Apply only_VP restriction if specified
    # ============================================================================
    if only_VP is not None:
        print(f"\nRestricting to phases: {only_VP}")
        ml_indexer.restrictVC(only_VP)
              # Rebuild indexer components after restriction
        label_indices = ml_indexer.label_indices
        label_indices_comp = ml_indexer.label_indices_comp
        compositionally_variable_phases = ml_indexer.compositionally_variable_phases
        compositional_component_subset = ml_indexer.compositional_component_subset
        
        # Subset labels to restricted VC components
        labelMap = labelMap[:, compositional_component_subset]
        print(f"Restricted Label Shape: {labelMap.shape}")

    # ============================================================================
    # Create feature normalizer
    # ============================================================================
    # Determine number of physical features (P, T, fO2) - rest are chemical
    n_physical_features = len(ml_indexer.featureNames) if hasattr(ml_indexer, 'featureNames') else 3
    n_total_features = featureMap.shape[1]

    # Calculate min/max for first n_physical_features only
    min_tensor = torch.zeros(n_total_features, device='cpu', dtype=torch.float)
    range_tensor = torch.ones(n_total_features, device='cpu', dtype=torch.float)

    # For physical features, calculate from training data
    if n_physical_features > 0:
        feature_tensor = torch.tensor(featureMap[:, :n_physical_features], device='cpu', dtype=torch.float)
        min_tensor[:n_physical_features] = torch.min(feature_tensor, dim=0).values
        range_tensor[:n_physical_features] = torch.max(feature_tensor, dim=0).values - min_tensor[:n_physical_features]
        
        # Avoid division by zero (Should be no zero-range columns- let the error fly.
        #range_tensor[:n_physical_features] = torch.clamp(range_tensor[:n_physical_features], min=1e-7)

    # For chemical features, use identity normalization (min=0, range=1)
    # (already set to these values above)

    # Store normalizer in ml_indexer
    ml_indexer.feature_normalizer = Normalizer(min_tensor=min_tensor, range_tensor=range_tensor)

    # ============================================================================
    # Normalize features
    # ============================================================================
    normf = ml_indexer.feature_normalizer
    Trainnormfeatures = normf.norm(torch.tensor(featureMap, device='cpu', dtype=torch.float))
    del featureMap
    gc.collect()

    Trainbinaryfeatures = torch.tensor(binaryMap, device='cpu', dtype=torch.float)
    del binaryMap
    gc.collect()

    Trainlabels = torch.tensor(labelMap, device='cpu', dtype=torch.float) @ torch.tensor(
        PxSpTransform[np.ix_(compositional_component_subset, compositional_component_subset)], dtype=torch.float
    )
    del labelMap
    gc.collect()

    Trainmoles = torch.tensor(np.log10(moleMap+molar_epsilon), device='cpu', dtype=torch.float)
    del moleMap
    gc.collect()

    if has_free_outputs:
        Trainfreeoutputs = torch.tensor(ml_indexer.freeOutputs, device='cpu', dtype=torch.float)
        
        # ============================================================================
        # Create output normalizer for free outputs
        # ============================================================================
        free_output_tensor = Trainfreeoutputs
        free_output_min = torch.min(free_output_tensor, dim=0).values
        free_output_range = torch.max(free_output_tensor, dim=0).values - free_output_min
        free_output_range = torch.clamp(free_output_range, min=1e-7)
        
        ml_indexer.output_normalizer = Normalizer(min_tensor=free_output_min, range_tensor=free_output_range)
        Trainfreeoutputs_normalized = ml_indexer.output_normalizer.norm(Trainfreeoutputs)
        

    else:
        Trainfreeoutputs = None
        Trainfreeoutputs_normalized = None

    del train_data
    gc.collect()


    # ============================================================================
    # Process in batches for validation
    # ============================================================================
    """def bulk_test_in_batches(Trainnormfeatures, Trainmoles, Trainlabels, batch_size=8192):

        # Precompute constant matrices as float32 tensors
        MM_t = torch.tensor(MM[:-1, :-1], dtype=torch.float32)
        compToOx_t = torch.tensor(compToOx, dtype=torch.float32)
        oxToEl_full_t = torch.tensor(oxToEl, dtype=torch.float32)

        n_samples = Trainnormfeatures.size(0)

        bulk_wt_ox_chunks = []
        GTReconBulk_chunks = []

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)

            # === Bulk weights ===
            bulk_wt_ox = (
                (Trainnormfeatures[start:end, len(ml_indexer.featureNames):] @ elToOx) @ MM_t
            )
            bulk_wt_ox = 100 * bulk_wt_ox / torch.sum(bulk_wt_ox, axis=1).reshape(-1, 1)
            bulk_wt_ox_chunks.append(bulk_wt_ox)

            # === Ground truth compositions ===
            GT_comps = torch.zeros(
                (end - start, ml_indexer.ncomps),
                dtype=torch.float32,
            )

            for phase in np.array(list(label_indices.keys())):
                moles = torch.tensor(
                    Trainmoles[start:end, ml_indexer.mass_phasedict[phase]].reshape(-1, 1),
                    dtype=torch.float32,
                )
                if phase in compositionally_variable_phases:
                    GT_comps[:, label_indices[phase]] = (
                        moles * Trainlabels[start:end, label_indices_comp[phase]].to(torch.float32)
                    )
                else:
                    GT_comps[:, label_indices[phase]] = moles

            # === Recon bulk oxides ===
            GTReconBulk_oxides = (
                ((GT_comps @ compToOx_t) @ oxToEl_full_t) @ elToOx
            ) @ MM_t
            GTReconBulk_oxides *= 100 / torch.sum(GTReconBulk_oxides, axis=1, keepdims=True)

            GTReconBulk_chunks.append(GTReconBulk_oxides)
            print(bulk_wt_ox)
            print(GTReconBulk_oxides)

        # Recombine all batches
        bulk_wt_ox = torch.cat(bulk_wt_ox_chunks, dim=0)
        GTReconBulk_oxides = torch.cat(GTReconBulk_chunks, dim=0)

        # === Compare rounded results ===
        train_mismatches = torch.unique(
            torch.where(
                torch.round(bulk_wt_ox, decimals=2) != torch.round(GTReconBulk_oxides, decimals=2)
            )[0]
        )

        return train_mismatches


    train_mismatches = bulk_test_in_batches(Trainnormfeatures, Trainmoles, Trainlabels, batch_size=2**13)


    print(f'Train mismatches: {train_mismatches.size()[0]}')"""

    OOB = ((Trainlabels > 1).to(float) + (Trainlabels < 0).to(float)).to(bool)
    badMap = torch.unique(torch.where(OOB)[0])
    goodMap = torch.ones(Trainlabels.size()[0]).to(torch.bool)
    goodMap[badMap] = False
    #goodMap[train_mismatches] = False

    print(f"Train Features: {Trainnormfeatures.size()}, Binaries {Trainbinaryfeatures.size()}, labels: {Trainlabels.size()}")
    Trainnormfeatures = Trainnormfeatures[goodMap]
    Trainbinaryfeatures = Trainbinaryfeatures[goodMap]
    Trainlabels = Trainlabels[goodMap]
    Trainmoles = Trainmoles[goodMap]
    if Trainfreeoutputs is not None:
        Trainfreeoutputs = Trainfreeoutputs[goodMap]
    print(f"Train Features: {Trainnormfeatures.size()}, Binaries {Trainbinaryfeatures.size()}, labels: {Trainlabels.size()}")

    # Create dataset objects based on whether free outputs exist
    if has_free_outputs:
        full_train_set = TensorDatasetFive(
            features=Trainnormfeatures,
            binarylabels=Trainbinaryfeatures,
            labels=Trainlabels,
            molelabels=Trainmoles,
            freeoutputs=Trainfreeoutputs_normalized,
        )
    else:
        full_train_set = TensorDatasetFour(
            features=Trainnormfeatures,
            binarylabels=Trainbinaryfeatures,
            labels=Trainlabels,
            molelabels=Trainmoles,
        )                       
    return full_train_set, ml_indexer