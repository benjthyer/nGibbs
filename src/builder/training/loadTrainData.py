#time.sleep(3600*3)
import sys
import os
import gc
import tarfile
import numpy as np
import torch
from pathlib import Path

# Ensure repo root and src are on path
repo_root = str(Path(__file__).resolve().parents[3])
#print(f"Repo root: {repo_root}")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)


from builder.training.torchDataClass import TensorDatasetFour, TensorDatasetFive
from ngibbs.config.ml_indexer import MLIndexer
from recipes.settings import external_base
from ngibbs.utils.file_utils import load_ml_bundle, MLDataBundle
from ngibbs.utils.math_utils import Normalizer
from tests.unit_tests.test_processing.ML_export_tests import sanity_check_bundle


MELTSModel = '102'
CalcType = 'FxCryst'
date = 'Nov9'
use_external = True  # Is data on external drive? Path defined in nMELTS.config.settings
subset = False
molar_epsilon = 0 #1e-4 # Used for log-scaling phase molar abundances. 0 means no log scaling!

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

# ============================================================================
# LOAD TRAINING DATA
# ============================================================================
def load_ML_data(Trainpath, only_VP=None, feature_normalizer=None):

    #sanity_check_bundle(Path(f'{Trainpath}.tar.gz')) # Good check. Cost time, so we skip for now, 

    bundle_path = f'{Trainpath}.tar.gz'
    print(f"Loading training data from {Trainpath}")

    # Cheap presence check (tar member names only, no extraction) purely for the
    # diagnostic below - free_outputs is never loaded into RAM here regardless,
    # since nothing downstream consumes it yet (see note below). Previously this
    # was checked via getattr(train_data, 'free_outputs', ...) - the wrong
    # (camelCase) attribute name, so it silently always evaluated to False
    # whether or not the bundle actually had one.
    with tarfile.open(bundle_path, 'r:gz') as tar:
        has_free_outputs = 'free_outputs.npy' in tar.getnames()
    if has_free_outputs:
        print("[load_ML_data] Bundle contains free_outputs, but training does not yet "
              "consume them - not loaded. (Loading/using free_outputs during training "
              "is planned to be configurable in the future.)")

    # mass_labels is unused by every current consumer; free_outputs is likewise
    # unused for now (see above) - neither is loaded into RAM.
    train_data = load_ml_bundle(bundle_path, arrays=('features', 'binary_labels', 'labels', 'molar_labels'))

    ml_indexer = train_data.ml_indexer
    featureMap = train_data.features
    binaryMap = train_data.binary_labels
    labelMap = train_data.labels
    moleMap = train_data.molar_labels

    ml_indexer.molar_epsilon = molar_epsilon # Save

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
        print(f"Free Outputs Shape: {train_data.free_outputs.shape}")

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
    # Create (or inherit) feature normalizer
    # ============================================================================
    n_total_features = featureMap.shape[1]

    if feature_normalizer is not None:
        # Inherit bounds from an already-fit normalizer (the Train set's) rather
        # than fitting a new one from this bundle's own data - the model is
        # trained on Train-normalized inputs and expects every later input
        # (Test, inference) normalized the *same* way. Computing an independent
        # min/max per-bundle (the old behavior below, still used when no
        # normalizer is supplied - e.g. loading Train itself) would silently
        # normalize Test differently from what the model was trained on.
        got = feature_normalizer.miner.shape[0]
        if got != n_total_features:
            raise ValueError(
                f"feature_normalizer has {got} columns but this bundle's features "
                f"have {n_total_features} - Train/Test featureNames or Elkeys must match."
            )
        ml_indexer.feature_normalizer = feature_normalizer
    else:
        # Determine number of physical features (P, T, fO2) - rest are chemical
        n_physical_features = len(ml_indexer.featureNames) if hasattr(ml_indexer, 'featureNames') else 3

        # Calculate min/max for first n_physical_features only
        min_tensor = torch.zeros(n_total_features, device='cpu', dtype=torch.float)
        range_tensor = torch.ones(n_total_features, device='cpu', dtype=torch.float)

        # For physical features, calculate from this bundle's own data
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

    if molar_epsilon:
        print(f"Applying log10 transform to mole labels with epsilon={molar_epsilon}")
        Trainmoles = torch.tensor(np.log10(moleMap+molar_epsilon), device='cpu', dtype=torch.float)
    else:
        print("No log transform applied to mole labels.")
        Trainmoles = torch.tensor(moleMap, device='cpu', dtype=torch.float)
    del moleMap
    gc.collect()

    # free_outputs is never loaded (see note above), regardless of has_free_outputs -
    # kept as stable no-op values until loading/using them is made configurable.
    Trainfreeoutputs = None
    Trainfreeoutputs_normalized = None

    del train_data
    gc.collect()


    # ============================================================================
    # Process in batches for validation
    # ============================================================================
    """def bulk_test_in_batches(Trainnormfeatures, Trainmoles, Trainlabels, batch_size=8192):

        # Precompute constant matrices as float32 tensors
        MM_t = torch.tensor(
            MM[:len(ml_indexer.Elkeys), :len(ml_indexer.Elkeys)],
            dtype=torch.float32
        )
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
    print(f"Total samples: {Trainlabels.size()[0]}, Bad samples: {badMap.size()[0]}, Good samples: {goodMap.sum()}")
    print(f"Train Features: {Trainnormfeatures.size()}, Binaries {Trainbinaryfeatures.size()}, labels: {Trainlabels.size()}")
    Trainnormfeatures = Trainnormfeatures[goodMap]
    Trainbinaryfeatures = Trainbinaryfeatures[goodMap]
    Trainlabels = Trainlabels[goodMap]
    Trainmoles = Trainmoles[goodMap]
    print(f"Train Features: {Trainnormfeatures.size()}, Binaries {Trainbinaryfeatures.size()}, labels: {Trainlabels.size()}")

    # Always TensorDatasetFour for now - free_outputs is never loaded (see note
    # above), so there's nothing for TensorDatasetFive to hold even when the
    # bundle has one.
    full_train_set = TensorDatasetFour(
        features=Trainnormfeatures,
        binarylabels=Trainbinaryfeatures,
        labels=Trainlabels,
        molelabels=Trainmoles,
    )
    return full_train_set, ml_indexer


def load_ML_data_auto(Trainpath, only_VP=None, molar_epsilon=0,
                       ram_threshold_bytes=8 * 1024 ** 3, workspace_dir=None,
                       chunk_size=1_000_000, batch_size=1024, chunk_rows=1_000_000):
    """
    Load Train data via the cached, pre-transformed working directory
    (see builder.training.dataset_workspace), then decide - based on the
    workspace's actual size, not the compressed bundle size - whether to
    materialize it fully in RAM (small datasets, today's behavior) or hand
    back an async chunked loader (large datasets).

    Unlike `load_ML_data`, every per-row transform (PxSp label transform,
    feature normalization, out-of-bounds row filtering) is already baked into
    the cached workspace by `get_or_build_train_workspace`, so this function
    does no further transform work itself.

    Parameters
    ----------
    Trainpath : str
        Bundle base path (without .tar.gz), same convention as load_ML_data.
    only_VP : list, optional
        Restrict to a phase subset - see load_ML_data. Changing this from a
        previous call invalidates the cached workspace and triggers a rebuild.
    molar_epsilon : float, default 0
        Log-scaling epsilon for mole labels - see load_ML_data. Changing this
        from a previous call invalidates the cached workspace and triggers a
        rebuild.
    ram_threshold_bytes : int, default 8 GiB
        If the workspace's 4 arrays total at or below this many bytes,
        return a full in-RAM TensorDatasetFour (as load_ML_data does). Above
        it, return a ChunkedMemmapTrainLoader instead. Measured against the
        *uncompressed* workspace size, not the .tar.gz bundle size - those
        can differ by several times (gzip on this kind of data compresses
        well), and the RAM-fit decision only cares about the former.
    workspace_dir, chunk_size :
        Forwarded to get_or_build_train_workspace.
    batch_size, chunk_rows : only used for the chunked-loader path.
        Starting batch_size for the returned ChunkedMemmapTrainLoader.
        main.py's episode loop reads a (possibly different) batch_size per
        episode from its own config, so trainer.py updates
        `train_loader.batch_size` before each train_Lower_MELTS/
        train_Upper_MELTS call rather than relying on the value given here -
        this is just what a bare/direct iteration would use.

    Returns
    -------
    (dataset_or_loader, ml_indexer)
        Either a TensorDatasetFour (small dataset) or a ChunkedMemmapTrainLoader
        (large dataset) - both are valid `trainData` arguments to
        trainer.train_Lower_MELTS/train_Upper_MELTS, and both yield
        (features, binary_labels, labels, molar_labels) batches.
    """
    from builder.training.dataset_workspace import get_or_build_train_workspace

    workspace = get_or_build_train_workspace(
        Trainpath, only_VP=only_VP, molar_epsilon=molar_epsilon,
        workspace_dir=workspace_dir, chunk_size=chunk_size,
    )

    total_bytes = workspace.total_bytes()
    print(f"[load_ML_data_auto] Workspace size: {total_bytes / 1024**3:.2f} GiB "
          f"({workspace.n_rows:,} rows), threshold: {ram_threshold_bytes / 1024**3:.2f} GiB")

    if total_bytes <= ram_threshold_bytes:
        print("[load_ML_data_auto] Within RAM threshold - materializing full in-RAM TensorDatasetFour")
        full_train_set = TensorDatasetFour(
            features=torch.tensor(np.array(workspace.features), dtype=torch.float),
            binarylabels=torch.tensor(np.array(workspace.binary_labels), dtype=torch.float),
            labels=torch.tensor(np.array(workspace.labels), dtype=torch.float),
            molelabels=torch.tensor(np.array(workspace.molar_labels), dtype=torch.float),
        )
        return full_train_set, workspace.ml_indexer

    print("[load_ML_data_auto] Above RAM threshold - returning ChunkedMemmapTrainLoader")
    loader = build_chunked_train_loader(workspace, batch_size=batch_size, chunk_rows=chunk_rows)
    return loader, workspace.ml_indexer


def build_chunked_train_loader(workspace_handle, batch_size, chunk_rows=1_000_000,
                                pin_memory=True, seed=None):
    """Wrap a WorkspaceHandle (as returned internally by load_ML_data_auto for
    large datasets) in a ChunkedMemmapTrainLoader with a concrete batch_size."""
    from builder.training.dataset_workspace import ChunkedMemmapTrainLoader
    return ChunkedMemmapTrainLoader(
        workspace_handle, batch_size=batch_size, chunk_rows=chunk_rows,
        pin_memory=pin_memory, seed=seed,
    )