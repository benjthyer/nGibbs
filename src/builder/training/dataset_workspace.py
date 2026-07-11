"""
Cached, transformed, memory-mapped training-data workspace.

Building a Train bundle's ready-to-train arrays (PxSp-transformed labels,
normalized features, out-of-bounds rows removed) is a one-time cost that this
module amortizes across every subsequent training run against the same
bundle+transform-params: the result is cached under `workspace_dir` (default:
a `dataset_workspace/` folder next to this file, mirroring the existing
`TEMP_MODELS_DIR` convention already used by `trainer.py`/`tuners.py`), keyed
by a small fingerprint file. A later call with the same bundle+params reuses
the cache directly; a call against a different bundle, or the same bundle
with different `only_VP`/`molar_epsilon`, wipes and rebuilds it - this holds
exactly one dataset's worth of cached workspace at a time, by design.

Also provides `ChunkedMemmapTrainLoader`: an async-prefetching, chunk-shuffled
iterator over a workspace's arrays, for training sets too large to fit fully
in RAM. Since the workspace's arrays are already normalized/transformed/
filtered, this loader does no per-row transform work at all - only chunked
reads, a random per-epoch chunk offset/order, an in-RAM shuffle within each
chunk, and a background-thread prefetch of the next chunk.
"""
from __future__ import annotations

import gc
import json
import queue
import shutil
import tarfile
import tempfile
import threading
from pathlib import Path

import numpy as np
import torch

WORKSPACE_DIR_DEFAULT = Path(__file__).parent / "dataset_workspace"
_FINGERPRINT_FILE = "_fingerprint.json"
_ARRAY_NAMES = ("features", "binary_labels", "labels", "molar_labels")


class WorkspaceHandle:
    """Open memmaps + metadata for a built training workspace."""

    def __init__(self, workspace_dir, arrays, ml_indexer, n_rows):
        self.workspace_dir = workspace_dir
        self.arrays = arrays  # name -> memmap
        self.ml_indexer = ml_indexer
        self.n_rows = n_rows

    @property
    def features(self):
        return self.arrays['features']

    @property
    def binary_labels(self):
        return self.arrays['binary_labels']

    @property
    def labels(self):
        return self.arrays['labels']

    @property
    def molar_labels(self):
        return self.arrays['molar_labels']

    def total_bytes(self):
        """Total bytes across the 4 workspace arrays - free to compute (shape
        x dtype size from the memmap headers), no data read."""
        return sum(arr.nbytes for arr in self.arrays.values())


def _fingerprint_for(bundle_path, only_VP, molar_epsilon):
    stat = bundle_path.stat()
    return {
        "bundle_path": str(bundle_path.resolve()),
        "bundle_mtime": stat.st_mtime,
        "bundle_size": stat.st_size,
        "only_VP": list(only_VP) if only_VP is not None else None,
        "molar_epsilon": molar_epsilon,
    }


def get_or_build_train_workspace(bundle_path, only_VP=None, molar_epsilon=0,
                                  workspace_dir=None, chunk_size=1_000_000):
    """Return a `WorkspaceHandle` for `bundle_path`, building (or rebuilding)
    the cached workspace first if it's missing or stale for the given
    bundle/only_VP/molar_epsilon combination."""
    bundle_path = Path(bundle_path)
    if not str(bundle_path).endswith('.tar.gz'):
        bundle_path = Path(str(bundle_path) + '.tar.gz')
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    workspace_dir = Path(workspace_dir) if workspace_dir is not None else WORKSPACE_DIR_DEFAULT
    fingerprint = _fingerprint_for(bundle_path, only_VP, molar_epsilon)
    fp_path = workspace_dir / _FINGERPRINT_FILE

    reuse = False
    if fp_path.exists():
        try:
            reuse = json.loads(fp_path.read_text()) == fingerprint
        except (json.JSONDecodeError, OSError):
            reuse = False

    if reuse:
        print(f"[dataset_workspace] Reusing cached workspace at {workspace_dir}")
    else:
        print(f"[dataset_workspace] Building fresh workspace at {workspace_dir} for {bundle_path}")
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        _build_workspace(bundle_path, workspace_dir, only_VP, molar_epsilon, chunk_size)
        fp_path.write_text(json.dumps(fingerprint, indent=2))

    from ngibbs.config.ml_indexer import load_ml_indexer_from_state
    ml_indexer = load_ml_indexer_from_state(str(workspace_dir / 'ml_indexer'))

    arrays = {name: np.load(workspace_dir / f"{name}.npy", mmap_mode='r') for name in _ARRAY_NAMES}
    n_rows = arrays['features'].shape[0]
    return WorkspaceHandle(workspace_dir, arrays, ml_indexer, n_rows)


def _build_workspace(bundle_path, workspace_dir, only_VP, molar_epsilon, chunk_size):
    from ngibbs.config.ml_indexer import load_ml_indexer_from_state
    from ngibbs.utils.file_utils import chunked_mask_copy
    from ngibbs.utils.math_utils import Normalizer

    repo_root = Path(__file__).resolve().parents[3]
    tmp_base = repo_root / "data" / "tmp"
    tmp_base.mkdir(parents=True, exist_ok=True)
    extract_dir = Path(tempfile.mkdtemp(dir=tmp_base))
    try:
        print(f"[dataset_workspace] Extracting {bundle_path} ...")
        with tarfile.open(bundle_path, 'r:gz') as tar:
            tar.extractall(path=extract_dir)

        ml_indexer = load_ml_indexer_from_state(str(extract_dir / 'ml_indexer'))
        if only_VP is not None:
            print(f"[dataset_workspace] Restricting to phases: {only_VP}")
            ml_indexer.restrictVC(only_VP)
        ml_indexer.molar_epsilon = molar_epsilon

        # compositional_component_subset defaults to every label column when no
        # restriction has been applied, so subsetting by it (then transforming by
        # PxSpTransform's matching sub-block) is correct whether or not only_VP was
        # given - matching loadTrainData.load_ML_data's two (previously separate)
        # code paths with a single unconditional one.
        compositional_component_subset = ml_indexer.compositional_component_subset
        PxSp_sub = np.asarray(
            ml_indexer.PxSpTransform[np.ix_(compositional_component_subset, compositional_component_subset)],
            dtype=np.float32,
        )

        bounds_path = extract_dir / 'feature_bounds.json'
        if not bounds_path.exists():
            raise FileNotFoundError(
                f"Bundle {bundle_path} has no feature_bounds.json - rebuild it with the "
                "current processing pipeline (generate_dataset_stats writes this file "
                "alongside stats.txt)."
            )
        bounds = json.loads(bounds_path.read_text())
        n_physical = len(bounds['featureNames'])

        raw_features = np.load(extract_dir / 'features.npy', mmap_mode='r')
        raw_binary = np.load(extract_dir / 'binary_labels.npy', mmap_mode='r')
        raw_labels = np.load(extract_dir / 'labels.npy', mmap_mode='r')
        raw_molar = np.load(extract_dir / 'molar_labels.npy', mmap_mode='r')

        n_rows, n_feat_cols = raw_features.shape
        n_label_cols = len(compositional_component_subset)

        # Same Normalizer construction as load_ML_data (zeros/ones defaults, only the
        # physical feature columns overwritten with real bounds) - but from the
        # bundle's precomputed feature_bounds.json rather than a fresh min/max scan.
        min_arr = np.zeros(n_feat_cols, dtype=np.float32)
        range_arr = np.ones(n_feat_cols, dtype=np.float32)
        min_arr[:n_physical] = np.asarray(bounds['min'], dtype=np.float32)
        range_arr[:n_physical] = np.asarray(bounds['max'], dtype=np.float32) - min_arr[:n_physical]
        normf = Normalizer(min_tensor=torch.tensor(min_arr), range_tensor=torch.tensor(range_arr))
        ml_indexer.feature_normalizer = normf  # round-trips via ml_indexer.save() below

        # --- Pass 1 (chunked): normalize features, PxSp-transform labels, and
        # accumulate the out-of-bounds keep_mask. Writes full-length temp arrays -
        # never more than chunk_size rows' worth of transform work in flight at once.
        tmp_features_path = workspace_dir / '_features_full.npy'
        tmp_labels_path = workspace_dir / '_labels_full.npy'
        tmp_features = np.lib.format.open_memmap(tmp_features_path, mode='w+', dtype=np.float32, shape=(n_rows, n_feat_cols))
        tmp_labels = np.lib.format.open_memmap(tmp_labels_path, mode='w+', dtype=np.float32, shape=(n_rows, n_label_cols))
        keep_mask = np.zeros(n_rows, dtype=bool)

        print(f"[dataset_workspace] Pass 1/2: normalizing features + PxSp-transforming labels over {n_rows:,} rows...")
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            tmp_features[start:end] = normf.norm(np.array(raw_features[start:end], dtype=np.float32))

            label_chunk = np.array(raw_labels[start:end]).astype(np.float32)
            label_chunk = label_chunk @ PxSp_sub
            tmp_labels[start:end] = label_chunk
            keep_mask[start:end] = ~np.any((label_chunk > 1) | (label_chunk < 0), axis=1)

        tmp_features.flush()
        tmp_labels.flush()
        del tmp_features, tmp_labels
        gc.collect()

        n_bad = n_rows - int(keep_mask.sum())
        print(f"[dataset_workspace] {n_bad:,} / {n_rows:,} rows out-of-bounds after PxSp transform; dropping them")

        # --- Pass 2 (chunked): compact down to the surviving rows, via the same
        # shared helper BigMetaTable.delete()/split() and filters use for exactly
        # this "keep only the rows a boolean mask selects" operation.
        print(f"[dataset_workspace] Pass 2/2: compacting to {int(keep_mask.sum()):,} surviving rows...")
        tmp_features_ro = np.load(tmp_features_path, mmap_mode='r')
        tmp_labels_ro = np.load(tmp_labels_path, mmap_mode='r')
        chunked_mask_copy(tmp_features_ro, workspace_dir / 'features.npy', keep_mask, chunk_size)
        chunked_mask_copy(tmp_labels_ro, workspace_dir / 'labels.npy', keep_mask, chunk_size)
        chunked_mask_copy(raw_binary, workspace_dir / 'binary_labels.npy', keep_mask, chunk_size)

        if molar_epsilon:
            tmp_molar_path = workspace_dir / '_molar_full.npy'
            tmp_molar = np.lib.format.open_memmap(tmp_molar_path, mode='w+', dtype=np.float32, shape=raw_molar.shape)
            for start in range(0, n_rows, chunk_size):
                end = min(start + chunk_size, n_rows)
                tmp_molar[start:end] = np.log10(np.array(raw_molar[start:end], dtype=np.float32) + molar_epsilon)
            tmp_molar.flush()
            del tmp_molar
            gc.collect()
            tmp_molar_ro = np.load(tmp_molar_path, mmap_mode='r')
            chunked_mask_copy(tmp_molar_ro, workspace_dir / 'molar_labels.npy', keep_mask, chunk_size)
            del tmp_molar_ro
            gc.collect()
            tmp_molar_path.unlink()
        else:
            chunked_mask_copy(raw_molar, workspace_dir / 'molar_labels.npy', keep_mask, chunk_size)

        del tmp_features_ro, tmp_labels_ro, raw_features, raw_binary, raw_labels, raw_molar
        gc.collect()
        tmp_features_path.unlink()
        tmp_labels_path.unlink()

        # mass_labels/free_outputs are never extracted into the workspace at all -
        # nothing downstream consumes them today (see loadTrainData.py).

        ml_indexer.save(str(workspace_dir / 'ml_indexer'))
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


class ChunkedMemmapTrainLoader:
    """Iterates shuffled minibatches from a `WorkspaceHandle`'s memmapped
    arrays, prefetching the next ~chunk_rows-row chunk on a background thread
    while the current (locally shuffled) chunk is consumed.

    Chunk boundaries use a random per-epoch start offset (plus a shuffled
    chunk visit order), so a row's chunk-neighbors vary epoch to epoch - on
    top of the one-time on-disk shuffle already applied during processing
    (see BigMetaTable.shuffle_rows).

    Yields the same 4-tuple shape as TensorDatasetFour + DataLoader:
    (features, binary_labels, labels, molar_labels).
    """

    def __init__(self, workspace: WorkspaceHandle, batch_size, chunk_rows=1_000_000,
                 pin_memory=True, seed=None):
        self.workspace = workspace
        self.batch_size = batch_size
        self.chunk_rows = min(chunk_rows, workspace.n_rows)
        self.pin_memory = pin_memory
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        """Approximate batch count (for tqdm's progress display), not exact:
        each chunk yields its own trailing partial batch independently, so the
        real total is ceil(n_rows / batch_size) plus up to one extra batch per
        chunk boundary - and the number of chunk boundaries itself varies
        epoch to epoch (random start offset). Good enough for a progress bar;
        don't rely on it for anything exact."""
        return -(-self.workspace.n_rows // self.batch_size)  # ceil division

    def _chunk_bounds(self):
        """(start, end) row ranges covering every row exactly once, split at a
        random offset each call rather than always at multiples of chunk_rows."""
        n = self.workspace.n_rows
        chunk_rows = self.chunk_rows
        if n <= chunk_rows:
            return [(0, n)]
        offset = int(self._rng.integers(0, chunk_rows))
        bounds = []
        if offset > 0:
            bounds.append((0, offset))
        pos = offset
        while pos < n:
            end = min(pos + chunk_rows, n)
            bounds.append((pos, end))
            pos = end
        return bounds

    def _read_chunk(self, start, end):
        # np.array(...) (not np.asarray) forces an actual copy out of the memmap
        # into a plain in-RAM array *now*, on this background thread - otherwise
        # torch.from_numpy would share memory with the still-lazy memmap view, and
        # the real disk read would happen later on the main thread instead,
        # defeating the whole point of prefetching.
        features = torch.from_numpy(np.array(self.workspace.features[start:end]))
        binary = torch.from_numpy(np.array(self.workspace.binary_labels[start:end]))
        labels = torch.from_numpy(np.array(self.workspace.labels[start:end]))
        moles = torch.from_numpy(np.array(self.workspace.molar_labels[start:end]))
        if self.pin_memory:
            features = features.pin_memory()
            binary = binary.pin_memory()
            labels = labels.pin_memory()
            moles = moles.pin_memory()
        return features, binary, labels, moles

    def _iterate_chunk_batches(self, chunk):
        features, binary, labels, moles = chunk
        n = features.shape[0]
        perm = torch.randperm(n)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            idx = perm[start:end]
            yield features[idx], binary[idx], labels[idx], moles[idx]

    def __iter__(self):
        bounds = self._chunk_bounds()
        self._rng.shuffle(bounds)

        q = queue.Queue(maxsize=1)
        stop_event = threading.Event()

        def _producer():
            try:
                for start, end in bounds:
                    if stop_event.is_set():
                        return
                    q.put(self._read_chunk(start, end))
                q.put(None)  # sentinel: no more chunks
            except Exception as exc:  # propagate to the consumer thread below
                q.put(exc)

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()

        try:
            while True:
                chunk = q.get()
                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    raise chunk
                yield from self._iterate_chunk_batches(chunk)
        finally:
            stop_event.set()
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
            thread.join(timeout=5)
