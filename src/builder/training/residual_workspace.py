"""Cached, memory-mapped training workspace for train_temperature_residual_fcnn.py.

Mirrors the pattern in `builder.training.dataset_workspace` (used by the main
episode-orchestration pipeline, see main.py/loadTrainData.py): a bundle's
ready-to-train arrays are built once in chunked passes over memmapped source
arrays, cached to disk under `workspace_dir`, and reused on a later call
against the same bundle+params (fingerprint-gated). This keeps peak RAM at
O(chunk_size) while building the workspace, however large the source bundle,
rather than the O(dataset) it costs to extract+concatenate everything in RAM
up front.

Per split (train/test/valid), the cached workspace holds:
  features_norm   (n, F)  float32 - normalized features, post row-filtering
  pred_molar      (n, M)  float32 - emulator-predicted phase moles
  pred_intensive  (n, C)  float32 - emulator-predicted chem/intensive output
  molar_labels    (n, M)  float32 - ground-truth phase moles (same filtering)
  labels          (n, C)  float32 - ground-truth chem/intensive labels
  y_norm          (n, 1)  float32 - normalized T residual target
  residual        (n,)    float32 - raw (unnormalized) T residual, for diagnostics
  T_true          (n,)    float32 - raw true temperature
  T_ref           (n,)    float32 - reference-adiabat temperature
  normalizer_state.npz    - feature_min/range, output_min/range actually used

Model 1's input is [features_norm | pred_molar | pred_intensive]; model 2's is
[features_norm | molar_labels | labels]. Neither concatenation is ever
materialized for the full dataset here - `ResidualChunkedLoader` builds it one
chunk at a time, and the RAM-fit path in the caller concatenates the full
(already small) arrays only when they comfortably fit in memory.
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
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from ngibbs.utils.file_utils import chunked_mask_copy
from ngibbs.utils.math_utils import Normalizer

from builder.training.adiabat_utils import apply_normalizer, compute_residuals

WORKSPACE_DIR_DEFAULT = Path(__file__).parent / "residual_workspace"
_FINGERPRINT_FILE = "_fingerprint.json"


def _fingerprint_for(bundle_path: Path, emulator_model_path: Path, *, temperature_label_idx: int,
                      adiabat_coefs, adiabat_elem_norm: float,
                      s_min, s_max, max_abs_residual) -> Dict:
    bstat = bundle_path.stat()
    estat = Path(emulator_model_path).stat()
    return {
        "bundle_path": str(bundle_path.resolve()),
        "bundle_mtime": bstat.st_mtime,
        "bundle_size": bstat.st_size,
        "emulator_model_path": str(Path(emulator_model_path).resolve()),
        "emulator_model_mtime": estat.st_mtime,
        "emulator_model_size": estat.st_size,
        "temperature_label_idx": temperature_label_idx,
        "adiabat_coefs": adiabat_coefs,
        "adiabat_elem_norm": adiabat_elem_norm,
        "s_min": s_min,
        "s_max": s_max,
        "max_abs_residual": max_abs_residual,
    }


class ResidualWorkspaceHandle:
    """Open memmaps + metadata for one split's cached residual-training workspace."""

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = Path(workspace_dir)
        self.features_norm = np.load(self.workspace_dir / "features_norm.npy", mmap_mode="r")
        self.pred_molar = np.load(self.workspace_dir / "pred_molar.npy", mmap_mode="r")
        self.pred_intensive = np.load(self.workspace_dir / "pred_intensive.npy", mmap_mode="r")
        self.molar_labels = np.load(self.workspace_dir / "molar_labels.npy", mmap_mode="r")
        self.labels = np.load(self.workspace_dir / "labels.npy", mmap_mode="r")
        self.y_norm = np.load(self.workspace_dir / "y_norm.npy", mmap_mode="r")
        self.residual = np.load(self.workspace_dir / "residual.npy", mmap_mode="r")
        self.T_true = np.load(self.workspace_dir / "T_true.npy", mmap_mode="r")
        self.T_ref = np.load(self.workspace_dir / "T_ref.npy", mmap_mode="r")
        self.n_rows = self.features_norm.shape[0]

    def secondary_arrays(self, kind: str) -> Tuple[np.ndarray, np.ndarray]:
        if kind == "emulator":
            return self.pred_molar, self.pred_intensive
        if kind == "gt":
            return self.molar_labels, self.labels
        raise ValueError(f"Unknown kind: {kind!r} (expected 'emulator' or 'gt')")

    def x_dim(self, kind: str) -> int:
        a, b = self.secondary_arrays(kind)
        return self.features_norm.shape[1] + a.shape[1] + b.shape[1]

    def total_bytes(self, kind: Optional[str] = None) -> int:
        """Bytes needed to fully materialize this split's arrays.

        With `kind=None` (the default), sums every array the workspace holds -
        i.e. what it would cost to materialize BOTH model-1 and model-2 inputs
        at once, matching how the caller currently builds both loaders back to
        back from the same split. Pass a specific kind to size only that one.
        """
        base = self.features_norm.nbytes + self.y_norm.nbytes
        if kind is None:
            extra = (self.pred_molar.nbytes + self.pred_intensive.nbytes
                     + self.molar_labels.nbytes + self.labels.nbytes)
        else:
            a, b = self.secondary_arrays(kind)
            extra = a.nbytes + b.nbytes
        return base + extra


def get_or_build_residual_workspace(
    bundle_path,
    workspace_root,
    *,
    split_label: str,
    n_named_features: int,
    p_idx: int,
    s_idx: int,
    temperature_label_idx: int,
    adiabat_coefs: Optional[Dict[str, float]],
    comp_indices: Optional[Dict[str, object]],
    adiabat_elem_norm: float,
    s_min: Optional[float],
    s_max: Optional[float],
    max_abs_residual: Optional[float],
    emulator,
    emulator_model_path,
    device: torch.device,
    emulator_batch_size: int,
    feature_normalizer: Optional[Normalizer],
    output_normalizer: Optional[Normalizer],
    fit_normalizers: bool,
    chunk_size: int = 1_000_000,
) -> Tuple[ResidualWorkspaceHandle, Normalizer, Normalizer]:
    """Return (workspace_handle, feature_normalizer, output_normalizer) for `bundle_path`.

    `fit_normalizers=True` (the train split) fits fresh normalizers from this
    split's own (filtered) data - reusing `feature_normalizer` if one is passed
    in (mirrors the "existing ml_indexer.feature_normalizer wins" rule the
    unchunked script used) - and returns them for the caller to pass into the
    test/valid calls with `fit_normalizers=False`, which apply them unchanged.

    Cache is keyed by a fingerprint (bundle identity, emulator identity, and
    every filtering/adiabat parameter that affects the output arrays) under
    `workspace_root/<split_label>/`; a later call with the same fingerprint
    reuses the cached arrays without recomputing anything, including the
    normalizer parameters actually used (loaded back from disk either way, so
    a cache hit still returns real Normalizer objects).
    """
    bundle_path = Path(bundle_path)
    if not str(bundle_path).endswith(".tar.gz"):
        bundle_path = Path(str(bundle_path) + ".tar.gz")
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    workspace_dir = Path(workspace_root) / split_label
    fingerprint = _fingerprint_for(
        bundle_path, emulator_model_path,
        temperature_label_idx=temperature_label_idx,
        adiabat_coefs=adiabat_coefs, adiabat_elem_norm=adiabat_elem_norm,
        s_min=s_min, s_max=s_max, max_abs_residual=max_abs_residual,
    )
    fp_path = workspace_dir / _FINGERPRINT_FILE

    reuse = False
    if fp_path.exists():
        try:
            reuse = json.loads(fp_path.read_text()) == fingerprint
        except (json.JSONDecodeError, OSError):
            reuse = False

    if reuse:
        print(f"[residual_workspace] Reusing cached '{split_label}' workspace at {workspace_dir}")
    else:
        print(f"[residual_workspace] Building fresh '{split_label}' workspace at {workspace_dir} "
              f"for {bundle_path}")
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        _build_residual_workspace(
            bundle_path, workspace_dir,
            n_named_features=n_named_features,
            p_idx=p_idx, s_idx=s_idx, temperature_label_idx=temperature_label_idx,
            adiabat_coefs=adiabat_coefs, comp_indices=comp_indices,
            s_min=s_min, s_max=s_max, max_abs_residual=max_abs_residual,
            emulator=emulator, device=device, emulator_batch_size=emulator_batch_size,
            feature_normalizer=feature_normalizer, output_normalizer=output_normalizer,
            fit_normalizers=fit_normalizers, chunk_size=chunk_size,
        )
        fp_path.write_text(json.dumps(fingerprint, indent=2))

    state = np.load(workspace_dir / "normalizer_state.npz")
    feature_normalizer = Normalizer(
        min_tensor=torch.tensor(state["feature_min"], dtype=torch.float32),
        range_tensor=torch.tensor(state["feature_range"], dtype=torch.float32),
    )
    output_normalizer = Normalizer(
        min_tensor=torch.tensor(state["output_min"], dtype=torch.float32),
        range_tensor=torch.tensor(state["output_range"], dtype=torch.float32),
    )
    handle = ResidualWorkspaceHandle(workspace_dir)
    return handle, feature_normalizer, output_normalizer


def _fit_feature_normalizer_chunked(features, n_named_features: int,
                                     existing: Optional[Normalizer], chunk_size: int) -> Normalizer:
    n_features_total = features.shape[1]
    if existing is not None:
        min_tensor = existing.miner.detach().cpu().to(torch.float32).clone()
        range_tensor = existing.ranger.detach().cpu().to(torch.float32).clone()
        if min_tensor.numel() != n_features_total or range_tensor.numel() != n_features_total:
            raise ValueError(
                f"Existing feature_normalizer dimension mismatch: "
                f"expected {n_features_total}, got min={min_tensor.numel()} range={range_tensor.numel()}"
            )
        if n_named_features < n_features_total:
            min_tensor[n_named_features:] = 0.0
            range_tensor[n_named_features:] = 1.0
        return Normalizer(min_tensor=min_tensor, range_tensor=range_tensor)

    min_tensor = torch.zeros(n_features_total, dtype=torch.float32)
    range_tensor = torch.ones(n_features_total, dtype=torch.float32)
    if n_named_features > 0:
        n_rows = features.shape[0]
        mins = maxs = None
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            chunk = np.asarray(features[start:end, :n_named_features], dtype=np.float32)
            chunk_min = chunk.min(axis=0)
            chunk_max = chunk.max(axis=0)
            mins = chunk_min if mins is None else np.minimum(mins, chunk_min)
            maxs = chunk_max if maxs is None else np.maximum(maxs, chunk_max)
        min_tensor[:n_named_features] = torch.tensor(mins, dtype=torch.float32)
        range_tensor[:n_named_features] = torch.clamp(
            torch.tensor(maxs, dtype=torch.float32) - torch.tensor(mins, dtype=torch.float32), min=1e-7)
    return Normalizer(min_tensor=min_tensor, range_tensor=range_tensor)


def _fit_output_normalizer_chunked(residual, chunk_size: int) -> Normalizer:
    n = residual.shape[0]
    y_min = y_max = None
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = np.asarray(residual[start:end], dtype=np.float32)
        cmin = float(chunk.min())
        cmax = float(chunk.max())
        y_min = cmin if y_min is None else min(y_min, cmin)
        y_max = cmax if y_max is None else max(y_max, cmax)
    y_range = 1.0 if (y_max - y_min) < 1e-7 else (y_max - y_min)
    return Normalizer(
        min_tensor=torch.tensor([y_min], dtype=torch.float32),
        range_tensor=torch.tensor([y_range], dtype=torch.float32),
    )


def _build_residual_workspace(
    bundle_path: Path, workspace_dir: Path, *, n_named_features: int, p_idx: int, s_idx: int,
    temperature_label_idx: int, adiabat_coefs, comp_indices,
    s_min, s_max, max_abs_residual, emulator, device, emulator_batch_size: int,
    feature_normalizer: Optional[Normalizer], output_normalizer: Optional[Normalizer],
    fit_normalizers: bool, chunk_size: int,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    tmp_base = repo_root / "data" / "tmp"
    tmp_base.mkdir(parents=True, exist_ok=True)
    extract_dir = Path(tempfile.mkdtemp(dir=tmp_base))
    try:
        print(f"[residual_workspace] Extracting {bundle_path} ...")
        with tarfile.open(bundle_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)

        raw_features = np.load(extract_dir / "features.npy", mmap_mode="r")
        raw_free_outputs = np.load(extract_dir / "free_outputs.npy", mmap_mode="r")
        raw_molar = np.load(extract_dir / "molar_labels.npy", mmap_mode="r")
        raw_labels = np.load(extract_dir / "labels.npy", mmap_mode="r")
        n_rows, n_feat_cols = raw_features.shape

        # --- Pass 1/3: residual + keep_mask (entropy range + residual magnitude), chunked.
        tmp_residual_path = workspace_dir / "_residual_full.npy"
        tmp_tref_path = workspace_dir / "_tref_full.npy"
        tmp_ttrue_path = workspace_dir / "_ttrue_full.npy"
        tmp_residual = np.lib.format.open_memmap(tmp_residual_path, mode="w+", dtype=np.float32, shape=(n_rows,))
        tmp_tref = np.lib.format.open_memmap(tmp_tref_path, mode="w+", dtype=np.float32, shape=(n_rows,))
        tmp_ttrue = np.lib.format.open_memmap(tmp_ttrue_path, mode="w+", dtype=np.float32, shape=(n_rows,))
        keep_mask = np.ones(n_rows, dtype=bool)

        print(f"[residual_workspace] Pass 1/3: computing residuals over {n_rows:,} rows...")
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            feats_chunk = np.array(raw_features[start:end], dtype=np.float32)
            T_true_chunk = np.array(raw_free_outputs[start:end, temperature_label_idx], dtype=np.float32)
            residual_chunk, T_ref_chunk = compute_residuals(
                T_true_chunk, feats_chunk, p_idx, s_idx,
                adiabat_coefs=adiabat_coefs, comp_indices=comp_indices,
            )
            tmp_residual[start:end] = residual_chunk
            tmp_tref[start:end] = T_ref_chunk
            tmp_ttrue[start:end] = T_true_chunk

            mask_chunk = np.ones(end - start, dtype=bool)
            if s_min is not None or s_max is not None:
                S = feats_chunk[:, s_idx]
                if s_min is not None:
                    mask_chunk &= S > s_min
                if s_max is not None:
                    mask_chunk &= S < s_max
            if max_abs_residual is not None:
                mask_chunk &= np.abs(residual_chunk) <= max_abs_residual
            keep_mask[start:end] = mask_chunk
        tmp_residual.flush(); tmp_tref.flush(); tmp_ttrue.flush()
        del tmp_residual, tmp_tref, tmp_ttrue
        gc.collect()

        n_keep = int(keep_mask.sum())
        n_removed = n_rows - n_keep
        if n_removed:
            print(f"[residual_workspace] Dropping {n_removed:,}/{n_rows:,} rows "
                  f"(entropy-range / max-abs-residual filters); keeping {n_keep:,}")
        if n_keep == 0:
            raise ValueError(f"All rows filtered out of {bundle_path} - check --s-min/--s-max/--max-abs-residual")

        # --- Compact everything to the surviving rows.
        tmp_residual_ro = np.load(tmp_residual_path, mmap_mode="r")
        tmp_tref_ro = np.load(tmp_tref_path, mmap_mode="r")
        tmp_ttrue_ro = np.load(tmp_ttrue_path, mmap_mode="r")
        chunked_mask_copy(tmp_residual_ro, workspace_dir / "residual.npy", keep_mask, chunk_size)
        chunked_mask_copy(tmp_tref_ro, workspace_dir / "T_ref.npy", keep_mask, chunk_size)
        chunked_mask_copy(tmp_ttrue_ro, workspace_dir / "T_true.npy", keep_mask, chunk_size)
        chunked_mask_copy(raw_features, workspace_dir / "_features_raw.npy", keep_mask, chunk_size)
        chunked_mask_copy(raw_molar, workspace_dir / "molar_labels.npy", keep_mask, chunk_size)
        chunked_mask_copy(raw_labels, workspace_dir / "labels.npy", keep_mask, chunk_size)
        del tmp_residual_ro, tmp_tref_ro, tmp_ttrue_ro
        gc.collect()
        tmp_residual_path.unlink()
        tmp_tref_path.unlink()
        tmp_ttrue_path.unlink()

        features_raw_filtered = np.load(workspace_dir / "_features_raw.npy", mmap_mode="r")
        residual_filtered = np.load(workspace_dir / "residual.npy", mmap_mode="r")

        # --- Normalizers: fit fresh from this (filtered) split, or apply the given ones.
        if fit_normalizers:
            feature_normalizer = _fit_feature_normalizer_chunked(
                features_raw_filtered, n_named_features, feature_normalizer, chunk_size)
            output_normalizer = _fit_output_normalizer_chunked(residual_filtered, chunk_size)
        if feature_normalizer is None or output_normalizer is None:
            raise ValueError("feature_normalizer/output_normalizer must be provided when fit_normalizers=False")

        # --- Pass 2/3: normalized features + normalized residual target, chunked.
        features_norm = np.lib.format.open_memmap(
            workspace_dir / "features_norm.npy", mode="w+", dtype=np.float32, shape=(n_keep, n_feat_cols))
        y_norm = np.lib.format.open_memmap(
            workspace_dir / "y_norm.npy", mode="w+", dtype=np.float32, shape=(n_keep, 1))
        print("[residual_workspace] Pass 2/3: normalizing features + residual target...")
        for start in range(0, n_keep, chunk_size):
            end = min(start + chunk_size, n_keep)
            feats_chunk = np.array(features_raw_filtered[start:end], dtype=np.float32)
            features_norm[start:end] = apply_normalizer(feats_chunk, feature_normalizer)
            y_norm[start:end] = apply_normalizer(
                np.asarray(residual_filtered[start:end], dtype=np.float32).reshape(-1, 1), output_normalizer)
        features_norm.flush(); y_norm.flush()
        del features_norm, y_norm
        gc.collect()

        # --- Pass 3/3: emulator forward pass, chunked, writing straight to memmap
        # (never accumulated as a python list of full-size batches in RAM).
        print(f"[residual_workspace] Pass 3/3: running emulator forward pass over {n_keep:,} rows...")
        pred_molar_out = None
        pred_intensive_out = None
        with torch.no_grad():
            for start in range(0, n_keep, emulator_batch_size):
                end = min(start + emulator_batch_size, n_keep)
                feats_t = torch.tensor(
                    np.array(features_raw_filtered[start:end], dtype=np.float32), device=device)
                predicted = emulator.forwardMB(
                    feats_t, Normalize=True, WtPercent=False, outputs=["phase_moles", "chem_out"])
                pm = predicted["phase_moles"].detach().cpu().clamp(0, 1).numpy().astype(np.float32)
                co = predicted["chem_out"].detach().cpu().clamp(0, 1).numpy().astype(np.float32)
                if pred_molar_out is None:
                    pred_molar_out = np.lib.format.open_memmap(
                        workspace_dir / "pred_molar.npy", mode="w+", dtype=np.float32,
                        shape=(n_keep, pm.shape[1]))
                    pred_intensive_out = np.lib.format.open_memmap(
                        workspace_dir / "pred_intensive.npy", mode="w+", dtype=np.float32,
                        shape=(n_keep, co.shape[1]))
                pred_molar_out[start:end] = pm
                pred_intensive_out[start:end] = co
        pred_molar_out.flush(); pred_intensive_out.flush()
        del pred_molar_out, pred_intensive_out
        gc.collect()

        del features_raw_filtered, residual_filtered, raw_features, raw_free_outputs, raw_molar, raw_labels
        gc.collect()
        (workspace_dir / "_features_raw.npy").unlink()

        np.savez(
            workspace_dir / "normalizer_state.npz",
            feature_min=feature_normalizer.miner.detach().cpu().numpy().astype(np.float32),
            feature_range=feature_normalizer.ranger.detach().cpu().numpy().astype(np.float32),
            output_min=output_normalizer.miner.detach().cpu().numpy().astype(np.float32),
            output_range=output_normalizer.ranger.detach().cpu().numpy().astype(np.float32),
        )
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


class ResidualChunkedLoader:
    """Iterates shuffled minibatches of (x, y) for one model's input, built one
    chunk at a time from a ResidualWorkspaceHandle's memmapped arrays.

    Modeled directly on dataset_workspace.ChunkedMemmapTrainLoader (same
    random-offset chunk boundaries, single in-chunk permutation, background-
    thread prefetch of the next chunk) but yields a 2-tuple (x, y) with x
    already concatenated from [features_norm | secondary_a | secondary_b] for
    the given `kind` ("emulator" or "gt"), since that's the batch shape
    train_temperature_residual_fcnn.py's `_train_model` loop expects - unlike
    the main pipeline's loader, which yields the raw named arrays separately.

    `chunk_rows` is capped to the split's row count, so a small dataset
    collapses to exactly one chunk per epoch - equivalent to full in-RAM
    shuffling, just re-copied out of the memmap once per epoch instead of
    once total. For datasets that fit comfortably in RAM, prefer materializing
    once and using a plain DataLoader (see _make_loader_for_split in the
    training script); this loader exists for the case that no longer fits.
    """

    def __init__(self, workspace: ResidualWorkspaceHandle, kind: str, batch_size: int,
                 chunk_rows: int = 1_000_000, pin_memory: bool = True, seed: Optional[int] = None):
        self.workspace = workspace
        self.kind = kind
        self.batch_size = batch_size
        self.chunk_rows = min(chunk_rows, workspace.n_rows)
        self.pin_memory = pin_memory
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return -(-self.workspace.n_rows // self.batch_size)  # ceil division; see class docstring

    def _chunk_bounds(self):
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

    def _read_chunk(self, start: int, end: int):
        feats = np.array(self.workspace.features_norm[start:end])
        a, b = self.workspace.secondary_arrays(self.kind)
        extra_a = np.array(a[start:end])
        extra_b = np.array(b[start:end])
        x = np.concatenate([feats, extra_a, extra_b], axis=1).astype(np.float32)
        y = np.array(self.workspace.y_norm[start:end], dtype=np.float32)
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)
        if self.pin_memory:
            x_t = x_t.pin_memory()
            y_t = y_t.pin_memory()
        return x_t, y_t

    def _iterate_chunk_batches(self, chunk):
        x_t, y_t = chunk
        n = x_t.shape[0]
        perm = torch.randperm(n)
        for start in range(0, n, self.batch_size):
            idx = perm[start:min(start + self.batch_size, n)]
            yield x_t[idx], y_t[idx]

    def __iter__(self):
        bounds = self._chunk_bounds()
        self._rng.shuffle(bounds)

        q: "queue.Queue" = queue.Queue(maxsize=1)
        stop_event = threading.Event()

        def _producer():
            try:
                for start, end in bounds:
                    if stop_event.is_set():
                        return
                    q.put(self._read_chunk(start, end))
                q.put(None)
            except Exception as exc:
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
