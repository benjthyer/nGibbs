"""Unit tests for VRAM-aware batch tuning (ngibbs.utils.batch_tuning) and the
batch-size resolution / OOM back-off in EmulatorAPI._staged_forward.

No GPU or model checkpoint needed: CUDA-only pieces are monkeypatched.

Run:  pytest tests/unit_tests/test_batch_tuning.py
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ngibbs.utils import batch_tuning as bt

FAKE_FP = "FakeGPU|8192MiB|torch0|cuda0"


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("NGIBBS_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("NGIBBS_BATCH_TUNING", raising=False)
    monkeypatch.setattr(bt, "_STATE", None)
    yield tmp_path
    monkeypatch.setattr(bt, "_STATE", None)


@pytest.fixture
def fake_cuda(monkeypatch):
    """Pretend a CUDA device exists with `avail` bytes free (mutable)."""
    state = {"avail": 4 * 2 ** 30}
    monkeypatch.setattr(bt, "_is_cuda", lambda device: True)
    monkeypatch.setattr(bt, "fingerprint", lambda device: FAKE_FP)
    monkeypatch.setattr(bt, "available_bytes", lambda device: state["avail"])
    return state


def _entry(per_row=1000.0, fixed=0.0, plateau=2 ** 16, safety=0.8):
    return {"bytes_per_row": per_row, "fixed_bytes": fixed, "plateau_rows": plateau, "safety": safety}


# ---------------------------------------------------------------- persistence
def test_config_dir_env_and_roundtrip(isolated_config, fake_cuda):
    assert bt.config_path() == isolated_config / bt.FILENAME
    prof = bt.profile("cuda", create=True)
    prof["workloads"]["nn/x"] = _entry()
    path = bt.save()
    data = json.loads(path.read_text())
    assert data["version"] == bt.SCHEMA_VERSION
    assert data["profiles"][FAKE_FP]["workloads"]["nn/x"]["plateau_rows"] == 2 ** 16

    bt._STATE = None  # simulate a fresh interpreter
    assert bt.summary("cuda")["nn/x"]["bytes_per_row"] == 1000.0


def test_other_gpu_profile_is_ignored(fake_cuda, monkeypatch):
    bt.profile("cuda", create=True)["workloads"]["nn/x"] = _entry(plateau=2 ** 17)
    monkeypatch.setattr(bt, "fingerprint", lambda device: "OtherGPU|24576MiB|torch0|cuda0")
    assert bt.resolve_batch_size("nn/x", "cuda", default=123) == 123


def test_clear(fake_cuda):
    prof = bt.profile("cuda", create=True)
    prof["workloads"].update({"a": _entry(), "b": _entry()})
    bt.clear("cuda", workloads=["a"])
    assert set(bt.summary("cuda")) == {"b"}
    bt.clear()
    assert bt.summary("cuda") == {}


def test_unreadable_file_falls_back(isolated_config):
    (isolated_config / bt.FILENAME).write_text("{not json")
    with pytest.warns(UserWarning):
        state = bt.load(force=True)
    assert state["profiles"] == {}


# ---------------------------------------------------------------- resolution
def test_cpu_device_always_default():
    assert bt.resolve_batch_size("nn/x", "cpu", default=777) == 777


def test_rows_memory_vs_plateau_limited():
    # memory-limited: 1 GiB * 0.8 / 1000 B ~ 858k rows, plateau 2**16 wins
    assert bt.rows_from_entry(_entry(plateau=2 ** 16), 2 ** 30) == 2 ** 16
    # tight memory: (100 MB * 0.8 - 10 MB) / 1000 B = 73886 -> floor to 1024 multiple
    rows = bt.rows_from_entry(_entry(fixed=10e6, plateau=2 ** 20), int(100e6))
    assert rows == (int((100e6 * 0.8 - 10e6) / 1000) // 1024) * 1024
    # never below MIN_ROWS
    assert bt.rows_from_entry(_entry(fixed=1e12), 2 ** 30) == bt.MIN_ROWS


def test_resolve_uses_live_memory_and_key_order(fake_cuda):
    bt.profile("cuda", create=True)["workloads"]["mb"] = _entry(per_row=1e4, plateau=2 ** 20)
    fake_cuda["avail"] = 2 ** 30
    big = bt.resolve_batch_size([None, "missing", "mb"], "cuda", default=5)
    fake_cuda["avail"] = 2 ** 28  # something else now occupies VRAM
    small = bt.resolve_batch_size("mb", "cuda", default=5)
    assert big > small >= bt.MIN_ROWS
    assert bt.resolve_batch_size("missing", "cuda", default=5) == 5


def test_env_disable(fake_cuda, monkeypatch):
    bt.profile("cuda", create=True)["workloads"]["mb"] = _entry()
    monkeypatch.setenv("NGIBBS_BATCH_TUNING", "0")
    assert bt.resolve_batch_size("mb", "cuda", default=9) == 9


def test_eos_tier():
    meta, fast = {"cptot", "KStot"}, {"Vp_fast"}
    assert bt.eos_tier(["rho", "Vp", "S"], meta, fast) == "isomorphic"
    assert bt.eos_tier(["rho", "cptot"], meta, fast) == "metamorphic"
    assert bt.eos_tier(["Vp_fast", "cptot"], meta, fast) == "fast"


def test_is_oom():
    assert bt.is_oom(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert not bt.is_oom(RuntimeError("shape mismatch"))


# ---------------------------------------------------------------- sweep logic
def _synthetic_measure(fixed, per_row, sat_rows, overhead=0.01, cliff_above=None):
    """Time = overhead + rows / throughput, throughput saturating at sat_rows."""
    def measure(fn, b, device, repeats):
        thr = 1e6 * min(b, sat_rows) / sat_rows
        t = overhead + b / thr
        if cliff_above is not None and b > cliff_above:
            t *= 5
        return t, int(fixed + per_row * b)
    return measure


def test_sweep_finds_plateau_and_memory_model(fake_cuda, monkeypatch):
    monkeypatch.setattr(bt, "_measure", _synthetic_measure(5e6, 2000.0, sat_rows=2 ** 15, overhead=0.0))
    fake_cuda["avail"] = 64 * 2 ** 30
    entry = bt.sweep(lambda b: None, "cuda", max_rows=2 ** 19, verbose=False)
    assert entry["plateau_rows"] == 2 ** 15
    assert entry["bytes_per_row"] == pytest.approx(2000.0, rel=1e-6)
    assert entry["fixed_bytes"] == pytest.approx(5e6, rel=1e-3)
    assert entry["stop_reason"] == "max_rows"


def test_sweep_stops_at_memory_limit(fake_cuda, monkeypatch):
    monkeypatch.setattr(bt, "_measure", _synthetic_measure(0, 4000.0, sat_rows=2 ** 22, overhead=0.0))
    fake_cuda["avail"] = 2 ** 29  # 512 MiB * 0.8 / 4000 B ~ 107k rows
    entry = bt.sweep(lambda b: None, "cuda", max_rows=2 ** 22, verbose=False)
    assert entry["stop_reason"] == "memory"
    assert entry["max_rows_tested"] == 2 ** 16
    assert entry["plateau_rows"] == 2 ** 16  # still climbing: largest safe size


def test_sweep_stops_at_throughput_cliff(fake_cuda, monkeypatch):
    monkeypatch.setattr(bt, "_measure", _synthetic_measure(0, 100.0, sat_rows=2 ** 16, overhead=0.0,
                                                           cliff_above=2 ** 16))
    fake_cuda["avail"] = 64 * 2 ** 30
    entry = bt.sweep(lambda b: None, "cuda", max_rows=2 ** 20, verbose=False)
    assert entry["stop_reason"] == "cliff"
    assert entry["plateau_rows"] == 2 ** 16


def test_sweep_stops_on_oom(fake_cuda, monkeypatch):
    base = _synthetic_measure(0, 100.0, sat_rows=2 ** 13, overhead=0.0)

    def measure(fn, b, device, repeats):
        if b > 2 ** 14:
            raise RuntimeError("CUDA out of memory")
        return base(fn, b, device, repeats)
    monkeypatch.setattr(bt, "_measure", measure)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    fake_cuda["avail"] = 64 * 2 ** 30
    entry = bt.sweep(lambda b: None, "cuda", max_rows=2 ** 20, verbose=False)
    assert entry["stop_reason"] == "oom"
    assert entry["max_rows_tested"] == 2 ** 14


# ------------------------------------------------- EmulatorAPI._staged_forward
API = pytest.importorskip("ngibbs.engine.API")


def _bare_api(device="cpu"):
    api = object.__new__(API.EmulatorAPI)
    api.device = torch.device(device)
    api._cpu_api = None
    api.isothermal_emulator = api.isentropic_emulator = api.open_emulator = None
    api._iso_path = api._isen_path = api._open_path = None
    api._CPU_BATCH_THRESHOLD = 0
    return api


def _recording_func(limit=None):
    sizes = []

    def f(x):
        sizes.append(x.shape[0])
        if limit is not None and x.shape[0] > limit:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return {"y": x * 2.0, "s": x.sum(1)}
    return f, sizes


def test_staged_forward_default_and_explicit():
    api = _bare_api()
    x = torch.randn(10_000, 3)
    f, sizes = _recording_func()
    out = api._staged_forward(f, x, None, _default_batch_size=4096)
    assert sizes == [4096, 4096, 1808]
    assert torch.equal(out["y"], x * 2.0)
    sizes.clear()
    api._staged_forward(f, x, 3000)
    assert sizes == [3000, 3000, 3000, 1000]


def test_staged_forward_uses_tuned_value(monkeypatch):
    api = _bare_api()
    seen = {}

    def fake_resolve(keys, device, default):
        seen["keys"], seen["default"] = keys, default
        return 2048
    monkeypatch.setattr(API._bt, "resolve_batch_size", fake_resolve)
    f, sizes = _recording_func()
    api._staged_forward(f, torch.randn(5000, 2), None, _workload="eos/isomorphic/X", _default_batch_size=99)
    assert seen == {"keys": "eos/isomorphic/X", "default": 99}
    assert sizes == [2048, 2048, 904]


def test_staged_forward_oom_backoff_matches_clean_run(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    api = _bare_api()
    x = torch.randn(20_000, 4)
    f_ok, _ = _recording_func()
    ref = api._staged_forward(f_ok, x, 4096)
    f_oom, sizes = _recording_func(limit=5000)
    with pytest.warns(RuntimeWarning, match="out of memory"):
        out = api._staged_forward(f_oom, x, 16384)
    assert sizes[0] == 16384 and 8192 in sizes and max(sizes[-3:]) <= 4096
    assert torch.equal(out["y"], ref["y"]) and torch.equal(out["s"], ref["s"])


def test_staged_forward_single_batch_oom_falls_back_to_chunks(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    api = _bare_api()
    x = torch.randn(6000, 2)
    f, sizes = _recording_func(limit=2000)
    with pytest.warns(RuntimeWarning):
        out = api._staged_forward(f, x, 8192)
    assert sizes[:3] == [6000, 4096, 2048]
    assert torch.equal(out["y"], x * 2.0)


def test_staged_forward_reraises_non_oom():
    api = _bare_api()

    def bad(x):
        raise RuntimeError("shape mismatch")
    with pytest.raises(RuntimeError, match="shape mismatch"):
        api._staged_forward(bad, torch.randn(5000, 2), 1024)


def test_nn_workload_keys_forwardNN_falls_back_to_forwardMB():
    api = _bare_api()

    class Emu:
        mass_balance = "iterative"

        def forwardMB(self, x):
            return x

        def forwardNN(self, x):
            return x
    api.isentropic_emulator = Emu()
    api._isen_path = "/models/HeFESTo_NPS_v3.tar"
    keys = api._nn_workload_keys(api.isentropic_emulator.forwardNN, {})
    assert keys == ["nn/forwardNN/isentropic/HeFESTo_NPS_v3.tar/mb=iterative",
                    "nn/forwardMB/isentropic/HeFESTo_NPS_v3.tar/mb=iterative"]
    keys = api._nn_workload_keys(api.isentropic_emulator.forwardMB, {"mass_balance": "pinv"})
    assert keys == ["nn/forwardMB/isentropic/HeFESTo_NPS_v3.tar/mb=pinv"]
    assert api._nn_workload_keys(lambda x: x, {}) is None


# ------------------------------------------------------------- get_T staging
def _temperature_api(with_reference):
    from ngibbs.utils.math_utils import Normalizer
    api = _bare_api()
    torch.manual_seed(3)
    F = 6
    api.temperature_model = torch.nn.Sequential(torch.nn.Linear(F, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
    api.temp_input_normalizer = Normalizer(torch.zeros(F), torch.full((F,), 2.0))
    api.temp_output_normalizer = Normalizer(torch.tensor([1000.0]), torch.tensor([500.0]))
    api.temperature_payload = {"p_feature_idx": 0, "s_feature_idx": 1} if with_reference else {}
    api._temp_path = "/models/temperature_fcnn.pt"
    return api


@pytest.mark.parametrize("with_reference", [False, True])
def test_get_T_chunked_matches_unbatched(with_reference):
    api = _temperature_api(with_reference)
    x = torch.rand(10_000, 6)
    x[:, 0] *= 100.0          # P (GPa)
    x[:, 1] = 2.0 + x[:, 1]   # S (J/g/K)
    ref = api._get_T_unstaged(x)
    out = api.get_T(x, batch_size=1500)
    assert out.shape == (10_000,)
    assert out.device == api.device
    assert torch.allclose(out, ref, rtol=1e-6, atol=0)  # float32 matmul blocking differs per chunk


def test_get_T_uses_its_own_workload_key(monkeypatch):
    api = _temperature_api(False)
    seen = {}

    def fake_resolve(keys, device, default):
        seen["keys"], seen["default"] = keys, default
        return 4096
    monkeypatch.setattr(API._bt, "resolve_batch_size", fake_resolve)
    api.get_T(torch.rand(9000, 6))
    assert seen == {"keys": "nn/get_T/temperature/temperature_fcnn.pt/mb=None",
                    "default": API.EmulatorAPI._T_BATCH_SIZE}
