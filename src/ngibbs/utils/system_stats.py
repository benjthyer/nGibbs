"""Best-effort instantaneous system resource stats: CPU/RAM via `psutil`, GPU via
`pynvml`. Used by `builder.training.benchmark` to attach a hardware snapshot to each
training-speed benchmark record, and safe to call anywhere else that wants a quick
"what's the machine doing right now" reading.

Every field is None rather than raised when this environment can't provide it, rather
than the caller needing its own try/except around each metric. The one field that's
*always* None on this project's actual dev environment (WSL2) is `cpu_temp_c` -- WSL2
does not expose a hardware thermal zone to the Linux guest kernel, so
`psutil.sensors_temperatures()` returns an empty dict there regardless of the host's
real CPU temperature. Not fixable from inside the guest; this is a genuine platform
limitation, not a bug in this module.
"""
from __future__ import annotations

from typing import Optional, TypedDict


class SystemStats(TypedDict):
    cpu_percent: Optional[float]
    cpu_temp_c: Optional[float]
    ram_used_gb: Optional[float]
    ram_total_gb: Optional[float]
    gpu_name: Optional[str]
    gpu_temp_c: Optional[float]
    gpu_util_percent: Optional[float]
    gpu_mem_used_gb: Optional[float]
    gpu_mem_total_gb: Optional[float]
    gpu_power_w: Optional[float]


def _cpu_ram_stats() -> dict:
    out = {'cpu_percent': None, 'cpu_temp_c': None, 'ram_used_gb': None, 'ram_total_gb': None}
    try:
        import psutil
    except ImportError:
        return out

    try:
        # interval=None: non-blocking, compares against the last call rather than
        # sleeping to measure -- fine for a one-shot-per-benchmark snapshot, and
        # avoids stalling the training loop this gets called from.
        out['cpu_percent'] = psutil.cpu_percent(interval=None)
    except Exception:
        pass

    try:
        vm = psutil.virtual_memory()
        out['ram_used_gb'] = vm.used / 1e9
        out['ram_total_gb'] = vm.total / 1e9
    except Exception:
        pass

    try:
        temps = psutil.sensors_temperatures()
        if temps:
            # No fixed sensor name across machines -- take the first reading of
            # whichever sensor group psutil found, which is all "current CPU
            # temperature" means here anyway on the machines where this works.
            first_group = next(iter(temps.values()))
            if first_group:
                out['cpu_temp_c'] = float(first_group[0].current)
    except Exception:
        pass  # e.g. WSL2: sensors_temperatures() itself never raises, just returns {}

    return out


def _gpu_stats(gpu_index: int = 0) -> dict:
    out = {'gpu_name': None, 'gpu_temp_c': None, 'gpu_util_percent': None,
           'gpu_mem_used_gb': None, 'gpu_mem_total_gb': None, 'gpu_power_w': None}
    try:
        import pynvml
    except ImportError:
        return out

    try:
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            name = pynvml.nvmlDeviceGetName(h)
            out['gpu_name'] = name.decode() if isinstance(name, bytes) else name
            out['gpu_temp_c'] = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            out['gpu_util_percent'] = float(util.gpu)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            out['gpu_mem_used_gb'] = mem.used / 1e9
            out['gpu_mem_total_gb'] = mem.total / 1e9
            out['gpu_power_w'] = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass  # no NVIDIA GPU, driver mismatch, permissions, etc. -- degrade to None

    return out


def get_system_stats(gpu_index: int = 0) -> SystemStats:
    """One instantaneous reading of CPU%, RAM, and GPU temp/util/VRAM/power.

    Cheap enough to call once per training epoch or benchmark point (a few ms,
    dominated by `nvmlInit`/`nvmlShutdown`), but not intended for a tight polling
    loop -- for that, keep one `pynvml.nvmlInit()` alive across calls instead of
    re-initializing every time.
    """
    stats: dict = {}
    stats.update(_cpu_ram_stats())
    stats.update(_gpu_stats(gpu_index))
    return stats  # type: ignore[return-value]


if __name__ == "__main__":
    import json
    print(json.dumps(get_system_stats(), indent=2))
