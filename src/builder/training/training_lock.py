"""
Single-process lock for training.

Training entry points (main.py, train_temperature_residual_fcnn.py) share on-disk
state that is not safe to use from two processes at once: the cached dataset
workspaces (which a second run may wipe and rebuild), fixed-name temp checkpoints
in temp_models/ (temp_upper_train.pt etc.), and `{tarname}_{episode}.tar` outputs.
Running two trainings concurrently buys no speed, so instead of making each piece
process-safe, one lock file serializes whole training runs.

`training_lock()` is held from before the cache is loaded/built until the process
exits (released in a `finally`). A second process that finds the lock waits,
re-checking every `poll_seconds` (5 min), and raises TimeoutError after
`timeout_seconds` (24 h). A lock left behind by a process that no longer exists
(kill -9, crash of the whole interpreter, reboot) is detected by its PID and
taken over rather than waited on.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

LOCK_PATH_DEFAULT = Path(__file__).parent / "training.lock"
POLL_SECONDS_DEFAULT = 5 * 60
TIMEOUT_SECONDS_DEFAULT = 24 * 60 * 60


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def _read_lock(lock_path):
    try:
        return json.loads(lock_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _is_stale(info):
    """True only when we can tell the owner is gone: same host, PID not running."""
    if not info or info.get("host") != socket.gethostname():
        return False
    pid = info.get("pid")
    return isinstance(pid, int) and not _pid_alive(pid)


def _try_create(lock_path):
    info = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "command": " ".join(sys.argv),
    }
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        json.dump(info, f, indent=2)
    return True


@contextmanager
def training_lock(lock_path=None, poll_seconds=POLL_SECONDS_DEFAULT,
                  timeout_seconds=TIMEOUT_SECONDS_DEFAULT):
    lock_path = Path(lock_path) if lock_path is not None else LOCK_PATH_DEFAULT
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds

    while not _try_create(lock_path):
        info = _read_lock(lock_path)
        if _is_stale(info):
            print(f"[training_lock] Removing stale lock {lock_path} "
                  f"(PID {info['pid']} is no longer running).")
            lock_path.unlink(missing_ok=True)
            continue
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Training lock {lock_path} still held after {timeout_seconds / 3600:.1f} h "
                f"by {info}. If that process is gone, delete the lock file and rerun.")
        print(f"[training_lock] Another training process holds {lock_path}: {info}. "
              f"Checking again in {poll_seconds / 60:.0f} min "
              f"({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}).", flush=True)
        time.sleep(poll_seconds)

    print(f"[training_lock] Acquired {lock_path} (PID {os.getpid()}).")
    try:
        yield lock_path
    finally:
        # Only remove our own lock - never one another process took over after
        # deciding ours was stale.
        info = _read_lock(lock_path)
        if info is not None and info.get("pid") == os.getpid():
            lock_path.unlink(missing_ok=True)
