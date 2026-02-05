import numpy as np
import sys
from pathlib import Path
from typing import Optional
import atexit

def is_almost_equal(truth, estimate, rel_tol=1e-3, abs_tol=None):
    if rel_tol is None and abs_tol is None:
        raise ValueError("At least one of rel_tol or abs_tol must be specified")

    truth_arr = np.asarray(truth)
    est_arr = np.asarray(estimate)

    if abs_tol is not None:
        agree_mask = np.abs(truth_arr - est_arr) <= abs_tol
    else:
        agree_mask = np.abs(truth_arr - est_arr) <= (rel_tol * np.abs(truth_arr))

    agree_mask = np.asarray(agree_mask, dtype=bool)

    if np.all(agree_mask):
        return True, agree_mask
    else:
        failed = np.count_nonzero(~agree_mask)
        total = agree_mask.size
        pct = 100.0 * failed / total if total else 0.0
        print(f"\n❌ FAILED: {pct}% of values are not equal within tolerance!")
        return False, agree_mask


def setup_test_logging(log_filename: str, log_dir: Optional[Path] = None) -> Path:
    """
    Tee stdout/stderr to a log file.

    Parameters
    ----------
    log_filename : str
        Name of the log file to create.
    log_dir : Optional[Path]
        Directory to write the log file. Defaults to current working directory.

    Returns
    -------
    Path
        Full path to the log file.
    """
    if log_dir is None:
        log_dir = Path.cwd()
    log_path = Path(log_dir) / log_filename
    log_file = open(log_path, 'w', encoding='utf-8')

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams

        def write(self, data):
            for stream in self._streams:
                stream.write(data)

        def flush(self):
            for stream in self._streams:
                stream.flush()

    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    atexit.register(log_file.close)
    return log_path
