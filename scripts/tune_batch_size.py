"""Tune and persist VRAM-aware batch sizes for the HeFESTo GPU emulator.

    python scripts/tune_batch_size.py                      # nn + eos_isomorphic
    python scripts/tune_batch_size.py --workloads nn eos_isomorphic eos_metamorphic
    python scripts/tune_batch_size.py --verify             # tune, then time default vs tuned
    python scripts/tune_batch_size.py --show               # print the saved profile only

The profile is written to ngibbs.utils.batch_tuning.config_path() and picked
up automatically by every later session on this GPU (explicit batch_size=
arguments still win).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ngibbs.utils import batch_tuning as bt  # noqa: E402


def show(device='cuda'):
    print(f"profile file: {bt.config_path()}")
    print(f"fingerprint:  {bt.fingerprint(device)}")
    entries = bt.summary(device)
    if not entries:
        print("  (no tuned workloads for this GPU)")
    for key, e in sorted(entries.items()):
        now = bt.rows_from_entry(e, bt.available_bytes(device))
        print(f"  {key:<60} plateau {e['plateau_rows']:>8d}  {e['bytes_per_row']:>9.0f} B/row  "
              f"-> {now:>8d} rows now  (stop: {e['stop_reason']}, {e['tuned_at']})")


def verify(api, n):
    """Wall time with tuning disabled (built-in defaults) vs enabled."""
    names = list(api.isothermal_emulator.ml_indexer.featureNames)
    feats = bt._synthetic_features(api, api.isothermal_emulator, n, None, seed=1)
    p_idx = names.index('P(GPa)(System_main)')
    t_idx = next(i for i, c in enumerate(names) if c.startswith('T('))
    PT = torch.stack([feats[:, p_idx], feats[:, t_idx]], 1).double()
    iso = bt.EOS_TIER_PROPERTIES['isomorphic']

    def nn():
        return api._staged_forward(api.isothermal_emulator.forwardMB, feats, None, Normalize=True,
                                   outputs=['component_moles', 'phase_moles', 'chem_out'])
    cm = nn()['component_moles'].to('cuda', torch.float64)

    def eos():
        return api._compute_bulk_EOS_properties(cm, PT, property_names=iso)

    for label, fn in (('forwardMB (isothermal)', nn), ('EOS isomorphic', eos)):
        row = []
        for mode in ('0', '1'):
            os.environ['NGIBBS_BATCH_TUNING'] = mode
            torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
            row.append(time.perf_counter() - t0)
        os.environ['NGIBBS_BATCH_TUNING'] = '1'
        print(f"  {label:<24} N={n}: default {row[0]:7.2f} s   tuned {row[1]:7.2f} s   "
              f"speed-up x{row[0] / row[1]:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workloads', nargs='+', default=['nn', 'eos_isomorphic'])
    ap.add_argument('--safety', type=float, default=bt.DEFAULT_SAFETY)
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--verify-n', type=int, default=2 ** 18)
    ap.add_argument('--show', action='store_true')
    a = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("No CUDA device visible.")
    if a.show:
        show()
        return
    from ngibbs.engine.models import HeFESToEmulatorGPU
    api = HeFESToEmulatorGPU
    if api is None:
        sys.exit("HeFESTo GPU emulator unavailable (model files missing?).")
    api.tune_batch_size(workloads=a.workloads, safety=a.safety, repeats=a.repeats)
    print()
    show()
    if a.verify:
        print("\nverification (one timed run each, after the tuning warm-up):")
        verify(api, a.verify_n)


if __name__ == '__main__':
    main()
