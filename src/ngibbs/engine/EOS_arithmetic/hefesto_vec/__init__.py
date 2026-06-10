"""
hefesto_vec — Vectorised Python implementation of the HeFESTo EOS.

Quick start
-----------
>>> from hefesto_vec import compute, load_control
>>> params = load_control('BENCHMARK/control',
...                        param_dir_override='HeFESTo_Parameters_010123')
>>> result = compute(P, T, X, params)
"""

from .params  import load_control, HeFESToParams
from .compute import compute

__all__ = ['load_control', 'HeFESToParams', 'compute']
