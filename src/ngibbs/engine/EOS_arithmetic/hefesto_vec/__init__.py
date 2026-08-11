"""
hefesto_vec — Vectorised Python implementation of the HeFESTo EOS.

Quick start
-----------
>>> from hefesto_vec import compute, load_control
>>> params = load_control('BENCHMARK/control',
...                        param_dir_override='HeFESTo_Parameters_010123')
>>> result = compute(P, T, X, params)

Metamorphic (latent-heat / phase-change) totals
-----------------------------------------------
``compute()`` is isomorphic: it evaluates everything at fixed species moles,
reproducing fort.59's ``alpiso / cpiso / btiso / bsiso``.  To also get the
phase-change contributions that fort.56 reports (``alptot / cptot / btot /
bstot``), build the static tables once and augment the result:

>>> from hefesto_vec import build_tables, add_metamorphic
>>> tables = build_tables(params, params.param_dir)
>>> result = compute(P, T, X, params)
>>> result.update(add_metamorphic(result, X, T, P, tables, include_fast=True))

``include_fast=True`` additionally applies the intra-phase order-disorder
("fast") relaxation to the per-phase moduli, overwriting ``Vp``/``Vb``/``Kv``/
``Kr``/``Kh`` with the values fort.56 actually reports; the pre-correction
values are kept as ``Vp_iso``/``Vb_iso``/``Kh_iso``.  ``Vs`` is unaffected --
the shear modulus carries no metamorphic term.  Without it, Vp is biased high
by up to 0.26% in the lower mantle, where pv and mw have active ordering.
"""

from .params       import load_control, HeFESToParams
from .compute      import compute
from .aggregate    import vrh_average, apply_fast_metamorphic
from .metamorphic  import (
    MetamorphicTables,
    NSMALL_REL,
    prune_active_set,
    trace_species_leverage,
    build_tables,
    add_metamorphic,
    molar_derivatives,
    hessian_blocks,
    mixing_terms,
    metamorphic_terms,
    combine_totals,
    clapeyron_terms,
)

__all__ = [
    'load_control', 'HeFESToParams', 'compute',
    'vrh_average', 'apply_fast_metamorphic',
    'MetamorphicTables', 'NSMALL_REL', 'prune_active_set',
    'trace_species_leverage',
    'build_tables', 'add_metamorphic',
    'molar_derivatives', 'hessian_blocks', 'mixing_terms',
    'metamorphic_terms', 'combine_totals', 'clapeyron_terms',
]
