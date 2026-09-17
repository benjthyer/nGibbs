### melts_vec Performance Benchmark
#
# Mirrors `scripts/performance_benchmarking.py`'s structure and style (same
# batch-size sweep, same "N assemblages in 10 s" target line, same CSV/PNG
# output convention via `_plot_output`), but for the vectorized MELTS EOS
# (`EOS_arithmetic_MELTS/melts_vec`) rather than the HeFESTo NN emulator.
#
# The HeFESTo script times an NN forward pass (`ForwardMB`) plus the
# subsequent property solve. melts_vec has no NN step of its own -- it IS
# the property solve (`get_property_melts_vectorized_from_assemblage`'s
# actual per-phase work, whether the composition came from a real MELTS
# emulator or anywhere else) -- so this script instead times the underlying
# `compute_liquid_bulk()` / `compute_*_solution()` calls directly, one
# phase at a time and combined into a synthetic "typical assemblage"
# (liquid + olivine + orthopyroxene + clinopyroxene + spinel + plagioclase
# + k-feldspar), which is what `get_property_melts_vectorized_from_
# assemblage()` dispatches to per phase present. Compositions are
# Dirichlet-sampled (uniform on each phase's own endmember simplex) --
# fine for a *speed* benchmark, since the vectorized math's wall time
# depends on batch SHAPE, not on whether the numbers are a physically
# realistic magma composition (unlike `reproduce_meltstables.py`, which
# needs real MELTS-computed compositions for a *correctness* check).
#
# Target (Ben's own words): 32,768 (2^15) assemblages in 10 seconds.
#
# IMPORTANT: per Ben's explicit instruction, this script is built but NOT
# run as part of this pass -- do not execute the __main__ block without
# being asked.

from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Handle both notebook and script contexts, exactly as performance_benchmarking.py does.
try:
    repo_root = Path(__file__).parent.parent
except NameError:
    repo_root = Path.cwd().parent

src_root = repo_root / "src"
ngibbs_root = src_root / "ngibbs"
engine_root = ngibbs_root / "engine"

for p in (src_root, ngibbs_root, engine_root, repo_root / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from _plot_output import plot_path

from EOS_arithmetic_MELTS.melts_vec import (
    load_solids, load_liquid, compute_liquid_bulk,
    feldspar, olivine, clinopyroxene, orthopyroxene, spinel, rhomsghiorso,
    compute_feldspar_solution, compute_olivine_solution,
    compute_clinopyroxene_solution, compute_orthopyroxene_solution,
    compute_spinel_solution, compute_rhm_oxide_solution,
)

print("Imports successful!")
print(f"Repository root: {repo_root}")


# ── Synthetic batch construction ────────────────────────────────────────────
# T in [1073.15, 1773.15] K (800-1500 C), P in [1, 10000] bar -- a broad but
# ordinary magmatic range spanning most of the MELTStables tables used for
# `reproduce_meltstables.py`. Compositions are Dirichlet(alpha=1) draws
# (uniform on the simplex) per phase, independent of any other phase's
# composition -- a synthetic assemblage, not a physically equilibrated one
# (this script measures wall time, not correctness; see module docstring).

def random_TP(rng: np.random.Generator, n: int):
    T = rng.uniform(1073.15, 1773.15, size=n)
    P = rng.uniform(1.0, 10000.0, size=n)
    return T, P


def random_composition(rng: np.random.Generator, n_endmembers: int, n: int) -> np.ndarray:
    return rng.dirichlet(np.ones(n_endmembers), size=n)


# name -> (compute_fn, endmembers, extra_args_fn(solid_params) -> tuple of
# positional args after (T,P,X))
SOLID_PHASES = {
    "feldspar_plag": (compute_feldspar_solution, feldspar.ENDMEMBERS),
    "feldspar_kfs":  (compute_feldspar_solution, feldspar.ENDMEMBERS),
    "olivine":       (compute_olivine_solution, olivine.ENDMEMBERS),
    "orthopyroxene": (compute_orthopyroxene_solution, orthopyroxene.ENDMEMBERS),
    "clinopyroxene": (compute_clinopyroxene_solution, clinopyroxene.ENDMEMBERS),
    "spinel":        (compute_spinel_solution, spinel.ENDMEMBERS),
    "rhm_oxide":     (compute_rhm_oxide_solution, rhomsghiorso.ENDMEMBERS),
}


def time_solid_phase(rng, phase_key, batch_size, solid_params):
    compute_fn, endmembers = SOLID_PHASES[phase_key]
    T, P = random_TP(rng, batch_size)
    X = random_composition(rng, len(endmembers), batch_size)
    t0 = time.time()
    compute_fn(T, P, X, solid_params)
    return time.time() - t0


def time_liquid(rng, batch_size, liquid_params, liquid_names):
    T, P = random_TP(rng, batch_size)
    X = random_composition(rng, len(liquid_names), batch_size)
    t0 = time.time()
    compute_liquid_bulk(T, P, X, liquid_params, names=liquid_names)
    return time.time() - t0


def run_benchmark(batch_sizes, seed=0):
    """Returns a dict: {'liquid':[...], 'olivine':[...], ..., 'full_assemblage':[...]}
    each a list of wall times, one per batch size, aligned with batch_sizes.

    'full_assemblage' is the sum of that batch size's own per-phase timings
    (liquid + every SOLID_PHASES entry) -- a faithful proxy for what
    `get_property_melts_vectorized_from_assemblage()` costs for a 'typical'
    multi-phase assemblage, since its own per-phase dispatch loop has only
    O(1) overhead (dict bookkeeping / coverage-fraction accounting) beyond
    these same per-phase calls, and reusing the already-collected timings
    here avoids computing everything twice."""
    rng = np.random.default_rng(seed)
    solid_params = load_solids()
    liquid_params = load_liquid()
    liquid_names = liquid_params.labels

    timings = {k: [] for k in list(SOLID_PHASES) + ["liquid", "full_assemblage"]}
    for i, batch_size in enumerate(batch_sizes):
        print(f"batch_size = {batch_size}")
        timings["liquid"].append(time_liquid(rng, batch_size, liquid_params, liquid_names))
        for phase_key in SOLID_PHASES:
            t = time_solid_phase(rng, phase_key, batch_size, solid_params)
            timings[phase_key].append(t)
            print(f"    {phase_key:16s} {t:8.3f} s")
        timings["full_assemblage"].append(sum(timings[k][i] for k in list(SOLID_PHASES) + ["liquid"]))
        print(f"    {'full_assemblage':16s} {timings['full_assemblage'][i]:8.3f} s")

    for k in timings:
        timings[k] = np.array(timings[k], dtype=float)
    return timings


def plot_results(batch_sizes, timings, out_prefix="melts_vec"):
    plt.figure()
    for k, t in timings.items():
        ls = '-' if k == "full_assemblage" else '--'
        lw = 2.0 if k == "full_assemblage" else 1.0
        plt.loglog(batch_sizes, t, label=k, linestyle=ls, linewidth=lw)

    # Ben's stated target: 2^15 (32768) assemblages in 10 s.
    plt.axvline(2**15, color='crimson', lw=0.8, alpha=0.6)
    plt.axhline(10, color='crimson', lw=0.8, alpha=0.6)
    plt.plot([2**15], [10], marker='*', ms=12, color='crimson', linestyle='none',
             label='target: 2^15 in 10 s')

    plt.xlabel('Number of Assemblages (N)')
    plt.ylabel('Wall Time (s)')
    plt.grid(True)
    plt.title('melts_vec Performance: Total Time per Phase')
    plt.legend(fontsize=7)
    plt.savefig(plot_path(f'{out_prefix}_performance_results.png'), dpi=300)
    plt.show()

    plt.figure()
    for k, t in timings.items():
        ls = '-' if k == "full_assemblage" else '--'
        lw = 2.0 if k == "full_assemblage" else 1.0
        plt.loglog(batch_sizes, batch_sizes/t, label=k, linestyle=ls, linewidth=lw)
    plt.axvline(2**15, color='crimson', lw=0.8, alpha=0.6)
    plt.xlabel('Number of Assemblages (N)')
    plt.ylabel('Assemblages per second')
    plt.grid(True)
    plt.title('melts_vec Performance: Throughput per Phase')
    plt.legend(fontsize=7)
    plt.savefig(plot_path(f'{out_prefix}_per_second_performance_results.png'), dpi=300)
    plt.show()


def main():
    batch_sizes = (2**np.linspace(4, 18, 15)).astype(int)
    timings = run_benchmark(batch_sizes)

    cols = {'Batch Size': batch_sizes}
    for k, t in timings.items():
        cols[f'melts_vec ({k})'] = t
    pd.DataFrame(cols).to_csv('melts_vec_performance_results.csv', index=False)

    plot_results(batch_sizes, timings)

    print('\n' + '=' * 72)
    print(f"Target: 2^15 = {2**15} assemblages in 10 s")
    print('=' * 72)
    idx15 = int(np.argmin(np.abs(batch_sizes - 2**15)))
    print(f"full_assemblage at N={batch_sizes[idx15]}: {timings['full_assemblage'][idx15]:.3f} s "
          f"({'PASS' if timings['full_assemblage'][idx15] <= 10.0 else 'over target'})")


if __name__ == "__main__":
    # NOT run as part of this pass -- built and left here for Ben to invoke
    # explicitly when ready (see module docstring).
    main()
