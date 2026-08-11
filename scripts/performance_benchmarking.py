### Performance Testing

import time
import sys
from pathlib import Path
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Note: Will need to move HeFESTo_functions.py to nMELTS/utils so it will be ported properly. 

# Handle both notebook and script contexts
try:
    repo_root = Path(__file__).parent.parent
except NameError:
    # Running in Jupyter notebook context
    repo_root = Path.cwd().parent

# Add both src and src/ngibbs to path
src_root = repo_root / "src"
ngibbs_root = src_root / "ngibbs"

if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
if str(ngibbs_root) not in sys.path:
    sys.path.insert(0, str(ngibbs_root))

# Import utilities and API
from ngibbs.utils.math_utils import grid_sample, grid_sample_explicit
from ngibbs.engine.API import HeFESToAPI, HeFESToEmulatorCPU, HeFESToEmulatorGPU
from ngibbs.config.constants import HEFESTO_ABBREVIATION_TO_SHORT_NAMES
REV_HEFESTO_ABBREVIATION_TO_SHORT_NAMES = {v: k for k, v in HEFESTO_ABBREVIATION_TO_SHORT_NAMES.items()}


print("Imports successful!")
print(f"Repository root: {repo_root}")
print(f"ngibbs root: {ngibbs_root}")

# Check emulator availability
if HeFESToEmulatorCPU is None:
    print("\nNote: Pre-instantiated emulators not available (model files missing).")
    print("Create HeFESToAPI instances with valid model paths when needed.")
else:
    print("\nEmulators available:")
    print(f"  - HeFESToEmulatorCPU")
    print(f"  - HeFESToEmulatorGPU")

input_dict = {
    'S(J/g/K)(System_main)': 2.5,
    'Si': 3.57887,
    'Mg': 5.64233,
    'Fe': 0.58052,
    'Ca': 0.07969,
    'Al': 0.09740,
    'Na': 0.00160,
    'Cr': 0.01960,
    'O': 13.63947
    }

def example_array(batch_size):
    pressures = np.linspace(0,140, batch_size)
    headers = []
    comp_array = np.zeros((1, len(input_dict)), dtype=np.float32)
    for idx, (element, value) in enumerate(input_dict.items()):
        headers.append(element)
        comp_array[0, idx] = value
        
    input_grid = np.concatenate([pressures.reshape(-1, 1), np.tile(comp_array, (len(pressures), 1))], axis=1)

    headers = ['P(GPa)'] + headers

    return input_grid, headers

# ── Property variants ────────────────────────────────────────────────────────
# The EOS is now three routines, not one, and they cost very different amounts.
# Requesting a metamorphic key (cptot/KStot/...) makes the API solve for dn/dT
# and dn/dP; requesting Vp_fast makes it solve a second time restricted to the
# order-disorder species.  Timing them separately shows what the phase-change
# terms actually cost on top of the isomorphic EOS.
#
#   isomorphic  — fixed composition.  What the EOS did before this work.
#   metamorphic — + the aggregate phase-change terms (Cp, KS, alpha).  One extra
#                 projected-Hessian solve over all species.
#   +fast       — + the intra-phase order-disorder relaxation that corrects
#                 VP/VB.  A second, smaller solve.  This is the full fort.56
#                 reproduction.
PROPERTY_VARIANTS = {
    'isomorphic':  ('rho', 'Vp', 'Vs', 'S'),
    'metamorphic': ('rho', 'Vp', 'Vs', 'S', 'cptot', 'KStot'),
    '+fast':       ('rho', 'Vp_fast', 'Vs', 'S', 'cptot', 'KStot'),
}
VARIANT_STYLE = {'isomorphic': ':', 'metamorphic': '--', '+fast': '-.'}

batch_sizes = (2**np.linspace(4, 18, 15)).astype(int)
GPU_emulator = []
CPU_emulator = []
NN_emulator = []
NN_properties = []
# device -> variant -> list of wall times, one per batch size
timings = {dev: {v: [] for v in PROPERTY_VARIANTS} for dev in ('GPU', 'CPU')}
skip = np.zeros_like(batch_sizes, dtype=bool)
batch_sizes = np.array(batch_sizes)


def sync(device):
    """Block until queued CUDA work is done.

    CUDA kernel launches are asynchronous, so without this the GPU timings
    measure how long it takes to *enqueue* the work, not to run it — which is
    why an un-synchronised GPU benchmark can look impossibly fast at large
    batch sizes.
    """
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()


def time_properties(api, component_moles, PT, device):
    """Wall time for each property variant on one batch, in order."""
    out = {}
    for variant, names in PROPERTY_VARIANTS.items():
        sync(device)
        t0 = time.time()
        api._compute_bulk_EOS_properties(component_moles, PT=PT, property_names=names)
        sync(device)
        out[variant] = time.time() - t0
        print(f"    {variant:12s} {out[variant]:8.3f} s")
    return out

HAVE_GPU = torch.cuda.is_available()
if not HAVE_GPU:
    print('\n[warn] No CUDA device visible — GPU columns will be zero-filled.')

try:
    # first GPU
    for i, batch_size in enumerate(batch_sizes):

        if batch_size < 8000 or not HAVE_GPU:  # GPU isn't active on smaller batches, so skip for clarity
            GPU_emulator.append(0)
            for v in PROPERTY_VARIANTS:
                timings['GPU'][v].append(0)
            skip[i] = True
            continue

        print(f'GPU: {batch_size}')
        input_array, headers = example_array(batch_size)

        sync('cuda')
        begin_emulator = time.time()
        output = HeFESToEmulatorGPU.ForwardMB(torch.tensor(input_array, dtype=torch.float32, device='cuda'), headers=headers, outputs=['component_moles', 'temperature'])
        sync('cuda')
        end_emulator = time.time()
        GPU_emulator.append(end_emulator - begin_emulator)

        PT = torch.concatenate([torch.tensor(input_array[:,0], dtype=torch.float32, device='cuda').reshape(-1, 1), torch.tensor(output['temperature'], dtype=torch.float32, device='cuda').reshape(-1, 1)], dim=1)

        print(f"{time.time() - end_emulator:.4f} IO time")
        got = time_properties(
            HeFESToEmulatorGPU,
            torch.tensor(output['component_moles'], dtype=torch.float32, device='cuda'),
            PT, 'cuda',
        )
        for v, t in got.items():
            timings['GPU'][v].append(t)

    for batch_size in batch_sizes:
        print(f'CPU: {batch_size}')
        input_array, headers = example_array(batch_size)

        begin_emulator = time.time()
        output = HeFESToEmulatorCPU.ForwardMB(torch.tensor(input_array, dtype=torch.float32, device='cpu'), headers=headers, outputs=['component_moles', 'temperature'])
        end_emulator = time.time()
        CPU_emulator.append(end_emulator - begin_emulator)

        PT = torch.concatenate([torch.tensor(input_array[:,0], dtype=torch.float32, device='cpu').reshape(-1, 1), torch.tensor(output['temperature'], dtype=torch.float32, device='cpu').reshape(-1, 1)], dim=1)

        print(f"{time.time() - end_emulator:.4f} IO time")
        got = time_properties(
            HeFESToEmulatorCPU,
            torch.tensor(output['component_moles'], dtype=torch.float32, device='cpu'),
            PT, 'cpu',
        )
        for v, t in got.items():
            timings['CPU'][v].append(t)

finally:
    # A run interrupted partway leaves short lists; pad so the arrays still
    # line up with batch_sizes and the partial results stay plottable.
    for dev in timings:
        for v in timings[dev]:
            timings[dev][v] += [np.nan] * (len(batch_sizes) - len(timings[dev][v]))
            timings[dev][v] = np.array(timings[dev][v], dtype=float)
    GPU_emulator = np.array(GPU_emulator + [np.nan] * (len(batch_sizes) - len(GPU_emulator)), dtype=float)
    CPU_emulator = np.array(CPU_emulator + [np.nan] * (len(batch_sizes) - len(CPU_emulator)), dtype=float)
    # Back-compat aliases: the full-fidelity variant is what 'Properties' meant.
    GPU_properties = timings['GPU']['+fast']
    CPU_properties = timings['CPU']['+fast']
    print(f"GPU emulator: {GPU_emulator}")
    print(f"CPU emulator: {CPU_emulator}")
    for dev in ('GPU', 'CPU'):
        for v in PROPERTY_VARIANTS:
            print(f"{dev} properties [{v}]: {timings[dev][v]}")
    print(f"batch sizes: {batch_sizes}")

    print('\n' + '=' * 72)
    print('Cost of the phase-change terms (ratio to isomorphic)')
    print('=' * 72)
    print(f"{'batch':>8s}  " + '  '.join(f'{d} {v:>11s}' for d in ('CPU',) for v in ('metamorphic', '+fast')))
    for i, bs in enumerate(batch_sizes):
        iso = timings['CPU']['isomorphic'][i]
        if not np.isfinite(iso) or iso <= 0:
            continue
        print(f'{bs:8d}  ' + '  '.join(
            f'{timings["CPU"][v][i] / iso:15.2f}x' for v in ('metamorphic', '+fast')))

_cols = {'Batch Size': batch_sizes, 'GPU Emulator': GPU_emulator, 'CPU Emulator': CPU_emulator}
for dev in ('GPU', 'CPU'):
    for v in PROPERTY_VARIANTS:
        _cols[f'{dev} Properties ({v})'] = timings[dev][v]
pd.DataFrame(_cols).to_csv('performance_results.csv', index=False)


plt.loglog(batch_sizes, batch_sizes, label='Sequential HeFESTo', color = 'gray', linestyle='-', alpha=0.5)
plt.loglog(batch_sizes, batch_sizes*0.8, label='Sequential HeFESTo on Four Cores', color = 'lightgray', linestyle='-', alpha=0.5)

plt.loglog(batch_sizes[~skip], GPU_emulator[~skip], label='GPU Emulator', color = 'forestgreen', linestyle='-')
plt.loglog(batch_sizes, CPU_emulator, label='CPU Emulator', color = 'darkorange', linestyle='-')
for v, ls in VARIANT_STYLE.items():
    plt.loglog(batch_sizes[~skip], timings['GPU'][v][~skip],
               label=f'GPU Properties ({v})', color='forestgreen', linestyle=ls)
    plt.loglog(batch_sizes, timings['CPU'][v],
               label=f'CPU Properties ({v})', color='darkorange', linestyle=ls)
#plt.loglog(batch_sizes, NN_emulator, label='NN Emulator', color = 'teal', linestyle='-')
#plt.loglog(batch_sizes, NN_properties, label='NN Properties', color = 'teal', linestyle='--')

# The project target: 2**15 assemblages in 10 s.
plt.axvline(2**15, color='crimson', lw=0.8, alpha=0.6)
plt.axhline(10, color='crimson', lw=0.8, alpha=0.6)
plt.plot([2**15], [10], marker='*', ms=12, color='crimson', linestyle='none',
         label='target: 2^15 in 10 s')

plt.xlabel('Number of Assemblages (N)')
plt.ylabel('Wall Time (s)')
plt.grid(True)
plt.title('HeFESTo Emulator Performance Comparison: Total Time')
plt.legend(fontsize=7)
plt.savefig('performance_results.png', dpi=300)
plt.show()

plt.plot(batch_sizes[~skip], np.array(batch_sizes[~skip])/np.array(GPU_emulator[~skip]), label='GPU Emulator', color = 'forestgreen', linestyle='-')
plt.plot(batch_sizes, np.array(batch_sizes)/np.array(CPU_emulator), label='CPU Emulator', color = 'darkorange', linestyle='-')
for v, ls in VARIANT_STYLE.items():
    plt.plot(batch_sizes[~skip], batch_sizes[~skip] / timings['GPU'][v][~skip],
             label=f'GPU Properties ({v})', color='forestgreen', linestyle=ls)
    plt.plot(batch_sizes, batch_sizes / timings['CPU'][v],
             label=f'CPU Properties ({v})', color='darkorange', linestyle=ls)
#plt.plot(batch_sizes, np.array(batch_sizes)/np.array(NN_emulator), label='NN Emulator', color = 'teal', linestyle='-')
#plt.plot(batch_sizes, np.array(batch_sizes)/np.array(NN_properties), label='NN Properties', color = 'teal', linestyle='--')
plt.xlabel('Batch Size')
plt.ylabel('Assemblages per second')
plt.grid(True)
plt.title('Emulator Performance Comparison: Assemblages per Second')
plt.legend(fontsize=7)
plt.savefig('per_second_performance_results.png', dpi=300)
plt.show()

properties_rate = batch_sizes*(10/4000) #seconds

plt.loglog(batch_sizes[~skip], GPU_emulator[~skip]/batch_sizes[~skip], label='GPU Emulator speedup', color = 'forestgreen', linestyle='-')
plt.loglog(batch_sizes, CPU_emulator/batch_sizes, label='CPU Emulator speedup', color = 'darkorange', linestyle='-')
for v, ls in VARIANT_STYLE.items():
    plt.loglog(batch_sizes[~skip], timings['GPU'][v][~skip] / properties_rate[~skip],
               label=f'Properties GPU speedup ({v})', color='forestgreen', linestyle=ls)
    plt.loglog(batch_sizes, timings['CPU'][v] / properties_rate,
               label=f'Properties CPU speedup ({v})', color='darkorange', linestyle=ls)
#plt.loglog(batch_sizes, NN_emulator, label='NN Emulator', color = 'teal', linestyle='-')
#plt.loglog(batch_sizes, NN_properties, label='NN Properties', color = 'teal', linestyle='--')

plt.xlabel('Number of Assemblages (N)')
plt.ylabel('Relative Speed Up: nGibbsMin Equilibria/Properties')
plt.grid(True)
plt.title('Relative Speedup of nGibbsMin')
plt.legend(fontsize=7)
plt.savefig('performance_results_relative.png', dpi=300)
plt.show()