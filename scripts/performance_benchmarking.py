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

# Add both src and src/nMELTS to path
src_root = repo_root / "src"
nmelts_root = src_root / "nMELTS"

if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
if str(nmelts_root) not in sys.path:
    sys.path.insert(0, str(nmelts_root))

# Import utilities and API
from nMELTS.utils.math_utils import grid_sample, grid_sample_explicit
from nMELTS.engine.API import HeFESToAPI, HeFESToEmulatorCPU, HeFESToEmulatorGPU
from nMELTS.config.constants import HEFESTO_ABBREVIATION_TO_SHORT_NAMES
REV_HEFESTO_ABBREVIATION_TO_SHORT_NAMES = {v: k for k, v in HEFESTO_ABBREVIATION_TO_SHORT_NAMES.items()}


print("Imports successful!")
print(f"Repository root: {repo_root}")
print(f"nMELTS root: {nmelts_root}")

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

batch_sizes = np.array([2**19.5]).astype(int)#.5, 2**20, 2**20.25, 2**20.5]).astype(int) #np.append(2**np.linspace(4, 15, 20), 2**np.array([18.25,18.5])).astype(int)

GPU_emulator = []
CPU_emulator = []
NN_emulator = []
GPU_properties = []
CPU_properties = []
NN_properties = []

try:
    # first GPU
    for batch_size in batch_sizes:
        print(f'GPU: {batch_size}')
        input_array, headers = example_array(batch_size)

        begin_emulator = time.time()
        output = HeFESToEmulatorGPU.ForwardMB(torch.tensor(input_array, dtype=torch.float32, device='cuda'), headers=headers, outputs=['component_moles', 'temperature'])
        end_emulator = time.time()
        GPU_emulator.append(end_emulator - begin_emulator)

        PT = torch.concatenate([torch.tensor(input_array[:,0], dtype=torch.float32, device='cuda').reshape(-1, 1), torch.tensor(output['temperature'], dtype=torch.float32, device='cuda').reshape(-1, 1)], dim=1)

        begin_properties = time.time()
        print(f"{begin_properties - end_emulator:.4f} IO time")
        #for i in range(5):
            #print(f"Property calculation {i+1}: {time.time() - begin_properties:.4f} seconds")
        output_properties = HeFESToEmulatorGPU.get_property_burnman_vectorized_from_assemblage(torch.tensor(output['component_moles'], dtype=torch.float32, device='cuda'), PT=PT)
        GPU_properties.append(time.time() - begin_properties)

    """for batch_size in batch_sizes:
        print(f'GPU: {batch_size}')
        input_array, headers = example_array(batch_size)

        begin_emulator = time.time()
        output = HeFESToEmulatorGPU.ForwardNN(torch.tensor(input_array, dtype=torch.float32, device='cuda'), headers=headers, outputs=['component_moles', 'temperature'])
        end_emulator = time.time()
        NN_emulator.append(end_emulator - begin_emulator)

        PT = torch.concatenate([torch.tensor(input_array[:,0], dtype=torch.float32, device='cuda').reshape(-1, 1), torch.tensor(output['temperature'], dtype=torch.float32, device='cuda').reshape(-1, 1)], dim=1)

        begin_properties = time.time()
        print(f"{begin_properties - end_emulator:.4f} IO time")
        output_properties = HeFESToEmulatorGPU.get_property_burnman_vectorized_from_assemblage(torch.tensor(output['component_moles'], dtype=torch.float32, device='cuda'), PT=PT)
        NN_properties.append(time.time() - begin_properties)"""


    for batch_size in batch_sizes:
        print(f'CPU: {batch_size}')
        input_array, headers = example_array(batch_size)

        begin_emulator = time.time()
        output = HeFESToEmulatorCPU.ForwardMB(torch.tensor(input_array, dtype=torch.float32, device='cpu'), headers=headers, outputs=['component_moles', 'temperature'])
        end_emulator = time.time()
        CPU_emulator.append(end_emulator - begin_emulator)

        PT = torch.concatenate([torch.tensor(input_array[:,0], dtype=torch.float32, device='cpu').reshape(-1, 1), torch.tensor(output['temperature'], dtype=torch.float32, device='cpu').reshape(-1, 1)], dim=1)

        begin_properties = time.time()
        print(f"{begin_properties - end_emulator:.4f} IO time")
        #output_properties = HeFESToEmulatorCPU.get_property_burnman_vectorized_from_assemblage(torch.tensor(output['component_moles'], dtype=torch.float32, device='cpu'), PT=PT)
        CPU_properties.append(0)#time.time() - begin_properties)

finally:
    print(f"GPU properties: {GPU_properties}")
    print(f"CPU properties: {CPU_properties}")
    print(f"GPU emulator: {GPU_emulator}")
    print(f"CPU emulator: {CPU_emulator}")
    print(f"batch sizes: {batch_sizes}")

pd.DataFrame({'Batch Size': batch_sizes, 'GPU Emulator': GPU_emulator, 'CPU Emulator': CPU_emulator, 'GPU Properties': GPU_properties, 'CPU Properties': CPU_properties}).to_csv('performance_results.csv', index=False)


plt.loglog(batch_sizes, batch_sizes, label='Sequential HeFESTo', color = 'gray', linestyle='-', alpha=0.5)
plt.loglog(batch_sizes, batch_sizes*0.8, label='Sequential HeFESTo on Four Cores', color = 'lightgray', linestyle='-', alpha=0.5)

plt.loglog(batch_sizes, GPU_emulator, label='GPU Emulator', color = 'forestgreen', linestyle='-')
plt.loglog(batch_sizes, CPU_emulator, label='CPU Emulator', color = 'darkorange', linestyle='-')
plt.loglog(batch_sizes, GPU_properties, label='GPU Properties', color = 'forestgreen', linestyle='--')
plt.loglog(batch_sizes, CPU_properties, label='CPU Properties', color = 'darkorange', linestyle='--')
#plt.loglog(batch_sizes, NN_emulator, label='NN Emulator', color = 'teal', linestyle='-')
#plt.loglog(batch_sizes, NN_properties, label='NN Properties', color = 'teal', linestyle='--')

plt.xlabel('Number of Assemblages (N)')
plt.ylabel('Wall Time (s)')
plt.grid(True)
plt.title('HeFESTo Emulator Performance Comparison: Total Time')
plt.legend()
plt.savefig('performance_results.png', dpi=300)
plt.show()

plt.plot(batch_sizes, np.array(batch_sizes)/np.array(GPU_emulator), label='GPU Emulator', color = 'forestgreen', linestyle='-')
plt.plot(batch_sizes, np.array(batch_sizes)/np.array(CPU_emulator), label='CPU Emulator', color = 'darkorange', linestyle='-')
plt.plot(batch_sizes, np.array(batch_sizes)/np.array(GPU_properties), label='GPU Properties', color = 'forestgreen', linestyle='--')
plt.plot(batch_sizes, np.array(batch_sizes)/np.array(CPU_properties), label='CPU Properties', color = 'darkorange', linestyle='--')
#plt.plot(batch_sizes, np.array(batch_sizes)/np.array(NN_emulator), label='NN Emulator', color = 'teal', linestyle='-')
#plt.plot(batch_sizes, np.array(batch_sizes)/np.array(NN_properties), label='NN Properties', color = 'teal', linestyle='--')
plt.xlabel('Batch Size')
plt.ylabel('Assemblages per second')
plt.grid(True)
plt.title('Emulator Performance Comparison: Assemblages per Second')
plt.legend()
plt.show()

burnman_rate = batch_sizes*(10/4000) #seconds

plt.loglog(batch_sizes, GPU_emulator/batch_sizes, label='GPU Emulator speedup', color = 'forestgreen', linestyle='-')
plt.loglog(batch_sizes, CPU_emulator/batch_sizes, label='CPU Emulator speedup', color = 'darkorange', linestyle='-')
plt.loglog(batch_sizes, GPU_properties/burnman_rate, label='Burnman GPU speedup', color = 'forestgreen', linestyle='--')
plt.loglog(batch_sizes, CPU_properties/burnman_rate, label='Burnman CPU speedup', color = 'darkorange', linestyle='--')
#plt.loglog(batch_sizes, NN_emulator, label='NN Emulator', color = 'teal', linestyle='-')
#plt.loglog(batch_sizes, NN_properties, label='NN Properties', color = 'teal', linestyle='--')

plt.xlabel('Number of Assemblages (N)')
plt.ylabel('Relative Speed Up: nGibbsMin/(Hefesto | Burnman)')
plt.grid(True)
plt.title('Relative Speedup of nGibbsMin')
plt.legend()
plt.savefig('performance_results_relative.png', dpi=300)
plt.show()