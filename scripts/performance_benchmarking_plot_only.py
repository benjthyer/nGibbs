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

#pd.DataFrame({'Batch Size': batch_sizes, 'GPU Emulator': GPU_emulator, 'CPU Emulator': CPU_emulator, 'GPU Properties': GPU_properties, 'CPU Properties': CPU_properties}).to_csv('performance_results.csv', index=False)

data = pd.read_csv('scripts/performance_results1.csv')
batch_sizes = data['Batch Size']
GPU_emulator = data['GPU Emulator']
CPU_emulator = data['CPU Emulator']
GPU_properties = data['GPU Properties']
CPU_properties = data['CPU Properties']
zeros = CPU_properties == 0

#plt.loglog(batch_sizes, batch_sizes, label='Sequential HeFESTo', color = 'gray', linestyle='-', alpha=0.5)
#plt.loglog(batch_sizes, batch_sizes*0.8, label='Sequential HeFESTo on Four Cores', color = 'lightgray', linestyle='-', alpha=0.5)

plt.loglog(batch_sizes, GPU_emulator, label='GPU Emulator', color = 'forestgreen', linestyle='-')
plt.loglog(batch_sizes, CPU_emulator, label='CPU Emulator', color = 'darkorange', linestyle='-')
plt.loglog(batch_sizes, GPU_properties, label='GPU Properties', color = 'forestgreen', linestyle='--')
plt.loglog(batch_sizes[~zeros], CPU_properties[~zeros], label='CPU Properties', color = 'darkorange', linestyle='--')
#plt.loglog(batch_sizes, NN_emulator, label='NN Emulator', color = 'teal', linestyle='-')
#plt.loglog(batch_sizes, NN_properties, label='NN Properties', color = 'teal', linestyle='--')

plt.xlabel('Number of Assemblages (N)')
plt.ylabel('Wall Time (s)')
plt.xlim(batch_sizes.min(), batch_sizes.max())
plt.minorticks_on()
plt.grid(True, which='major')
plt.grid(True, which='minor', color='lightgrey', linewidth=0.5)
plt.title('HeFESTo Emulator Performance Comparison: Total Time')
plt.legend()
plt.savefig('scripts/performance_results.png', dpi=300)
plt.show()

plt.semilogx(batch_sizes, np.array(batch_sizes)/np.array(GPU_emulator), label='GPU Emulator', color = 'forestgreen', linestyle='-')
plt.semilogx(batch_sizes, np.array(batch_sizes)/np.array(CPU_emulator), label='CPU Emulator', color = 'darkorange', linestyle='-')
plt.semilogx(batch_sizes, np.array(batch_sizes)/np.array(GPU_properties), label='GPU Properties', color = 'forestgreen', linestyle='--')
plt.semilogx(batch_sizes[~zeros], np.array(batch_sizes[~zeros])/np.array(CPU_properties[~zeros]), label='CPU Properties', color = 'darkorange', linestyle='--')
#plt.semilogx(batch_sizes, np.array(batch_sizes)/np.array(NN_emulator), label='NN Emulator', color = 'teal', linestyle='-')
#plt.semilogx(batch_sizes, np.array(batch_sizes)/np.array(NN_properties), label='NN Properties', color = 'teal', linestyle='--')
plt.xlabel('Batch Size')
plt.ylabel('Assemblages per second')
plt.xlim(batch_sizes.min(), batch_sizes.max())
plt.grid(True, which='major')
plt.grid(True, which='minor', color='lightgrey', linewidth=0.5)
plt.title('Emulator Performance Comparison: Assemblages per Second')
plt.legend()
plt.savefig('scripts/performance_results_assemblages_per_second.png', dpi=300)
plt.show()

burnman_rate = batch_sizes*(10/4000) #seconds

plt.loglog(batch_sizes, batch_sizes/GPU_emulator, label='GPU Emulator speedup', color = 'forestgreen', linestyle='-')
plt.loglog(batch_sizes, batch_sizes/CPU_emulator, label='CPU Emulator speedup', color = 'darkorange', linestyle='-')
plt.loglog(batch_sizes, burnman_rate/GPU_properties, label='GPU Properties speedup', color = 'forestgreen', linestyle='--')
plt.loglog(batch_sizes, burnman_rate/CPU_properties, label='CPU Properties speedup', color = 'darkorange', linestyle='--')
plt.hlines(1, batch_sizes.min(), batch_sizes.max(), colors='black', linewidth=2)
plt.xlim(batch_sizes.min(), batch_sizes.max())
#plt.loglog(batch_sizes, NN_emulator, label='NN Emulator', color = 'teal', linestyle='-')
#plt.loglog(batch_sizes, NN_properties, label='NN Properties', color = 'teal', linestyle='--')

plt.xlabel('Number of Assemblages (N)')
plt.ylabel('Relative Speed Up: nGibbsMin/(Hefesto | Burnman)')
plt.grid(True, which='major')
plt.grid(True, which='minor', color='lightgrey', linewidth=0.5)
plt.title('Relative Speedup of nGibbsMin')
plt.legend()
plt.savefig('performance_results_relative.png', dpi=300)
plt.show()