from importlib import import_module
from pathlib import Path
import sys

# Robustly locate the repository root by walking up until we find pyproject.toml or README.md
p = Path(__file__).resolve()
repo_root = None
for _ in range(6):
    if (p / 'pyproject.toml').exists() or (p / 'README.md').exists():
        repo_root = p
        break
    p = p.parent
if repo_root is None:
    repo_root = Path(__file__).resolve().parents[3]
REPO_ROOT = repo_root
SRC_ROOT = REPO_ROOT / 'src'
BUILDER_ROOT = SRC_ROOT / 'builder'
for p in (str(REPO_ROOT), str(SRC_ROOT), str(BUILDER_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

he = import_module('nMELTS.engine.EOS_arithmetic.hefesto_physub')
no = import_module('nMELTS.engine.EOS_arithmetic.hefesto_physub_no_thermal')
hef_funcs = import_module('HeFESTo.HeFESTo_functions')

bench = SRC_ROOT / 'nMELTS' / 'engine' / 'EOS_arithmetic' / 'BENCHMARK'
extracted = hef_funcs.extract_bulk_properties_from_simulation_dir(str(bench))
component_names = list(extracted['component_names'])
component_moles = torch.tensor(extracted['component_moles'], dtype=torch.float32)
temps = list(extracted.get('T(K)', []))
print('n_comps', len(component_names))

# pick first row
row = 0
mol_m = torch.tensor([float(v) for v in extracted['component_moles'][row]], dtype=torch.float32)
moles_row = torch.tensor([extracted['component_moles'][row]], dtype=torch.float32)
T = float(temps[row]) if temps else 1600.0

full_ctx = he.get_hefesto_physub_context()
# align
aligned = full_ctx.align_component_tensor(moles_row, component_names)
print('aligned sum', aligned.sum().item())
# full attributes (batch=1)
attrs_full = full_ctx.compute_component_attributes_at_temperature(T, batch_size=1, device=torch.device('cpu'))
for name in ('molar_volume','bulk_modulus','shear_modulus'):
    v = attrs_full[name][0]
    print(name, 'first10', v[:10].tolist())

# call full compute with attributes prepared per-row
full_out, full_names = he.compute_physub_bulk_matrix(aligned, full_ctx.formula_mass_g_mol, attrs_full)
print('full_out[0]')
for n,val in zip(full_names, full_out[0].tolist()):
    print(n, val)

# call simplified: with hefesto_context and temperature (no precomputed attrs)
print('component_moles.shape', aligned.shape)
print('len(component_names)', len(component_names))
print('context component count', len(full_ctx.component_names))
no_out, no_names = no.compute_physub_bulk_matrix(
    aligned,
    full_ctx.formula_mass_g_mol,
    component_attributes=None,
    component_names=component_names,
    hefesto_context=full_ctx,
    temperature_k=T,
)
print('no_out[0]')
for n, val in zip(no_names, no_out[0].tolist()):
    print(n, val)

# show difference
print('diff')
for n, f, s in zip(full_names, full_out[0].tolist(), no_out[0].tolist()):
    print(n, f - s)
