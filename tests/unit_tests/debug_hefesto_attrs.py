from importlib import import_module
import sys
import torch
from pathlib import Path

# Ensure repository and src are on sys.path (mimic test file setup)
REPO_ROOT = Path(__file__).resolve().parents[2]
print(REPO_ROOT)
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
BUILDER_ROOT = SRC_ROOT / "builder"
if str(BUILDER_ROOT) not in sys.path:
    sys.path.insert(0, str(BUILDER_ROOT))
sys.path.insert(0, str(SRC_ROOT / "builder" / "HeFESTo"))
import HeFESTo_functions as hef_funcs


extracted = hef_funcs.extract_bulk_properties_from_simulation_dir(str(SRC_ROOT / "nMELTS" / "engine" / "EOS_arithmetic" / "BENCHMARK"))
component_names = list(extracted["component_names"])
comp_moles_np = extracted["component_moles"]
component_moles = torch.tensor(comp_moles_np, dtype=torch.float32)
from nMELTS.engine.EOS_arithmetic import hefesto_physub as full_physub
full_ctx = full_physub.get_hefesto_physub_context()
aligned_moles = full_ctx.align_component_tensor(component_moles, component_names)

temps = list(extracted.get('T(K)', []))
if not temps:
    temps = [1600.0]*component_moles.shape[0]

# print first row details
i = 0
print('temps[0]=', temps[0])
attrs = full_ctx.compute_component_attributes_at_temperature(float(temps[0]), batch_size=1, device=torch.device('cpu'))
print('molar_mass(ctx)[:10]=', full_ctx.formula_mass_g_mol[:10])
print('aligned_moles[0,:10]=', aligned_moles[0,:10])
print('molar_volume[0,:10]=', attrs['molar_volume'][0,:10])
print('bulk_modulus[0,:10]=', attrs['bulk_modulus'][0,:10])
print('shear_modulus[0,:10]=', attrs['shear_modulus'][0,:10])

# compute total mass and volume
molar_mass = full_ctx.formula_mass_g_mol
mass_components = aligned_moles * molar_mass.unsqueeze(0)
print('total_mass=', mass_components[0].sum().item())
print('total_volume=', (aligned_moles*attrs['molar_volume']).sum(dim=1)[0].item())
print('density=', (mass_components.sum(dim=1)/(aligned_moles*attrs['molar_volume']).sum(dim=1))[0].item())

# now replicate original pipeline: build phases via full module helper
from nMELTS.engine.EOS_arithmetic import hefesto_physub as full
# use internal helper to build phase states? It's internal; instead run compare function to get full's first-row density
param_records = full.load_hefesto_parameter_directory(use_cache=True)
result = full.compare_physub_against_benchmark_directory(full.DEFAULT_PARAMETER_DIR.parent/'BENCHMARK', param_records=param_records, hefesto_context=full_ctx, verbose=False)
print('compare result mean_errors sample:', result.mean_errors)
print('done')
