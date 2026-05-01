from importlib import import_module
from pathlib import Path
import sys

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

import numpy as np
import torch

htp = import_module('nMELTS.engine.EOS_arithmetic.hefesto_thermal_properties')
physub = import_module('nMELTS.engine.EOS_arithmetic.hefesto_physub')
hef_funcs = import_module('HeFESTo.HeFESTo_functions')

bench = SRC_ROOT / 'nMELTS' / 'engine' / 'EOS_arithmetic' / 'BENCHMARK'
extracted = hef_funcs.extract_bulk_properties_from_simulation_dir(str(bench))
component_names = list(extracted['component_names'])
component_moles_row = [float(v) for v in extracted['component_moles'][0]]
T = float(extracted.get('T(K)', [1600.0])[0])

print('T=', T)
from nMELTS.config.constants import HEFESTO_ABBREVIATION_TO_SHORT_NAMES
param_records = physub.load_hefesto_parameter_directory(use_cache=True)

# Build reverse lookup: short name -> list of abbreviations
rev = {}
for abbr, short in HEFESTO_ABBREVIATION_TO_SHORT_NAMES.items():
    rev.setdefault(short.lower(), []).append(abbr)

# Find non-zero components in first benchmark row
nonzero = [(name, amt) for name, amt in zip(component_names, component_moles_row) if float(amt) > 0.0]
print('nonzero count', len(nonzero))
for name, amt in nonzero:
    rec = param_records.get(name)
    if rec is None:
        # try reverse lookup by short mineral name
        aliases = rev.get(str(name).lower(), ())
        for a in aliases:
            if a in param_records:
                rec = param_records[a]
                break
    if rec is None:
        rec = param_records.get(name.lower())
    if rec is None:
        print('no record for', name)
        continue
    fm = rec.value('formula_mass_g_mol')
    v0 = rec.value('v0_cm3_mol')
    k0 = rec.value('k0_gpa')
    theta0 = rec.value('theta0_k')
    fu = rec.value('formula_units_per_cell')
    atoms = rec.value('atoms_per_formula_unit')
    print('---', name, 'amt', amt, 'fm', fm, 'fu', fu)
    # compute via component_thermodynamic_state (uses current volume later, but we'll use v0)
    state = htp.compute_component_thermodynamic_state(
        temperature=T,
        volume=v0,
        reference_volume=v0,
        bulk_modulus=k0,
        atoms_per_formula=atoms,
        formula_units_per_cell=rec.value('formula_units_per_cell'),
        debye_temp=theta0,
    )
    print('therm_state cp (J/mol/K)=', state.heat_capacity_p, 'cv=', state.heat_capacity_v)
    # compute via compute_component_heat_capacity_p directly
    cp = htp.compute_component_heat_capacity_p(temperature=T, atoms_per_formula=atoms, debye_temp=theta0)
    print('cp direct (J/mol/K)=', cp)
    # mass-basis
    print('cp mass-basis from state (J/g/K)=', state.heat_capacity_p / fm)
    print('cp mass-basis direct (J/g/K)=', cp / fm)

print('done')
