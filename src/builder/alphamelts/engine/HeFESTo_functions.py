"""
HeFESTo workspace parser utilities.

Parses SimulationN folders and compiles HeFESTo outputs into one CSV table
matching a DatasetIndexer header layout.
"""

import os
import re
import shutil
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from nMELTS.config.constants import (
	COMPOSITIONAL_COMPONENTS_IN_PHASES_HEFESTO,
	HEFESTO_ABBREVIATION_TO_SHORT_NAMES,
	get_oxide_molar_mass,
)


PHASE_ABBREVIATION_OVERRIDES: Dict[str, str] = {
	'c2c': 'hp-clinopyroxene',
	'il': 'akimotoite',
	'pv': 'bridgmanite',
	'mw': 'ferropericlase',
	'fea': 'iron',
	'feg': 'iron',
	'fee': 'iron',
}

COMPONENT_ABBREVIATION_OVERRIDES: Dict[str, str] = {
	'mgil': 'mg-akimotoite',
	'feil': 'fe-akimotoite',
	'mgpv': 'mg-bridgmanite',
	'fepv': 'fe-bridgmanite',
	'alpv': 'al-bridgmanite',
	'hepv': 'ferric-bridgmanite',
	'hlpv': 'ferric-bridgmanite-ls',
	'fapv': 'ferric-al-bridgmanite',
	'crpv': 'cr-bridgmanite',
	'smag': 'magnetite',
	'fea': 'alpha-iron',
	'feg': 'gamma-iron',
	'fee': 'epsilon-iron',
	'mgc2': 'hp-clinoenstatite',
	'fec2': 'hp-clinoferrosilite',
}


EnsembleLocation = None


def _clean_workspace(workspace_dir: str) -> None:
	if os.path.exists(workspace_dir):
		for item in os.listdir(workspace_dir):
			item_path = os.path.join(workspace_dir, item)
			if os.path.isdir(item_path):
				shutil.rmtree(item_path)
			else:
				os.remove(item_path)


def _normalize_element_label(label: str) -> str:
	text = str(label).strip()
	if not text:
		return text
	if text.upper() == 'O':
		return 'O'
	return text[0].upper() + text[1:].lower()


def _build_control_lines(
	template_lines: List[str],
	element_values: Dict[str, float],
	run_code: List,
) -> List[str]:
	if len(template_lines) == 0:
		raise ValueError('Control template file is empty')

	lines = list(template_lines)
	lines[0] = ','.join(str(value) for value in run_code)

	oxides_start = None
	for i, line in enumerate(lines):
		if line.strip().lower() == 'oxides':
			oxides_start = i + 1
			break
	if oxides_start is None:
		raise ValueError("No 'oxides' block found in control template")

	for i in range(oxides_start, len(lines)):
		stripped = lines[i].strip()
		if not stripped:
			continue
		if stripped.lower().startswith('phase '):
			break
		if ',' in stripped:
			continue

		parts = stripped.split()
		if len(parts) < 3:
			continue

		element = _normalize_element_label(parts[0])
		if element not in element_values:
			continue

		value = float(element_values[element])
		value_text = f'{value:.5f}'
		third_col = parts[3] if len(parts) >= 4 else '0'
		lines[i] = f'{element:<2} {value_text:>12} {value_text:>11} {third_col}'

	return lines


def forward_HeFESTo(input_array, keys, run_code, EnsembleLocation=EnsembleLocation):
	"""
	Create and execute an ensemble of HeFESTo simulations.

	Parameters
	----------
	input_array : np.ndarray
		Array of elemental conditions, shape (n_simulations, n_elements).
	keys : np.ndarray or list
		Element labels corresponding to columns in input_array.
	run_code : list
		Comma-separated first-line control values written to each control file.
	EnsembleLocation : str
		Directory where SimulationN folders and runall.sh are written.
	"""
	if EnsembleLocation is None:
		raise ValueError('EnsembleLocation must be set for forward_HeFESTo()')

	input_array = np.asarray(input_array)
	if input_array.ndim != 2:
		raise ValueError('input_array must be a 2D array')

	if input_array.shape[1] != len(keys):
		raise IndexError("Condition columns don't match keys")

	if not isinstance(run_code, (list, tuple, np.ndarray)):
		raise TypeError('run_code must be list-like')

	normalized_keys = [_normalize_element_label(key) for key in keys]
	run_code_list = list(run_code)

	control_template = os.path.join(Path(__file__).parent.parent.absolute(), 'batch', 'control')
	if not os.path.exists(control_template):
		raise FileNotFoundError(f'Missing HeFESTo control template: {control_template}')

	with open(control_template, 'r', encoding='utf-8', errors='ignore') as handle:
		template_lines = [line.rstrip('\n') for line in handle]

	_clean_workspace(EnsembleLocation)
	os.makedirs(EnsembleLocation, exist_ok=True)

	runall_lines: List[str] = []
	for i in range(input_array.shape[0]):
		sim_dir = os.path.join(EnsembleLocation, f'Simulation{i}')
		os.makedirs(sim_dir, exist_ok=True)

		control_path = os.path.join(sim_dir, 'control')
		shutil.copy(control_template, control_path)

		element_values = {
			normalized_keys[j]: float(input_array[i, j])
			for j in range(input_array.shape[1])
		}
		updated_control_lines = _build_control_lines(template_lines, element_values, run_code_list)

		with open(control_path, 'w', encoding='utf-8') as handle:
			handle.write('\n'.join(updated_control_lines) + '\n')

		runall_lines.append(f'cd "{sim_dir}" ; HeFESTo')

	runall_path = os.path.join(EnsembleLocation, 'runall.sh')
	with open(runall_path, 'w', encoding='utf-8') as handle:
		handle.write('\n'.join(runall_lines) + '\n')

	os.system('cd "' + EnsembleLocation + '"; parallel < runall.sh; cd -')


def _safe_read_ws_table(path: str, skiprows: int = 0) -> pd.DataFrame:
	return pd.read_csv(path, sep=r'\s+', engine='python', skiprows=skiprows)


def _extract_sim_id(sim_name: str) -> Optional[int]:
	match = re.search(r'simulation(\d+)$', sim_name.lower())
	if match is None:
		return None
	return int(match.group(1))


def _list_simulation_dirs(workspace_dir: str) -> List[Tuple[int, str]]:
	sims: List[Tuple[int, str]] = []
	for name in os.listdir(workspace_dir):
		full = os.path.join(workspace_dir, name)
		if not os.path.isdir(full):
			continue
		sim_id = _extract_sim_id(name)
		if sim_id is None:
			continue
		sims.append((sim_id, full))
	sims.sort(key=lambda x: x[0])
	return sims


def _resolve_phase_name_from_abbr(phase_abbr: str) -> str:
	if phase_abbr in PHASE_ABBREVIATION_OVERRIDES:
		return PHASE_ABBREVIATION_OVERRIDES[phase_abbr]
	return HEFESTO_ABBREVIATION_TO_SHORT_NAMES.get(phase_abbr, phase_abbr)


def _resolve_component_name_from_abbr(component_abbr: str) -> str:
	if component_abbr in COMPONENT_ABBREVIATION_OVERRIDES:
		return COMPONENT_ABBREVIATION_OVERRIDES[component_abbr]
	return HEFESTO_ABBREVIATION_TO_SHORT_NAMES.get(component_abbr, component_abbr)


def _build_reverse_component_phase_map() -> Dict[str, List[str]]:
	reverse_map: Dict[str, List[str]] = {}
	for phase_name, comp_list in COMPOSITIONAL_COMPONENTS_IN_PHASES_HEFESTO.items():
		if phase_name in {'System_main', 'Bulk_comp', 'Bulk_comp_elements'}:
			continue
		for component in comp_list:
			reverse_map.setdefault(component, []).append(phase_name)
	return reverse_map


def _parse_control_file(control_path: str) -> Tuple[Dict[str, float], Dict[str, str]]:
	with open(control_path, 'r', encoding='utf-8', errors='ignore') as handle:
		lines = [line.rstrip('\n') for line in handle]

	element_moles: Dict[str, float] = {}
	control_component_to_phase_abbr: Dict[str, str] = {}

	oxides_start = None
	for i, line in enumerate(lines):
		if line.strip().lower() == 'oxides':
			oxides_start = i + 1
			break
	if oxides_start is None:
		raise ValueError(f"No 'oxides' block found in {control_path}")

	for i in range(oxides_start, len(lines)):
		stripped = lines[i].strip()
		if not stripped:
			continue
		if stripped.lower().startswith('phase '):
			break
		if ',' in stripped:
			continue
		parts = stripped.split()
		if len(parts) < 2:
			continue
		symbol = parts[0]
		if symbol.upper() == 'O':
			symbol = 'O'
		try:
			value = float(parts[1])
		except ValueError:
			continue
		element_moles[symbol] = value

	active_phase_abbr: Optional[str] = None
	expect_flag_line = False
	for line in lines:
		stripped = line.strip()
		if not stripped:
			continue
		if stripped.lower().startswith('phase '):
			fields = stripped.split()
			if len(fields) >= 2:
				active_phase_abbr = fields[1].strip()
				expect_flag_line = True
			continue
		if active_phase_abbr is None:
			continue
		if expect_flag_line:
			expect_flag_line = False
			continue
		component_abbr = stripped.split()[0]
		control_component_to_phase_abbr[component_abbr] = active_phase_abbr

	return element_moles, control_component_to_phase_abbr


def _compute_bulk_from_elements(element_moles: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float], float]:
	required = ['Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr', 'O']
	missing = [el for el in required if el not in element_moles]
	if missing:
		raise ValueError(f'Missing required element(s) in control oxides block: {missing}')

	n_si = float(element_moles['Si'])
	n_mg = float(element_moles['Mg'])
	n_fe = float(element_moles['Fe'])
	n_ca = float(element_moles['Ca'])
	n_al = float(element_moles['Al'])
	n_na = float(element_moles['Na'])
	n_cr = float(element_moles['Cr'])
	n_o = float(element_moles['O'])

	# Oxygen tied up in non-iron oxides.
	o_non_fe = 2.0 * n_si + n_mg + n_ca + 1.5 * n_al + 0.5 * n_na + 1.5 * n_cr
	o_for_fe = n_o - o_non_fe

	# Fe2O3 moles are the oxygen excess relative to all-FeO partitioning.
	n_fe2o3 = max(0.0, o_for_fe - n_fe)
	n_feo = max(0.0, n_fe - 2.0 * n_fe2o3)

	if n_feo + 2.0 * n_fe2o3 > n_fe + 1e-9:
		n_fe2o3 = 0.0
		n_feo = n_fe

	oxide_moles = {
		'SiO2': n_si,
		'MgO': n_mg,
		'FeO': n_feo,
		'Fe2O3': n_fe2o3,
		'CaO': n_ca,
		'Al2O3': n_al / 2.0,
		'Na2O': n_na / 2.0,
		'Cr2O3': n_cr / 2.0,
	}

	oxide_masses = {ox: moles * get_oxide_molar_mass(ox) for ox, moles in oxide_moles.items()}
	system_mass = float(np.sum(list(oxide_masses.values())))
	if system_mass <= 0:
		raise ValueError('Computed non-positive system mass from control oxides block')

	bulk_comp_wt = {ox: 100.0 * mass / system_mass for ox, mass in oxide_masses.items()}
	bulk_elements = {el: float(element_moles[el]) for el in required}
	return bulk_comp_wt, bulk_elements, system_mass


def _resolve_component_phase(
	component_abbr: str,
	component_name: str,
	reverse_component_phase_map: Dict[str, List[str]],
	control_component_to_phase_abbr: Dict[str, str],
) -> Optional[str]:
	candidates = reverse_component_phase_map.get(component_name, [])

	if len(candidates) == 1:
		return candidates[0]

	if component_abbr in control_component_to_phase_abbr:
		phase_abbr = control_component_to_phase_abbr[component_abbr]
		phase_name = _resolve_phase_name_from_abbr(phase_abbr)
		if len(candidates) == 0:
			return phase_name
		if phase_name in candidates:
			return phase_name

	if len(candidates) > 0:
		return candidates[0]

	return None


def _safe_assign(
	table: np.ndarray,
	indexer,
	phase: str,
	component: str,
	values: np.ndarray,
	add: bool = False,
) -> None:
	phase_map = indexer.MELTS_indices.get(phase, None)
	if phase_map is None:
		return
	col_idx = phase_map.get(component, None)
	if col_idx is None:
		return
	if add:
		table[:, col_idx] += values
	else:
		table[:, col_idx] = values


def _parse_fort56(path: str) -> pd.DataFrame:
	return _safe_read_ws_table(path, skiprows=1)


def _extract_phase_series_from_prefixed_columns(
	df: pd.DataFrame,
	prefix: str,
) -> Dict[str, np.ndarray]:
	out: Dict[str, np.ndarray] = {}
	for col in df.columns:
		col_str = str(col)
		if not col_str.startswith(prefix):
			continue
		abbr = col_str[len(prefix):].strip()
		phase_name = _resolve_phase_name_from_abbr(abbr)
		values = pd.to_numeric(df[col], errors='coerce').fillna(0.0).to_numpy(dtype=float)
		out.setdefault(phase_name, np.zeros_like(values))
		out[phase_name] += values
	return out


def _write_block_to_csv(dataname: str, headers: List[str], block: np.ndarray) -> None:
	if not os.path.exists(dataname):
		pd.DataFrame(columns=headers).to_csv(dataname, index=False)
	pd.DataFrame(block, columns=headers).to_csv(dataname, mode='a', header=False, index=False)


def _ensure_existing_csv_headers_match(dataname: str, expected_headers: List[str]) -> None:
	if not os.path.exists(dataname):
		return

	existing_headers = list(pd.read_csv(dataname, nrows=0).columns)
	if existing_headers != list(expected_headers):
		raise ValueError(
			'Existing output CSV headers do not match indexer.database_headers. '
			'Refuse to append due to schema mismatch.'
		)


def import_HeFESTo_components(workspace_dir, indexer, dataname='DefaultHeFESTostorage.csv'):
	"""
	Parse HeFESTo SimulationN directories into one CSV table.

	File usage per simulation:
	- control: bulk element inventory (and phase/component abbreviation blocks)
	- fort.56: System_main intensive/thermo attributes
	- fort.61: phase densities (rh* columns)
	- fort.68: phase volumes (vol* columns)
	- fort.99: phase-component molar abundances

	Parameters
	----------
	workspace_dir : str
		Directory containing SimulationN folders.
	indexer : DatasetIndexer
		DatasetIndexer with MELTS_indices/database_headers defining output schema.
	dataname : str, default='DefaultHeFESTostorage.csv'
		Output CSV path.

	Returns
	-------
	np.ndarray
		Unique failing Simulation IDs.
	"""
	faultIDs: List[int] = []
	reverse_component_phase_map = _build_reverse_component_phase_map()
	sim_dirs = _list_simulation_dirs(workspace_dir)
	_ensure_existing_csv_headers_match(dataname, indexer.database_headers)

	if not os.path.exists(dataname):
		pd.DataFrame(columns=indexer.database_headers).to_csv(dataname, index=False)

	for sim_id, sim_dir in sim_dirs:
		control_path = os.path.join(sim_dir, 'control')
		fort56_path = os.path.join(sim_dir, 'fort.56')
		fort61_path = os.path.join(sim_dir, 'fort.61')
		fort68_path = os.path.join(sim_dir, 'fort.68')
		fort99_path = os.path.join(sim_dir, 'fort.99')

		try:
			for req_path in [control_path, fort56_path, fort61_path, fort68_path, fort99_path]:
				if not os.path.exists(req_path):
					raise FileNotFoundError(f'Missing required file: {req_path}')

			element_moles, control_component_to_phase_abbr = _parse_control_file(control_path)
			bulk_comp_wt, bulk_elements, system_mass = _compute_bulk_from_elements(element_moles)

			sys_df = _parse_fort56(fort56_path)
			rho_df = _safe_read_ws_table(fort61_path, skiprows=0)
			vol_df = _safe_read_ws_table(fort68_path, skiprows=0)
			comp_df = _safe_read_ws_table(fort99_path, skiprows=0)

			nrows = min(len(sys_df), len(rho_df), len(vol_df), len(comp_df))
			if nrows <= 0:
				raise ValueError('No rows found across required HeFESTo tables')

			sys_df = sys_df.iloc[:nrows].reset_index(drop=True)
			rho_df = rho_df.iloc[:nrows].reset_index(drop=True)
			vol_df = vol_df.iloc[:nrows].reset_index(drop=True)
			comp_df = comp_df.iloc[:nrows].reset_index(drop=True)

			out = np.zeros((nrows, indexer.get_max_index() + 1), dtype=float)

			# System_main from fort.56
			_safe_assign(out, indexer, 'System_main', 'P(GPa)', pd.to_numeric(sys_df.get('P(GPa)'), errors='coerce').fillna(0.0).to_numpy(dtype=float))
			_safe_assign(out, indexer, 'System_main', 'T(K)', pd.to_numeric(sys_df.get('T(K)'), errors='coerce').fillna(0.0).to_numpy(dtype=float))
			_safe_assign(out, indexer, 'System_main', 'rho(g/cm^3)', pd.to_numeric(sys_df.get('rho(g/cm^3)'), errors='coerce').fillna(0.0).to_numpy(dtype=float))
			_safe_assign(out, indexer, 'System_main', 'mass (gm)', np.full(nrows, system_mass, dtype=float))
			_safe_assign(out, indexer, 'System_main', 'VS(km/s)', pd.to_numeric(sys_df.get('VS(km/s)'), errors='coerce').fillna(0.0).to_numpy(dtype=float))
			_safe_assign(out, indexer, 'System_main', 'VP(km/s)', pd.to_numeric(sys_df.get('VP(km/s)'), errors='coerce').fillna(0.0).to_numpy(dtype=float))
			_safe_assign(out, indexer, 'System_main', 'H(kJ/g)', pd.to_numeric(sys_df.get('H(kJ/g)'), errors='coerce').fillna(0.0).to_numpy(dtype=float))
			_safe_assign(out, indexer, 'System_main', 'cp(J/g/K)', pd.to_numeric(sys_df.get('cp(J/g/K)'), errors='coerce').fillna(0.0).to_numpy(dtype=float))
			_safe_assign(out, indexer, 'System_main', 'S(J/g/K)', pd.to_numeric(sys_df.get('S(J/g/K)'), errors='coerce').fillna(0.0).to_numpy(dtype=float))
			_safe_assign(out, indexer, 'System_main', 'KS(GPa)', pd.to_numeric(sys_df.get('KS(GPa)'), errors='coerce').fillna(0.0).to_numpy(dtype=float))
			_safe_assign(out, indexer, 'System_main', 'alpha(1e5_K^-1)', pd.to_numeric(sys_df.get('alpha(1e5_K^-1)'), errors='coerce').fillna(0.0).to_numpy(dtype=float))

			# Bulk_comp and Bulk_comp_elements from control-derived chemistry.
			for oxide, wt in bulk_comp_wt.items():
				_safe_assign(out, indexer, 'Bulk_comp', oxide, np.full(nrows, wt, dtype=float))

			for element, value in bulk_elements.items():
				_safe_assign(out, indexer, 'Bulk_comp_elements', element, np.full(nrows, value, dtype=float))

			# Phase density and volume tables.
			rho_by_phase = _extract_phase_series_from_prefixed_columns(rho_df, 'rh')
			vol_by_phase = _extract_phase_series_from_prefixed_columns(vol_df, 'vol')

			for phase_name, rho_values in rho_by_phase.items():
				_safe_assign(out, indexer, phase_name, 'rho (gm/cc)', rho_values)

			for phase_name, vol_values in vol_by_phase.items():
				_safe_assign(out, indexer, phase_name, 'V (cc)', vol_values)

			# Rebuild phase mass from rho*V where available.
			for phase_name in indexer.MELTS_indices.keys():
				if phase_name in {'System_main', 'Bulk_comp', 'Bulk_comp_elements'}:
					continue
				phase_map = indexer.MELTS_indices[phase_name]
				mass_idx = phase_map.get('mass (gm)')
				vol_idx = phase_map.get('V (cc)')
				rho_idx = phase_map.get('rho (gm/cc)')
				if mass_idx is None or vol_idx is None or rho_idx is None:
					continue
				out[:, mass_idx] = out[:, rho_idx] * out[:, vol_idx]

			# fort.99 components: skip first 3 and last 2 columns.
			if comp_df.shape[1] < 6:
				raise ValueError('fort.99 has insufficient columns to parse components')

			component_cols = list(comp_df.columns)[3:-2]
			for comp_abbr in component_cols:
				comp_abbr_str = str(comp_abbr).strip()
				component_name = _resolve_component_name_from_abbr(comp_abbr_str)
				phase_name = _resolve_component_phase(
					component_abbr=comp_abbr_str,
					component_name=component_name,
					reverse_component_phase_map=reverse_component_phase_map,
					control_component_to_phase_abbr=control_component_to_phase_abbr,
				)
				if phase_name is None:
					continue
				values = pd.to_numeric(comp_df[comp_abbr], errors='coerce').fillna(0.0).to_numpy(dtype=float)
				_safe_assign(out, indexer, phase_name, component_name, values)

			_write_block_to_csv(dataname, indexer.database_headers, out)

		except Exception as exc:
			print(f'Simulation{sim_id} FAILED at {sim_dir}: {type(exc).__name__}: {exc}')
			faultIDs.append(sim_id)

	return np.unique(np.asarray(faultIDs, dtype=int))

