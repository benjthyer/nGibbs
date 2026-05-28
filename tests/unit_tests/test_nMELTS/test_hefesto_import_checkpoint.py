from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / 'src'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from builder.HeFESTo import HeFESTo_functions as hef


class FakeIndexer:
    def __init__(self) -> None:
        self.database_headers = [
            'P(GPa)',
            'T(K)',
            'rho(g/cm^3)',
            'mass (gm)',
            'VS(km/s)',
            'VP(km/s)',
            'H(kJ/g)',
            'cp(J/g/K)',
            'S(J/g/K)',
            'KS(GPa)',
            'alpha(1e5_K^-1)',
            'Bulk_SiO2',
            'Bulk_MgO',
            'Bulk_FeO',
            'Bulk_Fe2O3',
            'Bulk_CaO',
            'Bulk_Al2O3',
            'Bulk_Na2O',
            'Bulk_Cr2O3',
            'Elem_Si',
            'Elem_Mg',
            'Elem_Fe',
            'Elem_Ca',
            'Elem_Al',
            'Elem_Na',
            'Elem_Cr',
            'Elem_O',
            'PhaseA_rho',
            'PhaseA_vol',
            'PhaseA_mass',
            'PhaseA_COMPA',
            'PhaseA_COMPB',
            'PhaseA_total_moles',
            'PhaseA_moles',
            'PhaseA_phase_moles',
            'PhaseA_total_moles_2',
        ]
        self.MELTS_indices = {
            'System_main': {
                'P(GPa)': 0,
                'T(K)': 1,
                'rho(g/cm^3)': 2,
                'mass (gm)': 3,
                'VS(km/s)': 4,
                'VP(km/s)': 5,
                'H(kJ/g)': 6,
                'cp(J/g/K)': 7,
                'S(J/g/K)': 8,
                'KS(GPa)': 9,
                'alpha(1e5_K^-1)': 10,
            },
            'Bulk_comp': {
                'SiO2': 11,
                'MgO': 12,
                'FeO': 13,
                'Fe2O3': 14,
                'CaO': 15,
                'Al2O3': 16,
                'Na2O': 17,
                'Cr2O3': 18,
            },
            'Bulk_comp_elements': {
                'Si': 19,
                'Mg': 20,
                'Fe': 21,
                'Ca': 22,
                'Al': 23,
                'Na': 24,
                'Cr': 25,
                'O': 26,
            },
            'PhaseA': {
                'rho (gm/cc)': 27,
                'V (cc)': 28,
                'mass (gm)': 29,
                'COMPA': 30,
                'COMPB': 31,
                'total (moles)': 32,
                'moles': 33,
                'phase moles': 34,
                'total moles': 35,
            },
        }
        self.label_names = ['COMPA', 'COMPB']
        self.label_indices = {'PhaseA': [0, 1]}
        self.phaseToCompMap = np.array([[1.0, 1.0]], dtype=float)

    def get_max_index(self) -> int:
        return len(self.database_headers) - 1


def _make_simulation_tree(root: Path, count: int) -> None:
    for sim_id in range(1, count + 1):
        sim_dir = root / f'Simulation{sim_id}'
        sim_dir.mkdir(parents=True, exist_ok=True)
        for name in ['control', 'fort.56', 'fort.61', 'fort.68', 'fort.99']:
            (sim_dir / name).write_text('stub\n', encoding='utf-8')


def test_import_heuristic_checkpoint_resume_and_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace_dir = tmp_path / 'workspace'
    workspace_dir.mkdir()
    _make_simulation_tree(workspace_dir, 130)

    output_csv = tmp_path / 'hefesto_output.csv'
    phase_change_csv = tmp_path / 'hefesto_phase_changes.csv'
    indexer = FakeIndexer()

    def fake_parse_control_file(path: str):
        return (
            {'Si': 1.0, 'Mg': 2.0, 'Fe': 3.0, 'Ca': 4.0, 'Al': 5.0, 'Na': 6.0, 'Cr': 7.0, 'O': 8.0},
            {},
        )

    def fake_compute_bulk_from_elements(element_moles):
        return (
            {
                'SiO2': 11.0,
                'MgO': 12.0,
                'FeO': 13.0,
                'Fe2O3': 14.0,
                'CaO': 15.0,
                'Al2O3': 16.0,
                'Na2O': 17.0,
                'Cr2O3': 18.0,
            },
            {key: float(value) for key, value in element_moles.items()},
            99.0,
        )

    def fake_parse_fort56(path: str) -> pd.DataFrame:
        sim_id = int(Path(path).parent.name.removeprefix('Simulation'))
        base = float(sim_id)
        return pd.DataFrame(
            {
                'P(GPa)': [base],
                'T(K)': [base + 1000.0],
                'rho(g/cm^3)': [3.0],
                'VS(km/s)': [4.0],
                'VP(km/s)': [5.0],
                'H(kJ/g)': [6.0],
                'cp(J/g/K)': [7.0],
                'S(J/g/K)': [8.0],
                'KS(GPa)': [9.0],
                'alpha(1e5_K^-1)': [10.0],
            }
        )

    def fake_safe_read_ws_table(path: str, skiprows: int = 0) -> pd.DataFrame:
        name = Path(path).name
        if name == 'fort.61':
            return pd.DataFrame({'rhPhaseA': [2.0]})
        if name == 'fort.68':
            return pd.DataFrame({'volPhaseA': [5.0]})
        if name == 'fort.99':
            return pd.DataFrame(
                {
                    'skip1': [0.0],
                    'skip2': [0.0],
                    'skip3': [0.0],
                    'compa': [1.0],
                    'compb': [2.0],
                    'tail1': [0.0],
                    'tail2': [0.0],
                }
            )
        raise AssertionError(f'Unexpected ws table path: {path}')

    monkeypatch.setattr(hef, '_parse_control_file', fake_parse_control_file)
    monkeypatch.setattr(hef, '_compute_bulk_from_elements', fake_compute_bulk_from_elements)
    monkeypatch.setattr(hef, '_parse_fort56', fake_parse_fort56)
    monkeypatch.setattr(hef, '_safe_read_ws_table', fake_safe_read_ws_table)
    monkeypatch.setattr(hef, '_resolve_component_name_from_abbr', lambda abbr: str(abbr).strip().upper())
    monkeypatch.setattr(hef, '_resolve_phase_name_from_abbr', lambda abbr: 'PhaseA')
    monkeypatch.setattr(hef, '_resolve_component_phase', lambda **kwargs: 'PhaseA')
    monkeypatch.setattr(hef, '_build_reverse_component_phase_map', lambda: {})

    real_write_block = hef._write_block_to_csv
    write_calls = {'count': 0}

    def flaky_write_block_to_csv(dataname, headers, block):
        write_calls['count'] += 1
        if write_calls['count'] == 2:
            raise RuntimeError('simulated interruption during flush')
        return real_write_block(dataname, headers, block)

    monkeypatch.setattr(hef, '_write_block_to_csv', flaky_write_block_to_csv)

    checkpoint_path = Path(hef._checkpoint_path_for_csv(str(output_csv)))

    with pytest.raises(RuntimeError, match='simulated interruption during flush'):
        hef.import_HeFESTo_components(
            str(workspace_dir),
            indexer,
            dataname=str(output_csv),
            phase_change_dataname=str(phase_change_csv),
        )

    assert checkpoint_path.exists()
    assert checkpoint_path.read_text(encoding='utf-8').strip() == '128'

    partial_df = pd.read_csv(output_csv)
    assert len(partial_df) == 128
    assert partial_df['P(GPa)'].tolist() == list(range(1, 129))

    def stable_write_block_to_csv(dataname, headers, block):
        return real_write_block(dataname, headers, block)

    monkeypatch.setattr(hef, '_write_block_to_csv', stable_write_block_to_csv)

    passed_ids, malformed_ids, empty_ids = hef.import_HeFESTo_components(
        str(workspace_dir),
        indexer,
        dataname=str(output_csv),
        phase_change_dataname=str(phase_change_csv),
    )

    assert len(passed_ids) == 2
    assert len(malformed_ids) == 0
    assert len(empty_ids) == 0
    assert not checkpoint_path.exists()

    final_df = pd.read_csv(output_csv)
    assert len(final_df) == 130
    assert final_df['P(GPa)'].tolist() == list(range(1, 131))
    assert phase_change_csv.exists()