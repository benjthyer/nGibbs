from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nMELTS.engine.EOS_arithmetic import HCOK, Heat, gamset, parset
from nMELTS.engine.EOS_arithmetic.hefesto_physub import HeFESToParameterRecord
from nMELTS.engine.EOS_arithmetic.hefesto_thermal_data import HeFESToThermalModeTensors


def _make_parameter_record() -> HeFESToParameterRecord:
    values = [0.0] * 43
    values[0] = 2.0
    values[1] = 1.0
    values[2] = 100.0
    values[3] = 300.0
    values[4] = -10.0
    values[5] = 12.0
    values[6] = 160.0
    values[7] = 4.0
    values[8] = 0.0
    values[9] = 900.0
    values[10] = 910.0
    values[11] = 920.0
    values[12] = 100.0
    values[13] = 110.0
    values[14] = 120.0
    values[15] = 130.0
    values[16] = 0.10
    values[17] = 140.0
    values[18] = 0.20
    values[19] = 150.0
    values[20] = 0.30
    values[21] = 160.0
    values[22] = 0.40
    values[23] = 170.0
    values[24] = 180.0
    values[25] = 1.5
    values[26] = 0.25
    values[27] = 0.0
    values[28] = 0.0
    values[29] = 0.0
    values[30] = 0.0
    values[31] = 3.0
    values[32] = 2.0
    values[33] = 1.0
    values[34] = 50.0
    values[35] = 60.0
    values[36] = 70.0
    values[37] = 0.0
    values[38] = 0.0
    values[39] = 0.0
    values[40] = 0.0
    values[41] = 0.0
    values[42] = 0.0
    return HeFESToParameterRecord(
        source_path=Path("synthetic"),
        species_label="test",
        phase_label="phase",
        values=tuple(values),
        raw_lines=tuple(),
    )


def test_thermal_record_can_be_parsed_into_named_fields():
    record = _make_parameter_record()
    parsed = parset(record)

    assert parsed.fn == 2.0
    assert parsed.zu == 1.0
    assert parsed.wm == 100.0
    assert parsed.To == 300.0
    assert parsed.modes.wd1 == 900.0
    assert parsed.modes.ws1 == 100.0 * HCOK
    assert parsed.modes.we4 == 160.0 * HCOK
    assert parsed.modes.qe4 == 0.40
    assert parsed.ibv == 3
    assert parsed.ied == 2
    assert parsed.izp == 1
    assert parsed.Go == 50.0
    assert parsed.Gop == 60.0
    assert parsed.Got == 70.0


def test_thermal_tensor_bundle_preserves_order_and_shapes():
    record = parset(_make_parameter_record())
    bundle = HeFESToThermalModeTensors.from_parset_results(["test"], [record])

    assert bundle.names == ("test",)
    assert bundle.wd.shape == (1, 3)
    assert bundle.ws.shape == (1, 3)
    assert bundle.we.shape == (1, 4)
    assert bundle.qe.shape == (1, 4)
    assert bundle.wd[0, 0].item() == 900.0
    assert bundle.ws[0, 0].item() == 100.0 * HCOK
    assert bundle.Go[0].item() == 50.0


def test_heat_function_handles_edge_cases_exactly():
    assert Heat(0.0, 1.0, 1) == 0.0
    assert Heat(0.0, 1.0, 5) == 0.0
    assert Heat(3.0, 1.0, 5) == 1.0


def test_heat_function_optic_branch_degenerates_to_einstein_when_width_is_small():
    value = Heat(25.0, 0.001, 4)
    expected = Heat(25.0, 0.001, 2)
    assert value == expected


def test_gamset_constant_q_branch_scales_all_modes_identically():
    result = gamset(
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        100.0,
        110.0,
        120.0,
        1.0,
        1.5,
        0.25,
        8.0,
        4.0,
        ityp=3,
    )

    assert result.q == 1.5
    assert result.gamma == pytest.approx(1.0 * (8.0 / 4.0) ** 1.5)
    scale = result.wd1 / 10.0
    assert result.wd2 / 20.0 == pytest.approx(scale)
    assert result.we4 / 100.0 == pytest.approx(scale)
    assert result.wou / 110.0 == pytest.approx(scale)
    assert result.wol / 120.0 == pytest.approx(scale)