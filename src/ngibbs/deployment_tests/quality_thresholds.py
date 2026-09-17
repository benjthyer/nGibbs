"""Pass/fail quality gate applied on top of the raw deployment-test metrics.

``evaluate_emulator_quality`` (per-phase precision/recall/abundance/oxide
tables) and the MELTStable EOS comparison (``run_meltstable_property_comparison``)
report metrics only -- see the "Quality metrics only" note on
``EmulatorAPI.test()``. This module is that "separate layer": it turns those
tables into a list of ``QualityFailure`` objects (empty == pass) against fixed
tolerances, bucketed by how abundant a phase actually is in the ground truth.

Thresholds
----------
Phase presence (precision / recall), bucketed by a phase's average modal
abundance in the GT bundle (``abundance_gt_mean_wtpct``, wt% of system,
zero-padded over rows where the phase is absent):

    > 10 wt%   ("major")     precision and recall must both exceed 0.97
    5-10 wt%   ("moderate")  precision and recall must both exceed 0.95
    < 5 wt%    ("minor")     precision and recall must both exceed 0.70

Major phases only, additionally:
    * mean relative phase-abundance error must not exceed 20%
    * mean relative within-phase error of MgO, FeO, and SiO2 (whichever are
      tracked for that phase) must not exceed 5%

HeFESTo MELTStable EOS comparison, restricted to the ``HTZadiabat`` and
``DMMadiabat`` ground-truth tables (the ``BASALTadiabat`` sample is not yet
gated -- see ``MELTSTABLE_TABLES``):
    * rho, VP, VS, S, T: mean relative error must be below 0.1%
    * alpha, Cp, KS:     mean relative error must be below 10%
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# phase precision / recall / abundance thresholds
# --------------------------------------------------------------------------- #
MAJOR_ABUNDANCE_PCT = 10.0      # > this wt% -> "major"
MODERATE_ABUNDANCE_PCT = 5.0    # [this, MAJOR_ABUNDANCE_PCT] wt% -> "moderate"; below -> "minor"

MAJOR_PR_MIN = 0.97
MODERATE_PR_MIN = 0.95
MINOR_PR_MIN = 0.70

MAJOR_ABUNDANCE_RELERR_MAX_PCT = 20.0
MAJOR_OXIDES = ('MgO', 'FeO', 'SiO2')
MAJOR_OXIDE_RELERR_MAX_PCT = 5.0

# --------------------------------------------------------------------------- #
# HeFESTo MELTStable EOS thresholds
# --------------------------------------------------------------------------- #
MELTSTABLE_TABLES = ('HTZadiabat', 'DMMadiabat')
MELTSTABLE_TIGHT_PROPS = ('rho', 'VP', 'VS', 'S', 'T')
MELTSTABLE_TIGHT_MAX_PCT = 0.1
MELTSTABLE_LOOSE_PROPS = ('alpha', 'Cp', 'KS')
MELTSTABLE_LOOSE_MAX_PCT = 10.0


@dataclass
class QualityFailure:
    """One violated threshold."""
    check: str      # e.g. 'precision[major]', 'MgO_comp_relerr_pct[major]', 'meltstable_rho_mean_rel%'
    where: str       # e.g. 'isothermal/bridgmanite', 'HTZadiabat/emulation_isentropic'
    value: float
    threshold: float

    def __str__(self) -> str:
        return f"{self.where}: {self.check} = {self.value:.4g} (limit {self.threshold:.4g})"


class EmulatorQualityError(AssertionError):
    """Raised by ``EmulatorAPI.assert_quality()`` when any threshold is missed."""

    def __init__(self, model_name: str, failures: List[QualityFailure], result: Optional[dict] = None):
        self.model_name = model_name
        self.failures = failures
        self.result = result
        lines = [f"{model_name}: {len(failures)} deployment quality check(s) failed:"]
        lines += [f"  - {f}" for f in failures]
        super().__init__('\n'.join(lines))


def _abundance_bucket(avg_abundance_pct: float):
    if avg_abundance_pct > MAJOR_ABUNDANCE_PCT:
        return 'major', MAJOR_PR_MIN
    if avg_abundance_pct >= MODERATE_ABUNDANCE_PCT:
        return 'moderate', MODERATE_PR_MIN
    return 'minor', MINOR_PR_MIN


def check_phase_quality(
    quality_metrics: Dict[str, pd.DataFrame],
    emulator_name: Optional[str] = None,
) -> List[QualityFailure]:
    """Check every (emulator bundle, phase) pair in an ``EmulatorAPI.test()``
    result's ``quality_metrics`` dict against the precision/recall/abundance/
    oxide thresholds. Phases absent from the GT bundle entirely
    (``n_rows_present == 0``) are skipped -- there is nothing to score.
    """
    failures: List[QualityFailure] = []
    for name, table in quality_metrics.items():
        if emulator_name is not None and name != emulator_name:
            continue
        for phase in table.columns:
            n_present = table.loc['n_rows_present', phase]
            if not np.isfinite(n_present) or n_present <= 0:
                continue
            avg_abund = float(table.loc['abundance_gt_mean_wtpct', phase])
            bucket, min_pr = _abundance_bucket(avg_abund)
            where = f'{name}/{phase}'

            for metric in ('precision', 'recall'):
                val = table.loc[metric, phase]
                if np.isfinite(val) and not (val > min_pr):
                    failures.append(QualityFailure(f'{metric}[{bucket}]', where, float(val), min_pr))

            if bucket != 'major':
                continue

            relerr = table.loc['abundance_relerr_pct', phase]
            if np.isfinite(relerr) and relerr > MAJOR_ABUNDANCE_RELERR_MAX_PCT:
                failures.append(QualityFailure(
                    'abundance_relerr_pct[major]', where, float(relerr), MAJOR_ABUNDANCE_RELERR_MAX_PCT,
                ))

            for ox in MAJOR_OXIDES:
                row = f'{ox}_comp_relerr_pct'
                if row not in table.index:
                    continue
                ox_val = table.loc[row, phase]
                if np.isfinite(ox_val) and not (ox_val < MAJOR_OXIDE_RELERR_MAX_PCT):
                    failures.append(QualityFailure(
                        f'{ox}_comp_relerr_pct[major]', where, float(ox_val), MAJOR_OXIDE_RELERR_MAX_PCT,
                    ))
    return failures


def check_meltstable_quality(property_errors: pd.DataFrame) -> List[QualityFailure]:
    """Check the HTZ/DMM rows of a ``run_meltstable_property_comparison()``
    ``property_errors`` table (index: ``(table, source)``) against the EOS
    property thresholds. Tables outside ``MELTSTABLE_TABLES`` (currently just
    ``BASALTadiabat``) are not gated.
    """
    failures: List[QualityFailure] = []
    thresholds = (
        [(p, MELTSTABLE_TIGHT_MAX_PCT) for p in MELTSTABLE_TIGHT_PROPS]
        + [(p, MELTSTABLE_LOOSE_MAX_PCT) for p in MELTSTABLE_LOOSE_PROPS]
    )
    for (table, source), row in property_errors.iterrows():
        if table not in MELTSTABLE_TABLES:
            continue
        where = f'{table}/{source}'
        for prop, limit in thresholds:
            col = f'{prop} mean_rel%'
            if col not in row.index:
                continue
            val = row[col]
            if pd.notna(val) and val >= limit:
                failures.append(QualityFailure(f'meltstable_{prop}_mean_rel%', where, float(val), limit))
    return failures
