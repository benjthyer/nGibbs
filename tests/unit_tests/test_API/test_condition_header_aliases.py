"""Condition-header normalization: '<quantity>(<unit>)' forms, with or without
'(System_main)', map onto a model's canonical feature names only in the units
that model was trained in; nothing is converted."""

import pytest

from ngibbs.engine.API import _canonicalize_condition_headers as canon

HEFESTO = ['P(GPa)(System_main)', 'T(K)(System_main)', 'S(J/g/K)(System_main)']
MELTS = ['Pressure(System_main)', 'Temperature(System_main)']
MELTS_OPEN = MELTS + ['logfO2-QFM(System_main)']


@pytest.mark.parametrize('features, headers, expected', [
    (HEFESTO, ['P(GPa)', 'T(K)'], ['P(GPa)(System_main)', 'T(K)(System_main)']),
    (HEFESTO, ['Pressure(GPa)', 'Temperature(K)(System_main)'], ['P(GPa)(System_main)', 'T(K)(System_main)']),
    (HEFESTO, ['p (GPa)', 'temp(k)', 'Entropy(J/g/K)'],
     ['P(GPa)(System_main)', 'T(K)(System_main)', 'S(J/g/K)(System_main)']),
    (MELTS, ['P(bar)', 'T(C)'], MELTS),
    (MELTS, ['Pressure(bars)', 'Temperature(°C)'], MELTS),
    (MELTS, ['P(bar)', 'T(degC)'], MELTS),
    (MELTS_OPEN, ['P(bar)', 'T(C)', 'logfO2(QFM)'], MELTS_OPEN),
    (MELTS_OPEN, ['P(bar)', 'T(C)', 'ΔQFM'], MELTS_OPEN),
    (MELTS_OPEN, ['P(bar)', 'T(C)', 'logfO2-FMQ'], MELTS_OPEN),
])
def test_accepted_forms(features, headers, expected):
    assert canon(headers, features) == expected


@pytest.mark.parametrize('features, headers', [
    (HEFESTO, HEFESTO), (MELTS_OPEN, MELTS_OPEN),
    (MELTS, ['logfO2-QFM(System_main)']),     # canonical names pass through (routing reports it)
    (HEFESTO, ['SiO2', 'Fe', 'P', 'S']),      # compositions and element symbols untouched
])
def test_passthrough(features, headers):
    assert canon(headers, features) == headers


@pytest.mark.parametrize('features, header, match', [
    (HEFESTO, 'T(C)', 'in K'),                # no centigrade for HeFESTo
    (HEFESTO, 'P(bar)', 'in GPa'),
    (MELTS, 'T(K)', 'in °C'),                 # no Kelvin for MELTS
    (MELTS, 'P(GPa)', 'in bar'),
    (MELTS_OPEN, 'logfO2(NNO)', 'relative to QFM'),
    (HEFESTO, 'Temperature', 'no units'),     # unitless forms that were never accepted
    (MELTS, 'T(System_main)', 'no units'),
    (MELTS_OPEN, 'logfO2', 'no units'),
    (MELTS, 'logfO2(QFM)', 'does not take'),  # quantity the model has no input for
    (MELTS, 'S(J/g/K)', 'does not take'),
])
def test_rejected(features, header, match):
    with pytest.raises(ValueError, match=match):
        canon([header], features)
