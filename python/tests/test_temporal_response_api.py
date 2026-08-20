"""Python facade known-truth pin for temporal ResponseCurve / InterventionResponse."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent
from antecedent.errors import CausalUnsupportedError, CausalValueError
from antecedent.estimation import PreparedAnalysis
from antecedent.intervention import Sequence, Set, Soft

_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = json.loads(
    (_ROOT / "conformance" / "response" / "temporal_dose_horizon" / "expected.json").read_text()
)


def _fixture_data() -> dict[str, np.ndarray]:
    n = int(_FIXTURE["generation"]["n"])
    t = np.array([0.0 if i % 4 in (0, 2) else (1.0 if i % 4 == 1 else -1.0) for i in range(n)])
    y = np.zeros(n)
    for i in range(n):
        y[i] = 1.0 + 2.0 * (t[i - 1] if i >= 1 else 0.0) + 3.0 * (t[i - 2] if i >= 2 else 0.0)
    return {"t": t, "y": y}


_EDGES = [("t", 1, "y", 0), ("t", 2, "y", 0)]
_ATOL = float(_FIXTURE["tolerance"]["atol"])
_MEAN = np.asarray(_FIXTURE["contract"]["surface"]["mean"], dtype=float)
_SET_PATH = np.asarray(_FIXTURE["contract"]["intervention_paths"]["set_1"], dtype=float)
_SOFT_CONST = np.asarray(_FIXTURE["contract"]["intervention_paths"]["soft_constant_1"], dtype=float)


def _response_means(result: Any) -> np.ndarray:
    assert result.response is not None
    return np.asarray([row[0] for row in result.response.values], dtype=float)


def test_temporal_response_curve_matches_fixture_and_prepared_reuse():
    data = _fixture_data()
    query = antecedent.ResponseCurve(
        "t",
        "y",
        grid=[0.0, 1.0],
        horizons=[1, 2],
        policy="pulse",
        treatment_lag=1,
    )
    direct = antecedent.analyze(data, graph=_EDGES, query=query, refute=False, bootstrap=0, seed=21)
    np.testing.assert_allclose(_response_means(direct), _MEAN, atol=_ATOL)
    assert direct.support.status == "supported"
    assert list(direct.support.point_status) == ["supported"] * 4

    prepared = PreparedAnalysis.prepare(data, graph=_EDGES, query=query, refute=False, seed=21)
    click = prepared.estimate(data, seed=21)
    np.testing.assert_allclose(_response_means(click), _MEAN, atol=_ATOL)
    assert prepared.structure_source == "explicit"


def test_temporal_intervention_set_and_single_step_sequence_match_fixture():
    data = _fixture_data()
    set_query = antecedent.InterventionResponse(
        "y",
        intervention=Set("t", 1.0),
        horizons=[1, 2],
        policy="pulse",
        treatment_lag=1,
    )
    seq_query = antecedent.InterventionResponse(
        "y",
        intervention=Sequence([Set("t", 1.0)]),
        horizons=[1, 2],
        policy="pulse",
        treatment_lag=1,
    )
    soft_query = antecedent.InterventionResponse(
        "y",
        intervention=Soft("t", "constant", parameters=[1.0]),
        horizons=[1, 2],
        policy="pulse",
        treatment_lag=1,
    )
    set_result = antecedent.analyze(
        data, graph=_EDGES, query=set_query, refute=False, bootstrap=0, seed=22
    )
    seq_result = antecedent.analyze(
        data, graph=_EDGES, query=seq_query, refute=False, bootstrap=0, seed=22
    )
    soft_result = antecedent.analyze(
        data, graph=_EDGES, query=soft_query, refute=False, bootstrap=0, seed=22
    )
    np.testing.assert_allclose(_response_means(set_result), _SET_PATH, atol=_ATOL)
    np.testing.assert_allclose(_response_means(seq_result), _SET_PATH, atol=_ATOL)
    np.testing.assert_allclose(_response_means(soft_result), _SOFT_CONST, atol=_ATOL)

    prepared = PreparedAnalysis.prepare(data, graph=_EDGES, query=seq_query, refute=False, seed=22)
    click = prepared.estimate(data, seed=22)
    np.testing.assert_allclose(_response_means(click), _SET_PATH, atol=_ATOL)


def test_multi_step_sequence_refuses():
    data = _fixture_data()
    query = antecedent.InterventionResponse(
        "y",
        intervention=Sequence([Set("t", 0.0), Set("t", 1.0)]),
        horizons=[1],
        policy="pulse",
        treatment_lag=1,
    )
    with pytest.raises(CausalUnsupportedError, match="multi-step Sequence"):
        antecedent.analyze(data, graph=_EDGES, query=query, refute=False, bootstrap=0)


def test_pulse_projection_matches_surface_contrast():
    data = _fixture_data()
    surface = antecedent.analyze(
        data,
        graph=_EDGES,
        query=antecedent.ResponseCurve(
            "t",
            "y",
            grid=[0.0, 1.0],
            horizons=[1, 2],
            policy="pulse",
            treatment_lag=1,
        ),
        refute=False,
        bootstrap=0,
        seed=23,
    )
    pulse = antecedent.analyze(
        data,
        graph=_EDGES,
        query=antecedent.PulseEffect("t", "y", treatment_lag=1, horizon_steps=1),
        refute=False,
        bootstrap=0,
        seed=23,
    )
    means = _response_means(surface)
    # dose-major: mean[dose=1,h=1] - mean[dose=0,h=1] == index 2 - index 0
    contrast = float(means[2] - means[0])
    expected = float(_FIXTURE["contract"]["pulse_effect_projection"]["contrast"])
    assert contrast == pytest.approx(expected, abs=_ATOL)
    assert pulse.ate == pytest.approx(expected, abs=_ATOL)


def test_temporal_response_bands_match_fixture():
    data = _fixture_data()
    result = antecedent.analyze(
        data,
        graph=_EDGES,
        query=antecedent.ResponseCurve(
            "t",
            "y",
            grid=[0.0, 1.0],
            horizons=[1, 2],
            policy="pulse",
            treatment_lag=1,
        ),
        refute=False,
        bootstrap=0,
        seed=21,
    )
    assert result.response is not None
    assert result.uncertainty.lower is not None and result.uncertainty.upper is not None
    expected_lower = np.asarray(_FIXTURE["contract"]["surface"]["lower"], dtype=float)
    expected_upper = np.asarray(_FIXTURE["contract"]["surface"]["upper"], dtype=float)
    lower = np.asarray([row[0] for row in result.uncertainty.lower], dtype=float)
    upper = np.asarray([row[0] for row in result.uncertainty.upper], dtype=float)
    np.testing.assert_allclose(lower, expected_lower, atol=_ATOL)
    np.testing.assert_allclose(upper, expected_upper, atol=_ATOL)
    assert upper[0] - lower[0] > 0.0
    assert result.identification.method == "temporal.backdoor.unfolded"
    assert "identify.temporal_backdoor" in result.provenance["operation_ids"]
    assert result.validation is not None
    assert any(check.id == "refute.temporal_response.skipped" for check in result.validation.checks)


def test_temporal_default_lag_matches_pulse():
    curve = antecedent.ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=[1])
    path = antecedent.InterventionResponse("y", intervention=Set("t", 1.0), horizons=[1])
    assert curve.treatment_lag == 1
    assert path.treatment_lag == 1
    assert antecedent.PulseEffect("t", "y").treatment_lag == 1


def test_temporal_horizons_reject_bool_and_dynamic_policy():
    with pytest.raises(CausalValueError, match="positive integers"):
        antecedent.ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=[True])
    with pytest.raises(CausalValueError, match="pulse"):
        antecedent.ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=[1], policy="dynamic")


def test_prepared_temporal_rejects_wrong_identifier_and_refute_click():
    data = _fixture_data()
    query = antecedent.ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=[1], policy="pulse")
    with pytest.raises(CausalValueError, match="temporal.backdoor.unfolded"):
        PreparedAnalysis.prepare(data, graph=_EDGES, query=query, identifier="response.backdoor")
    prepared = PreparedAnalysis.prepare(data, graph=_EDGES, query=query, refute=False)
    with pytest.raises(CausalUnsupportedError, match="AverageEffect-only"):
        prepared.refute(data)
