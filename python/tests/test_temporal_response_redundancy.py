"""Supporting temporal-response fixtures: confounding and horizon-varying support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent
from antecedent.estimation import PreparedAnalysis
from antecedent.intervention import Set

_ROOT = Path(__file__).resolve().parents[2]
_CONFOUNDED = json.loads(
    (_ROOT / "conformance" / "response" / "temporal_confounded_pulse" / "expected.json").read_text()
)
_SUPPORT = json.loads(
    (_ROOT / "conformance" / "response" / "temporal_horizon_support" / "expected.json").read_text()
)

_CONFOUNDED_EDGES = [("z", 0, "t", 0), ("z", 1, "y", 0), ("t", 1, "y", 0)]
_SUPPORT_EDGES = [("t", 1, "y", 0), ("t", 2, "y", 0)]
_IDENT = _CONFOUNDED["contract"]["identification"]
_UNADJUSTED = _CONFOUNDED["contract"]["unadjusted"]


def _confounded_data() -> dict[str, np.ndarray]:
    n = int(_CONFOUNDED["generation"]["n"])
    z = np.array([1.0 if i % 4 in (0, 1) else -1.0 for i in range(n)])
    u = np.array([1.0 if i % 4 in (0, 2) else -1.0 for i in range(n)])
    t = z + u
    y = np.zeros(n)
    for i in range(n):
        y[i] = 1.0 + 2.0 * (t[i - 1] if i >= 1 else 0.0) + 5.0 * (z[i - 1] if i >= 1 else 0.0)
    return {"t": t, "y": y, "z": z}


def _support_data() -> dict[str, np.ndarray]:
    n = int(_SUPPORT["generation"]["n"])
    t = np.array([10.0 if i >= n - 7 else 0.05 * np.sin(i) for i in range(n)])
    y = np.zeros(n)
    for i in range(n):
        y[i] = 1.0 + 2.0 * (t[i - 1] if i >= 1 else 0.0) + 3.0 * (t[i - 2] if i >= 2 else 0.0)
    return {"t": t, "y": y}


def _means(result: Any) -> np.ndarray:
    assert result.response is not None
    return np.asarray([row[0] for row in result.response.values], dtype=float)


def _adjustment_names(spec: dict[str, Any]) -> list[str]:
    return [item["variable"] for item in spec["adjustment_set"]]


def _assert_identified(result: Any, spec: dict[str, Any], *, estimator: str | None = None) -> None:
    view = result.identification
    assert view.status == spec["status"]
    assert view.method == spec["method"]
    assert list(view.adjustment_set) == _adjustment_names(spec)
    if spec["adjustment_set"]:
        assert view.adjustment_set, (
            "empty Z with method temporal.backdoor.unfolded is the schedule-ID relabel bug"
        )
    if estimator is not None:
        assert result.estimate.estimator_id == estimator
        assert result.estimate.method == spec["method"]


def test_confounded_pulse_recovers_structural_surface_unadjusted_does_not():
    data = _confounded_data()
    atol = float(_CONFOUNDED["tolerance"]["atol"])
    query = antecedent.ResponseCurve(
        "t",
        "y",
        grid=[0.0, 1.0],
        horizons=[1],
        policy="pulse",
        treatment_lag=1,
    )
    adjusted = antecedent.analyze(
        data, graph=_CONFOUNDED_EDGES, query=query, refute=False, bootstrap=0, seed=21
    )
    np.testing.assert_allclose(
        _means(adjusted), _CONFOUNDED["contract"]["surface"]["mean"], atol=atol
    )
    _assert_identified(adjusted, _IDENT)
    assert adjusted.support.status == "supported"

    unadjusted = antecedent.analyze(
        data, graph=[("t", 1, "y", 0)], query=query, refute=False, bootstrap=0, seed=21
    )
    np.testing.assert_allclose(
        _means(unadjusted),
        _UNADJUSTED["surface"]["mean"],
        atol=atol,
    )
    _assert_identified(unadjusted, _UNADJUSTED["identification"])

    tdag = antecedent.TemporalDag.from_lagged_edges(["t", "y", "z"], _CONFOUNDED_EDGES)
    accepted = antecedent.AcceptedGraph.from_graph(tdag, algorithm_id="hand")
    click = PreparedAnalysis.prepare(
        data, graph=accepted, query=query, refute=False, seed=21
    ).estimate(data, seed=21)
    np.testing.assert_allclose(_means(click), _CONFOUNDED["contract"]["surface"]["mean"], atol=atol)
    _assert_identified(click, _IDENT)


def test_confounded_multi_horizon_keeps_z_at_short_horizon():
    data = _confounded_data()
    atol = float(_CONFOUNDED["tolerance"]["atol"])
    spec = _CONFOUNDED["contract"]["multi_horizon"]
    h2_atol = float(spec["horizon_2_atol"])
    query = antecedent.ResponseCurve(
        "t",
        "y",
        grid=[0.0, 1.0],
        horizons=[1, 2],
        policy="pulse",
        treatment_lag=1,
    )
    result = antecedent.analyze(
        data, graph=_CONFOUNDED_EDGES, query=query, refute=False, bootstrap=0, seed=29
    )
    mean = _means(result)
    expected = spec["surface"]["mean"]
    assert result.identification.adjustment_set == ["z"]
    assert result.identification.horizon_adjustment_sets == (("z@-1",), ())
    np.testing.assert_allclose(mean[0], expected[0], atol=atol)
    np.testing.assert_allclose(mean[2], expected[2], atol=atol)
    np.testing.assert_allclose(mean[1], expected[1], atol=h2_atol)
    np.testing.assert_allclose(mean[3], expected[3], atol=h2_atol)
    assert abs(mean[2] - mean[0] - 2.0) <= atol

    click = PreparedAnalysis.prepare(
        data,
        graph=antecedent.AcceptedGraph.from_graph(
            antecedent.TemporalDag.from_lagged_edges(["t", "y", "z"], _CONFOUNDED_EDGES),
            algorithm_id="hand",
        ),
        query=query,
        refute=False,
        seed=29,
    ).estimate(data, seed=29)
    np.testing.assert_allclose(_means(click), mean, atol=atol)
    assert click.identification.horizon_adjustment_sets == (("z@-1",), ())


def test_confounded_pulse_and_sustained_match_structural_contrast():
    data = _confounded_data()
    atol = float(_CONFOUNDED["tolerance"]["atol"])
    expected = float(_CONFOUNDED["contract"]["pulse_effect_projection"]["contrast"])
    estimators = _CONFOUNDED["contract"]["estimators"]
    pulse = antecedent.analyze(
        data,
        graph=_CONFOUNDED_EDGES,
        query=antecedent.PulseEffect("t", "y", treatment_lag=1, horizon_steps=1),
        refute=False,
        bootstrap=0,
        seed=23,
    )
    sustained = antecedent.analyze(
        data,
        graph=_CONFOUNDED_EDGES,
        query=antecedent.SustainedEffect("t", "y", treatment_lag=1, horizon_steps=1),
        refute=False,
        bootstrap=0,
        seed=23,
    )
    _assert_identified(pulse, _IDENT, estimator=estimators["pulse"])
    _assert_identified(sustained, _IDENT, estimator=estimators["sustained"])
    assert pulse.identification.adjustment_set == sustained.identification.adjustment_set
    assert pulse.ate == pytest.approx(expected, abs=atol)
    assert sustained.ate == pytest.approx(expected, abs=atol)

    tdag = antecedent.TemporalDag.from_lagged_edges(["t", "y", "z"], _CONFOUNDED_EDGES)
    accepted = antecedent.AcceptedGraph.from_graph(tdag, algorithm_id="hand")
    accepted_pulse = antecedent.analyze(
        data,
        graph=accepted,
        query=antecedent.PulseEffect("t", "y", treatment_lag=1, horizon_steps=1),
        refute=False,
        bootstrap=0,
        seed=23,
    )
    _assert_identified(accepted_pulse, _IDENT, estimator=estimators["pulse"])
    assert accepted_pulse.ate == pytest.approx(expected, abs=atol)

    unadjusted = antecedent.analyze(
        data,
        graph=[("t", 1, "y", 0)],
        query=antecedent.PulseEffect("t", "y", treatment_lag=1, horizon_steps=1),
        refute=False,
        bootstrap=0,
        seed=23,
    )
    _assert_identified(unadjusted, _UNADJUSTED["identification"], estimator=estimators["pulse"])
    assert unadjusted.ate == pytest.approx(_UNADJUSTED["pulse_contrast"], abs=atol)


def test_horizon_support_is_per_cell_not_union():
    data = _support_data()
    atol = float(_SUPPORT["tolerance"]["atol"])
    query = antecedent.ResponseCurve(
        "t",
        "y",
        grid=[0.0, 5.0],
        horizons=[1, 8],
        policy="pulse",
        treatment_lag=1,
    )
    result = antecedent.analyze(
        data, graph=_SUPPORT_EDGES, query=query, refute=False, bootstrap=0, seed=11
    )
    spec = _SUPPORT["contract"]["support"]
    assert result.support.status == spec["status"]
    assert list(result.support.point_status) == spec["point_status"]
    ranges = next(
        d.values
        for d in result.support.diagnostics
        if d.id == "response.temporal.horizon_treatment_range"
    )
    np.testing.assert_allclose(ranges, spec["horizon_treatment_range"], atol=atol)
    union_min = min(ranges[0], ranges[2])
    union_max = max(ranges[1], ranges[3])
    assert union_min <= 5.0 <= union_max
    assert any("inspect support.point_status" in warning for warning in result.support.warnings)

    path = antecedent.analyze(
        data,
        graph=_SUPPORT_EDGES,
        query=antecedent.InterventionResponse(
            "y",
            intervention=Set("t", 5.0),
            horizons=[1, 8],
            policy="pulse",
            treatment_lag=1,
        ),
        refute=False,
        bootstrap=0,
        seed=11,
    )
    assert path.support.status == "extrapolative"
    assert list(path.support.point_status) == ["supported", "outside_empirical_support"]
