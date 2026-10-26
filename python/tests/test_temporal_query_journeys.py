"""Generic temporal targets use their existing Rust scientific families, not the fixed SCM API."""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalUnsupportedError, CausalValueError
from antecedent.graph import TemporalDag
from antecedent.outcomes import require_success
from antecedent.query import TemporalQuery


def _derivative_fixture():
    n = 80
    x = 1 + np.sin(np.arange(n) * 0.37)
    b = np.arange(n) % 2
    y, v = np.zeros(n), np.zeros(n)
    y[1:] = 3 + 2 * x[:-1] + 0.5 * b[:-1]
    v[1:] = 1 + x[:-1] - b[:-1]
    data = ant.data.table({"x": x, "b": b, "y": y, "v": v})
    edges = [(a, 1, outcome, 0) for a in ("x", "b") for outcome in ("y", "v")]
    return data, TemporalDag.from_lagged_edges(list(data), edges)


def _derivatives():
    return [
        (ant.AverageDerivative("x", "y"), [2.0]),
        (ant.PointDerivative("x", "y", at=2.0), [2.0]),
        (ant.Elasticity("x", "y", at=2.0), [16 / 29]),
        (ant.SemiElasticity("x", "y", at=2.0, log_scale="treatment"), [4.0]),
        (ant.SemiElasticity("x", "y", at=2.0, log_scale="outcome"), [8 / 29]),
        (
            ant.DirectionalDerivative(["x", "b"], ["y", "v"], at=[2.0, 1.0], direction=[1.0, 2.0]),
            [3.0, -1.0],
        ),
        (ant.ResponseJacobian(["x", "b"], ["y", "v"], at=[2.0, 1.0]), [2.0, 0.5, 1.0, -1.0]),
    ]


@pytest.mark.parametrize(
    "base, expected", _derivatives(), ids=lambda value: getattr(value, "kind", None)
)
@pytest.mark.parametrize("bayesian", [False, True])
def test_each_temporal_derivative_matches_its_structural_target_and_independent_artifact(
    base, expected, bayesian
):
    data, graph = _derivative_fixture()
    query = TemporalQuery(base, max_history_lag=1)
    inference = (
        ant.Bayesian(n_draws=32, prior_scale=1000, backend="conjugate")
        if bayesian
        else ant.Frequentist()
    )
    study = ant.prepare(
        data, graph=graph, query=query, inference=inference, bootstrap=0, refute="none", seed=11
    )
    assert study.last_result is None, "preparation and inspection do not execute a numerical fit"
    plan = study.plan_view()
    assert study.last_result is None
    result = require_success(study.estimate(seed=11))
    np.testing.assert_allclose(np.asarray(result.estimate).reshape(-1), expected, atol=0.05, rtol=0)
    assert result.program_id
    assert result.data_snapshot_id
    assert study.plan_view().identity == plan.identity
    loaded = require_success(ant.load(result.export()))
    assert loaded.program_id == result.program_id
    assert loaded.claim_id == result.claim_id
    assert loaded.answer == result.answer
    assert any("temporal" in str(item).lower() for item in result.diagnostics)
    if bayesian:
        assert result.uncertainty.kind != "none"
    else:
        assert result.uncertainty.kind == "none"


def _cross_world_fixture():
    n = 200
    x = np.sin(np.arange(n) * 0.37)
    m, y = np.zeros(n), np.zeros(n)
    shock = np.where(np.arange(1, n) % 2 == 0, 0.3, -0.3)
    m[1:] = 0.5 * x[:-1] + shock
    y[1:] = 1.5 * x[:-1] + 4 * m[1:]
    data = ant.data.table({"x": x, "m": m, "y": y})
    graph = TemporalDag.from_lagged_edges(
        list(data),
        [
            ("x", 1, "m", 0),
            ("x", 1, "y", 0),
            ("m", 0, "y", 0),
        ],
    )
    return data, graph


@pytest.mark.parametrize(
    "base, expected",
    [
        (ant.Counterfactual("x", "y"), 3.5),
        (ant.PathSpecificEffect("x", "y", path_nodes=["m"]), 2.0),
        (ant.InterventionalDistribution("y", interventions={"x": 1.0}), 3.5),
        (ant.NestedCounterfactual("x", "m", "y"), 1.5),
    ],
    ids=lambda value: getattr(value, "kind", None),
)
@pytest.mark.parametrize("bayesian", [False, True])
def test_unfolded_cross_world_targets_keep_their_own_mechanism_and_consumer(
    base, expected, bayesian
):
    data, graph = _cross_world_fixture()
    inference = (
        ant.Bayesian(n_draws=64, prior_scale=1000, backend="conjugate")
        if bayesian
        else ant.Frequentist()
    )
    study = ant.prepare(
        data,
        graph=graph,
        query=TemporalQuery(base),
        inference=inference,
        bootstrap=0,
        refute="none",
        seed=11,
    )
    assert study.last_result is None
    result = require_success(study.estimate(seed=11))
    assert result.estimate.ate == pytest.approx(expected, abs=0.05)
    loaded = require_success(ant.load(result.export()))
    assert loaded.program_id == result.program_id
    assert loaded.claim_id == result.claim_id
    assert loaded.answer == result.answer
    assert any("temporal" in str(note).lower() for note in result.diagnostics)


@pytest.mark.parametrize(
    "geometry",
    [
        {"horizons": [2]},
        {"treatment_lag": 2},
        {"policy": "sustained"},
        {"max_history_lag": 1},
        {"window": (-1, 0), "policy": "sustained"},
    ],
)
def test_unfolded_geometry_is_refused_instead_of_silently_changing_the_target(geometry):
    data, graph = _cross_world_fixture()
    with pytest.raises(CausalUnsupportedError, match="one lag and one horizon"):
        ant.prepare(
            data,
            graph=graph,
            query=TemporalQuery(ant.Counterfactual("x", "y"), **geometry),
            inference=ant.Frequentist(bootstrap=0),
            refute="none",
        )


def test_temporal_attachment_retains_typed_request_and_detaches_mutable_coordinates():
    at = {"x": 2.0, "b": 1.0}
    base = ant.ResponseJacobian(["x", "b"], ["y", "v"], at=at)
    horizons = [1]
    query = TemporalQuery(base, horizons=horizons, max_history_lag=1)
    at["x"] = 99
    horizons.append(2)
    assert isinstance(query.query, ant.ResponseJacobian)
    assert query.query.at["x"] == 2.0
    assert query.horizons == (1,)
    assert query.max_history_lag == 1
    with pytest.raises(TypeError, match="does not support item assignment"):
        query.query.at["x"] = 7


@pytest.mark.parametrize("horizons", [[], [1, 1], [2, 1], [0]])
def test_bad_temporal_geometry_refuses_before_native_work(horizons):
    with pytest.raises(CausalValueError):
        TemporalQuery(ant.PointDerivative("x", "y", at=2.0), horizons=horizons)
