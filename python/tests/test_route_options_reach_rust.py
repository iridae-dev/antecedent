"""Options the Rust study honours reach it on every route that licenses them.

A query field or ``estimator_config`` key that one prepare route forwards and
another drops (or refuses although Rust would honour it) is a capability the
Python API does not reach. Each test changes one option and checks that the
executed study changed accordingly.
"""

from __future__ import annotations

from typing import Any

import antecedent as ant
import numpy as np
import pytest
from antecedent.discovery import DbnPosterior, ExactDagPosterior


def _series(n: int = 600, seed: int = 11) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.9 * x[i - 1] + 0.3 * rng.normal()
    return {"x": x, "y": y}


def _continuous(n: int = 900, seed: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = z + rng.normal(size=n)
    y = 1.5 * t + np.sin(t) + z + rng.normal(size=n)
    return {"z": z, "t": t, "y": y}


def _curve(result: Any) -> list[float]:
    return [float(v) for v in np.asarray(result.response.values).flatten()]


@pytest.mark.parametrize(
    ("query_type", "options"),
    [
        (ant.PulseEffect, {"graph": [("x", 1, "y", 0)]}),
        (ant.SustainedEffect, {"graph": [("x", 1, "y", 0)]}),
        (ant.PulseEffect, {"graph": [("x", 1, "y", 0)], "inference": ant.Bayesian(n_draws=200)}),
        (
            ant.PulseEffect,
            {"discovery": DbnPosterior(max_lag=1), "inference": ant.Bayesian(n_draws=200)},
        ),
    ],
    ids=["pulse", "sustained", "pulse-bayesian", "pulse-dbn-posterior"],
)
def test_temporal_control_level_sets_the_contrast(query_type: Any, options: dict) -> None:
    """A linear lagged effect scales with ``active_level - control_level``."""
    data = _series()

    def effect(control: float) -> float:
        query = query_type("x", "y", treatment_lag=1, control_level=control, active_level=1.0)
        return float(ant.analyze(data, query=query, refute=False, seed=3, **options).ate)

    full, half = effect(0.0), effect(0.5)
    assert full == pytest.approx(0.9, abs=0.1)
    assert half == pytest.approx(full / 2, rel=0.05)


def test_response_nuisance_options_reach_the_static_curve() -> None:
    data = _continuous()
    graph = [("z", "t"), ("z", "y"), ("t", "y")]
    query = ant.ResponseCurve("t", "y", grid=[-1.0, 0.0, 1.0])
    base = ant.analyze(data, graph=graph, query=query, refute=False)
    default_folds = ant.analyze(
        data, graph=graph, query=query, refute=False, estimator_config={"folds": 5}
    )
    other = ant.analyze(
        data,
        graph=graph,
        query=query,
        refute=False,
        estimator_config={"folds": 3, "nuisance_basis": 8},
    )
    assert _curve(default_folds) == _curve(base)
    assert _curve(other) != _curve(base)


def test_response_options_reach_a_class_envelope_and_a_graph_posterior() -> None:
    data = _continuous()
    query = ant.ResponseCurve("t", "y", grid=[-1.0, 0.0, 1.0])
    cpdag = ant.Cpdag.from_directed_undirected(
        ["z", "t", "y"], [("z", "y"), ("t", "y")], [("z", "t")]
    )
    for route in ({"graph": cpdag}, {"discovery": ExactDagPosterior()}):
        base = ant.analyze(data, query=query, refute=False, **route)
        other = ant.analyze(data, query=query, refute=False, estimator_config={"folds": 3}, **route)
        assert _curve(other) != _curve(base), route
        with pytest.raises(ValueError, match="unknown response estimator_config keys"):
            ant.analyze(data, query=query, refute=False, estimator_config={"nope": 1}, **route)


def test_derivatives_take_the_nuisance_options() -> None:
    data = _continuous()
    graph = [("z", "t"), ("z", "y"), ("t", "y")]
    query = ant.AverageDerivative("t", "y")
    base = ant.analyze(data, graph=graph, query=query, refute=False)
    other = ant.analyze(data, graph=graph, query=query, refute=False, estimator_config={"folds": 3})
    assert float(other.estimate) != float(base.estimate)
    with pytest.raises(ValueError, match="prepared derivatives accept only"):
        ant.analyze(
            data,
            graph=graph,
            query=query,
            refute=False,
            estimator_config={"export_row_diagnostics": True},
        )
