"""AverageEffect on one series under a temporal graph: the lag-0 contrast.

``analyze(series, graph=TemporalDag | TemporalCpdag | TemporalPag, query=AverageEffect(x, y))``
compiles the contrast ``E[Y_t | do(X_t = 1)] - E[Y_t | do(X_t = 0)]`` as the pulse at offset 0
read at horizon 1. The answer must equal the explicit lag-0 ``PulseEffect``; the evidence status
stays ``refused`` (the AverageEffect cell executes and is unlicensed) and the result discloses
the reading. Coverage is not measured here.
"""

from __future__ import annotations

import math

import antecedent
import numpy as np
import pytest
from antecedent.errors import CausalUnsupportedError

DISCLOSURE = "estimate.temporal_class.lag_zero_average_effect"
N = 1200
BETA = 2.0


def _series() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(5)
    z = rng.standard_normal(N)
    e = rng.standard_normal(N)
    u = rng.standard_normal(N)
    x = np.zeros(N)
    y = np.zeros(N)
    x[1:] = 0.6 * z[:-1] + e[1:]
    y[1:] = BETA * x[1:] + 0.8 * z[:-1] + 0.5 * u[1:]
    return {"x": x, "y": y, "z": z}


_DIRECTED = [("z", 1, "x", 0), ("z", 1, "y", 0)]


def _dag() -> antecedent.graph.TemporalDag:
    return antecedent.graph.TemporalDag.from_lagged_edges(
        ["x", "y", "z"], [*_DIRECTED, ("x", 0, "y", 0)]
    )


def _cpdag() -> antecedent.graph.TemporalCpdag:
    return antecedent.graph.TemporalCpdag.from_lagged_edges(
        ["x", "y", "z"], _DIRECTED, [("x", 0, "y", 0)]
    )


def _run(graph, query, inference=None):
    return antecedent.analyze(
        _series(),
        graph=graph,
        query=query,
        inference=inference,
        refute=False,
        bootstrap=0,
        seed=7,
    )


def _same(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return (math.isnan(left) and math.isnan(right)) or left == right


@pytest.mark.parametrize("graph_factory", [_dag, _cpdag])
@pytest.mark.parametrize("bayesian", [False, True])
def test_the_lag_zero_average_effect_is_the_explicit_pulse(graph_factory, bayesian) -> None:
    inference = antecedent.Bayesian(n_draws=64, backend="conjugate") if bayesian else None
    average = _run(graph_factory(), antecedent.AverageEffect("x", "y"), inference)
    pulse = _run(
        graph_factory(),
        antecedent.PulseEffect("x", "y", treatment_lag=0, horizon_steps=1),
        inference,
    )
    assert _same(average.ate, pulse.ate)
    assert average.structural_identified_set_interval == pulse.structural_identified_set_interval
    assert average.evidence_status == "refused"
    assert any(DISCLOSURE in str(diagnostic) for diagnostic in average.diagnostics)
    assert not any(DISCLOSURE in str(diagnostic) for diagnostic in pulse.diagnostics)


def test_the_temporal_dag_recovers_the_contemporaneous_effect() -> None:
    result = _run(_dag(), antecedent.AverageEffect("x", "y"))
    assert abs(result.ate - BETA) < 0.1


def test_another_functional_is_refused_with_the_next_attempt() -> None:
    exceedance = antecedent.AverageEffect(
        "x", "y", outcome_functional={"kind": "exceedance", "threshold": 1.0}
    )
    with pytest.raises(CausalUnsupportedError, match="Next attempt"):
        _run(_cpdag(), exceedance)
