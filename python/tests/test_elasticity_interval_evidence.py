"""Public-route checks for the calibrated Frequentist elasticity interval."""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.outcomes import require_success


def test_elasticity_interval_matches_retained_and_accepted_routes_and_artifact() -> None:
    rng = np.random.default_rng(19)
    x = rng.normal(size=500)
    t = 0.5 * x + rng.normal(size=500)
    y = 5.0 + 2.0 * np.sin(t) + x + rng.normal(size=500)
    data = {"t": t, "x": x, "y": y}
    dag = antecedent.Dag.from_edges(["t", "x", "y"], [("x", "t"), ("x", "y"), ("t", "y")])
    query = antecedent.Elasticity("t", "y", at=0.5)
    options = {
        "query": query,
        "estimator_config": {"bandwidth": 0.35},
        "refute": False,
        "execution": antecedent.Execution(seed=29),
    }

    explicit = require_success(antecedent.analyze(data, graph=dag, **options))
    prepared = require_success(antecedent.estimate(antecedent.prepare(data, graph=dag, **options)))
    accepted = require_success(
        antecedent.analyze(
            data,
            graph=antecedent.AcceptedGraph.from_graph(dag, algorithm_id="ges"),
            **options,
        )
    )

    truth = 0.5 * 2.0 * np.cos(0.5) / (5.0 + 2.0 * np.sin(0.5))
    for result in (explicit, prepared, accepted):
        assert result.estimate == pytest.approx(truth, abs=0.05)
        assert result.uncertainty.interpretation == "confidence"
        assert result.uncertainty.level == pytest.approx(0.95)
        assert result.uncertainty.standard_error > 0
        assert result.uncertainty.lower[0][0] < truth < result.uncertainty.upper[0][0]
        assert any(
            "response.derivative_interval_fieller" in warning for warning in result.support.warnings
        )
        artifact = antecedent.artifacts.loads(bytes(result.export()))
        assert artifact.payload_kind == "analysis_result"
        assert (
            artifact.to_dict()["payload"]["response"]["uncertainty"]["scalar"]["interpretation"]
            == "confidence"
        )

    assert explicit.estimate == prepared.estimate == accepted.estimate
    assert explicit.uncertainty == prepared.uncertainty == accepted.uncertainty
