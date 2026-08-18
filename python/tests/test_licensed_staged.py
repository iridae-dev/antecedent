"""Every licensed cell must run identify → prepare → estimate, not analyze-only."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def _confounded(n: int = 240, seed: int = 19):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.random(n) < 1.0 / (1.0 + np.exp(-(-0.4 + 0.9 * z)))).astype(np.float64)
    y = 2.0 * t + z + rng.normal(scale=0.4, size=n)
    return {"t": t, "y": y, "z": z}


_ATE_DATA = _confounded()
_EDGES = [("z", "t"), ("z", "y"), ("t", "y")]
_DAG = antecedent.Dag.from_edges(["z", "t", "y"], _EDGES)
_ACCEPTED = antecedent.AcceptedGraph.from_graph(_DAG, algorithm_id="hand")
_ATE = antecedent.AverageEffect(treatment="t", outcome="y")


def _curve_table(n: int = 400, seed: int = 19):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = 0.7 * z + rng.normal(size=n)
    y = 2.0 * t + z + rng.normal(scale=0.2, size=n)
    return {"t": t, "y": y, "z": z}


_CURVE_DATA = _curve_table()
_CURVE = antecedent.ResponseCurve("t", "y", grid=[-0.5, 0.0, 0.5])


_BAYES = antecedent.Bayesian(backend="conjugate", n_draws=64)


@pytest.mark.parametrize(
    "data, graph, query, refute, inference",
    [
        (_ATE_DATA, _DAG, _ATE, False, None),
        (_ATE_DATA, _DAG, _ATE, "cheap", None),
        (_ATE_DATA, _DAG, _ATE, "full", None),
        (_ATE_DATA, _ACCEPTED, _ATE, False, None),
        (_ATE_DATA, _ACCEPTED, _ATE, "cheap", None),
        (_ATE_DATA, _ACCEPTED, _ATE, "full", None),
        (_ATE_DATA, _DAG, _ATE, False, _BAYES),
        (_ATE_DATA, _ACCEPTED, _ATE, False, _BAYES),
        (_CURVE_DATA, _EDGES, _CURVE, False, None),
        (_CURVE_DATA, _ACCEPTED, _CURVE, False, None),
    ],
    ids=[
        "ate_explicit_none",
        "ate_explicit_cheap",
        "ate_explicit_full",
        "ate_accepted_none",
        "ate_accepted_cheap",
        "ate_accepted_full",
        "ate_explicit_bayesian_none",
        "ate_accepted_bayesian_none",
        "curve_explicit_none",
        "curve_accepted_none",
    ],
)
def test_licensed_cell_prepare_matches_analyze(data, graph, query, refute, inference):
    fresh = antecedent.analyze(
        data,
        graph=graph,
        query=query,
        refute=refute,
        bootstrap=0,
        seed=1,
        inference=inference,
    )
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        graph=graph,
        query=query,
        refute=refute,
        seed=1,
        latency="interactive",
        inference=inference,
    )
    if isinstance(graph, antecedent.AcceptedGraph) and hasattr(prepared, "structure_source"):
        assert prepared.structure_source == "accepted"
    click = prepared.estimate(data, seed=1)
    if isinstance(query, antecedent.ResponseCurve):
        assert click.response is not None and fresh.response is not None
        assert click.response.values == fresh.response.values
        return
    assert math.isfinite(click.ate) and math.isfinite(fresh.ate)
    assert abs(click.ate - fresh.ate) < 1e-12
