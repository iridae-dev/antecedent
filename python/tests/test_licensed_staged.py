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


# -- ConditionalEffect / PathSpecificEffect / InterventionalDistribution fixtures --------
#
# Small deterministic tables mirroring the Rust prepared_analysis.rs fixtures
# (conditional_effect_fixture / path_specific_fixture / distribution_fixture in
# crates/antecedent/tests/prepared_analysis.rs): a `t -> y <- w` interaction table, a
# discrete `t -> m -> y` chain (no direct edge), and a confounded `z -> t -> y <- z` table.


def _conditional_table(n: int = 120):
    t = np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)])
    w = np.array([float(i % 5) for i in range(n)])
    y = 1.0 + 2.0 * t + 0.5 * t * w
    return {"t": t, "y": y, "w": w}


_CONDITIONAL_DATA = _conditional_table()
_CONDITIONAL_EDGES = [("t", "y"), ("w", "y")]
_CONDITIONAL_DAG = antecedent.Dag.from_edges(["t", "y", "w"], _CONDITIONAL_EDGES)
_CONDITIONAL_ACCEPTED = antecedent.AcceptedGraph.from_graph(_CONDITIONAL_DAG, algorithm_id="hand")
_CONDITIONAL = antecedent.ConditionalEffect(treatment="t", outcome="y", modifier="w")


def _path_specific_table():
    t_vals: list[float] = []
    m_vals: list[float] = []
    y_vals: list[float] = []
    for t in (0.0, 1.0):
        for _ in range(50):
            t_vals.append(t)
            m_vals.append(t)
            y_vals.append(t)
    return {"t": np.array(t_vals), "m": np.array(m_vals), "y": np.array(y_vals)}


_PATH_DATA = _path_specific_table()
_PATH_EDGES = [("t", "m"), ("m", "y")]
_PATH_DAG = antecedent.Dag.from_edges(["t", "m", "y"], _PATH_EDGES)
_PATH_SPECIFIC = antecedent.PathSpecificEffect(treatment="t", outcome="y", path_nodes=["m"])


def _distribution_table():
    combos = [
        (0.0, 0.0, 0.0, 21),
        (0.0, 0.0, 1.0, 9),
        (0.0, 1.0, 0.0, 4),
        (0.0, 1.0, 1.0, 16),
        (1.0, 0.0, 0.0, 12),
        (1.0, 0.0, 1.0, 3),
        (1.0, 1.0, 0.0, 14),
        (1.0, 1.0, 1.0, 21),
    ]
    t_vals: list[float] = []
    y_vals: list[float] = []
    z_vals: list[float] = []
    for z, t, y, count in combos:
        for _ in range(count):
            z_vals.append(z)
            t_vals.append(t)
            y_vals.append(y)
    return {"t": np.array(t_vals), "y": np.array(y_vals), "z": np.array(z_vals)}


_DISTRIBUTION_DATA = _distribution_table()
_DISTRIBUTION_EDGES = [("z", "t"), ("z", "y"), ("t", "y")]
_DISTRIBUTION_DAG = antecedent.Dag.from_edges(["t", "y", "z"], _DISTRIBUTION_EDGES)
_DISTRIBUTION = antecedent.InterventionalDistribution(outcome="y", interventions={"t": 1.0})


# -- InterventionResponse fixture ------------------------------------------------------
#
# Same deterministic generator as the Rust known-truth pin
# (conformance/response/intervention_response/expected.json /
# crates/antecedent/tests/response_facade.rs::intervention_response_conforms_to_known_truth_fixture):
# zero-noise linear structural mean, hard do(t := 0.25).


def _intervention_response_table(n: int = 240):
    z = np.array([math.sin(i / 17.0) for i in range(n)])
    t = z + np.array([math.cos(i / 11.0) for i in range(n)])
    y = 1.0 + 2.0 * t + 0.8 * z
    return {"t": t, "y": y, "z": z}


_IR_DATA = _intervention_response_table()
_IR_EDGES = [("z", "t"), ("z", "y"), ("t", "y")]
_IR_DAG = antecedent.Dag.from_edges(["t", "y", "z"], _IR_EDGES)
_IR_ACCEPTED = antecedent.AcceptedGraph.from_graph(_IR_DAG, algorithm_id="hand")
_IR = antecedent.InterventionResponse(
    outcome="y", intervention=antecedent.intervention.Set("t", 0.25)
)


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
        (_ATE_DATA, _DAG, _ATE, "cheap", _BAYES),
        (_ATE_DATA, _DAG, _ATE, "full", _BAYES),
        (_ATE_DATA, _ACCEPTED, _ATE, False, _BAYES),
        (_ATE_DATA, _ACCEPTED, _ATE, "cheap", _BAYES),
        (_ATE_DATA, _ACCEPTED, _ATE, "full", _BAYES),
        (_CURVE_DATA, _EDGES, _CURVE, False, None),
        (_CURVE_DATA, _ACCEPTED, _CURVE, False, None),
        (_CONDITIONAL_DATA, _CONDITIONAL_DAG, _CONDITIONAL, False, None),
        (_CONDITIONAL_DATA, _CONDITIONAL_DAG, _CONDITIONAL, "cheap", None),
        (_CONDITIONAL_DATA, _CONDITIONAL_DAG, _CONDITIONAL, "full", None),
        (_CONDITIONAL_DATA, _CONDITIONAL_ACCEPTED, _CONDITIONAL, False, None),
        (_CONDITIONAL_DATA, _CONDITIONAL_ACCEPTED, _CONDITIONAL, "cheap", None),
        (_CONDITIONAL_DATA, _CONDITIONAL_ACCEPTED, _CONDITIONAL, "full", None),
        (_PATH_DATA, _PATH_DAG, _PATH_SPECIFIC, False, None),
        (_DISTRIBUTION_DATA, _DISTRIBUTION_DAG, _DISTRIBUTION, False, None),
        (_IR_DATA, _IR_DAG, _IR, False, None),
        (_IR_DATA, _IR_ACCEPTED, _IR, False, None),
    ],
    ids=[
        "ate_explicit_none",
        "ate_explicit_cheap",
        "ate_explicit_full",
        "ate_accepted_none",
        "ate_accepted_cheap",
        "ate_accepted_full",
        "ate_explicit_bayesian_none",
        "ate_explicit_bayesian_cheap",
        "ate_explicit_bayesian_full",
        "ate_accepted_bayesian_none",
        "ate_accepted_bayesian_cheap",
        "ate_accepted_bayesian_full",
        "curve_explicit_none",
        "curve_accepted_none",
        "conditional_explicit_none",
        "conditional_explicit_cheap",
        "conditional_explicit_full",
        "conditional_accepted_none",
        "conditional_accepted_cheap",
        "conditional_accepted_full",
        "path_specific_explicit_none",
        "distribution_explicit_none",
        "intervention_response_explicit_none",
        "intervention_response_accepted_none",
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
    if hasattr(prepared, "evidence_status"):
        assert prepared.evidence_status == "licensed"
    click = prepared.estimate(data, seed=1)
    assert getattr(click, "evidence_status", None) == "licensed"
    if isinstance(query, antecedent.ResponseCurve):
        assert click.response is not None and fresh.response is not None
        assert click.response.values == fresh.response.values
        return
    if isinstance(query, antecedent.InterventionResponse):
        assert isinstance(click.estimate, float) and isinstance(fresh.estimate, float)
        assert math.isfinite(click.estimate) and math.isfinite(fresh.estimate)
        assert click.estimate == fresh.estimate
        return
    assert math.isfinite(click.ate) and math.isfinite(fresh.ate)
    assert abs(click.ate - fresh.ate) < 1e-12
