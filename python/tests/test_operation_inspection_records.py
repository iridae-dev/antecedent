"""Prepared inspection records published by the remaining checked operation routes.

Each test prepares a small study, estimates it through the canonical lifecycle and reads the
sealed route's ``plan_view().operation`` record. The source preparation keeps the generic
``estimator_route`` record: estimation is what installs the checked operation.
"""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent.discovery import GraphPosterior
from antecedent.graph import TemporalCpdag
from antecedent.outcomes import require_success

from known_truth import temporal_mediation_series

_NAMES = ["t", "y", "z"]
_EDGES = [("z", "t"), ("z", "y"), ("t", "y")]


def _confounded(n: int = 600, seed: int = 5) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = 0.6 * z + rng.normal(size=n)
    y = 1.0 + 2.0 * t + 1.5 * z + rng.normal(scale=0.5, size=n)
    return {"t": t, "y": y, "z": z, "w": rng.normal(size=n)}


def _estimated_operation(data, **options):
    options.setdefault("refute", "none")
    options.setdefault("execution", ant.Execution(seed=3, threads=1))
    prepared = ant.prepare(data, **options)
    assert prepared.plan_view().operation == {"kind": "estimator_route"}
    outcome = ant.estimate(prepared)
    require_success(outcome)
    return outcome.study.plan_view().operation


@pytest.mark.parametrize("horizons", [[1], [1, 2]])
def test_temporal_class_mediation_publishes_its_class_proof_and_per_horizon_targets(horizons):
    graph = TemporalCpdag.from_lagged_edges(
        ["t", "m", "y"], [("t", 1, "m", 0), ("t", 1, "y", 0), ("m", 0, "y", 0)], None
    )
    query = ant.TemporalMediationEffect("t", "m", "y", contrast="mediated", horizons=horizons)
    operation = _estimated_operation(
        temporal_mediation_series(160),
        query=query,
        graph=graph,
        inference=ant.Frequentist(bootstrap=0),
    )
    assert operation["kind"] == "temporal_class_mediation"
    assert operation["graph_class"] == "TemporalCpdag"
    assert operation["structure_source"] == "explicit"
    assert operation["estimator"] == "temporal.mediation"
    assert operation["validation"] == "none"
    assert operation["bootstrap_replicates"] == 0
    assert operation["horizons"] == horizons
    masses = operation["proof_masses"]
    assert [record["horizon"] for record in masses] == horizons
    for record in masses:
        assert record["atom_count"] == 1
        assert record["basis"] == "completion_enumeration_weight"
        assert record["identified_mass"] == pytest.approx(1.0)
        assert record["unidentified_mass"] == pytest.approx(0.0)
        assert record["truncated_completions"] == 0
    if len(horizons) == 1:
        assert operation["atom_count"] == 1
        assert operation["identified_mass"] == pytest.approx(1.0)
        assert operation["unidentified_mass"] == pytest.approx(0.0)
    else:
        assert operation["atom_count"] is None
        assert operation["identified_mass"] is None
        assert operation["unidentified_mass"] is None
    assert operation["proof_signature"].startswith("class=TemporalCpdag;")
    # One scientific target per horizon: the same single completion, proved identified.
    targets = operation["scientific_targets"]
    assert [target["horizon"] for target in targets] == horizons
    for target in targets:
        assert target["completion_index"] == 0
        assert target["atom_weight"] == pytest.approx(1.0)
        assert target["completion_weight"] == pytest.approx(1.0)
        assert target["identification_status"] == "IdentifiedUnderParametricRestrictions"
        assert "Mediation" in target["scientific_target"]


def test_bayesian_class_conditional_route_publishes_its_class_and_completion_mass():
    data = _confounded()
    graph = ant.Cpdag.from_directed_undirected(["t", "y", "z", "w"], [*_EDGES, ("w", "y")], [])
    operation = _estimated_operation(
        data,
        graph=graph,
        query=ant.ConditionalEffect("t", "y", modifier="w"),
        inference=ant.Bayesian(backend="conjugate", n_draws=64),
    )
    assert operation["kind"] == "bayesian_class_conditional"
    assert operation["graph_class"] == "Cpdag"
    assert operation["estimator"] == "conditional.bayesian"
    assert operation["adjustment_set"] == ["z"]
    assert "n_draws: 64" in operation["inference"]
    assert operation["completion_count"] == 1
    assert operation["identified_mass"] == pytest.approx(1.0)
    assert operation["unresolved_mass"] == pytest.approx(0.0)
    assert "ConditionalEffectQuery" in operation["query"]


def test_bayesian_specialist_route_publishes_its_identifier_estimator_and_draw_budget():
    n = 1200
    instrument = np.arange(n) % 2 * 1.0
    confounder = ((np.arange(n) % 17) - 8.0) / 5.0
    treatment = 0.6 * instrument + confounder
    outcome = 2.0 * treatment + confounder
    operation = _estimated_operation(
        {"t": treatment, "y": outcome, "z": instrument},
        graph=ant.Dag.from_edges(["t", "y", "z"], [("z", "t"), ("t", "y")]),
        query=ant.AverageEffect("t", "y"),
        identifier="iv",
        estimator=ant.Estimator.IV_BAYESIAN_JOINT_LINEAR,
        inference=ant.Bayesian(backend="conjugate", n_draws=64),
    )
    assert operation["kind"] == "bayesian_specialist"
    assert operation["identifier"] == "iv"
    assert operation["estimator"] == "iv.bayesian_joint_linear"
    assert operation["n_draws"] == 64
    assert operation["prior_scale"] == pytest.approx(10.0)
    assert operation["rd"] is None
    assert "AverageEffectQuery" in operation["query"]


def test_class_graph_posterior_effect_publishes_its_atoms_weights_and_identification():
    data = _confounded()
    class_graph = ant.Cpdag.from_directed_undirected(_NAMES, _EDGES, [])
    posterior = GraphPosterior.from_graphs(["t", "y", "z"], [0.7, 0.3], [class_graph, class_graph])
    operation = _estimated_operation(
        {name: data[name] for name in _NAMES},
        discovery=posterior,
        query=ant.AverageEffect("t", "y"),
    )
    assert operation["kind"] == "class_graph_posterior_effect"
    assert operation["conditional"] is False
    assert operation["estimator"] == "linear.adjustment.ate"
    assert operation["weights"] == pytest.approx([0.7, 0.3])
    assert len(operation["graph_keys"]) == 2
    assert operation["identified"] == ["Identified", "Identified"]
    assert operation["class_atom_count"] == 1, "equal class graphs share one proved atom"
    assert operation["bootstrap_replicates"] > 0


def test_derivative_response_publishes_its_adjustment_and_cell_budget():
    data = _confounded()
    operation = _estimated_operation(
        data,
        graph=ant.Dag.from_edges(_NAMES, _EDGES),
        query=ant.AverageDerivative("t", "y"),
    )
    assert operation["kind"] == "derivative_response"
    assert operation["identifier"] == "response.backdoor"
    assert operation["estimator"] == "response.riesz_ade"
    assert operation["adjustment_set"] == ["z"]
    assert operation["max_derivative_cells"] >= 1
    assert "AverageDerivative" in operation["query"]
