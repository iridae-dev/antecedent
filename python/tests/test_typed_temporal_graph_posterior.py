"""Supplied temporal atoms retain typed graph geometry and feed real execution."""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent.discovery import GraphPosterior
from antecedent.graph import TemporalAdmg, TemporalCpdag, TemporalDag, TemporalPag
from antecedent.outcomes import require_success
from antecedent.query import TemporalQuery


def _edge(n, source, target):
    return 1 << (source * (n - 1) + (target if target < source else target - 1))


@pytest.mark.parametrize("family", ["dag", "cpdag", "pag", "admg"])
def test_typed_temporal_atoms_preserve_weights_lags_and_family(family):
    names = ["x", "m", "y"]
    directed = [("x", 2, "m", 0), ("m", 0, "y", 0)]
    if family == "dag":
        graph = TemporalDag.from_lagged_edges(names, directed)
    elif family == "cpdag":
        graph = TemporalCpdag.from_lagged_edges(names, directed)
    elif family == "pag":
        graph = TemporalPag.from_marked_lagged_edges(
            names,
            [
                ("x", 2, "m", 0, "tail", "arrow"),
                ("m", 0, "y", 0, "circle", "arrow"),
            ],
        )
    else:
        graph = TemporalAdmg.from_lagged_edges(names, directed, [("y", 1, "x", 0)])
    posterior = GraphPosterior.from_graphs(names, [0.7, 0.3], [graph, graph])
    assert posterior.names == names
    assert posterior.weights == [0.7, 0.3]
    assert (
        posterior.atom_kind
        == {"dag": "Dag", "cpdag": "Cpdag", "pag": "Pag", "admg": "TemporalAdmg"}[family]
    )
    assert posterior.n_graphs == 2
    assert posterior.max_lag == 2
    assert posterior.lag_masks == [1 << 10, 1 << 10]
    assert posterior.adjacency == [_edge(3, 1, 2)] * 2
    assert posterior.lagged_edge_marginals[10] == pytest.approx(1.0)
    assert sum(posterior.lagged_edge_marginals) == pytest.approx(1.0)
    assert posterior.ess == pytest.approx(1 / (0.7**2 + 0.3**2))
    if family == "pag":
        assert posterior.mark_masks == [_edge(3, 1, 2)] * 2
    if family == "admg":
        assert posterior.lag_bidirected_masks == [1 << 6] * 2


def test_temporal_class_posterior_executes_same_derivative_as_explicit_class_and_exports():
    n = 80
    x = 1 + np.sin(np.arange(n) * 0.37)
    y = np.zeros(n)
    y[1:] = 3 + 2 * x[:-1]
    table = ant.data.table({"x": x, "y": y})
    graph = TemporalCpdag.from_lagged_edges(["x", "y"], [("x", 1, "y", 0)])
    posterior = GraphPosterior.from_graphs(["x", "y"], [1.0], [graph])
    query = TemporalQuery(ant.AverageDerivative("x", "y"), max_history_lag=1)
    options = dict(
        query=query,
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
        execution=ant.Execution(seed=11),
    )
    explicit = ant.prepare(table, graph=graph, **options)
    supplied = ant.prepare(table, discovery=posterior, **options)
    assert explicit.inspect().stage == "prepared" and supplied.inspect().stage == "prepared"
    assert supplied.plan_view()
    assert supplied.inspect().stage == "prepared"
    result = require_success(supplied.estimate())
    reference = require_success(explicit.estimate())
    np.testing.assert_allclose(np.asarray(result.estimate).reshape(-1), [2.0], atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        np.asarray(result.estimate), np.asarray(reference.estimate), atol=1e-12, rtol=0
    )
    restored = require_success(ant.load(result.export()))
    assert restored.program_id == result.program_id
    assert restored.claim_id == result.claim_id
    assert restored.answer == result.answer
    assert posterior.lag_masks == [2]


def test_typed_temporal_posterior_refuses_mixed_families_and_reordered_names():
    names = ["x", "y"]
    edges = [("x", 1, "y", 0)]
    dag = TemporalDag.from_lagged_edges(names, edges)
    cpdag = TemporalCpdag.from_lagged_edges(names, edges)
    with pytest.raises(ValueError, match="share one graph family"):
        GraphPosterior.from_graphs(names, [0.5, 0.5], [dag, cpdag])
    with pytest.raises(ValueError, match="names and order"):
        GraphPosterior.from_graphs(names[::-1], [1.0], [dag])


def test_temporal_pag_factory_refuses_lagged_circle_marks_instead_of_orienting_them():
    graph = TemporalPag.from_marked_lagged_edges(
        ["x", "y"],
        [
            ("x", 1, "y", 0, "circle", "arrow"),
        ],
    )
    with pytest.raises(ValueError, match="cannot retain these lagged endpoint marks"):
        GraphPosterior.from_graphs(["x", "y"], [1.0], [graph])
