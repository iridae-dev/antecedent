"""1.4 numeric evidence for licensed Cpdag/Pag ConditionalEffect cells."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import pytest

antecedent = pytest.importorskip("antecedent")


_ROOT = pathlib.Path(__file__).resolve().parents[2]
_ATE = json.loads(
    (_ROOT / "conformance" / "estimate" / "cpdag_ate_envelope" / "expected.json").read_text()
)
_CLASS = json.loads(
    (_ROOT / "conformance" / "response" / "class_aware_envelope" / "expected.json").read_text()
)


def _expand_contingency(pin: dict[str, Any]) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {name: [] for name in pin["columns"]}
    for cell in pin["contingency_table"]:
        count = int(cell["count"])
        for name in pin["columns"]:
            values[name].extend([float(cell[name])] * count)
    return {name: np.asarray(column, dtype=np.float64) for name, column in values.items()}


def _cpdag(*, accepted: bool):
    spec = _ATE["graph"]
    graph = antecedent.Cpdag.from_directed_undirected(
        _ATE["columns"],
        [tuple(edge) for edge in spec["directed_edges"]],
        [tuple(edge) for edge in spec["undirected_edges"]],
    )
    if not accepted:
        return graph
    return antecedent.AcceptedGraph.from_graph(graph, algorithm_id="fixture.cpdag-conditional")


def _pag(*, accepted: bool):
    spec = _CLASS["pag"]["graph"]
    graph = antecedent.Pag.from_marked_edges(
        _CLASS["columns"],
        [tuple(edge) for edge in spec["marked_edges"]],
    )
    if not accepted:
        return graph
    return antecedent.AcceptedGraph.from_graph(graph, algorithm_id="fixture.pag-conditional")


def _query() -> antecedent.ConditionalEffect:
    spec = _ATE["query"]
    return antecedent.ConditionalEffect(
        spec["treatment"],
        spec["outcome"],
        _ATE["conditional"]["modifier"],
        control_level=float(spec["control_level"]),
        active_level=float(spec["active_level"]),
    )


@pytest.mark.parametrize("accepted", [False, True], ids=["explicit", "accepted"])
@pytest.mark.parametrize("class_name", ["cpdag", "pag"])
@pytest.mark.parametrize("refute", [False, "cheap", "full"], ids=["none", "cheap", "full"])
def test_class_aware_conditional_pins(accepted: bool, class_name: str, refute) -> None:
    data = _expand_contingency(_ATE)
    graph = _cpdag(accepted=accepted) if class_name == "cpdag" else _pag(accepted=accepted)
    query = _query()
    freq = _ATE["conditional"]["frequentist"]
    fresh = antecedent.analyze(data, graph=graph, query=query, refute=refute, bootstrap=0, seed=1)
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        graph=graph,
        query=query,
        refute=refute,
        bootstrap=0,
        seed=1,
        latency="interactive",
    )
    click = prepared.estimate(data, seed=1)
    assert prepared.structure_source == ("accepted" if accepted else "explicit")
    assert fresh.evidence_status == click.evidence_status == "licensed"
    assert fresh.plan.identifier == "generalized.adjustment"
    assert fresh.plan.estimator == freq["estimator"]
    assert fresh.identification.status == "GraphDependent"
    assert fresh.ate == pytest.approx(freq["expected_ate"], abs=freq["absolute_tolerance"])
    assert click.ate == pytest.approx(freq["expected_ate"], abs=freq["absolute_tolerance"])


@pytest.mark.parametrize("accepted", [False, True], ids=["explicit", "accepted"])
@pytest.mark.parametrize("class_name", ["cpdag", "pag"])
def test_class_aware_conditional_bayesian_pins(accepted: bool, class_name: str) -> None:
    data = _expand_contingency(_ATE)
    graph = _cpdag(accepted=accepted) if class_name == "cpdag" else _pag(accepted=accepted)
    bayes = _ATE["conditional"]["bayesian"]
    inference = antecedent.Bayesian(
        backend="conjugate",
        n_draws=int(bayes["n_draws"]),
        prior_scale=float(bayes["prior_scale"]),
    )
    result = antecedent.analyze(
        data,
        graph=graph,
        query=_query(),
        inference=inference,
        refute=False,
        bootstrap=0,
        seed=int(bayes["seed"]),
    )
    assert result.plan.estimator == bayes["estimator"]
    assert result.ate == pytest.approx(bayes["expected_ate"], abs=bayes["absolute_tolerance"])


@pytest.mark.parametrize("class_name", ["dag", "cpdag", "pag"])
def test_conditional_identification_does_not_reuse_mediator_ate(class_name: str) -> None:
    names = ["t", "w", "y"]
    edges = [("t", "w"), ("w", "y")]
    if class_name == "dag":
        graph = antecedent.Dag.from_edges(names, edges)
    elif class_name == "cpdag":
        graph = antecedent.Cpdag.from_directed_undirected(names, edges, [])
    else:
        graph = antecedent.Pag.from_marked_edges(names, [(a, b, "tail", "arrow") for a, b in edges])
    query = antecedent.ConditionalEffect("t", "y", "w")
    if class_name == "dag":
        with pytest.raises(antecedent.errors.CausalCompileError, match="not identified"):
            antecedent.identify(graph=graph, query=query)
    else:
        result = antecedent.identify(graph=graph, query=query)
        assert result.status == "NotIdentified"
