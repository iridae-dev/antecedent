"""Numerical evidence for joint class-aware response identification and execution."""

import json
from pathlib import Path

import antecedent
import numpy as np
import pytest

_PIN = json.loads(
    (
        Path(__file__).resolve().parents[2] / "conformance/response/class_aware_envelope/joint.json"
    ).read_text()
)


def _data():
    rows = []
    for mask in range(32):
        t, z, w, r, v = [(mask >> i) & 1 for i in range(5)]
        rows.extend([[t, z, 1 + 2 * t + 3 * z + 4 * w, w, r, v]] * (160 if z == w else 40))
    matrix = np.asarray(rows, dtype=np.float64)
    return {name: matrix[:, i].copy() for i, name in enumerate(_PIN["columns"])}


def _graph(kind, accepted):
    names = _PIN["columns"]
    edges = [(names[a], names[b]) for a, b in _PIN["directed"]]
    if kind == "cpdag":
        graph = antecedent.Cpdag.from_directed_undirected(names, edges, [])
    elif kind == "pag":
        graph = antecedent.Pag.from_marked_edges(names, [(a, b, "tail", "arrow") for a, b in edges])
    else:
        graph = antecedent.Dag.from_edges(names, edges)
    return antecedent.AcceptedGraph.from_graph(graph) if accepted else graph


@pytest.mark.parametrize("kind", ["dag", "cpdag", "pag"])
@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("bayesian", [False, True])
def test_joint_response_identify_prepare_and_estimate(kind, accepted, bayesian):
    graph = _graph(kind, accepted)
    query = antecedent.InterventionResponse(
        "y", intervention=[antecedent.intervention.Set("t", 1), antecedent.intervention.Set("z", 1)]
    )
    identified = antecedent.identify(graph=graph, query=query)
    assert identified.adjustment_set == ["w"]
    kwargs = {"refute": False, "bootstrap": 0, "seed": 7}
    if bayesian:
        kwargs["inference"] = antecedent.Bayesian()
    data = _data()
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data, graph=graph, query=query, **kwargs
    )
    result = prepared.estimate(data, seed=7)
    tolerance = _PIN["bayesian_tolerance"] if bayesian else _PIN["tolerance"]
    assert float(np.asarray(result.response.values).reshape(-1)[0]) == pytest.approx(
        _PIN["mean"], abs=tolerance
    )
    fresh = antecedent.analyze(data, graph=graph, query=query, **kwargs)
    assert float(np.asarray(fresh.response.values).reshape(-1)[0]) == pytest.approx(
        _PIN["mean"], abs=tolerance
    )
    for executed in (result, fresh):
        assert executed.certificate is not None
        assert [item["name"] for item in executed.certificate["treatments"]] == ["t", "z"]
        spec = antecedent.handoff.econml(executed)
        assert spec.treatments == ("t", "z")
        columns = spec.columns(data)
        np.testing.assert_array_equal(columns["T"], np.column_stack([data["t"], data["z"]]))
        np.testing.assert_array_equal(columns["W"][:, 0], data["w"])


@pytest.mark.parametrize("kind", ["dag", "cpdag", "pag"])
@pytest.mark.parametrize(
    "policy",
    [
        antecedent.intervention.Bernoulli("t", 0.5),
        antecedent.intervention.Gaussian("t", 0.5, 0.25),
        antecedent.intervention.Categorical("t", [0.5, 0.5]),
    ],
)
def test_bayesian_joint_policy_mean(kind, policy):
    query = antecedent.InterventionResponse(
        "y", intervention=[policy, antecedent.intervention.Set("z", 1)]
    )
    result = antecedent.analyze(
        _data(),
        graph=_graph(kind, False),
        query=query,
        inference=antecedent.Bayesian(),
        refute=False,
        bootstrap=0,
        seed=7,
    )
    assert float(np.asarray(result.response.values).reshape(-1)[0]) == pytest.approx(
        7.0, abs=_PIN["bayesian_tolerance"]
    )
