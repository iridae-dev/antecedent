"""Python facade lifecycle checks for sealed static DAG response operations."""

from __future__ import annotations

import math

import antecedent as ant
import numpy as np
import pytest


def _table(n: int = 240) -> dict[str, np.ndarray]:
    z = np.array([math.sin(i / 17.0) for i in range(n)])
    t = z + np.array([math.cos(i / 11.0) for i in range(n)])
    y = 1.0 + 2.0 * t + 0.8 * z
    return {"t": t, "y": y, "z": z}


_EDGES = [("z", "t"), ("z", "y"), ("t", "y")]


@pytest.mark.parametrize("accepted", [False, True], ids=["explicit", "accepted"])
@pytest.mark.parametrize("kind", ["curve", "intervention"])
def test_checked_static_dag_response_python_lifecycle(accepted: bool, kind: str) -> None:
    data = _table()
    dag = ant.Dag.from_edges(["t", "y", "z"], _EDGES)
    graph = (
        ant.AcceptedGraph.from_graph(dag, algorithm_id="response-lifecycle") if accepted else _EDGES
    )
    if kind == "curve":
        query = ant.ResponseCurve("t", "y", grid=[-0.5, 0.0, 0.5])
        expected = np.array([1.0 + 2.0 * level + 0.8 * np.mean(data["z"]) for level in query.grid])
        dependency = "dependencies.checked_response_grid_operation"
    else:
        query = ant.InterventionResponse("y", intervention=ant.intervention.Set("t", 0.25))
        expected = np.array([1.0 + 2.0 * 0.25 + 0.8 * np.mean(data["z"])])
        dependency = "dependencies.checked_intervention_response_operation"

    # Deliberately release the builder/reference before executing the prepared handle.
    builder = ant.prepare(
        data,
        graph=graph,
        query=query,
        estimator="response.kennedy_dr" if kind == "curve" else "response.intervention_gcomp",
        refute="none",
        bootstrap=0,
        seed=4,
    )
    prepared = builder
    del builder, query, graph, dag

    info = prepared.checked_static_dag_response_info()
    assert info is not None
    assert (
        info["query"].startswith("MeanCurve")
        if kind == "curve"
        else info["query"].startswith("InterventionResponse")
    )
    assert info["identifier"] == "response.backdoor"
    assert info["estimator"] == (
        "response.kennedy_dr" if kind == "curve" else "response.intervention_gcomp"
    )
    assert info["validation"] == "none"
    assert info["grid_members"] == ([-0.5, 0.0, 0.5] if kind == "curve" else [])

    first = prepared.estimate(data, seed=4)
    if kind == "curve":
        assert first.response is not None
        observed = np.asarray([row[0] for row in first.response.values])
    else:
        observed = np.asarray([first.effect])
    # The response estimator fits its declared nuisance model; the exact structural
    # law is the independent truth reference, with finite-sample nuisance tolerance.
    np.testing.assert_allclose(observed, expected, atol=0.03, rtol=0)

    shifted = {**data, "y": data["y"] + 0.3}
    refreshed = prepared.refresh(shifted, seed=4)
    if kind == "curve":
        assert refreshed.response is not None
        refreshed_values = np.asarray([row[0] for row in refreshed.response.values])
    else:
        refreshed_values = np.asarray([refreshed.effect])
    np.testing.assert_allclose(refreshed_values, observed + 0.3, atol=1e-9, rtol=0)

    consumed = ant.artifacts.accept(refreshed.export())
    assert consumed["accepts_as_verified_program"] == "false"
    assert dependency in consumed["unresolved"]
