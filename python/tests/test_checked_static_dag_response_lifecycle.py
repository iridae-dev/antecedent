"""Python facade lifecycle checks for sealed static DAG response operations."""

from __future__ import annotations

import math

import antecedent as ant
import numpy as np
import pytest
from antecedent import joint_panel
from antecedent.errors import CausalValueError
from antecedent.outcomes import require_success


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
        inference=ant.Frequentist(bootstrap=0),
        execution=ant.Execution(seed=4),
    )
    prepared = builder
    del builder, query, graph, dag

    info = prepared.plan_view().operation
    assert info is not None
    assert info["kind"] == "static_dag_response"
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

    original_inspection = prepared.inspect()
    first = require_success(prepared.estimate())
    if kind == "curve":
        assert first.response is not None
        observed = np.asarray([row[0] for row in first.response.values])
    else:
        observed = np.asarray([first.effect])
    # The response estimator fits its declared nuisance model; the exact structural
    # law is the independent truth reference, with finite-sample nuisance tolerance.
    np.testing.assert_allclose(observed, expected, atol=0.03, rtol=0)

    shifted = {**data, "y": data["y"] + 0.3}
    refreshed = require_success(prepared.rebind(data=shifted).estimate())
    if kind == "curve":
        assert refreshed.response is not None
        refreshed_values = np.asarray([row[0] for row in refreshed.response.values])
    else:
        refreshed_values = np.asarray([refreshed.effect])
    np.testing.assert_allclose(refreshed_values, observed + 0.3, atol=1e-9, rtol=0)

    consumed = ant.artifacts.accept(bytes(refreshed.export()))
    assert consumed["accepts_as_verified_program"] == "false"
    assert dependency in consumed["unresolved"]
    assert prepared.inspect() == original_inspection
    assert first.study.inspect().data_snapshot_id == first.data_snapshot_id
    repeated = require_success(prepared.estimate())
    if kind == "curve":
        assert repeated.response.values == first.response.values
    else:
        assert repeated.effect == first.effect


def _noisy_table(n: int = 240) -> dict[str, np.ndarray]:
    table = _table(n)
    table["y"] = table["y"] + 0.1 * np.array([math.sin(i / 3.7) for i in range(n)])
    return table


def _worlds(*levels: float) -> list[dict[str, object]]:
    return [
        {"id": index, "reference": index == 0, "levels": {"t": level}}
        for index, level in enumerate(levels)
    ]


def test_joint_response_worlds_share_one_fit_and_carry_their_joint_covariance() -> None:
    data = _noisy_table()
    study = ant.prepare(
        data,
        graph=_EDGES,
        query=ant.InterventionResponse("y", intervention=ant.intervention.Set("t", 0.25)),
        estimator="response.intervention_gcomp",
        refute="none",
        inference=ant.Frequentist(bootstrap=0),
        execution=ant.Execution(seed=4),
    )
    # Independent oracle: ordinary least squares of y on [1, t, z]; a world's mean is the
    # fitted response at its level with every observed covariate row kept.
    design = np.column_stack([np.ones(len(data["t"])), data["t"], data["z"]])
    beta, *_ = np.linalg.lstsq(design, data["y"], rcond=None)

    def truth(level: float) -> float:
        return float(beta[0] + beta[1] * level + beta[2] * np.mean(data["z"]))

    levels = (-0.5, 0.0, 0.5)
    coefficients = [-2.0, 1.0, 1.0]
    compiled = require_success(
        joint_panel.compile_joint_response(
            study,
            outcome="y",
            worlds=_worlds(*levels),
            contrast={
                "kind": "linear",
                "terms": [[i, c] for i, c in enumerate(coefficients)],
                "reference": 0,
            },
        )
    )
    before = compiled.inspect()
    assert before["world_ids"] == [0, 1, 2]
    assert before["contrast_coefficients"] == coefficients
    assert before["fitted"] is False
    assert before["calibration"] == "unmeasured"

    law = require_success(compiled.execute(execution=ant.Execution(seed=7)))
    assert law.world_ids == (0, 1, 2)
    for got, level in zip(law.means, levels, strict=True):
        assert got == pytest.approx(truth(level), abs=1e-8)
    assert compiled.inspect()["fitted"] is True
    assert law.calibration == "unmeasured"
    assert law.uncertainty == "sampling_covariance"
    # One coefficient covariance gives every cross-world term: symmetric and positive.
    cov = np.asarray(law.covariance)
    assert cov.shape == (3, 3)
    np.testing.assert_allclose(cov, cov.T, atol=1e-14)
    assert np.all(np.diag(cov) > 0.0)
    assert np.all(np.linalg.eigvalsh(cov) > -1e-12)
    # The assembled contrast is the same linear functional of the same joint law.
    value, se = law.contrast
    assert value == pytest.approx(
        sum(c * truth(x) for c, x in zip(coefficients, levels, strict=True)), abs=1e-8
    )
    c = np.asarray(coefficients)
    assert se**2 == pytest.approx(float(c @ cov @ c), rel=1e-9, abs=1e-14)
    assert require_success(compiled.execute(execution=ant.Execution(seed=7))).means == law.means

    # Retargeting reuses the source, prior and fit and moves only the world projections.
    other = require_success(compiled.retarget(outcome="y", worlds=_worlds(0.0, 0.25)))
    again = require_success(other.execute(execution=ant.Execution(seed=7)))
    assert again.means == pytest.approx((truth(0.0), truth(0.25)), abs=1e-8)
    assert again.source_digest == law.source_digest
    assert again.target != law.target

    artifact = bytes(law.export())
    replayed = require_success(compiled.consume_current(artifact))
    assert replayed.means == law.means
    assert replayed.identification == law.identification
    with pytest.raises(CausalValueError):
        require_success(other.consume_current(artifact))

    # A world outside the observed range of its intervention coordinate is never extrapolated.
    with pytest.raises(ant.CausalError, match="observed marginal support"):
        require_success(
            joint_panel.compile_joint_response(study, outcome="y", worlds=_worlds(0.0, 25.0))
        )
    with pytest.raises(ant.CausalError):
        require_success(
            joint_panel.compile_joint_response(
                study,
                outcome="y",
                worlds=[{"id": 0, "reference": True, "levels": {"z": 0.0}}],
            )
        )
