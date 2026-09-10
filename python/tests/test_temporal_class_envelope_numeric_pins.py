"""1.4 numeric evidence for licensed TemporalCpdag/Pag Pulse cells."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

antecedent = pytest.importorskip("antecedent")


_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PIN = json.loads(
    (_ROOT / "conformance" / "estimate" / "temporal_class_envelope" / "expected.json").read_text()
)


def _series(pin: dict) -> dict[str, np.ndarray]:
    n = int(pin["n"])
    z = np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)], dtype=np.float64)
    t = 0.3 + 0.4 * z + 0.05 * np.sin(np.arange(n, dtype=np.float64) * 0.017)
    y = np.zeros(n, dtype=np.float64)
    y[1:] = 1.0 + 2.0 * t[:-1] + 0.5 * z[:-1]
    return {"t": t, "y": y, "z": z}


def _pulse(pin: dict) -> antecedent.PulseEffect:
    spec = pin["query"]
    return antecedent.PulseEffect(
        treatment=spec["treatment"],
        outcome=spec["outcome"],
        treatment_lag=abs(int(spec["treatment_offset"])),
        horizon_steps=int(spec["horizon_steps"]),
        active_level=float(spec["active_level"]),
    )


def _sustained(pin: dict) -> antecedent.SustainedEffect:
    spec = pin["query"]
    offset = int(spec["treatment_offset"])
    return antecedent.SustainedEffect(
        treatment=spec["treatment"],
        outcome=spec["outcome"],
        window=(offset, offset),
        treatment_lag=abs(offset),
        horizon_steps=int(spec["horizon_steps"]),
        active_level=float(spec["active_level"]),
    )


def _cpdag(*, accepted: bool):
    graph = antecedent.graph.TemporalCpdag.from_lagged_edges(
        _PIN["columns"],
        [tuple(edge) for edge in _PIN["cpdag"]["directed"]],
        [tuple(edge) for edge in _PIN["cpdag"]["undirected"]],
    )
    if not accepted:
        return graph
    return antecedent.AcceptedGraph.from_graph(graph, algorithm_id="fixture.temporal-cpdag")


def _pag(*, accepted: bool):
    directed = [(a, la, b, lb, "tail", "arrow") for a, la, b, lb in _PIN["pag"]["directed"]]
    circles = [(a, la, b, lb, "circle", "circle") for a, la, b, lb in _PIN["pag"]["circle_circle"]]
    graph = antecedent.graph.TemporalPag.from_marked_lagged_edges(
        _PIN["columns"], directed + circles
    )
    if not accepted:
        return graph
    return antecedent.AcceptedGraph.from_graph(graph, algorithm_id="fixture.temporal-pag")


@pytest.mark.parametrize("accepted", [False, True], ids=["explicit", "accepted"])
@pytest.mark.parametrize(
    "class_name,graph_fn,diag",
    [
        ("cpdag", _cpdag, "identify.temporal_cpdag.envelope"),
        ("pag", _pag, "identify.temporal_pag.envelope"),
    ],
)
def test_temporal_class_pulse_pin(accepted: bool, class_name: str, graph_fn, diag: str) -> None:
    data = _series(_PIN)
    graph = graph_fn(accepted=accepted)
    query = _pulse(_PIN)
    if class_name == "pag":
        identified = antecedent.identify(graph=graph, query=query)
        assert identified.status == "NotIdentified"
        assert len(identified.certificate["cases"]) == 3
        with pytest.raises(antecedent.errors.CausalCompileError, match="no identified mass"):
            antecedent.analyze(data, graph=graph, query=query, refute=False, bootstrap=0)
        return
    section = _PIN[class_name]
    expected = float(section["pulse"]["ate"])
    tol = float(section["pulse"]["absolute_tolerance"])

    fresh = antecedent.analyze(data, graph=graph, query=query, refute=False, bootstrap=0, seed=1)
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        graph=graph,
        query=query,
        refute=False,
        bootstrap=0,
        seed=1,
        latency="interactive",
    )
    click = prepared.estimate(data, seed=1)

    expected_source = "accepted" if accepted else "explicit"
    assert prepared.structure_source == expected_source
    assert fresh.plan.identifier == click.plan.identifier == section["identification"]["identifier"]
    assert fresh.plan.estimator == click.plan.estimator == section["pulse"]["estimator"]
    assert fresh.identification.status == click.identification.status == "PartiallyIdentified"
    assert fresh.ate == pytest.approx(expected, abs=tol)
    assert click.ate == pytest.approx(expected, abs=tol)
    assert any(diag in diagnostic for diagnostic in fresh.diagnostics)
    assert any(diag in diagnostic for diagnostic in click.diagnostics)
    assert all(
        not diagnostic.startswith("exec.identify.cached") for diagnostic in fresh.diagnostics
    )
    assert any(diagnostic.startswith("exec.identify.cached") for diagnostic in click.diagnostics)
    assert any(
        "estimate.envelope.se_omits_between_atom_variance" in diagnostic
        for diagnostic in fresh.diagnostics
    )

    cheap = antecedent.analyze(data, graph=graph, query=query, refute="cheap", bootstrap=0, seed=1)
    assert cheap.validation.ran
    assert any("refute.envelope.effect_mixture" in diagnostic for diagnostic in cheap.diagnostics)
    assert cheap.ate == pytest.approx(expected, abs=tol)


@pytest.mark.parametrize("class_name,graph_fn", [("cpdag", _cpdag), ("pag", _pag)])
def test_temporal_class_single_step_sustained_matches_pulse(class_name: str, graph_fn) -> None:
    data = _series(_PIN)
    graph = graph_fn(accepted=False)
    if class_name == "pag":
        for query in (_pulse(_PIN), _sustained(_PIN)):
            with pytest.raises(antecedent.errors.CausalCompileError, match="no identified mass"):
                antecedent.analyze(data, graph=graph, query=query, refute=False, bootstrap=0)
        return
    pulse = antecedent.analyze(
        data, graph=graph, query=_pulse(_PIN), refute=False, bootstrap=0, seed=1
    )
    sustained = antecedent.analyze(
        data, graph=graph, query=_sustained(_PIN), refute=False, bootstrap=0, seed=1
    )
    assert pulse.ate == pytest.approx(sustained.ate, abs=1e-10)
    assert class_name in {"cpdag", "pag"}


@pytest.mark.parametrize("class_name,graph_fn", [("cpdag", _cpdag), ("pag", _pag)])
def test_temporal_class_bayesian_refuses_at_build(class_name: str, graph_fn) -> None:
    data = _series(_PIN)
    graph = graph_fn(accepted=False)
    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="1.7"):
        antecedent.analyze(
            data,
            graph=graph,
            query=_pulse(_PIN),
            inference=antecedent.Bayesian(n_draws=16, backend="conjugate"),
            refute=False,
            bootstrap=0,
            seed=1,
        )
    assert class_name in {"cpdag", "pag"}
