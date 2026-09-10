"""1.4 numeric evidence for licensed Cpdag/Pag response cells."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import pytest

antecedent = pytest.importorskip("antecedent")


_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PIN = json.loads(
    (_ROOT / "conformance" / "response" / "class_aware_envelope" / "expected.json").read_text()
)


def _expand_contingency(pin: dict[str, Any]) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {name: [] for name in pin["columns"]}
    for cell in pin["contingency_table"]:
        count = int(cell["count"])
        for name in pin["columns"]:
            values[name].extend([float(cell[name])] * count)
    return {name: np.asarray(column, dtype=np.float64) for name, column in values.items()}


def _continuous_linear(pin: dict[str, Any]) -> dict[str, np.ndarray]:
    n = int(pin["continuous"]["n"])
    z = np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)], dtype=np.float64)
    t = 0.3 + 0.4 * z + 0.2 * np.sin(np.arange(n, dtype=np.float64) * 0.017)
    y = 0.10 + 0.40 * t + 0.20 * z
    return {"t": t, "y": y, "z": z}


def _cpdag(*, accepted: bool):
    spec = _PIN["cpdag"]["graph"]
    graph = antecedent.Cpdag.from_directed_undirected(
        _PIN["columns"],
        [tuple(edge) for edge in spec["directed_edges"]],
        [tuple(edge) for edge in spec["undirected_edges"]],
    )
    if not accepted:
        return graph
    return antecedent.AcceptedGraph.from_graph(graph, algorithm_id="fixture.cpdag-response")


def _pag(*, accepted: bool):
    spec = _PIN["pag"]["graph"]
    graph = antecedent.Pag.from_marked_edges(
        _PIN["columns"],
        [tuple(edge) for edge in spec["marked_edges"]],
    )
    if not accepted:
        return graph
    return antecedent.AcceptedGraph.from_graph(graph, algorithm_id="fixture.pag-response")


def _curve() -> antecedent.ResponseCurve:
    spec = _PIN["query"]
    return antecedent.ResponseCurve(
        spec["treatment"],
        spec["outcome"],
        grid=[float(v) for v in _PIN["grid"]],
    )


def _intervention(level: float) -> antecedent.InterventionResponse:
    return antecedent.InterventionResponse(
        _PIN["query"]["outcome"],
        intervention=antecedent.intervention.Set(_PIN["query"]["treatment"], level),
    )


def _values(result) -> np.ndarray:
    assert result.response is not None
    return np.asarray(result.response.values, dtype=np.float64).reshape(-1)


@pytest.mark.parametrize("accepted", [False, True], ids=["explicit", "accepted"])
@pytest.mark.parametrize("class_name", ["cpdag", "pag"])
def test_class_aware_intervention_pins(accepted: bool, class_name: str) -> None:
    data = _expand_contingency(_PIN)
    graph = _cpdag(accepted=accepted) if class_name == "cpdag" else _pag(accepted=accepted)
    if class_name == "pag":
        identified = antecedent.identify(graph=graph, query=antecedent.AverageEffect("t", "y"))
        assert identified.status == "NotIdentified"
        assert identified.certificate["identified_weight"] == 0.0
        return
    section = _PIN[class_name]
    high = antecedent.analyze(
        data, graph=graph, query=_intervention(1.0), refute=False, bootstrap=0, seed=1
    )
    low = antecedent.analyze(
        data, graph=graph, query=_intervention(0.0), refute=False, bootstrap=0, seed=1
    )
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        graph=graph,
        query=_intervention(1.0),
        refute=False,
        bootstrap=0,
        seed=1,
        latency="interactive",
    )
    click = prepared.estimate(data, seed=1)
    assert prepared.structure_source == ("accepted" if accepted else "explicit")
    assert prepared.evidence_status == "licensed"
    tol = section["intervention"]["absolute_tolerance"]
    assert high.evidence_status == low.evidence_status == click.evidence_status == "licensed"
    assert high.identification.status == section["identification"]["status"]
    assert float(_values(high)[0]) == pytest.approx(section["intervention"]["do_1"], abs=tol)
    assert float(_values(low)[0]) == pytest.approx(section["intervention"]["do_0"], abs=tol)
    assert float(_values(high)[0]) - float(_values(low)[0]) == pytest.approx(
        section["ate_contrast"], abs=section["intervention"]["contrast_tolerance"]
    )
    assert float(_values(click)[0]) == pytest.approx(section["intervention"]["do_1"], abs=tol)


@pytest.mark.parametrize("accepted", [False, True], ids=["explicit", "accepted"])
@pytest.mark.parametrize("class_name", ["cpdag", "pag"])
def test_class_aware_curve_pins(accepted: bool, class_name: str) -> None:
    data = _continuous_linear(_PIN)
    graph = _cpdag(accepted=accepted) if class_name == "cpdag" else _pag(accepted=accepted)
    if class_name == "pag":
        identified = antecedent.identify(graph=graph, query=antecedent.AverageEffect("t", "y"))
        assert identified.status == "NotIdentified"
        assert identified.certificate["identified_weight"] == 0.0
        return
    section = _PIN[class_name]
    query = _curve()
    fresh = antecedent.analyze(data, graph=graph, query=query, refute=False, bootstrap=0, seed=1)
    ate = antecedent.analyze(
        data,
        graph=graph,
        query=antecedent.AverageEffect(
            _PIN["query"]["treatment"],
            _PIN["query"]["outcome"],
        ),
        refute=False,
        bootstrap=0,
        seed=1,
    )
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
    assert prepared.structure_source == ("accepted" if accepted else "explicit")
    expected = np.asarray(section["curve"]["mean"], dtype=np.float64)
    for result in (fresh, click):
        assert result.evidence_status == "licensed"
        assert result.identification.status == section["identification"]["status"]
        values = _values(result)
        assert values == pytest.approx(expected, abs=section["curve"]["absolute_tolerance"])
        assert values[1] - values[0] == pytest.approx(
            ate.ate, abs=section["curve"]["contrast_tolerance"]
        )


@pytest.mark.parametrize("accepted", [False, True], ids=["explicit", "accepted"])
@pytest.mark.parametrize("class_name", ["cpdag", "pag"])
def test_class_aware_bayesian_intervention_pins(accepted: bool, class_name: str) -> None:
    data = _expand_contingency(_PIN)
    graph = _cpdag(accepted=accepted) if class_name == "cpdag" else _pag(accepted=accepted)
    if class_name == "pag":
        identified = antecedent.identify(graph=graph, query=antecedent.AverageEffect("t", "y"))
        assert identified.status == "NotIdentified"
        assert identified.certificate["identified_weight"] == 0.0
        return
    bayes = _PIN["bayesian"]
    inference = antecedent.Bayesian(
        backend="conjugate",
        n_draws=int(bayes["n_draws"]),
        prior_scale=float(bayes["prior_scale"]),
    )
    high = antecedent.analyze(
        data,
        graph=graph,
        query=_intervention(1.0),
        inference=inference,
        refute=False,
        bootstrap=0,
        seed=int(bayes["seed"]),
    )
    low = antecedent.analyze(
        data,
        graph=graph,
        query=_intervention(0.0),
        inference=inference,
        refute=False,
        bootstrap=0,
        seed=int(bayes["seed"]),
    )
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        graph=graph,
        query=_intervention(1.0),
        inference=inference,
        refute=False,
        bootstrap=0,
        seed=int(bayes["seed"]),
        latency="interactive",
    )
    click = prepared.estimate(data, seed=int(bayes["seed"]))
    assert high.evidence_status == click.evidence_status == "licensed"
    assert any(str(d).startswith(bayes["diagnostic"]) for d in high.diagnostics)
    expected = bayes["cpdag_ate_contrast" if class_name == "cpdag" else "pag_ate_contrast"]
    contrast = float(_values(high)[0]) - float(_values(low)[0])
    assert contrast == pytest.approx(expected, abs=bayes["contrast_tolerance"])
    assert float(_values(click)[0]) == pytest.approx(float(_values(high)[0]), abs=1e-12)
