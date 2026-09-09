"""1.4 EconML adjustment-set handoff."""

from __future__ import annotations

import numpy as np
import pytest

antecedent = pytest.importorskip("antecedent")


def _backdoor_data(n: int = 200) -> dict[str, np.ndarray]:
    z = np.linspace(0.0, 1.0, n, dtype=np.float64)
    t = (z > 0.5).astype(np.float64)
    y = 1.0 + 2.0 * t + 3.0 * z
    return {"t": t, "y": y, "z": z}


def _backdoor_graph() -> antecedent.Dag:
    return antecedent.Dag.from_edges(["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")])


def test_backdoor_ate_emits_adjustment_set() -> None:
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    identified = antecedent.identify(graph=_backdoor_graph(), query=query)
    spec = antecedent.handoff.econml(identified)
    assert spec.treatment == "t"
    assert spec.outcome == "y"
    assert spec.confounders == ("z",)
    assert spec.identifier == "backdoor.adjustment"
    assert "Identified" in spec.status
    cols = spec.columns(_backdoor_data())
    assert cols["T"].shape == cols["Y"].shape
    assert cols["W"] is not None and len(cols["W"]) == 1


def test_analyze_result_handoff_matches_identify() -> None:
    data = _backdoor_data()
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    result = antecedent.analyze(
        data, graph=_backdoor_graph(), query=query, refute=False, bootstrap=0, seed=1
    )
    spec = antecedent.handoff.econml(result, treatment="t", outcome="y")
    assert result.plan.structure_source == "explicit"
    assert spec.confounders == ("z",)
    assert spec.identifier == "backdoor.adjustment"


def test_partial_id_refuses() -> None:
    pin = __import__("json").loads(
        __import__("pathlib")
        .Path(__file__)
        .resolve()
        .parents[2]
        .joinpath("conformance", "estimate", "cpdag_ate_envelope", "expected.json")
        .read_text()
    )
    graph = antecedent.Cpdag.from_directed_undirected(
        pin["columns"],
        [tuple(edge) for edge in pin["graph"]["directed_edges"]],
        [tuple(edge) for edge in pin["graph"]["undirected_edges"]],
    )
    values: dict[str, list[float]] = {name: [] for name in pin["columns"]}
    for cell in pin["contingency_table"]:
        count = int(cell["count"])
        for name in pin["columns"]:
            values[name].extend([float(cell[name])] * count)
    data = {name: np.asarray(col, dtype=np.float64) for name, col in values.items()}
    result = antecedent.analyze(
        data,
        graph=graph,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        refute=False,
        bootstrap=0,
        seed=1,
    )
    assert result.plan.structure_source == "explicit"
    assert result.identification.adjustment_set == []
    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="point identification"):
        antecedent.handoff.econml(result, treatment="t", outcome="y")


def test_frontdoor_refuses() -> None:
    graph = antecedent.Dag.from_edges(
        ["u", "t", "m", "y"], [("u", "t"), ("u", "y"), ("t", "m"), ("m", "y")]
    )
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    identified = antecedent.identify(graph=graph, query=query, identifier="frontdoor")
    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="frontdoor"):
        antecedent.handoff.econml(identified)


def test_frequentist_graph_posterior_refuses() -> None:
    from known_truth import FREQ, STATIC, static_data, static_posterior

    result = antecedent.analyze(
        static_data(int(STATIC["n"])),
        discovery=static_posterior(),
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=FREQ,
        refute=False,
        bootstrap=0,
        seed=1,
    )
    assert result.plan.structure_source == "graph_posterior"
    assert result.identification.status == "GraphDependent"
    assert result.identification.adjustment_set == []
    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="graph-posterior"):
        antecedent.handoff.econml(result, treatment="t", outcome="y")


def test_graph_posterior_refuses() -> None:
    data = _backdoor_data()
    result = antecedent.analyze(
        data,
        discovery=antecedent.discovery.ExactDagPosterior(),
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Bayesian(n_draws=32, prior_scale=100.0, backend="conjugate"),
        refute=False,
        bootstrap=0,
        seed=7,
    )
    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="graph-posterior"):
        antecedent.handoff.econml(result, treatment="t", outcome="y")


def test_temporal_dag_pulse_emits_adjustment_set() -> None:
    graph = antecedent.TemporalDag.from_lagged_edges(
        ["t", "y", "z"],
        [("z", 0, "t", 0), ("z", 1, "y", 0), ("t", 1, "y", 0)],
    )
    query = antecedent.PulseEffect("t", "y", treatment_lag=1)
    identified = antecedent.identify(graph=graph, query=query)
    spec = antecedent.handoff.econml(identified)
    assert spec.treatment == "t"
    assert spec.outcome == "y"
    assert spec.confounders == ("z",)
    assert spec.identifier == "temporal.backdoor.unfolded"
    assert "Identified" in spec.status


def test_fully_oriented_cpdag_stays_cpdag_and_emits_w() -> None:
    graph = antecedent.Cpdag.from_directed_undirected(
        ["z", "t", "y"],
        [("z", "t"), ("z", "y"), ("t", "y")],
        [],
    )
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    identified = antecedent.identify(graph=graph, query=query)
    assert isinstance(identified.graph, antecedent.Cpdag)
    spec = antecedent.handoff.econml(identified)
    assert spec.confounders == ("z",)
    assert spec.identifier == "generalized.adjustment"
    assert "Identified" in spec.status
    assert "Partial" not in spec.status


def test_identify_result_needs_names() -> None:
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    identified = antecedent.identify(graph=_backdoor_graph(), query=query)
    legacy = identified.to_identify_result()
    with pytest.raises(antecedent.errors.CausalValueError, match="treatment"):
        antecedent.handoff.econml(legacy)
    spec = antecedent.handoff.econml(legacy, treatment="t", outcome="y")
    assert spec.confounders == ("z",)
