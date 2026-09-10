"""Identification must preserve the query passed to the new typed-graph API."""

import antecedent
import pytest


def test_sustained_identification_preserves_explicit_window() -> None:
    graph = antecedent.TemporalDag.from_lagged_edges(
        ["t", "y", "z"],
        [("z", 0, "t", 0), ("z", 2, "y", 0), ("t", 2, "y", 0)],
    )
    pulse = antecedent.identify(
        graph=graph, query=antecedent.PulseEffect("t", "y", treatment_lag=2)
    )
    sustained = antecedent.identify(
        graph=graph, query=antecedent.SustainedEffect("t", "y", window=(-2, -2))
    )
    assert sustained.adjustment_set == pulse.adjustment_set == ["z"]
    assert sustained.status == pulse.status


def test_incomplete_class_refuses_multi_step_identification() -> None:
    graph = antecedent.graph.TemporalCpdag.from_lagged_edges(["t", "y"], [("t", 1, "y", 0)])
    with pytest.raises(antecedent.errors.CausalError, match="single-step"):
        antecedent.identify(
            graph=graph, query=antecedent.SustainedEffect("t", "y", window=(-2, -1))
        )


@pytest.mark.parametrize("interventions", [[]])
def test_class_response_requires_an_intervention(interventions) -> None:
    graph = antecedent.Cpdag.from_directed_undirected(
        ["t", "y", "z"], [("t", "y"), ("z", "t"), ("z", "y")], []
    )
    with pytest.raises(antecedent.errors.CausalError, match="intervention"):
        antecedent.identify(
            graph=graph,
            query=antecedent.InterventionResponse("y", intervention=interventions),
        )


def test_native_certificate_keeps_temporal_coordinates_and_derivation():
    graph = antecedent.TemporalDag.from_lagged_edges(
        ["t", "y", "z"],
        [("z", 0, "t", 0), ("z", 1, "y", 0), ("t", 1, "y", 0)],
    )
    identified = antecedent.identify(
        graph=graph, query=antecedent.PulseEffect("t", "y", treatment_lag=1)
    )
    certificate = identified.certificate
    assert certificate["treatments"] == [{"name": "t", "variable": 0, "offset": -1}]
    assert certificate["outcome"] == {"name": "y", "variable": 1, "offset": 0}
    case = certificate["cases"][0]
    assert case["adjustment_coordinates"] == [[{"name": "z", "variable": 2, "offset": -1}]]
    assert case["indexer"]["history"] >= 1
    assert case["identification"]["derivation"]
    assert case["identification"]["required_assumptions"]
