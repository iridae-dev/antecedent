"""Identification must preserve the query passed to the new typed-graph API."""

import antecedent
import pytest

from _repo_text import REPO_ROOT, load_json

_CLASS_PIN = load_json(
    REPO_ROOT / "conformance" / "response" / "class_aware_envelope" / "expected.json"
)


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
    assert sustained.adjustment_set == pulse.adjustment_set == tuple(["z"])
    assert sustained.status == pulse.status


def test_incomplete_class_preserves_multi_step_identification() -> None:
    graph = antecedent.graph.TemporalCpdag.from_lagged_edges(["t", "y"], [("t", 1, "y", 0)])
    identified = antecedent.identify(
        graph=graph, query=antecedent.SustainedEffect("t", "y", window=(-2, -1))
    )
    assert len(identified.completion_keys) == 1
    assert [node["offset"] for node in identified.certificate["treatments"]] == [-2, -1]
    assert identified.certificate["cases"][0]["identification"]["estimands"]


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
    certificate = identified.to_dict()["certificate"]
    assert certificate["treatments"] == [{"name": "t", "variable": 0, "offset": -1}]
    assert certificate["outcome"] == {"name": "y", "variable": 1, "offset": 0}
    case = certificate["cases"][0]
    assert case["adjustment_coordinates"] == [[{"name": "z", "variable": 2, "offset": -1}]]
    assert case["indexer"]["history"] >= 1
    assert case["identification"]["derivation"]
    assert case["identification"]["required_assumptions"]
    assert identified.statement.startswith("PulseEffect of y from t is identified")
    assert identified.assumption_statements
    assert identified.derivation_statements
    assert identified.to_dict()["assumption_statements"] == list(identified.assumption_statements)


def _class_graph(class_name: str):
    spec = _CLASS_PIN[class_name]["graph"]
    if class_name == "cpdag":
        return antecedent.Cpdag.from_directed_undirected(
            _CLASS_PIN["columns"],
            [tuple(edge) for edge in spec["directed_edges"]],
            [tuple(edge) for edge in spec["undirected_edges"]],
        )
    return antecedent.Pag.from_marked_edges(
        _CLASS_PIN["columns"], [tuple(edge) for edge in spec["marked_edges"]]
    )


@pytest.mark.parametrize("class_name", ["cpdag", "pag"])
def test_class_identification_keeps_the_pinned_completion_mass_for_effect_and_response(
    class_name: str,
) -> None:
    pin = _CLASS_PIN[class_name]["identification"]
    graph = _class_graph(class_name)
    queries = [
        antecedent.AverageEffect("t", "y"),
        antecedent.InterventionResponse("y", intervention=antecedent.intervention.Set("t", 1.0)),
        antecedent.ResponseCurve("t", "y", grid=[float(value) for value in _CLASS_PIN["grid"]]),
    ]
    for query in queries:
        identified = antecedent.identify(graph=graph, query=query)
        assert identified.status == pin["status"], type(query).__name__
        assert identified.certificate["identified_weight"] == pin["identified_mass"]
        assert identified.certificate["unidentified_weight"] == pin["unidentified_mass"]


@pytest.mark.parametrize("class_name", ["cpdag", "pag"])
@pytest.mark.parametrize("query_kind", ["average", "response"])
def test_a_dag_only_identifier_is_refused_on_class_graphs_and_names_the_class_identifier(
    class_name: str, query_kind: str
) -> None:
    graph = _class_graph(class_name)
    query = (
        antecedent.AverageEffect("t", "y")
        if query_kind == "average"
        else antecedent.InterventionResponse(
            "y", intervention=antecedent.intervention.Set("t", 1.0)
        )
    )
    with pytest.raises((antecedent.errors.CausalError, ValueError), match="generalized.adjustment"):
        antecedent.identify(graph=graph, query=query, identifier="backdoor.adjustment")
    default = antecedent.identify(graph=graph, query=query)
    assert default.status == _CLASS_PIN[class_name]["identification"]["status"]
