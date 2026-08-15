from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent import interference, transport


def test_graphical_transport_returns_direct_formula_and_certificate() -> None:
    graph = antecedent.graph.Admg.from_edges(["a", "y"], [("a", "y")])
    query = transport.TransportQuery(
        antecedent.ResponseCurve("a", "y", grid=[0.0, 1.0]),
        transport.SelectionDiagram("trial", "target", []),
        source_experiments=["a"],
    )

    result = transport.identify(graph=graph, query=query)

    assert result.transportable
    assert isinstance(result.formula, transport.DirectFormula)
    assert result.formula.factor.population == "trial"
    assert result.formula.factor.interventions == ["a"]
    assert isinstance(result.certificate, transport.TransportCertificate)
    assert result.certificate.rule == "transport.sid.direct"


def test_graphical_transport_refuses_a_missing_source_experiment() -> None:
    graph = antecedent.graph.Admg.from_edges(["a", "y"], [("a", "y")])
    query = transport.TransportQuery(
        antecedent.ResponseCurve("a", "y", grid=[0.0, 1.0]),
        transport.SelectionDiagram("trial", "target", []),
    )

    result = transport.identify(graph=graph, query=query)

    assert not result.transportable
    assert isinstance(result.certificate, transport.NonTransportableCertificate)
    assert result.certificate.reason == "transport.source_experiment_missing"


def test_transport_refuses_embedded_response_semantics_it_cannot_preserve() -> None:
    graph = antecedent.graph.Admg.from_edges(["a", "y"], [("a", "y")])
    diagram = transport.SelectionDiagram("trial", "target", [])
    with pytest.raises(ValueError, match="target_population"):
        transport.identify(
            graph=graph,
            query=transport.TransportQuery(
                antecedent.ResponseCurve(
                    "a", "y", grid=[0.0, 1.0], target_population="embedded-target"
                ),
                diagram,
                source_experiments=["a"],
            ),
        )


def _certified_direct_identification() -> transport.TransportIdentification:
    graph = antecedent.graph.Admg.from_edges(["a", "y"], [("a", "y")])
    query = transport.TransportQuery(
        antecedent.ResponseCurve("a", "y", grid=[0.0, 1.0]),
        transport.SelectionDiagram("trial", "target", []),
        source_experiments=["a"],
    )
    return transport.identify(graph=graph, query=query)


def test_trial_transport_keeps_selection_and_treatment_overlap_separate() -> None:
    identification = _certified_direct_identification()
    assert identification.transportable

    result = transport.estimate_trial_effect(
        identification,
        [False, True, False, False],
        [1.0, 3.0, 0.0, 0.0],
        [True, True, False, False],
        [0.5] * 4,
        [0.5] * 4,
        mu0=[1.0] * 4,
        mu1=[3.0] * 4,
    )

    assert result.rule == identification.certificate.rule
    assert result.ipw == pytest.approx(2.0)
    assert result.aipw == pytest.approx(2.0)
    assert result.overlap.selection.probability_min == 0.5
    assert result.overlap.treatment.probability_min == 0.5


def test_trial_transport_refuses_uncertified_identification() -> None:
    graph = antecedent.graph.Admg.from_edges(["a", "y"], [("a", "y")])
    query = transport.TransportQuery(
        antecedent.ResponseCurve("a", "y", grid=[0.0, 1.0]),
        transport.SelectionDiagram("trial", "target", []),
        # No source_experiments: identify() cannot certify a formula, so this
        # NotCertified result must not be usable to obtain an estimate.
    )
    identification = transport.identify(graph=graph, query=query)
    assert not identification.transportable

    with pytest.raises(antecedent.errors.CausalEstimateError):
        transport.estimate_trial_effect(
            identification,
            [False, True, False, False],
            [1.0, 3.0, 0.0, 0.0],
            [True, True, False, False],
            [0.5] * 4,
            [0.5] * 4,
        )


def test_randomized_interference_matches_empty_network_difference() -> None:
    query = interference.InterferenceQuery(
        interference.CompleteRandomization(2),
        interference.NeighborFraction(),
        interference.ExposureContrast(
            "y",
            interference.ExposureLevel(0.0),
            interference.ExposureLevel(1.0),
        ),
    )

    result = interference.estimate(
        {"y": np.array([1.0, 3.0, 2.0, 4.0])},
        assignment=[False, True, False, True],
        edges=[],
        query=query,
        seed=9,
    )

    assert result.contrast.hajek == pytest.approx(2.0)
    assert result.from_probability_method == "exact"
    assert result.to_probability_method == "exact"


def test_randomized_interference_uses_directed_network_edges() -> None:
    query = interference.InterferenceQuery(
        interference.BernoulliAssignment(0.5),
        interference.NeighborCount(),
        interference.ExposureContrast(
            "y",
            interference.ExposureLevel(0.0, 1.0),
            interference.ExposureLevel(1.0, 0.0),
        ),
    )
    edges = [
        interference.NetworkEdge(0, 1),
        interference.NetworkEdge(1, 0),
    ]

    result = interference.estimate(
        {"y": np.array([1.0, 4.0])},
        assignment=[True, False],
        edges=edges,
        query=query,
    )

    assert result.contrast.hajek == pytest.approx(-3.0)
