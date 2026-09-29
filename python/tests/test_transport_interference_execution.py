from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent import interference
from antecedent.transport import advanced as transport


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


def test_graphical_transport_without_a_catalog_identifies_on_the_target() -> None:
    graph = antecedent.graph.Admg.from_edges(["a", "y"], [("a", "y")])
    query = transport.TransportQuery(
        antecedent.ResponseCurve("a", "y", grid=[0.0, 1.0]),
        transport.SelectionDiagram("trial", "target", []),
    )

    result = transport.identify(graph=graph, query=query)

    assert result.transportable
    assert result.outcome == "identified"
    assert isinstance(result.formula, transport.RecursiveFactorizationFormula)
    assert all(f.population == "target" and not f.interventions for f in result.formula.factors)


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
    graph = antecedent.graph.Admg.from_edges(
        ["a", "z", "y"],
        [("a", "y"), ("z", "y")],
        bidirected=[("z", "y")],
    )
    query = transport.TransportQuery(
        antecedent.ResponseCurve("a", "y", grid=[0.0, 1.0]),
        transport.SelectionDiagram("trial", "target", ["z"]),
        source_experiments=["a"],
    )
    identification = transport.identify(graph=graph, query=query)
    assert not identification.transportable
    assert identification.outcome == "not_certified"
    assert isinstance(identification.certificate, transport.NonTransportableCertificate)

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


def test_cluster_randomization_partial_interference_reports_support_and_bound() -> None:
    clusters = [10, 10, 20, 20]
    query = interference.InterferenceQuery(
        interference.ClusterRandomization(clusters, treated_clusters=1),
        interference.NeighborFraction(),
        interference.ExposureContrast(
            "y",
            interference.ExposureLevel(0.0, 0.0),
            interference.ExposureLevel(1.0, 1.0),
        ),
        partial_interference=interference.PartialInterference(clusters),
    )
    result = interference.estimate(
        {"y": np.array([1.0, 2.0, 3.0, 4.0])},
        assignment=[False, False, True, True],
        edges=[(0, 1), (1, 0), (2, 3), (3, 2)],
        query=query,
    )

    assert result.contrast.hajek == pytest.approx(2.0)
    assert result.from_probability_method == result.to_probability_method == "exact"
    assert result.support.from_exposed_units == result.support.to_exposed_units == 2
    assert result.support.from_observed_clusters == result.support.to_observed_clusters == 1
    assert result.support.minimum_exposure_probability > 0.0
    assert result.support.partial_interference_checked
    assert "covariance-free Young variance bound" in result.uncertainty_semantics
    assert "not a cluster-robust interval" in result.uncertainty_semantics


def test_partial_interference_rejects_cross_cluster_edges_and_partition_mismatch() -> None:
    clusters = [0, 0, 1, 1]
    query = interference.InterferenceQuery(
        interference.ClusterRandomization(clusters, 1),
        interference.NeighborFraction(),
        interference.ExposureContrast(
            "y", interference.ExposureLevel(0.0), interference.ExposureLevel(1.0)
        ),
        partial_interference=interference.PartialInterference(clusters),
    )
    args = {"y": np.array([0.0, 1.0, 2.0, 3.0])}
    assignment = [False, False, True, True]
    with pytest.raises(ValueError, match="crosses cluster boundary"):
        interference.estimate(args, assignment=assignment, edges=[(0, 2)], query=query)

    mismatched = interference.InterferenceQuery(
        interference.ClusterRandomization(clusters, 1),
        interference.NeighborFraction(),
        query.functional,
        partial_interference=interference.PartialInterference([0, 1, 0, 1]),
    )
    with pytest.raises(ValueError, match="partition"):
        interference.estimate(args, assignment=assignment, edges=[], query=mismatched)


def test_partial_interference_requires_explicit_cluster_assignment() -> None:
    query = interference.InterferenceQuery(
        interference.BernoulliAssignment(0.5),
        interference.OwnTreatment(),
        interference.ExposureContrast(
            "y", interference.ExposureLevel(0.0), interference.ExposureLevel(1.0)
        ),
        partial_interference=interference.PartialInterference([0, 0]),
    )
    with pytest.raises(ValueError, match="requires ClusterRandomization"):
        interference.estimate(
            {"y": np.array([0.0, 1.0])}, assignment=[False, True], edges=[], query=query
        )
