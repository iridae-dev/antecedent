from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent import interference


def _fixture():
    clusters = [0, 0, 1, 1]
    edges = [(0, 1), (1, 0), (2, 3), (3, 2)]
    assignment = [False, False, True, True]
    # Potential outcome is 2*own treatment + 4*treated-neighbor count.
    neighbor_count = [0, 0, 1, 1]
    outcome = np.array(
        [2 * own + 4 * neighbors for own, neighbors in zip(assignment, neighbor_count, strict=True)],
        dtype=float,
    )
    return {"y": outcome}, clusters, edges, assignment


def _estimate(**overrides):
    data, clusters, edges, assignment = _fixture()
    values = {
        "data": data,
        "outcome": "y",
        "assignment": assignment,
        "edges": edges,
        "clusters": clusters,
        "partial_interference": interference.PartialInterference(clusters),
        "exposure": interference.NeighborCount(),
        "from_level": interference.ExposureLevel(0.0, 0.0),
        "to_level": interference.ExposureLevel(1.0, 1.0),
        "propensity_from": [0.5] * 4,
        "propensity_to": [0.5] * 4,
        "propensity_provenance": "externally_estimated",
    }
    values.update(overrides)
    return interference.estimate_observational_network_exposure(**values)


def test_observational_network_exposure_recovers_known_contrast_and_cluster_variance():
    result = _estimate()
    assert result.horvitz_thompson == pytest.approx(6.0)
    assert result.hajek == pytest.approx(6.0)
    assert result.cluster_robust_variance == pytest.approx(36.0)
    assert result.from_exposed_units == result.to_exposed_units == 2
    assert result.from_exposed_clusters == result.to_exposed_clusters == 1
    assert result.clusters == 2
    assert result.minimum_exposure_probability == pytest.approx(0.5)
    assert result.propensity_provenance == "externally_estimated"
    assert result.support_status == "unlicensed_point_utility"
    assert "No unmeasured network confounding" in result.assumptions[0]
    assert "no interval" in result.uncertainty_semantics


def test_observational_network_exposure_refuses_cross_cluster_edges():
    with pytest.raises(ValueError, match="crosses cluster boundary"):
        _estimate(edges=[(0, 1), (1, 0), (2, 3), (3, 2), (0, 2)])


def test_observational_network_exposure_refuses_zero_propensity_and_missing_support():
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        _estimate(propensity_from=[0.0, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="observed support"):
        _estimate(assignment=[False] * 4)


def test_observational_network_exposure_requires_multiple_clusters_and_provenance():
    with pytest.raises(ValueError, match="at least two"):
        _estimate(clusters=[0] * 4, partial_interference=interference.PartialInterference([0] * 4))
    with pytest.raises(ValueError, match="known or externally_estimated"):
        _estimate(propensity_provenance="guessed")


def _query(*, edges=None, assume_network_exchangeability=True):
    _, clusters, fixture_edges, assignment = _fixture()
    design = interference.ObservedExposureDesign(
        clusters=clusters,
        propensity_from=[0.5] * 4,
        propensity_to=[0.5] * 4,
        propensity_provenance="externally_estimated",
        assume_network_exchangeability=assume_network_exchangeability,
    )
    return interference.InterferenceQuery(
        design,
        interference.NeighborCount(),
        interference.ExposureContrast(
            "y", interference.ExposureLevel(0.0, 0.0), interference.ExposureLevel(1.0, 1.0)
        ),
        network=fixture_edges if edges is None else edges,
        realized_assignment=assignment,
        partial_interference=interference.PartialInterference(clusters),
    )


def test_observational_network_analyze_matches_native_utility_and_reports_support():
    data, _, _, _ = _fixture()
    result = ant.analyze(data, graph=[], query=_query())
    utility = _estimate()
    assert result.estimate.ate == pytest.approx(utility.horvitz_thompson)
    assert result.interference.contrast.hajek == pytest.approx(utility.hajek)
    assert result.interference.contrast.conservative_variance == pytest.approx(
        utility.cluster_robust_variance
    )
    assert result.interference.from_probability_method == "supplied_externally_estimated"
    assert result.interference.support.from_exposed_units == utility.from_exposed_units
    assert result.interference.support.to_observed_clusters == utility.to_exposed_clusters
    assert result.interference.support.clusters == utility.clusters
    assert result.interference.support.maximum_exposure_probability == pytest.approx(
        utility.maximum_exposure_probability
    )
    assert result.evidence_status == "off_axis"
    assert np.isnan(result.estimate.se_analytic)
    assert "no interval" in result.interference.uncertainty_semantics
    assert "network_exchangeability" in " ".join(result.assumptions or [])


def test_observational_network_analyze_refuses_unsupported_design():
    data, _, edges, _ = _fixture()
    with pytest.raises(ValueError, match="assume_network_exchangeability"):
        _query(assume_network_exchangeability=False)
    with pytest.raises(ValueError, match="crosses cluster boundary"):
        ant.analyze(data, graph=[], query=_query(edges=[*edges, (0, 2)]))


def test_known_exposure_probabilities_publish_retained_cluster_interval():
    clusters = [cluster for cluster in range(60) for _ in range(2)]
    assignment = [cluster % 2 == 1 for cluster in clusters]
    edges = [(2 * cluster + 1 - unit, 2 * cluster + unit) for cluster in range(60) for unit in range(2)]
    outcomes = np.array([
        1.0 + (cluster % 7) / 10.0 + 2.0 * float(cluster % 2 == 1)
        for cluster in clusters
    ])
    design = interference.ObservedExposureDesign(
        clusters=clusters,
        propensity_from=[0.5] * 120,
        propensity_to=[0.5] * 120,
        propensity_provenance="known",
        assume_network_exchangeability=True,
    )
    query = interference.InterferenceQuery(
        design,
        interference.NeighborCount(),
        interference.ExposureContrast(
            "y", interference.ExposureLevel(0.0, 0.0), interference.ExposureLevel(1.0, 1.0)
        ),
        network=edges,
        realized_assignment=assignment,
        partial_interference=interference.PartialInterference(clusters),
    )
    result = ant.analyze({"y": outcomes}, graph=[], query=query)
    interval = result.interference.pointwise_interval
    assert interval is not None
    assert interval.lower < 2.0 < interval.upper
    assert interval.first_stage_arm_clusters == (30, 30)
    assert interval.degrees_of_freedom == 59.0
    assert result.interference.interval_unavailable_reason is None
    assert "Pointwise 95% cluster t interval" in result.interference.uncertainty_semantics
    assert result.evidence_status == "off_axis"


def test_observational_network_refuses_incoherent_exposure_probabilities():
    with pytest.raises(ValueError, match="summing above one"):
        _estimate(propensity_from=[0.7] * 4, propensity_to=[0.7] * 4)
