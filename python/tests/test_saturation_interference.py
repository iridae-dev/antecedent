from __future__ import annotations

import numpy as np
import pytest
import antecedent as ant
from antecedent import interference


def _fixture():
    clusters = [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3
    assignment = [False] * 3 + [True, False, False] + [True, True, False] + [True] * 3
    edges = [
        (source, target)
        for start in range(0, 12, 3)
        for source in range(start, start + 3)
        for target in range(start, start + 3)
        if source != target
    ]
    neighbors = []
    for row in range(12):
        incoming = [source for source, target in edges if target == row]
        neighbors.append(sum(assignment[source] for source in incoming) / len(incoming))
    outcome = np.array(
        [1.0 + 2.0 * own + 3.0 * neighbor + 4.0 * own * neighbor
         for own, neighbor in zip(assignment, neighbors, strict=True)]
    )
    design = interference.SaturationDesign(
        clusters=clusters,
        low_probability=0.2,
        high_probability=0.8,
        high_clusters=2,
        realized_saturation=[0.2] * 6 + [0.8] * 6,
    )
    partial = interference.PartialInterference(clusters)
    return {"y": outcome}, assignment, edges, design, partial


def _estimate(*, exposure=None, edges=None):
    data, assignment, fixture_edges, design, partial = _fixture()
    return interference.estimate_saturation_effects(
        data,
        assignment=assignment,
        edges=fixture_edges if edges is None else edges,
        design=design,
        partial_interference=partial,
        exposure=exposure or interference.NeighborFraction(),
        outcome="y",
        reference_neighbor_exposure=0.5,
        low_neighbor_exposure=0.0,
        high_neighbor_exposure=1.0,
    )


def test_two_stage_saturation_recovers_direct_spillover_and_total_effects() -> None:
    result = _estimate()

    assert result.direct.hajek == pytest.approx(4.0)
    assert result.spillover.hajek == pytest.approx(3.0)
    assert result.total.hajek == pytest.approx(9.0)
    for estimate in (result.direct, result.spillover, result.total):
        assert estimate.conservative_variance > 0.0
        assert estimate.from_exposed_units > 0
        assert estimate.to_exposed_units > 0
        assert estimate.from_exposed_clusters == 1
        assert estimate.to_exposed_clusters == 1
        assert estimate.minimum_exposure_probability > 0.0
    assert "independently Bernoulli assigned" in result.assumptions[1]
    assert "no confidence interval" in result.uncertainty_semantics


@pytest.mark.parametrize(
    ("from_level", "to_level", "truth", "utility_field"),
    [
        ((0.0, 0.5), (1.0, 0.5), 4.0, "direct"),
        ((0.0, 0.0), (0.0, 1.0), 3.0, "spillover"),
        ((0.0, 0.0), (1.0, 1.0), 9.0, "total"),
    ],
)
def test_saturation_analyze_matches_native_utility_and_reports_point_only(
    from_level, to_level, truth, utility_field
) -> None:
    data, assignment, edges, design, partial = _fixture()
    query = interference.InterferenceQuery(
        design,
        interference.NeighborFraction(),
        interference.ExposureContrast(
            "y", interference.ExposureLevel(*from_level), interference.ExposureLevel(*to_level)
        ),
        network=edges,
        realized_assignment=assignment,
        partial_interference=partial,
    )
    result = ant.analyze(data, graph=[], query=query)
    utility = getattr(_estimate(), utility_field)
    assert result.interference.contrast.hajek == pytest.approx(truth)
    assert result.interference.contrast.hajek == pytest.approx(utility.hajek)
    assert result.interference.contrast.conservative_variance == pytest.approx(
        utility.conservative_variance
    )
    assert any("two-stage" in value.lower() or "saturation" in value.lower() for value in result.assumptions)
    assert result.estimate.se_analytic != result.estimate.se_analytic


def test_saturation_analyze_refuses_cross_cluster_edge() -> None:
    data, assignment, edges, design, partial = _fixture()
    query = interference.InterferenceQuery(
        design, interference.NeighborFraction(),
        interference.ExposureContrast(
            "y", interference.ExposureLevel(0.0, 0.5), interference.ExposureLevel(1.0, 0.5)
        ),
        network=[*edges, (0, 3)],
        realized_assignment=assignment,
        partial_interference=partial,
    )
    with pytest.raises(ValueError, match="crosses cluster boundary"):
        ant.analyze(data, graph=[], query=query)


def test_saturation_refuses_cross_cluster_interference_edges() -> None:
    _, _, edges, _, _ = _fixture()
    with pytest.raises(ValueError, match="crosses cluster boundary"):
        _estimate(edges=[*edges, (0, 3)])


def test_saturation_refuses_misaligned_cluster_labels() -> None:
    data, assignment, edges, design, _ = _fixture()
    wrong = interference.PartialInterference([0, 1, 0, 1] * 3)
    with pytest.raises(ValueError, match="must match saturation-design clusters"):
        interference.estimate_saturation_effects(
            data,
            assignment=assignment,
            edges=edges,
            design=design,
            partial_interference=wrong,
            exposure=interference.NeighborFraction(),
            outcome="y",
            reference_neighbor_exposure=0.5,
            low_neighbor_exposure=0.0,
            high_neighbor_exposure=1.0,
        )


def test_saturation_refuses_own_treatment_mapping_without_spillover_exposure() -> None:
    data, assignment, edges, design, partial = _fixture()
    with pytest.raises(TypeError, match="require NeighborCount"):
        interference.estimate_saturation_effects(
            data,
            assignment=assignment,
            edges=edges,
            design=design,
            partial_interference=partial,
            exposure=interference.OwnTreatment(),
            outcome="y",
            reference_neighbor_exposure=0.5,
            low_neighbor_exposure=0.0,
            high_neighbor_exposure=1.0,
        )
