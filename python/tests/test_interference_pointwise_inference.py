"""Public analyze and artifact evidence for independent-cluster pointwise inference."""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent import interference
from antecedent.outcomes import require_success


def _network(clusters: int = 80) -> tuple[list[int], list[tuple[int, int]]]:
    labels = [cluster for cluster in range(clusters) for _ in range(3)]
    edges = [
        (first + source, first + target)
        for first in range(0, 3 * clusters, 3)
        for source in range(3)
        for target in range(3)
        if source != target
    ]
    return labels, edges


def test_cluster_total_interval_is_retained_and_sealed() -> None:
    labels, edges = _network()
    assignment = [cluster >= 40 for cluster in labels]
    outcome = np.array(
        [
            5.0 + 0.2 * (cluster % 11) + (2.0 + 0.1 * (cluster % 7)) * assigned
            for cluster, assigned in zip(labels, assignment, strict=True)
        ]
    )
    query = interference.InterferenceQuery(
        interference.ClusterRandomization(labels, treated_clusters=40),
        interference.NeighborFraction(),
        interference.ExposureContrast(
            "y", interference.ExposureLevel(0.0, 0.0), interference.ExposureLevel(1.0, 1.0)
        ),
        network=edges,
        realized_assignment=assignment,
        partial_interference=interference.PartialInterference(labels),
    )
    result = require_success(ant.analyze({"y": outcome}, graph=[], query=query))
    interval = result.interference.pointwise_interval
    assert interval is not None
    assert interval.level == 0.95
    assert interval.first_stage_arm_clusters == (40, 40)
    assert interval.lower < result.interference.contrast.horvitz_thompson < interval.upper
    # The HT contrast of a completely randomized cluster design is the treated-minus-control
    # cluster-mean difference, computed here from the outcomes: 2.3775.
    treated = [value for value, flag in zip(outcome, assignment, strict=True) if flag]
    control = [value for value, flag in zip(outcome, assignment, strict=True) if not flag]
    assert result.interference.contrast.horvitz_thompson == pytest.approx(
        np.mean(treated) - np.mean(control), abs=1e-9
    )
    # The population effect (mean of 2 + 0.1 (c % 7) over the 80 clusters) is 2.2925.
    assert interval.lower < 2.2925 < interval.upper
    assert interval.standard_error > 0
    assert interval.degrees_of_freedom > 0
    assert "cluster" in result.interference.uncertainty_semantics.lower()
    body = require_success(ant.load(bytes(result.export()))).decoded_artifact.payload
    assert body["interference_inference"]["interval"]["lower"] == pytest.approx(interval.lower)
    assert body["interference_inference"]["interval"]["upper"] == pytest.approx(interval.upper)


@pytest.mark.parametrize(
    ("name", "from_level", "to_level", "truth"),
    [
        ("direct", (0.0, 0.5), (1.0, 0.5), 2.0),
        ("spillover", (0.0, 0.0), (0.0, 1.0), 3.0),
        ("total", (0.0, 0.0), (1.0, 1.0), 5.0),
    ],
)
@pytest.mark.parametrize("cluster_count", [48, 80])
def test_saturation_interval_and_support_round_trip(
    name: str,
    from_level: tuple[float, float],
    to_level: tuple[float, float],
    truth: float,
    cluster_count: int,
) -> None:
    labels, edges = _network(cluster_count)
    patterns = (
        [False, False, False],
        [True, False, False],
        [True, True, False],
        [True, True, True],
    )
    assignment = [value for cluster in range(cluster_count) for value in patterns[cluster % 4]]
    realized = [0.2 if cluster < cluster_count // 2 else 0.8 for cluster in labels]
    outcome = []
    for row, own in enumerate(assignment):
        first = row // 3 * 3
        neighbors = sum(assignment[j] for j in range(first, first + 3) if j != row) / 2
        outcome.append(5.0 + 0.1 * (labels[row] % 9) + 2.0 * own + 3.0 * neighbors)
    query = interference.InterferenceQuery(
        interference.SaturationDesign(labels, 0.2, 0.8, cluster_count // 2, realized),
        interference.NeighborFraction(),
        interference.ExposureContrast(
            "y", interference.ExposureLevel(*from_level), interference.ExposureLevel(*to_level)
        ),
        network=edges,
        realized_assignment=assignment,
        partial_interference=interference.PartialInterference(labels),
    )
    result = require_success(ant.analyze({"y": np.asarray(outcome)}, graph=[], query=query))
    interval = result.interference.pointwise_interval
    assert interval is not None
    assert interval.first_stage_arm_clusters == (cluster_count // 2, cluster_count // 2)
    assert interval.lower < truth < interval.upper, name
    # The saturation HT estimate is design-random (observed errors <= 0.58 at these sizes).
    assert abs(result.interference.contrast.horvitz_thompson - truth) < 0.75, name
    assert interval.lower < result.interference.contrast.horvitz_thompson < interval.upper
    assert result.interference.support.from_observed_clusters >= 8
    assert result.interference.support.to_observed_clusters >= 8
    body = require_success(ant.load(bytes(result.export()))).decoded_artifact.payload
    assert body["interference_inference"]["method"] == "saturation_cluster_neyman_welch"
    assert body["interference_inference"]["from_exposed_clusters"] >= 8
    assert body["interference_inference"]["interval"]["lower"] == pytest.approx(interval.lower)
    tampered = bytearray(bytes(result.export()))
    tampered[len(tampered) // 2] ^= 0x40
    with pytest.raises(ant.CausalError, match="checksum mismatch"):
        require_success(ant.load(bytes(tampered)))


@pytest.mark.parametrize(
    ("from_level", "to_level"),
    [((0.0, 0.5), (1.0, 0.5)), ((0.0, 0.0), (0.0, 1.0)), ((0.0, 0.0), (1.0, 1.0))],
)
@pytest.mark.parametrize("cluster_count", [4, 40])
def test_saturation_thin_arms_remain_explicitly_point_only(
    from_level: tuple[float, float],
    to_level: tuple[float, float],
    cluster_count: int,
) -> None:
    labels, edges = _network(cluster_count)
    patterns = (
        [False, False, False],
        [True, False, False],
        [True, True, False],
        [True, True, True],
    )
    assignment = [value for cluster in range(cluster_count) for value in patterns[cluster % 4]]
    realized = [0.2 if cluster < cluster_count // 2 else 0.8 for cluster in labels]
    query = interference.InterferenceQuery(
        interference.SaturationDesign(labels, 0.2, 0.8, cluster_count // 2, realized),
        interference.NeighborFraction(),
        interference.ExposureContrast(
            "y", interference.ExposureLevel(*from_level), interference.ExposureLevel(*to_level)
        ),
        network=edges,
        realized_assignment=assignment,
        partial_interference=interference.PartialInterference(labels),
    )
    result = require_success(
        ant.analyze({"y": np.arange(3 * cluster_count, dtype=float)}, graph=[], query=query)
    )
    assert result.interference.pointwise_interval is None
    assert "24 independent clusters" in result.interference.interval_unavailable_reason
    body = require_success(ant.load(bytes(result.export()))).decoded_artifact.payload
    assert body["interference_inference"]["interval"] is None
