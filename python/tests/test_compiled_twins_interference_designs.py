"""Direct interference estimation warns and agrees with analyze for every design it serves.

Complete randomization and Bernoulli assignment run the same design-based estimator through the
compiled route under every built-in exposure mapping. Eight units keep the exposure probabilities
exact, so the Monte Carlo seed (the caller's for the direct entry, derived from the execution
context for the route) does not enter and the numbers agree exactly. Cluster randomization keeps
the direct path undeprecated.
"""

from __future__ import annotations

import warnings

import antecedent as ant
import numpy as np
import pytest
from antecedent import interference
from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.execution import Execution
from antecedent.outcomes import require_success

UNITS = 8
#: Treated units 0, 1, 2 and 5 of the ring.
ASSIGNMENT = [True, True, True, False, False, True, False, False]


def _ring() -> list[tuple[int, int, float]]:
    return [
        (source, unit, 1.0 + 0.5 * ((source + 2 * unit) % 3))
        for unit in range(UNITS)
        for source in ((unit - 1) % UNITS, (unit + 1) % UNITS)
    ]


def _outcomes() -> dict[str, np.ndarray]:
    y = [
        1.0
        + 0.25 * unit
        + 2.0 * ASSIGNMENT[unit]
        + 1.5 * (ASSIGNMENT[(unit - 1) % UNITS] + ASSIGNMENT[(unit + 1) % UNITS])
        for unit in range(UNITS)
    ]
    return {"y": np.array(y)}


CASES = {
    "complete_own": (
        interference.CompleteRandomization(4),
        interference.OwnTreatment(),
        (0.0, 0.0),
        (1.0, 0.0),
    ),
    "complete_count": (
        interference.CompleteRandomization(4),
        interference.NeighborCount(),
        (0.0, 1.0),
        (1.0, 1.0),
    ),
    "complete_fraction": (
        interference.CompleteRandomization(4),
        interference.NeighborFraction(),
        (0.0, 0.5),
        (1.0, 0.5),
    ),
    "complete_weighted": (
        interference.CompleteRandomization(4),
        interference.WeightedNeighborExposure(),
        (1.0, 0.0),
        (1.0, 1.0),
    ),
    "bernoulli_own": (
        interference.BernoulliAssignment(0.5),
        interference.OwnTreatment(),
        (0.0, 0.0),
        (1.0, 0.0),
    ),
    "bernoulli_fraction": (
        interference.BernoulliAssignment(0.5),
        interference.NeighborFraction(),
        (0.0, 0.5),
        (1.0, 0.5),
    ),
    "bernoulli_weighted": (
        interference.BernoulliAssignment(0.5),
        interference.WeightedNeighborExposure(),
        (1.0, 0.0),
        (1.0, 1.0),
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_estimate_warns_and_matches_the_compiled_route(case: str) -> None:
    design, exposure, from_level, to_level = CASES[case]
    query = interference.InterferenceQuery(
        design,
        exposure,
        interference.ExposureContrast(
            "y", interference.ExposureLevel(*from_level), interference.ExposureLevel(*to_level)
        ),
        network=_ring(),
        realized_assignment=ASSIGNMENT,
    )
    data = _outcomes()
    with pytest.warns(AntecedentDeprecationWarning, match="antecedent.interference.estimate"):
        direct = interference.estimate(
            data, assignment=ASSIGNMENT, edges=_ring(), query=query, execution=Execution(seed=9)
        )
    compiled = require_success(ant.analyze(data, graph=[], query=query)).interference
    assert direct.from_probability_method == compiled.from_probability_method == "exact"
    assert direct.to_probability_method == compiled.to_probability_method == "exact"
    assert direct.contrast.horvitz_thompson == compiled.contrast.horvitz_thompson
    assert direct.contrast.hajek == compiled.contrast.hajek
    assert direct.contrast.conservative_variance == compiled.contrast.conservative_variance
    assert direct.minimum_exposure_probability == compiled.minimum_exposure_probability


def test_cluster_randomization_keeps_the_direct_path_without_a_warning() -> None:
    clusters = [unit // 2 for unit in range(UNITS)]
    query = interference.InterferenceQuery(
        interference.ClusterRandomization(clusters, treated_clusters=2),
        interference.NeighborFraction(),
        interference.ExposureContrast(
            "y", interference.ExposureLevel(0.0, 0.0), interference.ExposureLevel(1.0, 1.0)
        ),
    )
    assignment = [True, True, False, False, True, True, False, False]
    edges = [(unit, unit ^ 1) for unit in range(UNITS)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", AntecedentDeprecationWarning)
        result = interference.estimate(
            {"y": np.arange(UNITS, dtype=float)}, assignment=assignment, edges=edges, query=query
        )
    assert result.from_probability_method == result.to_probability_method == "exact"
