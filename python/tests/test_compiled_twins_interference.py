"""The direct interference functions warn and agree with the compiled analyze route."""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent import interference
from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.execution import Execution
from antecedent.outcomes import require_success


def _bernoulli_neighbor_count():
    # A four-unit ring under Bernoulli(1/2): treated units with no treated neighbour
    # against control units with two.
    data = {"y": np.array([1.0, 3.0, 2.0, 4.0])}
    assignment = [False, True, False, True]
    network = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 0), (0, 3)]
    query = interference.InterferenceQuery(
        interference.BernoulliAssignment(0.5),
        interference.NeighborCount(),
        interference.ExposureContrast(
            "y", interference.ExposureLevel(1.0, 0.0), interference.ExposureLevel(0.0, 2.0)
        ),
        network=network,
        realized_assignment=assignment,
    )
    return data, assignment, network, query


def test_estimate_warns_and_matches_the_compiled_route() -> None:
    data, assignment, network, query = _bernoulli_neighbor_count()
    with pytest.warns(AntecedentDeprecationWarning, match="antecedent.interference.estimate"):
        direct = interference.estimate(
            data, assignment=assignment, edges=network, query=query, execution=Execution(seed=9)
        )
    compiled = require_success(ant.analyze(data, graph=[], query=query)).interference
    # The exposure probabilities are exact here, so the seed (the caller's for the direct
    # entry, derived from the execution context for the route) does not enter.
    assert direct.from_probability_method == compiled.from_probability_method == "exact"
    assert direct.to_probability_method == compiled.to_probability_method == "exact"
    assert direct.contrast.horvitz_thompson == compiled.contrast.horvitz_thompson
    assert direct.contrast.hajek == compiled.contrast.hajek
    assert direct.contrast.conservative_variance == compiled.contrast.conservative_variance


def _saturation():
    clusters = [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3
    assignment = [False] * 3 + [True, False, False] + [True, True, False] + [True] * 3
    edges = [
        (s, t)
        for start in range(0, 12, 3)
        for s in range(start, start + 3)
        for t in range(start, start + 3)
        if s != t
    ]
    fractions = [sum(assignment[s] for s, t in edges if t == row) / 2.0 for row in range(12)]
    y = np.array(
        [
            1.0 + 2.0 * own + 3.0 * f + 4.0 * own * f
            for own, f in zip(assignment, fractions, strict=True)
        ]
    )
    design = interference.SaturationDesign(
        clusters=clusters,
        low_probability=0.2,
        high_probability=0.8,
        high_clusters=2,
        realized_saturation=[0.2] * 6 + [0.8] * 6,
    )
    return {"y": y}, assignment, edges, design, interference.PartialInterference(clusters)


def test_saturation_effects_warn_and_match_the_compiled_route() -> None:
    data, assignment, edges, design, partial = _saturation()
    with pytest.warns(
        AntecedentDeprecationWarning, match="antecedent.interference.estimate_saturation_effects"
    ):
        direct = interference.estimate_saturation_effects(
            data,
            assignment=assignment,
            edges=edges,
            design=design,
            partial_interference=partial,
            exposure=interference.NeighborFraction(),
            outcome="y",
            reference_neighbor_exposure=0.5,
            low_neighbor_exposure=0.0,
            high_neighbor_exposure=1.0,
        )
    for name, (from_level, to_level) in {
        "direct": ((0.0, 0.5), (1.0, 0.5)),
        "spillover": ((0.0, 0.0), (0.0, 1.0)),
        "total": ((0.0, 0.0), (1.0, 1.0)),
    }.items():
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
        compiled = require_success(ant.analyze(data, graph=[], query=query)).interference
        utility = getattr(direct, name)
        assert compiled.contrast.hajek == utility.hajek, name
        assert compiled.contrast.horvitz_thompson == utility.horvitz_thompson, name
        assert compiled.contrast.conservative_variance == utility.conservative_variance, name


def _observed():
    clusters = [0, 0, 1, 1]
    edges = [(0, 1), (1, 0), (2, 3), (3, 2)]
    assignment = [False, False, True, True]
    data = {"y": np.array([0.0, 0.0, 6.0, 6.0])}
    levels = (interference.ExposureLevel(0.0, 0.0), interference.ExposureLevel(1.0, 1.0))
    return data, clusters, edges, assignment, levels


def test_observational_network_exposure_warns_and_matches_the_compiled_route() -> None:
    data, clusters, edges, assignment, (from_level, to_level) = _observed()
    partial = interference.PartialInterference(clusters)
    with pytest.warns(
        AntecedentDeprecationWarning,
        match="antecedent.interference.estimate_observational_network_exposure",
    ):
        direct = interference.estimate_observational_network_exposure(
            data,
            outcome="y",
            assignment=assignment,
            edges=edges,
            clusters=clusters,
            partial_interference=partial,
            exposure=interference.NeighborCount(),
            from_level=from_level,
            to_level=to_level,
            propensity_from=[0.5] * 4,
            propensity_to=[0.5] * 4,
            propensity_provenance="externally_estimated",
        )
    query = interference.InterferenceQuery(
        interference.ObservedExposureDesign(
            clusters=clusters,
            propensity_from=[0.5] * 4,
            propensity_to=[0.5] * 4,
            propensity_provenance="externally_estimated",
            assume_network_exchangeability=True,
        ),
        interference.NeighborCount(),
        interference.ExposureContrast("y", from_level, to_level),
        network=edges,
        realized_assignment=assignment,
        partial_interference=partial,
    )
    compiled = require_success(ant.analyze(data, graph=[], query=query))
    assert compiled.estimate.ate == direct.horvitz_thompson
    assert compiled.interference.contrast.hajek == direct.hajek
    assert compiled.interference.contrast.conservative_variance == direct.cluster_robust_variance
