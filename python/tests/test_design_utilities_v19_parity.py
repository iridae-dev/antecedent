"""Parity of the unlicensed design utilities.

``transport.estimate_trial_effect`` and ``interference.estimate`` are the
unlicensed utilities: augmented IPW, every assignment design and exposure
mapping, and ``seed`` as the exposure-probability Monte Carlo seed. They keep
that capability and those numbers. The expected values in
``fixtures/design_utilities_v1_9_0.json`` were recorded by running these exact
closed-form calls against a recorded build.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import antecedent
import pytest
from antecedent import interference
from antecedent.transport import advanced as transport

from _repo_text import read_text

EXPECTED = json.loads(
    read_text(Path(__file__).parent / "fixtures" / "design_utilities_v1_9_0.json")
)


def transport_inputs(n):
    treatment, outcome, trial, sel, prop, mu0, mu1 = [], [], [], [], [], [], []
    for i in range(n):
        x = math.sin(1.7 * i + 0.3)
        s = 1.0 / (1.0 + math.exp(-0.5 * x))
        in_trial = math.cos(2.3 * i) < (2 * s - 1)
        a = in_trial and (i % 3 != 0)
        e = 0.4 + 0.2 * (1.0 + math.sin(0.9 * i)) / 2.0
        y = (1.0 + (1.0 + x) * a + x + 0.3 * math.cos(3.1 * i)) if in_trial else 0.0
        treatment.append(bool(a))
        outcome.append(y)
        trial.append(bool(in_trial))
        sel.append(s)
        prop.append(e)
        mu0.append(1.0 + x)
        mu1.append(2.0 + 2.0 * x)
    return treatment, outcome, trial, sel, prop, mu0, mu1


def transport_case(aipw):
    graph = antecedent.graph.Admg.from_edges(["a", "y"], [("a", "y")])
    query = transport.TransportQuery(
        antecedent.ResponseCurve("a", "y", grid=[0.0, 1.0]),
        transport.SelectionDiagram("trial", "target", []),
        source_experiments=["a"],
    )
    ident = transport.identify(graph=graph, query=query)
    t, y, tr, s, e, m0, m1 = transport_inputs(40)
    kwargs = {"mu0": m0, "mu1": m1} if aipw else {}
    r = transport.estimate_trial_effect(ident, t, y, tr, s, e, **kwargs)

    def diag(d):
        return [
            d.probability_min,
            d.probability_max,
            d.effective_sample_size,
            d.extreme_weight_count,
        ]

    return {
        "rule": r.rule,
        "ipw": r.ipw,
        "aipw": r.aipw,
        "selection": diag(r.overlap.selection),
        "treatment": diag(r.overlap.treatment),
    }


def ring(units):
    return [((i + 1) % units, i, 1.0) for i in range(units)] + [
        ((i - 1) % units, i, 1.0) for i in range(units)
    ]


def interference_case(units, design, exposure, from_, to, seed, draws=10_000):
    assignment = [((i // 2) % 4) < 2 for i in range(units)]
    y = [1.0 + 2.0 * assignment[i] + 0.4 * math.cos(1.3 * i) + 0.1 * i for i in range(units)]
    query = interference.InterferenceQuery(
        design,
        exposure,
        interference.ExposureContrast(
            "y", interference.ExposureLevel(*from_), interference.ExposureLevel(*to)
        ),
        probability_draws=draws,
    )
    r = interference.estimate(
        {"y": y}, assignment=assignment, edges=ring(units), query=query, seed=seed
    )
    return {
        "horvitz_thompson": r.contrast.horvitz_thompson,
        "hajek": r.contrast.hajek,
        "conservative_variance": r.contrast.conservative_variance,
        "from_probability_method": r.from_probability_method,
        "to_probability_method": r.to_probability_method,
        "minimum_exposure_probability": r.minimum_exposure_probability,
    }


def _cases():
    cases = {
        "transport_ipw": transport_case(False),
        "transport_aipw": transport_case(True),
        "bernoulli_neighbor_count_exact": interference_case(
            6,
            interference.BernoulliAssignment(0.5),
            interference.NeighborCount(),
            (0.0, 1.0),
            (1.0, 1.0),
            1,
        ),
        "bernoulli_neighbor_count_monte_carlo": interference_case(
            40,
            interference.BernoulliAssignment(0.5),
            interference.NeighborCount(),
            (0.0, 1.0),
            (1.0, 1.0),
            7,
            4000,
        ),
        "complete_neighbor_fraction": interference_case(
            8,
            interference.CompleteRandomization(4),
            interference.NeighborFraction(),
            (0.0, 0.5),
            (1.0, 0.5),
            3,
        ),
        "cluster_own_treatment": interference_case(
            8,
            interference.ClusterRandomization([0, 0, 1, 1, 2, 2, 3, 3], 2),
            interference.OwnTreatment(),
            (0.0, 0.0),
            (1.0, 0.0),
            5,
        ),
        "bernoulli_weighted_exposure_monte_carlo": interference_case(
            30,
            interference.BernoulliAssignment([0.3 + 0.4 * (i % 2) for i in range(30)]),
            interference.WeightedNeighborExposure(),
            (0.0, 0.5),
            (1.0, 0.5),
            11,
            3000,
        ),
    }
    return cases


def _close(actual, expected):
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _close(a, e) for a, e in zip(actual, expected, strict=True)
        )
    if isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    return actual == expected


@pytest.mark.parametrize("name", sorted(EXPECTED["cases"]))
def test_design_utility_matches_v1_9_0(name):
    assert EXPECTED["version"] == "1.9.0"
    actual = _cases()[name]
    expected = EXPECTED["cases"][name]
    assert set(actual) == set(expected)
    for key, value in expected.items():
        assert _close(actual[key], value), (name, key, actual[key], value)
