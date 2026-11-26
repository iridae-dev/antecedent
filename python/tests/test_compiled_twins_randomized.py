"""The direct randomized-design utilities warn and agree with their compiled replacements."""

from __future__ import annotations

import warnings

import antecedent as ant
import numpy as np
import pytest
from antecedent import interference
from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.outcomes import require_success


def _bernoulli_data(n: int):
    assigned = [(i * 7 + i // 5) % 2 == 0 for i in range(n)]
    x1 = np.sin(0.17 * np.arange(n)) + 0.01 * np.arange(n)
    x2 = np.cos(0.11 * np.arange(n))
    outcome = (
        2.0
        + 1.3 * x1
        - 0.7 * x2
        + 0.4 * np.sin(0.29 * np.arange(n))
        + 1.4 * np.asarray(assigned, dtype=float)
    )
    design = ant.experiment.ExperimentDesign(
        interference.BernoulliAssignment(0.5),
        assigned,
        [f"unit-{i}" for i in range(n)],
        [f"row-{i}" for i in range(n)],
    )
    return {"outcome": outcome, "baseline_a": x1, "baseline_b": x2}, design, assigned


def test_estimate_cuped_effect_warns_and_matches_the_compiled_estimated_slope_route():
    data, design, assigned = _bernoulli_data(400)
    with pytest.warns(AntecedentDeprecationWarning, match="estimate_cuped_effect"):
        direct = ant.experiment.estimate_cuped_effect(
            data["outcome"], data["baseline_a"], assigned, 0.5
        )
    query = ant.experiment.RandomizedEffect(
        "outcome", design, estimated_cuped=ant.experiment.EstimatedCUPED("baseline_a")
    )
    fit = require_success(ant.analyze(data, query=query, refute="none")).randomized_effect
    assert fit.effect == direct.effect
    assert fit.adjustment_coefficient == direct.adjustment_coefficient
    assert fit.standard_error == direct.standard_error
    assert fit.interval_95 is not None
    assert fit.interval_95[0] < fit.effect < fit.interval_95[1]


def test_estimate_cuped_effect_still_returns_its_numbers_after_the_warning():
    data, _, assigned = _bernoulli_data(400)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AntecedentDeprecationWarning)
        first = ant.experiment.estimate_cuped_effect(
            data["outcome"], data["baseline_a"], assigned, 0.5
        )
        second = ant.experiment.estimate_cuped_effect(
            data["outcome"], data["baseline_a"], assigned, 0.5
        )
    assert first == second
    assert first.standard_error > 0


def test_estimated_cuped_refuses_what_the_direct_estimator_cannot_combine():
    _, design, _ = _bernoulli_data(400)
    with pytest.raises(ValueError, match="estimated-slope CUPED"):
        ant.experiment.RandomizedEffect(
            "outcome",
            design,
            estimated_cuped=ant.experiment.EstimatedCUPED("baseline_a"),
            cuped=ant.experiment.FixedCUPED("baseline_a", 1.0),
        )
    with pytest.raises(ValueError, match="distinct pre-assignment covariate"):
        ant.experiment.RandomizedEffect(
            "outcome", design, estimated_cuped=ant.experiment.EstimatedCUPED("outcome")
        )


def test_estimate_ancova_effect_warns_and_matches_the_compiled_route():
    data, design, _ = _bernoulli_data(400)
    covariates = ("baseline_a", "baseline_b")
    with pytest.warns(AntecedentDeprecationWarning, match="estimate_ancova_effect"):
        direct = ant.experiment.estimate_ancova_effect(
            data, outcome="outcome", design=design, covariates=covariates
        )
    query = ant.experiment.RandomizedEffect("outcome", design, ancova_covariates=covariates)
    fit = require_success(ant.analyze(data, query=query, refute="none")).randomized_effect
    assert fit.effect == direct.effect
    assert fit.standard_error == direct.standard_error
    assert fit.adjustment_coefficient is None


def test_exact_randomization_test_warns_and_matches_the_compiled_bernoulli_test():
    n = 12
    assigned = [i % 3 != 0 for i in range(n)]
    probabilities = [0.3 + 0.04 * i for i in range(n)]
    outcome = [1.0 + 0.7 * np.sin(i) + float(assigned[i]) for i in range(n)]
    with pytest.warns(AntecedentDeprecationWarning, match="exact_randomization_test"):
        direct = ant.experiment.exact_randomization_test(outcome, assigned, probabilities)
    design = ant.experiment.ExperimentDesign(
        interference.BernoulliAssignment(probabilities),
        assigned,
        [f"unit-{i}" for i in range(n)],
        [f"row-{i}" for i in range(n)],
    )
    query = ant.experiment.RandomizedEffect("outcome", design, exact_randomization_test=True)
    fit = require_success(
        ant.analyze({"outcome": np.asarray(outcome)}, query=query, refute="none")
    ).randomized_effect
    assert fit.adjustment_coefficient is None
    assert fit.effect == direct.observed_effect
    assert fit.randomization_p_value == direct.exact_two_sided_p_value
    assert fit.randomization_allocations == direct.assignments_enumerated


def test_switchback_estimate_warns_and_matches_the_compiled_route():
    sequences, periods = 40, 20
    n = sequences * periods
    assigned = [(i * 137 + (i // periods) * 31) % 11 < 5 for i in range(n)]
    outcome = [
        1.0
        + 0.5 * np.sin((i // periods) * 0.7)
        + 0.3 * np.cos((i % periods) * 0.4)
        + (1.4 if assigned[i] else 0.0)
        for i in range(n)
    ]
    design = ant.experiment.SwitchbackDesign(
        assigned,
        [f"s{i // periods}" for i in range(n)],
        [f"p{i % periods}" for i in range(n)],
        [0.5] * n,
        treatment_arms=("off", "on"),
    )
    query = ant.experiment.SwitchbackEffect("y", design)
    data = {"y": np.asarray(outcome)}
    with pytest.warns(AntecedentDeprecationWarning, match="SwitchbackEffect.estimate"):
        direct = require_success(query.estimate(data))
    fit = require_success(ant.analyze(data, query=query, refute="none")).randomized_effect
    assert fit.effect == direct.effect
    assert fit.standard_error == direct.standard_error


def test_stratified_estimate_warns_and_matches_the_compiled_route():
    assignment = [True, False, True, False, True, False, True, False]
    blocks = ["north"] * 4 + ["south"] * 4
    design = ant.experiment.ExperimentDesign(
        ant.experiment.StratifiedRandomization({"north": 2, "south": 2}),
        assignment,
        [f"a{i}" for i in range(8)],
        [f"y{i}" for i in range(8)],
        blocks=blocks,
    )
    data = {"outcome": [3.0, 0.0, 3.5, 0.5, 13.0, 10.0, 13.5, 10.5]}
    query = ant.experiment.RandomizedEffect("outcome", design)
    with pytest.warns(AntecedentDeprecationWarning, match="RandomizedEffect.estimate"):
        direct = require_success(query.estimate(data))
    fit = require_success(ant.analyze(data, query=query, refute="none")).randomized_effect
    assert fit.effect == direct.effect
