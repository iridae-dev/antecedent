"""``RandomizedEffect.estimate`` warns on the stratified design, and only there.

The stratified branch of the direct estimator runs the blocked Neyman kernel; ``analyze`` with
the same query runs the compiled ``RandomizedEffect`` route, which returns the same effect and
variance bound. The Bernoulli, complete and cluster branches keep their own direct path and do
not carry this warning.
"""

from __future__ import annotations

import warnings

import antecedent as ant
from antecedent import interference
from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.outcomes import require_success

_WARNING = "RandomizedEffect.estimate (stratified design)"


def _stratified():
    assignment = [True, False, True, False, True, False, True, False]
    design = ant.experiment.ExperimentDesign(
        ant.experiment.StratifiedRandomization({"north": 2, "south": 2}),
        assignment,
        [f"a{i}" for i in range(8)],
        [f"y{i}" for i in range(8)],
        blocks=["north"] * 4 + ["south"] * 4,
    )
    data = {"outcome": [3.0, 0.0, 3.25, 0.5, 13.0, 10.0, 13.5, 10.75]}
    return data, ant.experiment.RandomizedEffect("outcome", design)


def test_stratified_estimate_warns_once_and_keeps_its_numbers():
    data, query = _stratified()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        direct = require_success(query.estimate(data))
    named = [w for w in caught if issubclass(w.category, AntecedentDeprecationWarning)]
    assert len(named) == 1
    assert _WARNING in str(named[0].message)
    assert direct.assignment_design == "stratified"
    assert direct.uncertainty == "stratified_neyman_variance_upper_bound_no_interval"
    # Block means: north 3.125 - 0.25, south 13.25 - 10.375, equal block weights.
    assert direct.effect == 0.5 * (3.125 - 0.25) + 0.5 * (13.25 - 10.375)


def test_stratified_replacement_reproduces_the_effect_and_variance_bound():
    data, query = _stratified()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AntecedentDeprecationWarning)
        direct = require_success(query.estimate(data))
    fit = require_success(ant.analyze(data, query=query, refute="none")).randomized_effect
    assert fit.effect == direct.effect
    assert fit.variance_upper_bound == direct.variance_upper_bound
    assert fit.assignment_design == "stratified"


def test_bernoulli_estimate_does_not_carry_the_stratified_warning():
    assignment = [i % 2 == 0 for i in range(40)]
    design = ant.experiment.ExperimentDesign(
        interference.BernoulliAssignment(0.5),
        assignment,
        [f"unit-{i}" for i in range(40)],
        [f"row-{i}" for i in range(40)],
    )
    data = {"outcome": [1.0 + 2.0 * float(z) + 0.1 * (i % 3) for i, z in enumerate(assignment)]}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        require_success(ant.experiment.RandomizedEffect("outcome", design).estimate(data))
    assert not any(_WARNING in str(w.message) for w in caught)
