"""The compiled interval entries reproduce the deprecated direct intervals.

Each deprecated direct interval (``CountFittedObservation.interval`` and
``calibrated_probability``, ``CohortDecisionSource.interval``,
``AdaptiveInferenceSession.measured_interval``, ``ChoiceInferenceSession.measured_interval`` and
``antecedent.transport.temporal_dependent_interval``) warns with
``AntecedentDeprecationWarning`` and still returns its measured interval; the compiled entry
(``compiled_interval`` and ``TemporalUnitPanel.compiled_dependent_interval``) runs the same
measured consumer through its program family, does not warn, and returns the identical numbers,
calibration slots and exported envelope. Each test also checks a value known without the
library: the count-fitted posterior mean ``31 / 42``, the cohort's mean of the true effect on a
design whose noise cancels in pairs, the recounted Beta posterior mean of the Thompson arm and of
the first choice share, and the temporal panel's unit value ``0.49``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from antecedent import adaptive_inference as ai
from antecedent import choice_inference as ci
from antecedent import cohort_decision as cd
from antecedent import observation_query as oq
from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.adaptive_inference import AdaptiveInferenceRefusal
from antecedent.execution import Execution
from antecedent.transport.advanced import InitialStateLaw, temporal_dependent_interval

from test_adaptive_inference import prepare, thompson_design
from test_temporal_interval_activation import panel


def assert_identical(direct, compiled) -> None:
    assert compiled.route == direct.route
    assert len(compiled.scalars) == len(direct.scalars)
    for left, right in zip(direct.scalars, compiled.scalars, strict=True):
        assert right.name == left.name
        assert right.point == left.point
        assert right.interval == left.interval
        assert right.level == left.level
        assert right.calibration == left.calibration
        assert right.record_id == left.record_id
    assert compiled.inspect() == direct.inspect()
    assert bytes(compiled.export()) == bytes(direct.export())


def compiled_quietly(call):
    with warnings.catch_warnings():
        warnings.simplefilter("error", AntecedentDeprecationWarning)
        return call()


# -- count-fitted ------------------------------------------------------------------------------


def fitted() -> oq.CountFittedObservation:
    return oq.count_fitted_observation(
        {"Z": ["0", "1"], "Y": ["0", "1"]},
        parents={"Y": ["Z"]},
        counts={"Z": [60, 40], "Y": [48, 12, 10, 30]},
        rows=100,
        target="Y",
        target_level="1",
        evidence={"Z": "1"},
        execution=Execution(seed=7),
    )


def test_the_direct_count_fitted_interval_warns_and_equals_the_compiled_interval():
    claim = fitted()
    for level in (0.95, 0.9):
        with pytest.warns(AntecedentDeprecationWarning, match="compiled_interval"):
            direct = claim.interval(level=level)
        with pytest.warns(AntecedentDeprecationWarning, match="compiled_interval"):
            calibrated = claim.calibrated_probability(level=level)
        compiled = compiled_quietly(lambda level=level: claim.compiled_interval(level=level))
        assert_identical(direct, compiled)
        assert_identical(calibrated, compiled)
        # (30 + 1) / (40 + 2) under the uniform Dirichlet prior.
        assert compiled.scalars[0].point == pytest.approx(31.0 / 42.0, abs=1e-12)


# -- cohort mean -------------------------------------------------------------------------------


def native_fit_in_pairs(pairs: int = 200):
    """Each design point twice with opposite noise: least squares recovers the truth exactly."""
    rng = np.random.default_rng(29)
    x = np.repeat(rng.uniform(-2.0, 2.0, pairs), 2)
    t = np.repeat((np.arange(pairs) % 2).astype(float), 2)
    noise = np.repeat(rng.uniform(-0.5, 0.5, pairs), 2) * np.tile([1.0, -1.0], pairs)
    y = 2.0 + x + t * (1.0 + 0.5 * x) + noise
    fit = cd.fit_native_cohort_effect(
        outcome=y,
        treatment=t,
        features={"x": x},
        outcome_name="y",
        treatment_name="t",
        outcome_units="mmHg",
        population="target",
    )
    return x, fit


def test_the_direct_cohort_interval_warns_and_equals_the_compiled_interval():
    x, fit = native_fit_in_pairs()
    cohort = [float(v) for v in x[::2][:80]]
    binding = cd.CohortBinding(
        snapshot_id="clinic",
        population="target",
        columns={"x": cohort},
        weights=[1.0 / len(cohort)] * len(cohort),
    )
    source = cd.decision_source_from_native_fit(fit, cohort=binding, outcome_units="mmHg")
    for level in (0.95, 0.9):
        with pytest.warns(AntecedentDeprecationWarning, match="compiled_interval"):
            direct = source.interval(level=level)
        compiled = compiled_quietly(lambda level=level: source.compiled_interval(level=level))
        assert_identical(direct, compiled)
    truth = sum(1.0 + 0.5 * v for v in cohort) / len(cohort)
    scalar = compiled.scalars[0]
    assert scalar.point == pytest.approx(truth, abs=1e-9)
    lower, upper = scalar.interval
    assert lower < truth < upper
    assert scalar.point - lower == pytest.approx(upper - scalar.point, abs=1e-12)


# -- adaptive ----------------------------------------------------------------------------------


def test_the_direct_adaptive_interval_warns_and_equals_the_compiled_interval():
    design = thompson_design()
    session = prepare(design)
    request = ai.BayesianPosterior(look=1, prior_alpha=1.0, prior_beta=1.0)
    with pytest.warns(AntecedentDeprecationWarning, match="compiled_interval"):
        direct = session.measured_interval(request)
    compiled = compiled_quietly(lambda: session.compiled_interval(request))
    assert_identical(direct, compiled)
    # The second arm's posterior mean, recounted from the retained history.
    successes = sum(
        1 for arm, y in zip(design.assigned_arm, design.outcomes, strict=True) if arm == 1 and y[0]
    )
    trials = sum(1 for arm in design.assigned_arm if arm == 1)
    expected = (1 + successes) / (2 + trials)
    assert compiled.scalars[0].point == pytest.approx(expected, abs=1e-12)
    # A paradigm the cell is not supported for refuses through both entries alike.
    weighting = ai.AdaptiveWeighting(
        look=1, plus_arm=1, minus_arm=0, scheme="variance_stabilizing", prior_mean=0.5
    )
    with pytest.warns(AntecedentDeprecationWarning), pytest.raises(AdaptiveInferenceRefusal) as a:
        session.measured_interval(weighting)
    with pytest.raises(AdaptiveInferenceRefusal) as b:
        session.compiled_interval(weighting)
    assert (a.value.reason_code, a.value.detail) == (b.value.reason_code, b.value.detail)


# -- choice ------------------------------------------------------------------------------------


def test_the_direct_choice_interval_warns_and_equals_the_compiled_interval():
    respondents = tuple(
        ci.Respondent(f"r{i}", [1, 0, 0] if i < 4 else ([0, 1, 0] if i == 4 else [0, 0, 1]))
        for i in range(6)
    )
    study = ci.ChoiceStudy(
        label="pricing",
        outcome="categorical_choice",
        options=("a", "b", "c"),
        meaning="chosen in the pricing task",
        population="customers",
        respondents=respondents,
        initial_prior=(1.0, 1.0, 1.0),
        sources=(),
        looks=(6,),
        stopping_rule=ai.StoppingRule.no_stopping(),
    )
    session = ci.ChoiceInferenceSession.prepare(study)
    request = ci.Posterior(look=0)
    with pytest.warns(AntecedentDeprecationWarning, match="compiled_interval"):
        direct = session.measured_interval(request)
    compiled = compiled_quietly(lambda: session.compiled_interval(request))
    assert_identical(direct, compiled)
    # Dirichlet(1, 1, 1) and counts (4, 1, 1): the first share is Beta(5, 4), mean 5 / 9.
    assert compiled.scalars[0].point == pytest.approx(5.0 / 9.0, abs=1e-12)


# -- temporal ----------------------------------------------------------------------------------


def test_the_direct_temporal_interval_warns_and_equals_the_compiled_interval():
    law = InitialStateLaw("fixed_target", {0: 0.3, 1: 0.7})
    units = panel()
    execution = Execution(seed=901)
    with pytest.warns(AntecedentDeprecationWarning, match="compiled_dependent_interval"):
        direct = temporal_dependent_interval(
            units, sequence=(0, 0), target_law=law, execution=execution
        )
    compiled = compiled_quietly(
        lambda: units.compiled_dependent_interval(
            sequence=(0, 0), target_law=law, execution=execution
        )
    )
    assert_identical(direct, compiled)
    # Every unit's response is 0.49 plus its shock of +-0.2, half each way.
    point = compiled.scalar("response").point
    assert math.isclose(point, 0.49, abs_tol=1e-12)
