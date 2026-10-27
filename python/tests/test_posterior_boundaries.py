"""Typed boundaries of two posteriors that have no same-target counterpart.

The temporal response-curve posterior is a tempered generalized posterior, so an ordinary
posterior of a joint law of the series is refused unless independent innovations are attested;
a trajectory survival contrast under a prescribed regime has a conditional subject-bootstrap
interval and no credible interval. Each refusal names the missing premise and the next attempt.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent.errors import CausalError
from antecedent.regimes import (
    trajectory_hazard_credible_interval,
    trajectory_survival_bootstrap,
    trajectory_survival_contrast,
)
from antecedent.temporal import require_ordinary_temporal_curve_posterior


def test_unmodeled_serial_dependence_refuses_the_ordinary_reading():
    with pytest.raises(CausalError) as caught:
        require_ordinary_temporal_curve_posterior("unmodeled")
    error = caught.value
    assert error.reason_code == "cell_not_licensed"
    text = str(error)
    assert "temporal_curve.ordinary_posterior_not_licensed" in text
    assert "next attempt" in text and "autoregressive innovation law" in text


def test_attested_independent_innovations_license_the_ordinary_reading():
    assert require_ordinary_temporal_curve_posterior("independent_innovations") is None
    with pytest.raises(ValueError, match="serial_error_model must be"):
        require_ordinary_temporal_curve_posterior("nonsense")  # type: ignore[arg-type]


def _one_period(subjects: int = 600, seed: int = 5):
    rng = np.random.default_rng(seed)
    arm = np.arange(subjects) % 2 == 0
    event = rng.random(subjects) < np.where(arm, 0.25, 0.40)
    shape = (subjects, 1)
    return (
        dict(
            treatment=arm.reshape(shape),
            event=event.reshape(shape),
            at_risk=np.ones(shape, dtype=bool),
            treatment_probability=np.full(shape, 0.5),
            censoring_weight=np.ones(shape),
            minimum_probability=0.2,
            active=[True],
            control=[False],
        ),
        arm,
        event,
    )


def test_bootstrap_interval_width_matches_the_binomial_standard_error():
    # One period, equal treatment probabilities and unit censoring weights: the product limit is
    # the plain event share among followers, so the independent reference is the two-sample
    # binomial standard error of the survival difference.
    data, arm, event = _one_period()
    intervals = trajectory_survival_bootstrap(**data, replicates=999, seed=17)
    p1, n1 = event[arm].mean(), arm.sum()
    p0, n0 = event[~arm].mean(), (~arm).sum()
    standard_error = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    point, _ = trajectory_survival_contrast(**data)
    assert point == pytest.approx((1 - p1) - (1 - p0))
    lower, upper = intervals.survival_difference
    assert lower < point < upper
    assert (upper - lower) / (2 * 1.959964 * standard_error) == pytest.approx(1.0, abs=0.15)
    assert intervals.replicates_requested == 999 and intervals.replicates_ok >= 899


def test_credible_interval_always_refuses_and_names_the_missing_likelihood():
    data, _, _ = _one_period()
    with pytest.raises(CausalError) as caught:
        trajectory_hazard_credible_interval(**data)
    error = caught.value
    assert error.reason_code == "cell_not_licensed"
    text = str(error)
    assert "trajectory_survival.credible_interval_not_licensed" in text
    assert "joint likelihood" in text and "g-formula" in text and "next attempt" in text
    # Invalid trajectories are a data error, not the boundary.
    with pytest.raises(CausalError) as invalid:
        trajectory_hazard_credible_interval(**{**data, "active": [False], "control": [False]})
    assert getattr(invalid.value, "reason_code", None) != "cell_not_licensed"


def test_survival_contrast_and_bootstrap_refuse_trajectories_that_follow_no_one():
    data, _, _ = _one_period()
    nobody = {**data, "active": [False], "control": [False]}
    with pytest.raises(CausalError) as contrast:
        trajectory_survival_contrast(**nobody)
    with pytest.raises(CausalError) as bootstrap:
        trajectory_survival_bootstrap(**nobody, replicates=50, seed=1)
    for error in (contrast.value, bootstrap.value):
        assert getattr(error, "reason_code", None) != "cell_not_licensed"
