"""Observational survival effects at a declared horizon against a closed-form truth.

Exponential event times with a proportional-hazards treatment effect and two discrete
confounders, so each arm's survival probability and restricted mean survival time are finite
sums of exponentials over the six confounder cells, computed here independently of the
estimator. Censoring depends on the arm and the confounders (coarsening at random).
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import observational_survival
from antecedent.errors import CausalError

HORIZON = 2.0
BASE, BETA, G1, G2 = 0.25, -0.5, 0.7, 0.5


def _rate(a: np.ndarray | float, z1: np.ndarray | float, z2: np.ndarray | float):
    return BASE * np.exp(BETA * a + G1 * z1 + G2 * z2)


def _truth(treated: float) -> tuple[float, float]:
    survival = rmst = 0.0
    for z1 in (0.0, 1.0):
        for z2 in (-1.0, 0.0, 1.0):
            rate = _rate(treated, z1, z2)
            survival += np.exp(-rate * HORIZON) / 6.0
            rmst += (1.0 - np.exp(-rate * HORIZON)) / rate / 6.0
    return survival, rmst


def _draw(n: int = 5000, seed: int = 12) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z1 = rng.integers(0, 2, n).astype(float)
    z2 = rng.integers(-1, 2, n).astype(float)
    a = (rng.uniform(size=n) < 1 / (1 + np.exp(-(-0.2 + 0.9 * z1 + 0.6 * z2)))).astype(float)
    t = rng.exponential(1 / _rate(a, z1, z2))
    c = rng.exponential(1 / (0.12 * np.exp(0.9 * z1 - 0.6 * z2 + 0.4 * a)))
    return {
        "duration": np.minimum(t, c),
        "event": (t <= c).astype(float),
        "treatment": a,
        "z1": z1,
        "z2": z2,
    }


def _effect(data: dict[str, np.ndarray], **options):
    return observational_survival.survival_effect(
        data["duration"],
        data["event"],
        data["treatment"],
        active=1.0,
        control=0.0,
        adjust={"z1": data["z1"], "z2": data["z2"]},
        **options,
    )


def test_survival_and_rmst_differences_recover_the_closed_form_truth() -> None:
    data = _draw()
    effect = _effect(data, horizon=HORIZON, censoring="at_random_given_covariates", seed=4)
    s1, r1 = _truth(1.0)
    s0, r0 = _truth(0.0)
    assert abs(effect.survival_difference.estimate - (s1 - s0)) < 4 * effect.survival_difference.se
    assert abs(effect.rmst_difference.estimate - (r1 - r0)) < 4 * effect.rmst_difference.se
    assert abs(effect.active.survival - s1) < 4 * effect.active.survival_se
    assert abs(effect.control.rmst - r0) < 4 * effect.control.rmst_se
    assert effect.horizon == HORIZON
    assert effect.censoring == "at_random_given_covariates"
    assert effect.calibration == "unmeasured"
    assert any("coarsening at random" in text for text in effect.assumptions)
    assert effect.n_obs == 5000 and len(effect.row_index) == 5000
    assert (
        effect.to_dict()["survival_difference"]["estimate"] == effect.survival_difference.estimate
    )


def test_the_horizon_and_the_censoring_assumption_have_no_default() -> None:
    data = _draw(400, seed=3)
    with pytest.raises(CausalError) as raised:
        _effect(data, censoring="independent_within_arm")
    assert raised.value.reason_code == "required_option_missing"
    assert "survival_effect.horizon_missing" in str(raised.value)
    with pytest.raises(CausalError) as raised:
        _effect(data, horizon=HORIZON)
    assert raised.value.reason_code == "required_option_missing"
    assert "survival_effect.censoring_assumption_missing" in str(raised.value)


def test_support_violations_are_refused_with_their_own_details() -> None:
    data = _draw(600, seed=8)
    with pytest.raises(CausalError) as raised:
        _effect(data, horizon=500.0, censoring="independent_within_arm")
    assert raised.value.reason_code == "positivity_violated"
    assert "survival_effect.horizon_beyond_follow_up" in str(raised.value)
    bad = dict(data, treatment=np.where(np.arange(600) == 4, 2.0, data["treatment"]))
    with pytest.raises(CausalError) as raised:
        _effect(bad, horizon=HORIZON, censoring="independent_within_arm")
    assert raised.value.reason_code == "treatment_level_undeclared"
