"""Mechanism change and rollout on a fitted TemporalDag from Python."""

from __future__ import annotations

import numpy as np
import pytest
from antecedent.temporal_model import detect_mechanism_change, rollout_dynamics

EDGES = [("x", 1, "y", 0)]


def _series(shift: float, n: int = 300, seed: int = 4):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + rng.normal()
        intercept = 1.0 if t < n // 2 else 1.0 + shift
        y[t] = intercept + 2.0 * x[t - 1] + rng.normal()
    return ["x", "y"], [x, y]


@pytest.mark.parametrize(
    "test",
    [
        "likelihood_ratio",
        "mean_difference",
        "kernel_two_sample",
        "classifier_two_sample",
        "change_point",
    ],
)
def test_a_shifted_series_is_detected_by_every_statistic_with_a_block_null(test):
    names, cols = _series(5.0)
    (found,) = detect_mechanism_change(names, cols, EDGES, ["y"], (0, 150), (150, 300), test=test)
    assert found.changed and found.p_value is not None and found.p_value < 0.05
    assert "block_permutation" in found.method
    assert found.posterior_probability is None


def test_the_posterior_probability_needs_a_declared_prior():
    names, cols = _series(5.0)
    (shifted,) = detect_mechanism_change(
        names, cols, EDGES, ["y"], (0, 150), (150, 300), posterior_draws=4, change_prior=0.5
    )
    assert shifted.posterior_probability is not None and shifted.posterior_probability > 0.95
    names, cols = _series(0.0)
    (calm,) = detect_mechanism_change(
        names, cols, EDGES, ["y"], (0, 150), (150, 300), posterior_draws=4, change_prior=0.5
    )
    assert calm.posterior_probability is not None and calm.posterior_probability < 0.5
    with pytest.raises(Exception, match="prior"):
        detect_mechanism_change(names, cols, EDGES, ["y"], (0, 150), (150, 300), posterior_draws=4)


def test_the_rollout_steps_the_dynamics_and_carries_block_refit_uncertainty():
    names, cols = _series(0.0)
    hold = [{"variable": "x", "value": 1.0, "from": 0, "until": 3}]
    rolled = rollout_dynamics(names, cols, EDGES, n_steps=4, interventions=hold)
    assert rolled.mean(2, "y") == pytest.approx(3.0, abs=0.4)  # y = 1 + 2 * x_{t-1}
    bayes = rollout_dynamics(names, cols, EDGES, n_steps=4, interventions=hold, posterior_draws=10)
    assert bayes.uncertainty is not None and bayes.uncertainty.draws == 10
    assert len(bayes.uncertainty.mean) == 4 * 2 * 2  # step means, then simulated-path draws
    assert sum(name.startswith("path.") for name in bayes.uncertainty.names) == 4 * 2
    idx = bayes.uncertainty.names.index("path.step[2].y")
    assert bayes.uncertainty.sd[idx] > 0.5  # a path carries the innovation noise
    assert (
        bayes.uncertainty.sd[idx]
        > 3 * bayes.uncertainty.sd[bayes.uncertainty.names.index("step[2].y")]
    )
    assert bayes.uncertainty.ci_low[idx] < 3.0 < bayes.uncertainty.ci_high[idx]
    boot = rollout_dynamics(
        names, cols, EDGES, n_steps=4, interventions=hold, bootstrap_replicates=10
    )
    assert boot.uncertainty is not None and boot.uncertainty.se
