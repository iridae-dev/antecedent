"""PanelFrame Bayesian pulse via analyze_panel (P2)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def _lag1_unit(n: int = 200, seed: int = 3, coef: float = 0.9):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.empty(n)
    y[0] = rng.normal()
    for t in range(1, n):
        y[t] = coef * x[t - 1] + 0.05 * rng.normal()
    return {"x": x, "y": y}


def test_panel_bayesian_pulse_smoke():
    panel = antecedent.panel(
        [
            _lag1_unit(seed=3),
            _lag1_unit(seed=4),
            _lag1_unit(seed=5),
        ]
    )
    result = antecedent.analyze(
        panel,
        graph=[("x", 1, "y", 0)],
        query=antecedent.PulseEffect(
            treatment="x",
            outcome="y",
            treatment_lag=1,
            horizon_steps=1,
            active_level=1.0,
        ),
        inference=antecedent.Bayesian(n_draws=128),
        refute=False,
        bootstrap=0,
        seed=42,
    )
    assert result.posterior is not None
    assert np.isfinite(result.posterior.effect_mean)
    assert np.isfinite(result.ate)
    assert abs(result.ate - result.posterior.effect_mean) < 1e-12
    assert "bayesian" in result.estimate.estimator_id.lower()
    assert abs(result.posterior.effect_mean - 0.9) < 0.15


def test_panel_frequentist_pulse_baseline():
    panel = antecedent.panel([_lag1_unit(seed=3), _lag1_unit(seed=4)])
    result = antecedent.analyze(
        panel,
        graph=[("x", 1, "y", 0)],
        query=antecedent.PulseEffect(
            treatment="x",
            outcome="y",
            treatment_lag=1,
            horizon_steps=1,
            active_level=1.0,
        ),
        refute=False,
        bootstrap=0,
        seed=1,
    )
    assert np.isfinite(result.ate)
    assert result.posterior is None
