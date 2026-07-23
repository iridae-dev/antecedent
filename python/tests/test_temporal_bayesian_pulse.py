"""Manufacturing-style Bayesian temporal pulse dual (P0)."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def test_manufacturing_bayesian_pulse_recovers_effect():
    n = 400
    pressure = np.array([math.sin(0.04 * t) for t in range(n)], dtype=np.float64)
    defect = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        defect[t] = 0.9 * pressure[t - 1]

    result = causal.analyze(
        {"pressure": pressure, "defect": defect},
        graph=[("pressure", 1, "defect", 0)],
        query=causal.PulseEffect(
            treatment="pressure",
            outcome="defect",
            treatment_lag=1,
            horizon_steps=1,
            active_level=1.0,
        ),
        inference=causal.Bayesian(n_draws=256),
        refute=False,
        bootstrap=0,
        seed=42,
    )
    assert result.posterior is not None
    assert abs(result.posterior.effect_mean - 0.9) < 0.05
    assert abs(result.ate - result.posterior.effect_mean) < 1e-12
    assert np.isfinite(result.posterior.p_below_zero)
    assert result.estimate.estimator_id == "bayesian.temporal.gcomp"
    assert result.identification.method  # non-empty
    # Full draw artifacts are opt-in on static analyze; temporal defaults to summaries.
    assert result.posterior.n_draws is not None and result.posterior.n_draws > 0
    assert result.posterior.artifact is None
