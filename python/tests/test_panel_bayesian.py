"""PanelFrame Bayesian pulse through the prepared panel route (P2)."""

from __future__ import annotations

import antecedent
import numpy as np
from antecedent.outcomes import require_success


def _lag1_unit(n: int = 200, seed: int = 3, coef: float = 0.9):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.empty(n)
    y[0] = rng.normal()
    for t in range(1, n):
        y[t] = coef * x[t - 1] + 0.05 * rng.normal()
    return {"x": x, "y": y}


def test_panel_bayesian_pulse_smoke():
    panel = antecedent.data.panel(
        [
            _lag1_unit(seed=3),
            _lag1_unit(seed=4),
            _lag1_unit(seed=5),
        ]
    )
    result = require_success(
        antecedent.analyze(
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
            execution=antecedent.Execution(seed=42),
        )
    )
    assert result.posterior is not None
    assert np.isfinite(result.posterior.effect_mean)
    assert np.isfinite(result.ate)
    assert abs(result.ate - result.posterior.effect_mean) < 1e-12
    assert "bayesian" in result.estimate.estimator_id.lower()
    # Independent pooled least squares of y_t on x_{t-1} over the three units' rows.
    units = [_lag1_unit(seed=seed) for seed in (3, 4, 5)]
    lagged = np.concatenate([u["x"][:-1] for u in units])
    response = np.concatenate([u["y"][1:] for u in units])
    slope = np.linalg.lstsq(np.column_stack([np.ones_like(lagged), lagged]), response, rcond=None)[
        0
    ][1]
    assert abs(result.posterior.effect_mean - slope) < 1e-3


def test_panel_frequentist_pulse_baseline():
    panel = antecedent.data.panel([_lag1_unit(seed=3), _lag1_unit(seed=4)])
    result = require_success(
        antecedent.analyze(
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
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    units = [_lag1_unit(seed=seed) for seed in (3, 4)]
    lagged = np.concatenate([u["x"][:-1] for u in units])
    response = np.concatenate([u["y"][1:] for u in units])
    slope = np.linalg.lstsq(np.column_stack([np.ones_like(lagged), lagged]), response, rcond=None)[
        0
    ][1]
    assert abs(result.ate - slope) < 1e-9
    assert result.posterior is None
