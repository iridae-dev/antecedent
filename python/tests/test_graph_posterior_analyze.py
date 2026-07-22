"""P1-D: graph-posterior × Bayesian effect mixture via analyze."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("causal")
import causal


def test_exact_dag_posterior_bayesian_ate_mixture():
    n = 200
    z = np.linspace(0.0, 1.0, n, dtype=np.float64)
    t = (z > 0.5).astype(np.float64)
    y = 1.0 + 2.0 * t + 3.0 * z

    result = causal.analyze(
        {"t": t, "y": y, "z": z},
        discovery=causal.ExactDagPosterior(),
        query=causal.AverageEffect(treatment="t", outcome="y"),
        inference=causal.Bayesian(n_draws=80, prior_scale=100.0, backend="conjugate"),
        refute=False,
        bootstrap=0,
        seed=7,
    )
    assert result.posterior is not None
    mass = result.posterior.unidentified_mass
    assert mass is not None
    assert 0.0 <= mass <= 1.0
    assert np.isfinite(result.posterior.effect_mean)
    assert np.isfinite(result.ate)
    if mass > 0.0:
        assert result.posterior.envelope is not None
        assert result.posterior.envelope.unidentified_mass == mass


def test_exact_dag_posterior_rejects_frequentist():
    n = 80
    z = np.linspace(0.0, 1.0, n, dtype=np.float64)
    t = (z > 0.5).astype(np.float64)
    y = 1.0 + 2.0 * t + 3.0 * z
    with pytest.raises(TypeError, match="Bayesian"):
        causal.analyze(
            {"t": t, "y": y, "z": z},
            discovery=causal.ExactDagPosterior(),
            query=causal.AverageEffect(treatment="t", outcome="y"),
            inference=causal.Frequentist(),
            refute=False,
            bootstrap=0,
            seed=1,
        )


def test_dbn_posterior_bayesian_pulse_mixture():
    n = 400
    rng = np.random.default_rng(42)
    # White-noise treatment keeps BIC mass on the lag edge (AR loops often
    # fail temporal backdoor history caps under the DBN mixture).
    pressure = rng.normal(size=n).astype(np.float64)
    defect = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        defect[t] = 0.9 * pressure[t - 1]

    result = causal.analyze(
        {"pressure": pressure, "defect": defect},
        discovery=causal.DbnPosterior(max_lag=1),
        query=causal.PulseEffect(
            treatment="pressure",
            outcome="defect",
            treatment_lag=1,
            horizon_steps=1,
            active_level=1.0,
        ),
        inference=causal.Bayesian(n_draws=64, prior_scale=100.0, backend="conjugate"),
        refute=False,
        bootstrap=0,
        seed=11,
    )
    assert result.posterior is not None
    mass = result.posterior.unidentified_mass
    assert mass is not None
    assert 0.0 <= mass <= 1.0
    assert np.isfinite(result.posterior.effect_mean)
    assert abs(result.posterior.effect_mean - 0.9) < 0.35
