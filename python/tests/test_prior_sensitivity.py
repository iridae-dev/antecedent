"""Prior sensitivity on the Bayesian analyze path (Shared UX)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("causal")
import causal


def _confounded(n: int = 140, seed: int = 11):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = 2.0 * t + z + 0.3 * rng.normal(size=n)
    return {"z": z, "t": t, "y": y}, [("z", "t"), ("z", "y"), ("t", "y")]


def test_prior_sensitivity_on_refute_full():
    data, edges = _confounded()
    result = causal.analyze(
        data,
        graph=edges,
        query=causal.AverageEffect(treatment="t", outcome="y"),
        inference=causal.Bayesian(n_draws=48, backend="conjugate"),
        refute="full",
        seed=1,
    )
    sens = result.validation.prior_sensitivity
    assert sens is not None
    assert len(sens.scales) >= 3
    assert sens.alphas is None
    assert len(sens.effect_means) == len(sens.scales)
    assert len(sens.effect_sds) == len(sens.scales)
    assert all(np.isfinite(m) for m in sens.effect_means)


def test_prior_sensitivity_skipped_on_default_refute():
    data, edges = _confounded()
    result = causal.analyze(
        data,
        graph=edges,
        query=causal.AverageEffect(treatment="t", outcome="y"),
        inference=causal.Bayesian(n_draws=32),
        refute=True,
        seed=2,
    )
    assert result.validation.prior_predictive is not None
    assert result.validation.prior_sensitivity is None
