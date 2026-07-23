"""Prior / posterior predictive checks on the Bayesian analyze path (P1-A/B)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def _confounded(n: int = 200, seed: int = 7):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = 2.0 * t + z + 0.3 * rng.normal(size=n)
    return {"z": z, "t": t, "y": y}, [("z", "t"), ("z", "y"), ("t", "y")]


def test_bayesian_ate_validation_includes_prior_and_posterior_ppc():
    data, edges = _confounded()
    result = causal.analyze(
        data,
        graph=edges,
        query=causal.AverageEffect(treatment="t", outcome="y"),
        inference=causal.Bayesian(n_draws=64),
        seed=1,
    )
    prior = result.validation.prior_predictive
    assert prior is not None
    assert prior.kind == "prior_predictive"
    assert np.isfinite(prior.p_value)
    assert np.isfinite(prior.observed)
    assert np.isfinite(prior.predictive_mean)
    assert np.isfinite(prior.predictive_sd)
    assert prior.n_sims > 0

    post = result.validation.posterior_predictive
    assert post is not None
    assert post.kind == "posterior_predictive"
    assert np.isfinite(post.p_value)
    assert np.isfinite(post.observed)
    assert np.isfinite(post.predictive_mean)
    assert np.isfinite(post.predictive_sd)
    assert post.n_sims > 0

    assert result.validation.ran is True
    assert result.validation.count >= 2


def test_bayesian_ate_refute_false_skips_ppc():
    data, edges = _confounded()
    result = causal.analyze(
        data,
        graph=edges,
        query=causal.AverageEffect(treatment="t", outcome="y"),
        inference=causal.Bayesian(n_draws=32),
        refute=False,
        seed=1,
    )
    assert result.validation.prior_predictive is None
    assert result.validation.posterior_predictive is None
    assert result.posterior is not None
