"""Prior / posterior predictive checks on the Bayesian analyze path (P1-A/B)."""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.outcomes import require_success


def _confounded(n: int = 200, seed: int = 7):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = 2.0 * t + z + 0.3 * rng.normal(size=n)
    return {"z": z, "t": t, "y": y}, [("z", "t"), ("z", "y"), ("t", "y")]


def test_bayesian_ate_validation_includes_prior_and_posterior_ppc():
    data, edges = _confounded()
    result = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Bayesian(n_draws=64),
            execution=antecedent.Execution(seed=1),
        )
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
    result = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Bayesian(n_draws=32),
            refute=False,
            execution=antecedent.Execution(seed=1),
        )
    )
    assert result.validation.prior_predictive is None
    assert result.validation.posterior_predictive is None
    assert result.posterior is not None


def _analyze(data, edges, **bayesian):
    return require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Bayesian(n_draws=200, **bayesian),
            execution=antecedent.Execution(seed=1),
        )
    )


def test_predictive_checks_pin_the_observed_outcome_mean_and_pass_a_well_specified_model():
    data, edges = _confounded()
    result = _analyze(data, edges)
    observed = float(np.mean(data["y"]))
    prior = result.validation.prior_predictive
    post = result.validation.posterior_predictive
    assert (prior.kind, post.kind) == ("prior_predictive", "posterior_predictive")
    # The location statistic is the outcome mean, computed here independently of the library.
    assert prior.observed == pytest.approx(observed, abs=1e-9)
    assert post.observed == pytest.approx(observed, abs=1e-9)
    # A regression with an intercept reproduces the outcome mean, so the posterior replicate mean
    # sits on it to within Monte Carlo error and the check does not flag the model.
    assert abs(post.predictive_mean - observed) < 3.0 * post.predictive_sd
    assert post.p_value > 0.05
    assert 0.0 <= post.dispersion_p_value <= 1.0
    # Posterior replicates always carry the likelihood's noise; a prior check under an improper
    # residual-variance prior replicates the fitted means only.
    assert post.noise == "likelihood"
    assert prior.noise in {"likelihood", "mean_only_improper_scale"}
    assert 0.0 <= prior.dispersion_p_value <= 1.0


def test_predictive_checks_flag_a_prior_that_pins_the_coefficients_far_from_the_data():
    data, edges = _confounded()
    observed = float(np.mean(data["y"]))
    assert observed > 0.7
    # A prior concentrated at zero cannot reproduce an outcome whose mean is about three quarters, before or
    # after the data are seen, so both predictive checks put the observed mean in the tail.
    result = _analyze(data, edges, prior_scale=0.01)
    for check in (result.validation.prior_predictive, result.validation.posterior_predictive):
        assert check.observed == pytest.approx(observed, abs=1e-9)
        assert abs(check.predictive_mean) < 0.5 * observed
        assert check.p_value < 0.1
