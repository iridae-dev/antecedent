"""Bayesian backend selection on the analyze path (Shared UX)."""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.estimation import _bayesian_inference_kwargs
from antecedent.outcomes import require_success


def _confounded(n: int = 120, seed: int = 5):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = 2.0 * t + z + 0.3 * rng.normal(size=n)
    return {"z": z, "t": t, "y": y}, [("z", "t"), ("z", "y"), ("t", "y")]


def _ols_effect(data) -> float:
    """Adjusted OLS coefficient of ``t`` in ``y ~ 1 + t + z``: the diffuse-prior posterior mean."""
    design = np.column_stack([np.ones(len(data["y"])), data["t"], data["z"]])
    return float(np.linalg.lstsq(design, data["y"], rcond=None)[0][1])


def test_bayesian_conjugate_vs_laplace_backend():
    data, edges = _confounded()
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    laplace = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=query,
            inference=antecedent.Bayesian(n_draws=64, backend="laplace"),
            refute=False,
            execution=antecedent.Execution(seed=1),
        )
    )
    conjugate = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=query,
            inference=antecedent.Bayesian(n_draws=64, backend="conjugate"),
            refute=False,
            execution=antecedent.Execution(seed=1),
        )
    )
    assert laplace.posterior is not None
    assert conjugate.posterior is not None
    assert "conjugate" in (conjugate.posterior.backend or "").lower()
    assert np.isfinite(laplace.posterior.effect_mean)
    assert np.isfinite(conjugate.posterior.effect_mean)
    # Unknown-σ² GaussianIdentity Laplace is the NIG conjugate posterior (not a
    # block-diagonal Laplace Hessian). Requesting backend="laplace" must not
    # invent a different approximation.
    assert laplace.posterior.backend == conjugate.posterior.backend
    assert laplace.posterior.backend == "conjugate_gaussian"
    assert abs(laplace.posterior.effect_mean - conjugate.posterior.effect_mean) < 0.05
    # Independent reference: with the default weak prior the posterior mean sits on the
    # adjusted least-squares coefficient (observed gap 1e-3); the unadjusted contrast is far
    # from it, so this fails for a backend that drops the confounder.
    reference = _ols_effect(data)
    assert abs(conjugate.posterior.effect_mean - reference) < 0.01
    assert abs(laplace.posterior.effect_mean - reference) < 0.01


def test_bayesian_hmc_smoke():
    data, edges = _confounded(n=100, seed=9)
    result = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            # Publication-gate schedule (Ř≤1.01, ESS≥100); Rust floors HMC draws
            # at 3000 when callers ask for less.
            inference=antecedent.Bayesian(n_draws=4000, backend="hmc"),
            refute=False,
            execution=antecedent.Execution(seed=2),
        )
    )
    assert result.posterior is not None
    assert np.isfinite(result.posterior.effect_mean)
    assert result.posterior.n_draws is not None and result.posterior.n_draws >= 4000
    assert "hmc" in (result.posterior.backend or "").lower() or result.posterior.backend
    # The sampled posterior mean recovers the adjusted least-squares coefficient (observed gap 1e-3).
    assert abs(result.posterior.effect_mean - _ols_effect(data)) < 0.02


def test_bayesian_unknown_backend_rejected():
    class _Bad:
        n_draws = 16
        prior_scale = 10.0
        prior_from = None
        backend = "not-a-backend"

    with pytest.raises(ValueError, match="backend"):
        _bayesian_inference_kwargs(_Bad())  # type: ignore[arg-type]


def test_hmc_execution_work_ceiling_refuses_before_sampling():
    data, edges = _confounded(n=100, seed=9)
    with pytest.raises(antecedent.errors.CausalResourceError, match="hmc_work_budget"):
        require_success(antecedent.analyze(
            data, graph=edges,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Bayesian(n_draws=4000, backend="hmc"),
            refute=False, execution=antecedent.Execution(seed=2, hmc_work_limit=1),
        ))
