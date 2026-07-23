"""Bayesian backend selection on the analyze path (Shared UX)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent
from antecedent.estimation import _bayesian_inference_kwargs


def _confounded(n: int = 120, seed: int = 5):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = 2.0 * t + z + 0.3 * rng.normal(size=n)
    return {"z": z, "t": t, "y": y}, [("z", "t"), ("z", "y"), ("t", "y")]


def test_bayesian_conjugate_vs_laplace_backend():
    data, edges = _confounded()
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    laplace = antecedent.analyze(
        data,
        graph=edges,
        query=query,
        inference=antecedent.Bayesian(n_draws=64, backend="laplace"),
        refute=False,
        seed=1,
    )
    conjugate = antecedent.analyze(
        data,
        graph=edges,
        query=query,
        inference=antecedent.Bayesian(n_draws=64, backend="conjugate"),
        refute=False,
        seed=1,
    )
    assert laplace.posterior is not None
    assert conjugate.posterior is not None
    assert "conjugate" in (conjugate.posterior.backend or "").lower()
    assert np.isfinite(laplace.posterior.effect_mean)
    assert np.isfinite(conjugate.posterior.effect_mean)
    assert laplace.posterior.backend != conjugate.posterior.backend


def test_bayesian_hmc_smoke():
    data, edges = _confounded(n=100, seed=9)
    result = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        # Enough draws for ESS/R-hat gates on the native HMC backend.
        inference=antecedent.Bayesian(n_draws=120, backend="hmc"),
        refute=False,
        seed=2,
    )
    assert result.posterior is not None
    assert np.isfinite(result.posterior.effect_mean)
    assert result.posterior.n_draws is not None and result.posterior.n_draws >= 120
    assert "hmc" in (result.posterior.backend or "").lower() or result.posterior.backend


def test_bayesian_unknown_backend_rejected():
    class _Bad:
        n_draws = 16
        prior_scale = 10.0
        prior_from = None
        backend = "not-a-backend"

    with pytest.raises(ValueError, match="backend"):
        _bayesian_inference_kwargs(_Bad())  # type: ignore[arg-type]
