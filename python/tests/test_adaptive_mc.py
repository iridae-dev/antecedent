"""Adaptive Monte Carlo (Python dual of Rust backlog C bootstrap pin)."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def _confounded_scm(n: int = 500, seed: int = 19):
    rng = random.Random(seed)
    z = np.empty(n, dtype=np.float64)
    t = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        zi = rng.gauss(0.0, 1.0)
        p = 1.0 / (1.0 + math.exp(-(-0.4 + 0.9 * zi)))
        ti = 1.0 if rng.random() < p else 0.0
        yi = 2.0 * ti + zi + rng.gauss(0.0, 0.4)
        z[i] = zi
        t[i] = ti
        y[i] = yi
    return {"t": t, "y": y, "z": z}, [("z", "t"), ("z", "y"), ("t", "y")]


def test_adaptive_bootstrap_records_early_stop():
    """Production context enables adaptive bootstrap; effort fields are honest."""
    data, edges = _confounded_scm()
    max_reps = 80
    result = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        bootstrap=max_reps,
        refute=False,
        seed=5,
    )
    ok = result.performance.bootstrap_replicates_ok
    assert ok is not None
    assert 2 <= ok <= max_reps
    assert result.performance.bootstrap_replicates_requested == max_reps
    # Production adaptive may early-stop; when it does, flag must be set and
    # actual count must be strictly below the requested max.
    if result.performance.early_stopped:
        assert ok < max_reps
    assert result.estimate.se_bootstrap is not None
    assert math.isfinite(result.estimate.se_bootstrap)


def test_adaptive_bayesian_draws_records_early_stop():
    """Production context enables adaptive Laplace draws; effort fields are honest."""
    data, edges = _confounded_scm()
    max_draws = 256
    result = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Bayesian(backend="laplace", n_draws=max_draws),
        refute=False,
        seed=9,
    )
    n = result.performance.n_draws
    assert n is not None
    assert 2 <= n <= max_draws
    if result.performance.early_stopped:
        assert n < max_draws
    assert result.posterior is not None
    assert math.isfinite(result.ate)
