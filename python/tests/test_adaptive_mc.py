"""Production contexts evaluate the requested Monte Carlo effort.

Python dual of the Rust `production_replicates` conformance tests: the
production context carries no early-stop budget, so the replicate and draw
counts users get are the counts the calibration gates certify.
"""

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


def _mediation_scm(n: int = 500, seed: int = 11):
    rng = random.Random(seed)
    a = np.empty(n, dtype=np.float64)
    m = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        ai = 1.0 if rng.random() < 0.5 else 0.0
        mi = 0.8 * ai + rng.gauss(0.0, 0.5)
        a[i] = ai
        m[i] = mi
        y[i] = ai + 0.6 * mi + rng.gauss(0.0, 0.6)
    return {"a": a, "m": m, "y": y}, [("a", "m"), ("a", "y"), ("m", "y")]


def _assert_full(result, requested: int) -> None:
    assert result.performance.bootstrap_replicates_requested == requested
    assert result.performance.bootstrap_replicates_ok == requested
    assert not result.performance.early_stopped
    assert result.estimate.se_bootstrap is not None
    assert math.isfinite(result.estimate.se_bootstrap)


@pytest.mark.parametrize("estimator", [None, "propensity.weighting", "aipw"])
def test_static_ate_evaluates_requested_bootstrap(estimator):
    data, edges = _confounded_scm()
    kwargs = {} if estimator is None else {"estimator": estimator}
    result = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        bootstrap=199,
        refute=False,
        seed=5,
        **kwargs,
    )
    _assert_full(result, 199)


@pytest.mark.parametrize("contrast", ["total", "mediated"])
def test_static_mediation_evaluates_requested_bootstrap(contrast):
    data, edges = _mediation_scm()
    result = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.MediationEffect("a", "y", mediators=["m"], contrast=contrast),
        bootstrap=199,
        refute=False,
        seed=5,
    )
    _assert_full(result, 199)


def test_bayesian_laplace_materializes_requested_draws():
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
    assert result.performance.n_draws == max_draws
    assert not result.performance.early_stopped
    assert result.posterior is not None
    assert math.isfinite(result.ate)
