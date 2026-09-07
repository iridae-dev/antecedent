"""Conditional / context effect OO dual of conformance/context/conditional_effect."""

from __future__ import annotations

import antecedent
import numpy as np
import pytest


def test_conditional_effect_recovers_interaction():
    n = 200
    t = np.asarray([0.0 if i % 2 == 0 else 1.0 for i in range(n)], dtype=np.float64)
    w = np.asarray([(i % 5) for i in range(n)], dtype=np.float64)
    y = 1.0 + 2.0 * t + 0.5 * t * w
    data = {"t": t, "y": y, "w": w}
    edges = [("t", "y"), ("w", "y")]
    result = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.ConditionalEffect("t", "y", "w"),
        refute=False,
        bootstrap=0,
        seed=1,
    )
    assert abs(result.ate - 3.0) < 0.3


def test_conditional_bayesian_consults_prior():
    rng = np.random.default_rng(12)
    t, w = rng.normal(size=(2, 100))
    data = {"t": t, "w": w, "y": 2 * t + 0.5 * t * w + rng.normal(scale=0.2, size=100)}
    result = antecedent.analyze(
        data,
        graph=[("t", "y"), ("w", "y")],
        query=antecedent.ConditionalEffect("t", "y", "w"),
        inference=antecedent.Bayesian(backend="conjugate", n_draws=512),
        refute=False,
        bootstrap=0,
        return_posterior_artifact=True,
    )
    assert result.posterior is not None
    assert result.posterior.artifact is not None
    assert result.ate == pytest.approx(2 + 0.5 * w.mean(), abs=0.1)
