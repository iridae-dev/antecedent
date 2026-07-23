"""Conditional / context effect OO dual of conformance/context/conditional_effect."""

from __future__ import annotations

import numpy as np
import pytest

import antecedent


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


def test_conditional_rejects_bayesian():
    data = {"t": np.zeros(10), "y": np.zeros(10), "w": np.zeros(10)}
    with pytest.raises(TypeError, match="Bayesian"):
        antecedent.analyze(
            data,
            graph=[("t", "y")],
            query=antecedent.ConditionalEffect("t", "y", "w"),
            inference=antecedent.Bayesian(n_draws=8),
            refute=False,
            bootstrap=0,
        )
