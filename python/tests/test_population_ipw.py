"""Population registry + custom-distribution IPW via analyze()."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

pytest.importorskip("causal")
import causal


def _confounded_scm(n: int = 1200, seed: int = 21):
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
    return {"t": t, "y": y, "z": z}


def test_custom_distribution_ipw_via_analyze():
    data = _confounded_scm()
    n = len(data["t"])
    weights = [1.0] * n
    for i in range(n // 2):
        weights[i] = 0.5
    reg = causal.PopulationRegistry()
    reg.insert_distribution(7, weights)
    result = causal.analyze(
        data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=causal.AverageEffect(
            treatment="t",
            outcome="y",
            target_population=causal.target_custom_distribution(7),
        ),
        population_registry=reg,
        estimator="propensity.weighting",
        refute=False,
        bootstrap=0,
        seed=11,
    )
    assert abs(result.ate - 2.0) < 0.35
