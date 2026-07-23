"""Second-click refute via PreparedAnalysis (BACKLOG E)."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def _confounded(n: int = 400, seed: int = 11):
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


def test_prepared_refute_second_click():
    data, edges = _confounded()
    prepared = causal.PreparedAnalysis.prepare(
        data,
        graph=edges,
        query=causal.AverageEffect(treatment="t", outcome="y"),
        latency="interactive",
        refute=False,
        bootstrap=0,
        seed=5,
    )
    first = prepared.estimate(data, seed=5)
    assert not first.validation.ran
    ate = first.ate

    second = prepared.refute(data, suite="placebo", seed=5)
    assert abs(second.ate - ate) < 1e-12
    assert second.validation.ran
    assert second.validation.count >= 1

    # Convenience on the result object.
    third = first.refute(data, suite="placebo", seed=5)
    assert abs(third.ate - ate) < 1e-12
    assert third.validation.ran
