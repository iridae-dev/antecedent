"""Regression discontinuity via analyze(..., estimator='rd.sharp')."""

from __future__ import annotations

import numpy as np

import antecedent


def test_rd_sharp_via_analyze():
    rng = np.random.default_rng(25)
    n = 3000
    r = rng.uniform(-2.0, 2.0, size=n)
    t = (r >= 0.0).astype(np.float64)
    y = 1.0 + 2.0 * t + 0.3 * r + rng.normal(scale=0.2, size=n)
    data = {"t": t, "y": y, "r": r}
    # Empty DAG — RD does not use backdoor ID.
    result = causal.analyze(
        data,
        graph=[],
        query=causal.AverageEffect("t", "y"),
        estimator="rd.sharp",
        identifier="rd.sharp",
        running_variable="r",
        cutoff=0.0,
        bandwidth=1.5,
        refute=False,
        bootstrap=0,
        seed=26,
    )
    assert abs(result.ate - 2.0) < 0.35
    assert result.estimate.estimator_id in ("rd.sharp", "rd.sharp.local_linear", "")
