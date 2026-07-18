"""discovery= on temporal analyze() and enriched result fields."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("causal")
import causal


def _lag1_series(n: int = 120, seed: int = 3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.empty(n)
    y[0] = rng.normal()
    for t in range(1, n):
        y[t] = 0.6 * x[t - 1] + 0.2 * rng.normal()
    return {"x": x, "y": y}


def test_analyze_discovery_pcmci_smoke():
    data = _lag1_series()
    result = causal.analyze(
        data,
        discovery=causal.PCMCI(max_lag=1, alpha=0.2, fdr=False),
        query=causal.PulseEffect(
            treatment="x",
            outcome="y",
            treatment_lag=1,
            horizon_steps=1,
            active_level=1.0,
        ),
        bootstrap=0,
        seed=1,
    )
    assert isinstance(result.ate, float)
    assert result.performance.plan_id
    assert isinstance(result.diagnostics, list)
    assert "node_count" in result.provenance


def test_analyze_ate_enriched_fields():
    n = 200
    rng = np.random.default_rng(1)
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = 2.0 * t + z + rng.normal(size=n) * 0.3
    result = causal.analyze(
        {"t": t, "y": y, "z": z},
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=causal.AverageEffect(treatment="t", outcome="y"),
        refute=False,
        bootstrap=0,
        seed=1,
    )
    assert result.performance.modality
    assert result.performance.plan_id
    assert isinstance(result.diagnostics, list)
    assert result.provenance["node_count"] >= 0
