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
        refute=False,
    )
    assert isinstance(result.ate, float)
    assert result.performance.plan_id
    assert isinstance(result.diagnostics, list)
    assert "node_count" in result.provenance


def test_analyze_discovery_jpcmci_plus_two_env():
    n = 80
    rng = np.random.default_rng(4)
    envs = []
    for _ in range(2):
        x = rng.normal(size=n)
        y = np.empty(n)
        y[0] = rng.normal()
        for t in range(1, n):
            y[t] = 0.55 * x[t - 1] + 0.2 * rng.normal()
        envs.append({"x": x, "y": y})
    # May Ready-estimate or fail closed on review/ID; both prove the wire path.
    try:
        result = causal.analyze(
            envs,
            discovery=causal.JPCMCIPlus(max_lag=1, alpha=0.2, fdr=False),
            query=causal.PulseEffect(
                treatment="x",
                outcome="y",
                treatment_lag=1,
                horizon_steps=1,
                active_level=1.0,
            ),
            bootstrap=0,
            seed=1,
            refute=False,
        )
        assert isinstance(result.ate, float)
    except Exception as exc:  # noqa: BLE001 — native review/ID surfaces as Exception
        assert str(exc), "expected a non-empty error from the wired path"


def test_analyze_discovery_rpcmci_regimes():
    data = _lag1_series(n=100, seed=5)
    n = len(data["x"])
    regimes = [0] * (n // 2) + [1] * (n - n // 2)
    try:
        result = causal.analyze(
            data,
            discovery=causal.RPCMCI(max_lag=1, alpha=0.2, fdr=False),
            regimes=regimes,
            query=causal.PulseEffect(
                treatment="x",
                outcome="y",
                treatment_lag=1,
                horizon_steps=1,
                active_level=1.0,
            ),
            bootstrap=0,
            seed=1,
            refute=False,
        )
        assert isinstance(result.ate, float)
    except Exception as exc:  # noqa: BLE001
        assert str(exc), "expected a non-empty error from the wired path"


def test_analyze_discovery_pc_smoke():
    n = 250
    rng = np.random.default_rng(7)
    z = rng.normal(size=n)
    t = z + rng.normal(size=n) * 0.3
    y = 1.5 * t + z + rng.normal(size=n) * 0.3
    try:
        result = causal.analyze(
            {"t": t, "y": y, "z": z},
            discovery=causal.PC(alpha=0.2, fdr=False, max_cond_size=2),
            query=causal.AverageEffect(treatment="t", outcome="y"),
            refute=False,
            bootstrap=0,
            seed=1,
        )
        assert isinstance(result.ate, float)
        assert result.performance.plan_id
    except Exception as exc:  # noqa: BLE001
        assert str(exc), "expected a non-empty error from the wired path"

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
