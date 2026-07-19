"""RPCMCI requires explicit regimes (no silent half-split)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("causal")
import causal


def _lag1_series(n: int = 80, seed: int = 9):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.empty(n)
    y[0] = rng.normal()
    for t in range(1, n):
        y[t] = 0.55 * x[t - 1] + 0.25 * rng.normal()
    return {"x": x, "y": y}


def test_discover_rpcmci_requires_regimes():
    data = _lag1_series()
    with pytest.raises(TypeError):
        causal.discover_rpcmci(data=data, max_lag=1, alpha=0.2, fdr=False)


def test_discover_rpcmci_with_half_split_helper():
    data = _lag1_series(n=160)
    n = len(data["x"])
    regimes = causal.two_regime_half_split(n)
    assert len(regimes) == n
    assert set(regimes) == {0, 1}
    summary = causal.discover_rpcmci(
        data=data,
        regimes=regimes,
        max_lag=1,
        alpha=0.2,
        fdr=False,
        seed=1,
    )
    assert len(summary.regime_ids) >= 1
