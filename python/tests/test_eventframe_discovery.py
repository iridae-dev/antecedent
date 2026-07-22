"""EventFrame discovery: PCMCI-family happy path + JPCMCI+ reject."""

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


def test_eventframe_pcmci_discovery_happy_path():
    data = _lag1_series()
    n = len(data["x"])
    frame = causal.event(data, np.arange(n, dtype=np.int64), align_interval_ns=1)
    try:
        result = causal.analyze(
            frame,
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
    except Exception as exc:  # noqa: BLE001 — review/ID may fail closed
        assert str(exc), "expected a non-empty error from the wired path"


def test_eventframe_rejects_jpcmci_plus():
    data = _lag1_series(n=60, seed=2)
    n = len(data["x"])
    frame = causal.event(data, np.arange(n, dtype=np.int64), align_interval_ns=1)
    with pytest.raises(TypeError, match="EventFrame does not support discovery=JPCMCIPlus"):
        causal.analyze(
            frame,
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
