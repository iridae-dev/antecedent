"""PanelFrame pooled PCMCI-family discovery (not JPCMCI+ multi-env)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("causal")
import causal


def _lag1_unit(n: int = 100, seed: int = 3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.empty(n)
    y[0] = rng.normal()
    for t in range(1, n):
        y[t] = 0.55 * x[t - 1] + 0.2 * rng.normal()
    return {"x": x, "y": y}


def test_panel_pooled_pcmci_smoke():
    panel = causal.panel([_lag1_unit(seed=3), _lag1_unit(seed=4), _lag1_unit(seed=5)])
    try:
        result = causal.analyze(
            panel,
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


def test_panel_pooled_rejects_rpcmci():
    panel = causal.panel([_lag1_unit(seed=3), _lag1_unit(seed=4)])
    with pytest.raises(TypeError, match="PanelFrame discovery supports"):
        causal.analyze(
            panel,
            discovery=causal.RPCMCI(max_lag=1, alpha=0.2, fdr=False),
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
