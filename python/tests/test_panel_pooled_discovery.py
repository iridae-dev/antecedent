"""PanelFrame pooled PCMCI-family discovery (not JPCMCI+ multi-env)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def _lag1_unit(n: int = 100, seed: int = 3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.empty(n)
    y[0] = rng.normal()
    for t in range(1, n):
        y[t] = 0.55 * x[t - 1] + 0.2 * rng.normal()
    return {"x": x, "y": y}


def test_panel_pooled_pcmci_smoke():
    """Inline discovery on pooled panel data fails closed with a documented refusal.

    Two documented gates can fire on this path: panel analysis wants a supplied
    `TemporalDag` rather than an inline-discovered graph, and temporal unfolding
    refuses to certify backdoor identification once confounder ancestry crosses its
    history cap. Which one is reached first depends on unit count, so both messages
    are accepted — what is pinned is that the call fails closed with one of them
    rather than returning a number.
    """
    panel = antecedent.data.panel([_lag1_unit(seed=3), _lag1_unit(seed=4), _lag1_unit(seed=5)])
    with pytest.raises(
        (antecedent.errors.CausalCompileError, antecedent.errors.CausalIdentifyError),
        match="panel data supports only a supplied TemporalDag|not certified",
    ):
        antecedent.analyze(
            panel,
            discovery=antecedent.discovery.PCMCI(max_lag=1, alpha=0.2, fdr=False),
            query=antecedent.PulseEffect(
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


def test_panel_pooled_rejects_rpcmci():
    panel = antecedent.data.panel([_lag1_unit(seed=3), _lag1_unit(seed=4)])
    with pytest.raises(TypeError, match="PanelFrame discovery supports"):
        antecedent.analyze(
            panel,
            discovery=antecedent.discovery.RPCMCI(max_lag=1, alpha=0.2, fdr=False),
            query=antecedent.PulseEffect(
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
