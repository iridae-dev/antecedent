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
    """Inline discovery on pooled panel data recovers the lag-one effect.

    At ``alpha=0.2`` pooled PCMCI keeps a lag-one self edge on ``x``, so the
    treatment's ancestry is unbounded and unfolding cannot certify. The single-step
    pulse is then identified by adjusting for the treatment's parents (``x[t-2]``)
    and the pooled fit recovers the generating effect 0.55; the exported contract
    names the ``temporal.parent_adjustment`` derivation.
    """
    panel = antecedent.data.panel([_lag1_unit(seed=3), _lag1_unit(seed=4), _lag1_unit(seed=5)])
    result = antecedent.analyze(
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
    assert result.answer.kind == "point"
    assert result.answer.value == pytest.approx(0.55, abs=0.05)
    assert result.identification.adjustment_set == ["x"]
    product = (
        antecedent.load(result.export()).inspect().to_dict()["contract"]["identification_product"]
    )
    assert product["derivation_rules"][0] == "temporal.parent_adjustment"


def test_panel_pooled_rejects_rpcmci():
    panel = antecedent.data.panel([_lag1_unit(seed=3), _lag1_unit(seed=4)])
    with pytest.raises(
        antecedent.errors.CausalUnsupportedError, match="single observation sequence"
    ):
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
