"""Validation / refute suite and discovery stability surface."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def _confounded(n: int = 400, seed: int = 11):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = 2.0 * t + z + 0.3 * rng.normal(size=n)
    return {"z": z, "t": t, "y": y}, [("z", "t"), ("z", "y"), ("t", "y")]


def test_analyze_refute_full_runs():
    data, edges = _confounded()
    result = causal.analyze(
        data,
        graph=edges,
        query=causal.AverageEffect(treatment="t", outcome="y"),
        refute="full",
        bootstrap=5,
        seed=1,
    )
    # Full suite includes more than placebo+rcc when applicable.
    assert result.validation.ran is True
    assert result.validation.count >= 2


def test_validate_pcmci_block_bootstrap_smoke():
    rng = np.random.default_rng(2)
    n = 80
    x = rng.normal(size=n)
    y = np.empty(n)
    y[0] = rng.normal()
    for t in range(1, n):
        y[t] = 0.5 * x[t - 1] + 0.2 * rng.normal()
    report = causal.validate_pcmci_block_bootstrap(
        {"x": x, "y": y},
        max_lag=1,
        alpha=0.2,
        fdr=False,
        replicates=5,
        block_size=10,
        seed=1,
    )
    assert report["replicates"] == 5
    assert report["block_size"] == 10
    assert isinstance(report["frequencies"], list)


def test_validate_synthetic_null_calibration_smoke():
    report = causal.validate_synthetic_null_calibration(
        max_lag=1,
        alpha=0.2,
        fdr=False,
        n_sim=5,
        n_obs=60,
        n_vars=2,
        seed=3,
    )
    assert report["n_sim"] == 5
    assert np.isfinite(report["empirical_fpr"])
