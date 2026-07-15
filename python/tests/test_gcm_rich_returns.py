"""Rich GCM returns: unit ITEs and interventional draws."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("causal")
import causal


def _gcm_linear(n: int = 200, seed: int = 3):
    rng = np.random.default_rng(seed)
    t = (rng.random(n) > 0.5).astype(np.float64)
    z = rng.normal(size=n)
    y = 1.5 * t + 0.5 * z + rng.normal(scale=0.1, size=n)
    names = ["t", "z", "y"]
    cols = [t, z, y]
    edges = [("t", "y"), ("z", "y"), ("z", "t")]
    return names, cols, edges


def test_gcm_counterfactual_ite_returns_unit_effects():
    names, cols, edges = _gcm_linear()
    result = causal.gcm_counterfactual_ite(
        names, cols, edges, "t", "y", active=1.0, control=0.0, seed=1
    )
    assert result.n_units == len(cols[0])
    assert result.unit_effects.shape == (result.n_units,)
    assert np.isclose(result.unit_effects.mean(), result.mean_ite, rtol=1e-9)


def test_gcm_sample_do_returns_draws():
    names, cols, edges = _gcm_linear(n=80)
    n_draws = 50
    result = causal.gcm_sample_do(
        names, cols, edges, "t", do_value=1.0, n_draws=n_draws, seed=2
    )
    assert result.n_draws == n_draws
    assert result.draws.shape == (result.n_nodes, n_draws)
    means = result.draws.mean(axis=1)
    assert np.allclose(means, result.column_means, rtol=1e-9)
