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


def test_counterfactual_ite_returns_unit_effects():
    names, cols, edges = _gcm_linear()
    result = causal.counterfactual_ite(
        names, cols, edges, "t", "y", 1.0, 0.0, seed=1
    )
    assert result.n_units == len(cols[0])
    assert result.unit_effects.shape == (result.n_units,)
    assert np.isclose(result.unit_effects.mean(), result.mean_ite, rtol=1e-9)
    # Structural β=1.5; rtol=0.15 is stable on this fixture (was 0.2).
    assert np.isclose(result.mean_ite, 1.5, rtol=0.15)


def test_sample_do_returns_draws():
    names, cols, edges = _gcm_linear(n=80)
    n_draws = 50
    result = causal.sample_do(
        names, cols, edges, "t", 1.0, n_draws, seed=2
    )
    assert result.n_draws == n_draws
    assert result.draws.shape == (result.n_nodes, n_draws)
    means = result.draws.mean(axis=1)
    assert np.allclose(means, result.column_means, rtol=1e-9)


def test_sample_interventional_distribution():
    names, cols, edges = _gcm_linear(n=80)
    n_draws = 40
    result = causal.sample_interventional_distribution(
        names, cols, edges, "t", 1.0, n_draws, outcome="y", seed=2
    )
    assert result.n_draws == n_draws
    assert result.draws.shape == (result.n_nodes, n_draws)


def test_attribute_path_specific():
    rng = np.random.default_rng(4)
    n = 60
    t = rng.normal(size=n)
    m = 0.8 * t + rng.normal(scale=0.1, size=n)
    y = 0.6 * m + 0.2 * t + rng.normal(scale=0.1, size=n)
    names = ["t", "m", "y"]
    cols = [t, m, y]
    edges = [("t", "m"), ("m", "y"), ("t", "y")]
    total, paths = causal.attribute_path_specific(
        names, cols, edges, "t", "y", path_nodes=["m"], seed=1
    )
    assert isinstance(total, float)
    assert paths
    assert all(isinstance(p, list) and isinstance(c, float) for p, c in paths)
    mediated = next(
        (c for p, c in paths if p == ["t", "m", "y"] or (len(p) == 3 and p[1] == "m")),
        None,
    )
    assert mediated is not None, f"expected t→m→y path contribution, got {paths}"
    # Linear SEM path product 0.8×0.6 = 0.48 (MonteCarlo tolerance).
    assert abs(mediated - 0.48) < 0.25
