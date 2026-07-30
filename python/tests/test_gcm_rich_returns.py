"""Rich GCM returns: unit ITEs and interventional draws."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def _gcm_linear(n: int = 200, seed: int = 3):
    rng = np.random.default_rng(seed)
    t = (rng.random(n) > 0.5).astype(np.float64)
    z = rng.normal(size=n)
    y = 1.5 * t + 0.5 * z + rng.normal(scale=0.1, size=n)
    names = ["t", "z", "y"]
    cols = [t, z, y]
    edges = [("t", "y"), ("z", "y"), ("z", "t")]
    return names, cols, edges


def _linear_chain(n: int = 800, seed: int = 5, coef: float = 2.0):
    """A single-parent chain `x -> y` with a known structural coefficient."""
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=3.0, scale=1.0, size=n)
    y = 1.0 + coef * x + rng.normal(scale=0.1, size=n)
    names = ["x", "y"]
    cols = [x, y]
    edges = [("x", "y")]
    return names, cols, edges, coef


def test_counterfactual_ite_returns_unit_effects():
    names, cols, edges = _gcm_linear()
    result = antecedent.counterfactual.counterfactual_ite(
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
    result = antecedent.model.sample_do(names, cols, edges, "t", 1.0, n_draws, seed=2)
    assert result.n_draws == n_draws
    assert result.draws.shape == (result.n_nodes, n_draws)
    means = result.draws.mean(axis=1)
    assert np.allclose(means, result.column_means, rtol=1e-9)


def test_sample_interventional_distribution():
    names, cols, edges = _gcm_linear(n=80)
    n_draws = 40
    result = antecedent.model.sample_interventional_distribution(
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
    result = antecedent.attribution.attribute_path_specific(
        names, cols, edges, "t", "y", path_nodes=["m"], seed=1
    )
    assert isinstance(result.total_change, float)
    paths = result.path_breakdown
    assert paths
    assert all(isinstance(p, list) and isinstance(c, float) for p, c in paths)
    mediated = next(
        (c for p, c in paths if p == ["t", "m", "y"] or (len(p) == 3 and p[1] == "m")),
        None,
    )
    assert mediated is not None, f"expected t→m→y path contribution, got {paths}"
    # Linear SEM path product 0.8×0.6 = 0.48 (MonteCarlo tolerance).
    assert abs(mediated - 0.48) < 0.25


def test_fit_gcm_oo_sample_do():
    names, cols, edges = _gcm_linear(n=80)
    gcm = antecedent.model.fit_gcm(names, cols, edges)
    result = gcm.sample_do({"t": 1.0}, 40, seed=2)
    assert result.n_draws == 40
    assert result.draws.shape == (result.n_nodes, 40)
    ite = gcm.counterfactual_ite("t", "y", 1.0, 0.0, seed=1)
    assert ite.n_units == len(cols[0])
    assert np.isclose(ite.mean_ite, 1.5, rtol=0.15)


def test_fit_gcm_oo_sample_do_shift_moves_outcome_by_coefficient_times_delta():
    """`shifts={x: delta}` adds delta to x's structural assignment (`do(x := x + delta)`),
    moving the downstream mean by `coefficient * delta` relative to baseline. A hard
    `interventions={x: v}` instead pins x to v outright — the two are observably
    different, which is the whole point of exposing shift interventions.
    """
    names, cols, edges, coef = _linear_chain()
    gcm = antecedent.model.fit_gcm(names, cols, edges)
    y_idx = gcm.names.index("y")
    n_draws = 4000
    delta = 3.0

    baseline = gcm.sample_do({}, n_draws, seed=7)
    shifted = gcm.sample_do({}, n_draws, seed=7, shifts={"x": delta})
    pinned = gcm.sample_do({"x": delta}, n_draws, seed=7)

    baseline_mean = baseline.column_means[y_idx]
    shifted_mean = shifted.column_means[y_idx]
    pinned_mean = pinned.column_means[y_idx]

    # Shift moves the outcome mean by coefficient * delta relative to baseline.
    assert np.isclose(shifted_mean - baseline_mean, coef * delta, atol=0.5)
    # Hard set pins the outcome regardless of baseline x; on this fixture that
    # lands far from what the shift produces.
    assert abs(pinned_mean - shifted_mean) > 1.0


def test_fit_gcm_oo_sample_do_rejects_variable_in_both_interventions_and_shifts():
    names, cols, edges, _coef = _linear_chain(n=100)
    gcm = antecedent.model.fit_gcm(names, cols, edges)
    with pytest.raises(antecedent.CausalError):
        gcm.sample_do({"x": 1.0}, 10, shifts={"x": 2.0}, seed=1)


def test_sample_do_free_function_shift():
    names, cols, edges, coef = _linear_chain()
    n_draws = 4000
    delta = 3.0
    baseline = antecedent.model.sample_do(names, cols, edges, "x", 0.0, n_draws, seed=7, shift=True)
    shifted = antecedent.model.sample_do(
        names, cols, edges, "x", delta, n_draws, seed=7, shift=True
    )
    y_idx = names.index("y")
    assert np.isclose(
        shifted.column_means[y_idx] - baseline.column_means[y_idx], coef * delta, atol=0.5
    )
