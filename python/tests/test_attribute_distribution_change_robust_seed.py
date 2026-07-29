"""Regression test for `attribute_distribution_change_robust`'s `n_samples`/`seed` kwargs.

Before the fix, the Rust binding for this function discarded `n_samples` entirely
(`let _ = n_samples;`) and hardcoded `ShapleyConfig::monte_carlo(200)` with no
`.with_seed(seed)`, so the underlying Monte Carlo Shapley estimate ignored both
knobs. Its sibling `attribute_distribution_change` threads them through correctly;
this test would fail against the old, broken binding and passes against the fix.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def _mechanism_shift_fixture(n: int = 400, seed: int = 11):
    """Five parents of `y`, whose structural coefficients change halfway through
    the data — a real mechanism shift for `distribution_change_robust` to attribute
    across five Shapley players (5! = 120 permutations), so a handful of Monte Carlo
    samples clearly cannot exhaust the permutation space and must depend on `seed`.
    """
    rng = np.random.default_rng(seed)
    xs = rng.normal(size=(n, 5))
    half = n // 2
    baseline_coefs = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    comparison_coefs = np.array([3.0, 0.2, 2.0, 0.1, 4.0])
    y = np.empty(n)
    y[:half] = xs[:half] @ baseline_coefs + rng.normal(scale=0.05, size=half)
    y[half:] = xs[half:] @ comparison_coefs + rng.normal(scale=0.05, size=n - half)
    names = ["x1", "x2", "x3", "x4", "x5", "y"]
    cols = [xs[:, i] for i in range(5)] + [y]
    edges = [(f"x{i}", "y") for i in range(1, 6)]
    return names, cols, edges, half


def _run(names, cols, edges, half, n, *, n_samples, seed):
    return antecedent.attribution.attribute_distribution_change_robust(
        names,
        cols,
        edges,
        "y",
        baseline_start=0,
        baseline_end=half,
        comparison_start=half,
        comparison_end=n,
        n_samples=n_samples,
        seed=seed,
    )


def test_distribution_change_robust_seed_is_reproducible_and_changes_output():
    names, cols, edges, half = _mechanism_shift_fixture()
    n = len(cols[0])

    same_a = _run(names, cols, edges, half, n, n_samples=3, seed=1)
    same_b = _run(names, cols, edges, half, n, n_samples=3, seed=1)
    diff_seed = _run(names, cols, edges, half, n, n_samples=3, seed=2)

    # Same seed + same n_samples -> byte-identical contributions (seeding is real).
    assert same_a.contributions == same_b.contributions
    assert same_a.total_change == same_b.total_change

    # Different seed, same n_samples -> different per-component Shapley estimates.
    # (Before the fix, `seed` never reached `ShapleyConfig`, so this would have been
    # equal to `same_a.contributions`.)
    assert diff_seed.contributions != same_a.contributions


def test_distribution_change_robust_n_samples_has_observable_effect():
    names, cols, edges, half = _mechanism_shift_fixture()
    n = len(cols[0])

    few_samples = _run(names, cols, edges, half, n, n_samples=3, seed=1)
    many_samples = _run(names, cols, edges, half, n, n_samples=50, seed=1)

    # Different n_samples, same seed -> different per-component Shapley estimates,
    # since each Monte Carlo sample consumes the seeded RNG stream. Before the fix,
    # `n_samples` was discarded entirely (`let _ = n_samples;`), so both calls used
    # the same hardcoded 200-sample config and would have produced identical output.
    assert few_samples.contributions != many_samples.contributions
