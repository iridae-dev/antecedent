""" slow-path Python callback extensibility."""

from __future__ import annotations

import numpy as np
import pytest

import causal
from causal._native import (
    analyze_ate,
    discover_pcmci,
    sample_do,
)


def _indep_ci(columns, queries):
    """Always-independent batch CI (empty / sparse discovery)."""
    return [(0.0, 1.0) for _ in queries]


def test_custom_ci_independence_sparse_discovery():
    rng = np.random.default_rng(0)
    n = 80
    names = ["a", "b", "c"]
    columns = [rng.normal(size=n) for _ in names]
    # Strong dependence in data, but callback reports independence.
    columns[1] = columns[0] + 0.01 * rng.normal(size=n)
    result = discover_pcmci(
        names,
        columns,
        max_lag=1,
        alpha=0.05,
        fdr=False,
        seed=1,
        ci=_indep_ci,
        threads=4,
    )
    assert result.ci_name == "python.callback"
    assert result.links_retained == 0 or len(result.links) == 0


class _ConstMech:
    def sample_noise(self, n: int) -> np.ndarray:
        return np.zeros(n, dtype=np.float64)

    def evaluate(self, parents, noise: np.ndarray) -> np.ndarray:
        return np.full(noise.shape, 42.0, dtype=np.float64)


def test_mechanism_wrapper_sample_do():
    rng = np.random.default_rng(1)
    n = 60
    z = rng.normal(size=n)
    t = z + rng.normal(size=n)
    y = t + rng.normal(size=n)
    names = ["z", "t", "y"]
    columns = [z, t, y]
    edges = [("z", "t"), ("t", "y")]
    out = sample_do(
        names,
        columns,
        edges,
        "t",
        1.0,
        40,
        seed=2,
        mechanism_wrappers={"y": _ConstMech()},
    )
    # Column means for y under do(t) should be ~42 from the wrapper.
    y_idx = names.index("y")
    assert abs(out.column_means[y_idx] - 42.0) < 1e-9


def test_evaluate_decision_utility_callback():
    def util(actions, outcomes):
        # Prefer action index 1 (value 1.0): utility = action * mean(outcome)
        a = np.asarray(actions, dtype=np.float64)
        o = np.asarray(outcomes, dtype=np.float64)
        return np.outer(a, o).ravel()

    eu = causal.evaluate_decision([0.0, 1.0], [2.0, 4.0], util)
    assert eu.chosen_action == 1
    assert eu.expected_utility == pytest.approx(3.0)  # 1 * mean([2,4])
    assert eu.posterior_regret == pytest.approx(0.0)


def test_custom_validator_on_analyze_ate():
    rng = np.random.default_rng(3)
    n = 120
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = 1.5 * t + 0.5 * z + rng.normal(size=n)
    names = ["z", "t", "y"]
    columns = [z, t, y]
    edges = [("z", "t"), ("z", "y"), ("t", "y")]

    def always_fail(*, ate, se_analytic, method, adjustment_set):
        return {
            "passed": False,
            "refuted_ate": 0.0,
            "comparison": 0.0,
            "failure_condition": "custom fail",
        }

    result = analyze_ate(
        names,
        columns,
        edges,
        "t",
        "y",
        refute=False,
        validators=[always_fail],
        bootstrap=0,
        seed=4,
        threads=4,
    )
    assert result.refutation_ran
    assert result.refutation_count >= 1
    assert result.refutation_passed is False
    assert result.worker_threads == 0
    assert result.expected_python_crossings >= 2  # base + validator mark
    assert any("exec.python_callback_serial" in d for d in result.diagnostics)
