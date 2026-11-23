"""slow-path Python callback extensibility."""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent._native import (
    discover_pcmci,
    sample_do,
)
from antecedent.outcomes import require_success

from _native_ate import native_ate


def _indep_ci(columns, queries):
    """Always-independent batch CI (empty / sparse discovery)."""
    return [(0.0, 1.0) for _ in queries]


def test_custom_ci_independence_sparse_discovery():
    rng = np.random.default_rng(0)
    n = 200
    names = ["a", "b", "c"]
    columns = [rng.normal(size=n) for _ in names]
    # Strong lagged dependence a(t-1) -> b(t) in the data, but the callback reports independence.
    columns[1][1:] = columns[0][:-1] + 0.01 * rng.normal(size=n - 1)
    kwargs = {"max_lag": 1, "alpha": 0.05, "fdr": False, "seed": 1, "threads": 4}
    # Control: the native test finds the planted link, so zero links below is the callback's doing.
    native = discover_pcmci(names, columns, **kwargs)
    assert native.links_retained >= 1
    assert len(native.links) >= 1
    batches = []

    def recording_ci(cols, queries):
        batches.append(len(queries))
        return _indep_ci(cols, queries)

    result = discover_pcmci(names, columns, ci=recording_ci, **kwargs)
    assert result.ci_name == "python.callback"
    assert sum(batches) > 0  # the callback, not the native test, decided every query
    assert result.links_retained == 0
    assert len(result.links) == 0


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
    # The intervention itself is untouched by the wrapper on y: t is fixed at 1.0 exactly.
    assert abs(out.column_means[names.index("t")] - 1.0) < 1e-12


def test_evaluate_decision_utility_callback():
    def util(actions, outcomes):
        # Prefer action index 1 (value 1.0): utility = action * mean(outcome)
        a = np.asarray(actions, dtype=np.float64)
        o = np.asarray(outcomes, dtype=np.float64)
        return np.outer(a, o).ravel()

    eu = antecedent.design.evaluate_decision([0.0, 1.0], [2.0, 4.0], util)
    assert eu.chosen_action == 1
    assert eu.expected_utility == pytest.approx(3.0)  # 1 * mean([2,4])
    assert eu.posterior_regret == pytest.approx(0.0)


def test_evaluate_decision_regret_is_expected_value_of_perfect_information():
    def util(actions, outcomes):
        return np.outer(np.asarray(actions), np.asarray(outcomes)).ravel()

    # Bayes action a=2 (EU 2); perfect information earns ((−1) + 6) / 2 = 2.5.
    ev = antecedent.design.evaluate_decision([1.0, 2.0], [-1.0, 3.0], util)
    assert ev.chosen_action == 1
    assert ev.expected_utility == pytest.approx(2.0)
    assert ev.posterior_regret == pytest.approx(0.5, abs=1e-15)

    with pytest.raises(antecedent.errors.CausalDesignError, match="draw"):
        antecedent.design.evaluate_decision([1.0, 2.0], [], util)


def test_evaluate_decision_utility_failure_is_typed():
    def boom(_actions, _outcomes):
        raise RuntimeError("utility exploded")

    with pytest.raises(antecedent.errors.CausalDesignError, match="callback utility"):
        antecedent.design.evaluate_decision([0.0, 1.0], [2.0, 4.0], boom)


def test_custom_validator_on_native_ate():
    rng = np.random.default_rng(3)
    n = 120
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = 1.5 * t + 0.5 * z + rng.normal(size=n)
    names = ["z", "t", "y"]
    columns = [z, t, y]
    edges = [("z", "t"), ("z", "y"), ("t", "y")]

    seen = []

    def always_fail(*, ate, se_analytic, method, adjustment_set):
        seen.append((ate, se_analytic, method, adjustment_set))
        return {
            "passed": False,
            "refuted_ate": 0.0,
            "comparison": 0.0,
            "failure_condition": "custom fail",
        }

    result = native_ate(
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
    # The validator received the estimate: the OLS coefficient of t in y ~ 1 + t + z.
    design = np.column_stack([np.ones(n), t, z])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    assert len(seen) >= 1
    assert seen[0][0] == pytest.approx(beta[1], abs=1e-9)
    assert list(seen[0][3]) == ["z"]
    assert result.refutation_ran
    assert result.refutation_count >= 1
    assert result.refutation_passed is False
    assert result.worker_threads == 0
    assert result.expected_python_crossings >= 2  # base + validator mark
    assert any("exec.python_callback_serial" in d for d in result.diagnostics)


@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_decision_utility_preserves_original_callback_exception(exception_type):
    original = exception_type("utility callback identity")

    def failing_utility(_actions, _outcomes):
        raise original

    expected = (
        antecedent.errors.CausalDesignError if exception_type is RuntimeError else exception_type
    )
    with pytest.raises(expected) as raised:
        antecedent.design.evaluate_decision([0.0, 1.0], [2.0, 4.0], failing_utility)
    if exception_type is RuntimeError:
        assert raised.value.__cause__ is original
    else:
        assert raised.value is original
    assert original.__traceback__ is not None


@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_custom_validator_failures_propagate_the_original_exception(exception_type):
    rng = np.random.default_rng(932)
    n = 80
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = t + 0.5 * z + rng.normal(size=n)
    original = exception_type("validator callback identity")
    invoked = []

    def failing_validator(**_kwargs):
        invoked.append(True)
        raise original

    def estimate():
        return native_ate(
            ["z", "t", "y"],
            [z, t, y],
            [("z", "t"), ("z", "y"), ("t", "y")],
            "t",
            "y",
            validators=[failing_validator],
            refute=False,
            bootstrap=0,
            threads=4,
        )

    # A callback failure is operational, never a refuted result: every exception
    # class propagates as the original object.
    with pytest.raises(exception_type) as raised:
        estimate()
    assert raised.value is original
    assert invoked
    assert original.__traceback__ is not None


@pytest.mark.parametrize("sink", ["on_progress", "on_stage"])
@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_execution_observer_failures_propagate_the_original_exception(sink, exception_type):
    from antecedent._native import PreparedAnalysis

    rng = np.random.default_rng(934)
    n = 80
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = t + 0.5 * z + rng.normal(size=n)
    names = ["z", "t", "y"]
    columns = [z, t, y]
    prepared = PreparedAnalysis.prepare(
        names,
        columns,
        [("z", "t"), ("z", "y"), ("t", "y")],
        "t",
        "y",
        options={"refute": False, "bootstrap": 40},
        threads=4,
    )
    original = exception_type(f"{sink} callback identity")
    invoked = []

    def failing_observer(*_args):
        invoked.append(True)
        raise original

    def estimate():
        return require_success(
            prepared.estimate(names, columns, threads=4, **{sink: failing_observer})
        )

    with pytest.raises(exception_type) as raised:
        estimate()
    assert raised.value is original
    assert invoked
    assert original.__traceback__ is not None
