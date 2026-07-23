"""Typed exception mapping at the Python boundary."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def test_unknown_ci_raises_compile_error():
    n = 40
    x = np.linspace(0.0, 1.0, n)
    y = x + 0.01
    with pytest.raises(antecedent.CausalCompileError):
        antecedent.discover_pcmci(["x", "y"], [x, y], max_lag=1, ci="not_a_real_ci", seed=1)


def test_unknown_edge_variable_raises_data_error():
    n = 30
    t = np.zeros(n)
    y = np.ones(n)
    with pytest.raises(antecedent.CausalDataError):
        antecedent.analyze(
            {"t": t, "y": y},
            graph=[("missing", "y")],
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            refute=False,
            bootstrap=0,
            seed=1,
        )


def test_column_length_mismatch_is_value_error():
    with pytest.raises(ValueError, match="column length"):
        antecedent.load_float64_columns(
            ["a", "b"],
            [np.array([1.0, 2.0]), np.array([1.0])],
        )


def test_exceptions_subclass_causal_error():
    assert issubclass(antecedent.CausalCompileError, antecedent.CausalError)
    assert issubclass(antecedent.CausalDataError, antecedent.CausalError)
    assert issubclass(antecedent.CausalDiscoveryError, antecedent.CausalError)
    assert issubclass(antecedent.CausalAttributionError, antecedent.CausalError)
    assert issubclass(antecedent.CausalDesignError, antecedent.CausalError)
    assert issubclass(antecedent.CausalStateError, antecedent.CausalError)


def test_causal_state_append_smoke():
    version, stale = antecedent.state.causal_state_append(n_appends=2)
    assert version >= 1
    assert stale >= 0
