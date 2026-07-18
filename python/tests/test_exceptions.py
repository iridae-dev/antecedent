"""Typed exception mapping at the Python boundary."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("causal")
import causal


def test_unknown_ci_raises_compile_error():
    n = 40
    x = np.linspace(0.0, 1.0, n)
    y = x + 0.01
    with pytest.raises(causal.CausalCompileError):
        causal.discover_pcmci(["x", "y"], [x, y], max_lag=1, ci="not_a_real_ci", seed=1)


def test_unknown_edge_variable_raises_data_error():
    n = 30
    t = np.zeros(n)
    y = np.ones(n)
    with pytest.raises(causal.CausalDataError):
        causal.analyze(
            {"t": t, "y": y},
            graph=[("missing", "y")],
            query=causal.AverageEffect(treatment="t", outcome="y"),
            refute=False,
            bootstrap=0,
            seed=1,
        )


def test_column_length_mismatch_is_value_error():
    with pytest.raises(ValueError, match="column length"):
        causal.load_float64_columns(
            ["a", "b"],
            [np.array([1.0, 2.0]), np.array([1.0])],
        )


def test_exceptions_subclass_causal_error():
    assert issubclass(causal.CausalCompileError, causal.CausalError)
    assert issubclass(causal.CausalDataError, causal.CausalError)
    assert issubclass(causal.CausalDiscoveryError, causal.CausalError)
    assert issubclass(causal.CausalAttributionError, causal.CausalError)
    assert issubclass(causal.CausalDesignError, causal.CausalError)
    assert issubclass(causal.CausalStateError, causal.CausalError)


def test_causal_state_append_smoke():
    version, stale = causal.state.causal_state_append(n_appends=2)
    assert version >= 1
    assert stale >= 0
