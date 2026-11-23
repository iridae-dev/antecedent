"""The callback-failure capacity is an Execution-owned limit with a native round trip."""

from __future__ import annotations

import antecedent as ac
import numpy as np
import pytest
from antecedent import _native
from antecedent.execution import native_controls


def _columns(n: int = 120, k: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    names = [f"v{i}" for i in range(k)]
    return names, [rng.normal(size=n) for _ in names]


def test_the_capacity_defaults_to_the_native_default():
    assert ac.Execution().callback_failure_capacity == 256


@pytest.mark.parametrize("bad", [-1, True, 1.5, "8", 2**64])
def test_the_capacity_must_be_a_nonnegative_integer(bad):
    with pytest.raises((ac.errors.CausalTypeError, ac.errors.CausalValueError)):
        ac.Execution(callback_failure_capacity=bad)


def test_native_controls_pass_the_capacity_under_the_named_keyword():
    execution = ac.Execution(callback_failure_capacity=7)
    controls = native_controls(execution, callback_failure_capacity="capacity")
    assert controls == {"capacity": 7}
    assert native_controls(execution) == {}


@pytest.mark.parametrize("capacity", [0, 1, 256])
def test_a_tiny_capacity_still_handles_a_callback_failure(capacity):
    names, columns = _columns()
    execution = ac.Execution(callback_failure_capacity=capacity)

    def boom(_columns, _queries):
        raise ValueError("ci exploded")

    with pytest.raises(_native.CausalError) as raised:
        execution._call(_native.discover_pc, names, columns, ci=boom, fdr=False, seed=1)
    if capacity >= 1:
        cause = raised.value.__cause__
        assert isinstance(cause, ValueError)
        assert str(cause) == "ci exploded"
