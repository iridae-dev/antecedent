"""Explicit execution policies survive detached work and restore nested scopes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import antecedent as ant
import numpy as np
import pytest
from antecedent.execution import CachePolicy, Execution, ResourceLimits


def _prepare(execution=None):
    t = np.tile([0.0, 1.0], 32)
    z = np.repeat(np.linspace(-1.0, 1.0, len(t) // 2), 2)
    return ant.prepare(
        {"t": t, "y": 2 * t + z, "z": z},
        graph=[("t", "y"), ("z", "y")],
        query=ant.AverageEffect("t", "y"),
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
        execution=execution,
    )


def test_memory_policy_reaches_native_prepare_and_restores_after_failure():
    tiny = Execution(limits=ResourceLimits(memory_bytes=0), cache=CachePolicy.disabled())
    roomy = Execution(limits=ResourceLimits(memory_bytes=1_000_000))
    with pytest.raises(ant.errors.CausalResourceError, match="admission wait"):
        _prepare(tiny)
    study = _prepare(roomy)
    assert ant.estimate(study).answer.value == pytest.approx(2.0)
    assert ant.estimate(_prepare()).answer.value == pytest.approx(2.0)


def test_nested_policy_restores_the_outer_caller_after_exception():
    outer = Execution(limits=ResourceLimits(memory_bytes=1_000_000))
    inner = Execution(limits=ResourceLimits(memory_bytes=0))

    def operation():
        with pytest.raises(ant.errors.CausalResourceError, match="admission wait"):
            _prepare(inner)
        return ant.estimate(_prepare(outer)).answer.value

    assert outer._call(operation) == pytest.approx(2.0)


def test_concurrent_callers_do_not_inherit_each_others_resource_policy():
    barrier = Barrier(2)

    def run(memory):
        policy = Execution(limits=ResourceLimits(memory_bytes=memory))

        def operation():
            barrier.wait(timeout=10)
            return ant.estimate(_prepare(policy)).answer.value

        try:
            return policy._call(operation)
        except ant.errors.CausalResourceError:
            return "resource_refused"

    with ThreadPoolExecutor(max_workers=2) as pool:
        tiny = pool.submit(run, 0)
        roomy = pool.submit(run, 1_000_000)
        assert tiny.result(timeout=20) == "resource_refused"
        assert roomy.result(timeout=20) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "kwargs", [{"seed": True}, {"threads": False}, {"threads": 0}, {"threads": 2**32}, {"seed": -1}]
)
def test_execution_rejects_ambiguous_or_out_of_range_integers(kwargs):
    with pytest.raises((TypeError, ValueError), match="integer|range"):
        Execution(**kwargs)
