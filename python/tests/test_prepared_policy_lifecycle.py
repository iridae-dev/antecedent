# SPDX-License-Identifier: MIT OR Apache-2.0
"""Policy preparation owns and checks inputs before its first statistical run."""

import antecedent as ac
import numpy as np
import pytest
from antecedent.errors import CausalResourceError, CausalUnsupportedError
from antecedent.execution import Execution, ResourceLimits
from antecedent.intervention_programs import (
    PolicyProgramValue,
    assign,
    generated,
    transform,
    var,
)


def observations():
    rng = np.random.default_rng(417)
    x = rng.normal(size=160)
    return {"x": x, "y": 2 * x + rng.normal(size=len(x))}


def question():
    return PolicyProgramValue((transform("x", generated() + 1),), "y", 100)


def test_policy_preparation_defers_fit_and_forks_share_actual_native_fit():
    study = ac.prepare(observations(), query=question(), graph=[("x", "y")])
    assert study.inspect().stage == "prepared"
    assert not study._engine._native.fit_materialized
    first = ac.estimate(study)
    assert first.accepted
    assert first.answer.kind == "policy_value"
    assert study._engine._native.fit_materialized
    assert first.study.inspect().stage == "executed"
    assert study.inspect().stage == "prepared"
    second = ac.estimate(study)
    assert second.value.value == first.value.value
    assert first.value.preparation.program_compilations == 1


def test_policy_owned_rows_and_pure_data_branch():
    rows = observations()
    baseline = ac.prepare(rows, query=question(), graph=[("x", "y")])
    source = ac.prepare(rows, query=question(), graph=[("x", "y")])
    rows["y"][:] = 1e8
    assert ac.estimate(source).value.value == ac.estimate(baseline).value.value
    changed = observations()
    changed["y"] += 20
    branch = ac.rebind(source, data=changed)
    assert not branch._engine._native.fit_materialized
    assert (
        ac.estimate(branch).value.value.policy_mean
        > ac.estimate(source).value.value.policy_mean + 10
    )


def test_program_dependency_cycle_is_checked_before_fitting():
    query = PolicyProgramValue((assign("x", var("y")),), "y", 100)
    study = ac.prepare(observations(), query=query, graph=[("x", "y")])
    refused = ac.estimate(study)
    assert not refused.accepted


def test_policy_execution_preserves_progress_callback_exception():
    class ObserverFailed(RuntimeError):
        pass

    error = ObserverFailed("policy observer failed")

    def fail(*_):
        raise error

    study = ac.prepare(
        observations(),
        query=question(),
        graph=[("x", "y")],
        execution=Execution(on_progress=fail),
    )
    with pytest.raises(ObserverFailed) as caught:
        ac.estimate(study)
    assert caught.value is error


def test_policy_fit_resource_limit_remains_an_operational_exception():
    study = ac.prepare(observations(), query=question(), graph=[("x", "y")])
    # Bind a strict run context after input admission, so the fit itself is checked.
    branch = study._engine._branch(Execution(limits=ResourceLimits(memory_bytes=0)))
    with pytest.raises(CausalResourceError):
        branch.estimate()


def test_policy_program_export_does_not_claim_a_saved_value():
    result = ac.estimate(ac.prepare(observations(), query=question(), graph=[("x", "y")]))
    artifact = result.export(quantity="program")
    assert bytes(artifact).startswith(b"ANTPOLICYPROG")
    with pytest.raises(CausalUnsupportedError):
        result.export(quantity="value")


def test_policy_progress_failure_keeps_the_user_traceback():
    import traceback

    original = RuntimeError("observer traceback retained")

    def observer_with_traceback(*_):
        raise original

    study = ac.prepare(
        observations(),
        query=question(),
        graph=[("x", "y")],
        execution=Execution(on_progress=observer_with_traceback),
    )
    with pytest.raises(RuntimeError) as caught:
        ac.estimate(study)
    assert caught.value is original
    frames = traceback.extract_tb(caught.value.__traceback__)
    assert any(frame.name == "observer_with_traceback" for frame in frames)


def test_policy_observer_scientific_error_remains_operational():
    original = CausalUnsupportedError(
        "raised by observer, not by policy licensing",
        reason_code="policy_route_not_licensed",
    )

    def observer_raises_scientific_error(*_):
        raise original

    study = ac.prepare(
        observations(),
        query=question(),
        graph=[("x", "y")],
        execution=Execution(on_progress=observer_raises_scientific_error),
    )
    with pytest.raises(CausalUnsupportedError) as caught:
        ac.estimate(study)
    assert caught.value is original
    assert caught.value._antecedent_operational_callback
