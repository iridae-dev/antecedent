"""Independent consumers keep scientific receipts and the caller's execution scope."""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent import _native
from antecedent import joint_law_propagation as jl
from antecedent._artifact_protocol import LoadContext
from antecedent.execution import CachePolicy, Execution
from antecedent.outcomes import Outcome, require_success

from test_joint_law_propagation import _artifact, _draws


def _producer():
    artifact = _artifact("canonical-load", np.array([0.0, 1.0]), np.array([2.0, 4.0]))
    x, y = _draws(artifact)
    return jl.aligned_functional("product", x, y, "product")


def test_canonical_load_returns_independent_outcome_with_actual_receipt_and_no_study():
    produced = _producer()
    artifact = produced.export()
    execution = Execution(cache=CachePolicy.disabled())
    loaded = ant.load(
        artifact,
        expected=artifact.expectation,
        context=LoadContext(execution=execution),
        execution=execution,
    )
    assert isinstance(loaded, Outcome)
    assert loaded.study is None
    assert loaded.answer == produced.answer
    reading = require_success(loaded)
    assert reading.acceptance.verified
    assert reading.acceptance.replayable
    assert reading.acceptance.identity_bound
    assert bytes(loaded.export()) == bytes(artifact)


def test_canonical_load_rejects_conflicting_execution_owners_before_parsing():
    first, second = Execution(), Execution()
    with pytest.raises(TypeError, match="one Execution"):
        ant.load(b"not an artifact", execution=first, context=LoadContext(execution=second))
    with pytest.raises(TypeError, match="Execution"):
        LoadContext(execution=object())


def test_canonical_load_carries_cancellation_into_actual_native_replay_and_restores_scope():
    produced = _producer()
    artifact = produced.export()
    token = _native.CancellationToken()
    token.cancel()
    policy = Execution(cancel=token, cache=CachePolicy.disabled())
    # This is an operational interruption from the actual independent native
    # replay, so it raises rather than becoming a scientific Refused answer.
    with pytest.raises(jl.JointLawPropagationRefusal) as stopped:
        ant.load(artifact, context=LoadContext(execution=policy))
    assert stopped.value.reason_code == "cancelled_no_claim"
    assert ant.load(artifact).answer == produced.answer
