"""Carried population weights compose with exact immutable completed score snapshots."""

from __future__ import annotations

import gc

import antecedent as ant
import numpy as np
import pytest
from antecedent import artifacts
from antecedent.errors import (
    CausalCancelledError,
    CausalResourceError,
    CausalStateError,
    CausalTypeError,
    CausalUnsupportedError,
)
from antecedent.execution import Execution, ResourceLimits
from antecedent.execution_attempt import observe_native_attempts
from antecedent.outcomes import Outcome, require_success
from antecedent.population import replay_retarget


def _draw(seed=1, shift=0.0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=300)
    t = (rng.uniform(size=300) < 1 / (1 + np.exp(-0.8 * z))).astype(float)
    y = (2 + shift * z) * t + z + rng.normal(scale=0.5, size=300)
    return {"t": t, "y": y, "z": z}


def _prepare(data):
    return ant.prepare(
        data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.AverageEffect("t", "y"),
        estimator="aipw",
        inference=ant.Frequentist(bootstrap=0),
        execution=Execution(seed=1, threads=1),
        refute=False,
    )


def _completed(data=None):
    return ant.estimate(_prepare(_draw() if data is None else data))


def _targeted(completed, weights):
    return completed.study.retarget(weights, depends_on=["z"])


def _assert_no_fits(attempt):
    assert attempt.report.is_complete
    for name in ("glm_fit", "adjusted_fit", "learner_fit", "posterior_fit", "posterior_draw"):
        assert attempt.report.operations[name].attempted == 0, name


def test_replay_composes_with_completed_outcome_and_typed_artifact_without_fitting():
    completed = _completed()
    targeted = _targeted(completed, np.exp(_draw()["z"] / 3))
    carried = targeted.export()
    before = require_success(completed).claim_id
    attempted = observe_native_attempts(lambda: replay_retarget(completed, carried))
    replayed = attempted.unwrap()
    _assert_no_fits(attempted)
    result = require_success(replayed)
    expected = require_success(targeted)
    assert result.ate == expected.ate
    assert result.inspect().target_weights_id == expected.inspect().target_weights_id
    assert replayed.study is not completed.study
    assert require_success(completed).claim_id == before
    assert completed.study.inspect().stage == "executed"
    assert isinstance(replayed, Outcome)
    assert not isinstance(result, Outcome)
    # An outcome containing the typed artifact is accepted without caller reconstruction.
    again = require_success(replay_retarget(completed, Outcome(value=carried)))
    assert again.ate == result.ate


def test_replay_uses_scores_of_the_completed_data_branch_and_preserves_previous_study():
    first_data, second_data = _draw(1, 0), _draw(2, 3)
    declaration = _prepare(first_data)
    first = ant.estimate(declaration)
    second = ant.estimate(ant.rebind(declaration, data=second_data))
    weights = np.exp(second_data["z"])
    current_target = _targeted(second, weights)
    fresh_target = _targeted(_completed(second_data), weights)
    stale_target = _targeted(first, weights)
    assert abs(require_success(fresh_target).ate - require_success(stale_target).ate) > 0.5
    replayed = require_success(replay_retarget(second, current_target.export()))
    assert replayed.ate == require_success(fresh_target).ate
    actual = artifacts.loads(bytes(replayed.export())).contract
    reference = artifacts.loads(bytes(fresh_target.export())).contract
    assert actual["identities"]["data_snapshot"] == reference["identities"]["data_snapshot"]
    assert actual["identities"]["target_weights"] == reference["identities"]["target_weights"]
    assert require_success(first).ate == require_success(_completed(first_data)).ate


def test_replay_refuses_new_snapshot_with_same_shape_without_refitting_or_mutating_source():
    data = _draw()
    targeted = _targeted(_completed(data), np.exp(data["z"] / 3))
    carried = targeted.export()
    same = _completed(data)
    assert (
        require_success(replay_retarget(same.study, bytes(carried))).ate
        == require_success(targeted).ate
    )
    moved = _completed({**data, "y": data["y"] + 1})
    before = require_success(moved).claim_id
    attempted = observe_native_attempts(lambda: replay_retarget(moved, carried))
    refusal = attempted.unwrap()
    _assert_no_fits(attempted)
    assert refusal.value is None and refusal.refusal is not None
    assert refusal.refusal.code == "row_weights_bound_to_snapshot"
    assert require_success(moved).claim_id == before
    assert require_success(ant.estimate(moved.study)).ate == require_success(moved).ate


def test_replay_requires_live_scores_and_preserves_independent_load_semantics():
    completed = _completed()
    targeted = _targeted(completed, np.exp(_draw()["z"] / 3))
    loaded = ant.load(targeted.export())
    assert loaded.study is None
    with pytest.raises(CausalStateError, match="live completed study"):
        replay_retarget(loaded, targeted.export())
    with pytest.raises(CausalTypeError, match=r"requires\ one\ immutable\ prepared\ study\ or\ outcome"):
        replay_retarget({}, targeted.export())
    with pytest.raises(CausalTypeError, match=r"execution\ must\ be\ an\ Execution\ declaration"):
        replay_retarget(completed, targeted.export(), execution={"seed": 3})


def test_replay_operational_failures_preserve_original_observer_and_release_leases():
    completed = _completed()
    targeted = _targeted(completed, np.exp(_draw()["z"] / 3))
    carried = targeted.export()
    tiny = Execution(limits=ResourceLimits(memory_bytes=0))
    for _ in range(2):
        with pytest.raises(CausalResourceError, match=r"exceeds\ the\ remainder"):
            replay_retarget(completed, carried, execution=tiny)
        assert tiny._handle.available_memory_bytes == 0
    token = ant.CancellationToken()
    token.cancel()
    with pytest.raises(CausalCancelledError, match=r"cancelled\ before\ returning\ a\ claim"):
        replay_retarget(completed, carried, execution=Execution(cancel=token))
    original = CausalUnsupportedError(
        "observer interrupted replay", reason_code="effect_not_identified"
    )

    def fail(_fraction, _stage):
        raise original

    with pytest.raises(CausalUnsupportedError) as raised:
        replay_retarget(completed, carried, execution=Execution(on_progress=fail))
    assert raised.value is original
    assert any(frame.name == "fail" for frame in raised.traceback)
    policy = Execution(limits=ResourceLimits(memory_bytes=16_000_000))
    budget = policy._handle.available_memory_bytes
    replayed = replay_retarget(completed, carried, execution=policy)
    assert require_success(replayed).ate == require_success(targeted).ate
    del replayed
    gc.collect()
    assert policy._handle.available_memory_bytes == budget
    assert require_success(replay_retarget(completed, carried)).ate == require_success(targeted).ate
