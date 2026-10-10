"""Foreign callbacks execute once under complete dependencies; claims stay attested."""

import gc
import json
import os
import subprocess
import sys
import weakref
from dataclasses import replace

import antecedent as ac
import numpy as np
import pytest
from antecedent import external
from antecedent.program_claims import ProgramBinding
from antecedent.recalc import RecalcReceipt
from antecedent.recalc_external import (
    CallbackDescriptor,
    CallbackProvider,
    ExternalCallbackRefusal,
    ExternalCallbackRequest,
    ExternalCallbackSession,
)


def fixture(policy="deterministic", branch=0):
    ident = ac.identify(
        graph=[("a", "y")], names=["a", "y"], query=ac.ResponseCurve("a", "y", grid=[0.0, 1.0])
    )
    spec = external.response(ident, outcome_units="score", dose_units="dose", population="target")
    program = ProgramBinding.from_spec(spec)
    obj = external.ProviderObject(
        provider_id="lab",
        object_id="ols",
        version="v1",
        snapshot="source-1",
        request="request-1",
        meaning="interventional_predictive",
        capabilities=("mean",),
    )
    descriptor = CallbackDescriptor(obj, "python-test-v1", policy)
    a = np.tile([0.0, 1.0], 40)
    request = ExternalCallbackRequest(
        program, spec, descriptor, {"a": a, "y": 3.0 + 2.0 * a}, branch=branch
    )
    calls = []

    def callback(inputs):
        calls.append(inputs)
        assert not inputs.data["a"].flags.writeable
        design = np.column_stack([np.ones(len(inputs.data["a"])), inputs.data["a"]])
        coefficients = np.linalg.lstsq(design, inputs.data["y"], rcond=None)[0]
        values = [float(coefficients @ [1.0, dose]) for dose in inputs.doses]
        return external.Response(
            obj, values, attested_by="lab", support=("supported",) * len(values)
        )

    return request, CallbackProvider(callback, descriptor), calls


def test_callback_actual_invocation_cache_branch_receipt_and_full_rerun():
    request, provider, calls = fixture(branch=1)
    session = ExternalCallbackSession()
    session.plan(request, provider=provider)
    assert calls == [] and not session.is_live
    first = session.execute(request, provider=provider)
    assert first.receipt.totals.external_invocations == 1
    np.testing.assert_allclose(first.claim.values, [3.0, 5.0], atol=1e-12)
    assert first.claim.trust.value == "externally_attested" and not first.claim.native
    assert first.claim.identity_fields["causal_contract_id"] == request.program.contract_id
    assert first.claim.support == ("supported", "supported")
    same = session.execute(request)
    assert same.receipt.totals.external_invocations == 0 and len(calls) == 1
    changed = replace(
        request, data={"a": request.data["a"], "y": np.asarray(request.data["y"]) + 7.0}
    )
    updated = session.execute(changed, provider=provider)
    fresh = ExternalCallbackSession().execute(changed, provider=provider)
    np.testing.assert_allclose(updated.claim.values, [10.0, 12.0], atol=1e-12)
    np.testing.assert_array_equal(updated.claim.values, fresh.claim.values)
    assert (
        updated.receipt.totals.external_invocations
        == fresh.receipt.totals.external_invocations
        == 1
    )
    consumed = RecalcReceipt.consume(first.receipt.export())
    assert consumed.totals.external_invocations == 1


def test_callback_policy_descriptor_and_bound_output_failures_count_actual_attempts():
    request, provider, calls = fixture()
    session = ExternalCallbackSession()
    session.execute(request, provider=provider)
    other = replace(request, descriptor=replace(request.descriptor, environment_id="different"))
    with pytest.raises(ExternalCallbackRefusal) as error:
        session.execute(other, provider=provider)
    assert (error.value.reason_code, error.value.detail, error.value.stage) == (
        "invalid_argument",
        "external_recalc.provider_mismatch",
        "provider_request.0",
    )
    assert len(calls) == 1 and session.is_live
    unknown, unlicensed, unknown_calls = fixture("unknown")
    with pytest.raises(ExternalCallbackRefusal) as error:
        ExternalCallbackSession().execute(unknown, provider=unlicensed)
    assert error.value.detail == "external_recalc.policy_unsupported"
    assert error.value.reason_code == "route_not_supported" and unknown_calls == []
    bad = CallbackProvider(
        lambda inputs: external.Response(request.descriptor.provider, [1.0], attested_by="lab"),
        request.descriptor,
    )
    with pytest.raises(ExternalCallbackRefusal) as error:
        ExternalCallbackSession().execute(request, provider=bad)
    assert error.value.attempt["invocations"] == 1
    assert len(error.value.attempt["request_digest"]) == 64


def test_callback_stateful_execution_and_unsafe_side_effect_retry():
    request, provider, calls = fixture("stateful")
    session = ExternalCallbackSession()
    session.execute(request, provider=provider)
    again = session.execute(request, provider=provider)
    assert again.receipt.totals.external_invocations == 1 and len(calls) == 2
    with pytest.raises(ExternalCallbackRefusal) as error:
        ExternalCallbackSession.resume(session.export_output(), request)
    assert error.value.detail == "external_recalc.replay_unsupported"
    side, _, _ = fixture("side_effecting")
    attempts = []

    def fail(inputs):
        attempts.append(inputs)
        raise RuntimeError("effect failed after invocation")

    actual = CallbackProvider(fail, side.descriptor)
    failed = ExternalCallbackSession()
    with pytest.raises(ExternalCallbackRefusal) as error:
        failed.execute(side, provider=actual)
    assert error.value.detail == "external_recalc.callback_failed"
    assert error.value.attempt["invocations"] == 1 and not failed.is_live
    with pytest.raises(ExternalCallbackRefusal) as error:
        failed.execute(side, provider=actual)
    assert error.value.detail == "external_recalc.retry_unsafe" and len(attempts) == 1


def test_callback_fresh_process_replay_requires_real_provider_and_matches_output(tmp_path):
    request, provider, _ = fixture()
    session = ExternalCallbackSession()
    result = session.execute(request, provider=provider)
    artifact = tmp_path / "callback.art"
    artifact.write_bytes(session.export_output())
    child = """
import json
from test_recalc_external import fixture
from antecedent.recalc_external import ExternalCallbackSession, ExternalCallbackRefusal
request, provider, calls = fixture()
session = ExternalCallbackSession.resume(open(PATH, 'rb').read(), request)
assert not session.is_live
try:
    session.execute(request)
except ExternalCallbackRefusal as error:
    assert error.reason_code == 'external_capability_missing'
else:
    raise AssertionError('portable labels cannot issue authority')
result = session.execute(request, provider=provider)
again = session.execute(request)
print(json.dumps({'values':result.claim.values.tolist(),'calls':len(calls),'actual':result.receipt.totals.external_invocations,'repeat':again.receipt.totals.external_invocations,'native':result.claim.native}))
""".replace("PATH", repr(str(artifact)))
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [str(__import__("pathlib").Path(__file__).parent), os.environ.get("PYTHONPATH", "")]
        ),
    )
    ran = subprocess.run(
        [sys.executable, "-c", child], check=True, text=True, capture_output=True, env=env
    )
    output = json.loads(ran.stdout)
    np.testing.assert_allclose(output["values"], result.claim.values, atol=1e-12)
    assert (output["calls"], output["actual"], output["repeat"], output["native"]) == (
        1,
        1,
        0,
        False,
    )


def test_callback_native_handle_cycle_is_collectible():
    request, _, _ = fixture()

    class Callable:
        def __call__(self, inputs):
            return external.Response(request.descriptor.provider, [3.0, 5.0], attested_by="lab")

    callback = Callable()
    provider = CallbackProvider(callback, request.descriptor)
    callback.provider = provider
    reference = weakref.ref(callback)
    del callback, provider
    gc.collect()
    assert reference() is None


def test_callback_cancellation_bounds_and_replay_mismatch_are_typed():
    from antecedent import _native
    from antecedent.errors import CausalValueError

    request, provider, calls = fixture()
    cancel = _native.CancellationToken()
    cancel.cancel()
    with pytest.raises(ExternalCallbackRefusal) as error:
        ExternalCallbackSession().execute(request, provider=provider, cancel=cancel)
    assert (error.value.reason_code, error.value.detail, error.value.stage) == (
        "cancelled_no_claim",
        "recalc.cancelled",
        "provider_request.0",
    )
    assert calls == [] and error.value.attempt is None

    def during(inputs):
        inputs.cancellation.cancel()
        return external.Response(request.descriptor.provider, [3.0, 5.0], attested_by="lab")

    with pytest.raises(ExternalCallbackRefusal) as error:
        ExternalCallbackSession().execute(
            request, provider=CallbackProvider(during, request.descriptor)
        )
    assert (
        error.value.detail == "recalc.cancelled" and error.value.reason_code == "cancelled_no_claim"
    )
    assert error.value.attempt["invocations"] == 1
    oversized = replace(request, data={"a": np.zeros(100_001)})
    with pytest.raises(CausalValueError) as error:
        ExternalCallbackSession().execute(oversized, provider=provider)
    assert error.value.reason_code == "invalid_argument" and "recalc.limits_exceeded" in str(
        error.value
    )
    session = ExternalCallbackSession()
    session.execute(request, provider=provider)
    artifact = session.export_output()
    replay = ExternalCallbackSession.resume(artifact, request)
    bad = CallbackProvider(
        lambda inputs: external.Response(
            request.descriptor.provider,
            [30.0, 50.0],
            attested_by="lab",
            support=("supported", "supported"),
        ),
        request.descriptor,
    )
    with pytest.raises(ExternalCallbackRefusal) as error:
        replay.execute(request, provider=bad)
    assert (error.value.reason_code, error.value.detail, error.value.stage) == (
        "invalid_argument",
        "external_recalc.replay_mismatch",
        "provider_request.0",
    )
    assert error.value.attempt["invocations"] == 1 and not replay.is_live


def test_callback_capability_requires_actual_issued_native_state():
    from antecedent.recalc import RecalcUnavailable
    from antecedent.recalc_capabilities import (
        Family,
        Operation,
        RetainedKind,
        require_adapter,
        retained_kind,
    )

    request, provider, _ = fixture()
    session = ExternalCallbackSession()
    assert retained_kind(session) == RetainedKind.READABLE
    with pytest.raises(RecalcUnavailable):
        require_adapter(Family.EXTERNAL, Operation.DATA, session)
    session.execute(request, provider=provider)
    assert retained_kind(session) == RetainedKind.LIVE_EXTERNAL_OUTPUT
    assert (
        require_adapter(Family.EXTERNAL, Operation.DATA, session).state_type
        is ExternalCallbackSession
    )
    replay = ExternalCallbackSession.resume(session.export_output(), request)
    assert retained_kind(replay) == RetainedKind.READABLE
    with pytest.raises(RecalcUnavailable):
        require_adapter(Family.EXTERNAL, Operation.INFERENCE, session)
