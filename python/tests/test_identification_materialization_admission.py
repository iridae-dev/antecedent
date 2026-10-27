"""Actual native full-proof output admission and requested Python conversion."""

from __future__ import annotations

import gc
import json

import antecedent as ant
import pytest
from antecedent.errors import CausalCancelledError, CausalResourceError
from antecedent.execution import ResourceLimits
from antecedent.execution_attempt import observe_native_attempts

from test_prepared_identification_views import _FAMILIES, _prepare


@pytest.mark.parametrize("family", _FAMILIES)
def test_full_native_materialization_preserves_all_records_under_real_execution_controls(family):
    study = _prepare(family)
    view = study.identification
    before = view.inspect().to_dict()
    tiny = ant.Execution(limits=ResourceLimits(memory_bytes=1))
    available = tiny._handle.available_memory_bytes
    with pytest.raises(CausalResourceError):
        view.to_dict(execution=tiny)
    assert tiny._handle.available_memory_bytes == available
    assert view.inspect().to_dict() == before
    assert study._engine.last_result is None
    roomy = ant.Execution(limits=ResourceLimits(memory_bytes=64_000_000))
    available = roomy._handle.available_memory_bytes
    observed = observe_native_attempts(lambda: view.to_dict(execution=roomy))
    converted = observed.unwrap()
    assert converted["names"] == study._engine._native.identification_view().schema_names()
    assert converted["summary"] == before
    assert converted["children"]
    assert roomy._handle.available_memory_bytes == available
    for operation in ("glm_fit", "posterior_fit", "posterior_draw", "score_construction"):
        assert observed.report.operations[operation].attempted == 0


@pytest.mark.parametrize("family", _FAMILIES)
def test_native_payload_owner_funds_exact_capacity_after_preparation_is_dropped(family):
    study = _prepare(family)
    native = study._engine._native.identification_view()
    policy = ant.Execution(limits=ResourceLimits(memory_bytes=64_000_000))
    available = policy._handle.available_memory_bytes
    payload = policy._call(native.materialize, seed=policy.seed, threads=policy.threads)
    retained = payload.retained_bytes()
    assert retained >= payload.encoded_bytes() > 0
    assert policy._handle.available_memory_bytes == available - retained
    assert len(repr(payload)) < 200
    del study, native
    gc.collect()
    first = payload.to_dict()
    original = first["summary"].copy()
    first["summary"]["status"] = "fabricated"
    assert payload.to_dict()["summary"] == original
    assert policy._handle.available_memory_bytes == available - retained
    del payload
    gc.collect()
    assert policy._handle.available_memory_bytes == available


def test_real_native_materialization_honors_pre_cancelled_execution_and_preserves_source():
    from antecedent import _native

    study = _prepare("static")
    before = study.identification.inspect().to_dict()
    token = _native.CancellationToken()
    token.cancel()
    policy = ant.Execution(cancel=token, limits=ResourceLimits(memory_bytes=64_000_000))
    available = policy._handle.available_memory_bytes
    with pytest.raises(CausalCancelledError):
        study.identification.to_dict(execution=policy)
    assert policy._handle.available_memory_bytes == available
    assert study.identification.inspect().to_dict() == before
    assert study._engine.last_result is None


def test_admission_lives_through_python_decoding_and_late_cancellation_publishes_nothing(
    monkeypatch,
):
    from antecedent import _native

    study = _prepare("static")
    token = _native.CancellationToken()
    policy = ant.Execution(cancel=token, limits=ResourceLimits(memory_bytes=64_000_000))
    available = policy._handle.available_memory_bytes
    original_loads = json.loads
    decoded = []
    published = []

    def cancel_after_decode(text, *args, **kwargs):
        assert policy._handle.available_memory_bytes < available
        value = original_loads(text, *args, **kwargs)
        decoded.append(value["summary"]["kind"])
        token.cancel()
        return value

    with monkeypatch.context() as patch:
        patch.setattr(json, "loads", cancel_after_decode)
        with pytest.raises(CausalCancelledError):
            published.append(study.identification.to_dict(execution=policy))
    assert decoded == ["identification"]
    assert published == []
    assert policy._handle.available_memory_bytes == available
    assert study.identification.inspect().kind == "identification"
    assert study._engine.last_result is None
