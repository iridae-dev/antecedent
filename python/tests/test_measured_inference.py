"""Measured authority guards; candidate JSON and user basis never mint a license."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest
from antecedent._measured_inference import (
    MeasuredInference,
    MeasuredInferenceIdentity,
    MeasuredScalar,
)
from antecedent.errors import CausalTypeError, CausalUnsupportedError, CausalValueError


def identity():
    return MeasuredInferenceIdentity(
        route="antecedent.transport.joint_bayesian",
        candidate_digest="a" * 64,
        premises_digest="b" * 64,
        data_digest="c" * 64,
        level=0.95,
        scalars=("target_effect",),
        seal="d" * 64,
    )


@pytest.mark.parametrize("kind", [MeasuredInference, MeasuredScalar])
def test_public_constructors_cannot_mint_authority(kind):
    for arguments in [
        {},
        {"calibration": "calibrated", "record_id": "fake", "basis": {}},
    ]:
        with pytest.raises(CausalTypeError):
            kind(**arguments)


def test_foreign_payload_callback_is_not_invoked():
    calls = []
    native = SimpleNamespace(
        payload=lambda: calls.append("payload"), export=lambda: calls.append("export")
    )
    with pytest.raises(CausalTypeError, match="original native authority"):
        MeasuredInference._from_native(native)
    assert calls == []


@pytest.mark.parametrize(
    "changed",
    [
        {"route": ""},
        {"candidate_digest": "f" * 63},
        {"premises_digest": "F" * 64},
        {"level": 0},
        {"level": float("nan")},
        {"level": 10**1000},
        {"scalars": ()},
        {"scalars": ("target_effect", "target_effect")},
        {"seal": "z" * 64},
    ],
)
def test_expected_identity_domain_guards(changed):
    with pytest.raises(CausalValueError):
        replace(identity(), **changed)


def test_expectation_immutable_and_independent_declaration():
    original = identity()
    copy = MeasuredInferenceIdentity._from_wire(original._wire())
    assert copy == original
    with pytest.raises(FrozenInstanceError):
        original.seal = "e" * 64
    with pytest.raises(CausalTypeError):
        replace(original, level=True)
    with pytest.raises(CausalTypeError):
        replace(original, scalars="target_effect")


def test_bounds_and_missing_native_cannot_be_overridden_by_basis(monkeypatch):
    import antecedent._measured_inference as module

    monkeypatch.delattr(module._native, "consume_measured_inference", raising=False)
    for kwargs in [
        {"max_bytes": 64 * 1024 * 1024 + 1},
        {"max_bytes": 0},
        {"memory_limit_bytes": -1},
    ]:
        with pytest.raises(CausalValueError):
            MeasuredInference.load(b"", expected=identity(), **kwargs)
    with pytest.raises(CausalTypeError):
        MeasuredInference.load(b"", expected=identity(), cancel=object())
    with pytest.raises(CausalTypeError):
        MeasuredInference.load(b"", expected=identity()._wire())
    with pytest.raises(CausalUnsupportedError) as caught:
        MeasuredInference.load(b"", expected=identity())
    assert caught.value.reason_code == "cell_not_licensed"


@pytest.mark.parametrize(
    ("expected", "limits", "exception", "detail"),
    [
        ("{}", {"max_bytes": 0}, "CausalResourceError", "consumer_bounds_exceeded"),
        ("{}", {"memory_limit_bytes": 0}, "CausalResourceError", "memory_budget_exceeded"),
        ("{}", {}, "CausalValueError", "invalid_expectation"),
    ],
)
def test_native_input_exceptions_preserve_class_without_scientific_reason(
    expected, limits, exception, detail
):
    from antecedent import _native, errors

    with pytest.raises(getattr(errors, exception)) as caught:
        _native.consume_measured_inference(b"x", expected, **limits)
    assert f"measured_inference.{detail}" in str(caught.value)
    assert getattr(caught.value, "reason_code", None) is None


def test_source_report_unreachable_route_guard_is_class_only():
    import re
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2] / "python/src/measured_inference_api.rs"
    ).read_text(encoding="utf-8")
    # No public constructor can manufacture a seventh route. This exact fallback
    # remains a defensive class-only input error, never a scientific refusal.
    assert '_ => Err(crate::value_err("measured_inference.source_report_not_supported"))' in source
    routes = {
        "joint_bayesian",
        "learned_joint",
        "nested_fisher",
        "nested_bayesian",
        "sampled_recovery",
        "temporal_interval",
    }
    assert set(re.findall(r'"([a-z_]+)"\s*=>', source)) == routes
    assert "#[new]" not in source
    consumer = (
        Path(__file__).resolve().parents[2] / "crates/antecedent-io/src/measured_inference.rs"
    ).read_text(encoding="utf-8")
    start = consumer.index("match expected.route.as_str()")
    end = consumer.index("measured_inference.unsupported_route", start)
    assert set(re.findall(r'"([a-z_]+)"\s*=>', consumer[start:end])) == routes
    assert '_ => Err(refusal("measured_inference.unsupported_route"' in consumer
