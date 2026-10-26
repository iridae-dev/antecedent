"""The common read/persistence boundary must not change native authority or errors."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from antecedent._artifact_protocol import (
    ArtifactAcceptance,
    ExportedArtifact,
    acceptance_from_consumer,
    artifact_export,
    artifact_from_receipt,
    exported_artifact,
)
from antecedent._scientific import ScientificResult
from antecedent.errors import (
    CausalCancelledError,
    CausalResourceError,
    CausalSerializationError,
    CausalStateError,
    CausalValueError,
    EffectNotIdentified,
    StructuredRefusal,
)
from antecedent.outcomes import (
    Answer,
    Outcome,
    Refused,
    as_outcome,
    capture_outcome,
    outcome_from_error,
    require_success,
)
from antecedent.results._execution import ResultAPI, answer_from_artifact
from antecedent.results._report import as_inspection
from antecedent.results._slots import ReasoningSlots


def fail(error):
    raise error


def test_outcomes_retain_the_exact_family_reading_and_backing_handle():
    handle = object()
    reading = SimpleNamespace(answer=Answer("point", value=1.25), _native=handle)
    outcome = as_outcome(reading)
    assert isinstance(outcome, Outcome)
    assert require_success(outcome) is reading
    assert require_success(outcome)._native is handle
    assert outcome.answer is reading.answer
    assert as_outcome(outcome) is outcome


def test_scientific_refusal_has_an_explicit_answer_and_original_error():
    original = EffectNotIdentified("not identified")
    original.reason_code = "effect_not_identified"
    original.search_capped = True
    original.search_complete = False
    outcome = capture_outcome(fail, original)
    assert outcome.accepted is False
    assert isinstance(outcome.answer, Refused)
    assert outcome.answer.kind == "refused"
    assert outcome.refusal.fields["search_complete"] is False
    assert outcome.refusal.fields["search_capped"] is True
    with pytest.raises(EffectNotIdentified) as error:
        require_success(outcome)
    assert error.value is original
    assert outcome.inspect().stage == "refused"


@pytest.mark.parametrize(
    "error",
    [
        CausalValueError("argument"),
        TypeError("argument"),
        CausalResourceError("memory"),
        CausalCancelledError("cancelled"),
        CausalSerializationError("corrupt container"),
        CausalStateError("not retained"),
        RuntimeError("unexpected implementation failure"),
        StructuredRefusal({"code": "invalid_argument", "detail": "test.argument"}),
    ],
)
def test_operational_failures_raise_the_original_object(error):
    with pytest.raises(type(error)) as raised:
        capture_outcome(fail, error)
    assert raised.value is error


def test_callback_failure_cause_is_not_converted_to_a_scientific_value():
    callback = ValueError("user callback failed")
    error = EffectNotIdentified("domain wrapper")
    error.__cause__ = callback
    with pytest.raises(EffectNotIdentified) as raised:
        outcome_from_error(error)
    assert raised.value is error
    assert raised.value.__cause__ is callback


def test_portable_refusal_is_distinct_from_unavailable_and_never_upgraded_by_a_payload():
    answer = answer_from_artifact(
        {"claim": {"kind": "refusal", "detail": "not_licensed"}},
        {"quasi_rollout": {"uncertainty": "point_only"}},
    )
    assert isinstance(answer, Refused)
    assert answer.kind == "refused"
    assert answer.value is None


def test_inspect_and_repr_do_not_serialize_or_export_scientific_readings():
    @dataclass(repr=False)
    class Reading(ScientificResult):
        status: str
        rows: object

        def to_dict(self):
            raise AssertionError("inspect/repr must not serialize the reading")

        def export(self):
            raise AssertionError("inspect/repr must not export")

    class NoDisplay:
        def __repr__(self):
            raise AssertionError("nested payload must not be rendered")

    reading = Reading("identified", NoDisplay())
    assert "identified" in repr(reading)
    assert reading.inspect().details["status"] == "identified"
    assert "Reading" in repr(as_outcome(reading))


def test_execution_inspection_uses_retained_contract_without_exporting():
    class ExplodingExecution:
        def export_contracted_artifact(self):
            raise AssertionError("inspection must not export a native execution")

    class Reading(ResultAPI):
        _execution = ExplodingExecution()
        reasoning = ReasoningSlots.from_contract({})
        answer = Answer("point", value=2.0)

    report = Reading().inspect()
    assert report.answer.value == 2.0


def test_reasoning_inspection_does_not_call_to_dict(monkeypatch):
    slots = ReasoningSlots.from_contract({})
    monkeypatch.setattr(ReasoningSlots, "to_dict", lambda _: pytest.fail("hidden serialization"))
    assert as_inspection(slots).stage == "prepared"


def test_export_receipt_adapter_preserves_receipt_and_does_not_export_twice(monkeypatch):
    import antecedent._artifact_protocol as protocol

    monkeypatch.setattr(protocol, "inspect_artifact", lambda _: SimpleNamespace(kind="test_family"))

    @dataclass
    class Receipt:
        artifact: bytes
        identity: str
        carried: tuple[str, ...]

    class Producer:
        def __init__(self):
            self.calls = 0

        @artifact_export
        def export(self):
            self.calls += 1
            return Receipt(b"native bytes", "independent identity", ("scientific claim",))

    source = Producer()
    artifact = exported_artifact(source)
    assert isinstance(artifact, ExportedArtifact)
    assert source.calls == 1
    assert bytes(artifact) == b"native bytes"
    assert artifact.expectation.identity == "independent identity"
    assert artifact.metadata["carried"] == ("scientific claim",)
    assert artifact.export() is artifact
    assert artifact_from_receipt(source, artifact) is artifact


def test_acceptance_requires_a_real_consumer_receipt_not_a_kind_label():
    no_receipt = SimpleNamespace(kind="selective_discovery_v1", status="verified")
    assert acceptance_from_consumer(no_receipt).status == "unavailable"
    assert acceptance_from_consumer(no_receipt).resumable is False
    receipt = ArtifactAcceptance("sealed", True, False, False)
    actual = acceptance_from_consumer(SimpleNamespace(acceptance=receipt))
    assert actual == receipt
    explicit = SimpleNamespace(
        receipt={"accepts_as_verified_program": "true", "recognized": "true"}
    )
    assert acceptance_from_consumer(explicit).verified is True
    assert acceptance_from_consumer(explicit).resumable is False
