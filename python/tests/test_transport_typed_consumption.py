"""Transport acceptance is earned by the family consumer, separately from bytes."""

import importlib

import pytest
from antecedent._artifact_protocol import ExportedArtifact, artifact_from_receipt
from antecedent.errors import CausalSerializationError
from antecedent.outcomes import require_success
from antecedent.transport import advanced as transport

from test_transport_z import fixture

identify_module = importlib.import_module("antecedent.transport._impl.identify")


def test_z_typed_consumer_invokes_the_actual_verifier_once(monkeypatch):
    _, _, _, prepared, _ = fixture(False)
    require_success(prepared.estimate())
    raw = bytes(prepared.export())
    exported = artifact_from_receipt(prepared, raw)
    assert isinstance(exported, ExportedArtifact)
    assert bytes(exported) == raw
    calls = []
    original = identify_module._consume_z_transport_artifact

    def checked(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(identify_module, "_consume_z_transport_artifact", checked)
    consumed = transport.consume_z_transport_artifact(exported)
    assert calls == [raw]
    assert consumed.acceptance.verified
    assert consumed.acceptance.replayable
    assert not consumed.acceptance.resumable
    assert not consumed.acceptance.identity_bound
    assert consumed["probabilities"]
    assert bytes(consumed.export()) == raw
    with pytest.raises(TypeError, match="does not support item assignment"):
        consumed.payload["status"] = "forged"


def test_typed_transport_export_does_not_bypass_native_refusal():
    _, _, _, prepared, _ = fixture(False)
    require_success(prepared.estimate())
    raw = bytes(prepared.export())
    damaged = ExportedArtifact(raw[:-1], "ANTECEDENT-Z-TRANSPORT")
    with pytest.raises(CausalSerializationError, match="cbor error"):
        transport.consume_z_transport_artifact(damaged)


def test_restricted_whole_export_preserves_verified_coordinates_and_every_member():
    import antecedent
    from antecedent.transport._restricted import consume_restricted_artifacts

    _, stage, catalog, left, laws = fixture(False)
    right = stage.prepare_exact(catalog, laws, {"x": 1.0})
    require_success(left.estimate())
    require_success(right.estimate())
    execution = consume_restricted_artifacts([left.export(), right.export()])
    assert tuple(dict(point["at"]) for point in execution.points) == ({"x": 0.0}, {"x": 1.0})
    assert len(execution.member_acceptance) == 2
    assert all(receipt.verified for receipt in execution.member_acceptance)
    exported = execution.export()
    assert isinstance(exported, ExportedArtifact)
    loaded = require_success(antecedent.load(exported))
    assert tuple(point.to_dict() for point in loaded.points) == tuple(
        point.to_dict() for point in execution.points
    )
    assert loaded.acceptance.verified
    with pytest.raises(CausalSerializationError, match="coordinates"):
        consume_restricted_artifacts([left.export(), right.export()], [{"x": 1.0}, {"x": 0.0}])


def test_z_fork_and_snapshot_binding_leave_the_parent_without_a_run_header():
    from antecedent.errors import CausalUnsupportedError

    _, _, _, parent, laws = fixture(False)
    branch = parent.bind_snapshot(laws)
    run = branch.fork()
    require_success(run.estimate())
    assert bytes(run.export())
    for preparation in (parent, branch):
        with pytest.raises(CausalUnsupportedError, match="estimate before exporting a z-transport artifact"):
            preparation.export()
