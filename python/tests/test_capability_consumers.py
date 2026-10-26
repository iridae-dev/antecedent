"""Counterexamples for two exact native consuming methods in the inventory."""

import json
import struct

import pytest
from antecedent import _native
from antecedent.errors import (
    CausalCancelledError,
    CausalResourceError,
    CausalSerializationError,
)
from antecedent.execution import Execution, ResourceLimits
from antecedent.mechanism_composition import MeanContrast, MechanismCompositionRefusal
from antecedent.outcomes import require_success
from antecedent.transport import advanced as transport

from test_mechanism_composition import chain, compose, finite_chain
from test_transport_exact import fixture

_EXACT_PREFIX = b"ANTECEDENT-EXACT-TRANSPORT\x01"


def _cbor_read(data):
    """Decode only the definite CBOR forms emitted by this native test producer."""
    offset = 0

    def read():
        nonlocal offset
        marker = data[offset]
        offset += 1
        major, extra = marker >> 5, marker & 31
        if major == 7:
            if extra in (20, 21, 22):
                return {20: False, 21: True, 22: None}[extra]
            width, fmt = {25: (2, ">e"), 26: (4, ">f"), 27: (8, ">d")}[extra]
            value = struct.unpack(fmt, data[offset : offset + width])[0]
            offset += width
            return value
        if extra < 24:
            length = extra
        else:
            width = {24: 1, 25: 2, 26: 4, 27: 8}[extra]
            length = int.from_bytes(data[offset : offset + width], "big")
            offset += width
        if major in (0, 1):
            return length if major == 0 else -1 - length
        if major in (2, 3):
            value = data[offset : offset + length]
            offset += length
            return value if major == 2 else value.decode()
        if major == 4:
            return [read() for _ in range(length)]
        if major == 5:
            return {read(): read() for _ in range(length)}
        raise AssertionError(f"unexpected native fixture CBOR major type {major}")

    value = read()
    assert offset == len(data)
    return value


def _cbor_write(value):
    """Reframe deliberately modified fixture records, without scientific checks."""

    def head(major, size):
        if size < 24:
            return bytes([(major << 5) | size])
        for extra, width in ((24, 1), (25, 2), (26, 4), (27, 8)):
            if size < 1 << (8 * width):
                return bytes([(major << 5) | extra]) + size.to_bytes(width, "big")
        raise AssertionError("native fixture integer exceeded u64")

    if value is None or isinstance(value, bool):
        return {None: b"\xf6", False: b"\xf4", True: b"\xf5"}[value]
    if isinstance(value, int):
        return head(0, value) if value >= 0 else head(1, -1 - value)
    if isinstance(value, float):
        return b"\xfb" + struct.pack(">d", value)
    if isinstance(value, str):
        encoded = value.encode()
        return head(3, len(encoded)) + encoded
    if isinstance(value, bytes):
        return head(2, len(value)) + value
    if isinstance(value, list):
        return head(4, len(value)) + b"".join(_cbor_write(item) for item in value)
    if isinstance(value, dict):
        return head(5, len(value)) + b"".join(
            _cbor_write(key) + _cbor_write(item) for key, item in value.items()
        )
    raise AssertionError(f"unexpected native fixture type {type(value)}")


def _exact_bytes():
    identified, catalog, data = fixture()
    prepared = transport.prepare_exact(identified, catalog, data, at={"x": 1.0})
    result = require_success(prepared.estimate())
    assert result.probabilities == pytest.approx((0.3, 0.7))
    assert result.mean("y") == pytest.approx(0.35 / (0.15 + 0.35))
    return bytes(prepared.export())


@pytest.mark.parametrize("changed", ["proof", "law", "target_claim", "identity"])
def test_native_exact_consumer_rejects_semantically_changed_records(changed, monkeypatch):
    artifact = _exact_bytes()
    names, payload = _cbor_read(artifact.removeprefix(_EXACT_PREFIX))
    record = _cbor_read(bytes(payload))
    # The test codec itself must preserve a valid native semantic consumer path.
    unchanged = _EXACT_PREFIX + _cbor_write([names, list(_cbor_write(record))])
    replay = transport.consume_exact(unchanged)
    assert require_success(replay.estimate()).probabilities == pytest.approx((0.3, 0.7))
    if changed == "proof":
        record["proof"]["proof"]["graph_signature"] = "substituted-graph"
    elif changed == "law":
        # Still normalized: rejection must bind actual contents, not just total mass.
        record["laws"][0]["probabilities"] = [0.1, 0.4, 0.4, 0.1]
    elif changed == "target_claim":
        record["probabilities"] = [0.1, 0.9]
    else:
        record["identities"]["snapshot"] = "0" * 64
    damaged = _EXACT_PREFIX + _cbor_write([names, list(_cbor_write(record))])
    native_consume = _native.consume_exact_transport
    calls = []

    def checked_native_consume(data, **kwargs):
        calls.append(data)
        return native_consume(data, **kwargs)

    monkeypatch.setattr(_native, "consume_exact_transport", checked_native_consume)
    expected = {
        "proof": "transport.invalid_derivation",
        "law": "transport artifact identity mismatch",
        "target_claim": "transport artifact execution claim mismatch",
        "identity": "transport artifact identity mismatch",
    }
    with pytest.raises(CausalSerializationError, match=expected[changed]):
        transport.consume_exact(damaged)
    assert calls == [damaged]


@pytest.mark.parametrize("control", ["operations", "depth", "memory", "shared_memory", "cancel"])
def test_native_exact_consumer_honors_limits_and_cancellation(control):
    artifact = _exact_bytes()
    if control == "operations":
        with pytest.raises(CausalResourceError, match="limit|budget"):
            transport.consume_exact(artifact, max_operations=1)
    elif control == "depth":
        with pytest.raises(CausalResourceError, match="limit|budget"):
            transport.consume_exact(artifact, max_depth=1)
    elif control == "memory":
        with pytest.raises(CausalResourceError, match="memory budget"):
            transport.consume_exact(artifact, memory_bytes=1)
    elif control == "shared_memory":
        # Shared execution reaches Rust's ctx limit, not the explicit byte precheck.
        policy = Execution(limits=ResourceLimits(memory_bytes=1))
        with pytest.raises(CausalResourceError, match="budget|memory|admission"):
            policy._call(transport.consume_exact, artifact)
    else:
        token = _native.CancellationToken()
        token.cancel()
        with pytest.raises(CausalCancelledError, match="cancel"):
            transport.consume_exact(artifact, cancel=token)
    # A rejected consumer does not withdraw or mutate the independent producer bytes.
    replay = transport.consume_exact(artifact)
    assert require_success(replay.estimate()).mean("y") == pytest.approx(0.7)


def test_native_exact_consumer_replays_the_embedded_law_without_a_producer():
    # _exact_bytes releases its model, catalog, providers and prepared producer.
    artifact = _exact_bytes()
    consumed = transport.consume_exact(artifact)
    assert consumed.acceptance.verified
    assert consumed.acceptance.replayable
    assert consumed.acceptance.resumable
    plan = consumed.inspect()
    assert plan.identification.available
    assert not plan.uncertainty.available
    replayed = require_success(consumed.estimate())
    assert replayed.probabilities == pytest.approx((0.3, 0.7))
    assert replayed.mean("y") == pytest.approx(0.35 / (0.15 + 0.35))
    assert replayed.inspect().program_id == plan.program_id
    assert bytes(consumed.export()) == artifact


@pytest.mark.parametrize("unsupported", ["unknown_outcome", "finite_family"])
def test_mean_contrast_refusals_come_from_the_actual_native_method(unsupported):
    prepared = compose(finite_chain() if unsupported == "finite_family" else chain())
    contrast = MeanContrast.mean_at("absent" if unsupported == "unknown_outcome" else "Y", {})
    # Call the actual handle first: these are valid wire requests, refused by Rust.
    value, raw_refusal = prepared._handle.mean_contrast(json.dumps(contrast.to_wire()))
    assert value is None
    refusal = json.loads(raw_refusal)
    assert refusal["code"] == "route_not_supported"
    assert "mechanism_composition.request_unsupported" in str(refusal)
    with pytest.raises(MechanismCompositionRefusal) as caught:
        require_success(prepared.mean_contrast(contrast))
    assert caught.value.reason_code == refusal["code"]
    assert caught.value.details == ("mechanism_composition.request_unsupported",)
