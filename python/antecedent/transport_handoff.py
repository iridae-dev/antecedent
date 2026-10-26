"""Hand a transported claim to a callback input, and consume the record of the connection.

A transported claim (a value about a target population derived from other populations under a
declared selection diagram) may be offered to an input of a callback
(:mod:`antecedent.provider_callbacks`) that is declared to accept transported claims. The handoff
is a *record of one connection*: it seals the offer, the callback's declaration and the transport
identity, and a fresh consumer re-derives the connection and refuses an edit, a relabelled
calibration, a consumer that needs native target data and another callback declaration::

    from antecedent import transport_handoff as th

    handoff = th.hand_to_callback(offer, callback_declaration, "effect")
    handoff.standing, handoff.native_target_data       # "transported_claim", False
    consumed = th.consume_for_callback(
        handoff.export(), callback_declaration, expected_identity=handoff.identity
    )

What the record is, and is not:

* **Never native target data.** The claim arrives with its transport identity (source
  populations, target population, selection diagram, assumptions); ``native_target_data`` is
  always ``False``. A callback input that accepts native target data only refuses it
  (``provider_transport.native_target_required``); with ``accepts_transported=False`` the seal
  refuses.
* **The calibration label is the producer's.** A transported claim offered as a point carries
  ``point_only``; the handoff never upgrades a label.
* **The identity binds the whole connection**: the offer, the callback declaration, the input and
  the transport identity. A changed dependency version of the callback at an unchanged shape is
  another declaration and refuses ``provider_transport.identity_mismatch``.

**Refresh.** A refreshed offer (new outcome-only data of the source) is a new offer: recompute the
offer's identity and the handoff identity; keep the callback declaration and the consumed input;
refuse an offer whose quantity is not in the target population of its transport identity and a
changed callback declaration.

Rust owns the connection check, the identities, the container and every refusal; this module
declares the offer and presents the record. Refusals raise
:class:`~antecedent.provider_callbacks.CallbackRefusal` with a registered ``reason_code``, a
``provider_transport.*`` (or ``interface_contract.*``) ``detail`` and a ``remedy``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from . import _native
from ._artifact_protocol import (
    ExportedArtifact,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from ._columns import coerce_fields
from ._scientific import ScientificResult
from .errors import CausalTypeError
from .provider_callbacks import (
    CallbackDeclaration,
    Support,
    _dumps,
    _name,
    _raise,
)
from .scientific_bridge import _quantity_wire

__all__ = [
    "TransportHandoff",
    "TransportIdentity",
    "TransportedClaimOffer",
    "consume_for_callback",
    "hand_to_callback",
]


@dataclass(frozen=True, slots=True)
class TransportIdentity:
    """How a claim was carried: where it was learned, where it was carried, the selection
    diagram (a 64-hex identity) and the declared invariance and selection assumptions."""

    source_populations: tuple[str, ...]
    target_population: str
    selection_diagram: str
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        coerce_fields(
            self,
            source_populations=tuple(self.source_populations),
            assumptions=tuple(self.assumptions),
        )

    def _wire(self) -> dict[str, Any]:
        return {
            "source_populations": list(self.source_populations),
            "target_population": self.target_population,
            "selection_diagram": self.selection_diagram,
            "assumptions": list(self.assumptions),
        }

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> TransportIdentity:
        return cls(
            tuple(wire["source_populations"]),
            wire["target_population"],
            wire["selection_diagram"],
            tuple(wire["assumptions"]),
        )


@dataclass(frozen=True, slots=True)
class TransportedClaimOffer:
    """A claim about the target population that was carried there by transport.

    ``quantity`` is the coordinate of the claim, in the target population; ``operations`` the
    interface-contract operations offered (``point``, ``mean``, ``law``, ``draws``,
    ``joint_draws``, ``covariance``, ``interval``, ``posterior``); ``uncertainty`` the meaning of
    an offered law or posterior; ``evidence`` the evidence-ledger sources (64-hex) the claim
    rests on, never empty: a transported claim is derived from something.
    """

    component: str
    port: str
    quantity: Any
    semantic_identity: str
    support: Support
    operations: tuple[str, ...]
    transport: TransportIdentity
    evidence: tuple[str, ...]
    uncertainty: str | None = None

    def __post_init__(self) -> None:
        coerce_fields(
            self,
            operations=tuple(sorted(set(self.operations))),
            evidence=tuple(sorted(set(self.evidence))),
        )

    def _wire(self) -> dict[str, Any]:
        return {
            "component": _name(self.component, "component"),
            "port": _name(self.port, "port"),
            "quantity": _quantity_wire(self.quantity, "quantity"),
            "semantic_identity": _name(self.semantic_identity, "semantic_identity"),
            "support": self.support._wire(),
            "operations": list(self.operations),
            "uncertainty": self.uncertainty,
            "transport": self.transport._wire(),
            "evidence": list(self.evidence),
        }


@dataclass(repr=False, frozen=True, slots=True, eq=False)
class TransportHandoff(ScientificResult):
    """A sealed handoff of a transported claim to a callback input.

    ``standing`` is always ``transported_claim`` and ``native_target_data`` always ``False``.
    ``identity`` is the artifact identity: retain it independently of the bytes.
    ``calibration`` is the producer's label (``point_only`` for a point), never upgraded.
    """

    identity: str
    standing: str
    native_target_data: bool
    transport: TransportIdentity
    calibration: str
    _data: bytes = field(repr=False, compare=False, default=b"")

    @artifact_export
    def export(self) -> ExportedArtifact:
        """The checksummed ``transport_handoff_v1`` container."""
        return artifact_from_receipt(self, self._data)


def _summary(info: Mapping[str, Any], data: bytes) -> TransportHandoff:
    data = container_bytes(data)
    return TransportHandoff(
        identity=str(info["identity"]),
        standing=str(info["standing"]),
        native_target_data=bool(info["native_target_data"]),
        transport=TransportIdentity._from_wire(info["transport"]),
        calibration=str(info["calibration"]),
        _data=data,
    )


def hand_to_callback(
    offer: TransportedClaimOffer,
    callback: CallbackDeclaration,
    input: str,  # noqa: A002 - the declared input name, as the protocol spells it
    *,
    accepts_transported: bool = True,
    artifact_id: str = "transport-handoff",
) -> TransportHandoff:
    """Seal ``offer`` to ``input`` of ``callback`` and verify it against the declaration.

    Raises:
        ~antecedent.provider_callbacks.CallbackRefusal: a malformed offer or callback, an input
            policy of native target data only (``accepts_transported=False``), a quantity
            outside the target population, an input another callback supplies, or a connection
            defect (population, regime, units, operations, uncertainty meaning), each with its
            registered code and detail.
    """
    if not isinstance(offer, TransportedClaimOffer):
        raise CausalTypeError("offer must be a TransportedClaimOffer")
    if not isinstance(callback, CallbackDeclaration):
        raise CausalTypeError("callback must be a provider_callbacks.CallbackDeclaration")
    parts, refusal = _native.transport_handoff_seal(
        _dumps(offer._wire()),
        _dumps(callback._wire()),
        _name(input, "input"),
        bool(accepts_transported),
        _name(artifact_id, "artifact_id"),
    )
    _raise(refusal)
    assert parts is not None
    data, _identity, summary = parts
    return _summary(json.loads(summary), bytes(data))


def consume_for_callback(
    data: bytes,
    callback: CallbackDeclaration,
    *,
    expected_identity: str | None = None,
) -> TransportHandoff:
    """Consume a handoff's bytes and verify them against the callback declaration held.

    ``expected_identity`` is an artifact identity retained independently of the bytes; with it
    a consistently resealed handoff of another claim is refused.

    Raises:
        ~antecedent.provider_callbacks.CallbackRefusal: corruption, another version, a
            declaration the library refuses, a stored identity that does not match, a handoff
            sealed for another callback (``provider_transport.identity_mismatch``) or a
            connection that no longer holds.
    """
    data = container_bytes(data)
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise CausalTypeError("data must be bytes")
    if not isinstance(callback, CallbackDeclaration):
        raise CausalTypeError("callback must be a provider_callbacks.CallbackDeclaration")
    summary, refusal = _native.transport_handoff_consume(
        bytes(data), _dumps(callback._wire()), expected_identity
    )
    _raise(refusal)
    assert summary is not None
    return _summary(json.loads(summary), bytes(data))
