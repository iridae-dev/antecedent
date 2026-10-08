"""Portable composed result: build, export and independently consume a bundle (C3).

A bundle links the parts of a composed decision as a typed directed graph: the
artifacts embedded from their bytes (a joint distribution, an external claim, a
decision contract and result, a sensitivity artifact, a study ranking, an inverse
query, a repair report), references to a provider or data source the consumer must
supply, the dependency edges between them and the declared evidence relationships.
Each edge carries the Merkle digest of the upstream node, so a changed upstream
identity changes every dependent node and the bundle identity::

    builder = Bundle.builder()
    builder.add_artifact("distribution", law_bytes, node_id="law")
    builder.add_artifact("decision_contract", contract_bytes, node_id="contract")
    builder.add_artifact("decision_result", result_bytes, node_id="result")
    builder.connect("law", "result").connect("contract", "result")
    bundle = builder.build()
    data, identity = bundle.export(), bundle.identity

    # In another process, under the identity retained independently of the bytes:
    consumed = consume_bundle(data, expected_identity=identity)
    consumed.require_verified()
    consumed.value("result", "risky.expected_utility")
    consumed.claim_label        # "joint_draw" or "point_only_attested"

Every embedded artifact is decoded through its own consumer and cross-checked
against the nodes upstream of it. A node is :class:`Verified`,
:class:`ReferenceUnresolved` (a reference whose provider or data source was not
supplied: never claimed replayed, and any declared ``inspected`` value is kept
unverified) or :class:`Failed` at a named stage (``tampered_quantity``,
``swapped_evidence``, ``graph_or_snapshot_mismatch``, ``unsupported_law``,
``provider_request_changed``, ``callback_unavailable``, ``incompatible_version``,
``oversized``, ``unknown_node_kind``, ``edge_digest_mismatch``,
``expected_identity_mismatch``). A failed or unresolved node never hides the rest:
``consumed.value`` still reads a node that verified.

A decision resting on an external mean is labelled ``point_only_attested``: the
claim publishes a mean-only law and no stronger trust than its attestation (it is
``verified_extension`` only when the artifact retains its exact-request
verification receipt), and a decision whose functional needs an aligned joint law
is refused over a mean-only claim as ``unsupported_law``. A bundle never claims
that a serialized result recreates an executable study.

Rust owns the graph, the digests, the verifiers and every refusal; this module
raises each as a :class:`CompositionBundleRefusal`, a
:class:`~antecedent.errors.CausalUnsupportedError` with its registered
``reason_code`` and the namespaced ``composition_bundle.<stage>`` ``detail``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from inspect import signature as _signature
from typing import Any, Literal, TypeAlias

from . import _native
from .decision import Contract
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError

NodeKind = Literal[
    "causal_contract",
    "execution_or_fit",
    "external_claim",
    "attestation",
    "evidence_relationship",
    "quantity_coordinates",
    "support_trust_calibration",
    "distribution",
    "transformation",
    "decision_contract",
    "decision_result",
    "sensitivity",
    "study_ranking",
    "inverse_query",
    "repair_report",
]
Relationship = Literal[
    "shared_data", "shared_prior", "shared_fitted_model", "unknown_dependence", "independent"
]
ClaimLabel = Literal["point_only_attested", "joint_draw"]

NODE_KINDS: tuple[str, ...] = (
    "causal_contract",
    "execution_or_fit",
    "external_claim",
    "attestation",
    "evidence_relationship",
    "quantity_coordinates",
    "support_trust_calibration",
    "distribution",
    "transformation",
    "decision_contract",
    "decision_result",
    "sensitivity",
    "study_ranking",
    "inverse_query",
    "repair_report",
)
#: Kinds whose artifacts can be embedded (the rest are carried as references).
EMBEDDABLE_KINDS: tuple[str, ...] = (
    "external_claim",
    "evidence_relationship",
    "distribution",
    "decision_contract",
    "decision_result",
    "sensitivity",
    "study_ranking",
    "inverse_query",
    "repair_report",
)
RELATIONSHIPS: tuple[str, ...] = (
    "shared_data",
    "shared_prior",
    "shared_fitted_model",
    "unknown_dependence",
    "independent",
)
STAGES: tuple[str, ...] = (
    "tampered_quantity",
    "swapped_evidence",
    "graph_or_snapshot_mismatch",
    "unsupported_law",
    "provider_request_changed",
    "callback_unavailable",
    "incompatible_version",
    "oversized",
    "unknown_node_kind",
    "edge_digest_mismatch",
    "expected_identity_mismatch",
)

# --------------------------------------------------------------------------- refusals


_INVALID = "composition_bundle.invalid_container"
_STAGE_DETAILS: dict[str, str] = {
    "tampered_quantity": "composition_bundle.tampered_quantity",
    "swapped_evidence": "composition_bundle.swapped_evidence",
    "graph_or_snapshot_mismatch": "composition_bundle.graph_or_snapshot_mismatch",
    "unsupported_law": "composition_bundle.unsupported_law",
    "provider_request_changed": "composition_bundle.provider_request_changed",
    "callback_unavailable": "composition_bundle.callback_unavailable",
    "incompatible_version": "composition_bundle.incompatible_version",
    "oversized": "composition_bundle.oversized",
    "unknown_node_kind": "composition_bundle.unknown_node_kind",
    "edge_digest_mismatch": "composition_bundle.edge_digest_mismatch",
    "expected_identity_mismatch": "composition_bundle.expected_identity_mismatch",
}


class CompositionBundleRefusal(CausalUnsupportedError):
    """A refused bundle, node or composition carrying the structured Rust fields.

    ``reason_code`` is registered; ``detail`` is ``composition_bundle.<stage>`` and
    :attr:`bundle_stage` is that stage's short name. ``offending`` names the node
    when there is one.
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        offending = refusal.get("offending")
        supplied = refusal.get("supplied")
        text = str(refusal["detail"])
        if supplied:
            text += f": {supplied}"
        if offending:
            text += f" (node `{offending}`)"
        super().__init__(text, reason_code=refusal["code"], remedy=refusal.get("remedy"))
        self.stage: str = refusal.get("stage", "")
        self.detail: str = refusal["detail"]
        self.offending: str | None = offending
        self.expected: str | None = refusal.get("expected")
        self.supplied: str | None = supplied

    @property
    def bundle_stage(self) -> str:
        """The failed stage's short name, for example ``tampered_quantity``."""
        return self.detail.rpartition(".")[2]


class TamperedQuantityRefusal(CompositionBundleRefusal):
    """A quantity coordinate or a value-bearing artifact differs from what it binds."""


class SwappedEvidenceRefusal(CompositionBundleRefusal):
    """Evidence differs from the evidence the bundle declares."""


class GraphOrSnapshotMismatchRefusal(CompositionBundleRefusal):
    """A graph, contract or data snapshot differs between connected nodes."""


class UnsupportedLawRefusal(CompositionBundleRefusal):
    """A node requires a law (an aligned joint law) its upstream nodes do not supply."""


class ProviderRequestChangedRefusal(CompositionBundleRefusal):
    """A supplied provider answered a different request than the node retains."""


class CallbackUnavailableRefusal(CompositionBundleRefusal):
    """A provider callback or data source the bundle needs was not supplied."""


class IncompatibleVersionRefusal(CompositionBundleRefusal):
    """The bundle or an artifact has an unsupported version."""


class OversizedRefusal(CompositionBundleRefusal):
    """The bundle exceeds a size or count bound."""


class UnknownNodeKindRefusal(CompositionBundleRefusal):
    """A node kind is unknown or has no registered verifier."""


class EdgeDigestMismatchRefusal(CompositionBundleRefusal):
    """A node or edge digest differs from its recomputation, or the graph is malformed."""


class ExpectedIdentityMismatchRefusal(CompositionBundleRefusal):
    """The bundle identity differs from the identity the consumer retained."""


_STAGE_TYPES: dict[str, type[CompositionBundleRefusal]] = {
    "tampered_quantity": TamperedQuantityRefusal,
    "swapped_evidence": SwappedEvidenceRefusal,
    "graph_or_snapshot_mismatch": GraphOrSnapshotMismatchRefusal,
    "unsupported_law": UnsupportedLawRefusal,
    "provider_request_changed": ProviderRequestChangedRefusal,
    "callback_unavailable": CallbackUnavailableRefusal,
    "incompatible_version": IncompatibleVersionRefusal,
    "oversized": OversizedRefusal,
    "unknown_node_kind": UnknownNodeKindRefusal,
    "edge_digest_mismatch": EdgeDigestMismatchRefusal,
    "expected_identity_mismatch": ExpectedIdentityMismatchRefusal,
}


def _refusal(wire: Mapping[str, Any]) -> CompositionBundleRefusal:
    stage = str(wire["detail"]).rpartition(".")[2]
    return _STAGE_TYPES.get(stage, CompositionBundleRefusal)(wire)


def _raise(refusal: str | None) -> None:
    if refusal is not None:
        raise _refusal(json.loads(refusal))


# ------------------------------------------------------------------ declarations


@dataclass(frozen=True)
class ProviderRequirement:
    """A provider or callback that must answer one exact request."""

    provider_id: str
    snapshot_id: str
    request_fingerprint: str


@dataclass(frozen=True)
class DataRequirement:
    """A data snapshot that must have the stated content digest."""

    snapshot_id: str
    digest: str


Requirement: TypeAlias = ProviderRequirement | DataRequirement


def _requirement_json(requirement: Requirement) -> str:
    if isinstance(requirement, ProviderRequirement):
        body = {
            "provider": {
                "provider_id": requirement.provider_id,
                "snapshot_id": requirement.snapshot_id,
                "request_fingerprint": requirement.request_fingerprint,
            }
        }
    elif isinstance(requirement, DataRequirement):
        body = {"data": {"snapshot_id": requirement.snapshot_id, "digest": requirement.digest}}
    else:
        raise CausalTypeError("a reference requires a ProviderRequirement or a DataRequirement")
    return json.dumps(body)


def _requirement_from(wire: Mapping[str, Any]) -> Requirement:
    if "provider" in wire:
        p = wire["provider"]
        return ProviderRequirement(p["provider_id"], p["snapshot_id"], p["request_fingerprint"])
    d = wire["data"]
    return DataRequirement(d["snapshot_id"], d["digest"])


@dataclass(frozen=True)
class SuppliedSources:
    """Providers and data snapshots the consumer can supply to resolve references.

    A provider is matched on its identity and snapshot and must answer the exact
    request fingerprint the reference retains; supplied data must have the
    referenced digest. Supplying a source never claims the reference was replayed.
    """

    providers: tuple[ProviderRequirement, ...] = ()
    data: tuple[DataRequirement, ...] = ()

    def with_provider(
        self, provider_id: str, snapshot_id: str, request_fingerprint: str
    ) -> SuppliedSources:
        """Add a provider that will answer ``request_fingerprint``."""
        added = ProviderRequirement(provider_id, snapshot_id, request_fingerprint)
        return SuppliedSources((*self.providers, added), self.data)

    def with_data(self, snapshot_id: str, digest: str) -> SuppliedSources:
        """Add a data snapshot with content digest ``digest``."""
        return SuppliedSources(self.providers, (*self.data, DataRequirement(snapshot_id, digest)))

    def _json(self) -> str:
        return json.dumps(
            {
                "providers": [
                    {
                        "provider_id": p.provider_id,
                        "snapshot_id": p.snapshot_id,
                        "request_fingerprint": p.request_fingerprint,
                    }
                    for p in self.providers
                ],
                "data": [{"snapshot_id": d.snapshot_id, "digest": d.digest} for d in self.data],
            }
        )


def _bytes(data: object) -> bytes:
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    export = getattr(data, "export", None)
    if callable(export):
        try:
            _signature(export).bind(artifact_id="composition-bundle-node")
        except (TypeError, ValueError) as error:
            raise CausalTypeError("artifact export must accept an artifact_id keyword") from error
        exported = export(artifact_id="composition-bundle-node")
        if isinstance(exported, bytes):
            return exported
    raise CausalTypeError("an artifact is its container bytes (or an object that exports them)")


def _kind(kind: str, *, allow_auto: bool) -> str:
    if kind == "auto" and allow_auto:
        return kind
    if kind not in NODE_KINDS:
        raise CausalValueError(f"`{kind}` is not a bundle node kind; use one of {NODE_KINDS}")
    return kind


# ------------------------------------------------------------------------ building


class BundleBuilder:
    """Collects artifacts, references, dependency edges and evidence relationships.

    Every method returns the builder, so calls chain; :attr:`last_node_id` names the
    node the last ``add_*`` created. A node id defaults to
    ``<kind>:<first 12 hex of the artifact's recomputed identity>``, so a bundle
    built from the same parts has the same identity whatever order they were added.
    """

    def __init__(self) -> None:
        self._native = _native.CompositionBundleBuilder()
        self._ids: list[str] = []
        self._last: str | None = None

    @property
    def node_ids(self) -> tuple[str, ...]:
        """Ids of the nodes added so far, in insertion order."""
        return tuple(self._ids)

    @property
    def last_node_id(self) -> str:
        """Id of the node the last ``add_artifact``, ``add_reference`` or ``relate`` made."""
        if self._last is None:
            raise CausalValueError("no node has been added yet")
        return self._last

    def add_artifact(self, kind: str, data: object, *, node_id: str | None = None) -> BundleBuilder:
        """Embed an artifact from its container bytes.

        ``kind`` is a node kind (``auto`` reads it from the container). The artifact
        is decoded through its own consumer and its identity recomputed, so bytes
        that are not that kind of artifact refuse here with the failed stage.
        """
        added, refusal = self._native.add_artifact(
            _kind(kind, allow_auto=True), _bytes(data), node_id
        )
        _raise(refusal)
        assert added is not None
        self._ids.append(added)
        self._last = added
        return self

    def add_reference(
        self,
        node_id: str,
        kind: str,
        *,
        identity: str,
        requires: Requirement,
        facts: Mapping[str, str] | None = None,
        inspected: Mapping[str, float] | None = None,
    ) -> BundleBuilder:
        """Add a node held elsewhere that needs a supplied provider or data source.

        ``facts`` are declared (unverified) facts such as ``law`` or
        ``graph_or_snapshot`` that the bundle cross-checks against connected nodes;
        ``inspected`` values are kept, unverified, so an inspected result survives
        when the reference cannot be resolved.
        """
        refusal = self._native.add_reference(
            node_id,
            _kind(kind, allow_auto=False),
            identity,
            _requirement_json(requires),
            json.dumps(dict(facts)) if facts else None,
            json.dumps([[key, float(value)] for key, value in inspected.items()])
            if inspected
            else None,
        )
        _raise(refusal)
        self._ids.append(node_id)
        self._last = node_id
        return self

    def connect(self, upstream: str, dependent: str) -> BundleBuilder:
        """Declare that ``dependent`` was derived from ``upstream``."""
        _raise(self._native.connect(upstream, dependent))
        return self

    def relate(self, left: str, right: str, relationship: str) -> BundleBuilder:
        """Declare how the evidence behind two nodes depends on each other.

        The declaration becomes a node both are connected to, and is checked against
        them: a pair declared ``independent`` that rests on the same evidence is
        refused as ``swapped_evidence``. The pair is unordered.
        """
        if relationship not in RELATIONSHIPS:
            raise CausalValueError(f"`{relationship}` is not one of {RELATIONSHIPS}")
        added, refusal = self._native.relate(left, right, relationship)
        _raise(refusal)
        assert added is not None
        self._ids.append(added)
        self._last = added
        return self

    def build(self) -> Bundle:
        """Seal the graph into a :class:`Bundle`."""
        built, refusal = self._native.build()
        _raise(refusal)
        assert built is not None
        return Bundle(built)


@dataclass(frozen=True)
class NodeSummary:
    """One node of a built bundle."""

    id: str
    kind: str
    identity: str
    embedded: bool
    chain_digest: str


@dataclass(frozen=True)
class Edge:
    """A dependency carrying the Merkle digest of the upstream node."""

    source: str
    target: str
    upstream_digest: str


class Bundle:
    """A sealed composed result: a bounded, versioned, BLAKE3-identified graph."""

    def __init__(self, native: _native.CompositionBundle) -> None:
        self._native = native

    @staticmethod
    def builder() -> BundleBuilder:
        """A new, empty :class:`BundleBuilder`."""
        return BundleBuilder()

    @property
    def identity(self) -> str:
        """The bundle identity (lowercase hex), covering every node, edge and digest."""
        return self._native.identity

    @property
    def nodes(self) -> tuple[NodeSummary, ...]:
        """Nodes ordered by id."""
        return tuple(
            NodeSummary(
                n["id"], n["kind"], n["identity"], bool(n["embedded"]), n["chain_digest"] or ""
            )
            for n in json.loads(self._native.nodes_json)
        )

    @property
    def edges(self) -> tuple[Edge, ...]:
        """Edges ordered by ``(source, target)``."""
        wire = json.loads(self._native.edges_json)
        return tuple(Edge(e["from"], e["to"], e["upstream_digest"]) for e in wire)

    def export(self, artifact_id: str = "composition-bundle") -> bytes:
        """Serialize through the checksummed container (one section per embedded node)."""
        data, refusal = self._native.export(artifact_id)
        _raise(refusal)
        assert data is not None
        return data


# ------------------------------------------------------------------------ consuming


@dataclass(frozen=True)
class Verified:
    """An embedded artifact decoded and recomputed to its declared identity, or a
    reference that matched a supplied source. A resolved reference is never claimed
    replayed."""


@dataclass(frozen=True)
class ReferenceUnresolved:
    """A reference whose provider or data source was not supplied."""

    requires: Requirement


@dataclass(frozen=True)
class Failed:
    """Verification failed at a named stage."""

    stage: str
    reason: str
    detail: str = ""
    code: str = ""


NodeStatus: TypeAlias = Verified | ReferenceUnresolved | Failed


@dataclass(frozen=True)
class ConsumedNode:
    """One consumed node."""

    id: str
    kind: str
    identity: str
    chain_digest: str
    status: NodeStatus
    facts: Mapping[str, str] = field(default_factory=dict)
    values: Mapping[str, float] = field(default_factory=dict)
    inspected: Mapping[str, float] = field(default_factory=dict)
    claim_label: ClaimLabel | None = None

    @property
    def verified(self) -> bool:
        """Whether the node verified."""
        return isinstance(self.status, Verified)


def _status(wire: Mapping[str, Any]) -> NodeStatus:
    state = wire["state"]
    if state == "verified":
        return Verified()
    if state == "reference_unresolved":
        return ReferenceUnresolved(_requirement_from(wire["requires"]))
    return Failed(wire["stage"], wire["reason"], wire.get("detail", ""), wire.get("code", ""))


def _label(value: str | None) -> ClaimLabel | None:
    if value == "point_only_attested":
        return "point_only_attested"
    if value == "joint_draw":
        return "joint_draw"
    return None


class ConsumedBundle:
    """A consumed bundle: every node's status, with inspected results preserved."""

    def __init__(self, wire: Mapping[str, Any]) -> None:
        self._identity: str = wire["identity"]
        self._nodes: tuple[ConsumedNode, ...] = tuple(
            ConsumedNode(
                n["id"],
                n["kind"],
                n["identity"],
                n["chain_digest"],
                _status(n["status"]),
                dict(n["facts"]),
                {key: float(value) for key, value in n["values"]},
                {key: float(value) for key, value in n["inspected"]},
                _label(n["claim_label"]),
            )
            for n in wire["nodes"]
        )
        self._edges: tuple[Edge, ...] = tuple(
            Edge(e["from"], e["to"], e["upstream_digest"]) for e in wire["edges"]
        )

    @property
    def identity(self) -> str:
        """The recomputed bundle identity."""
        return self._identity

    @property
    def nodes(self) -> tuple[ConsumedNode, ...]:
        """Nodes ordered by id."""
        return self._nodes

    @property
    def edges(self) -> tuple[Edge, ...]:
        """Edges with their recomputed upstream digests."""
        return self._edges

    def node(self, node_id: str) -> ConsumedNode:
        """One node by id."""
        for node in self._nodes:
            if node.id == node_id:
                return node
        raise CausalValueError(f"the bundle has no node `{node_id}`")

    def value(self, node_id: str, key: str) -> float | None:
        """A number read from a node that verified; ``None`` for any other node."""
        node = self.node(node_id)
        return node.values.get(key) if node.verified else None

    @property
    def all_verified(self) -> bool:
        """Whether every node verified."""
        return all(node.verified for node in self._nodes)

    @property
    def claim_label(self) -> ClaimLabel | None:
        """The claim the bundle's decision results carry, derived from their ancestors.

        ``point_only_attested`` when any decision rests on an external mean (the
        conservative label: a mixed bundle is point-only), ``joint_draw`` when every
        decision rests on aligned joint draws, ``None`` without a decision result.
        """
        labels = {node.claim_label for node in self._nodes if node.claim_label is not None}
        if "point_only_attested" in labels:
            return "point_only_attested"
        if "joint_draw" in labels:
            return "joint_draw"
        return None

    def require_verified(self) -> None:
        """Require every node to have verified.

        Raises the first failed node's typed refusal at its stage, or, for the first
        unresolved reference, :class:`CallbackUnavailableRefusal`.
        """
        for node in self._nodes:
            status = node.status
            if isinstance(status, Failed):
                raise _refusal(
                    {
                        "code": status.code,
                        "stage": "bind",
                        "detail": status.detail or _STAGE_DETAILS.get(status.stage, _INVALID),
                        "offending": node.id,
                        "supplied": status.reason,
                    }
                )
            if isinstance(status, ReferenceUnresolved):
                raise CallbackUnavailableRefusal(
                    {
                        "code": "external_capability_missing",
                        "stage": "bind",
                        "detail": "composition_bundle.callback_unavailable",
                        "offending": node.id,
                        "supplied": "a node needs a provider or data source that was not supplied",
                    }
                )


def consume_bundle(
    data: bytes, *, expected_identity: str, supplied: SuppliedSources | None = None
) -> ConsumedBundle:
    """Consume exported bundle bytes under the identity the consumer retained.

    The container, version, node kinds, node and edge digests and the bundle
    identity are checked first and refuse as a typed
    :class:`CompositionBundleRefusal`; each embedded artifact is then verified
    through its own consumer, and a node-level failure is reported on that node
    without hiding the rest of the bundle.
    """
    wire, refusal = _native.consume_composition_bundle(
        bytes(data),
        expected_identity,
        None if supplied is None else supplied._json(),
    )
    _raise(refusal)
    assert wire is not None
    return ConsumedBundle(json.loads(wire))


@dataclass(frozen=True)
class ArtifactDescription:
    """What an artifact's own consumer establishes about it, read from its bytes."""

    kind: str
    identity: str
    facts: Mapping[str, str]
    values: Mapping[str, float]


def describe_artifact(kind: str, data: object) -> ArtifactDescription:
    """Decode one artifact on its own and report its identity and published facts.

    Raises the typed refusal of the artifact's own consumer for bytes that are not an
    artifact of that kind.
    """
    wire, refusal = _native.composition_describe_artifact(
        _kind(kind, allow_auto=True), _bytes(data)
    )
    _raise(refusal)
    assert wire is not None
    body = json.loads(wire)
    return ArtifactDescription(
        body["kind"],
        body["identity"],
        dict(body["facts"]),
        {key: float(value) for key, value in body["values"]},
    )


def mean_decision(
    contract: Contract, claim: object, *, artifact_id: str = "point-only-decision"
) -> bytes:
    """A decision over an external claim's means, as embeddable point-only result bytes.

    ``claim`` is the exported bytes of a bound external claim (or the claim itself).
    The result is bound to the digest of the claim's means, never to aligned draws:
    placed beneath the claim in a bundle it is labelled ``point_only_attested``.
    A contract whose functional a mean cannot answer (a quantile, a probability or a
    nonlinear utility) refuses as ``unsupported_law``; :meth:`Decision.export`
    refuses the same result by design, because only joint draws replay.
    """
    data, refusal = _native.composition_mean_decision(
        json.dumps(contract._wire()), _bytes(claim), artifact_id
    )
    _raise(refusal)
    assert data is not None
    return data


def detect_kind(data: object) -> str | None:
    """The node kind a container's artifact fills, or ``None`` when it is not one."""
    return _native.composition_detect_kind(_bytes(data))


__all__ = [
    "EMBEDDABLE_KINDS",
    "NODE_KINDS",
    "RELATIONSHIPS",
    "STAGES",
    "ArtifactDescription",
    "Bundle",
    "BundleBuilder",
    "CallbackUnavailableRefusal",
    "ClaimLabel",
    "CompositionBundleRefusal",
    "ConsumedBundle",
    "ConsumedNode",
    "DataRequirement",
    "Edge",
    "EdgeDigestMismatchRefusal",
    "ExpectedIdentityMismatchRefusal",
    "Failed",
    "GraphOrSnapshotMismatchRefusal",
    "IncompatibleVersionRefusal",
    "NodeKind",
    "NodeStatus",
    "NodeSummary",
    "OversizedRefusal",
    "ProviderRequestChangedRefusal",
    "ProviderRequirement",
    "ReferenceUnresolved",
    "Relationship",
    "Requirement",
    "SuppliedSources",
    "SwappedEvidenceRefusal",
    "TamperedQuantityRefusal",
    "UnknownNodeKindRefusal",
    "UnsupportedLawRefusal",
    "Verified",
    "consume_bundle",
    "describe_artifact",
    "detect_kind",
    "mean_decision",
]
