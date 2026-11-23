"""Portable composed result: build, export and independently consume a bundle.

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
that a serialized result recreates an executable study. Independently checked
``recalculation_receipt`` nodes report historical work and supply no live state
or provider authentication. ``frozen_scores`` nodes retain their original
snapshot and executed-fit identities for the separately supported same-row
retarget operation. Connecting a receipt to scores checks both bindings; a
changed fit refuses even when the data snapshot is the same.

A sequential policy reliability
(:func:`antecedent.decision_reliability.sequential_policy_reliability`) embeds as three
connected nodes with :meth:`BundleBuilder.add_sequential_policy_reliability`: its sealed raw
producer, its decision draws and its diagnostic, each verified by recomputation.
:func:`compile_sequential_reliability_node_bundle` compiles the built bundle into one program
that replays the producer and reassesses the diagnostic from the retained draws
(:class:`SequentialReliabilityBundleReceipt`, a point diagnostic), and
:func:`consume_portable_sequential_reliability_node_bundle` reconstructs and recomputes the
sealed bundle under the retained bundle and program identities.

Rust owns the graph, the digests, the verifiers and every refusal; this module
raises each as a :class:`CompositionBundleRefusal`, a
:class:`~antecedent.errors.StructuredRefusal` (a ``CausalUnsupportedError``) with its
registered ``code`` / ``reason_code``, the namespaced ``composition_bundle.<stage>`` ``detail``,
``offending`` and ``remedy``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from inspect import signature as _signature
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from antecedent import _reason_details

from . import _native
from ._artifact_protocol import (
    ArtifactAcceptance,
    ArtifactExpectation,
    ExportedArtifact,
    artifact_consumer,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from ._refusal import raise_refusal
from ._scientific import ScientificResult
from ._unit_support import read_unit_support
from ._validate import count
from ._validate import name as _name
from .decision import Contract
from .errors import CausalTypeError, CausalValueError, StructuredRefusal
from .execution import Execution, native_controls, resolve_execution
from .joint_distribution import JointDistributionArtifact
from .outcomes import outcome_operation, require_success

if TYPE_CHECKING:
    from .propagation_chain import PropagationChain

if TYPE_CHECKING:
    from .execution import Execution
    from .graph_imputation_decision import GraphImputationDecisionArtifact

if TYPE_CHECKING:
    from .decision_reliability import SequentialPolicyReliability

_UNSET: Any = object()

NodeKind = Literal[
    "causal_contract",
    "execution_or_fit",
    "cyclic_posterior_fit",
    "trial_ipw_transport_fit",
    "graph_imputation_transport_fit",
    "joint_transport_effect_fit",
    "joint_transport_predictive_fit",
    "trial_ipw_predictive_fit",
    "cyclic_target_feedback_fit",
    "graph_imputation_predictive_fit",
    "assessed_bridge_posterior_fit",
    "sequential_policy_reliability_fit",
    "sequential_decision_draws",
    "sequential_reliability_diagnostic",
    "predictive_chain_decision",
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
    "rollout",
    "inverse_query",
    "repair_report",
    "recalculation_receipt",
    "frozen_scores",
    "observation_query",
    "scientific_bridge",
    "propagation_chain",
]
"""The kind of a bundle node, named for the part of a composed decision it carries (a
distribution, a decision contract or result, a sensitivity artifact, a study ranking, ...).
:data:`EMBEDDABLE_KINDS` can be embedded from bytes; the rest are carried as references."""
Relationship = Literal[
    "shared_data", "shared_prior", "shared_fitted_model", "unknown_dependence", "independent"
]
"""How the evidence behind two nodes depends on each other, as declared with
:meth:`BundleBuilder.relate`; ``independent`` is refused when the two rest on the same evidence."""
ClaimLabel = Literal["point_only_attested", "joint_draw"]
"""The claim a decision result carries: ``joint_draw`` when it rests on aligned joint draws,
``point_only_attested`` when it rests on an external mean (the conservative label)."""

NODE_KINDS: tuple[str, ...] = (
    "causal_contract",
    "execution_or_fit",
    "cyclic_posterior_fit",
    "trial_ipw_transport_fit",
    "graph_imputation_transport_fit",
    "joint_transport_effect_fit",
    "joint_transport_predictive_fit",
    "trial_ipw_predictive_fit",
    "cyclic_target_feedback_fit",
    "graph_imputation_predictive_fit",
    "assessed_bridge_posterior_fit",
    "sequential_policy_reliability_fit",
    "sequential_decision_draws",
    "sequential_reliability_diagnostic",
    "predictive_chain_decision",
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
    "rollout",
    "inverse_query",
    "repair_report",
    "recalculation_receipt",
    "frozen_scores",
    "observation_query",
    "scientific_bridge",
    "propagation_chain",
)
#: Kinds whose artifacts can be embedded (the rest are carried as references).
EMBEDDABLE_KINDS: tuple[str, ...] = (
    "causal_contract",
    "attestation",
    "quantity_coordinates",
    "transformation",
    "external_claim",
    "evidence_relationship",
    "distribution",
    "decision_contract",
    "decision_result",
    "sensitivity",
    "study_ranking",
    "rollout",
    "inverse_query",
    "repair_report",
    "recalculation_receipt",
    "frozen_scores",
    "observation_query",
    "scientific_bridge",
    "propagation_chain",
)
RELATIONSHIPS: tuple[str, ...] = (
    "shared_data",
    "shared_prior",
    "shared_fitted_model",
    "unknown_dependence",
    "independent",
)
PredictiveFitKind = Literal[
    "joint_transport_predictive_fit",
    "trial_ipw_predictive_fit",
    "cyclic_target_feedback_fit",
    "graph_imputation_predictive_fit",
    "assessed_bridge_posterior_fit",
]
"""The node kind of a native predictive producer embedded with
:meth:`BundleBuilder.add_predictive_fit`."""
#: Kinds :meth:`BundleBuilder.add_predictive_fit` embeds.
PREDICTIVE_FIT_KINDS: tuple[str, ...] = (
    "joint_transport_predictive_fit",
    "trial_ipw_predictive_fit",
    "cyclic_target_feedback_fit",
    "graph_imputation_predictive_fit",
    "assessed_bridge_posterior_fit",
)
RecomputationRoute = Literal[
    "chain", "predictive_decision", "posterior_decision", "predictive_fit_decision"
]
"""The terminal a :class:`CompiledBundleRecomputation` executes live: a propagation chain, a
decision result scored on a predictive law or on a parameter posterior, or a predictive producer
-> chain -> decision path."""
#: Routes :func:`consume_bundle_recomputation` replays.
RECOMPUTATION_ROUTES: tuple[str, ...] = (
    "chain",
    "predictive_decision",
    "posterior_decision",
    "predictive_fit_decision",
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


_INVALID = _reason_details.composition_bundle.INVALID_CONTAINER
_STAGE_DETAILS: dict[str, str] = {
    "tampered_quantity": _reason_details.composition_bundle.TAMPERED_QUANTITY,
    "swapped_evidence": _reason_details.composition_bundle.SWAPPED_EVIDENCE,
    "graph_or_snapshot_mismatch": _reason_details.composition_bundle.GRAPH_OR_SNAPSHOT_MISMATCH,
    "unsupported_law": _reason_details.composition_bundle.UNSUPPORTED_LAW,
    "provider_request_changed": _reason_details.composition_bundle.PROVIDER_REQUEST_CHANGED,
    "callback_unavailable": _reason_details.composition_bundle.CALLBACK_UNAVAILABLE,
    "incompatible_version": _reason_details.composition_bundle.INCOMPATIBLE_VERSION,
    "oversized": _reason_details.composition_bundle.OVERSIZED,
    "unknown_node_kind": _reason_details.composition_bundle.UNKNOWN_NODE_KIND,
    "edge_digest_mismatch": _reason_details.composition_bundle.EDGE_DIGEST_MISMATCH,
    "expected_identity_mismatch": _reason_details.composition_bundle.EXPECTED_IDENTITY_MISMATCH,
}


class CompositionBundleRefusal(StructuredRefusal):
    """A refused bundle, node or composition carrying the structured Rust fields.

    A :class:`~antecedent.errors.StructuredRefusal`: ``reason_code`` (alias ``code``) is
    registered; ``detail`` is ``composition_bundle.<stage>`` and :attr:`bundle_stage` is that
    stage's short name. ``offending`` names the node when there is one.
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        offending = refusal.get("offending")
        supplied = refusal.get("supplied") or refusal.get("message")
        text = str(refusal["detail"])
        if supplied:
            text += f": {supplied}"
        if offending:
            text += f" (node `{offending}`)"
        super().__init__(refusal, text=text)
        self.expected: str | None = refusal.get("expected")
        self.supplied: str | None = refusal.get("supplied")

    @property
    def bundle_stage(self) -> str:
        """The failed stage's short name, for example ``tampered_quantity``."""
        return (self.detail or "").rpartition(".")[2]


class NodeNotFoundRefusal(CompositionBundleRefusal):
    """An edge or relationship names a node that was never added (``node_not_found``).

    The reason code is ``invalid_argument``, so it is also a
    :class:`~antecedent.errors.CausalValueError`; ``offending`` is the missing node id.
    """


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


def _refusal_type(wire: Mapping[str, Any]) -> type[CompositionBundleRefusal]:
    stage = str(wire["detail"]).rpartition(".")[2]
    return _STAGE_TYPES.get(stage, CompositionBundleRefusal)


def _refusal(wire: Mapping[str, Any]) -> CompositionBundleRefusal:
    return _refusal_type(wire)(wire)


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
    from .outcomes import require_success

    data = require_success(data)
    from ._artifact_protocol import ExportedArtifact, container_bytes

    if isinstance(data, (bytes, bytearray, memoryview, ExportedArtifact)):
        return container_bytes(data)
    from ._scientific import ScientificResult

    if (
        isinstance(data, ScientificResult)
        and type(data).export is ScientificResult.export
        and getattr(data, "_artifact_bytes", None) is None
    ):
        raise CausalTypeError(
            "this scientific object has no portable artifact; export its owning session or selected record"
        )
    export = getattr(data, "export", None)
    if callable(export):
        try:
            _signature(export).bind(artifact_id="composition-bundle-node")
        except (TypeError, ValueError):
            try:
                _signature(export).bind()
            except (TypeError, ValueError) as error:
                raise CausalTypeError(
                    "artifact export requires explicit query or record arguments; pass its bytes"
                ) from error
            return container_bytes(export())
        return container_bytes(export(artifact_id="composition-bundle-node"))
    raise CausalTypeError("an artifact is bytes, ExportedArtifact, or an exporting object")


def _controls(execution: Execution) -> dict[str, Any]:
    return native_controls(
        execution, seed="seed", threads="threads", cancel="cancel", memory="memory_bytes"
    )


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

    def add_artifact(
        self,
        kind: NodeKind | Literal["auto"] | object,
        data: object = _UNSET,
        *,
        node_id: str | None = None,
    ) -> BundleBuilder:
        """Embed an artifact from its container bytes (or an object that exports them).

        Call ``add_artifact(artifact)`` to infer the node kind from the container, or
        ``add_artifact(kind, artifact)`` to name it (``"auto"`` reads it from the container
        too). The artifact is decoded through its own consumer and its identity recomputed,
        so bytes that are not that kind of artifact refuse here with the failed stage.

        Raises:
            CausalTypeError: the artifact is neither container bytes nor an object whose
                ``export`` returns them, or ``kind`` is given without the artifact.
            CausalValueError: ``kind`` is not a bundle node kind.
            CompositionBundleRefusal: the bytes are not an artifact of that kind.
        """
        if data is _UNSET:
            if isinstance(kind, str):
                raise CausalTypeError(
                    "add_artifact(kind, data) needs the artifact after the kind; call "
                    "add_artifact(artifact) to infer the kind from the container"
                )
            kind, data = "auto", kind
        if not isinstance(kind, str):
            raise CausalTypeError(f"kind must be a node kind string, one of {NODE_KINDS}")
        added, refusal = self._native.add_artifact(
            _kind(kind, allow_auto=True), _bytes(data), node_id
        )
        raise_refusal(refusal, _refusal_type)
        if added is None:  # pragma: no cover - the native contract
            raise CausalValueError("the native builder returned neither a node nor a refusal")
        self._ids.append(added)
        self._last = added
        return self

    def add_reference(
        self,
        node_id: str,
        kind: NodeKind,
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

        Raises:
            CausalValueError: ``kind`` is not a bundle node kind.
            CausalTypeError: ``requires`` is not a requirement.
            CompositionBundleRefusal: the node id is already used or the reference is refused.
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
        raise_refusal(refusal, _refusal_type)
        self._ids.append(node_id)
        self._last = node_id
        return self

    def add_graph_imputation_transport_fit(
        self,
        artifact: GraphImputationDecisionArtifact,
        *,
        node_id: str,
        execution: Execution | None = None,
    ) -> BundleBuilder:
        """Embed a transported graph-imputation effect decision as a refittable fit node.

        ``artifact`` is the export of a ``transport`` route of
        :func:`antecedent.graph_imputation_decision.compile_graph_imputation_decision`. It is
        consumed under its program identity before it is added (the source reimputed, the
        graphs reselected, the target standardized and the decision replayed), and the replay
        seed (``execution.seed``) is recorded so a consumer refits it the same way. Connect it
        to the ``propagation_chain`` node its decision read
        (:meth:`~antecedent.graph_imputation_decision.CompiledGraphImputationDecision.chain_artifact`).

        Raises:
            CausalTypeError: ``artifact`` is not a graph-imputation decision artifact.
            CompositionBundleRefusal: the fit does not replay, or ``node_id`` is already used.
        """
        from .execution import resolve_execution
        from .graph_imputation_decision import GraphImputationDecisionArtifact

        if not isinstance(artifact, GraphImputationDecisionArtifact):
            raise CausalTypeError(
                "artifact must be the GraphImputationDecisionArtifact of a transported decision"
            )
        execution = resolve_execution(execution)
        refusal = execution._call(
            _native.graph_imputation_transport_bundle_add_fit,
            self._native,
            node_id,
            artifact.data,
            artifact.program,
            execution.seed,
        )
        raise_refusal(refusal, _refusal_type)
        self._ids.append(node_id)
        self._last = node_id
        return self

    def add_sequential_policy_reliability(
        self,
        reliability: SequentialPolicyReliability,
        *,
        producer: str = "sequential_producer",
        draws: str = "sequential_draws",
        diagnostic: str = "sequential_diagnostic",
        execution: Execution | None = None,
    ) -> BundleBuilder:
        """Embed a sequential policy reliability as three connected nodes.

        ``reliability`` comes from
        :func:`antecedent.decision_reliability.sequential_policy_reliability`. ``producer``
        holds its sealed raw declaration, ``draws`` its aligned decision draws and
        ``diagnostic`` its report on those draws, connected producer to draws to diagnostic.
        Each node is verified by recomputation before it is accepted: the producer is
        re-evaluated under the seed it was computed with and the diagnostic is reassessed from
        the draws. On a refusal the builder is unchanged. ``execution`` supplies threads,
        limits and cancellation. :func:`compile_sequential_reliability_node_bundle` recomputes
        the three nodes of the built bundle.

        Raises:
            CausalTypeError: ``reliability`` is not a computed sequential policy reliability.
            CompositionBundleRefusal: a node id already in use, or a node that does not verify.
        """
        native = getattr(reliability, "_native", None)
        if not isinstance(native, _native.SequentialPolicyReliabilityHandle):
            raise CausalTypeError(
                "reliability must be a SequentialPolicyReliability computed in this process by "
                "decision_reliability.sequential_policy_reliability"
            )
        ids = (_name(producer, "producer"), _name(draws, "draws"), _name(diagnostic, "diagnostic"))
        execution = resolve_execution(execution)
        refusal = execution._call(
            _native.composition_add_sequential_policy_reliability,
            self._native,
            native,
            *ids,
            execution.cancel,
        )
        raise_refusal(refusal, _refusal_type)
        self._ids.extend(ids)
        self._last = ids[-1]
        return self

    def _require_nodes(self, *node_ids: str) -> None:
        for node_id in node_ids:
            if node_id not in self._ids:
                raise NodeNotFoundRefusal(
                    {
                        "code": "invalid_argument",
                        "stage": "build",
                        "detail": _reason_details.composition_bundle.NODE_NOT_FOUND,
                        "offending": node_id,
                        "message": f"no node `{node_id}` has been added to this builder",
                        "remedy": f"add `{node_id}` first; nodes so far: {sorted(self._ids)}",
                    }
                )

    def connect(self, upstream: str, dependent: str) -> BundleBuilder:
        """Declare that ``dependent`` was derived from ``upstream``.

        Raises:
            NodeNotFoundRefusal: either node has not been added (also a
                :class:`~antecedent.errors.CausalValueError`).
            EdgeDigestMismatchRefusal: the edge would make the graph malformed (a cycle).
        """
        self._require_nodes(upstream, dependent)
        raise_refusal(self._native.connect(upstream, dependent), _refusal_type)
        return self

    def relate(self, left: str, right: str, relationship: Relationship) -> BundleBuilder:
        """Declare how the evidence behind two nodes depends on each other.

        The declaration becomes a node both are connected to, and is checked against
        them: a pair declared ``independent`` that rests on the same evidence is
        refused as ``swapped_evidence``. The pair is unordered.

        Raises:
            CausalValueError: ``relationship`` is not one of :data:`RELATIONSHIPS`.
            NodeNotFoundRefusal: either node has not been added.
            SwappedEvidenceRefusal: the declaration contradicts the evidence.
        """
        if relationship not in RELATIONSHIPS:
            raise CausalValueError(f"`{relationship}` is not one of {RELATIONSHIPS}")
        self._require_nodes(left, right)
        added, refusal = self._native.relate(left, right, relationship)
        raise_refusal(refusal, _refusal_type)
        if added is None:  # pragma: no cover - the native contract
            raise CausalValueError("the native builder returned neither a node nor a refusal")
        self._ids.append(added)
        self._last = added
        return self

    def add_predictive_fit(
        self,
        kind: PredictiveFitKind,
        artifact: object,
        *,
        program: str,
        node_id: str,
        execution: Execution | None = None,
    ) -> BundleBuilder:
        """Embed a native predictive producer: its exported artifact and its program identity.

        ``kind`` is one of :data:`PREDICTIVE_FIT_KINDS` (a joint transport, a trial IPW transport,
        a cyclic target-feedback fit, a graph-imputation fit or an assessed scientific bridge
        posterior) and ``program`` the identity of the producer's compiled program. The producer
        is reidentified and refitted from its raw evidence under ``execution`` before the node is
        accepted. The execution's seed is recorded with the node and every replay refits under
        it, so pass the execution the producer ran under. The producer's chain and its
        :meth:`add_predictive_chain_decision` terminal follow it by :meth:`connect` edges, and
        :func:`compile_predictive_fit_decision_bundle` recomputes the path.

        Raises:
            CausalValueError: ``kind`` is not a predictive fit kind or ``program`` is not a
                64-hex identity.
            CausalTypeError: the artifact is neither container bytes nor an exporting object.
            CompositionBundleRefusal: the node id is used, or the artifact does not replay as that
                producer under ``program`` (``swapped_evidence``).
        """
        if kind not in PREDICTIVE_FIT_KINDS:
            raise CausalValueError(
                f"`{kind}` is not a predictive fit kind; use one of {PREDICTIVE_FIT_KINDS}"
            )
        if not isinstance(program, str) or not isinstance(node_id, str):
            raise CausalTypeError("program and node_id must be strings")
        policy = resolve_execution(execution)
        refusal = policy._call(
            self._native.add_predictive_fit,
            kind,
            node_id,
            _bytes(artifact),
            program,
            **_controls(policy),
        )
        raise_refusal(refusal, _refusal_type)
        self._ids.append(node_id)
        self._last = node_id
        return self

    def add_predictive_chain_decision(
        self,
        decision: object,
        *,
        node_id: str,
        execution: Execution | None = None,
    ) -> BundleBuilder:
        """Embed a chain decision as a terminal that is recompiled and scored live.

        ``decision`` is a :class:`~antecedent.propagation_chain.ChainDecision` (or its exported
        bytes) over a ``predictive`` chain law: an action-world decision follows a joint
        transport, trial IPW, cyclic target-feedback or graph-imputation producer, a plain one an
        assessed bridge posterior. The decision is consumed independently before the node is
        accepted; the execution's seed is recorded with the node.

        Raises:
            CausalTypeError: ``decision`` is neither container bytes nor an exporting object.
            CompositionBundleRefusal: the node id is used, or the artifact does not replay as a
                predictive chain decision (``swapped_evidence``).
        """
        if not isinstance(node_id, str):
            raise CausalTypeError("node_id must be a string")
        policy = resolve_execution(execution)
        refusal = policy._call(
            self._native.add_predictive_chain_decision,
            node_id,
            _bytes(decision),
            **_controls(policy),
        )
        raise_refusal(refusal, _refusal_type)
        self._ids.append(node_id)
        self._last = node_id
        return self

    def build(self) -> Bundle:
        """Seal the graph into a :class:`Bundle`.

        Raises:
            CompositionBundleRefusal: the graph is malformed or exceeds a bound.
        """
        built, refusal = self._native.build()
        raise_refusal(refusal, _refusal_type)
        if built is None:  # pragma: no cover - the native contract
            raise CausalValueError("the native builder returned neither a bundle nor a refusal")
        return Bundle(built)


@dataclass(frozen=True)
class NodeSummary:
    """One node of a built bundle."""

    id: str
    kind: NodeKind
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

    @artifact_export
    def export(
        self, *, artifact_id: str = "composition-bundle"
    ) -> ExportedArtifact[ConsumedBundle]:
        """Serialize through the checksummed container (one section per embedded node).

        Raises:
            CompositionBundleRefusal: the bundle exceeds a size or count bound.
        """
        data, refusal = self._native.export(artifact_id)
        raise_refusal(refusal, _refusal_type)
        if data is None:  # pragma: no cover - the native contract
            raise CausalValueError("the native export returned neither a result nor a refusal")
        return artifact_from_receipt(self, data)

    def __repr__(self) -> str:
        return (
            f"Bundle(identity={self.identity[:12]!r}, nodes={len(self.nodes)}, "
            f"edges={len(self.edges)})"
        )

    def explain(self) -> str:
        """A plain-text account of the nodes and the dependency edges."""
        lines = [
            f"Bundle {self.identity[:12]}: {len(self.nodes)} node(s), {len(self.edges)} edge(s)."
        ]
        for node in self.nodes:
            how = "embedded" if node.embedded else "reference (needs a supplied source)"
            lines.append(f"- {node.id} ({node.kind}): {how}")
        for edge in self.edges:
            lines.append(f"- {edge.source} -> {edge.target}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready mapping of the identity, nodes and edges."""
        return {
            "identity": self.identity,
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [
                {"from": e.source, "to": e.target, "upstream_digest": e.upstream_digest}
                for e in self.edges
            ],
        }


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


@dataclass(repr=False, frozen=True)
class ConsumedNode(ScientificResult):
    """One consumed node."""

    id: str
    kind: NodeKind
    identity: str
    chain_digest: str
    status: NodeStatus
    facts: Mapping[str, str] = field(default_factory=dict)
    values: Mapping[str, float] = field(default_factory=dict)
    inspected: Mapping[str, float] = field(default_factory=dict)
    claim_label: ClaimLabel | None = None
    replayed: bool = False

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


def _status_dict(status: NodeStatus) -> dict[str, Any]:
    if isinstance(status, Failed):
        return {
            "state": "failed",
            "stage": status.stage,
            "reason": status.reason,
            "detail": status.detail,
            "code": status.code,
        }
    if isinstance(status, ReferenceUnresolved):
        return {"state": "reference_unresolved", "requires": asdict(status.requires)}
    return {"state": "verified"}


def _label(value: str | None) -> ClaimLabel | None:
    if value == "point_only_attested":
        return "point_only_attested"
    if value == "joint_draw":
        return "joint_draw"
    return None


class ConsumedBundle(ScientificResult):
    """A consumed bundle: every node's status, with inspected results preserved."""

    def __init__(self, wire: Mapping[str, Any]) -> None:
        self._identity: str = wire["identity"]
        self._all_verified: bool = wire["all_verified"]
        self._all_replayed: bool = wire["all_replayed"]
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
                replayed=n["replayed"],
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
        """One node by id.

        Raises:
            CausalValueError: the bundle has no node with that id.
        """
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

    def __repr__(self) -> str:
        verified = sum(1 for node in self._nodes if node.verified)
        return (
            f"ConsumedBundle(identity={self._identity[:12]!r}, nodes={len(self._nodes)}, "
            f"verified={verified}, claim_label={self.claim_label!r})"
        )

    def explain(self) -> str:
        """A plain-text account of every node's status and the claim the bundle carries."""
        lines = [
            f"Consumed bundle {self._identity[:12]}: {len(self._nodes)} node(s), "
            f"{len(self._edges)} edge(s), claim label {self.claim_label!r}."
        ]
        for node in self._nodes:
            status = node.status
            if isinstance(status, Failed):
                state = f"FAILED at {status.stage}: {status.reason}"
            elif isinstance(status, ReferenceUnresolved):
                state = "reference unresolved (provider or data source not supplied)"
            else:
                state = "verified"
            lines.append(f"- {node.id} ({node.kind}): {state}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready mapping of the identity, nodes (with status) and edges."""
        return {
            "identity": self._identity,
            "claim_label": self.claim_label,
            "all_verified": self.all_verified,
            "all_replayed": self._all_replayed,
            "nodes": [
                {
                    "id": node.id,
                    "kind": node.kind,
                    "identity": node.identity,
                    "chain_digest": node.chain_digest,
                    "status": _status_dict(node.status),
                    "replayed": node.replayed,
                    "facts": dict(node.facts),
                    "values": dict(node.values),
                    "inspected": dict(node.inspected),
                    "claim_label": node.claim_label,
                }
                for node in self._nodes
            ],
            "edges": [
                {"from": e.source, "to": e.target, "upstream_digest": e.upstream_digest}
                for e in self._edges
            ],
        }

    @property
    def all_replayed(self) -> bool:
        """Native receipt: every node passed its embedded artifact consumer."""
        return self._all_replayed

    def require_verified(self) -> None:
        """Require every node to have verified.

        Raises:
            CompositionBundleRefusal: the first failed node's typed refusal at its stage, or,
                for the first unresolved reference, :class:`CallbackUnavailableRefusal`.
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
                        "detail": _reason_details.composition_bundle.CALLBACK_UNAVAILABLE,
                        "offending": node.id,
                        "supplied": "a node needs a provider or data source that was not supplied",
                    }
                )


def _bundle_acceptance(result: ConsumedBundle, identity_bound: bool) -> ArtifactAcceptance:
    unresolved = tuple(
        node.id
        + ":"
        + (node.status.stage if isinstance(node.status, Failed) else "reference_unresolved")
        for node in result.nodes
        if not node.verified
    )
    return ArtifactAcceptance(
        "replayed" if result._all_verified else "unavailable",
        identity_bound,
        result._all_replayed,
        False,
        proof="BundleConsumer::consume runtime node replay receipts",
        unresolved=unresolved,
    )


@outcome_operation
@artifact_consumer(
    proof="BundleConsumer::consume runtime node replay receipts",
    acceptance=_bundle_acceptance,
    input="data",
)
def consume_bundle(
    data: bytes,
    *,
    expected_identity: str | None = None,
    supplied: SuppliedSources | None = None,
    actual_claims: Mapping[str, object] | None = None,
    max_bundle_bytes: int | None = None,
    max_nodes: int | None = None,
    max_edges: int | None = None,
    max_embedded_bytes: int | None = None,
) -> ConsumedBundle:
    """Consume exported bundle bytes under the identity the consumer retained.

    The container, version, node kinds, node and edge digests and the bundle
    identity are checked first and refuse as a typed
    :class:`CompositionBundleRefusal`; each embedded artifact is then verified
    through its own consumer, and a node-level failure is reported on that node
    without hiding the rest of the bundle. ``actual_claims`` binds original source
    digests to opaque claims from actual executions; each source's original engine
    executes once to resolve its native operation. Serialized metadata is insufficient.

    ``max_bundle_bytes``, ``max_nodes``, ``max_edges`` and ``max_embedded_bytes`` are the
    consumer's own bounds (each omitted one keeps its default); a bundle beyond any of them
    refuses as :class:`OversizedRefusal` before a node is verified. They are not available
    together with ``actual_claims``.

    ``expected_identity`` is required: the bundle identity (``Bundle.identity``) the consumer
    retained independently of the bytes. It is a keyword so a forgotten identity is named in
    the error rather than silently trusting the bytes.

    Raises:
        CausalTypeError: ``data`` is not bytes (for example a path or ``str``: read the file
            with ``Path.read_bytes()``), or ``expected_identity`` is not a string.
        CausalValueError: ``expected_identity`` was not given.
        CompositionBundleRefusal: the container does not verify (see the stage-specific
            subclasses, for example :class:`ExpectedIdentityMismatchRefusal`).
    """
    data = container_bytes(data)
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise CausalTypeError(
            f"consume_bundle expects the exported bundle bytes, not {type(data).__name__}; "
            "read a file with Path.read_bytes() or pass Bundle.export()"
        )
    if expected_identity is None:
        raise CausalValueError(
            "consume_bundle needs expected_identity=: the bundle identity you retained "
            "independently of the bytes (for example the Bundle.identity recorded at export)"
        )
    if not isinstance(expected_identity, str):
        raise CausalTypeError("expected_identity must be the bundle identity string")
    bounds = {
        name: count(value, name, minimum=1)
        for name, value in (
            ("max_bundle_bytes", max_bundle_bytes),
            ("max_nodes", max_nodes),
            ("max_edges", max_edges),
            ("max_embedded_bytes", max_embedded_bytes),
        )
        if value is not None
    }
    if actual_claims is None:
        wire, refusal = _native.consume_composition_bundle(
            bytes(data),
            expected_identity,
            None if supplied is None else supplied._json(),
            **bounds,
        )
    else:
        if bounds:
            raise CausalValueError("consumer bounds are not available with actual_claims")
        from .program_claims import NativeClaim

        if len(actual_claims) > 64:
            raise CausalValueError("source resolver count exceeds its bound")
        handles: dict[str, _native.NativeResponseClaim] = {}
        for digest, claim in actual_claims.items():
            if not isinstance(digest, str) or not isinstance(claim, NativeClaim):
                raise CausalTypeError(
                    "source resolution requires source digest to issued NativeClaim mapping"
                )
            handles[digest] = claim._native
        wire, refusal = _native.consume_source_projection_bundle(
            bytes(data), expected_identity, None if supplied is None else supplied._json(), handles
        )
    raise_refusal(refusal, _refusal_type)
    if wire is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native consumer returned neither a result nor a refusal")
    return ConsumedBundle(json.loads(wire))


@dataclass(frozen=True)
class ArtifactDescription:
    """What an artifact's own consumer establishes about it, read from its bytes."""

    kind: str
    identity: str
    facts: Mapping[str, str]
    values: Mapping[str, float]


def describe_artifact(kind: NodeKind | Literal["auto"], data: object) -> ArtifactDescription:
    """Decode one artifact on its own and report its identity and published facts.

    ``kind`` is a node kind, or ``"auto"`` to read it from the container.

    Raises:
        CausalValueError: ``kind`` is not a bundle node kind.
        CausalTypeError: ``data`` is neither container bytes nor an object whose ``export``
            returns them.
        CompositionBundleRefusal: the bytes are not an artifact of that kind (the typed
            refusal of the artifact's own consumer).
    """
    wire, refusal = _native.composition_describe_artifact(
        _kind(kind, allow_auto=True), _bytes(data)
    )
    raise_refusal(refusal, _refusal_type)
    if wire is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native describe returned neither a result nor a refusal")
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

    Raises:
        CausalTypeError: ``claim`` is neither container bytes nor an exporting object.
        CompositionBundleRefusal: the contract needs more than a mean (``unsupported_law``).
    """
    data, refusal = _native.composition_mean_decision(
        json.dumps(contract._wire()), _bytes(claim), artifact_id
    )
    raise_refusal(refusal, _refusal_type)
    if data is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native decision returned neither a result nor a refusal")
    return data


def detect_kind(data: object) -> str | None:
    """The node kind a container's artifact fills, or ``None`` when it is not one.

    Raises:
        CausalTypeError: ``data`` is neither container bytes nor an exporting object.
    """
    return _native.composition_detect_kind(_bytes(data))


# ------------------------------------------------------------------- recomputation


def _recomputed(pair: tuple[Any, str | None]) -> Any:
    value, refusal = pair
    raise_refusal(refusal, _refusal_type)
    if value is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native recomputation returned neither a result nor a refusal")
    return value


@dataclass(frozen=True)
class ReplayedDependency:
    """One embedded dependency that passed its own artifact consumer in this run."""

    node_id: str
    artifact_identity: str
    chain_digest: str


@dataclass(repr=False, frozen=True, slots=True)
class BundleRecomputation(ScientificResult):
    """The receipt of a bundle terminal executed live by its compiled program.

    Every embedded dependency passed its own consumer in this run (``dependencies``, in bundle
    order, with the recomputed identity and Merkle digest of each) and the terminal was executed
    by the program ``program_identity``. A ``chain`` route names the recomputed
    ``terminal_identity``; a ``predictive_decision`` or ``posterior_decision`` route the
    ``decision_fingerprint`` of the decision freshly scored on the bound source law; a
    ``predictive_fit_decision`` route the refitted ``chain_identity``, the live
    ``decision_fingerprint`` and the ``producer_program_identity``. Nothing outside the embedded
    evidence is refitted and no calibration is earned. ``replayed`` is ``True`` for a receipt an
    independent consumer recomputed from a portable artifact.
    """

    route: RecomputationRoute
    bundle_identity: str
    program_identity: str
    dependencies: tuple[ReplayedDependency, ...]
    terminal_node: str | None = None
    terminal_identity: str | None = None
    chain_identity: str | None = None
    decision_fingerprint: str | None = None
    producer_program_identity: str | None = None
    replayed: bool = False
    _handle: Any = field(repr=False, compare=False, default=None)

    @artifact_export
    def export(
        self, *, execution: Execution | None = None
    ) -> ExportedArtifact[BundleRecomputation]:
        """Re-execute, require this receipt, and embed the bundle with it for a fresh process.

        :func:`consume_bundle_recomputation` rebuilds the graph, recompiles the terminal program
        and requires the same live receipt; its expectation is :attr:`program_identity`. A
        replayed receipt forwards the bytes it was read from.

        Raises:
            CompositionBundleRefusal: the live receipt changed or the artifact exceeds its bound.
        """
        kind = f"{self.route}_bundle_recomputation"
        if self._handle is None:
            data = getattr(self, "_artifact_bytes", None)
            if data is None:
                raise CausalTypeError("this receipt retains neither its compiled program nor bytes")
        else:
            plan, run = self._handle
            policy = resolve_execution(execution)
            data = _recomputed(policy._call(plan.export_portable, run, **_controls(policy)))
        return ExportedArtifact(bytes(data), kind, ArtifactExpectation(kind, self.program_identity))


def _recomputation_of(run: Any, *, plan: Any) -> BundleRecomputation:
    body = json.loads(run.receipt_json())
    return BundleRecomputation(
        route=body["route"],
        bundle_identity=str(body["bundle_identity"]),
        program_identity=str(body["program_identity"]),
        dependencies=tuple(
            ReplayedDependency(
                str(item["node_id"]), str(item["artifact_identity"]), str(item["chain_digest"])
            )
            for item in body["dependencies"]
        ),
        terminal_node=body["terminal_node"],
        terminal_identity=body["terminal_identity"],
        chain_identity=body["chain_identity"],
        decision_fingerprint=body["decision_fingerprint"],
        producer_program_identity=body["producer_program_identity"],
        replayed=plan is None,
        _handle=None if plan is None else (plan, run),
    )


class CompiledBundleRecomputation:
    """A sealed bundle and its live terminal compiled into one six-layer program.

    The graph, every node identity and the terminal declaration are bound before anything runs.
    :attr:`route` names the terminal, :attr:`program_identity` is the identity a consumer retains
    with :attr:`bundle_identity`, :meth:`plan_view` shows every layer and :meth:`execute` replays
    the dependencies and runs the terminal.
    """

    def __init__(self, handle: Any, bundle_identity: str) -> None:
        self._handle = handle
        self._bundle_identity = bundle_identity

    @property
    def route(self) -> RecomputationRoute:
        """``chain``, ``predictive_decision``, ``posterior_decision`` or
        ``predictive_fit_decision``."""
        route: RecomputationRoute = self._handle.route
        return route

    @property
    def program_identity(self) -> str:
        """Identity of the compiled program (lowercase hex)."""
        return str(self._handle.program_identity)

    @property
    def bundle_identity(self) -> str:
        """Identity of the compiled bundle."""
        return self._bundle_identity

    def plan_view(self) -> dict[str, Any]:
        """Every layer of the compiled program, with nothing executed."""
        return dict(json.loads(self._handle.plan_view_json()))

    def unit_support(self) -> dict[str, Any] | None:
        """The support decision of the unit this compiled program runs under.

        ``{"unit", "supported", "missing_evidence"}``, read from the program's route; ``None``
        while no unit is bound to it.
        """
        return read_unit_support(self._handle)

    @outcome_operation
    def execute(self, *, execution: Execution | None = None) -> BundleRecomputation:
        """Replay every embedded dependency through its own consumer and execute the terminal.

        A reference node, a failed dependency or a terminal that differs from its live result
        refuses; a cancelled run raises :class:`~antecedent.errors.CausalCancelledError`.

        Raises:
            CompositionBundleRefusal: a dependency or the live terminal does not replay.
        """
        policy = resolve_execution(execution)
        run = _recomputed(policy._call(self._handle.execute, **_controls(policy)))
        return _recomputation_of(run, plan=self._handle)

    def __repr__(self) -> str:
        return (
            f"CompiledBundleRecomputation(route={self.route!r}, "
            f"program_identity={self.program_identity[:12]!r})"
        )


def _bundle(bundle: object) -> Bundle:
    bundle = require_success(bundle)
    if not isinstance(bundle, Bundle):
        raise CausalTypeError(f"bundle must be a built Bundle, not {type(bundle).__name__}")
    return bundle


def _node(value: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise CausalTypeError(f"{what} must be a node id")
    return value


@outcome_operation
def compile_chain_bundle_recomputation(
    bundle: Bundle,
    terminal: str,
    *,
    chain: PropagationChain | None = None,
    execution: Execution | None = None,
) -> CompiledBundleRecomputation:
    """Compile a fully embedded bundle with its propagation-chain terminal into one program.

    ``terminal`` is an embedded ``propagation_chain`` node. Its own record is compiled unless
    ``chain`` gives the live :class:`~antecedent.propagation_chain.PropagationChain` the terminal
    must equal; either way execution requires the embedded record and the live result to agree.

    Raises:
        CausalTypeError: ``bundle``, ``terminal`` or ``chain`` has the wrong type.
        CompositionBundleRefusal: the terminal is absent, a reference or not a chain, or the chain
            declaration refuses.
    """
    from .propagation_chain import PropagationChain

    built = _bundle(bundle)
    record = None
    if chain is not None:
        chain = require_success(chain)
        if not isinstance(chain, PropagationChain):
            raise CausalTypeError("chain must be a PropagationChain")
        record = chain._record
    policy = resolve_execution(execution)
    handle = _recomputed(
        policy._call(
            _native.chain_bundle_recomputation_compile,
            built._native,
            _node(terminal, "terminal"),
            record,
            cancel=policy.cancel,
        )
    )
    return CompiledBundleRecomputation(handle, built.identity)


@outcome_operation
def compile_decision_bundle_recomputation(
    bundle: Bundle,
    terminal: str,
    *,
    contract: Contract,
    source: JointDistributionArtifact,
    carrier: Literal["predictive", "parameter_posterior"] = "predictive",
) -> CompiledBundleRecomputation:
    """Compile a bundle whose terminal decision result is rescored live on its source law.

    ``terminal`` is an embedded ``decision_result`` node evaluated from ``contract`` on
    ``source``, a jointly aligned :class:`~antecedent.joint_distribution.JointDistributionArtifact`.
    ``carrier="predictive"`` needs a predictive outcome law; ``"parameter_posterior"`` a
    parameter-posterior law, an expected-utility or expected-loss criterion and no ``outcome``
    functional, because posterior parameter draws are not outcome risk. The contract, the source,
    the graph and the node all enter the program identity before any utility is scored. A
    portable consumer reconstructs the contract and the source from the bundle's own
    ``decision_contract`` and ``distribution`` nodes.

    Raises:
        CausalTypeError: an argument has the wrong type.
        CausalValueError: ``carrier`` is not ``predictive`` or ``parameter_posterior``.
        CompositionBundleRefusal: the terminal is absent or a reference, the contract is invalid,
            or the law cannot carry the decision (``decision_contract_unsatisfied``).
    """
    built = _bundle(bundle)
    contract = require_success(contract)
    source = require_success(source)
    if not isinstance(contract, Contract):
        raise CausalTypeError("contract must be a decision.Contract")
    if not isinstance(source, JointDistributionArtifact):
        raise CausalTypeError("source must be a JointDistributionArtifact")
    if carrier not in ("predictive", "parameter_posterior"):
        raise CausalValueError("carrier must be predictive or parameter_posterior")
    handle = _recomputed(
        _native.decision_bundle_recomputation_compile(
            built._native,
            _node(terminal, "terminal"),
            json.dumps(contract._wire()),
            source._native,
            posterior=carrier == "parameter_posterior",
        )
    )
    return CompiledBundleRecomputation(handle, built.identity)


@outcome_operation
def compile_predictive_fit_decision_bundle(
    bundle: Bundle,
    *,
    producer: str,
    chain: str,
    terminal: str,
    execution: Execution | None = None,
) -> CompiledBundleRecomputation:
    """Compile a native producer -> predictive chain -> live decision path of a bundle.

    ``producer`` is a node added with :meth:`BundleBuilder.add_predictive_fit`; its kind selects
    the producer's compiler. ``chain`` is the embedded ``propagation_chain`` the producer's
    decision retained and ``terminal`` the :meth:`BundleBuilder.add_predictive_chain_decision`
    node, joined by the edges ``producer -> chain -> terminal``. Execution refits the producer
    from its raw evidence, executes the predictive chain and rescores the decision; it earns no
    calibration.

    Raises:
        CausalTypeError: ``bundle`` or a node id has the wrong type.
        CompositionBundleRefusal: the producer is not a predictive fit node, an edge is missing,
            a node is a reference or of another kind, or the decision and chain differ.
    """
    built = _bundle(bundle)
    policy = resolve_execution(execution)
    handle = _recomputed(
        policy._call(
            _native.predictive_fit_decision_bundle_compile,
            built._native,
            _node(producer, "producer"),
            _node(chain, "chain"),
            _node(terminal, "terminal"),
            **_controls(policy),
        )
    )
    return CompiledBundleRecomputation(handle, built.identity)


# ------------------------------------------------------------ sequential reliability bundle

_SEQUENTIAL_KIND = "sequential_reliability_bundle_v1"


@dataclass(repr=False, frozen=True)
class SequentialReliabilityBundleReceipt(ScientificResult):
    """A recomputed producer, draw carrier and diagnostic of a sequential reliability bundle.

    ``bundle_identity`` is the bundle that was consumed, ``draws_digest`` and
    ``diagnostic_digest`` identify the retained decision draws and the live reassessment of
    them, and ``producer_program_identity`` the producer program that was re-evaluated.
    ``program_identity`` is the compiled recomputation a consumer retains to replay it, and
    ``dependencies`` lists every replayed node as ``node_id``, ``artifact_identity`` and
    ``chain_digest``. The diagnostic is a point diagnostic: no calibration is claimed.
    ``acceptance`` is set on a replayed receipt.
    """

    bundle_identity: str
    draws_digest: str
    diagnostic_digest: str
    producer_program_identity: str
    program_identity: str
    dependencies: tuple[Mapping[str, str], ...]
    acceptance: ArtifactAcceptance | None = field(default=None, compare=False)
    _native: Any = field(default=None, repr=False, compare=False)
    _artifact_bytes: bytes | None = field(default=None, repr=False, compare=False)

    def unit_support(self) -> dict[str, Any] | None:
        """The support decision of the unit the compiled program runs under.

        ``{"unit", "supported", "missing_evidence"}``, read from the program's route; ``None``
        for a result replayed from an artifact, which retains no program.
        """
        return read_unit_support(self._native)

    @property
    def route(self) -> str | None:
        """The compiled route of a receipt computed here; ``None`` for a replayed one."""
        return None if self._native is None else str(self._native.route)

    def export(
        self, *, execution: Execution | None = None
    ) -> ExportedArtifact[SequentialReliabilityBundleReceipt]:
        """The bundle and this receipt, sealed for
        :func:`consume_portable_sequential_reliability_node_bundle`.

        The bundle is recomputed again and must reproduce this receipt before it is sealed. A
        replayed receipt exports the bytes it was consumed from. The expectation carries the
        bundle identity and :attr:`program_identity`.
        """
        expectation = ArtifactExpectation(
            _SEQUENTIAL_KIND,
            {"bundle_identity": self.bundle_identity, "program_identity": self.program_identity},
        )
        if self._native is None:
            if self._artifact_bytes is None:
                raise CausalValueError("this receipt holds neither a program nor its artifact")
            return ExportedArtifact(self._artifact_bytes, _SEQUENTIAL_KIND, expectation)
        execution = resolve_execution(execution)
        data, refusal = execution._call(self._native.export, execution.cancel)
        raise_refusal(refusal, _refusal_type)
        assert data is not None
        return ExportedArtifact(bytes(data), _SEQUENTIAL_KIND, expectation)


def _sequential_receipt(text: str, native: Any = None) -> SequentialReliabilityBundleReceipt:
    wire = json.loads(text)
    return SequentialReliabilityBundleReceipt(
        bundle_identity=wire["bundle_identity"],
        draws_digest=wire["draws_digest"],
        diagnostic_digest=wire["diagnostic_digest"],
        producer_program_identity=wire["producer_program_identity"],
        program_identity=wire["program_identity"],
        dependencies=tuple(wire["dependencies"]),
        _native=native,
    )


@outcome_operation
def compile_sequential_reliability_node_bundle(
    bundle: Bundle,
    *,
    producer: str = "sequential_producer",
    draws: str = "sequential_draws",
    diagnostic: str = "sequential_diagnostic",
    execution: Execution | None = None,
) -> SequentialReliabilityBundleReceipt:
    """Compile and recompute the producer, draws and diagnostic nodes of a built bundle.

    The three nodes are those :meth:`BundleBuilder.add_sequential_policy_reliability` embeds:
    an embedded sequential producer, its decision draws and its diagnostic, joined by direct
    producer to draws and draws to diagnostic edges. The compiled program consumes the whole
    bundle, replays every node, re-evaluates the producer and reassesses the diagnostic from
    the draws; the producer, the carrier and the live diagnostic must agree.

    Raises:
        CausalTypeError: ``bundle`` is not a :class:`Bundle`.
        CompositionBundleRefusal: a missing, misplaced or non-embedded node or edge
            (``edge_digest_mismatch``), a node that does not replay or a producer, carrier and
            diagnostic that disagree (``swapped_evidence``).
    """
    if not isinstance(bundle, Bundle):
        raise CausalTypeError("bundle must be a built composition_bundle.Bundle")
    execution = resolve_execution(execution)
    handle, refusal = execution._call(
        _native.sequential_reliability_bundle_compile,
        bundle._native,
        _name(producer, "producer"),
        _name(draws, "draws"),
        _name(diagnostic, "diagnostic"),
        execution.seed,
        execution.cancel,
    )
    raise_refusal(refusal, _refusal_type)
    assert handle is not None
    return _sequential_receipt(handle.receipt_json, native=handle)


@outcome_operation
@artifact_consumer(
    proof="composition_bundle.consume_bundle_recomputation: rebuilds the portable bundle under the "
    "retained bundle identity, recompiles its terminal program under the retained program "
    "identity, replays every dependency, executes the terminal and requires the stored receipt",
    replayable=True,
    identity=("expected_bundle", "expected_program"),
)
def consume_bundle_recomputation(
    artifact: bytes,
    *,
    route: RecomputationRoute,
    expected_bundle: str,
    expected_program: str,
    execution: Execution | None = None,
) -> BundleRecomputation:
    """Independently consume a portable bundle recomputation.

    The bundle is rebuilt under ``expected_bundle``, the terminal's program recompiled and
    compared with ``expected_program``, every dependency replayed by its own consumer and the
    terminal executed again; the live receipt must equal the stored one. ``route`` is the
    :attr:`BundleRecomputation.route` the producer handed over with the two identities: a
    ``predictive_decision`` or ``posterior_decision`` rebuilds the contract and the source law from
    the bundle's own nodes, and a ``predictive_fit_decision`` refits its producer.

    Raises:
        CausalValueError: ``route`` is not one of :data:`RECOMPUTATION_ROUTES`, or
            ``expected_program`` is not a 64-hex identity.
        CompositionBundleRefusal: a foreign feature, a changed graph or program
            (``external_binding_mismatch``), a failed dependency or a changed live terminal.
    """
    if route not in RECOMPUTATION_ROUTES:
        raise CausalValueError(f"route must be one of {RECOMPUTATION_ROUTES}")
    if not isinstance(expected_bundle, str) or not isinstance(expected_program, str):
        raise CausalTypeError("expected_bundle and expected_program must be identity strings")
    data = container_bytes(artifact)
    policy = resolve_execution(execution)
    controls = _controls(policy)
    if route == "chain":
        pair = policy._call(
            _native.chain_bundle_recomputation_consume,
            data,
            expected_bundle,
            expected_program,
            **controls,
        )
    elif route == "predictive_fit_decision":
        pair = policy._call(
            _native.predictive_fit_decision_bundle_consume,
            data,
            expected_bundle,
            expected_program,
            **controls,
        )
    else:
        pair = policy._call(
            _native.decision_bundle_recomputation_consume,
            data,
            expected_bundle,
            expected_program,
            posterior=route == "posterior_decision",
            **controls,
        )
    return _recomputation_of(_recomputed(pair), plan=None)


@outcome_operation
@artifact_consumer(
    proof="composition_bundle.consume_portable_sequential_reliability_node_bundle: the native "
    "consumer rebuilds the bundle under the retained identity, recompiles the recomputation "
    "under the retained program identity, re-evaluates the producer and compares the receipt",
    replayable=True,
    identity=("expected_bundle", "expected_identity"),
)
def consume_portable_sequential_reliability_node_bundle(
    artifact: bytes | ExportedArtifact[Any],
    *,
    expected_bundle: str,
    expected_identity: str,
    execution: Execution | None = None,
) -> SequentialReliabilityBundleReceipt:
    """Independently reconstruct and recompute a sealed sequential reliability bundle.

    ``expected_bundle`` is the bundle identity and ``expected_identity`` the
    :attr:`SequentialReliabilityBundleReceipt.program_identity`, both retained beside the
    bytes. The bundle is rebuilt and recompiled, the producer is re-evaluated and the live
    receipt must equal the sealed one.

    Raises:
        CausalTypeError: ``artifact`` is not bytes or an exported artifact.
        CausalValueError: ``expected_identity`` is not 64 hexadecimal characters.
        ExpectedIdentityMismatchRefusal: another bundle identity, program identity or receipt.
        CompositionBundleRefusal: a node that does not replay.
    """
    execution = resolve_execution(execution)
    text, refusal = execution._call(
        _native.sequential_reliability_bundle_consume,
        container_bytes(artifact),
        _name(expected_bundle, "expected_bundle"),
        _name(expected_identity, "expected_identity"),
        execution.seed,
        execution.cancel,
    )
    raise_refusal(refusal, _refusal_type)
    assert text is not None
    # The consumer decorator records the acceptance and the consumed bytes on the result.
    return _sequential_receipt(text)


__all__ = [
    "EMBEDDABLE_KINDS",
    "NODE_KINDS",
    "PREDICTIVE_FIT_KINDS",
    "RECOMPUTATION_ROUTES",
    "RELATIONSHIPS",
    "STAGES",
    "ArtifactDescription",
    "Bundle",
    "BundleBuilder",
    "BundleRecomputation",
    "CallbackUnavailableRefusal",
    "ClaimLabel",
    "CompiledBundleRecomputation",
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
    "NodeNotFoundRefusal",
    "NodeStatus",
    "NodeSummary",
    "OversizedRefusal",
    "PredictiveFitKind",
    "ProviderRequestChangedRefusal",
    "ProviderRequirement",
    "RecomputationRoute",
    "ReferenceUnresolved",
    "Relationship",
    "ReplayedDependency",
    "Requirement",
    "SequentialReliabilityBundleReceipt",
    "SuppliedSources",
    "SwappedEvidenceRefusal",
    "TamperedQuantityRefusal",
    "UnknownNodeKindRefusal",
    "UnsupportedLawRefusal",
    "Verified",
    "compile_chain_bundle_recomputation",
    "compile_decision_bundle_recomputation",
    "compile_predictive_fit_decision_bundle",
    "compile_sequential_reliability_node_bundle",
    "consume_bundle",
    "consume_bundle_recomputation",
    "consume_portable_sequential_reliability_node_bundle",
    "describe_artifact",
    "detect_kind",
    "mean_decision",
]
