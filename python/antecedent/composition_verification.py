"""End-to-end composition verification: which downstream claims survive a change upstream.

A composed analysis rests on a cohort and its data, a measurement model, a mechanism composition, a
bridge, ledger results, a transported result and the decision that reads them. When one of them
changes, :func:`verify_composition` classifies every downstream claim over one claim-level
dependency graph::

    from antecedent import composition_verification as cv

    graph = cv.CompositionGraph(
        nodes=[
            cv.Node("cohort", "cohort_source", identity=cv.digest("cohort", "v1")),
            cv.Node("effect", "claim", identity=cv.digest("effect", "v1")),
            cv.Node("unrelated", "claim", identity=cv.digest("unrelated")),
        ],
        edges=[cv.Edge("effect", "cohort", "data")],
    )
    changed = graph.replace_identity("cohort", cv.digest("cohort", "v2"))
    result = cv.verify_composition(changed, previous=graph)
    result.classify("effect").klass       # "recompute"
    result.classify("unrelated").klass    # "still_valid"

What each class means:

* ``still_valid``: nothing it rests on changed and no license it states was lost; the earlier
  numbers and uncertainty stand.
* ``recompute``: data, evidence, a model or representation version or a structural derivation it
  uses changed. The earlier fit and uncertainty carry no authority; reusing a structural derivation
  after compatible data replacement recomputes and never retains the old fit.
* ``withdrawn``: an input it cannot do without was retired, coordinates it answers are outside the
  support, a provider's trust no longer holds, a stage cannot be executed, or the calibration it
  states does not hold at its coordinate. Numerical invalidation and withdrawal are reported
  apart (:attr:`ClaimClassification.numerical`, :attr:`ClaimClassification.withdrawals`).

A calibration record held at one coordinate is never carried to another: a claim that inherits a
record from an upstream object that no longer holds it, or a transported claim retargeted to
another population, is refused with ``cell_not_licensed``. The module measures no coverage and
reports no interval or posterior; a calibration declaration names a record identity and a
coordinate that the producer supplies.

Observed effects come from the evidence ledger (:func:`ledger_effects`), a recalculation plan or
claim freshness (declared with :class:`Effect`). The classification, lineage, identities, artifact
and refusals are computed in Rust; this module builds the declarations and presents the results.
Refusals are :class:`CompositionVerificationRefusal` (a
:class:`~antecedent.errors.CausalUnsupportedError`) with a registered ``reason_code``, a
``composition_verification.*`` ``detail`` and a ``remedy``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from . import _native
from ._refusal import StructuredRefusal, raise_refusal
from .errors import CausalTypeError, CausalValueError

__all__ = [
    "CALIBRATION_STATES",
    "CLASSES",
    "DEPENDENCY_KINDS",
    "NODE_KINDS",
    "Calibration",
    "ClaimClassification",
    "CompositionGraph",
    "CompositionVerification",
    "CompositionVerificationRefusal",
    "CompositionView",
    "Edge",
    "Effect",
    "Node",
    "SharedData",
    "consume_composition_verification_artifact",
    "digest",
    "ledger_effects",
    "shared_data_from_ledger",
    "verify_composition",
]

#: What a node can stand for.
NODE_KINDS: tuple[str, ...] = (
    "cohort_source",
    "data",
    "measurement_model",
    "mechanism_composition",
    "bridge",
    "ledger_result",
    "transported_result",
    "decision",
    "external_provider",
    "selection_diagram",
    "target_population",
    "claim",
)

#: How a node depends on another.
DEPENDENCY_KINDS: tuple[str, ...] = (
    "data",
    "evidence",
    "model_version",
    "support",
    "provider_trust",
    "calibration",
    "structural_derivation",
)

#: The classification of a claim.
CLASSES: tuple[str, ...] = ("still_valid", "recompute", "withdrawn")

#: The state of a claim's uncertainty guarantee.
CALIBRATION_STATES: tuple[str, ...] = (
    "not_applicable",
    "held",
    "inherited",
    "lost",
    "required",
)

_TRUST = ("not_applicable", "held", "withdrawn")
_HEX = re.compile(r"\A[0-9a-fA-F]{64}\Z")
_DEFAULT_ARTIFACT_ID = "composition_verification"


class CompositionVerificationRefusal(StructuredRefusal):
    """A typed refusal of composition verification.

    A :class:`~antecedent.errors.CausalUnsupportedError` with a registered ``reason_code``
    (``claim_revalidation_required``, ``claim_license_withdrawn``, ``cell_not_licensed``,
    ``invalid_argument``, ``route_not_supported``) and a ``composition_verification.*``
    ``detail``; ``remedy`` names what would license the request.
    """


# --------------------------------------------------------------------------- validation


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalTypeError(f"{what} must be a non-empty string")
    return value


def _choice(value: object, what: str, allowed: Sequence[str]) -> str:
    text = _name(value, what)
    if text not in allowed:
        raise CausalValueError(f"{what} must be one of {tuple(allowed)}, got {text!r}")
    return text


def _hex(value: object, what: str) -> str:
    text = _name(value, what)
    if not _HEX.match(text):
        raise CausalValueError(f"{what} must be 64 hexadecimal characters")
    return text.lower()


def _texts(values: Iterable[str], what: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise CausalTypeError(f"{what} must be a sequence of strings, not a string")
    return tuple(_name(v, what) for v in values)


def _dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _unwrap(pair: tuple[Any, str | None]) -> Any:
    value, refusal = pair
    raise_refusal(refusal, CompositionVerificationRefusal)
    return value


def digest(*parts: str | bytes) -> str:
    """A 64-hex identity for declared content: SHA-256 over length-prefixed parts.

    A convenience for producers that have no identity of their own to declare; the library
    treats every identity as an opaque 64-hex declaration.
    """
    hasher = hashlib.sha256()
    for part in parts:
        raw = part.encode("utf-8") if isinstance(part, str) else bytes(part)
        hasher.update(len(raw).to_bytes(8, "little"))
        hasher.update(raw)
    return hasher.hexdigest()


# --------------------------------------------------------------------------- declarations


@dataclass(frozen=True, slots=True)
class Calibration:
    """What a node says about its uncertainty guarantee.

    Use :meth:`point_only` (no guarantee claimed), :meth:`required` (one is wanted and nothing
    backs it), :meth:`held` (a record established at ``coordinate``) or :meth:`inherited` (a record
    claimed from the upstream node ``source``).
    """

    kind: str
    record: str | None = None
    coordinate: str | None = None
    source: str | None = None

    @classmethod
    def point_only(cls) -> Calibration:
        return cls("point_only")

    @classmethod
    def required(cls) -> Calibration:
        return cls("required")

    @classmethod
    def held(cls, record: str, coordinate: str) -> Calibration:
        return cls("held", _hex(record, "record"), _hex(coordinate, "coordinate"))

    @classmethod
    def inherited(cls, source: str, record: str, coordinate: str) -> Calibration:
        return cls(
            "inherited",
            _hex(record, "record"),
            _hex(coordinate, "coordinate"),
            _name(source, "source"),
        )

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {"kind": self.kind}
        if self.record is not None:
            wire["record"] = self.record
            wire["coordinate"] = self.coordinate
        if self.source is not None:
            wire["from"] = self.source
        return wire


@dataclass(frozen=True, slots=True)
class Node:
    """One claim or object of a composition.

    ``identity`` is the content identity (it changes whenever the data, fit, numbers or
    declarations it came from change); ``structure`` the identity of the structural derivation
    alone (it survives a data replacement); ``coordinate`` where the claim is evaluated (target
    population, regime, quantity); ``unsupported`` the coordinates outside the support;
    ``trust`` (``not_applicable``, ``held`` or ``withdrawn``) a provider's trust standing;
    ``evidence`` the ledger source ids (hex) the node counts.
    """

    key: str
    kind: str
    identity: str
    structure: str | None = None
    representation: str | None = None
    coordinate: str | None = None
    unsupported: tuple[str, ...] = ()
    trust: str = "not_applicable"
    calibration: Calibration = field(default_factory=Calibration.point_only)
    assumptions: tuple[str, ...] = ()
    transformation: str | None = None
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _name(self.key, "key")
        _choice(self.kind, "kind", NODE_KINDS)
        object.__setattr__(self, "identity", _hex(self.identity, "identity"))
        if self.structure is not None:
            object.__setattr__(self, "structure", _hex(self.structure, "structure"))
        if self.coordinate is not None:
            object.__setattr__(self, "coordinate", _hex(self.coordinate, "coordinate"))
        _choice(self.trust, "trust", _TRUST)
        if not isinstance(self.calibration, Calibration):
            raise CausalTypeError("calibration must be a Calibration")
        object.__setattr__(self, "unsupported", _texts(self.unsupported, "unsupported"))
        object.__setattr__(self, "assumptions", _texts(self.assumptions, "assumption"))
        object.__setattr__(
            self, "evidence", tuple(_hex(e, "evidence source") for e in self.evidence)
        )

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {
            "key": self.key,
            "kind": self.kind,
            "identity": self.identity,
            "trust": self.trust,
            "calibration": self.calibration.to_wire(),
            "assumptions": list(self.assumptions),
            "evidence": list(self.evidence),
        }
        if self.structure is not None:
            wire["structure"] = self.structure
        if self.representation is not None:
            wire["representation"] = self.representation
        if self.coordinate is not None:
            wire["coordinate"] = self.coordinate
        if self.unsupported:
            wire["unsupported"] = list(self.unsupported)
        if self.transformation is not None:
            wire["transformation"] = self.transformation
        return wire


@dataclass(frozen=True, slots=True)
class Edge:
    """``downstream`` depends on ``upstream`` in the stated way."""

    downstream: str
    upstream: str
    kind: str

    def __post_init__(self) -> None:
        _name(self.downstream, "downstream")
        _name(self.upstream, "upstream")
        _choice(self.kind, "kind", DEPENDENCY_KINDS)

    def to_wire(self) -> dict[str, str]:
        return {"downstream": self.downstream, "upstream": self.upstream, "kind": self.kind}


@dataclass(frozen=True, slots=True)
class CompositionGraph:
    """A set of nodes with typed dependencies. Declaration order carries no meaning."""

    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...] = ()

    def __post_init__(self) -> None:
        nodes = tuple(self.nodes)
        edges = tuple(self.edges)
        if not all(isinstance(n, Node) for n in nodes):
            raise CausalTypeError("nodes must be Node values")
        if not all(isinstance(e, Edge) for e in edges):
            raise CausalTypeError("edges must be Edge values")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)

    def node(self, key: str) -> Node:
        """The node under ``key``."""
        for node in self.nodes:
            if node.key == key:
                return node
        raise CausalValueError(f"no node {key!r}")

    def replace_node(self, key: str, **changes: Any) -> CompositionGraph:
        """A copy with the fields of one node replaced."""
        self.node(key)
        return CompositionGraph(
            nodes=tuple(replace(n, **changes) if n.key == key else n for n in self.nodes),
            edges=self.edges,
        )

    def replace_identity(self, key: str, identity: str) -> CompositionGraph:
        """A copy in which one node's content identity changed (its data or fit was replaced)."""
        return self.replace_node(key, identity=identity)

    def to_wire(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_wire() for n in self.nodes],
            "edges": [e.to_wire() for e in self.edges],
        }


@dataclass(frozen=True, slots=True)
class Effect:
    """What an observation outside the graph says about one node.

    ``kind`` is ``recomputed`` (its numbers are stale), ``withdrawn`` (it was withdrawn or
    replaced) or ``refused`` (the stage cannot be executed); ``text`` the cause.
    """

    key: str
    kind: str
    text: str

    def __post_init__(self) -> None:
        _name(self.key, "key")
        _choice(self.kind, "kind", ("recomputed", "withdrawn", "refused"))
        _name(self.text, "text")

    @classmethod
    def recomputed(cls, key: str, cause: str) -> Effect:
        return cls(key, "recomputed", cause)

    @classmethod
    def withdrawn(cls, key: str, cause: str) -> Effect:
        return cls(key, "withdrawn", cause)

    @classmethod
    def refused(cls, key: str, detail: str) -> Effect:
        return cls(key, "refused", detail)

    def to_wire(self) -> dict[str, str]:
        return {"key": self.key, "kind": self.kind, "text": self.text}


@dataclass(frozen=True, slots=True)
class SharedData:
    """Two nodes whose evidence overlaps (``first`` < ``second``); independent pairs are absent."""

    first: str
    second: str
    relation: str
    shared_observations: int

    def to_wire(self) -> dict[str, Any]:
        return {
            "first": self.first,
            "second": self.second,
            "relation": self.relation,
            "shared_observations": self.shared_observations,
        }


# --------------------------------------------------------------------------- ledger integration

_RANK = {"shared_data": 0, "shared_fitted_model": 1, "shared_prior": 2, "unknown_dependence": 3}
_SELF_RELATION = {"prior": "shared_prior", "fitted_model": "shared_fitted_model"}


def ledger_effects(ledger: Any, bindings: Mapping[str, str]) -> tuple[Effect, ...]:
    """The effects of ledger standings on the nodes bound to ledger results.

    ``bindings`` maps a node key to a result id. A result that needs recomputation is a numerical
    effect; a withdrawn or replaced result is a withdrawal.
    """
    effects: list[Effect] = []
    for key, result in sorted(bindings.items()):
        standing = ledger.standing(result)
        if standing is None:
            raise CausalValueError(f"result {result!r} is not in the ledger")
        if standing == "recompute":
            effects.append(Effect.recomputed(key, f"ledger.standing:{result}"))
        elif standing == "withdrawn":
            effects.append(Effect.withdrawn(key, f"ledger.standing:{result}"))
    return tuple(effects)


def shared_data_from_ledger(graph: CompositionGraph, ledger: Any) -> tuple[SharedData, ...]:
    """Shared-data relations among the nodes' evidence sources, from the ledger's overlap.

    An overlap that is not computable is an ``unknown_dependence``, never independent.
    """
    carriers = sorted((n for n in graph.nodes if n.evidence), key=lambda n: n.key)
    every = sorted({s for n in carriers for s in n.evidence})
    if not every:
        return ()
    account = ledger.account_observations(every)
    pairs = {(p.first, p.second): p for p in account.pairs}
    pairs.update({(p.second, p.first): p for p in account.pairs})
    records = {s.id: s for s in ledger.sources}
    out: list[SharedData] = []
    for i, first in enumerate(carriers):
        for second in carriers[i + 1 :]:
            best: str | None = None
            shared = 0
            for a in first.evidence:
                for b in second.evidence:
                    relation, count = _source_relation(a, b, pairs, records)
                    if relation is None:
                        continue
                    shared += count
                    if best is None or _RANK[relation] < _RANK[best]:
                        best = relation
            if best is not None:
                out.append(SharedData(first.key, second.key, best, shared))
    return tuple(out)


def _source_relation(
    a: str, b: str, pairs: Mapping[tuple[str, str], Any], records: Mapping[str, Any]
) -> tuple[str | None, int]:
    if a == b:
        record = records[a]
        if record.effective_observations is None:
            return "unknown_dependence", 0
        return _SELF_RELATION.get(record.kind, "shared_data"), int(record.effective_observations)
    pair = pairs[(a, b)]
    if pair.overlap == "independent":
        return None, 0
    if pair.overlap == "unknown":
        return "unknown_dependence", 0
    return str(pair.relation), int(pair.shared or 0)


# --------------------------------------------------------------------------- results


@dataclass(frozen=True, slots=True)
class NumericalReason:
    """Why a claim's numbers are stale."""

    origin: str
    via_node: str | None
    via_kind: str | None
    cause: str


@dataclass(frozen=True, slots=True)
class Withdrawal:
    """A license a claim lost: ``retired``, ``support``, ``provider_trust``, ``calibration`` or
    ``unexecutable``."""

    kind: str
    origin: str
    via_node: str | None
    via_kind: str | None
    detail: str


@dataclass(frozen=True, slots=True)
class CalibrationStanding:
    """A claim's uncertainty guarantee: inherited, lost or newly required."""

    state: str
    source: str | None
    newly_required: bool
    detail: str | None


@dataclass(frozen=True, slots=True)
class ClaimClassification:
    """The classification of one claim.

    ``klass`` is ``still_valid``, ``recompute`` or ``withdrawn``. ``authority_retained`` is true
    only for a ``still_valid`` claim: reusing a structural derivation after the data were
    replaced never retains the earlier fit or uncertainty.
    """

    key: str
    kind: str
    klass: str
    numerical: tuple[NumericalReason, ...]
    withdrawals: tuple[Withdrawal, ...]
    authority_retained: bool
    calibration: CalibrationStanding

    def withdrawn_for(self, kind: str) -> bool:
        """Whether a withdrawal of this kind stands."""
        return any(w.kind == kind for w in self.withdrawals)


def _classification(wire: Mapping[str, Any]) -> ClaimClassification:
    calibration = wire["calibration"]
    return ClaimClassification(
        key=wire["key"],
        kind=wire["kind"],
        klass=wire["class"],
        numerical=tuple(
            NumericalReason(r["origin"], r["via_node"], r["via_kind"], r["cause"])
            for r in wire["numerical"]
        ),
        withdrawals=tuple(
            Withdrawal(w["kind"], w["origin"], w["via_node"], w["via_kind"], w["detail"])
            for w in wire["withdrawals"]
        ),
        authority_retained=bool(wire["authority_retained"]),
        calibration=CalibrationStanding(
            calibration["state"],
            calibration["from"],
            bool(calibration["newly_required"]),
            calibration["detail"],
        ),
    )


@dataclass(frozen=True, slots=True)
class LineageEntry:
    """One object in a claim's lineage."""

    key: str
    kind: str
    identity: str
    representation: str | None
    klass: str
    calibration: CalibrationStanding


@dataclass(frozen=True, slots=True)
class CompositionView:
    """The lineage of one claim: what stands behind it, and how.

    ``upstream`` lists upstream objects in dependency order; ``transformations`` the bridges and
    models on the way; ``shared_data`` the overlap among them; ``versions`` the model and
    representation versions; ``unsupported`` the coordinates outside any support; ``assumptions``
    what each boundary introduced; each entry's ``calibration`` says what was inherited, lost or
    newly required.
    """

    claim: LineageEntry
    upstream: tuple[LineageEntry, ...]
    edges: tuple[tuple[str, str, str], ...]
    evidence: tuple[str, ...]
    transformations: tuple[tuple[str, str, str], ...]
    shared_data: tuple[SharedData, ...]
    versions: tuple[tuple[str, str], ...]
    unsupported: tuple[tuple[str, str], ...]
    assumptions: tuple[tuple[str, str], ...]
    identity: str


def _entry(wire: Mapping[str, Any]) -> LineageEntry:
    c = wire["calibration"]
    return LineageEntry(
        wire["key"],
        wire["kind"],
        wire["identity"],
        wire["representation"],
        wire["class"],
        CalibrationStanding(c["state"], c["from"], bool(c["newly_required"]), c["detail"]),
    )


def _shared(items: Iterable[Mapping[str, Any]]) -> tuple[SharedData, ...]:
    return tuple(
        SharedData(i["first"], i["second"], i["relation"], int(i["shared_observations"]))
        for i in items
    )


def _view(wire: Mapping[str, Any]) -> CompositionView:
    return CompositionView(
        claim=_entry(wire["claim"]),
        upstream=tuple(_entry(e) for e in wire["upstream"]),
        edges=tuple((e["downstream"], e["upstream"], e["kind"]) for e in wire["edges"]),
        evidence=tuple(wire["evidence"]),
        transformations=tuple((t["node"], t["kind"], t["name"]) for t in wire["transformations"]),
        shared_data=_shared(wire["shared_data"]),
        versions=tuple((v["node"], v["version"]) for v in wire["versions"]),
        unsupported=tuple((u["node"], u["coordinate"]) for u in wire["unsupported"]),
        assumptions=tuple((a["node"], a["statement"]) for a in wire["assumptions"]),
        identity=wire["identity"],
    )


@dataclass(frozen=True, slots=True)
class VerificationExport:
    """The bytes of ``composition_verification_v1`` and its identity (retain it independently)."""

    artifact: bytes
    identity: str


@dataclass(frozen=True, slots=True, eq=False)
class CompositionVerification:
    """A sealed classification of a composed analysis."""

    claims: Mapping[str, ClaimClassification]
    removed: tuple[str, ...]
    identity: str
    has_previous: bool
    ledger_state: str | None
    shared_data: tuple[SharedData, ...]
    _handle: Any = field(repr=False, compare=False)

    def classify(self, key: str) -> ClaimClassification:
        """The classification of one claim."""
        try:
            return self.claims[key]
        except KeyError:
            raise CausalValueError(f"no claim {key!r} in the current graph") from None

    def keys_of(self, klass: str) -> tuple[str, ...]:
        """Claim keys with a class, sorted."""
        _choice(klass, "klass", CLASSES)
        return tuple(sorted(k for k, c in self.claims.items() if c.klass == klass))

    def view(self, key: str) -> CompositionView:
        """The lineage of one claim.

        Raises:
            CompositionVerificationRefusal: ``invalid_argument`` for an unknown claim.
        """
        value, refusal = self._handle.view_json(_name(key, "key"))
        raise_refusal(refusal, CompositionVerificationRefusal)
        return _view(json.loads(value))

    def require_valid(self, key: str) -> ClaimClassification:
        """The claim, when it is usable as it stands.

        Raises:
            CompositionVerificationRefusal: ``claim_revalidation_required`` for a claim to be
                recomputed; ``claim_license_withdrawn`` for a withdrawn support, trust, retired
                input or unexecutable stage; ``cell_not_licensed`` for calibration inherited from
                an object that no longer holds it or established at another coordinate.
        """
        raise_refusal(self._handle.require_valid(_name(key, "key")), CompositionVerificationRefusal)
        return self.classify(key)

    def export(self, artifact_id: str = _DEFAULT_ARTIFACT_ID) -> VerificationExport:
        """Write ``composition_verification_v1``."""
        pair = _unwrap(self._handle.export(_name(artifact_id, "artifact_id")))
        return VerificationExport(artifact=bytes(pair[0]), identity=str(pair[1]))

    def verify_ledger(self, graph: CompositionGraph, ledger: Any) -> None:
        """Check the declared shared-data relations and ledger state against a ledger.

        Raises:
            CompositionVerificationRefusal: ``invalid_argument`` with detail
                ``composition_verification.ledger_mismatch`` when the ledger is at another state
                than the one the relations were read from, or its overlap differs.
        """
        if self.ledger_state is not None and self.ledger_state != ledger.state_digest:
            _raise_ledger_mismatch("the ledger state digest differs from the declared one")
        if tuple(shared_data_from_ledger(graph, ledger)) != self.shared_data:
            _raise_ledger_mismatch("the shared-data relations differ from the ledger's overlap")


def _raise_ledger_mismatch(message: str) -> None:
    raise CompositionVerificationRefusal(
        {
            "code": "invalid_argument",
            "stage": "composition_verification",
            "detail": "composition_verification.ledger_mismatch",
            "message": message,
            "offending": None,
            "remedy": (
                "verify against the ledger the relations were read from, or regenerate the "
                "verification"
            ),
        }
    )


def _wrap(handle: Any) -> CompositionVerification:
    summary = json.loads(handle.summary_json())
    return CompositionVerification(
        claims={c["key"]: _classification(c) for c in summary["claims"]},
        removed=tuple(summary["removed"]),
        identity=summary["identity"],
        has_previous=bool(summary["has_previous"]),
        ledger_state=summary["ledger_state"],
        shared_data=_shared(summary["shared_data"]),
        _handle=handle,
    )


# --------------------------------------------------------------------------- entry points


def verify_composition(
    current: CompositionGraph,
    *,
    previous: CompositionGraph | None = None,
    effects: Iterable[Effect] = (),
    shared_data: Iterable[SharedData] = (),
    ledger: Any = None,
) -> CompositionVerification:
    """Classify every claim of ``current`` against ``previous``.

    Without ``previous`` this is a first verification and nothing is reported stale. With a
    ``ledger`` the shared-data relations among the nodes' evidence sources and the ledger's state
    digest are read from it (an unknown overlap is an unknown dependence, never independent).

    Raises:
        CompositionVerificationRefusal: ``invalid_argument`` for a graph with a duplicate node,
            an unknown dependency, a cycle, or inherited calibration without a calibration
            dependency.
    """
    if not isinstance(current, CompositionGraph):
        raise CausalTypeError("current must be a CompositionGraph")
    if previous is not None and not isinstance(previous, CompositionGraph):
        raise CausalTypeError("previous must be a CompositionGraph or None")
    effect_list = list(effects)
    if not all(isinstance(e, Effect) for e in effect_list):
        raise CausalTypeError("effects must be Effect values")
    relations = list(shared_data)
    state: str | None = None
    if ledger is not None:
        relations = list(shared_data_from_ledger(current, ledger))
        state = str(ledger.state_digest)
    request: dict[str, Any] = {
        "current": current.to_wire(),
        "previous": None if previous is None else previous.to_wire(),
        "effects": [e.to_wire() for e in effect_list],
        "shared_data": [r.to_wire() for r in relations],
        "ledger_state": state,
    }
    return _wrap(_unwrap(_native.composition_verification_verify(_dumps(request))))


def consume_composition_verification_artifact(
    artifact: bytes, expected_identity: str | None = None
) -> CompositionVerification:
    """Independently consume a composition verification artifact by classifying again.

    ``expected_identity`` is the artifact identity the consumer retained independently of the
    bytes; a consistently resealed artifact of another verification is refused.

    Raises:
        CompositionVerificationRefusal: ``route_not_supported`` for an identity mismatch,
            ``invalid_argument`` for corrupt or foreign bytes, a graph the library refuses, or a
            stored classification or lineage identity that does not replay.
    """
    if not isinstance(artifact, (bytes, bytearray, memoryview)):
        raise CausalTypeError("artifact must be bytes")
    if expected_identity is not None:
        _name(expected_identity, "expected_identity")
    return _wrap(
        _unwrap(_native.composition_verification_consume(bytes(artifact), expected_identity))
    )
