"""Revalidation of transported claims after a refresh or a state change.

A transported claim is a claim about a target population derived from source evidence under a
selection diagram, possibly carrying a calibration record established at one target. When the
target, a source's evidence, the diagram or the support changes, every transported claim and
every decision resting on one is classified ``still_valid``, ``recompute`` or ``withdrawn``, in
transport terms::

    from antecedent.transport import revalidation as rv

    scene = (
        rv.TransportScene.builder("clinic_b", target_data_identity)
        .source("trial", trial_evidence_identity)
        .transported("effect", response, rv.Calibration.point_only())
        .decision("treat_or_wait", identity, quantity, rests_on=["effect"])
        .build()
    )
    report = rv.revalidate(previous_scene, scene)
    report.claim("effect").classification   # "recompute" when the trial's evidence changed
    report.require_valid("effect")          # raises with a stated remedy otherwise

The rule this runs: **transported calibration is established at one target.** A record held at
one target is never read as holding at another: a claim whose record was established at a
different target (or for a different quantity) than the one it now answers is withdrawn for
calibration (``calibration_from_another_target``) and its point value is left as it was; a claim
that inherits a record from another claim holds it only while that claim holds it at the same
target and quantity. Nothing here declares or measures a calibration record; a record is an
identity the caller holds.

**Lifecycle.**

* A refresh with outcome-only data of a source population changes that source's evidence
  identity: every claim counting the source **recomputes** (the earlier fit and uncertainty
  carry no authority), the declared selection diagram, structure and quantity are **kept**, and
  a claim about another target is **refused** its calibration rather than inheriting it.
* A change of target population or diagram recomputes every transported claim; a target outside
  the support withdraws them.
* :class:`TransportClaimState` holds the current scene and advances it only when a revalidation
  succeeds, so a refused refresh leaves the state unchanged.
* :meth:`TransportRevalidation.export` writes the claim-level verification as
  ``composition_verification_v1``; :func:`antecedent.composition_verification.consume` replays
  the classification from the bytes.

Rust owns the identities, the classification, the sealed verification and every refusal; this
module declares the scene and presents the result. Refusals raise
:class:`~antecedent.transported_response.TransportedResponseRefusal` with a registered
``reason_code``, a ``transport_revalidation.*`` ``detail`` and a ``remedy``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .. import _native
from .._artifact_protocol import ExportedArtifact, artifact_export, artifact_from_receipt
from .._columns import coerce_fields
from .._refusal import raise_refusal
from .._scientific import ScientificResult
from .._validate import choice, text
from ..errors import CausalTypeError
from ..transported_response import TransportedResponse, TransportedResponseRefusal

__all__ = [
    "Calibration",
    "ClaimRevalidation",
    "ObservedEffect",
    "TARGET_NODE_KEY",
    "DIAGRAM_NODE_KEY",
    "TransportClaimState",
    "TransportRevalidation",
    "TransportScene",
    "TransportSceneBuilder",
    "claim_node_key",
    "revalidate",
    "source_node_key",
]

#: The graph key of the target population, as used by observed effects.
TARGET_NODE_KEY = "target_population"
#: The graph key of the selection diagram, as used by observed effects.
DIAGRAM_NODE_KEY = "selection_diagram"

TrustStanding = Literal["not_applicable", "held", "withdrawn", "undeclared"]


def claim_node_key(key: str) -> str:
    """The graph key of a claim, as used by observed effects."""
    return f"claim:{key}"


def source_node_key(population: str) -> str:
    """The graph key of a source population, as used by observed effects."""
    return f"source:{population}"


def _unwrap(pair: tuple[Any, str | None]) -> Any:
    value, refusal = pair
    raise_refusal(refusal, TransportedResponseRefusal)
    return value


def _hex(value: str, what: str) -> str:
    text(value, what, non_empty=True)
    return value


# --------------------------------------------------------------------------- declarations


@dataclass(frozen=True, slots=True)
class Calibration:
    """How a transported claim's uncertainty guarantee is backed.

    ``point_only`` states no guarantee; ``required`` wants one and no record backs it;
    ``held`` is a record the claim holds itself, established at ``established_at_target``
    (default: the target of the scene it is declared in); ``inherited`` is a record claimed from
    another claim of the scene.
    """

    kind: Literal["point_only", "required", "held", "inherited"]
    record: str | None = None
    established_at_target: str | None = None
    from_claim: str | None = None

    @classmethod
    def point_only(cls) -> Calibration:
        """A point value; no guarantee is claimed."""
        return cls("point_only")

    @classmethod
    def required(cls) -> Calibration:
        """A guarantee is wanted and no record backs it."""
        return cls("required")

    @classmethod
    def held(cls, record: str, established_at_target: str | None = None) -> Calibration:
        """A record the claim holds itself (a 64-hex identity)."""
        return cls("held", _hex(record, "record"), established_at_target)

    @classmethod
    def inherited(
        cls, from_claim: str, record: str, established_at_target: str | None = None
    ) -> Calibration:
        """A record claimed from the claim ``from_claim`` of the same scene."""
        return cls("inherited", _hex(record, "record"), established_at_target, from_claim)

    def _wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {"kind": self.kind}
        if self.kind in ("held", "inherited"):
            wire["record"] = self.record
            if self.established_at_target is not None:
                wire["established_at_target"] = self.established_at_target
        if self.kind == "inherited":
            wire["from"] = self.from_claim
        return wire


@dataclass(frozen=True, slots=True)
class ObservedEffect:
    """An effect observed on one node of the scene (a stale stage, a withdrawn premise).

    ``key`` is a node key (:func:`claim_node_key`, :func:`source_node_key`,
    :data:`TARGET_NODE_KEY`, :data:`DIAGRAM_NODE_KEY`); ``kind`` is ``recomputed``,
    ``withdrawn`` or ``refused``; ``text`` is where the observation came from.
    """

    key: str
    kind: Literal["recomputed", "withdrawn", "refused"]
    text: str

    def __post_init__(self) -> None:
        choice(self.kind, "kind", ("recomputed", "withdrawn", "refused"))

    def _wire(self) -> dict[str, str]:
        return {"key": self.key, "kind": self.kind, "text": self.text}


@dataclass(frozen=True, slots=True)
class TransportSceneBuilder:
    """Declares a :class:`TransportScene` from transported responses.

    Each :class:`~antecedent.transported_response.TransportedResponse` contributes its receipt
    (source populations, target population, selection-diagram identities, support label), its
    value digest and the calibration the caller states. The builder is immutable: every method
    returns a new builder.
    """

    target_population: str
    target_data_identity: str
    unsupported_target: tuple[str, ...] | None = None
    sources: tuple[tuple[str, str], ...] = ()
    claims: tuple[dict[str, Any], ...] = ()
    responses: tuple[TransportedResponse, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        text(self.target_population, "target_population", non_empty=True)
        _hex(self.target_data_identity, "target_data_identity")
        coerce_fields(self, claims=tuple(dict(c) for c in self.claims))

    @property
    def target_identity(self) -> str:
        """The identity the scene gives its target: the population and the content identity of
        its declaration and data. A calibration record established at this target names it."""
        return str(
            _native.transport_scene_target_identity(
                self.target_population, self.target_data_identity
            )
        )

    def target_support(
        self, unsupported_coordinates: Sequence[str] | None
    ) -> TransportSceneBuilder:
        """Declare the target outside the support the transport needs, naming the coordinates
        (``None`` restores a supported target)."""
        coordinates = None if unsupported_coordinates is None else tuple(unsupported_coordinates)
        return replace(self, unsupported_target=coordinates)

    def source(self, population: str, evidence_identity: str) -> TransportSceneBuilder:
        """Declare a source population with the content identity of its evidence."""
        text(population, "population", non_empty=True)
        entry = (population, _hex(evidence_identity, "evidence_identity"))
        kept = tuple(s for s in self.sources if s[0] != population)
        return replace(self, sources=(*kept, entry))

    def transported(
        self,
        key: str,
        response: TransportedResponse,
        calibration: Calibration | None = None,
        trust: TrustStanding = "not_applicable",
    ) -> TransportSceneBuilder:
        """Add a transported claim from a response and the calibration it states."""
        if not isinstance(response, TransportedResponse):
            raise CausalTypeError("response must be a TransportedResponse")
        stated = calibration or Calibration.point_only()
        claim: dict[str, Any] = {
            "kind": "transported",
            "key": text(key, "key", non_empty=True),
            "response": len(self.responses),
            "calibration": stated._wire(),
            "trust": trust,
        }
        return replace(self, claims=(*self.claims, claim), responses=(*self.responses, response))

    def decision(
        self, key: str, identity: str, quantity: str, rests_on: Sequence[str] = ()
    ) -> TransportSceneBuilder:
        """Add a decision resting on claims of the scene."""
        claim: dict[str, Any] = {
            "kind": "decision",
            "key": text(key, "key", non_empty=True),
            "identity": _hex(identity, "identity"),
            "quantity": _hex(quantity, "quantity"),
            "rests_on": list(rests_on),
        }
        return replace(self, claims=(*self.claims, claim))

    def build(self) -> TransportScene:
        """The scene.

        Raises:
            TransportedResponseRefusal: a claim about another target
                (``transport_revalidation.claim_target_mismatch``), a repeated key
                (``.duplicate_claim``), a claim counting a source population the scene does not
                declare (``.source_undeclared``) or a graph that does not build
                (``.scene_invalid``).
        """
        wire: dict[str, Any] = {
            "target_population": self.target_population,
            "target_data_identity": self.target_data_identity,
            "unsupported_target": (
                None if self.unsupported_target is None else list(self.unsupported_target)
            ),
            "sources": [list(s) for s in self.sources],
            "claims": list(self.claims),
        }
        handle = _unwrap(
            _native.transport_scene_build(json.dumps(wire), [r._handle for r in self.responses])
        )
        return TransportScene._of(handle)


# --------------------------------------------------------------------------- scene


@dataclass(repr=False, frozen=True, slots=True, eq=False)
class TransportScene(ScientificResult):
    """Everything that decides which claims about one target are licensed.

    ``target_identity`` is the identity of the target (population and data); ``selection_diagram``
    the content identity of the diagrams the claims name; ``sources`` the declared source
    populations with their evidence identities; ``claims`` every claim and decision with its
    identities and dependencies.
    """

    target_identity: str
    target_supported: bool
    selection_diagram: str
    sources: tuple[Mapping[str, Any], ...]
    claims: tuple[Mapping[str, Any], ...]
    _handle: Any = field(repr=False, compare=False, default=None)

    @classmethod
    def builder(cls, target_population: str, target_data_identity: str) -> TransportSceneBuilder:
        """A builder for a scene about ``target_population``; ``target_data_identity`` is the
        content identity (64 hex) of the target's declaration and data."""
        return TransportSceneBuilder(target_population, target_data_identity)

    @classmethod
    def _of(cls, handle: Any) -> TransportScene:
        info = json.loads(handle.summary_json())
        return cls(
            target_identity=info["target_identity"],
            target_supported=bool(info["target_supported"]),
            selection_diagram=info["selection_diagram"],
            sources=tuple(info["sources"]),
            claims=tuple(info["claims"]),
            _handle=handle,
        )

    def claim(self, key: str) -> Mapping[str, Any]:
        """One declared claim by key."""
        for item in self.claims:
            if item["key"] == key:
                return item
        raise CausalTypeError(f"no claim {key!r} in this scene")


# --------------------------------------------------------------------------- report


@dataclass(frozen=True, slots=True)
class ClaimRevalidation:
    """One claim's classification, in transport terms.

    ``classification`` is ``still_valid``, ``recompute`` or ``withdrawn``. ``reasons`` are the
    snake_case names of why (``source_evidence_changed``, ``target_population_changed``,
    ``calibration_from_another_target``, ...); ``remedies`` say what the caller can do about
    each. ``prior_authority`` is ``retained`` or ``not_retained``. ``calibration_withdrawn`` is
    ``True`` when the record the claim relied on no longer holds: its point value is left as it
    was.
    """

    key: str
    kind: Literal["transported", "decision"]
    classification: Literal["still_valid", "recompute", "withdrawn"]
    reasons: tuple[str, ...]
    remedies: tuple[str, ...]
    prior_authority: str
    calibration: str
    calibration_withdrawn: bool

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> ClaimRevalidation:
        return cls(
            key=wire["key"],
            kind=wire["kind"],
            classification=wire["class"],
            reasons=tuple(wire["reasons"]),
            remedies=tuple(wire["remedies"]),
            prior_authority=wire["prior_authority"],
            calibration=wire["calibration"],
            calibration_withdrawn=bool(wire["calibration_withdrawn"]),
        )


@dataclass(repr=False, frozen=True, slots=True, eq=False)
class TransportRevalidation(ScientificResult):
    """A scene's classification against the earlier scene.

    ``claims`` lists every claim of the current scene in key order; ``removed`` the claims of
    the earlier scene absent now. ``identity`` is the identity of the classification and
    ``verification_identity`` that of the sealed claim-level verification.
    """

    claims: tuple[ClaimRevalidation, ...]
    removed: tuple[str, ...]
    identity: str
    verification_identity: str
    _handle: Any = field(repr=False, compare=False, default=None)

    @classmethod
    def _of(cls, handle: Any) -> TransportRevalidation:
        info = json.loads(handle.inspect_json())
        return cls(
            claims=tuple(ClaimRevalidation._from_wire(c) for c in info["claims"]),
            removed=tuple(info["removed"]),
            identity=str(info["identity"]),
            verification_identity=str(info["verification_identity"]),
            _handle=handle,
        )

    def claim(self, key: str) -> ClaimRevalidation:
        """The classification of one claim."""
        for item in self.claims:
            if item.key == key:
                return item
        raise CausalTypeError(f"no claim {key!r} in this revalidation")

    def require_valid(self, key: str) -> ClaimRevalidation:
        """The claim when it is usable as it stands.

        Raises:
            TransportedResponseRefusal: the typed refusal of a claim that must be recomputed or
                whose license was withdrawn (a calibration record from another target is
                ``cell_not_licensed``), or ``invalid_argument`` for an unknown key.
        """
        _unwrap(self._handle.require_valid(key))
        return self.claim(key)

    @artifact_export
    def export(self, *, artifact_id: str = "transport-revalidation") -> ExportedArtifact[Any]:
        """The claim-level verification as ``composition_verification_v1``; a fresh consumer
        replays the classification."""
        return artifact_from_receipt(self, bytes(_unwrap(self._handle.export(artifact_id))))


def _effects(effects: Sequence[ObservedEffect] | None) -> str | None:
    if effects is None:
        return None
    if not all(isinstance(e, ObservedEffect) for e in effects):
        raise CausalTypeError("effects must be ObservedEffect objects")
    return json.dumps([e._wire() for e in effects])


def revalidate(
    previous: TransportScene | None,
    current: TransportScene,
    effects: Sequence[ObservedEffect] | None = None,
) -> TransportRevalidation:
    """Classify every claim of ``current`` against ``previous`` under the observed ``effects``.

    ``previous=None`` is a first verification. Refuses a scene whose graph does not build.
    """
    if previous is not None and not isinstance(previous, TransportScene):
        raise CausalTypeError("previous must be a TransportScene or None")
    if not isinstance(current, TransportScene):
        raise CausalTypeError("current must be a TransportScene")
    handle = _unwrap(
        _native.transport_revalidate(
            None if previous is None else previous._handle, current._handle, _effects(effects)
        )
    )
    return TransportRevalidation._of(handle)


class TransportClaimState:
    """The current scene of a set of transported claims, advanced only by a successful
    revalidation. A refused observation leaves the state exactly as it was."""

    def __init__(self) -> None:
        self._native = _native.TransportClaimStateHandle()

    @property
    def scene(self) -> TransportScene | None:
        """The scene the last successful observation recorded."""
        handle = self._native.scene()
        return None if handle is None else TransportScene._of(handle)

    def observe(
        self, scene: TransportScene, effects: Sequence[ObservedEffect] | None = None
    ) -> TransportRevalidation:
        """Classify ``scene`` against the recorded scene and, only on success, record it."""
        if not isinstance(scene, TransportScene):
            raise CausalTypeError("scene must be a TransportScene")
        handle = _unwrap(self._native.observe(scene._handle, _effects(effects)))
        return TransportRevalidation._of(handle)

    def __repr__(self) -> str:
        recorded = self.scene
        return f"<TransportClaimState recorded={recorded is not None}>"
