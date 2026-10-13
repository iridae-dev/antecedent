"""Typed bridges between scientific quantities.

A claim answers a question about one scientific quantity. A consumer often asks about another:
the construct a measurement stands for, the behavior a stated preference predicts, the terminal
outcome a surrogate tracks, the causal quantity a prediction approximates, the response a
parameter encodes, the utility an endpoint is scored against. The claim does not answer those
questions by being renamed. A bridge is the explicit, separately identified derivation that states
the extra assumptions under which the source answers the downstream question::

    from antecedent import scientific_bridge as sb
    from antecedent.joint_distribution import ScientificQuantity

    surrogate = ScientificQuantity.outcome(
        "trial:biomarker", units="mmol_per_l", population="adults",
        regime="do(t=1)", functional="average_effect",
    )
    terminal = ScientificQuantity.outcome(
        "trial:survival", units="log_hazard", population="adults",
        regime="do(t=1)", functional="average_effect",
    )
    contract = sb.BridgeContract(
        kind="surrogate_terminal_outcome",
        source=surrogate, source_class="surrogate_endpoint",
        target=terminal, target_class="terminal_outcome",
        evidence_scope=sb.Scope("adults", "do(t=1)"),
        mapping=sb.TrialLevelMapping(
            trials=[(0.0, 0.1), (0.5, 0.5), (1.0, 0.9), (1.5, 1.5)],
            min_trials=4, min_r_squared=0.9,
        ),
        assumptions=[sb.Assumption(
            "validation", "surrogate_validation",
            "the surrogate is validated at the trial level", evidence="meta-analysis:ckd-2024",
        )],
    )
    bridge = sb.prepare_scientific_bridge(contract)
    claim = bridge.derive(sb.SourceClaim.declared("effect-on-biomarker", 0.8, surrogate))
    claim.value                      # 0.796: 0.06 + 0.92 * 0.8 from the trial-level line
    exported = claim.export("surrogate-bridge")
    again = sb.consume_scientific_bridge_artifact(exported.artifact, exported.identity)

A bridge produces a *new* claim with its own identity (binding the contract, every declared
assumption and the source claim), its own provenance, and a receipt of what it did not establish.
The source claim is not modified. Everything is point-only: a bridge carries one mean. Asking for
an interval, a posterior, a predictive law, an identified set or a joint law is refused
(:meth:`BridgedClaim.interval`, :meth:`BridgedClaim.require_value_kind`).

Every licensing check, mapping, identity and refusal is computed in Rust; this module builds the
declaration and presents the result. Refusals are :class:`ScientificBridgeRefusal` (a
:class:`~antecedent.errors.CausalUnsupportedError`) with a registered ``reason_code``, a
``scientific_bridge.*`` ``detail`` and a ``remedy``.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import _native
from ._refusal import StructuredRefusal, raise_refusal
from .errors import CausalTypeError, CausalValueError
from .measurement_models import MeasurementModel

__all__ = [
    "ASSUMPTION_KINDS",
    "BRIDGE_KINDS",
    "QUANTITY_CLASSES",
    "VALUE_KINDS",
    "AffineMapping",
    "Assumption",
    "BridgeContract",
    "BridgeExport",
    "BridgedClaim",
    "ConsumedBridge",
    "Freshness",
    "IdentityMapping",
    "MeasurementCorrection",
    "PreparedBridge",
    "Scope",
    "ScientificBridgeRefusal",
    "SourceClaim",
    "TransportabilityPremise",
    "TrialFit",
    "TrialLevelMapping",
    "consume_scientific_bridge_artifact",
    "contract_identity",
    "prepare_scientific_bridge",
]

#: The six bridge kinds.
BRIDGE_KINDS: tuple[str, ...] = (
    "latent_measured_outcome",
    "preference_realized_behavior",
    "surrogate_terminal_outcome",
    "predictive_causal_quantity",
    "parameter_intervention_response",
    "scientific_decision_utility",
)

#: What a quantity is, for the purpose of a bridge.
QUANTITY_CLASSES: tuple[str, ...] = (
    "latent_construct",
    "measured_outcome",
    "stated_preference",
    "realized_behavior",
    "surrogate_endpoint",
    "terminal_outcome",
    "predictive_quantity",
    "predictive_importance",
    "model_score",
    "causal_quantity",
    "model_parameter",
    "intervention_response",
    "scientific_endpoint",
    "decision_utility",
)

#: The kinds of declared assumption.
ASSUMPTION_KINDS: tuple[str, ...] = (
    "measurement_model",
    "behavioral_linkage",
    "surrogate_validation",
    "identification",
    "parameter_interpretation",
    "utility_mapping",
    "additional",
)

#: The kinds of object a value can be. They are not interchangeable.
VALUE_KINDS: tuple[str, ...] = (
    "mean",
    "predictive_law",
    "posterior",
    "identified_set",
    "joint_outcome_law",
)

_CLAIM_KINDS = ("point", "bounds", "mixture", "response", "policy_value", "refusal", "incomplete")
_DOMAIN_STATUSES = (
    "identified",
    "supported",
    "evaluated",
    "outside_scope",
    "contradicted",
    "unknown",
)


class ScientificBridgeRefusal(StructuredRefusal):
    """A typed refusal of the scientific-bridge route.

    A :class:`~antecedent.errors.CausalUnsupportedError` with a registered ``reason_code`` and a
    ``scientific_bridge.*`` ``detail``; ``remedy`` names the declaration or evidence that would
    license the request when one is known.
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


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise CausalTypeError(f"{what} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise CausalValueError(f"{what} must be finite")
    return number


def _dumps(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def _digest(*parts: str) -> str:
    """A 64-hex identity of length-framed text parts (a declared, not a native, identity)."""
    framed = "".join(f"{len(part)}:{part}" for part in parts)
    return hashlib.sha256(framed.encode("utf-8")).hexdigest()


def _quantity_wire(quantity: Any, what: str) -> dict[str, Any]:
    needed = (
        "variable_id",
        "variable_name",
        "role",
        "units",
        "population_id",
        "regime_id",
        "horizon",
        "functional_id",
        "conditioning",
        "transform_id",
    )
    if any(not hasattr(quantity, attribute) for attribute in needed):
        raise CausalTypeError(
            f"{what} must be a ScientificQuantity (see antecedent.joint_distribution)"
        )
    return {
        "version": 1,
        "variable_id": str(quantity.variable_id),
        "variable_name": str(quantity.variable_name),
        "role": str(quantity.role),
        "units": str(quantity.units),
        "population_id": str(quantity.population_id),
        "regime_id": str(quantity.regime_id),
        "horizon": int(quantity.horizon),
        "functional_id": str(quantity.functional_id),
        "conditioning": [
            {"variable_id": str(c.variable_id), "value_id": str(c.value_id)}
            for c in quantity.conditioning
        ],
        "transform_id": str(quantity.transform_id),
    }


def _unwrap(pair: tuple[Any, str | None]) -> Any:
    value, refusal = pair
    raise_refusal(refusal, ScientificBridgeRefusal)
    return value


# --------------------------------------------------------------------------- declarations


@dataclass(frozen=True, slots=True)
class Scope:
    """A population and a regime."""

    population_id: str
    regime_id: str

    def __post_init__(self) -> None:
        _name(self.population_id, "population_id")
        _name(self.regime_id, "regime_id")

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {"population_id": self.population_id, "regime_id": self.regime_id}


@dataclass(frozen=True, slots=True)
class Assumption:
    """One declared assumption of a bridge, with its own identity.

    ``kind`` is one of :data:`ASSUMPTION_KINDS`. Assumptions of the measurement-model,
    surrogate-validation, identification and utility-mapping kinds must name the ``evidence``
    (the study, model identity, identification or utility function) they rest on.
    """

    id: str
    kind: str
    statement: str
    evidence: str | None = None

    def __post_init__(self) -> None:
        _name(self.id, "id")
        _choice(self.kind, "kind", ASSUMPTION_KINDS)
        _name(self.statement, "statement")
        if self.evidence is not None:
            _name(self.evidence, "evidence")

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "id": self.id,
            "kind": self.kind,
            "statement": self.statement,
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class TransportabilityPremise:
    """The premise under which a relation established in one scope is carried to another.

    A declaration, not a check: the derived claim carries it as an unresolved obligation.
    """

    id: str
    statement: str
    established_in: Scope
    applies_to: Scope
    evidence: str | None = None

    def __post_init__(self) -> None:
        _name(self.id, "id")
        _name(self.statement, "statement")
        if not isinstance(self.established_in, Scope) or not isinstance(self.applies_to, Scope):
            raise CausalTypeError("established_in and applies_to must be Scope values")

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "id": self.id,
            "statement": self.statement,
            "established_in": self.established_in.to_wire(),
            "applies_to": self.applies_to.to_wire(),
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class IdentityMapping:
    """The value is unchanged; licensed only where the assumptions make the target equal it."""

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {"kind": "identity"}


@dataclass(frozen=True, slots=True)
class AffineMapping:
    """A declared ``intercept + slope * value`` (exact for a mean)."""

    intercept: float
    slope: float

    def __post_init__(self) -> None:
        _number(self.intercept, "intercept")
        _number(self.slope, "slope")

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "kind": "affine",
            "intercept": float(self.intercept),
            "slope": float(self.slope),
        }


@dataclass(frozen=True, slots=True)
class TrialLevelMapping:
    """The trial-level surrogate relationship, fitted by least squares inside the validated range.

    ``trials`` are ``(surrogate_effect, terminal_effect)`` pairs, one per validation trial;
    ``min_trials`` (at least three) and ``min_r_squared`` are the declared minimums the evidence
    must meet.
    """

    trials: Sequence[tuple[float, float]]
    min_trials: int
    min_r_squared: float

    def __post_init__(self) -> None:
        if isinstance(self.min_trials, bool) or not isinstance(self.min_trials, int):
            raise CausalTypeError("min_trials must be an integer")
        if self.min_trials < 0:
            raise CausalValueError("min_trials must be non-negative")
        _number(self.min_r_squared, "min_r_squared")
        pairs = [tuple(pair) for pair in self.trials]
        if any(len(pair) != 2 for pair in pairs):
            raise CausalValueError("each trial is a (surrogate_effect, terminal_effect) pair")
        for pair in pairs:
            _number(pair[0], "a surrogate effect")
            _number(pair[1], "a terminal effect")

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "kind": "trial_level_linear",
            "trials": [[float(s), float(t)] for s, t in self.trials],
            "min_trials": int(self.min_trials),
            "min_r_squared": float(self.min_r_squared),
        }


@dataclass(frozen=True, slots=True)
class MeasurementCorrection:
    """The binary-misclassification correction of a prevalence or risk by a measurement model."""

    model: MeasurementModel

    def __post_init__(self) -> None:
        if not isinstance(self.model, MeasurementModel):
            raise CausalTypeError("model must be a measurement_models.MeasurementModel")

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {"kind": "measurement_correction", "model": self.model.to_wire()}


_MappingDeclaration = IdentityMapping | AffineMapping | TrialLevelMapping | MeasurementCorrection


@dataclass(frozen=True, slots=True)
class BridgeContract:
    """What is bridged, under which declared assumptions, by which mapping, for which scope.

    ``source`` and ``target`` are :class:`~antecedent.joint_distribution.ScientificQuantity`
    values; ``source_class`` and ``target_class`` say what each is (:data:`QUANTITY_CLASSES`).
    ``evidence_scope`` is the population and regime the bridge's own evidence was established
    for; when it is not the claim's, ``transport`` must state the premise.
    """

    kind: str
    source: Any
    source_class: str
    target: Any
    target_class: str
    evidence_scope: Scope
    mapping: _MappingDeclaration
    assumptions: Sequence[Assumption] = ()
    transport: TransportabilityPremise | None = None
    source_value: str = "mean"
    target_value: str = "mean"

    def __post_init__(self) -> None:
        _choice(self.kind, "kind", BRIDGE_KINDS)
        _choice(self.source_class, "source_class", QUANTITY_CLASSES)
        _choice(self.target_class, "target_class", QUANTITY_CLASSES)
        _choice(self.source_value, "source_value", VALUE_KINDS)
        _choice(self.target_value, "target_value", VALUE_KINDS)
        if not isinstance(self.evidence_scope, Scope):
            raise CausalTypeError("evidence_scope must be a Scope")
        if not isinstance(
            self.mapping,
            (IdentityMapping, AffineMapping, TrialLevelMapping, MeasurementCorrection),
        ):
            raise CausalTypeError("mapping must be one of the mapping declarations")
        if any(not isinstance(a, Assumption) for a in self.assumptions):
            raise CausalTypeError("assumptions must contain Assumption values")
        if self.transport is not None and not isinstance(self.transport, TransportabilityPremise):
            raise CausalTypeError("transport must be a TransportabilityPremise or None")
        _quantity_wire(self.source, "source")
        _quantity_wire(self.target, "target")

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "kind": self.kind,
            "source": {
                "quantity": _quantity_wire(self.source, "source"),
                "class": self.source_class,
            },
            "target": {
                "quantity": _quantity_wire(self.target, "target"),
                "class": self.target_class,
            },
            "source_value": self.source_value,
            "target_value": self.target_value,
            "evidence_scope": self.evidence_scope.to_wire(),
            "mapping": self.mapping.to_wire(),
            "assumptions": [a.to_wire() for a in self.assumptions],
            "transport": None if self.transport is None else self.transport.to_wire(),
        }


@dataclass(frozen=True, slots=True)
class SourceClaim:
    """A source claim: one point value about one quantity, with the identities it carries.

    Use :meth:`declared` for a value the caller states (its identities are content digests of the
    label, value and quantity, labelled as declared). A claim from an executed analysis supplies
    its own ``claim_id`` and identity layers.
    """

    claim_id: str
    value: float | None
    quantity: Any
    units: str | None
    identities: Mapping[str, str | None]
    kind: str = "point"
    evaluated: str = "unknown"
    evidence: Sequence[str] = ()
    execution: str | None = None

    def __post_init__(self) -> None:
        _name(self.claim_id, "claim_id")
        _choice(self.kind, "kind", _CLAIM_KINDS)
        _choice(self.evaluated, "evaluated", _DOMAIN_STATUSES)
        if self.value is not None:
            _number(self.value, "value")
        _quantity_wire(self.quantity, "quantity")
        for key in (
            "target",
            "identification",
            "inference_binding",
            "observation",
            "data_snapshot",
        ):
            if key not in self.identities:
                raise CausalValueError(f"identities must carry `{key}`")

    @classmethod
    def declared(
        cls,
        label: str,
        value: float | None,
        quantity: Any,
        *,
        data_snapshot: str | None = None,
        kind: str = "point",
        evaluated: str = "evaluated",
        evidence: Iterable[str] = (),
    ) -> SourceClaim:
        """A claim the caller declares, with deterministic content-digest identities.

        ``data_snapshot`` names the data the value came from (a label): two claims with the same
        snapshot share evidence, and a bridge of either shares it too.
        """
        _name(label, "label")
        wire = _quantity_wire(quantity, "quantity")
        text = "null" if value is None else repr(float(value))
        snapshot = _digest("declared.data_snapshot", data_snapshot or label)
        return cls(
            claim_id=_digest("declared.claim", label, text, _dumps(wire)),
            value=value,
            quantity=quantity,
            units=str(quantity.units),
            identities={
                "target": _digest("declared.target", _dumps(wire)),
                "identification": _digest("declared.identification", label),
                "identification_product": None,
                "inference_binding": _digest("declared.inference_binding", label),
                "observation": _digest("declared.observation", data_snapshot or label),
                "data_snapshot": snapshot,
            },
            kind=kind,
            evaluated=evaluated,
            evidence=tuple(evidence),
        )

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "claim": {
                "claim_id": self.claim_id,
                "kind": self.kind,
                "value": None if self.value is None else float(self.value),
                "units": self.units,
                "identities": dict(self.identities),
                "evaluated": self.evaluated,
                "evidence": list(self.evidence),
                "execution": self.execution,
            },
            "quantity": _quantity_wire(self.quantity, "quantity"),
        }


# --------------------------------------------------------------------------- results


@dataclass(frozen=True, slots=True)
class TrialFit:
    """The trial-level least-squares line and its coefficient of determination."""

    intercept: float
    slope: float
    r_squared: float
    trials: int
    surrogate_min: float
    surrogate_max: float


@dataclass(frozen=True, slots=True)
class Freshness:
    """Applicability of a bridged claim given the standing of its source."""

    historically_valid: bool
    currently_applicable: bool | None


@dataclass(frozen=True, slots=True)
class BridgeExport:
    """An exported bridge: the bytes and the artifact identity a consumer retains."""

    artifact: bytes
    identity: str


@dataclass(frozen=True, slots=True, eq=False)
class BridgedClaim:
    """A claim derived across a bridge.

    ``claim_id`` is not ``source_claim_id``: the derived claim is a different claim whose identity
    binds the contract (``contract_identity``), every declared assumption and the source claim.
    ``unresolved`` lists the declared assumptions the bridge did not check; ``unavailable``
    lists what the claim cannot offer. Identification, support and uncertainty are never
    established for the target quantity by the bridge itself.
    """

    kind: str
    value: float
    source_value: float
    value_kind: str
    mapping: str
    fit: TrialFit | None
    claim_id: str
    source_claim_id: str
    contract_identity: str
    transported: bool
    units: str | None
    assumptions: tuple[Assumption, ...]
    transportability_premise: str | None
    unresolved: tuple[str, ...]
    unavailable: tuple[str, ...]
    evidence: tuple[str, ...]
    domains: Mapping[str, str]
    identification_available: bool
    support_available: bool
    uncertainty_available: bool
    provenance: tuple[Mapping[str, Any], ...]
    _handle: Any = field(repr=False, compare=False)

    def require_value_kind(self, requested: str) -> None:
        """Raise the refusal when a consumer requires another kind of object than a mean.

        Raises:
            ScientificBridgeRefusal: ``bridge_substitution_refused`` /
                ``scientific_bridge.value_kind_substitution``.
        """
        _choice(requested, "requested", VALUE_KINDS)
        raise_refusal(self._handle.require_value_kind(requested), ScientificBridgeRefusal)

    def interval(self) -> None:
        """Always raises: no interval, posterior or joint law is licensed for a bridged claim.

        Raises:
            ScientificBridgeRefusal: ``cell_not_licensed`` /
                ``scientific_bridge.interval_not_licensed``.
        """
        raise_refusal(self._handle.interval(), ScientificBridgeRefusal)

    def freshness(
        self, *, source_withdrawn: bool = False, source_superseded: bool = False
    ) -> Freshness:
        """Applicability given the source's standing; history is preserved either way."""
        wire = json.loads(
            self._handle.freshness_json(bool(source_withdrawn), bool(source_superseded))
        )
        return Freshness(wire["historically_valid"], wire["currently_applicable"])

    def source(self) -> dict[str, Any]:
        """The source claim as the derivation saw it."""
        return dict(json.loads(self._handle.source_json()))

    def export(self, artifact_id: str = "scientific_bridge") -> BridgeExport:
        """Write the ``scientific_bridge_v1`` artifact.

        Raises:
            ScientificBridgeRefusal: the bridge does not seal.
        """
        export = getattr(self._handle, "export", None)
        if export is None:
            raise CausalValueError(
                "a consumed bridge is re-exported from the bytes it was consumed from"
            )
        done = _unwrap(export(_name(artifact_id, "artifact_id")))
        data, identity = done
        return BridgeExport(bytes(data), identity)


def _fit(wire: Mapping[str, Any] | None) -> TrialFit | None:
    if wire is None:
        return None
    return TrialFit(
        intercept=float(wire["intercept"]),
        slope=float(wire["slope"]),
        r_squared=float(wire["r_squared"]),
        trials=int(wire["trials"]),
        surrogate_min=float(wire["surrogate_min"]),
        surrogate_max=float(wire["surrogate_max"]),
    )


def _claim(summary: Mapping[str, Any], handle: Any) -> BridgedClaim:
    contract = summary["contract"]
    transport = contract["transport"]
    return BridgedClaim(
        kind=summary["kind"],
        value=float(summary["value"]),
        source_value=float(summary["source_value"]),
        value_kind=summary["value_kind"],
        mapping=summary["mapping"],
        fit=_fit(summary["fit"]),
        claim_id=summary["claim_id"],
        source_claim_id=summary["source_claim_id"],
        contract_identity=summary["contract_identity"],
        transported=bool(summary["transported"]),
        units=summary["units"],
        assumptions=tuple(
            Assumption(a["id"], a["kind"], a["statement"], a["evidence"])
            for a in contract["assumptions"]
        ),
        transportability_premise=None if transport is None else transport["id"],
        unresolved=tuple(summary["unresolved"]),
        unavailable=tuple(summary["unavailable"]),
        evidence=tuple(summary["evidence"]),
        domains=dict(summary["domains"]),
        identification_available=bool(summary["identification_available"]),
        support_available=bool(summary["support_available"]),
        uncertainty_available=bool(summary["uncertainty_available"]),
        provenance=tuple(summary["provenance"]),
        _handle=handle,
    )


# --------------------------------------------------------------------------- prepare


@dataclass(frozen=True, slots=True, eq=False)
class PreparedBridge:
    """A bridge contract checked once, with its licensing verdict kept.

    :meth:`derive` and :meth:`with_contract` return new values and leave this one untouched.
    """

    contract: BridgeContract
    contract_identity: str
    transported: bool
    _handle: Any = field(repr=False, compare=False)

    def derive(self, source: SourceClaim) -> BridgedClaim:
        """Derive the claim for ``source``.

        A source claim about the same quantity (a refreshed estimate) keeps the contract, its
        identity, the assumptions and the verdict, and recomputes the value, the fit and the
        claim identity.

        Raises:
            ScientificBridgeRefusal: a claim about another quantity or in other units, a claim
                that is not a finite point value, or a value outside the mapping's validated or
                attainable range.
        """
        if not isinstance(source, SourceClaim):
            raise CausalTypeError("source must be a SourceClaim")
        handle = _unwrap(self._handle.derive(_dumps(source.to_wire())))
        return _claim(json.loads(handle.summary_json()), handle)

    def with_contract(self, contract: BridgeContract) -> PreparedBridge:
        """A prepared bridge for a replacement contract, re-checked in full.

        Raises:
            ScientificBridgeRefusal: the first defect of the replacement; ``self`` is unchanged.
        """
        if not isinstance(contract, BridgeContract):
            raise CausalTypeError("contract must be a BridgeContract")
        handle = _unwrap(self._handle.with_contract(_dumps(contract.to_wire())))
        return _prepared(contract, handle)


def _prepared(contract: BridgeContract, handle: Any) -> PreparedBridge:
    return PreparedBridge(
        contract=contract,
        contract_identity=handle.contract_identity(),
        transported=bool(handle.transported()),
        _handle=handle,
    )


def prepare_scientific_bridge(contract: BridgeContract) -> PreparedBridge:
    """Check a bridge contract before any number is formed.

    Raises:
        ScientificBridgeRefusal: the first defect, with its registered ``reason_code``,
            ``scientific_bridge.*`` ``detail`` and the ``remedy`` that would license it: a
            missing or evidence-less assumption, a proxy offered as the construct, an
            association offered as an intervention, a stated preference offered as realized
            behavior, a predictive importance offered as a causal effect, a model score offered
            as a probability, a scientific endpoint offered as a utility, a population or regime
            mismatch, a transported bridge without its premise, or a value of another kind.
    """
    if not isinstance(contract, BridgeContract):
        raise CausalTypeError("contract must be a BridgeContract")
    handle = _unwrap(_native.scientific_bridge_prepare(_dumps(contract.to_wire())))
    return _prepared(contract, handle)


def contract_identity(contract: BridgeContract) -> str:
    """The identity of a contract declaration, without checking its licensing."""
    if not isinstance(contract, BridgeContract):
        raise CausalTypeError("contract must be a BridgeContract")
    return str(_native.scientific_bridge_contract_identity(_dumps(contract.to_wire())))


# --------------------------------------------------------------------------- consume


@dataclass(frozen=True, slots=True, eq=False)
class ConsumedBridge:
    """A consumed artifact: the replayed derivation and the source claim as the bridge saw it.

    ``source`` reports the identity layers, value, units and evidence the bridge depended on;
    the replayed source never claims identification or support it cannot show
    (``identification_domain`` and ``support_domain`` are ``unknown``).
    """

    claim: BridgedClaim
    source: Mapping[str, Any]
    identity: str
    _handle: Any = field(repr=False, compare=False)

    def interval(self) -> None:
        """Always raises: a consumed bridge carries no interval license.

        Raises:
            ScientificBridgeRefusal: ``cell_not_licensed``.
        """
        raise_refusal(self._handle.interval(), ScientificBridgeRefusal)


def consume_scientific_bridge_artifact(
    artifact: bytes, expected_identity: str | None = None
) -> ConsumedBridge:
    """Independently consume a scientific bridge artifact by replaying its derivation.

    ``expected_identity`` is the artifact identity the consumer retained independently of the
    bytes; a consistently resealed artifact of another bridge is refused. The consumer rebuilds
    the contract and the source claim, applies every licensing check again, derives the claim
    again and requires the stored identity, value bits, fit and transport flag to match.

    Raises:
        ScientificBridgeRefusal: ``route_not_supported`` for an identity mismatch,
            ``invalid_argument`` for corrupt or foreign bytes or a value that does not replay, or
            the library's own refusal of a stored declaration that no longer licenses.
    """
    if not isinstance(artifact, (bytes, bytearray, memoryview)):
        raise CausalTypeError("artifact must be bytes")
    if expected_identity is not None:
        _name(expected_identity, "expected_identity")
    handle = _unwrap(_native.scientific_bridge_consume(bytes(artifact), expected_identity))
    summary = json.loads(handle.summary_json())
    return ConsumedBridge(
        claim=_claim(summary, handle),
        source=summary["source"],
        identity=summary["identity"],
        _handle=handle,
    )
