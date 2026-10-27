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
    claim.answer.value               # 0.796: 0.06 + 0.92 * 0.8 from the trial-level line
    exported = claim.export("surrogate-bridge")
    again = antecedent.load(exported, expected=exported.expectation)

A bridge produces a *new* claim with its own identity (binding the contract, every declared
assumption and the source claim), its own provenance, and a receipt of what it did not establish.
The source claim is not modified. A bridge carries one mean. The interval of that mean
(:meth:`BridgedClaim.interval` with a ``level``) carries the source claim's declared
``standard_error`` through the mapping together with the mapping's own estimation uncertainty, and
is published only inside a measured design; with no ``level``, a posterior, a predictive law, an
identified set or a joint law the request is refused
(:meth:`BridgedClaim.interval`, :meth:`BridgedClaim.require_value_kind`).

A point source whose ``standard_error`` is declared, through a mapping that declares its
:class:`MappingUncertainty` and :class:`SourceDependence`, also carries the first-order propagated
variance of the bridged value (:attr:`BridgedClaim.propagated_variance`,
:meth:`BridgedClaim.propagated_uncertainty`), read as an ``uncalibrated_spread`` whose calibration
is unmeasured and never as an interval. The claim ``value`` stays the plug-in value; the propagated
mean (bias-corrected when ``SourceClaim.nonlinear`` asks for the expectation of a ``square``) is
reported beside it. The dependence is never assumed: a declared uncertainty with no declared
dependence is refused ``scientific_bridge.source_dependence_missing``.

A bridge whose source is the aligned draws of a posterior or a predictive law
(:func:`prepare_distribution_bridge`) derives the same draw table with the bridged column in
place of the source column, row for row and with the table's alignment identity unchanged, so the
bridged column still pairs by index with every column it was sampled with::

    contract = sb.BridgeContract(
        kind="surrogate_terminal_outcome",
        source=surrogate, source_class="surrogate_endpoint",
        target=terminal, target_class="terminal_outcome",
        evidence_scope=sb.Scope("adults", "do(t=1)"),
        mapping=sb.AffineMapping(intercept=0.5, slope=2.0),
        mapping_uncertainty=sb.MappingUncertainty("calibration-fit", slope_variance=0.04),
        source_dependence=sb.SourceDependence.independent("disjoint-samples"),
        source_value="posterior", target_value="posterior",
        assumptions=[...],
    )
    bridged = sb.prepare_distribution_bridge(contract).derive(posterior_draws, seed=7)
    bridged.distribution()      # the aligned draws of the terminal outcome
    bridged.reading             # "uncalibrated_spread": an interval needs a measured coverage record

An affine map with uncertain coefficients draws them jointly from its declared
:class:`MappingUncertainty` and :class:`SourceDependence`; a trial-level line draws them from the
Student-t law of its least-squares fit; a measurement correction draws its rates. The source's
spread comes only from its draws and the mapping's only from the mapping, so neither is counted
twice. :meth:`BridgedDistribution.chain_stage` expresses every bridge as a measurement-mapping stage
of a propagation chain: an affine map or trial-level line reads its slope and intercept as inputs, a
predictive law adds the new trial's residual as an input of its own, and a measurement correction is
the guarded ratio of the source and the drawn sensitivity and specificity.

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
from typing import Any, Literal

from . import _native
from ._artifact_protocol import (
    ExportedArtifact,
    artifact_consumer,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from ._freeze import thaw
from ._mean_binding import bind_mean_reading
from ._refusal import StructuredRefusal, raise_refusal
from ._scientific import ScientificResult
from ._validate import count as _count
from .errors import CausalTypeError, CausalValueError
from .measurement_models import MeasurementModel
from .outcomes import Answer, Outcome, outcome_operation

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
    "BridgedClaimInterval",
    "BridgedDistribution",
    "ChainStageParts",
    "ConsumedBridge",
    "ConsumedBridgeDistribution",
    "EstimandBridge",
    "ExposureSetting",
    "Freshness",
    "IdentityMapping",
    "MappingUncertainty",
    "MeasurementCorrection",
    "PreparedBridge",
    "PreparedDistributionBridge",
    "PropagatedUncertainty",
    "Scope",
    "ScientificBridgeRefusal",
    "SourceClaim",
    "SourceDependence",
    "TransportabilityPremise",
    "TrialFit",
    "TrialLevelMapping",
    "consume_scientific_bridge_artifact",
    "contract_identity",
    "prepare_distribution_bridge",
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
    "associational_quantity",
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

_CLAIM_KINDS = (
    "point",
    "bounds",
    "mixture",
    "response",
    "policy_value",
    "refusal",
    "incomplete",
    "associational",
)
_NONLINEAR_MAPS = ("square", "exponential")
_DOMAIN_STATUSES = (
    "identified",
    "supported",
    "evaluated",
    "outside_scope",
    "contradicted",
    "unknown",
    "not_claimed",
)


BridgeKindsName = Literal[
    "latent_measured_outcome",
    "preference_realized_behavior",
    "surrogate_terminal_outcome",
    "predictive_causal_quantity",
    "parameter_intervention_response",
    "scientific_decision_utility",
]
QuantityClassesName = Literal[
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
    "associational_quantity",
]
AssumptionKindsName = Literal[
    "measurement_model",
    "behavioral_linkage",
    "surrogate_validation",
    "identification",
    "parameter_interpretation",
    "utility_mapping",
    "additional",
]
ValueKindsName = Literal[
    "mean", "predictive_law", "posterior", "identified_set", "joint_outcome_law"
]
ClaimKindsName = Literal[
    "point",
    "bounds",
    "mixture",
    "response",
    "policy_value",
    "refusal",
    "incomplete",
    "associational",
]
DomainStatusesName = Literal[
    "identified",
    "supported",
    "evaluated",
    "outside_scope",
    "contradicted",
    "unknown",
    "limited",
    "not_claimed",
]


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
    return json.dumps(thaw(value), allow_nan=False, separators=(",", ":"))


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
            "min_trials": _count(
                self.min_trials,
                "min_trials",
                minimum=0,
                maximum=2**64 - 1,
                reason_code="invalid_argument",
            ),
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
class MappingUncertainty:
    """The variance of a declared affine map's coefficients, apart from the source estimate.

    A variance of zero states the coefficient exact. ``intercept_slope_covariance`` of zero
    declares the two estimators independent; a regression calibration usually has a negative one
    (about ``-mean(x) * slope_variance``). Its square must not exceed the product of the variances.
    """

    id: str
    intercept_variance: float = 0.0
    slope_variance: float = 0.0
    intercept_slope_covariance: float = 0.0

    def __post_init__(self) -> None:
        _name(self.id, "id")
        for label, value in (
            ("intercept_variance", self.intercept_variance),
            ("slope_variance", self.slope_variance),
        ):
            if _number(value, label) < 0:
                raise CausalValueError(f"{label} must be non-negative")
        _number(self.intercept_slope_covariance, "intercept_slope_covariance")

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "id": self.id,
            "intercept_variance": float(self.intercept_variance),
            "slope_variance": float(self.slope_variance),
            "intercept_slope_covariance": float(self.intercept_slope_covariance),
        }

    @classmethod
    def from_wire(cls, wire: Mapping[str, Any]) -> MappingUncertainty:
        """The declaration a native summary reports."""
        return cls(
            wire["id"],
            float(wire["intercept_variance"]),
            float(wire["slope_variance"]),
            float(wire["intercept_slope_covariance"]),
        )


@dataclass(frozen=True, slots=True)
class SourceDependence:
    """How the source estimate and the mapping share information.

    Independence is a declaration naming its evidence; ``slope_covariance`` is a different
    declaration that states a number, the covariance of the slope estimator and the source
    estimator. An absent dependence is not a declaration of independence.
    """

    kind: Literal["independent", "slope_covariance"]
    evidence: str
    covariance: float | None = None

    def __post_init__(self) -> None:
        _choice(self.kind, "kind", ("independent", "slope_covariance"))
        _name(self.evidence, "evidence")
        if self.kind == "slope_covariance":
            if self.covariance is None:
                raise CausalValueError("a slope covariance declares its value")
            _number(self.covariance, "covariance")
        elif self.covariance is not None:
            raise CausalValueError("an independence declaration states no covariance")

    @classmethod
    def independent(cls, evidence: str) -> SourceDependence:
        """The source estimate and the mapping's evidence are disjoint."""
        return cls("independent", evidence)

    @classmethod
    def slope_covariance(cls, covariance: float, evidence: str) -> SourceDependence:
        """The covariance of the slope estimator and the source estimator."""
        return cls("slope_covariance", evidence, covariance)

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        if self.kind == "independent":
            return {"kind": "independent", "evidence": self.evidence}
        return {
            "kind": "slope_covariance",
            "covariance": float(self.covariance or 0.0),
            "evidence": self.evidence,
        }

    @classmethod
    def from_wire(cls, wire: Mapping[str, Any]) -> SourceDependence:
        """The declaration a native summary reports."""
        if wire["kind"] == "independent":
            return cls.independent(wire["evidence"])
        return cls.slope_covariance(float(wire["covariance"]), wire["evidence"])


@dataclass(frozen=True, slots=True)
class ExposureSetting:
    """One side of an estimand bridge: population, window, dose, duration and exposure regime."""

    population_id: str
    time_start: str
    time_end: str
    dose: float
    dose_units: str
    duration: float
    duration_units: str
    exposure_regime: str

    def __post_init__(self) -> None:
        for label in (
            "population_id",
            "time_start",
            "time_end",
            "dose_units",
            "duration_units",
            "exposure_regime",
        ):
            _name(getattr(self, label), label)
        _number(self.dose, "dose")
        if _number(self.duration, "duration") < 0:
            raise CausalValueError("duration must be non-negative")

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "population_id": self.population_id,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "dose": float(self.dose),
            "dose_units": self.dose_units,
            "duration": float(self.duration),
            "duration_units": self.duration_units,
            "exposure_regime": self.exposure_regime,
        }

    @classmethod
    def from_wire(cls, wire: Mapping[str, Any]) -> ExposureSetting:
        """The declaration a native summary reports."""
        return cls(
            wire["population_id"],
            wire["time_start"],
            wire["time_end"],
            float(wire["dose"]),
            wire["dose_units"],
            float(wire["duration"]),
            wire["duration_units"],
            wire["exposure_regime"],
        )


@dataclass(frozen=True, slots=True)
class EstimandBridge:
    """The translation of an experiment's estimand into the estimand the bridge answers.

    Recency and spend are differences of the target relative to the experiment; zero declares that
    they do not differ. The spend difference must equal the target dose minus the experiment dose.
    """

    id: str
    experiment_id: str
    experiment: ExposureSetting
    target: ExposureSetting
    recency_delta: float
    recency_units: str
    spend_delta: float
    spend_units: str

    def __post_init__(self) -> None:
        _name(self.id, "id")
        _name(self.experiment_id, "experiment_id")
        if not isinstance(self.experiment, ExposureSetting) or not isinstance(
            self.target, ExposureSetting
        ):
            raise CausalTypeError("experiment and target must be ExposureSetting values")
        _number(self.recency_delta, "recency_delta")
        _name(self.recency_units, "recency_units")
        _number(self.spend_delta, "spend_delta")
        _name(self.spend_units, "spend_units")

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "experiment": self.experiment.to_wire(),
            "target": self.target.to_wire(),
            "recency_delta": float(self.recency_delta),
            "recency_units": self.recency_units,
            "spend_delta": float(self.spend_delta),
            "spend_units": self.spend_units,
        }

    @classmethod
    def from_wire(cls, wire: Mapping[str, Any]) -> EstimandBridge:
        """The declaration a native summary reports."""
        return cls(
            wire["id"],
            wire["experiment_id"],
            ExposureSetting.from_wire(wire["experiment"]),
            ExposureSetting.from_wire(wire["target"]),
            float(wire["recency_delta"]),
            wire["recency_units"],
            float(wire["spend_delta"]),
            wire["spend_units"],
        )


def _extension(contract: Mapping[str, Any]) -> dict[str, Any]:
    """The extended declarations a native contract summary reports, as typed values."""
    return {
        "mapping_uncertainty": (
            None
            if contract.get("mapping_uncertainty") is None
            else MappingUncertainty.from_wire(contract["mapping_uncertainty"])
        ),
        "source_dependence": (
            None
            if contract.get("source_dependence") is None
            else SourceDependence.from_wire(contract["source_dependence"])
        ),
        "estimand_bridge": (
            None
            if contract.get("estimand_bridge") is None
            else EstimandBridge.from_wire(contract["estimand_bridge"])
        ),
    }


@dataclass(frozen=True, slots=True)
class BridgeContract:
    """What is bridged, under which declared assumptions, by which mapping, for which scope.

    ``source`` and ``target`` are :class:`~antecedent.joint_distribution.ScientificQuantity`
    values; ``source_class`` and ``target_class`` say what each is (:data:`QUANTITY_CLASSES`).
    ``evidence_scope`` is the population and regime the bridge's own evidence was established
    for; when it is not the claim's, ``transport`` must state the premise.
    """

    kind: BridgeKindsName
    source: Any
    source_class: QuantityClassesName
    target: Any
    target_class: QuantityClassesName
    evidence_scope: Scope
    mapping: _MappingDeclaration
    assumptions: Sequence[Assumption] = ()
    transport: TransportabilityPremise | None = None
    source_value: ValueKindsName = "mean"
    target_value: ValueKindsName = "mean"
    mapping_uncertainty: MappingUncertainty | None = None
    source_dependence: SourceDependence | None = None
    estimand_bridge: EstimandBridge | None = None

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
        for label, value, expected in (
            ("mapping_uncertainty", self.mapping_uncertainty, MappingUncertainty),
            ("source_dependence", self.source_dependence, SourceDependence),
            ("estimand_bridge", self.estimand_bridge, EstimandBridge),
        ):
            if value is not None and not isinstance(value, expected):
                raise CausalTypeError(f"{label} must be a {expected.__name__} or None")
        _quantity_wire(self.source, "source")
        _quantity_wire(self.target, "target")

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads.

        The extended declarations are listed only where the contract makes them, so a contract
        that declares none reads exactly as before.
        """
        wire: dict[str, Any] = {
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
        if self.mapping_uncertainty is not None:
            wire["mapping_uncertainty"] = self.mapping_uncertainty.to_wire()
        if self.source_dependence is not None:
            wire["source_dependence"] = self.source_dependence.to_wire()
        if self.estimand_bridge is not None:
            wire["estimand_bridge"] = self.estimand_bridge.to_wire()
        return wire


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
    standard_error: float | None = None
    nonlinear: str | None = None

    def __post_init__(self) -> None:
        _name(self.claim_id, "claim_id")
        if self.standard_error is not None:
            _number(self.standard_error, "standard_error")
            if self.standard_error < 0:
                raise CausalValueError("standard_error must be non-negative")
        if self.nonlinear is not None:
            _choice(self.nonlinear, "nonlinear", _NONLINEAR_MAPS)
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
        standard_error: float | None = None,
        nonlinear: str | None = None,
    ) -> SourceClaim:
        """A claim the caller declares, with deterministic content-digest identities.

        ``data_snapshot`` names the data the value came from (a label): two claims with the same
        snapshot share evidence, and a bridge of either shares it too. ``standard_error`` is the
        claim's own sampling standard error, declared with it: it is what
        :meth:`BridgedClaim.interval` carries through the bridge and, where the mapping declares
        its uncertainty and dependence, what the propagated variance is built from (then it binds
        the derived claim's identity). ``nonlinear`` (``"square"``) asks for the second-moment
        expectation of a nonlinear map of the bridged quantity under the propagated law.
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
            standard_error=standard_error,
            nonlinear=nonlinear,
        )

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "standard_error": None if self.standard_error is None else float(self.standard_error),
            "nonlinear": self.nonlinear,
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


@dataclass(repr=False, frozen=True, slots=True)
class Freshness(ScientificResult):
    """Applicability of a bridged claim given the standing of its source."""

    historically_valid: bool
    currently_applicable: bool | None


@dataclass(repr=False, frozen=True, slots=True)
class BridgedClaimInterval(ScientificResult):
    """The interval of a bridged value.

    ``standard_error`` is the source claim's declared ``source_standard_error`` propagated through
    the mapping together with the mapping's own estimation uncertainty. ``degrees_of_freedom`` is
    ``None`` for the normal reference. ``licensed_by`` is ``coverage_record`` (a current record,
    named by ``coverage_record``, measured the construction) or
    ``exact_transform_of_declared_source_interval`` (an identity or declared affine map adds no
    uncertainty, so the level is the declared source interval's own). ``assumptions`` states what
    the interval rests on. It is an interval for the value under the bridge's declared
    assumptions: those stay unresolved obligations.
    """

    lower: float
    upper: float
    level: float
    source_value: float
    value: float
    source_standard_error: float
    standard_error: float
    degrees_of_freedom: float | None
    mapping: str
    construction: str
    licensed_by: str
    coverage_record: str
    calibration_sha: str
    assumptions: str


@dataclass(repr=False, frozen=True, slots=True)
class PropagatedUncertainty(ScientificResult):
    """The first-order propagated uncertainty of a bridged value.

    ``plug_in_value`` is the claim value; ``mean`` is the propagated mean (the plug-in value plus
    the declared slope-source covariance) and ``propagated_mean`` is the bias-corrected expectation
    when a ``nonlinear`` map was asked for. ``variance`` is read as ``reading``
    (``uncalibrated_spread``) with ``calibration`` ``unmeasured``: it is not an interval, which
    stays the separate, scoped :meth:`BridgedClaim.interval`. ``dependence`` is the declared
    dependence the variance was built from.
    """

    construction: str
    source_value: float
    source_standard_error: float
    plug_in_value: float
    mean: float
    variance: float
    propagated_mean: float
    reading: str
    calibration: str
    mapping_uncertainty: str | None
    dependence: Mapping[str, Any]
    nonlinear: Mapping[str, Any] | None

    @property
    def standard_deviation(self) -> float:
        """The square root of the propagated variance."""
        return math.sqrt(self.variance)


def _propagation(wire: Mapping[str, Any] | None) -> PropagatedUncertainty | None:
    if wire is None:
        return None
    return PropagatedUncertainty(
        construction=str(wire["construction"]),
        source_value=float(wire["source_value"]),
        source_standard_error=float(wire["source_standard_error"]),
        plug_in_value=float(wire["plug_in_value"]),
        mean=float(wire["mean"]),
        variance=float(wire["variance"]),
        propagated_mean=float(wire["propagated_mean"]),
        reading=str(wire["reading"]),
        calibration=str(wire["calibration"]),
        mapping_uncertainty=wire["mapping_uncertainty"],
        dependence=dict(wire["dependence"]),
        nonlinear=None if wire["nonlinear"] is None else dict(wire["nonlinear"]),
    )


@dataclass(frozen=True, slots=True)
class BridgeExport:
    """An exported bridge: the bytes and the artifact identity a consumer retains."""

    artifact: bytes
    identity: str


@dataclass(repr=False, frozen=True, slots=True, eq=False)
class BridgedClaim(ScientificResult):
    """A claim derived across a bridge.

    ``claim_id`` is not ``source_claim_id``: the derived claim is a different claim whose identity
    binds the contract (``contract_identity``), every declared assumption and the source claim.
    ``unresolved`` lists the declared assumptions the bridge did not check; ``unavailable``
    lists what the claim cannot offer. Identification, support and calibrated uncertainty are
    never established for the target quantity by the bridge itself. ``propagation`` (and
    ``propagated_variance`` / ``propagated_mean``) is the first-order propagated uncertainty of the
    bridged value when the source declared a ``standard_error`` and the mapping declared its
    uncertainty and dependence; ``components`` lists the recorded uncertainty components.
    """

    _mean_binding: Any = field(default=None, init=False, repr=False, compare=False)

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
    quantity: Any = None
    mapping_uncertainty: MappingUncertainty | None = None
    source_dependence: SourceDependence | None = None
    estimand_bridge: EstimandBridge | None = None
    propagation: PropagatedUncertainty | None = None
    components: tuple[Mapping[str, Any], ...] = ()

    @property
    def answer(self) -> Answer:
        return Answer("point", value=self.value, detail=self.value_kind)

    @property
    def identities(self) -> dict[str, str]:
        return {
            "claim_id": self.claim_id,
            "source_claim_id": self.source_claim_id,
            "contract_identity": self.contract_identity,
        }

    def inspect(self) -> Any:
        from .results._report import SlotModel

        propagation = self.propagation
        return ScientificResult.inspect(self).model_copy(
            update={
                "identification": SlotModel(
                    available=self.identification_available,
                    summary="bridge identification standing",
                ),
                "support": SlotModel(
                    available=self.support_available, summary="bridge support standing"
                ),
                "uncertainty": SlotModel(
                    available=self.uncertainty_available,
                    reason="no_calibrated_interval" if not self.uncertainty_available else None,
                    summary="point_only" if propagation is None else propagation.reading,
                    payload={}
                    if propagation is None
                    else {
                        "variance": propagation.variance,
                        "propagated_mean": propagation.propagated_mean,
                        "calibration": propagation.calibration,
                        "construction": propagation.construction,
                    },
                ),
                "assumptions": SlotModel(
                    available=True,
                    summary="declared bridge assumptions",
                    payload={"count": len(self.assumptions), "unresolved": self.unresolved},
                ),
            }
        )

    @property
    def propagated_variance(self) -> float | None:
        """The propagated variance of the bridged value, or ``None`` when none was carried."""
        return None if self.propagation is None else self.propagation.variance

    @property
    def propagated_mean(self) -> float | None:
        """The propagated mean (bias-corrected for a nonlinear map), or ``None``."""
        return None if self.propagation is None else self.propagation.propagated_mean

    @outcome_operation
    def propagated_uncertainty(self) -> PropagatedUncertainty:
        """The propagated variance and mean of the bridged value.

        Raises:
            ScientificBridgeRefusal: ``scientific_bridge.source_variance_invalid`` when the source
                claim declared no ``standard_error``;
                ``scientific_bridge.mapping_uncertainty_missing`` when an affine map declared no
                uncertainty; ``scientific_bridge.source_dependence_missing`` when the dependence
                the propagation needs is undeclared (it is never assumed independent).
        """
        if self.propagation is not None:
            return self.propagation
        payload = _unwrap(self._handle.propagated_uncertainty())
        found = _propagation(json.loads(payload))
        if found is None:  # pragma: no cover - the native layer refuses instead
            raise CausalValueError("the bridge carries no propagated uncertainty")
        return found

    def require_value_kind(self, requested: str) -> None:
        """Raise the refusal when a consumer requires another kind of object than a mean.

        Raises:
            ScientificBridgeRefusal: ``bridge_substitution_refused`` /
                ``scientific_bridge.value_kind_substitution``.
        """
        _choice(requested, "requested", VALUE_KINDS)
        raise_refusal(self._handle.require_value_kind(requested), ScientificBridgeRefusal)

    @outcome_operation
    def interval(self, level: float | None = None) -> BridgedClaimInterval:
        """The interval of the bridged value at ``level``.

        The source claim's declared ``standard_error`` is carried through the mapping together
        with the mapping's own estimation uncertainty (the trial-level line's prediction error,
        or the declared variances of the sensitivity and specificity). It is published only where
        a current coverage record measured this construction at this bridge kind, mapping,
        direction and validated trial count, and states what it rests on. With no ``level`` the
        request is unscoped and, like a posterior or joint law of a bridged claim, always raises.

        Raises:
            ScientificBridgeRefusal: ``cell_not_licensed`` /
                ``scientific_bridge.interval_not_licensed`` for an unscoped request;
                ``scientific_bridge.interval_source_uncertainty_required`` when the source claim
                declared no ``standard_error``; ``scientific_bridge.interval_scope_outside_measured``
                outside a measured design.
        """
        scoped = getattr(self._handle, "bridged_interval", None)
        if level is None or scoped is None:
            raise_refusal(self._handle.interval(), ScientificBridgeRefusal)
            raise AssertionError("an unscoped interval request always refuses")  # pragma: no cover
        payload = _unwrap(scoped(_number(level, "level")))
        return BridgedClaimInterval(**json.loads(payload))

    @outcome_operation
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

    @artifact_export
    def export(self, artifact_id: str = "scientific_bridge") -> ExportedArtifact[ConsumedBridge]:
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
        return artifact_from_receipt(self, BridgeExport(bytes(data), identity))


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


@bind_mean_reading
def _claim(summary: Mapping[str, Any], handle: Any) -> BridgedClaim:
    from .joint_distribution import ScientificQuantity

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
        quantity=ScientificQuantity.from_wire(contract["target"]["quantity"]),
        propagation=_propagation(summary.get("propagation")),
        components=tuple(summary.get("uncertainty_components", ())),
        **_extension(contract),
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

    @outcome_operation
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


@dataclass(repr=False, frozen=True, slots=True, eq=False)
class ConsumedBridge(ScientificResult):
    """A consumed artifact: the replayed derivation and the source claim as the bridge saw it.

    ``source`` reports the identity layers, value, units and evidence the bridge depended on;
    the replayed source never claims identification or support it cannot show
    (``identification_domain`` and ``support_domain`` are ``unknown``).
    """

    _mean_binding: Any = field(default=None, init=False, repr=False, compare=False)

    claim: BridgedClaim
    source: Mapping[str, Any]
    identity: str
    _handle: Any = field(repr=False, compare=False)

    @property
    def answer(self) -> Answer:
        return self.claim.answer

    def inspect(self) -> Any:
        return self.claim.inspect().model_copy(update={"stage": "consumed"})

    @outcome_operation
    def interval(self) -> None:
        """Always raises: a consumed bridge carries no interval license.

        Raises:
            ScientificBridgeRefusal: ``cell_not_licensed``.
        """
        raise_refusal(self._handle.interval(), ScientificBridgeRefusal)


@outcome_operation
@artifact_consumer(
    proof="scientific_bridge.consume_scientific_bridge_artifact: independent semantic consumer checks declarations, identities and stored result; recomputes the scientific result",
    replayable=True,
)
def consume_scientific_bridge_artifact(
    artifact: bytes, expected_identity: str | None = None
) -> ConsumedBridge | ConsumedBridgeDistribution:
    """Independently consume a scientific bridge artifact by replaying its derivation.

    A bridge of one mean returns a :class:`ConsumedBridge`; a bridge of draws maps its stored draws
    again and returns a :class:`ConsumedBridgeDistribution`. The artifact's container tells which.
    """
    artifact = container_bytes(artifact)
    if not isinstance(artifact, (bytes, bytearray, memoryview)):
        raise CausalTypeError("artifact must be bytes")
    shape, refusal = _native.scientific_bridge_shape(bytes(artifact))
    raise_refusal(refusal, ScientificBridgeRefusal)
    if shape == "draws":
        return _consume_distribution(bytes(artifact), expected_identity)
    return _consume_point(bytes(artifact), expected_identity)


@bind_mean_reading
def _consume_point(artifact: bytes, expected_identity: str | None = None) -> ConsumedBridge:
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
    artifact = container_bytes(artifact)
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


# ------------------------------------------------------------------ bridges of distributions


def _seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**64:
        raise CausalValueError("seed must be an integer in [0, 2**64)")
    return value


@dataclass(frozen=True, slots=True)
class ChainStageParts:
    """A bridge as a stage of a propagation chain, with the sources it reads.

    ``sources`` are the source draws and the draws of the mapping (the drawn slope and intercept,
    plus the new trial's residual for a predictive law; or the drawn sensitivity and specificity
    of a measurement correction), columns of one aligned draw table; ``stage`` is the
    measurement-mapping stage that reads them. Pass both to
    :func:`antecedent.propagation_chain.propagate` together with the stages that read
    ``stage.id``.
    """

    sources: tuple[Any, ...]
    stage: Any


@dataclass(repr=False, frozen=True, slots=True, eq=False)
class BridgedDistribution(ScientificResult):
    """A claim derived across a bridge from the aligned draws of a posterior or a predictive law.

    :meth:`distribution` is the source draw table with the bridged column in place of the source
    column, row for row and with the source table's alignment identity (``alignment_identity``),
    so the bridged column still pairs by index with every column it was sampled with. ``mean`` and
    ``variance`` summarize it. ``reading`` is ``uncalibrated_spread`` and ``calibration`` is
    ``unmeasured``: the spread is not an interval, and :meth:`interval` publishes one only inside
    a measured design. The bridge's declared assumptions stay in ``unresolved``, with the
    assumption that the mapping's evidence is independent of the other columns of the table.
    """

    kind: str
    mapping: str
    value_kind: str
    n_draws: int
    source_mean: float
    source_variance: float
    mean: float
    variance: float
    stream: int
    alignment_identity: str
    draws_digest: str
    claim_id: str
    source_claim_id: str
    contract_identity: str
    transported: bool
    reading: str
    calibration: str
    units: str | None
    assumptions: tuple[Assumption, ...]
    unresolved: tuple[str, ...]
    unavailable: tuple[str, ...]
    evidence: tuple[str, ...]
    domains: Mapping[str, str]
    provenance: tuple[Mapping[str, Any], ...]
    identity: str | None
    _handle: Any = field(repr=False, compare=False)
    quantity: Any = None
    mapping_uncertainty: MappingUncertainty | None = None
    source_dependence: SourceDependence | None = None
    estimand_bridge: EstimandBridge | None = None

    @property
    def answer(self) -> Answer:
        return Answer("distribution", detail=self.value_kind)

    @property
    def identities(self) -> dict[str, str]:
        return {
            "claim_id": self.claim_id,
            "source_claim_id": self.source_claim_id,
            "contract_identity": self.contract_identity,
            "draws_digest": self.draws_digest,
            "alignment_identity": self.alignment_identity,
        }

    def inspect(self) -> Any:
        from .results._report import SlotModel

        return ScientificResult.inspect(self).model_copy(
            update={
                "uncertainty": SlotModel(
                    available=True,
                    reason="no_calibrated_interval",
                    summary=self.reading,
                    payload={
                        "mean": self.mean,
                        "variance": self.variance,
                        "calibration": self.calibration,
                        "n_draws": self.n_draws,
                    },
                ),
                "assumptions": SlotModel(
                    available=True,
                    summary="declared bridge assumptions",
                    payload={"count": len(self.assumptions), "unresolved": self.unresolved},
                ),
            }
        )

    def distribution(self) -> Any:
        """The bridged draw table as a :class:`~antecedent.joint_distribution.JointDistributionArtifact`."""
        from .joint_distribution import JointDistributionArtifact

        return JointDistributionArtifact._from_native(self._handle.distribution())

    def source_distribution(self) -> Any:
        """The source draw table, unchanged."""
        from .joint_distribution import JointDistributionArtifact

        return JointDistributionArtifact._from_native(self._handle.source_distribution())

    def require_value_kind(self, requested: str) -> None:
        """Raise the refusal when a consumer requires another kind of object than these draws.

        Raises:
            ScientificBridgeRefusal: ``bridge_substitution_refused`` /
                ``scientific_bridge.value_kind_substitution``.
        """
        _choice(requested, "requested", VALUE_KINDS)
        raise_refusal(self._handle.require_value_kind(requested), ScientificBridgeRefusal)

    @outcome_operation
    def interval(self, level: float | None = None) -> BridgedClaimInterval:
        """The interval of the bridged value at ``level``, through the measured-coverage gate.

        The standard deviation of the source draws is carried through the mapping together with
        the mapping's own estimation uncertainty, at the source's mean: the construction the
        bridged-mean coverage records measured. It is published only where such a record exists.
        With no ``level`` the request is unscoped and always raises.

        Raises:
            ScientificBridgeRefusal: ``cell_not_licensed`` /
                ``scientific_bridge.interval_not_licensed`` for an unscoped request, and
                ``scientific_bridge.interval_scope_outside_measured`` for a map with uncertain
                coefficients, a declared source-mapping covariance, or a design no record measured.
        """
        if level is None:
            raise_refusal(self._handle.interval(), ScientificBridgeRefusal)
            raise AssertionError("an unscoped interval request always refuses")  # pragma: no cover
        payload = _unwrap(self._handle.bridged_interval(_number(level, "level")))
        return BridgedClaimInterval(**json.loads(payload))

    def chain_stage(self, stage_id: str = "bridge") -> ChainStageParts:
        """The bridge as a measurement-mapping stage of a propagation chain.

        The stage reads the source draws and the draws of the mapping (columns of one aligned draw
        table) and reproduces the bridged column row by row, so a chain of bridge, transport and
        utility propagates the source, the mapping and every later stage jointly:

        * an affine map or trial-level line applies ``source * slope + intercept`` with the slope
          and the intercept as the stage's inputs;
        * a predictive law through a trial-level line adds the new trial's residual, an input of
          its own that declares its own disturbance, so the residual variance is added once;
        * a measurement correction applies ``(q + specificity - 1) / (sensitivity + specificity - 1)``
          over the source and the two drawn rates. The division refuses a Youden index that reaches
          zero (or falls below the model's smallest licensed value) on any row, and a corrected rate
          outside ``[0, 1]`` is refused, never clipped.

        Raises:
            ScientificBridgeRefusal: the mapping has no draws propagation to express.
        """
        from . import propagation_chain as chain

        payload = json.loads(_unwrap(self._handle.chain_stage_json(_name(stage_id, "stage_id"))))
        sources = tuple(
            chain.ChainSource(
                id=wire["id"],
                representation=wire["representation"],
                values=tuple(chain._float(bits) for bits in wire["values"]),
                meaning=wire["meaning"],
                alignment=wire["alignment"],
                complete_disclosure=bool(wire["complete_disclosure"]),
                factors=tuple((f["kind"], f["id"]) for f in wire["factors"]),
            )
            for wire in payload["sources"]
        )
        wire = payload["stage"]
        stage = chain.ChainStage(
            id=wire["id"],
            kind=wire["kind"],
            inputs=tuple(wire["inputs"]),
            op=wire["op"],
            own_factors=tuple((f["kind"], f["id"]) for f in wire["own_factors"]),
        )
        return ChainStageParts(sources, stage)

    @outcome_operation
    def freshness(
        self, *, source_withdrawn: bool = False, source_superseded: bool = False
    ) -> Freshness:
        """Applicability given the source's standing; history is preserved either way."""
        wire = json.loads(
            self._handle.freshness_json(bool(source_withdrawn), bool(source_superseded))
        )
        return Freshness(wire["historically_valid"], wire["currently_applicable"])

    @artifact_export
    def export(
        self, artifact_id: str = "scientific_bridge"
    ) -> ExportedArtifact[ConsumedBridgeDistribution]:
        """Write the extended ``scientific_bridge_v1`` artifact: the contract, the source draw
        table, the seed and the summary of the mapped table.

        Raises:
            ScientificBridgeRefusal: the bridge does not seal.
        """
        done = _unwrap(self._handle.export(_name(artifact_id, "artifact_id")))
        data, identity = done
        return artifact_from_receipt(self, BridgeExport(bytes(data), identity))


def _distribution(summary: Mapping[str, Any], handle: Any) -> BridgedDistribution:
    from .joint_distribution import ScientificQuantity

    contract = summary["contract"]
    return BridgedDistribution(
        kind=summary["kind"],
        mapping=summary["mapping"],
        value_kind=summary["value_kind"],
        n_draws=int(summary["n_draws"]),
        source_mean=float(summary["source_mean"]),
        source_variance=float(summary["source_variance"]),
        mean=float(summary["mean"]),
        variance=float(summary["variance"]),
        stream=int(summary["stream"]),
        alignment_identity=summary["alignment_identity"],
        draws_digest=summary["draws_digest"],
        claim_id=summary["claim_id"],
        source_claim_id=summary["source_claim_id"],
        contract_identity=summary["contract_identity"],
        transported=bool(summary["transported"]),
        reading=summary["reading"],
        calibration=summary["calibration"],
        units=summary["units"],
        assumptions=tuple(
            Assumption(a["id"], a["kind"], a["statement"], a["evidence"])
            for a in contract["assumptions"]
        ),
        unresolved=tuple(summary["unresolved"]),
        unavailable=tuple(summary["unavailable"]),
        evidence=tuple(summary["evidence"]),
        domains=dict(summary["domains"]),
        provenance=tuple(summary["provenance"]),
        identity=summary.get("identity"),
        _handle=handle,
        quantity=ScientificQuantity.from_wire(contract["target"]["quantity"]),
        **_extension(contract),
    )


@dataclass(frozen=True, slots=True, eq=False)
class PreparedDistributionBridge:
    """A bridge of draws whose contract was checked once, with its licensing verdict kept.

    :meth:`derive` and :meth:`with_contract` return new values and leave this one untouched.
    """

    contract: BridgeContract
    contract_identity: str
    transported: bool
    _handle: Any = field(repr=False, compare=False)

    @outcome_operation
    def derive(
        self, draws: Any, *, seed: int, source_kind: str | None = None
    ) -> BridgedDistribution:
        """Bridge the draws of ``draws``, a :class:`~antecedent.joint_distribution.JointDistributionArtifact`.

        ``draws`` is an aligned table with a column about the contract's source quantity, of a
        meaning that is a posterior or a predictive law (a bootstrap, a sampling law or an empirical
        law is neither). ``source_kind`` is the kind of the claim the draws come from, when they
        come from one: an ``associational`` claim is never a bridge source. ``seed`` selects the
        random stream of the mapping's coefficients, so the result is a function of the contract,
        the table and the seed.

        Raises:
            ScientificBridgeRefusal: draws of another quantity or meaning, independent marginals,
                an associational source, a draw outside the range the mapping is licensed for, a
                dependence no joint law has, or a mapped draw that is not finite.
        """
        from .joint_distribution import JointDistributionArtifact
        from .outcomes import require_success

        draws = require_success(draws)
        if not isinstance(draws, JointDistributionArtifact):
            raise CausalTypeError("draws must be a JointDistributionArtifact")
        if source_kind is not None:
            _choice(source_kind, "source_kind", _CLAIM_KINDS)
        handle = _unwrap(self._handle.derive(draws._native, _seed(seed), source_kind))
        return _distribution(json.loads(handle.summary_json()), handle)

    def with_contract(self, contract: BridgeContract) -> PreparedDistributionBridge:
        """A prepared bridge for a replacement contract, re-checked in full.

        Raises:
            ScientificBridgeRefusal: the first defect of the replacement; ``self`` is unchanged.
        """
        if not isinstance(contract, BridgeContract):
            raise CausalTypeError("contract must be a BridgeContract")
        handle = _unwrap(self._handle.with_contract(_dumps(contract.to_wire())))
        return _prepared_distribution(contract, handle)


def _prepared_distribution(contract: BridgeContract, handle: Any) -> PreparedDistributionBridge:
    return PreparedDistributionBridge(
        contract=contract,
        contract_identity=handle.contract_identity(),
        transported=bool(handle.transported()),
        _handle=handle,
    )


def prepare_distribution_bridge(contract: BridgeContract) -> PreparedDistributionBridge:
    """Check a contract whose source is the draws of a posterior or a predictive law.

    ``source_value`` and ``target_value`` are both ``posterior`` or both ``predictive_law``. An
    affine map with uncertain coefficients declares its ``mapping_uncertainty`` and its
    ``source_dependence``; a trial-level line and a measurement correction take their uncertainty
    from their own evidence and declare only the dependence.

    Raises:
        ScientificBridgeRefusal: the first defect, as for :func:`prepare_scientific_bridge`, a
            value kind that is neither a posterior nor a predictive law, an uncertain map with no
            declared dependence, or a mapping that declares its uncertainty a second time.
    """
    if not isinstance(contract, BridgeContract):
        raise CausalTypeError("contract must be a BridgeContract")
    handle = _unwrap(_native.scientific_bridge_distribution_prepare(_dumps(contract.to_wire())))
    return _prepared_distribution(contract, handle)


@dataclass(repr=False, frozen=True, slots=True, eq=False)
class ConsumedBridgeDistribution(ScientificResult):
    """A consumed bridge-of-draws artifact: its stored draws mapped again, bit for bit."""

    claim: BridgedDistribution
    identity: str
    _handle: Any = field(repr=False, compare=False)

    @property
    def answer(self) -> Answer:
        return self.claim.answer

    def inspect(self) -> Any:
        return self.claim.inspect().model_copy(update={"stage": "consumed"})

    def interval(self, level: float | None = None) -> Outcome[BridgedClaimInterval]:
        """As :meth:`BridgedDistribution.interval`."""
        return self.claim.interval(level)


def _consume_distribution(
    artifact: bytes, expected_identity: str | None
) -> ConsumedBridgeDistribution:
    artifact = container_bytes(artifact)
    if expected_identity is not None:
        _name(expected_identity, "expected_identity")
    handle = _unwrap(_native.scientific_bridge_distribution_consume(artifact, expected_identity))
    summary = json.loads(handle.summary_json())
    return ConsumedBridgeDistribution(
        claim=_distribution(summary, handle), identity=summary["identity"], _handle=handle
    )
