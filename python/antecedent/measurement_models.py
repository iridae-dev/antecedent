"""Measurement models: corrected functionals of a latent construct measured with error.

A recorded column is not the scientific construct it is recorded for. A
:class:`MeasurementModel` states how one indicator relates to one construct, for which
population, regime and arm, under which error structure, in which representation version, and
where its numbers came from. Everything downstream is conditional on that statement::

    from antecedent import measurement_models as mm

    model = mm.MeasurementModel(
        indicator=mm.MeasurementVariable("schema:flag", "noisy_misclassified", "binary"),
        construct=mm.MeasurementVariable("construct:disease", "latent_scientific", "binary"),
        parameters=mm.BinaryParameters(
            rates=(mm.BinaryRates(sensitivity=0.9, specificity=0.8,
                                  sensitivity_variance=1e-4, specificity_variance=4e-4),),
            min_youden=0.1,
        ),
        population_id="cohort",
        regime_id="observational",
        evidence=mm.ParameterEvidence.validation_sample("val-2024", units=500),
    )
    analysis = mm.AnalysisBinding(
        construct_id="construct:disease", population_id="cohort", regime_id="observational"
    )
    prepared = mm.prepare_measurement_model(
        model, analysis, mm.BinaryPrevalenceData(events=410, total=1000)
    )
    estimate = prepared.estimate()
    estimate.point              # 0.3: the algebra of p* = Se*pi + (1 - Sp)*(1 - pi) inverted
    estimate.naive_point        # 0.41: the uncorrected value of the same functional
    artifact = estimate.export("prevalence")
    again = mm.consume_measurement_model_artifact(artifact, prepared.model_identity)

What is licensed, and what is not:

* **Prepare** refuses before any number is formed: a proxy, a learned representation or a derived
  column offered as the construct, a population, regime, arm or representation the model was not
  established for (a representation change needs an explicit :class:`RepresentationMapping`, which
  carries values and never intervention meaning unless it declares the evidence for it), error that
  differs while the analysis assumes it does not, and an operation the route does not license.
* **Estimate** forms one functional of the corrected quantity. The standard error is an
  *uncalibrated* delta-method estimate that includes the uncertainty of the estimated measurement
  parameters where the record says it was propagated. No interval is produced:
  :meth:`MeasurementEstimate.interval` raises ``cell_not_licensed`` until the whole
  measurement-plus-estimation method has a measured coverage record.
* **Refresh** keeps the model and recomputes the numbers on new outcome-only data of the same
  question; it never changes validation parameters. :meth:`PreparedMeasurementModel
  .with_validation_parameters` is the only way the numbers a model was established with change, and
  it rebuilds the model identity. A refresh that refuses raises and leaves the prepared object it
  was called on exactly as it was.
* **Consume** recomputes the point and standard error from the stored sufficient inputs against a
  model identity the consumer retained independently of the bytes.

Continuous classical error beyond the linear moment correction goes through
:meth:`PreparedMeasurementModel.estimate_continuous` with a :class:`ContinuousErrorData`: regression
calibration or SIMEX (Cook and Stefanski 1994) under a linear or logistic outcome model, with the
error in the treatment, a covariate or the outcome. Logistic calibration and SIMEX report a
bootstrap standard error over the whole procedure; the result is uncalibrated and has no interval.

A posterior from an external provider is trusted only as far as it is attested or verified, and
never as native: :class:`ProviderObject` states an attestation or carries a verification receipt the
library issued, and a claim of native trust is refused by name.

Every correction, identity and refusal is computed in Rust; this module builds the declaration and
presents the result. Refusals are :class:`MeasurementModelRefusal` (a
:class:`~antecedent.errors.CausalUnsupportedError`) with a registered ``reason_code`` and a
``measurement_model.*`` ``detail``.

A model established for one population is carried to another only under a declared invariance
premise (:func:`transport_measurement_model`)::

    result = mm.transport_measurement_model(
        [source_model],
        analysis_for_clinic_b,
        mm.BinaryPrevalenceData(events=520, total=1000),
        structure=mm.InvarianceDeclaration(
            "clinic_a", "clinic_b", "analyst", ("sensitivity", "specificity")
        ),
    )
    result.estimate.point, result.receipt.carried, result.identity

The premise must name every error parameter the route uses; it is declared, never tested; its own
uncertainty is not propagated; and the corrected value is never presented as native to the target.
A missing premise, an undeclared parameter, a declared population-differential error without a
model for the target, and a route with nothing to carry each refuse with the unmet premise.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import _native
from ._artifact_protocol import ExportedArtifact
from ._mean_binding import bind_mean_reading
from ._refusal import StructuredRefusal, raise_refusal
from ._scientific import ScientificResult
from ._validate import count as _shared_count
from .errors import CausalTypeError, CausalValueError

__all__ = [
    "AnalysisBinding",
    "DifferentialErrorDeclaration",
    "InvarianceDeclaration",
    "MeasurementTransportReceipt",
    "TransportedMeasurementEstimate",
    "BinaryParameters",
    "BinaryPrevalenceData",
    "BinaryRates",
    "CategoricalData",
    "CategoricalParameters",
    "CategoricalTarget",
    "ClassicalErrorParameters",
    "ClassicalLinearData",
    "ConsumedMeasurement",
    "ContinuousErrorData",
    "ContinuousMeasurementEstimate",
    "LatentPosteriorData",
    "MeasurementEstimate",
    "MeasurementIdentities",
    "MeasurementModel",
    "MeasurementModelRefusal",
    "MeasurementRefresh",
    "MeasurementVariable",
    "ParameterEvidence",
    "PosteriorParameters",
    "PosteriorReference",
    "PreparedMeasurementModel",
    "ProviderObject",
    "RepeatedLinearData",
    "RepeatedParameters",
    "RepresentationMapping",
    "RepresentationVersion",
    "RiskDifferenceData",
    "VarianceComponent",
    "consume_measurement_model_artifact",
    "model_identities",
    "prepare_measurement_model",
    "transport_measurement_model",
    "verify_measurement_transport",
    "PopulationValidationCounts",
    "PooledDifferentialCorrection",
    "transport_pooled_differential",
]

#: Downstream operations a route may license or refuse.
OPERATIONS: tuple[str, ...] = (
    "linear_adjusted_effect",
    "linear_functional_of_coefficients",
    "latent_prevalence",
    "risk_difference",
    "category_outcome_means",
    "condition_on_construct",
    "intervene_on_representation",
    "transport_to_other_population",
    "nonlinear_outcome_model",
    "interval",
)
_KINDS = (
    "observed",
    "latent_scientific",
    "noisy_misclassified",
    "probabilistic_feature_assignment",
    "learned_vector_representation",
    "derived",
)
_STRUCTURES = (
    "non_differential",
    "differential_arm",
    "differential_population",
    "differential_time",
)
_ROUTES = (
    "classical_continuous_error",
    "binary_misclassification",
    "repeated_measurements",
    "probabilistic_categorical",
    "external_latent_posterior",
)
_DEFAULT_ARTIFACT_ID = "measurement_model"


class MeasurementModelRefusal(StructuredRefusal):
    """A typed refusal of the measurement-model route.

    A :class:`~antecedent.errors.CausalUnsupportedError` with a registered ``reason_code`` and a
    ``measurement_model.*`` ``detail``; ``remedy`` names the declaration or evidence that would
    license the request when one is known.
    """


# --------------------------------------------------------------------------- validation


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalTypeError(f"{what} must be a non-empty string")
    return value


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise CausalTypeError(f"{what} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise CausalValueError(f"{what} must be finite")
    return number


def _count(value: object, what: str, **kwargs: Any) -> int:
    kwargs.setdefault("minimum", 0)
    kwargs.setdefault("maximum", 2**64 - 1)
    return _shared_count(value, what, **kwargs)


def _vector(values: object, what: str) -> list[float]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CausalTypeError(f"{what} must be a one-dimensional numeric array") from error
    if array.ndim != 1:
        raise CausalValueError(f"{what} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise CausalValueError(f"{what} must be finite")
    return [float(v) for v in array]


def _columns(values: object, n: int, what: str) -> tuple[list[float], int]:
    """Column-major flattening of an ``n x k`` covariate matrix, and ``k``."""
    if values is None:
        return [], 0
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CausalTypeError(f"{what} must be a numeric matrix") from error
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise CausalValueError(f"{what} must be a matrix with one row per unit")
    if array.shape[0] != n:
        raise CausalValueError(f"{what} has {array.shape[0]} rows for {n} units")
    if not np.all(np.isfinite(array)):
        raise CausalValueError(f"{what} must be finite")
    return [float(v) for v in array.T.reshape(-1)], int(array.shape[1])


def _choice(value: str, what: str, allowed: Sequence[str]) -> str:
    if value not in allowed:
        raise CausalValueError(f"{what} must be one of {', '.join(allowed)}; got {value!r}")
    return value


def _dumps(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def _unwrap(pair: tuple[Any, str | None]) -> Any:
    value, refusal = pair
    raise_refusal(refusal, MeasurementModelRefusal)
    return value


# --------------------------------------------------------------------------- declarations


@dataclass(frozen=True, slots=True)
class RepresentationVersion:
    """Coordinate system and version of an indicator's representation."""

    coordinate_system_id: str = "scalar"
    version: int = 1

    def __post_init__(self) -> None:
        _name(self.coordinate_system_id, "coordinate_system_id")
        _count(self.version, "version")

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "coordinate_system_id": self.coordinate_system_id,
            "version": _count(
                self.version,
                "version",
                minimum=0,
                maximum=2**64 - 1,
                reason_code="invalid_argument",
            ),
        }


@dataclass(frozen=True, slots=True)
class RepresentationMapping:
    """An explicit mapping between two representation versions.

    A mapping carries *values*. It carries intervention meaning only when
    ``intervention_evidence_id`` names the evidence that establishes the correspondence.
    """

    source: RepresentationVersion
    target: RepresentationVersion
    mapping_id: str
    intervention_evidence_id: str | None = None

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        if self.intervention_evidence_id is None:
            semantics: dict[str, Any] = {"kind": "coordinate_relabeling"}
        else:
            semantics = {
                "kind": "intervention_preserving",
                "evidence_id": self.intervention_evidence_id,
            }
        return {
            "from": self.source.to_wire(),
            "to": self.target.to_wire(),
            "mapping_id": self.mapping_id,
            "semantics": semantics,
        }


@dataclass(frozen=True, slots=True)
class MeasurementVariable:
    """An indicator or a construct: stable identity, kind and units.

    ``kind`` is one of ``observed``, ``latent_scientific``, ``noisy_misclassified``,
    ``probabilistic_feature_assignment``, ``learned_vector_representation`` and ``derived``. The
    construct of a model is ``latent_scientific``; offering any other kind as the construct is
    refused as ``proxy_offered_as_construct``.
    """

    id: str
    kind: str
    units: str

    def __post_init__(self) -> None:
        _choice(self.kind, "kind", _KINDS)

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {"id": self.id, "kind": self.kind, "units": self.units}


@dataclass(frozen=True, slots=True)
class ClassicalErrorParameters:
    """Classical additive continuous error: variance of one measurement and its uncertainty."""

    error_variance: float
    error_variance_standard_error: float = 0.0
    min_reliability: float = 0.1

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "kind": "classical_error",
            "error_variance": _number(self.error_variance, "error_variance"),
            "error_variance_standard_error": _number(
                self.error_variance_standard_error, "error_variance_standard_error"
            ),
            "min_reliability": _number(self.min_reliability, "min_reliability"),
        }


@dataclass(frozen=True, slots=True)
class BinaryRates:
    """Sensitivity and specificity of a binary indicator, for one arm or shared by all."""

    sensitivity: float
    specificity: float
    sensitivity_variance: float = 0.0
    specificity_variance: float = 0.0
    arm_id: str | None = None

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "arm_id": self.arm_id,
            "sensitivity": _number(self.sensitivity, "sensitivity"),
            "specificity": _number(self.specificity, "specificity"),
            "sensitivity_variance": _number(self.sensitivity_variance, "sensitivity_variance"),
            "specificity_variance": _number(self.specificity_variance, "specificity_variance"),
        }


@dataclass(frozen=True, slots=True)
class BinaryParameters:
    """Binary misclassification: one shared entry, or one per arm when differential by arm."""

    rates: tuple[BinaryRates, ...]
    min_youden: float = 0.1

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "kind": "binary",
            "rates": [r.to_wire() for r in self.rates],
            "min_youden": _number(self.min_youden, "min_youden"),
        }


@dataclass(frozen=True, slots=True)
class RepeatedParameters:
    """Replicated measurements: the error variance is estimated from the analysis sample."""

    min_replicated_units: int
    min_reliability: float = 0.1

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "kind": "repeated",
            "min_replicated_units": _count(self.min_replicated_units, "min_replicated_units"),
            "min_reliability": _number(self.min_reliability, "min_reliability"),
        }


@dataclass(frozen=True, slots=True)
class CategoricalParameters:
    """Assigned categorical labels with a declared column-stochastic confusion matrix.

    ``confusion`` is row-major ``[observed][true]``; ``column_counts`` are the validation units per
    true category (``None`` when the matrix is declared exact).
    """

    levels: tuple[str, ...]
    confusion: tuple[float, ...]
    column_counts: tuple[int, ...] | None = None
    max_condition: float = 50.0

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "kind": "categorical",
            "levels": [_name(level, "level") for level in self.levels],
            "confusion": [_number(v, "confusion entry") for v in self.confusion],
            "column_counts": (
                None
                if self.column_counts is None
                else [_count(c, "column count") for c in self.column_counts]
            ),
            "max_condition": _number(self.max_condition, "max_condition"),
        }


@dataclass(frozen=True, slots=True)
class PosteriorReference:
    """Which external provider object supplies the latent posterior."""

    provider_id: str
    object_id: str
    version_id: str
    snapshot_id: str
    request_id: str

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "provider_id": self.provider_id,
            "object_id": self.object_id,
            "version_id": self.version_id,
            "snapshot_id": self.snapshot_id,
            "request_id": self.request_id,
        }


@dataclass(frozen=True, slots=True)
class PosteriorParameters:
    """An external latent-variable posterior consumed as joint draws."""

    reference: PosteriorReference
    conditioned_on_outcome: bool
    min_draws: int

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "kind": "external_posterior",
            "reference": self.reference.to_wire(),
            "conditioned_on_outcome": bool(self.conditioned_on_outcome),
            "min_draws": _count(self.min_draws, "min_draws"),
        }


MeasurementParameters = (
    ClassicalErrorParameters
    | BinaryParameters
    | RepeatedParameters
    | CategoricalParameters
    | PosteriorParameters
)

_ROUTE_OF_PARAMETERS: dict[type, str] = {
    ClassicalErrorParameters: "classical_continuous_error",
    BinaryParameters: "binary_misclassification",
    RepeatedParameters: "repeated_measurements",
    CategoricalParameters: "probabilistic_categorical",
    PosteriorParameters: "external_latent_posterior",
}


@dataclass(frozen=True, slots=True)
class ParameterEvidence:
    """Where the parameters of a model came from.

    Build one with :meth:`declared`, :meth:`validation_sample`, :meth:`replicate_sample` or
    :meth:`external_provider`. The evidence identity is part of the model identity.
    """

    kind: str
    declarer: str | None = None
    sample_id: str | None = None
    units: int | None = None
    provider_id: str | None = None

    @classmethod
    def declared(cls, declarer: str) -> ParameterEvidence:
        """Declared by a named party without a sample behind it here."""
        return cls(kind="declared", declarer=_name(declarer, "declarer"))

    @classmethod
    def validation_sample(cls, sample_id: str, units: int) -> ParameterEvidence:
        """Estimated from a validation sample against a reference measurement."""
        return cls(
            kind="validation_sample",
            sample_id=_name(sample_id, "sample_id"),
            units=_count(units, "units"),
        )

    @classmethod
    def replicate_sample(cls, sample_id: str) -> ParameterEvidence:
        """Estimated from the analysis sample's own replicates."""
        return cls(kind="replicate_sample", sample_id=_name(sample_id, "sample_id"))

    @classmethod
    def external_provider(cls, provider_id: str) -> ParameterEvidence:
        """Supplied by an external provider; carries no native trust."""
        return cls(kind="external_provider", provider_id=_name(provider_id, "provider_id"))

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        if self.kind == "declared":
            return {"kind": "declared", "declarer": self.declarer}
        if self.kind == "validation_sample":
            return {"kind": "validation_sample", "sample_id": self.sample_id, "units": self.units}
        if self.kind == "replicate_sample":
            return {"kind": "replicate_sample", "sample_id": self.sample_id}
        if self.kind == "external_provider":
            return {"kind": "external_provider", "provider_id": self.provider_id}
        raise CausalValueError(f"unknown parameter evidence kind {self.kind!r}")


@dataclass(frozen=True, slots=True)
class MeasurementModel:
    """A typed relationship between one indicator and one construct.

    ``route`` is derived from ``parameters`` and is overridden only to state a route the
    parameters do not belong to (which the checks then refuse). ``structure`` is one of
    ``non_differential``, ``differential_arm``, ``differential_population`` and
    ``differential_time``; only the first two have a bounded route. ``model_id`` is a label and is
    never part of the semantic identity.
    """

    indicator: MeasurementVariable
    construct: MeasurementVariable
    parameters: MeasurementParameters
    population_id: str
    regime_id: str
    evidence: ParameterEvidence
    arm_id: str | None = None
    structure: str = "non_differential"
    representation: RepresentationVersion = field(default_factory=RepresentationVersion)
    model_id: str = "measurement-model"
    route: str | None = None

    def __post_init__(self) -> None:
        _choice(self.structure, "structure", _STRUCTURES)
        if self.route is not None:
            _choice(self.route, "route", _ROUTES)

    @property
    def resolved_route(self) -> str:
        """The route of the declared parameters (or the explicitly stated one)."""
        if self.route is not None:
            return self.route
        try:
            return _ROUTE_OF_PARAMETERS[type(self.parameters)]
        except KeyError as error:
            raise CausalTypeError(
                "parameters must be one of ClassicalErrorParameters, BinaryParameters, "
                "RepeatedParameters, CategoricalParameters or PosteriorParameters"
            ) from error

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "model_id": self.model_id,
            "route": self.resolved_route,
            "indicator": self.indicator.to_wire(),
            "construct": self.construct.to_wire(),
            "binding": {
                "population_id": self.population_id,
                "regime_id": self.regime_id,
                "arm": self.arm_id,
            },
            "structure": self.structure,
            "parameters": self.parameters.to_wire(),
            "representation": self.representation.to_wire(),
            "evidence": self.evidence.to_wire(),
        }


def _parameters_from_wire(wire: dict[str, Any]) -> MeasurementParameters:
    kind = wire["kind"]
    if kind == "classical_error":
        return ClassicalErrorParameters(
            error_variance=wire["error_variance"],
            error_variance_standard_error=wire["error_variance_standard_error"],
            min_reliability=wire["min_reliability"],
        )
    if kind == "binary":
        return BinaryParameters(
            rates=tuple(
                BinaryRates(
                    sensitivity=r["sensitivity"],
                    specificity=r["specificity"],
                    sensitivity_variance=r["sensitivity_variance"],
                    specificity_variance=r["specificity_variance"],
                    arm_id=r["arm_id"],
                )
                for r in wire["rates"]
            ),
            min_youden=wire["min_youden"],
        )
    if kind == "repeated":
        return RepeatedParameters(
            min_replicated_units=wire["min_replicated_units"],
            min_reliability=wire["min_reliability"],
        )
    if kind == "categorical":
        counts = wire["column_counts"]
        return CategoricalParameters(
            levels=tuple(wire["levels"]),
            confusion=tuple(wire["confusion"]),
            column_counts=None if counts is None else tuple(counts),
            max_condition=wire["max_condition"],
        )
    if kind == "external_posterior":
        return PosteriorParameters(
            reference=PosteriorReference(**wire["reference"]),
            conditioned_on_outcome=wire["conditioned_on_outcome"],
            min_draws=wire["min_draws"],
        )
    raise CausalValueError(f"unknown measurement parameters {kind!r}")


def _evidence_from_wire(wire: dict[str, Any]) -> ParameterEvidence:
    kind = wire["kind"]
    if kind == "declared":
        return ParameterEvidence.declared(wire["declarer"])
    if kind == "validation_sample":
        return ParameterEvidence.validation_sample(wire["sample_id"], wire["units"])
    if kind == "replicate_sample":
        return ParameterEvidence.replicate_sample(wire["sample_id"])
    if kind == "external_provider":
        return ParameterEvidence.external_provider(wire["provider_id"])
    raise CausalValueError(f"unknown parameter evidence {kind!r}")


def _model_from_wire(wire: dict[str, Any]) -> MeasurementModel:
    """The typed model of a declaration mapping the native layer wrote."""
    binding = wire["binding"]
    representation = wire["representation"]
    return MeasurementModel(
        indicator=MeasurementVariable(**wire["indicator"]),
        construct=MeasurementVariable(**wire["construct"]),
        parameters=_parameters_from_wire(wire["parameters"]),
        population_id=binding["population_id"],
        regime_id=binding["regime_id"],
        evidence=_evidence_from_wire(wire["evidence"]),
        arm_id=binding["arm"],
        structure=wire["structure"],
        representation=RepresentationVersion(
            representation["coordinate_system_id"], representation["version"]
        ),
        model_id=wire["model_id"],
        route=wire["route"],
    )


@dataclass(frozen=True, slots=True)
class AnalysisBinding:
    """What the analysis says about the data and the question it asks of a model.

    The construct, population, regime, arm and representation are stated once; the model's
    checks run against exactly these. ``mapping`` is supplied when ``representation`` differs from
    the model's. ``assumed_structure`` is the error structure the analysis assumes.
    """

    construct_id: str
    population_id: str
    regime_id: str
    arm_id: str | None = None
    representation: RepresentationVersion = field(default_factory=RepresentationVersion)
    mapping: RepresentationMapping | None = None
    assumed_structure: str = "non_differential"

    def __post_init__(self) -> None:
        _choice(self.assumed_structure, "assumed_structure", _STRUCTURES)

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "construct_id": self.construct_id,
            "population_id": self.population_id,
            "regime_id": self.regime_id,
            "arm_id": self.arm_id,
            "representation": self.representation.to_wire(),
            "representation_mapping": None if self.mapping is None else self.mapping.to_wire(),
            "assumed_structure": self.assumed_structure,
        }


# --------------------------------------------------------------------------- data


@dataclass(frozen=True, slots=True)
class ClassicalLinearData:
    """Outcome, error-prone regressor and error-free covariates of a linear outcome model.

    ``regressor`` is the row mean of replicates when ``replicate_counts`` says how many were
    averaged into each value. ``covariates`` is ``n x k`` (or a vector for one covariate).
    """

    outcome: Any
    regressor: Any
    covariates: Any = None
    replicate_counts: Any = None

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        outcome = _vector(self.outcome, "outcome")
        covariates, k = _columns(self.covariates, len(outcome), "covariates")
        counts = None
        if self.replicate_counts is not None:
            counts = [_count(c, "replicate count") for c in np.asarray(self.replicate_counts)]
        return {
            "kind": "classical_linear",
            "outcome": outcome,
            "regressor": _vector(self.regressor, "regressor"),
            "covariates": covariates,
            "covariate_count": k,
            "replicate_counts": counts,
        }


@dataclass(frozen=True, slots=True)
class RepeatedLinearData:
    """Repeated measurements: every unit's replicates back to back, and their counts."""

    outcome: Any
    replicate_values: Any
    replicate_counts: Any
    covariates: Any = None

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        outcome = _vector(self.outcome, "outcome")
        covariates, k = _columns(self.covariates, len(outcome), "covariates")
        return {
            "kind": "repeated_linear",
            "outcome": outcome,
            "replicate_values": _vector(self.replicate_values, "replicate_values"),
            "replicate_counts": [
                _count(c, "replicate count") for c in np.asarray(self.replicate_counts)
            ],
            "covariates": covariates,
            "covariate_count": k,
        }


@dataclass(frozen=True, slots=True)
class BinaryPrevalenceData:
    """Observed positives and units observed, for a corrected prevalence."""

    events: int
    total: int

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "kind": "binary_prevalence",
            "events": _count(self.events, "events"),
            "total": _count(self.total, "total"),
        }


@dataclass(frozen=True, slots=True)
class RiskDifferenceData:
    """Observed events and totals of two arms, for a corrected risk difference."""

    exposed_arm_id: str
    unexposed_arm_id: str
    exposed_events: int
    exposed_total: int
    unexposed_events: int
    unexposed_total: int

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "kind": "binary_risk_difference",
            "exposed_arm_id": _name(self.exposed_arm_id, "exposed_arm_id"),
            "unexposed_arm_id": _name(self.unexposed_arm_id, "unexposed_arm_id"),
            "exposed_events": _count(self.exposed_events, "exposed_events"),
            "exposed_total": _count(self.exposed_total, "exposed_total"),
            "unexposed_events": _count(self.unexposed_events, "unexposed_events"),
            "unexposed_total": _count(self.unexposed_total, "unexposed_total"),
        }


@dataclass(frozen=True, slots=True)
class CategoricalTarget:
    """A scalar functional of corrected categories, named by level.

    Build one with :meth:`proportion`, :meth:`outcome_mean` or :meth:`outcome_contrast`.
    """

    kind: str
    level: str | None = None
    first: str | None = None
    second: str | None = None

    @classmethod
    def proportion(cls, level: str) -> CategoricalTarget:
        """The proportion of one true category."""
        return cls(kind="proportion", level=_name(level, "level"))

    @classmethod
    def outcome_mean(cls, level: str) -> CategoricalTarget:
        """The mean outcome among units in one true category."""
        return cls(kind="outcome_mean", level=_name(level, "level"))

    @classmethod
    def outcome_contrast(cls, first: str, second: str) -> CategoricalTarget:
        """The difference of outcome means ``first - second``."""
        return cls(
            kind="outcome_contrast", first=_name(first, "first"), second=_name(second, "second")
        )

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        if self.kind == "outcome_contrast":
            return {"kind": self.kind, "first": self.first, "second": self.second}
        return {"kind": self.kind, "level": self.level}


@dataclass(frozen=True, slots=True)
class CategoricalData:
    """Observed-label summaries in level order and the functional to report.

    ``classes`` has one ``(count, mean_outcome, outcome_variance)`` per level of the model.
    """

    classes: Sequence[tuple[int, float, float]]
    target: CategoricalTarget

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "kind": "categorical",
            "classes": [
                {
                    "count": _count(count, "class count"),
                    "mean": _number(mean, "class mean"),
                    "variance": _number(variance, "class variance"),
                }
                for count, mean, variance in self.classes
            ],
            "target": self.target.to_wire(),
        }


@dataclass(frozen=True, slots=True)
class LatentPosteriorData:
    """An external latent posterior as joint draws, one row per draw and one column per unit."""

    outcome: Any
    draws: Any
    covariates: Any = None

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        outcome = _vector(self.outcome, "outcome")
        try:
            draws = np.asarray(self.draws, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise CausalTypeError("draws must be a numeric matrix") from error
        if draws.ndim != 2:
            raise CausalValueError("draws must be a matrix with one row per draw")
        if draws.shape[1] != len(outcome):
            raise CausalValueError(f"draws has {draws.shape[1]} columns for {len(outcome)} units")
        if not np.all(np.isfinite(draws)):
            raise CausalValueError("draws must be finite")
        covariates, k = _columns(self.covariates, len(outcome), "covariates")
        return {
            "kind": "latent_posterior",
            "outcome": outcome,
            "draws": [float(v) for v in draws.reshape(-1)],
            "draw_count": _count(
                draws.shape[0],
                "draws.shape[0]",
                minimum=0,
                maximum=2**64 - 1,
                reason_code="invalid_argument",
            ),
            "covariates": covariates,
            "covariate_count": k,
        }


_LOCATIONS = ("treatment", "covariate", "outcome")
_LINKS = ("identity", "logit")
_METHODS = ("regression_calibration", "simex")
_EXTRAPOLANTS = ("quadratic", "linear")


@dataclass(frozen=True, slots=True)
class ContinuousErrorData:
    """Data and question of one correction for continuous classical measurement error.

    The design is ``(1, regressor, covariates)``. ``location`` says which column carries additive
    error: ``"treatment"`` (``regressor`` is the error-prone treatment), ``"covariate"``
    (``regressor`` is the error-prone covariate and the treatment is covariate column
    ``treatment_column``) or ``"outcome"`` (``outcome`` is error-prone and ``regressor`` is the
    error-free treatment). ``replicate_values`` (every row's measurements back to back) with
    ``replicate_counts`` makes the error-prone column the row means; the column may then be left
    ``None``.

    ``link`` is ``"identity"`` (linear) or ``"logit"`` (0/1 outcome). ``method`` is
    ``"regression_calibration"`` or ``"simex"`` (grid ``simex_lambdas`` plus ``0``,
    ``simex_replicates`` remeasurements per value, ``"quadratic"`` or ``"linear"``
    ``extrapolant``). Logistic calibration and SIMEX report a bootstrap standard error over
    ``bootstrap_replicates`` (at least 50) resamples of the whole procedure, keyed by ``seed``.
    """

    outcome: Any = None
    regressor: Any = None
    covariates: Any = None
    replicate_values: Any = None
    replicate_counts: Any = None
    location: str = "treatment"
    treatment_column: int | None = None
    link: str = "identity"
    method: str = "regression_calibration"
    simex_lambdas: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    simex_replicates: int = 50
    extrapolant: str = "quadratic"
    bootstrap_replicates: int = 200
    seed: int = 0

    def __post_init__(self) -> None:
        _choice(self.location, "location", _LOCATIONS)
        _choice(self.link, "link", _LINKS)
        _choice(self.method, "method", _METHODS)
        _choice(self.extrapolant, "extrapolant", _EXTRAPOLANTS)
        if (self.location == "covariate") != (self.treatment_column is not None):
            raise CausalValueError(
                "treatment_column is stated exactly when the error is in a covariate"
            )

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        outcome = [] if self.outcome is None else _vector(self.outcome, "outcome")
        regressor = [] if self.regressor is None else _vector(self.regressor, "regressor")
        rows = len(regressor) if self.location == "outcome" else len(outcome)
        covariates, k = _columns(self.covariates, rows, "covariates")
        replicate_values = None
        if self.replicate_values is not None:
            replicate_values = _vector(self.replicate_values, "replicate_values")
        replicate_counts = None
        if self.replicate_counts is not None:
            replicate_counts = [
                _count(c, "replicate count", maximum=2**32 - 1)
                for c in np.asarray(self.replicate_counts)
            ]
        method: dict[str, Any] = {"kind": self.method}
        if self.method == "simex":
            method.update(
                lambdas=[_number(v, "simex_lambdas") for v in self.simex_lambdas],
                replicates=_count(self.simex_replicates, "simex_replicates", maximum=2**32 - 1),
                extrapolant=self.extrapolant,
            )
        return {
            "outcome": outcome,
            "regressor": regressor,
            "covariates": covariates,
            "covariate_count": k,
            "replicate_values": replicate_values,
            "replicate_counts": replicate_counts,
            "location": self.location,
            "treatment_column": None
            if self.treatment_column is None
            else _count(self.treatment_column, "treatment_column"),
            "link": self.link,
            "method": method,
            "bootstrap_replicates": _count(
                self.bootstrap_replicates, "bootstrap_replicates", maximum=2**32 - 1
            ),
            "seed": _count(self.seed, "seed"),
        }


MeasurementData = (
    ClassicalLinearData
    | RepeatedLinearData
    | BinaryPrevalenceData
    | RiskDifferenceData
    | CategoricalData
    | LatentPosteriorData
)


@dataclass(frozen=True, slots=True)
class ProviderObject:
    """The exact external object a posterior route names, and what is claimed for it.

    ``trust`` is ``"externally_attested"`` (with an ``attestor``) or ``"exact_request_verified"``
    (with a ``receipt`` passed to :func:`prepare_measurement_model`). ``"native_licensed"`` is
    accepted only so the contract can refuse it: metadata never confers native trust.
    """

    provider_id: str
    object_id: str
    version_id: str
    snapshot_id: str
    request_id: str
    variable_id: str
    variable_name: str
    units: str
    population_id: str
    regime_id: str
    capabilities: tuple[str, ...] = ("sample", "mean")
    kind: str = "parameter"
    trust: str = "externally_attested"
    attestor: str | None = None
    role: str = "covariate"
    functional_id: str = "value"
    transform_id: str = "identity"
    horizon: int = 0

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "identity": {
                "provider_id": self.provider_id,
                "object_id": self.object_id,
                "version_id": self.version_id,
                "snapshot_id": self.snapshot_id,
                "request_id": self.request_id,
            },
            "kind": self.kind,
            "quantities": [
                {
                    "version": 1,
                    "variable_id": self.variable_id,
                    "variable_name": self.variable_name,
                    "role": self.role,
                    "units": self.units,
                    "population_id": self.population_id,
                    "regime_id": self.regime_id,
                    "horizon": _count(
                        self.horizon,
                        "horizon",
                        minimum=0,
                        maximum=2**64 - 1,
                        reason_code="invalid_argument",
                    ),
                    "functional_id": self.functional_id,
                    "conditioning": [],
                    "transform_id": self.transform_id,
                }
            ],
            "capabilities": list(self.capabilities),
            "trust": self.trust,
            "attestor": self.attestor,
        }


# --------------------------------------------------------------------------- results


@dataclass(frozen=True, slots=True)
class VarianceComponent:
    """One named part of a variance."""

    name: str
    variance: float


@dataclass(frozen=True, slots=True)
class MeasurementIdentities:
    """The separately frozen identities of a model (lowercase hex).

    A change in the parameters moves ``measurement``; a change of population moves ``population``,
    of regime ``regime`` and of arm scope ``arm``, each alone; a representation version moves
    ``representation``; provenance moves ``evidence``. ``model`` binds all six and is the value a
    consumer retains.
    """

    measurement: str
    population: str
    regime: str
    arm: str
    representation: str
    evidence: str
    model: str


def model_identities(model: MeasurementModel) -> MeasurementIdentities:
    """The identities of a model declaration, without preparing anything."""
    wire = json.loads(_native.measurement_model_identities(_dumps(model.to_wire())))
    return MeasurementIdentities(**wire)


@dataclass(repr=False, frozen=True, slots=True, eq=False)
class MeasurementEstimate(ScientificResult):
    """A corrected functional bound to its model, with the inputs to replay it.

    ``standard_error`` is an uncalibrated delta-method estimate (``inference`` says whether one is
    reported); ``parameter_uncertainty`` says whether the measurement parameters' uncertainty is in
    it. There is no interval: :meth:`interval` raises ``cell_not_licensed``.
    """

    _mean_binding: Any = field(default=None, init=False, repr=False, compare=False)

    model_identity: str
    route: str
    operation: str
    functional_id: str
    point: float
    naive_point: float | None
    standard_error: float | None
    variance_components: tuple[VarianceComponent, ...]
    rows: int
    data_identity: str
    inference: str
    parameter_uncertainty: str
    provider_trust: str | None
    representation_mapping_id: str | None
    _handle: Any = field(repr=False, compare=False)

    _model: MeasurementModel | None = field(default=None, repr=False, compare=False)

    @property
    def model(self) -> MeasurementModel | None:
        """The retained measurement declaration, when produced by a prepared model."""
        return self._model

    def artifact(
        self, artifact_id: str = _DEFAULT_ARTIFACT_ID
    ) -> ExportedArtifact[ConsumedMeasurement]:
        """Bytes and producer metadata; load returns the typed consumed result."""
        from ._artifact_protocol import exported_artifact

        return exported_artifact(self, artifact_id)

    @property
    def variance(self) -> float | None:
        """The delta-method variance (the square of :attr:`standard_error`)."""
        return None if self.standard_error is None else self.standard_error**2

    @property
    def detail(self) -> dict[str, Any]:
        """The estimator's own parts for the route: coefficients, covariance, components."""
        detail: dict[str, Any] = json.loads(self._handle.detail_json())
        return detail

    def interval(self) -> None:
        """Always raises ``cell_not_licensed``: no coverage record measures the method."""
        raise_refusal(self._handle.interval(), MeasurementModelRefusal)

    def downstream_linear_functional(self, weights: Any) -> tuple[float, float]:
        """Value and variance of ``weights' beta`` over the corrected coefficient vector.

        The coefficient vector is the intercept, the latent regressor, then the covariates. The
        variance is the quadratic form under the joint covariance, which includes the
        measurement-parameter uncertainty where it was propagated.
        """
        value, variance = _unwrap(
            self._handle.downstream_linear_functional(_vector(weights, "weights"))
        )
        return float(value), float(variance)

    def export(self, artifact_id: str = _DEFAULT_ARTIFACT_ID) -> bytes:
        """The model, its identities and the result with the inputs a consumer recomputes it from."""
        return bytes(_unwrap(self._handle.export(_name(artifact_id, "artifact_id"))))


@bind_mean_reading
def _estimate(handle: Any) -> MeasurementEstimate:
    summary = json.loads(handle.summary_json())
    return MeasurementEstimate(
        model_identity=summary["model_identity"],
        route=summary["route"],
        operation=summary["operation"],
        functional_id=summary["functional_id"],
        point=float(summary["point"]),
        naive_point=summary["naive_point"],
        standard_error=summary["standard_error"],
        variance_components=tuple(
            VarianceComponent(c["name"], float(c["variance"]))
            for c in summary["variance_components"]
        ),
        rows=int(summary["rows"]),
        data_identity=summary["data_identity"],
        inference=summary["inference"],
        parameter_uncertainty=summary["parameter_uncertainty"],
        provider_trust=summary["provider_trust"],
        representation_mapping_id=summary["representation_mapping_id"],
        _handle=handle,
    )


@dataclass(frozen=True, slots=True)
class ContinuousMeasurementEstimate:
    """A corrected coefficient under continuous classical measurement error, bound to its model.

    ``point`` is the treatment coefficient (index ``coefficient_index`` of ``coefficients``, on
    ``(1, regressor, covariates)``) and ``naive_point`` the uncorrected one. ``standard_error`` is
    uncalibrated: delta method for linear regression calibration, a bootstrap of the whole
    procedure (``bootstrap_replicates`` used, ``failed_bootstrap_replicates`` failed) for logistic
    calibration and SIMEX. ``method`` is ``regression_calibration``, ``simex`` or
    ``outcome_error_no_correction`` (error in the outcome under an identity link, which does not
    bias the slope). ``simex_lambdas`` and ``simex_means`` are the SIMEX path (``0`` first), one
    mean coefficient vector per value. There is no interval: :meth:`interval` raises
    ``cell_not_licensed``.
    """

    model_identity: str
    route: str
    functional_id: str
    point: float
    naive_point: float | None
    standard_error: float | None
    variance_components: tuple[VarianceComponent, ...]
    rows: int
    data_identity: str
    inference: str
    parameter_uncertainty: str
    method: str
    link: str
    location: str
    treatment_column: int | None
    error_variance: float | None
    error_variance_standard_error: float | None
    reliability: float | None
    bootstrap_replicates: int
    failed_bootstrap_replicates: int
    seed: int | None
    coefficients: tuple[float, ...]
    naive_coefficients: tuple[float, ...]
    coefficient_index: int
    simex_lambdas: tuple[float, ...] | None
    simex_means: tuple[tuple[float, ...], ...] | None
    simex_extrapolant: str | None
    _handle: Any = field(repr=False, compare=False)

    @property
    def variance(self) -> float | None:
        """The reported variance (the square of :attr:`standard_error`)."""
        return None if self.standard_error is None else self.standard_error**2

    def interval(self) -> None:
        """Always raises ``cell_not_licensed``: no coverage record measures the method."""
        raise_refusal(self._handle.interval(), MeasurementModelRefusal)


def _continuous(summary: dict[str, Any], handle: Any) -> ContinuousMeasurementEstimate:
    simex = summary["simex"]
    return ContinuousMeasurementEstimate(
        model_identity=summary["model_identity"],
        route=summary["route"],
        functional_id=summary["functional_id"],
        point=float(summary["point"]),
        naive_point=summary["naive_point"],
        standard_error=summary["standard_error"],
        variance_components=tuple(
            VarianceComponent(c["name"], float(c["variance"]))
            for c in summary["variance_components"]
        ),
        rows=int(summary["rows"]),
        data_identity=summary["data_identity"],
        inference=summary["inference"],
        parameter_uncertainty=summary["parameter_uncertainty"],
        method=summary["method"],
        link=summary["link"],
        location=summary["location"],
        treatment_column=summary["treatment_column"],
        error_variance=summary["error_variance"],
        error_variance_standard_error=summary["error_variance_standard_error"],
        reliability=summary["reliability"],
        bootstrap_replicates=int(summary["bootstrap_replicates"]),
        failed_bootstrap_replicates=int(summary["failed_bootstrap_replicates"]),
        seed=summary["seed"],
        coefficients=tuple(float(v) for v in summary["coefficients"]),
        naive_coefficients=tuple(float(v) for v in summary["naive_coefficients"]),
        coefficient_index=int(summary["coefficient_index"]),
        simex_lambdas=None if simex is None else tuple(float(v) for v in simex["lambdas"]),
        simex_means=None
        if simex is None
        else tuple(tuple(float(v) for v in row) for row in simex["means"]),
        simex_extrapolant=None if simex is None else simex["extrapolant"],
        _handle=handle,
    )


@dataclass(frozen=True, slots=True, eq=False)
class MeasurementRefresh:
    """The new prepared state and the result recomputed from it."""

    prepared: PreparedMeasurementModel
    estimate: MeasurementEstimate


@dataclass(frozen=True, slots=True, eq=False)
class PreparedMeasurementModel:
    """A measurement model, the analysis binding and the data, checked.

    Holds nothing that has not passed the prepare checks. Every method that changes it returns a
    new object and leaves this one untouched, whether it succeeds or refuses.
    """

    model_identity: str
    identities: MeasurementIdentities
    operation: str
    provider_trust: str | None
    route: str
    _handle: Any = field(repr=False, compare=False)

    @property
    def model(self) -> MeasurementModel:
        """The checked model."""
        return _model_from_wire(json.loads(self._handle.model_json()))

    @property
    def route_card(self) -> dict[str, Any]:
        """The route's assumptions, allowed operations and refused operations (with the reason)."""
        card: dict[str, Any] = json.loads(self._handle.route_card_json())
        return card

    def check_operation(self, operation: str) -> None:
        """Raise the refusal naming what would license ``operation``; return when licensed."""
        _choice(operation, "operation", OPERATIONS)
        raise_refusal(self._handle.check_operation(operation), MeasurementModelRefusal)

    def interval(self) -> None:
        """Always raises ``cell_not_licensed``: no coverage record measures the method."""
        raise_refusal(self._handle.interval(), MeasurementModelRefusal)

    def estimate(self, *, cancel: Any = None) -> MeasurementEstimate:
        """Form the corrected functional from the retained data."""
        result = _estimate(_unwrap(self._handle.estimate(cancel=cancel)))
        from ._freeze import freeze

        object.__setattr__(result, "_model", freeze(self.model))
        return result

    def estimate_continuous(
        self, data: ContinuousErrorData, *, cancel: Any = None
    ) -> ContinuousMeasurementEstimate:
        """Correct the treatment coefficient for continuous classical measurement error.

        Re-runs the prepare checks of the linear adjusted effect on this model and binding, then
        corrects by regression calibration or SIMEX under a linear or logistic outcome model. The
        nonlinear outcome operation stays refused by :meth:`check_operation`; this is its entry.

        Raises:
            MeasurementModelRefusal: ``measurement_not_identified`` when the model's route
                identifies no error variance (remedy: supply replicates or a validation sample),
                ``route_not_supported`` for a logit outcome with the error in the outcome (remedy:
                the binary misclassification route), and the estimator's other coded refusals.
        """
        if not isinstance(data, ContinuousErrorData):
            raise CausalTypeError("data must be a ContinuousErrorData")
        summary = _unwrap(
            self._handle.estimate_continuous(_dumps(data.to_wire()), cancel=cancel)
        )
        return _continuous(json.loads(summary), self._handle)

    def refresh(self, data: MeasurementData, *, cancel: Any = None) -> MeasurementRefresh:
        """Recompute with new outcome-only data of the same question.

        Keeps the model and its identities; recomputes the point, standard error and data identity.
        Data of another route family, functional or arm pair is refused (``route_not_supported``).
        """
        return _refreshed(self._handle.refresh(_dumps(_wire(data)), cancel=cancel))

    def with_validation_parameters(
        self,
        parameters: MeasurementParameters,
        evidence: ParameterEvidence,
        *,
        cancel: Any = None,
    ) -> MeasurementRefresh:
        """Rebuild the model with new validation parameters and evidence, and recompute.

        The measurement and evidence identities, and so the model identity, change; every prepare
        check runs again on the new model.
        """
        return _refreshed(
            self._handle.with_validation_parameters(
                _dumps(parameters.to_wire()), _dumps(evidence.to_wire()), cancel=cancel
            )
        )


def _wire(data: object) -> dict[str, Any]:
    to_wire = getattr(data, "to_wire", None)
    if not callable(to_wire):
        raise CausalTypeError(
            "data must be one of ClassicalLinearData, RepeatedLinearData, BinaryPrevalenceData, "
            "RiskDifferenceData, CategoricalData or LatentPosteriorData"
        )
    wire: dict[str, Any] = to_wire()
    return wire


def _prepared(handle: Any) -> PreparedMeasurementModel:
    identities = json.loads(handle.identities_json())
    return PreparedMeasurementModel(
        model_identity=identities["model"],
        identities=MeasurementIdentities(
            measurement=identities["measurement"],
            population=identities["population"],
            regime=identities["regime"],
            arm=identities["arm"],
            representation=identities["representation"],
            evidence=identities["evidence"],
            model=identities["model"],
        ),
        operation=identities["operation"],
        provider_trust=identities["provider_trust"],
        route=json.loads(handle.model_json())["route"],
        _handle=handle,
    )


def _refreshed(triple: tuple[Any, Any, str | None]) -> MeasurementRefresh:
    prepared, estimate, refusal = triple
    raise_refusal(refusal, MeasurementModelRefusal)
    prepared_model = _prepared(prepared)
    result = _estimate(estimate)
    from ._freeze import freeze

    object.__setattr__(result, "_model", freeze(prepared_model.model))
    return MeasurementRefresh(prepared=prepared_model, estimate=result)


def prepare_measurement_model(
    model: MeasurementModel,
    analysis: AnalysisBinding,
    data: MeasurementData,
    *,
    provider: ProviderObject | None = None,
    receipt: Any = None,
    cancel: Any = None,
) -> PreparedMeasurementModel:
    """Prepare a corrected functional under a declared measurement model.

    Every prepare check runs here, before any number is formed. The posterior route needs
    ``provider``: the exact object the model names, attested or verified, never natively trusted.
    ``receipt`` is a verification receipt the library issued (see :mod:`antecedent.external`).

    Raises:
        MeasurementModelRefusal: the first failed check, with its registered ``reason_code``,
            ``measurement_model.*`` ``detail`` and ``remedy``.
    """
    provider_json = None if provider is None else _dumps(provider.to_wire())
    handle = _unwrap(
        _native.measurement_model_prepare(
            _dumps(model.to_wire()),
            _dumps(analysis.to_wire()),
            _dumps(_wire(data)),
            provider_json,
            receipt,
            cancel=cancel,
        )
    )
    return _prepared(handle)


# --------------------------------------------------------------------------- consume


@dataclass(repr=False, frozen=True, slots=True, eq=False)
class ConsumedMeasurement(ScientificResult):
    """A consumed artifact: the model, the stored record and the values a recomputation produced.

    ``point``, ``standard_error`` and ``rows`` are recomputed from the stored sufficient inputs
    with the statistics layer only; a stored number that does not recompute is refused at
    consumption. ``stored_point`` is what the producer wrote.
    """

    _mean_binding: Any = field(default=None, init=False, repr=False, compare=False)

    model: MeasurementModel
    model_identity: str
    point: float
    naive_point: float | None
    standard_error: float
    variance: float
    rows: int
    functional_id: str
    route: str | None
    stored_point: float | None
    stored_standard_error: float | None
    data_identity: str | None
    provider_trust: str | None
    _handle: Any = field(repr=False, compare=False)

    def covers_data_identity(self, identity: str) -> bool:
        """Whether the stored result was computed from data with this identity."""
        return bool(self._handle.covers_data_identity(_name(identity, "identity")))

    def check_use(self, analysis: AnalysisBinding, operation: str) -> None:
        """Raise the refusal when the result may not be used for ``analysis`` and ``operation``."""
        _choice(operation, "operation", OPERATIONS)
        raise_refusal(
            self._handle.check_use(_dumps(analysis.to_wire()), operation), MeasurementModelRefusal
        )


@bind_mean_reading
def consume_measurement_model_artifact(
    artifact: bytes, expected_identity: str
) -> ConsumedMeasurement:
    """Independently consume a measurement-model artifact.

    ``expected_identity`` is the 64-character model identity the consumer retained
    independently of the bytes. A swapped sensitivity or specificity, population, regime, arm or
    representation version is a different identity and is refused, as is a stored point or standard
    error that does not recompute from its stored inputs.

    Raises:
        MeasurementModelRefusal: ``measurement_binding_mismatch`` for an identity, binding or
            recomputation failure, ``invalid_argument`` for corrupt, truncated or foreign bytes.
    """
    if not isinstance(artifact, (bytes, bytearray, memoryview)):
        raise CausalTypeError("artifact must be bytes")
    handle = _unwrap(
        _native.measurement_model_consume(
            bytes(artifact), _name(expected_identity, "expected_identity")
        )
    )
    summary = json.loads(handle.summary_json())
    replayed = summary["replayed"]
    record = summary["record"]
    if replayed is None:
        raise MeasurementModelRefusal(
            {
                "code": "invalid_argument",
                "stage": "consume",
                "detail": "measurement_model.artifact_invalid",
                "message": "the artifact carries a model but no result to consume",
            }
        )
    return ConsumedMeasurement(
        model=_model_from_wire(summary["model"]),
        model_identity=summary["model_identity"],
        point=float(replayed["point"]),
        naive_point=replayed["naive_point"],
        standard_error=float(replayed["standard_error"]),
        variance=float(replayed["variance"]),
        rows=int(replayed["rows"]),
        functional_id=replayed["functional_id"],
        route=None if record is None else record["route"],
        stored_point=None if record is None else float(record["point"]),
        stored_standard_error=None if record is None else record["standard_error"],
        data_identity=None if record is None else record["data_identity"],
        provider_trust=summary["provider_trust"],
        _handle=handle,
    )


# --------------------------------------------------------------------------- transport

_ERROR_PARAMETERS = ("sensitivity", "specificity", "confusion_matrix", "error_variance")


@dataclass(frozen=True, slots=True)
class InvarianceDeclaration:
    """The declared premise that named error parameters are the same in two populations.

    ``invariant`` names every parameter the correction uses (``sensitivity`` and ``specificity``
    for a binary indicator, ``confusion_matrix`` for an assigned categorical label,
    ``error_variance`` for a continuous one). A premise about one parameter does not license a
    correction that uses another. The premise is declared, never tested.
    """

    source_population: str
    target_population: str
    declarer: str
    invariant: tuple[str, ...]

    def __post_init__(self) -> None:
        _name(self.source_population, "source_population")
        _name(self.target_population, "target_population")
        _name(self.declarer, "declarer")
        if isinstance(self.invariant, str):
            raise CausalTypeError("invariant must be an iterable of parameter names")
        object.__setattr__(
            self,
            "invariant",
            tuple(_choice(p, "invariant parameter", _ERROR_PARAMETERS) for p in self.invariant),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "kind": "invariant",
            "source_population": self.source_population,
            "target_population": self.target_population,
            "declarer": self.declarer,
            "invariant": list(self.invariant),
        }


@dataclass(frozen=True, slots=True)
class DifferentialErrorDeclaration:
    """The declared case that error differs by population: nothing is carried.

    Only a model established for the target population is used; a missing one refuses
    (``differential_error_not_pooled``) rather than falling back to another population's rates.
    """

    declarer: str

    def __post_init__(self) -> None:
        _name(self.declarer, "declarer")

    def to_wire(self) -> dict[str, Any]:
        return {"kind": "differential", "declarer": self.declarer}


@dataclass(frozen=True, slots=True)
class MeasurementTransportReceipt:
    """The audit record of one transport of a measurement model.

    ``basis`` is ``established`` (the target had its own model; nothing carried),
    ``declared_invariance`` (parameters carried under the premise) or
    ``differential_established`` (the target's own model, selected under a declared differential
    structure). ``support`` is ``extrapolative`` when anything was carried.
    """

    identity: str
    source_model: str
    source_population: str
    target_population: str
    basis: Mapping[str, Any]
    carried: bool
    support: str
    evidence: Mapping[str, Any]
    assumptions: tuple[str, ...]

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> MeasurementTransportReceipt:
        return cls(
            identity=str(wire["identity"]),
            source_model=str(wire["source_model"]),
            source_population=str(wire["source_population"]),
            target_population=str(wire["target_population"]),
            basis=dict(wire["basis"]),
            carried=bool(wire["carried"]),
            support=str(wire["support"]),
            evidence=dict(wire["evidence"]),
            assumptions=tuple(wire["assumptions"]),
        )


@dataclass(repr=False, frozen=True, slots=True, eq=False)
class TransportedMeasurementEstimate(ScientificResult):
    """A corrected functional of a transported measurement model, with its receipt.

    ``estimate`` is the existing :class:`MeasurementEstimate` computed on the re-bound model
    (``estimate.model`` is ``None``: the re-bound model is identified by
    ``estimate.model_identity``). ``receipt`` and ``identity`` say how the model reached the target.
    The standard error is conditional on the premise and uncalibrated; there is no interval.
    """

    identity: str
    receipt: MeasurementTransportReceipt
    estimate: MeasurementEstimate
    native_target_data: bool = False

    @property
    def point(self) -> float:
        return self.estimate.point

    def interval(self) -> None:
        """Always raises ``cell_not_licensed``."""
        self.estimate.interval()

    def export(self, artifact_id: str = _DEFAULT_ARTIFACT_ID) -> bytes:
        """The existing measurement-model artifact of the re-bound model.

        It carries no receipt: a consumer re-derives it with :func:`verify_measurement_transport`.
        """
        return self.estimate.export(artifact_id)


def _structure_wire(structure: InvarianceDeclaration | DifferentialErrorDeclaration | None) -> Any:
    if structure is None:
        return None
    if not isinstance(structure, (InvarianceDeclaration, DifferentialErrorDeclaration)):
        raise CausalTypeError(
            "structure must be an InvarianceDeclaration or a DifferentialErrorDeclaration"
        )
    return _dumps(structure.to_wire())


def _models_wire(models: Sequence[MeasurementModel]) -> str:
    listed = list(models)
    if not listed or not all(isinstance(m, MeasurementModel) for m in listed):
        raise CausalTypeError("models must be a non-empty sequence of MeasurementModel")
    return _dumps([m.to_wire() for m in listed])


def transport_measurement_model(
    models: Sequence[MeasurementModel],
    analysis: AnalysisBinding,
    data: MeasurementData,
    *,
    structure: InvarianceDeclaration | DifferentialErrorDeclaration | None = None,
    provider: ProviderObject | None = None,
    receipt: Any = None,
    cancel: Any = None,
) -> TransportedMeasurementEstimate:
    """Carry a measurement model to the population of ``analysis`` and correct ``data`` with it.

    ``models`` are the models established so far, each for its own population. A model established
    for the analysis population is used as is; otherwise ``structure`` decides (see
    :class:`InvarianceDeclaration`, :class:`DifferentialErrorDeclaration`). Every prepare check of
    :func:`prepare_measurement_model` then runs against the re-bound model.

    Raises:
        MeasurementModelRefusal: the unmet premise, with a registered ``reason_code`` (for example
            ``transport_missing_evidence`` / ``measurement_model.invariance_premise_required``,
            ``transport_not_certified`` / ``measurement_model.invariance_not_declared``,
            ``differential_error_not_pooled``) and a ``remedy``.
    """
    provider_json = None if provider is None else _dumps(provider.to_wire())
    handle, receipt_json, refusal = _native.measurement_model_transport(
        _models_wire(models),
        _structure_wire(structure),
        _dumps(analysis.to_wire()),
        _dumps(_wire(data)),
        provider_json,
        receipt,
        cancel=cancel,
    )
    raise_refusal(refusal, MeasurementModelRefusal)
    assert receipt_json is not None
    wire = json.loads(receipt_json)
    return TransportedMeasurementEstimate(
        identity=str(wire["identity"]),
        receipt=MeasurementTransportReceipt._from_wire(wire),
        estimate=_estimate(handle),
    )


def verify_measurement_transport(
    models: Sequence[MeasurementModel],
    analysis: AnalysisBinding,
    *,
    structure: InvarianceDeclaration | DifferentialErrorDeclaration | None,
    expected_identity: str,
) -> MeasurementTransportReceipt:
    """Re-derive the transport from the source models and declaration; require the identity.

    The fresh-consumer path for a receipt, which the measurement artifact does not carry.

    Raises:
        MeasurementModelRefusal: whatever the transport refuses, and
            ``measurement_model.transport_receipt_mismatch`` (``measurement_binding_mismatch``)
            when the recomputed identity is not ``expected_identity``.
    """
    text, refusal = _native.measurement_model_transport_verify(
        _models_wire(models),
        _structure_wire(structure),
        _dumps(analysis.to_wire()),
        _name(expected_identity, "expected_identity"),
    )
    raise_refusal(refusal, MeasurementModelRefusal)
    assert text is not None
    return MeasurementTransportReceipt._from_wire(json.loads(text))


# --------------------------------------------------------------------------- pooled differential


@dataclass(frozen=True, slots=True)
class PopulationValidationCounts:
    """One population's validation counts of a misclassified binary indicator."""

    population: str
    true_positive: int
    false_negative: int
    true_negative: int
    false_positive: int

    def __post_init__(self) -> None:
        _name(self.population, "population")
        for count_name in ("true_positive", "false_negative", "true_negative", "false_positive"):
            _count(getattr(self, count_name), count_name)

    def _counts(self) -> tuple[int, int, int, int]:
        return (self.true_positive, self.false_negative, self.true_negative, self.false_positive)


@dataclass(frozen=True, slots=True)
class PooledDifferentialCorrection:
    """The target's corrected prevalence with differential error pooled across populations.

    ``sensitivity`` and ``specificity`` are ``(pooled mean logit, its variance,
    between-population variance, target rate mean, target rate standard deviation)``. The
    interval is an uncalibrated simulation interval (``calibration == "unmeasured"``).
    """

    target_population: str
    prevalence: float
    interval: tuple[float, float]
    sensitivity: tuple[float, float, float, float, float]
    specificity: tuple[float, float, float, float, float]
    discarded_draws: int
    truncated_draws: int
    draws: tuple[float, ...]
    calibration: str


def transport_pooled_differential(
    sources: Sequence[PopulationValidationCounts],
    target_population: str,
    observed_positive: int,
    observed_total: int,
    *,
    target_validation: PopulationValidationCounts | None = None,
    draws: int = 4000,
    seed: int = 1,
    threads: int | None = None,
) -> PooledDifferentialCorrection:
    """Correct the target's recorded prevalence with error rates that differ by population.

    The sensitivity and specificity logits of two or more validated ``sources`` are pooled by
    random effects; the between-population variance (the uncertainty of an invariance premise)
    is propagated with the target's own validation sample, when supplied, and the recorded
    prevalence into the Rogan-Gladen correction. Fewer than two sources, or pooled rates that
    leave the target unidentified, are refused (``differential_error_not_pooled``).
    """
    section = _native.measurement_pooled_differential_transport(
        [(source.population, *source._counts()) for source in sources],
        _name(target_population, "target_population"),
        _count(observed_positive, "observed_positive"),
        _count(observed_total, "observed_total"),
        target_validation=None if target_validation is None else target_validation._counts(),
        draws=_count(draws, "draws"),
        seed=seed,
        threads=threads,
    )
    return PooledDifferentialCorrection(
        target_population=section.target_population,
        prevalence=section.prevalence,
        interval=section.interval,
        sensitivity=section.sensitivity,
        specificity=section.specificity,
        discarded_draws=section.discarded_draws,
        truncated_draws=section.truncated_draws,
        draws=tuple(section.draws),
        calibration=section.calibration,
    )
