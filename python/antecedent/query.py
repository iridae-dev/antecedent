"""Typed causal queries for the Python facade.

Every query dataclass keeps a short positional prefix of the fields that
identify *which variables* the query is about (treatment/outcome/mediator/…);
everything after that — contrasts, levels, options — is keyword-only, via the
``KW_ONLY`` sentinel. ``kind`` is a discriminator, never a caller-set value:
it is ``init=False`` on every class.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from math import isfinite
from typing import Literal

from .errors import CausalValueError


def _require_name(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CausalValueError(f"{field_name} must be a non-empty variable name")


def _require_names(field_name: str, values: Sequence[str]) -> None:
    if not values:
        raise CausalValueError(f"{field_name} must contain at least one variable name")
    for value in values:
        _require_name(field_name, value)


def _require_finite(field_name: str, value: float) -> None:
    if not isfinite(value):
        raise CausalValueError(f"{field_name} must be finite, got {value!r}")


def _coordinate_values(
    field_name: str,
    coordinates: Sequence[float] | Mapping[str, float],
    treatments: Sequence[str],
) -> Sequence[float]:
    if isinstance(coordinates, Mapping):
        expected, actual = set(treatments), set(coordinates)
        if actual != expected:
            raise CausalValueError(
                f"{field_name} mapping keys must exactly match treatments; "
                f"missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}"
            )
        return [coordinates[name] for name in treatments]
    if len(coordinates) != len(treatments):
        raise CausalValueError(f"{field_name} must have one value per treatment")
    return coordinates


@dataclass(frozen=True, slots=True)
class AverageEffect:
    """Average treatment effect (static tabular)."""

    treatment: str
    outcome: str
    _: KW_ONLY
    control_level: float = 0.0
    active_level: float = 1.0
    target_population: object | None = None
    kind: Literal["average"] = field(default="average", init=False, repr=False)


@dataclass(frozen=True, slots=True)
class PulseEffect:
    """Temporal pulse intervention effect."""

    treatment: str
    outcome: str
    _: KW_ONLY
    active_level: float = 1.0
    treatment_lag: int = 1
    horizon_steps: int = 1
    kind: Literal["pulse"] = field(default="pulse", init=False, repr=False)


@dataclass(frozen=True, slots=True)
class SustainedEffect:
    """Temporal sustained intervention effect."""

    treatment: str
    outcome: str
    _: KW_ONLY
    active_level: float = 1.0
    treatment_lag: int = 1
    horizon_steps: int = 1
    kind: Literal["sustained"] = field(default="sustained", init=False, repr=False)


@dataclass(frozen=True, slots=True)
class InterventionalDistribution:
    """Interventional distribution query (static)."""

    outcome: str
    _: KW_ONLY
    interventions: dict[str, float] = field(default_factory=dict)
    conditioning: Sequence[str] = ()
    kind: Literal["distribution"] = field(default="distribution", init=False, repr=False)


@dataclass(frozen=True, slots=True)
class PathSpecificEffect:
    """Path-specific effect query (static)."""

    treatment: str
    outcome: str
    _: KW_ONLY
    path_nodes: Sequence[str] | None = None
    control_level: float = 0.0
    active_level: float = 1.0
    max_paths: int = 64
    max_len: int = 16
    kind: Literal["path_specific"] = field(default="path_specific", init=False, repr=False)


@dataclass(frozen=True, slots=True)
class ConditionalEffect:
    """Conditional / context average effect with a single effect modifier."""

    treatment: str
    outcome: str
    modifier: str
    _: KW_ONLY
    control_level: float = 0.0
    active_level: float = 1.0
    kind: Literal["conditional"] = field(default="conditional", init=False, repr=False)


@dataclass(frozen=True, slots=True)
class MediationEffect:
    """Static mediation (treatment → mediator(s) → outcome)."""

    treatment: str
    outcome: str
    _: KW_ONLY
    mediators: Sequence[str]
    contrast: Literal["total", "direct", "mediated"] = "mediated"
    control_level: float = 0.0
    active_level: float = 1.0
    kind: Literal["mediation"] = field(default="mediation", init=False, repr=False)


@dataclass(frozen=True, slots=True)
class Counterfactual:
    """Unit-level ITE via GCM abduction–action–prediction."""

    treatment: str
    outcome: str
    _: KW_ONLY
    control_level: float = 0.0
    active_level: float = 1.0
    kind: Literal["counterfactual"] = field(default="counterfactual", init=False, repr=False)


@dataclass(frozen=True, slots=True)
class TemporalMediationEffect:
    """Temporal linear mediation (treatment → mediator → outcome)."""

    treatment: str
    mediator: str
    outcome: str
    _: KW_ONLY
    contrast: Literal["total", "direct", "mediated"] = "mediated"
    control_level: float = 0.0
    active_level: float = 1.0
    kind: Literal["temporal_mediation"] = field(
        default="temporal_mediation", init=False, repr=False
    )


@dataclass(frozen=True, slots=True)
class ResponseCurve:
    """Mean causal response ``a -> E[Y | do(A=a)]`` on an explicit grid."""

    treatment: str
    outcome: str
    _: KW_ONLY
    grid: Sequence[float]
    target_population: object | None = None
    observation: object | None = None
    observation_assumptions: Sequence[object] = ()
    kind: Literal["response_curve"] = field(default="response_curve", init=False, repr=False)

    def __post_init__(self) -> None:
        _require_name("outcome", self.outcome)
        _require_name("treatment", self.treatment)
        if len(self.grid) < 2:
            raise CausalValueError("grid must contain at least two treatment values")
        previous: float | None = None
        for value in self.grid:
            _require_finite("grid values", value)
            if previous is not None and value <= previous:
                raise CausalValueError("grid values must be strictly increasing")
            previous = value


@dataclass(frozen=True, slots=True)
class AverageDerivative:
    """Population-average derivative of a continuous-treatment response."""

    treatment: str
    outcome: str
    _: KW_ONLY
    weighting: object | None = None
    target_population: object | None = None
    observation: object | None = None
    observation_assumptions: Sequence[object] = ()
    kind: Literal["average_derivative"] = field(
        default="average_derivative", init=False, repr=False
    )

    def __post_init__(self) -> None:
        _require_name("outcome", self.outcome)
        _require_name("treatment", self.treatment)


@dataclass(frozen=True, slots=True)
class PointDerivative:
    """Local derivative of a response curve at one treatment value."""

    treatment: str
    outcome: str
    _: KW_ONLY
    at: float
    order: int = 1
    observation: object | None = None
    observation_assumptions: Sequence[object] = ()
    kind: Literal["point_derivative"] = field(default="point_derivative", init=False, repr=False)

    def __post_init__(self) -> None:
        _require_name("outcome", self.outcome)
        _require_name("treatment", self.treatment)
        _require_finite("at", self.at)
        if self.order not in (1, 2):
            raise CausalValueError(f"order must be 1 or 2, got {self.order!r}")


@dataclass(frozen=True, slots=True)
class Elasticity:
    """Log-outcome/log-treatment derivative at a positive treatment value."""

    treatment: str
    outcome: str
    _: KW_ONLY
    at: float
    observation: object | None = None
    observation_assumptions: Sequence[object] = ()
    kind: Literal["elasticity"] = field(default="elasticity", init=False, repr=False)

    def __post_init__(self) -> None:
        _require_name("outcome", self.outcome)
        _require_name("treatment", self.treatment)
        _require_finite("at", self.at)
        if self.at <= 0.0:
            raise CausalValueError("Elasticity.at must be positive for the log-treatment scale")


@dataclass(frozen=True, slots=True)
class SemiElasticity:
    """Derivative using one logarithmic scale and one identity scale."""

    treatment: str
    outcome: str
    _: KW_ONLY
    at: float
    log_scale: Literal["treatment", "outcome"] = "treatment"
    observation: object | None = None
    observation_assumptions: Sequence[object] = ()
    kind: Literal["semi_elasticity"] = field(default="semi_elasticity", init=False, repr=False)

    def __post_init__(self) -> None:
        _require_name("outcome", self.outcome)
        _require_name("treatment", self.treatment)
        _require_finite("at", self.at)
        if self.log_scale not in ("treatment", "outcome"):
            raise CausalValueError(
                f"log_scale must be 'treatment' or 'outcome', got {self.log_scale!r}"
            )
        if self.log_scale == "treatment" and self.at <= 0.0:
            raise CausalValueError("SemiElasticity.at must be positive when log_scale='treatment'")


@dataclass(frozen=True, slots=True)
class DirectionalDerivative:
    """Response derivative along a direction in a vector intervention space."""

    treatments: Sequence[str]
    outcomes: Sequence[str]
    _: KW_ONLY
    at: Sequence[float] | Mapping[str, float]
    direction: Sequence[float] | Mapping[str, float]
    observation: object | None = None
    observation_assumptions: Sequence[object] = ()
    kind: Literal["directional_derivative"] = field(
        default="directional_derivative", init=False, repr=False
    )

    def __post_init__(self) -> None:
        _require_names("outcomes", self.outcomes)
        _require_names("treatments", self.treatments)
        at = _coordinate_values("at", self.at, self.treatments)
        direction = _coordinate_values("direction", self.direction, self.treatments)
        for value in at:
            _require_finite("at values", value)
        for value in direction:
            _require_finite("direction values", value)
        if not any(value != 0.0 for value in direction):
            raise CausalValueError("direction must contain at least one non-zero value")


@dataclass(frozen=True, slots=True)
class ResponseJacobian:
    """Low-dimensional Jacobian of outcomes with respect to treatments."""

    treatments: Sequence[str]
    outcomes: Sequence[str]
    _: KW_ONLY
    at: Sequence[float] | Mapping[str, float]
    observation: object | None = None
    observation_assumptions: Sequence[object] = ()
    kind: Literal["response_jacobian"] = field(default="response_jacobian", init=False, repr=False)

    def __post_init__(self) -> None:
        _require_names("outcomes", self.outcomes)
        _require_names("treatments", self.treatments)
        at = _coordinate_values("at", self.at, self.treatments)
        for value in at:
            _require_finite("at values", value)


@dataclass(frozen=True, slots=True)
class InterventionResponse:
    """Mean outcome under an existing static, stochastic, or modified intervention."""

    outcome: str
    _: KW_ONLY
    intervention: object
    target_population: object | None = None
    observation: object | None = None
    observation_assumptions: Sequence[object] = ()
    kind: Literal["intervention_response"] = field(
        default="intervention_response", init=False, repr=False
    )

    def __post_init__(self) -> None:
        _require_name("outcome", self.outcome)
        if self.intervention is None:
            raise CausalValueError("intervention must not be None")


__all__ = [
    "AverageDerivative",
    "AverageEffect",
    "ConditionalEffect",
    "Counterfactual",
    "DirectionalDerivative",
    "Elasticity",
    "InterventionalDistribution",
    "InterventionResponse",
    "MediationEffect",
    "PathSpecificEffect",
    "PulseEffect",
    "PointDerivative",
    "ResponseCurve",
    "ResponseJacobian",
    "SemiElasticity",
    "SustainedEffect",
    "TemporalMediationEffect",
]
