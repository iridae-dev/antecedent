"""Observation-mechanism specifications for latent scientific outcomes.

These dataclasses describe what was recorded.  Independence and structural
assumptions are separate objects because a censoring/selection column does not
by itself justify MAR, independent censoring, or non-informative selection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from ._data import as_columns
from ._native import (
    gaussian_observation_log_likelihood as _gaussian_observation_log_likelihood,
)
from ._native import observation_adjusted_outcome as _observation_adjusted_outcome
from .errors import CausalValueError


def _name(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CausalValueError(f"{field_name} must be a non-empty variable name")


@dataclass(frozen=True, slots=True)
class Complete:
    """The scientific outcome is observed directly."""


@dataclass(frozen=True, slots=True)
class RightCensored:
    latent: str
    observed: str
    censoring: str
    event: str

    def __post_init__(self) -> None:
        for field_name in ("latent", "observed", "censoring", "event"):
            _name(field_name, getattr(self, field_name))


@dataclass(frozen=True, slots=True)
class LeftCensored:
    latent: str
    observed: str
    censoring: str
    event: str

    def __post_init__(self) -> None:
        for field_name in ("latent", "observed", "censoring", "event"):
            _name(field_name, getattr(self, field_name))


@dataclass(frozen=True, slots=True)
class IntervalCensored:
    latent: str
    lower: str
    upper: str

    def __post_init__(self) -> None:
        for field_name in ("latent", "lower", "upper"):
            _name(field_name, getattr(self, field_name))


@dataclass(frozen=True, slots=True)
class Truncated:
    latent: str
    observed: str
    lower: str | None = None
    upper: str | None = None

    def __post_init__(self) -> None:
        _name("latent", self.latent)
        _name("observed", self.observed)
        if self.lower is not None:
            _name("lower", self.lower)
        if self.upper is not None:
            _name("upper", self.upper)
        if self.lower is None and self.upper is None:
            raise CausalValueError("Truncated requires at least one of lower or upper")


@dataclass(frozen=True, slots=True)
class Selected:
    latent: str
    observed: str
    indicator: str

    def __post_init__(self) -> None:
        for field_name in ("latent", "observed", "indicator"):
            _name(field_name, getattr(self, field_name))


@dataclass(frozen=True, slots=True)
class IndependentGiven:
    """Claim that observation is independent given the named variables."""

    variables: Sequence[str]

    def __post_init__(self) -> None:
        for value in self.variables:
            _name("variables", value)


@dataclass(frozen=True, slots=True)
class OutcomeIndependentGiven:
    """Claim that observation is outcome-independent given named variables."""

    variables: Sequence[str]

    def __post_init__(self) -> None:
        for value in self.variables:
            _name("variables", value)


@dataclass(frozen=True, slots=True)
class Structural:
    """Reference to a caller-supplied structural observation model."""

    model_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise CausalValueError("model_id must be a non-empty string")


@dataclass(frozen=True, slots=True)
class AdjustedOutcome:
    """Observation-adjusted pseudo-values and diagnostic-only weights.

    ``values`` already contain their IPW/AIPW or IPCW correction. ``weights``
    are exposed for diagnostics and must not be applied to ``values`` again.
    This stage result is deliberately not a response-curve estimate.
    """

    values: Sequence[float]
    weights: Sequence[float]
    method: str

    def __post_init__(self) -> None:
        if len(self.values) != len(self.weights):
            raise CausalValueError("values and weights must have the same length")
        if not self.method.strip():
            raise CausalValueError("method must be a non-empty string")


def adjusted_outcome(
    data: Any,
    query: Any,
    *,
    delayed_entry: str | None = None,
    correction: Literal["ipw", "aipw"] = "aipw",
    observation_probability_floor: float = 0.01,
    censoring_survival_floor: float = 0.01,
) -> AdjustedOutcome:
    """Construct selected-outcome or censoring-adjusted pseudo-values.

    The response query must have one treatment and one outcome and carry an
    explicit observation mechanism plus exactly one assumption supported by
    the requested estimator. Unsupported compositions fail closed.
    """

    treatment, outcome, mechanism, assumption = _stage_inputs(query)
    names, columns = as_columns(data)
    names, columns = _ensure_latent_schema_column(names, columns, mechanism)
    kwargs = _mechanism_kwargs(mechanism)
    kwargs.update(_assumption_kwargs(assumption))
    raw = cast(Any, _observation_adjusted_outcome)(
        names,
        columns,
        treatment,
        outcome,
        **kwargs,
        delayed_entry=delayed_entry,
        correction=correction,
        observation_probability_floor=observation_probability_floor,
        censoring_survival_floor=censoring_survival_floor,
    )
    return AdjustedOutcome(tuple(raw.values), tuple(raw.weights), raw.method)


def gaussian_log_likelihood(
    data: Any,
    query: Any,
    *,
    means: Sequence[float],
    sigma: float,
) -> float:
    """Evaluate the opt-in Gaussian observation log likelihood.

    The query must include ``Structural("gaussian_observation_likelihood")``.
    This is the parametric likelihood path for censoring/truncation, not an
    implicit correction for a response estimator.
    """

    treatment, outcome, mechanism, assumption = _stage_inputs(query)
    if not (
        isinstance(assumption, Structural)
        and assumption.model_id == "gaussian_observation_likelihood"
    ):
        raise CausalValueError(
            "Gaussian observation likelihood requires "
            "Structural('gaussian_observation_likelihood')"
        )
    names, columns = as_columns(data)
    names, columns = _ensure_latent_schema_column(names, columns, mechanism)
    kwargs = _mechanism_kwargs(mechanism)
    kwargs.update(_assumption_kwargs(assumption))
    return float(
        cast(Any, _gaussian_observation_log_likelihood)(
            names,
            columns,
            treatment,
            outcome,
            means=list(means),
            sigma=sigma,
            **kwargs,
        )
    )


def _stage_inputs(query: Any) -> tuple[str, str, object, object]:
    treatment = getattr(query, "treatment", None)
    outcome = getattr(query, "outcome", None)
    if not isinstance(treatment, str) or not isinstance(outcome, str):
        raise CausalValueError(
            "observation stages require a scalar response query with one treatment and outcome"
        )
    mechanism = getattr(query, "observation", None)
    if mechanism is None or isinstance(mechanism, Complete):
        raise CausalValueError("query must carry a non-complete observation mechanism")
    assumptions = tuple(getattr(query, "observation_assumptions", ()))
    if len(assumptions) != 1:
        raise CausalValueError("observation stage requires exactly one explicit assumption")
    latent = getattr(mechanism, "latent", None)
    if latent != outcome:
        raise CausalValueError(
            "query outcome must equal the observation mechanism's latent outcome"
        )
    return treatment, outcome, mechanism, assumptions[0]


def _mechanism_kwargs(mechanism: object) -> dict[str, object]:
    if isinstance(mechanism, RightCensored):
        return {
            "observation_kind": "right_censored",
            "latent": mechanism.latent,
            "observed": mechanism.observed,
            "censoring": mechanism.censoring,
            "event": mechanism.event,
        }
    if isinstance(mechanism, LeftCensored):
        return {
            "observation_kind": "left_censored",
            "latent": mechanism.latent,
            "observed": mechanism.observed,
            "censoring": mechanism.censoring,
            "event": mechanism.event,
        }
    if isinstance(mechanism, IntervalCensored):
        return {
            "observation_kind": "interval_censored",
            "latent": mechanism.latent,
            "lower": mechanism.lower,
            "upper": mechanism.upper,
        }
    if isinstance(mechanism, Truncated):
        return {
            "observation_kind": "truncated",
            "latent": mechanism.latent,
            "observed": mechanism.observed,
            "lower": mechanism.lower,
            "upper": mechanism.upper,
        }
    if isinstance(mechanism, Selected):
        return {
            "observation_kind": "selected",
            "latent": mechanism.latent,
            "observed": mechanism.observed,
            "indicator": mechanism.indicator,
        }
    raise CausalValueError(f"unsupported observation mechanism {type(mechanism).__name__}")


def _assumption_kwargs(assumption: object) -> dict[str, object]:
    if isinstance(assumption, IndependentGiven):
        return {
            "assumption_kind": "independent_given",
            "assumption_variables": list(assumption.variables),
        }
    if isinstance(assumption, OutcomeIndependentGiven):
        return {
            "assumption_kind": "outcome_independent_given",
            "assumption_variables": list(assumption.variables),
        }
    if isinstance(assumption, Structural):
        return {
            "assumption_kind": "structural",
            "structural_model": assumption.model_id,
        }
    raise CausalValueError(f"unsupported observation assumption {type(assumption).__name__}")


def _ensure_latent_schema_column(
    names: list[str], columns: list[NDArray[np.float64]], mechanism: object
) -> tuple[list[str], list[NDArray[np.float64]]]:
    """Represent an unrecorded scientific outcome in the schema without imputing it."""

    latent = getattr(mechanism, "latent", None)
    if not isinstance(latent, str) or latent in names:
        return names, columns
    if not columns:
        raise CausalValueError("observation data must contain at least one recorded column")
    return [*names, latent], [*columns, np.full(len(columns[0]), np.nan, dtype=float)]


__all__ = [
    "AdjustedOutcome",
    "Complete",
    "IndependentGiven",
    "IntervalCensored",
    "LeftCensored",
    "OutcomeIndependentGiven",
    "RightCensored",
    "Selected",
    "Structural",
    "Truncated",
    "adjusted_outcome",
    "gaussian_log_likelihood",
]
