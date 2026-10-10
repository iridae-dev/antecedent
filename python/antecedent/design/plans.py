"""Typed design plans and the structure prior they are ranked against.

A plan says what a study would do; :func:`antecedent.design.rank_designs` ranks plans by how
much they raise the probability that a query becomes identified. The four plans mirror the
Rust plan types (``MeasurementPlan``, ``ExperimentPlan``, ``EnvironmentPlan`` and
``SamplingPlan``). Every plan is a frozen value validated on construction; the native wire
form is built internally and is not part of the public surface::

    plans = [
        design.Measurement([3], cost=2.0),
        design.Environment(7, additional_rows=50),
        design.Sampling(10),
        design.Experiment([0]),
    ]
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ..errors import CausalTypeError, CausalValueError

PlanKind = Literal["measure", "intervene", "observe_environment", "increase_sampling_rate"]


def _count(value: object, what: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CausalTypeError(f"{what} must be an integer, not {type(value).__name__}")
    if value < minimum:
        raise CausalValueError(f"{what} must be at least {minimum}, got {value}")
    return value


def _ids(values: Sequence[int], what: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise CausalTypeError(f"{what} must be a sequence of integer ids")
    out = tuple(_count(v, f"{what} entry") for v in values)
    if not out:
        raise CausalValueError(f"{what} must be non-empty")
    return out


def _cost(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CausalTypeError(f"cost must be a number, not {type(value).__name__}")
    if not math.isfinite(value) or value < 0:
        raise CausalValueError(f"cost must be finite and non-negative, got {value}")
    return float(value)


def _label(value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        raise CausalTypeError("id must be a string or None")
    return value


def _common(plan: Any) -> None:
    object.__setattr__(plan, "cost", _cost(plan.cost))
    _count(plan.sample_budget, "sample_budget")
    if plan.tag is not None:
        _count(plan.tag, "tag")
    _label(plan.id)


def _base_wire(plan: Any, kind: PlanKind, index: int) -> dict[str, Any]:
    return {
        "kind": kind,
        "cost": plan.cost,
        "sample_budget": plan.sample_budget,
        "tag": index if plan.tag is None else plan.tag,
    }


@dataclass(frozen=True, slots=True)
class Measurement:
    """Measure additional ``variables`` (raw variable ids) on the existing units."""

    variables: tuple[int, ...]
    cost: float = 0.0
    sample_budget: int = 0
    tag: int | None = None
    id: str | None = None

    kind = "measure"

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", _ids(self.variables, "variables"))
        _common(self)

    def _wire(self, index: int) -> dict[str, Any]:
        return {**_base_wire(self, "measure", index), "variables": list(self.variables)}


@dataclass(frozen=True, slots=True)
class Experiment:
    """Intervene on ``targets`` (raw variable ids)."""

    targets: tuple[int, ...]
    cost: float = 0.0
    sample_budget: int = 0
    tag: int | None = None
    id: str | None = None

    kind = "intervene"

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", _ids(self.targets, "targets"))
        _common(self)

    def _wire(self, index: int) -> dict[str, Any]:
        return {**_base_wire(self, "intervene", index), "targets": list(self.targets)}


@dataclass(frozen=True, slots=True)
class Environment:
    """Observe a further environment (raw environment id), optionally ``additional_rows``."""

    environment: int
    additional_rows: int = 0
    cost: float = 0.0
    sample_budget: int = 0
    tag: int | None = None
    id: str | None = None

    kind = "observe_environment"

    def __post_init__(self) -> None:
        _count(self.environment, "environment")
        _count(self.additional_rows, "additional_rows")
        _common(self)

    def _wire(self, index: int) -> dict[str, Any]:
        return {
            **_base_wire(self, "observe_environment", index),
            "environment": self.environment,
            "additional_rows": self.additional_rows,
        }


@dataclass(frozen=True, slots=True)
class Sampling:
    """Collect ``additional_samples`` more samples of the existing design."""

    additional_samples: int
    cost: float = 0.0
    sample_budget: int = 0
    tag: int | None = None
    id: str | None = None

    kind = "increase_sampling_rate"

    def __post_init__(self) -> None:
        _count(self.additional_samples, "additional_samples", minimum=1)
        _common(self)

    def _wire(self, index: int) -> dict[str, Any]:
        return {
            **_base_wire(self, "increase_sampling_rate", index),
            "additional_samples": self.additional_samples,
        }


#: Any structural design plan.
DesignPlan = Measurement | Experiment | Environment | Sampling
PLAN_TYPES = (Measurement, Experiment, Environment, Sampling)


def plan_id(plan: DesignPlan, index: int) -> str:
    """The plan's label, defaulting to ``<kind>-<position>``."""
    return plan.id if plan.id is not None else f"{plan.kind}-{index}"


@dataclass(frozen=True, slots=True)
class StructurePrior:
    """Posterior weights over candidate causal structures, each flagged identified or not.

    ``weights[i]`` is the weight of structure ``i``, ``identified[i]`` whether the query is
    identified in it and ``keys[i]`` a stable structure key. ``identified_under_intervention``
    optionally gives the same flag once the structure is intervened on, and ``features`` an
    integer feature per structure.
    """

    weights: tuple[float, ...]
    identified: tuple[bool, ...]
    keys: tuple[int, ...]
    identified_under_intervention: tuple[bool, ...] | None = None
    features: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        try:
            weights = tuple(float(w) for w in self.weights)
        except (TypeError, ValueError) as error:
            raise CausalTypeError("weights must be a sequence of numbers") from error
        if not weights:
            raise CausalValueError("a structure prior needs at least one structure")
        if not all(math.isfinite(w) and w >= 0 for w in weights) or sum(weights) <= 0:
            raise CausalValueError("weights must be finite, non-negative and not all zero")
        n = len(weights)
        identified = tuple(bool(v) for v in self.identified)
        keys = tuple(_count(k, "keys entry") for k in self.keys)
        if len(identified) != n or len(keys) != n:
            raise CausalValueError("weights, identified and keys must have the same length")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "identified", identified)
        object.__setattr__(self, "keys", keys)
        if self.identified_under_intervention is not None:
            under = tuple(bool(v) for v in self.identified_under_intervention)
            if len(under) != n:
                raise CausalValueError("identified_under_intervention must match weights")
            object.__setattr__(self, "identified_under_intervention", under)
        if self.features is not None:
            features = tuple(_count(f, "features entry") for f in self.features)
            if len(features) != n:
                raise CausalValueError("features must match weights")
            object.__setattr__(self, "features", features)

    @classmethod
    def uniform(cls, identified: Sequence[bool]) -> StructurePrior:
        """Equal weight on each structure, keyed by position."""
        n = len(identified)
        return cls((1.0,) * n, tuple(identified), tuple(range(n)))


__all__ = [
    "PLAN_TYPES",
    "DesignPlan",
    "Environment",
    "Experiment",
    "Measurement",
    "PlanKind",
    "Sampling",
    "StructurePrior",
    "plan_id",
]
