"""Typed intervention specifications for :class:`InterventionResponse`."""

from __future__ import annotations

import math
from collections.abc import Sequence as SequenceValues
from dataclasses import dataclass

from .errors import CausalValueError


def _variable(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise CausalValueError("variable must be a non-empty string")


def _finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise CausalValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class Set:
    """Hard assignment ``do(variable := value)``."""

    variable: str
    value: float

    def __post_init__(self) -> None:
        _variable(self.variable)
        _finite("value", self.value)


@dataclass(frozen=True, slots=True)
class Shift:
    """Additive modified-treatment policy ``variable := variable + delta``."""

    variable: str
    delta: float

    def __post_init__(self) -> None:
        _variable(self.variable)
        _finite("delta", self.delta)


@dataclass(frozen=True, slots=True)
class Bernoulli:
    """Bernoulli stochastic assignment with success probability ``p``."""

    variable: str
    p: float

    def __post_init__(self) -> None:
        _variable(self.variable)
        if not math.isfinite(self.p) or not 0.0 <= self.p <= 1.0:
            raise CausalValueError("p must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class Gaussian:
    """Gaussian stochastic assignment parameterized by mean and variance."""

    variable: str
    mean: float
    variance: float

    def __post_init__(self) -> None:
        _variable(self.variable)
        _finite("mean", self.mean)
        if not math.isfinite(self.variance) or self.variance <= 0.0:
            raise CausalValueError("variance must be finite and positive")


@dataclass(frozen=True, slots=True)
class Categorical:
    """Categorical assignment over integer values ``0, ..., k - 1``."""

    variable: str
    probabilities: SequenceValues[float]

    def __post_init__(self) -> None:
        _variable(self.variable)
        probs = tuple(float(value) for value in self.probabilities)
        if not probs or any(not math.isfinite(value) or value < 0.0 for value in probs):
            raise CausalValueError("probabilities must be non-empty, finite, and non-negative")
        if sum(probs) <= 0.0:
            raise CausalValueError("probabilities must have positive total mass")
        object.__setattr__(self, "probabilities", probs)


@dataclass(frozen=True, slots=True)
class Soft:
    """Mechanism replacement; represented here but not currently estimable."""

    variable: str
    mechanism: str

    def __post_init__(self) -> None:
        _variable(self.variable)
        if not isinstance(self.mechanism, str) or not self.mechanism:
            raise CausalValueError("mechanism must be a non-empty string")


@dataclass(frozen=True, slots=True)
class Sequence:
    """Temporal sequence; represented here but not currently estimable."""

    steps: SequenceValues[object]

    def __post_init__(self) -> None:
        steps = tuple(self.steps)
        if not steps:
            raise CausalValueError("steps must not be empty")
        object.__setattr__(self, "steps", steps)


Intervention = Set | Shift | Bernoulli | Gaussian | Categorical | Soft | Sequence

__all__ = [
    "Bernoulli",
    "Categorical",
    "Gaussian",
    "Intervention",
    "Sequence",
    "Set",
    "Shift",
    "Soft",
]
