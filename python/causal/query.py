"""Typed causal queries for the Python facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AverageEffect:
    """Average treatment effect (static tabular)."""

    treatment: str
    outcome: str
    control_level: float = 0.0
    active_level: float = 1.0
    kind: Literal["average"] = "average"


@dataclass(frozen=True)
class PulseEffect:
    """Temporal pulse intervention effect."""

    treatment: str
    outcome: str
    active_level: float = 1.0
    treatment_lag: int = 1
    horizon_steps: int = 1
    kind: Literal["pulse"] = "pulse"


@dataclass(frozen=True)
class SustainedEffect:
    """Temporal sustained intervention effect."""

    treatment: str
    outcome: str
    active_level: float = 1.0
    treatment_lag: int = 1
    horizon_steps: int = 1
    kind: Literal["sustained"] = "sustained"


__all__ = ["AverageEffect", "PulseEffect", "SustainedEffect"]
