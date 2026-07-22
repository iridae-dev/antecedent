"""Typed causal queries for the Python facade."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence


@dataclass(frozen=True)
class AverageEffect:
    """Average treatment effect (static tabular)."""

    treatment: str
    outcome: str
    control_level: float = 0.0
    active_level: float = 1.0
    target_population: object | None = None
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


@dataclass(frozen=True)
class InterventionalDistribution:
    """Interventional distribution query (static)."""

    outcome: str
    interventions: dict[str, float] = field(default_factory=dict)
    conditioning: Sequence[str] = ()
    kind: Literal["distribution"] = "distribution"


@dataclass(frozen=True)
class PathSpecificEffect:
    """Path-specific effect query (static)."""

    treatment: str
    outcome: str
    path_nodes: Sequence[str] | None = None
    control_level: float = 0.0
    active_level: float = 1.0
    max_paths: int = 64
    max_len: int = 16
    kind: Literal["path_specific"] = "path_specific"


@dataclass(frozen=True)
class ConditionalEffect:
    """Conditional / context average effect with a single effect modifier."""

    treatment: str
    outcome: str
    modifier: str
    control_level: float = 0.0
    active_level: float = 1.0
    kind: Literal["conditional"] = "conditional"


@dataclass(frozen=True)
class TemporalMediationEffect:
    """Temporal linear mediation (treatment → mediator → outcome)."""

    treatment: str
    mediator: str
    outcome: str
    contrast: Literal["total", "direct", "mediated"] = "mediated"
    control_level: float = 0.0
    active_level: float = 1.0
    kind: Literal["temporal_mediation"] = "temporal_mediation"


__all__ = [
    "AverageEffect",
    "ConditionalEffect",
    "InterventionalDistribution",
    "PathSpecificEffect",
    "PulseEffect",
    "SustainedEffect",
    "TemporalMediationEffect",
]
