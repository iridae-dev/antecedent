"""Typed causal queries for the Python facade.

Every query dataclass keeps a short positional prefix of the fields that
identify *which variables* the query is about (treatment/outcome/mediator/…);
everything after that — contrasts, levels, options — is keyword-only, via the
``KW_ONLY`` sentinel. ``kind`` is a discriminator, never a caller-set value:
it is ``init=False`` on every class.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Literal


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


__all__ = [
    "AverageEffect",
    "ConditionalEffect",
    "Counterfactual",
    "InterventionalDistribution",
    "MediationEffect",
    "PathSpecificEffect",
    "PulseEffect",
    "SustainedEffect",
    "TemporalMediationEffect",
]
