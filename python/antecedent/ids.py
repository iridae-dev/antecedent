"""Typed identifier / estimator / latency / refute wire ids (Pythonic enums)."""

from __future__ import annotations

from enum import StrEnum


class Identifier(StrEnum):
    """Identification strategies (wire ids match Rust ``IdentifierId``)."""

    BACKDOOR_ADJUSTMENT = "backdoor.adjustment"
    BACKDOOR_EFFICIENT = "backdoor.efficient"
    FRONTDOOR = "frontdoor"
    IV = "iv"
    RD_SHARP = "rd.sharp"
    TEMPORAL_BACKDOOR_UNFOLDED = "temporal.backdoor.unfolded"
    GENERALIZED_ADJUSTMENT = "generalized.adjustment"
    GENERAL_ID = "general.id"
    PATH_SPECIFIC_NATURAL = "path_specific.natural"
    AUTO = "auto"


class Estimator(StrEnum):
    """Estimation strategies (wire ids match Rust ``EstimatorId``)."""

    LINEAR_ADJUSTMENT_ATE = "linear.adjustment.ate"
    PROPENSITY_WEIGHTING = "propensity.weighting"
    PROPENSITY_MATCHING = "propensity.matching"
    PROPENSITY_STRATIFICATION = "propensity.stratification"
    DISTANCE_MATCHING = "distance.matching"
    AIPW = "aipw"
    GLM_ADJUSTMENT = "glm.adjustment"
    FRONTDOOR_TWO_STAGE = "frontdoor.two_stage"
    IV_WALD = "iv.wald"
    IV_2SLS = "iv.2sls"
    RD_SHARP = "rd.sharp"
    BAYESIAN_GCOMP = "bayesian.gcomp"
    TEMPORAL_LINEAR_ADJUSTMENT = "temporal.linear.adjustment"
    FUNCTIONAL_DISTRIBUTION = "functional.distribution"
    FUNCTIONAL_EFFECT = "functional.effect"
    CONDITIONAL_LINEAR_ADJUSTMENT = "conditional.linear.adjustment"
    TEMPORAL_MEDIATION = "temporal.mediation"


class Latency(StrEnum):
    """Latency tiers (wire ids match Rust ``LatencyMode``)."""

    INTERACTIVE = "interactive"
    STANDARD = "standard"
    REPORT = "report"


class Refute(StrEnum):
    """Refutation suite ids (also accept ``bool`` at call sites)."""

    FULL = "full"
    PLACEBO = "placebo"
    NONE = "none"
    CHEAP = "cheap"


__all__ = ["Estimator", "Identifier", "Latency", "Refute"]
