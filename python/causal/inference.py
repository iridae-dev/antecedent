"""Inference-mode configuration for ``causal.analyze``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Frequentist:
    """Frequentist point estimate + bootstrap SE (default)."""

    kind: Literal["frequentist"] = "frequentist"


@dataclass(frozen=True)
class Bayesian:
    """Bayesian g-computation via Laplace GLM."""

    n_draws: int = 1000
    prior_scale: float = 10.0
    kind: Literal["bayesian"] = "bayesian"


__all__ = ["Bayesian", "Frequentist"]
