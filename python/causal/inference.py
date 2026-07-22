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
    """Bayesian g-computation (Laplace / conjugate / HMC backends).

    Parameters
    ----------
    n_draws:
        Posterior draw count.
    prior_scale:
        Isotropic Gaussian coefficient prior scale when ``prior_from`` is unset.
        Ignored when ``prior_from`` is provided (sequential Bayes).
    prior_from:
        Posterior artifact bytes from a previous ``result.posterior.artifact``.
        Hydrates a Gaussian coefficient prior (same design / ``ncols`` only).
    backend:
        Inference backend: ``laplace`` (default), ``conjugate``, or ``hmc``.
    """

    n_draws: int = 1000
    prior_scale: float = 10.0
    prior_from: bytes | None = None
    backend: Literal["laplace", "conjugate", "hmc"] = "laplace"
    kind: Literal["bayesian"] = "bayesian"


__all__ = ["Bayesian", "Frequentist"]
