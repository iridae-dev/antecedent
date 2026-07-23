"""Inference-mode configuration for ``causal.analyze``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .prior_bank import ComposedPrior, PriorMapping


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
        Ignored when ``prior_from`` is provided.
    prior_from:
        Posterior artifact bytes from a previous ``result.posterior.artifact``,
        or a ``ComposedPrior`` from ``compose_external_priors``.
        Artifact hydrate is deferred until the target design is prepared.
    mapping:
        How to map an artifact into the target prior. ``None`` auto-selects:
        identical coefficient subspace when designs match (sequential Bayes),
        or ``PriorMapping.effect_functional(...)`` when designs differ and the
        artifact has an effect quantity. Never silent ``coef_i → coef_i`` across
        heterogeneous designs. Ignored when ``prior_from`` is a ``ComposedPrior``.
    backend:
        Inference backend: ``laplace`` (default), ``conjugate``, or ``hmc``.
    """

    n_draws: int = 1000
    prior_scale: float = 10.0
    prior_from: bytes | ComposedPrior | None = None
    mapping: PriorMapping | None = None
    backend: Literal["laplace", "conjugate", "hmc"] = "laplace"
    kind: Literal["bayesian"] = "bayesian"


__all__ = ["Bayesian", "Frequentist"]
