"""Inference-mode configuration for ``antecedent.analyze``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ._native import (
    PosteriorArtifact,
    decode_posterior_artifact,
    encode_posterior_artifact,
)

if TYPE_CHECKING:
    from .priors import ComposedPrior, PriorMapping


@dataclass(frozen=True, slots=True)
class Frequentist:
    """Frequentist point estimate + bootstrap SE (default)."""

    kind: Literal["frequentist"] = "frequentist"


@dataclass(frozen=True, slots=True)
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
        The ``hmc`` backend needs several thousand draws to clear the native
        MCMC publication gate (Ř ≤ 1.01, bulk/tail ESS ≥ 100); under-specified
        draw counts are floored in Rust.
    """

    n_draws: int = 1000
    prior_scale: float = 10.0
    prior_from: bytes | ComposedPrior | None = None
    mapping: PriorMapping | None = None
    backend: Literal["laplace", "conjugate", "hmc"] = "laplace"
    kind: Literal["bayesian"] = "bayesian"


__all__ = [
    "Bayesian",
    "Frequentist",
    "PosteriorArtifact",
    "decode_posterior_artifact",
    "encode_posterior_artifact",
]
