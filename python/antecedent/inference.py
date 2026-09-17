"""Inference-mode configuration for ``antecedent.analyze``."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any, Literal

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
        MCMC publication gate (Ř ≤ 1.01, bulk/tail ESS ≥ 100 **per chain**, so
        400 total at the 4-chain default, and every chain must have moved);
        under-specified draw counts are floored in Rust.
    likelihood:
        Outcome model of the g-computation fit: ``gaussian`` (identity link,
        default), ``logit`` or ``probit`` (Bernoulli, for a 0/1 outcome), or
        ``poisson`` (log link, for a count outcome). A non-Gaussian likelihood
        is fitted for a tabular ``AverageEffect`` mean on a ``Dag`` by the
        ``laplace`` or ``hmc`` backend, under the isotropic ``prior_scale``
        (Bayesian g-computation averages the inverse link over the rows, so the
        effect stays a mean difference on the outcome scale); every other
        route, the ``conjugate`` backend and ``prior_from`` refuse it with
        ``reason_code="likelihood_not_supported"``. A Gaussian fit to an
        outcome that is binary or count-valued reports the
        ``estimate.bayesian.gaussian_likelihood_discrete_outcome`` diagnostic.
    """

    n_draws: int | None = None
    prior_scale: float = 10.0
    prior_from: bytes | ComposedPrior | None = None
    mapping: PriorMapping | None = None
    backend: Literal["laplace", "conjugate", "hmc"] = "laplace"
    kind: Literal["bayesian"] = "bayesian"
    likelihood: Literal["gaussian", "logit", "probit", "poisson"] = "gaussian"

    @property
    def n_draws_explicit(self) -> bool:
        return self.n_draws is not None


@dataclass(frozen=True, slots=True)
class ClassPrior:
    """Caller-declared mass over incomplete-temporal class members.

    This is not a graph posterior and not completion enumeration. Mechanism
    priors stay on :class:`Bayesian`. Supply either ``ordered`` masses aligned
    to ``identify(...).completion_keys`` order, or ``pairs`` of
    ``(completion_key, mass)``.
    """

    ordered: tuple[float, ...] | None = None
    pairs: tuple[tuple[int, float], ...] | None = None

    def __post_init__(self) -> None:
        if (self.ordered is None) == (self.pairs is None):
            raise ValueError("ClassPrior requires exactly one of ordered or pairs")
        masses = (
            self.ordered
            if self.ordered is not None
            else tuple(mass for _, mass in self.pairs or ())
        )
        if not masses or any(not isfinite(mass) or mass < 0.0 for mass in masses):
            raise ValueError("ClassPrior masses must be finite and nonnegative")
        if self.pairs is not None:
            keys = [key for key, _ in self.pairs]
            if len(set(keys)) != len(keys) or any(
                type(key) is not int or not 0 <= key < 2**64 for key in keys
            ):
                raise ValueError("ClassPrior keys must be unique unsigned 64-bit integers")
        if not isfinite(sum(masses)) or sum(masses) <= 0.0:
            raise ValueError("ClassPrior total mass must be strictly positive")

    @classmethod
    def from_ordered(cls, masses: list[float] | tuple[float, ...]) -> ClassPrior:
        return cls(ordered=tuple(float(mass) for mass in masses))

    @classmethod
    def from_pairs(
        cls, pairs: list[tuple[int, float]] | tuple[tuple[int, float], ...]
    ) -> ClassPrior:
        return cls(pairs=tuple((key, float(mass)) for key, mass in pairs))


def _class_prior_kwargs(prior: ClassPrior | None) -> dict[str, Any]:
    if prior is None:
        return {}
    if prior.ordered is not None:
        return {"class_prior_ordered": list(prior.ordered)}
    return {"class_prior_pairs": list(prior.pairs or ())}


def _max_completions_kwargs(max_completions: int | None) -> dict[str, Any]:
    return {} if max_completions is None else {"max_completions": int(max_completions)}


__all__ = [
    "Bayesian",
    "ClassPrior",
    "Frequentist",
    "PosteriorArtifact",
    "decode_posterior_artifact",
    "encode_posterior_artifact",
]
