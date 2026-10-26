"""Inference-mode configuration for ``antecedent.analyze``."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any, Literal

from ._measured_inference import MeasuredInference, MeasuredInferenceIdentity, MeasuredScalar
from ._native import (
    PosteriorArtifact,
    decode_posterior_artifact,
    encode_posterior_artifact,
)

if TYPE_CHECKING:
    from .design_models import DesignModel
    from .priors import ComposedPrior, PriorMapping


def _validate_profile(profile: str | None) -> None:
    if profile is not None and profile not in ("interactive", "standard", "report"):
        from .errors import CausalValueError

        raise CausalValueError("inference profile must be interactive, standard, report or None")


@dataclass(frozen=True, slots=True)
class Frequentist:
    """Frequentist point estimate with the estimator's own standard error.

    Which standard error is reported, and when it is valid, is scoped by the chosen
    estimator (see its docstring), not by this marker. ``profile`` selects the
    native scientific budget preset, including default validation depth. Explicit
    inference and validation declarations take precedence. Omission preserves
    the native resolved defaults; profiles are unrelated to resource limits.
    """

    kind: Literal["frequentist"] = "frequentist"
    bootstrap: int | None = None
    profile: Literal["interactive", "standard", "report"] | None = None

    def __post_init__(self) -> None:
        _validate_profile(self.profile)
        if self.bootstrap is not None:
            from .execution import _integer

            _integer(self.bootstrap, "bootstrap")


@dataclass(frozen=True, slots=True)
class Bayesian:
    """Bayesian g-computation (Laplace / conjugate / HMC backends).

    Parameters
    ----------
    n_draws:
        Posterior draw count.
    profile:
        Native scientific budget preset (``interactive``, ``standard`` or ``report``),
        including default validation depth. Explicit draw counts, inference and
        validation declarations take precedence. Omission preserves native defaults.
        Resource limits and observers remain in :class:`~antecedent.execution.Execution`.
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
    draw_controls:
        ``"measured"`` bounds the draws behind the interval of a tabular
        ``AverageEffect`` on a ``Dag``. ``n_draws`` must reach the backend's
        supported minimum (2000 for ``laplace`` and ``conjugate``, 3000 for
        ``hmc``) or the study is refused with
        ``reason_code="posterior_draws_below_supported"`` naming it; an interval
        whose larger endpoint Monte Carlo standard error exceeds a twentieth of its
        width (draws discounted by their tail effective sample size under ``hmc``) is
        refused with ``reason_code="posterior_draws_not_converged"``. The endpoint
        errors are reported with the posterior's inference notes. Any other route
        refuses the option rather than ignore it.
    design_model:
        The declared likelihood and prior of a randomized-design Bayesian estimator, one of
        :class:`~antecedent.design_models.BernoulliArms`,
        :class:`~antecedent.design_models.GaussianArms` or
        :class:`~antecedent.design_models.SwitchbackModel`. A randomized trial's arm
        contrasts (``bayesian.randomized_finite_population``, the same target as the design-based
        row, or ``bayesian.randomized_superpopulation``) and its switchback effect
        (``bayesian.switchback_effect``) need it with the ``conjugate`` backend; there is no
        default, because a prior alone does not identify a contrast. The results are unlicensed
        and their calibration is unmeasured. A ``ComplierEffect`` or ``TreatmentOnTreated``
        under ``Bayesian`` runs ``bayesian.complier_effect``, a Bayesian bootstrap of the Wald
        ratio that needs no ``design_model`` and refuses a first stage whose posterior reaches
        zero; a cluster-randomized interference total runs
        ``interference.cluster_bayesian_bootstrap`` likewise.
    white_innovation:
        ``(max_lag, level)`` declares that the outcome innovation of a single-step temporal
        pulse, one-step sustained effect or one-step single-horizon ``ResponseCurve`` on a
        ``TemporalDag`` is serially independent, and asks for the untempered full-likelihood
        posterior instead of the tempered (generalized) one. A curve with several horizons keeps
        the tempered posterior and refuses the declaration. The premise is yours to declare and is never inferred: a Ljung-Box test of the
        least-squares residuals at ``max_lag`` lags and ``level`` can refute it, and a pass does
        not prove it. A rejecting test refuses with
        ``white_innovation.residual_autocorrelation``, a horizon above one step with
        ``white_innovation.multi_step``, and any other route with ``option_not_applicable``,
        each naming the next attempt. The result is reported as the unlicensed estimator
        ``bayesian.temporal.white_innovation_linear`` and carries the declared premise and the
        Ljung-Box statistic.
    """

    n_draws: int | None = None
    prior_scale: float = 10.0
    prior_from: bytes | ComposedPrior | None = None
    mapping: PriorMapping | None = None
    backend: Literal["laplace", "conjugate", "hmc"] = "laplace"
    kind: Literal["bayesian"] = "bayesian"
    likelihood: Literal["gaussian", "logit", "probit", "poisson"] = "gaussian"
    draw_controls: Literal["measured"] | None = None
    white_innovation: tuple[int, float] | None = None
    design_model: DesignModel | None = None
    profile: Literal["interactive", "standard", "report"] | None = None

    def __post_init__(self) -> None:
        _validate_profile(self.profile)

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
    "MeasuredInference",
    "MeasuredInferenceIdentity",
    "MeasuredScalar",
    "Bayesian",
    "ClassPrior",
    "Frequentist",
    "PosteriorArtifact",
    "decode_posterior_artifact",
    "encode_posterior_artifact",
]
