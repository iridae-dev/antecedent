"""Inference declarations: :class:`Frequentist`, :class:`Bayesian` and :class:`ClassPrior`.

Pass one of the first two as ``inference=`` to :func:`antecedent.analyze` or
:func:`antecedent.prepare`. They say how uncertainty is quantified; the estimand is fixed by
the query and its identification, and the estimator by ``estimator=``. Omitting ``inference``
means ``Frequentist()``.
"""

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
    from .predictive_assessment import ReplicateBounds
    from .priors import (
        BetaHyperparameters,
        ComposedCountPrior,
        ComposedPrior,
        GammaHyperparameters,
        PriorMapping,
    )


def _validate_profile(profile: str | None) -> None:
    if profile is not None and profile not in ("interactive", "standard", "report"):
        from .errors import CausalValueError

        raise CausalValueError("inference profile must be interactive, standard, report or None")


@dataclass(frozen=True, slots=True)
class Frequentist:
    """Frequentist inference: a point estimate with the estimator's own standard error.

    Which standard error is reported (analytic, influence-function or bootstrap), and the
    conditions under which it is valid, are properties of the chosen estimator; see its
    documentation in :mod:`antecedent.estimators`. Each interval carries its calibration status
    on the result.

    Fields:

    * ``bootstrap`` -- number of bootstrap replicates, a nonnegative integer. Default ``None``
      leaves the count to the engine's defaults for the route and ``profile``. Routes that do not
      support a bootstrap refuse a positive count.
    * ``profile`` -- scientific budget preset: ``"interactive"``, ``"standard"`` or ``"report"``.
      It sets default replicate counts and default validation depth only for values you leave
      unset; an explicit ``bootstrap`` or refutation declaration takes precedence. Default
      ``None`` keeps the engine's resolved defaults. Profiles are unrelated to resource limits,
      which belong to :class:`~antecedent.Execution`.
    * ``kind`` -- always ``"frequentist"``.

    Raises:
        CausalValueError: when ``profile`` is not one of the three names or ``None``, or
            ``bootstrap`` is negative.
        CausalTypeError: when ``bootstrap`` is not an integer.

    Example::

        antecedent.analyze(data, graph=g, query=q, inference=antecedent.Frequentist(bootstrap=500))
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
    """Bayesian inference: a posterior for the identified estimand by Bayesian g-computation.

    The posterior is fitted by one of three backends (Laplace approximation, conjugate update or
    Hamiltonian Monte Carlo) and summarized as a point estimate with a credible interval. Each
    interval carries its calibration status on the result. The estimand is fixed by the query
    and its identification; ``Bayesian`` changes only how its uncertainty is quantified.

    Fields:

    * ``n_draws`` -- posterior draw count. Default ``None`` leaves it to the engine's default for
      the backend and ``profile``.
    * ``profile`` -- scientific budget preset (``"interactive"``, ``"standard"`` or
      ``"report"``), including default validation depth. Explicit draw counts, inference and
      validation declarations take precedence. Default ``None`` keeps the engine's defaults.
      Resource limits and observers remain in :class:`~antecedent.execution.Execution`.
    * ``prior_scale`` -- scale of the isotropic Gaussian coefficient prior used when
      ``prior_from`` is unset (default ``10.0``). Ignored when ``prior_from`` is provided.
    * ``prior_from`` -- an earlier posterior to use as the prior: the artifact returned by
      ``result.export(quantity="posterior")`` (or its bytes), or a
      :class:`~antecedent.priors.ComposedPrior` from
      :func:`~antecedent.priors.compose_external_priors`. The artifact is hydrated only once the
      target design is prepared. A composed prior with named coordinates
      (``ExternalPriorSourceSpec.coefficient_names``) maps by name onto the design's coefficient
      columns, including a kernel- or row-weighted target composition and a prior composed at a
      power-prior discount selected from the data
      (:func:`~antecedent.priors.select_alpha_prior_linear`). A
      :class:`~antecedent.priors.BetaHyperparameters` / :class:`~antecedent.priors.GammaHyperparameters`
      prior (or a :class:`~antecedent.priors.ComposedCountPrior`) is the prior of a conditional
      or exceedance :class:`~antecedent.Association` under ``Bayesian``: each cell's posterior is
      the closed-form Beta-Binomial or Gamma-Poisson one. Default ``None``.
    * ``mapping`` -- a :class:`~antecedent.priors.PriorMapping` saying how the artifact maps into
      the target prior. ``None`` auto-selects: the identical coefficient subspace when designs
      match (sequential Bayes), or ``PriorMapping.effect_functional(...)`` when designs differ
      and the artifact has an effect quantity. Coefficients are never matched by position across
      heterogeneous designs. Ignored when ``prior_from`` is a ``ComposedPrior``.
    * ``backend`` -- ``"laplace"`` (default), ``"conjugate"`` or ``"hmc"``. The ``hmc`` backend
      needs several thousand draws to clear the native MCMC publication gate (R-hat at most
      1.01, bulk and tail ESS at least 100 **per chain**, so 400 total at the 4-chain default,
      and every chain must have moved); under-specified draw counts are raised to a floor by the
      engine.
    * ``likelihood`` -- outcome model of the g-computation fit: ``"gaussian"`` (identity link,
      default), ``"logit"`` or ``"probit"`` (Bernoulli, for a 0/1 outcome), or ``"poisson"``
      (log link, for a count outcome). A non-Gaussian likelihood is fitted for a tabular
      ``AverageEffect`` mean on a ``Dag`` by the ``laplace`` or ``hmc`` backend, under the
      isotropic ``prior_scale`` (Bayesian g-computation averages the inverse link over the rows,
      so the effect stays a mean difference on the outcome scale); every other route, the
      ``conjugate`` backend and ``prior_from`` refuse it with
      ``reason_code="likelihood_not_supported"``. A Gaussian fit to an outcome that is binary or
      count-valued reports the ``estimate.bayesian.gaussian_likelihood_discrete_outcome``
      diagnostic.
    * ``draw_controls`` -- ``"measured"`` bounds the draws behind the interval of a tabular
      ``AverageEffect`` on a ``Dag``. ``n_draws`` must reach the backend's supported minimum
      (2000 for ``laplace`` and ``conjugate``, 3000 for ``hmc``) or the study is refused with
      ``reason_code="posterior_draws_below_supported"`` naming it; an interval whose larger
      endpoint Monte Carlo standard error exceeds a twentieth of its width (draws discounted by
      their tail effective sample size under ``hmc``) is refused with
      ``reason_code="posterior_draws_not_converged"``. The endpoint errors are reported with the
      posterior's inference notes. Any other route refuses the option rather than ignore it.
      Default ``None``.
    * ``design_model`` -- the declared likelihood and prior of a randomized-design Bayesian
      estimator, one of :class:`~antecedent.design_models.BernoulliArms`,
      :class:`~antecedent.design_models.GaussianArms` or
      :class:`~antecedent.design_models.SwitchbackModel`. A randomized trial's arm contrasts
      (``bayesian.randomized_finite_population``, the same target as the design-based row, or
      ``bayesian.randomized_superpopulation``) and its switchback effect
      (``bayesian.switchback_effect``) need it with the ``conjugate`` backend; there is no
      default, because a prior alone does not identify a contrast. The results are unsupported
      and their calibration is not_measured. A ``ComplierEffect`` or ``TreatmentOnTreated`` under
      ``Bayesian`` runs ``bayesian.complier_effect``, a Bayesian bootstrap of the Wald ratio that
      needs no ``design_model`` and refuses a first stage whose posterior reaches zero; a
      cluster-randomized interference total runs ``interference.cluster_bayesian_bootstrap``
      likewise.
    * ``predictive_replicates`` --
      :class:`~antecedent.predictive_assessment.ReplicateBounds` that make the fit retain its
      posterior predictive replicates (``n_sims`` draws over at most ``max_rows`` design rows),
      admitted against the Execution memory pool and read back with
      :func:`antecedent.predictive_assessment.posterior_replicates`. Only the tabular
      ``AverageEffect`` mean on a fixed ``Dag`` under Bayesian g-computation builds them; any
      other route refuses the declaration (``option_not_applicable``), and a design with more
      rows than ``max_rows`` is refused (``invalid_argument``). Default ``None``.
    * ``white_innovation`` -- ``(max_lag, level)`` declares that the outcome innovation of a
      single-step temporal pulse, one-step sustained effect or one-step single-horizon
      ``ResponseCurve`` on a ``TemporalDag`` is serially independent, and asks for the
      untempered full-likelihood posterior instead of the tempered (generalized) one. A curve
      with several horizons keeps the tempered posterior and refuses the declaration. The
      premise is yours to declare and is never inferred: a Ljung-Box test of the least-squares
      residuals at ``max_lag`` lags and ``level`` can refute it, and a pass does not prove it. A
      rejecting test refuses with ``white_innovation.residual_autocorrelation``, a horizon above
      one step with ``white_innovation.multi_step``, and any other route with
      ``option_not_applicable``, each naming the next attempt. The result is reported as the
      unsupported estimator ``bayesian.temporal.white_innovation_linear`` and carries the
      declared premise and the Ljung-Box statistic. Default ``None``.
    * ``kind`` -- always ``"bayesian"``.

    Raises:
        CausalValueError: when ``profile`` is not one of the three names or ``None``.
        CausalTypeError: when ``predictive_replicates`` is not a ``ReplicateBounds``.

    Example::

        first = antecedent.analyze(data, graph=g, query=q, inference=antecedent.Bayesian())
        posterior = first.export(quantity="posterior")
        second = antecedent.analyze(
            new_data, graph=g, query=q, inference=antecedent.Bayesian(prior_from=posterior)
        )
    """

    n_draws: int | None = None
    prior_scale: float = 10.0
    prior_from: (
        bytes
        | ComposedPrior
        | BetaHyperparameters
        | GammaHyperparameters
        | ComposedCountPrior
        | None
    ) = None
    mapping: PriorMapping | None = None
    backend: Literal["laplace", "conjugate", "hmc"] = "laplace"
    kind: Literal["bayesian"] = "bayesian"
    likelihood: Literal["gaussian", "logit", "probit", "poisson"] = "gaussian"
    draw_controls: Literal["measured"] | None = None
    white_innovation: tuple[int, float] | None = None
    design_model: DesignModel | None = None
    profile: Literal["interactive", "standard", "report"] | None = None
    predictive_replicates: ReplicateBounds | None = None

    def __post_init__(self) -> None:
        _validate_profile(self.profile)
        if self.predictive_replicates is not None:
            from .predictive_assessment import ReplicateBounds  # noqa: PLC0415

            if not isinstance(self.predictive_replicates, ReplicateBounds):
                from .errors import CausalTypeError  # noqa: PLC0415

                raise CausalTypeError("predictive_replicates must be a ReplicateBounds")

    @property
    def n_draws_explicit(self) -> bool:
        """``True`` when ``n_draws`` was given, ``False`` when it is left to the defaults."""
        return self.n_draws is not None


@dataclass(frozen=True, slots=True)
class ClassPrior:
    """Prior mass over the member graphs of an incomplete temporal graph class.

    When the structure is a temporal CPDAG or PAG, several completed graphs are compatible with
    it. A ``ClassPrior`` states how much weight each completion receives when their answers are
    combined; pass it as ``class_prior=`` together with :class:`Bayesian` inference. It is your
    declaration, not a graph posterior learned from data, and it does not enumerate completions.
    Priors on mechanism parameters stay on :class:`Bayesian`.

    Give exactly one of:

    * ``ordered`` -- masses aligned with the order of ``identify(...).completion_keys``;
    * ``pairs`` -- ``(completion_key, mass)`` pairs with unique unsigned 64-bit integer keys.

    Masses must be finite and nonnegative with a strictly positive total. Use
    :meth:`from_ordered` or :meth:`from_pairs` to build one from lists.

    Raises:
        ValueError: when both or neither of ``ordered`` and ``pairs`` are given, a mass is
            negative or not finite, the total mass is not positive, or keys repeat or are not
            unsigned 64-bit integers.
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
        """Build a prior from masses in ``identify(...).completion_keys`` order."""
        return cls(ordered=tuple(float(mass) for mass in masses))

    @classmethod
    def from_pairs(
        cls, pairs: list[tuple[int, float]] | tuple[tuple[int, float], ...]
    ) -> ClassPrior:
        """Build a prior from ``(completion_key, mass)`` pairs."""
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
