"""Declared likelihoods and priors of the randomized-design Bayesian estimators.

A posterior needs a stated likelihood and prior. These declarations go into
``Bayesian(backend="conjugate", design_model=...)``; they are part of the study's inference
identity, and a prior alone never identifies a contrast, so none is defaulted.

``BernoulliArms`` / ``GaussianArms``
    Independent arm outcomes with a Beta prior (0/1 outcomes) or a Normal-Inverse-Gamma prior (real
    outcomes), the same prior on every arm. They select the arm-contrast posteriors of a
    :class:`~antecedent.experiment.RandomizedEffect` query: the default estimator
    ``bayesian.randomized_finite_population`` answers the same target as the design-based row (the
    trial's own units), ``bayesian.randomized_superpopulation`` the arm means of the sampled
    population.
``SwitchbackModel``
    Gaussian outcomes within independent switchback sequences with a declared carryover and a
    declared within-sequence dependence (``bayesian.switchback_effect``). Unrestricted carryover and
    an unspecified dependence are refused with ``cell_not_licensed``, naming the next attempt.

``PiecewiseHazards``
    The hazard partition and Gamma prior of the piecewise-exponential survival posterior of a
    :class:`~antecedent.survival.SurvivalOutcome` query (``bayesian.piecewise_survival``): hazards
    are constant on every interval and independent a priori. The last cut is the restriction
    horizon and must equal ``tau``; the partition and the prior are part of the answer.
``QuantileGrid``
    The threshold grid and symmetric Dirichlet concentration of the randomized quantile posterior
    (``bayesian.randomized_quantile``, selected with ``estimator=``) of an
    :class:`~antecedent.AverageEffect` whose outcome functional is a quantile: within each
    arm the outcomes fall into the bins cut by the grid, with a Dirichlet prior on the bin
    probabilities, and the quantile is read by piecewise-linear inversion of the grid
    distribution function. The grid is part of the model. The treatment must be unconfounded
    (an empty adjustment set).
``DosePolynomial``
    The dose support, coefficient prior scales and residual prior of the per-group polynomial
    dose-response posterior of a fixed group-dose policy
    (:class:`~antecedent.policy.ConditionalDoseResponse` with ``policy_doses`` and
    ``reference_doses``; ``bayesian.continuous_dose_policy``).

Everything numerical, and every refusal, is computed in Rust.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "BernoulliArms",
    "DesignModel",
    "DosePolynomial",
    "GaussianArms",
    "PiecewiseHazards",
    "QuantileGrid",
    "SwitchbackModel",
]


@dataclass(frozen=True, slots=True)
class BernoulliArms:
    """Bernoulli arms with a ``Beta(alpha, beta)`` prior on every arm's success probability."""

    alpha: float
    beta: float

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "arm_outcome",
            "outcome": "bernoulli",
            "alpha": float(self.alpha),
            "beta": float(self.beta),
        }


@dataclass(frozen=True, slots=True)
class GaussianArms:
    """Gaussian arms: ``mu | s2 ~ N(mean, s2 / pseudo_units)`` and ``s2 ~ InvGamma(shape, scale)``."""

    mean: float
    pseudo_units: float
    variance_shape: float
    variance_scale: float

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "arm_outcome",
            "outcome": "gaussian",
            "mean": float(self.mean),
            "pseudo_units": float(self.pseudo_units),
            "variance_shape": float(self.variance_shape),
            "variance_scale": float(self.variance_scale),
        }


@dataclass(frozen=True, slots=True)
class SwitchbackModel:
    """Declared carryover, within-sequence dependence, period adjustment and prior.

    ``dependence`` is ``"independent"``, ``"compound_symmetric"`` or ``"autoregressive"`` (the
    last two with their correlation ``rho``); ``"unspecified"`` is refused. The coefficient
    precisions are relative to the error variance, which has an ``InvGamma(variance_shape,
    variance_scale)`` prior.
    """

    carryover: Literal["none", "first_order", "unrestricted"]
    dependence: Literal["independent", "compound_symmetric", "autoregressive", "unspecified"]
    effect_precision: float
    nuisance_precision: float
    variance_shape: float
    variance_scale: float
    rho: float | None = None
    periods: Literal["none", "fixed_effects"] = "none"

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "switchback",
            "carryover": self.carryover,
            "dependence": self.dependence,
            "rho": None if self.rho is None else float(self.rho),
            "periods": self.periods,
            "effect_precision": float(self.effect_precision),
            "nuisance_precision": float(self.nuisance_precision),
            "variance_shape": float(self.variance_shape),
            "variance_scale": float(self.variance_scale),
        }


@dataclass(frozen=True, slots=True)
class PiecewiseHazards:
    """Interval right endpoints ``cuts`` and an independent ``Gamma(prior_shape, prior_rate)`` prior.

    ``prior_shape`` is the prior event count and ``prior_rate`` the prior time at risk, in the
    unit of the durations.
    """

    cuts: tuple[float, ...]
    prior_shape: float
    prior_rate: float

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "piecewise_hazard",
            "cuts": [float(cut) for cut in self.cuts],
            "prior_shape": float(self.prior_shape),
            "prior_rate": float(self.prior_rate),
        }


@dataclass(frozen=True, slots=True)
class DosePolynomial:
    """A polynomial dose curve on ``support`` with prior scales ``coefficient_scales``.

    The dose is rescaled to ``[-1, 1]`` on ``support``; coefficient ``j`` is a priori
    ``N(0, sigma^2 s_j^2)``. The residual variance is known (``residual_variance``) or has an
    ``InvGamma(residual_shape, residual_scale)`` prior; declare exactly one of the two.
    """

    support: tuple[float, float]
    coefficient_scales: tuple[float, ...]
    residual_variance: float | None = None
    residual_shape: float | None = None
    residual_scale: float | None = None

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "dose_polynomial",
            "support": [float(self.support[0]), float(self.support[1])],
            "coefficient_scales": [float(scale) for scale in self.coefficient_scales],
            "residual_variance": None
            if self.residual_variance is None
            else float(self.residual_variance),
            "residual_shape": None if self.residual_shape is None else float(self.residual_shape),
            "residual_scale": None if self.residual_scale is None else float(self.residual_scale),
        }


@dataclass(frozen=True, slots=True)
class QuantileGrid:
    """A strictly increasing ``thresholds`` grid and a ``Dirichlet(bin_concentration)`` prior."""

    thresholds: tuple[float, ...]
    bin_concentration: float

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "quantile_grid",
            "thresholds": [float(t) for t in self.thresholds],
            "bin_concentration": float(self.bin_concentration),
        }


DesignModel = (
    BernoulliArms
    | GaussianArms
    | SwitchbackModel
    | PiecewiseHazards
    | DosePolynomial
    | QuantileGrid
)
