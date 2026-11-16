"""Exposure-error corrections: misclassified exposures, soft exposures, evidence-dependent detection
and repeat codings as replicates.

Four corrections that :mod:`antecedent.measurement_models` does not cover. Each takes its
declarations as typed values and refuses, with a :class:`~antecedent.measurement_models
.MeasurementModelRefusal`, what it cannot identify::

    from antecedent import measurement_exposure as mx

    # A binary exposure misclassified differently among cases and controls, with the rates
    # estimated from one validation sample per outcome level.
    fit = mx.correct_exposure_misclassification(
        exposed_cases=480, cases=1000, exposed_controls=180, controls=1000,
        error=mx.ExposureError.by_outcome(
            cases=mx.ValidationCounts(90, 10, 80, 20),
            controls=mx.ValidationCounts(70, 30, 190, 10),
        ),
    )
    fit.odds_ratio          # the corrected exposure odds ratio
    fit.interval            # Wald interval of the log odds ratio, calibration not_measured

* **Exposure misclassification** (:func:`correct_exposure_misclassification`) corrects the
  exposure odds ratio by outcome level for a sensitivity and specificity that may differ between
  outcome levels (differential error), declared or estimated from validation samples whose
  sampling variance is propagated. A Youden index at or below the floor in either outcome level
  is not identified and is refused.
* **Soft exposures** (:func:`soft_exposure_effect`) take a per-unit probability that the
  exposure is present. Posterior-mean substitution reports a variance term for the exposure
  uncertainty; multiple imputation draws the exposure given the outcome and combines the fits
  with Rubin's rules. :func:`soft_exposure_reliability` compares the stated probabilities with
  repeat codings in bins; it is a report and never gates anything.
* **Evidence-dependent detection** (:func:`detection_dependent_prevalence`) corrects a prevalence
  for a detection probability that is a declared function of a covariate such as a count: a fully
  declared function (:class:`LogisticDetection`, :class:`TableDetection`) point-identifies it, a
  partially declared monotone function (:class:`MonotoneDetection`) gives sharp bounds.
* **Repeat codings as replicates** (:func:`repeat_codings_latent_class`) fits a latent-class
  model for binary and ordinal facets. Conditional independence of the codings given the latent
  level is an assumption. Three codings identify a binary facet; two identify it only with
  exchangeable codings and a declared law of the lowest level; fewer are refused.

Every standard error is an uncalibrated delta-method (or Rubin) estimate; intervals carry a
``not_measured`` calibration slot.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from . import _native
from ._refusal import raise_refusal
from .errors import CausalTypeError, CausalValueError
from .execution import Execution
from .measurement_models import (
    BinaryRates,
    MeasurementInterval,
    MeasurementModelRefusal,
    VarianceComponent,
    _columns,
    _count,
    _dumps,
    _number,
    _vector,
)

__all__ = [
    "DetectionPrevalence",
    "ExposureCorrection",
    "ExposureError",
    "LogisticDetection",
    "MonotoneDetection",
    "RepeatCodingFit",
    "ReliabilityBin",
    "SoftExposureEffect",
    "SoftExposureReliability",
    "TableDetection",
    "ValidationCounts",
    "correct_exposure_misclassification",
    "detection_dependent_prevalence",
    "repeat_codings_latent_class",
    "soft_exposure_effect",
    "soft_exposure_reliability",
]

_NOT_MEASURED: Mapping[str, Any] = {"status": "not_measured"}


def _call(function: Any, declaration: Mapping[str, Any]) -> dict[str, Any]:
    result, refusal = function(_dumps(declaration))
    raise_refusal(refusal, MeasurementModelRefusal)
    parsed: dict[str, Any] = json.loads(result)
    return parsed


def _level(level: float) -> float:
    value = _number(level, "level")
    if not 0.0 < value < 1.0:
        raise CausalValueError("level must lie strictly between 0 and 1")
    return value


def _estimate(
    wire: Mapping[str, Any],
) -> tuple[float, float | None, MeasurementInterval, tuple[VarianceComponent, ...]]:
    interval = wire["interval"]
    return (
        float(wire["point"]),
        None if wire["standard_error"] is None else float(wire["standard_error"]),
        MeasurementInterval(
            float(interval["lower"]),
            float(interval["upper"]),
            float(interval["level"]),
            dict(_NOT_MEASURED),
        ),
        tuple(
            VarianceComponent(c["name"], float(c["variance"])) for c in wire["variance_components"]
        ),
    )


# --------------------------------------------------------------------------- exposure error


@dataclass(frozen=True, slots=True)
class ValidationCounts:
    """A validation sample against a reference standard.

    ``true_positive + false_negative`` units are truly exposed and ``true_negative +
    false_positive`` truly unexposed. The binomial sampling variance of the estimated
    sensitivity and specificity is propagated into the corrected result.
    """

    true_positive: int
    false_negative: int
    true_negative: int
    false_positive: int

    def to_wire(self) -> list[int]:
        """The counts in the order the native layer reads them."""
        return [
            _count(self.true_positive, "true_positive"),
            _count(self.false_negative, "false_negative"),
            _count(self.true_negative, "true_negative"),
            _count(self.false_positive, "false_positive"),
        ]


@dataclass(frozen=True, slots=True)
class ExposureError:
    """How a binary exposure's error depends on the outcome.

    Build it with :meth:`shared` (one sensitivity and specificity for both outcome levels) or
    :meth:`by_outcome` (a pair per outcome level). Each pair is declared as
    :class:`~antecedent.measurement_models.BinaryRates` or estimated from
    :class:`ValidationCounts`; both outcome levels use the same kind.
    """

    shared_error: BinaryRates | ValidationCounts | None = None
    cases: BinaryRates | ValidationCounts | None = None
    controls: BinaryRates | ValidationCounts | None = None

    @classmethod
    def shared(cls, error: BinaryRates | ValidationCounts) -> ExposureError:
        """Error that does not depend on the outcome."""
        return cls(shared_error=error)

    @classmethod
    def by_outcome(
        cls,
        *,
        cases: BinaryRates | ValidationCounts,
        controls: BinaryRates | ValidationCounts,
    ) -> ExposureError:
        """Error that differs between outcome levels."""
        return cls(cases=cases, controls=controls)

    def __post_init__(self) -> None:
        if self.shared_error is not None:
            if self.cases is not None or self.controls is not None:
                raise CausalValueError("error is either shared or by outcome, not both")
            return
        if self.cases is None or self.controls is None:
            raise CausalValueError("by-outcome error needs rates for cases and for controls")
        if isinstance(self.cases, ValidationCounts) != isinstance(self.controls, ValidationCounts):
            raise CausalValueError(
                "declare both outcome levels' rates or estimate both from validation counts"
            )

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        if self.shared_error is not None:
            if isinstance(self.shared_error, ValidationCounts):
                return {"kind": "shared_validation", "counts": self.shared_error.to_wire()}
            return {"kind": "shared", "rates": _rates_wire(self.shared_error)}
        assert self.cases is not None and self.controls is not None
        if isinstance(self.cases, ValidationCounts):
            assert isinstance(self.controls, ValidationCounts)
            return {
                "kind": "by_outcome_validation",
                "cases": self.cases.to_wire(),
                "controls": self.controls.to_wire(),
            }
        assert isinstance(self.controls, BinaryRates)
        return {
            "kind": "by_outcome",
            "cases": _rates_wire(self.cases),
            "controls": _rates_wire(self.controls),
        }


def _rates_wire(rates: BinaryRates | ValidationCounts) -> dict[str, Any]:
    if not isinstance(rates, BinaryRates):
        raise CausalTypeError("declared rates must be BinaryRates")
    wire = rates.to_wire()
    wire.pop("arm_id", None)
    return wire


@dataclass(frozen=True, slots=True)
class ExposureCorrection:
    """A corrected exposure association.

    ``log_odds_ratio`` and ``odds_ratio`` are the exposure odds ratio between cases and controls
    after correcting the recorded exposure; ``naive_log_odds_ratio`` is the recorded one
    (``None`` when a recorded cell is empty). ``standard_error`` is the delta-method standard
    error of the log odds ratio including the validation uncertainty of the rates; it is
    uncalibrated and ``interval`` carries a ``not_measured`` calibration slot.
    """

    prevalence_cases: float
    prevalence_controls: float
    observed_prevalence_cases: float
    observed_prevalence_controls: float
    log_odds_ratio: float
    odds_ratio: float
    naive_log_odds_ratio: float | None
    standard_error: float
    variance_components: tuple[VarianceComponent, ...]
    youden_cases: float
    youden_controls: float
    differential: bool
    parameter_uncertainty: str
    interval: MeasurementInterval
    data_identity: str


def correct_exposure_misclassification(
    *,
    exposed_cases: int,
    cases: int,
    exposed_controls: int,
    controls: int,
    error: ExposureError,
    min_youden: float = 0.1,
    level: float = 0.95,
) -> ExposureCorrection:
    """Correct a misclassified binary exposure, including error that differs by outcome.

    ``exposed_cases`` of ``cases`` units with the outcome and ``exposed_controls`` of ``controls``
    units without it are recorded as exposed. Refuses (``measurement_not_identified``) when the
    Youden index ``sensitivity + specificity - 1`` is at or below ``min_youden`` in either outcome
    level, and when a corrected exposure share falls outside ``(0, 1)``.
    """
    wire = _call(
        _native.measurement_exposure_misclassification,
        {
            "counts": {
                "exposed_cases": _count(exposed_cases, "exposed_cases"),
                "cases": _count(cases, "cases"),
                "exposed_controls": _count(exposed_controls, "exposed_controls"),
                "controls": _count(controls, "controls"),
            },
            "structure": error.to_wire(),
            "min_youden": _number(min_youden, "min_youden"),
            "level": _level(level),
        },
    )
    _, se, interval, components = _estimate(wire["estimate"])
    return ExposureCorrection(
        prevalence_cases=wire["prevalence_cases"],
        prevalence_controls=wire["prevalence_controls"],
        observed_prevalence_cases=wire["observed_prevalence_cases"],
        observed_prevalence_controls=wire["observed_prevalence_controls"],
        log_odds_ratio=wire["log_odds_ratio"],
        odds_ratio=wire["odds_ratio"],
        naive_log_odds_ratio=wire["naive_log_odds_ratio"],
        standard_error=float(se) if se is not None else float("nan"),
        variance_components=components,
        youden_cases=wire["youden_cases"],
        youden_controls=wire["youden_controls"],
        differential=wire["differential"],
        parameter_uncertainty=wire["estimate"]["parameter_uncertainty"],
        interval=interval,
        data_identity=wire["estimate"]["data_identity"],
    )


# --------------------------------------------------------------------------- soft exposures


@dataclass(frozen=True, slots=True)
class SoftExposureEffect:
    """The coefficient of a per-unit exposure probability in a linear outcome model.

    ``coefficients`` start with the intercept, then the exposure, then the covariates.
    ``variance_components`` are ``sampling_given_exposure`` and ``exposure_uncertainty`` for
    posterior-mean substitution and ``within_imputation`` and ``between_imputation`` for
    multiple imputation.
    """

    method: str
    coefficient: float
    standard_error: float
    coefficients: tuple[float, ...]
    covariance: np.ndarray
    variance_components: tuple[VarianceComponent, ...]
    interval: MeasurementInterval
    parameter_uncertainty: str
    imputations: int | None
    draw_coefficients: tuple[float, ...] | None
    mean_posterior_variance: float | None
    data_identity: str


def _execution_seed(execution: Execution | None) -> int:
    if execution is not None and not isinstance(execution, Execution):
        raise CausalTypeError("execution must be an Execution declaration")
    return (Execution() if execution is None else execution).seed


def soft_exposure_effect(
    outcome: object,
    probability: object,
    covariates: object = None,
    *,
    method: Literal["posterior_mean", "multiple_imputation", "supplied_draws"] = "posterior_mean",
    draws: object = None,
    draw_count: int = 30,
    execution: Execution | None = None,
    level: float = 0.95,
) -> SoftExposureEffect:
    """Fit a linear outcome model on a per-unit probability that the exposure is present.

    ``probability[i]`` is the calibrated probability that unit ``i``'s exposure is present given
    what the analysis conditions on. ``method="posterior_mean"`` regresses on the probabilities
    (valid when they are calibrated given the covariates) and reports the exposure-uncertainty
    variance term; ``"multiple_imputation"`` draws ``draw_count`` outcome-conditioned exposures
    under a Gaussian outcome model and combines the fits by Rubin's rules (the draws are proper
    only when that model is right); ``"supplied_draws"`` combines your own outcome-conditioned
    ``(draw_count, n)`` 0/1 draws by Rubin's rules.
    """
    y = _vector(outcome, "outcome")
    q = _vector(probability, "probability")
    columns, covariate_count = _columns(covariates, len(y), "covariates")
    if method == "posterior_mean":
        method_wire: dict[str, Any] = {"kind": "posterior_mean"}
    elif method == "multiple_imputation":
        method_wire = {
            "kind": "multiple_imputation",
            "draw_count": _count(draw_count, "draw_count", minimum=2),
            "seed": _execution_seed(execution),
        }
    elif method == "supplied_draws":
        array = np.asarray(draws, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(y):
            raise CausalValueError("draws must be a (draw_count, n) array of 0/1 exposures")
        method_wire = {
            "kind": "supplied_draws",
            "draw_count": int(array.shape[0]),
            "draws": [float(v) for v in array.reshape(-1)],
        }
    else:
        raise CausalValueError(
            "method must be 'posterior_mean', 'multiple_imputation' or 'supplied_draws'"
        )
    wire = _call(
        _native.measurement_soft_exposure,
        {
            "outcome": y,
            "probability": q,
            "covariates": columns,
            "covariate_count": covariate_count,
            "method": method_wire,
            "level": _level(level),
        },
    )
    point, se, interval, components = _estimate(wire["estimate"])
    dim = covariate_count + 2
    imputations = wire["imputations"]
    substitution = wire["substitution"]
    return SoftExposureEffect(
        method=method,
        coefficient=point,
        standard_error=float(se) if se is not None else float("nan"),
        coefficients=tuple(wire["coefficients"]),
        covariance=np.asarray(wire["covariance"], dtype=np.float64).reshape(dim, dim),
        variance_components=components,
        interval=interval,
        parameter_uncertainty=wire["estimate"]["parameter_uncertainty"],
        imputations=None if imputations is None else int(imputations["draws"]),
        draw_coefficients=None if imputations is None else tuple(imputations["draw_coefficients"]),
        mean_posterior_variance=None
        if substitution is None
        else substitution["mean_posterior_variance"],
        data_identity=wire["estimate"]["data_identity"],
    )


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One bin of the soft-exposure reliability report."""

    lower: float
    upper: float
    units: int
    mean_stated: float
    mean_expected: float
    mean_observed: float
    gap: float
    standard_error: float | None


@dataclass(frozen=True, slots=True)
class SoftExposureReliability:
    """Stated exposure probabilities against repeat codings, in bins of the stated probability.

    A report: ``gates`` is always ``False`` and no threshold is applied. A gap is the mean repeat
    coding share minus the share the stated probabilities imply.
    """

    bins: tuple[ReliabilityBin, ...]
    units: int
    max_abs_gap: float
    mean_abs_gap: float
    chi_square: float
    chi_square_bins: int
    gates: bool


def soft_exposure_reliability(
    probability: object,
    repeat_codings: Sequence[Sequence[int]],
    *,
    bins: int = 10,
    reference_sensitivity: float = 1.0,
    reference_specificity: float = 1.0,
) -> SoftExposureReliability:
    """Check stated exposure probabilities against repeat codings of the same units.

    ``repeat_codings[i]`` holds unit ``i``'s 0/1 codings (units may have different numbers).
    Under calibration and conditionally independent codings, a coding of a unit with stated
    probability ``q`` is 1 with probability ``se * q + (1 - sp) * (1 - q)``; the reference
    sensitivity and specificity default to exact codings. Bins are equal-width in ``q``.
    """
    q = _vector(probability, "probability")
    if len(repeat_codings) != len(q):
        raise CausalValueError("repeat_codings needs one entry per unit")
    flat: list[int] = []
    sizes: list[int] = []
    for row in repeat_codings:
        values = [int(v) for v in row]
        if any(v not in (0, 1) for v in values):
            raise CausalValueError("repeat codings must be 0 or 1")
        flat.extend(values)
        sizes.append(len(values))
    declaration: dict[str, Any] = {
        "probability": q,
        "codings": flat,
        "sizes": sizes,
        "bins": _count(bins, "bins", minimum=1),
    }
    if reference_sensitivity != 1.0 or reference_specificity != 1.0:
        declaration["reference"] = {
            "sensitivity": _number(reference_sensitivity, "reference_sensitivity"),
            "specificity": _number(reference_specificity, "reference_specificity"),
        }
    wire = _call(_native.measurement_soft_exposure_reliability, declaration)
    return SoftExposureReliability(
        bins=tuple(
            ReliabilityBin(
                lower=b["lower"],
                upper=b["upper"],
                units=b["units"],
                mean_stated=b["mean_stated"],
                mean_expected=b["mean_expected"],
                mean_observed=b["mean_observed"],
                gap=b["gap"],
                standard_error=b["standard_error"],
            )
            for b in wire["bins"]
        ),
        units=wire["units"],
        max_abs_gap=wire["max_abs_gap"],
        mean_abs_gap=wire["mean_abs_gap"],
        chi_square=wire["chi_square"],
        chi_square_bins=wire["chi_square_bins"],
        gates=bool(wire["gates"]),
    )


# --------------------------------------------------------------------------- detection


@dataclass(frozen=True, slots=True)
class LogisticDetection:
    """Detection probability ``expit(intercept + slope * covariate)``, fully declared.

    The variances and covariance are those of the parameter estimates (all zero when declared
    exact); they are propagated into the prevalence's variance.
    """

    intercept: float
    slope: float
    intercept_variance: float = 0.0
    covariance: float = 0.0
    slope_variance: float = 0.0

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "kind": "logistic",
            "intercept": _number(self.intercept, "intercept"),
            "slope": _number(self.slope, "slope"),
            "intercept_variance": _number(self.intercept_variance, "intercept_variance"),
            "covariance": _number(self.covariance, "covariance"),
            "slope_variance": _number(self.slope_variance, "slope_variance"),
        }


@dataclass(frozen=True, slots=True)
class TableDetection:
    """Detection probability interpolated linearly between declared ``(knot, value)`` points."""

    knots: Sequence[float]
    values: Sequence[float]

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "kind": "table",
            "knots": _vector(self.knots, "knots"),
            "values": _vector(self.values, "values"),
        }


@dataclass(frozen=True, slots=True)
class MonotoneDetection:
    """A partially declared monotone detection function, bounded at anchor covariate values.

    ``anchors`` are ``(covariate, lower, upper)`` triples with increasing covariates; the
    prevalence is then bounded, not point-identified.
    """

    direction: Literal["nondecreasing", "nonincreasing"]
    anchors: Sequence[tuple[float, float, float]]

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        if self.direction not in ("nondecreasing", "nonincreasing"):
            raise CausalValueError("direction must be 'nondecreasing' or 'nonincreasing'")
        return {
            "kind": "envelope",
            "direction": self.direction,
            "anchors": [[_number(v, "anchor") for v in anchor] for anchor in self.anchors],
        }


@dataclass(frozen=True, slots=True)
class DetectionPrevalence:
    """A prevalence corrected for covariate-dependent detection.

    ``kind`` is ``"point"`` for a fully declared function (``prevalence``, ``standard_error`` and
    ``interval`` are set) and ``"bounds"`` for a partially declared one (``lower`` and ``upper``
    are the sharp bounds, with their sampling standard errors).
    """

    kind: Literal["point", "bounds"]
    units: int
    observed_rate: float
    prevalence: float | None
    standard_error: float | None
    interval: MeasurementInterval | None
    variance_components: tuple[VarianceComponent, ...]
    lower: float | None
    upper: float | None
    lower_standard_error: float | None
    upper_standard_error: float | None
    outside_unit_interval: bool


def detection_dependent_prevalence(
    detected: object,
    covariate: object,
    detection: LogisticDetection | TableDetection | MonotoneDetection,
    *,
    false_positive_rate: float = 0.0,
    false_positive_variance: float = 0.0,
    min_net_detection: float = 0.1,
    level: float = 0.95,
) -> DetectionPrevalence:
    """Correct a prevalence recorded with detection that depends on a covariate.

    ``detected[i]`` records whether unit ``i`` was found to have the condition and ``covariate[i]``
    is what drives detection (for example the number of records on the unit). The latent presence
    must not depend on the covariate. Refuses (``measurement_not_identified``) when the mean
    detection net of false positives is at or below ``min_net_detection``.
    """
    flags = np.asarray(detected, dtype=np.bool_)
    if flags.ndim != 1:
        raise CausalValueError("detected must be one-dimensional")
    x = _vector(covariate, "covariate")
    if len(x) != len(flags):
        raise CausalValueError("detected and covariate differ in length")
    if not isinstance(detection, (LogisticDetection, TableDetection, MonotoneDetection)):
        raise CausalTypeError(
            "detection must be a LogisticDetection, TableDetection or MonotoneDetection"
        )
    wire = _call(
        _native.measurement_detection_dependent,
        {
            "detected": [bool(v) for v in flags],
            "covariate": x,
            "declaration": detection.to_wire(),
            "false_positive": {
                "rate": _number(false_positive_rate, "false_positive_rate"),
                "variance": _number(false_positive_variance, "false_positive_variance"),
            },
            "min_net_detection": _number(min_net_detection, "min_net_detection"),
            "level": _level(level),
        },
    )
    if wire["kind"] == "point":
        point, se, interval, components = _estimate(wire["estimate"])
        return DetectionPrevalence(
            kind="point",
            units=wire["units"],
            observed_rate=wire["observed_rate"],
            prevalence=point,
            standard_error=se,
            interval=interval,
            variance_components=components,
            lower=None,
            upper=None,
            lower_standard_error=None,
            upper_standard_error=None,
            outside_unit_interval=bool(wire["outside_unit_interval"]),
        )
    return DetectionPrevalence(
        kind="bounds",
        units=wire["units"],
        observed_rate=wire["observed_rate"],
        prevalence=None,
        standard_error=None,
        interval=None,
        variance_components=(),
        lower=wire["lower"],
        upper=wire["upper"],
        lower_standard_error=wire["lower_standard_error"],
        upper_standard_error=wire["upper_standard_error"],
        outside_unit_interval=False,
    )


# --------------------------------------------------------------------------- repeat codings


@dataclass(frozen=True, slots=True)
class RepeatCodingFit:
    """A latent-class fit of repeat codings.

    Latent levels are ordered by expected coding score. ``response[slot, level, category]`` is
    ``P(category | level)``; with exchangeable codings there is a single slot. ``posterior`` holds
    each unit's posterior over the latent levels and :meth:`probabilities` the per-unit
    probability of a level, which can be passed to :func:`soft_exposure_effect`.
    """

    prevalence: np.ndarray
    response: np.ndarray
    posterior: np.ndarray
    prevalence_standard_error: np.ndarray | None
    response_standard_error: np.ndarray | None
    log_likelihood: float
    converged: bool
    iterations: int
    starts: int
    starts_agreeing: int
    kruskal_identified: bool
    constrained: bool
    informative_codings: int

    def response_of(self, coding: int, level: int, category: int) -> float:
        """``P(category | level)`` of one coding."""
        slot = 0 if self.response.shape[0] == 1 else coding
        return float(self.response[slot, level, category])

    def probabilities(self, level: int = -1) -> np.ndarray:
        """Each unit's posterior probability of latent ``level`` (the highest by default)."""
        return np.asarray(self.posterior[:, level], dtype=np.float64)


def repeat_codings_latent_class(
    codings: object,
    *,
    levels: int = 2,
    categories: int | None = None,
    exchangeable: bool = False,
    lowest_level: Sequence[float] | None = None,
    starts: int = 8,
    max_iterations: int = 2000,
    tolerance: float = 1e-10,
    min_separation: float = 0.05,
    execution: Execution | None = None,
) -> RepeatCodingFit:
    """Fit repeat codings of a binary or ordinal facet as replicates of one latent level.

    ``codings`` is an ``(n, k)`` integer array of category indices ``0..categories-1`` (every unit
    has all ``k`` codings). The model assumes the codings are conditionally independent given the
    latent level. It is identified by three or more codings (more for more latent levels, by
    Kruskal's condition) or, for two codings of a binary facet, only when ``exchangeable=True``
    and ``lowest_level`` declares the response law of the lowest level (for example the
    probabilities of recording 0 and 1 for a truly absent facet). Anything else is refused with
    ``measurement_not_identified``, as is a fit whose codings do not separate the levels.
    """
    array = np.asarray(codings)
    if array.ndim != 2 or array.dtype.kind not in "iub":
        raise CausalValueError("codings must be an (n, k) integer array")
    if array.min(initial=0) < 0:
        raise CausalValueError("codings must be non-negative category indices")
    n, k = int(array.shape[0]), int(array.shape[1])
    category_count = int(array.max(initial=0)) + 1 if categories is None else int(categories)
    category_count = max(category_count, 2)
    declaration: dict[str, Any] = {
        "codings": [int(v) for v in array.reshape(-1)],
        "units": n,
        "coding_count": k,
        "categories": category_count,
        "latent_levels": _count(levels, "levels", minimum=2),
        "exchangeable": bool(exchangeable),
        "starts": _count(starts, "starts", minimum=1),
        "max_iterations": _count(max_iterations, "max_iterations", minimum=1),
        "tolerance": _number(tolerance, "tolerance"),
        "min_separation": _number(min_separation, "min_separation"),
        "seed": _execution_seed(execution),
    }
    if lowest_level is not None:
        declaration["lowest_level"] = _vector(lowest_level, "lowest_level")
    wire = _call(_native.measurement_repeat_codings, declaration)
    j = int(wire["latent_levels"])
    c = int(wire["categories"])
    slots = int(wire["response_slots"])
    prevalence_se = wire["prevalence_standard_error"]
    response_se = wire["response_standard_error"]
    return RepeatCodingFit(
        prevalence=np.asarray(wire["prevalence"], dtype=np.float64),
        response=np.asarray(wire["response"], dtype=np.float64).reshape(slots, j, c),
        posterior=np.asarray(wire["posterior"], dtype=np.float64).reshape(n, j),
        prevalence_standard_error=None
        if prevalence_se is None
        else np.asarray(prevalence_se, dtype=np.float64),
        response_standard_error=None
        if response_se is None
        else np.asarray(response_se, dtype=np.float64).reshape(slots, j, c),
        log_likelihood=float(wire["log_likelihood"]),
        converged=bool(wire["converged"]),
        iterations=int(wire["iterations"]),
        starts=int(wire["starts"]),
        starts_agreeing=int(wire["starts_agreeing"]),
        kruskal_identified=bool(wire["identification"]["kruskal"]),
        constrained=bool(wire["identification"]["constrained"]),
        informative_codings=int(wire["identification"]["informative_codings"]),
    )
