"""Descriptive raw-versus-adjusted comparison and declared reporting-scale transforms.

Two small computations over estimates you already hold. Both are **point only** and
**descriptive**: they describe a gap and re-express a pair of means, they do not identify or
decompose a causal effect.

* :func:`raw_vs_adjusted` sets the unadjusted contrast ``mean(Y | T=1) - mean(Y | T=0)`` beside
  an adjusted estimate that was computed under the *same* estimand coding (difference of
  means, active 1 versus control 0, the all-observed population) and reports the gap
  ``raw - adjusted``. A different scale, level coding or population is refused. The gap has
  no standard error or interval (the covariance of the two estimates is not carried and is
  not assumed zero) and it is **never attributed to adjustment columns**:
  :func:`attribute_gap_to_columns` is a typed refusal.
* :func:`transform_mean_pair` puts a pair of arm means on declared reporting scales
  (``risk_difference``, ``log_risk_ratio``, ``log_odds_ratio``, ``mean_difference``) by the
  delta method. The first-order covariance ``J Sigma J'`` of the transformed family exists
  only when you pass the joint covariance of the pair that the source carried; otherwise the
  transformed points are returned and the covariance is ``None`` with the typed reason in
  ``covariance_unavailable`` (independence is never assumed for arms that share data).
  :func:`raw_reporting_transform` does this for the raw arm means of a sample, whose
  independent-groups covariance it can form itself. No interval is reported on a transformed
  scale: ``level=...`` is refused with ``cell_not_licensed``.

``DescriptiveComparison.export()`` writes a JSON artifact carrying the arm sufficient
statistics and a digest; :func:`replay_descriptive_comparison` recomputes the raw contrast, its
Welch standard error and the gap from them in pure Python and compares them with the stored
values.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

from ._native import attribute_gap_to_columns as _attribute_gap_to_columns
from ._native import raw_vs_adjusted as _raw_vs_adjusted
from ._native import transform_mean_pair as _transform_mean_pair
from ._native import transform_raw_arms as _transform_raw_arms
from .errors import CausalSerializationError, CausalTypeError

__all__ = [
    "ArmSummary",
    "DescriptiveComparison",
    "ReportingTransform",
    "attribute_gap_to_columns",
    "raw_reporting_transform",
    "raw_vs_adjusted",
    "replay_descriptive_comparison",
    "transform_mean_pair",
]

_FORMAT = "descriptive_comparison_v1"
_SCALES = ("mean_difference", "risk_difference", "log_risk_ratio", "log_odds_ratio")
_INTERPRETATION = "descriptive_not_causal_decomposition"

# (registered reason code, detail) of the typed absences carried as data.
_GAP_INTERVAL_UNAVAILABLE = ("cell_not_licensed", "descriptive_comparison.gap_interval_unavailable")
_JOINT_COVARIANCE_UNAVAILABLE = (
    "required_option_missing",
    "descriptive_comparison.joint_covariance_unavailable",
)


@dataclass(frozen=True, slots=True)
class ArmSummary:
    """Complete rows, mean and sum of squared deviations of one treatment arm."""

    n: int
    mean: float
    sum_squares: float


def _arm(raw: tuple[int, float, float]) -> ArmSummary:
    return ArmSummary(n=raw[0], mean=raw[1], sum_squares=raw[2])


def _variance_of_mean(arm: ArmSummary) -> float | None:
    """``s^2 / n``, the variance of an arm mean under independent sampling."""
    return arm.sum_squares / (arm.n - 1) / arm.n if arm.n >= 2 else None


@dataclass(frozen=True, slots=True)
class DescriptiveComparison:
    """The raw contrast beside an adjusted estimate; ``claim`` is ``point_only``.

    ``gap`` is ``raw_difference - adjusted_estimate``. It is descriptive: it mixes
    confounding, the adjustment model's form, estimator differences and sampling noise, and
    it is not split over adjustment columns. ``raw_standard_error`` is the Welch
    independent-groups standard error (``None`` when an arm has fewer than two rows).
    ``gap_interval_unavailable`` names why the gap has no interval.
    """

    active: ArmSummary
    control: ArmSummary
    raw_difference: float
    raw_standard_error: float | None
    adjusted_estimate: float
    adjusted_standard_error: float | None
    gap: float
    scale: str = "mean_difference"
    claim: str = "point_only"
    interpretation: str = _INTERPRETATION
    gap_interval_unavailable: tuple[str, str] = _GAP_INTERVAL_UNAVAILABLE

    def export(self) -> str:
        """A self-describing JSON artifact that :func:`replay_descriptive_comparison` re-derives."""
        body: dict[str, Any] = {
            "format": _FORMAT,
            "scale": self.scale,
            "claim": self.claim,
            "interpretation": self.interpretation,
            "active": [self.active.n, self.active.mean, self.active.sum_squares],
            "control": [self.control.n, self.control.mean, self.control.sum_squares],
            "adjusted_estimate": self.adjusted_estimate,
            "adjusted_standard_error": self.adjusted_standard_error,
            "raw_difference": self.raw_difference,
            "raw_standard_error": self.raw_standard_error,
            "gap": self.gap,
        }
        return json.dumps({**body, "digest": _digest(body)}, sort_keys=True)


@dataclass(frozen=True, slots=True)
class ReportingTransform:
    """A mean pair on declared scales; ``claim`` is ``point_only`` and no interval exists.

    ``values[i]`` and ``gradients[i]`` (the row of the Jacobian with respect to the active
    and control means) belong to ``scales[i]``. ``covariance`` is the first-order delta-method
    covariance ``J Sigma J'`` as ``k`` rows of ``k`` numbers, or ``None`` when the source
    carried no joint covariance of the pair; ``covariance_unavailable`` is then the
    ``(reason code, detail)`` of the absence.
    """

    scales: tuple[str, ...]
    values: tuple[float, ...]
    gradients: tuple[tuple[float, float], ...]
    covariance: tuple[tuple[float, ...], ...] | None
    covariance_unavailable: tuple[str, str] | None
    claim: str = "point_only"

    def standard_error(self, scale: str) -> float | None:
        """Delta-method standard error of one scale, or ``None`` without a covariance."""
        if self.covariance is None:
            return None
        i = self.scales.index(scale)
        return math.sqrt(max(self.covariance[i][i], 0.0))


def _floats(name: str, values: Iterable[Any]) -> list[float]:
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError) as error:
        raise CausalTypeError(f"{name} must be a sequence of numbers") from error


def _digest(body: dict[str, Any]) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def raw_vs_adjusted(
    *,
    outcome: Iterable[Any],
    treatment: Iterable[Any],
    adjusted_estimate: float,
    adjusted_standard_error: float | None = None,
    adjusted_scale: str = "mean_difference",
    active: float = 1.0,
    control: float = 0.0,
    population: str = "all_observed",
) -> DescriptiveComparison:
    """Set the unadjusted contrast beside an adjusted estimate of the same estimand coding.

    ``treatment`` is coded exactly 0/1. The adjusted estimate must be a difference of means
    (``adjusted_scale="mean_difference"``) at ``active=1`` versus ``control=0`` on the
    ``population="all_observed"``; any other coding is refused rather than compared.
    """
    (n1, mean1, m2_1), (n0, mean0, m2_0), difference, se, gap = _raw_vs_adjusted(
        _floats("outcome", outcome),
        _floats("treatment", treatment),
        float(adjusted_estimate),
        adjusted_se=None if adjusted_standard_error is None else float(adjusted_standard_error),
        scale=adjusted_scale,
        active=float(active),
        control=float(control),
        all_observed=population == "all_observed",
    )
    return DescriptiveComparison(
        active=ArmSummary(n=n1, mean=mean1, sum_squares=m2_1),
        control=ArmSummary(n=n0, mean=mean0, sum_squares=m2_0),
        raw_difference=difference,
        raw_standard_error=se,
        adjusted_estimate=float(adjusted_estimate),
        adjusted_standard_error=(
            None if adjusted_standard_error is None else float(adjusted_standard_error)
        ),
        gap=gap,
    )


def attribute_gap_to_columns(*_args: object, **_kwargs: object) -> NoReturn:
    """Always refused (``effect_not_identified``): the gap is not split over columns.

    The adjusted estimate is not an additive function of the adjustment columns, so a
    leave-one-out or other per-column share of the raw-minus-adjusted gap is not a causal
    quantity. Whatever is passed is ignored.
    """
    _attribute_gap_to_columns()
    raise AssertionError("the native refusal always raises")  # pragma: no cover


def _scales(scales: Sequence[str]) -> list[str]:
    if isinstance(scales, str):
        raise CausalTypeError("scales must be a sequence of scale names, not one string")
    return [str(s) for s in scales]


def _transform(
    scales: list[str],
    raw: tuple[list[float], list[tuple[float, float]], list[float] | None],
    unavailable: tuple[str, str] | None,
) -> ReportingTransform:
    values, gradients, covariance = raw
    k = len(values)
    rows = (
        None
        if covariance is None
        else tuple(tuple(covariance[j * k + i] for j in range(k)) for i in range(k))
    )
    return ReportingTransform(
        scales=tuple(scales),
        values=tuple(values),
        gradients=tuple((g[0], g[1]) for g in gradients),
        covariance=rows,
        covariance_unavailable=unavailable if rows is None else None,
    )


def _covariance_triple(
    covariance: Sequence[Sequence[float]] | None,
) -> tuple[float, float, float] | None:
    if covariance is None:
        return None
    try:
        (a, b), (c, d) = ([float(v) for v in row] for row in covariance)
    except (TypeError, ValueError) as error:
        raise CausalTypeError("covariance must be a 2 x 2 matrix of numbers") from error
    if b != c and not math.isclose(b, c, rel_tol=1e-9, abs_tol=0.0):
        raise CausalTypeError("covariance must be symmetric")
    return (a, b, d)


def transform_mean_pair(
    mean_active: float,
    mean_control: float,
    *,
    scales: Sequence[str] = _SCALES[1:],
    covariance: Sequence[Sequence[float]] | None = None,
    level: float | None = None,
) -> ReportingTransform:
    """Put a pair of arm means on declared scales, with the delta-method covariance if carried.

    ``covariance`` is the ``2 x 2`` joint covariance of ``(mean_active, mean_control)`` exactly
    as the source estimate published it; ``None`` is not read as independence. ``level`` exists
    only so that asking for an interval is a typed refusal (``cell_not_licensed``).

    ``risk_difference`` needs both means in ``[0, 1]``, ``log_risk_ratio`` both means positive and
    ``log_odds_ratio`` both inside ``(0, 1)``; outside that the call is refused.
    """
    names = _scales(scales)
    raw = _transform_mean_pair(
        float(mean_active),
        float(mean_control),
        names,
        covariance=_covariance_triple(covariance),
        interval=level is not None,
    )
    return _transform(names, raw, _JOINT_COVARIANCE_UNAVAILABLE)


def raw_reporting_transform(
    *,
    outcome: Iterable[Any],
    treatment: Iterable[Any],
    scales: Sequence[str] = _SCALES[1:],
    level: float | None = None,
) -> ReportingTransform:
    """Put the raw arm means of a 0/1-treatment sample on declared scales.

    The two arms are disjoint groups of one sample, so their means are uncorrelated and the
    covariance is ``diag(s1^2/n1, s0^2/n0)``; an arm of fewer than two rows has no sample
    variance and the covariance is then unavailable. Descriptive, like the raw contrast.
    """
    names = _scales(scales)
    _, _, raw = _transform_raw_arms(
        _floats("outcome", outcome),
        _floats("treatment", treatment),
        names,
        interval=level is not None,
    )
    return _transform(names, raw, ("invalid_argument", "descriptive_comparison.arm_too_small"))


def _bad(why: str) -> CausalSerializationError:
    return CausalSerializationError(f"descriptive comparison artifact: {why}")


def _arm_stats(name: str, value: object) -> ArmSummary:
    if not (
        isinstance(value, list)
        and len(value) == 3
        and isinstance(value[0], int)
        and not isinstance(value[0], bool)
        and value[0] >= 1
        and all(isinstance(v, int | float) and not isinstance(v, bool) for v in value[1:])
        and all(math.isfinite(v) for v in value[1:])
        and value[2] >= 0
    ):
        raise _bad(f"{name} must be [count >= 1, mean, non-negative sum of squares]")
    return ArmSummary(n=value[0], mean=float(value[1]), sum_squares=float(value[2]))


def replay_descriptive_comparison(artifact: str) -> DescriptiveComparison:
    """Re-derive an exported comparison from its stored arm sufficient statistics.

    The digest is checked, then the raw difference, the Welch standard error and the gap are
    recomputed in pure Python from the stored counts, means and sums of squares and compared
    with the stored values. A tampered, unknown-format, non-reproducing or differently-claimed
    artifact is refused with :class:`~antecedent.errors.CausalSerializationError`.
    """
    try:
        body = json.loads(artifact)
    except (TypeError, ValueError) as error:
        raise _bad("not valid JSON") from error
    if not isinstance(body, dict) or body.get("format") != _FORMAT:
        raise _bad(f"format must be {_FORMAT!r}")
    stored = body.pop("digest", None)
    if stored != _digest(body):
        raise _bad("digest does not match the content")
    expected_keys = {
        "format",
        "scale",
        "claim",
        "interpretation",
        "active",
        "control",
        "adjusted_estimate",
        "adjusted_standard_error",
        "raw_difference",
        "raw_standard_error",
        "gap",
    }
    if (
        set(body) != expected_keys
        or body["claim"] != "point_only"
        or body["scale"] != "mean_difference"
        or body["interpretation"] != _INTERPRETATION
    ):
        raise _bad("unexpected fields, scale, claim or interpretation")
    active = _arm_stats("active", body["active"])
    control = _arm_stats("control", body["control"])
    adjusted = body["adjusted_estimate"]
    if not (isinstance(adjusted, int | float) and math.isfinite(adjusted)):
        raise _bad("adjusted_estimate must be a finite number")
    adjusted_se = body["adjusted_standard_error"]
    if adjusted_se is not None and not (
        isinstance(adjusted_se, int | float) and math.isfinite(adjusted_se) and adjusted_se >= 0
    ):
        raise _bad("adjusted_standard_error must be a non-negative number or null")
    difference = active.mean - control.mean
    va, vc = _variance_of_mean(active), _variance_of_mean(control)
    se = None if va is None or vc is None else math.sqrt(va + vc)
    if (
        body["raw_difference"] != difference
        or body["raw_standard_error"] != se
        or body["gap"] != difference - float(adjusted)
    ):
        raise _bad("the stored result does not reproduce from its sufficient statistics")
    return DescriptiveComparison(
        active=active,
        control=control,
        raw_difference=difference,
        raw_standard_error=se,
        adjusted_estimate=float(adjusted),
        adjusted_standard_error=None if adjusted_se is None else float(adjusted_se),
        gap=difference - float(adjusted),
    )
