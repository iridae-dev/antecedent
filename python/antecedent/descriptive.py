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

* :class:`Association` is the compiled noncausal route: an outcome against predictors, stated as a
  query to :func:`antecedent.analyze`. A declared selection rule picks a stratified mean or
  probability, linear, logistic or Poisson regression, a Pearson or distance correlation, or a
  predictive model scored on held-out folds, from the outcome scale and the functional. The
  answer (``result.answer.kind == "associational"``, ``result.association``) describes the
  observed law: identification is not claimed, it is never the effect of an intervention, it ranks
  no action, and every uncertainty it reports is unlicensed with calibration ``unmeasured``.
  Rows are independent units (``temporal=False``) or the consecutive points of one series
  (``temporal=True``, resampled in circular blocks under the declared dependent-series design).
  After a causal query was refused, ``refused_after=`` keeps the refusal's reason code beside the
  description; the description answers none of what the refusal declined to claim.

``DescriptiveComparison.export()`` writes a JSON artifact carrying the arm sufficient
statistics and a digest; :func:`replay_descriptive_comparison` recomputes the raw contrast, its
Welch standard error and the gap from them in pure Python and compares them with the stored
values.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NoReturn

from ._digest import canonical_digest
from ._native import CancellationToken
from ._native import attribute_gap_to_columns as _attribute_gap_to_columns
from ._native import raw_vs_adjusted as _raw_vs_adjusted
from ._native import transform_mean_pair as _transform_mean_pair
from ._native import transform_raw_arms as _transform_raw_arms
from ._scientific import ScientificResult
from .errors import CausalSerializationError, CausalTypeError, CausalValueError

if TYPE_CHECKING:
    from ._artifact_protocol import ExportedArtifact

__all__ = [
    "ArmSummary",
    "Association",
    "AssociationEstimate",
    "AssociationPosterior",
    "AssociationPrediction",
    "AssociationResampling",
    "AssociationTerm",
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
class DescriptiveComparison(ScientificResult):
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
    claim: Literal["point_only"] = "point_only"
    interpretation: str = _INTERPRETATION
    gap_interval_unavailable: tuple[str, str] = _GAP_INTERVAL_UNAVAILABLE

    def explain(self) -> str:
        """The raw contrast, the adjusted estimate and the gap, and what the gap is not."""
        text = (
            f"Raw {self.scale.replace('_', ' ')} {self.raw_difference:.4g} "
            f"(active n={self.active.n}, control n={self.control.n}) against adjusted estimate "
            f"{self.adjusted_estimate:.4g}: gap {self.gap:.4g}. "
        )
        if self.raw_standard_error is not None:
            text += f"Raw Welch standard error {self.raw_standard_error:.4g}. "
        return (
            text + "The gap is descriptive and point only: it mixes confounding, model form, "
            "estimator differences and sampling noise, is not attributed to adjustment columns, "
            f"and has no interval ({self.gap_interval_unavailable[1]})."
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form (the export body without its digest)."""
        return {
            "scale": self.scale,
            "claim": self.claim,
            "interpretation": self.interpretation,
            "active": {"n": self.active.n, "mean": self.active.mean},
            "control": {"n": self.control.n, "mean": self.control.mean},
            "raw_difference": self.raw_difference,
            "raw_standard_error": self.raw_standard_error,
            "adjusted_estimate": self.adjusted_estimate,
            "adjusted_standard_error": self.adjusted_standard_error,
            "gap": self.gap,
            "gap_interval_unavailable": list(self.gap_interval_unavailable),
        }

    def __repr__(self) -> str:
        return (
            f"<DescriptiveComparison raw={self.raw_difference:.4g} "
            f"adjusted={self.adjusted_estimate:.4g} gap={self.gap:.4g} {self.claim}>"
        )

    @property
    def identity(self) -> str:
        """Content identity retained independently for artifact consumption."""
        return str(json.loads(self.export())["digest"])

    def artifact(self) -> ExportedArtifact[DescriptiveComparison]:
        """Typed bytes envelope for shared scientific consumption."""
        from ._artifact_protocol import exported_artifact

        return exported_artifact(self)

    def export(self) -> bytes:
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
        return json.dumps({**body, "digest": _digest(body)}, sort_keys=True).encode("utf-8")


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
    claim: Literal["point_only"] = "point_only"

    def explain(self) -> str:
        """The transformed values and whether a covariance (and so a standard error) exists."""
        pairs = ", ".join(f"{s} = {v:.4g}" for s, v in zip(self.scales, self.values, strict=True))
        if self.covariance is None:
            reason = (self.covariance_unavailable or ("", ""))[1]
            tail = f"No covariance is available ({reason}); independence is never assumed."
        else:
            tail = "First-order delta-method covariance is available."
        return f"Reporting transform of a mean pair: {pairs}. Point only, no interval. {tail}"

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form of this transform."""
        return {
            "scales": list(self.scales),
            "values": list(self.values),
            "gradients": [list(g) for g in self.gradients],
            "covariance": None if self.covariance is None else [list(r) for r in self.covariance],
            "covariance_unavailable": None
            if self.covariance_unavailable is None
            else list(self.covariance_unavailable),
            "claim": self.claim,
        }

    def __repr__(self) -> str:
        return f"<ReportingTransform {list(self.scales)!r} {self.claim}>"

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
    return canonical_digest("antecedent.descriptive.report.v2", body)


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
    cancel: CancellationToken | None = None,
) -> DescriptiveComparison:
    """Set the unadjusted contrast beside an adjusted estimate of the same estimand coding.

    ``treatment`` is coded exactly 0/1. The adjusted estimate must be a difference of means
    (``adjusted_scale="mean_difference"``) at ``active=1`` versus ``control=0`` on the
    ``population="all_observed"``; any other coding is refused rather than compared. A cancelled
    ``cancel`` token stops the pass over the rows with ``cancelled_no_claim`` and no comparison.
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
        cancel=cancel,
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
    cancel: CancellationToken | None = None,
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
        cancel=cancel,
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


def replay_descriptive_comparison(
    artifact: bytes | str, *, expected_identity: str | None = None
) -> DescriptiveComparison:
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
    if expected_identity is not None and stored != expected_identity:
        raise _bad("identity differs from independently retained expected_identity")
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


# --- the compiled associational route ---------------------------------------------------------

_ASSOCIATION_FUNCTIONALS = (
    "auto",
    "conditional",
    "coefficients",
    "pearson_correlation",
    "distance_correlation",
    "predictive",
)
_ASSOCIATION_SCALES = ("schema", "continuous", "binary", "count")


def _names(value: str | Iterable[str], what: str) -> tuple[str, ...]:
    names = (value,) if isinstance(value, str) else tuple(value)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise CausalValueError(f"declare non-empty {what} names")
    return names


@dataclass(frozen=True, slots=True)
class Association:
    """A noncausal association of an outcome with predictors in the observed data.

    ``outcome`` and ``predictors`` name table columns (or the columns of one series when
    ``temporal=True``); ``conditioning`` names columns held fixed arithmetically (adjusted for in
    a regression, crossed into the strata of a conditional law). ``functional`` is ``"auto"`` (the
    declared selection rule picks from the outcome scale and the predictor scales),
    ``"conditional"`` (the mean of a real or count outcome, or the probability of a 0/1 outcome,
    in each stratum), ``"coefficients"`` (linear, logistic or Poisson regression by the outcome
    scale), ``"pearson_correlation"`` or ``"distance_correlation"`` (one predictor, nothing held
    fixed), or ``"predictive"`` (a model scored on ``folds`` held-out folds). ``outcome_scale``
    declares the scale the schema does not (``"binary"`` for a 0/1 float column, ``"count"``).

    The answer describes the observed law: ``result.answer.kind == "associational"``, identification
    is not claimed, and it is never the effect of an intervention or a ranking of actions. Pass
    ``refused_after=`` the reason code (or the error) of a refused causal query to keep the
    refusal beside the description. The route is unlicensed and its calibration is unmeasured.
    """

    outcome: str
    predictors: tuple[str, ...] = ()
    conditioning: tuple[str, ...] = ()
    functional: Literal[
        "auto",
        "conditional",
        "coefficients",
        "pearson_correlation",
        "distance_correlation",
        "predictive",
    ] = "auto"
    folds: int = 5
    outcome_scale: Literal["schema", "continuous", "binary", "count"] = "schema"
    weight: str | None = None
    temporal: bool = False
    refused_after: str | None = None
    kind: Literal["association"] = field(default="association", init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, str) or not self.outcome.strip():
            raise CausalValueError("declare a non-empty outcome name")
        predictors = _names(self.predictors, "predictor")
        conditioning = _names(self.conditioning, "conditioning")
        object.__setattr__(self, "predictors", predictors)
        object.__setattr__(self, "conditioning", conditioning)
        columns = [self.outcome, *predictors, *conditioning]
        if self.weight is not None:
            columns.append(self.weight)
        if len(set(columns)) != len(columns):
            raise CausalValueError(
                "outcome, predictors, conditioning and weight must be distinct columns"
            )
        if self.functional not in _ASSOCIATION_FUNCTIONALS:
            raise CausalValueError(
                f"unknown association functional {self.functional!r}; use one of "
                f"{', '.join(_ASSOCIATION_FUNCTIONALS)}"
            )
        if self.outcome_scale not in _ASSOCIATION_SCALES:
            raise CausalValueError(
                f"unknown outcome scale {self.outcome_scale!r}; use one of "
                f"{', '.join(_ASSOCIATION_SCALES)}"
            )
        if self.functional in ("coefficients", "predictive") and not predictors:
            raise CausalValueError(f"{self.functional} needs at least one predictor")
        if self.functional in ("pearson_correlation", "distance_correlation") and (
            len(predictors) != 1 or conditioning
        ):
            raise CausalValueError(
                "a correlation relates the outcome to exactly one predictor and holds nothing "
                "fixed; use functional='coefficients' to adjust for conditioning columns"
            )
        if self.functional == "predictive" and not 2 <= self.folds <= 20:
            raise CausalValueError("a predictive model is scored over 2 to 20 folds")
        refused = self.refused_after
        if refused is not None and not isinstance(refused, str):
            code = getattr(refused, "reason_code", None)
            if not isinstance(code, str) or not code:
                raise CausalValueError(
                    "refused_after is the reason code of the refused causal query, or the error"
                )
            object.__setattr__(self, "refused_after", code)
        if self.refused_after is not None and not 1 <= len(self.refused_after) <= 256:
            raise CausalValueError("refused_after must be 1 to 256 characters")


@dataclass(frozen=True, slots=True)
class AssociationTerm:
    """One reported quantity: a stratum value, a coefficient, a correlation.

    ``standard_error`` is the construction named by ``standard_error_method``;
    ``spread_lower`` / ``spread_upper`` are the uncalibrated percentile spread of the resampling
    replicates and the ``posterior_*`` fields the summary of a posterior. None is a coverage claim.
    """

    label: str
    scale: str
    estimate: float
    rows: int
    standard_error: float | None = None
    standard_error_method: str | None = None
    spread_lower: float | None = None
    spread_upper: float | None = None
    posterior_mean: float | None = None
    posterior_sd: float | None = None
    posterior_lower: float | None = None
    posterior_upper: float | None = None


@dataclass(frozen=True, slots=True)
class AssociationResampling:
    """Resampling behind the standard errors and spreads of a frequentist association."""

    method: str
    unit: str
    replicates_requested: int
    replicates_ok: int
    spread_level: float
    block_length: int | None = None


@dataclass(frozen=True, slots=True)
class AssociationPosterior:
    """Posterior behind the summaries of a Bayesian association."""

    method: str
    draws: int
    level: float


@dataclass(frozen=True, slots=True)
class AssociationPrediction:
    """Held-out proper scores of a predictive model beside the training-mean baseline."""

    folds: int
    fold_design: str
    cases: int
    score: str
    mean_score: float
    baseline_score: float
    calibration: str = "unmeasured"
    mean_crps: float | None = None


@dataclass(frozen=True, slots=True)
class AssociationEstimate:
    """A noncausal association and the evidence behind it. Unlicensed, calibration unmeasured.

    ``analysis`` is what the declared selection rule (``rule``) picked for the ``outcome_kind``
    and ``modality``; ``requested`` is the functional the query asked for. The result is never an
    intervention effect (``is_intervention_effect`` is ``False``) and ranks no action
    (``ranks_actions`` is ``False``); ``assumptions`` states what the description rests on.
    """

    analysis: str
    rule: str
    requested: str
    outcome_kind: str
    modality: str
    terms: tuple[AssociationTerm, ...]
    rows: int
    rows_dropped: int
    assumptions: tuple[str, ...]
    uncertainty: str
    resampling: AssociationResampling | None = None
    posterior: AssociationPosterior | None = None
    prediction: AssociationPrediction | None = None
    refused_after: str | None = None
    calibration: str = "unmeasured"
    licensed: bool = False
    ranks_actions: bool = False
    is_intervention_effect: bool = False


def association_from_raw(raw: Any) -> AssociationEstimate | None:
    """The :class:`AssociationEstimate` of a native result, or ``None`` for another route."""
    section = getattr(raw, "association", None)
    if section is None:
        return None
    resampling = section.resampling
    posterior = section.posterior
    prediction = section.prediction
    return AssociationEstimate(
        analysis=section.analysis,
        rule=section.rule,
        requested=section.requested,
        outcome_kind=section.outcome_kind,
        modality=section.modality,
        terms=tuple(
            AssociationTerm(
                label=term.label,
                scale=term.scale,
                estimate=term.estimate,
                rows=term.rows,
                standard_error=term.standard_error,
                standard_error_method=term.standard_error_method,
                spread_lower=term.spread_lower,
                spread_upper=term.spread_upper,
                posterior_mean=term.posterior_mean,
                posterior_sd=term.posterior_sd,
                posterior_lower=term.posterior_lower,
                posterior_upper=term.posterior_upper,
            )
            for term in section.terms
        ),
        rows=section.rows,
        rows_dropped=section.rows_dropped,
        assumptions=tuple(section.assumptions),
        uncertainty=section.uncertainty,
        resampling=None
        if resampling is None
        else AssociationResampling(
            method=resampling.method,
            unit=resampling.unit,
            replicates_requested=resampling.replicates_requested,
            replicates_ok=resampling.replicates_ok,
            spread_level=resampling.spread_level,
            block_length=resampling.block_length,
        ),
        posterior=None
        if posterior is None
        else AssociationPosterior(
            method=posterior.method, draws=posterior.draws, level=posterior.level
        ),
        prediction=None
        if prediction is None
        else AssociationPrediction(
            folds=prediction.folds,
            fold_design=prediction.fold_design,
            cases=prediction.cases,
            score=prediction.score,
            mean_score=prediction.mean_score,
            baseline_score=prediction.baseline_score,
            calibration=prediction.calibration,
            mean_crps=prediction.mean_crps,
        ),
        refused_after=section.refused_after,
        calibration=section.calibration,
        licensed=section.licensed,
        ranks_actions=section.ranks_actions,
        is_intervention_effect=section.is_intervention_effect,
    )


def describe_association(association: AssociationEstimate) -> str:
    """One sentence stating what an association says, and what it does not."""
    first = association.terms[0]
    head = (
        f"Noncausal {association.analysis.replace('_', ' ')} of the observed data "
        f"({association.rows} rows, {association.modality.replace('_', ' ')}): "
        f"{first.label} = {first.estimate:g}"
    )
    if len(association.terms) > 1:
        head += f" and {len(association.terms) - 1} more quantit"
        head += "ies" if len(association.terms) > 2 else "y"
    spread = {
        "posterior": " with a posterior",
        "resampling_standard_error": " with resampling standard errors",
        "analytic_standard_error": " with analytic standard errors",
    }.get(association.uncertainty, " with no uncertainty")
    follows = (
        f" It follows a causal query refused with {association.refused_after}."
        if association.refused_after
        else ""
    )
    return (
        f"{head}{spread}. Identification is not claimed: this is not the effect of changing any "
        f"variable and ranks no action.{follows} Unlicensed, calibration unmeasured."
    )
