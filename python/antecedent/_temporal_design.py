"""Query dataclasses, design binding and result mapping of the temporal designs.

One table row per unit and period. Every query names columns: the outcome, the unit and the
period, and the columns that record the assignment law. The design columns stay frozen query
metadata (a refreshed table must repeat them); only the outcome (and, for regime survival,
the treatment, censoring and confounder columns) reaches the numeric ingest.

The public names are re-exported by :mod:`antecedent.temporal_designs`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import Any

from ._data import ingest_columns, read_columns
from .errors import CausalValueError


def _column_name(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalValueError(f"{what} must be a non-empty column name")
    return value


def _check_common(query: Any) -> None:
    for what in ("outcome", "unit", "period"):
        _column_name(getattr(query, what), what)
    lag = getattr(query, "lag", 0)
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 0:
        raise CausalValueError("lag must be a non-negative integer")
    level = getattr(query, "quantile_level", None)
    if level is not None and not (0.0 < float(level) < 1.0):
        raise CausalValueError("quantile_level must lie strictly between 0 and 1")
    floor = float(query.min_probability)
    if not (0.0 < floor <= 0.5):
        raise CausalValueError("min_probability must lie in (0, 0.5]")
    misclassification = getattr(query, "misclassification", None)
    if misclassification is not None and not isinstance(
        misclassification, OutcomeMisclassification
    ):
        raise CausalValueError("misclassification must be an OutcomeMisclassification")


# -- dose targets and options --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SmoothedDoseCurve:
    """The kernel-smoothed dose-response curve on ``grid`` (the bandwidth is in the target)."""

    grid: Sequence[float]

    def _wire(self) -> dict[str, Any]:
        return {"kind": "smoothed_curve", "grid": [float(value) for value in self.grid]}


@dataclass(frozen=True, slots=True)
class SmoothedDoseContrast:
    """The smoothed contrast of the response at ``active`` against ``control``."""

    active: float
    control: float

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "smoothed_contrast",
            "active": float(self.active),
            "control": float(self.control),
        }


@dataclass(frozen=True, slots=True)
class LocalQuadraticDoseGrid:
    """The local-quadratic level (or ``derivative``) on ``grid``.

    The frequentist grid needs ``curvature_bound``, a declared bound on ``|m'''|``: each
    interval adds the worst-case smoothing bias the bound allows, and the band widens by it.
    """

    grid: Sequence[float]
    derivative: bool = False
    curvature_bound: float | None = None

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "local_quadratic",
            "grid": [float(value) for value in self.grid],
            "derivative": bool(self.derivative),
            "curvature_bound": None
            if self.curvature_bound is None
            else float(self.curvature_bound),
        }


DoseTarget = SmoothedDoseCurve | SmoothedDoseContrast | LocalQuadraticDoseGrid


@dataclass(frozen=True, slots=True)
class OutcomeMisclassification:
    """Counts from an independent validation sample of a misclassified binary outcome.

    The excursion contrast is corrected for non-differential misclassification with the
    sensitivity and specificity these counts estimate; their uncertainty is propagated.
    """

    true_positive: int
    false_negative: int
    true_negative: int
    false_positive: int

    def __post_init__(self) -> None:
        counts = (self.true_positive, self.false_negative, self.true_negative, self.false_positive)
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts
        ):
            raise CausalValueError("misclassification counts must be non-negative integers")

    def _wire(self) -> tuple[int, int, int, int]:
        return (self.true_positive, self.false_negative, self.true_negative, self.false_positive)


# -- queries -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TwoArmExcursion:
    """Two arms assigned with a recorded, possibly history-dependent probability.

    ``assignment`` is 0/1 per row and ``probability`` the recorded probability of arm 1. The
    target is the excursion effect of the decision on the same unit's outcome ``lag`` periods
    later, carryover inside the target. ``quantile_level`` asks for the quantile contrast.
    """

    outcome: str
    unit: str
    period: str
    _: KW_ONLY
    assignment: str
    probability: str
    lag: int = 0
    quantile_level: float | None = None
    min_probability: float = 0.05
    misclassification: OutcomeMisclassification | None = None

    def __post_init__(self) -> None:
        _check_common(self)
        _column_name(self.assignment, "assignment")
        _column_name(self.probability, "probability")


@dataclass(frozen=True, slots=True)
class MultiArmExcursion:
    """Several arms (or categorical levels) with recorded per-arm probabilities.

    ``assignment`` holds the arm label of each row, ``arms`` names the arms in order and
    ``probabilities`` names one column per arm with its recorded probability. Each arm is
    contrasted with the first; ``categorical`` reads the arms as unordered levels and
    ``ordered`` as an ordered scale. The contrasts carry Holm-adjusted p-values and a
    simultaneous band.
    """

    outcome: str
    unit: str
    period: str
    _: KW_ONLY
    assignment: str
    probabilities: Sequence[str]
    arms: Sequence[Any]
    categorical: bool = False
    ordered: bool = False
    lag: int = 0
    quantile_level: float | None = None
    min_probability: float = 0.05
    misclassification: OutcomeMisclassification | None = None

    def __post_init__(self) -> None:
        _check_common(self)
        _column_name(self.assignment, "assignment")
        if len(self.probabilities) != len(self.arms) or len(self.arms) < 2:
            raise CausalValueError(
                "a multi-arm design names two or more arms, one probability each"
            )
        for name in self.probabilities:
            _column_name(name, "probabilities")


@dataclass(frozen=True, slots=True)
class FactorialExcursion:
    """Two factors, each assigned with its own recorded probability, and their interaction."""

    outcome: str
    unit: str
    period: str
    _: KW_ONLY
    first: str
    first_probability: str
    second: str
    second_probability: str
    lag: int = 0
    quantile_level: float | None = None
    min_probability: float = 0.05
    misclassification: OutcomeMisclassification | None = None

    def __post_init__(self) -> None:
        _check_common(self)
        for what in ("first", "first_probability", "second", "second_probability"):
            _column_name(getattr(self, what), what)


@dataclass(frozen=True, slots=True)
class InterferenceExcursion:
    """Within-cluster interference: the direct and spillover excursion effects.

    ``cluster`` labels the rows that share exposure in a period; assignments are
    independent with recorded probabilities.
    """

    outcome: str
    unit: str
    period: str
    _: KW_ONLY
    assignment: str
    probability: str
    cluster: str
    lag: int = 0
    quantile_level: float | None = None
    min_probability: float = 0.05
    misclassification: OutcomeMisclassification | None = None

    def __post_init__(self) -> None:
        _check_common(self)
        for what in ("assignment", "probability", "cluster"):
            _column_name(getattr(self, what), what)


@dataclass(frozen=True, slots=True)
class DoseExcursion:
    """A continuous dose drawn from a recorded density: smoothed curve, contrast or grid."""

    outcome: str
    unit: str
    period: str
    _: KW_ONLY
    dose: str
    density: str
    bandwidth: float
    target: DoseTarget
    lag: int = 0
    quantile_level: float | None = None
    min_probability: float = 0.05
    misclassification: OutcomeMisclassification | None = None

    def __post_init__(self) -> None:
        _check_common(self)
        _column_name(self.dose, "dose")
        _column_name(self.density, "density")
        if not (math.isfinite(float(self.bandwidth)) and float(self.bandwidth) > 0.0):
            raise CausalValueError("bandwidth must be finite and positive")
        if not isinstance(
            self.target, (SmoothedDoseCurve, SmoothedDoseContrast, LocalQuadraticDoseGrid)
        ):
            raise CausalValueError(
                "target must be SmoothedDoseCurve, SmoothedDoseContrast or LocalQuadraticDoseGrid"
            )


@dataclass(frozen=True, slots=True)
class RegimeSurvivalContrast:
    """Always-treat against never-treat survival at ``horizon`` from subject trajectories.

    ``outcome`` is the event indicator; ``treatment`` and ``censored`` are 0/1 columns and
    ``confounders`` time-varying covariates. Frequentist: inverse-probability-of-regime
    weighting with a whole-procedure subject bootstrap (``bootstrap=`` of at least 99).
    Bayesian: the joint-likelihood parametric g-formula.
    """

    outcome: str
    unit: str
    period: str
    _: KW_ONLY
    treatment: str
    censored: str
    horizon: int
    confounders: Sequence[str] = ()
    min_probability: float = 0.05

    def __post_init__(self) -> None:
        _check_common(self)
        _column_name(self.treatment, "treatment")
        _column_name(self.censored, "censored")
        for name in self.confounders:
            _column_name(name, "confounders")
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon < 1:
            raise CausalValueError("horizon must be a positive integer")


@dataclass(frozen=True, slots=True)
class ObservationalExcursion:
    """The excursion effect of an observed decision whose assignment law is not recorded.

    ``treatment`` is a 0/1 column and ``history`` the decision-time covariates that make the
    decision exchangeable (declared, not checked). The assignment law is a pooled logistic fit
    of the decision on ``history`` and is refitted inside every interval: frequentist inference
    is a unit bootstrap (``bootstrap=`` of at least 99) and Bayesian inference the
    weighted-likelihood bootstrap. ``lag`` pairs a decision with the outcome that many periods
    later; ``min_probability`` is the positivity floor of the fitted probabilities.
    """

    outcome: str
    unit: str
    period: str
    _: KW_ONLY
    treatment: str
    history: Sequence[str] = ()
    lag: int = 0
    min_probability: float = 0.05

    def __post_init__(self) -> None:
        _check_common(self)
        _column_name(self.treatment, "treatment")
        for name in self.history:
            _column_name(name, "history")
        if len(self.history) == 0:
            raise CausalValueError("an observational excursion needs at least one history column")


REGIME_ESTIMATORS = ("g_formula", "inverse_probability_weighting", "doubly_robust")


@dataclass(frozen=True, slots=True)
class ObservationalRegimeContrast:
    """The value of whole static treatment regimes under time-varying confounding.

    Subjects are observed in the same periods. In each period the ``confounders`` are measured
    at the start of the period, then the 0/1 ``treatment`` is taken; ``outcome`` is the
    continuous end-of-follow-up outcome, read at each subject's last period. ``active`` and
    ``reference`` are whole treatment paths, one 0/1 action per period in period order, and the
    result is the mean outcome under ``active`` minus that under ``reference``
    (``regime_effect``) with both means. Identification is the declared sequential
    exchangeability given the measured confounder and treatment history (no unmeasured
    time-varying confounding), sequential positivity of the path actions (held to
    ``min_probability`` in the fitted probabilities), consistency, no interference and complete
    follow-up.

    ``method`` is ``"doubly_robust"`` (recursive augmented inverse probability, consistent when
    either the outcome regressions or the per-period treatment models are right),
    ``"g_formula"`` (iterated conditional expectations) or ``"inverse_probability_weighting"``
    (the saturated marginal structural model of each path). Every nuisance model is refitted
    inside each interval: frequentist inference is a unit bootstrap (``bootstrap=`` of at least
    99) and Bayesian inference the weighted-likelihood bootstrap.
    """

    outcome: str
    unit: str
    period: str
    _: KW_ONLY
    treatment: str
    confounders: Sequence[str]
    active: Sequence[int]
    reference: Sequence[int]
    method: str = "doubly_robust"
    min_probability: float = 0.05

    def __post_init__(self) -> None:
        _check_common(self)
        _column_name(self.treatment, "treatment")
        for name in self.confounders:
            _column_name(name, "confounders")
        if len(self.confounders) == 0:
            raise CausalValueError("a whole-regime contrast needs at least one confounder column")
        for label, path in (("active", self.active), ("reference", self.reference)):
            if len(path) == 0 or any(step not in (0, 1) for step in path):
                raise CausalValueError(f"{label} must be a non-empty path of 0/1 actions")
        if len(self.active) != len(self.reference):
            raise CausalValueError("active and reference must name the same number of periods")
        if [int(step) for step in self.active] == [int(step) for step in self.reference]:
            raise CausalValueError("active and reference must differ in some period")
        if self.method not in REGIME_ESTIMATORS:
            raise CausalValueError(f"method must be one of {REGIME_ESTIMATORS}")


TEMPORAL_DESIGN_QUERIES = (
    TwoArmExcursion,
    MultiArmExcursion,
    FactorialExcursion,
    InterferenceExcursion,
    DoseExcursion,
    RegimeSurvivalContrast,
    ObservationalExcursion,
    ObservationalRegimeContrast,
)

TemporalDesignQuery = (
    TwoArmExcursion
    | MultiArmExcursion
    | FactorialExcursion
    | InterferenceExcursion
    | DoseExcursion
    | RegimeSurvivalContrast
    | ObservationalExcursion
    | ObservationalRegimeContrast
)


# -- result ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TemporalDesignEstimate:
    """A temporal-design result: named coordinates with their uncertainty.

    Field names are the artifact's, so a loaded artifact reads back equal to the live result.
    ``estimate`` is the point estimate (frequentist) or the posterior mean (Bayesian);
    ``standard_error`` is the standard error (the whole-procedure bootstrap standard
    deviation when ``replicates`` is set) or the posterior standard deviation (when ``draws``
    is set). Unlicensed; calibration unmeasured.
    """

    design: str
    functional: str
    coordinates: tuple[str, ...]
    estimate: tuple[float, ...]
    standard_error: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    covariance: tuple[float, ...] | None
    band_coordinates: tuple[int, ...]
    band_critical_value: float | None
    band_lower: tuple[float, ...] | None
    band_upper: tuple[float, ...] | None
    holm_adjusted_p: tuple[float, ...] | None
    smoothing_bias_bound: tuple[float, ...] | None
    pairs: int
    clusters: int
    unpaired: int
    minimum_probability: float | None
    #: Successful and failed whole-procedure bootstrap replicates.
    replicates: tuple[int, int] | None
    #: Posterior draws.
    draws: int | None
    uncertainty: str
    calibration: str
    licensed: bool

    def coordinate(self, name: str) -> tuple[float, float, float]:
        """``(estimate, lower, upper)`` of the named coordinate."""
        try:
            index = self.coordinates.index(name)
        except ValueError:
            raise CausalValueError(
                f"unknown coordinate {name!r}; the result has {list(self.coordinates)}",
                reason_code="invalid_argument",
            ) from None
        return (self.estimate[index], self.lower[index], self.upper[index])

    @property
    def bootstrap_standard_error(self) -> tuple[float, ...] | None:
        """Standard deviation of the whole-procedure bootstrap replicates, per coordinate.

        Set when the result was resampled (the quantile functional and frequentist regime
        survival); ``None`` otherwise.
        """
        return self.standard_error if self.replicates is not None else None

    @property
    def posterior_standard_deviation(self) -> tuple[float, ...] | None:
        """Posterior standard deviation per coordinate, for a Bayesian result."""
        return self.standard_error if self.draws is not None else None


def _floats(values: Any) -> tuple[float, ...] | None:
    return None if values is None else tuple(float(value) for value in values)


def _probability(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def temporal_design_from_raw(raw: Any) -> TemporalDesignEstimate | None:
    """The native ``temporal_design`` section as a frozen result, or ``None``."""
    section = getattr(raw, "temporal_design", None)
    if section is None:
        return None
    replicates = section.replicates
    return TemporalDesignEstimate(
        design=str(section.design),
        functional=str(section.functional),
        coordinates=tuple(str(name) for name in section.coordinates),
        estimate=tuple(float(value) for value in section.estimate),
        standard_error=tuple(float(value) for value in section.standard_error),
        lower=tuple(float(value) for value in section.lower),
        upper=tuple(float(value) for value in section.upper),
        covariance=_floats(section.covariance),
        band_coordinates=tuple(int(index) for index in section.band_coordinates),
        band_critical_value=section.band_critical_value,
        band_lower=_floats(section.band_lower),
        band_upper=_floats(section.band_upper),
        holm_adjusted_p=_floats(section.holm_adjusted_p),
        smoothing_bias_bound=_floats(section.smoothing_bias_bound),
        pairs=int(section.pairs),
        clusters=int(section.clusters),
        unpaired=int(section.unpaired),
        minimum_probability=_probability(section.minimum_probability),
        replicates=None if replicates is None else (int(replicates[0]), int(replicates[1])),
        draws=None if section.draws is None else int(section.draws),
        uncertainty=str(section.uncertainty),
        calibration=str(section.calibration),
        licensed=bool(section.licensed),
    )


def temporal_design_from_wire(wire: Any) -> TemporalDesignEstimate | None:
    """A loaded artifact's ``temporal_design`` body section as the live frozen result."""
    if not isinstance(wire, Mapping):
        return None
    replicates = wire.get("replicates")
    critical = wire.get("band_critical_value")
    return TemporalDesignEstimate(
        design=str(wire["design"]),
        functional=str(wire["functional"]),
        coordinates=tuple(str(name) for name in wire["coordinates"]),
        estimate=tuple(float(value) for value in wire["estimate"]),
        standard_error=tuple(float(value) for value in wire["standard_error"]),
        lower=tuple(float(value) for value in wire["lower"]),
        upper=tuple(float(value) for value in wire["upper"]),
        covariance=_floats(wire.get("covariance")),
        band_coordinates=tuple(int(index) for index in wire.get("band_coordinates") or ()),
        band_critical_value=None if critical is None else float(critical),
        band_lower=_floats(wire.get("band_lower")),
        band_upper=_floats(wire.get("band_upper")),
        holm_adjusted_p=_floats(wire.get("holm_adjusted_p")),
        smoothing_bias_bound=_floats(wire.get("smoothing_bias_bound")),
        pairs=int(wire["pairs"]),
        clusters=int(wire["clusters"]),
        unpaired=int(wire["unpaired"]),
        minimum_probability=_probability(wire.get("minimum_probability")),
        replicates=None if replicates is None else (int(replicates[0]), int(replicates[1])),
        draws=None if wire.get("draws") is None else int(wire["draws"]),
        uncertainty=str(wire["uncertainty"]),
        calibration=str(wire.get("calibration", "unmeasured")),
        licensed=bool(wire.get("licensed", False)),
    )


def describe_temporal_design(estimate: TemporalDesignEstimate) -> str:
    """One-sentence claim of a temporal-design result."""
    if not estimate.coordinates:
        return f"{estimate.design} temporal design with no coordinates."
    name = estimate.coordinates[0]
    value, lower, upper = estimate.coordinate(name)
    spread = estimate.standard_error[0]
    if estimate.replicates is not None:
        ok, failed = estimate.replicates
        scale = (
            f"whole-procedure bootstrap standard error {spread:g} ({ok} replicates, "
            f"{failed} failed)"
        )
    elif estimate.draws is not None:
        scale = f"posterior standard deviation {spread:g} ({estimate.draws} draws)"
    else:
        scale = f"standard error {spread:g}"
    return (
        f"{estimate.design} temporal design, {name} = {value:g}: 95% interval "
        f"[{lower:g}, {upper:g}], {scale} ({estimate.uncertainty}). Unlicensed; calibration "
        f"is {estimate.calibration}."
    )


# -- design binding ------------------------------------------------------------------------


def _binary(values: Any, what: str) -> list[bool]:
    out = []
    for value in values:
        number = float(value)
        if number not in (0.0, 1.0):
            raise CausalValueError(f"{what} must be 0/1; got {value!r}")
        out.append(number == 1.0)
    return out


def _numbers(values: Any) -> list[float]:
    return [float(value) for value in values]


def _label(value: Any) -> str:
    if isinstance(value, str):
        return value
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)


def _periods(values: Any) -> list[int]:
    out = []
    for value in values:
        number = float(value)
        if not number.is_integer():
            raise CausalValueError(f"period must be an integer; got {value!r}")
        out.append(int(number))
    return out


def _spec(query: Any, raw: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(query, TwoArmExcursion):
        return {
            "design": "two_arm",
            "assignment": _binary(raw[query.assignment], query.assignment),
            "probability": _numbers(raw[query.probability]),
        }
    if isinstance(query, MultiArmExcursion):
        arms = [_label(arm) for arm in query.arms]
        index = {arm: position for position, arm in enumerate(arms)}
        assignment = []
        for value in raw[query.assignment]:
            label = _label(value)
            if label not in index:
                raise CausalValueError(f"assignment {value!r} is not one of the arms {arms}")
            assignment.append(index[label])
        columns = [_numbers(raw[name]) for name in query.probabilities]
        return {
            "design": "multi_arm",
            "assignment": assignment,
            "probabilities": [list(row) for row in zip(*columns, strict=True)],
            "arms": arms,
            "categorical": bool(query.categorical),
            "ordered": bool(query.ordered),
        }
    if isinstance(query, FactorialExcursion):
        return {
            "design": "factorial",
            "first": _binary(raw[query.first], query.first),
            "first_probability": _numbers(raw[query.first_probability]),
            "second": _binary(raw[query.second], query.second),
            "second_probability": _numbers(raw[query.second_probability]),
        }
    if isinstance(query, InterferenceExcursion):
        return {
            "design": "interference",
            "assignment": _binary(raw[query.assignment], query.assignment),
            "probability": _numbers(raw[query.probability]),
            "clusters": [_label(value) for value in raw[query.cluster]],
        }
    if isinstance(query, DoseExcursion):
        return {
            "design": "dose",
            "dose": _numbers(raw[query.dose]),
            "density": _numbers(raw[query.density]),
            "bandwidth": float(query.bandwidth),
            "target": query.target._wire(),
        }
    if isinstance(query, RegimeSurvivalContrast):
        return {
            "design": "regime_survival",
            "treatment": query.treatment,
            "censored": query.censored,
            "confounders": list(query.confounders),
            "horizon": int(query.horizon),
        }
    if isinstance(query, ObservationalExcursion):
        return {
            "design": "observational_excursion",
            "treatment": query.treatment,
            "history": list(query.history),
        }
    if isinstance(query, ObservationalRegimeContrast):
        return {
            "design": "observational_regime",
            "treatment": query.treatment,
            "confounders": list(query.confounders),
            "active": [bool(step) for step in query.active],
            "reference": [bool(step) for step in query.reference],
            "method": query.method,
        }
    raise CausalValueError(f"not a temporal design query: {type(query).__name__}")


def _design_columns(query: Any) -> tuple[str, ...]:
    if isinstance(query, TwoArmExcursion | InterferenceExcursion):
        names: tuple[str, ...] = (query.assignment, query.probability)
        return (*names, query.cluster) if isinstance(query, InterferenceExcursion) else names
    if isinstance(query, MultiArmExcursion):
        return (query.assignment, *query.probabilities)
    if isinstance(query, FactorialExcursion):
        return (query.first, query.first_probability, query.second, query.second_probability)
    if isinstance(query, DoseExcursion):
        return (query.dose, query.density)
    return ()


def _numeric_columns(query: Any) -> tuple[str, ...]:
    if isinstance(query, RegimeSurvivalContrast):
        return (query.outcome, query.treatment, query.censored, *query.confounders)
    if isinstance(query, ObservationalExcursion):
        return (query.outcome, query.treatment, *query.history)
    if isinstance(query, ObservationalRegimeContrast):
        return (query.outcome, query.treatment, *query.confounders)
    return (query.outcome,)


def temporal_design_payload(data: Any, query: Any) -> tuple[list[str], list[Any], dict[str, Any]]:
    """Numeric outcome (and survival) columns; units, periods and the design as metadata."""
    numeric = tuple(dict.fromkeys(_numeric_columns(query)))
    required = tuple(dict.fromkeys((*numeric, query.unit, query.period, *_design_columns(query))))
    raw = read_columns(data, required, what="temporal design")
    units = tuple(_label(value) for value in raw[query.unit])
    if any(not unit.strip() for unit in units):
        raise CausalValueError("unit labels must be non-empty")
    design = {
        "units": units,
        "periods": tuple(_periods(raw[query.period])),
        "spec": json.dumps(_spec(query, raw), allow_nan=False),
    }
    names, columns = ingest_columns({name: raw[name] for name in numeric})
    return names, columns, design


def temporal_design_keywords(query: Any) -> dict[str, Any]:
    """The query-level keywords of the native prepare."""
    misclassification = getattr(query, "misclassification", None)
    level = getattr(query, "quantile_level", None)
    return {
        "lag": int(getattr(query, "lag", 0)),
        "quantile_level": None if level is None else float(level),
        "min_probability": float(query.min_probability),
        "misclassification": None if misclassification is None else misclassification._wire(),
    }
