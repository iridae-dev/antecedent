"""Missing-not-at-random (MNAR) sensitivity of a two-arm mean contrast.

The missing-at-random (MAR) estimate of an average treatment effect takes the missing outcomes
of each arm to follow that arm's observed outcomes. Nothing in the data can test that. This
module relaxes it by an explicit, one-parameter, bounded departure and re-runs the declared
estimand (the difference of the arms' full-population mean outcomes) across a declared grid::

    result = mnar_ate_sensitivity(
        treatment, outcome,                  # outcome entries None / NaN are missing
        model="delta_pattern_mixture",       # or "exponential_tilting_selection"
        loading="treated_only",              # which arm departs
        lower=-5.0, upper=5.0,               # a declared bounded range that contains 0 (MAR)
        decisions={"treat": 1.0},            # act when the effect exceeds the threshold
    )
    result.mar.effect                        # parameter 0: the MAR estimate
    result.grid                              # the effect as a function of the parameter
    result.tipping_point("sign")             # where the sign of the effect changes
    result.decisions[0].changes_within_range

Three things stay distinct and are never merged:

* the **assumption range** (``result.assumption_range``) is the envelope of the point effects
  over the declared parameter range, *conditional on the departure model*: it is not a
  confidence interval and it is not empirically established;
* the **departure assumption** is recorded in ``result.assumptions`` with status ``untestable``;
* the **sampling interval** at each grid point (``point.sampling``) is a delta-method interval
  conditional on that parameter value. It is reported beside the range, and composing it with the
  range refuses (``cell_not_licensed`` / ``mnar_sensitivity.composition_not_licensed``).

Rust owns the estimates, the tipping search and every refusal; this module builds declarations
and raises each refusal as :class:`MnarSensitivityRefusal`, a
:class:`~antecedent.errors.CausalUnsupportedError` with its registered ``reason_code`` and the
namespaced ``detail``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

from ._native import mnar_sensitivity_run as _run
from ._refusal import raise_refusal
from ._validate import real
from .errors import CausalTypeError, CausalValueError, StructuredRefusal

__all__ = [
    "MnarAssumption",
    "MnarAssumptionRange",
    "MnarDecisionResult",
    "MnarPoint",
    "MnarResult",
    "MnarSamplingInterval",
    "MnarSensitivityRefusal",
    "MnarTipping",
    "MnarTippingSide",
    "from_analysis",
    "mnar_ate_sensitivity",
]

_DEFAULT_GRID_POINTS = 21
_DEFAULT_LEVEL = 0.95
_DEFAULT_TOLERANCE = 1e-12
_LOADINGS: dict[str, tuple[float, float]] = {
    "common": (1.0, 1.0),
    "treated_only": (1.0, 0.0),
    "control_only": (0.0, 1.0),
    "opposite": (1.0, -1.0),
}


class MnarSensitivityRefusal(StructuredRefusal):
    """A missing-not-at-random sensitivity refusal carrying the structured Rust fields.

    A :class:`~antecedent.errors.StructuredRefusal`: ``code``, ``detail``, ``offending`` and
    ``remedy`` are machine-readable. ``detail`` is the namespaced ``mnar_sensitivity.<slot>``:
    ``mechanism_not_declared``, ``unsupported_mechanism``, ``unsupported_estimator``,
    ``unbounded_range``, ``range_excludes_mar``, ``composition_not_licensed``, ``invalid_data``,
    ``invalid_loading``, ``invalid_level``, ``invalid_decision``, ``invalid_tolerance``,
    ``insufficient_observed`` or ``bounds_exceeded``. A refusal with code ``invalid_argument``
    is also a :class:`~antecedent.errors.CausalValueError`.
    """


_real = partial(real, type_message="{what} must be a number")


@dataclass(frozen=True, slots=True)
class MnarSamplingInterval:
    """The delta-method sampling interval of the effect at one parameter value.

    Conditional on the parameter; distinct from the assumption range.
    """

    standard_error: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class MnarPoint:
    """The effect at one value of the sensitivity parameter."""

    parameter: float
    treated_mean: float
    control_mean: float
    effect: float
    sampling: MnarSamplingInterval
    acts: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class MnarTippingSide:
    """The first departure on one side of MAR at which a conclusion changes."""

    parameter: float
    bracket_lower: float
    iterations: int
    changes_to: int


@dataclass(frozen=True, slots=True)
class MnarTipping:
    """The tipping points of one conclusion (``sign``, ``excludes_null`` or a decision).

    ``at_mar`` is the conclusion at parameter 0: the sign of the effect for ``sign``; ``-1``,
    ``0`` or ``1`` for a sampling interval entirely below, including or entirely above the null
    for ``excludes_null``; ``1`` when a decision acts and ``0`` when it does not.
    """

    kind: str
    label: str
    threshold: float
    at_mar: int
    negative: MnarTippingSide | None
    positive: MnarTippingSide | None


@dataclass(frozen=True, slots=True)
class MnarDecisionResult:
    """A declared decision's behavior across the range."""

    name: str
    threshold: float
    acts_at_mar: bool
    changes_within_range: bool
    tipping: MnarTipping


@dataclass(frozen=True, slots=True)
class MnarAssumptionRange:
    """The envelope of the point effects over the declared range: an assumption range.

    ``is_confidence_interval`` and ``empirically_established`` are always ``False``.
    """

    parameter_lower: float
    parameter_upper: float
    effect_lower: float
    effect_upper: float
    is_confidence_interval: bool
    empirically_established: bool


@dataclass(frozen=True, slots=True)
class MnarAssumption:
    """An assumption the result rests on and whether the data can test it."""

    id: str
    statement: str
    status: str


@dataclass(frozen=True, slots=True)
class MnarResult:
    """The effect as a function of the sensitivity parameter, with its tipping points."""

    family: str
    target: str
    model: str
    loading: tuple[float, float]
    estimator: str
    mar: MnarPoint
    observed_case_effect: float
    rows: int
    treated_rows: int
    treated_missing: int
    control_missing: int
    grid: tuple[MnarPoint, ...]
    assumption_range: MnarAssumptionRange
    tipping: tuple[MnarTipping, ...]
    decisions: tuple[MnarDecisionResult, ...]
    level: float
    tolerance: float
    assumptions: tuple[MnarAssumption, ...]
    method: str
    interpretation: str
    inference_claim: str
    empirical_status: str
    sampling_basis: str

    def tipping_point(self, label: str = "sign") -> MnarTipping:
        """The tipping points of the conclusion named ``label``.

        ``"sign"``, ``"excludes_null"`` or the name of a declared decision.

        Raises:
            CausalValueError: no such conclusion was analysed.
        """
        for tipping in self.tipping:
            if tipping.label == label:
                return tipping
        known = [tipping.label for tipping in self.tipping]
        raise CausalValueError(f"no conclusion named {label!r}; analysed: {known}")


def _side(wire: Mapping[str, Any] | None) -> MnarTippingSide | None:
    if wire is None:
        return None
    return MnarTippingSide(
        parameter=float(wire["parameter"]),
        bracket_lower=float(wire["bracket_lower"]),
        iterations=int(wire["iterations"]),
        changes_to=int(wire["changes_to"]),
    )


def _tipping(wire: Mapping[str, Any]) -> MnarTipping:
    return MnarTipping(
        kind=str(wire["kind"]),
        label=str(wire["label"]),
        threshold=float(wire["threshold"]),
        at_mar=int(wire["at_mar"]),
        negative=_side(wire["negative"]),
        positive=_side(wire["positive"]),
    )


def _point(wire: Mapping[str, Any]) -> MnarPoint:
    sampling = wire["sampling"]
    return MnarPoint(
        parameter=float(wire["parameter"]),
        treated_mean=float(wire["treated_mean"]),
        control_mean=float(wire["control_mean"]),
        effect=float(wire["effect"]),
        sampling=MnarSamplingInterval(
            standard_error=float(sampling["standard_error"]),
            lower=float(sampling["lower"]),
            upper=float(sampling["upper"]),
        ),
        acts=tuple(bool(a) for a in wire["acts"]),
    )


def _result(wire: Mapping[str, Any]) -> MnarResult:
    rng = wire["assumption_range"]
    return MnarResult(
        family=str(wire["family"]),
        target=str(wire["target"]),
        model=str(wire["model"]),
        loading=(float(wire["loading"]["treated"]), float(wire["loading"]["control"])),
        estimator=str(wire["estimator"]),
        mar=_point(wire["mar"]),
        observed_case_effect=float(wire["observed_case_effect"]),
        rows=int(wire["rows"]),
        treated_rows=int(wire["treated_rows"]),
        treated_missing=int(wire["treated_missing"]),
        control_missing=int(wire["control_missing"]),
        grid=tuple(_point(p) for p in wire["grid"]),
        assumption_range=MnarAssumptionRange(
            parameter_lower=float(rng["parameter_lower"]),
            parameter_upper=float(rng["parameter_upper"]),
            effect_lower=float(rng["effect_lower"]),
            effect_upper=float(rng["effect_upper"]),
            is_confidence_interval=bool(rng["is_confidence_interval"]),
            empirically_established=bool(rng["empirically_established"]),
        ),
        tipping=tuple(_tipping(t) for t in wire["tipping"]),
        decisions=tuple(
            MnarDecisionResult(
                name=str(d["name"]),
                threshold=float(d["threshold"]),
                acts_at_mar=bool(d["acts_at_mar"]),
                changes_within_range=bool(d["changes_within_range"]),
                tipping=_tipping(d["tipping"]),
            )
            for d in wire["decisions"]
        ),
        level=float(wire["level"]),
        tolerance=float(wire["tolerance"]),
        assumptions=tuple(
            MnarAssumption(str(a["id"]), str(a["statement"]), str(a["status"]))
            for a in wire["assumptions"]
        ),
        method=str(wire["method"]),
        interpretation=str(wire["interpretation"]),
        inference_claim=str(wire["inference_claim"]),
        empirical_status=str(wire["empirical_status"]),
        sampling_basis=str(wire["sampling_basis"]),
    )


def _column(values: object, name: str) -> list[Any]:
    if isinstance(values, (str, bytes)):
        raise CausalTypeError(f"{name} must be a sequence of numbers")
    if hasattr(values, "tolist"):
        return list(values.tolist())
    if not isinstance(values, Sequence):
        raise CausalTypeError(f"{name} must be a sequence of numbers")
    return list(values)


def _treatment(values: object) -> list[float]:
    out: list[float] = []
    for value in _column(values, "treatment"):
        if isinstance(value, bool):
            out.append(1.0 if value else 0.0)
        else:
            out.append(_real(value, "treatment"))
    return out


def _outcome(values: object) -> list[float | None]:
    out: list[float | None] = []
    for value in _column(values, "outcome"):
        if value is None:
            out.append(None)
            continue
        number = _real(value, "outcome")
        out.append(None if math.isnan(number) else number)
    return out


def _covariate_columns(covariates: Any) -> list[list[float]]:
    if covariates is None:
        return []
    columns = list(covariates.values()) if isinstance(covariates, Mapping) else covariates
    if isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence):
        raise CausalTypeError("covariates must be a sequence of columns or a mapping of columns")
    return [[_real(v, "covariate") for v in _column(column, "covariate")] for column in columns]


def _loading(loading: str | tuple[float, float]) -> tuple[float, float]:
    if isinstance(loading, str):
        if loading not in _LOADINGS:
            raise CausalValueError(f"loading must be one of {sorted(_LOADINGS)} or a pair")
        return _LOADINGS[loading]
    if not isinstance(loading, Sequence) or len(loading) != 2:
        raise CausalTypeError("loading must be a name or a (treated, control) pair")
    return (_real(loading[0], "loading"), _real(loading[1], "loading"))


def _decisions(
    decisions: Mapping[str, float] | Sequence[tuple[str, float]] | None,
) -> list[tuple[str, float]]:
    if decisions is None:
        return []
    items = list(decisions.items()) if isinstance(decisions, Mapping) else list(decisions)
    out: list[tuple[str, float]] = []
    for item in items:
        if not (isinstance(item, Sequence) and len(item) == 2 and isinstance(item[0], str)):
            raise CausalTypeError("decisions must map names to thresholds")
        out.append((item[0], _real(item[1], "decision threshold")))
    return out


def mnar_ate_sensitivity(
    treatment: Any,
    outcome: Any,
    *,
    model: str | None,
    lower: float,
    upper: float,
    loading: str | tuple[float, float] = "common",
    estimator: str = "difference_in_means",
    grid_points: int = _DEFAULT_GRID_POINTS,
    level: float = _DEFAULT_LEVEL,
    decisions: Mapping[str, float] | Sequence[tuple[str, float]] | None = None,
    tolerance: float = _DEFAULT_TOLERANCE,
    covariates: Sequence[Any] | Mapping[str, Any] | None = None,
    bootstrap_replicates: int = 200,
    seed: int = 0,
    sampling_composition: str | None = None,
) -> MnarResult:
    """Re-run the arm-mean contrast across a bounded grid of an MNAR departure.

    ``treatment`` is 0/1 with no missing entry; ``outcome`` is numeric with ``None`` or NaN for a
    missing outcome. ``model`` declares the outcome-missingness departure and is required:
    ``"delta_pattern_mixture"`` shifts each arm's missing outcomes by ``loading * delta`` from
    that arm's observed mean (the parameter is in outcome units), and
    ``"exponential_tilting_selection"`` tilts the arm's observed outcome law by
    ``exp(loading * theta * y)`` (the parameter is per outcome unit). ``loading`` is
    ``"common"``, ``"treated_only"``, ``"control_only"``, ``"opposite"`` or a
    ``(treated, control)`` pair in ``[-1, 1]``. ``lower`` and ``upper`` bound the parameter
    range, which must be finite and contain ``0`` (MAR); the grid has ``grid_points``
    equally spaced values (2 to 4096). ``decisions`` maps a decision name to the effect
    threshold above which it acts. ``level`` is the level of the per-point sampling interval and
    ``tolerance`` the tipping bracket width relative to the range.

    ``estimator`` is ``"difference_in_means"`` (no covariates), or an adjusted estimator that
    reads ``covariates`` (a sequence of columns, or a mapping of name to column):
    ``"regression_adjusted"``, ``"ipw"`` or ``"aipw"``. Each is re-run at every grid point on
    the table completed by the departure: a missing outcome is imputed from the arm's outcome
    regression plus the shift, or, for the selection model, plus the mean residual under the
    tilted weights. An adjusted estimator's per-point sampling interval is a seeded bootstrap
    (``bootstrap_replicates`` 20 to 5000, stream ``seed``) that repeats the imputation fit.
    Any other estimator refuses (``unsupported_estimator``).

    ``sampling_composition`` requests composing the sampling interval with the assumption range
    by a named method; no method is licensed, so any value refuses
    (``cell_not_licensed`` / ``mnar_sensitivity.composition_not_licensed``).

    Raises:
        CausalTypeError: an argument has the wrong type.
        MnarSensitivityRefusal: the analysis is refused: no declared mechanism
            (``mechanism_not_declared``), an unsupported model, estimator (for example an
            adjusted or doubly robust one: ``unsupported_estimator``), an unbounded or
            MAR-excluding range, malformed data or options, or exceeded bounds.
    """
    if model is not None and not isinstance(model, str):
        raise CausalTypeError("model must be a string or None")
    if not isinstance(estimator, str):
        raise CausalTypeError("estimator must be a string")
    if isinstance(grid_points, bool) or not isinstance(grid_points, int):
        raise CausalTypeError("grid_points must be an integer")
    if grid_points < 0:
        raise CausalValueError("grid_points must be non-negative")
    if sampling_composition is not None and not isinstance(sampling_composition, str):
        raise CausalTypeError("sampling_composition must be a string or None")
    treated_loading, control_loading = _loading(loading)
    for name, value in (("bootstrap_replicates", bootstrap_replicates), ("seed", seed)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CausalTypeError(f"{name} must be a non-negative integer")
    text, refusal = _run(
        _treatment(treatment),
        _outcome(outcome),
        model,
        treated_loading,
        control_loading,
        estimator,
        _real(lower, "lower"),
        _real(upper, "upper"),
        grid_points,
        _real(level, "level"),
        _decisions(decisions),
        _real(tolerance, "tolerance"),
        _covariate_columns(covariates),
        bootstrap_replicates,
        seed,
        sampling_composition,
    )
    raise_refusal(refusal, MnarSensitivityRefusal)
    if text is None:  # pragma: no cover - the native contract
        raise CausalValueError(
            "the native MNAR sensitivity returned neither a result nor a refusal"
        )
    return _result(json.loads(text))


def from_analysis(
    result: Any,
    *,
    data: Any,
    model: str | None,
    lower: float,
    upper: float,
    estimator: str | None = None,
    **options: Any,
) -> MnarResult:
    """The sensitivity of a checked binary-treatment ``AverageEffect`` result's own estimand.

    The result supplies the query (treatment, outcome, control and active levels) and the
    checked adjustment set; ``data`` is the table whose outcome column holds the missing
    values (``None`` or NaN). With a non-empty adjustment set the default estimator is
    ``"aipw"`` on the adjustment set as covariates; with an empty one it is
    ``"difference_in_means"``. This is the route a ``missing_not_at_random`` imputation
    declaration is refused toward. Remaining ``options`` are those of
    :func:`mnar_ate_sensitivity`.

    Raises:
        CausalTypeError: ``result`` is not an ``AverageEffect`` analysis result.
        CausalValueError: the data lack a required column or the treatment is not coded by the
            query's two levels.
    """
    from ._data import as_columns
    from .outcomes import require_success
    from .query import AverageEffect
    from .results import AnalysisResult

    result = require_success(result)
    if not isinstance(result, AnalysisResult) or not isinstance(result.query, AverageEffect):
        raise CausalTypeError("MNAR from_analysis requires an AverageEffect analysis result")
    if not result.identification:
        raise CausalValueError("MNAR from_analysis requires a point-identified adjustment query")
    query = result.query
    adjustment = tuple(result.identification.adjustment_set)
    names, columns = as_columns(data)
    table = dict(zip(names, columns, strict=True))
    required = (query.treatment, query.outcome, *adjustment)
    if any(name not in table for name in required):
        raise CausalValueError(f"data require columns {required!r}")
    levels = {float(query.control_level): 0.0, float(query.active_level): 1.0}
    try:
        treatment = [levels[float(v)] for v in _column(table[query.treatment], "treatment")]
    except KeyError as error:
        raise CausalValueError(
            "the treatment must take only the query's control and active levels"
        ) from error
    return mnar_ate_sensitivity(
        treatment,
        table[query.outcome],
        model=model,
        lower=lower,
        upper=upper,
        estimator=estimator or ("aipw" if adjustment else "difference_in_means"),
        covariates=[table[name] for name in adjustment] or None,
        **options,
    )
