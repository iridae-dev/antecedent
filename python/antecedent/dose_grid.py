"""Dose-grid functional row: a named functional of a randomized-dose response curve.

One complete response-grid row. For a continuous dose ``D`` and outcome ``Y`` of a
*randomized* dose design, the dose response is ``m(d) = E[Y(d)] = E[Y | D = d]``. Exactly one
named functional is requested per call:

* ``"level"``: ``m(d)`` at each grid dose;
* ``"derivative"``: ``m'(d)`` at each grid dose;
* ``"contrast"``: ``m(to) - m(from)`` between two named doses.

The fitted object is the Gaussian-kernel local quadratic smoother at the *declared* fixed
bandwidth (never data-selected here); the functional is that smoother's level, first
derivative or difference of levels. The smoothing bias ``m_h - m`` is not estimated and is not
included in any interval.

:func:`quadratic_dose_functional` adds a caller-attested exact quadratic conditional
mean premise. Polynomial reproduction then derives zero smoothing bias at the fixed
bandwidth; the premise and its required feature travel in the artifact. Sampling
calibration remains unmeasured.

Every requested dose is labelled ``supported`` / ``weak_overlap`` / ``outside_empirical_support``
(:func:`dose_support_table` shows the labels without refusing). The row *refuses* any dose that
is not supported; it never extrapolates. Intervals are pointwise normal intervals from a
heteroskedasticity-robust standard error with calibration ``"unmeasured"``: no coverage claim is
made. Level/contrast claims, derivative claims and the simultaneous band are separate
(:class:`DoseClaims`). Without a declared smoothness bound a simultaneous band refuses
(``dose_grid.simultaneous_band_closed``).

Passing ``curvature_bound`` (a declared bound ``B`` on ``|m'''|``) to :func:`dose_functional`
adds bias-aware intervals (:class:`DoseBiasAware`): each level or derivative interval is widened
by the worst-case smoothing bias the bound allows, using the folded-normal critical value, and a
level or derivative grid of two or more doses also carries a Gaussian-multiplier sup-t band over
the grid, widened by the same bias bound. The band is drawn from ``band_seed`` and replayed from it
when the artifact is consumed. Its calibration is unmeasured. The bound is a declaration, not
something the data establish.

Optional ``threads`` bounds grid workers, ``max_bytes`` bounds execution scratch, and
``cancel`` accepts a cancellation token. Independent grid fits preserve request order;
small grids use serial execution. The same controls apply during artifact recomputation.

Rows are independent: clustered, serial or linked rows are outside this row. An observational
dose is refused (``dose_grid.graph_not_certified``): this row does not adjust.

:meth:`DoseFunctionalResult.export` writes an artifact embedding the compact dose/outcome table;
:func:`consume_dose_grid_artifact` recomputes the local-quadratic fit from it and refuses any
change, even one resealed with fresh digests.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

import numpy as np

from ._native import consume_dose_grid_artifact as _consume
from ._native import consume_dose_grid_posterior_artifact as _consume_posterior
from ._native import dose_support_labels as _support_labels
from ._native import evaluate_dose_grid_functional as _evaluate
from ._native import evaluate_dose_grid_posterior as _evaluate_posterior
from ._posterior_row import PosteriorRowEstimate
from ._refusal import raise_refusal
from ._validate import real
from .errors import CausalTypeError, CausalValueError, StructuredRefusal

__all__ = [
    "DoseGridPosteriorResult",
    "DoseBiasAware",
    "DoseClaims",
    "DoseContrast",
    "DoseFunctionalResult",
    "DoseGridRefusal",
    "DosePoint",
    "DoseSupport",
    "consume_dose_grid_artifact",
    "consume_dose_grid_posterior_artifact",
    "dose_functional",
    "dose_grid_posterior",
    "quadratic_dose_functional",
    "dose_support_table",
]

Functional = Literal["level", "derivative", "contrast"]
Design = Literal["randomized_dose", "observational"]

#: Default smallest Kish effective sample size of the local weights at a requested dose.
DEFAULT_MINIMUM_LOCAL_ESS = 10.0
_FUNCTIONALS = ("level", "derivative", "contrast")
_DESIGNS = ("randomized_dose", "observational")


class DoseGridRefusal(StructuredRefusal):
    """A :class:`StructuredRefusal` for the structured Rust refusal fields.

    ``reason_code`` is the inherited, registered code; ``detail`` is the namespaced
    ``dose_grid.*`` slot (for example ``dose_grid.unsupported_dose``,
    ``dose_grid.simultaneous_band_closed`` or ``dose_grid.graph_not_certified``).
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        detail = str(refusal["detail"])
        message = refusal.get("message")
        super().__init__(refusal, text=f"{detail}: {message}" if message else detail)


def _invalid(message: str, offending: str | None = None) -> DoseGridRefusal:
    return DoseGridRefusal(
        {
            "code": "invalid_argument",
            "stage": "dose_grid",
            "detail": "dose_grid.invalid_request",
            "offending": offending,
            "message": message,
        }
    )


@dataclass(frozen=True, slots=True)
class DoseClaims:
    """Which inference claims the caller declares. Each is separate; none implies another.

    ``pointwise_level`` licenses a level or contrast, ``derivative`` licenses a derivative, and
    ``simultaneous_band`` asks for a band over the grid; it refuses unless a curvature bound is
    declared.
    """

    pointwise_level: bool = False
    derivative: bool = False
    simultaneous_band: bool = False

    def __post_init__(self) -> None:
        for name in ("pointwise_level", "derivative", "simultaneous_band"):
            if not isinstance(getattr(self, name), bool):
                raise CausalTypeError(f"{name} must be a bool")

    def _wire(self) -> dict[str, bool]:
        return {
            "pointwise_level": self.pointwise_level,
            "derivative": self.derivative,
            "simultaneous_band": self.simultaneous_band,
        }


@dataclass(frozen=True, slots=True)
class DoseSupport:
    """The support label of one requested dose."""

    dose: float
    label: str
    local_ess: float


@dataclass(frozen=True, slots=True)
class DosePoint:
    """A pointwise level ``m(d)`` or derivative ``m'(d)`` at one dose.

    ``lower`` / ``upper`` are a pointwise normal interval at ``nominal_level`` with
    ``calibration == "unmeasured"``: the nominal level is not a coverage claim, and the
    smoothing bias is not included.
    """

    dose: float
    support: str
    value: float
    standard_error: float
    local_ess: float
    lower: float
    upper: float
    nominal_level: float
    calibration: str
    smoothing_bias_included: bool


@dataclass(frozen=True, slots=True)
class DoseContrast:
    """A named contrast ``m(to) - m(from)`` with its covariance-aware standard error."""

    from_dose: float
    to_dose: float
    from_support: str
    to_support: str
    estimate: float
    standard_error: float
    lower: float
    upper: float
    nominal_level: float
    calibration: str
    smoothing_bias_included: bool


@dataclass(frozen=True, slots=True)
class DoseBiasAware:
    """Bias-aware intervals and the optional band under a declared curvature bound.

    ``smoothing_bias_bound``, ``lower`` and ``upper`` align with the reported coordinates (the
    levels, the derivatives, or the one contrast). The band fields are ``None`` unless a level or
    derivative grid of two or more doses was requested. Calibration is unmeasured.
    """

    curvature_bound: float
    smoothing_bias_bound: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    band_lower: tuple[float, ...] | None
    band_upper: tuple[float, ...] | None
    band_critical_value: float | None
    band_replicates: int

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready mapping."""
        return {
            "curvature_bound": self.curvature_bound,
            "smoothing_bias_bound": list(self.smoothing_bias_bound),
            "lower": list(self.lower),
            "upper": list(self.upper),
            "band_lower": None if self.band_lower is None else list(self.band_lower),
            "band_upper": None if self.band_upper is None else list(self.band_upper),
            "band_critical_value": self.band_critical_value,
            "band_replicates": self.band_replicates,
        }


def _bias_aware(section: Mapping[str, Any] | None) -> DoseBiasAware | None:
    if section is None:
        return None
    band_lower = section.get("band_lower")
    band_upper = section.get("band_upper")
    critical = section.get("band_critical_value")
    return DoseBiasAware(
        curvature_bound=float(section["curvature_bound"]),
        smoothing_bias_bound=tuple(float(x) for x in section["smoothing_bias_bound"]),
        lower=tuple(float(x) for x in section["lower"]),
        upper=tuple(float(x) for x in section["upper"]),
        band_lower=None if band_lower is None else tuple(float(x) for x in band_lower),
        band_upper=None if band_upper is None else tuple(float(x) for x in band_upper),
        band_critical_value=None if critical is None else float(critical),
        band_replicates=int(section["band_replicates"]),
    )


@dataclass(frozen=True, slots=True)
class DoseFunctionalResult:
    """One named functional of the dose response, its support, claims and artifact."""

    functional: str
    design: str
    n_rows: int
    bandwidth: float
    levels: tuple[DosePoint, ...]
    derivatives: tuple[DosePoint, ...]
    contrast: DoseContrast | None
    support: tuple[DoseSupport, ...]
    support_status: str
    fits: int
    max_abs_influence_sum: float
    calibration: str
    smoothing_bias_included: bool
    #: ``"closed"``, or ``"bias_aware"`` when a curvature bound produced a band.
    simultaneous_band: str
    inference_claim: str
    claims: DoseClaims
    bandwidth_range: tuple[float, float]
    minimum_local_ess: float
    premises_digest: str
    data_digest: str
    artifact: bytes = field(repr=False, compare=False)
    #: Caller attests an exact quadratic conditional mean; smoothing bias is zero under it.
    quadratic_mean: bool = False
    #: Bias-aware intervals and band under a declared curvature bound, when one was declared.
    bias_aware: DoseBiasAware | None = None

    def export(self) -> bytes:
        """The artifact: premises, compact dose/outcome table, result and both digests."""
        return self.artifact

    @property
    def support_table(self) -> tuple[DoseSupport, ...]:
        """The per-dose support labels, in request order."""
        return self.support

    def explain(self) -> str:
        """What was estimated, how well supported it is, and which claims are not made."""
        text = (
            f"Dose-response functional {self.functional!r} ({self.design}) from {self.n_rows} "
            f"rows at bandwidth {self.bandwidth:g}: worst support {self.support_status} over "
            f"{len(self.support)} dose(s), smallest local effective sample size "
            f"{self.minimum_local_ess:.3g}."
        )
        if self.contrast is not None:
            c = self.contrast
            text += (
                f" Contrast m({c.to_dose:g}) - m({c.from_dose:g}) = {c.estimate:.4g} "
                f"(pointwise standard error {c.standard_error:.4g})."
            )
        text += (
            f" Pointwise intervals are nominal only (calibration {self.calibration}, smoothing "
            f"bias {'included' if self.smoothing_bias_included else 'not included'}); the "
            f"simultaneous band is {self.simultaneous_band}; inference claim "
            f"{self.inference_claim!r}."
        )
        if self.bias_aware is not None:
            text += (
                f" Under the declared bound {self.bias_aware.curvature_bound:g} on |m'''|, "
                f"bias-aware intervals add a worst-case smoothing bias of up to "
                f"{max(self.bias_aware.smoothing_bias_bound, default=0.0):.4g}"
            )
            if self.bias_aware.band_critical_value is not None:
                text += (
                    f" and a simultaneous band uses critical value "
                    f"{self.bias_aware.band_critical_value:.4g}"
                )
            text += " (calibration unmeasured)."
        return text

    def __repr__(self) -> str:
        return (
            f"<DoseFunctionalResult {self.functional} n={self.n_rows} "
            f"support={self.support_status} claim={self.inference_claim}>"
        )

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready mapping of the result (the table is not repeated)."""
        return {
            "functional": self.functional,
            "design": self.design,
            "n_rows": self.n_rows,
            "bandwidth": self.bandwidth,
            "levels": [_point_dict(p) for p in self.levels],
            "derivatives": [_point_dict(p) for p in self.derivatives],
            "contrast": None
            if self.contrast is None
            else {
                "from": self.contrast.from_dose,
                "to": self.contrast.to_dose,
                "estimate": self.contrast.estimate,
                "standard_error": self.contrast.standard_error,
                "lower": self.contrast.lower,
                "upper": self.contrast.upper,
                "calibration": self.contrast.calibration,
            },
            "support": [
                {"dose": s.dose, "label": s.label, "local_ess": s.local_ess} for s in self.support
            ],
            "support_status": self.support_status,
            "calibration": self.calibration,
            "simultaneous_band": self.simultaneous_band,
            "inference_claim": self.inference_claim,
            "quadratic_mean": self.quadratic_mean,
            "bias_aware": None if self.bias_aware is None else self.bias_aware.to_dict(),
        }


def _point_dict(point: DosePoint) -> dict[str, Any]:
    return {
        "dose": point.dose,
        "support": point.support,
        "value": point.value,
        "standard_error": point.standard_error,
        "local_ess": point.local_ess,
        "lower": point.lower,
        "upper": point.upper,
        "calibration": point.calibration,
    }


def _column(values: object, what: str) -> list[float]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CausalTypeError(f"{what} must be a numeric sequence") from error
    if array.ndim != 1:
        raise CausalValueError(f"{what} must be one-dimensional")
    return array.tolist()


def _non_finite(what: str) -> DoseGridRefusal:
    return _invalid(f"{what} is not finite", what)


_number = partial(real, finite=True, non_finite=_non_finite)


def _finite_list(values: list[float], what: str) -> list[float]:
    if not all(math.isfinite(x) for x in values):
        raise _invalid(f"{what} has a non-finite entry", what)
    return values


def _point(item: Mapping[str, Any]) -> DosePoint:
    return DosePoint(
        dose=float(item["dose"]),
        support=item["support"],
        value=float(item["value"]),
        standard_error=float(item["standard_error"]),
        local_ess=float(item["local_ess"]),
        lower=float(item["lower"]),
        upper=float(item["upper"]),
        nominal_level=float(item["nominal_level"]),
        calibration=item["calibration"],
        smoothing_bias_included=bool(item["smoothing_bias_included"]),
    )


def _result(report_json: str, artifact: bytes) -> DoseFunctionalResult:
    report = json.loads(report_json)
    request = report["request"]
    result = report["result"]
    contrast = result["contrast"]
    return DoseFunctionalResult(
        functional=result["functional"],
        design=result["design"],
        n_rows=int(result["n_rows"]),
        bandwidth=float(result["bandwidth"]),
        levels=tuple(_point(p) for p in result["levels"]),
        derivatives=tuple(_point(p) for p in result["derivatives"]),
        contrast=None
        if contrast is None
        else DoseContrast(
            from_dose=float(contrast["from"]),
            to_dose=float(contrast["to"]),
            from_support=contrast["from_support"],
            to_support=contrast["to_support"],
            estimate=float(contrast["estimate"]),
            standard_error=float(contrast["standard_error"]),
            lower=float(contrast["lower"]),
            upper=float(contrast["upper"]),
            nominal_level=float(contrast["nominal_level"]),
            calibration=contrast["calibration"],
            smoothing_bias_included=bool(contrast["smoothing_bias_included"]),
        ),
        support=tuple(
            DoseSupport(dose=float(s["dose"]), label=s["label"], local_ess=float(s["local_ess"]))
            for s in result["support"]
        ),
        support_status=result["support_status"],
        fits=int(result["fits"]),
        max_abs_influence_sum=float(result["max_abs_influence_sum"]),
        calibration=result["calibration"],
        smoothing_bias_included=bool(result["smoothing_bias_included"]),
        simultaneous_band=result["simultaneous_band"],
        inference_claim=result["inference_claim"],
        claims=DoseClaims(**request["claims"]),
        bandwidth_range=(
            float(request["bandwidth_range"][0]),
            float(request["bandwidth_range"][1]),
        ),
        minimum_local_ess=float(request["minimum_local_ess"]),
        premises_digest=report["premises_digest"],
        data_digest=report["data_digest"],
        artifact=artifact,
        quadratic_mean=bool(request.get("quadratic_mean", False)),
        bias_aware=_bias_aware(result.get("bias_aware")),
    )


def _dose_functional(
    dose: Sequence[float] | np.ndarray,
    outcome: Sequence[float] | np.ndarray,
    *,
    bandwidth: float,
    bandwidth_range: tuple[float, float],
    functional: Functional = "level",
    grid: Sequence[float] | None = None,
    contrast: tuple[float, float] | None = None,
    minimum_local_ess: float = DEFAULT_MINIMUM_LOCAL_ESS,
    claims: DoseClaims | None = None,
    design: Design = "randomized_dose",
    quadratic_mean: bool = False,
    curvature_bound: float | None = None,
    band_replicates: int = 2000,
    band_seed: int = 1,
    threads: int | None = None,
    max_bytes: int | None = None,
    cancel: Any = None,
) -> DoseFunctionalResult:
    """Estimate one named functional of a randomized-dose response curve.

    ``dose`` and ``outcome`` are aligned finite rows (independent; at least three).
    ``functional`` selects ``"level"`` (``m(d)`` on ``grid``), ``"derivative"`` (``m'(d)`` on
    ``grid``) or ``"contrast"`` (``m(to) - m(from)`` for ``contrast=(from, to)``; no grid).
    ``bandwidth`` is the fixed Gaussian bandwidth and must lie in the caller-declared inclusive
    ``bandwidth_range``; it is never selected from the data. ``minimum_local_ess`` is the
    smallest admissible Kish effective sample size at any requested dose. ``claims`` defaults to
    the one pointwise claim the functional needs (a derivative claim for ``"derivative"``, a
    pointwise-level claim otherwise); a level without a level claim, a derivative without a
    derivative claim and a simultaneous band without ``curvature_bound`` refuse.
    ``curvature_bound`` declares a bound on ``|m'''|`` and adds :class:`DoseBiasAware` intervals
    and, on a grid of two or more doses, a band of ``band_replicates`` multiplier draws seeded by
    ``band_seed``.

    Raises :class:`DoseGridRefusal` (a :class:`~antecedent.errors.CausalUnsupportedError`) with
    ``detail`` one of ``dose_grid.graph_not_certified`` (observational design),
    ``dose_grid.simultaneous_band_closed``, ``dose_grid.derivative_without_claim``,
    ``dose_grid.level_without_claim``, ``dose_grid.invalid_request``,
    ``dose_grid.bandwidth_outside_range``, ``dose_grid.unsupported_dose`` (outside the observed
    dose range) or ``dose_grid.insufficient_local_weight`` (too little local weight).
    """
    if functional not in _FUNCTIONALS:
        raise CausalValueError(f"functional must be one of {_FUNCTIONALS}")
    if design not in _DESIGNS:
        raise CausalValueError(f"design must be one of {_DESIGNS}")
    if claims is None:
        claims = DoseClaims(
            pointwise_level=functional != "derivative", derivative=functional == "derivative"
        )
    elif not isinstance(claims, DoseClaims):
        raise CausalTypeError("claims must be a DoseClaims")
    if functional == "contrast":
        if grid is not None or contrast is None:
            raise CausalValueError("a contrast takes contrast=(from, to) and no grid")
        if len(contrast) != 2:
            raise CausalValueError("contrast must be a (from, to) pair")
        spec: dict[str, Any] = {
            "kind": "contrast",
            "from": _number(contrast[0], "contrast[0]"),
            "to": _number(contrast[1], "contrast[1]"),
        }
        points: list[float] = []
    else:
        if contrast is not None or grid is None:
            raise CausalValueError("a level or derivative takes a grid and no contrast")
        points = _finite_list(_column(grid, "grid"), "grid")
        spec = {"kind": functional, "from": None, "to": None}
    if len(bandwidth_range) != 2:
        raise CausalValueError("bandwidth_range must be a (low, high) pair")
    low = _number(bandwidth_range[0], "bandwidth_range")
    high = _number(bandwidth_range[1], "bandwidth_range")
    request = {
        "design": design,
        "functional": spec,
        "grid": points,
        "bandwidth": _number(bandwidth, "bandwidth"),
        "bandwidth_range": [low, high],
        "minimum_local_ess": _number(minimum_local_ess, "minimum_local_ess"),
        "claims": claims._wire(),
        "dose": [],
        "outcome": [],
    }
    if quadratic_mean:
        request["quadratic_mean"] = True
    if curvature_bound is not None:
        if isinstance(band_replicates, bool) or not isinstance(band_replicates, int):
            raise CausalTypeError("band_replicates must be an int")
        if isinstance(band_seed, bool) or not isinstance(band_seed, int):
            raise CausalTypeError("band_seed must be an int")
        if band_replicates < 0 or band_seed < 0:
            raise CausalValueError("band_replicates and band_seed must be non-negative")
        request["bias_aware"] = {
            "curvature_bound": _number(curvature_bound, "curvature_bound"),
            "band_replicates": band_replicates,
            "band_seed": band_seed,
        }
    report, artifact, refusal = _evaluate(
        json.dumps(request, allow_nan=False),
        _column(dose, "dose"),
        _column(outcome, "outcome"),
        threads=threads,
        max_bytes=max_bytes,
        cancel=cancel,
    )
    raise_refusal(refusal, DoseGridRefusal)
    if report is None or artifact is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native dose-grid row returned neither a result nor a refusal")
    return _result(report, bytes(artifact))


def dose_functional(
    dose: Sequence[float] | np.ndarray,
    outcome: Sequence[float] | np.ndarray,
    *,
    bandwidth: float,
    bandwidth_range: tuple[float, float],
    functional: Functional = "level",
    grid: Sequence[float] | None = None,
    contrast: tuple[float, float] | None = None,
    minimum_local_ess: float = DEFAULT_MINIMUM_LOCAL_ESS,
    claims: DoseClaims | None = None,
    design: Design = "randomized_dose",
    curvature_bound: float | None = None,
    band_replicates: int = 2000,
    band_seed: int = 1,
    threads: int | None = None,
    max_bytes: int | None = None,
    cancel: Any = None,
) -> DoseFunctionalResult:
    """Estimate a randomized-dose smoother functional.

    Level, derivative and contrast claims, fixed bandwidth and support refusals are
    described in the module documentation. Without ``curvature_bound`` the smoothing bias
    is not included; with it, :attr:`DoseFunctionalResult.bias_aware` carries bias-aware
    intervals and, on a grid of two or more doses, a simultaneous band. Sampling calibration
    remains unmeasured.
    """
    return _dose_functional(
        dose,
        outcome,
        bandwidth=bandwidth,
        bandwidth_range=bandwidth_range,
        functional=functional,
        grid=grid,
        contrast=contrast,
        minimum_local_ess=minimum_local_ess,
        claims=claims,
        design=design,
        curvature_bound=curvature_bound,
        band_replicates=band_replicates,
        band_seed=band_seed,
        threads=threads,
        max_bytes=max_bytes,
        cancel=cancel,
    )


def quadratic_dose_functional(
    dose: Sequence[float] | np.ndarray,
    outcome: Sequence[float] | np.ndarray,
    *,
    bandwidth: float,
    bandwidth_range: tuple[float, float],
    functional: Functional = "level",
    grid: Sequence[float] | None = None,
    contrast: tuple[float, float] | None = None,
    minimum_local_ess: float = DEFAULT_MINIMUM_LOCAL_ESS,
    claims: DoseClaims | None = None,
    design: Design = "randomized_dose",
    threads: int | None = None,
    max_bytes: int | None = None,
    cancel: Any = None,
) -> DoseFunctionalResult:
    """Attest an exact quadratic conditional mean and estimate its named functional.

    By calling this function the caller declares E[Y|D=d] = beta0 + beta1*d + beta2*d²
    throughout the randomized dose support. The local quadratic reproduces that mean
    at every fixed bandwidth, so smoothing bias is zero under this explicit premise.
    The observed residuals do not establish the premise. Sampling uncertainty uses
    the same residual sandwich and remains unmeasured; no coverage claim is made.
    All ordinary support, claim, independence and bandwidth restrictions still apply.
    The premise and its required feature are bound into the replayable artifact.
    """
    return _dose_functional(
        dose,
        outcome,
        bandwidth=bandwidth,
        bandwidth_range=bandwidth_range,
        functional=functional,
        grid=grid,
        contrast=contrast,
        minimum_local_ess=minimum_local_ess,
        claims=claims,
        design=design,
        threads=threads,
        max_bytes=max_bytes,
        cancel=cancel,
        quadratic_mean=True,
    )


def dose_support_table(
    dose: Sequence[float] | np.ndarray,
    points: Sequence[float],
    *,
    bandwidth: float,
    minimum_local_ess: float = DEFAULT_MINIMUM_LOCAL_ESS,
) -> tuple[DoseSupport, ...]:
    """Label every requested dose without refusing.

    One :class:`DoseSupport` per entry of ``points``, in order: ``outside_empirical_support``
    outside the observed dose range, ``weak_overlap`` when the Kish effective sample size of the
    Gaussian local weights is below ``minimum_local_ess``, else ``supported``.
    :func:`dose_functional` refuses every dose that is not ``supported``.
    """
    labels, refusal = _support_labels(
        _column(dose, "dose"),
        _finite_list(_column(points, "points"), "points"),
        _number(bandwidth, "bandwidth"),
        _number(minimum_local_ess, "minimum_local_ess"),
    )
    raise_refusal(refusal, DoseGridRefusal)
    if labels is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native support labeller returned neither labels nor a refusal")
    return tuple(
        DoseSupport(dose=float(s["dose"]), label=s["label"], local_ess=float(s["local_ess"]))
        for s in json.loads(labels)
    )


def consume_dose_grid_artifact(
    artifact: bytes,
    *,
    max_rows: int = 100_000,
    max_grid: int = 1_024,
    threads: int | None = None,
    max_bytes: int | None = None,
    cancel: Any = None,
) -> DoseFunctionalResult:
    """Recompute an exported dose-grid artifact and accept only an identical one.

    The Gaussian local-quadratic fit is recomputed from the embedded dose/outcome table and
    every stored value (support labels, levels, derivatives or contrast, standard errors,
    intervals, numerical residual) must reproduce bit for bit. A changed design, functional,
    grid, bandwidth, claim, table entry or stored value is refused
    (:class:`DoseGridRefusal`, ``invalid_argument``, ``dose_grid.premises_mismatch``,
    ``.data_identity_mismatch`` or ``.result_replay_mismatch``) even when the digests were
    resealed. Corruption and unknown versions raise
    :class:`~antecedent.errors.CausalSerializationError`; a stored table above ``max_rows`` or
    ``max_grid`` refuses (``dose_grid.consumer_limit_exceeded``).
    """
    if not isinstance(artifact, bytes | bytearray | memoryview):
        raise CausalTypeError("artifact must be bytes")
    data = bytes(artifact)
    report, refusal = _consume(
        data,
        max_rows=max_rows,
        max_grid=max_grid,
        threads=threads,
        max_bytes=max_bytes,
        cancel=cancel,
    )
    raise_refusal(refusal, DoseGridRefusal)
    if report is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native consumer returned neither a result nor a refusal")
    return _result(report, data)


# --- the unlicensed Bayesian counterpart of the same randomized-dose row -------------------------


@dataclass(frozen=True, slots=True)
class DoseGridPosteriorResult:
    """The joint posterior of one named functional of a randomized-dose response curve.

    The target is the dose response ``m`` itself (no bandwidth is part of the estimand), read
    through the same ``level`` / ``derivative`` / ``contrast`` functionals as
    :func:`dose_functional`. ``posterior`` carries the posterior mean and equal-tailed 95%
    credible interval of every evaluation dose and the covariance across them, from the joint
    draws; the exact Gaussian moments are added when the residual variance is known. The row is
    unlicensed and its calibration is ``"unmeasured"``.
    """

    functional: str
    design: str
    posterior: PosteriorRowEstimate
    request: Mapping[str, Any] = field(repr=False, compare=False)
    premises_digest: str
    data_digest: str
    artifact: bytes = field(repr=False, compare=False)
    execution_facts: Any = field(default=None, repr=False, compare=False)

    def export(self) -> bytes:
        """The artifact: premises, compact dose/outcome table, posterior and both digests."""
        return self.artifact


def _posterior_result(report_json: str, artifact: bytes, execution_facts: Any = None) -> DoseGridPosteriorResult:
    report = json.loads(report_json)
    request = report["request"]["dose_grid"]
    return DoseGridPosteriorResult(
        functional=str(request["functional"]["kind"]),
        design=str(request["design"]),
        posterior=PosteriorRowEstimate._from_mapping(report["result"]),
        request=request,
        premises_digest=report["premises_digest"],
        data_digest=report["data_digest"],
        artifact=artifact,
        execution_facts=execution_facts,
    )


def dose_grid_posterior(
    dose: Sequence[float] | np.ndarray,
    outcome: Sequence[float] | np.ndarray,
    *,
    support: tuple[float, float],
    coefficient_scales: Sequence[float],
    residual_variance: float | None = None,
    residual_shape: float | None = None,
    residual_scale: float | None = None,
    functional: Functional = "level",
    grid: Sequence[float] | None = None,
    contrast: tuple[float, float] | None = None,
    draws: int = 4_000,
    seed: int = 0,
    design: Design = "randomized_dose",
    threads: int | None = None,
    max_bytes: int | None = None,
    cancel: Any = None,
) -> DoseGridPosteriorResult:
    """Joint Bayesian posterior of a named functional of a randomized-dose response curve.

    The likelihood is ``y = m(d) + e`` with independent ``N(0, sigma^2)`` errors; ``m`` is a
    polynomial in the dose rescaled to ``[-1, 1]`` on the declared ``support`` with coefficient
    ``j`` a priori ``N(0, sigma^2 s_j^2)`` for ``s = coefficient_scales`` (a decaying scale
    vector is the smoothing prior and is part of the answer). Declare the residual variance
    (``residual_variance``) or its inverse-gamma prior (``residual_shape`` and
    ``residual_scale``). One coefficient draw gives every grid dose at once, so the covariance
    across the grid is preserved in ``posterior.covariance``.

    ``threads``, ``max_bytes`` and ``cancel`` control execution policy. Refused resources
    and cancellation raise ``CausalResourceError`` and ``CausalCancelledError``.

    Raises :class:`DoseGridRefusal` with ``detail`` one of ``dose_grid.graph_not_certified``
    (observational dose: this row does not adjust), ``dose_grid.unsupported_dose`` (a dose
    outside the observed range, never extrapolated) or ``dose_grid.invalid_request``.
    """
    if functional not in _FUNCTIONALS:
        raise CausalValueError(f"functional must be one of {_FUNCTIONALS}")
    if design not in _DESIGNS:
        raise CausalValueError(f"design must be one of {_DESIGNS}")
    if functional == "contrast":
        if grid is not None or contrast is None or len(contrast) != 2:
            raise CausalValueError("a contrast takes contrast=(from, to) and no grid")
        spec: dict[str, Any] = {
            "kind": "contrast",
            "from": _number(contrast[0], "contrast[0]"),
            "to": _number(contrast[1], "contrast[1]"),
        }
        points: list[float] = []
    else:
        if contrast is not None or grid is None:
            raise CausalValueError("a level or derivative takes a grid and no contrast")
        points = _finite_list(_column(grid, "grid"), "grid")
        spec = {"kind": functional, "from": None, "to": None}
    if residual_variance is not None and residual_shape is None and residual_scale is None:
        residual: dict[str, Any] = {
            "known_variance": _number(residual_variance, "residual_variance")
        }
    elif residual_variance is None and residual_shape is not None and residual_scale is not None:
        residual = {
            "inverse_gamma": {
                "shape": _number(residual_shape, "residual_shape"),
                "scale": _number(residual_scale, "residual_scale"),
            }
        }
    else:
        raise CausalValueError(
            "declare either residual_variance, or residual_shape and residual_scale"
        )
    request = {
        "dose_grid": {
            "design": design,
            "functional": spec,
            "grid": points,
            "support": [_number(support[0], "support"), _number(support[1], "support")],
            "coefficient_scales": _finite_list(
                _column(coefficient_scales, "coefficient_scales"), "coefficient_scales"
            ),
            "residual": residual,
            "draws": int(draws),
            "seed": int(seed),
            "dose": [],
            "outcome": [],
        }
    }
    report, artifact, refusal, facts = _evaluate_posterior(
        json.dumps(request, allow_nan=False),
        _column(dose, "dose"),
        _column(outcome, "outcome"),
        threads=threads,
        max_bytes=max_bytes,
        cancel=cancel,
    )
    try:
        raise_refusal(refusal, DoseGridRefusal)
    except DoseGridRefusal as error:
        error.execution_facts = facts
        raise
    if report is None or artifact is None:  # pragma: no cover - the native contract
        raise CausalValueError(
            "the native dose-grid posterior returned neither a result nor a refusal"
        )
    return _posterior_result(report, bytes(artifact), facts)


def consume_dose_grid_posterior_artifact(
    artifact: bytes,
    *,
    max_rows: int = 100_000,
    max_grid: int = 1_024,
    threads: int | None = None,
    max_bytes: int | None = None,
    cancel: Any = None,
) -> DoseGridPosteriorResult:
    """Recompute an exported dose-grid posterior and accept only an identical one.

    The posterior is recomputed from the embedded table with the stored seed and every stored
    value (means, intervals, joint covariance, exact moments, evaluation doses) must reproduce
    bit for bit. A changed premise, table entry or stored value is refused
    (:class:`DoseGridRefusal`, ``bayesian_row.premises_mismatch``,
    ``.data_identity_mismatch`` or ``.result_replay_mismatch``) even when the digests were
    resealed. Corruption and unknown versions raise
    :class:`~antecedent.errors.CausalSerializationError`.
    ``threads``, ``max_bytes`` and ``cancel`` apply to the independent replay.
    """
    if not isinstance(artifact, bytes | bytearray | memoryview):
        raise CausalTypeError("artifact must be bytes")
    data = bytes(artifact)
    report, refusal, facts = _consume_posterior(
        data, max_rows=max_rows, max_grid=max_grid,
        threads=threads, max_bytes=max_bytes, cancel=cancel,
    )
    try:
        raise_refusal(refusal, DoseGridRefusal)
    except DoseGridRefusal as error:
        error.execution_facts = facts
        raise
    if report is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native consumer returned neither a result nor a refusal")
    return _posterior_result(report, data, facts)
