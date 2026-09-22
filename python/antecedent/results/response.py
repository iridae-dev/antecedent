"""Public result views for causal-response queries.

These are language-native projections of the response, support, and
uncertainty axes.  They intentionally do not imply that empirical support and
structural identification are the same status.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite, isnan
from typing import Any, ClassVar, Literal, Self

from pydantic import Field, PrivateAttr, model_validator

from .._verdict import describe_status
from ..errors import CausalValueError
from ._execution import ResultAPI
from ._format import fmt_float, fmt_pct
from ._report import ResultModel
from ._slots import ReasoningSlots, mass_limitation
from ._views import IdentificationView

SupportStatus = Literal["supported", "weak_overlap", "extrapolative", "outside_empirical_support"]
UncertaintyKind = Literal["none", "pointwise", "simultaneous", "identified_set", "posterior"]
#: What an interval's level means: repeated-sampling ``"confidence"`` or posterior
#: ``"credible"``. A credible interval's ``standard_error`` is a posterior standard
#: deviation, and neither reading transfers to the other.
IntervalInterpretation = Literal["confidence", "credible"]


class ResponseView(ResultModel):
    """A scalar or vector causal response evaluated at explicit intervention points."""

    treatments: Sequence[str]
    outcomes: Sequence[str]
    points: Sequence[Sequence[float]]
    values: Sequence[Sequence[float]]

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not self.treatments or not self.outcomes:
            raise CausalValueError("treatments and outcomes must not be empty")
        if len(self.points) != len(self.values):
            raise CausalValueError("points and values must have the same number of rows")
        for point in self.points:
            # Static curves use one coordinate per treatment. Temporal dose ×
            # horizon surfaces use a multi-axis grid (dose, horizon, …) with a
            # single treatment name; allow len(point) >= len(treatments).
            if len(point) < len(self.treatments):
                raise CausalValueError(
                    "each response point must have at least one value per treatment"
                )
            if len(self.treatments) > 1 and len(point) != len(self.treatments):
                raise CausalValueError("each response point must have one value per treatment")
            if not all(isfinite(value) for value in point):
                raise CausalValueError("response points must be finite")
        for row in self.values:
            if len(row) != len(self.outcomes):
                raise CausalValueError("each response row must have one value per outcome")
            if not all(isfinite(value) for value in row):
                raise CausalValueError("response values must be finite")
        return self

    def __len__(self) -> int:
        return len(self.points)

    def to_columns(self) -> dict[str, list[Any]]:
        """Grid as name → column. Twin of dict-in; no frame dep."""
        data: dict[str, list[Any]] = {}
        width = len(self.points[0]) if self.points else len(self.treatments)
        for i in range(width):
            name = self.treatments[i] if i < len(self.treatments) else f"axis_{i}"
            data[name] = [point[i] for point in self.points]
        for i, name in enumerate(self.outcomes):
            data[name] = [row[i] for row in self.values]
        return data

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
        try:
            import pyarrow as pa
        except ImportError as exc:
            raise ImportError(
                "Arrow stream export requires pyarrow; install it with "
                "`pip install pyarrow` (or `uv add pyarrow`)"
            ) from exc
        return pa.table(self.to_columns()).__arrow_c_stream__(requested_schema)

    def __repr__(self) -> str:
        return (
            f"<ResponseView {len(self)} points treatments={list(self.treatments)!r} "
            f"outcomes={list(self.outcomes)!r}>"
        )


class ResponseEnvelopeView(ResultModel):
    """Function-valued identified envelope over one shared intervention grid."""

    treatments: Sequence[str]
    outcomes: Sequence[str]
    points: Sequence[Sequence[float]]
    lower: Sequence[Sequence[float]]
    upper: Sequence[Sequence[float]]
    identified_mass: float
    unidentified_mass: float
    completion_count: int
    truncated_completions: int = 0
    enumeration_capped: bool = False
    mass_scope: Literal["full_class", "examined_completions"] = "full_class"
    weight_basis: Literal[
        "posterior_probability", "completion_enumeration", "caller_supplied_class_prior"
    ] = "completion_enumeration"
    atom_keys: Sequence[int] = ()
    atom_weights: Sequence[float] = ()
    atom_statuses: Sequence[str] = ()
    atom_values: Sequence[Sequence[float]] = ()
    #: Mass identified in theory whose estimation failed. It is not
    #: unidentified mass and is not mixed into the published value.
    unevaluable_mass: float = 0.0
    #: Mass on identified atoms the Interactive latency tier left out of its
    #: graph subsample and never evaluated. Neither unidentified nor a failed
    #: estimate, and not mixed into the published value.
    subsampled_out_mass: float = 0.0

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if len(self.points) != len(self.lower) or len(self.points) != len(self.upper):
            raise CausalValueError("points, lower, and upper must have the same number of rows")
        if not 0.0 <= self.identified_mass <= 1.0:
            raise CausalValueError("identified_mass must be in [0, 1]")
        if not 0.0 <= self.unidentified_mass <= 1.0:
            raise CausalValueError("unidentified_mass must be in [0, 1]")
        if not 0.0 <= self.unevaluable_mass <= 1.0:
            raise CausalValueError("unevaluable_mass must be in [0, 1]")
        if not 0.0 <= self.subsampled_out_mass <= 1.0:
            raise CausalValueError("subsampled_out_mass must be in [0, 1]")
        total = (
            self.identified_mass
            + self.unidentified_mass
            + self.unevaluable_mass
            + self.subsampled_out_mass
        )
        if abs(total - 1.0) > 1e-9:
            raise CausalValueError(
                "identified, unidentified, unevaluable, and subsampled-out mass must sum to one"
            )
        if (
            self.completion_count < 1
            or not 0 <= self.truncated_completions <= self.completion_count
        ):
            raise CausalValueError("invalid PAG completion counts")
        if self.enumeration_capped != (self.mass_scope == "examined_completions"):
            raise CausalValueError("capped enumeration must label mass as examined completions")
        atom_count = len(self.atom_keys)
        if not (
            len(self.atom_weights) == len(self.atom_statuses) == len(self.atom_values) == atom_count
        ):
            raise CausalValueError("structural atom metadata must have equal lengths")
        for lo, hi in zip(self.lower, self.upper, strict=True):
            if len(lo) != len(self.outcomes) or len(hi) != len(self.outcomes):
                raise CausalValueError("each envelope row must have one value per outcome")
            if any(a > b for a, b in zip(lo, hi, strict=True)):
                raise CausalValueError("each envelope lower value must not exceed its upper value")
        return self

    def __len__(self) -> int:
        return len(self.points)


class ResponseValidationCheck(ResultModel):
    """One estimand-aware response validation diagnostic."""

    id: str
    status: Literal["passed", "failed", "informative", "skipped"]
    statistic: float | None
    threshold: float | None
    detail: str
    replicates: int = 0


class ResponseValidationView(ResultModel):
    """Curve-legal checks, including explicit scalar-refuter skips."""

    checks: Sequence[ResponseValidationCheck] = ()

    @property
    def passed(self) -> bool:
        return not any(check.status == "failed" for check in self.checks)

    @property
    def skipped(self) -> list[ResponseValidationCheck]:
        return [check for check in self.checks if check.status == "skipped"]


class SupportDiagnostic(ResultModel):
    """One named empirical overlap/support diagnostic."""

    id: str
    values: Sequence[float]
    detail: str

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not self.id.strip():
            raise CausalValueError("diagnostic name must be a non-empty string")
        if not all(isfinite(value) for value in self.values):
            raise CausalValueError("diagnostic values must be finite")
        return self


class SupportReport(ResultModel):
    """Estimand-aware empirical support, separate from identification status.

    ``status`` on a static curve is the worst label over requested points. On a
    temporal dose × horizon surface it summarizes ``point_status``: fully
    supported, partially extrapolative, or outside empirical support. Cell
    labels share the mean-surface layout (dose-major).
    """

    status: str
    query_region: Mapping[str, tuple[float, float]]
    diagnostics: Sequence[SupportDiagnostic] = ()
    warnings: Sequence[str] = ()
    point_status: Sequence[str] | None = None

    @model_validator(mode="after")
    def _validate(self) -> Self:
        allowed = {
            "supported",
            "weak_overlap",
            "extrapolative",
            "outside_empirical_support",
        }
        if self.status not in allowed:
            raise CausalValueError(f"unknown support status {self.status!r}")
        if self.point_status is not None:
            for status in self.point_status:
                if status not in allowed:
                    raise CausalValueError(f"unknown support status {status!r}")
        for variable, bounds in self.query_region.items():
            if (
                not variable.strip()
                or len(bounds) != 2
                or not all(isfinite(value) for value in bounds)
                or bounds[0] > bounds[1]
            ):
                raise CausalValueError(f"invalid support query region for {variable!r}")
        return self

    def __bool__(self) -> bool:
        return self.status == "supported"

    def __repr__(self) -> str:
        cells = "" if self.point_status is None else f" cells={len(self.point_status)}"
        return (
            f"<SupportReport status={self.status!r} "
            f"diagnostics={len(self.diagnostics)} warnings={len(self.warnings)}{cells}>"
        )


class ResponseUncertainty(ResultModel):
    """Pointwise, simultaneous, set-valued, or posterior response uncertainty."""

    kind: str
    lower: Sequence[Sequence[float]] | None = None
    upper: Sequence[Sequence[float]] | None = None
    level: float | None = None
    standard_error: float | None = None
    replicates: int | None = None
    artifact_id: str | None = None
    #: ``"confidence"`` or ``"credible"`` for an interval-bearing kind; ``None`` for
    #: ``none`` and ``posterior``. Left untyped as ``IntervalInterpretation`` (a strict
    #: pydantic ``Literal``) so an unknown value reaches ``_validate`` below and raises
    #: ``CausalValueError`` with a specific message, instead of a generic pydantic
    #: ``ValidationError`` short-circuiting field assembly first.
    interpretation: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> Self:
        allowed = {"none", "pointwise", "simultaneous", "identified_set", "posterior"}
        if self.kind not in allowed:
            raise CausalValueError(f"unknown uncertainty kind {self.kind!r}")
        if self.interpretation is not None:
            if self.interpretation not in ("confidence", "credible"):
                raise CausalValueError(f"unknown interval interpretation {self.interpretation!r}")
            if self.kind in ("none", "posterior"):
                raise CausalValueError(f"kind={self.kind!r} carries no interval to interpret")
        if (self.lower is None) != (self.upper is None):
            raise CausalValueError("lower and upper must either both be provided or both be None")
        if self.lower is not None and len(self.lower) != len(self.upper or ()):
            raise CausalValueError("lower and upper must have the same number of rows")
        if self.lower is not None and self.upper is not None:
            for lower, upper in zip(self.lower, self.upper, strict=True):
                if len(lower) != len(upper):
                    raise CausalValueError("lower and upper rows must have the same width")
                if any(
                    isnan(lo) or isnan(hi) or lo > hi for lo, hi in zip(lower, upper, strict=True)
                ):
                    raise CausalValueError(
                        "uncertainty bounds must be ordered and cannot contain NaN"
                    )
        if self.level is not None and not 0.0 < self.level < 1.0:
            raise CausalValueError("level must be strictly between 0 and 1")
        if self.kind == "none" and any(
            value is not None
            for value in (
                self.lower,
                self.level,
                self.standard_error,
                self.replicates,
                self.artifact_id,
            )
        ):
            raise CausalValueError("kind='none' cannot carry uncertainty data")
        if self.standard_error is not None and (
            not isfinite(self.standard_error) or self.standard_error < 0.0
        ):
            raise CausalValueError("standard_error must be finite and non-negative")
        if self.replicates is not None and self.replicates < 1:
            raise CausalValueError("replicates must be at least 1")
        return self

    def __repr__(self) -> str:
        level = "" if self.level is None else f" level={fmt_pct(self.level)}"
        rows = 0 if self.lower is None else len(self.lower)
        return f"<ResponseUncertainty kind={self.kind!r}{level} rows={rows}>"


SIMULTANEOUS_BAND_LOWER = "response.simultaneous_band.lower"
SIMULTANEOUS_BAND_UPPER = "response.simultaneous_band.upper"
SIMULTANEOUS_BAND_CRITICAL = "response.simultaneous_band.critical"


class SimultaneousBand(ResultModel):
    """A band that covers every response cell at once, next to a pointwise band.

    Temporal ``ResponseCurve`` / ``InterventionResponse`` surfaces keep their
    pointwise band in :attr:`CausalResponseView.uncertainty` and publish this
    max-studentized-deviation (sup-t) band from the same joint replicates
    (Frequentist circular-block bootstrap) or posterior draws (Bayesian). It is
    carried natively as the ``response.simultaneous_band.{lower,upper,critical}``
    support diagnostics; this view reads them back. ``lower`` / ``upper`` share the
    row layout of ``uncertainty.lower`` / ``uncertainty.upper`` (one value per
    outcome, dose-major on a temporal surface). ``critical`` is the sup-t critical
    value, floored at the one-cell normal quantile, so the band is never narrower
    than the pointwise band. ``replicates`` counts the joint replicates or draws
    it was computed from. ``detail`` names the construction.
    """

    level: float
    critical: float
    replicates: int
    lower: Sequence[Sequence[float]]
    upper: Sequence[Sequence[float]]
    detail: str = ""

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not 0.0 < self.level < 1.0:
            raise CausalValueError("level must be strictly between 0 and 1")
        if not isfinite(self.critical) or self.critical < 0.0:
            raise CausalValueError("critical value must be finite and non-negative")
        if self.replicates < 1:
            raise CausalValueError("replicates must be at least 1")
        if len(self.lower) != len(self.upper):
            raise CausalValueError("lower and upper must have the same number of rows")
        for lower, upper in zip(self.lower, self.upper, strict=True):
            if len(lower) != len(upper) or any(
                lo > hi for lo, hi in zip(lower, upper, strict=True)
            ):
                raise CausalValueError("simultaneous band rows must be ordered and aligned")
        return self

    @classmethod
    def from_support(cls, support: SupportReport) -> SimultaneousBand | None:
        """Rebuild the band from its support diagnostics, or ``None`` if none was published."""
        by_id = {diagnostic.id: diagnostic for diagnostic in support.diagnostics}
        lower = by_id.get(SIMULTANEOUS_BAND_LOWER)
        upper = by_id.get(SIMULTANEOUS_BAND_UPPER)
        critical = by_id.get(SIMULTANEOUS_BAND_CRITICAL)
        if lower is None or upper is None or critical is None or len(critical.values) < 3:
            return None
        level, value, replicates = critical.values[:3]
        return cls(
            level=float(level),
            critical=float(value),
            replicates=int(replicates),
            lower=tuple((float(v),) for v in lower.values),
            upper=tuple((float(v),) for v in upper.values),
            detail=lower.detail,
        )

    def __len__(self) -> int:
        return len(self.lower)

    def __repr__(self) -> str:
        return (
            f"<SimultaneousBand level={fmt_pct(self.level)} critical={fmt_float(self.critical)} "
            f"replicates={self.replicates} rows={len(self)}>"
        )


class CausalResponseView(ResultModel, ResultAPI):
    """Top-level result projection shared by response-family estimands."""

    _function_valued: ClassVar[bool] = True

    estimand: object
    response: ResponseView | None
    estimate: float | Sequence[float] | Sequence[Sequence[float]] | None
    uncertainty: ResponseUncertainty
    support: SupportReport
    identification: IdentificationView
    assumptions: Sequence[str] = ()
    provenance: Mapping[str, Any] = Field(default_factory=dict)
    envelope: ResponseEnvelopeView | None = None
    validation: ResponseValidationView | None = None
    evidence_status: str | None = None
    allowlist_reason: str | None = None
    allowlist_parent: str | None = None
    diagnostics: Sequence[str] = ()
    certificate: dict[str, Any] | None = None
    reasoning: ReasoningSlots | None = None
    #: Compiled program identity. Distinct from ``claim_id``.
    program_id: str | None = None
    #: Execution claim identity. Distinct from ``program_id``.
    claim_id: str | None = None
    #: Identity of the data snapshot this execution ran on.
    data_snapshot_id: str | None = None
    #: T5–T9 transport lineage when the estimand was ``transport.Transport``.
    transport: Any = None
    _prepared: Any = PrivateAttr(default=None)
    _execution: Any = PrivateAttr(default=None)

    @property
    def simultaneous_band(self) -> SimultaneousBand | None:
        """Band over the whole response grid published next to the pointwise band.

        ``None`` when no such band was published: a Frequentist temporal surface
        run with ``bootstrap=0`` (see the ``estimate.temporal_response.band_withheld``
        warning), too few surviving replicates
        (``response.simultaneous_band_withheld``), or a response whose
        ``uncertainty.kind`` is already ``"simultaneous"`` (the static Kennedy-DR
        multiplier band) and so needs no second band.
        """
        return SimultaneousBand.from_support(self.support)

    def __repr__(self) -> str:
        limitation = self.rendering_limitation()
        if limitation is not None:
            return (
                f"<CausalResponseView {describe_status(self.identification.status)} "
                f"answer={self.answer.kind} limitation={limitation}>"
            )
        if isinstance(self.estimate, (float, int)):
            estimate = fmt_float(float(self.estimate))
        elif self.response is not None:
            estimate = f"{len(self.response)} response points"
        else:
            estimate = "structured"
        extra = ""
        if self.evidence_status == "allowed_unlicensed":
            extra = " unlicensed"
        if self.support.warnings:
            extra += f" warnings={len(self.support.warnings)}"
        rbc = any(
            "derivative_interval_bias_corrected" in str(warning)
            for warning in self.support.warnings
        )
        if rbc and isinstance(self.estimate, (float, int)) and self.uncertainty.lower:
            lo = self.uncertainty.lower[0][0]
            hi = self.uncertainty.upper[0][0] if self.uncertainty.upper else float("nan")
            extra += (
                f" interval=[{fmt_float(lo)}, {fmt_float(hi)}] around RBC center, "
                "not the conventional point"
            )
        return (
            f"<CausalResponseView estimate={estimate} support={self.support.status!r} "
            f"uncertainty={self.uncertainty.kind!r}{extra}>"
        )

    def rendering_limitation(self) -> str | None:
        if self.reasoning is not None:
            limit = self.reasoning.rendering_limitation()
            if limit is not None:
                return limit
        mass = None if self.envelope is None else self.envelope.unidentified_mass
        return mass_limitation(mass, identified_set=self.envelope is not None)


__all__ = [
    "CausalResponseView",
    "IntervalInterpretation",
    "ResponseEnvelopeView",
    "ResponseUncertainty",
    "ResponseView",
    "ResponseValidationCheck",
    "ResponseValidationView",
    "SimultaneousBand",
    "SupportDiagnostic",
    "SupportReport",
    "SupportStatus",
    "UncertaintyKind",
]
