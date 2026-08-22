"""Public result views for causal-response queries.

These are language-native projections of the response, support, and
uncertainty axes.  They intentionally do not imply that empirical support and
structural identification are the same status.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal

from ..errors import CausalValueError
from ._format import fmt_float, fmt_pct
from ._views import IdentificationView

SupportStatus = Literal["supported", "weak_overlap", "extrapolative", "outside_empirical_support"]
UncertaintyKind = Literal["none", "pointwise", "simultaneous", "identified_set", "posterior"]


@dataclass(frozen=True, slots=True)
class ResponseView:
    """A scalar or vector causal response evaluated at explicit intervention points."""

    treatments: Sequence[str]
    outcomes: Sequence[str]
    points: Sequence[Sequence[float]]
    values: Sequence[Sequence[float]]

    def __post_init__(self) -> None:
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

    def __len__(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        return (
            f"<ResponseView {len(self)} points treatments={list(self.treatments)!r} "
            f"outcomes={list(self.outcomes)!r}>"
        )


@dataclass(frozen=True, slots=True)
class ResponseEnvelopeView:
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

    def __post_init__(self) -> None:
        if len(self.points) != len(self.lower) or len(self.points) != len(self.upper):
            raise CausalValueError("points, lower, and upper must have the same number of rows")
        if not 0.0 <= self.identified_mass <= 1.0:
            raise CausalValueError("identified_mass must be in [0, 1]")
        if not 0.0 <= self.unidentified_mass <= 1.0:
            raise CausalValueError("unidentified_mass must be in [0, 1]")
        if abs(self.identified_mass + self.unidentified_mass - 1.0) > 1e-9:
            raise CausalValueError("identified and unidentified mass must sum to one")
        if (
            self.completion_count < 1
            or not 0 <= self.truncated_completions <= self.completion_count
        ):
            raise CausalValueError("invalid PAG completion counts")
        if self.enumeration_capped != (self.mass_scope == "examined_completions"):
            raise CausalValueError("capped enumeration must label mass as examined completions")
        for lo, hi in zip(self.lower, self.upper, strict=True):
            if len(lo) != len(self.outcomes) or len(hi) != len(self.outcomes):
                raise CausalValueError("each envelope row must have one value per outcome")
            if any(a > b for a, b in zip(lo, hi, strict=True)):
                raise CausalValueError("each envelope lower value must not exceed its upper value")

    def __len__(self) -> int:
        return len(self.points)


@dataclass(frozen=True, slots=True)
class ResponseValidationCheck:
    """One estimand-aware response validation diagnostic."""

    id: str
    status: Literal["passed", "failed", "informative", "skipped"]
    statistic: float | None
    threshold: float | None
    detail: str
    replicates: int = 0


@dataclass(frozen=True, slots=True)
class ResponseValidationView:
    """Curve-legal checks, including explicit scalar-refuter skips."""

    checks: Sequence[ResponseValidationCheck] = ()

    @property
    def passed(self) -> bool:
        return not any(check.status == "failed" for check in self.checks)

    @property
    def skipped(self) -> list[ResponseValidationCheck]:
        return [check for check in self.checks if check.status == "skipped"]


@dataclass(frozen=True, slots=True)
class SupportDiagnostic:
    """One named empirical overlap/support diagnostic."""

    id: str
    values: Sequence[float]
    detail: str

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise CausalValueError("diagnostic name must be a non-empty string")
        if not all(isfinite(value) for value in self.values):
            raise CausalValueError("diagnostic values must be finite")


@dataclass(frozen=True, slots=True)
class SupportReport:
    """Estimand-aware empirical support, separate from identification status.

    ``status`` on a static curve is the worst label over requested points. On a
    temporal dose × horizon surface it summarizes ``point_status``: fully
    supported, partially extrapolative, or outside empirical support. Cell
    labels share the mean-surface layout (dose-major).
    """

    status: SupportStatus
    query_region: Mapping[str, tuple[float, float]]
    diagnostics: Sequence[SupportDiagnostic] = ()
    warnings: Sequence[str] = ()
    point_status: Sequence[SupportStatus] | None = None

    def __post_init__(self) -> None:
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

    def __bool__(self) -> bool:
        return self.status == "supported"

    def __repr__(self) -> str:
        cells = "" if self.point_status is None else f" cells={len(self.point_status)}"
        return (
            f"<SupportReport status={self.status!r} "
            f"diagnostics={len(self.diagnostics)} warnings={len(self.warnings)}{cells}>"
        )


@dataclass(frozen=True, slots=True)
class ResponseUncertainty:
    """Pointwise, simultaneous, set-valued, or posterior response uncertainty."""

    kind: UncertaintyKind
    lower: Sequence[Sequence[float]] | None = None
    upper: Sequence[Sequence[float]] | None = None
    level: float | None = None
    standard_error: float | None = None
    replicates: int | None = None
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        allowed = {"none", "pointwise", "simultaneous", "identified_set", "posterior"}
        if self.kind not in allowed:
            raise CausalValueError(f"unknown uncertainty kind {self.kind!r}")
        if (self.lower is None) != (self.upper is None):
            raise CausalValueError("lower and upper must either both be provided or both be None")
        if self.lower is not None and len(self.lower) != len(self.upper or ()):
            raise CausalValueError("lower and upper must have the same number of rows")
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
        if self.standard_error is not None and self.standard_error < 0.0:
            raise CausalValueError("standard_error must be non-negative")
        if self.replicates is not None and self.replicates < 1:
            raise CausalValueError("replicates must be at least 1")

    def __repr__(self) -> str:
        level = "" if self.level is None else f" level={fmt_pct(self.level)}"
        rows = 0 if self.lower is None else len(self.lower)
        return f"<ResponseUncertainty kind={self.kind!r}{level} rows={rows}>"


@dataclass(frozen=True, slots=True)
class CausalResponseView:
    """Top-level result projection shared by response-family estimands."""

    estimand: object
    response: ResponseView | None
    estimate: float | Sequence[float] | Sequence[Sequence[float]] | None
    uncertainty: ResponseUncertainty
    support: SupportReport
    identification: IdentificationView
    assumptions: Sequence[str] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    envelope: ResponseEnvelopeView | None = None
    validation: ResponseValidationView | None = None
    evidence_status: str | None = None
    allowlist_reason: str | None = None
    allowlist_parent: str | None = None

    def __repr__(self) -> str:
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
        return (
            f"<CausalResponseView estimate={estimate} support={self.support.status!r} "
            f"uncertainty={self.uncertainty.kind!r}{extra}>"
        )


__all__ = [
    "CausalResponseView",
    "ResponseEnvelopeView",
    "ResponseUncertainty",
    "ResponseView",
    "ResponseValidationCheck",
    "ResponseValidationView",
    "SupportDiagnostic",
    "SupportReport",
    "SupportStatus",
    "UncertaintyKind",
]
