"""Typed immutable reliability report sections; names and meanings follow Rust."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields
from types import UnionType
from typing import Any, cast, get_args, get_origin, get_type_hints

from ._freeze import freeze
from ._scientific import ScientificResult


class ReportRecord(ScientificResult, Mapping[str, Any]):
    def __getitem__(self, key: str) -> Any:
        if key not in {f.name for f in fields(cast(Any, self))}:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(f.name for f in fields(cast(Any, self)))

    def __len__(self) -> int:
        return len(fields(cast(Any, self)))

    @classmethod
    def from_wire(cls, wire: Mapping[str, Any]) -> Any:
        hints = get_type_hints(cls)
        return cls(**{f.name: _decode(hints[f.name], wire[f.name]) for f in fields(cast(Any, cls))})


def _decode(annotation: Any, value: Any) -> Any:
    if value is None:
        return None
    origin = get_origin(annotation)
    if origin is tuple:
        return tuple(_decode(get_args(annotation)[0], item) for item in value)
    if origin is UnionType:
        for member in get_args(annotation):
            if (
                isinstance(member, type)
                and issubclass(member, ReportRecord)
                and isinstance(value, Mapping)
            ):
                return member.from_wire(value)
        return freeze(value)
    if isinstance(annotation, type) and issubclass(annotation, ReportRecord):
        return annotation.from_wire(value)
    return freeze(value)


@dataclass(frozen=True, slots=True)
class McEstimate(ReportRecord):
    value: float
    standard_error: float
    effective_sample_size: float
    n_draws: int
    method: str | Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class FunctionalEstimate(ReportRecord):
    functional: str | Mapping[str, Any]
    estimate: McEstimate


@dataclass(frozen=True, slots=True)
class DifferenceEstimate(ReportRecord):
    functional: str | Mapping[str, Any]
    value: float
    standard_error: float
    effective_sample_size: float | None
    standard_error_a: float
    standard_error_b: float
    coupling: str | Mapping[str, Any]
    variance_ratio_vs_independent: float | None
    implied_correlation: float | None
    method: str | Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CandidateEstimate(ReportRecord):
    id: str
    estimate: McEstimate


@dataclass(frozen=True, slots=True)
class DifferenceVsDecision(ReportRecord):
    id: str
    difference: DifferenceEstimate


@dataclass(frozen=True, slots=True)
class StreamAccounting(ReportRecord):
    distinct_streams: int
    shared_noise: bool
    n_draws: int
    n_candidates: int
    weights_kish_ess: float | None


@dataclass(frozen=True, slots=True)
class MonteCarloErrorReport(ReportRecord):
    functional: str | Mapping[str, Any]
    candidates: tuple[CandidateEstimate, ...]
    decision: McEstimate
    differences: tuple[DifferenceVsDecision, ...]
    streams: StreamAccounting


@dataclass(frozen=True, slots=True)
class SourceSpread(ReportRecord):
    id: str
    mean: float
    standard_deviation: float
    difference_standard_deviation: float | None


@dataclass(frozen=True, slots=True)
class SelectionAssessment(ReportRecord):
    rule: str | Mapping[str, Any]
    n_considered: int
    in_sample: McEstimate
    in_sample_label: str | Mapping[str, Any]
    status: str | Mapping[str, Any]
    winners_curse: WinnersCurseBound | None
    held_out: HeldOutAssessment | None


@dataclass(frozen=True, slots=True)
class WinnersCurseBound(ReportRecord):
    n_considered: int
    n_supplied: int
    padded_candidates: int
    noise_dependence: str | Mapping[str, Any]
    gaussian_expected_maximum: float | None
    union_bound: float
    bound: float
    bound_kind: str | Mapping[str, Any]
    in_sample_value: float
    adjusted_value: float


@dataclass(frozen=True, slots=True)
class HeldOutAssessment(ReportRecord):
    record: str | Mapping[str, Any]
    held_out: FunctionalEstimate
    in_sample: McEstimate
    label: str | Mapping[str, Any]
    optimism: float
    optimism_standard_error: float


@dataclass(frozen=True, slots=True)
class TopKEstimate(ReportRecord):
    k: int
    estimate: McEstimate


@dataclass(frozen=True, slots=True)
class CandidateRankStatement(ReportRecord):
    id: str
    expected_utility: McEstimate
    probability_best: McEstimate
    probability_top_k: tuple[TopKEstimate, ...]
    probability_better_than_baseline: McEstimate | None


@dataclass(frozen=True, slots=True)
class DependenceStatement(ReportRecord):
    coupling: str | Mapping[str, Any]
    within_stream: str | Mapping[str, Any]
    repeated_evaluations: int
    candidates_adaptively_generated: bool
    validation_independent_of_selection: bool | None


@dataclass(frozen=True, slots=True)
class CalibrationCoordinate(ReportRecord):
    candidate_set_size: int
    mc_budget: int
    selection_rule: str
    functional: str
    coupling: str | Mapping[str, Any]
    paradigm: str | Mapping[str, Any]
    within_stream: str | Mapping[str, Any]
    importance_weighted: bool
    adaptively_generated: bool


@dataclass(frozen=True, slots=True)
class SpreadReport(ReportRecord):
    paradigm: str
    candidates: tuple[SourceSpread, ...]


@dataclass(frozen=True, slots=True)
class SourceUncertainty(ReportRecord):
    spread: SpreadReport


@dataclass(frozen=True, slots=True)
class SelectionUncertainty(ReportRecord):
    selected: SelectionAssessment
