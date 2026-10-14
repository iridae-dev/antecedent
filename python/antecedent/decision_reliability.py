"""Reliability of sampled and adaptively selected decisions.

Computing a decision from a supplied law is not the same as establishing that the decision is
reliable. :func:`assess` takes per-draw utilities of a finite candidate set
(:class:`DecisionDraws`, built from numpy arrays with the stream each came from and the declared
dependence) and reports three uncertainties **separately**, never summed::

    import numpy as np
    from antecedent import decision_reliability as dr

    stream = dr.StreamId(run_seed=1, purpose="selection", replicate=0, label="posterior")
    draws = dr.DecisionDraws.from_arrays(
        {"a": a_utilities, "b": b_utilities},
        stream=stream,
        source="posterior",
        coupling="shared_simulation",
        within_stream="independent",
    )
    result = dr.assess(draws, dr.ReliabilityRequest(ranking=dr.RankingRequest(baseline="b")))
    result.report.source_uncertainty      # spread of the utility under the supplied law
    result.report.monte_carlo_error       # standard errors and effective sample sizes
    result.report.selection_uncertainty   # in-sample value, winner's-curse bound, validation

* **Source-distribution uncertainty** is the spread of the utility under the supplied law (none
  when the law is declared ``known_law``). It does not shrink with more draws.
* **Monte Carlo error** is the standard error and effective sample size of utility means,
  differences, threshold probabilities and risk measures (``cvar`` by its influence function,
  quantiles by Maritz-Jarrett), accounting for common random numbers, dependent streams (batch
  means) and importance weights (Kish).
* **Selection-induced uncertainty** labels the unadjusted maximum ``in_sample`` and reports a
  winner's-curse bound, a held-out re-evaluation on an independent stream, or a sample split, with
  the streams that selected and validated.

Everything is a point diagnostic labelled uncalibrated. Ranking probabilities (best, top-``k``,
better than a baseline) are Monte Carlo estimates under the supplied law over the candidates as
supplied, refused for adaptively generated candidates without a :class:`CandidateSearchRecord`.
:meth:`DecisionReliability.publish` and :func:`publish` always raise ``cell_not_licensed`` until a
coverage record measures the route at its coordinate, and
:meth:`DecisionReliability.combined_uncertainty` always raises
``uncertainty_components_not_combined``.

Every validation, number, identity and refusal is computed in Rust; this module builds the
declarations and presents the results. :meth:`DecisionReliability.export` writes
``decision_reliability_v1``; :func:`consume_decision_reliability_artifact` recomputes the report
from the carried draws and refuses any edited draw, stream, declaration or claimed value.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from . import _native
from ._freeze import freeze, thaw
from ._refusal import StructuredRefusal, raise_refusal
from ._reliability_records import (
    CalibrationCoordinate,
    CandidateRankStatement,
    DependenceStatement,
    MonteCarloErrorReport,
    SelectionUncertainty,
    SourceUncertainty,
)
from ._scientific import ScientificResult
from ._validate import count as _count
from .errors import CausalTypeError, CausalValueError

__all__ = [
    "MonteCarloErrorReport",
    "SourceUncertainty",
    "SelectionUncertainty",
    "CandidateRankStatement",
    "DependenceStatement",
    "CalibrationCoordinate",
    "BatchPlan",
    "CandidateSearchRecord",
    "ConsumedDecisionReliability",
    "DecisionDraws",
    "DecisionReliability",
    "DecisionReliabilityRefusal",
    "DrawSegment",
    "Functional",
    "RankingReport",
    "RankingRequest",
    "ReliabilityReport",
    "ReliabilityRequest",
    "StreamId",
    "assess",
    "consume_decision_reliability_artifact",
    "publish",
    "rank",
]

_PURPOSES = ("source", "selection", "validation", "evaluation")
_SOURCES = ("posterior", "sampling_distribution", "known_law")
_COUPLINGS = ("shared_simulation", "declared_independent", "unknown")
_WITHIN = ("independent", "autocorrelated", "unknown")
_ROUTES = (
    "ranking_probability",
    "optimality_confidence_set",
    "utility_interval",
    "selection_corrected_interval",
)
_DEFAULT_ARTIFACT_ID = "decision_reliability"


PurposesName = Literal["source", "selection", "validation", "evaluation"]
SourcesName = Literal["posterior", "sampling_distribution", "known_law"]
CouplingsName = Literal["shared_simulation", "declared_independent", "unknown"]
WithinName = Literal["independent", "autocorrelated", "unknown"]
RoutesName = Literal[
    "ranking_probability",
    "optimality_confidence_set",
    "utility_interval",
    "selection_corrected_interval",
]


class DecisionReliabilityRefusal(StructuredRefusal):
    """A typed refusal of decision reliability.

    A :class:`~antecedent.errors.CausalUnsupportedError` with a registered ``reason_code``
    (``candidate_search_record_required``, ``validation_stream_not_independent``,
    ``uncertainty_components_not_combined``, ``joint_law_required``,
    ``sampling_dependence_unknown``, ``cell_not_licensed``, ...) and a
    ``decision_reliability.*`` ``detail``.
    """


def _refuse(payload: str | None) -> None:
    raise_refusal(payload, DecisionReliabilityRefusal)


def _check_choice(value: object, allowed: tuple[str, ...], what: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise CausalValueError(f"{what} must be one of {allowed}; got {value!r}")
    return value


def _real(value: object, what: str) -> float:
    if isinstance(value, bool) or not hasattr(value, "__float__"):
        raise CausalTypeError(f"{what} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise CausalValueError(f"{what} must be finite")
    return number


def _dumps(value: object) -> str:
    return json.dumps(thaw(value), allow_nan=False, separators=(",", ":"))


# --------------------------------------------------------------------------- streams


@dataclass(frozen=True)
class StreamId:
    """Semantic coordinates of one random stream: seed, purpose, replicate and label.

    ``purpose`` is one of ``source``, ``selection``, ``validation`` and ``evaluation``; it is a
    coordinate of the stream's identity, so a selection stream and a validation stream never share
    noise.
    """

    run_seed: int
    purpose: str
    replicate: int = 0
    label: str = "draws"

    def __post_init__(self) -> None:
        _check_choice(self.purpose, _PURPOSES, "purpose")
        if not isinstance(self.label, str) or not self.label.strip():
            raise CausalTypeError("label must be a non-empty string")
        for name in ("run_seed", "replicate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CausalValueError(f"{name} must be a non-negative integer")

    def to_wire(self) -> dict[str, Any]:
        """The stream as a wire mapping."""
        return {
            "run_seed": self.run_seed,
            "purpose": self.purpose,
            "replicate": self.replicate,
            "label": self.label,
        }


@dataclass(frozen=True)
class DrawSegment:
    """Draws ``[start, start + length)`` of a stream."""

    stream: StreamId
    start: int = 0
    length: int | None = None


# --------------------------------------------------------------------------- functionals


@dataclass(frozen=True)
class Functional:
    """A functional of the utility distribution.

    Build one with :meth:`expected_utility`, :meth:`threshold_probability`, :meth:`quantile` or
    :meth:`cvar`.
    """

    _wire: Any = field(repr=False)
    kind: str = "expected_utility"

    @classmethod
    def expected_utility(cls) -> Functional:
        """The expected utility."""
        return cls("expected_utility", "expected_utility")

    @classmethod
    def threshold_probability(cls, threshold: float, side: str = "above") -> Functional:
        """``P(utility > threshold)`` (``side="above"``) or ``P(utility <= threshold)``."""
        side = _check_choice(side, ("above", "at_or_below"), "side")
        wire = {"threshold_probability": {"threshold": _real(threshold, "threshold"), "side": side}}
        return cls(wire, "threshold_probability")

    @classmethod
    def quantile(cls, p: float) -> Functional:
        """The ``p``-quantile, ``p`` in ``(0, 1)``."""
        level = _real(p, "p")
        if not 0.0 < level < 1.0:
            raise CausalValueError("p must lie in (0, 1)")
        return cls({"quantile": {"p": level}}, "quantile")

    @classmethod
    def cvar(cls, alpha: float, tail: str = "lower") -> Functional:
        """Conditional value at risk at tail mass ``alpha`` in ``(0, 1]``."""
        level = _real(alpha, "alpha")
        if not 0.0 < level <= 1.0:
            raise CausalValueError("alpha must lie in (0, 1]")
        tail = _check_choice(tail, ("lower", "upper"), "tail")
        return cls({"cvar": {"alpha": level, "tail": tail}}, "cvar")

    def to_wire(self) -> Any:
        """The functional as a wire value."""
        return self._wire


@dataclass(frozen=True)
class BatchPlan:
    """How many non-overlapping batches a dependent stream is cut into."""

    batches: int | None = None

    def to_wire(self) -> Any:
        """The plan as a wire value (``floor(sqrt(n))`` batches when ``batches`` is ``None``)."""
        if self.batches is None:
            return "square_root"
        if isinstance(self.batches, bool) or not isinstance(self.batches, int) or self.batches < 2:
            raise CausalValueError("batches must be an integer of at least 2")
        return {"fixed": self.batches}


# --------------------------------------------------------------------------- the carrier


@dataclass(frozen=True)
class CandidateSearchRecord:
    """The identity of an adaptive candidate search.

    ``generator`` names the algorithm and configuration, ``budget`` the most candidates it may
    evaluate, ``n_evaluated`` how many it did evaluate (retained or not) and ``selection_rule`` the
    rule that chose the reported decision (``maximize_expected_utility``,
    ``maximize_threshold_probability``, ``maximize_quantile``, ``maximize_cvar`` or any other
    label, declared as such).
    """

    generator: str
    budget: int
    n_evaluated: int
    selection_rule: str = "maximize_expected_utility"
    generator_stream: StreamId | None = None

    def to_wire(self) -> dict[str, Any]:
        """The record as a wire mapping."""
        known = {
            "maximize_expected_utility",
            "maximize_threshold_probability",
            "maximize_quantile",
            "maximize_cvar",
        }
        rule: Any = (
            self.selection_rule
            if self.selection_rule in known
            else {"declared": self.selection_rule}
        )
        return {
            "generator": self.generator,
            "budget": _count(
                self.budget, "budget", minimum=0, maximum=2**64 - 1, reason_code="invalid_argument"
            ),
            "n_evaluated": _count(
                self.n_evaluated,
                "n_evaluated",
                minimum=0,
                maximum=2**64 - 1,
                reason_code="invalid_argument",
            ),
            "selection_rule": rule,
            "generator_stream": None
            if self.generator_stream is None
            else self.generator_stream.to_wire(),
        }


def _segment_wire(stream: StreamId | DrawSegment, n: int) -> dict[str, Any]:
    if isinstance(stream, DrawSegment):
        length = n if stream.length is None else int(stream.length)
        return {"stream": stream.stream.to_wire(), "start": int(stream.start), "len": length}
    if isinstance(stream, StreamId):
        return {"stream": stream.to_wire(), "start": 0, "len": n}
    raise CausalTypeError("a stream must be a StreamId or a DrawSegment")


class DecisionDraws:
    """Per-draw utilities of a finite candidate set, with stream identities and declarations.

    Dependence is declared, never inferred: ``coupling`` and ``within_stream`` default to
    ``"unknown"``, which every assessment refuses. Build one with :meth:`from_arrays`.
    """

    def __init__(self, wire: Mapping[str, Any]) -> None:
        self._wire: dict[str, Any] = json.loads(_dumps(dict(wire)))

    @classmethod
    def from_arrays(
        cls,
        utilities: Mapping[str, Any],
        *,
        stream: StreamId | DrawSegment | Mapping[str, StreamId | DrawSegment],
        utility_unit: str = "utility",
        producer: str = "python",
        source: str = "posterior",
        coupling: str = "unknown",
        within_stream: str = "unknown",
        weights: Any = None,
        repeated_evaluations: int = 0,
        origin: str = "enumerated",
        search_record: CandidateSearchRecord | None = None,
    ) -> DecisionDraws:
        """Build a carrier from ``{candidate id: 1-D array of per-draw utilities}``.

        ``stream`` is the one stream every candidate was evaluated on (a shared simulation) or a
        mapping from candidate id to that candidate's own stream (declared independent).
        """
        _check_choice(source, _SOURCES, "source")
        _check_choice(coupling, _COUPLINGS, "coupling")
        _check_choice(within_stream, _WITHIN, "within_stream")
        _check_choice(origin, ("enumerated", "adaptively_generated"), "origin")
        if not isinstance(utilities, Mapping) or not utilities:
            raise CausalTypeError("utilities must be a non-empty mapping of id -> array")
        candidates = []
        for candidate_id, values in utilities.items():
            array = np.asarray(values, dtype=np.float64)
            if array.ndim != 1:
                raise CausalValueError(f"utilities of {candidate_id!r} must be one-dimensional")
            if not np.all(np.isfinite(array)):
                raise CausalValueError(f"utilities of {candidate_id!r} must be finite")
            own = stream[candidate_id] if isinstance(stream, Mapping) else stream
            candidates.append(
                {
                    "id": str(candidate_id),
                    "utilities": array.tolist(),
                    "segment": _segment_wire(own, int(array.shape[0])),
                }
            )
        weight_list = None
        if weights is not None:
            warray = np.asarray(weights, dtype=np.float64)
            if warray.ndim != 1 or not np.all(np.isfinite(warray)):
                raise CausalValueError("weights must be a finite one-dimensional array")
            weight_list = warray.tolist()
        return cls(
            {
                "utility_unit": utility_unit,
                "producer": producer,
                "source": source,
                "coupling": coupling,
                "within_stream": within_stream,
                "weights": weight_list,
                "repeated_evaluations": _count(
                    repeated_evaluations,
                    "repeated_evaluations",
                    minimum=0,
                    maximum=2**64 - 1,
                    reason_code="invalid_argument",
                ),
                "origin": origin,
                "search_record": None if search_record is None else search_record.to_wire(),
                "candidates": candidates,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        """The carrier as a wire mapping."""
        return json.loads(_dumps(self._wire))

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        """Candidate identities in the order supplied."""
        return tuple(c["id"] for c in self._wire["candidates"])

    def digests(self) -> tuple[str, str]:
        """``(draws digest, candidate-set digest)``; refuses an invalid or undeclared carrier."""
        digests, refusal = _native.decision_reliability_digests(_dumps(self._wire))
        _refuse(refusal)
        assert digests is not None
        return digests[0], digests[1]


# --------------------------------------------------------------------------- requests


@dataclass(frozen=True)
class RankingRequest:
    """Which ranking statements to report: a baseline and top-``k`` sizes."""

    baseline: str | None = None
    top_k: Sequence[int] = ()
    batch_plan: BatchPlan = field(default_factory=BatchPlan)

    def to_wire(self) -> dict[str, Any]:
        """The request as a wire mapping."""
        return {
            "baseline": self.baseline,
            "top_k": [
                _count(k, "k", minimum=0, maximum=2**64 - 1, reason_code="invalid_argument")
                for k in self.top_k
            ],
            "batch_plan": self.batch_plan.to_wire(),
        }


@dataclass(frozen=True)
class ReliabilityRequest:
    """What to assess.

    ``decision`` is ``None`` for the candidate maximizing ``functional`` on the supplied draws or
    the identity of a decision fixed independently of the draws. ``validation`` is ``None``,
    ``"provided"`` (re-evaluate on the ``validation`` carrier passed to :func:`assess`, with
    ``gap`` draws demanded between ranges of one shared stream) or ``"split"`` (select on the
    leading ``selection_fraction`` of the draws, validate on the rest after ``gap`` draws).
    ``reporting="combined"`` is refused.
    """

    functional: Functional = field(default_factory=Functional.expected_utility)
    decision: str | None = None
    batch_plan: BatchPlan = field(default_factory=BatchPlan)
    validation: str | None = None
    validation_kind: str = "independent_simulation_stream"
    selection_fraction: float = 0.5
    gap: int = 0
    ranking: RankingRequest | None = None
    reporting: str = "separate"

    def to_wire(self) -> dict[str, Any]:
        """The request as a wire mapping."""
        _check_choice(self.reporting, ("separate", "combined"), "reporting")
        if self.validation is None:
            validation: Any = "no_validation"
        elif self.validation == "provided":
            _check_choice(
                self.validation_kind,
                ("independent_simulation_stream", "split_of_selection_stream", "held_out_data"),
                "validation_kind",
            )
            validation = {
                "provided_stream": {
                    "kind": self.validation_kind,
                    "gap": _count(
                        self.gap,
                        "gap",
                        minimum=0,
                        maximum=2**64 - 1,
                        reason_code="invalid_argument",
                    ),
                }
            }
        elif self.validation == "split":
            validation = {
                "split_selection_draws": {
                    "selection_fraction": _real(self.selection_fraction, "selection_fraction"),
                    "gap": _count(
                        self.gap,
                        "gap",
                        minimum=0,
                        maximum=2**64 - 1,
                        reason_code="invalid_argument",
                    ),
                }
            }
        else:
            raise CausalValueError("validation must be None, 'provided' or 'split'")
        decision: Any = (
            "selected_by_rule" if self.decision is None else {"pre_specified": self.decision}
        )
        return {
            "functional": self.functional.to_wire(),
            "decision": decision,
            "batch_plan": self.batch_plan.to_wire(),
            "validation": validation,
            "ranking": None if self.ranking is None else self.ranking.to_wire(),
            "reporting": self.reporting,
        }


# --------------------------------------------------------------------------- results


class ReliabilityReport(ScientificResult):
    """The report: three uncertainty components as separate attributes.

    The attributes are plain decoded mappings; :attr:`raw` is the full wire report.
    """

    def __init__(self, wire: Mapping[str, Any]) -> None:
        self.raw: Mapping[str, Any] = freeze(wire)

    @property
    def decided_id(self) -> str:
        """The reported decision."""
        return str(self.raw["decided_id"])

    @property
    def source_uncertainty(self) -> SourceUncertainty | str:
        """Component one: spread under the supplied law (``"known_law"`` when declared known)."""
        value = self.raw["source_uncertainty"]
        return SourceUncertainty.from_wire(value) if isinstance(value, Mapping) else value

    @property
    def monte_carlo_error(self) -> MonteCarloErrorReport:
        """Component two: numerical error of the reported values."""
        return MonteCarloErrorReport.from_wire(self.raw["monte_carlo_error"])

    @property
    def selection_uncertainty(self) -> SelectionUncertainty | str:
        """Component three: selection-induced uncertainty (``"not_selected"`` or a mapping)."""
        value = self.raw["selection_uncertainty"]
        return SelectionUncertainty.from_wire(value) if isinstance(value, Mapping) else value

    @property
    def ranking(self) -> RankingReport | None:
        """Ranking statements, when requested."""
        value = self.raw.get("ranking")
        return None if value is None else RankingReport(value)

    @property
    def dependence(self) -> DependenceStatement:
        """The dependence the claim rests on."""
        return DependenceStatement.from_wire(self.raw["dependence"])

    @property
    def coordinate(self) -> CalibrationCoordinate:
        """The calibration coordinate the report sits at."""
        return CalibrationCoordinate.from_wire(self.raw["coordinate"])

    @property
    def closed_routes(self) -> list[dict[str, Any]]:
        """Published routes closed at the coordinate, each with its refusal code."""
        return [dict(route) for route in self.raw["closed_routes"]]

    @property
    def label(self) -> dict[str, Any]:
        """Standing of every number: a point diagnostic, uncalibrated."""
        return dict(self.raw["label"])

    def __repr__(self) -> str:
        return f"ReliabilityReport(decided_id={self.decided_id!r}, label={self.label})"


class RankingReport(ScientificResult):
    """Ranking statements over a finite candidate set (uncalibrated diagnostics)."""

    def __init__(self, wire: Mapping[str, Any]) -> None:
        self.raw: Mapping[str, Any] = freeze(wire)

    @property
    def label(self) -> dict[str, Any]:
        """Paradigm, coupling, scope and calibration standing."""
        return dict(self.raw["label"])

    @property
    def best_by_expected_utility(self) -> str:
        """The candidate with the largest estimated expected utility."""
        return str(self.raw["best_by_expected_utility"])

    @property
    def statements(self) -> Mapping[str, CandidateRankStatement]:
        """Per-candidate statements, keyed by candidate id."""
        return freeze(
            {s["id"]: CandidateRankStatement.from_wire(s) for s in self.raw["statements"]}
        )

    def probability_best(self, candidate: str) -> float:
        """Monte Carlo probability that ``candidate`` is best (ties shared)."""
        return float(self.statements[candidate]["probability_best"]["value"])


class DecisionReliability(ScientificResult):
    """An assessed decision with the draws it was computed from."""

    def __init__(
        self,
        report: ReliabilityReport,
        inspection: Mapping[str, Any],
        artifact: bytes,
        draws: DecisionDraws,
        request: ReliabilityRequest,
        validation: DecisionDraws | None,
    ) -> None:
        #: The three-component report.
        self.report = report
        #: Plain-data summary of the assessment.
        self.inspection: Mapping[str, Any] = freeze(inspection)
        self._artifact = artifact
        self._draws = draws
        self._request = request
        self._validation = validation

    def export(self) -> bytes:
        """The ``decision_reliability_v1`` artifact bytes."""
        return self._artifact

    def digests(self) -> tuple[str, str]:
        """``(draws digest, candidate-set digest)`` of the selection draws."""
        return self._draws.digests()

    def publish(self, route: str) -> None:
        """Publish a ranking probability, confidence set or interval as a calibrated claim.

        Raises:
            DecisionReliabilityRefusal: ``cell_not_licensed`` until a coverage record measures the
                route at this report's coordinate.
        """
        publish(route, self.report.coordinate)

    def combined_uncertainty(self) -> float:
        """One combined uncertainty number.

        Raises:
            DecisionReliabilityRefusal: always ``uncertainty_components_not_combined``.
        """
        request = ReliabilityRequest(
            functional=self._request.functional,
            decision=self._request.decision,
            batch_plan=self._request.batch_plan,
            reporting="combined",
        )
        assess(self._draws, request, validation=None)
        raise AssertionError("a combined request is always refused")  # pragma: no cover


@dataclass(repr=False, frozen=True)
class ConsumedDecisionReliability(ScientificResult):
    """A consumed ``decision_reliability_v1`` artifact: the report recomputed from its draws."""

    report: ReliabilityReport
    inspection: dict[str, Any]


# --------------------------------------------------------------------------- operations


def assess(
    draws: DecisionDraws,
    request: ReliabilityRequest | None = None,
    *,
    validation: DecisionDraws | None = None,
    seed: int = 0,
    artifact_id: str = _DEFAULT_ARTIFACT_ID,
    cancel: Any = None,
) -> DecisionReliability:
    """Assess the reliability of a decision from its draws.

    Raises:
        DecisionReliabilityRefusal: undeclared or contradicted dependence
            (``sampling_dependence_unknown``, ``joint_law_required``), adaptively generated
            candidates without a search record (``candidate_search_record_required``),
            validation draws that are not independent of the selection draws
            (``validation_stream_not_independent``), a combined request
            (``uncertainty_components_not_combined``) or a malformed carrier or request.
    """
    if not isinstance(draws, DecisionDraws):
        raise CausalTypeError("draws must be a DecisionDraws")
    request = request or ReliabilityRequest()
    result, refusal = _native.decision_reliability_assess(
        _dumps(draws.to_wire()),
        None if validation is None else _dumps(validation.to_wire()),
        _dumps(request.to_wire()),
        artifact_id,
        _count(seed, "seed", minimum=0, maximum=2**64 - 1, reason_code="invalid_argument"),
        cancel,
    )
    _refuse(refusal)
    assert result is not None
    artifact, report_json, inspection_json = result
    return DecisionReliability(
        ReliabilityReport(json.loads(report_json)),
        json.loads(inspection_json),
        bytes(artifact),
        draws,
        request,
        validation,
    )


def rank(
    draws: DecisionDraws,
    request: RankingRequest | None = None,
    *,
    seed: int = 0,
    cancel: Any = None,
) -> RankingReport:
    """Rank a finite candidate set: best, top-``k`` and better-than-baseline probabilities.

    Raises:
        DecisionReliabilityRefusal: as :func:`assess` for the carrier and its dependence.
    """
    request = request or RankingRequest()
    report, refusal = _native.decision_reliability_rank(
        _dumps(draws.to_wire()),
        _dumps(request.to_wire()),
        _count(seed, "seed", minimum=0, maximum=2**64 - 1, reason_code="invalid_argument"),
        cancel,
    )
    _refuse(refusal)
    assert report is not None
    return RankingReport(json.loads(report))


def publish(route: str, coordinate: Mapping[str, Any]) -> None:
    """Publish a ranking or optimality route at a calibration coordinate.

    Every route refuses with ``cell_not_licensed`` until a coverage record measures it at its
    coordinate (candidate-set size, Monte Carlo budget, selection rule).

    Raises:
        DecisionReliabilityRefusal: ``cell_not_licensed``.
    """
    _check_choice(route, _ROUTES, "route")
    _refuse(_native.decision_reliability_publish(route, _dumps(dict(coordinate))))


def consume_decision_reliability_artifact(
    artifact: bytes,
    *,
    draws_digest: str | None = None,
    candidate_set_digest: str | None = None,
    selection_stream: StreamId | None = None,
    validation_stream: StreamId | None = None,
    utility_unit: str | None = None,
    decided_id: str | None = None,
    seed: int = 0,
) -> ConsumedDecisionReliability:
    """Independently consume ``decision_reliability_v1`` bytes by recomputation.

    The report is recomputed from the carried draws and compared bit for bit; every retained
    expectation that is given must equal the artifact's.

    Raises:
        DecisionReliabilityRefusal: a damaged container, an edited draw, stream, declaration or
            claimed value, or a retained expectation that differs.
    """
    expected = {
        "draws_digest": draws_digest,
        "candidate_set_digest": candidate_set_digest,
        "selection_stream": None if selection_stream is None else selection_stream.to_wire(),
        "validation_stream": None if validation_stream is None else validation_stream.to_wire(),
        "utility_unit": utility_unit,
        "decided_id": decided_id,
    }
    result, refusal = _native.decision_reliability_consume(
        bytes(artifact),
        _dumps(expected),
        _count(seed, "seed", minimum=0, maximum=2**64 - 1, reason_code="invalid_argument"),
        None,
    )
    _refuse(refusal)
    assert result is not None
    return ConsumedDecisionReliability(
        ReliabilityReport(json.loads(result[0])), json.loads(result[1])
    )
