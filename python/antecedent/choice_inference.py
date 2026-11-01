"""Inference from repeated categorical choices.

A choice study is declared once (:class:`ChoiceStudy`: ordered options, what a choice means, the
population, every respondent's choices, the retained sources reused as prior and the looks and
stopping rule) and validated by the Rust contract: the same observation row is never counted
twice (the evidence ledger judges overlap), a source from another meaning or population is refused, and a missing element is a typed refusal.

:class:`ChoiceInferenceSession` then runs one route at a time, each authorized by the procedure
matrix: the pooled Dirichlet posterior, the hierarchical posterior of repeated choices, or the
contract's stopping decision on either. The exact law of the sequential procedure under a declared
kernel (:meth:`~ChoiceInferenceSession.operating_characteristics`) and the expected value of
sample information of candidate studies (:meth:`~ChoiceInferenceSession.rank_studies`) come from
the same contract. Results export as ``choice_inference_result_v1`` and are consumed by replay.

Every result is point-only with calibration ``unmeasured``; the interval route refuses with
``cell_not_licensed`` until coverage is measured. Stated preferences do not identify behavioural or
causal outcomes. Every number, identity and refusal is computed by the Rust core.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from . import _native
from ._artifact_protocol import (
    ArtifactExpectation,
    ExportedArtifact,
    artifact_consumer,
    artifact_export,
    container_bytes,
)
from ._refusal import StructuredRefusal, raise_refusal
from ._scientific import ScientificResult
from .adaptive_inference import StoppingRule
from .errors import CausalTypeError, CausalValueError
from .execution import Execution, resolve_execution
from .outcomes import outcome_operation

__all__ = [
    "Candidate",
    "ChoiceInferenceRefusal",
    "ChoiceStudy",
    "ConsumedChoiceResult",
    "Hierarchical",
    "HierarchicalPosterior",
    "HierarchicalStopping",
    "Kernel",
    "OperatingCharacteristics",
    "Posterior",
    "PosteriorResult",
    "Respondent",
    "RetainedSource",
    "StudyRank",
    "Stopping",
    "StoppingResult",
    "ChoiceInferenceSession",
    "consume_choice_inference_result_artifact",
]

CALIBRATION = "unmeasured"


class ChoiceInferenceRefusal(StructuredRefusal):
    """A typed refusal of choice inference.

    A :class:`~antecedent.errors.CausalUnsupportedError` with a registered ``reason_code`` and a
    ``choice_inference.*`` ``detail``. A refused contract lists every ``(code, detail)`` in
    :attr:`refusals`; this refusal is the first.
    """

    def __init__(self, refusal: Any, *, text: str | None = None) -> None:
        super().__init__(refusal, text=text)
        self.refusals: tuple[tuple[str, str], ...] = tuple(
            (str(r["code"]), str(r["detail"])) for r in (refusal.get("refusals") or ())
        )

    @property
    def details(self) -> tuple[str, ...]:
        """Every refused detail, in the order the defects were found."""
        return tuple(detail for _, detail in self.refusals) or ((self.detail or ""),)


def _unwrap(pair: tuple[Any, str | None]) -> Any:
    value, refusal = pair
    raise_refusal(refusal, ChoiceInferenceRefusal)
    return value


def _bits(value: float) -> int:
    number = float(value)
    if not math.isfinite(number):
        raise CausalValueError("a declared number must be finite")
    return int(struct.unpack("<Q", struct.pack("<d", number))[0])


def _json(wire: Any) -> str:
    return json.dumps(wire, allow_nan=False, separators=(",", ":"))


# ------------------------------------------------------------------------------ declaration


@dataclass(frozen=True)
class Respondent:
    """One respondent and how often they chose each option, in the declared option order."""

    id: str
    counts: Sequence[int]


@dataclass(frozen=True)
class RetainedSource:
    """Retained choice counts reused as prior evidence at a power-prior ``discount``.

    A source whose meaning or population differs from the study is refused. A free-text
    rationale does not establish transport; the contract also refuses a source that
    shares observation rows with the respondents or another source.
    """

    label: str
    options: Sequence[str]
    meaning: str
    population: str
    counts: Sequence[float]
    observation_ids: Sequence[str]
    discount: float
    transfer_rationale: str | None = None
    posterior_artifact: bytes | None = None
    posterior_identity: str | None = None

    @classmethod
    def from_result(
        cls,
        artifact: bytes,
        expected_identity: str,
        label: str,
        discount: float,
        *,
        execution: Execution | None = None,
    ) -> RetainedSource:
        """Replay a native pooled posterior and retain its exact source and identity."""
        execution = resolve_execution(execution)
        artifact = container_bytes(artifact)
        wire = json.loads(
            _unwrap(
                execution._call(
                    _native.choice_inference_retained_source,
                    bytes(artifact),
                    expected_identity,
                    label,
                    float(discount),
                    execution.cancel,
                )
            )
        )
        return cls(**wire, posterior_artifact=bytes(artifact), posterior_identity=expected_identity)


@dataclass(frozen=True)
class ChoiceStudy:
    """The retained analysis contract of one choice study.

    ``outcome`` is ``categorical_choice`` (one choice per respondent, pooled) or
    ``repeated_choice`` (several choices per respondent, hierarchical). ``looks`` are respondents
    enrolled at each interim analysis (the last is final). ``stopping_rule`` is a
    :class:`~antecedent.adaptive_inference.StoppingRule`; a study without a rule is a final
    analysis only when it declares :meth:`StoppingRule.no_stopping`. Nothing is defaulted: an
    omitted ``looks`` or ``stopping_rule`` is refused by the contract with its own detail.
    """

    label: str
    outcome: Literal["categorical_choice", "repeated_choice"]
    options: Sequence[str]
    meaning: str
    population: str
    respondents: Sequence[Respondent]
    initial_prior: Sequence[float]
    sources: Sequence[RetainedSource] = ()
    looks: Sequence[int] | None = None
    stopping_rule: StoppingRule | None = None

    def to_wire(self) -> dict[str, Any]:
        stopping = "no_stopping" if self.stopping_rule is None else self.stopping_rule.kind
        monitoring = "final_analysis_only" if stopping == "no_stopping" else "fixed_interim_looks"
        return {
            "label": self.label,
            "assignment": "fixed_randomization",
            "outcome": self.outcome,
            "monitoring": monitoring,
            "stopping": stopping,
            "option_labels": list(self.options),
            "meaning": self.meaning,
            "population": self.population,
            "respondents": [
                {"id": r.id, "counts": [int(c) for c in r.counts]} for r in self.respondents
            ],
            "initial_prior_bits": [_bits(a) for a in self.initial_prior],
            "sources": [
                {
                    "label": s.label,
                    "option_labels": list(s.options),
                    "meaning": s.meaning,
                    "population": s.population,
                    "count_bits": [_bits(c) for c in s.counts],
                    "observation_ids": list(s.observation_ids),
                    "discount_bits": _bits(s.discount),
                    "transfer_rationale": s.transfer_rationale,
                    "posterior_artifact": None
                    if s.posterior_artifact is None
                    else list(s.posterior_artifact),
                    "posterior_identity": s.posterior_identity,
                }
                for s in self.sources
            ],
            "looks": None if self.looks is None else [int(n) for n in self.looks],
            "stopping_rule": None if self.stopping_rule is None else self.stopping_rule._wire(),
        }


# --------------------------------------------------------------------------------- requests


@dataclass(frozen=True)
class Posterior:
    """The pooled Dirichlet posterior after ``look`` on the prior the retained sources qualify."""

    look: int

    def _wire(self) -> dict[str, Any]:
        return {"kind": "posterior", "look": int(self.look), "shape_bits": [], "parameter_bits": []}


@dataclass(frozen=True)
class Stopping:
    """The contract's posterior stopping rule on the pooled posterior after ``look``."""

    look: int

    def _wire(self) -> dict[str, Any]:
        return {"kind": "stopping", "look": int(self.look), "shape_bits": [], "parameter_bits": []}


@dataclass(frozen=True)
class Hierarchical:
    """The hierarchical posterior of repeated choices after ``look``.

    ``shapes`` are the Dirichlet shapes of the population shares; ``ln kappa`` (the concentration)
    is Normal(``log_concentration_mean``, ``log_concentration_sd``).
    """

    look: int
    shapes: Sequence[float]
    log_concentration_mean: float
    log_concentration_sd: float
    _kind = "hierarchical"

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": self._kind,
            "look": int(self.look),
            "shape_bits": [_bits(a) for a in self.shapes],
            "parameter_bits": [
                _bits(self.log_concentration_mean),
                _bits(self.log_concentration_sd),
            ],
        }


@dataclass(frozen=True)
class HierarchicalStopping(Hierarchical):
    """The contract's expected-loss rule on the hierarchical posterior after ``look``."""

    _kind = "hierarchical_stopping"


Request = Posterior | Stopping | Hierarchical | HierarchicalStopping


# ---------------------------------------------------------------------------------- results


@dataclass(repr=False, frozen=True)
class PosteriorResult(ScientificResult):
    """The pooled Dirichlet posterior: per-option functionals and the prior's provenance."""

    alpha: tuple[float, ...]
    mean: tuple[float, ...]
    probability_best: tuple[float, ...]
    expected_loss: tuple[float, ...]
    borrowed_observations: float
    conflicting_sources: int
    max_conflict_log_bayes_factor: float
    method: str
    calibration: str = CALIBRATION


@dataclass(repr=False, frozen=True)
class HierarchicalPosterior(ScientificResult):
    """The hierarchical Dirichlet-multinomial posterior of repeated choices."""

    mean: tuple[float, ...]
    probability_best: tuple[float, ...]
    expected_loss: tuple[float, ...]
    log_concentration_mean: float
    log_concentration_sd: float
    correlation_mean: float
    log_marginal_likelihood: float
    method: str
    calibration: str = CALIBRATION


@dataclass(repr=False, frozen=True)
class StoppingResult(ScientificResult):
    """The stopping decision: the best option, its posterior summaries and whether to stop."""

    best_option: int
    probability_best: float
    expected_loss: float
    stop: bool
    method: str
    calibration: str = CALIBRATION


ChoiceResult = PosteriorResult | HierarchicalPosterior | StoppingResult


def _result(method: str, values: Sequence[float]) -> ChoiceResult:
    v = [float(x) for x in values]
    if method.endswith("_stopping"):
        return StoppingResult(int(v[0]), v[1], v[2], bool(v[3]), method)
    if method.startswith("hierarchical"):
        k = (len(v) - 4) // 3
        return HierarchicalPosterior(
            tuple(v[0 : 3 * k : 3]),
            tuple(v[1 : 3 * k : 3]),
            tuple(v[2 : 3 * k : 3]),
            v[3 * k],
            v[3 * k + 1],
            v[3 * k + 2],
            v[3 * k + 3],
            method,
        )
    k = (len(v) - 3) // 4
    return PosteriorResult(
        tuple(v[0 : 4 * k : 4]),
        tuple(v[1 : 4 * k : 4]),
        tuple(v[2 : 4 * k : 4]),
        tuple(v[3 : 4 * k : 4]),
        v[4 * k],
        int(v[4 * k + 1]),
        v[4 * k + 2],
        method,
    )


@dataclass(frozen=True)
class Kernel:
    """The outcome mechanism of a hypothesized truth: fixed ``shares`` or a Dirichlet prior
    predictive with ``alpha`` shapes."""

    shares: Sequence[float] | None = None
    alpha: Sequence[float] | None = None

    def _wire(self) -> dict[str, Any]:
        if (self.shares is None) == (self.alpha is None):
            raise CausalValueError("a kernel is either fixed shares or a prior predictive alpha")
        if self.shares is not None:
            return {"kind": "fixed", "shares": [float(s) for s in self.shares]}
        assert self.alpha is not None
        return {"kind": "prior_predictive", "alpha": [float(a) for a in self.alpha]}


@dataclass(repr=False, frozen=True)
class OperatingCharacteristics(ScientificResult):
    """Exact probabilities of the declared procedure under the declared kernel only.

    Not a bound over other truths and not a coverage statement.
    """

    selection: tuple[float, ...]
    undecided: float
    expected_enrolled: float
    rule_stop: tuple[float, ...]
    horizon_probability: float
    work_units: int
    statement: str


@dataclass(frozen=True)
class Candidate:
    """A candidate additional study: ``observations`` choices at ``cost``."""

    label: str
    observations: int
    cost: float = 0.0


@dataclass(repr=False, frozen=True)
class StudyRank(ScientificResult):
    """The expected value of sample information of one candidate, net of cost."""

    label: str
    observations: int
    cost: float
    evsi: float
    standard_error: float
    net_value: float
    rank_uncertain: bool
    method: dict[str, Any]
    calibration: str = CALIBRATION


# ---------------------------------------------------------------------------------- session


class ChoiceInferenceSession:
    """A prepared choice-study analysis over one validated contract."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self._summary: dict[str, Any] = json.loads(handle.summary_json())

    @classmethod
    def prepare(
        cls, study: ChoiceStudy, *, execution: Execution | None = None
    ) -> ChoiceInferenceSession:
        """Validate the retained choice study.

        Raises:
            ChoiceInferenceRefusal: one detail per missing element and per inconsistency
                (frame, rows, source scope, ledger overlap, looks, rule).
        """
        if not isinstance(study, ChoiceStudy):
            raise CausalTypeError("study must be a ChoiceStudy")
        execution = resolve_execution(execution)
        return cls(
            _unwrap(
                execution._call(
                    _native.choice_inference_prepare, _json(study.to_wire()), execution.cancel
                )
            )
        )

    @property
    def contract_identity(self) -> str:
        """Canonical identity (hex) of the contract."""
        return str(self._handle.contract_identity())

    @property
    def procedure(self) -> str:
        """The procedure cell identifier."""
        return str(self._summary["procedure"])

    @property
    def status(self) -> dict[str, Any]:
        """What the procedure matrix does with this cell."""
        return dict(self._summary["status"])

    @outcome_operation
    def analyze(self, request: Request, *, execution: Execution | None = None) -> ChoiceResult:
        """Run the route a request names.

        Raises:
            ChoiceInferenceRefusal: ``cell_not_licensed`` for a paradigm the cell has no route
                for, a route of the wrong outcome structure, or a numerical refusal.
        """
        execution = resolve_execution(execution)
        wire = json.loads(
            _unwrap(
                execution._call(
                    self._handle.analyse, _json(request._wire()), cancel=execution.cancel
                )
            )
        )
        return _result(wire["method"], wire["values"])

    @outcome_operation
    def operating_characteristics(
        self, kernel: Kernel, *, execution: Execution | None = None
    ) -> OperatingCharacteristics:
        """The exact law of the contract's sequential procedure under ``kernel``."""
        execution = resolve_execution(execution)
        wire = json.loads(
            _unwrap(
                execution._call(
                    self._handle.operating_characteristics,
                    _json(kernel._wire()),
                    cancel=execution.cancel,
                )
            )
        )
        return OperatingCharacteristics(
            selection=tuple(wire["selection"]),
            undecided=float(wire["undecided"]),
            expected_enrolled=float(wire["expected_enrolled"]),
            rule_stop=tuple(wire["rule_stop"]),
            horizon_probability=float(wire["horizon_probability"]),
            work_units=int(wire["work_units"]),
            statement=str(wire["statement"]),
        )

    @outcome_operation
    def rank_studies(
        self,
        look: int,
        values: Sequence[float],
        candidates: Sequence[Candidate],
        *,
        seed: int = 0,
        execution: Execution | None = None,
    ) -> tuple[StudyRank, ...]:
        """Rank candidate studies by EVSI about the pooled posterior at ``look``, net of cost.

        ``values`` is the value of each option when chosen, per unit share.
        """
        execution = resolve_execution(execution)
        payload = {
            "look": int(look),
            "values": [float(v) for v in values],
            "candidates": [
                {"label": c.label, "observations": int(c.observations), "cost": float(c.cost)}
                for c in candidates
            ],
            "seed": int(seed),
        }
        rows = json.loads(
            _unwrap(
                execution._call(self._handle.rank_studies, _json(payload), cancel=execution.cancel)
            )
        )
        return tuple(
            StudyRank(
                label=r["label"],
                observations=int(r["observations"]),
                cost=float(r["cost"]),
                evsi=float(r["evsi"]),
                standard_error=float(r["standard_error"]),
                net_value=float(r["net_value"]),
                rank_uncertain=bool(r["rank_uncertain"]),
                method=dict(r["method"]),
            )
            for r in rows
        )

    @outcome_operation
    def rank_hierarchical_studies(
        self,
        request: Hierarchical,
        values: Sequence[float],
        candidates: Sequence[Candidate],
        *,
        execution: Execution | None = None,
    ) -> tuple[StudyRank, ...]:
        """Rank one- or two-choice studies of one new independent respondent.

        All choices in each candidate come from the same respondent and preserve the
        hierarchical model's within-respondent dependence.
        """
        if type(request) is not Hierarchical:
            raise CausalTypeError("request must be a Hierarchical posterior request")
        execution = resolve_execution(execution)
        payload = {
            "request": request._wire(),
            "values": [float(v) for v in values],
            "candidates": [
                {"label": c.label, "observations": int(c.observations), "cost": float(c.cost)}
                for c in candidates
            ],
        }
        rows = json.loads(
            _unwrap(
                execution._call(
                    self._handle.rank_hierarchical_studies, _json(payload), cancel=execution.cancel
                )
            )
        )
        return tuple(
            StudyRank(
                label=r["label"],
                observations=int(r["observations"]),
                cost=float(r["cost"]),
                evsi=float(r["evsi"]),
                standard_error=float(r["standard_error"]),
                net_value=float(r["net_value"]),
                rank_uncertain=bool(r["rank_uncertain"]),
                method=dict(r["method"]),
            )
            for r in rows
        )

    def withdraw_source(self, label: str, reason: str) -> None:
        """Withdraw a retained source from this session's current evidence ledger."""
        _unwrap(self._handle.withdraw_source(label, reason))

    @outcome_operation
    def consume_current(
        self, artifact: bytes, expected_identity: str, *, execution: Execution | None = None
    ) -> ConsumedChoiceResult:
        """Replay a result only while this session's required sources remain active."""
        execution = resolve_execution(execution)
        artifact = container_bytes(artifact)
        handle = _unwrap(
            execution._call(
                self._handle.consume_current, bytes(artifact), expected_identity, execution.cancel
            )
        )
        return _consumed_result(handle)

    def interval(self) -> None:
        """Ask for an interval or coverage statement without naming a result.

        Raises:
            ChoiceInferenceRefusal: always ``cell_not_licensed``.
        """
        raise_refusal(self._handle.interval(), ChoiceInferenceRefusal)

    def measured_interval(self, request: Request, *, execution: Execution | None = None) -> Any:
        """The interval of one result of a measured choice cell, bound to attesting coverage.

        Raises:
            ChoiceInferenceRefusal: ``cell_not_licensed`` for a study outside the measured cells,
                or evidence that is absent or no longer attesting.
        """
        execution = resolve_execution(execution)
        from ._measured_inference import MeasuredInference

        native = _unwrap(
            execution._call(
                self._handle.measured_interval, _json(request._wire()), cancel=execution.cancel
            )
        )
        return MeasuredInference._from_native(native)

    @artifact_export
    def export(
        self,
        request: Request,
        artifact_id: str = "choice-result",
        *,
        execution: Execution | None = None,
    ) -> ExportedArtifact[ConsumedChoiceResult]:
        """Write ``choice_inference_result_v1`` for a request, with its producer identity."""
        execution = resolve_execution(execution)
        data, identity = _unwrap(
            execution._call(
                self._handle.export_result,
                _json(request._wire()),
                artifact_id,
                cancel=execution.cancel,
            )
        )
        kind = "choice_inference_result_v1"
        return ExportedArtifact(bytes(data), kind, ArtifactExpectation(kind, identity))


# -------------------------------------------------------------------------------- consumer


@dataclass(repr=False, frozen=True)
class ConsumedChoiceResult(ScientificResult):
    """A replayed result: procedure, route and the values recomputed by the consumer."""

    identity: str
    procedure: str
    contract_identity: str
    method: str
    result: ChoiceResult
    closed_guarantee: str
    calibration: str = field(default=CALIBRATION)


def _consumed_result(handle: Any) -> ConsumedChoiceResult:
    wire = json.loads(handle.summary_json())
    return ConsumedChoiceResult(
        identity=wire["identity"],
        procedure=wire["procedure"],
        contract_identity=wire["contract_identity"],
        method=wire["method"],
        result=_result(wire["method"], wire["values"]),
        closed_guarantee=wire["closed_guarantee"],
    )


@outcome_operation
@artifact_consumer(
    proof="choice_inference.consume_choice_inference_result_artifact: independent semantic consumer replays the contract through the evidence ledger and recomputes the stored result",
    replayable=True,
)
def consume_choice_inference_result_artifact(
    artifact: bytes, expected_identity: str | None = None, *, execution: Execution | None = None
) -> ConsumedChoiceResult:
    """Independently consume a choice inference result by replay and recomputation.

    ``expected_identity`` is an identity the consumer retained independently of the bytes.

    Raises:
        ChoiceInferenceRefusal: the contract's own refusal, ``choice_inference.wrong_result``
            for another identity, or a stored value that does not recompute.
    """
    execution = resolve_execution(execution)
    artifact = container_bytes(artifact)
    if not isinstance(artifact, (bytes, bytearray, memoryview)):
        raise CausalTypeError("artifact must be bytes")
    handle = _unwrap(
        execution._call(
            _native.choice_inference_result_consume,
            bytes(artifact),
            expected_identity,
            execution.cancel,
        )
    )
    return _consumed_result(handle)
