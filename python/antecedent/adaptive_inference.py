"""Inference from adaptive experiments.

Data from an experiment whose assignment adapts to outcomes, that is looked at before it ends, or
that stops on a rule, are not a fixed sample. :class:`AdaptiveDesign` is the retained analysis
contract: the assigned arms, the policy's realized assignment probabilities at every unit, the
exposure history, outcome availability and delay, the interim analyses and the stopping rule::

    from antecedent import adaptive_inference as ai

    design = ai.AdaptiveDesign(
        label="trial", assignment="thompson_beta_bernoulli", outcome="single_outcome",
        monitoring="fixed_interim_looks", stopping="efficacy",
        arm_labels=("control", "treatment"), assigned_arm=arms, outcomes=[[y] for y in ys],
        policy_history=[ai.PolicyEpoch(0, ai.policy_identity("thompson/uniform"))],
        assignment_probabilities=rows, exposure=arms,
        availability=ai.Availability(look_of_each_unit, "immediate"),
        interim_analyses=[ai.InterimAnalysis(50, 0.5), ai.InterimAnalysis(100, 1.0)],
        stopping_rule=ai.StoppingRule.posterior_probability_best(0.95),
    )
    session = ai.AdaptiveInferenceSession.prepare(design)
    posterior = session.analyse(ai.BayesianPosterior(look=1))

What it decides, and what it does not:

* **Complete contract.** A missing element is refused with ``adaptive_contract_incomplete`` and its
  own detail (``adaptive_inference.policy_history_missing``, ...), one per element; nothing is
  defaulted. The final sample is never analysed as if the design had been fixed:
  :meth:`AdaptiveInferenceSession.fixed_sample` raises ``adaptive_design_ignored``.
* **One paradigm per route.** Bayesian posterior (:class:`BayesianPosterior`), decision-theoretic
  stopping (:class:`PosteriorStopping`), adaptively weighted frequentist scores
  (:class:`AdaptiveWeighting`), an anytime-valid confidence sequence (:class:`ConfidenceSequence`)
  and group-sequential boundaries (:class:`GroupSequential`). The procedure matrix
  (:func:`procedure_matrix`) says which paradigms each assignment x outcome x monitoring x
  stopping cell is licensed for; a guarantee for two components never licenses their pairing.
* **Intervals only where measured.** :meth:`AdaptiveInferenceSession.interval` (an unqualified
  request) raises ``cell_not_licensed``. :meth:`AdaptiveInferenceSession.measured_interval`
  returns the interval of one result for the procedure cells whose whole-procedure coverage was
  measured, bound to its coverage record, and refuses every other procedure.
* **A participant-level family.** :class:`ParticipantProcedure` declares three or four arms,
  participant-level randomization, three or six binary responses per participant (a
  beta-binomial count with a declared or jointly estimated intra-participant correlation), historical informative
  priors with declared discounts, Thompson allocation with a minimum exploration probability, and
  expected-loss, futility and recruitment-cap decisions at completed-participant batch boundaries.
  Every look reads a posterior claim and an anytime-valid confidence-sequence claim, in separate
  labelled fields. Borrowing is frozen before recruitment (:meth:`ParticipantProcedure.amend_borrowing`
  refuses a change after the first look), and a history is replayed look by look
  (:meth:`ParticipantProcedure.read_history`) and exported with its procedure section
  (:meth:`AdaptiveInferenceSession.export_contract_with_procedure`).
* **Artifacts.** :meth:`AdaptiveInferenceSession.export_contract` writes
  ``adaptive_monitoring_contract_v1``; :meth:`AdaptiveInferenceSession.export_result` writes
  ``adaptive_inference_result_v1``. The consumers replay the contract and recompute the values, and
  the consumed object names the procedure, the stopping rule and the paradigm.

Every number, identity and refusal is computed in Rust; this module builds the declarations and
presents the results.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, overload

from . import _native
from ._refusal import StructuredRefusal, raise_refusal
from ._scientific import ScientificResult
from ._validate import count as _count
from .errors import CausalTypeError, CausalValueError

if TYPE_CHECKING:
    from ._artifact_protocol import ExportedArtifact

__all__ = [
    "ArmPosterior",
    "PosteriorSummary",
    "StoppingSummary",
    "WeightingSummary",
    "ArmSequence",
    "SequenceSummary",
    "SequentialLook",
    "SequentialSummary",
    "AdaptiveDesign",
    "AdaptiveInferenceRefusal",
    "AdaptiveInferenceSession",
    "AdaptiveResult",
    "AdaptiveProcedure",
    "AdaptiveWeighting",
    "AnytimeProcedureReport",
    "AnytimeSettings",
    "ArmPrior",
    "AssignmentRule",
    "Availability",
    "BayesianProcedureReport",
    "CorrelationPrior",
    "DERIVED_UNMEASURED",
    "DecisionProcedureReport",
    "ExposureModel",
    "FinalSelection",
    "FinalAnalysis",
    "FixedSampleProcedureReport",
    "FutilityRule",
    "GroupSequentialProcedureReport",
    "HistoricalPrior",
    "InversionProcedureReport",
    "OutcomeObservation",
    "OutcomeStructure",
    "ParticipantDeclaration",
    "ParticipantDecisions",
    "ParticipantHistory",
    "ParticipantProcedure",
    "ParticipantReadings",
    "ParticipantSimulation",
    "ProcedureDeclaration",
    "ProcedureElements",
    "ProcedureLaw",
    "ProcedureObservedTerminal",
    "ProcedureReport",
    "ProcedureSimulation",
    "ProcedureStopping",
    "RepeatedMeasuresDeclaration",
    "RepeatedMeasuresProcedure",
    "RepeatedMeasuresReport",
    "UnitAssignment",
    "WeightingProcedureReport",
    "WithinUnitCorrelation",
    "BayesianPosterior",
    "ConfidenceSequence",
    "ConsumedAdaptiveResult",
    "ConsumedMonitoringContract",
    "GroupSequential",
    "InterimAnalysis",
    "PolicyEpoch",
    "PosteriorStopping",
    "ProcedureCell",
    "StoppingRule",
    "consume_adaptive_inference_result_artifact",
    "consume_adaptive_monitoring_contract_artifact",
    "policy_identity",
    "procedure_matrix",
]


class AdaptiveInferenceRefusal(StructuredRefusal):
    """A typed refusal of inference from an adaptive experiment.

    A :class:`~antecedent.errors.CausalUnsupportedError` with a registered ``reason_code`` (for
    example ``adaptive_contract_incomplete``, ``adaptive_design_ignored``,
    ``adaptive_assumption_violated``, ``cell_not_licensed``) and an ``adaptive_inference.*``
    ``detail``. When a contract is refused for several defects, :attr:`refusals` lists every
    ``(code, detail)`` pair and this refusal is the first.
    """

    def __init__(self, refusal: Any, *, text: str | None = None) -> None:
        super().__init__(refusal, text=text)
        #: ``(code, detail)`` of every refusal of a refused contract.
        self.refusals: tuple[tuple[str, str], ...] = tuple(
            (str(r["code"]), str(r["detail"])) for r in (refusal.get("refusals") or ())
        )

    @property
    def details(self) -> tuple[str, ...]:
        """Every refused detail, in the order the defects were found."""
        return tuple(detail for _, detail in self.refusals) or ((self.detail or ""),)


def _unwrap(pair: tuple[Any, str | None]) -> Any:
    value, refusal = pair
    raise_refusal(refusal, AdaptiveInferenceRefusal)
    return value


def _raise_if(payload: str | None) -> None:
    raise_refusal(payload, AdaptiveInferenceRefusal)


# --------------------------------------------------------------------------- declarations


def _bits(value: float) -> int:
    number = float(value)
    if not math.isfinite(number):
        raise CausalValueError("a declared number must be finite")
    return int(struct.unpack("<Q", struct.pack("<d", number))[0])


def policy_identity(text: str) -> str:
    """A 64-hex policy-state identity derived from a stable description of the policy state."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PolicyEpoch:
    """The first unit governed by one policy state and that state's identity."""

    first_unit: int
    identity: str


@dataclass(frozen=True)
class Availability:
    """The look at which each outcome became available (``None``: never) and the delay mechanism.

    ``delay`` is ``immediate``, ``ignorable_administrative`` or ``outcome_dependent``.
    """

    available_at_look: Sequence[int | None]
    delay: str


@dataclass(frozen=True)
class InterimAnalysis:
    """Units enrolled at one interim analysis and the information fraction it reached."""

    units_enrolled: int
    information_fraction: float


@dataclass(frozen=True)
class StoppingRule:
    """A stopping rule: its kind, parameters and whether it ended the trial."""

    kind: str
    spec: str
    ended_by_rule: bool = False
    spending: str | None = None
    sidedness: str | None = None
    parameter: float | None = None
    max_units: int | None = None

    @classmethod
    def no_stopping(cls) -> StoppingRule:
        """No rule: the planned horizon is run."""
        return cls("no_stopping", "no_stopping")

    @classmethod
    def alpha_spending(
        cls,
        spending: str,
        total_alpha: float,
        sidedness: str,
        *,
        ended_by_rule: bool = False,
    ) -> StoppingRule:
        """Efficacy boundaries by alpha spending (``obrien_fleming_type`` or ``pocock_type``)."""
        return cls(
            "efficacy",
            "alpha_spending",
            ended_by_rule,
            spending=spending,
            sidedness=sidedness,
            parameter=total_alpha,
        )

    @classmethod
    def posterior_probability_best(
        cls, threshold: float, *, ended_by_rule: bool = False
    ) -> StoppingRule:
        """Stop for efficacy when an arm's posterior probability of being best exceeds a level."""
        return cls("efficacy", "posterior_probability_best", ended_by_rule, parameter=threshold)

    @classmethod
    def posterior_expected_loss(
        cls, loss_threshold: float, *, ended_by_rule: bool = False
    ) -> StoppingRule:
        """Stop when the best arm's posterior expected loss falls below a threshold."""
        return cls(
            "decision_theoretic", "posterior_expected_loss", ended_by_rule, parameter=loss_threshold
        )

    @classmethod
    def resource_cap(cls, max_units: int, *, ended_by_rule: bool = False) -> StoppingRule:
        """Stop at a resource cap."""
        return cls("resource", "resource_cap", ended_by_rule, max_units=max_units)

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "ended_by_rule": bool(self.ended_by_rule),
            "spec": self.spec,
            "spending": self.spending,
            "sidedness": self.sidedness,
            "parameter_bits": None if self.parameter is None else _bits(self.parameter),
            "max_units": self.max_units,
        }


AssignmentName = Literal[
    "fixed_randomization",
    "thompson_beta_bernoulli",
    "epsilon_greedy",
    "upper_confidence_bound",
    "contextual_bandit",
    "batched_adaptive",
]
OutcomeName = Literal["single_outcome", "repeated_measures", "hierarchical_clustered"]
MonitoringName = Literal[
    "final_analysis_only", "fixed_interim_looks", "group_sequential", "continuous_monitoring"
]
StoppingName = Literal["no_stopping", "efficacy", "futility", "resource", "decision_theoretic"]


@dataclass(frozen=True)
class AdaptiveDesign:
    """The retained analysis contract of one adaptive experiment.

    ``assignment``, ``outcome``, ``monitoring`` and ``stopping`` name the procedure cell (see
    :func:`procedure_matrix`). Each of ``policy_history``, ``assignment_probabilities``,
    ``exposure``, ``availability``, ``interim_analyses`` and ``stopping_rule`` may be ``None``;
    preparing the design then refuses with that element's own detail.
    """

    label: str
    assignment: AssignmentName
    outcome: OutcomeName
    monitoring: MonitoringName
    stopping: StoppingName
    arm_labels: Sequence[str]
    assigned_arm: Sequence[int]
    outcomes: Sequence[Sequence[float]]
    policy_history: Sequence[PolicyEpoch] | None = None
    assignment_probabilities: Sequence[Sequence[float]] | None = None
    exposure: Sequence[int] | None = None
    availability: Availability | None = None
    interim_analyses: Sequence[InterimAnalysis] | None = None
    stopping_rule: StoppingRule | None = None
    probability_floor: float = 0.01

    def to_wire(self) -> dict[str, Any]:
        """The declaration as the native wire mapping (floats as IEEE bit patterns)."""
        return {
            "label": self.label,
            "assignment": self.assignment,
            "outcome": self.outcome,
            "monitoring": self.monitoring,
            "stopping": self.stopping,
            "arm_labels": list(self.arm_labels),
            "assigned_arm": [
                _count(a, "a", minimum=0, maximum=2**64 - 1, reason_code="invalid_argument")
                for a in self.assigned_arm
            ],
            "policy_history": None
            if self.policy_history is None
            else [
                {
                    "first_unit": _count(
                        e.first_unit,
                        "e.first_unit",
                        minimum=0,
                        maximum=2**64 - 1,
                        reason_code="invalid_argument",
                    ),
                    "identity": e.identity,
                }
                for e in self.policy_history
            ],
            "assignment_probability_bits": None
            if self.assignment_probabilities is None
            else [[_bits(p) for p in row] for row in self.assignment_probabilities],
            "exposure": None
            if self.exposure is None
            else [
                _count(a, "a", minimum=0, maximum=2**64 - 1, reason_code="invalid_argument")
                for a in self.exposure
            ],
            "availability": None
            if self.availability is None
            else {
                "available_at_look": [
                    None
                    if look is None
                    else _count(
                        look, "look", minimum=0, maximum=2**64 - 1, reason_code="invalid_argument"
                    )
                    for look in self.availability.available_at_look
                ],
                "delay": self.availability.delay,
            },
            "outcome_bits": [[_bits(y) for y in row] for row in self.outcomes],
            "interim_analyses": None
            if self.interim_analyses is None
            else [
                {
                    "units_enrolled": _count(
                        a.units_enrolled,
                        "a.units_enrolled",
                        minimum=0,
                        maximum=2**64 - 1,
                        reason_code="invalid_argument",
                    ),
                    "information_fraction_bits": _bits(a.information_fraction),
                }
                for a in self.interim_analyses
            ],
            "stopping_rule": None if self.stopping_rule is None else self.stopping_rule._wire(),
            "probability_floor_bits": _bits(self.probability_floor),
        }


# --------------------------------------------------------------------------- requests


@dataclass(frozen=True)
class BayesianPosterior:
    """Beta-Bernoulli posterior of every arm after ``look`` (prior shapes at least one)."""

    look: int
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "bayesian_posterior",
            "look": _count(
                self.look, "look", minimum=0, maximum=2**64 - 1, reason_code="invalid_argument"
            ),
            "prior_alpha": float(self.prior_alpha),
            "prior_beta": float(self.prior_beta),
        }


@dataclass(frozen=True)
class PosteriorStopping:
    """The contract's posterior stopping rule evaluated after ``look``."""

    look: int
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "posterior_stopping",
            "look": _count(
                self.look, "look", minimum=0, maximum=2**64 - 1, reason_code="invalid_argument"
            ),
            "prior_alpha": float(self.prior_alpha),
            "prior_beta": float(self.prior_beta),
        }


@dataclass(frozen=True)
class AdaptiveWeighting:
    """Adaptively weighted augmented inverse-probability arm mean or contrast after ``look``.

    ``scheme`` is ``uniform`` or ``variance_stabilizing``. ``minus_arm=None`` estimates the mean of
    ``plus_arm``.
    """

    look: int
    plus_arm: int
    minus_arm: int | None = None
    scheme: str = "variance_stabilizing"
    prior_mean: float = 0.5
    prior_strength: float = 1.0
    size_weighted: bool = False

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "adaptive_weighting",
            "look": _count(
                self.look, "look", minimum=0, maximum=2**64 - 1, reason_code="invalid_argument"
            ),
            "plus_arm": _count(
                self.plus_arm,
                "plus_arm",
                minimum=0,
                maximum=2**64 - 1,
                reason_code="invalid_argument",
            ),
            "minus_arm": None
            if self.minus_arm is None
            else _count(
                self.minus_arm,
                "minus_arm",
                minimum=0,
                maximum=2**64 - 1,
                reason_code="invalid_argument",
            ),
            "scheme": self.scheme,
            "prior_mean": float(self.prior_mean),
            "prior_strength": float(self.prior_strength),
            "size_weighted": bool(self.size_weighted),
        }


@dataclass(frozen=True)
class ConfidenceSequence:
    """Time-uniform confidence sequences for bounded arm means after ``look``."""

    look: int
    alpha: float = 0.05
    tuning_samples: float = 100.0
    contrast: tuple[int, int] | None = None

    def _wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {
            "kind": "confidence_sequence",
            "look": _count(
                self.look, "look", minimum=0, maximum=2**64 - 1, reason_code="invalid_argument"
            ),
            "alpha": float(self.alpha),
            "tuning_samples": float(self.tuning_samples),
        }
        if self.contrast is not None:
            wire["plus_arm"], wire["minus_arm"] = (
                _count(a, "a", minimum=0, maximum=2**64 - 1, reason_code="invalid_argument")
                for a in self.contrast
            )
        return wire


@dataclass(frozen=True)
class GroupSequential:
    """The contract's alpha-spending rule applied to the looks up to ``look``."""

    look: int
    control_arm: int = 0
    treatment_arm: int = 1

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "group_sequential",
            "look": _count(
                self.look, "look", minimum=0, maximum=2**64 - 1, reason_code="invalid_argument"
            ),
            "control_arm": _count(
                self.control_arm,
                "control_arm",
                minimum=0,
                maximum=2**64 - 1,
                reason_code="invalid_argument",
            ),
            "treatment_arm": _count(
                self.treatment_arm,
                "treatment_arm",
                minimum=0,
                maximum=2**64 - 1,
                reason_code="invalid_argument",
            ),
        }


Request = (
    BayesianPosterior | PosteriorStopping | AdaptiveWeighting | ConfidenceSequence | GroupSequential
)


def _request_json(request: Request) -> str:
    wire = getattr(request, "_wire", None)
    if wire is None:
        raise CausalTypeError("a request is one of the adaptive inference request classes")
    return json.dumps(wire(), allow_nan=False, separators=(",", ":"))


# --------------------------------------------------------------------------- results


@dataclass(repr=False, frozen=True, slots=True)
class ArmPosterior(ScientificResult):
    alpha: float
    beta: float
    mean: float
    variance: float
    probability_best: float
    expected_loss: float


@dataclass(repr=False, frozen=True, slots=True)
class PosteriorSummary(ScientificResult):
    arms: tuple[ArmPosterior, ...]


@dataclass(repr=False, frozen=True, slots=True)
class StoppingSummary(ScientificResult):
    best_arm: int
    probability_best: float
    expected_loss: float
    stop: bool


@dataclass(repr=False, frozen=True, slots=True)
class WeightingSummary(ScientificResult):
    estimate: float
    variance: float
    units: int
    effective_units: float
    max_weight_share: float


@dataclass(repr=False, frozen=True, slots=True)
class ArmSequence(ScientificResult):
    count: int
    mean: float
    lower: float
    upper: float


@dataclass(repr=False, frozen=True, slots=True)
class SequenceSummary(ScientificResult):
    arms: tuple[ArmSequence, ...]
    contrast_lower: float
    contrast_upper: float


@dataclass(repr=False, frozen=True, slots=True)
class SequentialLook(ScientificResult):
    critical: float
    statistic: float
    spent: float


@dataclass(repr=False, frozen=True, slots=True)
class SequentialSummary(ScientificResult):
    stopped_at: int | None
    looks: tuple[SequentialLook, ...]


AdaptivePayload = (
    PosteriorSummary | StoppingSummary | WeightingSummary | SequenceSummary | SequentialSummary
)
PayloadT = TypeVar("PayloadT", bound=AdaptivePayload)


@dataclass(repr=False, frozen=True)
class AdaptiveResult(ScientificResult, Generic[PayloadT]):
    """A named method result; all readings retain the whole-procedure point-only label."""

    paradigm: str
    procedure: str
    method: str
    payload: PayloadT
    closed_guarantee: str
    contract_identity: str
    look: int | None
    stopping_rule: dict[str, Any] | None
    claim: str = "point_only"
    _wire_values: tuple[float, ...] = field(default=(), repr=False, compare=False)
    _export_session: Any = field(default=None, repr=False, compare=False)
    _export_request: Request | None = field(default=None, repr=False, compare=False)

    def export(self, artifact_id: str = "adaptive-result", *, cancel: Any = None) -> bytes:
        """Export this request under the immutable prepared adaptive contract."""
        if self._export_session is None or self._export_request is None:
            raise CausalValueError(
                "this detached adaptive reading has no retained request to export"
            )
        return bytes(
            self._export_session.export_result(self._export_request, artifact_id, cancel=cancel)
        )

    def artifact(
        self, artifact_id: str = "adaptive-result", *, cancel: Any = None
    ) -> ExportedArtifact[ConsumedAdaptiveResult]:
        from ._artifact_protocol import ExportedArtifact

        return ExportedArtifact(
            self.export(artifact_id, cancel=cancel),
            "adaptive_inference_result_v1",
            metadata={"contract_identity": self.contract_identity, "method": self.method},
        )

    def to_rows(self) -> list[dict[str, Any]]:
        return [
            {**row, "paradigm": self.paradigm, "method": self.method, "claim": self.claim}
            for row in self.payload.to_rows()
        ]


def _payload(paradigm: str, method: str, values: tuple[float, ...]) -> AdaptivePayload:
    if "posterior" in paradigm and "stopp" not in method:
        return PosteriorSummary(
            tuple(ArmPosterior(*values[i : i + 6]) for i in range(0, len(values), 6))
        )
    if "stopp" in method or "stopp" in paradigm:
        return StoppingSummary(int(values[0]), values[1], values[2], bool(values[3]))
    if "weight" in paradigm:
        return WeightingSummary(values[0], values[1], int(values[2]), values[3], values[4])
    if "anytime" in paradigm:
        return SequenceSummary(
            tuple(
                ArmSequence(int(values[i]), *values[i + 1 : i + 4])
                for i in range(0, len(values) - 2, 4)
            ),
            values[-2],
            values[-1],
        )
    return SequentialSummary(
        None if values[0] < 0 else int(values[0]),
        tuple(SequentialLook(*values[i : i + 3]) for i in range(1, len(values), 3)),
    )


@dataclass(frozen=True)
class ProcedureCell:
    """One cell of the procedure matrix and what the engine does with it."""

    id: str
    assignment: str
    outcome: str
    monitoring: str
    stopping: str
    status: str
    paradigms: tuple[str, ...] = ()
    owner: str | None = None
    reason: str | None = None


def procedure_matrix() -> tuple[ProcedureCell, ...]:
    """Every combination of assignment, outcome structure, monitoring and stopping rule.

    Each cell is ``licensed`` (with its paradigms), ``closed_incoherent``, ``closed_pairing`` or
    ``carried_forward`` (with its owner); none is unlisted.
    """
    cells = []
    for cell in json.loads(_native.adaptive_procedure_matrix()):
        status = cell["status"]
        cells.append(
            ProcedureCell(
                id=cell["id"],
                assignment=cell["assignment"],
                outcome=cell["outcome"],
                monitoring=cell["monitoring"],
                stopping=cell["stopping"],
                status=status["class"],
                paradigms=tuple(status.get("paradigms", ())),
                owner=status.get("owner"),
                reason=status.get("reason"),
            )
        )
    return tuple(cells)


# --------------------------------------------------------------------------- session


@dataclass(repr=False, frozen=True)
class InterimEvidence(ScientificResult):
    """The observations behind each interim analysis, counted once."""

    look_sizes: tuple[int, ...]
    declared_slots: int
    distinct_observations: int
    repeated_slots: int
    state_digest: str
    reuse_result: str
    #: ``(code, detail)`` of the refusal to pool the looks as independent, when there is one.
    independent_pooling_refusal: tuple[str, str] | None


def _result(wire: dict[str, Any]) -> AdaptiveResult:
    return AdaptiveResult(
        paradigm=wire["paradigm"],
        procedure=wire["procedure"],
        method=wire["method"],
        payload=_payload(
            wire["paradigm"],
            wire["method"],
            tuple(float("nan") if v is None else float(v) for v in wire["values"]),
        ),
        _wire_values=tuple(float("nan") if v is None else float(v) for v in wire["values"]),
        closed_guarantee=wire["closed_guarantee"],
        contract_identity=wire["contract_identity"],
        look=int(wire["look"]),
        stopping_rule=wire.get("stopping_rule"),
    )


class AdaptiveInferenceSession:
    """A prepared adaptive-experiment analysis over one validated contract."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self._summary: dict[str, Any] = json.loads(handle.summary_json())

    @classmethod
    def prepare(cls, design: AdaptiveDesign) -> AdaptiveInferenceSession:
        """Validate the retained analysis contract.

        Raises:
            AdaptiveInferenceRefusal: ``adaptive_contract_incomplete`` with one detail per missing
                element, or ``invalid_argument`` details for an inconsistent record.
        """
        if not isinstance(design, AdaptiveDesign):
            raise CausalTypeError("design must be an AdaptiveDesign")
        wire = json.dumps(design.to_wire(), allow_nan=False, separators=(",", ":"))
        return cls(_unwrap(_native.adaptive_inference_prepare(wire)))

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
        """What the procedure matrix does with this cell (``class`` and its paradigms or owner)."""
        return dict(self._summary["status"])

    @property
    def is_adaptive(self) -> bool:
        """Whether the design adapted, was looked at or stopped."""
        return bool(self._summary["adaptive"])

    def derive_procedure(
        self, elements: ProcedureElements, *, cancel: Any = None
    ) -> AdaptiveProcedure:
        """The whole data-collection procedure of this experiment.

        The contract supplies the looks, delay and stopping parameters; ``elements`` supplies what
        the contract does not keep (the assignment rule, loss retention, the exposure-failure
        model, final selection, posterior priors). The uncertainty of the procedure is then
        derived by :meth:`AdaptiveProcedure.report`.

        Raises:
            AdaptiveInferenceRefusal: one detail per missing element and per disagreement between
                the elements and the retained contract.
        """
        if not isinstance(elements, ProcedureElements):
            raise CausalTypeError("elements must be a ProcedureElements")
        wire = _procedure_json(elements.to_wire())
        handle = _unwrap(_native.adaptive_procedure_from_contract(self._handle, wire, cancel))
        return AdaptiveProcedure(handle)

    @overload
    def analyze(
        self, request: BayesianPosterior, *, cancel: Any = None
    ) -> AdaptiveResult[PosteriorSummary]: ...

    @overload
    def analyze(
        self, request: PosteriorStopping, *, cancel: Any = None
    ) -> AdaptiveResult[StoppingSummary]: ...

    @overload
    def analyze(
        self, request: AdaptiveWeighting, *, cancel: Any = None
    ) -> AdaptiveResult[WeightingSummary]: ...

    @overload
    def analyze(
        self, request: ConfidenceSequence, *, cancel: Any = None
    ) -> AdaptiveResult[SequenceSummary]: ...

    @overload
    def analyze(
        self, request: GroupSequential, *, cancel: Any = None
    ) -> AdaptiveResult[SequentialSummary]: ...

    def analyze(self, request: Request, *, cancel: Any = None) -> AdaptiveResult:
        """Run the paradigm a request names.

        Raises:
            AdaptiveInferenceRefusal: ``cell_not_licensed`` for a paradigm the cell has no route
                for, ``adaptive_assumption_violated`` for noncompliance, outcome-dependent delay
                or a probability below the floor, ``cancelled_no_claim`` on cancellation.
        """
        wire = _unwrap(self._handle.analyse(_request_json(request), cancel=cancel))
        result = _result(json.loads(wire))
        from ._freeze import freeze

        object.__setattr__(result, "_export_session", self)
        object.__setattr__(result, "_export_request", freeze(request))
        return result

    def analyse(self, request: Request, *, cancel: Any = None) -> AdaptiveResult[Any]:
        """British spelling of :meth:`analyze`."""
        return self.analyze(request, cancel=cancel)

    def fixed_sample(self) -> None:
        """Analyse as if the design had been fixed in advance.

        Raises:
            AdaptiveInferenceRefusal: ``adaptive_design_ignored`` for data from an adaptive design.
        """
        _raise_if(self._handle.fixed_sample())

    def interval(self) -> None:
        """Ask for an interval or coverage statement for the whole procedure.

        Raises:
            AdaptiveInferenceRefusal: always ``cell_not_licensed``.
        """
        _raise_if(self._handle.interval())

    def measured_interval(self, request: Request, *, cancel: Any = None) -> Any:
        """The interval of one result of a measured procedure cell.

        The result is sealed and replayed, its interval is built from the retained history with
        the construction whose whole-procedure coverage was measured, and it is bound to the
        currently attesting coverage record. Measured cells: Thompson sampling with a uniform prior
        and a posterior-probability stopping rule (credible interval of the second arm),
        epsilon-greedy repeated measures with a final analysis (weighted contrast), epsilon-greedy
        confidence sequences (arm mean or contrast, valid at every look) and fixed randomization
        with group-sequential monitoring (repeated confidence interval).

        Returns:
            A :class:`antecedent.MeasuredInference` carrying the scalar, its interval and the
            governing record.

        Raises:
            AdaptiveInferenceRefusal: ``cell_not_licensed`` for any other procedure, prior,
                threshold, exploration mass or look schedule, or for evidence that is absent or no
                longer attesting.
        """
        from ._measured_inference import MeasuredInference

        native = _unwrap(self._handle.measured_interval(_request_json(request), cancel=cancel))
        return MeasuredInference._from_native(native)

    def interim_evidence(self) -> InterimEvidence:
        """The observations behind each look, counted once."""
        wire = json.loads(_unwrap(self._handle.interim_evidence()))
        pooling = wire.get("independent_pooling_refusal")
        return InterimEvidence(
            look_sizes=tuple(int(n) for n in wire["look_sizes"]),
            declared_slots=int(wire["declared_slots"]),
            distinct_observations=int(wire["distinct_observations"]),
            repeated_slots=int(wire["repeated_slots"]),
            state_digest=wire["state_digest"],
            reuse_result=wire["reuse_result"],
            independent_pooling_refusal=None
            if pooling is None
            else (str(pooling["code"]), str(pooling["detail"])),
        )

    def export(
        self, request: Request, artifact_id: str = "adaptive-result", *, cancel: Any = None
    ) -> bytes:
        """Export a selected adaptive request with its whole-procedure declarations."""
        return self.export_result(request, artifact_id, cancel=cancel)

    def artifact(
        self, request: Request, artifact_id: str = "adaptive-result", *, cancel: Any = None
    ) -> ExportedArtifact[ConsumedAdaptiveResult]:
        from ._artifact_protocol import ExportedArtifact

        return ExportedArtifact(
            self.export_result(request, artifact_id, cancel=cancel),
            "adaptive_inference_result_v1",
            metadata={
                "contract_identity": self.contract_identity,
                "request": json.loads(_request_json(request)),
            },
        )

    def artifact_contract(
        self, paradigm: str, artifact_id: str = "adaptive-monitoring"
    ) -> ExportedArtifact[ConsumedMonitoringContract]:
        from ._artifact_protocol import ExportedArtifact

        return ExportedArtifact(
            self.export_contract(paradigm, artifact_id),
            "adaptive_monitoring_contract_v1",
            metadata={"contract_identity": self.contract_identity, "paradigm": paradigm},
        )

    def export_contract(self, paradigm: str, artifact_id: str = "adaptive-monitoring") -> bytes:
        """Write ``adaptive_monitoring_contract_v1`` for a paradigm."""
        return bytes(_unwrap(self._handle.export_contract(paradigm, artifact_id)))

    def export_contract_with_procedure(
        self,
        paradigm: str,
        procedure: ParticipantProcedure,
        artifact_id: str = "adaptive-monitoring",
    ) -> bytes:
        """Write ``adaptive_monitoring_contract_v1`` with the participant-level procedure section.

        The section holds the arms' priors with every historical source, discount and effective
        sample size, the allocation floor, the decisions and the reading at every performed look.
        A consumer replays it: it recomputes each batch's allocation from the completed
        participants and the declared priors and compares it with the retained probabilities.

        Raises:
            AdaptiveInferenceRefusal: the cell is not licensed for the paradigm, or the retained
                history is not the declared procedure's (``assignment_not_replayed``,
                ``stopping_not_replayed``).
        """
        if not isinstance(procedure, ParticipantProcedure):
            raise CausalTypeError("procedure must be a ParticipantProcedure")
        return bytes(
            _unwrap(
                self._handle.export_contract_with_procedure(
                    paradigm, procedure._handle, artifact_id
                )
            )
        )

    def export_result(
        self, request: Request, artifact_id: str = "adaptive-result", *, cancel: Any = None
    ) -> bytes:
        """Write ``adaptive_inference_result_v1`` for a request."""
        return bytes(
            _unwrap(self._handle.export_result(_request_json(request), artifact_id, cancel=cancel))
        )


# --------------------------------------------------------------------------- consumers


@dataclass(repr=False, frozen=True)
class ConsumedMonitoringContract(ScientificResult):
    """A replayed monitoring contract: the procedure, the stopping rule and the paradigm."""

    identity: str
    procedure: str
    paradigm: str
    stopping_rule: dict[str, Any] | None
    contract_identity: str
    units: int
    arms: int
    looks: int
    #: ``critical`` and ``spend_increments`` of an alpha-spending rule, else ``None``.
    boundaries: dict[str, tuple[float, ...]] | None
    declaration: dict[str, Any] = field(repr=False, default_factory=dict)
    #: The replayed participant-level procedure section (priors, readings), else ``None``.
    participant_procedure: dict[str, Any] | None = field(repr=False, default=None)


@dataclass(repr=False, frozen=True)
class ConsumedAdaptiveResult(ScientificResult):
    """A replayed result: procedure, rule, paradigm and the values recomputed by the consumer."""

    identity: str
    procedure: str
    paradigm: str
    stopping_rule: dict[str, Any] | None
    method: str
    payload: AdaptivePayload
    _wire_values: tuple[float, ...]
    closed_guarantee: str


def consume_adaptive_monitoring_contract_artifact(
    artifact: bytes, expected_identity: str | None = None
) -> ConsumedMonitoringContract:
    """Independently consume a monitoring contract by replay and recomputation.

    ``expected_identity`` is an identity the consumer retained independently of the bytes; with it
    a consistently resealed contract of another experiment is refused.

    Raises:
        AdaptiveInferenceRefusal: ``adaptive_inference.wrong_contract`` for another identity,
            ``adaptive_inference.report_replay_mismatch`` for a stored value that does not replay,
            the contract's own refusal for a missing or inconsistent element, and
            ``adaptive_design_ignored`` or ``cell_not_licensed`` for a paradigm the design may not
            use.
    """
    if not isinstance(artifact, (bytes, bytearray, memoryview)):
        raise CausalTypeError("artifact must be bytes")
    handle = _unwrap(
        _native.adaptive_monitoring_contract_consume(bytes(artifact), expected_identity)
    )
    wire = json.loads(handle.summary_json())
    boundaries = wire.get("boundaries")
    return ConsumedMonitoringContract(
        identity=wire["identity"],
        procedure=wire["procedure"],
        paradigm=wire["paradigm"],
        stopping_rule=wire.get("stopping_rule"),
        contract_identity=wire["contract_identity"],
        units=int(wire["units"]),
        arms=int(wire["arms"]),
        looks=int(wire["looks"]),
        boundaries=None
        if boundaries is None
        else {k: tuple(float(x) for x in v) for k, v in boundaries.items()},
        declaration=wire.get("declaration") or {},
        participant_procedure=wire.get("participant_procedure"),
    )


def consume_adaptive_inference_result_artifact(
    artifact: bytes, expected_identity: str | None = None, *, cancel: Any = None
) -> ConsumedAdaptiveResult:
    """Independently consume an adaptive inference result by replay and recomputation.

    Raises:
        AdaptiveInferenceRefusal: as :func:`consume_adaptive_monitoring_contract_artifact`, and
            ``adaptive_inference.report_replay_mismatch`` for a stored value that does not
            recompute.
    """
    if not isinstance(artifact, (bytes, bytearray, memoryview)):
        raise CausalTypeError("artifact must be bytes")
    handle = _unwrap(
        _native.adaptive_inference_result_consume(bytes(artifact), expected_identity, cancel)
    )
    wire = json.loads(handle.summary_json())
    return ConsumedAdaptiveResult(
        identity=wire["identity"],
        procedure=wire["procedure"],
        paradigm=wire["paradigm"],
        stopping_rule=wire.get("stopping_rule"),
        method=wire["method"],
        payload=_payload(
            wire["paradigm"],
            wire["method"],
            tuple(float("nan") if v is None else float(v) for v in wire["values"]),
        ),
        _wire_values=tuple(float("nan") if v is None else float(v) for v in wire["values"]),
        closed_guarantee=wire["closed_guarantee"],
    )


# ------------------------------------------------------------------ whole-procedure uncertainty
#
# The uncertainty of an adaptively collected result is a property of the whole data-collection
# procedure (policy, looks, delay and loss, exposure failures, stopping, selection), not of the
# final sample. The classes below declare that procedure and ask for its derived statements. Every
# result is labelled ``derived_unmeasured``: exact (or simulated, with a Monte Carlo error) under
# the stated model, with no interval or coverage licence.

DERIVED_UNMEASURED = "derived_unmeasured"


def _procedure_json(wire: Any) -> str:
    try:
        return json.dumps(wire, allow_nan=False, separators=(",", ":"))
    except ValueError as error:
        raise CausalValueError("a declared number must be finite") from error


def _prior_wire(prior: tuple[float, float]) -> dict[str, Any]:
    alpha, beta = prior
    return {"prior_alpha": float(alpha), "prior_beta": float(beta)}


@dataclass(frozen=True)
class AssignmentRule:
    """How units are assigned to arms, as a function of the observed history."""

    kind: str
    probabilities: Sequence[float] | None = None
    epsilon: float | None = None
    exploration: float | None = None
    prior: tuple[float, float] = (1.0, 1.0)
    min_probability: float | None = None

    @classmethod
    def fixed(cls, probabilities: Sequence[float]) -> AssignmentRule:
        """Probabilities fixed in advance."""
        return cls("fixed", probabilities=tuple(float(p) for p in probabilities))

    @classmethod
    def epsilon_greedy(
        cls, epsilon: float, prior: tuple[float, float] = (1.0, 1.0)
    ) -> AssignmentRule:
        """Epsilon-greedy over prior-shrunk running arm means."""
        return cls("epsilon_greedy", epsilon=float(epsilon), prior=prior)

    @classmethod
    def thompson_beta(cls, prior: tuple[float, float] = (1.0, 1.0)) -> AssignmentRule:
        """Beta-Bernoulli Thompson sampling."""
        return cls("thompson_beta", prior=prior)

    @classmethod
    def thompson_beta_floored(
        cls, min_probability: float, prior: tuple[float, float] = (1.0, 1.0)
    ) -> AssignmentRule:
        """Thompson sampling with a minimum exploration probability.

        The posterior probabilities that each arm is best are clamped from below at
        ``min_probability`` and renormalized, so every arm keeps at least that probability.
        """
        return cls("thompson_beta_floored", prior=prior, min_probability=float(min_probability))

    @classmethod
    def upper_confidence_bound(cls, exploration: float = 2.0) -> AssignmentRule:
        """Deterministic UCB1 (assignment probabilities are zero or one)."""
        return cls("upper_confidence_bound", exploration=float(exploration))

    def _wire(self) -> dict[str, Any]:
        if self.kind == "fixed":
            return {"kind": "fixed", "probabilities": list(self.probabilities or ())}
        if self.kind == "epsilon_greedy":
            return {"kind": self.kind, "epsilon": self.epsilon, **_prior_wire(self.prior)}
        if self.kind == "thompson_beta":
            return {"kind": self.kind, **_prior_wire(self.prior)}
        if self.kind == "thompson_beta_floored":
            return {
                "kind": self.kind,
                "min_probability": self.min_probability,
                **_prior_wire(self.prior),
            }
        return {"kind": self.kind, "exploration": self.exploration}


@dataclass(frozen=True)
class OutcomeObservation:
    """When and whether each outcome is observed: a delay in looks and two retention levels."""

    lag_looks: int = 0
    retain_success: float = 1.0
    retain_failure: float = 1.0

    def _wire(self) -> dict[str, Any]:
        return {
            "lag_looks": self.lag_looks,
            "retain_success": float(self.retain_success),
            "retain_failure": float(self.retain_failure),
        }


@dataclass(frozen=True)
class ExposureModel:
    """Exploration failures: the probability each assigned arm is not delivered, and what is."""

    failure_probability: Sequence[float]
    fallback_arm: int
    attribution: Literal["assigned", "delivered"]

    def _wire(self) -> dict[str, Any]:
        return {
            "failure_probability": [float(p) for p in self.failure_probability],
            "fallback_arm": self.fallback_arm,
            "attribution": self.attribution,
        }


@dataclass(frozen=True)
class ProcedureStopping:
    """What a look can decide, with the parameters the derivation needs."""

    kind: str
    threshold: float | None = None
    prior: tuple[float, float] = (1.0, 1.0)
    margin: float | None = None
    critical: Sequence[float] | None = None
    sidedness: str | None = None
    max_units: int | None = None

    @classmethod
    def no_stopping(cls) -> ProcedureStopping:
        """Nothing: the planned horizon is run."""
        return cls("no_stopping")

    @classmethod
    def posterior_probability_best(
        cls, threshold: float, prior: tuple[float, float] = (1.0, 1.0)
    ) -> ProcedureStopping:
        """Stop when an arm's posterior probability of being best reaches ``threshold``."""
        return cls("posterior_probability_best", threshold=float(threshold), prior=prior)

    @classmethod
    def posterior_expected_loss(
        cls, loss_threshold: float, prior: tuple[float, float] = (1.0, 1.0)
    ) -> ProcedureStopping:
        """Stop when the smallest posterior expected loss falls below ``loss_threshold``."""
        return cls("posterior_expected_loss", threshold=float(loss_threshold), prior=prior)

    @classmethod
    def posterior_futility(
        cls, margin: float, threshold: float, prior: tuple[float, float] = (1.0, 1.0)
    ) -> ProcedureStopping:
        """Stop for futility, deciding for arm 0 (the control).

        The procedure stops when every other arm has posterior probability below ``threshold`` of
        exceeding the control by more than ``margin``.
        """
        return cls(
            "posterior_futility", threshold=float(threshold), margin=float(margin), prior=prior
        )

    @classmethod
    def wald_boundaries(cls, critical: Sequence[float], sidedness: str) -> ProcedureStopping:
        """Stop when arm 1's Wald statistic against arm 0 crosses a per-look critical value."""
        return cls(
            "wald_boundaries", critical=tuple(float(c) for c in critical), sidedness=sidedness
        )

    @classmethod
    def resource_cap(cls, max_units: int) -> ProcedureStopping:
        """Stop when the enrolled units reach the cap."""
        return cls("resource_cap", max_units=max_units)

    def _wire(self) -> dict[str, Any]:
        if self.kind == "no_stopping":
            return {"kind": self.kind}
        if self.kind == "posterior_probability_best":
            return {"kind": self.kind, "threshold": self.threshold, **_prior_wire(self.prior)}
        if self.kind == "posterior_expected_loss":
            return {"kind": self.kind, "loss_threshold": self.threshold, **_prior_wire(self.prior)}
        if self.kind == "posterior_futility":
            return {
                "kind": self.kind,
                "margin": self.margin,
                "threshold": self.threshold,
                **_prior_wire(self.prior),
            }
        if self.kind == "wald_boundaries":
            return {
                "kind": self.kind,
                "critical": list(self.critical or ()),
                "sidedness": self.sidedness,
            }
        return {"kind": self.kind, "max_units": self.max_units}


@dataclass(frozen=True)
class FinalSelection:
    """How an arm is selected when no stopping rule decided."""

    kind: str
    prior: tuple[float, float] = (1.0, 1.0)

    @classmethod
    def no_selection(cls) -> FinalSelection:
        """No arm is selected."""
        return cls("no_selection")

    @classmethod
    def highest_observed_mean(cls) -> FinalSelection:
        """The arm with the highest observed mean."""
        return cls("highest_observed_mean")

    @classmethod
    def highest_posterior_mean(cls, prior: tuple[float, float] = (1.0, 1.0)) -> FinalSelection:
        """The arm with the highest posterior mean under ``prior``."""
        return cls("highest_posterior_mean", prior=prior)

    def _wire(self) -> dict[str, Any]:
        if self.kind == "highest_posterior_mean":
            return {"kind": self.kind, **_prior_wire(self.prior)}
        return {"kind": self.kind}


@dataclass(frozen=True)
class ProcedureDeclaration:
    """Everything a producer states about the data-collection procedure.

    Each element but ``arms`` may be ``None``; declaring the procedure then refuses with that
    element's own detail (``adaptive_contract_incomplete``).
    """

    arms: int
    assignment: AssignmentRule | None = None
    enrolment: Sequence[int] | None = None
    observation: OutcomeObservation | None = None
    exposure: ExposureModel | None = None
    stopping: ProcedureStopping | None = None
    selection: FinalSelection | None = None
    context_probabilities: Sequence[float] | None = None

    def to_wire(self) -> dict[str, Any]:
        """The declaration as the native wire mapping."""

        def wire(value: Any) -> Any:
            return None if value is None else value._wire()

        return {
            "arms": self.arms,
            "context_probabilities": None
            if self.context_probabilities is None
            else [float(p) for p in self.context_probabilities],
            "assignment": wire(self.assignment),
            "enrolment": None if self.enrolment is None else [int(n) for n in self.enrolment],
            "observation": wire(self.observation),
            "exposure": wire(self.exposure),
            "stopping": wire(self.stopping),
            "selection": wire(self.selection),
        }


@dataclass(frozen=True)
class ProcedureElements:
    """What the retained analysis contract does not keep.

    The contract holds realized probabilities, exposures, availability, looks and the stopping
    rule's parameters. These are the rest: the assignment rule itself, how lost outcomes relate to
    their values (``retention``), the exposure-failure model, the final selection and the prior of
    a posterior stopping rule (``stopping_prior``).
    """

    assignment: AssignmentRule | None = None
    selection: FinalSelection | None = None
    retention: tuple[float, float] | None = None
    exposure: ExposureModel | None = None
    stopping_prior: tuple[float, float] | None = None
    context_probabilities: Sequence[float] | None = None

    def to_wire(self) -> dict[str, Any]:
        """The elements as the native wire mapping."""
        return {
            "assignment": None if self.assignment is None else self.assignment._wire(),
            "selection": None if self.selection is None else self.selection._wire(),
            "retention": None
            if self.retention is None
            else {"success": float(self.retention[0]), "failure": float(self.retention[1])},
            "exposure": None if self.exposure is None else self.exposure._wire(),
            "stopping_prior": None
            if self.stopping_prior is None
            else {"alpha": float(self.stopping_prior[0]), "beta": float(self.stopping_prior[1])},
            "context_probabilities": None
            if self.context_probabilities is None
            else [float(p) for p in self.context_probabilities],
        }


# --- requests, one per paradigm


@dataclass(frozen=True)
class ProcedureObservedTerminal:
    """The terminal state a real procedure ended in: look, whether the rule stopped, the arm
    decided (if any) and the observed contrast estimate."""

    look: int
    estimate: float
    rule_stopped: bool = False
    decision: int | None = None

    def _wire(self) -> dict[str, Any]:
        return {
            "look": self.look,
            "rule_stopped": bool(self.rule_stopped),
            "decision": self.decision,
            "estimate": float(self.estimate),
        }


@dataclass(frozen=True)
class BayesianProcedureReport:
    """Posterior at ``observed`` counts (success, failure per cell) and the posterior-mean gap."""

    prior: tuple[float, float] = (1.0, 1.0)
    observed: Sequence[int] | None = None

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "bayesian_posterior",
            **_prior_wire(self.prior),
            "observed": None if self.observed is None else [int(n) for n in self.observed],
        }


@dataclass(frozen=True)
class DecisionProcedureReport:
    """Stopping-time law, expected sample size, decision law and regret under ``truth``."""

    truth: Sequence[float]

    def _wire(self) -> dict[str, Any]:
        return {"kind": "decision_theoretic", "truth": [float(p) for p in self.truth]}


@dataclass(frozen=True)
class FixedSampleProcedureReport:
    """Exact sampling law of the fixed design's contrast ``plus - minus`` under ``truth``."""

    truth: Sequence[float]
    plus: int = 1
    minus: int = 0
    observed: ProcedureObservedTerminal | None = None

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "fixed_sample",
            "truth": [float(p) for p in self.truth],
            "plus": self.plus,
            "minus": self.minus,
            "observed": None if self.observed is None else self.observed._wire(),
        }


@dataclass(frozen=True)
class GroupSequentialProcedureReport:
    """Actual crossing probabilities of the declared boundaries and stagewise inference."""

    truth: Sequence[float]
    level: float = 0.95
    observed_look: int | None = None
    observed_statistic: float | None = None

    def _wire(self) -> dict[str, Any]:
        observed = None
        if self.observed_look is not None and self.observed_statistic is not None:
            observed = {"look": self.observed_look, "statistic": float(self.observed_statistic)}
        return {
            "kind": "group_sequential",
            "truth": [float(p) for p in self.truth],
            "level": float(self.level),
            "observed": observed,
        }


@dataclass(frozen=True)
class WeightingProcedureReport:
    """Simulated law of the adaptively weighted estimate under the whole procedure."""

    truth: Sequence[float]
    replications: int
    seed: int
    estimand: tuple[Literal["arm_mean", "contrast"], int, int] = ("contrast", 1, 0)
    scheme: Literal["uniform", "variance_stabilizing"] = "uniform"
    outcome_prior_mean: float = 0.5
    outcome_prior_strength: float = 2.0
    probability_floor: float = 0.01
    z: float = 1.959963984540054

    def _simulation_wire(self) -> dict[str, Any]:
        kind, first, second = self.estimand
        estimand = (
            {"kind": "arm_mean", "arm": first}
            if kind == "arm_mean"
            else {"kind": "contrast", "plus": first, "minus": second}
        )
        return {
            "replications": self.replications,
            "seed": self.seed,
            "estimand": estimand,
            "scheme": self.scheme,
            "outcome_prior_mean": float(self.outcome_prior_mean),
            "outcome_prior_strength": float(self.outcome_prior_strength),
            "probability_floor": float(self.probability_floor),
        }

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "adaptive_weighting",
            "truth": [float(p) for p in self.truth],
            "z": float(self.z),
            **self._simulation_wire(),
        }


@dataclass(frozen=True)
class AnytimeProcedureReport:
    """Simultaneous time-uniform bounds and their exact exclusion probability under ``truth``."""

    alpha: float = 0.05
    tuning_samples: float = 50.0
    observed: Sequence[int] | None = None
    truth: Sequence[float] | None = None

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "anytime_valid",
            "alpha": float(self.alpha),
            "tuning_samples": float(self.tuning_samples),
            "observed": None if self.observed is None else [int(n) for n in self.observed],
            "truth": None if self.truth is None else [float(p) for p in self.truth],
        }


@dataclass(frozen=True)
class InversionProcedureReport:
    """Confidence set from inverting the whole-procedure law over declared grids.

    Not an inferential paradigm of the procedure matrix; the set carries no coverage claim.
    """

    observed: ProcedureObservedTerminal
    contrasts: Sequence[float]
    baselines: Sequence[float]
    alpha: float = 0.05

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "whole_procedure_inversion",
            "observed": self.observed._wire(),
            "contrasts": [float(c) for c in self.contrasts],
            "baselines": [float(b) for b in self.baselines],
            "alpha": float(self.alpha),
        }


ProcedureReportRequest = (
    BayesianProcedureReport
    | DecisionProcedureReport
    | FixedSampleProcedureReport
    | GroupSequentialProcedureReport
    | WeightingProcedureReport
    | AnytimeProcedureReport
    | InversionProcedureReport
)


# --- results


@dataclass(repr=False, frozen=True)
class ProcedureReport(ScientificResult):
    """One paradigm's derived statement about the whole procedure.

    ``body`` holds the paradigm's numbers; ``claim`` is always ``derived_unmeasured``.
    """

    paradigm: str
    estimand: str
    procedure_identity: str
    contract_identity: str | None
    body: dict[str, Any]
    claim: str = DERIVED_UNMEASURED


@dataclass(repr=False, frozen=True)
class ProcedureLaw(ScientificResult):
    """The exact law of the terminal state under a hypothesized truth."""

    procedure_identity: str
    atoms: tuple[dict[str, Any], ...]
    expected_enrolled: float
    rule_stop_probabilities: tuple[float, ...]
    horizon_probability: float
    cumulative_enrolment: tuple[int, ...]
    work_units: int
    peak_states: int
    claim: str = DERIVED_UNMEASURED


@dataclass(repr=False, frozen=True)
class ProcedureSimulation(ScientificResult):
    """Replicates of the whole procedure with the weighted estimate of each."""

    procedure_identity: str
    replicates: tuple[dict[str, Any], ...]
    mean_enrolled: float
    rule_stop_fraction: float
    claim: str = DERIVED_UNMEASURED
    #: The work the simulation was admitted for and did (``work_planned``, ``work_done``, ...).
    receipt: dict[str, Any] = field(default_factory=dict)


class AdaptiveProcedure:
    """A declared adaptive procedure and the uncertainty derived from it.

    Declare it directly (:meth:`declare`) or from a prepared contract
    (:meth:`AdaptiveInferenceSession.procedure`). A paradigm whose validity premise the procedure
    does not meet raises :class:`AdaptiveInferenceRefusal` with ``cell_not_licensed`` and the
    premise's ``adaptive_inference.*`` detail.
    """

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self._summary: dict[str, Any] = json.loads(handle.summary_json())

    @classmethod
    def declare(cls, declaration: ProcedureDeclaration) -> AdaptiveProcedure:
        """Declare the procedure.

        Raises:
            AdaptiveInferenceRefusal: ``adaptive_contract_incomplete`` with one detail per missing
                element, or ``invalid_argument`` details for an inconsistent element.
        """
        if not isinstance(declaration, ProcedureDeclaration):
            raise CausalTypeError("declaration must be a ProcedureDeclaration")
        wire = _procedure_json(declaration.to_wire())
        return cls(_unwrap(_native.adaptive_procedure_declare(wire)))

    @property
    def identity(self) -> str:
        """Canonical identity (hex) of the declared procedure."""
        return str(self._handle.identity())

    @property
    def contract_identity(self) -> str | None:
        """Identity (hex) of the retained contract the procedure was built from, if any."""
        value = self._summary.get("contract_identity")
        return None if value is None else str(value)

    @property
    def taken_from_contract(self) -> tuple[str, ...]:
        """The elements taken from the retained contract rather than declared."""
        return tuple(self._summary.get("taken_from_contract") or ())

    @property
    def summary(self) -> dict[str, Any]:
        """The validated elements and matrix labels (``family``, ``monitoring``, ...)."""
        return dict(self._summary)

    def law(self, truth: Sequence[float], *, cancel: Any = None) -> ProcedureLaw:
        """The exact law of the terminal state under ``truth`` (one probability per cell)."""
        wire = json.loads(_unwrap(self._handle.law([float(p) for p in truth], cancel)))
        return ProcedureLaw(
            procedure_identity=wire["procedure_identity"],
            atoms=tuple(wire["atoms"]),
            expected_enrolled=float(wire["expected_enrolled"]),
            rule_stop_probabilities=tuple(float(p) for p in wire["rule_stop_probabilities"]),
            horizon_probability=float(wire["horizon_probability"]),
            cumulative_enrolment=tuple(int(n) for n in wire["cumulative_enrolment"]),
            work_units=int(wire["work_units"]),
            peak_states=int(wire["peak_states"]),
        )

    def simulate(
        self, request: WeightingProcedureReport, *, cancel: Any = None
    ) -> ProcedureSimulation:
        """Run the whole procedure under ``request.truth`` and estimate by adaptive weighting."""
        if not isinstance(request, WeightingProcedureReport):
            raise CausalTypeError("request must be a WeightingProcedureReport")
        wire = json.loads(
            _unwrap(
                self._handle.simulate(
                    [float(p) for p in request.truth],
                    _procedure_json(request._simulation_wire()),
                    cancel,
                )
            )
        )
        return ProcedureSimulation(
            procedure_identity=wire["procedure_identity"],
            replicates=tuple(wire["replicates"]),
            mean_enrolled=float(wire["mean_enrolled"]),
            rule_stop_fraction=float(wire["rule_stop_fraction"]),
            receipt=wire.get("receipt") or {},
        )

    def report(self, request: ProcedureReportRequest, *, cancel: Any = None) -> ProcedureReport:
        """One paradigm's derived statement.

        Raises:
            AdaptiveInferenceRefusal: ``cell_not_licensed`` when the procedure does not meet the
                paradigm's premise (its detail names which), or the derivation's own refusal.
        """
        wire = getattr(request, "_wire", None)
        if wire is None:
            raise CausalTypeError("a request is one of the procedure report request classes")
        payload = json.loads(_unwrap(self._handle.report(_procedure_json(wire()), cancel)))
        return ProcedureReport(
            paradigm=payload["paradigm"],
            estimand=payload["estimand"],
            procedure_identity=payload["procedure_identity"],
            contract_identity=payload.get("contract_identity"),
            body=payload["body"],
        )


# --------------------------------------------------------------------------- repeated measures
#
# Units that contribute several correlated measurements (repeated measures, or members of a
# cluster with a unit-level random effect) are a separate outcome structure: the within-unit
# dependence is declared, never defaulted, and every guarantee is about one cell of the procedure
# matrix. Fixed-sample inference after response-adaptive assignment, and an independence
# assumption for dependent measurements, are refused. Results are ``derived_unmeasured``.


@dataclass(frozen=True)
class WithinUnitCorrelation:
    """Correlation of two measurements of one unit."""

    kind: str
    rho: float

    @classmethod
    def compound_symmetric(cls, rho: float) -> WithinUnitCorrelation:
        """Every pair of measurements has correlation ``rho`` (``-1/(m-1) < rho < 1``)."""
        return cls("compound_symmetric", float(rho))

    @classmethod
    def autoregressive(cls, rho: float) -> WithinUnitCorrelation:
        """Measurements ``j`` and ``k`` have correlation ``rho ** |j - k|``."""
        return cls("autoregressive", float(rho))

    def _wire(self) -> dict[str, Any]:
        return {"kind": self.kind, "rho": self.rho}


@dataclass(frozen=True)
class OutcomeStructure:
    """The outcome structure of a unit with several measurements."""

    kind: str
    count: int
    variance: float | None = None
    correlation: WithinUnitCorrelation | None = None
    between_variance: float | None = None
    within_variance: float | None = None

    @classmethod
    def repeated_measures(
        cls,
        measurements: int,
        variance: float,
        correlation: WithinUnitCorrelation | None = None,
    ) -> OutcomeStructure:
        """``measurements`` per unit with marginal ``variance`` and a within-unit correlation.

        Leaving ``correlation`` out is refused at declaration with
        ``adaptive_inference.within_unit_correlation_missing``.
        """
        return cls("repeated_measures", int(measurements), float(variance), correlation)

    @classmethod
    def hierarchical(
        cls,
        members: int,
        between_variance: float | None = None,
        within_variance: float | None = None,
    ) -> OutcomeStructure:
        """A cluster of ``members`` with a unit-level random effect.

        ``y_ij = mu + b_i + e_ij``; a missing variance is refused with its own detail.
        """
        return cls(
            "hierarchical",
            int(members),
            between_variance=None if between_variance is None else float(between_variance),
            within_variance=None if within_variance is None else float(within_variance),
        )

    def _wire(self) -> dict[str, Any]:
        if self.kind == "repeated_measures":
            return {
                "kind": self.kind,
                "measurements": self.count,
                "variance": self.variance,
                "correlation": None if self.correlation is None else self.correlation._wire(),
            }
        return {
            "kind": self.kind,
            "members": self.count,
            "between_variance": self.between_variance,
            "within_variance": self.within_variance,
        }


@dataclass(frozen=True)
class UnitAssignment:
    """Unit-level assignment, a function of the unit summaries of completed units."""

    kind: str
    probabilities: Sequence[float] | None = None
    epsilon: float | None = None
    prior_mean: float = 0.0
    prior_strength: float = 1.0

    @classmethod
    def fixed(cls, probabilities: Sequence[float]) -> UnitAssignment:
        """Probabilities fixed in advance."""
        return cls("fixed", probabilities=tuple(float(p) for p in probabilities))

    @classmethod
    def epsilon_greedy(
        cls, epsilon: float, prior_mean: float = 0.0, prior_strength: float = 1.0
    ) -> UnitAssignment:
        """Epsilon-greedy over prior-shrunk running means of the unit summaries."""
        return cls(
            "epsilon_greedy",
            epsilon=float(epsilon),
            prior_mean=float(prior_mean),
            prior_strength=float(prior_strength),
        )

    def _wire(self) -> dict[str, Any]:
        if self.kind == "fixed":
            return {"kind": "fixed", "probabilities": list(self.probabilities or ())}
        return {
            "kind": self.kind,
            "epsilon": self.epsilon,
            "prior_mean": self.prior_mean,
            "prior_strength": self.prior_strength,
        }


@dataclass(frozen=True)
class FinalAnalysis:
    """The declared final-analysis estimator of the contrast ``plus - minus``."""

    kind: str
    plus: int = 1
    minus: int = 0
    summary: Literal["mean", "gls"] = "mean"
    scheme: Literal["uniform", "variance_stabilizing"] = "uniform"
    prior_mean: float = 0.0
    prior_strength: float = 1.0
    probability_floor: float = 0.05

    @classmethod
    def cluster_robust(
        cls, summary: Literal["mean", "gls"] = "mean", *, plus: int = 1, minus: int = 0
    ) -> FinalAnalysis:
        """Contrast of arm means of unit summaries with the unit-level (cluster-robust) variance.

        A fixed-sample analysis: refused after response-adaptive assignment.
        """
        return cls("cluster_robust", plus=plus, minus=minus, summary=summary)

    @classmethod
    def pooled_independent(cls, *, plus: int = 1, minus: int = 0) -> FinalAnalysis:
        """All measurements pooled as independent; refused when dependence is declared."""
        return cls("pooled_independent", plus=plus, minus=minus)

    @classmethod
    def adaptively_weighted(
        cls,
        summary: Literal["mean", "gls"] = "mean",
        *,
        plus: int = 1,
        minus: int = 0,
        scheme: Literal["uniform", "variance_stabilizing"] = "uniform",
        prior_mean: float = 0.0,
        prior_strength: float = 1.0,
        probability_floor: float = 0.05,
    ) -> FinalAnalysis:
        """The adaptively weighted augmented score over unit summaries (the weighting route)."""
        return cls(
            "adaptively_weighted",
            plus=plus,
            minus=minus,
            summary=summary,
            scheme=scheme,
            prior_mean=float(prior_mean),
            prior_strength=float(prior_strength),
            probability_floor=float(probability_floor),
        )

    def _wire(self) -> dict[str, Any]:
        estimator: dict[str, Any] = {"kind": self.kind}
        if self.kind in ("cluster_robust", "adaptively_weighted"):
            estimator["summary"] = self.summary
        if self.kind == "adaptively_weighted":
            estimator.update(
                scheme=self.scheme,
                prior_mean=self.prior_mean,
                prior_strength=self.prior_strength,
                probability_floor=self.probability_floor,
            )
        return {"estimator": estimator, "plus": self.plus, "minus": self.minus}


@dataclass(frozen=True)
class RepeatedMeasuresDeclaration:
    """Everything a producer states about a repeated-measures or clustered procedure.

    Each element but ``arms`` may be ``None``; declaring then refuses with that element's own
    detail. ``enrolment`` is the units per policy-update batch (all ones is fully sequential);
    ``lag_looks`` is the follow-up delay in batches. Only no stopping is derived for this cell.
    """

    arms: int
    assignment: UnitAssignment | None = None
    enrolment: Sequence[int] | None = None
    structure: OutcomeStructure | None = None
    analysis: FinalAnalysis | None = None
    stopping: ProcedureStopping | None = None
    lag_looks: int = 0

    def to_wire(self) -> dict[str, Any]:
        """The declaration as the native wire mapping."""

        def wire(value: Any) -> Any:
            return None if value is None else value._wire()

        return {
            "arms": self.arms,
            "assignment": wire(self.assignment),
            "enrolment": None if self.enrolment is None else [int(n) for n in self.enrolment],
            "lag_looks": self.lag_looks,
            "structure": wire(self.structure),
            "analysis": wire(self.analysis),
            "stopping": wire(self.stopping),
        }


@dataclass(repr=False, frozen=True)
class RepeatedMeasuresReport(ScientificResult):
    """A derived statement about a repeated-measures or clustered procedure.

    ``cell`` names the matrix cell the statement is about (``id``, ``family``, ``outcome``,
    ``monitoring``, ``stopping``, ``paradigm``); ``route`` names the analysis route; ``body``
    holds the exact closed forms and, for a derived report, the simulated law with its Monte
    Carlo error reported separately. ``claim`` is always ``derived_unmeasured``.
    """

    procedure_identity: str
    cell: dict[str, Any]
    route: str
    body: dict[str, Any]
    claim: str = DERIVED_UNMEASURED


class RepeatedMeasuresProcedure:
    """A declared repeated-measures or clustered adaptive procedure."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self._cell: dict[str, Any] = json.loads(handle.cell_json())

    @classmethod
    def declare(cls, declaration: RepeatedMeasuresDeclaration) -> RepeatedMeasuresProcedure:
        """Declare the procedure.

        Raises:
            AdaptiveInferenceRefusal: ``adaptive_contract_incomplete`` with one detail per missing
                element (``adaptive_inference.within_unit_correlation_missing``,
                ``...between_unit_variance_missing``, ...), or ``invalid_argument`` details.
        """
        if not isinstance(declaration, RepeatedMeasuresDeclaration):
            raise CausalTypeError("declaration must be a RepeatedMeasuresDeclaration")
        wire = _procedure_json(declaration.to_wire())
        return cls(_unwrap(_native.repeated_measures_procedure_declare(wire)))

    @property
    def identity(self) -> str:
        """Canonical identity (hex) of the declared procedure."""
        return str(self._handle.identity())

    @property
    def cell(self) -> dict[str, Any]:
        """The matrix cell of the procedure and the paradigm of its declared analysis."""
        return dict(self._cell)

    def _report(self, wire: dict[str, Any], cancel: Any) -> RepeatedMeasuresReport:
        payload = json.loads(_unwrap(self._handle.report(_procedure_json(wire), cancel)))
        return RepeatedMeasuresReport(
            procedure_identity=payload["procedure_identity"],
            cell=payload["cell"],
            route=payload["route"],
            body=payload["body"],
        )

    def closed_form(
        self, arm_means: Sequence[float], *, cancel: Any = None
    ) -> RepeatedMeasuresReport:
        """The exact closed forms: unit-summary variance, design effect, exact contrast law."""
        wire = {"kind": "closed_form", "arm_means": [float(m) for m in arm_means]}
        return self._report(wire, cancel)

    def derive(
        self,
        arm_means: Sequence[float],
        *,
        replications: int,
        seed: int,
        z: float = 1.96,
        cancel: Any = None,
    ) -> RepeatedMeasuresReport:
        """The whole-procedure simulation under the paradigm of the declared analysis.

        Raises:
            AdaptiveInferenceRefusal: ``cell_not_licensed`` with the premise's detail when the
                declared analysis does not meet its paradigm's premise.
        """
        wire = {
            "kind": "derived",
            "arm_means": [float(m) for m in arm_means],
            "replications": int(replications),
            "seed": int(seed),
            "z": float(z),
        }
        return self._report(wire, cancel)

    def derive_paradigm(
        self,
        paradigm: str,
        arm_means: Sequence[float],
        *,
        replications: int,
        seed: int,
        z: float = 1.96,
        cancel: Any = None,
    ) -> RepeatedMeasuresReport:
        """As :meth:`derive`, naming the paradigm asked for; any other is refused."""
        wire = {
            "kind": "paradigm",
            "paradigm": paradigm,
            "arm_means": [float(m) for m in arm_means],
            "replications": int(replications),
            "seed": int(seed),
            "z": float(z),
        }
        return self._report(wire, cancel)

    def transfer_guarantee(
        self, cell_id: str, paradigm: str, *, cancel: Any = None
    ) -> RepeatedMeasuresReport:
        """Ask whether this cell's guarantee may be stated for another cell (it may not)."""
        wire = {"kind": "cell_transfer", "cell_id": str(cell_id), "paradigm": str(paradigm)}
        return self._report(wire, cancel)


# ---------------------------------------------------------------------- participant-level family
#
# Three or four arms, participant-level randomization, a few binary responses per participant,
# historical informative priors, adaptive allocation and decision-based stopping, all at once. The
# model, the posterior update, the allocation, the decisions and the confidence sequences are
# Rust's (``antecedent_stats::adaptive_inference::participant_procedure`` documents the update
# precisely); this section declares the procedure and presents what it reads. Every result is
# ``derived_unmeasured``: the posterior claim and the anytime-valid frequentist claim of each look
# are separate fields, and neither is a measured coverage of the whole procedure.


@dataclass(frozen=True)
class HistoricalPrior:
    """Historical binary responses an arm borrows from, at a declared power-prior discount.

    ``discount`` is the power-prior exponent ``a0`` in ``[0, 1]``: the source counts as
    ``a0 * trials`` current responses (``0`` ignores it, ``1`` counts it fully). The discount is
    declared and frozen before recruitment.
    """

    source: str
    successes: int
    trials: int
    discount: float

    def _wire(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "successes": int(self.successes),
            "trials": int(self.trials),
            "discount": float(self.discount),
        }


@dataclass(frozen=True)
class ArmPrior:
    """The prior of one arm: an initial Beta prior and the historical sources it borrows from.

    An empty ``historical`` declares no borrowing.
    """

    initial: tuple[float, float] = (1.0, 1.0)
    historical: Sequence[HistoricalPrior] = ()

    def _wire(self) -> dict[str, Any]:
        return {
            "initial_alpha": float(self.initial[0]),
            "initial_beta": float(self.initial[1]),
            "historical": [source._wire() for source in self.historical],
        }


@dataclass(frozen=True)
class FutilityRule:
    """Stop when no treatment arm is likely to beat the control by ``margin``.

    The procedure stops, deciding for the control, when every treatment arm has posterior
    probability below ``threshold`` of exceeding the control by more than ``margin``.
    """

    margin: float
    threshold: float


@dataclass(frozen=True)
class ParticipantDecisions:
    """What the procedure decides at each look, in this order.

    ``loss_threshold``: stop when the smallest posterior expected loss of choosing an arm falls
    below it (decide that arm); ``futility``: stop for futility (decide the control);
    ``participant_cap``: stop at the first look at which this many participants are enrolled.
    """

    loss_threshold: float
    futility: FutilityRule | None = None
    participant_cap: int | None = None

    def _wire(self) -> dict[str, Any]:
        return {
            "loss_threshold": float(self.loss_threshold),
            "futility": None
            if self.futility is None
            else {
                "margin": float(self.futility.margin),
                "threshold": float(self.futility.threshold),
            },
            "participant_cap": None if self.participant_cap is None else int(self.participant_cap),
        }


@dataclass(frozen=True)
class AnytimeSettings:
    """The confidence sequences read at every look.

    ``alpha`` is the total miscoverage of the simultaneous family over the ``K - 1`` comparisons
    against the control; ``tuning_samples`` is the participants per arm at which the sequences are
    tightest.
    """

    alpha: float
    tuning_samples: float


@dataclass(frozen=True)
class CorrelationPrior:
    """A ``Beta(alpha, beta)`` prior on the intra-participant correlation, shapes at least 1.

    Declared in place of a fixed correlation, it is shared by the arms and integrated jointly
    with the arm success probabilities, so the correlation is estimated from the data at every
    look. Frozen before recruitment like the arm priors.
    """

    alpha: float
    beta: float

    def _wire(self) -> dict[str, float]:
        return {"alpha": float(self.alpha), "beta": float(self.beta)}


@dataclass(frozen=True)
class ParticipantDeclaration:
    """Everything a producer states about the participant-level procedure.

    Each element but ``arms`` and ``lag_looks`` may be ``None``; declaring then refuses with that
    element's own detail. Arm 0 is the control. ``responses`` is the binary responses per
    participant; ``correlation`` the declared intra-participant correlation (``0`` declares the
    responses independent) or ``correlation_prior`` a :class:`CorrelationPrior` under which the
    correlation is estimated jointly with the arm means (exactly one of the two);
    ``borrowing`` one :class:`ArmPrior` per arm, frozen before
    recruitment; ``enrolment`` the participants per look; ``lag_looks`` the looks between
    enrolment and completed follow-up; ``min_exploration`` the smallest probability any arm
    keeps under Thompson allocation.
    """

    arms: int
    responses: int | None = None
    correlation: float | None = None
    correlation_prior: CorrelationPrior | None = None
    borrowing: Sequence[ArmPrior] | None = None
    enrolment: Sequence[int] | None = None
    lag_looks: int = 0
    min_exploration: float | None = None
    decisions: ParticipantDecisions | None = None
    anytime: AnytimeSettings | None = None

    def to_wire(self) -> dict[str, Any]:
        """The declaration as the native wire mapping."""
        return {
            "arms": self.arms,
            "responses": self.responses,
            "correlation": None if self.correlation is None else float(self.correlation),
            "correlation_prior": None
            if self.correlation_prior is None
            else self.correlation_prior._wire(),
            "borrowing": None
            if self.borrowing is None
            else [prior._wire() for prior in self.borrowing],
            "enrolment": None if self.enrolment is None else [int(n) for n in self.enrolment],
            "lag_looks": self.lag_looks,
            "min_exploration": None
            if self.min_exploration is None
            else float(self.min_exploration),
            "decisions": None if self.decisions is None else self.decisions._wire(),
            "anytime": None
            if self.anytime is None
            else {
                "alpha": float(self.anytime.alpha),
                "tuning_samples": float(self.anytime.tuning_samples),
            },
        }


@dataclass(frozen=True)
class ParticipantHistory:
    """What a participant-level procedure retained: batches, arms, outcomes and allocations.

    ``successes`` holds each participant's success count among the responses, or ``None`` while
    their follow-up had not completed when the procedure ended; ``assignment`` holds the
    allocation probabilities each participant was randomized under.
    """

    batches: Sequence[int]
    arms: Sequence[int]
    successes: Sequence[int | None]
    assignment: Sequence[Sequence[float]]

    def to_wire(self) -> dict[str, Any]:
        """The history as the native wire mapping."""
        return {
            "batches": [int(n) for n in self.batches],
            "arms": [int(a) for a in self.arms],
            "successes": [None if s is None else int(s) for s in self.successes],
            "assignment": [[float(p) for p in row] for row in self.assignment],
        }

    @classmethod
    def from_wire(cls, wire: dict[str, Any]) -> ParticipantHistory:
        """Rebuild a history from its wire mapping."""
        return cls(
            batches=tuple(int(n) for n in wire["batches"]),
            arms=tuple(int(a) for a in wire["arms"]),
            successes=tuple(None if s is None else int(s) for s in wire["successes"]),
            assignment=tuple(tuple(float(p) for p in row) for row in wire["assignment"]),
        )


@dataclass(repr=False, frozen=True)
class ParticipantReadings(ScientificResult):
    """The reading at every performed look of a retained history.

    Each entry of ``looks`` carries the look's ``assignment_probabilities``, its ``decision``, a
    ``posterior`` claim (paradigm ``bayesian_posterior``) and a ``frequentist`` claim (paradigm
    ``anytime_valid_frequentist``, with the simultaneous comparisons against the control and the
    comparison of the data-selected best treatment, which the simultaneous family covers).
    ``priors`` records every arm's resolved prior with its sources, discounts and effective sample
    size. ``claim`` is always ``derived_unmeasured``.
    """

    procedure_identity: str
    priors: tuple[dict[str, Any], ...]
    looks: tuple[dict[str, Any], ...]
    decision: dict[str, Any]
    claim: str = DERIVED_UNMEASURED


@dataclass(repr=False, frozen=True)
class ParticipantSimulation(ScientificResult):
    """Replicates of the whole participant-level procedure under a hypothesized truth.

    ``receipt`` records the work the simulation was admitted for and did (``work_planned``,
    ``work_done``, ``replications``, ``looks``, ``states``, ``peak_units``,
    ``estimated_peak_bytes``); ``summary`` holds
    measured properties of this simulation, not a coverage statement.
    """

    procedure_identity: str
    receipt: dict[str, Any]
    summary: dict[str, Any]
    replicates: tuple[dict[str, Any], ...]
    claim: str = DERIVED_UNMEASURED


class ParticipantProcedure:
    """A declared participant-level adaptive procedure.

    Declare it (:meth:`declare`), simulate the whole of it (:meth:`simulate`), retain one run
    (:meth:`simulate_history`), recompute a retained history look by look
    (:meth:`read_history`), and seal it with its retained contract
    (:meth:`prepare_session` then
    :meth:`AdaptiveInferenceSession.export_contract_with_procedure`).
    """

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self._summary: dict[str, Any] = json.loads(handle.summary_json())

    @classmethod
    def declare(cls, declaration: ParticipantDeclaration) -> ParticipantProcedure:
        """Declare the procedure.

        Raises:
            AdaptiveInferenceRefusal: ``adaptive_contract_incomplete`` with one detail per missing
                element, or ``invalid_argument`` details for an inconsistent element or a size
                beyond a declaration capacity (``declared_units_exceeded``,
                ``declared_looks_exceeded``).
        """
        if not isinstance(declaration, ParticipantDeclaration):
            raise CausalTypeError("declaration must be a ParticipantDeclaration")
        wire = _procedure_json(declaration.to_wire())
        return cls(_unwrap(_native.participant_procedure_declare(wire)))

    @property
    def identity(self) -> str:
        """Canonical identity (hex) of the declared procedure, borrowing included."""
        return str(self._handle.identity())

    @property
    def summary(self) -> dict[str, Any]:
        """The validated elements, the resolved priors and the matrix cell."""
        return dict(self._summary)

    @property
    def priors(self) -> tuple[dict[str, Any], ...]:
        """Each arm's resolved Beta prior, its sources, discounts and effective sample size."""
        return tuple(self._summary["priors"])

    def simulate(
        self,
        truth: Sequence[float],
        replications: int,
        seed: int,
        *,
        true_correlation: float | None = None,
        memory_bytes: int | None = None,
        cancel: Any = None,
    ) -> ParticipantSimulation:
        """Run the whole procedure under the arm means ``truth``.

        ``true_correlation`` is the intra-participant correlation the simulated participants
        have; absent, the declared one is used. A procedure that estimates the correlation has
        no declared value and needs it.

        The request is admitted against the library work bound (replications x participants x
        responses) and the memory budget before any replicate runs.

        Raises:
            AdaptiveInferenceRefusal: ``simulation_work_exceeded`` or
                ``simulation_memory_exceeded`` for a request over a budget,
                ``simulation_replications_exceeded`` over the replication bound, or
                ``cancelled_no_claim``.
        """
        request = _procedure_json(
            {
                "replications": int(replications),
                "seed": int(seed),
                "true_correlation": None if true_correlation is None else float(true_correlation),
            }
        )
        wire = json.loads(
            _unwrap(
                self._handle.simulate(
                    [float(p) for p in truth],
                    request,
                    None if memory_bytes is None else int(memory_bytes),
                    cancel,
                )
            )
        )
        return ParticipantSimulation(
            procedure_identity=wire["procedure_identity"],
            receipt=wire["receipt"],
            summary=wire["summary"],
            replicates=tuple(wire["replicates"]),
        )

    def simulate_history(
        self,
        truth: Sequence[float],
        seed: int,
        *,
        true_correlation: float | None = None,
        cancel: Any = None,
    ) -> ParticipantHistory:
        """Run the procedure once under ``truth`` and retain what happened.

        ``true_correlation`` is as in :meth:`simulate`.
        """
        wire = json.loads(
            _unwrap(
                self._handle.simulate_history(
                    [float(p) for p in truth],
                    int(seed),
                    None if true_correlation is None else float(true_correlation),
                    cancel,
                )
            )
        )
        return ParticipantHistory.from_wire(wire)

    def read_history(
        self, history: ParticipantHistory, *, cancel: Any = None
    ) -> ParticipantReadings:
        """Recompute every look of a retained history under the declared procedure.

        Raises:
            AdaptiveInferenceRefusal: ``adaptive_assumption_violated`` with
                ``adaptive_inference.assignment_not_replayed`` when a retained allocation
                probability is not the declared policy's, or
                ``adaptive_inference.stopping_not_replayed`` when the procedure continued past, or
                stopped without, its declared decisions.
        """
        if not isinstance(history, ParticipantHistory):
            raise CausalTypeError("history must be a ParticipantHistory")
        wire = json.loads(
            _unwrap(self._handle.read_history(_procedure_json(history.to_wire()), cancel))
        )
        return ParticipantReadings(
            procedure_identity=wire["procedure_identity"],
            priors=tuple(wire["priors"]),
            looks=tuple(wire["readings"]),
            decision=wire["decision"],
        )

    def amend_borrowing(
        self, borrowing: Sequence[ArmPrior], looks_performed: int
    ) -> ParticipantProcedure:
        """Re-declare the procedure with another borrowing plan while borrowing is still open.

        Raises:
            AdaptiveInferenceRefusal: ``adaptive_assumption_violated`` with
                ``adaptive_inference.borrowing_frozen`` when ``looks_performed`` is above zero and
                the plan differs.
        """
        wire = _procedure_json([prior._wire() for prior in borrowing])
        return ParticipantProcedure(
            _unwrap(self._handle.amend_borrowing(wire, int(looks_performed)))
        )

    def prepare_session(
        self,
        label: str,
        arm_labels: Sequence[str],
        history: ParticipantHistory,
        *,
        cancel: Any = None,
    ) -> AdaptiveInferenceSession:
        """The prepared analysis of the retained contract of a history.

        The history is read under the declared procedure first, so only a history that replays
        can be retained. Responses are retained as their success count.
        """
        if not isinstance(history, ParticipantHistory):
            raise CausalTypeError("history must be a ParticipantHistory")
        handle = _unwrap(
            self._handle.prepare_session(
                label,
                [str(a) for a in arm_labels],
                _procedure_json(history.to_wire()),
                cancel,
            )
        )
        return AdaptiveInferenceSession(handle)
