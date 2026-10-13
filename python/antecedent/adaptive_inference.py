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
* **Point-only.** No interval, credible interval or coverage statement:
  :meth:`AdaptiveInferenceSession.interval` raises ``cell_not_licensed``.
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
from typing import Any

from . import _native
from ._refusal import StructuredRefusal, raise_refusal
from .errors import CausalTypeError, CausalValueError

__all__ = [
    "AdaptiveDesign",
    "AdaptiveInferenceRefusal",
    "AdaptiveInferenceSession",
    "AdaptiveResult",
    "AdaptiveWeighting",
    "Availability",
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
        return cls(
            "efficacy", "posterior_probability_best", ended_by_rule, parameter=threshold
        )

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


@dataclass(frozen=True)
class AdaptiveDesign:
    """The retained analysis contract of one adaptive experiment.

    ``assignment``, ``outcome``, ``monitoring`` and ``stopping`` name the procedure cell (see
    :func:`procedure_matrix`). Each of ``policy_history``, ``assignment_probabilities``,
    ``exposure``, ``availability``, ``interim_analyses`` and ``stopping_rule`` may be ``None``;
    preparing the design then refuses with that element's own detail.
    """

    label: str
    assignment: str
    outcome: str
    monitoring: str
    stopping: str
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
            "assigned_arm": [int(a) for a in self.assigned_arm],
            "policy_history": None
            if self.policy_history is None
            else [
                {"first_unit": int(e.first_unit), "identity": e.identity}
                for e in self.policy_history
            ],
            "assignment_probability_bits": None
            if self.assignment_probabilities is None
            else [[_bits(p) for p in row] for row in self.assignment_probabilities],
            "exposure": None if self.exposure is None else [int(a) for a in self.exposure],
            "availability": None
            if self.availability is None
            else {
                "available_at_look": [
                    None if look is None else int(look)
                    for look in self.availability.available_at_look
                ],
                "delay": self.availability.delay,
            },
            "outcome_bits": [[_bits(y) for y in row] for row in self.outcomes],
            "interim_analyses": None
            if self.interim_analyses is None
            else [
                {
                    "units_enrolled": int(a.units_enrolled),
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
            "look": int(self.look),
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
            "look": int(self.look),
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
            "look": int(self.look),
            "plus_arm": int(self.plus_arm),
            "minus_arm": None if self.minus_arm is None else int(self.minus_arm),
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
            "look": int(self.look),
            "alpha": float(self.alpha),
            "tuning_samples": float(self.tuning_samples),
        }
        if self.contrast is not None:
            wire["plus_arm"], wire["minus_arm"] = (int(a) for a in self.contrast)
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
            "look": int(self.look),
            "control_arm": int(self.control_arm),
            "treatment_arm": int(self.treatment_arm),
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


@dataclass(frozen=True)
class AdaptiveResult:
    """One result: the values in the route's fixed order, labelled with paradigm and procedure.

    ``values`` order: Bayesian posterior ``alpha, beta, mean, variance, probability_best,
    expected_loss`` per arm; posterior stopping ``best_arm, probability_best, expected_loss,
    stop``; adaptive weighting ``estimate, variance, units, effective_units, max_weight_share``;
    confidence sequence ``count, mean, lower, upper`` per arm then the contrast ``lower, upper``;
    group sequential ``stopped_at`` (``-1`` for none) then ``critical, statistic, spent`` per look.
    """

    paradigm: str
    procedure: str
    method: str
    values: tuple[float, ...]
    closed_guarantee: str
    contract_identity: str
    look: int
    stopping_rule: dict[str, Any] | None
    #: Always ``point_only``: no interval or coverage statement is licensed.
    claim: str = "point_only"


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


@dataclass(frozen=True)
class InterimEvidence:
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
        values=tuple(float("nan") if v is None else float(v) for v in wire["values"]),
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

    def analyse(self, request: Request, *, cancel: Any = None) -> AdaptiveResult:
        """Run the paradigm a request names.

        Raises:
            AdaptiveInferenceRefusal: ``cell_not_licensed`` for a paradigm the cell has no route
                for, ``adaptive_assumption_violated`` for noncompliance, outcome-dependent delay
                or a probability below the floor, ``cancelled_no_claim`` on cancellation.
        """
        wire = _unwrap(self._handle.analyse(_request_json(request), cancel=cancel))
        return _result(json.loads(wire))

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

    def export_contract(self, paradigm: str, artifact_id: str = "adaptive-monitoring") -> bytes:
        """Write ``adaptive_monitoring_contract_v1`` for a paradigm."""
        return bytes(_unwrap(self._handle.export_contract(paradigm, artifact_id)))

    def export_result(
        self, request: Request, artifact_id: str = "adaptive-result", *, cancel: Any = None
    ) -> bytes:
        """Write ``adaptive_inference_result_v1`` for a request."""
        return bytes(
            _unwrap(self._handle.export_result(_request_json(request), artifact_id, cancel=cancel))
        )


# --------------------------------------------------------------------------- consumers


@dataclass(frozen=True)
class ConsumedMonitoringContract:
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


@dataclass(frozen=True)
class ConsumedAdaptiveResult:
    """A replayed result: procedure, rule, paradigm and the values recomputed by the consumer."""

    identity: str
    procedure: str
    paradigm: str
    stopping_rule: dict[str, Any] | None
    method: str
    values: tuple[float, ...]
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
    handle = _unwrap(_native.adaptive_monitoring_contract_consume(bytes(artifact), expected_identity))
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
        values=tuple(float("nan") if v is None else float(v) for v in wire["values"]),
        closed_guarantee=wire["closed_guarantee"],
    )
