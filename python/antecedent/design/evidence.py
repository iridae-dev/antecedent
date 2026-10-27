"""Evidence plans: the studies that would close an identification gap, verified and ordered.

A failed identification owes evidence: measure the adjustment set, measure the front-door
mediators, find an instrument, run an experiment on the hedge, measure the confounder a
sensitivity analysis names. Each is a theorem route. :func:`evidence_plan` takes a
:class:`~antecedent.repair.FailureContract` and the studies you declare (cost, timing,
recruitment and sample size are yours; Antecedent never defaults them), asks every route's own
theorem checker which studies would identify the effect, and returns one :class:`EvidencePlan`::

    contract = repair.FailureContract(graph=graph, treatment="t", outcome="y",
                                      population="study", observed=["t", "y"])
    templates = repair.candidate_templates(contract)        # what to collect
    cohort = templates[0].complete(sample_size=200, recruitment="...", timing="...",
                                   unit="patient", cost=5, cost_unit="usd",
                                   sample_budget=200, feasible=True)
    plan = design.evidence_plan(contract, [cohort])
    plan.best                                  # first verified option, or None
    plan.options[0].resolves, plan.options[0].unresolved
    plan.export()                              # portable, independently replayable artifact

Options are ordered by the **identification gate** first (``verified_sufficient``, then
``not_certified`` with its reason, ``insufficient``, ``unevaluated``, ``invalid``) and only
then by value: a study never outranks a verified one because it promises more information.
With ``value=`` (a declared decision, a prior and each study's signal) the value of a single
study is its expected value of sample information, net of the declared
:class:`~antecedent.design.CostMap`; costs are compared only in one unit. A subset of several
studies has no value of its own and is listed unvalued. The prior is the one you declare and
is recorded; an assumption range from a sensitivity analysis is never a probability, so
:func:`confounder_decision` values measuring the named confounder only under a
:class:`ConfounderPrior` you declare.

A plan is hypothetical identification, not an estimate: ``inference_claim`` is ``none``, no
proposed study establishes an assumption (an instrument's validity stays an assumption), a
search that a budget stopped is ``budget_exhausted`` and never an impossibility claim, the
ordering is ``point_only`` and its calibration ``unmeasured``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from .. import _native
from .._artifact_protocol import (
    ExportedArtifact,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from .._columns import coerce_fields
from .._refusal import raise_refusal
from ..errors import CausalTypeError, CausalValueError
from ..execution import Execution, native_controls
from ..repair import (
    CandidateGeneration,
    Derivation,
    EvidenceObligation,
    FailureContract,
    RepairLimits,
    StudyCandidate,
    StudyTemplate,
    UnmatchedObligation,
    _call,
    _count,
)
from ._declarations import (
    CostMap,
    DesignDecision,
    MonteCarlo,
    SignalProvider,
    SignalSpec,
    StatePrior,
)
from .plans import StructurePrior
from .ranking import ConstraintViolation, _bound_native, _target_digest

#: What a plan can say about closing the gap; never an impossibility claim.
PlanOutcome = Literal["closing", "none_certified", "budget_exhausted"]
#: What the options are valued by.
PlanBasis = Literal["none", "evsi", "net_value"]
_OBJECTIVES = ("minimize_cost", "minimize_sample_budget")
_CONSUME_MAX_OPERATIONS = 100_000
_CONSUME_MAX_DEPTH = 4


# -- value declarations ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StudyValue:
    """The signal a declared study would produce about the decision state.

    ``provider`` and ``signal`` are the ordinary design-ranking declarations
    (:class:`~antecedent.design.GaussianMeanSignal`, :class:`~antecedent.design.BinomialSignal`
    or an attested :class:`~antecedent.design.ExternalSignal`, and its
    :class:`~antecedent.design.SignalSpec`); the study's sample size and cost are taken from
    its :class:`~antecedent.repair.StudyCandidate`.
    """

    provider: SignalProvider
    signal: SignalSpec
    reused_observations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.provider, SignalProvider):
            raise CausalTypeError("provider must be a design.SignalProvider")
        if not isinstance(self.signal, SignalSpec):
            raise CausalTypeError("signal must be a design.SignalSpec")
        coerce_fields(self, reused_observations=tuple(self.reused_observations))


@dataclass(frozen=True, slots=True)
class ConfounderPrior:
    """The belief about the confounding coordinate, declared by you.

    ``draws_per_coordinate[k]`` equally weighted prior draws sit at the ``k``-th evaluated
    grid coordinate of the sensitivity surface (ascending); zero gives it no mass.
    ``declared_by`` and ``basis`` say who asserts the prior and what it rests on. An
    assumption range says what is possible, not what is probable; this is the only place a
    probability over the coordinate enters, and it is recorded with the value.
    """

    declared_by: str
    basis: str
    draws_per_coordinate: tuple[int, ...]

    def __post_init__(self) -> None:
        for what, value in (("declared_by", self.declared_by), ("basis", self.basis)):
            if not isinstance(value, str) or not value.strip():
                raise CausalValueError(f"{what} must be a non-blank string")
        counts = tuple(_count(c, "draws_per_coordinate entry") for c in self.draws_per_coordinate)
        coerce_fields(self, draws_per_coordinate=counts)

    def _wire(self) -> dict[str, Any]:
        return {
            "declared_by": self.declared_by,
            "basis": self.basis,
            "draws_per_coordinate": list(self.draws_per_coordinate),
        }


@dataclass(frozen=True, slots=True)
class ConfounderDecision:
    """A sensitivity surface read as a decision over the confounding coordinate.

    Built by :func:`confounder_decision`: the states are the grid coordinates the surface was
    evaluated at, the actions its declared actions and the utilities its own exact utilities
    there (nothing is interpolated between grid points). ``prior`` is the
    :class:`~antecedent.design.StatePrior` of the declared :class:`ConfounderPrior`. Use it as
    ``EvidenceValue(decision=...)``.
    """

    contract_identity: str
    utility_units: str
    action_ids: tuple[str, ...]
    coordinates: tuple[float, ...]
    utilities: tuple[tuple[float, ...], ...]
    prior: StatePrior
    prior_declaration: ConfounderPrior
    prior_identity: str
    artifact_identity: str
    source_digests: tuple[str, ...]
    outcome: Mapping[str, Any]

    def _wire(self) -> dict[str, Any]:
        return {
            "contract_identity": self.contract_identity,
            "utility_unit": self.utility_units,
            "action_ids": list(self.action_ids),
            "utility_table": {
                "states": list(self.coordinates),
                "rows": [list(row) for row in self.utilities],
            },
            "prior": self.prior._wire(),
        }


def confounder_decision(
    artifact: Any, prior: ConfounderPrior, *, contract_identity: str | None = None
) -> ConfounderDecision:
    """Value measuring the named confounder under a prior you declare.

    ``artifact`` is a :class:`~antecedent.sensitivity_decision.SensitivityArtifact` with a point
    surface (no assumption range). Its declared actions and exact per-coordinate utilities
    define the decision; ``prior`` declares how likely each evaluated coordinate is. Pair the
    result with a :class:`StudyValue` for each study that would measure the confounder. A
    surface with an assumption range, or with an unevaluated grid point, refuses.
    """
    from ..sensitivity_decision import SensitivityArtifact, SensitivityRefusal

    if not isinstance(artifact, SensitivityArtifact):
        raise CausalTypeError("artifact must be a sensitivity_decision.SensitivityArtifact")
    if not isinstance(prior, ConfounderPrior):
        raise CausalTypeError("prior must be a design.ConfounderPrior")
    units = artifact.utility_units
    if units is None:
        raise CausalValueError(
            "the surface's actions read no quantity, so utility units are undetermined"
        )
    text, refusal = _native.sensitivity_confounder_value(
        artifact._native, json.dumps(prior._wire(), allow_nan=False)
    )
    raise_refusal(refusal, SensitivityRefusal)
    assert text is not None
    wire = json.loads(text)
    return ConfounderDecision(
        contract_identity=contract_identity or f"sensitivity:{wire['artifact_identity']}",
        utility_units=str(units),
        action_ids=tuple(wire["action_ids"]),
        coordinates=tuple(float(x) for x in wire["coordinates"]),
        utilities=tuple(tuple(float(x) for x in row) for row in wire["utilities"]),
        prior=StatePrior.draws([float(x) for x in wire["prior_draws"]]),
        prior_declaration=prior,
        prior_identity=str(wire["prior_identity"]),
        artifact_identity=str(wire["artifact_identity"]),
        source_digests=tuple(wire["source_digests"]),
        outcome=wire["decision"],
    )


@dataclass(frozen=True, slots=True)
class EvidenceValue:
    """A declared decision, prior and signals that value the studies of a plan.

    ``studies`` maps a :class:`~antecedent.repair.StudyCandidate` label to the signal it would
    produce; a study with no entry is listed unvalued. ``decision`` is a
    :class:`~antecedent.design.DesignDecision` or a :class:`ConfounderDecision`. With a
    ``cost_map`` the value is net of the declared cost (its ``cost_unit`` must be the studies'
    cost unit); without one, value and cost are reported separately.
    """

    decision: DesignDecision | ConfounderDecision
    studies: Mapping[str, StudyValue]
    cost_map: CostMap | None = None
    require_net_value: bool = False
    prior_observations: Sequence[str] = ()
    source_digests: Sequence[str] = ()
    rng_seed: int = 0
    monte_carlo: MonteCarlo | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, (DesignDecision, ConfounderDecision)):
            raise CausalTypeError("decision must be a design.DesignDecision or ConfounderDecision")
        if any(not isinstance(v, StudyValue) for v in self.studies.values()):
            raise CausalTypeError("studies must map labels to StudyValue")
        _count(self.rng_seed, "rng_seed")

    def _wire(self) -> dict[str, Any]:
        digests = list(self.source_digests)
        if isinstance(self.decision, ConfounderDecision):
            digests.extend(self.decision.source_digests)
        return {
            "decision": self.decision._wire(),
            "studies": [
                {
                    "label": label,
                    "signal": value.signal._wire(),
                    "provider": value.provider._wire(),
                    "reused_observation_ids": list(value.reused_observations),
                }
                for label, value in self.studies.items()
            ],
            "cost_map": None if self.cost_map is None else self.cost_map._wire(),
            "require_net_value": bool(self.require_net_value),
            "prior_observation_ids": list(self.prior_observations),
            "source_digests": sorted(set(digests)),
            "rng_seed": int(self.rng_seed),
            "monte_carlo": None if self.monte_carlo is None else self.monte_carlo._wire(),
        }


# -- the plan -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OptionValue:
    """The value of a single study under the declared decision, prior and signal."""

    evsi: float
    evpi: float
    mc_stderr: float
    integration: str
    study_cost_utility: float | None
    net_value: float | None
    provider_trust: str


@dataclass(frozen=True, slots=True)
class PlanOption:
    """One option: a study or bounded subset of studies on one theorem route.

    ``gate`` is the route's own verdict (``verified_sufficient``, ``not_certified`` with
    ``reasons``, ``insufficient``, ``unevaluated`` or ``invalid``). ``resolves`` are the
    obligation ids the studies' evidence passes the necessary screen for and ``unresolved``
    every other obligation of the route; ``assumptions_remaining`` are those no study can
    establish. ``cost`` is in ``cost_unit``. ``value`` is set for a single valued study;
    ``value_note`` says why an option is unvalued. ``rank_uncertain`` marks a value within
    error of the next option of its gate.
    """

    rank: int
    route: str
    members: tuple[str, ...]
    labels: tuple[str, ...]
    gate: str
    reasons: tuple[str, ...]
    derivation: Derivation | None
    resolves: tuple[str, ...]
    unresolved: tuple[str, ...]
    assumptions_remaining: tuple[str, ...]
    cost: int
    cost_unit: str
    sample_budget: int
    value: OptionValue | None
    value_note: str | None
    rank_uncertain: bool

    def __repr__(self) -> str:
        return f"<PlanOption #{self.rank} {self.route} {self.gate} {list(self.labels)!r}>"

    @classmethod
    def _from_native(cls, value: Mapping[str, Any]) -> PlanOption:
        derivation = value["derivation"]
        v = value["value"]
        return cls(
            rank=int(value["rank"]),
            route=str(value["route"]),
            members=tuple(value["members"]),
            labels=tuple(value["labels"]),
            gate=str(value["gate"]),
            reasons=tuple(value["reasons"]),
            derivation=None
            if derivation is None
            else Derivation(
                derivation["checker"], tuple(derivation["steps"]), derivation["verified"]
            ),
            resolves=tuple(value["resolves"]),
            unresolved=tuple(value["unresolved"]),
            assumptions_remaining=tuple(value["assumptions_remaining"]),
            cost=int(value["cost"]["units"]),
            cost_unit=str(value["cost"]["unit"]),
            sample_budget=int(value["cost"]["sample_budget"]),
            value=None
            if v is None
            else OptionValue(
                evsi=float(v["evsi"]),
                evpi=float(v["evpi"]),
                mc_stderr=float(v["mc_stderr"]),
                integration=str(v["integration"]),
                study_cost_utility=v["study_cost_utility"],
                net_value=v["net_value"],
                provider_trust=str(v["provider_trust"]),
            ),
            value_note=value["value_note"],
            rank_uncertain=bool(value["rank_uncertain"]),
        )


@dataclass(frozen=True, slots=True)
class PlanRoute:
    """One theorem route the plan covers and what it owes."""

    family: str
    contract: str
    obligations: tuple[EvidenceObligation, ...]
    unevaluated_total: int
    stop: str | None
    beyond_declared_depth: bool


@dataclass(frozen=True, slots=True)
class UnverifiedRoute:
    """A route no theorem checker decides (the repeated-measurement alternative)."""

    route: str
    obligations: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class PlanUnlocks:
    """The ranker's unlock list, derived from the repair reports instead of declared by hand.

    ``variables`` (and ``ids``, their indices in the contract's variable names) are the
    variables a study verified sufficient on its own measures outside the observed law or
    sets. Only such a study unlocks identification: a study that merely names a variable, or
    only adds rows, is in ``unverified`` and is never credited.
    """

    variables: tuple[str, ...]
    ids: tuple[int, ...]
    verified: tuple[str, ...]
    unverified: tuple[str, ...]


class EvidencePlan:
    """The ordered options that close an identification gap.

    :attr:`outcome` is ``closing`` (a verified option exists), ``none_certified`` (nothing
    declared was certified; never an impossibility claim) or ``budget_exhausted`` (a budget
    stopped a search with nothing certified; never a verdict). :attr:`options` are ordered by
    the identification gate and then by value net of the declared cost map;
    :attr:`ranking_guarantee` is ``point_only`` and :attr:`calibration` ``unmeasured``.
    """

    __slots__ = (
        "_report",
        "_stage",
        "calibration",
        "generation",
        "inference_claim",
        "options",
        "ranking_guarantee",
        "routes",
        "unlocks",
        "unverified",
    )

    def __init__(self, stage: Any) -> None:
        report = stage.report()
        self._stage = stage
        self._report = report
        self.options: tuple[PlanOption, ...] = tuple(
            PlanOption._from_native(o) for o in report["options"]
        )
        self.routes: tuple[PlanRoute, ...] = tuple(
            PlanRoute(
                family=r["family"],
                contract=r["contract"],
                obligations=tuple(EvidenceObligation._from_native(o) for o in r["obligations"]),
                unevaluated_total=int(r["budget"]["unevaluated_total"]),
                stop=r["budget"]["stop"],
                beyond_declared_depth=bool(r["budget"]["beyond_declared_depth"]),
            )
            for r in report["routes"]
        )
        self.unverified: tuple[UnverifiedRoute, ...] = tuple(
            UnverifiedRoute(u["route"], tuple(u["obligations"]), u["reason"])
            for u in report["unverified"]
        )
        generation = report.get("generation")
        self.generation: CandidateGeneration | None = (
            None
            if generation is None
            else CandidateGeneration(
                templates=tuple(StudyTemplate._from_native(t) for t in generation["templates"]),
                unmatched=tuple(
                    UnmatchedObligation(u["id"], u["kind"], u["route"], u["reason"])
                    for u in generation["unmatched"]
                ),
            )
        )
        unlocks = report.get("unlocks")
        self.unlocks: PlanUnlocks | None = (
            None
            if unlocks is None
            else PlanUnlocks(
                variables=tuple(unlocks["variables"]),
                ids=tuple(int(i) for i in unlocks["ids"]),
                verified=tuple(unlocks["verified"]),
                unverified=tuple(unlocks["unverified"]),
            )
        )
        self.inference_claim: str = str(report["inference_claim"])
        self.ranking_guarantee: str = str(report["ranking_guarantee"])
        self.calibration: str = str(report["calibration"])

    @property
    def outcome(self) -> PlanOutcome:
        """``closing``, ``none_certified`` or ``budget_exhausted``."""
        return cast(PlanOutcome, str(self._report["outcome"]))

    @property
    def reason_code(self) -> str | None:
        """The registered reason code when nothing was verified, else ``None``."""
        return self._report["reason"]

    @property
    def detail(self) -> str | None:
        """The stable detail when nothing was verified, else ``None``."""
        return self._report["detail"]

    @property
    def cost_unit(self) -> str:
        """The one unit every option's cost is compared in."""
        return str(self._report["cost_unit"])

    @property
    def basis(self) -> PlanBasis:
        """``none`` (gate, then cost), ``evsi`` or ``net_value``."""
        return cast(PlanBasis, str(self._report["basis"]))

    @property
    def cost_map(self) -> Mapping[str, Any] | None:
        """The declared cost-to-utility mapping, when one was used."""
        return self._report["cost_map"]

    @property
    def utility_unit(self) -> str | None:
        """Utility unit of the values, when valued."""
        return self._report["utility_unit"]

    @property
    def identity(self) -> str:
        """The plan's identity."""
        return str(self._report["identity"])

    @property
    def value_identity(self) -> str | None:
        """Digest of the value evaluation the values came from."""
        return self._report["value_identity"]

    @property
    def closing(self) -> tuple[PlanOption, ...]:
        """The verified-sufficient options, best first."""
        return tuple(o for o in self.options if o.gate == "verified_sufficient")

    @property
    def best(self) -> PlanOption | None:
        """The first verified option, or ``None``."""
        return self.closing[0] if self.closing else None

    def variable_unlocks(self, query_id: int = 0) -> dict[int, tuple[int, ...]]:
        """``{query_id: ids}`` for ``rank_designs(variable_unlocks=...)``, from verified studies.

        Raises :class:`~antecedent.errors.CausalValueError` for a replayed plan, which holds no
        declared studies to derive the list from.
        """
        if self.unlocks is None:
            raise CausalValueError("a consumed plan holds no declared studies to derive unlocks")
        return {int(query_id): self.unlocks.ids} if self.unlocks.ids else {}

    @artifact_export
    def export(self) -> ExportedArtifact:
        """A portable ``evidence_plan_v1`` artifact; see :func:`consume`."""
        return artifact_from_receipt(self, bytes(_call(self._stage.export)))

    def explain(self) -> str:
        """What the plan found, what it did not, and why it is not an estimate."""
        if self.outcome == "closing" and self.best is not None:
            best = self.best
            text = (
                f"{len(self.closing)} option(s) would make the effect identify if the studies "
                f"deliver exactly their declared evidence; first: {', '.join(best.labels)} "
                f"on the {best.route} route (cost {best.cost} {best.cost_unit}"
            )
            if best.value is not None:
                net = best.value.net_value
                text += f", EVSI {best.value.evsi:.6g}" + (
                    "" if net is None else f", net value {net:.6g}"
                )
            text += ")."
        elif self.outcome == "budget_exhausted":
            text = (
                "A declared search budget stopped before anything was certified; the subsets it "
                "left unexamined are unknown, not impossible."
            )
        else:
            text = (
                f"None of {len(self.options)} option(s) was certified; that does not show that no "
                "study can close the gap."
            )
        if self.reason_code is not None:
            text += f" Reason: {self.reason_code} ({self.detail})."
        remaining = sorted({a for o in self.options for a in o.assumptions_remaining})
        if remaining:
            text += (
                f" {len(remaining)} assumption(s) remain that no study establishes; a study "
                "supplies data, never the assumption."
            )
        if self.unverified:
            text += (
                " Not scored: "
                + "; ".join(f"{u.route} ({u.reason})" for u in self.unverified)
                + "."
            )
        return (
            text + " A plan is hypothetical identification, not an estimate: inference claim is "
            f"{self.inference_claim!r}, the ordering is {self.ranking_guarantee} and its "
            f"calibration {self.calibration}."
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form."""
        return json.loads(json.dumps(self._report))

    def __repr__(self) -> str:
        return (
            f"<EvidencePlan outcome={self.outcome} options={len(self.options)} "
            f"closing={len(self.closing)}>"
        )


def _run(execution: Execution, operation: Any, *args: Any, **kwargs: Any) -> Any:
    """Run one native entry under the execution's policy with repair refusals typed."""
    return _call(execution._call, operation, *args, **kwargs)


def evidence_plan(
    contract: FailureContract,
    candidates: Sequence[StudyCandidate],
    objective: Literal["minimize_cost", "minimize_sample_budget"] = "minimize_cost",
    limits: RepairLimits | None = None,
    *,
    value: EvidenceValue | None = None,
    execution: Execution | None = None,
) -> EvidencePlan:
    """Verify declared studies on every theorem route and order the options that close the gap.

    Each route's own theorem checker decides whether a candidate (or a bounded subset) would
    identify the effect; a variable-name match never does, and a proposed study never
    establishes an assumption. Options are ordered by the identification gate and then, with
    ``value=``, by expected value of sample information net of the declared cost map; all
    studies must declare one cost unit (``design_cost_units_mismatch`` otherwise). A search a
    budget stopped is ``budget_exhausted``, never an impossibility claim.

    Refusals carry their reason code: ``invalid_argument`` for a request beyond the declared
    bounds, ``design_cost_units_mismatch`` for studies in different cost units,
    ``transport_budget_cancel`` when the budget stops before a search
    (:class:`~antecedent.repair.RepairBudgetRefusal`) and the value evaluation's own
    ``design_ranking`` refusals.
    """
    if not isinstance(contract, FailureContract):
        raise CausalTypeError("evidence_plan requires a repair.FailureContract")
    candidates = tuple(candidates)
    if any(not isinstance(c, StudyCandidate) for c in candidates):
        raise CausalTypeError("candidates must be StudyCandidate values")
    if objective not in _OBJECTIVES:
        raise CausalValueError(f"objective must be one of {', '.join(_OBJECTIVES)}")
    if value is not None and not isinstance(value, EvidenceValue):
        raise CausalTypeError("value must be a design.EvidenceValue")
    limits = RepairLimits() if limits is None else limits
    try:
        value_text = None if value is None else json.dumps(value._wire(), allow_nan=False)
    except ValueError as error:
        raise CausalValueError("a declaration contains a non-finite number") from error
    execution = Execution() if execution is None else execution
    stage = _run(
        execution,
        contract._stage.plan,
        json.dumps([c._wire() for c in candidates]),
        objective,
        max_operations=_count(limits.max_operations, "max_operations"),
        max_depth=_count(limits.max_depth, "max_depth"),
        memory_bytes=execution.limits.memory_bytes,
        value_json=value_text,
        cancel=execution.cancel,
    )
    return EvidencePlan(stage)


def consume_plan(
    artifact: bytes,
    *,
    max_operations: int = _CONSUME_MAX_OPERATIONS,
    max_depth: int = _CONSUME_MAX_DEPTH,
    execution: Execution | None = None,
) -> EvidencePlan:
    """Independently replay an exported evidence plan and return it.

    The consumer refuses stored limits above its own before any work, replays every embedded
    repair report with the families' own theorem checkers, replays the embedded value
    ranking by independent recomputation, rebuilds the plan and accepts only identical
    options, outcome and identity. A resealed edit of a gate, a cost or a value is
    :class:`~antecedent.repair.RepairArtifactRefusal` (``evidence_plan_artifact.replay_mismatch``,
    naming the section); a cancelled replay is :class:`~antecedent.repair.RepairBudgetRefusal`,
    never a verdict. Replay does not protect against a producer that states another contract or
    candidate universe and re-seals honestly, nor does it check that arriving studies will
    deliver their declared evidence.
    """
    artifact = container_bytes(artifact)
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    execution = Execution() if execution is None else execution
    stage = _run(
        execution,
        _native.consume_evidence_plan,
        artifact,
        max_operations=_count(max_operations, "max_operations"),
        max_depth=_count(max_depth, "max_depth"),
        **native_controls(execution, memory="memory_bytes", cancel="cancel"),
    )
    return EvidencePlan(stage)


@dataclass(frozen=True, slots=True)
class RankedStudy:
    """One verified study ranked by the gain in the probability the query is identified."""

    label: str
    #: Position of the study in the sequence supplied.
    index: int
    #: Zero-based rank, best first.
    rank: int
    #: Gain in the identified mass of the structure prior; not a calibrated probability.
    score: float
    stderr: float
    rank_uncertain: bool
    evaluation: str


@dataclass(frozen=True, slots=True)
class UnverifiedStudy:
    """A declared study the repair search did not verify sufficient on its own.

    It is never scored: a study that merely names a variable, or only adds rows, is not credited
    with an unlock.
    """

    index: int
    label: str
    semantic_id: str
    #: The checker's classification, e.g. ``insufficient`` or ``not_certified``.
    classification: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StudyRanking:
    """Declared studies ranked through a prepared study's contract.

    ``ranked`` holds only the studies the repair search verified sufficient, best first;
    ``unverified`` every other declared study with the checker's reasons, and ``unlock`` the
    variables the verified studies measure (outside the base law) or set, which the ranking used
    as the unlock list. ``violations`` are the studies the ranking refused to score.
    """

    ranked: tuple[RankedStudy, ...]
    verified: tuple[int, ...]
    unverified: tuple[UnverifiedStudy, ...]
    unlock: tuple[str, ...]
    violations: tuple[ConstraintViolation, ...]
    mc_samples: int
    early_stopped: bool
    calibration: Literal["unmeasured"] = "unmeasured"

    @property
    def best(self) -> RankedStudy | None:
        """The highest-ranked verified study, or ``None`` when no study was verified."""
        return self.ranked[0] if self.ranked else None


def rank_studies(
    study: Any,
    contract: FailureContract,
    candidates: Sequence[StudyCandidate],
    prior: StructurePrior,
    *,
    route: str | None = None,
    query_id: int = 0,
    objective: Literal["minimize_cost", "minimize_sample_budget"] = "minimize_cost",
    limits: RepairLimits | None = None,
    max_cost: float | None = None,
    max_sample_budget: int | None = None,
    monte_carlo: MonteCarlo | None = None,
    rng_seed: int = 0,
    decision_target: str | None = None,
    execution: Execution | None = None,
) -> StudyRanking:
    """Rank declared studies by identification gain through a prepared study's contract.

    The repair search of ``contract`` decides which studies are sufficient on their own; only
    those are scored, with their measured variables outside the observed base law and their
    set variables as the unlock list for ``query_id``. Every other study is reported in
    ``unverified`` and as an ``unlicensed_candidate`` violation. ``study`` is the immutable
    prepared study the ranking is bound to: an unlicensed support cell or one without an
    identification product raises :class:`~antecedent.errors.CausalUnsupportedError`. ``route``
    names the failure route that decides sufficiency (``backdoor``, ``frontdoor``,
    ``instrument``, ``hedge``) and may be omitted when the failure has exactly one.
    ``prior`` is the :class:`~antecedent.design.StructurePrior` ranked against; ordering is
    ``point_only`` and its calibration ``unmeasured``.
    """
    if not isinstance(contract, FailureContract):
        raise CausalTypeError("rank_studies requires a repair.FailureContract")
    if not isinstance(prior, StructurePrior):
        raise CausalTypeError("prior must be a design.StructurePrior")
    candidates = tuple(candidates)
    if not candidates:
        raise CausalValueError("rank_studies needs at least one candidate")
    if any(not isinstance(c, StudyCandidate) for c in candidates):
        raise CausalTypeError("candidates must be StudyCandidate values")
    if objective not in _OBJECTIVES:
        raise CausalValueError(f"objective must be one of {', '.join(_OBJECTIVES)}")
    limits = RepairLimits() if limits is None else limits
    if monte_carlo is not None and not isinstance(monte_carlo, MonteCarlo):
        raise CausalTypeError("monte_carlo must be design.MonteCarlo")
    options: dict[str, Any] = {}
    if monte_carlo is not None:
        options.update(
            min_batches=monte_carlo.min_batches,
            max_batches=monte_carlo.max_batches,
            batch_size=monte_carlo.batch_size,
            rank_uncertainty_threshold=monte_carlo.rank_uncertainty_threshold,
        )
    execution = Execution() if execution is None else execution
    native_ranking, record = _run(
        execution,
        _native.rank_studies_bound,
        contract._stage,
        _bound_native(study),
        json.dumps([c._wire() for c in candidates]),
        objective,
        _count(limits.max_operations, "max_operations"),
        _count(limits.max_depth, "max_depth"),
        list(prior.weights),
        [int(v) for v in prior.identified],
        list(prior.keys),
        route=route,
        query_id=_count(query_id, "query_id"),
        identified_under_intervention=(
            [int(v) for v in prior.identified_under_intervention]
            if prior.identified_under_intervention is not None
            else None
        ),
        graph_features=list(prior.features) if prior.features is not None else None,
        max_cost=max_cost,
        max_sample_budget=max_sample_budget,
        seed=_count(rng_seed, "rng_seed"),
        threads=execution.threads,
        memory_bytes=execution.limits.memory_bytes,
        decision_target=_target_digest(decision_target),
        cancel=execution.cancel,
        **options,
    )
    facts = json.loads(record)
    labels = [c.label for c in candidates]
    return StudyRanking(
        ranked=tuple(
            RankedStudy(
                label=labels[r.candidate_index],
                index=int(r.candidate_index),
                rank=int(r.rank),
                score=float(r.score),
                stderr=float(r.stderr),
                rank_uncertain=bool(r.rank_uncertain),
                evaluation=r.evaluation,
            )
            for r in sorted(native_ranking.ranked, key=lambda row: int(row.rank))
        ),
        verified=tuple(int(i) for i in facts["verified"]),
        unverified=tuple(
            UnverifiedStudy(
                index=int(u["index"]),
                label=labels[u["index"]],
                semantic_id=u["semantic_id"],
                classification=u["classification"],
                reasons=tuple(u["reasons"]),
            )
            for u in sorted(facts["unverified"], key=lambda row: int(row["index"]))
        ),
        unlock=tuple(facts["unlock"]),
        violations=tuple(
            ConstraintViolation(
                labels[v.candidate_index], int(v.candidate_index), v.constraint, v.detail
            )
            for v in native_ranking.violations
        ),
        mc_samples=int(native_ranking.mc_samples),
        early_stopped=bool(native_ranking.early_stopped),
    )


__all__ = [
    "ConfounderDecision",
    "ConfounderPrior",
    "EvidencePlan",
    "EvidenceValue",
    "OptionValue",
    "PlanBasis",
    "PlanOption",
    "PlanOutcome",
    "PlanRoute",
    "PlanUnlocks",
    "RankedStudy",
    "StudyRanking",
    "StudyValue",
    "UnverifiedRoute",
    "UnverifiedStudy",
    "confounder_decision",
    "consume_plan",
    "evidence_plan",
    "rank_studies",
]
