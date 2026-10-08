"""Evidence obligations, durable study candidates and identification repair.

A causal contract that fails to identify owes concrete evidence. This module
exposes that debt as machine-readable :class:`EvidenceObligation` values, lets
you describe a study that might supply it (:class:`StudyCandidate`; Antecedent
does not conduct it) and asks :func:`repair` which candidates, singly or in
bounded subsets, would make the contract identify::

    contract = repair.BackdoorContract(
        graph=Dag.from_edges(names, edges), treatment="t", outcome="y",
        population="clinic", observed=["t", "y"],
    )
    repair.obligations(contract)            # one joint law over t, y and the adjustment set
    result = repair.repair(contract, [cohort, registry])
    result.best                              # cheapest verified-sufficient subset, or None
    result.export()                          # portable, independently replayable artifact

Three families are supported, each deferring to its own theorem checker rather
than matching variable names: :class:`TransportContract` (the catalog-aware
classical transport identifier, re-verified after the hypothetical evidence is
applied) and :class:`BackdoorContract` (the back-door identifier on the one
joint law a study would deliver; covariates measured in separate studies never
combine), and :class:`ZTransportContract` (a checked z-transport failure snapshot
whose original expression leaves are re-verified and bound). Every candidate lands in exactly one class: ``verified_sufficient``,
``insufficient``, ``not_certified``, ``invalid`` or ``unevaluated``. A search
that a budget stopped keeps its unevaluated subsets and reports ``exhausted``:
exhaustion is never an impossibility claim, and ``none_certified`` only says
nothing in the declared candidates was certified. A proposed study never
establishes an assumption: an ``establish_assumption`` obligation can only be
met outside this module.

Refusals are :class:`~antecedent.errors.CausalUnsupportedError` subclasses
carrying the registered ``reason_code`` and a stable ``detail``
(``identification_repair.*``, ``evidence_obligations.*``, ``study_candidate.*``
or ``repair_artifact.*``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import _native
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .graph import Admg, Dag
from .joint_distribution import ScientificQuantity
from .transport._impl import EvidenceCatalog

_KINDS = ("experiment", "observation", "sample_increase")
_OBJECTIVES = ("minimize_cost", "minimize_sample_budget")
_REFUSAL = re.compile(
    r"^(?:refused: )?reason=(?P<code>[a-z_]+): (?P<detail>[a-z_]+\.[a-z_]+): (?P<message>.*)$",
    re.DOTALL,
)
_CONSUME_MAX_OPERATIONS = 100_000
_CONSUME_MAX_DEPTH = 4


class RepairRefusal(CausalUnsupportedError):
    """A refused repair request, obligation or candidate.

    ``reason_code`` is the registered runtime-refusal code and :attr:`detail` the
    stable ``<family>.<slot>`` detail (for example
    ``identification_repair.not_a_failure`` or
    ``identification_repair.bounds_exceeded``).
    """

    def __init__(
        self,
        message: str = "",
        *,
        reason_code: str | None = None,
        detail: str | None = None,
        remedy: str | None = None,
    ) -> None:
        text = f"{detail}: {message}" if detail else message
        super().__init__(text, reason_code=reason_code, remedy=remedy)
        #: Stable ``<family>.<slot>`` detail, when the refusal names one.
        self.detail: str | None = detail


class RepairBudgetRefusal(RepairRefusal):
    """The repair or its replay was stopped before it could decide.

    The artifact or request is neither accepted nor refuted: a stop is a resource
    outcome, never a finding about the contract.
    """


class RepairArtifactRefusal(RepairRefusal):
    """An exported repair report that cannot be read or does not replay.

    ``detail`` is one of ``repair_artifact.invalid_artifact``,
    ``repair_artifact.unsupported_version``, ``repair_artifact.bounds_exceeded``
    or ``repair_artifact.replay_mismatch`` (the message names the stored section
    the replay did not reproduce: contract, candidates, obligations,
    classification or receipt).
    """


class StudyCandidateRefusal(RepairRefusal):
    """A study candidate that declares no unit, timing, recruitment or cost semantics."""


def _call(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a native call, re-raising its refusals as typed repair refusals."""
    try:
        return function(*args, **kwargs)
    except CausalUnsupportedError as error:
        parsed = _REFUSAL.match(str(error))
        if parsed is None or isinstance(error, RepairRefusal):
            raise
        detail = parsed["detail"]
        if detail.endswith(".budget"):
            kind: type[RepairRefusal] = RepairBudgetRefusal
        elif detail.split(".", 1)[0] == "repair_artifact":
            kind = RepairArtifactRefusal
        else:
            kind = RepairRefusal
        raise kind(parsed["message"], reason_code=parsed["code"], detail=detail) from error


def _text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise CausalTypeError(f"{name} must be a string")
    if not value.strip():
        raise CausalValueError(f"{name} must be non-empty")
    return value


def _names(name: str, values: Sequence[str], *, allow_empty: bool = True) -> tuple[str, ...]:
    if isinstance(values, str):
        raise CausalTypeError(f"{name} must be a sequence of names, not a single string")
    out = tuple(values)
    if any(not isinstance(v, str) or not v.strip() for v in out):
        raise CausalTypeError(f"{name} must be non-empty strings")
    if len(set(out)) != len(out):
        raise CausalValueError(f"{name} must be distinct")
    if not allow_empty and not out:
        raise CausalValueError(f"{name} must not be empty")
    return out


def _count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CausalTypeError(f"{name} must be an integer")
    return value


def _semantics(name: str, text: object) -> str:
    if not isinstance(text, str) or not text.strip():
        raise StudyCandidateRefusal(
            f"a candidate needs {name}",
            reason_code="design_signal_invalid",
            detail="study_candidate.wrong_contract",
        )
    return text


@dataclass(frozen=True, slots=True)
class EvidenceObligation:
    """One machine-readable evidence request of a failed contract.

    ``kind`` is one of ``measure``, ``intervene``, ``observe_population``,
    ``observe_environment``, ``increase_sample``, ``provide_joint_law``,
    ``provide_conditional_law``, ``establish_support`` or (sparingly)
    ``establish_assumption``. ``id`` is a stable digest of the canonical content.
    ``interventions``, ``conditioned_on`` and ``joint`` describe the requested
    regime; ``required_slots`` are the contract slots the evidence fills.
    ``family``, ``source`` and ``proof_step`` are the provenance: the theorem
    route, the contract it was read from and the source proof step (spelled in
    the checker's own identifiers). An obligation is a request, never a verdict;
    ``satisfiable_by_study`` is ``False`` for an assumption.
    """

    id: str
    kind: str
    scope: str
    variables: tuple[str, ...]
    population: str | None
    interventions: tuple[str, ...]
    conditioned_on: tuple[str, ...]
    joint: bool
    reason: str
    required_slots: tuple[str, ...]
    min_additional_samples: int | None
    family: str
    source: str
    proof_step: str | None
    satisfiable_by_study: bool
    quantities: Mapping[str, ScientificQuantity] = field(default_factory=dict)

    @classmethod
    def _from_native(cls, value: Mapping[str, Any]) -> EvidenceObligation:
        regime = value["regime"]
        provenance = value["provenance"]
        return cls(
            id=value["id"],
            kind=value["kind"],
            scope=value["scope"],
            variables=tuple(value["variables"]),
            population=value["population"],
            interventions=tuple(regime["interventions"]),
            conditioned_on=tuple(regime["conditioned_on"]),
            joint=regime["joint"],
            reason=value["reason"],
            required_slots=tuple(value["required_slots"]),
            min_additional_samples=value["min_additional_samples"],
            family=provenance["family"],
            source=provenance["source"],
            proof_step=provenance["proof_step"],
            satisfiable_by_study=value["satisfiable_by_study"],
            quantities={
                name: ScientificQuantity._from_wire(wire)
                for name, wire in value.get("quantities", {}).items()
            },
        )


@dataclass(frozen=True, slots=True)
class ExpectedEvidence:
    """One law a study declares it would deliver.

    ``interventions`` is the study's own hard-intervention set (empty for an
    observation); ``levels`` the concrete levels, empty for the unrestricted
    domain; ``conditioned_on`` the coordinates the law is already conditioned
    on. ``joint=False`` declares separate marginals, which can never claim a
    joint law.
    """

    population: str
    measured: Sequence[str]
    interventions: Sequence[str] = ()
    levels: Mapping[str, float] = field(default_factory=dict)
    conditioned_on: Sequence[str] = ()
    joint: bool = True

    def __post_init__(self) -> None:
        _text("population", self.population)
        object.__setattr__(self, "measured", _names("measured", self.measured, allow_empty=False))
        object.__setattr__(self, "interventions", _names("interventions", self.interventions))
        object.__setattr__(self, "conditioned_on", _names("conditioned_on", self.conditioned_on))
        object.__setattr__(self, "levels", {str(k): float(v) for k, v in self.levels.items()})

    def _wire(self) -> dict[str, Any]:
        return {
            "population": self.population,
            "interventions": list(self.interventions),
            "levels": dict(self.levels),
            "conditioned_on": list(self.conditioned_on),
            "measured": list(self.measured),
            "joint": bool(self.joint),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class StudyCandidate:
    """A durable description of a study that could produce evidence.

    ``kind`` is ``experiment`` (hard ``interventions`` plus a measured margin),
    ``observation`` (measurement with no intervention) or ``sample_increase``
    (more rows of an existing design; produces no new regime). ``joint=True``
    declares that ``measured`` is observed jointly on each unit. ``unit`` (and
    ``cluster``), ``timing``, ``recruitment`` and ``cost`` with its ``cost_unit``
    are required semantics: a candidate without them is refused with
    ``study_candidate.wrong_contract``, and a cost unit is never assumed to
    equal outcome utility. ``evidence`` lists the laws the study would deliver;
    by default one law over the study's own population, interventions and
    margin. A candidate observing separate regimes cannot claim the joint law
    (it is classified ``invalid`` by :func:`repair`). ``feasible`` and ``notes``
    record feasibility; an infeasible candidate is never searched.
    """

    label: str
    kind: str
    population: str
    measured: Sequence[str]
    sample_size: int
    recruitment: str
    timing: str
    unit: str
    cost: int
    cost_unit: str
    interventions: Sequence[str] = ()
    joint: bool = True
    cluster: str | None = None
    whole_cluster_sampling: bool = False
    sample_budget: int | None = None
    feasible: bool = True
    notes: Sequence[str] = ()
    evidence: Sequence[ExpectedEvidence] | None = None
    provider: str | None = None

    def __post_init__(self) -> None:
        _text("label", self.label)
        if self.kind not in _KINDS:
            raise CausalValueError(f"kind must be one of {', '.join(_KINDS)}")
        _text("population", self.population)
        object.__setattr__(self, "measured", _names("measured", self.measured))
        object.__setattr__(self, "interventions", _names("interventions", self.interventions))
        object.__setattr__(self, "notes", tuple(_text("notes", n) for n in self.notes))
        if self.kind != "sample_increase" and self.evidence is None and not self.measured:
            raise CausalValueError("an experiment or observation measures at least one variable")
        for name in ("recruitment", "timing", "unit", "cost_unit"):
            _semantics(name, getattr(self, name))
        if self.cluster is not None:
            _semantics("a non-blank cluster identity", self.cluster)
        if self.provider is not None:
            _semantics("a non-blank provider", self.provider)
        for name in ("sample_size", "cost"):
            if _count(name, getattr(self, name)) <= 0:
                raise StudyCandidateRefusal(
                    f"a candidate needs a positive {name.replace('_', ' ')}",
                    reason_code="design_signal_invalid",
                    detail="study_candidate.wrong_contract",
                )
        if self.sample_budget is not None and _count("sample_budget", self.sample_budget) < 0:
            raise CausalValueError("sample_budget must be non-negative")
        if self.evidence is not None:
            evidence = tuple(self.evidence)
            if any(not isinstance(e, ExpectedEvidence) for e in evidence):
                raise CausalTypeError("evidence must be ExpectedEvidence values")
            object.__setattr__(self, "evidence", evidence)

    @classmethod
    def experiment(
        cls, label: str, *, population: str, interventions: Sequence[str], **fields: Any
    ) -> StudyCandidate:
        """An experiment setting ``interventions`` in ``population``."""
        return cls(
            label=label, kind="experiment", population=population,
            interventions=interventions, **fields,
        )  # fmt: skip

    @classmethod
    def observation(cls, label: str, *, population: str, **fields: Any) -> StudyCandidate:
        """An observational study (no hard intervention) in ``population``."""
        return cls(label=label, kind="observation", population=population, **fields)

    @classmethod
    def sample_increase(cls, label: str, *, population: str, **fields: Any) -> StudyCandidate:
        """More rows of an existing design; delivers no new regime."""
        fields.setdefault("measured", ())
        return cls(label=label, kind="sample_increase", population=population, **fields)

    def _wire(self) -> dict[str, Any]:
        if self.evidence is not None:
            evidence = [e._wire() for e in self.evidence]
        elif self.kind == "sample_increase":
            evidence = []
        else:
            evidence = [
                ExpectedEvidence(
                    self.population, self.measured, self.interventions, joint=self.joint
                )._wire()
            ]
        return {
            "label": self.label,
            "kind": self.kind,
            "population": self.population,
            "interventions": list(self.interventions),
            "measured": list(self.measured),
            "joint": bool(self.joint),
            "sample_size": self.sample_size,
            "recruitment": self.recruitment,
            "timing": self.timing,
            "unit": self.unit,
            "cluster": self.cluster,
            "whole_cluster_sampling": bool(self.whole_cluster_sampling),
            "cost": self.cost,
            "cost_unit": self.cost_unit,
            "sample_budget": self.sample_size if self.sample_budget is None else self.sample_budget,
            "feasible": bool(self.feasible),
            "notes": list(self.notes),
            "evidence": evidence,
            "provider": self.provider,
        }


@dataclass(frozen=True, slots=True)
class UnresolvedAssumption:
    """An assumption the contract still owes, never met by a proposed study."""

    id: str
    description: str
    required_check: str | None = None

    def __post_init__(self) -> None:
        _text("id", self.id)
        _text("description", self.description)


@dataclass(frozen=True, slots=True)
class TransportContract:
    """A failed catalog-aware classical transport contract.

    The query asks for ``P^target(outcomes | do(treatments))`` from the complete
    experimental family of ``source`` under the selection diagram
    (``graph`` plus ``selections``, the variables whose mechanisms may differ).
    ``catalog`` is the evidence supplied so far. It must not already identify
    the query (``identification_repair.not_a_failure``).
    """

    graph: Admg
    selections: Sequence[str]
    source: str
    target: str
    outcomes: Sequence[str]
    treatments: Sequence[str]
    catalog: EvidenceCatalog
    max_steps: int = 100_000
    max_depth: int = 256
    _stage: Any = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.graph, Admg):
            raise CausalTypeError("TransportContract requires graph=Admg(...)")
        if not isinstance(self.catalog, EvidenceCatalog):
            raise CausalTypeError("catalog must be an EvidenceCatalog")
        object.__setattr__(self, "selections", _names("selections", self.selections))
        object.__setattr__(self, "outcomes", _names("outcomes", self.outcomes, allow_empty=False))
        object.__setattr__(
            self, "treatments", _names("treatments", self.treatments, allow_empty=False)
        )
        stage = _call(
            _native.repair_transport_contract,
            self.graph,
            list(self.selections),
            _text("source", self.source),
            _text("target", self.target),
            list(self.outcomes),
            list(self.treatments),
            self.catalog,
            max_steps=_count("max_steps", self.max_steps),
            max_depth=_count("max_depth", self.max_depth),
        )
        object.__setattr__(self, "_stage", stage)

    @property
    def family(self) -> str:
        """``transport``."""
        return "transport"

    @property
    def contract_id(self) -> str:
        """The failed contract's stable identity."""
        return str(self._stage.contract_id)


@dataclass(frozen=True, slots=True)
class BackdoorContract:
    """A failed back-door contract on a fixed graph.

    The contract is to identify the effect of ``treatment`` on ``outcome`` in
    ``population`` by back-door adjustment over the variables ``observed``
    jointly with them so far. An adjustment set is read from one joint law, so a
    repair needs a study that observes treatment, outcome and the covariates
    together in that population. ``assumptions`` are unresolved declarations; a
    study cannot satisfy them.
    """

    graph: Dag
    treatment: str
    outcome: str
    population: str
    observed: Sequence[str]
    assumptions: Sequence[UnresolvedAssumption] = ()
    _stage: Any = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.graph, Dag):
            raise CausalTypeError("BackdoorContract requires graph=Dag(...)")
        object.__setattr__(self, "observed", _names("observed", self.observed))
        assumptions = tuple(self.assumptions)
        if any(not isinstance(a, UnresolvedAssumption) for a in assumptions):
            raise CausalTypeError("assumptions must be UnresolvedAssumption values")
        object.__setattr__(self, "assumptions", assumptions)
        stage = _call(
            _native.repair_backdoor_contract,
            self.graph,
            _text("treatment", self.treatment),
            _text("outcome", self.outcome),
            _text("population", self.population),
            list(self.observed),
            [(a.id, a.description, a.required_check) for a in assumptions],
        )
        object.__setattr__(self, "_stage", stage)

    @classmethod
    def from_identification(
        cls,
        identification: Any,
        *,
        population: str,
        observed: Sequence[str],
        assumptions: Sequence[UnresolvedAssumption] = (),
    ) -> BackdoorContract:
        """Read the graph, treatment and outcome off an :func:`antecedent.identify` result.

        ``identification.graph`` is a :class:`~antecedent.Dag` or an edge list
        (then ``identification.names`` names the nodes) and
        ``identification.query`` carries ``treatment`` and ``outcome``.
        """
        graph = identification.graph
        if not isinstance(graph, Dag):
            if not isinstance(graph, (list, tuple)) or not identification.names:
                raise CausalValueError(
                    "the identification's graph must be a Dag or an edge list with names"
                )
            graph = Dag.from_edges(list(identification.names), [tuple(e) for e in graph])
        query = identification.query
        return cls(
            graph=graph,
            treatment=query.treatment,
            outcome=query.outcome,
            population=population,
            observed=observed,
            assumptions=assumptions,
        )

    @property
    def family(self) -> str:
        """``backdoor``."""
        return "backdoor"

    @property
    def contract_id(self) -> str:
        """The failed contract's stable identity."""
        return str(self._stage.contract_id)


@dataclass(frozen=True, slots=True)
class ZTransportContract:
    """A checked z-transport missing-evidence snapshot as a repair contract.

    ``failure_snapshot`` is exported by an existing z-transport identification
    stage's ``failure_snapshot(catalog)``. ``names`` supplies its exact graph
    coordinate order. Construction independently replays the snapshot; a graph
    obstruction, exhausted search or failure without a checked formula refuses.
    The generic repair report exports and independently replays the same source
    proof leaves and hypothetical candidate evidence.
    """

    names: Sequence[str]
    failure_snapshot: bytes
    max_steps: int = 100_000
    max_depth: int = 256
    memory_bytes: int | None = None
    _stage: Any = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        names = _names("names", self.names, allow_empty=False)
        if not isinstance(self.failure_snapshot, bytes):
            raise CausalTypeError("failure_snapshot must be bytes from a z-transport stage")
        object.__setattr__(self, "names", names)
        object.__setattr__(
            self,
            "_stage",
            _call(
                _native.repair_z_transport_contract,
                list(names),
                self.failure_snapshot,
                max_steps=_count("max_steps", self.max_steps),
                max_depth=_count("max_depth", self.max_depth),
                memory_bytes=self.memory_bytes,
            ),
        )

    @classmethod
    def from_identification(
        cls, identification: Any, *, catalog: EvidenceCatalog, names: Sequence[str]
    ) -> ZTransportContract:
        """Use the actual identification stage's checked failure snapshot."""
        if not callable(getattr(identification, "failure_snapshot", None)):
            raise CausalTypeError("from_identification requires a z-transport identification stage")
        if not isinstance(catalog, EvidenceCatalog):
            raise CausalTypeError("catalog must be EvidenceCatalog")
        return cls(names=names, failure_snapshot=bytes(identification.failure_snapshot(catalog)))

    @property
    def family(self) -> str:
        """``z_transport``."""
        return "z_transport"

    @property
    def contract_id(self) -> str:
        """The frozen failure's semantic identity."""
        return str(self._stage.contract_id)


Contract = TransportContract | BackdoorContract | ZTransportContract


def _contract_of(failure: Any, population: str | None, observed: Sequence[str] | None) -> Any:
    if isinstance(failure, (TransportContract, BackdoorContract, ZTransportContract)):
        return failure
    if hasattr(failure, "graph") and hasattr(failure, "query") and hasattr(failure, "status"):
        if population is None or observed is None:
            raise CausalValueError(
                "an identification names no population or observed variables; pass "
                "population= and observed= to build its back-door contract"
            )
        return BackdoorContract.from_identification(
            failure, population=population, observed=observed
        )
    raise CausalTypeError(
        "obligations() requires a TransportContract, a BackdoorContract, a ZTransportContract or a failed identification"
    )


def obligations(
    contract_failure: Any,
    *,
    quantities: Mapping[str, ScientificQuantity] | None = None,
    population: str | None = None,
    observed: Sequence[str] | None = None,
) -> tuple[EvidenceObligation, ...]:
    """The unresolved evidence obligations of a failed contract.

    Accepts a :class:`TransportContract`, :class:`BackdoorContract`, :class:`ZTransportContract` or a failed
    identification (then ``population=`` and ``observed=`` say where and what is
    measured so far). Each obligation retains its source proof step; a contract
    that already identifies refuses with ``identification_repair.not_a_failure``
    when it is built. ``quantities`` explicitly maps every requested variable to
    its scientific descriptor; Rust validates the binding and includes every
    semantic dimension in the obligation ID. With no declaration, obligations
    remain structural requests with an empty coordinate map, not measurements
    in inferred units.
    """
    contract = _contract_of(contract_failure, population, observed)
    if quantities is not None and not isinstance(quantities, Mapping):
        raise CausalTypeError(
            "quantities must be a mapping of variable names to ScientificQuantity descriptors"
        )
    if quantities is not None and any(
        not isinstance(name, str) or not isinstance(quantity, ScientificQuantity)
        for name, quantity in quantities.items()
    ):
        raise CausalTypeError(
            "quantities must map variable names to ScientificQuantity descriptors"
        )
    declarations = json.dumps(
        {name: quantity._wire() for name, quantity in (quantities or {}).items()}
    )
    return tuple(
        EvidenceObligation._from_native(o) for o in contract._stage.obligations(declarations)
    )


@dataclass(frozen=True, slots=True)
class RepairLimits:
    """Declared bounds of the subset search.

    ``max_operations`` bounds the subsets evaluated, ``max_depth`` the largest
    subset size (at most four) and ``memory_bytes`` the live-state memory. A bound
    that stops the search is reported as ``exhausted`` with every unevaluated
    subset listed; subsets beyond ``max_depth`` are never examined and say so.
    """

    max_operations: int = 4096
    max_depth: int = 3
    memory_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class Derivation:
    """The re-verified derivation behind a sufficient subset."""

    checker: str
    steps: tuple[str, ...]
    verified: bool


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    """One candidate or candidate subset and how it fared.

    ``classification`` is ``verified_sufficient`` (the family's checker identified
    the hypothetical evidence and the derivation re-verified), ``insufficient``
    (the checker ran and the contract is still unmet), ``not_certified`` (the
    checker could not certify, or an unresolved assumption remains; no
    impossibility is implied), ``invalid`` (a malformed, duplicate or infeasible
    candidate) or ``unevaluated`` (a budget stop left it unexamined).
    ``addressed`` and ``unmet`` are obligation ids that pass, or fail, the
    necessary population/regime/joint-law screen: a screen, not a verdict.
    """

    candidates: tuple[str, ...]
    labels: tuple[str, ...]
    classification: str
    cost: int
    cost_unit: str | None
    sample_budget: int
    reasons: tuple[str, ...]
    addressed: tuple[str, ...]
    unmet: tuple[str, ...]
    derivation: Derivation | None

    @classmethod
    def _from_native(cls, value: Mapping[str, Any]) -> CandidateOutcome:
        derivation = value["derivation"]
        return cls(
            candidates=tuple(value["candidates"]),
            labels=tuple(value["labels"]),
            classification=value["classification"],
            cost=value["cost_units"],
            cost_unit=value["cost_unit"],
            sample_budget=value["sample_budget"],
            reasons=tuple(value["reasons"]),
            addressed=tuple(value["addressed"]),
            unmet=tuple(value["unmet"]),
            derivation=None
            if derivation is None
            else Derivation(
                derivation["checker"], tuple(derivation["steps"]), derivation["verified"]
            ),
        )


@dataclass(frozen=True, slots=True)
class SearchReceipt:
    """What the subset search consumed and left.

    ``unevaluated`` lists subsets a stop left unexamined (``unevaluated_total``
    counts all of them); ``beyond_declared_depth`` says subsets larger than the
    declared depth exist and were not examined. Neither is ever a verdict.
    ``stop`` is ``search.operations``, ``search.depth``, ``search.memory`` or
    ``search.cancelled`` when a bound ended the search, else ``None``.
    """

    operations_limit: int
    depth_limit: int
    memory_limit_bytes: int
    operations_consumed: int
    depth_reached: int
    explored: tuple[str, ...]
    unevaluated: tuple[str, ...]
    unevaluated_total: int
    dominated_skipped: int
    beyond_declared_depth: bool
    stop: str | None


class RepairResult:
    """A finished repair: classification table, ranking, receipt and artifact.

    :attr:`outcome` is ``repaired`` (a verified-sufficient subset exists),
    ``none_certified`` (nothing in the declared candidates was certified; never
    an impossibility claim) or ``exhausted`` (a budget stop with nothing
    certified; never a verdict). ``inference_claim`` is always ``none``: a
    repair says the contract would identify if the studies deliver exactly their
    declared evidence, not that any estimate is licensed.
    """

    __slots__ = ("_report", "_stage", "obligations", "ranked", "receipt", "table")

    def __init__(self, stage: Any) -> None:
        report = stage.report()
        self._stage = stage
        self._report = report
        self.obligations: tuple[EvidenceObligation, ...] = tuple(
            EvidenceObligation._from_native(o) for o in report["obligations"]
        )
        self.table: tuple[CandidateOutcome, ...] = tuple(
            CandidateOutcome._from_native(o) for o in report["outcomes"]
        )
        self.ranked: tuple[CandidateOutcome, ...] = tuple(self.table[i] for i in report["ranked"])
        receipt = report["receipt"]
        self.receipt = SearchReceipt(
            operations_limit=receipt["operations_limit"],
            depth_limit=receipt["depth_limit"],
            memory_limit_bytes=receipt["memory_limit_bytes"],
            operations_consumed=receipt["operations_consumed"],
            depth_reached=receipt["depth_reached"],
            explored=tuple(receipt["explored"]),
            unevaluated=tuple(receipt["unevaluated"]),
            unevaluated_total=receipt["unevaluated_total"],
            dominated_skipped=receipt["dominated_skipped"],
            beyond_declared_depth=receipt["beyond_declared_depth"],
            stop=receipt["stop"],
        )

    @property
    def outcome(self) -> str:
        """``repaired``, ``none_certified`` or ``exhausted``."""
        return str(self._report["outcome"])

    @property
    def family(self) -> str:
        """The family whose theorem checker decided (``transport``, ``backdoor`` or ``z_transport``)."""
        return str(self._report["family"])

    @property
    def contract_id(self) -> str:
        """The failed contract's stable identity."""
        return str(self._report["contract"])

    @property
    def reason_code(self) -> str | None:
        """The registered reason code when nothing was verified, else ``None``."""
        return self._report["reason"]

    @property
    def detail(self) -> str | None:
        """The stable detail when nothing was verified, else ``None``."""
        return self._report["detail"]

    @property
    def inference_claim(self) -> str:
        """Always ``none``."""
        return str(self._report["inference_claim"])

    @property
    def best(self) -> CandidateOutcome | None:
        """The best verified-sufficient subset under the objective, if any."""
        return self.ranked[0] if self.ranked else None

    @property
    def sufficient(self) -> tuple[CandidateOutcome, ...]:
        """Alias of :attr:`ranked`: verified-sufficient subsets, best first."""
        return self.ranked

    def classification(self, *labels: str) -> str:
        """The classification of the subset made of the candidates named ``labels``."""
        wanted = frozenset(labels)
        for row in self.table:
            if frozenset(row.labels) == wanted and len(row.labels) == len(labels):
                return row.classification
        raise CausalValueError(f"no outcome for the subset {sorted(wanted)}")

    def export(self) -> bytes:
        """A portable ``repair_search_receipt_v1`` artifact; see :func:`consume`."""
        return bytes(_call(self._stage.export))

    def __repr__(self) -> str:
        return (
            f"<RepairResult {self.family} outcome={self.outcome} "
            f"subsets={len(self.table)} sufficient={len(self.ranked)}>"
        )


def repair(
    contract: TransportContract | BackdoorContract,
    candidates: Sequence[StudyCandidate],
    objective: str = "minimize_cost",
    limits: RepairLimits | None = None,
    *,
    cancel: Any = None,
) -> RepairResult:
    """Which candidates, singly or in bounded subsets, repair the failed contract.

    Each candidate's hypothetical evidence is applied and the contract's own
    theorem checker re-run: a variable-name match never repairs anything, and
    two separate studies never supply one joint-regime law. ``objective`` ranks
    sufficient subsets by ``minimize_cost`` (all candidates must declare the same
    cost unit) or ``minimize_sample_budget``.

    Refusals carry their reason code: ``invalid_argument`` for a request beyond
    the declared bounds (``identification_repair.bounds_exceeded``: at most 16
    candidates, depth 4, 100000 operations), ``design_cost_units_mismatch`` for
    cost ranking over different cost units and ``transport_budget_cancel`` when
    the budget stops before the search (:class:`RepairBudgetRefusal`). A stop
    during the search is a result with ``outcome == "exhausted"``.
    """
    if not isinstance(contract, (TransportContract, BackdoorContract, ZTransportContract)):
        raise CausalTypeError(
            "repair requires a TransportContract, BackdoorContract or ZTransportContract"
        )
    candidates = tuple(candidates)
    if any(not isinstance(c, StudyCandidate) for c in candidates):
        raise CausalTypeError("candidates must be StudyCandidate values")
    if objective not in _OBJECTIVES:
        raise CausalValueError(f"objective must be one of {', '.join(_OBJECTIVES)}")
    limits = RepairLimits() if limits is None else limits
    stage = _call(
        contract._stage.repair,
        json.dumps([c._wire() for c in candidates]),
        objective,
        max_operations=_count("max_operations", limits.max_operations),
        max_depth=_count("max_depth", limits.max_depth),
        memory_bytes=limits.memory_bytes,
        cancel=cancel,
    )
    return RepairResult(stage)


def consume(
    artifact: bytes,
    *,
    max_operations: int = _CONSUME_MAX_OPERATIONS,
    max_depth: int = _CONSUME_MAX_DEPTH,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> RepairResult:
    """Independently replay an exported repair report and return it.

    The consumer refuses stored limits above its own before any work, checks
    every digest, replays each stored subset's family check from its stored
    hypothetical delta, then replays the whole search under the stored limits and
    accepts only an identical report. A resealed edit of a candidate, an
    obligation, a classification or the receipt is
    :class:`RepairArtifactRefusal` (``repair_artifact.replay_mismatch`` or
    ``repair_artifact.invalid_artifact``); a cancelled replay is
    :class:`RepairBudgetRefusal`, never a verdict. Replay does not protect against
    a producer that states another contract or candidate universe and re-seals
    honestly, nor does it check that arriving studies will deliver their declared
    evidence.
    """
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    stage = _call(
        _native.consume_repair_artifact,
        artifact,
        max_operations=_count("max_operations", max_operations),
        max_depth=_count("max_depth", max_depth),
        memory_bytes=memory_bytes,
        cancel=cancel,
    )
    return RepairResult(stage)


__all__ = [
    "BackdoorContract",
    "CandidateOutcome",
    "Derivation",
    "EvidenceObligation",
    "ExpectedEvidence",
    "RepairArtifactRefusal",
    "RepairBudgetRefusal",
    "RepairLimits",
    "RepairRefusal",
    "RepairResult",
    "SearchReceipt",
    "StudyCandidate",
    "StudyCandidateRefusal",
    "TransportContract",
    "ZTransportContract",
    "UnresolvedAssumption",
    "consume",
    "obligations",
    "repair",
]
