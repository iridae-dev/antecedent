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

Seven families are supported, each deferring to its own theorem checker rather
than matching variable names: :class:`TransportContract` (the catalog-aware
classical transport identifier, re-verified after the hypothetical evidence is
applied) and :class:`BackdoorContract` (the back-door identifier on the one
joint law a study would deliver; covariates measured in separate studies never
combine), and :class:`ZTransportContract` (a checked z-transport failure snapshot
whose original expression leaves are re-verified and bound). A failed
identification also reads as one contract per identifier:
:class:`FrontDoorContract` (a mediator set measured jointly with treatment and
outcome), :class:`InstrumentContract` (an instrument measured jointly, with its
relevance and exclusion left as ``establish_assumption`` obligations that no
study satisfies), :class:`HedgeContract` (experiments on the treatment or on a
verified surrogate on the hedge) and :class:`SensitivityContract` (the
unmeasured confounder a non-robust sensitivity analysis names). Each refuses
``identification_repair.not_a_failure`` when the law already observed identifies
the effect. Every candidate lands in exactly one class: ``verified_sufficient``,
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
from functools import partial
from typing import Any, Literal, Protocol, cast

from . import _native
from ._artifact_protocol import (
    ExportedArtifact,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from ._columns import coerce_fields
from ._validate import count, names, text
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError, StructuredRefusal
from .graph import Admg, Dag
from .joint_distribution import ScientificQuantity
from .transport._impl import EvidenceCatalog

#: What an evidence obligation asks for.
ObligationKind = Literal[
    "measure",
    "intervene",
    "observe_population",
    "observe_environment",
    "increase_sample",
    "provide_joint_law",
    "provide_conditional_law",
    "establish_support",
    "establish_assumption",
]
#: How a candidate (subset) fared in the repair search.
Classification = Literal[
    "verified_sufficient", "insufficient", "not_certified", "invalid", "unevaluated"
]
#: How a whole repair search ended; never an impossibility claim.
RepairOutcome = Literal["repaired", "none_certified", "exhausted"]
#: Which theorem checker decided a repair.
RepairFamily = Literal[
    "transport",
    "backdoor",
    "z_transport",
    "frontdoor",
    "instrument",
    "hedge",
    "sensitivity",
]
#: The sensitivity analysis a :class:`SensitivityContract` finding came from.
SensitivityMethod = Literal["e_value", "omitted_variable_bias", "tipping"]
#: The bound that ended a search early.
SearchStop = Literal["search.operations", "search.depth", "search.memory", "search.cancelled"]


class CancelToken(Protocol):
    """A cancellation token (for example :class:`antecedent.CancellationToken`)."""

    def cancel(self) -> None:
        """Request cancellation."""
        ...

    def is_cancelled(self) -> bool:
        """Whether cancellation was requested."""
        ...


class FailedIdentification(Protocol):
    """A failed identification stage: it exposes its ``graph``, ``query`` and ``status``."""

    @property
    def graph(self) -> object:
        """The identification graph."""
        ...

    @property
    def query(self) -> Any:
        """The identified query (carries ``treatment`` and ``outcome``)."""
        ...

    @property
    def status(self) -> object:
        """The identification status."""
        ...

    @property
    def names(self) -> Sequence[str] | None:
        """The node names an edge-list graph refers to, when the graph is not a ``Dag``."""
        ...


class ZTransportIdentification(Protocol):
    """A z-transport identification stage: it can freeze its checked failure."""

    def failure_snapshot(self, catalog: EvidenceCatalog) -> bytes:
        """The checked failure snapshot under ``catalog``."""
        ...


_KINDS = (
    "experiment",
    "observation",
    "sample_increase",
    "instrument",
    "natural_experiment",
    "panel",
)
_OBJECTIVES = ("minimize_cost", "minimize_sample_budget")
_REFUSAL = re.compile(
    r"^(?:refused: )?reason=(?P<code>[a-z_]+): (?P<detail>[a-z_]+\.[a-z_]+): (?P<message>.*)$",
    re.DOTALL,
)
_CONSUME_MAX_OPERATIONS = 100_000
_CONSUME_MAX_DEPTH = 4


class RepairRefusal(StructuredRefusal):
    """A refused repair request, obligation or candidate.

    A :class:`~antecedent.errors.StructuredRefusal`: ``reason_code`` (alias ``code``) is the
    registered runtime-refusal code and :attr:`detail` the stable ``<family>.<slot>`` detail
    (for example ``identification_repair.not_a_failure`` or
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
        super().__init__(
            {
                "code": reason_code,
                "detail": detail or "",
                "message": message,
                "remedy": remedy,
            },
            text=text,
        )
        #: Stable ``<family>.<slot>`` detail, when the refusal names one.
        self.detail = detail


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
        elif detail.split(".", 1)[0] in ("repair_artifact", "evidence_plan_artifact"):
            kind = RepairArtifactRefusal
        else:
            kind = RepairRefusal
        raise kind(parsed["message"], reason_code=parsed["code"], detail=detail) from error


_text = partial(text, non_empty=True)

_names = names

_count = count


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
    kind: ObligationKind
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

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form of this request."""
        return {
            "id": self.id,
            "kind": self.kind,
            "scope": self.scope,
            "variables": list(self.variables),
            "population": self.population,
            "interventions": list(self.interventions),
            "conditioned_on": list(self.conditioned_on),
            "joint": self.joint,
            "reason": self.reason,
            "required_slots": list(self.required_slots),
            "min_additional_samples": self.min_additional_samples,
            "family": self.family,
            "source": self.source,
            "proof_step": self.proof_step,
            "satisfiable_by_study": self.satisfiable_by_study,
            "quantities": {name: q._wire() for name, q in self.quantities.items()},
        }

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
        _text(self.population, "population")
        coerce_fields(self, measured=_names(self.measured, "measured", allow_empty=False))
        coerce_fields(self, interventions=_names(self.interventions, "interventions"))
        coerce_fields(self, conditioned_on=_names(self.conditioned_on, "conditioned_on"))
        coerce_fields(self, levels={str(k): float(v) for k, v in self.levels.items()})

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
    ``observation`` (measurement with no intervention), ``sample_increase``
    (more rows of an existing design; produces no new regime), ``instrument``
    (an instrument, the treatment and the outcome measured jointly; it never
    establishes that the instrument is valid), ``natural_experiment`` (variation
    the investigator does not set; no hard intervention, and its as-if-random
    assignment is an assumption) or ``panel`` (repeated measurement of the same
    units; time invariance of the unmeasured causes is an assumption). ``joint=True``
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
        _text(self.label, "label")
        if self.kind not in _KINDS:
            raise CausalValueError(f"kind must be one of {', '.join(_KINDS)}")
        _text(self.population, "population")
        coerce_fields(self, measured=_names(self.measured, "measured"))
        coerce_fields(self, interventions=_names(self.interventions, "interventions"))
        coerce_fields(self, notes=tuple(_text(n, "notes") for n in self.notes))
        if self.kind != "sample_increase" and self.evidence is None and not self.measured:
            raise CausalValueError("an experiment or observation measures at least one variable")
        for name in ("recruitment", "timing", "unit", "cost_unit"):
            _semantics(name, getattr(self, name))
        if self.cluster is not None:
            _semantics("a non-blank cluster identity", self.cluster)
        if self.provider is not None:
            _semantics("a non-blank provider", self.provider)
        for name in ("sample_size", "cost"):
            if _count(getattr(self, name), name) <= 0:
                raise StudyCandidateRefusal(
                    f"a candidate needs a positive {name.replace('_', ' ')}",
                    reason_code="design_signal_invalid",
                    detail="study_candidate.wrong_contract",
                )
        if self.sample_budget is not None and _count(self.sample_budget, "sample_budget") < 0:
            raise CausalValueError("sample_budget must be non-negative")
        if self.evidence is not None:
            evidence = tuple(self.evidence)
            if any(not isinstance(e, ExpectedEvidence) for e in evidence):
                raise CausalTypeError("evidence must be ExpectedEvidence values")
            coerce_fields(self, evidence=evidence)

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

    @classmethod
    def instrument(cls, label: str, *, population: str, **fields: Any) -> StudyCandidate:
        """An instrument, the treatment and the outcome measured jointly."""
        return cls(label=label, kind="instrument", population=population, **fields)

    @classmethod
    def natural_experiment(cls, label: str, *, population: str, **fields: Any) -> StudyCandidate:
        """Observation exploiting variation the investigator does not set."""
        return cls(label=label, kind="natural_experiment", population=population, **fields)

    @classmethod
    def panel(cls, label: str, *, population: str, **fields: Any) -> StudyCandidate:
        """Repeated measurement of the same units, observed jointly."""
        return cls(label=label, kind="panel", population=population, **fields)

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
        _text(self.id, "id")
        _text(self.description, "description")


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
        coerce_fields(self, selections=_names(self.selections, "selections"))
        coerce_fields(self, outcomes=_names(self.outcomes, "outcomes", allow_empty=False))
        coerce_fields(self, treatments=_names(self.treatments, "treatments", allow_empty=False))
        stage = _call(
            _native.repair_transport_contract,
            self.graph,
            list(self.selections),
            _text(self.source, "source"),
            _text(self.target, "target"),
            list(self.outcomes),
            list(self.treatments),
            self.catalog,
            max_steps=_count(self.max_steps, "max_steps"),
            max_depth=_count(self.max_depth, "max_depth"),
        )
        coerce_fields(self, _stage=stage)

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
        coerce_fields(self, observed=_names(self.observed, "observed"))
        assumptions = tuple(self.assumptions)
        if any(not isinstance(a, UnresolvedAssumption) for a in assumptions):
            raise CausalTypeError("assumptions must be UnresolvedAssumption values")
        coerce_fields(self, assumptions=assumptions)
        stage = _call(
            _native.repair_backdoor_contract,
            self.graph,
            _text(self.treatment, "treatment"),
            _text(self.outcome, "outcome"),
            _text(self.population, "population"),
            list(self.observed),
            [(a.id, a.description, a.required_check) for a in assumptions],
        )
        coerce_fields(self, _stage=stage)

    @classmethod
    def from_identification(
        cls,
        identification: FailedIdentification,
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
        names = _names(self.names, "names", allow_empty=False)
        if not isinstance(self.failure_snapshot, bytes):
            raise CausalTypeError("failure_snapshot must be bytes from a z-transport stage")
        coerce_fields(self, names=names)
        coerce_fields(
            self,
            _stage=_call(
                _native.repair_z_transport_contract,
                list(names),
                self.failure_snapshot,
                max_steps=_count(self.max_steps, "max_steps"),
                max_depth=_count(self.max_depth, "max_depth"),
                memory_bytes=self.memory_bytes,
            ),
        )

    @classmethod
    def from_identification(
        cls,
        identification: ZTransportIdentification,
        *,
        catalog: EvidenceCatalog,
        names: Sequence[str],
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


def _dag_of(identification: FailedIdentification) -> Dag:
    """The directed graph of a failed identification (a ``Dag`` or a named edge list)."""
    graph = identification.graph
    if isinstance(graph, Dag):
        return graph
    if not isinstance(graph, (list, tuple)) or not identification.names:
        raise CausalValueError(
            "the identification's graph must be a Dag or an edge list with names"
        )
    return Dag.from_edges(list(identification.names), [tuple(e) for e in graph])


@dataclass(frozen=True, slots=True)
class FrontDoorContract:
    """A failed front-door contract on a fixed graph.

    The contract is to identify the effect of ``treatment`` on ``outcome`` in
    ``population`` through a mediator set that intercepts every directed path and
    is shielded from the treatment's confounding. The criterion is read from one
    joint law, so a repair needs a study that observes treatment, outcome and the
    mediators together; the graph may hold variables nobody measures (the
    unmeasured confounder, an unmeasured mediator). ``observed`` are the
    variables measured jointly so far; if they already identify the effect the
    contract refuses with ``identification_repair.not_a_failure``.
    """

    graph: Dag
    treatment: str
    outcome: str
    population: str
    observed: Sequence[str]
    _stage: Any = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.graph, Dag):
            raise CausalTypeError("FrontDoorContract requires graph=Dag(...)")
        coerce_fields(self, observed=_names(self.observed, "observed"))
        stage = _call(
            _native.repair_frontdoor_contract,
            self.graph,
            _text(self.treatment, "treatment"),
            _text(self.outcome, "outcome"),
            _text(self.population, "population"),
            list(self.observed),
        )
        coerce_fields(self, _stage=stage)

    @classmethod
    def from_identification(
        cls,
        identification: FailedIdentification,
        *,
        population: str,
        observed: Sequence[str],
    ) -> FrontDoorContract:
        """Read the graph, treatment and outcome off an :func:`antecedent.identify` result."""
        query = identification.query
        return cls(
            graph=_dag_of(identification),
            treatment=query.treatment,
            outcome=query.outcome,
            population=population,
            observed=observed,
        )

    @property
    def family(self) -> str:
        """``frontdoor``."""
        return "frontdoor"

    @property
    def contract_id(self) -> str:
        """The failed contract's stable identity."""
        return str(self._stage.contract_id)


@dataclass(frozen=True, slots=True)
class InstrumentContract:
    """A failed instrumental-variable contract on a fixed graph.

    A study can deliver the joint law of an instrument, the treatment and the
    outcome, but the instrument's relevance, its exclusion and the effect
    restriction are ``establish_assumption`` obligations that no study satisfies:
    a repair report over this contract never certifies on data alone. A graph with
    no instrument owes the search for one (``iv.existence``).
    """

    graph: Dag
    treatment: str
    outcome: str
    population: str
    observed: Sequence[str]
    _stage: Any = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.graph, Dag):
            raise CausalTypeError("InstrumentContract requires graph=Dag(...)")
        coerce_fields(self, observed=_names(self.observed, "observed"))
        stage = _call(
            _native.repair_instrument_contract,
            self.graph,
            _text(self.treatment, "treatment"),
            _text(self.outcome, "outcome"),
            _text(self.population, "population"),
            list(self.observed),
        )
        coerce_fields(self, _stage=stage)

    @classmethod
    def from_identification(
        cls,
        identification: FailedIdentification,
        *,
        population: str,
        observed: Sequence[str],
    ) -> InstrumentContract:
        """Read the graph, treatment and outcome off an :func:`antecedent.identify` result."""
        query = identification.query
        return cls(
            graph=_dag_of(identification),
            treatment=query.treatment,
            outcome=query.outcome,
            population=population,
            observed=observed,
        )

    @property
    def family(self) -> str:
        """``instrument``."""
        return "instrument"

    @property
    def contract_id(self) -> str:
        """The failed contract's stable identity."""
        return str(self._stage.contract_id)


@dataclass(frozen=True, slots=True)
class HedgeContract:
    """A hedge: an ADMG in which observation cannot identify the effect.

    Only experiments answer it. An experiment that sets ``treatment`` and
    measures ``outcome`` observes the interventional law directly; an experiment
    on a hedge node that intercepts every directed path from treatment to
    outcome, while its response to the treatment is identified from observation,
    identifies the effect through ``P(y | do(x)) = sum_w P(w | do(x)) P(y |
    do(w))``. An observation never does. ``observed`` are the variables measured
    jointly so far (the surrogate route needs all of them).
    """

    graph: Admg
    treatment: str
    outcome: str
    population: str
    observed: Sequence[str]
    _stage: Any = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.graph, Admg):
            raise CausalTypeError("HedgeContract requires graph=Admg(...)")
        coerce_fields(self, observed=_names(self.observed, "observed"))
        stage = _call(
            _native.repair_hedge_contract,
            self.graph,
            _text(self.treatment, "treatment"),
            _text(self.outcome, "outcome"),
            _text(self.population, "population"),
            list(self.observed),
        )
        coerce_fields(self, _stage=stage)

    @classmethod
    def from_identification(
        cls,
        identification: FailedIdentification,
        *,
        population: str,
        observed: Sequence[str],
    ) -> HedgeContract:
        """Read the ADMG, treatment and outcome off a failed :func:`antecedent.identify` result."""
        graph = identification.graph
        if not isinstance(graph, Admg):
            raise CausalValueError("the identification's graph must be an Admg to read a hedge")
        query = identification.query
        return cls(
            graph=graph,
            treatment=query.treatment,
            outcome=query.outcome,
            population=population,
            observed=observed,
        )

    @property
    def family(self) -> str:
        """``hedge``."""
        return "hedge"

    @property
    def contract_id(self) -> str:
        """The failed contract's stable identity."""
        return str(self._stage.contract_id)


@dataclass(frozen=True, slots=True)
class SensitivityContract:
    """A non-robust sensitivity finding: the unmeasured confounder to measure.

    ``strength`` is the confounder strength at which the conclusion is overturned
    (the e-value, the robustness value or the tipping parameter named by
    ``method``). The contract models one more common cause of ``treatment`` and
    ``outcome`` and is repaired by a study that measures it jointly with them. A
    ``robust`` finding owes nothing and refuses with
    ``identification_repair.not_a_failure``.
    """

    graph: Dag
    treatment: str
    outcome: str
    population: str
    observed: Sequence[str]
    method: SensitivityMethod
    strength: float
    robust: bool = False
    _stage: Any = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.graph, Dag):
            raise CausalTypeError("SensitivityContract requires graph=Dag(...)")
        coerce_fields(self, observed=_names(self.observed, "observed"))
        if isinstance(self.strength, bool) or not isinstance(self.strength, (int, float)):
            raise CausalTypeError("strength must be a number")
        stage = _call(
            _native.repair_sensitivity_contract,
            self.graph,
            _text(self.treatment, "treatment"),
            _text(self.outcome, "outcome"),
            _text(self.population, "population"),
            list(self.observed),
            str(self.method),
            float(self.strength),
            bool(self.robust),
        )
        coerce_fields(self, _stage=stage)

    @classmethod
    def from_identification(
        cls,
        identification: FailedIdentification,
        *,
        population: str,
        observed: Sequence[str],
        method: SensitivityMethod,
        strength: float,
        robust: bool = False,
    ) -> SensitivityContract:
        """Read the graph, treatment and outcome off an :func:`antecedent.identify` result."""
        query = identification.query
        return cls(
            graph=_dag_of(identification),
            treatment=query.treatment,
            outcome=query.outcome,
            population=population,
            observed=observed,
            method=method,
            strength=strength,
            robust=robust,
        )

    @property
    def family(self) -> str:
        """``sensitivity``."""
        return "sensitivity"

    @property
    def contract_id(self) -> str:
        """The failed contract's stable identity."""
        return str(self._stage.contract_id)


@dataclass(frozen=True, slots=True)
class SensitivityFinding:
    """A sensitivity analysis whose conclusion an unmeasured confounder would overturn.

    ``strength`` is the confounder strength at which the conclusion is overturned (the
    e-value, the robustness value or the tipping parameter named by ``method``). A
    ``robust`` finding owes nothing and adds no route.
    """

    method: SensitivityMethod
    strength: float
    robust: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.strength, bool) or not isinstance(self.strength, (int, float)):
            raise CausalTypeError("strength must be a number")

    def _wire(self) -> tuple[str, float, bool]:
        return (str(self.method), float(self.strength), bool(self.robust))


@dataclass(frozen=True, slots=True)
class FailureContract:
    """A failed identification as every theorem route a study could repair.

    Where :class:`BackdoorContract` and its siblings freeze one route, this freezes
    the failure: the back-door, front-door, instrument and hedge families (and, given a
    non-robust ``sensitivity`` finding, the family that measures the confounder it
    names) are rebuilt from ``graph``, ``treatment``, ``outcome`` and the variables
    ``observed`` jointly so far. :func:`obligations` lists what every route owes,
    :func:`generate_candidates` the unfinished studies those obligations determine and
    :func:`antecedent.design.evidence_plan` verifies declared studies on every route and
    orders the options that close the gap. The unmeasured common cause of each
    bidirected edge is nameable in candidates as ``common_cause_of_<a>_<b>`` and the
    sensitivity confounder as ``unmeasured_confounder``. An effect the graph identifies
    refuses ``identification_repair.not_a_failure``.
    """

    graph: Dag | Admg
    treatment: str
    outcome: str
    population: str
    observed: Sequence[str]
    sensitivity: SensitivityFinding | None = None
    _stage: Any = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.graph, (Dag, Admg)):
            raise CausalTypeError("FailureContract requires graph=Dag(...) or Admg(...)")
        if self.sensitivity is not None and not isinstance(self.sensitivity, SensitivityFinding):
            raise CausalTypeError("sensitivity must be a SensitivityFinding")
        coerce_fields(self, observed=_names(self.observed, "observed"))
        native = (
            _native.evidence_failure_admg
            if isinstance(self.graph, Admg)
            else _native.evidence_failure_dag
        )
        stage = _call(
            native,
            self.graph,
            _text(self.treatment, "treatment"),
            _text(self.outcome, "outcome"),
            _text(self.population, "population"),
            list(self.observed),
            None if self.sensitivity is None else self.sensitivity._wire(),
        )
        coerce_fields(self, _stage=stage)

    @classmethod
    def from_identification(
        cls,
        identification: FailedIdentification,
        *,
        population: str,
        observed: Sequence[str],
        sensitivity: SensitivityFinding | None = None,
    ) -> FailureContract:
        """Read the graph, treatment and outcome off a failed :func:`antecedent.identify` result."""
        graph = identification.graph
        if not isinstance(graph, Admg):
            graph = _dag_of(identification)
        query = identification.query
        return cls(
            graph=graph,
            treatment=query.treatment,
            outcome=query.outcome,
            population=population,
            observed=observed,
            sensitivity=sensitivity,
        )

    @property
    def family(self) -> str:
        """``failure``."""
        return "failure"

    @property
    def contract_id(self) -> str:
        """The failed contract's stable identity."""
        return str(self._stage.contract_id)

    @property
    def routes(self) -> tuple[str, ...]:
        """The theorem routes a study could repair, then the routes no checker decides."""
        return tuple(self._stage.routes)

    @property
    def variable_names(self) -> tuple[str, ...]:
        """Every name a candidate may use, the nameable unmeasured causes included."""
        return tuple(self._stage.variable_names)

    def next_actions(self) -> tuple[dict[str, Any], ...]:
        """The evidence-gathering next actions the obligations imply.

        ``measure_variable:<name>`` for a variable the obligations name that is not measured
        today, ``run_experiment:<name>`` for a variable an obligation sets, ``find_instrument``
        and ``collect_panel``, each with a description that names the obligation and its proof
        step. An assumption obligation (an instrument's validity, the time invariance a panel
        relies on) yields no action: no study satisfies it. Advice only; nothing is executed.
        """
        return tuple(dict(action) for action in self._stage.next_actions())

    def capability_report(self) -> dict[str, Any]:
        """The failure as the capability report a host reads beside the refusal.

        ``next_actions`` are the evidence-gathering actions of :meth:`next_actions`;
        ``neighbors`` lists one alternative per route family that could repair the failure
        (``identified_by:<family>``, the evidence premise it changes and the action that
        starts it); ``unmet_obligations`` lists every owed obligation with its proof step as
        ``required_check``. An assumption obligation is ``uncheckable`` with status
        ``untestable``: no study establishes it. Advice only; nothing is executed.
        """
        report = self._stage.capability_report()
        return {key: tuple(dict(item) for item in report[key]) for key in report}


@dataclass(frozen=True, slots=True)
class UnmatchedObligation:
    """An obligation no study can address, with why (an assumption, for example)."""

    id: str
    kind: ObligationKind
    route: str
    reason: str


@dataclass(frozen=True, slots=True)
class StudyTemplate:
    """An unfinished study: everything an obligation determines, nothing it does not.

    The kind, population, hard interventions, measured variables, joint measurement and
    expected evidence come from the obligations (``addresses`` lists them). Sample size,
    recruitment, timing, unit rules, cost and feasibility are facts about a real study
    that only the caller knows: a template :meth:`validate` refuses and only
    :meth:`complete` turns it into a :class:`StudyCandidate`, with every declaration
    required and none defaulted.
    """

    label: str
    kind: str
    population: str
    interventions: tuple[str, ...]
    measured: tuple[str, ...]
    joint: bool
    evidence: tuple[ExpectedEvidence, ...]
    addresses: tuple[str, ...]
    routes: tuple[str, ...]
    minimum_sample_size: int | None

    def validate(self) -> None:
        """Always refuses ``study_candidate.declarations_required``: a template is not a study."""
        raise StudyCandidateRefusal(
            f"template {self.label!r} has no sample size, recruitment, timing, unit rules, "
            "cost or feasibility; those are caller declarations and are never defaulted",
            reason_code="design_signal_invalid",
            detail="study_candidate.declarations_required",
        )

    def complete(
        self,
        *,
        sample_size: int,
        recruitment: str,
        timing: str,
        unit: str,
        cost: int,
        cost_unit: str,
        sample_budget: int,
        feasible: bool,
        label: str | None = None,
        cluster: str | None = None,
        whole_cluster_sampling: bool = False,
        notes: Sequence[str] = (),
        provider: str | None = None,
    ) -> StudyCandidate:
        """The :class:`StudyCandidate` this template becomes under the caller's declarations."""
        if not isinstance(feasible, bool):
            raise CausalTypeError("feasible must be declared as a bool")
        if self.minimum_sample_size is not None and _count(sample_size, "sample_size") < (
            self.minimum_sample_size
        ):
            raise StudyCandidateRefusal(
                f"template {self.label!r} addresses an obligation that needs at least "
                f"{self.minimum_sample_size} additional rows; the declared sample size is "
                f"{sample_size}",
                reason_code="design_signal_invalid",
                detail="study_candidate.sample_below_obligation",
            )
        return StudyCandidate(
            label=self.label if label is None else label,
            kind=self.kind,
            population=self.population,
            interventions=self.interventions,
            measured=self.measured,
            joint=self.joint,
            sample_size=sample_size,
            recruitment=recruitment,
            timing=timing,
            unit=unit,
            cost=cost,
            cost_unit=cost_unit,
            cluster=cluster,
            whole_cluster_sampling=whole_cluster_sampling,
            sample_budget=sample_budget,
            feasible=feasible,
            notes=notes,
            evidence=self.evidence,
            provider=provider,
        )

    @classmethod
    def _from_native(cls, value: Mapping[str, Any]) -> StudyTemplate:
        return cls(
            label=value["label"],
            kind=value["kind"],
            population=value["population"],
            interventions=tuple(value["interventions"]),
            measured=tuple(value["measured"]),
            joint=bool(value["joint"]),
            evidence=tuple(
                ExpectedEvidence(
                    e["population"],
                    e["measured"],
                    e["interventions"],
                    e["levels"],
                    e["conditioned_on"],
                    e["joint"],
                )
                for e in value["evidence"]
            ),
            addresses=tuple(value["addresses"]),
            routes=tuple(value["routes"]),
            minimum_sample_size=value["minimum_sample_size"],
        )


@dataclass(frozen=True, slots=True)
class CandidateGeneration:
    """The unfinished studies a failed identification's obligations determine."""

    templates: tuple[StudyTemplate, ...]
    unmatched: tuple[UnmatchedObligation, ...]


def generate_candidates(contract: FailureContract) -> CandidateGeneration:
    """Restate the obligations of a failed identification as unfinished study candidates.

    An experiment is proposed for each experiment an obligation asks for, an observation
    for each joint, conditional or marginal law, and, on the instrument route, an instrument
    study and a natural experiment; the repeated-measurement alternative yields a panel. An
    assumption (an instrument's validity, the time invariance a panel relies on) is never
    addressed by a study and is returned in ``unmatched`` with its reason. Completing a
    template with :meth:`StudyTemplate.complete` is the caller's declaration of cost,
    timing, recruitment, unit rules and sample size.
    """
    if not isinstance(contract, FailureContract):
        raise CausalTypeError("generate_candidates requires a FailureContract")
    return _generation_of(contract._stage.templates())


def _generation_of(wire: Mapping[str, Any]) -> CandidateGeneration:
    """The candidate generation Rust wrote as ``wire`` (templates and unmatched obligations)."""
    return CandidateGeneration(
        templates=tuple(StudyTemplate._from_native(t) for t in wire["templates"]),
        unmatched=tuple(
            UnmatchedObligation(u["id"], u["kind"], u["route"], u["reason"])
            for u in wire["unmatched"]
        ),
    )


def candidate_templates(contract: FailureContract) -> tuple[StudyTemplate, ...]:
    """The templates of :func:`generate_candidates`."""
    return generate_candidates(contract).templates


Contract = (
    TransportContract
    | BackdoorContract
    | ZTransportContract
    | FrontDoorContract
    | InstrumentContract
    | HedgeContract
    | SensitivityContract
    | FailureContract
)
_CONTRACTS = (
    TransportContract,
    BackdoorContract,
    ZTransportContract,
    FrontDoorContract,
    InstrumentContract,
    HedgeContract,
    SensitivityContract,
    FailureContract,
)


def _contract_of(
    failure: Contract | FailedIdentification,
    population: str | None,
    observed: Sequence[str] | None,
) -> Contract:
    if isinstance(failure, _CONTRACTS):
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
        "obligations() requires a repair contract (TransportContract, BackdoorContract, "
        "ZTransportContract, FrontDoorContract, InstrumentContract, HedgeContract or "
        "SensitivityContract) or a failed identification"
    )


def obligations(
    contract_failure: Contract | FailedIdentification,
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
    classification: Classification
    cost: int
    cost_unit: str | None
    sample_budget: int
    reasons: tuple[str, ...]
    addressed: tuple[str, ...]
    unmet: tuple[str, ...]
    derivation: Derivation | None

    def __repr__(self) -> str:
        return f"<CandidateOutcome {self.classification} {list(self.labels)!r}>"

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form of this row."""
        return {
            "candidates": list(self.candidates),
            "labels": list(self.labels),
            "classification": self.classification,
            "cost": self.cost,
            "cost_unit": self.cost_unit,
            "sample_budget": self.sample_budget,
            "reasons": list(self.reasons),
            "addressed": list(self.addressed),
            "unmet": list(self.unmet),
            "derivation": None
            if self.derivation is None
            else {
                "checker": self.derivation.checker,
                "steps": list(self.derivation.steps),
                "verified": self.derivation.verified,
            },
        }

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
    stop: SearchStop | None


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
            stop=cast("SearchStop | None", receipt["stop"]),
        )

    @property
    def outcome(self) -> RepairOutcome:
        """``repaired``, ``none_certified`` or ``exhausted``."""
        return cast(RepairOutcome, str(self._report["outcome"]))

    @property
    def family(self) -> RepairFamily:
        """The family whose theorem checker decided (``transport``, ``backdoor`` or ``z_transport``)."""
        return cast(RepairFamily, str(self._report["family"]))

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

    def classification(self, *labels: str) -> Classification:
        """The classification of the subset made of the candidates named ``labels``."""
        wanted = frozenset(labels)
        for row in self.table:
            if frozenset(row.labels) == wanted and len(row.labels) == len(labels):
                return row.classification
        raise CausalValueError(f"no outcome for the subset {sorted(wanted)}")

    @artifact_export
    def export(self) -> ExportedArtifact:
        """A portable ``repair_search_receipt_v1`` artifact; see :func:`consume`."""
        return artifact_from_receipt(self, bytes(_call(self._stage.export)))

    def explain(self) -> str:
        """What the search found, what it did not, and why it is not an estimate."""
        if self.outcome == "repaired" and self.best is not None:
            labels = ", ".join(self.best.labels) or "no candidates"
            text = (
                f"{len(self.ranked)} of {len(self.table)} candidate subset(s) would make the "
                f"{self.family} contract identify if the studies deliver exactly their declared "
                f"evidence; best: {labels}"
            )
            if self.best.cost_unit is not None:
                text += f" (cost {self.best.cost} {self.best.cost_unit})"
            text += "."
        elif self.outcome == "exhausted":
            text = (
                f"The search for a {self.family} repair stopped on a budget "
                f"({self.receipt.stop}) with {self.receipt.unevaluated_total} subset(s) "
                "unexamined; exhaustion is not an impossibility claim."
            )
        else:
            text = (
                f"None of {len(self.table)} candidate subset(s) was certified to repair the "
                f"{self.family} contract; that does not show no repair exists."
            )
        if self.reason_code is not None:
            text += f" Reason: {self.reason_code} ({self.detail})."
        return (
            text + " A repair is a hypothetical identification, not an estimate: "
            f"inference claim is {self.inference_claim!r}, and no proposed study "
            "establishes an assumption."
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary of the outcome, ranked subsets and search receipt."""
        return {
            "outcome": self.outcome,
            "family": self.family,
            "contract_id": self.contract_id,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "inference_claim": self.inference_claim,
            "obligations": [o.to_dict() for o in self.obligations],
            "table": [row.to_dict() for row in self.table],
            "ranked": [row.to_dict() for row in self.ranked],
            "receipt": {
                "operations_limit": self.receipt.operations_limit,
                "depth_limit": self.receipt.depth_limit,
                "memory_limit_bytes": self.receipt.memory_limit_bytes,
                "operations_consumed": self.receipt.operations_consumed,
                "depth_reached": self.receipt.depth_reached,
                "explored": list(self.receipt.explored),
                "unevaluated": list(self.receipt.unevaluated),
                "unevaluated_total": self.receipt.unevaluated_total,
                "dominated_skipped": self.receipt.dominated_skipped,
                "beyond_declared_depth": self.receipt.beyond_declared_depth,
                "stop": self.receipt.stop,
            },
        }

    def __repr__(self) -> str:
        return (
            f"<RepairResult {self.family} outcome={self.outcome} "
            f"subsets={len(self.table)} sufficient={len(self.ranked)}>"
        )


def repair(
    contract: Contract,
    candidates: Sequence[StudyCandidate],
    objective: Literal["minimize_cost", "minimize_sample_budget"] = "minimize_cost",
    limits: RepairLimits | None = None,
    *,
    cancel: CancelToken | None = None,
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
    if isinstance(contract, FailureContract):
        raise CausalTypeError(
            "a FailureContract spans several theorem routes; plan it with "
            "antecedent.design.evidence_plan, which verifies every route"
        )
    if not isinstance(contract, _CONTRACTS):
        raise CausalTypeError(
            "repair requires a repair contract: TransportContract, BackdoorContract, "
            "ZTransportContract, FrontDoorContract, InstrumentContract, HedgeContract or "
            "SensitivityContract"
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
        max_operations=_count(limits.max_operations, "max_operations"),
        max_depth=_count(limits.max_depth, "max_depth"),
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
    cancel: CancelToken | None = None,
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
    artifact = container_bytes(artifact)
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    stage = _call(
        _native.consume_repair_artifact,
        artifact,
        max_operations=_count(max_operations, "max_operations"),
        max_depth=_count(max_depth, "max_depth"),
        memory_bytes=memory_bytes,
        cancel=cancel,
    )
    return RepairResult(stage)


__all__ = [
    "BackdoorContract",
    "CandidateGeneration",
    "CandidateOutcome",
    "Derivation",
    "EvidenceObligation",
    "ExpectedEvidence",
    "FailureContract",
    "FrontDoorContract",
    "HedgeContract",
    "InstrumentContract",
    "RepairArtifactRefusal",
    "RepairBudgetRefusal",
    "RepairFamily",
    "RepairLimits",
    "RepairOutcome",
    "RepairRefusal",
    "RepairResult",
    "SearchReceipt",
    "SearchStop",
    "SensitivityContract",
    "SensitivityFinding",
    "SensitivityMethod",
    "StudyCandidate",
    "StudyCandidateRefusal",
    "StudyTemplate",
    "UnmatchedObligation",
    "TransportContract",
    "ZTransportContract",
    "UnresolvedAssumption",
    "candidate_templates",
    "consume",
    "generate_candidates",
    "obligations",
    "repair",
]
