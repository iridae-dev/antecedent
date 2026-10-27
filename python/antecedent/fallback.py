"""The opt-in descriptive fallback: what can be said, and what evidence would help, when the
causal question is refused for want of identification.

A causal query the declared structure does not identify ends in a refusal
(:class:`~antecedent.errors.EffectNotIdentified`). That refusal is the answer to the causal
question and stays so. ``analyze(..., on_refusal="describe")`` adds two things *beside* it, only
when asked for, and keeps the causal refusal as the primary outcome::

    outcome = antecedent.analyze(data, graph=bow, query=AverageEffect("t", "y"),
                                 on_refusal="describe")
    outcome.refusal           # the causal refusal: code, remedy and the original exception
    outcome.associational     # an Outcome: a labelled-noncausal Association, or why there is none
    outcome.evidence_plan     # obligations, next actions and (with candidates) a ranked plan
    described(outcome)        # the three read together, with an explicit report and host record

* ``associational`` is the compiled associational route
  (:class:`~antecedent.descriptive.Association`) over the same outcome, treatment and
  adjustment variables, chosen by the declared rule of :func:`fallback_roles`, with the
  refusal's reason code retained in its identification layer. It describes the observed law.
  It is not the effect of an intervention, it is not identified, and it ranks no action
  (``result.answer.kind == "associational"``).
* ``evidence_plan`` (:class:`EvidenceNeeds`) is the evidence the failure determines: the
  verified evidence obligations, the evidence-gathering next actions and the unfinished study
  templates. Declare candidate studies with ``evidence_candidates=`` and it also holds the
  ranked :class:`~antecedent.design.EvidencePlan` of the options that would close the gap.

Each of the two parts is either present or a :class:`FallbackUnavailable` that says why not (no
association form exists for the query kind; a variable is not measured; the declared graph is
not one the identifiers read; the graph identifies the effect, so the refusal came from the
estimator). A query that succeeds returns the ordinary result and never an
:class:`AnalysisWithFallback`. A search that stopped at a budget, a cancellation, a resource
limit, an invalid argument and every data or estimation error are not identification-class
refusals and are raised exactly as without the option.

The default, ``on_refusal="refuse"``, is unchanged: the refusal is the outcome.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from . import _native
from .errors import (
    CausalCancelledError,
    CausalError,
    CausalResourceError,
    EffectNotIdentified,
)

if TYPE_CHECKING:
    from .design import EvidencePlan
    from .graph import Admg, Dag
    from .repair import (
        CandidateGeneration,
        EvidenceObligation,
        FailureContract,
        StudyCandidate,
    )
    from .results import AnalysisResult

__all__ = [
    "IDENTIFICATION_REFUSAL_CODES",
    "NONCAUSAL_NOTICE",
    "AnalysisWithFallback",
    "EvidenceNeeds",
    "FallbackRoles",
    "FallbackUnavailable",
    "described",
    "fallback_declaration",
    "fallback_roles",
    "from_native",
    "is_identification_refusal",
]

#: Reason codes of the identification-class refusals the fallback converts (Rust owns the list).
IDENTIFICATION_REFUSAL_CODES: tuple[str, ...] = tuple(_native.identification_refusal_codes())

#: What the descriptive part is, in the one sentence every report repeats.
NONCAUSAL_NOTICE = (
    "The associational result describes the observed law of the columns. It is not an answer "
    "to the causal question: it is not the effect of an intervention, it is not identified, "
    "and it ranks no action."
)

_POPULATION = "study"


def is_identification_refusal(error: BaseException) -> bool:
    """Whether ``error`` is an identification-class refusal the fallback converts.

    The declared structure does not identify the question (``effect_not_identified``, or a
    cross-world query it does not identify) and the search finished. A search a budget stopped
    (``search_capped``), a cancellation, a resource limit, an invalid argument and every data
    or estimation error are not.
    """
    if isinstance(error, (CausalCancelledError, CausalResourceError)):
        return False
    if isinstance(error, EffectNotIdentified) and getattr(error, "search_capped", False):
        return False
    return getattr(error, "reason_code", None) in IDENTIFICATION_REFUSAL_CODES


@dataclass(frozen=True, slots=True)
class FallbackUnavailable:
    """A part of a fallback that is not provided, with the typed reason it is not.

    ``part`` is ``"associational"`` or ``"evidence_plan"``; ``code`` is a registered
    runtime-refusal code and ``detail`` a stable ``fallback.*`` name (or the detail of the
    refusal behind it); ``message`` says what is missing and the next attempt.
    """

    part: Literal["associational", "evidence_plan"]
    code: str
    detail: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        """The host record of an unavailable part (the schema's ``fallback_unavailable``)."""
        return {
            "available": False,
            "part": self.part,
            "reason_code": self.code,
            "detail": self.detail,
            "message": self.message,
        }

    def __str__(self) -> str:
        return f"{self.detail}: {self.message}"


@dataclass(frozen=True, slots=True)
class FallbackRoles:
    """The roles the declared selection rule reads off the refused query.

    ``adjustment`` holds the measured parents of the treatment in the declared directed or
    bidirected graph together with the query's declared effect modifier, in column order
    (``rule`` names the rule; with a graph of another class the declared modifiers alone).
    """

    outcome: str
    treatment: str
    adjustment: tuple[str, ...]
    rule: str


def _roles_of(query: Any) -> tuple[str, str, tuple[str, ...]] | None:
    from .query import AverageEffect, ConditionalEffect

    if isinstance(query, AverageEffect):
        return query.treatment, query.outcome, ()
    if isinstance(query, ConditionalEffect):
        return query.treatment, query.outcome, (query.modifier,)
    return None


def _no_form(part: Literal["associational", "evidence_plan"], query: Any) -> FallbackUnavailable:
    if part == "associational":
        detail = "the description relates one outcome to one treatment, and this query has no such reading"
        next_attempt = "state the associational question directly with descriptive.Association"
    else:
        detail = (
            "evidence obligations are read off the identification of one treatment and one "
            "outcome, and this query has neither"
        )
        next_attempt = "ask the average effect of the treatment on the outcome"
    return FallbackUnavailable(
        part,
        "route_not_supported",
        "fallback.no_association_form",
        f"no association form exists for a {type(query).__name__} query: {detail}. "
        f"Next attempt: {next_attempt}",
    )


def _unmeasured(part: Literal["associational", "evidence_plan"], what: str) -> FallbackUnavailable:
    return FallbackUnavailable(
        part,
        "invalid_argument",
        "fallback.variable_not_measured",
        f"{what} is not a measured column of the data, so there is nothing to describe or to "
        "plan over. Next attempt: measure it, or state the query over the measured columns",
    )


def _directed_graph(graph: Any, columns: Sequence[str]) -> Dag | Admg | None:
    """The declared graph when it is a directed or bidirected one over the data's columns."""
    from .graph import Admg, Dag

    if isinstance(graph, (Dag, Admg)):
        return graph
    if (
        isinstance(graph, Sequence)
        and not isinstance(graph, (str, bytes))
        and all(
            isinstance(edge, tuple)
            and len(edge) == 2
            and all(isinstance(end, str) and end in columns for end in edge)
            for edge in graph
        )
    ):
        try:
            return Dag.from_edges(list(columns), [(str(a), str(b)) for a, b in graph])
        except (CausalError, ValueError, TypeError):
            return None
    return None


def fallback_roles(
    query: Any, graph: Any, columns: Sequence[str]
) -> FallbackRoles | FallbackUnavailable:
    """The roles the declared selection rule reads off a refused query.

    An average or conditional effect has an association form: the outcome against the
    treatment, holding fixed the measured parents of the treatment (the covariates the declared
    graph says move it) and the declared effect modifier. Every other query kind has none and
    yields a :class:`FallbackUnavailable`, as does a variable that is not a data column.
    """
    roles = _roles_of(query)
    if roles is None:
        return _no_form("associational", query)
    treatment, outcome, modifiers = roles
    columns = tuple(columns)
    for what, name in (
        ("the treatment", treatment),
        ("the outcome", outcome),
        *(("an effect modifier", modifier) for modifier in modifiers),
    ):
        if name not in columns:
            return _unmeasured("associational", what)
    adjustment = set(modifiers)
    rule = "declared_modifiers_only"
    directed = _directed_graph(graph, columns)
    if directed is not None:
        try:
            adjustment.update(p for p in directed.parents(treatment) if p in columns)
            rule = "measured_parents_of_treatment_and_declared_modifiers"
        except (CausalError, ValueError, KeyError):
            pass
    adjustment.discard(treatment)
    adjustment.discard(outcome)
    return FallbackRoles(
        outcome=outcome,
        treatment=treatment,
        adjustment=tuple(sorted(adjustment, key=columns.index)),
        rule=rule,
    )


@dataclass(frozen=True, slots=True)
class EvidenceNeeds:
    """The evidence that would resolve the identification failure.

    ``obligations`` are the verified evidence obligations, each with its proof step (an
    assumption obligation is listed and satisfiable by no study); ``next_actions`` the
    evidence-gathering actions they imply (``measure_variable:<name>``,
    ``run_experiment:<name>``, ``find_instrument``, ``collect_panel``); ``templates`` the
    unfinished studies the obligations determine. ``plan`` is the ranked plan over the studies
    the caller declared with ``evidence_candidates=``: ``None`` when none were declared (not
    requested), the :class:`~antecedent.design.EvidencePlan` when ranked, or the
    :class:`FallbackUnavailable` reason it refused. ``contract`` is the frozen failure, to be
    planned over again with other candidates or a value declaration.
    """

    contract: FailureContract
    obligations: tuple[EvidenceObligation, ...]
    next_actions: tuple[Mapping[str, Any], ...]
    templates: CandidateGeneration
    plan: EvidencePlan | FallbackUnavailable | None

    def to_dict(self) -> dict[str, Any]:
        """The host record of the evidence part (the schema's ``evidence_plan``)."""
        plan = self.plan
        if plan is None:
            ranked: dict[str, Any] = {"available": False, "requested": False}
        elif isinstance(plan, FallbackUnavailable):
            ranked = plan.to_dict()
        else:
            ranked = {
                "available": True,
                "requested": True,
                "outcome": plan.outcome,
                "options": len(plan.options),
            }
        return {
            "available": True,
            "obligations": [
                {
                    "id": o.id,
                    "kind": o.kind,
                    "scope": o.scope,
                    "variables": list(o.variables),
                    "reason": o.reason,
                    "family": o.family,
                    "proof_step": o.proof_step,
                    "satisfiable_by_study": o.satisfiable_by_study,
                }
                for o in self.obligations
            ],
            "next_actions": [
                {"id": str(a["id"]), "description": str(a["description"])}
                for a in self.next_actions
            ],
            "ranked_plan": ranked,
        }


def _figure(value: float) -> float | str:
    """A reported number as the renderer reads it: finite as is, else its text."""
    number = float(value)
    if math.isfinite(number):
        return number
    return "nan" if math.isnan(number) else ("inf" if number > 0 else "-inf")


def _claim_of(result: Any) -> dict[str, Any]:
    """The contract claim of an associational result, or the one its kind states."""
    try:
        contract = result.inspect().contract
    except (CausalError, AttributeError, ValueError):
        contract = None
    claim = contract.get("claim") if isinstance(contract, Mapping) else None
    if isinstance(claim, Mapping) and claim.get("kind") is not None:
        return dict(claim)
    return {
        "kind": "associational",
        "domains": {
            "identification": "not_claimed",
            "support": "unknown",
            "evaluated": "evaluated",
        },
    }


@dataclass(frozen=True, slots=True, repr=False)
class AnalysisWithFallback:
    """An identification-class refusal, unchanged, beside a noncausal description and the plan.

    ``refusal`` is the original refusal object: same class, reason code, remedy and ``report``
    it would have had if raised. ``associational`` is the compiled
    :class:`~antecedent.descriptive.Association` result (``answer.kind == "associational"``)
    or the :class:`FallbackUnavailable` reason there is none. ``evidence_plan`` is the
    :class:`EvidenceNeeds` or the reason there are none.

    The associational result is **not an answer to the causal question**: it describes the
    observed law, is not the effect of an intervention, is not identified, and ranks no action.
    The causal question stays refused.
    """

    refusal: CausalError
    associational: AnalysisResult | FallbackUnavailable
    evidence_plan: EvidenceNeeds | FallbackUnavailable
    roles: FallbackRoles | None = None

    @property
    def noncausal(self) -> bool:
        """Always ``True``: nothing here is a causal answer."""
        return True

    @property
    def answers_causal_question(self) -> bool:
        """Always ``False``: the refusal is the answer to the causal question."""
        return False

    @property
    def ranks_actions(self) -> bool:
        """Always ``False``: an association ranks no action."""
        return False

    @property
    def notice(self) -> str:
        """The sentence that states what the associational result is not."""
        return NONCAUSAL_NOTICE

    def explain(self) -> str:
        """Plain statement of what was returned and what it is not."""
        code = getattr(self.refusal, "reason_code", None) or type(self.refusal).__name__
        text = f"The causal question was refused ({code}): {self.refusal}. {NONCAUSAL_NOTICE}"
        if isinstance(self.associational, FallbackUnavailable):
            text += f" No associational description: {self.associational}"
        else:
            section = getattr(self.associational, "association", None)
            analysis = getattr(section, "analysis", "association")
            held = len(self.roles.adjustment) if self.roles is not None else 0
            rule = self.roles.rule if self.roles is not None else "declared"
            text += (
                f" Associational description: {analysis} of the outcome on the treatment "
                f"holding {held} column(s) fixed (rule {rule})."
            )
        needs = self.evidence_plan
        if isinstance(needs, FallbackUnavailable):
            return text + f" No evidence plan: {needs}"
        text += (
            f" Evidence that would resolve the refusal: {len(needs.obligations)} obligation(s), "
            f"{len(needs.next_actions)} next action(s)."
        )
        if needs.plan is None:
            text += " Declare candidate studies to rank the options."
        elif isinstance(needs.plan, FallbackUnavailable):
            text += f" No ranked plan: {needs.plan}"
        else:
            text += f" Ranked plan over the declared studies: {len(needs.plan.options)} option(s)."
        return text

    def report(self, format: Literal["markdown", "html"] = "markdown") -> str:
        """The report of the fallback: the refusal first, then the two parts.

        ``format`` is ``"markdown"`` or ``"html"`` (one self-contained document with inline style
        and no script). Rust writes the document from the values this object holds, so the
        wording, escaping and layout are the same as for a fallback produced by the Rust
        facade. Nothing is executed and no figure is computed: every number is read from the
        fallback Rust produced, or, for an object assembled by hand, from the parts it holds.
        """
        if format not in ("markdown", "html"):
            from .errors import CausalValueError

            raise CausalValueError("report supports format='markdown' or format='html'")
        stage = getattr(self.refusal, "fallback", None)
        if stage is not None:
            # The refusal came with the fallback Rust ran on the refused study: Rust reads it
            # and writes the document (``report::render_fallback_report``).
            return cast("str", stage.report(format))
        return _native.render_fallback_report(json.dumps(self._report_parts()), format)

    def _report_parts(self) -> dict[str, Any]:
        """The plain values of the report, in the shape the Rust renderer reads."""
        refusal = self.refusal
        associational: dict[str, Any]
        if isinstance(self.associational, FallbackUnavailable):
            associational = {"unavailable": str(self.associational)}
        else:
            claim = _claim_of(self.associational)
            domains = claim.get("domains")
            identification = domains.get("identification") if isinstance(domains, Mapping) else None
            section = self.associational.association
            figures = None
            if section is not None:
                figures = {
                    "analysis": section.analysis,
                    "rows": section.rows,
                    "uncertainty": section.uncertainty,
                    "calibration": section.calibration,
                    "refused_after": section.refused_after,
                    "terms": [[term.label, _figure(term.estimate)] for term in section.terms],
                }
            associational = {
                "described": {
                    "claim_kind": str(claim.get("kind")),
                    "identification": str(identification),
                    "rule": self.roles.rule if self.roles is not None else "declared",
                    "figures": figures,
                }
            }
        needs = self.evidence_plan
        evidence: dict[str, Any]
        if isinstance(needs, FallbackUnavailable):
            evidence = {"unavailable": str(needs)}
        else:
            ranking: Any
            if needs.plan is None:
                ranking = "not_requested"
            elif isinstance(needs.plan, FallbackUnavailable):
                ranking = {"unavailable": str(needs.plan)}
            else:
                ranking = {
                    "ranked": [
                        {
                            "rank": str(option.rank),
                            "route": option.route,
                            "studies": list(option.labels),
                            "gate": option.gate,
                            "cost": f"{option.cost} {option.cost_unit}",
                        }
                        for option in needs.plan.options
                    ]
                }
            evidence = {
                "needs": {
                    "obligations": [
                        {
                            "id": o.id,
                            "kind": o.kind,
                            "route": o.family,
                            "proof_step": o.proof_step,
                            "satisfiable_by_study": bool(o.satisfiable_by_study),
                        }
                        for o in needs.obligations
                    ],
                    "next_actions": [
                        [str(a["id"]), str(a["description"])] for a in needs.next_actions
                    ],
                    "ranking": ranking,
                }
            }
        return {
            "reason_code": getattr(refusal, "reason_code", None),
            "message": str(refusal),
            "remedy": getattr(refusal, "remedy", None),
            "associational": associational,
            "evidence": evidence,
        }

    def to_dict(self) -> dict[str, Any]:
        """The host record of the fallback (the schema's ``fallback_response``).

        The causal question stays refused: ``status`` and ``causal_status`` are
        ``"unsupported"`` and ``refusal`` is the unchanged refusal record; beside them
        ``associational_status`` shows the status of the noncausal claim.
        """
        from . import host

        return host.fallback_response(self)

    def _associational_record(self) -> tuple[str, dict[str, Any]]:
        if isinstance(self.associational, FallbackUnavailable):
            return "unavailable", self.associational.to_dict()
        from . import host

        claim = host.claim_record(_claim_of(self.associational))
        return str(claim["status"]), {
            "available": True,
            "noncausal": True,
            "ranks_actions": False,
            "rule": self.roles.rule if self.roles is not None else "declared",
            "claim": claim,
        }

    def __repr__(self) -> str:
        code = getattr(self.refusal, "reason_code", None) or type(self.refusal).__name__
        associational = (
            f"unavailable({self.associational.detail})"
            if isinstance(self.associational, FallbackUnavailable)
            else "associational"
        )
        evidence = (
            f"unavailable({self.evidence_plan.detail})"
            if isinstance(self.evidence_plan, FallbackUnavailable)
            else f"{len(self.evidence_plan.obligations)} obligations"
        )
        return (
            f"<AnalysisWithFallback refused={code} associational={associational} "
            f"evidence_plan={evidence}; not an answer to the causal question>"
        )


def fallback_declaration(candidates: Sequence[StudyCandidate] | None) -> dict[str, Any]:
    """The ``fallback`` prepare option: the declared studies, ranked by Rust when the refusal comes.

    Rust runs the fallback on the refused study itself (``Study::analyze_with_fallback``); this
    only spells the caller's candidate studies and the bounds of their plan, the defaults of
    :func:`antecedent.design.evidence_plan`.
    """
    from .errors import CausalTypeError
    from .repair import RepairLimits, StudyCandidate

    limits = RepairLimits()
    wire: str | None = None
    if candidates is not None:
        declared = tuple(candidates)
        if any(not isinstance(c, StudyCandidate) for c in declared):
            raise CausalTypeError("candidates must be StudyCandidate values")
        wire = json.dumps([c._wire() for c in declared])
    return {
        "candidates": wire,
        "objective": "minimize_cost",
        "max_operations": int(limits.max_operations),
        "max_depth": int(limits.max_depth),
        "memory_bytes": limits.memory_bytes,
    }


def _unavailable(part: tuple[str, str, str, str]) -> FallbackUnavailable:
    return FallbackUnavailable(cast("Any", part[0]), part[1], part[2], part[3])


def from_native(
    refusal: CausalError,
    stage: Any,
    *,
    query: Any,
    graph: Any,
    columns: Sequence[str],
    seed: int = 1,
    threads: int | None = None,
    controls: Any = None,
) -> AnalysisWithFallback:
    """Read the fallback Rust ran on the refused study as an :class:`AnalysisWithFallback`.

    ``stage`` is the ``fallback`` attribute the native prepare attached to the refusal: Rust's
    ``AnalysisWithFallback``, produced by ``Study::analyze_with_fallback``. The associational
    description is its compiled study and result, the evidence needs its obligations, next
    actions, templates and ranked plan; nothing is recomputed here.
    """
    from .descriptive import Association
    from .estimation import _Controls, _PreparedEngine

    roles_wire = stage.roles()
    roles = (
        None
        if roles_wire is None
        else FallbackRoles(
            outcome=roles_wire[0],
            treatment=roles_wire[1],
            adjustment=tuple(roles_wire[2]),
            rule=roles_wire[3],
        )
    )
    associational: AnalysisResult | FallbackUnavailable
    missing = stage.associational_unavailable()
    if missing is not None:
        associational = _unavailable(missing)
    else:
        native, raw = stage.associational()
        assert roles is not None
        reason = getattr(refusal, "reason_code", None) or "effect_not_identified"
        description = Association(
            roles.outcome,
            (roles.treatment,),
            conditioning=roles.adjustment,
            refused_after=str(reason),
        )
        engine: _PreparedEngine[Any] = _PreparedEngine(
            native,
            kind="average",
            query=description,
            seed=seed,
            threads=threads,
            controls=controls if isinstance(controls, _Controls) else None,
        )
        associational = cast("AnalysisResult", engine._wrap(raw))
    return AnalysisWithFallback(
        refusal=refusal,
        associational=associational,
        evidence_plan=_evidence_needs(stage, query=query, graph=graph, columns=columns),
        roles=roles,
    )


def _evidence_needs(
    stage: Any, *, query: Any, graph: Any, columns: Sequence[str]
) -> EvidenceNeeds | FallbackUnavailable:
    """The evidence part of Rust's fallback, with the frozen failure to plan over again."""
    from . import design, repair
    from .accepted_graph import AcceptedGraph
    from .errors import CausalStateError

    missing = stage.evidence_unavailable()
    if missing is not None:
        return _unavailable(missing)
    declared = graph.graph if isinstance(graph, AcceptedGraph) else graph
    directed = _directed_graph(declared, columns)
    roles = _roles_of(query)
    if directed is None or roles is None:
        raise CausalStateError("the evidence needs were produced for a study this call cannot read")
    treatment, outcome, _ = roles
    nodes = set(directed.nodes())
    contract = repair.FailureContract(
        graph=directed,
        treatment=treatment,
        outcome=outcome,
        population=_POPULATION,
        observed=[name for name in columns if name in nodes],
    )
    kind = stage.ranking_kind()
    plan: EvidencePlan | FallbackUnavailable | None
    if kind == "ranked":
        plan = design.EvidencePlan(stage.ranked_plan())
    elif kind == "unavailable":
        plan = _unavailable(stage.ranking_unavailable())
    else:
        plan = None
    return EvidenceNeeds(
        contract=contract,
        obligations=tuple(repair.EvidenceObligation._from_native(o) for o in stage.obligations()),
        next_actions=tuple(dict(action) for action in stage.next_actions()),
        templates=repair._generation_of(stage.templates()),
        plan=plan,
    )


def described(outcome: Any) -> AnalysisWithFallback:
    """The combined refusal-and-description view of a ``describe`` outcome.

    ``ant.analyze(..., on_refusal="describe")`` keeps the causal refusal as the primary outcome
    and attaches ``outcome.associational`` and ``outcome.evidence_plan``. This reads them
    together as an :class:`AnalysisWithFallback`: the unchanged refusal object, the noncausal
    association (or the typed reason there is none), the evidence plan, and the explicit report
    and host record. It executes nothing and re-reads no data; a refusal that carries no
    description is an invalid argument.
    """
    from .errors import CausalTypeError, CausalValueError
    from .outcomes import Outcome

    if not isinstance(outcome, Outcome):
        raise CausalTypeError("described needs the Outcome of an on_refusal='describe' analysis")
    if outcome.refusal is None or outcome.associational is None or outcome.evidence_plan is None:
        raise CausalValueError(
            "this outcome carries no descriptive fallback; request it with on_refusal='describe' "
            "on a question refused for want of identification"
        )
    error = outcome.refusal._error
    associational: AnalysisResult | FallbackUnavailable
    if outcome.associational.refusal is not None:
        unavailable = outcome.associational.refusal
        associational = FallbackUnavailable(
            "associational", unavailable.code, unavailable.detail or "", unavailable.message
        )
    else:
        associational = cast("AnalysisResult", outcome.associational.value)
    # The roles are Rust's: the stage the native prepare attached to the refusal read them off
    # the refused query.
    stage = getattr(error, "fallback", None)
    found = None if stage is None else stage.roles()
    roles = (
        None
        if found is None
        else FallbackRoles(
            outcome=found[0], treatment=found[1], adjustment=tuple(found[2]), rule=found[3]
        )
    )
    return AnalysisWithFallback(
        refusal=cast("CausalError", error),
        associational=associational,
        evidence_plan=outcome.evidence_plan,
        roles=roles,
    )
