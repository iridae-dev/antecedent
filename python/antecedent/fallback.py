"""The opt-in descriptive fallback: what can be said, and what evidence would help, when the
causal question is refused for want of identification.

A causal query the declared structure does not identify ends in a refusal
(:class:`~antecedent.errors.EffectNotIdentified`). That refusal is the answer to the causal
question and stays so. ``analyze(..., on_refusal="describe")`` adds two things *beside* it, only
when asked for, and returns an :class:`AnalysisWithFallback`::

    outcome = antecedent.analyze(data, graph=bow, query=AverageEffect("t", "y"),
                                 on_refusal="describe")
    outcome.refusal           # the unchanged refusal: same class, code, remedy and report
    outcome.associational     # a labelled-noncausal Association result, or why there is none
    outcome.evidence_plan     # obligations, next actions and (with candidates) a ranked plan

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

The default, ``on_refusal="raise"``, is unchanged: the refusal is raised.
"""

from __future__ import annotations

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
    from .results import Analysis

__all__ = [
    "IDENTIFICATION_REFUSAL_CODES",
    "NONCAUSAL_NOTICE",
    "AnalysisWithFallback",
    "EvidenceNeeds",
    "FallbackRoles",
    "FallbackUnavailable",
    "describe",
    "fallback_roles",
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
    associational: Analysis | FallbackUnavailable
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

    def report(self, format: Literal["markdown"] = "markdown") -> str:
        """The report of the fallback as Markdown: the refusal first, then the two parts."""
        if format != "markdown":
            from .errors import CausalValueError

            raise CausalValueError("report supports format='markdown'")
        code = getattr(self.refusal, "reason_code", None)
        lines = [
            "# Refusal with descriptive fallback",
            "",
            "## Causal question",
            "",
            "The causal query is refused. The refusal below is the answer to the causal "
            "question and is unchanged by anything else in this report.",
            "",
            "- Status: unsupported",
            f"- Reason code: {code}",
            f"- Message: {' '.join(str(self.refusal).split())}",
            f"- Remedy: {getattr(self.refusal, 'remedy', None)}",
            "",
            "## Noncausal description",
            "",
            NONCAUSAL_NOTICE,
            "",
        ]
        if isinstance(self.associational, FallbackUnavailable):
            lines += [f"The associational description is unavailable: {self.associational}", ""]
        else:
            section = self.associational.association
            lines += [
                "- Claim kind: associational",
                "- Identification: not_claimed",
                f"- Analysis: {section.analysis}" if section is not None else "- Analysis: n/a",
                f"- Refusal retained: {section.refused_after}" if section is not None else "",
                "- Ranks actions: no",
                "- Is an intervention effect: no",
                "",
            ]
        lines += ["## Evidence that would resolve the refusal", ""]
        needs = self.evidence_plan
        if isinstance(needs, FallbackUnavailable):
            lines += [f"The evidence plan is unavailable: {needs}", ""]
        else:
            lines += ["| Obligation | Kind | Route | Proof step | A study can satisfy it |"]
            lines += ["| --- | --- | --- | --- | --- |"]
            for o in needs.obligations:
                lines.append(
                    f"| {o.id} | {o.kind} | {o.family} | {o.proof_step or 'unrecorded'} | "
                    f"{'yes' if o.satisfiable_by_study else 'no'} |"
                )
            lines.append("")
            for action in needs.next_actions:
                lines.append(f"- {action['id']}: {action['description']}")
            lines.append("")
            if needs.plan is None:
                lines += ["No candidate studies were declared, so the options are not ranked.", ""]
            elif isinstance(needs.plan, FallbackUnavailable):
                lines += [f"The ranked plan is unavailable: {needs.plan}", ""]
            else:
                lines += [
                    "| Rank | Route | Studies | Gate | Cost |",
                    "| --- | --- | --- | --- | --- |",
                ]
                for option in needs.plan.options:
                    lines.append(
                        f"| {option.rank} | {option.route} | {', '.join(option.labels)} | "
                        f"{option.gate} | {option.cost} {option.cost_unit} |"
                    )
                lines.append("")
        return "\n".join(lines)

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


def _stopped(cancel: Any) -> bool:
    is_cancelled = getattr(cancel, "is_cancelled", None)
    return bool(is_cancelled()) if callable(is_cancelled) else False


def _associational(
    refusal: CausalError,
    roles: FallbackRoles,
    data: Any,
    options: Mapping[str, Any],
) -> Analysis | FallbackUnavailable:
    from .descriptive import Association
    from .estimation import PreparedAnalysis

    reason = getattr(refusal, "reason_code", None) or "effect_not_identified"
    try:
        query = Association(
            roles.outcome,
            (roles.treatment,),
            conditioning=roles.adjustment,
            refused_after=str(reason),
        )
        prepared = PreparedAnalysis.prepare(data, query=query, **options)
        result = prepared.estimate()
    except (CausalCancelledError, CausalResourceError):
        raise
    except (CausalError, ValueError, TypeError) as error:
        if _stopped(options.get("cancel")):
            raise
        code = getattr(error, "reason_code", None)
        return FallbackUnavailable(
            "associational",
            str(code) if code else "route_not_supported",
            "fallback.association_refused",
            str(error),
        )
    return cast("Analysis", result)


def _evidence(
    query: Any,
    graph: Any,
    columns: Sequence[str],
    candidates: Sequence[StudyCandidate] | None,
    cancel: Any,
) -> EvidenceNeeds | FallbackUnavailable:
    from . import design, repair

    roles = _roles_of(query)
    if roles is None:
        return _no_form("evidence_plan", query)
    treatment, outcome, _ = roles
    if treatment not in columns or outcome not in columns:
        return _unmeasured("evidence_plan", "the treatment or the outcome")
    directed = _directed_graph(graph, columns)
    if directed is None:
        return FallbackUnavailable(
            "evidence_plan",
            "route_not_supported",
            "fallback.graph_class_not_read",
            "the evidence routes (back-door, front-door, instrument, hedge) are read from a "
            "declared directed or bidirected graph, and this study declares another graph "
            "class. Next attempt: declare a Dag or an Admg",
        )
    nodes = set(directed.nodes())
    try:
        contract = repair.FailureContract(
            graph=directed,
            treatment=treatment,
            outcome=outcome,
            population=_POPULATION,
            observed=[name for name in columns if name in nodes],
        )
        obligations = repair.obligations(contract)
        actions = contract.next_actions()
        templates = repair.generate_candidates(contract)
    except (CausalCancelledError, CausalResourceError):
        raise
    except (CausalError, ValueError, TypeError) as error:
        if _stopped(cancel):
            raise
        code = getattr(error, "reason_code", None)
        detail = getattr(error, "detail", None) or "fallback.evidence_refused"
        return FallbackUnavailable(
            "evidence_plan",
            str(code) if code else "invalid_argument",
            str(detail),
            str(error),
        )
    plan: EvidencePlan | FallbackUnavailable | None = None
    if candidates is not None:
        try:
            plan = design.evidence_plan(contract, list(candidates), cancel=cancel)
        except (CausalCancelledError, CausalResourceError):
            raise
        except (CausalError, ValueError, TypeError) as error:
            if _stopped(cancel):
                raise
            code = getattr(error, "reason_code", None)
            detail = getattr(error, "detail", None) or "fallback.evidence_refused"
            plan = FallbackUnavailable(
                "evidence_plan",
                str(code) if code else "invalid_argument",
                str(detail),
                str(error),
            )
    return EvidenceNeeds(
        contract=contract,
        obligations=obligations,
        next_actions=actions,
        templates=templates,
        plan=plan,
    )


def describe(
    refusal: CausalError,
    data: Any,
    *,
    query: Any,
    graph: Any = None,
    candidates: Sequence[StudyCandidate] | None = None,
    options: Mapping[str, Any] | None = None,
) -> AnalysisWithFallback:
    """Keep ``refusal`` and add the noncausal description and the evidence plan beside it.

    ``options`` are the ``analyze`` arguments the description inherits (``inference``,
    ``seed``, ``bootstrap``, ``threads``, ``latency``, ``cancel``, ``participant``,
    ``missing``, ``sampling_design``); the graph, estimator and refutation choices belong to the
    causal route and are not passed on. Cancellation and resource limits are raised, never
    described.
    """
    from ._names import table_names

    options = {} if options is None else options
    columns = table_names(data)
    roles = fallback_roles(query, graph, columns)
    associational: Analysis | FallbackUnavailable
    if isinstance(roles, FallbackUnavailable):
        associational, kept = roles, None
    else:
        associational, kept = _associational(refusal, roles, data, options), roles
    evidence = _evidence(query, graph, columns, candidates, options.get("cancel"))
    return AnalysisWithFallback(
        refusal=refusal, associational=associational, evidence_plan=evidence, roles=kept
    )
