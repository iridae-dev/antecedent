"""Selection differences and invariances behind each transport-scenario answer (2.3.0 A1).

A transport scenario is a selection diagram: a graph plus the *selection targets*, the
variables whose generating mechanism may differ between the source and the target. The
scenario report says what each scenario answered and the range those answers span;
:func:`invariance_report` says **why**. For every scenario it reports

* the *selection differences*: the selection targets, the shared mechanisms (every other
  variable) and the directed and bidirected edges the scenario assumes;
* the *invariances the identified formula relies on*: each factor the formula takes from a
  SOURCE population, with its variables, the conditioning variables, the experimental regime
  (the ``do`` set and the catalog regime id), the district selection targets and the rule of
  the proof step that produced it. The mechanisms of a factor's variables are assumed shared
  between source and target; the checker's S-admissibility test has already established that
  no selection node reaches them. The factors taken from the TARGET population are reported
  separately, because a target law is not an invariance;
* for a scenario that is proven not transportable, the structural *obstruction* (the s-hedge
  forests, or the conditional two-model witness) instead of an invariance list.

For each extreme of the structural envelope the same report of the scenario that produced it
is attached, so "the answer ranges over 0.35 to 0.56" reads "0.35 from the scenario selecting
on ``z`` (it relies on ``P_s(y | do(x), z)`` and ``P*(z)``), 0.56 from the scenario with no
selection (it relies on ``P_s(y | do(x))``)".

The report is **derived, not stored**. It is a function of the decided scenario set (read from
each checked derivation, never from the evaluated numbers) and the estimated report, so it is
recomputed from the stage on demand and from the scenario artifact by consuming it into a
stage; no artifact byte changes and nothing here is sealed. ``identity`` is a digest of the
report's canonical text; it does not depend on the order edges or selection targets were
supplied in, nor on the scenario's name.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from . import _native
from .errors import CausalUnsupportedError

__all__ = [
    "ConditionalReduction",
    "EnvelopeExtremeInvariance",
    "InvarianceReport",
    "ObstructionWitness",
    "ScenarioInvariance",
    "ScenarioInvarianceRefusal",
    "ScenarioSetInvarianceReport",
    "SelectionDifferences",
    "SourceInvariance",
    "TargetFactor",
    "invariance_report",
]

Edge = tuple[str, str]


class ScenarioInvarianceRefusal(CausalUnsupportedError):
    """The invariance report could not be produced.

    ``detail`` is the namespaced ``scenario_invariance.*`` slot:
    ``scenario_invariance.not_estimated`` (``reason_code="not_executed"``; call ``estimate()``
    on the stage first), ``scenario_invariance.wrong_result_type`` (not a prepared transport
    scenario stage), or ``scenario_invariance.derivation_inconsistent`` /
    ``scenario_invariance.non_static_node`` for a derivation that contradicts its own formula,
    which a checked derivation cannot contain.
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        detail = str(refusal["detail"])
        message = refusal.get("message")
        super().__init__(
            f"{detail}: {message}" if message else detail,
            reason_code=refusal["code"],
            remedy=refusal.get("remedy"),
        )
        #: Namespaced ``family.slot`` detail.
        self.detail: str = detail
        #: Offending variable or scenario, when there is one.
        self.offending: str | None = refusal.get("offending")


@dataclass(frozen=True, slots=True)
class SelectionDifferences:
    """What may differ between source and target under one scenario.

    ``targets`` are the variables whose mechanism may differ, ``shared_mechanisms`` every
    other variable (assumed identical in both populations), and the edge lists the graph the
    scenario assumes. All lists are sorted by the scenario's variable order.
    """

    targets: tuple[str, ...]
    shared_mechanisms: tuple[str, ...]
    directed_edges: tuple[Edge, ...]
    bidirected_edges: tuple[Edge, ...]


@dataclass(frozen=True, slots=True)
class SourceInvariance:
    """One factor the identified formula takes from a source population.

    ``variables`` are the factor's variables, ``conditioned_on`` the conditioning variables
    (their mechanisms are NOT assumed shared: a pretreatment variable is conditioned on
    because it may differ), ``do_set`` the variables held by ``do(.)`` in the regime, and
    ``regime`` the catalog regime id the factor is read from. ``invariant_mechanisms`` are the
    mechanisms assumed identical in source and target (the factor's variables; none is a
    selection target) and ``district_selection_targets`` the selection targets inside the
    factor's bidirected district (empty when the district is selection-free). ``rule`` is the
    proof step that produced the factor (``sid.line10``, ``transport.direct`` or
    ``transport.pretreatment_standardize``).
    """

    population: str
    variables: tuple[str, ...]
    conditioned_on: tuple[str, ...]
    do_set: tuple[str, ...]
    regime: int | None
    invariant_mechanisms: tuple[str, ...]
    district_selection_targets: tuple[str, ...]
    rule: str


@dataclass(frozen=True, slots=True)
class TargetFactor:
    """One factor taken from the target population (a target law, not an invariance)."""

    variables: tuple[str, ...]
    conditioned_on: tuple[str, ...]
    regime: int | None


@dataclass(frozen=True, slots=True)
class ConditionalReduction:
    """The rule-2 reduction of a conditional question: coordinates moved into ``do(.)``."""

    moves: tuple[str, ...]
    remaining: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ObstructionWitness:
    """Why a scenario is not transportable: the obstruction's variables.

    ``kind`` is ``s_hedge``, ``conditional_two_model_witness`` or (an inspection-only
    candidate, never an impossibility claim) ``conditional_reduced_s_hedge_candidate``. The
    larger and smaller s-hedge forests are given by their nodes and edges, and
    ``selection_targets_in_larger`` lists the scenario's selection targets inside the larger
    forest. ``moves`` and ``remaining`` are the rule-2 reduction of a conditional question
    (empty otherwise).
    """

    kind: str
    larger_nodes: tuple[str, ...]
    smaller_nodes: tuple[str, ...]
    larger_directed: tuple[Edge, ...]
    larger_bidirected: tuple[Edge, ...]
    smaller_directed: tuple[Edge, ...]
    smaller_bidirected: tuple[Edge, ...]
    selection_targets_in_larger: tuple[str, ...]
    moves: tuple[str, ...]
    remaining: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class InvarianceReport:
    """Selection differences and invariances of one scenario.

    ``kind`` says how the scenario was decided:

    * ``"identified"``: ``invariances`` (the source factors the answer relies on),
      ``target_factors`` (the factors taken from the target) and ``rules`` are set, and
      ``conditional`` for a conditional question;
    * ``"obstructed"``: proven not transportable; ``obstruction`` is the witness and
      ``invariances`` / ``target_factors`` are ``None`` (there is no invariance list);
    * ``"undecided"``: neither identified nor proven non-transportable (``missing_evidence``,
      ``not_certified``, ``unevaluated``); ``obligations`` are the unmet obligations or scope
      notes and ``obstruction`` an inspection-only candidate when the conditional route found
      one.

    ``identity`` is the digest of ``canonical_text``.
    """

    status: str
    kind: Literal["identified", "obstructed", "undecided"]
    selection: SelectionDifferences
    rules: tuple[str, ...]
    invariances: tuple[SourceInvariance, ...] | None
    target_factors: tuple[TargetFactor, ...] | None
    conditional: ConditionalReduction | None
    obstruction: ObstructionWitness | None
    obligations: tuple[str, ...]
    identity: str
    canonical_text: str


@dataclass(frozen=True, slots=True)
class ScenarioInvariance:
    """One scenario's report beside the status its evaluated result reports.

    ``result_status`` can be ``support_failure`` or ``unsupported_provider`` for a scenario
    whose invariance report is still that of its identified formula.
    """

    name: str
    result_status: str
    report: InvarianceReport


@dataclass(frozen=True, slots=True)
class EnvelopeExtremeInvariance:
    """The selections and invariances that produced one envelope extreme."""

    outcome: str
    side: Literal["lower", "upper"]
    value: float
    scenario: str
    report: InvarianceReport


@dataclass(frozen=True, slots=True)
class ScenarioSetInvarianceReport:
    """Invariance reports of a whole scenario set.

    ``scenarios`` holds one report per scenario in canonical order (failed as well as
    successful); ``extremes`` both extremes of each outcome's envelope (empty when no scenario
    identified).
    """

    scenarios: tuple[ScenarioInvariance, ...]
    extremes: tuple[EnvelopeExtremeInvariance, ...]
    canonical_text: str

    def scenario(self, name: str) -> ScenarioInvariance | None:
        """The report of the scenario called ``name``, or ``None``."""
        return next((s for s in self.scenarios if s.name == name), None)

    def extreme(
        self, outcome: str, side: Literal["lower", "upper"]
    ) -> EnvelopeExtremeInvariance | None:
        """The extreme of ``outcome`` on ``side``, or ``None``."""
        return next((e for e in self.extremes if e.outcome == outcome and e.side == side), None)


def _names(items: Sequence[str]) -> tuple[str, ...]:
    return tuple(items)


def _edges(items: Sequence[Sequence[str]]) -> tuple[Edge, ...]:
    return tuple((a, b) for a, b in items)


def _witness(raw: Mapping[str, Any]) -> ObstructionWitness:
    return ObstructionWitness(
        kind=raw["kind"],
        larger_nodes=_names(raw["larger_nodes"]),
        smaller_nodes=_names(raw["smaller_nodes"]),
        larger_directed=_edges(raw["larger_directed"]),
        larger_bidirected=_edges(raw["larger_bidirected"]),
        smaller_directed=_edges(raw["smaller_directed"]),
        smaller_bidirected=_edges(raw["smaller_bidirected"]),
        selection_targets_in_larger=_names(raw["selection_targets_in_larger"]),
        moves=_names(raw["moves"]),
        remaining=_names(raw["remaining"]),
    )


def _report(raw: Mapping[str, Any]) -> InvarianceReport:
    selection = raw["selection"]
    body = raw["body"]
    kind = body["kind"]
    invariances: tuple[SourceInvariance, ...] | None = None
    target_factors: tuple[TargetFactor, ...] | None = None
    conditional: ConditionalReduction | None = None
    obstruction: ObstructionWitness | None = None
    rules: tuple[str, ...] = ()
    obligations: tuple[str, ...] = ()
    if kind == "identified":
        rules = _names(body["rules"])
        invariances = tuple(
            SourceInvariance(
                population=item["population"],
                variables=_names(item["variables"]),
                conditioned_on=_names(item["conditioned_on"]),
                do_set=_names(item["do_set"]),
                regime=item["regime"],
                invariant_mechanisms=_names(item["invariant_mechanisms"]),
                district_selection_targets=_names(item["district_selection_targets"]),
                rule=item["rule"],
            )
            for item in body["invariances"]
        )
        target_factors = tuple(
            TargetFactor(
                variables=_names(item["variables"]),
                conditioned_on=_names(item["conditioned_on"]),
                regime=item["regime"],
            )
            for item in body["target_factors"]
        )
        if body["conditional"] is not None:
            conditional = ConditionalReduction(
                moves=_names(body["conditional"]["moves"]),
                remaining=_names(body["conditional"]["remaining"]),
            )
    elif kind == "obstructed":
        obstruction = _witness(body["witness"])
    else:
        obligations = _names(body["obligations"])
        if body["candidate"] is not None:
            obstruction = _witness(body["candidate"])
    return InvarianceReport(
        status=raw["status"],
        kind=kind,
        selection=SelectionDifferences(
            targets=_names(selection["targets"]),
            shared_mechanisms=_names(selection["shared_mechanisms"]),
            directed_edges=_edges(selection["directed_edges"]),
            bidirected_edges=_edges(selection["bidirected_edges"]),
        ),
        rules=rules,
        invariances=invariances,
        target_factors=target_factors,
        conditional=conditional,
        obstruction=obstruction,
        obligations=obligations,
        identity=raw["identity"],
        canonical_text=raw["canonical_text"],
    )


def invariance_report(scenario_result: Any) -> ScenarioSetInvarianceReport:
    """The selection differences and invariances behind each scenario answer.

    ``scenario_result`` is the stage returned by
    :func:`antecedent.transport.advanced.prepare_transport_scenarios` after ``estimate()``.
    The stage, its decisions and its last report are not changed. The result holds one
    :class:`InvarianceReport` per scenario and, for each extreme of the structural envelope,
    the report of the scenario that produced it. It is derived (nothing is sealed or exported)
    and deterministic: the same decision reports the same invariances, and a refreshed stage
    with the same decisions reports the same ones again.

    Raises :class:`ScenarioInvarianceRefusal` (a
    :class:`~antecedent.errors.CausalUnsupportedError`) when the stage was not estimated
    (``scenario_invariance.not_estimated``) or is not a scenario stage
    (``scenario_invariance.wrong_result_type``).
    """
    payload, refusal = _native.scenario_invariance_report(scenario_result)
    if refusal is not None:
        raise ScenarioInvarianceRefusal(json.loads(refusal))
    if payload is None:  # pragma: no cover - the native contract
        raise ScenarioInvarianceRefusal(
            {
                "code": "invalid_argument",
                "detail": "scenario_invariance.wrong_result_type",
                "message": "the native report returned neither a result nor a refusal",
            }
        )
    raw = json.loads(payload)
    return ScenarioSetInvarianceReport(
        scenarios=tuple(
            ScenarioInvariance(
                name=item["name"],
                result_status=item["result_status"],
                report=_report(item["report"]),
            )
            for item in raw["scenarios"]
        ),
        extremes=tuple(
            EnvelopeExtremeInvariance(
                outcome=item["outcome"],
                side=item["side"],
                value=float(item["value"]),
                scenario=item["scenario"],
                report=_report(item["report"]),
            )
            for item in raw["extremes"]
        ),
        canonical_text=raw["canonical_text"],
    )
