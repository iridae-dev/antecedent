"""The evidence plan of a real study, reached through ``analyze`` and ``prepare``.

Nothing here hand-builds a contract: each study declares data, an ``Admg`` and an average effect
the structure does not identify. The refusal is the one ``prepare`` and ``analyze`` have always
raised; ``on_refusal="describe"`` keeps it unchanged and reads the evidence that would resolve it
off the same study: the verified obligations, the ``measure_variable:<name>`` /
``run_experiment:<name>`` / ``find_instrument`` actions, the candidate templates and, over the
studies the caller declares (cost included), a ranked plan that exports and replays.

Expected verdicts follow from the theorems: measuring the unmeasured common cause closes a back
door, a hedge is answered by an experiment, measuring the mediator's own unmeasured cause repairs
the front-door route, and an instrument's validity is an assumption no study establishes.
"""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent import design, repair
from antecedent.errors import CausalUnsupportedError, EffectNotIdentified
from antecedent.fallback import (
    AnalysisWithFallback,
    EvidenceNeeds,
    FallbackUnavailable,
    described,
)
from antecedent.outcomes import require_success

_QUERY = ant.AverageEffect("t", "y")


def _confounded_data(n: int = 240):
    rng = np.random.default_rng(21)
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(float)
    y = 1.0 + 2.0 * t + z + rng.normal(scale=0.3, size=n)
    return {"z": z, "t": t, "y": y}


def _bow():
    # z -> t, z -> y, t -> y and t <-> y: the back door through z is closed by adjustment, the
    # unmeasured common cause of t and y is not.
    return ant.Admg.from_edges(["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")], [("t", "y")])


def _instrument():
    # z -> t -> y with t <-> y: z is measured, yet no nonparametric argument identifies the effect.
    return ant.Admg.from_edges(["z", "t", "y"], [("z", "t"), ("t", "y")], [("t", "y")])


def _hedge_data(n: int = 240):
    rng = np.random.default_rng(22)
    t = (rng.normal(size=n) > 0).astype(float)
    w = (t + rng.normal(size=n) > 0.4).astype(float)
    y = 0.5 + 1.5 * w + rng.normal(scale=0.3, size=n)
    return {"t": t, "w": w, "y": y}


def _hedge():
    return ant.Admg.from_edges(["t", "w", "y"], [("t", "w"), ("w", "y")], [("t", "y"), ("w", "y")])


def _declared(label, measured, cost, *, kind="observation", interventions=()):
    make = getattr(repair.StudyCandidate, kind)
    fields = {"interventions": list(interventions)} if kind == "experiment" else {}
    return make(
        label,
        population="study",
        measured=measured,
        cost=cost,
        sample_size=100,
        recruitment="consecutive recruitment",
        timing="baseline",
        unit="patient",
        cost_unit="usd",
        sample_budget=100,
        feasible=True,
        **fields,
    )


def _described(data, graph, **kwargs) -> EvidenceNeeds:
    primary = ant.analyze(data, graph=graph, query=_QUERY, on_refusal="describe", **kwargs)
    # The causal refusal stays the primary outcome; the description is attached beside it.
    assert not primary.accepted and primary.refusal.code == "effect_not_identified"
    outcome = described(primary)
    assert isinstance(outcome, AnalysisWithFallback)
    needs = outcome.evidence_plan
    assert primary.evidence_plan is needs
    assert isinstance(needs, EvidenceNeeds), needs
    return needs


def _actions(needs: EvidenceNeeds) -> set[str]:
    return {action["id"] for action in needs.next_actions}


def _assert_refusal_unchanged(data, graph) -> None:
    with pytest.raises(EffectNotIdentified) as analyzed:
        require_success(ant.analyze(data, graph=graph, query=_QUERY))
    study = ant.prepare(data, query=_QUERY, graph=graph)
    assert study.refusal is not None and study.refusal.code == "effect_not_identified"
    with pytest.raises(EffectNotIdentified) as prepared:
        require_success(ant.estimate(study))
    assert prepared.value.reason_code == analyzed.value.reason_code == "effect_not_identified"
    assert str(prepared.value) == str(analyzed.value)
    primary = ant.analyze(data, graph=graph, query=_QUERY, on_refusal="describe")
    outcome = described(primary)
    assert isinstance(outcome, AnalysisWithFallback)
    assert str(outcome.refusal) == str(analyzed.value)
    assert primary.refusal.code == outcome.refusal.reason_code == "effect_not_identified"


def _assert_gate_then_cost(plan) -> None:
    options = plan.options
    verified = [o for o in options if o.gate == "verified_sufficient"]
    assert [o.rank for o in options] == list(range(len(options)))
    assert list(options[: len(verified)]) == verified, "a verified option is ranked below another"
    costs = [o.cost for o in verified]
    assert costs == sorted(costs)


def _assert_replays(plan) -> None:
    replayed = design.consume_plan(bytes(plan.export()))
    assert replayed.identity == plan.identity
    assert [o.labels for o in replayed.options] == [o.labels for o in plan.options]
    assert replayed.outcome == plan.outcome


def test_an_unmeasured_confounder_is_closed_by_the_study_that_measures_it():
    data, graph = _confounded_data(), _bow()
    _assert_refusal_unchanged(data, graph)
    candidates = [
        _declared("survey", ["z", "t", "y"], 1),
        _declared("assay", ["z", "t", "y", "common_cause_of_t_y"], 5),
        _declared("trial", ["y"], 20, kind="experiment", interventions=["t"]),
    ]
    needs = _described(data, graph, evidence_candidates=candidates)

    # The needs are the verified obligations of the failure, each with its proof step.
    assert needs.obligations and all(o.proof_step for o in needs.obligations)
    assert {"measure_variable:common_cause_of_t_y", "run_experiment:t"} <= _actions(needs)
    assert all(
        not o.satisfiable_by_study for o in needs.obligations if o.kind == "establish_assumption"
    )
    assert needs.templates.templates
    assert {"experiment", "observation"} <= {t.kind for t in needs.templates.templates}

    plan = needs.plan
    assert plan is not None and not isinstance(plan, FallbackUnavailable)
    assert plan.outcome == "closing"
    assert plan.best is not None and plan.best.labels == ("assay",)
    assert any(
        o.route == "backdoor" and o.labels == ("assay",) and o.gate == "verified_sufficient"
        for o in plan.options
    )
    assert all(o.gate != "verified_sufficient" for o in plan.options if o.labels == ("survey",))
    _assert_gate_then_cost(plan)
    _assert_replays(plan)


def test_a_front_door_route_is_repaired_by_measuring_the_cause_of_the_mediator():
    data, graph = _hedge_data(), _hedge()
    _assert_refusal_unchanged(data, graph)
    # The mediator is measured today and the treatment's own unmeasured cause stays unmeasured:
    # the back door stays open, the front door closes once the mediator's cause is measured.
    candidates = [
        _declared("survey", ["t", "w", "y"], 1),
        _declared("mediator cohort", ["t", "w", "y", "common_cause_of_w_y"], 4),
    ]
    needs = _described(data, graph, evidence_candidates=candidates)
    assert "measure_variable:common_cause_of_w_y" in _actions(needs)
    plan = needs.plan
    assert plan is not None and not isinstance(plan, FallbackUnavailable)
    cohort = next(
        o for o in plan.options if o.route == "frontdoor" and o.labels == ("mediator cohort",)
    )
    assert cohort.gate == "verified_sufficient"
    assert all(o.gate != "verified_sufficient" for o in plan.options if o.route == "backdoor")
    assert plan.outcome == "closing"
    _assert_gate_then_cost(plan)
    _assert_replays(plan)


def test_a_measured_instrument_is_proposed_but_its_validity_is_an_assumption_no_study_establishes():
    data, graph = _confounded_data(), _instrument()
    _assert_refusal_unchanged(data, graph)
    candidates = [
        _declared("instrument cohort", ["z", "t", "y"], 2, kind="instrument"),
        _declared("assay", ["z", "t", "y", "common_cause_of_t_y"], 6),
    ]
    needs = _described(data, graph, evidence_candidates=candidates)
    assert {"find_instrument", "measure_variable:common_cause_of_t_y"} <= _actions(needs)
    plan = needs.plan
    assert plan is not None and not isinstance(plan, FallbackUnavailable)
    assert all(o.gate != "verified_sufficient" for o in plan.options if o.route == "instrument")
    assert plan.outcome == "closing"
    assert plan.best is not None and plan.best.labels == ("assay",)
    _assert_gate_then_cost(plan)
    _assert_replays(plan)


def test_a_hedge_is_answered_by_an_experiment_and_a_cheaper_survey_cannot_answer_it():
    data, graph = _hedge_data(), _hedge()
    candidates = [
        _declared("randomized trial", ["y"], 9, kind="experiment", interventions=["t"]),
        _declared("bigger survey", ["t", "w", "y"], 1),
    ]
    needs = _described(data, graph, evidence_candidates=candidates)
    assert _actions(needs) == {
        "collect_panel",
        "find_instrument",
        "measure_variable:common_cause_of_t_y",
        "measure_variable:common_cause_of_w_y",
        "run_experiment:t",
        "run_experiment:w",
    }
    plan = needs.plan
    assert plan is not None and not isinstance(plan, FallbackUnavailable)
    assert plan.best is not None and plan.best.labels == ("randomized trial",)
    assert plan.best.gate == "verified_sufficient" and plan.outcome == "closing"
    survey = [o for o in plan.options if o.labels == ("bigger survey",)]
    assert survey and all(o.gate != "verified_sufficient" for o in survey)
    _assert_gate_then_cost(plan)
    _assert_replays(plan)

    # The caller's declarations are the caller's: studies in two cost units are not compared.
    euro = repair.StudyCandidate.observation(
        "survey in euros",
        population="study",
        measured=["t", "w", "y"],
        cost=1,
        sample_size=100,
        recruitment="consecutive recruitment",
        timing="baseline",
        unit="patient",
        cost_unit="eur",
        sample_budget=100,
        feasible=True,
    )
    mixed = _described(data, graph, evidence_candidates=[candidates[0], euro])
    assert isinstance(mixed.plan, FallbackUnavailable)
    assert mixed.plan.code == "design_cost_units_mismatch"


def test_an_identified_study_returns_the_ordinary_result_and_owes_no_evidence():
    rng = np.random.default_rng(23)
    t = (rng.normal(size=200) > 0).astype(float)
    data = {"t": t, "y": 1.0 + 2.0 * t + rng.normal(scale=0.3, size=200)}
    graph = ant.Admg.from_edges(["t", "y"], [("t", "y")], [])
    outcome = require_success(
        ant.analyze(
            data, graph=graph, query=_QUERY, on_refusal="describe", execution=ant.Execution(seed=5)
        )
    )
    assert not isinstance(outcome, AnalysisWithFallback)
    assert (
        outcome.answer.kind
        == require_success(
            ant.analyze(data, graph=graph, query=_QUERY, execution=ant.Execution(seed=5))
        ).answer.kind
    )

    # The evidence routes of a graph that identifies the effect refuse by name.
    with pytest.raises(CausalUnsupportedError) as caught:
        repair.FailureContract(
            graph=graph, treatment="t", outcome="y", population="study", observed=["t", "y"]
        )
    assert caught.value.detail == "identification_repair.not_a_failure"
