"""The opt-in descriptive fallback: an identification-class refusal, unchanged, beside a
labelled-noncausal association and the evidence that would resolve it.

The fixtures are the two structural failures an observational study meets. A bow (an unmeasured
common cause of the treatment and the outcome that no adjustment closes) and a hedge (the effect
passes through a mediator that shares an unmeasured cause with the outcome) are both refused by
the complete identification algorithm. The default behaviour is unchanged (the refusal is raised),
a successful query returns the ordinary result and never a fallback, a cancellation or a search a
budget stopped is never converted, and the association never ranks actions.
"""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent import host, repair
from antecedent.descriptive import AssociationEstimate
from antecedent.errors import (
    CausalCancelled,
    CausalError,
    EffectNotIdentified,
)
from antecedent.fallback import (
    IDENTIFICATION_REFUSAL_CODES,
    NONCAUSAL_NOTICE,
    AnalysisWithFallback,
    EvidenceNeeds,
    FallbackRoles,
    FallbackUnavailable,
    described,
    fallback_roles,
    is_identification_refusal,
)
from antecedent.outcomes import require_success
from antecedent.state import CancellationToken


def _confounded_data(n: int = 240):
    rng = np.random.default_rng(11)
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(float)
    y = 1.0 + 2.0 * t + z + rng.normal(scale=0.3, size=n)
    return {"z": z, "t": t, "y": y}


def _confounded_graph():
    # z -> t, z -> y, t -> y and t <-> y: adjustment closes the back door through z but not the
    # unmeasured common cause of t and y.
    return ant.Admg.from_edges(["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")], [("t", "y")])


def _hedge_data(n: int = 240):
    rng = np.random.default_rng(12)
    t = (rng.normal(size=n) > 0).astype(float)
    w = (t + rng.normal(size=n) > 0.4).astype(float)
    y = 0.5 + 1.5 * w + rng.normal(scale=0.3, size=n)
    return {"t": t, "w": w, "y": y}


def _hedge_graph():
    return ant.Admg.from_edges(["t", "w", "y"], [("t", "w"), ("w", "y")], [("t", "y"), ("w", "y")])


def _trial():
    return repair.StudyCandidate.experiment(
        "randomized trial",
        population="study",
        interventions=["t"],
        measured=["y"],
        cost=9,
        sample_size=100,
        recruitment="consecutive recruitment",
        timing="baseline",
        unit="patient",
        cost_unit="usd",
        sample_budget=100,
        feasible=True,
    )


def _survey(measured):
    return repair.StudyCandidate.observation(
        "bigger survey",
        population="study",
        measured=measured,
        cost=1,
        sample_size=100,
        recruitment="consecutive recruitment",
        timing="baseline",
        unit="patient",
        cost_unit="usd",
        sample_budget=100,
        feasible=True,
    )


def _describe(outcome) -> AnalysisWithFallback:
    """The refusal outcome of an ``on_refusal='describe'`` analysis, read with its description.

    The causal refusal stays the primary outcome: it is not accepted and its original
    exception is retained. The separate associational outcome and evidence plan are attached.
    """
    assert not outcome.accepted
    assert outcome.refusal is not None
    assert outcome.answer.kind == "refused"
    assert outcome.associational is not None and outcome.evidence_plan is not None
    description = described(outcome)
    assert description.refusal is outcome.refusal._error
    return description


def _association(outcome: AnalysisWithFallback) -> AssociationEstimate:
    section = outcome.associational.association  # type: ignore[union-attr]
    assert isinstance(section, AssociationEstimate)
    return section


def test_an_unmeasured_confounder_returns_the_refusal_the_description_and_the_evidence_needs():
    data, graph = _confounded_data(), _confounded_graph()
    query = ant.AverageEffect("t", "y")
    with pytest.raises(EffectNotIdentified) as raised:
        require_success(ant.analyze(data, graph=graph, query=query))
    primary = ant.analyze(data, graph=graph, query=query, on_refusal="describe")
    outcome = _describe(primary)
    assert isinstance(outcome, AnalysisWithFallback)
    assert primary.refusal.code == "effect_not_identified"
    assert primary.associational.accepted and primary.associational.answer.kind == "associational"
    assert primary.evidence_plan is outcome.evidence_plan

    # The refusal object is the one analyze raises: same class, code, remedy and report.
    refusal = outcome.refusal
    assert type(refusal) is type(raised.value)
    assert refusal.reason_code == raised.value.reason_code == "effect_not_identified"
    assert getattr(refusal, "remedy", None) == getattr(raised.value, "remedy", None)
    assert str(refusal) == str(raised.value)
    assert refusal.report.to_dict() == raised.value.report.to_dict()

    # The description is the compiled associational route, labelled noncausal, holding the
    # measured parent of the treatment fixed, with the refusal retained.
    assert outcome.roles == FallbackRoles(
        outcome="y",
        treatment="t",
        adjustment=("z",),
        rule="measured_parents_of_treatment_and_declared_modifiers",
    )
    result = outcome.associational
    assert not isinstance(result, FallbackUnavailable)
    assert result.answer.kind == "associational"
    assert result.answer.value is None
    section = _association(outcome)
    assert section.refused_after == "effect_not_identified"
    assert section.ranks_actions is False and section.is_intervention_effect is False
    assert section.licensed is False and section.calibration == "unmeasured"
    # The report lists the intercept and the treatment; the held-fixed parent is a nuisance
    # column adjusted arithmetically and not reported. The treatment coefficient (about 2.0,
    # against about 2.8 unadjusted) shows the parent was held fixed.
    assert [term.label for term in section.terms] == ["intercept", "t"]
    assert section.terms[1].estimate == pytest.approx(2.0, abs=0.15)
    assert outcome.noncausal and not outcome.answers_causal_question
    assert outcome.ranks_actions is False

    # The evidence needs: verified obligations with proof steps, next actions, templates.
    needs = outcome.evidence_plan
    assert isinstance(needs, EvidenceNeeds)
    assert needs.obligations and all(o.proof_step for o in needs.obligations)
    ids = {action["id"] for action in needs.next_actions}
    assert "run_experiment:t" in ids and "measure_variable:common_cause_of_t_y" in ids
    assert needs.templates.templates
    assert needs.plan is None  # no candidates declared, none ranked
    # A study supplies data, never an assumption.
    assert all(
        not o.satisfiable_by_study for o in needs.obligations if o.kind == "establish_assumption"
    )


def test_a_hedge_returns_the_refusal_the_description_and_a_ranked_plan():
    outcome = _describe(
        ant.analyze(
            _hedge_data(),
            graph=_hedge_graph(),
            query=ant.AverageEffect("t", "y"),
            on_refusal="describe",
            evidence_candidates=[_trial(), _survey(["t", "w", "y"])],
        )
    )
    assert isinstance(outcome, AnalysisWithFallback)
    assert outcome.refusal.reason_code == "effect_not_identified"
    assert outcome.roles is not None and outcome.roles.adjustment == ()
    assert _association(outcome).refused_after == "effect_not_identified"
    assert outcome.ranks_actions is False and not outcome.answers_causal_question
    assert _association(outcome).ranks_actions is False
    needs = outcome.evidence_plan
    assert isinstance(needs, EvidenceNeeds)
    plan = needs.plan
    assert plan is not None and not isinstance(plan, FallbackUnavailable)
    assert plan.best is not None and plan.best.labels == ("randomized trial",)
    assert plan.best.gate == "verified_sufficient"
    # The cheap survey cannot answer a hedge, whatever it costs.
    survey = next(o for o in plan.options if o.labels == ("bigger survey",))
    assert survey.gate != "verified_sufficient"
    assert any(a["id"].startswith("run_experiment:") for a in needs.next_actions)


def test_a_conditional_effect_holds_its_modifier_fixed():
    rng = np.random.default_rng(13)
    data = {name: rng.normal(size=200) for name in ("t", "y", "w")}
    bow = ant.Admg.from_edges(["t", "y", "w"], [("t", "y")], [("t", "y")])
    outcome = _describe(
        ant.analyze(
            data, graph=bow, query=ant.ConditionalEffect("t", "y", "w"), on_refusal="describe"
        )
    )
    assert isinstance(outcome, AnalysisWithFallback)
    assert outcome.roles is not None and outcome.roles.adjustment == ("w",)


def test_a_successful_causal_query_returns_the_ordinary_result_never_a_fallback():
    rng = np.random.default_rng(14)
    t = (rng.normal(size=200) > 0).astype(float)
    data = {"t": t, "y": 1.0 + 2.0 * t + rng.normal(scale=0.3, size=200)}
    graph = ant.Dag.from_edges(["t", "y"], [("t", "y")])
    ordinary = require_success(
        ant.analyze(
            data, graph=graph, query=ant.AverageEffect("t", "y"), execution=ant.Execution(seed=3)
        )
    )
    optional_outcome = ant.analyze(
        data,
        graph=graph,
        query=ant.AverageEffect("t", "y"),
        execution=ant.Execution(seed=3),
        on_refusal="describe",
    )
    assert optional_outcome.accepted
    assert optional_outcome.associational is None and optional_outcome.evidence_plan is None
    optional = require_success(optional_outcome)
    assert not isinstance(optional, AnalysisWithFallback)
    assert optional.answer.kind == ordinary.answer.kind
    assert optional.answer.value == ordinary.answer.value


def test_the_default_still_raises_and_the_refusal_points_at_the_opt_in():
    data, graph = _confounded_data(), _confounded_graph()
    for kwargs in ({}, {"on_refusal": "refuse"}):
        with pytest.raises(EffectNotIdentified) as caught:
            require_success(
                ant.analyze(data, graph=graph, query=ant.AverageEffect("t", "y"), **kwargs)
            )
        assert "on_refusal='describe'" in caught.value.report.next
        assert "not an answer to the causal question" in caught.value.report.next
    # A capped search is not converted, so its text does not offer the option.
    capped = EffectNotIdentified("capped")
    capped.identification_status = "not_identified"
    capped.search_capped = True
    assert "on_refusal" not in ant.errors.next_action(capped)


def test_only_identification_class_refusals_are_converted():
    assert IDENTIFICATION_REFUSAL_CODES == ("effect_not_identified", "cross_world_not_identified")
    complete = EffectNotIdentified("bow")
    complete.reason_code = "effect_not_identified"
    assert is_identification_refusal(complete)
    capped = EffectNotIdentified("capped")
    capped.reason_code = "effect_not_identified"
    capped.search_capped = True
    assert not is_identification_refusal(capped)
    assert not is_identification_refusal(CausalCancelled("cancelled"))
    assert not is_identification_refusal(ant.errors.CausalValueError("bad argument"))

    # A cancelled study is the cancellation it is, never described.
    token = CancellationToken()
    token.cancel()
    with pytest.raises(CausalCancelled):
        require_success(
            ant.analyze(
                _confounded_data(),
                graph=_confounded_graph(),
                query=ant.AverageEffect("t", "y"),
                on_refusal="describe",
                execution=ant.Execution(cancel=token),
            )
        )
    # An unknown column is an invalid argument, not an identification gap.
    with pytest.raises(CausalError) as invalid:
        require_success(
            ant.analyze(
                _confounded_data(),
                graph=_confounded_graph(),
                query=ant.AverageEffect("t", "nope"),
                on_refusal="describe",
            )
        )
    assert not isinstance(invalid.value, EffectNotIdentified)


def test_a_query_with_no_association_form_or_an_unmeasured_variable_says_so():
    unavailable = fallback_roles(ant.PulseEffect("t", "y"), None, ["t", "y"])
    assert isinstance(unavailable, FallbackUnavailable)
    assert unavailable.part == "associational"
    assert unavailable.code == "route_not_supported"
    assert unavailable.detail == "fallback.no_association_form"
    assert "Next attempt" in unavailable.message
    missing = fallback_roles(ant.AverageEffect("t", "nope"), None, ["t", "y"])
    assert isinstance(missing, FallbackUnavailable)
    assert missing.detail == "fallback.variable_not_measured" and missing.code == "invalid_argument"

    from antecedent import fallback

    other = fallback._evidence(ant.AverageEffect("t", "y"), None, ["t", "y"], None, None)
    assert isinstance(other, FallbackUnavailable)
    assert other.part == "evidence_plan" and other.detail == "fallback.graph_class_not_read"
    identified = ant.Dag.from_edges(["t", "y"], [("t", "y")])
    already = fallback._evidence(ant.AverageEffect("t", "y"), identified, ["t", "y"], None, None)
    assert isinstance(already, FallbackUnavailable)
    assert "not_a_failure" in already.detail or "not_a_failure" in already.message


def test_the_returned_object_states_plainly_that_it_is_not_a_causal_answer():
    outcome = _describe(
        ant.analyze(
            _confounded_data(),
            graph=_confounded_graph(),
            query=ant.AverageEffect("t", "y"),
            on_refusal="describe",
        )
    )
    assert isinstance(outcome, AnalysisWithFallback)
    assert NONCAUSAL_NOTICE.startswith("The associational result describes the observed law")
    for text in (repr(outcome), outcome.explain(), outcome.report(), outcome.notice):
        assert "answer to the causal question" in text
    assert "effect_not_identified" in outcome.report()
    assert "Ranks actions: no" in outcome.report()


def test_the_host_record_shows_both_statuses_and_satisfies_the_schema():
    primary = ant.analyze(
        _confounded_data(),
        graph=_confounded_graph(),
        query=ant.AverageEffect("t", "y"),
        on_refusal="describe",
    )
    outcome = _describe(primary)
    record = host.fallback_response(primary)
    assert outcome.to_dict() == record
    assert record["status"] == "unsupported" and record["causal_status"] == "unsupported"
    assert record["associational_status"] == "supported"
    assert record["refusal"]["reason_code"] == "effect_not_identified"
    assert record["associational"]["claim"]["causal"] is False
    assert record["associational"]["ranks_actions"] is False
    assert record["evidence_plan"]["obligations"]
    required = host.schema()["$defs"]["fallback_response"]["required"]
    assert all(key in record for key in required)


def test_a_bayesian_study_is_described_with_its_posterior():
    outcome = _describe(
        ant.analyze(
            _confounded_data(),
            graph=_confounded_graph(),
            query=ant.AverageEffect("t", "y"),
            inference=ant.Bayesian(backend="conjugate", n_draws=200),
            on_refusal="describe",
        )
    )
    assert isinstance(outcome, AnalysisWithFallback)
    assert outcome.refusal.reason_code == "effect_not_identified"
    assert _association(outcome).posterior is not None
