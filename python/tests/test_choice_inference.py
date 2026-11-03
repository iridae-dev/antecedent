"""Inference from repeated categorical choices, through the Python facade.

Truth is computed here in pure Python: the Dirichlet posterior of one choice per respondent is
``initial + sum(discount * source counts) + respondent counts`` and its mean is that vector over
its sum. Every route is point-only with calibration ``unmeasured``.
"""

from __future__ import annotations

import pytest
from antecedent import choice_inference as ci
from antecedent.adaptive_inference import StoppingRule
from antecedent.choice_inference import ChoiceInferenceRefusal
from antecedent.outcomes import require_success

OPTIONS = ("a", "b", "c")


def respondents(n: int) -> tuple[ci.Respondent, ...]:
    rows = []
    for i in range(n):
        counts = [0, 0, 0]
        counts[0 if i % 4 else 1] = 1
        rows.append(ci.Respondent(f"r{i}", counts))
    return tuple(rows)


def source(ids=None, **changes) -> ci.RetainedSource:
    if ids is None:
        ids = tuple(f"s{i}" for i in range(20))
    fields = {
        "label": "wave0",
        "options": OPTIONS,
        "meaning": "chosen in the pricing task",
        "population": "customers",
        "counts": (10.0, 5.0, 5.0) if len(ids) == 20 else (float(len(ids)), 0.0, 0.0),
        "observation_ids": ids,
        "discount": 0.5,
    }
    fields.update(changes)
    return ci.RetainedSource(**fields)


def study(**changes) -> ci.ChoiceStudy:
    fields = {
        "label": "wave",
        "outcome": "categorical_choice",
        "options": OPTIONS,
        "meaning": "chosen in the pricing task",
        "population": "customers",
        "respondents": respondents(12),
        "initial_prior": (1.0, 1.0, 1.0),
        "sources": (source(),),
        "looks": (12,),
        "stopping_rule": StoppingRule.no_stopping(),
    }
    fields.update(changes)
    return ci.ChoiceStudy(**fields)


def test_the_posterior_is_the_conjugate_closed_form_of_the_counts():
    session = ci.ChoiceInferenceSession.prepare(study())
    result = require_success(session.analyze(ci.Posterior(look=0)))
    counts = [sum(r.counts[k] for r in respondents(12)) for k in range(3)]
    alpha = [1.0 + 0.5 * s + c for s, c in zip((10.0, 5.0, 5.0), counts, strict=True)]
    assert result.alpha == pytest.approx(alpha)
    assert result.mean == pytest.approx([a / sum(alpha) for a in alpha])
    assert result.borrowed_observations == pytest.approx(10.0)
    assert result.calibration == "unmeasured"


def test_a_source_sharing_rows_with_the_respondents_is_refused():
    with pytest.raises(ChoiceInferenceRefusal) as raised:
        ci.ChoiceInferenceSession.prepare(study(sources=(source(ids=("r0#0", "r1#0")),)))
    assert raised.value.reason_code
    assert raised.value.details


def test_a_rationale_does_not_support_cross_population_transfer():
    other = source(population="another market")
    with pytest.raises(ChoiceInferenceRefusal):
        ci.ChoiceInferenceSession.prepare(study(sources=(other,)))
    declared = source(population="another market", transfer_rationale="same price ladder")
    with pytest.raises(ChoiceInferenceRefusal):
        ci.ChoiceInferenceSession.prepare(study(sources=(declared,)))


def test_an_interval_request_is_not_supported_without_coverage():
    session = ci.ChoiceInferenceSession.prepare(study())
    with pytest.raises(ChoiceInferenceRefusal) as raised:
        session.interval()
    assert raised.value.reason_code == "not_supported"


def test_a_stopping_study_has_exact_operating_characteristics():
    session = ci.ChoiceInferenceSession.prepare(
        study(
            looks=(4, 8, 12),
            stopping_rule=StoppingRule.posterior_probability_best(0.9),
        )
    )
    law = require_success(session.operating_characteristics(ci.Kernel(shares=(0.6, 0.3, 0.1))))
    assert sum(law.selection) + law.undecided == pytest.approx(1.0)
    assert 4.0 <= law.expected_enrolled <= 12.0
    assert "not a bound over other truths" in law.statement


def test_study_ranking_orders_by_net_value():
    session = ci.ChoiceInferenceSession.prepare(study())
    rows = require_success(
        session.rank_studies(
            0,
            (1.0, 0.5, 0.0),
            (ci.Candidate("small", 5, 0.0), ci.Candidate("large", 40, 0.0)),
        )
    )
    assert [r.label for r in rows][0] == "large"
    assert all(r.evsi >= 0.0 for r in rows)


def test_a_result_round_trips_through_the_artifact_by_replay():
    session = ci.ChoiceInferenceSession.prepare(study())
    request = ci.Posterior(look=0)
    exported = session.export(request)
    consumed = require_success(
        ci.consume_choice_inference_result_artifact(exported.data, exported.expectation.identity)
    )
    assert consumed.result == require_success(session.analyze(request))
    assert consumed.contract_identity == session.contract_identity


def test_an_omitted_look_schedule_is_refused_not_defaulted():
    with pytest.raises(ChoiceInferenceRefusal) as raised:
        ci.ChoiceInferenceSession.prepare(study(looks=None))
    assert "choice_inference.looks_missing" in raised.value.details


def test_native_retained_posterior_replays_and_preserves_source_identity():
    from dataclasses import replace

    old = tuple(ci.Respondent(f"old{i}", (1, 0, 0) if i % 2 else (0, 1, 0)) for i in range(12))
    old_session = ci.ChoiceInferenceSession.prepare(study(respondents=old, sources=()))
    exported = old_session.export(ci.Posterior(look=0))
    retained = ci.RetainedSource.from_result(
        exported.data, exported.expectation.identity, "previous", 0.5
    )
    current = ci.ChoiceInferenceSession.prepare(study(sources=(retained,)))
    result = require_success(current.analyze(ci.Posterior(look=0)))
    counts = [sum(r.counts[k] for r in respondents(12)) for k in range(3)]
    assert result.alpha == pytest.approx([1 + 3 + counts[0], 1 + 3 + counts[1], 1 + counts[2]])
    result_artifact = current.export(ci.Posterior(look=0))
    replayed = require_success(
        ci.consume_choice_inference_result_artifact(
            result_artifact.data, result_artifact.expectation.identity
        )
    )
    assert replayed.result == result
    assert (
        require_success(
            current.consume_current(result_artifact.data, result_artifact.expectation.identity)
        ).result
        == result
    )
    nested = ci.RetainedSource.from_result(
        result_artifact.data, result_artifact.expectation.identity, "nested", 0.5
    )
    assert nested.counts == pytest.approx([12, 6, 0])
    assert len(nested.observation_ids) == 24
    new_rows = tuple(ci.Respondent(f"new{i}", r.counts) for i, r in enumerate(respondents(12)))
    third = ci.ChoiceInferenceSession.prepare(study(respondents=new_rows, sources=(nested,)))
    third_result = require_success(third.analyze(ci.Posterior(look=0)))
    assert third_result.alpha == pytest.approx([16, 7, 1])
    current.withdraw_source("previous", "source retracted")
    with pytest.raises(ChoiceInferenceRefusal):
        require_success(current.analyze(ci.Posterior(look=0)))
    with pytest.raises(ChoiceInferenceRefusal):
        require_success(
            current.consume_current(result_artifact.data, result_artifact.expectation.identity)
        )
    # Historical replay remains possible, but does not use the withdrawn session as authority.
    assert (
        require_success(
            ci.consume_choice_inference_result_artifact(
                result_artifact.data, result_artifact.expectation.identity
            )
        ).result
        == result
    )
    with pytest.raises(ChoiceInferenceRefusal):
        ci.ChoiceInferenceSession.prepare(study(sources=(replace(retained, counts=(12, 0, 0)),)))
    with pytest.raises(ChoiceInferenceRefusal):
        ci.ChoiceInferenceSession.prepare(
            study(sources=(replace(retained, posterior_identity="bad"),))
        )
    with pytest.raises(ChoiceInferenceRefusal):
        ci.ChoiceInferenceSession.prepare(study(sources=(retained,), initial_prior=(2, 1, 1)))


def test_choice_preparation_obeys_caller_resource_limit():
    from antecedent.execution import Execution, ResourceLimits

    with pytest.raises(ChoiceInferenceRefusal) as refusal:
        ci.ChoiceInferenceSession.prepare(
            study(), execution=Execution(limits=ResourceLimits(memory_bytes=1))
        )
    assert refusal.value.reason_code == "execution_resource_limit"


def test_hierarchical_next_study_keeps_repeated_choice_dependence():
    import math

    counts = [(3, 1), (1, 3), (2, 2), (4, 0), (0, 4), (3, 1), (2, 2)]
    design = study(
        outcome="repeated_choice",
        options=("a", "b"),
        initial_prior=(1.0, 1.0),
        respondents=tuple(ci.Respondent(f"h{i}", row) for i, row in enumerate(counts)),
        sources=(),
        looks=(len(counts),),
    )
    session = ci.ChoiceInferenceSession.prepare(design)
    request = ci.Hierarchical(
        look=0, shapes=(1.0, 1.0), log_concentration_mean=math.log(5), log_concentration_sd=1.0
    )
    mismatched = ci.Hierarchical(
        look=0, shapes=(2.0, 1.0), log_concentration_mean=math.log(5), log_concentration_sd=1.0
    )
    with pytest.raises(ChoiceInferenceRefusal):
        require_success(session.analyze(mismatched))
    with pytest.raises(ChoiceInferenceRefusal):
        require_success(
            session.rank_hierarchical_studies(
                mismatched, (1.0, 1.0), (ci.Candidate("one", 1, 0.0),)
            )
        )
    rows = require_success(
        session.rank_hierarchical_studies(
            request,
            (1.0, 1.0),
            (
                ci.Candidate("one", 1, 0.0),
                ci.Candidate("two", 2, 0.0),
            ),
        )
    )
    assert rows[0].label == "two"
    assert rows[0].evsi == pytest.approx(0.006765558737430455, abs=2e-6)
    assert rows[0].method["kind"] == "hierarchical_quadrature"
    assert rows[0].calibration == "unmeasured"
    assert rows[0].rank_uncertain
