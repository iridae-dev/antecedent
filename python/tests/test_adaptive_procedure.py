"""Derived uncertainty of an adaptive procedure through the Python facade.

Truth is computed here in pure Python, never by native code:

* a two-unit fixed-allocation tree enumerated by hand (contrast ``+1`` with probability 0.24,
  ``0`` with 0.22, ``-1`` with 0.04, undefined with 0.5);
* the Beta integral ``B(2, 3) = 1/12`` for the expected loss after one success of an arm with a
  uniform competitor, which decides the stopping time and decision law of an expected-loss rule;
* the standard normal distribution function for a single-look stagewise p-value;
* the martingale property of a posterior mean, and the anytime-valid bound ``<= alpha``.

Every result is labelled ``derived_unmeasured`` and every premise failure is refused with
``cell_not_licensed`` and the premise's own detail.
"""

from __future__ import annotations

import math

import pytest
from antecedent import adaptive_inference as ai
from antecedent.adaptive_inference import AdaptiveInferenceRefusal

from _refusal import assert_registered_refusal

TOL = 1e-9


def normal_sf(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def declare(
    assignment: ai.AssignmentRule,
    enrolment: tuple[int, ...],
    stopping: ai.ProcedureStopping | None = None,
    observation: ai.OutcomeObservation | None = None,
    selection: ai.FinalSelection | None = None,
    arms: int = 2,
) -> ai.AdaptiveProcedure:
    return ai.AdaptiveProcedure.declare(
        ai.ProcedureDeclaration(
            arms=arms,
            assignment=assignment,
            enrolment=enrolment,
            observation=observation or ai.OutcomeObservation(),
            exposure=ai.ExposureModel((0.0,) * arms, 0, "delivered"),
            stopping=stopping or ai.ProcedureStopping.no_stopping(),
            selection=selection or ai.FinalSelection.no_selection(),
        )
    )


EVEN = ai.AssignmentRule.fixed((0.5, 0.5))
THOMPSON = ai.AssignmentRule.thompson_beta()


def refused(call, detail: str, code: str = "cell_not_licensed") -> AdaptiveInferenceRefusal:
    with pytest.raises(AdaptiveInferenceRefusal) as caught:
        call()
    assert_registered_refusal(caught.value)
    assert caught.value.reason_code == code
    assert caught.value.detail == detail
    return caught.value


def test_each_missing_element_is_its_own_refusal():
    with pytest.raises(AdaptiveInferenceRefusal) as caught:
        ai.AdaptiveProcedure.declare(ai.ProcedureDeclaration(arms=2))
    assert caught.value.details == (
        "adaptive_inference.policy_history_missing",
        "adaptive_inference.interim_analyses_missing",
        "adaptive_inference.outcome_availability_missing",
        "adaptive_inference.exposure_history_missing",
        "adaptive_inference.stopping_rule_missing",
        "adaptive_inference.policy_selection_missing",
    )
    assert_registered_refusal(caught.value)
    assert caught.value.reason_code == "adaptive_contract_incomplete"


def test_the_fixed_design_contrast_law_matches_the_hand_enumeration():
    procedure = declare(EVEN, (2,))
    report = procedure.report(ai.FixedSampleProcedureReport(truth=(0.2, 0.6)))
    assert report.claim == ai.DERIVED_UNMEASURED
    assert report.procedure_identity == procedure.identity
    law = report.body["law"]
    assert law["undefined_mass"] == pytest.approx(0.5, abs=TOL)
    support = {round(v, 12): p for v, p in law["support"]}
    assert support[-1.0] == pytest.approx(0.04, abs=TOL)
    assert support[0.0] == pytest.approx(0.22, abs=TOL)
    assert support[1.0] == pytest.approx(0.24, abs=TOL)
    assert report.body["mean"] == pytest.approx(0.4, abs=TOL)


def test_the_identity_changes_with_an_element_of_unchanged_shape():
    a = declare(ai.AssignmentRule.epsilon_greedy(0.1), (1, 1, 1))
    assert a.identity == declare(ai.AssignmentRule.epsilon_greedy(0.1), (1, 1, 1)).identity
    assert a.identity != declare(ai.AssignmentRule.epsilon_greedy(0.2), (1, 1, 1)).identity
    assert a.contract_identity is None


def test_the_expected_loss_rule_decides_with_hand_integrated_losses():
    # All units go to arm 0; after one outcome the smallest expected loss is B(2, 3) = 1/12.
    procedure = declare(
        ai.AssignmentRule.fixed((1.0, 0.0)),
        (1, 1),
        stopping=ai.ProcedureStopping.posterior_expected_loss(0.1),
    )
    report = procedure.report(ai.DecisionProcedureReport(truth=(0.7, 0.4)))
    body = report.body
    assert body["stop_probabilities"][0] == pytest.approx(1.0, abs=1e-8)
    assert body["expected_enrolled"] == pytest.approx(1.0, abs=1e-8)
    assert body["decision_probabilities"] == pytest.approx([0.7, 0.3], abs=1e-8)
    assert body["wrong_decision_probability"] == pytest.approx(0.3, abs=1e-8)
    assert body["expected_regret"] == pytest.approx(0.3 * (0.7 - 0.4), abs=1e-8)


def test_the_posterior_mean_is_a_martingale_of_an_adaptive_stopped_procedure():
    procedure = declare(
        THOMPSON,
        (1, 1, 1, 1),
        stopping=ai.ProcedureStopping.posterior_probability_best(0.8),
        observation=ai.OutcomeObservation(lag_looks=1),
        selection=ai.FinalSelection.highest_observed_mean(),
    )
    report = procedure.report(ai.BayesianProcedureReport(prior=(2.0, 1.0), observed=(3, 1, 1, 2)))
    assert report.paradigm == "bayesian_posterior"
    for gap in report.body["mean_gaps"]:
        assert gap["prior_mean"] == pytest.approx(2.0 / 3.0, abs=TOL)
        assert abs(gap["gap"]) < 1e-9
    assert report.body["posterior"]["mean"] == pytest.approx([5.0 / 7.0, 3.0 / 6.0], abs=TOL)


def test_every_paradigm_whose_premise_fails_is_refused():
    adaptive = declare(
        THOMPSON,
        (1, 1, 1),
        stopping=ai.ProcedureStopping.posterior_probability_best(0.9),
    )
    truth = (0.3, 0.6)
    refused(
        lambda: adaptive.report(ai.FixedSampleProcedureReport(truth=truth)),
        "adaptive_inference.fixed_sample_design_adaptive",
    )
    refused(
        lambda: adaptive.report(ai.GroupSequentialProcedureReport(truth=truth)),
        "adaptive_inference.group_sequential_assignment_adaptive",
    )
    refused(
        lambda: adaptive.report(ai.DecisionProcedureReport(truth=truth)),
        "adaptive_inference.rule_mismatch",
    )
    lossy = declare(
        THOMPSON,
        (1, 1),
        observation=ai.OutcomeObservation(retain_success=1.0, retain_failure=0.5),
    )
    refused(
        lambda: lossy.report(ai.BayesianProcedureReport()),
        "adaptive_inference.posterior_premise_outcome_dependent_delay",
    )
    refused(
        lambda: lossy.report(ai.AnytimeProcedureReport(truth=truth)),
        "adaptive_inference.outcome_dependent_delay",
    )
    refused(
        lambda: lossy.simulate(ai.WeightingProcedureReport(truth, 10, 1)),
        "adaptive_inference.outcome_dependent_delay",
    )


def test_a_single_look_stagewise_inference_is_the_normal_interval():
    procedure = declare(
        EVEN,
        (10,),
        stopping=ai.ProcedureStopping.wald_boundaries((1.96,), "one_sided_upper"),
    )
    report = procedure.report(
        ai.GroupSequentialProcedureReport(
            truth=(0.5, 0.5), level=0.95, observed_look=0, observed_statistic=1.5
        )
    )
    stagewise = report.body["stagewise"]
    z = 1.959963984540054
    assert stagewise["median_unbiased"] == pytest.approx(1.5, abs=1e-5)
    assert stagewise["lower"] == pytest.approx(1.5 - z, abs=1e-4)
    assert stagewise["upper"] == pytest.approx(1.5 + z, abs=1e-4)
    assert stagewise["p_value_at_null"] == pytest.approx(normal_sf(1.5), abs=1e-6)
    assert report.body["stop_probabilities"][0] + report.body["horizon_probability"] == (
        pytest.approx(1.0, abs=TOL)
    )


def test_the_anytime_bounds_exclude_the_truth_with_probability_at_most_alpha():
    procedure = declare(
        THOMPSON,
        (1,) * 8,
        stopping=ai.ProcedureStopping.posterior_probability_best(0.9),
    )
    for truth in ((0.0, 1.0), (0.3, 0.6), (0.5, 0.5)):
        report = procedure.report(
            ai.AnytimeProcedureReport(
                alpha=0.1, tuning_samples=8.0, observed=(2, 1, 1, 3), truth=truth
            )
        )
        assert 0.0 <= report.body["exclusion_probability"] <= 0.1 + 1e-12
        assert report.body["bounds"]["alpha_per_cell"] == pytest.approx(0.05, abs=TOL)


def test_the_simulated_weighted_estimate_recovers_the_contrast_and_replays():
    procedure = declare(
        ai.AssignmentRule.epsilon_greedy(0.5),
        (10, 10, 10, 10),
        selection=ai.FinalSelection.highest_observed_mean(),
    )
    request = ai.WeightingProcedureReport(truth=(0.3, 0.6), replications=600, seed=20240607)
    first = procedure.report(request)
    body = first.body
    assert body["truth_value"] == pytest.approx(0.3, abs=TOL)
    assert body["replications"] == 600
    assert abs(body["bias"]) <= 5.0 * body["monte_carlo_standard_error"] + 0.02
    assert procedure.report(request).body == body
    moved = procedure.report(
        ai.WeightingProcedureReport(truth=(0.3, 0.6), replications=600, seed=20240608)
    )
    assert moved.body != body
    simulation = procedure.simulate(request)
    assert len(simulation.replicates) == 600


def test_a_deterministic_rule_has_no_weighted_estimate():
    procedure = declare(
        ai.AssignmentRule.upper_confidence_bound(2.0),
        (1,) * 6,
        selection=ai.FinalSelection.highest_observed_mean(),
    )
    refused(
        lambda: procedure.simulate(ai.WeightingProcedureReport((0.3, 0.6), 5, 1)),
        "adaptive_inference.positivity_floor",
        code="adaptive_assumption_violated",
    )


def test_the_exact_law_reports_atoms_and_work():
    procedure = declare(EVEN, (1, 1), observation=ai.OutcomeObservation(lag_looks=1))
    law = procedure.law((0.3, 0.7))
    assert law.claim == ai.DERIVED_UNMEASURED
    assert law.cumulative_enrolment == (1, 2)
    assert sum(atom["probability"] for atom in law.atoms) == pytest.approx(1.0, abs=TOL)
    assert all(sum(atom["observed"]) == 1 for atom in law.atoms)
    assert law.work_units > 0
