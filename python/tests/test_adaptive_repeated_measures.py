"""Derived uncertainty of repeated-measures and clustered adaptive procedures.

Truth is computed here in pure Python, never by native code:

* the variance of a unit's mean measurement under compound symmetry,
  ``sigma^2 (1 + (m - 1) rho) / m`` (design effect ``1 + (m - 1) rho``);
* the variance ``tau^2 + sigma^2 / m`` of a cluster mean with a unit-level random effect;
* the enumerated binomial mixture of a two-unit fixed 50/50 allocation (one unit in each arm with
  probability 1/2; the contrast is then normal with variance ``2 v``);
* the martingale property of the adaptively weighted score (unbiased under epsilon-greedy
  assignment), with its Monte Carlo standard error.

Every result is ``derived_unmeasured``; every refusal is ``cell_not_licensed`` (premise) or
``adaptive_contract_incomplete`` / ``invalid_argument`` (declaration) with its own detail.
"""

from __future__ import annotations

import pytest
from antecedent import adaptive_inference as ai
from antecedent.adaptive_inference import AdaptiveInferenceRefusal

from _refusal import assert_registered_refusal

TOL = 1e-9
FIXED_CELL = "fixed_randomization/repeated_measures/final_analysis_only/no_stopping"
GREEDY_CELL = "epsilon_greedy/repeated_measures/final_analysis_only/no_stopping"


def declare(
    *,
    assignment: ai.UnitAssignment | None,
    enrolment: tuple[int, ...] | None,
    structure: ai.OutcomeStructure | None,
    analysis: ai.FinalAnalysis | None,
    lag_looks: int = 0,
) -> ai.RepeatedMeasuresProcedure:
    return ai.RepeatedMeasuresProcedure.declare(
        ai.RepeatedMeasuresDeclaration(
            arms=2,
            assignment=assignment,
            enrolment=enrolment,
            structure=structure,
            analysis=analysis,
            stopping=ai.ProcedureStopping.no_stopping(),
            lag_looks=lag_looks,
        )
    )


def rm(measurements: int, variance: float, rho: float | None) -> ai.OutcomeStructure:
    correlation = None if rho is None else ai.WithinUnitCorrelation.compound_symmetric(rho)
    return ai.OutcomeStructure.repeated_measures(measurements, variance, correlation)


EVEN = ai.UnitAssignment.fixed((0.5, 0.5))
GREEDY = ai.UnitAssignment.epsilon_greedy(0.4)


def refused(call, detail: str, code: str = "cell_not_licensed") -> AdaptiveInferenceRefusal:
    with pytest.raises(AdaptiveInferenceRefusal) as caught:
        call()
    assert_registered_refusal(caught.value)
    assert caught.value.reason_code == code
    assert caught.value.detail == detail
    return caught.value


def test_a_missing_within_unit_correlation_is_its_own_refusal():
    with pytest.raises(AdaptiveInferenceRefusal) as caught:
        declare(
            assignment=EVEN,
            enrolment=(10,),
            structure=rm(4, 1.0, None),
            analysis=ai.FinalAnalysis.cluster_robust(),
        )
    assert caught.value.details == ("adaptive_inference.within_unit_correlation_missing",)
    assert_registered_refusal(caught.value)
    assert caught.value.reason_code == "adaptive_contract_incomplete"


def test_every_missing_element_gives_its_own_detail():
    with pytest.raises(AdaptiveInferenceRefusal) as caught:
        ai.RepeatedMeasuresProcedure.declare(ai.RepeatedMeasuresDeclaration(arms=2))
    assert caught.value.details == (
        "adaptive_inference.policy_history_missing",
        "adaptive_inference.interim_analyses_missing",
        "adaptive_inference.outcome_structure_missing",
        "adaptive_inference.final_analysis_missing",
        "adaptive_inference.stopping_rule_missing",
    )
    assert caught.value.reason_code == "adaptive_contract_incomplete"


def test_a_missing_variance_component_of_a_cluster_is_its_own_refusal():
    with pytest.raises(AdaptiveInferenceRefusal) as caught:
        declare(
            assignment=EVEN,
            enrolment=(10,),
            structure=ai.OutcomeStructure.hierarchical(4),
            analysis=ai.FinalAnalysis.cluster_robust(),
        )
    assert caught.value.details == (
        "adaptive_inference.between_unit_variance_missing",
        "adaptive_inference.within_unit_variance_missing",
    )


def test_an_invalid_correlation_is_invalid_argument():
    with pytest.raises(AdaptiveInferenceRefusal) as caught:
        declare(
            assignment=EVEN,
            enrolment=(10,),
            structure=rm(4, 1.0, 1.0),
            analysis=ai.FinalAnalysis.cluster_robust(),
        )
    assert caught.value.details == ("adaptive_inference.correlation_invalid",)
    assert caught.value.reason_code == "invalid_argument"


def test_the_closed_forms_are_the_design_effect_and_the_enumerated_mixture():
    # m = 4, sigma^2 = 2, rho = 0.3: v = 2 * (1 + 3 * 0.3) / 4 = 0.95; two units, 50/50.
    procedure = declare(
        assignment=EVEN,
        enrolment=(2,),
        structure=rm(4, 2.0, 0.3),
        analysis=ai.FinalAnalysis.cluster_robust(),
    )
    assert procedure.cell["id"] == FIXED_CELL
    report = procedure.closed_form((0.0, 0.5))
    assert report.claim == ai.DERIVED_UNMEASURED
    assert report.procedure_identity == procedure.identity
    assert report.route == "cluster_robust_unit_summaries"
    body = report.body
    assert body["unit_summary_variance"] == pytest.approx(0.95, abs=TOL)
    assert body["design_effect"] == pytest.approx(1.9, abs=TOL)
    assert body["marginal_variance"] == pytest.approx(2.0, abs=TOL)
    assert body["measurements"] == 4
    assert body["revealed_units"] == 2
    assert body["contrast_value"] == pytest.approx(0.5, abs=TOL)
    law = body["exact_law"]
    assert law["undefined_mass"] == pytest.approx(0.5, abs=TOL)
    assert len(law["components"]) == 1
    assert law["components"][0]["weight"] == pytest.approx(0.5, abs=TOL)
    assert law["components"][0]["variance"] == pytest.approx(1.9, abs=TOL)
    assert law["mean"] == pytest.approx(0.5, abs=TOL)


def test_a_cluster_mean_has_variance_between_plus_within_over_members():
    procedure = declare(
        assignment=EVEN,
        enrolment=(2,),
        structure=ai.OutcomeStructure.hierarchical(5, 0.5, 1.5),
        analysis=ai.FinalAnalysis.cluster_robust(),
    )
    assert procedure.cell["outcome"] == "hierarchical_clustered"
    body = procedure.closed_form((1.0, 1.0)).body
    assert body["unit_summary_variance"] == pytest.approx(0.5 + 1.5 / 5, abs=TOL)
    assert body["design_effect"] == pytest.approx(2.0, abs=TOL)


def test_autoregressive_gls_has_the_closed_form_variance():
    # m = 3, rho = 0.5, sigma^2 = 1: GLS (1 + rho) / (2 + (m - 2)(1 - rho)) = 0.6;
    # the mean has 5.5 / 9.
    def variance(summary: str) -> float:
        procedure = declare(
            assignment=EVEN,
            enrolment=(4,),
            structure=ai.OutcomeStructure.repeated_measures(
                3, 1.0, ai.WithinUnitCorrelation.autoregressive(0.5)
            ),
            analysis=ai.FinalAnalysis.cluster_robust(summary),
        )
        return procedure.closed_form((0.0, 0.0)).body["unit_summary_variance"]

    assert variance("gls") == pytest.approx(0.6, abs=TOL)
    assert variance("mean") == pytest.approx(5.5 / 9.0, abs=TOL)


def test_the_derived_fixed_sample_report_separates_exact_values_from_monte_carlo_error():
    procedure = declare(
        assignment=EVEN,
        enrolment=(12,),
        structure=rm(3, 2.0, 0.5),
        analysis=ai.FinalAnalysis.cluster_robust(),
    )
    report = procedure.derive((0.0, 0.25), replications=4000, seed=13)
    assert report.claim == ai.DERIVED_UNMEASURED
    body = report.body
    assert body["kind"] == "derived"
    assert body["paradigm"] == "frequentist_fixed_sample"
    assert body["unit_summary_variance"] == pytest.approx(2.0 * 2.0 / 3.0, abs=TOL)
    assert body["design_effect"] == pytest.approx(2.0, abs=TOL)
    exact = body["exact_law"]
    simulated = body["simulated"]
    assert simulated["monte_carlo_standard_error"] > 0.0
    assert abs(simulated["bias"]) <= 5.0 * simulated["monte_carlo_standard_error"]
    assert simulated["variance"] == pytest.approx(exact["variance"], rel=0.1)


def test_the_weighting_route_is_labelled_and_unbiased_under_epsilon_greedy():
    procedure = declare(
        assignment=GREEDY,
        enrolment=(1,) * 30,
        structure=rm(3, 1.0, 0.4),
        analysis=ai.FinalAnalysis.adaptively_weighted(),
    )
    assert procedure.cell["id"] == GREEDY_CELL
    assert procedure.cell["paradigm"] == "frequentist_adaptive_weighting"
    report = procedure.derive((0.0, 0.5), replications=2000, seed=13)
    assert report.route == "adaptively_weighted_unit_scores"
    body = report.body
    assert body["paradigm"] == "frequentist_adaptive_weighting"
    assert body["exact_law"] is None
    simulated = body["simulated"]
    assert abs(simulated["bias"]) <= 5.0 * simulated["monte_carlo_standard_error"]
    assert body["enrolled"] == 30


def test_a_fixed_sample_analysis_after_adaptive_assignment_is_refused():
    procedure = declare(
        assignment=GREEDY,
        enrolment=(1,) * 20,
        structure=rm(3, 1.0, 0.4),
        analysis=ai.FinalAnalysis.cluster_robust(),
    )
    detail = "adaptive_inference.fixed_sample_design_adaptive"
    refused(lambda: procedure.derive((0.0, 0.5), replications=10, seed=1), detail)
    refused(
        lambda: procedure.derive_paradigm(
            "frequentist_fixed_sample", (0.0, 0.5), replications=10, seed=1
        ),
        detail,
    )


def test_an_independence_assumption_for_dependent_measurements_is_refused():
    procedure = declare(
        assignment=EVEN,
        enrolment=(10,),
        structure=rm(4, 1.0, 0.3),
        analysis=ai.FinalAnalysis.pooled_independent(),
    )
    refused(
        lambda: procedure.derive((0.0, 0.0), replications=10, seed=1),
        "adaptive_inference.independence_assumed_for_dependent_measurements",
    )
    # Independence declared as a zero correlation is admitted.
    independent = declare(
        assignment=EVEN,
        enrolment=(10,),
        structure=rm(4, 1.0, 0.0),
        analysis=ai.FinalAnalysis.pooled_independent(),
    )
    report = independent.derive((0.0, 0.0), replications=200, seed=1)
    assert report.claim == ai.DERIVED_UNMEASURED
    assert report.body["design_effect"] == pytest.approx(1.0, abs=TOL)


def test_a_paradigm_that_is_not_derived_for_the_cell_is_refused():
    procedure = declare(
        assignment=GREEDY,
        enrolment=(1,) * 20,
        structure=rm(3, 1.0, 0.4),
        analysis=ai.FinalAnalysis.adaptively_weighted(),
    )
    refused(
        lambda: procedure.derive_paradigm(
            "anytime_valid_frequentist", (0.0, 0.5), replications=10, seed=1
        ),
        "adaptive_inference.repeated_measures_paradigm_not_derived",
    )


def test_a_guarantee_is_not_paired_across_cells():
    procedure = declare(
        assignment=GREEDY,
        enrolment=(1,) * 20,
        structure=rm(3, 1.0, 0.4),
        analysis=ai.FinalAnalysis.adaptively_weighted(),
    )
    accepted = procedure.transfer_guarantee(GREEDY_CELL, "frequentist_adaptive_weighting")
    assert accepted.body == {"kind": "cell_transfer", "cell_id": GREEDY_CELL}
    for cell_id, paradigm in (
        ("epsilon_greedy/single_outcome/final_analysis_only/no_stopping",
         "frequentist_adaptive_weighting"),
        ("epsilon_greedy/hierarchical_clustered/final_analysis_only/no_stopping",
         "frequentist_adaptive_weighting"),
        ("epsilon_greedy/repeated_measures/fixed_interim_looks/no_stopping",
         "frequentist_adaptive_weighting"),
        (GREEDY_CELL, "anytime_valid_frequentist"),
    ):
        refused(
            lambda cell_id=cell_id, paradigm=paradigm: procedure.transfer_guarantee(
                cell_id, paradigm
            ),
            "adaptive_inference.guarantee_not_transferable",
        )


def test_the_identity_changes_with_the_declared_dependence_at_unchanged_shape():
    def at(rho: float) -> str:
        return declare(
            assignment=EVEN,
            enrolment=(10,),
            structure=rm(4, 1.0, rho),
            analysis=ai.FinalAnalysis.cluster_robust(),
        ).identity

    assert at(0.3) == at(0.3)
    assert at(0.3) != at(0.4)
