//! Derived uncertainty of repeated-measures and clustered adaptive procedures
//! through the facade: the declaration with one refusal per missing element
//! (the within-unit correlation among them), the closed forms and the
//! simulated whole-procedure law, the per-paradigm premise refusals, and the
//! refusal to pair a guarantee across cells.
//!
//! Oracles are independent: the compound-symmetric unit-mean variance
//! `sigma^2 (1 + (m - 1) rho) / m` (design effect `1 + (m - 1) rho`), the
//! enumerated binomial mixture of a two-unit fixed allocation, and the
//! martingale property of the adaptively weighted score.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::adaptive_inference::AdaptiveInferenceRefusal;
use antecedent::analysis::adaptive_procedure_report::{
    RepeatedMeasuresProcedure, RepeatedMeasuresReport, RepeatedMeasuresReportBody,
    RepeatedMeasuresReportRequest,
};
use antecedent_core::ExecutionContext;
use antecedent_stats::adaptive_inference::{
    AnalysisEstimator, DERIVED_UNMEASURED, FinalAnalysis, OutcomeModelPrior, OutcomeStructure,
    RepeatedMeasuresDeclaration, RepeatedMeasuresRequest, StoppingRule, UncertaintyParadigm,
    UnitAssignment, UnitSummary, WeightScheme, WithinUnitCorrelation,
};

const TOL: f64 = 1e-9;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(3)
}

fn structure(measurements: u32, rho: Option<f64>) -> OutcomeStructure {
    OutcomeStructure::RepeatedMeasures {
        measurements,
        variance: 2.0,
        correlation: rho.map(|rho| WithinUnitCorrelation::CompoundSymmetric { rho }),
    }
}

fn weighted() -> AnalysisEstimator {
    AnalysisEstimator::AdaptivelyWeighted {
        summary: UnitSummary::Mean,
        scheme: WeightScheme::Uniform,
        prior: OutcomeModelPrior { mean: 0.0, strength: 1.0 },
        probability_floor: 0.05,
    }
}

fn declaration(
    assignment: UnitAssignment,
    enrolment: Vec<u32>,
    structure: OutcomeStructure,
    estimator: AnalysisEstimator,
) -> RepeatedMeasuresDeclaration {
    RepeatedMeasuresDeclaration {
        arms: 2,
        assignment: Some(assignment),
        enrolment: Some(enrolment),
        lag_looks: 0,
        structure: Some(structure),
        analysis: Some(FinalAnalysis { estimator, plus: 1, minus: 0 }),
        stopping: Some(StoppingRule::NoStopping),
    }
}

fn even() -> UnitAssignment {
    UnitAssignment::Fixed { probabilities: vec![0.5, 0.5] }
}

fn greedy() -> UnitAssignment {
    UnitAssignment::EpsilonGreedy {
        epsilon: 0.4,
        prior: OutcomeModelPrior { mean: 0.0, strength: 1.0 },
    }
}

fn contrast(summary: UnitSummary) -> AnalysisEstimator {
    AnalysisEstimator::ClusterRobust { summary }
}

fn declared(d: RepeatedMeasuresDeclaration) -> RepeatedMeasuresProcedure {
    RepeatedMeasuresProcedure::declare(d).expect("the declaration is valid")
}

fn refused(
    procedure: &RepeatedMeasuresProcedure,
    request: &RepeatedMeasuresReportRequest,
) -> AdaptiveInferenceRefusal {
    procedure.report(request, &ctx()).expect_err("the request is refused")
}

fn derive(arm_means: &[f64], replications: u32) -> RepeatedMeasuresReportRequest {
    RepeatedMeasuresReportRequest::Derived {
        arm_means: arm_means.to_vec(),
        simulation: RepeatedMeasuresRequest { replications, seed: 13 },
        z: 1.96,
    }
}

fn report_of(
    procedure: &RepeatedMeasuresProcedure,
    request: &RepeatedMeasuresReportRequest,
) -> RepeatedMeasuresReport {
    procedure.report(request, &ctx()).expect("the report is derived")
}

#[test]
fn a_missing_within_unit_correlation_is_refused_with_its_own_detail_and_remedy() {
    let refusals = RepeatedMeasuresProcedure::declare(declaration(
        even(),
        vec![10],
        structure(4, None),
        contrast(UnitSummary::Mean),
    ))
    .expect_err("the correlation is not declared");
    assert_eq!(refusals.len(), 1);
    assert_eq!(refusals[0].code, antecedent_core::reason_code!("adaptive_contract_incomplete"));
    assert_eq!(refusals[0].detail, "adaptive_inference.within_unit_correlation_missing");
    assert!(refusals[0].message.contains("within-unit correlation"));
}

#[test]
fn an_invalid_element_is_refused_as_invalid_argument() {
    let refusals = RepeatedMeasuresProcedure::declare(declaration(
        even(),
        vec![10],
        structure(4, Some(1.0)),
        contrast(UnitSummary::Mean),
    ))
    .expect_err("rho = 1 is singular");
    assert_eq!(refusals[0].code, antecedent_core::reason_code!("invalid_argument"));
    assert_eq!(refusals[0].detail, "adaptive_inference.correlation_invalid");
}

#[test]
fn the_identity_changes_with_the_declared_dependence_at_unchanged_shape() {
    let at = |rho: f64| {
        declared(declaration(
            even(),
            vec![10],
            structure(4, Some(rho)),
            contrast(UnitSummary::Mean),
        ))
    };
    let a = at(0.3);
    assert_eq!(a.identity(), at(0.3).identity());
    assert_ne!(a.identity(), at(0.4).identity());
    // The same numbers under another outcome structure are another procedure.
    let clustered = declared(declaration(
        even(),
        vec![10],
        OutcomeStructure::Hierarchical {
            members: 4,
            between_variance: Some(0.3),
            within_variance: Some(1.0),
        },
        contrast(UnitSummary::Mean),
    ));
    assert_ne!(a.identity(), clustered.identity());
    assert!(!a.identity().is_empty());
}

#[test]
fn the_closed_forms_are_the_design_effect_and_the_enumerated_mixture() {
    // m = 4, sigma^2 = 2, rho = 0.3: Var(unit mean) = 2 * 1.9 / 4 = 0.95; two
    // units, a 50/50 allocation: exactly one in each arm with probability 1/2,
    // contrast variance 2 * 0.95 = 1.9.
    let procedure = declared(declaration(
        even(),
        vec![2],
        structure(4, Some(0.3)),
        contrast(UnitSummary::Mean),
    ));
    let report = report_of(
        &procedure,
        &RepeatedMeasuresReportRequest::ClosedForm { arm_means: vec![0.0, 0.5] },
    );
    assert_eq!(report.calibration, DERIVED_UNMEASURED);
    assert_eq!(report.route, "cluster_robust_unit_summaries");
    assert_eq!(
        report.cell.id(),
        "fixed_randomization/repeated_measures/final_analysis_only/no_stopping"
    );
    let RepeatedMeasuresReportBody::ClosedForm {
        marginal_variance,
        unit_summary_variance,
        design_effect,
        measurements,
        revealed_units,
        contrast_value,
        exact_law,
    } = report.body
    else {
        panic!("a closed-form body");
    };
    assert!((marginal_variance - 2.0).abs() < TOL);
    assert!((unit_summary_variance - 0.95).abs() < TOL);
    assert!((design_effect - 1.9).abs() < TOL);
    assert_eq!(measurements, 4);
    assert_eq!(revealed_units, 2);
    assert!((contrast_value - 0.5).abs() < TOL);
    let law = exact_law.expect("a fixed two-arm allocation has a closed form");
    assert_eq!(law.components.len(), 1);
    assert!((law.components[0].weight - 0.5).abs() < TOL);
    assert!((law.components[0].variance - 1.9).abs() < TOL);
    assert!((law.undefined_mass - 0.5).abs() < TOL);
}

#[test]
fn the_derived_fixed_sample_report_separates_exact_values_from_monte_carlo_error() {
    // 12 units, m = 3, sigma^2 = 2, rho = 0.5: v = 2 * 2 / 3.
    let procedure = declared(declaration(
        even(),
        vec![12],
        structure(3, Some(0.5)),
        contrast(UnitSummary::Mean),
    ));
    let report = report_of(&procedure, &derive(&[0.0, 0.25], 4_000));
    let RepeatedMeasuresReportBody::Derived(derived) = report.body else {
        panic!("a derived body");
    };
    assert_eq!(derived.calibration, DERIVED_UNMEASURED);
    assert_eq!(derived.paradigm, UncertaintyParadigm::FrequentistFixedSample);
    assert!((derived.unit_summary_variance - 4.0 / 3.0).abs() < TOL);
    assert!((derived.design_effect - 2.0).abs() < TOL);
    let exact = derived.exact_law.expect("closed form for a fixed allocation");
    let simulated = derived.simulated;
    assert!(simulated.monte_carlo_standard_error > 0.0);
    assert!(
        simulated.bias.abs() <= 5.0 * simulated.monte_carlo_standard_error,
        "bias {} mcse {}",
        simulated.bias,
        simulated.monte_carlo_standard_error
    );
    assert!(
        (simulated.variance / exact.variance() - 1.0).abs() < 0.1,
        "simulated {} exact {}",
        simulated.variance,
        exact.variance()
    );
}

#[test]
fn the_adaptive_weighting_route_is_labelled_and_unbiased() {
    let procedure =
        declared(declaration(greedy(), vec![1; 30], structure(3, Some(0.4)), weighted()));
    assert_eq!(
        procedure.cell().id(),
        "epsilon_greedy/repeated_measures/final_analysis_only/no_stopping"
    );
    let report = report_of(&procedure, &derive(&[0.0, 0.5], 2_000));
    assert_eq!(report.route, "adaptively_weighted_unit_scores");
    assert_eq!(report.calibration, DERIVED_UNMEASURED);
    let RepeatedMeasuresReportBody::Derived(derived) = report.body else {
        panic!("a derived body");
    };
    assert_eq!(derived.paradigm, UncertaintyParadigm::FrequentistAdaptiveWeighting);
    assert!(derived.exact_law.is_none());
    assert!(
        derived.simulated.bias.abs() <= 5.0 * derived.simulated.monte_carlo_standard_error,
        "bias {} mcse {}",
        derived.simulated.bias,
        derived.simulated.monte_carlo_standard_error
    );
    assert_eq!(derived.enrolled, 30);
}

#[test]
fn a_fixed_sample_analysis_after_adaptive_assignment_is_refused_by_the_facade() {
    let procedure = declared(declaration(
        greedy(),
        vec![1; 20],
        structure(3, Some(0.4)),
        contrast(UnitSummary::Mean),
    ));
    for request in [
        derive(&[0.0, 0.5], 10),
        RepeatedMeasuresReportRequest::Paradigm {
            paradigm: UncertaintyParadigm::FrequentistFixedSample,
            arm_means: vec![0.0, 0.5],
            simulation: RepeatedMeasuresRequest { replications: 10, seed: 1 },
            z: 1.96,
        },
    ] {
        let refusal = refused(&procedure, &request);
        assert_eq!(refusal.code, antecedent_core::reason_code!("cell_not_licensed"));
        assert_eq!(refusal.detail, "adaptive_inference.fixed_sample_design_adaptive");
        assert!(refusal.message.contains("remedy"), "{}", refusal.message);
    }
}

#[test]
fn an_independence_assumption_for_dependent_measurements_is_refused() {
    let procedure = declared(declaration(
        even(),
        vec![10],
        structure(4, Some(0.3)),
        AnalysisEstimator::PooledIndependent,
    ));
    let refusal = refused(&procedure, &derive(&[0.0, 0.0], 10));
    assert_eq!(refusal.code, antecedent_core::reason_code!("cell_not_licensed"));
    assert_eq!(
        refusal.detail,
        "adaptive_inference.independence_assumed_for_dependent_measurements"
    );
}

#[test]
fn a_paradigm_that_is_not_derived_for_this_cell_is_refused() {
    let procedure =
        declared(declaration(greedy(), vec![1; 20], structure(3, Some(0.4)), weighted()));
    let refusal = refused(
        &procedure,
        &RepeatedMeasuresReportRequest::Paradigm {
            paradigm: UncertaintyParadigm::AnytimeValidFrequentist,
            arm_means: vec![0.0, 0.5],
            simulation: RepeatedMeasuresRequest { replications: 10, seed: 1 },
            z: 1.96,
        },
    );
    assert_eq!(refusal.code, antecedent_core::reason_code!("cell_not_licensed"));
    assert_eq!(refusal.detail, "adaptive_inference.repeated_measures_paradigm_not_derived");
}

#[test]
fn a_guarantee_is_not_paired_across_cells() {
    let procedure =
        declared(declaration(greedy(), vec![1; 20], structure(3, Some(0.4)), weighted()));
    let same = RepeatedMeasuresReportRequest::CellTransfer {
        cell_id: "epsilon_greedy/repeated_measures/final_analysis_only/no_stopping".to_owned(),
        paradigm: "frequentist_adaptive_weighting".to_owned(),
    };
    let accepted = report_of(&procedure, &same);
    assert!(matches!(accepted.body, RepeatedMeasuresReportBody::CellTransfer { .. }));
    for (cell_id, paradigm) in [
        (
            "epsilon_greedy/single_outcome/final_analysis_only/no_stopping",
            "frequentist_adaptive_weighting",
        ),
        (
            "epsilon_greedy/hierarchical_clustered/final_analysis_only/no_stopping",
            "frequentist_adaptive_weighting",
        ),
        (
            "epsilon_greedy/repeated_measures/fixed_interim_looks/no_stopping",
            "frequentist_adaptive_weighting",
        ),
        (
            "epsilon_greedy/repeated_measures/final_analysis_only/no_stopping",
            "anytime_valid_frequentist",
        ),
    ] {
        let refusal = refused(
            &procedure,
            &RepeatedMeasuresReportRequest::CellTransfer {
                cell_id: cell_id.to_owned(),
                paradigm: paradigm.to_owned(),
            },
        );
        assert_eq!(refusal.code, antecedent_core::reason_code!("cell_not_licensed"));
        assert_eq!(refusal.detail, "adaptive_inference.guarantee_not_transferable", "{cell_id}");
    }
}

#[test]
fn a_clustered_outcome_has_its_own_cell_and_closed_form() {
    // tau^2 = 0.5, sigma^2 = 1.5, m = 5: Var(cluster mean) = 0.8, design effect 2.
    let procedure = declared(declaration(
        even(),
        vec![2],
        OutcomeStructure::Hierarchical {
            members: 5,
            between_variance: Some(0.5),
            within_variance: Some(1.5),
        },
        contrast(UnitSummary::Mean),
    ));
    assert_eq!(
        procedure.cell().id(),
        "fixed_randomization/hierarchical_clustered/final_analysis_only/no_stopping"
    );
    let report = report_of(
        &procedure,
        &RepeatedMeasuresReportRequest::ClosedForm { arm_means: vec![1.0, 1.0] },
    );
    let RepeatedMeasuresReportBody::ClosedForm { unit_summary_variance, design_effect, .. } =
        report.body
    else {
        panic!("a closed-form body");
    };
    assert!((unit_summary_variance - 0.8).abs() < TOL);
    assert!((design_effect - 2.0).abs() < TOL);
}
