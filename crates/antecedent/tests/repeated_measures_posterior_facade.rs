//! Posterior of an adaptive repeated-measures trial through the facade: the conjugate arm and
//! contrast posteriors against a hand calculation, the likelihood principle (the same posterior
//! under a fixed and an adaptive declared procedure, selection evidence retained and unused), the
//! typed refusals with their named next attempt, and the retained artifact with a recomputing
//! loader that refuses edits, resealed forgeries and another procedure.
//!
//! Oracle (pure arithmetic, no library code): two compound-symmetric measurements per unit with
//! variance 2 and correlation 1/2 give the unit-mean variance `v = 2 (1 + 1/2) / 2 = 1.5`. With
//! arm 0 units of mean 2 and 2 under the prior `N(0, 4)` and arm 1 units of mean 5, 6 and 4 under
//! `N(1, 2)`, arm `k` has precision `1 / prior_variance + n_k / v` and mean
//! `(prior_mean / prior_variance + sum_k / v) / precision`, so arm 0 is `N(1.68421052631579,
//! 0.79471941423903^2)`, arm 1 is `N(4.2, 0.63245553203368^2)` and the contrast is normal with
//! mean `2.51578947368421`, standard deviation `1.01566675015402`, 95% interval
//! `[0.52511922308750, 4.50645972428092]` and positive probability `0.99337509301353`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::adaptive_procedure_report::RepeatedMeasuresProcedure;
use antecedent::analysis::repeated_measures_posterior_report::{
    PosteriorUnit, REPEATED_MEASURES_POSTERIOR_ESTIMATOR_ID, RepeatedMeasuresPosteriorSpec,
    encode_repeated_measures_posterior_report, load_repeated_measures_posterior_report,
    seal_report_without_validation,
};
use antecedent_stats::adaptive_inference::{
    AnalysisEstimator, FinalAnalysis, NormalMeanPrior, OutcomeModelPrior, OutcomeStructure,
    RepeatedMeasuresDeclaration, SelectionMechanism, StoppingRule, UnitAssignment, UnitSummary,
    WeightScheme, WithinUnitCorrelation,
};

const TOL: f64 = 1e-9;

fn close(actual: f64, expected: f64, what: &str) {
    assert!((actual - expected).abs() <= TOL, "{what}: {actual} vs {expected}");
}

fn structure(variance: f64) -> OutcomeStructure {
    OutcomeStructure::RepeatedMeasures {
        measurements: 2,
        variance,
        correlation: Some(WithinUnitCorrelation::CompoundSymmetric { rho: 0.5 }),
    }
}

fn declaration(
    assignment: UnitAssignment,
    estimator: AnalysisEstimator,
    variance: f64,
) -> RepeatedMeasuresDeclaration {
    RepeatedMeasuresDeclaration {
        arms: 2,
        assignment: Some(assignment),
        enrolment: Some(vec![5]),
        lag_looks: 0,
        structure: Some(structure(variance)),
        analysis: Some(FinalAnalysis { estimator, plus: 1, minus: 0 }),
        stopping: Some(StoppingRule::NoStopping),
    }
}

fn fixed_procedure(variance: f64) -> RepeatedMeasuresProcedure {
    RepeatedMeasuresProcedure::declare(declaration(
        UnitAssignment::Fixed { probabilities: vec![0.5, 0.5] },
        AnalysisEstimator::ClusterRobust { summary: UnitSummary::Mean },
        variance,
    ))
    .expect("the declaration is valid")
}

fn adaptive_procedure() -> RepeatedMeasuresProcedure {
    RepeatedMeasuresProcedure::declare(declaration(
        UnitAssignment::EpsilonGreedy {
            epsilon: 0.4,
            prior: OutcomeModelPrior { mean: 0.0, strength: 1.0 },
        },
        AnalysisEstimator::AdaptivelyWeighted {
            summary: UnitSummary::Mean,
            scheme: WeightScheme::Uniform,
            prior: OutcomeModelPrior { mean: 0.0, strength: 1.0 },
            probability_floor: 0.05,
        },
        2.0,
    ))
    .expect("the declaration is valid")
}

fn unit(arm: usize, measurements: [f64; 2], probability: f64) -> PosteriorUnit {
    PosteriorUnit { arm, measurements: measurements.to_vec(), assignment_probability: probability }
}

fn spec(probability: [f64; 5]) -> RepeatedMeasuresPosteriorSpec {
    RepeatedMeasuresPosteriorSpec {
        priors: vec![
            NormalMeanPrior { mean: 0.0, variance: 4.0 },
            NormalMeanPrior { mean: 1.0, variance: 2.0 },
        ],
        units: vec![
            unit(0, [1.0, 3.0], probability[0]),
            unit(0, [2.0, 2.0], probability[1]),
            unit(1, [4.0, 6.0], probability[2]),
            unit(1, [5.0, 7.0], probability[3]),
            unit(1, [3.0, 5.0], probability[4]),
        ],
        selection: None,
        credible_level: Some(0.95),
    }
}

#[test]
fn the_posterior_matches_the_hand_calculation() {
    let report = fixed_procedure(2.0).posterior(&spec([0.5; 5])).expect("posterior");
    assert_eq!(report.estimator_id, REPEATED_MEASURES_POSTERIOR_ESTIMATOR_ID);
    assert_eq!(report.uncertainty, "posterior_equal_tailed_credible_interval");
    assert!(!report.licensed && !report.calibration_measured);
    assert_eq!(report.calibration, "unmeasured");
    close(report.unit_summary_variance, 1.5, "unit summary variance");
    close(report.arms[0].mean, 1.684_210_526_315_789_4, "arm 0 mean");
    close(report.arms[0].standard_deviation, 0.794_719_414_239_026_4, "arm 0 sd");
    close(report.arms[1].mean, 4.2, "arm 1 mean");
    close(report.arms[1].standard_deviation, 0.632_455_532_033_675_9, "arm 1 sd");
    assert_eq!((report.arms[0].units, report.arms[1].units), (2, 3));
    close(report.contrast.mean, 2.515_789_473_684_211, "contrast mean");
    close(report.contrast.standard_deviation, 1.015_666_750_154_016_4, "contrast sd");
    close(report.contrast.interval[0], 0.525_119_223_087_497_4, "interval lower");
    close(report.contrast.interval[1], 4.506_459_724_280_924, "interval upper");
    close(report.contrast.probability_positive, 0.993_375_093_013_53, "probability positive");
    assert_eq!(report.priors.len(), 2);
    assert!(report.estimand.contains("plus and minus arm means"));
}

#[test]
fn the_posterior_is_the_same_under_an_adaptive_procedure_and_other_assignment_probabilities() {
    let fixed = fixed_procedure(2.0).posterior(&spec([0.5; 5])).expect("fixed");
    let probabilities = [0.9, 0.2, 0.35, 0.8, 0.6];
    let adaptive = adaptive_procedure().posterior(&spec(probabilities)).expect("adaptive");
    assert_eq!(fixed.arms, adaptive.arms);
    assert_eq!(fixed.contrast, adaptive.contrast);
    assert!(!fixed.adaptive && adaptive.adaptive);
    assert_eq!(adaptive.selection, "depends_on_observed_history");
    assert_eq!(adaptive.assignment_probabilities, probabilities.to_vec());
    assert!(adaptive.selection_premise.contains("observed earlier data"));
}

#[test]
fn an_assignment_that_depends_on_an_unobserved_quantity_is_refused_with_the_next_attempt() {
    let mut request = spec([0.5; 5]);
    request.selection = Some(SelectionMechanism::DependsOnUnobserved);
    let refusal = fixed_procedure(2.0).posterior(&request).expect_err("refused");
    assert_eq!(refusal.code, antecedent_core::reason_code!("cell_not_licensed"));
    assert_eq!(refusal.detail, "adaptive_inference.selection_not_ignorable");
    assert!(refusal.message.contains("model the selection jointly"));
    assert!(refusal.message.contains("randomize"));
}

#[test]
fn a_fixed_selection_cannot_be_declared_for_an_adaptive_procedure() {
    let mut request = spec([0.5; 5]);
    request.selection = Some(SelectionMechanism::Fixed);
    let refusal = adaptive_procedure().posterior(&request).expect_err("refused");
    assert_eq!(refusal.code, antecedent_core::reason_code!("invalid_argument"));
    assert!(refusal.message.contains("adaptive assignment rule"));
}

#[test]
fn a_contrast_arm_without_a_unit_is_refused_because_a_prior_does_not_identify_it() {
    let mut request = spec([0.5; 5]);
    request.units.retain(|u| u.arm == 1);
    let refusal = fixed_procedure(2.0).posterior(&request).expect_err("refused");
    assert_eq!(refusal.code, antecedent_core::reason_code!("cell_not_licensed"));
    assert_eq!(refusal.detail, "adaptive_inference.arm_not_observed");
    assert!(refusal.message.contains("prior alone does not identify"));
}

#[test]
fn a_prior_variance_that_is_not_positive_is_invalid() {
    let mut request = spec([0.5; 5]);
    request.priors[0].variance = 0.0;
    let refusal = fixed_procedure(2.0).posterior(&request).expect_err("refused");
    assert_eq!(refusal.code, antecedent_core::reason_code!("invalid_argument"));
}

#[test]
fn the_artifact_round_trips_and_a_fresh_loader_recomputes_it() {
    let procedure = fixed_procedure(2.0);
    let report = procedure.posterior(&spec([0.5; 5])).expect("posterior");
    let text = encode_repeated_measures_posterior_report(&report, &procedure).expect("encode");
    assert!(text.contains("\"licensed\":false"));
    assert!(text.contains(REPEATED_MEASURES_POSTERIOR_ESTIMATOR_ID));
    // A loader that only holds the declaration (re-declared independently) reproduces the report.
    let loaded = load_repeated_measures_posterior_report(&text, &fixed_procedure(2.0))
        .expect("the independent loader recomputes the posterior");
    assert_eq!(loaded, report);
}

#[test]
fn an_edited_artifact_a_resealed_forgery_and_another_procedure_are_refused() {
    let procedure = fixed_procedure(2.0);
    let report = procedure.posterior(&spec([0.5; 5])).expect("posterior");
    let text = encode_repeated_measures_posterior_report(&report, &procedure).expect("encode");

    let edited = text.replace("\"probability_positive\":0.99", "\"probability_positive\":0.50");
    assert_ne!(edited, text);
    assert!(load_repeated_measures_posterior_report(&edited, &procedure).is_err());

    // A forger who recomputes the envelope digest is caught by the recomputation.
    let mut forged = report.clone();
    forged.contrast.mean += 1.0;
    let resealed =
        seal_report_without_validation(&forged).expect("the envelope accepts any finite report");
    let refusal = load_repeated_measures_posterior_report(&resealed, &procedure)
        .expect_err("the forged contrast is refused");
    assert!(refusal.message.contains("not the posterior of the retained units"));

    // The same retained report under a different declared variance is another procedure.
    let other = fixed_procedure(3.0);
    let refusal =
        load_repeated_measures_posterior_report(&text, &other).expect_err("another procedure");
    assert!(refusal.message.contains("different declared procedure"));

    // A claimed license is refused by the envelope.
    let claimed = text.replace("\"licensed\":false", "\"licensed\":true");
    assert!(load_repeated_measures_posterior_report(&claimed, &procedure).is_err());
}
