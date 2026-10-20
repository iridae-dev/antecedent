//! Posterior of an adaptive repeated-measures trial against independent oracles.
//!
//! * The conjugate arm posteriors are checked against a brute-force grid integration of the
//!   *full* multivariate-normal likelihood over the arm mean (400001 points), which does not
//!   use the generalized-least-squares sufficiency the implementation relies on.
//! * The likelihood principle is checked two ways: the posterior is bit-identical under different
//!   recorded assignment probabilities and unit orders, and a simulation-based calibration run of
//!   an epsilon-greedy adaptive procedure (arm means drawn from the analysis prior) has the
//!   nominal credible coverage of the true contrast, with no correction for the adaptation.
//! * The frequentist side is not exchangeable with this statement: the fixed-sample analysis after
//!   adaptive assignment is refused by `admit_repeated_measures` (see
//!   `a_fixed_sample_analysis_after_adaptive_assignment_is_refused` in
//!   `adaptive_repeated_measures_oracle.rs`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CausalRng;
use antecedent_stats::adaptive_inference::{
    AdaptiveInferenceError as Error, NormalMeanPrior, OutcomeStructure,
    RepeatedMeasuresPosteriorRequest, SelectionMechanism, UnitRecord, WithinUnitCorrelation,
    repeated_measures_posterior,
};

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!((actual - expected).abs() <= tolerance, "{what}: {actual} vs {expected}");
}

const ARM0: [[f64; 3]; 3] = [[0.4, 1.1, 0.2], [1.0, 0.3, 0.9], [-0.5, 0.2, 0.6]];
const ARM1: [[f64; 3]; 4] = [[1.9, 2.4, 1.2], [2.8, 1.7, 2.2], [0.9, 1.5, 2.6], [2.1, 3.0, 1.8]];

fn autoregressive() -> OutcomeStructure {
    OutcomeStructure::RepeatedMeasures {
        measurements: 3,
        variance: 2.0,
        correlation: Some(WithinUnitCorrelation::Autoregressive { rho: 0.5 }),
    }
}

fn random_effect() -> OutcomeStructure {
    OutcomeStructure::Hierarchical {
        members: 3,
        between_variance: Some(1.0),
        within_variance: Some(2.0),
    }
}

fn units(probability: f64) -> Vec<UnitRecord<'static>> {
    let mut out = Vec::new();
    for row in &ARM0 {
        out.push(UnitRecord { arm: 0, measurements: row, assignment_probability: probability });
    }
    for row in &ARM1 {
        out.push(UnitRecord { arm: 1, measurements: row, assignment_probability: probability });
    }
    out
}

const PRIORS: [NormalMeanPrior; 2] =
    [NormalMeanPrior { mean: -0.2, variance: 2.0 }, NormalMeanPrior { mean: 0.5, variance: 3.0 }];

fn request<'a>(
    structure: &'a OutcomeStructure,
    units: &'a [UnitRecord<'a>],
    selection: SelectionMechanism,
) -> RepeatedMeasuresPosteriorRequest<'a> {
    RepeatedMeasuresPosteriorRequest {
        structure,
        selection,
        priors: &PRIORS,
        units,
        plus: 1,
        minus: 0,
        credible_level: 0.9,
    }
}

fn detail(error: Error) -> &'static str {
    match error {
        Error::PremiseFailed { detail, .. } => detail,
        other => panic!("expected a premise failure, got {other:?}"),
    }
}

#[test]
fn autoregressive_posterior_matches_full_likelihood_grid_integration() {
    let data = units(0.5);
    let structure = autoregressive();
    let posterior =
        repeated_measures_posterior(&request(&structure, &data, SelectionMechanism::Fixed))
            .unwrap();
    close(posterior.arms[1].mean, 1.845_454_545_455, 1e-9, "arm 1 mean");
    close(posterior.arms[1].standard_deviation, 0.522_232_967_867, 1e-9, "arm 1 sd");
    close(posterior.arms[0].mean, 0.344_444_444_444, 1e-9, "arm 0 mean");
    close(posterior.arms[0].standard_deviation, 0.577_350_269_190, 1e-9, "arm 0 sd");
    let c = posterior.contrast;
    close(c.mean, 1.501_010_101_010, 1e-9, "contrast mean");
    close(c.standard_deviation, 0.778_498_944_162, 1e-9, "contrast sd");
    close(c.interval[0], 0.220_493_289_128, 1e-8, "5%");
    close(c.interval[1], 2.781_526_912_892, 1e-8, "95%");
    close(c.probability_positive, 0.973_077_560_187, 1e-9, "P(> 0)");
    // GLS unit-summary variance of an AR(1) triple: sigma^2 (1 + rho) / (2 + (m - 2)(1 - rho)).
    close(posterior.unit_summary_variance, 2.0 * 1.5 / 2.5, 1e-12, "unit summary variance");
    assert_eq!(posterior.arms[0].units, 3);
    assert_eq!(posterior.arms[1].units, 4);
}

#[test]
fn random_effect_cluster_posterior_matches_full_likelihood_grid_integration() {
    let data = units(0.5);
    let structure = random_effect();
    let posterior =
        repeated_measures_posterior(&request(&structure, &data, SelectionMechanism::Fixed))
            .unwrap();
    close(posterior.arms[1].mean, 1.824_390_243_902, 1e-9, "arm 1 mean");
    close(posterior.arms[1].standard_deviation, 0.604_858_378_909, 1e-9, "arm 1 sd");
    close(posterior.arms[0].mean, 0.321_739_130_435, 1e-9, "arm 0 mean");
    let c = posterior.contrast;
    close(c.mean, 1.502_651_113_468, 1e-9, "contrast mean");
    close(c.standard_deviation, 0.894_782_804_502, 1e-8, "contrast sd");
    close(c.interval[0], 0.030_864_372_148, 1e-8, "5%");
    close(c.interval[1], 2.974_437_854_786, 1e-8, "95%");
    close(c.probability_positive, 0.953_457_814_078, 1e-8, "P(> 0)");
    // tau^2 + sigma^2 / m = 1 + 2 / 3 is the variance of a cluster mean.
    close(posterior.unit_summary_variance, 1.0 + 2.0 / 3.0, 1e-12, "cluster mean variance");
}

#[test]
fn posterior_ignores_the_assignment_mechanism_but_retains_its_evidence() {
    let structure = autoregressive();
    let fixed = units(0.5);
    let adaptive = units(0.17);
    let a = repeated_measures_posterior(&request(&structure, &fixed, SelectionMechanism::Fixed))
        .unwrap();
    let b = repeated_measures_posterior(&request(
        &structure,
        &adaptive,
        SelectionMechanism::DependsOnObservedHistory,
    ))
    .unwrap();
    assert_eq!(a.arms, b.arms);
    assert_eq!(a.contrast, b.contrast);
    assert!(!a.evidence.adaptive && b.evidence.adaptive);
    assert!(b.evidence.assignment_probabilities.iter().all(|p| (p - 0.17).abs() < 1e-15));
    assert_eq!(b.evidence.assignment_probabilities.len(), 7);
    assert_eq!(b.evidence.selection, SelectionMechanism::DependsOnObservedHistory);

    // Unit order is irrelevant: the data enter through arm sums.
    let mut reversed = fixed.clone();
    reversed.reverse();
    let c = repeated_measures_posterior(&request(&structure, &reversed, SelectionMechanism::Fixed))
        .unwrap();
    close(c.contrast.mean, a.contrast.mean, 1e-12, "order-free mean");
    close(c.contrast.standard_deviation, a.contrast.standard_deviation, 1e-12, "order-free sd");
}

fn standard_normal(rng: &mut CausalRng) -> f64 {
    let u1 = 1.0 - rng.next_f64();
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

#[test]
fn credible_coverage_is_nominal_under_epsilon_greedy_assignment_without_correction() {
    // Arm means drawn from the analysis prior N(0, 1); compound-symmetric triples with variance
    // 1.5 and correlation 0.3; 30 units sequentially assigned epsilon-greedy (epsilon 0.2) on the
    // running mean of the completed units' mean measurements.
    let structure = OutcomeStructure::RepeatedMeasures {
        measurements: 3,
        variance: 1.5,
        correlation: Some(WithinUnitCorrelation::CompoundSymmetric { rho: 0.3 }),
    };
    let priors = [NormalMeanPrior { mean: 0.0, variance: 1.0 }; 2];
    let (replicates, epsilon) = (600_usize, 0.2);
    let mut rng = CausalRng::from_seed(20_240_611);
    let (mut covered, mut used, mut adaptive_imbalance) = (0_usize, 0_usize, 0_usize);
    for _ in 0..replicates {
        let truth = [standard_normal(&mut rng), standard_normal(&mut rng)];
        let mut measurements: Vec<[f64; 3]> = Vec::new();
        let mut arms: Vec<usize> = Vec::new();
        let mut probabilities: Vec<f64> = Vec::new();
        let (mut sums, mut counts) = ([0.0_f64; 2], [0_usize; 2]);
        for _ in 0..30 {
            let means = [
                if counts[0] > 0 { sums[0] / counts[0] as f64 } else { 0.0 },
                if counts[1] > 0 { sums[1] / counts[1] as f64 } else { 0.0 },
            ];
            let (arm, probability) = if counts[0] == 0 || counts[1] == 0 {
                let arm = usize::from(rng.next_f64() < 0.5);
                (arm, 0.5)
            } else {
                let greedy = usize::from(means[1] > means[0]);
                let explore = rng.next_f64() < epsilon;
                let arm = if explore { usize::from(rng.next_f64() < 0.5) } else { greedy };
                (arm, if arm == greedy { 1.0 - epsilon / 2.0 } else { epsilon / 2.0 })
            };
            let shared = (0.3_f64 * 1.5).sqrt() * standard_normal(&mut rng);
            let row = [(); 3]
                .map(|()| truth[arm] + shared + (0.7_f64 * 1.5).sqrt() * standard_normal(&mut rng));
            sums[arm] += row.iter().sum::<f64>() / 3.0;
            counts[arm] += 1;
            measurements.push(row);
            arms.push(arm);
            probabilities.push(probability);
        }
        adaptive_imbalance += usize::from(counts[0].abs_diff(counts[1]) > 6);
        let records: Vec<UnitRecord<'_>> = measurements
            .iter()
            .zip(arms.iter().zip(&probabilities))
            .map(|(row, (arm, p))| UnitRecord {
                arm: *arm,
                measurements: row,
                assignment_probability: *p,
            })
            .collect();
        let request = RepeatedMeasuresPosteriorRequest {
            structure: &structure,
            selection: SelectionMechanism::DependsOnObservedHistory,
            priors: &priors,
            units: &records,
            plus: 1,
            minus: 0,
            credible_level: 0.9,
        };
        if let Ok(posterior) = repeated_measures_posterior(&request) {
            let target = truth[1] - truth[0];
            used += 1;
            covered += usize::from(
                posterior.contrast.interval[0] <= target
                    && target <= posterior.contrast.interval[1],
            );
        }
    }
    assert!(used >= replicates - 5, "{used}");
    let coverage = covered as f64 / used as f64;
    // Binomial standard error is 0.012 at 600 replicates; the band is four of them.
    close(coverage, 0.9, 0.05, "adaptive credible coverage");
    // The policy really adapted (arm counts are not balanced in a share of the runs).
    assert!(adaptive_imbalance > replicates / 10, "{adaptive_imbalance}");
}

#[test]
fn unobserved_dependence_missing_structure_and_unobserved_arms_are_refused() {
    let data = units(0.5);
    let structure = autoregressive();
    let error = repeated_measures_posterior(&request(
        &structure,
        &data,
        SelectionMechanism::DependsOnUnobserved,
    ))
    .unwrap_err();
    assert_eq!(detail(error), "adaptive_inference.selection_not_ignorable");

    let undeclared =
        OutcomeStructure::RepeatedMeasures { measurements: 3, variance: 2.0, correlation: None };
    let error =
        repeated_measures_posterior(&request(&undeclared, &data, SelectionMechanism::Fixed))
            .unwrap_err();
    assert_eq!(detail(error), "adaptive_inference.structure_incomplete");

    let only_arm_one: Vec<UnitRecord<'static>> = units(0.5).into_iter().skip(3).collect();
    let error =
        repeated_measures_posterior(&request(&structure, &only_arm_one, SelectionMechanism::Fixed))
            .unwrap_err();
    assert_eq!(detail(error), "adaptive_inference.arm_not_observed");

    let short = [1.0, 2.0];
    let wrong_length =
        [UnitRecord { arm: 0, measurements: &short, assignment_probability: 0.5 }; 2];
    assert!(matches!(
        repeated_measures_posterior(&request(&structure, &wrong_length, SelectionMechanism::Fixed)),
        Err(Error::Shape(_))
    ));
    let mut bad_level = request(&structure, &data, SelectionMechanism::Fixed);
    bad_level.credible_level = 1.0;
    assert!(matches!(repeated_measures_posterior(&bad_level), Err(Error::InvalidParameter(_))));
}
