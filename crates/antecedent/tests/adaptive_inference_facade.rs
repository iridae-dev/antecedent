//! Inference from adaptive experiments through the facade: simulated trials
//! with a known truth, every paradigm's route, the refusals that keep the
//! final sample from being analysed as if the design were fixed, and the
//! artifacts a fresh consumer replays.
//!
//! Simulations use fixed seeds and tolerances written in terms of the Monte
//! Carlo standard error. Boundary constants are independent: published
//! values, or an independent grid recursion (`SciPy`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_range_loop, reason = "assertions index per-arm columns by arm")]

use antecedent::analysis::adaptive_inference::{
    AdaptiveDesignDeclaration, AdaptiveInferenceRefusal, AdaptiveInferenceSession, AdaptiveRequest,
    AssignmentFamily, BoundarySidedness, DelayMechanism, InferenceParadigm, InterimAnalysis,
    MonitoringStrategy, OutcomeAvailability, OutcomeStructure, PolicyEpoch, ProcedureCell,
    SpendingKind, StoppingRuleDeclaration, StoppingRuleKind, StoppingSpec,
    consume_adaptive_inference_result_artifact, consume_adaptive_monitoring_contract_artifact,
};
use antecedent_core::{CausalRng, ExecutionContext, SemanticDigest};
use antecedent_kernels::standard_normal;
use antecedent_stats::adaptive_inference::{BetaPosterior, best_arm_probabilities};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(3)
}

fn looks_for(sizes: &[usize], total: usize) -> (Vec<InterimAnalysis>, Vec<Option<u32>>) {
    let looks = sizes
        .iter()
        .map(|&n| InterimAnalysis {
            units_enrolled: u32::try_from(n).unwrap(),
            information_fraction: n as f64 / total as f64,
        })
        .collect();
    let availability = (0..total)
        .map(|unit| {
            let look = sizes.iter().position(|&n| unit < n).unwrap();
            Some(u32::try_from(look).unwrap())
        })
        .collect();
    (looks, availability)
}

#[allow(
    clippy::too_many_arguments,
    reason = "a test fixture builder takes each column of the declaration"
)]
fn skeleton(
    label: &str,
    cell: ProcedureCell,
    arms: Vec<u32>,
    probabilities: Vec<Vec<f64>>,
    outcomes: Vec<Vec<f64>>,
    sizes: &[usize],
    rule: StoppingRuleDeclaration,
    floor: f64,
) -> AdaptiveDesignDeclaration {
    let total = arms.len();
    let (looks, available_at_look) = looks_for(sizes, total);
    AdaptiveDesignDeclaration {
        label: label.to_owned(),
        cell,
        arm_labels: vec!["control".to_owned(), "treatment".to_owned()],
        assigned_arm: arms.clone(),
        policy_history: Some(vec![PolicyEpoch {
            first_unit: 0,
            policy_identity: SemanticDigest::from_bytes([u8::try_from(label.len()).unwrap(); 32]),
        }]),
        assignment_probabilities: Some(probabilities),
        exposure: Some(arms),
        availability: Some(OutcomeAvailability {
            available_at_look,
            delay: DelayMechanism::Immediate,
        }),
        outcomes,
        interim_analyses: Some(looks),
        stopping_rule: Some(rule),
        probability_floor: floor,
    }
}

/// Beta-Bernoulli Thompson sampling with uniform priors.
fn thompson_trial(
    truth: [f64; 2],
    total: usize,
    sizes: &[usize],
    seed: u64,
) -> AdaptiveDesignDeclaration {
    let mut rng = CausalRng::from_seed(seed);
    let mut counts = [[0_u64; 2]; 2];
    let (mut arms, mut rows, mut outcomes) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..total {
        let posteriors = [0, 1]
            .map(|a| BetaPosterior::from_counts(1.0, 1.0, counts[a][0], counts[a][1]).unwrap());
        let best = best_arm_probabilities(&posteriors, &|| false).unwrap();
        let sum = best[0] + best[1];
        let e = [best[0] / sum, best[1] / sum];
        let arm = usize::from(rng.next_f64() >= e[0]);
        let success = rng.next_f64() < truth[arm];
        counts[arm][usize::from(!success)] += 1;
        arms.push(u32::try_from(arm).unwrap());
        rows.push(e.to_vec());
        outcomes.push(vec![f64::from(u8::from(success))]);
    }
    let cell = ProcedureCell {
        assignment: AssignmentFamily::ThompsonBetaBernoulli,
        outcome: OutcomeStructure::SingleOutcome,
        monitoring: MonitoringStrategy::FixedInterimLooks,
        stopping: StoppingRuleKind::Efficacy,
    };
    let rule = StoppingRuleDeclaration {
        kind: StoppingRuleKind::Efficacy,
        spec: StoppingSpec::PosteriorProbabilityBest { threshold: 0.95 },
        ended_by_rule: false,
    };
    skeleton("thompson", cell, arms, rows, outcomes, sizes, rule, 1e-6)
}

/// Epsilon-greedy over running arm means with Bernoulli outcomes.
fn epsilon_greedy_trial(
    truth: [f64; 2],
    epsilon: f64,
    total: usize,
    sizes: &[usize],
    seed: u64,
) -> AdaptiveDesignDeclaration {
    let mut rng = CausalRng::from_seed(seed);
    let (mut sum, mut count) = ([0.0_f64; 2], [0.0_f64; 2]);
    let (mut arms, mut rows, mut outcomes) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..total {
        let means = [0, 1].map(|a| if count[a] > 0.0 { sum[a] / count[a] } else { 0.5 });
        let mut e = [epsilon / 2.0; 2];
        if (means[0] - means[1]).abs() < f64::EPSILON {
            e = [0.5, 0.5];
        } else {
            e[usize::from(means[1] > means[0])] += 1.0 - epsilon;
        }
        let arm = usize::from(rng.next_f64() >= e[0]);
        let y = f64::from(u8::from(rng.next_f64() < truth[arm]));
        sum[arm] += y;
        count[arm] += 1.0;
        arms.push(u32::try_from(arm).unwrap());
        rows.push(e.to_vec());
        outcomes.push(vec![y]);
    }
    let cell = ProcedureCell {
        assignment: AssignmentFamily::EpsilonGreedy,
        outcome: OutcomeStructure::SingleOutcome,
        monitoring: MonitoringStrategy::FixedInterimLooks,
        stopping: StoppingRuleKind::Efficacy,
    };
    let rule = StoppingRuleDeclaration {
        kind: StoppingRuleKind::Efficacy,
        spec: StoppingSpec::PosteriorProbabilityBest { threshold: 0.95 },
        ended_by_rule: false,
    };
    skeleton("epsilon_greedy", cell, arms, rows, outcomes, sizes, rule, 0.05)
}

/// Balanced fixed randomization with normal outcomes of sd one.
fn group_sequential_trial(
    effect: f64,
    total: usize,
    sizes: &[usize],
    seed: u64,
) -> AdaptiveDesignDeclaration {
    let mut rng = CausalRng::from_seed(seed);
    let arms: Vec<u32> = (0..total).map(|unit| u32::from(unit % 2 == 1)).collect();
    let outcomes =
        arms.iter().map(|&arm| vec![effect * f64::from(arm) + standard_normal(&mut rng)]).collect();
    let cell = ProcedureCell {
        assignment: AssignmentFamily::FixedRandomization,
        outcome: OutcomeStructure::SingleOutcome,
        monitoring: MonitoringStrategy::GroupSequential,
        stopping: StoppingRuleKind::Efficacy,
    };
    let rule = StoppingRuleDeclaration {
        kind: StoppingRuleKind::Efficacy,
        spec: StoppingSpec::AlphaSpending {
            spending: SpendingKind::OBrienFlemingType,
            total_alpha: 0.05,
            sidedness: BoundarySidedness::TwoSidedSymmetric,
        },
        ended_by_rule: false,
    };
    skeleton(
        "group_sequential",
        cell,
        arms,
        vec![vec![0.5, 0.5]; total],
        outcomes,
        sizes,
        rule,
        0.5,
    )
}

fn session(declaration: AdaptiveDesignDeclaration) -> AdaptiveInferenceSession {
    AdaptiveInferenceSession::prepare(declaration).expect("complete contract")
}

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{what}: expected {expected}, got {actual} (tolerance {tolerance})"
    );
}

// -- Bayesian posterior ---------------------------------------------------------------------

#[test]
fn the_posterior_of_a_thompson_trial_is_the_conjugate_posterior_of_its_counts() {
    let declaration = thompson_trial([0.3, 0.6], 120, &[40, 80, 120], 17);
    // Independent counts straight from the retained history.
    let mut counts = [[0_u32; 2]; 2];
    for (arm, outcome) in declaration.assigned_arm.iter().zip(&declaration.outcomes) {
        counts[*arm as usize][usize::from(outcome[0] < 0.5)] += 1;
    }
    let session = session(declaration);
    let request = AdaptiveRequest::BayesianPosterior { look: 2, prior_alpha: 1.0, prior_beta: 1.0 };
    let outcome = session.analyse(&request, &ctx()).unwrap();
    assert_eq!(outcome.paradigm, InferenceParadigm::BayesianPosterior);
    let v = &outcome.result.values;
    for arm in 0..2 {
        let (alpha, beta) = (1.0 + f64::from(counts[arm][0]), 1.0 + f64::from(counts[arm][1]));
        close(v[arm * 6], alpha, 0.0, "alpha");
        close(v[arm * 6 + 1], beta, 0.0, "beta");
        close(v[arm * 6 + 2], alpha / (alpha + beta), 1e-15, "mean");
        close(
            v[arm * 6 + 3],
            alpha * beta / ((alpha + beta).powi(2) * (alpha + beta + 1.0)),
            1e-15,
            "variance",
        );
    }
    close(v[4] + v[10], 1.0, 1e-9, "probabilities of best sum to one");
    // The better arm is identified and its mean is near the truth.
    assert!(v[10] > 0.9, "probability that the treatment is best {}", v[10]);
    close(v[8], 0.6, 0.12, "posterior mean of the treatment arm");
    // The contract's own accounting: each observation counted once.
    let evidence = session.interim_evidence().unwrap();
    assert_eq!(evidence.mass.declared_slots, 40 + 80 + 120);
    assert_eq!(evidence.mass.distinct_observations, 120);
}

#[test]
fn a_posterior_stopping_decision_follows_the_declared_expected_loss_rule() {
    let mut declaration = thompson_trial([0.2, 0.8], 100, &[50, 100], 4);
    declaration.cell.stopping = StoppingRuleKind::DecisionTheoretic;
    declaration.stopping_rule = Some(StoppingRuleDeclaration {
        kind: StoppingRuleKind::DecisionTheoretic,
        spec: StoppingSpec::PosteriorExpectedLoss { loss_threshold: 0.05 },
        ended_by_rule: true,
    });
    let session = session(declaration);
    let request = AdaptiveRequest::PosteriorStopping { look: 1, prior_alpha: 1.0, prior_beta: 1.0 };
    let outcome = session.analyse(&request, &ctx()).unwrap();
    assert_eq!(outcome.paradigm, InferenceParadigm::DecisionTheoreticStopping);
    let v = &outcome.result.values;
    // [best arm, probability best, expected loss, stop]: arm 1 is much better
    // and the expected loss of choosing it is far below the declared 0.05.
    close(v[0], 1.0, 0.0, "best arm");
    assert!(v[1] > 0.99 && v[2] < 0.05, "probability best {} loss {}", v[1], v[2]);
    close(v[3], 1.0, 0.0, "stop");
    assert_eq!(outcome.result.closed_guarantee, "adaptive_inference.stopping_reliability_closed");
}

#[test]
fn decision_theoretic_stopping_is_not_licensed_for_a_probability_best_rule() {
    let session = session(thompson_trial([0.2, 0.8], 100, &[50, 100], 4));
    let request = AdaptiveRequest::PosteriorStopping { look: 1, prior_alpha: 1.0, prior_beta: 1.0 };
    let refusal = session.analyse(&request, &ctx()).unwrap_err();
    assert_eq!(refusal.code, "cell_not_licensed");
    assert_eq!(refusal.detail, "adaptive_inference.paradigm_not_licensed_for_cell");
}

// -- the paradigm gate ----------------------------------------------------------------------

#[test]
fn fixed_sample_inference_of_adaptive_data_is_refused_while_the_posterior_runs() {
    let session = session(thompson_trial([0.3, 0.6], 80, &[40, 80], 1));
    let refusal: AdaptiveInferenceRefusal = session.fixed_sample().unwrap_err();
    assert_eq!(refusal.code, "adaptive_design_ignored");
    assert_eq!(refusal.detail, "adaptive_inference.fixed_sample_on_adaptive_data");
    let request = AdaptiveRequest::BayesianPosterior { look: 1, prior_alpha: 1.0, prior_beta: 1.0 };
    assert!(session.analyse(&request, &ctx()).is_ok());
    // No interval is licensed for the whole procedure.
    let interval = session.interval().unwrap_err();
    assert_eq!(interval.code, "cell_not_licensed");
    assert_eq!(interval.detail, "adaptive_inference.interval_closed");
}

#[test]
fn each_missing_element_is_refused_at_prepare_with_its_own_detail() {
    let mut declaration = thompson_trial([0.3, 0.6], 40, &[20, 40], 2);
    declaration.policy_history = None;
    declaration.availability = None;
    let refusals = AdaptiveInferenceSession::prepare(declaration).unwrap_err();
    let details: Vec<&str> = refusals.iter().map(|r| r.detail.as_str()).collect();
    assert_eq!(
        details,
        [
            "adaptive_inference.policy_history_missing",
            "adaptive_inference.outcome_availability_missing"
        ]
    );
    assert!(refusals.iter().all(|r| r.code == "adaptive_contract_incomplete"));
}

// -- adaptively weighted scores --------------------------------------------------------------

#[test]
fn epsilon_greedy_trials_recover_the_contrast_over_seeded_replicates() {
    let weighting = |scheme: &str| AdaptiveRequest::AdaptiveWeighting {
        look: 1,
        plus_arm: 1,
        minus_arm: Some(0),
        scheme: scheme.to_owned(),
        prior_mean: 0.5,
        prior_strength: 1.0,
        size_weighted: false,
    };
    let replicates = 120_u64;
    let mut estimates = Vec::new();
    let mut variances = Vec::new();
    for replicate in 0..replicates {
        let declaration =
            epsilon_greedy_trial([0.4, 0.6], 0.3, 200, &[100, 200], 9_000 + replicate);
        let session = session(declaration);
        let outcome = session.analyse(&weighting("variance_stabilizing"), &ctx()).unwrap();
        estimates.push(outcome.result.values[0]);
        variances.push(outcome.result.values[1]);
    }
    let n = replicates as f64;
    let mean = estimates.iter().sum::<f64>() / n;
    let empirical = estimates.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let reported = variances.iter().sum::<f64>() / n;
    // Monte Carlo standard error of the mean is about 0.1 / sqrt(120) = 0.009.
    close(mean, 0.2, 0.04, "mean of the weighted contrast");
    let ratio = reported / empirical;
    assert!((0.7..1.4).contains(&ratio), "reported over empirical variance {ratio}");
}

#[test]
fn the_probability_floor_is_enforced_and_never_clipped() {
    let mut declaration = epsilon_greedy_trial([0.4, 0.6], 0.3, 60, &[30, 60], 5);
    declaration.probability_floor = 0.4;
    let session = session(declaration);
    let request = AdaptiveRequest::AdaptiveWeighting {
        look: 1,
        plus_arm: 1,
        minus_arm: Some(0),
        scheme: "uniform".to_owned(),
        prior_mean: 0.5,
        prior_strength: 1.0,
        size_weighted: false,
    };
    let refusal = session.analyse(&request, &ctx()).unwrap_err();
    assert_eq!(refusal.code, "adaptive_assumption_violated");
    assert_eq!(refusal.detail, "adaptive_inference.positivity_floor");
}

// -- repeated measurements --------------------------------------------------------------------

/// Subjects with `measurements` repeated measurements, a random subject
/// intercept (sd 0.3) and noise (sd 0.5), epsilon-greedy over subject means.
fn repeated_measures_trial(truth: [f64; 2], total: usize, seed: u64) -> AdaptiveDesignDeclaration {
    let mut rng = CausalRng::from_seed(seed);
    let epsilon = 0.3;
    let (mut sum, mut count) = ([0.0_f64; 2], [0.0_f64; 2]);
    let (mut arms, mut rows, mut outcomes) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..total {
        let means = [0, 1].map(|a| if count[a] > 0.0 { sum[a] / count[a] } else { 0.0 });
        let mut e = [epsilon / 2.0; 2];
        if (means[0] - means[1]).abs() < f64::EPSILON {
            e = [0.5, 0.5];
        } else {
            e[usize::from(means[1] > means[0])] += 1.0 - epsilon;
        }
        let arm = usize::from(rng.next_f64() >= e[0]);
        let intercept = 0.3 * standard_normal(&mut rng);
        let measurements: Vec<f64> =
            (0..3).map(|_| truth[arm] + intercept + 0.5 * standard_normal(&mut rng)).collect();
        let subject_mean = measurements.iter().sum::<f64>() / 3.0;
        sum[arm] += subject_mean;
        count[arm] += 1.0;
        arms.push(u32::try_from(arm).unwrap());
        rows.push(e.to_vec());
        outcomes.push(measurements);
    }
    let cell = ProcedureCell {
        assignment: AssignmentFamily::EpsilonGreedy,
        outcome: OutcomeStructure::RepeatedMeasures,
        monitoring: MonitoringStrategy::FinalAnalysisOnly,
        stopping: StoppingRuleKind::NoStopping,
    };
    let rule = StoppingRuleDeclaration {
        kind: StoppingRuleKind::NoStopping,
        spec: StoppingSpec::NoStopping,
        ended_by_rule: false,
    };
    skeleton("repeated_measures", cell, arms, rows, outcomes, &[total], rule, 0.05)
}

fn contrast_request(size_weighted: bool) -> AdaptiveRequest {
    AdaptiveRequest::AdaptiveWeighting {
        look: 0,
        plus_arm: 1,
        minus_arm: Some(0),
        scheme: "uniform".to_owned(),
        prior_mean: 0.0,
        prior_strength: 1.0,
        size_weighted,
    }
}

#[test]
fn repeated_measures_subjects_are_the_unit_and_the_contrast_recovers_the_truth() {
    let replicates = 100_u64;
    let (mut estimates, mut variances) = (Vec::new(), Vec::new());
    for replicate in 0..replicates {
        let declaration = repeated_measures_trial([0.0, 0.4], 150, 4_000 + replicate);
        if replicate == 0 {
            // Reversing each subject's measurement order changes the
            // identity of the data but not the subject mean or the estimate.
            let mut reversed = declaration.clone();
            for measurements in &mut reversed.outcomes {
                measurements.reverse();
            }
            let a = session(declaration.clone()).analyse(&contrast_request(false), &ctx()).unwrap();
            let b = session(reversed).analyse(&contrast_request(false), &ctx()).unwrap();
            assert_ne!(a.contract_identity, b.contract_identity);
            close(a.result.values[0], b.result.values[0], 1e-12, "order of measurements");
        }
        let outcome = session(declaration).analyse(&contrast_request(false), &ctx()).unwrap();
        estimates.push(outcome.result.values[0]);
        variances.push(outcome.result.values[1]);
    }
    let n = replicates as f64;
    let mean = estimates.iter().sum::<f64>() / n;
    let empirical = estimates.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let reported = variances.iter().sum::<f64>() / n;
    // Subject-level sd of a mean is about sqrt(0.09 + 0.25 / 3) = 0.415; the
    // contrast over ~75 subjects per arm has sd near 0.07, so the Monte Carlo
    // standard error of the mean is about 0.007.
    close(mean, 0.4, 0.04, "mean of the weighted subject-level contrast");
    let ratio = reported / empirical;
    assert!((0.7..1.4).contains(&ratio), "reported over empirical variance {ratio}");
}

#[test]
fn the_hierarchical_cluster_cell_is_carried_forward_not_analysed_as_repeated_measures() {
    let mut declaration = repeated_measures_trial([0.0, 0.4], 40, 12);
    declaration.cell.outcome = OutcomeStructure::HierarchicalClustered;
    let session = session(declaration);
    let refusal = session.analyse(&contrast_request(false), &ctx()).unwrap_err();
    assert_eq!(refusal.code, "cell_not_licensed");
    assert_eq!(refusal.detail, "adaptive_inference.cell_carried_forward");
}

// -- anytime-valid confidence sequences -------------------------------------------------------------

#[test]
fn confidence_sequences_of_a_known_truth_trial_contain_the_arm_means() {
    let truth = [0.35, 0.65];
    let session = session(epsilon_greedy_trial(truth, 0.4, 600, &[300, 600], 77));
    let request = AdaptiveRequest::ConfidenceSequence {
        look: 1,
        alpha: 0.05,
        tuning_samples: 100.0,
        contrast: Some((1, 0)),
    };
    let outcome = session.analyse(&request, &ctx()).unwrap();
    let v = &outcome.result.values;
    // Per arm: count, mean, lower, upper; then the contrast bounds.
    assert_eq!(v.len(), 10);
    for arm in 0..2 {
        assert!(v[arm * 4 + 2] <= truth[arm] && truth[arm] <= v[arm * 4 + 3], "arm {arm}");
    }
    assert!(v[8] <= 0.3 && 0.3 <= v[9], "contrast bounds {} {}", v[8], v[9]);
    assert_eq!(outcome.result.closed_guarantee, "adaptive_inference.interval_closed");
}

// -- group-sequential monitoring --------------------------------------------------------------------

/// Welch statistic of the units enrolled by `enrolled`, written out here.
fn welch(declaration: &AdaptiveDesignDeclaration, enrolled: usize) -> f64 {
    let moments = |arm: u32| {
        let values: Vec<f64> = (0..enrolled)
            .filter(|&u| declaration.assigned_arm[u] == arm)
            .map(|u| declaration.outcomes[u][0])
            .collect();
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|y| (y - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (n, mean, variance)
    };
    let (n0, m0, v0) = moments(0);
    let (n1, m1, v1) = moments(1);
    (m1 - m0) / (v0 / n0 + v1 / n1).sqrt()
}

#[test]
fn group_sequential_decisions_use_the_independent_boundaries_and_statistics() {
    // Two-sided Lan-DeMets O'Brien-Fleming-type boundaries at information
    // 1/3, 2/3 and 1, alpha 0.05 (independent grid recursion).
    let published = [3.394_757_202_228_558, 2.406_732_888_064_830_4, 2.015_246_938_851_660_7];
    for (seed, effect) in [(31_u64, 0.0), (32, 0.0), (33, 1.5)] {
        let declaration = group_sequential_trial(effect, 240, &[80, 160, 240], seed);
        let statistics: Vec<f64> = [80, 160, 240].map(|n| welch(&declaration, n)).to_vec();
        let expected = statistics
            .iter()
            .zip(published)
            .position(|(z, c)| z.abs() >= c)
            .map_or(-1.0, |k| k as f64);
        let session = session(declaration);
        let request =
            AdaptiveRequest::GroupSequential { look: 2, control_arm: 0, treatment_arm: 1 };
        let outcome = session.analyse(&request, &ctx()).unwrap();
        let v = &outcome.result.values;
        close(v[0], expected, 0.0, "stopped_at");
        for k in 0..3 {
            close(v[1 + 3 * k], published[k], 5e-5, "critical value");
            close(v[2 + 3 * k], statistics[k], 1e-9, "statistic");
        }
        close(v.iter().skip(3).step_by(3).sum::<f64>(), 0.05, 1e-9, "alpha spent over the looks");
        if effect > 1.0 {
            assert!(v[0] >= 0.0, "a large effect must cross a boundary");
            assert!(v[0] < 1.5, "a large effect crosses at the first or second look");
        }
    }
}

#[test]
fn a_group_sequential_rule_is_not_licensed_for_a_thompson_assignment() {
    let mut declaration = thompson_trial([0.3, 0.6], 60, &[30, 60], 8);
    declaration.cell.monitoring = MonitoringStrategy::GroupSequential;
    declaration.stopping_rule = Some(StoppingRuleDeclaration {
        kind: StoppingRuleKind::Efficacy,
        spec: StoppingSpec::AlphaSpending {
            spending: SpendingKind::PocockType,
            total_alpha: 0.05,
            sidedness: BoundarySidedness::OneSidedUpper,
        },
        ended_by_rule: false,
    });
    let session = session(declaration);
    let request = AdaptiveRequest::BayesianPosterior { look: 1, prior_alpha: 1.0, prior_beta: 1.0 };
    let refusal = session.analyse(&request, &ctx()).unwrap_err();
    assert_eq!(refusal.code, "cell_not_licensed");
    assert_eq!(refusal.detail, "adaptive_inference.pairing_not_licensed");
}

// -- artifacts -------------------------------------------------------------------------------

#[test]
fn exported_contract_and_result_are_consumed_by_a_fresh_reader() {
    let session = session(group_sequential_trial(1.5, 240, &[80, 160, 240], 33));
    let request = AdaptiveRequest::GroupSequential { look: 2, control_arm: 0, treatment_arm: 1 };
    let contract_bytes = session
        .export_contract(InferenceParadigm::FrequentistGroupSequential, "gs-contract")
        .unwrap();
    let consumed = consume_adaptive_monitoring_contract_artifact(&contract_bytes, None).unwrap();
    assert_eq!(consumed.paradigm(), InferenceParadigm::FrequentistGroupSequential);
    assert_eq!(
        consumed.procedure().identifier(),
        "fixed_randomization/single_outcome/group_sequential/efficacy"
    );
    let result_bytes = session.export_result(&request, "gs-result", &ctx()).unwrap();
    let result = consume_adaptive_inference_result_artifact(&result_bytes, None, &ctx()).unwrap();
    assert_eq!(result.paradigm(), InferenceParadigm::FrequentistGroupSequential);
    let produced = session.analyse(&request, &ctx()).unwrap();
    assert_eq!(result.result().values.len(), produced.result.values.len());
    for (a, b) in result.result().values.iter().zip(&produced.result.values) {
        close(*a, *b, 1e-9, "consumer value");
    }
    // A consumer that kept the identity refuses another experiment's bytes.
    let other =
        AdaptiveInferenceSession::prepare(group_sequential_trial(1.5, 240, &[80, 160, 240], 34))
            .unwrap();
    let other_bytes = other.export_result(&request, "gs-result-other", &ctx()).unwrap();
    let refusal =
        consume_adaptive_inference_result_artifact(&other_bytes, Some(result.identity()), &ctx())
            .unwrap_err();
    assert_eq!(refusal.code, "route_not_supported");
    assert_eq!(refusal.detail, "adaptive_inference.wrong_contract");
}
