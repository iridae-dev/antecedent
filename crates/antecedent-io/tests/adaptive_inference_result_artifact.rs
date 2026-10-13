//! `adaptive_inference_result_v1`: results recomputed from the retained
//! history by a fresh consumer, bound to the monitoring contract, with every
//! semantic edit refused.
//!
//! Oracles: the Beta-Bernoulli posterior of counts, whose parameters, mean and
//! variance are closed forms and whose probability-best and expected loss were
//! integrated with `SciPy`; and the hand calculation of augmented
//! inverse-probability scores on a four-unit history.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::adaptive_design::{
    AdaptiveDesignContract, AdaptiveDesignDeclaration, AssignmentFamily, DelayMechanism,
    InferenceParadigm, InterimAnalysis, MonitoringStrategy, OutcomeAvailability, OutcomeStructure,
    PolicyEpoch, ProcedureCell, StoppingRuleDeclaration, StoppingRuleKind, StoppingSpec,
};
use antecedent_core::{ExecutionContext, SemanticDigest};
use antecedent_io::adaptive_inference_result_artifact::{
    AdaptiveInferenceResultArtifact, AdaptiveInferenceResultArtifactError, AdaptiveResultSpec,
    compute_adaptive_result,
};

type Artifact = AdaptiveInferenceResultArtifact;
type Error = AdaptiveInferenceResultArtifactError;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(11)
}

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{what}: expected {expected}, got {actual} (tolerance {tolerance})"
    );
}

/// Twelve units alternating arms; arm 0 always fails, arm 1 always succeeds.
fn thompson() -> AdaptiveDesignDeclaration {
    let assigned_arm: Vec<u32> = (0..12).map(|unit| unit % 2).collect();
    AdaptiveDesignDeclaration {
        label: "thompson_trial".to_owned(),
        cell: ProcedureCell {
            assignment: AssignmentFamily::ThompsonBetaBernoulli,
            outcome: OutcomeStructure::SingleOutcome,
            monitoring: MonitoringStrategy::FixedInterimLooks,
            stopping: StoppingRuleKind::Efficacy,
        },
        arm_labels: vec!["a".to_owned(), "b".to_owned()],
        assigned_arm: assigned_arm.clone(),
        policy_history: Some(vec![PolicyEpoch {
            first_unit: 0,
            policy_identity: SemanticDigest::from_bytes([3; 32]),
        }]),
        assignment_probabilities: Some(vec![vec![0.5, 0.5]; 12]),
        exposure: Some(assigned_arm),
        availability: Some(OutcomeAvailability {
            available_at_look: (0..12).map(|unit| Some(u32::from(unit >= 6))).collect(),
            delay: DelayMechanism::Immediate,
        }),
        outcomes: (0..12).map(|unit| vec![f64::from(unit % 2)]).collect(),
        interim_analyses: Some(vec![
            InterimAnalysis { units_enrolled: 6, information_fraction: 0.5 },
            InterimAnalysis { units_enrolled: 12, information_fraction: 1.0 },
        ]),
        stopping_rule: Some(StoppingRuleDeclaration {
            kind: StoppingRuleKind::Efficacy,
            spec: StoppingSpec::PosteriorProbabilityBest { threshold: 0.95 },
            ended_by_rule: false,
        }),
        probability_floor: 0.05,
    }
}

/// The four-unit history of the hand calculation, as an epsilon-greedy trial.
fn weighted() -> AdaptiveDesignDeclaration {
    let assigned_arm = vec![0, 1, 1, 0];
    AdaptiveDesignDeclaration {
        label: "epsilon_greedy_trial".to_owned(),
        cell: ProcedureCell {
            assignment: AssignmentFamily::EpsilonGreedy,
            outcome: OutcomeStructure::SingleOutcome,
            monitoring: MonitoringStrategy::FixedInterimLooks,
            stopping: StoppingRuleKind::Efficacy,
        },
        arm_labels: vec!["a".to_owned(), "b".to_owned()],
        assigned_arm: assigned_arm.clone(),
        policy_history: Some(vec![PolicyEpoch {
            first_unit: 0,
            policy_identity: SemanticDigest::from_bytes([4; 32]),
        }]),
        assignment_probabilities: Some(vec![
            vec![0.5, 0.5],
            vec![0.5, 0.5],
            vec![0.3, 0.7],
            vec![0.2, 0.8],
        ]),
        exposure: Some(assigned_arm),
        availability: Some(OutcomeAvailability {
            available_at_look: vec![Some(0), Some(0), Some(1), Some(1)],
            delay: DelayMechanism::Immediate,
        }),
        outcomes: vec![vec![1.0], vec![0.0], vec![1.0], vec![0.0]],
        interim_analyses: Some(vec![
            InterimAnalysis { units_enrolled: 2, information_fraction: 0.5 },
            InterimAnalysis { units_enrolled: 4, information_fraction: 1.0 },
        ]),
        stopping_rule: Some(StoppingRuleDeclaration {
            kind: StoppingRuleKind::Efficacy,
            spec: StoppingSpec::PosteriorProbabilityBest { threshold: 0.95 },
            ended_by_rule: false,
        }),
        probability_floor: 0.05,
    }
}

fn contract(declaration: AdaptiveDesignDeclaration) -> AdaptiveDesignContract {
    AdaptiveDesignContract::new(declaration).expect("complete contract")
}

fn posterior(look: u32) -> AdaptiveResultSpec {
    AdaptiveResultSpec::BayesianPosterior { look, prior_alpha: 1.0, prior_beta: 1.0 }
}

#[test]
fn the_posterior_is_the_conjugate_closed_form() {
    let result = compute_adaptive_result(&contract(thompson()), &posterior(1), &ctx()).unwrap();
    assert_eq!(result.paradigm, InferenceParadigm::BayesianPosterior);
    // Six failures of arm a and six successes of arm b under uniform priors:
    // Beta(1, 7) and Beta(7, 1).
    let v = &result.values;
    assert_eq!(v.len(), 12);
    close(v[0], 1.0, 0.0, "alpha of a");
    close(v[1], 7.0, 0.0, "beta of a");
    close(v[2], 1.0 / 8.0, 1e-15, "mean of a");
    close(v[3], 7.0 / (64.0 * 9.0), 1e-15, "variance of a");
    close(v[6], 7.0, 0.0, "alpha of b");
    close(v[7], 1.0, 0.0, "beta of b");
    close(v[8], 7.0 / 8.0, 1e-15, "mean of b");
    // scipy quad: P(Beta(7,1) > Beta(1,7)) and the expected losses.
    close(v[10], 0.999_708_624_708_624_8, 1e-9, "probability b is best");
    close(v[4] + v[10], 1.0, 1e-9, "probabilities of best sum to one");
    close(v[11], 1.942_501_942_501_718_5e-5, 1e-9, "loss of choosing b");
    close(v[5], 0.750_019_425_019_425, 1e-9, "loss of choosing a");
    // The first look saw only half the data.
    let early = compute_adaptive_result(&contract(thompson()), &posterior(0), &ctx()).unwrap();
    close(early.values[1], 4.0, 0.0, "beta of a at the first look");
    close(early.values[10], 0.985_714_285_714_285_4, 1e-9, "probability b is best, look 0");
}

#[test]
fn the_posterior_rule_stops_when_its_threshold_is_met() {
    let spec = AdaptiveResultSpec::PosteriorStopping { look: 1, prior_alpha: 1.0, prior_beta: 1.0 };
    // Decision-theoretic stopping is licensed only for the Thompson cell with
    // a decision-theoretic rule; the efficacy cell refuses it.
    let refused = compute_adaptive_result(&contract(thompson()), &spec, &ctx()).unwrap_err();
    assert_eq!(refused.detail, "adaptive_inference.paradigm_not_licensed_for_cell");
    let mut declaration = thompson();
    declaration.cell.stopping = StoppingRuleKind::DecisionTheoretic;
    declaration.stopping_rule = Some(StoppingRuleDeclaration {
        kind: StoppingRuleKind::DecisionTheoretic,
        spec: StoppingSpec::PosteriorExpectedLoss { loss_threshold: 0.05 },
        ended_by_rule: false,
    });
    let decision = compute_adaptive_result(&contract(declaration), &spec, &ctx()).unwrap();
    assert_eq!(decision.paradigm, InferenceParadigm::DecisionTheoreticStopping);
    // Arm b is best and its expected loss (1.9e-5) is below the 0.05 threshold.
    close(decision.values[0], 1.0, 0.0, "best arm");
    close(decision.values[3], 1.0, 0.0, "stop");
}

#[test]
fn weighted_scores_match_the_hand_calculation_through_the_contract() {
    let spec = AdaptiveResultSpec::AdaptiveWeighting {
        look: 1,
        plus_arm: 1,
        minus_arm: Some(0),
        scheme: "uniform".to_owned(),
        prior_mean: 0.5,
        prior_strength: 1.0,
        size_weighted: false,
    };
    let result = compute_adaptive_result(&contract(weighted()), &spec, &ctx()).unwrap();
    assert_eq!(result.paradigm, InferenceParadigm::FrequentistAdaptiveWeighting);
    close(result.values[0], 0.455_357_142_857_142_85, 1e-14, "contrast");
    close(result.values[1], 0.894_351_881_377_551, 1e-14, "variance");
    close(result.values[2], 4.0, 0.0, "units");
    // The first look had two units observed, one per arm: the same route runs
    // on exactly those.
    let first = AdaptiveResultSpec::AdaptiveWeighting {
        look: 0,
        plus_arm: 1,
        minus_arm: Some(0),
        scheme: "uniform".to_owned(),
        prior_mean: 0.5,
        prior_strength: 1.0,
        size_weighted: false,
    };
    let early = compute_adaptive_result(&contract(weighted()), &first, &ctx()).unwrap();
    close(early.values[2], 2.0, 0.0, "units at the first look");
}

#[test]
fn a_probability_floor_failure_is_a_typed_refusal_not_a_clip() {
    let mut declaration = weighted();
    declaration.probability_floor = 0.25;
    let spec = AdaptiveResultSpec::AdaptiveWeighting {
        look: 1,
        plus_arm: 1,
        minus_arm: Some(0),
        scheme: "uniform".to_owned(),
        prior_mean: 0.5,
        prior_strength: 1.0,
        size_weighted: false,
    };
    let error = compute_adaptive_result(&contract(declaration), &spec, &ctx()).unwrap_err();
    assert_eq!(error.code, "adaptive_assumption_violated");
    assert_eq!(error.detail, "adaptive_inference.positivity_floor");
}

#[test]
fn a_paradigm_of_another_cell_is_refused_before_any_number() {
    let weighting = AdaptiveResultSpec::AdaptiveWeighting {
        look: 1,
        plus_arm: 1,
        minus_arm: Some(0),
        scheme: "uniform".to_owned(),
        prior_mean: 0.5,
        prior_strength: 1.0,
        size_weighted: false,
    };
    let error = compute_adaptive_result(&contract(thompson()), &weighting, &ctx()).unwrap_err();
    assert_eq!(error.code, "cell_not_licensed");
    assert_eq!(error.detail, "adaptive_inference.paradigm_not_licensed_for_cell");
    let error = compute_adaptive_result(&contract(weighted()), &posterior(1), &ctx()).unwrap_err();
    assert_eq!(error.detail, "adaptive_inference.paradigm_not_licensed_for_cell");
    // A look the contract does not have.
    let error = compute_adaptive_result(&contract(thompson()), &posterior(5), &ctx()).unwrap_err();
    assert_eq!(error.detail, "adaptive_inference.input_invalid");
}

#[test]
fn cancellation_is_a_refusal_not_a_verdict() {
    let ctx = ctx();
    ctx.cancellation.cancel();
    let error = compute_adaptive_result(&contract(thompson()), &posterior(1), &ctx).unwrap_err();
    assert_eq!(error.code, "cancelled_no_claim");
    assert_eq!(error.detail, "adaptive_inference.cancelled");
}

// -- the artifact ----------------------------------------------------------------------------

#[test]
fn a_fresh_consumer_sees_the_procedure_rule_and_paradigm_and_recomputes_the_values() {
    let source = contract(thompson());
    let encoded = Artifact::to_bytes(&source, &posterior(1), "adaptive-result", &ctx()).unwrap();
    let consumed = Artifact::from_bytes(&encoded, None, &ctx()).unwrap();
    assert_eq!(consumed.contract().paradigm(), InferenceParadigm::BayesianPosterior);
    assert_eq!(
        consumed.contract().procedure().identifier(),
        "thompson_beta_bernoulli/single_outcome/fixed_interim_looks/efficacy"
    );
    assert_eq!(
        consumed.contract().stopping_rule().unwrap().spec,
        StoppingSpec::PosteriorProbabilityBest { threshold: 0.95 }
    );
    close(consumed.result().values[10], 0.999_708_624_708_624_8, 1e-9, "recomputed");
    assert_eq!(consumed.result().closed_guarantee, "adaptive_inference.posterior_interval_closed");
    // Deterministic, and accepted against its own retained identity.
    assert_eq!(
        Artifact::to_bytes(&source, &posterior(1), "adaptive-result", &ctx()).unwrap(),
        encoded
    );
    assert!(Artifact::from_bytes(&encoded, Some(consumed.identity()), &ctx()).is_ok());
}

fn resealed(
    edit: impl FnOnce(&mut antecedent_io::adaptive_inference_result_artifact::AdaptiveResultMeta),
) -> Result<Artifact, Error> {
    let source = contract(thompson());
    let encoded = Artifact::to_bytes(&source, &posterior(1), "adaptive-result", &ctx()).unwrap();
    let mut meta = Artifact::decode_unchecked(&encoded).unwrap();
    edit(&mut meta);
    Artifact::reseal_identity(&mut meta);
    Artifact::from_meta(&meta, None, &ctx())
}

#[test]
fn every_resealed_semantic_mutation_is_refused() {
    // A stored value that does not recompute.
    let outcome = resealed(|meta| meta.value_bits[10] = 0.5_f64.to_bits());
    assert_eq!(outcome.unwrap_err(), Error::ResultMismatch("values"));
    // A stored value count that does not match the route.
    let outcome = resealed(|meta| {
        meta.value_bits.pop();
    });
    assert_eq!(outcome.unwrap_err(), Error::ResultMismatch("values"));
    // A different look than the one the values describe.
    let outcome = resealed(|meta| meta.spec.look = 0);
    assert_eq!(outcome.unwrap_err(), Error::ResultMismatch("values"));
    // A different prior.
    let outcome = resealed(|meta| meta.spec.parameter_bits[0] = 3.0_f64.to_bits());
    assert_eq!(outcome.unwrap_err(), Error::ResultMismatch("values"));
    // A different stopping rule inside the embedded contract.
    let outcome = resealed(|meta| {
        let rule = meta.contract.declaration.stopping_rule.as_mut().unwrap();
        rule.parameter_bits = Some(0.99_f64.to_bits());
    });
    assert!(matches!(outcome, Err(Error::IdentityMismatch { field: "contract_identity" })));
    // A different outcome inside the embedded contract.
    let outcome =
        resealed(|meta| meta.contract.declaration.outcome_bits[0] = vec![1.0_f64.to_bits()]);
    assert!(matches!(outcome, Err(Error::IdentityMismatch { field: "contract_identity" })));
    // The request's paradigm and the contract's paradigm must be the same.
    let outcome = resealed(|meta| {
        meta.contract.paradigm = InferenceParadigm::AnytimeValidFrequentist.as_str().to_owned();
        antecedent_io::adaptive_monitoring_contract_artifact::AdaptiveMonitoringContractArtifact::reseal_identity(
            &mut meta.contract,
        );
    });
    assert_eq!(outcome.unwrap_err(), Error::ResultMismatch("paradigm"));
    // A claim stronger than point-only, or a calibration claim.
    let outcome = resealed(|meta| meta.claim = "calibrated".to_owned());
    assert!(matches!(outcome, Err(Error::UnsupportedSemantics(_))));
    let outcome = resealed(|meta| meta.calibration = "measured".to_owned());
    assert!(matches!(outcome, Err(Error::UnsupportedSemantics(_))));
    // A removed contract element.
    let outcome = resealed(|meta| meta.contract.declaration.exposure = None);
    match outcome.unwrap_err() {
        Error::Refused { code, message } => {
            assert_eq!(code, "adaptive_contract_incomplete");
            assert!(message.starts_with("adaptive_inference.exposure_history_missing"));
        }
        other => panic!("{other}"),
    }
}

#[test]
fn a_consumer_that_retained_the_identity_refuses_another_experiment() {
    let a = Artifact::to_bytes(&contract(thompson()), &posterior(1), "a", &ctx()).unwrap();
    let identity = Artifact::from_bytes(&a, None, &ctx()).unwrap().identity().to_owned();
    let mut other = thompson();
    other.label = "another_trial".to_owned();
    let b = Artifact::to_bytes(&contract(other), &posterior(1), "b", &ctx()).unwrap();
    let error = Artifact::from_bytes(&b, Some(&identity), &ctx()).unwrap_err();
    assert!(matches!(error, Error::IdentityMismatch { field: "retained_identity" }));
    let (code, detail, _) = error.refusal().unwrap();
    assert_eq!(code, "route_not_supported");
    assert_eq!(detail, "adaptive_inference.wrong_contract");
}
