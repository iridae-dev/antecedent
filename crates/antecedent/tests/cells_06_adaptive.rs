// SPDX-License-Identifier: MIT OR Apache-2.0
//! Adaptive-experiment inference under missing outcomes, declared weights, degenerate outcomes
//! and arms the design never exercised. Oracles are written by hand from the Beta-Bernoulli
//! conjugacy (`Beta(a + successes, b + failures)`, variance `ab / ((a + b)^2 (a + b + 1))`) and
//! from the retained counts of each look, never from the code under test.
use antecedent::analysis::adaptive_inference::{
    AdaptiveDesignDeclaration, AdaptiveInferenceSession, AdaptiveRequest, AssignmentFamily,
    BoundarySidedness, DelayMechanism, InterimAnalysis, MonitoringStrategy, OutcomeAvailability,
    OutcomeStructure, PolicyEpoch, ProcedureCell, SpendingKind, StoppingRuleDeclaration,
    StoppingRuleKind, StoppingSpec,
};
use antecedent_core::{ExecutionContext, SemanticDigest};
use antecedent_io::adaptive_monitoring_contract_artifact::{
    DeclarationWire, declaration_from_wire, declaration_to_wire,
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(3)
}

fn digest() -> Vec<PolicyEpoch> {
    vec![PolicyEpoch { first_unit: 0, policy_identity: SemanticDigest::from_bytes([1; 32]) }]
}

fn looks(sizes: &[usize], total: usize) -> Vec<InterimAnalysis> {
    sizes
        .iter()
        .map(|&n| InterimAnalysis {
            units_enrolled: u32::try_from(n).unwrap(),
            information_fraction: n as f64 / total as f64,
        })
        .collect()
}

/// Alternating arms over eight units, looks after four and eight units, every outcome observed
/// when its unit is enrolled.
fn thompson(outcomes: [f64; 8]) -> AdaptiveDesignDeclaration {
    let arms: Vec<u32> = (0..8).map(|u| u32::from(u % 2 == 1)).collect();
    AdaptiveDesignDeclaration {
        label: "cells_06_thompson".into(),
        cell: ProcedureCell {
            assignment: AssignmentFamily::ThompsonBetaBernoulli,
            outcome: OutcomeStructure::SingleOutcome,
            monitoring: MonitoringStrategy::FixedInterimLooks,
            stopping: StoppingRuleKind::Efficacy,
        },
        arm_labels: vec!["control".into(), "treatment".into()],
        assigned_arm: arms.clone(),
        policy_history: Some(digest()),
        assignment_probabilities: Some(vec![vec![0.5, 0.5]; 8]),
        exposure: Some(arms),
        availability: Some(OutcomeAvailability {
            available_at_look: (0..8).map(|u| Some(u32::from(u >= 4))).collect(),
            delay: DelayMechanism::Immediate,
        }),
        outcomes: outcomes.iter().map(|&y| vec![y]).collect(),
        interim_analyses: Some(looks(&[4, 8], 8)),
        stopping_rule: Some(StoppingRuleDeclaration {
            kind: StoppingRuleKind::Efficacy,
            spec: StoppingSpec::PosteriorProbabilityBest { threshold: 0.95 },
            ended_by_rule: false,
        }),
        probability_floor: 1e-6,
    }
}

/// Fixed one-to-one randomization with a group-sequential rule over twenty units.
fn group_sequential(outcomes: Vec<f64>) -> AdaptiveDesignDeclaration {
    let total = outcomes.len();
    let arms: Vec<u32> = (0..total).map(|u| u32::from(u % 2 == 1)).collect();
    AdaptiveDesignDeclaration {
        label: "cells_06_group_sequential".into(),
        cell: ProcedureCell {
            assignment: AssignmentFamily::FixedRandomization,
            outcome: OutcomeStructure::SingleOutcome,
            monitoring: MonitoringStrategy::GroupSequential,
            stopping: StoppingRuleKind::Efficacy,
        },
        arm_labels: vec!["control".into(), "treatment".into()],
        assigned_arm: arms.clone(),
        policy_history: Some(digest()),
        assignment_probabilities: Some(vec![vec![0.5, 0.5]; total]),
        exposure: Some(arms),
        availability: Some(OutcomeAvailability {
            available_at_look: (0..total).map(|u| Some(u32::from(u >= total / 2))).collect(),
            delay: DelayMechanism::Immediate,
        }),
        outcomes: outcomes.iter().map(|&y| vec![y]).collect(),
        interim_analyses: Some(looks(&[total / 2, total], total)),
        stopping_rule: Some(StoppingRuleDeclaration {
            kind: StoppingRuleKind::Efficacy,
            spec: StoppingSpec::AlphaSpending {
                spending: SpendingKind::OBrienFlemingType,
                total_alpha: 0.05,
                sidedness: BoundarySidedness::TwoSidedSymmetric,
            },
            ended_by_rule: false,
        }),
        probability_floor: 0.5,
    }
}

fn posterior(look: u32) -> AdaptiveRequest {
    AdaptiveRequest::BayesianPosterior { look, prior_alpha: 1.0, prior_beta: 1.0 }
}

fn beta_variance(a: f64, b: f64) -> f64 {
    a * b / ((a + b).powi(2) * (a + b + 1.0))
}

/// Alpha, beta, mean and variance of one arm of a posterior result.
fn arm_beta(values: &[f64], arm: usize) -> [f64; 4] {
    [values[arm * 6], values[arm * 6 + 1], values[arm * 6 + 2], values[arm * 6 + 3]]
}

fn assert_beta(values: &[f64], arm: usize, a: f64, b: f64) {
    let [alpha, beta, mean, variance] = arm_beta(values, arm);
    assert_eq!((alpha, beta), (a, b), "arm {arm} shape");
    assert!((mean - a / (a + b)).abs() < 1e-15, "arm {arm} mean {mean}");
    assert!((variance - beta_variance(a, b)).abs() < 1e-15, "arm {arm} variance {variance}");
}

#[test]
fn cells_06_delayed_outcomes_enter_only_when_observed_and_an_outcome_dependent_delay_is_refused() {
    // Arms alternate 0,1,0,1,...; outcomes 1,1,0,0,1,1,0,(never observed).
    let mut declaration = thompson([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
    let availability = declaration.availability.as_mut().unwrap();
    // Units 1 and 3 report one look late; unit 7 never reports and carries no outcome.
    availability.available_at_look =
        vec![Some(0), Some(1), Some(0), Some(1), Some(1), Some(1), Some(1), None];
    availability.delay = DelayMechanism::IgnorableAdministrative;
    declaration.outcomes[7] = Vec::new();
    let session = AdaptiveInferenceSession::prepare(declaration.clone()).unwrap();

    // Look 0 analyses units {0, 2}: control saw one success and one failure; treatment, no unit.
    let first = session.analyse(&posterior(0), &ctx()).unwrap();
    assert_beta(&first.result.values, 0, 2.0, 2.0);
    assert_beta(&first.result.values, 1, 1.0, 1.0);
    // Look 1 analyses units 0..=6: control {1, 0, 1, 0} gives Beta(3, 3); treatment {1, 0, 1}
    // gives Beta(3, 2). The unit that never reported does not enter, as a success or a failure.
    let second = session.analyse(&posterior(1), &ctx()).unwrap();
    assert_beta(&second.result.values, 0, 3.0, 3.0);
    assert_beta(&second.result.values, 1, 3.0, 2.0);

    // A delay that depends on the outcome cannot be ignored, so the units are never dropped.
    let mut dependent = declaration.clone();
    dependent.availability.as_mut().unwrap().delay = DelayMechanism::OutcomeDependent;
    let session = AdaptiveInferenceSession::prepare(dependent).unwrap();
    let refusal = session.analyse(&posterior(1), &ctx()).unwrap_err();
    assert_eq!(refusal.code, "adaptive_assumption_violated");
    assert_eq!(refusal.detail, "adaptive_inference.delay_not_ignorable");

    // A non-finite outcome is not a missing one: it is refused at the contract.
    let mut not_a_number = declaration.clone();
    not_a_number.outcomes[2] = vec![f64::NAN];
    let refusals = AdaptiveInferenceSession::prepare(not_a_number).unwrap_err();
    assert!(refusals.iter().any(|r| r.detail == "adaptive_inference.outcome_invalid"));

    // The availability must agree with the outcomes: an outcome declared never available but
    // present, or available but empty, is a shape refusal.
    let mut present = declaration.clone();
    present.outcomes[7] = vec![1.0];
    let refusals = AdaptiveInferenceSession::prepare(present).unwrap_err();
    assert!(refusals.iter().any(|r| r.detail == "adaptive_inference.shape_mismatch"));
    let mut empty = declaration;
    empty.outcomes[0] = Vec::new();
    let refusals = AdaptiveInferenceSession::prepare(empty).unwrap_err();
    assert!(refusals.iter().any(|r| r.detail == "adaptive_inference.shape_mismatch"));
}

#[test]
fn cells_06_a_constant_outcome_has_its_closed_form_and_a_zero_variance_statistic_is_refused() {
    // Every unit fails: the conjugate posterior of each arm after four units is Beta(1, 5) (look
    // 1 has all eight units, four per arm).
    let failures = AdaptiveInferenceSession::prepare(thompson([0.0; 8])).unwrap();
    let result = failures.analyse(&posterior(1), &ctx()).unwrap();
    assert_beta(&result.result.values, 0, 1.0, 5.0);
    assert_beta(&result.result.values, 1, 1.0, 5.0);
    // Every unit succeeds: Beta(5, 1) for both arms.
    let successes = AdaptiveInferenceSession::prepare(thompson([1.0; 8])).unwrap();
    let result = successes.analyse(&posterior(1), &ctx()).unwrap();
    assert_beta(&result.result.values, 0, 5.0, 1.0);
    assert_beta(&result.result.values, 1, 5.0, 1.0);

    // A normal-theory group-sequential statistic divides by the sample variance: with no
    // variation in either arm it has no value, and the route says so instead of inventing one.
    let request = AdaptiveRequest::GroupSequential { look: 1, control_arm: 0, treatment_arm: 1 };
    for outcomes in [vec![3.0; 20], (0..20).map(|u| f64::from(u32::from(u % 2 == 1))).collect()] {
        let session = AdaptiveInferenceSession::prepare(group_sequential(outcomes)).unwrap();
        let refusal = session.analyse(&request, &ctx()).unwrap_err();
        assert_eq!(refusal.code, "invalid_argument");
        assert_eq!(refusal.detail, "adaptive_inference.input_invalid");
        assert!(refusal.message.contains("zero sample variance"), "{}", refusal.message);
    }
    // With variation the same route answers: control {0, 2, ...} and treatment {1, 3, ...} are
    // separated, so the statistic exists.
    let varied: Vec<f64> =
        (0..20).map(|u| f64::from(u32::from(u % 2 == 1)) + f64::from(u % 5) * 0.1).collect();
    let session = AdaptiveInferenceSession::prepare(group_sequential(varied)).unwrap();
    assert!(session.analyse(&request, &ctx()).is_ok());
}

#[test]
fn cells_06_a_declared_weight_has_no_field_and_the_wire_declaration_refuses_it() {
    let declaration = thompson([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
    let wire = declaration_to_wire(&declaration);
    let json = serde_json::to_value(&wire).unwrap();
    // The untouched declaration reads back to the same contract.
    let reread: DeclarationWire = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(declaration_from_wire(&reread).unwrap(), declaration);
    // The only weights of the design are the realized assignment probabilities it retains. A
    // survey or analysis weight has no field, so it cannot be declared and then ignored.
    for field in ["weights", "sampling_weights", "unit_weights"] {
        let mut weighted = json.clone();
        weighted[field] = serde_json::json!([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        assert!(
            serde_json::from_value::<DeclarationWire>(weighted).is_err(),
            "`{field}` must be refused by the declaration"
        );
    }
}

#[test]
fn cells_06_an_arm_or_look_the_design_never_exercised_is_refused_not_extrapolated() {
    let session = AdaptiveInferenceSession::prepare(group_sequential(
        (0..20).map(|u| f64::from(u32::from(u % 2 == 1)) + f64::from(u % 5) * 0.1).collect(),
    ))
    .unwrap();
    // An arm index the design never declared.
    let unknown_arm =
        AdaptiveRequest::GroupSequential { look: 1, control_arm: 0, treatment_arm: 5 };
    let refusal = session.analyse(&unknown_arm, &ctx()).unwrap_err();
    assert_eq!(refusal.code, "invalid_argument");
    assert_eq!(refusal.detail, "adaptive_inference.input_invalid");
    assert!(refusal.message.contains("two declared arms"));
    // A look beyond the declared interim analyses.
    let late = AdaptiveRequest::GroupSequential { look: 7, control_arm: 0, treatment_arm: 1 };
    let refusal = session.analyse(&late, &ctx()).unwrap_err();
    assert_eq!(refusal.detail, "adaptive_inference.input_invalid");
    assert!(refusal.message.contains("not one of the contract's interim analyses"));
    // A contrast of an arm with itself.
    let same = AdaptiveRequest::GroupSequential { look: 1, control_arm: 1, treatment_arm: 1 };
    let refusal = session.analyse(&same, &ctx()).unwrap_err();
    assert_eq!(refusal.detail, "adaptive_inference.input_invalid");

    // An arm the policy all but never assigned has inverse-probability weights beyond the
    // declared floor: the weighted route refuses it and never clips the weight.
    let mut starved = thompson([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
    starved.cell.assignment = AssignmentFamily::EpsilonGreedy;
    starved.assigned_arm = vec![0; 8];
    starved.exposure = Some(vec![0; 8]);
    starved.assignment_probabilities = Some(vec![vec![0.99, 0.01]; 8]);
    starved.probability_floor = 0.05;
    let session = AdaptiveInferenceSession::prepare(starved).unwrap();
    let weighted = AdaptiveRequest::AdaptiveWeighting {
        look: 1,
        plus_arm: 1,
        minus_arm: Some(0),
        scheme: "uniform".to_owned(),
        prior_mean: 0.5,
        prior_strength: 1.0,
        size_weighted: false,
    };
    let refusal = session.analyse(&weighted, &ctx()).unwrap_err();
    assert_eq!(refusal.code, "adaptive_assumption_violated");
    assert_eq!(refusal.detail, "adaptive_inference.positivity_floor");
}
