//! Facade integration of sequential policy evaluation: a producer evaluates a
//! supplied policy by exact enumeration, seals and exports it, and an
//! independent call consumes the bytes under the retained identity.
//!
//! Hand oracle: a unit is `sound` or `faulty` (one half each). `inspect` costs
//! 1/4 and raises an alarm with probability 3/4 if faulty and 1/4 if sound;
//! `ship` earns +10 if sound and -10 if faulty; `repair` earns +6 if sound and
//! -4 if faulty. Inspect, then repair after an alarm (P(faulty) = 3/4, value
//! -3/2) and ship after none (P(faulty) = 1/4, value 5), is worth
//! `-1/4 + 1/2 * (-3/2) + 1/2 * 5 = 3/2`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::sequential_evaluation::{
    ActionRole, ActionSpec, ActionSupport, CensoringRule, ContingentPolicy, ExpectedEvaluation,
    FiniteSequentialModel, HistoryKey, ObservationChannel, consume, evaluate, export, seal,
};
use antecedent_core::ExecutionContext;
use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, ErrorStructure, MeasurementBinding, MeasurementKind,
    MeasurementModel, MeasurementParameters, MeasurementRoute, MeasurementVariable,
    ParameterEvidence, RepresentationVersion,
};

const INSPECT: usize = 0;
const SHIP: usize = 1;
const REPAIR: usize = 2;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(3)
}

fn alarm_panel() -> MeasurementModel {
    MeasurementModel {
        model_id: "alarm".into(),
        route: MeasurementRoute::BinaryMisclassification,
        indicator: MeasurementVariable {
            id: "alarm".into(),
            kind: MeasurementKind::NoisyMisclassified,
            units: "category".into(),
        },
        construct: MeasurementVariable {
            id: "unit_condition".into(),
            kind: MeasurementKind::LatentScientific,
            units: "category".into(),
        },
        binding: MeasurementBinding {
            population_id: "plant_a".into(),
            regime_id: "operations".into(),
            arm: ArmBinding::AllArms,
        },
        structure: ErrorStructure::NonDifferential,
        parameters: MeasurementParameters::Binary {
            rates: vec![BinaryArmRates {
                arm_id: None,
                sensitivity: 0.75,
                specificity: 0.75,
                sensitivity_variance: 0.0,
                specificity_variance: 0.0,
            }],
            min_youden: 0.1,
        },
        representation: RepresentationVersion {
            coordinate_system_id: "alarm_panel".into(),
            version: 1,
        },
        evidence: ParameterEvidence::Declared { declarer: "engineering".into() },
    }
}

fn model() -> FiniteSequentialModel {
    let identity = vec![1.0, 0.0, 0.0, 1.0];
    FiniteSequentialModel {
        label: "inspection".into(),
        population: "plant_a".into(),
        regime: "operations".into(),
        period: "2026".into(),
        state_variable: "unit_condition".into(),
        states: vec!["sound".into(), "faulty".into()],
        initial: vec![0.5, 0.5],
        horizon: 2,
        actions: vec![
            ActionSpec {
                label: "inspect".into(),
                role: ActionRole::Continue,
                utility_cost: 0.25,
                resource_use: 0.0,
                transition: identity.clone(),
                channel: Some(ObservationChannel { measurement: alarm_panel() }),
                immediate_utility: vec![0.0, 0.0],
                delayed: None,
            },
            ActionSpec {
                label: "ship".into(),
                role: ActionRole::Commit,
                utility_cost: 0.0,
                resource_use: 0.0,
                transition: identity.clone(),
                channel: None,
                immediate_utility: vec![10.0, -10.0],
                delayed: None,
            },
            ActionSpec {
                label: "repair".into(),
                role: ActionRole::Commit,
                utility_cost: 0.0,
                resource_use: 0.0,
                transition: identity,
                channel: None,
                immediate_utility: vec![6.0, -4.0],
                delayed: None,
            },
        ],
        terminal_utility: vec![0.0, 0.0],
        censoring: CensoringRule::Refuse,
        resource_budget: None,
        stopping: Vec::new(),
        support: ActionSupport::everywhere(3),
    }
}

fn policy() -> ContingentPolicy {
    let alarm = HistoryKey::root().extended(INSPECT, 1);
    let quiet = HistoryKey::root().extended(INSPECT, 0);
    ContingentPolicy::new("inspect_then_act_v1")
        .decide(HistoryKey::root(), INSPECT)
        .decide(alarm, REPAIR)
        .decide(quiet, SHIP)
}

#[test]
fn the_facade_evaluates_exports_and_consumes_a_policy() {
    let model = model();
    let policy = policy();
    let evaluation = evaluate(&model, &policy, &ctx()).unwrap();
    assert!((evaluation.value - 1.5).abs() < 1e-12, "{}", evaluation.value);
    assert_eq!(evaluation.inference, "point_only");
    assert_eq!(evaluation.policy_version, "inspect_then_act_v1");

    let artifact = seal(&model, &policy, None, None, &ctx()).unwrap();
    let bytes = export(&artifact, "inspection-1").unwrap();
    let expected = ExpectedEvaluation {
        model: Some(artifact.identities().model.clone()),
        policy: Some(artifact.identities().policy.clone()),
        evaluation: Some(artifact.identities().evaluation.clone()),
    };
    let consumed = consume(&bytes, Some(&expected), &ctx()).unwrap();
    assert_eq!(consumed.value().to_bits(), evaluation.value.to_bits());

    // another policy's identity is refused under the retained expectation
    let other = seal(
        &model,
        &ContingentPolicy::new("ship_now_v1").decide(HistoryKey::root(), SHIP),
        None,
        None,
        &ctx(),
    )
    .unwrap();
    let other_bytes = export(&other, "ship-now").unwrap();
    assert!(consume(&other_bytes, Some(&expected), &ctx()).is_err());
    assert!(consume(&other_bytes, None, &ctx()).is_ok());
}

#[test]
fn a_policy_with_a_missing_history_is_refused_through_the_facade() {
    let incomplete = ContingentPolicy::new("incomplete_v1").decide(HistoryKey::root(), INSPECT);
    let err = evaluate(&model(), &incomplete, &ctx()).unwrap_err();
    assert_eq!(err.code, "invalid_argument");
    assert_eq!(err.detail, "sequential_evaluation.policy_incomplete");
    assert!(!err.remedy.is_empty());
}
