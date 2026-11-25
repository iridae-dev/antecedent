//! The three policy-transport evaluators against the kernels they call.
//!
//! Oracles, copied from `crates/antecedent-design/tests/transported_evaluation.rs`:
//! inspection value `7.5 - 10 p` (source `2.5` at `p = 1/2`); a period shift to mass `0.6`
//! worth `1.5`; an adaptive trial whose treatment and control means are `29/56` and `1/2`
//! and whose contrast is `1/56`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines, reason = "each test is one copied fixture")]
#![allow(clippy::float_cmp, reason = "the bits are compared exactly")]

use antecedent::Study;
use antecedent::planner::PlanView;
use antecedent_contract::adaptive_design::{
    AdaptiveDesignContract, AdaptiveDesignDeclaration, AssignmentFamily, DelayMechanism,
    InterimAnalysis, MonitoringStrategy, OutcomeAvailability, OutcomeStructure, PolicyEpoch,
    ProcedureCell, StoppingRuleDeclaration, StoppingRuleKind, StoppingSpec,
};
use antecedent_core::ExecutionContext;
use antecedent_core::SemanticDigest;
use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, ErrorStructure, MeasurementBinding, MeasurementKind,
    MeasurementModel, MeasurementParameters, MeasurementRoute, MeasurementVariable,
    ParameterEvidence, RepresentationVersion,
};
use antecedent_design::adaptive_trial_transport::{
    AdaptiveTransportRequest, TrialMechanism, TrialSelectionDiagram, transport_adaptive_trial,
};
use antecedent_design::sequential_evaluation::{
    ActionRole, ActionSpec, ActionSupport, CensoringRule, ContingentPolicy, EvaluationLimits,
    FiniteSequentialModel, HistoryKey, HistoryStep, ObservationChannel,
};
use antecedent_design::temporal_policy_transport::{
    HistoryAlphabet, TemporalPolicyRequest, TemporalTarget, TimeInvariance,
    evaluate_temporal_policy,
};
use antecedent_design::transported_policy::{
    MechanismStatus, ModelMechanism, SelectionDiagram, TransportedPolicyRequest,
    evaluate_transported_policy,
};

const INSPECT: usize = 0;
const SHIP: usize = 1;
const HOLD: usize = 2;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

#[track_caller]
fn bits(left: f64, right: f64) {
    assert_eq!(left.to_bits(), right.to_bits(), "{left} vs {right}");
}

fn assert_unit(view: &PlanView, unit: &str) {
    let support = view.unit_support().expect("the route is bound to its capability record");
    assert_eq!(support.unit(), unit);
    assert!(support.supported());
    assert!(support.missing_evidence().is_empty());
}

fn alarm_model() -> MeasurementModel {
    MeasurementModel {
        model_id: "inspection_alarm".into(),
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
        evidence: ParameterEvidence::Declared { declarer: "reliability_engineering".into() },
    }
}

fn commit(label: &str, immediate: [f64; 2]) -> ActionSpec {
    ActionSpec {
        label: label.into(),
        role: ActionRole::Commit,
        utility_cost: 0.0,
        resource_use: 0.0,
        transition: vec![1.0, 0.0, 0.0, 1.0],
        channel: None,
        immediate_utility: immediate.to_vec(),
        delayed: None,
    }
}

fn model(initial: [f64; 2]) -> FiniteSequentialModel {
    FiniteSequentialModel {
        label: "inspection".into(),
        population: "plant_a".into(),
        regime: "operations".into(),
        period: "2026".into(),
        state_variable: "unit_condition".into(),
        states: vec!["sound".into(), "faulty".into()],
        initial: initial.to_vec(),
        horizon: 2,
        actions: vec![
            ActionSpec {
                label: "inspect".into(),
                role: ActionRole::Continue,
                utility_cost: 0.0,
                resource_use: 0.0,
                transition: vec![1.0, 0.0, 0.0, 1.0],
                channel: Some(ObservationChannel { measurement: alarm_model() }),
                immediate_utility: vec![0.0, 0.0],
                delayed: None,
            },
            commit("ship", [10.0, -10.0]),
            commit("hold", [0.0, 0.0]),
        ],
        terminal_utility: vec![0.0, 0.0],
        censoring: CensoringRule::Refuse,
        resource_budget: None,
        stopping: Vec::new(),
        support: ActionSupport::everywhere(3),
    }
}

fn policy() -> ContingentPolicy {
    let after =
        |observation: usize| HistoryKey::from_steps([HistoryStep { action: INSPECT, observation }]);
    ContingentPolicy::new("inspect_then_ship_on_quiet")
        .decide(HistoryKey::root(), INSPECT)
        .decide(after(0), SHIP)
        .decide(after(1), HOLD)
}

fn rows(model: &FiniteSequentialModel) -> Vec<(ModelMechanism, MechanismStatus)> {
    let mut rows = vec![
        (ModelMechanism::InitialState, MechanismStatus::Shifted),
        (ModelMechanism::Utility, MechanismStatus::Invariant),
    ];
    for (action, spec) in model.actions.iter().enumerate() {
        rows.push((ModelMechanism::Transition { action }, MechanismStatus::Invariant));
        if spec.channel.is_some() {
            rows.push((ModelMechanism::Observation { action }, MechanismStatus::Invariant));
        }
    }
    rows
}

#[test]
fn the_compiled_policy_matches_evaluate_transported_policy() {
    let source = model([0.5, 0.5]);
    let policy = policy();
    let diagram = SelectionDiagram {
        source_population: "plant_a".into(),
        target_population: "plant_b".into(),
        mechanisms: rows(&source),
    };
    for faulty in [0.0, 0.5, 1.0] {
        let law = [1.0 - faulty, faulty];
        let request = TransportedPolicyRequest {
            source: &source,
            policy: &policy,
            target_population: "plant_b",
            target_initial: Some(&law),
            diagram: Some(&diagram),
            target_utility: None,
            limits: EvaluationLimits::default(),
        };
        let direct = evaluate_transported_policy(&request, &ctx()).unwrap();
        let compiled = Study::compile_transported_policy(&request, &ctx()).unwrap();
        assert_unit(&compiled.plan_view(), "transport.policy_evaluation");
        let evaluation = compiled.execute(&ctx()).unwrap();
        bits(evaluation.target_value, direct.target_value);
        bits(evaluation.source_value, direct.source_value);
        bits(evaluation.target_value, 7.5 - 10.0 * faulty);
        bits(evaluation.source_value, 2.5);
        assert_eq!(evaluation.nodes_visited, direct.nodes_visited);
    }
}

#[test]
fn the_compiled_period_shift_matches_evaluate_temporal_policy() {
    let source = model([0.5, 0.5]);
    let policy = policy();
    let invariance = TimeInvariance {
        source_period: "2026".into(),
        target_period: "2027".into(),
        horizon: source.horizon,
        mechanisms: rows(&source),
    };
    let target = TemporalTarget {
        period: "2027".into(),
        horizon: source.horizon,
        alphabet: HistoryAlphabet::of_model(&source).unwrap(),
    };
    let law = [0.4, 0.6];
    let request = TemporalPolicyRequest {
        source: &source,
        policy: &policy,
        target: &target,
        target_initial: Some(&law),
        invariance: Some(&invariance),
        limits: EvaluationLimits::default(),
    };
    let direct = evaluate_temporal_policy(&request, &ctx()).unwrap();
    let compiled = Study::compile_temporal_policy(&request, &ctx()).unwrap();
    assert_unit(&compiled.plan_view(), "transport.period_shift_policy");
    let evaluation = compiled.execute(&ctx()).unwrap();
    bits(evaluation.target_value, direct.target_value);
    bits(evaluation.source_value, direct.source_value);
    assert!((evaluation.target_value - 1.5).abs() < 1e-12);
    assert!((evaluation.source_value - 2.5).abs() < 1e-12);
}

fn digest(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

fn epsilon_greedy() -> ProcedureCell {
    ProcedureCell {
        assignment: AssignmentFamily::EpsilonGreedy,
        outcome: OutcomeStructure::SingleOutcome,
        monitoring: MonitoringStrategy::FixedInterimLooks,
        stopping: StoppingRuleKind::Efficacy,
    }
}

fn declaration() -> AdaptiveDesignDeclaration {
    let assigned_arm = vec![0, 1, 0, 1, 1, 0, 1, 0];
    AdaptiveDesignDeclaration {
        label: "trial".to_owned(),
        cell: epsilon_greedy(),
        arm_labels: vec!["control".to_owned(), "treatment".to_owned()],
        assigned_arm: assigned_arm.clone(),
        policy_history: Some(vec![
            PolicyEpoch { first_unit: 0, policy_identity: digest(1) },
            PolicyEpoch { first_unit: 4, policy_identity: digest(2) },
        ]),
        assignment_probabilities: Some(
            (0..8)
                .map(|unit| match unit {
                    4 | 5 => vec![0.2, 0.8],
                    6 | 7 => vec![0.4, 0.6],
                    _ => vec![0.5, 0.5],
                })
                .collect(),
        ),
        exposure: Some(assigned_arm),
        availability: Some(OutcomeAvailability {
            available_at_look: (0..8).map(|unit| Some(u32::from(unit >= 4))).collect(),
            delay: DelayMechanism::Immediate,
        }),
        outcomes: [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0].iter().map(|&y| vec![y]).collect(),
        interim_analyses: Some(vec![
            InterimAnalysis { units_enrolled: 4, information_fraction: 0.5 },
            InterimAnalysis { units_enrolled: 8, information_fraction: 1.0 },
        ]),
        stopping_rule: Some(StoppingRuleDeclaration {
            kind: StoppingRuleKind::Efficacy,
            spec: StoppingSpec::PosteriorProbabilityBest { threshold: 0.95 },
            ended_by_rule: false,
        }),
        probability_floor: 0.05,
    }
}

#[test]
fn the_compiled_adaptive_trial_matches_transport_adaptive_trial() {
    let contract = AdaptiveDesignContract::new(declaration()).unwrap();
    let unit_strata: Vec<usize> = (0..8).map(|unit| unit % 2).collect();
    let labels = vec!["mild".into(), "severe".into()];
    let diagram = TrialSelectionDiagram {
        source_population: "trial_sites".into(),
        target_population: "clinic_network".into(),
        mechanisms: vec![
            (TrialMechanism::StratumLaw, MechanismStatus::Shifted),
            (TrialMechanism::OutcomeGivenStratumAndArm, MechanismStatus::Invariant),
        ],
    };
    let law = [0.25, 0.75];
    let request = AdaptiveTransportRequest {
        contract: Some(&contract),
        trial_population: "trial_sites",
        unit_strata: &unit_strata,
        stratum_labels: &labels,
        target_stratum_law: Some(&law),
        diagram: Some(&diagram),
        treatment_arm: 1,
        control_arm: 0,
    };
    let direct = transport_adaptive_trial(&request).unwrap();
    let compiled = Study::compile_adaptive_trial(&request, &ctx()).unwrap();
    assert_unit(&compiled.plan_view(), "transport.adaptive_trial_source");
    let estimate = compiled.execute(&ctx()).unwrap();
    bits(estimate.treatment_mean, direct.treatment_mean);
    bits(estimate.control_mean, direct.control_mean);
    bits(estimate.contrast, direct.contrast);
    assert!((estimate.treatment_mean - 29.0 / 56.0).abs() < 1e-12);
    assert!((estimate.control_mean - 0.5).abs() < 1e-12);
    assert!((estimate.contrast - 1.0 / 56.0).abs() < 1e-12);
    assert_eq!(estimate.analysed_units, direct.analysed_units);
}
