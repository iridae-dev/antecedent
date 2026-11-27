//! Compiled sequential policy evaluation is the same computation as the kernel.
//!
//! Each `Study::compile_*` binds the declared inputs under `SequentialPolicyEvaluation`.
//! Executing passes the support gate of `sequential_evaluation.policy_evaluation` and calls the
//! kernel function the direct path calls, with the same inputs and the same
//! `ExecutionContext::for_tests(11)` seed the design crate's inspection-tree tests use. The
//! model is that tree without stopping boundaries (`base_spec(false)` in
//! `crates/antecedent-design/tests/sequential_evaluation.rs`): ship now is 0, repair now is 1,
//! inspect-then-act is 1.5, and collecting the inspection before choosing is worth 0.75 of
//! sample information. Those numbers are that file's hand oracle, not a new one.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(deprecated, reason = "the twin compares the direct kernel call with the compiled execute")]
#![allow(clippy::float_cmp, reason = "the bits are compared exactly, including the hand values")]

use antecedent::Study;
use antecedent::planner::PlanView;
use antecedent_contract::evidence_ledger::EvidenceLedger;
use antecedent_core::ExecutionContext;
use antecedent_core::measurement_model::{
    ArmBinding, ErrorStructure, MeasurementBinding, MeasurementKind, MeasurementModel,
    MeasurementParameters, MeasurementRoute, MeasurementVariable, ParameterEvidence,
    RepresentationVersion,
};
use antecedent_core::reason_details::sequential_evaluation::POLICY_INCOMPLETE;
use antecedent_design::sequential_evaluation::{
    ActionRole, ActionSpec, ActionSupport, CensoringRule, ComparisonKind, ComparisonSpec,
    ContingentPolicy, DelayedOutcome, EvaluationLimits, FiniteSequentialModel, HistoryKey,
    LoggedStep, LoggedTrajectory, ObservationChannel, PolicySpec, RolloutConfig, SequentialRequest,
    VoiRequest, compare_policies, evaluate_off_policy, evaluate_policy, rollout_policy,
    value_of_collecting_evidence,
};
use antecedent_design::sequential_evaluation_artifact::{
    OffPolicyInput, SequentialEvaluationArtifact,
};

const UNIT: &str = "sequential_evaluation.policy_evaluation";
const WAIT: usize = 0;
const INSPECT: usize = 1;
const SHIP: usize = 2;
const REPAIR: usize = 3;
const PILOT: usize = 4;
const NEGATIVE: usize = 0;
const ALARM: usize = 1;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(11)
}

fn assert_bound(view: &PlanView) {
    assert_eq!(view.route().as_str(), "SequentialPolicyEvaluation");
    let decision = view.unit_support().expect("the route is bound to a unit");
    assert_eq!(decision.unit(), UNIT);
    assert!(decision.supported(), "{UNIT} is supported");
}

fn channel(states: &[String]) -> ObservationChannel {
    ObservationChannel {
        measurement: MeasurementModel {
            model_id: "inspection_alarm".into(),
            route: MeasurementRoute::ProbabilisticCategorical,
            indicator: MeasurementVariable {
                id: "alarm".into(),
                kind: MeasurementKind::ProbabilisticFeatureAssignment,
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
            parameters: MeasurementParameters::Categorical {
                levels: states.to_vec(),
                confusion: vec![0.75, 0.25, 0.25, 0.75],
                column_counts: None,
                max_condition: 100.0,
            },
            representation: RepresentationVersion {
                coordinate_system_id: "alarm_panel".into(),
                version: 1,
            },
            evidence: ParameterEvidence::Declared { declarer: "reliability_engineering".into() },
        },
    }
}

/// What taking an action costs in utility and in the shared resource budget.
#[derive(Clone, Copy)]
struct Charge {
    utility: f64,
    resource: f64,
}

fn act(
    label: &str,
    role: ActionRole,
    charge: Charge,
    transition: [f64; 4],
    immediate: [f64; 2],
    delayed: Option<(usize, [f64; 2])>,
    observed: Option<ObservationChannel>,
) -> ActionSpec {
    ActionSpec {
        label: label.into(),
        role,
        utility_cost: charge.utility,
        resource_use: charge.resource,
        transition: transition.to_vec(),
        channel: observed,
        immediate_utility: immediate.to_vec(),
        delayed: delayed
            .map(|(delay, utility)| DelayedOutcome { delay, utility: utility.to_vec() }),
    }
}

/// The design crate's inspection tree with no stopping boundary.
fn model() -> FiniteSequentialModel {
    let states = vec!["sound".into(), "faulty".into()];
    let identity = [1.0, 0.0, 0.0, 1.0];
    FiniteSequentialModel {
        label: "inspection".into(),
        population: "plant_a".into(),
        regime: "operations".into(),
        period: "2026".into(),
        state_variable: "unit_condition".into(),
        states: states.clone(),
        initial: vec![0.5, 0.5],
        horizon: 3,
        actions: vec![
            act(
                "wait",
                ActionRole::Continue,
                Charge { utility: 0.0, resource: 0.0 },
                [0.75, 0.25, 0.0, 1.0],
                [0.0, 0.0],
                None,
                None,
            ),
            act(
                "inspect",
                ActionRole::Continue,
                Charge { utility: 0.25, resource: 1.0 },
                identity,
                [0.0, 0.0],
                None,
                Some(channel(&states)),
            ),
            act(
                "ship",
                ActionRole::Commit,
                Charge { utility: 0.0, resource: 0.0 },
                identity,
                [10.0, -10.0],
                None,
                None,
            ),
            act(
                "repair",
                ActionRole::Commit,
                Charge { utility: 4.0, resource: 0.0 },
                [1.0, 0.0, 0.5, 0.5],
                [0.0, 0.0],
                Some((1, [10.0, -10.0])),
                None,
            ),
            act(
                "pilot",
                ActionRole::Continue,
                Charge { utility: 0.25, resource: 0.0 },
                identity,
                [0.0, 0.0],
                Some((2, [3.0, -1.0])),
                None,
            ),
        ],
        terminal_utility: vec![2.0, -2.0],
        censoring: CensoringRule::Refuse,
        resource_budget: Some(2.0),
        stopping: Vec::new(),
        support: ActionSupport::everywhere(5),
    }
}

fn ship() -> ContingentPolicy {
    ContingentPolicy::new("act_now_ship").decide(HistoryKey::root(), SHIP)
}

fn repair() -> ContingentPolicy {
    ContingentPolicy::new("act_now_repair").decide(HistoryKey::root(), REPAIR)
}

fn inspect_then_act() -> ContingentPolicy {
    ContingentPolicy::new("inspect_then_act_v1")
        .decide(HistoryKey::root(), INSPECT)
        .decide(HistoryKey::root().extended(INSPECT, ALARM), REPAIR)
        .decide(HistoryKey::root().extended(INSPECT, NEGATIVE), SHIP)
}

fn ship_log() -> Vec<LoggedTrajectory> {
    vec![
        LoggedTrajectory {
            id: "sound".into(),
            steps: vec![LoggedStep {
                action: SHIP,
                observation: None,
                assignment_probability: 1.0,
            }],
            outcome_utility: 10.0,
        },
        LoggedTrajectory {
            id: "faulty".into(),
            steps: vec![LoggedStep {
                action: SHIP,
                observation: None,
                assignment_probability: 1.0,
            }],
            outcome_utility: -10.0,
        },
    ]
}

#[test]
fn compiled_g_formula_matches_evaluate_policy() {
    let model = model();
    let limits = EvaluationLimits::default();
    for (policy, hand) in [(ship(), 0.0), (repair(), 1.0), (inspect_then_act(), 1.5)] {
        let direct_ctx = ctx();
        let compiled_ctx = ctx();
        let direct = evaluate_policy(&model, &policy, limits, &direct_ctx).unwrap();
        let compiled =
            Study::compile_evaluate_policy(model.clone(), policy.clone(), limits, &compiled_ctx)
                .unwrap();
        assert_bound(&compiled.plan_view());
        let compiled = compiled.evaluate(&compiled_ctx).unwrap();
        assert_eq!(direct.value, hand);
        assert_eq!(direct, compiled);
    }
    // The indices are the design tree's, so an unused action is still part of the fixture.
    assert_eq!(model.actions[WAIT].label, "wait");
    assert_eq!(model.actions[PILOT].label, "pilot");
}

#[test]
fn compiled_off_policy_matches_evaluate_off_policy() {
    let model = model();
    let target = ship();
    let behaviour = ship();
    let log = ship_log();
    let limits = EvaluationLimits::default();
    let direct_ctx = ctx();
    let compiled_ctx = ctx();
    let mut direct_ledger = EvidenceLedger::new();
    let direct = evaluate_off_policy(
        &model,
        &target,
        &behaviour,
        &log,
        &mut direct_ledger,
        limits,
        &direct_ctx,
    )
    .unwrap();
    let compiled =
        Study::compile_evaluate_off_policy(model, target, behaviour, log, limits, &compiled_ctx)
            .unwrap();
    assert_bound(&compiled.plan_view());
    let mut compiled_ledger = EvidenceLedger::new();
    let compiled = compiled.evaluate(&mut compiled_ledger, &compiled_ctx).unwrap();
    assert_eq!(direct.ordinary, 0.0);
    assert_eq!(direct, compiled);
}

#[test]
fn compiled_rollout_matches_rollout_policy() {
    let model = model();
    let policy = ship();
    let config = RolloutConfig { seed: 11, rollouts: 8 };
    let direct_ctx = ctx();
    let compiled_ctx = ctx();
    let direct = rollout_policy(&model, &policy, config, &direct_ctx).unwrap();
    let compiled = Study::compile_rollout_policy(model, policy, config, &compiled_ctx).unwrap();
    assert_bound(&compiled.plan_view());
    assert_eq!(direct, compiled.evaluate(&compiled_ctx).unwrap());
}

#[test]
fn compiled_comparison_matches_compare_policies() {
    let model = model();
    let specs = vec![
        ComparisonSpec {
            label: "ship".into(),
            kind: ComparisonKind::ActNow,
            policy: PolicySpec::Contingent(ship()),
        },
        ComparisonSpec {
            label: "repair".into(),
            kind: ComparisonKind::ActNow,
            policy: PolicySpec::Contingent(repair()),
        },
    ];
    let request = SequentialRequest::default();
    let limits = EvaluationLimits::default();
    let direct_ctx = ctx();
    let compiled_ctx = ctx();
    let direct =
        compare_policies(&model, specs.clone(), None, &request, limits, &direct_ctx).unwrap();
    let compiled =
        Study::compile_compare_policies(model, specs, None, request, limits, &compiled_ctx)
            .unwrap();
    assert_bound(&compiled.plan_view());
    let compiled = compiled.evaluate(&compiled_ctx).unwrap();
    assert_eq!(direct.rows[0].value, 0.0);
    assert_eq!(direct.rows[1].value, 1.0);
    assert_eq!(direct.best, 1);
    assert_eq!(direct, compiled);
}

#[test]
fn compiled_value_of_evidence_matches_the_kernel() {
    let model = model();
    let request = VoiRequest { collect_action: INSPECT, commit_actions: vec![SHIP, REPAIR] };
    let limits = EvaluationLimits::default();
    let direct_ctx = ctx();
    let compiled_ctx = ctx();
    let direct = value_of_collecting_evidence(&model, &request, limits, &direct_ctx).unwrap();
    let compiled =
        Study::compile_value_of_collecting_evidence(model, request, limits, &compiled_ctx).unwrap();
    assert_bound(&compiled.plan_view());
    let compiled = compiled.evaluate(&compiled_ctx).unwrap();
    assert_eq!(direct.expected_value_of_sample_information, 0.75);
    assert_eq!(direct.enumerated_policy_value, 1.5);
    assert_eq!(direct, compiled);
}

#[test]
fn compiled_seal_matches_the_artifact_constructor() {
    let model = model();
    let policy = ship();
    let off_policy = OffPolicyInput { behaviour: ship(), trajectories: ship_log() };
    let rollout = RolloutConfig { seed: 11, rollouts: 4 };
    let limits = EvaluationLimits::default();
    let direct_ctx = ctx();
    let compiled_ctx = ctx();
    let direct = SequentialEvaluationArtifact::new(
        &model,
        &policy,
        Some(&off_policy),
        Some(rollout),
        limits,
        &direct_ctx,
    )
    .unwrap();
    let compiled = Study::compile_sequential_evaluation_seal(
        model,
        policy,
        Some(off_policy),
        Some(rollout),
        limits,
        &compiled_ctx,
    )
    .unwrap();
    assert_bound(&compiled.plan_view());
    assert_eq!(direct, compiled.seal(&compiled_ctx).unwrap());
}

#[test]
fn an_incomplete_policy_is_the_kernels_refusal() {
    let model = model();
    let policy = ContingentPolicy::new("incomplete");
    let limits = EvaluationLimits::default();
    let direct_ctx = ctx();
    let compiled_ctx = ctx();
    let direct = evaluate_policy(&model, &policy, limits, &direct_ctx).unwrap_err();
    let compiled = Study::compile_evaluate_policy(model, policy, limits, &compiled_ctx).unwrap();
    assert_bound(&compiled.plan_view());
    let compiled = compiled.evaluate(&compiled_ctx).unwrap_err();
    assert_eq!(direct.detail, POLICY_INCOMPLETE);
    assert_eq!(direct, compiled);
}
