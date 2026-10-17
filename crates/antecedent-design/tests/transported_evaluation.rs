//! Transported policy evaluation, temporal transport of a policy value and
//! adaptive-trial transport against hand-computed values.
//!
//! The decision problem is a two-state inspection. A unit is `sound` or
//! `faulty`. Actions: 0 `inspect` (continue, free, state unchanged, a binary
//! alarm with sensitivity and specificity three quarters), 1 `ship` (commit,
//! +10 sound, -10 faulty), 2 `hold` (commit, zero). Horizon two. The policy
//! inspects, ships on no alarm and holds on an alarm.
//!
//! With `p = P(faulty)` at the start the value is worked out by hand:
//! the no-alarm joint masses are `0.75 (1 - p)` (sound) and `0.25 p`
//! (faulty), so `V = 10 * 0.75 (1 - p) - 10 * 0.25 p = 7.5 - 10 p`.
//! The alarm branch holds and is worth zero. At `p = 1/2` the value is
//! `2.5`; at `p = 0.8` it is `-0.5`.
//!
//! The adaptive trial has eight units, two arms and strata `unit % 2`.
//! Weighted cell means are worked out in the test.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::too_many_lines,
    reason = "the oracles compare dyadic constants and each test is one scenario"
)]

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
    verify_adaptive_transport,
};
use antecedent_design::sequential_evaluation::{
    ActionRole, ActionSpec, ActionSupport, CensoringRule, ContingentPolicy, EvaluationLimits,
    FiniteSequentialModel, HistoryKey, HistoryStep, ObservationChannel,
};
use antecedent_design::temporal_policy_transport::{
    HistoryAlphabet, TemporalPolicyRequest, TemporalTarget, TimeInvariance,
    evaluate_temporal_policy, verify_temporal_policy,
};
use antecedent_design::transported_policy::{
    ActionUtility, MechanismStatus, ModelMechanism, SelectionDiagram, TargetUtility,
    TransportPremises, TransportedPolicyRequest, UtilityProvenance, evaluate_transported_policy,
    verify_transported_policy,
};

const INSPECT: usize = 0;
const SHIP: usize = 1;
const HOLD: usize = 2;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(5)
}

// ---------------------------------------------------------------------------
// The inspection model and policy
// ---------------------------------------------------------------------------

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

fn all_invariant_but_initial(
    model: &FiniteSequentialModel,
) -> Vec<(ModelMechanism, MechanismStatus)> {
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

fn diagram(model: &FiniteSequentialModel) -> SelectionDiagram {
    SelectionDiagram {
        source_population: "plant_a".into(),
        target_population: "plant_b".into(),
        mechanisms: all_invariant_but_initial(model),
    }
}

// ---------------------------------------------------------------------------
// Transported policy evaluation
// ---------------------------------------------------------------------------

#[test]
fn transported_value_matches_the_hand_value_across_target_laws() {
    let source = model([0.5, 0.5]);
    let policy = policy();
    let diagram = diagram(&source);
    for faulty in [0.0, 0.2, 0.5, 0.8, 1.0] {
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
        let evaluation = evaluate_transported_policy(&request, &ctx()).unwrap();
        let hand = 7.5 - 10.0 * faulty;
        assert!(
            (evaluation.target_value - hand).abs() < 1e-12,
            "p = {faulty}: {} vs {hand}",
            evaluation.target_value
        );
        assert!((evaluation.source_value - 2.5).abs() < 1e-12);
        assert_eq!(evaluation.inference, "point_only");
        assert_eq!(evaluation.shifted_mechanisms, vec![ModelMechanism::InitialState]);
        assert!(evaluation.nodes_visited > 0);
        assert_eq!(
            evaluation.nodes_visited,
            2 * 3,
            "root and two observed histories per enumeration"
        );
        verify_transported_policy(&evaluation, &request, &ctx()).unwrap();
    }
}

#[test]
fn missing_selection_contract_is_refused_by_name() {
    let source = model([0.5, 0.5]);
    let policy = policy();
    let law = [0.2, 0.8];
    let diagram = diagram(&source);
    let base = TransportedPolicyRequest {
        source: &source,
        policy: &policy,
        target_population: "plant_b",
        target_initial: Some(&law),
        diagram: Some(&diagram),
        target_utility: None,
        limits: EvaluationLimits::default(),
    };
    let err =
        evaluate_transported_policy(&TransportedPolicyRequest { diagram: None, ..base }, &ctx())
            .unwrap_err();
    assert_eq!(err.code, "transport_missing_evidence");
    assert_eq!(err.detail, "transported_policy.selection_diagram_missing");
    assert!(err.remedy.contains("selection diagram"));

    let err = evaluate_transported_policy(
        &TransportedPolicyRequest { target_initial: None, ..base },
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(err.code, "transport_missing_evidence");
    assert_eq!(err.detail, "transported_policy.target_initial_law_missing");

    // an undeclared mechanism is never assumed invariant
    let mut partial = diagram.clone();
    partial.mechanisms.retain(|(m, _)| *m != ModelMechanism::Observation { action: INSPECT });
    let err = evaluate_transported_policy(
        &TransportedPolicyRequest { diagram: Some(&partial), ..base },
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(err.detail, "transported_policy.mechanism_status_missing");
    assert!(err.message.contains("observation[0]"));

    // a diagram for another pair of populations
    let mut other = diagram.clone();
    other.target_population = "plant_c".into();
    let err = evaluate_transported_policy(
        &TransportedPolicyRequest { diagram: Some(&other), ..base },
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(err.detail, "transported_policy.diagram_binding_mismatch");

    // the model's own population is not a transport
    let err = evaluate_transported_policy(
        &TransportedPolicyRequest { target_population: "plant_a", ..base },
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(err.detail, "transported_policy.target_is_source");
}

#[test]
fn a_shifted_dynamic_mechanism_is_refused_with_the_mechanism_named() {
    let source = model([0.5, 0.5]);
    let policy = policy();
    let law = [0.2, 0.8];
    for shifted in [
        ModelMechanism::Transition { action: INSPECT },
        ModelMechanism::Observation { action: INSPECT },
    ] {
        let mut diagram = diagram(&source);
        for row in &mut diagram.mechanisms {
            if row.0 == shifted {
                row.1 = MechanismStatus::Shifted;
            }
        }
        let request = TransportedPolicyRequest {
            source: &source,
            policy: &policy,
            target_population: "plant_b",
            target_initial: Some(&law),
            diagram: Some(&diagram),
            target_utility: None,
            limits: EvaluationLimits::default(),
        };
        let err = evaluate_transported_policy(&request, &ctx()).unwrap_err();
        assert_eq!(err.code, "transport_missing_evidence");
        assert_eq!(err.detail, "transported_policy.mechanism_not_invariant");
        assert!(err.message.contains(&shifted.name()));
    }
}

#[test]
fn support_and_declaration_consistency_of_the_target_law_are_checked() {
    let policy = policy();
    // a source that never reached `faulty`: no evidence covers it
    let source = model([1.0, 0.0]);
    let law = [0.5, 0.5];
    let diagram = diagram(&source);
    let request = TransportedPolicyRequest {
        source: &source,
        policy: &policy,
        target_population: "plant_b",
        target_initial: Some(&law),
        diagram: Some(&diagram),
        target_utility: None,
        limits: EvaluationLimits::default(),
    };
    let err = evaluate_transported_policy(&request, &ctx()).unwrap_err();
    assert_eq!(err.code, "transport_support_failure");
    assert_eq!(err.detail, "transported_policy.target_support_outside_source");
    assert!(err.message.contains("faulty"));

    // a target law that differs although the initial law is declared invariant
    let source = model([0.5, 0.5]);
    let mut invariant = self::diagram(&source);
    invariant.mechanisms[0].1 = MechanismStatus::Invariant;
    let law = [0.2, 0.8];
    let request = TransportedPolicyRequest {
        source: &source,
        policy: &policy,
        target_population: "plant_b",
        target_initial: Some(&law),
        diagram: Some(&invariant),
        target_utility: None,
        limits: EvaluationLimits::default(),
    };
    let err = evaluate_transported_policy(&request, &ctx()).unwrap_err();
    assert_eq!(err.code, "invalid_argument");
    assert_eq!(err.detail, "transported_policy.initial_law_declared_invariant");

    // the same law is accepted as invariant
    let same = [0.5, 0.5];
    let request = TransportedPolicyRequest { target_initial: Some(&same), ..request };
    let evaluation = evaluate_transported_policy(&request, &ctx()).unwrap();
    assert!((evaluation.target_value - 2.5).abs() < 1e-12);

    // a law that does not sum to one
    let bad = [0.5, 0.4];
    let request = TransportedPolicyRequest { target_initial: Some(&bad), ..request };
    let err = evaluate_transported_policy(&request, &ctx()).unwrap_err();
    assert_eq!(err.detail, "transported_policy.target_initial_law");
}

#[test]
fn the_evaluation_identity_follows_the_inputs_and_replays() {
    let source = model([0.5, 0.5]);
    let policy = policy();
    let diagram = diagram(&source);
    let run = |law: &[f64]| {
        let request = TransportedPolicyRequest {
            source: &source,
            policy: &policy,
            target_population: "plant_b",
            target_initial: Some(law),
            diagram: Some(&diagram),
            target_utility: None,
            limits: EvaluationLimits::default(),
        };
        (
            evaluate_transported_policy(&request, &ctx()).unwrap(),
            request.target_initial.unwrap().to_vec(),
        )
    };
    let (first, _) = run(&[0.2, 0.8]);
    let (again, _) = run(&[0.2, 0.8]);
    let (changed, _) = run(&[0.3, 0.7]);
    assert_eq!(first.identity, again.identity);
    assert_ne!(first.identity, changed.identity, "same shape, changed law");
    assert_ne!(first.target_value, changed.target_value);

    // a stored value that was mutated does not replay
    let law = [0.2, 0.8];
    let request = TransportedPolicyRequest {
        source: &source,
        policy: &policy,
        target_population: "plant_b",
        target_initial: Some(&law),
        diagram: Some(&diagram),
        target_utility: None,
        limits: EvaluationLimits::default(),
    };
    let mut forged = first.clone();
    forged.target_value += 0.5;
    let err = verify_transported_policy(&forged, &request, &ctx()).unwrap_err();
    assert_eq!(err.detail, "transported_policy.value_mismatch");

    // the diagram identity does not depend on listing order
    let mut reversed = diagram.clone();
    reversed.mechanisms.reverse();
    assert_eq!(diagram.identity_hex(), reversed.identity_hex());
}

// ---------------------------------------------------------------------------
// Target-native utility
// ---------------------------------------------------------------------------
//
// The target values the outcomes differently: shipping a sound unit is worth
// +6 and a faulty one costs -14, holding costs 1 whatever the unit, inspecting
// costs 0.5 and has a small state-dependent immediate value (+0.4 sound,
// -0.4 faulty). The policy inspects once, ships on a quiet panel and holds on
// an alarm.
//
// Dynamic programming over the two-stage tree, by hand. With prior faulty
// mass p, the alarm fires with probability 1/4 on a sound unit and 3/4 on a
// faulty one. Stage one (inspect) earns -0.5 + 0.4 (1 - p) - 0.4 p whatever
// is observed. Stage two earns, by branch,
//   quiet:  ship, 6 (sound) or -14 (faulty), weights 3/4 (1 - p) and 1/4 p;
//   alarm:  hold, -1 either way, weight 1/4 (1 - p) + 3/4 p.
// So V(p) = -0.5 + 0.4 (1 - 2p) + 4.5 (1 - p) - 3.5 p - (1/4 (1 - p) + 3/4 p).
// At p = 0.2 the first stage is -0.5 + 0.24 = -0.26, the quiet branch is
// 3.6 - 0.7 = 2.9 and the alarm branch is -(0.2 + 0.15) = -0.35, so V = 2.29.

const TARGET_DIGEST: &str = "5d41402abc4b2a76b9719d911017c5925d41402abc4b2a76b9719d911017c592";

fn target_utility(digest: &str) -> TargetUtility {
    let action = |cost, immediate: [f64; 2]| ActionUtility {
        cost,
        immediate: immediate.to_vec(),
        delayed: None,
    };
    TargetUtility {
        provenance: UtilityProvenance {
            source: "plant_b_loss_schedule".into(),
            digest: digest.into(),
        },
        terminal: vec![0.0, 0.0],
        actions: vec![
            action(0.5, [0.4, -0.4]),
            action(0.0, [6.0, -14.0]),
            action(0.0, [-1.0, -1.0]),
        ],
    }
}

/// The value by explicit dynamic programming over the two-stage tree: the
/// sum over the latent state and the panel outcome, no library code.
fn target_value_by_hand(p_faulty: f64) -> f64 {
    let mut total = 0.0;
    for (faulty, mass) in [(false, 1.0 - p_faulty), (true, p_faulty)] {
        let alarm = if faulty { 0.75 } else { 0.25 };
        let state = usize::from(faulty);
        let first = -0.5 + [0.4, -0.4][state];
        let ship = [6.0, -14.0][state];
        let hold = -1.0;
        total += mass * (first + (1.0 - alarm) * ship + alarm * hold);
    }
    total
}

fn diagram_with_shifted_utility(model: &FiniteSequentialModel) -> SelectionDiagram {
    let mut diagram = diagram(model);
    for row in &mut diagram.mechanisms {
        if row.0 == ModelMechanism::Utility {
            row.1 = MechanismStatus::Shifted;
        }
    }
    diagram
}

#[test]
fn a_target_native_utility_is_evaluated_against_hand_dynamic_programming() {
    let source = model([0.5, 0.5]);
    let policy = policy();
    let diagram = diagram_with_shifted_utility(&source);
    let utility = target_utility(TARGET_DIGEST);
    assert!((target_value_by_hand(0.2) - 2.29).abs() < 1e-12);
    for faulty in [0.0, 0.2, 0.5, 0.8, 1.0] {
        let law = [1.0 - faulty, faulty];
        let request = TransportedPolicyRequest {
            source: &source,
            policy: &policy,
            target_population: "plant_b",
            target_initial: Some(&law),
            diagram: Some(&diagram),
            target_utility: Some(&utility),
            limits: EvaluationLimits::default(),
        };
        let evaluation = evaluate_transported_policy(&request, &ctx()).unwrap();
        let hand = target_value_by_hand(faulty);
        assert!(
            (evaluation.target_value - hand).abs() < 1e-12,
            "p = {faulty}: {} vs {hand}",
            evaluation.target_value
        );
        // The source keeps its own utility: 7.5 - 10 * 0.5 = 2.5.
        assert!((evaluation.source_value - 2.5).abs() < 1e-12);
        assert_eq!(
            evaluation.shifted_mechanisms,
            vec![ModelMechanism::InitialState, ModelMechanism::Utility]
        );
        assert_eq!(evaluation.target_utility.as_ref(), Some(&utility));
        verify_transported_policy(&evaluation, &request, &ctx()).unwrap();
    }

    // The utility alone shifts: with the source's law as the target's initial
    // law the value is the source model's under the target's valuation.
    let mut utility_only = diagram.clone();
    utility_only.mechanisms[0].1 = MechanismStatus::Invariant;
    let same = [0.5, 0.5];
    let request = TransportedPolicyRequest {
        source: &source,
        policy: &policy,
        target_population: "plant_b",
        target_initial: Some(&same),
        diagram: Some(&utility_only),
        target_utility: Some(&utility),
        limits: EvaluationLimits::default(),
    };
    let evaluation = evaluate_transported_policy(&request, &ctx()).unwrap();
    assert!((evaluation.target_value - target_value_by_hand(0.5)).abs() < 1e-12);
    assert_eq!(evaluation.shifted_mechanisms, vec![ModelMechanism::Utility]);
}

#[test]
fn a_shifted_utility_needs_its_target_utility_and_relaxes_nothing_else() {
    let source = model([0.5, 0.5]);
    let policy = policy();
    let law = [0.2, 0.8];
    let shifted = diagram_with_shifted_utility(&source);
    let utility = target_utility(TARGET_DIGEST);
    let base = TransportedPolicyRequest {
        source: &source,
        policy: &policy,
        target_population: "plant_b",
        target_initial: Some(&law),
        diagram: Some(&shifted),
        target_utility: Some(&utility),
        limits: EvaluationLimits::default(),
    };

    // shifted, none supplied
    let err = evaluate_transported_policy(
        &TransportedPolicyRequest { target_utility: None, ..base },
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(err.code, "transport_missing_evidence");
    assert_eq!(err.detail, "transported_policy.target_utility_missing");

    // supplied, declared invariant
    let invariant = diagram(&source);
    let err = evaluate_transported_policy(
        &TransportedPolicyRequest { diagram: Some(&invariant), ..base },
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(err.code, "invalid_argument");
    assert_eq!(err.detail, "transported_policy.utility_declared_invariant");

    // a shifted transition or observation stays refused, utility supplied or not
    for dynamic in [
        ModelMechanism::Transition { action: INSPECT },
        ModelMechanism::Observation { action: INSPECT },
    ] {
        let mut diagram = shifted.clone();
        for row in &mut diagram.mechanisms {
            if row.0 == dynamic {
                row.1 = MechanismStatus::Shifted;
            }
        }
        let err = evaluate_transported_policy(
            &TransportedPolicyRequest { diagram: Some(&diagram), ..base },
            &ctx(),
        )
        .unwrap_err();
        assert_eq!(err.detail, "transported_policy.mechanism_not_invariant");
        assert!(err.message.contains(&dynamic.name()));
    }

    // the target law still has to lie in the source's support
    let narrow = model([1.0, 0.0]);
    let narrow_diagram = diagram_with_shifted_utility(&narrow);
    let err = evaluate_transported_policy(
        &TransportedPolicyRequest {
            source: &narrow,
            diagram: Some(&narrow_diagram),
            target_initial: Some(&[0.5, 0.5]),
            ..base
        },
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(err.detail, "transported_policy.target_support_outside_source");
}

#[test]
fn a_target_utility_must_match_the_source_and_declare_its_provenance() {
    let source = model([0.5, 0.5]);
    let policy = policy();
    let law = [0.2, 0.8];
    let diagram = diagram_with_shifted_utility(&source);
    let refusal = |utility: &TargetUtility| {
        let request = TransportedPolicyRequest {
            source: &source,
            policy: &policy,
            target_population: "plant_b",
            target_initial: Some(&law),
            diagram: Some(&diagram),
            target_utility: Some(utility),
            limits: EvaluationLimits::default(),
        };
        evaluate_transported_policy(&request, &ctx()).unwrap_err()
    };
    let good = target_utility(TARGET_DIGEST);

    let mut unsourced = good.clone();
    unsourced.provenance.source = "  ".into();
    let err = refusal(&unsourced);
    assert_eq!(err.code, "transport_missing_evidence");
    assert_eq!(err.detail, "transported_policy.target_utility_provenance_missing");

    for digest in ["", "abc", &"G".repeat(64), &TARGET_DIGEST.to_uppercase()] {
        let mut bad = good.clone();
        bad.provenance.digest = digest.into();
        let err = refusal(&bad);
        assert_eq!(err.detail, "transported_policy.target_utility_digest", "{digest}");
    }

    // the target cannot add or drop an action, add a state or invent a delayed outcome
    let mut dropped = good.clone();
    dropped.actions.pop();
    let mut extra_state = good.clone();
    extra_state.terminal.push(0.0);
    let mut short_immediate = good.clone();
    short_immediate.actions[SHIP].immediate.pop();
    let mut invented_delay = good.clone();
    invented_delay.actions[SHIP].delayed = Some(vec![1.0, 1.0]);
    for bad in [dropped, extra_state, short_immediate, invented_delay] {
        assert_eq!(refusal(&bad).detail, "transported_policy.target_utility_support");
    }

    let mut infinite = good.clone();
    infinite.actions[SHIP].immediate[0] = f64::INFINITY;
    let mut not_a_number = good.clone();
    not_a_number.terminal[1] = f64::NAN;
    let mut negative_cost = good.clone();
    negative_cost.actions[INSPECT].cost = -0.5;
    for bad in [infinite, not_a_number, negative_cost] {
        let err = refusal(&bad);
        assert_eq!(err.code, "invalid_argument");
        assert_eq!(err.detail, "transported_policy.target_utility_value");
    }
}

#[test]
fn the_target_utility_and_its_provenance_are_bound_into_the_identity_and_replay() {
    let source = model([0.5, 0.5]);
    let policy = policy();
    let law = [0.2, 0.8];
    let diagram = diagram_with_shifted_utility(&source);
    let utility = target_utility(TARGET_DIGEST);
    let request = |source: &FiniteSequentialModel, utility: &TargetUtility| {
        evaluate_transported_policy(
            &TransportedPolicyRequest {
                source,
                policy: &policy,
                target_population: "plant_b",
                target_initial: Some(&law),
                diagram: Some(&diagram),
                target_utility: Some(utility),
                limits: EvaluationLimits::default(),
            },
            &ctx(),
        )
        .unwrap()
    };
    let first = request(&source, &utility);
    assert_eq!(first.identity, request(&source, &utility).identity);

    // Provenance alone changes the identity, not the value.
    let mut relabelled = utility.clone();
    relabelled.provenance.source = "plant_b_loss_schedule_v2".into();
    let mut redigested = utility.clone();
    redigested.provenance.digest = "0".repeat(64);
    for other in [&relabelled, &redigested] {
        let changed = request(&source, other);
        assert_ne!(changed.identity, first.identity);
        assert_eq!(changed.target_value.to_bits(), first.target_value.to_bits());
    }

    // Values change both.
    let mut repriced = utility.clone();
    repriced.actions[SHIP].immediate[1] = -20.0;
    let changed = request(&source, &repriced);
    assert_ne!(changed.identity, first.identity);
    assert!(changed.target_value < first.target_value);

    // A target-utility evaluation and a utility-invariant one never share an identity.
    let invariant = self::diagram(&source);
    let plain = evaluate_transported_policy(
        &TransportedPolicyRequest {
            source: &source,
            policy: &policy,
            target_population: "plant_b",
            target_initial: Some(&law),
            diagram: Some(&invariant),
            target_utility: None,
            limits: EvaluationLimits::default(),
        },
        &ctx(),
    )
    .unwrap();
    assert_ne!(plain.identity, first.identity);

    // Replay: the same request verifies; a tampered utility, in the request or
    // in the stored evaluation, does not.
    let replay = |stored: &antecedent_design::transported_policy::TransportedPolicyEvaluation,
                  utility: &TargetUtility| {
        verify_transported_policy(
            stored,
            &TransportedPolicyRequest {
                source: &source,
                policy: &policy,
                target_population: "plant_b",
                target_initial: Some(&law),
                diagram: Some(&diagram),
                target_utility: Some(utility),
                limits: EvaluationLimits::default(),
            },
            &ctx(),
        )
    };
    replay(&first, &utility).unwrap();
    for tampered in [&repriced, &relabelled, &redigested] {
        let err = replay(&first, tampered).unwrap_err();
        assert_eq!(err.detail, "transported_policy.value_mismatch");
    }
    let mut forged = first.clone();
    forged.target_utility.as_mut().unwrap().provenance.source = "someone_else".into();
    assert_eq!(replay(&forged, &utility).unwrap_err().detail, "transported_policy.value_mismatch");
    let mut dropped = first.clone();
    dropped.target_utility = None;
    assert_eq!(replay(&dropped, &utility).unwrap_err().detail, "transported_policy.value_mismatch");
}

#[test]
fn refreshing_the_source_model_keeps_the_utility_declaration_and_recomputes_the_rest() {
    // Outcome-only data moves the source prior; the policy, the diagram, the
    // target law and the target utility are kept as declarations.
    let policy = policy();
    let law = [0.2, 0.8];
    let utility = target_utility(TARGET_DIGEST);
    let run = |source: &FiniteSequentialModel| {
        let diagram = diagram_with_shifted_utility(source);
        let request = TransportedPolicyRequest {
            source,
            policy: &policy,
            target_population: "plant_b",
            target_initial: Some(&law),
            diagram: Some(&diagram),
            target_utility: Some(&utility),
            limits: EvaluationLimits::default(),
        };
        (evaluate_transported_policy(&request, &ctx()).unwrap(), diagram)
    };
    let before = model([0.5, 0.5]);
    let after = model([0.6, 0.4]);
    let (old, old_diagram) = run(&before);
    let (new, _) = run(&after);
    // The target value depends on the target law, the target utility and the
    // invariant mechanisms, none of which moved; the source value and the
    // identity follow the refreshed model.
    assert_eq!(old.target_value.to_bits(), new.target_value.to_bits());
    assert!((new.target_value - target_value_by_hand(0.8)).abs() < 1e-12);
    assert!((old.source_value - 2.5).abs() < 1e-12);
    assert!((new.source_value - (7.5 - 10.0 * 0.4)).abs() < 1e-12);
    assert_ne!(old.identity, new.identity);
    assert_eq!(new.target_utility, old.target_utility);

    // The stale evaluation no longer replays against the refreshed model.
    let request = TransportedPolicyRequest {
        source: &after,
        policy: &policy,
        target_population: "plant_b",
        target_initial: Some(&law),
        diagram: Some(&old_diagram),
        target_utility: Some(&utility),
        limits: EvaluationLimits::default(),
    };
    assert_eq!(
        verify_transported_policy(&old, &request, &ctx()).unwrap_err().detail,
        "transported_policy.value_mismatch"
    );
    verify_transported_policy(&new, &request, &ctx()).unwrap();
}

#[test]
fn a_declared_target_utility_is_validated_then_accepted_and_echoed() {
    let source = model([0.5, 0.5]);
    let policy = policy();
    let law = [0.2, 0.8];
    let diagram = diagram_with_shifted_utility(&source);
    let utility = target_utility(TARGET_DIGEST);
    let request = TransportedPolicyRequest {
        source: &source,
        policy: &policy,
        target_population: "plant_b",
        target_initial: Some(&law),
        diagram: Some(&diagram),
        target_utility: Some(&utility),
        limits: EvaluationLimits::default(),
    };
    // Validation passes for a declaration that matches the source's actions and states,
    // carries provenance and has finite values: the evaluation runs and reports the
    // declaration it accepted.
    let evaluation = evaluate_transported_policy(&request, &ctx()).unwrap();
    assert_eq!(evaluation.target_utility.as_ref(), Some(&utility));
    assert!((evaluation.target_value - target_value_by_hand(0.8)).abs() < 1e-12);
    verify_transported_policy(&evaluation, &request, &ctx()).unwrap();

    // Each validated property, broken alone, is refused with its own detail.
    let broken = |edit: &dyn Fn(&mut TargetUtility)| {
        let mut bad = utility.clone();
        edit(&mut bad);
        evaluate_transported_policy(
            &TransportedPolicyRequest { target_utility: Some(&bad), ..request },
            &ctx(),
        )
        .unwrap_err()
        .detail
    };
    assert_eq!(
        broken(&|u| u.provenance.source.clear()),
        "transported_policy.target_utility_provenance_missing"
    );
    assert_eq!(
        broken(&|u| u.provenance.digest = "zz".into()),
        "transported_policy.target_utility_digest"
    );
    assert_eq!(broken(&|u| u.terminal.push(0.0)), "transported_policy.target_utility_support");
    assert_eq!(
        broken(&|u| u.actions[SHIP].immediate[0] = f64::NAN),
        "transported_policy.target_utility_value"
    );
}

#[test]
fn a_target_utility_keeps_the_transition_observation_and_support_premises() {
    let source = model([0.5, 0.5]);
    let policy = policy();
    let law = [0.2, 0.8];
    let diagram = diagram_with_shifted_utility(&source);
    let utility = target_utility(TARGET_DIGEST);
    let request = TransportedPolicyRequest {
        source: &source,
        policy: &policy,
        target_population: "plant_b",
        target_initial: Some(&law),
        diagram: Some(&diagram),
        target_utility: Some(&utility),
        limits: EvaluationLimits::default(),
    };
    let evaluation = evaluate_transported_policy(&request, &ctx()).unwrap();
    // The premises the value is conditional on still name the invariant transitions and
    // observation channels, the target-initial-law overlap and the g-formula premises;
    // only the utility moved out of the invariant set.
    let premises = evaluation.premises;
    assert!(premises.invariance.contains("transitions, observation channels"));
    assert!(premises.invariance.contains("unless a target utility is supplied"));
    assert!(premises.selection.contains("when a target utility is supplied"));
    assert!(premises.overlap.contains("absolutely continuous"));
    assert_eq!(premises, TransportPremises::DECLARED);
    assert_eq!(
        evaluation.shifted_mechanisms,
        vec![ModelMechanism::InitialState, ModelMechanism::Utility]
    );

    // A shifted transition or observation is still refused with the mechanism named, and a
    // target law outside the source's support is still refused.
    for dynamic in [
        ModelMechanism::Transition { action: INSPECT },
        ModelMechanism::Observation { action: INSPECT },
    ] {
        let mut shifted = diagram.clone();
        for row in &mut shifted.mechanisms {
            if row.0 == dynamic {
                row.1 = MechanismStatus::Shifted;
            }
        }
        let err = evaluate_transported_policy(
            &TransportedPolicyRequest { diagram: Some(&shifted), ..request },
            &ctx(),
        )
        .unwrap_err();
        assert_eq!(err.detail, "transported_policy.mechanism_not_invariant");
        assert!(err.message.contains(&dynamic.name()));
    }
    let narrow = model([1.0, 0.0]);
    let narrow_diagram = diagram_with_shifted_utility(&narrow);
    let err = evaluate_transported_policy(
        &TransportedPolicyRequest {
            source: &narrow,
            diagram: Some(&narrow_diagram),
            target_initial: Some(&[0.5, 0.5]),
            ..request
        },
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(err.detail, "transported_policy.target_support_outside_source");
}

// ---------------------------------------------------------------------------
// Temporal transport
// ---------------------------------------------------------------------------

fn invariance(model: &FiniteSequentialModel) -> TimeInvariance {
    TimeInvariance {
        source_period: "2026".into(),
        target_period: "2027".into(),
        horizon: model.horizon,
        mechanisms: all_invariant_but_initial(model),
    }
}

fn target(model: &FiniteSequentialModel) -> TemporalTarget {
    TemporalTarget {
        period: "2027".into(),
        horizon: model.horizon,
        alphabet: HistoryAlphabet::of_model(model).unwrap(),
    }
}

#[test]
fn a_period_shift_under_time_invariant_mechanisms_has_the_hand_value() {
    let source = model([0.5, 0.5]);
    let policy = policy();
    let invariance = invariance(&source);
    let target = target(&source);
    let law = [0.4, 0.6];
    let request = TemporalPolicyRequest {
        source: &source,
        policy: &policy,
        target: &target,
        target_initial: Some(&law),
        invariance: Some(&invariance),
        limits: EvaluationLimits::default(),
    };
    let evaluation = evaluate_temporal_policy(&request, &ctx()).unwrap();
    // 7.5 - 10 * 0.6
    assert!((evaluation.target_value - 1.5).abs() < 1e-12);
    assert!((evaluation.source_value - 2.5).abs() < 1e-12);
    assert_eq!(evaluation.horizon, 2);
    assert_eq!(evaluation.inference, "point_only");
    verify_temporal_policy(&evaluation, &request, &ctx()).unwrap();

    let law_changed = [0.45, 0.55];
    let changed = evaluate_temporal_policy(
        &TemporalPolicyRequest { target_initial: Some(&law_changed), ..request },
        &ctx(),
    )
    .unwrap();
    assert_ne!(evaluation.identity, changed.identity);
}

#[test]
fn time_varying_horizon_and_history_incompatibilities_are_refused() {
    let source = model([0.5, 0.5]);
    let policy = policy();
    let invariance_ok = invariance(&source);
    let target_ok = target(&source);
    let law = [0.4, 0.6];
    let base = TemporalPolicyRequest {
        source: &source,
        policy: &policy,
        target: &target_ok,
        target_initial: Some(&law),
        invariance: Some(&invariance_ok),
        limits: EvaluationLimits::default(),
    };

    let err = evaluate_temporal_policy(&TemporalPolicyRequest { invariance: None, ..base }, &ctx())
        .unwrap_err();
    assert_eq!(err.code, "transport_missing_evidence");
    assert_eq!(err.detail, "temporal_policy_transport.time_invariance_missing");

    // a time-varying transition
    let mut varying = invariance_ok.clone();
    for row in &mut varying.mechanisms {
        if row.0 == (ModelMechanism::Transition { action: INSPECT }) {
            row.1 = MechanismStatus::Shifted;
        }
    }
    let err = evaluate_temporal_policy(
        &TemporalPolicyRequest { invariance: Some(&varying), ..base },
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(err.code, "transport_missing_evidence");
    assert_eq!(err.detail, "temporal_policy_transport.mechanism_time_varying");
    assert!(err.message.contains("transition[0]"));

    // a different horizon in the target period
    let mut longer = target_ok.clone();
    longer.horizon = 3;
    let err = evaluate_temporal_policy(&TemporalPolicyRequest { target: &longer, ..base }, &ctx())
        .unwrap_err();
    assert_eq!(err.code, "route_not_supported");
    assert_eq!(err.detail, "temporal_policy_transport.horizon_mismatch");

    // histories over a different alphabet
    let mut renamed = target_ok.clone();
    renamed.alphabet.actions[INSPECT].symbols.push("fault_code".into());
    let err = evaluate_temporal_policy(&TemporalPolicyRequest { target: &renamed, ..base }, &ctx())
        .unwrap_err();
    assert_eq!(err.detail, "temporal_policy_transport.history_alphabet_incompatible");
    let mut fewer = target_ok.clone();
    fewer.alphabet.actions.pop();
    let err = evaluate_temporal_policy(&TemporalPolicyRequest { target: &fewer, ..base }, &ctx())
        .unwrap_err();
    assert_eq!(err.detail, "temporal_policy_transport.history_alphabet_incompatible");

    // a declaration for other periods
    let mut wrong = invariance_ok.clone();
    wrong.target_period = "2030".into();
    let err = evaluate_temporal_policy(
        &TemporalPolicyRequest { invariance: Some(&wrong), ..base },
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(err.detail, "temporal_policy_transport.invariance_binding_mismatch");

    // the model's own period
    let mut same = target_ok.clone();
    same.period = "2026".into();
    let err = evaluate_temporal_policy(&TemporalPolicyRequest { target: &same, ..base }, &ctx())
        .unwrap_err();
    assert_eq!(err.detail, "temporal_policy_transport.target_is_source");

    // the existing request check still names the missing contract
    assert!(
        antecedent_design::sequential_evaluation::SequentialRequest {
            target_population: None,
            target_period: Some("2027".into()),
        }
        .check(&source)
        .is_err()
    );
}

// ---------------------------------------------------------------------------
// Adaptive trial as a transport source
// ---------------------------------------------------------------------------

fn digest(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

fn cell(
    assignment: AssignmentFamily,
    monitoring: MonitoringStrategy,
    stopping: StoppingRuleKind,
) -> ProcedureCell {
    ProcedureCell { assignment, outcome: OutcomeStructure::SingleOutcome, monitoring, stopping }
}

fn epsilon_greedy() -> ProcedureCell {
    cell(
        AssignmentFamily::EpsilonGreedy,
        MonitoringStrategy::FixedInterimLooks,
        StoppingRuleKind::Efficacy,
    )
}

/// Eight units, arms {control, treatment}. Units 0-3 are assigned at
/// (0.5, 0.5), units 4-5 at (0.2, 0.8) and units 6-7 at (0.4, 0.6) under an
/// adaptive cell; every unit is at (0.5, 0.5) under fixed randomization.
fn declaration(cell: ProcedureCell) -> AdaptiveDesignDeclaration {
    let assigned_arm = vec![0, 1, 0, 1, 1, 0, 1, 0];
    let adaptive = cell.assignment != AssignmentFamily::FixedRandomization;
    let probabilities = (0..8)
        .map(|unit| match (adaptive, unit) {
            (true, 4 | 5) => vec![0.2, 0.8],
            (true, 6 | 7) => vec![0.4, 0.6],
            _ => vec![0.5, 0.5],
        })
        .collect();
    let final_only = cell.monitoring == MonitoringStrategy::FinalAnalysisOnly;
    let looks = if final_only {
        vec![InterimAnalysis { units_enrolled: 8, information_fraction: 1.0 }]
    } else {
        vec![
            InterimAnalysis { units_enrolled: 4, information_fraction: 0.5 },
            InterimAnalysis { units_enrolled: 8, information_fraction: 1.0 },
        ]
    };
    let available_at_look = (0..8).map(|unit| Some(u32::from(!(final_only || unit < 4)))).collect();
    let outcomes = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0].iter().map(|&y| vec![y]).collect();
    let spec = match cell.stopping {
        StoppingRuleKind::Efficacy => StoppingSpec::PosteriorProbabilityBest { threshold: 0.95 },
        _ => StoppingSpec::NoStopping,
    };
    AdaptiveDesignDeclaration {
        label: "trial".to_owned(),
        cell,
        arm_labels: vec!["control".to_owned(), "treatment".to_owned()],
        assigned_arm: assigned_arm.clone(),
        policy_history: Some(vec![
            PolicyEpoch { first_unit: 0, policy_identity: digest(1) },
            PolicyEpoch { first_unit: 4, policy_identity: digest(2) },
        ]),
        assignment_probabilities: Some(probabilities),
        exposure: Some(assigned_arm),
        availability: Some(OutcomeAvailability {
            available_at_look,
            delay: DelayMechanism::Immediate,
        }),
        outcomes,
        interim_analyses: Some(looks),
        stopping_rule: Some(StoppingRuleDeclaration {
            kind: cell.stopping,
            spec,
            ended_by_rule: false,
        }),
        probability_floor: 0.05,
    }
}

fn trial_diagram() -> TrialSelectionDiagram {
    TrialSelectionDiagram {
        source_population: "trial_sites".into(),
        target_population: "clinic_network".into(),
        mechanisms: vec![
            (TrialMechanism::StratumLaw, MechanismStatus::Shifted),
            (TrialMechanism::OutcomeGivenStratumAndArm, MechanismStatus::Invariant),
        ],
    }
}

fn strata() -> (Vec<usize>, Vec<String>) {
    ((0..8).map(|unit| unit % 2).collect(), vec!["mild".into(), "severe".into()])
}

#[test]
fn adaptive_trial_means_standardize_to_the_target_stratum_law() {
    // Stratum 0 = units 0, 2, 4, 6; stratum 1 = units 1, 3, 5, 7.
    // Arm 1 (treatment) in stratum 0: unit 4 (y 0, pi 0.8) and unit 6
    //   (y 1, pi 0.6): weights 1.25 and 5/3, mean (5/3) / (35/12) = 4/7.
    // Arm 1 in stratum 1: units 1 and 3 at pi 0.5, y 0 and 1: mean 1/2.
    // Arm 0 (control) in stratum 0: units 0 and 2 at pi 0.5, y 1 and 1: 1.
    // Arm 0 in stratum 1: unit 5 (y 0, pi 0.2) and unit 7 (y 1, pi 0.4):
    //   weights 5 and 2.5, mean 2.5 / 7.5 = 1/3.
    let contract = AdaptiveDesignContract::new(declaration(epsilon_greedy())).unwrap();
    let (unit_strata, labels) = strata();
    let diagram = trial_diagram();
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
    let estimate = transport_adaptive_trial(&request).unwrap();
    // treatment: 0.25 * 4/7 + 0.75 * 1/2 = 29/56; control: 0.25 + 0.75/3 = 1/2
    assert!((estimate.treatment_mean - 29.0 / 56.0).abs() < 1e-12);
    assert!((estimate.control_mean - 0.5).abs() < 1e-12);
    assert!((estimate.contrast - 1.0 / 56.0).abs() < 1e-12);
    assert_eq!(estimate.analysed_units, 8);
    assert_eq!(estimate.cells.len(), 4);
    assert_eq!(estimate.inference, "point_only");
    verify_adaptive_transport(&estimate, &request).unwrap();

    // a target of only mild units uses only stratum 0 cells
    let mild = [1.0, 0.0];
    let request_mild = AdaptiveTransportRequest { target_stratum_law: Some(&mild), ..request };
    let estimate_mild = transport_adaptive_trial(&request_mild).unwrap();
    assert!((estimate_mild.treatment_mean - 4.0 / 7.0).abs() < 1e-12);
    assert!((estimate_mild.control_mean - 1.0).abs() < 1e-12);
    assert_eq!(estimate_mild.cells.len(), 2);
    assert_ne!(estimate.identity, estimate_mild.identity);
}

#[test]
fn the_weighting_uses_the_recorded_probabilities_not_the_arm_counts() {
    // The same data under fixed (0.5, 0.5) recorded probabilities, relabelled
    // as an adaptive cell, weights every unit equally: the stratum 0
    // treatment mean is the plain average 1/2 rather than 4/7.
    let mut flat = declaration(epsilon_greedy());
    flat.assignment_probabilities = Some(vec![vec![0.5, 0.5]; 8]);
    let contract = AdaptiveDesignContract::new(flat).unwrap();
    let (unit_strata, labels) = strata();
    let diagram = trial_diagram();
    let law = [1.0, 0.0];
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
    let estimate = transport_adaptive_trial(&request).unwrap();
    assert!((estimate.treatment_mean - 0.5).abs() < 1e-12);
}

#[test]
fn an_adaptive_source_without_its_contract_or_with_open_questions_is_refused() {
    let contract = AdaptiveDesignContract::new(declaration(epsilon_greedy())).unwrap();
    let (unit_strata, labels) = strata();
    let diagram = trial_diagram();
    let law = [0.25, 0.75];
    let base = AdaptiveTransportRequest {
        contract: Some(&contract),
        trial_population: "trial_sites",
        unit_strata: &unit_strata,
        stratum_labels: &labels,
        target_stratum_law: Some(&law),
        diagram: Some(&diagram),
        treatment_arm: 1,
        control_arm: 0,
    };

    let err =
        transport_adaptive_trial(&AdaptiveTransportRequest { contract: None, ..base }).unwrap_err();
    assert_eq!(err.code, "adaptive_contract_incomplete");
    assert_eq!(err.detail, "adaptive_transport.assignment_monitoring_contract_missing");
    assert!(err.remedy.contains("realized assignment probabilities"));

    let err =
        transport_adaptive_trial(&AdaptiveTransportRequest { diagram: None, ..base }).unwrap_err();
    assert_eq!(err.detail, "adaptive_transport.selection_diagram_missing");
    let err =
        transport_adaptive_trial(&AdaptiveTransportRequest { target_stratum_law: None, ..base })
            .unwrap_err();
    assert_eq!(err.detail, "adaptive_transport.target_stratum_law_missing");

    // a shifted outcome mechanism cannot be supplied by the trial
    let mut shifted = trial_diagram();
    shifted.mechanisms[1].1 = MechanismStatus::Shifted;
    let err =
        transport_adaptive_trial(&AdaptiveTransportRequest { diagram: Some(&shifted), ..base })
            .unwrap_err();
    assert_eq!(err.code, "transport_missing_evidence");
    assert_eq!(err.detail, "adaptive_transport.outcome_mechanism_not_invariant");

    // an undeclared mechanism
    let mut partial = trial_diagram();
    partial.mechanisms.pop();
    let err =
        transport_adaptive_trial(&AdaptiveTransportRequest { diagram: Some(&partial), ..base })
            .unwrap_err();
    assert_eq!(err.detail, "adaptive_transport.mechanism_status_missing");

    // a fixed design is not an adaptive source
    let fixed = AdaptiveDesignContract::new(declaration(cell(
        AssignmentFamily::FixedRandomization,
        MonitoringStrategy::FinalAnalysisOnly,
        StoppingRuleKind::NoStopping,
    )))
    .unwrap();
    let err =
        transport_adaptive_trial(&AdaptiveTransportRequest { contract: Some(&fixed), ..base })
            .unwrap_err();
    assert_eq!(err.code, "route_not_supported");
    assert_eq!(err.detail, "adaptive_transport.design_not_adaptive");

    // a procedure cell with no adaptively weighted route
    let thompson = AdaptiveDesignContract::new(declaration(cell(
        AssignmentFamily::ThompsonBetaBernoulli,
        MonitoringStrategy::FixedInterimLooks,
        StoppingRuleKind::Efficacy,
    )))
    .unwrap();
    let err =
        transport_adaptive_trial(&AdaptiveTransportRequest { contract: Some(&thompson), ..base })
            .unwrap_err();
    assert_eq!(err.code, "cell_not_licensed");

    // exposure that differs from assignment
    let mut noncompliant = declaration(epsilon_greedy());
    noncompliant.exposure.as_mut().unwrap()[0] = 1;
    let noncompliant = AdaptiveDesignContract::new(noncompliant).unwrap();
    let err = transport_adaptive_trial(&AdaptiveTransportRequest {
        contract: Some(&noncompliant),
        ..base
    })
    .unwrap_err();
    assert_eq!(err.code, "adaptive_assumption_violated");

    // an assigned arm below the declared floor
    let mut thin = declaration(epsilon_greedy());
    thin.probability_floor = 0.3;
    let thin = AdaptiveDesignContract::new(thin).unwrap();
    let err = transport_adaptive_trial(&AdaptiveTransportRequest { contract: Some(&thin), ..base })
        .unwrap_err();
    assert_eq!(err.code, "adaptive_assumption_violated");
    assert_eq!(err.detail, "adaptive_transport.assignment_below_floor");
}

#[test]
fn a_stratum_the_adaptive_assignment_starved_is_a_support_failure() {
    // Stratum = assigned arm: each stratum was only ever assigned one arm.
    let contract = AdaptiveDesignContract::new(declaration(epsilon_greedy())).unwrap();
    let unit_strata = vec![0, 1, 0, 1, 1, 0, 1, 0];
    let labels: Vec<String> = vec!["got_control".into(), "got_treatment".into()];
    let diagram = trial_diagram();
    let law = [0.5, 0.5];
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
    let err = transport_adaptive_trial(&request).unwrap_err();
    assert_eq!(err.code, "transport_support_failure");
    assert_eq!(err.detail, "adaptive_transport.stratum_arm_unobserved");
    assert!(err.message.contains("got_control"));
}

#[test]
fn the_trial_population_is_bound_to_the_diagram_and_distinct_from_the_target() {
    let contract = AdaptiveDesignContract::new(declaration(epsilon_greedy())).unwrap();
    let (unit_strata, labels) = strata();
    let diagram = trial_diagram();
    let law = [0.25, 0.75];
    let base = AdaptiveTransportRequest {
        contract: Some(&contract),
        trial_population: "trial_sites",
        unit_strata: &unit_strata,
        stratum_labels: &labels,
        target_stratum_law: Some(&law),
        diagram: Some(&diagram),
        treatment_arm: 1,
        control_arm: 0,
    };
    assert!(transport_adaptive_trial(&base).is_ok());

    // a diagram for some other source population than the trial's
    let err = transport_adaptive_trial(&AdaptiveTransportRequest {
        trial_population: "other_sites",
        ..base
    })
    .unwrap_err();
    assert_eq!(err.code, "transport_missing_evidence");
    assert_eq!(err.detail, "adaptive_transport.diagram_binding_mismatch");

    // target equal to source
    let mut same = trial_diagram();
    same.target_population = "trial_sites".into();
    let err = transport_adaptive_trial(&AdaptiveTransportRequest { diagram: Some(&same), ..base })
        .unwrap_err();
    assert_eq!(err.detail, "adaptive_transport.target_is_source");

    // blank label
    let mut blank = trial_diagram();
    blank.target_population = "  ".into();
    let err = transport_adaptive_trial(&AdaptiveTransportRequest { diagram: Some(&blank), ..base })
        .unwrap_err();
    assert_eq!(err.detail, "adaptive_transport.population_label");
}

#[test]
fn an_invariant_stratum_law_must_equal_the_trials_empirical_law() {
    // unit_strata alternate 0/1 over 8 units: the empirical law is (0.5, 0.5).
    let contract = AdaptiveDesignContract::new(declaration(epsilon_greedy())).unwrap();
    let (unit_strata, labels) = strata();
    let mut invariant = trial_diagram();
    invariant.mechanisms[0].1 = MechanismStatus::Invariant;
    let same = [0.5, 0.5];
    let request = AdaptiveTransportRequest {
        contract: Some(&contract),
        trial_population: "trial_sites",
        unit_strata: &unit_strata,
        stratum_labels: &labels,
        target_stratum_law: Some(&same),
        diagram: Some(&invariant),
        treatment_arm: 1,
        control_arm: 0,
    };
    assert!(transport_adaptive_trial(&request).is_ok());

    let different = [0.25, 0.75];
    let err = transport_adaptive_trial(&AdaptiveTransportRequest {
        target_stratum_law: Some(&different),
        ..request
    })
    .unwrap_err();
    assert_eq!(err.detail, "adaptive_transport.stratum_law_declared_invariant");
    assert!(err.remedy.contains("shifted"));
}
