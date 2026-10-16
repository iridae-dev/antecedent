//! Verifying a stored temporal policy value and a stored adaptive-trial transport estimate:
//! the honest stored object replays; a forged value, a forged identity and a request whose
//! target law or horizon changed are refused with the value-mismatch detail (or the
//! evaluation's own refusal).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines, reason = "each test is one scenario over shared fixtures")]

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
use antecedent_design::transported_policy::{MechanismStatus, ModelMechanism};

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

// ---------------------------------------------------------------------------
// Transported policy evaluation
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
fn a_stored_temporal_policy_value_replays_and_forgeries_are_refused() {
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
    let stored = evaluate_temporal_policy(&request, &ctx()).unwrap();
    assert!((stored.target_value - 1.5).abs() < 1e-12);
    verify_temporal_policy(&stored, &request, &ctx()).unwrap();

    let mismatch = "temporal_policy_transport.value_mismatch";
    let mut value = stored.clone();
    value.target_value += 0.25;
    let err = verify_temporal_policy(&value, &request, &ctx()).unwrap_err();
    assert_eq!((err.code, err.detail.as_str()), ("invalid_argument", mismatch));

    let mut source_value = stored.clone();
    source_value.source_value -= 1.0;
    assert_eq!(
        verify_temporal_policy(&source_value, &request, &ctx()).unwrap_err().detail,
        mismatch
    );

    let mut identity = stored.clone();
    identity.identity.push('0');
    assert_eq!(verify_temporal_policy(&identity, &request, &ctx()).unwrap_err().detail, mismatch);

    // The stored value does not replay under another target law.
    let other_law = [0.45, 0.55];
    let other = TemporalPolicyRequest { target_initial: Some(&other_law), ..request };
    assert_eq!(verify_temporal_policy(&stored, &other, &ctx()).unwrap_err().detail, mismatch);

    // A horizon the declaration no longer shares is refused by the evaluation itself.
    let mut longer = invariance.clone();
    longer.horizon += 1;
    let changed = TemporalPolicyRequest { invariance: Some(&longer), ..request };
    assert_eq!(
        verify_temporal_policy(&stored, &changed, &ctx()).unwrap_err().detail,
        "temporal_policy_transport.horizon_mismatch"
    );
}

#[test]
fn a_stored_adaptive_transport_estimate_replays_and_forgeries_are_refused() {
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
    let stored = transport_adaptive_trial(&request).unwrap();
    assert!((stored.contrast - 1.0 / 56.0).abs() < 1e-12);
    verify_adaptive_transport(&stored, &request).unwrap();

    let mismatch = "adaptive_transport.value_mismatch";
    for which in 0..4 {
        let mut forged = stored.clone();
        match which {
            0 => forged.treatment_mean += 0.01,
            1 => forged.control_mean += 0.01,
            2 => forged.contrast += 0.01,
            _ => forged.identity.push('0'),
        }
        let err = verify_adaptive_transport(&forged, &request).unwrap_err();
        assert_eq!((err.code, err.detail.as_str()), ("invalid_argument", mismatch));
    }

    // The stored estimate does not replay under another target stratum law.
    let other_law = [0.5, 0.5];
    let other = AdaptiveTransportRequest { target_stratum_law: Some(&other_law), ..request };
    assert_eq!(verify_adaptive_transport(&stored, &other).unwrap_err().detail, mismatch);

    // A law that is not a distribution over the strata is refused by the evaluation.
    let bad_law = [0.5, 0.6];
    let bad = AdaptiveTransportRequest { target_stratum_law: Some(&bad_law), ..request };
    assert_eq!(
        verify_adaptive_transport(&stored, &bad).unwrap_err().detail,
        "adaptive_transport.target_stratum_law"
    );
}
