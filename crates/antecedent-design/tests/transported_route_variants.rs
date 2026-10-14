//! Transported response routes for a temporal period shift, an adaptive-trial
//! source and a surrogate bridge: each has its own receipt route and typed
//! constructor, a receipt identity that differs by route, and a sensitivity
//! route with an exact assumption range.
//!
//! Oracles (hand values, shared with the producing routes' own tests):
//! the period shift of the inspection policy under time-invariant dynamics and a
//! target initial law of `(0.4, 0.6)` is `7.5 - 10 * 0.6 = 1.5`; the adaptive
//! trial standardised to the stratum law `(0.25, 0.75)` has treatment mean
//! `29/56`, control mean `1/2`; a surrogate effect of `0.8` maps to
//! `0.06 + 0.92 * 0.8 = 0.796`.

#![allow(
    clippy::float_cmp,
    clippy::too_many_lines,
    reason = "the oracles compare dyadic constants and each test is one scenario"
)]

use std::sync::Arc;

use antecedent_contract::adaptive_design::{
    AdaptiveDesignContract, AdaptiveDesignDeclaration, AssignmentFamily, DelayMechanism,
    InterimAnalysis, MonitoringStrategy, OutcomeAvailability, OutcomeStructure, PolicyEpoch,
    ProcedureCell, StoppingRuleDeclaration, StoppingRuleKind, StoppingSpec,
};
use antecedent_contract::bridge::{BridgeScope, TrialEffectPair, TrialLevelValidation};
use antecedent_contract::{
    ClaimDomains, ClaimEnvelope, ClaimKind, DomainStatus, QuantityRole, ReasoningView,
    ScientificQuantity, SlotAvailability,
};
use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, ErrorStructure, MeasurementBinding, MeasurementKind,
    MeasurementModel, MeasurementParameters, MeasurementRoute, MeasurementVariable,
    ParameterEvidence, RepresentationVersion,
};
use antecedent_core::{ContractIdentities, ExecutionContext, SemanticDigest, SupportStatus};
use antecedent_design::adaptive_trial_transport::{
    AdaptiveTransportRequest, TrialMechanism, TrialSelectionDiagram, transport_adaptive_trial,
};
use antecedent_design::decision_transported_source::{
    AdaptiveTransportedValue, TransportDeclaration, TransportedRefusal, TransportedResponse,
    TransportedRoute, TransportedValue, period_population_label,
};
use antecedent_design::sequential_evaluation::{
    ActionRole, ActionSpec, ActionSupport, CensoringRule, ContingentPolicy, EvaluationLimits,
    FiniteSequentialModel, HistoryKey, HistoryStep, ObservationChannel,
};
use antecedent_design::temporal_policy_transport::{
    HistoryAlphabet, TemporalPolicyRequest, TemporalTarget, TimeInvariance,
    evaluate_temporal_policy,
};
use antecedent_design::transport_sensitivity::{
    TransportAssumptionRange, TransportPremise, TransportRoute, TransportSensitivitySpec,
    ViolationFamily, transport_assumption_range,
};
use antecedent_design::transported_bridge::{
    BridgeSelectionDiagram, SURROGATE_TRANSPORT_PREMISE_ID, TransportedSurrogateBridge,
};
use antecedent_design::transported_policy::{MechanismStatus, ModelMechanism};

const INSPECT: usize = 0;
const SHIP: usize = 1;
const HOLD: usize = 2;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(5)
}

fn digest(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
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

fn quantity(variable: &str, units: &str, population: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: units.into(),
        population_id: population.into(),
        regime_id: "do(treatment=1)".into(),
        horizon: 0,
        functional_id: "average_effect".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn claim(value: f64) -> ClaimEnvelope {
    ClaimEnvelope::new(
        digest(1),
        ContractIdentities::new(
            digest(10),
            digest(11),
            Some(digest(12)),
            Some(digest(13)),
            digest(14),
            digest(15),
            digest(16),
        ),
        ClaimKind::Point,
        Some(value),
        Some(Arc::from("mmol_per_l")),
        ReasoningView::new(
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
        ),
        ClaimDomains::new(
            DomainStatus::Identified,
            DomainStatus::Supported,
            DomainStatus::Evaluated,
        ),
        None,
        [Arc::<str>::from("prior:p0")],
    )
}

fn validation() -> TrialLevelValidation {
    TrialLevelValidation {
        trials: [(0.0, 0.1), (0.5, 0.5), (1.0, 0.9), (1.5, 1.5)]
            .into_iter()
            .map(|(surrogate_effect, terminal_effect)| TrialEffectPair {
                surrogate_effect,
                terminal_effect,
            })
            .collect(),
        min_trials: 4,
        min_r_squared: 0.9,
    }
}

fn bridge(claim_population: &str, selection_targets: &[&str]) -> TransportedSurrogateBridge {
    TransportedSurrogateBridge {
        surrogate: quantity("trial:biomarker", "mmol_per_l", claim_population),
        terminal: quantity("trial:survival", "log_hazard", claim_population),
        validation: validation(),
        validated_in: BridgeScope {
            population_id: "adults".into(),
            regime_id: "do(treatment=1)".into(),
        },
        validation_evidence: "meta-analysis:ckd-2024".into(),
        diagram: BridgeSelectionDiagram {
            source_population: "adults".into(),
            target_population: claim_population.into(),
            selection_targets: selection_targets.iter().map(|s| (*s).to_owned()).collect(),
        },
        terminal_variable: "survival".into(),
        relation_modifiers: vec!["age".into()],
    }
}

// ---------------------------------------------------------------------------
// Responses built through each typed constructor
// ---------------------------------------------------------------------------

fn temporal_response(
    coordinate: ScientificQuantity,
) -> Result<TransportedResponse, TransportedRefusal> {
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
    TransportedResponse::from_temporal_policy(
        &evaluation,
        "plant_a",
        &invariance,
        coordinate,
        "provider",
    )
}

fn adaptive_response(
    value: AdaptiveTransportedValue,
    coordinate: ScientificQuantity,
) -> Result<TransportedResponse, TransportedRefusal> {
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
    TransportedResponse::from_adaptive_trial(
        &estimate,
        value,
        "trial_sites",
        &diagram,
        coordinate,
        "provider",
    )
}

fn bridge_response(
    claim_population: &str,
    selection_targets: &[&str],
) -> Result<TransportedResponse, TransportedRefusal> {
    let bridge = bridge(claim_population, selection_targets);
    let bridged = bridge.derive(&claim(0.8)).unwrap();
    TransportedResponse::from_surrogate_bridge(&bridge, &bridged, "provider")
}

fn means(response: &TransportedResponse) -> Vec<f64> {
    match response.value() {
        TransportedValue::Means(source) => source.means.clone(),
        TransportedValue::Law(_) => panic!("a typed route builds a point"),
    }
}

fn temporal() -> TransportedResponse {
    temporal_response(quantity("effect", "utility", &period_population_label("plant_a", "2027")))
        .unwrap()
}

fn adaptive() -> TransportedResponse {
    adaptive_response(
        AdaptiveTransportedValue::TreatmentMean,
        quantity("effect", "fraction", "clinic_network"),
    )
    .unwrap()
}

fn surrogate() -> TransportedResponse {
    bridge_response("children", &["biomarker"]).unwrap()
}

#[test]
fn a_period_shift_is_its_own_route_with_period_qualified_populations() {
    let response = temporal();
    assert!((means(&response)[0] - 1.5).abs() < 1e-12);
    let receipt = response.receipt();
    assert_eq!(receipt.route, TransportedRoute::TemporalPolicy);
    assert_eq!(receipt.route.as_str(), "temporal_policy");
    assert_eq!(receipt.source_populations, vec!["plant_a@2026".to_owned()]);
    assert_eq!(receipt.target_population, "plant_a@2027");
    let source = model([0.5, 0.5]);
    assert_eq!(receipt.transport_identities, vec![invariance(&source).identity_hex()]);
    assert!(receipt.assumptions.contains(&"fixed_horizon:2".to_owned()));
    assert!(receipt.assumptions.contains(&"utility:invariant".to_owned()));
    assert_eq!(receipt.support, "extrapolative");
    assert_eq!(receipt.validate(), Ok(()));

    // The bare population is neither period: refused, not silently accepted.
    let error = temporal_response(quantity("effect", "utility", "plant_a")).unwrap_err();
    assert!(matches!(error, TransportedRefusal::QuantityNotInTarget { .. }), "{error:?}");
    // The source period is not the target.
    let error = temporal_response(quantity("effect", "utility", "plant_a@2026")).unwrap_err();
    assert!(matches!(error, TransportedRefusal::QuantityNotInTarget { .. }), "{error:?}");
}

#[test]
fn a_period_shift_response_requires_the_invariance_the_evaluation_was_computed_under() {
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
    let mut other = invariance.clone();
    other.mechanisms.retain(|(mechanism, _)| *mechanism != ModelMechanism::Utility);
    other.mechanisms.push((ModelMechanism::Utility, MechanismStatus::Shifted));
    let error = TransportedResponse::from_temporal_policy(
        &evaluation,
        "plant_a",
        &other,
        quantity("effect", "utility", "plant_a@2027"),
        "provider",
    )
    .unwrap_err();
    assert!(matches!(error, TransportedRefusal::InvalidValue(_)), "{error:?}");
    assert_eq!(error.detail(), "transported_decision.invalid_value");
}

#[test]
fn an_adaptive_trial_is_its_own_route_with_the_diagram_digest_as_identity() {
    let treatment = adaptive();
    assert!((means(&treatment)[0] - 29.0 / 56.0).abs() < 1e-12);
    let control = adaptive_response(
        AdaptiveTransportedValue::ControlMean,
        quantity("control_effect", "fraction", "clinic_network"),
    )
    .unwrap();
    assert!((means(&control)[0] - 0.5).abs() < 1e-12);
    let contrast = adaptive_response(
        AdaptiveTransportedValue::Contrast,
        quantity("effect", "fraction", "clinic_network"),
    )
    .unwrap();
    assert!((means(&contrast)[0] - 1.0 / 56.0).abs() < 1e-12);

    let receipt = treatment.receipt();
    assert_eq!(receipt.route, TransportedRoute::AdaptiveTrial);
    assert_eq!(receipt.route.as_str(), "adaptive_trial");
    assert_eq!(receipt.source_populations, vec!["trial_sites".to_owned()]);
    assert_eq!(receipt.target_population, "clinic_network");
    assert_eq!(receipt.transport_identities, vec![trial_diagram().identity_hex()]);
    assert!(receipt.assumptions.contains(&"stratum_law:shifted".to_owned()));
    assert!(receipt.assumptions.contains(&"outcome_given_stratum_and_arm:invariant".to_owned()));
    assert_eq!(receipt.validate(), Ok(()));

    // The merged grid of one estimate stays on the route.
    let merged = TransportedResponse::merge_points(&[treatment, control]).unwrap();
    assert_eq!(merged.receipt().route, TransportedRoute::AdaptiveTrial);

    // A coordinate in the trial's population is not the transported target.
    let error = adaptive_response(
        AdaptiveTransportedValue::Contrast,
        quantity("effect", "fraction", "trial_sites"),
    )
    .unwrap_err();
    assert!(matches!(error, TransportedRefusal::QuantityNotInTarget { .. }), "{error:?}");
}

#[test]
fn an_adaptive_response_requires_its_diagram_and_trial_population() {
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
    let coordinate = quantity("effect", "fraction", "clinic_network");

    let mut other = diagram.clone();
    other.mechanisms[0].1 = MechanismStatus::Invariant;
    let error = TransportedResponse::from_adaptive_trial(
        &estimate,
        AdaptiveTransportedValue::Contrast,
        "trial_sites",
        &other,
        coordinate.clone(),
        "provider",
    )
    .unwrap_err();
    assert!(matches!(error, TransportedRefusal::InvalidValue(_)), "{error:?}");

    let error = TransportedResponse::from_adaptive_trial(
        &estimate,
        AdaptiveTransportedValue::Contrast,
        "another_trial",
        &diagram,
        coordinate,
        "provider",
    )
    .unwrap_err();
    assert!(matches!(error, TransportedRefusal::InvalidValue(_)), "{error:?}");
}

#[test]
fn a_surrogate_bridge_is_its_own_route_with_the_premise_as_an_assumption() {
    let response = surrogate();
    assert!((means(&response)[0] - 0.796).abs() < 1e-12);
    let receipt = response.receipt();
    assert_eq!(receipt.route, TransportedRoute::SurrogateBridge);
    assert_eq!(receipt.route.as_str(), "surrogate_bridge");
    assert_eq!(receipt.source_populations, vec!["adults".to_owned()]);
    assert_eq!(receipt.target_population, "children");
    assert_eq!(
        receipt.transport_identities,
        vec![bridge("children", &["biomarker"]).diagram.identity_hex()]
    );
    assert!(receipt.assumptions.contains(&format!("premise:{SURROGATE_TRANSPORT_PREMISE_ID}")));
    assert!(receipt.assumptions.contains(&"relation_modifier:age".to_owned()));
    assert!(
        receipt
            .assumptions
            .contains(&"surrogate_validation_evidence:meta-analysis:ckd-2024".to_owned())
    );
    assert_eq!(receipt.validate(), Ok(()));
}

#[test]
fn a_native_bridged_claim_and_a_foreign_bridge_are_not_transported_responses() {
    // A claim about the validated population is native: nothing is carried.
    let error = bridge_response("adults", &[]).unwrap_err();
    assert_eq!(error, TransportedRefusal::NotTransported);
    assert_eq!(error.detail(), "transported_decision.not_transported");

    // A bridged claim offered with another bridge's declaration is refused.
    let bridge_a = bridge("children", &["biomarker"]);
    let bridged = bridge_a.derive(&claim(0.8)).unwrap();
    let bridge_b = bridge("children", &["unrelated"]);
    let error =
        TransportedResponse::from_surrogate_bridge(&bridge_b, &bridged, "provider").unwrap_err();
    assert!(matches!(error, TransportedRefusal::InvalidValue(_)), "{error:?}");
}

#[test]
fn the_receipt_identity_differs_by_route() {
    let temporal = temporal();
    let adaptive = adaptive();
    let surrogate = surrogate();
    let identities = [
        temporal.receipt().canonical_identity(),
        adaptive.receipt().canonical_identity(),
        surrogate.receipt().canonical_identity(),
    ];
    assert_ne!(identities[0], identities[1]);
    assert_ne!(identities[0], identities[2]);
    assert_ne!(identities[1], identities[2]);

    // At unchanged populations, identities, assumptions and value, the route
    // alone changes the receipt identity.
    for route in [
        TransportedRoute::AdaptiveTrial,
        TransportedRoute::SurrogateBridge,
        TransportedRoute::Policy,
        TransportedRoute::Declared,
    ] {
        let mut receipt = temporal.receipt().clone();
        receipt.route = route;
        assert_ne!(receipt.canonical_identity(), identities[0], "{route:?}");
    }
}

#[test]
fn a_typed_route_cannot_be_declared_by_hand_and_the_declared_route_still_refuses_what_it_did() {
    let declaration = |route| TransportDeclaration {
        route,
        source_identity: "transport-1".into(),
        source_populations: vec!["trial".into()],
        target_population: "clinic_b".into(),
        transport_identities: vec!["diagram-1".into()],
        assumptions: vec!["selection acts on age only".into()],
        support: SupportStatus::Extrapolative,
        uncalibrated_standard_error: None,
        provider_id: "provider".into(),
        snapshot_id: "snapshot-1".into(),
        causal_contract_id: "contract-1".into(),
    };
    let coordinate = || vec![quantity("effect", "utility", "clinic_b")];

    for route in [
        TransportedRoute::TemporalPolicy,
        TransportedRoute::AdaptiveTrial,
        TransportedRoute::SurrogateBridge,
    ] {
        let error =
            TransportedResponse::declared_point(coordinate(), vec![1.0], &declaration(route))
                .unwrap_err();
        assert_eq!(error, TransportedRefusal::RouteNeedsTypedSource(route));
        assert_eq!(error.detail(), "transported_decision.route_needs_typed_source");
    }

    // The declared route: free text is accepted, an incomplete identity, a
    // target among the sources and an inflated label are refused as before.
    assert!(
        TransportedResponse::declared_point(
            coordinate(),
            vec![1.0],
            &declaration(TransportedRoute::Declared)
        )
        .is_ok()
    );
    let mut blank = declaration(TransportedRoute::Declared);
    blank.assumptions.clear();
    assert_eq!(
        TransportedResponse::declared_point(coordinate(), vec![1.0], &blank).unwrap_err(),
        TransportedRefusal::TransportIdentityIncomplete
    );
    let mut native = declaration(TransportedRoute::Declared);
    native.source_populations = vec!["clinic_b".into()];
    assert_eq!(
        TransportedResponse::declared_point(coordinate(), vec![1.0], &native).unwrap_err(),
        TransportedRefusal::TransportIdentityIncomplete
    );
    let mut inflated = declaration(TransportedRoute::Declared);
    inflated.support = SupportStatus::Supported;
    assert!(matches!(
        TransportedResponse::declared_point(coordinate(), vec![1.0], &inflated).unwrap_err(),
        TransportedRefusal::SupportLabelInflated(_)
    ));
}

// ---------------------------------------------------------------------------
// Sensitivity route and exact assumption range per route
// ---------------------------------------------------------------------------

fn range_of(response: &TransportedResponse, family: ViolationFamily) -> TransportAssumptionRange {
    let receipt = response.receipt();
    let route = TransportRoute::of_response_route(receipt.route).expect("a typed route");
    let TransportedValue::Means(source) = response.value() else { panic!("point") };
    transport_assumption_range(&TransportSensitivitySpec {
        route,
        quantity: source.coordinates[0].clone(),
        point_value: source.means[0],
        family,
        grid_points: 3,
        source_population: receipt.source_populations[0].clone(),
        target_population: receipt.target_population.clone(),
        query_binding: receipt.source_identity.clone(),
        provider_snapshot: source.snapshot_id.clone(),
        causal_contract_id: source.causal_contract_id.clone(),
    })
    .unwrap()
}

#[test]
fn every_typed_response_route_names_its_sensitivity_route() {
    let expected = [
        (TransportedRoute::CohortEffect, Some(TransportRoute::CohortDecision)),
        (TransportedRoute::ShiftedCohort, Some(TransportRoute::CohortDecision)),
        (TransportedRoute::Policy, Some(TransportRoute::PolicyTransport)),
        (TransportedRoute::Synthesis, Some(TransportRoute::SynthesisTransport)),
        (TransportedRoute::Mechanism, Some(TransportRoute::MechanismTransport)),
        (TransportedRoute::TemporalPolicy, Some(TransportRoute::TemporalPolicyTransport)),
        (TransportedRoute::AdaptiveTrial, Some(TransportRoute::AdaptiveTrialSource)),
        (TransportedRoute::SurrogateBridge, Some(TransportRoute::BridgeTransport)),
        (TransportedRoute::Declared, None),
    ];
    for (route, sensitivity) in expected {
        assert_eq!(TransportRoute::of_response_route(route), sensitivity, "{route:?}");
    }
}

#[test]
fn a_period_shift_range_is_the_time_invariance_band_with_exact_endpoints() {
    let range = range_of(&temporal(), ViolationFamily::AdditiveShift { bound: 0.5 });
    assert_eq!(range.route(), TransportRoute::TemporalPolicyTransport);
    assert_eq!(range.premise(), TransportPremise::TemporalInvariance);
    assert_eq!(range.coordinate().id, "temporal_invariance.shift");
    assert_eq!(range.grid(), &[0.0, 0.25, 0.5]);
    // The point value is an enumeration sum (7.5 - 10 * 0.6), exact only to rounding.
    let v = range.point_value();
    assert!((v - 1.5).abs() < 1e-12);
    assert_eq!(range.lower(), &[v, v - 0.25, v - 0.5]);
    assert_eq!(range.upper(), &[v, v + 0.25, v + 0.5]);
    let (low, high) = range.range_at_bound();
    assert!((low - 1.0).abs() < 1e-12 && (high - 2.0).abs() < 1e-12);
}

#[test]
fn an_adaptive_trial_range_perturbs_the_stratum_law_and_selection_premise() {
    let range = range_of(&adaptive(), ViolationFamily::AdditiveShift { bound: 0.25 });
    // The estimate is a weighted sum, equal to 29/56 only to rounding.
    let value = range.point_value();
    assert!((value - 29.0 / 56.0).abs() < 1e-12);
    assert_eq!(range.route(), TransportRoute::AdaptiveTrialSource);
    assert_eq!(range.premise(), TransportPremise::SelectionDiagramInvariance);
    assert_eq!(range.grid(), &[0.0, 0.125, 0.25]);
    assert_eq!(range.lower(), &[value, value - 0.125, value - 0.25]);
    assert_eq!(range.upper(), &[value, value + 0.125, value + 0.25]);
}

#[test]
fn a_surrogate_bridge_range_perturbs_the_invariance_of_the_surrogate_terminal_relation() {
    let range = range_of(&surrogate(), ViolationFamily::ProportionalShift { bound: 0.5 });
    let value = range.point_value();
    assert!((value - 0.796).abs() < 1e-12);
    assert_eq!(range.route(), TransportRoute::BridgeTransport);
    assert_eq!(range.premise(), TransportPremise::BridgeInvariance);
    assert_eq!(range.coordinate().id, "bridge_invariance.shift");
    assert_eq!(range.coordinate().units, "fraction");
    assert_eq!(range.grid(), &[0.0, 0.25, 0.5]);
    assert_eq!(range.lower()[0], value);
    assert!((range.lower()[2] - value * 0.5).abs() < 1e-12);
    assert!((range.upper()[2] - value * 1.5).abs() < 1e-12);
    // The shift at which the range reaches zero is the whole proportional bound
    // at 1.0, beyond the declared 0.5.
    assert_eq!(range.reference_reached_at(0.0), None);
    assert_eq!(range.reference_reached_at(value * 0.75).map(|s| (s * 1e12).round()), Some(2.5e11));
}

#[test]
fn the_range_identity_differs_by_route_at_unchanged_value_and_bound() {
    let family = ViolationFamily::AdditiveShift { bound: 0.5 };
    let receipt = temporal().receipt().clone();
    let TransportedValue::Means(source) = temporal().value().clone() else { panic!() };
    let build = |route| {
        transport_assumption_range(&TransportSensitivitySpec {
            route,
            quantity: source.coordinates[0].clone(),
            point_value: source.means[0],
            family,
            grid_points: 3,
            source_population: receipt.source_populations[0].clone(),
            target_population: receipt.target_population.clone(),
            query_binding: "derivation:1".into(),
            provider_snapshot: "snapshot:1".into(),
            causal_contract_id: "contract:1".into(),
        })
        .unwrap()
        .identity()
    };
    assert_ne!(
        build(TransportRoute::TemporalPolicyTransport),
        build(TransportRoute::AdaptiveTrialSource)
    );
    assert_ne!(build(TransportRoute::AdaptiveTrialSource), build(TransportRoute::BridgeTransport));
}
