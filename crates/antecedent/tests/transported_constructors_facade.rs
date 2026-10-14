//! Typed constructors of a transported response from the library transport routes:
//! each route runs, and the response's populations, identities and assumptions are
//! copied from the producing result.
//!
//! Oracles (hand-computed).
//!
//! * Inspection policy. States `sound`/`faulty`, `inspect` raises an alarm with
//!   sensitivity and specificity three quarters, the policy ships on no alarm and
//!   holds on an alarm, `ship` pays +10 (sound) or -10 (faulty), horizon two. With
//!   `p = P(faulty)` the value is `7.5 - 10 p`: `-0.5` at `p = 0.8`, `1.5` at
//!   `p = 0.6`, `2.5` at the source's `p = 0.5`.
//! * Adaptive trial. Eight units, strata `unit % 2`; target stratum law
//!   `(0.25, 0.75)` gives treatment mean `0.25 * 4/7 + 0.75 * 1/2 = 29/56`, control
//!   mean `0.25 * 1 + 0.75 * 1/3 = 1/2`, contrast `1/56`.
//! * Synthesis. Native `y = 1, v = 1` and transported `y = 2, v = 1` with transport
//!   variance `0.5` pool to `1.4` with variance `0.6`.
//! * Surrogate bridge. Trial-level regression `0.06 + 0.92 x`: a surrogate effect of
//!   `0.8` maps to `0.796`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::too_many_lines,
    reason = "closed-form oracles; one scenario per test"
)]

use std::sync::Arc;

use antecedent::Study;
use antecedent::analysis::sequential_evaluation::{
    ActionRole, ActionSpec, ActionSupport, CensoringRule, ContingentPolicy, EvaluationLimits,
    FiniteSequentialModel, HistoryKey, HistoryStep, ObservationChannel,
};
use antecedent::analysis::transported_constructors::{
    AdaptiveTransportRequest, AdaptiveTransportedValue, BridgeSelectionDiagram, HistoryAlphabet,
    MechanismStatus, ModelMechanism, SelectionDiagram, TemporalPolicyRequest, TemporalTarget,
    TimeInvariance, TransportedContribution, TransportedPolicyRequest, TransportedSurrogateBridge,
    TransportedSynthesisMethod, TrialMechanism, TrialSelectionDiagram, adaptive_trial_response,
    period_population_label, surrogate_bridge_response, temporal_policy_response,
    transported_policy_response, transported_synthesis_response,
};
use antecedent::analysis::{
    AdmissibilityRules, AdmissibleDecisionContract, BoundDecisionContract, SupportPolicy,
    TransportedDemand, TransportedResponse, TransportedRoute,
};
use antecedent_contract::adaptive_design::{
    AdaptiveDesignContract, AdaptiveDesignDeclaration, AssignmentFamily, DelayMechanism,
    InterimAnalysis, MonitoringStrategy, OutcomeAvailability, OutcomeStructure, PolicyEpoch,
    ProcedureCell, StoppingRuleDeclaration, StoppingRuleKind, StoppingSpec,
};
use antecedent_contract::bridge::{BridgeScope, TrialEffectPair, TrialLevelValidation};
use antecedent_contract::evidence_ledger::{
    EvidenceLedger, Lineage, ObservationId, SourceDeclaration, SourceId, SourceKind,
    TransportIdentity,
};
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
use antecedent_design::composition_boundary::{SupportedVerdict, UnsupportedActionPolicy};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::Verdict;
use antecedent_design::decision_transported_source::{
    DETAIL_NOT_IN_TARGET, DETAIL_NOT_TRANSPORTED,
};

#[path = "fixtures_common/mod.rs"]
mod common;

use common::fixtures::confounded_scm;

const INSPECT: usize = 0;
const SHIP: usize = 1;
const HOLD: usize = 2;
const PROVIDER: &str = "transport-constructors";

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(5)
}

fn quantity(variable: &str, population: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: population.into(),
        regime_id: "do(treatment=1)".into(),
        horizon: 0,
        functional_id: "average_effect".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn is_digest(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn point_of(response: &TransportedResponse) -> Vec<f64> {
    match response.value() {
        antecedent::analysis::inverse_sources::TransportedValue::Means(mean) => mean.means.clone(),
        antecedent::analysis::inverse_sources::TransportedValue::Law(_) => {
            panic!("a transported point")
        }
    }
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

fn model() -> FiniteSequentialModel {
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
fn a_policy_value_is_carried_with_the_populations_and_digest_of_its_evaluation() {
    let source = model();
    let policy = policy();
    let diagram = diagram(&source);
    let law = [0.2, 0.8];
    let request = TransportedPolicyRequest {
        source: &source,
        policy: &policy,
        target_population: "plant_b",
        target_initial: Some(&law),
        diagram: Some(&diagram),
        limits: EvaluationLimits::default(),
    };
    let built =
        transported_policy_response(&request, quantity("value", "plant_b"), PROVIDER, &ctx())
            .unwrap();
    assert!((point_of(&built.response)[0] - (-0.5)).abs() < 1e-12);
    assert!((built.evaluation.source_value - 2.5).abs() < 1e-12);

    let receipt = built.response.receipt();
    assert_eq!(receipt.route, TransportedRoute::Policy);
    assert_eq!(receipt.source_populations, vec!["plant_a".to_owned()]);
    assert_eq!(receipt.target_population, "plant_b");
    assert_eq!(receipt.transport_identities, vec![diagram.identity_hex()]);
    assert!(is_digest(&receipt.transport_identities[0]));
    assert_eq!(receipt.support, SupportStatus::Extrapolative.as_str());
    assert_eq!(receipt.source_identity, built.evaluation.identity);
    assert!(receipt.assumptions.iter().any(|a| a == "shifted:initial_state"));

    // The same request at a changed target law is another identity.
    let other_law = [0.3, 0.7];
    let other = transported_policy_response(
        &TransportedPolicyRequest { target_initial: Some(&other_law), ..request },
        quantity("value", "plant_b"),
        PROVIDER,
        &ctx(),
    )
    .unwrap();
    assert!((point_of(&other.response)[0] - 0.5).abs() < 1e-12);
    assert_ne!(other.response.receipt().canonical_identity(), receipt.canonical_identity());
    assert_ne!(other.response.receipt().source_identity, receipt.source_identity);
}

#[test]
fn a_constructed_policy_response_feeds_the_transported_decision() {
    let source = model();
    let policy = policy();
    let diagram = diagram(&source);
    let law = [0.2, 0.8];
    let request = TransportedPolicyRequest {
        source: &source,
        policy: &policy,
        target_population: "plant_b",
        target_initial: Some(&law),
        diagram: Some(&diagram),
        limits: EvaluationLimits::default(),
    };
    let value = quantity("value", "plant_b");
    let built = transported_policy_response(&request, value.clone(), PROVIDER, &ctx()).unwrap();

    // `adopt` earns the transported value (-0.5); `keep` earns 0.
    let action = |id: &str, utility: UtilityExpr| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![value.clone()],
        utility,
    };
    let contract = DecisionContract {
        actions: vec![
            action(
                "adopt",
                UtilityExpr::difference(UtilityExpr::Input(0), UtilityExpr::Const(0.0)),
            ),
            action("keep", UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(0.0))),
        ],
        utility_units: "utility".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "plant_b".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    };
    let (data, dag, query) = confounded_scm(64, 5);
    let causal = Study::tabular(data).query(query).graph(dag).inspect().unwrap();
    let bound = BoundDecisionContract::bind(
        &causal,
        AdmissibleDecisionContract { contract, rules: AdmissibilityRules::default() },
    )
    .unwrap();
    let accepting = SupportPolicy {
        unsupported: UnsupportedActionPolicy::RequireAllActions,
        weakest_support: SupportStatus::Extrapolative,
    };
    let decided = bound
        .evaluate_transported(&built.response, TransportedDemand::ExpectedUtility, &accepting)
        .unwrap();
    let outcomes = &decided.decision().decision.outcomes;
    assert!((outcomes[0].expected_utility - (-0.5)).abs() < 1e-12);
    assert!(outcomes[1].expected_utility.abs() < 1e-12);
    assert_eq!(
        decided.decision().decision.verdict,
        SupportedVerdict::Compared(Verdict::UniquelyOptimal("keep".into()))
    );
    let inspection = decided.inspect();
    assert!(!inspection.native_target_data);
    assert_eq!(inspection.source_populations, vec!["plant_a".to_owned()]);
    assert_eq!(inspection.target_population, "plant_b");
}

#[test]
fn a_policy_route_refusal_is_the_routes_own_and_a_wrong_coordinate_is_refused() {
    let source = model();
    let policy = policy();
    let diagram = diagram(&source);
    let law = [0.2, 0.8];
    let base = TransportedPolicyRequest {
        source: &source,
        policy: &policy,
        target_population: "plant_b",
        target_initial: Some(&law),
        diagram: Some(&diagram),
        limits: EvaluationLimits::default(),
    };
    let refused = transported_policy_response(
        &TransportedPolicyRequest { diagram: None, ..base },
        quantity("value", "plant_b"),
        PROVIDER,
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(refused.code, "transport_missing_evidence");
    assert_eq!(refused.detail, "transported_policy.selection_diagram_missing");
    assert!(refused.remedy.is_some());

    // The value is a plant_b value; stating it for the source plant is refused.
    let refused =
        transported_policy_response(&base, quantity("value", "plant_a"), PROVIDER, &ctx())
            .unwrap_err();
    assert_eq!(refused.detail, DETAIL_NOT_IN_TARGET);
    assert_eq!(refused.expected.as_deref(), Some("plant_b"));
    assert_eq!(refused.supplied.as_deref(), Some("plant_a"));
}

// ---------------------------------------------------------------------------
// Period shift
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
fn a_period_shift_is_stated_in_period_qualified_populations_with_the_invariance_digest() {
    let source = model();
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
    let in_2027 = period_population_label("plant_a", "2027");
    assert_eq!(in_2027, "plant_a@2027");
    let built =
        temporal_policy_response(&request, quantity("value", &in_2027), PROVIDER, &ctx()).unwrap();
    assert!((point_of(&built.response)[0] - 1.5).abs() < 1e-12);
    assert!((built.evaluation.source_value - 2.5).abs() < 1e-12);

    let receipt = built.response.receipt();
    assert_eq!(receipt.source_populations, vec!["plant_a@2026".to_owned()]);
    assert_eq!(receipt.target_population, "plant_a@2027");
    assert_eq!(receipt.transport_identities, vec![invariance.identity_hex()]);
    assert!(is_digest(&receipt.transport_identities[0]));
    assert_eq!(receipt.support, SupportStatus::Extrapolative.as_str());
    assert!(receipt.assumptions.iter().any(|a| a == "fixed_horizon:2"));
    assert_eq!(receipt.route, TransportedRoute::TemporalPolicy);
    assert!(receipt.assumptions.iter().any(|a| a == "initial_state:shifted"));
    assert!(receipt.assumptions.iter().any(|a| a == "transition[0]:invariant"));

    // A value stated for the unqualified population is not in the target period.
    let refused =
        temporal_policy_response(&request, quantity("value", "plant_a"), PROVIDER, &ctx())
            .unwrap_err();
    assert_eq!(refused.detail, DETAIL_NOT_IN_TARGET);

    // Same shape, changed target law: another response identity.
    let changed_law = [0.45, 0.55];
    let changed = temporal_policy_response(
        &TemporalPolicyRequest { target_initial: Some(&changed_law), ..request },
        quantity("value", &in_2027),
        PROVIDER,
        &ctx(),
    )
    .unwrap();
    assert_ne!(
        changed.response.receipt().source_identity,
        built.response.receipt().source_identity
    );
}

#[test]
fn a_time_varying_mechanism_is_refused_by_the_period_route() {
    let source = model();
    let policy = policy();
    let mut varying = invariance(&source);
    for row in &mut varying.mechanisms {
        if row.0 == (ModelMechanism::Transition { action: INSPECT }) {
            row.1 = MechanismStatus::Shifted;
        }
    }
    let target = target(&source);
    let law = [0.4, 0.6];
    let request = TemporalPolicyRequest {
        source: &source,
        policy: &policy,
        target: &target,
        target_initial: Some(&law),
        invariance: Some(&varying),
        limits: EvaluationLimits::default(),
    };
    let refused =
        temporal_policy_response(&request, quantity("value", "plant_a@2027"), PROVIDER, &ctx())
            .unwrap_err();
    assert_eq!(refused.code, "transport_missing_evidence");
    assert_eq!(refused.detail, "temporal_policy_transport.mechanism_time_varying");
    assert!(refused.remedy.is_some());
}

// ---------------------------------------------------------------------------
// Adaptive trial
// ---------------------------------------------------------------------------

fn digest(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

fn declaration() -> AdaptiveDesignDeclaration {
    let cell = ProcedureCell {
        assignment: AssignmentFamily::EpsilonGreedy,
        outcome: OutcomeStructure::SingleOutcome,
        monitoring: MonitoringStrategy::FixedInterimLooks,
        stopping: StoppingRuleKind::Efficacy,
    };
    let assigned_arm = vec![0, 1, 0, 1, 1, 0, 1, 0];
    let probabilities = (0..8)
        .map(|unit| match unit {
            4 | 5 => vec![0.2, 0.8],
            6 | 7 => vec![0.4, 0.6],
            _ => vec![0.5, 0.5],
        })
        .collect();
    let available_at_look = (0..8).map(|unit| Some(u32::from(unit >= 4))).collect();
    let outcomes = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0].iter().map(|&y| vec![y]).collect();
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

#[test]
fn an_adaptive_trial_mean_and_contrast_carry_the_trial_and_diagram_identities() {
    let contract = AdaptiveDesignContract::new(declaration()).unwrap();
    let unit_strata: Vec<usize> = (0..8).map(|unit| unit % 2).collect();
    let labels = vec!["mild".to_owned(), "severe".to_owned()];
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
    let build = |value: AdaptiveTransportedValue, variable: &str| {
        adaptive_trial_response(&request, value, quantity(variable, "clinic_network"), PROVIDER)
            .unwrap()
    };
    let treatment = build(AdaptiveTransportedValue::TreatmentMean, "treatment_mean");
    let control = build(AdaptiveTransportedValue::ControlMean, "control_mean");
    let contrast = build(AdaptiveTransportedValue::Contrast, "contrast");
    assert!((point_of(&treatment.response)[0] - 29.0 / 56.0).abs() < 1e-12);
    assert!((point_of(&control.response)[0] - 0.5).abs() < 1e-12);
    assert!((point_of(&contrast.response)[0] - 1.0 / 56.0).abs() < 1e-12);

    let receipt = treatment.response.receipt();
    assert_eq!(receipt.source_populations, vec!["trial_sites".to_owned()]);
    assert_eq!(receipt.target_population, "clinic_network");
    assert_eq!(receipt.transport_identities, vec![diagram.identity_hex()]);
    assert!(is_digest(&receipt.transport_identities[0]));
    assert_eq!(receipt.source_identity, treatment.evaluation.identity);
    assert!(receipt.assumptions.iter().any(|a| a == "stratum_law:shifted"));
    assert!(receipt.assumptions.iter().any(|a| a == "outcome_given_stratum_and_arm:invariant"));
    assert_eq!(receipt.support, SupportStatus::Extrapolative.as_str());

    // The arms of one estimate were carried by one transport, so they merge into one grid.
    let merged =
        TransportedResponse::merge_points(&[treatment.response.clone(), control.response.clone()])
            .unwrap();
    assert_eq!(point_of(&merged).len(), 2);

    // A changed target law at unchanged shape is another estimate.
    let mild = [1.0, 0.0];
    let other = adaptive_trial_response(
        &AdaptiveTransportRequest { target_stratum_law: Some(&mild), ..request },
        AdaptiveTransportedValue::TreatmentMean,
        quantity("treatment_mean", "clinic_network"),
        PROVIDER,
    )
    .unwrap();
    assert!((point_of(&other.response)[0] - 4.0 / 7.0).abs() < 1e-12);
    assert_ne!(other.response.receipt().source_identity, receipt.source_identity);
}

#[test]
fn adaptive_refusals_are_the_routes_and_a_source_population_coordinate_is_refused() {
    let contract = AdaptiveDesignContract::new(declaration()).unwrap();
    let unit_strata: Vec<usize> = (0..8).map(|unit| unit % 2).collect();
    let labels = vec!["mild".to_owned(), "severe".to_owned()];
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
    let refused = adaptive_trial_response(
        &AdaptiveTransportRequest { contract: None, ..base },
        AdaptiveTransportedValue::Contrast,
        quantity("contrast", "clinic_network"),
        PROVIDER,
    )
    .unwrap_err();
    assert_eq!(refused.code, "adaptive_contract_incomplete");
    assert_eq!(refused.detail, "adaptive_transport.assignment_monitoring_contract_missing");

    let mut shifted = trial_diagram();
    shifted.mechanisms[1].1 = MechanismStatus::Shifted;
    let refused = adaptive_trial_response(
        &AdaptiveTransportRequest { diagram: Some(&shifted), ..base },
        AdaptiveTransportedValue::Contrast,
        quantity("contrast", "clinic_network"),
        PROVIDER,
    )
    .unwrap_err();
    assert_eq!(refused.detail, "adaptive_transport.outcome_mechanism_not_invariant");

    let refused = adaptive_trial_response(
        &base,
        AdaptiveTransportedValue::Contrast,
        quantity("contrast", "trial_sites"),
        PROVIDER,
    )
    .unwrap_err();
    assert_eq!(refused.detail, DETAIL_NOT_IN_TARGET);
}

// ---------------------------------------------------------------------------
// Transported synthesis
// ---------------------------------------------------------------------------

fn rows(start: u32, end: u32) -> Vec<ObservationId> {
    (start..end).map(|i| ObservationId::from_key("registry", &i.to_le_bytes())).collect()
}

fn carried_transport() -> TransportIdentity {
    TransportIdentity {
        source_populations: vec!["trial_a".to_owned()],
        target_population: "registry".to_owned(),
        selection_diagram: digest(1),
        assumptions: vec!["outcome mechanism invariant".to_owned()],
    }
}

fn ledger_with_claim() -> (EvidenceLedger, SourceId, SourceId) {
    let mut ledger = EvidenceLedger::new();
    let cohort = ledger
        .add_source(
            SourceDeclaration::new("registry cohort", SourceKind::ObservationSet)
                .with_observations(rows(0, 100)),
        )
        .unwrap();
    let carried = ledger
        .add_source(
            SourceDeclaration::new("trial a claim", SourceKind::ExternalClaim)
                .with_observations(rows(1_000, 1_100))
                .with_lineage(Lineage::Original)
                .with_transport(carried_transport()),
        )
        .unwrap();
    (ledger, cohort, carried)
}

fn contribution(
    source: SourceId,
    estimate: f64,
    transport: Option<f64>,
) -> TransportedContribution {
    TransportedContribution {
        source,
        estimate,
        sampling_variance: 1.0,
        transport_variance: transport,
    }
}

#[test]
fn a_transported_synthesis_takes_its_populations_from_the_ledger_identities() {
    let (mut ledger, cohort, carried) = ledger_with_claim();
    let method = TransportedSynthesisMethod::Pool { sampling_covariance: None };
    let built = transported_synthesis_response(
        &mut ledger,
        "pooled effect",
        &[contribution(cohort, 1.0, None), contribution(carried, 2.0, Some(0.5))],
        &method,
        Some("registry"),
        quantity("pooled", "registry"),
        PROVIDER,
    )
    .unwrap();
    assert!((point_of(&built.response)[0] - 1.4).abs() < 1e-12);
    assert!((built.evaluation.variance - 0.6).abs() < 1e-12);

    let receipt = built.response.receipt();
    assert_eq!(receipt.route, TransportedRoute::Synthesis);
    assert_eq!(receipt.source_populations, vec!["trial_a".to_owned()]);
    assert_eq!(receipt.target_population, "registry");
    assert_eq!(receipt.transport_identities, vec![carried_transport().digest().to_hex()]);
    assert_eq!(receipt.source_identity, built.evaluation.result.to_hex());
    // The model variance is an uncalibrated diagnostic only.
    assert!((receipt.uncalibrated_standard_error.unwrap() - 0.6_f64.sqrt()).abs() < 1e-12);
    // The ledger holds the registered result.
    assert!(ledger.result(built.evaluation.result).is_some());
}

#[test]
fn a_refused_synthesis_response_leaves_the_ledger_untouched() {
    let (mut ledger, cohort, carried) = ledger_with_claim();
    let before = ledger.state_digest();
    let method = TransportedSynthesisMethod::Pool { sampling_covariance: None };
    let both = [contribution(cohort, 1.0, None), contribution(carried, 2.0, Some(0.5))];

    // The synthesis registers, but the value is stated for another population.
    let refused = transported_synthesis_response(
        &mut ledger,
        "pooled effect",
        &both,
        &method,
        None,
        quantity("pooled", "trial_a"),
        PROVIDER,
    )
    .unwrap_err();
    assert_eq!(refused.detail, DETAIL_NOT_IN_TARGET);
    assert_eq!(ledger.state_digest(), before, "no result registered on a refused response");

    // Native sources only: nothing was transported.
    let second = ledger
        .add_source(
            SourceDeclaration::new("second cohort", SourceKind::ObservationSet)
                .with_observations(rows(5_000, 5_100)),
        )
        .unwrap();
    let before = ledger.state_digest();
    let refused = transported_synthesis_response(
        &mut ledger,
        "native only",
        &[contribution(cohort, 1.0, None), contribution(second, 1.5, None)],
        &method,
        None,
        quantity("pooled", "registry"),
        PROVIDER,
    )
    .unwrap_err();
    assert_eq!(refused.detail, DETAIL_NOT_TRANSPORTED);
    assert_eq!(ledger.state_digest(), before);

    // The route's own refusal: a transported claim without its transport variance.
    let refused = transported_synthesis_response(
        &mut ledger,
        "no transport variance",
        &[contribution(cohort, 1.0, None), contribution(carried, 2.0, None)],
        &method,
        None,
        quantity("pooled", "registry"),
        PROVIDER,
    )
    .unwrap_err();
    assert_eq!(refused.code, "transport_missing_evidence");
    assert_eq!(refused.detail, "transported_synthesis.transport_variance_undeclared");
    assert!(refused.remedy.is_some());
    assert_eq!(ledger.state_digest(), before);

    // A native population that is not the transports' target.
    let refused = transported_synthesis_response(
        &mut ledger,
        "wrong native population",
        &both,
        &method,
        Some("clinic"),
        quantity("pooled", "registry"),
        PROVIDER,
    )
    .unwrap_err();
    assert_eq!(refused.detail, "transported_synthesis.target_population_mismatch");
    assert_eq!(ledger.state_digest(), before);
}

// ---------------------------------------------------------------------------
// Surrogate bridge
// ---------------------------------------------------------------------------

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

fn bridge(claim_population: &str, selection_targets: &[&str]) -> TransportedSurrogateBridge {
    TransportedSurrogateBridge {
        surrogate: ScientificQuantity {
            units: "mmol_per_l".into(),
            ..quantity("trial:biomarker", claim_population)
        },
        terminal: ScientificQuantity {
            units: "log_hazard".into(),
            ..quantity("trial:survival", claim_population)
        },
        validation: TrialLevelValidation {
            trials: [(0.0, 0.1), (0.5, 0.5), (1.0, 0.9), (1.5, 1.5)]
                .into_iter()
                .map(|(surrogate_effect, terminal_effect)| TrialEffectPair {
                    surrogate_effect,
                    terminal_effect,
                })
                .collect(),
            min_trials: 4,
            min_r_squared: 0.9,
        },
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

#[test]
fn a_bridged_terminal_value_is_a_transported_point_in_the_claims_population() {
    let bridge = bridge("children", &["biomarker"]);
    let built = surrogate_bridge_response(&bridge, &claim(0.8), PROVIDER).unwrap();
    assert!((point_of(&built.response)[0] - 0.796).abs() < 1e-12);
    assert!((built.evaluation.value - 0.796).abs() < 1e-12);

    let receipt = built.response.receipt();
    assert_eq!(receipt.source_populations, vec!["adults".to_owned()]);
    assert_eq!(receipt.target_population, "children");
    assert_eq!(receipt.transport_identities, vec![bridge.diagram.identity_hex()]);
    assert!(is_digest(&receipt.transport_identities[0]));
    assert_eq!(receipt.support, SupportStatus::Extrapolative.as_str());
    assert!(receipt.assumptions.iter().any(|a| a == "relation_modifier:age"));
    assert!(
        receipt.assumptions.iter().any(|a| a == "premise:surrogate_terminal_relation_invariant")
    );

    // A different selection diagram at unchanged shape is another transport identity.
    let other =
        surrogate_bridge_response(&self::bridge("children", &["unrelated"]), &claim(0.8), PROVIDER)
            .unwrap();
    assert_ne!(other.response.receipt().transport_identities, receipt.transport_identities);
    assert_ne!(other.response.receipt().canonical_identity(), receipt.canonical_identity());
}

#[test]
fn bridge_refusals_are_the_bridges_and_a_native_bridge_is_not_transported() {
    // A selection arrow into the terminal outcome contradicts the premise.
    let refused =
        surrogate_bridge_response(&bridge("children", &["survival"]), &claim(0.8), PROVIDER)
            .unwrap_err();
    assert_eq!(refused.code, "bridge_assumption_missing");
    assert_eq!(
        refused.detail,
        "scientific_bridge.transport_premise_contradicted_by_selection_diagram"
    );
    assert!(refused.remedy.is_some());

    // A claim in the validated population is a native bridged claim.
    let refused =
        surrogate_bridge_response(&bridge("adults", &[]), &claim(0.8), PROVIDER).unwrap_err();
    assert_eq!(refused.detail, DETAIL_NOT_TRANSPORTED);
}
