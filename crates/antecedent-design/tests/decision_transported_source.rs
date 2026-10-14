//! A transported response as a decision source: hand-calculated expected utilities, the
//! transport identity in the decision claim, and an executed refusal for every request a
//! transported point cannot answer.
//!
//! Oracles.
//!
//! - Declared point. The transported mean of `effect` in `clinic_b` is `2.0`. The utility of
//!   `treat` is `4 x - 1.5`, so its expected utility is `6.5`; `wait` is `0 * x = 0`.
//! - Transported policy. The inspection model of the transported-evaluation tests: the policy
//!   inspects, ships on a quiet alarm and holds on an alarm, worth `7.5 - 10 p` where `p` is
//!   the target probability of a faulty unit (`5.5` at `p = 0.2`, `-0.5` at `p = 0.8`); the
//!   `hold` policy is worth `0` everywhere.
//! - Transported law. Four equiprobable draws of the outcome under `a` are `1, 1, 0, 0` and
//!   under `b` are `1, 1, 1, 0`, so `P(Y >= 1)` is `1/2` and `3/4`.

#![allow(
    clippy::float_cmp,
    clippy::too_many_lines,
    reason = "dyadic oracles; one scenario per test"
)]

use antecedent_contract::ScientificQuantity;
use antecedent_contract::evidence_ledger::{
    EvidenceLedger, Lineage, ObservationId, SourceDeclaration, SourceId, SourceKind,
    TransportIdentity,
};
use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, ErrorStructure, MeasurementBinding, MeasurementKind,
    MeasurementModel, MeasurementParameters, MeasurementRoute, MeasurementVariable,
    ParameterEvidence, RepresentationVersion,
};
use antecedent_core::{ExecutionContext, SemanticDigest, SupportStatus};
use antecedent_design::composition_boundary::{
    ActionStatus, SupportPolicy, SupportedVerdict, UnsupportedActionPolicy,
};
use antecedent_design::decision_adapters::{AdapterError, adapt_transported_claim};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::Verdict;
use antecedent_design::decision_transported_source::{
    DETAIL_CALIBRATION_INFLATED, DETAIL_IDENTITY_INCOMPLETE, DETAIL_INTERVAL_NOT_LICENSED,
    DETAIL_LAW_NOT_SUPPLIED, DETAIL_MIXED_TRANSPORTS, DETAIL_NATIVE_LABEL, DETAIL_NOT_IN_TARGET,
    DETAIL_SUPPORT_BELOW_POLICY, DETAIL_SUPPORT_INFLATED, TransportDeclaration, TransportedDemand,
    TransportedRefusal, TransportedRepresentation, TransportedResponse, TransportedRoute,
    contract_needs_law, evaluate_transported_decision,
};
use antecedent_design::sequential_evaluation::{
    ActionRole, ActionSpec, ActionSupport, CensoringRule, ContingentPolicy, EvaluationLimits,
    FiniteSequentialModel, HistoryKey, HistoryStep, ObservationChannel,
};
use antecedent_design::transported_policy::{
    MechanismStatus, ModelMechanism, SelectionDiagram, TransportedPolicyRequest,
    evaluate_transported_policy,
};
use antecedent_design::transported_synthesis::{
    TransportedContribution, TransportedSynthesisMethod, synthesize_transported,
};
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io_core::quantity_wire::DistributionMeaningWire;

mod common;

fn effect(population: &str) -> ScientificQuantity {
    ScientificQuantity {
        population_id: population.into(),
        functional_id: "mean".into(),
        regime_id: "contrast:do(t=1)-do(t=0)".into(),
        ..common::quantity("effect")
    }
}

fn declaration() -> TransportDeclaration {
    TransportDeclaration {
        route: TransportedRoute::Declared,
        source_identity: "transport-1".into(),
        source_populations: vec!["trial".into()],
        target_population: "clinic_b".into(),
        transport_identities: vec!["diagram-1".into()],
        assumptions: vec!["selection acts on age only".into()],
        support: SupportStatus::Extrapolative,
        uncalibrated_standard_error: Some(0.3),
        provider_id: "transport-provider".into(),
        snapshot_id: "snapshot-1".into(),
        causal_contract_id: "contract-1".into(),
    }
}

fn point(value: f64) -> TransportedResponse {
    TransportedResponse::declared_point(vec![effect("clinic_b")], vec![value], &declaration())
        .unwrap()
}

/// `4 x - 1.5`.
fn affine() -> UtilityExpr {
    UtilityExpr::difference(
        UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(4.0)),
        UtilityExpr::Const(1.5),
    )
}

fn contract_on(coordinate: &ScientificQuantity, treat: UtilityExpr) -> DecisionContract {
    let action = |id: &str, utility: UtilityExpr| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![coordinate.clone()],
        utility,
    };
    DecisionContract {
        actions: vec![
            action("treat", treat),
            action("wait", UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(0.0))),
        ],
        utility_units: "utility".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "clinic_b".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

fn accepting() -> SupportPolicy {
    SupportPolicy {
        unsupported: UnsupportedActionPolicy::RequireAllActions,
        weakest_support: SupportStatus::Extrapolative,
    }
}

#[test]
fn a_transported_point_decides_by_the_utility_of_the_mean_and_carries_its_identity() {
    let response = point(2.0);
    assert_eq!(response.representation(), TransportedRepresentation::PointOnly);
    let contract = contract_on(&response.coordinates()[0], affine());
    let decision = evaluate_transported_decision(
        &contract,
        &response,
        TransportedDemand::ExpectedUtility,
        &accepting(),
    )
    .unwrap();
    // 4 * 2.0 - 1.5 = 6.5 and 0 * 2.0 = 0.
    assert!((decision.decision.outcomes[0].expected_utility - 6.5).abs() < 1e-12);
    assert!(decision.decision.outcomes[1].expected_utility.abs() < 1e-12);
    assert_eq!(
        decision.decision.verdict,
        SupportedVerdict::Compared(Verdict::UniquelyOptimal("treat".into()))
    );
    for disposition in &decision.decision.dispositions {
        assert_eq!(disposition.status, ActionStatus::Evaluated);
    }
    // The transport identity is in the decision claim, not only in the input.
    assert_eq!(decision.receipt.target_population, "clinic_b");
    assert_eq!(decision.receipt.source_populations, vec!["trial".to_owned()]);
    assert_eq!(decision.receipt.transport_identities, vec!["diagram-1".to_owned()]);
    assert_eq!(decision.receipt.support, "extrapolative");
    assert_eq!(decision.receipt.uncalibrated_standard_error, Some(0.3));
    assert_eq!(decision.inference, "point_only");
    assert!(decision.assumptions.iter().any(|a| a == "selection acts on age only"));
    assert!(decision.assumptions.iter().any(|a| a.contains("not native target data")));
    // The source model's standard error is a diagnostic: no utility inherits it.
    for outcome in &decision.decision.outcomes {
        assert_eq!(outcome.standard_error, None);
    }
}

#[test]
fn the_decision_identity_changes_with_the_transport_at_unchanged_values() {
    let base = point(2.0);
    let mut other = declaration();
    other.transport_identities = vec!["diagram-2".into()];
    let changed =
        TransportedResponse::declared_point(vec![effect("clinic_b")], vec![2.0], &other).unwrap();
    let contract = contract_on(&base.coordinates()[0], affine());
    let run = |response: &TransportedResponse| {
        evaluate_transported_decision(
            &contract,
            response,
            TransportedDemand::ExpectedUtility,
            &accepting(),
        )
        .unwrap()
    };
    let (a, b) = (run(&base), run(&changed));
    assert_eq!(a.decision.outcomes, b.decision.outcomes);
    assert_ne!(a.identity, b.identity);
    assert_ne!(a.receipt.canonical_identity(), b.receipt.canonical_identity());
    // Repeating the same request reproduces the identity bit for bit.
    assert_eq!(a.identity, run(&base).identity);
}

#[test]
fn a_point_refuses_every_request_that_needs_a_law_or_an_interval() {
    let response = point(2.0);
    let coordinate = response.coordinates()[0].clone();
    let contract = contract_on(&coordinate, affine());

    // An explicit law demand.
    let error =
        evaluate_transported_decision(&contract, &response, TransportedDemand::Law, &accepting())
            .unwrap_err();
    assert_eq!(error, TransportedRefusal::LawNotSupplied { needed: "law_request" });
    assert_eq!(error.code(), "decision_contract_unsatisfied");
    assert_eq!(error.detail(), DETAIL_LAW_NOT_SUPPLIED);
    assert!(error.to_refusal().remedy.is_some());

    // A nonlinear utility is a law question.
    let squared = UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(0));
    let nonlinear = contract_on(&coordinate, squared);
    assert_eq!(contract_needs_law(&nonlinear), Some("nonlinear_utility"));
    let error = evaluate_transported_decision(
        &nonlinear,
        &response,
        TransportedDemand::ExpectedUtility,
        &accepting(),
    )
    .unwrap_err();
    assert_eq!(error, TransportedRefusal::LawNotSupplied { needed: "nonlinear_utility" });

    // A probability criterion is a law question.
    let mut probability = contract_on(&coordinate, affine());
    probability.criterion = DecisionCriterion::ThresholdProbability { threshold: 1.0 };
    let error = evaluate_transported_decision(
        &probability,
        &response,
        TransportedDemand::ExpectedUtility,
        &accepting(),
    )
    .unwrap_err();
    assert_eq!(error, TransportedRefusal::LawNotSupplied { needed: "criterion" });

    // An interval is refused for a point.
    let error = evaluate_transported_decision(
        &contract,
        &response,
        TransportedDemand::Interval,
        &accepting(),
    )
    .unwrap_err();
    assert_eq!(error, TransportedRefusal::IntervalNotLicensed);
    assert_eq!(error.code(), "cell_not_licensed");
    assert_eq!(error.detail(), DETAIL_INTERVAL_NOT_LICENSED);
}

#[test]
fn the_default_policy_does_not_accept_a_transported_claim() {
    let response = point(2.0);
    let contract = contract_on(&response.coordinates()[0], affine());
    let error = evaluate_transported_decision(
        &contract,
        &response,
        TransportedDemand::ExpectedUtility,
        &SupportPolicy::compare_supported(),
    )
    .unwrap_err();
    assert_eq!(
        error,
        TransportedRefusal::SupportBelowPolicy {
            supplied: SupportStatus::Extrapolative,
            required: SupportStatus::Supported,
        }
    );
    assert_eq!(error.code(), "cell_not_licensed");
    assert_eq!(error.detail(), DETAIL_SUPPORT_BELOW_POLICY);
}

#[test]
fn an_incomplete_or_inflated_transport_is_refused_at_construction() {
    let build = |edit: &dyn Fn(&mut TransportDeclaration)| {
        let mut declared = declaration();
        edit(&mut declared);
        TransportedResponse::declared_point(vec![effect("clinic_b")], vec![2.0], &declared)
    };
    let incomplete: [&dyn Fn(&mut TransportDeclaration); 4] = [
        &|d| d.assumptions.clear(),
        &|d| d.source_populations.clear(),
        &|d| d.source_populations = vec!["clinic_b".into()],
        &|d| d.transport_identities.clear(),
    ];
    for edit in incomplete {
        let error = build(edit).unwrap_err();
        assert_eq!(error, TransportedRefusal::TransportIdentityIncomplete);
        assert_eq!(error.code(), "transport_missing_evidence");
        assert_eq!(error.detail(), DETAIL_IDENTITY_INCOMPLETE);
    }
    // Transport never raises a support label.
    let error = build(&|d| d.support = SupportStatus::Supported).unwrap_err();
    assert_eq!(error, TransportedRefusal::SupportLabelInflated("supported".into()));
    assert_eq!(error.code(), "transport_not_certified");
    assert_eq!(error.detail(), DETAIL_SUPPORT_INFLATED);
    // The coordinate must be stated in the target population.
    let error =
        TransportedResponse::declared_point(vec![effect("trial")], vec![2.0], &declaration())
            .unwrap_err();
    assert!(matches!(error, TransportedRefusal::QuantityNotInTarget { .. }));
    assert_eq!(error.code(), "quantity_semantics_mismatch");
    assert_eq!(error.detail(), DETAIL_NOT_IN_TARGET);
    // A non-finite mean and a repeated coordinate are malformed.
    assert!(matches!(
        TransportedResponse::declared_point(
            vec![effect("clinic_b")],
            vec![f64::NAN],
            &declaration()
        ),
        Err(TransportedRefusal::InvalidValue(_))
    ));
    assert!(matches!(
        TransportedResponse::declared_point(
            vec![effect("clinic_b"), effect("clinic_b")],
            vec![1.0, 2.0],
            &declaration()
        ),
        Err(TransportedRefusal::InvalidValue(_))
    ));
}

fn outcome(arm: &str) -> ScientificQuantity {
    ScientificQuantity {
        population_id: "clinic_b".into(),
        regime_id: format!("do({arm})"),
        functional_id: "outcome".into(),
        ..common::quantity("y")
    }
}

fn transported_law(
    calibration: DistributionCalibration,
    trust: DistributionTrust,
) -> DistributionArtifact {
    let columns = [outcome("a"), outcome("b")];
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &columns,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "transported-law".into(),
            provider_id: "transport-provider".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "snapshot-1".into(),
            causal_contract_id: "contract-1".into(),
        },
    )
    .unwrap();
    // rows: (a, b) = (1,1), (1,1), (0,1), (0,0)
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [4, 2],
            weights: None,
            supported: None,
            calibration,
            trust,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    )
    .unwrap()
}

fn law_contract() -> DecisionContract {
    let action = |id: &str, arm: &str| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![outcome(arm)],
        utility: UtilityExpr::Input(0),
    };
    DecisionContract {
        actions: vec![action("a", "a"), action("b", "b")],
        utility_units: "units".into(),
        criterion: DecisionCriterion::ThresholdProbability { threshold: 1.0 },
        constraints: vec![],
        target_population: "clinic_b".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

#[test]
fn a_transported_law_answers_a_probability_criterion_and_refuses_an_interval() {
    let law = transported_law(DistributionCalibration::PointOnly, DistributionTrust::Unverified);
    let response = TransportedResponse::declared_law(law, &declaration()).unwrap();
    assert_eq!(response.representation(), TransportedRepresentation::Law);
    let decision = evaluate_transported_decision(
        &law_contract(),
        &response,
        TransportedDemand::Law,
        &accepting(),
    )
    .unwrap();
    // P(Y_a >= 1) = 2/4 and P(Y_b >= 1) = 3/4.
    assert!((decision.decision.outcomes[0].value - 0.5).abs() < 1e-12);
    assert!((decision.decision.outcomes[1].value - 0.75).abs() < 1e-12);
    assert_eq!(
        decision.decision.verdict,
        SupportedVerdict::Compared(Verdict::UniquelyOptimal("b".into()))
    );
    assert_eq!(decision.receipt.representation, TransportedRepresentation::Law);
    // Even a law carries no interval: the transport step has no measured coverage.
    let error = evaluate_transported_decision(
        &law_contract(),
        &response,
        TransportedDemand::Interval,
        &accepting(),
    )
    .unwrap_err();
    assert_eq!(error, TransportedRefusal::IntervalNotLicensed);
}

#[test]
fn a_transported_law_is_never_labelled_exact_measured_or_native() {
    for calibration in [DistributionCalibration::Exact, DistributionCalibration::Measured] {
        let law = transported_law(calibration, DistributionTrust::Unverified);
        let error = TransportedResponse::declared_law(law, &declaration()).unwrap_err();
        assert_eq!(error, TransportedRefusal::CalibrationLabelInflated);
        assert_eq!(error.detail(), DETAIL_CALIBRATION_INFLATED);
    }
    let law =
        transported_law(DistributionCalibration::PointOnly, DistributionTrust::NativeLicensed);
    let error = TransportedResponse::declared_law(law, &declaration()).unwrap_err();
    assert_eq!(error, TransportedRefusal::NativeLabelOnTransported);
    assert_eq!(error.detail(), DETAIL_NATIVE_LABEL);
}

#[test]
fn a_structural_claim_takes_a_transported_law_and_refuses_a_transported_mean() {
    let law = transported_law(DistributionCalibration::PointOnly, DistributionTrust::Unverified);
    let response = TransportedResponse::declared_law(law, &declaration()).unwrap();
    let adapted =
        adapt_transported_claim("transported", &response, StructuralPolicy::ReportOnly).unwrap();
    assert_eq!(adapted.atoms.len(), 1);
    assert_eq!(adapted.profile.support[0].0, "transported");
    assert_eq!(adapted.profile.support[0].1.overall, SupportStatus::Extrapolative);
    let error = adapt_transported_claim("transported", &point(2.0), StructuralPolicy::ReportOnly)
        .unwrap_err();
    assert_eq!(
        error,
        AdapterError::Transported(TransportedRefusal::LawNotSupplied {
            needed: "structural_evaluation"
        })
    );
}

// ---------------------------------------------------------------------------
// Transported policy values
// ---------------------------------------------------------------------------

const INSPECT: usize = 0;
const SHIP: usize = 1;
const HOLD: usize = 2;

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

fn inspection_model() -> FiniteSequentialModel {
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

fn inspect_policy() -> ContingentPolicy {
    let after =
        |observation: usize| HistoryKey::from_steps([HistoryStep { action: INSPECT, observation }]);
    ContingentPolicy::new("inspect_then_ship_on_quiet")
        .decide(HistoryKey::root(), INSPECT)
        .decide(after(0), SHIP)
        .decide(after(1), HOLD)
}

fn hold_policy() -> ContingentPolicy {
    ContingentPolicy::new("hold_always").decide(HistoryKey::root(), HOLD)
}

fn policy_value_coordinate(policy: &str) -> ScientificQuantity {
    ScientificQuantity {
        population_id: "plant_b".into(),
        regime_id: format!("policy:{policy}"),
        functional_id: "mean".into(),
        ..common::quantity("policy_value")
    }
}

/// Both policies' transported values at target faulty probability `faulty`, merged into one
/// grid carried by one selection diagram.
fn transported_policies(faulty: f64) -> TransportedResponse {
    let source = inspection_model();
    let mut rows = vec![
        (ModelMechanism::InitialState, MechanismStatus::Shifted),
        (ModelMechanism::Utility, MechanismStatus::Invariant),
    ];
    for (action, spec) in source.actions.iter().enumerate() {
        rows.push((ModelMechanism::Transition { action }, MechanismStatus::Invariant));
        if spec.channel.is_some() {
            rows.push((ModelMechanism::Observation { action }, MechanismStatus::Invariant));
        }
    }
    let diagram = SelectionDiagram {
        source_population: "plant_a".into(),
        target_population: "plant_b".into(),
        mechanisms: rows,
    };
    let law = [1.0 - faulty, faulty];
    let ctx = ExecutionContext::for_tests(5);
    let members: Vec<TransportedResponse> = [inspect_policy(), hold_policy()]
        .iter()
        .map(|policy| {
            let request = TransportedPolicyRequest {
                source: &source,
                policy,
                target_population: "plant_b",
                target_initial: Some(&law),
                diagram: Some(&diagram),
                limits: EvaluationLimits::default(),
            };
            let evaluation = evaluate_transported_policy(&request, &ctx).unwrap();
            TransportedResponse::from_transported_policy(
                &evaluation,
                policy_value_coordinate(&policy_label(policy)),
                "sequential_evaluation",
            )
            .unwrap()
        })
        .collect();
    TransportedResponse::merge_points(&members).unwrap()
}

fn policy_label(policy: &ContingentPolicy) -> String {
    policy.version().to_owned()
}

fn policy_contract() -> DecisionContract {
    let action = |id: &str, policy: &str| DecisionAction {
        id: id.into(),
        kind: ActionKind::Policy,
        inputs: vec![policy_value_coordinate(policy)],
        utility: UtilityExpr::Input(0),
    };
    DecisionContract {
        actions: vec![
            action("inspect", "inspect_then_ship_on_quiet"),
            action("hold", "hold_always"),
        ],
        utility_units: "utility".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "plant_b".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

#[test]
fn a_transported_policy_value_decides_between_policies_and_keeps_the_diagram() {
    for (faulty, best, inspect_value) in [(0.2, "inspect", 5.5), (0.8, "hold", -0.5)] {
        let response = transported_policies(faulty);
        assert_eq!(response.receipt().route, TransportedRoute::Policy);
        assert_eq!(response.receipt().target_population, "plant_b");
        assert_eq!(response.receipt().source_populations, vec!["plant_a".to_owned()]);
        assert_eq!(response.receipt().transport_identities.len(), 1);
        let decision = evaluate_transported_decision(
            &policy_contract(),
            &response,
            TransportedDemand::ExpectedUtility,
            &accepting(),
        )
        .unwrap();
        // V = 7.5 - 10 p for the inspecting policy; 0 for holding.
        assert!((decision.decision.outcomes[0].expected_utility - inspect_value).abs() < 1e-12);
        assert!(decision.decision.outcomes[1].expected_utility.abs() < 1e-12);
        assert_eq!(
            decision.decision.verdict,
            SupportedVerdict::Compared(Verdict::UniquelyOptimal(best.into()))
        );
        assert_eq!(decision.receipt.transport_identities, response.receipt().transport_identities);
    }
}

#[test]
fn values_carried_by_different_transports_do_not_merge() {
    let one = point(2.0);
    let mut other = declaration();
    other.transport_identities = vec!["diagram-2".into()];
    let two = TransportedResponse::declared_point(
        vec![ScientificQuantity { regime_id: "other".into(), ..effect("clinic_b") }],
        vec![1.0],
        &other,
    )
    .unwrap();
    let error = TransportedResponse::merge_points(&[one.clone(), two]).unwrap_err();
    assert_eq!(error, TransportedRefusal::MixedTransports);
    assert_eq!(error.detail(), DETAIL_MIXED_TRANSPORTS);
    // The same coordinate twice is not a grid.
    assert!(matches!(
        TransportedResponse::merge_points(&[one.clone(), one]),
        Err(TransportedRefusal::InvalidValue(_))
    ));
}

// ---------------------------------------------------------------------------
// Transported synthesis
// ---------------------------------------------------------------------------

fn registry_rows(start: u32, end: u32) -> Vec<ObservationId> {
    (start..end).map(|i| ObservationId::from_key("registry", &i.to_le_bytes())).collect()
}

fn native_source(ledger: &mut EvidenceLedger, label: &str, start: u32, end: u32) -> SourceId {
    ledger
        .add_source(
            SourceDeclaration::new(label, SourceKind::ObservationSet)
                .with_observations(registry_rows(start, end)),
        )
        .unwrap()
}

fn selection_identity() -> TransportIdentity {
    TransportIdentity {
        source_populations: vec!["trial_a".to_owned()],
        target_population: "registry".to_owned(),
        selection_diagram: SemanticDigest::from_bytes([1; 32]),
        assumptions: vec!["outcome mechanism invariant".to_owned()],
    }
}

#[test]
fn a_synthesis_with_a_transported_claim_is_a_transported_decision_source() {
    // Native `y = 1, v = 1` with a claim `y = 2, v = 1` carried under transport variance
    // 0.5: pooled mean 1.4 and variance 0.6.
    let mut ledger = EvidenceLedger::new();
    let cohort = native_source(&mut ledger, "registry cohort", 0, 100);
    let carried = ledger
        .add_source(
            SourceDeclaration::new("trial a claim", SourceKind::ExternalClaim)
                .with_observations(registry_rows(1_000, 1_100))
                .with_lineage(Lineage::Original)
                .with_transport(selection_identity()),
        )
        .unwrap();
    let synthesis = synthesize_transported(
        &mut ledger,
        "pooled effect",
        &[
            TransportedContribution {
                source: cohort,
                estimate: 1.0,
                sampling_variance: 1.0,
                transport_variance: None,
            },
            TransportedContribution {
                source: carried,
                estimate: 2.0,
                sampling_variance: 1.0,
                transport_variance: Some(0.5),
            },
        ],
        &TransportedSynthesisMethod::Pool { sampling_covariance: None },
    )
    .unwrap();
    let coordinate = ScientificQuantity {
        population_id: "registry".into(),
        functional_id: "mean".into(),
        ..common::quantity("effect")
    };
    let response = TransportedResponse::from_transported_synthesis(
        &synthesis,
        coordinate.clone(),
        "synthesis",
    )
    .unwrap();
    assert_eq!(response.receipt().route, TransportedRoute::Synthesis);
    assert_eq!(
        response.receipt().transport_identities,
        vec![selection_identity().digest().to_hex()]
    );
    assert_eq!(response.receipt().source_populations, vec!["trial_a".to_owned()]);
    assert!(
        (response.receipt().uncalibrated_standard_error.unwrap() - 0.6_f64.sqrt()).abs() < 1e-12
    );
    let decision = evaluate_transported_decision(
        &{
            let mut contract = contract_on(&coordinate, affine());
            contract.target_population = "registry".into();
            contract
        },
        &response,
        TransportedDemand::ExpectedUtility,
        &SupportPolicy {
            unsupported: UnsupportedActionPolicy::RequireAllActions,
            weakest_support: SupportStatus::Extrapolative,
        },
    )
    .unwrap();
    // 4 * 1.4 - 1.5 = 4.1; the pooled variance is not a utility standard error.
    assert!((decision.decision.outcomes[0].expected_utility - 4.1).abs() < 1e-12);
    assert_eq!(decision.decision.outcomes[0].standard_error, None);

    // A synthesis of native sources only was never carried anywhere.
    let mut native_only = EvidenceLedger::new();
    let first = native_source(&mut native_only, "first", 0, 100);
    let second = native_source(&mut native_only, "second", 100, 200);
    let pooled = synthesize_transported(
        &mut native_only,
        "native pool",
        &[
            TransportedContribution {
                source: first,
                estimate: 1.0,
                sampling_variance: 1.0,
                transport_variance: None,
            },
            TransportedContribution {
                source: second,
                estimate: 2.0,
                sampling_variance: 1.0,
                transport_variance: None,
            },
        ],
        &TransportedSynthesisMethod::Pool { sampling_covariance: None },
    )
    .unwrap();
    let error = TransportedResponse::from_transported_synthesis(
        &pooled,
        coordinate,
        "synthesis",
    )
    .unwrap_err();
    assert_eq!(error, TransportedRefusal::NotTransported);
}

#[test]
fn a_route_with_a_typed_constructor_cannot_be_declared_by_hand() {
    for route in
        [TransportedRoute::Policy, TransportedRoute::ShiftedCohort, TransportedRoute::Synthesis]
    {
        let mut declared = declaration();
        declared.route = route;
        let error = TransportedResponse::declared_point(
            vec![effect("clinic_b")],
            vec![2.0],
            &declared,
        )
        .unwrap_err();
        assert_eq!(error, TransportedRefusal::RouteNeedsTypedSource(route));
        assert_eq!(error.detail(), "transported_decision.route_needs_typed_source");
    }
}

#[test]
fn a_typed_route_identity_must_be_a_digest_while_a_declared_route_may_be_text() {
    // Free text is a declaration and stays allowed on the declared route.
    assert!(point(1.0).receipt().transport_identities == vec!["diagram-1".to_owned()]);
    // A mechanism or cohort-effect route names a producing result's digest.
    for route in [TransportedRoute::Mechanism, TransportedRoute::CohortEffect] {
        let mut declared = declaration();
        declared.route = route;
        let error =
            TransportedResponse::declared_point(vec![effect("clinic_b")], vec![2.0], &declared)
                .unwrap_err();
        assert_eq!(error, TransportedRefusal::IdentityNotDigest);
        assert_eq!(error.detail(), "transported_decision.transport_identity_not_digest");
        declared.transport_identities = vec![format!("{:064x}", 0xabc)];
        assert!(
            TransportedResponse::declared_point(vec![effect("clinic_b")], vec![2.0], &declared)
                .is_ok()
        );
    }
}

#[test]
fn a_support_floor_weakens_a_laws_coordinates_and_never_strengthens_them() {
    use antecedent_design::composition_boundary::{
        DecisionInput, TrustEvidence, TrustRequirement,
    };
    let law = transported_law(DistributionCalibration::PointOnly, DistributionTrust::Unverified);
    let input = DecisionInput::from_distribution_artifact(
        "law",
        &law,
        &TrustEvidence::None,
        TrustRequirement::Unrestricted,
    )
    .unwrap();
    // The artifact's own mask says supported.
    assert!(input.support().entries().iter().all(|(_, s)| *s == SupportStatus::Supported));
    let floored = input.clone().with_support_floor(SupportStatus::Extrapolative);
    assert!(floored.support().entries().iter().all(|(_, s)| *s == SupportStatus::Extrapolative));
    // A weaker status is kept, and a floor that is stronger changes nothing.
    let again = floored.with_support_floor(SupportStatus::Supported);
    assert!(again.support().entries().iter().all(|(_, s)| *s == SupportStatus::Extrapolative));
}

#[test]
fn a_synthesis_receipt_takes_its_populations_from_the_ledger_not_the_caller() {
    let mut ledger = EvidenceLedger::new();
    let cohort = native_source(&mut ledger, "registry cohort", 0, 100);
    let carried = ledger
        .add_source(
            SourceDeclaration::new("trial a claim", SourceKind::ExternalClaim)
                .with_observations(registry_rows(1_000, 1_100))
                .with_lineage(Lineage::Original)
                .with_transport(selection_identity()),
        )
        .unwrap();
    let contribution = |source, estimate, transport_variance| TransportedContribution {
        source,
        estimate,
        sampling_variance: 1.0,
        transport_variance,
    };
    let synthesis = synthesize_transported(
        &mut ledger,
        "pooled effect",
        &[contribution(cohort, 1.0, None), contribution(carried, 2.0, Some(0.5))],
        &TransportedSynthesisMethod::Pool { sampling_covariance: None },
    )
    .unwrap();
    assert_eq!(synthesis.target_population.as_deref(), Some("registry"));
    assert_eq!(synthesis.source_populations, vec!["trial_a".to_owned()]);

    // A coordinate stated in some other population than the ledger's target
    // is refused: the receipt cannot be pointed at a different target.
    let elsewhere = ScientificQuantity {
        population_id: "some_other_clinic".into(),
        functional_id: "mean".into(),
        ..common::quantity("effect")
    };
    let error =
        TransportedResponse::from_transported_synthesis(&synthesis, elsewhere, "synthesis")
            .unwrap_err();
    assert!(matches!(error, TransportedRefusal::QuantityNotInTarget { .. }));
}

#[test]
fn an_exported_transported_decision_replays_and_refuses_resealed_edits() {
    use antecedent_design::transported_decision_artifact::{
        consume_transported_decision, export_transported_decision,
    };
    let response = point(2.0);
    let contract = contract_on(&response.coordinates()[0], affine());
    let bytes =
        export_transported_decision(&contract, &response, &accepting(), "transported-1").unwrap();
    let consumed = consume_transported_decision(&bytes, None).unwrap();
    // 4 * 2.0 - 1.5 = 6.5, recomputed by the consumer.
    assert!((consumed.decision.decision.outcomes[0].expected_utility - 6.5).abs() < 1e-12);
    assert_eq!(consumed.decision.receipt.target_population, "clinic_b");
    let retained = consumed.decision.identity.clone();
    assert!(consume_transported_decision(&bytes, Some(&retained)).is_ok());
    assert!(consume_transported_decision(&bytes, Some("not-the-identity")).is_err());

    // The same decision at another carried value has another identity, so a
    // consumer that retained the first refuses the second's bytes.
    let other = point(3.0);
    let other_bytes =
        export_transported_decision(&contract, &other, &accepting(), "transported-2").unwrap();
    assert!(consume_transported_decision(&other_bytes, Some(&retained)).is_err());

    // A law has its own export.
    let law = transported_law(DistributionCalibration::PointOnly, DistributionTrust::Unverified);
    let law_response = TransportedResponse::declared_law(law, &declaration()).unwrap();
    let error =
        export_transported_decision(&law_contract(), &law_response, &accepting(), "law").unwrap_err();
    assert!(error.to_string().contains("law_not_exportable"), "{error}");

    // A policy that does not admit the label cannot even export.
    let strict = SupportPolicy {
        unsupported: UnsupportedActionPolicy::RequireAllActions,
        weakest_support: SupportStatus::Supported,
    };
    assert!(export_transported_decision(&contract, &response, &strict, "strict").is_err());
}
