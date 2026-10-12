//! Dirichlet posterior means of the finite laws the decision routes already read.
//! The published number is the functional of that mean. A tiny concentration
//! stays at the hand pin. A mapped prior refuses.

use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, DecisionFunctional,
    StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::evaluate_functional;
use antecedent_design::inverse_query::{
    Comparison, ForwardClaim, ForwardEvidence, GridScope, InverseConstraint, InverseQuery,
    SelectionRule, evaluate_inverse_query,
};
use antecedent_estimate::posterior_mean::{
    DirichletNamespace, PosteriorRequest, dirichlet_posterior_mean,
};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

fn quantity(regime: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "y".into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn law(weights: &[f64]) -> DistributionArtifact {
    let columns = [quantity("do(a=A)"), quantity("do(a=B)")];
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &columns,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "enumerated".into(),
            provider_id: "exact-law".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "enumeration-1".into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .unwrap();
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [2, 2],
            weights: Some(weights.to_vec()),
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        vec![0.0, 1.0, 2.0, 1.0],
    )
    .unwrap()
}

fn contract() -> DecisionContract {
    DecisionContract {
        actions: vec![
            DecisionAction {
                id: "Y".into(),
                kind: ActionKind::Intervention,
                inputs: vec![quantity("do(a=A)")],
                utility: UtilityExpr::Input(0),
            },
            DecisionAction {
                id: "ref".into(),
                kind: ActionKind::Intervention,
                inputs: vec![quantity("do(a=B)")],
                utility: UtilityExpr::Input(0),
            },
        ],
        utility_units: "units".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

fn expectation(weights: &[f64]) -> f64 {
    evaluate_functional(&contract(), "Y", DecisionFunctional::Expectation, &law(weights))
        .unwrap()
        .value
}

fn refusal(error: antecedent_estimate::EstimationError) -> String {
    let antecedent_estimate::EstimationError::Refused { code, message } = error else {
        panic!("expected a coded refusal, got {error:?}");
    };
    format!("{code} {message}")
}

#[test]
fn decision_posterior_mean_keeps_the_finite_law_expectation() {
    let mean = dirichlet_posterior_mean(
        &[1.0, 3.0],
        PosteriorRequest::IndependentNormal { scale: 1e-8 },
        DirichletNamespace::Decision,
    )
    .unwrap();
    let value = expectation(&mean);
    assert!((value - 1.5).abs() < 1e-6, "{value}");
    let pulled = dirichlet_posterior_mean(
        &[1.0, 3.0],
        PosteriorRequest::IndependentNormal { scale: 100.0 },
        DirichletNamespace::Decision,
    )
    .unwrap();
    let pulled_value = expectation(&pulled);
    assert!((pulled_value - 1.0).abs() < (value - 1.0).abs());
}

#[test]
fn decision_posterior_refuses_a_mapped_prior() {
    let message = refusal(
        dirichlet_posterior_mean(
            &[1.0, 3.0],
            PosteriorRequest::MappedCoefficient,
            DirichletNamespace::Decision,
        )
        .unwrap_err(),
    );
    assert!(message.contains("route_not_supported"), "{message}");
    assert!(message.contains("decision_posterior.mapped_prior"), "{message}");
}

#[test]
fn decision_posterior_consumer_replays_the_dirichlet_weights() {
    let produced = dirichlet_posterior_mean(
        &[1.0, 3.0],
        PosteriorRequest::IndependentNormal { scale: 1.0 },
        DirichletNamespace::Decision,
    )
    .unwrap();
    let total = 1.0 + 3.0 + 2.0;
    let consumed = [(1.0 + 1.0) / total, (3.0 + 1.0) / total];
    for (left, right) in produced.iter().zip(consumed) {
        assert!((left - right).abs() < 1e-12);
    }
    assert!((expectation(&produced) - (0.0 * consumed[0] + 2.0 * consumed[1])).abs() < 1e-9);
}

fn inverse_on(weights: &[f64]) -> Vec<String> {
    let evidence = ForwardEvidence {
        point: Some(ForwardClaim::Law(Box::new(law(weights)))),
        ..ForwardEvidence::default()
    };
    let query = InverseQuery {
        contract: contract(),
        grid_order: vec!["Y".into(), "ref".into()],
        grid_scope: GridScope::FiniteEnumeration,
        constraints: vec![InverseConstraint::TargetMean {
            target: 1.0,
            comparison: Comparison::AtLeast,
        }],
        selection: SelectionRule::FirstInGridOrder,
        tolerance: 1e-12,
        max_evaluations: None,
    };
    evaluate_inverse_query(&query, &evidence)
        .unwrap()
        .actions
        .into_iter()
        .filter(|action| action.point.is_some())
        .map(|action| action.id)
        .collect()
}

#[test]
fn functional_inverse_posterior_uses_the_dirichlet_mean() {
    let mean = dirichlet_posterior_mean(
        &[1.0, 3.0],
        PosteriorRequest::IndependentNormal { scale: 1e-8 },
        DirichletNamespace::FunctionalInverse,
    )
    .unwrap();
    assert_eq!(inverse_on(&mean), inverse_on(&[1.0, 3.0]));
    assert!((expectation(&mean) - 1.5).abs() < 1e-6);
}

#[test]
fn functional_inverse_posterior_refuses_an_external_power_prior() {
    let message = refusal(
        dirichlet_posterior_mean(
            &[1.0, 3.0],
            PosteriorRequest::ExternalPower,
            DirichletNamespace::FunctionalInverse,
        )
        .unwrap_err(),
    );
    assert!(message.contains("functional_inverse_posterior.external_power_prior"), "{message}");
}

#[test]
fn functional_inverse_posterior_consumer_replays_the_mean_law() {
    let produced = dirichlet_posterior_mean(
        &[1.0, 3.0],
        PosteriorRequest::IndependentNormal { scale: 0.5 },
        DirichletNamespace::FunctionalInverse,
    )
    .unwrap();
    let total = 4.0 + 1.0;
    assert!((produced[0] - 1.5 / total).abs() < 1e-12);
    assert_eq!(inverse_on(&produced), inverse_on(&[(1.0 + 0.5) / total, (3.0 + 0.5) / total]));
}

#[test]
fn finite_action_inverse_posterior_uses_the_dirichlet_mean() {
    let mean = dirichlet_posterior_mean(
        &[1.0, 3.0],
        PosteriorRequest::IndependentNormal { scale: 1e-8 },
        DirichletNamespace::FiniteActionInverse,
    )
    .unwrap();
    assert_eq!(inverse_on(&mean), inverse_on(&[1.0, 3.0]));
    assert!((expectation(&mean) - 1.5).abs() < 1e-6);
}

#[test]
fn finite_action_inverse_posterior_refuses_an_unmeasured_band() {
    let message = refusal(
        dirichlet_posterior_mean(
            &[1.0, 3.0],
            PosteriorRequest::UnmeasuredBand,
            DirichletNamespace::FiniteActionInverse,
        )
        .unwrap_err(),
    );
    assert!(message.contains("finite_action_posterior.unmeasured_band"), "{message}");
}

#[test]
fn finite_action_inverse_posterior_consumer_replays_the_selected_action() {
    let produced = dirichlet_posterior_mean(
        &[1.0, 3.0],
        PosteriorRequest::IndependentNormal { scale: 1e-8 },
        DirichletNamespace::FiniteActionInverse,
    )
    .unwrap();
    let hand = {
        let total = 4.0 + 2.0 * 1e-8;
        vec![(1.0 + 1e-8) / total, (3.0 + 1e-8) / total]
    };
    assert_eq!(inverse_on(&produced), inverse_on(&hand));
}
