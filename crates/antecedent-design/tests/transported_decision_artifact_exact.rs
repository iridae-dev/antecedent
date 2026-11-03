//! A transported decision whose carried value is computed by the transport-aware pooling
//! formula, exported and replayed.
//!
//! Hand derivation. A native cohort reports 0.6 with sampling variance 0.2 (weight 5). A claim
//! carried from another population reports 1.4 with sampling variance 0.1 and transport
//! variance 0.3: its total variance is 0.1 + 0.3 = 0.4 (weight 2.5). Generalized least squares
//! over independent errors gives (5 * 0.6 + 2.5 * 1.4) / 7.5 = 6.5 / 7.5 = 13/15 and variance
//! 1 / 7.5. With `treat = 4 x - 3.5` the target value is 52/15 - 3.5 = -1/30 and `wait` is 0,
//! so `wait` is uniquely optimal. Ignoring the transport variance would weight the claim by 10
//! and give (3 + 14) / 15 = 17/15 and `treat` = 4 * 17/15 - 3.5 = 31/30 > 0, the opposite choice; the
//! native mean alone gives 4 * 0.6 - 3.5 = -1.1.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ScientificQuantity;
use antecedent_contract::evidence_ledger::{
    EvidenceLedger, Lineage, ObservationId, SourceDeclaration, SourceKind, TransportIdentity,
};
use antecedent_core::{SemanticDigest, SupportStatus};
use antecedent_design::composition_boundary::{
    SupportPolicy, SupportedVerdict, UnsupportedActionPolicy,
};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::Verdict;
use antecedent_design::decision_transported_source::{
    TransportedDemand, TransportedResponse, evaluate_transported_decision,
};
use antecedent_design::transported_decision_artifact::{
    consume_transported_decision, export_transported_decision,
};
use antecedent_design::transported_synthesis::{
    TransportedContribution, TransportedSynthesisMethod, synthesize_transported,
};

mod common;

fn rows(start: u32, end: u32) -> Vec<ObservationId> {
    (start..end).map(|i| ObservationId::from_key("registry", &i.to_le_bytes())).collect()
}

fn coordinate() -> ScientificQuantity {
    ScientificQuantity {
        population_id: "registry".into(),
        functional_id: "mean".into(),
        regime_id: "contrast:do(t=1)-do(t=0)".into(),
        ..common::quantity("effect")
    }
}

fn contract(coordinate: &ScientificQuantity) -> DecisionContract {
    let action = |id: &str, utility: UtilityExpr| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![coordinate.clone()],
        utility,
    };
    DecisionContract {
        actions: vec![
            action(
                "treat",
                UtilityExpr::difference(
                    UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(4.0)),
                    UtilityExpr::Const(3.5),
                ),
            ),
            action("wait", UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(0.0))),
        ],
        utility_units: "utility".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "registry".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

fn policy() -> SupportPolicy {
    SupportPolicy {
        unsupported: UnsupportedActionPolicy::RequireAllActions,
        weakest_support: SupportStatus::Extrapolative,
    }
}

fn pooled_response(transport_variance: f64) -> TransportedResponse {
    let mut ledger = EvidenceLedger::new();
    let cohort = ledger
        .add_source(
            SourceDeclaration::new("registry cohort", SourceKind::ObservationSet)
                .with_observations(rows(0, 100)),
        )
        .unwrap();
    let carried = ledger
        .add_source(
            SourceDeclaration::new("trial claim", SourceKind::ExternalClaim)
                .with_observations(rows(1_000, 1_100))
                .with_lineage(Lineage::Original)
                .with_transport(TransportIdentity {
                    source_populations: vec!["trial_a".to_owned()],
                    target_population: "registry".to_owned(),
                    selection_diagram: SemanticDigest::from_bytes([1; 32]),
                    assumptions: vec!["outcome mechanism invariant".to_owned()],
                }),
        )
        .unwrap();
    let synthesis = synthesize_transported(
        &mut ledger,
        "pooled effect",
        &[
            TransportedContribution {
                source: cohort,
                estimate: 0.6,
                sampling_variance: 0.2,
                transport_variance: None,
            },
            TransportedContribution {
                source: carried,
                estimate: 1.4,
                sampling_variance: 0.1,
                transport_variance: Some(transport_variance),
            },
        ],
        &TransportedSynthesisMethod::Pool { sampling_covariance: None },
    )
    .unwrap();
    TransportedResponse::from_transported_synthesis(&synthesis, coordinate(), "synthesis").unwrap()
}

#[test]
fn the_exported_decision_replays_the_hand_pooled_target_value() {
    let response = pooled_response(0.3);
    let contract = contract(&coordinate());
    let bytes = export_transported_decision(&contract, &response, &policy(), "pooled").unwrap();
    let consumed = consume_transported_decision(&bytes, None).unwrap();
    let outcomes = &consumed.decision.decision.outcomes;
    let (treat, wait) = (&outcomes[0], &outcomes[1]);
    assert_eq!((treat.id.as_str(), wait.id.as_str()), ("treat", "wait"));
    assert!((treat.expected_utility - (-1.0 / 30.0)).abs() < 1e-12, "{}", treat.expected_utility);
    assert!(wait.expected_utility.abs() < 1e-12);
    assert_eq!(
        consumed.decision.decision.verdict,
        SupportedVerdict::Compared(Verdict::UniquelyOptimal("wait".into()))
    );
    // The retained identity re-admits the bytes.
    let identity = consumed.decision.identity.clone();
    consume_transported_decision(&bytes, Some(&identity)).unwrap();
}

#[test]
fn the_choice_depends_on_counting_the_transport_variance() {
    // Without the 0.3 transport variance the claim weighs 10 and the pooled mean is 17/15,
    // whose treat value 4 * 17/15 - 3.5 = 31/30 is positive: treat is chosen instead.
    let contract = contract(&coordinate());
    let decide = |transport_variance: f64| {
        evaluate_transported_decision(
            &contract,
            &pooled_response(transport_variance),
            TransportedDemand::ExpectedUtility,
            &policy(),
        )
        .unwrap()
        .decision
        .outcomes[0]
            .expected_utility
    };
    assert!((decide(0.3) - (-1.0 / 30.0)).abs() < 1e-12);
    // Transport variance 1e-300 stands in for none (zero is refused): weights 5 and 10.
    assert!((decide(1e-300) - 31.0 / 30.0).abs() < 1e-9);
}
