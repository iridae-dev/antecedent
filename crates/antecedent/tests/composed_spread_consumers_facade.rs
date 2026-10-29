//! What the consumers of other artifact kinds do with a transported spread claim.
//!
//! A spread of a composed quantity is a labelled `uncalibrated_spread` value of
//! a `joint_law_propagation_v1` artifact. It is not a decision law, an interval
//! or a composition result, so the decision-reliability consumer, the mechanism
//! composition consumer and the inverse-query consumer each refuse its bytes
//! rather than read them. The joint-law consumer replays it to identical bits.

use antecedent::analysis::decision_reliability::{
    DecisionReliabilityExpectation, consume_decision_reliability_artifact,
};
use antecedent::analysis::inverse_query;
use antecedent::analysis::joint_law_propagation::{
    MomentInput, PropagationClaim, PropagationDraws, consume_joint_law_propagation_artifact,
    propagate_transported_spread, propagate_transported_spread_aligned,
};
use antecedent::analysis::mechanism_composition::consume_mechanism_composition_artifact;
use antecedent_contract::{InputRole, QuantityRole, ScientificQuantity};
use antecedent_core::ExecutionContext;
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(5)
}

fn quantity(name: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: name.into(),
        variable_name: name.into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: "do(a=1)".into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn pieces_artifact(snapshot: &str) -> DistributionArtifact {
    let columns: [[f64; 4]; 3] = [[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0], [0.0, 0.0, 1.0, 2.0]];
    let quantities: Vec<ScientificQuantity> =
        ["a", "b", "c"].iter().map(|name| quantity(name)).collect();
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &quantities,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: format!("study-{snapshot}"),
            provider_id: "engine".into(),
            rng_id: format!("splitmix-{snapshot}"),
            snapshot_id: snapshot.into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .expect("identity");
    let mut draws = Vec::new();
    for row in 0..4 {
        for column in &columns {
            draws.push(column[row]);
        }
    }
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [4, 3],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Unmeasured,
            trust: DistributionTrust::NativeLicensed,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .expect("artifact")
}

fn covariance_claim() -> PropagationClaim {
    let piece = |id: &str, evidence: &str, role: InputRole| MomentInput {
        id: id.into(),
        evidence_id: evidence.into(),
        role,
        mean: 0.0,
    };
    propagate_transported_spread(
        &[
            piece("source", "snap-source", InputRole::SourceSampling),
            piece("target", "snap-target", InputRole::TargetSampling),
            piece("functional", "snap-source", InputRole::TransportFunctional),
        ],
        &[1.0, 1.0, 1.0],
        &[1.0, 0.5, 0.0, 0.5, 4.0, 0.0, 0.0, 0.0, 9.0],
        "piece-covariance",
        &ctx(),
    )
    .expect("transported spread")
}

fn aligned_claim() -> PropagationClaim {
    let joint = pieces_artifact("snap-pieces");
    let column = |name: &str, index: usize| {
        PropagationDraws::from_artifact_column(name, &joint, index).expect("column")
    };
    propagate_transported_spread_aligned(
        &column("source", 0),
        &column("target", 1),
        &column("functional", 2),
        [1.0, 1.0, 1.0],
        &ctx(),
    )
    .expect("aligned transported spread")
}

fn offered_to_other_consumers(claim: &PropagationClaim) {
    let bytes = claim.export().expect("export");
    // Each consumer refuses a recognised artifact of another kind with a typed
    // `artifact_kind_mismatch` detail, both kinds, and the consumer that accepts it.
    let refusal = consume_decision_reliability_artifact(
        &bytes,
        &DecisionReliabilityExpectation::default(),
        &ctx(),
    )
    .expect_err("a spread is not a decision-reliability artifact");
    assert_eq!(refusal.code, "invalid_argument");
    assert_eq!(refusal.detail, "decision_reliability.artifact_kind_mismatch");
    assert!(refusal.message.contains("expected decision_reliability_v1"));
    assert!(refusal.message.contains("supplied joint_law_propagation_v1"));
    assert!(refusal.remedy.is_some_and(|r| r.contains("joint-law propagation consumer")));

    let refusal = consume_mechanism_composition_artifact(&bytes, None, &ctx())
        .expect_err("a spread is not a mechanism composition");
    assert_eq!(refusal.code(), "invalid_argument");
    assert_eq!(refusal.details(), vec!["mechanism_composition.artifact_kind_mismatch"]);
    let first = &refusal.refusals[0];
    assert_eq!(first.supplied.as_deref(), Some("joint_law_propagation_v1"));
    assert_eq!(first.expected.as_deref(), Some("mechanism_composition_v1"));
    assert!(first.remedy.is_some_and(|r| r.contains("joint-law propagation consumer")));

    let error = inverse_query::consume(&bytes, None)
        .expect_err("a spread is not an inverse-query artifact");
    assert_eq!(error.reason_code(), Some("invalid_argument"));
    let refusal = antecedent_design::inverse_query_artifact::artifact_refusal(&error);
    assert_eq!(refusal.code, "invalid_argument");
    assert_eq!(refusal.detail, "functional_inverse_query.artifact_kind_mismatch");
    assert_eq!(refusal.expected.as_deref(), Some("inverse_functional_query_v1"));
    assert_eq!(refusal.supplied.as_deref(), Some("joint_law_propagation_v1"));
    assert!(refusal.remedy.is_some_and(|r| r.contains("joint-law propagation consumer")));
}

#[test]
fn a_transported_spread_with_a_covariance_is_not_read_as_a_decision_or_a_composition() {
    let claim = covariance_claim();
    assert_eq!(claim.label(), "uncalibrated_spread");
    offered_to_other_consumers(&claim);
}

#[test]
fn an_aligned_transported_spread_is_not_read_as_a_decision_or_a_composition() {
    let claim = aligned_claim();
    assert_eq!(claim.label(), "uncalibrated_spread");
    offered_to_other_consumers(&claim);
}

#[test]
fn the_joint_law_consumer_replays_both_transported_spreads_to_identical_bits() {
    // Hand values. Covariance form: weights (1, 1, 1) on the stated covariance, whose entries
    // sum to 1 + 4 + 9 + 2 * 0.5 = 15. Aligned form: the draws of a + b + c are 3, 3, 8, 9,
    // mean 5.75, squared deviations 7.5625, 7.5625, 5.0625, 10.5625, population variance
    // 30.75 / 4 = 7.6875.
    for (claim, variance) in [(covariance_claim(), 15.0), (aligned_claim(), 7.6875)] {
        assert!((claim.value("variance").unwrap() - variance).abs() < 1e-12, "{:?}", claim.value("variance"));
        let bytes = claim.export().expect("export");
        let consumed =
            consume_joint_law_propagation_artifact(&bytes, Some(claim.identity()), &ctx())
                .expect("consume");
        assert_eq!(consumed.label, "uncalibrated_spread");
        assert!(!consumed.is_envelope());
        assert_eq!(
            consumed.value("variance").map(f64::to_bits),
            claim.value("variance").map(f64::to_bits)
        );
        assert!((consumed.value("variance").unwrap() - variance).abs() < 1e-12);
    }
}

#[test]
fn malformed_bytes_keep_their_generic_refusals() {
    let junk = [0_u8, 1, 2, 3];
    let refusal = consume_decision_reliability_artifact(
        &junk,
        &DecisionReliabilityExpectation::default(),
        &ctx(),
    )
    .expect_err("junk");
    assert_eq!(refusal.detail, "decision_reliability.artifact");
    let refusal = consume_mechanism_composition_artifact(&junk, None, &ctx()).expect_err("junk");
    assert_eq!(refusal.details(), vec!["mechanism_composition.artifact_invalid"]);
    let error = inverse_query::consume(&junk, None).expect_err("junk");
    assert_ne!(
        antecedent_design::inverse_query_artifact::artifact_refusal(&error).detail,
        "functional_inverse_query.artifact_kind_mismatch"
    );
}
