//! Transported claims carry the standing their provenance states into the composition graph.
//!
//! A transported response is a point (or a transported law) carried under a declared premise.
//! Its node states support (the receipt's label), provider trust (the route's provenance) and an
//! uncertainty guarantee (a point unless a record established at exactly this target and
//! quantity is declared). A moved standing is a change the verification sees even when the
//! claim's numbers are untouched.
//!
//! Oracles (hand-classified):
//!
//! - a library route has no provider (trust not applicable); a declared route holds the trust
//!   level its provider was verified at, and states none as undeclared, which is not a license;
//! - nothing moved: every claim is `still_valid`;
//! - one claim's trust held then withdrawn: that claim is `withdrawn` for provider trust, its
//!   sibling stays `still_valid`;
//! - one claim's calibration record replaced at the same target: that claim `recompute`s, its
//!   sibling stays `still_valid`;
//! - a record established at the earlier target while the claim now answers another target:
//!   `withdrawn` for calibration, the point value untouched, and a heir of that record is
//!   refused the record it inherited.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent::analysis::composition_verification::{
    CalibrationDeclaration, CompositionGraph, CompositionVerification, NodeKind, Revalidation,
    SupportStanding, TrustStanding, WithdrawalKind,
};
use antecedent::analysis::transport_revalidation::{
    DETAIL_TRUST_NOT_APPLICABLE, SceneCalibration, TransportReason, TransportSceneBuilder,
    provenance_trust, response_coordinate, response_support, revalidate, transported_claim_node,
    transported_claim_standing,
};
use antecedent::analysis::{TransportDeclaration, TransportedResponse, TransportedRoute};
use antecedent_contract::{ProviderTrust, QuantityRole, ScientificQuantity};
use antecedent_core::{SemanticDigest, SupportStatus};

fn digest(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

fn quantity(variable: &str, population: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: population.into(),
        regime_id: "contrast".into(),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn response(variable: &str, source: &str, target: &str, value: f64) -> TransportedResponse {
    let declaration = TransportDeclaration {
        route: TransportedRoute::Declared,
        source_identity: format!("transport:{variable}:{source}"),
        source_populations: vec![source.into()],
        target_population: target.into(),
        transport_identities: vec!["diagram-1".into()],
        assumptions: vec!["selection acts on age only".into()],
        support: SupportStatus::Extrapolative,
        uncalibrated_standard_error: None,
        provider_id: "transport-provider".into(),
        snapshot_id: format!("snapshot:{variable}"),
        causal_contract_id: "contract-1".into(),
    };
    TransportedResponse::declared_point(vec![quantity(variable, target)], vec![value], &declaration)
        .unwrap()
}

const VERIFIED: Option<ProviderTrust> = Some(ProviderTrust::ExactRequestVerified);

#[test]
fn provenance_states_trust_and_never_assumes_it() {
    // A library route has no provider whose trust could hold or be withdrawn.
    assert_eq!(
        provenance_trust(TransportedRoute::Policy, None).unwrap(),
        TrustStanding::NotApplicable
    );
    assert_eq!(
        provenance_trust(TransportedRoute::CohortEffect, None).unwrap(),
        TrustStanding::NotApplicable
    );
    // A declared route holds the level its provider was verified at.
    assert_eq!(
        provenance_trust(TransportedRoute::Declared, VERIFIED).unwrap(),
        TrustStanding::Held
    );
    assert_eq!(
        provenance_trust(TransportedRoute::Declared, Some(ProviderTrust::Unverified)).unwrap(),
        TrustStanding::Withdrawn
    );
    // A declared route that states no level is not trusted by default.
    assert_eq!(
        provenance_trust(TransportedRoute::Declared, None).unwrap(),
        TrustStanding::Undeclared
    );

    // A provider level for a library route is refused: nothing it could describe.
    let refusal = provenance_trust(TransportedRoute::Policy, VERIFIED).unwrap_err();
    assert_eq!(refusal.code, "invalid_argument");
    assert_eq!(refusal.detail, DETAIL_TRUST_NOT_APPLICABLE);
    assert!(refusal.remedy.is_some());
}

#[test]
fn a_response_node_states_its_support_trust_and_calibration() {
    let carried = response("effect", "trial", "clinic_b", 2.0);
    let standing =
        transported_claim_standing(&carried, CalibrationDeclaration::PointOnly, VERIFIED).unwrap();
    assert_eq!(standing.support, SupportStanding::Supported);
    assert_eq!(standing.trust, TrustStanding::Held);
    assert_eq!(standing.calibration, CalibrationDeclaration::PointOnly);
    assert_eq!(response_support(&carried), SupportStanding::Supported);

    let node =
        transported_claim_node("effect", &carried, CalibrationDeclaration::PointOnly, VERIFIED)
            .unwrap();
    assert_eq!(node.kind, NodeKind::TransportedResult);
    assert_eq!(node.standing(), standing);
    assert_eq!(node.coordinate, response_coordinate(&carried).unwrap());
    assert_eq!(node.assumptions, vec!["selection acts on age only".to_owned()]);
    // The identity moves with the value and is stable otherwise.
    let same =
        transported_claim_node("effect", &carried, CalibrationDeclaration::PointOnly, VERIFIED)
            .unwrap();
    assert_eq!(node.identity, same.identity);
    let moved = transported_claim_node(
        "effect",
        &response("effect", "trial", "clinic_b", 2.5),
        CalibrationDeclaration::PointOnly,
        VERIFIED,
    )
    .unwrap();
    assert_ne!(node.identity, moved.identity);
    // The coordinate is the target and the quantity, not the value: a record established for
    // this response at this target holds at the same coordinate for a re-estimated value.
    assert_eq!(node.coordinate, moved.coordinate);
    let other_target = response("effect", "trial", "clinic_c", 2.0);
    assert_ne!(node.coordinate, response_coordinate(&other_target).unwrap());
}

fn graph_of(trust_a: Option<ProviderTrust>, record_a: u8) -> CompositionGraph {
    let a = response("effect_a", "trial", "clinic_b", 2.0);
    let b = response("effect_b", "registry", "clinic_b", 1.5);
    let coordinate_a = response_coordinate(&a).unwrap();
    let mut graph = CompositionGraph::new();
    graph
        .add_node(
            transported_claim_node(
                "effect_a",
                &a,
                CalibrationDeclaration::Held { record: digest(record_a), coordinate: coordinate_a },
                trust_a,
            )
            .unwrap(),
        )
        .unwrap();
    graph
        .add_node(
            transported_claim_node("effect_b", &b, CalibrationDeclaration::PointOnly, VERIFIED)
                .unwrap(),
        )
        .unwrap();
    graph
}

fn class_of(verification: &CompositionVerification, key: &str) -> Revalidation {
    verification.classify(key).unwrap().class
}

#[test]
fn a_moved_standing_changes_the_classification_and_leaves_the_sibling_valid() {
    let none = BTreeMap::new();
    let before = graph_of(VERIFIED, 70);

    // Nothing moved: every claim is still valid.
    let same = CompositionVerification::verify(Some(&before), &graph_of(VERIFIED, 70), &none, None)
        .unwrap();
    assert_eq!(class_of(&same, "effect_a"), Revalidation::StillValid);
    assert_eq!(class_of(&same, "effect_b"), Revalidation::StillValid);

    // The provider's trust held, then no longer: the claim is withdrawn for provider trust.
    let lost = CompositionVerification::verify(
        Some(&before),
        &graph_of(Some(ProviderTrust::Unverified), 70),
        &none,
        None,
    )
    .unwrap();
    assert_eq!(class_of(&lost, "effect_a"), Revalidation::Withdrawn);
    assert!(lost.classify("effect_a").unwrap().withdrawn_for(WithdrawalKind::ProviderTrust));
    assert_eq!(class_of(&lost, "effect_b"), Revalidation::StillValid);
    let refused = lost.require_valid("effect_a").unwrap_err();
    assert!(
        format!("{refused:?}").contains("composition_verification.provider_trust_withdrawn"),
        "{refused:?}"
    );
    assert!(lost.require_valid("effect_b").is_ok(), "the sibling stays valid");

    // The record behind the claim's uncertainty statement was replaced at the same target:
    // the claim recomputes; its sibling is untouched.
    let replaced =
        CompositionVerification::verify(Some(&before), &graph_of(VERIFIED, 71), &none, None)
            .unwrap();
    assert_eq!(class_of(&replaced, "effect_a"), Revalidation::Recompute);
    assert_eq!(class_of(&replaced, "effect_b"), Revalidation::StillValid);
}

/// Claim `effect_trial` (a record held at `held_at`) with `effect_heir` inheriting it, at a
/// target whose data identity is `target_data`.
fn scene_with_record(
    target_data: u8,
    held_at: Option<SemanticDigest>,
    record: u8,
    heir_record: u8,
) -> antecedent::analysis::transport_revalidation::TransportScene {
    let builder = TransportSceneBuilder::new("clinic_b", digest(target_data))
        .source("trial", digest(10))
        .source("registry", digest(20));
    let own_target = held_at.unwrap_or_else(|| builder.target_identity());
    builder
        .transported_claim(
            "effect_trial",
            &response("effect", "trial", "clinic_b", 2.0),
            SceneCalibration::Held { record: digest(record), established_at_target: own_target },
            VERIFIED,
        )
        .transported_claim(
            "effect_heir",
            &response("effect", "trial", "clinic_b", 2.0),
            SceneCalibration::Inherited {
                from: "effect_trial".into(),
                record: digest(heir_record),
                established_at_target: own_target,
            },
            VERIFIED,
        )
        .transported_claim(
            "effect_registry",
            &response("effect_r", "registry", "clinic_b", 1.5),
            SceneCalibration::PointOnly,
            VERIFIED,
        )
        .build()
        .unwrap()
}

#[test]
fn a_record_from_another_target_is_withdrawn_and_its_heir_is_refused_the_record() {
    let earlier_target = TransportSceneBuilder::new("clinic_b", digest(1)).target_identity();
    let before = scene_with_record(1, Some(earlier_target), 77, 77);
    let none = BTreeMap::new();

    // At the same target the record holds, and so does the heir's claim to it.
    let held =
        revalidate(Some(&before), &scene_with_record(1, Some(earlier_target), 77, 77), &none)
            .unwrap();
    for key in ["effect_trial", "effect_heir", "effect_registry"] {
        assert_eq!(held.claim(key).unwrap().class, Revalidation::StillValid, "{key}");
    }

    // The target's data changed while the record names the earlier target: the claim is
    // withdrawn for calibration with its point value untouched; a claim that carries no
    // guarantee only recomputes.
    let after = scene_with_record(2, Some(earlier_target), 77, 77);
    let moved = revalidate(Some(&before), &after, &none).unwrap();
    let trial = moved.claim("effect_trial").unwrap();
    assert_eq!(trial.class, Revalidation::Withdrawn);
    assert!(trial.calibration_withdrawn());
    assert!(trial.reasons.contains(&TransportReason::CalibrationFromAnotherTarget));
    assert_eq!(moved.claim("effect_registry").unwrap().class, Revalidation::Recompute);
    assert!(!moved.claim("effect_registry").unwrap().calibration_withdrawn());

    // A replaced record at the same target recomputes its owner and refuses the heir the
    // record it inherited.
    let replaced =
        revalidate(Some(&before), &scene_with_record(1, Some(earlier_target), 78, 77), &none)
            .unwrap();
    assert_eq!(replaced.claim("effect_trial").unwrap().class, Revalidation::Recompute);
    let heir = replaced.claim("effect_heir").unwrap();
    assert_eq!(heir.class, Revalidation::Withdrawn);
    assert!(heir.calibration_withdrawn());
    assert_eq!(replaced.claim("effect_registry").unwrap().class, Revalidation::StillValid);
}

#[test]
fn a_declared_claim_with_no_stated_trust_is_not_a_license_in_a_scene() {
    let scene = TransportSceneBuilder::new("clinic_b", digest(1))
        .source("trial", digest(10))
        .transported_claim(
            "effect",
            &response("effect", "trial", "clinic_b", 2.0),
            SceneCalibration::PointOnly,
            None,
        )
        .build()
        .unwrap();
    assert_eq!(scene.claims[0].trust, TrustStanding::Undeclared);
    let report = revalidate(None, &scene, &BTreeMap::new()).unwrap();
    assert_eq!(report.claim("effect").unwrap().class, Revalidation::Withdrawn);
    let refused = report.require_valid("effect").unwrap_err();
    assert!(
        format!("{refused:?}").contains("composition_verification.provider_trust_withdrawn"),
        "{refused:?}"
    );

    // Stating the level makes the same claim usable.
    let stated = TransportSceneBuilder::new("clinic_b", digest(1))
        .source("trial", digest(10))
        .transported_claim(
            "effect",
            &response("effect", "trial", "clinic_b", 2.0),
            SceneCalibration::PointOnly,
            VERIFIED,
        )
        .build()
        .unwrap();
    assert_eq!(stated.claims[0].trust, TrustStanding::Held);
    let report = revalidate(None, &stated, &BTreeMap::new()).unwrap();
    assert_eq!(report.claim("effect").unwrap().class, Revalidation::StillValid);
}
