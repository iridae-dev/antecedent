//! Transported claims are revalidated through the facade when a source, the target or
//! the support changes, so the rule that a calibration record is never inherited by
//! another target runs in the product path and its classification surfaces through
//! `inspect` and the exported composition verification.
//!
//! Oracles (hand-classified):
//!
//! - nothing changed: every claim and the decision resting on it is `still_valid`;
//! - one source's evidence changed: the claims counting it and the decision resting on
//!   them `recompute`; the claim of another source is untouched;
//! - the target's identity changed while a record was established at the earlier target:
//!   the claim is `withdrawn` for calibration with its point value left as it was;
//! - the target outside its support: every transported claim is `withdrawn`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent::analysis::composition_verification::{
    Revalidation, SupportStanding, TrustStanding, consume_composition_verification_artifact,
};
use antecedent::analysis::transport_revalidation::{
    DETAIL_DUPLICATE_CLAIM, DETAIL_SOURCE_UNDECLARED, DETAIL_TARGET_MISMATCH, SceneCalibration,
    SceneClaim, SceneClaimKind, SceneTarget, TransportClaimState, TransportReason, TransportScene,
    TransportSceneBuilder, claim_node_key, revalidate,
};
use antecedent::analysis::{TransportDeclaration, TransportedResponse, TransportedRoute};
use antecedent_contract::{QuantityRole, ScientificQuantity};
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

/// Two transported claims (one per source population) and a decision resting on both.
fn scene(
    target_data: u8,
    trial_evidence: u8,
    calibration_at: Option<SemanticDigest>,
) -> TransportScene {
    let builder = TransportSceneBuilder::new("clinic_b", digest(target_data))
        .source("trial", digest(trial_evidence))
        .source("registry", digest(20));
    let held = calibration_at.map_or(SceneCalibration::PointOnly, |established_at_target| {
        SceneCalibration::Held { record: digest(77), established_at_target }
    });
    builder
        .transported(
            "effect_trial",
            &response("effect", "trial", "clinic_b", 2.0),
            held,
            TrustStanding::NotApplicable,
        )
        .transported(
            "effect_registry",
            &response("effect_r", "registry", "clinic_b", 1.5),
            SceneCalibration::PointOnly,
            TrustStanding::NotApplicable,
        )
        .decision("treat_or_wait", digest(40), digest(41), &["effect_trial", "effect_registry"])
        .build()
        .unwrap()
}

fn class(
    report: &antecedent::analysis::transport_revalidation::TransportRevalidationReport,
    key: &str,
) -> Revalidation {
    report.claim(key).unwrap().class
}

#[test]
fn an_unchanged_scene_is_still_valid_and_its_classification_replays_from_bytes() {
    let first = scene(1, 10, None);
    let report = revalidate(Some(&first), &scene(1, 10, None), &BTreeMap::new()).unwrap();
    for key in ["effect_trial", "effect_registry", "treat_or_wait"] {
        assert_eq!(class(&report, key), Revalidation::StillValid, "{key}");
        assert!(report.claim(key).unwrap().reasons.is_empty());
        assert!(report.require_valid(key).is_ok());
    }
    let inspection = report.inspect();
    assert_eq!(inspection.claims.len(), 3);
    assert!(inspection.claims.iter().all(|c| c.class == "still_valid" && c.reasons.is_empty()));
    assert!(inspection.removed.is_empty());

    // The same graphs sealed as `composition_verification_v1`: a fresh consumer replays.
    let bytes = report.export("transport-revalidation").unwrap();
    let consumed =
        consume_composition_verification_artifact(&bytes, Some(report.verification().identity()))
            .unwrap();
    assert_eq!(
        consumed.revalidation().claim(&claim_node_key("effect_trial")).unwrap().class,
        Revalidation::StillValid
    );
    // A first observation (nothing earlier) classifies the same scene the same way.
    let first_report = revalidate(None, &first, &BTreeMap::new()).unwrap();
    assert_eq!(class(&first_report, "treat_or_wait"), Revalidation::StillValid);
}

#[test]
fn a_changed_source_recomputes_only_the_claims_that_count_it() {
    let before = scene(1, 10, None);
    let after = scene(1, 11, None);
    let report = revalidate(Some(&before), &after, &BTreeMap::new()).unwrap();
    assert_eq!(class(&report, "effect_trial"), Revalidation::Recompute);
    assert_eq!(class(&report, "effect_registry"), Revalidation::StillValid);
    assert_eq!(class(&report, "treat_or_wait"), Revalidation::Recompute);
    let reasons = &report.claim("effect_trial").unwrap().reasons;
    assert!(
        reasons.contains(&TransportReason::SourceEvidenceChanged { source: "trial".into() }),
        "{reasons:?}"
    );
    // The earlier fit and uncertainty carry no authority and a refusal names the remedy.
    let refusal = report.require_valid("effect_trial").unwrap_err();
    assert!(refusal.remedy.is_some());
    let inspection = report.inspect();
    let effect = inspection.claims.iter().find(|c| c.key == "effect_trial").unwrap();
    assert_eq!(effect.class, "recompute");
    assert_eq!(effect.prior_authority, "not_retained");
    assert!(effect.reasons.contains(&"source_evidence_changed"));
}

#[test]
fn a_calibration_record_is_never_inherited_by_another_target() {
    // A record established at the target of the earlier scene.
    let builder_before = TransportSceneBuilder::new("clinic_b", digest(1));
    let own_target = builder_before.target_identity();
    let before = scene(1, 10, Some(own_target));
    let held =
        revalidate(Some(&before), &scene(1, 10, Some(own_target)), &BTreeMap::new()).unwrap();
    assert_eq!(class(&held, "effect_trial"), Revalidation::StillValid);
    assert!(!held.claim("effect_trial").unwrap().calibration_withdrawn());

    // The target's data changed: the claim now answers another target, and the record
    // still names the earlier one.
    let after = scene(2, 10, Some(own_target));
    let report = revalidate(Some(&before), &after, &BTreeMap::new()).unwrap();
    let claim = report.claim("effect_trial").unwrap();
    assert_eq!(claim.class, Revalidation::Withdrawn);
    assert!(claim.reasons.contains(&TransportReason::CalibrationFromAnotherTarget));
    assert!(claim.calibration_withdrawn());
    let refusal = report.require_valid("effect_trial").unwrap_err();
    assert_eq!(refusal.code, "cell_not_licensed");
    // A point-only claim of the same scene is recomputed, not withdrawn: it states no
    // guarantee to lose.
    assert_eq!(class(&report, "effect_registry"), Revalidation::Recompute);

    let inspection = report.inspect();
    let effect = inspection.claims.iter().find(|c| c.key == "effect_trial").unwrap();
    assert!(effect.calibration_withdrawn);
    assert!(effect.reasons.contains(&"calibration_from_another_target"));
}

#[test]
fn an_unsupported_target_withdraws_every_transported_claim() {
    let before = scene(1, 10, None);
    let builder = TransportSceneBuilder::new("clinic_b", digest(1))
        .target_support(SupportStanding::Unsupported { coordinates: vec!["age".into()] })
        .source("trial", digest(10))
        .source("registry", digest(20))
        .transported(
            "effect_trial",
            &response("effect", "trial", "clinic_b", 2.0),
            SceneCalibration::PointOnly,
            TrustStanding::NotApplicable,
        );
    let after = builder.build().unwrap();
    let report = revalidate(Some(&before), &after, &BTreeMap::new()).unwrap();
    let claim = report.claim("effect_trial").unwrap();
    assert_eq!(claim.class, Revalidation::Withdrawn);
    assert!(claim.reasons.iter().any(|r| matches!(r, TransportReason::TargetUnsupported { .. })));
    // The claims the later scene dropped are reported as removed.
    assert_eq!(
        report.revalidation().removed(),
        &["effect_registry".to_owned(), "treat_or_wait".to_owned()]
    );
}

#[test]
fn a_refused_observation_leaves_the_state_unchanged() {
    let mut state = TransportClaimState::new();
    let first = scene(1, 10, None);
    state.observe(first.clone(), &BTreeMap::new()).unwrap();
    assert_eq!(state.scene(), Some(&first));

    // A scene whose decision rests on a claim it does not declare does not build.
    let mut broken = scene(1, 11, None);
    let mut ghost =
        SceneClaim::new("ghost_decision", SceneClaimKind::Decision, digest(5), digest(6));
    ghost.rests_on = vec!["not_declared".into()];
    broken.claims.push(ghost);
    let error = state.observe(broken, &BTreeMap::new()).unwrap_err();
    assert_eq!(error.detail, "transport_revalidation.scene_invalid");
    assert_eq!(state.scene(), Some(&first));

    // A good scene advances the state.
    let next = scene(1, 11, None);
    let report = state.observe(next.clone(), &BTreeMap::new()).unwrap();
    assert_eq!(class(&report, "effect_trial"), Revalidation::Recompute);
    assert_eq!(state.scene(), Some(&next));
}

#[test]
fn the_builder_refuses_what_a_scene_cannot_state() {
    // A claim about another target is not revalidated against this one.
    let error = TransportSceneBuilder::new("clinic_b", digest(1))
        .source("trial", digest(10))
        .transported(
            "elsewhere",
            &response("effect", "trial", "clinic_c", 2.0),
            SceneCalibration::PointOnly,
            TrustStanding::NotApplicable,
        )
        .build()
        .unwrap_err();
    assert_eq!(error.detail, DETAIL_TARGET_MISMATCH);
    assert_eq!(error.code, "quantity_semantics_mismatch");

    // A claim counting a source population the scene does not declare.
    let error = TransportSceneBuilder::new("clinic_b", digest(1))
        .transported(
            "orphan",
            &response("effect", "trial", "clinic_b", 2.0),
            SceneCalibration::PointOnly,
            TrustStanding::NotApplicable,
        )
        .build()
        .unwrap_err();
    assert_eq!(error.detail, DETAIL_SOURCE_UNDECLARED);

    // A repeated key.
    let error = TransportSceneBuilder::new("clinic_b", digest(1))
        .source("trial", digest(10))
        .transported(
            "effect",
            &response("effect", "trial", "clinic_b", 2.0),
            SceneCalibration::PointOnly,
            TrustStanding::NotApplicable,
        )
        .transported(
            "effect",
            &response("effect", "trial", "clinic_b", 3.0),
            SceneCalibration::PointOnly,
            TrustStanding::NotApplicable,
        )
        .build()
        .unwrap_err();
    assert_eq!(error.detail, DETAIL_DUPLICATE_CLAIM);
}

#[test]
fn the_scene_identity_follows_the_response_value_at_an_unchanged_shape() {
    let build = |value: f64| {
        TransportSceneBuilder::new("clinic_b", digest(1))
            .source("trial", digest(10))
            .transported(
                "effect",
                &response("effect", "trial", "clinic_b", value),
                SceneCalibration::PointOnly,
                TrustStanding::NotApplicable,
            )
            .build()
            .unwrap()
    };
    let a = build(2.0);
    let b = build(2.5);
    assert_ne!(a.claims[0].identity, b.claims[0].identity);
    // Target, quantity and structure are unchanged: only the claim's own numbers moved.
    assert_eq!(a.claims[0].quantity, b.claims[0].quantity);
    assert_eq!(a.claims[0].structure, b.claims[0].structure);
    let report = revalidate(Some(&a), &b, &BTreeMap::new()).unwrap();
    assert_eq!(class(&report, "effect"), Revalidation::Recompute);
    let target: &SceneTarget = &a.target;
    assert_eq!(target.identity, b.target.identity);
}
