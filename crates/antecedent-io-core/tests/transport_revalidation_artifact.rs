//! Transport revalidation artifact: a fresh consumer re-derives the
//! classification of every claim from the stored scenes and refuses semantic
//! mutations of a resealed report.
//!
//! The scene is hand computable. One source `s1` feeds the transported claim
//! `effect`; the decision `choice` rests on `effect`; `sibling` stands alone.
//! Replacing the data behind `s1` recomputes `effect` and `choice` and leaves
//! `sibling` valid. A calibration record established at another target is
//! withdrawn at this one and never inherited.

use std::collections::BTreeMap;

use antecedent_contract::composition_verification::Revalidation;
use antecedent_contract::composition_verification::SupportStanding;
use antecedent_contract::transport_revalidation::{
    SceneCalibration, SceneClaim, SceneClaimKind, SceneSource, SceneTarget, TransportScene,
};
use antecedent_core::SemanticDigest;
use antecedent_io_core::container_kind;
use antecedent_io_core::transport_revalidation_artifact::{
    TRANSPORT_REVALIDATION_ARTIFACT_FEATURE, TransportRevalidationArtifact,
    TransportRevalidationArtifactError, TransportRevalidationMeta,
};

fn d(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

fn scene(source_data: u8, effect_calibration: SceneCalibration) -> TransportScene {
    let mut effect = SceneClaim::new("effect", SceneClaimKind::Transported, d(30), d(20));
    effect.sources = vec!["s1".to_owned()];
    effect.calibration = effect_calibration;
    let mut choice = SceneClaim::new("choice", SceneClaimKind::Decision, d(40), d(21));
    choice.rests_on = vec!["effect".to_owned()];
    let sibling = SceneClaim::new("sibling", SceneClaimKind::Transported, d(50), d(22));
    TransportScene {
        target: SceneTarget { identity: d(1), support: SupportStanding::Supported },
        selection_diagram: d(2),
        sources: vec![SceneSource { key: "s1".to_owned(), identity: d(source_data) }],
        claims: vec![effect, choice, sibling],
    }
}

fn sealed_after_new_data() -> (TransportRevalidationArtifact, TransportRevalidationMeta) {
    let before = scene(10, SceneCalibration::PointOnly);
    let after = scene(11, SceneCalibration::PointOnly);
    TransportRevalidationArtifact::seal(Some(&before), &after, &BTreeMap::new()).expect("seals")
}

fn reseal(mut meta: TransportRevalidationMeta) -> TransportRevalidationMeta {
    TransportRevalidationArtifact::reseal_identity(&mut meta).expect("identity");
    meta
}

fn claim<'a>(
    meta: &'a TransportRevalidationMeta,
    key: &str,
) -> &'a antecedent_io_core::transport_revalidation_artifact::ClaimReportWire {
    meta.claims.iter().find(|c| c.key == key).expect("claim in report")
}

#[test]
fn new_source_data_recomputes_its_claims_and_leaves_an_unrelated_claim_valid() {
    let (artifact, meta) = sealed_after_new_data();
    assert_eq!(meta.feature, TRANSPORT_REVALIDATION_ARTIFACT_FEATURE);

    let effect = claim(&meta, "effect");
    assert_eq!(effect.class, "recompute");
    assert_eq!(effect.reasons, vec!["source_evidence_changed:s1".to_owned()]);
    assert!(!effect.authority_retained);

    assert_eq!(claim(&meta, "choice").class, "recompute");

    let sibling = claim(&meta, "sibling");
    assert_eq!(sibling.class, "still_valid");
    assert!(sibling.reasons.is_empty());
    assert!(sibling.authority_retained);

    let report = artifact.revalidation();
    assert_eq!(report.keys_of(Revalidation::Recompute), vec!["choice", "effect"]);
    assert_eq!(report.keys_of(Revalidation::StillValid), vec!["sibling"]);
    assert!(report.removed().is_empty());
}

#[test]
fn a_fresh_consumer_rederives_the_report_from_bytes() {
    let (artifact, meta) = sealed_after_new_data();
    let bytes = TransportRevalidationArtifact::meta_to_bytes(&meta, "report-1").expect("encodes");
    assert_eq!(container_kind(&bytes).as_deref(), Some("transport_revalidation_v1"));
    let consumed = TransportRevalidationArtifact::from_bytes(&bytes, Some(artifact.identity()))
        .expect("a fresh consumer accepts the sealed report");
    assert_eq!(consumed.identity(), artifact.identity());
    assert_eq!(consumed.revalidation_identity(), artifact.revalidation_identity());
    assert_eq!(consumed.current(), artifact.current());
    assert_eq!(consumed.previous(), artifact.previous());
    assert_eq!(
        TransportRevalidationArtifact::inspect(&bytes).expect("inspect").claims,
        meta.claims
    );
}

#[test]
fn a_calibration_record_from_another_target_is_withdrawn_and_never_inherited() {
    let current = scene(10, SceneCalibration::Held { record: d(90), established_at_target: d(99) });
    let (_, meta) =
        TransportRevalidationArtifact::seal(None, &current, &BTreeMap::new()).expect("seals");
    let effect = claim(&meta, "effect");
    assert_eq!(effect.class, "withdrawn");
    assert_eq!(effect.reasons, vec!["calibration_from_another_target".to_owned()]);
    assert_eq!(effect.calibration_state, "lost");
    assert!(effect.refusal_code.is_some());
    // The point value of the withdrawn claim is not recomputed by this report.
    assert_eq!(claim(&meta, "sibling").class, "still_valid");
}

#[test]
fn a_resealed_report_that_upgrades_a_claim_does_not_replay() {
    let (_, meta) = sealed_after_new_data();
    let mut edited = meta.clone();
    let effect = edited.claims.iter_mut().find(|c| c.key == "effect").unwrap();
    effect.class = "still_valid".to_owned();
    effect.reasons.clear();
    effect.authority_retained = true;
    let error = TransportRevalidationArtifact::from_meta(&reseal(edited), None)
        .expect_err("upgraded classification");
    assert_eq!(error, TransportRevalidationArtifactError::ReplayMismatch("claim classification"));
    let (code, detail, _) = error.refusal().expect("a registered refusal");
    assert_eq!(code, "invalid_argument");
    assert_eq!(detail, "transport_revalidation.replay_mismatch");
}

#[test]
fn a_resealed_scene_edit_does_not_replay_and_an_unsealed_edit_changes_identity() {
    let (_, meta) = sealed_after_new_data();

    // Another source identity at unchanged shape: the stored classification no longer matches.
    let mut moved = meta.clone();
    moved.current.sources[0].identity = d(12).to_hex();
    let error =
        TransportRevalidationArtifact::from_meta(&reseal(moved), None).expect_err("scene edited");
    assert!(matches!(error, TransportRevalidationArtifactError::ReplayMismatch(_)), "{error:?}");

    // The same edit without resealing is an identity mismatch or a replay mismatch, never accepted.
    let mut unsealed = meta.clone();
    unsealed.removed.push("ghost".to_owned());
    assert!(TransportRevalidationArtifact::from_meta(&unsealed, None).is_err());

    // A retained identity of another report.
    let error = TransportRevalidationArtifact::from_meta(&meta, Some(&"0".repeat(64)))
        .expect_err("another report");
    assert_eq!(
        error,
        TransportRevalidationArtifactError::IdentityMismatch { field: "retained_identity" }
    );
}

#[test]
fn unknown_features_versions_and_noncanonical_scenes_are_refused() {
    let (_, meta) = sealed_after_new_data();

    let mut required = meta.clone();
    required.required_features.push("future".to_owned());
    assert_eq!(
        TransportRevalidationArtifact::from_meta(&reseal(required), None).expect_err("feature"),
        TransportRevalidationArtifactError::UnsupportedSemantics("required feature")
    );

    let mut version = meta.clone();
    version.version = 2;
    assert_eq!(
        TransportRevalidationArtifact::from_meta(&version, None).expect_err("version"),
        TransportRevalidationArtifactError::UnsupportedVersion { version: 2 }
    );

    let mut unordered = meta.clone();
    unordered.current.claims.reverse();
    let error = TransportRevalidationArtifact::from_meta(&reseal(unordered), None)
        .expect_err("claims not in key order");
    assert!(matches!(error, TransportRevalidationArtifactError::Malformed(_)), "{error:?}");
}
