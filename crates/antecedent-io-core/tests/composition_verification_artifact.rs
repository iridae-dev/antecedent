//! Composition verification artifact: a fresh consumer replays the
//! classification of every claim from the stored graphs and refuses semantic
//! mutations.
//!
//! The graph is hand computable. `cohort` feeds `model` (data), `fit` reuses
//! `model`'s structural derivation and reads `cohort` (data), `claim` reads
//! `fit` and inherits its calibration record; `sibling` stands alone. Replacing
//! the cohort's data recomputes `model`, `fit` and `claim` and leaves
//! `sibling` valid; withdrawing the record `fit` holds withdraws `claim` for
//! calibration only.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_contract::composition_verification::{
    CalibrationDeclaration, CompositionGraph, CompositionNode, DependencyKind, NodeKind,
    ObservedEffect, PriorAuthority, Revalidation, WithdrawalKind, shared_data_relations,
};
use antecedent_contract::evidence_ledger::{
    EvidenceLedger, ObservationId, SourceDeclaration, SourceKind,
};
use antecedent_core::{CancellationToken, SemanticDigest};
use antecedent_io_core::composition_verification_artifact::{
    COMPOSITION_VERIFICATION_ARTIFACT_FEATURE, CompositionVerificationArtifact,
    CompositionVerificationArtifactError, CompositionVerificationMeta,
};
use antecedent_io_core::container_kind;

fn d(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

fn graph(cohort: u8, fit_calibration: CalibrationDeclaration) -> CompositionGraph {
    let mut g = CompositionGraph::new();
    let node = |key: &str, kind, id| CompositionNode::new(key, kind, d(id));
    g.add_node(node("cohort", NodeKind::CohortSource, cohort)).expect("node");
    g.add_node(node("model", NodeKind::MeasurementModel, 2).with_structure(d(3))).expect("node");
    g.add_node(
        node("fit", NodeKind::LedgerResult, 4)
            .with_structure(d(5))
            .with_coordinate(d(100))
            .with_calibration(fit_calibration),
    )
    .expect("node");
    g.add_node(node("claim", NodeKind::Claim, 6).with_coordinate(d(100)).with_calibration(
        CalibrationDeclaration::Inherited {
            from: "fit".to_owned(),
            record: d(90),
            coordinate: d(100),
        },
    ))
    .expect("node");
    g.add_node(node("sibling", NodeKind::Claim, 7)).expect("node");
    g.add_edge("model", "cohort", DependencyKind::Data).expect("edge");
    g.add_edge("fit", "model", DependencyKind::StructuralDerivation).expect("edge");
    g.add_edge("fit", "cohort", DependencyKind::Data).expect("edge");
    g.add_edge("claim", "fit", DependencyKind::Data).expect("edge");
    g.add_edge("claim", "fit", DependencyKind::Calibration).expect("edge");
    g
}

fn held() -> CalibrationDeclaration {
    CalibrationDeclaration::Held { record: d(90), coordinate: d(100) }
}

struct Sealed {
    meta: CompositionVerificationMeta,
    identity: String,
    before: CompositionGraph,
    after: CompositionGraph,
}

fn sealed() -> Sealed {
    let before = graph(1, held());
    let after = graph(9, held());
    let (artifact, meta) =
        CompositionVerificationArtifact::seal(Some(&before), &after, &BTreeMap::new(), &[], None)
            .expect("seal");
    Sealed { identity: artifact.identity().to_owned(), meta, before, after }
}

fn reseal(mut meta: CompositionVerificationMeta) -> CompositionVerificationMeta {
    CompositionVerificationArtifact::reseal_identity(&mut meta);
    meta
}

fn replay_mismatch(meta: &CompositionVerificationMeta) -> bool {
    matches!(
        CompositionVerificationArtifact::from_meta(&reseal(meta.clone()), None),
        Err(CompositionVerificationArtifactError::ReplayMismatch(_))
    )
}

#[test]
fn a_fresh_consumer_sees_the_classification_and_the_lineage() {
    let s = sealed();
    let bytes = CompositionVerificationArtifact::to_bytes(
        Some(&s.before),
        &s.after,
        &BTreeMap::new(),
        &[],
        None,
        "composition-1",
    )
    .expect("bytes");
    assert_eq!(container_kind(&bytes).as_deref(), Some("composition_verification_v1"));
    let artifact =
        CompositionVerificationArtifact::from_bytes(&bytes, Some(&s.identity)).expect("consume");
    let r = artifact.revalidation();
    assert_eq!(r.keys_of(Revalidation::Recompute), vec!["claim", "cohort", "fit", "model"]);
    assert_eq!(r.keys_of(Revalidation::StillValid), vec!["sibling"]);
    assert!(r.keys_of(Revalidation::Withdrawn).is_empty());
    assert_eq!(r.claim("claim").expect("claim").authority, PriorAuthority::NotRetained);
    assert_eq!(r.claim("sibling").expect("claim").authority, PriorAuthority::Retained);
    let view = artifact.view("claim").expect("view");
    assert_eq!(
        view.upstream.iter().map(|e| e.key.as_str()).collect::<Vec<_>>(),
        vec!["cohort", "model", "fit"]
    );
    assert_eq!(artifact.previous().expect("previous").identity(), s.before.identity());
    assert_eq!(artifact.current().identity(), s.after.identity());
    // sealing is deterministic
    let again = CompositionVerificationArtifact::to_bytes(
        Some(&s.before),
        &s.after,
        &BTreeMap::new(),
        &[],
        None,
        "composition-1",
    )
    .expect("bytes");
    assert_eq!(bytes, again);
}

#[test]
fn a_first_verification_has_nothing_stale_and_a_withdrawn_calibration_is_carried() {
    let first = graph(1, held());
    let (artifact, meta) =
        CompositionVerificationArtifact::seal(None, &first, &BTreeMap::new(), &[], None)
            .expect("seal");
    assert!(meta.previous.is_none());
    assert_eq!(artifact.revalidation().keys_of(Revalidation::StillValid).len(), 5);

    // the record `fit` held is dropped: `claim` is withdrawn for calibration only
    let dropped = graph(1, CalibrationDeclaration::PointOnly);
    let (artifact, _) =
        CompositionVerificationArtifact::seal(Some(&first), &dropped, &BTreeMap::new(), &[], None)
            .expect("seal");
    let claim = artifact.revalidation().claim("claim").expect("claim");
    assert_eq!(claim.class, Revalidation::Withdrawn);
    assert!(claim.numerical.is_empty());
    assert!(claim.withdrawn_for(WithdrawalKind::Calibration));
    let refusal = artifact.revalidation().require_valid("claim").expect_err("refused");
    assert_eq!(refusal.code, "cell_not_licensed");
}

#[test]
fn observed_effects_are_part_of_the_replayed_classification() {
    let g = graph(1, held());
    let mut effects = BTreeMap::new();
    effects.insert(
        "cohort".to_owned(),
        ObservedEffect::Withdrawn { cause: "ledger.operation:consent_withdrawn".to_owned() },
    );
    let (artifact, meta) =
        CompositionVerificationArtifact::seal(Some(&g), &g, &effects, &[], None).expect("seal");
    assert_eq!(
        artifact.revalidation().keys_of(Revalidation::Withdrawn),
        vec!["claim", "cohort", "fit", "model"]
    );
    // changing the stored effect kind changes the classification: it does not replay
    let mut edited = meta.clone();
    edited.effects[0].kind = "recomputed".to_owned();
    assert!(replay_mismatch(&edited));
    // dropping the effect is also a different classification
    let mut dropped = meta;
    dropped.effects.clear();
    assert!(replay_mismatch(&dropped));
}

#[test]
fn a_resealed_edit_of_a_stored_classification_does_not_replay() {
    let s = sealed();
    let claim = s.meta.claims.iter().position(|c| c.key == "claim").expect("claim");
    let sibling = s.meta.claims.iter().position(|c| c.key == "sibling").expect("sibling");

    // an upgraded class
    let mut upgraded = s.meta.clone();
    upgraded.claims[claim].class = "still_valid".to_owned();
    assert!(replay_mismatch(&upgraded));
    // retained authority claimed for a stale claim
    let mut authority = s.meta.clone();
    authority.claims[claim].authority_retained = true;
    assert!(replay_mismatch(&authority));
    // a numerical reason removed
    let mut reasons = s.meta.clone();
    reasons.claims[claim].numerical.pop();
    assert!(replay_mismatch(&reasons));
    // a withdrawal planted on the sibling
    let mut planted = s.meta.clone();
    planted.claims[sibling].class = "withdrawn".to_owned();
    assert!(replay_mismatch(&planted));
    // a calibration state rewritten
    let mut calibration = s.meta.clone();
    calibration.claims[claim].calibration_state = "held".to_owned();
    assert!(replay_mismatch(&calibration));
    // a lineage identity swapped for another claim's
    let mut lineage = s.meta.clone();
    lineage.claims[claim].lineage = s.meta.claims[sibling].lineage.clone();
    assert!(replay_mismatch(&lineage));
    // the classification identity
    let mut whole = s.meta.clone();
    whole.revalidation = d(1).to_hex();
    assert!(replay_mismatch(&whole));
}

#[test]
fn a_resealed_edit_of_the_graph_changes_the_classification_or_is_refused() {
    let s = sealed();

    // a node's calibration dropped in the current graph
    let mut calibration = s.meta.clone();
    let fit = calibration.current.nodes.iter_mut().find(|n| n.key == "fit").expect("fit");
    fit.calibration.kind = "point_only".to_owned();
    fit.calibration.record = None;
    fit.calibration.coordinate = None;
    assert!(replay_mismatch(&calibration));

    // a dependency removed
    let mut edge = s.meta.clone();
    edge.current.edges.retain(|e| !(e.downstream == "claim" && e.kind == "data"));
    assert!(replay_mismatch(&edge));

    // an identity edited back to the earlier one: nothing changed, nothing stale
    let mut identity = s.meta.clone();
    let cohort = identity.current.nodes.iter_mut().find(|n| n.key == "cohort").expect("cohort");
    cohort.identity = d(1).to_hex();
    assert!(replay_mismatch(&identity));

    // a cycle is not a graph
    let mut cycle = s.meta.clone();
    cycle.current.edges.push(antecedent_io_core::composition_verification_artifact::EdgeWire {
        downstream: "cohort".to_owned(),
        upstream: "claim".to_owned(),
        kind: "data".to_owned(),
    });
    assert!(matches!(
        CompositionVerificationArtifact::from_meta(&reseal(cycle), None),
        Err(CompositionVerificationArtifactError::Malformed(_))
    ));

    // an unknown dependency kind and an unknown node kind
    let mut kind = s.meta.clone();
    kind.current.edges[0].kind = "teleport".to_owned();
    assert!(matches!(
        CompositionVerificationArtifact::from_meta(&reseal(kind), None),
        Err(CompositionVerificationArtifactError::UnsupportedSemantics("dependency kind"))
    ));
    let mut node_kind = s.meta.clone();
    node_kind.current.nodes[0].kind = "oracle".to_owned();
    assert!(matches!(
        CompositionVerificationArtifact::from_meta(&reseal(node_kind), None),
        Err(CompositionVerificationArtifactError::UnsupportedSemantics("node kind"))
    ));

    // a graph not in canonical order was not written by this writer
    let mut unordered = s.meta.clone();
    unordered.current.nodes.swap(0, 1);
    assert!(matches!(
        CompositionVerificationArtifact::from_meta(&reseal(unordered), None),
        Err(CompositionVerificationArtifactError::Malformed(_))
    ));

    // inheriting calibration with no calibration dependency
    let mut inherit = s.meta.clone();
    inherit.current.edges.retain(|e| e.kind != "calibration");
    assert!(matches!(
        CompositionVerificationArtifact::from_meta(&reseal(inherit), None),
        Err(CompositionVerificationArtifactError::Malformed(_))
    ));
}

#[test]
fn an_unsealed_edit_is_an_identity_mismatch_and_a_retained_identity_binds_the_history() {
    let s = sealed();
    let mut edited = s.meta.clone();
    edited.effects.push(antecedent_io_core::composition_verification_artifact::EffectWire {
        key: "absent".to_owned(),
        kind: "recomputed".to_owned(),
        text: "stage".to_owned(),
    });
    // an effect on a node outside the graph is not consulted by replay, but it is stored data
    assert_eq!(
        CompositionVerificationArtifact::from_meta(&edited, None).expect_err("unsealed"),
        CompositionVerificationArtifactError::IdentityMismatch { field: "identity" }
    );
    let resealed = reseal(edited);
    let other = CompositionVerificationArtifact::from_meta(&resealed, None).expect("consistent");
    assert_ne!(other.identity(), s.identity);
    assert_eq!(
        CompositionVerificationArtifact::from_meta(&resealed, Some(&s.identity))
            .expect_err("retained"),
        CompositionVerificationArtifactError::IdentityMismatch { field: "retained_identity" }
    );
    let (code, detail, _) =
        CompositionVerificationArtifactError::IdentityMismatch { field: "identity" }
            .refusal()
            .expect("registered");
    assert_eq!(code, "route_not_supported");
    assert_eq!(detail, "composition_verification.identity_mismatch");
}

#[test]
fn versions_markers_and_corruption_are_refused() {
    let s = sealed();
    let mut newer = s.meta.clone();
    newer.version = 2;
    assert!(matches!(
        CompositionVerificationArtifact::from_meta(&newer, None),
        Err(CompositionVerificationArtifactError::UnsupportedVersion { version: 2 })
    ));
    let mut foreign = s.meta.clone();
    foreign.feature = "something_else".to_owned();
    assert!(matches!(
        CompositionVerificationArtifact::from_meta(&foreign, None),
        Err(CompositionVerificationArtifactError::UnsupportedSemantics("feature marker"))
    ));
    assert_eq!(COMPOSITION_VERIFICATION_ARTIFACT_FEATURE, "composition_verification_v1");
    let mut bytes = CompositionVerificationArtifact::to_bytes(
        Some(&s.before),
        &s.after,
        &BTreeMap::new(),
        &[],
        None,
        "composition-2",
    )
    .expect("bytes");
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0xff;
    assert!(CompositionVerificationArtifact::from_bytes(&bytes, None).is_err());
    assert!(CompositionVerificationArtifact::from_bytes(&bytes[..bytes.len() / 3], None).is_err());
    assert!(
        CompositionVerificationArtifact::to_bytes(None, &s.after, &BTreeMap::new(), &[], None, " ")
            .is_err()
    );
}

fn obs(range: std::ops::Range<u32>) -> Vec<ObservationId> {
    range.map(|i| ObservationId::from_key("cohort", &i.to_le_bytes())).collect()
}

#[test]
fn declared_shared_data_is_checked_against_the_consumers_ledger() {
    let mut ledger = EvidenceLedger::new();
    let a = ledger
        .add_source(
            SourceDeclaration::new("a", SourceKind::ObservationSet).with_observations(obs(0..100)),
        )
        .expect("a");
    let b = ledger
        .add_source(
            SourceDeclaration::new("b", SourceKind::ObservationSet).with_observations(obs(50..150)),
        )
        .expect("b");
    let mut g = CompositionGraph::new();
    g.add_node(CompositionNode::new("na", NodeKind::CohortSource, d(1)).with_evidence(vec![a]))
        .expect("node");
    g.add_node(CompositionNode::new("nb", NodeKind::CohortSource, d(2)).with_evidence(vec![b]))
        .expect("node");
    g.add_node(CompositionNode::new("pooled", NodeKind::LedgerResult, d(3))).expect("node");
    g.add_edge("pooled", "na", DependencyKind::Evidence).expect("edge");
    g.add_edge("pooled", "nb", DependencyKind::Evidence).expect("edge");

    let cancel = CancellationToken::new();
    let relations = shared_data_relations(&g, &ledger, &cancel).expect("relations");
    assert_eq!(relations.len(), 1);
    assert_eq!(relations[0].relation, "shared_data");
    assert_eq!(relations[0].shared_observations, 50);

    let state = ledger.state_digest();
    let (artifact, meta) =
        CompositionVerificationArtifact::seal(None, &g, &BTreeMap::new(), &relations, Some(state))
            .expect("seal");
    artifact.verify_ledger(&ledger, &cancel).expect("same ledger");
    let view = artifact.view("pooled").expect("view");
    assert_eq!(view.shared_data.len(), 1);
    assert_eq!(view.evidence.len(), 2);

    // a ledger at another state is not the one the relations were read from
    let mut moved = ledger.clone();
    moved.withdraw_source(a, "consent withdrawn").expect("withdraw");
    assert_eq!(
        artifact.verify_ledger(&moved, &cancel).expect_err("state"),
        CompositionVerificationArtifactError::LedgerMismatch("state digest")
    );

    // a resealed edit of the declared overlap changes the lineage identity
    let mut edited = meta.clone();
    edited.shared_data[0].shared_observations = 0;
    assert!(replay_mismatch(&edited));
    // a relation naming unknown nodes is malformed
    let mut unknown = meta;
    unknown.shared_data[0].second = "zzz".to_owned();
    assert!(matches!(
        CompositionVerificationArtifact::from_meta(&reseal(unknown), None),
        Err(CompositionVerificationArtifactError::Malformed(_))
    ));
    // relations declared with no ledger state still check against a ledger's overlap
    let (undated, _) =
        CompositionVerificationArtifact::seal(None, &g, &BTreeMap::new(), &[], None).expect("seal");
    assert_eq!(
        undated.verify_ledger(&ledger, &cancel).expect_err("relations"),
        CompositionVerificationArtifactError::LedgerMismatch("shared-data relations")
    );
}
