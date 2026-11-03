//! Composition bundle: the claim label, node statuses and digest reach of a composed graph,
//! derived by hand from the declared facts and edges.
//!
//! Components (upstream -> dependent), every embedded node publishing the facts written in its
//! own bytes:
//!
//! ```text
//! snapshot(ref S1) -> meanA (law mean_only, S1, v=3)  -> r_mean, r_mix, r_strict
//! snapshot         -> jointB (law joint_draw, S1, v=5) -> r_mix, r_joint, r_strict
//! snapshot         -> jointC (law joint_draw, S2, v=7) -> r_bad
//! ```
//!
//! Hand derivation:
//! - jointC states snapshot S2 against its upstream S1, so it fails at `GraphOrSnapshotMismatch`.
//! - r_strict requires a joint law but meanA supplies mean_only: `UnsupportedLaw`, no label.
//! - r_mean rests on a mean only: `PointOnlyAttested`. r_mix rests on a mean and a joint law; the
//!   mean dominates: `PointOnlyAttested`. r_joint rests on joint draws only: `JointDraw`.
//! - r_bad's only ancestor with a law is the failed jointC, which is ignored: verified, no label.
//! - Values are readable only from verified nodes: meanA = 3 and jointB = 5, not jointC.
//! - Changing meanA's bytes changes exactly meanA and its descendants r_mean, r_mix and r_strict.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_design::composition_bundle::{
    BundleConsumer, BundleLimits, BundleNode, BundleStage, ClaimLabel, CompositionBundle,
    ConsumedBundle, FACT_LAW, FACT_REQUIRES_LAW, FACT_SNAPSHOT, NodeFailure, NodeKind, NodeStatus,
    NodeVerifier, ProviderOrData, SuppliedSources, UpstreamNode, VerifiedNode,
};

/// Reads `law=<x>;snap=<y>;requires=<z>;v=<n>` from the node's own bytes.
struct TextVerifier(NodeKind);

impl NodeVerifier for TextVerifier {
    fn kind(&self) -> NodeKind {
        self.0
    }

    fn verify(
        &self,
        _node_id: &str,
        bytes: &[u8],
        _upstream: &[UpstreamNode<'_>],
    ) -> Result<VerifiedNode, NodeFailure> {
        let text = std::str::from_utf8(bytes).map_err(|_| NodeFailure {
            stage: BundleStage::TamperedQuantity,
            reason: "not text".into(),
        })?;
        let mut facts = BTreeMap::new();
        let mut values = Vec::new();
        for part in text.split(';') {
            let (key, value) = part.split_once('=').unwrap();
            match key {
                "law" => drop(facts.insert(FACT_LAW.to_owned(), value.to_owned())),
                "snap" => drop(facts.insert(FACT_SNAPSHOT.to_owned(), value.to_owned())),
                "requires" => drop(facts.insert(FACT_REQUIRES_LAW.to_owned(), value.to_owned())),
                "v" => values.push(("v".to_owned(), value.parse::<f64>().unwrap())),
                _ => {}
            }
        }
        Ok(VerifiedNode { identity: text.to_owned(), facts, values })
    }
}

fn node(id: &str, kind: NodeKind, text: &str) -> BundleNode {
    BundleNode::embedded(id, kind, text, text.as_bytes().to_vec())
}

fn bundle(mean_a: &str) -> CompositionBundle {
    let nodes = vec![
        BundleNode::reference(
            "snapshot",
            NodeKind::CausalContract,
            "program-1",
            ProviderOrData::Data { snapshot_id: "S1".into(), digest: "d1".into() },
        )
        .with_fact(FACT_SNAPSHOT, "S1"),
        node("meanA", NodeKind::ExternalClaim, mean_a),
        node("jointB", NodeKind::Distribution, "law=joint_draw;snap=S1;v=5"),
        node("jointC", NodeKind::Distribution, "law=joint_draw;snap=S2;v=7"),
        node("r_mean", NodeKind::DecisionResult, "snap=S1;v=1"),
        node("r_mix", NodeKind::DecisionResult, "snap=S1;v=2"),
        node("r_joint", NodeKind::DecisionResult, "requires=joint_draw;snap=S1;v=3"),
        node("r_strict", NodeKind::DecisionResult, "requires=joint_draw;snap=S1;v=4"),
        node("r_bad", NodeKind::DecisionResult, "snap=S1;v=6"),
    ];
    let edges: Vec<(String, String)> = [
        ("snapshot", "meanA"),
        ("snapshot", "jointB"),
        ("snapshot", "jointC"),
        ("meanA", "r_mean"),
        ("meanA", "r_mix"),
        ("jointB", "r_mix"),
        ("jointB", "r_joint"),
        ("meanA", "r_strict"),
        ("jointB", "r_strict"),
        ("jointC", "r_bad"),
    ]
    .iter()
    .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
    .collect();
    CompositionBundle::new(nodes, &edges, &BundleLimits::default()).unwrap()
}

fn consume(bundle: &CompositionBundle, supplied: &SuppliedSources) -> ConsumedBundle {
    let consumer = BundleConsumer::new()
        .register(Box::new(TextVerifier(NodeKind::ExternalClaim)))
        .register(Box::new(TextVerifier(NodeKind::Distribution)))
        .register(Box::new(TextVerifier(NodeKind::DecisionResult)));
    let limits = BundleLimits::default();
    let bytes = bundle.to_bytes("composed", &limits).unwrap();
    consumer.consume(&bytes, &limits, bundle.identity(), supplied).unwrap()
}

const MEAN_A: &str = "law=mean_only;snap=S1;v=3";

#[test]
fn claim_labels_statuses_and_readable_values_follow_the_hand_derivation() {
    let supplied = SuppliedSources::default().with_data("S1", "d1");
    let consumed = consume(&bundle(MEAN_A), &supplied);
    let label = |id: &str| consumed.node(id).unwrap().claim_label;
    assert_eq!(label("r_mean"), Some(ClaimLabel::PointOnlyAttested));
    assert_eq!(label("r_mix"), Some(ClaimLabel::PointOnlyAttested));
    assert_eq!(label("r_joint"), Some(ClaimLabel::JointDraw));
    assert_eq!(label("r_strict"), None);
    assert_eq!(label("r_bad"), None);

    let status = |id: &str| consumed.node(id).unwrap().status.clone();
    assert!(matches!(
        status("jointC"),
        NodeStatus::Failed { stage: BundleStage::GraphOrSnapshotMismatch, .. }
    ));
    assert!(matches!(
        status("r_strict"),
        NodeStatus::Failed { stage: BundleStage::UnsupportedLaw, .. }
    ));
    for id in ["snapshot", "meanA", "jointB", "r_mean", "r_mix", "r_joint", "r_bad"] {
        assert_eq!(status(id), NodeStatus::Verified, "{id}");
    }
    assert!(!consumed.all_verified());

    assert_eq!(consumed.value("meanA", "v"), Some(3.0));
    assert_eq!(consumed.value("jointB", "v"), Some(5.0));
    assert_eq!(consumed.value("jointC", "v"), None);
    assert_eq!(consumed.value("r_strict", "v"), None);
    assert_eq!(consumed.value("r_joint", "v"), Some(3.0));
}

#[test]
fn an_unsupplied_snapshot_reference_is_unresolved_but_does_not_fail_its_descendants() {
    let consumed = consume(&bundle(MEAN_A), &SuppliedSources::default());
    assert!(matches!(
        consumed.node("snapshot").unwrap().status,
        NodeStatus::ReferenceUnresolved { .. }
    ));
    assert_eq!(consumed.node("meanA").unwrap().status, NodeStatus::Verified);
    assert_eq!(consumed.node("r_joint").unwrap().claim_label, Some(ClaimLabel::JointDraw));
    // A wrong digest for the supplied snapshot fails the reference itself.
    let wrong = consume(&bundle(MEAN_A), &SuppliedSources::default().with_data("S1", "other"));
    assert!(matches!(
        wrong.node("snapshot").unwrap().status,
        NodeStatus::Failed { stage: BundleStage::GraphOrSnapshotMismatch, .. }
    ));
}

#[test]
fn changing_one_component_changes_the_digest_of_exactly_itself_and_its_descendants() {
    let before = bundle(MEAN_A);
    let after = bundle("law=mean_only;snap=S1;v=4");
    let ids = [
        "snapshot", "meanA", "jointB", "jointC", "r_mean", "r_mix", "r_joint", "r_strict", "r_bad",
    ];
    let changed: Vec<&str> = ids
        .into_iter()
        .filter(|id| before.chain_digest(id).unwrap() != after.chain_digest(id).unwrap())
        .collect();
    assert_eq!(changed, vec!["meanA", "r_mean", "r_mix", "r_strict"]);
    assert_ne!(before.identity(), after.identity());
}
