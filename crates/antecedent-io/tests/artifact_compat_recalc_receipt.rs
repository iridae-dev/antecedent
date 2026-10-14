//! Compatibility class of the recalculation receipt across the introduction of typed
//! standing inputs and the change of the composite decision's own identity.
//!
//! Class: byte-compatible. A receipt stores the declared own digest of every stage as an
//! opaque value and recomputes effective identities from them. A workflow that never
//! declared a standing input has the stages and dependency edges it always had, so a
//! receipt written before they existed is consumed unchanged; a decision whose own digest
//! moved is an ordinary recomputation, not a refusal.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::recalc::{
    RecalcCapabilities, RetargetSupport, Stage, StageIdentities, StageIdentity,
};
use antecedent_io::recalc_receipt_artifact::{CountsWire, RecalcReceiptArtifact};

fn digest(label: &str, text: &str) -> StageIdentity {
    StageIdentity::of(label, &[text.as_bytes()])
}

/// A workflow without any standing input, as every workflow was before they existed.
fn workflow(decision_own: StageIdentity) -> StageIdentities {
    let mut ids = StageIdentities::new();
    for stage in [
        Stage::Graph,
        Stage::Query,
        Stage::Regime,
        Stage::Evidence,
        Stage::SourcePopulation,
        Stage::TargetPopulation,
        Stage::DataSnapshot,
        Stage::RowDesign,
        Stage::TreatmentGrid,
        Stage::LearnerFoldsRng,
        Stage::Utility,
    ] {
        ids.set(stage, digest(&stage.label(), "v1"));
    }
    for stage in [Stage::Identification, Stage::ScoreArtifact, Stage::Law] {
        ids.set(stage, digest(&stage.label(), "derived"));
    }
    ids.set(Stage::Decision, decision_own);
    ids
}

fn decided() -> BTreeMap<Stage, CountsWire> {
    let mut counts = BTreeMap::new();
    counts.insert(Stage::Decision, CountsWire { decisions: 1, ..CountsWire::default() });
    counts
}

#[test]
fn a_receipt_that_predates_standing_inputs_is_consumed_unchanged() {
    // The previous terminal-decision own identity and the current one are both opaque
    // digests to the receipt.
    let previous = StageIdentity::of("composite.original_support_terminal.v1", &[]);
    let requested = StageIdentity::of(
        "composite.original_support_terminal.v2",
        &[b"contract", b"policy", b"order", b"terminal"],
    );
    assert_ne!(previous, requested);

    let artifact = RecalcReceiptArtifact::seal(
        &workflow(previous),
        &workflow(requested),
        &RecalcCapabilities::in_process(RetargetSupport::Licensed),
        &decided(),
    )
    .expect("a moved decision identity is a recomputation, not a refusal");

    // Only the stages the workflow declared are in the receipt: no standing input appears.
    let stages: Vec<&str> = artifact.meta().entries.iter().map(|e| e.stage.as_str()).collect();
    for standing in ["calibration", "representation", "coupling"] {
        assert!(!stages.contains(&standing), "{standing} must not appear: {stages:?}");
    }
    assert!(stages.iter().all(|stage| !stage.starts_with("provider_standing")));
    let decision = artifact
        .meta()
        .entries
        .iter()
        .find(|entry| entry.stage == "decision")
        .expect("the decision stage");
    assert!(decision.status.starts_with("recomputed"), "{}", decision.status);

    // The sealed receipt round-trips and a consumer retaining its identity accepts it.
    let bytes = artifact.to_bytes("compat-receipt").expect("encodes");
    let consumed = RecalcReceiptArtifact::from_bytes(&bytes, Some(artifact.receipt_identity()))
        .expect("a fresh consumer re-derives the same plan");
    assert_eq!(consumed.receipt_identity(), artifact.receipt_identity());
    assert_eq!(consumed.plan_identity(), artifact.plan_identity());
}
