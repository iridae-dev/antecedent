//! A previous-release artifact adopts current digests and refuses a fresh-process
//! reuse of a derived stage.
//!
//! Payload format 3 is the current scientific-target identity. Score-row ids are
//! eight bytes. A receipt that says a derived stage
//! is in-process only cannot be reused after a fresh process loads it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::recalc::{
    RecalcCapabilities, ResumeContext, RetargetSupport, Stage, StageIdentities, StageIdentity,
};
use antecedent_core::{
    AssumptionSet, Conditioning, GraphFamily, InferenceSelection, Intervention, InterventionWorld,
    ModelAssumptions, ObservationUnit, OutcomeQuantity, QuantityUnits, ScientificTarget,
    ScientificWorldId, TargetForm, TargetPopulation, TimeFrame, Value, VariableId, WorldAction,
    WorldBinding,
};
use antecedent_io::recalc_receipt_artifact::{
    encode_parts, CountsWire, RecalcReceiptArtifact, RecalcReceiptArtifactError,
};
use antecedent_io::release_migration::{
    consume_previous_release_artifact, current_score_row_digest, narrow_score_row_digest,
    AdoptedRelease, MigrationError, PreviousReleaseError,
};

const ROW_IDS: [u64; 3] = [7, 11, 13];

fn digest(label: &str, text: &str) -> StageIdentity {
    StageIdentity::of(label, &[text.as_bytes()])
}

fn workflow(utility: &str, seed: &str) -> StageIdentities {
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
    ] {
        ids.set(stage, digest(&stage.label(), "v1"));
    }
    ids.set(Stage::LearnerFoldsRng, digest("learner", seed));
    ids.set(Stage::Utility, digest("utility", utility));
    for stage in [Stage::Identification, Stage::ScoreArtifact, Stage::Law, Stage::Decision] {
        ids.set(stage, digest(&stage.label(), "derived"));
    }
    ids
}

fn world(level: f64) -> InterventionWorld {
    let binding = WorldBinding::new(
        TargetPopulation::AllObserved,
        ObservationUnit::new("subject").unwrap(),
        OutcomeQuantity::mean(VariableId::from_raw(1), QuantityUnits::dimensionless()).unwrap(),
        Conditioning::unconditional(),
    )
    .unwrap();
    InterventionWorld::new(
        ScientificWorldId::from_raw(1),
        WorldAction::at_decision([Intervention::set(VariableId::from_raw(0), Value::f64(level))])
            .unwrap(),
        false,
        binding,
        TimeFrame::static_decision(),
    )
    .unwrap()
}

fn target(level: f64) -> ScientificTarget {
    ScientificTarget::new(
        TargetForm::World(world(level)),
        ModelAssumptions::new(GraphFamily::Dag, AssumptionSet::new()),
        InferenceSelection::Unspecified,
    )
}

fn counts(pairs: &[(Stage, CountsWire)]) -> BTreeMap<Stage, CountsWire> {
    pairs.iter().copied().collect()
}

fn decided() -> CountsWire {
    CountsWire { decisions: 1, ..CountsWire::default() }
}

fn numbers(bytes: &[u8]) -> String {
    bytes.iter().map(std::string::ToString::to_string).collect::<Vec<_>>().join(",")
}

fn artifact(score: &[u8; 32], target_digest: &[u8; 32], receipt: &[u8]) -> Vec<u8> {
    let rows = ROW_IDS.iter().map(std::string::ToString::to_string).collect::<Vec<_>>().join(",");
    format!(
        concat!(
            r#"{{"row_ids":[{}],"retained_score_digest":[{}],"#,
            r#""carried_target_digest":[{}],"receipt":[{}]}}"#,
        ),
        rows,
        numbers(score),
        numbers(target_digest),
        numbers(receipt),
    )
    .into_bytes()
}

fn status<'a>(adopted: &'a AdoptedRelease, stage: &str) -> &'a str {
    adopted.stage_status.iter().find(|(name, _)| name == stage).unwrap().1.as_str()
}

#[test]
fn a_previous_release_artifact_adopts_current_digests_and_refuses_fresh_reuse() {
    let declared = target(0.0);
    let current_target = *declared.identity().as_bytes();
    let other_target = *target(1.0).identity().as_bytes();
    let current_score = current_score_row_digest(&ROW_IDS);
    let narrow = narrow_score_row_digest(&[7, 11, 13]);
    assert_ne!(narrow, current_score);

    let stale =
        consume_previous_release_artifact(&artifact(&narrow, &current_target, &[]), &declared)
            .unwrap_err();
    assert_eq!(stale, PreviousReleaseError::Migration(MigrationError::ScoreRowDigestNotTranslated));

    let carried =
        consume_previous_release_artifact(&artifact(&current_score, &other_target, &[]), &declared)
            .unwrap_err();
    assert_eq!(
        carried,
        PreviousReleaseError::Migration(MigrationError::CarriedTargetDigestNotTranslated)
    );

    let in_process = RecalcReceiptArtifact::seal(
        &workflow("u1", "seed1"),
        &workflow("u2", "seed1"),
        &RecalcCapabilities::in_process(RetargetSupport::Licensed),
        &counts(&[(Stage::Decision, decided())]),
    )
    .unwrap();
    let adopted = consume_previous_release_artifact(
        &artifact(&current_score, &current_target, &in_process.to_bytes("in-process").unwrap()),
        &declared,
    )
    .unwrap();
    assert_eq!(adopted.score_row_digest, current_score);
    assert_eq!(adopted.target_digest, current_target);
    assert_eq!(adopted.boundary_name, "in_process");
    for stage in ["identification", "score_artifact", "law"] {
        assert!(status(&adopted, stage).starts_with("reused"), "{stage}");
    }
    assert!(status(&adopted, "decision").starts_with("recomputed"));

    let previous = workflow("u1", "seed1");
    let supplied = RecalcCapabilities::fresh_process(
        RetargetSupport::Licensed,
        ResumeContext { supplied_data: true, ..ResumeContext::default() },
    );
    let fresh = RecalcReceiptArtifact::seal(
        &previous,
        &previous,
        &supplied,
        &counts(&[
            (Stage::Identification, CountsWire { identifications: 1, ..CountsWire::default() }),
            (
                Stage::ScoreArtifact,
                CountsWire { fold_fits: 10, score_computations: 1, ..CountsWire::default() },
            ),
            (Stage::Law, CountsWire { reweights: 1, ..CountsWire::default() }),
            (Stage::Decision, decided()),
        ]),
    )
    .unwrap();
    let adopted = consume_previous_release_artifact(
        &artifact(&current_score, &current_target, &fresh.to_bytes("fresh-process").unwrap()),
        &declared,
    )
    .unwrap();
    assert_eq!(adopted.boundary_name, "fresh_process");
    for stage in ["identification", "score_artifact", "law", "decision"] {
        assert_eq!(status(&adopted, stage), "recomputed(fresh_process)");
    }

    let mut meta = fresh.meta().clone();
    let row = meta.entries.iter().position(|entry| entry.stage == "law").unwrap();
    meta.entries[row].tag = "reused".into();
    meta.entries[row].detail = "score_artifact".into();
    meta.entries[row].status = "reused(score_artifact)".into();
    meta.entries[row].counts = CountsWire::default();
    let tampered = encode_parts(&meta, "previous-release-receipt").unwrap();
    let reused = consume_previous_release_artifact(
        &artifact(&current_score, &current_target, &tampered),
        &declared,
    )
    .unwrap_err();
    assert_eq!(
        reused,
        PreviousReleaseError::FreshProcessReusedDerivedStage { stage: "law".into() }
    );

    let portable = RecalcCapabilities::fresh_process(
        RetargetSupport::Licensed,
        ResumeContext { portable_scores: true, supplied_data: true, ..ResumeContext::default() },
    );
    let error =
        RecalcReceiptArtifact::seal(&previous, &previous, &portable, &BTreeMap::new()).unwrap_err();
    assert!(matches!(
        error,
        RecalcReceiptArtifactError::FreshProcessReuse { stage } if stage == "score_artifact"
    ));
}
