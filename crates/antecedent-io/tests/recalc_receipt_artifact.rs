//! 2.3 C2: the portable selective-recalculation receipt.
//!
//! The consumer recomputes the plan from the stored previous and requested digests and the
//! stored capabilities and requires the stored table, counts, totals and identities to match;
//! every mutation below is resealed consistently (valid container, recomputed receipt
//! identity) so only the semantic checks can refuse it.

use std::collections::BTreeMap;

use antecedent_core::recalc::{
    Boundary, MissingDependency, RecalcCapabilities, RefusalReason, RequestSupport, ResumeContext,
    RetargetSupport, Stage, StageIdentities, StageIdentity,
};
use antecedent_io::recalc_receipt_artifact::{
    CountsWire, RECALC_RECEIPT_ARTIFACT_VERSION, RecalcReceiptArtifact, RecalcReceiptArtifactError,
    RecalcReceiptMeta, decode_parts, encode_parts, identities_from_wire, identities_to_wire,
    plan_from_wire, plan_to_wire,
};

const ID: &str = "recalc-receipt-test";

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

fn counts(pairs: &[(Stage, CountsWire)]) -> BTreeMap<Stage, CountsWire> {
    pairs.iter().copied().collect()
}

fn decided() -> CountsWire {
    CountsWire { decisions: 1, ..CountsWire::default() }
}

fn in_process() -> RecalcCapabilities {
    RecalcCapabilities::in_process(RetargetSupport::Licensed)
}

/// A utility-only change: the decision is recomputed, nothing else runs.
fn utility_only() -> RecalcReceiptArtifact {
    RecalcReceiptArtifact::seal(
        &workflow("u1", "seed1"),
        &workflow("u2", "seed1"),
        &in_process(),
        &counts(&[(Stage::Decision, decided())]),
    )
    .unwrap()
}

fn table(artifact: &RecalcReceiptArtifact) -> Vec<(String, String)> {
    artifact.meta().entries.iter().map(|e| (e.stage.clone(), e.status.clone())).collect()
}

fn oracle_identity(meta: &RecalcReceiptMeta) -> String {
    let mut parts: Vec<Vec<u8>> = Vec::new();
    for entry in &meta.entries {
        parts.push(entry.stage.clone().into_bytes());
        parts.push(entry.status.clone().into_bytes());
        parts.push(StageIdentity::from_hex(&entry.identity).unwrap().as_bytes().to_vec());
        let mut counted: Vec<u8> =
            entry.counts.as_array().iter().flat_map(|n| n.to_le_bytes()).collect();
        if entry.counts.model_fits > 0 {
            counted.extend_from_slice(&entry.counts.model_fits.to_le_bytes());
        }
        parts.push(counted);
    }
    let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    StageIdentity::of("recalc_receipt", &refs).to_hex()
}

/// Recompute the totals and the receipt identity so the container and the identity are
/// self-consistent after an edit.
fn reseal(mut meta: RecalcReceiptMeta) -> RecalcReceiptMeta {
    let mut totals = CountsWire::default();
    for entry in &meta.entries {
        totals.identifications += entry.counts.identifications;
        totals.fold_fits += entry.counts.fold_fits;
        totals.model_fits += entry.counts.model_fits;
        totals.score_computations += entry.counts.score_computations;
        totals.reweights += entry.counts.reweights;
        totals.decisions += entry.counts.decisions;
    }
    meta.totals = totals;
    meta.receipt_identity = oracle_identity(&meta);
    meta
}

fn consume(meta: &RecalcReceiptMeta) -> Result<RecalcReceiptArtifact, RecalcReceiptArtifactError> {
    RecalcReceiptArtifact::from_bytes(&encode_parts(meta, ID).unwrap(), None)
}

#[test]
fn c2_artifact_round_trip_preserves_table_counts_and_identities() {
    let artifact = utility_only();
    let reused = |dep: &str| format!("reused({dep})");
    let expected: Vec<(String, String)> = [
        ("graph", reused("graph")),
        ("query", reused("query")),
        ("regime", reused("regime")),
        ("evidence", reused("evidence")),
        ("source_population", reused("source_population")),
        ("target_population", reused("target_population")),
        ("data_snapshot", reused("data_snapshot")),
        ("row_design", reused("row_design")),
        ("treatment_grid", reused("treatment_grid")),
        ("learner_folds_rng", reused("learner_folds_rng")),
        ("utility", "recomputed(own:utility:modified)".to_owned()),
        ("identification", reused("graph")),
        ("score_artifact", reused("identification")),
        ("law", reused("score_artifact")),
        ("decision", "recomputed(upstream:utility<-utility:modified)".to_owned()),
    ]
    .into_iter()
    .map(|(stage, status)| (stage.to_owned(), status))
    .collect();
    assert_eq!(table(&artifact), expected);
    assert_eq!(artifact.meta().totals, decided());
    assert_eq!(artifact.meta().totals.total(), 1);
    assert!(artifact.plan().is_executable());
    assert_eq!(artifact.plan().recomputed_computations(), vec![Stage::Decision]);

    let bytes = artifact.to_bytes(ID).unwrap();
    let loaded =
        RecalcReceiptArtifact::from_bytes(&bytes, Some(artifact.receipt_identity())).unwrap();
    assert_eq!(loaded, artifact);
    assert_eq!(loaded.to_bytes(ID).unwrap(), bytes, "the container is canonical");
    assert_eq!(loaded.capabilities().unwrap(), in_process());
    assert_eq!(loaded.plan_identity(), artifact.plan().canonical_identity().to_hex());

    // The wire planner agrees with the sealed plan.
    let wire = plan_from_wire(
        &identities_to_wire(&workflow("u1", "seed1")),
        &identities_to_wire(&workflow("u2", "seed1")),
        &loaded.meta().capabilities,
    )
    .unwrap();
    assert_eq!(&wire, loaded.plan());
    assert_eq!(plan_to_wire(&wire).identity, loaded.plan_identity());
    let statuses: Vec<String> = plan_to_wire(&wire).entries.into_iter().map(|e| e.status).collect();
    assert_eq!(statuses, table(&loaded).into_iter().map(|(_, s)| s).collect::<Vec<_>>());
}

#[test]
fn c2_artifact_off_grid_route_and_a_changed_boundary_round_trip_in_the_capabilities() {
    let caps = RecalcCapabilities {
        retarget: RetargetSupport::NotDeclared,
        request: RequestSupport::OffGrid { licensed_route: Some("transport.smoothed_dose") },
        boundary: Boundary::InProcess,
    };
    let artifact = RecalcReceiptArtifact::seal(
        &workflow("u1", "seed1"),
        &workflow("u1", "seed1"),
        &caps,
        &BTreeMap::new(),
    )
    .unwrap();
    let plan = artifact.plan();
    assert_eq!(
        plan.first_refusal(),
        Some((
            Stage::TreatmentGrid,
            RefusalReason::OffGrid { licensed_route: Some("transport.smoothed_dose") }
        ))
    );
    let loaded = RecalcReceiptArtifact::from_bytes(&artifact.to_bytes(ID).unwrap(), None).unwrap();
    assert_eq!(loaded.capabilities().unwrap(), caps);
    // A changed capability declaration recomputes another plan and so refuses the old table.
    let mut meta = artifact.meta().clone();
    meta.capabilities.request = "on_grid".into();
    meta.capabilities.licensed_route = None;
    assert!(matches!(consume(&reseal(meta)), Err(RecalcReceiptArtifactError::PlanMismatch { .. })));
}

#[test]
fn c2_artifact_resealed_mutation_is_refused() {
    let artifact = utility_only();
    let base = artifact.meta().clone();
    let at = |stage: &str| base.entries.iter().position(|e| e.stage == stage).unwrap();

    // A reused stage that did work.
    let mut meta = base.clone();
    meta.entries[at("law")].counts.reweights = 1;
    assert!(matches!(
        consume(&reseal(meta)),
        Err(RecalcReceiptArtifactError::CountsInconsistent { stage }) if stage == "law"
    ));

    // A recomputed computation that did none of its work.
    let mut meta = base.clone();
    meta.entries[at("decision")].counts.decisions = 0;
    assert!(matches!(
        consume(&reseal(meta)),
        Err(RecalcReceiptArtifactError::CountsInconsistent { stage }) if stage == "decision"
    ));

    // Fit work counted against the decision, which does not own it.
    let mut meta = base.clone();
    meta.entries[at("decision")].counts.fold_fits = 10;
    assert!(matches!(
        consume(&reseal(meta)),
        Err(RecalcReceiptArtifactError::CountsInconsistent { .. })
    ));

    // A status rewritten to reuse the decision.
    let mut meta = base.clone();
    let row = at("decision");
    meta.entries[row].tag = "reused".into();
    meta.entries[row].detail = "law".into();
    meta.entries[row].status = "reused(law)".into();
    meta.entries[row].counts = CountsWire::default();
    assert!(matches!(
        consume(&reseal(meta)),
        Err(RecalcReceiptArtifactError::PlanMismatch { stage, field })
            if stage == "decision" && field == "status"
    ));

    // The determining dependency rewritten while the tag stays.
    let mut meta = base.clone();
    let row = at("decision");
    meta.entries[row].detail = "law".into();
    meta.entries[row].status = "recomputed(law)".into();
    assert!(matches!(
        consume(&reseal(meta)),
        Err(RecalcReceiptArtifactError::PlanMismatch { field: "status", .. })
    ));

    // A stage identity swapped for another stage's.
    let mut meta = base.clone();
    meta.entries[at("law")].identity = base.entries[at("graph")].identity.clone();
    assert!(matches!(
        consume(&reseal(meta)),
        Err(RecalcReceiptArtifactError::PlanMismatch { stage, field })
            if stage == "law" && field == "identity"
    ));

    // A previous digest edited: the recomputed plan no longer reuses the graph.
    let mut meta = base.clone();
    let row = meta.previous.iter().position(|d| d.stage == "graph").unwrap();
    meta.previous[row].own = digest("graph", "tampered").to_hex();
    assert!(matches!(
        consume(&reseal(meta)),
        Err(RecalcReceiptArtifactError::PlanMismatch { stage, field })
            if stage == "graph" && field == "status"
    ));

    // A dropped row.
    let mut meta = base.clone();
    meta.entries.remove(at("utility"));
    assert!(matches!(
        consume(&reseal(meta)),
        Err(RecalcReceiptArtifactError::PlanMismatch { stage, .. }) if stage == "*"
    ));

    // Stale totals, stale receipt identity and a swapped plan identity.
    let mut meta = base.clone();
    meta.totals.decisions = 5;
    assert!(matches!(consume(&meta), Err(RecalcReceiptArtifactError::TotalsMismatch)));
    let mut meta = base.clone();
    meta.receipt_identity = base.plan_identity.clone();
    assert!(matches!(
        consume(&meta),
        Err(RecalcReceiptArtifactError::IdentityMismatch { field: "receipt_identity" })
    ));
    let mut meta = base.clone();
    meta.plan_identity = base.receipt_identity.clone();
    assert!(matches!(
        consume(&meta),
        Err(RecalcReceiptArtifactError::IdentityMismatch { field: "plan_identity" })
    ));

    // Every refusal above is a registered, typed refusal with a namespaced detail.
    let error = consume(&{
        let mut meta = base.clone();
        meta.totals.decisions = 5;
        meta
    })
    .unwrap_err();
    let (code, detail, _) = error.refusal().unwrap();
    assert_eq!((code, detail), ("route_not_supported", "recalc_receipt.totals_mismatch"));

    // The untouched artifact still consumes.
    assert_eq!(consume(&base).unwrap().receipt_identity(), artifact.receipt_identity());
}

#[test]
fn c2_artifact_a_consistently_resealed_other_run_is_refused_against_a_retained_identity() {
    let first = utility_only();
    // Another run sealed consistently from scratch: new utility, different counts.
    let other = RecalcReceiptArtifact::seal(
        &workflow("u1", "seed1"),
        &workflow("u3", "seed1"),
        &in_process(),
        &counts(&[(Stage::Decision, CountsWire { decisions: 2, ..CountsWire::default() })]),
    )
    .unwrap();
    assert_ne!(other.receipt_identity(), first.receipt_identity());
    let bytes = other.to_bytes(ID).unwrap();
    assert!(RecalcReceiptArtifact::from_bytes(&bytes, None).is_ok());
    assert!(matches!(
        RecalcReceiptArtifact::from_bytes(&bytes, Some(first.receipt_identity())),
        Err(RecalcReceiptArtifactError::IdentityMismatch { field: "retained_receipt_identity" })
    ));
}

#[test]
fn c2_artifact_a_loaded_receipt_never_claims_reuse_in_a_fresh_process() {
    let previous = workflow("u1", "seed1");
    let requested = workflow("u1", "seed1");
    let first_run = [
        (Stage::Identification, CountsWire { identifications: 1, ..CountsWire::default() }),
        (
            Stage::ScoreArtifact,
            CountsWire { fold_fits: 10, score_computations: 1, ..CountsWire::default() },
        ),
        (Stage::Law, CountsWire { reweights: 1, ..CountsWire::default() }),
        (Stage::Decision, decided()),
    ];

    // Supplied data resumes by recomputing every derived stage, never by reuse.
    let supplied = RecalcCapabilities::fresh_process(
        RetargetSupport::Licensed,
        ResumeContext { supplied_data: true, ..ResumeContext::default() },
    );
    let resumed =
        RecalcReceiptArtifact::seal(&previous, &requested, &supplied, &counts(&first_run)).unwrap();
    for entry in resumed.meta().entries.iter().filter(|e| e.tag != "refused") {
        let stage = Stage::all().into_iter().find(|s| s.label() == entry.stage).unwrap();
        if !stage.is_input() {
            assert_eq!(entry.status, "recomputed(fresh_process)", "{}", entry.stage);
        }
    }
    let loaded = RecalcReceiptArtifact::from_bytes(&resumed.to_bytes(ID).unwrap(), None).unwrap();
    assert_eq!(loaded.meta().totals.fold_fits, 10);

    // A derived stage rewritten to reused, resealed consistently, is refused as a reuse claim
    // before anything else is compared.
    let mut meta = resumed.meta().clone();
    let row = meta.entries.iter().position(|e| e.stage == "law").unwrap();
    meta.entries[row].tag = "reused".into();
    meta.entries[row].detail = "score_artifact".into();
    meta.entries[row].status = "reused(score_artifact)".into();
    meta.entries[row].counts = CountsWire::default();
    let error = consume(&reseal(meta)).unwrap_err();
    assert!(
        matches!(&error, RecalcReceiptArtifactError::FreshProcessReuse { stage } if stage == "law")
    );
    let (code, detail, _) = error.refusal().unwrap();
    assert_eq!((code, detail), ("score_table_unavailable", "recalc_receipt.fresh_process_reuse"));

    // Even portable scores, for which the planner would reuse the score artifact, cannot be
    // sealed: a loaded receipt recreates no score table.
    let portable = RecalcCapabilities::fresh_process(
        RetargetSupport::Licensed,
        ResumeContext { portable_scores: true, supplied_data: true, ..ResumeContext::default() },
    );
    let error = RecalcReceiptArtifact::seal(&previous, &requested, &portable, &BTreeMap::new())
        .unwrap_err();
    assert!(
        matches!(&error, RecalcReceiptArtifactError::FreshProcessReuse { stage } if stage == "score_artifact"),
        "{error:?}"
    );

    // Nothing supplied: the specific unavailable results, not a reuse claim.
    let bare =
        RecalcCapabilities::fresh_process(RetargetSupport::Licensed, ResumeContext::default());
    let artifact = RecalcReceiptArtifact::seal(
        &previous,
        &requested,
        &bare,
        &counts(&[(
            Stage::Identification,
            CountsWire { identifications: 1, ..CountsWire::default() },
        )]),
    )
    .unwrap();
    assert_eq!(
        artifact.plan().first_refusal(),
        Some((
            Stage::DataSnapshot,
            RefusalReason::Unavailable { missing: MissingDependency::Data }
        ))
    );
    assert_eq!(
        artifact.meta().entries.iter().find(|e| e.stage == "score_artifact").unwrap().status,
        "refused(recalc.unavailable_fit)"
    );
    assert!(artifact.meta().entries.iter().all(|e| e.tag != "reused" || {
        Stage::all().into_iter().find(|s| s.label() == e.stage).unwrap().is_input()
    }));
}

#[test]
fn c2_artifact_identity_is_order_independent() {
    // Declarations built in opposite orders are the same workflow.
    let forward = workflow("u1", "seed1");
    let mut backward = StageIdentities::new();
    for row in identities_to_wire(&forward).into_iter().rev() {
        let stage = Stage::all().into_iter().find(|s| s.label() == row.stage).unwrap();
        backward.set(stage, StageIdentity::from_hex(&row.own).unwrap());
    }
    assert_eq!(forward, backward);

    let artifact = utility_only();
    let bytes = artifact.to_bytes(ID).unwrap();
    // The same receipt with entries and both declaration lists shuffled loads to the same
    // canonical receipt, identity and bytes.
    let mut meta = decode_parts(&bytes).unwrap();
    meta.entries.reverse();
    meta.entries.rotate_left(4);
    meta.previous.reverse();
    meta.requested.rotate_left(3);
    let shuffled =
        RecalcReceiptArtifact::from_bytes(&encode_parts(&meta, ID).unwrap(), None).unwrap();
    assert_eq!(shuffled.receipt_identity(), artifact.receipt_identity());
    assert_eq!(shuffled.to_bytes(ID).unwrap(), bytes);
    assert_eq!(identities_from_wire(&meta.previous).unwrap(), workflow("u1", "seed1"));

    // The identity covers the counts, the status and the stage identities.
    let heavier = RecalcReceiptArtifact::seal(
        &workflow("u1", "seed1"),
        &workflow("u2", "seed1"),
        &in_process(),
        &counts(&[(Stage::Decision, CountsWire { decisions: 2, ..CountsWire::default() })]),
    )
    .unwrap();
    assert_ne!(heavier.receipt_identity(), artifact.receipt_identity());
    assert_eq!(heavier.plan_identity(), artifact.plan_identity(), "same plan, other work");
}

#[test]
fn c2_artifact_is_bounded_and_versioned() {
    let artifact = utility_only();
    let bytes = artifact.to_bytes(ID).unwrap();
    assert!(artifact.to_bytes("  ").is_err());
    assert!(RecalcReceiptArtifact::from_bytes(&bytes[..bytes.len() / 2], None).is_err());
    assert!(matches!(
        RecalcReceiptArtifact::from_bytes(&vec![0_u8; 300 * 1024], None),
        Err(RecalcReceiptArtifactError::LimitsExceeded(_))
    ));

    let mut meta = artifact.meta().clone();
    meta.version = RECALC_RECEIPT_ARTIFACT_VERSION + 1;
    assert!(matches!(
        RecalcReceiptArtifact::from_bytes(&encode_parts(&meta, ID).unwrap(), None),
        Err(RecalcReceiptArtifactError::UnsupportedVersion { version: 2 })
    ));
    let mut meta = artifact.meta().clone();
    meta.feature = "recalc_receipt_v2".into();
    assert!(matches!(
        consume(&meta),
        Err(RecalcReceiptArtifactError::UnsupportedSemantics("feature marker"))
    ));
    let mut meta = artifact.meta().clone();
    meta.previous[0].stage = "no_such_stage".into();
    assert!(matches!(consume(&meta), Err(RecalcReceiptArtifactError::Malformed(_))));
    let mut meta = artifact.meta().clone();
    meta.previous[0].own = "xyz".into();
    assert!(matches!(consume(&meta), Err(RecalcReceiptArtifactError::Malformed(_))));
    let mut meta = artifact.meta().clone();
    meta.capabilities.boundary = "fresh_process".into();
    assert!(matches!(consume(&meta), Err(RecalcReceiptArtifactError::Malformed(_))));
}

#[test]
fn c2_artifact_model_fits_are_bound_and_legacy_counts_remain_unchanged() {
    let legacy = utility_only();
    let wire = serde_json::to_string(legacy.meta()).unwrap();
    assert!(!wire.contains("model_fits"));
    assert_eq!(legacy.receipt_identity(), oracle_identity(legacy.meta()));

    let artifact = RecalcReceiptArtifact::seal(
        &workflow("u1", "seed1"),
        &workflow("u1", "seed2"),
        &in_process(),
        &counts(&[
            (Stage::ScoreArtifact, CountsWire { model_fits: 1, ..CountsWire::default() }),
            (Stage::Law, CountsWire { reweights: 1, ..CountsWire::default() }),
            (Stage::Decision, decided()),
        ]),
    )
    .unwrap();
    assert_eq!(artifact.meta().totals.model_fits, 1);
    assert_eq!(artifact.meta().totals.total(), 3);
    assert_eq!(artifact.receipt_identity(), oracle_identity(artifact.meta()));
    let bytes = artifact.to_bytes(ID).unwrap();
    let loaded =
        RecalcReceiptArtifact::from_bytes(&bytes, Some(artifact.receipt_identity())).unwrap();
    assert_eq!(loaded, artifact);

    // A valid count for another execution cannot replace a retained receipt.
    let mut edited = artifact.meta().clone();
    let row = edited.entries.iter_mut().find(|r| r.stage == "score_artifact").unwrap();
    row.counts.model_fits = 2;
    let edited = reseal(edited);
    assert!(consume(&edited).is_ok());
    assert!(matches!(
        RecalcReceiptArtifact::from_bytes(
            &encode_parts(&edited, ID).unwrap(),
            Some(artifact.receipt_identity())
        ),
        Err(RecalcReceiptArtifactError::IdentityMismatch { field: "retained_receipt_identity" })
    ));

    // Fits assigned to a reused identification stage contradict actual work ownership.
    let mut edited = artifact.meta().clone();
    edited.entries.iter_mut().find(|r| r.stage == "identification").unwrap().counts.model_fits = 1;
    assert!(matches!(
        consume(&reseal(edited)),
        Err(RecalcReceiptArtifactError::CountsInconsistent { .. })
    ));
}
