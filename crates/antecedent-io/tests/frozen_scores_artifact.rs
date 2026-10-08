//! 2.3 C2: the portable frozen score table.
//!
//! The consumer checks the format bounds and the table shape, recomputes the fit identity from
//! the stored stage digests, requires the stored snapshot digest to be the declared one, and
//! recomputes the artifact identity. Edits below are resealed consistently (valid container,
//! recomputed identity) where noted, so only the semantic checks can refuse them.

use antecedent_core::recalc::{RetargetSupport, Stage, StageIdentities, StageIdentity};
use antecedent_estimate::{ScoreColumn, ScoreTableWire};
use antecedent_io::frozen_scores_artifact::{
    FROZEN_SCORES_ARTIFACT_VERSION, FrozenScoreParts, FrozenScoreTable, FrozenScoresArtifactError,
    MAX_FROZEN_SCORE_COLUMNS, MAX_FROZEN_SCORE_ROWS, decode_parts, encode_parts,
};
use antecedent_io::recalc_receipt_artifact::{identities_from_wire, identities_to_wire};

const ID: &str = "frozen-scores-test";
const ROWS: usize = 12;

fn digest(label: &str, text: &str) -> StageIdentity {
    StageIdentity::of(label, &[text.as_bytes()])
}

fn f(i: usize) -> f64 {
    f64::from(u32::try_from(i).unwrap())
}

fn declared(seed: &str) -> StageIdentities {
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
        Stage::Utility,
    ] {
        ids.set(stage, digest(&stage.label(), "v1"));
    }
    ids.set(Stage::LearnerFoldsRng, digest("learner", seed));
    for stage in [Stage::Identification, Stage::ScoreArtifact, Stage::Law, Stage::Decision] {
        ids.set(stage, digest(&stage.label(), "derived"));
    }
    ids
}

fn wire() -> ScoreTableWire {
    ScoreTableWire {
        observed_arm: (0..ROWS).map(|i| u32::try_from(i % 2).unwrap()).collect(),
        propensities: vec![0.5; 2 * ROWS],
        observed_outcome: (0..ROWS).map(|i| f(i) * 0.25).collect(),
        n_rows: ROWS as u64,
        row_index: (0..ROWS).map(|i| u32::try_from(2 * i).unwrap()).collect(),
        fold_ids: (0..ROWS).map(|i| u32::try_from(i % 2).unwrap()).collect(),
        n_folds: 2,
        scores: (0..2 * ROWS).map(|i| f(i).sin()).collect(),
        columns: vec![
            ScoreColumn { arm: 0, threshold: None },
            ScoreColumn { arm: 1, threshold: None },
        ],
        adjustment_set: vec![2],
        nuisance_provenance: "test.scores.v1".to_owned(),
        propensity_clip: Some(0.01),
        treatment: 0,
        intervened: Vec::new(),
    }
}

fn parts_with(declared: StageIdentities, table: ScoreTableWire) -> FrozenScoreParts {
    let fit =
        declared.effective(RetargetSupport::Licensed).get(&Stage::ScoreArtifact).copied().unwrap();
    FrozenScoreParts {
        estimator: "test.scores.v1".to_owned(),
        fit_identity: fit,
        declared,
        input_rows: 30,
        table,
    }
}

fn sealed() -> FrozenScoreTable {
    FrozenScoreTable::seal(parts_with(declared("seed-1"), wire())).unwrap()
}

#[test]
fn c2_scores_round_trip_is_bit_exact_and_carries_every_identity() {
    let artifact = sealed();
    let bytes = artifact.to_bytes(ID).unwrap();
    let again = FrozenScoreTable::from_bytes(&bytes, Some(artifact.identity())).unwrap();
    assert_eq!(again, artifact);
    assert_eq!(again.identity(), artifact.identity());
    assert_eq!(again.estimator(), "test.scores.v1");
    assert_eq!(again.input_rows(), 30);
    assert_eq!(again.snapshot_digest().unwrap(), declared("seed-1").own(Stage::DataSnapshot));
    let effective = declared("seed-1").effective(RetargetSupport::Licensed);
    assert_eq!(again.fit_identity().unwrap(), effective[&Stage::ScoreArtifact]);
    assert_eq!(again.declared().unwrap(), declared("seed-1"));

    let original = wire();
    let back = again.table_wire();
    assert_eq!(back.row_index, original.row_index, "row ids survive");
    assert_eq!(back.n_rows, original.n_rows);
    assert_eq!(back.columns, original.columns);
    for (a, b) in back.scores.iter().zip(&original.scores) {
        assert_eq!(a.to_bits(), b.to_bits(), "scores are bit exact");
    }
    for (a, b) in back.propensities.iter().zip(&original.propensities) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    let table = again.score_table().unwrap();
    assert_eq!(table.n_rows, ROWS);
    assert_eq!(table.n_columns(), 2);
    // The same bytes are produced for the same content.
    assert_eq!(artifact.to_bytes(ID).unwrap(), bytes);
}

#[test]
fn c2_scores_identity_is_independent_of_declaration_order() {
    let base = sealed();
    let mut reversed_wire = identities_to_wire(&declared("seed-1"));
    reversed_wire.reverse();
    let reversed = identities_from_wire(&reversed_wire).unwrap();
    let again = FrozenScoreTable::seal(parts_with(reversed, wire())).unwrap();
    assert_eq!(again.identity(), base.identity());

    // Declared in a different insertion order: the map holds the same set.
    let mut shuffled = StageIdentities::new();
    for stage in Stage::all().into_iter().rev() {
        if declared("seed-1").contains(stage) {
            shuffled.set(stage, declared("seed-1").own(stage));
        }
    }
    let third = FrozenScoreTable::seal(parts_with(shuffled, wire())).unwrap();
    assert_eq!(third.identity(), base.identity());

    // A stored declaration list in another order is accepted and canonicalized.
    let mut meta = base.meta().clone();
    meta.declared.reverse();
    let from_meta = FrozenScoreTable::from_meta(meta, Some(base.identity())).unwrap();
    assert_eq!(from_meta.identity(), base.identity());
    assert_eq!(from_meta.meta().declared, base.meta().declared);

    // Content does change it: another seed, another scores.
    let other_seed = FrozenScoreTable::seal(parts_with(declared("seed-2"), wire()))
        .unwrap()
        .identity()
        .to_owned();
    assert_ne!(other_seed, base.identity());
    let mut other = wire();
    other.scores[0] += 1e-9;
    let other_scores = FrozenScoreTable::seal(parts_with(declared("seed-1"), other)).unwrap();
    assert_ne!(other_scores.identity(), base.identity());
}

#[test]
fn c2_scores_resealed_mutation_is_refused_by_a_retained_identity() {
    let artifact = sealed();
    let mut meta = artifact.meta().clone();
    meta.table.scores[3] += 0.5;

    // Stale identity: the edit alone is caught by recomputation.
    let stale = FrozenScoreTable::from_meta(meta.clone(), None).unwrap_err();
    assert_eq!(stale, FrozenScoresArtifactError::IdentityMismatch { field: "identity" });
    let (code, detail, _) = stale.refusal().unwrap();
    assert_eq!(detail, "frozen_scores.identity_mismatch");
    assert_eq!(code, antecedent_core::reason_code!("route_not_supported"));

    // Resealed consistently: valid on its own, refused by the identity the consumer kept.
    FrozenScoreTable::reseal_identity(&mut meta).unwrap();
    let bytes = encode_parts(&meta, ID).unwrap();
    assert!(FrozenScoreTable::from_bytes(&bytes, None).is_ok(), "a consistent reseal is valid");
    let error = FrozenScoreTable::from_bytes(&bytes, Some(artifact.identity())).unwrap_err();
    assert_eq!(error, FrozenScoresArtifactError::IdentityMismatch { field: "retained_identity" });
}

#[test]
fn c2_scores_stage_and_snapshot_bindings_are_checked() {
    let artifact = sealed();

    let mut snapshot = artifact.meta().clone();
    snapshot.snapshot_digest = digest("data_snapshot", "other").to_hex();
    FrozenScoreTable::reseal_identity(&mut snapshot).unwrap();
    assert_eq!(
        FrozenScoreTable::from_meta(snapshot, None).unwrap_err(),
        FrozenScoresArtifactError::IdentityMismatch { field: "snapshot_digest" }
    );

    let mut fit = artifact.meta().clone();
    fit.fit_identity = digest("score_artifact", "other").to_hex();
    FrozenScoreTable::reseal_identity(&mut fit).unwrap();
    assert_eq!(
        FrozenScoreTable::from_meta(fit, None).unwrap_err(),
        FrozenScoresArtifactError::IdentityMismatch { field: "fit_identity" }
    );

    // A changed declared stage moves the fit identity it must match.
    let mut learner = artifact.meta().clone();
    for row in &mut learner.declared {
        if row.stage == Stage::LearnerFoldsRng.label() {
            row.own = digest("learner", "tampered").to_hex();
        }
    }
    FrozenScoreTable::reseal_identity(&mut learner).unwrap();
    assert_eq!(
        FrozenScoreTable::from_meta(learner, None).unwrap_err(),
        FrozenScoresArtifactError::IdentityMismatch { field: "fit_identity" }
    );

    // The producing workflow must declare the stages a resume plans against.
    let mut missing = artifact.meta().clone();
    missing.declared.retain(|row| row.stage != Stage::RowDesign.label());
    FrozenScoreTable::reseal_identity(&mut missing).unwrap();
    assert!(matches!(
        FrozenScoreTable::from_meta(missing, None).unwrap_err(),
        FrozenScoresArtifactError::Malformed(_)
    ));
}

#[test]
fn c2_scores_malformed_tables_are_refused_even_when_resealed() {
    let artifact = sealed();
    let reseal = |edit: &dyn Fn(&mut antecedent_io::frozen_scores_artifact::FrozenScoresMeta)| {
        let mut meta = artifact.meta().clone();
        edit(&mut meta);
        FrozenScoreTable::reseal_identity(&mut meta).unwrap();
        FrozenScoreTable::from_meta(meta, None).unwrap_err()
    };
    let malformed = |error: FrozenScoresArtifactError| {
        assert!(matches!(error, FrozenScoresArtifactError::Malformed(_)), "{error:?}");
    };
    malformed(reseal(&|m| m.table.scores[0] = f64::NAN));
    malformed(reseal(&|m| m.table.propensities[0] = 1.5));
    malformed(reseal(&|m| m.table.row_index[3] = m.table.row_index[2]));
    malformed(reseal(&|m| m.table.row_index[ROWS - 1] = 30));
    malformed(reseal(&|m| m.table.fold_ids[0] = 7));
    malformed(reseal(&|m| {
        m.table.scores.pop();
    }));
    malformed(reseal(&|m| m.table.observed_arm.clear()));
    malformed(reseal(&|m| m.table.observed_outcome.clear()));
    malformed(reseal(&|m| m.table.propensity_clip = Some(0.9)));
    malformed(reseal(&|m| m.table.columns[0].threshold = Some(f64::INFINITY)));
    malformed(reseal(&|m| m.estimator.clear()));
    // A digest that is not hex cannot even be resealed; the consumer refuses it as stored.
    let mut not_hex = artifact.meta().clone();
    not_hex.fit_identity = "not-hex".to_owned();
    malformed(FrozenScoreTable::from_meta(not_hex, None).unwrap_err());
}

#[test]
fn c2_scores_decode_is_bounded_and_versioned() {
    let artifact = sealed();

    let mut rows = artifact.meta().clone();
    rows.table.n_rows = (MAX_FROZEN_SCORE_ROWS + 1) as u64;
    assert_eq!(
        FrozenScoreTable::from_meta(rows, None).unwrap_err(),
        FrozenScoresArtifactError::LimitsExceeded("score rows")
    );

    let mut columns = artifact.meta().clone();
    columns.table.columns = (0..=MAX_FROZEN_SCORE_COLUMNS)
        .map(|i| antecedent_io::frozen_scores_artifact::ScoreColumnPlain {
            arm: u32::try_from(i).unwrap(),
            threshold: None,
        })
        .collect();
    assert_eq!(
        FrozenScoreTable::from_meta(columns, None).unwrap_err(),
        FrozenScoresArtifactError::LimitsExceeded("score columns")
    );

    let mut stages = artifact.meta().clone();
    let row = stages.declared[0].clone();
    for _ in 0..70 {
        stages.declared.push(row.clone());
    }
    assert_eq!(
        FrozenScoreTable::from_meta(stages, None).unwrap_err(),
        FrozenScoresArtifactError::LimitsExceeded("stage declarations")
    );

    let mut version = artifact.meta().clone();
    version.version = FROZEN_SCORES_ARTIFACT_VERSION + 1;
    let bytes = encode_parts(&version, ID).unwrap();
    assert_eq!(
        FrozenScoreTable::from_bytes(&bytes, None).unwrap_err(),
        FrozenScoresArtifactError::UnsupportedVersion { version: version.version }
    );

    let mut feature = artifact.meta().clone();
    feature.feature = "frozen_scores_v2".to_owned();
    assert_eq!(
        FrozenScoreTable::from_meta(feature, None).unwrap_err(),
        FrozenScoresArtifactError::UnsupportedSemantics("feature marker")
    );
}

#[test]
fn c2_scores_corrupt_or_truncated_bytes_are_refused() {
    let artifact = sealed();
    let bytes = artifact.to_bytes(ID).unwrap();
    assert!(FrozenScoreTable::from_bytes(&bytes[..bytes.len() / 2], None).is_err());
    assert!(FrozenScoreTable::from_bytes(&[], None).is_err());
    assert!(FrozenScoreTable::from_bytes(b"not an artifact at all", None).is_err());
    let mut flipped = bytes.clone();
    let middle = flipped.len() / 2;
    flipped[middle] ^= 0xFF;
    assert!(FrozenScoreTable::from_bytes(&flipped, None).is_err());
    assert!(encode_parts(artifact.meta(), "  ").is_err(), "an artifact needs an id");
    // The decoded metadata is exactly what was stored.
    assert_eq!(&decode_parts(&bytes).unwrap(), artifact.meta());
}
