//! Portable frozen score table (`frozen_scores_v1`).
//!
//! A bounded, checksummed container holding the frozen cross-fitted (or cell) AIPW score
//! table of one run, so a fresh process can retarget it by row weights without the original
//! data and without refitting:
//!
//! * the score columns, held-out propensities, observed cells and outcomes, row ids and fold
//!   ids (plain data, bit-exact);
//! * the estimator identity (the nuisance provenance label), the fit identity (the effective
//!   identity of the score-artifact stage that produced the scores) and the data snapshot
//!   digest;
//! * the producing workflow's own-input stage digests, so a consumer can plan a requested
//!   workflow against them;
//! * a BLAKE3 identity over all of it, independent of the order the stage declarations were
//!   supplied in.
//!
//! A consumer trusts none of the stored table. It checks the format bounds and the table
//! shape, recomputes the fit identity from the stored stage digests, requires the stored
//! snapshot digest to be the declared one, and recomputes the artifact identity. A consumer
//! that retained the identity independently also refuses a consistently resealed artifact of
//! another run. The artifact decides nothing and computes nothing: it carries scores.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::recalc::{RetargetSupport, Stage, StageIdentities, StageIdentity};
use serde::{Deserialize, Serialize};

use crate::convert::to_cbor;
use crate::recalc_receipt_artifact::{DeclaredStageWire, identities_from_wire, identities_to_wire};
use crate::score_wire::{ScoreColumn, ScoreTableWire};

/// The artifact major version this reader writes and accepts.
pub const FROZEN_SCORES_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const FROZEN_SCORES_ARTIFACT_FEATURE: &str = "frozen_scores_v1";
/// Most bytes a frozen score artifact may occupy, enforced on export and on consumption.
pub const MAX_FROZEN_SCORES_ARTIFACT_BYTES: usize = 128 * 1024 * 1024;
/// Most complete-case rows a frozen score table may carry.
pub const MAX_FROZEN_SCORE_ROWS: usize = 1 << 21;
/// Most score columns a frozen score table may carry.
pub const MAX_FROZEN_SCORE_COLUMNS: usize = 64;
/// Most score values (rows times columns) a frozen score table may carry.
pub const MAX_FROZEN_SCORE_VALUES: usize = 1 << 22;
/// Most stage declarations the artifact may carry.
pub const MAX_FROZEN_SCORE_STAGES: usize = 64;

const ARTIFACT_KIND: &str = crate::artifact_kinds::FROZEN_SCORES;
const META_SECTION: &str = "frozen_scores_meta";
const MAX_TEXT_BYTES: usize = 256;
const MAX_ADJUSTMENT: usize = 1024;

crate::sectioned_artifact_enum! {
    /// Why a frozen score artifact was refused or could not be produced.
    #[derive(Clone, Debug, PartialEq, Eq)]
    FrozenScoresArtifactError {
        noun: "frozen scores",
        identity: "frozen_scores.identity_mismatch",
        variants {}
        display(f) {}
    }
}

impl FrozenScoresArtifactError {
    /// The registered refusal this error carries: `(code, detail, explanation)`.
    ///
    /// Present for a changed identity; absent for corruption, unsupported versions, malformed
    /// values, limits and encoding failures.
    #[must_use]
    pub fn refusal(&self) -> Option<(&'static str, &'static str, String)> {
        let text = self.to_string();
        let explanation =
            text.split_once(": ").map_or_else(|| text.clone(), |(_, rest)| rest.to_owned());
        match self {
            Self::IdentityMismatch { .. } => Some((
                antecedent_core::reason_code!("route_not_supported"),
                "frozen_scores.identity_mismatch",
                explanation,
            )),
            _ => None,
        }
    }
}

fn malformed(text: impl Into<String>) -> FrozenScoresArtifactError {
    FrozenScoresArtifactError::Malformed(text.into())
}

// -- wire types ---------------------------------------------------------------------------

/// One score column key on the wire.
pub type ScoreColumnPlain = ScoreColumn;

/// The score table as plain data (the shape of the estimator's `ScoreTableWire`).
pub type ScoreTablePlain = ScoreTableWire;

/// Stored metadata of a frozen score artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenScoresMeta {
    /// Metadata format version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// Estimator identity label (for example `cell.aipw.crossfit.multinomial_logit.ols.v1`).
    pub estimator: String,
    /// Effective identity (hex) of the score-artifact stage that produced the scores.
    pub fit_identity: String,
    /// Data snapshot own-input digest (hex) the scores were fitted on.
    pub snapshot_digest: String,
    /// Rows of the input data frame (the row-design count), at least the complete-case rows.
    pub input_rows: u64,
    /// Own-input digests of the producing workflow, in topological order.
    pub declared: Vec<DeclaredStageWire>,
    /// The frozen scores.
    pub table: ScoreTablePlain,
    /// BLAKE3 identity (hex) of everything above.
    pub identity: String,
}

// -- identity -----------------------------------------------------------------------------

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn u64_bytes(values: &[u64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f64_bytes(values: &[f64]) -> Vec<u8> {
    crate::artifact_codec::f64s_to_le_bytes(values)
}

fn opt_f64_bytes(value: Option<f64>) -> Vec<u8> {
    match value {
        None => vec![0],
        Some(v) => {
            let mut out = vec![1];
            out.extend_from_slice(&v.to_bits().to_le_bytes());
            out
        }
    }
}

fn columns_bytes(columns: &[ScoreColumnPlain]) -> Vec<u8> {
    columns
        .iter()
        .flat_map(|c| {
            let mut out = c.arm.to_le_bytes().to_vec();
            out.extend(opt_f64_bytes(c.threshold));
            out
        })
        .collect()
}

/// A stored stage identity read back from its hex spelling, refused with the artifact's own
/// error. This is a parser with a typed refusal (the encoder is `StageIdentity::to_hex` over
/// `antecedent_core::to_hex`), not a second hex encoder, so it stays here.
fn hex_identity(
    text: &str,
    what: &'static str,
) -> Result<StageIdentity, FrozenScoresArtifactError> {
    StageIdentity::from_hex(text).ok_or_else(|| malformed(format!("{what} is not 64 hex digits")))
}

/// BLAKE3 identity over the canonical content: the declarations enter in topological order,
/// so the order they were supplied in never matters.
fn compute_identity(
    meta: &FrozenScoresMeta,
    declared: &StageIdentities,
) -> Result<StageIdentity, FrozenScoresArtifactError> {
    let fit = hex_identity(&meta.fit_identity, "fit identity")?;
    let snapshot = hex_identity(&meta.snapshot_digest, "snapshot digest")?;
    let t = &meta.table;
    let mut parts: Vec<Vec<u8>> = vec![
        meta.feature.clone().into_bytes(),
        meta.estimator.clone().into_bytes(),
        fit.as_bytes().to_vec(),
        snapshot.as_bytes().to_vec(),
        meta.input_rows.to_le_bytes().to_vec(),
    ];
    for stage in Stage::all() {
        if declared.contains(stage) {
            parts.push(stage.label().into_bytes());
            parts.push(declared.own(stage).as_bytes().to_vec());
        }
    }
    parts.extend([
        u32_bytes(&t.observed_arm),
        f64_bytes(&t.propensities),
        f64_bytes(&t.observed_outcome),
        t.n_rows.to_le_bytes().to_vec(),
        u64_bytes(&t.row_index),
        u32_bytes(&t.fold_ids),
        t.n_folds.to_le_bytes().to_vec(),
        f64_bytes(&t.scores),
        columns_bytes(&t.columns),
        u32_bytes(&t.adjustment_set),
        t.nuisance_provenance.clone().into_bytes(),
        opt_f64_bytes(t.propensity_clip),
        t.treatment.to_le_bytes().to_vec(),
        u32_bytes(&t.intervened),
    ]);
    let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    Ok(StageIdentity::of("frozen_scores", &refs))
}

// -- validation ---------------------------------------------------------------------------

fn check_bounds(table: &ScoreTablePlain) -> Result<usize, FrozenScoresArtifactError> {
    let limit = |what| Err(FrozenScoresArtifactError::LimitsExceeded(what));
    let n = usize::try_from(table.n_rows).unwrap_or(usize::MAX);
    if n > MAX_FROZEN_SCORE_ROWS {
        return limit("score rows");
    }
    if table.columns.len() > MAX_FROZEN_SCORE_COLUMNS {
        return limit("score columns");
    }
    if n.saturating_mul(table.columns.len()) > MAX_FROZEN_SCORE_VALUES {
        return limit("score values");
    }
    if table.adjustment_set.len() > MAX_ADJUSTMENT || table.intervened.len() > MAX_ADJUSTMENT {
        return limit("adjustment variables");
    }
    if table.nuisance_provenance.len() > MAX_TEXT_BYTES {
        return limit("provenance bytes");
    }
    Ok(n)
}

fn check_shape(
    table: &ScoreTablePlain,
    n: usize,
    input_rows: u64,
) -> Result<(), FrozenScoresArtifactError> {
    let cols = table.columns.len();
    if cols == 0 || n == 0 {
        return Err(malformed("a frozen score table needs rows and columns"));
    }
    // The retarget gate reads the observed cells, the outcome and the held-out propensities,
    // so a table without them cannot be retargeted and is refused here.
    if table.observed_arm.len() != n
        || table.observed_outcome.len() != n
        || table.row_index.len() != n
        || table.fold_ids.len() != n
        || table.scores.len() != n.saturating_mul(cols)
        || table.propensities.len() != table.scores.len()
    {
        return Err(malformed("score table shape mismatch"));
    }
    if table.row_index.windows(2).any(|w| w[0] >= w[1]) {
        return Err(malformed("row ids must be strictly increasing"));
    }
    let last = table.row_index.last().copied().unwrap_or(0);
    if input_rows == 0 || last >= input_rows || n as u64 > input_rows {
        return Err(malformed("row ids must lie inside the input row count"));
    }
    if table.columns.iter().any(|c| c.threshold.is_some_and(|v| !v.is_finite())) {
        return Err(malformed("score column thresholds must be finite"));
    }
    Ok(())
}

fn check_identities(meta: &FrozenScoresMeta) -> Result<StageIdentities, FrozenScoresArtifactError> {
    if meta.declared.len() > MAX_FROZEN_SCORE_STAGES {
        return Err(FrozenScoresArtifactError::LimitsExceeded("stage declarations"));
    }
    if meta.estimator.is_empty() || meta.estimator.len() > MAX_TEXT_BYTES {
        return Err(malformed("estimator identity must be 1..=256 bytes"));
    }
    let declared = identities_from_wire(&meta.declared)
        .map_err(|e| malformed(format!("stage declarations: {e}")))?;
    for stage in
        [Stage::DataSnapshot, Stage::RowDesign, Stage::Identification, Stage::ScoreArtifact]
    {
        if !declared.contains(stage) {
            return Err(malformed(format!("declaration for stage {stage} is missing")));
        }
    }
    let snapshot = hex_identity(&meta.snapshot_digest, "snapshot digest")?;
    if snapshot != declared.own(Stage::DataSnapshot) {
        return Err(FrozenScoresArtifactError::IdentityMismatch { field: "snapshot_digest" });
    }
    let fit = hex_identity(&meta.fit_identity, "fit identity")?;
    let effective = declared.effective(RetargetSupport::Licensed);
    if effective.get(&Stage::ScoreArtifact) != Some(&fit) {
        return Err(FrozenScoresArtifactError::IdentityMismatch { field: "fit_identity" });
    }
    Ok(declared)
}

fn check(
    mut meta: FrozenScoresMeta,
    expected_identity: Option<&str>,
) -> Result<FrozenScoreTable, FrozenScoresArtifactError> {
    if meta.version != FROZEN_SCORES_ARTIFACT_VERSION {
        return Err(FrozenScoresArtifactError::UnsupportedVersion { version: meta.version });
    }
    if meta.feature != FROZEN_SCORES_ARTIFACT_FEATURE {
        return Err(FrozenScoresArtifactError::UnsupportedSemantics("feature marker"));
    }
    let n = check_bounds(&meta.table)?;
    check_shape(&meta.table, n, meta.input_rows)?;
    let declared = check_identities(&meta)?;
    // The estimator's decoder applies the same checks (finite scores, propensities in
    // [0, 1], fold ids below the fold count, a sane clip); the text keeps its
    // `invalid argument` rendering.
    meta.table.check().map_err(|e| malformed(format!("invalid argument: {e}")))?;
    let identity = compute_identity(&meta, &declared)?.to_hex();
    if meta.identity != identity {
        return Err(FrozenScoresArtifactError::IdentityMismatch { field: "identity" });
    }
    if expected_identity.is_some_and(|expected| expected != identity) {
        return Err(FrozenScoresArtifactError::IdentityMismatch { field: "retained_identity" });
    }
    meta.declared = identities_to_wire(&declared);
    Ok(FrozenScoreTable { meta })
}

// -- container ----------------------------------------------------------------------------

/// Encode a metadata section as a checksummed container.
///
/// Hidden: the producer path is [`FrozenScoreTable::to_bytes`]; tests use this to build
/// deliberately inconsistent or resealed artifacts.
///
/// # Errors
/// An empty id, an encoding failure or an oversized payload.
#[cfg(feature = "test-util")]
#[doc(hidden)]
pub fn encode_parts(
    meta: &FrozenScoresMeta,
    artifact_id: &str,
) -> Result<Vec<u8>, FrozenScoresArtifactError> {
    encode_parts_impl(meta, artifact_id)
}

fn encode_parts_impl(
    meta: &FrozenScoresMeta,
    artifact_id: &str,
) -> Result<Vec<u8>, FrozenScoresArtifactError> {
    if artifact_id.trim().is_empty() {
        return Err(FrozenScoresArtifactError::Encode("missing artifact id".into()));
    }
    let meta_bytes =
        to_cbor(meta).map_err(|error| FrozenScoresArtifactError::Encode(error.to_string()))?;
    crate::encode_container(
        &crate::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "frozen_scores",
            max_bytes: MAX_FROZEN_SCORES_ARTIFACT_BYTES,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        &[crate::OutSection {
            id: META_SECTION,
            content_type: crate::CONTENT_CBOR,
            bytes: &meta_bytes,
        }],
    )
    .map_err(|error| crate::map_artifact_check!(error, FrozenScoresArtifactError))
}

/// Decode the metadata section of a container, refusing another version before the metadata
/// is interpreted.
///
/// Hidden: see [`encode_parts`].
///
/// # Errors
/// Oversized, truncated, corrupt, differently laid out or other-version artifacts.
#[cfg(feature = "test-util")]
#[doc(hidden)]
pub fn decode_parts(bytes: &[u8]) -> Result<FrozenScoresMeta, FrozenScoresArtifactError> {
    decode_parts_impl(bytes)
}

fn decode_parts_impl(bytes: &[u8]) -> Result<FrozenScoresMeta, FrozenScoresArtifactError> {
    let parts = crate::decode_container(
        ARTIFACT_KIND,
        MAX_FROZEN_SCORES_ARTIFACT_BYTES,
        &[META_SECTION],
        bytes,
    )
    .map_err(|error| crate::map_artifact_check!(error, FrozenScoresArtifactError))?;
    let Ok([meta_bytes]) = <[Vec<u8>; 1]>::try_from(parts) else {
        return Err(FrozenScoresArtifactError::Malformed("unsupported container layout".into()));
    };
    crate::decode_with_version(&meta_bytes, |version| {
        if version == FROZEN_SCORES_ARTIFACT_VERSION {
            Ok(())
        } else {
            Err(FrozenScoresArtifactError::UnsupportedVersion { version })
        }
    })
}

// -- public type --------------------------------------------------------------------------

/// What a producer hands to [`FrozenScoreTable::seal`].
#[derive(Clone, Debug)]
pub struct FrozenScoreParts {
    /// Estimator identity label.
    pub estimator: String,
    /// Effective identity of the score-artifact stage under a licensed retarget.
    pub fit_identity: StageIdentity,
    /// Own-input digests of the producing workflow.
    pub declared: StageIdentities,
    /// Rows of the input data frame.
    pub input_rows: u64,
    /// The frozen scores.
    pub table: ScoreTableWire,
}

/// A validated frozen score table: scores, row ids, estimator and fit identity, snapshot
/// digest and the BLAKE3 identity of all of them.
#[derive(Clone, Debug, PartialEq)]
pub struct FrozenScoreTable {
    meta: FrozenScoresMeta,
}

impl FrozenScoreTable {
    /// Seal the scores of a run. The result passes the same checks a consumer applies.
    ///
    /// # Errors
    /// The consumer's refusals (shape, identities, bounds).
    pub fn seal(parts: FrozenScoreParts) -> Result<Self, FrozenScoresArtifactError> {
        let snapshot = parts.declared.own(Stage::DataSnapshot);
        let mut meta = FrozenScoresMeta {
            version: FROZEN_SCORES_ARTIFACT_VERSION,
            feature: FROZEN_SCORES_ARTIFACT_FEATURE.to_owned(),
            estimator: parts.estimator,
            fit_identity: parts.fit_identity.to_hex(),
            snapshot_digest: snapshot.to_hex(),
            input_rows: parts.input_rows,
            declared: identities_to_wire(&parts.declared),
            table: parts.table,
            identity: String::new(),
        };
        meta.identity = compute_identity(&meta, &parts.declared)?.to_hex();
        check(meta, None)
    }

    /// The stored metadata, with declarations in topological order.
    #[must_use]
    pub const fn meta(&self) -> &FrozenScoresMeta {
        &self.meta
    }

    /// BLAKE3 identity (hex), independent of the order declarations were supplied in.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.meta.identity
    }

    /// Estimator identity label.
    #[must_use]
    pub fn estimator(&self) -> &str {
        &self.meta.estimator
    }

    /// Fit identity: the effective identity of the producing score-artifact stage.
    ///
    /// # Errors
    /// Never for a sealed or consumed artifact.
    pub fn fit_identity(&self) -> Result<StageIdentity, FrozenScoresArtifactError> {
        hex_identity(&self.meta.fit_identity, "fit identity")
    }

    /// Data snapshot digest the scores were fitted on.
    ///
    /// # Errors
    /// Never for a sealed or consumed artifact.
    pub fn snapshot_digest(&self) -> Result<StageIdentity, FrozenScoresArtifactError> {
        hex_identity(&self.meta.snapshot_digest, "snapshot digest")
    }

    /// Rows of the input data frame the scores were built from.
    #[must_use]
    pub const fn input_rows(&self) -> u64 {
        self.meta.input_rows
    }

    /// Own-input digests of the producing workflow.
    ///
    /// # Errors
    /// Never for a sealed or consumed artifact.
    pub fn declared(&self) -> Result<StageIdentities, FrozenScoresArtifactError> {
        identities_from_wire(&self.meta.declared)
            .map_err(|e| malformed(format!("stage declarations: {e}")))
    }

    /// The scores in the estimator's wire form.
    #[must_use]
    pub fn table_wire(&self) -> ScoreTableWire {
        self.meta.table.clone()
    }

    /// Serialize through the checksummed sectioned container.
    ///
    /// # Errors
    /// An empty id, an encoding failure or an oversized payload.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, FrozenScoresArtifactError> {
        encode_parts_impl(&self.meta, artifact_id)
    }

    /// Consume an artifact by recomputation.
    ///
    /// `expected_identity` is an identity (hex) the consumer retained independently; when
    /// given, a consistently resealed artifact of another run is refused.
    ///
    /// # Errors
    /// Corruption, another major version, a bound exceeded, a malformed table or declaration,
    /// or an identity that does not match the recomputed one.
    pub fn from_bytes(
        bytes: &[u8],
        expected_identity: Option<&str>,
    ) -> Result<Self, FrozenScoresArtifactError> {
        check(decode_parts_impl(bytes)?, expected_identity)
    }

    /// Validate metadata as a consumer would, for hidden test reseals.
    ///
    /// # Errors
    /// The same refusals as [`Self::from_bytes`].
    #[cfg(feature = "test-util")]
    #[doc(hidden)]
    pub fn from_meta(
        meta: FrozenScoresMeta,
        expected_identity: Option<&str>,
    ) -> Result<Self, FrozenScoresArtifactError> {
        check(meta, expected_identity)
    }

    /// Recompute the identity of `meta` after an edit, for hidden test reseals.
    ///
    /// # Errors
    /// Malformed digests or declarations.
    #[cfg(feature = "test-util")]
    #[doc(hidden)]
    pub fn reseal_identity(meta: &mut FrozenScoresMeta) -> Result<(), FrozenScoresArtifactError> {
        let declared = identities_from_wire(&meta.declared)
            .map_err(|e| malformed(format!("stage declarations: {e}")))?;
        meta.identity = compute_identity(meta, &declared)?.to_hex();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::FrozenScoresArtifactError as Error;

    /// The texts the hand-written enum produced before it shared the generator.
    #[test]
    fn the_shared_variants_keep_their_historical_texts() {
        assert_eq!(
            Error::Decode("x".into()).to_string(),
            "frozen scores artifact does not decode: x"
        );
        assert_eq!(
            Error::UnsupportedVersion { version: 3 }.to_string(),
            "unsupported frozen scores artifact version 3"
        );
        assert_eq!(
            Error::UnsupportedSemantics("f").to_string(),
            "unsupported frozen scores semantics: f"
        );
        assert_eq!(
            Error::LimitsExceeded("rows").to_string(),
            "frozen scores consumer limit exceeded: rows"
        );
        assert_eq!(Error::Malformed("m".into()).to_string(), "malformed frozen scores artifact: m");
        assert_eq!(
            Error::IdentityMismatch { field: "identity" }.to_string(),
            "frozen_scores.identity_mismatch: identity changed"
        );
        assert_eq!(
            Error::Encode("e".into()).to_string(),
            "frozen scores artifact does not encode: e"
        );
    }
}
