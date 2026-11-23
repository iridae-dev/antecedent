//! Score-reuse and target-weight identity wire forms.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Deserialize, IDENTITY_FORMAT, IdentityDomain, IoError, PayloadDigestWire, SemanticDigest,
    Serialize, VariableId, digest_wire, vars_to_raw,
};

/// Score / batch reuse key. Stricter than identification: matching names and
/// shapes are not enough. Shared design is not shared nuisance fits.
///
/// Row-sized vectors enter as length-prefixed digests so the payload stays
/// small enough to travel in every contract section that advertises it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ScoreReuseIdentityWire {
    /// Identity format.
    pub format: u16,
    /// `score_table` or `batch_share`.
    pub kind: String,
    /// Identification digest when the key is query-specific.
    pub identification: Option<[u8; 32]>,
    /// Data snapshot the scores or folds were fit on.
    pub data_snapshot: [u8; 32],
    /// Complete-case row index, or the full-table index for a shared design.
    pub row_index: PayloadDigestWire,
    /// Fold assignment aligned with [`Self::row_index`].
    pub fold_ids: PayloadDigestWire,
    /// Fold count used to assign [`Self::fold_ids`].
    pub n_folds: u32,
    /// Certified adjustment set, writer order.
    pub adjustment_set: Vec<u32>,
    /// Nuisance provenance tag. Absent on a design-only share key.
    pub nuisance_provenance: Option<String>,
    /// Treatment variable. Absent on a design-only share key.
    pub treatment: Option<u32>,
    /// Additional intervened coordinates.
    pub intervened: Vec<u32>,
    /// Requested bootstrap / shared-draw count.
    pub bootstrap_replicates: u32,
    /// Inference binding the scores were produced under: the estimator configuration, GLM
    /// options, overlap policy and backend all change the cross-fitted scores while the
    /// provenance tag stays constant. Absent on a design-only share key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inference_binding: Option<[u8; 32]>,
}

/// Everything a query-specific score-table key is derived from.
#[derive(Debug, Clone, Copy)]
pub struct ScoreTableKey<'a> {
    /// Identification the scores answer.
    pub identification: SemanticDigest,
    /// Data snapshot the nuisances were fitted on.
    pub data_snapshot: SemanticDigest,
    /// Row index of the scored rows.
    pub row_index: &'a [u64],
    /// Cross-fitting fold of each row.
    pub fold_ids: &'a [u32],
    /// Number of folds.
    pub n_folds: u32,
    /// Adjustment covariates.
    pub adjustment_set: &'a [VariableId],
    /// Provenance tag of the cross-fitted nuisances.
    pub nuisance_provenance: &'a str,
    /// Treatment variable.
    pub treatment: VariableId,
    /// Additional intervened coordinates.
    pub intervened: &'a [VariableId],
    /// Requested bootstrap / shared-draw count.
    pub bootstrap_replicates: u32,
    /// Inference binding (estimator configuration, overlap policy, backend).
    pub inference_binding: SemanticDigest,
}

impl ScoreReuseIdentityWire {
    /// Query-specific score-table key.
    #[must_use]
    pub fn score_table(key: &ScoreTableKey<'_>) -> Self {
        let ScoreTableKey {
            identification,
            data_snapshot,
            row_index,
            fold_ids,
            n_folds,
            adjustment_set,
            nuisance_provenance,
            treatment,
            intervened,
            bootstrap_replicates,
            inference_binding,
        } = *key;
        Self {
            format: IDENTITY_FORMAT,
            kind: "score_table".into(),
            identification: Some(*identification.as_bytes()),
            data_snapshot: *data_snapshot.as_bytes(),
            row_index: PayloadDigestWire::u64s("score_reuse.row_index", row_index),
            fold_ids: PayloadDigestWire::u32s("score_reuse.fold_ids", fold_ids),
            n_folds,
            adjustment_set: vars_to_raw(adjustment_set),
            nuisance_provenance: Some(nuisance_provenance.into()),
            treatment: Some(treatment.raw()),
            intervened: vars_to_raw(intervened),
            bootstrap_replicates,
            inference_binding: Some(*inference_binding.as_bytes()),
        }
    }

    /// Shared-design key: folds and covariates, not per-query nuisances.
    #[must_use]
    pub fn batch_share(
        data_snapshot: SemanticDigest,
        fold_ids: &[u32],
        n_folds: u32,
        adjustment_set: &[VariableId],
    ) -> Self {
        let row_index: Vec<u32> =
            (0..fold_ids.len()).map(|index| u32::try_from(index).unwrap_or(u32::MAX)).collect();
        Self {
            format: IDENTITY_FORMAT,
            kind: "batch_share".into(),
            identification: None,
            data_snapshot: *data_snapshot.as_bytes(),
            row_index: PayloadDigestWire::u32s("score_reuse.row_index", &row_index),
            fold_ids: PayloadDigestWire::u32s("score_reuse.fold_ids", fold_ids),
            n_folds,
            adjustment_set: vars_to_raw(adjustment_set),
            nuisance_provenance: None,
            treatment: None,
            intervened: Vec::new(),
            bootstrap_replicates: 0,
            inference_binding: None,
        }
    }
}

/// Digest a score / batch reuse identity.
///
/// # Errors
///
/// CBOR encode failure.
pub fn score_reuse_digest(wire: &ScoreReuseIdentityWire) -> Result<SemanticDigest, IoError> {
    digest_wire(IdentityDomain::ScoreReuse, wire)
}

/// Reuse identity of an immutable fitted adjustment model. It is not a score table.
/// The legacy score-reuse digest slot can name this separately typed payload.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdjustedFitReuseIdentityWire {
    /// Format of this fitted-model identity (one).
    pub format: u16,
    /// `additive_linear_mean_v1` or `glm_response_mean_v1`.
    pub model: String,
    /// Estimator-specific covariance method; numerical options are in the inference binding.
    pub covariance: String,
    /// Frozen source snapshot.
    pub data_snapshot: [u8; 32],
    /// Certified identification and roles.
    pub identification: [u8; 32],
    /// Estimator, coding, support and uncertainty configuration.
    pub inference_binding: [u8; 32],
    /// Original source row positions, after complete-case selection.
    pub row_index: PayloadDigestWire,
    /// Digest of the actual coded column-major design, including intercept and treatment.
    pub design: PayloadDigestWire,
    /// Number of design columns.
    pub n_columns: u64,
    /// Certified adjustment roles in design order.
    pub adjustment_set: Vec<u32>,
    /// Frozen treatment role.
    pub treatment: u32,
    /// Frozen outcome role.
    pub outcome: u32,
}

/// Digest a fitted-model reuse identity in the common reuse identity slot.
/// Its distinct typed payload and model/version tags prevent a score table from aliasing it.
///
/// # Errors
/// Canonical encoding failure.
pub fn adjusted_fit_reuse_digest(
    wire: &AdjustedFitReuseIdentityWire,
) -> Result<SemanticDigest, IoError> {
    digest_wire(IdentityDomain::ScoreReuse, wire)
}

/// Row-weight identity bound to one data snapshot and one score table.
///
/// Its digest is the `RowWeights` population's `weights` field, so the target
/// identity and every claim over it name exactly this weighting.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TargetWeightsIdentityWire {
    /// Identity format.
    pub format: u16,
    /// The weight vector under the one data-sized payload convention
    /// ([`PayloadDigestWire::f64s`], kind `target_weights.rows`).
    pub weights: [u8; 32],
    /// Row count the weights cover (score-table rows).
    pub row_count: u64,
    /// Data snapshot digest.
    pub data_snapshot: [u8; 32],
    /// Score-reuse digest of the table the weights reweight.
    pub score_reuse: [u8; 32],
    /// Covariates the weights are declared to depend on.
    pub depends_on: Vec<u32>,
}

impl TargetWeightsIdentityWire {
    /// Bind `weights` to a snapshot and score table.
    #[must_use]
    pub fn new(
        weights: &[f64],
        data_snapshot: SemanticDigest,
        score_reuse: SemanticDigest,
        depends_on: &[VariableId],
    ) -> Self {
        let rows = PayloadDigestWire::f64s(ROW_WEIGHTS_PAYLOAD, weights);
        Self {
            format: IDENTITY_FORMAT,
            weights: rows.digest,
            row_count: rows.len,
            data_snapshot: *data_snapshot.as_bytes(),
            score_reuse: *score_reuse.as_bytes(),
            depends_on: vars_to_raw(depends_on),
        }
    }
}

/// Payload kind of the row-weight vector inside a `RowWeights` population and
/// the target-weights identity. One owner, so the population digest a query
/// carries and the identity that binds it agree by construction.
pub const ROW_WEIGHTS_PAYLOAD: &str = "target_weights.rows";

/// Digest target weights.
///
/// # Errors
///
/// CBOR encode failure.
pub fn target_weights_digest(wire: &TargetWeightsIdentityWire) -> Result<SemanticDigest, IoError> {
    digest_wire(IdentityDomain::TargetWeights, wire)
}
