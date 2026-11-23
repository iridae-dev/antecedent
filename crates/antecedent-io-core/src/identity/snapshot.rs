//! Observation and data-snapshot identity wire forms.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CausalSchema, Deserialize, DistributionFactorLawsWire, IDENTITY_FORMAT, IdentityDomain,
    IoError, PayloadDigestWire, SchemaWire, SemanticDigest, Serialize, digest_wire, schema_to_wire,
};

/// Observation-contract payload (schema only; not row contents).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ObservationIdentityWire {
    /// Identity format.
    pub format: u16,
    /// Schema.
    pub schema: SchemaWire,
    /// Observation tags with their variable arguments, sorted and deduplicated.
    pub observation: Vec<String>,
}

/// Observation-contract payload for digest and durable rehash.
///
/// Tags are a set: they are sorted and deduplicated. Each tag carries its
/// variable arguments (for example `independent_given:0,2`), so assumptions
/// that differ only in their conditioning set are distinct.
#[must_use]
pub fn observation_identity_wire(
    schema: &CausalSchema,
    observation: impl IntoIterator<Item = impl Into<String>>,
) -> ObservationIdentityWire {
    let mut observation: Vec<String> = observation.into_iter().map(Into::into).collect();
    observation.sort();
    observation.dedup();
    ObservationIdentityWire { format: IDENTITY_FORMAT, schema: schema_to_wire(schema), observation }
}

/// Ordered storage partition in a data snapshot.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DataPartitionIdentityWire {
    /// Digest from `OwnedColumnarStorage::content_digest` (storage encoding v1).
    pub content: [u8; 32],
    /// Row count for this partition.
    pub row_count: u64,
    /// Panel unit label; absent for tabular/series/environment partitions.
    pub unit_id: Option<u32>,
    /// Sampling regularity for this partition, when temporal.
    pub regularity: Option<String>,
}

/// Fixed interference network and realized assignment bound with the data.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct InterferenceSnapshotWire {
    /// Unit table content partition.
    pub units: DataPartitionIdentityWire,
    /// Edges `(from, to, weight bits)` in `(from, to)` order.
    pub edges: PayloadDigestWire,
    /// Realized binary assignment in unit-row order.
    pub assignment: PayloadDigestWire,
}

/// Participant labels bound to the rows of a data snapshot: the row count, a digest of the
/// distinct labels in first-seen order, and a digest of every row's label code. Labels that
/// differ in text or in row assignment give another digest.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParticipantBindingWire {
    /// Rows labelled.
    pub rows: u64,
    /// Digest (hex) of the distinct labels in first-seen order.
    pub labels: String,
    /// Digest (hex) of the per-row label codes.
    pub codes: String,
}

impl ParticipantBindingWire {
    /// The binding a table's declared unit key states.
    #[must_use]
    pub fn from_unit_key(key: &antecedent_data::UnitKey) -> Self {
        Self {
            rows: u64::try_from(key.rows()).unwrap_or(u64::MAX),
            labels: key.labels_digest().to_owned(),
            codes: key.codes_digest().to_owned(),
        }
    }

    /// Whether `key` is the unit assignment this binding declares.
    #[must_use]
    pub fn names(&self, key: &antecedent_data::UnitKey) -> bool {
        *self == Self::from_unit_key(key)
    }
}

/// A declared missing-data policy that differs from the route's default, with the covariates
/// of its observation model. Identifies how the analysed rows were chosen or weighted.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MissingPolicyWire {
    /// Policy name (`error`, `complete_case`, `inverse_probability`, `balanced_panel`,
    /// `available_case`).
    pub policy: String,
    /// Names of the observation-model covariates (inverse probability) or of the auxiliary
    /// variables of the imputation model (multiple imputation), in declared order.
    pub covariates: Vec<String>,
    /// The imputation specification of a multiple-imputation policy. Absent for every other
    /// policy, so their digests are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imputation: Option<ImputationSpecWire>,
}

#[allow(clippy::trivially_copy_pass_by_ref, reason = "serde skip predicates take a reference")]
const fn imputation_replicates_unset(replicates: &u32) -> bool {
    *replicates == 0
}

/// The declared inputs of a multiple-imputation analysis: each enters the data-snapshot
/// identity, so two analyses that differ in any of them are different claims.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImputationSpecWire {
    /// Imputation model (`chained_equations`, `joint_normal`).
    pub model: String,
    /// Number of completed tables.
    pub imputations: u32,
    /// Sweeps per completed table.
    pub iterations: u32,
    /// Seed of the keyed random streams.
    pub seed: u64,
    /// Declared missingness mechanism (`missing_at_random`).
    pub assumption: String,
    /// Sampling unit the parameter draws resample (`row`, `group`).
    pub unit: String,
    /// Bootstrap replicates run inside each completed table by an estimator with no analytic
    /// variance. Absent when zero, so earlier digests are unchanged.
    #[serde(default, skip_serializing_if = "imputation_replicates_unset")]
    pub bootstrap_replicates: u32,
    /// Digest (hex) of the model and design assumptions the imputation rests on: the model,
    /// tables, sweeps, seed, missingness assumption, resampling unit, declared sampling design,
    /// the neighbouring steps of a series and every variable's kind and imputed-cell count.
    /// Absent on older artifacts, and when the data cannot be imputed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assumptions_digest: Option<String>,
    /// Explicit model-column domains; absent for legacy inferred domains.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub domains: Vec<(String, String)>,
}

/// Data-snapshot payload binding contents and ordered unit/environment partitions.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DataSnapshotIdentityWire {
    /// Identity format.
    pub format: u16,
    /// Observation digest.
    pub observation: [u8; 32],
    /// `tabular`, `series`, or `panel`.
    pub modality: String,
    /// Sampling regularity label, when temporal.
    pub regularity: Option<String>,
    /// Row or time-step count.
    pub row_count: u64,
    /// Unit count for panels; `None` for tabular/series.
    pub unit_count: Option<u64>,
    /// Ordered partitions, including content, masks, weights, and temporal metadata.
    pub partitions: Vec<DataPartitionIdentityWire>,
    /// Interference network and assignment, when the study carries one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interference: Option<InterferenceSnapshotWire>,
    /// Portable finite-discrete provider tables for checked distribution or scalar
    /// functional replay. Bound to this data snapshot and absent on older artifacts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distribution_factor_laws: Option<DistributionFactorLawsWire>,
    /// Portable outcome-regression moments for the checked linear-Gaussian
    /// natural direct effect. Older artifacts omit them and remain readable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nested_counterfactual_fit: Option<NestedCounterfactualFitWire>,
    /// Portable moments for replaying checked static linear adjustment when its
    /// fit and uncertainty contract is inside the supported replay subset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear_fit_moments: Option<LinearFitMomentsWire>,
    /// Portable cross-product moments for replaying the point estimate of a checked
    /// instrumental-variable or front-door linear fit. Absent on older artifacts and on
    /// routes that do not replay from moments, whose digests are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cross_product_moments: Option<CrossProductMomentsWire>,
    /// Participant labels bound to the rows, when the study declares them. Absent on older
    /// artifacts and on studies without participants, whose digests are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub participants: Option<ParticipantBindingWire>,
    /// Missing-data policy, when the study declares one that differs from its route's default.
    /// Absent on older artifacts and on studies that keep the default, whose digests are
    /// unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub missing: Option<MissingPolicyWire>,
    /// Declared sampling design (`snake_case` name), when the study declares one. Absent on
    /// older artifacts and on studies that declare none, whose digests are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_design: Option<String>,
}

/// Sufficient statistics for replaying a checked homoskedastic OLS fit.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LinearFitMomentsWire {
    /// Wire format.
    pub format: u16,
    /// Number of complete-case rows.
    pub complete_case_rows: u64,
    /// Number of ordered design columns.
    pub columns: u32,
    /// Row-major `X'X`.
    pub gram: Vec<f64>,
    /// `X'Y` in design order.
    pub outcome_cross: Vec<f64>,
    /// `Y'Y`.
    pub outcome_square: f64,
}

/// Column marker of the constant column in [`CrossProductMomentsWire::columns`].
pub const CROSS_PRODUCT_INTERCEPT: u32 = u32::MAX;

/// Cross-product moments `V'V` of the complete-case columns a checked IV or front-door fit
/// reads, the outcome last.
///
/// Every column is a variable id, or [`CROSS_PRODUCT_INTERCEPT`] for the constant. IV carries
/// `[1, adjustment..., instrument..., treatment, outcome]`; front-door carries
/// `[1, treatment, mediators..., outcome]`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CrossProductMomentsWire {
    /// Wire format.
    pub format: u16,
    /// Number of complete-case rows.
    pub complete_case_rows: u64,
    /// Column identities, in order; the last column is the outcome.
    pub columns: Vec<u32>,
    /// Row-major `V'V`, `columns.len()` squared entries.
    pub gram: Vec<f64>,
}

/// Sufficient statistics for the ordered design `[intercept, treatment, mediator]`.
/// These are bound to the data snapshot, rather than the causal program.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NestedCounterfactualFitWire {
    /// Wire format.
    pub format: u16,
    /// Number of complete rows used by the outcome regression.
    pub complete_case_rows: u64,
    /// Row-major `X'X` for the three design columns.
    pub gram: [f64; 9],
    /// `X'Y` in design order.
    pub outcome_cross: [f64; 3],
}

/// Digest a data snapshot identity.
///
/// # Errors
///
/// CBOR encode failure.
pub fn data_snapshot_digest(wire: &DataSnapshotIdentityWire) -> Result<SemanticDigest, IoError> {
    digest_wire(IdentityDomain::DataSnapshot, wire)
}
