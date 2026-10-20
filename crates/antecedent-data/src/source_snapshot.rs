//! Immutable scientific source snapshot.
//!
//! A snapshot binds schema, modality, cohort and row order, category domains,
//! validity policy, weights, units, time index, and split lineage. Scientific
//! equality is [`SourceIdentity`]: the same full-content digest, schema, modality,
//! order, split lineage, and validity policy.
//!
//! The content digest is the existing storage digest of every original column,
//! validity bitmap, analysis mask, and weight vector. A column projection does not
//! reuse that digest. A chunk is a row range into the same source, not a new population.
//!
//! Stored `f64` columns are invalid exactly when the value is `NaN`.
//! [`ValidityPolicy::NullDistinctFromNan`] can be named, and constructing a snapshot
//! with it fails: null-aware floats are not adopted.
//!
//! Owned, borrowed, and chunked snapshots only hand out shared references.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

// Constructors and views are the snapshot contract. Later preparation call
// sites use them; until then only the unit tests do, which the non-test build
// does not count.
#![allow(dead_code)]

use std::sync::Arc;

use antecedent_core::{CausalSchema, VariableId};

use crate::categorical::CategoryDomain;
use crate::column::ColumnView;
use crate::dataset::{TabularData, TimeSeriesData};
use crate::error::DataError;
use crate::multi_env::MultiEnvironmentData;
use crate::panel::PanelData;
use crate::reference::ReferencePointPolicy;
use crate::split::{DiscoveryEstimationSplit, EnvHoldoutSplit, TimeRange};
use crate::storage::OwnedColumnarStorage;
use crate::table::TableView;
use crate::temporal::TimeIndex;

/// How missing cells are declared for a source.
///
/// The adopted rule matches [`crate::Float64Column`]: a row is invalid exactly when its
/// value is `NaN`. The null-distinct rule is named so a caller can ask for it; snapshots
/// refuse it instead of treating today's columns as null-aware.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub(crate) enum ValidityPolicy {
    /// Adopted rule: an `f64` row is invalid exactly when its value is `NaN`.
    #[default]
    InvalidIffNan,
    /// A null and an observed `NaN` would be distinct. Columnar storage does not implement this.
    NullDistinctFromNan,
}

/// Scientific container the rows came from.
///
/// Time indexes and panel or environment partitions live here, so two sources with the
/// same storage bytes and different sampling or unit identity stay distinct.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SourceModality {
    /// One static table ([`TabularData`] or bare [`OwnedColumnarStorage`]).
    StaticTable,
    /// One time series, including its [`TimeIndex`].
    TimeSeries {
        /// Sampling metadata. Its length matches the series.
        time_index: TimeIndex,
    },
    /// Panel units in declared order. Rows are not one shared axis.
    Panel {
        /// One entry per unit, in [`PanelData`] order.
        units: Arc<[PanelUnitMeta]>,
    },
    /// Environments in declared order. Rows are not one shared axis.
    MultiEnvironment {
        /// One time index per environment, in container order.
        time_indexes: Arc<[TimeIndex]>,
    },
}

/// Unit id and time index copied from a [`crate::PanelUnit`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PanelUnitMeta {
    unit_id: u32,
    time_index: TimeIndex,
}

impl PanelUnitMeta {
    /// Panel unit id.
    #[must_use]
    pub(crate) const fn unit_id(&self) -> u32 {
        self.unit_id
    }

    /// That unit's time index.
    #[must_use]
    pub(crate) const fn time_index(&self) -> &TimeIndex {
        &self.time_index
    }
}

/// Declared cohort label, row order, and reference-point alignment.
///
/// Order is not folded into the storage digest. The digest stays in storage order;
/// a different cohort or permutation is a different [`SourceIdentity`] with the same bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CohortOrder {
    label: Arc<str>,
    rows: RowOrder,
    reference: ReferencePointPolicy,
}

/// Where each scientific position sits in storage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum RowOrder {
    /// Storage row `i` is scientific position `i`.
    AsStored,
    /// `positions[i]` is the storage row at scientific position `i`.
    ///
    /// The slice is a permutation of `0..row_count`.
    Permuted(Arc<[u32]>),
}

impl CohortOrder {
    /// Cohort `label` in storage order, with [`ReferencePointPolicy::SeriesOrigin`].
    #[must_use]
    pub(crate) fn as_stored(label: impl Into<Arc<str>>) -> Self {
        Self {
            label: label.into(),
            rows: RowOrder::AsStored,
            reference: ReferencePointPolicy::SeriesOrigin,
        }
    }

    /// Cohort `label` with an explicit storage-row permutation.
    ///
    /// The permutation is checked when the snapshot is built.
    #[must_use]
    pub(crate) fn permuted(label: impl Into<Arc<str>>, positions: impl Into<Arc<[u32]>>) -> Self {
        Self {
            label: label.into(),
            rows: RowOrder::Permuted(positions.into()),
            reference: ReferencePointPolicy::SeriesOrigin,
        }
    }

    /// Replace the reference-point policy.
    #[must_use]
    pub(crate) const fn with_reference(mut self, reference: ReferencePointPolicy) -> Self {
        self.reference = reference;
        self
    }

    /// Cohort label.
    #[must_use]
    pub(crate) fn label(&self) -> &str {
        &self.label
    }

    /// Storage-row order.
    #[must_use]
    pub(crate) const fn rows(&self) -> &RowOrder {
        &self.rows
    }

    /// Reference-point alignment.
    #[must_use]
    pub(crate) const fn reference(&self) -> ReferencePointPolicy {
        self.reference
    }
}

/// Discovery / estimation split recorded with the source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SplitLineage {
    /// No split has been declared.
    Unsplit,
    /// Temporal discovery and estimation windows over one row axis.
    DiscoveryEstimation(DiscoveryEstimationSplit),
    /// Environment holdout over a panel or multi-environment source.
    EnvironmentHoldout(EnvHoldoutSplit),
}

/// Cohort, split, and validity policy supplied by the caller.
///
/// Modality is taken from the dataset, not from this record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SourceContract {
    cohort: CohortOrder,
    split: SplitLineage,
    validity: ValidityPolicy,
}

impl SourceContract {
    /// Contract with the adopted [`ValidityPolicy::InvalidIffNan`].
    #[must_use]
    pub(crate) const fn new(cohort: CohortOrder, split: SplitLineage) -> Self {
        Self { cohort, split, validity: ValidityPolicy::InvalidIffNan }
    }

    /// Replace the validity policy. [`ValidityPolicy::NullDistinctFromNan`] is refused at build.
    #[must_use]
    pub(crate) const fn with_validity(mut self, validity: ValidityPolicy) -> Self {
        self.validity = validity;
        self
    }
}

/// Full-source content digest.
///
/// [`Self::Table`] is [`OwnedColumnarStorage::content_digest`]: original columns, validity,
/// analysis mask, and weights. [`Self::Partitioned`] stores each member's full storage
/// digest in container order. Neither variant is a hash of a column subset.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum FullContentDigest {
    /// One table or series.
    Table([u8; 32]),
    /// Panel units or environments, each entry a full storage digest.
    Partitioned(Arc<[[u8; 32]]>),
}

/// Equality key of a source snapshot.
///
/// Chunks of one source share this value; a chunk's own key is [`ChunkIdentity`], which
/// adds the row range. A projected column subset does not have one.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SourceIdentity {
    content: FullContentDigest,
    schema: CausalSchema,
    modality: SourceModality,
    cohort: CohortOrder,
    split: SplitLineage,
    validity: ValidityPolicy,
}

/// Equality key of a row range of a source: the parent's [`SourceIdentity`] and the
/// absolute storage-row range. Two ranges of one source are different keys, and an
/// empty range at one row differs from an empty range at another.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ChunkIdentity {
    source: SourceIdentity,
    rows: TimeRange,
}

impl ChunkIdentity {
    /// Identity of the parent source. The chunk is not a new population.
    #[must_use]
    pub(crate) const fn source(&self) -> &SourceIdentity {
        &self.source
    }

    /// Absolute storage-row range.
    #[must_use]
    pub(crate) const fn row_range(&self) -> TimeRange {
        self.rows
    }
}

impl SourceIdentity {
    /// Full-source digest. This is not a projected-column digest.
    #[must_use]
    pub(crate) const fn content(&self) -> &FullContentDigest {
        &self.content
    }

    /// Schema, including variable units.
    #[must_use]
    pub(crate) const fn schema(&self) -> &CausalSchema {
        &self.schema
    }

    /// Modality, including time indexes and unit or environment partitions.
    #[must_use]
    pub(crate) const fn modality(&self) -> &SourceModality {
        &self.modality
    }

    /// Cohort label, row order, and reference point.
    #[must_use]
    pub(crate) const fn cohort(&self) -> &CohortOrder {
        &self.cohort
    }

    /// Split lineage.
    #[must_use]
    pub(crate) const fn split(&self) -> &SplitLineage {
        &self.split
    }

    /// Validity policy.
    #[must_use]
    pub(crate) const fn validity(&self) -> ValidityPolicy {
        self.validity
    }
}

/// Column ids named by a projection.
///
/// This is not a content digest. The projected-content contract is later work; this type
/// exists so a column subset cannot be passed off as the full source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProjectedIdentity {
    columns: Arc<[VariableId]>,
}

impl ProjectedIdentity {
    /// Requested variables, in request order.
    #[must_use]
    pub(crate) fn columns(&self) -> &[VariableId] {
        &self.columns
    }
}

/// A column subset of a source. It does not carry [`FullContentDigest`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProjectedSource {
    columns: Arc<[VariableId]>,
}

impl ProjectedSource {
    /// Requested variables, in request order.
    #[must_use]
    pub(crate) fn columns(&self) -> &[VariableId] {
        &self.columns
    }

    /// Identity of the request. It has no storage digest.
    #[must_use]
    pub(crate) fn identity(&self) -> ProjectedIdentity {
        ProjectedIdentity { columns: Arc::clone(&self.columns) }
    }

    /// Refuse the full-source digest.
    ///
    /// A projected column list is not the source, even when it names every column:
    /// masks, weights, and the unprojected columns are part of the storage digest.
    ///
    /// # Errors
    ///
    /// Always. The error is [`DataError::InvalidArgument`].
    pub(crate) fn full_content_digest(&self) -> Result<FullContentDigest, DataError> {
        let n = self.columns.len();
        Err(DataError::InvalidArgument {
            message: format!(
                "a projected column subset of {n} variables is not the full source and cannot reuse its content digest"
            ),
        })
    }
}

/// Where observation weights live.
#[derive(Clone, Debug)]
pub(crate) enum WeightView<'a> {
    /// One row axis. `None` means the source declared no weights.
    Rows(Option<&'a [f64]>),
    /// One optional weight vector per panel unit or environment, in container order.
    Partitioned(Vec<Option<&'a [f64]>>),
}

/// Owned immutable source.
#[derive(Clone, Debug)]
pub(crate) struct OwnedSourceSnapshot {
    body: OwnedBody,
    modality: SourceModality,
    contract: SourceContract,
    content: FullContentDigest,
}

/// Shared view of an [`OwnedSourceSnapshot`].
///
/// The view holds shared references only, so it cannot mutate the frozen source.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BorrowedSourceSnapshot<'a> {
    body: &'a OwnedBody,
    modality: &'a SourceModality,
    contract: &'a SourceContract,
    content: &'a FullContentDigest,
}

/// A row range into an existing table or series.
///
/// Empty ranges and ranges that end on the last row are representable. The range does
/// not allocate columns and does not replace [`SourceIdentity`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct ChunkedSourceSnapshot<'a> {
    body: &'a OwnedBody,
    modality: &'a SourceModality,
    contract: &'a SourceContract,
    content: &'a FullContentDigest,
    rows: TimeRange,
}

#[derive(Clone, Debug)]
enum OwnedBody {
    Table(OwnedColumnarStorage),
    TimeSeries(TimeSeriesData),
    Panel(PanelData),
    MultiEnvironment(MultiEnvironmentData),
}

impl OwnedSourceSnapshot {
    /// Static table from [`TabularData`].
    ///
    /// # Errors
    ///
    /// Unadopted validity policy, an order that is not a permutation of the rows,
    /// a reference point outside the table, or a split that does not belong to one table.
    pub(crate) fn from_tabular(
        data: &TabularData,
        contract: SourceContract,
    ) -> Result<Self, DataError> {
        Self::try_new(OwnedBody::Table(data.storage().clone()), contract)
    }

    /// Static table from owned storage.
    ///
    /// # Errors
    ///
    /// See [`Self::from_tabular`].
    pub(crate) fn from_storage(
        storage: OwnedColumnarStorage,
        contract: SourceContract,
    ) -> Result<Self, DataError> {
        Self::try_new(OwnedBody::Table(storage), contract)
    }

    /// Time series, including its time index.
    ///
    /// # Errors
    ///
    /// See [`Self::from_tabular`]. The split's series length must match the series.
    pub(crate) fn from_time_series(
        data: &TimeSeriesData,
        contract: SourceContract,
    ) -> Result<Self, DataError> {
        Self::try_new(OwnedBody::TimeSeries(data.clone()), contract)
    }

    /// Panel. Row order stays [`RowOrder::AsStored`]; units are not one row axis.
    ///
    /// # Errors
    ///
    /// Unadopted validity policy, a row permutation or absolute origin, a temporal split,
    /// or an environment holdout whose indexes fall outside the units.
    pub(crate) fn from_panel(
        data: &PanelData,
        contract: SourceContract,
    ) -> Result<Self, DataError> {
        Self::try_new(OwnedBody::Panel(data.clone()), contract)
    }

    /// Multi-environment source.
    ///
    /// # Errors
    ///
    /// See [`Self::from_panel`], with environments in place of units.
    pub(crate) fn from_multi_environment(
        data: &MultiEnvironmentData,
        contract: SourceContract,
    ) -> Result<Self, DataError> {
        Self::try_new(OwnedBody::MultiEnvironment(data.clone()), contract)
    }

    fn try_new(body: OwnedBody, contract: SourceContract) -> Result<Self, DataError> {
        reject_unadopted_validity(contract.validity)?;
        validate_contract(&body, &contract)?;
        let modality = modality_of(&body);
        let content = content_of(&body);
        Ok(Self { body, modality, contract, content })
    }

    /// Shared view of this source.
    #[must_use]
    pub(crate) const fn borrowed(&self) -> BorrowedSourceSnapshot<'_> {
        BorrowedSourceSnapshot {
            body: &self.body,
            modality: &self.modality,
            contract: &self.contract,
            content: &self.content,
        }
    }

    /// Row range into this table or series. `start == end` is an empty chunk.
    ///
    /// # Errors
    ///
    /// The source has no single row axis, `start > end`, or `end` is past the row count.
    pub(crate) fn try_chunk(
        &self,
        start: usize,
        end: usize,
    ) -> Result<ChunkedSourceSnapshot<'_>, DataError> {
        chunk_from(&self.body, &self.modality, &self.contract, &self.content, start, end)
    }

    /// Scientific identity.
    #[must_use]
    pub(crate) fn identity(&self) -> SourceIdentity {
        snapshot_identity(&self.body, &self.modality, &self.contract, &self.content)
    }

    /// Full-source digest.
    #[must_use]
    pub(crate) const fn full_content_digest(&self) -> &FullContentDigest {
        &self.content
    }

    /// Schema, including units.
    #[must_use]
    pub(crate) fn schema(&self) -> &CausalSchema {
        schema_of(&self.body)
    }

    /// Row count of the single table or series.
    ///
    /// # Errors
    ///
    /// The source is a panel or multi-environment collection.
    pub(crate) fn row_count(&self) -> Result<usize, DataError> {
        population_len(&self.body).ok_or_else(|| DataError::InvalidArgument {
            message: "a partitioned source has no single row count".into(),
        })
    }

    /// Modality.
    #[must_use]
    pub(crate) const fn modality(&self) -> &SourceModality {
        &self.modality
    }

    /// Cohort and order.
    #[must_use]
    pub(crate) const fn cohort(&self) -> &CohortOrder {
        &self.contract.cohort
    }

    /// Split lineage.
    #[must_use]
    pub(crate) const fn split(&self) -> &SplitLineage {
        &self.contract.split
    }

    /// Validity policy.
    #[must_use]
    pub(crate) const fn validity_policy(&self) -> ValidityPolicy {
        self.contract.validity
    }

    /// Time index when the modality is a single series.
    #[must_use]
    pub(crate) const fn time_index(&self) -> Option<&TimeIndex> {
        match &self.modality {
            SourceModality::TimeSeries { time_index } => Some(time_index),
            SourceModality::StaticTable
            | SourceModality::Panel { .. }
            | SourceModality::MultiEnvironment { .. } => None,
        }
    }

    /// Panel unit metadata, in container order.
    #[must_use]
    pub(crate) fn panel_units(&self) -> Option<&[PanelUnitMeta]> {
        match &self.modality {
            SourceModality::Panel { units } => Some(units.as_ref()),
            SourceModality::StaticTable
            | SourceModality::TimeSeries { .. }
            | SourceModality::MultiEnvironment { .. } => None,
        }
    }

    /// Environment time indexes, in container order.
    #[must_use]
    pub(crate) fn environment_time_indexes(&self) -> Option<&[TimeIndex]> {
        match &self.modality {
            SourceModality::MultiEnvironment { time_indexes } => Some(time_indexes.as_ref()),
            SourceModality::StaticTable
            | SourceModality::TimeSeries { .. }
            | SourceModality::Panel { .. } => None,
        }
    }

    /// Physical unit label of one variable, from the schema.
    ///
    /// # Errors
    ///
    /// Unknown variable.
    pub(crate) fn variable_unit(&self, id: VariableId) -> Result<Option<&str>, DataError> {
        let variable =
            schema_of(&self.body).get(id).map_err(|_| DataError::UnknownVariable { id })?;
        Ok(variable.unit.as_deref())
    }

    /// Category domains in column order, walking each member of a partitioned source.
    #[must_use]
    pub(crate) fn category_domains(&self) -> Vec<&CategoryDomain> {
        category_domains_of(&self.body)
    }

    /// Observation weights.
    #[must_use]
    pub(crate) fn weights(&self) -> WeightView<'_> {
        weight_view(&self.body)
    }

    /// Borrow one column of a table or series.
    ///
    /// # Errors
    ///
    /// Unknown variable, or the source is partitioned.
    pub(crate) fn column(&self, id: VariableId) -> Result<ColumnView<'_>, DataError> {
        column_of(&self.body, id)
    }

    /// Borrow the `f64` values on the single row axis.
    ///
    /// # Errors
    ///
    /// Unknown variable, a non-float column, or a partitioned source.
    pub(crate) fn float64_rows(&self, id: VariableId) -> Result<&[f64], DataError> {
        let rows = full_rows(&self.body).ok_or_else(|| DataError::InvalidArgument {
            message: "a partitioned source has no single row axis".into(),
        })?;
        float_rows(&self.body, rows, id)
    }

    /// Name a column subset. The result is not this source's digest.
    ///
    /// # Errors
    ///
    /// The list is empty, repeats a variable, or names an unknown variable.
    pub(crate) fn project(&self, columns: &[VariableId]) -> Result<ProjectedSource, DataError> {
        project_columns(schema_of(&self.body), columns)
    }
}

impl<'a> BorrowedSourceSnapshot<'a> {
    /// Scientific identity of the owned source.
    #[must_use]
    pub(crate) fn identity(&self) -> SourceIdentity {
        snapshot_identity(self.body, self.modality, self.contract, self.content)
    }

    /// Full-source digest.
    #[must_use]
    pub(crate) const fn full_content_digest(&self) -> &'a FullContentDigest {
        self.content
    }

    /// Schema.
    #[must_use]
    pub(crate) fn schema(&self) -> &'a CausalSchema {
        schema_of(self.body)
    }

    /// Validity policy.
    #[must_use]
    pub(crate) const fn validity_policy(&self) -> ValidityPolicy {
        self.contract.validity
    }

    /// Observation weights.
    #[must_use]
    pub(crate) fn weights(&self) -> WeightView<'a> {
        weight_view(self.body)
    }

    /// Borrow one column.
    ///
    /// # Errors
    ///
    /// Unknown variable, or the source is partitioned.
    pub(crate) fn column(&self, id: VariableId) -> Result<ColumnView<'a>, DataError> {
        column_of(self.body, id)
    }

    /// Borrow the `f64` values on the single row axis.
    ///
    /// # Errors
    ///
    /// Unknown variable, a non-float column, or a partitioned source.
    pub(crate) fn float64_rows(&self, id: VariableId) -> Result<&'a [f64], DataError> {
        let rows = full_rows(self.body).ok_or_else(|| DataError::InvalidArgument {
            message: "a partitioned source has no single row axis".into(),
        })?;
        float_rows(self.body, rows, id)
    }

    /// Row range into the same source. Coordinates are absolute storage rows.
    ///
    /// # Errors
    ///
    /// See [`OwnedSourceSnapshot::try_chunk`].
    pub(crate) fn try_chunk(
        &self,
        start: usize,
        end: usize,
    ) -> Result<ChunkedSourceSnapshot<'a>, DataError> {
        chunk_from(self.body, self.modality, self.contract, self.content, start, end)
    }

    /// Name a column subset of the owned source.
    ///
    /// # Errors
    ///
    /// See [`OwnedSourceSnapshot::project`].
    pub(crate) fn project(&self, columns: &[VariableId]) -> Result<ProjectedSource, DataError> {
        project_columns(schema_of(self.body), columns)
    }
}

impl<'a> ChunkedSourceSnapshot<'a> {
    /// Identity of the parent source plus this row range. The chunk is not a new
    /// population, so [`ChunkIdentity::source`] is the parent's identity.
    #[must_use]
    pub(crate) fn identity(&self) -> ChunkIdentity {
        ChunkIdentity {
            source: snapshot_identity(self.body, self.modality, self.contract, self.content),
            rows: self.rows,
        }
    }

    /// Full-source digest of the parent, not a digest of the row range.
    #[must_use]
    pub(crate) const fn full_content_digest(&self) -> &'a FullContentDigest {
        self.content
    }

    /// Absolute storage-row range. Empty when `start == end`.
    #[must_use]
    pub(crate) const fn row_range(&self) -> TimeRange {
        self.rows
    }

    /// Parent column. Values outside [`Self::row_range`] are not part of the chunk.
    ///
    /// # Errors
    ///
    /// Unknown variable.
    pub(crate) fn column(&self, id: VariableId) -> Result<ColumnView<'a>, DataError> {
        column_of(self.body, id)
    }

    /// `f64` values inside [`Self::row_range`], borrowed from the parent buffer.
    ///
    /// # Errors
    ///
    /// Unknown variable or a non-float column.
    pub(crate) fn float64_rows(&self, id: VariableId) -> Result<&'a [f64], DataError> {
        float_rows(self.body, self.rows, id)
    }

    /// Column subset of the parent source. This does not become the parent digest.
    ///
    /// # Errors
    ///
    /// See [`OwnedSourceSnapshot::project`].
    pub(crate) fn project(&self, columns: &[VariableId]) -> Result<ProjectedSource, DataError> {
        project_columns(schema_of(self.body), columns)
    }
}

fn reject_unadopted_validity(policy: ValidityPolicy) -> Result<(), DataError> {
    match policy {
        ValidityPolicy::InvalidIffNan => Ok(()),
        ValidityPolicy::NullDistinctFromNan => Err(DataError::InvalidArgument {
            message: "null-distinct-from-NaN validity is declared but not adopted; f64 columns are invalid exactly when the value is NaN"
                .into(),
        }),
    }
}

fn validate_contract(body: &OwnedBody, contract: &SourceContract) -> Result<(), DataError> {
    validate_order(body, &contract.cohort)?;
    validate_split(body, &contract.split)
}

fn validate_order(body: &OwnedBody, order: &CohortOrder) -> Result<(), DataError> {
    match &order.rows {
        RowOrder::AsStored => {}
        RowOrder::Permuted(positions) => {
            let Some(n) = population_len(body) else {
                return Err(DataError::InvalidArgument {
                    message: "a row permutation needs one table or series".into(),
                });
            };
            validate_permutation(n, positions)?;
        }
    }
    match order.reference {
        ReferencePointPolicy::SeriesOrigin => Ok(()),
        ReferencePointPolicy::AbsoluteOrigin { origin_row } => {
            let Some(n) = population_len(body) else {
                return Err(DataError::InvalidArgument {
                    message: "an absolute origin needs one table or series".into(),
                });
            };
            if origin_row >= n {
                return Err(DataError::InvalidArgument {
                    message: "absolute origin must be a row of the source".into(),
                });
            }
            Ok(())
        }
    }
}

fn validate_permutation(n: usize, positions: &[u32]) -> Result<(), DataError> {
    if positions.len() != n {
        return Err(DataError::LengthMismatch {
            expected: n,
            actual: positions.len(),
            context: "row order",
        });
    }
    let mut seen = vec![false; n];
    for &position in positions {
        let position = position as usize;
        match seen.get_mut(position) {
            Some(slot) if !*slot => *slot = true,
            _ => {
                return Err(DataError::InvalidArgument {
                    message: "row order must be a permutation of the source rows".into(),
                });
            }
        }
    }
    Ok(())
}

fn validate_split(body: &OwnedBody, split: &SplitLineage) -> Result<(), DataError> {
    match split {
        SplitLineage::Unsplit => Ok(()),
        SplitLineage::DiscoveryEstimation(split) => {
            let Some(n) = population_len(body) else {
                return Err(DataError::InvalidArgument {
                    message: "a discovery/estimation split needs one row axis".into(),
                });
            };
            if split.series_len != n {
                return Err(DataError::LengthMismatch {
                    expected: n,
                    actual: split.series_len,
                    context: "split lineage",
                });
            }
            Ok(())
        }
        SplitLineage::EnvironmentHoldout(split) => {
            let Some(n) = partition_count(body) else {
                return Err(DataError::InvalidArgument {
                    message: "an environment holdout needs a panel or multi-environment source"
                        .into(),
                });
            };
            validate_holdout(n, split)
        }
    }
}

fn validate_holdout(n: usize, split: &EnvHoldoutSplit) -> Result<(), DataError> {
    if split.discovery_envs.is_empty() || split.estimation_envs.is_empty() {
        return Err(DataError::InvalidArgument {
            message: "environment holdout needs at least one discovery and one estimation \
                      environment"
                .into(),
        });
    }
    let mut seen = vec![false; n];
    for &index in split.discovery_envs.iter().chain(split.estimation_envs.iter()) {
        match seen.get_mut(index) {
            Some(slot) if !*slot => *slot = true,
            _ => {
                return Err(DataError::InvalidArgument {
                    message: "environment holdout indexes must be unique and inside the source"
                        .into(),
                });
            }
        }
    }
    if seen.iter().any(|assigned| !assigned) {
        return Err(DataError::InvalidArgument {
            message: "environment holdout leaves an environment unassigned to either side".into(),
        });
    }
    Ok(())
}

fn modality_of(body: &OwnedBody) -> SourceModality {
    match body {
        OwnedBody::Table(_) => SourceModality::StaticTable,
        OwnedBody::TimeSeries(series) => {
            SourceModality::TimeSeries { time_index: series.time_index().clone() }
        }
        OwnedBody::Panel(panel) => {
            let units = panel
                .units()
                .iter()
                .map(|unit| PanelUnitMeta {
                    unit_id: unit.unit_id,
                    time_index: unit.series.time_index().clone(),
                })
                .collect::<Vec<_>>();
            SourceModality::Panel { units: Arc::from(units) }
        }
        OwnedBody::MultiEnvironment(envs) => {
            let time_indexes =
                envs.environments().iter().map(|env| env.time_index().clone()).collect::<Vec<_>>();
            SourceModality::MultiEnvironment { time_indexes: Arc::from(time_indexes) }
        }
    }
}

fn content_of(body: &OwnedBody) -> FullContentDigest {
    match body {
        OwnedBody::Table(storage) => FullContentDigest::Table(storage.content_digest()),
        OwnedBody::TimeSeries(series) => {
            FullContentDigest::Table(series.storage().content_digest())
        }
        OwnedBody::Panel(panel) => {
            let digests = panel
                .units()
                .iter()
                .map(|unit| unit.series.storage().content_digest())
                .collect::<Vec<[u8; 32]>>();
            FullContentDigest::Partitioned(Arc::from(digests))
        }
        OwnedBody::MultiEnvironment(envs) => {
            let digests = envs
                .environments()
                .iter()
                .map(|env| env.storage().content_digest())
                .collect::<Vec<[u8; 32]>>();
            FullContentDigest::Partitioned(Arc::from(digests))
        }
    }
}

fn full_rows(body: &OwnedBody) -> Option<TimeRange> {
    let n = match body {
        OwnedBody::Table(storage) => storage.row_count(),
        OwnedBody::TimeSeries(series) => series.row_count(),
        OwnedBody::Panel(_) | OwnedBody::MultiEnvironment(_) => return None,
    };
    Some(TimeRange { start: 0, end: n })
}

fn population_len(body: &OwnedBody) -> Option<usize> {
    full_rows(body).map(TimeRange::len)
}

fn partition_count(body: &OwnedBody) -> Option<usize> {
    match body {
        OwnedBody::Panel(panel) => Some(panel.unit_count()),
        OwnedBody::MultiEnvironment(envs) => Some(envs.env_count()),
        OwnedBody::Table(_) | OwnedBody::TimeSeries(_) => None,
    }
}

fn schema_of(body: &OwnedBody) -> &CausalSchema {
    match body {
        OwnedBody::Table(storage) => storage.schema(),
        OwnedBody::TimeSeries(series) => series.schema(),
        OwnedBody::Panel(panel) => panel.schema(),
        OwnedBody::MultiEnvironment(envs) => envs.schema(),
    }
}

fn snapshot_identity(
    body: &OwnedBody,
    modality: &SourceModality,
    contract: &SourceContract,
    content: &FullContentDigest,
) -> SourceIdentity {
    SourceIdentity {
        content: content.clone(),
        schema: schema_of(body).clone(),
        modality: modality.clone(),
        cohort: contract.cohort.clone(),
        split: contract.split.clone(),
        validity: contract.validity,
    }
}

fn column_of(body: &OwnedBody, id: VariableId) -> Result<ColumnView<'_>, DataError> {
    match body {
        OwnedBody::Table(storage) => storage.column(id),
        OwnedBody::TimeSeries(series) => series.column(id),
        OwnedBody::Panel(_) | OwnedBody::MultiEnvironment(_) => Err(DataError::InvalidArgument {
            message: "a partitioned source has no single column axis".into(),
        }),
    }
}

fn float_rows(body: &OwnedBody, rows: TimeRange, id: VariableId) -> Result<&[f64], DataError> {
    match column_of(body, id)? {
        ColumnView::Float64(column) => {
            let values = column.values.as_slice();
            values.get(rows.start..rows.end).ok_or_else(|| DataError::InvalidArgument {
                message: "chunk range is outside the column".into(),
            })
        }
        _ => Err(DataError::TypeMismatch { id, expected: "float64" }),
    }
}

fn category_domains_of(body: &OwnedBody) -> Vec<&CategoryDomain> {
    let mut out = Vec::new();
    match body {
        OwnedBody::Table(storage) => push_domains(storage, &mut out),
        OwnedBody::TimeSeries(series) => push_domains(series.storage(), &mut out),
        OwnedBody::Panel(panel) => {
            for unit in panel.units() {
                push_domains(unit.series.storage(), &mut out);
            }
        }
        OwnedBody::MultiEnvironment(envs) => {
            for env in envs.environments() {
                push_domains(env.storage(), &mut out);
            }
        }
    }
    out
}

fn push_domains<'a>(storage: &'a OwnedColumnarStorage, out: &mut Vec<&'a CategoryDomain>) {
    for column in storage.columns() {
        if let ColumnView::Categorical(column) = column.as_view() {
            out.push(column.domain.as_ref());
        }
    }
}

fn weight_view(body: &OwnedBody) -> WeightView<'_> {
    match body {
        OwnedBody::Table(storage) => WeightView::Rows(storage.weights()),
        OwnedBody::TimeSeries(series) => WeightView::Rows(series.storage().weights()),
        OwnedBody::Panel(panel) => WeightView::Partitioned(
            panel.units().iter().map(|unit| unit.series.storage().weights()).collect(),
        ),
        OwnedBody::MultiEnvironment(envs) => WeightView::Partitioned(
            envs.environments().iter().map(|env| env.storage().weights()).collect(),
        ),
    }
}

fn chunk_from<'a>(
    body: &'a OwnedBody,
    modality: &'a SourceModality,
    contract: &'a SourceContract,
    content: &'a FullContentDigest,
    start: usize,
    end: usize,
) -> Result<ChunkedSourceSnapshot<'a>, DataError> {
    let Some(population) = full_rows(body) else {
        return Err(DataError::InvalidArgument {
            message: "a chunk is a row range of one table or series, not a new population".into(),
        });
    };
    if start > end || end > population.len() {
        return Err(DataError::InvalidArgument {
            message: "chunk range is outside the source".into(),
        });
    }
    Ok(ChunkedSourceSnapshot { body, modality, contract, content, rows: TimeRange { start, end } })
}

fn project_columns(
    schema: &CausalSchema,
    columns: &[VariableId],
) -> Result<ProjectedSource, DataError> {
    if columns.is_empty() {
        return Err(DataError::InvalidArgument {
            message: "a column projection names no variables and is not the full source".into(),
        });
    }
    let mut seen = vec![false; schema.len()];
    for &id in columns {
        // Schema ids are dense, so an in-range index is a variable of this schema.
        let index = id.as_usize();
        match seen.get_mut(index) {
            Some(slot) if !*slot => *slot = true,
            Some(_) => {
                return Err(DataError::InvalidArgument {
                    message: "column projection repeats a variable".into(),
                });
            }
            None => return Err(DataError::UnknownVariable { id }),
        }
    }
    Ok(ProjectedSource { columns: Arc::from(columns) })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{
        CategoryDomainId, CausalSchemaBuilder, MeasurementSpec, SmallRoleSet, ValueType, VariableId,
    };

    use super::*;
    use crate::categorical::CategoricalColumn;
    use crate::categorical::{CategoryCode, CategoryDomain, CategoryLevel, UnknownCategoryPolicy};
    use crate::column::{ColumnView, Float64Column, OwnedColumn, ValidityBitmap};
    use crate::dataset::{TabularData, TimeSeriesData};
    use crate::error::DataError;
    use crate::multi_env::MultiEnvironmentData;
    use crate::panel::{PanelData, PanelUnit};
    use crate::reference::ReferencePointPolicy;
    use crate::split::{DiscoveryEstimationSplit, EnvHoldoutSplit, TimeRange};
    use crate::storage::OwnedColumnarStorage;

    fn contract(label: &str) -> SourceContract {
        SourceContract::new(CohortOrder::as_stored(label), SplitLineage::Unsplit)
    }

    fn sample_table() -> TabularData {
        TabularData::from_f64_columns([
            ("x", &[1.0_f64, 2.0, 3.0][..]),
            ("y", &[4.0_f64, 5.0, 6.0][..]),
        ])
        .unwrap()
    }

    fn one_column(name: &str) -> OwnedColumnarStorage {
        let mut builder = CausalSchemaBuilder::new();
        builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::empty(),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        let column = Float64Column::new(
            VariableId::from_raw(0),
            Arc::<[f64]>::from([1.0, 2.0, 3.0]),
            ValidityBitmap::all_valid(3),
        )
        .unwrap();
        OwnedColumnarStorage::try_new(
            builder.build().unwrap(),
            vec![OwnedColumn::Float64(column)],
            None,
            None,
        )
        .unwrap()
    }

    #[test]
    fn owned_and_borrowed_share_identity() {
        let data = sample_table();
        let snap = OwnedSourceSnapshot::from_tabular(&data, contract("c")).unwrap();
        let borrowed = snap.borrowed();
        assert_eq!(snap.identity(), borrowed.identity());
        let identity = snap.identity();
        assert_eq!(identity.content(), snap.full_content_digest());
        assert_eq!(identity.schema().len(), snap.schema().len());
        assert_eq!(identity.modality(), snap.modality());
        assert_eq!(identity.cohort(), snap.cohort());
        assert_eq!(identity.split(), snap.split());
        assert_eq!(identity.validity(), snap.validity_policy());
        assert_eq!(snap.full_content_digest(), borrowed.full_content_digest());
        let FullContentDigest::Table(digest) = snap.full_content_digest() else {
            panic!("tabular data is one table");
        };
        assert_eq!(*digest, data.storage().content_digest());
        assert_eq!(snap.row_count().unwrap(), 3);
        assert_eq!(snap.validity_policy(), borrowed.validity_policy());
        assert_eq!(snap.validity_policy(), ValidityPolicy::InvalidIffNan);
        assert!(std::ptr::eq(snap.schema(), borrowed.schema()));
        assert!(matches!(snap.modality(), SourceModality::StaticTable));
        assert_eq!(snap.cohort().label(), "c");
        assert!(matches!(snap.cohort().rows(), RowOrder::AsStored));
        assert_eq!(snap.cohort().reference(), ReferencePointPolicy::SeriesOrigin);
        assert!(matches!(snap.split(), SplitLineage::Unsplit));
        assert!(snap.time_index().is_none());
        assert!(matches!(snap.weights(), WeightView::Rows(None)));
        assert!(matches!(borrowed.weights(), WeightView::Rows(None)));
        assert!(snap.category_domains().is_empty());
        let ColumnView::Float64(owned_column) = snap.column(VariableId::from_raw(0)).unwrap()
        else {
            panic!("expected float64");
        };
        let ColumnView::Float64(borrowed_column) =
            borrowed.column(VariableId::from_raw(0)).unwrap()
        else {
            panic!("expected float64");
        };
        assert!(std::ptr::eq(owned_column.values.as_slice(), borrowed_column.values.as_slice()));
        assert_eq!(snap.float64_rows(VariableId::from_raw(0)).unwrap().len(), 3);
    }

    #[test]
    fn schema_and_time_index_change_identity() {
        let x = OwnedSourceSnapshot::from_storage(one_column("x"), contract("c")).unwrap();
        let z = OwnedSourceSnapshot::from_storage(one_column("z"), contract("c")).unwrap();
        assert_eq!(x.full_content_digest(), z.full_content_digest());
        assert_ne!(x.identity(), z.identity());

        let table = sample_table();
        let static_snap = OwnedSourceSnapshot::from_tabular(&table, contract("c")).unwrap();
        let series = TimeSeriesData::from_f64_columns(
            [("x", &[1.0_f64, 2.0, 3.0][..]), ("y", &[4.0, 5.0, 6.0][..])],
            10,
        )
        .unwrap();
        let timed = OwnedSourceSnapshot::from_time_series(&series, contract("c")).unwrap();
        let digest = series.storage().content_digest();
        assert_eq!(timed.full_content_digest(), &FullContentDigest::Table(digest));
        assert_eq!(static_snap.full_content_digest(), timed.full_content_digest());
        assert_ne!(static_snap.identity(), timed.identity());
        assert_eq!(timed.time_index().map(|index| index.length), Some(3));
        let slower = TimeSeriesData::from_f64_columns(
            [("x", &[1.0_f64, 2.0, 3.0][..]), ("y", &[4.0, 5.0, 6.0][..])],
            11,
        )
        .unwrap();
        let slower_snap = OwnedSourceSnapshot::from_time_series(&slower, contract("c")).unwrap();
        assert_eq!(timed.full_content_digest(), slower_snap.full_content_digest());
        assert_ne!(timed.identity(), slower_snap.identity());
    }

    #[test]
    fn nan_changes_content_identity() {
        let data = sample_table();
        let clean = OwnedSourceSnapshot::from_tabular(&data, contract("c")).unwrap();
        let dirty_data = data
            .with_replaced_float(VariableId::from_raw(0), Arc::from([1.0, f64::NAN, 3.0]))
            .unwrap();
        let dirty = OwnedSourceSnapshot::from_tabular(&dirty_data, contract("c")).unwrap();
        assert_eq!(clean.validity_policy(), dirty.validity_policy());
        assert_ne!(clean.full_content_digest(), dirty.full_content_digest());
        assert_ne!(clean.identity(), dirty.identity());
    }

    #[test]
    fn weight_change_changes_content_identity() {
        let data = sample_table();
        let unweighted = OwnedSourceSnapshot::from_tabular(&data, contract("c")).unwrap();
        let weighted_data = data.with_weights(Arc::from([1.0, 1.0, 2.0])).unwrap();
        let weighted = OwnedSourceSnapshot::from_tabular(&weighted_data, contract("c")).unwrap();
        assert!(matches!(
            weighted.weights(),
            WeightView::Rows(Some(weights)) if weights.len() == 3
        ));
        assert_ne!(unweighted.full_content_digest(), weighted.full_content_digest());
        assert_ne!(unweighted.identity(), weighted.identity());
    }

    #[test]
    fn projection_is_not_the_full_identity() {
        let snap = OwnedSourceSnapshot::from_tabular(&sample_table(), contract("c")).unwrap();
        let full = snap.full_content_digest().clone();
        let x = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let projected = snap.project(&[x]).unwrap();
        let borrowed = snap.borrowed().project(&[x]).unwrap();
        assert_eq!(projected.columns(), &[x]);
        assert_eq!(projected.identity().columns(), projected.columns());
        assert_eq!(borrowed.identity(), projected.identity());
        assert!(matches!(projected.full_content_digest(), Err(DataError::InvalidArgument { .. })));
        assert!(snap.project(&[x, y]).unwrap().full_content_digest().is_err());
        assert!(
            snap.try_chunk(0, 1).unwrap().project(&[x]).unwrap().full_content_digest().is_err()
        );
        assert!(matches!(snap.project(&[]), Err(DataError::InvalidArgument { .. })));
        assert!(matches!(snap.project(&[x, x]), Err(DataError::InvalidArgument { .. })));
        assert!(matches!(
            snap.project(&[VariableId::from_raw(9)]),
            Err(DataError::UnknownVariable { .. })
        ));
        assert_eq!(snap.full_content_digest(), &full);
    }

    #[test]
    fn chunk_ranges_stay_inside_the_source() {
        let snap = OwnedSourceSnapshot::from_tabular(&sample_table(), contract("c")).unwrap();
        let x = VariableId::from_raw(0);
        let parent = snap.float64_rows(x).unwrap();
        let head = snap.try_chunk(0, 0).unwrap();
        let tail = snap.try_chunk(2, 3).unwrap();
        let empty_tail = snap.try_chunk(3, 3).unwrap();
        let borrowed_tail = snap.borrowed().try_chunk(2, 3).unwrap();
        assert!(head.row_range().is_empty());
        assert!(head.float64_rows(x).unwrap().is_empty());
        assert!(empty_tail.row_range().is_empty());
        assert_eq!(tail.row_range(), TimeRange { start: 2, end: 3 });
        assert_eq!(empty_tail.row_range(), TimeRange { start: 3, end: 3 });
        let tail_values = tail.float64_rows(x).unwrap();
        assert_eq!(tail_values.len(), 1);
        assert_eq!(tail_values.as_ptr(), parent[2..].as_ptr());
        assert_eq!(tail.identity().source(), &snap.identity());
        assert_eq!(empty_tail.identity().source(), &snap.identity());
        assert_eq!(borrowed_tail.identity(), tail.identity());
        // Different row ranges of one source are different chunks, empty ranges included.
        assert_eq!(tail.identity().row_range(), TimeRange { start: 2, end: 3 });
        assert_ne!(tail.identity(), empty_tail.identity());
        assert_ne!(head.identity(), empty_tail.identity());
        assert_ne!(tail.identity(), snap.try_chunk(1, 3).unwrap().identity());
        assert_eq!(tail.full_content_digest(), snap.full_content_digest());
        assert_ne!(head.row_range(), tail.row_range());
        let ColumnView::Float64(parent_column) = tail.column(x).unwrap() else {
            panic!("expected float64");
        };
        assert!(std::ptr::eq(parent_column.values.as_slice(), parent));
        assert!(matches!(snap.try_chunk(0, 4), Err(DataError::InvalidArgument { .. })));
        assert!(matches!(snap.try_chunk(2, 1), Err(DataError::InvalidArgument { .. })));
        assert!(matches!(snap.try_chunk(4, 4), Err(DataError::InvalidArgument { .. })));
    }

    #[test]
    fn split_lineage_is_part_of_the_snapshot() {
        let data = sample_table();
        let base = OwnedSourceSnapshot::from_tabular(&data, contract("c")).unwrap();
        let early = DiscoveryEstimationSplit::from_sizes(3, 1, 0, 2).unwrap();
        let late = DiscoveryEstimationSplit::from_sizes(3, 2, 0, 1).unwrap();
        let early_snap = OwnedSourceSnapshot::from_tabular(
            &data,
            SourceContract::new(
                CohortOrder::as_stored("c"),
                SplitLineage::DiscoveryEstimation(early),
            ),
        )
        .unwrap();
        let late_snap = OwnedSourceSnapshot::from_tabular(
            &data,
            SourceContract::new(
                CohortOrder::as_stored("c"),
                SplitLineage::DiscoveryEstimation(late),
            ),
        )
        .unwrap();
        assert_eq!(base.full_content_digest(), early_snap.full_content_digest());
        assert_ne!(base.identity(), early_snap.identity());
        assert_ne!(early_snap.identity(), late_snap.identity());
        assert_eq!(early_snap.split(), &SplitLineage::DiscoveryEstimation(early));
        let mismatched = DiscoveryEstimationSplit::try_new(
            4,
            TimeRange { start: 0, end: 1 },
            TimeRange { start: 1, end: 4 },
        )
        .unwrap();
        assert!(matches!(
            OwnedSourceSnapshot::from_tabular(
                &data,
                SourceContract::new(
                    CohortOrder::as_stored("c"),
                    SplitLineage::DiscoveryEstimation(mismatched),
                ),
            ),
            Err(DataError::LengthMismatch { .. })
        ));
        let holdout = EnvHoldoutSplit::try_prefix(2, 1).unwrap();
        assert!(matches!(
            OwnedSourceSnapshot::from_tabular(
                &data,
                SourceContract::new(
                    CohortOrder::as_stored("c"),
                    SplitLineage::EnvironmentHoldout(holdout),
                ),
            ),
            Err(DataError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn order_or_cohort_change_is_visible() {
        let data = sample_table();
        let base = OwnedSourceSnapshot::from_tabular(&data, contract("treated")).unwrap();
        let relabeled = OwnedSourceSnapshot::from_tabular(&data, contract("control")).unwrap();
        let flipped = OwnedSourceSnapshot::from_tabular(
            &data,
            SourceContract::new(
                CohortOrder::permuted("treated", Arc::<[u32]>::from([2, 1, 0])),
                SplitLineage::Unsplit,
            ),
        )
        .unwrap();
        let origin = CohortOrder::as_stored("treated")
            .with_reference(ReferencePointPolicy::AbsoluteOrigin { origin_row: 1 });
        let shifted = OwnedSourceSnapshot::from_tabular(
            &data,
            SourceContract::new(origin, SplitLineage::Unsplit),
        )
        .unwrap();
        assert_eq!(base.full_content_digest(), relabeled.full_content_digest());
        assert_eq!(base.full_content_digest(), flipped.full_content_digest());
        assert_eq!(base.full_content_digest(), shifted.full_content_digest());
        assert_ne!(base.identity(), relabeled.identity());
        assert_ne!(base.identity(), flipped.identity());
        assert_ne!(base.identity(), shifted.identity());
        assert!(matches!(flipped.cohort().rows(), RowOrder::Permuted(_)));
        assert_eq!(
            shifted.cohort().reference(),
            ReferencePointPolicy::AbsoluteOrigin { origin_row: 1 }
        );
        let repeated = CohortOrder::permuted("treated", Arc::<[u32]>::from([0, 0, 1]));
        assert!(matches!(
            OwnedSourceSnapshot::from_tabular(
                &data,
                SourceContract::new(repeated, SplitLineage::Unsplit),
            ),
            Err(DataError::InvalidArgument { .. })
        ));
        let past_end = CohortOrder::as_stored("treated")
            .with_reference(ReferencePointPolicy::AbsoluteOrigin { origin_row: 3 });
        assert!(
            OwnedSourceSnapshot::from_tabular(
                &data,
                SourceContract::new(past_end, SplitLineage::Unsplit),
            )
            .is_err()
        );
    }

    #[test]
    fn null_distinct_from_nan_is_not_adopted() {
        let contract = SourceContract::new(CohortOrder::as_stored("c"), SplitLineage::Unsplit)
            .with_validity(ValidityPolicy::NullDistinctFromNan);
        assert!(matches!(
            OwnedSourceSnapshot::from_tabular(&sample_table(), contract),
            Err(DataError::InvalidArgument { .. })
        ));
    }

    fn series(values: &[f64], interval_ns: u64) -> TimeSeriesData {
        TimeSeriesData::from_f64_columns([("x", values)], interval_ns).unwrap()
    }

    fn panel(first_id: u32, second_id: u32) -> PanelData {
        PanelData::try_new(Arc::from([
            PanelUnit { unit_id: first_id, series: series(&[1.0, 2.0], 1) },
            PanelUnit { unit_id: second_id, series: series(&[3.0, 4.0, 5.0], 1) },
        ]))
        .unwrap()
    }

    #[test]
    fn partitioned_sources_keep_each_members_full_digest() {
        let panel_data = panel(0, 1);
        let snap = OwnedSourceSnapshot::from_panel(&panel_data, contract("c")).unwrap();
        let first = panel_data.unit(0).unwrap().series.storage().content_digest();
        let second = panel_data.unit(1).unwrap().series.storage().content_digest();
        match snap.full_content_digest() {
            FullContentDigest::Partitioned(parts) => assert_eq!(parts.as_ref(), &[first, second]),
            FullContentDigest::Table(_) => panic!("a panel is not one table digest"),
        }
        assert_eq!(snap.panel_units().unwrap().len(), 2);
        assert_eq!(snap.panel_units().unwrap()[0].unit_id(), 0);
        assert_eq!(snap.panel_units().unwrap()[1].time_index().length, 3);
        assert!(snap.time_index().is_none());
        assert!(snap.environment_time_indexes().is_none());
        assert!(matches!(snap.weights(), WeightView::Partitioned(weights) if weights.len() == 2));
        assert!(snap.row_count().is_err());
        assert!(snap.column(VariableId::from_raw(0)).is_err());
        assert!(snap.float64_rows(VariableId::from_raw(0)).is_err());
        assert!(snap.try_chunk(0, 1).is_err());
        assert!(snap.project(&[VariableId::from_raw(0)]).unwrap().full_content_digest().is_err());
        let relabeled = OwnedSourceSnapshot::from_panel(&panel(7, 8), contract("c")).unwrap();
        assert_eq!(snap.full_content_digest(), relabeled.full_content_digest());
        assert_ne!(snap.identity(), relabeled.identity());

        let envs = MultiEnvironmentData::try_new(Arc::from([
            series(&[1.0, 2.0], 1),
            series(&[3.0, 4.0, 5.0], 1),
        ]))
        .unwrap();
        let holdout = EnvHoldoutSplit::try_prefix(2, 1).unwrap();
        let held = OwnedSourceSnapshot::from_multi_environment(
            &envs,
            SourceContract::new(
                CohortOrder::as_stored("c"),
                SplitLineage::EnvironmentHoldout(holdout),
            ),
        )
        .unwrap();
        let swapped = EnvHoldoutSplit {
            discovery_envs: Arc::from([1usize]),
            estimation_envs: Arc::from([0usize]),
        };
        let swapped_snap = OwnedSourceSnapshot::from_multi_environment(
            &envs,
            SourceContract::new(
                CohortOrder::as_stored("c"),
                SplitLineage::EnvironmentHoldout(swapped),
            ),
        )
        .unwrap();
        assert!(matches!(
            held.full_content_digest(),
            FullContentDigest::Partitioned(parts) if parts.len() == 2
        ));
        assert_eq!(held.environment_time_indexes().unwrap().len(), 2);
        assert!(held.panel_units().is_none());
        assert_ne!(held.identity(), swapped_snap.identity());
        assert_eq!(held.full_content_digest(), swapped_snap.full_content_digest());
    }

    #[test]
    fn holdout_needs_both_sides_and_every_environment() {
        let envs = MultiEnvironmentData::try_new(Arc::from([
            series(&[1.0, 2.0], 1),
            series(&[3.0, 4.0, 5.0], 1),
            series(&[6.0, 7.0], 1),
        ]))
        .unwrap();
        let build = |discovery: &[usize], estimation: &[usize]| {
            let split = EnvHoldoutSplit {
                discovery_envs: Arc::from(discovery),
                estimation_envs: Arc::from(estimation),
            };
            OwnedSourceSnapshot::from_multi_environment(
                &envs,
                SourceContract::new(
                    CohortOrder::as_stored("c"),
                    SplitLineage::EnvironmentHoldout(split),
                ),
            )
        };
        build(&[0, 1], &[2]).unwrap();
        for (discovery, estimation) in [
            (&[][..], &[0, 1, 2][..]),
            (&[0, 1, 2][..], &[][..]),
            (&[0][..], &[1][..]),
            (&[0, 1][..], &[1, 2][..]),
        ] {
            assert!(
                matches!(build(discovery, estimation), Err(DataError::InvalidArgument { .. })),
                "{discovery:?} / {estimation:?}"
            );
        }
    }

    fn categorical_storage(label: &str, unit: &str) -> OwnedColumnarStorage {
        let domain = CategoryDomain::try_new(
            CategoryDomainId::from_raw(0),
            Arc::from([CategoryLevel { label: Arc::from(label) }]),
            false,
            None,
            UnknownCategoryPolicy::Fail,
        )
        .unwrap();
        let mut builder = CausalSchemaBuilder::new();
        builder
            .add_variable(
                "group",
                ValueType::Categorical,
                SmallRoleSet::empty(),
                Some(Arc::from(unit)),
                Some(CategoryDomainId::from_raw(0)),
                MeasurementSpec::default(),
            )
            .unwrap();
        let column = CategoricalColumn::try_new(
            VariableId::from_raw(0),
            Arc::from([CategoryCode::from_raw(0), CategoryCode::from_raw(0)]),
            ValidityBitmap::all_valid(2),
            Arc::new(domain),
        )
        .unwrap();
        OwnedColumnarStorage::try_new(
            builder.build().unwrap(),
            vec![OwnedColumn::Categorical(column)],
            None,
            None,
        )
        .unwrap()
    }

    #[test]
    fn category_domains_and_units_come_from_storage() {
        let storage = categorical_storage("case", "mg");
        let snap = OwnedSourceSnapshot::from_storage(storage.clone(), contract("c")).unwrap();
        let domains = snap.category_domains();
        assert_eq!(domains.len(), 1);
        assert_eq!(domains[0].levels[0].label.as_ref(), "case");
        assert_eq!(snap.variable_unit(VariableId::from_raw(0)).unwrap(), Some("mg"));
        assert!(matches!(
            snap.column(VariableId::from_raw(0)).unwrap(),
            ColumnView::Categorical(_)
        ));
        assert!(matches!(
            snap.float64_rows(VariableId::from_raw(0)),
            Err(DataError::TypeMismatch { .. })
        ));
        assert!(matches!(
            snap.variable_unit(VariableId::from_raw(3)),
            Err(DataError::UnknownVariable { .. })
        ));
        let other =
            OwnedSourceSnapshot::from_storage(categorical_storage("control", "mg"), contract("c"))
                .unwrap();
        assert_ne!(snap.full_content_digest(), other.full_content_digest());
        assert_ne!(snap.identity(), other.identity());
        let grams = OwnedSourceSnapshot::from_storage(
            OwnedColumnarStorage::try_new(
                {
                    let mut builder = CausalSchemaBuilder::new();
                    builder
                        .add_variable(
                            "group",
                            ValueType::Categorical,
                            SmallRoleSet::empty(),
                            Some(Arc::from("g")),
                            Some(CategoryDomainId::from_raw(0)),
                            MeasurementSpec::default(),
                        )
                        .unwrap();
                    builder.build().unwrap()
                },
                storage.columns().to_vec(),
                None,
                None,
            )
            .unwrap(),
            contract("c"),
        )
        .unwrap();
        assert_eq!(snap.full_content_digest(), grams.full_content_digest());
        assert_ne!(snap.identity(), grams.identity());
        assert_eq!(grams.variable_unit(VariableId::from_raw(0)).unwrap(), Some("g"));
    }
}
