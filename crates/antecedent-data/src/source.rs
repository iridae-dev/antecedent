//! Source and snapshot contract over owned storage.
//!
//! A snapshot binds modality, cohort order, validity policy, units, splits, an
//! optional time index, the schema (names, value types, roles, units and
//! measurement), event timestamps when the source has them, and the existing
//! full-content digest (columns, mask, and weights). A projection does not
//! reuse that digest. A chunk is a row range of the same source, not a new
//! population.
//!
//! Every modality (table, series, panel, multi-environment, event series) reaches
//! this contract through its own `source` constructor, and every setter of the
//! owned source (`with_weights`, `with_analysis_mask`, `with_replaced_float`,
//! `with_schema`, `with_time_index`, `with_event_times`, `with_part`) yields a source
//! whose identity is recomputed from the new content, so a lazy view or an
//! incremental digest taken of the earlier source no longer matches it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::ops::Range;
use std::sync::Arc;

use antecedent_core::{CausalSchema, VariableId};

use crate::column::ValidityBitmap;
use crate::dataset::{TabularData, TimeSeriesData};
use crate::error::DataError;
use crate::event::EventData;
use crate::multi_env::MultiEnvironmentData;
use crate::panel::PanelData;
use crate::storage::OwnedColumnarStorage;
use crate::table::TableView;
use crate::temporal::{SamplingRegularity, TimeIndex};

/// How the rows are organized. This does not change the stored columns.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceModality {
    /// One row per unit, no time index on the snapshot.
    Tabular,
    /// Panel units.
    Panel,
    /// Several environments.
    MultiEnvironment,
    /// A time series.
    TimeSeries,
    /// Irregular events with their own timestamps.
    EventSeries,
}

/// What invalid means for the values this snapshot points at.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValidityPolicy {
    /// Current f64 rule: a value is invalid exactly when it is NaN.
    F64InvalidIffNan,
    /// Declared so a later adapter can tell null from NaN. Not a kernel view.
    NullDistinctFromNan,
}

/// Row order relative to storage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CohortOrder {
    /// Dense row `i` is storage row `i`.
    Storage,
    /// Named order. The label is part of the snapshot; the columns are not permuted here.
    Labeled(Arc<str>),
}

/// A named row span that belongs to the snapshot (a split, not a chunk).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SplitSpan {
    /// Split name.
    pub label: Arc<str>,
    /// Half-open row range in storage order.
    pub rows: Range<usize>,
}

/// Full-source digest, or a projection that must not be compared as that digest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceIdentity {
    /// [`OwnedColumnarStorage::content_digest`] of the original columns, mask, and weights.
    Full([u8; 32]),
    /// Columns drawn from `parent`. Not a full-source identity.
    Projected {
        /// Digest of the source the columns were taken from.
        parent: [u8; 32],
        /// Stable fingerprint of the selected dense column ids.
        columns: u64,
    },
}

/// Scientific description of one immutable source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SnapshotContract {
    /// Table, panel, environment, or series.
    pub modality: SourceModality,
    /// Invalid-value rule.
    pub validity: ValidityPolicy,
    /// Cohort order.
    pub cohort: CohortOrder,
    /// Unit or environment label, when the modality has one.
    pub unit: Option<Arc<str>>,
    /// Split lineage stored with the source.
    pub splits: Vec<SplitSpan>,
    /// Content identity. Projections use [`SourceIdentity::Projected`].
    pub identity: SourceIdentity,
    /// Sampling metadata. Absent when the source declares no time axis.
    ///
    /// The length, when present, is the source row count. It is not part of
    /// the storage digest.
    pub time_index: Option<TimeIndex>,
    /// Row count of the source. Chunks do not change it.
    pub rows: usize,
    /// Schema variable names in dense order.
    pub schema_names: Arc<[Arc<str>]>,
    /// Digest of the whole schema (names, value types, roles, units, category domains and
    /// measurement). The storage digest does not cover it, so two sources with the same
    /// bytes and different roles are different sources.
    pub schema: [u8; 32],
    /// Digest of the event timestamps, for an event series. The storage digest does not
    /// cover them.
    pub event_times: Option<[u8; 32]>,
}

/// Owned immutable source.
#[derive(Clone, Debug)]
pub struct OwnedSource {
    storage: OwnedColumnarStorage,
    contract: SnapshotContract,
}

/// Borrow of an owned source. The borrow does not expose mutation.
#[derive(Clone, Copy, Debug)]
pub struct BorrowedSource<'a> {
    storage: &'a OwnedColumnarStorage,
    contract: &'a SnapshotContract,
}

/// A row range of a source. Empty and tail ranges are allowed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowChunk {
    parent: [u8; 32],
    start: usize,
    end: usize,
}

impl OwnedSource {
    /// Snapshot `storage` under the current f64 validity rule and storage order.
    #[must_use]
    pub fn from_storage(storage: OwnedColumnarStorage, modality: SourceModality) -> Self {
        let contract = contract_for(&storage, modality, ValidityPolicy::F64InvalidIffNan);
        Self { storage, contract }
    }

    /// Snapshot a table with no time axis.
    ///
    /// The storage digest is computed on `data` first, so the snapshot carries it and
    /// `data` keeps its cached copy.
    #[must_use]
    pub fn from_tabular(data: &TabularData) -> Self {
        let _ = data.storage().content_digest();
        Self::from_storage(data.storage().clone(), SourceModality::Tabular)
    }

    /// Snapshot a time series under `modality`, binding the series time index.
    ///
    /// The index is the one the series carries, so a series and the same storage
    /// given [`Self::with_time_index`] have equal contracts.
    ///
    /// # Errors
    ///
    /// `modality` is [`SourceModality::Tabular`], which declares no time axis.
    pub fn from_time_series(
        series: &TimeSeriesData,
        modality: SourceModality,
    ) -> Result<Self, DataError> {
        if matches!(modality, SourceModality::Tabular) {
            return Err(DataError::unsupported(
                "building a tabular source from a time series",
                "a time series carries a time index",
            ));
        }
        Ok(series_source(series, modality))
    }

    /// Snapshot an event series: irregular time axis, with the event timestamps bound
    /// into the contract because the storage digest does not carry them.
    #[must_use]
    pub fn from_event_series(events: &EventData) -> Self {
        let _ = events.storage().content_digest();
        let mut source = Self::from_storage(events.storage().clone(), SourceModality::EventSeries);
        source.contract.time_index = Some(TimeIndex {
            regularity: SamplingRegularity::Irregular,
            length: events.storage().row_count(),
        });
        source.contract.event_times = Some(event_times_digest(events.event_times_ns()));
        source
    }

    /// Replace the event timestamps of an event series.
    ///
    /// # Errors
    ///
    /// The source is not an event series, the count is not the row count, or the timestamps
    /// decrease.
    pub fn with_event_times(mut self, times_ns: &[i64]) -> Result<Self, DataError> {
        if self.contract.modality != SourceModality::EventSeries {
            return Err(DataError::unsupported(
                "event timestamps on this source",
                "they belong to an event series",
            ));
        }
        if times_ns.len() != self.contract.rows {
            return Err(DataError::LengthMismatch {
                expected: self.contract.rows,
                actual: times_ns.len(),
                context: "event times",
            });
        }
        if times_ns.windows(2).any(|pair| pair[1] < pair[0]) {
            return Err(DataError::not_ordered("event times"));
        }
        self.contract.event_times = Some(event_times_digest(times_ns));
        Ok(self)
    }

    /// Replace the observation weights. The weights are part of the content, so the source
    /// identity is recomputed.
    ///
    /// # Errors
    ///
    /// A weight count that is not the row count, or weights that are not valid weights.
    pub fn with_weights(self, weights: Arc<[f64]>) -> Result<Self, DataError> {
        let storage = TabularData::new(self.storage.clone()).with_weights(weights)?;
        Ok(self.rebuilt(storage.storage().clone()))
    }

    /// Restrict analysis to the rows `mask` keeps (intersected with any existing mask). The
    /// mask is part of the content, so the source identity is recomputed.
    ///
    /// # Errors
    ///
    /// A mask whose length is not the row count.
    pub fn with_analysis_mask(self, mask: ValidityBitmap) -> Result<Self, DataError> {
        let storage = TabularData::new(self.storage.clone()).with_analysis_mask(mask)?;
        Ok(self.rebuilt(storage.storage().clone()))
    }

    /// Replace one float64 column. The source identity is recomputed.
    ///
    /// # Errors
    ///
    /// Unknown id, length mismatch, or non-float target.
    pub fn with_replaced_float(
        self,
        id: VariableId,
        values: Arc<[f64]>,
    ) -> Result<Self, DataError> {
        let storage = TabularData::new(self.storage.clone()).with_replaced_float(id, values)?;
        Ok(self.rebuilt(storage.storage().clone()))
    }

    /// Replace the schema over the same columns (a rename, a role or a unit change). The
    /// schema digest and the schema names of the contract follow it.
    ///
    /// # Errors
    ///
    /// A schema that does not fit the columns.
    pub fn with_schema(self, schema: CausalSchema) -> Result<Self, DataError> {
        let storage = self.storage.with_schema(schema)?;
        Ok(self.rebuilt(storage))
    }

    /// Install `storage` of the same shape and recompute everything the contract derives from
    /// it; cohort, unit, splits, validity, time index and event timestamps are kept.
    fn rebuilt(mut self, storage: OwnedColumnarStorage) -> Self {
        let fresh = contract_for(&storage, self.contract.modality, self.contract.validity);
        self.contract.identity = fresh.identity;
        self.contract.schema_names = fresh.schema_names;
        self.contract.schema = fresh.schema;
        self.contract.rows = fresh.rows;
        self.storage = storage;
        self
    }

    /// Borrow the snapshot.
    #[must_use]
    pub fn borrow(&self) -> BorrowedSource<'_> {
        BorrowedSource { storage: &self.storage, contract: &self.contract }
    }

    /// Contract.
    #[must_use]
    pub fn contract(&self) -> &SnapshotContract {
        &self.contract
    }

    /// Replace cohort order. Content digest is unchanged.
    #[must_use]
    pub fn with_cohort(mut self, cohort: CohortOrder) -> Self {
        self.contract.cohort = cohort;
        self
    }

    /// Replace the unit label.
    #[must_use]
    pub fn with_unit(mut self, unit: impl Into<Arc<str>>) -> Self {
        self.contract.unit = Some(unit.into());
        self
    }

    /// Attach splits. Ranges must lie inside the source.
    ///
    /// # Errors
    ///
    /// A split endpoint past the row count.
    pub fn with_splits(mut self, splits: Vec<SplitSpan>) -> Result<Self, DataError> {
        for split in &splits {
            if split.rows.end > self.contract.rows || split.rows.start > split.rows.end {
                return Err(DataError::invalid_value(
                    "split range",
                    "must lie inside the source with start not after end",
                ));
            }
        }
        self.contract.splits = splits;
        Ok(self)
    }

    /// Record sampling metadata. This does not reinterpret stored bytes.
    ///
    /// # Errors
    ///
    /// [`TimeIndex::length`] is not the source row count.
    pub fn with_time_index(mut self, time_index: TimeIndex) -> Result<Self, DataError> {
        if time_index.length != self.contract.rows {
            return Err(DataError::LengthMismatch {
                expected: self.contract.rows,
                actual: time_index.length,
                context: "time index length",
            });
        }
        self.contract.time_index = Some(time_index);
        Ok(self)
    }

    /// Declare a validity policy. This does not reinterpret stored bytes.
    ///
    /// # Errors
    ///
    /// [`ValidityPolicy::NullDistinctFromNan`] is named but not adopted: f64 columns are
    /// invalid exactly when the value is NaN. The snapshot contract refuses it too.
    pub fn with_validity(mut self, validity: ValidityPolicy) -> Result<Self, DataError> {
        if matches!(validity, ValidityPolicy::NullDistinctFromNan) {
            return Err(DataError::unsupported(
                "null-distinct-from-NaN validity",
                "it is declared but not adopted; f64 columns are invalid exactly when the value is NaN",
            ));
        }
        self.contract.validity = validity;
        Ok(self)
    }

    /// Projection identity for `columns`. It is not the full-source digest.
    ///
    /// # Errors
    ///
    /// An empty list, a repeated column, or a column id past the schema.
    pub fn project(&self, columns: &[u32]) -> Result<SnapshotContract, DataError> {
        if columns.is_empty() {
            return Err(DataError::empty("column projection variables"));
        }
        let width = u32::try_from(self.contract.schema_names.len()).unwrap_or(u32::MAX);
        for (index, &column) in columns.iter().enumerate() {
            if column >= width {
                return Err(DataError::out_of_range(
                    "projected column",
                    column as usize,
                    width as usize,
                ));
            }
            if columns[..index].contains(&column) {
                return Err(DataError::duplicate("projected column"));
            }
        }
        let SourceIdentity::Full(parent) = self.contract.identity else {
            return Err(DataError::precondition(
                "projecting a source",
                "a full source, not a projection",
            ));
        };
        let mut projected = self.contract.clone();
        projected.identity = SourceIdentity::Projected { parent, columns: fingerprint(columns) };
        Ok(projected)
    }

    /// Row chunk of this source. `end` may equal `rows` (the tail) and may equal `start`.
    ///
    /// # Errors
    ///
    /// A range outside the source, or a source that is not a full identity.
    pub fn chunk(&self, start: usize, end: usize) -> Result<RowChunk, DataError> {
        let SourceIdentity::Full(parent) = self.contract.identity else {
            return Err(DataError::precondition("chunking a source", "a full source"));
        };
        RowChunk::new(parent, self.contract.rows, start, end)
    }
}

/// Panel units or environments, each its own owned source with its own time
/// index, in declared order.
///
/// The parts are not concatenated into one table. Order, the part labels and
/// each part's contract are part of the identity (`SegmentedIdentity`).
#[derive(Clone, Debug)]
pub struct SegmentedSource {
    modality: SourceModality,
    parts: Vec<OwnedSource>,
    /// Panel unit id of each part. `None` for an environment.
    unit_ids: Vec<Option<u32>>,
}

impl SegmentedSource {
    /// One part per panel unit, labeled by `unit_id`.
    #[must_use]
    pub fn from_panel(panel: &PanelData) -> Self {
        let parts = panel
            .units()
            .iter()
            .map(|unit| {
                series_source(&unit.series, SourceModality::Panel)
                    .with_unit(format!("unit:{}", unit.unit_id))
            })
            .collect();
        let unit_ids = panel.units().iter().map(|unit| Some(unit.unit_id)).collect();
        Self { modality: SourceModality::Panel, parts, unit_ids }
    }

    /// One part per environment, labeled by its position.
    #[must_use]
    pub fn from_multi_environment(environments: &MultiEnvironmentData) -> Self {
        let parts: Vec<OwnedSource> = environments
            .environments()
            .iter()
            .enumerate()
            .map(|(index, series)| {
                series_source(series, SourceModality::MultiEnvironment)
                    .with_unit(format!("env:{index}"))
            })
            .collect();
        let unit_ids = vec![None; parts.len()];
        Self { modality: SourceModality::MultiEnvironment, parts, unit_ids }
    }

    /// Modality shared by every part.
    #[must_use]
    pub fn modality(&self) -> SourceModality {
        self.modality
    }

    /// Parts in declared order.
    #[must_use]
    pub fn parts(&self) -> &[OwnedSource] {
        &self.parts
    }

    /// Panel unit id of part `index`; `None` for an environment or an index past the parts.
    #[must_use]
    pub fn unit_id(&self, index: usize) -> Option<u32> {
        self.unit_ids.get(index).copied().flatten()
    }

    /// Total rows across the parts.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.parts.iter().map(|part| part.contract().rows).sum()
    }

    /// Replace part `index` (a setter on one unit or environment).
    ///
    /// # Errors
    ///
    /// `index` is past the parts, or `part` has another modality.
    pub fn with_part(mut self, index: usize, part: OwnedSource) -> Result<Self, DataError> {
        if part.contract().modality != self.modality {
            return Err(DataError::mismatch(
                "replacement part modality and the segmented modality",
            ));
        }
        let Some(slot) = self.parts.get_mut(index) else {
            return Err(DataError::out_of_range("segment", index, self.parts.len()));
        };
        *slot = part;
        Ok(self)
    }
}

fn series_source(series: &TimeSeriesData, modality: SourceModality) -> OwnedSource {
    // Digest the original first so the clone below carries it.
    let _ = series.storage().content_digest();
    let mut source = OwnedSource::from_storage(series.storage().clone(), modality);
    source.contract.time_index = Some(series.time_index().clone());
    source
}

impl SnapshotContract {
    /// Full-source digest, when this contract is not a projection.
    #[must_use]
    pub fn full_digest(&self) -> Option<[u8; 32]> {
        match self.identity {
            SourceIdentity::Full(digest) => Some(digest),
            SourceIdentity::Projected { .. } => None,
        }
    }
}

impl RowChunk {
    fn new(parent: [u8; 32], rows: usize, start: usize, end: usize) -> Result<Self, DataError> {
        if start > end || end > rows {
            return Err(DataError::invalid_value(
                "chunk range",
                "must satisfy start <= end <= row count",
            ));
        }
        Ok(Self { parent, start, end })
    }

    /// Parent full-source digest.
    #[must_use]
    pub fn parent(&self) -> [u8; 32] {
        self.parent
    }

    /// Half-open row range.
    #[must_use]
    pub fn rows(&self) -> Range<usize> {
        self.start..self.end
    }
}

impl BorrowedSource<'_> {
    /// Same contract the owner holds.
    #[must_use]
    pub fn contract(&self) -> &SnapshotContract {
        self.contract
    }

    /// Borrowed storage.
    #[must_use]
    pub fn storage(&self) -> &OwnedColumnarStorage {
        self.storage
    }
}

fn contract_for(
    storage: &OwnedColumnarStorage,
    modality: SourceModality,
    validity: ValidityPolicy,
) -> SnapshotContract {
    let schema_names: Arc<[Arc<str>]> =
        storage.schema().variables().iter().map(|variable| Arc::clone(&variable.name)).collect();
    SnapshotContract {
        modality,
        validity,
        cohort: CohortOrder::Storage,
        unit: None,
        splits: Vec::new(),
        identity: SourceIdentity::Full(storage.content_digest()),
        time_index: None,
        rows: storage.row_count(),
        schema_names,
        schema: schema_digest(storage.schema()),
        event_times: None,
    }
}

/// Digest of a whole schema, in dense order: name, value type, role bits, unit, category
/// domain and measurement of every variable.
#[must_use]
pub fn schema_digest(schema: &CausalSchema) -> [u8; 32] {
    fn text(hasher: &mut blake3::Hasher, value: &str) {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    fn optional(hasher: &mut blake3::Hasher, value: Option<&str>) {
        match value {
            None => {
                hasher.update(&[0]);
            }
            Some(value) => {
                hasher.update(&[1]);
                text(hasher, value);
            }
        }
    }
    let mut hasher = blake3::Hasher::new_derive_key("antecedent.data.source.schema.v1");
    hasher.update(&(schema.len() as u64).to_le_bytes());
    for variable in schema.variables() {
        hasher.update(&variable.id.raw().to_le_bytes());
        text(&mut hasher, &variable.name);
        text(&mut hasher, &format!("{:?}", variable.value_type));
        hasher.update(&variable.role_hints.bits().to_le_bytes());
        optional(&mut hasher, variable.unit.as_deref());
        text(&mut hasher, &format!("{:?}", variable.category_domain));
        optional(&mut hasher, variable.measurement.description.as_deref());
        hasher.update(&[u8::from(variable.measurement.noisy)]);
    }
    *hasher.finalize().as_bytes()
}

/// Digest of event timestamps, in order.
fn event_times_digest(times_ns: &[i64]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key("antecedent.data.source.event_times.v1");
    hasher.update(&(times_ns.len() as u64).to_le_bytes());
    for time in times_ns {
        hasher.update(&time.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

fn fingerprint(columns: &[u32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for column in columns {
        hash ^= u64::from(*column);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash ^ u64::try_from(columns.len()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::dataset::TabularData;
    use crate::storage::OwnedColumnarStorage;

    use super::*;

    fn table(values: &[f64]) -> OwnedColumnarStorage {
        TabularData::from_f64_columns([("x", values)]).unwrap().storage().clone()
    }

    fn weighted(values: &[f64], weights: &[f64]) -> OwnedColumnarStorage {
        let storage = table(values);
        OwnedColumnarStorage::try_new(
            storage.schema().clone(),
            storage.columns().to_vec(),
            None,
            Some(Arc::from(weights)),
        )
        .unwrap()
    }

    #[test]
    fn owned_and_borrowed_share_identity() {
        let owned = OwnedSource::from_storage(table(&[1.0, 2.0, 3.0]), SourceModality::Tabular);
        let borrowed = owned.borrow();
        assert_eq!(owned.contract(), borrowed.contract());
        assert!(matches!(borrowed.contract().identity, SourceIdentity::Full(_)));
        assert_eq!(borrowed.storage().row_count(), 3);
    }

    #[test]
    fn nan_and_weights_change_the_full_digest() {
        let plain = OwnedSource::from_storage(table(&[1.0, 2.0]), SourceModality::Tabular);
        let missing = OwnedSource::from_storage(table(&[1.0, f64::NAN]), SourceModality::Tabular);
        let weighted =
            OwnedSource::from_storage(weighted(&[1.0, 2.0], &[1.0, 1.0]), SourceModality::Tabular);
        assert_ne!(plain.contract().full_digest(), missing.contract().full_digest());
        assert_ne!(plain.contract().full_digest(), weighted.contract().full_digest());
    }

    #[test]
    fn projection_is_not_the_full_identity() {
        let owned = OwnedSource::from_storage(table(&[1.0, 2.0]), SourceModality::Tabular);
        let projected = owned.project(&[0]).unwrap();
        assert!(projected.full_digest().is_none());
        assert_ne!(projected.identity, owned.contract().identity);
        let again = owned.project(&[0]).unwrap();
        assert_eq!(projected, again);
        assert!(owned.project(&[3]).is_err());
        assert!(owned.project(&[]).is_err());
        assert!(owned.project(&[0, 0]).is_err());
    }

    #[test]
    fn chunks_stay_inside_the_source_including_empty_and_tail() {
        let owned = OwnedSource::from_storage(table(&[1.0, 2.0, 3.0]), SourceModality::Panel);
        let tail = owned.chunk(2, 3).unwrap();
        let empty = owned.chunk(1, 1).unwrap();
        assert_eq!(tail.rows(), 2..3);
        assert_eq!(empty.rows(), 1..1);
        assert_eq!(tail.parent(), owned.contract().full_digest().unwrap());
        assert!(owned.chunk(0, 4).is_err());
        assert!(owned.chunk(2, 1).is_err());
    }

    #[test]
    fn splits_cohort_and_validity_are_part_of_the_snapshot() {
        let owned =
            OwnedSource::from_storage(table(&[1.0, 2.0, 3.0, 4.0]), SourceModality::TimeSeries);
        let split = owned
            .clone()
            .with_splits(vec![SplitSpan { label: Arc::from("train"), rows: 0..2 }])
            .unwrap();
        let cohort = owned.clone().with_cohort(CohortOrder::Labeled(Arc::from("arrival")));
        assert!(matches!(
            owned.clone().with_validity(ValidityPolicy::NullDistinctFromNan),
            Err(DataError::Unsupported { .. })
        ));
        let same = owned.clone().with_validity(ValidityPolicy::F64InvalidIffNan).unwrap();
        assert_eq!(owned.contract(), same.contract());
        assert_ne!(owned.contract(), split.contract());
        assert_ne!(owned.contract(), cohort.contract());
        assert_eq!(owned.contract().full_digest(), split.contract().full_digest());
        assert!(
            owned.with_splits(vec![SplitSpan { label: Arc::from("bad"), rows: 0..9 }]).is_err()
        );
    }

    #[test]
    fn time_index_is_part_of_the_contract_not_the_content_digest() {
        use crate::temporal::{SamplingRegularity, TimeIndex};

        let owned = OwnedSource::from_storage(table(&[1.0, 2.0, 3.0]), SourceModality::TimeSeries);
        assert!(owned.contract().time_index.is_none());
        let fast =
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 10 }, length: 3 };
        let slow =
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 11 }, length: 3 };
        let timed = owned.clone().with_time_index(fast.clone()).unwrap();
        let slower = owned.clone().with_time_index(slow).unwrap();
        assert_eq!(owned.contract().full_digest(), timed.contract().full_digest());
        assert_ne!(owned.contract(), timed.contract());
        assert_ne!(timed.contract(), slower.contract());
        assert_eq!(timed.contract().time_index.as_ref(), Some(&fast));
        let short = TimeIndex { regularity: SamplingRegularity::Irregular, length: 2 };
        assert!(owned.with_time_index(short).is_err());
    }
}
