//! Internal source and snapshot contract over owned storage.
//!
//! A snapshot binds modality, cohort order, validity policy, units, splits, an
//! optional time index, and the existing full-content digest (columns, mask,
//! and weights). A projection does not reuse that digest. A chunk is a row
//! range of the same source, not a new population.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

// A materialized file-backed table (`file_source`) is snapshotted here with
// `from_storage` and addresses row ranges with `chunk`. The other modalities and
// setters are part of the contract; their preparation call sites are not wired
// yet, and only the unit tests use them.
#![allow(dead_code)]

use std::ops::Range;
use std::sync::Arc;

use crate::error::DataError;
use crate::storage::OwnedColumnarStorage;
use crate::table::TableView;
use crate::temporal::TimeIndex;

/// How the rows are organized. This does not change the stored columns.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SourceModality {
    /// One row per unit, no time index on the snapshot.
    Tabular,
    /// Panel units.
    Panel,
    /// Several environments.
    MultiEnvironment,
    /// A time series.
    TimeSeries,
}

/// What invalid means for the values this snapshot points at.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ValidityPolicy {
    /// Current f64 rule: a value is invalid exactly when it is NaN.
    F64InvalidIffNan,
    /// Declared so a later adapter can tell null from NaN. Not a kernel view.
    NullDistinctFromNan,
}

/// Row order relative to storage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum CohortOrder {
    /// Dense row `i` is storage row `i`.
    Storage,
    /// Named order. The label is part of the snapshot; the columns are not permuted here.
    Labeled(Arc<str>),
}

/// A named row span that belongs to the snapshot (a split, not a chunk).
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SplitSpan {
    /// Split name.
    pub label: Arc<str>,
    /// Half-open row range in storage order.
    pub rows: Range<usize>,
}

/// Full-source digest, or a projection that must not be compared as that digest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SourceIdentity {
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
pub(crate) struct SnapshotContract {
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
}

/// Owned immutable source.
#[derive(Clone, Debug)]
pub(crate) struct OwnedSource {
    storage: OwnedColumnarStorage,
    contract: SnapshotContract,
}

/// Borrow of an owned source. The borrow does not expose mutation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BorrowedSource<'a> {
    storage: &'a OwnedColumnarStorage,
    contract: &'a SnapshotContract,
}

/// A row range of a source. Empty and tail ranges are allowed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RowChunk {
    parent: [u8; 32],
    start: usize,
    end: usize,
}

impl OwnedSource {
    /// Snapshot `storage` under the current f64 validity rule and storage order.
    #[must_use]
    pub(crate) fn from_storage(storage: OwnedColumnarStorage, modality: SourceModality) -> Self {
        let contract = contract_for(&storage, modality, ValidityPolicy::F64InvalidIffNan);
        Self { storage, contract }
    }

    /// Borrow the snapshot.
    #[must_use]
    pub(crate) fn borrow(&self) -> BorrowedSource<'_> {
        BorrowedSource { storage: &self.storage, contract: &self.contract }
    }

    /// Contract.
    #[must_use]
    pub(crate) fn contract(&self) -> &SnapshotContract {
        &self.contract
    }

    /// Replace cohort order. Content digest is unchanged.
    #[must_use]
    pub(crate) fn with_cohort(mut self, cohort: CohortOrder) -> Self {
        self.contract.cohort = cohort;
        self
    }

    /// Replace the unit label.
    #[must_use]
    pub(crate) fn with_unit(mut self, unit: impl Into<Arc<str>>) -> Self {
        self.contract.unit = Some(unit.into());
        self
    }

    /// Attach splits. Ranges must lie inside the source.
    ///
    /// # Errors
    ///
    /// A split endpoint past the row count.
    pub(crate) fn with_splits(mut self, splits: Vec<SplitSpan>) -> Result<Self, DataError> {
        for split in &splits {
            if split.rows.end > self.contract.rows || split.rows.start > split.rows.end {
                return Err(DataError::InvalidArgument {
                    message: format!("split '{}' is outside the source", split.label),
                });
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
    pub(crate) fn with_time_index(mut self, time_index: TimeIndex) -> Result<Self, DataError> {
        if time_index.length != self.contract.rows {
            return Err(DataError::InvalidArgument {
                message: format!(
                    "time index length {} is not the source row count {}",
                    time_index.length, self.contract.rows
                ),
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
    pub(crate) fn with_validity(mut self, validity: ValidityPolicy) -> Result<Self, DataError> {
        if matches!(validity, ValidityPolicy::NullDistinctFromNan) {
            return Err(DataError::InvalidArgument {
                message: "null-distinct-from-NaN validity is declared but not adopted; f64 columns are invalid exactly when the value is NaN".into(),
            });
        }
        self.contract.validity = validity;
        Ok(self)
    }

    /// Projection identity for `columns`. It is not the full-source digest.
    ///
    /// # Errors
    ///
    /// An empty list, a repeated column, or a column id past the schema.
    pub(crate) fn project(&self, columns: &[u32]) -> Result<SnapshotContract, DataError> {
        if columns.is_empty() {
            return Err(DataError::InvalidArgument {
                message: "a column projection names no variables and is not the full source".into(),
            });
        }
        let width = u32::try_from(self.contract.schema_names.len()).unwrap_or(u32::MAX);
        for (index, &column) in columns.iter().enumerate() {
            if column >= width {
                return Err(DataError::InvalidArgument {
                    message: format!("projected column {column} is outside the schema"),
                });
            }
            if columns[..index].contains(&column) {
                return Err(DataError::InvalidArgument {
                    message: "column projection repeats a variable".into(),
                });
            }
        }
        let SourceIdentity::Full(parent) = self.contract.identity else {
            return Err(DataError::InvalidArgument {
                message: "project a full source, not a projection".into(),
            });
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
    pub(crate) fn chunk(&self, start: usize, end: usize) -> Result<RowChunk, DataError> {
        let SourceIdentity::Full(parent) = self.contract.identity else {
            return Err(DataError::InvalidArgument {
                message: "chunks address a full source".into(),
            });
        };
        RowChunk::new(parent, self.contract.rows, start, end)
    }
}

impl SnapshotContract {
    /// Full-source digest, when this contract is not a projection.
    #[must_use]
    pub(crate) fn full_digest(&self) -> Option<[u8; 32]> {
        match self.identity {
            SourceIdentity::Full(digest) => Some(digest),
            SourceIdentity::Projected { .. } => None,
        }
    }
}

impl RowChunk {
    fn new(parent: [u8; 32], rows: usize, start: usize, end: usize) -> Result<Self, DataError> {
        if start > end || end > rows {
            return Err(DataError::InvalidArgument {
                message: format!("chunk {start}..{end} is outside 0..{rows}"),
            });
        }
        Ok(Self { parent, start, end })
    }

    /// Parent full-source digest.
    #[must_use]
    pub(crate) fn parent(&self) -> [u8; 32] {
        self.parent
    }

    /// Half-open row range.
    #[must_use]
    pub(crate) fn rows(&self) -> Range<usize> {
        self.start..self.end
    }
}

impl BorrowedSource<'_> {
    /// Same contract the owner holds.
    #[must_use]
    pub(crate) fn contract(&self) -> &SnapshotContract {
        self.contract
    }

    /// Borrowed storage.
    #[must_use]
    pub(crate) fn storage(&self) -> &OwnedColumnarStorage {
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
    }
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
            Err(DataError::InvalidArgument { .. })
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
