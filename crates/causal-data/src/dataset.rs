//! Concrete tabular and time-series dataset types.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{CausalSchema, VariableId};

use crate::column::{ColumnView, ValidityBitmap};
#[cfg(test)]
use crate::column::OwnedColumn;
use crate::error::DataError;
use crate::storage::OwnedColumnarStorage;
use crate::table::TableView;
use crate::temporal::TimeIndex;

/// Tabular IID dataset.
#[derive(Clone, Debug)]
pub struct TabularData {
    storage: OwnedColumnarStorage,
}

impl TabularData {
    /// Wrap owned storage.
    #[must_use]
    pub fn new(storage: OwnedColumnarStorage) -> Self {
        Self { storage }
    }

    /// Borrow underlying storage.
    #[must_use]
    pub fn storage(&self) -> &OwnedColumnarStorage {
        &self.storage
    }
}

impl TableView for TabularData {
    fn schema(&self) -> &CausalSchema {
        self.storage.schema()
    }

    fn row_count(&self) -> usize {
        self.storage.row_count()
    }

    fn column(&self, id: VariableId) -> Result<ColumnView<'_>, DataError> {
        self.storage.column(id)
    }
}

/// Temporal / time-series dataset with time index metadata.
#[derive(Clone, Debug)]
pub struct TimeSeriesData {
    storage: OwnedColumnarStorage,
    time_index: TimeIndex,
}

impl TimeSeriesData {
    /// Construct ensuring time index length matches row count.
    ///
    /// # Errors
    ///
    /// Length mismatch.
    pub fn try_new(
        storage: OwnedColumnarStorage,
        time_index: TimeIndex,
    ) -> Result<Self, DataError> {
        if time_index.length != storage.row_count() {
            return Err(DataError::LengthMismatch {
                expected: storage.row_count(),
                actual: time_index.length,
                context: "time index",
            });
        }
        Ok(Self { storage, time_index })
    }

    /// Time index metadata.
    #[must_use]
    pub fn time_index(&self) -> &TimeIndex {
        &self.time_index
    }

    /// Borrow storage.
    #[must_use]
    pub fn storage(&self) -> &OwnedColumnarStorage {
        &self.storage
    }

    /// Pointer identity of the columnar Arc (tests: planning must not clone payloads).
    #[cfg(test)]
    #[must_use]
    pub(crate) fn columnar_ptr(&self) -> *const [OwnedColumn] {
        Arc::as_ptr(self.storage.columns_arc())
    }

    /// Restrict analysis to rows where `mask` is valid, intersected (AND) with any existing
    /// analysis mask; preserves columns, validity, weights, and time index.
    ///
    /// # Errors
    ///
    /// Mask length mismatch.
    pub fn with_analysis_mask(&self, mask: ValidityBitmap) -> Result<Self, DataError> {
        let storage = &self.storage;
        let n = storage.row_count();
        if mask.len() != n {
            return Err(DataError::LengthMismatch {
                expected: n,
                actual: mask.len(),
                context: "analysis mask",
            });
        }
        let combined = match storage.analysis_mask() {
            Some(existing) => {
                let mut bytes = vec![0u8; n.div_ceil(8)];
                for i in 0..n {
                    if existing.is_valid(i) && mask.is_valid(i) {
                        bytes[i / 8] |= 1 << (i % 8);
                    }
                }
                ValidityBitmap::from_bytes(bytes, n)?
            }
            None => mask,
        };
        let new_storage = OwnedColumnarStorage::try_new(
            storage.schema().clone(),
            storage.columns().to_vec(),
            Some(combined),
            storage.weights().map(Arc::from),
        )?;
        Self::try_new(new_storage, self.time_index.clone())
    }
}

impl TableView for TimeSeriesData {
    fn schema(&self) -> &CausalSchema {
        self.storage.schema()
    }

    fn row_count(&self) -> usize {
        self.storage.row_count()
    }

    fn column(&self, id: VariableId) -> Result<ColumnView<'_>, DataError> {
        self.storage.column(id)
    }
}
