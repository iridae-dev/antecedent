//! Concrete tabular and time-series dataset types (DESIGN.md §5.1).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{CausalSchema, VariableId};

use crate::column::{ColumnView, OwnedColumn};
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
    #[must_use]
    pub fn columnar_ptr(&self) -> *const [OwnedColumn] {
        Arc::as_ptr(self.storage.columns_arc())
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
