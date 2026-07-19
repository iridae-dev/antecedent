//! Irregular event-indexed datasets.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{CausalSchema, VariableId};

use crate::column::ColumnView;
use crate::error::DataError;
use crate::storage::OwnedColumnarStorage;
use crate::table::TableView;
use crate::temporal::SamplingRegularity;

/// Event-indexed dataset with irregular event times and mark columns.
///
/// Integer lag algorithms must not treat event indices as regular time steps
/// (§5.4). Use duration windows, explicit alignment, or native event models.
#[derive(Clone, Debug)]
pub struct EventData {
    storage: OwnedColumnarStorage,
    /// Event timestamps in nanoseconds (non-decreasing, length = row count).
    event_times_ns: Arc<[i64]>,
}

impl EventData {
    /// Construct from mark/covariate storage and event timestamps.
    ///
    /// # Errors
    ///
    /// Length mismatch, empty data, or timestamps that are not non-decreasing.
    pub fn try_new(
        storage: OwnedColumnarStorage,
        event_times_ns: impl Into<Arc<[i64]>>,
    ) -> Result<Self, DataError> {
        let event_times_ns = event_times_ns.into();
        let n = storage.row_count();
        if event_times_ns.len() != n {
            return Err(DataError::LengthMismatch {
                expected: n,
                actual: event_times_ns.len(),
                context: "event times",
            });
        }
        if n == 0 {
            return Err(DataError::InvalidArgument {
                message: "event data requires ≥1 event".into(),
            });
        }
        for w in event_times_ns.windows(2) {
            if w[1] < w[0] {
                return Err(DataError::InvalidArgument {
                    message: "event times must be non-decreasing".into(),
                });
            }
        }
        Ok(Self { storage, event_times_ns })
    }

    /// Event timestamps (nanoseconds).
    #[must_use]
    pub fn event_times_ns(&self) -> &[i64] {
        &self.event_times_ns
    }

    /// Borrow mark / covariate storage.
    #[must_use]
    pub fn storage(&self) -> &OwnedColumnarStorage {
        &self.storage
    }

    /// Sampling regularity is always irregular for event data.
    #[must_use]
    pub const fn regularity(&self) -> SamplingRegularity {
        SamplingRegularity::Irregular
    }

    /// Reject integer-lag sample planning (lags are not durations on irregular data).
    ///
    /// # Errors
    ///
    /// Always returns [`DataError::InvalidArgument`].
    pub fn reject_integer_lag_planning(&self) -> Result<(), DataError> {
        Err(DataError::InvalidArgument {
            message: "integer lags are not valid on EventData; use duration windows or event models"
                .into(),
        })
    }
}

impl TableView for EventData {
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use causal_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
    };

    use super::*;
    use crate::column::{Float64Column, OwnedColumn, ValidityBitmap};
    use crate::table::TableView;

    fn one_col(n: usize) -> OwnedColumnarStorage {
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "mark",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let col = Float64Column::new(
            VariableId::from_raw(0),
            Arc::<[f64]>::from(vec![1.0; n]),
            ValidityBitmap::all_valid(n),
        )
        .unwrap();
        OwnedColumnarStorage::try_new(schema, vec![OwnedColumn::Float64(col)], None, None).unwrap()
    }

    #[test]
    fn event_data_round_trip() {
        let times = Arc::<[i64]>::from(vec![0, 10, 10, 25]);
        let data = EventData::try_new(one_col(4), times).unwrap();
        assert_eq!(data.row_count(), 4);
        assert_eq!(data.event_times_ns(), &[0, 10, 10, 25]);
        assert_eq!(data.regularity(), SamplingRegularity::Irregular);
        assert!(data.reject_integer_lag_planning().is_err());
    }

    #[test]
    fn rejects_decreasing_times() {
        let err = EventData::try_new(one_col(3), Arc::<[i64]>::from(vec![0, 5, 4]));
        assert!(err.is_err());
    }
}
