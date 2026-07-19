//! Irregular event-indexed datasets.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use std::sync::Arc;

use causal_core::{CausalSchema, VariableId};

use crate::column::{
    BooleanColumn, ColumnView, FixedVectorColumn, Float64Column, Int64Column, OwnedColumn,
    TimestampColumn, ValidityBitmap,
};
use crate::dataset::TimeSeriesData;
use crate::error::DataError;
use crate::storage::OwnedColumnarStorage;
use crate::table::TableView;
use crate::temporal::{SamplingRegularity, TimeIndex};

/// Event-indexed dataset with irregular event times and mark columns.
///
/// Integer lag algorithms must not treat event indices as regular time steps
/// (§5.4). Use [`Self::align_to_grid`] (duration bins), or native event models.
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

    /// Align irregular events onto a regular grid via duration bins.
    ///
    /// Bins are half-open intervals `[t0 + k·Δ, t0 + (k+1)·Δ)` covering
    /// `[t0, t_last]` where `t0` / `t_last` are the first / last event times and
    /// `Δ = interval_ns`. Within each bin the **last** event's mark values are
    /// kept. Empty bins are marked invalid (value zeroed).
    ///
    /// The result is a regular [`TimeSeriesData`] suitable for integer-lag
    /// temporal discovery / identification / estimation.
    ///
    /// # Errors
    ///
    /// Zero interval, empty events, or column construction failures.
    pub fn align_to_grid(&self, interval_ns: u64) -> Result<TimeSeriesData, DataError> {
        if interval_ns == 0 {
            return Err(DataError::InvalidArgument {
                message: "align_to_grid requires interval_ns > 0".into(),
            });
        }
        let times = self.event_times_ns.as_ref();
        let t0 = times[0];
        let t_last = times[times.len() - 1];
        let span = (t_last - t0) as u64;
        let n_bins = (span / interval_ns) as usize + 1;

        // Last event index per bin (None = empty).
        let mut last_in_bin: Vec<Option<usize>> = vec![None; n_bins];
        for (i, &t) in times.iter().enumerate() {
            let bin = ((t - t0) as u64 / interval_ns) as usize;
            let bin = bin.min(n_bins - 1);
            last_in_bin[bin] = Some(i);
        }

        let mut out_cols = Vec::with_capacity(self.storage.columns().len());
        for col in self.storage.columns() {
            out_cols.push(align_column(col, &last_in_bin, n_bins)?);
        }

        let mut analysis_mask = None;
        if let Some(mask) = self.storage.analysis_mask() {
            let mut bytes = vec![0u8; n_bins.div_ceil(8)];
            for (b, &src) in last_in_bin.iter().enumerate() {
                if let Some(i) = src {
                    if mask.is_valid(i) {
                        bytes[b / 8] |= 1 << (b % 8);
                    }
                }
            }
            analysis_mask = Some(ValidityBitmap::from_bytes(bytes, n_bins)?);
        }

        let weights = self.storage.weights().map(|w| {
            let mut out = vec![0.0_f64; n_bins];
            for (b, &src) in last_in_bin.iter().enumerate() {
                if let Some(i) = src {
                    out[b] = w[i];
                }
            }
            Arc::<[f64]>::from(out)
        });

        let storage = OwnedColumnarStorage::try_new(
            self.storage.schema().clone(),
            out_cols,
            analysis_mask,
            weights,
        )?;
        TimeSeriesData::try_new(
            storage,
            TimeIndex {
                regularity: SamplingRegularity::Regular { interval_ns },
                length: n_bins,
            },
        )
    }
}

fn align_column(
    col: &OwnedColumn,
    last_in_bin: &[Option<usize>],
    n_bins: usize,
) -> Result<OwnedColumn, DataError> {
    let mut valid_bytes = vec![0u8; n_bins.div_ceil(8)];
    match col {
        OwnedColumn::Float64(c) => {
            let mut values = vec![0.0_f64; n_bins];
            for (b, &src) in last_in_bin.iter().enumerate() {
                if let Some(i) = src {
                    if c.validity.is_valid(i) {
                        values[b] = c.values.as_slice()[i];
                        valid_bytes[b / 8] |= 1 << (b % 8);
                    }
                }
            }
            Ok(OwnedColumn::Float64(Float64Column::new(
                c.id,
                Arc::<[f64]>::from(values),
                ValidityBitmap::from_bytes(valid_bytes, n_bins)?,
            )?))
        }
        OwnedColumn::Int64(c) => {
            let mut values = vec![0_i64; n_bins];
            for (b, &src) in last_in_bin.iter().enumerate() {
                if let Some(i) = src {
                    if c.validity.is_valid(i) {
                        values[b] = c.values[i];
                        valid_bytes[b / 8] |= 1 << (b % 8);
                    }
                }
            }
            Ok(OwnedColumn::Int64(Int64Column::new(
                c.id,
                Arc::<[i64]>::from(values),
                ValidityBitmap::from_bytes(valid_bytes, n_bins)?,
            )?))
        }
        OwnedColumn::Boolean(c) => {
            let mut values = vec![0u8; n_bins];
            for (b, &src) in last_in_bin.iter().enumerate() {
                if let Some(i) = src {
                    if c.validity.is_valid(i) {
                        values[b] = c.values[i];
                        valid_bytes[b / 8] |= 1 << (b % 8);
                    }
                }
            }
            Ok(OwnedColumn::Boolean(BooleanColumn::new(
                c.id,
                Arc::<[u8]>::from(values),
                ValidityBitmap::from_bytes(valid_bytes, n_bins)?,
            )?))
        }
        OwnedColumn::Timestamp(c) => {
            let mut values = vec![0_i64; n_bins];
            for (b, &src) in last_in_bin.iter().enumerate() {
                if let Some(i) = src {
                    if c.validity.is_valid(i) {
                        values[b] = c.values_ns[i];
                        valid_bytes[b / 8] |= 1 << (b % 8);
                    }
                }
            }
            Ok(OwnedColumn::Timestamp(TimestampColumn::new(
                c.id,
                Arc::<[i64]>::from(values),
                ValidityBitmap::from_bytes(valid_bytes, n_bins)?,
            )?))
        }
        OwnedColumn::FixedVector(c) => {
            let dim = c.dim;
            let mut values = vec![0.0_f64; n_bins * dim];
            for (b, &src) in last_in_bin.iter().enumerate() {
                if let Some(i) = src {
                    if c.validity.is_valid(i) {
                        let src_off = i * dim;
                        let dst_off = b * dim;
                        values[dst_off..dst_off + dim]
                            .copy_from_slice(&c.values[src_off..src_off + dim]);
                        valid_bytes[b / 8] |= 1 << (b % 8);
                    }
                }
            }
            Ok(OwnedColumn::FixedVector(FixedVectorColumn::new(
                c.id,
                dim,
                Arc::<[f64]>::from(values),
                ValidityBitmap::from_bytes(valid_bytes, n_bins)?,
            )?))
        }
        OwnedColumn::Categorical(_) => Err(DataError::InvalidArgument {
            message: "align_to_grid does not support categorical mark columns".into(),
        }),
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

    #[test]
    fn align_to_grid_last_event_in_bin() {
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
        // Events at 0, 5, 15 with marks 1, 2, 3 → bins of 10ns: [0,10), [10,20]
        // bin0 last=5→2.0, bin1 last=15→3.0
        let col = Float64Column::new(
            VariableId::from_raw(0),
            Arc::<[f64]>::from(vec![1.0, 2.0, 3.0]),
            ValidityBitmap::all_valid(3),
        )
        .unwrap();
        let storage =
            OwnedColumnarStorage::try_new(schema, vec![OwnedColumn::Float64(col)], None, None)
                .unwrap();
        let data = EventData::try_new(storage, Arc::<[i64]>::from(vec![0, 5, 15])).unwrap();
        let series = data.align_to_grid(10).unwrap();
        assert_eq!(series.row_count(), 2);
        assert_eq!(
            series.time_index().regularity,
            SamplingRegularity::Regular { interval_ns: 10 }
        );
        let ColumnView::Float64(c) = series.column(VariableId::from_raw(0)).unwrap() else {
            panic!("expected float");
        };
        assert!((c.values.as_slice()[0] - 2.0).abs() < f64::EPSILON);
        assert!((c.values.as_slice()[1] - 3.0).abs() < f64::EPSILON);
        assert!(c.validity.is_valid(0) && c.validity.is_valid(1));
    }

    #[test]
    fn align_to_grid_marks_empty_bins_invalid() {
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
        // Events at 0 and 20 with Δ=10 → bins [0,10), [10,20), [20,30): middle empty
        let col = Float64Column::new(
            VariableId::from_raw(0),
            Arc::<[f64]>::from(vec![1.0, 9.0]),
            ValidityBitmap::all_valid(2),
        )
        .unwrap();
        let storage =
            OwnedColumnarStorage::try_new(schema, vec![OwnedColumn::Float64(col)], None, None)
                .unwrap();
        let data = EventData::try_new(storage, Arc::<[i64]>::from(vec![0, 20])).unwrap();
        let series = data.align_to_grid(10).unwrap();
        assert_eq!(series.row_count(), 3);
        let ColumnView::Float64(c) = series.column(VariableId::from_raw(0)).unwrap() else {
            panic!("expected float");
        };
        assert!(c.validity.is_valid(0));
        assert!(!c.validity.is_valid(1));
        assert!(c.validity.is_valid(2));
        assert!((c.values.as_slice()[2] - 9.0).abs() < f64::EPSILON);
    }

    #[test]
    fn align_rejects_zero_interval() {
        let data = EventData::try_new(one_col(2), Arc::<[i64]>::from(vec![0, 1])).unwrap();
        assert!(data.align_to_grid(0).is_err());
    }
}
