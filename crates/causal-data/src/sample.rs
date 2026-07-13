//! Lag-aligned sample planning for temporal discovery (DESIGN.md §5.5).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use causal_core::{Lag, VariableId};

use crate::column::ColumnView;
use crate::dataset::TimeSeriesData;
use crate::error::DataError;
use crate::reference::ReferencePointPolicy;
use crate::table::TableView;

/// Lag-alignment cache for a regular series of length `series_len`.
///
/// For `max_lag = τ_max` under [`ReferencePointPolicy::SeriesOrigin`], effective
/// samples are times `t = τ_max .. series_len-1` (`n = series_len - τ_max`).
/// Sample `i` at lag `τ` reads raw row `base_t + i - τ`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LagMap {
    series_len: usize,
    max_lag: u32,
    n_effective: usize,
    base_t: usize,
    reference: ReferencePointPolicy,
}

impl LagMap {
    /// Build a lag map with the default series-origin reference policy.
    ///
    /// # Errors
    ///
    /// Empty series, or `max_lag >= series_len`.
    pub fn new(series_len: usize, max_lag: u32) -> Result<Self, DataError> {
        Self::with_reference(series_len, max_lag, ReferencePointPolicy::SeriesOrigin)
    }

    /// Build a lag map under an explicit reference-point policy.
    ///
    /// # Errors
    ///
    /// Empty series, invalid lag, or origin out of range.
    pub fn with_reference(
        series_len: usize,
        max_lag: u32,
        reference: ReferencePointPolicy,
    ) -> Result<Self, DataError> {
        let (base_t, n_effective) = reference.base_and_n(series_len, max_lag)?;
        Ok(Self { series_len, max_lag, n_effective, base_t, reference })
    }

    /// Series length.
    #[must_use]
    pub const fn series_len(&self) -> usize {
        self.series_len
    }

    /// Configured maximum lag.
    #[must_use]
    pub const fn max_lag(&self) -> u32 {
        self.max_lag
    }

    /// Number of aligned samples.
    #[must_use]
    pub const fn n_effective(&self) -> usize {
        self.n_effective
    }

    /// Reference-point policy used to build this map.
    #[must_use]
    pub const fn reference(&self) -> ReferencePointPolicy {
        self.reference
    }

    /// Raw row index for sample `i` at the given lag.
    ///
    /// # Panics
    ///
    /// Panics if `i >= n_effective` or `lag.raw() > max_lag`.
    #[must_use]
    pub fn row_index(&self, lag: Lag, sample_i: usize) -> usize {
        debug_assert!(sample_i < self.n_effective);
        debug_assert!(lag.raw() <= self.max_lag);
        self.base_t + sample_i - lag.raw() as usize
    }

    /// Fill `out` with raw row indexes for `lag` (`out.len()` must equal `n_effective`).
    ///
    /// # Errors
    ///
    /// Length mismatch or lag exceeding `max_lag`.
    pub fn fill_row_indexes(&self, lag: Lag, out: &mut [u32]) -> Result<(), DataError> {
        if out.len() != self.n_effective {
            return Err(DataError::LengthMismatch {
                expected: self.n_effective,
                actual: out.len(),
                context: "lag-map row index buffer",
            });
        }
        if lag.raw() > self.max_lag {
            return Err(DataError::InvalidValidity { message: "lag exceeds max_lag" });
        }
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = u32::try_from(self.row_index(lag, i)).expect("row fits u32 for series_len");
        }
        Ok(())
    }
}

/// One planned lagged column.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct LaggedColumn {
    /// Variable.
    pub variable: VariableId,
    /// Lag relative to the contemporaneous sample time.
    pub lag: Lag,
}

/// Reusable plan for gathering a fixed set of lagged columns.
#[derive(Clone, Debug)]
pub struct SamplePlan {
    columns: Arc<[LaggedColumn]>,
    lag_map: LagMap,
}

impl SamplePlan {
    /// Plan lagged columns for a series length / max lag (series-origin reference).
    ///
    /// # Errors
    ///
    /// Invalid lag map, empty column list, or a column lag exceeding `max_lag`.
    pub fn new(
        series_len: usize,
        max_lag: u32,
        columns: impl Into<Arc<[LaggedColumn]>>,
    ) -> Result<Self, DataError> {
        Self::with_reference(series_len, max_lag, ReferencePointPolicy::SeriesOrigin, columns)
    }

    /// Plan lagged columns under an explicit reference-point policy.
    ///
    /// # Errors
    ///
    /// Invalid lag map, empty column list, or a column lag exceeding `max_lag`.
    pub fn with_reference(
        series_len: usize,
        max_lag: u32,
        reference: ReferencePointPolicy,
        columns: impl Into<Arc<[LaggedColumn]>>,
    ) -> Result<Self, DataError> {
        let columns = columns.into();
        if columns.is_empty() {
            return Err(DataError::InvalidValidity { message: "sample plan needs ≥1 column" });
        }
        let lag_map = LagMap::with_reference(series_len, max_lag, reference)?;
        for c in columns.iter() {
            if c.lag.raw() > max_lag {
                return Err(DataError::InvalidValidity {
                    message: "planned column lag exceeds max_lag",
                });
            }
        }
        Ok(Self { columns, lag_map })
    }

    /// Planned columns.
    #[must_use]
    pub fn columns(&self) -> &[LaggedColumn] {
        &self.columns
    }

    /// Lag map.
    #[must_use]
    pub fn lag_map(&self) -> &LagMap {
        &self.lag_map
    }

    /// Effective sample size.
    #[must_use]
    pub fn n_effective(&self) -> usize {
        self.lag_map.n_effective
    }

    /// Gather planned columns into `workspace` (grows once, then reuses capacity).
    ///
    /// # Errors
    ///
    /// Missing / non-float64 columns, or series length mismatch.
    pub fn prepare<'a>(
        &'a self,
        data: &TimeSeriesData,
        workspace: &'a mut SampleWorkspace,
    ) -> Result<PreparedSample<'a>, DataError> {
        if data.row_count() != self.lag_map.series_len {
            return Err(DataError::LengthMismatch {
                expected: self.lag_map.series_len,
                actual: data.row_count(),
                context: "time series length vs sample plan",
            });
        }
        let n = self.lag_map.n_effective;
        let ncols = self.columns.len();
        workspace.prepare(n, ncols);

        for (c, col) in self.columns.iter().enumerate() {
            let ColumnView::Float64(src) = data.column(col.variable)? else {
                return Err(DataError::TypeMismatch { id: col.variable, expected: "float64" });
            };
            self.lag_map.fill_row_indexes(col.lag, &mut workspace.row_indexes)?;
            let dst = &mut workspace.values[c * n..(c + 1) * n];
            for (j, &row) in workspace.row_indexes.iter().enumerate() {
                dst[j] = src.values[row as usize];
            }
        }

        Ok(PreparedSample {
            n,
            ncols,
            values: &workspace.values[..n * ncols],
            columns: &self.columns,
            dropped: DropSummary { requested: self.lag_map.series_len, retained: n },
        })
    }
}

/// Caller-owned scratch for repeated [`SamplePlan::prepare`] calls.
#[derive(Clone, Debug, Default)]
pub struct SampleWorkspace {
    /// Reused row-index buffer (length = `n_effective`).
    pub row_indexes: Vec<u32>,
    /// Column-major gathered values (`ncols * n`).
    pub values: Vec<f64>,
    capacity_n: usize,
    capacity_cols: usize,
}

impl SampleWorkspace {
    /// Ensure capacity for `n` rows and `ncols` columns (grows, never shrinks).
    pub fn prepare(&mut self, n: usize, ncols: usize) {
        if self.row_indexes.len() < n {
            self.row_indexes.resize(n, 0);
        }
        let need = n.saturating_mul(ncols);
        if self.values.len() < need {
            self.values.resize(need, 0.0);
        }
        self.capacity_n = self.capacity_n.max(n);
        self.capacity_cols = self.capacity_cols.max(ncols);
    }

    /// Peak row capacity retained.
    #[must_use]
    pub const fn capacity_n(&self) -> usize {
        self.capacity_n
    }

    /// Peak column capacity retained.
    #[must_use]
    pub const fn capacity_cols(&self) -> usize {
        self.capacity_cols
    }
}

/// Rows dropped vs retained when aligning lags.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DropSummary {
    /// Raw series length.
    pub requested: usize,
    /// Effective aligned samples.
    pub retained: usize,
}

/// Borrowed prepared sample (views into a [`SampleWorkspace`]).
#[derive(Clone, Copy, Debug)]
pub struct PreparedSample<'a> {
    /// Effective sample size.
    pub n: usize,
    /// Number of lagged columns.
    pub ncols: usize,
    /// Column-major values (`ncols` blocks of length `n`).
    pub values: &'a [f64],
    /// Column descriptors aligned with `values` blocks.
    pub columns: &'a [LaggedColumn],
    /// Drop summary.
    pub dropped: DropSummary,
}

impl PreparedSample<'_> {
    /// Borrow column `c` as a contiguous slice of length `n`.
    #[must_use]
    pub fn column(&self, c: usize) -> &[f64] {
        &self.values[c * self.n..(c + 1) * self.n]
    }
}

impl TimeSeriesData {
    /// Plan a lagged sample for discovery / CI queries.
    ///
    /// # Errors
    ///
    /// Propagates [`SamplePlan::new`] errors.
    pub fn plan_lagged_sample(
        &self,
        max_lag: u32,
        columns: impl Into<Arc<[LaggedColumn]>>,
    ) -> Result<SamplePlan, DataError> {
        SamplePlan::new(self.row_count(), max_lag, columns)
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
mod tests {
    use std::sync::Arc;

    use causal_core::{Lag, VariableId};

    use super::*;
    use crate::testing::float_series;

    #[test]
    fn lag_map_row_indexes() {
        let map = LagMap::new(10, 2).unwrap();
        assert_eq!(map.n_effective(), 8);
        assert_eq!(map.row_index(Lag::CONTEMPORANEOUS, 0), 2);
        assert_eq!(map.row_index(Lag::from_raw(2), 0), 0);
        assert_eq!(map.row_index(Lag::from_raw(1), 3), 2 + 3 - 1);
    }

    #[test]
    fn prepare_gathers_lagged_values() {
        let data = float_series(20, 2);
        let cols = Arc::from([
            LaggedColumn { variable: VariableId::from_raw(0), lag: Lag::CONTEMPORANEOUS },
            LaggedColumn { variable: VariableId::from_raw(0), lag: Lag::from_raw(2) },
            LaggedColumn { variable: VariableId::from_raw(1), lag: Lag::from_raw(1) },
        ]);
        let plan = data.plan_lagged_sample(2, cols).unwrap();
        let mut ws = SampleWorkspace::default();
        let prep = plan.prepare(&data, &mut ws).unwrap();
        assert_eq!(prep.n, 18);
        assert!((prep.column(0)[0] - 2.0).abs() < 1e-12);
        assert!((prep.column(1)[0] - 0.0).abs() < 1e-12);
        assert!((prep.column(2)[0] - 101.0).abs() < 1e-12);
    }

    #[test]
    fn repeated_prepare_reuses_workspace_capacity() {
        let data = float_series(100, 3);
        let cols = Arc::from([
            LaggedColumn { variable: VariableId::from_raw(0), lag: Lag::CONTEMPORANEOUS },
            LaggedColumn { variable: VariableId::from_raw(1), lag: Lag::from_raw(3) },
            LaggedColumn { variable: VariableId::from_raw(2), lag: Lag::from_raw(1) },
        ]);
        let plan = data.plan_lagged_sample(3, cols).unwrap();
        let mut ws = SampleWorkspace::default();
        let _ = plan.prepare(&data, &mut ws).unwrap();
        let cap_n = ws.capacity_n();
        let cap_c = ws.capacity_cols();
        let values_cap = ws.values.capacity();
        let idx_cap = ws.row_indexes.capacity();
        for _ in 0..50 {
            let _ = plan.prepare(&data, &mut ws).unwrap();
            assert_eq!(ws.capacity_n(), cap_n);
            assert_eq!(ws.capacity_cols(), cap_c);
            assert_eq!(ws.values.capacity(), values_cap);
            assert_eq!(ws.row_indexes.capacity(), idx_cap);
        }
    }
}
