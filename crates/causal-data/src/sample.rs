//! Lag-aligned sample planning for temporal discovery (DESIGN.md §5.5).
//!
//! General x/y/z [`SampleRequest`](crate::SampleRequest) planning lives in
//! [`sample_request`](crate::sample_request). This module owns the lag-gather
//! hot path ([`LaggedSamplePlan`]).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use causal_core::{KernelPolicy, Lag, VariableId};
use causal_kernels::{F64VectorView, gather};

use crate::column::{ColumnView, Float64Column};
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
    pub fn fill_row_indexes(&self, lag: Lag, out: &mut [usize]) -> Result<(), DataError> {
        if out.len() != self.n_effective {
            return Err(DataError::LengthMismatch {
                expected: self.n_effective,
                actual: out.len(),
                context: "lag-map row index buffer",
            });
        }
        if lag.raw() > self.max_lag {
            return Err(DataError::InvalidArgument { message: "lag exceeds max_lag".into() });
        }
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.row_index(lag, i);
        }
        Ok(())
    }
}

/// Reject datasets whose analysis mask hides rows: lag gathers index raw rows,
/// so temporal discovery requires the full contiguous series.
pub(crate) fn ensure_unmasked(data: &TimeSeriesData) -> Result<(), DataError> {
    if let Some(mask) = data.storage().analysis_mask() {
        if !mask.is_all_valid() {
            return Err(DataError::IncompleteSeries {
                id: None,
                message: "analysis mask hides rows; temporal discovery requires complete series",
            });
        }
    }
    Ok(())
}

/// Reject float columns with missing values before a lag gather (values under
/// null slots are sentinels and must never be consumed).
pub(crate) fn ensure_complete_float(src: &Float64Column) -> Result<(), DataError> {
    if !src.validity.is_all_valid() {
        return Err(DataError::IncompleteSeries {
            id: Some(src.id),
            message: "missing values in series; temporal discovery requires complete series",
        });
    }
    Ok(())
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
pub struct LaggedSamplePlan {
    columns: Arc<[LaggedColumn]>,
    lag_map: Arc<LagMap>,
}

impl LaggedSamplePlan {
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
            return Err(DataError::InvalidArgument {
                message: "sample plan needs ≥1 column".into(),
            });
        }
        let lag_map = Arc::new(LagMap::with_reference(series_len, max_lag, reference)?);
        Self::validate_columns(lag_map.max_lag(), &columns)?;
        Ok(Self { columns, lag_map })
    }

    /// Build a plan that reuses a shared lag map and column list (multi-env / panel).
    ///
    /// # Errors
    ///
    /// Empty column list, or a column lag exceeding the shared map's `max_lag`.
    pub fn with_shared(
        lag_map: Arc<LagMap>,
        columns: Arc<[LaggedColumn]>,
    ) -> Result<Self, DataError> {
        if columns.is_empty() {
            return Err(DataError::InvalidArgument {
                message: "sample plan needs ≥1 column".into(),
            });
        }
        Self::validate_columns(lag_map.max_lag(), &columns)?;
        Ok(Self { columns, lag_map })
    }

    fn validate_columns(max_lag: u32, columns: &[LaggedColumn]) -> Result<(), DataError> {
        for c in columns {
            if c.lag.raw() > max_lag {
                return Err(DataError::InvalidArgument {
                    message: "planned column lag exceeds max_lag".into(),
                });
            }
        }
        Ok(())
    }

    /// Planned columns.
    #[must_use]
    pub fn columns(&self) -> &[LaggedColumn] {
        &self.columns
    }

    /// Shared column Arc (for multi-env plan reuse).
    #[must_use]
    pub fn columns_arc(&self) -> &Arc<[LaggedColumn]> {
        &self.columns
    }

    /// Lag map.
    #[must_use]
    pub fn lag_map(&self) -> &LagMap {
        &self.lag_map
    }

    /// Shared lag-map Arc (identical lengths can share one map).
    #[must_use]
    pub fn lag_map_arc(&self) -> &Arc<LagMap> {
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
    /// Missing / non-float64 columns, series length mismatch, or incomplete
    /// series (missing values or a row-hiding analysis mask).
    pub fn prepare<'a>(
        &'a self,
        data: &TimeSeriesData,
        workspace: &'a mut LaggedSampleWorkspace,
    ) -> Result<LaggedPreparedSample<'a>, DataError> {
        if data.row_count() != self.lag_map.series_len {
            return Err(DataError::LengthMismatch {
                expected: self.lag_map.series_len,
                actual: data.row_count(),
                context: "time series length vs sample plan",
            });
        }
        ensure_unmasked(data)?;
        let n = self.lag_map.n_effective;
        let ncols = self.columns.len();
        workspace.prepare(n, ncols);
        let policy = KernelPolicy::default_policy();

        for (c, col) in self.columns.iter().enumerate() {
            let ColumnView::Float64(src) = data.column(col.variable)? else {
                return Err(DataError::TypeMismatch { id: col.variable, expected: "float64" });
            };
            ensure_complete_float(src)?;
            self.lag_map.fill_row_indexes(col.lag, &mut workspace.row_indexes[..n])?;
            let dst = &mut workspace.values[c * n..(c + 1) * n];
            gather(
                &policy,
                F64VectorView::contiguous(src.values.as_slice()),
                &workspace.row_indexes[..n],
                dst,
            );
        }

        Ok(LaggedPreparedSample {
            n,
            ncols,
            values: &workspace.values[..n * ncols],
            columns: &self.columns,
            dropped: DropSummary { requested: self.lag_map.series_len, retained: n },
        })
    }
}

/// Caller-owned scratch for repeated [`LaggedSamplePlan::prepare`] calls.
#[derive(Clone, Debug, Default)]
pub struct LaggedSampleWorkspace {
    /// Reused row-index buffer (length = `n_effective`).
    pub row_indexes: Vec<usize>,
    /// Column-major gathered values (`ncols * n`).
    pub values: Vec<f64>,
    capacity_n: usize,
    capacity_cols: usize,
}

impl LaggedSampleWorkspace {
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

/// Borrowed prepared sample (views into a [`LaggedSampleWorkspace`]).
#[derive(Clone, Copy, Debug)]
pub struct LaggedPreparedSample<'a> {
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

impl LaggedPreparedSample<'_> {
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
    /// Propagates [`LaggedSamplePlan::new`] errors.
    pub fn plan_lagged_sample(
        &self,
        max_lag: u32,
        columns: impl Into<Arc<[LaggedColumn]>>,
    ) -> Result<LaggedSamplePlan, DataError> {
        LaggedSamplePlan::new(self.row_count(), max_lag, columns)
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
mod tests {
    use std::sync::Arc;

    use causal_core::{Lag, VariableId};

    use super::*;
    use crate::testing::{float_series, float_series_with_gap, float_series_with_mask};

    #[test]
    fn prepare_rejects_missing_values() {
        let data = float_series_with_gap(20, 1, 3);
        let cols = Arc::from([LaggedColumn {
            variable: VariableId::from_raw(0),
            lag: Lag::CONTEMPORANEOUS,
        }]);
        let plan = data.plan_lagged_sample(2, cols).unwrap();
        let mut ws = LaggedSampleWorkspace::default();
        let err = plan.prepare(&data, &mut ws).unwrap_err();
        assert!(matches!(
            err,
            DataError::IncompleteSeries { id: Some(v), .. } if v == VariableId::from_raw(0)
        ));
    }

    #[test]
    fn prepare_rejects_row_hiding_analysis_mask() {
        let data = float_series_with_mask(20, 1, 3);
        let cols = Arc::from([LaggedColumn {
            variable: VariableId::from_raw(0),
            lag: Lag::CONTEMPORANEOUS,
        }]);
        let plan = data.plan_lagged_sample(2, cols).unwrap();
        let mut ws = LaggedSampleWorkspace::default();
        let err = plan.prepare(&data, &mut ws).unwrap_err();
        assert!(matches!(err, DataError::IncompleteSeries { id: None, .. }));
    }

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
        let mut ws = LaggedSampleWorkspace::default();
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
        let mut ws = LaggedSampleWorkspace::default();
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
