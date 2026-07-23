//! Pre-materialized lagged columns for temporal discovery.
//!
//! Built once per discovery run so candidate CI tests index columns without
//! rebuilding lag alignment or re-gathering series. Optional analysis masks and
//! column missingness are recorded per lagged column so masked MCI can keep
//! complete-case windows without requiring a fully contiguous unmasked series.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use antecedent_core::{KernelPolicy, Lag, VariableId};
use antecedent_kernels::{F64VectorView, gather};

use crate::column::{ColumnView, ValidityBitmap};
use crate::dataset::TimeSeriesData;
use crate::error::DataError;
use crate::reference::ReferencePointPolicy;
use crate::sample::LagMap;
use crate::sample_policy::{MaskPolicy, MissingPolicy};
use crate::table::TableView;

/// Options for lag-aligned frame materialization.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct LaggedFrameOptions {
    /// How the optional analysis mask is applied when recording column validity.
    pub mask: MaskPolicy,
    /// How column missingness is recorded (`CompleteCase` marks invalid; `ErrorOnMissing` fails).
    pub missing: MissingPolicy,
}

impl Default for LaggedFrameOptions {
    fn default() -> Self {
        Self { mask: MaskPolicy::Honor, missing: MissingPolicy::CompleteCase }
    }
}

/// Pre-materialized lagged frame: one contiguous column per `(variable, lag)`.
///
/// Layout is column-major over slots `variable_slot * (max_lag + 1) + lag`.
/// Per-column [`ValidityBitmap`]s record whether each effective row's source
/// sample passed analysis-mask ∩ column-validity (under the build options).
#[derive(Clone, Debug)]
pub struct LaggedFrame {
    variables: Arc<[VariableId]>,
    max_lag: u32,
    n_effective: usize,
    n_lags: usize,
    /// Column-major values: `n_cols * n_effective`.
    values: Vec<f64>,
    /// Per lagged-column validity (`ncols` bitmaps, each length `n_effective`).
    validity: Vec<ValidityBitmap>,
}

impl LaggedFrame {
    /// Materialize all lags `0..=max_lag` for `variables` from `data`.
    ///
    /// Uses [`LaggedFrameOptions::default`] (Honor analysis mask + `CompleteCase`).
    ///
    /// # Errors
    ///
    /// Empty variable list, invalid lag map, missing/non-float64 columns, or
    /// `ErrorOnMissing` when a source column has nulls.
    pub fn from_series(
        data: &TimeSeriesData,
        variables: &[VariableId],
        max_lag: u32,
        policy: &KernelPolicy,
    ) -> Result<Self, DataError> {
        Self::from_series_with_options(
            data,
            variables,
            max_lag,
            ReferencePointPolicy::SeriesOrigin,
            LaggedFrameOptions::default(),
            policy,
        )
    }

    /// Materialize lagged columns under an explicit reference-point policy.
    ///
    /// # Errors
    ///
    /// Empty variable list, invalid lag map, missing/non-float64 columns, or
    /// `ErrorOnMissing` when a source column has nulls.
    pub fn from_series_with_reference(
        data: &TimeSeriesData,
        variables: &[VariableId],
        max_lag: u32,
        reference: ReferencePointPolicy,
        policy: &KernelPolicy,
    ) -> Result<Self, DataError> {
        Self::from_series_with_options(
            data,
            variables,
            max_lag,
            reference,
            LaggedFrameOptions::default(),
            policy,
        )
    }

    /// Materialize lagged columns with explicit mask / missingness policies.
    ///
    /// # Errors
    ///
    /// Empty variable list, invalid lag map, missing/non-float64 columns, or
    /// `ErrorOnMissing` when a source column has nulls.
    pub fn from_series_with_options(
        data: &TimeSeriesData,
        variables: &[VariableId],
        max_lag: u32,
        reference: ReferencePointPolicy,
        options: LaggedFrameOptions,
        policy: &KernelPolicy,
    ) -> Result<Self, DataError> {
        if variables.is_empty() {
            return Err(DataError::InvalidArgument {
                message: "lagged frame needs ≥1 variable".into(),
            });
        }
        let lag_map = LagMap::with_reference(data.row_count(), max_lag, reference)?;
        let n_effective = lag_map.n_effective();
        let n_lags = max_lag as usize + 1;
        let n_cols = variables.len().saturating_mul(n_lags);
        let mut values = vec![0.0; n_cols.saturating_mul(n_effective)];
        let mut validity = Vec::with_capacity(n_cols);

        // Row indexes depend only on the lag: compute each lag's gather once.
        let mut lag_rows = vec![vec![0usize; n_effective]; n_lags];
        for (lag, rows) in lag_rows.iter_mut().enumerate() {
            lag_map.fill_row_indexes(Lag::from_raw(lag as u32), rows)?;
        }

        let analysis = data.storage().analysis_mask();
        for (slot, &var) in variables.iter().enumerate() {
            let ColumnView::Float64(src) = data.column(var)? else {
                return Err(DataError::TypeMismatch { id: var, expected: "float64" });
            };
            if options.missing == MissingPolicy::ErrorOnMissing && !src.validity.is_all_valid() {
                return Err(DataError::IncompleteSeries {
                    id: Some(src.id),
                    message: "missing values under ErrorOnMissing policy",
                });
            }
            if options.mask == MaskPolicy::Honor && options.missing == MissingPolicy::ErrorOnMissing
            {
                if let Some(mask) = analysis {
                    if !mask.is_all_valid() {
                        return Err(DataError::IncompleteSeries {
                            id: None,
                            message: "analysis mask hides rows under ErrorOnMissing policy",
                        });
                    }
                }
            }
            let src_view = F64VectorView::contiguous(src.values.as_slice());
            for (lag, rows) in lag_rows.iter().enumerate() {
                let col = slot * n_lags + lag;
                let dst = &mut values[col * n_effective..(col + 1) * n_effective];
                gather(policy, src_view, rows, dst);
                let col_valid = gather_column_validity(src, analysis, options.mask, rows)?;
                validity.push(col_valid);
            }
        }

        Ok(Self { variables: Arc::from(variables), max_lag, n_effective, n_lags, values, validity })
    }

    /// Variables in slot order.
    #[must_use]
    pub fn variables(&self) -> &[VariableId] {
        &self.variables
    }

    /// Maximum lag materialised (inclusive).
    #[must_use]
    pub const fn max_lag(&self) -> u32 {
        self.max_lag
    }

    /// Effective aligned sample count.
    #[must_use]
    pub const fn n_effective(&self) -> usize {
        self.n_effective
    }

    /// Number of columns (`n_vars * (max_lag + 1)`).
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.variables.len().saturating_mul(self.n_lags)
    }

    /// Whether every lagged column is fully valid (fast path for unmasked series).
    #[must_use]
    pub fn is_fully_valid(&self) -> bool {
        self.validity.iter().all(ValidityBitmap::is_all_valid)
    }

    /// Byte size of the gathered value buffer.
    #[must_use]
    pub fn values_bytes(&self) -> u64 {
        (self.values.len() * core::mem::size_of::<f64>()) as u64
    }

    /// Dense column index for `(variable, lag)`, or `None` if unknown / out of range.
    #[must_use]
    pub fn column_index(&self, variable: VariableId, lag: Lag) -> Option<usize> {
        let slot = self.variables.iter().position(|&v| v == variable)?;
        let l = lag.raw() as usize;
        if l >= self.n_lags {
            return None;
        }
        Some(slot * self.n_lags + l)
    }

    /// Borrow column at dense index.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= ncols`.
    #[must_use]
    pub fn column(&self, idx: usize) -> &[f64] {
        let n = self.n_effective;
        &self.values[idx * n..(idx + 1) * n]
    }

    /// Borrow per-row validity for lagged column `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= ncols`.
    #[must_use]
    pub fn column_valid(&self, idx: usize) -> &ValidityBitmap {
        &self.validity[idx]
    }

    /// Keep-mask for a set of lagged column indexes: AND of their validity bitmaps.
    ///
    /// # Errors
    ///
    /// Empty column list or out-of-range index.
    pub fn keep_mask_for_columns(&self, cols: &[usize]) -> Result<Vec<bool>, DataError> {
        if cols.is_empty() {
            return Err(DataError::InvalidArgument {
                message: "keep_mask_for_columns needs ≥1 column".into(),
            });
        }
        let mut keep = vec![true; self.n_effective];
        for &c in cols {
            if c >= self.ncols() {
                return Err(DataError::InvalidArgument {
                    message: format!("keep_mask_for_columns: column {c} out of range"),
                });
            }
            let v = &self.validity[c];
            for (i, slot) in keep.iter_mut().enumerate() {
                if *slot && !v.is_valid(i) {
                    *slot = false;
                }
            }
        }
        Ok(keep)
    }

    /// Compact to effective rows where `keep[i]` is true.
    ///
    /// Used for regime-masked CI: the full series is materialized first so lag
    /// alignment is correct, then only windows wholly inside a regime are retained.
    ///
    /// # Errors
    ///
    /// Length mismatch or empty keep set.
    pub fn retain_effective(&self, keep: &[bool]) -> Result<Self, DataError> {
        if keep.len() != self.n_effective {
            return Err(DataError::InvalidArgument {
                message: format!(
                    "retain_effective keep length {} != n_effective {}",
                    keep.len(),
                    self.n_effective
                ),
            });
        }
        let n_new = keep.iter().filter(|&&k| k).count();
        if n_new == 0 {
            return Err(DataError::InvalidArgument {
                message: "retain_effective: no effective rows retained".into(),
            });
        }
        let n_cols = self.ncols();
        let mut values = vec![0.0; n_cols.saturating_mul(n_new)];
        let mut validity = Vec::with_capacity(n_cols);
        for c in 0..n_cols {
            let src = self.column(c);
            let dst = &mut values[c * n_new..(c + 1) * n_new];
            let mut j = 0;
            for (i, &k) in keep.iter().enumerate() {
                if k {
                    dst[j] = src[i];
                    j += 1;
                }
            }
            validity.push(self.validity[c].compact(keep)?);
        }
        Ok(Self {
            variables: Arc::clone(&self.variables),
            max_lag: self.max_lag,
            n_effective: n_new,
            n_lags: self.n_lags,
            values,
            validity,
        })
    }

    /// Vertically stack frames that share the same variable list and max lag.
    ///
    /// Used for multi-environment pooling without lag windows crossing env boundaries.
    ///
    /// # Errors
    ///
    /// Empty input, or mismatched variables / `max_lag` across frames.
    pub fn stack(frames: &[Self]) -> Result<Self, DataError> {
        let Some(first) = frames.first() else {
            return Err(DataError::InvalidArgument {
                message: "LaggedFrame::stack needs ≥1 frame".into(),
            });
        };
        for (i, f) in frames.iter().enumerate().skip(1) {
            if f.variables.as_ref() != first.variables.as_ref() {
                return Err(DataError::InvalidArgument {
                    message: format!("LaggedFrame::stack: variables mismatch at frame {i}"),
                });
            }
            if f.max_lag != first.max_lag || f.n_lags != first.n_lags {
                return Err(DataError::InvalidArgument {
                    message: format!("LaggedFrame::stack: max_lag mismatch at frame {i}"),
                });
            }
        }
        let n_eff: usize = frames.iter().map(Self::n_effective).sum();
        if n_eff == 0 {
            return Err(DataError::InvalidArgument {
                message: "LaggedFrame::stack: zero effective rows".into(),
            });
        }
        let n_cols = first.ncols();
        let mut values = vec![0.0; n_cols.saturating_mul(n_eff)];
        let mut validity = Vec::with_capacity(n_cols);
        for c in 0..n_cols {
            let mut offset = 0usize;
            for f in frames {
                let src = f.column(c);
                let dst = &mut values[c * n_eff + offset..c * n_eff + offset + f.n_effective];
                dst.copy_from_slice(src);
                offset += f.n_effective;
            }
            let parts: Vec<&ValidityBitmap> = frames.iter().map(|f| &f.validity[c]).collect();
            validity.push(ValidityBitmap::concat(&parts)?);
        }
        Ok(Self {
            variables: Arc::clone(&first.variables),
            max_lag: first.max_lag,
            n_effective: n_eff,
            n_lags: first.n_lags,
            values,
            validity,
        })
    }

    /// Append variables whose values are constant across lags (space/time dummies).
    ///
    /// Each entry is `(variable_id, contemporaneous column)` of length `n_effective`.
    /// The same values are copied into every lag slot so MCI can index any lag;
    /// link assumptions should still forbid lagged parents of space/time dummies.
    /// Appended columns are marked all-valid.
    ///
    /// # Errors
    ///
    /// Length mismatch, duplicate variable id, or empty column list.
    pub fn append_constant_lag_columns(
        &self,
        columns: &[(VariableId, Vec<f64>)],
    ) -> Result<Self, DataError> {
        if columns.is_empty() {
            return Ok(self.clone());
        }
        let mut vars = self.variables.to_vec();
        for (id, col) in columns {
            if col.len() != self.n_effective {
                return Err(DataError::InvalidArgument {
                    message: format!(
                        "append_constant_lag_columns: column len {} != n_effective {}",
                        col.len(),
                        self.n_effective
                    ),
                });
            }
            if vars.contains(id) {
                return Err(DataError::InvalidArgument {
                    message: format!("append_constant_lag_columns: duplicate variable {id}"),
                });
            }
            vars.push(*id);
        }
        let n_eff = self.n_effective;
        let n_lags = self.n_lags;
        let old_cols = self.ncols();
        let new_slots = columns.len();
        let n_cols = old_cols + new_slots * n_lags;
        let mut values = vec![0.0; n_cols.saturating_mul(n_eff)];
        values[..old_cols * n_eff].copy_from_slice(&self.values);
        let mut validity = self.validity.clone();
        for (s, (_id, col)) in columns.iter().enumerate() {
            for lag in 0..n_lags {
                let c = old_cols + s * n_lags + lag;
                values[c * n_eff..(c + 1) * n_eff].copy_from_slice(col);
                validity.push(ValidityBitmap::all_valid(n_eff));
            }
        }
        Ok(Self {
            variables: Arc::from(vars),
            max_lag: self.max_lag,
            n_effective: n_eff,
            n_lags,
            values,
            validity,
        })
    }
}

fn gather_column_validity(
    src: &crate::column::Float64Column,
    analysis: Option<&ValidityBitmap>,
    mask_policy: MaskPolicy,
    rows: &[usize],
) -> Result<ValidityBitmap, DataError> {
    let col_valid = src.validity.gather_rows(rows)?;
    match (mask_policy, analysis) {
        (MaskPolicy::Ignore, _) | (MaskPolicy::Honor, None) => Ok(col_valid),
        (MaskPolicy::Honor, Some(mask)) => {
            let mask_valid = mask.gather_rows(rows)?;
            let n = rows.len();
            let mut bytes = vec![0u8; n.div_ceil(8)];
            for i in 0..n {
                if col_valid.is_valid(i) && mask_valid.is_valid(i) {
                    bytes[i / 8] |= 1 << (i % 8);
                }
            }
            ValidityBitmap::from_bytes(bytes, n)
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
mod tests {
    use antecedent_core::{Lag, VariableId};

    use super::*;
    use crate::sample_policy::{MaskPolicy, MissingPolicy};
    use crate::testing::{float_series, float_series_with_gap, float_series_with_mask};

    #[test]
    fn builds_with_missing_values_marking_invalid() {
        let data = float_series_with_gap(20, 2, 5);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let frame = LaggedFrame::from_series(
            &data,
            &vars,
            2,
            &antecedent_core::KernelPolicy::default_policy(),
        )
        .unwrap();
        assert_eq!(frame.n_effective(), 18);
        // Row index 5 in the series is invalid for v0; lag-0 effective index = 5 - 2 = 3.
        let i0 = frame.column_index(vars[0], Lag::CONTEMPORANEOUS).unwrap();
        assert!(!frame.column_valid(i0).is_valid(3));
        assert!(frame.column_valid(i0).is_valid(0));
        let i1 = frame.column_index(vars[1], Lag::CONTEMPORANEOUS).unwrap();
        assert!(frame.column_valid(i1).is_all_valid());
    }

    #[test]
    fn builds_with_analysis_mask_marking_invalid() {
        let data = float_series_with_mask(20, 2, 5);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let frame = LaggedFrame::from_series(
            &data,
            &vars,
            2,
            &antecedent_core::KernelPolicy::default_policy(),
        )
        .unwrap();
        let i0 = frame.column_index(vars[0], Lag::CONTEMPORANEOUS).unwrap();
        assert!(!frame.column_valid(i0).is_valid(3));
        let keep = frame.keep_mask_for_columns(&[i0]).unwrap();
        assert!(!keep[3]);
        assert!(keep[0]);
    }

    #[test]
    fn ignore_mask_keeps_analysis_hidden_rows_valid() {
        let data = float_series_with_mask(20, 2, 5);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let frame = LaggedFrame::from_series_with_options(
            &data,
            &vars,
            2,
            ReferencePointPolicy::SeriesOrigin,
            LaggedFrameOptions { mask: MaskPolicy::Ignore, missing: MissingPolicy::CompleteCase },
            &KernelPolicy::default_policy(),
        )
        .unwrap();
        assert!(frame.is_fully_valid());
    }

    #[test]
    fn error_on_missing_rejects_gaps() {
        let data = float_series_with_gap(20, 2, 5);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let err = LaggedFrame::from_series_with_options(
            &data,
            &vars,
            2,
            ReferencePointPolicy::SeriesOrigin,
            LaggedFrameOptions { mask: MaskPolicy::Honor, missing: MissingPolicy::ErrorOnMissing },
            &KernelPolicy::default_policy(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            DataError::IncompleteSeries { id: Some(v), .. } if v == VariableId::from_raw(0)
        ));
    }

    #[test]
    fn frame_matches_lag_map_gather() {
        let data = float_series(20, 2);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let frame = LaggedFrame::from_series(
            &data,
            &vars,
            2,
            &antecedent_core::KernelPolicy::default_policy(),
        )
        .unwrap();
        assert_eq!(frame.n_effective(), 18);
        assert_eq!(frame.ncols(), 6);
        assert!(frame.is_fully_valid());
        let i = frame.column_index(vars[0], Lag::CONTEMPORANEOUS).unwrap();
        assert!((frame.column(i)[0] - 2.0).abs() < 1e-12);
        let j = frame.column_index(vars[1], Lag::from_raw(1)).unwrap();
        assert!((frame.column(j)[0] - 101.0).abs() < 1e-12);
    }

    #[test]
    fn retain_effective_compacts_validity() {
        let data = float_series_with_mask(20, 2, 5);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let frame = LaggedFrame::from_series(
            &data,
            &vars,
            2,
            &antecedent_core::KernelPolicy::default_policy(),
        )
        .unwrap();
        let cols: Vec<usize> = (0..frame.ncols()).collect();
        let keep = frame.keep_mask_for_columns(&cols).unwrap();
        let compacted = frame.retain_effective(&keep).unwrap();
        assert!(compacted.is_fully_valid());
        assert_eq!(compacted.n_effective(), keep.iter().filter(|&&k| k).count());
    }
}
