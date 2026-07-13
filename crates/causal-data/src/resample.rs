//! Temporal bootstrap / resampling index plans (DESIGN.md §11.4 subset).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use causal_core::CausalRng;

use crate::column::{ColumnView, Float64Column, OwnedColumn, ValidityBitmap};
use crate::dataset::TimeSeriesData;
use crate::error::DataError;
use crate::storage::OwnedColumnarStorage;
use crate::table::TableView;

/// Resampling plan producing row-index replicates.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ResamplingPlan {
    /// IID with-replacement row bootstrap.
    IidBootstrap,
    /// Moving-block bootstrap (contiguous blocks).
    MovingBlock {
        /// Block length in rows.
        length: usize,
    },
    /// Circular block bootstrap (wraps the series).
    CircularBlock {
        /// Block length in rows.
        length: usize,
    },
}

/// Fill `out` with a length-`n` row-index plan under `plan`.
///
/// # Errors
///
/// Zero series length or zero block length.
pub fn fill_resample_indexes(
    plan: ResamplingPlan,
    n: usize,
    rng: &mut CausalRng,
    out: &mut Vec<u32>,
) -> Result<(), DataError> {
    if n == 0 {
        return Err(DataError::InvalidValidity { message: "resample needs n > 0" });
    }
    out.clear();
    out.reserve(n);
    match plan {
        ResamplingPlan::IidBootstrap => {
            for _ in 0..n {
                out.push((rng.next_u64() as usize % n) as u32);
            }
        }
        ResamplingPlan::MovingBlock { length } | ResamplingPlan::CircularBlock { length } => {
            if length == 0 {
                return Err(DataError::InvalidValidity { message: "block length must be > 0" });
            }
            let circular = matches!(plan, ResamplingPlan::CircularBlock { .. });
            let n_starts = if circular {
                n
            } else {
                n.saturating_sub(length).saturating_add(1).max(1)
            };
            while out.len() < n {
                let start = (rng.next_u64() as usize) % n_starts;
                for k in 0..length {
                    if out.len() >= n {
                        break;
                    }
                    let idx = if circular {
                        (start + k) % n
                    } else if start + k < n {
                        start + k
                    } else {
                        n - 1
                    };
                    out.push(idx as u32);
                }
            }
            out.truncate(n);
        }
    }
    Ok(())
}

/// Apply a resampling plan to produce a new float64 time series.
///
/// # Errors
///
/// Non-float columns or construction failures.
pub fn resample_timeseries(
    data: &TimeSeriesData,
    plan: ResamplingPlan,
    rng: &mut CausalRng,
    index_scratch: &mut Vec<u32>,
) -> Result<TimeSeriesData, DataError> {
    let n = data.row_count();
    fill_resample_indexes(plan, n, rng, index_scratch)?;
    apply_row_map(data, index_scratch)
}

fn apply_row_map(data: &TimeSeriesData, row_map: &[u32]) -> Result<TimeSeriesData, DataError> {
    let n = row_map.len();
    if n != data.row_count() {
        return Err(DataError::LengthMismatch {
            expected: data.row_count(),
            actual: n,
            context: "resample row map",
        });
    }
    let schema = data.schema().clone();
    let mut cols = Vec::with_capacity(schema.len());
    for v in schema.variables() {
        let ColumnView::Float64(src) = data.column(v.id)? else {
            return Err(DataError::TypeMismatch { id: v.id, expected: "float64" });
        };
        let values: Vec<f64> = row_map.iter().map(|&r| src.values[r as usize]).collect();
        cols.push(OwnedColumn::Float64(Float64Column::new(
            v.id,
            Arc::from(values),
            ValidityBitmap::all_valid(n),
        )?));
    }
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None)?;
    TimeSeriesData::try_new(storage, data.time_index().clone())
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use crate::testing::float_series;

    #[test]
    fn moving_block_preserves_length() {
        let data = float_series(100, 1);
        let mut rng = CausalRng::from_seed(1);
        let mut idx = Vec::new();
        let out = resample_timeseries(
            &data,
            ResamplingPlan::MovingBlock { length: 10 },
            &mut rng,
            &mut idx,
        )
        .unwrap();
        assert_eq!(out.row_count(), 100);
        assert_eq!(idx.len(), 100);
    }
}
