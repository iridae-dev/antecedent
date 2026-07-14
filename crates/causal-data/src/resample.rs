//! Temporal bootstrap / resampling index plans (DESIGN.md §11.4 subset).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use causal_core::CausalRng;

use crate::column::{ColumnView, Float64Column, OwnedColumn};
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
            let n_starts =
                if circular { n } else { n.saturating_sub(length).saturating_add(1).max(1) };
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
        // A replicate slot inherits the source row's validity: values under
        // null slots must not resurface as valid observations.
        let validity = src.validity.gather(row_map)?;
        cols.push(OwnedColumn::Float64(Float64Column::new(v.id, Arc::from(values), validity)?));
    }
    // Analysis mask and weights follow the same row map as the values.
    let analysis_mask = data.storage().analysis_mask().map(|m| m.gather(row_map)).transpose()?;
    let weights = data
        .storage()
        .weights()
        .map(|w| Arc::from(row_map.iter().map(|&r| w[r as usize]).collect::<Vec<f64>>()));
    let storage = OwnedColumnarStorage::try_new(schema, cols, analysis_mask, weights)?;
    TimeSeriesData::try_new(storage, data.time_index().clone())
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use causal_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
    };

    use super::*;
    use crate::column::ValidityBitmap;
    use crate::temporal::{SamplingRegularity, TimeIndex};
    use crate::testing::float_series;

    /// One-variable series of length 4 with row 2 invalid, a mask hiding row 1,
    /// and per-row weights.
    fn series_with_missing() -> TimeSeriesData {
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "v0".to_owned(),
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        // Row 2 invalid (LSB-first bits: rows 0,1,3 set).
        let validity = ValidityBitmap::from_bytes(vec![0b1011u8], 4).unwrap();
        let col = OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(0),
                Arc::from(vec![10.0, 11.0, 0.0, 13.0]),
                validity,
            )
            .unwrap(),
        );
        // Analysis mask hides row 1 (LSB-first bits: rows 0,2,3 set).
        let mask = ValidityBitmap::from_bytes(vec![0b1101u8], 4).unwrap();
        let weights: Arc<[f64]> = Arc::from(vec![1.0, 2.0, 3.0, 4.0]);
        let storage =
            OwnedColumnarStorage::try_new(schema, vec![col], Some(mask), Some(weights)).unwrap();
        TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: 4 },
        )
        .unwrap()
    }

    #[test]
    fn row_map_gathers_validity_mask_and_weights() {
        let data = series_with_missing();
        let out = apply_row_map(&data, &[2, 0, 2, 1]).unwrap();
        let ColumnView::Float64(col) = out.column(VariableId::from_raw(0)).unwrap() else {
            panic!("expected float64 column");
        };
        // Replicate slots sourced from row 2 must stay invalid.
        assert!(!col.validity.is_valid(0));
        assert!(col.validity.is_valid(1));
        assert!(!col.validity.is_valid(2));
        assert!(col.validity.is_valid(3));
        // No invalid source value appears as a valid replicate value.
        for i in 0..4 {
            if col.validity.is_valid(i) {
                assert!(
                    (col.values[i] - 10.0).abs() < 1e-12 || (col.values[i] - 11.0).abs() < 1e-12
                );
            }
        }
        // Analysis mask follows the row map (source row 1 was hidden).
        let mask = out.storage().analysis_mask().unwrap();
        assert!(mask.is_valid(0));
        assert!(mask.is_valid(1));
        assert!(mask.is_valid(2));
        assert!(!mask.is_valid(3));
        // Weights follow the row map.
        assert_eq!(out.storage().weights().unwrap(), &[3.0, 1.0, 3.0, 2.0]);
    }

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
