//! Temporal bootstrap / resampling index plans (DESIGN.md §11.4 subset).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::neg_cmp_op_on_partial_ord
)]

use std::sync::Arc;

use causal_core::CausalRng;

use crate::column::{ColumnView, Float64Column, OwnedColumn};
use crate::dataset::TimeSeriesData;
use crate::error::DataError;
use crate::storage::OwnedColumnarStorage;
use crate::table::TableView;

/// Resampling plan producing row-index or weight replicates.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ResamplingPlan {
    /// IID with-replacement row bootstrap.
    IidBootstrap,
    /// Bayesian bootstrap: Dirichlet(1,…,1) observation weights (Rubin).
    BayesianBootstrap,
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

impl ResamplingPlan {
    /// Whether this plan yields observation weights rather than row indexes.
    #[must_use]
    pub const fn is_weight_plan(self) -> bool {
        matches!(self, Self::BayesianBootstrap)
    }
}

/// Fill `out` with a length-`n` row-index plan under `plan`.
///
/// # Errors
///
/// Zero series length, zero block length, or a weight-only plan
/// ([`ResamplingPlan::BayesianBootstrap`]).
pub fn fill_resample_indexes(
    plan: ResamplingPlan,
    n: usize,
    rng: &mut CausalRng,
    out: &mut Vec<u32>,
) -> Result<(), DataError> {
    if n == 0 {
        return Err(DataError::InvalidArgument { message: "resample needs n > 0".into() });
    }
    if plan.is_weight_plan() {
        return Err(DataError::InvalidArgument {
            message: "BayesianBootstrap yields weights; use fill_resample_weights".into(),
        });
    }
    out.clear();
    out.reserve(n);
    match plan {
        ResamplingPlan::IidBootstrap => {
            for _ in 0..n {
                out.push((rng.next_u64() as usize % n) as u32);
            }
        }
        ResamplingPlan::BayesianBootstrap => unreachable!("checked above"),
        ResamplingPlan::MovingBlock { length } | ResamplingPlan::CircularBlock { length } => {
            if length == 0 {
                return Err(DataError::InvalidArgument {
                    message: "block length must be > 0".into(),
                });
            }
            if length > n {
                return Err(DataError::InvalidArgument {
                    message: format!("block length {length} exceeds series length {n}"),
                });
            }
            let circular = matches!(plan, ResamplingPlan::CircularBlock { .. });
            let n_starts =
                if circular { n } else { n.saturating_sub(length).saturating_add(1) };
            debug_assert!(n_starts >= 1);
            while out.len() < n {
                let start = (rng.next_u64() as usize) % n_starts;
                for k in 0..length {
                    if out.len() >= n {
                        break;
                    }
                    let idx = if circular {
                        (start + k) % n
                    } else {
                        // length <= n and start <= n-length, so start+k < n.
                        start + k
                    };
                    out.push(idx as u32);
                }
            }
            out.truncate(n);
        }
    }
    Ok(())
}

/// Fill `out` with length-`n` Bayesian-bootstrap weights (normalized Exp(1) draws).
///
/// Weights sum to `n` (mean weight 1) so weighted estimators match unweighted
/// scale on the original sample size.
///
/// # Errors
///
/// Zero length, or a non-weight plan.
pub fn fill_resample_weights(
    plan: ResamplingPlan,
    n: usize,
    rng: &mut CausalRng,
    out: &mut Vec<f64>,
) -> Result<(), DataError> {
    if n == 0 {
        return Err(DataError::InvalidArgument { message: "resample needs n > 0".into() });
    }
    if !matches!(plan, ResamplingPlan::BayesianBootstrap) {
        return Err(DataError::InvalidArgument {
            message: "fill_resample_weights requires BayesianBootstrap".into(),
        });
    }
    out.clear();
    out.reserve(n);
    let mut sum = 0.0;
    for _ in 0..n {
        // Exp(1) via -ln(U); Dirichlet(1..1) = normalized exponentials.
        let u = rng.next_f64().max(f64::EPSILON);
        let e = -u.ln();
        out.push(e);
        sum += e;
    }
    if !(sum > 0.0) {
        return Err(DataError::InvalidArgument {
            message: "BayesianBootstrap weight sum non-positive".into(),
        });
    }
    let scale = n as f64 / sum;
    for w in out.iter_mut() {
        *w *= scale;
    }
    Ok(())
}

/// Apply a resampling plan to produce a new float64 time series.
///
/// Index plans gather rows. [`ResamplingPlan::BayesianBootstrap`] keeps row
/// order and replaces storage weights with a fresh weight plan (multiplied by
/// any existing weights).
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
    if plan.is_weight_plan() {
        let mut weights = Vec::new();
        fill_resample_weights(plan, n, rng, &mut weights)?;
        return apply_weight_plan(data, &weights);
    }
    fill_resample_indexes(plan, n, rng, index_scratch)?;
    apply_row_map(data, index_scratch)
}

fn apply_weight_plan(data: &TimeSeriesData, weights: &[f64]) -> Result<TimeSeriesData, DataError> {
    let n = data.row_count();
    if weights.len() != n {
        return Err(DataError::LengthMismatch {
            expected: n,
            actual: weights.len(),
            context: "BayesianBootstrap weights",
        });
    }
    let schema = data.schema().clone();
    let mut cols = Vec::with_capacity(schema.len());
    for v in schema.variables() {
        let ColumnView::Float64(src) = data.column(v.id)? else {
            return Err(DataError::TypeMismatch { id: v.id, expected: "float64" });
        };
        cols.push(OwnedColumn::Float64(Float64Column::new(
            v.id,
            Arc::clone(&src.values),
            src.validity.clone(),
        )?));
    }
    let analysis_mask = data.storage().analysis_mask().cloned();
    let combined: Arc<[f64]> = match data.storage().weights() {
        Some(existing) => {
            Arc::from(existing.iter().zip(weights.iter()).map(|(a, b)| a * b).collect::<Vec<_>>())
        }
        None => Arc::from(weights.to_vec()),
    };
    let storage = OwnedColumnarStorage::try_new(schema, cols, analysis_mask, Some(combined))?;
    TimeSeriesData::try_new(storage, data.time_index().clone())
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
    fn bayesian_bootstrap_weights_sum_to_n() {
        let mut rng = CausalRng::from_seed(99);
        let mut w = Vec::new();
        fill_resample_weights(ResamplingPlan::BayesianBootstrap, 50, &mut rng, &mut w).unwrap();
        assert_eq!(w.len(), 50);
        let sum: f64 = w.iter().sum();
        assert!((sum - 50.0).abs() < 1e-9);
        assert!(w.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn bayesian_bootstrap_rejects_index_fill() {
        let mut rng = CausalRng::from_seed(1);
        let mut idx = Vec::new();
        assert!(
            fill_resample_indexes(ResamplingPlan::BayesianBootstrap, 10, &mut rng, &mut idx)
                .is_err()
        );
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
