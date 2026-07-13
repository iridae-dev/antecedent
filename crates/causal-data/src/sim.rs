//! Structural process generators for conformance / toys (Phase 2).
//!
//! Full SCM mechanism registry is Phase 7; this module covers lagged linear
//! processes with known parents for fixtures and notebooks.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::sync::Arc;

use causal_core::{
    CausalSchemaBuilder, Lag, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
};

use crate::column::{Float64Column, OwnedColumn, ValidityBitmap};
use crate::dataset::TimeSeriesData;
use crate::error::DataError;
use crate::storage::OwnedColumnarStorage;
use crate::temporal::{SamplingRegularity, TimeIndex};

/// Known lagged parent used as ground truth in fixtures.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct KnownLaggedParent {
    /// Source variable index into the generator’s variable list.
    pub source: VariableId,
    /// Source lag.
    pub source_lag: Lag,
    /// Target variable.
    pub target: VariableId,
}

/// Two-variable lagged linear SCM: `Y_t = coef * X_{t-lag} + noise`.
#[derive(Clone, Debug)]
pub struct LaggedLinearPair {
    /// Series length.
    pub n: usize,
    /// Coupling coefficient.
    pub coef: f64,
    /// Lag of X → Y.
    pub lag: u32,
    /// Master seed for deterministic noise streams.
    pub seed: u64,
}

impl Default for LaggedLinearPair {
    fn default() -> Self {
        Self { n: 500, coef: 0.8, lag: 1, seed: 1 }
    }
}

impl LaggedLinearPair {
    /// Simulate `(X, Y)` and return data plus the true parent edge.
    ///
    /// # Errors
    ///
    /// Invalid lengths / lag.
    pub fn simulate(&self) -> Result<(TimeSeriesData, KnownLaggedParent), DataError> {
        if self.n < self.lag as usize + 2 {
            return Err(DataError::InvalidValidity {
                message: "series too short for configured lag",
            });
        }
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .map_err(|e| DataError::Schema(e.to_string()))?;
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .map_err(|e| DataError::Schema(e.to_string()))?;
        let schema = b.build().map_err(|e| DataError::Schema(e.to_string()))?;

        let mut x = vec![0.0; self.n];
        let mut y = vec![0.0; self.n];
        for t in 0..self.n {
            x[t] = det_noise(self.seed, t as u64, 1);
            let x_lag = if t >= self.lag as usize {
                x[t - self.lag as usize]
            } else {
                0.0
            };
            y[t] = self.coef * x_lag + 0.2 * det_noise(self.seed, t as u64, 2);
        }

        let cols = vec![
            OwnedColumn::Float64(Float64Column::new(
                VariableId::from_raw(0),
                Arc::from(x),
                ValidityBitmap::all_valid(self.n),
            )?),
            OwnedColumn::Float64(Float64Column::new(
                VariableId::from_raw(1),
                Arc::from(y),
                ValidityBitmap::all_valid(self.n),
            )?),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None)?;
        let data = TimeSeriesData::try_new(
            storage,
            TimeIndex {
                regularity: SamplingRegularity::Regular { interval_ns: 1 },
                length: self.n,
            },
        )?;
        let parent = KnownLaggedParent {
            source: VariableId::from_raw(0),
            source_lag: Lag::from_raw(self.lag),
            target: VariableId::from_raw(1),
        };
        Ok((data, parent))
    }
}

fn det_noise(seed: u64, t: u64, stream: u64) -> f64 {
    // SplitMix-style deterministic noise in (-1, 1).
    let mut z = seed
        .wrapping_add(t.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(stream.wrapping_mul(0xD6E8_FEB8_6659_FD93));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let u = (z >> 11) as f64 / ((1u64 << 53) as f64);
    u * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::TableView;

    #[test]
    fn lag1_pair_recovers_length_and_parent() {
        let (data, parent) = LaggedLinearPair::default().simulate().unwrap();
        assert_eq!(data.row_count(), 500);
        assert_eq!(parent.source_lag.raw(), 1);
        assert_eq!(parent.source.raw(), 0);
        assert_eq!(parent.target.raw(), 1);
    }
}
