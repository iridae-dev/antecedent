//! Pooled multi-environment lagged frames with space/time dummies (J-PCMCI+).
//!
//! Materializes each environment independently (no cross-env lag windows), stacks
//! effective rows, then appends synthetic dummy columns.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use causal_core::{KernelPolicy, VariableId};

use crate::error::DataError;
use crate::lagged_frame::LaggedFrame;
use crate::multi_env::MultiEnvironmentData;
use crate::table::TableView;

/// Options for synthesizing dataset / time dummy columns on a pooled frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DummyOptions {
    /// One-hot space (dataset) dummy with `M−1` columns when `M > 1`.
    pub include_space_dummy: bool,
    /// Integer time-index dummy (one column), opt-in — high-dimensional one-hot
    /// of `T` is deferred.
    pub include_time_dummy: bool,
}

impl Default for DummyOptions {
    fn default() -> Self {
        Self { include_space_dummy: true, include_time_dummy: false }
    }
}

/// Result of pooling multi-environment series into one lagged frame.
#[derive(Clone, Debug)]
pub struct PooledLaggedFrame {
    /// Stacked frame including system/context variables and any dummy columns.
    pub frame: LaggedFrame,
    /// Variable ids that were requested for pooling (system + observed context).
    pub observed_variables: Arc<[VariableId]>,
    /// Synthetic space-dummy variable ids (`M−1` one-hot columns), empty if off.
    pub space_dummy_variables: Arc<[VariableId]>,
    /// Synthetic time-dummy variable id (single integer column), if enabled.
    pub time_dummy_variable: Option<VariableId>,
    /// Per-environment effective row counts in stack order.
    pub env_effective_rows: Arc<[usize]>,
}

impl PooledLaggedFrame {
    /// All variable ids present in [`Self::frame`] (observed then space dummies then time).
    #[must_use]
    pub fn all_variables(&self) -> Vec<VariableId> {
        let mut v = self.observed_variables.to_vec();
        v.extend_from_slice(&self.space_dummy_variables);
        if let Some(t) = self.time_dummy_variable {
            v.push(t);
        }
        v
    }

    /// Whether `v` is a synthesized dummy column.
    #[must_use]
    pub fn is_dummy(&self, v: VariableId) -> bool {
        self.space_dummy_variables.iter().any(|&x| x == v)
            || self.time_dummy_variable == Some(v)
    }
}

/// Build a pooled lagged frame from multi-environment data.
///
/// Each environment is materialized with [`LaggedFrame::from_series`] at
/// `frame_depth`, then vertically stacked. Dummy columns are appended as
/// contemporaneous-constant (space) or integer time-index (time) series.
///
/// # Errors
///
/// Empty multi-env / variable list, per-env frame failures, or stack geometry mismatch.
pub fn pool_multi_env_lagged_frame(
    data: &MultiEnvironmentData,
    variables: &[VariableId],
    frame_depth: u32,
    dummies: DummyOptions,
    policy: &KernelPolicy,
) -> Result<PooledLaggedFrame, DataError> {
    if data.env_count() == 0 {
        return Err(DataError::InvalidArgument {
            message: "pooled lagged frame needs ≥1 environment".into(),
        });
    }
    if variables.is_empty() {
        return Err(DataError::InvalidArgument {
            message: "pooled lagged frame needs ≥1 variable".into(),
        });
    }

    let mut frames = Vec::with_capacity(data.env_count());
    let mut env_effective = Vec::with_capacity(data.env_count());
    for i in 0..data.env_count() {
        let series = data.environment(i)?;
        let frame = LaggedFrame::from_series(series, variables, frame_depth, policy)?;
        env_effective.push(frame.n_effective());
        frames.push(frame);
    }
    let mut pooled = LaggedFrame::stack(&frames)?;

    let next_id = next_synthetic_id(variables);
    let mut space_dummies = Vec::new();
    let mut time_dummy = None;
    let mut cursor = next_id;

    if dummies.include_space_dummy && data.env_count() > 1 {
        let m = data.env_count();
        let n_hot = m - 1;
        let mut cols: Vec<(VariableId, Vec<f64>)> = Vec::with_capacity(n_hot);
        for k in 0..n_hot {
            let id = VariableId::from_raw(cursor);
            cursor = cursor.saturating_add(1);
            space_dummies.push(id);
            let mut col = Vec::with_capacity(pooled.n_effective());
            for (env_i, &n_eff) in env_effective.iter().enumerate() {
                let val = if env_i == k { 1.0 } else { 0.0 };
                col.extend(std::iter::repeat_n(val, n_eff));
            }
            cols.push((id, col));
        }
        pooled = pooled.append_constant_lag_columns(&cols)?;
    }

    if dummies.include_time_dummy {
        let id = VariableId::from_raw(cursor);
        time_dummy = Some(id);
        let mut col = Vec::with_capacity(pooled.n_effective());
        for i in 0..data.env_count() {
            let series = data.environment(i)?;
            let n_raw = series.row_count();
            let n_eff = env_effective[i];
            // Effective sample `j` aligns to raw time `j + frame_depth` under SeriesOrigin.
            let base = frame_depth as usize;
            if base + n_eff > n_raw {
                return Err(DataError::InvalidArgument {
                    message: format!(
                        "time dummy: env {i} effective {n_eff} + depth {frame_depth} exceeds rows {n_raw}"
                    ),
                });
            }
            for j in 0..n_eff {
                col.push((base + j) as f64);
            }
        }
        pooled = pooled.append_constant_lag_columns(&[(id, col)])?;
    }

    Ok(PooledLaggedFrame {
        frame: pooled,
        observed_variables: Arc::from(variables.to_vec()),
        space_dummy_variables: Arc::from(space_dummies),
        time_dummy_variable: time_dummy,
        env_effective_rows: Arc::from(env_effective),
    })
}

fn next_synthetic_id(variables: &[VariableId]) -> u32 {
    variables.iter().map(|v| v.raw()).max().map(|m| m.saturating_add(1)).unwrap_or(0)
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use std::sync::Arc;

    use causal_core::{Lag, VariableId};

    use super::*;
    use crate::multi_env::MultiEnvironmentData;
    use crate::testing::float_series;

    #[test]
    fn pool_stacks_effective_rows_without_cross_env_bleed() {
        let a = float_series(20, 2);
        let b = float_series(30, 2);
        let multi = MultiEnvironmentData::try_new(Arc::from([a, b])).unwrap();
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let depth = 2u32;
        let pooled = pool_multi_env_lagged_frame(
            &multi,
            &vars,
            depth,
            DummyOptions { include_space_dummy: false, include_time_dummy: false },
            &KernelPolicy::default_policy(),
        )
        .unwrap();
        // n_effective = (20-2) + (30-2) = 46
        assert_eq!(pooled.frame.n_effective(), 46);
        assert_eq!(pooled.env_effective_rows.as_ref(), &[18, 28]);
        assert!(pooled.space_dummy_variables.is_empty());
        assert!(pooled.time_dummy_variable.is_none());
    }

    #[test]
    fn space_dummy_one_hot_m_minus_1() {
        let a = float_series(16, 2);
        let b = float_series(16, 2);
        let c = float_series(16, 2);
        let multi = MultiEnvironmentData::try_new(Arc::from([a, b, c])).unwrap();
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let pooled = pool_multi_env_lagged_frame(
            &multi,
            &vars,
            2,
            DummyOptions { include_space_dummy: true, include_time_dummy: false },
            &KernelPolicy::default_policy(),
        )
        .unwrap();
        assert_eq!(pooled.space_dummy_variables.len(), 2);
        let d0 = pooled.space_dummy_variables[0];
        let col = pooled
            .frame
            .column(pooled.frame.column_index(d0, Lag::CONTEMPORANEOUS).unwrap());
        // First env block: 14 effective rows of 1.0
        assert!((col[0] - 1.0).abs() < 1e-12);
        assert!((col[13] - 1.0).abs() < 1e-12);
        // Second env: 0.0 for first hot column
        assert!((col[14]).abs() < 1e-12);
    }

    #[test]
    fn time_dummy_tracks_raw_time() {
        let a = float_series(12, 1);
        let multi = MultiEnvironmentData::try_new(Arc::from([a])).unwrap();
        let vars = [VariableId::from_raw(0)];
        let pooled = pool_multi_env_lagged_frame(
            &multi,
            &vars,
            2,
            DummyOptions { include_space_dummy: false, include_time_dummy: true },
            &KernelPolicy::default_policy(),
        )
        .unwrap();
        let tid = pooled.time_dummy_variable.unwrap();
        let col = pooled
            .frame
            .column(pooled.frame.column_index(tid, Lag::CONTEMPORANEOUS).unwrap());
        assert_eq!(col.len(), 10);
        assert!((col[0] - 2.0).abs() < 1e-12);
        assert!((col[9] - 11.0).abs() < 1e-12);
    }
}
