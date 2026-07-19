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

/// Default cap on distinct time levels for [`TimeDummyEncoding::OneHot`] (fail-closed).
pub const DEFAULT_MAX_TIME_ONE_HOT_LEVELS: usize = 512;

/// How the synthetic time dummy is embedded in the pooled frame.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum TimeDummyEncoding {
    /// Single integer raw-time index column (legacy opt-in).
    #[default]
    IntegerIndex,
    /// One-hot over distinct raw times in the pooled effective rows (`T−1` columns;
    /// last level is the reference category — same convention as space dummies).
    /// Paper-faithful Günther embedding (up to the dropped reference column).
    OneHot,
}

/// Options for synthesizing dataset / time dummy columns on a pooled frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DummyOptions {
    /// One-hot space (dataset) dummy with `M−1` columns when `M > 1`.
    pub include_space_dummy: bool,
    /// When true, synthesize a time dummy under [`Self::time_dummy_encoding`].
    pub include_time_dummy: bool,
    /// Integer index vs one-hot of `T` (ignored unless [`Self::include_time_dummy`]).
    pub time_dummy_encoding: TimeDummyEncoding,
    /// Fail-closed cap on distinct time levels for [`TimeDummyEncoding::OneHot`].
    pub max_time_one_hot_levels: usize,
}

impl Default for DummyOptions {
    fn default() -> Self {
        Self {
            include_space_dummy: true,
            include_time_dummy: false,
            time_dummy_encoding: TimeDummyEncoding::IntegerIndex,
            max_time_one_hot_levels: DEFAULT_MAX_TIME_ONE_HOT_LEVELS,
        }
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
    /// Synthetic time-dummy variable ids (one integer column, or `T−1` one-hot columns).
    pub time_dummy_variables: Arc<[VariableId]>,
    /// Per-environment effective row counts in stack order.
    pub env_effective_rows: Arc<[usize]>,
}

impl PooledLaggedFrame {
    /// All variable ids present in [`Self::frame`] (observed then space dummies then time).
    #[must_use]
    pub fn all_variables(&self) -> Vec<VariableId> {
        let mut v = self.observed_variables.to_vec();
        v.extend_from_slice(&self.space_dummy_variables);
        v.extend_from_slice(&self.time_dummy_variables);
        v
    }

    /// Whether `v` is a synthesized dummy column.
    #[must_use]
    pub fn is_dummy(&self, v: VariableId) -> bool {
        self.space_dummy_variables.iter().any(|&x| x == v)
            || self.time_dummy_variables.iter().any(|&x| x == v)
    }
}

/// Build a pooled lagged frame from multi-environment data.
///
/// Each environment is materialized with [`LaggedFrame::from_series`] at
/// `frame_depth`, then vertically stacked. Dummy columns are appended as
/// contemporaneous-constant (space) or time-index / one-hot (time) series.
///
/// # Errors
///
/// Empty multi-env / variable list, per-env frame failures, stack geometry mismatch,
/// time alignment failure, or one-hot level count above the configured cap.
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
    let mut time_dummies = Vec::new();
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
        let times = effective_raw_times(data, &env_effective, frame_depth)?;
        match dummies.time_dummy_encoding {
            TimeDummyEncoding::IntegerIndex => {
                let id = VariableId::from_raw(cursor);
                cursor = cursor.saturating_add(1);
                time_dummies.push(id);
                let col: Vec<f64> = times.iter().map(|&t| f64::from(t)).collect();
                pooled = pooled.append_constant_lag_columns(&[(id, col)])?;
            }
            TimeDummyEncoding::OneHot => {
                let mut levels = times.clone();
                levels.sort_unstable();
                levels.dedup();
                if levels.len() > dummies.max_time_one_hot_levels {
                    return Err(DataError::InvalidArgument {
                        message: format!(
                            "time one-hot: {} distinct levels exceeds max_time_one_hot_levels={}",
                            levels.len(),
                            dummies.max_time_one_hot_levels
                        ),
                    });
                }
                // T≤1 → constant; no identifiable dummy columns.
                if levels.len() > 1 {
                    let n_hot = levels.len() - 1;
                    let mut cols: Vec<(VariableId, Vec<f64>)> = Vec::with_capacity(n_hot);
                    for &level in &levels[..n_hot] {
                        let id = VariableId::from_raw(cursor);
                        cursor = cursor.saturating_add(1);
                        time_dummies.push(id);
                        let col: Vec<f64> =
                            times.iter().map(|&t| if t == level { 1.0 } else { 0.0 }).collect();
                        cols.push((id, col));
                    }
                    pooled = pooled.append_constant_lag_columns(&cols)?;
                }
            }
        }
    }

    let _ = cursor; // advance tracked for future synthetic columns
    Ok(PooledLaggedFrame {
        frame: pooled,
        observed_variables: Arc::from(variables.to_vec()),
        space_dummy_variables: Arc::from(space_dummies),
        time_dummy_variables: Arc::from(time_dummies),
        env_effective_rows: Arc::from(env_effective),
    })
}

/// Absolute raw time index for each stacked effective row (`j + frame_depth` under SeriesOrigin).
fn effective_raw_times(
    data: &MultiEnvironmentData,
    env_effective: &[usize],
    frame_depth: u32,
) -> Result<Vec<u32>, DataError> {
    let mut times = Vec::new();
    let base = frame_depth as usize;
    for i in 0..data.env_count() {
        let series = data.environment(i)?;
        let n_raw = series.row_count();
        let n_eff = env_effective[i];
        if base + n_eff > n_raw {
            return Err(DataError::InvalidArgument {
                message: format!(
                    "time dummy: env {i} effective {n_eff} + depth {frame_depth} exceeds rows {n_raw}"
                ),
            });
        }
        for j in 0..n_eff {
            times.push(u32::try_from(base + j).map_err(|_| DataError::InvalidArgument {
                message: "time dummy: raw time index exceeds u32".into(),
            })?);
        }
    }
    Ok(times)
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
            DummyOptions { include_space_dummy: false, include_time_dummy: false, ..DummyOptions::default() },
            &KernelPolicy::default_policy(),
        )
        .unwrap();
        // n_effective = (20-2) + (30-2) = 46
        assert_eq!(pooled.frame.n_effective(), 46);
        assert_eq!(pooled.env_effective_rows.as_ref(), &[18, 28]);
        assert!(pooled.space_dummy_variables.is_empty());
        assert!(pooled.time_dummy_variables.is_empty());
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
            DummyOptions { include_space_dummy: true, include_time_dummy: false, ..DummyOptions::default() },
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
    fn time_dummy_integer_tracks_raw_time() {
        let a = float_series(12, 1);
        let multi = MultiEnvironmentData::try_new(Arc::from([a])).unwrap();
        let vars = [VariableId::from_raw(0)];
        let pooled = pool_multi_env_lagged_frame(
            &multi,
            &vars,
            2,
            DummyOptions {
                include_space_dummy: false,
                include_time_dummy: true,
                time_dummy_encoding: TimeDummyEncoding::IntegerIndex,
                ..DummyOptions::default()
            },
            &KernelPolicy::default_policy(),
        )
        .unwrap();
        assert_eq!(pooled.time_dummy_variables.len(), 1);
        let tid = pooled.time_dummy_variables[0];
        let col = pooled
            .frame
            .column(pooled.frame.column_index(tid, Lag::CONTEMPORANEOUS).unwrap());
        assert_eq!(col.len(), 10);
        assert!((col[0] - 2.0).abs() < 1e-12);
        assert!((col[9] - 11.0).abs() < 1e-12);
    }

    #[test]
    fn time_dummy_one_hot_t_minus_1() {
        // depth=2, n=6 → effective times {2,3,4,5} → T=4 → 3 one-hot columns.
        let a = float_series(6, 1);
        let multi = MultiEnvironmentData::try_new(Arc::from([a])).unwrap();
        let vars = [VariableId::from_raw(0)];
        let pooled = pool_multi_env_lagged_frame(
            &multi,
            &vars,
            2,
            DummyOptions {
                include_space_dummy: false,
                include_time_dummy: true,
                time_dummy_encoding: TimeDummyEncoding::OneHot,
                ..DummyOptions::default()
            },
            &KernelPolicy::default_policy(),
        )
        .unwrap();
        assert_eq!(pooled.time_dummy_variables.len(), 3);
        let t0 = pooled.time_dummy_variables[0];
        let col0 = pooled
            .frame
            .column(pooled.frame.column_index(t0, Lag::CONTEMPORANEOUS).unwrap());
        // Row 0 → time 2 → first level → 1
        assert!((col0[0] - 1.0).abs() < 1e-12);
        // Row 1 → time 3 → 0 on first column
        assert!(col0[1].abs() < 1e-12);
        // Last row → time 5 (reference) → all zeros
        let last = col0.len() - 1;
        for &tid in pooled.time_dummy_variables.iter() {
            let col = pooled
                .frame
                .column(pooled.frame.column_index(tid, Lag::CONTEMPORANEOUS).unwrap());
            assert!(col[last].abs() < 1e-12, "reference time should be all-zero");
        }
    }

    #[test]
    fn time_dummy_one_hot_respects_level_cap() {
        let a = float_series(20, 1);
        let multi = MultiEnvironmentData::try_new(Arc::from([a])).unwrap();
        let vars = [VariableId::from_raw(0)];
        let err = pool_multi_env_lagged_frame(
            &multi,
            &vars,
            2,
            DummyOptions {
                include_space_dummy: false,
                include_time_dummy: true,
                time_dummy_encoding: TimeDummyEncoding::OneHot,
                max_time_one_hot_levels: 4,
            },
            &KernelPolicy::default_policy(),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("max_time_one_hot_levels"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn time_dummy_one_hot_aligns_across_unequal_envs() {
        // Env A: times 2..5 (n=6, depth=2); Env B: times 2..7 (n=8, depth=2).
        // Union levels {2,3,4,5,6,7} → 5 one-hot columns.
        let a = float_series(6, 1);
        let b = float_series(8, 1);
        let multi = MultiEnvironmentData::try_new(Arc::from([a, b])).unwrap();
        let vars = [VariableId::from_raw(0)];
        let pooled = pool_multi_env_lagged_frame(
            &multi,
            &vars,
            2,
            DummyOptions {
                include_space_dummy: false,
                include_time_dummy: true,
                time_dummy_encoding: TimeDummyEncoding::OneHot,
                ..DummyOptions::default()
            },
            &KernelPolicy::default_policy(),
        )
        .unwrap();
        assert_eq!(pooled.frame.n_effective(), 4 + 6);
        assert_eq!(pooled.time_dummy_variables.len(), 5);
    }
}
