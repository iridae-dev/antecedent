//! Split strategies for discovery / estimation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss)]

use std::collections::BTreeMap;
use std::sync::Arc;

use causal_core::CausalRng;

use crate::error::DataError;

/// Half-open index range `[start, end)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TimeRange {
    /// Inclusive start index.
    pub start: usize,
    /// Exclusive end index.
    pub end: usize,
}

impl TimeRange {
    /// Length of the range.
    #[must_use]
    pub const fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Whether the range is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start >= self.end
    }
}

/// Train / test row-index partition (metadata only — no data copy).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowSplit {
    /// Training row indexes.
    pub train: Arc<[u32]>,
    /// Test / holdout row indexes.
    pub test: Arc<[u32]>,
}

impl RowSplit {
    /// Total rows covered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.train.len() + self.test.len()
    }

    /// Whether both sides are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.train.is_empty() && self.test.is_empty()
    }
}

/// One rolling-origin fold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemporalFold {
    /// Fold index (0-based).
    pub fold: usize,
    /// Training window (half-open).
    pub train: TimeRange,
    /// Test / forecast window (half-open).
    pub test: TimeRange,
    /// Gap between train end and test start.
    pub gap: usize,
}

/// Whether random row splits are allowed on temporal / panel / event data.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub enum TemporalRandomPolicy {
    /// Refuse random IID splits on temporal-like data (default, §5.6).
    #[default]
    Refuse,
    /// Explicit opt-in for random row splits on temporal / panel / event data.
    Allow,
}

/// Guard used by planners before applying [`RandomIidSplit`] to temporal data.
///
/// # Errors
///
/// When policy is [`TemporalRandomPolicy::Refuse`].
pub fn ensure_random_allowed_on_temporal(policy: TemporalRandomPolicy) -> Result<(), DataError> {
    match policy {
        TemporalRandomPolicy::Allow => Ok(()),
        TemporalRandomPolicy::Refuse => Err(DataError::InvalidArgument {
            message: "random IID split refused on temporal/panel/event data without explicit opt-in"
                .into(),
        }),
    }
}

/// Discovery / estimation split with a temporal gap.
///
/// Layout over `0..series_len`:
/// `[discovery) | gap | [estimation)`.
///
/// This is metadata only — no data copy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct DiscoveryEstimationSplit {
    /// Contiguous discovery window.
    pub discovery: TimeRange,
    /// Contiguous estimation window (after the gap).
    pub estimation: TimeRange,
    /// Number of time steps skipped between discovery end and estimation start.
    pub gap: usize,
    /// Full series length used to validate the split.
    pub series_len: usize,
}

impl DiscoveryEstimationSplit {
    /// Build a split from absolute ranges.
    ///
    /// # Errors
    ///
    /// Empty windows, inverted ranges, overlapping discovery/estimation, or
    /// ranges outside `series_len`.
    pub fn try_new(
        series_len: usize,
        discovery: TimeRange,
        estimation: TimeRange,
    ) -> Result<Self, DataError> {
        validate_range(series_len, discovery, "discovery")?;
        validate_range(series_len, estimation, "estimation")?;
        if discovery.end > estimation.start {
            return Err(DataError::InvalidArgument {
                message:
                    "discovery window must end at or before estimation start (with optional gap)"
                        .into(),
            });
        }
        let gap = estimation.start - discovery.end;
        Ok(Self { discovery, estimation, gap, series_len })
    }

    /// Split `series_len` into discovery / gap / estimation by sizes.
    ///
    /// # Errors
    ///
    /// When sizes do not sum to `series_len`, or either window is empty.
    pub fn from_sizes(
        series_len: usize,
        discovery_len: usize,
        gap: usize,
        estimation_len: usize,
    ) -> Result<Self, DataError> {
        if discovery_len == 0 || estimation_len == 0 {
            return Err(DataError::InvalidArgument {
                message: "discovery and estimation windows must be non-empty".into(),
            });
        }
        let need = discovery_len
            .checked_add(gap)
            .and_then(|v| v.checked_add(estimation_len))
            .ok_or(DataError::InvalidArgument { message: "split sizes overflow".into() })?;
        if need != series_len {
            return Err(DataError::InvalidArgument {
                message: "discovery + gap + estimation must equal series_len".into(),
            });
        }
        let discovery = TimeRange { start: 0, end: discovery_len };
        let estimation = TimeRange { start: discovery_len + gap, end: series_len };
        Self::try_new(series_len, discovery, estimation)
    }

    /// Proportional split: discovery gets `discovery_frac` of the length after
    /// reserving `gap` steps in the middle (remainder → estimation).
    ///
    /// # Errors
    ///
    /// Invalid fraction, insufficient length for gap + two non-empty windows.
    pub fn from_fraction(
        series_len: usize,
        discovery_frac: f64,
        gap: usize,
    ) -> Result<Self, DataError> {
        if !(discovery_frac > 0.0 && discovery_frac < 1.0) {
            return Err(DataError::InvalidArgument {
                message: "discovery_frac must be in (0, 1)".into(),
            });
        }
        if series_len <= gap + 1 {
            return Err(DataError::InvalidArgument {
                message: "series too short for gap and two non-empty windows".into(),
            });
        }
        let usable = series_len - gap;
        let mut discovery_len = ((usable as f64) * discovery_frac).floor() as usize;
        if discovery_len == 0 {
            discovery_len = 1;
        }
        if discovery_len >= usable {
            discovery_len = usable - 1;
        }
        let estimation_len = usable - discovery_len;
        Self::from_sizes(series_len, discovery_len, gap, estimation_len)
    }
}

/// Environment holdout: discovery vs estimation environment index sets (no data copy).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnvHoldoutSplit {
    /// Environment indices used for discovery / CI.
    pub discovery_envs: Arc<[usize]>,
    /// Environment indices held out for estimation / evaluation.
    pub estimation_envs: Arc<[usize]>,
}

impl EnvHoldoutSplit {
    /// Split environments: first `n_discovery` indices for discovery, rest for estimation.
    ///
    /// # Errors
    ///
    /// Empty total, or `n_discovery` not in `1..n_env`.
    pub fn try_prefix(n_env: usize, n_discovery: usize) -> Result<Self, DataError> {
        if n_env == 0 {
            return Err(DataError::InvalidArgument {
                message: "env holdout needs ≥1 environment".into(),
            });
        }
        if n_discovery == 0 || n_discovery >= n_env {
            return Err(DataError::InvalidArgument {
                message: "n_discovery must be in 1..n_env".into(),
            });
        }
        Ok(Self {
            discovery_envs: Arc::from((0..n_discovery).collect::<Vec<_>>()),
            estimation_envs: Arc::from((n_discovery..n_env).collect::<Vec<_>>()),
        })
    }
}

/// Regime holdout: discovery vs estimation regime id sets (no data copy).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegimeHoldoutSplit {
    /// Regime ids used for discovery.
    pub discovery_regimes: Arc<[u32]>,
    /// Regime ids held out for estimation.
    pub estimation_regimes: Arc<[u32]>,
}

impl RegimeHoldoutSplit {
    /// Partition a list of regime ids into discovery / estimation by prefix count.
    ///
    /// # Errors
    ///
    /// Empty list, or `n_discovery` not in `1..n_regimes`.
    pub fn try_prefix(regime_ids: &[u32], n_discovery: usize) -> Result<Self, DataError> {
        let n = regime_ids.len();
        if n == 0 {
            return Err(DataError::InvalidArgument {
                message: "regime holdout needs ≥1 regime".into(),
            });
        }
        if n_discovery == 0 || n_discovery >= n {
            return Err(DataError::InvalidArgument {
                message: "n_discovery must be in 1..n_regimes".into(),
            });
        }
        Ok(Self {
            discovery_regimes: Arc::from(regime_ids[..n_discovery].to_vec()),
            estimation_regimes: Arc::from(regime_ids[n_discovery..].to_vec()),
        })
    }
}

/// Random IID train/test split (seeded; metadata only).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RandomIidSplit {
    /// Resulting row partition.
    pub rows: RowSplit,
    /// Seed used to shuffle.
    pub seed: u64,
    /// Requested test fraction.
    pub test_frac: OrderedFrac,
}

/// Newtype so `f64` fraction can live on an `Eq` struct via bit pattern.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrderedFrac(pub f64);

impl Eq for OrderedFrac {}

impl std::hash::Hash for OrderedFrac {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl RandomIidSplit {
    /// Shuffle `0..n` and take the last `test_frac` as test.
    ///
    /// For temporal / panel / event data, callers must pass
    /// [`TemporalRandomPolicy::Allow`] via [`ensure_random_allowed_on_temporal`]
    /// before constructing this split.
    ///
    /// # Errors
    ///
    /// Empty `n`, invalid fraction, or empty train/test after rounding.
    pub fn try_new(n: usize, test_frac: f64, seed: u64) -> Result<Self, DataError> {
        if n == 0 {
            return Err(DataError::InvalidArgument {
                message: "random IID split needs n ≥ 1".into(),
            });
        }
        if !(test_frac > 0.0 && test_frac < 1.0) {
            return Err(DataError::InvalidArgument {
                message: "test_frac must be in (0, 1)".into(),
            });
        }
        let mut test_len = ((n as f64) * test_frac).round() as usize;
        if test_len == 0 {
            test_len = 1;
        }
        if test_len >= n {
            test_len = n - 1;
        }
        let mut idx: Vec<u32> = (0..n as u32).collect();
        fisher_yates(&mut idx, seed);
        let train = Arc::from(idx[..n - test_len].to_vec());
        let test = Arc::from(idx[n - test_len..].to_vec());
        Ok(Self {
            rows: RowSplit { train, test },
            seed,
            test_frac: OrderedFrac(test_frac),
        })
    }
}

/// Grouped split: entire groups assigned to train or test (no leakage).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupedSplit {
    /// Row partition.
    pub rows: RowSplit,
    /// Group ids assigned to train.
    pub train_groups: Arc<[i64]>,
    /// Group ids assigned to test.
    pub test_groups: Arc<[i64]>,
    /// Seed.
    pub seed: u64,
}

impl GroupedSplit {
    /// Assign whole groups by shuffling unique group ids.
    ///
    /// # Errors
    ///
    /// Length mismatch, fewer than two groups, or invalid fraction.
    pub fn try_new(group_ids: &[i64], test_frac: f64, seed: u64) -> Result<Self, DataError> {
        split_by_labels(group_ids, test_frac, seed).map(|(rows, train_groups, test_groups)| Self {
            rows,
            train_groups,
            test_groups,
            seed,
        })
    }
}

/// Cluster split: same mechanics as grouped, distinct type for cluster ids.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClusterSplit {
    /// Row partition.
    pub rows: RowSplit,
    /// Cluster ids in train.
    pub train_clusters: Arc<[i64]>,
    /// Cluster ids in test.
    pub test_clusters: Arc<[i64]>,
    /// Seed.
    pub seed: u64,
}

impl ClusterSplit {
    /// Assign whole clusters by shuffling unique cluster ids.
    ///
    /// # Errors
    ///
    /// Same as [`GroupedSplit::try_new`].
    pub fn try_new(cluster_ids: &[i64], test_frac: f64, seed: u64) -> Result<Self, DataError> {
        split_by_labels(cluster_ids, test_frac, seed).map(
            |(rows, train_clusters, test_clusters)| Self {
                rows,
                train_clusters,
                test_clusters,
                seed,
            },
        )
    }
}

/// Blocked temporal split: contiguous time blocks assigned to train / test.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockedTemporalSplit {
    /// Series length.
    pub series_len: usize,
    /// Block length in rows.
    pub block_len: usize,
    /// Training blocks as ranges.
    pub train_blocks: Arc<[TimeRange]>,
    /// Test blocks as ranges.
    pub test_blocks: Arc<[TimeRange]>,
    /// Flattened row partition.
    pub rows: RowSplit,
    /// Seed.
    pub seed: u64,
}

impl BlockedTemporalSplit {
    /// Partition `0..series_len` into blocks of `block_len` and assign by shuffle.
    ///
    /// The final block may be shorter. At least one train and one test block required.
    ///
    /// # Errors
    ///
    /// Invalid sizes / fraction.
    pub fn try_new(
        series_len: usize,
        block_len: usize,
        test_frac: f64,
        seed: u64,
    ) -> Result<Self, DataError> {
        if series_len == 0 || block_len == 0 {
            return Err(DataError::InvalidArgument {
                message: "blocked temporal split needs series_len, block_len ≥ 1".into(),
            });
        }
        if !(test_frac > 0.0 && test_frac < 1.0) {
            return Err(DataError::InvalidArgument {
                message: "test_frac must be in (0, 1)".into(),
            });
        }
        let mut blocks = Vec::new();
        let mut start = 0usize;
        while start < series_len {
            let end = (start + block_len).min(series_len);
            blocks.push(TimeRange { start, end });
            start = end;
        }
        let n_blocks = blocks.len();
        if n_blocks < 2 {
            return Err(DataError::InvalidArgument {
                message: "blocked temporal split needs ≥2 blocks".into(),
            });
        }
        let mut test_blocks_n = ((n_blocks as f64) * test_frac).round() as usize;
        if test_blocks_n == 0 {
            test_blocks_n = 1;
        }
        if test_blocks_n >= n_blocks {
            test_blocks_n = n_blocks - 1;
        }
        let mut order: Vec<usize> = (0..n_blocks).collect();
        fisher_yates_usize(&mut order, seed);
        let mut train_blocks = Vec::new();
        let mut test_blocks = Vec::new();
        let mut train_rows = Vec::new();
        let mut test_rows = Vec::new();
        for (rank, &bi) in order.iter().enumerate() {
            let block = blocks[bi];
            let rows = (block.start as u32)..(block.end as u32);
            if rank < n_blocks - test_blocks_n {
                train_blocks.push(block);
                train_rows.extend(rows);
            } else {
                test_blocks.push(block);
                test_rows.extend(rows);
            }
        }
        Ok(Self {
            series_len,
            block_len,
            train_blocks: Arc::from(train_blocks),
            test_blocks: Arc::from(test_blocks),
            rows: RowSplit {
                train: Arc::from(train_rows),
                test: Arc::from(test_rows),
            },
            seed,
        })
    }
}

/// Rolling-origin (expanding window) temporal folds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RollingOriginSplit {
    /// Series length.
    pub series_len: usize,
    /// Minimum initial training length.
    pub min_train: usize,
    /// Forecast / test horizon per fold.
    pub horizon: usize,
    /// Gap between train end and test start.
    pub gap: usize,
    /// Step between successive fold origins.
    pub step: usize,
    /// Ordered folds.
    pub folds: Arc<[TemporalFold]>,
}

impl RollingOriginSplit {
    /// Build expanding-origin folds.
    ///
    /// Fold `k` uses train `[0, origin)` where `origin = min_train + k * step`,
    /// then gap, then test of length `horizon`.
    ///
    /// # Errors
    ///
    /// When no complete fold fits in `series_len`.
    pub fn try_new(
        series_len: usize,
        min_train: usize,
        horizon: usize,
        gap: usize,
        step: usize,
    ) -> Result<Self, DataError> {
        if min_train == 0 || horizon == 0 || step == 0 {
            return Err(DataError::InvalidArgument {
                message: "rolling-origin requires min_train, horizon, step ≥ 1".into(),
            });
        }
        let mut folds = Vec::new();
        let mut origin = min_train;
        let mut fold = 0usize;
        while origin + gap + horizon <= series_len {
            folds.push(TemporalFold {
                fold,
                train: TimeRange { start: 0, end: origin },
                test: TimeRange {
                    start: origin + gap,
                    end: origin + gap + horizon,
                },
                gap,
            });
            fold += 1;
            origin = origin.saturating_add(step);
        }
        if folds.is_empty() {
            return Err(DataError::InvalidArgument {
                message: "series too short for any rolling-origin fold".into(),
            });
        }
        Ok(Self {
            series_len,
            min_train,
            horizon,
            gap,
            step,
            folds: Arc::from(folds),
        })
    }
}

fn validate_range(series_len: usize, range: TimeRange, label: &str) -> Result<(), DataError> {
    if range.is_empty() {
        return Err(DataError::InvalidArgument {
            message: format!("{label} window must be non-empty"),
        });
    }
    if range.end > series_len {
        return Err(DataError::InvalidArgument {
            message: format!("{label} window exceeds series_len"),
        });
    }
    Ok(())
}

fn fisher_yates(idx: &mut [u32], seed: u64) {
    let mut rng = CausalRng::from_seed(seed);
    for i in (1..idx.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        idx.swap(i, j);
    }
}

fn fisher_yates_usize(idx: &mut [usize], seed: u64) {
    let mut rng = CausalRng::from_seed(seed);
    for i in (1..idx.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        idx.swap(i, j);
    }
}

fn split_by_labels(
    labels: &[i64],
    test_frac: f64,
    seed: u64,
) -> Result<(RowSplit, Arc<[i64]>, Arc<[i64]>), DataError> {
    if labels.is_empty() {
        return Err(DataError::InvalidArgument {
            message: "grouped/cluster split needs ≥1 row".into(),
        });
    }
    if !(test_frac > 0.0 && test_frac < 1.0) {
        return Err(DataError::InvalidArgument {
            message: "test_frac must be in (0, 1)".into(),
        });
    }
    let mut members: BTreeMap<i64, Vec<u32>> = BTreeMap::new();
    for (i, &g) in labels.iter().enumerate() {
        members.entry(g).or_default().push(i as u32);
    }
    if members.len() < 2 {
        return Err(DataError::InvalidArgument {
            message: "grouped/cluster split needs ≥2 distinct labels".into(),
        });
    }
    let mut unique: Vec<i64> = members.keys().copied().collect();
    fisher_yates_i64(&mut unique, seed);
    let n_g = unique.len();
    let mut test_n = ((n_g as f64) * test_frac).round() as usize;
    if test_n == 0 {
        test_n = 1;
    }
    if test_n >= n_g {
        test_n = n_g - 1;
    }
    let test_groups = Arc::<[i64]>::from(unique[n_g - test_n..].to_vec());
    let train_groups = Arc::<[i64]>::from(unique[..n_g - test_n].to_vec());
    let mut train_rows = Vec::new();
    let mut test_rows = Vec::new();
    for &g in train_groups.iter() {
        train_rows.extend_from_slice(&members[&g]);
    }
    for &g in test_groups.iter() {
        test_rows.extend_from_slice(&members[&g]);
    }
    Ok((
        RowSplit { train: Arc::from(train_rows), test: Arc::from(test_rows) },
        train_groups,
        test_groups,
    ))
}

fn fisher_yates_i64(idx: &mut [i64], seed: u64) {
    let mut rng = CausalRng::from_seed(seed);
    for i in (1..idx.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        idx.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_sizes_layout() {
        let s = DiscoveryEstimationSplit::from_sizes(100, 60, 10, 30).unwrap();
        assert_eq!(s.discovery, TimeRange { start: 0, end: 60 });
        assert_eq!(s.gap, 10);
        assert_eq!(s.estimation, TimeRange { start: 70, end: 100 });
    }

    #[test]
    fn from_fraction_respects_gap() {
        let s = DiscoveryEstimationSplit::from_fraction(100, 0.5, 10).unwrap();
        assert_eq!(s.gap, 10);
        assert_eq!(s.discovery.len() + s.gap + s.estimation.len(), 100);
        assert!(!s.discovery.is_empty());
        assert!(!s.estimation.is_empty());
    }

    #[test]
    fn from_fraction_rejects_boundary_fractions() {
        assert!(DiscoveryEstimationSplit::from_fraction(100, 0.0, 10).is_err());
        assert!(DiscoveryEstimationSplit::from_fraction(100, 1.0, 10).is_err());
        assert!(DiscoveryEstimationSplit::from_fraction(100, f64::NAN, 10).is_err());
    }

    #[test]
    fn rejects_overlap() {
        let err = DiscoveryEstimationSplit::try_new(
            50,
            TimeRange { start: 0, end: 30 },
            TimeRange { start: 20, end: 50 },
        );
        assert!(err.is_err());
    }

    #[test]
    fn rejects_bad_sum() {
        assert!(DiscoveryEstimationSplit::from_sizes(100, 40, 10, 40).is_err());
    }

    #[test]
    fn env_holdout_prefix() {
        let s = EnvHoldoutSplit::try_prefix(4, 2).unwrap();
        assert_eq!(s.discovery_envs.as_ref(), &[0, 1]);
        assert_eq!(s.estimation_envs.as_ref(), &[2, 3]);
    }

    #[test]
    fn regime_holdout_prefix() {
        let s = RegimeHoldoutSplit::try_prefix(&[10, 20, 30], 1).unwrap();
        assert_eq!(s.discovery_regimes.as_ref(), &[10]);
        assert_eq!(s.estimation_regimes.as_ref(), &[20, 30]);
    }

    #[test]
    fn random_iid_seed_stable() {
        let a = RandomIidSplit::try_new(100, 0.2, 7).unwrap();
        let b = RandomIidSplit::try_new(100, 0.2, 7).unwrap();
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.rows.test.len(), 20);
        assert_eq!(a.rows.train.len(), 80);
    }

    #[test]
    fn temporal_random_refused_by_default() {
        assert!(ensure_random_allowed_on_temporal(TemporalRandomPolicy::Refuse).is_err());
        assert!(ensure_random_allowed_on_temporal(TemporalRandomPolicy::Allow).is_ok());
    }

    #[test]
    fn grouped_keeps_groups_intact() {
        let ids = [0i64, 0, 1, 1, 2, 2, 3, 3];
        let s = GroupedSplit::try_new(&ids, 0.25, 11).unwrap();
        assert_eq!(s.test_groups.len(), 1);
        assert_eq!(s.train_groups.len(), 3);
        for &g in s.test_groups.iter() {
            for &row in s.rows.test.iter() {
                if ids[row as usize] == g {
                    assert!(!s.rows.train.contains(&row));
                }
            }
        }
    }

    #[test]
    fn cluster_matches_grouped_mechanics() {
        let ids = [10i64, 10, 20, 20, 30, 30];
        let g = GroupedSplit::try_new(&ids, 0.5, 3).unwrap();
        let c = ClusterSplit::try_new(&ids, 0.5, 3).unwrap();
        assert_eq!(g.rows, c.rows);
    }

    #[test]
    fn blocked_temporal_covers_series() {
        let s = BlockedTemporalSplit::try_new(100, 10, 0.2, 5).unwrap();
        assert_eq!(s.rows.len(), 100);
        assert!(!s.train_blocks.is_empty());
        assert!(!s.test_blocks.is_empty());
    }

    #[test]
    fn rolling_origin_fold_count() {
        // min_train=40, horizon=10, gap=0, step=10, series=100
        // origins 40..=90 → 6 folds
        let s = RollingOriginSplit::try_new(100, 40, 10, 0, 10).unwrap();
        assert_eq!(s.folds.len(), 6);
        assert_eq!(s.folds[0].train, TimeRange { start: 0, end: 40 });
        assert_eq!(s.folds[0].test, TimeRange { start: 40, end: 50 });
        assert_eq!(s.folds.last().unwrap().train.end, 90);
    }
}
