//! Batch Fisher-Z tests, with one stable covariance per shared cohort.
//!
//! A cache entry is keyed by the data identity, the mask identity, the lag, and
//! the configuration: the ordered pair, the conditioning set, and the interval
//! level. The cache keeps at most `capacity` entries. Complete rows that every
//! test in a cohort shares are accumulated once; each test then reads its
//! submatrix. Testwise deletion drops a mask-valid row only for the test whose
//! own variables are non-finite there. A stored covariance is refused when a
//! later pair or conditioning set keeps a different row set.
//!
//! The columns are already the lag window. `lag` does not shift them; it only
//! stops a covariance from one lag answering another. The scalar Fisher-z
//! interval stays in the analytic module.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::VecDeque;

const COHORT_CHANGED: &str =
    "substituting a Fisher-Z covariance for a cohort that changed is refused";
const IDENTITY_BOUND: &str = "a Fisher-Z covariance is bound to one data identity, mask, and lag";
const COVERAGE_GAP: &str =
    "a complete-data Fisher-Z covariance does not cover this pair or conditioning set";
const SINGULAR: &str = "singular Fisher-Z covariance";
const TOO_FEW_ROWS: &str = "Fisher-Z needs more complete rows than the conditioning set plus three";

use antecedent_kernels::{norm_sf, update_scatter};

use super::analytic::analytic_parcorr_ci;
use crate::error::StatsError;
use crate::gram::invert_square;

/// Ordered pair and conditioning columns for one Fisher-Z test.
///
/// The conditioning slice is a set: order and duplicate indexes do not change
/// the test. The pair keeps the caller's order, so `(x, y)` and `(y, x)` are
/// different cache entries.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct FisherTestSpec {
    /// Column index of the first variable.
    pub x: usize,
    /// Column index of the second variable.
    pub y: usize,
    /// Conditioning column indexes.
    pub conditioning: Vec<usize>,
}

/// Sample a Fisher-Z batch reads, plus the identities that bound its cache entry.
#[derive(Clone, Copy, Debug)]
pub struct FisherSample<'a> {
    /// Equal-length columns. A non-finite value is missing for tests that use it.
    pub columns: &'a [&'a [f64]],
    /// Content identity of `columns`. Values may be reused only under this identity.
    pub data_id: u64,
    /// Identity of `mask`. A different mask is a different entry even when the bits match.
    pub mask_id: u64,
    /// Row inclusion. `None` treats every row as eligible. `false` drops that row for every test.
    pub mask: Option<&'a [bool]>,
    /// Lag of this window. Compared, not applied.
    pub lag: u32,
    /// Two-sided Fisher-z interval level in `(0, 1)`.
    pub level: f64,
}

/// One batch of Fisher-Z tests against a [`FisherSample`].
#[derive(Clone, Copy, Debug)]
pub struct FisherBatch<'a> {
    /// Sample and cache identity.
    pub sample: FisherSample<'a>,
    /// Tests, answered in this order.
    pub tests: &'a [FisherTestSpec],
}

/// Configuration half of a cache key: pair, conditioning set, and interval level.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct FisherConfiguration {
    /// First variable of the pair, in the caller's order.
    pub x: usize,
    /// Second variable of the pair, in the caller's order.
    pub y: usize,
    /// Conditioning set, sorted and de-duplicated.
    pub conditioning: Vec<usize>,
    /// [`f64::to_bits`] of the interval level.
    pub level_bits: u64,
}

/// Cache key: data identity, mask identity, lag, and configuration.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct FisherCacheKey {
    /// [`FisherSample::data_id`].
    pub data_id: u64,
    /// [`FisherSample::mask_id`].
    pub mask_id: u64,
    /// [`FisherSample::lag`].
    pub lag: u32,
    /// Pair, conditioning set, and interval level.
    pub configuration: FisherConfiguration,
}

impl FisherCacheKey {
    /// Key for `spec` on this data, mask, lag, and level.
    ///
    /// Conditioning order and duplicates are ignored. Column bounds are not
    /// checked here; [`FisherCovarianceCache::run`] refuses a spec the sample
    /// cannot index.
    #[must_use]
    pub fn new(data_id: u64, mask_id: u64, lag: u32, spec: &FisherTestSpec, level: f64) -> Self {
        Self {
            data_id,
            mask_id,
            lag,
            configuration: FisherConfiguration {
                x: spec.x,
                y: spec.y,
                conditioning: sorted_conditioning(&spec.conditioning),
                level_bits: level.to_bits(),
            },
        }
    }
}

/// Where a stored scatter got its rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum CovarianceOrigin {
    /// Every mask-valid row was finite on the variables in the scatter.
    CompleteData,
    /// At least one mask-valid row was dropped because a covered variable was non-finite.
    Testwise,
}

/// Fisher-Z partial correlation, normal test, and analytic interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FisherZResult {
    /// Partial correlation of the pair given the conditioning set.
    pub r: f64,
    /// Fisher `z`, `atanh(r)`.
    pub z: f64,
    /// `|z| * sqrt(n − q − 3)`.
    pub statistic: f64,
    /// Two-sided standard-normal p-value.
    pub p_value: f64,
    /// Rows kept for this test after the mask and testwise deletion.
    pub n: usize,
    /// Residual degrees of freedom `n − q − 2` passed to the Fisher-z interval.
    pub df: f64,
    /// Interval at the sample's level, `(lower, upper)`.
    pub ci: (f64, f64),
}

/// Answers of one [`FisherCovarianceCache::run`], in request order.
#[derive(Clone, Debug, PartialEq)]
pub struct FisherBatchResult {
    /// One result per requested test.
    pub results: Vec<FisherZResult>,
    /// Scatter formations. Tests that share a cohort count as one. A full cache hit is zero.
    pub covariance_passes: usize,
}

/// One retained covariance and the test it was stored under.
#[derive(Clone, Debug)]
pub struct FisherCacheEntry {
    /// Data, mask, lag, and configuration.
    pub key: FisherCacheKey,
    /// `CompleteData` when the mask-valid sample was finite on every covered column.
    pub origin: CovarianceOrigin,
    /// Ascending row indexes the scatter was accumulated on.
    pub rows: Vec<usize>,
    /// Covered columns, ascending. A test outside this set cannot use the scatter.
    pub columns: Vec<usize>,
    /// Prepared pair and conditioning set.
    pub spec: FisherTestSpec,
    /// Fisher-Z answer for `spec` at the key's level.
    pub result: FisherZResult,
    scatter: Vec<f64>,
}

impl FisherCacheEntry {
    /// Partial correlation of `spec` from this scatter.
    ///
    /// # Errors
    ///
    /// [`StatsError::Shape`] when the sample or the residual degrees of freedom are unusable.
    /// [`StatsError::Unsupported`] when the data, mask, or lag differ, when `spec`'s complete
    /// rows differ from [`Self::rows`], when a requested column is outside [`Self::columns`],
    /// or when the submatrix is singular.
    #[must_use = "a refused Fisher-Z covariance must not be read as a result"]
    pub fn reuse(
        &self,
        sample: &FisherSample<'_>,
        spec: &FisherTestSpec,
    ) -> Result<FisherZResult, StatsError> {
        let n = validate_sample(sample)?;
        if sample.data_id != self.key.data_id
            || sample.mask_id != self.key.mask_id
            || sample.lag != self.key.lag
        {
            return Err(StatsError::Unsupported { message: IDENTITY_BOUND });
        }
        let spec = prepare_spec(spec, sample.columns.len())?;
        let rows = included_rows(sample, n, &spec_vars(&spec));
        if rows != self.rows {
            return Err(StatsError::Unsupported { message: COHORT_CHANGED });
        }
        if spec == self.spec && sample.level.to_bits() == self.key.configuration.level_bits {
            return Ok(self.result);
        }
        let local = positions(&self.columns, &spec)?;
        let r = partial_correlation(&self.scatter, self.columns.len(), &local)?;
        fisher_z_result(r, rows.len(), spec.conditioning.len(), sample.level)
    }
}

/// Bounded Fisher-Z covariance cache.
///
/// Eviction drops the oldest entry. Repeating a key refreshes that entry and
/// does not grow the cache. Capacity zero computes each batch and retains nothing.
#[derive(Clone, Debug)]
pub struct FisherCovarianceCache {
    capacity: usize,
    entries: VecDeque<FisherCacheEntry>,
}

impl FisherCovarianceCache {
    /// Cache that retains at most `capacity` entries.
    #[must_use]
    pub const fn with_capacity(capacity: usize) -> Self {
        Self { capacity, entries: VecDeque::new() }
    }

    /// Entries currently retained.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether [`Self::len`] is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Maximum number of retained entries.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Entry for `key`, if it has not been evicted.
    #[must_use]
    pub fn get(&self, key: &FisherCacheKey) -> Option<&FisherCacheEntry> {
        self.entries.iter().find(|entry| &entry.key == key)
    }

    /// Run `batch`, reusing an entry only when its cohort is still that test's row set.
    ///
    /// # Errors
    ///
    /// [`StatsError::Shape`] for a bad sample, an illegal spec, or too few complete rows.
    /// [`StatsError::Unsupported`] for a singular submatrix, or when a stored entry's data
    /// identity matches but its rows are not the rows this test would keep now.
    #[must_use = "a refused Fisher-Z cohort must not be read as a result"]
    pub fn run(&mut self, batch: &FisherBatch<'_>) -> Result<FisherBatchResult, StatsError> {
        let n = validate_sample(&batch.sample)?;
        let mut answers: Vec<(usize, FisherZResult)> = Vec::with_capacity(batch.tests.len());
        let mut pending: Vec<Pending> = Vec::new();
        for (order, spec) in batch.tests.iter().enumerate() {
            let spec = prepare_spec(spec, batch.sample.columns.len())?;
            let rows = included_rows(&batch.sample, n, &spec_vars(&spec));
            let key = FisherCacheKey::new(
                batch.sample.data_id,
                batch.sample.mask_id,
                batch.sample.lag,
                &spec,
                batch.sample.level,
            );
            if let Some(entry) = self.get(&key) {
                if entry.rows != rows {
                    return Err(StatsError::Unsupported { message: COHORT_CHANGED });
                }
                answers.push((order, entry.result));
            } else {
                pending.push(Pending { order, spec, key, rows });
            }
        }
        let (built, covariance_passes) = covariances_for_misses(&batch.sample, n, pending)?;
        for item in built {
            answers.push((item.order, item.entry.result));
            self.insert(item.entry);
        }
        answers.sort_unstable_by_key(|(order, _)| *order);
        if answers.len() != batch.tests.len() {
            return Err(StatsError::Shape { message: "Fisher-Z batch lost a test" });
        }
        Ok(FisherBatchResult {
            results: answers.into_iter().map(|(_, result)| result).collect(),
            covariance_passes,
        })
    }

    fn insert(&mut self, entry: FisherCacheEntry) {
        if self.capacity == 0 {
            return;
        }
        if let Some(pos) = self.entries.iter().position(|kept| kept.key == entry.key) {
            self.entries.remove(pos);
        }
        while self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }
}

struct Pending {
    order: usize,
    spec: FisherTestSpec,
    key: FisherCacheKey,
    rows: Vec<usize>,
}

struct Built {
    order: usize,
    entry: FisherCacheEntry,
}

fn covariances_for_misses(
    sample: &FisherSample<'_>,
    n: usize,
    pending: Vec<Pending>,
) -> Result<(Vec<Built>, usize), StatsError> {
    for item in &pending {
        fisher_width(item.rows.len(), item.spec.conditioning.len())?;
    }
    let groups = group_by_rows(pending);
    let passes = groups.len();
    let mask_n = mask_valid_count(n, sample.mask);
    let mut built = Vec::new();
    for (rows, members) in groups {
        let specs: Vec<FisherTestSpec> = members.iter().map(|item| item.spec.clone()).collect();
        let columns = union_columns(&specs);
        let scatter = scatter(sample.columns, &rows, &columns)?;
        let origin = if rows.len() == mask_n {
            CovarianceOrigin::CompleteData
        } else {
            CovarianceOrigin::Testwise
        };
        for item in members {
            let local = positions(&columns, &item.spec)?;
            let r = partial_correlation(&scatter, columns.len(), &local)?;
            let result =
                fisher_z_result(r, rows.len(), item.spec.conditioning.len(), sample.level)?;
            built.push(Built {
                order: item.order,
                entry: FisherCacheEntry {
                    key: item.key,
                    origin,
                    rows: rows.clone(),
                    columns: columns.clone(),
                    spec: item.spec,
                    result,
                    scatter: scatter.clone(),
                },
            });
        }
    }
    Ok((built, passes))
}

fn group_by_rows(pending: Vec<Pending>) -> Vec<(Vec<usize>, Vec<Pending>)> {
    let mut groups: Vec<(Vec<usize>, Vec<Pending>)> = Vec::new();
    for item in pending {
        let found = groups.iter().position(|(rows, _)| *rows == item.rows);
        if let Some(index) = found {
            groups[index].1.push(item);
        } else {
            groups.push((item.rows.clone(), vec![item]));
        }
    }
    groups
}

fn validate_sample(sample: &FisherSample<'_>) -> Result<usize, StatsError> {
    if sample.columns.is_empty() {
        return Err(StatsError::Shape { message: "no columns" });
    }
    if !(sample.level > 0.0 && sample.level < 1.0) {
        return Err(StatsError::Shape { message: "Fisher-Z confidence level must lie in (0, 1)" });
    }
    let n = sample.columns[0].len();
    if sample.columns.iter().any(|column| column.len() != n) {
        return Err(StatsError::Shape { message: "column length mismatch" });
    }
    if sample.mask.is_some_and(|mask| mask.len() != n) {
        return Err(StatsError::Shape { message: "mask length mismatch" });
    }
    Ok(n)
}

fn prepare_spec(spec: &FisherTestSpec, ncols: usize) -> Result<FisherTestSpec, StatsError> {
    if spec.x >= ncols || spec.y >= ncols || spec.x == spec.y {
        return Err(StatsError::Shape {
            message: "Fisher-Z pair is outside the columns or names one column twice",
        });
    }
    let conditioning = sorted_conditioning(&spec.conditioning);
    if conditioning.iter().any(|&column| column >= ncols) {
        return Err(StatsError::Shape {
            message: "Fisher-Z conditioning column is outside the columns",
        });
    }
    if conditioning.contains(&spec.x) || conditioning.contains(&spec.y) {
        return Err(StatsError::Shape { message: "Fisher-Z conditioning repeats the pair" });
    }
    Ok(FisherTestSpec { x: spec.x, y: spec.y, conditioning })
}

fn sorted_conditioning(conditioning: &[usize]) -> Vec<usize> {
    let mut conditioning = conditioning.to_vec();
    conditioning.sort_unstable();
    conditioning.dedup();
    conditioning
}

fn spec_vars(spec: &FisherTestSpec) -> Vec<usize> {
    let mut vars = Vec::with_capacity(2 + spec.conditioning.len());
    vars.push(spec.x);
    vars.push(spec.y);
    vars.extend_from_slice(&spec.conditioning);
    vars
}

fn union_columns(specs: &[FisherTestSpec]) -> Vec<usize> {
    let mut columns = Vec::new();
    for spec in specs {
        for column in spec_vars(spec) {
            if !columns.contains(&column) {
                columns.push(column);
            }
        }
    }
    columns.sort_unstable();
    columns
}

fn mask_valid_count(n: usize, mask: Option<&[bool]>) -> usize {
    match mask {
        None => n,
        Some(mask) => mask.iter().filter(|&&bit| bit).count(),
    }
}

fn included_rows(sample: &FisherSample<'_>, n: usize, vars: &[usize]) -> Vec<usize> {
    let mut rows = Vec::new();
    for row in 0..n {
        if sample.mask.is_some_and(|mask| !mask[row]) {
            continue;
        }
        let finite = vars.iter().all(|&column| sample.columns[column][row].is_finite());
        if finite {
            rows.push(row);
        }
    }
    rows
}

fn scatter(columns: &[&[f64]], rows: &[usize], vars: &[usize]) -> Result<Vec<f64>, StatsError> {
    let dim = vars.len();
    let cells = dim
        .checked_mul(dim)
        .ok_or(StatsError::Shape { message: "Fisher-Z covariance is too large" })?;
    let mut mean = vec![0.0; dim];
    let mut m2 = vec![0.0; cells];
    let mut observation = vec![0.0; dim];
    let mut seen = 0u64;
    for &row in rows {
        for (slot, &column) in observation.iter_mut().zip(vars) {
            *slot = columns[column][row];
        }
        if seen == 0 {
            mean.clone_from_slice(&observation);
            seen = 1;
            continue;
        }
        update_scatter(&mut mean, &mut m2, seen, &observation);
        seen += 1;
    }
    Ok(m2)
}

fn positions(columns: &[usize], spec: &FisherTestSpec) -> Result<Vec<usize>, StatsError> {
    let mut local = Vec::with_capacity(2 + spec.conditioning.len());
    for column in spec_vars(spec) {
        let Some(index) = columns.iter().position(|&covered| covered == column) else {
            return Err(StatsError::Unsupported { message: COVERAGE_GAP });
        };
        local.push(index);
    }
    Ok(local)
}

fn partial_correlation(scatter: &[f64], stride: usize, local: &[usize]) -> Result<f64, StatsError> {
    let k = local.len();
    let cells = k
        .checked_mul(k)
        .ok_or(StatsError::Shape { message: "Fisher-Z covariance is too large" })?;
    let mut sub = vec![0.0; cells];
    for (row, &source_row) in local.iter().enumerate() {
        for (col, &source_col) in local.iter().enumerate() {
            sub[row * k + col] = scatter[source_row * stride + source_col];
        }
    }
    if k == 2 {
        let sxx = sub[0];
        let sxy = sub[1];
        let syy = sub[3];
        if !(sxx > 0.0 && syy > 0.0) {
            return Err(StatsError::Unsupported { message: SINGULAR });
        }
        let r = sxy / (sxx * syy).sqrt();
        if !r.is_finite() {
            return Err(StatsError::Unsupported { message: SINGULAR });
        }
        return Ok(r.clamp(-1.0, 1.0));
    }
    let inv = invert_square(&sub, k).ok_or(StatsError::Unsupported { message: SINGULAR })?;
    let pxx = inv[0];
    let pxy = inv[1];
    let pyy = inv[k + 1];
    if !(pxx > 0.0 && pyy > 0.0) {
        return Err(StatsError::Unsupported { message: SINGULAR });
    }
    let r = -pxy / (pxx * pyy).sqrt();
    if !r.is_finite() {
        return Err(StatsError::Unsupported { message: SINGULAR });
    }
    Ok(r.clamp(-1.0, 1.0))
}

fn fisher_width(n: usize, q: usize) -> Result<usize, StatsError> {
    let Some(denominator) = q.checked_add(3) else {
        return Err(StatsError::Shape { message: "Fisher-Z conditioning set is too large" });
    };
    match n.checked_sub(denominator) {
        Some(width) if width > 0 => Ok(width),
        _ => Err(StatsError::Shape { message: TOO_FEW_ROWS }),
    }
}

fn fisher_z_result(r: f64, n: usize, q: usize, level: f64) -> Result<FisherZResult, StatsError> {
    let width = fisher_width(n, q)?;
    let r = r.clamp(-1.0 + 1e-15, 1.0 - 1e-15);
    let z = 0.5 * ((1.0 + r) / (1.0 - r)).ln();
    let statistic = z.abs() * (width as f64).sqrt();
    let p_value = (2.0 * norm_sf(statistic)).clamp(0.0, 1.0);
    let df = (width + 1) as f64;
    let ci = analytic_parcorr_ci(r, df, level);
    Ok(FisherZResult { r, z, statistic, p_value, n, df, ci })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(x: usize, y: usize, conditioning: &[usize]) -> FisherTestSpec {
        FisherTestSpec { x, y, conditioning: conditioning.to_vec() }
    }

    fn sample<'a>(
        columns: &'a [&'a [f64]],
        data_id: u64,
        mask_id: u64,
        mask: Option<&'a [bool]>,
        lag: u32,
        level: f64,
    ) -> FisherSample<'a> {
        FisherSample { columns, data_id, mask_id, mask, lag, level }
    }

    fn pearson(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;
        let mut sxx = 0.0;
        let mut syy = 0.0;
        let mut sxy = 0.0;
        for (left, right) in x.iter().zip(y) {
            let dx = left - mean_x;
            let dy = right - mean_y;
            sxx += dx * dx;
            syy += dy * dy;
            sxy += dx * dy;
        }
        sxy / (sxx * syy).sqrt()
    }

    fn gather(column: &[f64], rows: &[usize]) -> Vec<f64> {
        rows.iter().map(|&row| column[row]).collect()
    }

    #[test]
    fn level_must_lie_strictly_inside_zero_and_one() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let columns: [&[f64]; 1] = [&x];
        for level in [0.0, 1.0, -0.1, 1.5, f64::NAN, f64::INFINITY] {
            let refused = validate_sample(&sample(&columns, 1, 1, None, 0, level));
            assert!(refused.is_err(), "level {level} must be refused");
        }
        for level in [f64::MIN_POSITIVE, 0.5, 0.95, 1.0 - f64::EPSILON] {
            assert_eq!(validate_sample(&sample(&columns, 1, 1, None, 0, level)).unwrap(), 4);
        }
    }

    #[test]
    fn complete_data_batch_uses_one_covariance_pass() {
        let x = [1.0, 3.0, 2.0, 5.0, 4.0, 7.0];
        let y = [2.0, 2.0, 4.0, 3.0, 6.0, 5.0];
        let z = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let columns = [&x[..], &y[..], &z[..]];
        let tests = [pair(0, 1, &[]), pair(0, 2, &[]), pair(1, 2, &[])];
        let observed = sample(&columns, 1, 7, None, 0, 0.95);
        let mut cache = FisherCovarianceCache::with_capacity(8);
        let out = cache.run(&FisherBatch { sample: observed, tests: &tests }).unwrap();
        assert_eq!(out.covariance_passes, 1);
        assert_eq!(cache.len(), 3);
        assert!((out.results[0].r - pearson(&x, &y)).abs() < 1e-9);
        assert_eq!(out.results[0].n, 6);
        assert!(out.results[0].ci.0 < out.results[0].r && out.results[0].r < out.results[0].ci.1);
        let stored = cache.get(&FisherCacheKey::new(1, 7, 0, &tests[0], 0.95)).unwrap();
        assert_eq!(stored.origin, CovarianceOrigin::CompleteData);
        let sibling = stored.reuse(&observed, &tests[1]).unwrap();
        assert!((sibling.r - out.results[1].r).abs() < 1e-12);
        assert!((sibling.r - pearson(&x, &z)).abs() < 1e-9);
    }

    #[test]
    fn cache_hit_skips_the_covariance_pass() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = [1.0, 1.0, 2.0, 3.0, 5.0, 8.0];
        let columns = [&x[..], &y[..]];
        let tests = [pair(0, 1, &[])];
        let observed = sample(&columns, 4, 4, None, 1, 0.95);
        let mut cache = FisherCovarianceCache::with_capacity(4);
        let first = cache.run(&FisherBatch { sample: observed, tests: &tests }).unwrap();
        let second = cache.run(&FisherBatch { sample: observed, tests: &tests }).unwrap();
        assert_eq!(first.covariance_passes, 1);
        assert_eq!(second.covariance_passes, 0);
        assert!((first.results[0].r - second.results[0].r).abs() < 1e-15);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn covariance_pass_keeps_correlation_under_a_shared_offset() {
        let offset = 1e12;
        let x: Vec<f64> = [0.0, 1.0, 2.0, 3.0].iter().map(|v| offset + v).collect();
        let y: Vec<f64> = [0.0, 1.0, 2.0, 4.0].iter().map(|v| offset + v).collect();
        let expected = pearson(&[0.0, 1.0, 2.0, 3.0], &[0.0, 1.0, 2.0, 4.0]);
        let columns = [&x[..], &y[..]];
        let tests = [pair(0, 1, &[])];
        let observed = sample(&columns, 9, 9, None, 0, 0.95);
        let mut cache = FisherCovarianceCache::with_capacity(1);
        let out = cache.run(&FisherBatch { sample: observed, tests: &tests }).unwrap();
        assert!((out.results[0].r - expected).abs() < 1e-9);
        assert!((expected - 6.5 / (5.0 * 8.75_f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn partial_correlation_matches_the_three_correlation_formula() {
        let x = [1.0, 3.0, 2.0, 5.0, 4.0, 7.0];
        let y = [2.0, 2.0, 4.0, 3.0, 6.0, 5.0];
        let z = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rxy = pearson(&x, &y);
        let rxz = pearson(&x, &z);
        let ryz = pearson(&y, &z);
        let expected = (rxy - rxz * ryz) / ((1.0 - rxz * rxz) * (1.0 - ryz * ryz)).sqrt();
        let columns = [&x[..], &y[..], &z[..]];
        let tests = [pair(0, 1, &[2])];
        let observed = sample(&columns, 3, 3, None, 0, 0.9);
        let mut cache = FisherCovarianceCache::with_capacity(2);
        let out = cache.run(&FisherBatch { sample: observed, tests: &tests }).unwrap();
        assert!((out.results[0].r - expected).abs() < 1e-9);
        assert!((out.results[0].r - rxy).abs() > 1e-3);
        assert_eq!(out.results[0].n, 6);
        let width = f64::from(6 - 1 - 3);
        assert!((out.results[0].statistic - out.results[0].z.abs() * width.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn uncorrelated_pair_has_unit_pvalue() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [1.0, 2.0, 2.0, 1.0];
        let columns = [&x[..], &y[..]];
        let tests = [pair(0, 1, &[])];
        let observed = sample(&columns, 1, 1, None, 0, 0.95);
        let mut cache = FisherCovarianceCache::with_capacity(1);
        let out = cache.run(&FisherBatch { sample: observed, tests: &tests }).unwrap();
        assert!(out.results[0].r.abs() < 1e-12);
        assert!((out.results[0].p_value - 1.0).abs() < 1e-12);
        assert!(out.results[0].statistic.abs() < 1e-12);
    }

    #[test]
    fn cache_entry_is_keyed_by_data_mask_lag_and_configuration() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = [1.0, 2.0, 1.0, 2.0, 3.0, 3.0];
        let columns = [&x[..], &y[..]];
        let tests = [pair(0, 1, &[])];
        let mut cache = FisherCovarianceCache::with_capacity(8);
        let base = sample(&columns, 10, 20, None, 0, 0.95);
        cache.run(&FisherBatch { sample: base, tests: &tests }).unwrap();
        cache
            .run(&FisherBatch { sample: sample(&columns, 11, 20, None, 0, 0.95), tests: &tests })
            .unwrap();
        let mask = [true, true, true, true, true, false];
        cache
            .run(&FisherBatch {
                sample: sample(&columns, 10, 21, Some(&mask), 0, 0.95),
                tests: &tests,
            })
            .unwrap();
        cache
            .run(&FisherBatch { sample: sample(&columns, 10, 20, None, 2, 0.95), tests: &tests })
            .unwrap();
        cache
            .run(&FisherBatch { sample: sample(&columns, 10, 20, None, 0, 0.8), tests: &tests })
            .unwrap();
        assert_eq!(cache.len(), 5);
        assert!(cache.get(&FisherCacheKey::new(10, 20, 0, &tests[0], 0.95)).is_some());
        assert!(cache.get(&FisherCacheKey::new(11, 20, 0, &tests[0], 0.95)).is_some());
        assert!(cache.get(&FisherCacheKey::new(10, 21, 0, &tests[0], 0.95)).is_some());
        assert!(cache.get(&FisherCacheKey::new(10, 20, 2, &tests[0], 0.95)).is_some());
        assert!(cache.get(&FisherCacheKey::new(10, 20, 0, &tests[0], 0.8)).is_some());
        let stored = cache.get(&FisherCacheKey::new(10, 20, 0, &tests[0], 0.95)).unwrap();
        let err = stored.reuse(&sample(&columns, 10, 20, None, 2, 0.95), &tests[0]).unwrap_err();
        assert_eq!(err, StatsError::Unsupported { message: IDENTITY_BOUND });
        let masked = cache.get(&FisherCacheKey::new(10, 21, 0, &tests[0], 0.95)).unwrap();
        assert_eq!(masked.origin, CovarianceOrigin::CompleteData);
        assert_eq!(masked.rows, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn different_pair_or_conditioning_set_is_a_different_entry() {
        let x = [1.0, 3.0, 2.0, 5.0, 4.0, 7.0];
        let y = [2.0, 2.0, 4.0, 3.0, 6.0, 5.0];
        let z = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let columns = [&x[..], &y[..], &z[..]];
        let marginal = pair(0, 1, &[]);
        let swapped = pair(1, 0, &[]);
        let conditional = pair(0, 1, &[2, 2]);
        let reordered = pair(0, 1, &[2]);
        let observed = sample(&columns, 1, 1, None, 0, 0.95);
        let mut cache = FisherCovarianceCache::with_capacity(8);
        cache
            .run(&FisherBatch { sample: observed, tests: &[marginal.clone(), swapped.clone()] })
            .unwrap();
        let first = FisherCacheKey::new(1, 1, 0, &marginal, 0.95);
        let second = FisherCacheKey::new(1, 1, 0, &swapped, 0.95);
        assert_ne!(first, second);
        assert!(cache.get(&first).is_some());
        assert!(cache.get(&second).is_some());
        cache
            .run(&FisherBatch { sample: observed, tests: &[conditional.clone(), reordered] })
            .unwrap();
        let conditioned = FisherCacheKey::new(1, 1, 0, &conditional, 0.95);
        assert_ne!(conditioned, first);
        assert_eq!(cache.get(&conditioned).unwrap().spec.conditioning, [2]);
        let marginal_entry = cache.get(&first).unwrap();
        let err = marginal_entry.reuse(&observed, &conditional).unwrap_err();
        assert_eq!(err, StatsError::Unsupported { message: COVERAGE_GAP });
    }

    #[test]
    fn testwise_deletion_drops_incomplete_rows_for_that_test_only() {
        let x = [1.0, f64::NAN, 2.0, 3.0, 4.0];
        let y = [1.0, 100.0, 2.0, 3.0, 4.0];
        let z = [1.0, 0.0, 2.0, 3.0, 4.0];
        let columns = [&x[..], &y[..], &z[..]];
        let tests = [pair(1, 2, &[]), pair(0, 1, &[])];
        let observed = sample(&columns, 5, 5, None, 0, 0.95);
        let mut cache = FisherCovarianceCache::with_capacity(4);
        let out = cache.run(&FisherBatch { sample: observed, tests: &tests }).unwrap();
        assert_eq!(out.covariance_passes, 2);
        assert_eq!(out.results[0].n, 5);
        assert_eq!(out.results[1].n, 4);
        assert!((out.results[0].r - pearson(&y, &z)).abs() < 1e-9);
        let reduced = pearson(&gather(&y, &[0, 2, 3, 4]), &gather(&z, &[0, 2, 3, 4]));
        assert!((reduced - 1.0).abs() < 1e-12);
        assert!((out.results[0].r - reduced).abs() > 0.1);
        let yz_entry = cache.get(&FisherCacheKey::new(5, 5, 0, &tests[0], 0.95)).unwrap();
        let xy_entry = cache.get(&FisherCacheKey::new(5, 5, 0, &tests[1], 0.95)).unwrap();
        assert_eq!(yz_entry.origin, CovarianceOrigin::CompleteData);
        assert_eq!(xy_entry.origin, CovarianceOrigin::Testwise);
        assert_eq!(yz_entry.rows, [0, 1, 2, 3, 4]);
        assert_eq!(xy_entry.rows, [0, 2, 3, 4]);
        let kept = [0usize, 2, 3, 4];
        let xy_r = pearson(&gather(&x, &kept), &gather(&y, &kept));
        assert!((out.results[1].r - xy_r).abs() < 1e-9);
    }

    #[test]
    fn global_covariance_refuses_a_changed_cohort() {
        let mut x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = [1.0, 2.0, 1.0, 2.0, 3.0, 4.0];
        let z = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let tests = [pair(0, 1, &[]), pair(0, 2, &[])];
        let mut cache = FisherCovarianceCache::with_capacity(4);
        let first = {
            let columns = [&x[..], &y[..], &z[..]];
            let observed = sample(&columns, 8, 8, None, 0, 0.95);
            cache.run(&FisherBatch { sample: observed, tests: &tests }).unwrap()
        };
        assert_eq!(first.covariance_passes, 1);
        assert_eq!(cache.len(), 2);
        x[1] = f64::NAN;
        let columns = [&x[..], &y[..], &z[..]];
        let changed = sample(&columns, 8, 8, None, 0, 0.95);
        let err = cache.run(&FisherBatch { sample: changed, tests: &tests }).unwrap_err();
        assert_eq!(err, StatsError::Unsupported { message: COHORT_CHANGED });
        assert_eq!(cache.len(), 2);
        let stored = cache.get(&FisherCacheKey::new(8, 8, 0, &tests[0], 0.95)).unwrap();
        assert_eq!(stored.rows, [0, 1, 2, 3, 4, 5]);
        let reuse_err = stored.reuse(&changed, &tests[0]).unwrap_err();
        assert_eq!(reuse_err, StatsError::Unsupported { message: COHORT_CHANGED });
        let mask = [true, false, true, true, true, true];
        let masked = sample(&columns, 8, 8, Some(&mask), 0, 0.95);
        let mask_err = stored.reuse(&masked, &tests[0]).unwrap_err();
        assert_eq!(mask_err, StatsError::Unsupported { message: COHORT_CHANGED });
    }

    #[test]
    fn cache_len_stays_within_capacity() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = [1.0, 2.0, 4.0, 4.0, 5.0, 7.0];
        let z = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let columns = [&x[..], &y[..], &z[..]];
        let tests = [pair(0, 1, &[]), pair(0, 2, &[]), pair(1, 2, &[])];
        let observed = sample(&columns, 2, 2, None, 0, 0.95);
        let mut bounded = FisherCovarianceCache::with_capacity(2);
        let out = bounded.run(&FisherBatch { sample: observed, tests: &tests }).unwrap();
        assert_eq!(bounded.len(), 2);
        assert!(bounded.capacity() >= bounded.len());
        assert!(bounded.get(&FisherCacheKey::new(2, 2, 0, &tests[0], 0.95)).is_none());
        assert!(bounded.get(&FisherCacheKey::new(2, 2, 0, &tests[1], 0.95)).is_some());
        assert!(bounded.get(&FisherCacheKey::new(2, 2, 0, &tests[2], 0.95)).is_some());
        let mut fresh = FisherCovarianceCache::with_capacity(8);
        let full = fresh.run(&FisherBatch { sample: observed, tests: &tests }).unwrap();
        for (left, right) in out.results.iter().zip(&full.results) {
            assert!((left.r - right.r).abs() < 1e-15);
        }
        let mut none = FisherCovarianceCache::with_capacity(0);
        let computed = none.run(&FisherBatch { sample: observed, tests: &tests }).unwrap();
        assert!(none.is_empty());
        let again = none.run(&FisherBatch { sample: observed, tests: &tests }).unwrap();
        assert_eq!(again.covariance_passes, 1);
        assert!((computed.results[0].r - full.results[0].r).abs() < 1e-15);
    }

    #[test]
    fn fisher_z_refuses_too_few_complete_rows() {
        let x = [1.0, 2.0, 3.0];
        let y = [1.0, 2.0, 4.0];
        let columns = [&x[..], &y[..]];
        let tests = [pair(0, 1, &[])];
        let observed = sample(&columns, 1, 1, None, 0, 0.95);
        let mut cache = FisherCovarianceCache::with_capacity(2);
        let err = cache.run(&FisherBatch { sample: observed, tests: &tests }).unwrap_err();
        assert_eq!(err, StatsError::Shape { message: TOO_FEW_ROWS });
        assert!(cache.is_empty());
    }

    #[test]
    fn singular_covariance_is_refused() {
        let x = [1.0, 1.0, 1.0, 1.0];
        let y = [1.0, 2.0, 3.0, 4.0];
        let columns = [&x[..], &y[..]];
        let tests = [pair(0, 1, &[])];
        let observed = sample(&columns, 1, 1, None, 0, 0.95);
        let mut cache = FisherCovarianceCache::with_capacity(2);
        let err = cache.run(&FisherBatch { sample: observed, tests: &tests }).unwrap_err();
        assert_eq!(err, StatsError::Unsupported { message: SINGULAR });
        assert!(cache.is_empty());
    }
}
