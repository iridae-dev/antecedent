//! Retained conditional-independence statistics, keyed by the exact input coordinate.
//!
//! A statistic is reused only when the test, its settings, the random seed, the kernel
//! policy, the column indexes the test is handed and the *content* of every column it reads
//! are all identical. The key is the 256-bit digest of that complete input, so a lookup
//! cannot collide and no retained-input validation is needed: a changed value at an unchanged
//! shape, an equal label over different rows, or a permuted row order is a different key.
//!
//! The store is the single place a statistic is computed or served from. A hit is answered
//! with the bit pattern computed earlier and is never counted as executed work; a miss is
//! counted as attempted when it is handed to the test and as completed when the test
//! returns.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_stats::{
    CiBatchRequest, CiBatchResult, CiPreparationPlan, CiQuery, CiResult, CiWorkspace,
    ConditionalIndependenceTest, ConfidenceMethod, PreparedCiTest, SignificanceMethod, StatsError,
};

use super::identity::{Digest, SELECTIVE_IDENTITY_VERSION, column_digest};
use super::refusal::SelectiveDiscoveryError;

fn statistics_invalid(detail: &'static str, remedy: impl Into<String>) -> SelectiveDiscoveryError {
    SelectiveDiscoveryError::refused(
        antecedent_core::reason_code!("discovery_statistics_invalid"),
        detail,
        remedy,
    )
}

const COORDINATE_TAG: &str = "antecedent.selective_discovery.ci_coordinate.v1";
const RECEIPTS_TAG: &str = "antecedent.selective_discovery.receipts.v1";

/// Most statistics one store retains. Beyond it a test still runs and is still counted, but
/// its result is not retained (and is reported in [`StoreCounters::not_retained`]).
pub const DEFAULT_STORE_CAPACITY: usize = 4_000_000;
/// Most receipts a portable state carries.
pub const MAX_PORTABLE_RECEIPTS: usize = 250_000;

/// One retained statistic with the digest of the coordinate it was computed at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StatisticReceipt {
    /// Digest of the complete input coordinate.
    pub coordinate: Digest,
    /// Test statistic.
    pub statistic: f64,
    /// P-value (or posterior probability for the Bayesian tests).
    pub p_value: f64,
    /// Residual degrees of freedom.
    pub df: f64,
    /// Optional analytic interval for the statistic.
    pub interval: Option<(f64, f64)>,
}

impl StatisticReceipt {
    fn from_result(coordinate: Digest, result: CiResult) -> Self {
        Self {
            coordinate,
            statistic: result.statistic,
            p_value: result.p_value,
            df: result.df,
            interval: result.ci,
        }
    }

    fn to_result(self) -> CiResult {
        CiResult {
            statistic: self.statistic,
            p_value: self.p_value,
            df: self.df,
            ci: self.interval,
        }
    }

    /// Whether the receipt is a usable statistic: finite statistic, p-value in `[0, 1]`, no
    /// NaN degrees of freedom and finite interval endpoints.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.statistic.is_finite()
            && self.p_value.is_finite()
            && (0.0..=1.0).contains(&self.p_value)
            && !self.df.is_nan()
            && self.interval.is_none_or(|(lo, hi)| lo.is_finite() && hi.is_finite())
    }
}

/// Whether a miss computes the statistic or is a refusal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StoreMode {
    /// A miss runs the test and retains the result.
    Compute,
    /// A miss is refused: the run may only consume statistics already retained. Used by
    /// resume, where nothing may be computed.
    ReplayOnly,
}

/// Counters of one store since creation (or since it was forked).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StoreCounters {
    /// Tests handed to the test implementation.
    pub attempted: u64,
    /// Tests the implementation returned a result for.
    pub completed: u64,
    /// Requests answered from a retained statistic.
    pub reused: u64,
    /// Completed results not retained because the store was at capacity.
    pub not_retained: u64,
}

#[derive(Debug, Default)]
struct StoreInner {
    entries: HashMap<Digest, CiResult>,
}

/// Retained statistics of one stage lineage.
#[derive(Debug)]
pub struct StatisticStore {
    inner: Mutex<StoreInner>,
    mode: StoreMode,
    capacity: usize,
    attempted: AtomicU64,
    completed: AtomicU64,
    reused: AtomicU64,
    not_retained: AtomicU64,
    replay_miss: AtomicBool,
}

impl StatisticStore {
    /// An empty store that computes on a miss.
    #[must_use]
    pub fn new() -> Self {
        Self::with_mode(StoreMode::Compute, DEFAULT_STORE_CAPACITY)
    }

    fn with_mode(mode: StoreMode, capacity: usize) -> Self {
        Self {
            inner: Mutex::new(StoreInner::default()),
            mode,
            capacity,
            attempted: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            reused: AtomicU64::new(0),
            not_retained: AtomicU64::new(0),
            replay_miss: AtomicBool::new(false),
        }
    }

    /// An empty store with an explicit capacity (tests and bounded callers).
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_mode(StoreMode::Compute, capacity)
    }

    /// A replay-only store holding exactly `receipts`.
    ///
    /// # Errors
    ///
    /// A receipt that is not a usable statistic, a repeated coordinate, or more receipts
    /// than [`MAX_PORTABLE_RECEIPTS`].
    pub fn replay_only(receipts: &[StatisticReceipt]) -> Result<Self, SelectiveDiscoveryError> {
        let store = Self::with_mode(StoreMode::ReplayOnly, receipts.len().max(1));
        store.load(receipts)?;
        Ok(store)
    }

    /// A computing store seeded with `receipts` (a resume that may continue to compute).
    ///
    /// # Errors
    ///
    /// As [`Self::replay_only`].
    pub fn seeded(receipts: &[StatisticReceipt]) -> Result<Self, SelectiveDiscoveryError> {
        let store = Self::new();
        store.load(receipts)?;
        Ok(store)
    }

    fn load(&self, receipts: &[StatisticReceipt]) -> Result<(), SelectiveDiscoveryError> {
        if receipts.len() > MAX_PORTABLE_RECEIPTS {
            return Err(statistics_invalid(
                "selective_discovery.receipt_limit",
                format!("a portable state carries at most {MAX_PORTABLE_RECEIPTS} receipts"),
            ));
        }
        let mut inner = self.lock();
        for receipt in receipts {
            if !receipt.is_valid() {
                return Err(statistics_invalid(
                    "selective_discovery.receipt_invalid",
                    "a retained statistic is not finite or its p-value is outside [0, 1]; \
                     recompute it from the raw data",
                ));
            }
            if inner.entries.insert(receipt.coordinate, receipt.to_result()).is_some() {
                return Err(statistics_invalid(
                    "selective_discovery.receipt_repeated",
                    "one input coordinate carries two receipts; the state is not a record of \
                     one run",
                ));
            }
        }
        Ok(())
    }

    /// A copy holding the same statistics with zeroed counters and the same mode.
    ///
    /// Candidate runs execute on a fork so that a failed or cancelled update leaves the live
    /// store untouched.
    #[must_use]
    pub fn fork(&self) -> Self {
        let entries = self.lock().entries.clone();
        let fork = Self::with_mode(self.mode, self.capacity);
        fork.lock().entries = entries;
        fork
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, StoreInner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Statistics currently retained.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    /// Whether nothing is retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The mode.
    #[must_use]
    pub const fn mode(&self) -> StoreMode {
        self.mode
    }

    /// Whether a replay-only run asked for a statistic that was not retained.
    #[must_use]
    pub fn replay_missed(&self) -> bool {
        self.replay_miss.load(Ordering::SeqCst)
    }

    /// Counters since creation.
    #[must_use]
    pub fn counters(&self) -> StoreCounters {
        StoreCounters {
            attempted: self.attempted.load(Ordering::SeqCst),
            completed: self.completed.load(Ordering::SeqCst),
            reused: self.reused.load(Ordering::SeqCst),
            not_retained: self.not_retained.load(Ordering::SeqCst),
        }
    }

    /// Every retained statistic, ascending by coordinate.
    #[must_use]
    pub fn receipts(&self) -> Vec<StatisticReceipt> {
        let mut out: Vec<StatisticReceipt> = self
            .lock()
            .entries
            .iter()
            .map(|(coordinate, result)| StatisticReceipt::from_result(*coordinate, *result))
            .collect();
        out.sort_unstable_by(|a, b| a.coordinate.cmp(&b.coordinate));
        out
    }

    /// Identity of a receipt list (order as given; pass [`Self::receipts`] output).
    #[must_use]
    pub fn receipts_identity(receipts: &[StatisticReceipt]) -> Digest {
        let mut writer = CanonicalWriter::new(RECEIPTS_TAG);
        writer.word(SELECTIVE_IDENTITY_VERSION).word(receipts.len() as u64);
        for receipt in receipts {
            writer
                .bytes(&receipt.coordinate)
                .real(receipt.statistic)
                .real(receipt.p_value)
                .real(receipt.df);
            match receipt.interval {
                Some((lo, hi)) => writer.word(1).real(lo).real(hi),
                None => writer.word(0),
            };
        }
        writer.finish()
    }

    fn lookup(&self, keys: &[Digest]) -> Vec<Option<CiResult>> {
        let inner = self.lock();
        keys.iter().map(|key| inner.entries.get(key).copied()).collect()
    }

    fn insert(&self, key: Digest, result: CiResult) {
        let mut inner = self.lock();
        if inner.entries.len() >= self.capacity && !inner.entries.contains_key(&key) {
            self.not_retained.fetch_add(1, Ordering::SeqCst);
            return;
        }
        inner.entries.insert(key, result);
    }
}

impl Default for StatisticStore {
    fn default() -> Self {
        Self::new()
    }
}

fn write_significance(writer: &mut CanonicalWriter, significance: SignificanceMethod) {
    match significance {
        SignificanceMethod::Analytic => {
            writer.word(0);
        }
        SignificanceMethod::BlockShuffle { replicates, block_size } => {
            writer.word(1).word(u64::from(replicates)).word(block_size as u64);
        }
    }
}

fn write_confidence(writer: &mut CanonicalWriter, confidence: ConfidenceMethod) {
    match confidence {
        ConfidenceMethod::None => {
            writer.word(0);
        }
        ConfidenceMethod::Analytic { level } => {
            writer.word(1).real(level);
        }
    }
}

/// A conditional-independence test that serves repeated requests from a [`StatisticStore`].
///
/// It implements the same trait as the test it wraps, so any discovery algorithm runs on it
/// unchanged. A request is answered from the store only at an identical input coordinate;
/// every other request runs the wrapped test on exactly that request and retains the result.
/// Results are therefore bit-identical to a run on the wrapped test alone, which holds as
/// long as the wrapped test's result is a function of its request, the context seed and the
/// kernel policy (the contract every test of this library states: a permutation stream is
/// keyed by the query, never by its position in a batch).
pub struct SelectiveCi {
    inner: Arc<dyn ConditionalIndependenceTest + Send + Sync>,
    store: Arc<StatisticStore>,
    test_identity: Arc<str>,
}

impl std::fmt::Debug for SelectiveCi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectiveCi")
            .field("test_identity", &self.test_identity)
            .field("retained", &self.store.len())
            .finish_non_exhaustive()
    }
}

impl SelectiveCi {
    /// Wrap `inner`, identified by `test_identity`, over `store`.
    ///
    /// `test_identity` must name the test and every setting that changes its result (a
    /// permutation count, a salt, a bandwidth). Two tests that can return different results
    /// for one request must not share an identity.
    #[must_use]
    pub fn new(
        inner: Arc<dyn ConditionalIndependenceTest + Send + Sync>,
        store: Arc<StatisticStore>,
        test_identity: impl Into<Arc<str>>,
    ) -> Self {
        Self { inner, store, test_identity: test_identity.into() }
    }

    /// The store this test serves from.
    #[must_use]
    pub const fn store(&self) -> &Arc<StatisticStore> {
        &self.store
    }

    fn coordinates(
        &self,
        request: &CiBatchRequest<'_>,
        ctx: &ExecutionContext,
    ) -> Result<Vec<Digest>, StatsError> {
        let mut base = CanonicalWriter::new(COORDINATE_TAG);
        base.word(SELECTIVE_IDENTITY_VERSION).text(&self.test_identity);
        write_significance(&mut base, request.significance);
        write_confidence(&mut base, request.confidence);
        base.word(ctx.rng.master_seed());
        let policy = ctx.kernel_policy;
        base.word(u64::from(policy.allow_portable_optimized))
            .word(u64::from(policy.allow_arch_simd))
            .word(u64::from(policy.force_scalar));
        let mut digests: Vec<Option<Digest>> = vec![None; request.columns.len()];
        let mut column = |index: usize| -> Result<Digest, StatsError> {
            let slot = digests
                .get_mut(index)
                .ok_or(StatsError::Shape { message: "CI query column index out of range" })?;
            Ok(*slot.get_or_insert_with(|| column_digest(request.columns[index])))
        };
        let mut keys = Vec::with_capacity(request.queries.len());
        for query in request.queries {
            let end = query.z_start.checked_add(query.z_len).filter(|e| *e <= request.z_flat.len());
            let Some(end) = end else {
                return Err(StatsError::Shape { message: "CI query conditioning range" });
            };
            let z = &request.z_flat[query.z_start..end];
            let mut writer = base.clone();
            writer.word(query.x as u64).word(query.y as u64).word(z.len() as u64);
            for &index in z {
                writer.word(index as u64);
            }
            writer.bytes(&column(query.x)?).bytes(&column(query.y)?);
            for &index in z {
                writer.bytes(&column(index)?);
            }
            keys.push(writer.finish());
        }
        Ok(keys)
    }
}

impl ConditionalIndependenceTest for SelectiveCi {
    fn prepare(
        &self,
        columns: &[&[f64]],
        plan: &CiPreparationPlan,
        ctx: &ExecutionContext,
    ) -> Result<PreparedCiTest, StatsError> {
        self.inner.prepare(columns, plan, ctx)
    }

    fn test_batch(
        &self,
        prepared: &PreparedCiTest,
        request: &CiBatchRequest<'_>,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        prepared.ensure_compatible(request)?;
        let keys = self.coordinates(request, ctx)?;
        let mut results = self.store.lookup(&keys);
        let missing: Vec<usize> =
            results.iter().enumerate().filter(|(_, r)| r.is_none()).map(|(i, _)| i).collect();
        self.store.reused.fetch_add((keys.len() - missing.len()) as u64, Ordering::SeqCst);
        if !missing.is_empty() {
            if self.store.mode == StoreMode::ReplayOnly {
                self.store.replay_miss.store(true, Ordering::SeqCst);
                return Err(StatsError::Backend(
                    "selective_discovery.replay_miss: a statistic that was not retained was \
                     requested during replay"
                        .to_owned(),
                ));
            }
            let queries: Vec<CiQuery> = missing.iter().map(|&i| request.queries[i]).collect();
            let sub = CiBatchRequest {
                columns: request.columns,
                queries: &queries,
                z_flat: request.z_flat,
                significance: request.significance,
                confidence: request.confidence,
            };
            self.store.attempted.fetch_add(queries.len() as u64, Ordering::SeqCst);
            let computed = self.inner.test_batch(prepared, &sub, workspace, ctx)?;
            if computed.results.len() != queries.len() {
                return Err(StatsError::Shape { message: "CI test returned a different count" });
            }
            self.store.completed.fetch_add(queries.len() as u64, Ordering::SeqCst);
            for (&slot, result) in missing.iter().zip(computed.results) {
                self.store.insert(keys[slot], result);
                results[slot] = Some(result);
            }
        }
        let results = results
            .into_iter()
            .collect::<Option<Vec<CiResult>>>()
            .ok_or(StatsError::Shape { message: "CI result missing after evaluation" })?;
        Ok(CiBatchResult { results })
    }

    fn min_attainable_p(&self, significance: SignificanceMethod) -> f64 {
        self.inner.min_attainable_p(significance)
    }

    fn series_aligned_weights_len(&self) -> Option<usize> {
        self.inner.series_aligned_weights_len()
    }

    fn p_value_is_frequentist(&self) -> bool {
        self.inner.p_value_is_frequentist()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_stats::PartialCorrelation;

    #[allow(
        clippy::cast_precision_loss,
        reason = "test generator maps a 53-bit integer onto the unit interval"
    )]
    fn series(seed: u64, n: usize) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                (state >> 11) as f64 / (1u64 << 53) as f64
            })
            .collect()
    }

    fn request<'a>(
        columns: &'a [&'a [f64]],
        queries: &'a [CiQuery],
        z: &'a [usize],
    ) -> CiBatchRequest<'a> {
        CiBatchRequest {
            columns,
            queries,
            z_flat: z,
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        }
    }

    #[test]
    fn hit_is_bit_identical_and_not_executed_work() {
        let (a, b, c) = (series(1, 60), series(2, 60), series(3, 60));
        let columns: Vec<&[f64]> = vec![&a, &b, &c];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
        let z = [2usize];
        let store = Arc::new(StatisticStore::new());
        let ci = SelectiveCi::new(Arc::new(PartialCorrelation), Arc::clone(&store), "parcorr");
        let ctx = ExecutionContext::for_tests(7);
        let plan = CiPreparationPlan::default();
        let prepared = ci.prepare(&columns, &plan, &ctx).unwrap();
        let mut ws = CiWorkspace::default();
        let first =
            ci.test_batch(&prepared, &request(&columns, &queries, &z), &mut ws, &ctx).unwrap();
        let direct = PartialCorrelation
            .test_batch(&prepared, &request(&columns, &queries, &z), &mut ws, &ctx)
            .unwrap();
        assert_eq!(first.results[0].p_value.to_bits(), direct.results[0].p_value.to_bits());
        assert_eq!(first.results[0].statistic.to_bits(), direct.results[0].statistic.to_bits());
        let again =
            ci.test_batch(&prepared, &request(&columns, &queries, &z), &mut ws, &ctx).unwrap();
        assert_eq!(again.results[0], first.results[0]);
        let counters = store.counters();
        assert_eq!((counters.attempted, counters.completed, counters.reused), (1, 1, 1));
    }

    #[test]
    fn changed_value_at_unchanged_shape_is_a_different_coordinate() {
        let (a, b, mut c) = (series(1, 60), series(2, 60), series(3, 60));
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
        let z = [2usize];
        let store = Arc::new(StatisticStore::new());
        let ci = SelectiveCi::new(Arc::new(PartialCorrelation), Arc::clone(&store), "parcorr");
        let ctx = ExecutionContext::for_tests(7);
        let mut ws = CiWorkspace::default();
        let plan = CiPreparationPlan::default();
        let columns: Vec<&[f64]> = vec![&a, &b, &c];
        let prepared = ci.prepare(&columns, &plan, &ctx).unwrap();
        let _ = ci.test_batch(&prepared, &request(&columns, &queries, &z), &mut ws, &ctx).unwrap();
        c[10] += 0.25;
        let columns: Vec<&[f64]> = vec![&a, &b, &c];
        let _ = ci.test_batch(&prepared, &request(&columns, &queries, &z), &mut ws, &ctx).unwrap();
        let counters = store.counters();
        assert_eq!((counters.attempted, counters.reused), (2, 0));
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn replay_only_refuses_a_missing_statistic_and_flags_it() {
        let (a, b) = (series(1, 40), series(2, 40));
        let columns: Vec<&[f64]> = vec![&a, &b];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let store = Arc::new(StatisticStore::replay_only(&[]).unwrap());
        let ci = SelectiveCi::new(Arc::new(PartialCorrelation), Arc::clone(&store), "parcorr");
        let ctx = ExecutionContext::for_tests(1);
        let prepared = ci.prepare(&columns, &CiPreparationPlan::default(), &ctx).unwrap();
        let err = ci
            .test_batch(
                &prepared,
                &request(&columns, &queries, &[]),
                &mut CiWorkspace::default(),
                &ctx,
            )
            .unwrap_err();
        assert!(err.to_string().contains("replay_miss"), "{err}");
        assert!(store.replay_missed());
        assert_eq!(store.counters().attempted, 0);
    }

    #[test]
    fn invalid_receipts_are_refused_on_load() {
        let bad = StatisticReceipt {
            coordinate: [1; 32],
            statistic: 0.1,
            p_value: 1.5,
            df: 10.0,
            interval: None,
        };
        assert!(StatisticStore::replay_only(&[bad]).is_err());
        let nan = StatisticReceipt { p_value: f64::NAN, ..bad };
        assert!(StatisticStore::seeded(&[nan]).is_err());
        let good = StatisticReceipt { p_value: 0.5, ..bad };
        assert!(StatisticStore::replay_only(&[good, good]).is_err());
        assert_eq!(StatisticStore::replay_only(&[good]).unwrap().len(), 1);
    }
}
