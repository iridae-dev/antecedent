//! The one checked nuisance-reuse store, shared by the AIPW, DML / doubly robust and
//! observational-policy estimators of a batch.
//!
//! Scoped nuisance reuse: the batch runner opens a scope around each query and every
//! compatible query borrows the earlier query's cross-fitted fit instead of fitting again.
//!
//! Reuse is exact, never approximate. A retained fit is keyed by a [`NuisanceCacheKey`] that
//! the builder derives from the inputs the fit actually reads, hashed value by value:
//!
//! - **Propensity** `e(Z)`: the complete-case `[1 | Z…]` design, the row identities, the
//!   0/1 treatment coding, the realized cross-fit fold assignment (a compact
//!   [`crate::nuisance_cache::FoldIdentity`], not a per-row list), the sampling weights and the
//!   GLM options.
//! - **Outcome** `μ_a(Z)`: everything the propensity key holds plus the transformed outcome
//!   column the regressions are fit to and the threshold target.
//! - **Learner bundle** (DML, doubly robust): the design, treatment and outcome columns, the
//!   weights, the learner specifications, the realized folds and their plan inputs, and the
//!   seed.
//! - **Policy nuisances**: the design, observed actions, outcome, declared support, the realized
//!   splits and the propensity options.
//!
//! A changed input is a miss, and a shared fit reproduces the per-query fit bit for bit because
//! the nuisance fitters are deterministic in exactly these inputs. Clipping is applied after
//! reuse, per query, to the raw out-of-fold propensity.
//!
//! Retention is charged. Each retained entry takes a lease on the caller's shared
//! [`antecedent_core::execution::AdmissionPool`] (see [`CrossfitNuisanceCache::charged_to`])
//! for its key and its predictions, and returns it when the entry is evicted or dropped. When
//! the pool or the local bound has no room, the least recently used idle entries are evicted;
//! an entry that a running fit still holds is never evicted. When nothing can be evicted the
//! fit runs unshared and the refusal is counted in [`CrossfitCacheStats`]. No cache holds
//! predictions the pool did not admit.
//!
//! A cache is used only inside [`CrossfitNuisanceCache::scope`], on the calling thread: the
//! batch runner opens one scope per query, so the single-query estimators need no new
//! parameters. A worker thread does not inherit the scope; it takes the cache explicitly with
//! [`CrossfitNuisanceCache::active`] and [`CrossfitNuisanceCache::run_in`]. Bootstrap
//! replicates never use the cache.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use antecedent_core::ExecutionContext;
use antecedent_core::execution::{AdmissionError, AdmissionLease};
use antecedent_core::identity::ReusableProduct;

use crate::error::EstimationError;
use crate::nuisance_cache::{NuisanceCacheKey, NuisanceKeyBuilder, TargetDependency};

/// Most bytes one cache retains across its keys and fitted values (256 MiB). Beyond it, idle
/// entries are evicted and, failing that, a fit runs unshared.
pub(crate) const MAX_RETAINED_BYTES: u64 = 1 << 28;

/// Most entries one shared cache keeps. Lookup is hashed, but eviction scans the entries, so
/// small fits cannot grow that scan without bound under the byte budget alone.
pub(crate) const MAX_ENTRIES: usize = 1 << 12;

/// Most entries a problem-local store (no active cache scope) retains.
pub(crate) const MAX_LOCAL_ENTRIES: usize = 8;

/// Out-of-fold propensity scores (raw, unclipped), one per complete-case row.
pub(crate) type PropensitySlot = Option<Arc<[f64]>>;
/// Out-of-fold `(μ_0, μ_1)` predictions, one pair per complete-case row.
pub(crate) type OutcomeSlot = Option<(Arc<[f64]>, Arc<[f64]>)>;
type ValueSlot = Option<Arc<dyn Any + Send + Sync>>;

pub(crate) struct PropensityEntry {
    id: u64,
    slot: Mutex<PropensitySlot>,
    scopes: Arc<AtomicUsize>,
}

pub(crate) struct OutcomeEntry {
    id: u64,
    slot: Mutex<OutcomeSlot>,
    scopes: Arc<AtomicUsize>,
}

/// A typed value (a learner bundle, the policy nuisances) behind one key.
pub(crate) struct ValueEntry {
    slot: Mutex<ValueSlot>,
}

#[derive(Clone)]
enum Handle {
    Propensity(Arc<PropensityEntry>),
    Outcome(Arc<OutcomeEntry>),
    Value(Arc<ValueEntry>),
}

impl Handle {
    /// Nobody but the store holds the entry, so evicting it frees memory no fit is using.
    fn idle(&self) -> bool {
        match self {
            Self::Propensity(entry) => Arc::strong_count(entry) == 1,
            Self::Outcome(entry) => Arc::strong_count(entry) == 1,
            Self::Value(entry) => Arc::strong_count(entry) == 1,
        }
    }
}

struct Slot {
    bytes: u64,
    last_used: u64,
    /// Returned to the shared pool when the slot is evicted or the store drops.
    _lease: Option<AdmissionLease>,
    handle: Handle,
}

struct StoreState {
    next_id: u64,
    tick: u64,
    bytes: u64,
    max_entries: usize,
    max_bytes: u64,
    slots: HashMap<NuisanceCacheKey, Slot>,
}

impl StoreState {
    /// Evict the least recently used idle entry. `false` when every entry is in use.
    fn evict_one(&mut self) -> bool {
        let Some(victim) = self
            .slots
            .iter()
            .filter(|(_, slot)| slot.handle.idle())
            .min_by_key(|(_, slot)| slot.last_used)
            .map(|(key, _)| key.clone())
        else {
            return false;
        };
        if let Some(slot) = self.slots.remove(&victim) {
            self.bytes = self.bytes.saturating_sub(slot.bytes);
        }
        true
    }
}

/// Keys, predictions and provider state retained for reuse, with eviction and charging.
pub(crate) struct NuisanceStore {
    state: Mutex<StoreState>,
    /// The caller's shared pool every retained entry is charged to. A problem-local store
    /// takes the pool of the execution context that asks for the entry instead.
    pool: Option<ExecutionContext>,
    evictions: AtomicUsize,
    refusals: AtomicUsize,
    value_fits: AtomicUsize,
    value_reuses: AtomicUsize,
}

// The mutable state sits behind a mutex that recovers from poisoning, and the pool handle is
// only read, so a panic observed through a shared reference cannot expose a broken invariant.
impl std::panic::RefUnwindSafe for NuisanceStore {}

impl Default for NuisanceStore {
    fn default() -> Self {
        Self::bounded(None, MAX_LOCAL_ENTRIES, MAX_RETAINED_BYTES)
    }
}

impl std::fmt::Debug for NuisanceStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NuisanceStore")
            .field("entries", &self.len())
            .field("bytes", &self.retained_bytes())
            .finish_non_exhaustive()
    }
}

impl NuisanceStore {
    pub(crate) fn bounded(
        pool: Option<ExecutionContext>,
        max_entries: usize,
        max_bytes: u64,
    ) -> Self {
        Self {
            state: Mutex::new(StoreState {
                next_id: 0,
                tick: 0,
                bytes: 0,
                max_entries,
                max_bytes,
                slots: HashMap::new(),
            }),
            pool,
            evictions: AtomicUsize::new(0),
            refusals: AtomicUsize::new(0),
            value_fits: AtomicUsize::new(0),
            value_reuses: AtomicUsize::new(0),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.state.lock().map_or(0, |state| state.slots.len())
    }

    pub(crate) fn retained_bytes(&self) -> u64 {
        self.state.lock().map_or(0, |state| state.bytes)
    }

    /// The retained entry for `key`, or a new one charged `payload_bytes` plus the key.
    ///
    /// `None` when the entry cannot be retained: it exceeds the local bound, the pool has no
    /// room after every idle entry was evicted, or the study is cancelled. The caller then
    /// runs unshared.
    fn entry(
        &self,
        key: NuisanceCacheKey,
        payload_bytes: u64,
        charge: Option<&ExecutionContext>,
        make: impl FnOnce(u64) -> Handle,
    ) -> Option<Handle> {
        let mut state = self.state.lock().ok()?;
        state.tick += 1;
        let tick = state.tick;
        if let Some(slot) = state.slots.get_mut(&key) {
            slot.last_used = tick;
            return Some(slot.handle.clone());
        }
        let cost = key.retained_bytes().saturating_add(payload_bytes);
        if cost > state.max_bytes || state.max_entries == 0 {
            self.refusals.fetch_add(1, Ordering::AcqRel);
            return None;
        }
        while state.slots.len() >= state.max_entries
            || state.bytes.saturating_add(cost) > state.max_bytes
        {
            if !state.evict_one() {
                self.refusals.fetch_add(1, Ordering::AcqRel);
                return None;
            }
            self.evictions.fetch_add(1, Ordering::AcqRel);
        }
        let lease = match self.pool.as_ref().or(charge) {
            None => None,
            Some(ctx) => loop {
                match ctx.acquire_admission(0, cost) {
                    Ok(lease) => break Some(lease),
                    Err(AdmissionError::ExceedsRemainder { .. }) => {
                        if !state.evict_one() {
                            self.refusals.fetch_add(1, Ordering::AcqRel);
                            return None;
                        }
                        self.evictions.fetch_add(1, Ordering::AcqRel);
                    }
                    Err(_) => {
                        self.refusals.fetch_add(1, Ordering::AcqRel);
                        return None;
                    }
                }
            },
        };
        let id = state.next_id;
        state.next_id += 1;
        state.bytes = state.bytes.saturating_add(cost);
        let handle = make(id);
        state.slots.insert(
            key,
            Slot { bytes: cost, last_used: tick, _lease: lease, handle: handle.clone() },
        );
        Some(handle)
    }

    fn propensity_entry(
        &self,
        key: NuisanceCacheKey,
        payload_bytes: u64,
        charge: Option<&ExecutionContext>,
    ) -> Option<Arc<PropensityEntry>> {
        let made = self.entry(key, payload_bytes, charge, |id| {
            Handle::Propensity(Arc::new(PropensityEntry {
                id,
                slot: Mutex::new(None),
                scopes: Arc::new(AtomicUsize::new(0)),
            }))
        })?;
        match made {
            Handle::Propensity(entry) => Some(entry),
            _ => None,
        }
    }

    fn outcome_entry(
        &self,
        key: NuisanceCacheKey,
        payload_bytes: u64,
        charge: Option<&ExecutionContext>,
    ) -> Option<Arc<OutcomeEntry>> {
        let made = self.entry(key, payload_bytes, charge, |id| {
            Handle::Outcome(Arc::new(OutcomeEntry {
                id,
                slot: Mutex::new(None),
                scopes: Arc::new(AtomicUsize::new(0)),
            }))
        })?;
        match made {
            Handle::Outcome(entry) => Some(entry),
            _ => None,
        }
    }

    /// The value computed under `key`, computing it once.
    ///
    /// Callers for one key wait on a single fit and then read it; a failed or cancelled fit
    /// leaves the entry empty and retryable. When the entry cannot be retained the value is
    /// computed for this caller alone. `wait` is polled while another caller holds the fit, so
    /// a cancelled caller stops waiting. The flag is `true` for a reused value.
    pub(crate) fn get_or_compute<T, E>(
        &self,
        key: NuisanceCacheKey,
        payload_bytes: u64,
        charge: Option<&ExecutionContext>,
        wait: &dyn Fn() -> Result<(), E>,
        compute: impl FnOnce() -> Result<T, E>,
    ) -> Result<(Arc<T>, bool), E>
    where
        T: Any + Send + Sync,
        E: From<EstimationError>,
    {
        let retained = self.entry(key, payload_bytes, charge, |_| {
            Handle::Value(Arc::new(ValueEntry { slot: Mutex::new(None) }))
        });
        let entry = match retained {
            Some(Handle::Value(entry)) => entry,
            _ => Arc::new(ValueEntry { slot: Mutex::new(None) }),
        };
        let mut slot = loop {
            wait()?;
            match entry.slot.try_lock() {
                Ok(slot) => break slot,
                Err(std::sync::TryLockError::WouldBlock) => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(std::sync::TryLockError::Poisoned(_)) => return Err(E::from(poisoned())),
            }
        };
        wait()?;
        if let Some(existing) = slot.as_ref() {
            if let Ok(value) = Arc::clone(existing).downcast::<T>() {
                self.value_reuses.fetch_add(1, Ordering::AcqRel);
                return Ok((value, true));
            }
        }
        let value = Arc::new(compute()?);
        wait()?;
        *slot = Some(Arc::clone(&value) as Arc<dyn Any + Send + Sync>);
        self.value_fits.fetch_add(1, Ordering::AcqRel);
        Ok((value, false))
    }

    /// Test hook: change the local bounds.
    #[cfg(test)]
    pub(crate) fn set_limits(&self, max_entries: usize, max_bytes: u64) {
        if let Ok(mut state) = self.state.lock() {
            state.max_entries = max_entries;
            state.max_bytes = max_bytes;
        }
    }

    /// Test hook: hold every retained entry as concurrent callers would during a fit.
    #[cfg(test)]
    pub(crate) fn hold_all(&self) -> Vec<Arc<ValueEntry>> {
        let state = self.state.lock().unwrap();
        state
            .slots
            .values()
            .filter_map(|slot| match &slot.handle {
                Handle::Value(entry) => Some(Arc::clone(entry)),
                _ => None,
            })
            .collect()
    }

    /// Test hook: drop every idle entry.
    #[cfg(test)]
    pub(crate) fn clear(&self) {
        let mut state = self.state.lock().unwrap();
        while state.evict_one() {}
    }
}

/// The retained learner-backed (DML / doubly robust) bundles of one prepared problem, or of
/// every compatible problem of a cache scope. The same store type as the cache's own.
pub(crate) type LearnerNuisanceStore = NuisanceStore;

/// Fit counters for one cache (see [`CrossfitNuisanceCache::stats`]).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CrossfitCacheStats {
    /// Cross-fitted propensity models actually fit (one per fold plan).
    pub propensity_fits: usize,
    /// Propensity requests answered by an earlier fit.
    pub propensity_reuses: usize,
    /// Cross-fitted outcome-regression pairs actually fit.
    pub outcome_fits: usize,
    /// Outcome requests answered by an earlier fit.
    pub outcome_reuses: usize,
    /// Learner-backed and policy nuisance bundles actually fit.
    pub bundle_fits: usize,
    /// Learner-backed and policy nuisance bundle requests answered by an earlier fit.
    pub bundle_reuses: usize,
    /// Entries evicted to make room under the local bound or the shared pool.
    pub evictions: usize,
    /// Entries that could not be retained (nothing idle to evict): the fit ran unshared.
    pub refused_retentions: usize,
}

#[derive(Clone, Debug)]
struct UsedEntry {
    id: u64,
    scopes: Arc<AtomicUsize>,
}

impl PartialEq for UsedEntry {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for UsedEntry {}

/// Which shared entries one [`CrossfitNuisanceCache::scope`] read.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NuisanceScopeUse {
    propensity: Vec<UsedEntry>,
    outcome: Vec<UsedEntry>,
}

/// Exact-key cache of cross-fitted nuisances shared by one batch.
pub struct CrossfitNuisanceCache {
    store: Arc<NuisanceStore>,
    propensity_fits: AtomicUsize,
    propensity_reuses: AtomicUsize,
    outcome_fits: AtomicUsize,
    outcome_reuses: AtomicUsize,
    completed_fold_fits: AtomicUsize,
}

impl Default for CrossfitNuisanceCache {
    fn default() -> Self {
        Self::build(None, MAX_ENTRIES, MAX_RETAINED_BYTES)
    }
}

impl std::fmt::Debug for CrossfitNuisanceCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossfitNuisanceCache").field("stats", &self.stats()).finish()
    }
}

struct ActiveScope {
    cache: Arc<CrossfitNuisanceCache>,
    used: NuisanceScopeUse,
}

thread_local! {
    static ACTIVE: RefCell<Option<ActiveScope>> = const { RefCell::new(None) };
}

/// Restores the enclosing scope (if any) even when the scoped work panics.
struct ScopeGuard {
    previous: Option<ActiveScope>,
    restored: bool,
}

impl ScopeGuard {
    fn finish(mut self) -> NuisanceScopeUse {
        self.restored = true;
        let previous = self.previous.take();
        ACTIVE.with(|slot| {
            let mut slot = slot.borrow_mut();
            let mine = std::mem::replace(&mut *slot, previous);
            mine.map(|scope| scope.used).unwrap_or_default()
        })
    }
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        if !self.restored {
            let previous = self.previous.take();
            ACTIVE.with(|slot| *slot.borrow_mut() = previous);
        }
    }
}

impl CrossfitNuisanceCache {
    fn build(pool: Option<ExecutionContext>, max_entries: usize, max_bytes: u64) -> Self {
        Self {
            store: Arc::new(NuisanceStore::bounded(pool, max_entries, max_bytes)),
            propensity_fits: AtomicUsize::new(0),
            propensity_reuses: AtomicUsize::new(0),
            outcome_fits: AtomicUsize::new(0),
            outcome_reuses: AtomicUsize::new(0),
            completed_fold_fits: AtomicUsize::new(0),
        }
    }

    /// An empty cache bounded only by its own defaults. Share it (behind an [`Arc`]) across the
    /// queries of one batch. Prefer [`Self::charged_to`] when a caller owns an admission pool.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// An empty cache whose retained keys, predictions and provider state are charged to the
    /// admission pool of `ctx`, the caller's shared account.
    ///
    /// Each retained entry holds a lease for its bytes until it is evicted or the cache drops.
    /// A request the pool cannot hold evicts the least recently used idle entries; if none
    /// is idle the fit runs unshared and [`CrossfitCacheStats::refused_retentions`] counts it.
    #[must_use]
    pub fn charged_to(ctx: &ExecutionContext) -> Arc<Self> {
        Arc::new(Self::build(Some(ctx.clone()), MAX_ENTRIES, MAX_RETAINED_BYTES))
    }

    /// [`Self::charged_to`] with explicit local bounds, for a caller that wants a tighter cap
    /// than the pool's. `ctx` may be `None` for a cache bounded locally only.
    #[must_use]
    pub fn bounded(
        ctx: Option<&ExecutionContext>,
        max_entries: usize,
        max_bytes: u64,
    ) -> Arc<Self> {
        Arc::new(Self::build(ctx.cloned(), max_entries, max_bytes))
    }

    /// Bytes this cache currently retains (keys and predictions), as charged.
    #[must_use]
    pub fn retained_bytes(&self) -> u64 {
        self.store.retained_bytes()
    }

    /// Entries this cache currently retains.
    #[must_use]
    pub fn retained_entries(&self) -> usize {
        self.store.len()
    }

    /// The cache whose scope is open on the calling thread, if any.
    ///
    /// A scope is thread-local. Hand this to a worker thread and open it there with
    /// [`Self::run_in`], or the worker fits unshared.
    #[must_use]
    pub fn active() -> Option<Arc<Self>> {
        active_cache()
    }

    /// Run `work` in `cache`'s scope when one is given, or plainly when `None`.
    pub fn run_in<R>(cache: Option<&Arc<Self>>, work: impl FnOnce() -> R) -> R {
        match cache {
            Some(cache) => cache.scope(work).0,
            None => work(),
        }
    }

    /// Run `work` with this cache active on the current thread and report which
    /// shared entries it read. Nested scopes restore the enclosing one on exit.
    pub fn scope<R>(self: &Arc<Self>, work: impl FnOnce() -> R) -> (R, NuisanceScopeUse) {
        let previous = ACTIVE.with(|slot| {
            slot.borrow_mut()
                .replace(ActiveScope { cache: Arc::clone(self), used: NuisanceScopeUse::default() })
        });
        let guard = ScopeGuard { previous, restored: false };
        let out = work();
        (out, guard.finish())
    }

    /// Run `work` under the cache already active on this thread, or under a
    /// fresh cache that lives exactly as long as `work` when none is.
    ///
    /// A caller that fits one table more than once (a click and the score
    /// table it freezes) uses this to share the nuisances between its own
    /// fits without hiding them from an enclosing batch scope: joining the
    /// active cache keeps that batch's counters and sharing intact, where a
    /// nested [`Self::scope`] would replace it for the duration of `work`.
    pub fn within_scope<R>(work: impl FnOnce() -> R) -> R {
        if active_cache().is_some() {
            return work();
        }
        Self::new().scope(work).0
    }

    /// `(propensity, outcome)`: whether a scope read a propensity / outcome fit that
    /// at least one other scope of this cache also read.
    #[must_use]
    #[allow(
        clippy::unused_self,
        reason = "the answer lives in the entries' own counters; the receiver keeps the API tied to the cache that issued the scope use"
    )]
    pub fn shared_with_another_scope(&self, used: &NuisanceScopeUse) -> (bool, bool) {
        let shared = |entries: &[UsedEntry]| {
            entries.iter().any(|entry| entry.scopes.load(Ordering::Acquire) > 1)
        };
        (shared(&used.propensity), shared(&used.outcome))
    }

    /// Fit and reuse counters.
    #[must_use]
    pub fn stats(&self) -> CrossfitCacheStats {
        CrossfitCacheStats {
            propensity_fits: self.propensity_fits.load(Ordering::Acquire),
            propensity_reuses: self.propensity_reuses.load(Ordering::Acquire),
            outcome_fits: self.outcome_fits.load(Ordering::Acquire),
            outcome_reuses: self.outcome_reuses.load(Ordering::Acquire),
            bundle_fits: self.store.value_fits.load(Ordering::Acquire),
            bundle_reuses: self.store.value_reuses.load(Ordering::Acquire),
            evictions: self.store.evictions.load(Ordering::Acquire),
            refused_retentions: self.store.refusals.load(Ordering::Acquire),
        }
    }

    /// Completed outer-fold nuisance model fits observed in active cache scopes.
    ///
    /// This counts fitted propensity models and fitted outcome-model pairs whether they used
    /// a cache lease or ran unshared. Cache hits and failed fit attempts are not counted.
    /// Penalized propensity inner-fold model fits are a separate operation and are not included.
    #[must_use]
    pub fn completed_fold_fits(&self) -> usize {
        self.completed_fold_fits.load(Ordering::Acquire)
    }
}

/// Record that the active scope read an entry (once per scope).
fn note_use(
    propensity: Option<(u64, &Arc<AtomicUsize>)>,
    outcome: Option<(u64, &Arc<AtomicUsize>)>,
) {
    ACTIVE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(scope) = slot.as_mut() else {
            return;
        };
        if let Some((id, scopes)) = propensity {
            if !scope.used.propensity.iter().any(|used| used.id == id) {
                scopes.fetch_add(1, Ordering::AcqRel);
                scope.used.propensity.push(UsedEntry { id, scopes: Arc::clone(scopes) });
            }
        }
        if let Some((id, scopes)) = outcome {
            if !scope.used.outcome.iter().any(|used| used.id == id) {
                scopes.fetch_add(1, Ordering::AcqRel);
                scope.used.outcome.push(UsedEntry { id, scopes: Arc::clone(scopes) });
            }
        }
    });
}

fn active_cache() -> Option<Arc<CrossfitNuisanceCache>> {
    ACTIVE.with(|slot| slot.borrow().as_ref().map(|scope| Arc::clone(&scope.cache)))
}

/// Record one completed outer-fold nuisance model fit in the active scope, if any.
pub(crate) fn record_completed_fold_fit() {
    let Some(cache) = active_cache() else {
        return;
    };
    cache.completed_fold_fits.fetch_add(1, Ordering::AcqRel);
}

fn poisoned() -> EstimationError {
    EstimationError::stats_msg("shared nuisance cache lock poisoned")
}

/// A propensity slot held for the duration of one score-table build.
///
/// While held, another query with the same key waits, then reads the finished fit
/// instead of fitting it again. Dropping the lease without [`Self::record`] and a stored
/// fit (an error) leaves the slot empty and retryable.
pub(crate) struct PropensityLease {
    cache: Arc<CrossfitNuisanceCache>,
    entry: Arc<PropensityEntry>,
    /// The inputs shared by the propensity and outcome keys; an outcome key adds its own.
    base: NuisanceKeyBuilder,
}

impl PropensityLease {
    /// Lease the active scope's propensity slot for the key `base` derives, if a scope is
    /// active, the key builds, and the cache has room. `base` carries every input the
    /// propensity and the outcome regressions share and is not run outside a scope.
    pub(crate) fn acquire(
        base: impl FnOnce() -> Option<NuisanceKeyBuilder>,
        nrows: usize,
    ) -> Option<Self> {
        let cache = active_cache()?;
        let base = base()?;
        let key = base.clone().with_product(ReusableProduct::PropensityFit).build().ok()?;
        let entry = cache.store.propensity_entry(key, 8 * nrows as u64, None)?;
        Some(Self { cache, entry, base })
    }

    pub(crate) fn lock(&self) -> Result<MutexGuard<'_, PropensitySlot>, EstimationError> {
        self.entry.slot.lock().map_err(|_| poisoned())
    }

    /// Count a fit or a reuse and mark the scope as a reader of this entry.
    pub(crate) fn record(&self, reused: bool) {
        if reused {
            antecedent_core::execution_attempt::record_physical(
                antecedent_core::execution_attempt::PhysicalEvent::CacheHit,
                1,
            );
        }
        let counter =
            if reused { &self.cache.propensity_reuses } else { &self.cache.propensity_fits };
        counter.fetch_add(1, Ordering::AcqRel);
        note_use(Some((self.entry.id, &self.entry.scopes)), None);
    }

    /// Lease the outcome slot for the regressions fit to `y` at `target`, on this
    /// propensity's inputs.
    pub(crate) fn outcome(&self, target: TargetDependency, y: &[f64]) -> Option<OutcomeLease> {
        let key = self
            .base
            .clone()
            .with_product(ReusableProduct::OutcomeFit)
            .target(target)
            .outcome(y)
            .build()
            .ok()?;
        let entry = self.cache.store.outcome_entry(key, 16 * y.len() as u64, None)?;
        Some(OutcomeLease { cache: Arc::clone(&self.cache), entry })
    }
}

/// An outcome-regression slot; see [`PropensityLease`].
pub(crate) struct OutcomeLease {
    cache: Arc<CrossfitNuisanceCache>,
    entry: Arc<OutcomeEntry>,
}

impl OutcomeLease {
    pub(crate) fn lock(&self) -> Result<MutexGuard<'_, OutcomeSlot>, EstimationError> {
        self.entry.slot.lock().map_err(|_| poisoned())
    }

    pub(crate) fn record(&self, reused: bool) {
        if reused {
            antecedent_core::execution_attempt::record_physical(
                antecedent_core::execution_attempt::PhysicalEvent::CacheHit,
                1,
            );
        }
        let counter = if reused { &self.cache.outcome_reuses } else { &self.cache.outcome_fits };
        counter.fetch_add(1, Ordering::AcqRel);
        note_use(None, Some((self.entry.id, &self.entry.scopes)));
    }
}

/// Give a freshly prepared learner problem the store of the active cache scope, so every
/// compatible problem of the scope reuses one set of bundles under the cache's budget.
/// Outside a scope the problem keeps its own small store.
pub(crate) fn share_learner_preparation(
    mut problem: crate::propensity::PreparedPropensityProblem,
) -> crate::propensity::PreparedPropensityProblem {
    if let Some(cache) = active_cache() {
        problem.learner_cache = Arc::clone(&cache.store);
    }
    problem
}

/// The value `compute` produces under the key `key` derives, reusing it from the cache active
/// on this thread. With no active scope, or when the key does not build, it is computed for
/// this caller alone.
///
/// # Errors
///
/// What `compute` or `wait` return. A failed fit leaves the entry empty.
pub(crate) fn reuse_in_active_scope<T, E>(
    key: impl FnOnce() -> Option<NuisanceCacheKey>,
    payload_bytes: u64,
    wait: &dyn Fn() -> Result<(), E>,
    compute: impl FnOnce() -> Result<T, E>,
) -> Result<Arc<T>, E>
where
    T: Any + Send + Sync,
    E: From<EstimationError>,
{
    let Some(cache) = active_cache() else {
        return compute().map(Arc::new);
    };
    let Some(key) = key() else {
        return compute().map(Arc::new);
    };
    let (value, reused) = cache.store.get_or_compute(key, payload_bytes, None, wait, compute)?;
    if reused {
        antecedent_core::execution_attempt::record_physical(
            antecedent_core::execution_attempt::PhysicalEvent::CacheHit,
            1,
        );
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nuisance_cache::{
        FoldIdentity, NuisanceOptions, NumericalDependencies, ProviderDependency,
    };
    use crate::propensity::{PreparedPropensityProblem, PropensityNuisance};
    use antecedent_core::execution::AdmissionPool;
    use antecedent_core::{NumericalPolicy, TargetPopulation, VariableId};

    const PAYLOAD: u64 = 1_000;

    fn key(tag: f64) -> NuisanceCacheKey {
        NuisanceKeyBuilder::new(ReusableProduct::PropensityFit)
            .design_matrix(2, 1, &[tag, 1.0])
            .rows_positional(2)
            .folds(FoldIdentity::of_fold_ids(2, &[0, 1]).unwrap())
            .treatment_values("t", &[&[0.0, 1.0]])
            .weights(None)
            .provider(ProviderDependency::named("test", "1"))
            .options(&NuisanceOptions::new("test"))
            .numerical(NumericalDependencies::try_new(NumericalPolicy::strict(), &[]).unwrap())
            .build()
            .unwrap()
    }

    fn ok() -> Result<(), EstimationError> {
        Ok(())
    }

    fn fetch(cache: &CrossfitNuisanceCache, tag: f64) -> (u64, bool) {
        let (value, reused) = cache
            .store
            .get_or_compute(key(tag), PAYLOAD, None, &ok, || Ok(tag as u64 * 10))
            .unwrap();
        (*value, reused)
    }

    fn pool_of(entries: u64) -> (AdmissionPool, ExecutionContext, u64, u64) {
        let cost = key(0.0).retained_bytes() + PAYLOAD;
        let budget = entries * cost + cost / 2;
        let pool = AdmissionPool::fixed(1, budget);
        let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
        (pool, ctx, cost, budget)
    }

    #[test]
    fn retained_entries_are_charged_to_the_shared_pool_and_returned_on_eviction_and_drop() {
        let (pool, ctx, cost, budget) = pool_of(2);
        let cache = CrossfitNuisanceCache::charged_to(&ctx);
        assert_eq!(fetch(&cache, 1.0), (10, false));
        assert_eq!(fetch(&cache, 2.0), (20, false));
        assert_eq!(cache.retained_bytes(), 2 * cost);
        assert_eq!(pool.available().bytes(), budget - 2 * cost);
        // Touch the first so the second is the least recently used.
        assert_eq!(fetch(&cache, 1.0), (10, true));
        assert_eq!(fetch(&cache, 3.0), (30, false));
        let stats = cache.stats();
        assert_eq!((stats.evictions, stats.refused_retentions), (1, 0));
        assert_eq!(cache.retained_entries(), 2);
        assert_eq!(pool.available().bytes(), budget - 2 * cost, "the evicted lease came back");
        assert_eq!(fetch(&cache, 1.0), (10, true), "the touched entry survived");
        assert_eq!(fetch(&cache, 2.0), (20, false), "the evicted entry is refit");
        assert_eq!((stats.bundle_fits, stats.bundle_reuses), (3, 1));
        drop(cache);
        assert_eq!(pool.available().bytes(), budget, "a dropped cache holds nothing");
    }

    #[test]
    fn an_entry_a_fit_still_holds_is_never_evicted_and_a_full_pool_refuses_retention() {
        let (pool, ctx, cost, budget) = pool_of(2);
        let cache = CrossfitNuisanceCache::charged_to(&ctx);
        fetch(&cache, 1.0);
        fetch(&cache, 2.0);
        let held = cache.store.hold_all();
        assert_eq!(held.len(), 2);
        for _ in 0..2 {
            assert_eq!(fetch(&cache, 3.0), (30, false), "refused retention still answers");
        }
        let stats = cache.stats();
        assert_eq!((stats.evictions, stats.refused_retentions), (0, 2));
        assert_eq!(cache.retained_bytes(), 2 * cost);
        assert_eq!(pool.available().bytes(), budget - 2 * cost, "nothing was over-admitted");
        drop(held);
        assert_eq!(fetch(&cache, 3.0), (30, false));
        assert_eq!(cache.stats().evictions, 1, "idle again, so one is evicted for the new key");
        assert_eq!(fetch(&cache, 3.0), (30, true));
    }

    #[test]
    fn a_pool_smaller_than_one_entry_retains_nothing_and_a_cancelled_study_retains_nothing() {
        let tiny = AdmissionPool::fixed(1, 8);
        let ctx = ExecutionContext::for_tests(1).with_admission(tiny.clone());
        let cache = CrossfitNuisanceCache::charged_to(&ctx);
        assert_eq!(fetch(&cache, 1.0), (10, false));
        assert_eq!(fetch(&cache, 1.0), (10, false));
        assert_eq!(cache.retained_entries(), 0);
        assert_eq!(cache.stats().refused_retentions, 2);
        assert_eq!(tiny.available().bytes(), 8);

        let (pool, ctx, _, budget) = pool_of(2);
        ctx.cancellation.cancel();
        let cancelled = CrossfitNuisanceCache::charged_to(&ctx);
        assert_eq!(fetch(&cancelled, 1.0), (10, false));
        assert_eq!(cancelled.retained_entries(), 0);
        assert_eq!(pool.available().bytes(), budget);
    }

    #[test]
    fn concurrent_callers_never_charge_more_than_the_pool_admits() {
        let (pool, ctx, cost, budget) = pool_of(2);
        let cache = CrossfitNuisanceCache::charged_to(&ctx);
        std::thread::scope(|scope| {
            for worker in 0..6u64 {
                let (cache, pool) = (&cache, &pool);
                scope.spawn(move || {
                    for round in 0..25u64 {
                        let tag = ((worker + round) % 5) as f64;
                        let (value, _) = fetch(cache, tag);
                        assert_eq!(value, tag as u64 * 10, "every caller reads its own value");
                        let charged = budget - pool.available().bytes();
                        assert!(charged <= 2 * cost, "{charged} bytes charged for two entries");
                    }
                });
            }
        });
        assert!(cache.retained_entries() <= 2);
        assert!(cache.retained_bytes() <= 2 * cost);
        let stats = cache.stats();
        assert!(stats.evictions + stats.refused_retentions > 0, "the budget was binding");
        drop(cache);
        assert_eq!(pool.available().bytes(), budget);
    }

    #[test]
    fn local_bounds_cap_entries_without_a_pool() {
        let cache = CrossfitNuisanceCache::bounded(None, 2, MAX_RETAINED_BYTES);
        fetch(&cache, 1.0);
        fetch(&cache, 2.0);
        fetch(&cache, 3.0);
        assert_eq!(cache.retained_entries(), 2);
        assert_eq!(cache.stats().evictions, 1);
        let none = CrossfitNuisanceCache::bounded(None, 0, MAX_RETAINED_BYTES);
        assert_eq!(fetch(&none, 1.0), (10, false));
        assert_eq!(fetch(&none, 1.0), (10, false));
        assert_eq!(none.stats().refused_retentions, 2);
    }

    #[test]
    fn a_cancelled_waiter_stops_waiting_and_a_failed_fit_stays_retryable() {
        let cache = CrossfitNuisanceCache::new();
        let failed = cache.store.get_or_compute::<u64, EstimationError>(
            key(1.0),
            PAYLOAD,
            None,
            &ok,
            || Err(EstimationError::stats_msg("fit failed")),
        );
        assert!(failed.is_err());
        assert_eq!(fetch(&cache, 1.0), (10, false), "the failed fit left the entry empty");
        assert_eq!(fetch(&cache, 1.0), (10, true));
        let stopped = cache.store.get_or_compute::<u64, EstimationError>(
            key(2.0),
            PAYLOAD,
            None,
            &|| Err(EstimationError::stats_msg("cancelled")),
            || Ok(0),
        );
        assert!(stopped.is_err());
    }

    #[test]
    fn a_scope_is_per_thread_and_a_worker_takes_the_cache_explicitly() {
        let cache = CrossfitNuisanceCache::new();
        assert!(CrossfitNuisanceCache::active().is_none());
        cache.scope(|| {
            let active = CrossfitNuisanceCache::active().expect("scope is open here");
            assert!(Arc::ptr_eq(&active, &cache));
            std::thread::scope(|workers| {
                let implicit = workers.spawn(|| CrossfitNuisanceCache::active().is_none());
                assert!(
                    implicit.join().unwrap(),
                    "a worker does not inherit the thread-local scope"
                );
                let explicit = workers.spawn(|| {
                    CrossfitNuisanceCache::run_in(Some(&active), || {
                        CrossfitNuisanceCache::active()
                            .is_some_and(|inner| Arc::ptr_eq(&inner, &active))
                    })
                });
                assert!(explicit.join().unwrap());
            });
        });
        assert!(CrossfitNuisanceCache::active().is_none());
    }

    fn problem() -> PreparedPropensityProblem {
        PreparedPropensityProblem {
            learner_cache: Arc::default(),
            design_matrix: Arc::from([1., 1., -1., 1.]),
            design_ncols: 2,
            nrows: 2,
            treatment: Arc::from([0., 1.]),
            arms: Arc::from([crate::arm::TreatmentArm::Control, crate::arm::TreatmentArm::Treated]),
            treatment_levels: crate::propensity::DeclaredTreatmentLevels {
                active: 1.0,
                control: 0.0,
                other_levels: crate::propensity::OtherLevels::Refuse,
                rows_at_active: 1,
                rows_at_control: 1,
                excluded_rows: 0,
            },
            outcome: Arc::from([1., 2.]),
            covariates: Arc::from([Arc::from([-1., 1.])]),
            method: Arc::from("backdoor"),
            adjustment_set: Arc::from([VariableId::from_raw(2)]),
            overlap: crate::OverlapPolicy::require_diagnostics(),
            target_population: TargetPopulation::AllObserved,
            target_weights: None,
            row_index: Arc::from([
                antecedent_data::GlobalId::new(0),
                antecedent_data::GlobalId::new(1),
            ]),
            treatment_id: VariableId::from_raw(0),
            fold_assignment: None,
            fold_units: None,
            fold_unit_kind: crate::cluster_dml_aipw::IndependenceUnit::Cluster,
            fold_seed: 0,
            shared_design: false,
            propensity: PropensityNuisance::default(),
            sampling_weights: None,
            preparation: antecedent_data::materialize::PreparationReceipt::composed_finite(2)
                .unwrap(),
        }
    }

    #[test]
    fn a_problem_prepared_in_a_scope_shares_the_scopes_store_and_one_outside_keeps_its_own() {
        let cache = CrossfitNuisanceCache::new();
        let (inside, outside) = {
            let (inside, _) = cache.scope(|| share_learner_preparation(problem()));
            (inside, share_learner_preparation(problem()))
        };
        assert!(Arc::ptr_eq(&inside.learner_cache, &cache.store));
        assert!(!Arc::ptr_eq(&outside.learner_cache, &cache.store));
        let (other, _) = cache.scope(|| share_learner_preparation(problem()));
        assert!(Arc::ptr_eq(&inside.learner_cache, &other.learner_cache));
    }
}
