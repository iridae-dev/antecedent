//! Shared cross-fitted AIPW nuisance fits for a batch of queries.
//!
//! A batch of average-effect queries on one table often fits the same nuisance
//! more than once: every query with the same treatment and adjustment set fits
//! the same propensity model, and a repeated `(treatment, outcome)` pair fits
//! the same outcome regressions. [`CrossfitNuisanceCache`] lets those queries
//! reuse one out-of-fold fit.
//!
//! Reuse is exact, never approximate. An entry is reused only when every input
//! the fit reads is bit-identical:
//!
//! - **Propensity** `e(Z)`: the complete-case `[1 | Z…]` design (so the
//!   adjustment set, its column order and the row set), the 0/1 treatment
//!   coding, the cross-fit fold ids and fold count (so the fold seed and the
//!   arm stratification), and the GLM options.
//! - **Outcome** `μ_a(Z)`: everything the propensity key holds (the outcome
//!   regressions train on the same design, treatment and folds) plus the
//!   transformed outcome column the regressions are fit to (the outcome, or
//!   its `1{Y > c}` indicator for a threshold).
//!
//! Different data, rows, learners, seeds or folds therefore never share a
//! fit, and a shared fit reproduces the per-query fit bit for bit, because
//! the nuisance fitters are deterministic in exactly these inputs. Clipping is
//! applied after reuse, per query, to the raw out-of-fold propensity.
//!
//! A cache is used only inside [`CrossfitNuisanceCache::scope`], on the
//! calling thread: the batch runner opens one scope per query, so the
//! single-query estimators need no new parameters. Bootstrap replicates never
//! use the cache. Retained inputs are bounded by [`MAX_RETAINED_VALUES`]; past
//! the bound a fit simply runs unshared.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use antecedent_stats::GlmOptions;

use crate::error::EstimationError;

/// Most `f64`/`u32` values one cache retains across its keys and fitted values
/// (about 256 MiB). Beyond it, new fits run unshared.
pub const MAX_RETAINED_VALUES: usize = 1 << 25;

/// Out-of-fold propensity scores (raw, unclipped), one per complete-case row.
pub(crate) type PropensitySlot = Option<Arc<[f64]>>;
/// Out-of-fold `(μ_0, μ_1)` predictions, one pair per complete-case row.
pub(crate) type OutcomeSlot = Option<(Arc<[f64]>, Arc<[f64]>)>;

pub(crate) struct PropensityKey {
    pub(crate) design: Arc<[f64]>,
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
    pub(crate) treatment: Arc<[f64]>,
    pub(crate) fold_ids: Arc<[u32]>,
    pub(crate) folds: usize,
    pub(crate) glm_options: GlmOptions,
}

impl PropensityKey {
    fn matches(&self, other: &Self) -> bool {
        self.nrows == other.nrows
            && self.ncols == other.ncols
            && self.folds == other.folds
            && self.glm_options == other.glm_options
            && self.fold_ids == other.fold_ids
            && same_bits(&self.treatment, &other.treatment)
            && same_bits(&self.design, &other.design)
    }

    fn retained_values(&self) -> usize {
        self.design.len() + self.treatment.len() + self.fold_ids.len()
    }
}

fn same_bits(a: &Arc<[f64]>, b: &Arc<[f64]>) -> bool {
    Arc::ptr_eq(a, b)
        || (a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits()))
}

pub(crate) struct PropensityEntry {
    id: usize,
    key: PropensityKey,
    slot: Mutex<PropensitySlot>,
    scopes: AtomicUsize,
}

pub(crate) struct OutcomeEntry {
    id: usize,
    propensity: usize,
    outcome: Arc<[f64]>,
    slot: Mutex<OutcomeSlot>,
    scopes: AtomicUsize,
}

#[derive(Default)]
struct Entries {
    propensity: Vec<Arc<PropensityEntry>>,
    outcome: Vec<Arc<OutcomeEntry>>,
    retained: usize,
}

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
}

/// Which shared entries one [`CrossfitNuisanceCache::scope`] read.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NuisanceScopeUse {
    propensity: Vec<usize>,
    outcome: Vec<usize>,
}

/// Exact-key cache of cross-fitted AIPW nuisances shared by one batch.
#[derive(Default)]
pub struct CrossfitNuisanceCache {
    entries: Mutex<Entries>,
    propensity_fits: AtomicUsize,
    propensity_reuses: AtomicUsize,
    outcome_fits: AtomicUsize,
    outcome_reuses: AtomicUsize,
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
    /// An empty cache. Share it (behind an [`Arc`]) across the queries of one batch.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
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

    /// `(propensity, outcome)`: whether a scope read a propensity / outcome fit that
    /// at least one other scope of this cache also read.
    #[must_use]
    pub fn shared_with_another_scope(&self, used: &NuisanceScopeUse) -> (bool, bool) {
        let Ok(entries) = self.entries.lock() else {
            return (false, false);
        };
        let propensity = used.propensity.iter().any(|id| {
            entries.propensity.get(*id).is_some_and(|e| e.scopes.load(Ordering::Acquire) > 1)
        });
        let outcome = used.outcome.iter().any(|id| {
            entries.outcome.get(*id).is_some_and(|e| e.scopes.load(Ordering::Acquire) > 1)
        });
        (propensity, outcome)
    }

    /// Fit and reuse counters.
    #[must_use]
    pub fn stats(&self) -> CrossfitCacheStats {
        CrossfitCacheStats {
            propensity_fits: self.propensity_fits.load(Ordering::Acquire),
            propensity_reuses: self.propensity_reuses.load(Ordering::Acquire),
            outcome_fits: self.outcome_fits.load(Ordering::Acquire),
            outcome_reuses: self.outcome_reuses.load(Ordering::Acquire),
        }
    }

    fn propensity_entry(&self, key: PropensityKey) -> Option<Arc<PropensityEntry>> {
        let mut entries = self.entries.lock().ok()?;
        if let Some(entry) = entries.propensity.iter().find(|e| e.key.matches(&key)) {
            return Some(Arc::clone(entry));
        }
        let cost = key.retained_values() + key.nrows;
        if entries.retained.saturating_add(cost) > MAX_RETAINED_VALUES {
            return None;
        }
        entries.retained += cost;
        let entry = Arc::new(PropensityEntry {
            id: entries.propensity.len(),
            key,
            slot: Mutex::new(None),
            scopes: AtomicUsize::new(0),
        });
        entries.propensity.push(Arc::clone(&entry));
        Some(entry)
    }

    fn outcome_entry(&self, propensity: usize, outcome: Arc<[f64]>) -> Option<Arc<OutcomeEntry>> {
        let mut entries = self.entries.lock().ok()?;
        if let Some(entry) = entries
            .outcome
            .iter()
            .find(|e| e.propensity == propensity && same_bits(&e.outcome, &outcome))
        {
            return Some(Arc::clone(entry));
        }
        let cost = 3 * outcome.len();
        if entries.retained.saturating_add(cost) > MAX_RETAINED_VALUES {
            return None;
        }
        entries.retained += cost;
        let entry = Arc::new(OutcomeEntry {
            id: entries.outcome.len(),
            propensity,
            outcome,
            slot: Mutex::new(None),
            scopes: AtomicUsize::new(0),
        });
        entries.outcome.push(Arc::clone(&entry));
        Some(entry)
    }
}

/// Record that the active scope read an entry (once per scope).
fn note_use(propensity: Option<usize>, outcome: Option<usize>, cache: &CrossfitNuisanceCache) {
    ACTIVE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(scope) = slot.as_mut() else {
            return;
        };
        let Ok(entries) = cache.entries.lock() else {
            return;
        };
        if let Some(id) = propensity.filter(|id| !scope.used.propensity.contains(id)) {
            scope.used.propensity.push(id);
            if let Some(entry) = entries.propensity.get(id) {
                entry.scopes.fetch_add(1, Ordering::AcqRel);
            }
        }
        if let Some(id) = outcome.filter(|id| !scope.used.outcome.contains(id)) {
            scope.used.outcome.push(id);
            if let Some(entry) = entries.outcome.get(id) {
                entry.scopes.fetch_add(1, Ordering::AcqRel);
            }
        }
    });
}

fn active_cache() -> Option<Arc<CrossfitNuisanceCache>> {
    ACTIVE.with(|slot| slot.borrow().as_ref().map(|scope| Arc::clone(&scope.cache)))
}

fn poisoned() -> EstimationError {
    EstimationError::stats_msg("shared nuisance cache lock poisoned")
}

/// A propensity slot held for the duration of one score-table build.
///
/// While held, another query with the same key waits, then reads the finished fit
/// instead of fitting it again. Dropping the lease without [`Self::store`] (an error)
/// leaves the slot empty and retryable.
pub(crate) struct PropensityLease {
    cache: Arc<CrossfitNuisanceCache>,
    entry: Arc<PropensityEntry>,
}

impl PropensityLease {
    /// Lease the active scope's propensity slot for `key`, if a scope is active and
    /// the cache has room.
    pub(crate) fn acquire(key: impl FnOnce() -> PropensityKey) -> Option<Self> {
        let cache = active_cache()?;
        let entry = cache.propensity_entry(key())?;
        Some(Self { cache, entry })
    }

    pub(crate) fn lock(&self) -> Result<MutexGuard<'_, PropensitySlot>, EstimationError> {
        self.entry.slot.lock().map_err(|_| poisoned())
    }

    /// Count a fit or a reuse and mark the scope as a reader of this entry.
    pub(crate) fn record(&self, reused: bool) {
        let counter =
            if reused { &self.cache.propensity_reuses } else { &self.cache.propensity_fits };
        counter.fetch_add(1, Ordering::AcqRel);
        note_use(Some(self.entry.id), None, &self.cache);
    }

    /// Lease the outcome slot for `outcome` fit on this propensity key.
    pub(crate) fn outcome(&self, outcome: impl FnOnce() -> Arc<[f64]>) -> Option<OutcomeLease> {
        let entry = self.cache.outcome_entry(self.entry.id, outcome())?;
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
        let counter = if reused { &self.cache.outcome_reuses } else { &self.cache.outcome_fits };
        counter.fetch_add(1, Ordering::AcqRel);
        note_use(None, Some(self.entry.id), &self.cache);
    }
}
