//! Shared cross-fitted nuisance fits for a batch of queries on one thread.
//!
//! Scoped nuisance reuse for the executing AIPW, DML, and policy estimators.
//! The batch runner opens a thread-local scope; compatible queries borrow its
//! propensity and outcome fits.
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
//! single-query estimators need no new parameters. Learner-backed DML and
//! doubly robust fits use the same cache: it retains their prepared problems under
//! the same retention budget, and their cross-fitted predictions live in the
//! [`LearnerNuisanceStore`] it shares among compatible problems. Bootstrap replicates never
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
pub(crate) const MAX_RETAINED_VALUES: usize = 1 << 25;

/// Most propensity entries, and most outcome entries, one cache keeps. Each lookup scans its
/// entries, so small fits cannot grow that scan without bound under the value budget alone.
pub(crate) const MAX_ENTRIES: usize = 1 << 12;

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
    learned_preparations: Mutex<Vec<crate::propensity::PreparedPropensityProblem>>,
    propensity_fits: AtomicUsize,
    propensity_reuses: AtomicUsize,
    outcome_fits: AtomicUsize,
    outcome_reuses: AtomicUsize,
    completed_fold_fits: AtomicUsize,
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

    /// Completed outer-fold nuisance model fits observed in active cache scopes.
    ///
    /// This counts fitted propensity models and fitted outcome-model pairs whether they used
    /// a cache lease or ran unshared. Cache hits and failed fit attempts are not counted.
    /// Penalized propensity inner-fold model fits are a separate operation and are not included.
    #[must_use]
    pub fn completed_fold_fits(&self) -> usize {
        self.completed_fold_fits.load(Ordering::Acquire)
    }

    fn propensity_entry(&self, key: PropensityKey) -> Option<Arc<PropensityEntry>> {
        let mut entries = self.entries.lock().ok()?;
        if let Some(entry) = entries.propensity.iter().find(|e| e.key.matches(&key)) {
            return Some(Arc::clone(entry));
        }
        let cost = key.retained_values() + key.nrows;
        if entries.propensity.len() >= MAX_ENTRIES
            || entries.retained.saturating_add(cost) > MAX_RETAINED_VALUES
        {
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
        if entries.outcome.len() >= MAX_ENTRIES
            || entries.retained.saturating_add(cost) > MAX_RETAINED_VALUES
        {
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
        if reused {
            antecedent_core::execution_attempt::record_physical(
                antecedent_core::execution_attempt::PhysicalEvent::CacheHit,
                1,
            );
        }
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
        if reused {
            antecedent_core::execution_attempt::record_physical(
                antecedent_core::execution_attempt::PhysicalEvent::CacheHit,
                1,
            );
        }
        let counter = if reused { &self.cache.outcome_reuses } else { &self.cache.outcome_fits };
        counter.fetch_add(1, Ordering::AcqRel);
        note_use(None, Some(self.entry.id), &self.cache);
    }
}

/// Most learner-backed nuisance bundles one prepared problem retains.
const MAX_LEARNER_BUNDLES: usize = 8;

/// Learner-backed (DML / doubly robust) cross-fitted bundles of one prepared problem.
///
/// [`share_learner_preparation`] gives every compatible problem of a cache scope the same
/// store, so the bundles are shared under the cache's retention budget
/// ([`learner_preparation_cost`]) and across threads. Lookup is exact on the entry's key.
#[derive(Default)]
pub(crate) struct LearnerNuisanceStore {
    entries: Mutex<Vec<Arc<crate::learn_nuisance::AipwCacheEntry>>>,
}

impl std::fmt::Debug for LearnerNuisanceStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LearnerNuisanceStore").finish_non_exhaustive()
    }
}

impl LearnerNuisanceStore {
    /// The retained entries, for inspection.
    #[cfg(test)]
    pub(crate) fn lock(
        &self,
    ) -> std::sync::LockResult<MutexGuard<'_, Vec<Arc<crate::learn_nuisance::AipwCacheEntry>>>>
    {
        self.entries.lock()
    }

    /// The entry whose key `matches`, or a new one from `make`.
    ///
    /// Idle entries are evicted to stay within [`MAX_LEARNER_BUNDLES`]; one that a caller
    /// still holds is never evicted, so a further key computes outside the store instead of
    /// exceeding the bound.
    pub(crate) fn entry(
        &self,
        matches: impl Fn(&crate::learn_nuisance::AipwCacheEntry) -> bool,
        make: impl FnOnce() -> crate::learn_nuisance::AipwCacheEntry,
    ) -> Result<Arc<crate::learn_nuisance::AipwCacheEntry>, EstimationError> {
        let mut entries = self.entries.lock().map_err(|_| poisoned())?;
        if let Some(entry) = entries.iter().find(|entry| matches(entry)) {
            return Ok(Arc::clone(entry));
        }
        while entries.len() >= MAX_LEARNER_BUNDLES {
            let Some(index) = entries.iter().position(|entry| Arc::strong_count(entry) == 1) else {
                break;
            };
            entries.remove(index);
        }
        let entry = Arc::new(make());
        if entries.len() < MAX_LEARNER_BUNDLES {
            entries.push(Arc::clone(&entry));
        }
        Ok(entry)
    }
}

/// Reuse exact physical row/design bindings only inside the explicitly active scoped cache.
pub(crate) fn share_learner_preparation(
    problem: crate::propensity::PreparedPropensityProblem,
) -> crate::propensity::PreparedPropensityProblem {
    let cache = ACTIVE.with(|scope| scope.borrow().as_ref().map(|scope| Arc::clone(&scope.cache)));
    let Some(cache) = cache else {
        return problem;
    };
    let Ok(mut entries) = cache.learned_preparations.lock() else {
        return problem;
    };
    if let Some(existing) =
        entries.iter().find(|existing| same_learner_preparation(existing, &problem))
    {
        return existing.clone();
    }
    let cost = learner_preparation_cost(&problem);
    let Ok(mut budget) = cache.entries.lock() else {
        return problem;
    };
    if entries.len() == 8 {
        let old = entries.remove(0);
        budget.retained = budget.retained.saturating_sub(learner_preparation_cost(&old));
    }
    if budget.retained.saturating_add(cost) > MAX_RETAINED_VALUES {
        return problem;
    }
    budget.retained = budget.retained.saturating_add(cost);
    entries.push(problem.clone());
    problem
}
fn same_learner_preparation(
    a: &crate::propensity::PreparedPropensityProblem,
    b: &crate::propensity::PreparedPropensityProblem,
) -> bool {
    let floats = |a: &[f64], b: &[f64]| {
        a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
    };
    a.design_ncols == b.design_ncols
        && a.nrows == b.nrows
        && floats(&a.design_matrix, &b.design_matrix)
        && floats(&a.treatment, &b.treatment)
        && floats(&a.outcome, &b.outcome)
        && a.covariates.len() == b.covariates.len()
        && a.covariates.iter().zip(b.covariates.iter()).all(|(a, b)| floats(a, b))
        && a.method == b.method
        && a.adjustment_set == b.adjustment_set
        && a.overlap == b.overlap
        && a.target_population == b.target_population
        && match (&a.target_weights, &b.target_weights) {
            (None, None) => true,
            (Some(a), Some(b)) => floats(a, b),
            _ => false,
        }
        && a.row_index == b.row_index
        && a.treatment_id == b.treatment_id
        && a.fold_assignment == b.fold_assignment
        && a.fold_units == b.fold_units
        && a.fold_unit_kind == b.fold_unit_kind
        && a.fold_seed == b.fold_seed
        && a.shared_design == b.shared_design
        && a.propensity == b.propensity
        && match (&a.sampling_weights, &b.sampling_weights) {
            (None, None) => true,
            (Some(a), Some(b)) => floats(&a.weights, &b.weights),
            _ => false,
        }
}

fn learner_preparation_cost(problem: &crate::propensity::PreparedPropensityProblem) -> usize {
    // Physical inputs and worst-case retention of eight cached OOF nuisance bundles.
    problem.nrows.saturating_mul(problem.design_ncols.saturating_mul(2).saturating_add(64))
}

#[cfg(test)]
mod learned_preparation_tests {
    use super::*;
    use crate::propensity::{PreparedPropensityProblem, PropensityNuisance};
    use antecedent_core::{TargetPopulation, VariableId};
    fn problem() -> PreparedPropensityProblem {
        PreparedPropensityProblem {
            learner_cache: Arc::default(),
            design_matrix: Arc::from([1., 1., -1., 1.]),
            design_ncols: 2,
            nrows: 2,
            treatment: Arc::from([0., 1.]),
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
    fn exact_shared_inputs_reuse_but_outcomes_and_fold_units_do_not_alias() {
        let cache = CrossfitNuisanceCache::new();
        cache.scope(|| {
            let a = share_learner_preparation(problem());
            let same = share_learner_preparation(problem());
            assert!(Arc::ptr_eq(&a.learner_cache, &same.learner_cache));
            let mut changed = problem();
            changed.outcome = Arc::from([1., 3.]);
            let changed = share_learner_preparation(changed);
            assert!(!Arc::ptr_eq(&a.learner_cache, &changed.learner_cache));
            let mut clustered = problem();
            clustered.fold_units = Some(Arc::from([4, 5]));
            let clustered = share_learner_preparation(clustered);
            assert!(!Arc::ptr_eq(&a.learner_cache, &clustered.learner_cache));
            let mut seeded = problem();
            seeded.fold_seed = 3;
            let seeded = share_learner_preparation(seeded);
            assert!(!Arc::ptr_eq(&a.learner_cache, &seeded.learner_cache));
        });
    }
    #[test]
    fn learner_preparations_share_the_existing_total_retention_budget() {
        let cache = CrossfitNuisanceCache::new();
        cache.entries.lock().unwrap().retained = MAX_RETAINED_VALUES;
        cache.scope(|| {
            let a = share_learner_preparation(problem());
            let b = share_learner_preparation(problem());
            assert!(!Arc::ptr_eq(&a.learner_cache, &b.learner_cache));
        });
        assert!(cache.learned_preparations.lock().unwrap().is_empty());
    }
}
