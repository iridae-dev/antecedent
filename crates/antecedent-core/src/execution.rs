//! Execution context: parallelism, determinism, RNG, budgets, kernel policy.
//!
//! No core algorithm creates a global thread pool, uses an implicit global RNG,
//! or selects architecture-specific behavior outside [`KernelPolicy`].
//!
//! CPU and RAM admission is a shared lease. Taking a lease decrements the
//! budget. [`MemoryBudget`] is a planner limit and is never decremented.
//! [`ExecutionContext::map_indexed`] takes one [`AdmissionPool`] lease for its
//! workers and returns that lease on success, failure, and cancellation. A
//! child lease is drawn from its parent's remainder, release returns that
//! remainder. Cancellation stops new work; live owners retain their reservations
//! until release or drop. Byte products and sums that
//! do not fit in `u64` are refused.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::fmt;
use std::num::NonZeroU32;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{Duration, Instant};

use thiserror::Error;

/// Parallel execution budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Parallelism {
    /// Maximum worker threads (1 = serial).
    pub max_threads: NonZeroThreadCount,
}

/// Thread count that is at least one.
///
/// This is [`NonZeroU32`]. A separate newtype would reimplement the same zero check.
pub type NonZeroThreadCount = NonZeroU32;

/// Cap on the default user thread count. An explicit `threads=` may exceed it.
pub const DEFAULT_USER_THREAD_CAP: u32 = 16;

/// Default worker count for the product path: `available_parallelism`,
/// clamped to [`DEFAULT_USER_THREAD_CAP`]. `threads=1` remains an explicit pin.
#[must_use]
pub fn default_user_threads() -> u32 {
    std::thread::available_parallelism()
        .map(|n| u32::try_from(n.get()).unwrap_or(DEFAULT_USER_THREAD_CAP))
        .unwrap_or(1)
        .clamp(1, DEFAULT_USER_THREAD_CAP)
}

impl Parallelism {
    /// Serial execution.
    #[must_use]
    pub const fn serial() -> Self {
        Self { max_threads: NonZeroThreadCount::MIN }
    }

    /// Bounded parallelism.
    #[must_use]
    pub const fn bounded(max_threads: NonZeroThreadCount) -> Self {
        Self { max_threads }
    }
}

/// Determinism requirements for reductions and scheduling.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Determinism {
    /// Prefer fastest path; reductions may be nondeterministic.
    PreferFast,
    /// Require bitwise-reproducible results for a fixed seed and thread count.
    Strict,
}

/// Memory budget for planned allocations.
///
/// These fields are planner limits. No path decrements them. Actual byte
/// reservations use [`AdmissionPool`], installed with [`ExecutionContext::with_admission`].
/// A parallel [`ExecutionContext::map_indexed`] leases worker slots; its callers
/// reserve their scratch bytes separately, so nested work can use the same account.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct MemoryBudget {
    /// Soft limit in bytes; planners should refuse or stream above this.
    pub soft_limit_bytes: Option<u64>,
    /// Hard limit in bytes; exceeding is a resource error.
    pub hard_limit_bytes: Option<u64>,
}

impl MemoryBudget {
    /// Unlimited budget (still subject to OS limits).
    #[must_use]
    pub const fn unlimited() -> Self {
        Self { soft_limit_bytes: None, hard_limit_bytes: None }
    }
}

/// Whether an architecture-SIMD kernel path is compiled into this build.
///
/// Always `false` until a justified `simd-runtime` kernel lands. Kernel dispatch and
/// execution identity both read it, so a request for SIMD that cannot be honored is
/// neither run nor recorded as if it were.
pub const ARCH_SIMD_COMPILED: bool = false;

/// Kernel selection policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct KernelPolicy {
    /// Allow portable optimized kernels.
    pub allow_portable_optimized: bool,
    /// Allow architecture-specific SIMD after feature detection.
    pub allow_arch_simd: bool,
    /// Force scalar reference path (for tests / debugging).
    pub force_scalar: bool,
}

impl KernelPolicy {
    /// Default: optimized allowed, SIMD allowed, scalar not forced.
    #[must_use]
    pub const fn default_policy() -> Self {
        Self { allow_portable_optimized: true, allow_arch_simd: true, force_scalar: false }
    }

    /// Force scalar kernels only.
    #[must_use]
    pub const fn scalar_only() -> Self {
        Self { allow_portable_optimized: false, allow_arch_simd: false, force_scalar: true }
    }

    /// Whether SIMD is both allowed and available in this build.
    #[must_use]
    pub const fn arch_simd_effective(&self) -> bool {
        self.allow_arch_simd && ARCH_SIMD_COMPILED && !self.force_scalar
    }
}

/// Cache usage policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CachePolicy {
    /// Whether semantic caches may be used.
    pub enabled: bool,
    /// Maximum cache bytes, if bounded.
    pub max_bytes: Option<u64>,
}

impl CachePolicy {
    /// Caching disabled.
    #[must_use]
    pub const fn disabled() -> Self {
        Self { enabled: false, max_bytes: None }
    }

    /// Caching enabled with optional byte cap.
    #[must_use]
    pub const fn enabled(max_bytes: Option<u64>) -> Self {
        Self { enabled: true, max_bytes }
    }
}

/// Bounded cache budget for incremental causal state.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CacheBudget {
    /// Maximum retained cache bytes.
    pub max_bytes: u64,
    /// Bytes currently retained (updated by the state crate).
    pub used_bytes: u64,
}

impl CacheBudget {
    /// Fresh budget with `max_bytes` capacity and zero usage.
    #[must_use]
    pub const fn new(max_bytes: u64) -> Self {
        Self { max_bytes, used_bytes: 0 }
    }

    /// Unlimited soft budget (still subject to OS limits).
    #[must_use]
    pub const fn unlimited() -> Self {
        Self { max_bytes: u64::MAX, used_bytes: 0 }
    }

    /// Remaining capacity in bytes.
    #[must_use]
    pub const fn remaining(self) -> u64 {
        self.max_bytes.saturating_sub(self.used_bytes)
    }

    /// Whether `additional` bytes would fit under the budget.
    #[must_use]
    pub const fn can_admit(self, additional: u64) -> bool {
        self.used_bytes.saturating_add(additional) <= self.max_bytes
    }
}

/// Shared Monte Carlo / approximate-compute budget report.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MonteCarloBudget {
    /// Scalar / batch evaluations performed.
    pub evaluations: u64,
    /// Monte Carlo samples drawn.
    pub samples: u64,
    /// Exact enumerations performed, if any.
    pub exact_enumerations: u64,
}

/// Per-estimate Monte Carlo uncertainty summary.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MonteCarloError {
    /// Estimated standard error of the reported score.
    pub stderr: f64,
    /// Samples contributing to this estimate.
    pub samples: u64,
}

/// Opt-in bootstrap early-stop budget bounded by Monte Carlo error.
///
/// For approximately normal replicates, a bootstrap SE computed from `B` of them
/// carries a relative Monte Carlo standard error of about `1/√(2(B − 1))` (the SD
/// of a sample SD); for a replicate distribution with kurtosis `κ` it is
/// `√((κ − 1)/(4B))`, so heavy-tailed replicates need more than the floor below
/// to reach the same precision. The only honest stopping rule is a replicate
/// floor: after
/// [`Self::required_replicates`] successful replicates — the larger of
/// [`Self::min_replicates`] and `⌈1 + 1/(2·se_rel_epsilon²)⌉`, the count at
/// which that relative error is at most [`Self::se_rel_epsilon`] — the loop
/// stops and reports `early_stopped = true` with the actual count. The cap
/// remains the requested replicate count, so a request below the floor is
/// evaluated in full.
///
/// Earlier releases stopped on a one-step relative change of the running SE
/// (`|SE_t − SE_{t−1}| / SE_{t−1} < ε`), which is not an error bound and
/// stopped static bootstraps at 10–22 replicates whatever was requested. That
/// rule is gone. Neither [`ExecutionContext::production`] nor
/// [`ExecutionContext::for_tests`] enables a budget: both evaluate the full
/// request, which is what the calibration gates certify.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveBootstrapBudget {
    /// When false, evaluate all requested replicates (no early-stop).
    pub enabled: bool,
    /// Minimum successful replicates before early-stop is eligible.
    pub min_replicates: u32,
    /// Target relative Monte Carlo standard error of the bootstrap SE
    /// (default `0.05`, met at 201 successful replicates). Zero or negative
    /// never stops.
    pub se_rel_epsilon: f64,
}

impl AdaptiveBootstrapBudget {
    /// Enabled defaults: stop once the relative Monte Carlo SE of the
    /// bootstrap SE is at most 5% (201 successful replicates).
    #[must_use]
    pub const fn enabled_default() -> Self {
        Self { enabled: true, min_replicates: 2, se_rel_epsilon: 0.05 }
    }

    /// Force full requested replicate count (tests / exact-N pins).
    #[must_use]
    pub const fn disabled() -> Self {
        Self { enabled: false, min_replicates: 0, se_rel_epsilon: 0.0 }
    }

    /// Successful replicates after which an enabled budget may stop:
    /// `max(min_replicates, ⌈1 + 1/(2·se_rel_epsilon²)⌉)`, from the relative
    /// Monte Carlo SE `1/√(2(B − 1))` of a bootstrap SE. `u32::MAX` when the
    /// budget is disabled or `se_rel_epsilon` is not positive.
    ///
    /// # Examples
    ///
    /// ```
    /// use antecedent_core::AdaptiveBootstrapBudget;
    ///
    /// assert_eq!(AdaptiveBootstrapBudget::enabled_default().required_replicates(), 201);
    /// assert_eq!(AdaptiveBootstrapBudget::disabled().required_replicates(), u32::MAX);
    /// ```
    #[must_use]
    pub fn required_replicates(&self) -> u32 {
        // NaN must read as "never stop", so test the positive branch directly.
        let positive = self.se_rel_epsilon > 0.0;
        if !self.enabled || !positive {
            return u32::MAX;
        }
        let bound = (1.0 + 0.5 / (self.se_rel_epsilon * self.se_rel_epsilon)).ceil();
        // Saturate rather than truncate: a tiny ε is "never stop", not "stop at 0".
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "the value is a non-negative ceil and the branch saturates at u32::MAX before the cast, so it can neither truncate nor lose sign"
        )]
        let bound = if bound >= f64::from(u32::MAX) { u32::MAX } else { bound as u32 };
        bound.max(self.min_replicates).max(2)
    }
}

impl Default for AdaptiveBootstrapBudget {
    /// Disabled, like every constructor: early stopping is opt-in, so
    /// `..Default::default()` never changes how many replicates run.
    fn default() -> Self {
        Self::disabled()
    }
}

/// Opt-in Bayesian draw budget for the Laplace Gaussian-redraw path.
///
/// When enabled, Laplace MVN redraws stop once the effective sample size of
/// the effect draws reaches [`Self::ess_target`] (independent draws, so ESS is
/// the draw count) after at least [`Self::min_draws`]; the requested
/// `n_draws` remains the cap. ESS is a genuine Monte Carlo error measure; the
/// earlier stop on a one-step relative change of the 95% quantile width was
/// not and is no longer consulted. Exact conjugate (NIG) sampling and HMC
/// always materialize the full request. Neither
/// [`ExecutionContext::production`] nor [`ExecutionContext::for_tests`]
/// enables this budget.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveDrawBudget {
    /// When false, materialize the full requested draw count.
    pub enabled: bool,
    /// Minimum draws before early-stop is eligible.
    pub min_draws: usize,
    /// Retained for API stability; no longer consulted (quantile-width
    /// convergence is not a Monte Carlo error bound).
    pub quantile_width_rel_epsilon: f64,
    /// Stop when ESS of the effect draws reaches this target (default `10_000`).
    pub ess_target: f64,
}

impl AdaptiveDrawBudget {
    /// Enabled defaults: min 32 draws, stop at ESS 10 000.
    #[must_use]
    pub const fn enabled_default() -> Self {
        Self { enabled: true, min_draws: 32, quantile_width_rel_epsilon: 0.0, ess_target: 10_000.0 }
    }

    /// Force full requested draw count.
    #[must_use]
    pub const fn disabled() -> Self {
        Self { enabled: false, min_draws: 0, quantile_width_rel_epsilon: 0.0, ess_target: 0.0 }
    }
}

impl Default for AdaptiveDrawBudget {
    /// Disabled, like every constructor: early stopping is opt-in, so
    /// `..Default::default()` never changes how many draws are materialized.
    fn default() -> Self {
        Self::disabled()
    }
}

/// CPU slots and RAM bytes requested from an admission budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct AdmissionRequest {
    threads: u32,
    bytes: u64,
}

impl AdmissionRequest {
    /// Request `threads` CPU slots and `bytes` of RAM.
    #[must_use]
    pub const fn new(threads: u32, bytes: u64) -> Self {
        Self { threads, bytes }
    }

    /// CPU slots.
    #[must_use]
    pub const fn threads(self) -> u32 {
        self.threads
    }

    /// RAM bytes.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    /// Sum `parts` into one byte request.
    ///
    /// # Errors
    ///
    /// [`AdmissionError::Overflow`] when the sum does not fit in `u64`. The sum is not wrapped.
    pub const fn try_sum(threads: u32, parts: &[u64]) -> Result<Self, AdmissionError> {
        let mut bytes = 0u64;
        let mut index = 0;
        while index < parts.len() {
            let Some(sum) = checked_byte_sum(bytes, parts[index]) else {
                return Err(AdmissionError::Overflow);
            };
            bytes = sum;
            index += 1;
        }
        Ok(Self::new(threads, bytes))
    }

    /// Request `count * width` bytes.
    ///
    /// # Errors
    ///
    /// [`AdmissionError::Overflow`] when the product does not fit in `u64`.
    pub const fn try_product(threads: u32, count: u64, width: u64) -> Result<Self, AdmissionError> {
        let Some(bytes) = checked_byte_product(count, width) else {
            return Err(AdmissionError::Overflow);
        };
        Ok(Self::new(threads, bytes))
    }
}

/// CPU slots and RAM bytes still held by one lease, or still free in a pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct AdmissionRemainder {
    threads: u32,
    bytes: u64,
}

impl AdmissionRemainder {
    /// Remainder of `threads` slots and `bytes` of RAM.
    #[must_use]
    pub const fn new(threads: u32, bytes: u64) -> Self {
        Self { threads, bytes }
    }

    /// Nothing held.
    #[must_use]
    pub const fn zero() -> Self {
        Self::new(0, 0)
    }

    /// CPU slots.
    #[must_use]
    pub const fn threads(self) -> u32 {
        self.threads
    }

    /// RAM bytes.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.bytes
    }
}

/// Why an admission lease was not granted, or a size was not accepted.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Error)]
#[non_exhaustive]
pub enum AdmissionError {
    /// The request is larger than the pool or parent remainder.
    #[error(
        "admission request of {requested_threads} threads and {requested_bytes} bytes exceeds \
         the remainder of {remaining_threads} threads and {remaining_bytes} bytes"
    )]
    ExceedsRemainder {
        /// CPU slots asked for.
        requested_threads: u32,
        /// RAM bytes asked for.
        requested_bytes: u64,
        /// CPU slots still available to this account.
        remaining_threads: u32,
        /// RAM bytes still available to this account.
        remaining_bytes: u64,
    },
    /// A sum or product did not fit. It was not wrapped.
    #[error("admission arithmetic overflowed and was refused")]
    Overflow,
    /// The wait ended and the budget still could not hold the request.
    #[error("admission wait of {bound:?} ended before the budget had room")]
    WaitBoundExceeded {
        /// Bound that elapsed.
        bound: Duration,
    },
    /// The study was cancelled before this admission completed.
    #[error("admission stopped because the study was cancelled")]
    Cancelled,
    /// The parent lease no longer holds a remainder.
    #[error("admission lease is already released")]
    Released,
}

/// `count * width`, or `None` when the product does not fit in `u64`.
#[must_use]
pub const fn checked_byte_product(count: u64, width: u64) -> Option<u64> {
    count.checked_mul(width)
}

/// `left + right`, or `None` when the sum does not fit in `u64`.
#[must_use]
pub const fn checked_byte_sum(left: u64, right: u64) -> Option<u64> {
    left.checked_add(right)
}

/// Shared CPU and RAM budget.
///
/// [`Self::available`] is what no open lease holds. Granting a lease decrements it. Two contexts
/// that install the same pool compete for that one remainder. [`MemoryBudget`] is not decremented.
/// [`ExecutionContext::map_indexed`] consults this pool for its one worker lease.
#[derive(Clone)]
pub struct AdmissionPool {
    shared: Arc<SharedAdmission>,
}

struct SharedAdmission {
    state: Mutex<AdmissionState>,
    released: Arc<Condvar>,
}

/// Free counters plus every open lease remainder equal the budget. That equality is the
/// enforcement: a descriptor that never changes these counters admits nothing.
struct AdmissionState {
    thread_budget: u32,
    byte_budget: u64,
    threads_free: u32,
    bytes_free: u64,
    leases: Vec<LeaseSlot>,
    next_id: u64,
    blocked_waiters: usize,
}

struct LeaseSlot {
    id: u64,
    parent: Option<u64>,
    threads: u32,
    bytes: u64,
    observation: Option<crate::execution_attempt::ObservedAdmission>,
    /// Set when a test asks the cancellation scan to return this slot.
    /// A live owner is not marked by [`CancellationToken::cancel`].
    #[allow(dead_code)]
    cancelled: bool,
}

enum GrantError {
    Short(AdmissionError),
    Overflow,
}

impl AdmissionPool {
    /// Budget of `threads` CPU slots and `bytes` of RAM, all of it free.
    #[must_use]
    pub fn fixed(threads: u32, bytes: u64) -> Self {
        Self {
            shared: Arc::new(SharedAdmission {
                state: Mutex::new(AdmissionState {
                    thread_budget: threads,
                    byte_budget: bytes,
                    threads_free: threads,
                    bytes_free: bytes,
                    leases: Vec::new(),
                    next_id: 1,
                    blocked_waiters: 0,
                }),
                released: Arc::new(Condvar::new()),
            }),
        }
    }

    /// CPU and RAM no open lease holds.
    ///
    /// Cancellation does not free reservations still held by workers or retained buffers.
    /// Their owners release them after cleanup, so another job cannot overcommit live memory.
    #[must_use]
    pub fn available(&self) -> AdmissionRemainder {
        let guard = lock_mutex(&self.shared.state);
        AdmissionRemainder::new(guard.threads_free, guard.bytes_free)
    }

    /// CPU slots this pool was created with.
    #[must_use]
    pub fn thread_budget(&self) -> u32 {
        lock_mutex(&self.shared.state).thread_budget
    }

    /// RAM bytes this pool was created with.
    #[must_use]
    pub fn byte_budget(&self) -> u64 {
        lock_mutex(&self.shared.state).byte_budget
    }

    /// Studies blocked inside a bounded acquire.
    #[must_use]
    pub fn blocked_waiters(&self) -> usize {
        lock_mutex(&self.shared.state).blocked_waiters
    }
}

impl fmt::Debug for AdmissionPool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let guard = lock_mutex(&self.shared.state);
        formatter
            .debug_struct("AdmissionPool")
            .field("thread_budget", &guard.thread_budget)
            .field("byte_budget", &guard.byte_budget)
            .field("threads_free", &guard.threads_free)
            .field("bytes_free", &guard.bytes_free)
            .field("open_leases", &guard.leases.len())
            .field("blocked_waiters", &guard.blocked_waiters)
            .finish()
    }
}

fn lock_mutex<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn lease_index(leases: &[LeaseSlot], id: u64) -> Option<usize> {
    leases.iter().position(|slot| slot.id == id)
}

fn exceeds(
    request: AdmissionRequest,
    remaining_threads: u32,
    remaining_bytes: u64,
) -> AdmissionError {
    AdmissionError::ExceedsRemainder {
        requested_threads: request.threads,
        requested_bytes: request.bytes,
        remaining_threads,
        remaining_bytes,
    }
}

fn reserve(
    available_threads: u32,
    available_bytes: u64,
    request: AdmissionRequest,
) -> Result<(u32, u64), AdmissionError> {
    let Some(threads) = available_threads.checked_sub(request.threads) else {
        return Err(exceeds(request, available_threads, available_bytes));
    };
    let Some(bytes) = available_bytes.checked_sub(request.bytes) else {
        return Err(exceeds(request, available_threads, available_bytes));
    };
    Ok((threads, bytes))
}

fn grant_root(state: &mut AdmissionState, request: AdmissionRequest) -> Result<u64, GrantError> {
    let Some(next_id) = state.next_id.checked_add(1) else {
        return Err(GrantError::Overflow);
    };
    let (threads_free, bytes_free) = match reserve(state.threads_free, state.bytes_free, request) {
        Ok(reserved) => reserved,
        Err(error) => return Err(GrantError::Short(error)),
    };
    state.threads_free = threads_free;
    state.bytes_free = bytes_free;
    Ok(push_lease(state, None, request, next_id))
}

fn grant_child(
    state: &mut AdmissionState,
    parent_id: u64,
    request: AdmissionRequest,
) -> Result<u64, AdmissionError> {
    let Some(next_id) = state.next_id.checked_add(1) else {
        return Err(AdmissionError::Overflow);
    };
    let Some(index) = lease_index(&state.leases, parent_id) else {
        return Err(AdmissionError::Released);
    };
    let remaining_threads = state.leases[index].threads;
    let remaining_bytes = state.leases[index].bytes;
    let (threads, bytes) = reserve(remaining_threads, remaining_bytes, request)?;
    state.leases[index].threads = threads;
    state.leases[index].bytes = bytes;
    Ok(push_lease(state, Some(parent_id), request, next_id))
}

fn push_lease(
    state: &mut AdmissionState,
    parent: Option<u64>,
    request: AdmissionRequest,
    next_id: u64,
) -> u64 {
    let id = state.next_id;
    state.next_id = next_id;
    let observation = match parent.and_then(|id| lease_index(&state.leases, id)) {
        Some(index) => state.leases[index]
            .observation
            .as_mut()
            .map(|held| held.split(request.threads, request.bytes)),
        None => {
            Some(crate::execution_attempt::ObservedAdmission::new(request.threads, request.bytes))
        }
    };
    state.leases.push(LeaseSlot {
        id,
        parent,
        threads: request.threads,
        bytes: request.bytes,
        observation,
        cancelled: false,
    });
    id
}

fn credit(
    state: &mut AdmissionState,
    parent: Option<u64>,
    threads: u32,
    bytes: u64,
) -> Result<(), AdmissionError> {
    let parent_index = parent.and_then(|id| lease_index(&state.leases, id));
    if let Some(index) = parent_index {
        let Some(held_threads) = state.leases[index].threads.checked_add(threads) else {
            return Err(AdmissionError::Overflow);
        };
        let Some(held_bytes) = state.leases[index].bytes.checked_add(bytes) else {
            return Err(AdmissionError::Overflow);
        };
        state.leases[index].threads = held_threads;
        state.leases[index].bytes = held_bytes;
        return Ok(());
    }
    let Some(threads_free) = state.threads_free.checked_add(threads) else {
        return Err(AdmissionError::Overflow);
    };
    if threads_free > state.thread_budget {
        return Err(AdmissionError::Overflow);
    }
    let Some(bytes_free) = state.bytes_free.checked_add(bytes) else {
        return Err(AdmissionError::Overflow);
    };
    if bytes_free > state.byte_budget {
        return Err(AdmissionError::Overflow);
    }
    state.threads_free = threads_free;
    state.bytes_free = bytes_free;
    Ok(())
}

fn close_slot(state: &mut AdmissionState, index: usize) -> AdmissionRemainder {
    let id = state.leases[index].id;
    let parent = state.leases[index].parent;
    let threads = state.leases[index].threads;
    let bytes = state.leases[index].bytes;
    if let Err(error) = credit(state, parent, threads, bytes) {
        panic!("admission credit overflow on lease {id}: {error}; the lease stays in the table");
    }
    for slot in &mut state.leases {
        if slot.parent == Some(id) {
            slot.parent = parent;
        }
    }
    let mut removed = state.leases.remove(index);
    if let Some(parent_index) = parent.and_then(|id| lease_index(&state.leases, id)) {
        if let (Some(held), Some(returned)) =
            (state.leases[parent_index].observation.as_mut(), removed.observation.as_mut())
        {
            held.merge(returned);
        }
    }
    AdmissionRemainder::new(threads, bytes)
}

/// Return every slot marked `cancelled`, visiting each slot once.
///
/// The index is not reset after a removal. Resetting it would rescan slots that
/// were already kept and could miss a slot that slid into the removed index
/// only on a later pass.
#[cfg(test)]
fn reap_cancelled(state: &mut AdmissionState) -> (usize, usize) {
    let mut index = 0;
    let mut removed = 0;
    let mut visits = 0;
    while index < state.leases.len() {
        visits += 1;
        if state.leases[index].cancelled {
            close_slot(state, index);
            removed += 1;
        } else {
            index += 1;
        }
    }
    (removed, visits)
}

fn close_and_notify(shared: &SharedAdmission, id: u64) -> AdmissionRemainder {
    let mut guard = lock_mutex(&shared.state);
    let Some(index) = lease_index(&guard.leases, id) else {
        return AdmissionRemainder::zero();
    };
    let remainder = close_slot(&mut guard, index);
    shared.released.notify_all();
    remainder
}

/// A hold on CPU slots and RAM bytes.
///
/// Dropping or explicitly releasing the lease returns its remainder. Cancellation
/// refuses new child work and wakes waiters, but keeps live CPU and memory charged
/// until the owner finishes cleanup. Retained buffers keep their leases until drop.
#[must_use = "dropping the lease releases its CPU and RAM"]
pub struct AdmissionLease {
    shared: Arc<SharedAdmission>,
    id: u64,
    flag: Arc<AtomicBool>,
    active: AtomicBool,
}

impl AdmissionLease {
    /// Whether this reservation is charged to the specified shared admission account.
    #[must_use]
    pub fn belongs_to(&self, pool: &AdmissionPool) -> bool {
        Arc::ptr_eq(&self.shared, &pool.shared)
    }

    fn new(shared: Arc<SharedAdmission>, id: u64, flag: Arc<AtomicBool>) -> Self {
        Self { shared, id, flag, active: AtomicBool::new(true) }
    }

    /// Include an existing reservation in the current native execution observation.
    /// This neither grants resources nor changes cancellation or release behavior.
    #[doc(hidden)]
    pub fn observe_current(&self) {
        let mut state = lock_mutex(&self.shared.state);
        if let Some(index) = lease_index(&state.leases, self.id) {
            if let Some(observation) = state.leases[index].observation.as_mut() {
                observation.attach_current();
            }
        }
    }

    /// Take a child lease from this lease's remainder.
    ///
    /// The child cannot use RAM or CPU the parent does not still hold, even when the pool has
    /// more free. The pool's free count stays put: the child is a portion of the parent.
    ///
    /// # Errors
    ///
    /// [`AdmissionError::ExceedsRemainder`] when the request is larger than this lease's
    /// remainder. [`AdmissionError::Cancelled`] when the study was cancelled.
    /// [`AdmissionError::Released`] when this lease is already gone.
    /// [`AdmissionError::Overflow`] when naming another lease does not fit in `u64`.
    pub fn acquire_child(&self, threads: u32, bytes: u64) -> Result<Self, AdmissionError> {
        self.acquire_child_request(AdmissionRequest::new(threads, bytes))
    }

    /// Child lease of `count * width` bytes, refused when the product overflows.
    ///
    /// # Errors
    ///
    /// [`AdmissionError::Overflow`] when `count * width` does not fit in `u64`. Otherwise the
    /// errors of [`Self::acquire_child`].
    pub fn acquire_child_product(
        &self,
        threads: u32,
        count: u64,
        width: u64,
    ) -> Result<Self, AdmissionError> {
        self.acquire_child_request(AdmissionRequest::try_product(threads, count, width)?)
    }

    /// Child lease for a request already checked for overflow.
    ///
    /// # Errors
    ///
    /// See [`Self::acquire_child`].
    pub fn acquire_child_request(&self, request: AdmissionRequest) -> Result<Self, AdmissionError> {
        let mut guard = lock_mutex(&self.shared.state);
        if self.flag.load(Ordering::SeqCst) {
            return Err(AdmissionError::Cancelled);
        }
        if lease_index(&guard.leases, self.id).is_none() {
            self.active.store(false, Ordering::SeqCst);
            return Err(AdmissionError::Released);
        }
        let id = grant_child(&mut guard, self.id, request)?;
        Ok(Self::new(Arc::clone(&self.shared), id, Arc::clone(&self.flag)))
    }

    /// Root lease on the pool that issued `self`.
    ///
    /// The new lease is not a portion of this remainder. This remainder stays
    /// put. The pool's free counters pay for it, under the same cancellation flag.
    ///
    /// # Errors
    ///
    /// [`AdmissionError::ExceedsRemainder`] when the pool's free counters are short.
    /// [`AdmissionError::Cancelled`] when the study was cancelled.
    /// [`AdmissionError::Released`] when this lease is already gone.
    /// [`AdmissionError::Overflow`] when naming another lease does not fit in `u64`.
    pub(crate) fn acquire_from_pool(
        &self,
        threads: u32,
        bytes: u64,
    ) -> Result<Self, AdmissionError> {
        let mut guard = lock_mutex(&self.shared.state);
        if self.flag.load(Ordering::SeqCst) {
            return Err(AdmissionError::Cancelled);
        }
        if lease_index(&guard.leases, self.id).is_none() {
            self.active.store(false, Ordering::SeqCst);
            return Err(AdmissionError::Released);
        }
        let request = AdmissionRequest::new(threads, bytes);
        let id = match grant_root(&mut guard, request) {
            Ok(id) => id,
            Err(GrantError::Overflow) => return Err(AdmissionError::Overflow),
            Err(GrantError::Short(error)) => return Err(error),
        };
        Ok(Self::new(Arc::clone(&self.shared), id, Arc::clone(&self.flag)))
    }

    /// CPU and RAM this lease still holds, including during cancellation cleanup.
    #[must_use]
    pub fn remainder(&self) -> AdmissionRemainder {
        let guard = lock_mutex(&self.shared.state);
        if let Some(index) = lease_index(&guard.leases, self.id) {
            let slot = &guard.leases[index];
            AdmissionRemainder::new(slot.threads, slot.bytes)
        } else {
            self.active.store(false, Ordering::SeqCst);
            AdmissionRemainder::zero()
        }
    }

    /// Return this lease's remainder to its parent, or to the pool when it is a root.
    ///
    /// Outstanding children keep their portion. That portion is not part of the returned
    /// remainder; each child returns it when that child is released.
    #[must_use = "the remainder is the CPU and RAM returned to the parent"]
    pub fn release(self) -> AdmissionRemainder {
        if !self.active.swap(false, Ordering::SeqCst) {
            return AdmissionRemainder::zero();
        }
        close_and_notify(&self.shared, self.id)
    }
}

impl Drop for AdmissionLease {
    fn drop(&mut self) {
        if self.active.swap(false, Ordering::SeqCst) {
            let _ = close_and_notify(&self.shared, self.id);
        }
    }
}

impl fmt::Debug for AdmissionLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AdmissionLease")
            .field("id", &self.id)
            .field("active", &self.active.load(Ordering::SeqCst))
            .finish_non_exhaustive()
    }
}

/// Shared wait state. A production token is this and nothing else.
struct CancelWait {
    cancelled: Arc<AtomicBool>,
    waiters: Mutex<Vec<Weak<SharedAdmission>>>,
}

impl CancelWait {
    fn register_pool(&self, shared: &Arc<SharedAdmission>) {
        let mut waiters = lock_mutex(&self.waiters);
        waiters.retain(|waiter| waiter.strong_count() > 0);
        let incoming = Arc::downgrade(shared);
        if !waiters.iter().any(|waiter| Weak::ptr_eq(waiter, &incoming)) {
            waiters.push(incoming);
        }
    }

    /// Store the cancelled flag and notify each pool while holding that pool's state lock.
    ///
    /// The flag is stored again under the pool lock, before `notify_all`. A waiter holds
    /// that same lock while it reads the flag and only then waits, so it cannot pass the
    /// read, miss the notify, and sleep until its bound.
    fn wake(&self) {
        let waiters = {
            let mut waiters = lock_mutex(&self.waiters);
            waiters.retain(|waiter| waiter.strong_count() > 0);
            waiters.clone()
        };
        // Publish before notifying so a waiter that registers after this snapshot
        // observes the flag under the pool lock and does not sleep.
        self.cancelled.store(true, Ordering::SeqCst);
        for waiter in waiters {
            let Some(shared) = waiter.upgrade() else {
                continue;
            };
            let _state = lock_mutex(&shared.state);
            self.cancelled.store(true, Ordering::SeqCst);
            shared.released.notify_all();
        }
    }
}

/// Test-only observation counter. It is not a field of [`CancelWait`].
struct AfterChecks {
    wait: CancelWait,
    left: AtomicUsize,
}

enum CancelFlag {
    Production(CancelWait),
    AfterChecks(AfterChecks),
}

impl Default for CancelFlag {
    fn default() -> Self {
        Self::Production(CancelWait {
            cancelled: Arc::new(AtomicBool::new(false)),
            waiters: Mutex::new(Vec::new()),
        })
    }
}

impl CancelFlag {
    fn wait(&self) -> &CancelWait {
        match self {
            Self::Production(wait) => wait,
            Self::AfterChecks(checks) => &checks.wait,
        }
    }

    fn register_pool(&self, shared: &Arc<SharedAdmission>) {
        self.wait().register_pool(shared);
    }

    fn wake(&self) {
        self.wait().wake();
    }

    fn is_requested(&self) -> bool {
        self.wait().cancelled.load(Ordering::SeqCst)
    }

    /// Production reads the flag. [`CancellationToken::cancel_after_checks`] spends one
    /// observation first. That counter lives only on [`Self::AfterChecks`].
    fn is_cancelled(&self) -> bool {
        if let Self::AfterChecks(checks) = self {
            let expired = checks
                .left
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                .is_err();
            if expired {
                checks.wait.wake();
            }
        }
        self.is_requested()
    }
}

/// Cooperative cancellation token.
///
/// [`Self::cancel`] wakes admission waits that registered on this token. The admission pool
/// lives on [`ExecutionContext`], so installing a pool does not retarget other clones of this
/// token.
#[derive(Clone, Default)]
pub struct CancellationToken {
    flag: Arc<CancelFlag>,
}

impl fmt::Debug for CancellationToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CancellationToken")
            .field("cancelled", &self.flag.is_requested())
            .finish()
    }
}

impl CancellationToken {
    /// Create a fresh token. It does not own an admission pool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A token that cancels itself on its `checks + 1`-th observation, so a
    /// test can stop a search deterministically in the middle (every budget
    /// charge observes the token once). The counter is a separate flag body.
    /// [`Self::new`] does not carry it.
    // doc(hidden): test helper called from integration tests across the workspace.
    #[doc(hidden)]
    #[must_use]
    pub fn cancel_after_checks(checks: usize) -> Self {
        Self {
            flag: Arc::new(CancelFlag::AfterChecks(AfterChecks {
                wait: CancelWait {
                    cancelled: Arc::new(AtomicBool::new(false)),
                    waiters: Mutex::new(Vec::new()),
                },
                left: AtomicUsize::new(checks),
            })),
        }
    }

    /// A non-owning handle that does not keep this token alive.
    #[must_use]
    pub fn downgrade(&self) -> WeakCancellationToken {
        WeakCancellationToken { flag: Arc::downgrade(&self.flag) }
    }

    /// Request cancellation and wake admission waits bound to this token.
    ///
    /// The cancelled flag is stored while the pool state lock is held, and the
    /// condvar is notified before that lock is released.
    pub fn cancel(&self) {
        self.flag.wake();
    }

    /// Whether cancellation was requested, read without spending one of the checks a
    /// [`Self::cancel_after_checks`] token counts.
    ///
    /// A hold that releases its lease on cancellation observes the request through this, so the
    /// number of checks a cell makes stays the cell's own planned count.
    #[must_use]
    pub fn is_requested(&self) -> bool {
        self.flag.is_requested()
    }

    /// Whether cancellation was requested.
    ///
    /// A production token only reads the flag. A token from [`Self::cancel_after_checks`] spends
    /// one observation first.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.flag.is_cancelled()
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.flag.wait().cancelled)
    }
}

/// A non-owning handle to a [`CancellationToken`].
///
/// It keeps the token's cancellation flag reachable without keeping the token
/// alive: once every owner of the token is gone, [`Self::upgrade`] returns
/// `None`. A queue of contexts waiting for a run can hold these without
/// leaking the contexts that were built and dropped.
#[derive(Clone, Debug)]
pub struct WeakCancellationToken {
    flag: Weak<CancelFlag>,
}

impl WeakCancellationToken {
    /// The token, if any owner of it is still alive.
    #[must_use]
    pub fn upgrade(&self) -> Option<CancellationToken> {
        Some(CancellationToken { flag: self.flag.upgrade()? })
    }

    /// Whether an owner of the token is still alive.
    #[must_use]
    pub fn is_alive(&self) -> bool {
        self.flag.strong_count() > 0
    }
}

/// Optional progress reporting sink.
pub trait ProgressSink: Send + Sync {
    /// Report progress in `[0.0, 1.0]` with an optional stage label.
    fn report(&self, fraction: f64, stage: &str);
}

/// RNG stream domain: a tag that keeps indexed stream families apart.
///
/// Indexed families must use [`RngFactory::stream_for`] with a dedicated domain
/// rather than `CONST.wrapping_add(index)` into [`RngFactory::stream`]: additive
/// family offsets collide across domains and, under an additive mixer, across
/// adjacent master seeds.
///
/// The domains the workspace's own crates draw from are the associated
/// constants below, tags `1..=19`. The set is open: a crate outside this
/// workspace names its own domain with [`Self::external`], which accepts only
/// tags from [`Self::EXTERNAL_BASE`] up so it cannot alias a built-in one. The
/// tag is the only input a domain contributes to a derived stream, so the
/// built-in constants derive exactly the streams the closed enum did.
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct StreamDomain(u64);

// The built-in domains keep their former variant spellings so no call site changes.
#[allow(
    non_upper_case_globals,
    reason = "the built-in domains keep their former variant spellings so no call site changes"
)]
impl StreamDomain {
    /// First tag available to [`Self::external`]; lower tags are reserved for
    /// the constants below.
    pub const EXTERNAL_BASE: u64 = 1 << 16;

    /// IID / block / cluster / Bayesian bootstrap replicates.
    pub const Resample: Self = Self(1);
    /// Lag-aligned temporal block-bootstrap families.
    pub const TemporalBlock: Self = Self(2);
    /// Bayesian draw / posterior simulation streams.
    pub const Bayesian: Self = Self(3);
    /// Structure-MCMC chains.
    pub const McmcStructure: Self = Self(4);
    /// Order-MCMC chains.
    pub const McmcOrder: Self = Self(5);
    /// DBN posterior MCMC chains.
    pub const McmcDbn: Self = Self(6);
    /// Graph-completion / identified-set posterior streams.
    pub const Completion: Self = Self(7);
    /// Counterfactual abduction / action / prediction noise.
    pub const Counterfactual: Self = Self(8);
    /// Attribution Monte Carlo arms.
    pub const Attribution: Self = Self(9);
    /// Design ranking and allocation draws.
    pub const Design: Self = Self(10);
    /// Statistical transport / retargeting draws.
    pub const Transport: Self = Self(11);
    /// Conditional-independence / CI null Monte Carlo.
    pub const StatsCi: Self = Self(12);
    /// Learner-internal randomness (forest, GBT, neural).
    pub const Learner: Self = Self(13);
    /// Estimator-local streams (AIPW, IV, RD, …).
    pub const Estimate: Self = Self(14);
    /// Temporal / DBN mediation block streams.
    pub const Mediation: Self = Self(15);
    /// Panel / interference path streams.
    pub const Panel: Self = Self(16);
    /// Facade execute-path streams.
    pub const Execute: Self = Self(17);
    /// Unit / integration test fixtures.
    pub const Test: Self = Self(18);
    /// Example programs.
    pub const Example: Self = Self(19);

    /// A domain outside the built-in set.
    ///
    /// # Panics
    /// If `tag` is below [`Self::EXTERNAL_BASE`]; in a `const` item this is a
    /// compile-time error.
    #[must_use]
    pub const fn external(tag: u64) -> Self {
        assert!(
            tag >= Self::EXTERNAL_BASE,
            "external stream domain tag collides with a built-in domain"
        );
        Self(tag)
    }

    /// The tag mixed into every stream this domain derives.
    #[must_use]
    pub const fn tag(self) -> u64 {
        self.0
    }
}

impl std::fmt::Debug for StreamDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self.0 {
            1 => "Resample",
            2 => "TemporalBlock",
            3 => "Bayesian",
            4 => "McmcStructure",
            5 => "McmcOrder",
            6 => "McmcDbn",
            7 => "Completion",
            8 => "Counterfactual",
            9 => "Attribution",
            10 => "Design",
            11 => "Transport",
            12 => "StatsCi",
            13 => "Learner",
            14 => "Estimate",
            15 => "Mediation",
            16 => "Panel",
            17 => "Execute",
            18 => "Test",
            19 => "Example",
            tag => return write!(f, "StreamDomain::external({tag})"),
        };
        f.write_str(name)
    }
}

/// Factory for deterministic RNG streams.
///
/// Streams are derived from a master seed and a stream id via a non-additive
/// mixer so algorithms can request reproducible substreams without a global RNG.
/// Prefer [`Self::stream_for`] so indexed families do not share additive ids.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RngFactory {
    master_seed: u64,
}

impl RngFactory {
    /// Create a factory from a master seed.
    #[must_use]
    pub const fn from_seed(master_seed: u64) -> Self {
        Self { master_seed }
    }

    /// Master seed.
    #[must_use]
    pub const fn master_seed(&self) -> u64 {
        self.master_seed
    }

    /// Derive a stream for a raw `stream_id`.
    ///
    /// Prefer [`Self::stream_for`] for new call sites. This entry point remains
    /// for low-level ids that are already domain-unique.
    #[must_use]
    pub fn stream(&self, stream_id: u64) -> CausalRng {
        CausalRng::from_seed(derive_stream(self.master_seed, stream_id))
    }

    /// Derive a stream for `(domain, index)`.
    ///
    /// Domain and index are packed without an additive family offset, then passed
    /// through the single [`derive_stream`] mixer with the master seed.
    #[must_use]
    pub fn stream_for(&self, domain: StreamDomain, index: u64) -> CausalRng {
        // Odd multiplier so domain tags stay distinct under xor with index.
        let stream_id = domain.tag().wrapping_mul(0xD1B5_4A32_D192_ED03) ^ index;
        self.stream(stream_id)
    }
}

/// The `SplitMix64` increment (the golden-ratio gamma).
pub const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// The `SplitMix64` output finalizer: a bijective avalanche mix of one word.
#[must_use]
pub const fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Advance a `SplitMix64` state by [`GOLDEN_GAMMA`] and return its next output.
///
/// [`CausalRng::next_u64`] is exactly this step on its own state.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(GOLDEN_GAMMA);
    mix64(*state)
}

/// Deterministic SplitMix64-based RNG for library algorithms.
#[derive(Clone, Debug)]
pub struct CausalRng {
    state: u64,
}

impl CausalRng {
    /// Create from a 64-bit seed.
    #[must_use]
    pub const fn from_seed(seed: u64) -> Self {
        // Avoid the all-zero fixed point of SplitMix by mixing once.
        Self { state: seed ^ GOLDEN_GAMMA }
    }

    /// Restore from a previously exported [`Self::state`].
    #[must_use]
    pub const fn from_state(state: u64) -> Self {
        Self { state }
    }

    /// Opaque stream state for checkpoint / CRN continuation.
    #[must_use]
    pub const fn state(&self) -> u64 {
        self.state
    }

    /// Next `u64` from the stream.
    #[must_use]
    pub fn next_u64(&mut self) -> u64 {
        splitmix64(&mut self.state)
    }

    /// Next `f64` in `[0, 1)`.
    #[must_use]
    pub fn next_f64(&mut self) -> f64 {
        // 53-bit mantissa extraction; precision loss vs full u64 is intentional.
        let bits = self.next_u64() >> 11;
        bits as f64 * (1.0 / ((1u64 << 53) as f64))
    }
}

/// Non-additive mix of `(master, stream_id)`.
///
/// The previous `master.wrapping_add(stream_id)` form made
/// `(master=s, stream=k)` identical to `(s−δ, k+δ)`, so adjacent user seeds
/// shared almost every bootstrap replicate stream.
///
/// This is the seed [`RngFactory::stream`] hands to [`CausalRng::from_seed`]; it
/// is a different mixer from the [`mix64`] finalizer (it keys two words), so the
/// two are not interchangeable.
#[must_use]
pub const fn derive_stream(master: u64, stream_id: u64) -> u64 {
    let mut z = master.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    z ^= stream_id.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    z = (z ^ (z >> 32)).wrapping_mul(0x1656_67B1_9E37_79F9);
    z = (z ^ (z >> 32)).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    z ^ (z >> 32)
}

#[derive(Clone, Copy)]
enum AdmissionWait {
    Immediate,
    Bounded(Duration),
    UntilAvailable,
}

fn admit(
    shared: &Arc<SharedAdmission>,
    request: AdmissionRequest,
    flag: &Arc<AtomicBool>,
    waiters: &CancelFlag,
    wait: AdmissionWait,
) -> Result<AdmissionLease, AdmissionError> {
    waiters.register_pool(shared);
    let mut guard = lock_mutex(&shared.state);
    let limit = match wait {
        AdmissionWait::Bounded(bound) => {
            let deadline = Instant::now().checked_add(bound).ok_or(AdmissionError::Overflow)?;
            Some((deadline, bound))
        }
        AdmissionWait::Immediate | AdmissionWait::UntilAvailable => None,
    };
    loop {
        if flag.load(Ordering::SeqCst) {
            return Err(AdmissionError::Cancelled);
        }
        match grant_root(&mut guard, request) {
            Ok(id) => {
                return Ok(AdmissionLease::new(Arc::clone(shared), id, Arc::clone(flag)));
            }
            Err(GrantError::Overflow) => return Err(AdmissionError::Overflow),
            Err(GrantError::Short(refusal)) => {
                let never_fits =
                    request.threads > guard.thread_budget || request.bytes > guard.byte_budget;
                if never_fits && matches!(wait, AdmissionWait::UntilAvailable) {
                    return Err(refusal);
                }
                guard = match wait {
                    AdmissionWait::Immediate => return Err(refusal),
                    AdmissionWait::Bounded(_) => wait_or_stop(shared, guard, refusal, limit)?,
                    AdmissionWait::UntilAvailable => block_on_release(shared, guard, None)?,
                };
            }
        }
    }
}

fn wait_or_stop<'a>(
    shared: &'a SharedAdmission,
    guard: MutexGuard<'a, AdmissionState>,
    refusal: AdmissionError,
    limit: Option<(Instant, Duration)>,
) -> Result<MutexGuard<'a, AdmissionState>, AdmissionError> {
    let Some((deadline, bound)) = limit else {
        return Err(refusal);
    };
    if Instant::now() >= deadline {
        return Err(AdmissionError::WaitBoundExceeded { bound });
    }
    wait_for_room(shared, guard, deadline)
}

fn wait_for_room<'a>(
    shared: &'a SharedAdmission,
    guard: MutexGuard<'a, AdmissionState>,
    deadline: Instant,
) -> Result<MutexGuard<'a, AdmissionState>, AdmissionError> {
    let timeout = deadline.saturating_duration_since(Instant::now());
    block_on_release(shared, guard, Some(timeout))
}

fn block_on_release<'a>(
    shared: &'a SharedAdmission,
    mut guard: MutexGuard<'a, AdmissionState>,
    timeout: Option<Duration>,
) -> Result<MutexGuard<'a, AdmissionState>, AdmissionError> {
    let Some(waiting) = guard.blocked_waiters.checked_add(1) else {
        return Err(AdmissionError::Overflow);
    };
    guard.blocked_waiters = waiting;
    let mut locked = match timeout {
        Some(timeout) => {
            let (locked, _timed_out) = shared
                .released
                .wait_timeout(guard, timeout)
                .unwrap_or_else(PoisonError::into_inner);
            locked
        }
        None => shared.released.wait(guard).unwrap_or_else(PoisonError::into_inner),
    };
    locked.blocked_waiters = locked.blocked_waiters.saturating_sub(1);
    Ok(locked)
}

/// Full execution context passed into algorithms.
#[derive(Clone)]
pub struct ExecutionContext {
    /// Parallelism budget.
    pub parallelism: Parallelism,
    /// Determinism policy.
    pub determinism: Determinism,
    /// RNG factory (no global RNG).
    pub rng: RngFactory,
    /// Memory budget.
    pub memory: MemoryBudget,
    /// Cancellation token.
    pub cancellation: CancellationToken,
    /// Admission pool later acquires decrement. Clones share this handle until
    /// [`Self::with_admission`] replaces it on that one context.
    pub admission: AdmissionPool,
    /// Optional progress sink.
    pub progress: Option<Arc<dyn ProgressSink>>,
    /// Kernel selection policy.
    pub kernel_policy: KernelPolicy,
    /// Cache policy.
    pub cache_policy: CachePolicy,
    /// Opt-in bootstrap early-stop (estimate SE path); disabled by every constructor.
    pub adaptive_bootstrap: AdaptiveBootstrapBudget,
    /// Opt-in Bayesian draw early-stop (Laplace redraws); disabled by every constructor.
    pub adaptive_draws: AdaptiveDrawBudget,
}

impl ExecutionContext {
    /// Construct a serial, strict, scalar-friendly context for tests.
    ///
    /// # Examples
    ///
    /// ```
    /// use antecedent_core::ExecutionContext;
    ///
    /// let ctx = ExecutionContext::for_tests(42);
    /// assert!(!ctx.cancellation.is_cancelled());
    /// ```
    #[must_use]
    pub fn for_tests(seed: u64) -> Self {
        Self {
            parallelism: Parallelism::serial(),
            determinism: Determinism::Strict,
            rng: RngFactory::from_seed(seed),
            memory: MemoryBudget::unlimited(),
            admission: AdmissionPool::fixed(1, u64::MAX),
            cancellation: CancellationToken::new(),
            progress: None,
            kernel_policy: KernelPolicy::scalar_only(),
            cache_policy: CachePolicy::disabled(),
            // Exact-N pins in unit tests; enable explicitly for adaptive MC tests.
            adaptive_bootstrap: AdaptiveBootstrapBudget::disabled(),
            adaptive_draws: AdaptiveDrawBudget::disabled(),
        }
    }

    /// Production context at [`default_user_threads`].
    #[must_use]
    pub fn production_default(seed: u64) -> Self {
        Self::production(seed, default_user_threads())
    }

    /// Clone with [`Parallelism::serial`]. Use for inner work when an outer
    /// map already saturates [`Self::parallelism`].
    #[must_use]
    pub fn serial_inner(&self) -> Self {
        let mut inner = self.clone();
        inner.parallelism = Parallelism::serial();
        inner
    }

    /// `f(0), …, f(n-1)` under this context. Results return in index order.
    ///
    /// When `n > 1` and `max_threads > 1`, work is chunked on `thread::scope`
    /// and `f` receives [`Self::serial_inner`] so nested pools stay serial.
    /// When `n <= 1` or the context is already serial, `f` receives a context
    /// bounded by this pool's worker budget; an already compliant context is reused.
    ///
    /// The parallel path acquires one admission lease for its workers before it
    /// spawns. The lease is dropped after the workers join, including when `f`
    /// fails. Cancellation keeps the lease charged until the workers join and cleanup
    /// drops it. A call that cannot take the lease,
    /// including one whose context is already cancelled, evaluates `f` on the
    /// caller with a serial inner context and spawns nothing. A second parallel call waits until the pool
    /// can hold that lease. The serial path does not take a lease, so a nested
    /// map on [`Self::serial_inner`] does not wait on the outer hold.
    /// [`MemoryBudget`] is not consulted.
    ///
    /// # Errors
    ///
    /// Returns the lowest-index error produced by `f`. Work above a known failure
    /// is skipped, so a failing run does not evaluate every remaining index.
    ///
    /// # Panics
    ///
    /// Panics if a worker fails to write a slot at or below the first error (a
    /// programming error in the pool, not in `f`).
    pub fn map_indexed<T, E, F>(&self, n: usize, f: F) -> Result<Vec<T>, E>
    where
        T: Send,
        E: Send,
        F: Fn(usize, &Self) -> Result<T, E> + Sync,
    {
        self.map_indexed_with_scratch(n, || (), |index, ctx, _scratch| f(index, ctx))
    }

    /// Execute an indexed map with shared admission even for a serial caller.
    /// Nested work reuses a live enclosing worker reservation and remains serial.
    ///
    /// # Errors
    /// Cancellation/admission refusal, or the original lowest-index task error.
    pub fn try_map_indexed<T, E, F>(&self, n: usize, f: F) -> Result<Vec<T>, MapExecutionError<E>>
    where
        T: Send,
        E: Send,
        F: Fn(usize, &Self) -> Result<T, E> + Sync,
    {
        self.try_map_indexed_with_scratch(n, || (), |index, ctx, ()| f(index, ctx))
    }

    /// Admitted map with one reusable scratch value per worker, including serial work.
    /// Scratch construction occurs only after admission and a cancellation check.
    ///
    /// # Errors
    /// Cancellation/admission refusal, or the original lowest-index task error.
    pub fn try_map_indexed_with_scratch<T, S, E, F>(
        &self,
        n: usize,
        make_scratch: impl Fn() -> S + Sync,
        f: F,
    ) -> Result<Vec<T>, MapExecutionError<E>>
    where
        T: Send,
        S: Send,
        E: Send,
        F: Fn(usize, &Self, &mut S) -> Result<T, E> + Sync,
    {
        if n == 0 {
            return Ok(Vec::new());
        }
        if self.cancellation.is_cancelled() {
            return Err(MapExecutionError::Admission(AdmissionError::Cancelled));
        }
        let call = |index, ctx: &Self, scratch: &mut S| {
            if ctx.cancellation.is_cancelled() {
                return Err(MapExecutionError::Admission(AdmissionError::Cancelled));
            }
            if !crate::numerical_admission::worker()
                .is_some_and(|lease| lease.belongs_to(&ctx.admission))
            {
                return Err(MapExecutionError::Admission(AdmissionError::Released));
            }
            f(index, ctx, scratch).map_err(MapExecutionError::Work)
        };
        if let Some(worker) =
            crate::numerical_admission::worker().filter(|lease| lease.belongs_to(&self.admission))
        {
            let inner = self.serial_inner();
            return crate::numerical_admission::with_worker(worker, || {
                inner.map_indexed_with_scratch(n, make_scratch, call)
            });
        }
        if self.map_worker_count(n) <= 1 {
            let lease = Arc::new(self.acquire_map_lease(1)?);
            let inner = self.serial_inner();
            return crate::numerical_admission::with_worker(lease, || {
                inner.map_indexed_with_scratch(n, make_scratch, call)
            });
        }
        self.map_indexed_with_scratch(n, make_scratch, call)
    }

    /// [`Self::map_indexed`] with one scratch value per worker.
    ///
    /// The serial path builds one scratch and reuses it. The parallel path builds
    /// one scratch inside each worker after the admission lease is held, and
    /// stops scheduling indexes above the first error.
    ///
    /// # Errors
    ///
    /// Returns the lowest-index error produced by `f`.
    ///
    /// # Panics
    ///
    /// Panics if a worker fails to write a slot at or below the first error.
    pub fn map_indexed_with_scratch<T, S, E, F>(
        &self,
        n: usize,
        make_scratch: impl Fn() -> S + Sync,
        f: F,
    ) -> Result<Vec<T>, E>
    where
        T: Send,
        S: Send,
        E: Send,
        F: Fn(usize, &Self, &mut S) -> Result<T, E> + Sync,
    {
        if n == 0 {
            return Ok(Vec::new());
        }
        let threads = self.map_worker_count(n);
        if threads <= 1 {
            let mut scratch = make_scratch();
            // A caller can request more workers than its shared account funds.
            // Keep nested numerical kernels within that account on the serial path too.
            let bounded;
            let pool_threads = self.admission.thread_budget().max(1);
            let ctx = if self.parallelism.max_threads.get() > pool_threads {
                bounded = Self {
                    parallelism: Parallelism::bounded(
                        NonZeroThreadCount::new(pool_threads).unwrap_or(NonZeroThreadCount::MIN),
                    ),
                    ..self.clone()
                };
                &bounded
            } else {
                self
            };
            return (0..n).map(|index| f(index, ctx, &mut scratch)).collect();
        }
        let Ok(worker_threads) = u32::try_from(threads) else {
            let mut scratch = make_scratch();
            return (0..n).map(|index| f(index, self, &mut scratch)).collect();
        };
        let Ok(lease) = self.acquire_map_lease(worker_threads) else {
            let mut scratch = make_scratch();
            let inner = self.serial_inner();
            return (0..n).map(|index| f(index, &inner, &mut scratch)).collect();
        };
        let inner = self.serial_inner();
        let mut slots: Vec<Option<Result<T, E>>> = (0..n).map(|_| None).collect();
        let first_failure = AtomicUsize::new(usize::MAX);
        let observation = crate::execution_attempt::ObservationToken::capture();
        let lease = Arc::new(lease);
        let numerical = crate::numerical_admission::with_worker(
            Arc::clone(&lease),
            crate::numerical_admission::NumericalAdmissionToken::capture,
        );
        std::thread::scope(|scope| {
            let f = &f;
            let inner = &inner;
            let first_failure = &first_failure;
            let make_scratch = &make_scratch;
            let mut rest = slots.as_mut_slice();
            let mut start = 0usize;
            for t in 0..threads {
                let take = rest.len().div_ceil(threads - t);
                let (mine, next) = rest.split_at_mut(take);
                let begin = start;
                let observation = &observation;
                let numerical = &numerical;
                scope.spawn(move || {
                    observation.execute(|| {
                        let mut scratch = make_scratch();
                        for (k, slot) in mine.iter_mut().enumerate() {
                            let index = begin + k;
                            if first_failure.load(Ordering::Relaxed) < index {
                                break;
                            }
                            let outcome =
                                numerical.execute(inner, || f(index, inner, &mut scratch));
                            if outcome.is_err() {
                                first_failure.fetch_min(index, Ordering::Relaxed);
                            }
                            *slot = Some(outcome);
                        }
                    })
                });
                rest = next;
                start += take;
                if rest.is_empty() {
                    break;
                }
            }
        });
        drop(lease);
        slots
            .into_iter()
            .map(|slot| slot.expect("every index below the first error was filled"))
            .collect()
    }

    /// Worker count for [`Self::map_indexed`], never above the admission pool.
    ///
    /// `n` is at least one. A count of one stays on the caller and takes no lease.
    fn map_worker_count(&self, n: usize) -> usize {
        let requested = usize::try_from(self.parallelism.max_threads.get()).unwrap_or(usize::MAX);
        let requested = requested.clamp(1, n);
        if requested == 1 {
            return 1;
        }
        let pool_slots = usize::try_from(self.admission().thread_budget()).unwrap_or(usize::MAX);
        requested.min(pool_slots)
    }

    fn acquire_map_lease(&self, threads: u32) -> Result<AdmissionLease, AdmissionError> {
        let request = AdmissionRequest::new(threads, 0);
        admit(
            &self.admission.shared,
            request,
            &self.cancellation.cancel_flag(),
            &self.cancellation.flag,
            AdmissionWait::UntilAvailable,
        )
    }

    /// Production context: optimized kernels allowed, cache enabled, bounded threads.
    ///
    /// Monte Carlo effort is evaluated in full: the requested bootstrap
    /// replicates and posterior draws are what the calibration gates certify,
    /// so no early-stop budget is enabled. Opt in by setting
    /// [`Self::adaptive_bootstrap`] / [`Self::adaptive_draws`] explicitly.
    #[must_use]
    pub fn production(seed: u64, max_threads: u32) -> Self {
        let threads =
            NonZeroThreadCount::new(max_threads.max(1)).unwrap_or(NonZeroThreadCount::MIN);
        Self {
            parallelism: Parallelism::bounded(threads),
            determinism: Determinism::Strict,
            rng: RngFactory::from_seed(seed),
            memory: MemoryBudget::unlimited(),
            admission: AdmissionPool::fixed(threads.get(), u64::MAX),
            cancellation: CancellationToken::new(),
            progress: None,
            kernel_policy: KernelPolicy::default_policy(),
            cache_policy: CachePolicy::enabled(None),
            adaptive_bootstrap: AdaptiveBootstrapBudget::disabled(),
            adaptive_draws: AdaptiveDrawBudget::disabled(),
        }
    }

    /// Install `pool` on this context.
    ///
    /// Later acquires, including [`Self::map_indexed`], decrement `pool`. A clone taken
    /// before this call keeps the pool it already held. [`Self::memory`] is not an
    /// admission account and is left unchanged.
    #[must_use]
    pub fn with_admission(mut self, pool: AdmissionPool) -> Self {
        self.admission = pool;
        self
    }

    /// The pool later acquires decrement.
    #[must_use]
    pub fn admission(&self) -> AdmissionPool {
        self.admission.clone()
    }

    /// Take a root lease from [`Self::admission`], decrementing that pool.
    ///
    /// # Errors
    ///
    /// [`AdmissionError::ExceedsRemainder`] when the pool's remainder is short.
    /// That refusal charges nothing. [`MemoryBudget`] is not consulted.
    /// [`AdmissionError::Cancelled`] when this context is already cancelled.
    /// [`AdmissionError::Overflow`] when the lease id does not fit in `u64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use antecedent_core::execution::AdmissionPool;
    /// use antecedent_core::ExecutionContext;
    ///
    /// let shared = AdmissionPool::fixed(4, 1_000);
    /// let study = ExecutionContext::for_tests(1).with_admission(shared.clone());
    /// let lease = study.acquire_admission(3, 250).unwrap();
    /// assert_eq!(shared.available().threads(), 1);
    /// assert_eq!(shared.available().bytes(), 750);
    /// assert_eq!(lease.release().threads(), 3);
    /// assert_eq!(shared.available().threads(), 4);
    /// ```
    pub fn acquire_admission(
        &self,
        threads: u32,
        bytes: u64,
    ) -> Result<AdmissionLease, AdmissionError> {
        self.acquire_request(AdmissionRequest::new(threads, bytes), None)
    }

    /// Wait up to `bound` for a root lease. Cancellation ends the wait with no hold.
    ///
    /// # Errors
    ///
    /// [`AdmissionError::WaitBoundExceeded`] when `bound` elapses first.
    /// [`AdmissionError::Cancelled`] when this context is cancelled. The other
    /// errors match [`Self::acquire_admission`].
    pub fn acquire_admission_bounded(
        &self,
        threads: u32,
        bytes: u64,
        bound: Duration,
    ) -> Result<AdmissionLease, AdmissionError> {
        self.acquire_request(AdmissionRequest::new(threads, bytes), Some(bound))
    }

    /// Root lease of `count * width` bytes.
    ///
    /// # Errors
    ///
    /// [`AdmissionError::Overflow`] when `count * width` does not fit in `u64`.
    /// The product is not wrapped, and the pool is not decremented. Otherwise
    /// the errors of [`Self::acquire_admission`].
    pub fn acquire_admission_product(
        &self,
        threads: u32,
        count: u64,
        width: u64,
    ) -> Result<AdmissionLease, AdmissionError> {
        self.acquire_request(AdmissionRequest::try_product(threads, count, width)?, None)
    }

    fn acquire_request(
        &self,
        request: AdmissionRequest,
        bound: Option<Duration>,
    ) -> Result<AdmissionLease, AdmissionError> {
        let wait = match bound {
            Some(bound) => AdmissionWait::Bounded(bound),
            None => AdmissionWait::Immediate,
        };
        admit(
            &self.admission.shared,
            request,
            &self.cancellation.cancel_flag(),
            &self.cancellation.flag,
            wait,
        )
    }
}

/// A shared-scheduler refusal or the original lowest-index task error.
#[derive(Debug, Error)]
pub enum MapExecutionError<E> {
    /// No work started without its required shared CPU reservation.
    #[error(transparent)]
    Admission(#[from] AdmissionError),
    /// The original task error, unchanged.
    #[error("worker execution failed: {0}")]
    Work(E),
}

/// Host request identity: scoped to the complete scientific + execution inputs.
///
/// A key reused for different inputs is a conflict. The request binds every
/// contract layer (target and population, premises, products, program,
/// inference binding, observation, data snapshot) plus execution lineage, so
/// a changed estimand, prior, or numeric knob under a reused key is detected.
/// The host owns scheduling; this record only names what the engine
/// considered one logical request.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct RequestIdentity {
    /// Caller-supplied idempotency key.
    pub idempotency_key: crate::identity::SemanticDigest,
    /// Every contract identity layer the request was bound to.
    pub contract: crate::identity::ContractIdentities,
    /// Execution-lineage identity (seed, threads, backend, budgets, version).
    pub execution: crate::identity::SemanticDigest,
}

impl RequestIdentity {
    /// Construct a complete request identity.
    #[must_use]
    pub const fn new(
        idempotency_key: crate::identity::SemanticDigest,
        contract: crate::identity::ContractIdentities,
        execution: crate::identity::SemanticDigest,
    ) -> Self {
        Self { idempotency_key, contract, execution }
    }

    /// Whether `other` reuses this key for a different scientific request.
    #[must_use]
    pub fn conflicts_with(&self, other: &Self) -> bool {
        self.idempotency_key == other.idempotency_key
            && (self.contract != other.contract || self.execution != other.execution)
    }
}

/// Lifecycle of an external execution request. Distinct from scientific result status.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum ExecutionRequestState {
    /// Accepted but not started.
    Pending,
    /// Currently executing.
    Running,
    /// Finished with a published scientific result.
    Completed,
    /// Refused before or during execution (scientific or license).
    Refused,
    /// Failed without publishing a completed claim.
    Failed,
    /// Cancelled; diagnostics may be retained, a claim must not be published.
    Cancelled,
}

impl ExecutionRequestState {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Refused => "refused",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    /// Whether this state may publish a completed claim.
    #[must_use]
    pub const fn publishes_claim(self) -> bool {
        matches!(self, Self::Completed)
    }
}

/// Receipt for one host request. Cancellation or failure cannot mark a claim current.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionReceipt {
    /// Request identity this receipt answers.
    pub request: RequestIdentity,
    /// Lifecycle state.
    pub state: ExecutionRequestState,
}

impl ExecutionReceipt {
    /// Construct a receipt.
    #[must_use]
    pub const fn new(request: RequestIdentity, state: ExecutionRequestState) -> Self {
        Self { request, state }
    }
}

impl core::fmt::Debug for ExecutionContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ExecutionContext")
            .field("parallelism", &self.parallelism)
            .field("determinism", &self.determinism)
            .field("rng", &self.rng)
            .field("memory", &self.memory)
            .field("cancellation_cancelled", &self.cancellation.is_cancelled())
            .field("progress_is_some", &self.progress.is_some())
            .field("kernel_policy", &self.kernel_policy)
            .field("cache_policy", &self.cache_policy)
            .field("adaptive_bootstrap", &self.adaptive_bootstrap)
            .field("adaptive_draws", &self.adaptive_draws)
            .field("admission", &self.admission())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::{Duration, Instant};

    #[test]
    fn rng_streams_are_deterministic() {
        let factory = RngFactory::from_seed(42);
        let mut a1 = factory.stream_for(StreamDomain::Test, 0);
        let mut a2 = factory.stream_for(StreamDomain::Test, 0);
        let mut b = factory.stream_for(StreamDomain::Test, 1);
        let seq_a1: Vec<u64> = (0..8).map(|_| a1.next_u64()).collect();
        let seq_a2: Vec<u64> = (0..8).map(|_| a2.next_u64()).collect();
        let seq_b: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        assert_eq!(seq_a1, seq_a2);
        assert_ne!(seq_a1, seq_b);
    }

    #[test]
    fn independent_factories_same_seed_match() {
        let f1 = RngFactory::from_seed(7);
        let f2 = RngFactory::from_seed(7);
        let mut s1 = f1.stream_for(StreamDomain::Test, 99);
        let mut s2 = f2.stream_for(StreamDomain::Test, 99);
        for _ in 0..32 {
            assert_eq!(s1.next_u64(), s2.next_u64());
        }
    }

    #[test]
    fn derive_stream_rejects_additive_master_stream_alias() {
        // (s, k) must not equal (s − 1, k + 1).
        for s in [1u64, 42, 1000, u64::MAX / 2, u64::MAX] {
            for k in [0u64, 1, 999, 0xA7E0_0001_1000] {
                let a = RngFactory::from_seed(s).stream(k).state();
                let b = RngFactory::from_seed(s.wrapping_sub(1)).stream(k.wrapping_add(1)).state();
                assert_ne!(a, b, "alias at master={s} stream={k}");
                let af = RngFactory::from_seed(s).stream_for(StreamDomain::Resample, k).state();
                let bf = RngFactory::from_seed(s.wrapping_sub(1))
                    .stream_for(StreamDomain::Resample, k.wrapping_add(1))
                    .state();
                assert_ne!(af, bf, "stream_for alias at master={s} index={k}");
            }
        }
    }

    #[test]
    fn stream_for_domains_separate_same_index() {
        let factory = RngFactory::from_seed(42);
        let a = factory.stream_for(StreamDomain::Resample, 7).state();
        let b = factory.stream_for(StreamDomain::TemporalBlock, 7).state();
        let c = factory.stream_for(StreamDomain::McmcStructure, 7).state();
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
    }

    #[test]
    fn built_in_domain_tags_are_pinned() {
        // The tag is the whole domain contribution to a derived stream; these
        // are the values the closed enum carried, so they may never change.
        let pinned = [
            (StreamDomain::Resample, 1),
            (StreamDomain::TemporalBlock, 2),
            (StreamDomain::Bayesian, 3),
            (StreamDomain::McmcStructure, 4),
            (StreamDomain::McmcOrder, 5),
            (StreamDomain::McmcDbn, 6),
            (StreamDomain::Completion, 7),
            (StreamDomain::Counterfactual, 8),
            (StreamDomain::Attribution, 9),
            (StreamDomain::Design, 10),
            (StreamDomain::Transport, 11),
            (StreamDomain::StatsCi, 12),
            (StreamDomain::Learner, 13),
            (StreamDomain::Estimate, 14),
            (StreamDomain::Mediation, 15),
            (StreamDomain::Panel, 16),
            (StreamDomain::Execute, 17),
            (StreamDomain::Test, 18),
            (StreamDomain::Example, 19),
        ];
        for (domain, tag) in pinned {
            assert_eq!(domain.tag(), tag);
        }
        let factory = RngFactory::from_seed(42);
        let by_domain = factory.stream_for(StreamDomain::Transport, 5).state();
        let by_hand = factory.stream(11u64.wrapping_mul(0xD1B5_4A32_D192_ED03) ^ 5).state();
        assert_eq!(by_domain, by_hand);
    }

    #[test]
    fn external_domains_stay_clear_of_built_in_tags() {
        const OURS: StreamDomain = StreamDomain::external(StreamDomain::EXTERNAL_BASE);
        assert_eq!(OURS.tag(), StreamDomain::EXTERNAL_BASE);
        assert_ne!(OURS, StreamDomain::Example);
        assert_eq!(format!("{:?}", StreamDomain::Resample), "Resample");
        let factory = RngFactory::from_seed(42);
        assert_ne!(
            factory.stream_for(OURS, 0).state(),
            factory.stream_for(StreamDomain::Example, 0).state()
        );
    }

    #[test]
    #[should_panic(expected = "collides with a built-in domain")]
    fn external_domain_below_the_base_is_refused() {
        let _ = StreamDomain::external(StreamDomain::Example.tag());
    }

    #[test]
    fn cancellation_is_shared_across_clones() {
        let token = CancellationToken::new();
        let clone = token.clone();
        assert!(!token.is_cancelled());
        clone.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn request_identity_conflicts_on_reused_key() {
        use crate::identity::{ContractIdentities, SemanticDigest};
        let digest = |byte: u8| SemanticDigest::from_bytes([byte; 32]);
        let identities = |target: u8, inference: u8, data: u8| {
            ContractIdentities::new(
                digest(target),
                digest(2),
                Some(digest(3)),
                Some(digest(4)),
                digest(inference),
                digest(6),
                digest(data),
            )
        };
        let key = digest(1);
        let execution = digest(8);
        let first = RequestIdentity::new(key, identities(10, 5, 7), execution);
        let same = RequestIdentity::new(key, identities(10, 5, 7), execution);
        assert!(!first.conflicts_with(&same));
        // Data snapshot, target population, and inference binding each change
        // the scientific request even when the program digest is shared.
        for changed in [identities(10, 5, 9), identities(11, 5, 7), identities(10, 12, 7)] {
            assert!(first.conflicts_with(&RequestIdentity::new(key, changed, execution)));
        }
        assert!(first.conflicts_with(&RequestIdentity::new(key, identities(10, 5, 7), digest(9))));
        assert!(!first.conflicts_with(&RequestIdentity::new(
            digest(99),
            identities(11, 12, 9),
            execution
        )));
        assert!(!ExecutionRequestState::Cancelled.publishes_claim());
        assert!(ExecutionRequestState::Completed.publishes_claim());
        let receipt = ExecutionReceipt::new(first, ExecutionRequestState::Cancelled);
        assert_eq!(receipt.state.as_str(), "cancelled");
    }

    #[test]
    fn production_token_does_not_trip_and_with_admission_keeps_clones() {
        let token = CancellationToken::new();
        for _ in 0..8 {
            assert!(!token.is_cancelled());
        }
        assert!(!format!("{token:?}").contains("trip"));
        let tripping = CancellationToken::cancel_after_checks(0);
        assert!(tripping.is_cancelled());

        let original = AdmissionPool::fixed(2, 10);
        let study = ExecutionContext::for_tests(1).with_admission(original.clone());
        let clone = study.clone();
        let replacement = AdmissionPool::fixed(8, 80);
        let moved = clone.with_admission(replacement.clone());
        assert_eq!(study.admission().thread_budget(), 2);
        assert_eq!(moved.admission().thread_budget(), 8);
        let _lease = moved.acquire_admission(8, 80).unwrap();
        assert_eq!(original.available().threads(), 2);
        assert_eq!(replacement.available().threads(), 0);
    }

    #[test]
    fn credit_overflow_panics_and_leaves_the_lease() {
        let mut state = AdmissionState {
            thread_budget: u32::MAX,
            byte_budget: 0,
            threads_free: u32::MAX,
            bytes_free: 0,
            leases: vec![LeaseSlot {
                observation: None,
                id: 7,
                parent: None,
                threads: 1,
                bytes: 0,
                cancelled: false,
            }],
            next_id: 8,
            blocked_waiters: 0,
        };
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            close_slot(&mut state, 0);
        }));
        let payload = panicked.expect_err("credit overflow must panic");
        let text = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or_default();
        assert!(text.contains("overflow"), "{text}");
        assert_eq!(state.leases.len(), 1);
        assert_eq!(state.leases[0].id, 7);
        assert!(!state.leases[0].cancelled);
    }

    #[test]
    fn cancel_wakes_a_blocked_waiter() {
        let shared = AdmissionPool::fixed(1, 8);
        let study = ExecutionContext::for_tests(1).with_admission(shared.clone());
        let held = study.acquire_admission(1, 8).unwrap();
        let waiting_study = study.clone();
        let started = Instant::now();
        let waiting = thread::spawn(move || {
            waiting_study.acquire_admission_bounded(1, 8, Duration::from_secs(30))
        });
        while shared.blocked_waiters() == 0 {
            assert!(started.elapsed() < Duration::from_secs(2), "waiter did not block");
            thread::yield_now();
        }
        study.cancellation.cancel();
        let result = waiting.join().unwrap();
        assert!(started.elapsed() < Duration::from_secs(2), "cancel left the waiter sleeping");
        assert!(matches!(result, Err(AdmissionError::Cancelled)));
        drop(held);
        assert_eq!(shared.available(), AdmissionRemainder::new(1, 8));
    }

    #[test]
    fn two_contexts_do_not_share_a_mutated_pool() {
        let original = AdmissionPool::fixed(2, 10);
        let first = ExecutionContext::for_tests(1).with_admission(original.clone());
        let sibling = first.clone();
        let replacement = AdmissionPool::fixed(4, 40);
        let moved = sibling.with_admission(replacement.clone());
        let held = first.acquire_admission(2, 10).unwrap();
        assert_eq!(original.available().threads(), 0);
        assert_eq!(original.available().bytes(), 0);
        assert_eq!(replacement.available().threads(), 4);
        let moved_hold = moved.acquire_admission(4, 40).unwrap();
        assert_eq!(replacement.available().threads(), 0);
        moved.cancellation.cancel();
        assert!(first.cancellation.is_requested());
        assert_eq!(first.admission().thread_budget(), 2);
        assert_eq!(moved.admission().thread_budget(), 4);
        drop(held);
        drop(moved_hold);
        assert_eq!(original.available().threads(), 2);
        assert_eq!(replacement.available().threads(), 4);
    }

    #[test]
    fn reap_removes_every_cancelled_lease_in_one_scan() {
        let mut state = AdmissionState {
            thread_budget: 10,
            byte_budget: 100,
            threads_free: 4,
            bytes_free: 40,
            leases: vec![
                LeaseSlot {
                    id: 3,
                    parent: Some(1),
                    threads: 1,
                    bytes: 10,
                    observation: None,
                    cancelled: true,
                },
                LeaseSlot {
                    id: 2,
                    parent: None,
                    threads: 3,
                    bytes: 30,
                    observation: None,
                    cancelled: false,
                },
                LeaseSlot {
                    id: 1,
                    parent: None,
                    threads: 2,
                    bytes: 20,
                    observation: None,
                    cancelled: true,
                },
                LeaseSlot {
                    id: 4,
                    parent: Some(2),
                    threads: 0,
                    bytes: 0,
                    observation: None,
                    cancelled: false,
                },
            ],
            next_id: 5,
            blocked_waiters: 0,
        };
        let (removed, visits) = reap_cancelled(&mut state);
        assert_eq!(removed, 2);
        assert_eq!(visits, 4, "a restart would visit kept slots again");
        assert_eq!(state.leases.iter().map(|slot| slot.id).collect::<Vec<_>>(), vec![2, 4]);
        assert!(state.leases.iter().all(|slot| !slot.cancelled));
        assert_eq!(state.leases[1].parent, Some(2));
        assert_eq!(state.threads_free, 7);
        assert_eq!(state.bytes_free, 70);
        let mut threads = state.threads_free;
        let mut bytes = state.bytes_free;
        for slot in &state.leases {
            threads += slot.threads;
            bytes += slot.bytes;
        }
        assert_eq!(threads, state.thread_budget);
        assert_eq!(bytes, state.byte_budget);
    }

    #[test]
    fn cancelled_parent_and_child_hold_live_memory_until_cleanup() {
        let pool = AdmissionPool::fixed(2, 100);
        let owner = ExecutionContext::for_tests(1).with_admission(pool.clone());
        let other = ExecutionContext::for_tests(2).with_admission(pool.clone());
        let parent = owner.acquire_admission(2, 100).unwrap();
        let child = parent.acquire_child(1, 40).unwrap();
        owner.cancellation.cancel();
        assert_eq!(pool.available(), AdmissionRemainder::zero());
        assert!(matches!(parent.acquire_child(0, 1), Err(AdmissionError::Cancelled)));
        assert!(other.acquire_admission(1, 1).is_err());
        drop(parent);
        assert_eq!(pool.available(), AdmissionRemainder::new(1, 60));
        let _replacement = other.acquire_admission(1, 60).unwrap();
        assert_eq!(pool.available(), AdmissionRemainder::zero());
        drop(child);
        assert_eq!(pool.available(), AdmissionRemainder::new(1, 40));
    }

    #[test]
    fn map_indexed_with_scratch_reuses_one_buffer_per_worker() {
        let ctx = ExecutionContext::production(1, 4);
        let built = AtomicUsize::new(0);
        let out = ctx
            .map_indexed_with_scratch(
                8,
                || {
                    built.fetch_add(1, Ordering::SeqCst);
                    Vec::<usize>::new()
                },
                |index, inner, scratch| {
                    assert_eq!(inner.parallelism.max_threads.get(), 1);
                    scratch.push(index);
                    Ok::<_, ()>(scratch.clone())
                },
            )
            .unwrap();
        assert!(built.load(Ordering::SeqCst) <= 4);
        assert!(built.load(Ordering::SeqCst) >= 1);
        assert_eq!(
            out.iter()
                .map(|row| *row.last().expect("scratch recorded the index"))
                .collect::<Vec<_>>(),
            (0..8).collect::<Vec<_>>()
        );
        assert!(out.iter().any(|row| row.len() > 1), "a worker reused its scratch");
    }

    #[test]
    fn map_indexed_preserves_order_and_serializes_inner() {
        let ctx = ExecutionContext::production(1, 4);
        let out = ctx
            .map_indexed(8, |i, inner| {
                assert_eq!(inner.parallelism.max_threads.get(), 1);
                Ok::<_, ()>(i * 10)
            })
            .unwrap();
        assert_eq!(out, vec![0, 10, 20, 30, 40, 50, 60, 70]);
        let serial = ExecutionContext::for_tests(1);
        let inner_threads = serial
            .map_indexed(1, |_, inner| Ok::<_, ()>(inner.parallelism.max_threads.get()))
            .unwrap();
        assert_eq!(inner_threads, vec![1]);
        assert_eq!(serial.serial_inner().parallelism.max_threads.get(), 1);
    }

    #[test]
    fn map_indexed_fallback_keeps_inner_parallelism_within_admission() {
        let ctx =
            ExecutionContext::production(1, 4).with_admission(AdmissionPool::fixed(1, u64::MAX));
        for n in [1, 8] {
            let threads = ctx
                .map_indexed(n, |_, inner| Ok::<_, ()>(inner.parallelism.max_threads.get()))
                .unwrap();
            assert_eq!(threads, vec![1; n]);
        }
        let ctx =
            ExecutionContext::production(1, 4).with_admission(AdmissionPool::fixed(2, u64::MAX));
        let threads = ctx
            .map_indexed(1, |_, inner| Ok::<_, ()>(inner.parallelism.max_threads.get()))
            .unwrap();
        assert_eq!(threads, vec![2]);
        ctx.cancellation.cancel();
        let threads = ctx
            .map_indexed(8, |_, inner| Ok::<_, ()>(inner.parallelism.max_threads.get()))
            .unwrap();
        assert_eq!(threads, vec![1; 8]);
    }

    #[test]
    fn map_indexed_stops_at_a_failure_and_reports_the_lowest_index() {
        let ctx = ExecutionContext::production(1, 4);
        let evaluated = AtomicUsize::new(0);
        let n = 2000;
        // Index 0 fails immediately; its own chunk (500 items) must stop there.
        let result = ctx.map_indexed(n, |i, _| {
            evaluated.fetch_add(1, Ordering::Relaxed);
            if i == 0 { Err(i) } else { Ok(i) }
        });
        assert_eq!(result, Err(0));
        assert!(evaluated.load(Ordering::Relaxed) <= n - 499, "the failing chunk must not run on");
        // Two failures in different chunks: the lowest index wins on every run.
        for _ in 0..20 {
            let result =
                ctx.map_indexed(n, |i, _| if i == 700 || i == 1500 { Err(i) } else { Ok(i) });
            assert_eq!(result, Err(700));
        }
        // No failure: everything is evaluated, in order.
        let all = ctx.map_indexed(n, |i, _| Ok::<_, ()>(i * 2)).unwrap();
        assert_eq!(all, (0..n).map(|i| i * 2).collect::<Vec<_>>());
    }

    #[test]
    fn adaptive_budgets_are_disabled_by_default() {
        assert_eq!(AdaptiveBootstrapBudget::default(), AdaptiveBootstrapBudget::disabled());
        assert_eq!(AdaptiveDrawBudget::default(), AdaptiveDrawBudget::disabled());
        let spread = AdaptiveBootstrapBudget { min_replicates: 5, ..Default::default() };
        assert!(!spread.enabled);
        assert_eq!(spread.required_replicates(), u32::MAX);
    }

    #[test]
    fn unavailable_arch_simd_is_not_an_effective_request() {
        const { assert!(!ARCH_SIMD_COMPILED) };
        assert!(!KernelPolicy::default_policy().arch_simd_effective());
        assert!(!KernelPolicy::scalar_only().arch_simd_effective());
    }

    #[test]
    fn default_user_threads_is_at_least_one_and_capped() {
        let n = default_user_threads();
        assert!(n >= 1);
        assert!(n <= DEFAULT_USER_THREAD_CAP);
        let ctx = ExecutionContext::production_default(1);
        assert_eq!(ctx.parallelism.max_threads.get(), n);
        assert_ne!(ctx.kernel_policy, KernelPolicy::scalar_only());
    }

    #[test]
    fn production_and_test_contexts_evaluate_full_monte_carlo_effort() {
        let production = ExecutionContext::production(1, 4);
        let tests = ExecutionContext::for_tests(1);
        assert_eq!(production.adaptive_bootstrap, AdaptiveBootstrapBudget::disabled());
        assert_eq!(production.adaptive_draws, AdaptiveDrawBudget::disabled());
        assert_eq!(production.adaptive_bootstrap, tests.adaptive_bootstrap);
        assert_eq!(production.adaptive_draws, tests.adaptive_draws);
    }

    #[test]
    fn bootstrap_budget_floor_is_the_monte_carlo_error_bound() {
        let at = |eps: f64, min: u32| {
            AdaptiveBootstrapBudget { enabled: true, min_replicates: min, se_rel_epsilon: eps }
                .required_replicates()
        };
        // 1/√(2(B−1)) ≤ ε  ⇔  B ≥ 1 + 1/(2ε²).
        assert_eq!(at(0.05, 0), 201);
        assert_eq!(at(0.10, 0), 51);
        assert_eq!(at(0.01, 0), 5_001);
        assert_eq!(at(0.5, 0), 3);
        assert_eq!(at(0.5, 10), 10, "explicit floor wins when larger");
        assert_eq!(at(0.0, 10), u32::MAX, "non-positive ε never stops");
        assert_eq!(at(f64::NAN, 10), u32::MAX);
        assert_eq!(at(1e-9, 0), u32::MAX, "saturates instead of truncating");
        assert_eq!(AdaptiveBootstrapBudget::disabled().required_replicates(), u32::MAX);
    }

    #[test]
    fn f64_draws_are_in_unit_interval() {
        let mut rng = CausalRng::from_seed(123);
        for _ in 0..1000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn simultaneous_studies_compete_for_a_fixed_admission_budget() {
        let shared = AdmissionPool::fixed(4, 500);
        let study_a = ExecutionContext::for_tests(1).with_admission(shared.clone());
        let study_b = ExecutionContext::for_tests(2).with_admission(shared.clone());
        assert!(study_a.memory.hard_limit_bytes.is_none());
        assert_eq!(shared.thread_budget(), 4);
        assert_eq!(shared.byte_budget(), 500);
        let gate = Arc::new(Barrier::new(2));
        let start = |study: ExecutionContext, gate: Arc<Barrier>| {
            thread::spawn(move || {
                gate.wait();
                study.acquire_admission(3, 300)
            })
        };
        let left = start(study_a, Arc::clone(&gate));
        let right = start(study_b, gate);
        let left = left.join().unwrap();
        let right = right.join().unwrap();
        let ((Ok(_), Err(refusal)) | (Err(refusal), Ok(_))) = (&left, &right) else {
            panic!("one study must take the budget and the other must be refused");
        };
        assert!(matches!(
            refusal,
            AdmissionError::ExceedsRemainder {
                requested_threads: 3,
                requested_bytes: 300,
                remaining_threads: 1,
                remaining_bytes: 200,
            }
        ));
        assert_eq!(shared.available().threads(), 1);
        assert_eq!(shared.available().bytes(), 200);
        drop(left);
        drop(right);
        assert_eq!(shared.available().threads(), 4);
        assert_eq!(shared.available().bytes(), 500);
    }

    #[test]
    fn cloned_context_shares_the_admission_decrement() {
        let study = ExecutionContext::for_tests(1).with_admission(AdmissionPool::fixed(2, 10));
        let clone = study.clone();
        let lease = study.acquire_admission(2, 10).unwrap();
        assert!(clone.acquire_admission(1, 0).is_err());
        assert_eq!(clone.admission().available().threads(), 0);
        assert_eq!(lease.release().threads(), 2);
        assert_eq!(clone.admission().available().threads(), 2);
        assert_eq!(clone.admission().available().bytes(), 10);
    }

    #[test]
    fn child_lease_cannot_exceed_parent_remainder() {
        let shared = AdmissionPool::fixed(10, 1_000);
        let study = ExecutionContext::for_tests(7).with_admission(shared.clone());
        let parent = study.acquire_admission(4, 400).unwrap();
        assert_eq!(shared.available().threads(), 6);
        let too_wide = parent.acquire_child(5, 1);
        assert!(matches!(
            too_wide,
            Err(AdmissionError::ExceedsRemainder {
                remaining_threads: 4,
                remaining_bytes: 400,
                ..
            })
        ));
        assert_eq!(shared.available().threads(), 6);
        let child = parent.acquire_child(3, 100).unwrap();
        assert_eq!(parent.remainder().threads(), 1);
        assert_eq!(parent.remainder().bytes(), 300);
        assert_eq!(shared.available().threads(), 6);
        let returned = child.release();
        assert_eq!(returned.threads(), 3);
        assert_eq!(returned.bytes(), 100);
        assert_eq!(parent.remainder().threads(), 4);
        assert_eq!(parent.remainder().bytes(), 400);
        let child = parent.acquire_child(4, 400).unwrap();
        assert_eq!(parent.release().threads(), 0);
        assert_eq!(shared.available().threads(), 6);
        assert_eq!(child.remainder().threads(), 4);
        drop(child);
        assert_eq!(shared.available().threads(), 10);
        assert_eq!(shared.available().bytes(), 1_000);
    }

    #[test]
    fn cancellation_keeps_admission_until_the_owner_releases() {
        let shared = AdmissionPool::fixed(4, 80);
        let study_a = ExecutionContext::for_tests(1).with_admission(shared.clone());
        let study_b = ExecutionContext::for_tests(2).with_admission(shared.clone());
        let lease = study_a.acquire_admission(4, 80).unwrap();
        study_a.cancellation.cancel();
        assert_eq!(shared.available(), AdmissionRemainder::zero());
        assert!(study_b.acquire_admission(4, 80).is_err());
        assert_eq!(lease.release(), AdmissionRemainder::new(4, 80));
        let taken = study_b.acquire_admission(4, 80).unwrap();
        assert_eq!(shared.available().threads(), 0);
        drop(taken);
        assert_eq!(shared.available().threads(), 4);
        study_b.cancellation.cancel();
        let refused = study_b.acquire_admission_bounded(1, 1, Duration::from_secs(30));
        assert!(matches!(refused, Err(AdmissionError::Cancelled)));
    }

    #[test]
    fn byte_overflow_is_refused_instead_of_wrapped() {
        let shared = AdmissionPool::fixed(2, 32);
        let study = ExecutionContext::for_tests(1).with_admission(shared.clone());
        assert!(checked_byte_product(u64::MAX, 2).is_none());
        assert!(checked_byte_sum(u64::MAX, 1).is_none());
        assert_eq!(u64::MAX.wrapping_add(1), 0);
        assert!(matches!(
            AdmissionRequest::try_sum(1, &[u64::MAX, 1]),
            Err(AdmissionError::Overflow)
        ));
        assert!(matches!(
            study.acquire_admission_product(1, u64::MAX, 2),
            Err(AdmissionError::Overflow)
        ));
        assert_eq!(shared.available().threads(), 2);
        assert_eq!(shared.available().bytes(), 32);
        let parent = study.acquire_admission(1, 32).unwrap();
        assert!(matches!(
            parent.acquire_child_product(1, u64::MAX, 2),
            Err(AdmissionError::Overflow)
        ));
        assert_eq!(parent.remainder().bytes(), 32);
        assert_eq!(shared.available().bytes(), 0);
        drop(parent);
        assert_eq!(shared.available().bytes(), 32);
    }

    #[test]
    fn bounded_wait_ends_on_release_and_on_the_bound() {
        let shared = AdmissionPool::fixed(2, 40);
        let study_a = ExecutionContext::for_tests(1).with_admission(shared.clone());
        let study_b = ExecutionContext::for_tests(2).with_admission(shared.clone());
        let lease = study_a.acquire_admission(2, 40).unwrap();
        let early = study_b.acquire_admission_bounded(2, 40, Duration::from_millis(20));
        assert!(matches!(early, Err(AdmissionError::WaitBoundExceeded { .. })));
        assert_eq!(shared.available().threads(), 0);
        let waiting =
            thread::spawn(move || study_b.acquire_admission_bounded(2, 40, Duration::from_secs(5)));
        let started = Instant::now();
        while shared.blocked_waiters() == 0 {
            assert!(started.elapsed() < Duration::from_secs(2), "waiter did not block");
            thread::yield_now();
        }
        assert_eq!(lease.release().threads(), 2);
        let got = waiting.join().unwrap().unwrap();
        assert_eq!(got.remainder().threads(), 2);
        assert_eq!(shared.available().threads(), 0);
        drop(got);
        assert_eq!(shared.available().threads(), 2);
        assert_eq!(shared.available().bytes(), 40);
    }

    fn parallel_context(pool: &AdmissionPool, seed: u64) -> ExecutionContext {
        let mut ctx = ExecutionContext::production(seed, 2).with_admission(pool.clone());
        ctx.memory.hard_limit_bytes = Some(pool.byte_budget());
        ctx
    }

    struct HoldState {
        entered: usize,
        release: bool,
    }

    struct Hold {
        state: Mutex<HoldState>,
        cv: Condvar,
    }

    impl Hold {
        const fn new() -> Self {
            Self { state: Mutex::new(HoldState { entered: 0, release: false }), cv: Condvar::new() }
        }

        fn enter(&self) {
            let mut state = lock_mutex(&self.state);
            state.entered = state.entered.saturating_add(1);
            self.cv.notify_all();
        }

        fn entered(&self) -> usize {
            lock_mutex(&self.state).entered
        }

        fn wait_until_released(&self) {
            let mut state = lock_mutex(&self.state);
            while !state.release {
                state = self.cv.wait(state).unwrap_or_else(PoisonError::into_inner);
            }
        }

        fn release(&self) {
            lock_mutex(&self.state).release = true;
            self.cv.notify_all();
        }
    }

    struct Release(Arc<Hold>);

    impl Drop for Release {
        fn drop(&mut self) {
            self.0.release();
        }
    }

    #[test]
    fn concurrent_map_indexed_runs_stay_within_the_pool() {
        let shared = AdmissionPool::fixed(2, 64);
        let hold = Arc::new(Hold::new());
        let inflight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let start = |seed: u64| {
            let ctx = parallel_context(&shared, seed);
            let hold = Arc::clone(&hold);
            let inflight = Arc::clone(&inflight);
            let peak = Arc::clone(&peak);
            thread::spawn(move || {
                ctx.map_indexed(2, move |index, inner| {
                    assert_eq!(inner.parallelism.max_threads.get(), 1);
                    let now = inflight.fetch_add(1, Ordering::SeqCst).saturating_add(1);
                    peak.fetch_max(now, Ordering::SeqCst);
                    hold.enter();
                    hold.wait_until_released();
                    inflight.fetch_sub(1, Ordering::SeqCst);
                    Ok::<_, ()>(index)
                })
            })
        };
        let left = start(1);
        let right = start(2);
        let _release = Release(Arc::clone(&hold));
        let started = Instant::now();
        loop {
            let held = shared.available().threads() == 0;
            let waiting = shared.blocked_waiters() >= 1;
            if hold.entered() == 2 && held && waiting {
                break;
            }
            assert!(
                started.elapsed() < Duration::from_secs(2),
                "second map_indexed did not wait for the admission lease"
            );
            thread::yield_now();
        }
        assert!(peak.load(Ordering::SeqCst) <= 2);
        hold.release();
        let left = left.join().unwrap().unwrap();
        let right = right.join().unwrap().unwrap();
        assert_eq!(left, vec![0, 1]);
        assert_eq!(right, vec![0, 1]);
        assert!(peak.load(Ordering::SeqCst) <= 2);
        assert_eq!(inflight.load(Ordering::SeqCst), 0);
        assert_eq!(shared.available(), AdmissionRemainder::new(2, 64));
        assert_eq!(shared.blocked_waiters(), 0);
    }

    #[test]
    fn serial_admitted_maps_share_capacity_and_nested_work_reuses_the_slot() {
        let pool = AdmissionPool::fixed(1, 64);
        let first = ExecutionContext::for_tests(1).with_admission(pool.clone());
        let second = ExecutionContext::for_tests(2).with_admission(pool.clone());
        let entered = Arc::new(Barrier::new(2));
        let finish = Arc::new(Barrier::new(2));
        let worker_entered = Arc::clone(&entered);
        let worker_finish = Arc::clone(&finish);
        let a = thread::spawn(move || {
            first.try_map_indexed(1, |_, inner| {
                assert_eq!(inner.try_map_indexed(2, |i, _| Ok::<_, ()>(i)).unwrap(), vec![0, 1]);
                worker_entered.wait();
                worker_finish.wait();
                Ok::<_, ()>(1)
            })
        });
        entered.wait();
        let b = thread::spawn(move || second.try_map_indexed(1, |_, _| Ok::<_, ()>(2)));
        let deadline = Instant::now() + Duration::from_secs(5);
        while pool.blocked_waiters() == 0 {
            assert!(Instant::now() < deadline, "second serial map must wait for shared admission");
            thread::yield_now();
        }
        assert_eq!(pool.available().threads(), 0);
        finish.wait();
        assert_eq!(a.join().unwrap().unwrap(), vec![1]);
        assert_eq!(b.join().unwrap().unwrap(), vec![2]);
        assert_eq!(pool.available(), AdmissionRemainder::new(1, 64));
    }

    #[test]
    fn admitted_map_refuses_cancellation_before_constructing_scratch() {
        let ctx = ExecutionContext::for_tests(1);
        ctx.cancellation.cancel();
        let result = ctx.try_map_indexed_with_scratch(
            1,
            || panic!("cancelled work must not allocate scratch"),
            |_, _, (): &mut ()| Ok::<_, ()>(()),
        );
        assert!(matches!(result, Err(MapExecutionError::Admission(AdmissionError::Cancelled))));
    }

    #[test]
    fn cancelled_map_indexed_leaves_no_lease() {
        let shared = AdmissionPool::fixed(2, 64);
        let ctx = parallel_context(&shared, 1);
        let entered = Arc::new(Barrier::new(3));
        let finish = Arc::new(Barrier::new(3));
        let running = ctx.clone();
        let entered_worker = Arc::clone(&entered);
        let finish_worker = Arc::clone(&finish);
        let handle = thread::spawn(move || {
            running.map_indexed(2, move |index, inner| {
                let _reservation = inner.acquire_admission(0, 32).unwrap();
                let scratch = vec![0_u8; 32];
                entered_worker.wait();
                finish_worker.wait();
                std::hint::black_box(scratch);
                Ok::<_, ()>(index)
            })
        });
        entered.wait();
        assert_eq!(shared.available().threads(), 0);
        assert_eq!(shared.available().bytes(), 0);
        ctx.cancellation.cancel();
        assert_eq!(shared.available(), AdmissionRemainder::zero());
        let competing = ExecutionContext::for_tests(2).with_admission(shared.clone());
        assert!(competing.acquire_admission(1, 1).is_err());
        finish.wait();
        assert_eq!(handle.join().unwrap().unwrap(), vec![0, 1]);
        assert_eq!(shared.available(), AdmissionRemainder::new(2, 64));
        let again = ctx.map_indexed(2, |index, _inner| Ok::<_, ()>(index)).unwrap();
        assert_eq!(again, vec![0, 1]);
        assert_eq!(shared.available(), AdmissionRemainder::new(2, 64));
        assert_eq!(shared.blocked_waiters(), 0);
    }

    #[test]
    fn failed_map_indexed_returns_the_lease() {
        let shared = AdmissionPool::fixed(2, 32);
        let ctx = parallel_context(&shared, 1);
        let saw_hold = AtomicBool::new(false);
        let result = ctx.map_indexed(4, |index, _inner| {
            if shared.available() == AdmissionRemainder::new(0, 32) {
                saw_hold.store(true, Ordering::SeqCst);
            }
            if index == 0 { Err(index) } else { Ok(index) }
        });
        assert_eq!(result, Err(0));
        assert!(saw_hold.load(Ordering::SeqCst));
        assert_eq!(shared.available(), AdmissionRemainder::new(2, 32));
    }

    #[test]
    fn nested_map_indexed_does_not_wait_on_the_outer_lease() {
        let shared = AdmissionPool::fixed(2, 32);
        let ctx = parallel_context(&shared, 1);
        let output = ctx
            .map_indexed(2, |index, inner| {
                let nested =
                    inner.map_indexed(2, |inner_index, _inner| Ok::<_, ()>(inner_index))?;
                Ok::<_, ()>((index, nested))
            })
            .unwrap();
        assert_eq!(output, vec![(0, vec![0, 1]), (1, vec![0, 1])]);
        assert_eq!(shared.available(), AdmissionRemainder::new(2, 32));
    }

    #[test]
    fn map_workers_stay_within_the_pool() {
        let shared = AdmissionPool::fixed(2, 1_000);
        let mut ctx = ExecutionContext::production(1, 8).with_admission(shared.clone());
        ctx.memory = MemoryBudget { soft_limit_bytes: Some(32), hard_limit_bytes: Some(64) };
        let barrier = Arc::new(Barrier::new(2));
        let inflight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let out = ctx
            .map_indexed(8, |index, inner| {
                assert_eq!(inner.parallelism.max_threads.get(), 1);
                let now = inflight.fetch_add(1, Ordering::SeqCst).saturating_add(1);
                peak.fetch_max(now, Ordering::SeqCst);
                assert_eq!(shared.available().threads(), 0);
                let scratch = inner.acquire_admission(0, 32).unwrap();
                barrier.wait();
                assert_eq!(shared.available().bytes(), 936);
                barrier.wait();
                drop(scratch);
                barrier.wait();
                inflight.fetch_sub(1, Ordering::SeqCst);
                Ok::<_, ()>(index)
            })
            .unwrap();
        assert_eq!(out, (0..8).collect::<Vec<_>>());
        assert_eq!(peak.load(Ordering::SeqCst), 2);
        assert_eq!(inflight.load(Ordering::SeqCst), 0);
        assert_eq!(ctx.memory.soft_limit_bytes, Some(32));
        assert_eq!(ctx.memory.hard_limit_bytes, Some(64));
        assert_eq!(shared.available(), AdmissionRemainder::new(2, 1_000));
    }

    #[test]
    fn explicit_byte_account_refuses_before_the_pool_is_charged() {
        let shared = AdmissionPool::fixed(4, 100);
        let mut study = ExecutionContext::for_tests(1).with_admission(shared.clone());
        study.memory = MemoryBudget { soft_limit_bytes: Some(10), hard_limit_bytes: Some(100) };
        let refused = study.acquire_admission(1, 101).unwrap_err();
        assert!(matches!(
            refused,
            AdmissionError::ExceedsRemainder {
                requested_threads: 1,
                requested_bytes: 101,
                remaining_threads: 4,
                remaining_bytes: 100,
            }
        ));
        let product = study.acquire_admission_product(1, 50, 3).unwrap_err();
        assert!(matches!(
            product,
            AdmissionError::ExceedsRemainder { requested_bytes: 150, remaining_bytes: 100, .. }
        ));
        assert_eq!(shared.available(), AdmissionRemainder::new(4, 100));
        assert_eq!(study.memory.soft_limit_bytes, Some(10));
        assert_eq!(study.memory.hard_limit_bytes, Some(100));
        let granted = study.acquire_admission(1, 100).unwrap();
        assert_eq!(granted.remainder().bytes(), 100);
        assert_eq!(shared.available().bytes(), 0);
        drop(granted);
        assert_eq!(shared.available().bytes(), 100);
        assert_eq!(study.memory.hard_limit_bytes, Some(100));
    }

    #[test]
    fn pool_refuses_a_request_it_cannot_fund() {
        let shared = AdmissionPool::fixed(2, 40);
        let study = ExecutionContext::for_tests(1).with_admission(shared.clone());
        assert!(study.memory.hard_limit_bytes.is_none());
        let threads = study.acquire_admission(3, 10).unwrap_err();
        assert!(matches!(
            threads,
            AdmissionError::ExceedsRemainder {
                requested_threads: 3,
                requested_bytes: 10,
                remaining_threads: 2,
                remaining_bytes: 40,
            }
        ));
        let bytes = study.acquire_admission(1, 41).unwrap_err();
        assert!(matches!(
            bytes,
            AdmissionError::ExceedsRemainder {
                requested_threads: 1,
                requested_bytes: 41,
                remaining_threads: 2,
                remaining_bytes: 40,
            }
        ));
        assert_eq!(shared.available(), AdmissionRemainder::new(2, 40));
        assert_eq!(shared.blocked_waiters(), 0);
        assert!(study.memory.hard_limit_bytes.is_none());
    }

    #[test]
    fn wrapped_byte_product_is_not_admitted() {
        let count = (u64::MAX / 2) + 1;
        assert_eq!(count.wrapping_mul(2), 0);
        let shared = AdmissionPool::fixed(2, 32);
        let study = ExecutionContext::for_tests(1).with_admission(shared.clone());
        assert!(matches!(
            study.acquire_admission_product(1, count, 2),
            Err(AdmissionError::Overflow)
        ));
        assert!(matches!(
            AdmissionRequest::try_product(1, count, 2),
            Err(AdmissionError::Overflow)
        ));
        assert_eq!(shared.available(), AdmissionRemainder::new(2, 32));
        assert_eq!(study.memory, MemoryBudget::unlimited());
    }

    #[test]
    fn concurrent_child_scratch_cannot_exceed_the_parent_lease() {
        let shared = AdmissionPool::fixed(4, 500);
        let study = ExecutionContext::for_tests(1).with_admission(shared.clone());
        let parent = study.acquire_admission(2, 100).unwrap();
        let first = parent.acquire_child(1, 60).unwrap();
        let second = parent.acquire_child(1, 50).unwrap_err();
        assert!(matches!(
            second,
            AdmissionError::ExceedsRemainder {
                requested_threads: 1,
                requested_bytes: 50,
                remaining_threads: 1,
                remaining_bytes: 40,
            }
        ));
        assert_eq!(first.remainder().bytes(), 60);
        assert_eq!(parent.remainder().bytes(), 40);
        assert_eq!(shared.available().bytes(), 400);
        drop(first);
        assert_eq!(parent.remainder(), AdmissionRemainder::new(2, 100));
        drop(parent);
        assert_eq!(shared.available(), AdmissionRemainder::new(4, 500));
    }
}
