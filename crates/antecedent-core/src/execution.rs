//! Execution context: parallelism, determinism, RNG, budgets, kernel policy.
//!
//! No core algorithm creates a global thread pool, uses an implicit global RNG,
//! or selects architecture-specific behavior outside [`KernelPolicy`].
//!
//! CPU and RAM admission is a shared lease. Taking a lease decrements the
//! budget; [`MemoryBudget`] is not that lease and is never decremented. A child
//! lease is drawn from its parent's remainder, release returns that remainder,
//! and cancellation releases the lease. Byte products and sums that do not fit
//! in `u64` are refused.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::fmt;
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
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NonZeroThreadCount(u32);

impl NonZeroThreadCount {
    /// Create from a positive thread count.
    #[must_use]
    pub const fn new(n: u32) -> Option<Self> {
        if n == 0 { None } else { Some(Self(n)) }
    }

    /// Single-threaded execution.
    #[must_use]
    pub const fn one() -> Self {
        Self(1)
    }

    /// Underlying count.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

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
        Self { max_threads: NonZeroThreadCount::one() }
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
/// that install the same pool compete for that one remainder. [`MemoryBudget`] is not consulted
/// and is not decremented.
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
    flag: Arc<AtomicBool>,
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
    /// Cancelled leases are released before the remainder is read, so a cancelled study no longer
    /// occupies the budget.
    #[must_use]
    pub fn available(&self) -> AdmissionRemainder {
        let mut guard = lock_mutex(&self.shared.state);
        if reap_cancelled(&mut guard) {
            self.shared.released.notify_all();
        }
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

fn grant_root(
    state: &mut AdmissionState,
    request: AdmissionRequest,
    flag: Arc<AtomicBool>,
) -> Result<u64, GrantError> {
    let Some(next_id) = state.next_id.checked_add(1) else {
        return Err(GrantError::Overflow);
    };
    let (threads_free, bytes_free) = match reserve(state.threads_free, state.bytes_free, request) {
        Ok(reserved) => reserved,
        Err(error) => return Err(GrantError::Short(error)),
    };
    state.threads_free = threads_free;
    state.bytes_free = bytes_free;
    Ok(push_lease(state, None, request, flag, next_id))
}

fn grant_child(
    state: &mut AdmissionState,
    parent_id: u64,
    request: AdmissionRequest,
    flag: Arc<AtomicBool>,
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
    Ok(push_lease(state, Some(parent_id), request, flag, next_id))
}

fn push_lease(
    state: &mut AdmissionState,
    parent: Option<u64>,
    request: AdmissionRequest,
    flag: Arc<AtomicBool>,
    next_id: u64,
) -> u64 {
    let id = state.next_id;
    state.next_id = next_id;
    state.leases.push(LeaseSlot {
        id,
        parent,
        threads: request.threads,
        bytes: request.bytes,
        flag,
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

fn close_slot(state: &mut AdmissionState, index: usize) -> Option<AdmissionRemainder> {
    let id = state.leases[index].id;
    let parent = state.leases[index].parent;
    let threads = state.leases[index].threads;
    let bytes = state.leases[index].bytes;
    credit(state, parent, threads, bytes).ok()?;
    for slot in &mut state.leases {
        if slot.parent == Some(id) {
            slot.parent = parent;
        }
    }
    state.leases.remove(index);
    Some(AdmissionRemainder::new(threads, bytes))
}

fn reap_cancelled(state: &mut AdmissionState) -> bool {
    let mut freed = false;
    loop {
        let Some(index) = state.leases.iter().position(|slot| slot.flag.load(Ordering::SeqCst))
        else {
            return freed;
        };
        if close_slot(state, index).is_none() {
            return freed;
        }
        freed = true;
    }
}

fn close_and_notify(shared: &SharedAdmission, id: u64) -> AdmissionRemainder {
    let mut guard = lock_mutex(&shared.state);
    let Some(index) = lease_index(&guard.leases, id) else {
        return AdmissionRemainder::zero();
    };
    let Some(remainder) = close_slot(&mut guard, index) else {
        return AdmissionRemainder::zero();
    };
    shared.released.notify_all();
    remainder
}

/// A hold on CPU slots and RAM bytes.
///
/// Dropping the lease releases whatever remainder it still holds. Cancellation of the study that
/// took it does the same: the next admission observation returns that remainder to the parent, or
/// to the pool when the lease is a root.
#[must_use = "dropping the lease releases its CPU and RAM"]
pub struct AdmissionLease {
    shared: Arc<SharedAdmission>,
    id: u64,
    flag: Arc<AtomicBool>,
    active: AtomicBool,
}

impl AdmissionLease {
    fn new(shared: Arc<SharedAdmission>, id: u64, flag: Arc<AtomicBool>) -> Self {
        Self { shared, id, flag, active: AtomicBool::new(true) }
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
        if reap_cancelled(&mut guard) {
            self.shared.released.notify_all();
        }
        if self.flag.load(Ordering::SeqCst) {
            self.active.store(false, Ordering::SeqCst);
            return Err(AdmissionError::Cancelled);
        }
        if lease_index(&guard.leases, self.id).is_none() {
            self.active.store(false, Ordering::SeqCst);
            return Err(AdmissionError::Released);
        }
        let id = grant_child(&mut guard, self.id, request, Arc::clone(&self.flag))?;
        Ok(Self::new(Arc::clone(&self.shared), id, Arc::clone(&self.flag)))
    }

    /// CPU and RAM this lease still holds, after cancelled leases are released.
    #[must_use]
    pub fn remainder(&self) -> AdmissionRemainder {
        let mut guard = lock_mutex(&self.shared.state);
        if reap_cancelled(&mut guard) {
            self.shared.released.notify_all();
        }
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

struct CancelFlag {
    cancelled: Arc<AtomicBool>,
    waiters: Mutex<Vec<Weak<Condvar>>>,
    admission: Mutex<Arc<SharedAdmission>>,
}

impl Default for CancelFlag {
    fn default() -> Self {
        Self::from_pool(AdmissionPool::fixed(u32::MAX, u64::MAX))
    }
}

impl CancelFlag {
    fn from_pool(pool: AdmissionPool) -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            waiters: Mutex::new(Vec::new()),
            admission: Mutex::new(pool.shared),
        }
    }

    fn register_waiter(&self, cv: &Arc<Condvar>) {
        let mut waiters = lock_mutex(&self.waiters);
        waiters.retain(|waiter| waiter.strong_count() > 0);
        let incoming = Arc::downgrade(cv);
        if !waiters.iter().any(|waiter| Weak::ptr_eq(waiter, &incoming)) {
            waiters.push(incoming);
        }
    }

    fn wake(&self) {
        let waiters = {
            let mut waiters = lock_mutex(&self.waiters);
            waiters.retain(|waiter| waiter.strong_count() > 0);
            waiters.clone()
        };
        for waiter in waiters {
            if let Some(cv) = waiter.upgrade() {
                cv.notify_all();
            }
        }
    }
}

/// Cooperative cancellation token.
///
/// The token also names the admission pool of the context that holds it. [`Self::cancel`] wakes
/// studies waiting on that pool.
#[derive(Clone, Default)]
pub struct CancellationToken {
    flag: Arc<CancelFlag>,
    /// Test hook: checks left before the token cancels itself.
    trip_after: Option<Arc<AtomicUsize>>,
}

impl fmt::Debug for CancellationToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CancellationToken")
            .field("cancelled", &self.flag.cancelled.load(Ordering::SeqCst))
            .field("trip_after", &self.trip_after)
            .finish()
    }
}

impl CancellationToken {
    /// Create a fresh token with an open CPU and RAM budget.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn with_pool(pool: AdmissionPool) -> Self {
        Self { flag: Arc::new(CancelFlag::from_pool(pool)), trip_after: None }
    }

    /// A token that cancels itself on its `checks + 1`-th observation, so a
    /// test can stop a search deterministically in the middle (every budget
    /// charge observes the token once). Not for production use.
    // doc(hidden): test helper called from integration tests across the workspace.
    #[doc(hidden)]
    #[must_use]
    pub fn cancel_after_checks(checks: usize) -> Self {
        Self {
            flag: Arc::new(CancelFlag::default()),
            trip_after: Some(Arc::new(AtomicUsize::new(checks))),
        }
    }

    /// A non-owning handle that does not keep this token alive.
    #[must_use]
    pub fn downgrade(&self) -> WeakCancellationToken {
        WeakCancellationToken {
            flag: Arc::downgrade(&self.flag),
            trip_after: self.trip_after.clone(),
        }
    }

    /// Request cancellation and wake admission waits bound to this token.
    pub fn cancel(&self) {
        self.flag.cancelled.store(true, Ordering::SeqCst);
        self.flag.wake();
    }

    /// Whether cancellation was requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        if let Some(left) = &self.trip_after {
            if left.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1)).is_err()
            {
                self.cancel();
            }
        }
        self.flag.cancelled.load(Ordering::SeqCst)
    }

    fn set_pool(&self, pool: AdmissionPool) {
        *lock_mutex(&self.flag.admission) = pool.shared;
    }

    fn admission_pool(&self) -> AdmissionPool {
        AdmissionPool { shared: Arc::clone(&lock_mutex(&self.flag.admission)) }
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.flag.cancelled)
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
    trip_after: Option<Arc<AtomicUsize>>,
}

impl WeakCancellationToken {
    /// The token, if any owner of it is still alive.
    #[must_use]
    pub fn upgrade(&self) -> Option<CancellationToken> {
        Some(CancellationToken { flag: self.flag.upgrade()?, trip_after: self.trip_after.clone() })
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

fn admit(
    shared: &Arc<SharedAdmission>,
    request: AdmissionRequest,
    flag: &Arc<AtomicBool>,
    waiters: &CancelFlag,
    bound: Option<Duration>,
) -> Result<AdmissionLease, AdmissionError> {
    waiters.register_waiter(&shared.released);
    let mut guard = lock_mutex(&shared.state);
    let limit = match bound {
        None => None,
        Some(bound) => {
            let deadline = Instant::now().checked_add(bound).ok_or(AdmissionError::Overflow)?;
            Some((deadline, bound))
        }
    };
    loop {
        if reap_cancelled(&mut guard) {
            shared.released.notify_all();
        }
        if flag.load(Ordering::SeqCst) {
            return Err(AdmissionError::Cancelled);
        }
        match grant_root(&mut guard, request, Arc::clone(flag)) {
            Ok(id) => return Ok(AdmissionLease::new(Arc::clone(shared), id, Arc::clone(flag))),
            Err(GrantError::Overflow) => return Err(AdmissionError::Overflow),
            Err(GrantError::Short(refusal)) => {
                guard = wait_or_stop(shared, guard, refusal, limit)?;
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
    mut guard: MutexGuard<'a, AdmissionState>,
    deadline: Instant,
) -> Result<MutexGuard<'a, AdmissionState>, AdmissionError> {
    let Some(waiting) = guard.blocked_waiters.checked_add(1) else {
        return Err(AdmissionError::Overflow);
    };
    guard.blocked_waiters = waiting;
    let timeout = deadline.saturating_duration_since(Instant::now());
    let (mut locked, _) =
        shared.released.wait_timeout(guard, timeout).unwrap_or_else(PoisonError::into_inner);
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
            cancellation: CancellationToken::with_pool(AdmissionPool::fixed(1, u64::MAX)),
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
    /// When `n <= 1` or the context is already serial, `f` receives `self`.
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
        if n == 0 {
            return Ok(Vec::new());
        }
        let threads = (self.parallelism.max_threads.get() as usize).clamp(1, n);
        if threads == 1 {
            return (0..n).map(|i| f(i, self)).collect();
        }
        let inner = self.serial_inner();
        let mut slots: Vec<Option<Result<T, E>>> = (0..n).map(|_| None).collect();
        // Lowest index known to have failed. A serial run stops at the first error,
        // so an item above a known failure can no longer change the result and is
        // skipped; items below it still run, so the returned error is always the
        // lowest-index one regardless of thread count or scheduling.
        let first_failure = AtomicUsize::new(usize::MAX);
        std::thread::scope(|scope| {
            let f = &f;
            let inner = &inner;
            let first_failure = &first_failure;
            let mut rest = slots.as_mut_slice();
            let mut start = 0usize;
            for t in 0..threads {
                let take = rest.len().div_ceil(threads - t);
                let (mine, next) = rest.split_at_mut(take);
                let begin = start;
                scope.spawn(move || {
                    for (k, slot) in mine.iter_mut().enumerate() {
                        let index = begin + k;
                        if first_failure.load(Ordering::Relaxed) < index {
                            break;
                        }
                        let outcome = f(index, inner);
                        if outcome.is_err() {
                            first_failure.fetch_min(index, Ordering::Relaxed);
                        }
                        *slot = Some(outcome);
                    }
                });
                rest = next;
                start += take;
                if rest.is_empty() {
                    break;
                }
            }
        });
        // Unfilled slots exist only above the lowest failure, which `collect` reaches first.
        slots
            .into_iter()
            .map(|slot| slot.expect("every index below the first error was filled"))
            .collect()
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
            NonZeroThreadCount::new(max_threads.max(1)).unwrap_or_else(NonZeroThreadCount::one);
        Self {
            parallelism: Parallelism::bounded(threads),
            determinism: Determinism::Strict,
            rng: RngFactory::from_seed(seed),
            memory: MemoryBudget::unlimited(),
            cancellation: CancellationToken::with_pool(AdmissionPool::fixed(
                threads.get(),
                u64::MAX,
            )),
            progress: None,
            kernel_policy: KernelPolicy::default_policy(),
            cache_policy: CachePolicy::enabled(None),
            adaptive_bootstrap: AdaptiveBootstrapBudget::disabled(),
            adaptive_draws: AdaptiveDrawBudget::disabled(),
        }
    }

    /// Install `pool` for this context and every clone of its cancellation token.
    ///
    /// Later acquires decrement `pool`. [`Self::memory`] is left unchanged.
    #[must_use]
    pub fn with_admission(self, pool: AdmissionPool) -> Self {
        self.cancellation.set_pool(pool);
        self
    }

    /// The pool later acquires decrement.
    #[must_use]
    pub fn admission(&self) -> AdmissionPool {
        self.cancellation.admission_pool()
    }

    /// Take a root lease from [`Self::admission`], decrementing that pool.
    ///
    /// # Errors
    ///
    /// [`AdmissionError::ExceedsRemainder`] when the pool's remainder is short.
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
        admit(
            &self.cancellation.admission_pool().shared,
            request,
            &self.cancellation.cancel_flag(),
            &self.cancellation.flag,
            bound,
        )
    }
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
    fn cancellation_releases_the_admission_lease() {
        let shared = AdmissionPool::fixed(4, 80);
        let study_a = ExecutionContext::for_tests(1).with_admission(shared.clone());
        let study_b = ExecutionContext::for_tests(2).with_admission(shared.clone());
        let lease = study_a.acquire_admission(4, 80).unwrap();
        study_a.cancellation.cancel();
        assert_eq!(shared.available().threads(), 4);
        assert_eq!(shared.available().bytes(), 80);
        assert_eq!(lease.release().threads(), 0);
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
}
