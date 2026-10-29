//! Execution context: the context type and its policy-carrying constructors.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::admission::{
    AdmissionError, AdmissionLease, AdmissionPool, AdmissionRequest, AdmissionWait, admit,
};
use super::budget::{
    AdaptiveBootstrapBudget, AdaptiveDrawBudget, CachePolicy, Determinism, KernelPolicy,
    NonZeroThreadCount, Parallelism, default_user_threads,
};
use super::cancel::{CancellationToken, ProgressSink};
use super::rng::RngFactory;
use std::sync::Arc;
use std::time::Duration;

/// Full execution context passed into algorithms.
#[derive(Clone)]
pub struct ExecutionContext {
    /// Parallelism budget.
    pub parallelism: Parallelism,
    /// Determinism policy.
    pub determinism: Determinism,
    /// RNG factory (no global RNG).
    pub rng: RngFactory,
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
    /// Identity of the lowered program whose dispatch this context runs inside.
    /// Clones carry it into workers. Private, so a context cannot be rebuilt field
    /// by field outside this crate and silently lose it.
    pub(super) lowered_scope: Option<crate::identity::SemanticDigest>,
    /// Bootstrap replicates a multiple-imputation run asks of an estimator that publishes no
    /// analytic variance, inside each completed table. Private, set by
    /// [`Self::with_imputation_bootstrap`]; every constructor leaves it unset.
    pub(super) imputation_bootstrap: Option<u32>,
    /// Most observed treatment ranges a pulse's active level may lie beyond the observed
    /// treatment interval. Private, set by [`Self::with_pulse_extrapolation_ranges`]; every
    /// constructor leaves it unset, which reads as [`Self::DEFAULT_PULSE_EXTRAPOLATION_RANGES`].
    pub(super) pulse_extrapolation_ranges: Option<f64>,
}

impl ExecutionContext {
    /// This context inside the dispatch of the lowered program `program`.
    ///
    /// Compile, prepare and one-shot run refuse a context that carries this scope:
    /// work inside a lowered dispatch must execute its retained program and must not
    /// build and run another public study.
    #[doc(hidden)]
    #[must_use]
    pub fn enter_lowered_dispatch(&self, program: crate::identity::SemanticDigest) -> Self {
        let mut scoped = self.clone();
        scoped.lowered_scope = Some(program);
        scoped
    }

    /// This context asking an estimator with no analytic variance for `replicates` bootstrap
    /// replicates inside each completed table of a multiple-imputation run; `0` clears it.
    #[must_use]
    pub fn with_imputation_bootstrap(&self, replicates: u32) -> Self {
        let mut scoped = self.clone();
        scoped.imputation_bootstrap = (replicates > 0).then_some(replicates);
        scoped
    }

    /// The pulse extrapolation bound a context carries unless one is set: the active level of a
    /// pulse may lie at most this many observed treatment ranges beyond the observed interval.
    pub const DEFAULT_PULSE_EXTRAPOLATION_RANGES: f64 = 10.0;

    /// This context bounding how far beyond the observed treatment interval a pulse's active
    /// level may lie, in observed treatment ranges (`max - min`); a level further out is refused
    /// rather than reached by linear extrapolation of the fitted dose response.
    ///
    /// `None` when `ranges` is negative or not finite.
    #[must_use]
    pub fn with_pulse_extrapolation_ranges(&self, ranges: f64) -> Option<Self> {
        if !(ranges.is_finite() && ranges >= 0.0) {
            return None;
        }
        let mut scoped = self.clone();
        scoped.pulse_extrapolation_ranges = Some(ranges);
        Some(scoped)
    }

    /// The pulse extrapolation bound in observed treatment ranges: the declared one, else
    /// [`Self::DEFAULT_PULSE_EXTRAPOLATION_RANGES`].
    #[must_use]
    pub fn pulse_extrapolation_ranges(&self) -> f64 {
        self.pulse_extrapolation_ranges.unwrap_or(Self::DEFAULT_PULSE_EXTRAPOLATION_RANGES)
    }

    /// The bootstrap replicates a multiple-imputation run asked for, if it did.
    #[must_use]
    pub const fn imputation_bootstrap(&self) -> Option<u32> {
        self.imputation_bootstrap
    }

    /// The lowered program whose dispatch this context runs inside, if any.
    #[must_use]
    pub const fn lowered_dispatch(&self) -> Option<crate::identity::SemanticDigest> {
        self.lowered_scope
    }

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
            admission: AdmissionPool::fixed(1, u64::MAX),
            cancellation: CancellationToken::new(),
            progress: None,
            kernel_policy: KernelPolicy::scalar_only(),
            cache_policy: CachePolicy::disabled(),
            // Exact-N pins in unit tests; enable explicitly for adaptive MC tests.
            adaptive_bootstrap: AdaptiveBootstrapBudget::disabled(),
            adaptive_draws: AdaptiveDrawBudget::disabled(),
            lowered_scope: None,
            imputation_bootstrap: None,
            pulse_extrapolation_ranges: None,
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
            admission: AdmissionPool::fixed(threads.get(), u64::MAX),
            cancellation: CancellationToken::new(),
            progress: None,
            kernel_policy: KernelPolicy::production_policy(),
            cache_policy: CachePolicy::enabled(None),
            adaptive_bootstrap: AdaptiveBootstrapBudget::disabled(),
            adaptive_draws: AdaptiveDrawBudget::disabled(),
            lowered_scope: None,
            imputation_bootstrap: None,
            pulse_extrapolation_ranges: None,
        }
    }

    /// Install `pool` on this context.
    ///
    /// Later acquires, including [`Self::map_indexed`], decrement `pool`. A clone taken
    /// before this call keeps the pool it already held. The pool's byte capacity becomes
    /// this context's planner limit ([`Self::memory_limit_bytes`]).
    #[must_use]
    pub fn with_admission(mut self, pool: AdmissionPool) -> Self {
        self.admission = pool;
        self
    }

    /// Planner byte limit: the byte capacity of [`Self::admission`], or `None` when that
    /// capacity is `u64::MAX` (unlimited).
    ///
    /// This is the pool's capacity, not what is free. Planners refuse or stream work
    /// whose bytes exceed it, and the same pool refuses a lease that would exceed it.
    #[must_use]
    pub fn memory_limit_bytes(&self) -> Option<u64> {
        let capacity = self.admission.byte_budget();
        (capacity != u64::MAX).then_some(capacity)
    }

    /// Replace this context's pool with a fresh one of the same thread budget and a byte
    /// capacity of `limit` (`None` is unlimited). Clones taken earlier keep their pool. A
    /// limit the pool already has leaves the pool, and everything sharing it, in place.
    #[must_use]
    pub fn with_memory_limit(mut self, limit: Option<u64>) -> Self {
        self.set_memory_limit(limit);
        self
    }

    /// In-place form of [`Self::with_memory_limit`].
    pub fn set_memory_limit(&mut self, limit: Option<u64>) {
        if self.memory_limit_bytes() == limit {
            // Already the capacity: keep the pool, and the leases and clones sharing it.
            return;
        }
        self.admission =
            AdmissionPool::fixed(self.admission.thread_budget(), limit.unwrap_or(u64::MAX));
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
    /// That refusal charges nothing.
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

    /// The one admission entry point: `request` against this context's pool and
    /// cancellation token under `wait`.
    pub(super) fn admit_request(
        &self,
        request: AdmissionRequest,
        wait: AdmissionWait,
    ) -> Result<AdmissionLease, AdmissionError> {
        admit(
            &self.admission.shared,
            request,
            &self.cancellation.cancel_flag(),
            &self.cancellation.flag,
            wait,
        )
    }

    pub(super) fn acquire_request(
        &self,
        request: AdmissionRequest,
        bound: Option<Duration>,
    ) -> Result<AdmissionLease, AdmissionError> {
        let wait = match bound {
            Some(bound) => AdmissionWait::Bounded(bound),
            None => AdmissionWait::Immediate,
        };
        self.admit_request(request, wait)
    }
}

impl core::fmt::Debug for ExecutionContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ExecutionContext")
            .field("parallelism", &self.parallelism)
            .field("determinism", &self.determinism)
            .field("rng", &self.rng)
            .field("memory_limit_bytes", &self.memory_limit_bytes())
            .field("cancellation_cancelled", &self.cancellation.is_cancelled())
            .field("progress_is_some", &self.progress.is_some())
            .field("kernel_policy", &self.kernel_policy)
            .field("cache_policy", &self.cache_policy)
            .field("adaptive_bootstrap", &self.adaptive_bootstrap)
            .field("adaptive_draws", &self.adaptive_draws)
            .field("lowered_scope", &self.lowered_scope)
            .field("imputation_bootstrap", &self.imputation_bootstrap)
            .field("pulse_extrapolation_ranges", &self.pulse_extrapolation_ranges)
            .field("admission", &self.admission())
            .finish()
    }
}
