//! Execution context: parallelism, determinism, RNG, budgets, kernel policy.
//!
//! No core algorithm creates a global thread pool, uses an implicit global RNG,
//! or selects architecture-specific behavior outside [`KernelPolicy`]
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use core::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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

/// Adaptive bootstrap early-stop budget (SE relative-change criterion).
///
/// After at least [`Self::min_replicates`] successful replicates, stop when
/// `|SE_t − SE_{t−1}| / max(|SE_{t−1}|, ε₀) < se_rel_epsilon`. Cap remains the
/// requested replicate count. Disabled runs always evaluate the full request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveBootstrapBudget {
    /// When false, evaluate all requested replicates (no early-stop).
    pub enabled: bool,
    /// Minimum successful replicates before early-stop is eligible.
    pub min_replicates: u32,
    /// Relative SE change threshold (default `0.01`).
    pub se_rel_epsilon: f64,
}

impl AdaptiveBootstrapBudget {
    /// Enabled defaults: min 10 replicates, 1% relative SE change.
    #[must_use]
    pub const fn enabled_default() -> Self {
        Self { enabled: true, min_replicates: 10, se_rel_epsilon: 0.01 }
    }

    /// Force full requested replicate count (tests / exact-N pins).
    #[must_use]
    pub const fn disabled() -> Self {
        Self { enabled: false, min_replicates: 0, se_rel_epsilon: 0.0 }
    }
}

impl Default for AdaptiveBootstrapBudget {
    fn default() -> Self {
        Self::enabled_default()
    }
}

/// Adaptive Bayesian draw budget (Laplace / conjugate path).
///
/// Cap by quantile-width relative change and/or ESS target under the latency
/// `n_draws` maximum. HMC ignores this and always draws the full request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveDrawBudget {
    /// When false, materialize the full requested draw count.
    pub enabled: bool,
    /// Minimum draws before early-stop is eligible.
    pub min_draws: usize,
    /// Relative 95% quantile-width change threshold (default `0.01`).
    pub quantile_width_rel_epsilon: f64,
    /// Stop when ESS of the effect draws reaches this target (default large).
    pub ess_target: f64,
}

impl AdaptiveDrawBudget {
    /// Enabled defaults: min 32 draws, 1% quantile-width change, high ESS target.
    #[must_use]
    pub const fn enabled_default() -> Self {
        Self {
            enabled: true,
            min_draws: 32,
            quantile_width_rel_epsilon: 0.01,
            ess_target: 10_000.0,
        }
    }

    /// Force full requested draw count.
    #[must_use]
    pub const fn disabled() -> Self {
        Self { enabled: false, min_draws: 0, quantile_width_rel_epsilon: 0.0, ess_target: 0.0 }
    }
}

impl Default for AdaptiveDrawBudget {
    fn default() -> Self {
        Self::enabled_default()
    }
}

/// Cooperative cancellation token.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Create a fresh token.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Whether cancellation was requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

/// Optional progress reporting sink.
pub trait ProgressSink: Send + Sync {
    /// Report progress in `[0.0, 1.0]` with an optional stage label.
    fn report(&self, fraction: f64, stage: &str);
}

/// Factory for deterministic, independently seeded RNG streams.
///
/// Streams are derived from a master seed and a stream id so algorithms can
/// request reproducible substreams without a global RNG.
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

    /// Derive an independent stream for `stream_id`.
    #[must_use]
    pub fn stream(&self, stream_id: u64) -> CausalRng {
        let seed = mix_seed(self.master_seed, stream_id);
        CausalRng::from_seed(seed)
    }
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
        Self { state: seed ^ 0x9E37_79B9_7F4A_7C15 }
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
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Next `f64` in `[0, 1)`.
    #[must_use]
    pub fn next_f64(&mut self) -> f64 {
        // 53-bit mantissa extraction; precision loss vs full u64 is intentional.
        let bits = self.next_u64() >> 11;
        #[allow(clippy::cast_precision_loss)]
        {
            bits as f64 * (1.0 / ((1u64 << 53) as f64))
        }
    }
}

fn mix_seed(master: u64, stream_id: u64) -> u64 {
    let mut z = master.wrapping_add(stream_id).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    z = (z ^ (z >> 32)).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    z ^ (z >> 32)
}

/// Full execution context passed into algorithms.
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
    /// Adaptive bootstrap early-stop (estimate SE path).
    pub adaptive_bootstrap: AdaptiveBootstrapBudget,
    /// Adaptive Bayesian draw early-stop (Laplace / conjugate).
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
            cancellation: CancellationToken::new(),
            progress: None,
            kernel_policy: KernelPolicy::scalar_only(),
            cache_policy: CachePolicy::disabled(),
            // Exact-N pins in unit tests; enable explicitly for adaptive MC tests.
            adaptive_bootstrap: AdaptiveBootstrapBudget::disabled(),
            adaptive_draws: AdaptiveDrawBudget::disabled(),
        }
    }

    /// Production context: optimized kernels allowed, cache enabled, bounded threads.
    #[must_use]
    pub fn production(seed: u64, max_threads: u32) -> Self {
        let threads =
            NonZeroThreadCount::new(max_threads.max(1)).unwrap_or_else(NonZeroThreadCount::one);
        Self {
            parallelism: Parallelism::bounded(threads),
            determinism: Determinism::Strict,
            rng: RngFactory::from_seed(seed),
            memory: MemoryBudget::unlimited(),
            cancellation: CancellationToken::new(),
            progress: None,
            kernel_policy: KernelPolicy::default_policy(),
            cache_policy: CachePolicy::enabled(None),
            adaptive_bootstrap: AdaptiveBootstrapBudget::enabled_default(),
            adaptive_draws: AdaptiveDrawBudget::enabled_default(),
        }
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
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_streams_are_deterministic() {
        let factory = RngFactory::from_seed(42);
        let mut a1 = factory.stream(0);
        let mut a2 = factory.stream(0);
        let mut b = factory.stream(1);
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
        let mut s1 = f1.stream(99);
        let mut s2 = f2.stream(99);
        for _ in 0..32 {
            assert_eq!(s1.next_u64(), s2.next_u64());
        }
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
    fn f64_draws_are_in_unit_interval() {
        let mut rng = CausalRng::from_seed(123);
        for _ in 0..1000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x));
        }
    }
}
