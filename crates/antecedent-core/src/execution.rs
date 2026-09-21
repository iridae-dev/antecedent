//! Execution context: parallelism, determinism, RNG, budgets, kernel policy.
//!
//! No core algorithm creates a global thread pool, uses an implicit global RNG,
//! or selects architecture-specific behavior outside [`KernelPolicy`]
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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

    /// [`default_user_threads`] as a [`Parallelism`].
    #[must_use]
    pub fn default_user() -> Self {
        Self {
            max_threads: NonZeroThreadCount::new(default_user_threads())
                .unwrap_or_else(NonZeroThreadCount::one),
        }
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
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
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

/// Closed set of RNG stream domains.
///
/// Indexed families must use [`RngFactory::stream_for`] with a dedicated variant
/// rather than `CONST.wrapping_add(index)` into [`RngFactory::stream`]: additive
/// family offsets collide across domains and, under an additive mixer, across
/// adjacent master seeds.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(u64)]
pub enum StreamDomain {
    /// IID / block / cluster / Bayesian bootstrap replicates.
    Resample = 1,
    /// Lag-aligned temporal block-bootstrap families.
    TemporalBlock = 2,
    /// Bayesian draw / posterior simulation streams.
    Bayesian = 3,
    /// Structure-MCMC chains.
    McmcStructure = 4,
    /// Order-MCMC chains.
    McmcOrder = 5,
    /// DBN posterior MCMC chains.
    McmcDbn = 6,
    /// Graph-completion / identified-set posterior streams.
    Completion = 7,
    /// Counterfactual abduction / action / prediction noise.
    Counterfactual = 8,
    /// Attribution Monte Carlo arms.
    Attribution = 9,
    /// Design ranking and allocation draws.
    Design = 10,
    /// Statistical transport / retargeting draws.
    Transport = 11,
    /// Conditional-independence / CI null Monte Carlo.
    StatsCi = 12,
    /// Learner-internal randomness (forest, GBT, neural).
    Learner = 13,
    /// Estimator-local streams (AIPW, IV, RD, …).
    Estimate = 14,
    /// Temporal / DBN mediation block streams.
    Mediation = 15,
    /// Panel / interference path streams.
    Panel = 16,
    /// Facade execute-path streams.
    Execute = 17,
    /// Unit / integration test fixtures.
    Test = 18,
    /// Example programs.
    Example = 19,
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
        CausalRng::from_seed(mix_seed(self.master_seed, stream_id))
    }

    /// Derive a stream for `(domain, index)`.
    ///
    /// Domain and index are packed without an additive family offset, then passed
    /// through the single [`mix_seed`] mixer with the master seed.
    #[must_use]
    pub fn stream_for(&self, domain: StreamDomain, index: u64) -> CausalRng {
        // Odd multiplier so domain tags stay distinct under xor with index.
        let stream_id = (domain as u64).wrapping_mul(0xD1B5_4A32_D192_ED03) ^ index;
        self.stream(stream_id)
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

/// Non-additive mix of `(master, stream_id)`.
///
/// The previous `master.wrapping_add(stream_id)` form made
/// `(master=s, stream=k)` identical to `(s−δ, k+δ)`, so adjacent user seeds
/// shared almost every bootstrap replicate stream.
fn mix_seed(master: u64, stream_id: u64) -> u64 {
    let mut z = master.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    z ^= stream_id.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    z = (z ^ (z >> 32)).wrapping_mul(0x1656_67B1_9E37_79F9);
    z = (z ^ (z >> 32)).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    z ^ (z >> 32)
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
            cancellation: CancellationToken::new(),
            progress: None,
            kernel_policy: KernelPolicy::default_policy(),
            cache_policy: CachePolicy::enabled(None),
            adaptive_bootstrap: AdaptiveBootstrapBudget::disabled(),
            adaptive_draws: AdaptiveDrawBudget::disabled(),
        }
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
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn mix_seed_rejects_additive_master_stream_alias() {
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
        assert!(!ARCH_SIMD_COMPILED);
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
}
