//! Execution context: parallelism, kernel and cache policy, and Monte Carlo budgets.
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(doc)]
use super::context::ExecutionContext;
use std::num::NonZeroU32;

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

    /// Environment variable that makes a production context run the scalar reference kernels.
    ///
    /// The performance lanes set it to `1` to measure the current tree's scalar baseline in
    /// the same process shape as the optimized run. Any other value, or no value, leaves the
    /// default policy. The policy in force is recorded with the run as always.
    pub const SCALAR_BASELINE_ENV: &'static str = "ANTECEDENT_BASELINE_SCALAR_KERNELS";

    /// The policy a production context starts with: [`Self::default_policy`], or
    /// [`Self::scalar_only`] when [`Self::SCALAR_BASELINE_ENV`] is `1`.
    #[must_use]
    pub fn production_policy() -> Self {
        if std::env::var(Self::SCALAR_BASELINE_ENV).is_ok_and(|value| value == "1") {
            Self::scalar_only()
        } else {
            Self::default_policy()
        }
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
