//! Execution context: parallelism, determinism, RNG, budgets, kernel policy.
//!
//! No core algorithm creates a global thread pool, uses an implicit global RNG,
//! or selects architecture-specific behavior outside [`KernelPolicy`].
//!
//! CPU and RAM admission is a shared lease. Taking a lease decrements the
//! budget. The [`AdmissionPool`] byte capacity is also the planner limit that
//! [`ExecutionContext::memory_limit_bytes`] reports; there is no second memory account.
//! [`ExecutionContext::map_indexed`] takes one [`AdmissionPool`] lease for its
//! workers and returns that lease on success, failure, and cancellation. A
//! child lease is drawn from its parent's remainder, release returns that
//! remainder. Cancellation stops new work; live owners retain their reservations
//! until release or drop. Byte products and sums that
//! do not fit in `u64` are refused.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod admission;
mod budget;
mod cancel;
mod context;
mod pipeline;
mod request;
mod rng;
#[cfg(test)]
mod tests;

pub use admission::{
    AdmissionError, AdmissionLease, AdmissionPool, AdmissionRemainder, AdmissionRequest,
    checked_byte_product, checked_byte_sum,
};
pub use budget::{
    ARCH_SIMD_COMPILED, AdaptiveBootstrapBudget, AdaptiveDrawBudget, CacheBudget, CachePolicy,
    DEFAULT_USER_THREAD_CAP, Determinism, KernelPolicy, MonteCarloBudget, MonteCarloError,
    NonZeroThreadCount, Parallelism, default_user_threads,
};
pub use cancel::{CancellationToken, ProgressSink, WeakCancellationToken};
pub use context::ExecutionContext;
pub use pipeline::{MapExecutionError, PipelineReceipt, PipelineStep};
pub use request::{ExecutionReceipt, ExecutionRequestState, RequestIdentity};
pub use rng::{CausalRng, GOLDEN_GAMMA, RngFactory, StreamDomain, derive_stream, mix64, splitmix64};
