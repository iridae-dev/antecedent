//! Execution plumbing shared by every route: callbacks, cancellation, budgets, deterministic
//! randomness, recorded execution facts, recalculation plans and admission.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// The one gate a `not_supported` route passes through.
pub use antecedent_core::BUILD_FLAVOR;
/// Execution context: the context type and its policy-carrying constructors.
pub use antecedent_core::RandomizationTestLimits;
/// `antecedent-core` items from `inference_realization.rs`.
pub use antecedent_core::RealizationKind;
/// Batch callbacks for intervention programs.
pub use antecedent_core::{
    BatchCallback, CallbackError, CallbackFailure, CallbackImplementation, CallbackOutput,
    CallbackSpec, PinnedDependency, Portability, RecoveryRecord,
};
/// Physical operator metadata published before R, D, N, Y, and G integration.
pub use antecedent_core::{ByteEstimates, PhysicalEffect, PhysicalOperator, SourceDependencyId};
/// Execution context: parallelism, kernel and cache policy, and Monte Carlo budgets.
pub use antecedent_core::{CacheBudget, CachePolicy, KernelPolicy, default_user_threads};
/// Execution context: cancellation tokens and progress reporting.
pub use antecedent_core::{CancellationToken, ProgressSink, WeakCancellationToken};
/// Execution context: deterministic RNG streams.
pub use antecedent_core::{CausalRng, RngFactory, StreamDomain};
/// Declared capabilities of an inference provider.
pub use antecedent_core::{ProviderCapability, ProviderDeclaration};
/// The bounded-search contract every bounded search runs under.
pub use antecedent_core::{SearchLimits, SearchReceipt, SearchStop};
/// Measured facts for one physical run.
pub mod execution_facts {
    pub use antecedent_core::execution_facts::{
        ExecutionBackend, ExecutionCounts, ExecutionFactError, ExecutionFactStatus, ExecutionFacts,
        ExecutionResources, MeasuredExecution, OutputRetention, PlannedResources,
    };
}
/// Stage model for selective recalculation.
pub mod recalc {
    pub use antecedent_core::recalc::{
        Boundary, Branch, ChangeKind, MissingDependency, RecalcCapabilities, RecalcPlan,
        RefusalReason, RequestSupport, ResumeContext, RetargetSupport, Stage, StageIdentities,
        StageIdentity,
    };
}
/// The one gate a `not_supported` route passes through.
pub mod support_gate {
    pub use antecedent_core::support_gate::{NotSupported, require_supported_unscoped};
}

/// Why the execution context did not admit a request for retained memory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdmissionRefusal {
    /// The study was cancelled before the admission completed.
    Cancelled,
    /// The budget could not hold the request; the text is the engine's explanation.
    Refused(String),
}

impl AdmissionRefusal {
    /// True when the refusal is a cancellation rather than a resource limit.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }
}

impl From<antecedent_core::execution::AdmissionError> for AdmissionRefusal {
    fn from(error: antecedent_core::execution::AdmissionError) -> Self {
        match error {
            antecedent_core::execution::AdmissionError::Cancelled => Self::Cancelled,
            other => Self::Refused(other.to_string()),
        }
    }
}

impl std::fmt::Display for AdmissionRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cancelled => {
                write!(f, "{}", antecedent_core::execution::AdmissionError::Cancelled)
            }
            Self::Refused(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for AdmissionRefusal {}

/// Memory admitted against the execution context's budget; the budget is returned on drop.
#[must_use = "dropping the retention releases its memory budget"]
pub struct Retention {
    lease: std::sync::Arc<antecedent_core::execution::AdmissionLease>,
}

/// Admit `bytes` of retained memory against `ctx`, waiting until the budget has room.
///
/// # Errors
/// [`AdmissionRefusal::Cancelled`] when the study is cancelled; otherwise the budget's refusal.
pub fn retain(
    ctx: &antecedent_core::ExecutionContext,
    bytes: u64,
) -> Result<Retention, AdmissionRefusal> {
    Ok(Retention { lease: std::sync::Arc::new(ctx.acquire_admission(0, bytes)?) })
}

/// Admit `bytes` of retained memory against `ctx`, waiting at most `bound` for room.
///
/// # Errors
/// As [`retain`], plus a refusal when the wait ends before the budget has room.
pub fn retain_bounded(
    ctx: &antecedent_core::ExecutionContext,
    bytes: u64,
    bound: std::time::Duration,
) -> Result<Retention, AdmissionRefusal> {
    Ok(Retention { lease: std::sync::Arc::new(ctx.acquire_admission_bounded(0, bytes, bound)?) })
}

/// Admit `count * width` bytes of retained memory against `ctx`; an overflowing product is refused.
///
/// # Errors
/// As [`retain`], plus a refusal when the product does not fit.
pub fn retain_product(
    ctx: &antecedent_core::ExecutionContext,
    count: u64,
    width: u64,
) -> Result<Retention, AdmissionRefusal> {
    Ok(Retention { lease: std::sync::Arc::new(ctx.acquire_admission_product(0, count, width)?) })
}

/// True when `error` (or any cause beneath it) is an admission refusal from the execution budget.
#[must_use]
pub fn is_admission_refusal(error: &(dyn std::error::Error + 'static)) -> bool {
    error.is::<antecedent_core::execution::AdmissionError>() || error.is::<AdmissionRefusal>()
}

/// A fixed CPU and memory budget a context can be given, with its remaining capacity readable.
#[derive(Clone)]
pub struct FixedBudget(antecedent_core::execution::AdmissionPool);

impl FixedBudget {
    /// A budget of `threads` CPU slots and `bytes` of RAM, all of it free.
    #[must_use]
    pub fn new(threads: u32, bytes: u64) -> Self {
        Self(antecedent_core::execution::AdmissionPool::fixed(threads, bytes))
    }

    /// RAM bytes still free in the budget.
    #[must_use]
    pub fn available_bytes(&self) -> u64 {
        self.0.available().bytes()
    }

    /// `ctx` drawing on this budget; every context given the same budget shares one account.
    #[must_use]
    pub fn apply(
        &self,
        ctx: antecedent_core::ExecutionContext,
    ) -> antecedent_core::ExecutionContext {
        ctx.with_admission(self.0.clone())
    }
}

/// Run `execute` with `ctx`'s admission policy installed for every numerical kernel it enters.
pub fn run_admitted<T>(ctx: &antecedent_core::ExecutionContext, execute: impl FnOnce() -> T) -> T {
    antecedent_core::numerical_admission::with_context(ctx, execute)
}

/// As [`run_admitted`], also returning the first admission refusal a kernel hit, which a
/// component's own error conversion may have erased.
pub fn run_admitted_recording_refusal<T>(
    ctx: &antecedent_core::ExecutionContext,
    execute: impl FnOnce() -> T,
) -> (T, Option<AdmissionRefusal>) {
    let (value, failure) =
        antecedent_core::numerical_admission::with_context_and_failure(ctx, execute);
    (value, failure.map(AdmissionRefusal::from))
}

/// Run `execute` under `ctx`'s admission policy while observing the source work it begins.
///
/// The original result is preserved on failure and cancellation; the observation grants no
/// scientific authority.
pub fn observe<T, E>(
    ctx: &antecedent_core::ExecutionContext,
    execute: impl FnOnce() -> Result<T, E>,
) -> antecedent_core::execution_attempt::ExecutionAttempt<T, E> {
    antecedent_core::execution_attempt::observe_with_context(ctx, execute)
}

/// Observation scope captured on one thread so work dispatched to another thread is counted.
#[derive(Clone)]
pub struct ObservationToken(antecedent_core::execution_attempt::ObservationToken);

impl core::fmt::Debug for ObservationToken {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ObservationToken").finish_non_exhaustive()
    }
}

impl ObservationToken {
    /// Capture the calling thread's observation scope.
    #[must_use]
    pub fn capture() -> Self {
        Self(antecedent_core::execution_attempt::ObservationToken::capture())
    }

    /// Run `execute` inside the captured scope.
    pub fn execute<T>(&self, execute: impl FnOnce() -> T) -> T {
        self.0.execute(execute)
    }
}

/// The artifact-IO error carrying an admission refusal: a cancellation, or a resource refusal
/// (an overflowing request) when `cancelled` is false.
#[must_use]
pub fn artifact_io_admission_error(cancelled: bool) -> antecedent_io_core::IoError {
    use antecedent_core::execution::AdmissionError;
    antecedent_io_core::IoError::Admission(if cancelled {
        AdmissionError::Cancelled
    } else {
        AdmissionError::Overflow
    })
}

impl Retention {
    pub(crate) fn shared_lease(
        &self,
    ) -> std::sync::Arc<antecedent_core::execution::AdmissionLease> {
        std::sync::Arc::clone(&self.lease)
    }
}
