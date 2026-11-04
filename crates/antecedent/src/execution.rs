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
/// Execution context: parallelism, determinism, RNG, budgets, kernel policy.
pub mod execution {
    pub use antecedent_core::execution::{AdmissionError, AdmissionLease, AdmissionPool};
}
/// Non-authoritative accounting of operations begun by executing source engines.
pub mod execution_attempt {
    pub use antecedent_core::execution_attempt::{ObservationToken, observe_with_context};
}
/// Measured facts for one physical run.
pub mod execution_facts {
    pub use antecedent_core::execution_facts::{
        ExecutionBackend, ExecutionCounts, ExecutionFactError, ExecutionFactStatus, ExecutionFacts,
        ExecutionResources, MeasuredExecution, OutputRetention, PlannedResources,
    };
}
/// Items of `antecedent-core::numerical_admission`.
pub mod numerical_admission {
    pub use antecedent_core::numerical_admission::{with_context, with_context_and_failure};
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
