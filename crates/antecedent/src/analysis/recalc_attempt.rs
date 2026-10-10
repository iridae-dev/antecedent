//! Failure-work observation around the original native recalculation operation.
//! No successful receipt, scientific claim or resumable state is created by this observer.
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_core::execution_attempt::{
    AttemptReport, ExecutionAttempt, Operation, OperationCounts,
};

/// Execute one original native operation once and retain its original result plus work report.
///
/// Pass the existing adjusted, DR/AIPW, static, Bayesian, temporal or design-specific
/// execution call. Their own scientific refusals and transactionality are unchanged.
/// Operation kinds are distinct component levels: a model invocation and its numerical
/// solves must not be summed into a manufactured fit count. Cancellation remains the
/// original error; abandoned component operations are failed, never successful artifacts.
/// Resolved factory observation follows joined workers; unrelated or uninstrumented
/// work remains outside the report. No callback is retried by this wrapper.
pub fn execute_with_attempt<T, E>(
    execute: impl FnOnce() -> Result<T, E>,
) -> ExecutionAttempt<T, E> {
    antecedent_core::execution_attempt::observe_execution(execute)
}
