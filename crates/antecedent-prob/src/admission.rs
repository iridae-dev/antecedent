// SPDX-License-Identifier: MIT OR Apache-2.0
//! Admission of draw buffers against the workflow's pool.
//!
//! A sampler that returns `n_draws * width` values holds that block in memory for the whole
//! call. This module charges it to the execution account the caller bound, so a request that
//! does not fit is refused before the buffer is allocated.

use antecedent_core::ExecutionContext;
use antecedent_core::execution::AdmissionLease;

use crate::error::ProbError;

/// Charge `cells` `f64` values to the bound execution account.
///
/// `ctx` is the caller's context when the entry point has one; otherwise the workflow
/// binding installed by [`antecedent_core::numerical_admission`] is used. A direct library
/// call with neither has no pool to charge and returns `None`; its allocation belongs to the
/// caller. Hold the returned lease until the buffer is dropped.
///
/// # Errors
///
/// The byte count overflows, the study is cancelled, or the pool cannot fund the buffer.
/// Nothing is allocated.
pub(crate) fn admit_cells(
    ctx: Option<&ExecutionContext>,
    cells: usize,
) -> Result<Option<AdmissionLease>, ProbError> {
    let bytes = cells
        .checked_mul(size_of::<f64>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| refused("draw buffer size overflow".to_owned()))?;
    let outcome = match ctx {
        Some(ctx) => Some(ctx.acquire_admission(0, bytes)),
        None => antecedent_core::numerical_admission::acquire(bytes),
    };
    outcome.transpose().map_err(|error| refused(error.to_string()))
}

fn refused(detail: String) -> ProbError {
    ProbError::Numerical { message: format!("prob.resource_refused: {detail}") }
}
