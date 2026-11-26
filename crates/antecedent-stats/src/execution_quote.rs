//! Quote structural state and work against the execution before a procedure is entered.
//!
//! A miss is a [`SearchReceipt`]. Its summary names the bound, the consumed work, and what
//! was not entered. The procedure is not entered, so the consumed counts are absent.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::execution::{AdmissionLease, ExecutionContext};
use antecedent_core::{SearchBudget, SearchLimits, SearchReceipt, SearchStop};

/// An execution with no declared byte limit. Any representable quote is admitted.
#[must_use]
pub fn open_execution() -> ExecutionContext {
    ExecutionContext::for_tests(0)
}

/// Receipt for a quote that cannot be represented. The procedure was not entered.
#[must_use]
pub fn unrepresentable_quote(unevaluated: &str) -> SearchReceipt {
    SearchReceipt {
        stop: SearchStop::Operations,
        operations_limit: 1,
        depth_limit: 1,
        memory_limit_bytes: Some(0),
        operations_consumed: None,
        depth_reached: None,
        explored: Vec::new(),
        unevaluated: vec![unevaluated.to_owned()],
    }
}

/// Admit `quoted_operations` at one level over `bytes` of live state, and lease those bytes.
///
/// The operation limit is `operation_limit` (at least one). Memory is the context's pool,
/// or unlimited when the pool declares none. Hold the lease for the work.
///
/// # Errors
///
/// The receipt of a stop before the work is entered.
pub fn admit_execution_quote(
    ctx: &ExecutionContext,
    operation_limit: u64,
    quoted_operations: u64,
    bytes: u64,
    unevaluated: &str,
) -> Result<AdmissionLease, SearchReceipt> {
    let memory = ctx.memory_limit_bytes().unwrap_or(u64::MAX);
    let operations = usize::try_from(operation_limit).unwrap_or(usize::MAX).max(1);
    let budget = match SearchBudget::with_memory(SearchLimits { operations, depth: 1 }, memory, ctx)
    {
        Ok(budget) => budget,
        Err(mut receipt) => {
            receipt.unevaluated = vec![unevaluated.to_owned()];
            return Err(receipt);
        }
    };
    if let Err(stop) = budget.admits(1, bytes, quoted_operations) {
        return Err(budget.receipt(stop, Vec::new(), vec![unevaluated.to_owned()]));
    }
    budget
        .lease_storage(bytes)
        .map_err(|stop| budget.receipt(stop, Vec::new(), vec![unevaluated.to_owned()]))
}

/// Admit the quote without leasing. The caller keeps its own leases.
///
/// # Errors
///
/// The receipt of a stop before the work is entered.
pub fn probe_execution_quote(
    ctx: &ExecutionContext,
    operation_limit: u64,
    quoted_operations: u64,
    bytes: u64,
    unevaluated: &str,
) -> Result<(), SearchReceipt> {
    let memory = ctx.memory_limit_bytes().unwrap_or(u64::MAX);
    let operations = usize::try_from(operation_limit).unwrap_or(usize::MAX).max(1);
    let budget = match SearchBudget::with_memory(SearchLimits { operations, depth: 1 }, memory, ctx)
    {
        Ok(budget) => budget,
        Err(mut receipt) => {
            receipt.unevaluated = vec![unevaluated.to_owned()];
            return Err(receipt);
        }
    };
    budget
        .admits(1, bytes, quoted_operations)
        .map_err(|stop| budget.receipt(stop, Vec::new(), vec![unevaluated.to_owned()]))
}

/// Whether the quote would be admitted, without leasing and without treating cancellation as a
/// size miss. Cancellation is `Ok` so a caller that already reports cancellation can enter and
/// report it.
pub fn execution_quote_fits(
    ctx: &ExecutionContext,
    operation_limit: u64,
    quoted_operations: u64,
    bytes: u64,
) -> bool {
    if ctx.cancellation.is_cancelled() {
        return true;
    }
    let memory = ctx.memory_limit_bytes().unwrap_or(u64::MAX);
    let operations = usize::try_from(operation_limit).unwrap_or(usize::MAX).max(1);
    let Ok(budget) = SearchBudget::with_memory(SearchLimits { operations, depth: 1 }, memory, ctx)
    else {
        return false;
    };
    budget.admits(1, bytes, quoted_operations).is_ok()
}
