//! Quote structural state and work against the execution before a procedure is entered.
//!
//! A miss is a [`SearchReceipt`]. Its summary names the bound, the consumed work, and what
//! was not entered. The procedure is not entered, so the consumed counts are absent.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::execution::{AdmissionError, AdmissionLease, ExecutionContext};
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

/// Bytes leased for one quoted piece of work. Dropping the hold releases them.
///
/// A later [`WorkLease::charge`] of a larger quote leases only the growth, so a subdivision
/// can raise its quote without giving the bytes back in between.
#[must_use = "dropping the hold releases the quote"]
pub struct WorkLease {
    /// Live leases. Their byte sum is the quote charged so far.
    #[allow(dead_code, reason = "retained so drop releases the quote")]
    leases: Vec<AdmissionLease>,
    held: u64,
}

impl WorkLease {
    /// A hold that has leased nothing.
    #[must_use]
    pub fn new() -> Self {
        Self { leases: Vec::new(), held: 0 }
    }

    /// Lease `operations` of work over `state_bytes` of live state.
    ///
    /// The quote is `operations * 16 + state_bytes`. Absent `ctx` is an execution with no
    /// byte limit. Bytes already held by this lease stay held when the quote is refused.
    ///
    /// # Errors
    ///
    /// The receipt of a stop before the additional bytes are leased.
    pub fn charge(
        &mut self,
        ctx: Option<&ExecutionContext>,
        operations: u64,
        state_bytes: u64,
        unevaluated: &str,
    ) -> Result<(), SearchReceipt> {
        let Some(bytes) = operations.checked_mul(16).and_then(|tail| tail.checked_add(state_bytes))
        else {
            return Err(unrepresentable_quote(unevaluated));
        };
        if bytes <= self.held {
            return Ok(());
        }
        let owned;
        let host = match ctx {
            Some(ctx) => ctx,
            None => {
                owned = open_execution();
                &owned
            }
        };
        probe_execution_quote(host, operations.max(1), operations, bytes, unevaluated)?;
        let delta = bytes - self.held;
        let lease = host
            .acquire_admission(0, delta)
            .map_err(|error| admission_receipt(host, unevaluated, error))?;
        self.leases.push(lease);
        self.held = bytes;
        Ok(())
    }
}

impl Default for WorkLease {
    fn default() -> Self {
        Self::new()
    }
}

fn admission_receipt(
    ctx: &ExecutionContext,
    unevaluated: &str,
    error: AdmissionError,
) -> SearchReceipt {
    let stop = match error {
        AdmissionError::Cancelled => SearchStop::Cancelled,
        _ => SearchStop::Memory,
    };
    let memory = ctx.memory_limit_bytes().unwrap_or(u64::MAX);
    match SearchBudget::with_memory(SearchLimits { operations: 1, depth: 1 }, memory, ctx) {
        Ok(budget) => budget.receipt(stop, Vec::new(), vec![unevaluated.to_owned()]),
        Err(mut receipt) => {
            receipt.stop = stop;
            receipt.unevaluated = vec![unevaluated.to_owned()];
            receipt
        }
    }
}
