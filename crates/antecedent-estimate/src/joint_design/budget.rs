//! Memory admission for the joint-design estimators.
//!
//! The estimators retain state that grows with the design (posterior draws by program, the
//! per-unit score matrix, the cohort-by-time grid). That state is leased against the execution
//! context's memory budget instead of being capped by a literal count, so a larger design runs
//! when the budget holds it and is refused, naming the quantity, when it does not.
// SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent_core::ExecutionContext;
use antecedent_core::execution::AdmissionLease;

/// Bytes of one `f64` cell.
const CELL_BYTES: usize = std::mem::size_of::<f64>();

/// Lease `cells` retained `f64` cells (times `copies`) for `what`.
///
/// # Errors
/// A refusal naming `what` when the context's memory budget cannot hold the state, or when the
/// cell count overflows.
pub(crate) fn admit_cells(
    ctx: &ExecutionContext,
    what: &str,
    cells: usize,
    copies: usize,
) -> Result<AdmissionLease, String> {
    let bytes = cells
        .checked_mul(copies)
        .and_then(|total| total.checked_mul(CELL_BYTES))
        .ok_or_else(|| format!("joint design {what} overflows the addressable size"))?;
    ctx.acquire_admission(0, u64::try_from(bytes).unwrap_or(u64::MAX))
        .map_err(|error| format!("joint design {what} exceeds the memory budget: {error}"))
}

/// Deepest schedule or history the recursive g-formula evaluators descend. It bounds the call
/// stack (a few hundred bytes a level), not the model: every other size is leased against the
/// context.
pub(crate) const MAX_RECURSION_DEPTH: usize = 512;

/// Cancellation check for the long per-unit and per-draw loops.
///
/// # Errors
/// `joint-design execution cancelled` when the context is cancelled.
pub(crate) fn check_cancelled(ctx: &ExecutionContext) -> Result<(), String> {
    if ctx.cancellation.is_cancelled() {
        Err("joint-design execution cancelled".into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_state_that_overflows_or_exceeds_the_budget_is_refused_by_name() {
        let ctx = ExecutionContext::for_tests(1);
        assert!(admit_cells(&ctx, "draws", 1_000, 8).is_ok());
        let overflow = admit_cells(&ctx, "draws", usize::MAX, 8).unwrap_err();
        assert!(overflow.contains("draws"), "{overflow}");
    }

    #[test]
    fn a_cancelled_context_stops_the_loops() {
        let ctx = ExecutionContext::for_tests(1);
        assert!(check_cancelled(&ctx).is_ok());
        ctx.cancellation.cancel();
        assert!(check_cancelled(&ctx).is_err());
    }
}
