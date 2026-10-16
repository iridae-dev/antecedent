//! Computation and export budgets for artifacts whose fit does not depend on their size.
//!
//! A fit and the artifact that carries it are separate things: the fit runs on any number of
//! rows the caller's [`ExecutionContext`] can pay for, and the artifact is exported only when
//! its bytes fit the export budget. Each refusal names the quantity that exceeded its bound; the
//! in-memory fit is unaffected by an export refusal.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;

/// A typed refusal: the registered reason code and `detail: explanation`.
pub(crate) type BudgetRefusal = (&'static str, String);

/// The context the size-independent entry points run under: no memory limit, not cancelled.
pub(crate) fn unbounded_context() -> ExecutionContext {
    ExecutionContext::production(0, 1)
}

/// Bytes of `rows` rows of `words_per_row` eight-byte words plus `fixed_words` more, or `None`
/// when the product does not fit.
pub(crate) fn planned_bytes(rows: usize, words_per_row: usize, fixed_words: usize) -> Option<u64> {
    let words = rows.checked_mul(words_per_row)?.checked_add(fixed_words)?;
    u64::try_from(words).ok()?.checked_mul(8)
}

/// Refuse before any work when the context is cancelled or `planned` bytes exceed its hard
/// memory limit.
///
/// `planned` is a conservative planning estimate of the working set, not allocator accounting;
/// `None` (an overflowing estimate) exceeds every limit.
pub(crate) fn admit_computation(
    feature: &str,
    planned: Option<u64>,
    ctx: &ExecutionContext,
) -> Result<(), BudgetRefusal> {
    if ctx.cancellation.is_cancelled() {
        return Err((
            antecedent_core::reason_code!("cancelled_no_claim"),
            format!("{feature}.cancelled: cancelled before the work started"),
        ));
    }
    let planned = planned.unwrap_or(u64::MAX);
    match ctx.memory.hard_limit_bytes {
        Some(limit) if planned > limit => Err((
            antecedent_core::reason_code!("route_not_supported"),
            format!(
                "{feature}.memory_budget_exceeded: the work needs about {planned} bytes, over \
                 the {limit}-byte memory budget of the execution context"
            ),
        )),
        _ => Ok(()),
    }
}

/// Admit the working set of a dense regression of `rows` rows on `columns` design columns: the
/// request's columns, the design built from them, the residuals and a few `columns`-square
/// matrices. A conservative planning estimate, not allocator accounting.
pub(crate) fn admit_regression(
    feature: &str,
    rows: usize,
    columns: usize,
    ctx: &ExecutionContext,
) -> Result<(), BudgetRefusal> {
    let per_row = columns.checked_mul(3).and_then(|words| words.checked_add(4));
    let fixed = columns.checked_mul(columns).and_then(|words| words.checked_mul(4));
    let planned =
        per_row.zip(fixed).and_then(|(per_row, fixed)| planned_bytes(rows, per_row, fixed));
    admit_computation(feature, planned, ctx)
}

/// Refuse an export of `needed` bytes above `budget`.
pub(crate) fn admit_export(
    feature: &str,
    needed: usize,
    budget: usize,
) -> Result<(), BudgetRefusal> {
    if needed > budget {
        return Err((
            antecedent_core::reason_code!("route_not_supported"),
            format!(
                "{feature}.export_budget_exceeded: the artifact needs {needed} bytes, over the \
                 export budget of {budget} bytes; the fit is complete and stays in memory"
            ),
        ));
    }
    Ok(())
}
