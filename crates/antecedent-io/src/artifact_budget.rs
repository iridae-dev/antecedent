//! Computation and export budgets for artifacts whose fit does not depend on their size.
//!
//! A fit and the artifact that carries it are separate things: the fit runs on any number of
//! rows the caller's [`ExecutionContext`] can pay for, and the artifact is exported only when
//! its bytes fit the export budget. Each refusal names the quantity that exceeded its bound; the
//! in-memory fit is unaffected by an export refusal.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;

/// The shape the budgeted treatment artifacts (`categorical_treatment`, `vector_treatment`)
/// share: a fit sealed under the computation budget of an [`ExecutionContext`], exported under
/// a byte budget and consumed by recomputation within a replay budget.
///
/// An implementor supplies the three budgeted operations; the unbudgeted `seal`, `to_bytes` and
/// `from_bytes` follow from them (no memory limit, the format's byte cap).
pub trait BudgetedTreatmentArtifact: Sized {
    /// What is fitted and sealed.
    type Request;
    /// The identity a consumer retains independently of the artifact.
    type Identity;
    /// Why sealing, exporting or consuming was refused.
    type Error;
    /// The format's byte cap, the most an export may occupy.
    const MAX_BYTES: usize;

    /// Fit the request and seal the answer under the computation budget of `ctx`.
    ///
    /// # Errors
    /// The core fit's refusals, the format's bounds and the memory or cancellation refusals.
    fn seal_with(request: &Self::Request, ctx: &ExecutionContext) -> Result<Self, Self::Error>;

    /// Serialize when the whole artifact fits `export_budget` bytes (at most
    /// [`Self::MAX_BYTES`]).
    ///
    /// # Errors
    /// An empty id, an encoding failure or an export budget exceeded.
    fn to_bytes_within(
        &self,
        artifact_id: &str,
        export_budget: usize,
    ) -> Result<Vec<u8>, Self::Error>;

    /// Consume an artifact by recomputation within the replay budget of `ctx`.
    ///
    /// # Errors
    /// Corruption, another major version, a changed identity, a stored result that does not
    /// replay, or a memory or cancellation refusal.
    fn from_bytes_with(
        bytes: &[u8],
        expected: Option<&Self::Identity>,
        ctx: &ExecutionContext,
    ) -> Result<Self, Self::Error>;

    /// [`Self::seal_with`] with no memory limit.
    ///
    /// # Errors
    /// As [`Self::seal_with`].
    fn seal(request: &Self::Request) -> Result<Self, Self::Error> {
        Self::seal_with(request, &unbounded_context())
    }

    /// [`Self::to_bytes_within`] up to the format's cap [`Self::MAX_BYTES`].
    ///
    /// # Errors
    /// As [`Self::to_bytes_within`].
    fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, Self::Error> {
        self.to_bytes_within(artifact_id, Self::MAX_BYTES)
    }

    /// [`Self::from_bytes_with`] with no memory limit.
    ///
    /// # Errors
    /// As [`Self::from_bytes_with`].
    fn from_bytes(bytes: &[u8], expected: Option<&Self::Identity>) -> Result<Self, Self::Error> {
        Self::from_bytes_with(bytes, expected, &unbounded_context())
    }
}

/// The refusal details one artifact reports, spelled out where the artifact is defined.
#[derive(Clone, Copy)]
pub(crate) struct BudgetDetails {
    /// Cancelled before the work started: the registered reason code and the detail.
    pub(crate) cancelled: (&'static str, &'static str),
    /// Planned working set over the memory limit.
    pub(crate) memory: &'static str,
    /// Artifact over the export budget.
    pub(crate) export: &'static str,
}

/// A typed refusal: the registered reason code, the detail identifier and the explanation.
pub(crate) type BudgetRefusal = (&'static str, &'static str, String);

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
    details: BudgetDetails,
    planned: Option<u64>,
    ctx: &ExecutionContext,
) -> Result<(), BudgetRefusal> {
    if ctx.cancellation.is_cancelled() {
        return Err((
            details.cancelled.0,
            details.cancelled.1,
            "cancelled before the work started".to_owned(),
        ));
    }
    let planned = planned.unwrap_or(u64::MAX);
    match ctx.memory_limit_bytes() {
        Some(limit) if planned > limit => Err((
            antecedent_core::reason_code!("route_not_supported"),
            details.memory,
            format!(
                "the work needs about {planned} bytes, over the {limit}-byte memory budget \
                 of the execution context"
            ),
        )),
        _ => Ok(()),
    }
}

/// Admit the working set of a dense regression of `rows` rows on `columns` design columns: the
/// request's columns, the design built from them, the residuals and a few `columns`-square
/// matrices. A conservative planning estimate, not allocator accounting.
pub(crate) fn admit_regression(
    details: BudgetDetails,
    rows: usize,
    columns: usize,
    ctx: &ExecutionContext,
) -> Result<(), BudgetRefusal> {
    let per_row = columns.checked_mul(3).and_then(|words| words.checked_add(4));
    let fixed = columns.checked_mul(columns).and_then(|words| words.checked_mul(4));
    let planned =
        per_row.zip(fixed).and_then(|(per_row, fixed)| planned_bytes(rows, per_row, fixed));
    admit_computation(details, planned, ctx)
}

/// Refuse an export of `needed` bytes above `budget`.
pub(crate) fn admit_export(
    details: BudgetDetails,
    needed: usize,
    budget: usize,
) -> Result<(), BudgetRefusal> {
    if needed > budget {
        return Err((
            antecedent_core::reason_code!("route_not_supported"),
            details.export,
            format!(
                "the artifact needs {needed} bytes, over the export budget of {budget} \
                 bytes; the fit is complete and stays in memory"
            ),
        ));
    }
    Ok(())
}
