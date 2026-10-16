//! Shared frozen shape of the graphless randomized-trial operations.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CausalError, ExecutionContext, IdentificationResult, IdentifiedEstimand,
    IdentifiedResultContext, PhysicalExecutionPlan, Study, TableView, TabularData,
};

/// Query-specific facts a [`RandomizedTrialOperation`] needs but cannot derive.
pub(crate) trait RandomizedTrialQuery: std::fmt::Debug + Sized {
    /// Type name reported by `Debug`.
    const OPERATION: &'static str;
    /// Refusal text when a refresh table differs from the prepared one.
    const REFRESH_MISMATCH: &'static str;

    fn describe(&self, rows: usize, debug: &mut std::fmt::DebugStruct<'_, '_>) {
        debug.field("query", self);
        debug.field("rows", &rows);
    }
}

/// Frozen query, prepared table identity, identification result and plan; `E`
/// carries any per-operation study setting that is also fixed at prepare time.
#[derive(Clone)]
pub(crate) struct RandomizedTrialOperation<Q, E = ()> {
    pub(super) query: Q,
    pub(super) schema: antecedent_core::CausalSchema,
    pub(super) rows: usize,
    pub(super) identification: IdentificationResult,
    pub(super) estimand: IdentifiedEstimand,
    pub(super) physical: std::sync::Arc<PhysicalExecutionPlan>,
    pub(super) result_context: IdentifiedResultContext,
    pub(super) extra: E,
}

impl<Q: RandomizedTrialQuery, E> std::fmt::Debug for RandomizedTrialOperation<Q, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct(Q::OPERATION);
        self.query.describe(self.rows, &mut debug);
        debug.finish_non_exhaustive()
    }
}

impl<Q: Clone + RandomizedTrialQuery, E> RandomizedTrialOperation<Q, E> {
    pub(super) fn freeze(
        study: &Study,
        query: &Q,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        identification: IdentificationResult,
        estimand: IdentifiedEstimand,
        extra: E,
    ) -> Self {
        Self {
            query: query.clone(),
            schema: data.schema().clone(),
            rows: data.row_count(),
            identification,
            estimand,
            physical: physical.clone(),
            result_context: IdentifiedResultContext::from_study(study),
            extra,
        }
    }

    /// The refresh gate shared by every operation: same schema and row order,
    /// and not already cancelled.
    pub(super) fn begin_execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        if data.schema() != &self.schema || data.row_count() != self.rows {
            return Err(CausalError::Unsupported { message: Q::REFRESH_MISMATCH });
        }
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: super::super::stage::STAGE_ESTIMATE_POINT,
            });
        }
        Ok(())
    }
}
