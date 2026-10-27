//! Builder independent execution of a checked temporal class response.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{Arc, CausalError, CausalQuery, ExecutionContext, StudyResult, TimeSeriesData};

impl super::Study {
    /// Execute a retained TemporalCpdag/TemporalPag response through the class
    /// envelope combiner, binding every procedure field from the checked
    /// operation rather than from the live builder.
    pub(in crate::analysis) fn execute_checked_temporal_class_response(
        &self,
        data: &TimeSeriesData,
        operation: &crate::analysis::CheckedTemporalClassResponseOperation,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let mut bound = self.clone();
        bound.graph = Arc::new(operation.source_graph().clone());
        bound.graph_posterior = None;
        bound.declared.tiered = None;
        bound.query = CausalQuery::Response(operation.query().clone());
        bound.install_identification(crate::analysis::IdentificationCache::TemporalClass(
            Arc::clone(operation.bundle()),
        ));
        operation.frozen().bind_study(&mut bound);
        bound.declared.class_prior = operation.class_prior().cloned();
        bound.execute_temporal_class_response(data, operation.query(), operation.physical(), ctx)
    }
}
