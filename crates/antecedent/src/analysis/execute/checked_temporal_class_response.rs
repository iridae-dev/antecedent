//! Builder independent execution of a checked temporal class response.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

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
        let (identifier, estimator, bootstrap_replicates, validation) = operation.procedure();
        let mut bound = self.clone();
        bound.graph = operation.source_graph().clone();
        bound.graph_posterior = None;
        bound.tiered = None;
        bound.query = CausalQuery::Response(operation.query().clone());
        bound.temporal_identification_cache = None;
        bound.temporal_class_identification_cache = Some(Arc::new(operation.bundle().clone()));
        bound.temporal_class_posterior_identification_cache = None;
        bound.dbn_posterior_identification_cache = None;
        bound.inference = operation.inference().clone();
        bound.identifier = Some(identifier);
        bound.estimator = Some(estimator);
        bound.bootstrap_replicates = bootstrap_replicates;
        bound.refute = validation;
        bound.custom_validators.clear();
        bound.max_completions = operation.max_completions();
        bound.class_prior = operation.class_prior().cloned();
        bound.execute_temporal_class_response(data, operation.query(), operation.physical(), ctx)
    }
}
