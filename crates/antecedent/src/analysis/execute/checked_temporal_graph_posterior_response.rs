//! Builder independent execution of a checked temporal graph-posterior response.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::analysis::TemporalPosteriorResponseProof;

impl super::Study {
    /// Execute a retained temporal graph-posterior response, binding the
    /// posterior, per-atom proofs, inference, and validation from the checked
    /// operation rather than from the live builder.
    pub(in crate::analysis) fn execute_checked_temporal_graph_posterior_response(
        &self,
        data: &TimeSeriesData,
        operation: &crate::analysis::CheckedTemporalGraphPosteriorResponse,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let mut bound = self.clone();
        bound.query = CausalQuery::Response(operation.query().clone());
        bound.graph_posterior = Some(operation.posterior().clone());
        bound.tiered = None;
        bound.inference = operation.inference().clone();
        bound.identifier = Some(operation.identifier());
        bound.estimator = Some(operation.estimator());
        bound.refute = operation.validation();
        bound.custom_validators.clear();
        bound.bootstrap_replicates = operation.bootstrap_replicates();
        bound.max_completions = operation.max_completions();
        bound.class_prior = operation.class_prior().cloned();
        bound.temporal_identification_cache = None;
        bound.temporal_class_identification_cache = None;
        match operation.proof() {
            TemporalPosteriorResponseProof::Dbn(cache) => {
                bound.dbn_posterior_identification_cache = Some(Arc::clone(cache));
                bound.temporal_class_posterior_identification_cache = None;
                bound.execute_dbn_posterior_response(
                    data,
                    operation.posterior(),
                    operation.query(),
                    operation.physical(),
                    ctx,
                )
            }
            TemporalPosteriorResponseProof::Class(cache) => {
                bound.dbn_posterior_identification_cache = None;
                bound.temporal_class_posterior_identification_cache = Some(Arc::clone(cache));
                bound.execute_temporal_class_graph_posterior_response(
                    data,
                    operation.posterior(),
                    operation.query(),
                    operation.physical(),
                    ctx,
                )
            }
        }
    }
}
