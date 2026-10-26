//! Builder independent execution of a checked temporal graph-posterior response.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{Arc, CausalError, CausalQuery, ExecutionContext, StudyResult, TimeSeriesData};
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
        operation.frozen().bind_study(&mut bound);
        bound.class_prior = operation.class_prior().cloned();
        match operation.proof() {
            TemporalPosteriorResponseProof::Dbn(cache) => {
                bound.install_identification(crate::analysis::IdentificationCache::DbnPosterior(Arc::clone(cache)));
                bound.execute_dbn_posterior_response(
                    data,
                    operation.posterior(),
                    operation.query(),
                    operation.physical(),
                    ctx,
                )
            }
            TemporalPosteriorResponseProof::Class(cache) => {
                bound.install_identification(crate::analysis::IdentificationCache::TemporalClassPosterior(Arc::clone(cache)));
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
