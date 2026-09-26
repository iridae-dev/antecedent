//! Builder-independent execution of a checked temporal graph-posterior effect.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::analysis::route_guards::series_input;
use crate::analysis::{CheckedTemporalGraphPosteriorEffect, CheckedTemporalGraphPosteriorProof};

impl super::Study {
    /// Execute a sealed pulse or sustained effect over frozen DBN or
    /// TemporalCpdag/TemporalPag posterior atoms from their retained proofs.
    ///
    /// The click study is rebuilt from the operation's fields: the posterior
    /// samples, the per-atom identification, the inference settings, and the
    /// validation suite all come from the sealed operation, and the retained
    /// physical plan drives every atom.
    pub(in crate::analysis) fn execute_checked_temporal_graph_posterior_effect(
        &self,
        data: &TimeSeriesData,
        operation: &CheckedTemporalGraphPosteriorEffect,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if !self
            .graph_posterior
            .as_ref()
            .is_some_and(|posterior| operation.matches_posterior(posterior))
        {
            return Err(CausalError::Compile {
                message: "checked temporal graph-posterior samples differ from the retained proof"
                    .into(),
            });
        }
        let query = operation.query();
        let mut click = self.clone();
        click.data = series_input(&self.data, data.clone());
        click.query = CausalQuery::TemporalEffect(query.clone());
        click.graph_posterior = Some(operation.posterior().clone());
        click.inference = operation.inference().clone();
        click.refute = operation.validation();
        click.bootstrap_replicates = operation.bootstrap_replicates();
        click.split = operation.split().copied();
        click.max_completions = operation.max_completions();
        click.custom_validators = Vec::new();
        click.estimator = Some(operation.estimator());
        click.identifier = Some(operation.identifier());
        match operation.proof() {
            CheckedTemporalGraphPosteriorProof::Dbn(cache) => {
                click.dbn_posterior_identification_cache = Some(Arc::clone(cache));
                click.temporal_class_posterior_identification_cache = None;
                match operation.inference() {
                    InferenceMode::Frequentist => click.execute_dbn_posterior_frequentist(
                        data,
                        operation.posterior(),
                        query,
                        operation.physical(),
                        ctx,
                    ),
                    InferenceMode::Bayesian(_) => click.execute_dbn_posterior_bayesian(
                        data,
                        operation.posterior(),
                        query,
                        operation.physical(),
                        ctx,
                    ),
                }
            }
            CheckedTemporalGraphPosteriorProof::Class(cache) => {
                click.temporal_class_posterior_identification_cache = Some(Arc::clone(cache));
                click.dbn_posterior_identification_cache = None;
                click.execute_temporal_class_graph_posterior(
                    data,
                    operation.posterior(),
                    query,
                    operation.physical(),
                    ctx,
                )
            }
        }
    }
}
