//! Builder-independent execution of sealed class and Bayesian graph-posterior
//! effects.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::analysis::{CheckedBayesianGraphPosteriorAte, CheckedClassGraphPosteriorEffect};

impl super::Study {
    /// Execute a sealed CPDAG/PAG graph-posterior effect from its retained
    /// atoms, per-atom envelopes, procedure, and validation suite.
    pub(in crate::analysis) fn execute_checked_class_graph_posterior_effect(
        &self,
        data: &TabularData,
        operation: &CheckedClassGraphPosteriorEffect,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let mut click_analysis = self.clone();
        click_analysis.data = DataInput::Tabular(data.clone());
        click_analysis.query = operation.target().causal_query();
        click_analysis.graph_posterior = Some(operation.posterior().clone());
        click_analysis.graph_posterior_identification_cache =
            Some(Arc::new(operation.identification().clone()));
        click_analysis.inference = operation.inference().clone();
        click_analysis.estimator_spec = Some(operation.procedure().clone());
        click_analysis.estimator = Some(operation.procedure().id());
        click_analysis.refute = operation.validation();
        click_analysis.overlap_policy = Some(operation.overlap());
        click_analysis.population_registry = operation.population_registry().cloned();
        click_analysis.latency_mode = operation.latency_mode();
        click_analysis.execute_class_graph_posterior(
            data,
            operation.posterior(),
            operation.query(),
            physical,
            ctx,
        )
    }

    /// Execute a sealed Bayesian DAG graph-posterior effect from its retained
    /// atoms, per-atom proofs, model settings, and plan.
    pub(in crate::analysis) fn execute_checked_bayesian_graph_posterior_ate(
        &self,
        data: &TabularData,
        operation: &CheckedBayesianGraphPosteriorAte,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let mut click_analysis = self.clone();
        click_analysis.data = DataInput::Tabular(data.clone());
        click_analysis.query = operation.target().causal_query();
        click_analysis.graph_posterior = Some(operation.posterior().clone());
        click_analysis.graph_posterior_identification_cache =
            Some(Arc::new(operation.identification().clone()));
        click_analysis.inference = operation.inference().clone();
        click_analysis.estimator_spec = Some(operation.procedure().clone());
        click_analysis.estimator = Some(operation.procedure().id());
        click_analysis.refute = operation.validation();
        click_analysis.overlap_policy = Some(operation.overlap());
        click_analysis.latency_mode = operation.latency_mode();
        click_analysis.execute_graph_posterior_bayesian(
            data,
            operation.posterior(),
            operation.query(),
            operation.physical(),
            ctx,
        )
    }
}
