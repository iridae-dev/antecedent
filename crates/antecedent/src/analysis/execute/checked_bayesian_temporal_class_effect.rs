//! Builder-independent execution of a checked Bayesian temporal class effect.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::analysis::CheckedBayesianTemporalClassEffectOperation;
use crate::analysis::route_guards::series_input;

impl super::Study {
    /// Execute a sealed Bayesian TemporalCpdag/TemporalPag pulse or sustained
    /// effect from its retained completion envelope, model settings, and plan.
    ///
    /// The click study is rebuilt from the operation's fields, never from the
    /// caller's builder: the envelope, unfolding indexers, class mass, and the
    /// validation suite all come from the sealed operation.
    pub(in crate::analysis) fn execute_checked_bayesian_temporal_class_effect(
        &self,
        data: &TimeSeriesData,
        operation: &CheckedBayesianTemporalClassEffectOperation,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let target = operation.target();
        if !target.matches_source_graph(&self.graph) {
            return Err(CausalError::Compile {
                message:
                    "checked Bayesian temporal class source graph differs from the retained proof"
                        .into(),
            });
        }
        let (_, _, bootstrap_replicates, split, _, custom_validators) = target.procedure();
        let mut click = self.clone();
        click.data = series_input(&self.data, data.clone());
        click.query = CausalQuery::TemporalEffect(target.query().clone());
        click.graph_posterior = None;
        click.temporal_class_identification_cache = Some(Arc::new(target.bundle().clone()));
        click.inference = InferenceMode::Bayesian(operation.config().clone());
        click.refute = operation.validation();
        click.class_prior = operation.class_prior().cloned();
        click.max_completions = target.max_completions();
        click.bootstrap_replicates = bootstrap_replicates;
        click.split = split.copied();
        click.custom_validators = custom_validators.to_vec();
        if target.query().is_multi_step_sustained() {
            click.execute_temporal_class_sequential(data, target.query(), operation.physical(), ctx)
        } else {
            click.execute_temporal_class_bayesian(data, target.query(), operation.physical(), ctx)
        }
    }
}
