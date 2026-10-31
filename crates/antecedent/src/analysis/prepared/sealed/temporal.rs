//! Sealed series routes: temporal effects, responses and mediation over the
//! retained time-series proofs.
//!
//! Each operation refuses a study whose graph, query or proof no longer matches
//! what preparation retained; the refusal reads the same on estimate and on
//! refresh up to the entry point it names.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{SealedOperation, SeriesCall, SeriesRefusal};
use crate::analysis::execute::{
    CheckedTemporalClassEffectExecution, CheckedTemporalClassMediationOperation,
    CheckedTemporalEffectExecution, CheckedTemporalMediationOperation,
    CheckedTemporalResponseExecution,
};
use crate::analysis::{
    CheckedBayesianTemporalClassEffectOperation, CheckedBayesianTemporalEffectOperation,
    CheckedTemporalClassResponseOperation, CheckedTemporalGraphPosteriorEffect,
    CheckedTemporalGraphPosteriorResponse,
};
use crate::error::CausalError;
use crate::result::StudyResult;
use antecedent_core::CausalQuery;
use std::sync::Arc;

/// Series operations own no click controls and bind no program: the study
/// supplies both at execution.
macro_rules! series_operation_hooks {
    () => {
        unbound_program!();
        study_owned_stage_sink!();
        study_owned_validators!();
    };
}

impl SealedOperation for CheckedTemporalMediationOperation {
    series_operation_hooks!();

    fn estimate_series(&self, call: &SeriesCall<'_>) -> Result<Option<StudyResult>, CausalError> {
        const LOST_GRAPH: SeriesRefusal = SeriesRefusal::new(
            "checked temporal mediation lost its prepared TemporalDag",
            "checked temporal mediation refresh lost its TemporalDag",
        );
        const GRAPH_DIFFERS: SeriesRefusal = SeriesRefusal::new(
            "checked temporal mediation graph differs from the prepared graph",
            "checked temporal mediation refresh graph differs from its retained proof",
        );
        let graph = call.temporal_dag(LOST_GRAPH)?;
        if !self.matches_graph(graph) {
            return Err(call.refusal(GRAPH_DIFFERS));
        }
        self.execute(call.series, call.ctx).map(Some)
    }
}

impl SealedOperation for CheckedTemporalClassMediationOperation {
    series_operation_hooks!();

    fn estimate_series(&self, call: &SeriesCall<'_>) -> Result<Option<StudyResult>, CausalError> {
        self.execute(call.study, call.input, call.ctx).map(Some)
    }
}

impl SealedOperation for CheckedBayesianTemporalEffectOperation {
    series_operation_hooks!();

    fn estimate_series(&self, call: &SeriesCall<'_>) -> Result<Option<StudyResult>, CausalError> {
        const QUERY_CHANGED: SeriesRefusal = SeriesRefusal::new(
            "checked Bayesian temporal query changed after preparation",
            "checked Bayesian temporal refresh query changed after preparation",
        );
        const LOST_GRAPH: SeriesRefusal = SeriesRefusal::new(
            "checked Bayesian temporal effect lost its prepared TemporalDag",
            "checked Bayesian temporal effect refresh lost its TemporalDag",
        );
        const LOST_QUERY: SeriesRefusal = SeriesRefusal::new(
            "checked Bayesian temporal effect lost its temporal query",
            "checked Bayesian temporal effect refresh lost its temporal query",
        );
        if !self.matches_query(&call.study.query) {
            return Err(call.refusal(QUERY_CHANGED));
        }
        let graph = call.temporal_dag(LOST_GRAPH)?;
        let CausalQuery::TemporalEffect(query) = &call.study.query else {
            return Err(call.refusal(LOST_QUERY));
        };
        call.study
            .execute_temporal(call.series, graph, query, &call.handle.plan, call.ctx, Some(self))
            .map(Some)
    }
}

impl SealedOperation for CheckedTemporalEffectExecution {
    series_operation_hooks!();

    fn estimate_series(&self, call: &SeriesCall<'_>) -> Result<Option<StudyResult>, CausalError> {
        const LOST_GRAPH: SeriesRefusal = SeriesRefusal::new(
            "checked temporal effect lost its prepared TemporalDag",
            "checked temporal effect refresh lost its TemporalDag",
        );
        const GRAPH_DIFFERS: SeriesRefusal = SeriesRefusal::new(
            "checked temporal effect graph differs from prepared graph",
            "checked temporal effect refresh graph differs from prepared graph",
        );
        let graph = call.temporal_dag(LOST_GRAPH)?;
        if !self.operation().matches_graph(graph) {
            return Err(call.refusal(GRAPH_DIFFERS));
        }
        self.execute(call.series, call.ctx).map(Some)
    }
}

impl SealedOperation for CheckedTemporalClassEffectExecution {
    unbound_program!();
    study_owned_stage_sink!();

    fn set_custom_validators(
        &mut self,
        validators: &[Arc<dyn antecedent_validate::CustomEffectValidator>],
    ) {
        Self::set_custom_validators(self, validators.to_vec());
    }

    fn estimate_series(&self, call: &SeriesCall<'_>) -> Result<Option<StudyResult>, CausalError> {
        const GRAPH_DIFFERS: SeriesRefusal = SeriesRefusal::new(
            "checked temporal class source graph differs from the retained proof",
            "checked temporal class refresh graph differs from its retained proof",
        );
        if !self.operation().matches_source_graph(&call.study.graph) {
            return Err(call.refusal(GRAPH_DIFFERS));
        }
        self.execute(call.series, call.ctx).map(Some)
    }
}

impl SealedOperation for CheckedTemporalResponseExecution {
    series_operation_hooks!();

    fn estimate_series(&self, call: &SeriesCall<'_>) -> Result<Option<StudyResult>, CausalError> {
        const LOST_GRAPH: SeriesRefusal = SeriesRefusal::new(
            "checked temporal response lost its prepared TemporalDag",
            "checked temporal response refresh lost its TemporalDag",
        );
        const GRAPH_DIFFERS: SeriesRefusal = SeriesRefusal::new(
            "checked temporal response graph differs from prepared graph",
            "checked temporal response refresh graph differs from prepared graph",
        );
        let graph = call.temporal_dag(LOST_GRAPH)?;
        if !self.operation().matches_graph(graph) {
            return Err(call.refusal(GRAPH_DIFFERS));
        }
        self.execute(call.series, call.ctx).map(Some)
    }
}

impl SealedOperation for CheckedTemporalClassResponseOperation {
    series_operation_hooks!();

    fn estimate_series(&self, call: &SeriesCall<'_>) -> Result<Option<StudyResult>, CausalError> {
        const GRAPH_DIFFERS: SeriesRefusal = SeriesRefusal::new(
            "checked temporal class response source graph differs from the retained proof",
            "checked temporal class response refresh graph differs from its retained proof",
        );
        if !self.matches_source_graph(&call.study.graph) {
            return Err(call.refusal(GRAPH_DIFFERS));
        }
        call.study.execute_checked_temporal_class_response(call.series, self, call.ctx).map(Some)
    }
}

impl SealedOperation for CheckedTemporalGraphPosteriorResponse {
    series_operation_hooks!();

    fn estimate_series(&self, call: &SeriesCall<'_>) -> Result<Option<StudyResult>, CausalError> {
        const ATOMS_DIFFER: SeriesRefusal = SeriesRefusal::new(
            "checked temporal graph-posterior response atoms differ from the retained proof",
            "checked temporal graph-posterior response refresh atoms differ from its retained proof",
        );
        if !self.matches_posterior(call.study.graph_posterior.as_ref()) {
            return Err(call.refusal(ATOMS_DIFFER));
        }
        call.study
            .execute_checked_temporal_graph_posterior_response(call.series, self, call.ctx)
            .map(Some)
    }
}

impl SealedOperation for CheckedBayesianTemporalClassEffectOperation {
    series_operation_hooks!();

    fn estimate_series(&self, call: &SeriesCall<'_>) -> Result<Option<StudyResult>, CausalError> {
        const QUERY_CHANGED: SeriesRefusal = SeriesRefusal::new(
            "checked Bayesian temporal class query changed after preparation",
            "checked Bayesian temporal class refresh query changed after preparation",
        );
        const ENVELOPE_DIFFERS: SeriesRefusal = SeriesRefusal::new(
            "checked Bayesian temporal class envelope differs from its retained proof",
            "checked Bayesian temporal class refresh envelope differs from its retained proof",
        );
        if !self.matches_query(&call.study.query) {
            return Err(call.refusal(QUERY_CHANGED));
        }
        if !call.study.temporal_class_cache().is_some_and(|proof| self.matches_class_proof(proof)) {
            return Err(call.refusal(ENVELOPE_DIFFERS));
        }
        call.study
            .execute_checked_bayesian_temporal_class_effect(call.series, self, call.ctx)
            .map(Some)
    }
}

impl SealedOperation for CheckedTemporalGraphPosteriorEffect {
    series_operation_hooks!();

    fn estimate_series(&self, call: &SeriesCall<'_>) -> Result<Option<StudyResult>, CausalError> {
        const QUERY_CHANGED: SeriesRefusal = SeriesRefusal::new(
            "checked temporal graph-posterior query changed after preparation",
            "checked temporal graph-posterior refresh query changed after preparation",
        );
        if !self.matches_query(&call.study.query) {
            return Err(call.refusal(QUERY_CHANGED));
        }
        call.study
            .execute_checked_temporal_graph_posterior_effect(call.series, self, call.ctx)
            .map(Some)
    }
}

impl SealedOperation for crate::analysis::execute::CheckedTemporalAdmgOperation {
    series_operation_hooks!();

    fn require_retained_program(
        &self,
        plan: &crate::planner::PhysicalExecutionPlan,
    ) -> Result<(), CausalError> {
        self.require_program(plan)
    }

    fn estimate_series(&self, call: &SeriesCall<'_>) -> Result<Option<StudyResult>, CausalError> {
        let mut result = self.execute_checked_temporal_admg(
            call.study,
            call.series,
            &call.handle.plan,
            call.ctx,
        )?;
        crate::analysis::execute::finish_family_result(
            call.study,
            call.input,
            &call.handle.plan,
            &mut result,
            call.ctx,
        )?;
        Ok(Some(result))
    }
}
