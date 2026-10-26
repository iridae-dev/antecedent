//! Sealed class routes: graph-posterior and class effects and responses,
//! Bayesian specialists, and trial transport.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{SealedOperation, stamp_tabular};
use crate::analysis::builder::DataInput;
use crate::analysis::execute::{
    CheckedBayesianSpecialistOperation, CheckedGraphPosteriorResponse, CheckedStaticClassResponse,
    CheckedTransportTrialOperation,
};
use crate::analysis::prepared::PreparedStudy;
use crate::analysis::prepared::estimate::ClickNotes;
use crate::analysis::{
    CheckedAdmgGraphPosteriorResponse, CheckedBayesianClassConditional,
    CheckedBayesianGraphPosteriorAte, CheckedClassGraphPosteriorEffect,
    CheckedGraphPosteriorEffect, CheckedStaticClassEffect, StaticClassGraph,
    IdentificationCache, StaticClassIdentification,
};
use crate::error::CausalError;
use crate::result::StudyResult;
use antecedent_core::{CausalQuery, ExecutionContext};
use antecedent_data::TabularData;
use std::sync::Arc;

impl SealedOperation for CheckedGraphPosteriorEffect {
    unbound_program!();
    study_owned_stage_sink!();
    study_owned_validators!();

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        let mut click_analysis = handle.analysis.clone();
        click_analysis.data = DataInput::Tabular(data.clone());
        click_analysis.query = self.target().causal_query();
        let result = if matches!(
            self.posterior().atom_kind(),
            antecedent_discovery::GraphPosteriorAtomKind::Admg
                | antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg
        ) {
            click_analysis.execute_checked_admg_graph_posterior(data, &handle.plan, self, ctx)?
        } else {
            click_analysis.execute_checked_graph_posterior_frequentist(
                data,
                &handle.plan,
                self,
                ctx,
            )?
        };
        stamp_tabular(handle, data, result, notes)
    }
}

impl SealedOperation for CheckedClassGraphPosteriorEffect {
    unbound_program!();
    study_owned_stage_sink!();
    study_owned_validators!();

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        let result = handle.analysis.execute_checked_class_graph_posterior_effect(
            data,
            self,
            &handle.plan,
            ctx,
        )?;
        stamp_tabular(handle, data, result, notes)
    }
}

impl SealedOperation for CheckedBayesianGraphPosteriorAte {
    unbound_program!();
    study_owned_stage_sink!();
    study_owned_validators!();

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        let result =
            handle.analysis.execute_checked_bayesian_graph_posterior_ate(data, self, ctx)?;
        stamp_tabular(handle, data, result, notes)
    }
}

impl SealedOperation for CheckedStaticClassEffect {
    unbound_program!();
    study_owned_stage_sink!();
    study_owned_validators!();

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        let mut click_analysis = handle.analysis.clone();
        click_analysis.data = DataInput::Tabular(data.clone());
        click_analysis.query = CausalQuery::AverageEffect(self.query().clone());
        click_analysis.inference = self.inference().clone();
        click_analysis.estimator = Some(self.procedure().id());
        click_analysis.estimator_spec = Some(Arc::new(self.procedure().clone()));
        click_analysis.refute = self.validation();
        click_analysis.bootstrap_replicates = self.bootstrap_replicates();
        click_analysis.overlap_policy = Some(self.overlap());
        let result = match (self.graph(), self.identification()) {
            (StaticClassGraph::Cpdag(graph), StaticClassIdentification::Cpdag(cache)) => {
                click_analysis
                    .install_identification(IdentificationCache::Cpdag(Arc::clone(cache)));
                click_analysis.execute_cpdag(data, graph, self.query(), self.physical(), ctx)?
            }
            (StaticClassGraph::Pag(graph), StaticClassIdentification::Pag(cache)) => {
                click_analysis.install_identification(IdentificationCache::Pag(Arc::clone(cache)));
                click_analysis.execute_pag(data, graph, self.query(), self.physical(), ctx)?
            }
            _ => {
                return Err(CausalError::Compile {
                    message: "checked static class effect graph and identification cache disagree"
                        .into(),
                });
            }
        };
        handle
            .stamp_with_likelihood(data, self.inference(), result, notes.defers_contract())
            .map(Some)
    }
}

impl SealedOperation for CheckedAdmgGraphPosteriorResponse {
    unbound_program!();
    study_owned_stage_sink!();
    study_owned_validators!();

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        let result =
            handle.analysis.execute_checked_admg_graph_posterior_response(data, self, ctx)?;
        stamp_tabular(handle, data, result, notes)
    }
}

impl SealedOperation for CheckedBayesianClassConditional {
    unbound_program!();
    study_owned_stage_sink!();
    study_owned_validators!();

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        let result = self.execute(&handle.analysis, data, ctx)?;
        stamp_tabular(handle, data, result, notes)
    }
}

impl SealedOperation for CheckedStaticClassResponse {
    unbound_program!();
    study_owned_stage_sink!();
    study_owned_validators!();

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        let result = handle.analysis.execute_checked_static_class_response(data, self, ctx)?;
        handle
            .stamp_with_likelihood(data, self.inference(), result, notes.defers_contract())
            .map(Some)
    }
}

impl SealedOperation for CheckedGraphPosteriorResponse {
    unbound_program!();
    study_owned_stage_sink!();
    study_owned_validators!();

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        let result = handle.analysis.execute_checked_graph_posterior_response(data, self, ctx)?;
        handle
            .stamp_with_likelihood(data, self.inference(), result, notes.defers_contract())
            .map(Some)
    }
}

impl SealedOperation for CheckedBayesianSpecialistOperation {
    unbound_program!();
    study_owned_stage_sink!();
    study_owned_validators!();

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        let result = self.execute(&handle.analysis, data, ctx)?;
        stamp_tabular(handle, data, result, notes)
    }
}

impl SealedOperation for CheckedTransportTrialOperation {
    unbound_program!();
    study_owned_stage_sink!();
    study_owned_validators!();

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        let result = self.execute(&handle.analysis, data, ctx)?;
        stamp_tabular(handle, data, result, notes)
    }
}
