//! Sealers of the graph-posterior, class-envelope and specialist routes.
//!
//! Their admission lives with the operations (`Study::seal_*` and the
//! `admits`/`seal` constructors); this file binds each to its route.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{SealContext, SealOutcome};
use crate::analysis::execute::Study;
use crate::analysis::prepared::PreparedExecution;

pub(super) fn graph_posterior_effect(cx: &SealContext<'_>) -> SealOutcome {
    Ok(Study::seal_graph_posterior_effect(cx.analysis, cx.plan)?
        .map(PreparedExecution::GraphPosteriorEffect))
}

pub(super) fn class_graph_posterior_effect(cx: &SealContext<'_>) -> SealOutcome {
    Ok(Study::seal_class_graph_posterior_effect(cx.analysis, cx.plan)?
        .map(PreparedExecution::ClassGraphPosteriorEffect))
}

pub(super) fn bayesian_graph_posterior_ate(cx: &SealContext<'_>) -> SealOutcome {
    Ok(Study::seal_bayesian_graph_posterior_ate(cx.analysis, cx.plan)?
        .map(PreparedExecution::BayesianGraphPosteriorAte))
}

pub(super) fn static_class_effect(cx: &SealContext<'_>) -> SealOutcome {
    Ok(Study::seal_static_class_effect(cx.analysis, cx.plan)?
        .map(PreparedExecution::StaticClassEffect))
}

pub(super) fn admg_graph_posterior_response(cx: &SealContext<'_>) -> SealOutcome {
    Ok(Study::seal_admg_graph_posterior_response(cx.analysis, cx.plan)?
        .map(PreparedExecution::AdmgGraphPosteriorResponse))
}

pub(super) fn temporal_class_response(cx: &SealContext<'_>) -> SealOutcome {
    Ok(Study::seal_temporal_class_response(cx.analysis, cx.plan)?
        .map(PreparedExecution::TemporalClassResponse))
}

pub(super) fn temporal_graph_posterior_response(cx: &SealContext<'_>) -> SealOutcome {
    Ok(Study::seal_temporal_graph_posterior_response(cx.analysis, cx.plan)?
        .map(PreparedExecution::TemporalGraphPosteriorResponse))
}

pub(super) fn static_class_response(cx: &SealContext<'_>) -> SealOutcome {
    Ok(Study::seal_static_class_response(cx.analysis, cx.plan)?
        .map(PreparedExecution::StaticClassResponse))
}

pub(super) fn graph_posterior_response(cx: &SealContext<'_>) -> SealOutcome {
    Ok(Study::seal_graph_posterior_response(cx.analysis, cx.plan)?
        .map(PreparedExecution::GraphPosteriorResponse))
}

pub(super) fn bayesian_class_conditional(cx: &SealContext<'_>) -> SealOutcome {
    Ok(crate::analysis::CheckedBayesianClassConditional::seal(cx.analysis, cx.plan)?
        .map(PreparedExecution::BayesianClassConditional))
}

pub(super) fn bayesian_temporal_class_effect(cx: &SealContext<'_>) -> SealOutcome {
    Ok(Study::seal_bayesian_temporal_class_effect(cx.analysis, cx.plan, cx.ctx)?
        .map(PreparedExecution::BayesianTemporalClassEffect))
}

pub(super) fn temporal_graph_posterior_effect(cx: &SealContext<'_>) -> SealOutcome {
    Ok(Study::seal_temporal_graph_posterior_effect(cx.analysis, cx.plan)?
        .map(PreparedExecution::TemporalGraphPosteriorEffect))
}

pub(super) fn bayesian_specialist(cx: &SealContext<'_>) -> SealOutcome {
    Ok(crate::analysis::execute::CheckedBayesianSpecialistOperation::checked(
        cx.analysis,
        cx.plan,
        cx.ctx,
    )?
    .map(PreparedExecution::BayesianSpecialist))
}

pub(super) fn transport_trial(cx: &SealContext<'_>) -> SealOutcome {
    Ok(crate::analysis::execute::CheckedTransportTrialOperation::checked(cx.analysis, cx.plan)?
        .map(PreparedExecution::TransportTrial))
}
