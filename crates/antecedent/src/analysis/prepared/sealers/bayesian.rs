//! Sealers of the Bayesian DAG routes: basis ATE and CATE, robust ATE,
//! g-computation, and the conditional effect.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{SealContext, SealOutcome};
use crate::analysis::builder::RefuteSuite;
use crate::analysis::prepared::{CheckedBayesianGcompOperation, PreparedExecution};
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::strategy_table::EstimatorId;
use antecedent_core::{OutcomeFunctional, TargetPopulation};
use std::sync::Arc;

pub(super) fn bayesian_basis_ate(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, plan, .. } = *cx;
    let sealed =
        match (cx.static_ate_dag(), analysis.identification_cache.as_ref(), &analysis.inference) {
            (Some((data, query)), Some(cache), InferenceMode::Bayesian(_))
                if analysis.graph_posterior.is_none()
                    && analysis.tiered.is_none()
                    && analysis.structure_source == crate::support::StructureSource::Explicit
                    && analysis.refute == RefuteSuite::None
                    && analysis.custom_validators.is_empty()
                    && cx.binding.estimator.is(EstimatorId::BayesianBasisGcomp) =>
            {
                let graph = analysis.graph.as_dag().ok_or(CausalError::Compile {
                    message: "checked Bayesian basis ATE requires its retained DAG".into(),
                })?;
                Some(crate::analysis::execute::CheckedBayesianBasisAteExecution::checked(
                    data,
                    graph,
                    query.clone(),
                    Arc::clone(cache),
                    analysis.inference.clone(),
                    crate::analysis::execute::IdentifiedResultContext::from_study(analysis),
                    plan.clone(),
                )?)
            }
            _ => None,
        };
    Ok(sealed.map(PreparedExecution::BayesianBasisAte))
}

pub(super) fn bayesian_basis_cate(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, plan, .. } = *cx;
    let sealed =
        match (cx.conditional_dag(), analysis.identification_cache.as_ref(), &analysis.inference) {
            (Some((data, query)), Some(cache), InferenceMode::Bayesian(_))
                if analysis.graph_posterior.is_none()
                    && analysis.tiered.is_none()
                    && analysis.structure_source == crate::support::StructureSource::Explicit
                    && analysis.refute == RefuteSuite::None
                    && analysis.custom_validators.is_empty()
                    && cx.binding.estimator.is(EstimatorId::BayesianBasisGcomp) =>
            {
                let graph = analysis.graph.as_dag().ok_or(CausalError::Compile {
                    message: "checked Bayesian basis CATE requires its retained DAG".into(),
                })?;
                Some(crate::analysis::execute::CheckedBayesianBasisCateExecution::checked(
                    data,
                    graph,
                    query.clone(),
                    Arc::clone(cache),
                    analysis.inference.clone(),
                    crate::analysis::execute::IdentifiedResultContext::from_study(analysis),
                    plan.clone(),
                )?)
            }
            _ => None,
        };
    Ok(sealed.map(PreparedExecution::BayesianBasisCate))
}

pub(super) fn bayesian_robust_ate(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, plan, .. } = *cx;
    let sealed =
        match (cx.static_ate_dag(), analysis.identification_cache.as_ref(), &analysis.inference) {
            (Some((data, query)), Some(cache), InferenceMode::Bayesian(_))
                if analysis.graph_posterior.is_none()
                    && analysis.tiered.is_none()
                    && analysis.structure_source == crate::support::StructureSource::Explicit
                    && analysis.refute == RefuteSuite::None
                    && analysis.custom_validators.is_empty()
                    && cx.binding.estimator.is(EstimatorId::BayesianRobustAte) =>
            {
                let graph = analysis.graph.as_dag().ok_or(CausalError::Compile {
                    message: "checked Bayesian robust ATE requires its retained DAG".into(),
                })?;
                Some(crate::analysis::execute::CheckedBayesianRobustAteExecution::checked(
                    data,
                    graph,
                    query.clone(),
                    Arc::clone(cache),
                    analysis.inference.clone(),
                    crate::analysis::execute::IdentifiedResultContext::from_study(analysis),
                    plan.clone(),
                )?)
            }
            _ => None,
        };
    Ok(sealed.map(PreparedExecution::BayesianRobustAte))
}

pub(super) fn bayesian_conditional(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, plan, .. } = *cx;
    let sealed =
        match (cx.conditional_dag(), analysis.identification_cache.as_ref(), &analysis.inference) {
            (Some((_, query)), Some(cache), InferenceMode::Bayesian(_))
                if analysis.graph_posterior.is_none()
                    && analysis.tiered.is_none()
                    && analysis.custom_validators.is_empty()
                    && matches!(
                        analysis.structure_source,
                        crate::support::StructureSource::Explicit
                            | crate::support::StructureSource::Accepted
                    )
                    && matches!(
                        analysis.refute,
                        RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full
                    )
                    && cx.binding.estimator.is(EstimatorId::BayesianConditional) =>
            {
                let graph = analysis.graph.as_dag().ok_or(CausalError::Compile {
                    message: "checked Bayesian conditional effect requires its retained DAG".into(),
                })?;
                Some(crate::analysis::execute::CheckedBayesianConditionalOperation::checked(
                    crate::analysis::execute::CheckedBayesianConditionalOperationPremises {
                        graph,
                        query: query.clone(),
                        cache: Arc::clone(cache),
                        inference: analysis.inference.clone(),
                        refute: analysis.refute,
                        context: crate::analysis::execute::IdentifiedResultContext::from_study(
                            analysis,
                        ),
                        physical: plan.clone(),
                        latency: analysis.latency_mode,
                        validators: analysis.custom_validators.clone(),
                        stage_sink: analysis.stage_sink.clone(),
                    },
                )?)
            }
            _ => None,
        };
    Ok(sealed.map(PreparedExecution::BayesianConditional))
}

pub(super) fn bayesian_gcomp(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, plan, .. } = *cx;
    let operation =
        match (cx.static_ate_dag(), analysis.identification_cache.as_ref(), &analysis.inference) {
            (Some((_, query)), Some(cache), InferenceMode::Bayesian(_))
                if matches!(query.outcome_functional, OutcomeFunctional::Mean)
                    && matches!(
                        query.target_population,
                        TargetPopulation::AllObserved
                            | TargetPopulation::Treated
                            | TargetPopulation::Untreated
                    )
                    && cx.binding.estimator.is(EstimatorId::BayesianGcomp) =>
            {
                CheckedBayesianGcompOperation {
                    query: query.clone(),
                    cache: Arc::clone(cache),
                    inference: analysis.inference.clone(),
                }
            }
            _ => return Ok(None),
        };
    let graph = analysis.graph.as_dag().ok_or(CausalError::Compile {
        message: "checked Bayesian g-computation requires its retained DAG".into(),
    })?;
    Ok(Some(PreparedExecution::BayesianGcomp(
        crate::analysis::execute::CheckedBayesianDagAteExecution::checked(
            crate::analysis::execute::CheckedBayesianDagAteExecutionPremises {
                graph,
                operation,
                context: crate::analysis::execute::IdentifiedResultContext::from_study(analysis),
                physical: plan.clone(),
                validation: analysis.refute,
                latency: analysis.latency_mode,
                validators: analysis.custom_validators.clone(),
                stage_sink: analysis.stage_sink.clone(),
                draw_controls: analysis.draw_controls,
            },
        )?,
    )))
}
