//! Sealers of the static response routes: derivative, cell-AIPW, DAG
//! response, and the legacy mean-curve operation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{SealContext, SealOutcome};
use crate::analysis::builder::RefuteSuite;
use crate::analysis::execute::RouteInput;
use crate::analysis::prepared::{CheckedStaticResponseCurve, PreparedExecution};
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::planner::RouteDefaults;
use crate::strategy_table::{EstimatorId, IdentifierId};
use antecedent_core::{OutcomeFunctional, TargetPopulation};
use std::sync::Arc;

pub(super) fn derivative_response(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, .. } = *cx;
    let sealed = match (cx.static_response_dag(), analysis.identification_cache.as_ref()) {
        (Some((query, graph)), Some(cache))
            if analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && analysis.refute == RefuteSuite::None
                && analysis.custom_validators.is_empty()
                && matches!(
                    analysis.inference,
                    InferenceMode::Frequentist | InferenceMode::Bayesian(_)
                )
                && query.temporal.is_none()
                && query.observation == antecedent_core::ObservationSpec::Complete
                && query.target_population == TargetPopulation::AllObserved
                && matches!(query.outcome_functional, OutcomeFunctional::Mean)
                && matches!(
                    query.functional,
                    antecedent_core::ResponseFunctional::PointDerivative { .. }
                        | antecedent_core::ResponseFunctional::AverageDerivative { .. }
                        | antecedent_core::ResponseFunctional::DirectionalDerivative { .. }
                        | antecedent_core::ResponseFunctional::Jacobian { .. }
                ) =>
        {
            let (identifier, estimator) = cx.binding.resolve(RouteDefaults::RESPONSE)?;
            Some(crate::analysis::execute::CheckedDerivativeResponseOperation::checked(
                graph,
                query,
                cache,
                identifier,
                estimator,
                analysis.inference.clone(),
                analysis.response_options.clone().unwrap_or_default(),
            )?)
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::DerivativeResponse))
}

pub(super) fn static_dag_response(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, .. } = *cx;
    let sealed = match (cx.static_response_dag(), analysis.identification_cache.as_ref()) {
        (Some((query, graph)), Some(cache))
            if analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && matches!(
                    analysis.structure_source,
                    crate::support::StructureSource::Explicit
                        | crate::support::StructureSource::Accepted
                )
                && analysis.custom_validators.is_empty()
                && matches!(
                    analysis.inference,
                    InferenceMode::Frequentist | InferenceMode::Bayesian(_)
                )
                && query.temporal.is_none()
                && query.observation == antecedent_core::ObservationSpec::Complete
                && query.target_population == TargetPopulation::AllObserved
                && matches!(query.outcome_functional, OutcomeFunctional::Mean)
                && (matches!(
                    query.functional,
                    antecedent_core::ResponseFunctional::MeanCurve { .. }
                ) && analysis.refute == RefuteSuite::None
                    || matches!(
                        query.functional,
                        antecedent_core::ResponseFunctional::InterventionResponse { .. }
                    ) && (matches!(analysis.inference, InferenceMode::Frequentist)
                        || matches!(
                            analysis.refute,
                            RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full
                        ))) =>
        {
            let (identifier, estimator) = cx.binding.resolve(RouteDefaults::RESPONSE)?;
            let bayesian = matches!(analysis.inference, InferenceMode::Bayesian(_));
            if bayesian && estimator == crate::strategy_table::EstimatorId::ResponseBayesian
                || !bayesian
                    && estimator == crate::strategy_table::EstimatorId::ResponseKennedyDr
                    && matches!(
                        query.functional,
                        antecedent_core::ResponseFunctional::MeanCurve { .. }
                    )
                || !bayesian
                    && estimator == crate::strategy_table::EstimatorId::ResponseInterventionGcomp
                    && matches!(
                        query.functional,
                        antecedent_core::ResponseFunctional::InterventionResponse { .. }
                    )
            {
                Some(crate::analysis::execute::CheckedStaticDagResponseOperation::checked(
                    crate::analysis::execute::CheckedStaticDagResponseOperationPremises {
                        graph,
                        query,
                        cache,
                        identifier,
                        estimator,
                        options: analysis.response_options.clone().unwrap_or_default(),
                        inference: analysis.inference.clone(),
                        refute: analysis.refute,
                    },
                )?)
            } else {
                None
            }
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::StaticDagResponse))
}

pub(super) fn cell_aipw_response(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, ctx, .. } = *cx;
    let sealed = match (cx.route, analysis.identification_cache.as_ref()) {
        (Some(RouteInput::StaticResponseDag { query, graph, .. }), Some(cache))
            if analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && matches!(
                    analysis.structure_source,
                    crate::support::StructureSource::Explicit
                        | crate::support::StructureSource::Accepted
                ) =>
        {
            let (identifier, estimator) = cx.binding.resolve(RouteDefaults::RESPONSE)?;
            if estimator == crate::strategy_table::EstimatorId::CellAipw
                && analysis.custom_validators.is_empty()
                && analysis.shared_batch_design.is_none()
                && analysis.continuous_cell.is_none()
                && matches!(analysis.inference, InferenceMode::Frequentist)
            {
                Some(crate::analysis::execute::CheckedCellAipwResponseOperation::checked(
                    crate::analysis::execute::CheckedCellAipwResponseOperationPremises {
                        ctx: cx.ctx,
                        structure: &crate::analysis::execute::CellAipwStructure::Dag(graph.clone()),
                        query,
                        cache,
                        identifier,
                        estimator,
                        fold_seed: ctx.rng.master_seed(),
                        validation: analysis.refute,
                        origin: if analysis.structure_source
                            == crate::support::StructureSource::Accepted
                        {
                            crate::analysis::execute::DagResponseOrigin::Accepted
                        } else {
                            crate::analysis::execute::DagResponseOrigin::Explicit
                        },
                        plan: Some(&**cx.plan),
                    },
                )?)
            } else {
                None
            }
        }
        (Some(RouteInput::CodeterminedJoint { query, .. }), Some(cache))
            if analysis.graph_posterior.is_none()
                && analysis.structure_source == crate::support::StructureSource::Explicit
                && analysis.custom_validators.is_empty()
                && analysis.shared_batch_design.is_none()
                && analysis.continuous_cell.is_none()
                && matches!(analysis.inference, InferenceMode::Frequentist)
                && cx.binding.names(IdentifierId::GeneralizedAdjustment, EstimatorId::CellAipw) =>
        {
            let (Some(admg), Some(background)) =
                (analysis.graph.as_admg(), analysis.tiered.as_ref())
            else {
                return Err(CausalError::Compile {
                    message: "CoDetermined joint cell prepare lost its closure ADMG or tiers"
                        .into(),
                });
            };
            Some(crate::analysis::execute::CheckedCellAipwResponseOperation::checked(
                crate::analysis::execute::CheckedCellAipwResponseOperationPremises {
                    ctx: cx.ctx,
                    structure: &crate::analysis::execute::CellAipwStructure::TierClosure {
                        admg: admg.clone(),
                        background: background.clone(),
                    },
                    query,
                    cache,
                    identifier: crate::strategy_table::IdentifierId::GeneralizedAdjustment,
                    estimator: crate::strategy_table::EstimatorId::CellAipw,
                    fold_seed: ctx.rng.master_seed(),
                    validation: analysis.refute,
                    origin: crate::analysis::execute::DagResponseOrigin::Explicit,
                    plan: Some(&**cx.plan),
                },
            )?)
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::CellAipwResponse))
}

pub(super) fn static_response_curve(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, .. } = *cx;
    let sealed = match (cx.static_response_dag(), analysis.identification_cache.as_ref()) {
        (Some((query, _)), Some(cache))
            if query.temporal.is_none()
                && query.observation == antecedent_core::ObservationSpec::Complete
                && query.target_population == TargetPopulation::AllObserved
                && matches!(
                    query.functional,
                    antecedent_core::ResponseFunctional::MeanCurve { .. }
                )
                && matches!(analysis.inference, InferenceMode::Frequentist)
                && cx.binding.runs_defaults(RouteDefaults::RESPONSE) =>
        {
            let antecedent_core::ResponseFunctional::MeanCurve { treatment, .. } =
                &query.functional
            else {
                unreachable!()
            };
            Some(CheckedStaticResponseCurve {
                query: query.clone(),
                grid: Arc::from(treatment.grid.values()?),
                cache: Arc::clone(cache),
                identifier: RouteDefaults::RESPONSE.identifier,
                estimator: RouteDefaults::RESPONSE.estimator,
            })
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::StaticResponseCurve))
}
