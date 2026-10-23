//! Sealers of the program routes: interventional distribution, functional
//! and path-specific effects, and the ADMG response curve.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{SealContext, SealOutcome};
use crate::analysis::builder::RefuteSuite;
use crate::analysis::execute::CheckedStaticCommon;
use crate::analysis::execute::RouteInput;
use crate::analysis::prepared::{
    CheckedAdmgResponseCurveOperation, CheckedDistributionOperation,
    CheckedFunctionalEffectOperation, CheckedFunctionalEffectResponseMember,
    CheckedPathSpecificEffectOperation, PreparedExecution,
};
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::planner::RouteDefaults;
use crate::strategy_table::{EstimatorId, IdentifierId};
use antecedent_core::{CausalQuery, Intervention, OutcomeFunctional, TargetPopulation, Value};
use std::sync::Arc;

pub(super) fn distribution(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, plan, .. } = *cx;
    let sealed = match (cx.distribution(), analysis.identification_cache.as_ref()) {
        (Some((data, query)), Some(cache)) => {
            let (identifier, estimator) = cx.binding.resolve(RouteDefaults::DISTRIBUTION)?;
            let mut fitter = antecedent_estimate::FunctionalDistribution::new();
            fitter.bootstrap_replicates = analysis.bootstrap_replicates;
            (fitter.row_resampling, fitter.sampling_design) = analysis.functional_row_resampling();
            let prepared = fitter.prepare(
                data,
                query,
                &cache.estimand,
                &cache.identification.arena,
                cache.identification.required_assumptions.clone(),
            )?;
            Some(CheckedDistributionOperation {
                query: query.clone(),
                cache: Arc::clone(cache),
                fitter,
                prepared,
                common: CheckedStaticCommon::from_analysis(analysis, plan, identifier, estimator),
            })
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::Distribution))
}

/// The mixed-graph target a retained functional effect answers: the data, the average-effect
/// query the program is prepared against, and the conditional effect it stands for, if any.
type FunctionalEffectTarget<'a> = (
    &'a antecedent_data::TabularData,
    antecedent_core::AverageEffectQuery,
    Option<antecedent_core::ConditionalEffectQuery>,
);

pub(super) fn functional_effect(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, plan, .. } = *cx;
    let target: Option<FunctionalEffectTarget<'_>> = match cx.route {
        Some(RouteInput::StaticAteMixed { data, query })
            if matches!(query.outcome_functional, OutcomeFunctional::Mean)
                && matches!(query.target_population, TargetPopulation::AllObserved) =>
        {
            Some((data, query.clone(), None))
        }
        Some(RouteInput::ConditionalMixed { data, query })
            if matches!(query.inner.outcome_functional, OutcomeFunctional::Mean)
                && matches!(query.inner.target_population, TargetPopulation::AllObserved) =>
        {
            Some((data, query.inner.clone(), Some(query.clone())))
        }
        _ => None,
    };
    let sealed = match (target, analysis.identification_cache.as_ref()) {
        (Some((data, query, conditional)), Some(cache))
            if analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && cache.estimand.method_kind().ok()
                    == Some(antecedent_expr::EstimandMethod::GeneralId)
                && cx.binding.resolves_to(
                    RouteDefaults::ADMG,
                    IdentifierId::GeneralId,
                    EstimatorId::FunctionalEffect,
                ) =>
        {
            let mut fitter = antecedent_estimate::FunctionalEffect::new();
            fitter.bootstrap_replicates = analysis.bootstrap_replicates;
            (fitter.row_resampling, fitter.sampling_design) = analysis.functional_row_resampling();
            let prepared = fitter.prepare(
                data,
                &cache.estimand,
                &cache.identification.arena,
                cache.identification.required_assumptions.clone(),
                &[query.treatment, query.outcome],
            )?;
            Some(CheckedFunctionalEffectOperation {
                query,
                conditional,
                cache: Arc::clone(cache),
                fitter,
                prepared,
                common: CheckedStaticCommon::from_analysis(
                    analysis,
                    plan,
                    crate::strategy_table::IdentifierId::GeneralId,
                    crate::strategy_table::EstimatorId::FunctionalEffect,
                ),
            })
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::FunctionalEffect))
}

pub(super) fn path_specific_effect(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, plan, .. } = *cx;
    let sealed = match (cx.route, analysis.identification_cache.as_ref()) {
        (Some(RouteInput::PathSpecificDag { data, query, .. }), Some(cache))
            if analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && cache.estimand.method_kind().ok()
                    == Some(antecedent_expr::EstimandMethod::PathSpecificNatural)
                && cx.binding.resolves_to(
                    RouteDefaults::PATH,
                    IdentifierId::PathSpecificNatural,
                    EstimatorId::FunctionalEffect,
                )
                && matches!(query.target_population, TargetPopulation::AllObserved) =>
        {
            let mut fitter = antecedent_estimate::FunctionalEffect::new();
            fitter.bootstrap_replicates = analysis.bootstrap_replicates;
            (fitter.row_resampling, fitter.sampling_design) = analysis.functional_row_resampling();
            let mut extra = vec![query.treatment, query.outcome];
            extra.extend(query.path_nodes.iter().copied());
            let prepared = fitter.prepare(
                data,
                &cache.estimand,
                &cache.identification.arena,
                cache.identification.required_assumptions.clone(),
                &extra,
            )?;
            Some(CheckedPathSpecificEffectOperation {
                query: query.clone(),
                cache: Arc::clone(cache),
                fitter,
                prepared,
                common: CheckedStaticCommon::from_analysis(
                    analysis,
                    plan,
                    crate::strategy_table::IdentifierId::PathSpecificNatural,
                    crate::strategy_table::EstimatorId::FunctionalEffect,
                ),
            })
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::PathSpecificEffect))
}

/// Treatment, outcome and evaluation grid of an ADMG response the checked route accepts.
fn admg_response_target(
    query: &antecedent_core::ResponseQuery,
) -> Result<(antecedent_core::VariableId, antecedent_core::VariableId, Vec<f64>), CausalError> {
    Ok(match &query.functional {
        antecedent_core::ResponseFunctional::MeanCurve { outcome, treatment } => {
            (treatment.variable, *outcome, treatment.grid.values()?)
        }
        antecedent_core::ResponseFunctional::InterventionResponse { outcome, interventions }
            if interventions.len() == 1 =>
        {
            let intervention = &interventions[0];
            let (treatment, _) = query.functional.primary_pair().ok_or_else(|| {
                CausalError::Compile { message: "scalar ADMG response lacks treatment role".into() }
            })?;
            let antecedent_core::Intervention::Set { variable, value } = intervention else {
                return Err(CausalError::Compile {
                    message: "checked ADMG scalar response requires a single hard-set intervention"
                        .into(),
                });
            };
            let value = value.as_f64().filter(|value| value.is_finite()).ok_or_else(|| CausalError::Compile {
                message: "checked ADMG scalar response requires one finite numeric intervention value".into(),
            })?;
            if treatment != *variable {
                return Err(CausalError::Compile {
                    message: "checked ADMG scalar response intervention must assign its declared treatment".into(),
                });
            }
            (treatment, *outcome, vec![value])
        }
        _ => unreachable!("guarded response operation shape"),
    })
}

/// One identified and prepared member per grid level of an ADMG response curve.
fn admg_response_members(
    fitter: &antecedent_estimate::FunctionalEffect,
    data: &antecedent_data::TabularData,
    query: &antecedent_core::ResponseQuery,
    admg: &antecedent_graph::Admg,
    grid: &[f64],
    treatment: antecedent_core::VariableId,
    outcome: antecedent_core::VariableId,
) -> Result<Vec<CheckedFunctionalEffectResponseMember>, CausalError> {
    let mut members = Vec::with_capacity(grid.len());
    for level in grid {
        let mut member_query = query.clone();
        if matches!(query.functional, antecedent_core::ResponseFunctional::MeanCurve { .. }) {
            member_query.functional = antecedent_core::ResponseFunctional::InterventionResponse {
                outcome,
                interventions: Arc::from([Intervention::set(treatment, Value::f64(*level))]),
            };
        }
        let member_causal_query = CausalQuery::Response(member_query.clone());
        let identification = crate::strategy_table::identify_admg_query(
            crate::strategy_table::IdentifierId::GeneralId,
            admg,
            &member_causal_query,
        )?;
        let estimand = crate::strategy_table::select_estimand(
            &identification,
            crate::strategy_table::EstimatorId::FunctionalEffect,
        )?;
        let prepared = fitter.prepare(
            data,
            &estimand,
            &identification.arena,
            identification.required_assumptions.clone(),
            &[treatment, outcome],
        )?;
        members.push(CheckedFunctionalEffectResponseMember {
            grid_value: *level,
            query: member_query,
            identification,
            estimand,
            prepared,
        });
    }
    Ok(members)
}

pub(super) fn admg_response_curve(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, plan, .. } = *cx;
    let sealed = match cx.route {
        Some(RouteInput::MixedAdmgResponse { data, query })
            if analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && query.temporal.is_none()
                && query.observation == antecedent_core::ObservationSpec::Complete
                && query.target_population == TargetPopulation::AllObserved
                && query.outcome_functional == OutcomeFunctional::Mean
                && matches!(
                    (&query.functional, analysis.refute),
                    (_, RefuteSuite::None)
                        | (
                            antecedent_core::ResponseFunctional::InterventionResponse { .. },
                            RefuteSuite::Cheap | RefuteSuite::Full,
                        )
                )
                && cx.binding.resolves_to(
                    RouteDefaults::ADMG,
                    IdentifierId::GeneralId,
                    EstimatorId::FunctionalEffect,
                )
                && matches!(
                    query.functional,
                    antecedent_core::ResponseFunctional::MeanCurve { .. }
                        | antecedent_core::ResponseFunctional::InterventionResponse { .. }
                )
                && matches!(
                    analysis.inference,
                    InferenceMode::Frequentist | InferenceMode::Bayesian(_)
                ) =>
        {
            let (treatment, outcome, grid) = admg_response_target(query)?;
            if grid.is_empty() {
                return Err(CausalError::Compile {
                    message: "ADMG response curve has no grid members".into(),
                });
            }
            let admg = analysis.graph.graph_ref().projected_admg()?.ok_or_else(|| {
                CausalError::Compile {
                    message: "ADMG response operation lacks an ADMG graph".into(),
                }
            })?;
            let mut fitter = antecedent_estimate::FunctionalEffect::new();
            fitter.bootstrap_replicates = analysis.bootstrap_replicates;
            (fitter.row_resampling, fitter.sampling_design) = analysis.functional_row_resampling();
            let members =
                admg_response_members(&fitter, data, query, &admg, &grid, treatment, outcome)?;
            Some(CheckedAdmgResponseCurveOperation {
                factor_snapshot_digest: data.storage().content_digest(),
                query: query.clone(),
                grid: Arc::from(grid),
                members: Arc::from(members),
                fitter,
                common: CheckedStaticCommon::from_analysis(
                    analysis,
                    plan,
                    crate::strategy_table::IdentifierId::GeneralId,
                    crate::strategy_table::EstimatorId::FunctionalEffect,
                ),
            })
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::AdmgResponseCurve))
}
