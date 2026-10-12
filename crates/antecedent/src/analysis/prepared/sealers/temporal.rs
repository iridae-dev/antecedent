//! Sealers of the series routes over a TemporalDag or a retained class
//! envelope: mediation, effect and response.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{SealContext, SealOutcome};
use crate::accepted::GraphClass;
use crate::analysis::builder::{DataInput, RefuteSuite};
use crate::analysis::prepared::{PreparedExecution, single_step_dose};
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::strategy_table::EstimatorId;
use antecedent_core::{CausalQuery, OutcomeFunctional, TargetPopulation, TemporalEffectQuery};

pub(super) fn temporal_mediation(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { source, analysis, plan, .. } = *cx;
    let sealed = match (
        &source.data,
        &source.query,
        analysis.temporal_identification_cache.as_deref(),
        analysis.graph.class(),
    ) {
        (
            DataInput::Temporal(data) | DataInput::Event(data),
            CausalQuery::Mediation(_),
            Some(cache),
            GraphClass::TemporalDag,
        ) if analysis.graph_posterior.is_none()
            && analysis.tiered.is_none()
            && analysis.split.is_none()
            && analysis.custom_validators.is_empty()
            && matches!(
                analysis.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && matches!(
                analysis.refute,
                RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full
            ) =>
        {
            Some(crate::analysis::execute::CheckedTemporalMediationOperation::checked(
                analysis, data, plan, cache,
            )?)
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::TemporalMediation))
}

pub(super) fn temporal_class_mediation(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { source, analysis, plan, .. } = *cx;
    let sealed = match &source.data {
        DataInput::Temporal(data) | DataInput::Event(data)
            if crate::analysis::execute::CheckedTemporalClassMediationOperation::admits(
                analysis,
            ) && (analysis.temporal_class_identification_cache.is_some()
                || analysis.temporal_class_posterior_identification_cache.is_some()
                || analysis.dbn_posterior_identification_cache.is_some()) =>
        {
            Some(Box::new(
                crate::analysis::execute::CheckedTemporalClassMediationOperation::checked(
                    analysis, data, plan,
                )?,
            ))
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::TemporalClassMediation))
}

pub(super) fn temporal_dag_effect(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { source, analysis, plan, .. } = *cx;
    let sealed = match (
        &source.data,
        &source.query,
        analysis.temporal_identification_cache.as_deref(),
        analysis.graph.class(),
    ) {
        (
            DataInput::Temporal(_) | DataInput::Event(_),
            CausalQuery::TemporalEffect(query),
            Some(cache),
            GraphClass::TemporalDag,
        ) if analysis.graph_posterior.is_none()
            && analysis.tiered.is_none()
            && matches!(
                analysis.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && analysis.split.is_none()
            && analysis.custom_validators.is_empty()
            && matches!(analysis.inference, InferenceMode::Frequentist)
            && matches!(
                analysis.refute,
                RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full
            )
            && query.policy.is_pulse_or_sustained() =>
        {
            let identifier = crate::strategy_table::IdentifierId::TemporalBackdoorUnfolded;
            let estimator = EstimatorId::temporal_effect_procedure(query, false);
            if plan
                .logical
                .record
                .identifier
                .as_deref()
                .is_some_and(|name| name != identifier.as_str())
                || plan
                    .logical
                    .record
                    .estimator
                    .as_deref()
                    .is_some_and(|name| name != estimator.as_str())
            {
                None
            } else {
                let member =
                    cache.get(query.horizon_steps).ok_or_else(|| CausalError::Compile {
                        message: "prepared temporal effect lacks its horizon proof".into(),
                    })?;
                let operation = crate::analysis::CheckedTemporalEffectOperation::checked(
                    analysis.graph.as_temporal_dag().expect("guarded TemporalDag"),
                    query,
                    member.identification.clone(),
                    member.estimand.clone(),
                    member.indexer.clone(),
                    identifier,
                    estimator,
                    analysis.bootstrap_replicates,
                    analysis.refute,
                )?;
                Some(crate::analysis::execute::CheckedTemporalEffectExecution::checked(
                    operation,
                    crate::analysis::execute::IdentifiedResultContext::from_study(analysis),
                    plan.clone(),
                )?)
            }
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::TemporalDagEffect))
}

pub(super) fn bayesian_temporal_dag_effect(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { source, analysis, plan, .. } = *cx;
    let sealed = match (
        &source.data,
        &source.query,
        analysis.temporal_identification_cache.as_deref(),
        analysis.graph.class(),
    ) {
        (
            DataInput::Temporal(_) | DataInput::Event(_),
            CausalQuery::TemporalEffect(query),
            Some(cache),
            GraphClass::TemporalDag,
        ) if analysis.graph_posterior.is_none()
            && analysis.tiered.is_none()
            && matches!(
                analysis.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && analysis.split.is_none()
            && analysis.custom_validators.is_empty()
            && matches!(analysis.inference, InferenceMode::Bayesian(_))
            && query.policy.is_pulse_or_sustained()
            && analysis.point_validation() =>
        {
            let identifier = crate::strategy_table::IdentifierId::TemporalBackdoorUnfolded;
            let estimator = EstimatorId::temporal_effect_procedure(query, true);
            if plan
                .logical
                .record
                .identifier
                .as_deref()
                .is_some_and(|name| name != identifier.as_str())
                || plan
                    .logical
                    .record
                    .estimator
                    .as_deref()
                    .is_some_and(|name| name != estimator.as_str())
            {
                None
            } else {
                let member =
                    cache.get(query.horizon_steps).ok_or_else(|| CausalError::Compile {
                        message: "prepared Bayesian temporal effect lacks its horizon proof".into(),
                    })?;
                let procedure = EstimatorId::temporal_effect_procedure(query, false);
                let target = crate::analysis::CheckedTemporalEffectOperation::checked(
                    analysis.graph.as_temporal_dag().expect("guarded TemporalDag"),
                    query,
                    member.identification.clone(),
                    member.estimand.clone(),
                    member.indexer.clone(),
                    identifier,
                    procedure,
                    0,
                    analysis.refute,
                )?;
                Some(crate::analysis::CheckedBayesianTemporalEffectOperation::checked(
                    target,
                    &analysis.inference,
                    analysis.refute,
                )?)
            }
        }
        _ => None,
    };
    Ok(sealed.map(|operation| PreparedExecution::BayesianTemporalDagEffect(Box::new(operation))))
}

pub(super) fn temporal_class_effect(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { source, analysis, plan, .. } = *cx;
    let sealed = match (
        &source.data,
        &source.query,
        analysis.temporal_class_identification_cache.as_deref(),
        analysis.graph.class(),
    ) {
        (
            DataInput::Temporal(_) | DataInput::Event(_),
            CausalQuery::TemporalEffect(query),
            Some(cache),
            GraphClass::TemporalCpdag | GraphClass::TemporalPag,
        ) if analysis.graph_posterior.is_none()
            && analysis.tiered.is_none()
            && matches!(
                analysis.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && matches!(analysis.inference, InferenceMode::Frequentist)
            && query.policy.is_pulse_or_sustained() =>
        {
            let identifier = crate::strategy_table::IdentifierId::GeneralizedAdjustment;
            let estimator = EstimatorId::temporal_effect_procedure(query, false);
            if plan
                .logical
                .record
                .identifier
                .as_deref()
                .is_some_and(|name| name != identifier.as_str())
                || plan
                    .logical
                    .record
                    .estimator
                    .as_deref()
                    .is_some_and(|name| name != estimator.as_str())
            {
                None
            } else {
                let operation = crate::analysis::CheckedTemporalClassEffectOperation::checked(
                    analysis.graph.clone(),
                    query,
                    cache,
                    analysis.max_completions,
                    identifier,
                    estimator,
                    analysis.bootstrap_replicates,
                    analysis.split,
                    analysis.refute,
                    analysis.custom_validators.clone(),
                )?;
                Some(crate::analysis::execute::CheckedTemporalClassEffectExecution::checked(
                    operation,
                    crate::analysis::execute::IdentifiedResultContext::from_study(analysis),
                    plan.clone(),
                )?)
            }
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::TemporalClassEffect))
}

pub(super) fn temporal_dag_response(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { source, analysis, plan, .. } = *cx;
    let sealed = match (
        &source.data,
        &source.query,
        analysis.temporal_identification_cache.as_deref(),
        analysis.graph.class(),
    ) {
        (
            DataInput::Temporal(_) | DataInput::Event(_),
            CausalQuery::Response(query),
            Some(cache),
            GraphClass::TemporalDag,
        ) if analysis.graph_posterior.is_none()
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
            && analysis.refute == RefuteSuite::None
            && query.temporal.is_some()
            && query.temporal.as_ref().is_some_and(|spec| match &spec.policy {
                antecedent_core::TemporalPolicy::Pulse { .. } => true,
                antecedent_core::TemporalPolicy::Sustained { from, until } => from == until,
                _ => false,
            })
            && query.observation == antecedent_core::ObservationSpec::Complete
            && query.target_population == TargetPopulation::AllObserved
            && matches!(query.outcome_functional, OutcomeFunctional::Mean)
            && crate::analysis::temporal_response_is_direct(query) =>
        {
            let identifier = crate::strategy_table::IdentifierId::TemporalBackdoorUnfolded;
            let estimator = EstimatorId::temporal_response_for(&analysis.inference);
            if plan
                .logical
                .record
                .identifier
                .as_deref()
                .is_some_and(|selected| selected != identifier.as_str())
                || plan
                    .logical
                    .record
                    .estimator
                    .as_deref()
                    .is_some_and(|selected| selected != estimator.as_str())
            {
                None
            } else {
                let temporal = query.temporal.as_ref().expect("guarded temporal response");
                let (treatment, outcome) =
                    query.functional.primary_pair().ok_or_else(|| CausalError::Compile {
                        message: "temporal response query has no treatment/outcome pair".into(),
                    })?;
                let dose = single_step_dose(query)?.unwrap_or(1.0);
                let evidence = cache
                    .by_horizon
                    .iter()
                    .map(|member| {
                        let effect_query = TemporalEffectQuery::pulse(treatment, outcome, dose)
                            .with_policy(temporal.policy.clone())
                            .with_horizon_steps(member.horizon)
                            .with_max_history_lag(temporal.max_history_lag)
                            .with_target_population(query.target_population.clone());
                        crate::analysis::TemporalResponseHorizonEvidence::checked(
                            member.horizon,
                            effect_query,
                            member.identification.clone(),
                            member.estimand.clone(),
                            member.indexer.clone(),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let operation = crate::analysis::CheckedTemporalResponseOperation::checked(
                    analysis.graph.as_temporal_dag().expect("checked TemporalDag response"),
                    query,
                    evidence,
                    identifier,
                    estimator,
                    analysis.bootstrap_replicates,
                    &analysis.inference,
                )?;
                let execution =
                    crate::analysis::execute::CheckedTemporalResponseExecution::checked(
                        operation,
                        crate::analysis::execute::IdentifiedResultContext::from_study(analysis),
                        plan.clone(),
                    )?;
                Some(execution)
            }
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::TemporalDagResponse))
}
