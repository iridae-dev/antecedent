//! Sealing of graph-posterior, class and temporal operations at prepare time.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::accepted::GraphClass;
use crate::analysis::builder::{DataInput, RefuteSuite};
use crate::analysis::execute::Study;
use crate::analysis::route_guards::{
    PlanRecordMode, complete_mean_response, mean_all_observed, plan_record_is,
};
use crate::analysis::{
    CheckedAdmgGraphPosteriorResponse, CheckedBayesianGraphPosteriorAte,
    CheckedClassGraphPosteriorEffect, CheckedGraphPosteriorEffect, CheckedStaticClassEffect,
    StaticClassGraph, StaticClassIdentification,
};
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::planner::PhysicalExecutionPlan;
use crate::strategy_table::{DEFAULT_ESTIMATOR, EstimatorId, IdentifierId};
use antecedent_core::{AverageEffectQuery, CausalQuery};
use antecedent_estimate::OverlapPolicy;
use std::sync::Arc;

impl Study {
    pub(super) fn seal_static_class_effect(
        analysis: &Study,
        plan: &PhysicalExecutionPlan,
    ) -> Result<Option<CheckedStaticClassEffect>, CausalError> {
        let CausalQuery::AverageEffect(query) = &analysis.query else {
            return Ok(None);
        };
        let expected_estimator = CheckedStaticClassEffect::expected_estimator(&analysis.inference);
        if analysis.graph_posterior.is_some()
            || !analysis.fixed_structure()
            || !mean_all_observed(&query.outcome_functional, &query.target_population)
            || !analysis.point_validation_or_placebo()
            || !analysis.custom_validators.is_empty()
            || !plan_record_is(
                plan,
                IdentifierId::GeneralizedAdjustment,
                PlanRecordMode::Exact,
                expected_estimator,
                PlanRecordMode::Exact,
            )
        {
            return Ok(None);
        }
        let (graph, identification) = match analysis.graph.class() {
            GraphClass::Cpdag => {
                let (Some(graph), Some(cache)) =
                    (analysis.graph.as_cpdag(), analysis.cpdag_identification_cache.as_deref())
                else {
                    return Ok(None);
                };
                (
                    StaticClassGraph::Cpdag(graph.clone()),
                    StaticClassIdentification::Cpdag(cache.clone()),
                )
            }
            GraphClass::Pag => {
                let (Some(graph), Some(cache)) =
                    (plan.static_pag(), analysis.pag_identification_cache.as_deref())
                else {
                    return Ok(None);
                };
                (
                    StaticClassGraph::Pag(graph.clone()),
                    StaticClassIdentification::Pag(cache.clone()),
                )
            }
            _ => return Ok(None),
        };
        let procedure = analysis
            .estimator_spec
            .clone()
            .unwrap_or(crate::EstimatorSpec::Default(expected_estimator));
        Ok(Some(CheckedStaticClassEffect::prepare(
            graph,
            query.clone(),
            identification,
            plan.clone(),
            procedure,
            analysis.inference.clone(),
            analysis.bootstrap_replicates,
            analysis.overlap_policy.unwrap_or(OverlapPolicy::ExplicitOverride),
            analysis.refute,
        )?))
    }

    /// Whether a graph-posterior effect target sits inside the point-estimate
    /// envelope every graph-posterior sealer shares: mean functional, the
    /// all-observed population, a built-in validation suite, and no custom
    /// validators.
    pub(crate) fn graph_posterior_target_is_sealable(
        analysis: &Study,
        query: &AverageEffectQuery,
    ) -> bool {
        mean_all_observed(&query.outcome_functional, &query.target_population)
            && analysis.point_validation()
            && analysis.custom_validators.is_empty()
    }

    /// Whether the builder's estimator choices resolve to exactly `estimator`.
    pub(crate) fn graph_posterior_procedure_is(analysis: &Study, estimator: EstimatorId) -> bool {
        analysis.estimator_spec.as_ref().is_none_or(|spec| spec.id() == estimator)
            && analysis.estimator.is_none_or(|id| id == estimator)
    }

    #[inline(never)]
    pub(super) fn seal_graph_posterior_effect(
        analysis: &Study,
        plan: &PhysicalExecutionPlan,
    ) -> Result<Option<CheckedGraphPosteriorEffect>, CausalError> {
        let (Some(posterior), Some(identification), Some(target)) = (
            analysis.graph_posterior.as_ref(),
            analysis.graph_posterior_identification_cache.as_deref(),
            crate::analysis::GraphPosteriorEffectTarget::from_query(&analysis.query),
        ) else {
            return Ok(None);
        };
        if !Self::graph_posterior_target_is_sealable(analysis, target.inner()) {
            return Ok(None);
        }
        match (posterior.atom_kind, &analysis.inference) {
            (antecedent_discovery::GraphPosteriorAtomKind::Dag, InferenceMode::Frequentist) => {
                let estimator = target.frequentist_estimator();
                if !Self::graph_posterior_procedure_is(analysis, estimator)
                    || !plan_record_is(
                        plan,
                        IdentifierId::BackdoorAdjustment,
                        PlanRecordMode::OrDefault(crate::strategy_table::DEFAULT_IDENTIFIER),
                        estimator,
                        PlanRecordMode::OrDefault(DEFAULT_ESTIMATOR),
                    )
                {
                    return Ok(None);
                }
                Ok(Some(CheckedGraphPosteriorEffect::prepare(
                    posterior.clone(),
                    target,
                    identification.clone(),
                    analysis
                        .estimator_spec
                        .clone()
                        .unwrap_or(crate::EstimatorSpec::Default(estimator)),
                    analysis.bootstrap_replicates,
                    analysis.overlap_policy.unwrap_or(OverlapPolicy::ExplicitOverride),
                    analysis.population_registry.clone(),
                    analysis.latency_mode,
                    analysis.refute,
                )?))
            }
            (
                antecedent_discovery::GraphPosteriorAtomKind::Admg
                | antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg,
                InferenceMode::Frequentist | InferenceMode::Bayesian(_),
            ) if !target.is_conditional()
                && plan_record_is(
                    plan,
                    IdentifierId::GeneralId,
                    PlanRecordMode::OrDefault(crate::strategy_table::DEFAULT_ADMG_IDENTIFIER),
                    EstimatorId::FunctionalEffect,
                    PlanRecordMode::OrDefault(crate::strategy_table::DEFAULT_ADMG_ESTIMATOR),
                ) =>
            {
                Ok(Some(CheckedGraphPosteriorEffect::prepare(
                    posterior.clone(),
                    target,
                    identification.clone(),
                    analysis.estimator_spec.clone().unwrap_or(crate::EstimatorSpec::Default(
                        crate::strategy_table::EstimatorId::FunctionalEffect,
                    )),
                    analysis.bootstrap_replicates,
                    analysis.overlap_policy.unwrap_or(OverlapPolicy::ExplicitOverride),
                    analysis.population_registry.clone(),
                    analysis.latency_mode,
                    analysis.refute,
                )?))
            }
            _ => Ok(None),
        }
    }

    #[inline(never)]
    pub(super) fn seal_class_graph_posterior_effect(
        analysis: &Study,
        plan: &PhysicalExecutionPlan,
    ) -> Result<Option<CheckedClassGraphPosteriorEffect>, CausalError> {
        let (Some(posterior), Some(identification), Some(target)) = (
            analysis.graph_posterior.as_ref(),
            analysis.graph_posterior_identification_cache.as_deref(),
            crate::analysis::GraphPosteriorEffectTarget::from_query(&analysis.query),
        ) else {
            return Ok(None);
        };
        if !matches!(
            posterior.atom_kind,
            antecedent_discovery::GraphPosteriorAtomKind::Cpdag
                | antecedent_discovery::GraphPosteriorAtomKind::Pag
        ) || !Self::graph_posterior_target_is_sealable(analysis, target.inner())
        {
            return Ok(None);
        }
        let estimator = match &analysis.inference {
            InferenceMode::Frequentist => target.frequentist_estimator(),
            InferenceMode::Bayesian(_) => target.bayesian_estimator(),
        };
        if !Self::graph_posterior_procedure_is(analysis, estimator) {
            return Ok(None);
        }
        let identifier = crate::strategy_table::IdentifierId::GeneralizedAdjustment;
        if !plan_record_is(
            plan,
            identifier,
            PlanRecordMode::Exact,
            estimator,
            PlanRecordMode::Exact,
        ) {
            return Ok(None);
        }
        Ok(Some(CheckedClassGraphPosteriorEffect::prepare(
            posterior.clone(),
            target,
            identification.clone(),
            analysis.inference.clone(),
            analysis.estimator_spec.clone().unwrap_or(crate::EstimatorSpec::Default(estimator)),
            analysis.bootstrap_replicates,
            analysis.overlap_policy.unwrap_or(OverlapPolicy::ExplicitOverride),
            analysis.population_registry.clone(),
            analysis.latency_mode,
            analysis.refute,
        )?))
    }

    #[inline(never)]
    pub(super) fn seal_bayesian_graph_posterior_ate(
        analysis: &Study,
        plan: &PhysicalExecutionPlan,
    ) -> Result<Option<Box<CheckedBayesianGraphPosteriorAte>>, CausalError> {
        let (Some(posterior), Some(identification), Some(target), InferenceMode::Bayesian(_)) = (
            analysis.graph_posterior.as_ref(),
            analysis.graph_posterior_identification_cache.as_deref(),
            crate::analysis::GraphPosteriorEffectTarget::from_query(&analysis.query),
            &analysis.inference,
        ) else {
            return Ok(None);
        };
        let estimator = target.bayesian_estimator();
        if posterior.atom_kind != antecedent_discovery::GraphPosteriorAtomKind::Dag
            || !Self::graph_posterior_target_is_sealable(analysis, target.inner())
            || !Self::graph_posterior_procedure_is(analysis, estimator)
            || !plan_record_is(
                plan,
                IdentifierId::BackdoorAdjustment,
                PlanRecordMode::OrDefault(crate::strategy_table::DEFAULT_IDENTIFIER),
                estimator,
                PlanRecordMode::OrDefault(DEFAULT_ESTIMATOR),
            )
        {
            return Ok(None);
        }
        let procedure =
            analysis.estimator_spec.clone().unwrap_or(crate::EstimatorSpec::Default(estimator));
        Ok(Some(Box::new(CheckedBayesianGraphPosteriorAte::prepare(
            posterior.clone(),
            target,
            identification.clone(),
            analysis.inference.clone(),
            procedure,
            plan.clone(),
            analysis.refute,
            analysis.overlap_policy.unwrap_or(OverlapPolicy::ExplicitOverride),
            analysis.latency_mode,
        )?)))
    }

    pub(super) fn seal_admg_graph_posterior_response(
        analysis: &Study,
        plan: &PhysicalExecutionPlan,
    ) -> Result<Option<Box<CheckedAdmgGraphPosteriorResponse>>, CausalError> {
        let (Some(posterior), CausalQuery::Response(query), Some(identification)) = (
            analysis.graph_posterior.as_ref(),
            &analysis.query,
            analysis.graph_posterior_identification_cache.as_deref(),
        ) else {
            return Ok(None);
        };
        if !matches!(
            posterior.atom_kind,
            antecedent_discovery::GraphPosteriorAtomKind::Admg
                | antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg
        ) || !analysis.custom_validators.is_empty()
            || !matches!(
                analysis.estimator_spec.as_ref().map(crate::EstimatorSpec::id),
                None | Some(crate::strategy_table::EstimatorId::FunctionalEffect)
            )
            || !matches!(
                analysis.estimator,
                None | Some(crate::strategy_table::EstimatorId::FunctionalEffect)
            )
            || !plan_record_is(
                plan,
                IdentifierId::GeneralId,
                PlanRecordMode::OrDefault(crate::strategy_table::DEFAULT_ADMG_IDENTIFIER),
                EstimatorId::FunctionalEffect,
                PlanRecordMode::OrDefault(crate::strategy_table::DEFAULT_ADMG_ESTIMATOR),
            )
        {
            return Ok(None);
        }
        crate::analysis::execute::graph_posterior_response_supported(query)?;
        Ok(Some(Box::new(CheckedAdmgGraphPosteriorResponse::prepare(
            posterior.clone(),
            query.clone(),
            identification.clone(),
            plan.clone(),
            analysis.inference.clone(),
            analysis.refute,
        )?)))
    }

    /// Seal a TemporalCpdag/TemporalPag response over its retained per-horizon
    /// completion envelopes (explicit or accepted structure, validation none).
    pub(super) fn seal_temporal_class_response(
        analysis: &Study,
        plan: &PhysicalExecutionPlan,
    ) -> Result<Option<Box<crate::analysis::CheckedTemporalClassResponseOperation>>, CausalError>
    {
        let (
            DataInput::Temporal(_) | DataInput::Event(_),
            CausalQuery::Response(query),
            Some(cache),
            GraphClass::TemporalCpdag | GraphClass::TemporalPag,
        ) = (
            &analysis.data,
            &analysis.query,
            analysis.temporal_class_identification_cache.as_deref(),
            analysis.graph.class(),
        )
        else {
            return Ok(None);
        };
        if analysis.graph_posterior.is_some()
            || analysis.tiered.is_some()
            || !analysis.fixed_structure()
            || !analysis.custom_validators.is_empty()
            || analysis.refute != RefuteSuite::None
            || analysis.observation_delayed_entry.is_some()
            || !complete_mean_response(query)
            || !crate::analysis::temporal_response_is_direct(query)
            || analysis
                .identifier
                .is_some_and(|id| id != crate::strategy_table::IdentifierId::GeneralizedAdjustment)
        {
            return Ok(None);
        }
        let expected_estimator = EstimatorId::temporal_response_for(&analysis.inference);
        if analysis.estimator.is_some_and(|id| id != expected_estimator) {
            return Ok(None);
        }
        Ok(Some(Box::new(crate::analysis::CheckedTemporalClassResponseOperation::checked(
            analysis.graph.clone(),
            query,
            cache,
            analysis.max_completions,
            &analysis.inference,
            analysis.bootstrap_replicates,
            analysis.class_prior.clone(),
            analysis.refute,
            plan.clone(),
        )?)))
    }

    /// Seal a temporal graph-posterior response over its frozen atoms and
    /// per-atom proofs (TemporalDag, TemporalCpdag, or TemporalPag atoms).
    pub(super) fn seal_temporal_graph_posterior_response(
        analysis: &Study,
        plan: &PhysicalExecutionPlan,
    ) -> Result<Option<Box<crate::analysis::CheckedTemporalGraphPosteriorResponse>>, CausalError>
    {
        let (
            DataInput::Temporal(_) | DataInput::Event(_),
            Some(posterior),
            CausalQuery::Response(query),
        ) = (&analysis.data, analysis.graph_posterior.as_ref(), &analysis.query)
        else {
            return Ok(None);
        };
        if !crate::analysis::CheckedTemporalGraphPosteriorResponse::admits(analysis) {
            return Ok(None);
        }
        let proof = match posterior.atom_kind {
            antecedent_discovery::GraphPosteriorAtomKind::Dag => {
                let Some(cache) = analysis.dbn_posterior_identification_cache.as_ref() else {
                    return Ok(None);
                };
                crate::analysis::TemporalPosteriorResponseProof::Dbn(Arc::clone(cache))
            }
            antecedent_discovery::GraphPosteriorAtomKind::Cpdag
            | antecedent_discovery::GraphPosteriorAtomKind::Pag => {
                let Some(cache) = analysis.temporal_class_posterior_identification_cache.as_ref()
                else {
                    return Ok(None);
                };
                crate::analysis::TemporalPosteriorResponseProof::Class(Arc::clone(cache))
            }
            _ => return Ok(None),
        };
        Ok(Some(Box::new(crate::analysis::CheckedTemporalGraphPosteriorResponse::prepare(
            posterior.clone(),
            query.clone(),
            proof,
            plan.clone(),
            analysis.inference.clone(),
            analysis.refute,
            analysis.bootstrap_replicates,
            analysis.max_completions,
            analysis.class_prior.clone(),
        )?)))
    }

    #[inline(never)]
    pub(super) fn seal_bayesian_temporal_class_effect(
        analysis: &Study,
        plan: &PhysicalExecutionPlan,
    ) -> Result<
        Option<Box<crate::analysis::CheckedBayesianTemporalClassEffectOperation>>,
        CausalError,
    > {
        let (
            DataInput::Temporal(_) | DataInput::Event(_),
            CausalQuery::TemporalEffect(query),
            Some(cache),
            GraphClass::TemporalCpdag | GraphClass::TemporalPag,
            InferenceMode::Bayesian(_),
        ) = (
            &analysis.data,
            &analysis.query,
            analysis.temporal_class_identification_cache.as_deref(),
            analysis.graph.class(),
            &analysis.inference,
        )
        else {
            return Ok(None);
        };
        if !crate::analysis::CheckedBayesianTemporalClassEffectOperation::admits(analysis) {
            return Ok(None);
        }
        let identifier = crate::strategy_table::IdentifierId::GeneralizedAdjustment;
        let estimator = EstimatorId::temporal_effect_procedure(query, true);
        let procedure = EstimatorId::temporal_effect_procedure(query, false);
        if !plan_record_is(
            plan,
            identifier,
            PlanRecordMode::Exact,
            estimator,
            PlanRecordMode::Exact,
        ) || !analysis.estimator_spec.as_ref().is_none_or(|spec| spec.id() == estimator)
        {
            return Ok(None);
        }
        let target = crate::analysis::CheckedTemporalClassEffectOperation::checked(
            analysis.graph.clone(),
            query,
            cache,
            analysis.max_completions,
            identifier,
            procedure,
            analysis.bootstrap_replicates,
            analysis.split,
            analysis.refute,
            Vec::new(),
        )?;
        Ok(Some(Box::new(crate::analysis::CheckedBayesianTemporalClassEffectOperation::checked(
            target,
            &analysis.inference,
            analysis.refute,
            analysis.class_prior.clone(),
            plan.clone(),
        )?)))
    }

    #[inline(never)]
    pub(super) fn seal_temporal_graph_posterior_effect(
        analysis: &Study,
        plan: &PhysicalExecutionPlan,
    ) -> Result<Option<Box<crate::analysis::CheckedTemporalGraphPosteriorEffect>>, CausalError>
    {
        let (
            DataInput::Temporal(_) | DataInput::Event(_),
            CausalQuery::TemporalEffect(query),
            Some(posterior),
        ) = (&analysis.data, &analysis.query, analysis.graph_posterior.as_ref())
        else {
            return Ok(None);
        };
        if !crate::analysis::CheckedTemporalGraphPosteriorEffect::admits(analysis) {
            return Ok(None);
        }
        let proof = match posterior.atom_kind {
            antecedent_discovery::GraphPosteriorAtomKind::Dag => analysis
                .dbn_posterior_identification_cache
                .clone()
                .map(crate::analysis::CheckedTemporalGraphPosteriorProof::Dbn),
            antecedent_discovery::GraphPosteriorAtomKind::Cpdag
            | antecedent_discovery::GraphPosteriorAtomKind::Pag => analysis
                .temporal_class_posterior_identification_cache
                .clone()
                .map(crate::analysis::CheckedTemporalGraphPosteriorProof::Class),
            _ => None,
        };
        let Some(proof) = proof else {
            return Ok(None);
        };
        let identifier = crate::strategy_table::IdentifierId::TemporalBackdoorUnfolded;
        let estimator = EstimatorId::temporal_effect_for(query, &analysis.inference);
        if !plan_record_is(
            plan,
            identifier,
            PlanRecordMode::IfPresent,
            estimator,
            PlanRecordMode::IfPresent,
        ) || !analysis.estimator_spec.as_ref().is_none_or(|spec| spec.id() == estimator)
        {
            return Ok(None);
        }
        Ok(Some(Box::new(crate::analysis::CheckedTemporalGraphPosteriorEffect::prepare(
            posterior.clone(),
            query.clone(),
            proof,
            analysis.inference.clone(),
            analysis.estimator_spec.as_ref(),
            analysis.bootstrap_replicates,
            analysis.split,
            analysis.max_completions,
            analysis.refute,
            plan.clone(),
        )?)))
    }

    pub(super) fn seal_static_class_response(
        analysis: &Study,
        plan: &PhysicalExecutionPlan,
    ) -> Result<Option<Box<crate::analysis::execute::CheckedStaticClassResponse>>, CausalError>
    {
        let CausalQuery::Response(query) = &analysis.query else {
            return Ok(None);
        };
        if !crate::analysis::execute::CheckedStaticClassResponse::admits(analysis) {
            return Ok(None);
        }
        let (graph, identification) = match analysis.graph.class() {
            GraphClass::Cpdag => {
                let (Some(graph), Some(cache)) =
                    (analysis.graph.as_cpdag(), analysis.cpdag_identification_cache.as_deref())
                else {
                    return Ok(None);
                };
                (
                    StaticClassGraph::Cpdag(graph.clone()),
                    StaticClassIdentification::Cpdag(cache.clone()),
                )
            }
            GraphClass::Pag => {
                let (Some(graph), Some(cache)) =
                    (plan.static_pag(), analysis.pag_identification_cache.as_deref())
                else {
                    return Ok(None);
                };
                (
                    StaticClassGraph::Pag(graph.clone()),
                    StaticClassIdentification::Pag(cache.clone()),
                )
            }
            _ => return Ok(None),
        };
        let expected_estimator =
            EstimatorId::static_response_for(&query.functional, &analysis.inference);
        let identifier: IdentifierId = plan
            .logical
            .record
            .identifier
            .as_deref()
            .unwrap_or(crate::strategy_table::DEFAULT_PAG_IDENTIFIER)
            .parse()?;
        let estimator: EstimatorId = plan
            .logical
            .record
            .estimator
            .as_deref()
            .unwrap_or(expected_estimator.as_str())
            .parse()?;
        if identifier != IdentifierId::GeneralizedAdjustment
            || estimator != expected_estimator
            || analysis.estimator.is_some_and(|selected| selected != estimator)
            || analysis.estimator_spec.as_ref().is_some_and(|spec| spec.id() != estimator)
        {
            return Ok(None);
        }
        Ok(Some(Box::new(crate::analysis::execute::CheckedStaticClassResponse::prepare(
            graph,
            query.clone(),
            identification,
            plan.clone(),
            identifier,
            estimator,
            analysis.inference.clone(),
            analysis.refute,
        )?)))
    }

    pub(super) fn seal_graph_posterior_response(
        analysis: &Study,
        plan: &PhysicalExecutionPlan,
    ) -> Result<Option<Box<crate::analysis::execute::CheckedGraphPosteriorResponse>>, CausalError>
    {
        let (Some(posterior), CausalQuery::Response(query), Some(identification)) = (
            analysis.graph_posterior.as_ref(),
            &analysis.query,
            analysis.graph_posterior_identification_cache.as_deref(),
        ) else {
            return Ok(None);
        };
        let class_atoms = match posterior.atom_kind {
            antecedent_discovery::GraphPosteriorAtomKind::Dag => false,
            antecedent_discovery::GraphPosteriorAtomKind::Cpdag
            | antecedent_discovery::GraphPosteriorAtomKind::Pag => true,
            _ => return Ok(None),
        };
        if analysis.tiered.is_some()
            || !matches!(
                analysis.inference,
                InferenceMode::Frequentist | InferenceMode::Bayesian(_)
            )
            || !analysis.custom_validators.is_empty()
            || !analysis.point_validation()
            || query.temporal.is_some()
            || !complete_mean_response(query)
            || !matches!(
                query.functional,
                antecedent_core::ResponseFunctional::MeanCurve { .. }
                    | antecedent_core::ResponseFunctional::InterventionResponse { .. }
            )
            || (matches!(query.functional, antecedent_core::ResponseFunctional::MeanCurve { .. })
                && analysis.refute != RefuteSuite::None)
        {
            return Ok(None);
        }
        crate::analysis::execute::graph_posterior_response_supported(query)?;
        let expected_estimator =
            EstimatorId::static_response_for(&query.functional, &analysis.inference);
        let expected_identifier = if class_atoms {
            crate::strategy_table::DEFAULT_PAG_IDENTIFIER_ID
        } else {
            crate::strategy_table::DEFAULT_RESPONSE_IDENTIFIER_ID
        };
        let identifier: IdentifierId = plan
            .logical
            .record
            .identifier
            .as_deref()
            .unwrap_or(expected_identifier.as_str())
            .parse()?;
        let estimator: EstimatorId = plan
            .logical
            .record
            .estimator
            .as_deref()
            .unwrap_or(expected_estimator.as_str())
            .parse()?;
        if identifier != expected_identifier
            || estimator != expected_estimator
            || analysis.estimator.is_some_and(|selected| selected != estimator)
            || analysis.estimator_spec.as_ref().is_some_and(|spec| spec.id() != estimator)
        {
            return Ok(None);
        }
        Ok(Some(Box::new(crate::analysis::execute::CheckedGraphPosteriorResponse::prepare(
            posterior.clone(),
            query.clone(),
            identification.clone(),
            plan.clone(),
            identifier,
            estimator,
            analysis.inference.clone(),
            analysis.refute,
            analysis.latency_mode,
        )?)))
    }
}
