//! Sealing of graph-posterior, class and temporal operations at prepare time.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::accepted::{GraphClass, GraphRef};
use crate::analysis::builder::{DataInput, RefuteSuite};
use crate::analysis::execute::Study;
use crate::analysis::route_guards::{complete_mean_response, mean_all_observed};
use crate::analysis::{
    CheckedAdmgGraphPosteriorResponse, CheckedBayesianGraphPosteriorAte,
    CheckedClassGraphPosteriorEffect, CheckedGraphPosteriorEffect, CheckedStaticClassEffect,
    StaticClassGraph, StaticClassIdentification,
};
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::planner::{PhysicalExecutionPlan, RouteDefaults};
use crate::strategy_table::{EstimatorId, IdentifierId};
use antecedent_core::{AverageEffectQuery, CausalQuery};
use antecedent_estimate::OverlapPolicy;
use std::sync::Arc;

impl Study {
    pub(super) fn seal_static_class_effect(
        analysis: &Study,
        plan: &Arc<PhysicalExecutionPlan>,
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
            || !plan.binding().names(IdentifierId::GeneralizedAdjustment, expected_estimator)
        {
            return Ok(None);
        }
        let (graph, identification) = match analysis.graph.graph_ref() {
            GraphRef::Cpdag(graph) => {
                let Some(cache) = analysis.cpdag_identification_cache.as_ref() else {
                    return Ok(None);
                };
                (
                    StaticClassGraph::Cpdag(graph.clone()),
                    StaticClassIdentification::Cpdag(cache.clone()),
                )
            }
            GraphRef::Pag(_) => {
                let (Some(graph), Some(cache)) =
                    (plan.static_pag(), analysis.pag_identification_cache.as_ref())
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
            .as_deref()
            .cloned()
            .unwrap_or(crate::EstimatorSpec::Default(expected_estimator));
        Ok(Some(CheckedStaticClassEffect::prepare(
            crate::analysis::CheckedStaticClassEffectPremises {
                graph,
                query: query.clone(),
                identification,
                physical: plan.clone(),
                procedure,
                inference: analysis.inference.clone(),
                bootstrap_replicates: analysis.bootstrap_replicates,
                overlap: analysis.overlap_policy.unwrap_or(OverlapPolicy::ExplicitOverride),
                validation: analysis.refute,
            },
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
        analysis.estimator_spec.as_deref().is_none_or(|spec| spec.id() == estimator)
            && analysis.estimator.is_none_or(|id| id == estimator)
    }

    #[inline(never)]
    pub(super) fn seal_graph_posterior_effect(
        analysis: &Study,
        plan: &Arc<PhysicalExecutionPlan>,
    ) -> Result<Option<CheckedGraphPosteriorEffect>, CausalError> {
        let (Some(posterior), Some(identification), Some(target)) = (
            analysis.graph_posterior.as_ref(),
            analysis.graph_posterior_identification_cache.as_ref(),
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
                    || !plan.binding().resolves_to(
                        RouteDefaults::STATIC_ATE,
                        IdentifierId::BackdoorAdjustment,
                        estimator,
                    )
                {
                    return Ok(None);
                }
                Ok(Some(CheckedGraphPosteriorEffect::prepare(
                    crate::analysis::CheckedGraphPosteriorEffectPremises {
                        posterior: posterior.clone(),
                        target,
                        identification: identification.clone(),
                        procedure: analysis
                            .estimator_spec
                            .as_deref()
                            .cloned()
                            .unwrap_or(crate::EstimatorSpec::Default(estimator)),
                        bootstrap_replicates: analysis.bootstrap_replicates,
                        overlap: analysis.overlap_policy.unwrap_or(OverlapPolicy::ExplicitOverride),
                        population_registry: analysis.population_registry.clone(),
                        latency_mode: analysis.latency_mode,
                        validation: analysis.refute,
                    },
                )?))
            }
            (
                antecedent_discovery::GraphPosteriorAtomKind::Admg
                | antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg,
                InferenceMode::Frequentist | InferenceMode::Bayesian(_),
            ) if !target.is_conditional()
                && plan.binding().resolves_to(
                    RouteDefaults::ADMG,
                    IdentifierId::GeneralId,
                    EstimatorId::FunctionalEffect,
                ) =>
            {
                Ok(Some(CheckedGraphPosteriorEffect::prepare(
                    crate::analysis::CheckedGraphPosteriorEffectPremises {
                        posterior: posterior.clone(),
                        target,
                        identification: identification.clone(),
                        procedure: analysis.estimator_spec.as_deref().cloned().unwrap_or(
                            crate::EstimatorSpec::Default(
                                crate::strategy_table::EstimatorId::FunctionalEffect,
                            ),
                        ),
                        bootstrap_replicates: analysis.bootstrap_replicates,
                        overlap: analysis.overlap_policy.unwrap_or(OverlapPolicy::ExplicitOverride),
                        population_registry: analysis.population_registry.clone(),
                        latency_mode: analysis.latency_mode,
                        validation: analysis.refute,
                    },
                )?))
            }
            _ => Ok(None),
        }
    }

    #[inline(never)]
    pub(super) fn seal_class_graph_posterior_effect(
        analysis: &Study,
        plan: &Arc<PhysicalExecutionPlan>,
    ) -> Result<Option<CheckedClassGraphPosteriorEffect>, CausalError> {
        let (Some(posterior), Some(identification), Some(target)) = (
            analysis.graph_posterior.as_ref(),
            analysis.graph_posterior_identification_cache.as_ref(),
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
        if !plan.binding().names(identifier, estimator) {
            return Ok(None);
        }
        Ok(Some(CheckedClassGraphPosteriorEffect::prepare(
            crate::analysis::CheckedClassGraphPosteriorEffectPremises {
                posterior: posterior.clone(),
                target,
                identification: identification.clone(),
                inference: analysis.inference.clone(),
                procedure: analysis
                    .estimator_spec
                    .as_deref()
                    .cloned()
                    .unwrap_or(crate::EstimatorSpec::Default(estimator)),
                bootstrap_replicates: analysis.bootstrap_replicates,
                overlap: analysis.overlap_policy.unwrap_or(OverlapPolicy::ExplicitOverride),
                population_registry: analysis.population_registry.clone(),
                latency_mode: analysis.latency_mode,
                validation: analysis.refute,
            },
        )?))
    }

    #[inline(never)]
    pub(super) fn seal_bayesian_graph_posterior_ate(
        analysis: &Study,
        plan: &Arc<PhysicalExecutionPlan>,
    ) -> Result<Option<Box<CheckedBayesianGraphPosteriorAte>>, CausalError> {
        let (Some(posterior), Some(identification), Some(target), InferenceMode::Bayesian(_)) = (
            analysis.graph_posterior.as_ref(),
            analysis.graph_posterior_identification_cache.as_ref(),
            crate::analysis::GraphPosteriorEffectTarget::from_query(&analysis.query),
            &analysis.inference,
        ) else {
            return Ok(None);
        };
        let estimator = target.bayesian_estimator();
        if posterior.atom_kind != antecedent_discovery::GraphPosteriorAtomKind::Dag
            || !Self::graph_posterior_target_is_sealable(analysis, target.inner())
            || !Self::graph_posterior_procedure_is(analysis, estimator)
            || !plan.binding().resolves_to(
                RouteDefaults::STATIC_ATE,
                IdentifierId::BackdoorAdjustment,
                estimator,
            )
        {
            return Ok(None);
        }
        let procedure = analysis
            .estimator_spec
            .as_deref()
            .cloned()
            .unwrap_or(crate::EstimatorSpec::Default(estimator));
        Ok(Some(Box::new(CheckedBayesianGraphPosteriorAte::prepare(
            crate::analysis::CheckedBayesianGraphPosteriorAtePremises {
                posterior: posterior.clone(),
                target,
                identification: identification.clone(),
                inference: analysis.inference.clone(),
                procedure,
                physical: plan.clone(),
                validation: analysis.refute,
                overlap: analysis.overlap_policy.unwrap_or(OverlapPolicy::ExplicitOverride),
                latency_mode: analysis.latency_mode,
            },
        )?)))
    }

    pub(super) fn seal_admg_graph_posterior_response(
        analysis: &Study,
        plan: &Arc<PhysicalExecutionPlan>,
    ) -> Result<Option<Box<CheckedAdmgGraphPosteriorResponse>>, CausalError> {
        let (Some(posterior), CausalQuery::Response(query), Some(identification)) = (
            analysis.graph_posterior.as_ref(),
            &analysis.query,
            analysis.graph_posterior_identification_cache.as_ref(),
        ) else {
            return Ok(None);
        };
        if !matches!(
            posterior.atom_kind,
            antecedent_discovery::GraphPosteriorAtomKind::Admg
                | antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg
        ) || !analysis.custom_validators.is_empty()
            || !matches!(
                analysis.estimator_spec.as_deref().map(crate::EstimatorSpec::id),
                None | Some(crate::strategy_table::EstimatorId::FunctionalEffect)
            )
            || !matches!(
                analysis.estimator,
                None | Some(crate::strategy_table::EstimatorId::FunctionalEffect)
            )
            || !plan.binding().resolves_to(
                RouteDefaults::ADMG,
                IdentifierId::GeneralId,
                EstimatorId::FunctionalEffect,
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
        plan: &Arc<PhysicalExecutionPlan>,
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
            analysis.temporal_class_identification_cache.as_ref(),
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
            crate::analysis::CheckedTemporalClassResponseOperationPremises {
                source_graph: (*analysis.graph).clone(),
                query,
                bundle: cache,
                max_completions: analysis.max_completions,
                inference: &analysis.inference,
                bootstrap_replicates: analysis.bootstrap_replicates,
                class_prior: analysis.class_prior.clone(),
                validation: analysis.refute,
                physical: plan.clone(),
            },
        )?)))
    }

    /// Seal a temporal graph-posterior response over its frozen atoms and
    /// per-atom proofs (`TemporalDag`, `TemporalCpdag`, or `TemporalPag` atoms).
    pub(super) fn seal_temporal_graph_posterior_response(
        analysis: &Study,
        plan: &Arc<PhysicalExecutionPlan>,
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
            crate::analysis::CheckedTemporalGraphPosteriorResponsePremises {
                posterior: posterior.clone(),
                query: query.clone(),
                proof,
                physical: plan.clone(),
                inference: analysis.inference.clone(),
                validation: analysis.refute,
                bootstrap_replicates: analysis.bootstrap_replicates,
                max_completions: analysis.max_completions,
                class_prior: analysis.class_prior.clone(),
            },
        )?)))
    }

    #[inline(never)]
    pub(super) fn seal_bayesian_temporal_class_effect(
        analysis: &Study,
        plan: &Arc<PhysicalExecutionPlan>,
        ctx: &antecedent_core::ExecutionContext,
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
            analysis.temporal_class_identification_cache.as_ref(),
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
        if !plan.binding().names(identifier, estimator)
            || !analysis.estimator_spec.as_deref().is_none_or(|spec| spec.id() == estimator)
        {
            return Ok(None);
        }
        let target = crate::analysis::CheckedTemporalClassEffectOperation::checked(
            crate::analysis::CheckedTemporalClassEffectOperationPremises {
                ctx,
                source_graph: (*analysis.graph).clone(),
                query,
                bundle: cache,
                max_completions: analysis.max_completions,
                identifier,
                estimator: procedure,
                bootstrap_replicates: analysis.bootstrap_replicates,
                split: analysis.split,
                refute: analysis.refute,
                custom_validators: Vec::new(),
            },
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
        plan: &Arc<PhysicalExecutionPlan>,
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
        if !plan.binding().names(identifier, estimator)
            || !analysis.estimator_spec.as_deref().is_none_or(|spec| spec.id() == estimator)
        {
            return Ok(None);
        }
        Ok(Some(Box::new(crate::analysis::CheckedTemporalGraphPosteriorEffect::prepare(
            crate::analysis::CheckedTemporalGraphPosteriorEffectPremises {
                posterior: posterior.clone(),
                query: query.clone(),
                proof,
                inference: analysis.inference.clone(),
                estimator_spec: analysis.estimator_spec.as_deref(),
                bootstrap_replicates: analysis.bootstrap_replicates,
                split: analysis.split,
                max_completions: analysis.max_completions,
                validation: analysis.refute,
                physical: plan.clone(),
            },
        )?)))
    }

    pub(super) fn seal_static_class_response(
        analysis: &Study,
        plan: &Arc<PhysicalExecutionPlan>,
    ) -> Result<Option<Box<crate::analysis::execute::CheckedStaticClassResponse>>, CausalError>
    {
        let CausalQuery::Response(query) = &analysis.query else {
            return Ok(None);
        };
        if !crate::analysis::execute::CheckedStaticClassResponse::admits(analysis) {
            return Ok(None);
        }
        let (graph, identification) = match analysis.graph.graph_ref() {
            GraphRef::Cpdag(graph) => {
                let Some(cache) = analysis.cpdag_identification_cache.as_ref() else {
                    return Ok(None);
                };
                (
                    StaticClassGraph::Cpdag(graph.clone()),
                    StaticClassIdentification::Cpdag(cache.clone()),
                )
            }
            GraphRef::Pag(_) => {
                let (Some(graph), Some(cache)) =
                    (plan.static_pag(), analysis.pag_identification_cache.as_ref())
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
        let (identifier, estimator) = plan
            .binding()
            .resolve(RouteDefaults { estimator: expected_estimator, ..RouteDefaults::PAG })?;
        if identifier != IdentifierId::GeneralizedAdjustment
            || estimator != expected_estimator
            || analysis.estimator.is_some_and(|selected| selected != estimator)
            || analysis.estimator_spec.as_deref().is_some_and(|spec| spec.id() != estimator)
        {
            return Ok(None);
        }
        Ok(Some(Box::new(crate::analysis::execute::CheckedStaticClassResponse::prepare(
            crate::analysis::execute::CheckedStaticClassResponsePremises {
                graph,
                query: query.clone(),
                identification,
                physical: plan.clone(),
                identifier,
                estimator,
                inference: analysis.inference.clone(),
                validation: analysis.refute,
            },
        )?)))
    }

    pub(super) fn seal_graph_posterior_response(
        analysis: &Study,
        plan: &Arc<PhysicalExecutionPlan>,
    ) -> Result<Option<Box<crate::analysis::execute::CheckedGraphPosteriorResponse>>, CausalError>
    {
        let (Some(posterior), CausalQuery::Response(query), Some(identification)) = (
            analysis.graph_posterior.as_ref(),
            &analysis.query,
            analysis.graph_posterior_identification_cache.as_ref(),
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
            RouteDefaults::PAG.identifier
        } else {
            RouteDefaults::RESPONSE.identifier
        };
        let (identifier, estimator) = plan.binding().resolve(RouteDefaults {
            identifier: expected_identifier,
            estimator: expected_estimator,
        })?;
        if identifier != expected_identifier
            || estimator != expected_estimator
            || analysis.estimator.is_some_and(|selected| selected != estimator)
            || analysis.estimator_spec.as_deref().is_some_and(|spec| spec.id() != estimator)
        {
            return Ok(None);
        }
        Ok(Some(Box::new(crate::analysis::execute::CheckedGraphPosteriorResponse::prepare(
            crate::analysis::execute::CheckedGraphPosteriorResponsePremises {
                posterior: posterior.clone(),
                query: query.clone(),
                identification: identification.clone(),
                physical: plan.clone(),
                identifier,
                estimator,
                inference: analysis.inference.clone(),
                validation: analysis.refute,
                latency_mode: analysis.latency_mode,
            },
        )?)))
    }
}
