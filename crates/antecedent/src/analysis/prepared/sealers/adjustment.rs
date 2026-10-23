//! Sealers of the adjustment-estimator routes: linear, GLM, RD, propensity,
//! conditional, AIPW, front-door, IV, and the Unknown-tier average.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{SealContext, SealOutcome};
use crate::accepted::GraphClass;
use crate::analysis::CheckedConditionalOperation;
use crate::analysis::builder::RefuteSuite;
use crate::analysis::checked_propensity::{
    CheckedPropensityContext, CheckedPropensityEstimator, CheckedPropensityOperation,
};
use crate::analysis::execute::CheckedStaticCommon;
use crate::analysis::execute::RouteInput;
use crate::analysis::prepared::{
    CheckedAipwOperation, CheckedFrontDoorOperation, CheckedGlmAdjustmentOperation,
    CheckedIvOperation, CheckedLinearOperation, CheckedRdOperation,
    CheckedUnknownTieredAverageOperation, PreparedExecution, checked_aipw_fitter,
    conditional_class_proof_is_complete,
};
use crate::analysis::proof_support::EstimandClaim;
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::planner::RouteDefaults;
use crate::strategy_table::{EstimatorId, IdentifierId};
use antecedent_core::{CausalQuery, Intervention, OutcomeFunctional, TargetPopulation};
use antecedent_estimate::OverlapPolicy;
use antecedent_identify::IdentificationStatus;
use std::sync::Arc;

pub(super) fn checked_linear(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { source, analysis, plan, .. } = *cx;
    let sealed = match (
        cx.static_ate_dag(),
        analysis.identification_cache.as_ref(),
        source.graph_posterior.is_none(),
        source.tiered.is_none(),
    ) {
        (Some((data, query)), Some(cache), true, true)
            if crate::analysis::compiled_linear::can_retain(source, query)
                && cx.binding.estimator.is_or(
                    RouteDefaults::STATIC_ATE.estimator,
                    EstimatorId::LinearAdjustmentAte,
                ) =>
        {
            let product = plan.linear.as_ref().ok_or_else(|| CausalError::Compile {
                message: "checked linear execution requires its compiled product".into(),
            })?;
            let (identifier, estimator) = cx.binding.resolve(RouteDefaults::STATIC_ATE)?;
            Some(CheckedLinearOperation {
                frozen_fit: Arc::new(std::sync::Mutex::new(None)),
                compiled: Arc::clone(&product.program),
                source: data.clone(),
                fitter: product.fitter.clone(),
                preparation: product.preparation.clone(),
                default_id: product.default_id,
                query: query.clone(),
                cache: Arc::clone(cache),
                stage_sink: analysis.stage_sink.clone(),
                common: CheckedStaticCommon::from_analysis(analysis, plan, identifier, estimator),
            })
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::CheckedLinear))
}

pub(super) fn checked_glm_adjustment(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, plan, .. } = *cx;
    let sealed = match (
        cx.static_ate_dag(),
        analysis.identification_cache.as_ref(),
        &analysis.inference,
    ) {
        (Some((data, query)), Some(cache), InferenceMode::Frequentist)
            if matches!(query.outcome_functional, OutcomeFunctional::Mean)
                && !matches!(query.active, Intervention::Set { variable, .. } if variable != query.treatment)
                && !matches!(query.control, Intervention::Set { variable, .. } if variable != query.treatment)
                && cx.binding.estimator.is(EstimatorId::GlmAdjustment)
                && analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && matches!(
                    analysis.structure_source,
                    crate::support::StructureSource::Explicit
                        | crate::support::StructureSource::Accepted
                )
                && analysis.custom_validators.is_empty()
                && matches!(
                    analysis.refute,
                    RefuteSuite::None
                        | RefuteSuite::Cheap
                        | RefuteSuite::PlaceboAndRcc
                        | RefuteSuite::Full
                ) =>
        {
            let mut fitter =
                if let Some(crate::estimator_spec::EstimatorSpec::GlmAdjustment(config)) =
                    analysis.estimator_spec.as_deref()
                {
                    (**config).clone()
                } else {
                    let mut fitter = antecedent_estimate::GlmAdjustmentAte::new();
                    fitter.sampling_design =
                        crate::estimator_spec::declared_sampling(analysis.sampling_design);
                    fitter.bootstrap_replicates = analysis.bootstrap_replicates;
                    fitter
                };
            if fitter.population_registry.is_none() {
                fitter.population_registry.clone_from(&analysis.population_registry);
            }
            let preparation = fitter.prepare(data, &cache.estimand, query)?;
            let (identifier, estimator) = cx.binding.resolve(RouteDefaults::STATIC_ATE)?;
            Some(CheckedGlmAdjustmentOperation {
                source_query: CausalQuery::AverageEffect(query.clone()),
                query: query.clone(),
                cache: Arc::clone(cache),
                fitter,
                preparation,
                common: CheckedStaticCommon::from_analysis(analysis, plan, identifier, estimator),
            })
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::CheckedGlmAdjustment))
}

pub(super) fn checked_rd(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, plan, .. } = *cx;
    let sealed = match (cx.route, analysis.rd, &analysis.inference) {
        (
            Some(RouteInput::StaticAteDag { data, query, graph }),
            Some(config),
            InferenceMode::Frequentist,
        ) if cx.binding.estimator.is(EstimatorId::RdSharp)
            && analysis.graph_posterior.is_none()
            && analysis.tiered.is_none()
            && matches!(
                analysis.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && analysis.custom_validators.is_empty()
            && matches!(
                analysis.refute,
                RefuteSuite::None
                    | RefuteSuite::Cheap
                    | RefuteSuite::PlaceboAndRcc
                    | RefuteSuite::Full
            ) =>
        {
            let identification = antecedent_identify::SharpRdIdentifier::new(
                antecedent_identify::SharpRdConfig::new(
                    config.running_variable,
                    config.cutoff,
                    config.bandwidth,
                ),
            )
            .identify_on(graph, CausalQuery::AverageEffect(query.clone()))
            .map_err(CausalError::from)?;
            if matches!(identification.status, IdentificationStatus::NotIdentified)
                || identification.estimands.is_empty()
                || !crate::strategy_table::identification_status_acceptable(identification.status)
            {
                let detail = identification
                    .diagnostics
                    .last()
                    .map_or("effect not identified", |diagnostic| diagnostic.message.as_ref());
                return Err(CausalError::not_identified(
                    identification.status,
                    antecedent_identify::search_truncated(&identification),
                    detail,
                ));
            }
            let estimand = crate::strategy_table::select_estimand(
                &identification,
                crate::strategy_table::EstimatorId::RdSharp,
            )?;
            let identified_query =
                identification.average_effect().cloned().unwrap_or_else(|| query.clone());
            let mut fitter = antecedent_estimate::SharpRegressionDiscontinuity::new(
                config.running_variable,
                config.cutoff,
                config.bandwidth,
            );
            fitter.bootstrap_replicates = analysis.bootstrap_replicates;
            fitter.se_kind = config.se_kind;
            // The design is checked against the data (sharp assignment,
            // rows inside the bandwidth window) when it executes, as the
            // route always did; a prepare on data the design does not fit
            // still succeeds and leaves the click to report that refusal.
            let identifier = crate::strategy_table::IdentifierId::RdSharp;
            let estimator = crate::strategy_table::EstimatorId::RdSharp;
            fitter.prepare_checked(data, &identification, 0).ok().map(|preparation| {
                CheckedRdOperation {
                    source_query: query.clone(),
                    query: identified_query,
                    cache: Arc::new(crate::analysis::prepared::CachedStaticIdentification {
                        identification,
                        estimand,
                    }),
                    fitter,
                    preparation,
                    common: CheckedStaticCommon::from_analysis(
                        analysis, plan, identifier, estimator,
                    ),
                }
            })
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::CheckedRd))
}

pub(super) fn checked_conditional(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, plan, .. } = *cx;
    let sealed = match (cx.conditional(), analysis.graph.class(), &analysis.inference) {
        (Some((data, query)), graph_class, InferenceMode::Frequentist)
            if analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && query.inner.effect_modifiers.len() == 1
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
                && cx.binding.estimator.is_or(
                    RouteDefaults::CONDITIONAL.estimator,
                    EstimatorId::ConditionalLinearAdjustment,
                ) =>
        {
            let class_proof = conditional_class_proof(cx, graph_class)?;
            let target = conditional_class_target(analysis, class_proof.as_ref(), graph_class);
            target
                .map(|cache| {
                    CheckedConditionalOperation::prepare(
                        data,
                        query.clone(),
                        cache,
                        CheckedStaticCommon::from_analysis(
                            analysis,
                            plan,
                            crate::strategy_table::IdentifierId::BackdoorAdjustment,
                            crate::strategy_table::EstimatorId::ConditionalLinearAdjustment,
                        ),
                        class_proof,
                    )
                })
                .transpose()?
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::CheckedConditional))
}

/// The CPDAG or PAG graph proof the conditional route keeps beside its identification, if any.
fn conditional_class_proof(
    cx: &SealContext<'_>,
    graph_class: GraphClass,
) -> Result<Option<crate::analysis::checked_conditional::ConditionalClassProof>, CausalError> {
    let SealContext { analysis, plan, .. } = *cx;
    Ok(match graph_class {
        GraphClass::Cpdag => {
            let (Some(graph), Some(cache)) =
                (analysis.graph.as_cpdag(), analysis.cpdag_identification_cache.as_ref())
            else {
                return Err(CausalError::Compile {
                    message: "conditional CPDAG preparation lost its graph proof".into(),
                });
            };
            Some(crate::analysis::checked_conditional::ConditionalClassProof::Cpdag {
                graph: graph.clone(),
                cache: cache.clone(),
            })
        }
        GraphClass::Pag => {
            let (Some(graph), Some(cache)) =
                (plan.static_pag(), analysis.pag_identification_cache.as_ref())
            else {
                return Err(CausalError::Compile {
                    message: "conditional PAG preparation lost its graph proof".into(),
                });
            };
            Some(crate::analysis::checked_conditional::ConditionalClassProof::Pag {
                graph: graph.clone(),
                cache: cache.clone(),
            })
        }
        _ => None,
    })
}

/// The cached identification a conditional route executes against, `None` when its class proof
/// is incomplete.
fn conditional_class_target(
    analysis: &crate::analysis::execute::Study,
    class_proof: Option<&crate::analysis::checked_conditional::ConditionalClassProof>,
    graph_class: GraphClass,
) -> Option<Arc<crate::analysis::prepared::CachedStaticIdentification>> {
    match (class_proof, graph_class) {
        (None, GraphClass::Dag) => analysis.identification_cache.clone(),
        (
            Some(crate::analysis::checked_conditional::ConditionalClassProof::Cpdag {
                cache, ..
            }),
            GraphClass::Cpdag,
        ) => {
            let estimand = cache.envelope.invariant.as_ref().and_then(|invariant| {
                cache
                    .identification
                    .estimands
                    .iter()
                    .find(|candidate| {
                        candidate.is_adjustment_shaped()
                            && candidate.method == invariant.method
                            && candidate.adjustment_set == invariant.adjustment_set
                    })
                    .cloned()
            });
            (conditional_class_proof_is_complete(&cache.envelope)
                && cache.identification.status
                    == antecedent_identify::IdentificationStatus::NonparametricallyIdentified)
                .then(|| {
                    estimand.map(|estimand| {
                        Arc::new(crate::analysis::prepared::CachedStaticIdentification {
                            identification: cache.identification.clone(),
                            estimand,
                        })
                    })
                })
                .flatten()
        }
        (
            Some(crate::analysis::checked_conditional::ConditionalClassProof::Pag {
                cache, ..
            }),
            GraphClass::Pag,
        ) => {
            let estimand = cache.envelope.invariant.as_ref().and_then(|invariant| {
                cache
                    .identification
                    .estimands
                    .iter()
                    .find(|candidate| {
                        candidate.is_adjustment_shaped()
                            && candidate.method == invariant.method
                            && candidate.adjustment_set == invariant.adjustment_set
                    })
                    .cloned()
            });
            (conditional_class_proof_is_complete(&cache.envelope)
                && cache.identification.status
                    == antecedent_identify::IdentificationStatus::NonparametricallyIdentified)
                .then(|| {
                    estimand.map(|estimand| {
                        Arc::new(crate::analysis::prepared::CachedStaticIdentification {
                            identification: cache.identification.clone(),
                            estimand,
                        })
                    })
                })
                .flatten()
        }
        _ => None,
    }
}

pub(super) fn checked_propensity(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, plan, .. } = *cx;
    let sealed =
        match (cx.static_ate_dag(), analysis.identification_cache.as_ref(), &analysis.inference) {
            (Some((data, query)), Some(cache), InferenceMode::Frequentist)
                if matches!(query.outcome_functional, OutcomeFunctional::Mean)
                    && analysis.graph_posterior.is_none()
                    && analysis.tiered.is_none()
                    && matches!(
                        analysis.structure_source,
                        crate::support::StructureSource::Explicit
                            | crate::support::StructureSource::Accepted
                    )
                    && analysis.custom_validators.is_empty()
                    && matches!(
                        analysis.refute,
                        RefuteSuite::None
                            | RefuteSuite::Cheap
                            | RefuteSuite::PlaceboAndRcc
                            | RefuteSuite::Full
                    )
                    && matches!(
                        cx.binding.estimator.named(),
                        Some(EstimatorId::PropensityWeighting | EstimatorId::PropensityMatching)
                    )
                    && !analysis.estimator_spec.as_deref().is_some_and(
                        crate::estimator_spec::EstimatorSpec::excludes_undeclared_levels,
                    ) =>
            {
                let estimator = cx.binding.estimator.named().expect("checked propensity estimator");
                let identifier = cx.binding.identifier_or(RouteDefaults::STATIC_ATE)?;
                let estimand_index = cache
                    .identification
                    .estimands
                    .iter()
                    .position(|candidate| candidate.same_claim(&cache.estimand))
                    .ok_or_else(|| CausalError::Compile {
                        message:
                            "selected propensity estimand is absent from its identification product"
                                .into(),
                    })?;
                let fitter = match (analysis.estimator_spec.as_deref(), estimator) {
                    (
                        Some(crate::estimator_spec::EstimatorSpec::PropensityWeighting(config)),
                        crate::strategy_table::EstimatorId::PropensityWeighting,
                    ) => CheckedPropensityEstimator::Weighting((**config).clone()),
                    (
                        Some(crate::estimator_spec::EstimatorSpec::PropensityMatching(config)),
                        crate::strategy_table::EstimatorId::PropensityMatching,
                    ) => CheckedPropensityEstimator::Matching((**config).clone()),
                    (_, crate::strategy_table::EstimatorId::PropensityWeighting) => {
                        let mut fitter = antecedent_estimate::PropensityWeighting::new();
                        fitter.sampling_design =
                            crate::estimator_spec::declared_sampling(analysis.sampling_design);
                        fitter.bootstrap_replicates = analysis.bootstrap_replicates;
                        CheckedPropensityEstimator::Weighting(fitter)
                    }
                    (_, crate::strategy_table::EstimatorId::PropensityMatching) => {
                        let mut fitter = antecedent_estimate::PropensityMatching::new();
                        fitter.bootstrap_replicates = analysis.bootstrap_replicates;
                        CheckedPropensityEstimator::Matching(fitter)
                    }
                    _ => unreachable!(),
                };
                let context = CheckedPropensityContext {
                    source_query: analysis.query.clone(),
                    cache: Arc::clone(cache),
                    estimand_index,
                    common: CheckedStaticCommon::from_analysis(
                        analysis, plan, identifier, estimator,
                    ),
                };
                Some(CheckedPropensityOperation::prepare(data, context, estimator, fitter)?)
            }
            _ => None,
        };
    Ok(sealed.map(PreparedExecution::CheckedPropensity))
}

pub(super) fn checked_aipw(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { source, analysis, plan, .. } = *cx;
    let sealed = match (
        cx.route,
        analysis.identification_cache.as_ref(),
        &analysis.inference,
        source.graph.class(),
        source.graph_posterior.is_none(),
        source.tiered.as_ref(),
    ) {
        (
            Some(
                RouteInput::StaticAteDag { data, query, .. }
                | RouteInput::StaticAteMixed { data, query },
            ),
            Some(cache),
            InferenceMode::Frequentist,
            graph_class,
            true,
            tiered_background,
        ) if matches!(
            (graph_class, tiered_background),
            (GraphClass::Dag, None) | (GraphClass::Admg, Some(_))
        ) && tiered_background.is_none_or(|background| {
            background.within_tier == antecedent_graph::WithinTier::CoDetermined
        }) && matches!(query.outcome_functional, OutcomeFunctional::Mean)
            && matches!(query.target_population, TargetPopulation::AllObserved)
            && crate::analysis::route_guards::distinct_numeric_arms(query)
            && !analysis
                .estimator_spec
                .as_deref()
                .is_some_and(crate::estimator_spec::EstimatorSpec::excludes_undeclared_levels)
            && cx.binding.estimator.is(EstimatorId::Aipw) =>
        {
            let fitter = checked_aipw_fitter(analysis);
            if matches!(fitter.overlap, OverlapPolicy::RequireDiagnostics { .. }) {
                let preparation = fitter.prepare_checked(data, &cache.identification, 0)?;
                let (identifier, estimator) = cx.binding.resolve(RouteDefaults::STATIC_ATE)?;
                Some(CheckedAipwOperation {
                    fitter,
                    preparation,
                    query: query.clone(),
                    cache: Arc::clone(cache),
                    common: CheckedStaticCommon::from_analysis(
                        analysis, plan, identifier, estimator,
                    ),
                    tiered_background: tiered_background.cloned(),
                })
            } else {
                None
            }
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::CheckedAipw))
}

pub(super) fn frontdoor_linear(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, .. } = *cx;
    let sealed =
        match (cx.static_ate_dag(), analysis.identification_cache.as_ref(), &analysis.inference) {
            (Some((data, query)), Some(cache), InferenceMode::Frequentist)
                if matches!(query.outcome_functional, OutcomeFunctional::Mean)
                    && matches!(query.target_population, TargetPopulation::AllObserved)
                    && cx.binding.estimator.is(EstimatorId::FrontDoorTwoStage) =>
            {
                let fitter =
                    if let Some(crate::estimator_spec::EstimatorSpec::FrontDoorTwoStage(config)) =
                        analysis.estimator_spec.as_deref()
                    {
                        (**config).clone()
                    } else {
                        let mut fitter = antecedent_estimate::FrontDoorTwoStage::new();
                        fitter.sampling_design =
                            crate::estimator_spec::declared_sampling(analysis.sampling_design);
                        fitter
                    };
                let preparation = fitter.prepare_checked(data, &cache.identification, 0)?;
                Some(CheckedFrontDoorOperation { fitter, preparation })
            }
            _ => None,
        };
    Ok(sealed.map(PreparedExecution::FrontDoorLinear))
}

pub(super) fn iv(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, .. } = *cx;
    let sealed = match (
        cx.static_ate_dag(),
        analysis.identification_cache.as_ref(),
        &analysis.inference,
    ) {
        (Some((data, _)), Some(cache), InferenceMode::Frequentist) => {
            match (analysis.estimator_spec.as_deref(), cx.binding.estimator.named()) {
                (
                    Some(crate::estimator_spec::EstimatorSpec::IvWald(cfg)),
                    Some(EstimatorId::IvWald),
                ) => Some(CheckedIvOperation::Wald {
                    fitter: (**cfg).clone(),
                    preparation: (**cfg).prepare_checked(data, &cache.identification, 0)?,
                }),
                (
                    None | Some(crate::estimator_spec::EstimatorSpec::Default(EstimatorId::IvWald)),
                    Some(EstimatorId::IvWald),
                ) => {
                    let fitter = antecedent_estimate::WaldIv::new();
                    let preparation = fitter.prepare_checked(data, &cache.identification, 0)?;
                    Some(CheckedIvOperation::Wald { fitter, preparation })
                }
                (
                    Some(crate::estimator_spec::EstimatorSpec::Iv2Sls(cfg)),
                    Some(EstimatorId::Iv2Sls),
                ) => match (**cfg).prepare_checked(data, &cache.identification, 0) {
                    Ok(preparation) => {
                        Some(CheckedIvOperation::TwoSls { fitter: (**cfg).clone(), preparation })
                    }
                    Err(antecedent_estimate::EstimationError::Unsupported {
                        message:
                            "checked IV currently requires a binary 0/1 instrument matching the checked Wald functional",
                    }) => None,
                    Err(error) => return Err(error.into()),
                },
                (
                    None | Some(crate::estimator_spec::EstimatorSpec::Default(EstimatorId::Iv2Sls)),
                    Some(EstimatorId::Iv2Sls),
                ) => {
                    let fitter = antecedent_estimate::TwoStageLeastSquares::new();
                    match fitter.prepare_checked(data, &cache.identification, 0) {
                        Ok(preparation) => Some(CheckedIvOperation::TwoSls { fitter, preparation }),
                        Err(antecedent_estimate::EstimationError::Unsupported {
                            message:
                                "checked IV currently requires a binary 0/1 instrument matching the checked Wald functional",
                        }) => None,
                        Err(error) => return Err(error.into()),
                    }
                }
                _ => None,
            }
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::Iv))
}

#[allow(
    clippy::unnecessary_wraps,
    reason = "every sealer shares the table's fallible `SealOutcome` signature"
)]
pub(super) fn unknown_tiered_average(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, plan, .. } = *cx;
    let sealed = match (
        cx.static_ate(),
        analysis.tiered.as_ref(),
        analysis.identification_cache.as_ref(),
        &analysis.inference,
    ) {
        (Some(query), Some(background), Some(cache), inference)
            if background.within_tier == antecedent_graph::WithinTier::Unknown
                && analysis.structure_source == crate::support::StructureSource::Explicit
                && analysis.graph_posterior.is_none()
                && analysis.custom_validators.is_empty()
                && matches!(analysis.refute, RefuteSuite::None)
                && cx.binding.resolves_to(
                    RouteDefaults::STATIC_ATE,
                    IdentifierId::GeneralizedAdjustment,
                    match inference {
                        InferenceMode::Frequentist => EstimatorId::LinearAdjustmentAte,
                        InferenceMode::Bayesian(_) => EstimatorId::BayesianGcomp,
                    },
                )
                && cache.identification.query == CausalQuery::AverageEffect(query.clone())
                && cache.identification.estimands.len() == 2 =>
        {
            Some(Box::new(CheckedUnknownTieredAverageOperation {
                query: query.clone(),
                background: background.clone(),
                cache: Arc::clone(cache),
                identifier: crate::strategy_table::IdentifierId::GeneralizedAdjustment,
                estimator: match inference {
                    InferenceMode::Frequentist => {
                        crate::strategy_table::EstimatorId::LinearAdjustmentAte
                    }
                    InferenceMode::Bayesian(_) => crate::strategy_table::EstimatorId::BayesianGcomp,
                },
                physical: plan.clone(),
            }))
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::UnknownTieredAverage))
}
