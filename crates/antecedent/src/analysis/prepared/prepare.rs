//! `Study::prepare`: compile once and seal the execution route.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CachedTemporalClassIdentification, CheckedAdmgResponseCurveOperation, CheckedAipwOperation,
    CheckedBayesianGcompOperation, CheckedDistributionOperation, CheckedFrontDoorOperation,
    CheckedFunctionalEffectOperation, CheckedFunctionalEffectResponseMember,
    CheckedGlmAdjustmentOperation, CheckedIvOperation, CheckedLinearOperation,
    CheckedPathSpecificEffectOperation, CheckedRdOperation, CheckedStaticResponseCurve,
    CheckedUnknownTieredAverageOperation, PreparedExecution, PreparedModality, PreparedStudy,
    SampledPreparedState, build_admg_graph_posterior_response_identification_cache,
    build_dbn_posterior_identification_cache, build_dbn_posterior_mediation_identification_cache,
    build_dbn_posterior_response_identification_cache, build_graph_posterior_identification_cache,
    build_temporal_class_posterior_identification_cache, checked_aipw_fitter,
    conditional_class_proof_is_complete, ensure_prepared_supported, multi_env_regularity,
    single_step_dose,
};
use crate::accepted::GraphClass;
use crate::analysis::CheckedConditionalOperation;
use crate::analysis::builder::{DataInput, RefuteSuite};
use crate::analysis::checked_propensity::{
    CheckedPropensityContext, CheckedPropensityEstimator, CheckedPropensityOperation,
};
use crate::analysis::execute::{CheckedStaticCommon, Study};
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::strategy_table::EstimatorId;
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, Intervention, OutcomeFunctional,
    TargetPopulation, TemporalEffectQuery, Value,
};
use antecedent_data::TableView;
use antecedent_estimate::OverlapPolicy;
use antecedent_identify::IdentificationStatus;
use std::sync::Arc;

impl Study {
    /// Compile once into a durable [`PreparedStudy`] for re-estimate-many.
    ///
    /// Supports:
    /// - tabular [`CausalQuery::AverageEffect`] on a supplied static graph
    ///   ([`GraphClass::Dag`], [`GraphClass::Cpdag`], [`GraphClass::Pag`], or
    ///   [`GraphClass::Admg`])
    /// - tabular [`CausalQuery::AverageEffect`], [`CausalQuery::ConditionalEffect`],
    ///   and static [`CausalQuery::Response`] on a supplied DAG graph posterior
    /// - tabular [`CausalQuery::Response`] and [`CausalQuery::ConditionalEffect`]
    ///   on a supplied [`GraphClass::Dag`], [`GraphClass::Cpdag`],
    ///   [`GraphClass::Pag`], or [`GraphClass::Admg`] (joint Response also on a
    ///   CoDetermined tier closure)
    /// - tabular [`CausalQuery::Distribution`] on a supplied [`GraphClass::Dag`]
    ///   or [`GraphClass::Admg`] (ADMG: unconditional, validation none)
    /// - tabular [`CausalQuery::PathSpecific`], static [`CausalQuery::Mediation`],
    ///   and [`CausalQuery::Counterfactual`] on a supplied [`GraphClass::Dag`]
    /// - series temporal [`CausalQuery::Response`] on a supplied
    ///   [`GraphClass::TemporalDag`], [`GraphClass::TemporalCpdag`], or
    ///   [`GraphClass::TemporalPag`]
    /// - series [`CausalQuery::TemporalEffect`] (Pulse / single-step Sustained)
    ///   on a supplied [`GraphClass::TemporalDag`], [`GraphClass::TemporalCpdag`],
    ///   or [`GraphClass::TemporalPag`]
    /// - series [`CausalQuery::TemporalEffect`] (Pulse / single-step or
    ///   multi-step Sustained) on a supplied DBN graph posterior
    /// - series temporal [`CausalQuery::Response`] on a supplied DBN graph
    ///   posterior (TemporalDag atoms; MeanCurve none, one-coordinate
    ///   InterventionResponse none/cheap/full)
    /// - scalar-horizon series [`CausalQuery::Mediation`] (`TemporalMediationEffect`) on a
    ///   supplied [`GraphClass::TemporalDag`] or [`GraphClass::TemporalCpdag`]
    ///   (horizon-specific `I(h)`)
    /// - scalar-horizon series [`CausalQuery::Mediation`] (`TemporalMediationEffect`) on a
    ///   supplied DBN graph posterior (per-atom `I(h)`, unidentified mass retained)
    /// - panel [`CausalQuery::TemporalEffect`] (Pulse / Sustained) on a supplied
    ///   [`GraphClass::TemporalDag`]; every unit must share one time-index regularity
    ///
    /// The handle records the prepare-time data modality (tabular, series, or
    /// panel) and its estimate / refresh entry points refuse other modalities.
    /// Discovery inputs and review-required compiles are refused.
    ///
    /// # Errors
    ///
    /// Unsupported combination, compile failure, or review-required plan.
    ///
    /// # Panics
    ///
    /// Panics only if an internal route invariant that preparation validates
    /// is violated while sealing the execution plan.
    pub fn prepare(&self, ctx: &ExecutionContext) -> Result<PreparedStudy, CausalError> {
        ensure_prepared_supported(self)?;
        let plan = self.compile(ctx)?;
        let counterfactual = match (&self.graph, &self.query) {
            (graph, CausalQuery::Counterfactual(_))
                if matches!(
                    graph.class(),
                    GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag
                ) =>
            {
                None
            }
            (_, CausalQuery::Counterfactual(query)) => {
                let graph = self
                    .graph
                    .as_dag()
                    .ok_or(CausalError::Unsupported { message: "counterfactual requires Dag" })?;
                Some(crate::gcm::CheckedCounterfactualOperation::compile(
                    graph.clone(),
                    query.clone(),
                )?)
            }
            _ => None,
        };
        let nested_counterfactual = match (&self.graph, &self.query) {
            (graph, CausalQuery::NestedCounterfactual(_))
                if matches!(
                    graph.class(),
                    GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag
                ) =>
            {
                None
            }
            (_, CausalQuery::NestedCounterfactual(query)) => {
                let graph = self.graph.as_dag().ok_or(crate::unsupported_reason!(
                    "cross_world_not_identified",
                    "natural direct effect requires the licensed three-node DAG"
                ))?;
                Some(crate::gcm::NestedCounterfactualOperation::compile(graph.clone(), *query)?)
            }
            _ => None,
        };
        let (schema, modality, time_regularity) = match &self.data {
            DataInput::Tabular(data) => (data.schema().clone(), PreparedModality::Tabular, None),
            DataInput::Temporal(data) | DataInput::Event(data) => (
                data.schema().clone(),
                PreparedModality::Series,
                Some(data.time_index().regularity.clone()),
            ),
            DataInput::Panel(panel) => (
                panel.schema().clone(),
                PreparedModality::Panel,
                Some(crate::analysis::builder::panel_shared_regularity(panel)?),
            ),
            DataInput::MultiEnv(multi) => (
                multi.schema().clone(),
                PreparedModality::MultiEnv,
                Some(multi_env_regularity(multi)?),
            ),
        };
        let mut analysis = self.clone();
        match (&self.data, &self.query, self.graph_posterior.as_ref()) {
            (DataInput::Tabular(_), CausalQuery::AverageEffect(query), Some(posterior)) => {
                analysis.graph_posterior_identification_cache = Some(Arc::new(
                    build_graph_posterior_identification_cache(posterior, query, ctx)?,
                ));
            }
            (DataInput::Tabular(_), CausalQuery::ConditionalEffect(query), Some(posterior)) => {
                analysis.graph_posterior_identification_cache = Some(Arc::new(
                    build_graph_posterior_identification_cache(posterior, &query.inner, ctx)?,
                ));
            }
            (DataInput::Tabular(_), CausalQuery::Response(query), Some(posterior))
                if !query.is_temporal() =>
            {
                analysis.graph_posterior_identification_cache = Some(Arc::new(
                    if matches!(
                        posterior.atom_kind,
                        antecedent_discovery::GraphPosteriorAtomKind::Admg
                            | antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg
                    ) {
                        build_admg_graph_posterior_response_identification_cache(
                            posterior, query, ctx,
                        )?
                    } else {
                        let (treatment, outcome) =
                            query.functional.primary_pair().ok_or_else(|| {
                                CausalError::Compile {
                                    message: "response query has no treatment/outcome pair".into(),
                                }
                            })?;
                        build_graph_posterior_identification_cache(
                            posterior,
                            &AverageEffectQuery::binary_ate(treatment, outcome),
                            ctx,
                        )?
                    },
                ));
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(query),
                Some(posterior),
            ) => {
                let variables: Vec<_> =
                    data.schema().variables().iter().map(|variable| variable.id).collect();
                if matches!(
                    posterior.atom_kind,
                    antecedent_discovery::GraphPosteriorAtomKind::Cpdag
                        | antecedent_discovery::GraphPosteriorAtomKind::Pag
                ) {
                    analysis.temporal_class_posterior_identification_cache =
                        Some(Arc::new(build_temporal_class_posterior_identification_cache(
                            posterior,
                            &variables,
                            query,
                            analysis.max_completions,
                            ctx,
                        )?));
                } else {
                    analysis.dbn_posterior_identification_cache =
                        Some(Arc::new(build_dbn_posterior_identification_cache(
                            posterior, &variables, query, ctx,
                        )?));
                }
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::Mediation(query),
                Some(posterior),
            ) => {
                let variables: Vec<_> =
                    data.schema().variables().iter().map(|variable| variable.id).collect();
                if matches!(
                    posterior.atom_kind,
                    antecedent_discovery::GraphPosteriorAtomKind::Cpdag
                        | antecedent_discovery::GraphPosteriorAtomKind::Pag
                ) {
                    let horizon =
                        query.horizons.first().copied().ok_or_else(|| CausalError::Compile {
                            message: "temporal class graph-posterior mediation requires a horizon"
                                .into(),
                        })?;
                    let mut witness =
                        TemporalEffectQuery::pulse(query.treatment, query.outcome, 1.0);
                    witness.horizon_steps = horizon;
                    analysis.temporal_class_posterior_identification_cache =
                        Some(Arc::new(build_temporal_class_posterior_identification_cache(
                            posterior,
                            &variables,
                            &witness,
                            analysis.max_completions,
                            ctx,
                        )?));
                } else {
                    analysis.dbn_posterior_identification_cache =
                        Some(Arc::new(build_dbn_posterior_mediation_identification_cache(
                            posterior, &variables, query, ctx,
                        )?));
                }
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::Response(query),
                Some(posterior),
            ) if query.is_temporal() => {
                crate::analysis::execute::dbn_posterior_response_supported(query)?;
                let variables: Vec<_> =
                    data.schema().variables().iter().map(|variable| variable.id).collect();
                if matches!(
                    posterior.atom_kind,
                    antecedent_discovery::GraphPosteriorAtomKind::Cpdag
                        | antecedent_discovery::GraphPosteriorAtomKind::Pag
                ) {
                    let (treatment, outcome) =
                        query.functional.primary_pair().ok_or_else(|| CausalError::Compile {
                            message: "temporal class graph-posterior response has no \
                                      treatment/outcome pair"
                                .into(),
                        })?;
                    let horizon = query
                        .temporal
                        .as_ref()
                        .and_then(|spec| spec.horizons.first())
                        .copied()
                        .unwrap_or(1);
                    let mut witness = TemporalEffectQuery::pulse(treatment, outcome, 1.0);
                    witness.horizon_steps = horizon;
                    if let Some(temporal) = query.temporal.as_ref() {
                        witness.policy = temporal.policy.clone();
                        witness.max_history_lag = temporal.max_history_lag;
                    }
                    let cache = build_temporal_class_posterior_identification_cache(
                        posterior,
                        &variables,
                        &witness,
                        analysis.max_completions,
                        ctx,
                    )?;
                    if let Some(atom) = cache.class_atoms.first() {
                        analysis.temporal_class_identification_cache =
                            Some(Arc::new(CachedTemporalClassIdentification {
                                envelope: atom.envelope.clone(),
                                by_horizon: vec![(horizon, atom.envelope.clone())],
                            }));
                    }
                    analysis.temporal_class_posterior_identification_cache = Some(Arc::new(cache));
                } else {
                    let estimator_id = EstimatorId::temporal_response_for(&self.inference);
                    analysis.dbn_posterior_identification_cache =
                        Some(Arc::new(build_dbn_posterior_response_identification_cache(
                            posterior,
                            &variables,
                            query,
                            estimator_id,
                            ctx,
                        )?));
                }
            }
            (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Mediation(_), None) => {
                analysis.temporal_identification_cache =
                    self.prepare_temporal_mediation_identification()?.map(Arc::new);
                analysis.temporal_class_identification_cache =
                    self.prepare_temporal_class_identification()?.map(Arc::new);
            }
            (DataInput::Tabular(_), CausalQuery::Transport(query), None) => {
                let diagram = self.selection_diagram.as_ref().ok_or(CausalError::Unsupported {
                    message: "TransportQuery prepare requires a selection diagram",
                })?;
                analysis.transport_identification_cache = Some(Arc::new(
                    crate::analysis::execute::live_transport_identification(diagram, query)?,
                ));
            }
            (DataInput::Tabular(_), CausalQuery::RandomizedEffect(_), None) => {
                // A randomized trial has design-based identification but no graph class to
                // inspect. Preserve its typed identification cache without invoking the
                // graph-only PAG/CPDAG cache builders below.
                analysis.identification_cache =
                    self.prepare_static_identification(&plan)?.map(Arc::new);
            }
            (
                DataInput::Tabular(_),
                CausalQuery::PolicyValue(_)
                | CausalQuery::ContinuousDoseResponse(_)
                | CausalQuery::Survival(_)
                | CausalQuery::LongitudinalRegime(_),
                None,
            ) => {}
            (DataInput::Tabular(_), _, None) => {
                analysis.identification_cache =
                    self.prepare_static_identification(&plan)?.map(Arc::new);
                analysis.pag_identification_cache =
                    self.prepare_pag_identification(&plan)?.map(Arc::new);
                analysis.cpdag_identification_cache =
                    self.prepare_cpdag_identification(&plan)?.map(Arc::new);
            }
            (
                DataInput::Panel(_),
                CausalQuery::TemporalEffect(_) | CausalQuery::Response(_),
                None,
            ) => {
                if analysis.graph.class().is_incomplete_temporal() {
                    analysis.temporal_class_identification_cache =
                        self.prepare_temporal_class_identification()?.map(Arc::new);
                } else {
                    analysis.temporal_identification_cache =
                        self.prepare_temporal_identification()?.map(Arc::new);
                }
            }
            (
                DataInput::Temporal(_) | DataInput::Event(_),
                CausalQuery::AnomalyAttribution(_) | CausalQuery::ChangeAttribution(_),
                None,
            ) => {
                analysis.identification_cache =
                    self.prepare_static_identification(&plan)?.map(Arc::new);
            }
            (_, _, None) => {
                analysis.temporal_identification_cache =
                    self.prepare_temporal_identification()?.map(Arc::new);
                analysis.temporal_class_identification_cache =
                    self.prepare_temporal_class_identification()?.map(Arc::new);
            }
            (
                DataInput::Temporal(_) | DataInput::Event(_),
                CausalQuery::NestedCounterfactual(_)
                | CausalQuery::PathSpecific(_)
                | CausalQuery::Counterfactual(_)
                | CausalQuery::Distribution(_),
                Some(_),
            ) => {
                // Identification runs at execute, on the cross-world probe, so the
                // cache matches the producer rather than an average-effect witness.
            }
            (_, _, Some(_)) => {
                // `ensure_prepared_supported` already refuses this coordinate; a
                // library must still answer with a typed refusal, never a panic.
                return Err(CausalError::Support {
                    id: crate::support::SupportRefusal::Refused,
                    message: "graph_posterior on the prepared handle is licensed only for \
                        tabular AverageEffect/Response/ConditionalEffect and series \
                        TemporalEffect, TemporalMediationEffect, or TemporalDag Response",
                });
            }
        }
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_IDENTIFY });
        }
        // The posterior builders report `identify.compute` themselves; the
        // single-graph, PAG, and temporal caches identify above without a
        // progress hook, so report once here when one of them was built. A
        // sharp-RD prepare builds no cache and reports nothing: its clicks do.
        if analysis.identification_cache.is_some()
            || analysis.pag_identification_cache.is_some()
            || analysis.cpdag_identification_cache.is_some()
            || analysis.temporal_identification_cache.is_some()
            || analysis.temporal_class_identification_cache.is_some()
            || analysis.transport_identification_cache.is_some()
        {
            crate::analysis::execute::report_identify_compute(ctx);
        }
        let checked_linear = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
            self.graph.class(),
            self.graph_posterior.is_none(),
            self.tiered.is_none(),
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(query),
                Some(cache),
                GraphClass::Dag,
                GraphClass::Dag,
                true,
                true,
            ) if matches!(query.outcome_functional, OutcomeFunctional::Mean)
                && matches!(query.target_population, TargetPopulation::AllObserved)
                && plan.logical.record.estimator.as_deref()
                    == Some(crate::strategy_table::EstimatorId::LinearAdjustmentAte.as_str()) =>
            {
                let default_id = !matches!(
                    &self.estimator_spec,
                    Some(crate::estimator_spec::EstimatorSpec::LinearAdjustmentAte(_))
                );
                let mut fitter = match &self.estimator_spec {
                    Some(crate::estimator_spec::EstimatorSpec::LinearAdjustmentAte(cfg)) => {
                        (**cfg).clone()
                    }
                    _ => antecedent_estimate::LinearAdjustmentAte::new(),
                };
                if default_id {
                    fitter.bootstrap_replicates = analysis.bootstrap_replicates;
                }
                let preparation = fitter.prepare_checked(data, &cache.identification, 0)?;
                let identifier = plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_IDENTIFIER)
                    .parse()?;
                let estimator = plan
                    .logical
                    .record
                    .estimator
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_ESTIMATOR)
                    .parse()?;
                Some(CheckedLinearOperation {
                    fitter,
                    preparation,
                    default_id,
                    query: query.clone(),
                    identification: cache.identification.clone(),
                    estimand: cache.estimand.clone(),
                    stage_sink: analysis.stage_sink.clone(),
                    common: CheckedStaticCommon::from_analysis(
                        &analysis, &plan, identifier, estimator,
                    ),
                })
            }
            _ => None,
        };
        let checked_glm = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
            &analysis.inference,
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(query),
                Some(cache),
                GraphClass::Dag,
                InferenceMode::Frequentist,
            ) if matches!(query.outcome_functional, OutcomeFunctional::Mean)
                && !matches!(query.active, Intervention::Set { variable, .. } if variable != query.treatment)
                && !matches!(query.control, Intervention::Set { variable, .. } if variable != query.treatment)
                && plan.logical.record.estimator.as_deref()
                    == Some(crate::strategy_table::EstimatorId::GlmAdjustment.as_str())
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
                        &analysis.estimator_spec
                    {
                        (**config).clone()
                    } else {
                        let mut fitter = antecedent_estimate::GlmAdjustmentAte::new();
                        fitter.bootstrap_replicates = analysis.bootstrap_replicates;
                        fitter
                    };
                if fitter.population_registry.is_none() {
                    fitter.population_registry.clone_from(&analysis.population_registry);
                }
                let preparation = fitter.prepare(data, &cache.estimand, query)?;
                let identifier = plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_IDENTIFIER)
                    .parse()?;
                let estimator = plan
                    .logical
                    .record
                    .estimator
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_ESTIMATOR)
                    .parse()?;
                Some(CheckedGlmAdjustmentOperation {
                    source_query: CausalQuery::AverageEffect(query.clone()),
                    query: query.clone(),
                    identification: cache.identification.clone(),
                    estimand: cache.estimand.clone(),
                    fitter,
                    preparation,
                    common: CheckedStaticCommon::from_analysis(
                        &analysis, &plan, identifier, estimator,
                    ),
                })
            }
            _ => None,
        };
        let checked_rd = match (
            &self.data,
            &self.query,
            analysis.graph.as_dag(),
            analysis.rd,
            &analysis.inference,
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(query),
                Some(graph),
                Some(config),
                InferenceMode::Frequentist,
            ) if plan.logical.record.estimator.as_deref()
                == Some(crate::strategy_table::EstimatorId::RdSharp.as_str())
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
                    || !crate::strategy_table::identification_status_acceptable(
                        identification.status,
                    )
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
                        identification,
                        estimand,
                        fitter,
                        preparation,
                        common: CheckedStaticCommon::from_analysis(
                            &analysis, &plan, identifier, estimator,
                        ),
                    }
                })
            }
            _ => None,
        };
        let checked_conditional = match (
            &self.data,
            &self.query,
            analysis.graph.class(),
            &analysis.inference,
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::ConditionalEffect(query),
                graph_class,
                InferenceMode::Frequentist,
            ) if analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && matches!(graph_class, GraphClass::Dag | GraphClass::Cpdag | GraphClass::Pag)
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
                && plan.logical.record.estimator.as_deref()
                    == Some(
                        crate::strategy_table::EstimatorId::ConditionalLinearAdjustment.as_str(),
                    ) =>
            {
                let class_proof = match graph_class {
                    GraphClass::Cpdag => {
                        let (Some(graph), Some(cache)) = (
                            analysis.graph.as_cpdag(),
                            analysis.cpdag_identification_cache.as_deref(),
                        ) else {
                            return Err(CausalError::Compile {
                                message: "conditional CPDAG preparation lost its graph proof"
                                    .into(),
                            });
                        };
                        Some(crate::analysis::checked_conditional::ConditionalClassProof::Cpdag {
                            graph: graph.clone(),
                            cache: cache.clone(),
                        })
                    }
                    GraphClass::Pag => {
                        let (Some(graph), Some(cache)) =
                            (plan.static_pag(), analysis.pag_identification_cache.as_deref())
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
                };
                let target = match (&class_proof, graph_class) {
                    (None, GraphClass::Dag) => analysis
                        .identification_cache
                        .as_deref()
                        .map(|cache| (cache.identification.clone(), cache.estimand.clone())),
                    (
                        Some(crate::analysis::checked_conditional::ConditionalClassProof::Cpdag {
                            cache,
                            ..
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
                                estimand.map(|estimand| (cache.identification.clone(), estimand))
                            })
                            .flatten()
                    }
                    (
                        Some(crate::analysis::checked_conditional::ConditionalClassProof::Pag {
                            cache,
                            ..
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
                                estimand.map(|estimand| (cache.identification.clone(), estimand))
                            })
                            .flatten()
                    }
                    _ => None,
                };
                target
                    .map(|(identification, estimand)| {
                        CheckedConditionalOperation::prepare(
                            data,
                            query.clone(),
                            identification,
                            estimand,
                            CheckedStaticCommon::from_analysis(
                                &analysis,
                                &plan,
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
        let checked_propensity = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
            &analysis.inference,
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(query),
                Some(cache),
                GraphClass::Dag,
                InferenceMode::Frequentist,
            ) if matches!(query.outcome_functional, OutcomeFunctional::Mean)
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
                    plan.logical.record.estimator.as_deref(),
                    Some("propensity.weighting" | "propensity.matching")
                ) =>
            {
                let estimator: crate::strategy_table::EstimatorId = plan
                    .logical
                    .record
                    .estimator
                    .as_deref()
                    .expect("checked propensity estimator")
                    .parse()?;
                let identifier = plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_IDENTIFIER)
                    .parse()?;
                let estimand_index = cache
                    .identification
                    .estimands
                    .iter()
                    .position(|candidate| {
                        candidate.functional == cache.estimand.functional
                            && candidate.method == cache.estimand.method
                            && candidate.adjustment_set == cache.estimand.adjustment_set
                    })
                    .ok_or_else(|| CausalError::Compile {
                        message:
                            "selected propensity estimand is absent from its identification product"
                                .into(),
                    })?;
                let fitter = match (&analysis.estimator_spec, estimator) {
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
                    identification: cache.identification.clone(),
                    estimand_index,
                    common: CheckedStaticCommon::from_analysis(
                        &analysis, &plan, identifier, estimator,
                    ),
                };
                Some(CheckedPropensityOperation::prepare(data, context, estimator, fitter)?)
            }
            _ => None,
        };
        let checked_aipw = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
            &analysis.inference,
            self.graph.class(),
            self.graph_posterior.is_none(),
            self.tiered.as_ref(),
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(query),
                Some(cache),
                graph_class,
                InferenceMode::Frequentist,
                source_graph_class,
                true,
                tiered_background,
            ) if matches!(
                (graph_class, source_graph_class, tiered_background),
                (GraphClass::Dag, GraphClass::Dag, None)
                    | (GraphClass::Admg, GraphClass::Admg, Some(_))
            ) && tiered_background.is_none_or(|background| {
                background.within_tier == antecedent_graph::WithinTier::CoDetermined
            }) && matches!(query.outcome_functional, OutcomeFunctional::Mean)
                && matches!(query.target_population, TargetPopulation::AllObserved)
                && matches!(&query.active, antecedent_core::Intervention::Set { variable, value }
                        if *variable == query.treatment && value.as_f64() == Some(1.0))
                && matches!(&query.control, antecedent_core::Intervention::Set { variable, value }
                        if *variable == query.treatment && value.as_f64() == Some(0.0))
                && plan.logical.record.estimator.as_deref()
                    == Some(crate::strategy_table::EstimatorId::Aipw.as_str()) =>
            {
                let fitter = checked_aipw_fitter(&analysis);
                if matches!(fitter.overlap, OverlapPolicy::RequireDiagnostics { .. }) {
                    let preparation = fitter.prepare_checked(data, &cache.identification, 0)?;
                    let identifier = plan
                        .logical
                        .record
                        .identifier
                        .as_deref()
                        .unwrap_or(crate::strategy_table::DEFAULT_IDENTIFIER)
                        .parse()?;
                    let estimator = plan
                        .logical
                        .record
                        .estimator
                        .as_deref()
                        .unwrap_or(crate::strategy_table::DEFAULT_ESTIMATOR)
                        .parse()?;
                    Some(CheckedAipwOperation {
                        fitter,
                        preparation,
                        query: query.clone(),
                        identification: cache.identification.clone(),
                        estimand: cache.estimand.clone(),
                        common: CheckedStaticCommon::from_analysis(
                            &analysis, &plan, identifier, estimator,
                        ),
                        tiered_background: tiered_background.cloned(),
                    })
                } else {
                    None
                }
            }
            _ => None,
        };
        let checked_frontdoor_linear = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
            &analysis.inference,
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(query),
                Some(cache),
                GraphClass::Dag,
                InferenceMode::Frequentist,
            ) if matches!(query.outcome_functional, OutcomeFunctional::Mean)
                && matches!(query.target_population, TargetPopulation::AllObserved)
                && plan.logical.record.estimator.as_deref()
                    == Some(crate::strategy_table::EstimatorId::FrontDoorTwoStage.as_str()) =>
            {
                let fitter = match &analysis.estimator_spec {
                    Some(crate::estimator_spec::EstimatorSpec::FrontDoorTwoStage(config)) => {
                        (**config).clone()
                    }
                    _ => antecedent_estimate::FrontDoorTwoStage::new(),
                };
                let preparation = fitter.prepare_checked(data, &cache.identification, 0)?;
                Some(CheckedFrontDoorOperation { fitter, preparation })
            }
            _ => None,
        };
        let checked_iv = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
            &analysis.inference,
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(_),
                Some(cache),
                GraphClass::Dag,
                InferenceMode::Frequentist,
            ) => {
                let estimator = plan.logical.record.estimator.as_deref().unwrap_or("");
                match (&analysis.estimator_spec, estimator) {
                    (Some(crate::estimator_spec::EstimatorSpec::IvWald(cfg)), "iv.wald") => {
                        Some(CheckedIvOperation::Wald {
                            fitter: (**cfg).clone(),
                            preparation: (**cfg).prepare_checked(data, &cache.identification, 0)?,
                        })
                    }
                    (
                        None
                        | Some(crate::estimator_spec::EstimatorSpec::Default(
                            crate::EstimatorId::IvWald,
                        )),
                        "iv.wald",
                    ) => {
                        let fitter = antecedent_estimate::WaldIv::new();
                        let preparation = fitter.prepare_checked(data, &cache.identification, 0)?;
                        Some(CheckedIvOperation::Wald { fitter, preparation })
                    }
                    (Some(crate::estimator_spec::EstimatorSpec::Iv2Sls(cfg)), "iv.2sls") => {
                        match (**cfg).prepare_checked(data, &cache.identification, 0) {
                            Ok(preparation) => Some(CheckedIvOperation::TwoSls {
                                fitter: (**cfg).clone(),
                                preparation,
                            }),
                            Err(antecedent_estimate::EstimationError::Unsupported {
                                message:
                                    "checked IV currently requires a binary 0/1 instrument matching the checked Wald functional",
                            }) => None,
                            Err(error) => return Err(error.into()),
                        }
                    }
                    (
                        None
                        | Some(crate::estimator_spec::EstimatorSpec::Default(
                            crate::EstimatorId::Iv2Sls,
                        )),
                        "iv.2sls",
                    ) => {
                        let fitter = antecedent_estimate::TwoStageLeastSquares::new();
                        match fitter.prepare_checked(data, &cache.identification, 0) {
                            Ok(preparation) => {
                                Some(CheckedIvOperation::TwoSls { fitter, preparation })
                            }
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
        let distribution_operation = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
            &analysis.inference,
        ) {
            (DataInput::Tabular(data), CausalQuery::Distribution(query), Some(cache), _, _) => {
                let identifier = plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_DISTRIBUTION_IDENTIFIER)
                    .parse()?;
                let estimator = plan
                    .logical
                    .record
                    .estimator
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_DISTRIBUTION_ESTIMATOR)
                    .parse()?;
                let mut fitter = antecedent_estimate::FunctionalDistribution::new();
                fitter.bootstrap_replicates = analysis.bootstrap_replicates;
                let prepared = fitter.prepare(
                    data,
                    query,
                    &cache.estimand,
                    &cache.identification.arena,
                    cache.identification.required_assumptions.clone(),
                )?;
                Some(CheckedDistributionOperation {
                    query: query.clone(),
                    identification: cache.identification.clone(),
                    estimand: cache.estimand.clone(),
                    fitter,
                    prepared,
                    common: CheckedStaticCommon::from_analysis(
                        &analysis, &plan, identifier, estimator,
                    ),
                })
            }
            _ => None,
        };
        let functional_effect_operation = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
            &analysis.inference,
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(query),
                Some(cache),
                GraphClass::Admg | GraphClass::TemporalAdmg,
                _,
            ) if analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && cache.estimand.method_kind().ok()
                    == Some(antecedent_expr::EstimandMethod::GeneralId)
                && plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_ADMG_IDENTIFIER)
                    == crate::strategy_table::IdentifierId::GeneralId.as_str()
                && plan
                    .logical
                    .record
                    .estimator
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_ADMG_ESTIMATOR)
                    == crate::strategy_table::EstimatorId::FunctionalEffect.as_str()
                && matches!(query.outcome_functional, OutcomeFunctional::Mean)
                && matches!(query.target_population, TargetPopulation::AllObserved) =>
            {
                let mut fitter = antecedent_estimate::FunctionalEffect::new();
                fitter.bootstrap_replicates = analysis.bootstrap_replicates;
                let prepared = fitter.prepare(
                    data,
                    &cache.estimand,
                    &cache.identification.arena,
                    cache.identification.required_assumptions.clone(),
                    &[query.treatment, query.outcome],
                )?;
                Some(CheckedFunctionalEffectOperation {
                    query: query.clone(),
                    identification: cache.identification.clone(),
                    estimand: cache.estimand.clone(),
                    fitter,
                    prepared,
                    common: CheckedStaticCommon::from_analysis(
                        &analysis,
                        &plan,
                        crate::strategy_table::IdentifierId::GeneralId,
                        crate::strategy_table::EstimatorId::FunctionalEffect,
                    ),
                })
            }
            _ => None,
        };
        let path_specific_effect_operation = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
            &analysis.inference,
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::PathSpecific(query),
                Some(cache),
                GraphClass::Dag,
                _,
            ) if analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && cache.estimand.method_kind().ok()
                    == Some(antecedent_expr::EstimandMethod::PathSpecificNatural)
                && plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_PATH_IDENTIFIER)
                    == crate::strategy_table::IdentifierId::PathSpecificNatural.as_str()
                && plan
                    .logical
                    .record
                    .estimator
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_PATH_ESTIMATOR)
                    == crate::strategy_table::EstimatorId::FunctionalEffect.as_str()
                && matches!(query.target_population, TargetPopulation::AllObserved) =>
            {
                let mut fitter = antecedent_estimate::FunctionalEffect::new();
                fitter.bootstrap_replicates = analysis.bootstrap_replicates;
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
                    identification: cache.identification.clone(),
                    estimand: cache.estimand.clone(),
                    fitter,
                    prepared,
                    common: CheckedStaticCommon::from_analysis(
                        &analysis,
                        &plan,
                        crate::strategy_table::IdentifierId::PathSpecificNatural,
                        crate::strategy_table::EstimatorId::FunctionalEffect,
                    ),
                })
            }
            _ => None,
        };
        let admg_response_curve_operation = match (&self.data, &self.query, analysis.graph.class())
        {
            (
                DataInput::Tabular(data),
                CausalQuery::Response(query),
                GraphClass::Admg | GraphClass::TemporalAdmg,
            ) if analysis.graph_posterior.is_none()
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
                && plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_ADMG_IDENTIFIER)
                    == crate::strategy_table::IdentifierId::GeneralId.as_str()
                && plan
                    .logical
                    .record
                    .estimator
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_ADMG_ESTIMATOR)
                    == crate::strategy_table::EstimatorId::FunctionalEffect.as_str()
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
                let (treatment, outcome, grid) = match &query.functional {
                    antecedent_core::ResponseFunctional::MeanCurve { outcome, treatment } => (
                        treatment.variable,
                        *outcome,
                        treatment
                            .grid
                            .values()
                            .map_err(|error| CausalError::Compile { message: error.to_string() })?,
                    ),
                    antecedent_core::ResponseFunctional::InterventionResponse {
                        outcome,
                        interventions,
                    } if interventions.len() == 1 => {
                        let intervention = &interventions[0];
                        let (treatment, _) = query.functional.primary_pair().ok_or_else(|| {
                            CausalError::Compile {
                                message: "scalar ADMG response lacks treatment role".into(),
                            }
                        })?;
                        let antecedent_core::Intervention::Set { variable, value } = intervention
                        else {
                            return Err(CausalError::Compile {
                                message: "checked ADMG scalar response requires a single hard-set intervention".into(),
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
                };
                if grid.is_empty() {
                    return Err(CausalError::Compile {
                        message: "ADMG response curve has no grid members".into(),
                    });
                }
                let admg = match analysis.graph.class() {
                    GraphClass::Admg => analysis.graph.as_admg().cloned(),
                    GraphClass::TemporalAdmg => analysis
                        .graph
                        .as_temporal_admg()
                        .map(|graph| crate::analysis::execute::temporal_admg_projection(graph))
                        .transpose()?,
                    _ => None,
                }
                .ok_or_else(|| CausalError::Compile {
                    message: "ADMG response operation lacks an ADMG graph".into(),
                })?;
                let mut fitter = antecedent_estimate::FunctionalEffect::new();
                fitter.bootstrap_replicates = analysis.bootstrap_replicates;
                let mut members = Vec::with_capacity(grid.len());
                for level in &grid {
                    let mut member_query = query.clone();
                    if matches!(
                        query.functional,
                        antecedent_core::ResponseFunctional::MeanCurve { .. }
                    ) {
                        member_query.functional =
                            antecedent_core::ResponseFunctional::InterventionResponse {
                                outcome,
                                interventions: Arc::from([Intervention::set(
                                    treatment,
                                    Value::f64(*level),
                                )]),
                            };
                    }
                    let member_causal_query = CausalQuery::Response(member_query.clone());
                    let identification = crate::strategy_table::identify_admg_query(
                        crate::strategy_table::IdentifierId::GeneralId,
                        &admg,
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
                Some(CheckedAdmgResponseCurveOperation {
                    factor_snapshot_digest: data.storage().content_digest(),
                    query: query.clone(),
                    grid: Arc::from(grid),
                    members: Arc::from(members),
                    fitter,
                    common: CheckedStaticCommon::from_analysis(
                        &analysis,
                        &plan,
                        crate::strategy_table::IdentifierId::GeneralId,
                        crate::strategy_table::EstimatorId::FunctionalEffect,
                    ),
                })
            }
            _ => None,
        };
        let bayesian_gcomp_operation = match (
            &self.query,
            analysis.identification_cache.as_deref(),
            &analysis.inference,
            analysis.graph.class(),
            plan.logical.record.estimator.as_deref(),
        ) {
            (
                CausalQuery::AverageEffect(query),
                Some(cache),
                InferenceMode::Bayesian(_),
                GraphClass::Dag,
                Some(estimator),
            ) if matches!(query.outcome_functional, OutcomeFunctional::Mean)
                && matches!(query.target_population, TargetPopulation::AllObserved)
                && estimator == crate::strategy_table::EstimatorId::BayesianGcomp.as_str() =>
            {
                Some(CheckedBayesianGcompOperation {
                    query: query.clone(),
                    identification: cache.identification.clone(),
                    estimand: cache.estimand.clone(),
                    inference: analysis.inference.clone(),
                })
            }
            _ => None,
        };
        let bayesian_gcomp_execution = if let Some(operation) = bayesian_gcomp_operation {
            let graph = analysis.graph.as_dag().ok_or(CausalError::Compile {
                message: "checked Bayesian g-computation requires its retained DAG".into(),
            })?;
            Some(crate::analysis::execute::CheckedBayesianDagAteExecution::checked(
                graph,
                operation,
                crate::analysis::execute::IdentifiedResultContext::from_study(&analysis),
                plan.clone(),
                analysis.refute,
                analysis.latency_mode,
                analysis.custom_validators.clone(),
                analysis.stage_sink.clone(),
            )?)
        } else {
            None
        };
        let bayesian_basis_ate_execution = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            &analysis.inference,
            analysis.graph.class(),
            plan.logical.record.estimator.as_deref(),
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(query),
                Some(cache),
                InferenceMode::Bayesian(_),
                GraphClass::Dag,
                Some(estimator),
            ) if analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && analysis.structure_source == crate::support::StructureSource::Explicit
                && analysis.refute == RefuteSuite::None
                && analysis.custom_validators.is_empty()
                && estimator == crate::strategy_table::EstimatorId::BayesianBasisGcomp.as_str() =>
            {
                let graph = analysis.graph.as_dag().ok_or(CausalError::Compile {
                    message: "checked Bayesian basis ATE requires its retained DAG".into(),
                })?;
                Some(crate::analysis::execute::CheckedBayesianBasisAteExecution::checked(
                    data,
                    graph,
                    query.clone(),
                    cache.identification.clone(),
                    cache.estimand.clone(),
                    analysis.inference.clone(),
                    crate::analysis::execute::IdentifiedResultContext::from_study(&analysis),
                    plan.clone(),
                )?)
            }
            _ => None,
        };
        let bayesian_basis_cate_execution = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            &analysis.inference,
            analysis.graph.class(),
            plan.logical.record.estimator.as_deref(),
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::ConditionalEffect(query),
                Some(cache),
                InferenceMode::Bayesian(_),
                GraphClass::Dag,
                Some(estimator),
            ) if analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && analysis.structure_source == crate::support::StructureSource::Explicit
                && analysis.refute == RefuteSuite::None
                && analysis.custom_validators.is_empty()
                && estimator == crate::strategy_table::EstimatorId::BayesianBasisGcomp.as_str() =>
            {
                let graph = analysis.graph.as_dag().ok_or(CausalError::Compile {
                    message: "checked Bayesian basis CATE requires its retained DAG".into(),
                })?;
                Some(crate::analysis::execute::CheckedBayesianBasisCateExecution::checked(
                    data,
                    graph,
                    query.clone(),
                    cache.identification.clone(),
                    cache.estimand.clone(),
                    analysis.inference.clone(),
                    crate::analysis::execute::IdentifiedResultContext::from_study(&analysis),
                    plan.clone(),
                )?)
            }
            _ => None,
        };
        let bayesian_robust_ate_execution = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            &analysis.inference,
            analysis.graph.class(),
            plan.logical.record.estimator.as_deref(),
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(query),
                Some(cache),
                InferenceMode::Bayesian(_),
                GraphClass::Dag,
                Some(estimator),
            ) if analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && analysis.structure_source == crate::support::StructureSource::Explicit
                && analysis.refute == RefuteSuite::None
                && analysis.custom_validators.is_empty()
                && estimator == crate::strategy_table::EstimatorId::BayesianRobustAte.as_str() =>
            {
                let graph = analysis.graph.as_dag().ok_or(CausalError::Compile {
                    message: "checked Bayesian robust ATE requires its retained DAG".into(),
                })?;
                Some(crate::analysis::execute::CheckedBayesianRobustAteExecution::checked(
                    data,
                    graph,
                    query.clone(),
                    cache.identification.clone(),
                    cache.estimand.clone(),
                    analysis.inference.clone(),
                    crate::analysis::execute::IdentifiedResultContext::from_study(&analysis),
                    plan.clone(),
                )?)
            }
            _ => None,
        };
        let bayesian_conditional_execution = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            &analysis.inference,
            analysis.graph.class(),
            plan.logical.record.estimator.as_deref(),
        ) {
            (
                DataInput::Tabular(_),
                CausalQuery::ConditionalEffect(query),
                Some(cache),
                InferenceMode::Bayesian(_),
                GraphClass::Dag,
                Some(estimator),
            ) if analysis.graph_posterior.is_none()
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
                && estimator
                    == crate::strategy_table::EstimatorId::BayesianConditional.as_str() =>
            {
                let graph = analysis.graph.as_dag().ok_or(CausalError::Compile {
                    message: "checked Bayesian conditional effect requires its retained DAG".into(),
                })?;
                Some(crate::analysis::execute::CheckedBayesianConditionalOperation::checked(
                    graph,
                    query.clone(),
                    cache.identification.clone(),
                    cache.estimand.clone(),
                    analysis.inference.clone(),
                    analysis.refute,
                    crate::analysis::execute::IdentifiedResultContext::from_study(&analysis),
                    plan.clone(),
                    analysis.latency_mode,
                    analysis.custom_validators.clone(),
                    analysis.stage_sink.clone(),
                )?)
            }
            _ => None,
        };
        let derivative_response_operation = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
        ) {
            (DataInput::Tabular(_), CausalQuery::Response(query), Some(cache), GraphClass::Dag)
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
                let identifier = plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_RESPONSE_IDENTIFIER)
                    .parse()?;
                let estimator = plan
                    .logical
                    .record
                    .estimator
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_RESPONSE_ESTIMATOR)
                    .parse()?;
                Some(crate::analysis::execute::CheckedDerivativeResponseOperation::checked(
                    analysis.graph.as_dag().expect("checked Dag response"),
                    query,
                    &cache.identification,
                    &cache.estimand,
                    identifier,
                    estimator,
                    analysis.inference.clone(),
                    analysis.response_options.clone().unwrap_or_default(),
                )?)
            }
            _ => None,
        };
        let static_dag_response_operation = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
        ) {
            (DataInput::Tabular(_), CausalQuery::Response(query), Some(cache), GraphClass::Dag)
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
                            || analysis.refute == RefuteSuite::None)) =>
            {
                let identifier = plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_RESPONSE_IDENTIFIER)
                    .parse()?;
                let estimator = plan
                    .logical
                    .record
                    .estimator
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_RESPONSE_ESTIMATOR)
                    .parse()?;
                let bayesian = matches!(analysis.inference, InferenceMode::Bayesian(_));
                if bayesian && estimator == crate::strategy_table::EstimatorId::ResponseBayesian
                    || !bayesian
                        && estimator == crate::strategy_table::EstimatorId::ResponseKennedyDr
                        && matches!(
                            query.functional,
                            antecedent_core::ResponseFunctional::MeanCurve { .. }
                        )
                    || !bayesian
                        && estimator
                            == crate::strategy_table::EstimatorId::ResponseInterventionGcomp
                        && matches!(
                            query.functional,
                            antecedent_core::ResponseFunctional::InterventionResponse { .. }
                        )
                {
                    Some(crate::analysis::execute::CheckedStaticDagResponseOperation::checked(
                        analysis.graph.as_dag().expect("checked DAG response"),
                        query,
                        &cache.identification,
                        &cache.estimand,
                        identifier,
                        estimator,
                        analysis.response_options.clone().unwrap_or_default(),
                        analysis.inference.clone(),
                        analysis.refute,
                    )?)
                } else {
                    None
                }
            }
            _ => None,
        };
        let cell_aipw_response_operation = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
        ) {
            (DataInput::Tabular(_), CausalQuery::Response(query), Some(cache), GraphClass::Dag)
                if analysis.graph_posterior.is_none()
                    && analysis.tiered.is_none()
                    && matches!(
                        analysis.structure_source,
                        crate::support::StructureSource::Explicit
                            | crate::support::StructureSource::Accepted
                    ) =>
            {
                let identifier = plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_RESPONSE_IDENTIFIER)
                    .parse()?;
                let estimator = plan
                    .logical
                    .record
                    .estimator
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_RESPONSE_ESTIMATOR)
                    .parse()?;
                if estimator == crate::strategy_table::EstimatorId::CellAipw
                    && analysis.custom_validators.is_empty()
                    && analysis.shared_batch_design.is_none()
                    && analysis.continuous_cell.is_none()
                    && matches!(analysis.inference, InferenceMode::Frequentist)
                {
                    Some(crate::analysis::execute::CheckedCellAipwResponseOperation::checked(
                        &crate::analysis::execute::CellAipwStructure::Dag(
                            analysis.graph.as_dag().expect("checked DAG response").clone(),
                        ),
                        query,
                        &cache.identification,
                        &cache.estimand,
                        identifier,
                        estimator,
                        ctx.rng.master_seed(),
                        analysis.refute,
                        if analysis.structure_source == crate::support::StructureSource::Accepted {
                            crate::analysis::execute::DagResponseOrigin::Accepted
                        } else {
                            crate::analysis::execute::DagResponseOrigin::Explicit
                        },
                    )?)
                } else {
                    None
                }
            }
            (
                DataInput::Tabular(_),
                CausalQuery::Response(query),
                Some(cache),
                GraphClass::Admg,
            ) if analysis.graph_posterior.is_none()
                && analysis.structure_source == crate::support::StructureSource::Explicit
                && analysis.tiered.as_ref().is_some_and(|background| {
                    background.within_tier == antecedent_graph::WithinTier::CoDetermined
                })
                && analysis.custom_validators.is_empty()
                && analysis.shared_batch_design.is_none()
                && analysis.continuous_cell.is_none()
                && matches!(analysis.inference, InferenceMode::Frequentist)
                && plan.logical.record.estimator.as_deref()
                    == Some(crate::strategy_table::EstimatorId::CellAipw.as_str())
                && plan.logical.record.identifier.as_deref()
                    == Some(
                        crate::strategy_table::IdentifierId::GeneralizedAdjustment.as_str(),
                    ) =>
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
                    &crate::analysis::execute::CellAipwStructure::TierClosure {
                        admg: admg.clone(),
                        background: background.clone(),
                    },
                    query,
                    &cache.identification,
                    &cache.estimand,
                    crate::strategy_table::IdentifierId::GeneralizedAdjustment,
                    crate::strategy_table::EstimatorId::CellAipw,
                    ctx.rng.master_seed(),
                    analysis.refute,
                    crate::analysis::execute::DagResponseOrigin::Explicit,
                )?)
            }
            _ => None,
        };
        let checked_interference = match (&self.data, &self.query, analysis.graph.class()) {
            (DataInput::Tabular(data), CausalQuery::Interference(_), GraphClass::Dag)
                if analysis.structure_source == crate::support::StructureSource::Explicit
                    && analysis.graph_posterior.is_none()
                    && analysis.tiered.is_none()
                    && analysis.split.is_none() =>
            {
                Some(crate::analysis::execute::CheckedInterferenceOperation::checked(
                    &analysis, data, &plan,
                )?)
            }
            _ => None,
        };
        let checked_randomized = match (&self.data, &self.query, analysis.graph.class()) {
            (
                DataInput::Tabular(data),
                CausalQuery::RandomizedEffect(_),
                GraphClass::RandomizedTrial,
            ) if analysis.structure_source == crate::support::StructureSource::RandomizedTrial
                && analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && analysis.split.is_none() =>
            {
                Some(crate::analysis::execute::CheckedRandomizedOperation::checked(
                    &analysis, data, &plan,
                )?)
            }
            _ => None,
        };
        let checked_policy_value = match (&self.data, &self.query, analysis.graph.class()) {
            (
                DataInput::Tabular(data),
                CausalQuery::PolicyValue(_),
                GraphClass::RandomizedTrial,
            ) if analysis.structure_source == crate::support::StructureSource::RandomizedTrial
                && analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && analysis.split.is_none() =>
            {
                Some(crate::analysis::execute::CheckedPolicyValueOperation::checked(
                    &analysis, data, &plan,
                )?)
            }
            _ => None,
        };
        let checked_continuous_dose = match (&self.data, &self.query, analysis.graph.class()) {
            (
                DataInput::Tabular(data),
                CausalQuery::ContinuousDoseResponse(_),
                GraphClass::RandomizedTrial,
            ) if analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && analysis.split.is_none() =>
            {
                Some(crate::analysis::execute::CheckedContinuousDoseOperation::checked(
                    &analysis, data, &plan,
                )?)
            }
            _ => None,
        };
        let checked_panel_did = match (&self.data, &self.query, analysis.graph.class()) {
            (DataInput::Tabular(data), CausalQuery::PanelDid(_), GraphClass::RandomizedTrial)
                if analysis.graph_posterior.is_none()
                    && analysis.tiered.is_none()
                    && analysis.split.is_none() =>
            {
                Some(crate::analysis::execute::CheckedPanelDidOperation::checked(
                    &analysis, data, &plan,
                )?)
            }
            _ => None,
        };
        let checked_synthetic_control = match (&self.data, &self.query, analysis.graph.class()) {
            (
                DataInput::Tabular(data),
                CausalQuery::SyntheticControl(_),
                GraphClass::RandomizedTrial,
            ) if analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && analysis.split.is_none() =>
            {
                Some(crate::analysis::execute::CheckedSyntheticControlOperation::checked(
                    &analysis, data, &plan,
                )?)
            }
            _ => None,
        };
        let checked_local_polynomial_ratio = match (&self.data, &self.query, analysis.graph.class())
        {
            (
                DataInput::Tabular(data),
                CausalQuery::LocalPolynomialRatio(_),
                GraphClass::RandomizedTrial,
            ) if analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && analysis.split.is_none() =>
            {
                Some(crate::analysis::execute::CheckedLocalPolynomialRatioOperation::checked(
                    &analysis, data, &plan,
                )?)
            }
            _ => None,
        };
        let checked_survival = match (&self.data, &self.query, analysis.graph.class()) {
            (DataInput::Tabular(data), CausalQuery::Survival(_), GraphClass::RandomizedTrial)
                if analysis.structure_source
                    == crate::support::StructureSource::RandomizedTrial
                    && analysis.graph_posterior.is_none()
                    && analysis.tiered.is_none()
                    && analysis.split.is_none() =>
            {
                Some(crate::analysis::execute::CheckedSurvivalOperation::checked(
                    &analysis, data, &plan,
                )?)
            }
            _ => None,
        };
        let checked_longitudinal_regime = match (&self.data, &self.query, analysis.graph.class()) {
            (
                DataInput::Tabular(data),
                CausalQuery::LongitudinalRegime(_),
                GraphClass::RandomizedTrial,
            ) if analysis.structure_source == crate::support::StructureSource::RandomizedTrial
                && analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && analysis.split.is_none() =>
            {
                Some(crate::analysis::execute::CheckedLongitudinalRegimeOperation::checked(
                    &analysis, data, &plan,
                )?)
            }
            _ => None,
        };
        let temporal_mediation_operation = match (
            &self.data,
            &self.query,
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
                    &analysis, data, &plan, cache,
                )?)
            }
            _ => None,
        };
        let temporal_class_mediation_operation = match &self.data {
            DataInput::Temporal(data) | DataInput::Event(data)
                if crate::analysis::execute::CheckedTemporalClassMediationOperation::admits(
                    &analysis,
                ) && (analysis.temporal_class_identification_cache.is_some()
                    || analysis.temporal_class_posterior_identification_cache.is_some()
                    || analysis.dbn_posterior_identification_cache.is_some()) =>
            {
                Some(Box::new(
                    crate::analysis::execute::CheckedTemporalClassMediationOperation::checked(
                        &analysis, data, &plan,
                    )?,
                ))
            }
            _ => None,
        };
        let temporal_dag_effect_operation = match (
            &self.data,
            &self.query,
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
                        crate::analysis::execute::IdentifiedResultContext::from_study(&analysis),
                        plan.clone(),
                    )?)
                }
            }
            _ => None,
        };
        let bayesian_temporal_dag_effect_operation = match (
            &self.data,
            &self.query,
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
                            message: "prepared Bayesian temporal effect lacks its horizon proof"
                                .into(),
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
        let temporal_class_effect_operation = match (
            &self.data,
            &self.query,
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
                        crate::analysis::execute::IdentifiedResultContext::from_study(&analysis),
                        plan.clone(),
                    )?)
                }
            }
            _ => None,
        };
        let temporal_dag_response_operation = match (
            &self.data,
            &self.query,
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
                            crate::analysis::execute::IdentifiedResultContext::from_study(
                                &analysis,
                            ),
                            plan.clone(),
                        )?;
                    Some(execution)
                }
            }
            _ => None,
        };
        let checked_response_curve = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
        ) {
            (DataInput::Tabular(_), CausalQuery::Response(query), Some(cache), GraphClass::Dag)
                if query.temporal.is_none()
                    && query.observation == antecedent_core::ObservationSpec::Complete
                    && query.target_population == TargetPopulation::AllObserved
                    && matches!(
                        query.functional,
                        antecedent_core::ResponseFunctional::MeanCurve { .. }
                    )
                    && matches!(analysis.inference, InferenceMode::Frequentist)
                    && plan
                        .logical
                        .record
                        .identifier
                        .as_deref()
                        .unwrap_or(crate::strategy_table::DEFAULT_RESPONSE_IDENTIFIER)
                        == crate::strategy_table::DEFAULT_RESPONSE_IDENTIFIER
                    && plan
                        .logical
                        .record
                        .estimator
                        .as_deref()
                        .unwrap_or(crate::strategy_table::DEFAULT_RESPONSE_ESTIMATOR)
                        == crate::strategy_table::DEFAULT_RESPONSE_ESTIMATOR =>
            {
                let antecedent_core::ResponseFunctional::MeanCurve { treatment, .. } =
                    &query.functional
                else {
                    unreachable!()
                };
                Some(CheckedStaticResponseCurve {
                    query: query.clone(),
                    grid: Arc::from(
                        treatment
                            .grid
                            .values()
                            .map_err(|e| CausalError::Compile { message: e.to_string() })?,
                    ),
                    identification: cache.identification.clone(),
                    estimand: cache.estimand.clone(),
                    identifier: crate::strategy_table::DEFAULT_RESPONSE_IDENTIFIER_ID,
                    estimator: crate::strategy_table::DEFAULT_RESPONSE_ESTIMATOR_ID,
                })
            }
            _ => None,
        };
        let checked_counterfactual = if let Some(operation) = counterfactual {
            let cache =
                analysis.identification_cache.as_deref().ok_or_else(|| CausalError::Compile {
                    message: "counterfactual preparation lost its identification product".into(),
                })?;
            let inference = analysis.inference.clone();
            let procedure =
                crate::analysis::execute::CounterfactualProcedure::for_inference(&inference)?;
            Some(crate::analysis::execute::CheckedCounterfactualPlan::new(
                operation,
                cache.identification.clone(),
                cache.estimand.clone(),
                true,
                inference,
                procedure,
                plan.clone(),
                crate::analysis::execute::IdentifiedResultContext::from_study(&analysis),
            )?)
        } else {
            None
        };
        let checked_graph_posterior_effect = Self::seal_graph_posterior_effect(&analysis, &plan)?;
        let checked_class_graph_posterior_effect =
            Self::seal_class_graph_posterior_effect(&analysis, &plan)?;
        let checked_bayesian_graph_posterior_ate =
            Self::seal_bayesian_graph_posterior_ate(&analysis, &plan)?;
        let checked_static_class_effect = Self::seal_static_class_effect(&analysis, &plan)?;
        let checked_admg_graph_posterior_response =
            Self::seal_admg_graph_posterior_response(&analysis, &plan)?;
        let checked_temporal_class_response = Self::seal_temporal_class_response(&analysis, &plan)?;
        let checked_temporal_graph_posterior_response =
            Self::seal_temporal_graph_posterior_response(&analysis, &plan)?;
        let checked_static_class_response = Self::seal_static_class_response(&analysis, &plan)?;
        let checked_graph_posterior_response =
            Self::seal_graph_posterior_response(&analysis, &plan)?;
        let checked_bayesian_static_mediation = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
            &analysis.inference,
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::Mediation(_),
                Some(cache),
                GraphClass::Dag,
                InferenceMode::Bayesian(_),
            ) => Some(Box::new(
                crate::analysis::execute::CheckedBayesianStaticMediationOperation::checked(
                    &analysis, data, &plan, cache,
                )?,
            )),
            _ => None,
        };
        let checked_static_mediation = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
        ) {
            (DataInput::Tabular(data), CausalQuery::Mediation(_), Some(cache), GraphClass::Dag)
                if matches!(analysis.inference, InferenceMode::Frequentist)
                    && matches!(
                        analysis.structure_source,
                        crate::support::StructureSource::Explicit
                            | crate::support::StructureSource::Accepted
                    )
                    && analysis.custom_validators.is_empty() =>
            {
                Some(crate::analysis::execute::CheckedStaticMediationOperation::checked(
                    &analysis, data, &plan, cache,
                )?)
            }
            _ => None,
        };
        let checked_attribution = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
            &analysis.inference,
        ) {
            (
                DataInput::Tabular(_),
                CausalQuery::AnomalyAttribution(_) | CausalQuery::ChangeAttribution(_),
                Some(cache),
                GraphClass::Dag,
                InferenceMode::Frequentist | InferenceMode::Bayesian(_),
            ) if analysis.structure_source == crate::support::StructureSource::Explicit => {
                Some(crate::analysis::execute::CheckedAttributionOperation::checked(
                    &analysis, &plan, cache,
                )?)
            }
            (
                DataInput::Temporal(_) | DataInput::Event(_),
                CausalQuery::AnomalyAttribution(_) | CausalQuery::ChangeAttribution(_),
                Some(cache),
                GraphClass::TemporalDag,
                InferenceMode::Frequentist | InferenceMode::Bayesian(_),
            ) if analysis.structure_source == crate::support::StructureSource::Explicit => {
                Some(crate::analysis::execute::CheckedAttributionOperation::checked(
                    &analysis, &plan, cache,
                )?)
            }
            _ => None,
        };
        let score_table = analysis.prepare_score_table(ctx)?;
        let checked_unknown_tiered_average = match (
            &self.data,
            &self.query,
            analysis.tiered.as_ref(),
            analysis.identification_cache.as_deref(),
            &analysis.inference,
        ) {
            (
                DataInput::Tabular(_),
                CausalQuery::AverageEffect(query),
                Some(background),
                Some(cache),
                InferenceMode::Frequentist,
            ) if background.within_tier == antecedent_graph::WithinTier::Unknown
                && analysis.structure_source == crate::support::StructureSource::Explicit
                && analysis.graph_posterior.is_none()
                && analysis.custom_validators.is_empty()
                && matches!(analysis.refute, RefuteSuite::None)
                && plan.logical.record.estimator.as_deref()
                    == Some(crate::strategy_table::EstimatorId::LinearAdjustmentAte.as_str())
                && plan.logical.record.identifier.as_deref()
                    == Some(
                        crate::strategy_table::IdentifierId::GeneralizedAdjustment.as_str(),
                    )
                && cache.identification.query == CausalQuery::AverageEffect(query.clone())
                && cache.identification.estimands.len() == 2 =>
            {
                Some(Box::new(CheckedUnknownTieredAverageOperation {
                    query: query.clone(),
                    background: background.clone(),
                    identification: cache.identification.clone(),
                    identifier: crate::strategy_table::IdentifierId::GeneralizedAdjustment,
                    estimator: crate::strategy_table::EstimatorId::LinearAdjustmentAte,
                    physical: plan.clone(),
                }))
            }
            _ => None,
        };
        let checked_bayesian_class_conditional =
            crate::analysis::CheckedBayesianClassConditional::seal(&analysis, &plan)?;
        let bayesian_temporal_class_effect_operation =
            Self::seal_bayesian_temporal_class_effect(&analysis, &plan)?;
        let temporal_graph_posterior_effect_operation =
            Self::seal_temporal_graph_posterior_effect(&analysis, &plan)?;
        let execution = if let Some(operation) = checked_unknown_tiered_average {
            PreparedExecution::UnknownTieredAverage(operation)
        } else if let Some(operation) = checked_linear {
            PreparedExecution::CheckedLinear(operation)
        } else if let Some(operation) = checked_glm {
            PreparedExecution::CheckedGlmAdjustment(operation)
        } else if let Some(operation) = checked_rd {
            PreparedExecution::CheckedRd(operation)
        } else if let Some(operation) = checked_propensity {
            PreparedExecution::CheckedPropensity(operation)
        } else if let Some(operation) = checked_conditional {
            PreparedExecution::CheckedConditional(operation)
        } else if let Some(operation) = checked_graph_posterior_effect {
            PreparedExecution::GraphPosteriorEffect(operation)
        } else if let Some(operation) = checked_class_graph_posterior_effect {
            PreparedExecution::ClassGraphPosteriorEffect(operation)
        } else if let Some(operation) = checked_bayesian_graph_posterior_ate {
            PreparedExecution::BayesianGraphPosteriorAte(operation)
        } else if let Some(operation) = checked_static_class_effect {
            PreparedExecution::StaticClassEffect(operation)
        } else if let Some(operation) = checked_aipw {
            PreparedExecution::CheckedAipw(operation)
        } else if let Some(operation) = checked_frontdoor_linear {
            PreparedExecution::FrontDoorLinear(operation)
        } else if let Some(operation) = checked_iv {
            PreparedExecution::Iv(operation)
        } else if let Some(operation) = checked_counterfactual {
            PreparedExecution::Counterfactual(operation)
        } else if let Some(operation) = nested_counterfactual {
            PreparedExecution::NestedCounterfactual(operation)
        } else if let Some(operation) = derivative_response_operation {
            PreparedExecution::DerivativeResponse(operation)
        } else if let Some(operation) = cell_aipw_response_operation {
            PreparedExecution::CellAipwResponse(operation)
        } else if let Some(operation) = static_dag_response_operation {
            PreparedExecution::StaticDagResponse(operation)
        } else if let Some(operation) = checked_bayesian_static_mediation {
            PreparedExecution::BayesianStaticMediation(operation)
        } else if let Some(operation) = checked_static_mediation {
            PreparedExecution::StaticMediation(operation)
        } else if let Some(operation) = checked_attribution {
            PreparedExecution::Attribution(operation)
        } else if let Some(operation) = temporal_dag_response_operation {
            PreparedExecution::TemporalDagResponse(operation)
        } else if let Some(operation) = checked_interference {
            PreparedExecution::Interference(operation)
        } else if let Some(operation) = checked_randomized {
            PreparedExecution::Randomized(operation)
        } else if let Some(operation) = checked_policy_value {
            PreparedExecution::PolicyValue(operation)
        } else if let Some(operation) = checked_continuous_dose {
            PreparedExecution::ContinuousDoseResponse(operation)
        } else if let Some(operation) = checked_panel_did {
            PreparedExecution::PanelDid(operation)
        } else if let Some(operation) = checked_synthetic_control {
            PreparedExecution::SyntheticControl(operation)
        } else if let Some(operation) = checked_local_polynomial_ratio {
            PreparedExecution::LocalPolynomialRatio(operation)
        } else if let Some(operation) = checked_survival {
            PreparedExecution::Survival(operation)
        } else if let Some(operation) = checked_longitudinal_regime {
            PreparedExecution::LongitudinalRegime(operation)
        } else if let Some(operation) = temporal_mediation_operation {
            PreparedExecution::TemporalMediation(operation)
        } else if let Some(operation) = temporal_dag_effect_operation {
            PreparedExecution::TemporalDagEffect(operation)
        } else if let Some(operation) = bayesian_temporal_dag_effect_operation {
            PreparedExecution::BayesianTemporalDagEffect(Box::new(operation))
        } else if let Some(operation) = temporal_class_effect_operation {
            PreparedExecution::TemporalClassEffect(operation)
        } else if let Some(operation) = distribution_operation {
            PreparedExecution::Distribution(operation)
        } else if let Some(operation) = functional_effect_operation {
            PreparedExecution::FunctionalEffect(operation)
        } else if let Some(operation) = path_specific_effect_operation {
            PreparedExecution::PathSpecificEffect(operation)
        } else if let Some(operation) = checked_admg_graph_posterior_response {
            PreparedExecution::AdmgGraphPosteriorResponse(operation)
        } else if let Some(operation) = admg_response_curve_operation {
            PreparedExecution::AdmgResponseCurve(operation)
        } else if let Some(operation) = bayesian_basis_ate_execution {
            PreparedExecution::BayesianBasisAte(operation)
        } else if let Some(operation) = bayesian_basis_cate_execution {
            PreparedExecution::BayesianBasisCate(operation)
        } else if let Some(operation) = bayesian_robust_ate_execution {
            PreparedExecution::BayesianRobustAte(operation)
        } else if let Some(operation) = bayesian_gcomp_execution {
            PreparedExecution::BayesianGcomp(operation)
        } else if let Some(operation) = bayesian_conditional_execution {
            PreparedExecution::BayesianConditional(operation)
        } else if let Some(operation) = checked_response_curve {
            PreparedExecution::StaticResponseCurve(operation)
        } else if let Some(operation) = checked_bayesian_class_conditional {
            PreparedExecution::BayesianClassConditional(operation)
        } else if let Some(operation) = checked_temporal_class_response {
            PreparedExecution::TemporalClassResponse(operation)
        } else if let Some(operation) = checked_temporal_graph_posterior_response {
            PreparedExecution::TemporalGraphPosteriorResponse(operation)
        } else if let Some(operation) = temporal_class_mediation_operation {
            PreparedExecution::TemporalClassMediation(operation)
        } else if let Some(operation) = bayesian_temporal_class_effect_operation {
            PreparedExecution::BayesianTemporalClassEffect(operation)
        } else if let Some(operation) = temporal_graph_posterior_effect_operation {
            PreparedExecution::TemporalGraphPosteriorEffect(operation)
        } else if let Some(operation) = checked_static_class_response {
            PreparedExecution::StaticClassResponse(operation)
        } else if let Some(operation) = checked_graph_posterior_response {
            PreparedExecution::GraphPosteriorResponse(operation)
        } else if let Some(operation) =
            crate::analysis::execute::CheckedBayesianSpecialistOperation::checked(&analysis, &plan)?
        {
            PreparedExecution::BayesianSpecialist(operation)
        } else if let Some(operation) =
            crate::analysis::execute::CheckedTransportTrialOperation::checked(&analysis, &plan)?
        {
            PreparedExecution::TransportTrial(operation)
        } else {
            PreparedExecution::OrdinaryDispatch
        };
        Ok(PreparedStudy {
            state: SampledPreparedState {
                analysis,
                program_cache: std::sync::OnceLock::new(),
                plan,
                schema,
                modality,
                time_regularity,
                score_table,
                execution,
            },
        })
    }
}
