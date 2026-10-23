//! `Study::prepare`: compile once and seal the execution route.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::sealers::{SealContext, seal_execution};
use super::{
    CachedTemporalClassIdentification, IdentificationCache, PreparedModality, PreparedStudy,
    SampledPreparedState, build_admg_graph_posterior_response_identification_cache,
    build_dbn_posterior_identification_cache, build_dbn_posterior_mediation_identification_cache,
    build_dbn_posterior_response_identification_cache, build_graph_posterior_identification_cache,
    build_temporal_class_posterior_identification_cache, ensure_prepared_supported,
    multi_env_regularity,
};
use crate::accepted::GraphClass;
use crate::analysis::builder::DataInput;
use crate::analysis::execute::Study;
use crate::analysis::query_license::{graphless_design, is_graphless_design};
use crate::error::CausalError;
use crate::planner::PhysicalExecutionPlan;
use crate::strategy_table::EstimatorId;
use antecedent_core::{
    AverageEffectQuery, CausalQuery, CausalSchema, ExecutionContext, TemporalEffectQuery,
};
use antecedent_data::TableView;
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
    ///   `CoDetermined` tier closure)
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
    ///   posterior (`TemporalDag` atoms; `MeanCurve` none, one-coordinate
    ///   `InterventionResponse` none/cheap/full)
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
        let plan = Arc::new(self.compile(ctx)?);
        let counterfactual = self.compile_counterfactual()?;
        let nested_counterfactual = self.compile_nested_counterfactual()?;
        let (schema, modality, time_regularity) = self.prepared_modality()?;
        let analysis = self.with_identification_caches(&plan, ctx)?;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_IDENTIFY });
        }
        // `identify_caches` reports `identify.compute` once wherever it identifies:
        // at compile when the lowered program retains the products, or above
        // when the plan retained none. A sharp-RD prepare builds no cache and
        // reports nothing: its clicks do.
        let binding = plan.binding();
        let execution = seal_execution(&SealContext {
            source: self,
            analysis: &analysis,
            plan: &plan,
            binding: &binding,
            ctx,
            route: self.route_input_classified(),
            counterfactual: counterfactual.as_ref(),
            nested_counterfactual: nested_counterfactual.as_ref(),
        })?;
        let score_table = analysis.prepare_score_table(ctx)?;
        Ok(PreparedStudy {
            state: SampledPreparedState {
                analysis,
                program_cache: std::sync::OnceLock::new(),
                contract_cache: std::sync::OnceLock::new(),
                plan,
                schema,
                modality,
                time_regularity,
                score_table,
                execution,
            },
        })
    }

    /// The cross-world operation of a tabular counterfactual query; temporal
    /// graph classes carry theirs through the temporal routes.
    fn compile_counterfactual(
        &self,
    ) -> Result<Option<crate::gcm::CheckedCounterfactualOperation>, CausalError> {
        Ok(match (&self.graph, &self.query) {
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
        })
    }

    /// The cross-world operation of a tabular nested counterfactual query.
    fn compile_nested_counterfactual(
        &self,
    ) -> Result<Option<crate::gcm::NestedCounterfactualOperation>, CausalError> {
        Ok(match (&self.graph, &self.query) {
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
        })
    }

    /// Schema, modality, and sampling regularity frozen by this prepare.
    fn prepared_modality(
        &self,
    ) -> Result<
        (CausalSchema, PreparedModality, Option<antecedent_data::SamplingRegularity>),
        CausalError,
    > {
        Ok(match &self.data {
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
        })
    }

    /// The study with the identification caches this query and data modality
    /// freeze at prepare time installed.
    ///
    /// A plan whose lowered program retained its identification at compile is
    /// installed from that program and is not identified again. Every other plan
    /// is identified here by [`Self::identify_caches`].
    fn with_identification_caches(
        &self,
        plan: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<Study, CausalError> {
        let mut analysis = self.clone();
        clear_identification_caches(&mut analysis);
        let retained = plan
            .lowered()
            .and_then(crate::planner::lowered::LoweredProgram::retained_identification);
        match retained {
            Some(retained) => {
                for cache in retained.caches() {
                    install_cache(&mut analysis, cache);
                }
            }
            None => {
                for cache in &self.identify_caches(plan, ctx)? {
                    install_cache(&mut analysis, cache);
                }
            }
        }
        Ok(analysis)
    }

    /// Identify the caches this query and data modality freeze at prepare time,
    /// without installing them. Compile calls this to retain identification with a
    /// lowered program; preparation calls it only for a plan that retained none.
    ///
    /// # Errors
    ///
    /// Identification refuses, or a graph-posterior coordinate is not licensed on
    /// the prepared handle.
    pub(crate) fn identify_caches(
        &self,
        plan: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<Vec<IdentificationCache>, CausalError> {
        let mut caches = Vec::new();
        if let Some(posterior) = self.graph_posterior.as_ref() {
            self.identify_graph_posterior_caches(posterior, ctx, &mut caches)?;
        } else {
            self.identify_single_graph_caches(plan, &mut caches)?;
            // The posterior builders report `identify.compute` themselves; the
            // single-graph, class, temporal and transport products identify
            // without a progress hook, so report once when any was built.
            if !caches.is_empty() {
                crate::analysis::execute::report_identify_compute(ctx);
            }
        }
        Ok(caches)
    }

    /// Identify the caches of a study that carries a graph posterior, refusing the
    /// coordinates the prepared handle does not license.
    fn identify_graph_posterior_caches(
        &self,
        posterior: &antecedent_discovery::GraphPosterior,
        ctx: &ExecutionContext,
        caches: &mut Vec<IdentificationCache>,
    ) -> Result<(), CausalError> {
        match (&self.data, &self.query) {
            (DataInput::Tabular(_), CausalQuery::AverageEffect(query)) => {
                caches.push(IdentificationCache::GraphPosterior(Arc::new(
                    build_graph_posterior_identification_cache(posterior, query, ctx)?,
                )));
            }
            (DataInput::Tabular(_), CausalQuery::ConditionalEffect(query)) => {
                caches.push(IdentificationCache::GraphPosterior(Arc::new(
                    build_graph_posterior_identification_cache(posterior, &query.inner, ctx)?,
                )));
            }
            (DataInput::Tabular(_), CausalQuery::Response(query)) if !query.is_temporal() => {
                caches
                    .push(Self::identify_tabular_response_posterior_cache(posterior, query, ctx)?);
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(query),
            ) => {
                let variables: Vec<_> =
                    data.schema().variables().iter().map(|variable| variable.id).collect();
                caches.push(
                    self.identify_temporal_effect_posterior_cache(
                        posterior, &variables, query, ctx,
                    )?,
                );
            }
            (DataInput::Temporal(data) | DataInput::Event(data), CausalQuery::Mediation(query)) => {
                let variables: Vec<_> =
                    data.schema().variables().iter().map(|variable| variable.id).collect();
                caches.push(self.identify_temporal_mediation_posterior_cache(
                    posterior, &variables, query, ctx,
                )?);
            }
            (DataInput::Temporal(data) | DataInput::Event(data), CausalQuery::Response(query))
                if query.is_temporal() =>
            {
                crate::analysis::execute::dbn_posterior_response_supported(query)?;
                let variables: Vec<_> =
                    data.schema().variables().iter().map(|variable| variable.id).collect();
                self.identify_temporal_response_posterior_cache(
                    posterior, &variables, query, ctx, caches,
                )?;
            }
            (
                DataInput::Temporal(_) | DataInput::Event(_),
                CausalQuery::NestedCounterfactual(_)
                | CausalQuery::PathSpecific(_)
                | CausalQuery::Counterfactual(_)
                | CausalQuery::Distribution(_),
            ) => {
                // Identification runs at execute, on the cross-world probe, so the
                // cache matches the producer rather than an average-effect witness.
            }
            _ => {
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
        Ok(())
    }

    /// Identify the caches of a study with a single supplied or discovered graph.
    fn identify_single_graph_caches(
        &self,
        plan: &PhysicalExecutionPlan,
        caches: &mut Vec<IdentificationCache>,
    ) -> Result<(), CausalError> {
        let mut push = |cache: Option<IdentificationCache>| caches.extend(cache);
        match (&self.data, &self.query) {
            (_, CausalQuery::Discover(_)) => {
                // A structure search identifies no contrast; its product states the
                // observational law it reads and the premises it is correct under.
                push(static_cache(self.prepare_static_identification(plan)?));
            }
            (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Mediation(_)) => {
                push(temporal(self.prepare_temporal_mediation_identification()?));
                push(temporal_class(self.prepare_temporal_class_identification()?));
            }
            (DataInput::Tabular(_), CausalQuery::Transport(query)) => {
                let diagram = self.selection_diagram.as_ref().ok_or(CausalError::Unsupported {
                    message: "TransportQuery prepare requires a selection diagram",
                })?;
                let identification =
                    crate::analysis::execute::live_transport_identification(diagram, query)?;
                push(Some(IdentificationCache::Transport(Arc::new(identification))));
            }
            (DataInput::Tabular(_), query) if is_graphless_design(query) => {
                // A design has no graph class to inspect, so the graph-only PAG/CPDAG
                // cache builders below do not apply. Its identification is typed by the
                // design itself and is cached at prepare when the design identifies once.
                if graphless_design(query).is_some_and(|design| design.identified_at_prepare) {
                    push(static_cache(self.prepare_static_identification(plan)?));
                }
            }
            (DataInput::Tabular(_), _) => {
                push(match &plan.linear {
                    Some(linear) => {
                        Some(IdentificationCache::Static(Arc::clone(&linear.identification)))
                    }
                    None => static_cache(self.prepare_static_identification(plan)?),
                });
                push(
                    self.prepare_pag_identification(plan)?
                        .map(|cache| IdentificationCache::Pag(Arc::new(cache))),
                );
                push(
                    self.prepare_cpdag_identification(plan)?
                        .map(|cache| IdentificationCache::Cpdag(Arc::new(cache))),
                );
            }
            (DataInput::Panel(_), CausalQuery::TemporalEffect(_) | CausalQuery::Response(_)) => {
                if self.graph.class().is_incomplete_temporal() {
                    push(temporal_class(self.prepare_temporal_class_identification()?));
                } else {
                    push(temporal(self.prepare_temporal_identification()?));
                }
            }
            (
                DataInput::Temporal(_) | DataInput::Event(_),
                CausalQuery::AnomalyAttribution(_) | CausalQuery::ChangeAttribution(_),
            ) => {
                push(static_cache(self.prepare_static_identification(plan)?));
            }
            (_, _) => {
                push(temporal(self.prepare_temporal_identification()?));
                push(temporal_class(self.prepare_temporal_class_identification()?));
            }
        }
        Ok(())
    }

    /// The graph-posterior cache of a non-temporal tabular response query.
    fn identify_tabular_response_posterior_cache(
        posterior: &antecedent_discovery::GraphPosterior,
        query: &antecedent_core::ResponseQuery,
        ctx: &ExecutionContext,
    ) -> Result<IdentificationCache, CausalError> {
        Ok(IdentificationCache::GraphPosterior(Arc::new(
            if matches!(
                posterior.atom_kind,
                antecedent_discovery::GraphPosteriorAtomKind::Admg
                    | antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg
            ) {
                build_admg_graph_posterior_response_identification_cache(posterior, query, ctx)?
            } else {
                let (treatment, outcome) =
                    crate::analysis::proof_support::require_primary_pair(&query.functional)?;
                build_graph_posterior_identification_cache(
                    posterior,
                    &AverageEffectQuery::binary_ate(treatment, outcome),
                    ctx,
                )?
            },
        )))
    }

    /// The temporal-effect graph-posterior cache for the atom kind of `posterior`.
    fn identify_temporal_effect_posterior_cache(
        &self,
        posterior: &antecedent_discovery::GraphPosterior,
        variables: &[antecedent_core::VariableId],
        query: &TemporalEffectQuery,
        ctx: &ExecutionContext,
    ) -> Result<IdentificationCache, CausalError> {
        Ok(if crate::analysis::execute::is_class_atom(posterior.atom_kind) {
            IdentificationCache::TemporalClassPosterior(Arc::new(
                build_temporal_class_posterior_identification_cache(
                    posterior,
                    variables,
                    query,
                    self.max_completions,
                    ctx,
                )?,
            ))
        } else {
            IdentificationCache::DbnPosterior(Arc::new(build_dbn_posterior_identification_cache(
                posterior, variables, query, ctx,
            )?))
        })
    }

    /// The temporal-mediation graph-posterior cache for the atom kind of `posterior`.
    fn identify_temporal_mediation_posterior_cache(
        &self,
        posterior: &antecedent_discovery::GraphPosterior,
        variables: &[antecedent_core::VariableId],
        query: &antecedent_core::MediationQuery,
        ctx: &ExecutionContext,
    ) -> Result<IdentificationCache, CausalError> {
        Ok(if crate::analysis::execute::is_class_atom(posterior.atom_kind) {
            let horizon = query.horizons.first().copied().ok_or_else(|| CausalError::Compile {
                message: "temporal class graph-posterior mediation requires a horizon".into(),
            })?;
            let mut witness = TemporalEffectQuery::pulse(query.treatment, query.outcome, 1.0);
            witness.horizon_steps = horizon;
            IdentificationCache::TemporalClassPosterior(Arc::new(
                build_temporal_class_posterior_identification_cache(
                    posterior,
                    variables,
                    &witness,
                    self.max_completions,
                    ctx,
                )?,
            ))
        } else {
            IdentificationCache::DbnPosterior(Arc::new(
                build_dbn_posterior_mediation_identification_cache(
                    posterior, variables, query, ctx,
                )?,
            ))
        })
    }

    /// The temporal-response graph-posterior caches for the atom kind of `posterior`.
    fn identify_temporal_response_posterior_cache(
        &self,
        posterior: &antecedent_discovery::GraphPosterior,
        variables: &[antecedent_core::VariableId],
        query: &antecedent_core::ResponseQuery,
        ctx: &ExecutionContext,
        caches: &mut Vec<IdentificationCache>,
    ) -> Result<(), CausalError> {
        if crate::analysis::execute::is_class_atom(posterior.atom_kind) {
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
                variables,
                &witness,
                self.max_completions,
                ctx,
            )?;
            if let Some(atom) = cache.class_atoms.first() {
                caches.push(IdentificationCache::TemporalClass(Arc::new(
                    CachedTemporalClassIdentification {
                        envelope: atom.envelope.clone(),
                        by_horizon: vec![(horizon, atom.envelope.clone())],
                    },
                )));
            }
            caches.push(IdentificationCache::TemporalClassPosterior(Arc::new(cache)));
        } else {
            let estimator_id = EstimatorId::temporal_response_for(&self.inference);
            caches.push(IdentificationCache::DbnPosterior(Arc::new(
                build_dbn_posterior_response_identification_cache(
                    posterior,
                    variables,
                    query,
                    estimator_id,
                    ctx,
                )?,
            )));
        }
        Ok(())
    }
}

fn static_cache(cache: Option<super::CachedStaticIdentification>) -> Option<IdentificationCache> {
    cache.map(|cache| IdentificationCache::Static(Arc::new(cache)))
}

fn temporal(cache: Option<super::CachedTemporalIdentification>) -> Option<IdentificationCache> {
    cache.map(|cache| IdentificationCache::Temporal(Arc::new(cache)))
}

fn temporal_class(cache: Option<CachedTemporalClassIdentification>) -> Option<IdentificationCache> {
    cache.map(|cache| IdentificationCache::TemporalClass(Arc::new(cache)))
}

/// Empty every identification slot a prepare installs, so a retained or freshly
/// identified set replaces whatever the source study carried.
fn clear_identification_caches(analysis: &mut Study) {
    analysis.identification_cache = None;
    analysis.pag_identification_cache = None;
    analysis.cpdag_identification_cache = None;
    analysis.temporal_identification_cache = None;
    analysis.temporal_class_identification_cache = None;
    analysis.graph_posterior_identification_cache = None;
    analysis.dbn_posterior_identification_cache = None;
    analysis.temporal_class_posterior_identification_cache = None;
    if analysis.config.transport_identification_cache.is_some() {
        Arc::make_mut(&mut analysis.config).transport_identification_cache = None;
    }
}

/// Install one identified cache into the study slot that holds it.
fn install_cache(analysis: &mut Study, cache: &IdentificationCache) {
    match cache {
        IdentificationCache::Static(cache) => {
            analysis.identification_cache = Some(Arc::clone(cache));
        }
        IdentificationCache::Pag(cache) => {
            analysis.pag_identification_cache = Some(Arc::clone(cache));
        }
        IdentificationCache::Cpdag(cache) => {
            analysis.cpdag_identification_cache = Some(Arc::clone(cache));
        }
        IdentificationCache::Temporal(cache) => {
            analysis.temporal_identification_cache = Some(Arc::clone(cache));
        }
        IdentificationCache::TemporalClass(cache) => {
            analysis.temporal_class_identification_cache = Some(Arc::clone(cache));
        }
        IdentificationCache::Transport(cache) => {
            Arc::make_mut(&mut analysis.config).transport_identification_cache =
                Some(Arc::clone(cache));
        }
        IdentificationCache::GraphPosterior(cache) => {
            analysis.graph_posterior_identification_cache = Some(Arc::clone(cache));
        }
        IdentificationCache::DbnPosterior(cache) => {
            analysis.dbn_posterior_identification_cache = Some(Arc::clone(cache));
        }
        IdentificationCache::TemporalClassPosterior(cache) => {
            analysis.temporal_class_posterior_identification_cache = Some(Arc::clone(cache));
        }
    }
}
