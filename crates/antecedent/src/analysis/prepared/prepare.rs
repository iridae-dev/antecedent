//! `Study::prepare`: compile once and seal the execution route.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::sealers::{SealContext, seal_deferred_execution, seal_execution};
use super::{
    CachedTemporalClassIdentification, IdentificationCache, PreparedModality, PreparedStudy,
    SampledPreparedState, build_admg_graph_posterior_response_identification_cache,
    build_dbn_posterior_identification_cache, build_dbn_posterior_mediation_identification_cache,
    build_dbn_posterior_response_identification_cache, build_graph_posterior_identification_cache,
    build_temporal_class_posterior_identification_cache, ensure_prepared_supported,
    multi_env_regularity,
};
use crate::accepted::{GraphClass, GraphRef};
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
        antecedent_estimate::prepare::with_preparation_budget(
            crate::planner::PhysicalExecutionPlan::preparation_budget(ctx),
            || self.prepare_metered(ctx, true, None),
        )
    }

    /// Compile and identify without fitting statistical operations or scores.
    ///
    /// # Errors
    /// Returns the same declaration, identification, and admission errors as preparation.
    pub fn prepare_deferred(&self, ctx: &ExecutionContext) -> Result<PreparedStudy, CausalError> {
        antecedent_estimate::prepare::with_preparation_budget(
            PhysicalExecutionPlan::preparation_budget(ctx),
            || self.prepare_metered(ctx, false, None),
        )
    }

    /// A pure prepared branch may offer its immutable proof, while all physical
    /// quotes and statistical declarations are rebuilt under the caller's context.
    pub(super) fn prepare_deferred_retaining(
        &self,
        previous: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<PreparedStudy, CausalError> {
        antecedent_estimate::prepare::with_preparation_budget(
            PhysicalExecutionPlan::preparation_budget(ctx),
            || self.prepare_metered(ctx, false, Some(previous)),
        )
    }

    /// The body of preparation, run under the plan's preparation budget.
    fn prepare_metered(
        &self,
        ctx: &ExecutionContext,
        materialized: bool,
        previous: Option<&PhysicalExecutionPlan>,
    ) -> Result<PreparedStudy, CausalError> {
        // A functional query under cheap or full validation is prepared as the same query at
        // validation none: its program is the one every refit re-executes, and the handle keeps
        // the requested suite so each estimate validates (see `functional_validation`).
        if crate::analysis::functional_validation::requested(self) {
            let mut base = self.clone();
            base.settings.refute = crate::analysis::builder::RefuteSuite::None;
            let mut handle = base.prepare_metered(ctx, materialized, previous)?;
            handle.analysis.settings.refute = self.settings.refute;
            handle.program_cache = std::sync::OnceLock::new();
            handle.contract_cache = std::sync::OnceLock::new();
            handle.inspection_header = std::sync::OnceLock::new();
            handle.seal_inspection_metadata()?;
            return Ok(handle);
        }
        ensure_prepared_supported(self)?;
        if let DataInput::Tabular(table) = &self.data {
            self.refuse_unread_sampling_weights_for_route(table)?;
        }
        let plan = Arc::new(self.compile_with_retained_identification(
            &crate::analysis::lowering::CompileGoal::Estimate,
            ctx,
            previous,
        )?);
        let posterior_programs = crate::analysis::contract::graph_posterior_programs(
            self,
            plan.logical.record.estimator.as_deref(),
            ctx,
        )?
        .map(|programs| {
            Ok::<_, CausalError>(Arc::new(super::RetainedPosteriorPrograms {
                query: crate::analysis::contract_identity::canonical_query_bytes(&self.query)?,
                programs,
            }))
        })
        .transpose()?;
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
        let seal_context = SealContext {
            source: self,
            analysis: &analysis,
            plan: &plan,
            binding: &binding,
            ctx,
            route: self.route_input_classified(),
            counterfactual: counterfactual.as_ref(),
            nested_counterfactual: nested_counterfactual.as_ref(),
            previous: None,
        };
        let execution = if materialized {
            seal_execution(&seal_context)?
        } else {
            seal_deferred_execution(&seal_context)?
        };
        // A plan that streams its design holds no rows to cross-fit scores over: it keeps no
        // score table, and a retarget refuses as unavailable.
        let score_table = if !materialized || plan.streams_design() {
            None
        } else {
            analysis.prepare_score_table(ctx)?
        };
        let ready = PreparedStudy {
            state: SampledPreparedState {
                analysis,
                materialized,
                posterior_programs,
                program_cache: std::sync::OnceLock::new(),
                contract_cache: std::sync::OnceLock::new(),
                inspection_header: std::sync::OnceLock::new(),
                plan,
                schema,
                modality,
                time_regularity,
                score_table,
                execution,
                temporal_log: antecedent_state::InvalidationLog::new(),
                retained: None,
            },
        };
        ready.seal_inspection_metadata()?;
        Ok(ready)
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
            // A counterfactual on a structure latent confounding or an equivalence class leaves
            // open is identified (or run per completion) at execution; it fits no shared
            // exogenous operation.
            (graph, CausalQuery::Counterfactual(_))
                if matches!(
                    graph.class(),
                    GraphClass::Admg
                        | GraphClass::TemporalAdmg
                        | GraphClass::Pag
                        | GraphClass::Cpdag
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
        // A graph posterior names no single graph: each atom answers the natural direct effect
        // on its own structure at every execution.
        if self.graph_posterior.is_some() {
            return Ok(None);
        }
        Ok(match (&self.graph, &self.query) {
            (graph, CausalQuery::NestedCounterfactual(_))
                if matches!(
                    graph.class(),
                    GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag
                ) =>
            {
                None
            }
            // The natural direct effect on a mixed graph is identified from the graph and
            // read by the functional plug-in; it fits no shared-exogenous operation.
            (graph, CausalQuery::NestedCounterfactual(_))
                if matches!(
                    graph.class(),
                    GraphClass::Admg
                        | GraphClass::TemporalAdmg
                        | GraphClass::Cpdag
                        | GraphClass::Pag
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
    pub(crate) fn with_identification_caches(
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
            self.identify_single_graph_caches(plan, ctx, &mut caches)?;
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
        if posterior.is_lagged_temporal_admg() {
            return self.identify_lagged_admg_posterior_caches(posterior, ctx, caches);
        }
        match (&self.data, &self.query) {
            (DataInput::Tabular(_), CausalQuery::AverageEffect(query)) => {
                caches.push(IdentificationCache::GraphPosterior(Arc::new(
                    build_graph_posterior_identification_cache(posterior, query, ctx)?,
                )));
            }
            // A latent-confounded conditional effect, a derivative or multi-coordinate response
            // and a counterfactual are identified on each atom's own structure by the executor
            // of that structure at every execution; an average-effect witness would state
            // another question.
            (DataInput::Tabular(_), CausalQuery::ConditionalEffect(_))
                if matches!(
                    posterior.atom_kind(),
                    antecedent_discovery::GraphPosteriorAtomKind::Admg
                        | antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg
                ) => {}
            (DataInput::Tabular(_), CausalQuery::Response(query))
                if crate::analysis::execute::atom_route_response(query) => {}
            (DataInput::Tabular(_), CausalQuery::ConditionalEffect(query)) => {
                caches.push(IdentificationCache::GraphPosterior(Arc::new(
                    build_graph_posterior_identification_cache(posterior, &query.inner, ctx)?,
                )));
            }
            (DataInput::Tabular(_), CausalQuery::Response(query)) if !query.is_temporal() => {
                caches
                    .push(Self::identify_tabular_response_posterior_cache(posterior, query, ctx)?);
            }
            (DataInput::Tabular(_), CausalQuery::Transport(query)) => {
                caches.push(self.identify_transport_posterior_cache(posterior, query, ctx)?);
            }
            (DataInput::Tabular(_), CausalQuery::Interference(query)) => {
                caches.push(self.identify_interference_posterior_cache(posterior, query, ctx)?);
            }
            // A one-modifier conditional effect over TemporalDag atoms identifies each atom's
            // regressions at every execution, with the interval of the whole procedure.
            (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::TemporalEffect(query))
                if crate::analysis::execute::dbn_atoms(posterior)
                    && !query.effect_modifiers.is_empty() => {}
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
                // A posterior of temporal PAGs identifies each atom on the window of its own
                // template at every execution, as a supplied temporal PAG does.
                if posterior.atom_kind() != antecedent_discovery::GraphPosteriorAtomKind::Pag {
                    caches.push(self.identify_temporal_mediation_posterior_cache(
                        posterior, &variables, query, ctx,
                    )?);
                }
            }
            (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Response(_))
                if crate::analysis::execute::dbn_atoms(posterior)
                    && crate::analysis::execute::dbn_route_query(&self.query) => {}
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
                DataInput::Tabular(_),
                CausalQuery::Mediation(_)
                | CausalQuery::PathSpecific(_)
                | CausalQuery::NestedCounterfactual(_)
                | CausalQuery::Counterfactual(_)
                | CausalQuery::Distribution(_)
                | CausalQuery::AnomalyAttribution(_)
                | CausalQuery::ChangeAttribution(_),
            )
            | (
                DataInput::Temporal(_) | DataInput::Event(_),
                CausalQuery::NestedCounterfactual(_)
                | CausalQuery::PathSpecific(_)
                | CausalQuery::Counterfactual(_)
                | CausalQuery::Distribution(_)
                | CausalQuery::AnomalyAttribution(_)
                | CausalQuery::ChangeAttribution(_),
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

    /// The design identifies the contrast; posterior atoms state only their membership.
    fn identify_interference_posterior_cache(
        &self,
        posterior: &antecedent_discovery::GraphPosterior,
        query: &antecedent_core::InterferenceQuery,
        ctx: &ExecutionContext,
    ) -> Result<IdentificationCache, CausalError> {
        // The design identifies the contrast; the atoms only state their membership.
        Ok(IdentificationCache::GraphPosterior(Arc::new(
                    crate::analysis::identification_cache::build_interference_graph_posterior_identification_cache(
                        posterior,
                        query,
                        &self.settings.inference,
                        ctx,
                    )?,
                )))
    }

    /// A transport atom uses its own selection diagram rather than an effect witness.
    fn identify_transport_posterior_cache(
        &self,
        posterior: &antecedent_discovery::GraphPosterior,
        query: &antecedent_core::TransportQuery,
        ctx: &ExecutionContext,
    ) -> Result<IdentificationCache, CausalError> {
        // Every atom is a selection diagram: sID decides it on its own graph.
        let targets: Vec<_> = self
            .selection_diagram
            .as_ref()
            .map(|diagram| diagram.selection_targets().to_vec())
            .unwrap_or_default();
        Ok(IdentificationCache::GraphPosterior(Arc::new(
                    crate::analysis::identification_cache::build_transport_graph_posterior_identification_cache(
                        posterior, query, &targets, ctx,
                    )?,
                )))
    }

    fn identify_lagged_admg_posterior_caches(
        &self,
        posterior: &antecedent_discovery::GraphPosterior,
        ctx: &ExecutionContext,
        caches: &mut Vec<IdentificationCache>,
    ) -> Result<(), CausalError> {
        // Lagged temporal-ADMG atoms are identified on the unrolled window of their own
        // templates: a contrast once per atom here, a mediation (three components per
        // atom) at each execution.
        let (DataInput::Temporal(data) | DataInput::Event(data)) = &self.data else {
            return Err(CausalError::Unsupported {
                message: "a lagged temporal-ADMG graph posterior reads a time series",
            });
        };
        match &self.query {
            CausalQuery::TemporalEffect(query) => {
                let variables: Vec<_> =
                    data.schema().variables().iter().map(|variable| variable.id).collect();
                caches.push(IdentificationCache::GraphPosterior(Arc::new(
                        crate::analysis::identification_cache::build_temporal_admg_posterior_identification_cache(
                            posterior, &variables, query, ctx,
                        )?,
                    )));
                Ok(())
            }
            CausalQuery::Mediation(_) => Ok(()),
            _ => Err(CausalError::Unsupported {
                message: "a lagged temporal-ADMG graph posterior answers a pulse, sustained \
                              or conditional temporal contrast and a temporal mediation",
            }),
        }
    }

    /// Identify the caches of a study with a single supplied or discovered graph.
    fn identify_single_graph_caches(
        &self,
        plan: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        caches: &mut Vec<IdentificationCache>,
    ) -> Result<(), CausalError> {
        let mut push = |cache: Option<IdentificationCache>| caches.extend(cache);
        match (&self.data, &self.query) {
            (_, CausalQuery::Discover(_)) => {
                // A structure search identifies no contrast; its product states the
                // observational law it reads and the premises it is correct under.
                push(static_cache(self.prepare_static_identification(plan)?));
            }
            (_, CausalQuery::Association(_)) => {
                // An association claims no identification: there is no product to cache, and
                // the lowered program records that the layer is not claimed.
            }
            (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Mediation(_)) => {
                push(temporal(self.prepare_temporal_mediation_identification()?));
                push(temporal_class(self.prepare_temporal_class_identification(ctx)?));
            }
            (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::TemporalEffect(query))
                if matches!(self.graph.graph_ref(), GraphRef::TemporalAdmg(_)) =>
            {
                // A temporal ADMG contrast is identified once on its unrolled window; the
                // functional over the window coordinates is the claim execution reads.
                push(temporal(Some(self.prepare_temporal_admg_identification(query)?)));
            }
            (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Response(query))
                if query.is_temporal()
                    && matches!(self.graph.graph_ref(), GraphRef::TemporalAdmg(_)) =>
            {
                // A temporal ADMG derivative is identified once by an adjustment set of the
                // unrolled window; the estimand over the window coordinates is the claim
                // execution reads.
                push(temporal(Some(self.prepare_temporal_admg_response_identification(query)?)));
            }
            (DataInput::Tabular(_), CausalQuery::Transport(_))
                if matches!(self.graph.class(), GraphClass::Cpdag | GraphClass::Pag) =>
            {
                // sID decides every member of the class on its own selection diagram.
                push(self.identify_class_transport(ctx)?);
            }
            (DataInput::Tabular(_), CausalQuery::Transport(query)) => {
                let diagram = self.selection_diagram.as_ref().ok_or(CausalError::Unsupported {
                    message: "TransportQuery prepare requires a selection diagram",
                })?;
                let identification =
                    crate::analysis::execute::live_transport_identification(diagram, query, ctx)?;
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
                    None => static_cache(self.prepare_static_identification_in(plan, Some(ctx))?),
                });
                push(
                    self.prepare_pag_identification(plan, Some(ctx))?
                        .map(|cache| IdentificationCache::Pag(Arc::new(cache))),
                );
                push(
                    self.prepare_cpdag_identification(plan, Some(ctx))?
                        .map(|cache| IdentificationCache::Cpdag(Arc::new(cache))),
                );
            }
            (DataInput::Panel(_), CausalQuery::TemporalEffect(_) | CausalQuery::Response(_)) => {
                if self.graph.class().is_incomplete_temporal() {
                    push(temporal_class(self.prepare_temporal_class_identification(ctx)?));
                } else {
                    push(temporal(self.prepare_temporal_identification()?));
                }
            }
            (
                DataInput::Temporal(_) | DataInput::Event(_),
                CausalQuery::AnomalyAttribution(_)
                | CausalQuery::ChangeAttribution(_)
                | CausalQuery::Attribute(_)
                | CausalQuery::UnitChange(_)
                | CausalQuery::MechanismChange(_)
                | CausalQuery::Simulate(_),
            ) => {
                push(static_cache(self.prepare_static_identification(plan)?));
            }
            (_, _) => {
                push(temporal(self.prepare_temporal_identification()?));
                push(temporal_class(self.prepare_temporal_class_identification(ctx)?));
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
                posterior.atom_kind(),
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
        Ok(if crate::analysis::execute::is_class_atom(posterior.atom_kind()) {
            IdentificationCache::TemporalClassPosterior(Arc::new(
                build_temporal_class_posterior_identification_cache(
                    posterior,
                    variables,
                    query,
                    self.declared.max_completions,
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
        Ok(if crate::analysis::execute::is_class_atom(posterior.atom_kind()) {
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
                    self.declared.max_completions,
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
        if crate::analysis::execute::is_class_atom(posterior.atom_kind()) {
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
            let mut cache = build_temporal_class_posterior_identification_cache(
                posterior,
                variables,
                &witness,
                self.declared.max_completions,
                ctx,
            )?;
            if let Some(atom) = cache.class_atoms.first() {
                cache.first_atom_class = Some(Arc::new(CachedTemporalClassIdentification {
                    envelope: atom.envelope.clone(),
                    by_horizon: vec![(horizon, atom.envelope.clone())],
                }));
            }
            caches.push(IdentificationCache::TemporalClassPosterior(Arc::new(cache)));
        } else {
            let estimator_id = EstimatorId::temporal_response_for(&self.settings.inference);
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

/// Empty the identification slot a prepare installs, so a retained or freshly
/// identified product replaces whatever the source study carried.
fn clear_identification_caches(analysis: &mut Study) {
    analysis.clear_identification();
}

/// Install one identified cache as the study's identification product.
///
/// The kinds are mutually exclusive, and a prepare produces one. Should a plan
/// ever hand over several, the first stands and the rest are dropped.
fn install_cache(analysis: &mut Study, cache: &IdentificationCache) {
    if analysis.identification.is_none() {
        analysis.install_identification(cache.clone());
    }
}

impl PreparedStudy {
    /// Materialize the retained statistical plan at an explicit execution boundary.
    /// Identification, declarations, and the physical plan remain those of preparation.
    ///
    /// # Errors
    /// Statistical, resource, or cancellation failures while building retained operations.
    pub fn materialize(&self, ctx: &ExecutionContext) -> Result<Self, CausalError> {
        self.materialize_scored(ctx, true)
    }

    /// Materialize for an execution that is about to estimate on the bound rows.
    ///
    /// A branch rebound from one that retained cross-fitted scores does not build them here:
    /// the estimate that follows constructs the same table from the same rows, and
    /// [`Self::retain_estimated_scores`] keeps it, so each branch constructs its scores once.
    /// The second value is `Some` while the table is still owed, saying whether it is read from a
    /// shared design.
    pub(crate) fn materialize_for_estimate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<(Self, Option<bool>), CausalError> {
        let owed = if self.materialized {
            None
        } else {
            self.retained.as_ref().and_then(|branch| branch.scored)
        };
        Ok((self.materialize_scored(ctx, owed.is_none())?, owed))
    }

    /// Keep the score table of the estimate this branch just ran as its prepared table, or
    /// build it when the estimate published none.
    pub(crate) fn retain_estimated_scores(
        &mut self,
        estimated: &crate::StudyResult,
        shared: bool,
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        self.score_table =
            match estimated.estimate.as_effect().and_then(|effect| effect.scores.table.clone()) {
                // The estimate reads the rows' own design, while the prepared table is named for
                // the shared design the branch it was rebound from read.
                Some(table) if shared => Some(table.named_for_shared_design()),
                Some(table) => Some(table),
                None => self.analysis.prepare_score_table(ctx)?,
            };
        self.program_cache = std::sync::OnceLock::new();
        self.contract_cache = std::sync::OnceLock::new();
        self.inspection_header = std::sync::OnceLock::new();
        self.seal_inspection_metadata()
    }

    fn materialize_scored(
        &self,
        ctx: &ExecutionContext,
        build_scores: bool,
    ) -> Result<Self, CausalError> {
        if self.materialized {
            return Ok(self.clone());
        }
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_IDENTIFY });
        }
        antecedent_estimate::prepare::with_preparation_budget(
            PhysicalExecutionPlan::preparation_budget(ctx),
            || {
                let mut ready = self.clone();
                let mut source = self.analysis.clone();
                if crate::analysis::functional_validation::requested(&source) {
                    source.settings.refute = crate::analysis::builder::RefuteSuite::None;
                }
                let counterfactual = source.compile_counterfactual()?;
                let nested_counterfactual = source.compile_nested_counterfactual()?;
                let binding = self.plan.binding();
                ready.execution = seal_execution(&SealContext {
                    source: &source,
                    analysis: &source,
                    plan: &self.plan,
                    binding: &binding,
                    ctx,
                    route: source.route_input_classified(),
                    counterfactual: counterfactual.as_ref(),
                    nested_counterfactual: nested_counterfactual.as_ref(),
                    previous: self.retained.as_ref().map(|retained| &retained.execution),
                })?;
                ready.retained = None;
                ready.score_table = if self.plan.streams_design() || !build_scores {
                    None
                } else {
                    source.prepare_score_table(ctx)?
                };
                if ctx.cancellation.is_cancelled() {
                    return Err(CausalError::Cancelled {
                        stage: crate::analysis::stage::STAGE_IDENTIFY,
                    });
                }
                ready.materialized = true;
                // Inspection may have cached a contract on the deferred
                // declaration. Materialization adds row-bound fit/score
                // payloads, so its executed contract must derive those actual
                // payloads rather than reuse the pre-execution projection.
                ready.program_cache = std::sync::OnceLock::new();
                ready.contract_cache = std::sync::OnceLock::new();
                ready.inspection_header = std::sync::OnceLock::new();
                ready.seal_inspection_metadata()?;
                Ok(ready)
            },
        )
    }
}
