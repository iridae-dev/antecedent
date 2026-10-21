//! `Study::prepare`: compile once and seal the execution route.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::sealers::{SealContext, seal_execution};
use super::{
    CachedTemporalClassIdentification, PreparedModality, PreparedStudy, SampledPreparedState,
    build_admg_graph_posterior_response_identification_cache,
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
    fn with_identification_caches(
        &self,
        plan: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<Study, CausalError> {
        let mut analysis = self.clone();
        match self.graph_posterior.as_ref() {
            Some(posterior) => {
                self.install_graph_posterior_caches(&mut analysis, posterior, ctx)?
            }
            None => self.install_identification_caches(&mut analysis, plan)?,
        }
        Ok(analysis)
    }

    /// Install the caches of a study that carries a graph posterior, refusing the coordinates
    /// the prepared handle does not license.
    fn install_graph_posterior_caches(
        &self,
        analysis: &mut Study,
        posterior: &antecedent_discovery::GraphPosterior,
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        match (&self.data, &self.query) {
            (DataInput::Tabular(_), CausalQuery::AverageEffect(query)) => {
                analysis.graph_posterior_identification_cache = Some(Arc::new(
                    build_graph_posterior_identification_cache(posterior, query, ctx)?,
                ));
            }
            (DataInput::Tabular(_), CausalQuery::ConditionalEffect(query)) => {
                analysis.graph_posterior_identification_cache = Some(Arc::new(
                    build_graph_posterior_identification_cache(posterior, &query.inner, ctx)?,
                ));
            }
            (DataInput::Tabular(_), CausalQuery::Response(query)) if !query.is_temporal() => {
                Self::install_tabular_response_posterior_cache(analysis, posterior, query, ctx)?;
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(query),
            ) => {
                let variables: Vec<_> =
                    data.schema().variables().iter().map(|variable| variable.id).collect();
                Self::install_temporal_effect_posterior_cache(
                    analysis, posterior, &variables, query, ctx,
                )?;
            }
            (DataInput::Temporal(data) | DataInput::Event(data), CausalQuery::Mediation(query)) => {
                let variables: Vec<_> =
                    data.schema().variables().iter().map(|variable| variable.id).collect();
                Self::install_temporal_mediation_posterior_cache(
                    analysis, posterior, &variables, query, ctx,
                )?;
            }
            (DataInput::Temporal(data) | DataInput::Event(data), CausalQuery::Response(query))
                if query.is_temporal() =>
            {
                crate::analysis::execute::dbn_posterior_response_supported(query)?;
                let variables: Vec<_> =
                    data.schema().variables().iter().map(|variable| variable.id).collect();
                self.install_temporal_response_posterior_cache(
                    analysis, posterior, &variables, query, ctx,
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

    /// Install the caches of a study with a single supplied or discovered graph.
    fn install_identification_caches(
        &self,
        analysis: &mut Study,
        plan: &PhysicalExecutionPlan,
    ) -> Result<(), CausalError> {
        match (&self.data, &self.query) {
            (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Mediation(_)) => {
                analysis.temporal_identification_cache =
                    self.prepare_temporal_mediation_identification()?.map(Arc::new);
                analysis.temporal_class_identification_cache =
                    self.prepare_temporal_class_identification()?.map(Arc::new);
            }
            (DataInput::Tabular(_), CausalQuery::Transport(query)) => {
                let diagram = self.selection_diagram.as_ref().ok_or(CausalError::Unsupported {
                    message: "TransportQuery prepare requires a selection diagram",
                })?;
                let identification =
                    crate::analysis::execute::live_transport_identification(diagram, query)?;
                Arc::make_mut(&mut analysis.config).transport_identification_cache =
                    Some(Arc::new(identification));
            }
            (DataInput::Tabular(_), query) if is_graphless_design(query) => {
                // A design has no graph class to inspect, so the graph-only PAG/CPDAG
                // cache builders below do not apply. Its identification is typed by the
                // design itself and is cached at prepare when the design identifies once.
                if graphless_design(query).is_some_and(|design| design.identified_at_prepare) {
                    analysis.identification_cache =
                        self.prepare_static_identification(plan)?.map(Arc::new);
                }
            }
            (DataInput::Tabular(_), _) => {
                analysis.identification_cache = match &plan.linear {
                    Some(linear) => Some(Arc::clone(&linear.identification)),
                    None => self.prepare_static_identification(plan)?.map(Arc::new),
                };
                analysis.pag_identification_cache =
                    self.prepare_pag_identification(plan)?.map(Arc::new);
                analysis.cpdag_identification_cache =
                    self.prepare_cpdag_identification(plan)?.map(Arc::new);
            }
            (DataInput::Panel(_), CausalQuery::TemporalEffect(_) | CausalQuery::Response(_)) => {
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
            ) => {
                analysis.identification_cache =
                    self.prepare_static_identification(plan)?.map(Arc::new);
            }
            (_, _) => {
                analysis.temporal_identification_cache =
                    self.prepare_temporal_identification()?.map(Arc::new);
                analysis.temporal_class_identification_cache =
                    self.prepare_temporal_class_identification()?.map(Arc::new);
            }
        }
        Ok(())
    }

    /// Install the graph-posterior cache of a non-temporal tabular response query.
    fn install_tabular_response_posterior_cache(
        analysis: &mut Study,
        posterior: &antecedent_discovery::GraphPosterior,
        query: &antecedent_core::ResponseQuery,
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        analysis.graph_posterior_identification_cache = Some(Arc::new(
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
        ));
        Ok(())
    }

    /// Install the temporal-effect graph-posterior cache for the atom kind of `posterior`.
    fn install_temporal_effect_posterior_cache(
        analysis: &mut Study,
        posterior: &antecedent_discovery::GraphPosterior,
        variables: &[antecedent_core::VariableId],
        query: &TemporalEffectQuery,
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        if crate::analysis::execute::is_class_atom(posterior.atom_kind) {
            analysis.temporal_class_posterior_identification_cache =
                Some(Arc::new(build_temporal_class_posterior_identification_cache(
                    posterior,
                    variables,
                    query,
                    analysis.max_completions,
                    ctx,
                )?));
        } else {
            analysis.dbn_posterior_identification_cache = Some(Arc::new(
                build_dbn_posterior_identification_cache(posterior, variables, query, ctx)?,
            ));
        }
        Ok(())
    }

    /// Install the temporal-mediation graph-posterior cache for the atom kind of `posterior`.
    fn install_temporal_mediation_posterior_cache(
        analysis: &mut Study,
        posterior: &antecedent_discovery::GraphPosterior,
        variables: &[antecedent_core::VariableId],
        query: &antecedent_core::MediationQuery,
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        if crate::analysis::execute::is_class_atom(posterior.atom_kind) {
            let horizon = query.horizons.first().copied().ok_or_else(|| CausalError::Compile {
                message: "temporal class graph-posterior mediation requires a horizon".into(),
            })?;
            let mut witness = TemporalEffectQuery::pulse(query.treatment, query.outcome, 1.0);
            witness.horizon_steps = horizon;
            analysis.temporal_class_posterior_identification_cache =
                Some(Arc::new(build_temporal_class_posterior_identification_cache(
                    posterior,
                    variables,
                    &witness,
                    analysis.max_completions,
                    ctx,
                )?));
        } else {
            analysis.dbn_posterior_identification_cache =
                Some(Arc::new(build_dbn_posterior_mediation_identification_cache(
                    posterior, variables, query, ctx,
                )?));
        }
        Ok(())
    }

    /// Install the temporal-response graph-posterior cache for the atom kind of `posterior`.
    fn install_temporal_response_posterior_cache(
        &self,
        analysis: &mut Study,
        posterior: &antecedent_discovery::GraphPosterior,
        variables: &[antecedent_core::VariableId],
        query: &antecedent_core::ResponseQuery,
        ctx: &ExecutionContext,
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
                    variables,
                    query,
                    estimator_id,
                    ctx,
                )?));
        }
        Ok(())
    }
}
