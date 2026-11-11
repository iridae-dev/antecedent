// SPDX-License-Identifier: MIT OR Apache-2.0

mod admission;
mod sealed;
mod tabular;

#[cfg(test)]
use admission::ValidationFamily;

use super::route_call::{RouteCall, TabularCall, route_mismatch, route_unsupported};
use super::{
    AnalysisRoute, Arc, CausalError, CausalQuery, CheckedInterferenceOperation,
    DEFAULT_ADMG_ESTIMATOR_ID, DEFAULT_ADMG_IDENTIFIER_ID, DEFAULT_CONDITIONAL_ESTIMATOR_ID,
    DEFAULT_CONDITIONAL_IDENTIFIER_ID, DEFAULT_DISTRIBUTION_ESTIMATOR_ID,
    DEFAULT_DISTRIBUTION_IDENTIFIER_ID, DEFAULT_ESTIMATOR_ID, DEFAULT_IDENTIFIER_ID,
    DEFAULT_PAG_ESTIMATOR_ID, DEFAULT_PAG_IDENTIFIER_ID, DEFAULT_PATH_ESTIMATOR_ID,
    DEFAULT_PATH_IDENTIFIER_ID, DEFAULT_RESPONSE_IDENTIFIER_ID, DataInput, DataModality,
    Diagnostic, DiagnosticKind, DiagnosticSeverity, EstimatorId, ExecutionContext, GraphClass,
    GraphPosterior, GraphRef, IdentificationResult, IdentifierId, InferenceMode, OverlapPolicy,
    PhysicalExecutionPlan, RefuteSuite, ResponseQuery, RouteInput, StaticMediationSource, Strategy,
    StudyResult, TableView, TabularData, TimeSeriesData, VariableId, admg_has_bidirected,
    admg_without_latents_to_dag, classify_route, graph_posterior_response_supported, is_class_atom,
    reject_dag_only_on_pag, response_functional_is_derivative, temporal_admg_projection,
};
use crate::analysis::prepared::ExecutionKind;
use crate::analysis::route_guards::{
    complete_mean_response, distinct_numeric_arms, mean_all_observed,
};
use crate::graph::Dag;
use crate::planner::RouteDefaults;
use antecedent_core::NestedCounterfactualQuery;

/// Refusal for a graph-posterior study whose data / query pair has no route.
pub(super) const GRAPH_POSTERIOR_QUERY_REFUSAL: &str = concat!(
    "graph-posterior analysis supports tabular average-effect, conditional-effect, static ",
    "response, transport, interference, mediation, path-specific, natural-direct or ",
    "attribution queries, and series temporal-effect, temporal-mediation, or temporal ",
    "response queries only",
);

/// The wire names of a typed identifier/estimator pair, as plan records carry them.
pub(super) fn wire_pair(
    (identifier, estimator): (IdentifierId, EstimatorId),
) -> (Arc<str>, Arc<str>) {
    (Arc::from(identifier.as_str()), Arc::from(estimator.as_str()))
}

impl super::Study {
    /// Execute a compiled physical plan.
    ///
    /// # Errors
    ///
    /// Identification / estimation / validation failures.
    ///
    /// # Panics
    ///
    /// Never in practice; see [`Study::compile_logical`]'s panics note — the same
    /// `AcceptedGraph::class`/accessor invariant applies here.
    pub fn execute(
        &self,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        // Validation executes through its own refit path instead of execute_on, which would
        // never look at the supplied plan: refuse an unlowered, foreign or differently bound
        // plan before taking that branch.
        if crate::analysis::functional_validation::requested(self) {
            self.require_plan_for_study(physical, true)?;
            return crate::analysis::functional_validation::run_validated(self, ctx);
        }
        self.execute_shared(&Arc::new(physical.clone()), None, ctx)
    }

    /// [`Self::execute`] for a plan the caller already holds behind an `Arc`.
    ///
    /// `execution` is the sealed operation of the prepared handle that runs the plan.
    /// `None` is a fresh run: no handle exists, so the route executes without a
    /// retained operation, inside the dispatch scope of the plan's lowered program, and
    /// its result carries no prepared-contract stamp or cached-identification claim.
    pub(crate) fn execute_shared(
        &self,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: Option<&super::super::prepared::PreparedExecution>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.execute_on(&self.data, physical, execution, None, ctx)
    }

    /// The lowered program of `physical`, checked against this study.
    ///
    /// # Errors
    ///
    /// `route_not_lowered`: the plan carries no lowered program, it was lowered for another
    /// route cell, it belongs to different live callback bindings, or (when `bound`) it was
    /// compiled for a study whose query, graph, data schema or declarations differ from
    /// this one's.
    fn require_plan_for_study<'p>(
        &self,
        physical: &'p PhysicalExecutionPlan,
        bound: bool,
    ) -> Result<&'p crate::planner::lowered::LoweredProgram, CausalError> {
        let program = physical.require_lowered(crate::analysis::lowering::expected_cell(self))?;
        if !program.accepts_live_query(&self.query) {
            return Err(crate::compile_reason!(
                "route_not_lowered",
                "the plan belongs to a different live callback query"
            ));
        }
        if bound && !crate::analysis::lowering::program_is_bound_to(program, self) {
            return Err(crate::compile_reason!(
                "route_not_lowered",
                "the plan was compiled for a different study: its query, graph, data schema or \
                 declarations differ from this study's"
            ));
        }
        Ok(program)
    }

    /// Execute against an explicit data slot (prepare/estimate must not clone `Study`).
    /// `provenance` names the source rows of a temporal slot when the caller holds them.
    pub(crate) fn execute_on(
        &self,
        data: &DataInput,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: Option<&super::super::prepared::PreparedExecution>,
        provenance: Option<&super::TemporalRowProvenance>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        // A requested functional validation is run by `functional_validation::run_validated`
        // (the study refitted at validation none); a route reaching here with the suite still
        // set would return an unvalidated result under a validated request.
        if crate::analysis::functional_validation::requested(self) {
            return Err(CausalError::Unsupported {
                message: "cheap and full validation of this query run through the functional \
                          refuter state: execute the study through run, a prepared estimate or \
                          refresh",
            });
        }
        // A fresh run has no sealed operation to carry the binding: the plan's program must
        // have been compiled for this study. A prepared handle runs its own retained plan.
        let program = self.require_plan_for_study(physical, execution.is_none())?;
        let ctx = &ctx.enter_lowered_dispatch(program.identity());
        let mut result = match data {
            DataInput::Tabular(tabular) if crate::analysis::missing_imputation::imputes(self) => {
                crate::analysis::missing_imputation::impute_fit_pool(
                    self,
                    tabular,
                    ctx,
                    |table, ctx| self.execute_on_table(table, physical, execution, ctx),
                )?
            }
            DataInput::Temporal(series)
                if crate::analysis::missing_imputation::imputes_series(self) =>
            {
                crate::analysis::missing_imputation::impute_fit_pool_series(
                    self,
                    series,
                    ctx,
                    |series, ctx| self.execute_on_series(series, physical, execution, ctx),
                )?
            }
            _ => self.execute_on_inner(data, physical, execution, provenance, ctx)?,
        };
        push_gaussian_likelihood_disclosure(&mut result, &self.settings.inference, data);
        push_lag_zero_average_effect_disclosure(self, &mut result);
        result.extras.custom_validator_names = self
            .declared
            .custom_validators
            .iter()
            .map(|validator| std::sync::Arc::from(validator.name()))
            .collect();
        result.stamp_lowered_program(
            physical.require_lowered(crate::analysis::lowering::expected_cell(self))?,
        );
        result.executed_input =
            Some(crate::analysis::target_validation::executed_input_identity(data, ctx)?);
        crate::analysis::target_validation::stamp_route_refutations(&mut result);
        Ok(result)
    }

    /// Graph-posterior dispatch over data modality, query, and posterior atom kind.
    fn execute_graph_posterior_route(
        &self,
        data: &DataInput,
        gp: &GraphPosterior,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        provenance: Option<&super::TemporalRowProvenance>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        match data {
            DataInput::Tabular(table) => {
                self.execute_graph_posterior_tabular(table, gp, physical, ctx)
            }
            DataInput::Temporal(series) | DataInput::Event(series) => {
                self.execute_graph_posterior_series(series, gp, physical, provenance, ctx)
            }
            DataInput::Panel(_) | DataInput::MultiEnv(_) => {
                Err(CausalError::Unsupported { message: GRAPH_POSTERIOR_QUERY_REFUSAL })
            }
        }
    }

    /// Series graph-posterior dispatch over query and posterior atom kind.
    fn execute_graph_posterior_series(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        provenance: Option<&super::TemporalRowProvenance>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let result =
            self.execute_graph_posterior_series_unrefuted(data, gp, physical, provenance, ctx)?;
        let CausalQuery::Response(q) = &self.query else { return Ok(result) };
        let Some(treatment) = super::curve_refutation::curve_treatment(q) else {
            return Ok(result);
        };
        let placebo = super::curve_refutation::series_placebo(treatment, ctx);
        let subset = super::curve_refutation::series_subset(ctx);
        self.attach_curve_refutation(
            result,
            data,
            q,
            (&placebo, &subset),
            &|study, perturbed, rctx| {
                study.execute_graph_posterior_series_unrefuted(
                    perturbed, gp, physical, provenance, rctx,
                )
            },
            ctx,
        )
    }

    fn execute_graph_posterior_series_unrefuted(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        provenance: Option<&super::TemporalRowProvenance>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if is_class_atom(gp.atom_kind())
            && super::temporal_class_point::temporal_class_point_family(&self.query)
        {
            return self.execute_temporal_class_graph_posterior_family(data, gp, physical, ctx);
        }
        // A one-modifier conditional effect or a cross-world contrast over `TemporalDag` atoms is
        // answered atom by atom, with the shared interval of the whole procedure.
        if super::dbn_atoms(gp) && super::dbn_route_query(&self.query) {
            return self.execute_dbn_route_graph_posterior(data, gp, physical, ctx);
        }
        match &self.query {
            CausalQuery::TemporalEffect(q) => match gp.atom_kind() {
                antecedent_discovery::GraphPosteriorAtomKind::Cpdag
                | antecedent_discovery::GraphPosteriorAtomKind::Pag => {
                    self.execute_temporal_class_graph_posterior(data, gp, q, physical, ctx)
                }
                antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg => {
                    self.execute_temporal_admg_graph_posterior(data, gp, q, physical, ctx)
                }
                _ => match self.settings.inference {
                    InferenceMode::Frequentist => {
                        self.execute_dbn_posterior_frequentist(data, gp, q, physical, ctx)
                    }
                    InferenceMode::Bayesian(_) => {
                        self.execute_dbn_posterior_bayesian(data, gp, q, physical, ctx)
                    }
                },
            },
            CausalQuery::Mediation(q) => match gp.atom_kind() {
                antecedent_discovery::GraphPosteriorAtomKind::Cpdag => self
                    .execute_temporal_class_graph_posterior_mediation(data, gp, q, physical, ctx),
                antecedent_discovery::GraphPosteriorAtomKind::Pag => {
                    self.execute_temporal_pag_graph_posterior_mediation(data, gp, q, physical, ctx)
                }
                antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg => {
                    self.execute_temporal_admg_graph_posterior_mediation(data, gp, q, physical, ctx)
                }
                _ => self.execute_dbn_posterior_mediation(data, gp, q, physical, ctx),
            },
            CausalQuery::Response(q) => match gp.atom_kind() {
                antecedent_discovery::GraphPosteriorAtomKind::Cpdag
                | antecedent_discovery::GraphPosteriorAtomKind::Pag => {
                    self.execute_temporal_class_graph_posterior_response(data, gp, q, physical, ctx)
                }
                _ => self.execute_dbn_posterior_response(data, gp, q, physical, ctx),
            },
            CausalQuery::AnomalyAttribution(_) | CausalQuery::ChangeAttribution(_) => {
                self.execute_series_attribution_graph_posterior(data, gp, physical, provenance, ctx)
            }
            _ => Err(CausalError::Unsupported { message: GRAPH_POSTERIOR_QUERY_REFUSAL }),
        }
    }

    /// A tabular route on a prepared handle, or on a fresh run whose operation is sealed to the
    /// plan's lowered program here.
    fn execute_tabular_prepared_or_fresh(
        &self,
        call: TabularCall<'_>,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: Option<&super::super::prepared::PreparedExecution>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        // A fresh run holds no prepared handle: its operation is sealed to the
        // plan's lowered program here, and stamps nothing as a prepared click.
        let fresh;
        let execution = if let Some(execution) = execution {
            execution
        } else {
            fresh = super::super::prepared::PreparedExecution::FreshRun(
                super::CheckedRouteOperation::lowered(self, physical)?,
            );
            &fresh
        };
        self.execute_tabular_route(call, data, physical, execution, ctx)
    }

    fn execute_on_inner(
        &self,
        data: &DataInput,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: Option<&super::super::prepared::PreparedExecution>,
        provenance: Option<&super::TemporalRowProvenance>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if let DataInput::Tabular(table) = data {
            return self.execute_on_table(table, physical, execution, ctx);
        }
        if let Some(gp) = &self.graph_posterior {
            return self.execute_graph_posterior_route(data, gp, physical, provenance, ctx);
        }
        if matches!(self.query, CausalQuery::Discover(_)) {
            return self.execute_discover(data, physical, ctx);
        }
        self.execute_route_call(self.route_call(data)?, physical, execution, provenance, ctx)
    }

    /// [`Self::execute_on_inner`] on tabular rows held by reference, such as the completed
    /// table of one multiple-imputation pass, without wrapping them in a new `DataInput`.
    fn execute_on_table(
        &self,
        table: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: Option<&super::super::prepared::PreparedExecution>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.refuse_unread_sampling_weights_for_route(table)?;
        if let Some(gp) = &self.graph_posterior {
            return self.execute_graph_posterior_tabular(table, gp, physical, ctx);
        }
        if matches!(self.query, CausalQuery::Discover(_)) {
            return self.execute_discover_tabular(table, physical, ctx);
        }
        self.execute_route_call(self.tabular_route_call(table)?, physical, execution, None, ctx)
    }

    /// [`Self::execute_on_inner`] on a series held by reference, such as the completed
    /// series of one multiple-imputation pass.
    fn execute_on_series(
        &self,
        series: &TimeSeriesData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: Option<&super::super::prepared::PreparedExecution>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if let Some(gp) = &self.graph_posterior {
            return self.execute_graph_posterior_series(series, gp, physical, None, ctx);
        }
        // A structure search is no imputation route, so a completed series never carries one:
        // `series_route_call` refuses it as an unserved combination.
        self.execute_route_call(self.series_route_call(series)?, physical, execution, None, ctx)
    }

    /// Run the executor a classified route names.
    fn execute_route_call(
        &self,
        call: RouteCall<'_>,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: Option<&super::super::prepared::PreparedExecution>,
        provenance: Option<&super::TemporalRowProvenance>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        match call {
            RouteCall::Tabular(call, table) => {
                self.execute_tabular_prepared_or_fresh(call, table, physical, execution, ctx)
            }
            RouteCall::CrossWorld(series) => {
                if self.graph.class().is_incomplete_temporal() {
                    return self.execute_temporal_class_cross_world(series, physical, ctx);
                }
                let graph = physical.temporal_graph().ok_or_else(|| CausalError::Compile {
                    message: "temporal cross-world plan is missing its TemporalDag".into(),
                })?;
                self.execute_temporal_cross_world(series, graph, physical, ctx)
            }
            RouteCall::SeriesMediation { series, query } => {
                self.execute_series_mediation_route(series, query, physical, ctx)
            }
            RouteCall::SeriesEffect { series, query } => {
                if self.graph.class() == GraphClass::TemporalAdmg {
                    return self.execute_temporal_admg_contrast(series, query, physical, ctx);
                }
                if self.graph.class().is_incomplete_temporal() {
                    return self.execute_temporal_class(series, query, physical, ctx);
                }
                let graph = physical.temporal_graph().ok_or(CausalError::Compile {
                    message: "Ready temporal plan missing resolved graph".into(),
                })?;
                self.execute_temporal(series, graph, query, physical, ctx, None)
            }
            RouteCall::SeriesResponse { series, query } => {
                self.execute_series_response_route(series, query, physical, ctx)
            }
            RouteCall::SeriesAssociation(series) => {
                self.execute_association_series(series, physical, ctx)
            }
            RouteCall::PanelEffect { panel, query } => {
                self.execute_panel_effect_route(panel, query, physical, ctx)
            }
            RouteCall::PanelResponse { panel, query } => {
                self.execute_panel_response_route(panel, query, physical, ctx)
            }
            RouteCall::MultiEnvEffect { multi, query } => {
                self.execute_multi_env_effect_route(multi, query, physical, ctx)
            }
            RouteCall::ClassAttribution(series) => {
                self.execute_temporal_class_attribution(series, physical, provenance, ctx)
            }
            RouteCall::SeriesAttribution(series) => {
                self.execute_series_attribution(series, physical, provenance, ctx)
            }
        }
    }

    /// Resolve the temporal response's graph family before its numerical executor.
    fn execute_series_response_route(
        &self,
        data: &TimeSeriesData,
        q: &ResponseQuery,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let result = self.execute_series_response_route_unrefuted(data, q, physical, ctx)?;
        let Some(treatment) = super::curve_refutation::curve_treatment(q) else {
            return Ok(result);
        };
        let placebo = super::curve_refutation::series_placebo(treatment, ctx);
        let subset = super::curve_refutation::series_subset(ctx);
        self.attach_curve_refutation(
            result,
            data,
            q,
            (&placebo, &subset),
            &|study, perturbed, rctx| {
                study.execute_series_response_route_unrefuted(perturbed, q, physical, rctx)
            },
            ctx,
        )
    }

    fn execute_series_response_route_unrefuted(
        &self,
        data: &TimeSeriesData,
        q: &ResponseQuery,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if self.graph.class() == GraphClass::TemporalAdmg {
            return self.execute_temporal_admg_derivative(data, q, physical, ctx);
        }
        if self.graph.class().is_incomplete_temporal() {
            return self.execute_temporal_class_response(data, q, physical, ctx);
        }
        let graph = physical.temporal_graph().ok_or(CausalError::Compile {
            message: "Ready temporal-response plan missing resolved graph".into(),
        })?;
        self.execute_temporal_response(data, graph, q, physical, ctx)
    }

    fn execute_series_mediation_route(
        &self,
        data: &TimeSeriesData,
        q: &antecedent_core::MediationQuery,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if self.graph.class() == GraphClass::TemporalCpdag {
            return self.execute_temporal_cpdag_mediation(data, q, physical, ctx);
        }
        if self.graph.class() == GraphClass::TemporalAdmg {
            return self.execute_temporal_admg_mediation(data, q, physical, ctx);
        }
        if self.graph.class() == GraphClass::TemporalPag {
            return self.execute_temporal_pag_mediation(data, q, physical, ctx);
        }
        let graph = physical.temporal_graph().ok_or(CausalError::Compile {
            message: "Ready temporal mediation plan missing resolved graph".into(),
        })?;
        self.execute_temporal_mediation(data, graph, q, physical, ctx)
    }

    /// Series anomaly or change attribution on a supplied `TemporalDag`: the sealed
    /// attribution operation over the lag-aligned unfolding, built from the static
    /// identification the plan's lowered program retained (or the prepared cache).
    /// The study identifies here only when neither carries the product.
    fn execute_series_attribution(
        &self,
        series: &TimeSeriesData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        provenance: Option<&super::TemporalRowProvenance>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let cache = if let Some((cache, _)) = self.reuse_static_identification(physical) {
            cache
        } else {
            super::report_identify_compute(ctx);
            std::sync::Arc::new(self.prepare_static_identification(physical)?.ok_or_else(|| {
                CausalError::Compile {
                    message: "series attribution identified no structural product".into(),
                }
            })?)
        };
        super::CheckedAttributionOperation::checked(self, physical, &cache, ctx)?
            .execute_series(series, provenance, None, None, ctx)
    }

    /// Panel temporal-effect execution: class graphs or the resolved `TemporalDag`.
    fn execute_panel_effect_route(
        &self,
        panel: &antecedent_data::PanelData,
        q: &antecedent_core::TemporalEffectQuery,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if self.graph.class().is_incomplete_temporal() {
            return self.execute_panel_class(panel, q, physical, ctx);
        }
        let graph = physical.temporal_graph().ok_or(CausalError::Compile {
            message: "Ready panel plan missing resolved graph".into(),
        })?;
        if matches!(
            self.declared.estimator,
            Some(
                EstimatorId::TemporalGlmLinkContrast | EstimatorId::BayesianTemporalGlmLinkContrast
            )
        ) {
            return self.execute_panel_link(panel, graph, q, physical, ctx);
        }
        self.execute_panel(panel, graph, q, physical, ctx)
    }

    /// Panel temporal-response execution: class graphs or the resolved `TemporalDag`.
    fn execute_panel_response_route(
        &self,
        panel: &antecedent_data::PanelData,
        q: &ResponseQuery,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let result = self.execute_panel_response_route_unrefuted(panel, q, physical, ctx)?;
        let Some(treatment) = super::curve_refutation::curve_treatment(q) else {
            return Ok(result);
        };
        let placebo = super::curve_refutation::panel_placebo(treatment, ctx);
        let subset = super::curve_refutation::panel_subset(ctx);
        self.attach_curve_refutation(
            result,
            panel,
            q,
            (&placebo, &subset),
            &|study, perturbed, rctx| {
                study.execute_panel_response_route_unrefuted(perturbed, q, physical, rctx)
            },
            ctx,
        )
    }

    fn execute_panel_response_route_unrefuted(
        &self,
        panel: &antecedent_data::PanelData,
        q: &ResponseQuery,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if self.graph.class().is_incomplete_temporal() {
            return self.execute_panel_class_response(panel, q, physical, ctx);
        }
        let graph = physical.temporal_graph().ok_or(CausalError::Compile {
            message: "Ready panel-response plan missing resolved graph".into(),
        })?;
        self.execute_panel_response(panel, graph, q, physical, ctx)
    }

    /// Multi-environment temporal-effect execution, which admits only a resolved `TemporalDag`.
    fn execute_multi_env_effect_route(
        &self,
        multi: &antecedent_data::MultiEnvironmentData,
        q: &antecedent_core::TemporalEffectQuery,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if self.graph.class().is_incomplete_temporal() {
            return Err(CausalError::Unsupported {
                message: "multi-environment data supports only a supplied TemporalDag",
            });
        }
        let graph = physical.temporal_graph().ok_or(CausalError::Compile {
            message: "Ready multi-env plan missing resolved graph".into(),
        })?;
        self.execute_multi_env_temporal(multi, graph, q, physical, ctx)
    }

    /// Compile and run.
    ///
    /// # Errors
    ///
    /// Compile / execute failures.
    pub fn run(&self, ctx: &ExecutionContext) -> Result<StudyResult, CausalError> {
        self.run_observed(ctx).result
    }

    /// Execute with engine-owned work observations, including refusal and cancellation.
    #[must_use]
    pub fn run_observed(
        &self,
        ctx: &ExecutionContext,
    ) -> antecedent_core::execution_attempt::ExecutionAttempt<StudyResult, CausalError> {
        let mut observed = antecedent_core::execution_attempt::observe_with_context(ctx, || {
            antecedent_estimate::prepare::with_preparation_budget(
                crate::planner::PhysicalExecutionPlan::preparation_budget(ctx),
                || self.run_unnamed(ctx),
            )
            .map_err(|error| crate::analysis::preflight::name_run_refusal(self, error, ctx))
        });
        if let Ok(result) = &mut observed.result {
            result.view.performance.native_execution.clone_from(&observed.facts);
            result.stamp_would_refuse(&self.would_refuse, ctx);
        }
        observed
    }

    fn run_unnamed(&self, ctx: &ExecutionContext) -> Result<StudyResult, CausalError> {
        crate::analysis::lowering::refuse_reentry(ctx)?;
        if crate::analysis::functional_validation::requested(self) {
            return crate::analysis::functional_validation::run_validated(self, ctx);
        }
        if let Some(result) = self.run_checked_bayesian_static_mediation(ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_checked_admg_response(ctx)? {
            return Ok(*result);
        }
        self.run_other_routes(ctx)
    }

    /// Prepare the study, require that `retained` sees the sealed checked
    /// operation, then estimate on tabular data. `message` is the compile
    /// error reported when preparation did not retain the operation.
    fn run_sealed_tabular(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
        retained: impl FnOnce(&crate::PreparedStudy) -> bool,
        message: &str,
    ) -> Result<StudyResult, CausalError> {
        let prepared = self.prepare(ctx)?;
        if !retained(&prepared) {
            return Err(CausalError::Compile { message: message.into() });
        }
        prepared.estimate(data, ctx)
    }

    /// Series twin of [`Self::run_sealed_tabular`].
    fn run_sealed_series(
        &self,
        data: &TimeSeriesData,
        ctx: &ExecutionContext,
        retained: impl FnOnce(&crate::PreparedStudy) -> bool,
        message: &str,
    ) -> Result<StudyResult, CausalError> {
        let prepared = self.prepare(ctx)?;
        if !retained(&prepared) {
            return Err(CausalError::Compile { message: message.into() });
        }
        prepared.estimate_series(data, ctx)
    }

    /// Estimator the caller selected, directly or through the estimator spec.
    fn selected_estimator_id(&self) -> Option<EstimatorId> {
        self.declared
            .estimator
            .or_else(|| self.declared.estimator_spec.as_deref().map(crate::EstimatorSpec::id))
    }

    /// [`Self::run_sealed_tabular`] boxed as a taken route, for the one-shot admission helpers.
    fn admit_sealed_tabular(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
        retained: impl FnOnce(&crate::PreparedStudy) -> bool,
        message: &str,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        let prepared = self.prepare(ctx)?;
        if !retained(&prepared) {
            // A design over the resident budget streams through a closed-form reduction and
            // retains no operation; the ordinary dispatcher runs it.
            if prepared.plan().streams_design() {
                return Ok(None);
            }
            return Err(CausalError::Compile { message: message.into() });
        }
        prepared.estimate(data, ctx).map(Box::new).map(Some)
    }

    /// Series twin of [`Self::admit_sealed_tabular`].
    fn admit_sealed_series(
        &self,
        data: &TimeSeriesData,
        ctx: &ExecutionContext,
        retained: impl FnOnce(&crate::PreparedStudy) -> bool,
        message: &str,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        self.run_sealed_series(data, ctx, retained, message).map(Box::new).map(Some)
    }

    #[inline(never)]
    fn run_other_routes(&self, ctx: &ExecutionContext) -> Result<StudyResult, CausalError> {
        // Shape is parsed once. Estimator, validation, and structure-source checks
        // stay here: they choose which sealed operation to commit to, and a match
        // that does not retain that operation is still a compile error.
        let route = self.route_input_classified();
        // Bind one-shot Bernoulli interference to the actual checked operation,
        // as prepared execution does, so its result carries that execution's contract.
        if self.plain_fixed_route()
            && self.declared.split.is_none()
            && self.settings.inference.is_frequentist()
            && let Some(RouteInput::InterferenceDag { data, query, .. }) = route
            && matches!(query.assignment, antecedent_core::AssignmentDesign::Bernoulli { .. })
            && query.exposure == antecedent_core::ExposureMapping::NeighborCount
            && let Some(result) = self.admit_sealed_tabular(
                data,
                ctx,
                |prepared| prepared.retains(ExecutionKind::Interference),
                "one-shot interference route did not retain its checked operation",
            )?
        {
            return Ok(*result);
        }
        if let Some(result) = self.run_sealed_bayesian_basis_and_robust_effect(route, ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_sealed_bayesian_dag_effect(route, ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_sealed_mediation_and_temporal_dag(route, ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_sealed_frequentist_graph_posterior_effect(route, ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_sealed_static_dag_conditional_and_response(route, ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_sealed_derivative_counterfactual_and_program(route, ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_sealed_aipw(route, ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_sealed_glm_rd_propensity(route, ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_sealed_class_and_posterior_response(route, ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_sealed_cell_aipw_and_bayesian_class_effect(route, ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_sealed_graph_posterior_conditional_effect(ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_sealed_temporal_class_response_and_posterior(route, ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_sealed_specialist_transport_and_linear(route, ctx)? {
            return Ok(*result);
        }
        // A fresh run executes through a prepared handle, so its result carries the executed
        // contract that attests what ran (program, data snapshot, validation mode).
        self.prepare(ctx)?.estimate_retained(ctx)
    }

    /// Identify only (no estimation). Supports static DAG and ADMG average-effect
    /// / related queries.
    ///
    /// The ADMG path matters for correctness, not just coverage: a DAG cannot
    /// express "this variable is unobservable", so a latent common cause
    /// flattened into one is identified by adjusting on a variable no study can
    /// measure. Routing an ADMG through general ID lets that case report
    /// `NotIdentified` instead. Mirrors `execute()`: an ADMG with no bidirected
    /// edges is just a DAG, and is coerced rather than forced down the general
    /// ID path.
    ///
    /// # Errors
    ///
    /// Missing graph structure, unsupported graph class, or identification failure.
    pub fn identify_only(&self) -> Result<IdentificationResult, CausalError> {
        self.identify_only_within(&ExecutionContext::production_default(0))
    }

    /// [`Self::identify_only`] under the caller's execution context: the identification is the
    /// compiled one ([`Self::identify_structure_within`]), so its exact searches lease their
    /// storage from the context's admission pool and stop at its cancellation token.
    ///
    /// # Errors
    ///
    /// As [`Self::identify_only`], plus resource or cancellation refusal.
    pub fn identify_only_within(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<IdentificationResult, CausalError> {
        if self.graph_posterior.is_some() {
            // `self.graph` is only the placeholder shape here (see `stub_accepted_graph_for`);
            // identification runs per-graph, against the real posterior atoms, inside `execute()`.
            return Err(crate::unsupported_reason!(
                "route_not_supported",
                "identify_only is not a graph-posterior cell; identification \
                 runs per-graph inside execute."
            ));
        }
        let identification = if self.declared.tiered.is_some() {
            self.identify_structure_within(ctx)?
        } else if let Some(admg) = self.graph.as_admg() {
            if admg_has_bidirected(admg) {
                if !matches!(&self.query, CausalQuery::AverageEffect(_)) {
                    return Err(crate::unsupported_reason!(
                        "route_not_supported",
                        "identify_only on an ADMG supports AverageEffect only."
                    ));
                }
                // Only general ID handles bidirected structure; the default
                // identifier is a backdoor strategy that would ignore it.
                let identifier = self.declared.identifier.unwrap_or(IdentifierId::GeneralId);
                crate::identify_api::identify_study_within(
                    &self.graph,
                    &self.query,
                    Some(identifier),
                    self.structure_source,
                    ctx,
                )?
            } else {
                // No bidirected edges: this ADMG *is* a DAG. Coercing keeps the
                // caller's identifier choice meaningful instead of forcing general
                // ID on a graph with no latent structure to reason about.
                let coerced = crate::AcceptedGraph::from(admg_without_latents_to_dag(admg)?);
                crate::identify_api::identify_study_within(
                    &coerced,
                    &self.query,
                    self.declared.identifier,
                    self.structure_source,
                    ctx,
                )?
            }
        } else {
            if self.graph.as_dag().is_none() {
                return Err(crate::unsupported_reason!(
                    "route_not_supported",
                    "identify_only supports static DAG and ADMG graphs only."
                ));
            }
            self.identify_structure_within(ctx)?
        };
        match identification {
            crate::Identification::Point { result, .. } => Ok(result),
            _ => Err(crate::unsupported_reason!(
                "route_not_supported",
                "identify_only returns a single-graph identification; use identify_structure \
                 for a class envelope."
            )),
        }
    }

    /// The identification of this study's query against its structure, for every graph
    /// class: the static and temporal DAG, the ADMG, and the CPDAG and PAG classes (whose
    /// answer is an envelope over completions).
    ///
    /// Identification is a function of structure and query, so the study's rows are never
    /// read: a study over a table with no observations (see
    /// [`StudyBuilder::identify_structure`](crate::StudyBuilder::identify_structure)) answers
    /// exactly as one over a full table. The caller's identifier is honoured; without one the
    /// class default applies.
    ///
    /// # Errors
    ///
    /// A graph-posterior study (identification runs per graph inside `execute`), an
    /// unsupported graph-class/query pair, or identification failure.
    pub fn identify_structure(&self) -> Result<crate::Identification, CausalError> {
        self.identify_structure_within(&ExecutionContext::production_default(0))
    }

    /// Identify with the caller's temporal materialization limits, pool and cancellation.
    /// # Errors
    /// The same identification refusals, including temporal resource or cancellation refusal.
    pub fn identify_structure_within(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<crate::Identification, CausalError> {
        if self.graph_posterior.is_some() {
            return Err(crate::unsupported_reason!(
                "route_not_supported",
                "identify_structure is not a graph-posterior cell; identification \
                 runs per-graph inside execute."
            ));
        }
        if let Some(background) = self.declared.tiered.as_ref() {
            let schema = match &self.data {
                DataInput::Tabular(data) => data.schema(),
                DataInput::Temporal(data) | DataInput::Event(data) => data.schema(),
                _ => {
                    return Err(CausalError::Unsupported {
                        message: "TieredBackground requires tabular or series data",
                    });
                }
            };
            return crate::identify_api::identify_tiered_study_within(
                &self.graph,
                background,
                schema,
                &self.query,
                ctx,
            );
        }
        crate::identify_api::identify_study_within(
            &self.graph,
            &self.query,
            self.declared.identifier,
            self.structure_source,
            ctx,
        )
    }

    /// Inspectable compile result (logical + physical).
    ///
    /// Prefer this over `compile` when documenting plan inspection in user code.
    ///
    /// # Errors
    ///
    /// Same as [`Self::compile`].
    pub fn plan(&self, ctx: &ExecutionContext) -> Result<PhysicalExecutionPlan, CausalError> {
        self.compile(ctx)
    }
}

/// Posterior backend of the Bayesian-bootstrap robust ATE.
pub(crate) const ROBUST_ATE_BOOTSTRAP_BACKEND: &str =
    "bayesian.robust_ate.modular_bootstrap_pushforward";
/// Posterior backend of Bayesian anomaly attribution.
pub(crate) const ANOMALY_DIRICHLET_BACKEND: &str = "gcm.attribution.shared_dirichlet_row_weights";
/// Posterior backend of Bayesian change attribution.
pub(crate) const CHANGE_DIRICHLET_BACKEND: &str =
    "gcm.attribution.shared_population_dirichlet_row_weights";
/// Posterior backend of the Bayesian GCM counterfactual.
pub(crate) const COUNTERFACTUAL_DIRICHLET_BACKEND: &str = "gcm.fit.bayesian";
/// Posterior backend of the Bayesian fitted-model attribution (intrinsic influence, change and
/// root causes): the posterior over the mechanisms under Dirichlet row weights.
pub(crate) const ATTRIBUTE_DIRICHLET_BACKEND: &str =
    "gcm.attribution.attribute_dirichlet_row_weights";

/// Posteriors drawn by Dirichlet row reweighting (a Bayesian bootstrap).
/// They fit no outcome likelihood, so no Gaussian fit is disclosed for them.
const BOOTSTRAP_LAW_BACKENDS: [&str; 5] = [
    ROBUST_ATE_BOOTSTRAP_BACKEND,
    ANOMALY_DIRICHLET_BACKEND,
    CHANGE_DIRICHLET_BACKEND,
    COUNTERFACTUAL_DIRICHLET_BACKEND,
    ATTRIBUTE_DIRICHLET_BACKEND,
];

/// Append [`gaussian_likelihood_disclosure`] to `result` once.
///
/// Fresh runs ([`Study::execute_on`]) and prepared clicks (which stamp their
/// contract after bypassing it) both call this, so it is idempotent.
pub(crate) fn push_gaussian_likelihood_disclosure(
    result: &mut StudyResult,
    inference: &InferenceMode,
    data: &DataInput,
) {
    const CODE: &str = "estimate.bayesian.gaussian_likelihood_discrete_outcome";
    if result.diagnostics.iter().any(|diagnostic| diagnostic.code.as_ref() == CODE) {
        return;
    }
    if result.posterior.as_ref().is_some_and(|posterior| {
        BOOTSTRAP_LAW_BACKENDS.contains(&posterior.diagnostics.backend_id.as_ref())
    }) {
        return;
    }
    if let Some(disclosure) = gaussian_likelihood_disclosure(inference, data, result.outcome) {
        result.view.diagnostics.push(disclosure);
    }
}

/// Disclose that a requested `AverageEffect` on a temporal graph was compiled and executed as
/// its lag-0 pulse contrast, and that the result is unsupported.
pub(crate) fn push_lag_zero_average_effect_disclosure(
    study: &crate::Study,
    result: &mut StudyResult,
) {
    const CODE: &str = "estimate.temporal_class.lag_zero_average_effect";
    let Some(average) = study.lag_zero_average_effect.as_ref() else {
        return;
    };
    if result.diagnostics.iter().any(|diagnostic| diagnostic.code.as_ref() == CODE) {
        return;
    }
    let treatment = average.treatment.raw().to_string();
    let outcome = average.outcome.raw().to_string();
    result.view.diagnostics.push(
        Diagnostic::new(
            CODE,
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "the AverageEffect on series data under a temporal graph denotes the lag-0 \
             contrast E[Y_t | do(X_t = active)] - E[Y_t | do(X_t = control)]; it was compiled \
             as the pulse at offset 0 read at horizon 1 and identified on every completion of \
             the graph. The AverageEffect cell on a temporal graph is unsupported: the result \
             reports support status refused and no coverage record names it"
                .to_owned(),
        )
        .with_fields([
            ("treatment", treatment.as_str()),
            ("outcome", outcome.as_str()),
            ("compiled_as", "PulseEffect(offset=0, horizon_steps=1)"),
        ]),
    );
}

/// Disclose a Gaussian likelihood fitted to a binary or count outcome.
///
/// A Bayesian execution under the Gaussian identity-link model (the default
/// likelihood, and the only one the conjugate backend fits) is reported with
/// `estimate.bayesian.gaussian_likelihood_discrete_outcome` when the outcome is
/// declared binary or count in the schema, or when every observed value is 0/1
/// (`binary`) or a nonnegative integer (`count`). The answer is unchanged; the
/// diagnostic names the likelihood that models the outcome.
fn gaussian_likelihood_disclosure(
    inference: &InferenceMode,
    data: &DataInput,
    outcome: VariableId,
) -> Option<Diagnostic> {
    let InferenceMode::Bayesian(cfg) = inference else {
        return None;
    };
    let gaussian = cfg.likelihood == antecedent_prob::BayesLikelihood::GaussianIdentity
        || cfg.backend == antecedent_estimate::BayesianBackendKind::ConjugateGaussian;
    if !gaussian {
        return None;
    }
    let kind = match data {
        DataInput::Tabular(table) => discrete_outcome_kind([table as &dyn TableView], outcome),
        DataInput::Temporal(series) | DataInput::Event(series) => {
            discrete_outcome_kind([series as &dyn TableView], outcome)
        }
        DataInput::Panel(panel) => {
            let views: Vec<antecedent_data::PanelUnitView<'_>> =
                panel.units().iter().map(antecedent_data::PanelUnitView::new).collect();
            discrete_outcome_kind(views.iter().map(|view| view as &dyn TableView), outcome)
        }
        DataInput::MultiEnv(_) => None,
    }?;
    let (advice, suggested) = if kind == "binary" {
        ("A Bernoulli (logit or probit) likelihood", "bernoulli_logit")
    } else {
        ("A Poisson log-link likelihood", "poisson_log")
    };
    Some(
        Diagnostic::new(
            "estimate.bayesian.gaussian_likelihood_discrete_outcome",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            format!(
                "the outcome is {kind}-valued but the Bayesian model was fitted with a Gaussian \
                 identity-link likelihood, so the posterior describes a linear-Gaussian outcome \
                 model. {advice} models this outcome and is available for a tabular \
                 AverageEffect on a Dag"
            ),
        )
        .with_fields([
            ("outcome_kind", kind),
            ("fitted_likelihood", "gaussian_identity"),
            ("suggested_likelihood", suggested),
        ]),
    )
}

/// `binary` / `count` when the outcome is declared or observed as such.
#[allow(
    clippy::float_cmp,
    reason = "exact comparison is the point: a coded 0/1 outcome, not values near 0 or 1"
)]
fn discrete_outcome_kind<'a>(
    tables: impl IntoIterator<Item = &'a dyn TableView>,
    outcome: VariableId,
) -> Option<&'static str> {
    let mut binary = true;
    let mut count = true;
    let mut seen = false;
    for table in tables {
        match table.schema().get(outcome).ok().map(|variable| &variable.value_type) {
            Some(antecedent_core::ValueType::Binary) => return Some("binary"),
            Some(antecedent_core::ValueType::Count) => return Some("count"),
            // An unspecified type claims nothing, so the observed values decide, as for a
            // declared-continuous variable.
            Some(
                antecedent_core::ValueType::Continuous | antecedent_core::ValueType::Unspecified,
            ) => {}
            _ => return None,
        }
        let values = table.float64_values(outcome).ok()?;
        for value in values.into_iter().filter(|value| value.is_finite()) {
            seen = true;
            binary &= value == 0.0 || value == 1.0;
            count &= value >= 0.0 && value.fract() == 0.0;
            if !binary && !count {
                return None;
            }
        }
    }
    if !seen {
        None
    } else if binary {
        Some("binary")
    } else if count {
        Some("count")
    } else {
        None
    }
}

#[cfg(test)]
mod validation_family_tests {
    use super::{RefuteSuite, ValidationFamily};

    #[test]
    fn a_family_names_its_full_and_cheap_suites_and_none_when_validation_is_off() {
        for (family, full, cheap) in [
            (ValidationFamily::PathSpecific, "path.full", "path.cheap"),
            (ValidationFamily::Distribution, "distribution.full", "distribution.cheap"),
            (
                ValidationFamily::TemporalMediation,
                "temporal.mediation.full",
                "temporal.mediation.cheap",
            ),
            (
                ValidationFamily::MediationDecomposition,
                "mediation.decomposition.full",
                "mediation.decomposition.cheap",
            ),
        ] {
            assert_eq!(family.suite_id(RefuteSuite::Full).as_deref(), Some(full));
            assert_eq!(family.suite_id(RefuteSuite::Cheap).as_deref(), Some(cheap));
            assert_eq!(family.suite_id(RefuteSuite::PlaceboAndRcc).as_deref(), Some(cheap));
            assert!(family.suite_id(RefuteSuite::None).is_none());
        }
    }
}
