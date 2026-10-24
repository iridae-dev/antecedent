// SPDX-License-Identifier: MIT OR Apache-2.0
// Compile of a supplied graph posterior: the logical plan and physical compile per data/query
// kind and posterior atom kind. Execution lives in the per-family `*_posterior` modules.

use super::{
    Arc, AverageEffectQuery, CausalError, CausalQuery, Dag, DataClassification, DataInput,
    EstimatorId, ExecutionContext, GraphClass, IdentifierId, InferenceMode, LogicalAnalysisPlan,
    Pag, PhysicalExecutionPlan, StaticAteCompileInput, StaticCpdagResponseCompileInput,
    StaticPagResponseCompileInput, StaticResponseCompileInput, TableView, TabularData, TemporalDag,
    TemporalEffectQuery, TimeSeriesData, compile_logical_static_ate,
    compile_logical_static_cpdag_response, compile_logical_static_pag_response,
    compile_logical_static_response, compile_logical_temporal_effect_classified,
    compile_logical_temporal_response, is_class_atom,
};

impl super::Study {
    /// Compile path for a supplied graph posterior ([`crate::StudyBuilder::graph_posterior`]).
    ///
    /// `self.graph` is not consulted here beyond its class (it only carries the
    /// placeholder shape `stub_accepted_graph_for` built at `build()` time) — the logical
    /// plan needs a structure argument for row-count / classification bookkeeping only.
    /// Real identification happens per-graph, against the posterior atoms, in
    /// [`Self::execute_graph_posterior_bayesian`],
    /// [`Self::execute_graph_posterior_frequentist`],
    /// [`Self::execute_admg_graph_posterior`],
    /// [`Self::execute_class_graph_posterior`],
    /// [`Self::execute_temporal_class_graph_posterior`], or
    /// [`Self::execute_dbn_posterior_bayesian`].
    ///
    /// # Errors
    ///
    /// An unsupported data/query combination (graph-posterior analysis
    /// supports tabular average-effect, temporal-effect, temporal-mediation, or
    /// temporal response queries).
    pub(super) fn compile_graph_posterior(
        &self,
        goal: &crate::analysis::lowering::CompileGoal,
        ctx: &ExecutionContext,
    ) -> Result<PhysicalExecutionPlan, CausalError> {
        let mut plan = self.compile_graph_posterior_record(ctx)?;
        crate::analysis::lowering::attach_graph_posterior(self, &mut plan, goal, ctx)?;
        Ok(plan)
    }

    /// The physical plan of the study's graph-posterior cell, before lowering.
    fn compile_graph_posterior_record(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<PhysicalExecutionPlan, CausalError> {
        match (&self.data, &self.query) {
            (DataInput::Tabular(data), CausalQuery::Response(q)) => {
                self.compile_graph_posterior_response(data, q, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::AverageEffect(q)) => {
                self.compile_graph_posterior_average_effect(data, q, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::ConditionalEffect(q)) => {
                self.compile_graph_posterior_conditional_effect(data, q, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::Transport(q)) => {
                self.compile_graph_posterior_transport(data, q, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::Interference(q)) => {
                self.compile_graph_posterior_interference(data, q, ctx)
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
            ) => self.compile_graph_posterior_temporal_effect(data, q, ctx),
            (DataInput::Temporal(data) | DataInput::Event(data), CausalQuery::Response(q)) => {
                self.compile_graph_posterior_temporal_response(data, q, ctx)
            }
            (DataInput::Temporal(data) | DataInput::Event(data), CausalQuery::Mediation(q)) => {
                self.compile_graph_posterior_temporal_mediation(data, q, ctx)
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::NestedCounterfactual(_)
                | CausalQuery::PathSpecific(_)
                | CausalQuery::Counterfactual(_)
                | CausalQuery::Distribution(_),
            ) if self
                .graph_posterior
                .as_ref()
                .map(|posterior| posterior.atom_kind)
                .is_some_and(is_class_atom) =>
            {
                self.compile_graph_posterior_cross_world(data, ctx)
            }
            _ => Err(CausalError::Unsupported {
                message: super::dispatch::GRAPH_POSTERIOR_QUERY_REFUSAL,
            }),
        }
    }

    /// Tabular response compile over placeholder shapes of the posterior's atom kind.
    fn compile_graph_posterior_response(
        &self,
        data: &TabularData,
        q: &antecedent_core::ResponseQuery,
        ctx: &ExecutionContext,
    ) -> Result<PhysicalExecutionPlan, CausalError> {
        let atom_kind = self.graph_posterior.as_ref().map(|gp| gp.atom_kind);
        let mut logical = if matches!(
            atom_kind,
            Some(
                antecedent_discovery::GraphPosteriorAtomKind::Admg
                    | antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg,
            )
        ) {
            super::compile_logical_admg_response(data, q, self.validation_suite_id(), self)?
        } else {
            let n_vars = u32::try_from(data.schema().len()).map_err(|_| CausalError::Compile {
                message: "too many variables for graph-posterior compile".into(),
            })?;
            let stub = Dag::with_variables(n_vars);
            if matches!(atom_kind, Some(antecedent_discovery::GraphPosteriorAtomKind::Cpdag)) {
                let cpdag = antecedent_graph::Cpdag::from_dag(&stub);
                let (identifier, estimator) = self.resolve_class_response_pair(q);
                compile_logical_static_cpdag_response(StaticCpdagResponseCompileInput {
                    data,
                    graph: &cpdag,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })?
            } else if matches!(atom_kind, Some(antecedent_discovery::GraphPosteriorAtomKind::Pag)) {
                let pag = Pag::with_variables(n_vars);
                let (identifier, estimator) = self.resolve_class_response_pair(q);
                compile_logical_static_pag_response(StaticPagResponseCompileInput {
                    data,
                    graph: &pag,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })?
            } else {
                let (identifier, estimator) = self.resolve_response_pair(q);
                compile_logical_static_response(StaticResponseCompileInput {
                    data,
                    graph: &stub,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })?
            }
        };
        logical.record.discovery_algorithm =
            Some(self.graph_posterior_discovery_algorithm("graph_posterior"));
        logical.compile_physical(ctx)
    }

    /// Tabular average-effect compile: the identifier / estimator follow the atom kind and inference mode.
    fn compile_graph_posterior_average_effect(
        &self,
        data: &TabularData,
        q: &AverageEffectQuery,
        ctx: &ExecutionContext,
    ) -> Result<PhysicalExecutionPlan, CausalError> {
        let n_vars = u32::try_from(data.schema().len()).map_err(|_| CausalError::Compile {
            message: "too many variables for graph-posterior compile".into(),
        })?;
        let stub = Dag::with_variables(n_vars);
        let atom_kind = self.graph_posterior.as_ref().map(|gp| gp.atom_kind);
        let (identifier, estimator) = match atom_kind {
            Some(
                antecedent_discovery::GraphPosteriorAtomKind::Admg
                | antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg,
            ) => (
                Arc::from(IdentifierId::GeneralId.as_str()),
                Arc::from(EstimatorId::FunctionalEffect.as_str()),
            ),
            Some(
                antecedent_discovery::GraphPosteriorAtomKind::Cpdag
                | antecedent_discovery::GraphPosteriorAtomKind::Pag,
            ) => (
                Arc::from(IdentifierId::GeneralizedAdjustment.as_str()),
                match &self.inference {
                    InferenceMode::Frequentist => {
                        Arc::from(EstimatorId::LinearAdjustmentAte.as_str())
                    }
                    InferenceMode::Bayesian(_) => Arc::from(EstimatorId::BayesianGcomp.as_str()),
                },
            ),
            _ => (
                Arc::from(IdentifierId::BackdoorAdjustment.as_str()),
                match &self.inference {
                    InferenceMode::Frequentist => {
                        Arc::from(EstimatorId::LinearAdjustmentAte.as_str())
                    }
                    InferenceMode::Bayesian(_) => Arc::from(EstimatorId::BayesianGcomp.as_str()),
                },
            ),
        };
        let mut logical = compile_logical_static_ate(StaticAteCompileInput {
            data,
            graph: &stub,
            query: q,
            validation_suite: self.validation_suite_id(),
            identifier,
            estimator,
        })?;
        logical.record.discovery_algorithm =
            Some(self.graph_posterior_discovery_algorithm("graph_posterior"));
        logical.compile_physical(ctx)
    }

    /// Tabular conditional-effect compile over a placeholder DAG.
    fn compile_graph_posterior_conditional_effect(
        &self,
        data: &TabularData,
        q: &antecedent_core::ConditionalEffectQuery,
        ctx: &ExecutionContext,
    ) -> Result<PhysicalExecutionPlan, CausalError> {
        let n_vars = u32::try_from(data.schema().len()).map_err(|_| CausalError::Compile {
            message: "too many variables for graph-posterior compile".into(),
        })?;
        let stub = Dag::with_variables(n_vars);
        let atom_kind = self.graph_posterior.as_ref().map(|gp| gp.atom_kind);
        let (identifier, estimator) = match atom_kind {
            Some(
                antecedent_discovery::GraphPosteriorAtomKind::Cpdag
                | antecedent_discovery::GraphPosteriorAtomKind::Pag,
            ) => self.resolve_class_conditional_pair(),
            _ => (
                Arc::from(IdentifierId::BackdoorAdjustment.as_str()),
                match &self.inference {
                    InferenceMode::Frequentist => {
                        Arc::from(EstimatorId::ConditionalLinearAdjustment.as_str())
                    }
                    InferenceMode::Bayesian(_) => {
                        Arc::from(EstimatorId::BayesianConditional.as_str())
                    }
                },
            ),
        };
        let mut logical = compile_logical_static_ate(StaticAteCompileInput {
            data,
            graph: &stub,
            query: &q.inner,
            validation_suite: self.validation_suite_id(),
            identifier,
            estimator,
        })?;
        logical.record.plan_id = Arc::from("static_conditional_graph_posterior");
        logical.query = CausalQuery::ConditionalEffect(q.clone());
        logical.record.discovery_algorithm =
            Some(self.graph_posterior_discovery_algorithm("graph_posterior"));
        logical.compile_physical(ctx)
    }

    /// Trial-transport compile: every posterior atom is a selection diagram, identified by
    /// sID and estimated by the trial-to-target weighting of the supplied probabilities.
    fn compile_graph_posterior_transport(
        &self,
        data: &TabularData,
        q: &antecedent_core::TransportQuery,
        ctx: &ExecutionContext,
    ) -> Result<PhysicalExecutionPlan, CausalError> {
        let mut logical = self.compile_logical_transport(data, q)?;
        logical.record.discovery_algorithm =
            Some(self.graph_posterior_discovery_algorithm("graph_posterior"));
        logical.compile_physical(ctx)
    }

    /// Interference compile: the exposure contrast is identified by the assignment design, so
    /// the plan is the design's own, recorded over the posterior.
    fn compile_graph_posterior_interference(
        &self,
        data: &TabularData,
        q: &antecedent_core::InterferenceQuery,
        ctx: &ExecutionContext,
    ) -> Result<PhysicalExecutionPlan, CausalError> {
        let mut logical = self.compile_logical_interference(data, q)?;
        logical.record.discovery_algorithm =
            Some(self.graph_posterior_discovery_algorithm("graph_posterior"));
        logical.compile_physical(ctx)
    }

    /// Temporal-effect compile over a DBN or temporal-class posterior.
    fn compile_graph_posterior_temporal_effect(
        &self,
        data: &TimeSeriesData,
        q: &TemporalEffectQuery,
        ctx: &ExecutionContext,
    ) -> Result<PhysicalExecutionPlan, CausalError> {
        let class = match &self.data {
            DataInput::Event(_) => DataClassification::Event,
            _ => DataClassification::Temporal,
        };
        let mut logical = compile_logical_temporal_effect_classified(
            data,
            &TemporalDag::empty(),
            q,
            self.split,
            false,
            class,
        )?;
        let class_atom =
            self.graph_posterior.as_ref().map(|gp| gp.atom_kind).is_some_and(is_class_atom);
        if self.lagged_admg_atoms()? {
            // Every atom is identified on the unrolled window of its own template and
            // evaluated by the discrete plug-in on the lag-aligned window table.
            logical.record.identifier = Some(Arc::from(IdentifierId::GeneralId.as_str()));
            logical.record.estimator = Some(Arc::from(EstimatorId::FunctionalEffect.as_str()));
        } else if class_atom && !q.effect_modifiers.is_empty() {
            // The conditional point is the completion-weighted linear producer,
            // including under a Bayesian isotropic prior. It is not the ATE g-computation.
            logical.record.identifier =
                Some(Arc::from(IdentifierId::GeneralizedAdjustment.as_str()));
            logical.record.estimator =
                Some(Arc::from(EstimatorId::TemporalLinearAdjustment.as_str()));
        } else if q.is_multi_step_sustained() {
            logical.record.estimator =
                Some(Arc::from(EstimatorId::TemporalSequentialGcomp.as_str()));
        } else if self.inference.is_bayesian() {
            logical.record.estimator = Some(Arc::from(EstimatorId::BayesianTemporalGcomp.as_str()));
        }
        logical.record.validation_suite = self.validation_suite_id();
        logical.record.discovery_algorithm =
            Some(self.graph_posterior_discovery_algorithm("dbn_posterior"));
        logical.compile_physical(ctx)
    }

    /// Whether the study's posterior is a lagged temporal-ADMG posterior.
    ///
    /// # Errors
    ///
    /// A `TemporalAdmg` posterior without per-atom lag masks: its atoms are contemporaneous
    /// and read tabular rows, not a series.
    pub(super) fn lagged_admg_atoms(&self) -> Result<bool, CausalError> {
        let Some(posterior) = self.graph_posterior.as_ref() else { return Ok(false) };
        if posterior.atom_kind != antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg {
            return Ok(false);
        }
        if !posterior.is_lagged_temporal_admg() {
            return Err(CausalError::Unsupported {
                message: "a TemporalAdmg graph posterior over a series needs per-atom lag masks \
                          and max_lag (lagged templates); contemporaneous TemporalAdmg atoms read \
                          tabular rows",
            });
        }
        Ok(true)
    }

    /// Temporal-response compile over a DBN or temporal-class posterior.
    fn compile_graph_posterior_temporal_response(
        &self,
        data: &TimeSeriesData,
        q: &antecedent_core::ResponseQuery,
        ctx: &ExecutionContext,
    ) -> Result<PhysicalExecutionPlan, CausalError> {
        if self.lagged_admg_atoms()? {
            return Err(CausalError::Unsupported {
                message: "a lagged temporal-ADMG graph posterior answers a pulse, sustained or \
                          conditional temporal contrast and a temporal mediation; a temporal \
                          response over latent-confounded atoms is not a per-atom window route",
            });
        }
        super::temporal_posterior_response::dbn_posterior_response_supported(q)?;
        let mut logical = compile_logical_temporal_response(data, &TemporalDag::empty(), q, false)?;
        if self.graph_posterior.as_ref().map(|gp| gp.atom_kind).is_some_and(is_class_atom) {
            logical.record.identifier =
                Some(Arc::from(IdentifierId::GeneralizedAdjustment.as_str()));
        }
        if self.inference.is_bayesian() {
            logical.record.estimator =
                Some(Arc::from(EstimatorId::TemporalResponseBayesian.as_str()));
        }
        logical.record.validation_suite = self.validation_suite_id();
        logical.record.discovery_algorithm =
            Some(self.graph_posterior_discovery_algorithm("dbn_posterior"));
        logical.compile_physical(ctx)
    }

    /// Temporal-mediation compile over a DBN or temporal-class posterior.
    fn compile_graph_posterior_temporal_mediation(
        &self,
        data: &TimeSeriesData,
        q: &antecedent_core::MediationQuery,
        ctx: &ExecutionContext,
    ) -> Result<PhysicalExecutionPlan, CausalError> {
        q.validate()?;
        let class = match &self.data {
            DataInput::Event(_) => DataClassification::Event,
            _ => DataClassification::Temporal,
        };
        let mut logical = compile_logical_temporal_effect_classified(
            data,
            &TemporalDag::empty(),
            &TemporalEffectQuery::pulse(q.treatment, q.outcome, 1.0),
            self.split,
            false,
            class,
        )?;
        if self.lagged_admg_atoms()? {
            // Each atom's three components are identified on its own window by the
            // path-specific identifier and evaluated by the discrete plug-in; one outcome
            // horizon names the window.
            super::temporal_admg_mediation::window_effect_query(q)?;
            logical.record.plan_id =
                Arc::from(super::plan_id::PlanId::TemporalAdmgMediation.as_str());
            logical.record.identifier =
                Some(Arc::from(IdentifierId::PathSpecificNatural.as_str()));
            logical.record.estimator = Some(Arc::from(EstimatorId::FunctionalEffect.as_str()));
        } else {
            logical.record.plan_id = Arc::from("temporal_mediation");
            logical.record.identifier = Some(Arc::from("temporal.mediation"));
            logical.record.estimator = Some(Arc::from(if self.inference.is_frequentist() {
                EstimatorId::TemporalMediation.as_str()
            } else {
                EstimatorId::BayesianTemporalMediation.as_str()
            }));
        }
        logical.record.validation_suite = self.validation_suite_id();
        logical.record.query_variables = Arc::from([q.treatment, q.outcome]);
        logical.query = CausalQuery::Mediation(q.clone());
        logical.record.discovery_algorithm =
            Some(self.graph_posterior_discovery_algorithm("dbn_posterior"));
        logical.compile_physical(ctx)
    }

    /// Cross-world compile over a temporal-class posterior: the linear temporal
    /// adjustment plan keyed by the posterior's class.
    fn compile_graph_posterior_cross_world(
        &self,
        data: &TimeSeriesData,
        ctx: &ExecutionContext,
    ) -> Result<PhysicalExecutionPlan, CausalError> {
        let (treatment, outcome) =
            super::temporal_class_point::cross_world_plan_variables(&self.query)?;
        let plan_id = match self.graph.class() {
            GraphClass::TemporalPag => "temporal_pag_cross_world",
            _ => "temporal_cpdag_cross_world",
        };
        let logical = LogicalAnalysisPlan::unsplit(
            antecedent_contract::LogicalAnalysisPlanRecord {
                plan_id: Arc::from(plan_id),
                data_classification: antecedent_contract::DataClassification::Temporal,
                discovery_algorithm: Some(
                    self.graph_posterior
                        .as_ref()
                        .and_then(|gp| gp.algorithm.clone())
                        .unwrap_or_else(|| Arc::from("graph_posterior")),
                ),
                graph_review_required: false,
                identifier: Some(Arc::from(IdentifierId::GeneralizedAdjustment.as_str())),
                estimator: Some(Arc::from(EstimatorId::TemporalLinearAdjustment.as_str())),
                validation_suite: self.validation_suite_id(),
                query_variables: Arc::from([treatment, outcome]),
            },
            self.query.clone(),
            data.row_count() as u64,
        );
        logical.compile_physical(ctx)
    }

    /// Discovery algorithm recorded on a graph-posterior plan, or `fallback` when the
    /// posterior names none.
    fn graph_posterior_discovery_algorithm(&self, fallback: &str) -> Arc<str> {
        self.graph_posterior
            .as_ref()
            .and_then(|gp| gp.algorithm.clone())
            .unwrap_or_else(|| Arc::from(fallback))
    }
}
