// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::graph::{Cpdag, Dag, Pag};
use antecedent_core::{
    AverageEffectQuery, ConditionalEffectQuery, ContinuousDoseResponseQuery, InterferenceQuery,
    InterventionalDistributionQuery, LocalPolynomialRatioQuery, LongitudinalRegimeQuery,
    MediationQuery, PanelDidQuery, PathSpecificEffectQuery, PolicyValueQuery,
    RandomizedEffectQuery, SurvivalQuery, SyntheticControlQuery, TransportQuery,
};
use antecedent_data::MultiEnvironmentData;

use super::dispatch::wire_pair;
use super::{
    Admg, Arc, CausalError, CausalQuery, DataClassification, EstimatorId, ExecutionContext,
    GraphClass, GraphRef, IdentifierId, InferenceMode, Intervention, LogicalAnalysisPlan,
    PanelData, PhysicalExecutionPlan, ResponseQuery, RouteInput, StaticAteCompileInput,
    StaticCpdagResponseCompileInput, StaticDistributionCompileInput, StaticMediationSource,
    StaticPagAteCompileInput, StaticPagResponseCompileInput, StaticPathSpecificCompileInput,
    StaticResponseCompileInput, TableView, TabularData, TemporalClassEffectCompileInput,
    TemporalDag, TemporalEffectQuery, TimeSeriesData, VariableId, admg_has_bidirected,
    admg_without_latents_to_dag, compile_logical_codetermined_joint, compile_logical_distribution,
    compile_logical_path_specific, compile_logical_static_ate, compile_logical_static_cpdag_ate,
    compile_logical_static_cpdag_response, compile_logical_static_pag_ate,
    compile_logical_static_pag_response, compile_logical_static_response,
    compile_logical_temporal_class_effect, compile_logical_temporal_effect,
    compile_logical_temporal_effect_classified, compile_logical_temporal_response, gcm_query_vars,
    reject_dag_only_on_pag, validate_static_pair,
};

fn mixed_admg(study: &super::Study) -> Result<Admg, CausalError> {
    study
        .graph
        .graph_ref()
        .projected_admg()?
        .ok_or_else(|| CausalError::Compile { message: "mixed graph required".into() })
}

impl super::Study {
    /// Compile logical plan only (inspectable semantics).
    ///
    /// # Errors
    ///
    /// Modality / query validation failures.
    ///
    /// # Panics
    ///
    /// Never in practice. [`Study::route_input`] asserts the
    /// [`crate::accepted::AcceptedGraph::class`] invariant once: the class tag is
    /// derived from the stored graph, so the matching accessor (`as_dag`, `as_pag`, …)
    /// is always `Some` for the variant that accessor feeds.
    pub fn compile_logical(&self) -> Result<LogicalAnalysisPlan, CausalError> {
        if self.graph_posterior.is_some() {
            // A graph posterior is a mixture over structures, not one structure — there is
            // no single logical plan to inspect. `self.graph` here is only the placeholder
            // shape built at `build()` time, never the real analysis input.
            return Err(CausalError::Unsupported {
                message: "graph-posterior analysis has no single inspectable logical plan; \
                          use compile()/run() instead",
            });
        }
        self.compile_logical_from(&self.route_input()?)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "forwarding table from each RouteInput variant to its named compile_logical_* builder; every arm only destructures one variant and passes its payload on, so the length is the enum's variant count"
    )]
    fn compile_logical_from(
        &self,
        input: &RouteInput<'_>,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        match *input {
            RouteInput::StaticResponseDag { data, query: q, graph } => {
                self.compile_logical_response_dag(data, q, graph)
            }
            RouteInput::StaticResponsePag { data, query: q, pag } => {
                self.compile_logical_response_pag(data, q, pag)
            }
            RouteInput::StaticResponseCpdag { data, query: q, cpdag } => {
                self.compile_logical_response_cpdag(data, q, cpdag)
            }
            RouteInput::CodeterminedJoint { data, query: q, admg } => {
                self.compile_logical_codetermined_joint_route(data, q, admg)
            }
            RouteInput::MixedAdmgResponse { data, query: q } => {
                self.compile_logical_mixed_admg_response(data, q)
            }
            RouteInput::StaticAteDag { data, query: q, graph } => {
                self.compile_logical_ate_dag(data, q, graph)
            }
            RouteInput::DistributionDag { data, query: q, graph } => {
                self.compile_logical_distribution_dag(data, q, graph)
            }
            RouteInput::DistributionMixed { data, query: q } => {
                self.compile_logical_distribution_mixed(data, q)
            }
            RouteInput::PathSpecificDag { data, query: q, graph } => {
                self.compile_logical_path_specific_dag(data, q, graph)
            }
            RouteInput::TemporalEffectClass { data, query, classification } => self
                .compile_logical_class_effect(
                    data,
                    query,
                    classification,
                    "temporal_cpdag_effect",
                    "temporal_pag_effect",
                ),
            RouteInput::TemporalEffectDag { data, query: q, graph, classification } => {
                self.compile_logical_temporal_effect_dag(data, q, graph, classification)
            }
            RouteInput::TemporalResponseDag { data, query: q, graph } => {
                self.compile_logical_temporal_response_dag(data, q, graph)
            }
            RouteInput::TemporalResponseClass { data, query: q } => {
                self.compile_logical_temporal_response_class(data, q)
            }
            RouteInput::MultiEnvTemporalEffect { multi, query: q, graph } => {
                self.compile_logical_multi_env_temporal_effect(multi, q, graph)
            }
            RouteInput::PanelTemporalResponseDag { panel, query: q, graph } => {
                self.compile_logical_panel_response_dag(panel, q, graph)
            }
            RouteInput::PanelTemporalResponseClass { panel, query } => {
                self.compile_logical_panel_class_response_from(panel, query)
            }
            RouteInput::PanelTemporalEffectClass { panel, query } => {
                self.compile_logical_panel_class_from(panel, query)
            }
            RouteInput::PanelTemporalEffectDag { panel, query: q, graph } => {
                self.compile_logical_panel_effect_dag(panel, q, graph)
            }
            RouteInput::StaticAtePag { data, query: q, pag } => {
                self.compile_logical_ate_pag(data, q, pag)
            }
            RouteInput::StaticAteCpdag { data, query: q, cpdag } => {
                self.compile_logical_ate_cpdag(data, q, cpdag)
            }
            RouteInput::ConditionalDag { data, query: q, graph } => {
                self.compile_logical_conditional_dag(data, q, graph)
            }
            RouteInput::ConditionalCpdag { data, query: q, cpdag } => {
                self.compile_logical_conditional_cpdag(data, q, cpdag)
            }
            RouteInput::ConditionalPag { data, query: q, pag } => {
                self.compile_logical_conditional_pag(data, q, pag)
            }
            RouteInput::TemporalMediationDag { data, query: q, graph } => {
                self.compile_logical_temporal_mediation_dag(data, q, graph)
            }
            RouteInput::TemporalMediationPag => Err(CausalError::Unsupported {
                message: "Latent-confounded TemporalPag natural mediation is outside the \
                              existing cross-world identification theory; the linear DAG \
                              mechanism product on completions cannot license it",
            }),
            RouteInput::TemporalMediationCpdag { data, query } => {
                self.compile_logical_temporal_mediation_class(data, query)
            }
            RouteInput::StaticMediationDag { data, graph, source } => {
                self.compile_logical_static_mediation(data, graph, source)
            }
            RouteInput::TransportMixed { data, query: q, .. } => {
                self.compile_logical_transport(data, q)
            }
            RouteInput::InterferenceDag { data, query: q, .. } => {
                self.compile_logical_interference(data, q)
            }
            RouteInput::RandomizedEffect { data, query: q } => {
                self.compile_logical_randomized(data, q)
            }
            RouteInput::PolicyValue { data, query: q } => {
                self.compile_logical_policy_value(data, q)
            }
            RouteInput::PanelDid { data, query: q } => self.compile_logical_panel_did(data, q),
            RouteInput::SyntheticControl { data, query: q } => {
                self.compile_logical_synthetic_control(data, q)
            }
            RouteInput::ContinuousDoseResponse { data, query: q } => {
                self.compile_logical_continuous_dose(data, q)
            }
            RouteInput::LocalPolynomialRatio { data, query: q } => {
                self.compile_logical_local_polynomial_ratio(data, q)
            }
            RouteInput::Survival { data, query: q } => self.compile_logical_survival(data, q),
            RouteInput::LongitudinalRegime { data, query: q } => {
                self.compile_logical_longitudinal_regime(data, q)
            }
            RouteInput::GcmDag { data, .. } => self.compile_logical_gcm(
                data.row_count() as u64,
                "gcm_query",
                DataClassification::Tabular,
            ),
            RouteInput::GcmTemporal { data } => self.compile_logical_gcm(
                data.row_count() as u64,
                "gcm_temporal_query",
                DataClassification::Temporal,
            ),
            RouteInput::TemporalCrossWorldClass { data, class } => {
                self.compile_logical_cross_world_class(data, class)
            }
            RouteInput::TemporalCrossWorldDag { data, .. } => {
                self.compile_logical_cross_world_dag(data)
            }
            // Physical compile builds this cell itself. Logical inspection keeps the
            // previous catch-all refusal.
            RouteInput::StaticAteMixed { .. } => Err(CausalError::Unsupported {
                message: "unsupported data/graph/query combination",
            }),
        }
    }

    /// Logical plan whose record rides on `self.query` and the study's validation suite;
    /// the study-query route arms differ only in the ids and query variables they pass.
    fn study_query_plan(
        &self,
        row_count_hint: u64,
        classification: DataClassification,
        plan_id: &str,
        identifier: &str,
        estimator: &str,
        query_variables: Arc<[VariableId]>,
    ) -> LogicalAnalysisPlan {
        LogicalAnalysisPlan::unsplit(
            antecedent_contract::LogicalAnalysisPlanRecord {
                plan_id: Arc::from(plan_id),
                data_classification: classification,
                discovery_algorithm: None,
                graph_review_required: false,
                identifier: Some(Arc::from(identifier)),
                estimator: Some(Arc::from(estimator)),
                validation_suite: self.validation_suite_id(),
                query_variables,
            },
            self.query.clone(),
            row_count_hint,
        )
    }

    fn compile_logical_response_dag(
        &self,
        data: &TabularData,
        q: &ResponseQuery,
        graph: &Dag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (identifier, estimator) = self.resolve_response_pair(q);
        compile_logical_static_response(StaticResponseCompileInput {
            data,
            graph,
            query: q,
            validation_suite: self.validation_suite_id(),
            identifier,
            estimator,
        })
    }

    fn compile_logical_response_pag(
        &self,
        data: &TabularData,
        q: &ResponseQuery,
        pag: &Pag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (identifier, estimator) = self.resolve_class_response_pair(q);
        compile_logical_static_pag_response(StaticPagResponseCompileInput {
            data,
            graph: pag,
            query: q,
            validation_suite: self.validation_suite_id(),
            identifier,
            estimator,
        })
    }

    fn compile_logical_response_cpdag(
        &self,
        data: &TabularData,
        q: &ResponseQuery,
        cpdag: &Cpdag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (identifier, estimator) = self.resolve_class_response_pair(q);
        compile_logical_static_cpdag_response(StaticCpdagResponseCompileInput {
            data,
            graph: cpdag,
            query: q,
            validation_suite: self.validation_suite_id(),
            identifier,
            estimator,
        })
    }

    fn compile_logical_codetermined_joint_route(
        &self,
        data: &TabularData,
        q: &ResponseQuery,
        admg: &Admg,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (identifier, estimator) = self.resolve_codetermined_joint_pair();
        compile_logical_codetermined_joint(
            data,
            admg,
            q,
            self.validation_suite_id(),
            identifier,
            estimator,
        )
    }

    fn compile_logical_mixed_admg_response(
        &self,
        data: &TabularData,
        q: &ResponseQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let admg = mixed_admg(self)?;
        if admg_has_bidirected(&admg) {
            compile_logical_admg_response(data, q, self.validation_suite_id(), self)
        } else {
            let dag = admg_without_latents_to_dag(&admg)?;
            let (identifier, estimator) = self.resolve_response_pair(q);
            compile_logical_static_response(StaticResponseCompileInput {
                data,
                graph: &dag,
                query: q,
                validation_suite: self.validation_suite_id(),
                identifier,
                estimator,
            })
        }
    }

    fn compile_logical_ate_dag(
        &self,
        data: &TabularData,
        q: &AverageEffectQuery,
        graph: &Dag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (identifier_id, estimator_id) = self.resolve_static_ids();
        self.ensure_rd_config_present(estimator_id)?;
        let (identifier, estimator) = wire_pair((identifier_id, estimator_id));
        compile_logical_static_ate(StaticAteCompileInput {
            data,
            graph,
            query: q,
            validation_suite: self.validation_suite_id(),
            identifier,
            estimator,
        })
    }

    fn compile_logical_distribution_dag(
        &self,
        data: &TabularData,
        q: &InterventionalDistributionQuery,
        graph: &Dag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (identifier, estimator) = self.resolve_distribution_pair();
        compile_logical_distribution(StaticDistributionCompileInput {
            data,
            graph,
            query: q,
            validation_suite: self.validation_suite_id(),
            identifier,
            estimator,
        })
    }

    fn compile_logical_distribution_mixed(
        &self,
        data: &TabularData,
        q: &InterventionalDistributionQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        super::static_path::ensure_admg_distribution_licensed(q, self.refute)?;
        let (identifier, estimator) = self.resolve_distribution_pair();
        let treatment =
            q.interventions.first().and_then(Intervention::primary_variable).ok_or_else(|| {
                CausalError::Compile {
                    message: "distribution query requires an intervention variable".into(),
                }
            })?;
        let outcome = *q.outcomes.first().ok_or_else(|| CausalError::Compile {
            message: "distribution query requires an outcome".into(),
        })?;
        let record = antecedent_contract::LogicalAnalysisPlanRecord {
            plan_id: Arc::from("static_admg_distribution"),
            data_classification: antecedent_contract::DataClassification::Tabular,
            discovery_algorithm: None,
            graph_review_required: false,
            identifier: Some(identifier),
            estimator: Some(estimator),
            validation_suite: self.validation_suite_id(),
            query_variables: Arc::from([treatment, outcome]),
        };
        let logical = LogicalAnalysisPlan::unsplit(
            record,
            CausalQuery::Distribution(q.clone()),
            data.row_count() as u64,
        );
        logical.validate()?;
        Ok(logical)
    }

    fn compile_logical_path_specific_dag(
        &self,
        data: &TabularData,
        q: &PathSpecificEffectQuery,
        graph: &Dag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (identifier, estimator) = self.resolve_path_pair();
        compile_logical_path_specific(StaticPathSpecificCompileInput {
            data,
            graph,
            query: q,
            validation_suite: self.validation_suite_id(),
            identifier,
            estimator,
        })
    }

    fn compile_logical_temporal_effect_dag(
        &self,
        data: &TimeSeriesData,
        q: &TemporalEffectQuery,
        graph: &TemporalDag,
        classification: DataClassification,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let mut plan = compile_logical_temporal_effect_classified(
            data,
            graph,
            q,
            self.split,
            false,
            classification,
        )?;
        if q.is_multi_step_sustained() {
            plan.record.estimator = Some(Arc::from(EstimatorId::TemporalSequentialGcomp.as_str()));
        } else if self.inference.is_bayesian() && q.effect_modifiers.is_empty() {
            plan.record.estimator = Some(Arc::from(EstimatorId::BayesianTemporalGcomp.as_str()));
        }
        Ok(plan)
    }

    fn compile_logical_temporal_response_dag(
        &self,
        data: &TimeSeriesData,
        q: &ResponseQuery,
        graph: &TemporalDag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let mut plan = compile_logical_temporal_response(data, graph, q, false)?;
        if self.inference.is_bayesian() {
            plan.record.estimator = Some(Arc::from(EstimatorId::TemporalResponseBayesian.as_str()));
        }
        Ok(plan)
    }

    fn compile_logical_temporal_response_class(
        &self,
        data: &TimeSeriesData,
        q: &ResponseQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let mut plan = compile_logical_temporal_response(data, &TemporalDag::empty(), q, false)?;
        plan.record.identifier = Some(Arc::from(IdentifierId::GeneralizedAdjustment.as_str()));
        plan.record.estimator =
            Some(Arc::from(EstimatorId::temporal_response_for(&self.inference).as_str()));
        Ok(plan)
    }

    fn compile_logical_multi_env_temporal_effect(
        &self,
        multi: &MultiEnvironmentData,
        q: &TemporalEffectQuery,
        graph: &TemporalDag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let data = multi
            .environment(0)
            .map_err(|e| CausalError::Compile { message: format!("multi-env: {e}") })?;
        compile_logical_temporal_effect_classified(
            data,
            graph,
            q,
            self.split,
            false,
            DataClassification::MultiEnvironment,
        )
    }

    fn compile_logical_panel_response_dag(
        &self,
        panel: &PanelData,
        q: &ResponseQuery,
        graph: &TemporalDag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let mut plan =
            compile_logical_temporal_response(panel_compile_series(panel)?, graph, q, false)?;
        plan.record.data_classification = DataClassification::Panel;
        plan.record.estimator =
            Some(Arc::from(EstimatorId::temporal_response_for(&self.inference).as_str()));
        Ok(plan)
    }

    fn compile_logical_panel_effect_dag(
        &self,
        panel: &PanelData,
        q: &TemporalEffectQuery,
        graph: &TemporalDag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let mut plan = compile_logical_temporal_effect_classified(
            panel_compile_series(panel)?,
            graph,
            q,
            self.split,
            false,
            DataClassification::Panel,
        )?;
        if self.inference.is_bayesian() {
            plan.record.estimator = Some(Arc::from(EstimatorId::BayesianTemporalGcomp.as_str()));
        }
        Ok(plan)
    }

    fn compile_logical_ate_pag(
        &self,
        data: &TabularData,
        q: &AverageEffectQuery,
        pag: &Pag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (identifier_id, estimator_id) = self.resolve_pag_ids();
        reject_dag_only_on_pag(&self.graph, identifier_id)?;
        let (identifier, estimator) = wire_pair((identifier_id, estimator_id));
        compile_logical_static_pag_ate(StaticPagAteCompileInput {
            data,
            graph: pag,
            query: q,
            validation_suite: self.validation_suite_id(),
            identifier,
            estimator,
        })
    }

    fn compile_logical_ate_cpdag(
        &self,
        data: &TabularData,
        q: &AverageEffectQuery,
        cpdag: &Cpdag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (identifier, estimator) = self.resolve_pag_pair();
        compile_logical_static_cpdag_ate(crate::planner::StaticCpdagAteCompileInput {
            data,
            graph: cpdag,
            query: q,
            validation_suite: self.validation_suite_id(),
            identifier,
            estimator,
        })
    }

    fn compile_logical_conditional_dag(
        &self,
        data: &TabularData,
        q: &ConditionalEffectQuery,
        graph: &Dag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (identifier, estimator) = self.resolve_conditional_pair();
        // Logical plan reuses static ATE metadata with conditional estimator.
        let mut plan = compile_logical_static_ate(StaticAteCompileInput {
            data,
            graph,
            query: &q.inner,
            validation_suite: self.validation_suite_id(),
            identifier,
            estimator,
        })?;
        plan.record.plan_id = Arc::from("static_conditional");
        plan.query = CausalQuery::ConditionalEffect(q.clone());
        Ok(plan)
    }

    fn compile_logical_conditional_cpdag(
        &self,
        data: &TabularData,
        q: &ConditionalEffectQuery,
        cpdag: &Cpdag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (identifier, estimator) = self.resolve_class_conditional_pair();
        let mut plan =
            compile_logical_static_cpdag_ate(crate::planner::StaticCpdagAteCompileInput {
                data,
                graph: cpdag,
                query: &q.inner,
                validation_suite: self.validation_suite_id(),
                identifier,
                estimator,
            })?;
        plan.record.plan_id = Arc::from("static_cpdag_conditional");
        plan.query = CausalQuery::ConditionalEffect(q.clone());
        Ok(plan)
    }

    fn compile_logical_conditional_pag(
        &self,
        data: &TabularData,
        q: &ConditionalEffectQuery,
        pag: &Pag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (identifier, estimator) = self.resolve_class_conditional_pair();
        let mut plan = compile_logical_static_pag_ate(StaticPagAteCompileInput {
            data,
            graph: pag,
            query: &q.inner,
            validation_suite: self.validation_suite_id(),
            identifier,
            estimator,
        })?;
        plan.record.plan_id = Arc::from("static_pag_conditional");
        plan.query = CausalQuery::ConditionalEffect(q.clone());
        Ok(plan)
    }

    fn compile_logical_temporal_mediation_dag(
        &self,
        data: &TimeSeriesData,
        q: &MediationQuery,
        graph: &TemporalDag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        let mut plan = compile_logical_temporal_effect(
            data,
            graph,
            &TemporalEffectQuery::pulse(q.treatment, q.outcome, 1.0),
            self.split,
            false,
        )?;
        plan.record.plan_id = Arc::from("temporal_mediation");
        plan.record.identifier = Some(Arc::from("temporal.mediation"));
        plan.record.estimator = Some(Arc::from(if self.inference.is_bayesian() {
            "temporal.mediation.bayesian"
        } else {
            "temporal.mediation"
        }));
        plan.record.validation_suite = self.validation_suite_id();
        plan.record.query_variables = Arc::from([q.treatment, q.outcome]);
        plan.query = CausalQuery::Mediation(q.clone());
        Ok(plan)
    }

    fn compile_logical_temporal_mediation_class(
        &self,
        data: &TimeSeriesData,
        query: &MediationQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        query.validate()?;
        let mut plan = compile_logical_temporal_effect_classified(
            data,
            &TemporalDag::empty(),
            &TemporalEffectQuery::pulse(query.treatment, query.outcome, 1.0),
            self.split,
            false,
            DataClassification::Temporal,
        )?;
        plan.record.plan_id = Arc::from("temporal_mediation_class");
        plan.record.identifier = Some(Arc::from(IdentifierId::GeneralizedAdjustment.as_str()));
        plan.record.estimator = Some(Arc::from(if self.inference.is_bayesian() {
            EstimatorId::BayesianTemporalMediation.as_str()
        } else {
            EstimatorId::TemporalMediation.as_str()
        }));
        plan.record.validation_suite = self.validation_suite_id();
        plan.record.query_variables = Arc::from([query.treatment, query.outcome]);
        plan.query = CausalQuery::Mediation(query.clone());
        Ok(plan)
    }

    fn compile_logical_static_mediation(
        &self,
        data: &TabularData,
        graph: &Dag,
        source: StaticMediationSource<'_>,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        if self.identifier.is_some_and(|id| id != IdentifierId::PathSpecificNatural)
            || self.estimator.is_some_and(|id| id != EstimatorId::StaticMediationLinear)
            || self.estimator_spec.is_some()
            || !self.custom_validators.is_empty()
            || self.split.is_some()
        {
            return Err(CausalError::Unsupported {
                message: "static mediation requires path_specific.natural and mediation.linear without custom estimators, scalar validators, or discovery split",
            });
        }
        let q = match source {
            StaticMediationSource::Mediation(query) => (*query).clone(),
            StaticMediationSource::Nested(query) => query.as_mediation_query(),
        };
        q.validate()?;
        let path = antecedent_core::PathSpecificEffectQuery {
            control: q.control.clone(),
            active: q.active.clone(),
            path_nodes: Arc::clone(&q.mediators),
            target_population: q.target_population.clone(),
            ..antecedent_core::PathSpecificEffectQuery::binary(q.treatment, q.outcome)
        };
        let mut plan = compile_logical_path_specific(StaticPathSpecificCompileInput {
            data,
            graph,
            query: &path,
            validation_suite: self.validation_suite_id(),
            identifier: Arc::from("path_specific.natural"),
            estimator: Arc::from("functional.effect"),
        })?;
        plan.record.plan_id = Arc::from("static_mediation");
        plan.record.estimator = Some(Arc::from("mediation.linear"));
        plan.query = match source {
            StaticMediationSource::Nested(nested) => CausalQuery::NestedCounterfactual(*nested),
            StaticMediationSource::Mediation(_) => CausalQuery::Mediation(q),
        };
        Ok(plan)
    }

    fn compile_logical_transport(
        &self,
        data: &TabularData,
        q: &TransportQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        if self.selection_diagram.is_none() || self.transport_trial.is_none() {
            return Err(CausalError::Unsupported {
                message: "TransportQuery requires a selection diagram and transport_trial columns",
            });
        }
        let (treatment, outcome) =
            q.response.functional.primary_pair().ok_or_else(|| CausalError::Compile {
                message: "TransportQuery inner response has no treatment/outcome".into(),
            })?;
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            "transport_trial",
            "transport.sid",
            if self.inference.is_bayesian() {
                "transport.trial_bayesian_bootstrap"
            } else {
                "transport.trial_ipw"
            },
            Arc::from([treatment, outcome]),
        ))
    }

    fn compile_logical_interference(
        &self,
        data: &TabularData,
        q: &InterferenceQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        if self.interference.is_none() {
            return Err(CausalError::Unsupported {
                message: "InterferenceQuery requires StudyBuilder::interference",
            });
        }
        let antecedent_core::InterferenceFunctional::ExposureContrast { outcome, .. } =
            q.functional;
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            "interference_design",
            "interference.design",
            if self.inference.is_bayesian() {
                "interference.bayesian_gaussian"
            } else if matches!(
                q.assignment,
                antecedent_core::AssignmentDesign::ClusterRandomization { .. }
            ) {
                "interference.cluster_neyman"
            } else if matches!(
                q.assignment,
                antecedent_core::AssignmentDesign::TwoStageSaturation { .. }
            ) {
                "interference.saturation_exact"
            } else if matches!(
                q.assignment,
                antecedent_core::AssignmentDesign::ObservedExposure { .. }
            ) {
                "interference.observational_ipw"
            } else {
                "interference.ht_hajek"
            },
            Arc::from([outcome, outcome]),
        ))
    }

    fn compile_logical_randomized(
        &self,
        data: &TabularData,
        q: &RandomizedEffectQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        if q.realized_assignment.len() != data.row_count() {
            return Err(CausalError::Compile {
                message: "randomized assignment metadata must align with the table rows".into(),
            });
        }
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            "randomized_trial_itt",
            "randomized.design",
            match &q.design {
                antecedent_core::RandomizationDesign::Bernoulli => {
                    if q.received_treatment.is_some() {
                        "randomized.wald_cace_late"
                    } else if q.fixed_cuped.is_some() {
                        "randomized.fixed_cuped_ht_itt"
                    } else if !q.ancova_covariates.is_empty() {
                        "randomized.ancova_itt"
                    } else {
                        "randomized.ht_itt"
                    }
                }
                antecedent_core::RandomizationDesign::Complete { .. }
                | antecedent_core::RandomizationDesign::Stratified { .. }
                | antecedent_core::RandomizationDesign::Factorial2x2 { .. }
                | antecedent_core::RandomizationDesign::Cluster { .. } => "randomized.neyman_itt",
                antecedent_core::RandomizationDesign::Switchback { .. } => {
                    "randomized.switchback_ht_itt"
                }
                antecedent_core::RandomizationDesign::MultiArm { .. } => "randomized.ht_itt",
            },
            Arc::from([q.outcome]),
        ))
    }

    fn compile_logical_policy_value(
        &self,
        data: &TabularData,
        q: &PolicyValueQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        if q.multi_action.as_ref().map_or(q.assignment.len(), |multi| multi.assignment.len())
            != data.row_count()
        {
            return Err(CausalError::Compile {
                message: "policy inputs must align with the table rows".into(),
            });
        }
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            if q.multi_action.is_some() {
                "randomized_policy_value_multi_action_ipw"
            } else if q.mu0.is_empty() {
                "randomized_policy_value_ipw"
            } else {
                "randomized_policy_value_dr"
            },
            "randomized.design",
            if q.multi_action.is_some() {
                "randomized.multi_action_ipw_policy"
            } else if q.mu0.is_empty() {
                "randomized.ipw_policy"
            } else {
                "randomized.dr_policy"
            },
            Arc::from([q.outcome]),
        ))
    }

    fn compile_logical_panel_did(
        &self,
        data: &TabularData,
        q: &PanelDidQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        if q.treated.len() != data.row_count() {
            return Err(CausalError::Compile {
                message: "panel DiD metadata must align with table rows".into(),
            });
        }
        let (plan_id, identifier, estimator) =
            super::checked_panel_did::panel_did_plan_ids(q.design);
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            plan_id,
            identifier,
            estimator,
            Arc::from([q.outcome]),
        ))
    }

    fn compile_logical_synthetic_control(
        &self,
        data: &TabularData,
        q: &SyntheticControlQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        if q.units.len() != data.row_count() {
            return Err(CausalError::Compile {
                message: "synthetic-control metadata must align with table rows".into(),
            });
        }
        let (plan_id, identifier, estimator) =
            super::checked_synthetic_control::synthetic_panel_plan_ids(q.method);
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            plan_id,
            identifier,
            estimator,
            Arc::from([q.outcome]),
        ))
    }

    fn compile_logical_continuous_dose(
        &self,
        data: &TabularData,
        q: &ContinuousDoseResponseQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        if q.baseline_groups.len() != data.row_count() {
            return Err(CausalError::Compile {
                message: "continuous-dose baseline groups must align with table rows".into(),
            });
        }
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            "policy.continuous_dose_response",
            IdentifierId::ContinuousDoseExchangeability.as_str(),
            EstimatorId::ContinuousDoseKernel.as_str(),
            Arc::from([q.outcome, q.dose, q.dose_density]),
        ))
    }

    fn compile_logical_local_polynomial_ratio(
        &self,
        data: &TabularData,
        q: &LocalPolynomialRatioQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        for variable in [q.outcome, q.treatment, q.running] {
            data.schema().get(variable).map_err(CausalError::rejected)?;
        }
        let (plan_id, identifier, estimator) =
            super::checked_local_polynomial_ratio::local_polynomial_ratio_plan_ids(q.kink);
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            &plan_id,
            &identifier,
            &estimator,
            Arc::from([q.outcome, q.treatment, q.running]),
        ))
    }

    fn compile_logical_survival(
        &self,
        data: &TabularData,
        q: &SurvivalQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        for id in [Some(q.duration), Some(q.event), Some(q.treatment), q.delayed_entry]
            .into_iter()
            .flatten()
        {
            data.schema().get(id).map_err(CausalError::rejected)?;
        }
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            "randomized.survival_product_limit",
            "randomized.design",
            "randomized.survival_product_limit",
            Arc::from([q.duration, q.event, q.treatment]),
        ))
    }

    fn compile_logical_longitudinal_regime(
        &self,
        data: &TabularData,
        q: &LongitudinalRegimeQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        if q.subject_ids.len() != data.row_count() {
            return Err(CausalError::Compile {
                message: "longitudinal subject histories must align with outcome rows".into(),
            });
        }
        data.schema().get(q.outcome).map_err(CausalError::rejected)?;
        let method_id = match q.method {
            antecedent_core::LongitudinalRegimeMethod::Ipw => "longitudinal.ipw_regime",
            antecedent_core::LongitudinalRegimeMethod::GFormula => "longitudinal.g_formula_regime",
            antecedent_core::LongitudinalRegimeMethod::SequentialDoublyRobust => {
                "longitudinal.sequential_dr_regime"
            }
            antecedent_core::LongitudinalRegimeMethod::MarginalStructuralModel => {
                "longitudinal.marginal_structural_model"
            }
        };
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            method_id,
            "randomized.design",
            method_id,
            Arc::from([q.outcome]),
        ))
    }

    /// Parametric SCM paths use a query- and inference-specific estimator identity.
    fn compile_logical_gcm(
        &self,
        row_count_hint: u64,
        plan_id: &str,
        classification: DataClassification,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        self.query.validate()?;
        let (treatment, outcome) = gcm_query_vars(&self.query)?;
        Ok(self.study_query_plan(
            row_count_hint,
            classification,
            plan_id,
            "gcm.parametric",
            match (&self.query, &self.inference) {
                (CausalQuery::AnomalyAttribution(_), InferenceMode::Bayesian(_)) => {
                    "gcm.fit.bayesian"
                }
                (CausalQuery::ChangeAttribution(_), InferenceMode::Bayesian(_)) => {
                    "gcm.attribution.bayesian"
                }
                _ => "gcm.fit",
            },
            Arc::from([treatment, outcome]),
        ))
    }

    fn compile_logical_cross_world_class(
        &self,
        data: &TimeSeriesData,
        class: GraphClass,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (treatment, outcome) = temporal_cross_world_endpoints(&self.query)?;
        let plan_id = match class {
            GraphClass::TemporalCpdag => "temporal_cpdag_cross_world",
            GraphClass::TemporalPag => "temporal_pag_cross_world",
            _ => {
                return Err(CausalError::Compile {
                    message: "temporal cross-world class plan needs a TemporalCpdag or TemporalPag"
                        .into(),
                });
            }
        };
        let mut plan = self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Temporal,
            plan_id,
            IdentifierId::GeneralizedAdjustment.as_str(),
            EstimatorId::TemporalLinearAdjustment.as_str(),
            Arc::from([treatment, outcome]),
        );
        plan.record.discovery_algorithm = self.graph.algorithm_id().map(Arc::from);
        Ok(plan)
    }

    fn compile_logical_cross_world_dag(
        &self,
        data: &TimeSeriesData,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (treatment, outcome) = temporal_cross_world_endpoints(&self.query)?;
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Temporal,
            "temporal.cross_world.linear",
            IdentifierId::TemporalBackdoorUnfolded.as_str(),
            EstimatorId::TemporalLinearAdjustment.as_str(),
            Arc::from([treatment, outcome]),
        ))
    }

    /// Compile logical → physical plan.
    ///
    /// # Errors
    ///
    /// Modality / resource failures.
    ///
    /// # Panics
    ///
    /// Never in practice; see [`Self::compile_logical`]'s panics note — the same
    /// class/accessor invariant applies here, and it is checked inside
    /// [`Study::route_input`], not again below.
    pub fn compile(&self, ctx: &ExecutionContext) -> Result<PhysicalExecutionPlan, CausalError> {
        if self.graph_posterior.is_some() {
            return self.compile_graph_posterior(ctx);
        }
        let input = self.route_input()?;
        self.compile_with_route(&input, ctx)
    }

    fn compile_with_route(
        &self,
        input: &RouteInput<'_>,
        ctx: &ExecutionContext,
    ) -> Result<PhysicalExecutionPlan, CausalError> {
        // Copy the parsed input so graph `.clone()` resolves to the owned graph,
        // not to cloning another reference.
        match *input {
            RouteInput::StaticAteMixed { data, query: q } => {
                self.compile_physical_static_ate_mixed(data, q, ctx)
            }
            RouteInput::MixedAdmgResponse { .. } => {
                let admg = mixed_admg(self)?;
                if admg_has_bidirected(&admg) {
                    self.compile_logical_from(input)?.compile_physical(ctx)
                } else {
                    let dag = admg_without_latents_to_dag(&admg)?;
                    self.compile_logical_from(input)?.compile_physical_with_graphs(
                        ctx,
                        None,
                        Some(dag),
                    )
                }
            }
            RouteInput::TemporalEffectDag { graph, .. }
            | RouteInput::TemporalCrossWorldDag { graph, .. }
            | RouteInput::TemporalResponseDag { graph, .. }
            | RouteInput::MultiEnvTemporalEffect { graph, .. }
            | RouteInput::PanelTemporalResponseDag { graph, .. }
            | RouteInput::PanelTemporalEffectDag { graph, .. }
            | RouteInput::TemporalMediationDag { graph, .. } => self
                .compile_logical_from(input)?
                .compile_physical_with_graph(ctx, Some(graph.clone())),
            RouteInput::StaticResponsePag { pag, .. }
            | RouteInput::StaticAtePag { pag, .. }
            | RouteInput::ConditionalPag { pag, .. } => self
                .compile_logical_from(input)?
                .compile_physical_with_all_graphs(ctx, None, None, Some(pag.clone())),
            RouteInput::ConditionalDag { graph, .. }
            | RouteInput::StaticMediationDag { graph, .. }
            | RouteInput::InterferenceDag { graph, .. }
            | RouteInput::GcmDag { graph, .. } => self
                .compile_logical_from(input)?
                .compile_physical_with_graphs(ctx, None, Some(graph.clone())),
            RouteInput::StaticResponseDag { .. }
            | RouteInput::StaticResponseCpdag { .. }
            | RouteInput::CodeterminedJoint { .. }
            | RouteInput::StaticAteDag { .. }
            | RouteInput::StaticAteCpdag { .. }
            | RouteInput::DistributionDag { .. }
            | RouteInput::DistributionMixed { .. }
            | RouteInput::PathSpecificDag { .. }
            | RouteInput::TemporalEffectClass { .. }
            | RouteInput::TemporalResponseClass { .. }
            | RouteInput::TemporalMediationPag
            | RouteInput::TemporalMediationCpdag { .. }
            | RouteInput::PanelTemporalResponseClass { .. }
            | RouteInput::PanelTemporalEffectClass { .. }
            | RouteInput::ConditionalCpdag { .. }
            | RouteInput::TransportMixed { .. }
            | RouteInput::RandomizedEffect { .. }
            | RouteInput::PolicyValue { .. }
            | RouteInput::ContinuousDoseResponse { .. }
            | RouteInput::PanelDid { .. }
            | RouteInput::SyntheticControl { .. }
            | RouteInput::LocalPolynomialRatio { .. }
            | RouteInput::Survival { .. }
            | RouteInput::LongitudinalRegime { .. }
            | RouteInput::GcmTemporal { .. }
            | RouteInput::TemporalCrossWorldClass { .. } => {
                self.compile_logical_from(input)?.compile_physical(ctx)
            }
        }
    }

    /// Physical compile of the mixed-graph ATE cell: latent-confounded graphs keep the ADMG
    /// plan, latent-free ones project to a DAG first.
    fn compile_physical_static_ate_mixed(
        &self,
        data: &TabularData,
        q: &AverageEffectQuery,
        ctx: &ExecutionContext,
    ) -> Result<PhysicalExecutionPlan, CausalError> {
        let admg = mixed_admg(self)?;
        if admg_has_bidirected(&admg) {
            let (identifier_id, estimator_id) = self.resolve_admg_ids();
            validate_static_pair(identifier_id, estimator_id)?;
            let (identifier, estimator) = wire_pair((identifier_id, estimator_id));
            q.validate()?;
            let record = antecedent_contract::LogicalAnalysisPlanRecord {
                plan_id: Arc::from("static_admg_ate"),
                data_classification: antecedent_contract::DataClassification::Tabular,
                discovery_algorithm: None,
                graph_review_required: false,
                identifier: Some(identifier),
                estimator: Some(estimator),
                validation_suite: self.validation_suite_id(),
                query_variables: Arc::from([q.treatment, q.outcome]),
            };
            let logical = LogicalAnalysisPlan::unsplit(
                record,
                CausalQuery::AverageEffect(q.clone()),
                data.row_count() as u64,
            );
            logical.validate()?;
            logical.compile_physical(ctx)
        } else {
            let dag = admg_without_latents_to_dag(&admg)?;
            let (identifier_id, estimator_id) = self.resolve_static_ids();
            self.ensure_rd_config_present(estimator_id)?;
            let (identifier, estimator) = wire_pair((identifier_id, estimator_id));
            let logical = compile_logical_static_ate(StaticAteCompileInput {
                data,
                graph: &dag,
                query: q,
                validation_suite: self.validation_suite_id(),
                identifier,
                estimator,
            })?;
            logical.compile_physical_with_graphs(ctx, None, Some(dag))
        }
    }

    fn compile_logical_panel_class_from(
        &self,
        panel: &PanelData,
        query: &TemporalEffectQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        super::super::builder::refuse_unlicensed_panel_route(
            &self.query,
            self.graph.class(),
            &self.inference,
            panel,
            self.split.as_ref(),
        )?;
        self.compile_logical_class_effect(
            panel_compile_series(panel)?,
            query,
            DataClassification::Panel,
            "panel_cpdag_effect",
            "panel_pag_effect",
        )
    }

    fn compile_logical_panel_class_response_from(
        &self,
        panel: &PanelData,
        query: &ResponseQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        super::super::builder::refuse_unlicensed_panel_route(
            &self.query,
            self.graph.class(),
            &self.inference,
            panel,
            self.split.as_ref(),
        )?;
        let mut plan = compile_logical_temporal_response(
            panel_compile_series(panel)?,
            &TemporalDag::empty(),
            query,
            false,
        )?;
        plan.record.data_classification = DataClassification::Panel;
        plan.record.identifier = Some(Arc::from(IdentifierId::GeneralizedAdjustment.as_str()));
        plan.record.estimator =
            Some(Arc::from(EstimatorId::temporal_response_for(&self.inference).as_str()));
        plan.record.plan_id = Arc::from(match self.graph.class() {
            GraphClass::TemporalCpdag => "panel_cpdag_response",
            GraphClass::TemporalPag => "panel_pag_response",
            _ => "panel_class_response",
        });
        Ok(plan)
    }

    fn compile_logical_class_effect(
        &self,
        data: &TimeSeriesData,
        query: &TemporalEffectQuery,
        classification: DataClassification,
        cpdag_plan: &'static str,
        pag_plan: &'static str,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (plan_id, nodes) = match self.graph.graph_ref() {
            GraphRef::TemporalCpdag(graph) => (cpdag_plan, graph.nodes()),
            GraphRef::TemporalPag(graph) => (pag_plan, graph.nodes()),
            _ => {
                return Err(CausalError::Compile {
                    message: "temporal class effect requires a TemporalCpdag or TemporalPag graph"
                        .into(),
                });
            }
        };
        let (identifier, estimator) = self.resolve_temporal_class_pair();
        let mut logical = compile_logical_temporal_class_effect(TemporalClassEffectCompileInput {
            data,
            nodes,
            query,
            split: self.split,
            data_classification: classification,
            plan_id,
            identifier,
            estimator,
        })?;
        if query.is_multi_step_sustained() {
            logical.record.estimator =
                Some(Arc::from(EstimatorId::TemporalSequentialGcomp.as_str()));
        } else if self.inference.is_bayesian() && query.effect_modifiers.is_empty() {
            logical.record.estimator = Some(Arc::from(EstimatorId::BayesianTemporalGcomp.as_str()));
        }
        logical.record.discovery_algorithm = self.graph.algorithm_id().map(Arc::from);
        Ok(logical)
    }
}

fn temporal_cross_world_endpoints(
    query: &CausalQuery,
) -> Result<(VariableId, VariableId), CausalError> {
    let pair = match query {
        CausalQuery::Distribution(q) => q
            .interventions
            .first()
            .and_then(Intervention::primary_variable)
            .zip(q.outcomes.first().copied()),
        CausalQuery::PathSpecific(q) => Some((q.treatment, q.outcome)),
        CausalQuery::NestedCounterfactual(q) => Some((q.treatment, q.outcome)),
        CausalQuery::Counterfactual(q) => q
            .interventions
            .first()
            .and_then(Intervention::primary_variable)
            .zip(q.outcomes.first().copied()),
        _ => None,
    };
    pair.ok_or_else(|| CausalError::Compile {
        message: "temporal cross-world query needs one treatment and one outcome".into(),
    })
}

pub(crate) fn compile_logical_admg_response(
    data: &TabularData,
    query: &ResponseQuery,
    validation_suite: Option<Arc<str>>,
    study: &super::Study,
) -> Result<LogicalAnalysisPlan, CausalError> {
    let (identifier_id, estimator_id) = study.resolve_admg_ids();
    validate_static_pair(identifier_id, estimator_id)?;
    let (identifier, estimator) = wire_pair((identifier_id, estimator_id));
    query.validate()?;
    let (treatment, outcome) =
        crate::analysis::proof_support::require_primary_pair(&query.functional)?;
    let record = antecedent_contract::LogicalAnalysisPlanRecord {
        plan_id: Arc::from("static_admg_response"),
        data_classification: antecedent_contract::DataClassification::Tabular,
        discovery_algorithm: None,
        graph_review_required: false,
        identifier: Some(identifier),
        estimator: Some(estimator),
        validation_suite,
        query_variables: Arc::from([treatment, outcome]),
    };
    let logical = LogicalAnalysisPlan::unsplit(
        record,
        CausalQuery::Response(query.clone()),
        data.row_count() as u64,
    );
    logical.validate()?;
    Ok(logical)
}

fn panel_compile_series(panel: &PanelData) -> Result<&TimeSeriesData, CausalError> {
    Ok(&panel.unit(0).map_err(|e| CausalError::Compile { message: format!("panel: {e}") })?.series)
}
