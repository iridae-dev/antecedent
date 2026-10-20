// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::graph::{Cpdag, Dag, Pag};
use antecedent_core::{
    AverageEffectQuery, ConditionalEffectQuery, ContinuousDoseResponseQuery, HistoryMatchQuery,
    InterferenceQuery, InterventionalDistributionQuery, LocalPolynomialRatioQuery,
    LongitudinalRegimeQuery, MediationQuery, PanelDidQuery, PathSpecificEffectQuery,
    PolicyValueQuery, RandomizedEffectQuery, SequentialContrastQuery, SurvivalQuery,
    SyntheticControlQuery, TransportQuery,
};
use antecedent_data::MultiEnvironmentData;

use super::dispatch::wire_pair;
use super::plan_id::PlanId;
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
use crate::analysis::conditional_admg::{
    FrequentistAdmgConditional, compile_temporal_admg_conditional_bayesian,
    compile_temporal_admg_conditional_frequentist,
};

#[path = "lowered_route.rs"]
mod lowered_route;

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
                    PlanId::TemporalCpdagEffect.as_str(),
                    PlanId::TemporalPagEffect.as_str(),
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
            RouteInput::ConditionalMixed { data, query: q } => {
                self.compile_logical_conditional_mixed(data, q)
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
            RouteInput::SequentialContrast { data, query: q } => {
                self.compile_logical_sequential_contrast(data, q)
            }
            RouteInput::HistoryMatch { data, query: q } => {
                self.compile_logical_history_match(data, q)
            }
            RouteInput::GcmDag { data, .. } => self.compile_logical_gcm(
                data.row_count() as u64,
                PlanId::GcmQuery.as_str(),
                DataClassification::Tabular,
            ),
            RouteInput::GcmTemporal { data } => self.compile_logical_gcm(
                data.row_count() as u64,
                PlanId::GcmTemporalQuery.as_str(),
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
            plan_id: Arc::from(PlanId::StaticAdmgDistribution.as_str()),
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
        if let Some(
            instrument @ (EstimatorId::TemporalLaggedInstrument
            | EstimatorId::BayesianTemporalLaggedInstrument),
        ) = self.estimator
        {
            plan.record.identifier = Some(Arc::from(IdentifierId::TemporalIvUnfolded.as_str()));
            plan.record.estimator = Some(Arc::from(instrument.as_str()));
        } else if q.is_multi_step_sustained() {
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
        if let Some(
            link @ (EstimatorId::TemporalGlmLinkContrast
            | EstimatorId::BayesianTemporalGlmLinkContrast),
        ) = self.estimator
        {
            plan.record.estimator = Some(Arc::from(link.as_str()));
        } else if self.inference.is_bayesian() {
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
        plan.record.plan_id = Arc::from(PlanId::StaticConditional.as_str());
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
        plan.record.plan_id = Arc::from(PlanId::StaticCpdagConditional.as_str());
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
        plan.record.plan_id = Arc::from(PlanId::StaticPagConditional.as_str());
        plan.query = CausalQuery::ConditionalEffect(q.clone());
        Ok(plan)
    }

    /// Identifier and estimator of a conditional effect on a latent-confounded ADMG. The
    /// identification is IDC at the two hard-set levels and the estimator evaluates the
    /// identified contrast; a different pair would answer another question.
    fn resolve_conditional_admg_ids(&self) -> Result<(IdentifierId, EstimatorId), CausalError> {
        let (identifier, estimator) = self.resolve_admg_ids();
        if identifier != FrequentistAdmgConditional::IDENTIFIER
            || estimator != FrequentistAdmgConditional::ESTIMATOR
        {
            return Err(crate::compile_reason!(
                "strategy_incompatible",
                "an ADMG ConditionalEffect is identified by {:?} and evaluated by {:?}; got \
                 identifier {:?} and estimator {:?}",
                FrequentistAdmgConditional::IDENTIFIER.as_str(),
                FrequentistAdmgConditional::ESTIMATOR.as_str(),
                identifier.as_str(),
                estimator.as_str()
            ));
        }
        Ok((identifier, estimator))
    }

    /// Logical plan of a conditional effect on a mixed graph.
    ///
    /// A latent-confounded `Admg` compiles the IDC contrast; a latent-free one is the
    /// Dag conditional route. A `TemporalAdmg` is refused: its contrast is a temporal one
    /// and is not identified, and the contemporaneous projection answers a different target.
    fn compile_logical_conditional_mixed(
        &self,
        data: &TabularData,
        q: &ConditionalEffectQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        if let GraphRef::TemporalAdmg(temporal) = self.graph.graph_ref() {
            let refusal = if self.inference.is_bayesian() {
                compile_temporal_admg_conditional_bayesian(temporal, q).map(drop)
            } else {
                compile_temporal_admg_conditional_frequentist(temporal, q).map(drop)
            };
            refusal?;
            return Err(CausalError::Compile {
                message: "a conditional effect on a TemporalAdmg has no compile arm".into(),
            });
        }
        let admg = mixed_admg(self)?;
        if !admg_has_bidirected(&admg) {
            let dag = admg_without_latents_to_dag(&admg)?;
            return self.compile_logical_conditional_dag(data, q, &dag);
        }
        let (identifier_id, estimator_id) = self.resolve_conditional_admg_ids()?;
        let (identifier, estimator) = wire_pair((identifier_id, estimator_id));
        q.validate()?;
        let record = antecedent_contract::LogicalAnalysisPlanRecord {
            plan_id: Arc::from(PlanId::StaticAdmgConditional.as_str()),
            data_classification: antecedent_contract::DataClassification::Tabular,
            discovery_algorithm: None,
            graph_review_required: false,
            identifier: Some(identifier),
            estimator: Some(estimator),
            validation_suite: self.validation_suite_id(),
            query_variables: Arc::from([q.inner.treatment, q.inner.outcome]),
        };
        let logical = LogicalAnalysisPlan::unsplit(
            record,
            CausalQuery::ConditionalEffect(q.clone()),
            data.row_count() as u64,
        );
        logical.validate()?;
        Ok(logical)
    }

    /// Physical compile of the mixed-graph conditional cell: a latent-confounded graph keeps
    /// the ADMG plan, a latent-free one projects to a DAG first.
    fn compile_physical_conditional_mixed(
        &self,
        data: &TabularData,
        q: &ConditionalEffectQuery,
        ctx: &ExecutionContext,
    ) -> Result<PhysicalExecutionPlan, CausalError> {
        let logical = self.compile_logical_conditional_mixed(data, q)?;
        let admg = mixed_admg(self)?;
        if admg_has_bidirected(&admg) {
            logical.compile_physical(ctx)
        } else {
            let dag = admg_without_latents_to_dag(&admg)?;
            logical.compile_physical_with_graphs(ctx, None, Some(dag))
        }
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
        plan.record.plan_id = Arc::from(PlanId::TemporalMediation.as_str());
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
        plan.record.plan_id = Arc::from(PlanId::TemporalMediationClass.as_str());
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
        plan.record.plan_id = Arc::from(PlanId::StaticMediation.as_str());
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
            crate::analysis::proof_support::require_primary_pair(&q.response.functional)?;
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            PlanId::TransportTrial.as_str(),
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
            PlanId::InterferenceDesign.as_str(),
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
            PlanId::RandomizedTrialItt.as_str(),
            "randomized.design",
            if self.inference.is_bayesian() {
                super::checked_randomized::bayesian_randomized_estimator(self.estimator, &q.design)
                    .as_str()
            } else {
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
                    | antecedent_core::RandomizationDesign::Cluster { .. } => {
                        "randomized.neyman_itt"
                    }
                    antecedent_core::RandomizationDesign::Switchback { .. } => {
                        "randomized.switchback_ht_itt"
                    }
                    antecedent_core::RandomizationDesign::MultiArm { .. } => "randomized.ht_itt",
                }
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
        if self.inference.is_bayesian() {
            return Ok(self.study_query_plan(
                data.row_count() as u64,
                DataClassification::Tabular,
                PlanId::BayesianPolicyValue.as_str(),
                "randomized.design",
                EstimatorId::BayesianBinaryPolicyValue.as_str(),
                Arc::from([q.outcome]),
            ));
        }
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            if q.multi_action.is_some() {
                PlanId::RandomizedPolicyValueMultiActionIpw.as_str()
            } else if q.mu0.is_empty() {
                PlanId::RandomizedPolicyValueIpw.as_str()
            } else {
                PlanId::RandomizedPolicyValueDr.as_str()
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
        let (plan_id, identifier, estimator) = if self.inference.is_bayesian() {
            super::checked_panel_did::bayesian_panel_did_plan_ids(q.design)
        } else {
            super::checked_panel_did::panel_did_plan_ids(q.design)
        };
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
        let (plan_id, identifier, estimator) = if self.inference.is_bayesian() {
            super::checked_synthetic_control::bayesian_synthetic_panel_plan_ids()
        } else {
            super::checked_synthetic_control::synthetic_panel_plan_ids(q.method)
        };
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
        let (plan, estimator) = if self.inference.is_bayesian() {
            (PlanId::BayesianDosePolicy, EstimatorId::BayesianDosePolicy)
        } else {
            (PlanId::PolicyContinuousDoseResponse, EstimatorId::ContinuousDoseKernel)
        };
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            plan.as_str(),
            IdentifierId::ContinuousDoseExchangeability.as_str(),
            estimator.as_str(),
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
        let (plan_id, identifier, estimator) = if self.inference.is_bayesian() {
            super::checked_local_polynomial_ratio::bayesian_local_polynomial_ratio_plan_ids(q.kink)
        } else {
            super::checked_local_polynomial_ratio::local_polynomial_ratio_plan_ids(q.kink)
        };
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
        let (plan, estimator) = if self.inference.is_bayesian() {
            (
                PlanId::BayesianPiecewiseSurvival.as_str(),
                EstimatorId::BayesianPiecewiseSurvival.as_str(),
            )
        } else {
            (
                PlanId::RandomizedSurvivalProductLimit.as_str(),
                PlanId::RandomizedSurvivalProductLimit.as_str(),
            )
        };
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            plan,
            "randomized.design",
            estimator,
            Arc::from([q.duration, q.event, q.treatment]),
        ))
    }

    fn compile_logical_history_match(
        &self,
        data: &TabularData,
        q: &HistoryMatchQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        for id in [q.outcome, q.treated]
            .into_iter()
            .chain(q.history.iter().copied())
            .chain(q.period)
            .chain(q.weight)
        {
            data.schema().get(id).map_err(CausalError::rejected)?;
        }
        let plan = PlanId::PeriodHistoryMatching.as_str();
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            plan,
            "randomized.design",
            plan,
            Arc::from([q.outcome, q.treated]),
        ))
    }

    fn compile_logical_sequential_contrast(
        &self,
        data: &TabularData,
        q: &SequentialContrastQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        if q.fold_of_row.len() != data.row_count() {
            return Err(CausalError::Compile {
                message: "sequential contrast folds must align with the table rows".into(),
            });
        }
        for id in [q.outcome, q.first_treatment, q.second_treatment]
            .into_iter()
            .chain(q.baseline.iter().copied())
            .chain(q.intermediate.iter().copied())
        {
            data.schema().get(id).map_err(CausalError::rejected)?;
        }
        let plan = if self.inference.is_bayesian() {
            PlanId::BayesianSequentialCrossfitContrast
        } else {
            PlanId::SequentialCrossfitContrastWald
        }
        .as_str();
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            plan,
            "randomized.design",
            plan,
            Arc::from([q.outcome, q.first_treatment, q.second_treatment]),
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
        let bayesian = self.inference.is_bayesian();
        let method_id = match q.method {
            _ if bayesian => PlanId::BayesianSequentialRegimeValue,
            antecedent_core::LongitudinalRegimeMethod::Ipw => PlanId::LongitudinalIpwRegime,
            antecedent_core::LongitudinalRegimeMethod::GFormula => {
                PlanId::LongitudinalGFormulaRegime
            }
            antecedent_core::LongitudinalRegimeMethod::SequentialDoublyRobust => {
                PlanId::LongitudinalSequentialDrRegime
            }
            antecedent_core::LongitudinalRegimeMethod::MarginalStructuralModel => {
                PlanId::LongitudinalMarginalStructuralModel
            }
        }
        .as_str();
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
            GraphClass::TemporalCpdag => PlanId::TemporalCpdagCrossWorld.as_str(),
            GraphClass::TemporalPag => PlanId::TemporalPagCrossWorld.as_str(),
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
        // The unit counterfactual of a linear additive-noise model has the posterior of its
        // slope times the contrast: the Bayesian twin runs the tempered temporal posterior,
        // not the ridge point of the frequentist fit.
        let estimator = if matches!(self.query, CausalQuery::Counterfactual(_))
            && self.inference.is_bayesian()
        {
            EstimatorId::BayesianTemporalGcomp
        } else {
            EstimatorId::TemporalLinearAdjustment
        };
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Temporal,
            PlanId::TemporalCrossWorldLinear.as_str(),
            IdentifierId::TemporalBackdoorUnfolded.as_str(),
            estimator.as_str(),
            Arc::from([treatment, outcome]),
        ))
    }

    /// Compile logical → physical plan.
    ///
    /// A static ordinary-least-squares intervention returns the physical record
    /// retained by the lowered linear/Gaussian program. Other routes keep their
    /// existing compile path.
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
        if self.admits_linear_gaussian() {
            return self
                .lower_linear_gaussian_intervention(ctx)
                .map(lowered_route::LoweredLinearGaussianRoute::into_physical);
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
            RouteInput::ConditionalMixed { data, query: q } => {
                self.compile_physical_conditional_mixed(data, q, ctx)
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
            | RouteInput::SequentialContrast { .. }
            | RouteInput::HistoryMatch { .. }
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
                plan_id: Arc::from(PlanId::StaticAdmgAte.as_str()),
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
            PlanId::PanelCpdagEffect.as_str(),
            PlanId::PanelPagEffect.as_str(),
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
            GraphClass::TemporalCpdag => PlanId::PanelCpdagResponse.as_str(),
            GraphClass::TemporalPag => PlanId::PanelPagResponse.as_str(),
            _ => PlanId::PanelClassResponse.as_str(),
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
        plan_id: Arc::from(PlanId::StaticAdmgResponse.as_str()),
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

#[cfg(test)]
mod dispatch_tests {
    use std::sync::Arc;

    use antecedent_core::{
        ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery, TemporalEffectQuery,
        TemporalPolicy,
    };
    use antecedent_data::TimeSeriesData;
    use antecedent_graph::ensure_lagged;

    use crate::{BayesianConfig, InferenceMode, RefuteSuite, Study};

    use super::{
        Dag, EstimatorId, ExecutionContext, PhysicalExecutionPlan, TabularData, TemporalDag,
    };

    fn estimator_id(plan: &PhysicalExecutionPlan) -> EstimatorId {
        plan.logical.record.estimator.as_deref().unwrap().parse().unwrap()
    }

    fn kernel_label(plan: &PhysicalExecutionPlan) -> &str {
        plan.record.kernels.first().unwrap().0.as_ref()
    }

    #[test]
    fn temporal_history_compile_dispatches_the_history_operator_estimator() {
        let treatment_col = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let outcome_col = [0.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0];
        let series = TimeSeriesData::from_f64_columns(
            [("t", treatment_col.as_slice()), ("y", outcome_col.as_slice())],
            1,
        )
        .unwrap();
        let treatment = antecedent_core::VariableId::from_raw(0);
        let outcome = antecedent_core::VariableId::from_raw(1);
        let mut graph = TemporalDag::empty();
        let lagged =
            ensure_lagged(&mut graph, treatment, antecedent_core::Lag::from_raw(1)).unwrap();
        let now =
            ensure_lagged(&mut graph, outcome, antecedent_core::Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(lagged, now).unwrap();
        let query = TemporalEffectQuery::pulse(treatment, outcome, 1.0)
            .with_policy(TemporalPolicy::pulse(-1))
            .with_horizon_steps(1);
        let study = Study::series(series)
            .graph(graph)
            .temporal_query(query)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let plan = study.compile(&ExecutionContext::for_tests(1)).unwrap();
        let id = estimator_id(&plan);
        // `TemporalLinearAdjustment::prepare` admits the sample anchor through
        // history operators. Compile records that estimator; it does not fit.
        assert_eq!(id, EstimatorId::TemporalLinearAdjustment);
        assert_eq!(kernel_label(&plan), id.kernel_label());
    }

    #[test]
    fn study_prediction_compile_dispatches_the_bayesian_response_grid() {
        let schema = antecedent_core::CausalSchemaBuilder::new()
            .continuous("t")
            .treatment()
            .continuous("y")
            .outcome()
            .build()
            .unwrap();
        let data = TabularData::try_from_schema_f64(
            schema.clone(),
            [("t", &[0.0, 1.0, 0.0, 1.0][..]), ("y", &[1.0, 3.0, 1.0, 3.0][..])],
        )
        .unwrap();
        let graph = Dag::from_named_edges(&schema, &[("t", "y")]).unwrap();
        let treatment = schema.id_of("t").unwrap();
        let outcome = schema.id_of("y").unwrap();
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome,
            treatment: ContinuousDomain::new(
                treatment,
                GridSpec::Values(Arc::from([0.0_f64, 1.0])),
            ),
        });
        let study = Study::tabular(data)
            .graph(graph)
            .query(query)
            .inference(InferenceMode::Bayesian(BayesianConfig::conjugate()))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let plan = study.compile(&ExecutionContext::for_tests(1)).unwrap();
        let id = estimator_id(&plan);
        // `ContinuousResponseEstimator::estimate_bayesian` streams this grid
        // through `stream_tile`. Compile records that estimator; it does not tile.
        assert_eq!(id, EstimatorId::ResponseBayesian);
        assert_eq!(kernel_label(&plan), id.kernel_label());
    }
}
