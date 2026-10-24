// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::analysis::builder::DataInput;

use crate::graph::{Cpdag, Dag, Pag};
use antecedent_core::{
    AverageEffectQuery, ConditionalEffectQuery, ContinuousDoseResponseQuery, HistoryMatchQuery,
    InterferenceQuery, InterventionalDistributionQuery, LocalPolynomialRatioQuery,
    LongitudinalRegimeQuery, MediationQuery, PanelDidQuery, PathSpecificEffectQuery,
    PolicyValueQuery, RandomizedEffectQuery, SequentialContrastQuery, SurvivalQuery,
    SyntheticControlQuery, TransportQuery,
};
use antecedent_data::MultiEnvironmentData;
use antecedent_graph::TemporalAdmg;

use super::dispatch::wire_pair;
use super::plan_id::PlanId;
use super::{
    Admg, Arc, CausalError, CausalQuery, DataClassification, EstimatorId, ExecutionContext,
    GraphClass, GraphRef, IdentifierId, InferenceMode, Intervention, LogicalAnalysisPlan,
    PanelData, PathSpecificMixedSource, PhysicalExecutionPlan, ResponseQuery, RouteInput,
    StaticAdmgPathSpecificCompileInput, StaticAteCompileInput, StaticCpdagPathSpecificCompileInput,
    StaticCpdagResponseCompileInput, StaticDistributionCompileInput, StaticMediationSource,
    StaticPagAteCompileInput, StaticPagPathSpecificCompileInput, StaticPagResponseCompileInput,
    StaticPathSpecificCompileInput, StaticResponseCompileInput, TableView, TabularData,
    TemporalClassEffectCompileInput,
    TemporalDag, TemporalEffectQuery, TimeSeriesData, VariableId, admg_has_bidirected,
    admg_without_latents_to_dag, compile_logical_codetermined_joint, compile_logical_distribution,
    compile_logical_path_specific, compile_logical_path_specific_admg,
    compile_logical_path_specific_cpdag, compile_logical_path_specific_pag,
    compile_logical_static_ate, compile_logical_static_cpdag_ate,
    compile_logical_static_cpdag_response,
    compile_logical_static_pag_ate, compile_logical_static_pag_response,
    compile_logical_static_response, compile_logical_temporal_class_effect,
    compile_logical_temporal_effect, compile_logical_temporal_effect_classified,
    compile_logical_temporal_response, gcm_query_vars, reject_dag_only_on_pag,
    validate_static_pair,
};
use crate::analysis::conditional_admg::{
    FrequentistAdmgConditional, compile_temporal_admg_conditional_bayesian,
    compile_temporal_admg_conditional_frequentist,
};

#[cfg(test)]
#[path = "prepared_gaussian_tests.rs"]
mod prepared_gaussian_tests;

/// The path-restricted query a natural-effect source identifies, and the query its plan
/// records: a natural direct effect reads its mediator as the path node.
fn path_specific_source_queries(
    source: PathSpecificMixedSource<'_>,
) -> Result<(PathSpecificEffectQuery, CausalQuery), CausalError> {
    Ok(match source {
        PathSpecificMixedSource::PathSpecific(query) => {
            (query.clone(), CausalQuery::PathSpecific(query.clone()))
        }
        PathSpecificMixedSource::Mediation(mediation) => {
            mediation.validate()?;
            (
                PathSpecificEffectQuery {
                    control: mediation.control.clone(),
                    active: mediation.active.clone(),
                    path_nodes: Arc::clone(&mediation.mediators),
                    target_population: mediation.target_population.clone(),
                    ..PathSpecificEffectQuery::binary(mediation.treatment, mediation.outcome)
                },
                CausalQuery::Mediation(mediation.clone()),
            )
        }
        PathSpecificMixedSource::Nested(nested) => {
            let natural_direct = nested.as_mediation_query();
            natural_direct.validate()?;
            (
                PathSpecificEffectQuery {
                    control: natural_direct.control.clone(),
                    active: natural_direct.active.clone(),
                    path_nodes: Arc::clone(&natural_direct.mediators),
                    target_population: natural_direct.target_population.clone(),
                    ..PathSpecificEffectQuery::binary(
                        natural_direct.treatment,
                        natural_direct.outcome,
                    )
                },
                CausalQuery::NestedCounterfactual(*nested),
            )
        }
    })
}

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
        let mut plan = match *input {
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
            RouteInput::PathSpecificMixed { data, source } => {
                self.compile_logical_path_specific_mixed(data, source)
            }
            RouteInput::PathSpecificCpdag { data, source } => {
                self.compile_logical_path_specific_cpdag_cell(data, source)
            }
            RouteInput::PathSpecificPag { data, source } => {
                self.compile_logical_path_specific_pag_cell(data, source)
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
            RouteInput::TemporalEffectAdmg { data, query: q, graph } => {
                self.compile_logical_temporal_effect_admg(data, q, graph)
            }
            RouteInput::TemporalResponseDag { data, query: q, graph } => {
                self.compile_logical_temporal_response_dag(data, q, graph)
            }
            RouteInput::TemporalResponseClass { data, query: q } => {
                self.compile_logical_temporal_response_class(data, q)
            }
            RouteInput::TemporalResponseAdmg { data, query: q, graph } => {
                self.compile_logical_temporal_response_admg(data, q, graph)
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
            RouteInput::TemporalMediationPag { data, query } => {
                self.compile_logical_temporal_mediation_pag(data, query)
            }
            RouteInput::TemporalMediationCpdag { data, query } => {
                self.compile_logical_temporal_mediation_class(data, query)
            }
            RouteInput::TemporalMediationAdmg { data, query } => {
                self.compile_logical_temporal_mediation_admg(data, query)
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
            RouteInput::TemporalDesign { data, query: q } => {
                self.compile_logical_temporal_design(data, q)
            }
            RouteInput::IvBounds { data, query: q } => self.compile_logical_iv_bounds(data, q),
            RouteInput::AssociationStatic { data, query: q } => {
                self.compile_logical_association(data, q, DataClassification::Tabular)
            }
            RouteInput::AssociationTemporal { data, query: q } => {
                self.compile_logical_association(data, q, DataClassification::Temporal)
            }
            RouteInput::CounterfactualClass { data, query: q } => {
                self.compile_logical_counterfactual_class(data, q)
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
            RouteInput::GcmClass { data, .. } => self.compile_logical_gcm(
                data.row_count() as u64,
                PlanId::GcmClassQuery.as_str(),
                DataClassification::Tabular,
            ),
            RouteInput::GcmTemporalClass { data, .. } => self.compile_logical_gcm(
                data.row_count() as u64,
                PlanId::GcmTemporalClassQuery.as_str(),
                DataClassification::Temporal,
            ),
            RouteInput::GcmAttributeDag { data, .. } => self.compile_logical_gcm(
                data.row_count() as u64,
                PlanId::GcmAttribute.as_str(),
                DataClassification::Tabular,
            ),
            RouteInput::GcmSimulateDag { data, .. } => self.compile_logical_gcm(
                data.row_count() as u64,
                PlanId::GcmSimulate.as_str(),
                DataClassification::Tabular,
            ),
            RouteInput::DiscoverStatic { query, .. } => {
                self.compile_logical_discover(query, DataClassification::Tabular)
            }
            RouteInput::DiscoverTemporal { query, classification } => {
                self.compile_logical_discover(query, classification)
            }
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
        }?;
        let schema = match &self.data {
            DataInput::Tabular(data) => data.schema(),
            DataInput::Temporal(data) | DataInput::Event(data) => data.schema(),
            DataInput::Panel(data) => data.schema(),
            DataInput::MultiEnv(data) => data.schema(),
        };
        plan.design_columns_hint = schema
            .variables()
            .iter()
            .map(|v| match v.value_type {
                antecedent_core::ValueType::Vector { width, .. } => u64::from(width.get()),
                _ => 1,
            })
            .try_fold(0_u64, u64::checked_add)
            .ok_or_else(|| CausalError::Resource {
                message: "encoded schema width overflow".into(),
            })?
            .max(1);
        Ok(plan)
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

    /// Logical plan of a path-restricted natural effect, or a natural direct effect, on a
    /// supplied mixed graph: the same identifier and estimator pair as the DAG cell, over the
    /// graph's ADMG.
    fn compile_logical_path_specific_mixed(
        &self,
        data: &TabularData,
        source: PathSpecificMixedSource<'_>,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        super::static_path::ensure_admg_path_specific_licensed(self.refute)?;
        let admg = mixed_admg(self)?;
        let (identifier, estimator) = self.resolve_path_pair();
        let (path, query) = path_specific_source_queries(source)?;
        let mut plan = compile_logical_path_specific_admg(
            StaticAdmgPathSpecificCompileInput {
                data,
                graph: &admg,
                query: &path,
                validation_suite: self.validation_suite_id(),
                identifier,
                estimator,
            },
            PlanId::StaticAdmgPathSpecific.as_str(),
        )?;
        plan.query = query;
        Ok(plan)
    }

    /// Logical plan of a path-restricted natural effect, or a natural direct effect, over the
    /// DAG completions of a supplied CPDAG.
    fn compile_logical_path_specific_cpdag_cell(
        &self,
        data: &TabularData,
        source: PathSpecificMixedSource<'_>,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        super::static_path::ensure_admg_path_specific_licensed(self.refute)?;
        let GraphRef::Cpdag(cpdag) = self.graph.graph_ref() else {
            return Err(CausalError::Compile {
                message: "class path-specific effect requires a Cpdag".into(),
            });
        };
        let (identifier, estimator) = self.resolve_path_pair();
        let (path, query) = path_specific_source_queries(source)?;
        let mut plan = compile_logical_path_specific_cpdag(
            StaticCpdagPathSpecificCompileInput {
                data,
                graph: cpdag,
                query: &path,
                validation_suite: self.validation_suite_id(),
                identifier,
                estimator,
            },
            PlanId::StaticCpdagPathSpecific.as_str(),
        )?;
        plan.query = query;
        Ok(plan)
    }

    /// Logical plan of a path-restricted natural effect, or a natural direct effect, over the
    /// ADMGs a supplied PAG leaves open.
    fn compile_logical_path_specific_pag_cell(
        &self,
        data: &TabularData,
        source: PathSpecificMixedSource<'_>,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        super::static_path::ensure_admg_path_specific_licensed(self.refute)?;
        let GraphRef::Pag(pag) = self.graph.graph_ref() else {
            return Err(CausalError::Compile {
                message: "PAG path-specific effect requires a Pag".into(),
            });
        };
        let (identifier, estimator) = self.resolve_path_pair();
        let (path, query) = path_specific_source_queries(source)?;
        let mut plan = compile_logical_path_specific_pag(
            StaticPagPathSpecificCompileInput {
                data,
                graph: pag,
                query: &path,
                validation_suite: self.validation_suite_id(),
                identifier,
                estimator,
            },
            PlanId::StaticPagPathSpecific.as_str(),
        )?;
        plan.query = query;
        Ok(plan)
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
            // A declared white-innovation premise changes the posterior, so it changes the
            // estimator the plan names; builder validation has already refused it elsewhere.
            let estimator = match &self.inference {
                InferenceMode::Bayesian(config) if config.white_innovation.is_some() => {
                    EstimatorId::BayesianTemporalWhiteInnovation
                }
                _ => EstimatorId::BayesianTemporalGcomp,
            };
            plan.record.estimator = Some(Arc::from(estimator.as_str()));
        }
        Ok(plan)
    }

    /// Logical plan of a pulse, sustained or conditional contrast on a temporal ADMG.
    ///
    /// The contrast is identified by the complete ID algorithm (IDC with modifiers) on the
    /// unrolled window of the template and evaluated by the functional plug-in on the lag-aligned
    /// window table, so the plan names `general.id` and `functional.effect`; a different pair
    /// would answer another question.
    fn compile_logical_temporal_effect_admg(
        &self,
        data: &TimeSeriesData,
        q: &TemporalEffectQuery,
        graph: &TemporalAdmg,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        let template_variables: std::collections::BTreeSet<VariableId> = graph
            .nodes()
            .iter()
            .filter_map(|node| match node {
                antecedent_core::NodeRef::Lagged { variable, .. } => Some(*variable),
                _ => None,
            })
            .collect();
        for variable in
            [q.treatment, q.outcome].into_iter().chain(q.effect_modifiers.iter().copied())
        {
            if !template_variables.contains(&variable) {
                return Err(crate::compile_reason!(
                    "unknown_variable",
                    "variable {} is not a variable of the temporal ADMG template; name a \
                     variable that appears in the template at some lag",
                    variable.raw()
                ));
            }
        }
        let schema_len = data.schema().len();
        if let Some(variable) = template_variables.iter().find(|v| v.as_usize() >= schema_len) {
            return Err(crate::compile_reason!(
                "unknown_variable",
                "template variable {} is outside the series schema of {schema_len} variables; \
                 the template names variables by the schema's dense ids",
                variable.raw()
            ));
        }
        if data.storage().weights().is_some() {
            return Err(crate::compile_reason!(
                "option_not_applicable",
                "row sampling weights do not apply to a temporal ADMG contrast: the window table \
                 is read from the unweighted series. Next attempt: drop the weights, or use a \
                 TemporalDag whose linear fit reads them"
            ));
        }
        if q.max_history_lag.is_some() {
            return Err(crate::compile_reason!(
                "option_not_applicable",
                "a declared history cap does not apply to a temporal ADMG contrast: the window \
                 holds the template's own lags, and a shorter history would silently turn their \
                 causes into latent ones. Next attempt: drop the cap, or edit the template"
            ));
        }
        if self.split.is_some() {
            return Err(crate::compile_reason!(
                "option_not_applicable",
                "a supplied TemporalAdmg needs no discovery/estimation split: the window table \
                 is built from the whole series. Next attempt: drop the split, or discover the \
                 structure first and analyse the accepted graph"
            ));
        }
        let (identifier_id, estimator_id) = self.resolve_admg_ids();
        if identifier_id != IdentifierId::GeneralId || estimator_id != EstimatorId::FunctionalEffect
        {
            return Err(crate::compile_reason!(
                "strategy_incompatible",
                "a contrast on a TemporalAdmg is identified by {:?} on the unrolled window and \
                 evaluated by {:?}; got identifier {:?} and estimator {:?}",
                IdentifierId::GeneralId.as_str(),
                EstimatorId::FunctionalEffect.as_str(),
                identifier_id.as_str(),
                estimator_id.as_str()
            ));
        }
        let (identifier, estimator) = wire_pair((identifier_id, estimator_id));
        let classification = match &self.data {
            DataInput::Event(_) => DataClassification::Event,
            _ => DataClassification::Temporal,
        };
        let record = antecedent_contract::LogicalAnalysisPlanRecord {
            plan_id: Arc::from(PlanId::TemporalAdmgContrast.as_str()),
            data_classification: classification,
            discovery_algorithm: None,
            graph_review_required: false,
            identifier: Some(identifier),
            estimator: Some(estimator),
            validation_suite: self.validation_suite_id(),
            query_variables: Arc::from([q.treatment, q.outcome]),
        };
        let mut logical = LogicalAnalysisPlan::unsplit(
            record,
            CausalQuery::TemporalEffect(q.clone()),
            data.row_count() as u64,
        );
        if let Ok(window) = antecedent_identify::temporal_admg::plan_temporal_admg_window(graph, q)
        {
            // The estimator reads one column per template variable and window period.
            logical.design_columns_hint =
                (u64::from(window.periods) * template_variables.len() as u64).max(1);
        }
        logical.validate()?;
        Ok(logical)
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

    /// Logical plan of a derivative at one horizon on a temporal ADMG.
    ///
    /// The derivative of a continuous treatment is the slope of the lag-aligned regression on an
    /// adjustment set of the unrolled window of the template, so the plan names the temporal
    /// response estimator of the inference mode; the window-ADMG adjustment is recorded in the
    /// identification derivation. Curves and intervention levels need a plug-in the
    /// latent-confounded window does not evaluate, so they are refused here.
    fn compile_logical_temporal_response_admg(
        &self,
        data: &TimeSeriesData,
        q: &ResponseQuery,
        graph: &TemporalAdmg,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        if !super::response_functional_is_derivative(&q.functional) {
            return Err(CausalError::Unsupported {
                message: "a temporal response on a TemporalAdmg is licensed for derivatives, \
                          elasticities, directional derivatives and Jacobians; a curve or an \
                          intervention level of a latent-confounded window has no \
                          continuous-treatment functional estimator",
            });
        }
        let template_variables: std::collections::BTreeSet<VariableId> = graph
            .nodes()
            .iter()
            .filter_map(|node| match node {
                antecedent_core::NodeRef::Lagged { variable, .. } => Some(*variable),
                _ => None,
            })
            .collect();
        for variable in q.functional.treatment_ids().into_iter().chain(q.functional.outcome_ids()) {
            if !template_variables.contains(&variable) {
                return Err(crate::compile_reason!(
                    "unknown_variable",
                    "variable {} is not a variable of the temporal ADMG template; name a \
                     variable that appears in the template at some lag",
                    variable.raw()
                ));
            }
        }
        let schema_len = data.schema().len();
        if let Some(variable) = template_variables.iter().find(|v| v.as_usize() >= schema_len) {
            return Err(crate::compile_reason!(
                "unknown_variable",
                "template variable {} is outside the series schema of {schema_len} variables; \
                 the template names variables by the schema's dense ids",
                variable.raw()
            ));
        }
        if data.storage().weights().is_some() {
            return Err(crate::compile_reason!(
                "option_not_applicable",
                "row sampling weights do not apply to a temporal ADMG derivative: the slope \
                 regression reads the unweighted series. Next attempt: drop the weights, or use \
                 a TemporalDag whose linear fit reads them"
            ));
        }
        if q.temporal.as_ref().is_some_and(|temporal| temporal.max_history_lag.is_some()) {
            return Err(crate::compile_reason!(
                "option_not_applicable",
                "a declared history cap does not apply to a temporal ADMG derivative: the window \
                 holds the template's own lags, and a shorter history would silently turn their \
                 causes into latent ones. Next attempt: drop the cap, or edit the template"
            ));
        }
        if self.split.is_some() {
            return Err(crate::compile_reason!(
                "option_not_applicable",
                "a supplied TemporalAdmg needs no discovery/estimation split: the slope \
                 regression reads the whole series. Next attempt: drop the split, or discover \
                 the structure first and analyse the accepted graph"
            ));
        }
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
    /// Dag conditional route. A `TemporalAdmg` compiles as its contemporaneous projection (the
    /// tabular rows carry no time index); the temporal
    /// conditional contrast (a `TemporalEffect` with modifiers on one series, identified by IDC
    /// on the unrolled window).
    fn compile_logical_conditional_mixed(
        &self,
        data: &TabularData,
        q: &ConditionalEffectQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        if let GraphRef::TemporalAdmg(temporal) = self.graph.graph_ref() {
            // The contemporaneous projection must compile (a mean contrast, identified by IDC);
            // the plan below is then the projection's.
            if self.inference.is_bayesian() {
                compile_temporal_admg_conditional_bayesian(temporal, q).map(drop)?;
            } else {
                compile_temporal_admg_conditional_frequentist(temporal, q).map(drop)?;
            }
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

    /// Logical plan of natural mediation under a temporal ADMG: the window plan of the pulse
    /// that shares its treatment, outcome and horizon, identified by the path-specific
    /// identifier over the unrolled window and evaluated by the functional plug-in.
    fn compile_logical_temporal_mediation_admg(
        &self,
        data: &TimeSeriesData,
        query: &MediationQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let GraphRef::TemporalAdmg(template) = self.graph.graph_ref() else {
            return Err(CausalError::Compile {
                message: "temporal ADMG mediation requires a supplied TemporalAdmg".into(),
            });
        };
        let effect = super::temporal_admg_mediation::window_effect_query(query)?;
        let mut plan = self.compile_logical_temporal_effect_admg(data, &effect, template)?;
        plan.record.plan_id = Arc::from(PlanId::TemporalAdmgMediation.as_str());
        plan.record.identifier = Some(Arc::from(IdentifierId::PathSpecificNatural.as_str()));
        plan.record.estimator = Some(Arc::from(EstimatorId::FunctionalEffect.as_str()));
        plan.record.query_variables = Arc::from([query.treatment, query.outcome]);
        plan.query = CausalQuery::Mediation(query.clone());
        Ok(plan)
    }

    /// Logical plan of natural mediation under a temporal PAG: the class plan of the pulse
    /// that shares its treatment, outcome and horizon, identified by the path-specific
    /// identifier over the ADMGs each completion of the unfolded window stands for and
    /// evaluated by the functional plug-in.
    fn compile_logical_temporal_mediation_pag(
        &self,
        data: &TimeSeriesData,
        query: &MediationQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let GraphRef::TemporalPag(template) = self.graph.graph_ref() else {
            return Err(CausalError::Compile {
                message: "temporal PAG mediation requires a supplied TemporalPag".into(),
            });
        };
        let effect = super::temporal_admg_mediation::window_effect_query(query)?;
        if self.identifier.is_some_and(|id| id != IdentifierId::PathSpecificNatural)
            || self.caller_selected_estimator().is_some_and(|id| id != EstimatorId::FunctionalEffect)
        {
            return Err(crate::compile_reason!(
                "strategy_incompatible",
                "mediation on a TemporalPag is identified by path_specific.natural and evaluated \
                 by functional.effect; got identifier {:?} and estimator {:?}",
                self.identifier,
                self.caller_selected_estimator()
            ));
        }
        if data.storage().weights().is_some() {
            return Err(crate::compile_reason!(
                "option_not_applicable",
                "row sampling weights do not apply to mediation on a temporal PAG: the window \
                 table is read from the unweighted series. Next attempt: drop the weights, or \
                 use a TemporalDag whose linear fit reads them"
            ));
        }
        if self.split.is_some() {
            return Err(crate::compile_reason!(
                "option_not_applicable",
                "a supplied TemporalPag needs no discovery/estimation split: the window table is \
                 built from the whole series. Next attempt: drop the split, or discover the \
                 structure first and analyse the accepted graph"
            ));
        }
        let variable_count = u32::try_from(data.schema().len()).map_err(|_| {
            CausalError::Compile { message: "series schema exceeds the variable id range".into() }
        })?;
        antecedent_identify::temporal_pag_mediation_window(template, &effect, query, variable_count)
            .map_err(CausalError::from)?;
        let mut plan = compile_logical_temporal_effect_classified(
            data,
            &TemporalDag::empty(),
            &effect,
            None,
            false,
            DataClassification::Temporal,
        )?;
        plan.record.plan_id = Arc::from(PlanId::TemporalPagMediation.as_str());
        plan.record.identifier = Some(Arc::from(IdentifierId::PathSpecificNatural.as_str()));
        plan.record.estimator = Some(Arc::from(EstimatorId::FunctionalEffect.as_str()));
        plan.record.validation_suite = self.validation_suite_id();
        plan.record.query_variables = Arc::from([query.treatment, query.outcome]);
        plan.query = CausalQuery::Mediation(query.clone());
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

    pub(super) fn compile_logical_transport(
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

    pub(super) fn compile_logical_interference(
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
            if self.inference.is_bayesian()
                && matches!(
                    q.assignment,
                    antecedent_core::AssignmentDesign::ClusterRandomization { .. }
                )
            {
                "interference.cluster_bayesian_bootstrap"
            } else if self.inference.is_bayesian() {
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
                super::checked_randomized::bayesian_randomized_estimator(self.estimator, q).as_str()
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
            super::checked_synthetic_control::synthetic_query_plan_ids(q)
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
            super::checked_local_polynomial_ratio::local_ratio_query_plan_ids(q)
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
        // The matching contrast has no posterior; a Bayesian request reaches the outcome-model
        // estimator only by selecting it, and otherwise keeps the matching plan so the typed
        // boundary refuses it at build with the estimator named as the next attempt.
        let plan = if self.inference.is_bayesian()
            && self.estimator == Some(EstimatorId::BayesianPeriodHistoryOutcomeModel)
        {
            PlanId::BayesianPeriodHistoryOutcomeModel
        } else {
            PlanId::PeriodHistoryMatching
        }
        .as_str();
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            plan,
            "randomized.design",
            plan,
            Arc::from([q.outcome, q.treated]),
        ))
    }

    fn compile_logical_iv_bounds(
        &self,
        data: &TabularData,
        q: &antecedent_core::IvBoundsQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        for id in super::checked_iv_bounds::query_columns(q) {
            data.schema().get(id).map_err(CausalError::rejected)?;
        }
        let plan = if self.inference.is_bayesian() {
            PlanId::BayesianIvBounds
        } else {
            PlanId::IvBoundsSharp
        }
        .as_str();
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            plan,
            "randomized.design",
            plan,
            Arc::from([q.outcome, q.treatment, q.instrument]),
        ))
    }

    /// Logical plan of a noncausal association. The plan names no causal identifier or
    /// estimator: the analysis is picked from the outcome scale by the declared selection rule
    /// and recorded on the lowered program.
    fn compile_logical_association(
        &self,
        data: &impl TableView,
        q: &antecedent_core::AssociationQuery,
        classification: DataClassification,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        for id in q.columns() {
            data.schema().get(id).map_err(CausalError::rejected)?;
        }
        let plan = if self.inference.is_bayesian() {
            PlanId::BayesianAssociation
        } else {
            PlanId::Association
        }
        .as_str();
        Ok(self.study_query_plan(
            data.row_count() as u64,
            classification,
            plan,
            super::checked_association::ASSOCIATION_IDENTIFIER,
            plan,
            Arc::from(q.columns()),
        ))
    }

    fn compile_logical_temporal_design(
        &self,
        data: &TabularData,
        q: &antecedent_core::TemporalDesignQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        if q.rows() != data.row_count() {
            return Err(CausalError::Compile {
                message: "temporal-design unit and period labels must align with the table rows"
                    .into(),
            });
        }
        for id in q.columns() {
            data.schema().get(id).map_err(CausalError::rejected)?;
        }
        let survival =
            matches!(q.assignment, antecedent_core::TemporalAssignment::RegimeSurvival { .. });
        let plan = match (survival, self.inference.is_bayesian()) {
            (false, false) => PlanId::TemporalDesignExcursion,
            (false, true) => PlanId::BayesianTemporalDesignExcursion,
            (true, false) => PlanId::TemporalRegimeSurvival,
            (true, true) => PlanId::BayesianTemporalRegimeSurvival,
        }
        .as_str();
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            plan,
            "randomized.design",
            plan,
            Arc::from(q.columns()),
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

    /// Structure search: the record names the method as the discovery algorithm
    /// and the estimator, and asks for review unless the answer is a posterior.
    fn compile_logical_discover(
        &self,
        query: &antecedent_core::DiscoverQuery,
        classification: DataClassification,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        use antecedent_core::DiscoveredClass;
        query.validate()?;
        let rows = match &self.data {
            DataInput::Tabular(data) => data.row_count() as u64,
            DataInput::Temporal(data) | DataInput::Event(data) => data.row_count() as u64,
            DataInput::Panel(panel) => panel.total_rows() as u64,
            DataInput::MultiEnv(multi) => {
                multi.environments().iter().map(|env| env.row_count() as u64).sum()
            }
        };
        let plan_id = if classification == DataClassification::Tabular {
            PlanId::StructureDiscovery
        } else {
            PlanId::TemporalStructureDiscovery
        };
        Ok(LogicalAnalysisPlan::unsplit(
            antecedent_contract::LogicalAnalysisPlanRecord {
                plan_id: Arc::from(plan_id.as_str()),
                data_classification: classification,
                discovery_algorithm: Some(Arc::from(query.method.id())),
                graph_review_required: !matches!(
                    query.output(),
                    DiscoveredClass::DagPosterior
                        | DiscoveredClass::DagEdgeMarginals
                        | DiscoveredClass::TemporalDagPosterior
                        | DiscoveredClass::CpdagPosterior
                        | DiscoveredClass::PagPosterior
                        | DiscoveredClass::StructureStability
                ),
                identifier: Some(Arc::from(super::structural_route::STRUCTURE_SEARCH_IDENTIFIER)),
                estimator: Some(Arc::from(format!("discovery.{}", query.method.id()))),
                validation_suite: None,
                query_variables: Arc::clone(&query.variables),
            },
            self.query.clone(),
            rows,
        ))
    }

    /// Logical plan of a counterfactual on an Admg, a temporal Admg (read as its contemporaneous
    /// projection), a Pag or a Cpdag. An Admg and a Pag are identified by ID* and evaluated by the
    /// discrete plug-in (`general.id`, `functional.effect`); a Cpdag runs the per-unit
    /// counterfactual on every DAG completion, so it names the structural-mechanism pair.
    fn compile_logical_counterfactual_class(
        &self,
        data: &TabularData,
        query: &antecedent_core::CounterfactualQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        if self.refute != super::RefuteSuite::None {
            return Err(CausalError::Unsupported {
                message: "a counterfactual on an Admg, Pag, Cpdag or TemporalAdmg runs at \
                          validation none; cheap/full remain Dag-only",
            });
        }
        query.validate()?;
        let (treatment, outcome) = gcm_query_vars(&self.query)?;
        let (identifier, estimator) = match self.graph.graph_ref() {
            GraphRef::Cpdag(_) => (IdentifierId::GcmParametric, EstimatorId::GcmFit),
            _ => (IdentifierId::GeneralId, EstimatorId::FunctionalEffect),
        };
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            PlanId::StaticClassCounterfactual.as_str(),
            identifier.as_str(),
            estimator.as_str(),
            Arc::from([treatment, outcome]),
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
                (
                    CausalQuery::Attribute(_) | CausalQuery::Simulate(_),
                    InferenceMode::Bayesian(_),
                ) if super::attribute_posterior_supported(&self.query) => "gcm.fit.bayesian",
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
    /// An admitted static ordinary-least-squares study returns the lowered
    /// linear/Gaussian physical plan. Its checked scientific and statistical program
    /// is the same retained product consumed by [`Study::prepare`]. No fitting occurs.
    ///
    /// Every returned plan carries the route's lowered program
    /// ([`PhysicalExecutionPlan::lowered`]); execution refuses a plan without one.
    ///
    /// # Errors
    ///
    /// Modality / resource failures; `public_builder_reentry` when `ctx` is already
    /// inside the dispatch of a lowered program.
    ///
    /// # Panics
    ///
    /// Never in practice; see [`Self::compile_logical`]'s panics note — the same
    /// class/accessor invariant applies here, and it is checked inside
    /// [`Study::route_input`], not again below.
    pub fn compile(&self, ctx: &ExecutionContext) -> Result<PhysicalExecutionPlan, CausalError> {
        self.compile_for_goal(&crate::analysis::lowering::CompileGoal::Estimate, ctx)
    }

    /// [`Self::compile`] with the lowered program's layer four built for `goal`.
    ///
    /// # Errors
    ///
    /// As [`Self::compile`]; and the route's statistical program cannot supply the
    /// carrier a decision goal demands.
    pub(super) fn compile_for_goal(
        &self,
        goal: &crate::analysis::lowering::CompileGoal,
        ctx: &ExecutionContext,
    ) -> Result<PhysicalExecutionPlan, CausalError> {
        crate::analysis::lowering::refuse_reentry(ctx)?;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "compile" });
        }
        if self.graph_posterior.is_some() {
            return self.compile_graph_posterior(goal, ctx);
        }
        let input = self.route_input()?;
        let checked_linear = matches!(&input, RouteInput::StaticAteDag { query, .. }
            if crate::analysis::compiled_linear::can_retain(self, query)
                && self.resolve_static_ids().1 == EstimatorId::LinearAdjustmentAte);
        // Selected-role and contrast quotes below replace the conservative whole-schema bound.
        let mut physical_context = ctx.clone();
        if checked_linear {
            physical_context.set_memory_limit(None);
        }
        let mut plan = self.compile_with_route(&input, &physical_context)?;
        if let RouteInput::StaticAteDag { data, query, .. } = input {
            if crate::analysis::compiled_linear::can_retain(self, query)
                && self.resolve_static_ids().1 == EstimatorId::LinearAdjustmentAte
            {
                plan.linear = crate::analysis::compiled_linear::CompiledLinearExecution::compile(
                    self, data, query, &mut plan, ctx,
                )?
                .map(std::sync::Arc::new);
            }
        }
        crate::analysis::lowering::attach(self, input, &mut plan, goal, ctx)?;
        Ok(plan)
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
            | RouteInput::GcmDag { graph, .. }
            | RouteInput::GcmAttributeDag { graph, .. }
            | RouteInput::GcmSimulateDag { graph, .. } => self
                .compile_logical_from(input)?
                .compile_physical_with_graphs(ctx, None, Some(graph.clone())),
            // A DAG schema is carried into the physical plan; every other structure class
            // compiles as the design alone, which never reads the structure.
            RouteInput::InterferenceDag { class, .. } => {
                let logical = self.compile_logical_from(input)?;
                match self.graph.as_dag() {
                    Some(graph) if class == GraphClass::Dag => {
                        logical.compile_physical_with_graphs(ctx, None, Some(graph.clone()))
                    }
                    _ => logical.compile_physical(ctx),
                }
            }
            RouteInput::DiscoverStatic { query, .. }
            | RouteInput::DiscoverTemporal { query, .. } => {
                let mut plan = self.compile_logical_from(input)?.compile_physical(ctx)?;
                // The search runs its own engine, not an adjustment design.
                plan.record.kernels = Arc::from([(
                    Arc::from(format!("discovery.{}", query.method.id())),
                    antecedent_contract::KernelSelection::DenseBackend,
                )]);
                Ok(plan)
            }
            RouteInput::StaticResponseDag { .. }
            | RouteInput::StaticResponseCpdag { .. }
            | RouteInput::CodeterminedJoint { .. }
            | RouteInput::StaticAteDag { .. }
            | RouteInput::StaticAteCpdag { .. }
            | RouteInput::DistributionDag { .. }
            | RouteInput::DistributionMixed { .. }
            | RouteInput::PathSpecificDag { .. }
            | RouteInput::PathSpecificMixed { .. }
            | RouteInput::PathSpecificCpdag { .. }
            | RouteInput::PathSpecificPag { .. }
            | RouteInput::TemporalEffectClass { .. }
            | RouteInput::TemporalEffectAdmg { .. }
            | RouteInput::TemporalResponseClass { .. }
            | RouteInput::TemporalResponseAdmg { .. }
            | RouteInput::TemporalMediationPag { .. }
            | RouteInput::TemporalMediationAdmg { .. }
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
            | RouteInput::TemporalDesign { .. }
            | RouteInput::IvBounds { .. }
            | RouteInput::AssociationStatic { .. }
            | RouteInput::AssociationTemporal { .. }
            | RouteInput::CounterfactualClass { .. }
            | RouteInput::GcmTemporal { .. }
            | RouteInput::GcmClass { .. }
            | RouteInput::GcmTemporalClass { .. }
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
    if super::response_functional_is_derivative(&query.functional) {
        return compile_logical_admg_derivative(data, query, validation_suite, study);
    }
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

/// Derivative, elasticity, directional derivative or Jacobian on a latent-confounded ADMG.
///
/// The identifier is generalized adjustment read on the ADMG's proper back-door graph, and the
/// estimator is the continuous response estimator the functional names (or the Bayesian
/// response under Bayesian inference), because the derivative of a continuous treatment is
/// evaluated on a regression over the adjustment set, not by the discrete general-ID plug-in.
fn compile_logical_admg_derivative(
    data: &TabularData,
    query: &ResponseQuery,
    validation_suite: Option<Arc<str>>,
    study: &super::Study,
) -> Result<LogicalAnalysisPlan, CausalError> {
    query.validate()?;
    let identifier_id = study.identifier.unwrap_or(IdentifierId::GeneralizedAdjustment);
    let estimator_id = study
        .caller_selected_estimator()
        .unwrap_or_else(|| EstimatorId::static_response_for(&query.functional, &study.inference));
    crate::strategy_table::validate_class_response_pair(identifier_id, estimator_id)?;
    let expected = EstimatorId::default_for_response(&query.functional);
    if estimator_id != expected && estimator_id != EstimatorId::ResponseBayesian {
        return Err(CausalError::Compile {
            message: format!(
                "response functional requires estimator {:?} or response.bayesian; got {:?}",
                expected.as_str(),
                estimator_id.as_str()
            ),
        });
    }
    let (identifier, estimator) = wire_pair((identifier_id, estimator_id));
    let query_variables: Arc<[VariableId]> = query
        .functional
        .treatment_ids()
        .into_iter()
        .chain(query.functional.outcome_ids())
        .collect::<Vec<_>>()
        .into();
    let record = antecedent_contract::LogicalAnalysisPlanRecord {
        plan_id: Arc::from(PlanId::StaticAdmgResponse.as_str()),
        data_classification: antecedent_contract::DataClassification::Tabular,
        discovery_algorithm: None,
        graph_review_required: false,
        identifier: Some(identifier),
        estimator: Some(estimator),
        validation_suite,
        query_variables,
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
