//! Static-graph logical and physical compilation.

use super::{
    Admg, Arc, AverageEffectQuery, CausalError, CausalQuery, ConditionalEffectQuery, Cpdag, Dag,
    EstimatorId, ExecutionContext, FrequentistAdmgConditional, GraphRef, IdentifierId,
    Intervention, InterventionalDistributionQuery, LogicalAnalysisPlan, Pag,
    PathSpecificEffectQuery, PathSpecificMixedSource, PhysicalExecutionPlan, PlanId, ResponseQuery,
    StaticAdmgPathSpecificCompileInput, StaticAteCompileInput, StaticCpdagPathSpecificCompileInput,
    StaticCpdagResponseCompileInput, StaticDistributionCompileInput, StaticPagAteCompileInput,
    StaticPagPathSpecificCompileInput, StaticPagResponseCompileInput,
    StaticPathSpecificCompileInput, StaticResponseCompileInput, TableView, TabularData,
    admg_has_bidirected, admg_without_latents_to_dag, compile_logical_admg_response,
    compile_logical_codetermined_joint, compile_logical_distribution,
    compile_logical_path_specific, compile_logical_path_specific_admg,
    compile_logical_path_specific_cpdag, compile_logical_path_specific_pag,
    compile_logical_static_ate, compile_logical_static_cpdag_ate,
    compile_logical_static_cpdag_response, compile_logical_static_pag_ate,
    compile_logical_static_pag_response, compile_logical_static_response,
    compile_temporal_admg_conditional_bayesian, compile_temporal_admg_conditional_frequentist,
    mixed_admg, path_specific_source_queries, reject_dag_only_on_pag, wire_pair,
};

impl super::super::Study {
    pub(super) fn compile_logical_response_dag(
        &self,
        data: &TabularData,
        q: &ResponseQuery,
        graph: &Dag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (identifier, estimator) = self.resolve_response_ids(q);
        compile_logical_static_response(StaticResponseCompileInput {
            data,
            graph,
            query: q,
            validation_suite: self.validation_suite_id(),
            identifier,
            estimator,
        })
    }

    pub(super) fn compile_logical_response_pag(
        &self,
        data: &TabularData,
        q: &ResponseQuery,
        pag: &Pag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (identifier, estimator) = self.resolve_class_response_ids(q);
        compile_logical_static_pag_response(StaticPagResponseCompileInput {
            data,
            graph: pag,
            query: q,
            validation_suite: self.validation_suite_id(),
            identifier,
            estimator,
        })
    }

    pub(super) fn compile_logical_response_cpdag(
        &self,
        data: &TabularData,
        q: &ResponseQuery,
        cpdag: &Cpdag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (identifier, estimator) = self.resolve_class_response_ids(q);
        compile_logical_static_cpdag_response(StaticCpdagResponseCompileInput {
            data,
            graph: cpdag,
            query: q,
            validation_suite: self.validation_suite_id(),
            identifier,
            estimator,
        })
    }

    pub(super) fn compile_logical_codetermined_joint_route(
        &self,
        data: &TabularData,
        q: &ResponseQuery,
        admg: &Admg,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (identifier, estimator) = self.resolve_codetermined_joint_ids();
        compile_logical_codetermined_joint(
            data,
            admg,
            q,
            self.validation_suite_id(),
            identifier,
            estimator,
        )
    }

    pub(super) fn compile_logical_mixed_admg_response(
        &self,
        data: &TabularData,
        q: &ResponseQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let admg = mixed_admg(self)?;
        if admg_has_bidirected(&admg) {
            compile_logical_admg_response(data, q, self.validation_suite_id(), self)
        } else {
            let dag = admg_without_latents_to_dag(&admg)?;
            let (identifier, estimator) = self.resolve_response_ids(q);
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

    pub(super) fn compile_logical_ate_dag(
        &self,
        data: &TabularData,
        q: &AverageEffectQuery,
        graph: &Dag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (identifier_id, estimator_id) = self.resolve_static_ids();
        self.ensure_rd_config_present(estimator_id)?;
        compile_logical_static_ate(StaticAteCompileInput {
            data,
            graph,
            query: q,
            validation_suite: self.validation_suite_id(),
            identifier: identifier_id,
            estimator: estimator_id,
        })
    }

    pub(super) fn compile_logical_distribution_dag(
        &self,
        data: &TabularData,
        q: &InterventionalDistributionQuery,
        graph: &Dag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (identifier, estimator) = self.resolve_distribution_ids();
        compile_logical_distribution(StaticDistributionCompileInput {
            data,
            graph,
            query: q,
            validation_suite: self.validation_suite_id(),
            identifier,
            estimator,
        })
    }

    pub(super) fn compile_logical_distribution_mixed(
        &self,
        data: &TabularData,
        q: &InterventionalDistributionQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        super::super::static_path::ensure_admg_distribution_supported(q, self.settings.refute)?;
        let (identifier, estimator) = wire_pair(self.resolve_distribution_ids());
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
    pub(super) fn compile_logical_path_specific_mixed(
        &self,
        data: &TabularData,
        source: PathSpecificMixedSource<'_>,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        if !matches!(source, PathSpecificMixedSource::Mediation(_)) {
            super::super::static_path::ensure_admg_path_specific_supported(self.settings.refute)?;
        }
        let admg = mixed_admg(self)?;
        let (identifier, estimator) = self.resolve_path_ids();
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
    pub(super) fn compile_logical_path_specific_cpdag_cell(
        &self,
        data: &TabularData,
        source: PathSpecificMixedSource<'_>,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        if !matches!(source, PathSpecificMixedSource::Mediation(_)) {
            super::super::static_path::ensure_admg_path_specific_supported(self.settings.refute)?;
        }
        let GraphRef::Cpdag(cpdag) = self.graph.graph_ref() else {
            return Err(CausalError::Compile {
                message: "class path-specific effect requires a Cpdag".into(),
            });
        };
        let (identifier, estimator) = self.resolve_path_ids();
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
    pub(super) fn compile_logical_path_specific_pag_cell(
        &self,
        data: &TabularData,
        source: PathSpecificMixedSource<'_>,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        if !matches!(source, PathSpecificMixedSource::Mediation(_)) {
            super::super::static_path::ensure_admg_path_specific_supported(self.settings.refute)?;
        }
        let GraphRef::Pag(pag) = self.graph.graph_ref() else {
            return Err(CausalError::Compile {
                message: "PAG path-specific effect requires a Pag".into(),
            });
        };
        let (identifier, estimator) = self.resolve_path_ids();
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

    pub(super) fn compile_logical_path_specific_dag(
        &self,
        data: &TabularData,
        q: &PathSpecificEffectQuery,
        graph: &Dag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (identifier, estimator) = self.resolve_path_ids();
        compile_logical_path_specific(StaticPathSpecificCompileInput {
            data,
            graph,
            query: q,
            validation_suite: self.validation_suite_id(),
            identifier,
            estimator,
        })
    }

    pub(super) fn compile_logical_ate_pag(
        &self,
        data: &TabularData,
        q: &AverageEffectQuery,
        pag: &Pag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (identifier_id, estimator_id) = self.resolve_pag_ids();
        reject_dag_only_on_pag(&self.graph, identifier_id)?;
        compile_logical_static_pag_ate(StaticPagAteCompileInput {
            data,
            graph: pag,
            query: q,
            validation_suite: self.validation_suite_id(),
            identifier: identifier_id,
            estimator: estimator_id,
        })
    }

    pub(super) fn compile_logical_ate_cpdag(
        &self,
        data: &TabularData,
        q: &AverageEffectQuery,
        cpdag: &Cpdag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (identifier, estimator) = self.resolve_pag_ids();
        compile_logical_static_cpdag_ate(crate::planner::StaticCpdagAteCompileInput {
            data,
            graph: cpdag,
            query: q,
            validation_suite: self.validation_suite_id(),
            identifier,
            estimator,
        })
    }

    pub(super) fn compile_logical_conditional_dag(
        &self,
        data: &TabularData,
        q: &ConditionalEffectQuery,
        graph: &Dag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (identifier, estimator) = self.resolve_conditional_ids();
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

    pub(super) fn compile_logical_conditional_cpdag(
        &self,
        data: &TabularData,
        q: &ConditionalEffectQuery,
        cpdag: &Cpdag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (identifier, estimator) = self.resolve_class_conditional_ids();
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

    pub(super) fn compile_logical_conditional_pag(
        &self,
        data: &TabularData,
        q: &ConditionalEffectQuery,
        pag: &Pag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (identifier, estimator) = self.resolve_class_conditional_ids();
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
    pub(super) fn compile_logical_conditional_mixed(
        &self,
        data: &TabularData,
        q: &ConditionalEffectQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        if let GraphRef::TemporalAdmg(temporal) = self.graph.graph_ref() {
            // The contemporaneous projection must compile (a mean contrast, identified by IDC);
            // the plan below is then the projection's.
            if self.settings.inference.is_bayesian() {
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
    pub(super) fn compile_physical_conditional_mixed(
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
}
