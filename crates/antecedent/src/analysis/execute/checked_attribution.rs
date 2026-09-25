//! Builder-independent execution for frequentist DAG attribution operations.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[derive(Clone, Debug)]
enum AttributionTarget {
    Anomaly(antecedent_core::AnomalyAttributionQuery),
    Change(antecedent_core::ChangeAttributionQuery),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum AttributionProcedure {
    Point,
    SharedDirichletRowWeights { draws: usize },
}

/// A frozen GCM attribution target and its prepare-time identification contract.
#[derive(Clone)]
pub(crate) struct CheckedAttributionOperation {
    graph: Dag,
    target: AttributionTarget,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    inference: InferenceMode,
    procedure: AttributionProcedure,
    estimator: EstimatorId,
    physical: PhysicalExecutionPlan,
    result_context: IdentifiedResultContext,
}

impl std::fmt::Debug for CheckedAttributionOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckedAttributionOperation")
            .field("target", &self.target)
            .field("estimand", &self.estimand)
            .field("identification_status", &self.identification.status)
            .field("identifier", &IdentifierId::GcmParametric)
            .field("inference", &self.inference)
            .field("procedure", &self.procedure)
            .field("estimator", &self.estimator)
            .finish_non_exhaustive()
    }
}

impl CheckedAttributionOperation {
    pub(crate) fn checked(
        study: &Study,
        physical: &PhysicalExecutionPlan,
        cache: &crate::analysis::prepared::CachedStaticIdentification,
    ) -> Result<Self, CausalError> {
        let graph = study.graph.as_dag().ok_or(CausalError::Unsupported {
            message: "checked attribution requires a supplied DAG",
        })?;
        let target = match &study.query {
            CausalQuery::AnomalyAttribution(query) => {
                query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
                AttributionTarget::Anomaly(query.clone())
            }
            CausalQuery::ChangeAttribution(query) => {
                query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
                AttributionTarget::Change(query.clone())
            }
            _ => {
                return Err(CausalError::Compile {
                    message: "checked attribution supports only anomaly or change targets".into(),
                });
            }
        };
        let target_query = target.query();
        if !study.custom_validators.is_empty() {
            return Err(CausalError::Unsupported {
                message: "checked attribution does not support custom validators",
            });
        }
        let (estimator, procedure) = match (&target, &study.inference) {
            (_, InferenceMode::Frequentist) => (EstimatorId::GcmFit, AttributionProcedure::Point),
            (AttributionTarget::Anomaly(_), InferenceMode::Bayesian(config)) => {
                if config.prior.is_some()
                    || config.prior_artifact.is_some()
                    || config.external_compose.is_some()
                {
                    return Err(CausalError::Unsupported {
                        message: "Bayesian attribution uses its shared-row Bayesian bootstrap and does not accept coefficient priors or external prior composition",
                    });
                }
                (
                    EstimatorId::GcmFitBayesian,
                    AttributionProcedure::SharedDirichletRowWeights {
                        draws: bayesian_draw_count(&study.inference)?,
                    },
                )
            }
            (AttributionTarget::Change(_), InferenceMode::Bayesian(config)) => {
                if config.prior.is_some()
                    || config.prior_artifact.is_some()
                    || config.external_compose.is_some()
                {
                    return Err(CausalError::Unsupported {
                        message: "Bayesian attribution uses its shared-row Bayesian bootstrap and does not accept coefficient priors or external prior composition",
                    });
                }
                (
                    EstimatorId::GcmAttributionBayesian,
                    AttributionProcedure::SharedDirichletRowWeights {
                        draws: bayesian_draw_count(&study.inference)?,
                    },
                )
            }
        };
        if study.structure_source != crate::support::StructureSource::Explicit
            || cache.identification.query != target_query
            || physical.logical.query != target_query
            || physical.logical.record.identifier.as_deref()
                != Some(IdentifierId::GcmParametric.as_str())
            || physical.logical.record.estimator.as_deref() != Some(estimator.as_str())
        {
            return Err(CausalError::Compile {
                message: "attribution target, source graph, identification, or estimator does not match the checked operation".into(),
            });
        }
        let estimand = cache.estimand.clone();
        if !cache.identification.estimands.iter().any(|candidate| {
            candidate.method == estimand.method && candidate.functional == estimand.functional
        }) {
            return Err(CausalError::Compile {
                message: "attribution estimand is absent from its identification result".into(),
            });
        }
        Ok(Self {
            graph: graph.clone(),
            target,
            identification: cache.identification.clone(),
            estimand,
            inference: study.inference.clone(),
            procedure,
            estimator,
            physical: physical.clone(),
            result_context: IdentifiedResultContext::from_study(study),
        })
    }

    pub(crate) fn query(&self) -> CausalQuery {
        self.target.query()
    }

    pub(crate) fn graph(&self) -> &Dag {
        &self.graph
    }

    pub(crate) fn identification(&self) -> &IdentificationResult {
        &self.identification
    }

    pub(crate) fn procedure(&self) -> (&'static str, &'static str) {
        (IdentifierId::GcmParametric.as_str(), self.estimator.as_str())
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if self.result_context.query != self.query()
            || self.identification.query != self.query()
            || !self.identification.estimands.iter().any(|candidate| {
                candidate.method == self.estimand.method
                    && candidate.functional == self.estimand.functional
            })
        {
            return Err(CausalError::Compile {
                message: "attribution operation lost its checked target or identification binding"
                    .into(),
            });
        }
        match &self.target {
            AttributionTarget::Anomaly(query) => {
                let started = Instant::now();
                let fitted = fit_gcm(self.graph.clone(), data)?;
                let scores = anomaly_attribution_with(
                    &fitted.model,
                    data,
                    query.targets.iter().copied(),
                    query.max_units,
                    ctx,
                )?;
                let outcome = *query.targets.first().ok_or_else(|| CausalError::Compile {
                    message: "checked anomaly target has no outcome variable".into(),
                })?;
                let posterior =
                    self.anomaly_posterior(query, data, &fitted, &scores, outcome, ctx)?;
                let mut extra_diagnostics = Vec::new();
                if posterior.is_some() {
                    extra_diagnostics.push(bayesian_attribution_diagnostic("Each draw refits all mechanisms under one shared Dirichlet(1,…,1) row-weight vector while holding observed marginal reference statistics fixed."));
                }
                Ok(finish_identified_execute_with_context(
                    &self.result_context,
                    Some(data),
                    IdentifiedExecuteFinish {
                        physical: &self.physical,
                        identification: self.identification.clone(),
                        estimand: self.estimand.clone(),
                        estimate: nan_effect(),
                        identifier_id: IdentifierId::GcmParametric,
                        estimator_id: self.estimator,
                        treatment: outcome,
                        outcome,
                        identify_cached: true,
                        extra_diagnostics,
                        refutations: Vec::new(),
                        distribution: None,
                        mediation: None,
                        wall_time_ns: u64::try_from(started.elapsed().as_nanos())
                            .unwrap_or(u64::MAX),
                        bootstrap_replicates_ok: None,
                        cancelled: false,
                        early_stopped: false,
                        extras: IdentifiedExecuteExtras {
                            posterior,
                            n_draws: match self.procedure {
                                AttributionProcedure::SharedDirichletRowWeights { draws } => {
                                    Some(u32::try_from(draws).unwrap_or(u32::MAX))
                                }
                                AttributionProcedure::Point => None,
                            },
                            gcm: Some(GcmSlot::Anomaly(scores)),
                            empty_provenance: true,
                            ..Default::default()
                        },
                    },
                ))
            }
            AttributionTarget::Change(query) => {
                let started = Instant::now();
                let fitted = fit_gcm(self.graph.clone(), data)?;
                let change = attribute_distribution_change(
                    &fitted.model,
                    data,
                    query,
                    &antecedent_attribution::DistributionChangeOptions::default(),
                    ctx,
                )?;
                let mut estimate = EffectEstimate::new(
                    change.total_change,
                    f64::NAN,
                    antecedent_core::AssumptionSet::default(),
                    OverlapPolicy::ExplicitOverride,
                );
                let posterior = self.change_posterior(query, data, &fitted.model, ctx)?;
                if let Some(posterior) = &posterior {
                    if let Some(index) = posterior.effect_column() {
                        estimate.ate = posterior.summaries.mean[index];
                        estimate.se_analytic = posterior.summaries.sd[index];
                    }
                }
                let mut extra_diagnostics = Vec::new();
                if posterior.is_some() {
                    extra_diagnostics.push(bayesian_attribution_diagnostic("Each population has one Dirichlet(1,…,1) row-weight vector per draw shared across mechanisms; total and Shapley components use the same weighted attribution."));
                }
                Ok(finish_identified_execute_with_context(
                    &self.result_context,
                    Some(data),
                    IdentifiedExecuteFinish {
                        physical: &self.physical,
                        identification: self.identification.clone(),
                        estimand: self.estimand.clone(),
                        estimate,
                        identifier_id: IdentifierId::GcmParametric,
                        estimator_id: self.estimator,
                        treatment: query.outcome,
                        outcome: query.outcome,
                        identify_cached: true,
                        extra_diagnostics,
                        refutations: Vec::new(),
                        distribution: None,
                        mediation: None,
                        wall_time_ns: u64::try_from(started.elapsed().as_nanos())
                            .unwrap_or(u64::MAX),
                        bootstrap_replicates_ok: None,
                        cancelled: false,
                        early_stopped: false,
                        extras: IdentifiedExecuteExtras {
                            posterior,
                            n_draws: match self.procedure {
                                AttributionProcedure::SharedDirichletRowWeights { draws } => {
                                    Some(u32::try_from(draws).unwrap_or(u32::MAX))
                                }
                                AttributionProcedure::Point => None,
                            },
                            gcm: Some(GcmSlot::Change(change)),
                            empty_provenance: true,
                            ..Default::default()
                        },
                    },
                ))
            }
        }
    }

    fn anomaly_posterior(
        &self,
        query: &antecedent_core::AnomalyAttributionQuery,
        data: &TabularData,
        fitted: &crate::gcm::FittedGcm,
        observed_scores: &[antecedent_attribution::AnomalyScores],
        outcome: VariableId,
        ctx: &ExecutionContext,
    ) -> Result<Option<CausalPosterior>, CausalError> {
        let AttributionProcedure::SharedDirichletRowWeights { draws } = self.procedure else {
            return Ok(None);
        };
        let mut columns = observed_scores
            .iter()
            .map(|score| {
                (format!("mean_anomaly_score[{}]", score.target.raw()), Vec::with_capacity(draws))
            })
            .collect::<Vec<_>>();
        let registry = crate::gcm::MechanismRegistry::standard();
        let mut rng = ctx.rng.stream_for(antecedent_core::StreamDomain::Attribution, 0xA110_7A7E);
        for _ in 0..draws {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: super::super::stage::STAGE_ESTIMATE_POINT,
                });
            }
            let weights =
                super::attribution_path::dirichlet_row_weights(data.row_count(), &mut rng);
            let store = registry
                .refit_weighted(&fitted.model, data, &fitted.assignments, &weights)
                .map_err(|e| CausalError::Compile { message: e.to_string() })?;
            let draw_scores = anomaly_attribution_with(
                &fitted.model.clone().with_mechanisms(store),
                data,
                query.targets.iter().copied(),
                query.max_units,
                ctx,
            )?;
            if draw_scores.len() != columns.len() {
                return Err(CausalError::Compile {
                    message: "Bayesian anomaly draw changed target set".into(),
                });
            }
            for (column, score) in columns.iter_mut().zip(draw_scores) {
                column.1.push(if score.scores.is_empty() {
                    f64::NAN
                } else {
                    score.scores.iter().sum::<f64>() / score.scores.len() as f64
                });
            }
        }
        let (identification, _) = parametric_scm_identification(
            CausalQuery::AnomalyAttribution(query.clone()),
            outcome,
            outcome,
        );
        Ok(Some(super::attribution_path::attribution_posterior(
            columns,
            identification.required_assumptions,
            identification.status,
            "gcm.attribution.shared_dirichlet_row_weights",
            false,
        )?))
    }

    fn change_posterior(
        &self,
        query: &antecedent_core::ChangeAttributionQuery,
        data: &TabularData,
        model: &crate::gcm::CompiledCausalModel,
        ctx: &ExecutionContext,
    ) -> Result<Option<CausalPosterior>, CausalError> {
        let AttributionProcedure::SharedDirichletRowWeights { draws } = self.procedure else {
            return Ok(None);
        };
        let baseline_n = antecedent_attribution::resolve_rows(data, &query.baseline)
            .map_err(|e| CausalError::Compile { message: e.to_string() })?
            .len();
        let comparison_n = antecedent_attribution::resolve_rows(data, &query.comparison)
            .map_err(|e| CausalError::Compile { message: e.to_string() })?
            .len();
        let options = antecedent_attribution::DistributionChangeOptions::default();
        let mut rng = ctx.rng.stream_for(antecedent_core::StreamDomain::Attribution, 0xC4A6_7A7E);
        let mut totals = Vec::with_capacity(draws);
        let mut components: Option<Vec<(String, Vec<f64>)>> = None;
        for _ in 0..draws {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: super::super::stage::STAGE_ESTIMATE_POINT,
                });
            }
            let baseline_weights =
                super::attribution_path::dirichlet_row_weights(baseline_n, &mut rng);
            let comparison_weights =
                super::attribution_path::dirichlet_row_weights(comparison_n, &mut rng);
            let draw = antecedent_attribution::distribution_change_with_row_weights(
                model,
                data,
                query,
                &options,
                &baseline_weights,
                &comparison_weights,
                ctx,
            )
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
            totals.push(draw.total_change);
            let columns = components.get_or_insert_with(|| {
                draw.contributions
                    .iter()
                    .map(|c| {
                        (
                            format!("component[{}]", c.component.variable().raw()),
                            Vec::with_capacity(draws),
                        )
                    })
                    .collect()
            });
            if draw.contributions.len() != columns.len() {
                return Err(CausalError::Compile {
                    message: "Bayesian change attribution draw changed component set".into(),
                });
            }
            for (column, component) in columns.iter_mut().zip(draw.contributions.iter()) {
                column.1.push(component.contribution);
            }
        }
        let mut posterior_columns = vec![("total_change".to_owned(), totals)];
        posterior_columns.extend(components.unwrap_or_default());
        let (identification, _) = parametric_scm_identification(
            CausalQuery::ChangeAttribution(query.clone()),
            query.outcome,
            query.outcome,
        );
        Ok(Some(super::attribution_path::attribution_posterior(
            posterior_columns,
            identification.required_assumptions,
            identification.status,
            "gcm.attribution.shared_population_dirichlet_row_weights",
            true,
        )?))
    }
}

fn bayesian_attribution_diagnostic(detail: &str) -> Diagnostic {
    Diagnostic::new(
        "gcm.attribution.bayesian",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "Posterior is conditional on mechanism-family selection from the original unweighted data. {detail} It does not address model misspecification or causal identification."
        ),
    )
}

impl AttributionTarget {
    fn query(&self) -> CausalQuery {
        match self {
            Self::Anomaly(query) => CausalQuery::AnomalyAttribution(query.clone()),
            Self::Change(query) => CausalQuery::ChangeAttribution(query.clone()),
        }
    }
}

#[cfg(test)]
mod bayesian_checked_attribution_tests {
    use super::*;
    use antecedent_core::{AnomalyAttributionQuery, CausalQuery, PopulationSelector, VariableId};
    use antecedent_data::TabularData;
    use antecedent_graph::DenseNodeId;

    fn context() -> ExecutionContext {
        ExecutionContext::for_tests(0xBA71_5EED)
    }

    fn anomaly_fixture() -> (TabularData, Dag) {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> =
            (0..20).map(|i| if i == 19 { 200.0 } else { 2.0 * f64::from(i) }).collect();
        let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())])
            .expect("fixture data");
        let mut graph = Dag::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        (data, graph)
    }

    fn change_fixture() -> (TabularData, Dag) {
        let x: Vec<f64> = (0..80).map(|i| f64::from(i % 40) * 0.1).collect();
        let y: Vec<f64> = (0..80)
            .map(|i| {
                let x = f64::from(i % 40) * 0.1;
                if i < 40 { 1.0 + 2.0 * x } else { 6.0 + 2.0 * x }
            })
            .collect();
        let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())])
            .expect("fixture data");
        let mut graph = Dag::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        (data, graph)
    }

    fn checked_operation(
        data: &TabularData,
        graph: Dag,
        query: CausalQuery,
        estimator: EstimatorId,
    ) -> CheckedAttributionOperation {
        let ctx = context();
        let builder = crate::Study::tabular(data.clone())
            .graph(graph)
            .query(query.clone())
            .inference(InferenceMode::Bayesian(crate::BayesianConfig::conjugate().n_draws(24)))
            .refute(RefuteSuite::None);
        let study = builder.build().expect("study build");
        let physical = study.plan(&ctx).expect("physical plan");
        let outcome = match &query {
            CausalQuery::AnomalyAttribution(query) => *query.targets.first().unwrap(),
            CausalQuery::ChangeAttribution(query) => query.outcome,
            _ => unreachable!(),
        };
        let (identification, estimand) =
            parametric_scm_identification(query.clone(), outcome, outcome);
        let cache =
            crate::analysis::prepared::CachedStaticIdentification { identification, estimand };
        let operation = CheckedAttributionOperation::checked(&study, &physical, &cache)
            .expect("Bayesian attribution plan");
        assert_eq!(operation.procedure(), ("gcm.parametric", estimator.as_str()));
        drop(study);
        operation
    }

    #[test]
    fn anomaly_posterior_executes_from_retained_operation_after_builder_drop() {
        let (data, graph) = anomaly_fixture();
        let query = CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new(
            [VariableId::from_raw(1)],
            100,
        ));
        let operation = checked_operation(&data, graph, query, EstimatorId::GcmFitBayesian);
        let result = operation.execute(&data, &context()).expect("retained execution");
        let posterior = result.posterior.as_ref().expect("shared-row posterior");
        assert_eq!(posterior.draws.n_draws, 24);
        assert_eq!(posterior.draws.schema.n_quantities(), 1);
        assert!(posterior.summaries.mean[0].is_finite() && posterior.summaries.mean[0] > 0.0);
        let scores = result.anomaly.as_ref().expect("anomaly scores");
        let target = scores.iter().find(|scores| scores.target == VariableId::from_raw(1)).unwrap();
        let top_row = target.scores.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        assert_eq!(target.rows[top_row], 19, "known outlier remains the highest-scoring row");
        assert!(result.diagnostics.iter().any(|d| d.code.as_ref() == "gcm.attribution.bayesian"));
    }

    #[test]
    fn change_posterior_keeps_total_and_shapley_components_coupled() {
        let (data, graph) = change_fixture();
        let query = CausalQuery::ChangeAttribution(antecedent_core::ChangeAttributionQuery::new(
            VariableId::from_raw(1),
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
        ));
        let operation = checked_operation(&data, graph, query, EstimatorId::GcmAttributionBayesian);
        let result = operation.execute(&data, &context()).expect("retained execution");
        let posterior = result.posterior.as_ref().expect("shared-population posterior");
        assert_eq!(posterior.draws.n_draws, 24);
        assert!(posterior.summaries.mean[0].is_finite());
        let truth: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../conformance/estimate/staged_attribution/expected.json"
        ))
        .unwrap();
        let expected = truth["change"]["total_change"].as_f64().unwrap();
        let tolerance = truth["change"]["tolerance"].as_f64().unwrap();
        assert!((posterior.summaries.mean[0] - expected).abs() < tolerance);
        for draw in 0..posterior.draws.n_draws {
            let total = posterior.draws.get(draw, 0).unwrap();
            let sum = (1..posterior.draws.schema.n_quantities())
                .map(|quantity| posterior.draws.get(draw, quantity).unwrap())
                .sum::<f64>();
            assert!((sum - total).abs() < 1e-8, "Shapley components must sum to total per draw");
        }
        assert!(result.diagnostics.iter().any(|d| d.code.as_ref() == "gcm.attribution.bayesian"));
    }
}
