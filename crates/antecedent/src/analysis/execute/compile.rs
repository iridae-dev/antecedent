// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

impl super::CausalAnalysis {
    /// Compile logical plan only (inspectable semantics).
    ///
    /// # Errors
    ///
    /// Modality / query validation failures. Does not run discovery.
    pub fn compile_logical(&self) -> Result<LogicalAnalysisPlan, CausalError> {
        self.ensure_supported_combination()?;
        match (&self.data, &self.query, &self.graph) {
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::Static(graph),
            ) => {
                let (identifier, estimator) = self.resolve_static_pair();
                self.ensure_rd_config_present(&estimator)?;
                compile_logical_static_ate(StaticAteCompileInput {
                    data,
                    graph,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })
            }
            (DataInput::Tabular(data), CausalQuery::Distribution(q), GraphInput::Static(graph)) => {
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
            (DataInput::Tabular(data), CausalQuery::PathSpecific(q), GraphInput::Static(graph)) => {
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
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
                GraphInput::Temporal(graph),
            ) => {
                let class = match &self.data {
                    DataInput::Event(_) => DataClassification::Event,
                    _ => DataClassification::Temporal,
                };
                compile_logical_temporal_effect_classified(data, graph, q, self.split, false, class)
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverPcmci { .. }
                | GraphInput::DiscoverPcmciPlus { .. }
                | GraphInput::DiscoverRpcmci { .. }
                | GraphInput::DiscoverLpcmci { .. }
                | GraphInput::TemporalPag(_),
            ) => {
                let class = match &self.data {
                    DataInput::Event(_) => DataClassification::Event,
                    _ => DataClassification::Temporal,
                };
                compile_logical_temporal_effect_classified(
                    data,
                    &TemporalDag::empty(),
                    q,
                    self.split,
                    true,
                    class,
                )
            }
            (
                DataInput::MultiEnv(multi),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverJpcmciPlus { .. },
            ) => {
                let data = multi.environment(0).map_err(|e| CausalError::Compile {
                    message: format!("jpcmci+ multi-env: {e}"),
                })?;
                compile_logical_temporal_effect_classified(
                    data,
                    &TemporalDag::empty(),
                    q,
                    self.split,
                    true,
                    DataClassification::MultiEnvironment,
                )
            }
            (
                DataInput::Panel(panel),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverJpcmciPlus { .. }
                | GraphInput::DiscoverPcmci { .. }
                | GraphInput::DiscoverPcmciPlus { .. }
                | GraphInput::DiscoverLpcmci { .. }
                | GraphInput::Temporal(_),
            ) => {
                let data = &panel
                    .unit(0)
                    .map_err(|e| CausalError::Compile { message: format!("panel: {e}") })?
                    .series;
                let review = matches!(
                    self.graph,
                    GraphInput::DiscoverJpcmciPlus { .. }
                        | GraphInput::DiscoverPcmci { .. }
                        | GraphInput::DiscoverPcmciPlus { .. }
                        | GraphInput::DiscoverLpcmci { .. }
                );
                compile_logical_temporal_effect_classified(
                    data,
                    &TemporalDag::empty(),
                    q,
                    self.split,
                    review,
                    DataClassification::Panel,
                )
            }
            (DataInput::Tabular(data), CausalQuery::AverageEffect(q), GraphInput::Pag(pag)) => {
                let (identifier, estimator) = self.resolve_pag_pair();
                reject_dag_only_on_pag(&self.graph, identifier.parse::<IdentifierId>()?)?;
                compile_logical_static_pag_ate(StaticPagAteCompileInput {
                    data,
                    pag,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })
            }
            (
                DataInput::Tabular(data),
                CausalQuery::ConditionalEffect(q),
                GraphInput::Static(graph),
            ) => {
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
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::Mediation(q),
                GraphInput::Temporal(graph),
            ) => {
                q.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
                let mut plan = compile_logical_temporal_effect(
                    data,
                    graph,
                    &TemporalEffectQuery::pulse(q.treatment, q.outcome, 1.0),
                    self.split,
                    false,
                )?;
                plan.record.plan_id = Arc::from("temporal_mediation");
                plan.record.identifier = Some(Arc::from("temporal.mediation"));
                plan.record.estimator = Some(Arc::from("temporal.mediation"));
                plan.record.query_variables = Arc::from([q.treatment, q.outcome]);
                plan.query = CausalQuery::Mediation(q.clone());
                Ok(plan)
            }
            (DataInput::Tabular(data), CausalQuery::Mediation(q), GraphInput::Static(graph)) => {
                q.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
                if !matches!(q.contrast, MediationContrast::Total) {
                    return Err(CausalError::Unsupported {
                        message: "static Mediation natural/direct/mediated contrasts require \
                             temporal data + TemporalDag; only MediationContrast::Total \
                             (front-door) is supported on a static DAG",
                    });
                }
                let ate = AverageEffectQuery::binary_ate(q.treatment, q.outcome);
                let mut plan = compile_logical_static_ate(StaticAteCompileInput {
                    data,
                    graph,
                    query: &ate,
                    validation_suite: self.validation_suite_id(),
                    identifier: Arc::from("frontdoor"),
                    estimator: Arc::from("frontdoor.two_stage"),
                })?;
                plan.record.plan_id = Arc::from("static_mediation_total");
                plan.query = CausalQuery::Mediation(q.clone());
                Ok(plan)
            }
            (
                DataInput::Tabular(data),
                CausalQuery::Counterfactual(_)
                | CausalQuery::AnomalyAttribution(_)
                | CausalQuery::ChangeAttribution(_)
                | CausalQuery::MechanismChange(_)
                | CausalQuery::UnitChange(_),
                GraphInput::Static(_),
            ) => {
                // Parametric SCM paths: logical metadata only (no classic identifier/estimator).
                let (treatment, outcome) = gcm_query_vars(&self.query)?;
                self.query
                    .validate()
                    .map_err(|e| CausalError::Compile { message: e.to_string() })?;
                Ok(LogicalAnalysisPlan {
                    record: antecedent_core::LogicalAnalysisPlanRecord {
                        plan_id: Arc::from("gcm_query"),
                        data_classification: antecedent_core::DataClassification::Tabular,
                        discovery_algorithm: None,
                        graph_review_required: false,
                        identifier: Some(Arc::from("gcm.parametric")),
                        estimator: Some(Arc::from("gcm.fit")),
                        validation_suite: self.validation_suite_id(),
                        query_variables: Arc::from([treatment, outcome]),
                    },
                    query: self.query.clone(),
                    split: None,
                    row_count_hint: data.row_count() as u64,
                })
            }
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::DiscoverPc { .. },
            ) => self.compile_discover_static_ate(data, q, "pc", None),
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::DiscoverGes { .. },
            ) => self.compile_discover_static_ate(data, q, "ges", None),
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::DiscoverLingam { .. },
            ) => self.compile_discover_static_ate(data, q, "direct_lingam", None),
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::DiscoverNotears { .. },
            ) => self.compile_discover_static_ate(data, q, "notears", None),
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                graph @ GraphInput::DiscoverFci { .. },
            ) => self.compile_discover_static_ate(data, q, "fci", Some(graph)),
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                graph @ GraphInput::DiscoverRfci { .. },
            ) => self.compile_discover_static_ate(data, q, "rfci", Some(graph)),
            _ => Err(CausalError::Unsupported {
                message: "unsupported data/graph/query combination",
            }),
        }
    }

    /// Empty-DAG logical plan for static Discover* → ATE (review required).
    pub(super) fn compile_discover_static_ate(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        algorithm: &str,
        pag_graph: Option<&GraphInput>,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let (identifier, estimator) = self.resolve_static_pair();
        if let Some(graph) = pag_graph {
            reject_dag_only_on_pag(graph, identifier.parse::<IdentifierId>()?)?;
        }
        let n_vars = u32::try_from(data.schema().len()).unwrap_or(0);
        let empty = Dag::with_variables(n_vars);
        let mut plan = compile_logical_static_ate(StaticAteCompileInput {
            data,
            graph: &empty,
            query,
            validation_suite: self.validation_suite_id(),
            identifier,
            estimator,
        })?;
        plan.record.discovery_algorithm = Some(Arc::from(algorithm));
        plan.record.graph_review_required = true;
        Ok(plan)
    }

    /// Compile logical → physical plan (or review-required).
    ///
    /// # Errors
    ///
    /// Modality / resource / discovery failures.
    pub fn compile(&self, ctx: &ExecutionContext) -> Result<CompiledAnalysis, CausalError> {
        self.ensure_supported_combination()?;
        match (&self.data, &self.query, &self.graph) {
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::Static(graph),
            ) => {
                let (identifier, estimator) = self.resolve_static_pair();
                self.ensure_rd_config_present(&estimator)?;
                let logical = compile_logical_static_ate(StaticAteCompileInput {
                    data,
                    graph,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })?;
                let physical = logical.compile_physical(ctx)?;
                Ok(CompiledAnalysis::Ready(physical))
            }
            (DataInput::Tabular(data), CausalQuery::Distribution(q), GraphInput::Static(graph)) => {
                let (identifier, estimator) = self.resolve_distribution_pair();
                let logical = compile_logical_distribution(StaticDistributionCompileInput {
                    data,
                    graph,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })?;
                let physical = logical.compile_physical(ctx)?;
                Ok(CompiledAnalysis::Ready(physical))
            }
            (DataInput::Tabular(data), CausalQuery::PathSpecific(q), GraphInput::Static(graph)) => {
                let (identifier, estimator) = self.resolve_path_pair();
                let logical = compile_logical_path_specific(StaticPathSpecificCompileInput {
                    data,
                    graph,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })?;
                let physical = logical.compile_physical(ctx)?;
                Ok(CompiledAnalysis::Ready(physical))
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
                GraphInput::Temporal(graph),
            ) => {
                let class = match &self.data {
                    DataInput::Event(_) => DataClassification::Event,
                    _ => DataClassification::Temporal,
                };
                let logical = compile_logical_temporal_effect_classified(
                    data, graph, q, self.split, false, class,
                )?;
                ensure_review_complete(&logical)?;
                let physical = logical.compile_physical_with_graph(ctx, Some(graph.clone()))?;
                Ok(CompiledAnalysis::Ready(physical))
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverPcmci { max_lag, alpha, max_cond_size, fdr, accept_discovered },
            ) => {
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let review =
                    run_pcmci_review(data, *max_lag, *alpha, *max_cond_size, *fdr, ci, ctx)?;
                if *accept_discovered {
                    PendingGraphReview::new(review, data.row_count(), q.clone(), self.split)
                        .accept_all()
                        .finish(data, ctx)
                } else {
                    Ok(compile_review_required(review))
                }
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverPcmciPlus {
                    max_lag,
                    alpha,
                    max_cond_size,
                    fdr,
                    accept_discovered,
                },
            ) => {
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let review =
                    run_pcmci_plus_review(data, *max_lag, *alpha, *max_cond_size, *fdr, ci, ctx)?;
                if *accept_discovered && review.pending_undirected.is_empty() {
                    PendingCpdagReview::new(review, data.row_count(), q.clone(), self.split)
                        .accept_all_directed()
                        .finish(data, ctx)
                } else {
                    Ok(compile_review_required_cpdag(review))
                }
            }
            (
                DataInput::MultiEnv(multi),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverJpcmciPlus {
                    max_lag,
                    alpha,
                    max_cond_size,
                    fdr,
                    accept_discovered,
                    multi_dataset,
                },
            ) => {
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let review = run_jpcmci_plus_review(
                    multi,
                    *max_lag,
                    *alpha,
                    *max_cond_size,
                    *fdr,
                    multi_dataset,
                    ci,
                    ctx,
                )?;
                let data = multi.environment(0).map_err(|e| CausalError::Compile {
                    message: format!("jpcmci+ multi-env: {e}"),
                })?;
                if *accept_discovered && review.pending_undirected.is_empty() {
                    PendingCpdagReview::new(review, data.row_count(), q.clone(), self.split)
                        .accept_all_directed()
                        .finish(data, ctx)
                } else {
                    Ok(compile_review_required_cpdag(review))
                }
            }
            (
                DataInput::Panel(panel),
                CausalQuery::TemporalEffect(q),
                GraphInput::Temporal(graph),
            ) => {
                let data = &panel
                    .unit(0)
                    .map_err(|e| CausalError::Compile { message: format!("panel: {e}") })?
                    .series;
                let logical = compile_logical_temporal_effect_classified(
                    data,
                    graph,
                    q,
                    self.split,
                    false,
                    DataClassification::Panel,
                )?;
                ensure_review_complete(&logical)?;
                let physical = logical.compile_physical_with_graph(ctx, Some(graph.clone()))?;
                Ok(CompiledAnalysis::Ready(physical))
            }
            (
                DataInput::Panel(panel),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverJpcmciPlus {
                    max_lag,
                    alpha,
                    max_cond_size,
                    fdr,
                    accept_discovered,
                    multi_dataset,
                },
            ) => {
                let multi = panel.as_multi_env().map_err(|e| CausalError::Compile {
                    message: format!("panel as multi-env: {e}"),
                })?;
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let review = run_jpcmci_plus_review(
                    &multi,
                    *max_lag,
                    *alpha,
                    *max_cond_size,
                    *fdr,
                    multi_dataset,
                    ci,
                    ctx,
                )?;
                let data = &panel
                    .unit(0)
                    .map_err(|e| CausalError::Compile { message: format!("panel: {e}") })?
                    .series;
                if *accept_discovered && review.pending_undirected.is_empty() {
                    let compiled =
                        PendingCpdagReview::new(review, data.row_count(), q.clone(), self.split)
                            .accept_all_directed()
                            .finish(data, ctx)?;
                    Ok(mark_panel_classification(compiled))
                } else {
                    Ok(compile_review_required_cpdag(review))
                }
            }
            (
                DataInput::Panel(panel),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverPcmci { max_lag, alpha, max_cond_size, fdr, accept_discovered },
            ) => {
                let pooled = stack_panel_tabular(panel).map_err(CausalError::from)?;
                let n = pooled.row_count();
                let series = TimeSeriesData::try_new(
                    pooled.storage().clone(),
                    antecedent_data::TimeIndex {
                        regularity: antecedent_data::SamplingRegularity::Regular { interval_ns: 1 },
                        length: n,
                    },
                )
                .map_err(CausalError::from)?;
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let review =
                    run_pcmci_review(&series, *max_lag, *alpha, *max_cond_size, *fdr, ci, ctx)?;
                let data = &panel
                    .unit(0)
                    .map_err(|e| CausalError::Compile { message: format!("panel: {e}") })?
                    .series;
                if *accept_discovered {
                    let compiled =
                        PendingGraphReview::new(review, data.row_count(), q.clone(), self.split)
                            .accept_all()
                            .finish(data, ctx)?;
                    Ok(mark_panel_classification(compiled))
                } else {
                    Ok(compile_review_required(review))
                }
            }
            (
                DataInput::Panel(panel),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverPcmciPlus {
                    max_lag,
                    alpha,
                    max_cond_size,
                    fdr,
                    accept_discovered,
                },
            ) => {
                let pooled = stack_panel_tabular(panel).map_err(CausalError::from)?;
                let n = pooled.row_count();
                let series = TimeSeriesData::try_new(
                    pooled.storage().clone(),
                    antecedent_data::TimeIndex {
                        regularity: antecedent_data::SamplingRegularity::Regular { interval_ns: 1 },
                        length: n,
                    },
                )
                .map_err(CausalError::from)?;
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let review = run_pcmci_plus_review(
                    &series,
                    *max_lag,
                    *alpha,
                    *max_cond_size,
                    *fdr,
                    ci,
                    ctx,
                )?;
                let data = &panel
                    .unit(0)
                    .map_err(|e| CausalError::Compile { message: format!("panel: {e}") })?
                    .series;
                if *accept_discovered && review.pending_undirected.is_empty() {
                    let compiled =
                        PendingCpdagReview::new(review, data.row_count(), q.clone(), self.split)
                            .accept_all_directed()
                            .finish(data, ctx)?;
                    Ok(mark_panel_classification(compiled))
                } else {
                    Ok(compile_review_required_cpdag(review))
                }
            }
            (
                DataInput::Panel(panel),
                CausalQuery::TemporalEffect(_q),
                GraphInput::DiscoverLpcmci {
                    max_lag,
                    alpha,
                    max_cond_size,
                    fdr,
                    accept_discovered: _,
                },
            ) => {
                let pooled = stack_panel_tabular(panel).map_err(CausalError::from)?;
                let n = pooled.row_count();
                let series = TimeSeriesData::try_new(
                    pooled.storage().clone(),
                    antecedent_data::TimeIndex {
                        regularity: antecedent_data::SamplingRegularity::Regular { interval_ns: 1 },
                        length: n,
                    },
                )
                .map_err(CausalError::from)?;
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let review =
                    run_lpcmci_review(&series, *max_lag, *alpha, *max_cond_size, *fdr, ci, ctx)?;
                Ok(compile_review_required_pag(review))
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(_q),
                GraphInput::DiscoverRpcmci {
                    max_lag,
                    alpha,
                    max_cond_size,
                    fdr,
                    accept_discovered,
                    regime_assignment,
                },
            ) => {
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let result = run_rpcmci_discovery(
                    data,
                    *max_lag,
                    *alpha,
                    *max_cond_size,
                    *fdr,
                    regime_assignment,
                    ci,
                    ctx,
                )?;
                // Multi-regime estimation is not auto-wired; surface the first regime's CPDAG
                // for review. Auto-accept only when a single fully-oriented regime exists.
                let Some(first) = result.per_regime.first() else {
                    return Err(CausalError::Compile {
                        message: "RPCMCI returned no regime graphs".into(),
                    });
                };
                let review = first.review.clone();
                if *accept_discovered
                    && result.per_regime.len() == 1
                    && review.pending_undirected.is_empty()
                {
                    let q = match &self.query {
                        CausalQuery::TemporalEffect(q) => q.clone(),
                        _ => unreachable!(),
                    };
                    PendingCpdagReview::new(review, data.row_count(), q, self.split)
                        .accept_all_directed()
                        .finish(data, ctx)
                } else {
                    Ok(compile_review_required_cpdag(review))
                }
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverDbnPosterior { .. },
            ) => self.compile_dbn_posterior_temporal(data, q, ctx),
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverLpcmci {
                    max_lag,
                    alpha,
                    max_cond_size,
                    fdr,
                    accept_discovered,
                },
            ) => {
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let review =
                    run_lpcmci_review(data, *max_lag, *alpha, *max_cond_size, *fdr, ci, ctx)?;
                // Temporal backdoor is DAG-only. Auto-accept only when the PAG is already
                // fully definite-directed (no circle/ambiguous marks) — never invent orientations.
                if *accept_discovered && review.is_complete() {
                    match review.graph.try_into_temporal_dag() {
                        Ok(dag) => {
                            let mut logical =
                                compile_logical_temporal_effect(data, &dag, q, self.split, false)?;
                            // Completion→DAG (not class-aware temporal PAG ID).
                            logical.record.discovery_algorithm =
                                Some(Arc::from("lpcmci.pag_completed_to_dag"));
                            let physical = logical.compile_physical_with_graph(ctx, Some(dag))?;
                            Ok(CompiledAnalysis::Ready(physical))
                        }
                        Err(_) => Ok(compile_review_required_pag(review)),
                    }
                } else {
                    Ok(compile_review_required_pag(review))
                }
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
                GraphInput::TemporalPag(pag),
            ) => {
                let review = antecedent_graph::TemporalPagReview::from_pag(
                    pag.clone(),
                    "supplied.temporal_pag",
                );
                if review.is_complete() {
                    match review.graph.try_into_temporal_dag() {
                        Ok(dag) => {
                            let mut logical =
                                compile_logical_temporal_effect(data, &dag, q, self.split, false)?;
                            logical.record.discovery_algorithm =
                                Some(Arc::from("supplied.temporal_pag.completed_to_dag"));
                            let physical = logical.compile_physical_with_graph(ctx, Some(dag))?;
                            Ok(CompiledAnalysis::Ready(physical))
                        }
                        Err(_) => Ok(compile_review_required_pag(review)),
                    }
                } else {
                    Ok(compile_review_required_pag(review))
                }
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
                GraphInput::TemporalCpdag(cpdag),
            ) => match cpdag.try_into_temporal_dag() {
                Ok(dag) => {
                    let logical =
                        compile_logical_temporal_effect(data, &dag, q, self.split, false)?;
                    let physical = logical.compile_physical_with_graph(ctx, Some(dag))?;
                    Ok(CompiledAnalysis::Ready(physical))
                }
                Err(_) => Ok(compile_review_required_cpdag(
                    antecedent_graph::TemporalCpdagReview::from_cpdag(
                        cpdag.clone(),
                        "supplied.temporal_cpdag",
                    ),
                )),
            },
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::DiscoverPc { alpha, max_cond_size, fdr, accept_discovered },
            ) => {
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let review = run_pc_review(data, *alpha, *max_cond_size, *fdr, ci, ctx)?;
                if *accept_discovered && review.pending_undirected.is_empty() {
                    let mut accepted = review;
                    accepted.pending_edges = Arc::from([]);
                    let dag = accepted
                        .try_into_dag()
                        .map_err(|e| CausalError::review_required_msg(e.to_string()))?;
                    let (identifier, estimator) = self.resolve_static_pair();
                    self.ensure_rd_config_present(&estimator)?;
                    let mut logical = compile_logical_static_ate(StaticAteCompileInput {
                        data,
                        graph: &dag,
                        query: q,
                        validation_suite: self.validation_suite_id(),
                        identifier,
                        estimator,
                    })?;
                    logical.record.discovery_algorithm = Some(Arc::from("pc"));
                    let physical = logical.compile_physical_with_graphs(ctx, None, Some(dag))?;
                    Ok(CompiledAnalysis::Ready(physical))
                } else {
                    Ok(compile_review_required_static_cpdag(review))
                }
            }
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::DiscoverGes { alpha, max_cond_size, fdr, accept_discovered },
            ) => {
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let review = run_ges_review(data, *alpha, *max_cond_size, *fdr, ci, ctx)?;
                if *accept_discovered && review.pending_undirected.is_empty() {
                    let mut accepted = review;
                    accepted.pending_edges = Arc::from([]);
                    let dag = accepted
                        .try_into_dag()
                        .map_err(|e| CausalError::review_required_msg(e.to_string()))?;
                    let (identifier, estimator) = self.resolve_static_pair();
                    self.ensure_rd_config_present(&estimator)?;
                    let mut logical = compile_logical_static_ate(StaticAteCompileInput {
                        data,
                        graph: &dag,
                        query: q,
                        validation_suite: self.validation_suite_id(),
                        identifier,
                        estimator,
                    })?;
                    logical.record.discovery_algorithm = Some(Arc::from("ges"));
                    let physical = logical.compile_physical_with_graphs(ctx, None, Some(dag))?;
                    Ok(CompiledAnalysis::Ready(physical))
                } else {
                    Ok(compile_review_required_static_cpdag(review))
                }
            }
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::DiscoverLingam { max_cond_size, prune_threshold, accept_discovered },
            ) => {
                let review = run_lingam_review(data, *max_cond_size, *prune_threshold, ctx)?;
                if *accept_discovered {
                    let dag = review
                        .accept_all()
                        .try_into_dag()
                        .map_err(|e| CausalError::review_required_msg(e.to_string()))?;
                    let (identifier, estimator) = self.resolve_static_pair();
                    self.ensure_rd_config_present(&estimator)?;
                    let mut logical = compile_logical_static_ate(StaticAteCompileInput {
                        data,
                        graph: &dag,
                        query: q,
                        validation_suite: self.validation_suite_id(),
                        identifier,
                        estimator,
                    })?;
                    logical.record.discovery_algorithm = Some(Arc::from("direct_lingam"));
                    let physical = logical.compile_physical_with_graphs(ctx, None, Some(dag))?;
                    Ok(CompiledAnalysis::Ready(physical))
                } else {
                    Ok(compile_review_required_static_dag(review))
                }
            }
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::DiscoverNotears {
                    max_cond_size,
                    lambda,
                    threshold,
                    standardize,
                    accept_discovered,
                },
            ) => {
                let review = run_notears_review(
                    data,
                    *max_cond_size,
                    *lambda,
                    *threshold,
                    *standardize,
                    ctx,
                )?;
                if *accept_discovered {
                    let dag = review
                        .accept_all()
                        .try_into_dag()
                        .map_err(|e| CausalError::review_required_msg(e.to_string()))?;
                    let (identifier, estimator) = self.resolve_static_pair();
                    self.ensure_rd_config_present(&estimator)?;
                    let mut logical = compile_logical_static_ate(StaticAteCompileInput {
                        data,
                        graph: &dag,
                        query: q,
                        validation_suite: self.validation_suite_id(),
                        identifier,
                        estimator,
                    })?;
                    logical.record.discovery_algorithm = Some(Arc::from("notears"));
                    let physical = logical.compile_physical_with_graphs(ctx, None, Some(dag))?;
                    Ok(CompiledAnalysis::Ready(physical))
                } else {
                    Ok(compile_review_required_static_dag(review))
                }
            }
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::DiscoverExactDagPosterior
                | GraphInput::DiscoverOrderMcmc { .. }
                | GraphInput::DiscoverStructureMcmc { .. }
                | GraphInput::DiscoverCiScreenedPosterior { .. },
            ) => self.compile_graph_posterior_static_ate(data, q, ctx),
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                graph @ GraphInput::DiscoverFci { alpha, max_cond_size, fdr, accept_discovered },
            ) => {
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let (identifier, estimator) = self.resolve_pag_pair();
                reject_dag_only_on_pag(graph, identifier.parse::<IdentifierId>()?)?;
                let review = run_fci_review(data, *alpha, *max_cond_size, *fdr, ci, ctx)?;
                // Accept-as-PAG: circle marks are handled by generalized adjustment over
                // MAG completions (same path as GraphInput::Pag). Review is only when
                // accept_discovered is false.
                if *accept_discovered {
                    let mut logical = compile_logical_static_pag_ate(StaticPagAteCompileInput {
                        data,
                        pag: &review.graph,
                        query: q,
                        validation_suite: self.validation_suite_id(),
                        identifier,
                        estimator,
                    })?;
                    logical.record.discovery_algorithm = Some(Arc::from("fci"));
                    let physical = logical.compile_physical_with_all_graphs(
                        ctx,
                        None,
                        None,
                        Some(review.graph.clone()),
                    )?;
                    Ok(CompiledAnalysis::Ready(physical))
                } else {
                    Ok(compile_review_required_static_pag(review))
                }
            }
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                graph @ GraphInput::DiscoverRfci { alpha, max_cond_size, fdr, accept_discovered },
            ) => {
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let (identifier, estimator) = self.resolve_pag_pair();
                reject_dag_only_on_pag(graph, identifier.parse::<IdentifierId>()?)?;
                let review = run_rfci_review(data, *alpha, *max_cond_size, *fdr, ci, ctx)?;
                if *accept_discovered {
                    let mut logical = compile_logical_static_pag_ate(StaticPagAteCompileInput {
                        data,
                        pag: &review.graph,
                        query: q,
                        validation_suite: self.validation_suite_id(),
                        identifier,
                        estimator,
                    })?;
                    logical.record.discovery_algorithm = Some(Arc::from("rfci"));
                    let physical = logical.compile_physical_with_all_graphs(
                        ctx,
                        None,
                        None,
                        Some(review.graph.clone()),
                    )?;
                    Ok(CompiledAnalysis::Ready(physical))
                } else {
                    Ok(compile_review_required_static_pag(review))
                }
            }
            (DataInput::Tabular(data), CausalQuery::AverageEffect(q), GraphInput::Pag(pag)) => {
                let (identifier, estimator) = self.resolve_pag_pair();
                reject_dag_only_on_pag(&self.graph, identifier.parse::<IdentifierId>()?)?;
                let logical = compile_logical_static_pag_ate(StaticPagAteCompileInput {
                    data,
                    pag,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })?;
                let physical =
                    logical.compile_physical_with_all_graphs(ctx, None, None, Some(pag.clone()))?;
                Ok(CompiledAnalysis::Ready(physical))
            }
            (DataInput::Tabular(data), CausalQuery::AverageEffect(q), GraphInput::Cpdag(cpdag)) => {
                match cpdag.try_into_dag() {
                    Ok(dag) => {
                        let (identifier, estimator) = self.resolve_static_pair();
                        self.ensure_rd_config_present(&estimator)?;
                        let logical = compile_logical_static_ate(StaticAteCompileInput {
                            data,
                            graph: &dag,
                            query: q,
                            validation_suite: self.validation_suite_id(),
                            identifier,
                            estimator,
                        })?;
                        let physical =
                            logical.compile_physical_with_graphs(ctx, None, Some(dag))?;
                        Ok(CompiledAnalysis::Ready(physical))
                    }
                    Err(_) => Ok(compile_review_required_static_cpdag(
                        antecedent_graph::CpdagReview::from_cpdag(cpdag.clone(), "supplied.cpdag"),
                    )),
                }
            }
            (DataInput::Tabular(data), CausalQuery::AverageEffect(q), GraphInput::Admg(admg)) => {
                if admg_has_bidirected(admg) {
                    let (identifier, estimator) = self.resolve_admg_pair();
                    validate_static_pair(
                        identifier.parse::<IdentifierId>()?,
                        estimator.parse::<EstimatorId>()?,
                    )?;
                    q.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
                    let record = antecedent_core::LogicalAnalysisPlanRecord {
                        plan_id: Arc::from("static_admg_ate"),
                        data_classification: antecedent_core::DataClassification::Tabular,
                        discovery_algorithm: None,
                        graph_review_required: false,
                        identifier: Some(identifier),
                        estimator: Some(estimator),
                        validation_suite: self.validation_suite_id(),
                        query_variables: Arc::from([q.treatment, q.outcome]),
                    };
                    let logical = LogicalAnalysisPlan {
                        record,
                        query: CausalQuery::AverageEffect(q.clone()),
                        split: None,
                        row_count_hint: data.row_count() as u64,
                    };
                    logical.validate()?;
                    let physical = logical.compile_physical(ctx)?;
                    Ok(CompiledAnalysis::Ready(physical))
                } else {
                    let dag = admg_to_dag(admg)?;
                    let (identifier, estimator) = self.resolve_static_pair();
                    self.ensure_rd_config_present(&estimator)?;
                    let logical = compile_logical_static_ate(StaticAteCompileInput {
                        data,
                        graph: &dag,
                        query: q,
                        validation_suite: self.validation_suite_id(),
                        identifier,
                        estimator,
                    })?;
                    let physical = logical.compile_physical_with_graphs(ctx, None, Some(dag))?;
                    Ok(CompiledAnalysis::Ready(physical))
                }
            }
            (
                DataInput::Tabular(data),
                CausalQuery::ConditionalEffect(q),
                GraphInput::Static(graph),
            ) => {
                let (identifier, estimator) = self.resolve_conditional_pair();
                let mut logical = compile_logical_static_ate(StaticAteCompileInput {
                    data,
                    graph,
                    query: &q.inner,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })?;
                logical.record.plan_id = Arc::from("static_conditional");
                logical.query = CausalQuery::ConditionalEffect(q.clone());
                let physical =
                    logical.compile_physical_with_graphs(ctx, None, Some(graph.clone()))?;
                Ok(CompiledAnalysis::Ready(physical))
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::Mediation(q),
                GraphInput::Temporal(graph),
            ) => {
                q.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
                let mut logical = compile_logical_temporal_effect(
                    data,
                    graph,
                    &TemporalEffectQuery::pulse(q.treatment, q.outcome, 1.0),
                    self.split,
                    false,
                )?;
                logical.record.plan_id = Arc::from("temporal_mediation");
                logical.record.identifier = Some(Arc::from("temporal.mediation"));
                logical.record.estimator = Some(Arc::from("temporal.mediation"));
                logical.record.query_variables = Arc::from([q.treatment, q.outcome]);
                logical.query = CausalQuery::Mediation(q.clone());
                let physical = logical.compile_physical_with_graph(ctx, Some(graph.clone()))?;
                Ok(CompiledAnalysis::Ready(physical))
            }
            (DataInput::Tabular(data), CausalQuery::Mediation(q), GraphInput::Static(graph)) => {
                q.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
                if !matches!(q.contrast, MediationContrast::Total) {
                    return Err(CausalError::Unsupported {
                        message: "static Mediation natural/direct/mediated contrasts require \
                             temporal data + TemporalDag; only MediationContrast::Total \
                             (front-door) is supported on a static DAG",
                    });
                }
                let ate = AverageEffectQuery::binary_ate(q.treatment, q.outcome);
                let mut logical = compile_logical_static_ate(StaticAteCompileInput {
                    data,
                    graph,
                    query: &ate,
                    validation_suite: self.validation_suite_id(),
                    identifier: Arc::from("frontdoor"),
                    estimator: Arc::from("frontdoor.two_stage"),
                })?;
                logical.record.plan_id = Arc::from("static_mediation_total");
                logical.query = CausalQuery::Mediation(q.clone());
                let physical =
                    logical.compile_physical_with_graphs(ctx, None, Some(graph.clone()))?;
                Ok(CompiledAnalysis::Ready(physical))
            }
            (
                DataInput::Tabular(data),
                CausalQuery::Counterfactual(_)
                | CausalQuery::AnomalyAttribution(_)
                | CausalQuery::ChangeAttribution(_)
                | CausalQuery::MechanismChange(_)
                | CausalQuery::UnitChange(_),
                GraphInput::Static(graph),
            ) => {
                let logical = self.compile_logical()?;
                let physical =
                    logical.compile_physical_with_graphs(ctx, None, Some(graph.clone()))?;
                let _ = data;
                Ok(CompiledAnalysis::Ready(physical))
            }
            _ => Err(CausalError::Unsupported {
                message: "unsupported data/graph/query combination",
            }),
        }
    }
}
