// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

impl super::Study {
    /// Compile logical plan only (inspectable semantics).
    ///
    /// # Errors
    ///
    /// Modality / query validation failures.
    ///
    /// # Panics
    ///
    /// Never in practice. Each `expect` below asserts an invariant that
    /// [`crate::accepted::AcceptedGraph::class`] itself establishes — the class tag is
    /// derived from the stored graph, so the matching accessor (`as_dag`, `as_pag`, …)
    /// is always `Some` once the match arm has already selected on that class.
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
        self.ensure_supported_combination()?;
        match (&self.data, &self.query, self.graph.class()) {
            (DataInput::Tabular(data), CausalQuery::AverageEffect(q), GraphClass::Dag) => {
                let graph = self.graph.as_dag().expect("class() == Dag implies as_dag() is Some");
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
            (DataInput::Tabular(data), CausalQuery::Distribution(q), GraphClass::Dag) => {
                let graph = self.graph.as_dag().expect("class() == Dag implies as_dag() is Some");
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
            (DataInput::Tabular(data), CausalQuery::PathSpecific(q), GraphClass::Dag) => {
                let graph = self.graph.as_dag().expect("class() == Dag implies as_dag() is Some");
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
                GraphClass::TemporalDag,
            ) => {
                let graph = self
                    .graph
                    .as_temporal_dag()
                    .expect("class() == TemporalDag implies as_temporal_dag() is Some");
                let class = match &self.data {
                    DataInput::Event(_) => DataClassification::Event,
                    _ => DataClassification::Temporal,
                };
                compile_logical_temporal_effect_classified(data, graph, q, self.split, false, class)
            }
            (
                DataInput::MultiEnv(multi),
                CausalQuery::TemporalEffect(q),
                GraphClass::TemporalDag,
            ) => {
                let graph = self
                    .graph
                    .as_temporal_dag()
                    .expect("class() == TemporalDag implies as_temporal_dag() is Some");
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
            (DataInput::Panel(panel), CausalQuery::TemporalEffect(q), GraphClass::TemporalDag) => {
                let graph = self
                    .graph
                    .as_temporal_dag()
                    .expect("class() == TemporalDag implies as_temporal_dag() is Some");
                let data = &panel
                    .unit(0)
                    .map_err(|e| CausalError::Compile { message: format!("panel: {e}") })?
                    .series;
                compile_logical_temporal_effect_classified(
                    data,
                    graph,
                    q,
                    self.split,
                    false,
                    DataClassification::Panel,
                )
            }
            (DataInput::Tabular(data), CausalQuery::AverageEffect(q), GraphClass::Pag) => {
                let pag = self.graph.as_pag().expect("class() == Pag implies as_pag() is Some");
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
            (DataInput::Tabular(data), CausalQuery::ConditionalEffect(q), GraphClass::Dag) => {
                let graph = self.graph.as_dag().expect("class() == Dag implies as_dag() is Some");
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
                GraphClass::TemporalDag,
            ) => {
                let graph = self
                    .graph
                    .as_temporal_dag()
                    .expect("class() == TemporalDag implies as_temporal_dag() is Some");
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
            (DataInput::Tabular(data), CausalQuery::Mediation(q), GraphClass::Dag) => {
                let graph = self.graph.as_dag().expect("class() == Dag implies as_dag() is Some");
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
                GraphClass::Dag,
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
            _ => Err(CausalError::Unsupported {
                message: "unsupported data/graph/query combination",
            }),
        }
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
    /// class/accessor invariant applies here.
    pub fn compile(&self, ctx: &ExecutionContext) -> Result<PhysicalExecutionPlan, CausalError> {
        if self.graph_posterior.is_some() {
            return self.compile_graph_posterior(ctx);
        }
        self.ensure_supported_combination()?;
        match (&self.data, &self.query, self.graph.class()) {
            (DataInput::Tabular(data), CausalQuery::AverageEffect(q), GraphClass::Dag) => {
                let graph = self.graph.as_dag().expect("class() == Dag implies as_dag() is Some");
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
                logical.compile_physical(ctx)
            }
            (DataInput::Tabular(data), CausalQuery::Distribution(q), GraphClass::Dag) => {
                let graph = self.graph.as_dag().expect("class() == Dag implies as_dag() is Some");
                let (identifier, estimator) = self.resolve_distribution_pair();
                let logical = compile_logical_distribution(StaticDistributionCompileInput {
                    data,
                    graph,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })?;
                logical.compile_physical(ctx)
            }
            (DataInput::Tabular(data), CausalQuery::PathSpecific(q), GraphClass::Dag) => {
                let graph = self.graph.as_dag().expect("class() == Dag implies as_dag() is Some");
                let (identifier, estimator) = self.resolve_path_pair();
                let logical = compile_logical_path_specific(StaticPathSpecificCompileInput {
                    data,
                    graph,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })?;
                logical.compile_physical(ctx)
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
                GraphClass::TemporalDag,
            ) => {
                let graph = self
                    .graph
                    .as_temporal_dag()
                    .expect("class() == TemporalDag implies as_temporal_dag() is Some");
                let class = match &self.data {
                    DataInput::Event(_) => DataClassification::Event,
                    _ => DataClassification::Temporal,
                };
                let logical = compile_logical_temporal_effect_classified(
                    data, graph, q, self.split, false, class,
                )?;
                logical.compile_physical_with_graph(ctx, Some(graph.clone()))
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
                GraphClass::TemporalCpdag,
            ) => {
                let cpdag = self
                    .graph
                    .as_temporal_cpdag()
                    .expect("class() == TemporalCpdag implies as_temporal_cpdag() is Some");
                // The review gate at `AcceptedGraph::temporal_cpdag` already refused any
                // TemporalCpdag carrying undirected or conflict marks, so this completion
                // is a lossless structural reinterpretation, never a coercion.
                let dag = cpdag.try_into_temporal_dag().map_err(|e| CausalError::Compile {
                    message: format!("accepted temporal CPDAG failed to complete: {e}"),
                })?;
                let class = match &self.data {
                    DataInput::Event(_) => DataClassification::Event,
                    _ => DataClassification::Temporal,
                };
                let logical = compile_logical_temporal_effect_classified(
                    data, &dag, q, self.split, false, class,
                )?;
                logical.compile_physical_with_graph(ctx, Some(dag))
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
                GraphClass::TemporalPag,
            ) => {
                let pag = self
                    .graph
                    .as_temporal_pag()
                    .expect("class() == TemporalPag implies as_temporal_pag() is Some");
                // Temporal backdoor is DAG-only. `AcceptedGraph::temporal_pag` already
                // refused any PAG carrying circle marks, so this conversion should not
                // fail; the arm is defensive, not the review gate (that lives at
                // construction, and reports `ReviewRequired` with a pending count).
                let dag = pag.try_into_temporal_dag().map_err(|_| CausalError::Compile {
                    message: "temporal PAG has unresolved circle marks; temporal backdoor \
                              requires a fully directed structure (no class-aware temporal PAG \
                              identifier is wired today)"
                        .into(),
                })?;
                let class = match &self.data {
                    DataInput::Event(_) => DataClassification::Event,
                    _ => DataClassification::Temporal,
                };
                let mut logical = compile_logical_temporal_effect_classified(
                    data, &dag, q, self.split, false, class,
                )?;
                // Name the algorithm that actually produced this structure. A PAG that
                // reached here via `AcceptedGraph::accept(lpcmci_review)` is not
                // "supplied" — recording it as such loses the provenance the record
                // exists to carry. Asserted graphs have no algorithm and keep the
                // `supplied.` prefix.
                logical.record.discovery_algorithm = Some(self.graph.algorithm_id().map_or_else(
                    || Arc::from("supplied.temporal_pag.completed_to_dag"),
                    |a| Arc::from(format!("{a}.pag_completed_to_dag").as_str()),
                ));
                logical.compile_physical_with_graph(ctx, Some(dag))
            }
            (
                DataInput::MultiEnv(multi),
                CausalQuery::TemporalEffect(q),
                GraphClass::TemporalDag,
            ) => {
                let graph = self
                    .graph
                    .as_temporal_dag()
                    .expect("class() == TemporalDag implies as_temporal_dag() is Some");
                let data = multi
                    .environment(0)
                    .map_err(|e| CausalError::Compile { message: format!("multi-env: {e}") })?;
                let logical = compile_logical_temporal_effect_classified(
                    data,
                    graph,
                    q,
                    self.split,
                    false,
                    DataClassification::MultiEnvironment,
                )?;
                logical.compile_physical_with_graph(ctx, Some(graph.clone()))
            }
            (DataInput::Panel(panel), CausalQuery::TemporalEffect(q), GraphClass::TemporalDag) => {
                let graph = self
                    .graph
                    .as_temporal_dag()
                    .expect("class() == TemporalDag implies as_temporal_dag() is Some");
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
                logical.compile_physical_with_graph(ctx, Some(graph.clone()))
            }
            (DataInput::Tabular(data), CausalQuery::AverageEffect(q), GraphClass::Pag) => {
                let pag = self.graph.as_pag().expect("class() == Pag implies as_pag() is Some");
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
                logical.compile_physical_with_all_graphs(ctx, None, None, Some(pag.clone()))
            }
            (DataInput::Tabular(data), CausalQuery::AverageEffect(q), GraphClass::Cpdag) => {
                let cpdag =
                    self.graph.as_cpdag().expect("class() == Cpdag implies as_cpdag() is Some");
                // Same completion guarantee as the temporal CPDAG branch above.
                let dag = cpdag.try_into_dag().map_err(|e| CausalError::Compile {
                    message: format!("accepted CPDAG failed to complete to DAG: {e}"),
                })?;
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
                logical.compile_physical_with_graphs(ctx, None, Some(dag))
            }
            (DataInput::Tabular(data), CausalQuery::AverageEffect(q), GraphClass::Admg) => {
                let admg = self.graph.as_admg().expect("class() == Admg implies as_admg() is Some");
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
                    logical.compile_physical(ctx)
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
                    logical.compile_physical_with_graphs(ctx, None, Some(dag))
                }
            }
            (DataInput::Tabular(data), CausalQuery::ConditionalEffect(q), GraphClass::Dag) => {
                let graph = self.graph.as_dag().expect("class() == Dag implies as_dag() is Some");
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
                logical.compile_physical_with_graphs(ctx, None, Some(graph.clone()))
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::Mediation(q),
                GraphClass::TemporalDag,
            ) => {
                let graph = self
                    .graph
                    .as_temporal_dag()
                    .expect("class() == TemporalDag implies as_temporal_dag() is Some");
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
                logical.compile_physical_with_graph(ctx, Some(graph.clone()))
            }
            (DataInput::Tabular(data), CausalQuery::Mediation(q), GraphClass::Dag) => {
                let graph = self.graph.as_dag().expect("class() == Dag implies as_dag() is Some");
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
                logical.compile_physical_with_graphs(ctx, None, Some(graph.clone()))
            }
            (
                DataInput::Tabular(data),
                CausalQuery::Counterfactual(_)
                | CausalQuery::AnomalyAttribution(_)
                | CausalQuery::ChangeAttribution(_)
                | CausalQuery::MechanismChange(_)
                | CausalQuery::UnitChange(_),
                GraphClass::Dag,
            ) => {
                let graph = self.graph.as_dag().expect("class() == Dag implies as_dag() is Some");
                let logical = self.compile_logical()?;
                let _ = data;
                logical.compile_physical_with_graphs(ctx, None, Some(graph.clone()))
            }
            _ => Err(CausalError::Unsupported {
                message: "unsupported data/graph/query combination",
            }),
        }
    }
}
