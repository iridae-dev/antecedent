//! Logical / physical analysis planning.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::large_enum_variant)]

use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, BufferMaterialization, CausalQuery, DataClassification, ExecutionContext,
    Intervention, KernelSelection, LogicalAnalysisPlanRecord, ParallelTaskSpec,
    PhysicalExecutionPlanRecord, ResponseFunctional, ResponseQuery, TargetPopulation,
    TemporalEffectQuery, VariableId,
};
use antecedent_data::{DiscoveryEstimationSplit, TableView, TabularData, TimeSeriesData};
use antecedent_graph::{Dag, Pag, TemporalDag};

use crate::accepted::{AcceptedGraph, GraphClass};
use crate::error::CausalError;
use crate::strategy_table::{
    EstimatorId, IdentifierId, validate_distribution_pair, validate_path_specific_pair,
    validate_response_pair, validate_static_pair,
};

/// Logical plan after compile (semantics only).
#[derive(Clone, Debug)]
pub struct LogicalAnalysisPlan {
    /// Record for results / serialization.
    pub record: LogicalAnalysisPlanRecord,
    /// Query being planned.
    pub query: CausalQuery,
    /// Optional temporal-gap split metadata.
    pub split: Option<DiscoveryEstimationSplit>,
    /// Row-count hint for memory / batch planning (estimation window when split).
    pub row_count_hint: u64,
}

impl LogicalAnalysisPlan {
    /// Validate logical semantics (modality × algorithm).
    ///
    /// # Errors
    ///
    /// Invalid combinations.
    pub fn validate(&self) -> Result<(), CausalError> {
        match (&self.query, self.record.data_classification) {
            (CausalQuery::TemporalEffect(_), DataClassification::Tabular) => {
                return Err(CausalError::Compile {
                    message: "temporal effect query requires temporal data".into(),
                });
            }
            (CausalQuery::AverageEffect(_), DataClassification::Temporal)
                if self.record.discovery_algorithm.is_some() =>
            {
                // Static ATE on temporal rows is allowed only without temporal discovery.
            }
            _ => {}
        }
        if matches!(
            self.record.discovery_algorithm.as_deref(),
            Some("pcmci" | "pcmci_plus" | "jpcmci_plus" | "rpcmci" | "lpcmci")
        ) && !matches!(
            self.record.data_classification,
            DataClassification::Temporal
                | DataClassification::Event
                | DataClassification::Panel
                | DataClassification::MultiEnvironment
        ) {
            return Err(CausalError::Compile {
                message: "PCMCI-family discovery requires temporal data metadata".into(),
            });
        }
        if matches!(self.record.discovery_algorithm.as_deref(), Some("pc"))
            && self.record.data_classification != DataClassification::Tabular
        {
            return Err(CausalError::Compile {
                message: "static PC discovery requires tabular data metadata".into(),
            });
        }
        self.query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        Ok(())
    }

    /// Compile a physical plan given execution capabilities / budget.
    ///
    /// # Errors
    ///
    /// Resource refusals or unsupported backends.
    pub fn compile_physical(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<PhysicalExecutionPlan, CausalError> {
        self.compile_physical_with_graph(ctx, None)
    }

    /// Compile a physical plan, optionally attaching a resolved temporal graph.
    ///
    /// # Errors
    ///
    /// Resource refusals or unsupported backends.
    pub fn compile_physical_with_graph(
        &self,
        ctx: &ExecutionContext,
        resolved_temporal_graph: Option<TemporalDag>,
    ) -> Result<PhysicalExecutionPlan, CausalError> {
        self.compile_physical_with_graphs(ctx, resolved_temporal_graph, None)
    }

    /// Compile a physical plan with optional resolved temporal and/or static graphs.
    ///
    /// # Errors
    ///
    /// Resource refusals or unsupported backends.
    pub fn compile_physical_with_graphs(
        &self,
        ctx: &ExecutionContext,
        resolved_temporal_graph: Option<TemporalDag>,
        resolved_static_graph: Option<Dag>,
    ) -> Result<PhysicalExecutionPlan, CausalError> {
        self.compile_physical_with_all_graphs(
            ctx,
            resolved_temporal_graph,
            resolved_static_graph,
            None,
        )
    }

    /// Compile a physical plan with optional resolved temporal DAG, static DAG, and/or static PAG.
    ///
    /// # Errors
    ///
    /// Resource refusals or unsupported backends.
    pub fn compile_physical_with_all_graphs(
        &self,
        ctx: &ExecutionContext,
        resolved_temporal_graph: Option<TemporalDag>,
        resolved_static_graph: Option<Dag>,
        resolved_static_pag: Option<Pag>,
    ) -> Result<PhysicalExecutionPlan, CausalError> {
        self.validate()?;
        let n_rows = self.row_count_hint.max(1);
        // Rough dense design: rows × ~8 f64 columns.
        let design_bytes = n_rows.saturating_mul(8).saturating_mul(8);
        let workspace = design_bytes.saturating_mul(2);
        let peak = design_bytes.saturating_add(workspace);
        let copy_bytes = design_bytes; // design matrix is CopiedContiguous

        if let Some(limit) = ctx.memory.soft_limit_bytes {
            if peak > limit {
                return Err(CausalError::Resource {
                    message: format!(
                        "estimated peak memory {peak} exceeds soft limit {limit}; no chunked path"
                    ),
                });
            }
        }

        let workers = if ctx.parallelism.max_threads.get() <= 1 {
            0
        } else {
            ctx.parallelism.max_threads.get()
        };

        // An unknown/unparseable estimator on the wire (or a missing one) must not fail
        // physical planning: fall back to the same labels the old `Other` escape produced
        // ("analysis" task dimension, "ols.faer" kernel) so previously-serialized artifacts
        // with an estimator name outside the current allowlist still load.
        let task_schedule: Arc<[ParallelTaskSpec]> = if workers == 0 {
            Arc::from([ParallelTaskSpec { dimension: Arc::from("serial"), units: 1 }])
        } else {
            let dimension = self
                .record
                .estimator
                .as_deref()
                .and_then(|s| s.parse::<EstimatorId>().ok())
                .map_or("analysis", |e| e.parallel_task_dimension());
            Arc::from([ParallelTaskSpec { dimension: Arc::from(dimension), units: workers }])
        };

        let kernel_label = self
            .record
            .estimator
            .as_deref()
            .and_then(|s| s.parse::<EstimatorId>().ok())
            .map_or("ols.faer", |e| e.kernel_label());
        let record = PhysicalExecutionPlanRecord {
            plan_id: Arc::clone(&self.record.plan_id),
            materializations: Arc::from([(
                Arc::from("design.matrix"),
                BufferMaterialization::CopiedContiguous,
            )]),
            kernels: Arc::from([(Arc::from(kernel_label), KernelSelection::DenseBackend)]),
            batch_size: Some(n_rows as usize),
            workspace_bytes: Some(workspace),
            estimated_peak_memory_bytes: Some(peak),
            estimated_copy_bytes: Some(copy_bytes),
            task_schedule,
            worker_threads: workers,
            deterministic_reductions: true,
            expected_python_crossings: 1,
        };
        Ok(PhysicalExecutionPlan {
            record,
            logical: self.clone(),
            resolved_temporal_graph,
            resolved_static_graph,
            resolved_static_pag,
        })
    }
}

/// Physical plan ready for execution.
#[derive(Clone, Debug)]
pub struct PhysicalExecutionPlan {
    /// Record for results.
    pub record: PhysicalExecutionPlanRecord,
    /// Logical plan this was derived from.
    pub logical: LogicalAnalysisPlan,
    /// Temporal DAG to estimate against (supplied or post-review). Avoids re-discovery.
    pub resolved_temporal_graph: Option<TemporalDag>,
    /// Static DAG from PC discovery auto-accept (avoids re-discovery at execute).
    pub resolved_static_graph: Option<Dag>,
    /// Static PAG from FCI/RFCI / supplied Pag (class-aware identification).
    pub resolved_static_pag: Option<Pag>,
}

impl PhysicalExecutionPlan {
    /// Borrow the resolved temporal graph when present.
    #[must_use]
    pub fn temporal_graph(&self) -> Option<&TemporalDag> {
        self.resolved_temporal_graph.as_ref()
    }

    /// Borrow the resolved static DAG when present (PC discovery path).
    #[must_use]
    pub fn static_graph(&self) -> Option<&Dag> {
        self.resolved_static_graph.as_ref()
    }

    /// Borrow the resolved static PAG when present (FCI / RFCI / supplied Pag).
    #[must_use]
    pub fn static_pag(&self) -> Option<&Pag> {
        self.resolved_static_pag.as_ref()
    }
}

/// Whether an identifier is DAG-only (cannot accept a PAG without completion / class-aware ID).
#[must_use]
pub fn is_dag_only_identifier(identifier: IdentifierId) -> bool {
    identifier.is_dag_only()
}

/// Refuse DAG-only identification on a PAG structure.
///
/// # Errors
///
/// [`CausalError::Compile`] when a DAG-only identifier is paired with a PAG structure.
pub fn reject_dag_only_on_pag(
    structure: &AcceptedGraph,
    identifier: IdentifierId,
) -> Result<(), CausalError> {
    let is_pag = matches!(structure.class(), GraphClass::Pag | GraphClass::TemporalPag);
    if is_pag && identifier.is_dag_only() {
        return Err(CausalError::Compile {
            message: format!(
                "DAG-only identification {:?} cannot accept a PAG without a completion \
                 or class-aware identifier (use generalized.adjustment)",
                identifier.as_str()
            ),
        });
    }
    Ok(())
}

/// Inputs needed to compile a logical plan for the static ATE path.
#[derive(Clone, Debug)]
pub struct StaticAteCompileInput<'a> {
    /// Tabular data (classification + row count).
    pub data: &'a TabularData,
    /// Graph.
    pub graph: &'a Dag,
    /// Query.
    pub query: &'a AverageEffectQuery,
    /// Validation suite id.
    pub validation_suite: Option<Arc<str>>,
    /// Identifier id selected by the builder (defaults to `backdoor.adjustment`).
    pub identifier: Arc<str>,
    /// Estimator id selected by the builder (defaults to `linear.adjustment.ate`).
    pub estimator: Arc<str>,
}

/// Compile logical plan for static ATE .
///
/// # Errors
///
/// Query validation failures, or an identifier/estimator pair not in the compile-time allowlist
/// (see [`crate::strategy_table::validate_static_pair`]).
pub fn compile_logical_static_ate(
    input: StaticAteCompileInput<'_>,
) -> Result<LogicalAnalysisPlan, CausalError> {
    input.query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
    validate_query_vars_in_dag(input.graph, input.query.treatment, input.query.outcome)?;
    let identifier: IdentifierId = input.identifier.parse()?;
    let estimator: EstimatorId = input.estimator.parse()?;
    validate_static_pair(identifier, estimator)?;
    if matches!(estimator, EstimatorId::LinearAdjustmentAte)
        && input.query.target_population != TargetPopulation::AllObserved
    {
        return Err(CausalError::Compile {
            message: format!(
                "estimator \"linear.adjustment.ate\" only supports TargetPopulation::AllObserved \
                 (got {:?}); use a propensity or AIPW estimator for ATT/ATC/Predicate",
                input.query.target_population
            ),
        });
    }
    let record = LogicalAnalysisPlanRecord {
        plan_id: Arc::from("static_ate"),
        data_classification: DataClassification::Tabular,
        discovery_algorithm: None,
        graph_review_required: false,
        identifier: Some(Arc::clone(&input.identifier)),
        estimator: Some(Arc::clone(&input.estimator)),
        validation_suite: input.validation_suite,
        query_variables: Arc::from([input.query.treatment, input.query.outcome]),
    };
    let plan = LogicalAnalysisPlan {
        record,
        query: CausalQuery::AverageEffect(input.query.clone()),
        split: None,
        row_count_hint: input.data.row_count() as u64,
    };
    plan.validate()?;
    Ok(plan)
}

/// Inputs for a static-DAG continuous-response plan.
#[derive(Clone, Debug)]
pub struct StaticResponseCompileInput<'a> {
    /// Tabular data.
    pub data: &'a TabularData,
    /// Accepted causal DAG.
    pub graph: &'a Dag,
    /// Response query.
    pub query: &'a ResponseQuery,
    /// Validation suite id.
    pub validation_suite: Option<Arc<str>>,
    /// Identifier selected by the builder.
    pub identifier: Arc<str>,
    /// Estimator selected by the builder.
    pub estimator: Arc<str>,
}

/// Compile a continuous-response query over tabular data and a static DAG.
///
/// # Errors
///
/// Invalid queries, out-of-range variables, or incompatible strategies.
pub fn compile_logical_static_response(
    input: StaticResponseCompileInput<'_>,
) -> Result<LogicalAnalysisPlan, CausalError> {
    input.query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
    let identifier: IdentifierId = input.identifier.parse()?;
    let estimator: EstimatorId = input.estimator.parse()?;
    validate_response_pair(identifier, estimator)?;
    let expected = match &input.query.functional {
        ResponseFunctional::MeanCurve { .. } | ResponseFunctional::PointDerivative { .. } => {
            EstimatorId::ResponseKennedyDr
        }
        ResponseFunctional::AverageDerivative { .. } => EstimatorId::ResponseRieszAde,
        ResponseFunctional::DirectionalDerivative { .. } | ResponseFunctional::Jacobian { .. } => {
            EstimatorId::ResponseGamDerivative
        }
        ResponseFunctional::InterventionResponse { .. } => EstimatorId::ResponseInterventionGcomp,
    };
    if estimator != expected {
        return Err(CausalError::Compile {
            message: format!(
                "response functional requires estimator {:?}; got {:?}",
                expected.as_str(),
                estimator.as_str()
            ),
        });
    }
    let (treatments, outcomes) = response_query_variables(&input.query.functional);
    for &treatment in &treatments {
        for &outcome in &outcomes {
            validate_query_vars_in_dag(input.graph, treatment, outcome)?;
        }
    }
    let query_variables: Arc<[VariableId]> =
        treatments.iter().chain(outcomes.iter()).copied().collect::<Vec<_>>().into();
    let plan = LogicalAnalysisPlan {
        record: LogicalAnalysisPlanRecord {
            plan_id: Arc::from("static_response"),
            data_classification: DataClassification::Tabular,
            discovery_algorithm: None,
            graph_review_required: false,
            identifier: Some(input.identifier),
            estimator: Some(input.estimator),
            validation_suite: input.validation_suite,
            query_variables,
        },
        query: CausalQuery::Response(input.query.clone()),
        split: None,
        row_count_hint: input.data.row_count() as u64,
    };
    plan.validate()?;
    Ok(plan)
}

fn response_query_variables(functional: &ResponseFunctional) -> (Vec<VariableId>, Vec<VariableId>) {
    match functional {
        ResponseFunctional::MeanCurve { outcome, treatment } => {
            (vec![treatment.variable], vec![*outcome])
        }
        ResponseFunctional::AverageDerivative { outcome, treatment, .. }
        | ResponseFunctional::PointDerivative { outcome, treatment, .. } => {
            (vec![*treatment], vec![*outcome])
        }
        ResponseFunctional::DirectionalDerivative { outcomes, treatments, .. }
        | ResponseFunctional::Jacobian { outcomes, treatments, .. } => {
            (treatments.to_vec(), outcomes.to_vec())
        }
        ResponseFunctional::InterventionResponse { outcome, interventions } => (
            interventions.iter().filter_map(Intervention::primary_variable).collect(),
            vec![*outcome],
        ),
    }
}

/// Inputs for PAG ATE compile (class-aware identification).
#[derive(Clone, Debug)]
pub struct StaticPagAteCompileInput<'a> {
    /// Tabular data.
    pub data: &'a TabularData,
    /// PAG.
    pub pag: &'a Pag,
    /// Query.
    pub query: &'a AverageEffectQuery,
    /// Validation suite id.
    pub validation_suite: Option<Arc<str>>,
    /// Identifier (must be generalized.adjustment).
    pub identifier: Arc<str>,
    /// Estimator id.
    pub estimator: Arc<str>,
}

/// Compile logical plan for static ATE on a PAG.
///
/// # Errors
///
/// Query validation or incompatible identifier/estimator.
pub fn compile_logical_static_pag_ate(
    input: StaticPagAteCompileInput<'_>,
) -> Result<LogicalAnalysisPlan, CausalError> {
    input.query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
    validate_query_vars_in_pag(input.pag, input.query.treatment, input.query.outcome)?;
    let identifier: IdentifierId = input.identifier.parse()?;
    let estimator: EstimatorId = input.estimator.parse()?;
    if !matches!(identifier, IdentifierId::GeneralizedAdjustment) {
        return Err(CausalError::Compile {
            message: format!(
                "PAG ATE requires identifier \"generalized.adjustment\"; got {:?}",
                identifier.as_str()
            ),
        });
    }
    validate_static_pair(identifier, estimator)?;
    let record = LogicalAnalysisPlanRecord {
        plan_id: Arc::from("static_pag_ate"),
        data_classification: DataClassification::Tabular,
        discovery_algorithm: None,
        graph_review_required: false,
        identifier: Some(Arc::clone(&input.identifier)),
        estimator: Some(Arc::clone(&input.estimator)),
        validation_suite: input.validation_suite,
        query_variables: Arc::from([input.query.treatment, input.query.outcome]),
    };
    let plan = LogicalAnalysisPlan {
        record,
        query: CausalQuery::AverageEffect(input.query.clone()),
        split: None,
        row_count_hint: input.data.row_count() as u64,
    };
    plan.validate()?;
    Ok(plan)
}

/// Compile logical plan for interventional-distribution queries.
#[derive(Clone, Debug)]
pub struct StaticDistributionCompileInput<'a> {
    /// Tabular data.
    pub data: &'a TabularData,
    /// Graph.
    pub graph: &'a Dag,
    /// Distribution query.
    pub query: &'a antecedent_core::InterventionalDistributionQuery,
    /// Validation suite id.
    pub validation_suite: Option<Arc<str>>,
    /// Identifier (`general.id` / `auto`).
    pub identifier: Arc<str>,
    /// Estimator (`functional.distribution`).
    pub estimator: Arc<str>,
}

/// Compile logical plan for an interventional distribution.
///
/// # Errors
///
/// Query validation or incompatible identifier/estimator.
pub fn compile_logical_distribution(
    input: StaticDistributionCompileInput<'_>,
) -> Result<LogicalAnalysisPlan, CausalError> {
    input.query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
    if input.query.target_population != TargetPopulation::AllObserved {
        return Err(CausalError::Compile {
            message: "functional.distribution only supports TargetPopulation::AllObserved".into(),
        });
    }
    let treatment =
        input.query.interventions.first().and_then(Intervention::primary_variable).ok_or_else(
            || CausalError::Compile {
                message:
                    "distribution query requires at least one intervention with a primary variable"
                        .into(),
            },
        )?;
    let outcome = *input.query.outcomes.first().ok_or_else(|| CausalError::Compile {
        message: "distribution query requires at least one outcome".into(),
    })?;
    validate_query_vars_in_dag(input.graph, treatment, outcome)?;
    let identifier: IdentifierId = input.identifier.parse()?;
    let estimator: EstimatorId = input.estimator.parse()?;
    validate_distribution_pair(identifier, estimator)?;
    let mut qvars = vec![treatment, outcome];
    for &z in input.query.conditioning.iter() {
        if !qvars.contains(&z) {
            qvars.push(z);
        }
    }
    let record = LogicalAnalysisPlanRecord {
        plan_id: Arc::from("static_distribution"),
        data_classification: DataClassification::Tabular,
        discovery_algorithm: None,
        graph_review_required: false,
        identifier: Some(Arc::clone(&input.identifier)),
        estimator: Some(Arc::clone(&input.estimator)),
        validation_suite: input.validation_suite,
        query_variables: Arc::from(qvars),
    };
    let plan = LogicalAnalysisPlan {
        record,
        query: CausalQuery::Distribution(input.query.clone()),
        split: None,
        row_count_hint: input.data.row_count() as u64,
    };
    plan.validate()?;
    Ok(plan)
}

/// Compile input for path-specific natural-effect queries.
#[derive(Clone, Debug)]
pub struct StaticPathSpecificCompileInput<'a> {
    /// Tabular data.
    pub data: &'a TabularData,
    /// Graph.
    pub graph: &'a Dag,
    /// Path-specific query.
    pub query: &'a antecedent_core::PathSpecificEffectQuery,
    /// Validation suite id.
    pub validation_suite: Option<Arc<str>>,
    /// Identifier.
    pub identifier: Arc<str>,
    /// Estimator.
    pub estimator: Arc<str>,
}

/// Compile logical plan for path-specific natural effects.
///
/// # Errors
///
/// Query validation or incompatible identifier/estimator.
pub fn compile_logical_path_specific(
    input: StaticPathSpecificCompileInput<'_>,
) -> Result<LogicalAnalysisPlan, CausalError> {
    input.query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
    if input.query.target_population != TargetPopulation::AllObserved {
        return Err(CausalError::Compile {
            message: "functional.effect only supports TargetPopulation::AllObserved".into(),
        });
    }
    validate_query_vars_in_dag(input.graph, input.query.treatment, input.query.outcome)?;
    let identifier: IdentifierId = input.identifier.parse()?;
    let estimator: EstimatorId = input.estimator.parse()?;
    validate_path_specific_pair(identifier, estimator)?;
    let mut qvars = vec![input.query.treatment, input.query.outcome];
    for &m in input.query.path_nodes.iter() {
        if !qvars.contains(&m) {
            qvars.push(m);
        }
    }
    let record = LogicalAnalysisPlanRecord {
        plan_id: Arc::from("static_path_specific"),
        data_classification: DataClassification::Tabular,
        discovery_algorithm: None,
        graph_review_required: false,
        identifier: Some(Arc::clone(&input.identifier)),
        estimator: Some(Arc::clone(&input.estimator)),
        validation_suite: input.validation_suite,
        query_variables: Arc::from(qvars),
    };
    let plan = LogicalAnalysisPlan {
        record,
        query: CausalQuery::PathSpecific(input.query.clone()),
        split: None,
        row_count_hint: input.data.row_count() as u64,
    };
    plan.validate()?;
    Ok(plan)
}

fn validate_query_vars_in_dag(
    dag: &Dag,
    treatment: antecedent_core::VariableId,
    outcome: antecedent_core::VariableId,
) -> Result<(), CausalError> {
    let mut has_t = false;
    let mut has_y = false;
    for node in dag.nodes() {
        if let antecedent_graph::NodeRef::Static(v) = node {
            if *v == treatment {
                has_t = true;
            }
            if *v == outcome {
                has_y = true;
            }
        }
    }
    if !has_t || !has_y {
        return Err(CausalError::Compile {
            message: format!(
                "query variables not in DAG (treatment present={has_t}, outcome present={has_y})"
            ),
        });
    }
    Ok(())
}

fn validate_query_vars_in_pag(
    pag: &Pag,
    treatment: antecedent_core::VariableId,
    outcome: antecedent_core::VariableId,
) -> Result<(), CausalError> {
    let mut has_t = false;
    let mut has_y = false;
    for node in pag.nodes() {
        if let antecedent_graph::NodeRef::Static(v) = node {
            if *v == treatment {
                has_t = true;
            }
            if *v == outcome {
                has_y = true;
            }
        }
    }
    if !has_t || !has_y {
        return Err(CausalError::Compile {
            message: format!(
                "query variables not in PAG (treatment present={has_t}, outcome present={has_y})"
            ),
        });
    }
    Ok(())
}

fn validate_query_vars_in_temporal_dag(
    dag: &TemporalDag,
    treatment: antecedent_core::VariableId,
    outcome: antecedent_core::VariableId,
) -> Result<(), CausalError> {
    // A node-less DAG is the placeholder the graph-posterior path supplies: the
    // structure lives in the `GraphPosterior` mixture, not here, so there is no
    // membership to check and rejecting would be a false negative.
    if dag.nodes().is_empty() {
        return Ok(());
    }
    let mut has_t = false;
    let mut has_y = false;
    for node in dag.nodes() {
        // Temporal graphs carry `Lagged` nodes, and `Context` nodes when an
        // environment is attached; both name a variable the query may reference.
        let variable = match node {
            antecedent_graph::NodeRef::Lagged { variable, .. }
            | antecedent_graph::NodeRef::Context { variable, .. } => *variable,
            antecedent_graph::NodeRef::Static(v) => *v,
        };
        if variable == treatment {
            has_t = true;
        }
        if variable == outcome {
            has_y = true;
        }
    }
    if !has_t || !has_y {
        return Err(CausalError::Compile {
            message: format!(
                "query variables not in temporal DAG (treatment present={has_t}, outcome \
                 present={has_y})"
            ),
        });
    }
    Ok(())
}

/// Compile logical plan for a temporal effect with a supplied temporal graph.
///
/// # Errors
///
/// Modality / query validation failures.
pub fn compile_logical_temporal_effect(
    data: &TimeSeriesData,
    graph: &TemporalDag,
    query: &TemporalEffectQuery,
    split: Option<DiscoveryEstimationSplit>,
    review_required: bool,
) -> Result<LogicalAnalysisPlan, CausalError> {
    compile_logical_temporal_effect_classified(
        data,
        graph,
        query,
        split,
        review_required,
        DataClassification::Temporal,
    )
}

/// Temporal effect plan with an explicit data classification (Event / Panel / Temporal).
///
/// # Errors
///
/// Query validation failures.
pub fn compile_logical_temporal_effect_classified(
    data: &TimeSeriesData,
    graph: &TemporalDag,
    query: &TemporalEffectQuery,
    split: Option<DiscoveryEstimationSplit>,
    review_required: bool,
    data_classification: DataClassification,
) -> Result<LogicalAnalysisPlan, CausalError> {
    query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
    validate_query_vars_in_temporal_dag(graph, query.treatment, query.outcome)?;
    if query.target_population != TargetPopulation::AllObserved {
        return Err(CausalError::Compile {
            message: format!(
                "temporal linear adjustment only supports TargetPopulation::AllObserved \
                 (got {:?})",
                query.target_population
            ),
        });
    }
    let row_count_hint =
        split.map_or_else(|| data.row_count() as u64, |s| s.estimation.len() as u64);
    let record = LogicalAnalysisPlanRecord {
        plan_id: Arc::from("temporal_effect"),
        data_classification,
        discovery_algorithm: None,
        graph_review_required: review_required,
        identifier: Some(Arc::from("temporal.backdoor.unfolded")),
        estimator: Some(Arc::from("temporal.linear.adjustment")),
        validation_suite: None,
        query_variables: Arc::from([query.treatment, query.outcome]),
    };
    let plan = LogicalAnalysisPlan {
        record,
        query: CausalQuery::TemporalEffect(query.clone()),
        split,
        row_count_hint,
    };
    plan.validate()?;
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        AverageEffectQuery, ExecutionContext, MemoryBudget, TemporalEffectQuery, VariableId,
    };
    use antecedent_graph::TemporalDag;

    use super::*;

    fn tabular_plan(rows: u64) -> LogicalAnalysisPlan {
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        LogicalAnalysisPlan {
            record: LogicalAnalysisPlanRecord {
                plan_id: Arc::from("test"),
                data_classification: DataClassification::Tabular,
                discovery_algorithm: None,
                graph_review_required: false,
                identifier: Some(Arc::from("backdoor.adjustment")),
                estimator: Some(Arc::from("linear.adjustment.ate")),
                validation_suite: None,
                query_variables: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
            },
            query: CausalQuery::AverageEffect(q),
            split: None,
            row_count_hint: rows,
        }
    }

    #[test]
    fn static_ate_compiles_with_schedule_and_copies() {
        let plan = tabular_plan(200);
        plan.validate().unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let physical = plan.compile_physical(&ctx).unwrap();
        assert!(physical.record.estimated_peak_memory_bytes.is_some());
        assert_eq!(physical.record.kernels.len(), 1);
        assert_eq!(physical.record.estimated_copy_bytes, Some(200 * 8 * 8));
        assert_eq!(physical.record.task_schedule.len(), 1);
        assert_eq!(&*physical.record.task_schedule[0].dimension, "serial");
        assert!(!physical.record.materializations.is_empty());
    }

    #[test]
    fn temporal_query_on_tabular_fails() {
        let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0);
        let plan = LogicalAnalysisPlan {
            record: LogicalAnalysisPlanRecord {
                plan_id: Arc::from("bad"),
                data_classification: DataClassification::Tabular,
                discovery_algorithm: None,
                graph_review_required: false,
                identifier: None,
                estimator: None,
                validation_suite: None,
                query_variables: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
            },
            query: CausalQuery::TemporalEffect(q),
            split: None,
            row_count_hint: 10,
        };
        assert!(matches!(plan.validate(), Err(CausalError::Compile { .. })));
    }

    #[test]
    fn pcmci_on_tabular_fails() {
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let plan = LogicalAnalysisPlan {
            record: LogicalAnalysisPlanRecord {
                plan_id: Arc::from("bad"),
                data_classification: DataClassification::Tabular,
                discovery_algorithm: Some(Arc::from("pcmci")),
                graph_review_required: true,
                identifier: None,
                estimator: None,
                validation_suite: None,
                query_variables: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
            },
            query: CausalQuery::AverageEffect(q),
            split: None,
            row_count_hint: 10,
        };
        assert!(matches!(plan.validate(), Err(CausalError::Compile { .. })));
    }

    #[test]
    fn soft_memory_limit_refuses_dense_plan() {
        let plan = tabular_plan(10_000);
        let mut ctx = ExecutionContext::for_tests(1);
        ctx.memory = MemoryBudget { soft_limit_bytes: Some(64), hard_limit_bytes: None };
        assert!(matches!(plan.compile_physical(&ctx), Err(CausalError::Resource { .. })));
    }

    #[test]
    fn split_row_hint_drives_batch_size() {
        let mut plan = tabular_plan(100);
        plan.split = Some(DiscoveryEstimationSplit::from_sizes(100, 50, 10, 40).unwrap());
        plan.row_count_hint = 40;
        let ctx = ExecutionContext::for_tests(1);
        let physical = plan.compile_physical(&ctx).unwrap();
        assert_eq!(physical.record.batch_size, Some(40));
    }

    fn toy_static_input() -> (TabularData, Dag, AverageEffectQuery) {
        use antecedent_core::{
            CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
        };
        use antecedent_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, ValidityBitmap};
        use antecedent_graph::DenseNodeId;
        use std::sync::Arc as StdArc;

        let n = 10usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let t: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * t[i]).collect();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    StdArc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    StdArc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        (TabularData::new(storage), dag, query)
    }

    #[test]
    fn refuses_iv_estimator_with_backdoor_identifier() {
        let (data, graph, query) = toy_static_input();
        let err = compile_logical_static_ate(StaticAteCompileInput {
            data: &data,
            graph: &graph,
            query: &query,
            validation_suite: None,
            identifier: Arc::from("backdoor.adjustment"),
            estimator: Arc::from("iv.2sls"),
        })
        .unwrap_err();
        assert!(matches!(err, CausalError::Compile { .. }));
    }

    #[test]
    fn refuses_propensity_estimator_with_frontdoor_identifier() {
        let (data, graph, query) = toy_static_input();
        let err = compile_logical_static_ate(StaticAteCompileInput {
            data: &data,
            graph: &graph,
            query: &query,
            validation_suite: None,
            identifier: Arc::from("frontdoor"),
            estimator: Arc::from("propensity.weighting"),
        })
        .unwrap_err();
        assert!(matches!(err, CausalError::Compile { .. }));
    }

    #[test]
    fn refuses_unknown_identifier_and_estimator() {
        let (data, graph, query) = toy_static_input();
        let err = compile_logical_static_ate(StaticAteCompileInput {
            data: &data,
            graph: &graph,
            query: &query,
            validation_suite: None,
            identifier: Arc::from("backdoor.adjustment"),
            estimator: Arc::from("not.a.real.estimator"),
        })
        .unwrap_err();
        assert!(matches!(err, CausalError::Compile { .. }));
    }

    #[test]
    fn refuses_att_target_population_with_linear_adjustment() {
        use antecedent_core::TargetPopulation;
        let (data, graph, query) = toy_static_input();
        let att_query = query.with_target_population(TargetPopulation::Treated);
        let err = compile_logical_static_ate(StaticAteCompileInput {
            data: &data,
            graph: &graph,
            query: &att_query,
            validation_suite: None,
            identifier: Arc::from("backdoor.adjustment"),
            estimator: Arc::from("linear.adjustment.ate"),
        })
        .unwrap_err();
        assert!(matches!(err, CausalError::Compile { .. }));
    }

    #[test]
    fn refuses_planned_target_population_on_temporal_effect() {
        use antecedent_core::{
            CausalSchemaBuilder, MeasurementSpec, PredicateExpr, RoleHint, SmallRoleSet,
            TargetPopulation, ValueType,
        };
        use antecedent_data::{
            Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
            ValidityBitmap,
        };
        use std::sync::Arc as StdArc;

        let n = 8usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    StdArc::from(vec![0.0; n]),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    StdArc::from(vec![0.0; n]),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap();
        let graph = TemporalDag::empty();
        let query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_target_population(TargetPopulation::Predicate(PredicateExpr::named(
                    "cohort_a",
                )));
        let err = compile_logical_temporal_effect(&data, &graph, &query, None, false).unwrap_err();
        assert!(matches!(err, CausalError::Compile { .. }));
    }

    #[test]
    fn refuses_temporal_query_vars_not_in_temporal_dag() {
        use antecedent_core::{
            CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
        };
        use antecedent_data::{
            Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
            ValidityBitmap,
        };
        use std::sync::Arc as StdArc;

        let n = 8usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    StdArc::from(vec![0.0; n]),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    StdArc::from(vec![0.0; n]),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap();
        // A populated temporal DAG that does NOT name the query variables: the plan must
        // fail at compile time rather than silently accepting an unfounded query
        // (target_population is left at the default AllObserved so the later population
        // check cannot be the one firing).
        //
        // The DAG must be non-empty: a node-less TemporalDag is the placeholder the
        // graph-posterior path supplies, and is deliberately exempt from this check —
        // see `validate_query_vars_in_temporal_dag`.
        let mut graph = TemporalDag::empty();
        graph.add_lagged(VariableId::from_raw(7), antecedent_core::Lag::CONTEMPORANEOUS).unwrap();
        graph.add_lagged(VariableId::from_raw(8), antecedent_core::Lag::CONTEMPORANEOUS).unwrap();
        let query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0);
        let err = compile_logical_temporal_effect(&data, &graph, &query, None, false).unwrap_err();
        let CausalError::Compile { message } = err else {
            panic!("expected CausalError::Compile, got {err:?}");
        };
        assert!(
            message.contains("not in temporal DAG"),
            "expected a temporal-DAG membership error, got: {message}"
        );
    }

    #[test]
    fn accepts_default_pair() {
        let (data, graph, query) = toy_static_input();
        let plan = compile_logical_static_ate(StaticAteCompileInput {
            data: &data,
            graph: &graph,
            query: &query,
            validation_suite: None,
            identifier: Arc::from("backdoor.adjustment"),
            estimator: Arc::from("linear.adjustment.ate"),
        })
        .unwrap();
        assert_eq!(plan.record.identifier.as_deref(), Some("backdoor.adjustment"));
        assert_eq!(plan.record.estimator.as_deref(), Some("linear.adjustment.ate"));
    }

    #[test]
    fn accepts_propensity_weighting_with_backdoor_adjustment() {
        let (data, graph, query) = toy_static_input();
        let plan = compile_logical_static_ate(StaticAteCompileInput {
            data: &data,
            graph: &graph,
            query: &query,
            validation_suite: None,
            identifier: Arc::from("backdoor.adjustment"),
            estimator: Arc::from("propensity.weighting"),
        })
        .unwrap();
        assert_eq!(plan.record.estimator.as_deref(), Some("propensity.weighting"));
    }

    #[test]
    fn refuses_dag_only_identifier_on_pag() {
        use antecedent_graph::Pag;
        let structure = AcceptedGraph::pag(Pag::with_variables(2));
        let err = reject_dag_only_on_pag(&structure, IdentifierId::BackdoorAdjustment).unwrap_err();
        assert!(matches!(err, CausalError::Compile { .. }));
        // Class-aware identifier is allowed through this gate.
        let structure = AcceptedGraph::pag(Pag::with_variables(2));
        reject_dag_only_on_pag(&structure, IdentifierId::GeneralizedAdjustment).unwrap();
    }
}
