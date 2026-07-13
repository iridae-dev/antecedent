//! Logical / physical analysis planning (DESIGN.md §21.2).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::large_enum_variant)]

use std::sync::Arc;

use causal_core::{
    AverageEffectQuery, BufferMaterialization, CausalQuery, DataClassification, ExecutionContext,
    KernelSelection, LogicalAnalysisPlanRecord, ParallelTaskSpec, PhysicalExecutionPlanRecord,
    TargetPopulation, TemporalEffectQuery,
};
use causal_data::{DiscoveryEstimationSplit, TableView, TabularData, TimeSeriesData};
use causal_graph::{Dag, TemporalCpdagReview, TemporalDag, TemporalGraphReview};

use crate::error::AnalysisError;

/// How the causal graph is supplied to the planner.
#[derive(Clone, Debug)]
pub enum GraphInput {
    /// Validated static DAG.
    Static(Dag),
    /// Validated temporal DAG (template).
    Temporal(TemporalDag),
    /// Discover with PCMCI (review usually required).
    DiscoverPcmci {
        /// Max lag for PCMCI.
        max_lag: u32,
        /// Significance level.
        alpha: f64,
        /// Apply FDR.
        fdr: bool,
        /// Auto-accept discovered edges (skip review).
        accept_discovered: bool,
    },
    /// Discover with PCMCI+ (temporal CPDAG; review/orientation usually required).
    DiscoverPcmciPlus {
        /// Max lag for PCMCI+.
        max_lag: u32,
        /// Significance level.
        alpha: f64,
        /// Apply FDR.
        fdr: bool,
        /// Auto-accept directed edges when no undirected marks remain.
        ///
        /// If undirected contemporaneous edges remain after orientation, compile still
        /// returns [`CompiledAnalysis::ReviewRequiredCpdag`] (never silently coerces).
        accept_discovered: bool,
    },
}

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
    pub fn validate(&self) -> Result<(), AnalysisError> {
        match (&self.query, self.record.data_classification) {
            (CausalQuery::TemporalEffect(_), DataClassification::Tabular) => {
                return Err(AnalysisError::Compile {
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
            Some("pcmci" | "pcmci_plus")
        ) && self.record.data_classification != DataClassification::Temporal
        {
            return Err(AnalysisError::Compile {
                message: "PCMCI / PCMCI+ requires temporal data metadata".into(),
            });
        }
        self.query.validate().map_err(|e| AnalysisError::Compile { message: e.to_string() })?;
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
    ) -> Result<PhysicalExecutionPlan, AnalysisError> {
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
    ) -> Result<PhysicalExecutionPlan, AnalysisError> {
        self.validate()?;
        let n_rows = self.row_count_hint.max(1);
        // Rough dense design: rows × ~8 f64 columns.
        let design_bytes = n_rows.saturating_mul(8).saturating_mul(8);
        let workspace = design_bytes.saturating_mul(2);
        let peak = design_bytes.saturating_add(workspace);
        let copy_bytes = design_bytes; // design matrix is CopiedContiguous

        if let Some(limit) = ctx.memory.soft_limit_bytes {
            if peak > limit {
                return Err(AnalysisError::Resource {
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

        let task_schedule: Arc<[ParallelTaskSpec]> = if workers == 0 {
            Arc::from([ParallelTaskSpec { dimension: Arc::from("serial"), units: 1 }])
        } else {
            Arc::from([ParallelTaskSpec {
                dimension: Arc::from(match self.record.estimator.as_deref() {
                    // Every Phase 3/4 estimator supports an optional IID bootstrap; parallelize
                    // over replicates when threads are available.
                    Some(
                        "temporal.linear.adjustment"
                        | "linear.adjustment.ate"
                        | "propensity.weighting"
                        | "propensity.matching"
                        | "propensity.stratification"
                        | "distance.matching"
                        | "aipw"
                        | "glm.adjustment"
                        | "frontdoor.two_stage"
                        | "iv.wald"
                        | "iv.2sls"
                        | "rd.sharp",
                    ) => "bootstrap.replicate",
                    _ => "analysis",
                }),
                units: workers,
            }])
        };

        let record = PhysicalExecutionPlanRecord {
            plan_id: Arc::clone(&self.record.plan_id),
            materializations: Arc::from([(
                Arc::from("design.matrix"),
                BufferMaterialization::CopiedContiguous,
            )]),
            kernels: Arc::from([(
                Arc::from(match self.record.estimator.as_deref() {
                    Some("temporal.linear.adjustment") => "ols.faer.temporal",
                    Some("propensity.weighting") => "ipw",
                    Some("propensity.matching" | "distance.matching") => "matching",
                    Some("propensity.stratification") => "propensity.stratification",
                    Some("aipw") => "aipw",
                    Some("glm.adjustment") => "glm.logit",
                    Some("frontdoor.two_stage") => "frontdoor.two_stage",
                    Some("iv.wald") => "iv.wald",
                    Some("iv.2sls") => "2sls",
                    Some("rd.sharp") => "rd.local_linear",
                    _ => "ols.faer",
                }),
                KernelSelection::DenseBackend,
            )]),
            batch_size: Some(n_rows as usize),
            workspace_bytes: Some(workspace),
            estimated_peak_memory_bytes: Some(peak),
            estimated_copy_bytes: Some(copy_bytes),
            task_schedule,
            worker_threads: workers,
            deterministic_reductions: true,
            expected_python_crossings: 1,
        };
        Ok(PhysicalExecutionPlan { record, logical: self.clone(), resolved_temporal_graph })
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
}

impl PhysicalExecutionPlan {
    /// Borrow the resolved temporal graph when present.
    #[must_use]
    pub fn temporal_graph(&self) -> Option<&TemporalDag> {
        self.resolved_temporal_graph.as_ref()
    }
}

/// Result of compilation: ready to run, or graph review required.
#[derive(Clone, Debug)]
pub enum CompiledAnalysis {
    /// Physical plan may execute.
    Ready(PhysicalExecutionPlan),
    /// Discovery / incomplete DAG needs human acceptance.
    ReviewRequired(TemporalGraphReview),
    /// PCMCI+ CPDAG needs acceptance of directed edges and orientation of undirected marks.
    ReviewRequiredCpdag(TemporalCpdagReview),
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

/// Compile-time allowlist of identifier/estimator pairs for the static ATE path (Phase 4;
/// DESIGN.md §21.2). Any pair not listed here — including unknown ids, an IV estimator paired
/// with a non-IV identifier, `frontdoor.two_stage` without `frontdoor`, or a propensity/AIPW
/// estimator without a `backdoor.*` identifier — is refused.
fn validate_static_pair(identifier: &str, estimator: &str) -> Result<(), AnalysisError> {
    let supported = matches!(
        (identifier, estimator),
        (
            "backdoor.adjustment" | "backdoor.efficient",
            "linear.adjustment.ate"
                | "propensity.weighting"
                | "propensity.matching"
                | "propensity.stratification"
                | "distance.matching"
                | "aipw"
                | "glm.adjustment"
        ) | ("frontdoor", "frontdoor.two_stage")
            | ("iv", "iv.wald" | "iv.2sls")
            | ("rd.sharp", "rd.sharp")
    );
    if !supported {
        return Err(AnalysisError::Compile {
            message: format!(
                "identifier {identifier:?} is not compatible with estimator {estimator:?}"
            ),
        });
    }
    Ok(())
}

/// Compile logical plan for static ATE (Phase 4 planner/estimator routing via DESIGN.md §21.2).
///
/// # Errors
///
/// Query validation failures, or an identifier/estimator pair not in the compile-time allowlist
/// (see [`validate_static_pair`]).
pub fn compile_logical_static_ate(
    input: StaticAteCompileInput<'_>,
) -> Result<LogicalAnalysisPlan, AnalysisError> {
    let _ = input.graph;
    input.query.validate().map_err(|e| AnalysisError::Compile { message: e.to_string() })?;
    validate_static_pair(&input.identifier, &input.estimator)?;
    if &*input.estimator == "linear.adjustment.ate"
        && input.query.target_population != TargetPopulation::AllObserved
    {
        return Err(AnalysisError::Compile {
            message: format!(
                "estimator \"linear.adjustment.ate\" only supports TargetPopulation::AllObserved \
                 (got {:?}); use a propensity or AIPW estimator for ATT/ATC",
                input.query.target_population
            ),
        });
    }
    let record = LogicalAnalysisPlanRecord {
        plan_id: Arc::from("phase4.static_ate"),
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

/// Compile logical plan for a temporal effect with a supplied temporal graph.
///
/// # Errors
///
/// Modality / query validation failures.
pub fn compile_logical_temporal_effect(
    data: &TimeSeriesData,
    _graph: &TemporalDag,
    query: &TemporalEffectQuery,
    split: Option<DiscoveryEstimationSplit>,
    review_required: bool,
) -> Result<LogicalAnalysisPlan, AnalysisError> {
    query.validate().map_err(|e| AnalysisError::Compile { message: e.to_string() })?;
    let row_count_hint =
        split.map_or_else(|| data.row_count() as u64, |s| s.estimation.len() as u64);
    let record = LogicalAnalysisPlanRecord {
        plan_id: Arc::from("phase3.temporal_effect"),
        data_classification: DataClassification::Temporal,
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
    use causal_core::{
        AverageEffectQuery, ExecutionContext, MemoryBudget, TemporalEffectQuery, VariableId,
    };

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
        assert!(matches!(plan.validate(), Err(AnalysisError::Compile { .. })));
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
        assert!(matches!(plan.validate(), Err(AnalysisError::Compile { .. })));
    }

    #[test]
    fn soft_memory_limit_refuses_dense_plan() {
        let plan = tabular_plan(10_000);
        let mut ctx = ExecutionContext::for_tests(1);
        ctx.memory = MemoryBudget { soft_limit_bytes: Some(64), hard_limit_bytes: None };
        assert!(matches!(plan.compile_physical(&ctx), Err(AnalysisError::Resource { .. })));
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
        use causal_core::{
            CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
        };
        use causal_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, ValidityBitmap};
        use causal_graph::DenseNodeId;
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
        assert!(matches!(err, AnalysisError::Compile { .. }));
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
        assert!(matches!(err, AnalysisError::Compile { .. }));
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
        assert!(matches!(err, AnalysisError::Compile { .. }));
    }

    #[test]
    fn refuses_att_target_population_with_linear_adjustment() {
        use causal_core::TargetPopulation;
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
        assert!(matches!(err, AnalysisError::Compile { .. }));
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
}
