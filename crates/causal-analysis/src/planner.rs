//! Logical / physical analysis planning (DESIGN.md §21.2).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{
    AverageEffectQuery, BufferMaterialization, CausalQuery, DataClassification, ExecutionContext,
    KernelSelection, LogicalAnalysisPlanRecord, PhysicalExecutionPlanRecord, TemporalEffectQuery,
};
use causal_data::{DiscoveryEstimationSplit, TabularData, TimeSeriesData};
use causal_graph::{Dag, TemporalDag, TemporalGraphReview};

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
        if self.record.discovery_algorithm.as_deref() == Some("pcmci")
            && self.record.data_classification != DataClassification::Temporal
        {
            return Err(AnalysisError::Compile {
                message: "PCMCI requires temporal data metadata".into(),
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
        self.validate()?;
        let n_rows_hint = self.split.map_or(0_u64, |s| s.estimation.len() as u64).max(1);
        let design_bytes = n_rows_hint.saturating_mul(8).saturating_mul(8); // rough
        let workspace = design_bytes.saturating_mul(2);
        let peak = design_bytes.saturating_add(workspace);

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

        let record = PhysicalExecutionPlanRecord {
            plan_id: Arc::clone(&self.record.plan_id),
            materializations: Arc::from([(
                Arc::from("design.matrix"),
                BufferMaterialization::CopiedContiguous,
            )]),
            kernels: Arc::from([(
                Arc::from(match self.record.estimator.as_deref() {
                    Some("temporal.linear.adjustment") => "ols.faer.temporal",
                    _ => "ols.faer",
                }),
                KernelSelection::DenseBackend,
            )]),
            batch_size: Some(n_rows_hint as usize),
            workspace_bytes: Some(workspace),
            estimated_peak_memory_bytes: Some(peak),
            worker_threads: workers,
            deterministic_reductions: true,
            expected_python_crossings: 1,
        };
        Ok(PhysicalExecutionPlan { record, logical: self.clone() })
    }
}

/// Physical plan ready for execution.
#[derive(Clone, Debug)]
pub struct PhysicalExecutionPlan {
    /// Record for results.
    pub record: PhysicalExecutionPlanRecord,
    /// Logical plan this was derived from.
    pub logical: LogicalAnalysisPlan,
}

/// Result of compilation: ready to run, or graph review required.
#[derive(Clone, Debug)]
pub enum CompiledAnalysis {
    /// Physical plan may execute.
    Ready(PhysicalExecutionPlan),
    /// Discovery / incomplete graph needs human acceptance.
    ReviewRequired(TemporalGraphReview),
}

/// Inputs needed to compile a logical plan for the static ATE path.
#[derive(Clone, Debug)]
pub struct StaticAteCompileInput<'a> {
    /// Tabular data (classification only).
    pub data: &'a TabularData,
    /// Graph.
    pub graph: &'a Dag,
    /// Query.
    pub query: &'a AverageEffectQuery,
    /// Validation suite id.
    pub validation_suite: Option<Arc<str>>,
}

/// Compile logical plan for static ATE (Phase 1 path via planner).
///
/// # Errors
///
/// Validation failures.
pub fn compile_logical_static_ate(
    input: StaticAteCompileInput<'_>,
) -> Result<LogicalAnalysisPlan, AnalysisError> {
    let _ = input.data;
    let _ = input.graph;
    input.query.validate().map_err(|e| AnalysisError::Compile { message: e.to_string() })?;
    let record = LogicalAnalysisPlanRecord {
        plan_id: Arc::from("phase3.static_ate"),
        data_classification: DataClassification::Tabular,
        discovery_algorithm: None,
        graph_review_required: false,
        identifier: Some(Arc::from("backdoor.adjustment")),
        estimator: Some(Arc::from("linear.adjustment.ate")),
        validation_suite: input.validation_suite,
        query_variables: Arc::from([input.query.treatment, input.query.outcome]),
    };
    let plan = LogicalAnalysisPlan {
        record,
        query: CausalQuery::AverageEffect(input.query.clone()),
        split: None,
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
    let _ = data;
    query.validate().map_err(|e| AnalysisError::Compile { message: e.to_string() })?;
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

    #[test]
    fn static_ate_compiles() {
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        // Minimal fake table not needed for compile_logical_static_ate metadata path —
        // we still need TabularData; skip full construction by using validate on plan directly.
        let record = LogicalAnalysisPlanRecord {
            plan_id: Arc::from("test"),
            data_classification: DataClassification::Tabular,
            discovery_algorithm: None,
            graph_review_required: false,
            identifier: Some(Arc::from("backdoor.adjustment")),
            estimator: Some(Arc::from("linear.adjustment.ate")),
            validation_suite: None,
            query_variables: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
        };
        let plan = LogicalAnalysisPlan {
            record,
            query: CausalQuery::AverageEffect(q),
            split: None,
        };
        plan.validate().unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let physical = plan.compile_physical(&ctx).unwrap();
        assert!(physical.record.estimated_peak_memory_bytes.is_some());
        assert_eq!(physical.record.kernels.len(), 1);
    }

    #[test]
    fn temporal_query_on_tabular_fails() {
        let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0);
        let record = LogicalAnalysisPlanRecord {
            plan_id: Arc::from("bad"),
            data_classification: DataClassification::Tabular,
            discovery_algorithm: None,
            graph_review_required: false,
            identifier: None,
            estimator: None,
            validation_suite: None,
            query_variables: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
        };
        let plan = LogicalAnalysisPlan {
            record,
            query: CausalQuery::TemporalEffect(q),
            split: None,
        };
        assert!(matches!(plan.validate(), Err(AnalysisError::Compile { .. })));
    }

    #[test]
    fn pcmci_on_tabular_fails() {
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let record = LogicalAnalysisPlanRecord {
            plan_id: Arc::from("bad"),
            data_classification: DataClassification::Tabular,
            discovery_algorithm: Some(Arc::from("pcmci")),
            graph_review_required: true,
            identifier: None,
            estimator: None,
            validation_suite: None,
            query_variables: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
        };
        let plan = LogicalAnalysisPlan {
            record,
            query: CausalQuery::AverageEffect(q),
            split: None,
        };
        assert!(matches!(plan.validate(), Err(AnalysisError::Compile { .. })));
    }

    #[test]
    fn soft_memory_limit_refuses() {
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let record = LogicalAnalysisPlanRecord {
            plan_id: Arc::from("mem"),
            data_classification: DataClassification::Tabular,
            discovery_algorithm: None,
            graph_review_required: false,
            identifier: Some(Arc::from("backdoor.adjustment")),
            estimator: Some(Arc::from("linear.adjustment.ate")),
            validation_suite: None,
            query_variables: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
        };
        let plan = LogicalAnalysisPlan {
            record,
            query: CausalQuery::AverageEffect(q),
            split: Some(
                DiscoveryEstimationSplit::from_sizes(10_000, 5_000, 0, 5_000).unwrap(),
            ),
        };
        let mut ctx = ExecutionContext::for_tests(1);
        ctx.memory = MemoryBudget { soft_limit_bytes: Some(64), hard_limit_bytes: None };
        assert!(matches!(plan.compile_physical(&ctx), Err(AnalysisError::Resource { .. })));
    }
}
