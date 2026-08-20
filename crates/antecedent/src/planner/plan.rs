//! Logical / physical analysis planning.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::large_enum_variant)]

use std::sync::Arc;

use antecedent_core::{
    BufferMaterialization, CausalQuery, DataClassification, ExecutionContext, KernelSelection,
    LogicalAnalysisPlanRecord, ParallelTaskSpec, PhysicalExecutionPlanRecord,
};
use antecedent_data::DiscoveryEstimationSplit;
use antecedent_graph::{Dag, Pag, TemporalDag};

use crate::accepted::{AcceptedGraph, GraphClass};
use crate::error::CausalError;
use crate::strategy_table::{EstimatorId, IdentifierId};

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
