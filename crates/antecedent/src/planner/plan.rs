//! Logical / physical analysis planning.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use antecedent_core::physical_choice::{
    AvailableCapabilities, CostQuotes, DeclaredCost, PhysicalAlternative, PhysicalChoice,
};

use std::sync::Arc;

use antecedent_contract::{
    BufferMaterialization, DataClassification, KernelSelection, LogicalAnalysisPlanRecord,
    ParallelTaskSpec, PhysicalExecutionPlanRecord,
};
use antecedent_core::{CausalQuery, ExecutionContext};
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
    /// Encoded design width (including intercept), or a conservative schema bound before preparation.
    pub design_columns_hint: u64,
}

impl LogicalAnalysisPlan {
    /// A plan over the whole data window (no discovery/estimation split).
    #[must_use]
    pub(crate) fn unsplit(
        record: LogicalAnalysisPlanRecord,
        query: CausalQuery,
        row_count_hint: u64,
    ) -> Self {
        Self {
            design_columns_hint: record.query_variables.len().max(1) as u64,
            record,
            query,
            split: None,
            row_count_hint,
        }
    }

    /// Validate logical semantics (modality × algorithm).
    ///
    /// # Errors
    ///
    /// Invalid combinations.
    pub fn validate(&self) -> Result<(), CausalError> {
        // A static average effect never carries a temporal classification (it compiles as
        // tabular), and a temporal graph under `AverageEffect` is an n/a support cell, so no
        // modality rule for it belongs here.
        if matches!(
            (&self.query, self.record.data_classification),
            (CausalQuery::TemporalEffect(_), DataClassification::Tabular)
        ) {
            return Err(CausalError::Compile {
                message: "temporal effect query requires temporal data".into(),
            });
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
        self.query.validate()?;
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
        let DesignEstimates { design: design_bytes, workspace, peak } =
            design_estimates(n_rows, self.design_columns_hint)?;
        let copy_bytes = design_bytes; // design matrix is CopiedContiguous
        check_memory_limit(peak, ctx)?;

        let materialized = PhysicalAlternative::MaterializedDesign;
        let available = AvailableCapabilities::empty()
            .with_operator(materialized.required_operator())
            .with_resource(materialized.required_resource());
        let quote = DeclaredCost::from_raw(peak);
        let choice = PhysicalChoice::choose(CostQuotes::new(quote, quote, quote, quote), available)
            .map_err(|error| CausalError::Resource { message: error.to_string() })?;
        let requested_workers =
            ctx.parallelism.max_threads.get().min(ctx.admission().thread_budget());
        let workers = if requested_workers <= 1 { 0 } else { requested_workers };

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
            batch_size: Some(usize::try_from(n_rows).unwrap_or(usize::MAX)),
            workspace_bytes: Some(workspace),
            estimated_peak_memory_bytes: Some(peak),
            estimated_copy_bytes: Some(copy_bytes),
            task_schedule,
            worker_threads: workers,
            deterministic_reductions: true,
            expected_python_crossings: 1,
        };
        Ok(PhysicalExecutionPlan {
            linear: None,
            lowered: None,
            choice,
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
    pub(crate) linear: Option<Arc<crate::analysis::compiled_linear::CompiledLinearExecution>>,
    /// Lowered program `Study::compile` attached. A plan built from a logical record
    /// alone carries none, and dispatch refuses it.
    pub(crate) lowered: Option<Arc<super::lowered::LoweredProgram>>,
    /// Executable backend selection, with rejected alternatives and reasons.
    pub choice: antecedent_core::physical_choice::PhysicalChoice,
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
    /// Checked scientific and statistical program shared with prepared execution, when applicable.
    #[must_use]
    pub fn compiled_linear_program(
        &self,
    ) -> Option<&crate::analysis::compiled_linear::CompiledLinearProgram> {
        self.linear.as_ref().map(|linear| linear.program.as_ref())
    }

    /// The lowered program `Study::compile` attached, when this plan has one.
    #[must_use]
    pub fn lowered(&self) -> Option<&super::lowered::LoweredProgram> {
        self.lowered.as_deref()
    }

    /// Identity of the attached lowered program, when this plan has one.
    #[must_use]
    pub fn lowered_program_identity(&self) -> Option<antecedent_core::SemanticDigest> {
        self.lowered().map(super::lowered::LoweredProgram::identity)
    }

    /// The lowered program dispatch may execute, checked against the route the
    /// study classifies as now (`expected`, when it classifies as one).
    ///
    /// # Errors
    ///
    /// `route_not_lowered`: the plan carries no lowered program, or its program was
    /// lowered for a different route.
    pub(crate) fn require_lowered(
        &self,
        expected: Option<crate::analysis::lowering::RouteCell>,
    ) -> Result<&super::lowered::LoweredProgram, CausalError> {
        let lowered = self.lowered().ok_or_else(|| {
            crate::compile_reason!(
                "route_not_lowered",
                "the physical plan carries no lowered program; compile it with Study::compile"
            )
        })?;
        match expected {
            Some(cell) if cell != lowered.cell() => Err(crate::compile_reason!(
                "route_not_lowered",
                "the plan was lowered for {} but the study executes as {}",
                lowered.route(),
                cell.as_str()
            )),
            _ => Ok(lowered),
        }
    }
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

impl PhysicalExecutionPlan {
    /// Choose an executable design implementation before checked preparation.
    pub(crate) fn bind_prepared_linear_views(
        &mut self,
        direct_eligible: bool,
        work: u64,
    ) -> Result<(), CausalError> {
        use antecedent_core::physical_choice::{
            AvailableCapabilities, CostQuotes, DeclaredCost, PhysicalAlternative, PhysicalChoice,
        };
        let direct = PhysicalAlternative::DirectPreparedNumericViews;
        let materialized = PhysicalAlternative::MaterializedDesign;
        let mut available = AvailableCapabilities::empty()
            .with_operator(materialized.required_operator())
            .with_resource(materialized.required_resource());
        if direct_eligible {
            available = available
                .with_operator(direct.required_operator())
                .with_resource(direct.required_resource());
        }
        let peak = self.record.estimated_peak_memory_bytes.ok_or_else(|| {
            CausalError::Resource { message: "prepared design omitted memory estimate".into() }
        })?;
        let copy = self.record.estimated_copy_bytes.ok_or_else(|| CausalError::Resource {
            message: "prepared design omitted copy estimate".into(),
        })?;
        let cost = peak.checked_add(work).ok_or_else(|| CausalError::Resource {
            message: "linear execution work quote overflow".into(),
        })?;
        let repeated = cost.checked_add(copy).ok_or_else(|| CausalError::Resource {
            message: "materialized execution cost overflow".into(),
        })?;
        self.choice = PhysicalChoice::choose(
            CostQuotes::new(
                DeclaredCost::from_raw(cost),
                DeclaredCost::from_raw(cost),
                DeclaredCost::from_raw(repeated),
                DeclaredCost::from_raw(cost),
            ),
            available,
        )
        .map_err(|error| CausalError::Resource { message: error.to_string() })?;
        self.record.materializations =
            Arc::from([(Arc::from("design.matrix"), BufferMaterialization::CopiedContiguous)]);
        Ok(())
    }

    /// Re-plan the design bytes and materialization from a quoted design shape, and
    /// apply the caller's memory limit to the re-planned peak.
    ///
    /// A direct design borrows its numeric source columns and copies once into the
    /// final design; a materialized design copies the selected and expanded columns
    /// before assembling the final design.
    ///
    /// # Errors
    ///
    /// The re-planned peak exceeds the memory limit, or the size arithmetic overflows.
    pub(crate) fn requote(
        &mut self,
        design: super::lowered::DesignShape,
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        // Only a quoted design re-plans the record; the planner's own sizing stays
        // for a route that is not design based.
        let (Some(rows), Some(columns)) = (design.rows(), design.columns()) else {
            return Ok(());
        };
        let rows = rows.max(1);
        let DesignEstimates { design: design_bytes, workspace, peak } =
            design_estimates(rows, columns)?;
        let (source, copies) = if design.direct() {
            (BufferMaterialization::Borrowed, 1)
        } else {
            (BufferMaterialization::CopiedContiguous, 2)
        };
        let copy_bytes = design_bytes
            .checked_mul(copies)
            .ok_or_else(|| CausalError::Resource { message: "copy estimate overflow".into() })?;
        let peak = peak
            .checked_add(copy_bytes - design_bytes)
            .ok_or_else(|| CausalError::Resource { message: "peak estimate overflow".into() })?;
        check_memory_limit(peak, ctx)?;
        self.logical.row_count_hint = rows;
        self.logical.design_columns_hint = columns.max(1);
        self.record.batch_size = Some(usize::try_from(rows).unwrap_or(usize::MAX));
        self.record.workspace_bytes = Some(workspace);
        self.record.estimated_peak_memory_bytes = Some(peak);
        self.record.estimated_copy_bytes = Some(copy_bytes);
        self.record.materializations = Arc::from([
            (Arc::from("source.columns"), source),
            (Arc::from("design.matrix"), BufferMaterialization::CopiedContiguous),
        ]);
        Ok(())
    }
}

/// Bytes of a dense `rows × columns` design, its workspace, and their peak.
struct DesignEstimates {
    design: u64,
    workspace: u64,
    peak: u64,
}

fn design_estimates(rows: u64, columns: u64) -> Result<DesignEstimates, CausalError> {
    let design =
        rows.max(1).checked_mul(columns.max(1)).and_then(|cells| cells.checked_mul(8)).ok_or_else(
            || CausalError::Resource { message: "design memory estimate overflow".into() },
        )?;
    let workspace = design
        .checked_mul(2)
        .ok_or_else(|| CausalError::Resource { message: "workspace estimate overflow".into() })?;
    let peak = design
        .checked_add(workspace)
        .ok_or_else(|| CausalError::Resource { message: "peak estimate overflow".into() })?;
    Ok(DesignEstimates { design, workspace, peak })
}

fn check_memory_limit(peak: u64, ctx: &ExecutionContext) -> Result<(), CausalError> {
    if let Some(limit) = ctx.memory_limit_bytes() {
        if peak > limit {
            return Err(CausalError::Resource {
                message: format!(
                    "estimated peak memory {peak} exceeds memory limit {limit}; no chunked path"
                ),
            });
        }
    }
    Ok(())
}
