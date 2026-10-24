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
    BackendSupport, PhysicalAlternative, RouteChoice, RouteWork,
};
use antecedent_core::route_memory::{
    RouteBudget, RouteMemoryError, RouteMemoryEstimate, RouteMemoryProfile,
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
        let columns = self.design_columns_hint.max(1);
        let requested_workers =
            ctx.parallelism.max_threads.get().min(ctx.admission().thread_budget());
        let workers = if requested_workers <= 1 { 0 } else { requested_workers };

        // The starting record is quoted from the planner's own row and column hints through
        // the same route-memory components a lowered program re-quotes with, and admitted
        // against the caller's limit before any buffer exists.
        let profile = RouteMemoryProfile::design(n_rows, columns)
            .workers(u64::from(requested_workers.max(1)));
        let memory = admit_memory(profile, ctx)?;
        let copy_bytes = profile.design_bytes().map_err(memory_refusal)?; // design matrix is CopiedContiguous
        let workspace = memory.scratch_copy_bytes();
        let peak = memory.total();
        let route_choice = RouteChoice::select(
            RouteWork::new(n_rows, columns, 1),
            BackendSupport::materialized_only(),
            RouteBudget::from_limit(ctx.memory_limit_bytes()),
        )
        .map_err(|error| CausalError::Resource { message: error.to_string() })?;
        let choice = *route_choice.choice();

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
            route_choice: Some(route_choice),
            memory_quote: Some(memory),
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
    /// The quote behind [`Self::choice`]: the cells and resident bytes each
    /// alternative was costed at, under the budget it was compared with.
    pub route_choice: Option<RouteChoice>,
    /// The route-memory components the record's peak and workspace were quoted
    /// from, admitted against the caller's limit before any allocation.
    pub memory_quote: Option<RouteMemoryEstimate>,
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
    ///
    /// The linear adjustment reads its design `1 + replicates` times, so a direct
    /// prepared view (when every design column is numeric) beats a materialized
    /// design by the staging copy it avoids, and a streamed reduction is not
    /// implemented for it: the estimator requires retained rows. Both are costed
    /// from the prepared design's own shape, and an alternative whose resident
    /// bytes exceed the caller's limit is rejected with the missing resource.
    pub(crate) fn bind_prepared_linear_views(
        &mut self,
        direct_eligible: bool,
        rows: u64,
        columns: u64,
        replicates: u32,
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        let support = BackendSupport::materialized_only()
            .with(PhysicalAlternative::DirectPreparedNumericViews, direct_eligible);
        let routed = RouteChoice::select(
            RouteWork::new(rows, columns, u64::from(replicates).saturating_add(1)),
            support,
            RouteBudget::from_limit(ctx.memory_limit_bytes()),
        )
        .map_err(|error| CausalError::Resource { message: error.to_string() })?;
        self.choice = *routed.choice();
        self.route_choice = Some(routed);
        self.record.materializations =
            Arc::from([(Arc::from("design.matrix"), BufferMaterialization::CopiedContiguous)]);
        Ok(())
    }

    /// Choose the streamed reduction for a linear adjustment whose resident design does not
    /// fit: the rows are read in batches straight from the source columns, no design is
    /// built, and the plan records the design as chunked.
    ///
    /// # Errors
    ///
    /// The limit cannot hold even the streamed reduction.
    pub(crate) fn bind_streamed_linear_views(
        &mut self,
        columns: u64,
        rows: u64,
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        let support = BackendSupport::none().with(PhysicalAlternative::StreamedReduction, true);
        let routed = RouteChoice::select(
            RouteWork::new(rows, columns, 2),
            support,
            RouteBudget::from_limit(ctx.memory_limit_bytes()),
        )
        .map_err(|error| CausalError::Resource { message: error.to_string() })?;
        self.choice = *routed.choice();
        self.route_choice = Some(routed);
        self.logical.row_count_hint = rows;
        self.logical.design_columns_hint = columns;
        self.record.materializations = Arc::from([
            (Arc::from("source.columns"), BufferMaterialization::Borrowed),
            (Arc::from("design.matrix"), BufferMaterialization::Chunked),
        ]);
        Ok(())
    }

    /// Whether the plan reads its design as a streamed reduction.
    #[must_use]
    pub(crate) fn streams_design(&self) -> bool {
        self.choice.selected() == PhysicalAlternative::StreamedReduction
    }

    /// Re-plan the design bytes and materialization from a quoted design shape, and
    /// admit the re-planned memory under the caller's limit.
    ///
    /// A direct design borrows its numeric source columns and copies once into the
    /// final design; a materialized design copies the selected and expanded columns
    /// before assembling the final design. Every memory component is quoted from the
    /// design's shape ([`RouteMemoryProfile`]) and the sum is refused before any
    /// buffer is allocated.
    ///
    /// # Errors
    ///
    /// The re-planned memory exceeds the limit, or the size arithmetic overflows.
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
        let columns = columns.max(1);
        let (source, copies) = if design.direct() {
            (BufferMaterialization::Borrowed, 1)
        } else {
            (BufferMaterialization::CopiedContiguous, 2)
        };
        let profile = RouteMemoryProfile::design(rows, columns)
            .expanded(!design.direct())
            .workers(self.quoted_workers(ctx));
        let memory = admit_memory(profile, ctx)?;
        let design_bytes = profile.design_bytes().map_err(memory_refusal)?;
        let copy_bytes = design_bytes
            .checked_mul(copies)
            .ok_or_else(|| CausalError::Resource { message: "copy estimate overflow".into() })?;
        self.logical.row_count_hint = rows;
        self.logical.design_columns_hint = columns;
        self.record.batch_size = Some(usize::try_from(rows).unwrap_or(usize::MAX));
        self.record.workspace_bytes = Some(memory.scratch_copy_bytes());
        self.record.estimated_peak_memory_bytes = Some(memory.total());
        self.record.estimated_copy_bytes = Some(copy_bytes);
        self.record.materializations = Arc::from([
            (Arc::from("source.columns"), source),
            (Arc::from("design.matrix"), BufferMaterialization::CopiedContiguous),
        ]);
        self.memory_quote = Some(memory);
        Ok(())
    }

    /// Re-quote the plan's memory and physical alternatives from its attached lowered
    /// program, and admit the quote under the caller's limit before anything is allocated.
    ///
    /// The statistical program supplies what the route will hold beyond its design: the
    /// cross-fitting folds and nuisance learners (a training copy per fold and learner,
    /// and out-of-fold predictions), the resampling replicates and posterior draws it
    /// retains, and the workers each holding a solver scratch copy. The physical
    /// alternatives are costed from the same rows, columns and number of passes the
    /// kernel reads the design (one, plus one per fold and per replicate): a direct
    /// prepared view is implemented only where every design column is numeric, and a
    /// streamed reduction only where the route reduces without retained rows, which no
    /// planned route does.
    ///
    /// A plan whose linear operation sized its own record from the prepared design keeps
    /// that quote.
    ///
    /// # Errors
    ///
    /// The quote exceeds the limit, no alternative is both implemented and within the
    /// limit, or the size arithmetic overflows.
    pub(crate) fn quote_route_memory(&mut self, ctx: &ExecutionContext) -> Result<(), CausalError> {
        if self.linear.is_some() {
            return Ok(());
        }
        let Some(lowered) = self.lowered.clone() else {
            return Ok(());
        };
        let statistical = lowered.statistical();
        let design = lowered.physical().design();
        let quoted = design.source() == super::lowered::WidthSource::Quoted;
        let (rows, columns) = match (design.rows(), design.columns()) {
            (Some(rows), Some(columns)) => (rows.max(1), columns.max(1)),
            _ => (self.logical.row_count_hint.max(1), self.logical.design_columns_hint.max(1)),
        };
        let direct = quoted && design.direct();
        let folds = statistical.folds().map_or(0, |plan| count(plan.folds()));
        let learners = count(statistical.models().len()).max(1);
        let replicates = statistical.resampling().map_or(0, |plan| u64::from(plan.replicates()));
        let draws = statistical.posterior().map_or(0, |plan| count(plan.draws()));
        let profile = RouteMemoryProfile::design(rows, columns)
            .expanded(quoted && !direct)
            .cross_fitting(folds, learners)
            .resampling(replicates)
            .posterior_draws(draws)
            .workers(self.quoted_workers(ctx));
        let memory = admit_memory(profile, ctx)?;
        let fold_passes = if folds >= 2 { folds } else { 0 };
        let passes = 1_u64.saturating_add(fold_passes).saturating_add(replicates);
        let support = BackendSupport::materialized_only()
            .with(PhysicalAlternative::DirectPreparedNumericViews, direct);
        let routed = RouteChoice::select(
            RouteWork::new(rows, columns, passes),
            support,
            RouteBudget::from_limit(ctx.memory_limit_bytes()),
        )
        .map_err(|error| CausalError::Resource { message: error.to_string() })?;
        self.record.workspace_bytes = Some(memory.scratch_copy_bytes());
        self.record.estimated_peak_memory_bytes = Some(memory.total());
        self.choice = *routed.choice();
        self.route_choice = Some(routed);
        self.memory_quote = Some(memory);
        Ok(())
    }

    /// Byte budget estimator preparation is admitted against: the caller's limit, or the
    /// largest budget when there is none, so the preparation receipt reports the bytes
    /// it admitted and the budget it was compared with. Execution wraps estimator preparation
    /// in [`antecedent_estimate::prepare::with_preparation_budget`] with this budget.
    pub(crate) fn preparation_budget(
        ctx: &ExecutionContext,
    ) -> antecedent_data::materialize::PreparationBudget {
        antecedent_data::materialize::PreparationBudget::new(
            RouteBudget::from_limit(ctx.memory_limit_bytes()).bytes(),
        )
    }

    /// Workers that each hold a solver scratch copy: the context's threads under the pool.
    fn quoted_workers(&self, ctx: &ExecutionContext) -> u64 {
        u64::from(ctx.parallelism.max_threads.get().min(ctx.admission().thread_budget()).max(1))
    }
}

/// Refuse a route-memory quote that exceeds `ctx`'s limit, or return the admitted quote.
///
/// Nothing is allocated either way: admission compares the declared sum with the budget.
fn admit_memory(
    profile: RouteMemoryProfile,
    ctx: &ExecutionContext,
) -> Result<RouteMemoryEstimate, CausalError> {
    profile
        .admit(RouteBudget::from_limit(ctx.memory_limit_bytes()))
        .map(antecedent_core::route_memory::AdmittedRouteMemory::estimate)
        .map_err(memory_refusal)
}

fn memory_refusal(error: RouteMemoryError) -> CausalError {
    let message = match error {
        RouteMemoryError::ExceedsBudget { estimated_bytes, budget_bytes } => format!(
            "estimated peak memory {estimated_bytes} exceeds memory limit {budget_bytes}; no \
             chunked path; next attempt: lower the declared copies, workers, or retained outputs"
        ),
        other => other.to_string(),
    };
    CausalError::Resource { message }
}

fn count(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

impl crate::StudyResult {
    /// Measured facts of the run that produced this result, when it was observed.
    ///
    /// These are what the engine counted while the analysis executed: scans, copies,
    /// fits, cache hits, retained output, admitted resources, spill, the backend and
    /// how the run ended. They are kept apart from the plan's quotes
    /// ([`Self::physical_plan`]).
    #[must_use]
    pub const fn execution_facts(
        &self,
    ) -> Option<&antecedent_core::execution_facts::ExecutionFacts> {
        self.performance.native_execution.as_ref()
    }

    /// Measured facts, admission reservations and the plan's estimates of this run,
    /// each quantity labelled with its basis so none is read as another.
    #[must_use]
    pub fn execution_accounting(
        &self,
    ) -> Option<antecedent_core::execution_facts::ExecutionAccounting> {
        use antecedent_core::execution_facts::PlannedResources;
        let plan = &self.physical_plan;
        self.execution_facts().map(|facts| {
            facts.accounting(Some(PlannedResources {
                peak_memory_bytes: plan.estimated_peak_memory_bytes,
                copy_bytes: plan.estimated_copy_bytes,
                workspace_bytes: plan.workspace_bytes,
                worker_threads: Some(u64::from(plan.worker_threads)),
            }))
        })
    }
}
