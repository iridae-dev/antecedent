//! Route lowering: every plan `Study::compile` returns carries a lowered program.
//!
//! [`attach`] lowers one parsed route; [`attach_graph_posterior`] lowers a study
//! that carries a graph posterior. Each cell is registered to a family lowering
//! in `registry`; a family that has not lowered its own layers yet delegates to
//! the generic lowering, which states the route record, the declared inference
//! and the planner's physical estimate, and marks the program
//! [`LoweringDepth::Generic`].
//!
//! Dispatch refuses a plan without a lowered program (`route_not_lowered`), and
//! compile and one-shot run refuse a context already inside a lowered dispatch
//! (`public_builder_reentry`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod decision;
mod design_based;
mod graph_posterior;
mod registry;
mod static_effect;
mod static_response;
mod structural;
mod temporal;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::Contrast;
use antecedent_identify::CompiledIdentificationProduct;

pub(crate) use decision::CompileGoal;
pub(crate) use registry::{GraphPosteriorCell, RouteCell};
pub(crate) use static_response::{response_world, response_world_binding};
pub(crate) use structural::theorem_mediation_target;
pub(crate) use temporal::history_window;
#[cfg(test)]
pub(crate) use temporal::{
    TemporalContrast, contrast_frame, contrast_outcome, target_contrast, temporal_contrast_target,
};

use crate::analysis::builder::DataInput;
use crate::analysis::compiled_linear::CompiledLinearProgram;
use crate::analysis::execute::RouteInput;
use crate::error::CausalError;
use crate::planner::PhysicalExecutionPlan;
use crate::planner::lowered::{
    ClaimDemand, DesignShape, IdentificationLayer, LoweredParts, LoweredProgram, LoweringDepth,
    PhysicalLayer, ProblemTarget, RetainedIdentification, ScientificProblem, UncertaintyReading,
    WidthSource,
};
use crate::planner::statistical::{DeclaredProgram, StatisticalProgram};
use crate::{EstimatorId, IdentifierId, InferenceMode, Study};

/// Why a sharp regression-discontinuity route retains no identification.
const RD_IDENTIFIES_PER_EXECUTION: &str =
    "sharp regression discontinuity identifies on each execution";

/// Everything a route lowering reads.
pub(crate) struct LoweringContext<'a> {
    /// The study being compiled.
    pub(crate) study: &'a Study,
    /// The cell the study compiles as.
    pub(crate) cell: RouteCell,
    /// The parsed route. `None` for a graph-posterior study.
    pub(crate) input: Option<RouteInput<'a>>,
    /// The physical plan compile built for the route record.
    pub(crate) plan: &'a PhysicalExecutionPlan,
    /// What compile was asked to produce.
    pub(crate) goal: &'a CompileGoal,
    /// Compile context.
    pub(crate) ctx: &'a ExecutionContext,
}

/// Lower the parsed route `input` for `goal` and attach the program to `plan`.
///
/// # Errors
///
/// The route's lowering refuses, the statistical program cannot supply the
/// carrier a decision goal demands, or compile was cancelled.
pub(crate) fn attach(
    study: &Study,
    input: RouteInput<'_>,
    plan: &mut PhysicalExecutionPlan,
    goal: &CompileGoal,
    ctx: &ExecutionContext,
) -> Result<(), CausalError> {
    let lowered = lower(&LoweringContext {
        study,
        cell: RouteCell::of(&input),
        input: Some(input),
        plan: &*plan,
        goal,
        ctx,
    })?;
    install(plan, lowered, ctx)
}

/// Lower a graph-posterior study for `goal` and attach the program to `plan`.
///
/// # Errors
///
/// Graph-posterior compile has no cell for the study's data and query, the
/// lowering refuses, the statistical program cannot supply the carrier a
/// decision goal demands, or compile was cancelled.
pub(crate) fn attach_graph_posterior(
    study: &Study,
    plan: &mut PhysicalExecutionPlan,
    goal: &CompileGoal,
    ctx: &ExecutionContext,
) -> Result<(), CausalError> {
    let cell = GraphPosteriorCell::classify(study).ok_or(CausalError::Unsupported {
        message: "graph-posterior compile has no route cell for this data and query",
    })?;
    let lowered = lower(&LoweringContext {
        study,
        cell: RouteCell::GraphPosterior(cell),
        input: None,
        plan: &*plan,
        goal,
        ctx,
    })?;
    install(plan, lowered, ctx)
}

/// The cell `study` executes as now, when it classifies as one.
pub(crate) fn expected_cell(study: &Study) -> Option<RouteCell> {
    if study.graph_posterior.is_some() {
        return GraphPosteriorCell::classify(study).map(RouteCell::GraphPosterior);
    }
    study.route_input_classified().map(|input| RouteCell::of(&input))
}

/// Refuse compile and one-shot run inside the dispatch of a lowered program.
///
/// # Errors
///
/// `public_builder_reentry` when `ctx` carries a lowered-dispatch scope.
pub(crate) fn refuse_reentry(ctx: &ExecutionContext) -> Result<(), CausalError> {
    if ctx.lowered_dispatch().is_some() {
        return Err(crate::unsupported_reason!(
            "public_builder_reentry",
            "a lowered dispatch executes its retained program; it does not compile, prepare or \
             run another study"
        ));
    }
    Ok(())
}

/// Attach `lowered` to `plan`. A quoted design re-plans the physical record from
/// the real width; the retained linear operation has already sized its own plan.
fn install(
    plan: &mut PhysicalExecutionPlan,
    lowered: LoweredProgram,
    ctx: &ExecutionContext,
) -> Result<(), CausalError> {
    let design = lowered.physical().design();
    if design.source() == WidthSource::Quoted && plan.linear.is_none() {
        plan.requote(design, ctx)?;
    }
    plan.lowered = Some(Arc::new(lowered));
    Ok(())
}

/// Identify at compile what preparation would identify, and retain it with the
/// program, so preparation installs it instead of identifying again.
///
/// `product` is the route's checked identification product bound to its
/// scientific target, when the route builds one.
///
/// # Errors
///
/// Identification refuses, as preparation would have.
pub(crate) fn retain_identification(
    cx: &LoweringContext<'_>,
    product: Option<CompiledIdentificationProduct>,
) -> Result<IdentificationLayer, CausalError> {
    let caches = cx.study.identify_caches(cx.plan, cx.ctx)?;
    Ok(IdentificationLayer::Retained(Arc::new(RetainedIdentification {
        product,
        caches: Arc::from(caches),
    })))
}

/// The real width of the linear adjustment design `[1 | T | Z…]` over `adjustment`
/// under `contrasts`, quoted from the study's tabular data. The attached program
/// re-plans the physical record from this shape.
///
/// # Errors
///
/// The study's data is not tabular, an adjustment column is missing, or a
/// categorical adjustment variable has no contrast coding.
pub(crate) fn quote_adjustment_width(
    cx: &LoweringContext<'_>,
    adjustment: &[VariableId],
    contrasts: &[(VariableId, Contrast)],
) -> Result<DesignShape, CausalError> {
    let DataInput::Tabular(data) = &cx.study.data else {
        return Err(CausalError::Compile {
            message: "an adjustment design width is quoted from tabular data".into(),
        });
    };
    let quote =
        antecedent_estimate::design_width::quote_adjustment_design(data, adjustment, contrasts)
            .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    Ok(DesignShape {
        rows: quote.rows as u64,
        columns: quote.columns as u64,
        source: WidthSource::Quoted,
        direct: quote.direct,
    })
}

fn lower(cx: &LoweringContext<'_>) -> Result<LoweredProgram, CausalError> {
    if cx.ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: "compile" });
    }
    let parsed = cx.input.as_ref().map(RouteCell::of);
    if let Some(parsed) = parsed.filter(|parsed| *parsed != cx.cell) {
        return Err(crate::compile_reason!(
            "route_not_lowered",
            "the lowering context names {} for a route that parses as {}",
            cx.cell.as_str(),
            parsed.as_str()
        ));
    }
    registry::lower_cell(cx)
}

/// The lowering every family delegates to until it lowers its own layers.
///
/// Layer one states the route record (and the checked target of a retained linear
/// operation). Layer two retains the identification product only where compile
/// already identified. Layer three is the declared inference. Layer five is the
/// planner's estimate.
fn generic(cx: &LoweringContext<'_>) -> Result<LoweredProgram, CausalError> {
    let study = cx.study;
    let plan = cx.plan;
    let record = &plan.logical.record;
    let linear = plan.linear.as_deref().map(|linear| linear.program.as_ref());
    let identifier =
        record.identifier.as_deref().and_then(|name| name.parse::<IdentifierId>().ok());
    let estimator = record.estimator.as_deref().and_then(|name| name.parse::<EstimatorId>().ok());
    let source_digest = linear.map(CompiledLinearProgram::source_digest);
    let scientific = ScientificProblem {
        target: linear.map_or(ProblemTarget::Unlowered, |program| {
            ProblemTarget::Target(program.target().clone())
        }),
        plan_id: Arc::clone(&record.plan_id),
        data: record.data_classification,
        graph_class: study.graph.class().as_str(),
        query_variables: Arc::clone(&record.query_variables),
        source_digest,
        support_coordinate: crate::analysis::contract::matrix_coordinate(study).map(Arc::from),
        licensed: matches!(study.support_status, Some(crate::support::CellStatus::Licensed)),
    };
    let identification = match (linear, estimator) {
        (Some(program), _) => retain_identification(cx, Some(program.identification().clone()))?,
        (None, Some(EstimatorId::RdSharp | EstimatorId::BayesianRdLocalLinear)) => {
            IdentificationLayer::PerExecution { reason: RD_IDENTIFIES_PER_EXECUTION }
        }
        (None, _) => IdentificationLayer::Deferred,
    };
    let design = match plan.linear.as_deref() {
        Some(linear) => quote_adjustment_width(
            cx,
            &linear.identification.estimand.adjustment_set,
            &linear.fitter.contrasts,
        )?,
        None => DesignShape {
            rows: plan.logical.row_count_hint,
            columns: plan.logical.design_columns_hint,
            source: WidthSource::SchemaBound,
            direct: false,
        },
    };
    let statistical = StatisticalProgram::declared(DeclaredProgram {
        identifier,
        estimator,
        inference: &study.inference,
        bootstrap_replicates: study.bootstrap_replicates,
        sampling_design: study.sampling_design,
        design,
        operators: linear.map_or_else(Vec::new, |program| program.operators().to_vec()),
    })?;
    let physical = PhysicalLayer {
        operators: Arc::from(
            linear.map_or_else(Vec::new, |program| program.physical_operators().to_vec()),
        ),
        design,
    };
    let claim = ClaimDemand {
        uncertainty: match study.inference {
            InferenceMode::Frequentist => UncertaintyReading::Sampling,
            InferenceMode::Bayesian(_) => UncertaintyReading::Posterior,
        },
        kind: None,
    };
    let evidence = decision::evidence_program(cx, source_digest, &statistical)?;
    Ok(LoweredParts {
        cell: cx.cell,
        depth: LoweringDepth::Generic,
        scientific,
        identification,
        statistical,
        evidence,
        physical,
        claim,
    }
    .finish())
}
