//! Route lowering: every plan `Study::compile` returns carries a lowered program.
//!
//! [`attach`] lowers one parsed route; [`attach_graph_posterior`] lowers a study
//! that carries a graph posterior. Each cell is registered to a family lowering
//! in `registry`, and every family lowers each layer from its own scientific
//! problem: a query the world algebra cannot state lowers to its typed functional
//! keyed by the query's canonical digest, and an identification that refuses at
//! compile is deferred with a recorded reason. There is no partial lowering.
//!
//! Dispatch refuses a plan without a lowered program (`route_not_lowered`), and
//! compile and one-shot run refuse a context already inside a lowered dispatch
//! (`public_builder_reentry`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) mod cyclic;
mod decision;
mod design_based;
mod discovery;
mod graph_posterior;
mod query_digest;
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
use antecedent_expr::IdentifiedEstimand;
use antecedent_identify::{
    CompiledIdentificationProduct, IdentificationResult, TransportIdentification,
};

pub(crate) use decision::CompileGoal;
#[cfg(test)]
pub(crate) use registry::REGISTERED;
pub(crate) use registry::RouteCell;
pub use registry::{GraphPosteriorCell, PlanRoute};
pub(crate) use static_response::{response_world, response_world_binding};
pub(crate) use structural::theorem_mediation_target;
pub(crate) use temporal::history_window;
#[cfg(test)]
pub(crate) use temporal::{contrast_frame, contrast_outcome, target_contrast};

use query_digest::{query_digest, retention_key};

use crate::Study;
use crate::analysis::IdentificationCache;
use crate::analysis::builder::DataInput;
use crate::analysis::execute::RouteInput;
use crate::analysis::prepared::{
    CachedCpdagIdentification, CachedDbnPosteriorIdentification,
    CachedGraphPosteriorIdentification, CachedPagIdentification, CachedStaticIdentification,
    CachedTemporalClassIdentification, CachedTemporalClassPosteriorIdentification,
    CachedTemporalIdentification,
};
use crate::error::CausalError;
use crate::planner::PhysicalExecutionPlan;
use crate::planner::lowered::{
    DesignShape, FunctionalKind, IdentificationLayer, LoweredProgram, ProblemTarget,
    RetainedIdentification, WidthSource,
};

/// Why a sharp regression-discontinuity route retains no identification.
const RD_IDENTIFIES_PER_EXECUTION: &str =
    "sharp regression discontinuity identifies on each execution";
/// Why a program whose identification refused at compile retains none.
const IDENTIFICATION_REFUSED_AT_COMPILE: &str = "identification refused at compile; preparation \
                                                  identifies and reports the same refusal";

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
        key: retention_key(cx.study),
    })))
}

/// The deferred layer of a program whose identification refused at compile.
pub(crate) const fn identification_refused_at_compile() -> IdentificationLayer {
    IdentificationLayer::Deferred { reason: IDENTIFICATION_REFUSED_AT_COMPILE }
}

/// The typed functional `kind` of the study's query on `outcome`, keyed by the
/// query's canonical digest so two queries of one kind lower to different programs.
pub(crate) fn query_functional(
    study: &Study,
    kind: FunctionalKind,
    outcome: Option<VariableId>,
) -> ProblemTarget {
    ProblemTarget::QueryFunctional { kind, outcome, query: query_digest(&study.query) }
}

/// The identification product `plan`'s lowered program retained for `study`, picked
/// by `pick`. `None` when the program retained none, or retained it for a study whose
/// query, structure or identification options differ from `study`'s.
fn retained_cache<T>(
    study: &Study,
    plan: &PhysicalExecutionPlan,
    pick: impl Fn(&IdentificationCache) -> Option<&Arc<T>>,
) -> Option<Arc<T>> {
    let retained = plan.lowered()?.retained_identification()?;
    if Some(retained.key?) != retention_key(study) {
        return None;
    }
    retained.caches().iter().find_map(pick).map(Arc::clone)
}

/// The static identification `plan` retained for `study`.
pub(crate) fn plan_retained_static(
    study: &Study,
    plan: &PhysicalExecutionPlan,
) -> Option<Arc<CachedStaticIdentification>> {
    retained_cache(study, plan, |cache| match cache {
        IdentificationCache::Static(cache) => Some(cache),
        _ => None,
    })
}

/// The PAG envelope `plan` retained for `study`.
pub(crate) fn plan_retained_pag(
    study: &Study,
    plan: &PhysicalExecutionPlan,
) -> Option<Arc<CachedPagIdentification>> {
    retained_cache(study, plan, |cache| match cache {
        IdentificationCache::Pag(cache) => Some(cache),
        _ => None,
    })
}

/// The CPDAG envelope `plan` retained for `study`.
pub(crate) fn plan_retained_cpdag(
    study: &Study,
    plan: &PhysicalExecutionPlan,
) -> Option<Arc<CachedCpdagIdentification>> {
    retained_cache(study, plan, |cache| match cache {
        IdentificationCache::Cpdag(cache) => Some(cache),
        _ => None,
    })
}

/// The temporal identification by horizon `plan` retained for `study`.
pub(crate) fn plan_retained_temporal(
    study: &Study,
    plan: &PhysicalExecutionPlan,
) -> Option<Arc<CachedTemporalIdentification>> {
    retained_cache(study, plan, |cache| match cache {
        IdentificationCache::Temporal(cache) => Some(cache),
        _ => None,
    })
}

/// The temporal class envelope `plan` retained for `study`.
pub(crate) fn plan_retained_temporal_class(
    study: &Study,
    plan: &PhysicalExecutionPlan,
) -> Option<Arc<CachedTemporalClassIdentification>> {
    retained_cache(study, plan, |cache| match cache {
        IdentificationCache::TemporalClass(cache) => Some(cache),
        _ => None,
    })
}

/// The transport formula and certificate `plan` retained for `study`.
pub(crate) fn plan_retained_transport(
    study: &Study,
    plan: &PhysicalExecutionPlan,
) -> Option<Arc<TransportIdentification>> {
    retained_cache(study, plan, |cache| match cache {
        IdentificationCache::Transport(cache) => Some(cache),
        _ => None,
    })
}

/// The prepared product when there is one, else the product the plan retained.
/// The flag is `true` only for a prepared product: a compiled plan is not a
/// prepared handle, so it publishes the diagnostics a fresh identification does.
fn prepared_or_retained<T>(
    prepared: Option<&Arc<T>>,
    retained: impl FnOnce() -> Option<Arc<T>>,
) -> Option<(Arc<T>, bool)> {
    match prepared {
        Some(cache) => Some((Arc::clone(cache), true)),
        None => retained().map(|cache| (cache, false)),
    }
}

/// Every single-graph executor takes its identification in one order: the
/// prepared cache, then the product `plan`'s lowered program retained for this
/// study (guarded by the retention key), and only then a fresh identification.
/// Each method returns the first two, with `true` for a prepared cache.
impl Study {
    /// Static identification to reuse.
    pub(crate) fn reuse_static_identification(
        &self,
        plan: &PhysicalExecutionPlan,
    ) -> Option<(Arc<CachedStaticIdentification>, bool)> {
        prepared_or_retained(self.identification_cache.as_ref(), || {
            plan_retained_static(self, plan)
        })
    }

    /// [`Self::reuse_static_identification`] for an executor whose own identifier,
    /// estimator or design inputs can differ from the ones preparation resolves: a
    /// retained product is taken only when `same_inputs` confirms it is the claim
    /// the executor would identify. A prepared cache is taken as before.
    pub(crate) fn reuse_static_identification_if(
        &self,
        plan: &PhysicalExecutionPlan,
        same_inputs: impl FnOnce(&CachedStaticIdentification) -> bool,
    ) -> Option<(Arc<CachedStaticIdentification>, bool)> {
        self.reuse_static_identification(plan)
            .filter(|(cache, prepared)| *prepared || same_inputs(cache))
    }

    /// PAG envelope to reuse.
    pub(crate) fn reuse_pag_identification(
        &self,
        plan: &PhysicalExecutionPlan,
    ) -> Option<(Arc<CachedPagIdentification>, bool)> {
        prepared_or_retained(self.pag_identification_cache.as_ref(), || {
            plan_retained_pag(self, plan)
        })
    }

    /// CPDAG envelope to reuse.
    pub(crate) fn reuse_cpdag_identification(
        &self,
        plan: &PhysicalExecutionPlan,
    ) -> Option<(Arc<CachedCpdagIdentification>, bool)> {
        prepared_or_retained(self.cpdag_identification_cache.as_ref(), || {
            plan_retained_cpdag(self, plan)
        })
    }

    /// Temporal identification by horizon to reuse.
    pub(crate) fn reuse_temporal_identification(
        &self,
        plan: &PhysicalExecutionPlan,
    ) -> Option<(Arc<CachedTemporalIdentification>, bool)> {
        prepared_or_retained(self.temporal_identification_cache.as_ref(), || {
            plan_retained_temporal(self, plan)
        })
    }

    /// Temporal class envelope to reuse.
    pub(crate) fn reuse_temporal_class_identification(
        &self,
        plan: &PhysicalExecutionPlan,
    ) -> Option<(Arc<CachedTemporalClassIdentification>, bool)> {
        prepared_or_retained(self.temporal_class_identification_cache.as_ref(), || {
            plan_retained_temporal_class(self, plan)
        })
    }

    /// The identification a design operation freezes: the reused static product
    /// (identified from the design query alone), else `fresh`.
    pub(crate) fn design_identification_or(
        &self,
        plan: &PhysicalExecutionPlan,
        fresh: impl FnOnce() -> (IdentificationResult, IdentifiedEstimand),
    ) -> (IdentificationResult, IdentifiedEstimand) {
        match self.reuse_static_identification(plan) {
            Some((cache, _)) => (cache.identification.clone(), cache.estimand.clone()),
            None => fresh(),
        }
    }

    /// Transport formula and certificate to reuse.
    pub(crate) fn reuse_transport_identification(
        &self,
        plan: &PhysicalExecutionPlan,
    ) -> Option<(Arc<TransportIdentification>, bool)> {
        prepared_or_retained(self.transport_identification_cache.as_ref(), || {
            plan_retained_transport(self, plan)
        })
    }
}

/// The static graph-posterior identification `plan` retained for `study`.
pub(crate) fn plan_retained_graph_posterior(
    study: &Study,
    plan: &PhysicalExecutionPlan,
) -> Option<Arc<CachedGraphPosteriorIdentification>> {
    retained_cache(study, plan, |cache| match cache {
        IdentificationCache::GraphPosterior(cache) => Some(cache),
        _ => None,
    })
}

/// The DBN-posterior identification `plan` retained for `study`.
pub(crate) fn plan_retained_dbn_posterior(
    study: &Study,
    plan: &PhysicalExecutionPlan,
) -> Option<Arc<CachedDbnPosteriorIdentification>> {
    retained_cache(study, plan, |cache| match cache {
        IdentificationCache::DbnPosterior(cache) => Some(cache),
        _ => None,
    })
}

/// The temporal class-posterior identification `plan` retained for `study`.
pub(crate) fn plan_retained_temporal_class_posterior(
    study: &Study,
    plan: &PhysicalExecutionPlan,
) -> Option<Arc<CachedTemporalClassPosteriorIdentification>> {
    retained_cache(study, plan, |cache| match cache {
        IdentificationCache::TemporalClassPosterior(cache) => Some(cache),
        _ => None,
    })
}

/// A refusal for a family lowering handed a cell it does not lower.
fn not_lowered_by(family: &str, cx: &LoweringContext<'_>) -> CausalError {
    crate::compile_reason!(
        "route_not_lowered",
        "the {} lowering does not lower {} for this study",
        family,
        cx.cell.as_str()
    )
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
    Ok(DesignShape::quoted(quote.rows as u64, quote.columns as u64, quote.direct))
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
