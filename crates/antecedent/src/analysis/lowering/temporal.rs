//! Lowering of the temporal cells: series effect, response and mediation on a
//! temporal DAG or class, temporal cross-world queries, multi-environment effects,
//! and panel effects and responses.
//!
//! Each cell lowers every layer from its own query:
//!
//! * layer one is the world-indexed target the query asks for: a pulse or
//!   sustained contrast at the query horizon for an effect, a dose-by-horizon
//!   surface for a response (one scheduled world per horizon for a single
//!   non-grid intervention), and one theorem-scoped mediation target per requested
//!   horizon for a mediation on a temporal DAG. The history window is the declared
//!   cap, or the planner's default resolved to the depth the retained
//!   identification unfolded, so the identity pins what the planner used. A query
//!   the target constructors cannot state as worlds lowers to its typed functional,
//!   keyed by the canonical digest of the query;
//! * layer two retains the identification preparation would compute; cross-world
//!   queries identify on each execution;
//! * layer three resamples the unit the study's sampling design declares (or the
//!   unit its modality implies), and splits a series with the declared embargo;
//! * layer five does not quote a design: each horizon or completion assembles its
//!   lagged design from the unfolded sample at execution.
//!
//! Mediation under a temporal ADMG or PAG identifies on its window at each execution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::ClaimKind;
use antecedent_core::{
    AssumptionSet, CausalQuery, CausalSchema, Conditioning, DoseTimeSurface, GraphFamily,
    HistoryWindow, InferenceSelection, InitialLaw, Intervention, InterventionContrast,
    InterventionWorld, LikelihoodModel, MediationQuery, ModelAssumptions, ObservationUnit,
    OrdinaryBayesianLikelihood, OutcomeQuantity, PartialIdentificationEnvelope, Prior,
    QuantityUnits, ResponseFunctional, ResponseQuery, ScientificTarget, ScientificTargetError,
    ScientificWorldId, TargetForm, TargetPopulation, TemporalEffectQuery, TemporalPolicy,
    TimeFrame, Value, VariableId, WorldAction, WorldBinding, WorldCollection,
};
use antecedent_data::TableView;

use super::LoweringContext;
use crate::accepted::GraphClass;
use crate::analysis::IdentificationCache;
use crate::analysis::builder::DataInput;
use crate::analysis::execute::{CheckedTemporalMediationOperation, RouteInput};
use crate::analysis::prepared::{
    CachedTemporalClassIdentification, CachedTemporalHorizonIdentification,
    CachedTemporalIdentification, conditional_class_proof_is_complete,
};
use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, DesignShape, FunctionalKind, IdentificationLayer, LoweredParts, LoweredProgram,
    LoweringDepth, PhysicalLayer, ProblemTarget, ScientificProblem, UncertaintyReading,
    linear_gaussian_physical_operators, linear_gaussian_program_operators,
};
use crate::planner::statistical::{
    DeclaredProgram, FoldPlan, ModelRole, ModelSlot, PosteriorAlgorithm, PosteriorPlan,
    Realization, ResampleUnit, ResamplingMethod, ResamplingPlan, StatisticalProgram,
};

/// Prior of the joint block-weight posterior over a temporal response surface.
const JOINT_BLOCK_WEIGHT_PRIOR: &str = "dirichlet_block_weights(1)";
/// Backend of that posterior: weighted least squares of every horizon under shared block weights.
const JOINT_BLOCK_WEIGHT_BACKEND: &str = "temporal.response.block_weight_surface";
use crate::{EstimatorId, IdentifierId, InferenceMode, Study};

#[cfg(test)]
#[path = "temporal_tests.rs"]
mod tests;

const CONTROL_WORLD: u32 = 0;
const ACTIVE_WORLD: u32 = 1;
/// Observation unit of a single series: one lag window.
const SERIES_UNIT: &str = "lag_window";
/// Observation unit of a pooled panel or multi-environment fit: one unit's lag window.
const POOLED_UNIT: &str = "unit_lag_window";
/// Model family when the route record names no estimator.
const UNNAMED_ESTIMATOR: &str = "unnamed";
/// A discovery/estimation split is two contiguous windows.
const SPLIT_WINDOWS: usize = 2;

const DAG_DESIGN_AT_EXECUTION: &str =
    "the lagged adjustment design is assembled from the unfolded sample at execution";
const WINDOW_DESIGN_AT_EXECUTION: &str = "the lag-aligned window table of the unrolled temporal \
     ADMG is assembled from the series at execution; the discrete plug-in fits no design matrix";
const CLASS_DESIGN_AT_EXECUTION: &str =
    "each completion assembles its own lagged design from the unfolded sample at execution";
const RESPONSE_DESIGN_AT_EXECUTION: &str =
    "each horizon assembles its own lagged design from the unfolded sample at execution";
const MEDIATION_DESIGN_AT_EXECUTION: &str =
    "each horizon assembles its mediator and outcome designs from the unfolded sample at execution";
const CROSS_WORLD_DESIGN_AT_EXECUTION: &str =
    "the cross-world probe identifies and fits its mechanisms at execution";
/// Why a temporal cross-world route retains no identification: the probe did not identify.
const CROSS_WORLD_IDENTIFIES_PER_EXECUTION: &str = "the temporal cross-world probe did not identify at compile; execution identifies it and \
     reports the refusal";
/// Why a temporal ADMG mediation route retains no identification.
const WINDOW_MEDIATION_IDENTIFIES_PER_EXECUTION: &str =
    "mediation under a temporal ADMG or PAG identifies on its window at each execution";

/// Lower one cell of this family.
pub(super) fn lower(cx: &LoweringContext<'_>) -> Result<LoweredProgram, CausalError> {
    let Some(input) = cx.input else {
        return Err(not_temporal(cx));
    };
    match input {
        RouteInput::TemporalEffectDag { query, .. }
        | RouteInput::TemporalEffectClass { query, .. }
        | RouteInput::TemporalEffectAdmg { query, .. }
        | RouteInput::PanelTemporalEffectDag { query, .. }
        | RouteInput::PanelTemporalEffectClass { query, .. }
        | RouteInput::MultiEnvTemporalEffect { query, .. } => lower_effect(cx, query),
        RouteInput::TemporalResponseDag { query, .. }
        | RouteInput::TemporalResponseClass { query, .. }
        | RouteInput::TemporalResponseAdmg { query, .. }
        | RouteInput::PanelTemporalResponseDag { query, .. }
        | RouteInput::PanelTemporalResponseClass { query, .. } => lower_response(cx, query),
        RouteInput::TemporalMediationDag { query, .. }
        | RouteInput::TemporalMediationCpdag { query, .. } => lower_mediation(cx, query),
        RouteInput::TemporalMediationAdmg { query, .. }
        | RouteInput::TemporalMediationPag { query, .. } => lower_mediation_window(cx, query),
        RouteInput::TemporalCrossWorldDag { .. } | RouteInput::TemporalCrossWorldClass { .. } => {
            lower_cross_world(cx)
        }
        _ => Err(not_temporal(cx)),
    }
}

fn not_temporal(cx: &LoweringContext<'_>) -> CausalError {
    CausalError::Compile { message: format!("{} is not a temporal route cell", cx.cell.as_str()) }
}

// ---------------------------------------------------------------------------
// Production target builder
// ---------------------------------------------------------------------------

/// One temporal contrast `E[Y_h(active)] - E[Y_h(control)]` under a pulse or
/// sustained schedule, at one horizon, over one history window.
pub(crate) struct TemporalContrast<'a> {
    /// Control intervention on the treatment.
    pub(crate) control: &'a Intervention,
    /// Active intervention on the treatment.
    pub(crate) active: &'a Intervention,
    /// Outcome whose mean is contrasted.
    pub(crate) outcome: VariableId,
    /// Pulse or sustained schedule both worlds share.
    pub(crate) policy: &'a TemporalPolicy,
    /// Steps after the policy origin.
    pub(crate) horizon_steps: u32,
    /// Retained history: declared, or the planner's resolved cap.
    pub(crate) history: HistoryWindow,
    /// Population both worlds bind.
    pub(crate) population: &'a TargetPopulation,
    /// Effect modifiers the contrast is conditioned on. Empty for an unconditional effect.
    pub(crate) modifiers: &'a [VariableId],
    /// Units of the outcome.
    pub(crate) units: QuantityUnits,
    /// Observation unit label.
    pub(crate) unit: &'static str,
    /// Graph family the target is stated under.
    pub(crate) graph: GraphFamily,
    /// Assumptions the identification required.
    pub(crate) assumptions: AssumptionSet,
    /// Inference label. The target identity does not read it.
    pub(crate) inference: InferenceSelection,
}

impl<'a> TemporalContrast<'a> {
    /// The contrast `query` asks for over `history`, on one series under a
    /// temporal DAG, with no recorded assumptions.
    pub(crate) fn of_query(
        query: &'a TemporalEffectQuery,
        history: HistoryWindow,
        units: QuantityUnits,
    ) -> Self {
        Self {
            control: &query.control,
            active: &query.active,
            outcome: query.outcome,
            policy: &query.policy,
            horizon_steps: query.horizon_steps,
            history,
            population: &query.target_population,
            modifiers: &query.effect_modifiers,
            units,
            unit: SERIES_UNIT,
            graph: GraphFamily::TemporalDag,
            assumptions: AssumptionSet::new(),
            inference: InferenceSelection::Frequentist,
        }
    }
}

/// History retained by a temporal target: the declared cap, or the planner's
/// default resolved to the `planner_used` lags it unfolded.
pub(crate) const fn history_window(declared: Option<u32>, planner_used: u32) -> HistoryWindow {
    planned_history(declared, Some(planner_used))
}

const fn planned_history(declared: Option<u32>, planner_used: Option<u32>) -> HistoryWindow {
    match (declared, planner_used) {
        (Some(lags), _) => HistoryWindow::max_lag(lags),
        (None, Some(used)) => HistoryWindow::planner_default().resolve(used),
        (None, None) => HistoryWindow::planner_default(),
    }
}

/// The scientific target of one temporal contrast.
///
/// # Errors
///
/// The worlds are not a coherent contrast: a callback action, a schedule that
/// reaches outside the horizon, conditioning on the outcome, or interventions on
/// different coordinates.
pub(crate) fn temporal_contrast_target(
    spec: TemporalContrast<'_>,
) -> Result<ScientificTarget, CausalError> {
    contrast_target(spec).map_err(target_error)
}

/// The contrast a target asks for, when it asks for one.
pub(crate) fn target_contrast(target: &ScientificTarget) -> Option<&InterventionContrast> {
    match target.form() {
        TargetForm::Contrast(contrast) => Some(contrast),
        _ => None,
    }
}

/// Decision time, horizon, history and initial law the contrasted worlds share.
pub(crate) fn contrast_frame(contrast: &InterventionContrast) -> Option<&TimeFrame> {
    contrast.worlds().worlds().first().map(InterventionWorld::time)
}

/// Outcome quantity the contrasted worlds bind.
pub(crate) fn contrast_outcome(contrast: &InterventionContrast) -> Option<&OutcomeQuantity> {
    contrast.worlds().worlds().first().map(|world| world.binding().outcome())
}

fn contrast_target(spec: TemporalContrast<'_>) -> Result<ScientificTarget, ScientificTargetError> {
    let conditioning = if spec.modifiers.is_empty() {
        Conditioning::unconditional()
    } else {
        Conditioning::on(spec.modifiers.iter().copied())
    };
    let binding = WorldBinding::new(
        spec.population.clone(),
        ObservationUnit::new(spec.unit)?,
        OutcomeQuantity::mean(spec.outcome, spec.units)?,
        conditioning,
    )?;
    let time = TimeFrame::temporal(
        spec.horizon_steps,
        spec.history,
        InitialLaw::contemporaneous_observed(),
    )?;
    let control = scheduled_world(CONTROL_WORLD, spec.control, spec.policy, true, &binding, &time)?;
    let active = scheduled_world(ACTIVE_WORLD, spec.active, spec.policy, false, &binding, &time)?;
    let contrast = InterventionContrast::difference(
        WorldCollection::new([control, active])?,
        ScientificWorldId::from_raw(ACTIVE_WORLD),
        ScientificWorldId::from_raw(CONTROL_WORLD),
        ScientificWorldId::from_raw(CONTROL_WORLD),
    )?;
    Ok(ScientificTarget::new(
        TargetForm::Contrast(contrast),
        ModelAssumptions::new(spec.graph, spec.assumptions),
        spec.inference,
    ))
}

fn scheduled_world(
    raw: u32,
    intervention: &Intervention,
    policy: &TemporalPolicy,
    reference: bool,
    binding: &WorldBinding,
    time: &TimeFrame,
) -> Result<InterventionWorld, ScientificTargetError> {
    InterventionWorld::new(
        ScientificWorldId::from_raw(raw),
        WorldAction::schedule(intervention.clone(), policy.clone())?,
        reference,
        binding.clone(),
        time.clone(),
    )
}

/// A rectangular dose-by-horizon grid of hard-set doses under one schedule.
struct DoseTimeGrid<'a> {
    dose: VariableId,
    levels: &'a [f64],
    horizons: &'a [u32],
    policy: &'a TemporalPolicy,
    history: HistoryWindow,
}

fn dose_time_target(
    grid: &DoseTimeGrid<'_>,
    binding: &WorldBinding,
    assumptions: ModelAssumptions,
    inference: InferenceSelection,
) -> Result<ScientificTarget, ScientificTargetError> {
    let mut worlds = Vec::with_capacity(grid.levels.len() * grid.horizons.len());
    let mut cells = Vec::with_capacity(worlds.capacity());
    for &level in grid.levels {
        for &horizon in grid.horizons {
            let raw = u32::try_from(cells.len()).map_err(|_| {
                ScientificTargetError::NotADoseTimeGrid {
                    reason: antecedent_core::DoseGridFault::TooLarge,
                }
            })?;
            let id = ScientificWorldId::from_raw(raw);
            let time =
                TimeFrame::temporal(horizon, grid.history, InitialLaw::contemporaneous_observed())?;
            let action = WorldAction::schedule(
                Intervention::set(grid.dose, Value::f64(level)),
                grid.policy.clone(),
            )?;
            worlds.push(InterventionWorld::new(id, action, false, binding.clone(), time)?);
            cells.push(id);
        }
    }
    let surface = DoseTimeSurface::new(
        WorldCollection::new(worlds)?,
        grid.dose,
        grid.levels.iter().copied(),
        grid.horizons.iter().copied(),
        cells,
    )?;
    Ok(ScientificTarget::new(TargetForm::DoseTime(surface), assumptions, inference))
}

#[allow(clippy::needless_pass_by_value, reason = "Result::map_err consumes its error")]
fn target_error(error: ScientificTargetError) -> CausalError {
    CausalError::Compile { message: format!("temporal scientific target: {error}") }
}

// ---------------------------------------------------------------------------
// Cells
// ---------------------------------------------------------------------------

/// Pulse or sustained effect on one series, a panel, or several environments,
/// under a temporal DAG or class.
fn lower_effect(
    cx: &LoweringContext<'_>,
    query: &TemporalEffectQuery,
) -> Result<LoweredProgram, CausalError> {
    let identification = retained(cx)?;
    let dag = temporal_cache(&identification).and_then(|cache| cache.get(query.horizon_steps));
    let class = class_cache(&identification);
    let planner_used = match (dag, class) {
        (Some(entry), _) => Some(entry.indexer.history()),
        (None, Some(class)) => class_history(class, false),
        (None, None) => None,
    };
    let history = planned_history(query.max_history_lag, planner_used);
    let spec = TemporalContrast {
        unit: observation_unit(cx.study),
        graph: graph_family(cx.study),
        assumptions: dag.map_or_else(AssumptionSet::new, |entry| {
            entry.identification.required_assumptions.clone()
        }),
        inference: selection(cx.study),
        ..TemporalContrast::of_query(query, history, outcome_units(cx.study, query.outcome)?)
    };
    let target = match temporal_contrast_target(spec) {
        Ok(target) => {
            if !pins(&target, history, query.outcome) {
                return Err(CausalError::Compile {
                    message: "temporal contrast target does not pin the planner's history and \
                              outcome"
                        .into(),
                });
            }
            ProblemTarget::Target(target)
        }
        // The target constructors refuse the query as a contrast of two scheduled
        // worlds (a callback action, a schedule outside the horizon, a zero horizon);
        // it lowers to the typed contrast functional of this query.
        Err(_) => super::query_functional(
            cx.study,
            FunctionalKind::InterventionContrast,
            Some(query.outcome),
        ),
    };
    let mut inputs = vec![query.treatment];
    inputs.extend(query.effect_modifiers.iter().copied());
    if let Some(entry) = dag {
        inputs.extend(adjustment_variables(entry)?);
    }
    let models = vec![slot(ModelRole::Outcome, estimator_family(cx), inputs, query.outcome)];
    let is_class = is_class(cx.study);
    assemble(
        cx,
        TemporalLowering {
            target,
            envelope: is_class && !class.is_some_and(class_point_identified),
            identification,
            models,
            claim: ClaimKind::Point,
            width: if cx.study.graph.class() == GraphClass::TemporalAdmg {
                WINDOW_DESIGN_AT_EXECUTION
            } else if is_class {
                CLASS_DESIGN_AT_EXECUTION
            } else {
                DAG_DESIGN_AT_EXECUTION
            },
        },
    )
}

/// Dose-by-horizon response on one series or a panel, under a temporal DAG or class.
fn lower_response(
    cx: &LoweringContext<'_>,
    query: &ResponseQuery,
) -> Result<LoweredProgram, CausalError> {
    let identification = retained(cx)?;
    let dag = temporal_cache(&identification);
    let class = class_cache(&identification);
    let planner_used = match (dag, class) {
        (Some(cache), _) => cache.by_horizon.iter().map(|entry| entry.indexer.history()).max(),
        (None, Some(class)) => class_history(class, true),
        (None, None) => None,
    };
    let declared = query.temporal.as_ref().and_then(|temporal| temporal.max_history_lag);
    let history = planned_history(declared, planner_used);
    let assumptions = dag
        .and_then(|cache| cache.by_horizon.first())
        .map_or_else(AssumptionSet::new, |entry| entry.identification.required_assumptions.clone());
    let target = response_target(cx, query, history, assumptions)?;
    let pair = query.functional.primary_pair();
    let mut inputs: Vec<VariableId> = pair.map(|(treatment, _)| treatment).into_iter().collect();
    if let Some(cache) = dag {
        for entry in cache.by_horizon.iter() {
            inputs.extend(adjustment_variables(entry)?);
        }
    }
    let models = pair
        .map(|(_, outcome)| slot(ModelRole::Outcome, estimator_family(cx), inputs, outcome))
        .into_iter()
        .collect();
    let is_class = is_class(cx.study);
    assemble(
        cx,
        TemporalLowering {
            target,
            envelope: is_class && !class.is_some_and(class_point_identified),
            identification,
            models,
            claim: ClaimKind::Response,
            width: if is_class { CLASS_DESIGN_AT_EXECUTION } else { RESPONSE_DESIGN_AT_EXECUTION },
        },
    )
}

/// Natural or controlled mediation on one series, under a temporal DAG or CPDAG.
fn lower_mediation(
    cx: &LoweringContext<'_>,
    query: &MediationQuery,
) -> Result<LoweredProgram, CausalError> {
    let identification = retained(cx)?;
    let dag = temporal_cache(&identification);
    let class = class_cache(&identification);
    let functional =
        super::query_functional(cx.study, FunctionalKind::PathSpecific, Some(query.outcome));
    // One theorem target per requested horizon. A class has one target per completion
    // and horizon, and lowers to the typed path-specific functional of the query.
    let target = match dag {
        Some(cache) if !cache.by_horizon.is_empty() => {
            CheckedTemporalMediationOperation::theorem_targets(
                query,
                schema_of(cx.study),
                cache
                    .by_horizon
                    .iter()
                    .map(|entry| (entry.horizon, &entry.identification, &entry.indexer)),
                selection(cx.study),
            )
            .ok()
            .and_then(horizon_targets)
            .unwrap_or(functional)
        }
        _ => functional,
    };
    let mut adjustment = Vec::new();
    if let Some(cache) = dag {
        for entry in cache.by_horizon.iter() {
            adjustment.extend(adjustment_variables(entry)?);
        }
    }
    let family = estimator_family(cx);
    let mut outcome_inputs = vec![query.treatment];
    outcome_inputs.extend(query.mediators.iter().copied());
    outcome_inputs.extend(adjustment.iter().copied());
    let mut models = vec![slot(ModelRole::Outcome, family, outcome_inputs, query.outcome)];
    for &mediator in query.mediators.iter() {
        let mut inputs = vec![query.treatment];
        inputs.extend(adjustment.iter().copied());
        models.push(slot(ModelRole::Mediator, family, inputs, mediator));
    }
    let is_class = is_class(cx.study);
    assemble(
        cx,
        TemporalLowering {
            target,
            envelope: is_class && !class.is_some_and(class_point_identified),
            identification,
            models,
            claim: ClaimKind::Point,
            width: if is_class { CLASS_DESIGN_AT_EXECUTION } else { MEDIATION_DESIGN_AT_EXECUTION },
        },
    )
}

/// Identification of the window, retained with the program when it identifies at compile.
/// A window that does not identify is left to execution, which reports the refusal.
fn window_identification(cx: &LoweringContext<'_>) -> Result<IdentificationLayer, CausalError> {
    let retained =
        cx.study.retain_window_mediation_identification(cx.plan, cx.ctx)?.and_then(|window| {
            super::RetainedIdentification::new(None, Vec::new(), super::retention_key(cx.study))
                .ok()
                .map(|retained| retained.with_execution_proof(window))
        });
    Ok(match retained {
        Some(retained) => IdentificationLayer::Retained(Arc::new(retained)),
        None => {
            IdentificationLayer::PerExecution { reason: WINDOW_MEDIATION_IDENTIFIES_PER_EXECUTION }
        }
    })
}

/// Identification of the cross-world probe, retained with the program when it identifies at
/// compile. A probe that does not identify is left to execution, which reports the refusal.
fn cross_world_identification(
    cx: &LoweringContext<'_>,
) -> Result<IdentificationLayer, CausalError> {
    let retained = cx.study.retain_cross_world_identification(cx.plan, cx.ctx)?.and_then(|proof| {
        super::RetainedIdentification::new(None, Vec::new(), super::retention_key(cx.study))
            .ok()
            .map(|retained| retained.with_execution_proof(proof))
    });
    Ok(match retained {
        Some(retained) => IdentificationLayer::Retained(Arc::new(retained)),
        None => IdentificationLayer::PerExecution { reason: CROSS_WORLD_IDENTIFIES_PER_EXECUTION },
    })
}

/// Natural mediation on one series under a temporal ADMG or PAG: identified on the unrolled
/// or unfolded window at each execution, estimated by the discrete plug-in on the lag-aligned
/// window table.
fn lower_mediation_window(
    cx: &LoweringContext<'_>,
    query: &MediationQuery,
) -> Result<LoweredProgram, CausalError> {
    let mut outcome_inputs = vec![query.treatment];
    outcome_inputs.extend(query.mediators.iter().copied());
    let family = estimator_family(cx);
    let mut models = vec![slot(ModelRole::Outcome, family, outcome_inputs, query.outcome)];
    for &mediator in query.mediators.iter() {
        models.push(slot(ModelRole::Mediator, family, vec![query.treatment], mediator));
    }
    assemble(
        cx,
        TemporalLowering {
            target: super::query_functional(
                cx.study,
                FunctionalKind::PathSpecific,
                Some(query.outcome),
            ),
            envelope: is_class(cx.study),
            identification: window_identification(cx)?,
            models,
            claim: ClaimKind::Point,
            width: WINDOW_DESIGN_AT_EXECUTION,
        },
    )
}

/// Unit counterfactual, nested counterfactual, path-specific effect or
/// interventional law on one series, under a temporal DAG or class.
fn lower_cross_world(cx: &LoweringContext<'_>) -> Result<LoweredProgram, CausalError> {
    let (kind, outcome) = match &cx.study.query {
        CausalQuery::NestedCounterfactual(query) => {
            (FunctionalKind::PathSpecific, Some(query.outcome))
        }
        CausalQuery::PathSpecific(query) => (FunctionalKind::PathSpecific, Some(query.outcome)),
        CausalQuery::Counterfactual(query) => {
            (FunctionalKind::Counterfactual, query.outcomes.first().copied())
        }
        CausalQuery::Distribution(query) => {
            (FunctionalKind::InterventionalDistribution, query.outcomes.first().copied())
        }
        _ => return Err(not_temporal(cx)),
    };
    let models = outcome
        .map(|outcome| {
            let inputs = cx
                .plan
                .logical
                .record
                .query_variables
                .iter()
                .copied()
                .filter(|variable| *variable != outcome)
                .collect();
            slot(ModelRole::Outcome, estimator_family(cx), inputs, outcome)
        })
        .into_iter()
        .collect();
    assemble(
        cx,
        TemporalLowering {
            target: super::query_functional(cx.study, kind, outcome),
            envelope: is_class(cx.study),
            identification: cross_world_identification(cx)?,
            models,
            claim: ClaimKind::Point,
            width: CROSS_WORLD_DESIGN_AT_EXECUTION,
        },
    )
}

// ---------------------------------------------------------------------------
// Shared assembly
// ---------------------------------------------------------------------------

/// What a cell lowers before the shared layers are assembled.
struct TemporalLowering {
    target: ProblemTarget,
    identification: IdentificationLayer,
    models: Vec<ModelSlot>,
    /// The answer is an identified set over the class's completions.
    envelope: bool,
    /// Claim kind of a point-identified answer.
    claim: ClaimKind,
    /// Why no design width is quoted at compile.
    width: &'static str,
}

/// The operators every temporal cell runs: the read, fit and intervention of the linear
/// route, and the shared history operators the temporal program executes over its declared
/// cadence, horizon and initial law (`antecedent_model::history_operators::HistoryProgram`):
/// one history scan, the sequential regression, the action timing and policy evaluation, the
/// transition scan, and the trajectory reduction. A cell does not carry its own scan.
fn temporal_program_operators() -> Result<Vec<antecedent_core::Operator>, CausalError> {
    use antecedent_model::history_operators::HistoryOperatorKind as Kind;
    let mut operators = linear_gaussian_program_operators()?;
    for kind in [
        Kind::HistoryConstruction,
        Kind::SequentialRegression,
        Kind::ActionTiming,
        Kind::PolicyEvaluation,
        Kind::Transition,
        Kind::TrajectoryReduction,
    ] {
        operators.push(
            kind.operator().map_err(|error| CausalError::Compile { message: error.to_string() })?,
        );
    }
    Ok(operators)
}

fn assemble(
    cx: &LoweringContext<'_>,
    parts: TemporalLowering,
) -> Result<LoweredProgram, CausalError> {
    let study = cx.study;
    let plan = cx.plan;
    let record = &plan.logical.record;
    let identifier =
        record.identifier.as_deref().and_then(|name| name.parse::<IdentifierId>().ok());
    let estimator = recorded_estimator(cx);
    let design = DesignShape::not_design_based(parts.width);
    let mut statistical = StatisticalProgram::declared(DeclaredProgram {
        identifier,
        estimator,
        inference: &study.inference,
        bootstrap_replicates: study.bootstrap_replicates,
        sampling_design: study.sampling_design,
        design,
        operators: temporal_program_operators()?,
    })?;
    statistical.models = Arc::from(parts.models);
    statistical.folds = fold_plan(study);
    statistical.resampling = resampling_plan(study);
    if let (true, InferenceMode::Bayesian(config)) = (study.retain_joint_horizons, &study.inference)
    {
        // The joint block-weight posterior over the whole surface replaces the per-horizon
        // conjugate fits: one Exp(1) weight per block of consecutive times, shared by every
        // horizon, per draw.
        let declaration = |error: antecedent_core::DeclarationError| CausalError::Compile {
            message: error.to_string(),
        };
        let prior = Prior::new(JOINT_BLOCK_WEIGHT_PRIOR).map_err(declaration)?;
        let likelihood = LikelihoodModel::new(JOINT_BLOCK_WEIGHT_BACKEND).map_err(declaration)?;
        statistical.realization =
            Realization::Bayesian(OrdinaryBayesianLikelihood::new(likelihood, prior.clone()));
        statistical.posterior = Some(PosteriorPlan {
            algorithm: PosteriorAlgorithm::Conjugate,
            draws: config.n_draws,
            prior,
        });
        statistical.resampling = None;
    }
    let bayesian = matches!(study.inference, InferenceMode::Bayesian(_));
    if parts.envelope && !bayesian {
        statistical.realization = Realization::Envelope(
            PartialIdentificationEnvelope::new(estimator_family(cx))
                .map_err(|error| CausalError::Compile { message: error.to_string() })?,
        );
    }
    let physical = PhysicalLayer {
        operators: Arc::from(linear_gaussian_physical_operators(
            plan,
            statistical.resampling.is_some(),
        )?),
        design,
    };
    let (uncertainty, kind) = match (parts.envelope, bayesian) {
        (true, true) => (UncertaintyReading::Posterior, ClaimKind::Mixture),
        (false, true) => (UncertaintyReading::Posterior, parts.claim),
        (true, false) => (UncertaintyReading::Envelope, ClaimKind::Bounds),
        // The dependence-honest temporal fits publish no analytic standard error (the iid
        // one ignores the serial dependence), so their sampling law is the block bootstrap:
        // a study that asks for no replicates declares a point claim, not a sampling one.
        (false, false) if statistical.resampling.is_none() && bootstrap_only(estimator) => {
            (UncertaintyReading::None, parts.claim)
        }
        (false, false) => (UncertaintyReading::Sampling, parts.claim),
    };
    let claim = ClaimDemand { uncertainty, kind: Some(kind) };
    let scientific = ScientificProblem {
        target: parts.target,
        plan_id: Arc::clone(&record.plan_id),
        data: record.data_classification,
        graph_class: study.graph.class().as_str(),
        query_variables: Arc::clone(&record.query_variables),
        source_digest: None,
        support_coordinate: crate::analysis::contract::matrix_coordinate(study).map(Arc::from),
        licensed: matches!(study.support_status, Some(crate::support::CellStatus::Licensed)),
        support_status: study.support_status,
    };
    let evidence = super::decision::evidence_program(cx, None, &statistical)?;
    Ok(LoweredParts {
        cell: cx.cell,
        depth: LoweringDepth::Complete,
        scientific,
        identification: parts.identification,
        statistical,
        evidence,
        physical,
        claim,
    }
    .finish())
}

/// Identification preparation would compute, retained with the program.
///
/// A refusal at compile is not raised here: the program defers, and preparation
/// identifies again and reports the same refusal it always reported.
fn retained(cx: &LoweringContext<'_>) -> Result<IdentificationLayer, CausalError> {
    match super::retain_identification(cx, None) {
        Ok(layer) => Ok(layer),
        Err(error @ (CausalError::Cancelled { .. } | CausalError::Resource { .. })) => Err(error),
        Err(_) => Ok(super::identification_refused_at_compile()),
    }
}

fn caches(layer: &IdentificationLayer) -> &[IdentificationCache] {
    match layer {
        IdentificationLayer::Retained(retained) => retained.caches(),
        IdentificationLayer::Deferred { .. }
        | IdentificationLayer::PerExecution { .. }
        | IdentificationLayer::NonGraphical(_)
        | IdentificationLayer::NotClaimed { .. } => &[],
    }
}

fn temporal_cache(layer: &IdentificationLayer) -> Option<&CachedTemporalIdentification> {
    caches(layer).iter().find_map(|cache| match cache {
        IdentificationCache::Temporal(cache) => Some(&**cache),
        _ => None,
    })
}

fn class_cache(layer: &IdentificationLayer) -> Option<&CachedTemporalClassIdentification> {
    caches(layer).iter().find_map(|cache| cache.temporal_class_cache().map(|cache| &**cache))
}

/// Deepest history any completion unfolded, over the effect envelope and, when
/// `per_horizon`, every horizon's certificate.
fn class_history(class: &CachedTemporalClassIdentification, per_horizon: bool) -> Option<u32> {
    let horizons = class.by_horizon.iter().filter(|_| per_horizon).map(|(_, envelope)| envelope);
    std::iter::once(&class.envelope)
        .chain(horizons)
        .flat_map(|envelope| {
            envelope.indexers.iter().map(antecedent_core::TemporalIndexer::history)
        })
        .max()
}

/// Every completion identifies by the same adjustment, so the class answer is a point.
fn class_point_identified(class: &CachedTemporalClassIdentification) -> bool {
    conditional_class_proof_is_complete(&class.envelope.envelope)
}

/// Variables of a horizon's lagged adjustment set.
fn adjustment_variables(
    entry: &CachedTemporalHorizonIdentification,
) -> Result<Vec<VariableId>, CausalError> {
    entry
        .estimand
        .adjustment_set
        .iter()
        .map(|dense| -> Result<VariableId, CausalError> {
            Ok(entry.indexer.key_of(dense.raw())?.variable)
        })
        .collect()
}

/// Whether a built contrast states `history` and `outcome`.
fn pins(target: &ScientificTarget, history: HistoryWindow, outcome: VariableId) -> bool {
    target_contrast(target).is_some_and(|contrast| {
        contrast_frame(contrast).and_then(TimeFrame::history) == Some(history)
            && contrast_outcome(contrast).map(OutcomeQuantity::outcome) == Some(outcome)
    })
}

/// One target per horizon, or the single target of a one-horizon request.
fn horizon_targets(targets: Vec<ScientificTarget>) -> Option<ProblemTarget> {
    match targets.len() {
        0 => None,
        1 => targets.into_iter().next().map(ProblemTarget::Target),
        _ => Some(ProblemTarget::Targets(Arc::from(targets))),
    }
}

/// The dose-by-horizon surface a response asks for; one scheduled world per horizon
/// for a single intervention that is not a hard-set grid (a shift, a soft or
/// stochastic action); otherwise its typed functional (a derivative, a joint or
/// callback action), keyed by the query.
fn response_target(
    cx: &LoweringContext<'_>,
    query: &ResponseQuery,
    history: HistoryWindow,
    assumptions: AssumptionSet,
) -> Result<ProblemTarget, CausalError> {
    let kind = match query.functional {
        ResponseFunctional::AverageDerivative { .. }
        | ResponseFunctional::PointDerivative { .. }
        | ResponseFunctional::DirectionalDerivative { .. }
        | ResponseFunctional::Jacobian { .. } => FunctionalKind::ResponseDerivative,
        ResponseFunctional::MeanCurve { .. } | ResponseFunctional::InterventionResponse { .. } => {
            FunctionalKind::InterventionalDistribution
        }
    };
    let functional = super::query_functional(
        cx.study,
        kind,
        query.functional.primary_pair().map(|(_, outcome)| outcome),
    );
    let Some(temporal) = query.temporal.as_ref() else {
        return Ok(functional);
    };
    let Some((dose, outcome, levels)) = response_grid(&query.functional) else {
        return Ok(per_horizon_response(cx, query, history, assumptions).unwrap_or(functional));
    };
    let units = outcome_units(cx.study, outcome)?;
    let grid = DoseTimeGrid {
        dose,
        levels: &levels,
        horizons: &temporal.horizons,
        policy: &temporal.policy,
        history,
    };
    let built = ObservationUnit::new(observation_unit(cx.study))
        .and_then(|unit| {
            WorldBinding::new(
                query.target_population.clone(),
                unit,
                OutcomeQuantity::new(outcome, query.outcome_functional.clone(), units)?,
                Conditioning::unconditional(),
            )
        })
        .and_then(|binding| {
            dose_time_target(
                &grid,
                &binding,
                ModelAssumptions::new(graph_family(cx.study), assumptions),
                selection(cx.study),
            )
        });
    Ok(built.map_or(functional, ProblemTarget::Target))
}

/// One scheduled world per requested horizon for an intervention response that sets one
/// non-grid action; `None` when the response is not that, or the worlds are refused.
fn per_horizon_response(
    cx: &LoweringContext<'_>,
    query: &ResponseQuery,
    history: HistoryWindow,
    assumptions: AssumptionSet,
) -> Option<ProblemTarget> {
    let temporal = query.temporal.as_ref()?;
    let ResponseFunctional::InterventionResponse { outcome, interventions } = &query.functional
    else {
        return None;
    };
    let [intervention] = &**interventions else { return None };
    let units = outcome_units(cx.study, *outcome).ok()?;
    let model = ModelAssumptions::new(graph_family(cx.study), assumptions);
    let built = ObservationUnit::new(observation_unit(cx.study)).and_then(|unit| {
        let binding = WorldBinding::new(
            query.target_population.clone(),
            unit,
            OutcomeQuantity::new(*outcome, query.outcome_functional.clone(), units)?,
            Conditioning::unconditional(),
        )?;
        temporal
            .horizons
            .iter()
            .map(|&horizon| {
                let time =
                    TimeFrame::temporal(horizon, history, InitialLaw::contemporaneous_observed())?;
                let world = InterventionWorld::new(
                    ScientificWorldId::from_raw(0),
                    WorldAction::schedule(intervention.clone(), temporal.policy.clone())?,
                    false,
                    binding.clone(),
                    time,
                )?;
                Ok(ScientificTarget::new(
                    TargetForm::World(world),
                    model.clone(),
                    selection(cx.study),
                ))
            })
            .collect::<Result<Vec<_>, ScientificTargetError>>()
    });
    built.ok().and_then(horizon_targets)
}

/// Dose, outcome and hard-set levels of a mean curve or a one-dose intervention response.
fn response_grid(functional: &ResponseFunctional) -> Option<(VariableId, VariableId, Vec<f64>)> {
    match functional {
        ResponseFunctional::MeanCurve { outcome, treatment } => {
            Some((treatment.variable, *outcome, treatment.grid.values().ok()?))
        }
        ResponseFunctional::InterventionResponse { outcome, interventions } => {
            match &**interventions {
                [Intervention::Set { variable, value: Value::Float64(level) }] => {
                    Some((*variable, *outcome, vec![*level]))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Resampling the study requests, over the unit its sampling design declares or,
/// without a declaration, the unit its modality implies: contiguous time blocks
/// of a series, whole units of a panel, whole environments of a multi-environment
/// study.
fn resampling_plan(study: &Study) -> Option<ResamplingPlan> {
    (study.bootstrap_replicates > 0).then(|| {
        let unit = resample_unit(study);
        ResamplingPlan {
            method: if unit == ResampleUnit::CalendarBlock {
                ResamplingMethod::BlockBootstrap
            } else {
                ResamplingMethod::Bootstrap
            },
            replicates: study.bootstrap_replicates,
            unit,
        }
    })
}

fn resample_unit(study: &Study) -> ResampleUnit {
    if study.sampling_design.is_some() {
        return ResampleUnit::of_design(study.sampling_design);
    }
    match &study.data {
        DataInput::Panel(_) => ResampleUnit::Subject,
        DataInput::MultiEnv(_) => ResampleUnit::Cluster,
        DataInput::Tabular(_) | DataInput::Temporal(_) | DataInput::Event(_) => {
            ResampleUnit::CalendarBlock
        }
    }
}

/// A declared discovery/estimation split: two contiguous windows, the gap between
/// them the embargo.
fn fold_plan(study: &Study) -> Option<FoldPlan> {
    study.split.as_ref().map(|split| FoldPlan {
        folds: SPLIT_WINDOWS,
        unit: ResampleUnit::CalendarBlock,
        embargo: Some(split.gap),
    })
}

fn slot(
    role: ModelRole,
    family: &str,
    mut inputs: Vec<VariableId>,
    output: VariableId,
) -> ModelSlot {
    inputs.sort_unstable_by_key(|variable| variable.raw());
    inputs.dedup();
    ModelSlot { role, family: Arc::from(family), inputs: Arc::from(inputs), output: Some(output) }
}

/// Whether the estimator's only calibrated sampling uncertainty is its block bootstrap.
const fn bootstrap_only(estimator: Option<EstimatorId>) -> bool {
    matches!(
        estimator,
        Some(EstimatorId::TemporalLinearAdjustment | EstimatorId::TemporalSequentialGcomp)
    )
}

fn recorded_estimator(cx: &LoweringContext<'_>) -> Option<EstimatorId> {
    cx.plan.logical.record.estimator.as_deref().and_then(|name| name.parse::<EstimatorId>().ok())
}

fn estimator_family(cx: &LoweringContext<'_>) -> &'static str {
    recorded_estimator(cx).map_or(UNNAMED_ESTIMATOR, |estimator| estimator.as_str())
}

fn selection(study: &Study) -> InferenceSelection {
    match study.inference {
        InferenceMode::Frequentist => InferenceSelection::Frequentist,
        InferenceMode::Bayesian(_) => InferenceSelection::Bayesian,
    }
}

fn is_class(study: &Study) -> bool {
    matches!(study.graph.class(), GraphClass::TemporalCpdag | GraphClass::TemporalPag)
}

fn graph_family(study: &Study) -> GraphFamily {
    match study.graph.class() {
        GraphClass::TemporalCpdag => GraphFamily::TemporalCpdag,
        GraphClass::TemporalPag => GraphFamily::TemporalPag,
        GraphClass::TemporalAdmg => GraphFamily::TemporalAdmg,
        _ => GraphFamily::TemporalDag,
    }
}

fn observation_unit(study: &Study) -> &'static str {
    match &study.data {
        DataInput::Panel(_) | DataInput::MultiEnv(_) => POOLED_UNIT,
        DataInput::Tabular(_) | DataInput::Temporal(_) | DataInput::Event(_) => SERIES_UNIT,
    }
}

fn schema_of(study: &Study) -> &CausalSchema {
    match &study.data {
        DataInput::Tabular(data) => data.schema(),
        DataInput::Temporal(data) | DataInput::Event(data) => data.schema(),
        DataInput::Panel(panel) => panel.schema(),
        DataInput::MultiEnv(multi) => multi.schema(),
    }
}

fn outcome_units(study: &Study, outcome: VariableId) -> Result<QuantityUnits, CausalError> {
    let units = schema_of(study).get(outcome)?.unit.as_deref().unwrap_or("unspecified");
    QuantityUnits::new(units).map_err(target_error)
}
