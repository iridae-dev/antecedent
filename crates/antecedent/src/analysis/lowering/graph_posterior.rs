//! Lowering of studies that carry a graph posterior instead of one graph.
//!
//! The program is a mixture over the posterior's graph atoms.
//!
//! * Layer one states the target every atom answers: the contrast, world or
//!   worlds of the query under the atoms' graph family, one mediation target per
//!   requested horizon, or the typed functional of the query (keyed by its
//!   canonical digest) for the cross-world queries no world-indexed target states.
//! * Layer two identifies every atom once, at compile, through the same per-atom
//!   builders preparation used, and retains the products so preparation installs
//!   them instead of identifying again. A cross-world query identifies on its probe
//!   at each execution, so it retains nothing.
//! * Layer three is the declared inference realized over the atoms: an ordinary
//!   Bayesian likelihood or a frequentist score per atom, or an identified-set
//!   envelope when the retained identification leaves posterior mass unidentified.
//!   The posterior plan mixes the atoms under the supplied structure posterior.
//! * Layer four is the atom mixture; layer five has no single design width, since
//!   every atom fits its own adjustment design.
//!
//! Identification is refused before any atom is identified when the posterior is
//! wider than the atom packing admits, or when the query is one every execution of
//! the cell refuses.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::ClaimKind;
use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CanonicalWriter, CausalQuery, CausalSchema, Conditioning,
    CrossWorldPremise, GraphFamily, HistoryWindow, InferenceSelection, InitialLaw, Intervention,
    InterventionContrast, InterventionWorld, MediationArms, MediationComponent, MediationContrast,
    MediationQuery, MediationTarget, ModelAssumptions, ObservationUnit, OutcomeFunctional,
    OutcomeQuantity, POSTERIOR_MASS_TOLERANCE, PartialIdentificationEnvelope, PhysicalOperator,
    Prior, QuantityUnits, ResponseFunctional, ResponseQuery, ScientificTarget,
    ScientificTargetError, ScientificWorldId, TargetForm, TargetPopulation, TemporalEffectQuery,
    TemporalPolicy, TimeFrame, Value, VariableId, WorldAction, WorldBinding, WorldCollection,
};
use antecedent_data::TableView;
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, MAX_POSTERIOR_NODES};

use super::{GraphPosteriorCell, LoweringContext, RouteCell};
use crate::analysis::IdentificationCache;
use crate::analysis::builder::DataInput;
use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, DesignShape, FunctionalKind, IdentificationLayer, LoweredParts, LoweredProgram,
    LoweringDepth, PhysicalLayer, ProblemTarget, ScientificProblem, UncertaintyReading,
};
use crate::planner::statistical::{
    DeclaredProgram, ModelRole, ModelSlot, PosteriorAlgorithm, PosteriorPlan, Realization,
    ResampleUnit, ResamplingMethod, StatisticalProgram,
};
use crate::{EstimatorId, IdentifierId, InferenceMode, Study};

#[cfg(test)]
#[path = "graph_posterior_tests.rs"]
mod tests;

/// Why a cross-world cell retains no identification.
const CROSS_WORLD_IDENTIFIES_PER_EXECUTION: &str =
    "a cross-world query over a temporal class posterior identifies on its probe at each execution";
/// Why a graph-posterior program has no single design width.
const ATOM_DESIGNS: &str = "every graph-posterior atom fits its own adjustment design";
/// Refusal when a posterior is wider than the atom packing.
const ATOM_CAP_REFUSAL: &str = "graph-posterior atoms are packed over at most \
                                antecedent_discovery::MAX_POSTERIOR_NODES variables; this \
                                posterior is wider, so no atom is identified";
/// Declaration name used when the route record names no estimator.
const UNNAMED_ESTIMATOR: &str = "unnamed";
/// Domain tag of the supplied structure posterior's digest.
const STRUCTURE_PRIOR_TAG: &str = "antecedent.lowered_program.graph_posterior.structure";
const CONTROL_WORLD: u32 = 0;
const ACTIVE_WORLD: u32 = 1;
/// Observation unit of a static target.
const ROW_UNIT: &str = "row";
/// Observation unit of a series target: the lag window each estimator averages over.
const LAG_WINDOW_UNIT: &str = "lag_window";

/// Lower one cell of this family.
///
/// # Errors
///
/// The study carries no graph posterior, the posterior or query is refused before
/// identification, an atom's identification refuses as preparation would have, a
/// declaration name is empty, or a decision goal demands a carrier the realization
/// cannot publish.
pub(super) fn lower(cx: &LoweringContext<'_>) -> Result<LoweredProgram, CausalError> {
    let RouteCell::GraphPosterior(cell) = cx.cell else {
        return Err(CausalError::Compile {
            message: format!("the graph-posterior lowering received {}", cx.cell.as_str()),
        });
    };
    let study = cx.study;
    let posterior = study.graph_posterior.as_ref().ok_or(CausalError::Unsupported {
        message: "a graph-posterior cell requires a study that carries a graph posterior",
    })?;
    refuse_before_identification(study, posterior)?;
    let plan = cx.plan;
    let record = &plan.logical.record;
    let identifier =
        record.identifier.as_deref().and_then(|name| name.parse::<IdentifierId>().ok());
    let estimator = record.estimator.as_deref().and_then(|name| name.parse::<EstimatorId>().ok());
    let family = atom_family(posterior);
    let target = problem_target(study, cell, family, &record.query_variables);
    let identification = if cell == GraphPosteriorCell::CrossWorld {
        IdentificationLayer::PerExecution { reason: CROSS_WORLD_IDENTIFIES_PER_EXECUTION }
    } else {
        super::retain_identification(cx, None)?
    };
    let envelope = unidentified_mass(&identification) > POSTERIOR_MASS_TOLERANCE;
    let design = DesignShape::not_design_based(ATOM_DESIGNS);
    let statistical = statistical_program(&ProgramInputs {
        study,
        cell,
        posterior,
        identifier,
        estimator,
        envelope,
        design,
        query_variables: &record.query_variables,
    })?;
    let scientific = ScientificProblem {
        target,
        plan_id: Arc::clone(&record.plan_id),
        data: record.data_classification,
        graph_class: study.graph.class().as_str(),
        query_variables: query_variables(&study.query, &record.query_variables),
        source_digest: None,
        support_coordinate: crate::analysis::contract::matrix_coordinate(study).map(Arc::from),
        licensed: matches!(study.support_status, Some(crate::support::CellStatus::Licensed)),
    };
    let physical = PhysicalLayer { operators: Arc::<[PhysicalOperator]>::from(Vec::new()), design };
    let claim = ClaimDemand {
        uncertainty: match (&statistical.realization, &study.inference) {
            (Realization::Envelope(_), _) => UncertaintyReading::Envelope,
            (_, InferenceMode::Bayesian(_)) => UncertaintyReading::Posterior,
            (_, InferenceMode::Frequentist) => UncertaintyReading::Sampling,
        },
        kind: Some(if is_response(cell) { ClaimKind::Response } else { ClaimKind::Mixture }),
    };
    let evidence = super::decision::evidence_program(cx, None, &statistical)?;
    Ok(LoweredParts {
        cell: cx.cell,
        depth: LoweringDepth::Complete,
        scientific,
        identification,
        statistical,
        evidence,
        physical,
        claim,
    }
    .finish())
}

/// Refuse, before any atom is identified, a posterior wider than the atom packing
/// and a query every execution of the cell refuses.
fn refuse_before_identification(
    study: &Study,
    posterior: &GraphPosterior,
) -> Result<(), CausalError> {
    if posterior.n_vars > MAX_POSTERIOR_NODES {
        return Err(CausalError::Unsupported { message: ATOM_CAP_REFUSAL });
    }
    match (&study.data, &study.query) {
        (DataInput::Tabular(_), CausalQuery::Response(query)) => {
            crate::analysis::execute::graph_posterior_response_supported(query)
        }
        (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Response(query)) => {
            crate::analysis::execute::dbn_posterior_response_supported(query)
        }
        _ => Ok(()),
    }
}

const fn is_response(cell: GraphPosteriorCell) -> bool {
    matches!(cell, GraphPosteriorCell::StaticResponse | GraphPosteriorCell::TemporalResponse)
}

const fn is_series(cell: GraphPosteriorCell) -> bool {
    matches!(
        cell,
        GraphPosteriorCell::TemporalEffect
            | GraphPosteriorCell::TemporalResponse
            | GraphPosteriorCell::TemporalMediation
            | GraphPosteriorCell::CrossWorld
    )
}

/// Graph family every atom of `posterior` belongs to.
fn atom_family(posterior: &GraphPosterior) -> GraphFamily {
    let temporal = posterior.lag_masks.is_some() || posterior.max_lag.is_some();
    match (posterior.atom_kind, temporal) {
        (GraphPosteriorAtomKind::Cpdag, false) => GraphFamily::Cpdag,
        (GraphPosteriorAtomKind::Cpdag, true) => GraphFamily::TemporalCpdag,
        (GraphPosteriorAtomKind::Pag, false) => GraphFamily::Pag,
        (GraphPosteriorAtomKind::Pag, true) => GraphFamily::TemporalPag,
        (GraphPosteriorAtomKind::Admg, _) => GraphFamily::Admg,
        (GraphPosteriorAtomKind::TemporalAdmg, _) => GraphFamily::TemporalAdmg,
        (_, true) => GraphFamily::TemporalDag,
        (_, false) => GraphFamily::Dag,
    }
}

/// Posterior mass the retained identification leaves unidentified; zero when
/// nothing was retained.
#[allow(
    clippy::match_same_arms,
    reason = "each posterior cache is a different type; the arms cannot share one pattern"
)]
fn unidentified_mass(identification: &IdentificationLayer) -> f64 {
    let IdentificationLayer::Retained(retained) = identification else { return 0.0 };
    retained
        .caches()
        .iter()
        .map(|cache| match cache {
            IdentificationCache::GraphPosterior(cache) => cache.graphs.unidentified_mass(),
            IdentificationCache::DbnPosterior(cache) => cache.graphs.unidentified_mass(),
            IdentificationCache::TemporalClassPosterior(cache) => cache.graphs.unidentified_mass(),
            _ => 0.0,
        })
        .fold(0.0, f64::max)
}

/// Layer-one variables: the route record's, plus the mediators a mediation query names.
fn query_variables(query: &CausalQuery, recorded: &Arc<[VariableId]>) -> Arc<[VariableId]> {
    let CausalQuery::Mediation(mediation) = query else { return Arc::clone(recorded) };
    let mut variables = recorded.to_vec();
    for mediator in mediation.mediators.iter() {
        if !variables.contains(mediator) {
            variables.push(*mediator);
        }
    }
    Arc::from(variables)
}

// ---------------------------------------------------------------------------
// Layer one: the target common to every atom.
// ---------------------------------------------------------------------------

/// The target every atom of the posterior answers.
///
/// A world-indexed target where the query states one; one target per requested
/// horizon for a multi-horizon mediation. A query the target algebra refuses (a
/// callback intervention, a zero horizon, a schedule outside its horizon) and the
/// cross-world queries take the typed functional of the query instead.
fn problem_target(
    study: &Study,
    cell: GraphPosteriorCell,
    family: GraphFamily,
    recorded: &[VariableId],
) -> ProblemTarget {
    let functional =
        |kind: FunctionalKind| super::query_functional(study, kind, recorded.get(1).copied());
    let fallback = match &study.query {
        CausalQuery::Counterfactual(_) => FunctionalKind::Counterfactual,
        CausalQuery::NestedCounterfactual(_) | CausalQuery::PathSpecific(_) => {
            FunctionalKind::PathSpecific
        }
        CausalQuery::Transport(_) => FunctionalKind::InterventionContrast,
        CausalQuery::Interference(_) => FunctionalKind::ExposureContrast,
        CausalQuery::Mediation(query) if query.contrast != MediationContrast::Total => {
            FunctionalKind::PathSpecific
        }
        CausalQuery::AverageEffect(_)
        | CausalQuery::ConditionalEffect(_)
        | CausalQuery::TemporalEffect(_)
        | CausalQuery::Mediation(_) => FunctionalKind::InterventionContrast,
        CausalQuery::Response(query)
            if !matches!(
                query.functional,
                ResponseFunctional::MeanCurve { .. }
                    | ResponseFunctional::InterventionResponse { .. }
            ) =>
        {
            FunctionalKind::ResponseDerivative
        }
        _ => FunctionalKind::InterventionalDistribution,
    };
    if cell == GraphPosteriorCell::CrossWorld {
        return functional(fallback);
    }
    let Some(schema) = schema(study) else { return functional(fallback) };
    let model = ModelAssumptions::new(family, AssumptionSet::new());
    let inference = if study.inference.is_bayesian() {
        InferenceSelection::Bayesian
    } else {
        InferenceSelection::Frequentist
    };
    let built = match (&study.query, cell) {
        (CausalQuery::AverageEffect(query), GraphPosteriorCell::AverageEffect) => {
            static_contrast(schema, query, &model, inference).map(ProblemTarget::Target)
        }
        (CausalQuery::ConditionalEffect(query), GraphPosteriorCell::ConditionalEffect) => {
            static_contrast(schema, &query.inner, &model, inference).map(ProblemTarget::Target)
        }
        (CausalQuery::Response(query), GraphPosteriorCell::StaticResponse) => {
            static_response(schema, query, &model, inference).map(ProblemTarget::Target)
        }
        (CausalQuery::Transport(query), GraphPosteriorCell::Transport) => {
            transport_contrast(schema, query, &model, inference).map(ProblemTarget::Target)
        }
        (CausalQuery::TemporalEffect(query), GraphPosteriorCell::TemporalEffect) => {
            temporal_contrast(schema, query, &model, inference).map(ProblemTarget::Target)
        }
        (CausalQuery::Response(query), GraphPosteriorCell::TemporalResponse) => {
            temporal_response(schema, query, &model, inference).map(ProblemTarget::Target)
        }
        (CausalQuery::Mediation(query), GraphPosteriorCell::TemporalMediation) => {
            temporal_mediation(schema, query, family, inference)
        }
        _ => Err(unrepresented()),
    };
    built.unwrap_or_else(|_| functional(fallback))
}

fn schema(study: &Study) -> Option<&CausalSchema> {
    match &study.data {
        DataInput::Tabular(data) => Some(data.schema()),
        DataInput::Temporal(data) | DataInput::Event(data) => Some(data.schema()),
        DataInput::MultiEnv(_) | DataInput::Panel(_) => None,
    }
}

#[allow(clippy::needless_pass_by_value, reason = "Result::map_err consumes its error")]
fn target_error(error: ScientificTargetError) -> CausalError {
    CausalError::Compile { message: format!("graph-posterior scientific target: {error}") }
}

fn unrepresented() -> CausalError {
    CausalError::Compile {
        message: "the query states no world-indexed graph-posterior target".into(),
    }
}

fn conditioning(modifiers: &[VariableId]) -> Conditioning {
    if modifiers.is_empty() {
        Conditioning::unconditional()
    } else {
        Conditioning::on(modifiers.iter().copied())
    }
}

/// Population, unit, outcome quantity and conditioning shared by the target's worlds.
fn binding(
    schema: &CausalSchema,
    population: TargetPopulation,
    unit: &str,
    outcome: VariableId,
    functional: OutcomeFunctional,
    conditioning: Conditioning,
) -> Result<WorldBinding, CausalError> {
    let units = QuantityUnits::new(schema.get(outcome)?.unit.as_deref().unwrap_or("unspecified"))
        .map_err(target_error)?;
    WorldBinding::new(
        population,
        ObservationUnit::new(unit).map_err(target_error)?,
        OutcomeQuantity::new(outcome, functional, units).map_err(target_error)?,
        conditioning,
    )
    .map_err(target_error)
}

fn world(
    raw: u32,
    action: WorldAction,
    reference: bool,
    binding: &WorldBinding,
    time: &TimeFrame,
) -> Result<InterventionWorld, CausalError> {
    InterventionWorld::new(
        ScientificWorldId::from_raw(raw),
        action,
        reference,
        binding.clone(),
        time.clone(),
    )
    .map_err(target_error)
}

fn world_id(index: usize) -> Result<u32, CausalError> {
    u32::try_from(index).map_err(|_| CausalError::Compile {
        message: "a graph-posterior target has more worlds than ids".into(),
    })
}

/// Active-minus-control contrast with the control world as the reference.
fn contrast(
    control: InterventionWorld,
    active: InterventionWorld,
    model: &ModelAssumptions,
    inference: InferenceSelection,
) -> Result<ScientificTarget, CausalError> {
    let worlds = WorldCollection::new([control, active]).map_err(target_error)?;
    let contrast = InterventionContrast::difference(
        worlds,
        ScientificWorldId::from_raw(ACTIVE_WORLD),
        ScientificWorldId::from_raw(CONTROL_WORLD),
        ScientificWorldId::from_raw(CONTROL_WORLD),
    )
    .map_err(target_error)?;
    Ok(ScientificTarget::new(TargetForm::Contrast(contrast), model.clone(), inference))
}

/// Static average or conditional effect: the contrast of the query's two worlds.
fn static_contrast(
    schema: &CausalSchema,
    query: &AverageEffectQuery,
    model: &ModelAssumptions,
    inference: InferenceSelection,
) -> Result<ScientificTarget, CausalError> {
    let binding = binding(
        schema,
        query.target_population.clone(),
        ROW_UNIT,
        query.outcome,
        query.outcome_functional.clone(),
        conditioning(&query.effect_modifiers),
    )?;
    let time = TimeFrame::static_decision();
    let at = |intervention: &Intervention| {
        WorldAction::at_decision([intervention.clone()]).map_err(target_error)
    };
    let control = world(CONTROL_WORLD, at(&query.control)?, true, &binding, &time)?;
    let active = world(ACTIVE_WORLD, at(&query.active)?, false, &binding, &time)?;
    contrast(control, active, model, inference)
}

/// Transported contrast: the inner response's treatment set to one against zero in the target
/// population, every atom answering the same target by its own selection diagram.
fn transport_contrast(
    schema: &CausalSchema,
    query: &antecedent_core::TransportQuery,
    model: &ModelAssumptions,
    inference: InferenceSelection,
) -> Result<ScientificTarget, CausalError> {
    let (treatment, outcome) =
        query.response.functional.primary_pair().ok_or_else(unrepresented)?;
    let binding = binding(
        schema,
        query.response.target_population.clone(),
        ROW_UNIT,
        outcome,
        OutcomeFunctional::Mean,
        Conditioning::unconditional(),
    )?;
    let time = TimeFrame::static_decision();
    let at = |level: f64| {
        WorldAction::at_decision([Intervention::set(treatment, Value::f64(level))])
            .map_err(target_error)
    };
    let control = world(CONTROL_WORLD, at(0.0)?, true, &binding, &time)?;
    let active = world(ACTIVE_WORLD, at(1.0)?, false, &binding, &time)?;
    contrast(control, active, model, inference)
}

/// The scalar treatment levels a response reads, one action per level.
fn response_actions(functional: &ResponseFunctional) -> Result<Vec<Intervention>, CausalError> {
    match functional {
        ResponseFunctional::InterventionResponse { interventions, .. } => {
            Ok(interventions.to_vec())
        }
        ResponseFunctional::MeanCurve { treatment, .. } => Ok(treatment
            .grid
            .values()?
            .into_iter()
            .map(|level| Intervention::set(treatment.variable, Value::f64(level)))
            .collect()),
        _ => Err(unrepresented()),
    }
}

fn response_outcome(query: &ResponseQuery) -> Result<VariableId, CausalError> {
    query.functional.outcome_ids().into_iter().next().ok_or_else(unrepresented)
}

/// Static response: the world of an intervention response, or one world per
/// level of a mean curve.
fn static_response(
    schema: &CausalSchema,
    query: &ResponseQuery,
    model: &ModelAssumptions,
    inference: InferenceSelection,
) -> Result<ScientificTarget, CausalError> {
    let binding = binding(
        schema,
        query.target_population.clone(),
        ROW_UNIT,
        response_outcome(query)?,
        query.outcome_functional.clone(),
        Conditioning::unconditional(),
    )?;
    let time = TimeFrame::static_decision();
    let form = match &query.functional {
        ResponseFunctional::InterventionResponse { interventions, .. } => {
            let action =
                WorldAction::at_decision(interventions.iter().cloned()).map_err(target_error)?;
            TargetForm::World(world(CONTROL_WORLD, action, false, &binding, &time)?)
        }
        functional => {
            let mut worlds = Vec::new();
            for (index, intervention) in response_actions(functional)?.into_iter().enumerate() {
                let action = WorldAction::at_decision([intervention]).map_err(target_error)?;
                worlds.push(world(world_id(index)?, action, false, &binding, &time)?);
            }
            TargetForm::Worlds(WorldCollection::new(worlds).map_err(target_error)?)
        }
    };
    Ok(ScientificTarget::new(form, model.clone(), inference))
}

fn history(max_lag: Option<u32>) -> HistoryWindow {
    max_lag.map_or_else(HistoryWindow::planner_default, HistoryWindow::max_lag)
}

/// Temporal effect: the contrast of the query's scheduled worlds at its horizon.
fn temporal_contrast(
    schema: &CausalSchema,
    query: &TemporalEffectQuery,
    model: &ModelAssumptions,
    inference: InferenceSelection,
) -> Result<ScientificTarget, CausalError> {
    let binding = binding(
        schema,
        query.target_population.clone(),
        LAG_WINDOW_UNIT,
        query.outcome,
        OutcomeFunctional::Mean,
        conditioning(&query.effect_modifiers),
    )?;
    let time = TimeFrame::temporal(
        query.horizon_steps,
        history(query.max_history_lag),
        InitialLaw::contemporaneous_observed(),
    )
    .map_err(target_error)?;
    let scheduled = |intervention: &Intervention| {
        WorldAction::schedule(intervention.clone(), query.policy.clone()).map_err(target_error)
    };
    let control = world(CONTROL_WORLD, scheduled(&query.control)?, true, &binding, &time)?;
    let active = world(ACTIVE_WORLD, scheduled(&query.active)?, false, &binding, &time)?;
    contrast(control, active, model, inference)
}

/// Temporal response: one scheduled world per (horizon, level).
fn temporal_response(
    schema: &CausalSchema,
    query: &ResponseQuery,
    model: &ModelAssumptions,
    inference: InferenceSelection,
) -> Result<ScientificTarget, CausalError> {
    let spec = query.temporal.as_ref().ok_or_else(unrepresented)?;
    let binding = binding(
        schema,
        query.target_population.clone(),
        LAG_WINDOW_UNIT,
        response_outcome(query)?,
        query.outcome_functional.clone(),
        Conditioning::unconditional(),
    )?;
    let actions = response_actions(&query.functional)?;
    let window = history(spec.max_history_lag);
    let mut worlds = Vec::new();
    for &horizon in spec.horizons.iter() {
        let time = TimeFrame::temporal(horizon, window, InitialLaw::contemporaneous_observed())
            .map_err(target_error)?;
        for intervention in &actions {
            let action = WorldAction::schedule(intervention.clone(), spec.policy.clone())
                .map_err(target_error)?;
            worlds.push(world(world_id(worlds.len())?, action, false, &binding, &time)?);
        }
    }
    Ok(ScientificTarget::new(
        TargetForm::Worlds(WorldCollection::new(worlds).map_err(target_error)?),
        model.clone(),
        inference,
    ))
}

/// Temporal mediation: one target per requested horizon, in request order, with the
/// cross-world premises a natural component carries. A one-horizon request is one
/// target. Every requested horizon is also a mediator and outcome model of layer three.
fn temporal_mediation(
    schema: &CausalSchema,
    query: &MediationQuery,
    family: GraphFamily,
    inference: InferenceSelection,
) -> Result<ProblemTarget, CausalError> {
    query.validate()?;
    let mut targets = query
        .horizons
        .iter()
        .map(|&horizon| temporal_mediation_at(schema, query, horizon, family, inference))
        .collect::<Result<Vec<_>, _>>()?;
    match targets.len() {
        0 => Err(unrepresented()),
        1 => targets.pop().map(ProblemTarget::Target).ok_or_else(unrepresented),
        _ => Ok(ProblemTarget::Targets(Arc::from(targets))),
    }
}

/// The mediation target of `query` at one `horizon`.
fn temporal_mediation_at(
    schema: &CausalSchema,
    query: &MediationQuery,
    horizon: u32,
    family: GraphFamily,
    inference: InferenceSelection,
) -> Result<ScientificTarget, CausalError> {
    let component = match query.contrast {
        MediationContrast::Total => MediationComponent::Total,
        MediationContrast::Direct => MediationComponent::Direct,
        MediationContrast::NaturalDirect => MediationComponent::NaturalDirect,
        MediationContrast::Mediated => MediationComponent::Mediated,
        MediationContrast::NaturalIndirect => MediationComponent::NaturalIndirect,
    };
    let binding = binding(
        schema,
        query.target_population.clone(),
        LAG_WINDOW_UNIT,
        query.outcome,
        OutcomeFunctional::Mean,
        Conditioning::unconditional(),
    )?;
    let time = TimeFrame::temporal(
        horizon,
        HistoryWindow::planner_default(),
        InitialLaw::contemporaneous_observed(),
    )
    .map_err(target_error)?;
    let arms = MediationArms::new(
        query.treatment,
        query.mediators.iter().copied(),
        query.control.clone(),
        query.active.clone(),
        component,
    )
    .map_err(target_error)?;
    let premises = if component == MediationComponent::Total {
        Vec::new()
    } else {
        vec![
            CrossWorldPremise::Composition,
            CrossWorldPremise::CrossWorldIndependence,
            CrossWorldPremise::PathSpecificNested,
        ]
    };
    // The temporal mediation identifier sets the treatment one step before the origin.
    let mediation =
        MediationTarget::new(binding, time, Some(TemporalPolicy::pulse(-1)), arms, premises)
            .map_err(target_error)?;
    Ok(ScientificTarget::new(
        TargetForm::Mediation(mediation),
        ModelAssumptions::new(family, AssumptionSet::new()),
        inference,
    ))
}

// ---------------------------------------------------------------------------
// Layer three: the statistical program over the atoms.
// ---------------------------------------------------------------------------

struct ProgramInputs<'a> {
    study: &'a Study,
    cell: GraphPosteriorCell,
    posterior: &'a GraphPosterior,
    identifier: Option<IdentifierId>,
    estimator: Option<EstimatorId>,
    envelope: bool,
    design: DesignShape,
    query_variables: &'a [VariableId],
}

fn statistical_program(inputs: &ProgramInputs<'_>) -> Result<StatisticalProgram, CausalError> {
    let ProgramInputs {
        study,
        cell,
        posterior,
        identifier,
        estimator,
        envelope,
        design,
        query_variables,
    } = *inputs;
    let mut program = StatisticalProgram::declared(DeclaredProgram {
        identifier,
        estimator,
        inference: &study.inference,
        bootstrap_replicates: study.bootstrap_replicates,
        sampling_design: study.sampling_design,
        design,
        operators: Vec::new(),
    })?;
    let name = estimator.map_or(UNNAMED_ESTIMATOR, |estimator| estimator.as_str());
    if envelope {
        program.realization = Realization::Envelope(
            PartialIdentificationEnvelope::new(format!("graph_posterior_identified_set({name})"))
                .map_err(declaration_error)?,
        );
    }
    // A series with no declared sampling design is resampled in contiguous blocks
    // of calendar time, shared across every atom.
    if is_series(cell) && study.sampling_design.is_none() {
        if let Some(plan) = program.resampling.as_mut() {
            plan.method = ResamplingMethod::BlockBootstrap;
            plan.unit = ResampleUnit::CalendarBlock;
        }
    }
    let structure = structure_prior_name(posterior);
    let declared = program.posterior.take();
    let per_atom = declared.as_ref().map(|plan| plan.algorithm.as_str());
    program.posterior = Some(match &study.inference {
        InferenceMode::Bayesian(config) => PosteriorPlan {
            algorithm: PosteriorAlgorithm::AtomMixture,
            draws: config.n_draws,
            prior: Prior::new(match &declared {
                Some(plan) => format!("{} | {structure}", plan.prior.name()),
                None => structure,
            })
            .map_err(declaration_error)?,
        },
        InferenceMode::Frequentist => PosteriorPlan {
            algorithm: PosteriorAlgorithm::AtomMixture,
            draws: posterior.n_graphs,
            prior: Prior::new(structure).map_err(declaration_error)?,
        },
    });
    let family: Arc<str> = match per_atom {
        Some(algorithm) => Arc::from(format!("{name}.{algorithm}")),
        None => Arc::from(name),
    };
    program.models = Arc::from(model_slots(&study.query, &family, query_variables));
    Ok(program)
}

/// The supplied structure posterior, named by its producer, atom kind, and a digest
/// of its weights, graph keys, lag and mark structure. A different posterior is a
/// different prior over graphs.
fn structure_prior_name(posterior: &GraphPosterior) -> String {
    let mut writer = CanonicalWriter::new(STRUCTURE_PRIOR_TAG);
    writer
        .text(posterior.atom_kind.as_str())
        .word(posterior.n_vars as u64)
        .word(posterior.n_graphs as u64)
        .opt_word(posterior.max_lag.map(u64::from))
        .reals(&posterior.weights);
    for key in posterior.graph_keys.iter() {
        writer.word(*key);
    }
    match &posterior.lag_masks {
        Some(masks) => {
            writer.word(1).word(masks.len() as u64);
            for mask in masks.iter() {
                writer.word(*mask);
            }
        }
        None => {
            writer.word(0);
        }
    }
    // Lagged latent-confounder edges distinguish two templates with the same directed edges;
    // a posterior without them writes nothing here, so its digest is the one it always had.
    if let Some(masks) = &posterior.lag_bidirected_masks {
        writer.word(2).word(masks.len() as u64);
        for mask in masks.iter() {
            writer.word(*mask);
        }
    }
    match &posterior.mark_atoms {
        Some(marks) => {
            writer.word(1).word(marks.len() as u64);
            for mark in marks.iter() {
                writer.word(mark.key());
            }
        }
        None => {
            writer.word(0);
        }
    }
    format!(
        "graph_posterior(algorithm={}, atom_kind={}, atoms={}, digest={})",
        posterior.algorithm.as_deref().unwrap_or("unnamed"),
        posterior.atom_kind.as_str(),
        posterior.n_graphs,
        writer.finish_hex()
    )
}

/// The models every atom fits: the outcome model of an effect or response, the
/// mediator and outcome models of each mediation horizon, or the mechanism a
/// cross-world probe evaluates.
fn model_slots(query: &CausalQuery, family: &Arc<str>, recorded: &[VariableId]) -> Vec<ModelSlot> {
    fn with_modifiers(treatment: VariableId, modifiers: &[VariableId]) -> Vec<VariableId> {
        let mut inputs = vec![treatment];
        inputs.extend(modifiers.iter().copied().filter(|modifier| *modifier != treatment));
        inputs
    }
    let slot = |role: ModelRole, family: Arc<str>, inputs: Vec<VariableId>, output| ModelSlot {
        role,
        family,
        inputs: Arc::from(inputs),
        output,
    };
    match query {
        CausalQuery::AverageEffect(query) => vec![slot(
            ModelRole::Outcome,
            Arc::clone(family),
            with_modifiers(query.treatment, &query.effect_modifiers),
            Some(query.outcome),
        )],
        CausalQuery::ConditionalEffect(query) => vec![slot(
            ModelRole::Outcome,
            Arc::clone(family),
            with_modifiers(query.inner.treatment, &query.inner.effect_modifiers),
            Some(query.inner.outcome),
        )],
        CausalQuery::TemporalEffect(query) => vec![slot(
            ModelRole::Outcome,
            Arc::clone(family),
            with_modifiers(query.treatment, &query.effect_modifiers),
            Some(query.outcome),
        )],
        CausalQuery::Response(query) => vec![slot(
            ModelRole::Outcome,
            Arc::clone(family),
            query.functional.treatment_ids(),
            query.functional.outcome_ids().into_iter().next(),
        )],
        CausalQuery::Mediation(query) => {
            let mut slots = Vec::new();
            for horizon in query.horizons.iter() {
                let at_horizon: Arc<str> = Arc::from(format!("{family}.horizon_{horizon}"));
                for mediator in query.mediators.iter() {
                    slots.push(slot(
                        ModelRole::Mediator,
                        Arc::clone(&at_horizon),
                        vec![query.treatment],
                        Some(*mediator),
                    ));
                }
                let mut inputs = vec![query.treatment];
                inputs.extend(query.mediators.iter().copied());
                slots.push(slot(ModelRole::Outcome, at_horizon, inputs, Some(query.outcome)));
            }
            slots
        }
        CausalQuery::Transport(query) => query.response.functional.primary_pair().map_or_else(
            Vec::new,
            |(treatment, outcome)| {
                vec![slot(ModelRole::Outcome, Arc::clone(family), vec![treatment], Some(outcome))]
            },
        ),
        CausalQuery::Interference(query) => {
            let antecedent_core::InterferenceFunctional::ExposureContrast { outcome, .. } =
                query.functional;
            vec![slot(ModelRole::Outcome, Arc::clone(family), Vec::new(), Some(outcome))]
        }
        _ => vec![slot(
            ModelRole::Mechanism,
            Arc::clone(family),
            recorded.first().copied().into_iter().collect(),
            recorded.get(1).copied(),
        )],
    }
}

fn declaration_error(error: antecedent_core::DeclarationError) -> CausalError {
    CausalError::Compile { message: error.to_string() }
}
