//! Lowering of the static response cells: DAG, PAG and CPDAG response curves, the
//! co-determined joint response, and the mixed-graph response.
//!
//! Layer one is the world-indexed target the query asks for: one world per grid
//! level of a mean curve, one joint world for an intervention response, and the
//! response-derivative functional for derivatives and Jacobians. An intervention the
//! world algebra cannot state (a callback action) lowers to the typed response
//! functional keyed by the query's digest. Layer two retains
//! the identification preparation would compute. Layer three reads the estimator the
//! route record resolved, its nuisance models over the identified adjustment set, the
//! cross-fitting split and the multiplier band the response options declare. Layer
//! five states that the response nuisances are not one quoted adjustment design.
//!
//! The shared assembly here also serves the structural family.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::ClaimKind;
use antecedent_core::{
    AssumptionSet, ByteEstimates, CancellationPoint, CausalSchema, Conditioning, EvidenceClaims,
    GraphFamily, InferenceSelection, Intervention, InterventionWorld, ModelAssumptions,
    ObservationUnit, Operator, OperatorDeclaration, OperatorEffect, OperatorGraph, OperatorPorts,
    OperatorRole, OutcomeQuantity, PhysicalEffect, PhysicalOperator, QuantityUnits,
    ResponseFunctional, ResponseQuery, ScientificTarget, ScientificTargetError, ScientificWorldId,
    SourceDependencyId, TargetForm, TimeFrame, Value, VariableId, WorldAction, WorldBinding,
    WorldCollection,
};
use antecedent_data::TableView;

use super::LoweringContext;
use crate::analysis::IdentificationCache;
use crate::analysis::execute::RouteInput;
use crate::analysis::prepared::CachedStaticIdentification;
use crate::error::CausalError;
use crate::planner::PhysicalExecutionPlan;
use crate::planner::lowered::{
    ClaimDemand, DesignShape, FunctionalKind, IdentificationLayer, LoweredParts, LoweredProgram,
    LoweringDepth, PhysicalLayer, ProblemTarget, ScientificProblem, UncertaintyReading,
    WidthSource,
};
use crate::planner::statistical::{
    DeclaredProgram, FoldPlan, ModelRole, ModelSlot, PosteriorPlan, Realization, ResampleUnit,
    ResamplingMethod, ResamplingPlan, StatisticalProgram,
};
use crate::{EstimatorId, GraphClass, IdentifierId, InferenceMode};

/// Why a response route quotes no adjustment-design width.
const RESPONSE_WIDTH: &str = "response nuisances are fitted on spline, cell or plug-in bases \
                              sized at execution, not on one quoted adjustment design";
/// Observation unit every static response world reads.
const ROW_UNIT: &str = "row";
/// Units of an outcome whose schema declares none.
const UNSPECIFIED_UNITS: &str = "unspecified";
/// Source dependency every read and fit of these programs names.
const SOURCE_DEPENDENCY: u64 = 1;

/// Lower one cell of this family.
pub(super) fn lower(cx: &LoweringContext<'_>) -> Result<LoweredProgram, CausalError> {
    let (data, query, family, class) = match cx.input {
        Some(RouteInput::StaticResponseDag { data, query, .. }) => {
            (data, query, GraphFamily::Dag, false)
        }
        Some(RouteInput::StaticResponsePag { data, query, .. }) => {
            (data, query, GraphFamily::Pag, true)
        }
        Some(RouteInput::StaticResponseCpdag { data, query, .. }) => {
            (data, query, GraphFamily::Cpdag, true)
        }
        Some(RouteInput::CodeterminedJoint { data, query, .. }) => {
            (data, query, GraphFamily::CoDetermined, false)
        }
        Some(RouteInput::MixedAdmgResponse { data, query }) => {
            let family = if cx.study.graph.class() == GraphClass::TemporalAdmg {
                GraphFamily::TemporalAdmg
            } else {
                GraphFamily::Admg
            };
            (data, query, family, false)
        }
        _ => return Err(super::not_lowered_by("static response", cx)),
    };
    let study = cx.study;
    let identification = retain_or_defer(cx)?;
    let cache = retained_static(&identification);
    let assumptions = cache
        .as_ref()
        .map_or_else(AssumptionSet::new, |cache| cache.identification.required_assumptions.clone());
    let key = super::query_digest(&study.query);
    // A query whose worlds are not portable (for example a callback intervention)
    // lowers to its typed functional rather than a target that misstates it.
    let target = response_target(
        data.schema(),
        query,
        family,
        assumptions,
        inference_selection(&study.inference),
        key,
    )
    .unwrap_or_else(|_| ProblemTarget::QueryFunctional {
        kind: FunctionalKind::InterventionalDistribution,
        outcome: query.functional.outcome_ids().first().copied(),
        query: key,
    });
    let estimator = record_estimator(cx);
    let adjustment: &[VariableId] = match cache.as_ref() {
        Some(cache) => &cache.estimand.adjustment_set,
        None => &[],
    };
    let models = response_models(estimator, query, adjustment);
    let (folds, resampling) = response_sampling(cx, estimator, query);
    assemble(
        cx,
        CellProgram {
            target,
            identification,
            realization: RealizationChoice::Declared,
            models,
            folds,
            resampling,
            intervenes: true,
            width: RESPONSE_WIDTH,
            claim: ClaimDemand {
                uncertainty: if class {
                    UncertaintyReading::Envelope
                } else {
                    declared_uncertainty(&study.inference)
                },
                kind: Some(ClaimKind::Response),
            },
        },
    )
}

/// The cross-fitting split and resampling the resolved response estimator runs.
///
/// The doubly robust and Riesz estimators cross-fit on the study's response
/// options, and the joint-cell estimator on its own fold count. A frequentist mean
/// curve with a simultaneous band draws row-level wild multipliers. The functional
/// plug-in resamples with the study's bootstrap request. Every other response
/// estimator reports influence-function or posterior uncertainty and draws nothing.
fn response_sampling(
    cx: &LoweringContext<'_>,
    estimator: Option<EstimatorId>,
    query: &ResponseQuery,
) -> (Option<FoldPlan>, Resampling) {
    let options = cx.study.response_options.clone().unwrap_or_default();
    let frequentist = matches!(cx.study.inference, InferenceMode::Frequentist);
    let folds = match estimator {
        Some(EstimatorId::ResponseKennedyDr | EstimatorId::ResponseRieszAde) if frequentist => {
            Some(FoldPlan { folds: options.folds, unit: ResampleUnit::Row, embargo: None })
        }
        Some(EstimatorId::CellAipw) => Some(FoldPlan {
            folds: antecedent_estimate::CellSaturatedAipw::new().folds,
            unit: ResampleUnit::Row,
            embargo: None,
        }),
        _ => None,
    };
    let resampling = match (estimator, &query.functional, options.simultaneous_replicates) {
        (Some(EstimatorId::FunctionalEffect), _, _) => Resampling::Declared,
        (
            Some(EstimatorId::ResponseKennedyDr),
            ResponseFunctional::MeanCurve { .. },
            Some(replicates),
        ) if frequentist && replicates > 0 => Resampling::Plan(ResamplingPlan {
            method: ResamplingMethod::Bootstrap,
            replicates,
            unit: ResampleUnit::Row,
        }),
        _ => Resampling::Off,
    };
    (folds, resampling)
}

/// The target a static response query asks for. `key` is the canonical digest of
/// the study's query, which keys a derivative functional.
///
/// # Errors
///
/// The outcome is not in `schema`, a grid does not resolve, or an intervention is not
/// a portable world action.
fn response_target(
    schema: &CausalSchema,
    query: &ResponseQuery,
    family: GraphFamily,
    assumptions: AssumptionSet,
    inference: InferenceSelection,
    key: [u8; 32],
) -> Result<ProblemTarget, CausalError> {
    let form = match &query.functional {
        ResponseFunctional::MeanCurve { outcome, treatment } => {
            let binding = response_world_binding(schema, query, *outcome)?;
            let worlds = treatment
                .grid
                .values()?
                .into_iter()
                .zip(0_u32..)
                .map(|(level, id)| {
                    response_world(
                        id,
                        [Intervention::set(treatment.variable, Value::f64(level))],
                        false,
                        &binding,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            TargetForm::Worlds(WorldCollection::new(worlds).map_err(target_error)?)
        }
        ResponseFunctional::InterventionResponse { outcome, interventions } => {
            let binding = response_world_binding(schema, query, *outcome)?;
            TargetForm::World(response_world(0, interventions.iter().cloned(), false, &binding)?)
        }
        ResponseFunctional::AverageDerivative { outcome, .. }
        | ResponseFunctional::PointDerivative { outcome, .. } => {
            return Ok(ProblemTarget::QueryFunctional {
                kind: FunctionalKind::ResponseDerivative,
                outcome: Some(*outcome),
                query: key,
            });
        }
        ResponseFunctional::DirectionalDerivative { outcomes, .. }
        | ResponseFunctional::Jacobian { outcomes, .. } => {
            return Ok(ProblemTarget::QueryFunctional {
                kind: FunctionalKind::ResponseDerivative,
                outcome: outcomes.first().copied(),
                query: key,
            });
        }
    };
    Ok(ProblemTarget::Target(ScientificTarget::new(
        form,
        ModelAssumptions::new(family, assumptions),
        inference,
    )))
}

/// Population, observation unit, outcome quantity and conditioning shared by every
/// world of one static response on `outcome`. A mean reads the outcome's declared
/// units; any other outcome functional is dimensionless.
///
/// # Errors
///
/// `outcome` is not in `schema`, or the population or units are not usable.
pub(crate) fn response_world_binding(
    schema: &CausalSchema,
    query: &ResponseQuery,
    outcome: VariableId,
) -> Result<WorldBinding, CausalError> {
    let units = if query.outcome_functional.is_mean() {
        QuantityUnits::new(schema.get(outcome)?.unit.as_deref().unwrap_or(UNSPECIFIED_UNITS))
            .map_err(target_error)?
    } else {
        QuantityUnits::dimensionless()
    };
    WorldBinding::new(
        query.target_population.clone(),
        ObservationUnit::new(ROW_UNIT).map_err(target_error)?,
        OutcomeQuantity::new(outcome, query.outcome_functional.clone(), units)
            .map_err(target_error)?,
        Conditioning::unconditional(),
    )
    .map_err(target_error)
}

/// One static decision-time world setting `interventions`.
///
/// # Errors
///
/// The interventions are empty, repeat a variable, or are not portable.
pub(crate) fn response_world(
    id: u32,
    interventions: impl IntoIterator<Item = Intervention>,
    reference: bool,
    binding: &WorldBinding,
) -> Result<InterventionWorld, CausalError> {
    let action = WorldAction::at_decision(interventions).map_err(target_error)?;
    InterventionWorld::new(
        ScientificWorldId::from_raw(id),
        action,
        reference,
        binding.clone(),
        TimeFrame::static_decision(),
    )
    .map_err(target_error)
}

/// Nuisance and outcome models the resolved response estimator fits.
fn response_models(
    estimator: Option<EstimatorId>,
    query: &ResponseQuery,
    adjustment: &[VariableId],
) -> Vec<ModelSlot> {
    let Some(estimator) = estimator else { return Vec::new() };
    let family = estimator.kernel_label();
    let treatments = query.functional.treatment_ids();
    let regressors = union(&treatments, adjustment);
    let mut models: Vec<_> = query
        .functional
        .outcome_ids()
        .into_iter()
        .map(|outcome| slot(ModelRole::Outcome, family, &regressors, Some(outcome)))
        .collect();
    match estimator {
        EstimatorId::ResponseKennedyDr => {
            models.push(slot(
                ModelRole::Propensity,
                family,
                adjustment,
                treatments.first().copied(),
            ));
        }
        EstimatorId::CellAipw => {
            // One propensity over the joint treatment cells, not one per treatment.
            models.push(slot(ModelRole::Propensity, family, adjustment, None));
        }
        EstimatorId::ResponseRieszAde => {
            models.push(slot(ModelRole::Nuisance, family, &regressors, None));
        }
        _ => {}
    }
    models
}

/// How a cell's resampling departs from the study's declared bootstrap request.
pub(super) enum Resampling {
    /// The route resamples as the study declares.
    Declared,
    /// The route does not resample; its uncertainty comes from elsewhere.
    Off,
    /// The route draws this plan.
    Plan(ResamplingPlan),
}

/// Whether a cell's realization is the study's declared inference.
pub(super) enum RealizationChoice {
    /// The declared inference realization and posterior plan.
    Declared,
    /// The realization the route actually runs.
    Override {
        /// Realization.
        realization: Realization,
        /// Posterior computation, when the realization is a posterior.
        posterior: Option<PosteriorPlan>,
    },
}

/// Everything one cell of the response, mediation or structural-model families states.
pub(super) struct CellProgram {
    pub(super) target: ProblemTarget,
    pub(super) identification: IdentificationLayer,
    pub(super) realization: RealizationChoice,
    pub(super) models: Vec<ModelSlot>,
    pub(super) folds: Option<FoldPlan>,
    pub(super) resampling: Resampling,
    /// Whether the program sets an intervention world.
    pub(super) intervenes: bool,
    /// Why the route quotes no adjustment-design width.
    pub(super) width: &'static str,
    pub(super) claim: ClaimDemand,
}

/// Assemble a complete lowered program for `cell`.
///
/// The identifier and estimator are the ones the route record resolved. The design
/// shape keeps the planner's sizing and marks it as not design-based, so the
/// physical record is not re-planned.
///
/// # Errors
///
/// An operator declaration is refused, a declaration name is blank, or a decision
/// goal demands a carrier the realization cannot publish.
pub(super) fn assemble(
    cx: &LoweringContext<'_>,
    cell: CellProgram,
) -> Result<LoweredProgram, CausalError> {
    let study = cx.study;
    let plan = cx.plan;
    let record = &plan.logical.record;
    let identifier =
        record.identifier.as_deref().and_then(|name| name.parse::<IdentifierId>().ok());
    let estimator = record_estimator(cx);
    let scientific = ScientificProblem {
        target: cell.target,
        plan_id: Arc::clone(&record.plan_id),
        data: record.data_classification,
        graph_class: study.graph.class().as_str(),
        query_variables: Arc::clone(&record.query_variables),
        source_digest: None,
        support_coordinate: crate::analysis::contract::matrix_coordinate(study).map(Arc::from),
        licensed: matches!(study.support_status, Some(crate::support::CellStatus::Licensed)),
    };
    let design = DesignShape {
        rows: plan.logical.row_count_hint,
        columns: plan.logical.design_columns_hint,
        source: WidthSource::NotDesignBased(cell.width),
        direct: false,
    };
    let mut statistical = StatisticalProgram::declared(DeclaredProgram {
        identifier,
        estimator,
        inference: &study.inference,
        bootstrap_replicates: study.bootstrap_replicates,
        sampling_design: study.sampling_design,
        design,
        operators: program_operators(cell.intervenes)?,
    })?;
    if let RealizationChoice::Override { realization, posterior } = cell.realization {
        statistical.realization = realization;
        statistical.posterior = posterior;
    }
    statistical.models = Arc::from(cell.models);
    statistical.folds = cell.folds;
    match cell.resampling {
        Resampling::Declared => {}
        Resampling::Off => statistical.resampling = None,
        Resampling::Plan(resampling) => statistical.resampling = Some(resampling),
    }
    let draws = statistical.resampling.is_some() || statistical.posterior.is_some();
    let physical = PhysicalLayer {
        operators: Arc::from(physical_operators(plan, cell.intervenes, draws)?),
        design,
    };
    let evidence = super::decision::evidence_program(cx, None, &statistical)?;
    Ok(LoweredParts {
        cell: cx.cell,
        depth: LoweringDepth::Complete,
        scientific,
        identification: cell.identification,
        statistical,
        evidence,
        physical,
        claim: cell.claim,
    }
    .finish())
}

/// Identify at compile what preparation would identify and retain it.
///
/// A study whose identification refuses keeps a deferred layer that records why:
/// preparation then identifies and refuses exactly as before, and a one-shot run
/// that never prepares is not refused at compile.
///
/// # Errors
///
/// Compile was cancelled.
pub(super) fn retain_or_defer(
    cx: &LoweringContext<'_>,
) -> Result<IdentificationLayer, CausalError> {
    match super::retain_identification(cx, None) {
        Ok(layer) => Ok(layer),
        Err(error @ CausalError::Cancelled { .. }) => Err(error),
        Err(_) => Ok(super::identification_refused_at_compile()),
    }
}

/// The static identification product a retained layer installs, when it has one.
pub(super) fn retained_static(
    layer: &IdentificationLayer,
) -> Option<Arc<CachedStaticIdentification>> {
    let IdentificationLayer::Retained(retained) = layer else { return None };
    retained.caches().iter().find_map(|cache| match cache {
        IdentificationCache::Static(cache) => Some(Arc::clone(cache)),
        _ => None,
    })
}

/// The estimator the route record resolved.
pub(super) fn record_estimator(cx: &LoweringContext<'_>) -> Option<EstimatorId> {
    cx.plan.logical.record.estimator.as_deref().and_then(|name| name.parse::<EstimatorId>().ok())
}

/// The inference label a scientific target carries.
pub(super) const fn inference_selection(inference: &InferenceMode) -> InferenceSelection {
    match inference {
        InferenceMode::Frequentist => InferenceSelection::Frequentist,
        InferenceMode::Bayesian(_) => InferenceSelection::Bayesian,
    }
}

/// How the uncertainty of the declared realization is read.
pub(super) const fn declared_uncertainty(inference: &InferenceMode) -> UncertaintyReading {
    match inference {
        InferenceMode::Frequentist => UncertaintyReading::Sampling,
        InferenceMode::Bayesian(_) => UncertaintyReading::Posterior,
    }
}

/// One fitted model.
pub(super) fn slot(
    role: ModelRole,
    family: &str,
    inputs: &[VariableId],
    output: Option<VariableId>,
) -> ModelSlot {
    ModelSlot { role, family: Arc::from(family), inputs: Arc::from(inputs), output }
}

/// `first` followed by the members of `second` it lacks, in order.
pub(super) fn union(first: &[VariableId], second: &[VariableId]) -> Vec<VariableId> {
    let mut joined = first.to_vec();
    for variable in second {
        if !joined.contains(variable) {
            joined.push(*variable);
        }
    }
    joined
}

/// Read and fit edges, with the intervention edge when the program sets a world.
///
/// The read is provenance, the fit is execution, and the intervention is causal.
///
/// # Errors
///
/// An operator declaration is refused. These declarations are fixed.
fn program_operators(intervenes: bool) -> Result<Vec<Operator>, CausalError> {
    let mut operators = vec![operator(
        OperatorEffect::Read,
        OperatorGraph::Provenance,
        &[OperatorRole::Population],
        &[OperatorRole::Outcome],
        EvidenceClaims::observation_read(),
    )?];
    if intervenes {
        operators.push(operator(
            OperatorEffect::Fit,
            OperatorGraph::Execution,
            &[OperatorRole::Outcome, OperatorRole::WorldOrSchedule],
            &[OperatorRole::FitHandle],
            EvidenceClaims::nuisance_fit(),
        )?);
        operators.push(operator(
            OperatorEffect::Intervene,
            OperatorGraph::Causal,
            &[OperatorRole::Population],
            &[OperatorRole::WorldOrSchedule],
            EvidenceClaims::none(),
        )?);
    } else {
        operators.push(operator(
            OperatorEffect::Fit,
            OperatorGraph::Execution,
            &[OperatorRole::Outcome],
            &[OperatorRole::FitHandle],
            EvidenceClaims::nuisance_fit(),
        )?);
    }
    Ok(operators)
}

fn operator(
    effect: OperatorEffect,
    graph: OperatorGraph,
    inputs: &[OperatorRole],
    outputs: &[OperatorRole],
    claims: EvidenceClaims,
) -> Result<Operator, CausalError> {
    let declaration = OperatorDeclaration::new(
        effect,
        graph,
        OperatorPorts::new(inputs.iter().copied(), outputs.iter().copied()),
    )
    .with_evidence_claims(claims);
    Operator::new(declaration).map_err(|error| CausalError::Compile { message: error.to_string() })
}

/// Physical read and fit, with the intervention when the program sets a world.
///
/// Byte estimates come from the plan record; a record without them leaves them
/// absent rather than declaring a size. `draws` declares a resample cancellation
/// point on the fit, for bootstrap replicates and posterior draws alike.
///
/// # Errors
///
/// An operator declaration is refused.
fn physical_operators(
    plan: &PhysicalExecutionPlan,
    intervenes: bool,
    draws: bool,
) -> Result<Vec<PhysicalOperator>, CausalError> {
    let copy = plan.record.estimated_copy_bytes;
    let workspace = plan.record.workspace_bytes;
    let read =
        copy.map_or_else(ByteEstimates::absent, |copy| ByteEstimates::absent().with_output(copy));
    let mut fit = ByteEstimates::absent();
    if let Some(workspace) = workspace {
        fit = fit.with_scratch(workspace);
    }
    if let Some(copy) = copy {
        fit = fit.with_retained(copy);
    }
    let source = SourceDependencyId::from_raw(SOURCE_DEPENDENCY);
    let mut fit_operator =
        PhysicalOperator::declare(PhysicalEffect::Fit).source(source).estimates(fit);
    if draws {
        fit_operator = fit_operator.cancellation_point(CancellationPoint::Resample);
    }
    let mut operators = vec![
        PhysicalOperator::declare(PhysicalEffect::Read).source(source).estimates(read).build(),
        fit_operator.build(),
    ];
    if intervenes {
        operators.push(PhysicalOperator::declare(PhysicalEffect::Intervene).build());
    }
    operators
        .into_iter()
        .map(|built| built.map_err(|error| CausalError::Compile { message: error.to_string() }))
        .collect()
}

pub(super) fn target_error(error: ScientificTargetError) -> CausalError {
    CausalError::Compile { message: format!("scientific target: {error}") }
}

#[cfg(test)]
#[path = "static_response_tests.rs"]
mod tests;
