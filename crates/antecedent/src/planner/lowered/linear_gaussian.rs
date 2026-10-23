//! Layer-1 and layer-5 descriptions for the static linear/Gaussian intervention route.
//!
//! The scientific target is the unconditional contrast of the query's two
//! intervention worlds. The program operators are the read, the ordinary
//! least-squares fit, and the intervention. The physical operators carry the
//! byte estimates already on the physical plan `Study::compile` returns.
//! Nothing here identifies, prepares, or compiles a second plan.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ByteEstimates, CancellationPoint, Conditioning,
    DesignDependencyId, EvidenceClaims, GraphFamily, InferenceSelection, InterventionContrast,
    InterventionWorld, ModelAssumptions, ObservationUnit, Operator, OperatorDeclaration,
    OperatorEffect, OperatorGraph, OperatorPorts, OperatorRole, OutcomeQuantity, PhysicalEffect,
    PhysicalOperator, QuantityUnits, ScientificTarget, ScientificWorldId, SourceDependencyId,
    TargetForm, TimeFrame, WorldAction, WorldCollection,
};

use crate::error::CausalError;
use crate::planner::PhysicalExecutionPlan;

const CONTROL_WORLD: u32 = 0;
const ACTIVE_WORLD: u32 = 1;
const SOURCE_DEPENDENCY: u64 = 1;
const DESIGN_DEPENDENCY: u64 = 1;
/// One `f64` contrast. A declared size, not a measured allocation.
const CONTRAST_BYTES: u64 = 8;

/// Scientific problem for one static average contrast.
///
/// # Errors
///
/// The query's interventions are not a portable static contrast.
pub(crate) fn linear_gaussian_scientific_problem(
    query: &AverageEffectQuery,
    bayesian: bool,
    schema: &antecedent_core::CausalSchema,
) -> Result<ScientificTarget, CausalError> {
    let outcome = schema.get(query.outcome)?;
    let units = QuantityUnits::new(outcome.unit.as_deref().unwrap_or("unspecified"))
        .map_err(|err| target_error(&err))?;
    let binding = antecedent_core::WorldBinding::new(
        query.target_population.clone(),
        ObservationUnit::new("row").map_err(|err| target_error(&err))?,
        OutcomeQuantity::mean(query.outcome, units).map_err(|err| target_error(&err))?,
        Conditioning::unconditional(),
    )
    .map_err(|err| target_error(&err))?;
    let time = TimeFrame::static_decision();
    let control = world(CONTROL_WORLD, &query.control, true, &binding, &time)?;
    let active = world(ACTIVE_WORLD, &query.active, false, &binding, &time)?;
    let worlds = WorldCollection::new([control, active]).map_err(|err| target_error(&err))?;
    let contrast = InterventionContrast::difference(
        worlds,
        ScientificWorldId::from_raw(ACTIVE_WORLD),
        ScientificWorldId::from_raw(CONTROL_WORLD),
        ScientificWorldId::from_raw(CONTROL_WORLD),
    )
    .map_err(|err| target_error(&err))?;
    let inference =
        if bayesian { InferenceSelection::Bayesian } else { InferenceSelection::Frequentist };
    Ok(ScientificTarget::new(
        TargetForm::Contrast(contrast),
        ModelAssumptions::new(GraphFamily::Dag, AssumptionSet::new()),
        inference,
    ))
}

fn world(
    raw: u32,
    intervention: &antecedent_core::Intervention,
    reference: bool,
    binding: &antecedent_core::WorldBinding,
    time: &TimeFrame,
) -> Result<InterventionWorld, CausalError> {
    let action =
        WorldAction::at_decision([intervention.clone()]).map_err(|err| target_error(&err))?;
    InterventionWorld::new(
        ScientificWorldId::from_raw(raw),
        action,
        reference,
        binding.clone(),
        time.clone(),
    )
    .map_err(|err| target_error(&err))
}

/// Read, fit, and intervene edges for the retained program.
///
/// The read is provenance, the fit is execution, and the intervention is causal.
/// None of them carries causal feedback.
///
/// # Errors
///
/// An operator declaration is illegal. These three declarations are fixed.
pub(crate) fn linear_gaussian_program_operators() -> Result<Vec<Operator>, CausalError> {
    Ok(vec![
        operator(
            OperatorEffect::Read,
            OperatorGraph::Provenance,
            &[OperatorRole::Population],
            &[OperatorRole::Outcome],
            EvidenceClaims::observation_read(),
        )?,
        operator(
            OperatorEffect::Fit,
            OperatorGraph::Execution,
            &[OperatorRole::Outcome, OperatorRole::WorldOrSchedule],
            &[OperatorRole::FitHandle],
            EvidenceClaims::nuisance_fit(),
        )?,
        operator(
            OperatorEffect::Intervene,
            OperatorGraph::Causal,
            &[OperatorRole::Population],
            &[OperatorRole::WorldOrSchedule],
            EvidenceClaims::none(),
        )?,
    ])
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
    Operator::new(declaration).map_err(|err| CausalError::Compile { message: err.to_string() })
}

/// Physical read, fit, and intervene metadata for `plan`.
///
/// Copy and workspace bytes come from the plan. An absent estimate is refused.
/// `resample` declares a bootstrap cancellation point on the fit.
///
/// # Errors
///
/// The plan omitted a byte estimate, or an operator declaration is illegal.
pub(crate) fn linear_gaussian_physical_operators(
    plan: &PhysicalExecutionPlan,
    resample: bool,
) -> Result<Vec<PhysicalOperator>, CausalError> {
    let copy = plan.record.estimated_copy_bytes.ok_or_else(|| CausalError::Compile {
        message: "linear/Gaussian physical plan omitted copy bytes".into(),
    })?;
    let workspace = plan.record.workspace_bytes.ok_or_else(|| CausalError::Compile {
        message: "linear/Gaussian physical plan omitted workspace bytes".into(),
    })?;
    let fit_estimates = ByteEstimates::declared(workspace, copy, CONTRAST_BYTES);
    let fit = if resample {
        physical_with_resample(fit_estimates)?
    } else {
        physical(PhysicalEffect::Fit, fit_estimates)?
    };
    Ok(vec![
        physical(PhysicalEffect::Read, ByteEstimates::declared(0, 0, copy))?,
        fit,
        physical(PhysicalEffect::Intervene, ByteEstimates::declared(0, 0, 0))?,
    ])
}

fn physical(
    effect: PhysicalEffect,
    estimates: ByteEstimates,
) -> Result<PhysicalOperator, CausalError> {
    PhysicalOperator::declare(effect)
        .source(SourceDependencyId::from_raw(SOURCE_DEPENDENCY))
        .design(DesignDependencyId::from_raw(DESIGN_DEPENDENCY))
        .estimates(estimates)
        .build()
        .map_err(|err| physical_error(&err))
}

fn physical_with_resample(estimates: ByteEstimates) -> Result<PhysicalOperator, CausalError> {
    PhysicalOperator::declare(PhysicalEffect::Fit)
        .source(SourceDependencyId::from_raw(SOURCE_DEPENDENCY))
        .design(DesignDependencyId::from_raw(DESIGN_DEPENDENCY))
        .estimates(estimates)
        .cancellation_point(CancellationPoint::Resample)
        .build()
        .map_err(|err| physical_error(&err))
}

fn target_error(err: &antecedent_core::ScientificTargetError) -> CausalError {
    CausalError::Compile { message: err.to_string() }
}

fn physical_error(err: &antecedent_core::PhysicalOperatorError) -> CausalError {
    CausalError::Compile { message: err.to_string() }
}
