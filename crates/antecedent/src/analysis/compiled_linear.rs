// SPDX-License-Identifier: MIT OR Apache-2.0
//! Production scientific, identification and operator binding of checked linear execution.
use crate::{CausalError, planner::PhysicalExecutionPlan};
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionSet, AssumptionStatus, AverageEffectQuery, Operator,
    PhysicalOperator, ScientificTarget,
};
use antecedent_estimate::CheckedLinearAdjustmentAte;
use antecedent_identify::{
    AssembledFunctionalObligation, CompiledIdentificationProduct, CompiledProductParts,
    IdentificationClaimClass, IdentificationOutcome, IdentificationResult, RequiredFactor,
    UnresolvedObligation, WorldObligation,
};
use std::sync::Arc;
#[path = "linear_resources.rs"]
pub(crate) mod resources;

const CONTROL_WORLD: u32 = 0;
const ACTIVE_WORLD: u32 = 1;

/// The single checked compilation product consumed by physical inspection and execution.
#[derive(Clone, Debug)]
pub(crate) struct CompiledLinearExecution {
    pub(crate) program: Arc<CompiledLinearProgram>,
    pub(crate) identification: Arc<super::prepared::CachedStaticIdentification>,
    pub(crate) fitter: antecedent_estimate::LinearAdjustmentAte,
    pub(crate) preparation: CheckedLinearAdjustmentAte,
    pub(crate) default_id: bool,
    pub(crate) retention: Arc<antecedent_core::execution::AdmissionLease>,
    pub(crate) resource_context: antecedent_core::ExecutionContext,
}
struct LinearRecipe {
    identification: Arc<super::prepared::CachedStaticIdentification>,
    fitter: antecedent_estimate::LinearAdjustmentAte,
    default_id: bool,
}
impl CompiledLinearExecution {
    pub(crate) fn compile(
        source: &crate::Study,
        data: &antecedent_data::TabularData,
        query: &AverageEffectQuery,
        plan: &mut PhysicalExecutionPlan,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Self, CausalError> {
        let identification =
            Arc::new(source.prepare_static_identification(plan)?.ok_or_else(|| {
                CausalError::Compile { message: "linear compilation omitted identification".into() }
            })?);
        let (mut fitter, default_id) = match source.estimator_spec.as_deref() {
            Some(crate::estimator_spec::EstimatorSpec::LinearAdjustmentAte(config)) => {
                ((**config).clone(), false)
            }
            _ => (
                antecedent_estimate::LinearAdjustmentAte::new()
                    .with_bootstrap_replicates(source.bootstrap_replicates),
                true,
            ),
        };
        if fitter.population_registry.is_none() {
            fitter.population_registry.clone_from(&source.population_registry);
        }
        Self::lower(
            data,
            query,
            plan,
            ctx,
            LinearRecipe { identification, fitter, default_id },
            None,
        )
    }

    pub(crate) fn rebind(
        &self,
        data: &antecedent_data::TabularData,
        query: &AverageEffectQuery,
        plan: &mut PhysicalExecutionPlan,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Self, CausalError> {
        Self::lower(
            data,
            query,
            plan,
            ctx,
            LinearRecipe {
                identification: Arc::clone(&self.identification),
                fitter: self.fitter.clone(),
                default_id: self.default_id,
            },
            Some(&self.preparation),
        )
    }

    fn lower(
        data: &antecedent_data::TabularData,
        query: &AverageEffectQuery,
        plan: &mut PhysicalExecutionPlan,
        ctx: &antecedent_core::ExecutionContext,
        recipe: LinearRecipe,
        previous: Option<&CheckedLinearAdjustmentAte>,
    ) -> Result<Self, CausalError> {
        let LinearRecipe { identification, fitter, default_id } = recipe;
        let (_, preparation_bytes, direct) =
            resources::source_layout(data, &identification.estimand.adjustment_set, &fitter, ctx)?;
        let reservation = ctx
            .acquire_admission(0, preparation_bytes)
            .map_err(|error| CausalError::Resource { message: error.to_string() })?;
        let preparation =
            antecedent_core::numerical_admission::with_context(ctx, || match previous {
                Some(previous) => fitter.rebind_checked(previous, data),
                None => fitter.prepare_checked(data, &identification.identification, 0),
            })?;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "compile" });
        }
        let geometry = resources::LinearResources::prepared(&preparation, &fitter, ctx)?;
        let retention = Arc::new(
            reservation
                .acquire_child(0, geometry.resident)
                .map_err(|error| CausalError::Resource { message: error.to_string() })?,
        );
        plan.logical.row_count_hint = geometry.rows;
        plan.logical.design_columns_hint = geometry.columns;
        plan.record.batch_size = Some(preparation.problem().design.nrows);
        plan.record.workspace_bytes = Some(geometry.workspace);
        plan.record.estimated_peak_memory_bytes = Some(geometry.peak()?);
        plan.record.estimated_copy_bytes = Some(geometry.resident);
        plan.bind_prepared_linear_views(direct, geometry.work)?;
        let program = Arc::new(CompiledLinearProgram::compile(
            query,
            &identification.identification,
            &preparation,
            plan,
            fitter.bootstrap_replicates > 0,
            data.storage().content_digest(),
        )?);
        Ok(Self {
            program,
            identification,
            fitter,
            preparation,
            default_id,
            retention,
            resource_context: ctx.clone(),
        })
    }
}

/// The scientific and execution contracts retained with a production linear operation.
#[derive(Clone, Debug)]
pub struct CompiledLinearProgram {
    source_digest: [u8; 32],
    target: ScientificTarget,
    identification: CompiledIdentificationProduct,
    operators: Vec<Operator>,
    physical: Vec<PhysicalOperator>,
}
impl CompiledLinearProgram {
    pub(crate) fn compile(
        query: &AverageEffectQuery,
        identification: &IdentificationResult,
        checked: &CheckedLinearAdjustmentAte,
        physical: &PhysicalExecutionPlan,
        resample: bool,
        source_digest: [u8; 32],
    ) -> Result<Self, CausalError> {
        Ok(Self {
            source_digest,
            target: crate::planner::linear_gaussian_scientific_problem(query, false)?,
            identification: retain_linear_gaussian(identification, checked)?,
            operators: crate::planner::linear_gaussian_program_operators()?,
            physical: crate::planner::linear_gaussian_physical_operators(physical, resample)?,
        })
    }
    /// Identity of the immutable source rows used to compile this statistical operation.
    #[must_use]
    pub fn source_digest(&self) -> [u8; 32] {
        self.source_digest
    }
    /// The inference-independent target executed by this operation.
    #[must_use]
    pub fn target(&self) -> &ScientificTarget {
        &self.target
    }
    /// Original derivation, world obligations and required factors.
    pub fn identification(&self) -> &CompiledIdentificationProduct {
        &self.identification
    }
    /// Scientific/evidence effects retained by execution.
    #[must_use]
    pub fn operators(&self) -> &[Operator] {
        &self.operators
    }
    /// Actual physical operator contracts for this route.
    #[must_use]
    pub fn physical_operators(&self) -> &[PhysicalOperator] {
        &self.physical
    }
}
fn retain_linear_gaussian(
    identification: &IdentificationResult,
    program: &CheckedLinearAdjustmentAte,
) -> Result<CompiledIdentificationProduct, CausalError> {
    if program.required_assumptions() != &identification.required_assumptions {
        return Err(CausalError::Compile {
            message: "checked linear program dropped an identification assumption".into(),
        });
    }
    let unresolved = unresolved_obligations(&identification.required_assumptions);
    let factors = required_factors(program);
    let (outcome, claim_class) = identification_claim(identification.status);
    let worlds = world_obligations(outcome, claim_class, &unresolved);
    let assembled =
        AssembledFunctionalObligation::new(outcome, claim_class, unresolved.clone(), None);
    let mut parts = CompiledProductParts::new(outcome, claim_class);
    parts.required_factors = factors;
    parts.unresolved_obligations = unresolved;
    parts.world_obligations = worlds;
    parts.assembled = Some(assembled);
    CompiledIdentificationProduct::retain(identification, parts)
        .map_err(|refusal| CausalError::Compile { message: refusal.to_string() })
}

fn required_factors(program: &CheckedLinearAdjustmentAte) -> Vec<RequiredFactor> {
    program
        .factor_requirements()
        .iter()
        .enumerate()
        .map(|(index, factor)| {
            RequiredFactor::new(format!("factor.{index}"), Arc::clone(&factor.variables))
        })
        .collect()
}

fn world_obligations(
    outcome: IdentificationOutcome,
    claim_class: IdentificationClaimClass,
    unresolved: &[UnresolvedObligation],
) -> Vec<WorldObligation> {
    [CONTROL_WORLD, ACTIVE_WORLD]
        .into_iter()
        .map(|raw| {
            WorldObligation::new(
                antecedent_core::ScientificWorldId::from_raw(raw),
                outcome,
                claim_class,
                unresolved.to_vec(),
                None,
            )
        })
        .collect()
}

fn identification_claim(
    status: antecedent_core::IdentificationStatus,
) -> (IdentificationOutcome, IdentificationClaimClass) {
    match status {
        antecedent_core::IdentificationStatus::NonparametricallyIdentified
        | antecedent_core::IdentificationStatus::IdentifiedUnderParametricRestrictions
        | antecedent_core::IdentificationStatus::IdentifiedUnderPriorRestrictions => {
            (IdentificationOutcome::Identified, IdentificationClaimClass::IdentifiedClaim)
        }
        antecedent_core::IdentificationStatus::PartiallyIdentified
        | antecedent_core::IdentificationStatus::GraphDependent
        | antecedent_core::IdentificationStatus::NotIdentified => {
            (IdentificationOutcome::Unidentified, IdentificationClaimClass::IdentifiedClaim)
        }
    }
}

fn unresolved_obligations(assumptions: &AssumptionSet) -> Vec<UnresolvedObligation> {
    assumptions
        .entries
        .iter()
        .enumerate()
        .filter(|(_, record)| {
            matches!(record.status, AssumptionStatus::Declared | AssumptionStatus::Contradicted)
        })
        .map(|(index, record)| {
            UnresolvedObligation::new(
                obligation_code(&record.assumption, index),
                obligation_detail(record),
            )
        })
        .collect()
}

fn obligation_code(assumption: &Assumption, index: usize) -> String {
    let stem = match assumption {
        Assumption::CausalMarkov => "causal_markov",
        Assumption::Faithfulness => "faithfulness",
        Assumption::CausalSufficiency => "causal_sufficiency",
        Assumption::Consistency => "consistency",
        Assumption::Positivity => "positivity",
        Assumption::NoInterference => "no_interference",
        Assumption::Stationarity => "stationarity",
        Assumption::PiecewiseStationarity => "piecewise_stationarity",
        Assumption::NoSelectionBias => "no_selection_bias",
        Assumption::ExclusionRestriction { .. } => "exclusion_restriction",
        Assumption::Monotonicity => "monotonicity",
        Assumption::ParametricRestriction(_) => "parametric_restriction",
        Assumption::PriorRestriction(_) => "prior_restriction",
        Assumption::Custom { .. } => "custom",
    };
    format!("{stem}.{index}")
}

fn obligation_detail(record: &AssumptionRecord) -> String {
    match &record.assumption {
        Assumption::ParametricRestriction(restriction) => {
            format!("{}: {}", record.status.as_str(), restriction.id)
        }
        Assumption::PriorRestriction(restriction) => {
            format!("{}: {}", record.status.as_str(), restriction.id)
        }
        Assumption::Custom { id, .. } => format!("{}: {id}", record.status.as_str()),
        _ => record.status.as_str().to_string(),
    }
}

/// Scientific eligibility shared by public planning and checked preparation.
pub(crate) fn can_retain(source: &crate::Study, query: &AverageEffectQuery) -> bool {
    source.graph_posterior.is_none()
        && source.tiered.is_none()
        && source.inference.is_frequentist()
        && source.custom_validators.is_empty()
        && matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
        && matches!(query.target_population, antecedent_core::TargetPopulation::AllObserved)
}
