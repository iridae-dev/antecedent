//! Retained target and estimation procedure for a static DAG conditional effect.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalQuery, ConditionalEffectQuery, ExecutionContext, OutcomeFunctional, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::{ConditionalLinearAdjustment, EffectEstimate};
use antecedent_expr::{
    FunctionalProgram, IdentifiedEstimand, ProgramLimits, ProgramSchema, ProgramVariable,
};
use antecedent_identify::{IdentificationResult, IdentificationStatus};

use crate::planner::PhysicalExecutionPlan;
use crate::support::{CellStatus, StructureSource};
use crate::{CausalError, EstimatorId, IdentifierId, RefuteSuite};

/// The actual point and uncertainty procedure is determined by the functional and arms.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ConditionalProcedure {
    LinearInteraction,
    CrossfitAipwDistribution,
}

/// A complete selected conditional target with bound semantic design roles.
#[derive(Clone, Debug)]
pub(crate) struct CheckedConditionalOperation {
    pub(crate) query: ConditionalEffectQuery,
    pub(crate) source_query: CausalQuery,
    pub(crate) identification: IdentificationResult,
    pub(crate) estimand: IdentifiedEstimand,
    pub(crate) identifier: IdentifierId,
    pub(crate) estimator: EstimatorId,
    pub(crate) physical: PhysicalExecutionPlan,
    pub(crate) program: FunctionalProgram,
    pub(crate) design_roles: Arc<[VariableId]>,
    pub(crate) source_rows: Arc<[u32]>,
    pub(crate) procedure: ConditionalProcedure,
    pub(crate) refute: RefuteSuite,
    pub(crate) graph_version: u32,
    pub(crate) support_status: Option<CellStatus>,
    pub(crate) structure_source: StructureSource,
    pub(crate) population_registry: Option<antecedent_core::PopulationRegistry>,
    pub(crate) latency_mode: Option<super::latency::LatencyMode>,
}

impl CheckedConditionalOperation {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        data: &TabularData,
        query: ConditionalEffectQuery,
        identification: IdentificationResult,
        estimand: IdentifiedEstimand,
        physical: PhysicalExecutionPlan,
        refute: RefuteSuite,
        graph_version: u32,
        support_status: Option<CellStatus>,
        structure_source: StructureSource,
        population_registry: Option<antecedent_core::PopulationRegistry>,
        latency_mode: Option<super::latency::LatencyMode>,
    ) -> Result<Self, CausalError> {
        let source_query = CausalQuery::ConditionalEffect(query.clone());
        if identification.status != IdentificationStatus::NonparametricallyIdentified
            || identification.average_effect() != Some(&query.inner)
            || !identification.estimands.iter().any(|candidate| {
                candidate.functional == estimand.functional
                    && candidate.method == estimand.method
                    && candidate.adjustment_set == estimand.adjustment_set
            })
            || !estimand.is_adjustment_shaped()
            || query.inner.effect_modifiers.len() != 1
            || !matches!(structure_source, StructureSource::Explicit | StructureSource::Accepted)
            || !matches!(refute, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
            || physical.logical.record.estimator.as_deref()
                != Some(EstimatorId::ConditionalLinearAdjustment.as_str())
        {
            return Err(compile_error(
                "conditional target or procedure is outside the checked DAG contract",
            ));
        }
        query.validate().map_err(|e| compile_error(&e.to_string()))?;
        let identifier: IdentifierId = physical
            .logical
            .record
            .identifier
            .as_deref()
            .unwrap_or(crate::strategy_table::DEFAULT_CONDITIONAL_IDENTIFIER)
            .parse()?;
        if identifier != IdentifierId::BackdoorAdjustment {
            return Err(compile_error(
                "checked conditional effect requires backdoor identification",
            ));
        }
        let mut design_roles =
            vec![query.inner.treatment, query.inner.outcome, query.inner.effect_modifiers[0]];
        design_roles.extend(estimand.adjustment_set.iter().copied());
        design_roles.sort_unstable();
        design_roles.dedup();
        let schema = program_schema(data);
        let program = FunctionalProgram::new(
            identification.arena.clone(),
            schema,
            estimand.functional,
            estimand.functional,
            ProgramLimits::default(),
        )
        .map_err(|e| compile_error(&format!("conditional functional program: {e}")))?;
        let source_rows = rows(data, &design_roles)?;
        let procedure = if super::helpers::conditional_uses_crossfit_aipw(&query) {
            ConditionalProcedure::CrossfitAipwDistribution
        } else {
            ConditionalProcedure::LinearInteraction
        };
        Ok(Self {
            query,
            source_query,
            identification,
            estimand,
            identifier,
            estimator: EstimatorId::ConditionalLinearAdjustment,
            physical,
            program,
            design_roles: design_roles.into(),
            source_rows,
            procedure,
            refute,
            graph_version,
            support_status,
            structure_source,
            population_registry,
            latency_mode,
        })
    }

    /// Rebind columns and rows under the same variable IDs and named semantic schema.
    pub(crate) fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        if self.program.schema() != &program_schema(data)
            || self.program.mapping().source != self.estimand.functional
            || self.program.mapping().executable != self.estimand.functional
            || self.procedure
                != if super::helpers::conditional_uses_crossfit_aipw(&self.query) {
                    ConditionalProcedure::CrossfitAipwDistribution
                } else {
                    ConditionalProcedure::LinearInteraction
                }
        {
            return Err(compile_error(
                "conditional refresh changed its target, schema, or procedure",
            ));
        }
        let mut rebound = self.clone();
        rebound.source_rows = rows(data, &self.design_roles)?;
        Ok(rebound)
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<EffectEstimate, CausalError> {
        let data_est = super::helpers::apply_scalar_outcome_functional(
            data,
            self.query.inner.outcome,
            &self.query.inner.outcome_functional,
        )?;
        let fitter = ConditionalLinearAdjustment::new().with_fold_seed(ctx.rng.master_seed());
        let mut mean_query = self.query.clone();
        mean_query.inner.outcome_functional = OutcomeFunctional::Mean;
        let estimate = fitter.estimate(&data_est, &self.estimand, &mean_query, ctx)?;
        super::helpers::attach_conditional_functional_grid(
            estimate,
            data,
            &self.query,
            &self.estimand,
            ctx,
        )
    }
}

fn program_schema(data: &TabularData) -> ProgramSchema {
    ProgramSchema::new(
        data.schema()
            .variables()
            .iter()
            .map(|v| (v.id, ProgramVariable { name: Arc::clone(&v.name) })),
    )
}

fn rows(data: &TabularData, roles: &[VariableId]) -> Result<Arc<[u32]>, CausalError> {
    let mask = data.complete_case_mask(roles)?;
    let kept = mask
        .iter()
        .enumerate()
        .filter(|(_, keep)| **keep)
        .map(|(i, _)| {
            u32::try_from(i)
                .map_err(|_| compile_error("conditional row index exceeds supported size"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(kept.into())
}

fn compile_error(message: &str) -> CausalError {
    CausalError::Compile { message: message.into() }
}
