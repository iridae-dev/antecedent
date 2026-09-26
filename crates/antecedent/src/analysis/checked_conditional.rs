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

/// Completion proof retained for a conditional effect identified on a static
/// equivalence class. The class graph and its complete prepare-time envelope
/// stay bound together through execution and refresh.
#[derive(Clone, Debug)]
pub(crate) enum ConditionalClassProof {
    Cpdag {
        graph: antecedent_graph::Cpdag,
        cache: crate::analysis::prepared::CachedCpdagIdentification,
    },
    Pag {
        graph: antecedent_graph::Pag,
        cache: crate::analysis::prepared::CachedPagIdentification,
    },
}

impl ConditionalClassProof {
    pub(crate) fn graph_class(&self) -> crate::GraphClass {
        match self {
            Self::Cpdag { .. } => crate::GraphClass::Cpdag,
            Self::Pag { .. } => crate::GraphClass::Pag,
        }
    }

    /// The identification result the retained envelope was folded into.
    pub(crate) fn identification(&self) -> &IdentificationResult {
        match self {
            Self::Cpdag { cache, .. } => &cache.identification,
            Self::Pag { cache, .. } => &cache.identification,
        }
    }

    /// Completion count, identified mass, and unresolved mass of the envelope.
    pub(crate) fn completion_mass(&self) -> (usize, f64, f64) {
        match self {
            Self::Cpdag { cache, .. } => (
                cache.envelope.cases.len(),
                cache.envelope.identified_weight.0,
                cache.envelope.unidentified_weight.0,
            ),
            Self::Pag { cache, .. } => (
                cache.envelope.cases.len(),
                cache.envelope.identified_weight.0,
                cache.envelope.unidentified_weight.0,
            ),
        }
    }

    /// Re-identify the retained graph and require the frozen envelope back.
    pub(crate) fn verify(
        &self,
        identifier: IdentifierId,
        query: &antecedent_core::AverageEffectQuery,
    ) -> Result<(), CausalError> {
        let matches = match self {
            Self::Cpdag { graph, cache } => {
                let fresh = crate::strategy_table::identify_cpdag(identifier, graph, query)?;
                same_class_envelope(&fresh, &cache.envelope)
            }
            Self::Pag { graph, cache } => {
                let fresh = crate::strategy_table::identify_pag(identifier, graph, query)?;
                same_class_envelope(&fresh, &cache.envelope)
            }
        };
        if !matches {
            return Err(compile_error(
                "retained conditional class graph no longer verifies its identification envelope",
            ));
        }
        Ok(())
    }
}

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
    pub(crate) class_proof: Option<ConditionalClassProof>,
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
        class_proof: Option<ConditionalClassProof>,
    ) -> Result<Self, CausalError> {
        let source_query = CausalQuery::ConditionalEffect(query.clone());
        if identification.status != IdentificationStatus::NonparametricallyIdentified
            || (identification.query != source_query
                && identification.query != CausalQuery::AverageEffect(query.inner.clone()))
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
        let expected_identifier = if class_proof.is_some() {
            IdentifierId::GeneralizedAdjustment
        } else {
            IdentifierId::BackdoorAdjustment
        };
        if identifier != expected_identifier {
            return Err(compile_error(
                "checked conditional effect identifier does not match its graph contract",
            ));
        }
        if let Some(proof) = &class_proof {
            let class_identification = match proof {
                ConditionalClassProof::Cpdag { cache, .. } => &cache.identification,
                ConditionalClassProof::Pag { cache, .. } => &cache.identification,
            };
            if class_identification.query != source_query
                || class_identification.status != IdentificationStatus::NonparametricallyIdentified
                || !class_identification.estimands.iter().any(|candidate| {
                    candidate.functional == estimand.functional
                        && candidate.method == estimand.method
                        && candidate.adjustment_set == estimand.adjustment_set
                        && candidate.is_adjustment_shaped()
                })
                || !estimand.is_adjustment_shaped()
            {
                return Err(compile_error(
                    "conditional class proof does not identify the retained adjustment target",
                ));
            }
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
            class_proof,
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

    pub(crate) fn graph_class(&self) -> crate::GraphClass {
        self.class_proof.as_ref().map_or(crate::GraphClass::Dag, ConditionalClassProof::graph_class)
    }

    fn verify_class_proof(&self) -> Result<(), CausalError> {
        match &self.class_proof {
            Some(proof) => proof.verify(self.identifier, &self.query.inner),
            None => Ok(()),
        }
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<EffectEstimate, CausalError> {
        self.verify_class_proof()?;
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

fn same_class_envelope<G: std::fmt::Debug>(
    actual: &antecedent_identify::IdentificationEnvelope<G>,
    retained: &antecedent_identify::IdentificationEnvelope<G>,
) -> bool {
    let same_invariant = match (&actual.invariant, &retained.invariant) {
        (Some(a), Some(b)) => {
            a.method == b.method
                && a.adjustment_set == b.adjustment_set
                && a.functional == b.functional
        }
        (None, None) => true,
        _ => false,
    };
    same_invariant
        && actual.status == retained.status
        && actual.truncated_completions == retained.truncated_completions
        && (actual.identified_weight.0 - retained.identified_weight.0).abs() <= f64::EPSILON
        && (actual.unidentified_weight.0 - retained.unidentified_weight.0).abs() <= f64::EPSILON
        && actual.cases.len() == retained.cases.len()
        && actual.cases.iter().zip(&retained.cases).all(|(a, b)| {
            format!("{:?}", a.graph) == format!("{:?}", b.graph)
                && (a.weight.0 - b.weight.0).abs() <= f64::EPSILON
                && a.result.status == b.result.status
                && a.result.estimands.iter().any(|candidate| {
                    b.result.estimands.iter().any(|expected| {
                        candidate.method == expected.method
                            && candidate.adjustment_set == expected.adjustment_set
                            && candidate.functional == expected.functional
                    })
                })
                && a.result.required_assumptions == b.result.required_assumptions
        })
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
