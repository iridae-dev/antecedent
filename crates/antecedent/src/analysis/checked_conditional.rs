//! Retained target and estimation procedure for a static DAG conditional effect.
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::analysis::prepared::Inspect;
use crate::planner::RouteDefaults;
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

use crate::analysis::proof_support::{EstimandClaim, proof_supplies};
use crate::analysis::route_guards::{compile_error, envelope_summary};
use crate::support::StructureSource;
use crate::{CausalError, EstimatorId, IdentifierId};

/// Completion proof retained for a conditional effect identified on a static
/// equivalence class. The class graph and its complete prepare-time envelope
/// stay bound together through execution and refresh.
#[derive(Clone, Debug)]
pub(crate) enum ConditionalClassProof {
    Cpdag {
        graph: antecedent_graph::Cpdag,
        cache: Arc<crate::analysis::prepared::CachedCpdagIdentification>,
    },
    Pag {
        graph: antecedent_graph::Pag,
        cache: Arc<crate::analysis::prepared::CachedPagIdentification>,
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

    /// Whether every enumerated completion identifies the same adjustment
    /// target (see `conditional_class_proof_is_complete`).
    pub(crate) fn is_complete(&self) -> bool {
        match self {
            Self::Cpdag { cache, .. } => {
                crate::analysis::prepared::conditional_class_proof_is_complete(&cache.envelope)
            }
            Self::Pag { cache, .. } => {
                crate::analysis::prepared::conditional_class_proof_is_complete(&cache.envelope)
            }
        }
    }

    /// The estimand shared by every identified completion, if any.
    pub(crate) fn invariant(&self) -> Option<&IdentifiedEstimand> {
        match self {
            Self::Cpdag { cache, .. } => cache.envelope.invariant.as_ref(),
            Self::Pag { cache, .. } => cache.envelope.invariant.as_ref(),
        }
    }

    /// Completion count, identified mass, and unresolved mass of the envelope.
    pub(crate) fn completion_mass(&self) -> (usize, f64, f64) {
        match self {
            Self::Cpdag { cache, .. } => envelope_summary(&cache.envelope),
            Self::Pag { cache, .. } => envelope_summary(&cache.envelope),
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

/// Whether the conditional estimator fits an interaction design over `count` effect
/// modifiers. This is the single place the checked operation reads the estimator's
/// modifier-set support from; the operation itself carries every modifier as a design role.
/// A modifier set is fitted as the vector generalization of the one-modifier model and its
/// interval carries the estimator's assumption record naming it uncalibrated.
const fn estimator_fits_modifier_set(count: usize) -> bool {
    count >= 1
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
    pub(crate) cache: Arc<crate::analysis::prepared::CachedStaticIdentification>,
    pub(crate) program: FunctionalProgram,
    pub(crate) design_roles: Arc<[VariableId]>,
    pub(crate) source_rows: Arc<[u32]>,
    pub(crate) procedure: ConditionalProcedure,
    pub(crate) class_proof: Option<ConditionalClassProof>,
    pub(crate) common: Arc<crate::analysis::execute::CheckedStaticCommon>,
}

impl std::ops::Deref for CheckedConditionalOperation {
    type Target = crate::analysis::execute::CheckedStaticCommon;
    fn deref(&self) -> &Self::Target {
        &self.common
    }
}

impl CheckedConditionalOperation {
    pub(crate) fn prepare(
        data: &TabularData,
        query: ConditionalEffectQuery,
        cache: Arc<crate::analysis::prepared::CachedStaticIdentification>,
        common: Arc<crate::analysis::execute::CheckedStaticCommon>,
        class_proof: Option<ConditionalClassProof>,
    ) -> Result<Self, CausalError> {
        let (identification, estimand) = (&cache.identification, &cache.estimand);
        let source_query = CausalQuery::ConditionalEffect(query.clone());
        if identification.status != IdentificationStatus::NonparametricallyIdentified
            || (identification.query != source_query
                && identification.query != CausalQuery::AverageEffect(query.inner.clone()))
            || !proof_supplies(identification, estimand)
            || !estimand.is_adjustment_shaped()
            || !estimator_fits_modifier_set(query.inner.effect_modifiers.len())
            || !matches!(
                common.structure_source,
                StructureSource::Explicit | StructureSource::Accepted
            )
            || !common.refute.is_point()
            || !common.physical.binding().estimator.is(EstimatorId::ConditionalLinearAdjustment)
        {
            return Err(compile_error(
                "conditional target or procedure is outside the checked DAG contract",
            ));
        }
        query.validate().map_err(|e| compile_error(&e.to_string()))?;
        let identifier = common.physical.binding().identifier_or(RouteDefaults::CONDITIONAL)?;
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
                    candidate.same_claim(estimand) && candidate.is_adjustment_shaped()
                })
                || !estimand.is_adjustment_shaped()
            {
                return Err(compile_error(
                    "conditional class proof does not identify the retained adjustment target",
                ));
            }
        }
        if query.inner.effect_modifiers.is_empty() {
            return Err(compile_error("conditional effect requires an effect modifier"));
        }
        // Every modifier is a design role, so the complete-case rows of a modifier set are
        // the rows the interaction design is built on.
        let mut design_roles = vec![query.inner.treatment, query.inner.outcome];
        design_roles.extend(query.inner.effect_modifiers.iter().copied());
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
        let mut common = Arc::try_unwrap(common).unwrap_or_else(|shared| (*shared).clone());
        common.identifier = identifier;
        common.estimator = EstimatorId::ConditionalLinearAdjustment;
        Ok(Self {
            query,
            source_query,
            cache,
            program,
            design_roles: design_roles.into(),
            source_rows,
            procedure,
            class_proof,
            common: Arc::new(common),
        })
    }

    /// Rebind columns and rows under the same variable IDs and named semantic schema.
    pub(crate) fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        if self.program.schema() != &program_schema(data)
            || self.program.mapping().source != self.cache.estimand.functional
            || self.program.mapping().executable != self.cache.estimand.functional
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
        Ok(Self {
            query: self.query.clone(),
            source_query: self.source_query.clone(),
            cache: Arc::clone(&self.cache),
            program: self.program.clone(),
            design_roles: Arc::clone(&self.design_roles),
            source_rows: rows(data, &self.design_roles)?,
            procedure: self.procedure,
            class_proof: self.class_proof.clone(),
            common: Arc::clone(&self.common),
        })
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
        let mut fitter = ConditionalLinearAdjustment::new().with_fold_seed(ctx.rng.master_seed());
        if self.query.inner.effect_modifiers.len() > 1 {
            fitter = fitter.with_joint_modifiers();
        }
        let mut mean_query = self.query.clone();
        mean_query.inner.outcome_functional = OutcomeFunctional::Mean;
        let estimate = fitter.estimate(&data_est, &self.cache.estimand, &mean_query, ctx)?;
        super::helpers::attach_conditional_functional_grid(
            estimate,
            data,
            &self.query,
            &self.cache.estimand,
            ctx,
        )
    }
}

fn same_class_envelope<G: PartialEq>(
    actual: &antecedent_identify::IdentificationEnvelope<G>,
    retained: &antecedent_identify::IdentificationEnvelope<G>,
) -> bool {
    let same_invariant = match (&actual.invariant, &retained.invariant) {
        (Some(a), Some(b)) => a.same_claim(b),
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
            a.graph == b.graph
                && (a.weight.0 - b.weight.0).abs() <= f64::EPSILON
                && a.result.status == b.result.status
                && a.result.estimands.iter().any(|candidate| {
                    b.result.estimands.iter().any(|expected| candidate.same_claim(expected))
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

/// Read-only checked target and procedure for a frequentist static conditional effect.
#[derive(Clone, Debug)]
pub struct CheckedConditionalEffectInfo {
    /// Exact conditional query, including modifiers, arms, and outcome functional.
    pub query: antecedent_core::ConditionalEffectQuery,
    /// Source graph class retained with the identification proof.
    pub graph_class: crate::accepted::GraphClass,
    /// Selected identification procedure.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Selected estimator identity.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Actual scalar or distribution score procedure.
    pub procedure: std::sync::Arc<str>,
    /// Invariant adjustment variables retained by the checked proof.
    pub adjustment_set: std::sync::Arc<[antecedent_core::VariableId]>,
    /// Number of class completions checked, or `None` for a fixed DAG.
    pub completion_count: Option<usize>,
    /// Identified completion mass for a class proof.
    pub identified_mass: Option<f64>,
    /// Unresolved completion mass for a class proof.
    pub unresolved_mass: Option<f64>,
    /// Prepared refutation suite.
    pub validation: crate::analysis::builder::RefuteSuite,
    /// Variables in the physical design, retaining their semantic IDs.
    pub design_roles: std::sync::Arc<[antecedent_core::VariableId]>,
    /// Complete source row indices at preparation or latest refresh.
    pub source_rows: std::sync::Arc<[u32]>,
}

impl Inspect for CheckedConditionalOperation {
    type Info = CheckedConditionalEffectInfo;

    fn info(&self) -> CheckedConditionalEffectInfo {
        let (completion_count, identified_mass, unresolved_mass) =
            self.class_proof.as_ref().map_or((None, None, None), |proof| {
                let (count, identified, unresolved) = proof.completion_mass();
                (Some(count), Some(identified), Some(unresolved))
            });
        CheckedConditionalEffectInfo {
            query: self.query.clone(),
            graph_class: self.graph_class(),
            identifier: self.identifier,
            estimator: self.estimator,
            procedure: Arc::from(match self.procedure {
                ConditionalProcedure::LinearInteraction => "linear_interaction_plugin",
                ConditionalProcedure::CrossfitAipwDistribution => {
                    "crossfit_aipw_distribution_scores"
                }
            }),
            adjustment_set: Arc::clone(&self.cache.estimand.adjustment_set),
            completion_count,
            identified_mass,
            unresolved_mass,
            validation: self.refute,
            design_roles: Arc::clone(&self.design_roles),
            source_rows: Arc::clone(&self.source_rows),
        }
    }
}
