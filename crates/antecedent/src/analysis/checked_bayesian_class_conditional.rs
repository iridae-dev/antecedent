//! Sealed Bayesian conditional effect over a static CPDAG or PAG completion envelope.
//!
//! The completion proof, invariant adjustment target, prior configuration,
//! validation suite, and physical plan are retained together. Execution
//! re-verifies the envelope against the retained graph and composes over the
//! class-aware conditional executor with the click study rebuilt from these
//! fields, so no builder state can change the procedure after preparation.
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::analysis::prepared::Inspect;
use antecedent_core::{
    CausalQuery, ConditionalEffectQuery, ExecutionContext, OutcomeFunctional, TargetPopulation,
};
use antecedent_data::TabularData;
use antecedent_expr::IdentifiedEstimand;
use antecedent_identify::IdentificationStatus;
use std::sync::Arc;

use super::builder::DataInput;
use super::checked_conditional::ConditionalClassProof;
use super::execute::Study;
use super::route_guards::compile_error;
use crate::planner::PhysicalExecutionPlan;
use crate::result::StudyResult;
use crate::{CausalError, EstimatorId, IdentifierId, InferenceMode, RefuteSuite};

/// Bayesian CATE target bound to a fully identified class completion proof.
#[derive(Clone, Debug)]
pub(crate) struct CheckedBayesianClassConditional {
    proof: ConditionalClassProof,
    query: ConditionalEffectQuery,
    estimand: IdentifiedEstimand,
    identifier: IdentifierId,
    physical: std::sync::Arc<PhysicalExecutionPlan>,
    inference: InferenceMode,
    validation: RefuteSuite,
    latency_mode: Option<super::latency::LatencyMode>,
}

impl CheckedBayesianClassConditional {
    pub(crate) fn prepare(
        proof: ConditionalClassProof,
        query: ConditionalEffectQuery,
        physical: std::sync::Arc<PhysicalExecutionPlan>,
        inference: InferenceMode,
        validation: RefuteSuite,
        latency_mode: Option<super::latency::LatencyMode>,
    ) -> Result<Self, CausalError> {
        let InferenceMode::Bayesian(_) = inference else {
            return Err(CausalError::unsupported_inference(
                "checked Bayesian class conditional effect requires Bayesian inference",
            ));
        };
        query.validate().map_err(|e| compile_error(&e.to_string()))?;
        if query.inner.target_population != TargetPopulation::AllObserved
            || !matches!(query.inner.outcome_functional, OutcomeFunctional::Mean)
        {
            return Err(CausalError::Unsupported {
                message: "checked Bayesian class conditional effects require a mean outcome over AllObserved",
            });
        }
        if physical.logical.query != CausalQuery::ConditionalEffect(query.clone())
            || !physical
                .binding()
                .names(IdentifierId::GeneralizedAdjustment, EstimatorId::BayesianConditional)
            || !validation.is_point()
        {
            return Err(compile_error(
                "checked Bayesian class conditional target, identifier, or procedure changed after identification",
            ));
        }
        let identification = proof.identification();
        if !proof.is_complete()
            || identification.query != CausalQuery::ConditionalEffect(query.clone())
            || identification.status != IdentificationStatus::NonparametricallyIdentified
        {
            return Err(compile_error(
                "checked Bayesian class conditional effect requires every completion to identify the same adjustment target",
            ));
        }
        let estimand = proof
            .invariant()
            .and_then(|invariant| {
                identification
                    .estimands
                    .iter()
                    .find(|candidate| {
                        candidate.is_adjustment_shaped()
                            && candidate.method == invariant.method
                            && candidate.adjustment_set == invariant.adjustment_set
                    })
                    .cloned()
            })
            .ok_or_else(|| {
                compile_error(
                    "checked Bayesian class conditional proof does not carry its invariant adjustment estimand",
                )
            })?;
        Ok(Self {
            proof,
            query,
            estimand,
            identifier: IdentifierId::GeneralizedAdjustment,
            physical,
            inference,
            validation,
            latency_mode,
        })
    }

    /// Seal a Bayesian conditional effect on a supplied or accepted CPDAG/PAG
    /// whose every completion identifies the same adjustment target. Partial
    /// identification and non-mean targets stay outside this point plan.
    pub(crate) fn seal(
        analysis: &Study,
        plan: &std::sync::Arc<PhysicalExecutionPlan>,
    ) -> Result<Option<Self>, CausalError> {
        let CausalQuery::ConditionalEffect(query) = &analysis.query else {
            return Ok(None);
        };
        if analysis.graph_posterior.is_some()
            || analysis.declared.tiered.is_some()
            || !matches!(analysis.settings.inference, InferenceMode::Bayesian(_))
            || !matches!(
                analysis.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            || !analysis.settings.refute.is_point()
            || !analysis.declared.custom_validators.is_empty()
            || query.inner.target_population != TargetPopulation::AllObserved
            || !matches!(query.inner.outcome_functional, OutcomeFunctional::Mean)
            || !plan
                .binding()
                .names(IdentifierId::GeneralizedAdjustment, EstimatorId::BayesianConditional)
        {
            return Ok(None);
        }
        let proof = match analysis.graph.graph_ref() {
            crate::GraphRef::Cpdag(graph) => {
                let Some(cache) = analysis.cpdag_cache() else {
                    return Ok(None);
                };
                ConditionalClassProof::Cpdag { graph: graph.clone(), cache: cache.clone() }
            }
            crate::GraphRef::Pag(_) => {
                let (Some(graph), Some(cache)) = (plan.static_pag(), analysis.pag_cache()) else {
                    return Ok(None);
                };
                ConditionalClassProof::Pag { graph: graph.clone(), cache: cache.clone() }
            }
            _ => return Ok(None),
        };
        if !proof.is_complete() {
            return Ok(None);
        }
        Self::prepare(
            proof,
            query.clone(),
            plan.clone(),
            analysis.settings.inference.clone(),
            analysis.settings.refute,
            analysis.latency_mode,
        )
        .map(Some)
    }

    pub(crate) fn proof(&self) -> &ConditionalClassProof {
        &self.proof
    }
    pub(crate) fn query(&self) -> &ConditionalEffectQuery {
        &self.query
    }
    pub(crate) fn estimand(&self) -> &IdentifiedEstimand {
        &self.estimand
    }
    pub(crate) const fn identifier(&self) -> IdentifierId {
        self.identifier
    }
    pub(crate) const fn inference(&self) -> &InferenceMode {
        &self.inference
    }
    pub(crate) const fn validation(&self) -> RefuteSuite {
        self.validation
    }

    /// Execute the retained class conditional procedure on `data`.
    ///
    /// The click study is rebuilt from the retained fields only; the graph is
    /// re-identified and must reproduce the frozen envelope before any fit.
    pub(crate) fn execute(
        &self,
        analysis: &Study,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.proof.verify(self.identifier, &self.query.inner)?;
        let mut click = analysis.clone();
        click.data = DataInput::Tabular(data.clone());
        click.query = CausalQuery::ConditionalEffect(self.query.clone());
        click.settings.inference = self.inference.clone();
        click.declared.identifier = Some(self.identifier);
        click.declared.estimator = Some(EstimatorId::BayesianConditional);
        click.declared.estimator_spec =
            Some(Arc::new(crate::EstimatorSpec::Default(EstimatorId::BayesianConditional)));
        click.settings.refute = self.validation;
        click.latency_mode = self.latency_mode;
        click.install_identification(match &self.proof {
            ConditionalClassProof::Cpdag { cache, .. } => {
                crate::analysis::IdentificationCache::Cpdag(std::sync::Arc::clone(cache))
            }
            ConditionalClassProof::Pag { cache, .. } => {
                crate::analysis::IdentificationCache::Pag(std::sync::Arc::clone(cache))
            }
        });
        let mut result = click.execute_class_conditional(data, &self.query, &self.physical, ctx)?;
        super::execute::push_gaussian_likelihood_disclosure(
            &mut result,
            &self.inference,
            &click.data,
        );
        Ok(result)
    }
}

/// Read-only checked target and procedure for a Bayesian static class conditional effect.
#[derive(Clone, Debug)]
pub struct CheckedBayesianClassConditionalInfo {
    /// Exact conditional query, including modifiers and arms.
    pub query: antecedent_core::ConditionalEffectQuery,
    /// Source graph class retained with the completion proof.
    pub graph_class: crate::accepted::GraphClass,
    /// Selected identification procedure.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Selected estimator identity.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Bayesian prior and inference settings retained by preparation.
    pub inference: crate::inference::InferenceMode,
    /// Prepared refutation suite.
    pub validation: crate::analysis::builder::RefuteSuite,
    /// Invariant adjustment variables shared by every completion.
    pub adjustment_set: std::sync::Arc<[antecedent_core::VariableId]>,
    /// Number of class completions checked.
    pub completion_count: usize,
    /// Identified completion mass for the class proof.
    pub identified_mass: f64,
    /// Unresolved completion mass for the class proof.
    pub unresolved_mass: f64,
}

impl Inspect for CheckedBayesianClassConditional {
    type Info = CheckedBayesianClassConditionalInfo;

    fn info(&self) -> CheckedBayesianClassConditionalInfo {
        let (completion_count, identified_mass, unresolved_mass) = self.proof().completion_mass();
        CheckedBayesianClassConditionalInfo {
            query: self.query().clone(),
            graph_class: self.proof().graph_class(),
            identifier: self.identifier(),
            estimator: EstimatorId::BayesianConditional,
            inference: self.inference().clone(),
            validation: self.validation(),
            adjustment_set: Arc::clone(&self.estimand().adjustment_set),
            completion_count,
            identified_mass,
            unresolved_mass,
        }
    }
}
