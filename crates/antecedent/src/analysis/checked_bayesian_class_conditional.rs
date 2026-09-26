//! Sealed Bayesian conditional effect over a static CPDAG or PAG completion envelope.
//!
//! The completion proof, invariant adjustment target, prior configuration,
//! validation suite, and physical plan are retained together. Execution
//! re-verifies the envelope against the retained graph and composes over the
//! class-aware conditional executor with the click study rebuilt from these
//! fields, so no builder state can change the procedure after preparation.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    CausalQuery, ConditionalEffectQuery, ExecutionContext, OutcomeFunctional, TargetPopulation,
};
use antecedent_data::TabularData;
use antecedent_expr::IdentifiedEstimand;
use antecedent_identify::IdentificationStatus;

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
    physical: PhysicalExecutionPlan,
    inference: InferenceMode,
    validation: RefuteSuite,
    latency_mode: Option<super::latency::LatencyMode>,
}

impl CheckedBayesianClassConditional {
    pub(crate) fn prepare(
        proof: ConditionalClassProof,
        query: ConditionalEffectQuery,
        physical: PhysicalExecutionPlan,
        inference: InferenceMode,
        validation: RefuteSuite,
        latency_mode: Option<super::latency::LatencyMode>,
    ) -> Result<Self, CausalError> {
        let InferenceMode::Bayesian(_) = inference else {
            return Err(compile_error(
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
            || physical.logical.record.estimator.as_deref()
                != Some(EstimatorId::BayesianConditional.as_str())
            || physical.logical.record.identifier.as_deref()
                != Some(IdentifierId::GeneralizedAdjustment.as_str())
            || !matches!(validation, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
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
        plan: &PhysicalExecutionPlan,
    ) -> Result<Option<Self>, CausalError> {
        let CausalQuery::ConditionalEffect(query) = &analysis.query else {
            return Ok(None);
        };
        if analysis.graph_posterior.is_some()
            || analysis.tiered.is_some()
            || !matches!(analysis.inference, InferenceMode::Bayesian(_))
            || !matches!(
                analysis.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            || !matches!(
                analysis.refute,
                RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full
            )
            || !analysis.custom_validators.is_empty()
            || query.inner.target_population != TargetPopulation::AllObserved
            || !matches!(query.inner.outcome_functional, OutcomeFunctional::Mean)
            || plan.logical.record.identifier.as_deref()
                != Some(IdentifierId::GeneralizedAdjustment.as_str())
            || plan.logical.record.estimator.as_deref()
                != Some(EstimatorId::BayesianConditional.as_str())
        {
            return Ok(None);
        }
        let proof = match analysis.graph.class() {
            crate::GraphClass::Cpdag => {
                let (Some(graph), Some(cache)) =
                    (analysis.graph.as_cpdag(), analysis.cpdag_identification_cache.as_deref())
                else {
                    return Ok(None);
                };
                ConditionalClassProof::Cpdag { graph: graph.clone(), cache: cache.clone() }
            }
            crate::GraphClass::Pag => {
                let (Some(graph), Some(cache)) =
                    (plan.static_pag(), analysis.pag_identification_cache.as_deref())
                else {
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
            analysis.inference.clone(),
            analysis.refute,
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
        click.inference = self.inference.clone();
        click.identifier = Some(self.identifier);
        click.estimator = Some(EstimatorId::BayesianConditional);
        click.estimator_spec =
            Some(crate::EstimatorSpec::Default(EstimatorId::BayesianConditional));
        click.refute = self.validation;
        click.latency_mode = self.latency_mode;
        click.cpdag_identification_cache = None;
        click.pag_identification_cache = None;
        match &self.proof {
            ConditionalClassProof::Cpdag { cache, .. } => {
                click.cpdag_identification_cache = Some(std::sync::Arc::new(cache.clone()));
            }
            ConditionalClassProof::Pag { cache, .. } => {
                click.pag_identification_cache = Some(std::sync::Arc::new(cache.clone()));
            }
        }
        let mut result = click.execute_class_conditional(data, &self.query, &self.physical, ctx)?;
        super::execute::push_gaussian_likelihood_disclosure(
            &mut result,
            &self.inference,
            &click.data,
        );
        Ok(result)
    }
}
