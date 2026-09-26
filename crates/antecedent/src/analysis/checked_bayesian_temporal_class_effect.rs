//! Checked Bayesian operation for TemporalCpdag/TemporalPag pulse and sustained effects.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CausalQuery;

use super::builder::{DataInput, RefuteSuite};
use super::checked_temporal_class_effect::{
    CheckedTemporalClassEffectOperation, same_temporal_class_proof,
};
use super::execute::Study;
use super::prepared::CachedTemporalClassIdentification;
use crate::GraphClass;
use crate::error::CausalError;
use crate::planner::PhysicalExecutionPlan;
use crate::strategy_table::EstimatorId;
use crate::{BayesianConfig, ClassPrior, InferenceMode};

/// A complete Bayesian temporal effect operation over an incomplete temporal
/// class. The retained class target owns the source graph, the query, and the
/// completion envelope with its aligned unfolding indexers; the Bayesian model
/// settings, the optional caller-supplied class mass, and the physical plan
/// are sealed alongside it. This is a separate route from the fixed-DAG
/// Bayesian operation and is never implied by it.
#[derive(Clone)]
pub(crate) struct CheckedBayesianTemporalClassEffectOperation {
    target: CheckedTemporalClassEffectOperation,
    config: BayesianConfig,
    validation: RefuteSuite,
    class_prior: Option<ClassPrior>,
    physical: PhysicalExecutionPlan,
}

impl std::fmt::Debug for CheckedBayesianTemporalClassEffectOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CheckedBayesianTemporalClassEffectOperation")
            .field("target", &self.target)
            .field("posterior_draws", &self.config.n_draws)
            .field("validation", &self.validation)
            .field("class_prior", &self.class_prior.is_some())
            .finish_non_exhaustive()
    }
}

impl CheckedBayesianTemporalClassEffectOperation {
    /// Whether `study` is a coordinate this operation seals: a series pulse or
    /// sustained effect on a fixed TemporalCpdag or TemporalPag under Bayesian
    /// inference with the default point-estimate settings. The one-shot facade
    /// and the prepared sealer share this predicate.
    #[must_use]
    pub(crate) fn admits(study: &Study) -> bool {
        let CausalQuery::TemporalEffect(query) = &study.query else {
            return false;
        };
        matches!(study.data, DataInput::Temporal(_) | DataInput::Event(_))
            && study.graph_posterior.is_none()
            && study.tiered.is_none()
            && matches!(study.graph.class(), GraphClass::TemporalCpdag | GraphClass::TemporalPag)
            && study.fixed_structure()
            && matches!(study.inference, InferenceMode::Bayesian(_))
            && study.custom_validators.is_empty()
            && study.point_validation()
            && query.policy.is_pulse_or_sustained()
            && study
                .estimator
                .is_none_or(|id| id == EstimatorId::temporal_effect_procedure(query, true))
    }

    pub(crate) fn checked(
        target: CheckedTemporalClassEffectOperation,
        inference: &InferenceMode,
        validation: RefuteSuite,
        class_prior: Option<ClassPrior>,
        physical: PhysicalExecutionPlan,
    ) -> Result<Self, CausalError> {
        let config = checked_settings(inference, validation)?;
        let (identifier, _, _, _, target_validation, custom_validators) = target.procedure();
        if target_validation != validation || !custom_validators.is_empty() {
            return Err(CausalError::Compile {
                message:
                    "checked Bayesian temporal class target disagrees with its validation suite"
                        .into(),
            });
        }
        let query = CausalQuery::TemporalEffect(target.query().clone());
        if physical.logical.query != query
            || physical.logical.record.identifier.as_deref() != Some(identifier.as_str())
            || physical.logical.record.estimator.as_deref()
                != Some(Self::estimator_for(&target).as_str())
        {
            return Err(CausalError::Compile {
                message: "Bayesian temporal class physical plan disagrees with its retained proof or procedure"
                    .into(),
            });
        }
        Ok(Self { target, config, validation, class_prior, physical })
    }

    fn estimator_for(target: &CheckedTemporalClassEffectOperation) -> EstimatorId {
        EstimatorId::temporal_effect_procedure(target.query(), true)
    }

    #[must_use]
    pub(crate) fn target(&self) -> &CheckedTemporalClassEffectOperation {
        &self.target
    }

    #[must_use]
    pub(crate) fn config(&self) -> &BayesianConfig {
        &self.config
    }

    #[must_use]
    pub(crate) const fn validation(&self) -> RefuteSuite {
        self.validation
    }

    #[must_use]
    pub(crate) fn class_prior(&self) -> Option<&ClassPrior> {
        self.class_prior.as_ref()
    }

    #[must_use]
    pub(crate) fn physical(&self) -> &PhysicalExecutionPlan {
        &self.physical
    }

    /// Estimator named by the sealed plan: the Bayesian temporal g-computation
    /// for pulse and one-step sustained contrasts, sequential g-computation for
    /// multi-step schedules.
    #[must_use]
    pub(crate) fn estimator(&self) -> EstimatorId {
        Self::estimator_for(&self.target)
    }

    #[must_use]
    pub(crate) fn matches_query(&self, query: &CausalQuery) -> bool {
        self.target.matches_query(query)
    }

    /// Confirm a prepared class envelope remains the exact proof retained by
    /// the operation. Refresh re-checks the handle's envelope through this.
    #[must_use]
    pub(crate) fn matches_class_proof(&self, proof: &CachedTemporalClassIdentification) -> bool {
        same_temporal_class_proof(&self.target.bundle().envelope, &proof.envelope)
    }
}

fn checked_settings(
    inference: &InferenceMode,
    validation: RefuteSuite,
) -> Result<BayesianConfig, CausalError> {
    super::checked_bayesian_temporal_effect::checked_bayesian_settings(
        inference,
        validation,
        "checked Bayesian temporal class operation requires Bayesian inference",
        "checked Bayesian temporal class operation supports none, cheap, or full validation",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bayesian_temporal_class_operation_rejects_frequentist_mode() {
        let error = checked_settings(&InferenceMode::Frequentist, RefuteSuite::None).unwrap_err();
        assert!(matches!(error, CausalError::Compile { .. }));
    }

    #[test]
    fn bayesian_temporal_class_operation_rejects_placebo_suite() {
        let error = checked_settings(
            &InferenceMode::Bayesian(BayesianConfig::conjugate()),
            RefuteSuite::PlaceboAndRcc,
        )
        .unwrap_err();
        assert!(matches!(error, CausalError::Unsupported { .. }));
    }
}
