//! Checked Bayesian operation for TemporalCpdag/TemporalPag pulse and sustained effects.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CausalQuery;

use super::builder::RefuteSuite;
use super::checked_temporal_class_effect::CheckedTemporalClassEffectOperation;
use super::prepared::CachedTemporalClassIdentification;
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
        if target.query().is_multi_step_sustained() {
            EstimatorId::TemporalSequentialGcomp
        } else {
            EstimatorId::BayesianTemporalGcomp
        }
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
        let expected = self.target.bundle();
        let left = &expected.envelope.envelope;
        let right = &proof.envelope.envelope;
        expected.envelope.indexers == proof.envelope.indexers
            && left.status == right.status
            && left.identified_weight.0.to_bits() == right.identified_weight.0.to_bits()
            && left.unidentified_weight.0.to_bits() == right.unidentified_weight.0.to_bits()
            && left.truncated_completions == right.truncated_completions
            && left.cases.len() == right.cases.len()
            && left.cases.iter().zip(&right.cases).all(|(a, b)| {
                a.graph.fingerprint() == b.graph.fingerprint()
                    && a.weight.0.to_bits() == b.weight.0.to_bits()
                    && a.result.query == b.result.query
                    && a.result.status == b.result.status
                    && a.result.estimands.len() == b.result.estimands.len()
                    && a.result.estimands.iter().zip(&b.result.estimands).all(|(x, y)| {
                        x.method == y.method
                            && x.adjustment_set == y.adjustment_set
                            && x.instruments == y.instruments
                            && x.mediators == y.mediators
                            && x.rd_design == y.rd_design
                            && format!("{:?}", a.result.arena.node(x.functional))
                                == format!("{:?}", b.result.arena.node(y.functional))
                    })
            })
    }
}

fn checked_settings(
    inference: &InferenceMode,
    validation: RefuteSuite,
) -> Result<BayesianConfig, CausalError> {
    let InferenceMode::Bayesian(config) = inference else {
        return Err(CausalError::Compile {
            message: "checked Bayesian temporal class operation requires Bayesian inference".into(),
        });
    };
    if !matches!(validation, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full) {
        return Err(CausalError::Unsupported {
            message: "checked Bayesian temporal class operation supports none, cheap, or full validation",
        });
    }
    Ok(config.clone())
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
