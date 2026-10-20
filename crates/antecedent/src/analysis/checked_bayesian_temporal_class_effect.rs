//! Checked Bayesian operation for TemporalCpdag/TemporalPag pulse and sustained effects.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::analysis::prepared::Inspect;
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
    physical: std::sync::Arc<PhysicalExecutionPlan>,
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
    /// sustained effect on a fixed `TemporalCpdag` or `TemporalPag` under Bayesian
    /// inference with the default point-estimate settings. The one-shot facade
    /// and the prepared sealer share this predicate.
    #[must_use]
    pub(crate) fn admits(study: &Study) -> bool {
        let CausalQuery::TemporalEffect(query) = &study.query else {
            return false;
        };
        matches!(study.data, DataInput::Temporal(_) | DataInput::Event(_))
            && study.plain_fixed_route()
            && matches!(study.graph.class(), GraphClass::TemporalCpdag | GraphClass::TemporalPag)
            && matches!(study.inference, InferenceMode::Bayesian(_))
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
        physical: std::sync::Arc<PhysicalExecutionPlan>,
    ) -> Result<Self, CausalError> {
        let config = checked_settings(inference, validation)?;
        let procedure = target.procedure();
        let (identifier, target_validation) = (procedure.identifier, procedure.validation);
        let custom_validators = procedure.custom_validators;
        if target_validation != validation || !custom_validators.is_empty() {
            return Err(CausalError::Compile {
                message:
                    "checked Bayesian temporal class target disagrees with its validation suite"
                        .into(),
            });
        }
        let query = CausalQuery::TemporalEffect(target.query().clone());
        if physical.logical.query != query
            || !physical.binding().names(identifier, Self::estimator_for(&target))
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

/// Read-only source graph, completion proof, and Bayesian model settings for a
/// temporal Cpdag/Pag pulse or sustained effect.
#[derive(Clone, Debug)]
pub struct CheckedBayesianTemporalClassEffectInfo {
    /// Exact policy, intervention levels, horizon, and target population.
    pub query: antecedent_core::TemporalEffectQuery,
    /// Source class retained with the completion proof.
    pub graph_class: crate::accepted::GraphClass,
    /// Identifier fixed during preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed during preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Validation suite fixed during preparation.
    pub validation: crate::analysis::builder::RefuteSuite,
    /// Requested posterior draws.
    pub posterior_draws: usize,
    /// Whether a caller-supplied class mass was sealed with the operation.
    pub class_prior_supplied: bool,
    /// Number of completion atoms retained by the checked proof.
    pub completion_count: usize,
    /// Maximum number of class completions searched, when configured.
    pub completion_limit: Option<usize>,
    /// Identified enumeration mass retained by the proof.
    pub identified_mass: f64,
    /// Unidentified enumeration mass retained by the proof.
    pub unidentified_mass: f64,
    /// Number of completions whose enumeration was cut off.
    pub truncated_completions: usize,
    /// Source graph version sealed with the proof.
    pub graph_version: u64,
}

impl Inspect for CheckedBayesianTemporalClassEffectOperation {
    type Info = CheckedBayesianTemporalClassEffectInfo;

    fn info(&self) -> CheckedBayesianTemporalClassEffectInfo {
        let target = self.target();
        let identifier = target.procedure().identifier;
        let envelope = &target.bundle().envelope.envelope;
        CheckedBayesianTemporalClassEffectInfo {
            query: target.query().clone(),
            graph_class: target.graph_class(),
            identifier,
            estimator: self.estimator(),
            validation: self.validation(),
            posterior_draws: self.config().n_draws,
            class_prior_supplied: self.class_prior().is_some(),
            completion_count: envelope.cases.len(),
            completion_limit: target.max_completions(),
            identified_mass: envelope.identified_weight.0,
            unidentified_mass: envelope.unidentified_weight.0,
            truncated_completions: envelope.truncated_completions,
            graph_version: u64::from(target.structure_version()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bayesian_temporal_class_operation_rejects_frequentist_mode() {
        let error = checked_settings(&InferenceMode::Frequentist, RefuteSuite::None).unwrap_err();
        let records = error.route_diagnostics();
        assert_eq!(records.len(), 1, "{error}");
        assert!(records[0].is_unsupported_inference(), "{error}");
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
