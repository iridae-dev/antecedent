//! Checked Bayesian operation for temporal pulse and sustained effects.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::analysis::prepared::Inspect;
use crate::strategy_table::EstimatorId;
use crate::strategy_table::IdentifierId;
use crate::{BayesianConfig, InferenceMode};
use antecedent_core::CausalQuery;
use std::sync::Arc;

use super::builder::RefuteSuite;
use super::checked_temporal_effect::CheckedTemporalEffectOperation;
use crate::error::CausalError;

#[path = "execute/checked_bayesian_temporal_kernel.rs"]
mod executable_kernel;
pub(crate) use executable_kernel::fit_temporal_dag_effect;

/// The target proof is retained alongside the Bayesian model settings. The
/// wrapped checked operations own the graph, query, full identification proof,
/// unfolding indexers, and the pulse versus sustained estimator choice.
/// A complete Bayesian temporal effect operation for a fixed temporal DAG.
/// Graph-posterior atoms have a separate contract and are not represented here.
#[derive(Clone, Debug)]
pub(crate) struct CheckedBayesianTemporalEffectOperation {
    target: CheckedTemporalEffectOperation,
    config: BayesianConfig,
    validation: RefuteSuite,
}

impl CheckedBayesianTemporalEffectOperation {
    pub(crate) fn checked(
        target: CheckedTemporalEffectOperation,
        inference: &InferenceMode,
        validation: RefuteSuite,
    ) -> Result<Self, CausalError> {
        let config = checked_settings(inference, validation)?;
        Ok(Self { target, config, validation })
    }

    #[must_use]
    pub(crate) fn target(&self) -> &CheckedTemporalEffectOperation {
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
    pub(crate) fn query(&self) -> &antecedent_core::TemporalEffectQuery {
        self.target.query()
    }

    #[must_use]
    pub(crate) fn matches_query(&self, query: &CausalQuery) -> bool {
        matches!(query, CausalQuery::TemporalEffect(candidate) if candidate == self.query())
    }
}

fn checked_settings(
    inference: &InferenceMode,
    validation: RefuteSuite,
) -> Result<BayesianConfig, CausalError> {
    checked_bayesian_settings(
        inference,
        validation,
        "checked Bayesian temporal operation requires Bayesian inference",
        "checked Bayesian temporal operation supports none, cheap, or full validation",
    )
}

/// Model settings of a checked Bayesian temporal operation: Bayesian
/// inference and a `none`, `cheap`, or `full` validation suite. The two
/// messages name the refusing family.
pub(super) fn checked_bayesian_settings(
    inference: &InferenceMode,
    validation: RefuteSuite,
    requires_bayesian: &str,
    validation_support: &'static str,
) -> Result<BayesianConfig, CausalError> {
    let InferenceMode::Bayesian(config) = inference else {
        return Err(CausalError::unsupported_inference(requires_bayesian));
    };
    if !validation.is_point() {
        return Err(CausalError::Unsupported { message: validation_support });
    }
    Ok(config.clone())
}

/// Read-only target and procedure of a sealed Bayesian `TemporalDag` effect.
#[derive(Clone, Debug)]
pub struct CheckedBayesianTemporalDagEffectInfo {
    /// Exact pulse or one-step sustained target.
    pub query: antecedent_core::TemporalEffectQuery,
    /// Identifier fixed during preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed during preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Validation suite fixed during preparation.
    pub validation: crate::analysis::builder::RefuteSuite,
    /// Requested posterior draws.
    pub posterior_draws: usize,
    /// Signature of the retained `TemporalDag`.
    pub graph_signature: std::sync::Arc<str>,
}

impl Inspect for CheckedBayesianTemporalEffectOperation {
    type Info = CheckedBayesianTemporalDagEffectInfo;

    fn info(&self) -> CheckedBayesianTemporalDagEffectInfo {
        let target = self.target();
        CheckedBayesianTemporalDagEffectInfo {
            query: target.query().clone(),
            identifier: IdentifierId::TemporalBackdoorUnfolded,
            estimator: EstimatorId::temporal_effect_procedure(target.query(), true),
            validation: self.validation(),
            posterior_draws: self.config().n_draws,
            graph_signature: Arc::from(target.graph_signature()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bayesian_temporal_operation_rejects_frequentist_mode() {
        let error = checked_settings(&InferenceMode::Frequentist, RefuteSuite::None).unwrap_err();
        let records = error.route_diagnostics();
        assert_eq!(records.len(), 1, "{error}");
        assert!(records[0].is_unsupported_inference(), "{error}");
    }
}
