//! Checked Bayesian operation for temporal pulse and sustained effects.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{BayesianConfig, ClassPrior, InferenceMode};
use antecedent_core::CausalQuery;

use super::builder::RefuteSuite;
use super::checked_temporal_class_effect::CheckedTemporalClassEffectOperation;
use super::checked_temporal_effect::CheckedTemporalEffectOperation;
use super::prepared::CachedTemporalClassIdentification;
use crate::error::CausalError;

#[path = "execute/checked_bayesian_temporal_kernel.rs"]
mod executable_kernel;
pub(crate) use executable_kernel::{CheckedBayesianTemporalDagFit, fit_temporal_dag_effect};

/// The target proof is retained alongside the Bayesian model settings. The
/// wrapped checked operations own the graph, query, full identification proof,
/// unfolding indexers, and the pulse versus sustained estimator choice.
#[derive(Clone, Debug)]
pub(crate) enum CheckedBayesianTemporalTarget {
    Dag(CheckedTemporalEffectOperation),
    Class(CheckedTemporalClassEffectOperation),
}

/// A complete Bayesian temporal effect operation for a fixed DAG or a supplied
/// TemporalCpdag/TemporalPag. Graph-posterior atoms intentionally have a
/// separate contract and are not represented here.
#[derive(Clone, Debug)]
pub(crate) struct CheckedBayesianTemporalEffectOperation {
    target: CheckedBayesianTemporalTarget,
    config: BayesianConfig,
    validation: RefuteSuite,
    class_prior: Option<ClassPrior>,
}

impl CheckedBayesianTemporalEffectOperation {
    pub(crate) fn checked(
        target: CheckedBayesianTemporalTarget,
        inference: &InferenceMode,
        validation: RefuteSuite,
        class_prior: Option<ClassPrior>,
    ) -> Result<Self, CausalError> {
        let config = checked_settings(
            inference,
            validation,
            class_prior.is_some(),
            matches!(&target, CheckedBayesianTemporalTarget::Class(_)),
        )?;
        Ok(Self { target, config, validation, class_prior })
    }

    #[must_use]
    pub(crate) fn target(&self) -> &CheckedBayesianTemporalTarget {
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
    pub(crate) fn query(&self) -> &antecedent_core::TemporalEffectQuery {
        match &self.target {
            CheckedBayesianTemporalTarget::Dag(operation) => operation.query(),
            CheckedBayesianTemporalTarget::Class(operation) => operation.query(),
        }
    }

    #[must_use]
    pub(crate) fn matches_query(&self, query: &CausalQuery) -> bool {
        matches!(query, CausalQuery::TemporalEffect(candidate) if candidate == self.query())
    }

    /// Confirm a prepared class envelope remains the exact proof retained by
    /// the operation. This is used by refresh and artifact verification hooks.
    #[must_use]
    pub(crate) fn matches_class_proof(&self, proof: &CachedTemporalClassIdentification) -> bool {
        let CheckedBayesianTemporalTarget::Class(operation) = &self.target else {
            return false;
        };
        let expected = operation.bundle();
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
    has_class_prior: bool,
    is_class_target: bool,
) -> Result<BayesianConfig, CausalError> {
    let InferenceMode::Bayesian(config) = inference else {
        return Err(CausalError::Compile {
            message: "checked Bayesian temporal operation requires Bayesian inference".into(),
        });
    };
    if !matches!(validation, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full) {
        return Err(CausalError::Unsupported {
            message: "checked Bayesian temporal operation supports none, cheap, or full validation",
        });
    }
    if has_class_prior && !is_class_target {
        return Err(CausalError::Compile {
            message: "temporal class prior requires a retained TemporalCpdag or TemporalPag proof"
                .into(),
        });
    }
    Ok(config.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bayesian_temporal_operation_rejects_frequentist_mode() {
        let error = checked_settings(&InferenceMode::Frequentist, RefuteSuite::None, false, true)
            .unwrap_err();
        assert!(matches!(error, CausalError::Compile { .. }));
    }

    #[test]
    fn bayesian_temporal_operation_rejects_class_prior_for_fixed_dag() {
        let error = checked_settings(
            &InferenceMode::Bayesian(BayesianConfig::conjugate()),
            RefuteSuite::None,
            true,
            false,
        )
        .unwrap_err();
        assert!(matches!(error, CausalError::Compile { .. }));
    }
}
