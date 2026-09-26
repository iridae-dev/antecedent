//! Effect target retained by the graph-posterior checked operations.
//!
//! Graph-posterior compositions identify every atom for the inner average
//! effect and then estimate either that average effect or a conditional
//! effect over the same atoms. The retained operation records which of the
//! two it was sealed for so execution rebuilds the click with the same query
//! kind and estimator procedure.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{AverageEffectQuery, CausalQuery, ConditionalEffectQuery};

use crate::{EstimatorId, InferenceMode};

/// Stable inspector label for a frozen inference mode.
#[must_use]
pub(crate) fn inference_label(inference: &InferenceMode) -> String {
    match inference {
        InferenceMode::Bayesian(config) => format!("bayesian:{:?}", config.backend),
        InferenceMode::Frequentist => "frequentist".to_owned(),
    }
}

/// Query kind a graph-posterior effect operation was sealed for.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum GraphPosteriorEffectTarget {
    /// Posterior-weighted average effect.
    Average(AverageEffectQuery),
    /// Posterior-weighted conditional effect, including its modifiers.
    Conditional(ConditionalEffectQuery),
}

impl GraphPosteriorEffectTarget {
    /// The tabular effect targets a graph posterior can be sealed for.
    #[must_use]
    pub(crate) fn from_query(query: &CausalQuery) -> Option<Self> {
        match query {
            CausalQuery::AverageEffect(query) => Some(Self::Average(query.clone())),
            CausalQuery::ConditionalEffect(query) => Some(Self::Conditional(query.clone())),
            _ => None,
        }
    }

    /// The average-effect query every posterior atom is identified for.
    #[must_use]
    pub(crate) const fn inner(&self) -> &AverageEffectQuery {
        match self {
            Self::Average(query) => query,
            Self::Conditional(query) => &query.inner,
        }
    }

    /// The query the click study executes.
    #[must_use]
    pub(crate) fn causal_query(&self) -> CausalQuery {
        match self {
            Self::Average(query) => CausalQuery::AverageEffect(query.clone()),
            Self::Conditional(query) => CausalQuery::ConditionalEffect(query.clone()),
        }
    }

    /// The query the prepare-time atom identification cache binds.
    #[must_use]
    pub(crate) fn atom_identification_query(&self) -> CausalQuery {
        CausalQuery::AverageEffect(self.inner().clone())
    }

    #[must_use]
    pub(crate) const fn is_conditional(&self) -> bool {
        matches!(self, Self::Conditional(_))
    }

    /// Frequentist adjustment procedure licensed for this target.
    #[must_use]
    pub(crate) const fn frequentist_estimator(&self) -> EstimatorId {
        match self {
            Self::Average(_) => EstimatorId::LinearAdjustmentAte,
            Self::Conditional(_) => EstimatorId::ConditionalLinearAdjustment,
        }
    }

    /// Bayesian g-computation procedure licensed for this target.
    #[must_use]
    pub(crate) const fn bayesian_estimator(&self) -> EstimatorId {
        match self {
            Self::Average(_) => EstimatorId::BayesianGcomp,
            Self::Conditional(_) => EstimatorId::BayesianConditional,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::VariableId;

    fn ate() -> AverageEffectQuery {
        AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
    }

    #[test]
    fn targets_record_query_kind_and_licensed_procedures() {
        let average = GraphPosteriorEffectTarget::from_query(&CausalQuery::AverageEffect(ate()))
            .expect("average effect target");
        assert!(!average.is_conditional());
        assert_eq!(average.frequentist_estimator(), EstimatorId::LinearAdjustmentAte);
        assert_eq!(average.bayesian_estimator(), EstimatorId::BayesianGcomp);
        assert_eq!(average.causal_query(), CausalQuery::AverageEffect(ate()));

        let conditional =
            ConditionalEffectQuery::try_new(ate().with_effect_modifiers([VariableId::from_raw(2)]))
                .unwrap();
        let target = GraphPosteriorEffectTarget::from_query(&CausalQuery::ConditionalEffect(
            conditional.clone(),
        ))
        .expect("conditional effect target");
        assert!(target.is_conditional());
        assert_eq!(target.inner(), &conditional.inner);
        assert_eq!(target.frequentist_estimator(), EstimatorId::ConditionalLinearAdjustment);
        assert_eq!(target.bayesian_estimator(), EstimatorId::BayesianConditional);
        assert_eq!(target.causal_query(), CausalQuery::ConditionalEffect(conditional.clone()));
        assert_eq!(
            target.atom_identification_query(),
            CausalQuery::AverageEffect(conditional.inner)
        );

        assert!(
            GraphPosteriorEffectTarget::from_query(&CausalQuery::Mediation(
                antecedent_core::MediationQuery::binary(
                    VariableId::from_raw(0),
                    VariableId::from_raw(1),
                    [VariableId::from_raw(2)],
                    antecedent_core::MediationContrast::Total,
                )
            ))
            .is_none()
        );
    }
}
