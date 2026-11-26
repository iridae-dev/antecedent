//! Covariate adjustments a randomized-effect study's estimator makes from the rows themselves.
//!
//! A [`RandomizedEffectQuery`] states the experiment: the assignment mechanism and its known
//! probabilities, the outcome and its unit labels, and any adjustment fixed outside the
//! experiment (a CUPED coefficient chosen before the outcomes were seen). An adjustment whose
//! coefficients are estimated from these same rows is a choice of method, not a property of
//! the experiment, so a study declares it with [`crate::StudyBuilder::randomized_adjustment`]
//! beside the query, as it declares an estimator.
//!
//! [`RandomizedAdjustment::EstimatedCuped`] (same-sample CUPED) residualizes the outcome by the
//! pooled outcome-on-covariate slope of one pre-assignment covariate and reads the independent
//! Bernoulli Horvitz-Thompson contrast with its score standard error: the
//! `randomized.estimated_cuped_ht_itt` estimator, which runs the kernel
//! `antecedent_estimate::randomized_scores::estimated_cuped_contrast`. The study's lowered
//! program names that method and covariate in the randomized cell's outcome slot, so a study
//! that declares the adjustment has a program identity of its own and a study that declares
//! none keeps the one it had; the contract's resolved estimator names the method, and the
//! estimator's diagnostic states the covariate as a structured field
//! (`antecedent_io::graphless_support::ESTIMATED_CUPED_COVARIATE_FIELD`), which the result
//! verifier requires.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    CausalQuery, RandomizationDesign, RandomizedEffectQuery, RandomizedEstimand, VariableId,
};

use crate::CausalError;
use crate::strategy_table::EstimatorId;

/// A covariate adjustment estimated from the rows of a randomized-effect study.
///
/// Declared with [`crate::StudyBuilder::randomized_adjustment`]. The query keeps its own
/// meaning: its `fixed_cuped` field only ever holds a coefficient fixed outside the experiment,
/// and the two adjustments are never combined.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RandomizedAdjustment {
    /// Same-sample CUPED: the outcome is residualized by the pooled slope of `covariate`
    /// estimated from these rows, and the contrast is the independent Bernoulli
    /// Horvitz-Thompson contrast of the residualized outcome with its score standard error.
    /// The slope's own estimation is not propagated into that standard error.
    ///
    /// Reads an intention-to-treat query under independent Bernoulli assignment with no
    /// fixed CUPED coefficient, ANCOVA covariates, treatment receipt or exact randomization
    /// test; `covariate` must be a complete continuous pre-assignment column other than the
    /// outcome. Frequentist inference only.
    EstimatedCuped {
        /// Pre-assignment covariate whose pooled outcome slope is estimated from the rows.
        covariate: VariableId,
    },
    /// Same kernel and estimator as [`Self::EstimatedCuped`]. The covariate is a prediction
    /// fixed from pre-assignment information; the predictive model's error is not in the
    /// reported variance.
    EstimatedCupac {
        /// Prediction column, one value per Bernoulli row.
        covariate: VariableId,
    },
    /// Slope of cluster outcome totals on cluster prediction totals, then the cluster Neyman
    /// contrast of the adjusted totals. The cluster stays the unit of analysis. The variance
    /// treats the slope as fixed.
    ClusterTotalCupac {
        /// Prediction column, summed to the assignment cluster.
        covariate: VariableId,
    },
}

impl RandomizedAdjustment {
    /// The pre-assignment covariate the adjustment reads.
    #[must_use]
    pub const fn covariate(&self) -> VariableId {
        match self {
            Self::EstimatedCuped { covariate }
            | Self::EstimatedCupac { covariate }
            | Self::ClusterTotalCupac { covariate } => *covariate,
        }
    }

    /// Bernoulli same-sample slope, whether or not the covariate is a supplied prediction.
    #[must_use]
    pub const fn bernoulli_slope(self) -> bool {
        matches!(self, Self::EstimatedCuped { .. } | Self::EstimatedCupac { .. })
    }
}

/// Graphless support coordinate of the same-sample CUPED reading of an independent Bernoulli
/// design: `(family, design, method, assignment_unit, inference_claim)`.
const ESTIMATED_CUPED_COORDINATE: (&str, &str, &str, &str, &str) = (
    "randomized_effect",
    "bernoulli",
    "estimated_cuped_ht_score",
    "unit",
    "pointwise_95_normal_interval",
);

/// Graphless coordinate of the cluster-total prediction slope. The interval is the cluster
/// Neyman interval of the adjusted totals; its coverage is not measured.
const CLUSTER_TOTAL_CUPAC_COORDINATE: (&str, &str, &str, &str, &str) = (
    "randomized_effect",
    "cluster",
    "estimated_cuped_cluster_totals",
    "cluster",
    "pointwise_95_normal_interval",
);

/// Why a declared adjustment does not apply to the study's query.
fn design_refusal(cluster: bool) -> CausalError {
    if cluster {
        crate::unsupported_reason!(
            "option_not_applicable",
            "cluster-total CUPAC requires complete cluster randomization, a distinct \
             pre-assignment prediction, and no other adjustment or exact test"
        )
    } else {
        crate::unsupported_reason!(
            "option_not_applicable",
            "estimated-slope CUPED requires independent Bernoulli assignment, a distinct \
             pre-assignment covariate, and no other adjustment or exact test"
        )
    }
}

/// The graphless support coordinate of a randomized query read under `adjustment`: the
/// query's own coordinate ([`antecedent_core::randomized_graphless_coordinate`]), or the
/// same-sample CUPED coordinate when the study declares that adjustment.
pub(crate) fn graphless_coordinate(
    query: &RandomizedEffectQuery,
    adjustment: Option<&RandomizedAdjustment>,
) -> (&'static str, &'static str, &'static str, &'static str, &'static str) {
    match adjustment {
        Some(adjustment) if adjustment.bernoulli_slope()
            && matches!(query.design, RandomizationDesign::Bernoulli) =>
        {
            ESTIMATED_CUPED_COORDINATE
        }
        Some(RandomizedAdjustment::ClusterTotalCupac { .. })
            if matches!(query.design, RandomizationDesign::Cluster { .. }) =>
        {
            CLUSTER_TOTAL_CUPAC_COORDINATE
        }
        _ => antecedent_core::randomized_graphless_coordinate(
            &query.design,
            query.estimand,
            query.received_treatment.is_some(),
            query.fixed_cuped.is_some(),
            !query.ancova_covariates.is_empty(),
        ),
    }
}

/// Refuse an adjustment the query cannot be read under.
///
/// # Errors
///
/// `option_not_applicable` when the query is not an unadjusted independent Bernoulli
/// intention-to-treat query, or when the covariate is the outcome. A fixed CUPED coefficient
/// and an estimated slope are two adjustments, so the pair is refused.
pub(crate) fn check(
    query: &RandomizedEffectQuery,
    adjustment: RandomizedAdjustment,
) -> Result<(), CausalError> {
    let covariate = adjustment.covariate();
    let design_ok = match adjustment {
        RandomizedAdjustment::EstimatedCuped { .. } | RandomizedAdjustment::EstimatedCupac { .. } => {
            matches!(query.design, RandomizationDesign::Bernoulli)
        }
        RandomizedAdjustment::ClusterTotalCupac { .. } => {
            matches!(query.design, RandomizationDesign::Cluster { .. })
        }
    };
    if !design_ok
        || query.estimand != RandomizedEstimand::IntentionToTreat
        || covariate == query.outcome
        || query.fixed_cuped.is_some()
        || query.cupac
        || !query.ancova_covariates.is_empty()
        || query.received_treatment.is_some()
        || query.exact_randomization_test
    {
        return Err(design_refusal(matches!(
            adjustment,
            RandomizedAdjustment::ClusterTotalCupac { .. }
        )));
    }
    Ok(())
}

/// Refuse at build a declared adjustment its study would not read, and the same-sample CUPED
/// estimator declared without the covariate it adjusts by.
///
/// # Errors
///
/// `option_not_applicable` when the study declares an adjustment on anything but a
/// frequentist [`CausalQuery::RandomizedEffect`] study its [`check`] admits, or with an
/// estimator other than `randomized.estimated_cuped_ht_itt`; `required_option_missing` when
/// that estimator is declared without an adjustment.
pub(crate) fn admit(study: &crate::Study) -> Result<(), CausalError> {
    let Some(adjustment) = study.declared.randomized_adjustment.as_ref() else {
        if matches!(
            study.declared.estimator,
            Some(
                EstimatorId::RandomizedEstimatedCupedHt
                    | EstimatorId::RandomizedClusterEstimatedCuped
            )
        ) {
            return Err(crate::unsupported_reason!(
                "required_option_missing",
                "the estimated-slope CUPED and cluster-total CUPAC estimators read a declared \
                 pre-assignment covariate"
            ));
        }
        return Ok(());
    };
    let CausalQuery::RandomizedEffect(query) = &study.query else {
        return Err(crate::unsupported_reason!(
            "option_not_applicable",
            "a randomized adjustment is read by a RandomizedEffectQuery study only"
        ));
    };
    check(query, *adjustment)?;
    if study.settings.inference.is_bayesian() {
        return Err(crate::unsupported_reason!(
            "option_not_applicable",
            "estimated-slope CUPED is a frequentist score estimator; the Bayesian randomized \
             estimators read no CUPED adjustment, so use inference=Frequentist"
        ));
    }
    let expected = match adjustment {
        RandomizedAdjustment::EstimatedCuped { .. } | RandomizedAdjustment::EstimatedCupac { .. } => {
            EstimatorId::RandomizedEstimatedCupedHt
        }
        RandomizedAdjustment::ClusterTotalCupac { .. } => EstimatorId::RandomizedClusterEstimatedCuped,
    };
    if study.declared.estimator.is_some_and(|estimator| estimator != expected) {
        return Err(crate::unsupported_reason!(
            "option_not_applicable",
            "a declared slope adjustment is estimated by its own route; omit estimator= or \
             declare that estimator"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bernoulli() -> RandomizedEffectQuery {
        RandomizedEffectQuery::bernoulli_itt(
            VariableId::from_raw(0),
            vec![true, false, true, false],
            vec![0.5; 4],
            ["a", "b", "c", "d"].map(std::sync::Arc::<str>::from).to_vec(),
            ["a", "b", "c", "d"].map(std::sync::Arc::<str>::from).to_vec(),
            ("control", "treated"),
        )
    }

    #[test]
    fn the_adjustment_selects_the_estimated_slope_coordinate_on_bernoulli_rows_only() {
        let query = bernoulli();
        let adjustment =
            RandomizedAdjustment::EstimatedCuped { covariate: VariableId::from_raw(1) };
        assert_eq!(graphless_coordinate(&query, Some(&adjustment)), ESTIMATED_CUPED_COORDINATE);
        assert_eq!(graphless_coordinate(&query, None).2, "independent_action_ht_score");
        assert_eq!(adjustment.covariate(), VariableId::from_raw(1));
    }

    #[test]
    fn the_adjustment_composes_with_no_other_declaration() {
        let adjustment =
            RandomizedAdjustment::EstimatedCuped { covariate: VariableId::from_raw(1) };
        assert!(check(&bernoulli(), adjustment).is_ok());
        for refused in [
            bernoulli().with_fixed_cuped(VariableId::from_raw(1), 1.3),
            bernoulli().with_ancova(vec![VariableId::from_raw(2)]),
            bernoulli().with_received_treatment(vec![true, false, true, false]),
            bernoulli().with_exact_randomization_test(),
        ] {
            assert!(check(&refused, adjustment).is_err(), "{refused:?}");
        }
        let outcome = RandomizedAdjustment::EstimatedCuped { covariate: VariableId::from_raw(0) };
        assert!(check(&bernoulli(), outcome).is_err());
    }
}
