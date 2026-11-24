//! Covariate adjustment a randomized-effect study's estimator makes from the rows.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::StudyBuilder;
use crate::analysis::randomized_adjustment::RandomizedAdjustment;

impl StudyBuilder {
    /// A covariate adjustment the estimator of a [`CausalQuery::RandomizedEffect`] study makes
    /// from the rows themselves, such as same-sample CUPED
    /// ([`RandomizedAdjustment::EstimatedCuped`]).
    ///
    /// The query states the experiment; this declares how its contrast is estimated. Same-sample
    /// CUPED selects the `randomized.estimated_cuped_ht_itt` estimator, which runs
    /// `antecedent_estimate::randomized_scores::estimated_cuped_contrast` on the outcome, the
    /// covariate column and the query's assignment and probabilities. The adjustment enters
    /// the study's program identity. [`Self::build`] refuses it (`option_not_applicable`) on any
    /// other query, on a query that declares a fixed CUPED coefficient, ANCOVA covariates,
    /// treatment receipt or an exact randomization test, under Bayesian inference, and with any
    /// other declared estimator.
    ///
    /// [`CausalQuery::RandomizedEffect`]: antecedent_core::CausalQuery::RandomizedEffect
    #[must_use]
    pub fn randomized_adjustment(mut self, adjustment: RandomizedAdjustment) -> Self {
        self.declared.randomized_adjustment = Some(adjustment);
        self
    }
}
