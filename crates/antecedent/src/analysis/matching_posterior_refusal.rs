//! The typed refusal of a posterior for a nearest-neighbor matching contrast.
//!
//! A matching rule has no likelihood for the outcomes or the assignment, so a prior has nothing
//! to update, and resampling the units is not a substitute. Both matching settings answer with
//! the premise that is missing and the next attempt that does reach a posterior.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::matching_posterior_boundary::{
    MatchingPosteriorRequest, refuse_matching_posterior,
};

use crate::error::CausalError;

/// Which matching setting a posterior was asked for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatchingSetting {
    /// Covariate or propensity-score matching on one cross-section.
    Static,
    /// Matching on pre-treatment history within calendar periods.
    WithinPeriodHistory,
}

/// The `route_not_supported` refusal naming the missing premise and the next attempt.
#[must_use]
pub fn matching_posterior_refusal(setting: MatchingSetting) -> CausalError {
    let request = match setting {
        MatchingSetting::Static => MatchingPosteriorRequest::Static,
        MatchingSetting::WithinPeriodHistory => MatchingPosteriorRequest::WithinPeriodHistory,
    };
    CausalError::Estimate(refuse_matching_posterior(request))
}
