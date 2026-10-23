//! The typed refusal of a posterior for a nearest-neighbor matching contrast.
//!
//! A matching contrast is the mean of observed treated-minus-matched-control outcome
//! differences on the units the distance, the tie rule, the caliper and donor reuse select. It
//! is a rule applied to the sample, not a model of the outcomes or of the assignment, so there
//! is no likelihood for a prior to update. A prior added to the weights is not a sampling
//! weight and changes the estimand it claims to inform. Resampling the units to stand in for a
//! posterior fails for a different reason: the matching functional is not smooth in the
//! empirical measure at a fixed number of matches, so a Bayesian-bootstrap or ordinary
//! bootstrap draw is inconsistent for its law (Abadie and Imbens 2008).
//!
//! Both matching settings answer the same way, with the next attempt that does reach a
//! posterior: change the estimand to one that has a likelihood and say so.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::reason_code;

use crate::EstimationError;

/// Which matching contrast a posterior was asked for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatchingPosteriorRequest {
    /// Covariate or propensity-score matching on one cross-section.
    Static,
    /// Matching on pre-treatment history within calendar periods.
    WithinPeriodHistory,
}

/// Refuse a posterior of the matching contrast, naming the missing premise and the next attempt.
#[must_use]
pub fn refuse_matching_posterior(request: MatchingPosteriorRequest) -> EstimationError {
    let setting = match request {
        MatchingPosteriorRequest::Static => {
            "the nearest-neighbor matching contrast on covariates or a propensity score"
        }
        MatchingPosteriorRequest::WithinPeriodHistory => {
            "the history-matched contrast, matched within calendar periods"
        }
    };
    let next = match request {
        MatchingPosteriorRequest::Static => {
            "next attempt: declare a Bayesian outcome model on the same covariates (the Bayesian \
             g-computation route with a Treated target population on the Gaussian identity \
             model) and report its model-based effect on the treated as that different estimand"
        }
        MatchingPosteriorRequest::WithinPeriodHistory => {
            "next attempt: declare a Bayesian outcome model on the pre-treatment history and \
             calendar period (a model with a stated likelihood and prior; select the estimator \
             bayesian.period_history_outcome_model) and report its model-based effect on the \
             treated as that different estimand"
        }
    };
    EstimationError::refused(
        reason_code!("route_not_supported"),
        format!(
            "no posterior for {setting}: a matching rule has no likelihood for the outcomes or \
             the assignment, so a prior has nothing to update and a prior is not a sampling \
             weight; resampling the units is not a substitute because the fixed-number \
             nearest-neighbor matching functional is not smooth in the empirical measure \
             (Abadie and Imbens 2008); {next}"
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(request: MatchingPosteriorRequest) -> String {
        match refuse_matching_posterior(request) {
            EstimationError::Refused { code, message } => {
                assert_eq!(code, "route_not_supported");
                message
            }
            other => panic!("expected a registered refusal, got {other}"),
        }
    }

    #[test]
    fn the_static_refusal_names_the_premise_and_the_next_attempt() {
        let message = text(MatchingPosteriorRequest::Static);
        assert!(message.contains("no likelihood"), "{message}");
        assert!(message.contains("a prior is not a sampling weight"), "{message}");
        assert!(message.contains("not smooth in the empirical measure"), "{message}");
        assert!(message.contains("next attempt: declare a Bayesian outcome model"), "{message}");
        assert!(message.contains("covariates or a propensity score"), "{message}");
    }

    #[test]
    fn the_history_refusal_keeps_its_own_setting_and_next_attempt() {
        let message = text(MatchingPosteriorRequest::WithinPeriodHistory);
        assert!(message.contains("matched within calendar periods"), "{message}");
        assert!(message.contains("pre-treatment history and calendar period"), "{message}");
        assert!(!message.contains("covariates or a propensity score"), "{message}");
    }
}
