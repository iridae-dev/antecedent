//! What a temporal response-curve posterior means, and when it may be called ordinary.
//!
//! The temporal curve posterior of [`crate::temporal_response`] is a tempered generalized
//! posterior: each horizon's Gaussian working likelihood is divided by a long-run-variance
//! factor so that serially dependent innovations do not make the credible interval too narrow.
//! A tempered posterior has a declared loss, prior and learning rate but it is not the
//! posterior of a stated joint law of the series, and a credible interval from it is not
//! interchangeable with the interval of an ordinary likelihood.
//!
//! An ordinary posterior for the curve needs the innovation law of the series to be part of
//! the likelihood. This module records that premise: [`SerialErrorModel::IndependentInnovations`]
//! (the caller attests that conditional on the modelled history the innovations are
//! independent, so the working likelihood is the true one) licenses the ordinary reading;
//! [`SerialErrorModel::Unmodeled`] does not, and a request for the ordinary reading is
//! refused with the premise that is missing and the next attempt that reaches it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::EstimationError;

/// The innovation dependence the temporal curve likelihood takes as given.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SerialErrorModel {
    /// Serial dependence of the innovations is not part of the likelihood.
    Unmodeled,
    /// The innovations are attested independent given the modelled history.
    IndependentInnovations,
}

/// The meaning of the posterior a temporal curve fit reports.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TemporalCurvePosteriorMeaning {
    /// The posterior of a stated joint likelihood.
    OrdinaryLikelihood,
    /// A generalized posterior with a long-run-variance tempering factor.
    TemperedGeneralized,
}

/// The meaning of the temporal curve posterior under the declared innovation model.
#[must_use]
pub const fn temporal_curve_posterior_meaning(
    serial: SerialErrorModel,
) -> TemporalCurvePosteriorMeaning {
    match serial {
        SerialErrorModel::IndependentInnovations => {
            TemporalCurvePosteriorMeaning::OrdinaryLikelihood
        }
        SerialErrorModel::Unmodeled => TemporalCurvePosteriorMeaning::TemperedGeneralized,
    }
}

/// Require the ordinary-likelihood reading of a temporal curve posterior.
///
/// # Errors
///
/// A refusal when the innovation dependence is [`SerialErrorModel::Unmodeled`]. The missing
/// premise is a likelihood for the serial dependence of the innovations. The next attempt is
/// one of: attest independent innovations given the modelled history, add an explicit
/// autoregressive innovation law to the likelihood, or keep the tempered generalized
/// reading that the temporal curve cell reports.
pub fn require_ordinary_temporal_curve_posterior(
    serial: SerialErrorModel,
) -> Result<(), EstimationError> {
    match temporal_curve_posterior_meaning(serial) {
        TemporalCurvePosteriorMeaning::OrdinaryLikelihood => Ok(()),
        TemporalCurvePosteriorMeaning::TemperedGeneralized => Err(EstimationError::refused(
            antecedent_core::reason_code!("cell_not_licensed"),
            "temporal_curve.ordinary_posterior_not_licensed: the innovation dependence is not \
             part of the likelihood, so the curve posterior is the tempered generalized \
             posterior and not the posterior of a joint law of the series; next attempt: \
             attest independent innovations given the modelled history, add an explicit \
             autoregressive innovation law to the likelihood, or read the tempered posterior \
             as a generalized posterior",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unmodeled_serial_dependence_refuses_the_ordinary_reading_and_names_the_next_attempt() {
        let refused =
            require_ordinary_temporal_curve_posterior(SerialErrorModel::Unmodeled).unwrap_err();
        let text = refused.to_string();
        assert!(text.contains("temporal_curve.ordinary_posterior_not_licensed"));
        assert!(text.contains("next attempt"));
        assert_eq!(
            temporal_curve_posterior_meaning(SerialErrorModel::Unmodeled),
            TemporalCurvePosteriorMeaning::TemperedGeneralized
        );
    }

    #[test]
    fn attested_independent_innovations_license_the_ordinary_reading() {
        assert!(
            require_ordinary_temporal_curve_posterior(SerialErrorModel::IndependentInnovations)
                .is_ok()
        );
        assert_eq!(
            temporal_curve_posterior_meaning(SerialErrorModel::IndependentInnovations),
            TemporalCurvePosteriorMeaning::OrdinaryLikelihood
        );
    }
}
