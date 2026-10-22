//! Typed ports a decision may consume, above [`crate::evidence_envelope::EvidenceEnvelope`].
//!
//! The envelope binds a quantity and keeps uncertainty on separate
//! [`UncertaintyAxis`] fields. A port is the representation offered to a
//! decision request: a mean, a covariance or influence, a confidence region,
//! a sampling law, a parameter posterior, a predictive law, an interventional
//! law, an identified set, a likelihood, or a scenario. A port may name the
//! envelope axis it reports. It is not a second copy of that axis, and two
//! ports may report the same axis without being the same port.
//!
//! [`EvidencePort::try_coerce`] does not retag a port. A confidence interval
//! or confidence region cannot become a parameter posterior: that request
//! stays a parameter posterior. Parameter draws cannot become individual
//! outcome risk: that request needs a predictive or interventional law. The
//! error is the failed attempt, and it names the legal port that can still
//! carry the same request.
//!
//! This module does not replace [`crate::evidence_envelope::EvidenceEnvelope`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{Refusal, impl_refused, named_variants};

use crate::evidence_envelope::UncertaintyAxis;

const PARAMETER_POSTERIOR_PORTS: &[EvidencePort] = &[EvidencePort::ParameterPosterior];
const INDIVIDUAL_OUTCOME_PORTS: &[EvidencePort] =
    &[EvidencePort::PredictiveLaw, EvidencePort::InterventionalLaw];

/// Representation a decision request may consume.
///
/// Declaration order is the order of [`Self::ALL`]. A confidence interval is a
/// one-coordinate [`Self::ConfidenceRegion`]. Draws from a parameter or
/// functional posterior are [`Self::ParameterPosterior`], not an outcome law.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum EvidencePort {
    /// Point or mean. Enough for a nominal affine comparison, not an uncertainty law.
    Mean,
    /// Covariance or influence contributions for linear frequentist propagation.
    ///
    /// Cross terms and sampling units belong to this port. A posterior
    /// covariance is part of [`Self::ParameterPosterior`].
    CovarianceInfluence,
    /// Coverage region or band, including a one-coordinate confidence interval.
    ///
    /// Coverage is not posterior probability mass.
    ConfidenceRegion,
    /// Repeated-sampling or bootstrap law. Not a belief over the true state.
    SamplingLaw,
    /// Parameter or functional posterior, including posterior draws.
    ///
    /// Bayesian uncertainty and expected loss stay on this port. The draws are
    /// not individual outcome risk.
    ParameterPosterior,
    /// Predictive law of outcomes. Outcome variation, not parameter uncertainty.
    PredictiveLaw,
    /// Interventional law of outcomes under an intervention.
    ///
    /// Distinct from [`Self::PredictiveLaw`] even though both report
    /// [`UncertaintyAxis::Predictive`].
    InterventionalLaw,
    /// Identified set or structural envelope. No implicit probabilities.
    IdentifiedSet,
    /// Likelihood or observation model. Not an uncertainty axis.
    Likelihood,
    /// Scenario or assumption range. Not an empirical probability.
    Scenario,
}

named_variants! {
    EvidencePort {
        Mean => "mean",
        CovarianceInfluence => "covariance_influence",
        ConfidenceRegion => "confidence_region",
        SamplingLaw => "sampling_law",
        ParameterPosterior => "parameter_posterior",
        PredictiveLaw => "predictive_law",
        InterventionalLaw => "interventional_law",
        IdentifiedSet => "identified_set",
        Likelihood => "likelihood",
        Scenario => "scenario",
    }
}

impl EvidencePort {
    /// Envelope axis this port reports, when the port is an uncertainty statement.
    ///
    /// [`Self::Mean`], [`Self::Likelihood`], and [`Self::Scenario`] report no
    /// axis. [`Self::PredictiveLaw`] and [`Self::InterventionalLaw`] both
    /// report [`UncertaintyAxis::Predictive`] and stay different ports.
    #[must_use]
    pub const fn uncertainty_axis(self) -> Option<UncertaintyAxis> {
        match self {
            Self::Mean | Self::Likelihood | Self::Scenario => None,
            Self::CovarianceInfluence | Self::ConfidenceRegion | Self::SamplingLaw => {
                Some(UncertaintyAxis::Sampling)
            }
            Self::ParameterPosterior => Some(UncertaintyAxis::Parameter),
            Self::PredictiveLaw | Self::InterventionalLaw => Some(UncertaintyAxis::Predictive),
            Self::IdentifiedSet => Some(UncertaintyAxis::Identification),
        }
    }

    /// Offer this port as the carrier of `request`.
    ///
    /// Succeeds only when this port is already a legal carrier. The value
    /// returned is this port, unchanged. A confidence region offered as a
    /// parameter posterior is refused, and the posterior request stays
    /// [`Self::ParameterPosterior`]. Parameter draws offered as individual
    /// outcome risk are refused; that request needs [`Self::PredictiveLaw`] or
    /// [`Self::InterventionalLaw`].
    ///
    /// # Errors
    ///
    /// Returns an error when this port is not a legal carrier of `request`.
    /// The error names the ports that can still carry the request.
    #[must_use = "an illegal port coercion is refused"]
    pub const fn try_coerce(self, request: DecisionRequest) -> Result<Self, EvidencePortError> {
        if request.admits(self) {
            Ok(self)
        } else {
            Err(EvidencePortError::coercion(self, request))
        }
    }
}

/// What the caller asked a port to answer.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DecisionRequest {
    /// Bayesian uncertainty or expected loss over a parameter or functional.
    ParameterPosterior,
    /// Risk, threshold, or quantile of an individual outcome.
    IndividualOutcomeRisk,
}

impl DecisionRequest {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ParameterPosterior => "parameter_posterior",
            Self::IndividualOutcomeRisk => "individual_outcome_risk",
        }
    }

    /// Ports that can carry this request.
    ///
    /// A parameter-posterior request stays [`EvidencePort::ParameterPosterior`].
    /// Individual outcome risk needs [`EvidencePort::PredictiveLaw`] or
    /// [`EvidencePort::InterventionalLaw`].
    #[must_use]
    pub const fn legal_ports(self) -> &'static [EvidencePort] {
        match self {
            Self::ParameterPosterior => PARAMETER_POSTERIOR_PORTS,
            Self::IndividualOutcomeRisk => INDIVIDUAL_OUTCOME_PORTS,
        }
    }

    /// Whether `port` is already a legal carrier of this request.
    #[must_use]
    pub const fn admits(self, port: EvidencePort) -> bool {
        matches!(
            (self, port),
            (Self::ParameterPosterior, EvidencePort::ParameterPosterior)
                | (
                    Self::IndividualOutcomeRisk,
                    EvidencePort::PredictiveLaw | EvidencePort::InterventionalLaw,
                )
        )
    }
}

/// A refused attempt to coerce a port into a decision request.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("{detail}: {message}")]
pub struct EvidencePortError {
    /// Registered runtime-refusal reason code.
    pub code: &'static str,
    /// Stable `evidence_ports.*` detail.
    pub detail: &'static str,
    /// What was refused.
    pub message: &'static str,
    /// Port the caller offered.
    pub source_port: EvidencePort,
    /// Decision request the offer tried to answer.
    pub request: DecisionRequest,
    /// Ports that can still carry `request`.
    ///
    /// The failed attempt does not retag `source_port` as one of these.
    pub legal_ports: &'static [EvidencePort],
}

impl EvidencePortError {
    /// The refusal as `(code, detail, message)`.
    #[must_use]
    pub const fn refusal(self) -> (&'static str, &'static str, &'static str) {
        Refusal::new(self.code, self.detail, self.message).into_parts()
    }

    const fn coercion(source: EvidencePort, request: DecisionRequest) -> Self {
        Self {
            code: antecedent_core::reason_code!("distribution_meaning_mismatch"),
            detail: "evidence_ports.coercion",
            message: coercion_message(source, request),
            source_port: source,
            request,
            legal_ports: request.legal_ports(),
        }
    }
}

impl_refused!(EvidencePortError, static_message);

const fn coercion_message(source: EvidencePort, request: DecisionRequest) -> &'static str {
    match (source, request) {
        (EvidencePort::ConfidenceRegion, DecisionRequest::ParameterPosterior) => {
            "a confidence interval or confidence region is not a parameter posterior"
        }
        (EvidencePort::ParameterPosterior, DecisionRequest::IndividualOutcomeRisk) => {
            "parameter draws are not individual outcome risk"
        }
        _ => "this evidence port cannot carry the decision request",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_region_cannot_become_a_parameter_posterior() {
        let region = EvidencePort::ConfidenceRegion;
        let request = DecisionRequest::ParameterPosterior;
        let err = region.try_coerce(request).unwrap_err();

        assert_eq!(err.source_port, region);
        assert_eq!(err.request, request);
        assert_eq!(err.legal_ports, &[EvidencePort::ParameterPosterior]);
        assert_eq!(
            err.refusal(),
            (
                "distribution_meaning_mismatch",
                "evidence_ports.coercion",
                "a confidence interval or confidence region is not a parameter posterior",
            )
        );
        assert!(antecedent_core::reason_code::is_runtime_refusal(err.code));
        assert_eq!(region.uncertainty_axis(), Some(UncertaintyAxis::Sampling));
        assert_ne!(region.uncertainty_axis(), EvidencePort::ParameterPosterior.uncertainty_axis());

        let legal = err.legal_ports[0];
        assert_eq!(legal.try_coerce(request), Ok(legal));
        assert_eq!(legal, EvidencePort::ParameterPosterior);
        assert_ne!(err.source_port, legal);

        let sampling = EvidencePort::SamplingLaw.try_coerce(request).unwrap_err();
        assert_eq!(sampling.legal_ports, err.legal_ports);
        assert_eq!(sampling.message, "this evidence port cannot carry the decision request");
    }

    #[test]
    fn parameter_draws_cannot_become_individual_outcome_risk() {
        let draws = EvidencePort::ParameterPosterior;
        let request = DecisionRequest::IndividualOutcomeRisk;
        let err = draws.try_coerce(request).unwrap_err();

        assert_eq!(err.source_port, draws);
        assert_eq!(err.request, request);
        assert_eq!(
            err.legal_ports,
            &[EvidencePort::PredictiveLaw, EvidencePort::InterventionalLaw]
        );
        assert_eq!(
            err.refusal(),
            (
                "distribution_meaning_mismatch",
                "evidence_ports.coercion",
                "parameter draws are not individual outcome risk",
            )
        );
        assert!(antecedent_core::reason_code::is_runtime_refusal(err.code));
        assert!(err.legal_ports.iter().all(|port| *port != err.source_port));

        let predictive = err.legal_ports[0];
        let interventional = err.legal_ports[1];
        assert_eq!(predictive.try_coerce(request), Ok(predictive));
        assert_eq!(interventional.try_coerce(request), Ok(interventional));
        assert_eq!(predictive.uncertainty_axis(), Some(UncertaintyAxis::Predictive));
        assert_eq!(interventional.uncertainty_axis(), Some(UncertaintyAxis::Predictive));
        assert_ne!(EvidencePort::PredictiveLaw, EvidencePort::InterventionalLaw);

        let posterior = draws.try_coerce(DecisionRequest::ParameterPosterior);
        assert_eq!(posterior, Ok(EvidencePort::ParameterPosterior));

        let mean = EvidencePort::Mean.try_coerce(request).unwrap_err();
        assert_eq!(mean.legal_ports, err.legal_ports);
        assert_eq!(mean.message, "this evidence port cannot carry the decision request");
    }

    #[test]
    fn ports_name_envelope_axes_without_replacing_them() {
        assert_eq!(EvidencePort::ALL.len(), 10);
        for port in EvidencePort::ALL {
            assert_eq!(EvidencePort::from_name(port.as_str()), Some(port));
        }
        assert_eq!(EvidencePort::from_name("parameter"), None);
        assert_eq!(EvidencePort::from_name("sampling"), None);
        assert_eq!(EvidencePort::from_name("predictive"), None);
        assert_eq!(EvidencePort::from_name(UncertaintyAxis::Identification.as_str()), None);

        assert_eq!(EvidencePort::Mean.uncertainty_axis(), None);
        assert_eq!(EvidencePort::Likelihood.uncertainty_axis(), None);
        assert_eq!(EvidencePort::Scenario.uncertainty_axis(), None);
        assert_eq!(
            EvidencePort::CovarianceInfluence.uncertainty_axis(),
            Some(UncertaintyAxis::Sampling)
        );
        assert_eq!(
            EvidencePort::IdentifiedSet.uncertainty_axis(),
            Some(UncertaintyAxis::Identification)
        );
        assert_eq!(
            DecisionRequest::ParameterPosterior.legal_ports(),
            &[EvidencePort::ParameterPosterior]
        );
        assert_eq!(
            DecisionRequest::IndividualOutcomeRisk.legal_ports(),
            &[EvidencePort::PredictiveLaw, EvidencePort::InterventionalLaw]
        );
        assert_eq!(DecisionRequest::IndividualOutcomeRisk.as_str(), "individual_outcome_risk");
    }
}
