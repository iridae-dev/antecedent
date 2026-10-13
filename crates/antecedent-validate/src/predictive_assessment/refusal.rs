//! Typed refusals of the predictive assessment contract.
//!
//! Every refusal carries a registered reason code, a namespaced
//! `predictive_assessment.*` detail and a message that states the declaration
//! or evidence that would license the request.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::reason_code;
use thiserror::Error;

/// Namespaced `detail` slots of the predictive assessment refusals.
pub mod detail {
    /// The law and the observations are not about the same quantity, population or units.
    pub const TARGET_MISMATCH: &str = "predictive_assessment.target_mismatch";
    /// The law and the observations are not about the same horizon.
    pub const HORIZON_MISMATCH: &str = "predictive_assessment.horizon_mismatch";
    /// A law is not a well formed predictive law.
    pub const INVALID_LAW: &str = "predictive_assessment.invalid_law";
    /// The realized observations are not a well formed set.
    pub const INVALID_OBSERVATIONS: &str = "predictive_assessment.invalid_observations";
    /// An outcome is not of the kind its law predicts.
    pub const OUTCOME_KIND: &str = "predictive_assessment.outcome_kind";
    /// A requested score is not defined for a law.
    pub const SCORE_NOT_DEFINED: &str = "predictive_assessment.score_not_defined";
    /// A requested diagnostic is not defined for a law.
    pub const DIAGNOSTIC_NOT_DEFINED: &str = "predictive_assessment.diagnostic_not_defined";
    /// The log score is infinite at the realized outcome.
    pub const LOG_SCORE_INFINITE: &str = "predictive_assessment.log_score_infinite";
    /// The law is supplied as a callback.
    pub const LAW_CALLBACK_NOT_BUILT: &str = "predictive_assessment.law_callback_not_built";
    /// A provider identity is incomplete.
    pub const PROVIDER_IDENTITY_INCOMPLETE: &str =
        "predictive_assessment.provider_identity_incomplete";
    /// The validation design is not well formed.
    pub const DESIGN_INVALID: &str = "predictive_assessment.design_invalid";
    /// An iid assumption was declared for a temporal or clustered design.
    pub const IID_NOT_LICENSED: &str =
        "predictive_assessment.iid_not_licensed_for_dependent_design";
    /// The declared serial dependence is shorter than the forecast horizon implies.
    pub const SERIAL_LAG_BELOW_HORIZON: &str = "predictive_assessment.serial_lag_below_horizon";
    /// Too few clusters for a cluster-robust variance.
    pub const TOO_FEW_CLUSTERS: &str = "predictive_assessment.too_few_clusters";
    /// Observations the law was fitted on are among the assessed ones.
    pub const IN_SAMPLE: &str = "predictive_assessment.in_sample";
    /// What the law was fitted on is not declared.
    pub const OVERLAP_UNKNOWN: &str = "predictive_assessment.overlap_unknown";
    /// A score or score-difference interval was requested.
    pub const INTERVAL_NOT_LICENSED: &str = "predictive_assessment.interval_not_licensed";
    /// A predictive score was offered as a causal claim.
    pub const NOT_A_CAUSAL_CLAIM: &str = "predictive_assessment.not_a_causal_claim";
    /// A predictive score was offered as a probability that a claim holds.
    pub const SCORE_NOT_PROBABILITY: &str = "predictive_assessment.score_not_probability";
    /// A declared bound was exceeded.
    pub const BOUNDS_EXCEEDED: &str = "predictive_assessment.bounds_exceeded";
    /// The assessment was cancelled before a result existed.
    pub const CANCELLED: &str = "predictive_assessment.cancelled";
}

/// Typed refusal of the predictive assessment contract.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("{detail}: {message}")]
pub struct PredictiveAssessmentError {
    /// Registered reason code.
    pub code: &'static str,
    /// Namespaced `predictive_assessment.*` detail.
    pub detail: &'static str,
    /// Human-readable context, including what would license the request.
    pub message: String,
}

impl PredictiveAssessmentError {
    /// Registered reason code.
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        self.code
    }

    /// Namespaced detail.
    #[must_use]
    pub const fn detail(&self) -> &'static str {
        self.detail
    }
}

pub(crate) fn refuse(
    code: &'static str,
    detail: &'static str,
    message: impl Into<String>,
) -> PredictiveAssessmentError {
    PredictiveAssessmentError { code, detail, message: message.into() }
}

pub(crate) fn invalid(
    detail: &'static str,
    message: impl Into<String>,
) -> PredictiveAssessmentError {
    refuse(reason_code!("invalid_argument"), detail, message)
}

pub(crate) fn unsupported(
    detail: &'static str,
    message: impl Into<String>,
) -> PredictiveAssessmentError {
    refuse(reason_code!("route_not_supported"), detail, message)
}

pub(crate) fn bounds(message: impl Into<String>) -> PredictiveAssessmentError {
    unsupported(detail::BOUNDS_EXCEEDED, message)
}

pub(crate) fn cancelled() -> PredictiveAssessmentError {
    refuse(
        reason_code!("cancelled_no_claim"),
        detail::CANCELLED,
        "the assessment was cancelled before a result existed; this is not a verdict on the \
         predictive law. Rerun without cancellation.",
    )
}
