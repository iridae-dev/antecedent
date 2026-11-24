//! Why a preference artifact is refused: one error for the record of a decision across a
//! preference set and the record of accepted preferences.
//!
//! The error is shared so both records keep one public refusal type. Its
//! [`PreferenceArtifactError::Accepted`] variant carries a refusal of accepted preferences; that
//! variant is the only reference from the decision record to the code that learns preferences.

// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ExternalRefusal;
use antecedent_io_core::error::IoError;

use crate::accepted_preferences::AcceptedPreferencesError;
use crate::preference_contract::PreferenceContractError;
use crate::preference_refusal::{CONSUME, EXPORT, refusal};
use crate::preference_route_choice::PreferenceEvaluationError;
use crate::signal::at_stage;

/// Why a preference artifact is refused.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum PreferenceArtifactError {
    /// The body could not be encoded or the container could not be written.
    #[error("encoding: {0}")]
    Encode(IoError),
    /// The container or body is corrupt, of another kind or an unsupported version.
    #[error("container: {0}")]
    Container(IoError),
    /// The bytes are not the artifact the consumer retained
    /// ([`antecedent_io_core::bind_retained_artifact`]); nothing was read.
    #[error("{0}")]
    Retained(IoError),
    /// A decoded field names nothing this reader knows.
    #[error("malformed field: {what}")]
    Malformed {
        /// What is malformed.
        what: String,
    },
    /// The decoded or relearned accepted preferences are refused.
    #[error("accepted preferences: {0}")]
    Accepted(AcceptedPreferencesError),
    /// The stored preference contract is refused.
    #[error("preference contract: {0}")]
    Contract(PreferenceContractError),
    /// The evaluation is refused.
    #[error("evaluation: {0}")]
    Evaluation(PreferenceEvaluationError),
    /// The identity recomputed from the decoded content differs from the stored identity.
    #[error("the stored identity does not match the stored content")]
    StoredIdentityMismatch,
    /// The identity differs from the consumer's retained identity.
    #[error("the identity differs from the consumer's retained identity")]
    IdentityExpected,
    /// The result was computed under a different contract than the consumer retained.
    #[error("the result was computed under a different preference contract")]
    ContractExpected,
    /// The result was computed from different draws than the consumer retained.
    #[error("the result was computed from different draws")]
    SourceExpected,
    /// The result names a different contract than the one stored with it.
    #[error("the result was computed under another contract than the one stored")]
    ResultContractMismatch,
    /// Replay was given different draws than the result was computed from.
    #[error("the replay source differs from the stored result's")]
    ReplaySourceDiffers,
    /// Replay was given different records than the preferences were learned from.
    #[error("the replay records differ from the ones the preferences were learned from")]
    ReplayRecordsDiffer,
    /// Recomputation does not reproduce the stored content.
    #[error("recomputation does not reproduce the stored {field}")]
    ReplayMismatch {
        /// The first stored field that differs.
        field: &'static str,
    },
    /// The execution context was cancelled during a replay; nothing was verified.
    #[error("the replay was cancelled")]
    Cancelled,
}

impl PreferenceArtifactError {
    /// Stable detail id, prefixed `preference_artifact.`; a wrapped refusal of another module
    /// keeps its own detail.
    #[must_use]
    pub fn detail(&self) -> &'static str {
        use antecedent_core::reason_details;
        match self {
            Self::Encode(_) => reason_details::preference_artifact::ENCODE_FAILED,
            Self::Container(_) => reason_details::preference_artifact::CONTAINER_REFUSED,
            Self::Retained(_) => reason_details::preference_artifact::RETAINED_ARTIFACT_MISMATCH,
            Self::Malformed { .. } => reason_details::preference_artifact::WIRE_MALFORMED,
            Self::Accepted(inner) => inner.detail(),
            Self::Contract(inner) => inner.detail(),
            Self::Evaluation(inner) => inner.detail(),
            Self::StoredIdentityMismatch => reason_details::preference_artifact::IDENTITY_RESEALED,
            Self::IdentityExpected => reason_details::preference_artifact::IDENTITY_EXPECTED,
            Self::ContractExpected => reason_details::preference_artifact::CONTRACT_EXPECTED,
            Self::SourceExpected => reason_details::preference_artifact::SOURCE_EXPECTED,
            Self::ResultContractMismatch => {
                reason_details::preference_artifact::RESULT_CONTRACT_MISMATCH
            }
            Self::ReplaySourceDiffers => reason_details::preference_artifact::REPLAY_SOURCE_DIFFERS,
            Self::ReplayRecordsDiffer => reason_details::preference_artifact::REPLAY_RECORDS_DIFFER,
            Self::ReplayMismatch { .. } => reason_details::preference_artifact::REPLAY_MISMATCH,
            Self::Cancelled => reason_details::preference_artifact::CANCELLED,
        }
    }
}

impl From<PreferenceArtifactError> for IoError {
    /// A container or retained-binding error stays itself; every other refusal keeps the registered code and
    /// detail of its structured refusal, with the detail id leading the message.
    fn from(error: PreferenceArtifactError) -> Self {
        match error {
            PreferenceArtifactError::Encode(inner)
            | PreferenceArtifactError::Container(inner)
            | PreferenceArtifactError::Retained(inner) => inner,
            other => {
                let refusal = other.to_refusal();
                Self::Refused {
                    code: refusal.code,
                    message: format!("{}: {other}", refusal.detail),
                }
            }
        }
    }
}

impl PreferenceArtifactError {
    /// Structured refusal for a failure to write, read, check or replay a preference artifact.
    ///
    /// An accepted-preferences, contract or evaluation refusal keeps its own refusal; an
    /// encoding or container failure carries the container's message as the supplied text.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let invalid = |stage, remedy| {
            refusal(antecedent_core::reason_code!("invalid_argument"), stage, self.detail())
                .remedy(remedy)
        };
        let unsatisfied = |remedy| {
            refusal(
                antecedent_core::reason_code!("decision_contract_unsatisfied"),
                CONSUME,
                self.detail(),
            )
            .remedy(remedy)
        };
        match self {
            Self::Encode(error) => {
                refusal(antecedent_core::reason_code!("invalid_argument"), EXPORT, self.detail())
                    .supplied(error.to_string())
                    .build()
            }
            Self::Container(error) => invalid(CONSUME, "supply bytes written by this library")
                .supplied(error.to_string())
                .build(),
            Self::Retained(error) => refusal(
                antecedent_core::reason_code!("external_binding_mismatch"),
                CONSUME,
                self.detail(),
            )
            .remedy("supply the bytes whose identity was retained")
            .supplied(error.to_string())
            .build(),
            Self::Malformed { what } => invalid(CONSUME, "supply bytes written by this library")
                .offending(what.clone())
                .build(),
            Self::Accepted(error) => at_stage(error.to_refusal(), CONSUME),
            Self::Contract(error) => at_stage(error.to_refusal(), CONSUME),
            Self::Evaluation(error) => at_stage(error.to_refusal(), CONSUME),
            Self::StoredIdentityMismatch => {
                invalid(CONSUME, "re-export from the producer instead of editing the bytes").build()
            }
            Self::IdentityExpected => {
                unsatisfied("supply the artifact whose identity was retained").build()
            }
            Self::ContractExpected => {
                unsatisfied("supply the result of the retained preference contract").build()
            }
            Self::SourceExpected => {
                unsatisfied("supply the result computed from the retained draws").build()
            }
            Self::ResultContractMismatch => {
                unsatisfied("re-export the result from the producer").build()
            }
            Self::ReplaySourceDiffers => {
                unsatisfied("replay with the draws the result was computed from").build()
            }
            Self::ReplayRecordsDiffer => {
                unsatisfied("replay with the records the preferences were learned from").build()
            }
            Self::ReplayMismatch { field } => {
                unsatisfied("re-export from the producer and replay again")
                    .offending(*field)
                    .build()
            }
            Self::Cancelled => {
                refusal(antecedent_core::reason_code!("cancelled_no_claim"), CONSUME, self.detail())
                    .build()
            }
        }
    }
}
