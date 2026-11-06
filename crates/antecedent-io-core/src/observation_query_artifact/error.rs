//! Observation-query error and refusal translation.

use antecedent_contract::{
    ExternalRefusal, level_unreachable_refusal, mixed_conditioning_refusal,
    zero_probability_evidence_refusal,
};
use antecedent_model::compose::CompositionError;
use antecedent_model::observation_query::{ObservationQueryError, UnreachableBound};

use crate::IoError;
use crate::mechanism_composition_artifact::refusals_for_findings;

/// Why an observation-query artifact was refused.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ObservationQueryArtifactError {
    /// The bytes do not decode as this format.
    #[error("observation-query artifact does not decode: {0}")]
    Decode(String),
    /// The feature marker is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// Another format version, refused before the payload is interpreted.
    #[error("unsupported artifact version {version}")]
    UnsupportedVersion {
        /// The stored version.
        version: u32,
    },
    /// A stored collection exceeds the consumer's bound.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The identity does not match the stored premises, query and reading, or
    /// the caller's expected identity.
    #[error("identity mismatch")]
    IdentityMismatch,
    /// The artifact could not be encoded.
    #[error("observation-query artifact does not encode: {0}")]
    Encode(String),
    /// A stored field is not a well formed declaration.
    #[error("malformed premises: {0}")]
    Malformed(String),
    /// The stored mechanisms do not establish a coherent joint law.
    #[error("mechanisms do not compose: {} refusal(s)", .refusals.len())]
    Incoherent {
        /// Every defect, as structured refusals.
        refusals: Vec<ExternalRefusal>,
    },
    /// The request is refused: each refusal has a registered code, a
    /// namespaced `observation_query.*` detail and a remedy.
    #[error("observation query refused: {} refusal(s)", .refusals.len())]
    Refused {
        /// The refusals (never empty).
        refusals: Vec<ExternalRefusal>,
    },
    /// The replayed result differs from the stored one.
    #[error("result does not replay")]
    ResultMismatch,
    /// Cancellation was observed at a work boundary.
    #[error("observation query cancelled")]
    Cancelled,
    /// The replay itself failed.
    #[error("replay failed: {0}")]
    Replay(String),
}

impl From<IoError> for ObservationQueryArtifactError {
    fn from(error: IoError) -> Self {
        Self::Decode(error.to_string())
    }
}

pub(super) fn refused(refusal: ExternalRefusal) -> ObservationQueryArtifactError {
    ObservationQueryArtifactError::Refused { refusals: vec![refusal] }
}

pub(super) fn registry_refusal(
    code: &'static str,
    detail: &str,
    offending: impl Into<String>,
    remedy: &'static str,
) -> ExternalRefusal {
    ExternalRefusal::new(code, antecedent_contract::OBSERVATION_QUERY_STAGE, detail)
        .with_offending(offending)
        .with_remedy(remedy)
}

/// The structured refusal for a numerical-layer error.
///
/// [`ObservationQueryError::Cancelled`] is not a refusal of the request and
/// has none; callers handle it separately.
#[must_use]
pub fn query_error_refusals(error: &ObservationQueryError) -> Vec<ExternalRefusal> {
    let invalid = antecedent_core::reason_code!("invalid_argument");
    let one = |r: ExternalRefusal| vec![r];
    match error {
        ObservationQueryError::UnknownVariable(name) => one(registry_refusal(
            antecedent_core::reason_code!("unknown_variable"),
            antecedent_core::reason_details::observation_query::UNKNOWN_VARIABLE,
            name.clone(),
            "name a variable of the composed model",
        )),
        ObservationQueryError::UnknownLevel { variable, level } => one(registry_refusal(
            invalid,
            antecedent_core::reason_details::observation_query::UNKNOWN_LEVEL,
            format!("{variable}={level}"),
            "name a declared level of the variable",
        )),
        ObservationQueryError::RepeatedVariable(name) => one(registry_refusal(
            invalid,
            antecedent_core::reason_details::observation_query::REPEATED_SLOT,
            name.clone(),
            "assign each variable at most once",
        )),
        ObservationQueryError::OutcomeObserved(name) => one(registry_refusal(
            invalid,
            antecedent_core::reason_details::observation_query::OUTCOME_ASSIGNED,
            name.clone(),
            "ask about an outcome that is not itself conditioned on",
        )),
        ObservationQueryError::ConstructObserved(name) => one(registry_refusal(
            invalid,
            antecedent_core::reason_details::observation_query::CONSTRUCT_OBSERVED_DIRECTLY,
            name.clone(),
            "condition on the construct directly or on its measured indicator, not both",
        )),
        ObservationQueryError::ZeroProbabilityEvidence => one(zero_probability_evidence_refusal()),
        ObservationQueryError::ObservedAndIntervened(name) => one(mixed_conditioning_refusal(name)),
        ObservationQueryError::EvidenceSingular => one(registry_refusal(
            invalid,
            antecedent_core::reason_details::observation_query::EVIDENCE_COVARIANCE_SINGULAR,
            "observed".to_owned(),
            "observe values whose joint covariance under the model is positive definite: drop an observed variable that is a deterministic function of the others, or give the mechanism a positive residual scale",
        )),
        ObservationQueryError::Invalid(what) => one(registry_refusal(
            invalid,
            antecedent_core::reason_details::observation_query::INVALID_DECLARATION,
            (*what).to_owned(),
            "supply a well formed declaration: probabilities in range, distributions that sum to one, named levels",
        )),
        ObservationQueryError::LevelUnreachable { level, extremum, direction, bound } => {
            let bound = match bound {
                UnreachableBound::SignalDoesNotMove => "signal_does_not_move",
                UnreachableBound::ObservationBudget { .. } => "observation_budget",
                UnreachableBound::NoObservationCrosses => "no_observation_crosses",
                _ => "unreachable",
            };
            one(level_unreachable_refusal(*level, *extremum, direction.as_str(), bound))
        }
        ObservationQueryError::Composition(CompositionError::Incoherent { findings }) => {
            refusals_for_findings(findings)
        }
        ObservationQueryError::Composition(other) => one(registry_refusal(
            antecedent_core::reason_code!("route_not_supported"),
            antecedent_core::reason_details::observation_query::REQUEST_UNSUPPORTED,
            other.to_string(),
            "state the query within the bounded finite family",
        )),
        _ => one(registry_refusal(
            antecedent_core::reason_code!("route_not_supported"),
            antecedent_core::reason_details::observation_query::REQUEST_UNSUPPORTED,
            error.to_string(),
            "state the query within the bounded family",
        )),
    }
}

pub(super) fn query_error(error: ObservationQueryError) -> ObservationQueryArtifactError {
    match error {
        ObservationQueryError::Cancelled => ObservationQueryArtifactError::Cancelled,
        ObservationQueryError::Composition(CompositionError::Incoherent { findings }) => {
            ObservationQueryArtifactError::Incoherent { refusals: refusals_for_findings(&findings) }
        }
        other => ObservationQueryArtifactError::Refused { refusals: query_error_refusals(&other) },
    }
}

pub(super) fn composition_error(error: CompositionError) -> ObservationQueryArtifactError {
    match error {
        CompositionError::Incoherent { findings } => {
            ObservationQueryArtifactError::Incoherent { refusals: refusals_for_findings(&findings) }
        }
        CompositionError::Cancelled => ObservationQueryArtifactError::Cancelled,
        other => ObservationQueryArtifactError::Replay(other.to_string()),
    }
}
