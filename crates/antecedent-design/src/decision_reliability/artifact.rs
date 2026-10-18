//! The `decision_reliability_v1` artifact.
//!
//! The artifact carries the selection draws (and the validation draws when there are
//! any), the request and the report. A consumer does not trust the report: it recomputes
//! it from the draws and compares the canonical encoding bit for bit, so an edited
//! utility, stream coordinate, declaration or claimed value is refused. A consumer that
//! retained the draws digest, the candidate-set digest or the selecting and validating
//! stream identities independently also refuses a consistently resealed forgery.
//! The carried closed-route snapshot may remain more conservative after a new license
//! is earned; publication always checks the current license independently.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, blake3_hex};
use antecedent_io_core::convert::to_cbor;
use antecedent_io_core::error::IoError;
use serde::{Deserialize, Serialize};

use super::ReliabilityError;
use super::calibration::{CalibrationCoordinate, ClosedRoute, PublishedRoute};
use super::draws::DecisionDraws;
use super::report::{ReliabilityReport, ReliabilityRequest, assess};
use super::stream::StreamId;

/// Container kind of a decision-reliability artifact.
pub const DECISION_RELIABILITY_KIND: &str = "decision_reliability_v1";
/// Byte bound checked before decoding.
pub const MAX_DECISION_RELIABILITY_BYTES: usize = 256 * 1024 * 1024;
const SECTION: &str = "reliability_body";
const BODY_VERSION: u32 = 1;

/// Durable decision-reliability claim with the draws it was computed from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionReliabilityArtifact {
    /// Body schema version.
    pub version: u32,
    /// Draws that selected the decision.
    pub draws: DecisionDraws,
    /// Independent validation draws, when a provided stream validated the decision.
    pub validation: Option<DecisionDraws>,
    /// What was assessed.
    pub request: ReliabilityRequest,
    /// The claimed report; recomputed by every consumer.
    pub report: ReliabilityReport,
    /// Canonical digest of the body.
    pub digest: String,
}

/// What a consumer retained independently of the artifact.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DecisionReliabilityExpectation {
    /// Digest of the selection draws ([`DecisionDraws::digest_hex`]).
    pub draws_digest: Option<String>,
    /// Digest of the candidate set and its search record.
    pub candidate_set_digest: Option<String>,
    /// The one stream every selection candidate must have been evaluated on.
    pub selection_stream: Option<StreamId>,
    /// The stream the selected candidate must have been validated on.
    pub validation_stream: Option<StreamId>,
    /// Utility unit.
    pub utility_unit: Option<String>,
    /// The decision the artifact must report.
    pub decided_id: Option<String>,
}

fn refused(detail: &str, message: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        message: format!("{detail}: {message}"),
    }
}

fn from_reliability(error: &ReliabilityError) -> IoError {
    IoError::Refused { code: error.reason_code(), message: error.to_string() }
}

fn conservative_closure_snapshot(
    carried: &[ClosedRoute],
    current: &[ClosedRoute],
    coordinate: &CalibrationCoordinate,
) -> bool {
    // Historical refusals can remain closed after a promotion, but cannot omit a
    // currently closed route or carry arbitrary metadata, duplicates or new ordering.
    let canonical = PublishedRoute::ALL.map(|route| ClosedRoute {
        route,
        coordinate: coordinate.describe(),
        code: antecedent_core::reason_code!("cell_not_licensed").to_owned(),
        remedy: "point diagnostics only: a coverage record at this candidate-set size, \
                 Monte Carlo budget and selection rule would be needed"
            .to_owned(),
    });
    let mut remaining = carried.iter();
    let mut next = remaining.next();
    for expected in &canonical {
        if next == Some(expected) {
            next = remaining.next();
        }
    }
    next.is_none() && current.iter().all(|required| carried.contains(required))
}

impl DecisionReliabilityArtifact {
    /// Assess the draws and seal the claim with them.
    ///
    /// # Errors
    /// Anything [`assess`] refuses, or encoding failure.
    pub fn new(
        draws: DecisionDraws,
        validation: Option<DecisionDraws>,
        request: ReliabilityRequest,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        let report = assess(&draws, validation.as_ref(), &request, ctx)
            .map_err(|error| from_reliability(&error))?;
        let mut artifact = Self {
            version: BODY_VERSION,
            draws,
            validation,
            request,
            report,
            digest: String::new(),
        };
        artifact.digest = artifact.recomputed_digest()?;
        Ok(artifact)
    }

    /// Canonical digest of the body with an empty digest field.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn recomputed_digest(&self) -> Result<String, IoError> {
        #[derive(Serialize)]
        struct DigestBody<'a> {
            version: u32,
            draws: &'a DecisionDraws,
            validation: &'a Option<DecisionDraws>,
            request: &'a ReliabilityRequest,
            report: &'a ReliabilityReport,
            digest: &'static str,
        }
        let body = DigestBody {
            version: self.version,
            draws: &self.draws,
            validation: &self.validation,
            request: &self.request,
            report: &self.report,
            digest: "",
        };
        Ok(blake3_hex(&to_cbor(&body)?))
    }

    /// Recompute the report from the draws and check every retained expectation.
    ///
    /// # Errors
    /// A changed digest or version, a noncanonical or insufficient historical refusal
    /// snapshot, any other report field that does not recompute bit for bit, an
    /// expectation that differs, or any refusal of [`assess`].
    pub fn verify(
        &self,
        expected: &DecisionReliabilityExpectation,
        ctx: &ExecutionContext,
    ) -> Result<(), IoError> {
        if self.version != BODY_VERSION {
            return Err(IoError::UnsupportedVersion { version: self.version });
        }
        if self.digest != self.recomputed_digest()? {
            return Err(refused("decision_reliability.identity_mismatch", "digest differs"));
        }
        let mut recomputed = assess(&self.draws, self.validation.as_ref(), &self.request, ctx)
            .map_err(|error| from_reliability(&error))?;
        if conservative_closure_snapshot(
            &self.report.closed_routes,
            &recomputed.closed_routes,
            &recomputed.coordinate,
        ) {
            // Preserve the exactly digested historical snapshot, never replace it
            // with newly granted licenses. Every other report field stays exact.
            recomputed.closed_routes.clone_from(&self.report.closed_routes);
        } else {
            return Err(refused(
                "decision_reliability.report_mismatch",
                "closed-route snapshot omits a current refusal or is noncanonical",
            ));
        }
        if to_cbor(&recomputed)? != to_cbor(&self.report)? {
            return Err(refused(
                "decision_reliability.report_mismatch",
                "report does not recompute from the draws",
            ));
        }
        self.check_expectation(expected)
    }

    fn check_expectation(&self, expected: &DecisionReliabilityExpectation) -> Result<(), IoError> {
        let differs = |detail: &str| Err(refused(detail, "differs from the retained expectation"));
        if expected.draws_digest.as_ref().is_some_and(|d| *d != self.draws.digest_hex()) {
            return differs("decision_reliability.draws_digest_mismatch");
        }
        if expected
            .candidate_set_digest
            .as_ref()
            .is_some_and(|d| *d != self.draws.candidate_set_digest())
        {
            return differs("decision_reliability.candidate_set_mismatch");
        }
        if expected.utility_unit.as_ref().is_some_and(|u| *u != self.draws.utility_unit) {
            return differs("decision_reliability.utility_unit_mismatch");
        }
        if expected.decided_id.as_ref().is_some_and(|d| *d != self.report.decided_id) {
            return differs("decision_reliability.decision_mismatch");
        }
        if let Some(stream) = &expected.selection_stream {
            if self.draws.candidates.iter().any(|c| c.segment.stream != *stream) {
                return differs("decision_reliability.selection_stream_mismatch");
            }
        }
        if let Some(stream) = &expected.validation_stream {
            let validated = self
                .validation
                .as_ref()
                .and_then(|v| v.candidate(&self.report.decided_id))
                .is_some_and(|c| c.segment.stream == *stream);
            if !validated {
                return differs("decision_reliability.validation_stream_mismatch");
            }
        }
        Ok(())
    }

    /// Write a verified, bounded, checksummed container.
    ///
    /// # Errors
    /// Verification failure, a blank display identity, or encoding failure.
    pub fn to_bytes(&self, artifact_id: &str, ctx: &ExecutionContext) -> Result<Vec<u8>, IoError> {
        self.verify(&DecisionReliabilityExpectation::default(), ctx)?;
        if artifact_id.trim().is_empty() {
            return Err(refused(
                "decision_reliability.artifact_id_missing",
                "artifact id is blank",
            ));
        }
        let body = to_cbor(self)?;
        antecedent_io_core::encode_container(
            &antecedent_io_core::ArtifactContainer {
                kind: DECISION_RELIABILITY_KIND,
                note: "decision reliability: separate uncertainty components over draws",
                max_bytes: MAX_DECISION_RELIABILITY_BYTES,
                artifact_id,
                library_version: env!("CARGO_PKG_VERSION"),
            },
            &[antecedent_io_core::OutSection {
                id: SECTION,
                content_type: antecedent_io_core::CONTENT_CBOR,
                bytes: &body,
            }],
        )
        .map_err(|error| match error {
            antecedent_io_core::CodecError::Layout => refused(
                "decision_reliability.artifact_kind_mismatch",
                "unsupported container layout",
            ),
            antecedent_io_core::CodecError::MissingArtifactId => {
                refused("decision_reliability.artifact_id_missing", "artifact id is blank")
            }
            other => other.into_io("decision reliability artifact"),
        })
    }

    /// Decode a bounded container; scientific verification remains explicit.
    ///
    /// # Errors
    /// Corruption, wrong kind or version, or byte limits.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, IoError> {
        let parts = antecedent_io_core::decode_container(
            DECISION_RELIABILITY_KIND,
            MAX_DECISION_RELIABILITY_BYTES,
            &[SECTION],
            bytes,
        )
        .map_err(|error| match error {
            antecedent_io_core::CodecError::Layout => refused(
                "decision_reliability.artifact_kind_mismatch",
                "unsupported container layout",
            ),
            other => other.into_io("decision reliability artifact"),
        })?;
        let Ok([body]) = <[Vec<u8>; 1]>::try_from(parts) else {
            return Err(refused(
                "decision_reliability.artifact_kind_mismatch",
                "unsupported container layout",
            ));
        };
        // The version is read from the untyped value and gated before the typed
        // shape is interpreted, so a newer body is `UnsupportedVersion`, not a
        // shape error.
        antecedent_io_core::decode_with_version(&body, |version: u32| {
            if version == BODY_VERSION {
                Ok(())
            } else {
                Err(IoError::UnsupportedVersion { version })
            }
        })
    }
}

/// Independently consume decision-reliability bytes: decode, recompute the report from
/// the carried draws, and check the retained expectations.
///
/// # Errors
/// Anything [`DecisionReliabilityArtifact::from_bytes`] or
/// [`DecisionReliabilityArtifact::verify`] refuses.
pub fn consume(
    bytes: &[u8],
    expected: &DecisionReliabilityExpectation,
    ctx: &ExecutionContext,
) -> Result<DecisionReliabilityArtifact, IoError> {
    let artifact = DecisionReliabilityArtifact::from_bytes(bytes)?;
    artifact.verify(expected, ctx)?;
    Ok(artifact)
}
