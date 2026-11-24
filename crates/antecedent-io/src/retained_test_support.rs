//! Shared check for the retained-identity (`*_expecting`) loader forms.
//!
//! Each form must give the original loader's outcome when the retained identity is that of
//! the bytes handed over, and refuse with the retained-artifact binding refusal, before the
//! bytes are read, when it is the identity of other bytes.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt::Display;

use antecedent_io_core::RetainedArtifactIdentity;
use antecedent_io_core::retained_artifact::RETAINED_ARTIFACT_REFUSAL;

/// Bytes the checks hand to every loader.
pub(crate) const BYTES: &[u8] = b"not an artifact";

/// The identity a consumer retained for [`BYTES`].
pub(crate) fn matching() -> RetainedArtifactIdentity {
    RetainedArtifactIdentity::of_bytes(BYTES)
}

/// The identity of another artifact.
pub(crate) fn other() -> RetainedArtifactIdentity {
    RetainedArtifactIdentity::of_bytes(b"another artifact")
}

fn outcome<T, E: Display>(result: Result<T, E>) -> Option<String> {
    result.err().map(|error| error.to_string())
}

/// The matching retained identity gives the original outcome; another identity is refused by
/// the binding.
#[track_caller]
pub(crate) fn assert_binds<T, E: Display>(
    original: Result<T, E>,
    matched: Result<T, E>,
    mismatched: Result<T, E>,
) {
    let original = outcome(original);
    if let Some(text) = &original {
        assert!(!text.contains(RETAINED_ARTIFACT_REFUSAL), "{text}");
    }
    assert_eq!(outcome(matched), original);
    let refused = outcome(mismatched).expect("another artifact is refused");
    assert!(refused.contains(RETAINED_ARTIFACT_REFUSAL), "{refused}");
}

/// A measured expectation for loaders that take one.
pub(crate) fn measured_expectation() -> crate::measured_inference::MeasuredExpectation {
    crate::measured_inference::MeasuredExpectation {
        route: String::new(),
        candidate_digest: String::new(),
        premises_digest: String::new(),
        data_digest: String::new(),
        level: 0.95,
        scalars: Vec::new(),
        seal: String::new(),
    }
}
