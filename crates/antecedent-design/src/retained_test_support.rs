//! Shared assertions for the `*_expecting` loaders: the retained bytes reach the loader's own
//! checks unchanged, and other bytes are refused as not the retained artifact.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_io_core::RetainedArtifactIdentity;
use antecedent_io_core::retained_artifact::RETAINED_ARTIFACT_REFUSAL;

/// The identity a consumer retained for bytes other than the ones under test.
pub(crate) fn other_artifact() -> RetainedArtifactIdentity {
    RetainedArtifactIdentity::of_bytes(b"another artifact")
}

/// The bound form reached the loader's own verdict: same success, and any refusal is the
/// loader's own, not the binding refusal.
pub(crate) fn same_outcome<T, U, E: std::fmt::Debug, F: std::fmt::Debug>(
    original: &Result<T, E>,
    bound: &Result<U, F>,
) {
    assert_eq!(
        original.is_ok(),
        bound.is_ok(),
        "{:?} vs {:?}",
        original.as_ref().err(),
        bound.as_ref().err()
    );
    if let Err(error) = bound {
        assert!(!format!("{error:?}").contains(RETAINED_ARTIFACT_REFUSAL), "{error:?}");
    }
}

/// The bound form refused bytes that are not the retained artifact.
pub(crate) fn refused_as_not_retained<T, E: std::fmt::Debug>(bound: &Result<T, E>) {
    match bound {
        Ok(_) => panic!("bytes other than the retained artifact were admitted"),
        Err(error) => {
            assert!(format!("{error:?}").contains(RETAINED_ARTIFACT_REFUSAL), "{error:?}")
        }
    }
}
