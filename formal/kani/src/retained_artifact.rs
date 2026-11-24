//! `antecedent_io_core::retained_artifact`: every `*_expecting` loader admits bytes only as the
//! artifact its consumer retained.
//!
//! The artifact digest and the facet digest are replaced by arbitrary values
//! (`artifact_identity_any`, `facet_digest_any`), so these proofs use no property of BLAKE3:
//! they hold for every digest function. That each `*_expecting` loader is
//! `load_expecting(bytes, retained, <its unbound form>)`, or calls `bind_retained_artifact`
//! first, is checked by `scripts/check_retained_artifact_loaders.py`.

use antecedent_core::SemanticDigest;
use antecedent_io_core::IoError;
use antecedent_io_core::bound_identity::BoundIdentity;
use antecedent_io_core::retained_artifact::{
    RetainedArtifactIdentity, bind_artifact_identity, load_expecting,
};

fn any_digest() -> SemanticDigest {
    SemanticDigest::from_bytes(kani::any())
}

/// An arbitrary digest function in place of BLAKE3, for the artifact digest.
fn artifact_identity_any(_bytes: &[u8]) -> SemanticDigest {
    any_digest()
}

/// An arbitrary digest function in place of BLAKE3, for the facet digest.
fn facet_digest_any(_bound: &BoundIdentity) -> SemanticDigest {
    any_digest()
}

/// The artifact slot binds exactly when the presented digest equals the retained one, for every
/// pair of digests.
#[kani::proof]
#[kani::stub(antecedent_io_core::bound_identity::facet_digest, facet_digest_any)]
#[kani::unwind(33)]
fn artifact_binds_only_the_retained_digest() {
    let presented = any_digest();
    let retained = any_digest();
    let outcome = bind_artifact_identity(presented, &RetainedArtifactIdentity::new(retained));
    assert_eq!(outcome.is_ok(), presented == retained);
}

/// `load_expecting` runs the loader only when the bytes bound, and every refusal it returns
/// before the loader runs is `external_binding_mismatch`.
#[kani::proof]
#[kani::stub(antecedent_io_core::retained_artifact::artifact_identity, artifact_identity_any)]
#[kani::stub(antecedent_io_core::bound_identity::facet_digest, facet_digest_any)]
#[kani::unwind(33)]
fn loader_runs_only_after_binding() {
    let bytes: [u8; 4] = kani::any();
    let retained = RetainedArtifactIdentity::new(any_digest());
    let mut read = false;
    let outcome: Result<(), IoError> = load_expecting(&bytes, &retained, |_| {
        read = true;
        Ok(())
    });
    assert_eq!(outcome.is_ok(), read);
    if let Err(refusal) = outcome {
        assert!(matches!(refusal, IoError::Refused { code: "external_binding_mismatch", .. }));
    }
}
