//! Fuzz the composition verification artifact: bounded decode with stored-identity re-derivation.
#![no_main]

use antecedent_io_core::composition_verification_artifact::CompositionVerificationArtifact;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = CompositionVerificationArtifact::from_bytes(data, None);
});
